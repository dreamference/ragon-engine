// A fan-out (BATCHING P4d): a fork of a session (no copy of its KV, a copy of its per-sequence state), and many
// sessions' plain steps, one row each, in one pass. Kept apart from speculate.cpp: no speculation here and nothing of
// the one-stream path.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/engine.hpp"
#include "core/kernels.cuh"
#include "core/sampling.hpp"

namespace ragon {

namespace {
// The shortest shared run worth a launch of its own (a range of attention_chunk()'s default size).
constexpr int kCascadeMinKeys = 4096;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("fan-out: ") + what + ": " + cudaGetErrorString(e));
}

// The top K a row's token needs: 1 for greedy, its top_k when the GPU's top K covers it, 0 when the host must see every
// logit (penalties, or a top_k above what the kernel returns).
int row_top_k(const SamplingParams& p) {
  if (p.presence_penalty != 0.f || p.repetition_penalty != 1.f) return 0;
  if (p.temperature <= 0.f) return 1;
  return p.top_k > 0 && p.top_k <= 64 ? p.top_k : 0;
}
}  // namespace

void Engine::alloc_verify_rows() {
  if (topk_scratch_) return;
  const ModelConfig& c = model_->config();
  const int B = kernels::kMaxStreamRows, C = c.lin_conv_channels();
  topk_scratch_ = alloc<float>(kernels::topk_scratch_floats(B, 64));
  vals_dev_ = alloc<float>(size_t(B) * 64);
  ids_out_dev_ = alloc<int>(size_t(B) * 64);
  for (size_t i = 0; i < s_->gdn.size(); ++i) {
    v_pre_.push_back(alloc<float>(size_t(B) * C));
    v_post_.push_back(alloc<float>(size_t(B) * C));
    v_g_.push_back(alloc<float>(size_t(B) * c.lin_v_heads));
    v_beta_.push_back(alloc<float>(size_t(B) * c.lin_v_heads));
  }
}

void Engine::commit_seqs(std::span<Session* const> rows, cudaStream_t st, float* out) {
  const ModelConfig& c = model_->config();
  kernels::SeqBatch sb;
  sb.n = static_cast<int>(rows.size());
  for (int i = 0; i < sb.n; ++i) sb.row0[i + 1] = i + 1;
  for (int li = 0; li < c.layers; ++li) {
    const LayerWeights& L = model_->layers()[li];
    if (L.full) continue;  // the verify already wrote the KV
    const int slot = layer_slot_[li];
    kernels::SeqPtrs<float> conv, gdn;
    for (int i = 0; i < sb.n; ++i) {
      conv.p[i] = rows[i]->conv[slot];
      gdn.p[i] = rows[i]->gdn[slot];
    }
    kernels::gdn_conv_commit_seqs(v_pre_[slot], conv, L.conv, sb, c.lin_conv_channels(), st);
    kernels::gdn_recurrent_commit_seqs(v_post_[slot], v_g_[slot], v_beta_[slot], gdn, out, sb, c.lin_k_heads,
                                       c.lin_v_heads, st);
  }
}

void Engine::commit_state_seqs(std::span<Session* const> ss, std::span<const int> row0, std::span<const int> rows) {
  const ModelConfig& c = model_->config();
  kernels::SeqBatch sb;
  sb.n = static_cast<int>(ss.size());
  for (int i = 0; i < sb.n; ++i) {
    sb.row0[i] = row0[i];
    sb.row1[i] = row0[i] + rows[i];
  }
  sb.row0[sb.n] = sb.row1[sb.n - 1];
  const bool defer = gdn_fused_ && gdn_defer_;
  const auto t_prof = std::chrono::steady_clock::now();
  // RAGON_MK_GDN_FUSED: the recurrence's inputs of rows lo .. hi - 1 are kept for the next verify, after stream_ has
  // left the previous ones (a flush or a fused verify reads them).
  int lo = sb.row0[0], hi = sb.row1[0];
  for (int i = 1; i < sb.n; ++i) lo = std::min(lo, sb.row0[i]), hi = std::max(hi, sb.row1[i]);
  if (defer) {
    if (gdn_deferred_) throw std::logic_error("commit_state_seqs: a recurrence still pending");
    check(cudaEventRecord(pend_free_, stream_), "pending event");
    check(cudaStreamWaitEvent(commit_stream_, pend_free_, 0), "pending wait");
  }
  const int C = c.lin_conv_channels(), HV = c.lin_v_heads;
  for (int li = 0; li < c.layers; ++li) {
    const LayerWeights& L = model_->layers()[li];
    if (L.full) continue;  // the verify already wrote the KV
    const int slot = layer_slot_[li];
    kernels::SeqPtrs<float> conv, gdn;
    for (int i = 0; i < sb.n; ++i) {
      conv.p[i] = ss[i]->conv[slot];
      gdn.p[i] = ss[i]->gdn[slot];
    }
    kernels::gdn_conv_commit_seqs(v_pre_[slot], conv, L.conv, sb, C, commit_stream_);
    if (defer) {
      const auto keep = [&](float* to, const float* from, int w) {
        check(cudaMemcpyAsync(to + size_t(lo) * w, from + size_t(lo) * w, size_t(hi - lo) * w * sizeof(float),
                              cudaMemcpyDeviceToDevice, commit_stream_),
              "pending rows");
      };
      keep(p_post_[slot], v_post_[slot], C);
      keep(p_g_[slot], v_g_[slot], HV);
      keep(p_beta_[slot], v_beta_[slot], HV);
      continue;
    }
    kernels::gdn_recurrent_commit_seqs(v_post_[slot], v_g_[slot], v_beta_[slot], gdn, commit_core_, sb, c.lin_k_heads,
                                       HV, commit_stream_);
  }
  check(cudaEventRecord(commit_done_, commit_stream_), "commit event");
  commit_pending_ = true;
  if (profile_) {  // RAGON_PROFILE: the commit alone (no longer beside the draft), its GPU time to the host
    check(cudaStreamSynchronize(commit_stream_), "commit profile");
    phase_seconds_["commit/state"] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prof).count();
    phase_seconds_["commit/passes"] += 1;
  }
  if (defer) {
    gdn_deferred_ = true;
    pend_s_.assign(ss.begin(), ss.end());
    pend_r0_.assign(row0.begin(), row0.end());
    pend_n_.assign(rows.begin(), rows.end());
  }
}

kernels::SeqGroups Engine::cascade_groups(std::span<const Seg> segs) const {
  kernels::SeqGroups g;
  // RAGON_CASCADE=0: every sequence over all its keys, for a comparison.
  static const bool on = [] {
    const char* e = std::getenv("RAGON_CASCADE");
    return !e || std::atoi(e) != 0;
  }();
  if (!kv_book_ || !on || kernels::attention_chunk() % kernels::kKvPageTokens) return g;
  const int n = static_cast<int>(segs.size());
  // Pages wholly behind a session's position: every row of it sees all their keys and nothing writes them.
  const auto behind = [](const Session& s) {
    return std::min<int>(s.pos / kernels::kKvPageTokens, static_cast<int>(s.kv_ids.size()));
  };
  for (int i = 0; i < n;) {
    const Session& a = *segs[i].s;
    int run = behind(a), j = i + 1;
    for (; j < n; ++j) {
      const Session& b = *segs[j].s;
      const int limit = std::min(run, behind(b));
      int common = 0;
      while (common < limit && a.kv_ids[common] != 0 && a.kv_ids[common] == b.kv_ids[common]) ++common;
      if (common * kernels::kKvPageTokens < kCascadeMinKeys) break;
      run = common;
    }
    if (j - i >= 2) {
      g.seq0[g.n] = i;
      g.seq1[g.n] = j;
      g.cut[g.n] = run * kernels::kKvPageTokens;
      ++g.n;
    }
    i = j;
  }
  return g;
}

bool Engine::can_step_batch(int session) const {
  return topk_scratch_ && opts_.max_context - sessions_.at(session).pos >= 1;
}

void Engine::step_batch(std::span<SeqStep> seqs) {
  if (seqs.empty()) return;
  const int n = static_cast<int>(seqs.size());
  if (n > kernels::kMaxSeqs) throw std::runtime_error("step_batch: more sequences than kMaxSeqs");
  for (const SeqStep& q : seqs) {
    if (q.session < 0 || q.session >= sessions() || !q.params) throw std::runtime_error("step_batch: bad entry");
    if (!can_step_batch(q.session)) throw std::runtime_error("step_batch: a sequence cannot join the batch");
    for (const SeqStep& o : seqs)
      if (&o != &q && o.session == q.session) throw std::runtime_error("step_batch: a session twice");
  }
  if (kv_book_) {  // every sequence's page first, all or nothing (see speculate_batch)
    int need = 0;
    for (const SeqStep& q : seqs) need += kv_step_need(q.session, false);
    if (need > kv_book_->free_pages()) kv_retire(false);
    if (need > kv_book_->free_pages()) kv_retire(true);
    if (need > kv_book_->free_pages())
      throw KvPoolExhausted("KV page pool exhausted: the batch's step needs " + std::to_string(need) + " pages, " +
                            std::to_string(kv_book_->free_pages()) + " free");
    for (const SeqStep& q : seqs) {
      Session& s = sessions_[q.session];
      kv_ensure(s, s.pos, s.pos + 1, stream_);
    }
  }
  const size_t vocab = model_->config().vocab;
  std::vector<Seg> segs;
  std::vector<int> ks(n);
  int kmax = 1;
  for (int i = 0; i < n; ++i) {
    pin_->rows[i] = seqs[i].anchor;
    segs.push_back({&sessions_[seqs[i].session], i, 1, nullptr});
    ks[i] = row_top_k(*seqs[i].params);
    kmax = std::max(kmax, ks[i]);
  }
  // Every row of two or more is kept, whatever its sampling does (a row that fails resets its session below), so the
  // pass writes the DeltaNet state itself; one row takes the single-sequence path and its commit.
  auto t_prof = std::chrono::steady_clock::now();
  auto prof = [&](const char* phase) {
    if (!profile_) return;
    check(cudaStreamSynchronize(stream_), phase);
    const auto now = std::chrono::steady_clock::now();
    phase_seconds_[std::string("fanout/") + phase] += std::chrono::duration<double>(now - t_prof).count();
    t_prof = now;
  };
  if (profile_ && fanout_last_.time_since_epoch().count())
    phase_seconds_["fanout/between"] += std::chrono::duration<double>(t_prof - fanout_last_).count();
  verify_commits_ = n > 1;
  forward_rows(pin_->rows, n, Pass::Verify, -1, segs);
  verify_commits_ = false;
  prof("forward");
  kernels::topk_rows(logits_dev_, n, static_cast<int>(vocab), kmax, topk_scratch_, vals_dev_, ids_out_dev_, stream_);
  const size_t top = size_t(n) * kmax;
  check(cudaMemcpyAsync(pin_->topk_vals, vals_dev_, top * sizeof(float), cudaMemcpyDeviceToHost, stream_), "topk");
  check(cudaMemcpyAsync(pin_->topk_ids, ids_out_dev_, top * sizeof(int), cudaMemcpyDeviceToHost, stream_), "topk");
  // A row chosen on the host from every logit, and a row whose logits were asked for.
  std::vector<std::vector<float>> full(n);
  for (int i = 0; i < n; ++i) {
    std::vector<float>* to = seqs[i].row_logits;
    if (!to && ks[i] != 0) continue;
    if (!to) to = &full[i];
    to->resize(vocab);
    check(cudaMemcpyAsync(to->data(), logits_dev_ + size_t(i) * vocab, vocab * sizeof(float), cudaMemcpyDeviceToHost,
                          stream_),
          "row logits");
  }
  check(cudaStreamSynchronize(stream_), "step");
  prof("topk");
  // Every row's token, as sample() after step() draws it: the history holds the anchor, the seed counts it.
  std::vector<Session*> ok;
  for (int i = 0; i < n; ++i) {
    Session& s = sessions_[seqs[i].session];
    const SamplingParams& p = *seqs[i].params;
    const float* vals = pin_->topk_vals + size_t(i) * kmax;
    const int* ids = pin_->topk_ids + size_t(i) * kmax;
    try {
      s.history.push_back(seqs[i].anchor);
      int next;
      if (p.seed != 0 && p.temperature > 0.f) s.rng.seed(p.seed + s.history.size());
      if (ks[i] == 0) {
        const std::vector<float>& l = seqs[i].row_logits ? *seqs[i].row_logits : full[i];
        const size_t output = std::min(s.history.size(), s.prompt_len);
        next = sample_logits(l, p, std::span<const int>(s.history).subspan(output), s.rng);
      } else {
        require_finite_topk({vals, size_t(ks[i])}, {ids, size_t(ks[i])}, static_cast<int>(vocab));
        next = ks[i] == 1 && p.temperature <= 0.f ? ids[0]
                                                  : sample_topk({vals, size_t(ks[i])}, {ids, size_t(ks[i])}, p, s.rng);
      }
      seqs[i].out = {next};
      seqs[i].rows = 1;
      ok.push_back(&s);
    } catch (const NonFiniteLogits&) {
      seqs[i].error = std::current_exception();
    }
  }
  prof("sample");
  quantized_ = false;
  if (n == 1 && !ok.empty()) {
    if (commit_overlap_) {
      commit_seqs(ok, commit_stream_, commit_core_);
      check(cudaEventRecord(commit_done_, commit_stream_), "commit event");
      commit_pending_ = true;
    } else {
      commit_seqs(ok, stream_, core_);
    }
  }
  prof("commit");
  // The drafter's rings follow every sequence (a fork may speculate again later): one pass over all the rows, as the
  // forward was (BATCHING §5.8); a sequence that failed resets instead, and its row is skipped.
  std::vector<Session*> upkeep;
  std::vector<int> upkeep_rows;
  for (int i = 0; i < n; ++i)
    if (!seqs[i].error) {
      upkeep.push_back(&sessions_[seqs[i].session]);
      upkeep_rows.push_back(i);
    }
  if (draft_) {
    // MULTI_DRAFTER §2.3: a ring is its drafter's, so one pass covers the rows only when every session uses one drafter.
    bool same_drafter = true;
    for (const Session* u : upkeep) same_drafter = same_drafter && u->drafter == upkeep[0]->drafter;
    if (n > 1 && static_cast<int>(upkeep.size()) == n && same_drafter) {
      bind_session(upkeep[0]);
      draft_materialize_seqs(upkeep);
    } else {
      for (size_t j = 0; j < upkeep.size(); ++j) {
        bind_session(upkeep[j]);
        draft_materialize(1, upkeep[j]->pos, upkeep_rows[j]);
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    Session& s = sessions_[seqs[i].session];
    bind_session(&s);
    if (seqs[i].drafter_off) {  // speculate()'s plain step: the guard's count to its next probe
      s.guard.plain();
      s.spec.drafter_off_steps += 1;
    }
    if (seqs[i].error) {
      reset_session();
      continue;
    }
    s.pos += 1;
    if (!seqs[i].drafter_off) s.spec.fanout_steps += 1;
  }
  check(cudaStreamSynchronize(stream_), "commit");
  prof("materialize");
  if (profile_) fanout_last_ = std::chrono::steady_clock::now();
}

int Engine::kv_fork_need(int from, int to, const KvGone* gone) const {
  if (!kv_book_) return 0;
  const Session& parent = sessions_.at(from);
  const int L = static_cast<int>(parent.history.size());
  std::vector<std::vector<int>> gone_tree;
  KvPool::Refs gone_refs;
  if (gone) kv_gone_refs(*gone, gone_tree, gone_refs);
  return kv_book_->resume_need(sessions_.at(to).kv_ids, L, kv_admit_pages(L), &parent.kv_ids, gone ? &gone_refs : nullptr);
}

void Engine::fork_session(int from) {
  Session& parent = sessions_.at(from);
  if (&parent == s_) throw std::runtime_error("fork_session: a session is not forked onto itself");
  if (parent.drafter != s_->drafter)  // the ring it adopts is its drafter's (MULTI_DRAFTER §2.3)
    throw std::runtime_error("fork_session: the fork and its parent use different drafters");
  side_join();
  join_commit();
  const int L = static_cast<int>(parent.history.size());
  int target = L;
  if (kv_book_) {
    // As begin_prefill: checked first, so a short pool changes nothing; the session's pages past the target go back.
    const int need = kv_fork_need(from, active_session());
    if (need > kv_book_->free_pages()) kv_retire(false);
    if (need > kv_book_->free_pages()) kv_retire(true);
    if (need > kv_book_->free_pages())
      throw KvPoolExhausted("KV page pool exhausted: a fork needs " + std::to_string(need) + " pages, " +
                            std::to_string(kv_book_->free_pages()) + " free of " +
                            std::to_string(kv_book_->capacity()));
    target = kv_target(L);
    kv_release(*s_, target);
    if (static_cast<int>(s_->history.size()) > target) truncate_history(target);
    s_->pos = std::min(s_->pos, target);
  }
  adopt_prefix(parent, L);  // the pages (a reference each, flat slots a copy), the drafter's ring, the history
  copy_state(StateSlot{parent.gdn, parent.conv}, false);
  if (parent.snap_pos > 0) {
    copy_slot(s_->snap, parent.snap);
    s_->snap_pos = parent.snap_pos;
  }
  s_->pos = L;
  s_->rope_delta = parent.rope_delta;
  s_->prompt_len = parent.prompt_len;
  s_->turn_start = parent.turn_start;
  s_->guard.reset();
  if (kv_book_) {
    // The page L falls in stays with the parent: the fork copies it now, before either writes it. At a page boundary
    // nothing is copied. Pages the fork mapped past L before are dropped, as a prefill drops them.
    const int pending = kv_book_->pending_pages();
    kv_book_->unshare(s_->kv_ids, L);
    kv_table_sync(*s_, stream_);
    if (kv_book_->pending_pages() > pending) kv_record_release();
    kv_ensure(*s_, L, target, stream_);
  }
  // The parent's state is read only until here: its next pass or commit may run on another stream.
  check(cudaStreamSynchronize(stream_), "fork");
}

}  // namespace ragon
