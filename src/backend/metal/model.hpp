// The hybrid Qwen3.8 model on the Metal backend (METAL spec, first lane): the checkpoint mapped as shipped
// (NVFP4 FFN and head, FP8 projections, BF16 embedding), unpacked in the kernels, and a forward pass over T
// tokens at a position (prefill chunks and M=1 decode share it). Objective-C++ only.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "backend/metal/device.hpp"

namespace ragon::metal {

struct Tensor {
  View v;
  std::string dtype;
  std::vector<int64_t> shape;
  size_t bytes = 0;
};

// safetensors shards, each mapped once and wrapped in a no-copy Metal buffer (copied when the OS refuses).
class Checkpoint {
 public:
  void open(Queue& q, const std::string& dir);
  bool has(const std::string& name) const { return where_.count(name) != 0; }
  Tensor get(const std::string& name);    // a view into the mapped shard (copied if not 16-byte aligned)
  View f32(const std::string& name);       // a small tensor converted to f32 on the host
  float scalar(const std::string& name);
  size_t mapped_bytes() const { return mapped_; }

 private:
  struct Shard {
    std::string path;
    const uint8_t* base = nullptr;
    size_t size = 0, data = 0;  // data = 8 + header length
    id<MTLBuffer> buf = nil;
    NSDictionary* hdr = nil;
  };
  Queue* q_ = nullptr;
  std::vector<Shard> shards_;
  std::map<std::string, int> where_;
  size_t mapped_ = 0;
};

struct Lin {
  enum Fmt { kNvfp4, kFp8, kBf16, kQ4 } fmt = kBf16;  // kQ4: built at load from kFp8
  View w, sc;
  float scale = 1.0f;
  uint32_t N = 0, K = 0;
  // layer-major prefill: the half weight already sits at wlay_ + dq_off (dequantized once for all chunks)
  bool dq_ready = false;
  size_t dq_off = 0;
};

struct Layer {
  bool attn = false;
  View in_norm, post_norm;
  // gated delta net
  Lin qkv, z, a, b, out;
  View conv_w, A_log, dt_bias, gnorm, conv_state, S;
  // full attention
  Lin q, k, v, o;
  View q_norm, k_norm, kc, vc;
  View ks, vs;  // int8 KV option: f32 scale a position and kv head
  // MLP
  Lin gate, up, down;
};

// One layer of the DFlash2 drafter (speculative decode): attention and MLP each wrapped in a grouped convolution
// whose per-row coefficients come from a BF16 kernel projection; sliding-window K/V in a ring of positions.
struct DraftLayer {
  View in_norm, post_norm, q_norm, k_norm, attn_base, mlp_base, kc, vc;
  Lin q, k, v, o, gate, up, down, attn_kproj, mlp_kproj;
};

class Model {
 public:
  using Dump = std::function<void(const std::string& name, View v, std::vector<int64_t> shape)>;

  void load(Queue& q, const std::string& dir, int max_ctx, int chunk);
  void reset();  // a new sequence: zero the recurrent and conv states (the KV cache is overwritten by position)
  // Runs T <= chunk tokens at positions pos0.. and leaves logits for the last row (all rows if all_logits) in
  // logits(). With dump set, syncs at each check point and hands the buffer to it.
  void forward(const uint32_t* ids, int T, int pos0, bool all_logits, const Dump& dump = nullptr);
  // A new sequence's whole prompt (P tokens from position 0) in chunk-sized pieces, layer by layer: each layer's
  // weights are dequantized once for all chunks. Same kernels on the same inputs as forward() chunk by chunk, so
  // the same results; RAGON_METAL_LAYER_MAJOR=0 runs the chunk loop instead. Leaves the last row's logits.
  void prefill(const uint32_t* ids, int P);
  // Greedy decode without a CPU round trip a token: greedy_begin(id0, banned, n) puts the first id on the device;
  // each greedy_step(pos) encodes one token, picks the next id on the GPU (the largest logit outside banned, ties to
  // the lower id) into the device's id slot and commits without waiting, so the next step is encoded while this one
  // runs. greedy_wait(k) blocks until step k has run; greedy_out(k) is then {id, top, second} (floats as bits).
  void greedy_begin(uint32_t id0, const std::vector<int>& banned, int steps);
  void greedy_step(int pos);
  void greedy_wait(int k);
  const uint32_t* greedy_out(int k) const { return gout_.host<uint32_t>() + size_t(k) * 3; }
  const float* logits() const { return logits_.host<float>(); }
  int vocab() const { return V_; }
  int chunk() const { return C_; }
  std::vector<int> eos_ids;
  // The DFlash2 drafter (METAL spec, lane 7): draft_load maps it; from then on every forward captures the
  // drafter's target features and writes its context K/V (verify rows wait for the commit). speculate() runs one
  // step at pos: the anchor (the next token, not fed yet) and B - 1 drafted tokens through the target at once, the
  // longest prefix the target's greedy pick agrees with, plus the target's own next token. out gets those n >= 1
  // tokens (the last is the next anchor, not fed); returns n, or -1 when a verify row's top logit is not finite.
  // The accepted rows' GDN state and drafter context are committed into the command buffer that the next GPU work
  // (the next step, or any forward) runs first.
  // With sampling set, the step samples instead: each drafted token is drawn from the selector's softmax over its
  // 16 candidates at the request's temperature, and the chain is accepted by the standard rule against the target's
  // distribution from each row's top-k logits (core/accept.hpp), so the tokens are distributed as plain sampling.
  struct SpecSampling {
    float temperature = 1.f, top_p = 1.f, min_p = 0.f;
    int top_k = 20;  // 1..64
    std::mt19937_64* rng = nullptr;
  };
  void draft_load(const std::string& dir);
  // The drafter's sparse head (BACKLOG 19, core/sparse_head.cu on CUDA): model.safetensors in dir with us, vt and
  // fixed. Each draft step then scores the fixed rows, the session's context vocabulary and the RAGON_SH_DYNAMIC rows
  // the rank-256 copy rates best, instead of the whole head; a session whose last RAGON_SH_WINDOW tokens fall
  // outside them too often (above RAGON_SH_HI, back below RAGON_SH_LO) drafts with the full head. The target
  // verifies every token, so only the proposals change. RAGON_DRAFT_HEAD=full keeps the full head.
  void sparse_head_load(const std::string& dir);
  bool sparse_head() const { return sh_on_; }
  long sh_sparse_steps = 0, sh_full_steps = 0, sh_fallbacks = 0;
  bool has_draft() const { return !dl_.empty(); }
  int draft_block() const { return dblock_; }
  int speculate(uint32_t anchor, int pos, int B, const std::vector<int>& banned, uint32_t* out,
                const SpecSampling* sampling = nullptr);
  double spec_s[3] = {0, 0, 0};  // host seconds in the draft, the verify and the accept (commit encoding included)
  long spec_steps = 0, spec_tokens = 0;
  // RAGON_METAL_PROF=1: sync after each block and add its GPU seconds to prof[0..3] = attention, GDN, MLP, head;
  // prof[4] = the GDN recurrence alone (part of prof[1]).
  double prof[5] = {0, 0, 0, 0, 0};
  double gpu_s = 0;  // GPU time of forward()'s last command buffer, summed (a decode step is one command buffer)
  // RAGON_METAL_PROF=2 also syncs after each prefill matmul pass: pm[0..4] = weight dequant, x to half, matmul,
  // store, silu_mul; pm[5] = the other work queued before each matmul (these seconds are then missing from
  // prof[0..3]). RAGON_METAL_PROF=4 also syncs around each GEMV (decode): pm[2] = the GEMVs, pm[5] = the rest.
  double pm[6] = {0, 0, 0, 0, 0, 0};
  double dpm[5] = {0, 0, 0, 0, 0};  // RAGON_METAL_PROF=5: draft-step sections (speculate)

 private:
  // y[T][N] (=|+=) x[T][K] W^T: GEMV kernels for decode and short chunks, tensor-op matmul for T >= mm_min_.
  // mode 2 (prefill matmul only, see mm_fused()): y = silu(y) * (x W^T), the MLP up projection over the gate.
  // reuse_x: x is the input of the previous gemv call and unchanged since, so its half copy is kept.
  // mode 3 (prefill matmul only): as mode 2, but the result goes only to xh2_ as half, the A of the next gemv;
  // xa: the half A that a mode-3 gemv wrote (skips the x_half pass; x is then unused).
  void gemv(const Lin& l, View x, View y, int T, int ldx, int ldy, bool add, int mode = 0, bool reuse_x = false,
            View xa = View{});
  bool mm_fused(const Lin& l, int T) const { return !mm_old_ && use_mm(l, T); }
  bool use_mm(const Lin& l, int T) const;
  void attention_mm(Layer& L, int T, int pos0);
  id<MTLComputeCommandEncoder> pm_sync(id<MTLComputeCommandEncoder> e, int k);
  // next/T: when next takes the matmul path over these T tokens, the norm also writes out's half copy into xh_
  // (no x_half pass for it); out must be [T][ldo_t] with next.K == ldo_t.
  void norm(View x, View w, View z, View out, int rows, int n, int rpt, int ldt, int ldh, int ldo_t, int ldo_h,
            int mode, float scale = 1.0f, const Lin* next = nullptr, int T = 0);
  void gdn(Layer& L, int T, const Dump& dump, const std::string& tag);
  void layer(size_t i, int T, int pos0, const Dump& dump);  // one decoder layer over h_'s T rows
  void dequant(id<MTLComputeCommandEncoder> e, const Lin& l, View out);
  void attention(Layer& L, int T, int pos0, const Dump& dump, const std::string& tag);
  Lin lin(const std::string& prefix) { return lin_of(ck_, prefix); }
  Lin lin_of(Checkpoint& ck, const std::string& prefix);
  // drafter
  void capture(int T, int pos0, size_t layer);  // the target layer's rows inside [cap_lo_, cap_hi_) into cap_
  void draft_materialize(int n, int pos0);        // cap_ rows 0 .. n-1 (positions pos0 ..) into the ring
  void draft_forward(int B, uint32_t anchor, int pos);
  void sh_observe(const uint32_t* ids, int n, int pos0);  // tokens of the history at pos0 ..
  void sh_reset();
  void sh_rewin();
  bool sh_decide();          // true: this step drafts with the sparse head
  void sh_head(int E);       // the selection and the selected rows' logits (E x sh_cap_) into logits_
  Checkpoint shck_;
  bool sh_on_ = false, sh_full_ = true;
  View sh_us_, sh_vt_, sh_mask_, sh_keys_, sh_hist_, sh_state_, sh_counts_, sh_rows_, sh_nrows_, sh_p_;
  std::vector<uint8_t> sh_fixed_, sh_win_;
  int sh_nfixed_ = 0, sh_cap_ = 0, sh_dyn_ = 16384, sh_ctxmax_ = 32768, sh_W_ = 64, sh_ctx_ = 0, sh_n_ = 0;
  int sh_wpos_ = 0, sh_wn_ = 0, sh_novel_ = 0;
  double sh_hi_ = 0.15, sh_lo_ = 0.08;
  bool sh_verbose_ = false;
  void commit(int n, int pos0);
  void copy_rows(View src, View dst, int rows, int cols, int lds, int ldd);
  Checkpoint dck_;
  std::vector<DraftLayer> dl_;
  std::vector<int> feat_;  // target layer -> feature slot (-1: none)
  Lin dfc_, dsel_;
  View dhnorm_, dnorm_, dpred_, dsucc_;
  int dH_ = 0, dI_ = 0, dHq_ = 0, dKV_ = 0, dblock_ = 8, dgroups_ = 0, dwin_ = 2048, ring_ = 0, dmask_ = 0, nfeat_ = 0;
  float dtheta_ = 1e7f, deps_ = 1e-6f;
  int cap_lo_ = 0, cap_hi_ = 0, cap_rows_ = 0;
  View cap_, dctx_, dres_, dh_, dh2_, dcoef_, dq_, dk_, dv_, dattn_, dg_, du_, dout_, dunary_, dcand_, dhp_, dscores_,
      dids_, vout_;
  bool verify_ = false;
  int vmax_ = 16;                       // verify rows the saved GDN inputs hold
  std::vector<View> vpre_, vg_, vbeta_;  // per GDN layer: the verify rows' conv input, g and beta

  Queue* q_ = nullptr;
  Checkpoint ck_;
  std::vector<Layer> layers_;
  View embed_, final_norm_;
  Lin head_;
  int H_ = 0, KV_ = 0, D_ = 0, Hk_ = 0, Hv_ = 0, Dk_ = 0, Dv_ = 0, hidden_ = 0, ffn_ = 0, V_ = 0, rot_ = 0;
  int max_ctx_ = 0, C_ = 0, nchunk_max_ = 32, mm_min_ = 64;
  bool attn_gqa_ = true;
  int mq_nch_ = 32;  // verify attention key chunks (RAGON_METAL_MQ_NCH, 0: the decode kernel per row)
  bool g8_ = true;  // the simdgroup-matrix GEMV for 2..8 rows (gemv8)
  uint32_t g8_min_n_ = 0;  // gemv8 only for N at least this (RAGON_METAL_G8_MIN_N); smaller N take the row GEMV
  View xp8_;
  size_t xp8_cap_ = 0;
  const void* xp8_buf_ = nullptr;  // what xp8_ holds: x's buffer, offset, K, T, ldx
  size_t xp8_off_ = 0;
  uint32_t xp8_k_ = 0, xp8_t_ = 0, xp8_ld_ = 0;
  bool mm_old_ = false;  // RAGON_METAL_MM_OLD=1: the lane 2 prefill matmul (64x64 tiles, separate store pass)
  const void* xh_buf_ = nullptr;  // what xh_ holds: x's buffer, offset, K, T, ldx
  size_t xh_off_ = 0;
  uint32_t xh_k_ = 0, xh_t_ = 0, xh_ld_ = 0;
  bool xh_fresh_ = false;  // a norm just wrote xh_ for the next gemv (as reuse_x for that one call)
  bool xh_fuse_ = true;    // RAGON_METAL_XH_OLD=1: the separate x_half passes (lane 3)
  // Decode (one row): the residual-stream rmsnorms after layer 0 fold into the GEMVs around them. The GEMV that adds
  // into h_ while fold_out_ is set writes fxw_ = h * (1 + fold_out_) and fpart_ (sums of h*h, 8 rows each); a GEMV
  // of xn_ while fold_in_ is set reads fxw_ and scales by the rms from fpart_. Not bit-exact with the norm pass (the
  // scale goes after the dot). RAGON_METAL_NORM_FOLD=0 runs the norm passes.
  bool fold_ = true, fold_in_ = false;
  View fold_out_, fxw_, fpart_;
  View xh2_;               // half silu(gate)*up, the MLP down projection's A (mode 3)
  bool kv8_ = false;  // int8 KV cache option (RAGON_METAL_KV=int8)
  uint32_t flash_dbg_ = 0;  // flash timing probes (RAGON_METAL_FLASH_DBG, wrong results): skip 1 rescale, 2 PV, 4 softmax, 16 QK; 8 QK via registers
  int attn_flash_ = 2;  // prefill attention (RAGON_METAL_ATTN_FLASH): 0 = S/P matmuls; 2 = flash2 (softmax in registers);
                        // 4, 8, 128, 192 = the first flash kernel
  View kdq_, vdq_;    // int8 KV: bfloat scratch of one layer's cache for the prefill matmuls
  void to_q4(Lin& l);  // the Mac-native 4-bit projection option (RAGON_METAL_PROJ=q4)
  int attn_nch_ = 16, attn_chunk_min_ = 64;  // fused decode attention: max key chunks, min keys a chunk
  int prof_ = 0;
  size_t wdq_cap_ = 0;
  float eps_ = 1e-6f;
  // activations [C][...] f32
  View gout_;
  std::vector<id<MTLCommandBuffer>> gcb_;  // the last command buffer of each greedy step
  std::vector<uint32_t> gban_;
  int gstep_ = 0;
  bool greedy_ = false;  // inside greedy_step: forward() ends with the pick and a commit instead of a sync
  View ids_, h_, xn_, qkv_, zb_, ab_, bb_, core_, gated_, qg_, kb_, vb_, ob_, part_, gb_, ub_, logits_, cos_, sin_;
  // prefill matmul scratch: half weights, half activations, f32 product; attention Q, scores, probabilities
  View wdq_, xh_, mmc_, qb_, S_, P_;
  // layer-major prefill: the whole prompt's hidden rows and one layer's half weights
  View hall_, wlay_;
  bool layer_major_ = true;
  bool conc_ = true;  // decode: a concurrent encoder, barriers only between dependent dispatches (RAGON_METAL_CONC)
  int flush_ = 4;  // decode: commit the command buffer every flush_ layers so the GPU starts while the CPU encodes
  // chunked GDN prefill (RAGON_METAL_GDN_OLD=1 keeps the token loop): half q/k/v, per (sub-chunk, head) W, Uv, A2, lg
  View gq_, gk_, gv_, gW_, gU_, gA2_, glg_;
  double gtp_[3] = {0, 0, 0};  // RAGON_METAL_PROF=3: the chunked kernels apart (measurement only)
  long gtn_ = 0;
  int gdn_min_ = 64;  // chunk sizes from which the chunked recurrence runs; 0 = never
};

}  // namespace ragon::metal
