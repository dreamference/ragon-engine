// ragon-metal-run: greedy decode on the Metal backend (METAL spec, first lane), no drafter.
//   ragon-metal-run --model DIR --prompts-ids FILE [--check FILE] [--max-tokens N] [--greedy-out FILE]
//                   [--ban-eos] [--ctx N] [--chunk C]
//   ragon-metal-run --model DIR --prompts-ids FILE --teacher FILE    (argmax agreement on a reference continuation)
//     [--teacher-chunk N]  feed the continuation N tokens a step (default --chunk; 1 scores the decode path)
//   ragon-metal-run --model DIR --prompt-ids 1,2,3 --dump DIR      (per-layer check dumps, .npy f32)
//   ragon-metal-run --model DIR --stdio [--ctx N] [--chunk C]      (a serving loop over JSON lines, see serve_stdio)
// A prompts-ids file holds one prompt a line as comma-separated token ids; a check file holds one JSON object a
// line with an "ids" array (the reference continuation), compared token by token.
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "backend/metal/model.hpp"

using ragon::metal::Model;
using ragon::metal::Queue;
using ragon::metal::View;

static std::vector<uint32_t> parse_ids(const std::string& s) {
  std::vector<uint32_t> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty() && t.find_first_not_of(" \t\r\n") != std::string::npos) v.push_back(uint32_t(std::stoul(t)));
  return v;
}

static std::vector<std::vector<int>> read_check(const std::string& path) {
  std::vector<std::vector<int>> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    NSData* d = [NSData dataWithBytes:line.data() length:line.size()];
    NSDictionary* j = [NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
    std::vector<int> ids;
    for (NSNumber* n in j[@"ids"]) ids.push_back([n intValue]);
    out.push_back(ids);
  }
  return out;
}

static void write_npy(const std::string& path, const float* p, const std::vector<int64_t>& shape) {
  std::string sh = "(";
  size_t n = 1;
  for (int64_t d : shape) {
    sh += std::to_string(d) + ",";
    n *= size_t(d);
  }
  if (shape.size() > 1) sh.pop_back();
  sh += ")";
  std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
  const size_t total = 10 + hdr.size() + 1;
  hdr += std::string((64 - total % 64) % 64, ' ') + "\n";
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(2);
  }
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  std::fwrite(magic, 1, 8, f);
  const uint16_t hl = uint16_t(hdr.size());
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(hdr.data(), 1, hdr.size(), f);
  std::fwrite(p, 4, n, f);
  std::fclose(f);
}

struct Top {
  int id = 0;
  float v = -INFINITY, second = -INFINITY;
};

static Top top2(const float* x, int n, const std::vector<int>& banned) {
  Top t;
  for (int i = 0; i < n; ++i) {
    if (!(x[i] > t.second)) continue;  // the banned check only for a value that would enter the top two
    if (std::find(banned.begin(), banned.end(), i) != banned.end()) continue;
    if (x[i] > t.v) {
      t.second = t.v;
      t.v = x[i];
      t.id = i;
    } else if (x[i] > t.second) {
      t.second = x[i];
    }
  }
  return t;
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void reply(NSDictionary* d) {
  NSData* j = [NSJSONSerialization dataWithJSONObject:d options:0 error:nil];
  std::fwrite([j bytes], 1, [j length], stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// A serving loop for a front end that tokenizes (the Mac node has no HTTP surface yet): one JSON request a line on
// stdin, {"ids":[...], "max_tokens":N, "stop":[ids]}, one reply a line on stdout, {"ids":[generated], "finish":
// "stop"|"length", "reused":n, "prefill_tokens":n, "prefill_s":f, "decode_s":f, "decode_tok_s":f} or {"error":...}.
// Greedy on the CPU pick. The sequence in the model stays between requests: a request that extends it (an agent's
// next turn) feeds only the new tokens. The recurrent state cannot be rewound, so any other request starts over.
// The last generated token (the stop token, or the one that hit max_tokens) is not fed; a next turn feeds it.
static int serve_stdio(Model& m, int ctx, int chunk) {
  reply(@{@"ready" : @YES, @"vocab" : @(m.vocab())});
  std::vector<uint32_t> fed;
  char* buf = nullptr;
  size_t cap = 0;
  ssize_t len;
  while ((len = getline(&buf, &cap, stdin)) > 0) {
    @autoreleasepool {
      NSData* d = [NSData dataWithBytes:buf length:size_t(len)];
      NSDictionary* j = [NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
      if (![j isKindOfClass:[NSDictionary class]] || ![j[@"ids"] isKindOfClass:[NSArray class]]) {
        reply(@{@"error" : @"bad request: want {\"ids\":[...]}"});
        continue;
      }
      std::vector<uint32_t> ids;
      for (NSNumber* n in j[@"ids"]) ids.push_back(uint32_t([n unsignedIntValue]));
      std::vector<int> stop = m.eos_ids;
      for (NSNumber* n in j[@"stop"]) stop.push_back([n intValue]);
      int max_tokens = j[@"max_tokens"] ? [j[@"max_tokens"] intValue] : 256;
      const int P = int(ids.size());
      if (P == 0 || P >= ctx) {
        reply(@{@"error" : [NSString stringWithFormat:@"prompt of %d tokens outside 1..%d", P, ctx - 1]});
        continue;
      }
      max_tokens = std::max(1, std::min(max_tokens, ctx - P));
      size_t reuse = 0;
      if (!fed.empty() && fed.size() < ids.size() && std::equal(fed.begin(), fed.end(), ids.begin()))
        reuse = fed.size();
      const double tp = now();
      if (reuse) {
        for (int s = int(reuse); s < P; s += chunk) m.forward(ids.data() + s, std::min(chunk, P - s), s, false);
      } else {
        m.reset();
        m.prefill(ids.data(), P);
      }
      const double prefill_s = now() - tp;
      fed = ids;
      std::vector<int> gen;
      const char* finish = "length";
      const double td = now();
      for (int k = 0; k < max_tokens; ++k) {
        const int id = top2(m.logits(), m.vocab(), {}).id;
        gen.push_back(id);
        if (std::find(stop.begin(), stop.end(), id) != stop.end()) {
          finish = "stop";
          break;
        }
        if (k + 1 == max_tokens) break;
        const uint32_t u = uint32_t(id);
        m.forward(&u, 1, int(fed.size()), false);
        fed.push_back(u);
      }
      const double decode_s = now() - td;
      NSMutableArray* g = [NSMutableArray arrayWithCapacity:gen.size()];
      for (int id : gen) [g addObject:@(id)];
      const int steps = int(gen.size()) - 1;
      reply(@{@"ids" : g, @"finish" : @(finish), @"reused" : @(reuse), @"prefill_tokens" : @(P - int(reuse)),
              @"prefill_s" : @(prefill_s), @"decode_s" : @(decode_s),
              @"decode_tok_s" : @(steps > 0 ? steps / decode_s : 0.0)});
    }
  }
  std::free(buf);
  return 0;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string model, prompts_file, check_file, greedy_out, dump_dir, teacher_file;
    std::vector<std::vector<uint32_t>> prompts;
    int max_tokens = 256, ctx = 40960, chunk = 128, teacher_chunk = 0, spec_block = 0;
    std::string draft_dir, sparse_head_dir;
    bool step_curve = false;
    bool ban_eos = false, stdio = false;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "%s needs a value\n", a.c_str());
          std::exit(2);
        }
        return argv[++i];
      };
      if (a == "--model") model = next();
      else if (a == "--prompts-ids") prompts_file = next();
      else if (a == "--prompt-ids") prompts.push_back(parse_ids(next()));
      else if (a == "--check") check_file = next();
      else if (a == "--teacher") teacher_file = next();
      else if (a == "--teacher-chunk") teacher_chunk = std::stoi(next());
      else if (a == "--greedy-out") greedy_out = next();
      else if (a == "--dump") dump_dir = next();
      else if (a == "--max-tokens") max_tokens = std::stoi(next());
      else if (a == "--ctx") ctx = std::stoi(next());
      else if (a == "--chunk") chunk = std::stoi(next());
      else if (a == "--ban-eos") ban_eos = true;
      else if (a == "--stdio") stdio = true;
      else if (a == "--step-curve") step_curve = true;
      else if (a == "--draft") draft_dir = next();
      else if (a == "--sparse-head-dir") sparse_head_dir = next();
      else if (a == "--spec-block") spec_block = std::stoi(next());
      else {
        std::fprintf(stderr, "unknown argument %s\n", a.c_str());
        return 2;
      }
    }
    if (!prompts_file.empty()) {
      std::ifstream f(prompts_file);
      std::string line;
      while (std::getline(f, line))
        if (!line.empty()) prompts.push_back(parse_ids(line));
    }
    if (model.empty() || (prompts.empty() && !stdio)) {
      std::fprintf(stderr, "usage: ragon-metal-run --model DIR (--prompts-ids FILE | --prompt-ids IDS) ...\n");
      return 2;
    }
    Queue q;
    q.open();
    std::fprintf(stderr, "device: %s\n", [[q.dev name] UTF8String]);
    Model m;
    const double t0 = now();
    m.load(q, model, ctx, chunk);
    if (!draft_dir.empty()) {
      m.draft_load(draft_dir);
      m.sparse_head_load(sparse_head_dir.empty() ? draft_dir + "/sparse-head" : sparse_head_dir);
    }
    if (!draft_dir.empty() && spec_block == 0) spec_block = m.draft_block();
    std::fprintf(stderr, "loaded in %.1f s\n", now() - t0);
    if (stdio) return serve_stdio(m, ctx, chunk);

    if (!dump_dir.empty()) {
      [[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:dump_dir.c_str()]
                                withIntermediateDirectories:YES attributes:nil error:nil];
      const std::vector<uint32_t>& ids = prompts[0];
      m.reset();
      m.forward(ids.data(), int(ids.size()), 0, true, [&](const std::string& name, View v, std::vector<int64_t> sh) {
        write_npy(dump_dir + "/" + name + ".npy", v.host<float>(), sh);
      });
      write_npy(dump_dir + "/logits.npy", m.logits(), {int64_t(ids.size()), m.vocab()});
      std::fprintf(stderr, "dumped to %s\n", dump_dir.c_str());
      return 0;
    }

    if (step_curve) {
      // The cost of a T-row forward with every row's logits (a speculative verify) after each prompt, T = 1..16:
      // five runs each at the same position (the rows' KV is overwritten), the median printed.
      for (size_t p = 0; p < prompts.size(); ++p) {
        const std::vector<uint32_t>& pr = prompts[p];
        const int P = int(pr.size());
        m.reset();
        m.prefill(pr.data(), P);
        std::vector<uint32_t> rows(pr.begin(), pr.begin() + 64);
        for (int T : {1, 2, 4, 8, 12, 16, 32, 64}) {
          std::vector<double> ts;
          std::fill(m.prof, m.prof + 5, 0.0);
          std::fill(m.pm, m.pm + 6, 0.0);
          for (int k = 0; k < 6; ++k) {
            const double t0 = now();
            m.forward(rows.data(), T, P, true);
            if (k) ts.push_back(now() - t0);
          }
          std::sort(ts.begin(), ts.end());
          std::printf("prompt %zu: pos %d, T %d: %.2f ms (min %.2f)\n", p, P, T, 1e3 * ts[ts.size() / 2], 1e3 * ts[0]);
          if (std::getenv("RAGON_METAL_PROF"))
            std::printf("  GPU ms a pass: attention %.2f, GDN %.2f (recurrence %.2f), MLP %.2f, head %.2f; gemv %.2f rest %.2f\n",
                        1e3 * m.prof[0] / 6, 1e3 * m.prof[1] / 6, 1e3 * m.prof[4] / 6, 1e3 * m.prof[2] / 6,
                        1e3 * m.prof[3] / 6, 1e3 * m.pm[2] / 6, 1e3 * m.pm[5] / 6);
          std::fflush(stdout);
        }
      }
      return 0;
    }

    if (!teacher_file.empty()) {
      // Teacher forcing: after each prompt, feed the reference continuation and count how often this backend's
      // argmax equals the reference's next token (a quality comparison that does not stop at the first near tie).
      std::vector<std::vector<int>> ref = read_check(teacher_file);
      for (size_t p = 0; p < prompts.size() && p < ref.size(); ++p) {
        const std::vector<uint32_t>& pr = prompts[p];
        const int P = int(pr.size());
        std::vector<uint32_t> cont(ref[p].begin(), ref[p].end());
        const int R = std::min(int(cont.size()), ctx - P);
        m.reset();
        m.prefill(pr.data(), P);
        int agree = 0;
        double margin_sum = 0;
        auto score = [&](const float* lg, int want) {
          Top t = top2(lg, m.vocab(), {});
          agree += t.id == want;
          margin_sum += t.v - lg[want];
        };
        score(m.logits(), int(cont[0]));
        const int tc = teacher_chunk > 0 ? std::min(teacher_chunk, chunk) : chunk;
        for (int s = 0; s + 1 < R; s += tc) {
          const int n = std::min(tc, R - 1 - s);
          m.forward(cont.data() + s, n, P + s, true);
          for (int r = 0; r < n; ++r) score(m.logits() + size_t(r) * m.vocab(), int(cont[s + r + 1]));
        }
        std::printf("prompt %zu: %d prompt tokens, teacher-forced %d reference tokens: argmax agrees on %d (%.1f %%), "
                    "mean logit margin to the reference token %.3f\n",
                    p, P, R, agree, 100.0 * agree / R, margin_sum / R);
        std::fflush(stdout);
      }
      return 0;
    }

    std::vector<std::vector<int>> check;
    if (!check_file.empty()) check = read_check(check_file);
    std::vector<int> banned = ban_eos ? m.eos_ids : std::vector<int>{};
    FILE* out = greedy_out.empty() ? nullptr : std::fopen(greedy_out.c_str(), "w");
    for (size_t p = 0; p < prompts.size(); ++p) {
      const std::vector<uint32_t>& pr = prompts[p];
      const int P = int(pr.size());
      if (P + max_tokens > ctx) {
        std::fprintf(stderr, "prompt %zu: %d + %d tokens exceed --ctx %d\n", p, P, max_tokens, ctx);
        return 2;
      }
      m.reset();
      std::fill(m.prof, m.prof + 5, 0.0);
      std::fill(m.pm, m.pm + 6, 0.0);
      const double tp = now();
      m.prefill(pr.data(), P);
      const double prefill_s = now() - tp;
      if (std::getenv("RAGON_METAL_LOGITS_HASH")) {  // exactness checks between prefill paths
        uint64_t hsh = 1469598103934665603ull;
        const unsigned char* b = reinterpret_cast<const unsigned char*>(m.logits());
        for (size_t k = 0; k < size_t(m.vocab()) * 4; ++k) hsh = (hsh ^ b[k]) * 1099511628211ull;
        std::fprintf(stderr, "prompt %zu: prefill logits hash %016llx\n", p, (unsigned long long)hsh);
      }
      if (std::getenv("RAGON_METAL_PROF"))
        std::fprintf(stderr, "prefill GPU s: attention %.3f, GDN %.3f (recurrence %.3f), MLP %.3f, head %.3f\n",
                     m.prof[0], m.prof[1], m.prof[4], m.prof[2], m.prof[3]);
      if (std::getenv("RAGON_METAL_PROF") && std::atoi(std::getenv("RAGON_METAL_PROF")) >= 2)
        std::fprintf(stderr, "prefill matmul passes s: dequant %.3f, x_half %.3f, matmul %.3f, store %.3f, silu_mul %.3f; other %.3f\n",
                     m.pm[0], m.pm[1], m.pm[2], m.pm[3], m.pm[4], m.pm[5]);
      std::vector<int> gen;
      float min_gap = INFINITY;
      int div = -1, div_ref = -1;
      float div_margin = 0, div_gap = 0;
      const std::vector<int>* ref = p < check.size() ? &check[p] : nullptr;
      double td = 0;
      std::fill(m.prof, m.prof + 5, 0.0);
      std::fill(m.pm, m.pm + 6, 0.0);
      m.gpu_s = 0;
      double top_s = 0;
      // Greedy on the GPU, pipelined two steps deep (RAGON_METAL_PIPE=0: the CPU picks each token). The CPU path
      // stays for the reference check (it needs the logits) and the profiles (they sync inside each step).
      const char* pe = std::getenv("RAGON_METAL_PIPE");
      const bool spec = spec_block > 0 && m.has_draft() && max_tokens > 1;
      const bool pipe = !spec && !(pe && std::atoi(pe) == 0) && !ref && !std::getenv("RAGON_METAL_PROF") && max_tokens > 1;
      if (spec) {
        // speculative decode: the drafter's block of spec_block rows verified by the target in one forward
        const Top t = top2(m.logits(), m.vocab(), banned);
        gen.push_back(t.id);
        std::fill(m.spec_s, m.spec_s + 3, 0.0);
        m.spec_steps = m.spec_tokens = 0;
        uint32_t anchor = uint32_t(t.id), buf[16];
        int pos = P;
        td = now();
        while (int(gen.size()) < max_tokens) {
          if (pos + spec_block > ctx) break;
          const int n = m.speculate(anchor, pos, spec_block, banned, buf);
          if (n < 0) {
            std::fprintf(stderr, "speculate: non-finite logits at position %d\n", pos);
            return 1;
          }
          for (int i = 0; i < n && int(gen.size()) < max_tokens; ++i) gen.push_back(int(buf[i]));
          pos += n;
          anchor = buf[n - 1];
        }
        if (ref)
          for (int k = 0; k < int(gen.size()) && k < int(ref->size()); ++k)
            if ((*ref)[k] != gen[k]) {
              div = k;
              div_ref = (*ref)[k];
              break;
            }
        const double ns = double(std::max(1L, m.spec_steps));
        std::fprintf(stderr, "spec B=%d: %ld steps, %.3f tokens a step; ms a step: draft %.3f, verify %.3f, accept %.3f\n",
                     spec_block, m.spec_steps, double(m.spec_tokens) / ns, 1e3 * m.spec_s[0] / ns,
                     1e3 * m.spec_s[1] / ns, 1e3 * m.spec_s[2] / ns);
        if (m.sparse_head())
          std::fprintf(stderr, "drafter head: %ld sparse steps, %ld full, %ld fallbacks\n", m.sh_sparse_steps,
                       m.sh_full_steps, m.sh_fallbacks);
        if (const char* pr = std::getenv("RAGON_METAL_PROF"); pr && std::atoi(pr) == 5)
          std::fprintf(stderr, "spec GPU ms a step: commit GDN %.3f, materialize %.3f, draft layers %.3f, head+selector %.3f, "
                       "top-16+lattice %.3f\n", 1e3 * m.dpm[0] / ns, 1e3 * m.dpm[1] / ns, 1e3 * m.dpm[2] / ns,
                       1e3 * m.dpm[3] / ns, 1e3 * m.dpm[4] / ns);
        if (const char* pr = std::getenv("RAGON_METAL_PROF"); pr && std::atoi(pr) == 6)
          std::fprintf(stderr, "draft layers GPU ms a step: attention GEMV %.3f, attention core %.3f, MLP GEMV %.3f, "
                       "other %.3f\n", 1e3 * m.dpm[0] / ns, 1e3 * m.dpm[1] / ns, 1e3 * m.dpm[2] / ns, 1e3 * m.dpm[3] / ns);
      }
      if (pipe) {
        const Top t = top2(m.logits(), m.vocab(), banned);
        min_gap = std::min(min_gap, t.v - t.second);
        gen.push_back(t.id);
        const int steps = max_tokens - 1;
        td = now();
        m.greedy_begin(uint32_t(t.id), banned, steps);
        for (int j = 0; j < std::min(steps, 2); ++j) m.greedy_step(P + j);
        for (int j = 0; j < steps; ++j) {
          m.greedy_wait(j);
          if (j + 2 < steps) m.greedy_step(P + j + 2);
          const uint32_t* o = m.greedy_out(j);
          float v, s2;
          std::memcpy(&v, o + 1, 4);
          std::memcpy(&s2, o + 2, 4);
          min_gap = std::min(min_gap, v - s2);
          gen.push_back(int(o[0]));
        }
      }
      for (int k = 0; k < (pipe || spec ? 0 : max_tokens); ++k) {
        const float* lg = m.logits();
        const double tt = now();
        Top t = top2(lg, m.vocab(), banned);
        if (k) top_s += now() - tt;
        min_gap = std::min(min_gap, t.v - t.second);
        if (ref && div < 0 && k < int(ref->size()) && (*ref)[k] != t.id) {
          div = k;
          div_ref = (*ref)[k];
          div_margin = t.v - lg[div_ref];
          div_gap = t.v - t.second;
        }
        gen.push_back(t.id);
        if (k + 1 == max_tokens) break;
        if (k == 0) td = now();
        const uint32_t id = uint32_t(t.id);
        m.forward(&id, 1, P + k, false);
      }
      const double decode_s = now() - td;
      const double dec_tps = max_tokens > 1 ? (max_tokens - 1) / decode_s : 0;
      if (spec)
        ;
      else if (pipe)  // two steps in flight: no per-step GPU time (RAGON_METAL_PIPE=0 for the wall/GPU split)
        std::fprintf(stderr, "decode ms a step: wall %.3f (pipelined)\n", decode_s * 1e3 / (max_tokens - 1));
      else if (max_tokens > 1)
        std::fprintf(stderr, "decode ms a step: wall %.3f, GPU (step command buffers) %.3f, top-2 %.3f\n",
                     decode_s * 1e3 / (max_tokens - 1), m.gpu_s * 1e3 / max_tokens, top_s * 1e3 / (max_tokens - 1));
      if (std::getenv("RAGON_METAL_PROF") && max_tokens > 1) {
        const double st = 1e3 / (max_tokens - 1);
        std::fprintf(stderr, "decode GPU ms a step: attention %.3f, GDN %.3f, MLP %.3f, head %.3f (wall %.3f)\n",
                     m.prof[0] * st, m.prof[1] * st, m.prof[2] * st, m.prof[3] * st, decode_s * st);
        if (std::atoi(std::getenv("RAGON_METAL_PROF")) == 4)
          std::fprintf(stderr, "decode GPU ms a step: GEMV %.3f, other %.3f, GDN recurrence %.3f\n", m.pm[2] * st,
                       m.pm[5] * st, m.prof[4] * st);
      }
      std::printf("prompt %zu: %d prompt tokens, prefill %.3f s (%.1f tok/s), decode %d tokens %.2f tok/s, "
                  "min top1-top2 gap %.3f",
                  p, P, prefill_s, P / prefill_s, int(gen.size()), dec_tps, min_gap);
      if (ref) {
        if (div < 0) std::printf(", matches the reference on all %d tokens\n", std::min(int(gen.size()), int(ref->size())));
        else
          std::printf(", first divergence at token %d: metal %d vs reference %d, metal logit margin %.4f "
                      "(metal top1-top2 gap there %.4f)\n",
                      div, gen[div], div_ref, div_margin, div_gap);
      } else {
        std::printf("\n");
      }
      std::fflush(stdout);
      if (out) {
        std::fprintf(out, "{\"ids\":[");
        for (size_t i = 0; i < gen.size(); ++i) std::fprintf(out, "%s%d", i ? "," : "", gen[i]);
        std::fprintf(out, "],\"prompt_tokens\":%d,\"prefill_s\":%.4f,\"decode_tok_s\":%.3f,\"first_divergence\":%d}\n",
                     P, prefill_s, dec_tps, div);
        std::fflush(out);
      }
    }
    if (out) std::fclose(out);
  }
  return 0;
}
