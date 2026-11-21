// ragon-metal-run: greedy decode on the Metal backend (METAL spec, first lane), no drafter.
//   ragon-metal-run --model DIR --prompts-ids FILE [--check FILE] [--max-tokens N] [--greedy-out FILE]
//                   [--ban-eos] [--ctx N] [--chunk C]
//   ragon-metal-run --model DIR --prompts-ids FILE --teacher FILE    (argmax agreement on a reference continuation)
//   ragon-metal-run --model DIR --prompt-ids 1,2,3 --dump DIR      (per-layer check dumps, .npy f32)
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

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string model, prompts_file, check_file, greedy_out, dump_dir, teacher_file;
    std::vector<std::vector<uint32_t>> prompts;
    int max_tokens = 256, ctx = 40960, chunk = 128;
    bool ban_eos = false;
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
      else if (a == "--greedy-out") greedy_out = next();
      else if (a == "--dump") dump_dir = next();
      else if (a == "--max-tokens") max_tokens = std::stoi(next());
      else if (a == "--ctx") ctx = std::stoi(next());
      else if (a == "--chunk") chunk = std::stoi(next());
      else if (a == "--ban-eos") ban_eos = true;
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
    if (model.empty() || prompts.empty()) {
      std::fprintf(stderr, "usage: ragon-metal-run --model DIR (--prompts-ids FILE | --prompt-ids IDS) ...\n");
      return 2;
    }
    Queue q;
    q.open();
    std::fprintf(stderr, "device: %s\n", [[q.dev name] UTF8String]);
    Model m;
    const double t0 = now();
    m.load(q, model, ctx, chunk);
    std::fprintf(stderr, "loaded in %.1f s\n", now() - t0);

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
        for (int s = 0; s < P; s += chunk) m.forward(pr.data() + s, std::min(chunk, P - s), s, false);
        int agree = 0;
        double margin_sum = 0;
        auto score = [&](const float* lg, int want) {
          Top t = top2(lg, m.vocab(), {});
          agree += t.id == want;
          margin_sum += t.v - lg[want];
        };
        score(m.logits(), int(cont[0]));
        for (int s = 0; s + 1 < R; s += chunk) {
          const int n = std::min(chunk, R - 1 - s);
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
      for (int s = 0; s < P; s += chunk) m.forward(pr.data() + s, std::min(chunk, P - s), s, false);
      const double prefill_s = now() - tp;
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
      for (int k = 0; k < max_tokens; ++k) {
        const float* lg = m.logits();
        Top t = top2(lg, m.vocab(), banned);
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
