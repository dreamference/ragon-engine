// attention_rows on one backend (BACKEND §7, W1): the rows path's attention over a paged, head-major BF16 pool, plain
// and with the FP8 KV cache (--kv-fp8, LOWBIT §8; rounded in place, and from the sealed E4M3 mirror).
//   1. kv_seal's codes for all 65536 BF16 patterns at six scales, hashed against the CUDA build's codes.
//   2. Row invariance: each row of a 24-row verify byte-equal to the same row alone (a decode at its position).
//   3. Two runs byte-identical.
//   4. The rows against a double-precision reference (BF16 queries and probabilities; FP8 over R of the older units' K and V), within 1%.
// --bench: one layer's attention_rows at 25.7K and 32K keys, M = 1, 16 and 24, BF16 and the sealed FP8 mirror, cycling over
// four layers' pools so no layer's KV sits in the last-level cache. --seal-ref FILE: the CUDA codes (6 x 65536 bytes) to
// count differences against when the hashes disagree. Exits 0 when every check holds.
#include "gpu_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.cuh"

namespace k = ragon::kernels;

namespace {

template <typename T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  ragon::test::alloc_into(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
  ragon::test::h2d(d, h.data(), h.size() * sizeof(T));
  return d;
}
template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  (void)ragon::test::sync_error();
  ragon::test::d2h(h.data(), d, n * sizeof(T));
  return h;
}

constexpr int Hq = 24, Hkv = 4, D = 256, PT = k::kKvPageTokens, kRows = 24;
std::mt19937 rng(7151);
// A normal sample from the generator's raw words (Box-Muller): std::normal_distribution differs between the
// standard libraries, and the HIP (Windows) and CUDA (Linux) builds must draw the same data.
float normal(std::mt19937& g) {
  const double u1 = (double(g()) + 1.0) / 4294967297.0, u2 = double(g()) / 4294967296.0;
  return float(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
}

uint16_t bf16_bits(float f) {  // round to nearest even (finite inputs)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float bf16_float(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// R(x) on the host: bf16(float(e4m3(x * inv)) * s), the value the FP8 KV paths compute with (round to nearest even,
// saturated to 448, as kv_seal; finite inputs). exact: float(e4m3(x * inv)) * s in double, no BF16 rounding.
double q8_round(float x, float s, float inv, bool exact = false) {
  const float v = x * inv;
  uint32_t u;
  std::memcpy(&u, &v, 4);
  const uint32_t a = u & 0x7fffffffu;
  float m;
  std::memcpy(&m, &a, 4);
  float e;  // |float(e4m3(v))|
  if (a >= 0x43e80000u) e = 448.f;
  else if (a < 0x3c800000u) e = std::nearbyint(m * 512.f) / 512.f;
  else {
    const uint32_t r = (a + 0x7ffffu + ((a >> 20) & 1u)) & ~0xfffffu;
    std::memcpy(&e, &r, 4);
    e = std::min(e, 448.f);
  }
  if (exact) return double(u >> 31 ? -e : e) * s;
  return bf16_float(bf16_bits((u >> 31 ? -e : e) * s));
}
uint64_t fnv(const uint8_t* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

int failures = 0;
void report(const std::string& what, bool ok, const char* yes = "ok", const char* no = "FAILED") {
  std::printf("%-96s %s\n", what.c_str(), ok ? yes : no);
  if (!ok) ++failures;
}
bool gpu_ok() {
  if (const char* e = ragon::test::sync_error()) {
    std::printf("GPU error: %s\n", e);
    return false;
  }
  return true;
}

// One layer's KV in a head-major pool ([K, V][head][page][PT][D], page 0 unused) of shuffled pages, its page table
// (entries in elements, units of one page), the E4M3 mirror and its sealed flags.
struct Pool {
  int ctx = 0, np = 0;
  size_t hs = 0, elems = 0;
  ragon::gpu::bf16* kv = nullptr;  // K at kv, V at kv + elems
  uint64_t* table = nullptr;
  uint8_t* mirror = nullptr;
  uint8_t* sealed = nullptr;
  std::vector<uint16_t> hk, hv;  // host copies (kept only when asked)
  ragon::gpu::bf16* k() const { return kv; }
  ragon::gpu::bf16* v() const { return kv + elems; }
};
Pool make_pool(int ctx, bool keep, float kscale = 1.5f, float vscale = 0.8f) {
  Pool p;
  p.ctx = ctx;
  p.np = (ctx + PT - 1) / PT;
  p.hs = size_t(p.np + 1) * PT * D;
  p.elems = size_t(Hkv) * p.hs;
  std::vector<uint16_t> h(2 * p.elems);
  for (size_t i = 0; i < p.elems; ++i) h[i] = bf16_bits(normal(rng) * kscale);
  for (size_t i = 0; i < p.elems; ++i) h[p.elems + i] = bf16_bits(normal(rng) * vscale);
  ragon::test::alloc_into(&p.kv, 2 * p.elems * 2);
  ragon::test::h2d(p.kv, h.data(), 2 * p.elems * 2);
  if (keep) {
    p.hk.assign(h.begin(), h.begin() + p.elems);
    p.hv.assign(h.begin() + p.elems, h.end());
  }
  std::vector<int> ids(p.np);
  for (int i = 0; i < p.np; ++i) ids[i] = i + 1;
  std::shuffle(ids.begin(), ids.end(), rng);
  std::vector<uint64_t> t(p.np);
  for (int i = 0; i < p.np; ++i) t[i] = uint64_t(ids[i]) * PT * D;
  p.table = upload(t);
  ragon::test::alloc_into(&p.mirror, 2 * p.elems);
  ragon::test::fill(p.mirror, 0xff, 2 * p.elems);  // NaN codes where nothing is sealed
  p.sealed = upload(std::vector<uint8_t>(p.np + 1, 0));
  return p;
}
void free_pool(Pool& p) {
  ragon::test::free_bytes(p.kv);
  ragon::test::free_bytes(p.table);
  ragon::test::free_bytes(p.mirror);
  ragon::test::free_bytes(p.sealed);
  p = Pool{};
}
// The layer's FP8 scales from its maxima (as bench/quant/kv_calib.py sets them), measured with kv_absmax.
std::vector<float> absmax(const Pool& p) {
  float* amax = upload(std::vector<float>{0.f, 0.f});
  k::kv_absmax(p.k(), p.v(), p.table, 0, p.ctx, Hkv, D, amax, nullptr);
  const std::vector<float> m = download(amax, 2);
  ragon::test::free_bytes(amax);
  return m;
}
k::KvQ8 scales(const Pool& p) {
  const std::vector<float> m = absmax(p);
  const float ks = m[0] / 448.f, vs = m[1] / 448.f;
  return k::KvQ8{ks, vs, 1.f / ks, 1.f / vs};
}
// Every page of the pool sealed; the KvQ8 that reads the mirror.
k::KvQ8 seal_all(const Pool& p, k::KvQ8 q8) {
  std::vector<int> ids(p.np);
  for (int i = 0; i < p.np; ++i) ids[i] = i + 1;
  const float kinv[1] = {q8.kinv}, vinv[1] = {q8.vinv};
  k::kv_seal(p.kv, p.mirror, p.sealed, ids.data(), p.np, 0, 1, Hkv, D, size_t(PT) * D, kinv, vinv, nullptr);
  q8.k8 = p.mirror;
  q8.v8 = p.mirror + p.elems;
  q8.sealed = p.sealed;
  q8.unit = size_t(PT) * D;
  return q8;
}

// ---- 1. the seal codes ----
const float kSealInv[6] = {1.0f, 0.37f, 17.25f, 1.0f / 3.0f, 3.0e-3f, 448.0f / 12.5f};
const uint64_t kSealCuda[6] = {0xd8b2feb674a31125ull, 0x8c7c0cf2b91b2701ull, 0x0088567593e61569ull,
                               0xad983abfd5e13881ull, 0xc15c3ce31e3b845dull, 0x9a5a20f2e6dd8aa5ull};
void seal_codes(const char* ref_path) {
  // Four pages a head hold the 65536 patterns: head h's page j + 1 holds patterns [j * PT * D, (j + 1) * PT * D).
  constexpr int np = 65536 / (PT * D);
  const size_t hs = size_t(np + 1) * PT * D, elems = size_t(Hkv) * hs;
  std::vector<uint16_t> h(2 * elems);
  for (size_t i = 0; i < 2 * elems; ++i) h[i] = static_cast<uint16_t>((i % hs) - size_t(PT) * D);
  ragon::gpu::bf16* pool = nullptr;
  ragon::test::alloc_into(&pool, h.size() * 2);
  ragon::test::h2d(pool, h.data(), h.size() * 2);
  uint8_t* mirror = nullptr;
  ragon::test::alloc_into(&mirror, 2 * elems);
  uint8_t* flags = upload(std::vector<uint8_t>(np + 1, 0));
  std::vector<int> ids;
  for (int i = 1; i <= np; ++i) ids.push_back(i);
  k::set_kv_layout(hs, D);
  std::vector<uint8_t> ref;
  if (ref_path) {
    if (FILE* f = std::fopen(ref_path, "rb")) {
      ref.resize(6 * 65536);
      if (std::fread(ref.data(), 1, ref.size(), f) != ref.size()) ref.clear();
      std::fclose(f);
    }
  }
  for (int c = 0; c < 6; ++c) {
    ragon::test::fill(mirror, 0, 2 * elems);
    const float inv[1] = {kSealInv[c]};
    k::kv_seal(pool, mirror, flags, ids.data(), np, 0, 1, Hkv, D, size_t(PT) * D, inv, inv, nullptr);
    const std::vector<uint8_t> m = download(mirror, 2 * elems);
    bool ok = gpu_ok();
    const uint64_t want = kSealCuda[c];
    for (int kvh = 0; kvh < 2 * Hkv; ++kvh) {  // K heads, then V heads: each the same 65536 codes
      const uint8_t* got = m.data() + size_t(kvh) * hs + size_t(PT) * D;
      const uint64_t hash = fnv(got, 65536);
      if (hash != want) {
        if (ok) {
          std::printf("  inv %g: head %d codes hash %016llx, CUDA %016llx\n", kSealInv[c], kvh,
                      static_cast<unsigned long long>(hash), static_cast<unsigned long long>(want));
          if (!ref.empty()) {
            int diff = 0;
            for (int i = 0; i < 65536; ++i)
              if (got[i] != ref[size_t(c) * 65536 + i] && diff++ < 8)
                std::printf("    pattern %04x: code %02x, CUDA %02x\n", i, got[i], ref[size_t(c) * 65536 + i]);
            std::printf("    %d of 65536 codes differ\n", diff);
          }
        }
        ok = false;
      }
    }
    char what[128];
    std::snprintf(what, sizeof what, "kv_seal codes of all 65536 BF16 patterns, inv %g, against the CUDA build's",
                  kSealInv[c]);
    report(what, ok, "identical", "DIFFERENT");
  }
  ragon::test::free_bytes(pool);
  ragon::test::free_bytes(mirror);
  ragon::test::free_bytes(flags);
}

// ---- 2-4. rows ----
struct Bufs {
  float *q = nullptr, *out = nullptr, *scratch = nullptr;
};
Bufs make_bufs(int max_ctx) {
  Bufs b;
  ragon::test::alloc_into(&b.q, size_t(kRows) * Hq * D * 4);
  ragon::test::alloc_into(&b.out, size_t(kRows) * Hq * D * 4);
  ragon::test::alloc_into(&b.scratch, k::attention_rows_scratch_floats(kRows, Hq, D, max_ctx) * 4);
  return b;
}
void free_bufs(Bufs& b) {
  ragon::test::free_bytes(b.q);
  ragon::test::free_bytes(b.out);
  ragon::test::free_bytes(b.scratch);
}
// Rows [m0, m0 + M) of the query block, at positions P + m0 .., over the pool: the output rows.
std::vector<float> rows(const Pool& p, const Bufs& b, int P, int m0, int M, const k::KvQ8& q8) {
  const size_t rowf = size_t(Hq) * D;
  ragon::test::fill(b.out, 0, size_t(M) * rowf * 4);
  k::attention_rows(b.q + size_t(m0) * rowf, p.k(), p.v(), P + m0, M, Hq, Hkv, D, b.scratch, b.out, nullptr, 0,
                    nullptr, p.table, q8);
  return download(b.out, size_t(M) * rowf);
}

// The reference of row m (position pos): softmax(q k / 16) v in double, with the kernel's BF16 roundings of q / 16
// and of the probabilities (relative to the row's maximum) before the V product.
// q8: the FP8 formats' scales, or nullptr for BF16. With them the keys of every 32-key unit below the row's own
// (j < pos & ~31) are taken as R of their values, its own unit's as stored (LOWBIT §8).
// exact: no rounding of q / 16 nor of the probabilities, and an FP8 value as its code times the scale (the
// attention the format means, whatever the kernel's operand types).
std::vector<double> reference(const Pool& p, const std::vector<float>& hq, int m, int pos, const k::KvQ8* q8,
                              bool exact = false) {
  const auto kv = [&](uint16_t b, bool v, int j) -> double {
    const float x = bf16_float(b);
    return q8 && j < (pos & ~31) ? q8_round(x, v ? q8->vs : q8->ks, v ? q8->vinv : q8->kinv, exact) : x;
  };
  std::vector<double> o(size_t(Hq) * D, 0.0);
  const std::vector<uint64_t> t = download(p.table, p.np);
  std::vector<double> s(pos + 1);
  for (int h = 0; h < Hq; ++h) {
    const int kvh = h / (Hq / Hkv);
    float qb[D];
    for (int d = 0; d < D; ++d) {
      const float x = hq[(size_t(m) * Hq + h) * D + d] * 0.0625f;
      qb[d] = exact ? x : bf16_float(bf16_bits(x));
    }
    double mx = -1e300;
    for (int j = 0; j <= pos; ++j) {
      const size_t off = size_t(kvh) * p.hs + t[j / PT] + size_t(j % PT) * D;
      double acc = 0.0;
      for (int d = 0; d < D; ++d) acc += double(qb[d]) * kv(p.hk[off + d], false, j);
      s[j] = acc;
      mx = std::max(mx, acc);
    }
    double l = 0.0;
    for (int j = 0; j <= pos; ++j) {
      s[j] = std::exp(s[j] - mx);
      l += s[j];
    }
    for (int j = 0; j <= pos; ++j) {
      const size_t off = size_t(kvh) * p.hs + t[j / PT] + size_t(j % PT) * D;
      const double pj = exact ? s[j] : bf16_float(bf16_bits(float(s[j])));
      for (int d = 0; d < D; ++d) o[size_t(h) * D + d] += pj * kv(p.hv[off + d], true, j);
    }
    for (int d = 0; d < D; ++d) o[size_t(h) * D + d] /= l;
  }
  return o;
}

void rows_checks() {
  const int max_ctx = 9000;
  Pool p = make_pool(max_ctx, true);
  k::set_kv_layout(p.hs, D);
  Bufs b = make_bufs(max_ctx);
  std::vector<float> hq(size_t(kRows) * Hq * D);
  for (auto& x : hq) x = normal(rng) * 2.f;
  ragon::test::h2d(b.q, hq.data(), hq.size() * 4);
  const k::KvQ8 q8 = scales(p);
  {  // kv_absmax against the host's maxima
    // The host maxima over the pool's used positions only (page 0 and the tail past ctx are not the layer's).
    float mk = 0.f, mv = 0.f;
    const std::vector<uint64_t> t = download(p.table, p.np);
    for (int h = 0; h < Hkv; ++h)
      for (int j = 0; j < p.ctx; ++j)
        for (int d = 0; d < D; ++d) {
          const size_t off = size_t(h) * p.hs + t[j / PT] + size_t(j % PT) * D + d;
          mk = std::max(mk, std::fabs(bf16_float(p.hk[off])));
          mv = std::max(mv, std::fabs(bf16_float(p.hv[off])));
        }
    const std::vector<float> m = absmax(p);
    report("kv_absmax: the layer's largest |K| and |V|", m[0] == mk && m[1] == mv);
  }
  const k::KvQ8 m8 = seal_all(p, q8);
  (void)gpu_ok();
  const k::KvQ8 formats[3] = {k::KvQ8{}, q8, m8};
  const char* names[3] = {"BF16 KV", "FP8 KV (rounded in place)", "FP8 KV (sealed E4M3 mirror)"};
  const size_t rowf = size_t(Hq) * D;
  const int starts[] = {0, 9, 41, 4072, 4085, 8192 - 30, max_ctx - kRows};
  for (int f = 0; f < 3; ++f) {
    bool inv_ok = gpu_ok(), det_ok = inv_ok;
    for (int P : starts) {
      const std::vector<float> verify = rows(p, b, P, 0, kRows, formats[f]);
      const std::vector<float> again = rows(p, b, P, 0, kRows, formats[f]);
      if (std::memcmp(verify.data(), again.data(), verify.size() * 4) != 0) {
        if (det_ok) std::printf("  %s, P %d: two verify runs differ\n", names[f], P);
        det_ok = false;
      }
      for (int m = 0; m < kRows; ++m) {
        const std::vector<float> one = rows(p, b, P, m, 1, formats[f]);
        if (std::memcmp(one.data(), &verify[size_t(m) * rowf], rowf * 4) != 0) {
          if (inv_ok) {
            int n = 0, first = -1;
            float worst = 0.f;
            for (size_t i = 0; i < rowf; ++i)
              if (std::memcmp(&one[i], &verify[size_t(m) * rowf + i], 4) != 0) {
                if (first < 0) first = int(i);
                ++n;
                worst = std::max(worst, std::fabs(one[i] - verify[size_t(m) * rowf + i]));
              }
            std::printf("  %s, P %d: row %d alone differs from the row in the 24-row verify: %d values, worst %.3e, "
                        "first head %d dim %d (%.6e vs %.6e)\n", names[f], P, m, n, worst, first / D, first % D,
                        one[first], verify[size_t(m) * rowf + first]);
          }
          inv_ok = false;
        }
      }
      for (int M : {5, 16}) {  // other verify widths: the same rows
        const std::vector<float> part = rows(p, b, P, 0, M, formats[f]);
        if (std::memcmp(part.data(), verify.data(), part.size() * 4) != 0) {
          if (inv_ok) std::printf("  %s, P %d: the %d-row verify's rows differ from the 24-row verify's\n", names[f], P, M);
          inv_ok = false;
        }
      }
      inv_ok &= gpu_ok();
    }
    report(std::string("row invariance (decode row == the row in a 5/16/24-row verify), ") + names[f], inv_ok,
           "identical", "DIFFERENT");
    report(std::string("two runs byte-identical, ") + names[f], det_ok, "identical", "DIFFERENT");
  }
  // The rows against the double reference (for FP8 over R of the keys and values).
  // Twice: with the BF16 kernel's roundings, and exact (no roundings; the FP8 values as code times scale), where the
  // RMS error over the rows is printed too.
  for (bool exact : {false, true})
    for (int f = 0; f < 3; ++f) {
      double worst = 0.0, e2 = 0.0, r2 = 0.0;
      for (int P : {0, 41, 4085}) {
        const std::vector<float> got = rows(p, b, P, 0, kRows, formats[f]);
        for (int m : {0, 7, kRows - 1}) {
          const std::vector<double> ref = reference(p, hq, m, P + m, f == 0 ? nullptr : &q8, exact);
          double mx = 0.0, err = 0.0;
          for (size_t i = 0; i < rowf; ++i) {
            const double e = got[size_t(m) * rowf + i] - ref[i];
            mx = std::max(mx, std::fabs(ref[i]));
            err = std::max(err, std::fabs(e));
            e2 += e * e, r2 += ref[i] * ref[i];
          }
          worst = std::max(worst, err / std::max(mx, 1e-30));
        }
      }
      const double tol = 1e-2;
      char what[200];
      std::snprintf(what, sizeof what, "against the %s reference, %s: worst error %.2e of the row's largest, RMS %.2e "
                    "(tol %.0e)", exact ? "exact" : "double", names[f], worst, std::sqrt(e2 / std::max(r2, 1e-300)), tol);
      report(what, worst < tol && std::isfinite(worst));
    }
  free_bufs(b);
  free_pool(p);
}

// ---- --bench ----
void bench() {
  constexpr int kLayers = 4;
  for (int ctx : {25700, 32768}) {
    Pool pools[kLayers];
    k::KvQ8 m8[kLayers];
    for (int l = 0; l < kLayers; ++l) {
      pools[l] = make_pool(ctx, false);
      k::set_kv_layout(pools[l].hs, D);
      m8[l] = seal_all(pools[l], scales(pools[l]));
    }
    Bufs b = make_bufs(ctx);
    ragon::test::fill(b.q, 0, size_t(kRows) * Hq * D * 4);
    (void)gpu_ok();
    for (int fmt = 0; fmt < 2; ++fmt)
      for (int M : {1, 16, kRows}) {
        const int P = ctx - M;
        auto run = [&](int l) {
          k::attention_rows(b.q, pools[l].k(), pools[l].v(), P, M, Hq, Hkv, D, b.scratch, b.out, nullptr, 0, nullptr,
                            pools[l].table, fmt ? m8[l] : k::KvQ8{});
        };
        for (int l = 0; l < kLayers; ++l) run(l);  // warm
        (void)gpu_ok();
        const int reps = 64;
        ragon::test::Timer timer;
        timer.start();
        for (int r = 0; r < reps; ++r) run(r % kLayers);
        const float ms = timer.stop() / reps;
        (void)gpu_ok();
        std::printf("bench attention_rows ctx %5d M %2d %-4s: %.4f ms per layer, %.3f ms per 16 layers\n", ctx, M,
                    fmt ? "FP8" : "BF16", ms, 16.f * ms);
      }
    free_bufs(b);
    for (auto& pl : pools) free_pool(pl);
  }
}

}  // namespace

int main(int argc, char** argv) {
  k::prepare_kernels();
  bool do_bench = false;
  const char* ref = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--bench")) do_bench = true;
    if (!std::strcmp(argv[i], "--seal-ref") && i + 1 < argc) ref = argv[++i];
  }
  if (do_bench) {
    bench();
    return gpu_ok() ? 0 : 1;
  }
  seal_codes(ref);
  rows_checks();
  std::printf("%s\n", failures ? "FAILED" : "all attention_rows checks hold");
  return failures ? 1 : 0;
}
