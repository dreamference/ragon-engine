// The prefill GEMMs and activation quantizers (src/core/prefill_gemm_hip.cu on HIP, prefill_gemm.cu on CUDA) against
// host references built from raw mt19937 words (the standard fixes them, so both backends see the same inputs):
//   - quant_act_nvfp4 / quant_act_fp8 / quant_act_fp8_bf16in byte for byte the CUDA recipe (bf16 rounding, the E4M3
//     block scale, E2M1 nearest-even satfinite), on random rows and on a sweep of the E2M1 midpoints, one BF16 ulp
//     either side of each, signed zeros, saturated scales and infinities. QHASH prints their bytes' FNV hash: equal on
//     both backends means CUDA's bytes.
//   - silu_mul_quant_nvfp4, rmsnorm_quant, gated_rmsnorm_fp8 and sigmoid_mul_fp8 (exp and rsqrt from the hardware)
//     within a code step of the host on a small share of values.
//   - HIP only: prefill_gemm_nvfp4 / _fp8 / _fp8_bf16out (with and without accumulate) against an FP64 product, every
//     row's bytes independent of M and of the launch's row slab; prefill_swiglu_nvfp4 byte for byte the plain GEMMs
//     followed by silu_mul_quant_nvfp4; narrow_bf16 against FP64.
// HASH prints every output's FNV hash so two runs compare byte for byte.
#include "gpu_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "backend/device.hpp"
#include "core/kernels.cuh"

namespace {

namespace gpu = ragon::gpu;
namespace K = ragon::kernels;
using ragon::test::download;
using ragon::test::e4m3;
using ragon::test::upload;

const float kFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

uint64_t fnv(const void* p, size_t n, uint64_t h) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}
constexpr uint64_t kFnv0 = 1469598103934665603ull;
uint64_t g_qhash = kFnv0, g_hash = kFnv0;

float unit(std::mt19937& rng) { return static_cast<float>(rng() >> 8) * (1.f / 16777216.f); }
float value(std::mt19937& rng) {
  const float v = (unit(rng) + unit(rng) + unit(rng) + unit(rng) - 2.f) * 1.7f;
  return (rng() % 97 == 0) ? v * 30.f : v;
}
uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// BF16 round to nearest even (no NaN in these tests; infinities stay).
float bf16r(float x) {
  uint32_t u = bits(x);
  if ((u & 0x7f800000u) == 0x7f800000u) return x;
  u += 0x7fffu + ((u >> 16) & 1u);
  return from_bits(u & 0xffff0000u);
}
uint16_t bf16_bits(float x) { return static_cast<uint16_t>(bits(bf16r(x)) >> 16); }

// E2M1 nearest, ties to the even code, saturated (cvt.rn.satfinite.e2m1x2), by search over the grid.
uint8_t e2m1(float x) {
  const float a = std::fabs(x);
  int best = 0;
  for (int i = 1; i < 8; ++i) {
    const float d = std::fabs(kFp4[i] - a), db = std::fabs(kFp4[best] - a);
    if (d < db || (d == db && (i % 2 == 0))) best = i;
  }
  if (a > 6.f) best = 7;
  return static_cast<uint8_t>(best | (std::signbit(x) ? 8 : 0));
}

// The CUDA recipe on the host.
void host_nvfp4(const std::vector<float>& v, float input_scale, std::vector<uint8_t>& q, std::vector<uint8_t>& sf) {
  const float g = 1.f / input_scale;
  q.assign(v.size() / 2, 0);
  sf.assign(v.size() / 16, 0);
  for (size_t b = 0; b < v.size() / 16; ++b) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(v[b * 16 + i]));
    const uint8_t sc = gpu::e4m3_from_float(g * (amax * (1.f / 6.f)));
    const float sv = e4m3(sc);
    const float os = sv != 0.f ? 1.f / (sv * (1.f / g)) : 0.f;
    for (int i = 0; i < 16; i += 2)
      q[(b * 16 + i) / 2] = static_cast<uint8_t>(e2m1(v[b * 16 + i] * os) | (e2m1(v[b * 16 + i + 1] * os) << 4));
    sf[b] = sc;
  }
}
std::vector<uint8_t> host_fp8(const std::vector<float>& v, float input_scale) {
  const float inv = 1.f / input_scale;
  std::vector<uint8_t> q(v.size());
  for (size_t i = 0; i < v.size(); ++i) q[i] = gpu::e4m3_from_float(v[i] * inv);
  return q;
}
std::vector<float> rounded(const std::vector<float>& x) {
  std::vector<float> v(x.size());
  for (size_t i = 0; i < x.size(); ++i) v[i] = bf16r(x[i]);
  return v;
}
size_t first_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return i;
  return a.size();
}

// The pure quantizers on x (M x K): byte for byte the host recipe; their bytes go into QHASH.
bool check_pure(const char* tag, const std::vector<float>& x, int M, int Kd, float in4, float in8) {
  float* dx = upload(x);
  std::vector<uint16_t> xb(x.size());
  for (size_t i = 0; i < x.size(); ++i) xb[i] = bf16_bits(x[i]);
  uint16_t* dxb = upload(xb);
  const size_t n = size_t(M) * Kd;
  uint8_t *q4 = ragon::test::alloc<uint8_t>(n / 2), *s4 = ragon::test::alloc<uint8_t>(n / 16),
          *q8 = ragon::test::alloc<uint8_t>(n), *q8b = ragon::test::alloc<uint8_t>(n);
  K::quant_act_nvfp4(dx, M, Kd, in4, q4, s4, nullptr);
  K::quant_act_fp8(dx, M, Kd, in8, q8, nullptr);
  K::quant_act_fp8_bf16in(reinterpret_cast<const gpu::bf16*>(dxb), M, Kd, in8, q8b, nullptr);
  const char* err = ragon::test::sync_error();
  const auto g4 = download(q4, n / 2), gs = download(s4, n / 16), g8 = download(q8, n), g8b = download(q8b, n);
  std::vector<uint8_t> h4, hs;
  const auto v = rounded(x);
  host_nvfp4(v, in4, h4, hs);
  const auto h8 = host_fp8(v, in8);
  const size_t d4 = first_diff(g4, h4), ds = first_diff(gs, hs), d8 = first_diff(g8, h8), d8b = first_diff(g8b, h8);
  const bool ok = !err && d4 == h4.size() && ds == hs.size() && d8 == h8.size() && d8b == h8.size();
  std::printf("quant %s M=%d K=%d: nvfp4 codes %s, scales %s, fp8 %s, fp8 bf16in %s%s%s %s\n", tag, M, Kd,
              d4 == h4.size() ? "=" : "DIFF", ds == hs.size() ? "=" : "DIFF", d8 == h8.size() ? "=" : "DIFF",
              d8b == h8.size() ? "=" : "DIFF", err ? ", error " : "", err ? err : "", ok ? "ok" : "FAIL");
  if (d4 != h4.size())
    std::printf("  first nvfp4 code diff at byte %zu: gpu %02x host %02x (x %a %a)\n", d4, g4[d4], h4[d4], x[2 * d4],
                x[2 * d4 + 1]);
  if (ds != hs.size()) std::printf("  first scale diff at block %zu: gpu %02x host %02x\n", ds, gs[ds], hs[ds]);
  if (d8 != h8.size()) std::printf("  first fp8 diff at %zu: gpu %02x host %02x (x %a)\n", d8, g8[d8], h8[d8], x[d8]);
  for (const auto* h : {&g4, &gs, &g8, &g8b}) g_qhash = fnv(h->data(), h->size(), g_qhash);
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dxb), static_cast<void*>(q4), static_cast<void*>(s4),
                  static_cast<void*>(q8), static_cast<void*>(q8b)})
    ragon::test::free_bytes(p);
  return ok;
}

bool test_quant_random() {
  std::mt19937 rng(41);
  const int M = 67, Kd = 512;
  std::vector<float> x(size_t(M) * Kd);
  for (auto& v : x) v = value(rng) * 0.003f;
  bool ok = check_pure("random", x, M, Kd, 0.0019763766f, 0.2053571f);
  std::vector<float> y(size_t(5) * 5120);
  for (auto& v : y) v = value(rng) * (1.f + 40.f * unit(rng));
  ok &= check_pure("wide", y, 5, 5120, 0.37f, 0.051f);
  return ok;
}

// E2M1 midpoints, one BF16 ulp either side, signed zeros, blocks whose scale saturates and infinities. A block's first
// value is 6s (s a power of two, input scale 1), so its scale is s and the rest scale by 1/s exactly.
bool test_quant_sweep() {
  std::vector<float> pool;
  for (float m : {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}) {
    pool.push_back(m);
    pool.push_back(from_bits(bits(m) + 0x10000u));
    pool.push_back(from_bits(bits(m) - 0x10000u));
  }
  pool.push_back(0.f);
  pool.push_back(-0.f);
  pool.push_back(1e-30f);
  pool.push_back(-1e-30f);
  const int Kd = 512, blocks_per_row = Kd / 16, M = 12;
  std::vector<float> x(size_t(M) * Kd);
  std::mt19937 rng(43);
  size_t at = 0;
  for (int b = 0; b < M * blocks_per_row; ++b) {
    float* blk = x.data() + size_t(b) * 16;
    const float s = std::ldexp(1.f, -6 + b % 11);
    if (b % 29 == 7) {  // saturated scale: amax / 6 above 448
      for (int i = 0; i < 16; ++i) blk[i] = (rng() & 1 ? -1.f : 1.f) * bf16r(3000.f + 977.f * i);
      continue;
    }
    if (b % 31 == 11) {  // an infinity in the block
      for (int i = 0; i < 16; ++i) blk[i] = pool[(at++) % pool.size()] * s;
      blk[5] = std::numeric_limits<float>::infinity();
      blk[9] = -std::numeric_limits<float>::infinity();
      continue;
    }
    if (b % 37 == 3) {  // all zeros: scale 0
      for (int i = 0; i < 16; ++i) blk[i] = (i & 1) ? -0.f : 0.f;
      continue;
    }
    blk[b % 16] = (b & 1 ? -6.f : 6.f) * s;
    for (int i = 0; i < 16; ++i)
      if (i != b % 16) blk[i] = (rng() & 1 ? -1.f : 1.f) * pool[(at++) % pool.size()] * s;
  }
  return check_pure("e2m1 sweep", x, M, Kd, 1.f, 1.f);
}

// Values at which two quantizations differ by more than one code step (NVFP4: magnitude index; FP8: byte).
size_t code_far(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, bool nibbles) {
  size_t far = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (nibbles) {
      for (int sh : {0, 4}) {
        const int ca = (a[i] >> sh) & 15, cb = (b[i] >> sh) & 15;
        if (ca != cb && ((ca ^ cb) & 8 ? (ca & 7) + (cb & 7) > 1 : std::abs((ca & 7) - (cb & 7)) > 1)) ++far;
      }
    } else if (a[i] != b[i] && std::abs(int(a[i] & 0x7f) - int(b[i] & 0x7f)) > 1) {
      ++far;
    }
  }
  return far;
}
size_t count_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  size_t d = 0;
  for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
  return d;
}

// The fused quantizers against the host with std::exp / 1/sqrt: no value more than one code step off, at most 1 %
// of the bytes different; the bytes go into HASH (same backend twice), not QHASH.
bool test_fused() {
  std::mt19937 rng(47);
  bool ok = true;
  {  // silu_mul_quant_nvfp4
    const int M = 33, Kd = 1024;
    std::vector<float> g(size_t(M) * Kd), u(g.size());
    for (auto& v : g) v = value(rng);
    for (auto& v : u) v = value(rng);
    const float in4 = 0.0117f;
    float *dg = upload(g), *du = upload(u);
    uint8_t *q = ragon::test::alloc<uint8_t>(g.size() / 2), *s = ragon::test::alloc<uint8_t>(g.size() / 16);
    K::silu_mul_quant_nvfp4(dg, du, M, Kd, in4, q, s, nullptr);
    const char* err = ragon::test::sync_error();
    const auto gq = download(q, g.size() / 2), gsf = download(s, g.size() / 16);
    std::vector<float> v(g.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = bf16r(g[i] / (1.f + std::exp(-g[i])) * u[i]);
    std::vector<uint8_t> hq, hs;
    host_nvfp4(v, in4, hq, hs);
    const size_t far = code_far(gq, hq, true) + code_far(gsf, hs, false), diff = count_diff(gq, hq) + count_diff(gsf, hs);
    const bool good = !err && far == 0 && diff * 100 <= gq.size() + gsf.size();
    std::printf("silu_mul_quant_nvfp4 M=%d K=%d: %zu bytes differ from the host, %zu beyond a step%s%s %s\n", M, Kd, diff,
                far, err ? ", error " : "", err ? err : "", good ? "ok" : "FAIL");
    g_hash = fnv(gq.data(), gq.size(), g_hash);
    g_hash = fnv(gsf.data(), gsf.size(), g_hash);
    ok &= good;
    for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(q), static_cast<void*>(s)})
      ragon::test::free_bytes(p);
  }
  for (bool fp4 : {true, false}) {  // rmsnorm_quant
    const int M = 9, H = 5120;
    std::vector<float> x(size_t(M) * H);
    for (auto& v : x) v = value(rng);
    std::vector<uint16_t> w(H);
    for (auto& b : w) b = bf16_bits(value(rng) * 0.1f);
    const float in = fp4 ? 0.0021f : 0.0313f, eps = 1e-6f;
    float* dx = upload(x);
    uint16_t* dw = upload(w);
    float* dout = ragon::test::alloc<float>(x.size());
    const size_t qn = fp4 ? x.size() / 2 : x.size();
    uint8_t *q = ragon::test::alloc<uint8_t>(qn), *s = ragon::test::alloc<uint8_t>(x.size() / 16);
    K::rmsnorm_quant(dx, reinterpret_cast<const gpu::bf16*>(dw), dout, M, H, eps, fp4, in, q, s, nullptr);
    const char* err = ragon::test::sync_error();
    const auto go = download(dout, x.size());
    const auto gq = download(q, qn);
    std::vector<float> v(x.size());
    double worst = 0;
    for (int r = 0; r < M; ++r) {
      double ss = 0;
      for (int i = 0; i < H; ++i) ss += double(x[size_t(r) * H + i]) * x[size_t(r) * H + i];
      const float rstd = float(1.0 / std::sqrt(ss / H + eps));
      for (int i = 0; i < H; ++i) {
        const float wv = from_bits(uint32_t(w[i]) << 16);
        const float o = x[size_t(r) * H + i] * rstd * (1.f + wv);
        worst = std::max(worst, std::fabs(double(go[size_t(r) * H + i]) - o) / (std::fabs(o) + 1e-3));
        v[size_t(r) * H + i] = bf16r(o);
      }
    }
    std::vector<uint8_t> hq, hs;
    size_t far, diff, total;
    if (fp4) {
      host_nvfp4(v, in, hq, hs);
      const auto gsf = download(s, x.size() / 16);
      far = code_far(gq, hq, true) + code_far(gsf, hs, false);
      diff = count_diff(gq, hq) + count_diff(gsf, hs);
      total = gq.size() + gsf.size();
      g_hash = fnv(gsf.data(), gsf.size(), g_hash);
    } else {
      hq = host_fp8(v, in);
      far = code_far(gq, hq, false);
      diff = count_diff(gq, hq);
      total = gq.size();
    }
    const bool good = !err && far == 0 && diff * 100 <= total && worst < 1e-5;
    std::printf("rmsnorm_quant %s M=%d H=%d: out rel %.2e, %zu bytes differ, %zu beyond a step%s%s %s\n",
                fp4 ? "nvfp4" : "fp8", M, H, worst, diff, far, err ? ", error " : "", err ? err : "", good ? "ok" : "FAIL");
    g_hash = fnv(gq.data(), gq.size(), g_hash);
    g_hash = fnv(go.data(), go.size() * 4, g_hash);
    ok &= good;
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dw), static_cast<void*>(dout), static_cast<void*>(q),
                    static_cast<void*>(s)})
      ragon::test::free_bytes(p);
  }
  {  // gated_rmsnorm_fp8 and sigmoid_mul_fp8
    const int rows = 77, D = 128;
    std::vector<float> x(size_t(rows) * D), z(x.size());
    for (auto& v : x) v = value(rng);
    for (auto& v : z) v = value(rng);
    std::vector<uint16_t> w(D);
    for (auto& b : w) b = bf16_bits(1.f + value(rng) * 0.1f);
    float *dx = upload(x), *dz = upload(z);
    uint16_t* dw = upload(w);
    uint8_t *q = ragon::test::alloc<uint8_t>(x.size()), *q2 = ragon::test::alloc<uint8_t>(x.size());
    const float in = 0.0097f, eps = 1e-6f;
    K::gated_rmsnorm_fp8(dx, dz, reinterpret_cast<const gpu::bf16*>(dw), rows, D, eps, in, q, nullptr);
    K::sigmoid_mul_fp8(dx, dz, x.size(), in, q2, nullptr);
    const char* err = ragon::test::sync_error();
    const auto gq = download(q, x.size()), gq2 = download(q2, x.size());
    std::vector<float> v(x.size()), v2(x.size());
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int i = 0; i < D; ++i) ss += double(x[size_t(r) * D + i]) * x[size_t(r) * D + i];
      const float rstd = float(1.0 / std::sqrt(ss / D + eps));
      for (int i = 0; i < D; ++i) {
        const size_t j = size_t(r) * D + i;
        const float wv = from_bits(uint32_t(w[i]) << 16);
        v[j] = bf16r(x[j] * rstd * wv * (z[j] / (1.f + std::exp(-z[j]))));
        v2[j] = bf16r(x[j] / (1.f + std::exp(-z[j])));
      }
    }
    const auto hq = host_fp8(v, in), hq2 = host_fp8(v2, in);
    const size_t far = code_far(gq, hq, false), diff = count_diff(gq, hq);
    const size_t far2 = code_far(gq2, hq2, false), diff2 = count_diff(gq2, hq2);
    const bool good = !err && far == 0 && far2 == 0 && diff * 100 <= gq.size() && diff2 * 100 <= gq2.size();
    std::printf("gated_rmsnorm_fp8 rows=%d: %zu differ, %zu beyond a step; sigmoid_mul_fp8: %zu differ, %zu beyond%s%s %s\n",
                rows, diff, far, diff2, far2, err ? ", error " : "", err ? err : "", good ? "ok" : "FAIL");
    g_hash = fnv(gq.data(), gq.size(), g_hash);
    g_hash = fnv(gq2.data(), gq2.size(), g_hash);
    ok &= good;
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dz), static_cast<void*>(dw), static_cast<void*>(q),
                    static_cast<void*>(q2)})
      ragon::test::free_bytes(p);
  }
  return ok;
}

#ifdef RAGON_BACKEND_HIP
// ---- The HIP GEMMs (weights in stream_gemm_hip.cu's tiled layout) ----

struct Weights {
  int N, Kd;
  std::vector<uint8_t> w4, s4, w8;
  uint8_t *t4 = nullptr, *t8 = nullptr;
  std::vector<double> d4, d8;  // dequantized [N][K]
};
Weights make_weights(std::mt19937& rng, int N, int Kd) {
  Weights m{N, Kd, std::vector<uint8_t>(size_t(N) * Kd / 2), std::vector<uint8_t>(size_t(N) * Kd / 16),
            std::vector<uint8_t>(size_t(N) * Kd)};
  for (auto& b : m.w4) b = static_cast<uint8_t>(rng());
  for (auto& b : m.s4) b = static_cast<uint8_t>(0x28 + rng() % 0x21);
  for (auto& b : m.w8) {
    do b = static_cast<uint8_t>(rng());
    while ((b & 0x7f) == 0x7f);
  }
  uint8_t *rw4 = upload(m.w4), *rs4 = upload(m.s4), *rw8 = upload(m.w8);
  m.t4 = ragon::test::alloc<uint8_t>(K::tiled_bytes_nvfp4(N, Kd));
  m.t8 = ragon::test::alloc<uint8_t>(K::tiled_bytes_fp8(N, Kd));
  K::retile_nvfp4(rw4, rs4, m.t4, N, Kd, nullptr);
  K::retile_fp8(rw8, m.t8, N, Kd, nullptr);
  (void)ragon::test::sync_error();
  for (void* p : {static_cast<void*>(rw4), static_cast<void*>(rs4), static_cast<void*>(rw8)}) ragon::test::free_bytes(p);
  m.d4.resize(size_t(N) * Kd);
  m.d8.resize(size_t(N) * Kd);
  for (size_t n = 0; n < size_t(N); ++n)
    for (int k = 0; k < Kd; ++k) {
      const uint8_t b = m.w4[(n * Kd + k) / 2];
      m.d4[n * Kd + k] = double(kFp4[(k & 1) ? b >> 4 : b & 15]) * e4m3(m.s4[n * (Kd / 16) + k / 16]);
      m.d8[n * Kd + k] = e4m3(m.w8[n * Kd + k]);
    }
  return m;
}

struct Acts {
  int M, Kd;
  std::vector<uint8_t> q4, s4, q8;
  std::vector<double> d4, d8;
};
Acts make_acts(std::mt19937& rng, int M, int Kd) {
  Acts a{M, Kd, std::vector<uint8_t>(size_t(M) * Kd / 2), std::vector<uint8_t>(size_t(M) * Kd / 16),
         std::vector<uint8_t>(size_t(M) * Kd)};
  for (auto& b : a.q4) b = static_cast<uint8_t>(rng());
  for (auto& b : a.s4) b = static_cast<uint8_t>(0x30 + rng() % 0x21);
  for (auto& b : a.q8) {
    do b = static_cast<uint8_t>(rng());
    while ((b & 0x7f) == 0x7f || (b & 0x7f) > 0x60);
  }
  a.d4.resize(size_t(M) * Kd);
  a.d8.resize(size_t(M) * Kd);
  for (size_t m = 0; m < size_t(M); ++m)
    for (int k = 0; k < Kd; ++k) {
      const uint8_t b = a.q4[(m * Kd + k) / 2];
      a.d4[m * Kd + k] = double(kFp4[(k & 1) ? b >> 4 : b & 15]) * e4m3(a.s4[m * (Kd / 16) + k / 16]);
      a.d8[m * Kd + k] = e4m3(a.q8[m * Kd + k]);
    }
  return a;
}

// |y - ref| within 1e-5 of the sum of |terms| (FP32 accumulation of exact products over K <= 5120).
bool near(const std::vector<float>& y, int ldy, const Acts& a, const Weights& w, bool fp4, float alpha,
          const std::vector<float>* old, double& worst) {
  const auto& xd = fp4 ? a.d4 : a.d8;
  const auto& wd = fp4 ? w.d4 : w.d8;
  bool ok = true;
  for (int m = 0; m < a.M; ++m)
    for (int n = 0; n < w.N; ++n) {
      double s = 0, sa = 0;
      for (int k = 0; k < w.Kd; ++k) {
        const double t = xd[size_t(m) * w.Kd + k] * wd[size_t(n) * w.Kd + k];
        s += t;
        sa += std::fabs(t);
      }
      double ref = s * alpha;
      if (old) ref += (*old)[size_t(m) * ldy + n];
      const double err = std::fabs(y[size_t(m) * ldy + n] - ref), tol = 1e-5 * sa * alpha + 1e-6 * std::fabs(ref) + 1e-30;
      worst = std::max(worst, err / tol);
      if (err > tol) ok = false;
    }
  return ok;
}

bool test_gemm(int N, int Kd, uint32_t seed, const std::vector<int>& Ms) {
  std::mt19937 rng(seed);
  Weights w = make_weights(rng, N, Kd);
  const int MMAX = *std::max_element(Ms.begin(), Ms.end());
  Acts a = make_acts(rng, MMAX, Kd);
  uint8_t *dq4 = upload(a.q4), *ds4 = upload(a.s4), *dq8 = upload(a.q8);
  const int ldy = N + 16;
  std::vector<float> old(size_t(MMAX) * ldy);
  for (auto& v : old) v = value(rng);
  float* dy = ragon::test::alloc<float>(old.size());
  uint16_t* dyb = ragon::test::alloc<uint16_t>(old.size());
  const float alpha = 0.0137f;
  bool ok = true;
  std::vector<float> full4, full8;
  std::vector<uint16_t> fullb;
  for (int idx = 0; idx < int(Ms.size()); ++idx) {
    const int M = Ms[idx];
    double w4 = 0, w8 = 0, wa = 0;
    ragon::test::h2d(dy, old.data(), old.size() * 4);
    K::prefill_gemm_nvfp4(dq4, ds4, M, w.t4, alpha, dy, ldy, N, Kd, false, nullptr, nullptr);
    auto y4 = download(dy, old.size());
    ragon::test::h2d(dy, old.data(), old.size() * 4);
    K::prefill_gemm_fp8(dq8, M, w.t8, alpha, dy, ldy, N, Kd, false, nullptr);
    auto y8 = download(dy, old.size());
    ragon::test::h2d(dy, old.data(), old.size() * 4);
    K::prefill_gemm_fp8(dq8, M, w.t8, alpha, dy, ldy, N, Kd, true, nullptr);
    auto ya = download(dy, old.size());
    K::prefill_gemm_fp8_bf16out(dq8, M, w.t8, alpha, reinterpret_cast<gpu::bf16*>(dyb), ldy, N, Kd, nullptr);
    auto yb = download(dyb, old.size());
    const char* err = ragon::test::sync_error();
    Acts am = a;
    am.M = M;
    bool good = !err;
    good &= near(y4, ldy, am, w, true, alpha, nullptr, w4);
    good &= near(y8, ldy, am, w, false, alpha, nullptr, w8);
    good &= near(ya, ldy, am, w, false, alpha, &old, wa);
    // bf16out: the bf16 of the float result, bit for bit; nothing written past M or past N in a row.
    bool bfe = true, untouched = true;
    for (int m = 0; m < MMAX; ++m)
      for (int n = 0; n < ldy; ++n) {
        const size_t j = size_t(m) * ldy + n;
        if (m < M && n < N) bfe &= yb[j] == bf16_bits(y8[j]);
        else untouched &= y4[j] == old[j] && y8[j] == old[j];
      }
    good &= bfe && untouched;
    if (idx == 0) full4 = y4, full8 = y8, fullb = yb;
    // Every row's bytes those of the largest launch (Ms[0] is the largest).
    bool rows_same = true;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        const size_t j = size_t(m) * ldy + n;
        rows_same &= bits(y4[j]) == bits(full4[j]) && bits(y8[j]) == bits(full8[j]) && yb[j] == fullb[j];
      }
    good &= rows_same;
    std::printf("gemm N=%d K=%d M=%d: nvfp4 %.2f, fp8 %.2f, accumulate %.2f of tolerance; bf16out %s, rest untouched %s, "
                "rows = M=%d's %s%s%s %s\n",
                N, Kd, M, w4, w8, wa, bfe ? "yes" : "NO", untouched ? "yes" : "NO", Ms[0], rows_same ? "yes" : "NO",
                err ? ", error " : "", err ? err : "", good ? "ok" : "FAIL");
    for (const auto* h : {&y4, &y8, &ya}) g_hash = fnv(h->data(), h->size() * 4, g_hash);
    g_hash = fnv(yb.data(), yb.size() * 2, g_hash);
    ok &= good;
  }
  for (void* p : {static_cast<void*>(dq4), static_cast<void*>(ds4), static_cast<void*>(dq8), static_cast<void*>(dy),
                  static_cast<void*>(dyb), static_cast<void*>(w.t4), static_cast<void*>(w.t8)})
    ragon::test::free_bytes(p);
  return ok;
}

// Over two row slabs (kSlabRows = 2048): the second slab's rows equal a launch of those rows alone.
bool test_gemm_slabs() {
  std::mt19937 rng(53);
  const int N = 128, Kd = 256, M = 2100, tail = M - 2048;
  Weights w = make_weights(rng, N, Kd);
  Acts a = make_acts(rng, M, Kd);
  uint8_t *dq4 = upload(a.q4), *ds4 = upload(a.s4);
  float *dy = ragon::test::alloc<float>(size_t(M) * N), *dt = ragon::test::alloc<float>(size_t(tail) * N);
  K::prefill_gemm_nvfp4(dq4, ds4, M, w.t4, 0.5f, dy, N, N, Kd, false, nullptr, nullptr);
  K::prefill_gemm_nvfp4(dq4 + size_t(2048) * Kd / 2, ds4 + size_t(2048) * Kd / 16, tail, w.t4, 0.5f, dt, N, N, Kd, false,
                        nullptr, nullptr);
  const char* err = ragon::test::sync_error();
  const auto y = download(dy, size_t(M) * N), t = download(dt, size_t(tail) * N);
  double worst = 0;
  bool good = !err && near(y, N, a, w, true, 0.5f, nullptr, worst) &&
              std::memcmp(y.data() + size_t(2048) * N, t.data(), t.size() * 4) == 0;
  std::printf("gemm slabs M=%d: %.2f of tolerance, the second slab = its own launch%s%s %s\n", M, worst,
              err ? ", error " : "", err ? err : "", good ? "ok" : "FAIL");
  g_hash = fnv(y.data(), y.size() * 4, g_hash);
  for (void* p : {static_cast<void*>(dq4), static_cast<void*>(ds4), static_cast<void*>(dy), static_cast<void*>(dt),
                  static_cast<void*>(w.t4), static_cast<void*>(w.t8)})
    ragon::test::free_bytes(p);
  return good;
}

// prefill_swiglu_nvfp4 = prefill_gemm_nvfp4 (gate, up) then silu_mul_quant_nvfp4, byte for byte.
bool test_swiglu(int M, int N, int Kd) {
  std::mt19937 rng(59 + M);
  Weights wg = make_weights(rng, N, Kd), wu = make_weights(rng, N, Kd);
  Acts a = make_acts(rng, M, Kd);
  uint8_t *dq4 = upload(a.q4), *ds4 = upload(a.s4);
  const float ag = 0.0091f, au = 0.0113f, in_out = 0.0043f;
  uint8_t *q = ragon::test::alloc<uint8_t>(size_t(M) * N / 2), *s = ragon::test::alloc<uint8_t>(size_t(M) * N / 16);
  uint8_t *q2 = ragon::test::alloc<uint8_t>(size_t(M) * N / 2), *s2 = ragon::test::alloc<uint8_t>(size_t(M) * N / 16);
  float *g = ragon::test::alloc<float>(size_t(M) * N), *u = ragon::test::alloc<float>(size_t(M) * N);
  K::prefill_swiglu_nvfp4(dq4, ds4, M, wg.t4, ag, wu.t4, au, N, Kd, in_out, q, s, nullptr, nullptr, nullptr);
  K::prefill_gemm_nvfp4(dq4, ds4, M, wg.t4, ag, g, N, N, Kd, false, nullptr, nullptr);
  K::prefill_gemm_nvfp4(dq4, ds4, M, wu.t4, au, u, N, N, Kd, false, nullptr, nullptr);
  K::silu_mul_quant_nvfp4(g, u, M, N, in_out, q2, s2, nullptr);
  const char* err = ragon::test::sync_error();
  const auto gq = download(q, size_t(M) * N / 2), gs = download(s, size_t(M) * N / 16);
  const auto tq = download(q2, size_t(M) * N / 2), ts = download(s2, size_t(M) * N / 16);
  const bool good = !err && gq == tq && gs == ts;
  std::printf("swiglu M=%d N=%d K=%d: fused = gemm + silu_mul_quant: codes %s, scales %s%s%s %s\n", M, N, Kd,
              gq == tq ? "yes" : "NO", gs == ts ? "yes" : "NO", err ? ", error " : "", err ? err : "", good ? "ok" : "FAIL");
  if (gq != tq) {
    const size_t d = first_diff(gq, tq);
    std::printf("  first code diff at byte %zu (row %zu): fused %02x two-step %02x\n", d, d / (N / 2), gq[d], tq[d]);
  }
  g_hash = fnv(gq.data(), gq.size(), g_hash);
  g_hash = fnv(gs.data(), gs.size(), g_hash);
  for (void* p : {static_cast<void*>(dq4), static_cast<void*>(ds4), static_cast<void*>(q), static_cast<void*>(s),
                  static_cast<void*>(q2), static_cast<void*>(s2), static_cast<void*>(g), static_cast<void*>(u),
                  static_cast<void*>(wg.t4), static_cast<void*>(wg.t8), static_cast<void*>(wu.t4),
                  static_cast<void*>(wu.t8)})
    ragon::test::free_bytes(p);
  return good;
}
#endif

bool test_narrow() {
  std::mt19937 rng(61);
  const int M = 37, N = 48, Kd = 5120;
  std::vector<float> x(size_t(M) * Kd);
  for (auto& v : x) v = value(rng);
  std::vector<uint16_t> xb(x.size()), wa(size_t(N) * Kd), wb(size_t(N) * Kd);
  for (size_t i = 0; i < x.size(); ++i) xb[i] = bf16_bits(x[i]);
  for (auto& b : wa) b = bf16_bits(value(rng) * 0.02f);
  for (auto& b : wb) b = bf16_bits(value(rng) * 0.02f);
  float* dx = upload(x);
  uint16_t *dxb = upload(xb), *dwa = upload(wa), *dwb = upload(wb);
  float *ya = ragon::test::alloc<float>(size_t(M) * N), *yb = ragon::test::alloc<float>(size_t(M) * N);
  float *za = ragon::test::alloc<float>(size_t(M) * N), *zb = ragon::test::alloc<float>(size_t(M) * N);
  K::narrow_bf16(dx, M, reinterpret_cast<const gpu::bf16*>(dwa), reinterpret_cast<const gpu::bf16*>(dwb), ya, yb, N, Kd,
                 nullptr);
  K::narrow_bf16_bf16in(reinterpret_cast<const gpu::bf16*>(dxb), M, reinterpret_cast<const gpu::bf16*>(dwa),
                        reinterpret_cast<const gpu::bf16*>(dwb), za, zb, N, Kd, nullptr);
  const char* err = ragon::test::sync_error();
  const auto ha = download(ya, size_t(M) * N), hb = download(yb, size_t(M) * N);
  const auto ga = download(za, size_t(M) * N), gb = download(zb, size_t(M) * N);
  double worst = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < 2 * N; ++n) {
      const auto& wm = n < N ? wa : wb;
      double s = 0, sa = 0;
      for (int k = 0; k < Kd; ++k) {
        const double t = double(bf16r(x[size_t(m) * Kd + k])) * from_bits(uint32_t(wm[size_t(n % N) * Kd + k]) << 16);
        s += t;
        sa += std::fabs(t);
      }
      const double y = n < N ? ha[size_t(m) * N + n] : hb[size_t(m) * N + n - N];
      worst = std::max(worst, std::fabs(y - s) / (1e-5 * sa + 1e-30));
    }
  const bool same = std::memcmp(ha.data(), ga.data(), ha.size() * 4) == 0 && std::memcmp(hb.data(), gb.data(), hb.size() * 4) == 0;
  const bool good = !err && worst <= 1.0 && same;
  std::printf("narrow_bf16 M=%d K=%d: %.2f of tolerance, bf16 input the same bytes %s%s%s %s\n", M, Kd, worst,
              same ? "yes" : "NO", err ? ", error " : "", err ? err : "", good ? "ok" : "FAIL");
  g_hash = fnv(ha.data(), ha.size() * 4, g_hash);
  g_hash = fnv(hb.data(), hb.size() * 4, g_hash);
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dxb), static_cast<void*>(dwa), static_cast<void*>(dwb),
                  static_cast<void*>(ya), static_cast<void*>(yb), static_cast<void*>(za), static_cast<void*>(zb)})
    ragon::test::free_bytes(p);
  return good;
}

}  // namespace

int main() {
  bool ok = test_quant_random();
  ok &= test_quant_sweep();
  ok &= test_fused();
  ok &= test_narrow();
#ifdef RAGON_BACKEND_HIP
  ok &= test_gemm(528, 640, 11, {130, 1, 15, 16, 17, 64, 65});  // partial wave blocks (33 tiles), the K % 256 tail
  ok &= test_gemm(256, 5120, 13, {70, 3});
  ok &= test_gemm_slabs();
  ok &= test_swiglu(67, 256, 512);
  ok &= test_swiglu(1, 384, 1024);
#endif
  std::printf("QHASH %016llx\n", static_cast<unsigned long long>(g_qhash));
  std::printf("HASH %016llx\n", static_cast<unsigned long long>(g_hash));
  std::printf(ok ? "all prefill HIP tests passed\n" : "PREFILL HIP TESTS FAILED\n");
  return ok ? 0 : 1;
}
