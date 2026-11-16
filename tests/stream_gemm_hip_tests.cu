// The HIP stream GEMM (src/core/stream_gemm_hip.cu, BACKEND §7) against FP64 host references: to_half_rows and
// rmsnorm_half bit for bit, the retile against a host-built blob (and twice the same bytes), the tiled dequant against
// the row-major one, stream_gemm_nvfp4 / _fp8 within stream_tests' tolerance at 1 to 64 rows with every row's bytes
// those of the 64-row launch, and stream_gemm_multi's targets bit for bit the single launches. The outputs are hashed
// so two runs compare byte for byte. Inputs come from raw mt19937 words only (the standard fixes them).
//
// `--bench [name,...]`: the served shapes (grouped as the engine launches them) at 1 and 16 rows, event-timed over
// enough weight copies that one pass walks >= 1.1 GB (the 32 MB MALL cannot hold it), 5 interleaved reps, the median
// GB/s of codes + scales (the gate's method, scratch p1-hip-gemm-gate/vgs.ps1).
#include "gpu_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}
uint64_t g_hash_gemm = 1469598103934665603ull, g_hash_retile = 1469598103934665603ull;

float unit(std::mt19937& rng) { return static_cast<float>(rng() >> 8) * (1.f / 16777216.f); }
// A normal-ish value from raw words (sum of four uniforms, centred), with an occasional outlier.
float value(std::mt19937& rng) {
  const float v = (unit(rng) + unit(rng) + unit(rng) + unit(rng) - 2.f) * 1.7f;
  return (rng() % 97 == 0) ? v * 30.f : v;
}
// float -> IEEE half bits, round to nearest even, in software: the Windows HIP link has no
// compiler-rt, so a host _Float16 conversion (__truncsfhf2) does not link. No NaN in these tests.
uint16_t half_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t a = u & 0x7FFFFFFFu;
  if (a >= 0x47800000u) return static_cast<uint16_t>(sign | 0x7C00u);  // >= 65536: inf
  if (a < 0x38800000u) {                                                 // half subnormal or zero
    if (a < 0x33000000u) return static_cast<uint16_t>(sign);             // < 2^-25: rounds to 0
    const uint32_t mant = (a & 0x7FFFFFu) | 0x800000u;
    const uint32_t drop = 13u + (113u - (a >> 23));                      // 14..24 bits
    const uint32_t q = mant >> drop, r = mant & ((1u << drop) - 1u), h = 1u << (drop - 1u);
    const uint32_t v = q + ((r > h || (r == h && (q & 1u))) ? 1u : 0u);
    return static_cast<uint16_t>(sign | v);
  }
  const uint32_t q = (a - 0x38000000u) >> 13, r = a & 0x1FFFu;
  const uint32_t v = q + ((r > 0x1000u || (r == 0x1000u && (q & 1u))) ? 1u : 0u);  // carry may reach inf: correct
  return static_cast<uint16_t>(sign | v);
}
double half_val(uint16_t b) {  // IEEE half bits -> value (finite inputs)
  const int e = (b >> 10) & 0x1f, m = b & 0x3ff;
  const double v = e ? std::ldexp(1024.0 + m, e - 25) : std::ldexp(double(m), -24);
  return (b & 0x8000) ? -v : v;
}
uint16_t bf16_bits(float f) {  // round to nearest even (no NaN in these tests)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float bf16_val(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
bool same_bytes(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

void to_half(const float* dx, int M, int Kd, uint16_t* xh, float* xinv) {
  K::to_half_rows(dx, M, Kd, reinterpret_cast<gpu::half*>(xh), xinv, nullptr);
}

bool test_to_half_rows() {
  std::mt19937 rng(5);
  const int Ms[] = {1, 2, 8, 12, 16, 24, 32}, Ks[] = {128, 1000, 4096, 5120, 6144, 17408, 25600};
  const int MMAX = 32, KMAX = 25600;
  std::vector<float> x(size_t(MMAX) * KMAX + 1);
  float* dx = ragon::test::alloc<float>(x.size());
  uint16_t* dh = ragon::test::alloc<uint16_t>(size_t(MMAX) * KMAX);
  float* dinv = ragon::test::alloc<float>(MMAX);
  bool ok = true;
  int cases = 0;
  for (int Kd : Ks)
    for (int M : Ms)
      for (int off = 0; off < 2; ++off) {
        for (auto& v : x) v = value(rng);
        if (M > 2) {
          for (int k = 0; k < Kd; ++k) x[off + size_t(1) * Kd + k] = 0.f;  // a zero row
          x[off + size_t(2) * Kd + Kd / 3] = 3.0e30f;                       // one huge value: the rest underflows
        }
        if (M > 3) x[off + size_t(3) * Kd + Kd - 1] = INFINITY;
        ragon::test::h2d(dx, x.data(), x.size() * sizeof(float));
        to_half(dx + off, M, Kd, dh, dinv);
        const auto h = download(dh, size_t(M) * Kd);
        const auto inv = download(dinv, M);
        for (int m = 0; m < M && ok; ++m) {
          const float* row = x.data() + off + size_t(m) * Kd;
          float mx = 0.f;
          for (int k = 0; k < Kd; ++k) mx = std::fmax(mx, std::fabs(row[k]));
          int e = 0;
          if (mx > 0.f && std::isfinite(mx)) std::frexp(mx, &e);
          const float s = std::ldexp(1.f, 15 - e);
          if (inv[m] != std::ldexp(1.f, e - 15)) {
            std::printf("to_half_rows M=%d K=%d off=%d row %d: scale differs FAIL\n", M, Kd, off, m);
            ok = false;
          }
          for (int k = 0; k < Kd && ok; ++k)
            if (half_bits(row[k] * s) != h[size_t(m) * Kd + k]) {
              std::printf("to_half_rows M=%d K=%d off=%d row %d col %d differs FAIL\n", M, Kd, off, m, k);
              ok = false;
            }
        }
        ++cases;
      }
  ragon::test::free_bytes(dx), ragon::test::free_bytes(dh), ragon::test::free_bytes(dinv);
  std::printf("to_half_rows: %d shapes bit for bit against the host %s\n", cases, ok ? "ok" : "FAIL");
  return ok;
}

// rmsnorm_half: out within FP32 rounding of the FP64 norm; xh and xinv bit for bit to_half_rows of that out.
bool test_rmsnorm_half() {
  std::mt19937 rng(7);
  bool ok = true;
  for (int H : {4096, 5120})
    for (bool gemma : {false, true}) {
      const int R = 5;
      std::vector<float> x(size_t(R) * H);
      std::vector<uint16_t> w(H);
      for (auto& v : x) v = value(rng);
      for (auto& b : w) b = bf16_bits(value(rng) * 0.3f);
      float *dx = upload(x), *dout = ragon::test::alloc<float>(x.size()), *dinv = ragon::test::alloc<float>(R),
            *dinv2 = ragon::test::alloc<float>(R);
      uint16_t *dw = upload(w), *dh = ragon::test::alloc<uint16_t>(x.size()), *dh2 = ragon::test::alloc<uint16_t>(x.size());
      K::rmsnorm_half(dx, reinterpret_cast<const gpu::bf16*>(dw), dout, reinterpret_cast<gpu::half*>(dh), dinv, R, H,
                      1e-6f, gemma, nullptr);
      to_half(dout, R, H, dh2, dinv2);
      const auto out = download(dout, x.size());
      double worst = 0;
      for (int r = 0; r < R; ++r) {
        double ss = 0;
        for (int h = 0; h < H; ++h) ss += double(x[size_t(r) * H + h]) * x[size_t(r) * H + h];
        const double rstd = 1.0 / std::sqrt(ss / H + 1e-6);
        double mx = 0;
        std::vector<double> want(H);
        for (int h = 0; h < H; ++h) {
          const double wv = bf16_val(w[h]);
          want[h] = x[size_t(r) * H + h] * rstd * (gemma ? 1.0 + wv : wv);
          mx = std::max(mx, std::abs(want[h]));
        }
        for (int h = 0; h < H; ++h) worst = std::max(worst, std::abs(out[size_t(r) * H + h] - want[h]) / mx);
      }
      const bool bits = download(dh, x.size()) == download(dh2, x.size()) && download(dinv, R) == download(dinv2, R);
      const bool good = worst < 1e-5 && bits;
      ok &= good;
      std::printf("rmsnorm_half H=%d gemma=%d: err/max %.2e, FP16 copy = to_half_rows(out) %s %s\n", H, int(gemma), worst,
                  bits ? "yes" : "NO", good ? "ok" : "FAIL");
      for (void* p : {static_cast<void*>(dx), static_cast<void*>(dout), static_cast<void*>(dinv),
                      static_cast<void*>(dinv2), static_cast<void*>(dw), static_cast<void*>(dh), static_cast<void*>(dh2)})
        ragon::test::free_bytes(p);
    }
  return ok;
}

// The HIP blob, built on the host from the layout in stream_gemm_hip.cu's header.
std::vector<uint8_t> host_tiled_nvfp4(const std::vector<uint8_t>& w, const std::vector<uint8_t>& s, int N, int Kd) {
  const size_t nt = N / 16, tiles = nt * (Kd / 64);
  std::vector<uint8_t> o(tiles * 576);
  for (size_t t = 0; t < tiles; ++t)
    for (int r = 0; r < 16; ++r) {
      const size_t n = (t % nt) * 16 + r, kg = t / nt;
      for (int j = 0; j < 32; ++j) o[t * 512 + r * 32 + j] = w[n * (Kd / 2) + kg * 32 + j];
      for (int q = 0; q < 4; ++q) o[tiles * 512 + t * 64 + r * 4 + q] = s[n * (Kd / 16) + kg * 4 + q];
    }
  return o;
}
std::vector<uint8_t> host_tiled_fp8(const std::vector<uint8_t>& w, int N, int Kd) {
  const size_t nt = N / 16, tiles = nt * (Kd / 64);
  std::vector<uint8_t> o(tiles * 1024);
  for (size_t t = 0; t < tiles; ++t)
    for (int r = 0; r < 16; ++r) {
      const size_t n = (t % nt) * 16 + r, kg = t / nt;
      for (int j = 0; j < 64; ++j) o[t * 1024 + r * 64 + j] = w[n * Kd + kg * 64 + j];
    }
  return o;
}

struct Mat {
  int N, Kd;
  std::vector<uint8_t> w4, s4, w8;
  uint8_t *dw4 = nullptr, *dw8 = nullptr;
};
Mat make_mat(std::mt19937& rng, int N, int Kd) {
  Mat m{N, Kd, std::vector<uint8_t>(size_t(N) * Kd / 2), std::vector<uint8_t>(size_t(N) * Kd / 16),
        std::vector<uint8_t>(size_t(N) * Kd)};
  for (auto& b : m.w4) b = static_cast<uint8_t>(rng());
  for (auto& b : m.s4) b = static_cast<uint8_t>(0x28 + rng() % 0x21);  // E4M3 2^-5 .. 2^-1 range, no NaN
  for (auto& b : m.w8) {
    do b = static_cast<uint8_t>(rng());
    while ((b & 0x7f) == 0x7f);
  }
  return m;
}

// Retile (twice: the same bytes), against the host blob; the tiled dequant against the row-major one.
bool retile(Mat& m, const char* tag) {
  const int N = m.N, Kd = m.Kd;
  uint8_t *rw4 = upload(m.w4), *rs4 = upload(m.s4), *rw8 = upload(m.w8);
  const size_t b4 = K::tiled_bytes_nvfp4(N, Kd), b8 = K::tiled_bytes_fp8(N, Kd);
  m.dw4 = ragon::test::alloc<uint8_t>(b4);
  m.dw8 = ragon::test::alloc<uint8_t>(b8);
  uint8_t *a4 = ragon::test::alloc<uint8_t>(b4), *a8 = ragon::test::alloc<uint8_t>(b8);
  ragon::test::zero(m.dw4, b4), ragon::test::zero(a4, b4), ragon::test::zero(m.dw8, b8), ragon::test::zero(a8, b8);
  K::retile_nvfp4(rw4, rs4, m.dw4, N, Kd, nullptr);
  K::retile_fp8(rw8, m.dw8, N, Kd, nullptr);
  K::retile_nvfp4(rw4, rs4, a4, N, Kd, nullptr);
  K::retile_fp8(rw8, a8, N, Kd, nullptr);
  const auto t4 = download(m.dw4, b4), t8 = download(m.dw8, b8);
  const bool twice = t4 == download(a4, b4) && t8 == download(a8, b8);
  const bool host = t4 == host_tiled_nvfp4(m.w4, m.s4, N, Kd) && t8 == host_tiled_fp8(m.w8, N, Kd);
  g_hash_retile = fnv(t4.data(), t4.size(), g_hash_retile);
  g_hash_retile = fnv(t8.data(), t8.size(), g_hash_retile);
  // Tiled dequant = row-major dequant (kernels.cu), bit for bit.
  uint16_t *da = ragon::test::alloc<uint16_t>(size_t(N) * Kd), *db = ragon::test::alloc<uint16_t>(size_t(N) * Kd);
  auto* ba = reinterpret_cast<gpu::bf16*>(da);
  auto* bb = reinterpret_cast<gpu::bf16*>(db);
  K::dequant_nvfp4(rw4, rs4, 0.37f, ba, N, Kd, nullptr);
  K::dequant_tiled_nvfp4(m.dw4, 0.37f, bb, N, Kd, nullptr);
  bool deq = download(da, size_t(N) * Kd) == download(db, size_t(N) * Kd);
  K::dequant_fp8(rw8, 0.021f, ba, N, Kd, nullptr);
  K::dequant_tiled_fp8(m.dw8, 0.021f, bb, N, Kd, nullptr);
  deq &= download(da, size_t(N) * Kd) == download(db, size_t(N) * Kd);
  const char* err = ragon::test::sync_error();
  const bool good = twice && host && deq && !err;
  std::printf("%s retile N=%d K=%d: host blob %s, two retiles identical %s, tiled dequant = row-major %s%s%s %s\n", tag, N,
              Kd, host ? "yes" : "NO", twice ? "yes" : "NO", deq ? "yes" : "NO", err ? ", error " : "", err ? err : "",
              good ? "ok" : "FAIL");
  for (void* p : {static_cast<void*>(rw4), static_cast<void*>(rs4), static_cast<void*>(rw8), static_cast<void*>(a4),
                  static_cast<void*>(a8), static_cast<void*>(da), static_cast<void*>(db)})
    ragon::test::free_bytes(p);
  return good;
}

template <class T>
std::vector<float> host_ref(const Mat& w, bool fp4, const std::vector<T>& x, int M, float gs) {
  const int N = w.N, Kd = w.Kd;
  std::vector<double> wd(size_t(N) * Kd);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < Kd; ++k) {
      if (fp4) {
        const uint8_t b = w.w4[(size_t(n) * Kd + k) / 2];
        wd[size_t(n) * Kd + k] = double(kFp4[(k & 1) ? b >> 4 : b & 0xf]) * e4m3(w.s4[size_t(n) * (Kd / 16) + k / 16]);
      } else {
        wd[size_t(n) * Kd + k] = e4m3(w.w8[size_t(n) * Kd + k]);
      }
    }
  std::vector<float> ref(size_t(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double a = 0;
      const double* wr = wd.data() + size_t(n) * Kd;
      const T* xr = x.data() + size_t(m) * Kd;
      for (int k = 0; k < Kd; ++k) a += wr[k] * double(xr[k]);
      ref[size_t(m) * N + n] = float(a * gs);
    }
  return ref;
}

void run(bool fp4, const float* dx, int M, int Kd, uint16_t* xh, float* xinv, const Mat& w, float gs, float* dy) {
  to_half(dx, M, Kd, xh, xinv);
  if (fp4) K::stream_gemm_nvfp4(reinterpret_cast<gpu::half*>(xh), xinv, M, w.dw4, nullptr, gs, dy, w.N, Kd, nullptr);
  else K::stream_gemm_fp8(reinterpret_cast<gpu::half*>(xh), xinv, M, w.dw8, gs, dy, w.N, Kd, nullptr);
}

// One shape: tolerance against the host at every M, every row's bytes those of the 64-row launch, and a repeat of the
// 64-row launch the same bytes.
bool test_gemm(int N, int Kd, uint32_t seed, bool split, const std::vector<int>& Ms) {
  std::mt19937 rng(seed);
  Mat w = make_mat(rng, N, Kd);
  const int MMAX = K::kMaxStreamRows;
  std::vector<float> x(size_t(MMAX) * Kd);
  for (auto& v : x) v = value(rng);
  bool ok = retile(w, "stream");
  if (split) K::stream_gemm_split_k(w.dw4, N, Kd), K::stream_gemm_split_k(w.dw8, N, Kd);
  float* dx = upload(x);
  uint16_t* xh = ragon::test::alloc<uint16_t>(size_t(MMAX) * Kd);
  float *xinv = ragon::test::alloc<float>(MMAX), *dy = ragon::test::alloc<float>(size_t(MMAX) * N);
  for (bool fp4 : {true, false}) {
    const float gs = fp4 ? 0.37f : 0.021f;
    const std::vector<float> ref = host_ref(w, fp4, x, MMAX, gs);
    run(fp4, dx, MMAX, Kd, xh, xinv, w, gs, dy);
    const std::vector<float> full = download(dy, size_t(MMAX) * N);
    // The activation as the kernel reads it (FP16 row times its xinv): against this the only error left is the
    // kernel's own FP32 arithmetic. The bound against the exact activation (2e-3, as stream_tests) also carries
    // the FP16 rounding of x, which on short K (K=640) alone reaches ~2e-3, so it applies from K=2048 on.
    std::vector<double> xq(size_t(MMAX) * Kd);
    {
      const std::vector<uint16_t> h = download(xh, xq.size());
      const std::vector<float> inv = download(xinv, MMAX);
      for (size_t i = 0; i < xq.size(); ++i) xq[i] = half_val(h[i]) * double(inv[i / Kd]);
    }
    const std::vector<float> refq = host_ref(w, fp4, xq, MMAX, gs);
    ragon::test::zero(dy, size_t(MMAX) * N * 4);
    run(fp4, dx, MMAX, Kd, xh, xinv, w, gs, dy);
    const bool again = download(dy, size_t(MMAX) * N) == full;
    g_hash_gemm = fnv(full.data(), full.size() * 4, g_hash_gemm);
    for (int M : Ms) {
      ragon::test::zero(dy, size_t(MMAX) * N * 4);
      run(fp4, dx, M, Kd, xh, xinv, w, gs, dy);
      const std::vector<float> got = download(dy, size_t(MMAX) * N);
      double worst = 0, worstq = 0;
      for (int m = 0; m < M; ++m) {
        double rms = 0;
        for (int n = 0; n < N; ++n) rms += double(ref[size_t(m) * N + n]) * ref[size_t(m) * N + n];
        rms = std::sqrt(rms / N);
        for (int n = 0; n < N; ++n) {
          const size_t i = size_t(m) * N + n;
          worst = std::max(worst, std::abs(got[i] - ref[i]) / rms);
          worstq = std::max(worstq, std::abs(got[i] - refq[i]) / rms);
        }
      }
      bool same = same_bytes(got.data(), full.data(), size_t(M) * N * 4);
      for (size_t i = size_t(M) * N; i < got.size(); ++i) same &= got[i] == 0.f;  // nothing past row M written
      const char* err = ragon::test::sync_error();
      // The gfx11 WMMA FP32 accumulate leaves an error growing about linearly with K (1.5e-5 at K=640, 8e-5 at
      // K=5120 measured); a wrong index or scale shows as 1e-1.
      const bool good = worstq < std::max(5e-5, 3e-8 * Kd) && (Kd < 2048 || worst < 2e-3) && same && again && !err;
      ok &= good;
      std::printf("stream_gemm %-5s N=%d K=%d%s M=%-2d err/rms %.2e (FP16 x %.1e), rows identical to M=64 %s, repeat identical %s%s%s %s\n",
                  fp4 ? "nvfp4" : "fp8", N, Kd, split ? " marked" : "", M, worst, worstq, same ? "yes" : "NO", again ? "yes" : "NO", err ? ", error " : "",
                  err ? err : "", good ? "ok" : "FAIL");
    }
  }
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(xh), static_cast<void*>(xinv), static_cast<void*>(dy),
                  static_cast<void*>(w.dw4), static_cast<void*>(w.dw8)})
    ragon::test::free_bytes(p);
  return ok;
}

// stream_gemm_multi: each target's bytes those of its single launch (grouping changes nothing), 2 to 4 targets.
bool test_multi() {
  std::mt19937 rng(23);
  const int Kd = 2048, Ns[] = {528, 4096, 1024, 16};  // 528 and 16 marked (K split in 8), 1024 not, 4096 too wide
  std::vector<Mat> ws;
  for (int N : Ns) ws.push_back(make_mat(rng, N, Kd));
  bool ok = true;
  for (auto& w : ws) ok &= retile(w, "multi");
  for (int i : {0, 3}) K::stream_gemm_split_k(ws[i].dw4, Ns[i], Kd), K::stream_gemm_split_k(ws[i].dw8, Ns[i], Kd);
  const int MMAX = K::kMaxStreamRows;
  std::vector<float> x(size_t(MMAX) * Kd);
  for (auto& v : x) v = value(rng);
  float* dx = upload(x);
  uint16_t* xh = ragon::test::alloc<uint16_t>(size_t(MMAX) * Kd);
  float* xinv = ragon::test::alloc<float>(MMAX);
  std::vector<float*> ya, yg;
  for (int N : Ns) ya.push_back(ragon::test::alloc<float>(size_t(MMAX) * N)), yg.push_back(ragon::test::alloc<float>(size_t(MMAX) * N));
  for (bool fp4 : {true, false})
    for (int n : {2, 3, 4})
      for (int M : {1, 5, 16, 17, 33, 64}) {
        K::StreamTarget t[4];
        for (int i = 0; i < n; ++i) {
          const float gs = 0.1f + 0.07f * i;
          run(fp4, dx, M, Kd, xh, xinv, ws[i], gs, ya[i]);
          t[i] = {fp4 ? ws[i].dw4 : ws[i].dw8, nullptr, gs, yg[i], Ns[i]};
        }
        K::stream_gemm_multi(fp4, reinterpret_cast<gpu::half*>(xh), xinv, M, t, n, Kd, nullptr);
        bool same = true;
        for (int i = 0; i < n; ++i) {
          const auto a = download(ya[i], size_t(M) * Ns[i]), g = download(yg[i], size_t(M) * Ns[i]);
          same &= a == g;
          g_hash_gemm = fnv(g.data(), g.size() * 4, g_hash_gemm);
        }
        const char* err = ragon::test::sync_error();
        ok &= same && !err;
        if (!same || err || M == 64)
          std::printf("stream_gemm_multi %-5s %d targets M=%-2d (and smaller M): grouped = alone %s%s%s\n",
                      fp4 ? "nvfp4" : "fp8", n, M, same ? "yes ok" : "NO FAIL", err ? ", error " : "", err ? err : "");
      }
  ragon::test::free_bytes(dx), ragon::test::free_bytes(xh), ragon::test::free_bytes(xinv);
  for (size_t i = 0; i < ws.size(); ++i) {
    ragon::test::free_bytes(ya[i]), ragon::test::free_bytes(yg[i]);
    ragon::test::free_bytes(ws[i].dw4), ragon::test::free_bytes(ws[i].dw8);
  }
  return ok;
}

// One split blob on two streams at once (a side prefill chunk beside the main step): each stream has its own split
// counters, so every launch's bytes are those of the blob alone on the null stream.
bool test_two_streams() {
  std::mt19937 rng(31);
  const int N = 1280, Kd = 5120, M = 16, reps = 40;  // the drafter conv projection: K split in 4
  Mat w = make_mat(rng, N, Kd);
  bool ok = retile(w, "streams");
  K::stream_gemm_split_k(w.dw4, N, Kd), K::stream_gemm_split_k(w.dw8, N, Kd);
  float* dx[2];
  uint16_t* xh[2];
  float *xinv[2], *dy[2];
  std::vector<float> want[2];
  for (int j = 0; j < 2; ++j) {
    std::vector<float> x(size_t(M) * Kd);
    for (auto& v : x) v = value(rng);
    dx[j] = upload(x);
    xh[j] = ragon::test::alloc<uint16_t>(size_t(M) * Kd);
    xinv[j] = ragon::test::alloc<float>(M);
    dy[j] = ragon::test::alloc<float>(size_t(M) * N);
  }
  gpu::Stream st[2];
  for (auto& s : st) ok &= gpu::stream_create(&s, gpu::kStreamNonBlocking) == gpu::kSuccess;
  for (bool fp4 : {true, false}) {
    const float gs = fp4 ? 0.37f : 0.021f;
    for (int j = 0; j < 2; ++j) {
      run(fp4, dx[j], M, Kd, xh[j], xinv[j], w, gs, dy[j]);
      want[j] = download(dy[j], size_t(M) * N);
    }
    (void)gpu::device_sync();  // an error shows in sync_error below
    bool same = true;
    for (int r = 0; r < reps && same; ++r) {
      for (int j = 0; j < 2; ++j) {
        const gpu::half* x = reinterpret_cast<gpu::half*>(xh[j]);
        if (fp4) K::stream_gemm_nvfp4(x, xinv[j], M, w.dw4, nullptr, gs, dy[j], N, Kd, st[j]);
        else K::stream_gemm_fp8(x, xinv[j], M, w.dw8, gs, dy[j], N, Kd, st[j]);
      }
      (void)gpu::device_sync();
      for (int j = 0; j < 2; ++j) same &= download(dy[j], size_t(M) * N) == want[j];
    }
    const char* err = ragon::test::sync_error();
    ok &= same && !err;
    std::printf("stream_gemm %-5s N=%d K=%d on two streams at once, %d launches each: bytes those of one stream %s%s%s\n",
                fp4 ? "nvfp4" : "fp8", N, Kd, reps, same ? "yes ok" : "NO FAIL", err ? ", error " : "", err ? err : "");
  }
  for (auto s : st) (void)gpu::stream_destroy(s);
  for (int j = 0; j < 2; ++j)
    ragon::test::free_bytes(dx[j]), ragon::test::free_bytes(xh[j]), ragon::test::free_bytes(xinv[j]),
        ragon::test::free_bytes(dy[j]);
  ragon::test::free_bytes(w.dw4), ragon::test::free_bytes(w.dw8);
  return ok;
}

// ---- bench ----
__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed, int mode) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ static_cast<uint32_t>(i >> 32) * 40503u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
    uint8_t b = static_cast<uint8_t>(h);
    if (mode == 1) b = static_cast<uint8_t>(0x28 + h % 0x21);   // NVFP4 block scales
    if (mode == 2 && (b & 0x7f) == 0x7f) b = 0x38;              // FP8 values, no NaN
    p[i] = b;
  }
}
void fill(uint8_t* p, size_t n, uint32_t seed, int mode) {
  fill_kernel<<<4096, 256>>>(p, n, seed, mode);
}

struct Group {
  const char* name;
  bool fp4;
  int K;
  std::vector<int> Ns;
  bool split = false;  // marked as the loader marks it (stream_gemm_split_k)
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void bench(const std::string& only) {
  const std::vector<Group> groups = {
      {"gu", true, 5120, {17408, 17408}},     {"down", true, 17408, {5120}},
      {"head", true, 5120, {248320}},         {"dn_in", false, 5120, {10240, 6144}},
      {"out", false, 6144, {5120}},           {"a_qkv", false, 5120, {12288, 1024, 1024}},
      {"d_qkv", true, 5120, {4096, 1024, 1024}}, {"d_o", true, 4096, {5120}},
      {"d_fc", true, 25600, {5120}},          {"d_kproj", true, 5120, {1280}, true},
      {"d_head48k", true, 5120, {49152}},     {"d_head80k", true, 5120, {81920}},
  };
  const long long kMinBytes = 1100000000LL;
  const int kReps = 5;
  hipEvent_t e0, e1;
  (void)hipEventCreate(&e0), (void)hipEventCreate(&e1);
  std::printf("ROW,name,fmt,K,N,bytes,copies,launches,M,us,ev_gbs,ev_min,ev_max\n");
  for (const Group& g : groups) {
    if (!only.empty() && ("," + only + ",").find("," + std::string(g.name) + ",") == std::string::npos) continue;
    const int Kd = g.K, nt = static_cast<int>(g.Ns.size());
    long long bytes = 0;
    std::vector<size_t> tb;
    for (int N : g.Ns) tb.push_back(g.fp4 ? K::tiled_bytes_nvfp4(N, Kd) : K::tiled_bytes_fp8(N, Kd)), bytes += tb.back();
    const int L = static_cast<int>(std::max<long long>(2, (kMinBytes + bytes - 1) / bytes));
    // One allocation per (copy, target); NVFP4 codes random, scales in range, FP8 without NaN.
    std::vector<uint8_t*> bufs;
    for (int c = 0; c < L; ++c)
      for (int i = 0; i < nt; ++i) {
        uint8_t* p = ragon::test::alloc<uint8_t>(tb[i]);
        const int N = g.Ns[i];
        if (g.fp4) {
          fill(p, size_t(N) * Kd / 2, 17u * c + i, 0);
          fill(p + size_t(N) * Kd / 2, size_t(N) * Kd / 16, 31u * c + i, 1);
        } else {
          fill(p, tb[i], 17u * c + i, 2);
        }
        if (g.split) K::stream_gemm_split_k(p, N, Kd);
        bufs.push_back(p);
      }
    float* dx = ragon::test::alloc<float>(size_t(16) * Kd);
    fill(reinterpret_cast<uint8_t*>(dx), size_t(16) * Kd * 4, 99u, 0);
    std::vector<float> xs(size_t(16) * Kd);
    std::mt19937 rng(3);
    for (auto& v : xs) v = value(rng);
    ragon::test::h2d(dx, xs.data(), xs.size() * 4);
    uint16_t* xh = ragon::test::alloc<uint16_t>(size_t(16) * Kd);
    float* xinv = ragon::test::alloc<float>(16);
    to_half(dx, 16, Kd, xh, xinv);
    std::vector<float*> ys;
    for (int N : g.Ns) ys.push_back(ragon::test::alloc<float>(size_t(16) * N));
    const auto* xhh = reinterpret_cast<const gpu::half*>(xh);
    auto launch = [&](int c, int M) {
      K::StreamTarget t[4];
      for (int i = 0; i < nt; ++i) t[i] = {bufs[size_t(c) * nt + i], nullptr, 0.5f, ys[i], g.Ns[i]};
      K::stream_gemm_multi(g.fp4, xhh, xinv, M, t, nt, Kd, nullptr);
    };
    const int per = static_cast<int>(
        std::max(1.0, std::min(std::max(1.0, 400.0 / L), std::ceil(8e9 / (double(bytes) * L)))));
    const int n = L * per;
    for (int M : {1, 16})
      for (int w = 0; w < 2; ++w)
        for (int c = 0; c < L; ++c) launch(c, M);
    (void)hipDeviceSynchronize();
    std::vector<double> ev[2];
    for (int rep = 0; rep < kReps; ++rep)
      for (int mi = 0; mi < 2; ++mi) {
        const int M = mi ? 16 : 1;
        (void)hipEventRecord(e0, nullptr);
        for (int i = 0; i < per; ++i)
          for (int c = 0; c < L; ++c) launch(c, M);
        (void)hipEventRecord(e1, nullptr);
        (void)hipEventSynchronize(e1);
        float ms = 0.f;
        (void)hipEventElapsedTime(&ms, e0, e1);
        ev[mi].push_back(double(bytes) * n / (ms / 1000.0) / 1e9);
      }
    const char* err = ragon::test::sync_error();
    std::string ns;
    for (int i = 0; i < nt; ++i) ns += (i ? "|" : "") + std::to_string(g.Ns[i]);
    for (int mi = 0; mi < 2; ++mi) {
      const double med = median(ev[mi]);
      std::printf("ROW,%s,%s,%d,%s,%lld,%d,%d,%d,%.1f,%.1f,%.1f,%.1f%s%s\n", g.name, g.fp4 ? "nvfp4" : "fp8", Kd,
                  ns.c_str(), bytes, L, n, mi ? 16 : 1, double(bytes) / (med * 1e3), med,
                  *std::min_element(ev[mi].begin(), ev[mi].end()), *std::max_element(ev[mi].begin(), ev[mi].end()),
                  err ? ",error " : "", err ? err : "");
    }
    std::fflush(stdout);
    for (uint8_t* p : bufs) ragon::test::free_bytes(p);
    for (float* y : ys) ragon::test::free_bytes(y);
    ragon::test::free_bytes(dx), ragon::test::free_bytes(xh), ragon::test::free_bytes(xinv);
  }
  (void)hipEventDestroy(e0), (void)hipEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--bench") {
    bench(argc > 2 ? argv[2] : "");
    return 0;
  }
  std::vector<int> all;
  for (int M = 1; M <= 16; ++M) all.push_back(M);
  for (int M : {17, 20, 24, 31, 32, 33, 48, 63, 64}) all.push_back(M);
  bool ok = test_to_half_rows();
  ok &= test_rmsnorm_half();
  // "marked": stream_gemm_split_k, as the loader marks the drafter's conv projections.
  ok &= test_gemm(528, 2048, 11, true, all);                // partial blocks (33 tiles), K split in 8
  ok &= test_gemm(528, 640, 13, true, {1, 7, 16, 17, 64});  // the K % 256 tail: no split even marked
  ok &= test_gemm(1040, 5120, 17, true, {1, 2, 9, 16, 32, 64});  // K split in 5
  ok &= test_gemm(1040, 5120, 17, false, {1, 16, 64});      // the same unmarked: no split
  ok &= test_gemm(4096, 1024, 19, true, {1, 16, 17, 64});   // 256 tiles: no split even marked
  ok &= test_gemm(1280, 5120, 29, true, {1, 16, 64});       // the drafter conv projection: K split in 4
  ok &= test_multi();
  ok &= test_two_streams();
  std::printf("HASH gemm %016llx retile %016llx\n", static_cast<unsigned long long>(g_hash_gemm),
              static_cast<unsigned long long>(g_hash_retile));
  std::printf(ok ? "all HIP stream GEMM tests passed\n" : "HIP STREAM GEMM TESTS FAILED\n");
  return ok ? 0 : 1;
}
