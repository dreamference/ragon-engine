// W0's kernels against FP32/FP64 host references, for either backend (BACKEND §7): norms, conv, the DeltaNet
// recurrence (chain, sequences, tree), gating, attention's prepare, the KV page moves, the drafter's non-GEMM kernels
// and the sparse head's selection. Each output is hashed and, with a path argument, its bytes are written there, so two
// runs (or two backends) compare byte for byte. Inputs come from raw mt19937 words only (the standard fixes them; the
// distributions are not), so every platform builds the same inputs.
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
using ragon::kernels::SeqBatch;
using ragon::kernels::SeqPtrs;
using ragon::kernels::TreeDesc;
using ragon::test::download;
using ragon::test::e4m3;
using bits16 = uint16_t;

// ---- inputs ----------------------------------------------------------------------------------------------------

struct Rng {
  std::mt19937 g;
  explicit Rng(uint32_t seed) : g(seed) {}
  float uni() { return static_cast<float>(g() >> 8) * (1.f / 8388608.f) - 1.f; }  // [-1, 1), exact
  int below(int n) { return static_cast<int>(g() % static_cast<uint32_t>(n)); }
  std::vector<float> vec(size_t n, float scale = 1.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = uni() * scale;
    return v;
  }
};

bits16 bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<bits16>(u >> 16);
}
float bf16_float(bits16 b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
std::vector<bits16> to_bits(const std::vector<float>& v) {
  std::vector<bits16> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = bf16_bits(v[i]);
  return b;
}
std::vector<float> from_bits(const std::vector<bits16>& b) {
  std::vector<float> v(b.size());
  for (size_t i = 0; i < b.size(); ++i) v[i] = bf16_float(b[i]);
  return v;
}

// ---- device memory: every buffer freed at exit ------------------------------------------------------------------

std::vector<void*> g_live;
template <typename T>
T* dev(const std::vector<T>& h) {
  T* d = ragon::test::upload(h);
  g_live.push_back(d);
  return d;
}
template <typename T>
T* dev_zero(size_t n) {
  T* d = ragon::test::alloc<T>(n);
  ragon::test::zero(d, n * sizeof(T));
  g_live.push_back(d);
  return d;
}
gpu::bf16* as_bf16(bits16* p) { return reinterpret_cast<gpu::bf16*>(p); }
gpu::bf16* dev_bf16(const std::vector<bits16>& h) { return as_bf16(dev(h)); }

// ---- checks -----------------------------------------------------------------------------------------------------

FILE* g_dump = nullptr;
int g_fail = 0;

uint64_t fnv(const void* p, size_t n) {
  const auto* b = static_cast<const unsigned char*>(p);
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

// One output: its hash and error on stdout, its bytes to the dump (name length, name, byte count, bytes).
void report(const char* name, const void* bytes, size_t n, double err, double tol) {
  const char* e = ragon::test::sync_error();
  const bool ok = !e && err <= tol;  // a NaN error fails
  if (!ok) ++g_fail;
  std::printf("%-30s %016llx err %.3e tol %.0e %s%s%s\n", name, static_cast<unsigned long long>(fnv(bytes, n)), err, tol,
              ok ? "ok" : "FAIL", e ? " " : "", e ? e : "");
  if (g_dump) {
    const uint32_t len = static_cast<uint32_t>(std::strlen(name));
    const uint64_t size = n;
    std::fwrite(&len, 4, 1, g_dump);
    std::fwrite(name, 1, len, g_dump);
    std::fwrite(&size, 8, 1, g_dump);
    std::fwrite(bytes, 1, n, g_dump);
  }
}

// max |got - ref| / (|ref| + 1): relative for large values, absolute below 1.
double err_of(const std::vector<float>& got, const std::vector<double>& ref) {
  if (got.size() != ref.size()) return INFINITY;
  double e = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - ref[i]) / (std::fabs(ref[i]) + 1.0);
    if (!(d <= e)) e = d;  // NaN sticks
  }
  return e;
}
template <typename T>
double mismatches(const std::vector<T>& got, const std::vector<T>& ref) {
  if (got.size() != ref.size()) return INFINITY;
  return static_cast<double>(std::memcmp(got.data(), ref.data(), got.size() * sizeof(T)) != 0);
}
template <typename T>
void check(const char* name, const std::vector<T>& got, double err, double tol) {
  report(name, got.data(), got.size() * sizeof(T), err, tol);
}
// A device result that must equal another device result bit for bit (a variant of a kernel checked against its
// reference elsewhere).
template <typename T>
void check_same(const char* name, const std::vector<T>& got, const std::vector<T>& want) {
  check(name, got, mismatches(got, want), 0);
}

double silu(double x) { return x / (1 + std::exp(-x)); }
double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }

// ---- the backend's element helpers --------------------------------------------------------------------------------

__global__ void primitives_kernel(const float* x, int n, uint8_t* enc, float* dec, uint32_t* pair, float* mul) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  enc[i] = gpu::e4m3_from_float(x[i]);
  dec[i] = gpu::e4m3_to_float(static_cast<uint8_t>(i & 0xff));
  const gpu::bf16x2 b = gpu::floats_to_bf16x2(x[i], -0.5f * x[i]);
  uint32_t u;
  memcpy(&u, &b, 4);
  pair[i] = u;
  mul[i] = gpu::mul_rn(x[i], x[(i + 1) % n]);
}

// E4M3 by search: the nearest of the 127 finite magnitudes, ties to the even code, saturating at 448.
uint8_t e4m3_ref(float v) {
  const uint8_t sign = std::signbit(v) ? 0x80 : 0;
  const double a = std::fabs(static_cast<double>(v));
  int best = 0;
  double bd = INFINITY;
  for (int c = 0; c <= 126; ++c) {
    const double d = std::fabs(a - e4m3(static_cast<uint8_t>(c)));
    if (d < bd || (d == bd && (c & 1) == 0 && (best & 1) == 1)) best = c, bd = d;
  }
  return static_cast<uint8_t>(sign | best);
}

void test_primitives(Rng& r) {
  std::vector<float> x;
  for (int c = 0; c < 126; ++c) {  // every code, and the midpoint above it (a tie)
    const double lo = e4m3(static_cast<uint8_t>(c)), hi = e4m3(static_cast<uint8_t>(c + 1));
    x.push_back(static_cast<float>(lo));
    x.push_back(static_cast<float>(-(lo + hi) / 2));
  }
  for (float v : {448.f, 464.f, 480.f, 500.f, 1e6f, -1e6f, 1.0f / 1024, 1.0f / 2048, 3.0f / 4096, 1.00390625f}) x.push_back(v);
  for (int i = 0; i < 512; ++i) x.push_back(r.uni() * std::ldexp(1.f, r.below(24) - 12));
  const int n = static_cast<int>(x.size());
  float* dx = dev(x);
  auto* enc = dev_zero<uint8_t>(n);
  auto* dec = dev_zero<float>(n);
  auto* pair = dev_zero<uint32_t>(n);
  auto* mul = dev_zero<float>(n);
  primitives_kernel<<<(n + 127) / 128, 128>>>(dx, n, enc, dec, pair, mul);
  std::vector<uint8_t> enc_ref(n);
  std::vector<float> dec_ref(n), mul_ref(n);
  std::vector<uint32_t> pair_ref(n);
  for (int i = 0; i < n; ++i) {
    enc_ref[i] = e4m3_ref(x[i]);
    const uint8_t b = static_cast<uint8_t>(i & 0xff);
    dec_ref[i] = ((b & 0x7f) == 0x7f) ? 0.f : e4m3(b);
    pair_ref[i] = bf16_bits(x[i]) | (static_cast<uint32_t>(bf16_bits(-0.5f * x[i])) << 16);
    mul_ref[i] = x[i] * x[(i + 1) % n];
  }
  auto dec_got = download(dec, n);
  for (int i = 0; i < n; ++i)
    if (((i & 0xff) & 0x7f) == 0x7f) dec_got[i] = std::isnan(dec_got[i]) ? 0.f : 1.f;  // NaN codes: NaN, any payload
  check("e4m3_from_float", download(enc, n), mismatches(download(enc, n), enc_ref), 0);
  check("e4m3_to_float", dec_got, mismatches(dec_got, dec_ref), 0);
  check("floats_to_bf16x2", download(pair, n), mismatches(download(pair, n), pair_ref), 0);
  check("mul_rn", download(mul, n), mismatches(download(mul, n), mul_ref), 0);
}

// ---- conversions ------------------------------------------------------------------------------------------------

void test_conversions(Rng& r) {
  {
    std::vector<float> x;
    for (int i = 0; i < 1000; ++i) x.push_back(r.uni() * std::ldexp(1.f, r.below(40) - 20));
    for (float v : {1.00390625f, 1.01171875f, -1.00390625f, 0.f, 3.0e38f}) x.push_back(v);  // ties, zero, near max
    const int n = static_cast<int>(x.size());
    auto* out = dev_zero<bits16>(n);
    K::to_bf16(dev(x), as_bf16(out), n, nullptr);
    const auto got = download(out, n);
    check("to_bf16", got, mismatches(got, to_bits(x)), 0);
  }
  {
    const int N = 16, Kd = 256;
    std::vector<uint8_t> w(size_t(N) * Kd);
    for (auto& b : w) {
      b = static_cast<uint8_t>(r.below(256));
      if ((b & 0x7f) == 0x7f) b ^= 1;  // no NaN codes
    }
    const float scale = 0.0371f;
    auto* out = dev_zero<bits16>(w.size());
    K::dequant_fp8(dev(w), scale, as_bf16(out), N, Kd, nullptr);
    std::vector<bits16> ref(w.size());
    for (size_t i = 0; i < w.size(); ++i) ref[i] = bf16_bits(e4m3(w[i]) * scale);
    const auto got = download(out, w.size());
    check("dequant_fp8", got, mismatches(got, ref), 0);
  }
  {
    // NVFP4: two E2M1 codes per byte (low nibble first), one E4M3 scale per 16 values, times the tensor's scale2.
    static const float kFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const int N = 8, Kd = 512;
    std::vector<uint8_t> w(size_t(N) * Kd / 2), ws(size_t(N) * Kd / 16);
    for (auto& b : w) b = static_cast<uint8_t>(r.below(256));
    for (auto& b : ws) b = static_cast<uint8_t>(r.below(0x7f));  // positive, finite
    const float scale2 = 0.00213f;
    auto* out = dev_zero<bits16>(size_t(N) * Kd);
    K::dequant_nvfp4(dev(w), dev(ws), scale2, as_bf16(out), N, Kd, nullptr);
    std::vector<bits16> ref(size_t(N) * Kd);
    for (size_t i = 0; i < w.size(); ++i) {
      const size_t row = i / (Kd / 2), col = (i % (Kd / 2)) * 2;
      const float sc = e4m3(ws[row * (Kd / 16) + col / 16]) * scale2;
      const float lo = kFp4[w[i] & 0xf] * sc, hi = kFp4[w[i] >> 4] * sc;
      ref[row * Kd + col] = bf16_bits(lo);
      ref[row * Kd + col + 1] = bf16_bits(hi);
    }
    const auto got = download(out, ref.size());
    check("dequant_nvfp4", got, mismatches(got, ref), 0);
  }
}

// ---- elementwise and norms --------------------------------------------------------------------------------------

std::vector<double> rmsnorm_ref(const std::vector<float>& x, const std::vector<bits16>& w, int rows, int H, float eps,
                                bool gemma) {
  std::vector<double> out(x.size());
  for (int m = 0; m < rows; ++m) {
    double ss = 0;
    for (int h = 0; h < H; ++h) ss += double(x[size_t(m) * H + h]) * x[size_t(m) * H + h];
    const double rstd = 1 / std::sqrt(ss / H + eps);
    for (int h = 0; h < H; ++h) {
      const double wv = bf16_float(w[h]);
      out[size_t(m) * H + h] = x[size_t(m) * H + h] * rstd * (gemma ? 1 + wv : wv);
    }
  }
  return out;
}

void test_elementwise(Rng& r) {
  {
    const int V = 50, H = 200, M = 5;
    const auto table = to_bits(r.vec(size_t(V) * H));
    std::vector<int> ids(M);
    for (auto& i : ids) i = r.below(V);
    auto* out = dev_zero<float>(size_t(M) * H);
    K::embed(dev_bf16(table), dev(ids), M, H, out, nullptr);
    std::vector<float> ref(size_t(M) * H);
    for (int m = 0; m < M; ++m)
      for (int h = 0; h < H; ++h) ref[size_t(m) * H + h] = bf16_float(table[size_t(ids[m]) * H + h]);
    const auto got = download(out, ref.size());
    check("embed", got, mismatches(got, ref), 0);
  }
  // H below 1024 is the block size, so it must be a whole number of warps (block_sum shuffles full warps).
  for (const auto& [H, gemma, name] : {std::tuple{2560, true, "rmsnorm_gemma_2560"}, std::tuple{480, false, "rmsnorm_480"},
                                       std::tuple{1024, true, "rmsnorm_gemma_1024"}}) {
    const int rows = 3;
    const auto x = r.vec(size_t(rows) * H, 3.f);
    const auto w = to_bits(r.vec(H, 0.5f));
    auto* out = dev_zero<float>(x.size());
    K::rmsnorm(dev(x), dev_bf16(w), out, rows, H, 1e-6f, gemma, nullptr);
    const auto got = download(out, x.size());
    check(name, got, err_of(got, rmsnorm_ref(x, w, rows, H, 1e-6f, gemma)), 2e-5);
  }
  {
    const int n = 3001;
    const auto h = r.vec(n, 4.f), d = r.vec(n, 4.f);
    float* dh = dev(h);
    K::add_inplace(dh, dev(d), n, nullptr);
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = h[i] + d[i];
    const auto got = download(dh, n);
    check("add_inplace", got, mismatches(got, ref), 0);
  }
  {
    const int n = 3001;
    const auto g = r.vec(n, 8.f), u = r.vec(n, 2.f);
    auto* out = dev_zero<float>(n);
    K::silu_mul(dev(g), dev(u), out, n, nullptr);
    std::vector<double> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = silu(g[i]) * u[i];
    const auto got = download(out, n);
    check("silu_mul", got, err_of(got, ref), 2e-6);
  }
  {
    const int n = 3001;
    const auto x = r.vec(n, 2.f), gate = r.vec(n, 8.f);
    float* dx = dev(x);
    K::sigmoid_mul(dx, dev(gate), n, nullptr);
    std::vector<double> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = x[i] * sigmoid(gate[i]);
    const auto got = download(dx, n);
    check("sigmoid_mul", got, err_of(got, ref), 2e-6);
  }
  {
    const int rows = 4, D = 128;
    const auto x = r.vec(size_t(rows) * D, 2.f), z = r.vec(size_t(rows) * D, 4.f);
    const auto w = to_bits(r.vec(D, 1.f));
    auto* out = dev_zero<float>(x.size());
    K::gated_rmsnorm(dev(x), dev(z), dev_bf16(w), out, rows, D, 1e-6f, nullptr);
    auto ref = rmsnorm_ref(x, w, rows, D, 1e-6f, false);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] *= silu(z[i]);
    const auto got = download(out, x.size());
    check("gated_rmsnorm", got, err_of(got, ref), 2e-5);
  }
}

// ---- the bf16 GEMVs ---------------------------------------------------------------------------------------------

std::vector<double> matvec_ref(const std::vector<float>& x, const std::vector<bits16>& w, int M, int N, int Kd) {
  std::vector<double> y(size_t(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      for (int k = 0; k < Kd; ++k) s += double(x[size_t(m) * Kd + k]) * bf16_float(w[size_t(n) * Kd + k]);
      y[size_t(m) * N + n] = s;
    }
  return y;
}

void test_gemv(Rng& r) {
  {
    const int M = 3, N = 40, Kd = 700;
    const auto x = r.vec(size_t(M) * Kd);
    const auto w = to_bits(r.vec(size_t(N) * Kd));
    auto* y = dev_zero<float>(size_t(M) * N);
    K::gemv_bf16(dev(x), M, dev_bf16(w), y, N, Kd, nullptr);
    const auto got = download(y, size_t(M) * N);
    check("gemv_bf16", got, err_of(got, matvec_ref(x, w, M, N, Kd)), 2e-5);
  }
  {
    const int M = 4, N = 24, Kd = 2048;
    // x is rounded to bf16 here so that the reference holds whether or not the kernel rounds it.
    const auto x = from_bits(to_bits(r.vec(size_t(M) * Kd)));
    const auto w = to_bits(r.vec(size_t(N) * Kd)), w2 = to_bits(r.vec(size_t(N) * Kd));
    auto* y = dev_zero<float>(size_t(M) * N);
    auto* y2 = dev_zero<float>(size_t(M) * N);
    K::bf16_rows(dev(x), M, dev_bf16(w), y, N, Kd, nullptr, dev_bf16(w2), y2);
    const auto got = download(y, size_t(M) * N), got2 = download(y2, size_t(M) * N);
    check("bf16_rows", got, err_of(got, matvec_ref(x, w, M, N, Kd)), 2e-5);
    check("bf16_rows_w2", got2, err_of(got2, matvec_ref(x, w2, M, N, Kd)), 2e-5);
  }
}

// ---- the gated DeltaNet ------------------------------------------------------------------------------------------

constexpr int kDk = 128, kDv = 128;

// gdn_conv over rows of mixed [M][C] from state [C][3]: both updated.
void conv_ref(std::vector<double>& out, std::vector<float>& state, const std::vector<float>& mixed,
              const std::vector<bits16>& w, int M, int C) {
  out.assign(size_t(M) * C, 0);
  for (int c = 0; c < C; ++c) {
    float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    for (int t = 0; t < M; ++t) {
      const float xv = mixed[size_t(t) * C + c];
      const double o = bf16_float(w[c * 4]) * double(s0) + bf16_float(w[c * 4 + 1]) * double(s1) +
                       bf16_float(w[c * 4 + 2]) * double(s2) + bf16_float(w[c * 4 + 3]) * double(xv);
      out[size_t(t) * C + c] = silu(o);
      s0 = s1, s1 = s2, s2 = xv;
    }
    state[c * 3] = s0, state[c * 3 + 1] = s1, state[c * 3 + 2] = s2;
  }
}

// The gated delta rule over rows of mixed [M][C]; state [HV][Dk][Dv] advanced in place.
std::vector<double> recurrent_ref(const std::vector<float>& mixed, const std::vector<float>& g,
                                  const std::vector<float>& beta, std::vector<double>& S, int M, int H, int HV) {
  const int C = 2 * H * kDk + HV * kDv;
  std::vector<double> out(size_t(M) * HV * kDv);
  for (int t = 0; t < M; ++t)
    for (int hv = 0; hv < HV; ++hv) {
      const int h = hv / (HV / H);
      const float* row = mixed.data() + size_t(t) * C;
      double qn = 0, kn = 0;
      for (int k = 0; k < kDk; ++k) qn += double(row[h * kDk + k]) * row[h * kDk + k];
      for (int k = 0; k < kDk; ++k) kn += double(row[H * kDk + h * kDk + k]) * row[H * kDk + h * kDk + k];
      const double qr = 1 / std::sqrt(qn + 1e-6) / std::sqrt(double(kDk)), kr = 1 / std::sqrt(kn + 1e-6);
      const double decay = std::exp(double(g[t * HV + hv])), b = beta[t * HV + hv];
      double* s = S.data() + size_t(hv) * kDk * kDv;
      for (int v = 0; v < kDv; ++v) {
        double kv = 0;
        for (int k = 0; k < kDk; ++k) {
          s[k * kDv + v] *= decay;
          kv += s[k * kDv + v] * row[H * kDk + h * kDk + k] * kr;
        }
        const double d = (row[2 * H * kDk + hv * kDv + v] - kv) * b;
        double o = 0;
        for (int k = 0; k < kDk; ++k) {
          s[k * kDv + v] += d * row[H * kDk + h * kDk + k] * kr;
          o += s[k * kDv + v] * row[h * kDk + k] * qr;
        }
        out[(size_t(t) * HV + hv) * kDv + v] = o;
      }
    }
  return out;
}

struct GdnCase {
  int H = 1, HV = 2, C = 0;
  std::vector<float> mixed, g, beta, state;
  GdnCase(Rng& r, int M, int H_, int HV_) : H(H_), HV(HV_), C(2 * H_ * kDk + HV_ * kDv) {
    mixed = r.vec(size_t(M) * C);
    g = r.vec(size_t(M) * HV, 0.2f);
    for (auto& x : g) x = -0.2f - x;  // decays in (0.67, 1]
    beta = r.vec(size_t(M) * HV, 0.5f);
    for (auto& x : beta) x += 0.5f;
    state = r.vec(size_t(HV) * kDk * kDv, 0.1f);
  }
  template <typename T>
  static std::vector<T> rows(const std::vector<T>& v, int width, const std::vector<int>& which) {
    std::vector<T> o;
    for (int t : which) o.insert(o.end(), v.begin() + size_t(t) * width, v.begin() + size_t(t + 1) * width);
    return o;
  }
};

void test_gdn(Rng& r) {
  // gdn_conv: in place over M rows, the window advanced.
  const int C = 96, M = 3;
  const auto mixed = r.vec(size_t(M) * C, 2.f);
  const auto cstate = r.vec(size_t(C) * 3, 2.f);
  const auto cw = to_bits(r.vec(size_t(C) * 4, 1.f));
  std::vector<float> conv_out, conv_state;
  {
    float* dm = dev(mixed);
    float* ds = dev(cstate);
    K::gdn_conv(dm, ds, dev_bf16(cw), M, C, nullptr);
    std::vector<double> ref;
    std::vector<float> sref = cstate;
    conv_ref(ref, sref, mixed, cw, M, C);
    conv_out = download(dm, mixed.size());
    conv_state = download(ds, cstate.size());
    check("gdn_conv", conv_out, err_of(conv_out, ref), 2e-6);
    check("gdn_conv_state", conv_state, mismatches(conv_state, sref), 0);
  }
  {
    const int GM = 3, HV = 8;
    auto a = r.vec(size_t(GM) * HV, 4.f), b = r.vec(size_t(GM) * HV, 6.f);
    a[3] = 25.f, a[11] = 19.5f;  // the softplus's linear branch, and just under it
    const auto A_log = to_bits(r.vec(HV, 1.f)), dt_bias = to_bits(r.vec(HV, 1.f));
    auto* g = dev_zero<float>(a.size());
    auto* beta = dev_zero<float>(a.size());
    K::gdn_gating(dev(a), dev(b), dev_bf16(A_log), dev_bf16(dt_bias), g, beta, GM, HV, nullptr);
    std::vector<double> gref(a.size()), bref(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      const int hv = static_cast<int>(i % HV);
      const double x = double(a[i]) + bf16_float(dt_bias[hv]);
      const double sp = x <= 20 ? std::log1p(std::exp(x)) : x;
      gref[i] = -std::exp(double(bf16_float(A_log[hv]))) * sp;
      bref[i] = sigmoid(b[i]);
    }
    const auto gg = download(g, a.size()), gb = download(beta, a.size());
    check("gdn_gating_g", gg, err_of(gg, gref), 2e-6);
    check("gdn_gating_beta", gb, err_of(gb, bref), 2e-6);
  }
  // gdn_recurrent: two value heads on one key head (the grouped case), three rows.
  const int RM = 3, H = 1, HV = 2;
  GdnCase rc(r, RM, H, HV);
  const size_t SN = size_t(HV) * kDk * kDv, ON = size_t(RM) * HV * kDv;
  std::vector<float> rec_out, rec_state;
  {
    float* sin = dev(rc.state);
    auto* sout = dev_zero<float>(SN);
    auto* out = dev_zero<float>(ON);
    K::gdn_recurrent(dev(rc.mixed), dev(rc.g), dev(rc.beta), sin, sout, out, RM, H, HV, nullptr);
    std::vector<double> S(rc.state.begin(), rc.state.end());
    const auto ref = recurrent_ref(rc.mixed, rc.g, rc.beta, S, RM, H, HV);
    rec_out = download(out, ON);
    rec_state = download(sout, SN);
    check("gdn_recurrent", rec_out, err_of(rec_out, ref), 2e-5);
    check("gdn_recurrent_state", rec_state, err_of(rec_state, S), 2e-5);
  }

  // The batched forms: sequence 0 is the chain above's rows, sequence 1 the conv's (its own state) reversed; each
  // sequence's results equal the single-sequence kernel's bit for bit.
  SeqBatch b;
  b.n = 2;
  b.row0[0] = 0, b.row0[1] = 2, b.row0[2] = 3;  // sequence 0: rows 0-1, sequence 1: row 2
  b.pos[0] = 7, b.pos[1] = 30;
  {
    const auto st1 = r.vec(size_t(C) * 3, 2.f);
    float* dm = dev(mixed);
    SeqPtrs<const float> sp;
    sp.p[0] = dev(cstate);
    sp.p[1] = dev(st1);
    K::gdn_conv_seqs(dm, sp, dev_bf16(cw), b, C, nullptr);
    // The singles: rows 0-1 from cstate, row 2 from st1.
    float* m0 = dev(GdnCase::rows(mixed, C, {0, 1}));
    float* m1 = dev(GdnCase::rows(mixed, C, {2}));
    float* s0 = dev(cstate);
    float* s1 = dev(st1);
    K::gdn_conv(m0, s0, dev_bf16(cw), 2, C, nullptr);
    K::gdn_conv(m1, s1, dev_bf16(cw), 1, C, nullptr);
    auto want = download(m0, size_t(2) * C);
    const auto w1 = download(m1, C);
    want.insert(want.end(), w1.begin(), w1.end());
    check_same("gdn_conv_seqs", download(dm, mixed.size()), want);

    float* dm2 = dev(mixed);
    SeqPtrs<float> spw;
    spw.p[0] = dev(cstate);
    spw.p[1] = dev(st1);
    K::gdn_conv_commit_seqs(dm2, spw, dev_bf16(cw), b, C, nullptr);
    check_same("gdn_conv_commit_seqs", download(dm2, mixed.size()), want);
    auto st_got = download(spw.p[0], cstate.size());
    const auto st1_got = download(spw.p[1], st1.size());
    st_got.insert(st_got.end(), st1_got.begin(), st1_got.end());
    auto st_want = download(s0, cstate.size());
    const auto st1_want = download(s1, st1.size());
    st_want.insert(st_want.end(), st1_want.begin(), st1_want.end());
    check_same("gdn_conv_commit_seqs_state", st_got, st_want);
  }
  {
    const auto st1 = r.vec(SN, 0.1f);
    const int C2 = rc.C;
    float* dmix = dev(rc.mixed);
    float* dg = dev(rc.g);
    float* db = dev(rc.beta);
    SeqPtrs<const float> sp;
    sp.p[0] = dev(rc.state);
    sp.p[1] = dev(st1);
    auto* out = dev_zero<float>(ON);
    K::gdn_recurrent_seqs(dmix, dg, db, sp, out, b, H, HV, nullptr);
    // The singles.
    float* s0 = dev(rc.state);
    float* s1 = dev(st1);
    auto* so0 = dev_zero<float>(SN);
    auto* so1 = dev_zero<float>(SN);
    auto* o0 = dev_zero<float>(size_t(2) * HV * kDv);
    auto* o1 = dev_zero<float>(size_t(HV) * kDv);
    K::gdn_recurrent(dev(GdnCase::rows(rc.mixed, C2, {0, 1})), dev(GdnCase::rows(rc.g, HV, {0, 1})),
                     dev(GdnCase::rows(rc.beta, HV, {0, 1})), s0, so0, o0, 2, H, HV, nullptr);
    K::gdn_recurrent(dev(GdnCase::rows(rc.mixed, C2, {2})), dev(GdnCase::rows(rc.g, HV, {2})),
                     dev(GdnCase::rows(rc.beta, HV, {2})), s1, so1, o1, 1, H, HV, nullptr);
    auto want = download(o0, size_t(2) * HV * kDv);
    const auto w1 = download(o1, size_t(HV) * kDv);
    want.insert(want.end(), w1.begin(), w1.end());
    check_same("gdn_recurrent_seqs", download(out, ON), want);

    SeqPtrs<float> spw;
    spw.p[0] = dev(rc.state);
    spw.p[1] = dev(st1);
    auto* out2 = dev_zero<float>(ON);
    K::gdn_recurrent_commit_seqs(dmix, dg, db, spw, out2, b, H, HV, nullptr);
    check_same("gdn_recurrent_commit_seqs", download(out2, ON), want);
    auto st_got = download(spw.p[0], SN);
    const auto st1_got = download(spw.p[1], SN);
    st_got.insert(st_got.end(), st1_got.begin(), st1_got.end());
    auto st_want = download(so0, SN);
    const auto st1_want = download(so1, SN);
    st_want.insert(st_want.end(), st1_want.begin(), st1_want.end());
    check_same("gdn_recurrent_commit_seqs_state", st_got, st_want);
  }

  // The tree forms: rows 1 and 2 are children of 0, row 3 a child of 1 (paths 0-1-3 and 0-2). Each row equals the
  // chain over its root path bit for bit.
  {
    const int TM = 4, TC = rc.C;
    TreeDesc td;
    td.n = TM;
    td.depth[0] = 0, td.depth[1] = 1, td.depth[2] = 1, td.depth[3] = 2;
    td.var[0] = 0, td.var[1] = 0, td.var[2] = 1, td.var[3] = 0;
    td.nvar = 2;
    td.path[0][0] = 0, td.path[0][1] = 1, td.path[0][2] = 3;
    td.path[1][0] = 0, td.path[1][1] = 2;
    td.ngdn = 2;
    td.gdn_len[0] = 3, td.gdn_len[1] = 2;
    td.gdn_step[0][0] = 0 | TreeDesc::kWrite, td.gdn_step[0][1] = 1 | TreeDesc::kWrite;
    td.gdn_step[0][2] = 3 | TreeDesc::kWrite | TreeDesc::kLeaf;
    td.gdn_step[1][0] = 0, td.gdn_step[1][1] = 2 | TreeDesc::kWrite | TreeDesc::kLeaf;
    std::vector<TreeDesc> tdv{td};
    const TreeDesc* dtd = dev(tdv);
    const std::vector<int> pa{0, 1, 3}, pb{0, 2};

    // The conv over the GDN width (its key/value channels): taps from each row's ancestors.
    GdnCase tc(r, TM, H, HV);
    const auto tstate = r.vec(size_t(TC) * 3, 1.f);
    const auto tw = to_bits(r.vec(size_t(TC) * 4, 1.f));
    auto* post = dev_zero<float>(size_t(TM) * TC);
    K::gdn_conv_tree(dev(tc.mixed), post, dev(tstate), dev_bf16(tw), TM, TC, dtd, nullptr);
    std::vector<float> want(size_t(TM) * TC);
    for (const auto& path : {pa, pb}) {
      float* m = dev(GdnCase::rows(tc.mixed, TC, path));
      K::gdn_conv(m, dev(tstate), dev_bf16(tw), static_cast<int>(path.size()), TC, nullptr);
      const auto got = download(m, path.size() * TC);
      for (size_t i = 0; i < path.size(); ++i)
        std::copy(got.begin() + i * TC, got.begin() + (i + 1) * TC, want.begin() + size_t(path[i]) * TC);
    }
    const auto tree_post = download(post, size_t(TM) * TC);
    check_same("gdn_conv_tree", tree_post, want);

    // The recurrence over the tree's rows (the conv's outputs as its inputs).
    auto* tout = dev_zero<float>(size_t(TM) * HV * kDv);
    K::gdn_recurrent_tree(post, dev(tc.g), dev(tc.beta), dev(tc.state), tout, H, HV, dtd, 2, nullptr);
    std::vector<float> rwant(size_t(TM) * HV * kDv);
    std::vector<double> rref(rwant.size());
    for (const auto& path : {pa, pb}) {
      const int n = static_cast<int>(path.size());
      auto* o = dev_zero<float>(size_t(n) * HV * kDv);
      auto* so = dev_zero<float>(SN);
      const auto pm = GdnCase::rows(tree_post, TC, path), pg = GdnCase::rows(tc.g, HV, path);
      const auto pbt = GdnCase::rows(tc.beta, HV, path);
      K::gdn_recurrent(dev(pm), dev(pg), dev(pbt), dev(tc.state), so, o, n, H, HV, nullptr);
      const auto got = download(o, size_t(n) * HV * kDv);
      std::vector<double> S(tc.state.begin(), tc.state.end());
      const auto ref = recurrent_ref(pm, pg, pbt, S, n, H, HV);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < HV * kDv; ++j) {
          rwant[size_t(path[i]) * HV * kDv + j] = got[size_t(i) * HV * kDv + j];
          rref[size_t(path[i]) * HV * kDv + j] = ref[size_t(i) * HV * kDv + j];
        }
    }
    const auto tree_out = download(tout, rwant.size());
    check("gdn_recurrent_tree_vs_ref", tree_out, err_of(tree_out, rref), 2e-5);
    check_same("gdn_recurrent_tree", tree_out, rwant);
  }
}

// ---- attention's prepare and the KV page moves --------------------------------------------------------------------

// One head's normalized and rotated vector (NeoX halves on the first rot dims), as attn_prepare computes it.
std::vector<double> norm_rope_ref(const float* x, const std::vector<bits16>& w, int D, int rot, float theta, float eps,
                                  int pos, bool one_plus) {
  double ss = 0;
  for (int d = 0; d < D; ++d) ss += double(x[d]) * x[d];
  const double rstd = 1 / std::sqrt(ss / D + eps);
  std::vector<double> xn(D), o(D);
  for (int d = 0; d < D; ++d) xn[d] = x[d] * rstd * (one_plus ? 1 + double(bf16_float(w[d])) : double(bf16_float(w[d])));
  for (int d = 0; d < D; ++d) {
    o[d] = xn[d];
    if (d < rot) {
      const int half = rot / 2, i = d % half;
      const float inv_freq = std::pow(theta, -2.f * i / rot);
      const double ang = double(static_cast<float>(pos) * inv_freq);
      o[d] = d < half ? xn[d] * std::cos(ang) - xn[d + half] * std::sin(ang)
                      : xn[d] * std::cos(ang) + xn[d - half] * std::sin(ang);
    }
  }
  return o;
}

void test_attn_prepare(Rng& r) {
  const int M = 3, Hq = 2, Hkv = 1, D = 128, rot = 64, P = 16, pos0 = 5;
  const float theta = 1e7f, eps = 1e-6f;
  const auto qg = r.vec(size_t(M) * Hq * 2 * D, 2.f), k = r.vec(size_t(M) * Hkv * D, 2.f);
  const auto v = r.vec(size_t(M) * Hkv * D, 2.f);
  const auto qn = to_bits(r.vec(D, 0.5f)), kn = to_bits(r.vec(D, 0.5f));
  float* dqg = dev(qg);
  float* dk = dev(k);
  float* dv = dev(v);
  gpu::bf16* dqn = dev_bf16(qn);
  gpu::bf16* dkn = dev_bf16(kn);
  auto* q = dev_zero<float>(size_t(M) * Hq * D);
  auto* gate = dev_zero<float>(size_t(M) * Hq * D);
  auto* kc = dev_zero<bits16>(size_t(P) * Hkv * D);
  auto* vc = dev_zero<bits16>(size_t(P) * Hkv * D);
  K::attn_prepare(dqg, dk, dv, dqn, dkn, pos0, M, Hq, Hkv, D, rot, theta, eps, q, gate, as_bf16(kc), as_bf16(vc),
                  nullptr);
  std::vector<double> qref(size_t(M) * Hq * D);
  std::vector<float> gref(size_t(M) * Hq * D);
  std::vector<double> kref(size_t(P) * Hkv * D, 0.0);
  std::vector<bits16> vref(size_t(P) * Hkv * D, 0);
  for (int m = 0; m < M; ++m) {
    for (int h = 0; h < Hq; ++h) {
      const float* src = qg.data() + (size_t(m) * Hq + h) * 2 * D;
      const auto o = norm_rope_ref(src, qn, D, rot, theta, eps, pos0 + m, true);
      for (int d = 0; d < D; ++d) qref[(size_t(m) * Hq + h) * D + d] = o[d], gref[(size_t(m) * Hq + h) * D + d] = src[D + d];
    }
    for (int h = 0; h < Hkv; ++h) {
      const auto o = norm_rope_ref(k.data() + (size_t(m) * Hkv + h) * D, kn, D, rot, theta, eps, pos0 + m, true);
      for (int d = 0; d < D; ++d) {
        kref[(size_t(pos0 + m) * Hkv + h) * D + d] = o[d];
        vref[(size_t(pos0 + m) * Hkv + h) * D + d] = bf16_bits(v[(size_t(m) * Hkv + h) * D + d]);
      }
    }
  }
  const auto gq = download(q, qref.size()), gg = download(gate, gref.size());
  const auto gkc = download(kc, kref.size()), gvc = download(vc, vref.size());
  check("attn_prepare_q", gq, err_of(gq, qref), 2e-5);
  check("attn_prepare_gate", gg, mismatches(gg, gref), 0);
  check("attn_prepare_kcache", gkc, err_of(from_bits(gkc), kref), 8e-3);
  check("attn_prepare_vcache", gvc, mismatches(gvc, vref), 0);

  // The batched form: sequence 0 rows 0-1 at 5, sequence 1 row 2 at 9 (own caches), each as attn_prepare's rows.
  SeqBatch b;
  b.n = 2;
  b.row0[0] = 0, b.row0[1] = 2, b.row0[2] = 3;
  b.pos[0] = pos0, b.pos[1] = 9;
  SeqPtrs<gpu::bf16> kcs, vcs;
  for (int i = 0; i < 2; ++i) {
    kcs.p[i] = as_bf16(dev_zero<bits16>(size_t(P) * Hkv * D));
    vcs.p[i] = as_bf16(dev_zero<bits16>(size_t(P) * Hkv * D));
  }
  auto* q2 = dev_zero<float>(size_t(M) * Hq * D);
  auto* gate2 = dev_zero<float>(size_t(M) * Hq * D);
  K::attn_prepare_seqs(dqg, dk, dv, dqn, dkn, b, Hq, Hkv, D, rot, theta, eps, q2, gate2, kcs, vcs, nullptr);
  // Row 2 alone at position 9.
  auto* q1 = dev_zero<float>(size_t(Hq) * D);
  auto* g1 = dev_zero<float>(size_t(Hq) * D);
  auto* kc1 = dev_zero<bits16>(size_t(P) * Hkv * D);
  auto* vc1 = dev_zero<bits16>(size_t(P) * Hkv * D);
  K::attn_prepare(dqg + size_t(2) * Hq * 2 * D, dk + size_t(2) * Hkv * D, dv + size_t(2) * Hkv * D, dqn, dkn, 9, 1, Hq,
                  Hkv, D, rot, theta, eps, q1, g1, as_bf16(kc1), as_bf16(vc1), nullptr);
  auto qwant = download(q, size_t(2) * Hq * D);
  const auto q1h = download(q1, size_t(Hq) * D);
  qwant.insert(qwant.end(), q1h.begin(), q1h.end());
  check_same("attn_prepare_seqs_q", download(q2, size_t(M) * Hq * D), qwant);
  auto kwant = gkc;  // sequence 0's cache: rows 0-1 of the single call at 5, 6 (row 2's slot 7 zero)
  for (size_t i = size_t(pos0 + 2) * Hkv * D; i < size_t(pos0 + 3) * Hkv * D; ++i) kwant[i] = 0;
  auto kgot = download(reinterpret_cast<bits16*>(kcs.p[0]), kref.size());
  const auto k1got = download(reinterpret_cast<bits16*>(kcs.p[1]), kref.size());
  kgot.insert(kgot.end(), k1got.begin(), k1got.end());
  const auto k1want = download(kc1, kref.size());
  kwant.insert(kwant.end(), k1want.begin(), k1want.end());
  check_same("attn_prepare_seqs_kcache", kgot, kwant);
}

void test_kv_moves(Rng& r) {
  {
    const int T = 300;
    std::vector<uint64_t> table(T);
    for (auto& x : table) x = (uint64_t(r.g()) << 32) | r.g();
    uint64_t* dt = dev(table);
    std::vector<int> idx;
    std::vector<uint64_t> val;
    for (int i = 0; i < T; i += 1 + (i % 3)) idx.push_back(i), val.push_back(uint64_t(i) * 0x9e3779b97f4a7c15ull);
    K::kv_table_set(dt, idx.data(), val.data(), static_cast<int>(idx.size()), nullptr);  // two launches (> 256)
    for (size_t i = 0; i < idx.size(); ++i) table[idx[i]] = val[i];
    const auto got = download(dt, T);
    check("kv_table_set", got, mismatches(got, table), 0);
  }
  {
    // Paged layout: a page holds L layer-and-K/V blocks of Hkv heads x 64 positions x D.
    const int D = 128, Hkv = 2, L = 2, lkv = L, npages = 3, page = L * Hkv * 64 * D;
    const int buf_start = 8, buf_len = 150, from = 10, to = 150;
    K::set_kv_layout(size_t(64) * D, D);
    std::vector<bits16> pool(size_t(npages) * page);
    for (auto& x : pool) x = static_cast<bits16>(r.g());
    const std::vector<uint64_t> pages{uint64_t(2) * page, 0, uint64_t(1) * page};
    std::vector<bits16> buf(size_t(lkv) * Hkv * buf_len * D);
    for (auto& x : buf) x = static_cast<bits16>(r.g());
    bits16* dpool = dev(pool);
    bits16* dbuf = dev(buf);
    uint64_t* dpages = dev(pages);
    auto pool_at = [&](int p, int l, int h) { return pages[p / 64] + size_t(l) * Hkv * 64 * D + size_t(p % 64) * D + size_t(h) * 64 * D; };
    auto buf_at = [&](int p, int l, int h) { return (size_t(l) * Hkv * buf_len + size_t(p - buf_start)) * D + size_t(h) * buf_len * D; };
    // Out of the pages into the flat buffer, then a fresh buffer back into the pages.
    K::kv_pages_copy(as_bf16(dpool), dpages, lkv, Hkv, D, as_bf16(dbuf), buf_start, buf_len, from, to, true, nullptr);
    for (int p = from; p < to; ++p)
      for (int l = 0; l < lkv; ++l)
        for (int h = 0; h < Hkv; ++h) std::copy_n(pool.begin() + pool_at(p, l, h), D, buf.begin() + buf_at(p, l, h));
    const auto got = download(dbuf, buf.size());
    check("kv_pages_copy_to_buf", got, mismatches(got, buf), 0);
    std::vector<bits16> buf2(buf.size());
    for (auto& x : buf2) x = static_cast<bits16>(r.g());
    K::kv_pages_copy(as_bf16(dpool), dpages, lkv, Hkv, D, as_bf16(dev(buf2)), buf_start, buf_len, from + 3, to - 5, false,
                     nullptr);
    for (int p = from + 3; p < to - 5; ++p)
      for (int l = 0; l < lkv; ++l)
        for (int h = 0; h < Hkv; ++h) std::copy_n(buf2.begin() + buf_at(p, l, h), D, pool.begin() + pool_at(p, l, h));
    const auto gpool = download(dpool, pool.size());
    check("kv_pages_copy_to_pool", gpool, mismatches(gpool, pool), 0);
    K::set_kv_layout(0, 0);
  }
  {
    // Two buffers: a flat one at row0 and a paged one at the position; rows moved in order j = 1 .. n - 1.
    const int n = 5, rows_flat = 16, stride = 96, row_bytes = 64;
    const std::vector<int> rows{0, 2, 2, 4, 3};
    const int pos = 60, page_unit = 16 * 1024;
    std::vector<uint8_t> a(size_t(rows_flat) * stride), pg(size_t(3) * page_unit);
    for (auto& x : a) x = static_cast<uint8_t>(r.g());
    for (auto& x : pg) x = static_cast<uint8_t>(r.g());
    uint8_t* da = dev(a);
    uint8_t* dp = dev(pg);
    const std::vector<uint64_t> table{2, 0, 1};
    std::vector<K::GatherBuf> bufs(2);
    bufs[0] = {reinterpret_cast<char*>(da), size_t(stride), row_bytes, 0};
    bufs[1] = {reinterpret_cast<char*>(dp), size_t(128), row_bytes, 1, dev(table), 1};
    bufs[1].page_unit = page_unit;
    const int row0 = 3;
    K::tree_gather(dev(bufs), 2, dev(rows), n, pos, nullptr, row0);
    auto move = [&](std::vector<uint8_t>& v, auto at, size_t off) {
      for (int j = 1; j < n; ++j)
        if (rows[j] != j) std::copy_n(v.begin() + at(off + rows[j]), row_bytes, v.begin() + at(off + j));
    };
    move(a, [&](size_t p) { return p * stride; }, row0);
    move(pg, [&](size_t p) { return table[p / 64] * page_unit + (p % 64) * 128; }, pos);
    auto got = download(da, a.size());
    const auto gp = download(dp, pg.size());
    got.insert(got.end(), gp.begin(), gp.end());
    auto want = a;
    want.insert(want.end(), pg.begin(), pg.end());
    check("tree_gather", got, mismatches(got, want), 0);
  }
}

// ---- the drafter ---------------------------------------------------------------------------------------------------

constexpr int kDD = 128;

void test_drafter(Rng& r) {
  const float theta = 1e6f, eps = 1e-6f;
  {
    const int rows = 3, heads = 4, row_stride = heads * kDD + 64, pos0 = 11;
    const auto x = r.vec(size_t(rows) * row_stride, 2.f);
    const auto w = to_bits(r.vec(kDD, 1.f));
    float* dx = dev(x);
    K::draft_qk_rope(dx, rows, heads, row_stride, dev_bf16(w), pos0, theta, eps, nullptr);
    std::vector<double> ref(x.begin(), x.end());
    for (int m = 0; m < rows; ++m)
      for (int h = 0; h < heads; ++h) {
        const auto o = norm_rope_ref(x.data() + size_t(m) * row_stride + h * kDD, w, kDD, kDD, theta, eps, pos0 + m, false);
        std::copy(o.begin(), o.end(), ref.begin() + size_t(m) * row_stride + h * kDD);
      }
    const auto got = download(dx, x.size());
    check("draft_qk_rope", got, err_of(got, ref), 2e-5);

    SeqBatch b;
    b.n = 2;
    b.row0[0] = 0, b.row0[1] = 1, b.row0[2] = 3;
    b.pos[0] = pos0, b.pos[1] = 40;
    float* dx2 = dev(x);
    K::draft_qk_rope_seqs(dx2, heads, row_stride, dev_bf16(w), b, theta, eps, nullptr);
    float* dx3 = dev(x);
    K::draft_qk_rope(dx3, 1, heads, row_stride, dev_bf16(w), pos0, theta, eps, nullptr);
    K::draft_qk_rope(dx3 + row_stride, 2, heads, row_stride, dev_bf16(w), 40, theta, eps, nullptr);
    check_same("draft_qk_rope_seqs", download(dx2, x.size()), download(dx3, x.size()));
  }
  {
    const int rows = 3, kv_size = 256, pos0 = 14, ring = 16;
    const auto k = r.vec(size_t(rows) * kv_size, 2.f), v = r.vec(size_t(rows) * kv_size, 2.f);
    auto* kc = dev_zero<bits16>(size_t(ring) * kv_size);
    auto* vc = dev_zero<bits16>(size_t(ring) * kv_size);
    K::draft_store_kv(dev(k), dev(v), rows, kv_size, pos0, as_bf16(kc), as_bf16(vc), ring, nullptr);
    std::vector<bits16> kref(size_t(ring) * kv_size, 0), vref(kref);
    for (int m = 0; m < rows; ++m)
      for (int i = 0; i < kv_size; ++i) {
        kref[size_t((pos0 + m) % ring) * kv_size + i] = bf16_bits(k[size_t(m) * kv_size + i]);
        vref[size_t((pos0 + m) % ring) * kv_size + i] = bf16_bits(v[size_t(m) * kv_size + i]);
      }
    auto got = download(kc, kref.size());
    const auto gv = download(vc, vref.size());
    got.insert(got.end(), gv.begin(), gv.end());
    kref.insert(kref.end(), vref.begin(), vref.end());
    check("draft_store_kv", got, mismatches(got, kref), 0);
  }
  {
    // A window of 6 over a 16-slot ring at L = 10, three block rows; Hq = 4 Hkv.
    const int Hkv = 2, Hq = 4 * Hkv, ring = 16, window_left = 6, L = 10, B = 3;
    const auto q = r.vec(size_t(B) * Hq * kDD, 2.f);
    const auto kc = to_bits(r.vec(size_t(ring) * Hkv * kDD, 2.f)), vc = to_bits(r.vec(size_t(ring) * Hkv * kDD, 2.f));
    const auto kb = r.vec(size_t(B) * Hkv * kDD, 2.f), vb = r.vec(size_t(B) * Hkv * kDD, 2.f);
    std::vector<double> ref(size_t(B) * Hq * kDD);
    for (int j = 0; j < B; ++j)
      for (int hq = 0; hq < Hq; ++hq) {
        const int kvh = hq / 4, lo = std::max(0, L + j - window_left);
        const float* qv = q.data() + (size_t(j) * Hq + hq) * kDD;
        std::vector<double> sc;
        std::vector<const float*> vrow;
        std::vector<std::vector<float>> ctx_v;
        auto dot = [&](auto key) {
          double s = 0;
          for (int d = 0; d < kDD; ++d) s += double(qv[d]) * key(d);
          return s / std::sqrt(double(kDD));
        };
        for (int p = lo; p < L; ++p) {
          const size_t at = (size_t(p % ring) * Hkv + kvh) * kDD;
          sc.push_back(dot([&](int d) { return double(bf16_float(kc[at + d])); }));
          ctx_v.emplace_back(from_bits(std::vector<bits16>(vc.begin() + at, vc.begin() + at + kDD)));
        }
        for (int i = 0; i < B; ++i) {
          const size_t at = (size_t(i) * Hkv + kvh) * kDD;
          sc.push_back(dot([&](int d) { return double(kb[at + d]); }));
          ctx_v.emplace_back(vb.begin() + at, vb.begin() + at + kDD);
        }
        const double mx = *std::max_element(sc.begin(), sc.end());
        double den = 0;
        for (auto& s : sc) den += (s = std::exp(s - mx));
        for (int d = 0; d < kDD; ++d) {
          double o = 0;
          for (size_t i = 0; i < sc.size(); ++i) o += sc[i] * ctx_v[i][d];
          ref[(size_t(j) * Hq + hq) * kDD + d] = o / den;
        }
      }
    float* dq = dev(q);
    gpu::bf16* dkc = dev_bf16(kc);
    gpu::bf16* dvc = dev_bf16(vc);
    float* dkb = dev(kb);
    float* dvb = dev(vb);
    std::vector<float> first;
    for (int keys : {0, 1, 2, 4, 8}) {
      K::set_draft_attention_keys(keys);
      auto* out = dev_zero<float>(ref.size());
      K::draft_attention(dq, dkc, dvc, ring, dkb, dvb, L, B, window_left, Hq, Hkv, out, nullptr);
      const auto got = download(out, ref.size());
      const std::string name = "draft_attention_keys" + std::to_string(keys);
      check(name.c_str(), got, err_of(got, ref), 2e-5);
      if (keys == 0) first = got;
    }
    K::set_draft_attention_keys(0);

    SeqBatch b;  // one sequence per launch row block: the batched form on one sequence equals the single launch
    b.n = 1;
    b.row0[0] = 0, b.row0[1] = B;
    b.pos[0] = L;
    SeqPtrs<const gpu::bf16> kcs, vcs;
    kcs.p[0] = dkc, vcs.p[0] = dvc;
    auto* out = dev_zero<float>(ref.size());
    K::draft_attention_seqs(dq, kcs, vcs, ring, dkb, dvb, b, window_left, Hq, Hkv, out, nullptr);
    check_same("draft_attention_seqs", download(out, ref.size()), first);
  }
  {
    const int rows = 6, C = 64, groups = 4, block = 3;
    const auto h = r.vec(size_t(rows) * C, 2.f), coef = r.vec(size_t(rows) * 4 * groups, 1.f);
    const auto base = to_bits(r.vec(size_t(4) * C, 1.f));
    for (int side = 0; side < 2; ++side) {
      auto* out = dev_zero<float>(size_t(rows) * C);
      K::grouped_conv(dev(h), dev(coef), dev_bf16(base), side, out, rows, C, groups, block, nullptr);
      std::vector<double> ref(size_t(rows) * C);
      for (int n = 0; n < rows; ++n)
        for (int c = 0; c < C; ++c) {
          const float* cf = coef.data() + size_t(n) * 4 * groups + side * 2 * groups;
          const int grp = c / (C / groups);
          double o = (double(bf16_float(base[size_t(side * 2) * C + c])) + cf[grp]) * h[size_t(n) * C + c];
          if (n % block >= 1)
            o += (double(bf16_float(base[size_t(side * 2 + 1) * C + c])) + cf[groups + grp]) * h[size_t(n - 1) * C + c];
          ref[size_t(n) * C + c] = o;
        }
      const auto got = download(out, ref.size());
      check(side ? "grouped_conv_side1" : "grouped_conv_side0", got, err_of(got, ref), 2e-6);
    }
  }
  {
    const int E = 3, V = 20, R = 256, anchor = 7;
    const auto hp = r.vec(size_t(E) * R, 1.f), unary = r.vec(size_t(E) * 16, 1.f);
    std::vector<int> cand(size_t(E) * 16);
    for (auto& c : cand) c = r.below(V);
    const auto P = to_bits(r.vec(size_t(V) * R, 1.f)), S = to_bits(r.vec(size_t(V) * R, 1.f));
    auto* scores = dev_zero<float>(size_t(E) * 256);
    K::selector_lattice(dev(hp), dev(cand), dev(unary), dev_bf16(P), dev_bf16(S), anchor, E, scores, nullptr);
    std::vector<double> ref(size_t(E) * 256);
    for (int e = 0; e < E; ++e)
      for (int p = 0; p < 16; ++p)
        for (int c = 0; c < 16; ++c) {
          const int pid = e == 0 ? anchor : cand[size_t(e - 1) * 16 + p], cid = cand[size_t(e) * 16 + c];
          double s = unary[size_t(e) * 16 + c];
          for (int k = 0; k < R; ++k)
            s += double(bf16_float(P[size_t(pid) * R + k])) * hp[size_t(e) * R + k] * bf16_float(S[size_t(cid) * R + k]);
          ref[(size_t(e) * 16 + p) * 16 + c] = s;
        }
    const auto got = download(scores, ref.size());
    check("selector_lattice", got, err_of(got, ref), 2e-5);
    std::vector<int> anc{anchor};
    auto* scores2 = dev_zero<float>(ref.size());
    K::selector_lattice(dev(hp), dev(cand), dev(unary), dev_bf16(P), dev_bf16(S), 0, E, scores2, nullptr, dev(anc));
    check_same("selector_lattice_anchor_dev", download(scores2, ref.size()), got);
  }
  {
    const int rows = 3, H = 300, dst_stride = 320;
    const auto src = r.vec(size_t(rows) * H, 4.f);
    std::vector<bits16> dst(size_t(rows) * dst_stride);
    for (auto& x : dst) x = static_cast<bits16>(r.g());
    bits16* dd = dev(dst);
    K::copy_rows_bf16(dev(src), rows, H, as_bf16(dd), dst_stride, nullptr);
    for (int m = 0; m < rows; ++m)
      for (int h = 0; h < H; ++h) dst[size_t(m) * dst_stride + h] = bf16_bits(src[size_t(m) * H + h]);
    const auto got = download(dd, dst.size());
    check("copy_rows_bf16", got, mismatches(got, dst), 0);
  }
  for (const auto& [K_, name] : {std::pair{4096, "bf16_to_half_rows_4096"}, std::pair{1000, "bf16_to_half_rows_1000"}}) {
    // Each row scaled by a power of two so its largest magnitude lands in [2^14, 2^15), then rounded to FP16; xinv
    // undoes the scale exactly. Row 2 is zero (scale 2^15).
    const int M = 3, Kd = K_;
    auto xf = r.vec(size_t(M) * Kd, 3.f);
    for (int k = 0; k < Kd; ++k) xf[size_t(2) * Kd + k] = 0.f;
    const auto x = to_bits(xf);
    auto* out = dev_zero<bits16>(x.size());
    auto* xinv = dev_zero<float>(M);
    K::bf16_to_half_rows(dev_bf16(x), M, Kd, reinterpret_cast<gpu::half*>(out), xinv, nullptr);
    const auto h = download(out, x.size());
    const auto xi = download(xinv, M);
    std::vector<float> back(x.size()), xi_ref(M);
    std::vector<double> ref(x.size());
    for (int m = 0; m < M; ++m) {
      float mx = 0.f;
      for (int k = 0; k < Kd; ++k) mx = std::max(mx, std::fabs(bf16_float(x[size_t(m) * Kd + k])));
      int e = 0;
      if (mx > 0.f) std::frexp(mx, &e);
      xi_ref[m] = std::ldexp(1.f, e - 15);
      for (int k = 0; k < Kd; ++k) {
        const size_t i = size_t(m) * Kd + k;
        const uint16_t b = h[i];  // FP16 decode
        const int ex = (b >> 10) & 0x1f, man = b & 0x3ff;
        const double mag = ex ? std::ldexp(1.0 + man / 1024.0, ex - 15) : std::ldexp(man / 1024.0, -14);
        back[i] = static_cast<float>(((b & 0x8000) ? -mag : mag) * xi[m]);
        ref[i] = bf16_float(x[i]);
      }
    }
    check(name, h, err_of(back, ref) + mismatches(xi, xi_ref), 5e-4);  // FP16's half ulp is 2^-11 relative
  }
}

// ---- the sparse head's selection -------------------------------------------------------------------------------------

void test_sparse_head(Rng& r) {
  // Small integers throughout: every product and sum is exact in FP32 and bf16, whatever the order.
  const int M = 3, N = 3000, R = 256;
  std::vector<float> p(size_t(M) * R);
  for (auto& x : p) x = static_cast<float>(r.below(17) - 8);
  std::vector<float> usf(size_t(N) * R);
  for (auto& x : usf) x = static_cast<float>(r.below(17) - 8);
  const auto us = to_bits(usf);
  std::vector<uint8_t> m0(N, 0), m1(N, 0);
  for (int i = 0; i < 25; ++i) m0[r.below(N)] = 1, m1[r.below(N)] = 1;
  std::vector<double> approx(N);
  for (int n = 0; n < N; ++n) {
    double best = -INFINITY;
    for (int m = 0; m < M; ++m) {
      double s = 0;
      for (int k = 0; k < R; ++k) s += double(p[size_t(m) * R + k]) * usf[size_t(n) * R + k];
      best = std::max(best, s);
    }
    approx[n] = best;
  }
  float* dp = dev(p);
  gpu::bf16* dus = dev_bf16(us);
  ragon::kernels::SelMasks masks{};
  masks.m[0] = dev(m0);
  masks.m[1] = dev(m1);
  masks.n = 2;
  void* scratch = dev_zero<uint8_t>(K::sparse_head_scratch_bytes(N));
  for (const auto& [dynamic, cap, name] : {std::tuple{40, 200, "sparse_head_select"}, std::tuple{300, 120, "sparse_head_select_cap"}}) {
    auto* rows = dev_zero<int>(N);
    auto* nrows = dev_zero<int>(1);
    K::sparse_head_select(dp, M, dus, N, R, masks, dynamic, cap, scratch, rows, nrows, nullptr);
    std::vector<double> unmarked;
    for (int n = 0; n < N; ++n)
      if (!m0[n] && !m1[n]) unmarked.push_back(approx[n]);
    std::sort(unmarked.begin(), unmarked.end(), std::greater<double>());
    const double thr = unmarked[dynamic - 1];
    std::vector<int> want;
    for (int n = 0; n < N; ++n)
      if (m0[n] || m1[n] || approx[n] >= thr) want.push_back(n);
    if (static_cast<int>(want.size()) > cap) want.resize(cap);
    const int got_n = download(nrows, 1)[0];
    auto got = download(rows, std::max(0, std::min(got_n, N)));
    got.push_back(got_n);
    want.push_back(static_cast<int>(want.size()));
    check(name, got, mismatches(got, want), 0);
  }
  {
    const std::vector<int> rows{5, 9, 13, 2000, 2999, 7};
    std::vector<int> ids{3, 0, 5, 1};
    int* dids = dev(ids);
    K::sparse_head_ids(dev(rows), dids, 4, nullptr);
    for (auto& i : ids) i = rows[i];
    const auto got = download(dids, 4);
    check("sparse_head_ids", got, mismatches(got, ids), 0);
    std::vector<uint8_t> mask(N, 0);
    uint8_t* dm = dev(mask);
    K::sparse_head_mark(dm, dids, 4, nullptr);
    for (int i : ids) mask[i] = 1;
    const auto gm = download(dm, N);
    check("sparse_head_mark", gm, mismatches(gm, mask), 0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
#if defined(_WIN32)
    if (fopen_s(&g_dump, argv[1], "wb") != 0) g_dump = nullptr;
#else
    g_dump = std::fopen(argv[1], "wb");
#endif
  }
  if (argc > 1 && !g_dump) {
    std::fprintf(stderr, "w0_tests: cannot write %s\n", argv[1]);
    return 2;
  }
  Rng r(20261114u);
  test_primitives(r);
  test_conversions(r);
  test_elementwise(r);
  test_gemv(r);
  test_gdn(r);
  test_attn_prepare(r);
  test_kv_moves(r);
  test_drafter(r);
  test_sparse_head(r);
  for (void* p : g_live) ragon::test::free_bytes(p);
  if (g_dump) std::fclose(g_dump);
  std::printf("w0_tests: %s (%d failed)\n", g_fail ? "FAIL" : "ok", g_fail);
  return g_fail ? 1 : 0;
}
