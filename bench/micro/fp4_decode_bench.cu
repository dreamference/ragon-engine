// fp4_decode_bench: can decode use the FP4 tensor cores? The engine's GEMM groups at a sweep of row counts, on the two
// paths the engine has, through the engine's own entry points:
//   fp16  to_half_rows + stream_gemm_multi: decode today. Weights unpacked to FP16 in registers, FP16 MMA, activations
//         FP16 (W4A16). At most 32 rows a launch (kMaxStreamRows), so M > 32 is ceil(M / 32) launches, each reading
//         the weights again.
//   fp4   quant_act_* + prefill_gemm_*: the prefill path. Block-scaled MMA over NVFP4 or FP8 weights and activations
//         quantized to the same format (W4A4, W8A8). Blocks of 128 rows, so M > 128 reads the weights ceil(M / 128)
//         times. The name is the owner's question's; the FP8 groups use the FP8 tensor core.
// Every call reads the next of enough copies of the weights to defeat the L2, as a pass does. Timing data is random;
// the numerics columns say what changing the path would do, not what a real layer's activations would lose:
//   relerr    ||y_fp4 - y_fp16|| / ||y_fp16||, first target of the group, the same x
//   rowdiff   floats of the fp4 path's first M rows that differ, bit for bit, from the same rows of a run at the
//             largest M. 0 means the path is row-invariant, which is what a wide draft tree needs of a target.
//
// Usage: fp4_decode_bench [ROWS=1,8,16,32,64,128,256] [REPS]  (REPS scales the iterations; default 1)
// One CSV line per (group, rows, arm); bus_gbps counts every read of the weights (so fp4 past 128 rows overstates it:
// its blocks share one read through the L2), TFLOPS = 2 M N K / ms. Then '#' lines: the step (each group times its
// count in a pass) at each M, and two crossovers, the smallest M where fp4 is the faster: against fp16 as the engine
// launches it, which at M > 32 only says what a second read costs; and against `bound`, the strongest FP16 kernel the
// machine allows, which no kernel here reaches: each group at the larger of its one-read time at fp16's bus rate (the
// first M asked for) and its FLOPs at the measured rate of FP16 MMA with no load at all. The bound leaves out bringing
// x to the tensor cores, which is what limits a wide FP16 kernel, so the real crossover lies between.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.cuh"

namespace {

namespace rk = ragon::kernels;

// ---- The rate of FP16 MMA with nothing to load: what no FP16 GEMM can exceed ----------------------------------
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %4, %4, %4}, {%5, %5}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a), "r"(b));
}

template <int NT>
__global__ void __launch_bounds__(256) fp16_mma_rate_kernel(uint32_t one, int steps, float* out) {
  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  for (int s = 0; s < steps; ++s) {
#pragma unroll
    for (int j = 0; j < NT; ++j) mma16816(acc[j], one, one);
  }
  // Every tensor stays live: the assembler drops an MMA whose result nothing reads, even in volatile asm.
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < NT; ++j) sum += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = sum;
}

// TFLOPS of FP16 MMA with NT tensors in registers, as a kernel of 256 threads holds them.
template <int NT>
double fp16_mma_rate() {
  constexpr int kBlocks = 4096, kInstr = 16000;
  float* out;
  cudaMalloc(&out, sizeof(float));
  const auto run = [&] { fp16_mma_rate_kernel<NT><<<kBlocks, 256>>>(0x3c003c00u, kInstr / NT, out); };
  run();
  cudaDeviceSynchronize();
  double best = 1e30;
  for (int pass = 0; pass < 3; ++pass) {
    const auto t0 = std::chrono::steady_clock::now();
    run();
    cudaDeviceSynchronize();
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  cudaFree(out);
  return double(kBlocks) * 8 * (kInstr / NT) * NT * 4096.0 / best / 1e12;
}

constexpr int kStreamRows = rk::kMaxStreamRows;
constexpr int kPrefillRows = 128;
constexpr size_t kCopiesBytes = size_t(1024) << 20;

struct Group {
  const char* name;
  bool fp4;
  int K, n, count;
  int N[3];
};

// The groups of a Qwen3.8-27B step, as tests/stream_tests.cu's bench_groups.
const Group kGroups[] = {
    {"attn_qkv", false, 5120, 3, 16, {12288, 1024, 1024}}, {"attn_o", false, 6144, 1, 16, {5120}},
    {"gdn_in", false, 5120, 2, 48, {10240, 6144}},          {"gdn_out", false, 6144, 1, 48, {5120}},
    {"ffn_gate_up", true, 5120, 2, 64, {17408, 17408}},     {"ffn_down", true, 17408, 1, 64, {5120}},
    {"lm_head", true, 5120, 1, 1, {248320}},
};

// The activation scales of tests/prefill_tests.cu, for x of the same distribution.
constexpr float kIn4 = 0.0019763766f, kIn8 = 0.2053571f;

void check(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "fp4_decode_bench: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(2);
  }
}

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) {
    std::fprintf(stderr, "fp4_decode_bench: cannot allocate %zu MB\n", n * sizeof(T) >> 20);
    std::exit(2);
  }
  return p;
}

// xorshift64*: gigabytes of bytes, where the quality of a Mersenne twister buys nothing.
void fill_bytes(std::vector<uint8_t>& v, uint64_t seed, bool no_fp8_nan) {
  uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  for (size_t i = 0; i < v.size(); i += 8) {
    s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
    uint64_t r = s * 0x2545F4914F6CDD1Dull;
    for (size_t j = i; j < std::min(i + 8, v.size()); ++j, r >>= 8) {
      uint8_t b = static_cast<uint8_t>(r);
      if (no_fp8_nan && (b & 0x7f) == 0x7f) b = 0x7e;
      v[j] = b;
    }
  }
}

// `copies` tiled copies of a random matrix, one after another.
uint8_t* make_weight(bool fp4, int N, int K, int copies, uint64_t seed, size_t* bytes) {
  std::vector<uint8_t> w(fp4 ? size_t(N) * K / 2 : size_t(N) * K);
  fill_bytes(w, seed, !fp4);
  uint8_t* dw = dev_alloc<uint8_t>(w.size());
  cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
  *bytes = fp4 ? rk::tiled_bytes_nvfp4(N, K) : rk::tiled_bytes_fp8(N, K);
  uint8_t* out = dev_alloc<uint8_t>(*bytes * copies);
  if (fp4) {
    std::vector<uint8_t> s(size_t(N) * K / 16);
    fill_bytes(s, seed + 1, false);
    for (auto& b : s) b = static_cast<uint8_t>(0x28 + b % 25);  // E4M3 scales around 1
    uint8_t* ds = dev_alloc<uint8_t>(s.size());
    cudaMemcpy(ds, s.data(), s.size(), cudaMemcpyHostToDevice);
    rk::retile_nvfp4(dw, ds, out, N, K, nullptr);
    cudaDeviceSynchronize();
    cudaFree(ds);
  } else {
    rk::retile_fp8(dw, out, N, K, nullptr);
    cudaDeviceSynchronize();
  }
  cudaFree(dw);
  for (int c = 1; c < copies; ++c) cudaMemcpy(out + *bytes * c, out, *bytes, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  check("weights");
  return out;
}

// Milliseconds per iteration of f(it): one warm-up pass, then the better of two.
template <class F>
double time_ms(int iters, F&& f) {
  double best = 1e30;
  for (int pass = 0; pass < 3; ++pass) {
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) f(it);
    cudaDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (pass > 0) best = std::min(best, ms / iters);
  }
  check("timed section");
  return best;
}

struct Cell {
  double gemm = 0, prep = 0;
  double total() const { return gemm + prep; }
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> Ms;
  for (const char* p = argc > 1 ? argv[1] : "1,8,16,32,64,128,256"; *p;) {
    Ms.push_back(std::atoi(p));
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  const int scale = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;
  const int Mmax = *std::max_element(Ms.begin(), Ms.end());
  const int nM = static_cast<int>(Ms.size());
  std::vector<Cell> step_a(nM), step_b(nM);
  std::vector<double> step_bound(nM, 0.0);
  const double rate8 = fp16_mma_rate<8>(), rate32 = fp16_mma_rate<32>(), rate = std::max(rate8, rate32);
  std::printf("# FP16 MMA with nothing to load: %.1f TFLOPS at 8 tensors in registers, %.1f at 32\n", rate8, rate32);
  double step_gb = 0;
  std::printf("group,fmt,N,K,count,rows,arm,reads,bytes,ms_gemm,ms_prep,bus_gbps,tflops,relerr,rowdiff\n");
  std::mt19937 rng(1234);
  uint64_t seed = 7;
  for (const Group& g : kGroups) {
    size_t sizes[3], bytes = 0;
    {
      // Sizes first: the copies are as many as a pass's worth of the L2 needs, per group.
      for (int i = 0; i < g.n; ++i)
        bytes += sizes[i] = g.fp4 ? rk::tiled_bytes_nvfp4(g.N[i], g.K) : rk::tiled_bytes_fp8(g.N[i], g.K);
    }
    const int copies = int(std::max<size_t>(2, kCopiesBytes / bytes));
    uint8_t* w[3];
    for (int i = 0; i < g.n; ++i) w[i] = make_weight(g.fp4, g.N[i], g.K, copies, seed++, &sizes[i]);
    step_gb += double(bytes) * g.count / 1e9;

    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx(size_t(Mmax) * g.K);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = nd(rng) * (i % 97 == 0 ? 40.f : 1.f) * 0.003f;
    float* x = dev_alloc<float>(hx.size());
    cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice);
    __half* xh = dev_alloc<__half>(hx.size());
    float* xinv = dev_alloc<float>(Mmax);
    uint8_t* q = dev_alloc<uint8_t>(g.fp4 ? size_t(Mmax) * g.K / 2 : size_t(Mmax) * g.K);
    uint8_t* sf = g.fp4 ? dev_alloc<uint8_t>(size_t(Mmax) * g.K / 16) : nullptr;
    float* y[3];
    for (int i = 0; i < g.n; ++i) y[i] = dev_alloc<float>(size_t(Mmax) * g.N[i]);

    const float in = g.fp4 ? kIn4 : kIn8;
    const auto quant = [&](int M) {
      if (g.fp4) rk::quant_act_nvfp4(x, M, g.K, in, q, sf, nullptr);
      else rk::quant_act_fp8(x, M, g.K, in, q, nullptr);
    };
    const auto run_a = [&](int M, int copy) {
      for (int m = 0; m < M; m += kStreamRows) {
        const int rows = std::min(kStreamRows, M - m);
        rk::StreamTarget t[3];
        for (int i = 0; i < g.n; ++i) t[i] = {w[i] + sizes[i] * copy, nullptr, 1.f, y[i] + size_t(m) * g.N[i], g.N[i]};
        rk::stream_gemm_multi(g.fp4, xh + size_t(m) * g.K, xinv + m, rows, t, g.n, g.K, nullptr);
      }
    };
    const auto run_b = [&](int M, int copy) {
      for (int i = 0; i < g.n; ++i) {
        if (g.fp4)
          rk::prefill_gemm_nvfp4(q, sf, M, w[i] + sizes[i] * copy, in, y[i], g.N[i], g.N[i], g.K, false, nullptr);
        else rk::prefill_gemm_fp8(q, M, w[i] + sizes[i] * copy, in, y[i], g.N[i], g.N[i], g.K, false, nullptr);
      }
    };

    // The largest M's rows, the reference of row invariance.
    rk::to_half_rows(x, Mmax, g.K, xh, xinv, nullptr);
    quant(Mmax);
    run_b(Mmax, 0);
    cudaDeviceSynchronize();
    std::vector<float> ref(size_t(Mmax) * g.N[0]), ya(ref.size()), yb(ref.size());
    cudaMemcpy(ref.data(), y[0], ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
    check("reference");

    size_t flops_nk = 0;
    for (int i = 0; i < g.n; ++i) flops_nk += size_t(g.N[i]) * g.K;

    double one_read_ms = 0;
    for (int mi = 0; mi < nM; ++mi) {
      const int M = Ms[mi];
      const int iters = std::max(40, copies * 4) * scale;
      Cell a, b;
      a.prep = time_ms(iters, [&](int) { rk::to_half_rows(x, M, g.K, xh, xinv, nullptr); });
      a.gemm = time_ms(iters, [&](int it) { run_a(M, it % copies); });
      b.prep = time_ms(iters, [&](int) { quant(M); });
      b.gemm = time_ms(iters, [&](int it) { run_b(M, it % copies); });

      // Numerics: the same x and the same copy through both paths.
      const size_t floats = size_t(M) * g.N[0];
      run_a(M, 0);
      cudaDeviceSynchronize();
      cudaMemcpy(ya.data(), y[0], floats * sizeof(float), cudaMemcpyDeviceToHost);
      run_b(M, 0);
      cudaDeviceSynchronize();
      cudaMemcpy(yb.data(), y[0], floats * sizeof(float), cudaMemcpyDeviceToHost);
      check("numerics");
      double dd = 0, nn = 0;
      size_t rowdiff = 0;
      for (size_t i = 0; i < floats; ++i) {
        const double d = double(yb[i]) - double(ya[i]);
        dd += d * d, nn += double(ya[i]) * double(ya[i]);
        rowdiff += yb[i] != ref[i] && !(std::isnan(yb[i]) && std::isnan(ref[i]));
      }

      const int reads_a = (M + kStreamRows - 1) / kStreamRows, reads_b = (M + kPrefillRows - 1) / kPrefillRows;
      const auto line = [&](const char* arm, int reads, const Cell& c, bool nums) {
        std::printf("%s,%s,%d,%d,%d,%d,%s,%d,%zu,%.4f,%.4f,%.1f,%.1f,", g.name, g.fp4 ? "nvfp4" : "fp8", g.N[0], g.K,
                    g.count, M, arm, reads, bytes, c.gemm, c.prep, double(bytes) * reads / (c.gemm * 1e-3) / 1e9,
                    2.0 * M * double(flops_nk) / (c.gemm * 1e-3) / 1e12);
        if (nums) std::printf("%.4f,%zu\n", std::sqrt(dd / std::max(nn, 1e-30)), rowdiff);
        else std::printf(",\n");
      };
      line("fp16", reads_a, a, false);
      line("fp4", reads_b, b, true);
      std::fflush(stdout);
      step_a[mi].gemm += a.gemm * g.count, step_a[mi].prep += a.prep * g.count;
      step_b[mi].gemm += b.gemm * g.count, step_b[mi].prep += b.prep * g.count;
      if (mi == 0) one_read_ms = a.gemm;
      step_bound[mi] += (std::max(one_read_ms, 2.0 * M * double(flops_nk) / (rate * 1e12) * 1e3) + a.prep) * g.count;
    }

    for (int i = 0; i < g.n; ++i) cudaFree(w[i]), cudaFree(y[i]);
    cudaFree(x), cudaFree(xh), cudaFree(xinv), cudaFree(q);
    if (sf) cudaFree(sf);
  }

  std::printf("# step GEMMs, %.2f GB of weights a pass (each group times its count; preparation of x once per group)\n",
              step_gb);
  int crossover = 0, cross_bound = 0;
  for (int mi = 0; mi < nM; ++mi) {
    std::printf("# M=%-4d fp16 %8.2f ms (+%6.2f)  bound %8.2f ms  fp4 %8.2f ms (+%6.2f)\n", Ms[mi], step_a[mi].gemm,
                step_a[mi].prep, step_bound[mi], step_b[mi].gemm, step_b[mi].prep);
    if (!crossover && step_b[mi].total() < step_a[mi].total()) crossover = Ms[mi];
    if (!cross_bound && step_b[mi].total() < step_bound[mi]) cross_bound = Ms[mi];
  }
  std::printf("# crossover against fp16: %s%d\n", crossover ? "M=" : "none up to M=", crossover ? crossover : Mmax);
  std::printf("# crossover against the bound: %s%d\n", cross_bound ? "M=" : "none up to M=",
              cross_bound ? cross_bound : Mmax);
  return 0;
}
