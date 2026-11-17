// gemm_bf16 for the hip backend (W1, correctness first; gemm_bf16.cu is the cuda form with TMA and ldmatrix):
// y[M][N] = x[M][K] . W[N][K]^T, BF16 inputs (both K-contiguous), FP32 accumulation and output.
//
// A block of 8 waves covers 64 rows x 128 columns: the rows' activations staged in LDS 256 values of K at a time,
// each wave one 16-column tile with its weight fragments read straight from global memory, and 16x16x16 BF16 WMMAs
// over K in ascending order of 16 into one FP32 accumulator per output. No split-K, so a row's result does not
// depend on M or on the rows beside it. A launch takes a row slab sized so that its work stays far under the
// Windows display-timeout limit, whatever N and K are.
#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "backend/device.hpp"

namespace ragon::kernels {
namespace {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int v4u __attribute__((ext_vector_type(4)));

constexpr int kWaves = 8, kBR = 64, kKC = 256, kXSt = kKC + 16;
constexpr double kLaunchMacs = 2e11;  // multiply-adds per launch at most (a few tens of ms)

__global__ void __launch_bounds__(kWaves * 32)
    gemm_bf16_kernel(const short* __restrict__ x, int M, const short* __restrict__ w, float* __restrict__ y, int N,
                     int K) {
  __shared__ __attribute__((aligned(32))) short XS[kBR * kXSt];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, r16 = lane & 15;
  const int nt = blockIdx.x * kWaves + wave;
  const bool valid = nt < (N + 15) / 16;  // wave-uniform
  const int col = nt * 16 + r16;
  const bool colok = valid && col < N;
  const int m0 = blockIdx.y * kBR;
  const short* wr = w + static_cast<size_t>(colok ? col : 0) * K;
  v8f acc[4] = {};
  for (int k0 = 0; k0 < K; k0 += kKC) {
    const int kc = min(kKC, K - k0), per_row = kc / 8;
    for (int i = tid; i < kBR * per_row; i += kWaves * 32) {
      const int r = i / per_row, c8 = i - r * per_row, row = m0 + r;
      v4u v = {0, 0, 0, 0};
      if (row < M) v = *reinterpret_cast<const v4u*>(x + static_cast<size_t>(row) * K + k0 + c8 * 8);
      *reinterpret_cast<v4u*>(XS + r * kXSt + c8 * 8) = v;
    }
    __syncthreads();
    if (valid) {
      for (int kk = 0; kk < kc; kk += 16) {
        v16s b = {};
        if (colok) b = *reinterpret_cast<const v16s*>(wr + k0 + kk);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          const v16s a = *reinterpret_cast<const v16s*>(XS + (g * 16 + r16) * kXSt + kk);
          acc[g] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[g]);
        }
      }
    }
    __syncthreads();
  }
  if (!colok) return;
  // acc[g][r]: row m0 + 16 g + 2 r + half, column col.
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int row = m0 + g * 16 + 2 * r + half;
      if (row < M) y[static_cast<size_t>(row) * N + col] = acc[g][r];
    }
}

}  // namespace

void gemm_bf16(const gpu::bf16* x, int M, const gpu::bf16* w, float* y, int N, int K, gpu::Stream s) {
  if (M < 1 || N < 1) return;
  if (K < 16 || K % 16 != 0) throw std::runtime_error("gemm_bf16: K must be a positive multiple of 16, got " + std::to_string(K));
  const double per_row = static_cast<double>(N) * K;
  const int slab = std::max(kBR, std::min(2048, static_cast<int>(kLaunchMacs / per_row) / kBR * kBR));
  const int nblk = (N + 16 * kWaves - 1) / (16 * kWaves);
  for (int r0 = 0; r0 < M; r0 += slab) {
    const int m = std::min(slab, M - r0);
    launch_kernel("gemm_bf16", gemm_bf16_kernel, dim3(nblk, (m + kBR - 1) / kBR), dim3(kWaves * 32), 0, s,
                  reinterpret_cast<const short*>(x) + static_cast<size_t>(r0) * K, m, reinterpret_cast<const short*>(w),
                  y + static_cast<size_t>(r0) * N, N, K);
  }
}

}  // namespace ragon::kernels
