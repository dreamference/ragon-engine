// The CUDA backend's device helpers (BACKEND §3.1): one-line forwards to the CUDA spellings the kernels used before,
// inlined, so the CUDA objects keep the same code (§6).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "backend/cuda/runtime.hpp"  // <cuda_runtime.h> and the runtime calls (Error, memset_async, copy_d2d_async: §5.2)

#include <cstdint>

#include "backend/cuda/types.hpp"

namespace ragon::gpu {

using bf16x2 = __nv_bfloat162;  // two bfloat16 values, low first

// Two floats to a bf16x2, each rounded to nearest even.
__device__ __forceinline__ bf16x2 floats_to_bf16x2(float lo, float hi) { return __floats2bfloat162_rn(lo, hi); }

// OCP FP8 E4M3 (the checkpoints' and the KV mirror's format), decoded exactly.
__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}
// Two E4M3 values (low byte first) to two floats, exactly.
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t pair) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}
// A float to E4M3, round to nearest even, saturated to the largest finite value (+-448); NaN stays NaN.
__host__ __device__ __forceinline__ uint8_t e4m3_from_float(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// x * y rounded to nearest, opaque to the compiler: never fused into an FMA and never kept across a loop (recomputed).
__device__ __forceinline__ float mul_rn(float x, float y) {
  float r;
  asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y));
  return r;
}

// A hint that the line at p is read soon (L2). No effect on results.
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// Launch sizing for the kernels' own files (the runtime calls are in backend/cuda/runtime.hpp, §5.2).
// Device 0's multiprocessor count into *n; the blocks of `kernel` that fit on one at once into *per.
inline Error sm_count(int* n) { return cudaDeviceGetAttribute(n, cudaDevAttrMultiProcessorCount, 0); }
template <typename K>
inline Error blocks_per_sm(int* per, K kernel, int threads, size_t smem) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(per, kernel, threads, smem);
}

}  // namespace ragon::gpu
