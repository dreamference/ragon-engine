// The HIP backend's device helpers (BACKEND §3.1, §7): the CUDA spellings HIP lacks, with the same results.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "backend/hip/types.hpp"

namespace ragon::gpu {

using bf16x2 = __hip_bfloat162;  // two bfloat16 values, low first

// Two floats to a bf16x2, each rounded to nearest even (HIP has no __floats2bfloat162_rn).
__device__ __forceinline__ bf16x2 floats_to_bf16x2(float lo, float hi) {
  return __float22bfloat162_rn(make_float2(lo, hi));
}

// OCP FP8 E4M3 through HIP's type (software conversion on gfx1151, exact: every E4M3 value is a float).
__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __hip_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t pair) {
  return make_float2(e4m3_to_float(static_cast<uint8_t>(pair & 0xff)), e4m3_to_float(static_cast<uint8_t>(pair >> 8)));
}
__host__ __device__ __forceinline__ uint8_t e4m3_from_float(float v) {
  return __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
}

// x * y rounded to nearest, never fused into an FMA (the multiply carries no contract flag).
__device__ __forceinline__ float mul_rn(float x, float y) {
#pragma clang fp contract(off)
  return x * y;
}

// No L2 prefetch instruction is used on HIP: the hint is dropped (results unchanged).
__device__ __forceinline__ void prefetch_l2(const void*) {}

using Error = hipError_t;
inline constexpr Error kSuccess = hipSuccess;
inline Error last_error() { return hipGetLastError(); }
inline const char* error_string(Error e) { return hipGetErrorString(e); }
inline Error memset_async(void* p, int v, size_t bytes, Stream s) { return hipMemsetAsync(p, v, bytes, s); }
inline Error copy_d2d_async(void* dst, const void* src, size_t bytes, Stream s) {
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, s);
}
// Device 0's multiprocessor count into *n; the blocks of `kernel` that fit on one at once into *per.
inline Error sm_count(int* n) { return hipDeviceGetAttribute(n, hipDeviceAttributeMultiprocessorCount, 0); }
template <typename K>
inline Error blocks_per_sm(int* per, K kernel, int threads, size_t smem) {
  return hipOccupancyMaxActiveBlocksPerMultiprocessor(per, kernel, threads, smem);
}

}  // namespace ragon::gpu
