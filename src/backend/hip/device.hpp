// The HIP backend's device helpers (BACKEND §3.1, §7): the CUDA spellings HIP lacks, with the same results.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include "backend/hip/runtime.hpp"  // <hip/hip_runtime.h> and the runtime calls (Error, memset_async, copy_d2d_async: §5.2)

#include <cstdint>

#include "backend/hip/types.hpp"

namespace ragon::gpu {

using bf16x2 = __hip_bfloat162;  // two bfloat16 values, low first
using fp8_e4m3 = __hip_fp8_e4m3;  // OCP FP8 E4M3, convertible to float on the host and the device

// A float to bf16, rounded to nearest even (on the host and the device; HIP's __float2bfloat16 rounds so).
__host__ __device__ __forceinline__ bf16 bf16_from_float(float v) { return __float2bfloat16(v); }

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
// A float to OCP FP8 E4M3, rounded to nearest even and saturated to +-448 (infinity too), NaN to 0x7f:
// the result of CUDA's __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3). Written out because HIP's
// software conversion on gfx1151 encodes magnitudes far below the smallest subnormal (2^-9) as 0x01,
// where the rounding gives zero.
__host__ __device__ __forceinline__ uint8_t e4m3_from_float(float v) {
  uint32_t u;
  __builtin_memcpy(&u, &v, 4);
  const uint8_t sign = static_cast<uint8_t>((u >> 24) & 0x80);
  const uint32_t a = u & 0x7fffffffu;
  if (a > 0x7f800000u) return 0x7f;            // NaN
  if (a >= 0x43e80000u) return sign | 0x7e;    // 464 and above (infinity too) saturate to 448
  if (a < 0x3c800000u) {                       // below 2^-6: subnormal codes m * 2^-9, m = 0..8
    float m;
    __builtin_memcpy(&m, &a, 4);
    return sign | static_cast<uint8_t>(__builtin_rintf(m * 512.0f));  // exact scaling; rint rounds to even
  }
  const uint32_t r = a + 0x7ffffu + ((a >> 20) & 1u);  // round the mantissa to 3 bits, ties to even
  const uint32_t code = (r >> 20) - (120u << 3);       // rebias the exponent from 127 to 7
  return sign | static_cast<uint8_t>(code > 0x7eu ? 0x7eu : code);
}

// x * y rounded to nearest, never fused into an FMA (the multiply carries no contract flag).
__device__ __forceinline__ float mul_rn(float x, float y) {
#pragma clang fp contract(off)
  return x * y;
}

// No L2 prefetch instruction is used on HIP: the hint is dropped (results unchanged).
__device__ __forceinline__ void prefetch_l2(const void*) {}

// Launch sizing for the kernels' own files (the runtime calls are in backend/hip/runtime.hpp, §5.2).
// Device 0's multiprocessor count into *n; the blocks of `kernel` that fit on one at once into *per.
inline Error sm_count(int* n) { return hipDeviceGetAttribute(n, hipDeviceAttributeMultiprocessorCount, 0); }
template <typename K>
inline Error blocks_per_sm(int* per, K kernel, int threads, size_t smem) {
  return hipOccupancyMaxActiveBlocksPerMultiprocessor(per, kernel, threads, smem);
}

}  // namespace ragon::gpu
