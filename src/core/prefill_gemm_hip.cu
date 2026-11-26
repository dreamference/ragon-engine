// The prefill GEMMs and activation quantizers on HIP (gfx1151, wave32; BACKEND §7, W1): prefill_gemm.cu's entry
// points for the HIP build, compiled in its place. The CUDA kernels are built around sm_121's block-scaled
// mma.sync, ldmatrix and cp.async; this file keeps their arithmetic where it is a fixed recipe and replaces the
// tensor-core part with WMMA 16x16x16 FP16 into FP32.
//
// Quantizers (quant_act_*, rmsnorm_quant*, gated_rmsnorm_fp8*, sigmoid_mul_fp8, silu_mul_quant_nvfp4): the CUDA
// kernels line for line, with the two conversions written out: E2M1 (cvt.rn.satfinite.e2m1x2: nearest, ties to
// the even code, saturated to 6, the sign kept) and E4M3 (gpu::e4m3_from_float, __nv_cvt_float_to_fp8's
// satfinite rounding). The pure ones (quant_act_*) give CUDA's bytes; the fused ones take exp or rsqrt from the
// hardware, so they are held to the host reference's tolerance (DECISIONS 2026-11-14), and silu_mul_quant_nvfp4
// equals silu then quant_act_nvfp4 on the same backend.
//
// GEMMs: the activations (NVFP4 codes times their E4M3 scales, or E4M3) and the HIP tiled weights
// (stream_gemm_hip.cu's layout and decoders) decode to FP16 exactly, so every WMMA product is exact in FP32; one
// wave walks all of K for its 16 output columns in ascending order, rows past M are zeros in LDS, and a row's
// bytes do not depend on M, on the launch's row slab or on the run. The host entries launch at most kSlabRows
// rows per kernel, so no launch comes near the Windows TDR limit whatever the prompt length.
#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "backend/device.hpp"
#include "core/elem.cuh"
#include "core/lora_kernels.cuh"

namespace ragon::kernels {
namespace {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef _Float16 hf;
typedef hf v16h __attribute__((ext_vector_type(16)));
typedef hf v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef u32 v4u __attribute__((ext_vector_type(4)));
typedef u32 v8u __attribute__((ext_vector_type(8)));
typedef short v16s __attribute__((ext_vector_type(16)));

constexpr int kTileK = 64;           // K per weight tile
constexpr int kKC = 256;             // K per LDS chunk
constexpr int kXSt = kKC + 16;       // LDS row stride (halves)
constexpr int kBR = 64;              // rows per block (four 16-row WMMA groups)
constexpr int kWaves = 8;            // waves per block (16 in the NVFP4 GEMMs, see waves_of)
constexpr int kSlabRows = 2048;      // rows per launch (TDR)

// ---- Conversions ----

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// E2M1 of one value: nearest, ties to the even code, saturated to 6 (cvt.rn.satfinite.e2m1x2.f32); a negative
// value that rounds to zero keeps its sign. Grid 0, .5, 1, 1.5, 2, 3, 4, 6; the midpoints .25, 1.25, 2.5, 5 go
// down (to the even codes 0, 2, 4, 6) and .75, 1.75, 3.5 go up (to 2, 4, 6).
__device__ __forceinline__ u32 e2m1(float x) {
  const u32 sign = (__float_as_uint(x) >> 28) & 8u;
  const float a = fabsf(x);
  u32 c;
  if (a <= 0.25f) c = 0;
  else if (a < 0.75f) c = 1;
  else if (a <= 1.25f) c = 2;
  else if (a < 1.75f) c = 3;
  else if (a <= 2.5f) c = 4;
  else if (a < 3.5f) c = 5;
  else if (a <= 5.f) c = 6;
  else c = 7;  // above 5, infinity and NaN
  return sign | c;
}
// Two values to one byte, the first in the low nibble.
__device__ __forceinline__ u32 e2m1x2(float lo, float hi) { return e2m1(lo) | (e2m1(hi) << 4); }

// E4M3 byte to float, exact.
__device__ __forceinline__ float e4m3f(u32 b) {
  const u32 e = (b >> 3) & 15u, m = b & 7u;
  const float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : static_cast<float>(m) * 0.001953125f;
  return (b & 0x80u) ? -v : v;
}

// ---- Quantizers (prefill_gemm.cu's, with the conversions above) ----

// The NVFP4 encoding of four values held by each of four adjacent lanes (one block of 16).
__device__ __forceinline__ void put_nvfp4(const float (&v)[4], float gscale, size_t quad, bool act, uint8_t* __restrict__ q,
                                          uint8_t* __restrict__ sf) {
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  amax = fmaxf(amax, __shfl_xor_sync(gpu::kWarpMask, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(gpu::kWarpMask, amax, 2));
  const u32 sc = gpu::e4m3_from_float(gpu::mul_rn(gscale, gpu::mul_rn(amax, 1.f / 6.f)));
  const float sv = e4m3f(sc);
  const float os = sv != 0.f ? 1.f / gpu::mul_rn(sv, 1.f / gscale) : 0.f;
  if (!act) return;
  const u32 lo = e2m1x2(gpu::mul_rn(v[0], os), gpu::mul_rn(v[1], os));
  const u32 hi = e2m1x2(gpu::mul_rn(v[2], os), gpu::mul_rn(v[3], os));
  reinterpret_cast<uint16_t*>(q)[quad] = static_cast<uint16_t>(lo | (hi << 8));
  if ((quad & 3) == 0) sf[quad >> 2] = static_cast<uint8_t>(sc);
}

__device__ __forceinline__ u32 put_fp8x4(const float (&v)[4], float inv_scale) {
  u32 w = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) w |= static_cast<u32>(gpu::e4m3_from_float(gpu::mul_rn(v[k], inv_scale))) << (8 * k);
  return w;
}

// Grid-stride over quads (four values each); every lane of a wave runs the same iterations, so the four lanes of
// an NVFP4 block always shuffle together (the tail lanes past `quads` take part with zeros and store nothing).
template <bool SILU>
__global__ void quant_nvfp4_kernel(const float* __restrict__ x, const float* __restrict__ up, size_t quads, float gscale,
                                   uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const u32 lane = threadIdx.x & 31;
  for (size_t i0 = blockIdx.x * static_cast<size_t>(blockDim.x) + (threadIdx.x & ~31u); i0 < quads; i0 += stride) {
    const size_t i = i0 + lane;
    const bool act = i < quads;
    float4 f = act ? reinterpret_cast<const float4*>(x)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (SILU) {
      const float4 u = act ? reinterpret_cast<const float4*>(up)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
      f = make_float4(gpu::mul_rn(silu(f.x), u.x), gpu::mul_rn(silu(f.y), u.y), gpu::mul_rn(silu(f.z), u.z),
                      gpu::mul_rn(silu(f.w), u.w));
    }
    const float v[4] = {bf16_round(f.x), bf16_round(f.y), bf16_round(f.z), bf16_round(f.w)};
    put_nvfp4(v, gscale, i, act, q, sf);
  }
}

// RMSNorm of each row (x * rstd * (1 + w)) into the next GEMM's encoding, and into `out` when given. One block of
// 256 per row (eight waves); H % 1024 == 0, so every lane runs the same iterations.
template <bool FP4, typename OutT = float>
__global__ void __launch_bounds__(256)
    norm_quant_kernel(const float* __restrict__ x, const gpu::bf16* __restrict__ w, OutT* __restrict__ out, int H,
                      float eps, float scale, uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  __shared__ float red[8];
  const size_t r = blockIdx.x;
  const float4* row = reinterpret_cast<const float4*>(x + r * H);
  float ss = 0.f;
  for (int i = threadIdx.x; i < H / 4; i += 256) {
    const float4 f = row[i];
    ss = fmaf(f.x, f.x, fmaf(f.y, f.y, fmaf(f.z, f.z, fmaf(f.w, f.w, ss))));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(gpu::kWarpMask, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / H + eps);
  for (int i = threadIdx.x; i < H / 4; i += 256) {
    const float4 f = row[i];
    const float xs[4] = {f.x, f.y, f.z, f.w};
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = gpu::mul_rn(gpu::mul_rn(xs[k], rstd), 1.f + __bfloat162float(w[4 * i + k]));
    if (out) elem::store4(out + r * H, i, make_float4(v[0], v[1], v[2], v[3]));
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = bf16_round(v[k]);
    const size_t quad = r * (H / 4) + i;
    if constexpr (FP4) put_nvfp4(v, scale, quad, true, q, sf);
    else reinterpret_cast<u32*>(q)[quad] = put_fp8x4(v, scale);
  }
}

// The DeltaNet output norm (rmsnorm(x) * w * silu(z) per head of 128) into FP8: one wave per head row.
template <typename InT>
__global__ void gated_norm_fp8_kernel(const InT* __restrict__ x, const InT* __restrict__ z, const gpu::bf16* __restrict__ w,
                                      size_t rows, float eps, float inv_scale, uint8_t* __restrict__ q) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x / 32) + threadIdx.x / 32;
  if (r >= rows) return;  // whole waves
  const int lane = threadIdx.x & 31;
  const float4 f = elem::load4(x + r * 128, lane);
  const float4 g = elem::load4(z + r * 128, lane);
  float ss = fmaf(f.w, f.w, fmaf(f.z, f.z, fmaf(f.y, f.y, gpu::mul_rn(f.x, f.x))));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(gpu::kWarpMask, ss, o);
  const float rstd = rsqrtf(ss / 128 + eps);
  const float xs[4] = {f.x, f.y, f.z, f.w}, zs[4] = {g.x, g.y, g.z, g.w};
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    v[k] = bf16_round(gpu::mul_rn(gpu::mul_rn(gpu::mul_rn(xs[k], rstd), __bfloat162float(w[4 * lane + k])), silu(zs[k])));
  reinterpret_cast<u32*>(q)[r * 32 + lane] = put_fp8x4(v, inv_scale);
}

// x * sigmoid(gate) into FP8 (the attention output, for the o-projection).
__global__ void sigmoid_mul_fp8_kernel(const float* __restrict__ x, const float* __restrict__ gate, size_t quads,
                                       float inv_scale, uint8_t* __restrict__ q) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float4 f = reinterpret_cast<const float4*>(x)[i], g = reinterpret_cast<const float4*>(gate)[i];
    const float v[4] = {bf16_round(f.x / (1.f + __expf(-g.x))), bf16_round(f.y / (1.f + __expf(-g.y))),
                        bf16_round(f.z / (1.f + __expf(-g.z))), bf16_round(f.w / (1.f + __expf(-g.w)))};
    reinterpret_cast<u32*>(q)[i] = put_fp8x4(v, inv_scale);
  }
}

// Four values per thread: E4M3(bf16(x) / input_scale), saturated to +-448.
template <typename InT>
__global__ void quant_fp8_kernel(const InT* __restrict__ x, size_t quads, float inv_scale, uint8_t* __restrict__ q) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float4 f = elem::load4(x, i);
    const float v[4] = {bf16_round(f.x), bf16_round(f.y), bf16_round(f.z), bf16_round(f.w)};
    reinterpret_cast<u32*>(q)[i] = put_fp8x4(v, inv_scale);
  }
}

// ---- GEMMs ----

__device__ __forceinline__ u32 as_u(v2h x) { return __builtin_bit_cast(u32, x); }
__device__ __forceinline__ v2h as_h(u32 x) { return __builtin_bit_cast(v2h, x); }

// stream_gemm_hip.cu's decoders: 8 E2M1 codes (low nibble first) times a scale; 4 E4M3 bytes; one E4M3 scale.
__device__ __forceinline__ void dec_fp4(u32 c, v2h sc, v2h* o) {
  const u32 T0 = 0x3E3C3800u, T1 = 0x46444240u;  // 0, .5, 1, 1.5 | 2, 3, 4, 6
  const u32 lo = c & 0x07070707u, hi = (c >> 4) & 0x07070707u;
  const u32 hl = __builtin_amdgcn_perm(T1, T0, lo) | ((c & 0x08080808u) << 4);
  const u32 hh = __builtin_amdgcn_perm(T1, T0, hi) | (c & 0x80808080u);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const u32 d = __builtin_amdgcn_perm(hh, hl, ((4u + j) << 24) | 0x000C0000u | ((u32)j << 8) | 0x0Cu);
    o[j] = as_h(d) * sc;
  }
}
__device__ __forceinline__ void dec_fp8(u32 c, v2h* o) {
  const v2h k8 = {(hf)256.0f, (hf)256.0f};
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const u32 d = __builtin_amdgcn_perm(0u, c, 0x0C010C00u + j * 0x00020002u);
    const u32 r = ((d & 0x007F007Fu) << 7) | ((d & 0x00800080u) << 8);
    o[j] = as_h(r) * k8;
  }
}
__device__ __forceinline__ v2h e4m3_h2(u32 b) {
  const u32 r = ((b & 0x7Fu) << 7) | ((b & 0x80u) << 8);
  const hf v = __builtin_bit_cast(hf, (unsigned short)r) * (hf)256.0f;
  const v2h o = {v, v};
  return o;
}
__device__ __forceinline__ u32 xswap(u32 v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}

enum Mode { kPlain = 0, kBf16Out = 1, kSwiglu = 2 };

struct GemmArgs {
  const uint8_t* xq;  // activations: NVFP4 codes [M][K/2] or E4M3 [M][K]
  const uint8_t* xs;  // NVFP4 activation scales [M][K/16]
  const uint8_t* w;   // tiled weights (the gate in SwiGLU mode)
  const uint8_t* w2;  // SwiGLU: the up projection
  float alpha, alpha2, gscale_out;
  float* y;           // [M][ldy]
  gpu::bf16* yb;      // kBf16Out: [M][ldy]
  uint8_t* q_out;     // SwiGLU: NVFP4 [M][N/2] and scales [M][N/16]
  uint8_t* sf_out;
  int ldy, M, N, K;
  bool accumulate;
};

// The raw bytes of one lane's share of weight tile `tile` (FP4: 16 code bytes and two scales; FP8: 32 bytes).
template <bool FP4> struct WRaw;
template <> struct WRaw<true> { v4u c; u32 s2; };
template <> struct WRaw<false> { v4u c0, c1; };

template <bool FP4>
__device__ __forceinline__ void weight_load(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, u64 tile,
                                            u32 r16, u32 half, WRaw<FP4>& r) {
  if constexpr (FP4) {
    r.c = *(const v4u*)(w + tile * 512 + r16 * 32 + half * 16);
    r.s2 = *(const unsigned short*)(sc + tile * 64 + r16 * 4 + half * 2);
  } else {
    const uint8_t* base = w + tile * 1024 + r16 * 64 + half * 32;
    r.c0 = *(const v4u*)base;
    r.c1 = *(const v4u*)(base + 16);
  }
}

// The four B fragments (16 weight rows x 64 K) of a tile's raw bytes, as stream_kernel decodes them.
template <bool FP4>
__device__ __forceinline__ void weight_decode(const WRaw<FP4>& r, u32 half, v16h (&bq)[4]) {
  v8u own0, own1;
  if constexpr (FP4) {
    const v2h sa = e4m3_h2(r.s2 & 0xFF), sb = e4m3_h2(r.s2 >> 8);
    v2h o[8];
    dec_fp4(r.c.x, sa, o);
    dec_fp4(r.c.y, sa, o + 4);
#pragma unroll
    for (int e = 0; e < 8; e++) own0[e] = as_u(o[e]);
    dec_fp4(r.c.z, sb, o);
    dec_fp4(r.c.w, sb, o + 4);
#pragma unroll
    for (int e = 0; e < 8; e++) own1[e] = as_u(o[e]);
  } else {
    v2h o[8];
    dec_fp8(r.c0.x, o); dec_fp8(r.c0.y, o + 2); dec_fp8(r.c0.z, o + 4); dec_fp8(r.c0.w, o + 6);
#pragma unroll
    for (int e = 0; e < 8; e++) own0[e] = as_u(o[e]);
    dec_fp8(r.c1.x, o); dec_fp8(r.c1.y, o + 2); dec_fp8(r.c1.z, o + 4); dec_fp8(r.c1.w, o + 6);
#pragma unroll
    for (int e = 0; e < 8; e++) own1[e] = as_u(o[e]);
  }
  v8u oth0, oth1;
#pragma unroll
  for (int e = 0; e < 8; e++) { oth0[e] = xswap(own0[e]); oth1[e] = xswap(own1[e]); }
  bq[0] = __builtin_bit_cast(v16h, half ? oth0 : own0);
  bq[1] = __builtin_bit_cast(v16h, half ? oth1 : own1);
  bq[2] = __builtin_bit_cast(v16h, half ? own0 : oth0);
  bq[3] = __builtin_bit_cast(v16h, half ? own1 : oth1);
}

// One weight tile's 16 WMMAs: four 16-row groups x four 16-K steps, K ascending per group.
__device__ __forceinline__ void mma_tile(const hf* XS, u32 r16, int kk, const v16h (&bq)[4], v8f (&acc)[4]) {
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const v16h a = *(const v16h*)(XS + (g * 16 + r16) * kXSt + kk + q * 16);
      acc[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bq[q], acc[g]);
    }
}

// Activation staging in two halves: the raw codes of a chunk into registers (issued before the previous chunk's
// WMMAs), then decoded into LDS. An item is 32 values of a row (16 bytes of FP4 codes and two scales, or 32 bytes of
// FP8): item i of a chunk is row i / (kc / 32), K offset 32 (i % (kc / 32)). Wide loads: the loads, not the decode,
// are what staging costs.
template <int W>
constexpr int kStageItems = kBR * (kKC / 32) / (W * 32);

// Waves per block: both NVFP4 GEMMs (the down projection and the SwiGLU gate/up) gain from 16, as activation staging
// per output is halved (SwiGLU gate/up -3 % at 2048 rows, measured 2026-11-26); the FP8 GEMM loses with it.
template <bool FP4, int MODE>
constexpr int waves_of() { return FP4 ? 16 : kWaves; }

template <bool FP4, int W>
__device__ __forceinline__ void stage_load(const GemmArgs& p, int m0, int k0, int kc, u32 tid,
                                           v4u (&a)[kStageItems<W>][2], u32 (&b)[kStageItems<W>]) {
  const int per_row = kc / 32, K = p.K;
#pragma unroll
  for (int j = 0; j < kStageItems<W>; ++j) {
    const int i = static_cast<int>(tid) + j * W * 32;
    a[j][0] = a[j][1] = v4u{0, 0, 0, 0}, b[j] = 0;
    if (i < kBR * per_row) {
      const int r = i / per_row, c32 = i - r * per_row, row = m0 + r, k = k0 + c32 * 32;
      if (row < p.M) {
        if constexpr (FP4) {
          a[j][0] = *(const v4u*)(p.xq + static_cast<size_t>(row) * (K / 2) + k / 2);
          b[j] = *(const uint16_t*)(p.xs + static_cast<size_t>(row) * (K / 16) + k / 16);
        } else {
          const v4u* src = (const v4u*)(p.xq + static_cast<size_t>(row) * K + k);
          a[j][0] = src[0], a[j][1] = src[1];
        }
      }
    }
  }
}

template <bool FP4, int W>
__device__ __forceinline__ void stage_store(const GemmArgs& p, int m0, int kc, u32 tid,
                                            const v4u (&a)[kStageItems<W>][2], const u32 (&b)[kStageItems<W>], hf* XS) {
  const int per_row = kc / 32;
#pragma unroll
  for (int j = 0; j < kStageItems<W>; ++j) {
    const int i = static_cast<int>(tid) + j * W * 32;
    if (i < kBR * per_row) {
      const int r = i / per_row, c32 = i - r * per_row;
      const bool in = m0 + r < p.M;
#pragma unroll
      for (int g = 0; g < 4; ++g) {  // the item's four 8-value groups
        v4u v = {0, 0, 0, 0};
        if (in) {
          v2h o[4];
          if constexpr (FP4) {
            dec_fp4(a[j][0][g], e4m3_h2((b[j] >> (8 * (g >> 1))) & 0xFF), o);
          } else {
            dec_fp8(a[j][g >> 1][(g & 1) * 2], o);
            dec_fp8(a[j][g >> 1][(g & 1) * 2 + 1], o + 2);
          }
          v = v4u{as_u(o[0]), as_u(o[1]), as_u(o[2]), as_u(o[3])};
        }
        *(v4u*)(XS + r * kXSt + c32 * 32 + g * 8) = v;
      }
    }
  }
}

// A block: 64 rows x W waves. Each wave holds two 16-column weight tiles (SwiGLU: the gate's and the up projection's
// tile of its 16 columns; otherwise two adjacent column tiles) and feeds every A fragment it reads from LDS to both.
// The activations of a 256-K chunk are decoded to FP16 into LDS (rows past M as zeros); the next chunk's codes are
// loaded into registers while this chunk's WMMAs run, and the next 64-K weight tiles' bytes while this tile's are
// decoded and used. Row blocks run fastest in the grid (blockIdx.x), so the blocks resident together read the same
// weight tiles. K ascends and nothing is split: each output is the plain WMMA chain over K.
template <bool FP4, int MODE>
__global__ void __launch_bounds__((waves_of<FP4, MODE>() * 32)) prefill_gemm_kernel(GemmArgs p) {
  constexpr int W = waves_of<FP4, MODE>();
  __shared__ __attribute__((aligned(16))) hf XS[kBR * kXSt];
  constexpr bool kSw = MODE == kSwiglu;
  constexpr int kTN = kSw ? 1 : 2;  // output tiles per wave
  const u32 tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, r16 = lane & 15;
  const int N = p.N, K = p.K, M = p.M;
  const u32 ntiles = N / 16;
  const u32 nt = (blockIdx.y * W + wave) * kTN;
  const bool valid = nt < ntiles;              // wave-uniform
  const bool second = kSw || nt + 1 < ntiles;  // the second weight tile (always there in SwiGLU)
  const int m0 = blockIdx.x * kBR;
  const uint8_t* __restrict__ sc1 = p.w + static_cast<size_t>(N) * K / 2;
  // The second tile: the up projection's tile nt (SwiGLU) or the gate matrix's tile nt + 1.
  const uint8_t* __restrict__ w2 = kSw ? p.w2 : p.w;
  const uint8_t* __restrict__ sc2 = kSw ? p.w2 + static_cast<size_t>(N) * K / 2 : sc1;
  const u32 nt2 = kSw ? nt : nt + 1;

  v8f acc[kTN][4] = {}, acc2[4] = {};
  v4u sa[kStageItems<W>][2];
  u32 sb[kStageItems<W>];
  stage_load<FP4, W>(p, m0, 0, min(kKC, K), tid, sa, sb);
  WRaw<FP4> wn[2] = {};
  if (valid) {
    weight_load<FP4>(p.w, sc1, nt, r16, half, wn[0]);
    if (second) weight_load<FP4>(w2, sc2, nt2, r16, half, wn[1]);
  }
  for (int k0 = 0; k0 < K; k0 += kKC) {
    const int kc = min(kKC, K - k0);
    stage_store<FP4, W>(p, m0, kc, tid, sa, sb, XS);
    __syncthreads();
    if (k0 + kKC < K) stage_load<FP4, W>(p, m0, k0 + kKC, min(kKC, K - k0 - kKC), tid, sa, sb);
    if (valid) {
      for (int kk = 0; kk < kc; kk += kTileK) {
        const WRaw<FP4> wc0 = wn[0], wc1 = wn[1];
        const int kn = k0 + kk + kTileK;
        if (kn < K) {
          const u64 tb = static_cast<u64>(kn / kTileK) * ntiles;
          weight_load<FP4>(p.w, sc1, tb + nt, r16, half, wn[0]);
          if (second) weight_load<FP4>(w2, sc2, tb + nt2, r16, half, wn[1]);
        }
        v16h bq[4], bq2[4];
        weight_decode<FP4>(wc0, half, bq);
        if (second) {
          weight_decode<FP4>(wc1, half, bq2);
#pragma unroll
          for (int g = 0; g < 4; ++g)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const v16h a = *(const v16h*)(XS + (g * 16 + r16) * kXSt + kk + q * 16);
              acc[0][g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bq[q], acc[0][g]);
              if constexpr (kSw) acc2[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bq2[q], acc2[g]);
              else acc[kTN - 1][g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bq2[q], acc[kTN - 1][g]);
            }
        } else {
          mma_tile(XS, r16, kk, bq, acc[0]);
        }
      }
    }
    __syncthreads();
  }
  if (!valid) return;
#pragma unroll
  for (int t = 0; t < kTN; ++t) {
    if (t == 1 && !second) break;
    // acc[t][g][r]: row m0 + 16 g + 2 r + half, column 16 (nt + t) + r16.
    const int col = (nt + t) * 16 + r16;
#pragma unroll
    for (int g = 0; g < 4; ++g)
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int row = m0 + g * 16 + 2 * r + half;
        if constexpr (kSw) {
          // The 16 columns of a row are one NVFP4 block, held by the 16 lanes of a half-wave.
          const float gv = gpu::mul_rn(acc[0][g][r], p.alpha), uv = gpu::mul_rn(acc2[g][r], p.alpha2);
          const float v = bf16_round(gpu::mul_rn(silu(gv), uv));
          float amax = fabsf(v);
#pragma unroll
          for (int o = 1; o < 16; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(gpu::kWarpMask, amax, o));
          const u32 s = gpu::e4m3_from_float(gpu::mul_rn(p.gscale_out, gpu::mul_rn(amax, 1.f / 6.f)));
          const float sv = e4m3f(s);
          const float os = sv != 0.f ? 1.f / gpu::mul_rn(sv, 1.f / p.gscale_out) : 0.f;
          const u32 code = e2m1(gpu::mul_rn(v, os));
          const u32 other = __shfl_xor_sync(gpu::kWarpMask, code, 1);
          if (row < M) {
            if ((r16 & 1) == 0) p.q_out[static_cast<size_t>(row) * (N / 2) + col / 2] = static_cast<uint8_t>(code | (other << 4));
            if (r16 == 0) p.sf_out[static_cast<size_t>(row) * (N / 16) + nt] = static_cast<uint8_t>(s);
          }
        } else if constexpr (MODE == kBf16Out) {
          if (row < M) p.yb[static_cast<size_t>(row) * p.ldy + col] = gpu::bf16_from_float(gpu::mul_rn(acc[t][g][r], p.alpha));
        } else {
          if (row < M) {
            float* d = p.y + static_cast<size_t>(row) * p.ldy + col;
            float v = gpu::mul_rn(acc[t][g][r], p.alpha);
            if (p.accumulate) v = v + *d;
            *d = v;
          }
        }
      }
  }
}

template <bool FP4, int MODE>
void launch_gemm(GemmArgs p, gpu::Stream s, const char* what) {
  if (p.N <= 0 || p.K <= 0 || p.N % 16 != 0 || p.K % 128 != 0)
    throw std::runtime_error(std::string(what) + ": N % 16 and K % 128 must be 0");
  const int M = p.M;
  constexpr int tn = MODE == kSwiglu ? 1 : 2;  // the kernel's kTN
  constexpr int W = waves_of<FP4, MODE>();
  for (int r0 = 0; r0 < M; r0 += kSlabRows) {
    GemmArgs q = p;
    q.M = std::min(kSlabRows, M - r0);
    q.xq = p.xq + static_cast<size_t>(r0) * (FP4 ? p.K / 2 : p.K);
    if (FP4) q.xs = p.xs + static_cast<size_t>(r0) * (p.K / 16);
    if (p.y) q.y = p.y + static_cast<size_t>(r0) * p.ldy;
    if (p.yb) q.yb = p.yb + static_cast<size_t>(r0) * p.ldy;
    if (p.q_out) q.q_out = p.q_out + static_cast<size_t>(r0) * (p.N / 2);
    if (p.sf_out) q.sf_out = p.sf_out + static_cast<size_t>(r0) * (p.N / 16);
    const unsigned cb = (p.N / 16 + W * tn - 1) / (W * tn), rb = (q.M + kBR - 1) / kBR;
    launch_kernel(what, prefill_gemm_kernel<FP4, MODE>, dim3(rb, cb), dim3(W * 32), 0, s, q);
  }
}

// Narrow BF16 matrices (the DeltaNet a and b projections, 2 x 48 rows): one thread per output, K ascending, one
// FMA per term (x rounded to BF16 as production's activations; BF16 x read as it is). A row's result does not
// depend on M.
template <typename InT>
__global__ void __launch_bounds__(96) narrow_bf16_kernel(const InT* __restrict__ x, int M, const gpu::bf16* __restrict__ wa,
                                                         const gpu::bf16* __restrict__ wb, float* __restrict__ ya,
                                                         float* __restrict__ yb, int N, int K) {
  const int row = blockIdx.x, col = threadIdx.x;
  if (row >= M || col >= 2 * N) return;
  const InT* xr = x + static_cast<size_t>(row) * K;
  const gpu::bf16* wr = col < N ? wa + static_cast<size_t>(col) * K : wb + static_cast<size_t>(col - N) * K;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    const float xv = bf16_round(static_cast<float>(xr[k]));
    acc = fmaf(xv, __bfloat162float(wr[k]), acc);
  }
  if (col < N) ya[static_cast<size_t>(row) * N + col] = acc;
  else yb[static_cast<size_t>(row) * N + col - N] = acc;
}

// The same sums, row-blocked: a block holds 16 rows x 96 outputs (two groups of 96 threads, 8 rows per thread); x
// staged in LDS (rounded to BF16 as above), weights read 16 bytes at a time. Each output still adds its terms one
// FMA at a time with K ascending, so every bit equals narrow_bf16_kernel's.
constexpr int kNarrowRows = 8, kNarrowKC = 256;
template <typename InT>
__global__ void __launch_bounds__(192) narrow_bf16_rows_kernel(const InT* __restrict__ x, int M,
                                                               const gpu::bf16* __restrict__ wa,
                                                               const gpu::bf16* __restrict__ wb, float* __restrict__ ya,
                                                               float* __restrict__ yb, int N, int K) {
  __shared__ __attribute__((aligned(16))) float XN[2 * kNarrowRows][kNarrowKC];
  const int tid = threadIdx.x, col = tid % 96, grp = tid / 96;
  const int r0 = blockIdx.x * 2 * kNarrowRows;
  const gpu::bf16* wr = col < N ? wa + static_cast<size_t>(col) * K : wb + static_cast<size_t>(col - N) * K;
  float acc[kNarrowRows];
#pragma unroll
  for (int r = 0; r < kNarrowRows; ++r) acc[r] = 0.f;
  for (int k0 = 0; k0 < K; k0 += kNarrowKC) {
    const int kc = min(kNarrowKC, K - k0);
    for (int i = tid; i < 2 * kNarrowRows * kc; i += 192) {
      const int r = i / kc, k = i - r * kc, row = r0 + r;
      XN[r][k] = row < M ? bf16_round(static_cast<float>(x[static_cast<size_t>(row) * K + k0 + k])) : 0.f;
    }
    __syncthreads();
    for (int kk = 0; kk < kc; kk += 8) {
      const v4u wv = *(const v4u*)(wr + k0 + kk);
      float wf[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        wf[2 * e] = __uint_as_float(wv[e] << 16);
        wf[2 * e + 1] = __uint_as_float(wv[e] & 0xFFFF0000u);
      }
#pragma unroll
      for (int r = 0; r < kNarrowRows; ++r) {
        const float4 x0 = *(const float4*)&XN[grp * kNarrowRows + r][kk];
        const float4 x1 = *(const float4*)&XN[grp * kNarrowRows + r][kk + 4];
        float a = acc[r];
        a = fmaf(x0.x, wf[0], a); a = fmaf(x0.y, wf[1], a); a = fmaf(x0.z, wf[2], a); a = fmaf(x0.w, wf[3], a);
        a = fmaf(x1.x, wf[4], a); a = fmaf(x1.y, wf[5], a); a = fmaf(x1.z, wf[6], a); a = fmaf(x1.w, wf[7], a);
        acc[r] = a;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < kNarrowRows; ++r) {
    const int row = r0 + grp * kNarrowRows + r;
    if (row < M) {
      if (col < N) ya[static_cast<size_t>(row) * N + col] = acc[r];
      else yb[static_cast<size_t>(row) * N + col - N] = acc[r];
    }
  }
}

template <typename InT>
void launch_narrow(const char* what, const InT* x, int M, const gpu::bf16* wa, const gpu::bf16* wb, float* ya,
                   float* yb, int N, int K, gpu::Stream s) {
  if (2 * N != 96 || K % 16 != 0) throw std::runtime_error("narrow_bf16: built for two 48-row matrices, K % 16 == 0");
  const bool rows = (reinterpret_cast<uintptr_t>(wa) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(wb) % 16 == 0);
  for (int r0 = 0; r0 < M; r0 += kSlabRows) {
    const int m = std::min(kSlabRows, M - r0);
    const InT* xs = x + static_cast<size_t>(r0) * K;
    float *a = ya + static_cast<size_t>(r0) * N, *b = yb + static_cast<size_t>(r0) * N;
    if (rows)
      launch_kernel(what, narrow_bf16_rows_kernel<InT>, dim3((m + 2 * kNarrowRows - 1) / (2 * kNarrowRows)), dim3(192),
                    0, s, xs, m, wa, wb, a, b, N, K);
    else
      launch_kernel(what, narrow_bf16_kernel<InT>, dim3(m), dim3(96), 0, s, xs, m, wa, wb, a, b, N, K);
  }
}

int grid_for(size_t n) { return static_cast<int>(std::min<size_t>((n + 255) / 256, 48 * 64)); }

}  // namespace

void quant_act_nvfp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  if (!quads) return;
  launch_kernel("quant_act_nvfp4", quant_nvfp4_kernel<false>, dim3(grid_for(quads)), dim3(256), 0, s, x,
                static_cast<const float*>(nullptr), quads, 1.f / input_scale, q, sf);
}

void silu_mul_quant_nvfp4(const float* g, const float* u, int M, int K, float input_scale, uint8_t* q, uint8_t* sf,
                          gpu::Stream s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  if (!quads) return;
  launch_kernel("silu_mul_quant_nvfp4", quant_nvfp4_kernel<true>, dim3(grid_for(quads)), dim3(256), 0, s, g, u, quads,
                1.f / input_scale, q, sf);
}

void quant_act_fp8(const float* x, int M, int K, float input_scale, uint8_t* q, gpu::Stream s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  if (!quads) return;
  launch_kernel("quant_act_fp8", quant_fp8_kernel<float>, dim3(grid_for(quads)), dim3(256), 0, s, x, quads,
                1.f / input_scale, q);
}

void quant_act_fp8_bf16in(const gpu::bf16* x, int M, int K, float input_scale, uint8_t* q, gpu::Stream s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  if (!quads) return;
  launch_kernel("quant_act_fp8_bf16in", quant_fp8_kernel<gpu::bf16>, dim3(grid_for(quads)), dim3(256), 0, s, x, quads,
                1.f / input_scale, q);
}

void rmsnorm_quant(const float* x, const gpu::bf16* w, float* out, int M, int H, float eps, bool nvfp4,
                   float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s) {
  if (H % 1024 != 0) throw std::runtime_error("rmsnorm_quant: H % 1024 must be 0");
  if (M < 1) return;
  if (nvfp4)
    launch_kernel("rmsnorm_quant", norm_quant_kernel<true>, dim3(M), dim3(256), 0, s, x, w, out, H, eps,
                  1.f / input_scale, q, sf);
  else
    launch_kernel("rmsnorm_quant", norm_quant_kernel<false>, dim3(M), dim3(256), 0, s, x, w, out, H, eps,
                  1.f / input_scale, q, static_cast<uint8_t*>(nullptr));
}

void rmsnorm_quant_bf16(const float* x, const gpu::bf16* w, gpu::bf16* out, int M, int H, float eps, bool nvfp4,
                        float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s) {
  if (H % 1024 != 0) throw std::runtime_error("rmsnorm_quant_bf16: H % 1024 must be 0");
  if (M < 1) return;
  if (nvfp4)
    launch_kernel("rmsnorm_quant_bf16", norm_quant_kernel<true, gpu::bf16>, dim3(M), dim3(256), 0, s, x, w, out, H,
                  eps, 1.f / input_scale, q, sf);
  else
    launch_kernel("rmsnorm_quant_bf16", norm_quant_kernel<false, gpu::bf16>, dim3(M), dim3(256), 0, s, x, w, out, H,
                  eps, 1.f / input_scale, q, static_cast<uint8_t*>(nullptr));
}

void gated_rmsnorm_fp8(const float* x, const float* z, const gpu::bf16* w, int rows, int D, float eps,
                       float input_scale, uint8_t* q, gpu::Stream s) {
  if (D != 128) throw std::runtime_error("gated_rmsnorm_fp8: D must be 128");
  if (rows < 1) return;
  launch_kernel("gated_rmsnorm_fp8", gated_norm_fp8_kernel<float>, dim3((rows + 7) / 8), dim3(256), 0, s, x, z, w,
                static_cast<size_t>(rows), eps, 1.f / input_scale, q);
}

void gated_rmsnorm_fp8_bf16(const gpu::bf16* x, const gpu::bf16* z, const gpu::bf16* w, int rows, int D, float eps,
                            float input_scale, uint8_t* q, gpu::Stream s) {
  if (D != 128) throw std::runtime_error("gated_rmsnorm_fp8_bf16: D must be 128");
  if (rows < 1) return;
  launch_kernel("gated_rmsnorm_fp8_bf16", gated_norm_fp8_kernel<gpu::bf16>, dim3((rows + 7) / 8), dim3(256), 0, s, x,
                z, w, static_cast<size_t>(rows), eps, 1.f / input_scale, q);
}

void sigmoid_mul_fp8(const float* x, const float* gate, size_t n, float input_scale, uint8_t* q, gpu::Stream s) {
  if (n < 4) return;
  launch_kernel("sigmoid_mul_fp8", sigmoid_mul_fp8_kernel, dim3(grid_for(n / 4)), dim3(256), 0, s, x, gate, n / 4,
                1.f / input_scale, q);
}

void prefill_gemm_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, float alpha, float* y, int ldy,
                        int N, int K, bool accumulate, gpu::Stream s, const uint8_t* wsf) {
  if (wsf) throw std::runtime_error("prefill_gemm_nvfp4: no coded NVFP4 blobs on HIP");
  GemmArgs p{xq, xs, w, nullptr, alpha, 0.f, 0.f, y, nullptr, nullptr, nullptr, ldy, M, N, K, accumulate};
  launch_gemm<true, kPlain>(p, s, "prefill_gemm_nvfp4");
}

void narrow_bf16(const float* x, int M, const gpu::bf16* wa, const gpu::bf16* wb, float* ya, float* yb, int N, int K,
                 gpu::Stream s) {
  launch_narrow("narrow_bf16", x, M, wa, wb, ya, yb, N, K, s);
}

void narrow_bf16_bf16in(const gpu::bf16* x, int M, const gpu::bf16* wa, const gpu::bf16* wb, float* ya, float* yb,
                        int N, int K, gpu::Stream s) {
  launch_narrow("narrow_bf16_bf16in", x, M, wa, wb, ya, yb, N, K, s);
}

void prefill_swiglu_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* gate, float alpha_gate,
                          const uint8_t* up, float alpha_up, int N, int K, float out_input_scale, uint8_t* q_out,
                          uint8_t* sf_out, gpu::Stream s, const uint8_t* gate_sf, const uint8_t* up_sf) {
  if (N % 128 != 0 || K % 128 != 0) throw std::runtime_error("prefill_swiglu_nvfp4: N % 128 and K % 128 must be 0");
  if (gate_sf || up_sf) throw std::runtime_error("prefill_swiglu_nvfp4: no coded NVFP4 blobs on HIP");
  if (M < 1) return;
  GemmArgs p{xq, xs, gate, up, alpha_gate, alpha_up, 1.f / out_input_scale, nullptr, nullptr, q_out, sf_out, 0, M, N, K,
             false};
  launch_gemm<true, kSwiglu>(p, s, "prefill_swiglu_nvfp4");
}

void prefill_gemm_fp8(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                      bool accumulate, gpu::Stream s) {
  GemmArgs p{xq, nullptr, w, nullptr, alpha, 0.f, 0.f, y, nullptr, nullptr, nullptr, ldy, M, N, K, accumulate};
  launch_gemm<false, kPlain>(p, s, "prefill_gemm_fp8");
}

void prefill_gemm_fp8_lora(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                           bool accumulate, const LoraGemm& lora, gpu::Stream s) {
  if (lora.nsegs != 0) throw std::runtime_error("prefill_gemm_fp8_lora: LoRA adapters are not built on HIP");
  prefill_gemm_fp8(xq, M, w, alpha, y, ldy, N, K, accumulate, s);
}

void prefill_gemm_fp8_bf16out(const uint8_t* xq, int M, const uint8_t* w, float alpha, gpu::bf16* y, int ldy, int N,
                              int K, gpu::Stream s) {
  GemmArgs p{xq, nullptr, w, nullptr, alpha, 0.f, 0.f, nullptr, y, nullptr, nullptr, ldy, M, N, K, false};
  launch_gemm<false, kBf16Out>(p, s, "prefill_gemm_fp8_bf16out");
}

}  // namespace ragon::kernels
