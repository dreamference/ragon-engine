// The GDN prefill kernels for the hip backend (prefill_gdn.cu is the cuda form).
//
// The conv is prefill_gdn.cu's, unchanged but for the launch. The recurrence is the token-sequential delta
// rule of kernels.cu (gdn_recurrent_rows: the same per-token updates, one 32-lane wave per 32 value columns of a
// head), not the cuda form's 32-token chunks on mma.sync tensor cores: it has no chunked form, so its result never
// depends on how a prompt is split (a stronger invariance than the chunk-end one cuda keeps). The state is held
// k-major (see gdn_prefill_kernel), so its sums run in another order than gdn_recurrent's: the two agree to
// rounding, not bit for bit, as the cuda chunked form does.
// A launch takes at most kSlab tokens and carries the state through global memory (FP32, exact), so no single
// kernel runs long whatever the prompt length (the Windows display-timeout limit).
#include "core/kernels.cuh"
#include "core/launch.cuh"
#include "core/attention_body.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "backend/device.hpp"
#include "core/elem.cuh"

namespace ragon::kernels {
namespace {

constexpr int kSlab = 1024;  // tokens per recurrence launch
constexpr int kT = 32;       // the cuda form's chunk; state_at keeps its contract

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

template <typename InT, typename OutT>
__global__ void conv_prefill_kernel(const InT* __restrict__ in, OutT* __restrict__ out, const float* __restrict__ state,
                                    const gpu::bf16* __restrict__ w, int M, int C) {
  const size_t n = static_cast<size_t>(M) * C;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(i / C), c = static_cast<int>(i % C);
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int tt = t - 3 + j;
      x[j] = tt >= 0 ? elem::load(in + static_cast<size_t>(tt) * C + c) : state[c * 3 + 3 + tt];
    }
    const float o = __bfloat162float(w[c * 4]) * x[0] + __bfloat162float(w[c * 4 + 1]) * x[1] +
                    __bfloat162float(w[c * 4 + 2]) * x[2] + __bfloat162float(w[c * 4 + 3]) * x[3];
    elem::store(out + i, silu(o));
  }
}

template <typename InT>
__global__ void conv_window_kernel(const InT* __restrict__ in, const float* state, float* out, int M, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int tt = M - 3 + j;
    s[j] = tt >= 0 ? elem::load(in + static_cast<size_t>(tt) * C + c) : state[c * 3 + 3 + tt];
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) out[c * 3 + j] = s[j];
}

// The k-major form: lane l of the wave holds rows 4 l .. 4 l + 3 of the state for the wave's 32 value columns, so a
// lane's own q / k elements (its load4) are all it reads of them. The column sums kv and o are its four products
// added, then a reduce-scatter across the 32 lanes (lane c ends with column c); d goes back to every lane through LDS
// (8 broadcast reads per token instead of the v-major form's 96). The sums are in a different order from
// gdn_recurrent's, so agreement is to rounding, not bit for bit. Measured 2026-11-26 on gfx1151 (2048 tokens, 48
// heads): 3.9 ms the v-major form, 2.9 ms this one over __shfl_xor (ds_bpermute), 2.3 ms with the lane exchange below.
//
// x from lane ^ W without LDS: DPP row_xmask within a row of 16 lanes, permlanex16 across the two rows (wave32).
template <int W>
__device__ __forceinline__ float xor_lane(float x) {
  const int i = __float_as_int(x);
  if constexpr (W == 16)
    return __int_as_float(__builtin_amdgcn_permlanex16(i, i, 0x76543210u, 0xfedcba98u, false, false));
  else
    return __int_as_float(__builtin_amdgcn_update_dpp(0, i, 0x160 | W, 0xf, 0xf, false));
}

template <int W>
__device__ __forceinline__ float xor_sum(float x) {
  x += xor_lane<W>(x);
  if constexpr (W > 1) x = xor_sum<W / 2>(x);
  return x;
}

template <int W>
__device__ __forceinline__ void reduce_scatter_step(float (&p)[32], int lane) {
  const bool hi = lane & W;
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const float keep = hi ? p[W + i] : p[i], send = hi ? p[i] : p[W + i];
    p[i] = keep + xor_lane<W>(send);
  }
  if constexpr (W > 1) reduce_scatter_step<W / 2>(p, lane);
}

__device__ __forceinline__ float reduce_scatter32(float (&p)[32], int lane) {
  reduce_scatter_step<16>(p, lane);
  return p[0];
}

template <typename InT, typename OutT>
__global__ void __launch_bounds__(32)
    gdn_prefill_kernel(const InT* __restrict__ mixed, const float* __restrict__ g, const float* __restrict__ beta,
                          float* state, float* state_copy, OutT* __restrict__ out, int M, int H, int HV) {
  __shared__ __align__(16) float ds[32];
  const int hv = blockIdx.x / 4, part = blockIdx.x % 4, lane = threadIdx.x, v0 = part * 32;
  float* st = state + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
  float S[4][32];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int c = 0; c < 32; c += 4) {
      const float4 x = *reinterpret_cast<const float4*>(st + (4 * lane + j) * kGdnDv + v0 + c);
      S[j][c] = x.x, S[j][c + 1] = x.y, S[j][c + 2] = x.z, S[j][c + 3] = x.w;
    }
  const int h = hv / (HV / H);
  const int C = 2 * H * kGdnDk + HV * kGdnDv;
  const float scale = rsqrtf(static_cast<float>(kGdnDk));
  float4 q4n{}, k4n{};
  float gn = 0.f, bn = 0.f, vn = 0.f;
  auto fetch = [&](int t) {
    const InT* row = mixed + static_cast<size_t>(t) * C;
    q4n = elem::load4(row + h * kGdnDk, lane);
    k4n = elem::load4(row + H * kGdnDk + h * kGdnDk, lane);
    vn = elem::load(row + 2 * H * kGdnDk + hv * kGdnDv + v0 + lane);
    gn = g[t * HV + hv];
    bn = beta[t * HV + hv];
  };
  if (M > 0) fetch(0);
  for (int t = 0; t < M; ++t) {
    const float4 q4 = q4n, k4 = k4n;
    const float gt = gn, b = bn, vv = vn;
    if (t + 1 < M) fetch(t + 1);
    float qn = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
    float kn = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
    qn = xor_sum<16>(qn), kn = xor_sum<16>(kn);
    const float qr = rsqrtf(qn + 1e-6f) * scale, kr = rsqrtf(kn + 1e-6f);
    const float kk[4] = {k4.x * kr, k4.y * kr, k4.z * kr, k4.w * kr};
    const float qq[4] = {q4.x * qr, q4.y * qr, q4.z * qr, q4.w * qr};
    const float decay = __expf(gt);
    float p[32];
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      S[0][c] *= decay, S[1][c] *= decay, S[2][c] *= decay, S[3][c] *= decay;
      p[c] = fmaf(S[3][c], kk[3], fmaf(S[2][c], kk[2], fmaf(S[1][c], kk[1], S[0][c] * kk[0])));
    }
    const float d = (vv - reduce_scatter32(p, lane)) * b;
    __syncwarp();
    ds[lane] = d;
    __syncwarp();
#pragma unroll
    for (int c = 0; c < 32; c += 4) {
      const float4 d4 = *reinterpret_cast<const float4*>(ds + c);
      const float dc[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int j = 0; j < 4; ++j) S[j][c + e] = fmaf(dc[e], kk[j], S[j][c + e]);
        p[c + e] = fmaf(S[3][c + e], qq[3], fmaf(S[2][c + e], qq[2], fmaf(S[1][c + e], qq[1], S[0][c + e] * qq[0])));
      }
    }
    elem::store(out + static_cast<size_t>(t) * HV * kGdnDv + hv * kGdnDv + v0 + lane, reduce_scatter32(p, lane));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int c = 0; c < 32; c += 4) {
      const float4 x = make_float4(S[j][c], S[j][c + 1], S[j][c + 2], S[j][c + 3]);
      *reinterpret_cast<float4*>(st + (4 * lane + j) * kGdnDv + v0 + c) = x;
      if (state_copy)
        *reinterpret_cast<float4*>(state_copy + static_cast<size_t>(hv) * kGdnDk * kGdnDv + (4 * lane + j) * kGdnDv +
                                   v0 + c) = x;
    }
}

template <typename InT, typename OutT>
void conv_prefill(const InT* in, OutT* out, float* conv_state, const gpu::bf16* w, int M, int C, gpu::Stream s,
                  float* window_at, int at) {
  const size_t n = static_cast<size_t>(M) * C;
  launch_kernel("gdn_conv_prefill", conv_prefill_kernel<InT, OutT>,
                dim3(static_cast<int>(std::min<size_t>((n + 255) / 256, 48 * 64))), dim3(256), 0, s, in, out,
                conv_state, w, M, C);
  if (window_at && at > 0 && at <= M)
    launch_kernel("gdn_conv_window_at", conv_window_kernel<InT>, dim3((C + 255) / 256), dim3(256), 0, s, in,
                  static_cast<const float*>(conv_state), window_at, at, C);
  launch_kernel("gdn_conv_window", conv_window_kernel<InT>, dim3((C + 255) / 256), dim3(256), 0, s, in,
                static_cast<const float*>(conv_state), conv_state, M, C);
}

// Tokens [t0, t1) in slabs of kSlab, the last slab also writing its state to copy_at_end.
template <typename InT, typename OutT>
void seq_range(const InT* mixed, const float* g, const float* beta, float* state, OutT* out, int t0, int t1, int H,
               int HV, float* copy_at_end, gpu::Stream s) {
  const size_t C = 2 * static_cast<size_t>(H) * kGdnDk + static_cast<size_t>(HV) * kGdnDv;
  if (t0 == t1 && copy_at_end)  // an empty range: the state copied as it stands
    launch_kernel("gdn_recurrent_prefill", gdn_prefill_kernel<InT, OutT>, dim3(HV * 4), dim3(32), 0, s, mixed, g,
                  beta, state, copy_at_end, out, 0, H, HV);
  for (int t = t0; t < t1; t += kSlab) {
    const int n = std::min(kSlab, t1 - t);
    launch_kernel("gdn_recurrent_prefill", gdn_prefill_kernel<InT, OutT>, dim3(HV * 4), dim3(32), 0, s,
                  mixed + static_cast<size_t>(t) * C, g + static_cast<size_t>(t) * HV,
                  beta + static_cast<size_t>(t) * HV, state, t + n == t1 ? copy_at_end : nullptr,
                  out + static_cast<size_t>(t) * HV * kGdnDv, n, H, HV);
  }
}

template <typename InT, typename OutT>
void recurrent_prefill(const InT* mixed, const float* g, const float* beta, float* state, OutT* out, int M, int H, int HV,
                       int pos0, gpu::Stream s, float* state_at, int at) {
  if (M < 1) return;
  if ((pos0 + at) % kT != 0 && state_at) throw std::runtime_error("gdn_recurrent_prefill: state_at must be at a chunk end");
  if (!state_at || at < 0 || at > M) {  // a state_at past the prompt is never reached, as in the chunked form
    seq_range(mixed, g, beta, state, out, 0, M, H, HV, nullptr, s);
    return;
  }
  seq_range(mixed, g, beta, state, out, 0, at, H, HV, state_at, s);
  seq_range(mixed, g, beta, state, out, at, M, H, HV, nullptr, s);
}

}  // namespace

void gdn_conv_prefill(const float* in, float* out, float* conv_state, const gpu::bf16* w, int M, int C, gpu::Stream s,
                      float* window_at, int at) {
  conv_prefill<float, float>(in, out, conv_state, w, M, C, s, window_at, at);
}

void gdn_conv_prefill_bf16(const gpu::bf16* in, gpu::bf16* out, float* conv_state, const gpu::bf16* w, int M, int C,
                           gpu::Stream s, float* window_at, int at) {
  conv_prefill<gpu::bf16, gpu::bf16>(in, out, conv_state, w, M, C, s, window_at, at);
}

void gdn_recurrent_prefill(const float* mixed, const float* g, const float* beta, float* state, float* out, int M, int H,
                           int HV, int pos0, gpu::Stream s, float* state_at, int at) {
  recurrent_prefill<float, float>(mixed, g, beta, state, out, M, H, HV, pos0, s, state_at, at);
}

void gdn_recurrent_prefill_bf16(const gpu::bf16* mixed, const float* g, const float* beta, float* state,
                                gpu::bf16* out, int M, int H, int HV, int pos0, gpu::Stream s, float* state_at,
                                int at) {
  recurrent_prefill<gpu::bf16, gpu::bf16>(mixed, g, beta, state, out, M, H, HV, pos0, s, state_at, at);
}

}  // namespace ragon::kernels
