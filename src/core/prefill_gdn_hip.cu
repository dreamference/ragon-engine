// The GDN prefill kernels for the hip backend (W1, correctness first; prefill_gdn.cu is the cuda form).
//
// The conv is prefill_gdn.cu's, unchanged but for the launch. The recurrence is the token-sequential delta
// rule of kernels.cu (gdn_recurrent_rows: the same per-token arithmetic, one 32-lane wave per 32 value
// columns of a head), not the cuda form's 32-token chunks on mma.sync tensor cores: it has no chunked form, so
// its result never depends on how a prompt is split (a stronger invariance than the chunk-end one cuda keeps),
// and over FP32 inputs it is gdn_recurrent bit for bit. Against the cuda chunked form it agrees to tolerance.
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

// Tokens 0 .. M - 1 of a slab (the pointers already at its first token), the state read from and written back
// to `state` (and also to `state_copy` when given). The loop body is gdn_recurrent_rows over InT/OutT.
template <typename InT, typename OutT>
__global__ void __launch_bounds__(32)
    gdn_prefill_seq_kernel(const InT* __restrict__ mixed, const float* __restrict__ g, const float* __restrict__ beta,
                           float* state, float* state_copy, OutT* __restrict__ out, int M, int H, int HV) {
  __shared__ float qs[kGdnDk], ks[kGdnDk];
  const int hv = blockIdx.x / 4, part = blockIdx.x % 4, lane = threadIdx.x, v = part * 32 + lane;
  float* st = state + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
  float S[kGdnDk];
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) S[k] = st[k * kGdnDv + v];
  const int h = hv / (HV / H);
  const int C = 2 * H * kGdnDk + HV * kGdnDv;
  const float scale = rsqrtf(static_cast<float>(kGdnDk));
  for (int t = 0; t < M; ++t) {
    const InT* row = mixed + static_cast<size_t>(t) * C;
    const float4 q4 = elem::load4(row + h * kGdnDk, lane);
    const float4 k4 = elem::load4(row + H * kGdnDk + h * kGdnDk, lane);
    float qn = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
    float kn = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      qn += __shfl_xor_sync(gpu::kWarpMask, qn, o);
      kn += __shfl_xor_sync(gpu::kWarpMask, kn, o);
    }
    const float qr = rsqrtf(qn + 1e-6f) * scale, kr = rsqrtf(kn + 1e-6f);
    __syncwarp();
    qs[lane * 4] = q4.x * qr, qs[lane * 4 + 1] = q4.y * qr, qs[lane * 4 + 2] = q4.z * qr, qs[lane * 4 + 3] = q4.w * qr;
    ks[lane * 4] = k4.x * kr, ks[lane * 4 + 1] = k4.y * kr, ks[lane * 4 + 2] = k4.z * kr, ks[lane * 4 + 3] = k4.w * kr;
    __syncwarp();
    const float decay = __expf(g[t * HV + hv]);
    const float b = beta[t * HV + hv];
    const float vv = elem::load(row + 2 * H * kGdnDk + hv * kGdnDv + v);
    float kv = 0.f;
#pragma unroll
    for (int k = 0; k < kGdnDk; ++k) {
      S[k] *= decay;
      kv = fmaf(S[k], ks[k], kv);
    }
    const float d = (vv - kv) * b;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < kGdnDk; ++k) {
      S[k] = fmaf(d, ks[k], S[k]);
      o = fmaf(S[k], qs[k], o);
    }
    elem::store(out + static_cast<size_t>(t) * HV * kGdnDv + hv * kGdnDv + v, o);
  }
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) st[k * kGdnDv + v] = S[k];
  if (state_copy == nullptr) return;
  float* sc = state_copy + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) sc[k * kGdnDv + v] = S[k];
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
    launch_kernel("gdn_recurrent_prefill", gdn_prefill_seq_kernel<InT, OutT>, dim3(HV * 4), dim3(32), 0, s, mixed, g,
                  beta, state, copy_at_end, out, 0, H, HV);
  for (int t = t0; t < t1; t += kSlab) {
    const int n = std::min(kSlab, t1 - t);
    launch_kernel("gdn_recurrent_prefill", gdn_prefill_seq_kernel<InT, OutT>, dim3(HV * 4), dim3(32), 0, s,
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
