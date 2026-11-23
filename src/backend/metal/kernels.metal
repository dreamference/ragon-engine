// The Metal backend's kernels (METAL spec, first lane): the M=1 decode path and a row-blocked prefill path.
// Activations are f32, the KV cache bfloat, the GDN state f32. Weights are read as the checkpoint ships them:
//   NVFP4  [N][K/2] bytes, two e2m1 codes a byte (low nibble first), one e4m3 scale per 16 values [N][K/16],
//          times a global f32 scale (weight_scale_2);
//   FP8    [N][K] e4m3 bytes times a per-tensor f32 scale;
//   BF16   [N][K]. Small vectors (norm weights, conv taps, A_log, dt_bias) arrive as f32 from the loader.
// e2m1 and e4m3 are decoded in registers by moving their bits into an IEEE half and rescaling once:
//   e4m3 bits s.eeee.mmm -> half bits s.0eeee.mmm0000000, value * 2^8;
//   e2m1 bits s.ee.m     -> half bits s.000ee.m000000000, value * 2^14 (subnormal codes land on half subnormals).
#include <metal_stdlib>
using namespace metal;

constant uint kRows [[function_constant(0)]];  // activation rows per GEMV pass (1 for decode, up to 8 for prefill)
// Decode (kRows 1): the residual-stream rmsnorm folded into the GEMVs around it. The GEMV that adds into the
// residual (kFoldOut) also writes the next norm's input times its weight, xw = h * (1 + w), and per threadgroup
// the sum of h*h over its 8 rows; the GEMVs that read the normed row (kFoldIn) dot with xw and scale by
// rsqrt(mean(h*h) + eps) from those partial sums. No norm pass and no barrier for it between the big GEMVs.
constant bool kFoldIn [[function_constant(1)]];
constant bool kFold = is_function_constant_defined(kFoldIn) && kFoldIn;
constant bool kFoldOutIn [[function_constant(2)]];
constant bool kFoldOut = is_function_constant_defined(kFoldOutIn) && kFoldOutIn;

constant float kFp8Unscale = 256.0f;
constant float kFp4Unscale = 16384.0f;

inline float2 fp8x2(uint w) {  // the two low bytes of w, unscaled (times 2^-8)
  uint h = ((w & 0x7Fu) << 7) | ((w & 0x80u) << 8) | ((w & 0x7F00u) << 15) | ((w & 0x8000u) << 16);
  return float2(as_type<half2>(h));
}

inline float2 fp4x2(uint v) {  // the two low nibbles of v, unscaled (times 2^-14)
  uint h = ((v & 0x7u) << 9) | ((v & 0x8u) << 12) | ((v & 0x70u) << 21) | ((v & 0x80u) << 24);
  return float2(as_type<half2>(h));
}

inline float e4m3(uint b) {
  return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)))) * kFp8Unscale;
}

inline float2 bf16x2(uint w) { return float2(as_type<float>(w << 16), as_type<float>(w & 0xFFFF0000u)); }

struct GemvArgs {
  uint N, K;       // y[m][n] for n < N, reduction over K
  uint ldx, ldy;   // row strides of x and y (floats)
  uint rows;       // activation rows in total; threadgroup row y takes rows 8y .. 8y+7 (kRows of them)
  float scale;     // global scale (NVFP4 scale_2, the FP8 per-tensor scale, 1 for BF16)
  uint add;        // 1: y += result (the residual adds)
  float eps;       // kFold: the rmsnorm epsilon
  uint npart;      // kFold: partial sums to add (K / 8)
};

// Fold buffers: kFoldIn reads the partial sums at 6; kFoldOut reads the next norm's weight at 7 and writes xw at 8
// and the partial sums at 9.
#define FOLD_ARGS                                                                                          \
  device const float *fpart [[buffer(6)]], device const float *fnw [[buffer(7)]],                          \
      device float *fxw [[buffer(8)]], device float *fpo [[buffer(9)]]

// The 8 simdgroups of the threadgroup share the partial sums' read (N is a multiple of 8 when kFold is on).
inline void fold_scale(thread float* acc, device const float* part, constant GemvArgs& a, uint lane, uint sg,
                       threadgroup float* tp) {
  if (!kFold) return;
  float ss = 0;
  for (uint i = sg * 32 + lane; i < a.npart; i += 256) ss += part[i];
  ss = simd_sum(ss);
  if (lane == 0) tp[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0;
  for (uint i = 0; i < 8; ++i) t += tp[i];
  acc[0] *= rsqrt(t / float(a.K) + a.eps);
}

inline void gemv_store(device float* y, constant GemvArgs& a, uint rows, uint n, thread float* acc, float unscale,
                       uint lane, threadgroup float* tp = nullptr, device const float* nw = nullptr,
                       device float* xw = nullptr, device float* po = nullptr, uint sg = 0, uint tgx = 0) {
  for (uint m = 0; m < kRows; ++m) {
    if (m >= rows) break;
    const float r = simd_sum(acc[m]) * a.scale * unscale;
    if (lane == 0) {
      device float* p = y + (ulong)m * a.ldy + n;
      const float v = a.add ? *p + r : r;
      *p = v;
      if (kFoldOut) {
        xw[n] = v * (1.0f + nw[n]);
        tp[sg] = v * v;
      }
    }
  }
  if (kFoldOut) {  // N is a multiple of 8 here: every simdgroup of the threadgroup has a row
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
      float t = 0;
      for (uint i = 0; i < 8; ++i) t += tp[i];
      po[tgx] = t;
    }
  }
}

// One simdgroup per output row n, 8 rows a threadgroup; each lane takes 16 weight bytes (32 values) a step.
kernel void gemv_nvfp4(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                       device const uchar* sc [[buffer(2)]], device float* y [[buffer(3)]],
                       constant GemvArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], FOLD_ARGS) {
  threadgroup float ftp[8];
  const uint n = tg.x * 8 + sg;
  if (n >= a.N) return;
  const uint rows = min(kRows, a.rows - tg.y * 8);
  x += (ulong)tg.y * 8 * a.ldx;
  y += (ulong)tg.y * 8 * a.ldy;
  device const uint4* wr = (device const uint4*)(w + (ulong)n * (a.K / 2));
  device const ushort* sr = (device const ushort*)(sc + (ulong)n * (a.K / 16));
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint steps = a.K / 32;
  for (uint s = lane; s < steps; s += 32) {
    const uint4 q = wr[s];
    const uint sv = sr[s];
    const float s0 = e4m3(sv & 0xFFu), s1 = e4m3(sv >> 8);
    float2 v[16];  // v[k] = values 2k, 2k+1 of the 32
    for (uint i = 0; i < 4; ++i)
      for (uint j = 0; j < 4; ++j) v[i * 4 + j] = fp4x2(q[i] >> (8 * j));
    for (uint m = 0; m < kRows; ++m) {
      if (m >= rows) break;
      device const float4* xr = (device const float4*)(x + (ulong)m * a.ldx + s * 32);
      float b0 = 0, b1 = 0;
      for (uint i = 0; i < 4; ++i) {
        const float4 xa = xr[i];
        b0 += dot(v[2 * i], xa.xy) + dot(v[2 * i + 1], xa.zw);
        const float4 xb = xr[i + 4];
        b1 += dot(v[8 + 2 * i], xb.xy) + dot(v[8 + 2 * i + 1], xb.zw);
      }
      acc[m] += s0 * b0 + s1 * b1;
    }
  }
  fold_scale(acc, fpart, a, lane, sg, ftp);
  gemv_store(y, a, rows, n, acc, kFp4Unscale, lane, ftp, fnw, fxw, fpo, sg, tg.x);
}

// FP8 rows: each lane takes 16 weight bytes (16 values) a step.
kernel void gemv_fp8(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                     device float* y [[buffer(2)]], constant GemvArgs& a [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], FOLD_ARGS) {
  threadgroup float ftp[8];
  const uint n = tg.x * 8 + sg;
  if (n >= a.N) return;
  const uint rows = min(kRows, a.rows - tg.y * 8);
  x += (ulong)tg.y * 8 * a.ldx;
  y += (ulong)tg.y * 8 * a.ldy;
  device const uint4* wr = (device const uint4*)(w + (ulong)n * a.K);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint steps = a.K / 16;
  for (uint s = lane; s < steps; s += 32) {
    const uint4 q = wr[s];
    float2 v[8];
    for (uint i = 0; i < 4; ++i) {
      v[2 * i] = fp8x2(q[i]);
      v[2 * i + 1] = fp8x2(q[i] >> 16);
    }
    for (uint m = 0; m < kRows; ++m) {
      if (m >= rows) break;
      device const float4* xr = (device const float4*)(x + (ulong)m * a.ldx + s * 16);
      float b = 0;
      for (uint i = 0; i < 4; ++i) {
        const float4 xa = xr[i];
        b += dot(v[2 * i], xa.xy) + dot(v[2 * i + 1], xa.zw);
      }
      acc[m] += b;
    }
  }
  fold_scale(acc, fpart, a, lane, sg, ftp);
  gemv_store(y, a, rows, n, acc, kFp8Unscale, lane, ftp, fnw, fxw, fpo, sg, tg.x);
}

// ---- Q4: the Mac-native 4-bit projection option (affine, groups of 64) -------------------------------
// Built at load time from an FP8 projection (q4_quant). Layout per row n:
//   w  [N][K/2] bytes: each 32-bit word holds 8 values; value 2j sits in nibble j, value 2j+1 in nibble j+4, so
//      (word >> 4j) & 0x000F000F gives the pair (2j, 2j+1) as the low bits of two halves;
//   sb [N][K/64] half2 (scale, bias) in raw e4m3 units: value = scale * q + bias; the per-tensor FP8 scale stays
//      in GemvArgs.scale / mm_store.
// A nibble n becomes a half via the bits 0x6400 | n = 1024 + n, minus 1024 (exact).
inline half2 q4x2(uint w, uint j) { return as_type<half2>(0x64006400u | ((w >> (4 * j)) & 0x000F000Fu)) - half2(1024.0h); }

// One simdgroup per output row n, 8 rows a threadgroup; each lane takes 16 weight bytes (32 values, half a group)
// a step: acc += scale * sum(q x) + bias * sum(x).
kernel void gemv_q4(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                    device const uchar* sb [[buffer(2)]], device float* y [[buffer(3)]],
                    constant GemvArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], FOLD_ARGS) {
  threadgroup float ftp[8];
  const uint n = tg.x * 8 + sg;
  if (n >= a.N) return;
  const uint rows = min(kRows, a.rows - tg.y * 8);
  x += (ulong)tg.y * 8 * a.ldx;
  y += (ulong)tg.y * 8 * a.ldy;
  device const uint4* wr = (device const uint4*)(w + (ulong)n * (a.K / 2));
  device const half2* sr = (device const half2*)(sb + (ulong)n * (a.K / 64) * 4);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint steps = a.K / 32;
  for (uint s = lane; s < steps; s += 32) {
    const uint4 q = wr[s];
    const float2 cb = float2(sr[s >> 1]);
    half2 v[16];  // v[4i + j] = values 8i + 2j, 8i + 2j + 1 of the 32
    for (uint i = 0; i < 4; ++i)
      for (uint j = 0; j < 4; ++j) v[i * 4 + j] = q4x2(q[i], j);
    for (uint m = 0; m < kRows; ++m) {
      if (m >= rows) break;
      device const float4* xr = (device const float4*)(x + (ulong)m * a.ldx + s * 32);
      float b = 0, sx = 0;
      for (uint i = 0; i < 8; ++i) {
        const float4 xa = xr[i];
        b += dot(float2(v[2 * i]), xa.xy) + dot(float2(v[2 * i + 1]), xa.zw);
        sx += (xa.x + xa.y) + (xa.z + xa.w);
      }
      acc[m] += cb.x * b + cb.y * sx;
    }
  }
  fold_scale(acc, fpart, a, lane, sg, ftp);
  gemv_store(y, a, rows, n, acc, 1.0f, lane, ftp, fnw, fxw, fpo, sg, tg.x);
}

// Q4 [N][K] -> half [N][K] (raw units, without the per-tensor scale): one thread per 32 values.
kernel void dq_q4(device const uchar* w [[buffer(0)]], device const uchar* sb [[buffer(1)]],
                  device half* out [[buffer(2)]], constant uint& n32 [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
  if (i >= n32) return;
  const uint4 q = ((device const uint4*)w)[i];
  const float2 cb = float2(((device const half2*)sb)[i >> 1]);
  device half2* o = (device half2*)(out + (ulong)i * 32);
  for (uint k = 0; k < 4; ++k)
    for (uint j = 0; j < 4; ++j) o[k * 4 + j] = half2(float2(q4x2(q[k], j)) * cb.x + cb.y);
}

// ---- Small-T GEMV (2..8 activation rows, the speculative verify) on the simdgroup matrix units ----------------
// The weights are decoded once into the 8x8 A fragments of a simdgroup that owns 8 output rows; each 8x8x8 MMA
// takes 8 k values of 8 output rows against those k of the 8 activation rows, so the per-row cost is one MMA, not a
// pass of FMAs over decoded values. Fragment coordinates (lane -> row fm, columns fn, fn+1) follow the Apple layout.
// k order: the sum runs over k, so a lane's fragment pairs take a contiguous run of its row's weights; tile t of
// step s covers k = 32 (s + i/2) + 2t + i%2 for k-index i. x_pack8 lays x out to match: half2 pairs of activation
// rows (2rp, 2rp+1) at ((s*2 + e)*4 + rp)*16 + t for k = 32s + 2t + e, rows >= T zero.
kernel void x_pack8(device const float* x [[buffer(0)]], device half2* xp [[buffer(1)]],
                    constant uint4& a [[buffer(2)]], uint2 id [[thread_position_in_grid]]) {  // a: K, T, ldx
  const uint k = id.x, rp = id.y;
  if (k >= a.x) return;
  const uint r0 = 2 * rp, r1 = r0 + 1;
  const float v0 = r0 < a.y ? x[(ulong)r0 * a.z + k] : 0.0f;
  const float v1 = r1 < a.y ? x[(ulong)r1 * a.z + k] : 0.0f;
  const uint s = k >> 5, t = (k & 31u) >> 1, e = k & 1u;
  xp[((s * 2 + e) * 4 + rp) * 16 + t] = half2(clamp(float2(v0, v1), -65504.0f, 65504.0f));
}

template <uint F>  // 0: NVFP4, 1: Q4
kernel void gemv8(device const half2* xp [[buffer(0)]], device const uchar* w [[buffer(1)]],
                  device const uchar* sc [[buffer(2)]], device float* y [[buffer(3)]],
                  constant GemvArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 8 + sg) * 8;
  if (n0 >= a.N) return;
  const uint qid = lane / 4;
  const uint fm = (qid & 4u) + ((lane / 2) % 4), fn = (qid & 2u) * 2 + (lane % 2) * 2;
  const uint n = min(n0 + fm, a.N - 1);
  device const uint4* wr = (device const uint4*)(w + (ulong)n * (a.K / 2));
  const uint cw = fn / 2, cx = fm / 2, ex = fm & 1u, rp = fn / 2;
  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const uint steps = a.K / 32;
  for (uint it = 0; it < steps; it += 4) {
    const uint4 q = wr[it + cw];
    half2 wv[16];
    if (F == 0) {
      const uint sv = ((device const ushort*)(sc + (ulong)n * (a.K / 16)))[it + cw];
      const float s0 = e4m3(sv & 0xFFu) * kFp4Unscale, s1 = e4m3(sv >> 8) * kFp4Unscale;
      for (uint t = 0; t < 16; ++t) wv[t] = half2(fp4x2(q[t / 4] >> (8 * (t % 4))) * (t < 8 ? s0 : s1));
    } else {
      const float2 cb = float2(((device const half2*)(sc + (ulong)n * (a.K / 64) * 4))[(it + cw) >> 1]);
      for (uint t = 0; t < 16; ++t) wv[t] = half2(float2(q4x2(q[t / 4], t % 4)) * cb.x + cb.y);
    }
    device const half2* xr = xp + (((it + cx) * 2 + ex) * 4 + rp) * 16;
    for (uint t = 0; t < 16; ++t) {
      simdgroup_half8x8 A, B;
      const half2 xv = xr[t];
      A.thread_elements()[0] = wv[t].x;
      A.thread_elements()[1] = wv[t].y;
      B.thread_elements()[0] = xv.x;
      B.thread_elements()[1] = xv.y;
      simdgroup_multiply_accumulate(acc, A, B, acc);
    }
  }
  const uint nn = n0 + fm;
  if (nn >= a.N) return;
  const float c[2] = {acc.thread_elements()[0] * a.scale, acc.thread_elements()[1] * a.scale};
  for (uint e = 0; e < 2; ++e) {
    const uint r = fn + e;
    if (r >= a.rows) continue;
    device float* p = y + (ulong)r * a.ldy + nn;
    *p = a.add ? *p + c[e] : c[e];
  }
}
template [[host_name("gemv8_nvfp4")]] kernel void gemv8<0>(device const half2*, device const uchar*, device const uchar*,
                                                          device float*, constant GemvArgs&, uint2, uint, uint);
template [[host_name("gemv8_q4")]] kernel void gemv8<1>(device const half2*, device const uchar*, device const uchar*,
                                                       device float*, constant GemvArgs&, uint2, uint, uint);

// FP8 [N][K] -> Q4: one thread per group of 64 values. Searches a shrunken [min, max] range for the least squared
// error, refines (scale, bias) by least squares on the chosen codes, and quantizes with the half-rounded pair.
kernel void q4_quant(device const uchar* src [[buffer(0)]], device uchar* w [[buffer(1)]],
                     device half2* sb [[buffer(2)]], constant uint& ngroups [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= ngroups) return;
  device const uint4* sr = (device const uint4*)(src + (ulong)i * 64);
  float v[64];
  for (uint k = 0; k < 4; ++k) {
    const uint4 q = sr[k];
    for (uint t = 0; t < 4; ++t) {
      const float2 a = fp8x2(q[t]) * kFp8Unscale, b = fp8x2(q[t] >> 16) * kFp8Unscale;
      v[k * 16 + t * 4 + 0] = a.x;
      v[k * 16 + t * 4 + 1] = a.y;
      v[k * 16 + t * 4 + 2] = b.x;
      v[k * 16 + t * 4 + 3] = b.y;
    }
  }
  float mn = v[0], mx = v[0];
  for (uint k = 1; k < 64; ++k) {
    mn = min(mn, v[k]);
    mx = max(mx, v[k]);
  }
  float bs = 1.0f, bb = mn, be = INFINITY;
  if (mx > mn) {
    for (uint f = 0; f < 16; ++f) {
      const float sh = 1.0f - 0.0125f * f;
      const float lo = mn * sh, sc = (mx * sh - lo) / 15.0f;
      if (!(sc > 0)) continue;
      // candidate: the shrunken range, then one least-squares refinement on its codes
      for (uint r = 0; r < 2; ++r) {
        float s = sc, b = lo;
        if (r == 1) {
          float Sq = 0, Sqq = 0, Sv = 0, Sqv = 0;
          for (uint k = 0; k < 64; ++k) {
            const float q = clamp(rint((v[k] - lo) / sc), 0.0f, 15.0f);
            Sq += q;
            Sqq += q * q;
            Sv += v[k];
            Sqv += q * v[k];
          }
          const float det = 64.0f * Sqq - Sq * Sq;
          if (!(det > 0)) continue;
          s = (64.0f * Sqv - Sq * Sv) / det;
          b = (Sv - s * Sq) / 64.0f;
          if (!(s > 0)) continue;
        }
        s = float(half(s));
        b = float(half(b));
        if (!(s > 0)) continue;
        float e = 0;
        for (uint k = 0; k < 64; ++k) {
          const float q = clamp(rint((v[k] - b) / s), 0.0f, 15.0f);
          const float d = q * s + b - v[k];
          e += d * d;
        }
        if (e < be) {
          be = e;
          bs = s;
          bb = b;
        }
      }
    }
  }
  if (!(be < INFINITY)) {  // constant group (or no valid candidate): every code 0, bias = the value
    bs = 1.0f;
    bb = float(half(mn));
  }
  sb[i] = half2(half(bs), half(bb));
  device uint* wo = (device uint*)(w + (ulong)i * 32);
  for (uint word = 0; word < 8; ++word) {
    uint o = 0;
    for (uint j = 0; j < 4; ++j) {
      const uint q0 = uint(clamp(rint((v[word * 8 + 2 * j] - bb) / bs), 0.0f, 15.0f));
      const uint q1 = uint(clamp(rint((v[word * 8 + 2 * j + 1] - bb) / bs), 0.0f, 15.0f));
      o |= (q0 << (4 * j)) | (q1 << (16 + 4 * j));
    }
    wo[word] = o;
  }
}

// BF16 rows: each lane takes 16 bytes (8 values) a step.
kernel void gemv_bf16(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                      device float* y [[buffer(2)]], constant GemvArgs& a [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], FOLD_ARGS) {
  threadgroup float ftp[8];
  const uint n = tg.x * 8 + sg;
  if (n >= a.N) return;
  const uint rows = min(kRows, a.rows - tg.y * 8);
  x += (ulong)tg.y * 8 * a.ldx;
  y += (ulong)tg.y * 8 * a.ldy;
  device const uint4* wr = (device const uint4*)(w + (ulong)n * a.K * 2);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint steps = a.K / 8;
  for (uint s = lane; s < steps; s += 32) {
    const uint4 q = wr[s];
    for (uint m = 0; m < kRows; ++m) {
      if (m >= rows) break;
      device const float4* xr = (device const float4*)(x + (ulong)m * a.ldx + s * 8);
      const float4 xa = xr[0], xb = xr[1];
      acc[m] += dot(bf16x2(q[0]), xa.xy) + dot(bf16x2(q[1]), xa.zw) + dot(bf16x2(q[2]), xb.xy) +
                dot(bf16x2(q[3]), xb.zw);
    }
  }
  fold_scale(acc, fpart, a, lane, sg, ftp);
  gemv_store(y, a, rows, n, acc, 1.0f, lane, ftp, fnw, fxw, fpo, sg, tg.x);
}

// ---- embedding, norms, elementwise ----------------------------------------------------------------

kernel void embed_bf16(device const uint* ids [[buffer(0)]], device const ushort* emb [[buffer(1)]],
                       device float* h [[buffer(2)]], constant uint& dim [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dim) return;
  const ulong row = ids[gid.y];
  h[(ulong)gid.y * dim + gid.x] = as_type<float>(uint(emb[row * dim + gid.x]) << 16);
}

struct NormArgs {
  uint n;          // row length
  uint rpt;        // rows a token (heads)
  uint ldt, ldh;   // token stride and head stride of x (and z)
  uint ldo_t, ldo_h;
  float eps;
  uint mode;       // 0: x*rms*(1+w); 1: x*rms*w*silu(z); 2: x/|x| * scale (no weight); 3: x*rms*w
  float scale;
  uint half_out;   // 1: also write half(out) to oh at the same offset (the next matmul's A, see x_half)
  uint tvalid;     // with half_out: rows of tokens >= tvalid (the pad to 64) only zero oh
};

// One threadgroup (256 threads) a row; x and out may alias (in place).
kernel void rmsnorm_rows(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                         device const float* z [[buffer(2)]], device float* out [[buffer(3)]],
                         constant NormArgs& a [[buffer(4)]], device half* oh [[buffer(5)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const ulong xo = (ulong)(row / a.rpt) * a.ldt + (ulong)(row % a.rpt) * a.ldh;
  const ulong oo = (ulong)(row / a.rpt) * a.ldo_t + (ulong)(row % a.rpt) * a.ldo_h;
  if (a.half_out && row / a.rpt >= a.tvalid) {
    for (uint i = tid; i < a.n; i += 256) oh[oo + i] = half(0);
    return;
  }
  float ss = 0;
  for (uint i = tid; i < a.n; i += 256) {
    const float v = x[xo + i];
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0;
  for (uint i = 0; i < 8; ++i) tot += part[i];
  float r;
  if (a.mode == 2) r = a.scale * rsqrt(tot + a.eps);
  else r = rsqrt(tot / float(a.n) + a.eps);
  for (uint i = tid; i < a.n; i += 256) {
    float v = x[xo + i] * r;
    if (a.mode != 2) {
      const float wv = w[i];
      if (a.mode == 0) v *= 1.0f + wv;
      else v *= wv;
      if (a.mode == 1) {
        const float zv = z[xo + i];
        v *= zv / (1.0f + exp(-zv));
      }
    }
    out[oo + i] = v;
    if (a.half_out) oh[oo + i] = half(clamp(v, -65504.0f, 65504.0f));
  }
}

kernel void silu_mul(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                     device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = g[i];
  out[i] = v / (1.0f + exp(-v)) * u[i];
}

// o[t][h*D + d] *= sigmoid(qg[t][h*2D + D + d]) (the attention output gate)
kernel void sigmoid_gate(device float* o [[buffer(0)]], device const float* qg [[buffer(1)]],
                         constant uint3& hd [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint H = hd.x, D = hd.y;
  if (gid.x >= H * D) return;
  const uint h = gid.x / D, d = gid.x % D;
  const float g = qg[(ulong)gid.y * H * 2 * D + h * 2 * D + D + d];
  o[(ulong)gid.y * H * D + gid.x] *= 1.0f / (1.0f + exp(-g));
}

// ---- attention ----------------------------------------------------------------------------------

struct RopeArgs {
  uint heads, ldt, ldh;  // rows a token, token stride, head stride (floats)
  uint half_rot;         // rotated pairs (i, i + half_rot), i < half_rot
  uint pos0;             // position of the first token
};

// x[.. + i], x[.. + i + half] rotated by the host's cos/sin table [pos][half_rot].
kernel void rope_rows(device float* x [[buffer(0)]], device const float* cs [[buffer(1)]],
                      device const float* sn [[buffer(2)]], constant RopeArgs& a [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  if (i >= a.half_rot) return;
  const uint t = gid.y / a.heads, h = gid.y % a.heads;
  device float* p = x + (ulong)t * a.ldt + (ulong)h * a.ldh;
  const ulong ti = (ulong)(a.pos0 + t) * a.half_rot + i;
  const float c = cs[ti], s = sn[ti];
  const float x1 = p[i], x2 = p[i + a.half_rot];
  p[i] = x1 * c - x2 * s;
  p[i + a.half_rot] = x2 * c + x1 * s;
}

// k, v rows [T][KV*D] f32 -> the layer's cache [pos][KV*D] bfloat
kernel void kv_write(device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
                     device bfloat* kc [[buffer(2)]], device bfloat* vc [[buffer(3)]],
                     constant uint2& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const uint width = a.x, pos0 = a.y;
  if (gid.x >= width) return;
  const ulong src = (ulong)gid.y * width + gid.x, dst = (ulong)(pos0 + gid.y) * width + gid.x;
  kc[dst] = bfloat(k[src]);
  vc[dst] = bfloat(v[src]);
}

struct AttnArgs {
  uint H, KV, D;     // query heads, kv heads, head dim (D == 256: 8 dims a lane)
  uint ldq;          // token stride of q (floats); q head h at h * qh
  uint qh;           // head stride of q
  uint pos0;         // position of the first query row; row t sees keys 0 .. pos0 + t
  uint chunk;        // keys a threadgroup
  uint nchunks;
  float scale;
};

// Partial softmax-attention over one key chunk: grid (nchunks, H, T), 256 threads; part = [T][H][nchunks] x (m, l, D).
kernel void attn_partial(device const float* q [[buffer(0)]], device const bfloat* kc [[buffer(1)]],
                         device const bfloat* vc [[buffer(2)]], device float* part [[buffer(3)]],
                         constant AttnArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x, h = tg.y, t = tg.z;
  const uint kvh = h / (a.H / a.KV);
  const uint last = a.pos0 + t;  // inclusive
  const uint j0 = c * a.chunk;
  const uint j1 = min(j0 + a.chunk, last + 1);
  device const float* qr = q + (ulong)t * a.ldq + (ulong)h * a.qh + lane * 8;
  float qv[8];
  for (uint i = 0; i < 8; ++i) qv[i] = qr[i] * a.scale;
  float m = -INFINITY, l = 0, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint W = a.KV * a.D;
  for (uint j = j0 + sg; j < j1; j += 8) {
    device const bfloat* kr = kc + (ulong)j * W + kvh * a.D + lane * 8;
    float s = 0;
    for (uint i = 0; i < 8; ++i) s += qv[i] * float(kr[i]);
    s = simd_sum(s);
    const float mn = max(m, s);
    const float corr = exp(m - mn), p = exp(s - mn);
    device const bfloat* vr = vc + (ulong)j * W + kvh * a.D + lane * 8;
    for (uint i = 0; i < 8; ++i) acc[i] = acc[i] * corr + p * float(vr[i]);
    l = l * corr + p;
    m = mn;
  }
  threadgroup float tm[8], tl[8];
  threadgroup float tacc[8][256];
  if (lane == 0) {
    tm[sg] = m;
    tl[sg] = l;
  }
  for (uint i = 0; i < 8; ++i) tacc[sg][lane * 8 + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint d = sg * 32 + lane;  // 256 threads, one output dim each
  float M = -INFINITY;
  for (uint g = 0; g < 8; ++g) M = max(M, tm[g]);
  float L = 0, A = 0;
  if (M > -INFINITY) {
    for (uint g = 0; g < 8; ++g) {
      const float f = tm[g] > -INFINITY ? exp(tm[g] - M) : 0.0f;
      L += tl[g] * f;
      A += tacc[g][d] * f;
    }
  }
  device float* pr = part + (((ulong)t * a.H + h) * a.nchunks + c) * (2 + a.D);
  if (d == 0) {
    pr[0] = M;
    pr[1] = L;
  }
  pr[2 + d] = A;
}

// Combine the chunks: grid (H, T), 256 threads (one per dim); out [T][H*D].
kernel void attn_combine(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                         constant AttnArgs& a [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint2 tid [[thread_position_in_threadgroup]]) {
  const uint h = tg.x, t = tg.y, d = tid.x;
  device const float* pr = part + ((ulong)t * a.H + h) * a.nchunks * (2 + a.D);
  float M = -INFINITY;
  for (uint c = 0; c < a.nchunks; ++c) M = max(M, pr[c * (2 + a.D)]);
  float L = 0, A = 0;
  for (uint c = 0; c < a.nchunks; ++c) {
    device const float* p = pr + c * (2 + a.D);
    if (p[0] == -INFINITY) continue;
    const float f = exp(p[0] - M);
    L += p[1] * f;
    A += p[2 + d] * f;
  }
  out[(ulong)t * a.H * a.D + h * a.D + d] = A / L;
}

// GQA-fused partial attention (decode and short rows): one threadgroup serves the G query heads of one kv head over
// one key chunk, so each K/V row is read once instead of G times. Grid (nchunks, KV, T), 256 threads; D == 256
// (8 dims a lane). Each simdgroup takes blocks of 4 consecutive keys (loads issued together, one max/rescale a
// block); scores are kept in the log2 domain. Writes the same part layout as attn_partial (m in natural log units),
// so attn_combine folds the chunks unchanged. Q8: the int8 KV cache option (int8 [pos][KV*D] with one f32 scale a
// row and kv head, ks/vs [pos][KV]); otherwise bfloat.
template <bool Q8>
inline void kv_row8(device const uchar* c, device const float* cs, uint j, uint W, uint KV, uint kvh, uint off,
                    thread float* o) {
  if (Q8) {
    const uint2 r = *(device const uint2*)(c + (ulong)j * W + off);
    const float sc = cs[(ulong)j * KV + kvh];
    const float4 a = float4(as_type<char4>(r.x)) * sc, b = float4(as_type<char4>(r.y)) * sc;
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
  } else {
    const uint4 r = *(device const uint4*)(c + ((ulong)j * W + off) * 2);
    const float2 a = bf16x2(r.x), b = bf16x2(r.y), d = bf16x2(r.z), e = bf16x2(r.w);
    o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y; o[4] = d.x; o[5] = d.y; o[6] = e.x; o[7] = e.y;
  }
}

template <uint G, bool Q8>
kernel void attn_gqa(device const float* q [[buffer(0)]], device const uchar* kc [[buffer(1)]],
                     device const uchar* vc [[buffer(2)]], device float* part [[buffer(3)]],
                     constant AttnArgs& a [[buffer(4)]], device const float* ks [[buffer(5)]],
                     device const float* vs [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x, kvh = tg.y, t = tg.z;
  const uint last = a.pos0 + t;  // inclusive
  const uint j0 = c * a.chunk;
  const uint j1 = min(j0 + a.chunk, last + 1);
  const float qs = a.scale * M_LOG2E_F;
  float qv[G][8];
  for (uint g = 0; g < G; ++g) {
    device const float4* qr = (device const float4*)(q + (ulong)t * a.ldq + (ulong)(kvh * G + g) * a.qh + lane * 8);
    const float4 q0 = qr[0], q1 = qr[1];
    qv[g][0] = q0.x * qs; qv[g][1] = q0.y * qs; qv[g][2] = q0.z * qs; qv[g][3] = q0.w * qs;
    qv[g][4] = q1.x * qs; qv[g][5] = q1.y * qs; qv[g][6] = q1.z * qs; qv[g][7] = q1.w * qs;
  }
  float m[G], l[G], acc[G][8];
  for (uint g = 0; g < G; ++g) {
    m[g] = -INFINITY;
    l[g] = 0;
    for (uint i = 0; i < 8; ++i) acc[g][i] = 0;
  }
  const uint W = a.KV * a.D, off = kvh * a.D + lane * 8;
  for (uint jb = j0 + sg * 4; jb < j1; jb += 32) {
    float kr[4][8];
    for (uint u = 0; u < 4; ++u) kv_row8<Q8>(kc, ks, min(jb + u, j1 - 1), W, a.KV, kvh, off, kr[u]);
    float s[G][4];
    for (uint u = 0; u < 4; ++u)
      for (uint g = 0; g < G; ++g) {
        float d = 0;
        for (uint i = 0; i < 8; ++i) d += qv[g][i] * kr[u][i];
        s[g][u] = simd_sum(d);
      }
    float vr[4][8];
    for (uint u = 0; u < 4; ++u) kv_row8<Q8>(vc, vs, min(jb + u, j1 - 1), W, a.KV, kvh, off, vr[u]);
    const uint nv = min(4u, j1 - jb);
    for (uint g = 0; g < G; ++g) {
      float bm = s[g][0];
      for (uint u = 1; u < nv; ++u) bm = max(bm, s[g][u]);
      const float mn = max(m[g], bm);
      const float corr = exp2(m[g] - mn);
      float p[4];
      for (uint u = 0; u < 4; ++u) p[u] = u < nv ? exp2(s[g][u] - mn) : 0.0f;
      l[g] = l[g] * corr + p[0] + p[1] + p[2] + p[3];
      m[g] = mn;
      for (uint i = 0; i < 8; ++i)
        acc[g][i] = acc[g][i] * corr + p[0] * vr[0][i] + p[1] * vr[1][i] + p[2] * vr[2][i] + p[3] * vr[3][i];
    }
  }
  threadgroup float tm[8], tl[8];
  threadgroup float tacc[8][256];
  const uint d = sg * 32 + lane;  // 256 threads, one output dim each
  for (uint g = 0; g < G; ++g) {
    if (g) threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      tm[sg] = m[g];
      tl[sg] = l[g];
    }
    for (uint i = 0; i < 8; ++i) tacc[sg][lane * 8 + i] = acc[g][i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -INFINITY;
    for (uint r = 0; r < 8; ++r) M = max(M, tm[r]);
    float L = 0, A = 0;
    if (M > -INFINITY) {
      for (uint r = 0; r < 8; ++r) {
        const float f = tm[r] > -INFINITY ? exp2(tm[r] - M) : 0.0f;
        L += tl[r] * f;
        A += tacc[r][d] * f;
      }
    }
    device float* pr = part + (((ulong)t * a.H + kvh * G + g) * a.nchunks + c) * (2 + a.D);
    if (d == 0) {
      pr[0] = M * M_LN2_F;
      pr[1] = L;
    }
    pr[2 + d] = A;
  }
}
template [[host_name("attn_gqa6")]] kernel void attn_gqa<6, false>(device const float*, device const uchar*,
                                                                   device const uchar*, device float*,
                                                                   constant AttnArgs&, device const float*,
                                                                   device const float*, uint3, uint, uint);
template [[host_name("attn_gqa6_q8")]] kernel void attn_gqa<6, true>(device const float*, device const uchar*,
                                                                     device const uchar*, device float*,
                                                                     constant AttnArgs&, device const float*,
                                                                     device const float*, uint3, uint, uint);

// int8 KV cache write: grid (KV, T) threadgroups of D == 256 threads; symmetric per row and kv head (absmax / 127).
kernel void kv_write8(device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
                      device char* kc [[buffer(2)]], device char* vc [[buffer(3)]], device float* ks [[buffer(4)]],
                      device float* vs [[buffer(5)]], constant uint4& a [[buffer(6)]],  // (KV, D, pos0, -)
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kvh = tg.x, t = tg.y, W = a.x * a.y;
  const ulong src = (ulong)t * W + kvh * a.y + tid, dst = (ulong)(a.z + t) * W + kvh * a.y + tid;
  const float kx = k[src], vx = v[src];
  threadgroup float rk[8], rv[8];
  const float mk = simd_max(fabs(kx)), mv = simd_max(fabs(vx));
  if (lane == 0) {
    rk[sg] = mk;
    rv[sg] = mv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ak = 0, av = 0;
  for (uint i = 0; i < 8; ++i) {
    ak = max(ak, rk[i]);
    av = max(av, rv[i]);
  }
  const float sk = ak > 0 ? ak / 127.0f : 1.0f, sv = av > 0 ? av / 127.0f : 1.0f;
  kc[dst] = char(clamp(rint(kx / sk), -127.0f, 127.0f));
  vc[dst] = char(clamp(rint(vx / sv), -127.0f, 127.0f));
  if (tid == 0) {
    const ulong si = (ulong)(a.z + t) * a.x + kvh;
    ks[si] = sk;
    vs[si] = sv;
  }
}

// int8 KV -> bfloat scratch for the prefill matmuls: grid (KV*D, kvp); rows >= n zero.
kernel void kv_dq8(device const char* kc [[buffer(0)]], device const char* vc [[buffer(1)]],
                   device const float* ks [[buffer(2)]], device const float* vs [[buffer(3)]],
                   device bfloat* ko [[buffer(4)]], device bfloat* vo [[buffer(5)]],
                   constant uint4& a [[buffer(6)]],  // (KV, D, n, -)
                   uint2 gid [[thread_position_in_grid]]) {
  const uint W = a.x * a.y;
  if (gid.x >= W) return;
  const ulong i = (ulong)gid.y * W + gid.x, si = (ulong)gid.y * a.x + gid.x / a.y;
  const bool in = gid.y < a.z;
  ko[i] = bfloat(in ? float(kc[i]) * ks[si] : 0.0f);
  vo[i] = bfloat(in ? float(vc[i]) * vs[si] : 0.0f);
}

// ---- gated delta net ----------------------------------------------------------------------------

// Causal depthwise conv (kernel 4) over [state(3), x(T)] then SiLU; state keeps the last 3 inputs. One thread a channel.
kernel void gdn_conv(device float* x [[buffer(0)]], device float* state [[buffer(1)]],
                     device const float* w [[buffer(2)]], constant uint2& a [[buffer(3)]],
                     uint c [[thread_position_in_grid]]) {
  const uint C = a.x, T = a.y;
  if (c >= C) return;
  const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  float s0 = state[c], s1 = state[C + c], s2 = state[2 * C + c];
  for (uint t = 0; t < T; ++t) {
    const float v = x[(ulong)t * C + c];
    const float o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
    x[(ulong)t * C + c] = o / (1.0f + exp(-o));
    s0 = s1;
    s1 = s2;
    s2 = v;
  }
  state[c] = s0;
  state[C + c] = s1;
  state[2 * C + c] = s2;
}

// g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b); in place over a [T][H] and b [T][H].
kernel void gdn_gates(device float* av [[buffer(0)]], device float* bv [[buffer(1)]],
                      device const float* A_log [[buffer(2)]], device const float* dt_bias [[buffer(3)]],
                      constant uint2& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  const uint H = a.x, T = a.y;
  if (i >= H * T) return;
  const uint h = i % H;
  const float A = exp(A_log[h]);
  const float x = av[i] + dt_bias[h];
  const float sp = x > 20.0f ? x : log(1.0f + exp(x));
  av[i] = exp(-A * sp);
  bv[i] = 1.0f / (1.0f + exp(-bv[i]));
}

struct GdnArgs {
  uint T, Hk, Hv, Dk, Dv;  // Dk == 128: 4 k dims a lane
  uint ldc;                // token stride of the conv output (q | k | v)
};

// The recurrence, one simdgroup a state row (v head h, row r): S_r *= g; kv = S_r.k; S_r += (v_r - kv) beta k;
// y_r = S_r.q. q and k are normalized here (l2, q times Dk^-0.5). grid: Hv*Dv/8 threadgroups of 256.
kernel void gdn_recurrent(device const float* conv [[buffer(0)]], device const float* g [[buffer(1)]],
                          device const float* beta [[buffer(2)]], device float* S [[buffer(3)]],
                          device float* y [[buffer(4)]], constant GdnArgs& a [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 8 + sg;  // h * Dv + r
  const uint h = row / a.Dv, r = row % a.Dv;
  const uint hk = h / (a.Hv / a.Hk);
  device float* Sr = S + (ulong)row * a.Dk + lane * 4;
  float4 s = *(device float4*)Sr;
  const float qscale = rsqrt(float(a.Dk));
  for (uint t = 0; t < a.T; ++t) {
    device const float* ct = conv + (ulong)t * a.ldc;
    float4 q = *(device const float4*)(ct + hk * a.Dk + lane * 4);
    float4 k = *(device const float4*)(ct + a.Hk * a.Dk + hk * a.Dk + lane * 4);
    const float vr = ct[2 * a.Hk * a.Dk + h * a.Dv + r];
    const float qn = simd_sum(dot(q, q)), kn = simd_sum(dot(k, k));
    q *= qscale * rsqrt(qn + 1e-6f);
    k *= rsqrt(kn + 1e-6f);
    s *= g[t * a.Hv + h];
    const float kv = simd_sum(dot(s, k));
    s += k * ((vr - kv) * beta[t * a.Hv + h]);
    const float o = simd_sum(dot(s, q));
    if (lane == 0) y[(ulong)t * a.Hv * a.Dv + row] = o;
  }
  *(device float4*)Sr = s;
}

// ---- prefill: Metal 4 tensor-op matmul (MetalPerformancePrimitives) ---------------------------------------------
// Prefill chunks of 64+ rows run the projections and attention as tiled matmuls on the GPU's matrix units: the
// NVFP4/FP8 weights are expanded to half once a chunk (exact: e2m1 x e4m3 and e4m3 fit a half), activations are
// rounded to half, accumulation is f32. Attention runs as S = Q K^T, a masked softmax, O = P V in bfloat (the
// KV cache's type). Every extent is padded to the 64x64 tile.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmArgs {
  uint M, N, K;        // C[M][N] = A[M][K] B, all multiples of 64
  uint lda, ldb, ldc;  // row strides (elements)
  uint sa, sb, sc;     // batch strides (elements); batch z reads B batch z / bdiv (grouped-query attention)
  uint bdiv;
};

// B is [N][K] (C = A B^T: weights, keys)
template <typename T>
kernel void mm_nt(device T* A [[buffer(0)]], device T* B [[buffer(1)]], device float* C [[buffer(2)]],
                  constant MmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
  device T* Ab = A + (ulong)tg.z * a.sa;
  device T* Bb = B + (ulong)(tg.z / a.bdiv) * a.sb;
  device float* Cb = C + (ulong)tg.z * a.sc;
  auto tA = tensor<device T, dextents<int32_t, 2>, tensor_inline>(Ab, dextents<int32_t, 2>(a.K, a.M),
                                                                  array<int32_t, 2>{1, int(a.lda)});
  auto tB = tensor<device T, dextents<int32_t, 2>, tensor_inline>(Bb, dextents<int32_t, 2>(a.K, a.N),
                                                                  array<int32_t, 2>{1, int(a.ldb)});
  auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Cb, dextents<int32_t, 2>(a.N, a.M),
                                                                      array<int32_t, 2>{1, int(a.ldc)});
  constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<d, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, tg.y * 64);
  auto mB = tB.slice(0, tg.x * 64);
  auto mC = tC.slice(tg.x * 64, tg.y * 64);
  op.run(mA, mB, mC);
}

// B is [K][N] (C = A B: probabilities times values)
template <typename T>
kernel void mm_nn(device T* A [[buffer(0)]], device T* B [[buffer(1)]], device float* C [[buffer(2)]],
                  constant MmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
  device T* Ab = A + (ulong)tg.z * a.sa;
  device T* Bb = B + (ulong)(tg.z / a.bdiv) * a.sb;
  device float* Cb = C + (ulong)tg.z * a.sc;
  auto tA = tensor<device T, dextents<int32_t, 2>, tensor_inline>(Ab, dextents<int32_t, 2>(a.K, a.M),
                                                                  array<int32_t, 2>{1, int(a.lda)});
  auto tB = tensor<device T, dextents<int32_t, 2>, tensor_inline>(Bb, dextents<int32_t, 2>(a.N, a.K),
                                                                  array<int32_t, 2>{1, int(a.ldb)});
  auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Cb, dextents<int32_t, 2>(a.N, a.M),
                                                                      array<int32_t, 2>{1, int(a.ldc)});
  constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false, false);
  matmul2d<d, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, tg.y * 64);
  auto mB = tB.slice(tg.x * 64, 0);
  auto mC = tC.slice(tg.x * 64, tg.y * 64);
  op.run(mA, mB, mC);
}

template [[host_name("mm_nt_h")]] kernel void mm_nt<half>(device half*, device half*, device float*,
                                                          constant MmArgs&, uint3);
template [[host_name("mm_nt_b")]] kernel void mm_nt<bfloat>(device bfloat*, device bfloat*, device float*,
                                                            constant MmArgs&, uint3);
template [[host_name("mm_nn_b")]] kernel void mm_nn<bfloat>(device bfloat*, device bfloat*, device float*,
                                                            constant MmArgs&, uint3);

// Prefill weight matmul with the store fused in: y[t][n] = scale * c (mode 0), y + scale * c (mode 1) or
// silu(y) * scale * c (mode 2, the MLP up projection writing over the gate) for t < T; c = A[t][:] . W[n][:].
// A is half [Mp][K] (Mp a multiple of BM), W half [N][K]. The y tensor's extents (N, T) clip the tail rows.
struct EpArgs {
  uint M, N, K, T, ldy, mode;
  float scale;
};
template <uint BM, uint BN, uint NSG>
kernel void mm_ep(device half* A [[buffer(0)]], device half* W [[buffer(1)]], device float* y [[buffer(2)]],
                  constant EpArgs& a [[buffer(3)]], device half* yh [[buffer(4)]],
                  uint2 tg [[threadgroup_position_in_grid]]) {
  auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(a.K, a.M));
  auto tW = tensor<device half, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(a.K, a.N));
  auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y, dextents<int32_t, 2>(a.N, a.T),
                                                                      array<int32_t, 2>{1, int(a.ldy)});
  constexpr auto d = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<d, execution_simdgroups<NSG>> op;
  auto mA = tA.slice(0, tg.y * BM);
  auto mW = tW.slice(0, tg.x * BN);
  auto mY = tY.slice(tg.x * BN, tg.y * BM);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mW), float>();
  op.run(mA, mW, cT);
  if (a.mode == 0) {
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
      if (cT.is_valid_element(i)) cT[i] *= a.scale;
  } else {
    auto yT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mW), float>();
    yT.load(mY);
    if (a.mode == 1) {
#pragma unroll
      for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = a.scale * cT[i] + yT[i];
    } else {
#pragma unroll
      for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) {
          const float g = yT[i];
          cT[i] = g / (1.0f + exp(-g)) * (a.scale * cT[i]);
        }
      if (a.mode == 3) {
        // the half copy only, [M][N] with zero rows from T: the next matmul's A (the MLP down projection)
#pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
          if (cT.is_valid_element(i)) {
            const auto ix = cT.get_multidimensional_index(i);
            const uint r = tg.y * BM + uint(ix[1]);
            yh[(ulong)r * a.N + tg.x * BN + uint(ix[0])] = r < a.T ? half(clamp(cT[i], -65504.0f, 65504.0f)) : half(0);
          }
        return;
      }
    }
  }
  cT.store(mY);
}
template [[host_name("mm_ep_64x128")]] kernel void mm_ep<64, 128, 4>(device half*, device half*, device float*,
                                                                     constant EpArgs&, device half*, uint2);
template [[host_name("mm_ep_64x64")]] kernel void mm_ep<64, 64, 4>(device half*, device half*, device float*,
                                                                   constant EpArgs&, device half*, uint2);

// NVFP4 [N][K] -> half [N][K] (without the global scale): one thread per 32 values.
kernel void dq_nvfp4(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                     device half* out [[buffer(2)]], constant uint& n32 [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= n32) return;
  const uint4 q = ((device const uint4*)w)[i];
  const uint sv = ((device const ushort*)sc)[i];
  const float s0 = e4m3(sv & 0xFFu) * kFp4Unscale, s1 = e4m3(sv >> 8) * kFp4Unscale;
  device half2* o = (device half2*)(out + (ulong)i * 32);
  for (uint k = 0; k < 4; ++k)
    for (uint j = 0; j < 4; ++j) o[k * 4 + j] = half2(fp4x2(q[k] >> (8 * j)) * (k < 2 ? s0 : s1));
}

// FP8 [N][K] -> half [N][K] (without the per-tensor scale): one thread per 16 values.
kernel void dq_fp8(device const uchar* w [[buffer(0)]], device half* out [[buffer(1)]],
                   constant uint& n16 [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n16) return;
  const uint4 q = ((device const uint4*)w)[i];
  device half2* o = (device half2*)(out + (ulong)i * 16);
  for (uint k = 0; k < 4; ++k) {
    o[2 * k] = half2(fp8x2(q[k]) * kFp8Unscale);
    o[2 * k + 1] = half2(fp8x2(q[k] >> 16) * kFp8Unscale);
  }
}

// x [T][ldx] f32 -> half [Mpad][K] (rows >= T zero; saturated to the half range).
kernel void x_half(device const float* x [[buffer(0)]], device half* out [[buffer(1)]],
                   constant uint4& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {  // a = (K, T, ldx, -)
  if (gid.x >= a.x) return;
  const float v = gid.y < a.y ? x[(ulong)gid.y * a.z + gid.x] : 0.0f;
  out[(ulong)gid.y * a.x + gid.x] = half(clamp(v, -65504.0f, 65504.0f));
}

// y[t][n] (=|+=) scale * c[t][n] for t < T.
kernel void mm_store(device const float* c [[buffer(0)]], device float* y [[buffer(1)]],
                     constant uint4& a [[buffer(2)]], constant float2& f [[buffer(3)]],  // a = (N, T, ldy, add)
                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= a.x || gid.y >= a.y) return;
  const float r = c[(ulong)gid.y * a.x + gid.x] * f.x;
  device float* p = y + (ulong)gid.y * a.z + gid.x;
  *p = a.w ? *p + r : r;
}

struct PrefAttnArgs {
  uint T, Tp, H, D, ldq, qh, pos0, kvp;  // kvp: padded key count (row stride of S and P)
  float scale;
};

// q [T][ldq] (head h at h * qh) -> bfloat [H][Tp][D] times the softmax scale (rows >= T zero).
kernel void q_bf16(device const float* q [[buffer(0)]], device bfloat* out [[buffer(1)]],
                   constant PrefAttnArgs& a [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
  const uint d = gid.x, t = gid.y, h = gid.z;
  if (d >= a.D) return;
  const float v = t < a.T ? q[(ulong)t * a.ldq + h * a.qh + d] * a.scale : 0.0f;
  out[((ulong)h * a.Tp + t) * a.D + d] = bfloat(v);
}

// q [T][ldq] -> bfloat [KV][Tp * G][D] times the softmax scale, row t * G + g = query head kvh * G + g at token t
// (rows of tokens >= T zero): the grouped layout of attn_flash. Grid (D, Tp, H).
kernel void q_bf16g(device const float* q [[buffer(0)]], device bfloat* out [[buffer(1)]],
                    constant PrefAttnArgs& a [[buffer(2)]], constant uint& G [[buffer(3)]],
                    uint3 gid [[thread_position_in_grid]]) {
  const uint d = gid.x, t = gid.y, h = gid.z;
  if (d >= a.D) return;
  const float v = t < a.T ? q[(ulong)t * a.ldq + h * a.qh + d] * a.scale : 0.0f;
  out[(((ulong)(h / G) * a.Tp + t) * G + h % G) * a.D + d] = bfloat(v);
}

// Flash prefill attention: the G query heads of one kv head are the rows r = t * G + g of one [Tp * G][256] block, so
// a 64-row tile reads each key and value once for all its heads. Per 64-key block: S = Q K^T into threadgroup memory
// (tensor-op matmul), the online softmax in the log2 domain by simdgroups, P (bfloat) into threadgroup memory, and
// O += P V into the matmul's cooperative destination, rescaled per row first. Nothing of size T x keys reaches
// device memory. Grid (Tp * G / 64, KV), 32 * NSG threads. Writes o [Tp][H * 256] (rows of tokens >= T zero).
template <uint G, uint NSG, uint BR, uint BC>
kernel void attn_flash(device bfloat* qg [[buffer(0)]], device bfloat* kc [[buffer(1)]],
                       device bfloat* vc [[buffer(2)]], device float* o [[buffer(3)]],
                       constant PrefAttnArgs& a [[buffer(4)]], constant uint& dbg [[buffer(5)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 256, RPS = BR / NSG, E = (BC + 31) / 32;
  threadgroup float sS[BR * BC];
  threadgroup bfloat sP[BR * BC];
  threadgroup float sm[BR], sl[BR], sa[BR];
  const uint kvh = tg.y, r0 = tg.x * BR, rows = a.Tp * G, kw = (a.H / G) * D;
  auto tQ = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
      qg + (ulong)kvh * rows * D, dextents<int32_t, 2>(D, rows), array<int32_t, 2>{1, int(D)});
  auto tK = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(kc + kvh * D, dextents<int32_t, 2>(D, a.kvp),
                                                                       array<int32_t, 2>{1, int(kw)});
  auto tV = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(vc + kvh * D, dextents<int32_t, 2>(D, a.kvp),
                                                                       array<int32_t, 2>{1, int(kw)});
  auto tS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sS, dextents<int32_t, 2>(BC, BR),
                                                                           array<int32_t, 2>{1, int(BC)});
  auto tP = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sP, dextents<int32_t, 2>(BC, BR),
                                                                            array<int32_t, 2>{1, int(BC)});
  constexpr auto d1 = matmul2d_descriptor(BR, BC, D, false, true, false);
  constexpr auto d2 = matmul2d_descriptor(BR, D, BC, false, false, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d1, execution_simdgroups<NSG>> op1;
  matmul2d<d2, execution_simdgroups<NSG>> op2;
  auto mQ = tQ.slice(0, r0);
  auto mV0 = tV.slice(0, 0);
  auto cO = op2.template get_destination_cooperative_tensor<decltype(tP), decltype(mV0), float>();
#pragma unroll
  for (uint16_t i = 0; i < cO.get_capacity(); ++i)
    if (cO.is_valid_element(i)) cO[i] = 0.0f;
  if (tid < BR) {
    sm[tid] = -INFINITY;
    sl[tid] = 0.0f;
  }
  const uint tfirst = r0 / G;
  const uint kend = tfirst < a.T ? a.pos0 + min((r0 + BR - 1) / G, a.T - 1) + 1 : 0;  // keys [0, kend)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j0 = 0; j0 < kend; j0 += BC) {
    auto mK = tK.slice(0, j0);
    if (dbg & 8) {
      auto cS = op1.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
      op1.run(mQ, mK, cS);
      cS.store(tS);
    } else if (!(dbg & 16)) {
      op1.run(mQ, mK, tS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!(dbg & 4)) for (uint i = 0; i < RPS; ++i) {
      const uint r = sg * RPS + i;
      const uint t = (r0 + r) / G;
      const uint n = t < a.T ? a.pos0 + t + 1 : 0;  // valid keys j < n
      float sv[E];
      float mx = -INFINITY;
      for (uint e = 0; e < E; ++e) {
        const uint c = lane + 32 * e;
        sv[e] = c < BC && j0 + c < n ? sS[r * BC + c] * M_LOG2E_F : -INFINITY;
        mx = max(mx, sv[e]);
      }
      const float mo = sm[r];
      const float mn = max(mo, simd_max(mx));
      const bool none = mn == -INFINITY;
      const float al = none ? 1.0f : exp2(mo - mn);
      float pl = 0.0f;
      for (uint e = 0; e < E; ++e) {
        const uint c = lane + 32 * e;
        const float pv = none ? 0.0f : exp2(sv[e] - mn);
        if (c < BC) sP[r * BC + c] = bfloat(pv);
        pl += c < BC ? pv : 0.0f;
      }
      const float ps = simd_sum(pl);
      if (lane == 0) {
        sm[r] = mn;
        sl[r] = sl[r] * al + ps;
        sa[r] = al;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!(dbg & 1)) {
#pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i)
      if (cO.is_valid_element(i)) {
        const auto ix = cO.get_multidimensional_index(i);
        cO[i] *= sa[ix[1]];
      }
    }
    auto mV = tV.slice(0, j0);
    if (!(dbg & 2)) op2.run(tP, mV, cO);
  }
  const uint H = a.H;
#pragma unroll
  for (uint16_t i = 0; i < cO.get_capacity(); ++i)
    if (cO.is_valid_element(i)) {
      const auto ix = cO.get_multidimensional_index(i);
      const uint r = r0 + uint(ix[1]), t = r / G, c = uint(ix[0]);
      const float l = sl[ix[1]];
      o[(ulong)t * H * D + (kvh * G + r % G) * D + c] = t < a.T && l > 0.0f ? cO[i] / l : 0.0f;
    }
}

template [[host_name("attn_flash6")]] kernel void attn_flash<6, 4, 64, 64>(device bfloat*, device bfloat*, device bfloat*, device float*, constant PrefAttnArgs&, constant uint&, uint2, uint, uint, uint);
template [[host_name("attn_flash6_8")]] kernel void attn_flash<6, 8, 64, 64>(device bfloat*, device bfloat*, device bfloat*, device float*, constant PrefAttnArgs&, constant uint&, uint2, uint, uint, uint);
template [[host_name("attn_flash6_128")]] kernel void attn_flash<6, 8, 128, 32>(device bfloat*, device bfloat*, device bfloat*, device float*, constant PrefAttnArgs&, constant uint&, uint2, uint, uint, uint);
template [[host_name("attn_flash6_192")]] kernel void attn_flash<6, 12, 192, 16>(device bfloat*, device bfloat*, device bfloat*, device float*, constant PrefAttnArgs&, constant uint&, uint2, uint, uint, uint);

// attn_flash2: as attn_flash, with the online softmax on the QK^T cooperative tile in registers (no S round trip
// through threadgroup memory), one cross-simdgroup row-max exchange per key block, per-thread partial row sums (combined
// once at the end), and a lazy rescale: a row's reference max moves only when the block max exceeds it by more than
// 2^8 (log2 domain), so O is rescaled only in the blocks where some row's max moved that far. Exact math: the final
// division uses the same reference max as the numerator. Layout of the MPP destination (measured for 64x64 and 64x128
// over 8 simdgroups; 128-key blocks were slower, register pressure): elements 4g..4g+3 of a thread are 4 consecutive columns of one row, and lanes l, l^1, l^8, l^9
// hold the same row's 16 consecutive columns starting at a multiple of 16 (a column chunk).
// SPLIT (the verify rows, T <= 10): one 64-row tile, keys split in chunks of sp.x over grid x (sp.y chunks); o is
// then the attn_combine partials [T][H][nchunks][2 + D] (m in natural log units, the row sum, the unnormalized O).
template <uint G, uint BC, bool SPLIT>
kernel void attn_flash2(device bfloat* qg [[buffer(0)]], device bfloat* kc [[buffer(1)]],
                        device bfloat* vc [[buffer(2)]], device float* o [[buffer(3)]],
                        constant PrefAttnArgs& a [[buffer(4)]], constant uint& dbg [[buffer(5)]],
                        constant uint2& sp [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 256, BR = 64, NSG = 8, NG = BR * BC / (32 * NSG * 4), NCH = BC / 16;
  constexpr float TAU = 8.0f;
  threadgroup bfloat sP[BR * BC];
  threadgroup float smx[NCH * BR];  // per column chunk row maxima, then the final per chunk row sums
  threadgroup float sa[BR];
  threadgroup uint sflag;
  const uint kvh = tg.y, r0 = SPLIT ? 0 : tg.x * BR, rows = a.Tp * G, kw = (a.H / G) * D;
  auto tQ = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
      qg + (ulong)kvh * rows * D, dextents<int32_t, 2>(D, rows), array<int32_t, 2>{1, int(D)});
  auto tK = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(kc + kvh * D, dextents<int32_t, 2>(D, a.kvp),
                                                                       array<int32_t, 2>{1, int(kw)});
  auto tV = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(vc + kvh * D, dextents<int32_t, 2>(D, a.kvp),
                                                                       array<int32_t, 2>{1, int(kw)});
  auto tP = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sP, dextents<int32_t, 2>(BC, BR),
                                                                            array<int32_t, 2>{1, int(BC)});
  constexpr auto d1 = matmul2d_descriptor(BR, BC, D, false, true, false);
  constexpr auto d2 = matmul2d_descriptor(BR, D, BC, false, false, false,
                                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d1, execution_simdgroups<NSG>> op1;
  matmul2d<d2, execution_simdgroups<NSG>> op2;
  auto mQ = tQ.slice(0, r0);
  auto mK0 = tK.slice(0, 0);
  auto mV0 = tV.slice(0, 0);
  auto cO = op2.template get_destination_cooperative_tensor<decltype(tP), decltype(mV0), float>();
#pragma unroll
  for (uint16_t i = 0; i < cO.get_capacity(); ++i)
    if (cO.is_valid_element(i)) cO[i] = 0.0f;
  auto cS = op1.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
  // this thread's element groups: row, first column, valid key count (fixed for every block); per group row state
  uint rr[NG], cc[NG], nn[NG];
  float m[NG], l[NG];
#pragma unroll
  for (uint k = 0; k < NG; ++k) {
    const auto ix = cS.get_multidimensional_index(uint16_t(4 * k));
    rr[k] = uint(ix[1]);
    cc[k] = uint(ix[0]);
    const uint t = (r0 + rr[k]) / G;
    nn[k] = t < a.T ? a.pos0 + t + 1 : 0;  // valid keys j < nn
    m[k] = -INFINITY;
    l[k] = 0.0f;
  }
  const bool wr = (lane & 9) == 0;
  if (tid == 0) sflag = ~0u;
  const uint tfirst = r0 / G;
  const uint kend = tfirst < a.T ? a.pos0 + min((r0 + BR - 1) / G, a.T - 1) + 1 : 0;  // keys [0, kend)
  const uint nmin = tfirst < a.T ? a.pos0 + tfirst + 1 : 0;  // blocks ending at or below nmin need no mask
  const uint jb = SPLIT ? tg.x * sp.x : 0, je = SPLIT ? min(kend, jb + sp.x) : kend;
  for (uint j0 = jb, it = 0; j0 < je; j0 += BC, ++it) {
    auto mK = tK.slice(0, j0);
    op1.run(mQ, mK, cS);
    const bool full = j0 + BC <= nmin;
#pragma unroll
    for (uint k = 0; k < NG; ++k) {
      float x = -INFINITY;
#pragma unroll
      for (uint e = 0; e < 4; ++e) {
        const uint i = 4 * k + e;
        const float s = (full || j0 + cc[k] + e < nn[k]) ? cS[i] * M_LOG2E_F : -INFINITY;
        cS[i] = s;
        x = max(x, s);
      }
      x = max(x, simd_shuffle_xor(x, 1));
      x = max(x, simd_shuffle_xor(x, 8));
      if (wr) smx[(cc[k] / 16) * BR + rr[k]] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool resc = false;
#pragma unroll
    for (uint k = 0; k < NG; ++k) {
      const uint r = rr[k];
      float mx = smx[r];
#pragma unroll
      for (uint c = 1; c < NCH; ++c) mx = max(mx, smx[c * BR + r]);
      float al = 1.0f;
      if (mx > m[k] + TAU) {  // also the first finite block (m = -inf); O and l are still 0 then
        if (m[k] != -INFINITY) {
          al = exp2(m[k] - mx);
          resc = true;
        }
        m[k] = mx;
      }
      float4 p4;
#pragma unroll
      for (uint e = 0; e < 4; ++e) p4[e] = m[k] == -INFINITY ? 0.0f : exp2(cS[4 * k + e] - m[k]);
      *(threadgroup bfloat4*)(sP + r * BC + cc[k]) = bfloat4(p4);
      l[k] = l[k] * al + (p4[0] + p4[1] + p4[2] + p4[3]);
      if (wr && cc[k] < 16) sa[r] = al;
    }
    if (resc) sflag = it;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sflag == it) {
#pragma unroll
      for (uint16_t i = 0; i < cO.get_capacity(); ++i)
        if (cO.is_valid_element(i)) cO[i] *= sa[cO.get_multidimensional_index(i)[1]];
    }
    auto mV = tV.slice(0, j0);
    op2.run(tP, mV, cO);
  }
  // row sums: lanes, then column chunks
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (uint k = 0; k < NG; ++k) {
    float x = l[k];
    x += simd_shuffle_xor(x, 1);
    x += simd_shuffle_xor(x, 8);
    if (wr) smx[(cc[k] / 16) * BR + rr[k]] = x;
    if (SPLIT && wr && cc[k] < 16) sa[rr[k]] = m[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint H = a.H;
  if (SPLIT) {
#pragma unroll
    for (uint16_t i = 0; i < cO.get_capacity(); ++i)
      if (cO.is_valid_element(i)) {
        const auto ix = cO.get_multidimensional_index(i);
        const uint rl = uint(ix[1]), t = rl / G, c = uint(ix[0]);
        if (t >= a.T) continue;
        device float* pr = o + (((ulong)t * H + kvh * G + rl % G) * sp.y + tg.x) * (2 + D);
        pr[2 + c] = cO[i];
        if (c == 0) {
          float s = 0.0f;
#pragma unroll
          for (uint q = 0; q < NCH; ++q) s += smx[q * BR + rl];
          pr[0] = sa[rl] == -INFINITY ? -INFINITY : sa[rl] * M_LN2_F;
          pr[1] = s;
        }
      }
    return;
  }
#pragma unroll
  for (uint16_t i = 0; i < cO.get_capacity(); ++i)
    if (cO.is_valid_element(i)) {
      const auto ix = cO.get_multidimensional_index(i);
      const uint rl = uint(ix[1]), r = r0 + rl, t = r / G, c = uint(ix[0]);
      float s = 0.0f;
#pragma unroll
      for (uint q = 0; q < NCH; ++q) s += smx[q * BR + rl];
      o[(ulong)t * H * D + (kvh * G + r % G) * D + c] = t < a.T && s > 0.0f ? cO[i] / s : 0.0f;
    }
}
template [[host_name("attn_flash2_6")]] kernel void attn_flash2<6, 64, false>(device bfloat*, device bfloat*, device bfloat*, device float*, constant PrefAttnArgs&, constant uint&, constant uint2&, uint2, uint, uint, uint);
template [[host_name("attn_flash2_6s")]] kernel void attn_flash2<6, 64, true>(device bfloat*, device bfloat*, device bfloat*, device float*, constant PrefAttnArgs&, constant uint&, constant uint2&, uint2, uint, uint, uint);

// Causal softmax over S [H][Tp][kvp] f32 -> P bfloat (masked and padded entries 0). Grid (Tp, H), 256 threads.
kernel void softmax_causal(device const float* S [[buffer(0)]], device bfloat* P [[buffer(1)]],
                           constant PrefAttnArgs& a [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint t = tg.x, h = tg.y;
  const ulong row = ((ulong)h * a.Tp + t) * a.kvp;
  const uint n = t < a.T ? a.pos0 + t + 1 : 0;
  threadgroup float red[8];
  float m = -INFINITY;
  for (uint j = tid; j < n; j += 256) m = max(m, S[row + j]);
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0];
  for (uint g = 1; g < 8; ++g) m = max(m, red[g]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float l = 0;
  for (uint j = tid; j < n; j += 256) l += exp(S[row + j] - m);
  l = simd_sum(l);
  if (lane == 0) red[sg] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  l = 0;
  for (uint g = 0; g < 8; ++g) l += red[g];
  const float inv = n ? 1.0f / l : 0.0f;
  for (uint j = tid; j < a.kvp; j += 256) P[row + j] = bfloat(j < n ? exp(S[row + j] - m) * inv : 0.0f);
}

// gdn_recurrent for prefill chunks: four state rows of one head per simdgroup, so the q/k loads and norms are shared
// and the four rows' reductions overlap. Grid Hv*Dv/32 threadgroups of 256 threads.
kernel void gdn_recurrent4(device const float* conv [[buffer(0)]], device const float* g [[buffer(1)]],
                           device const float* beta [[buffer(2)]], device float* S [[buffer(3)]],
                           device float* y [[buffer(4)]], constant GdnArgs& a [[buffer(5)]],
                           uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (tg * 8 + sg) * 4;  // h * Dv + r, four consecutive r of one head (Dv % 4 == 0)
  const uint h = row0 / a.Dv;
  const uint hk = h / (a.Hv / a.Hk);
  float4 s[4];
  for (uint i = 0; i < 4; ++i) s[i] = *(device float4*)(S + (ulong)(row0 + i) * a.Dk + lane * 4);
  const float qscale = rsqrt(float(a.Dk));
  for (uint t = 0; t < a.T; ++t) {
    device const float* ct = conv + (ulong)t * a.ldc;
    float4 q = *(device const float4*)(ct + hk * a.Dk + lane * 4);
    float4 k = *(device const float4*)(ct + a.Hk * a.Dk + hk * a.Dk + lane * 4);
    const float4 vr = *(device const float4*)(ct + 2 * a.Hk * a.Dk + row0);
    const float gt = g[t * a.Hv + h], bt = beta[t * a.Hv + h];
    const float qn = simd_sum(dot(q, q)), kn = simd_sum(dot(k, k));
    q *= qscale * rsqrt(qn + 1e-6f);
    k *= rsqrt(kn + 1e-6f);
    float kv[4], o[4];
    for (uint i = 0; i < 4; ++i) {
      s[i] *= gt;
      kv[i] = simd_sum(dot(s[i], k));
    }
    for (uint i = 0; i < 4; ++i) {
      s[i] += k * ((vr[i] - kv[i]) * bt);
      o[i] = simd_sum(dot(s[i], q));
    }
    if (lane == 0) *(device float4*)(y + (ulong)t * a.Hv * a.Dv + row0) = float4(o[0], o[1], o[2], o[3]);
  }
  for (uint i = 0; i < 4; ++i) *(device float4*)(S + (ulong)(row0 + i) * a.Dk + lane * 4) = s[i];
}

// ---- chunked GDN recurrence for prefill (Dk = Dv = 128, sub-chunks of 64 tokens) -------------------------------------
// Per sub-chunk c and v head h, with lg the cumulative log gate inside the sub-chunk (G_t = exp(lg_t)), q and k
// normalized, A the strictly lower beta_t (G_t/G_j) k_t.k_j and X = (I + A)^-1 (f32 forward substitution):
//   U = X diag(beta) V - X diag(beta G) K S^T = Uv - W S^T,  O = diag(G) Q S^T + (Q K^T * G_t/G_j, j <= t) U,
//   S <- G_C S + (diag(G_C/G) U)^T K.
// gdn_conv_h runs the conv in token blocks and writes raw q, k, v as half (pad rows zero); gdn_chunk_prep folds the
// l2 norms (q times Dk^-0.5) into its scalars and builds Uv, W (X diag(..) as hi + lo halves, so the factor keeps ~f32
// precision), A2 = [G Q | masked Q K^T] and the S-update scales for every (c, h) in parallel; gdn_chunk_seq walks
// the sub-chunks of one head and a block of state rows with S in registers (f32).
// Pad rows (t >= T) carry beta 0, gate 1 and zero q/k/v: an exact identity on S.
struct GdnChunkArgs {
  uint T, Hk, Hv, ldc;  // ldc: channels of the conv output (q | k | v)
};

// grid (ldc, token blocks of 64 over the padded chunk). The conv state moves in gdn_conv_state (several blocks) or
// here (one block).
kernel void gdn_conv_h(device const float* x [[buffer(0)]], device float* state [[buffer(1)]],
                       device const float* w [[buffer(2)]], device half* qr [[buffer(3)]], device half* kr [[buffer(4)]],
                       device half* vr [[buffer(5)]], constant GdnChunkArgs& a [[buffer(6)]],
                       uint2 pos [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  const uint C = a.ldc, c = pos.x, t0 = pos.y * 64, qw = a.Hk * 128;
  if (c >= C) return;
  const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  float s0, s1, s2;
  if (t0 == 0) {
    s0 = state[c], s1 = state[C + c], s2 = state[2 * C + c];
  } else {
    s0 = x[(ulong)(t0 - 3) * C + c], s1 = x[(ulong)(t0 - 2) * C + c], s2 = x[(ulong)(t0 - 1) * C + c];
  }
  device half* dst = c < qw ? qr + c : (c < 2 * qw ? kr + (c - qw) : vr + (c - 2 * qw));
  const uint ld = c < 2 * qw ? qw : a.Hv * 128;
  for (uint t = t0; t < t0 + 64; ++t) {
    float o = 0.0f;
    if (t < a.T) {
      const float v = x[(ulong)t * C + c];
      o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
      o = o / (1.0f + exp(-o));
      s0 = s1;
      s1 = s2;
      s2 = v;
    }
    dst[(ulong)t * ld] = half(o);
  }
  if (grid.y == 1) {
    state[c] = s0;
    state[C + c] = s1;
    state[2 * C + c] = s2;
  }
}

// After a multi-block gdn_conv_h (T >= 3): the state is the last three inputs.
kernel void gdn_conv_state(device const float* x [[buffer(0)]], device float* state [[buffer(1)]],
                           constant GdnChunkArgs& a [[buffer(2)]], uint c [[thread_position_in_grid]]) {
  const uint C = a.ldc;
  if (c >= C) return;
  for (uint i = 0; i < 3; ++i) state[i * C + c] = x[(ulong)(a.T - 3 + i) * C + c];
}

// grid (sub-chunks, Hv), 256 threads. Outputs a block (c, h) = c * Hv + h: W [64][128], Uv [64][128], A2 [64][192] half,
// sc [128] f32: [0, 64) the S-update scale G_C/G_t / |k_t|, [64] G_C.
kernel void gdn_chunk_prep(device const half* qr [[buffer(0)]], device const half* kr [[buffer(1)]],
                           device const half* vr [[buffer(2)]], device const float* g [[buffer(3)]],
                           device const float* beta [[buffer(4)]], device half* W [[buffer(5)]],
                           device half* Uv [[buffer(6)]], device half* A2 [[buffer(7)]], device float* sco [[buffer(8)]],
                           constant GdnChunkArgs& a [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint C = 64, D = 128, NSG = 8;
  threadgroup float sX[C * C];  // A; then two half matrices (hi | lo)
  threadgroup float sLg[C], sBe[C], sRq[C], sRk[C];
  const uint c = tg.x, h = tg.y, hk = h / (a.Hv / a.Hk), t0 = c * C;
  const ulong blk = (ulong)c * a.Hv + h;
  const uint ldk = a.Hk * D, ldv = a.Hv * D;
  {  // inverse norms: q rows by threads 0..127, k rows by 128..255, two lanes a row
    const uint isk = tid >> 7, r = (tid >> 1) & 63, p = tid & 1;
    device const half* src = (isk ? kr : qr) + (ulong)(t0 + r) * ldk + hk * D + p * 64;
    float n = 0.0f;
    for (uint k = 0; k < 64; k += 4) {
      const float4 v = float4(*(device const half4*)(src + k));
      n += dot(v, v);
    }
    n += simd_shuffle_xor(n, 1);
    if (p == 0) (isk ? sRk : sRq)[r] = rsqrt(n + 1e-6f) * (isk ? 1.0f : rsqrt(float(D)));
  }
  if (sg == 0) {
    float x[2], b[2];
    for (uint u = 0; u < 2; ++u) {
      const uint t = t0 + lane + 32 * u;
      x[u] = t < a.T ? max(log(g[(ulong)t * a.Hv + h]), -80.0f) : 0.0f;
      b[u] = t < a.T ? beta[(ulong)t * a.Hv + h] : 0.0f;
    }
    const float p0 = simd_prefix_inclusive_sum(x[0]);
    const float p1 = simd_prefix_inclusive_sum(x[1]) + simd_broadcast(p0, 31);
    sLg[lane] = p0;
    sLg[lane + 32] = p1;
    sBe[lane] = b[0];
    sBe[lane + 32] = b[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < C) {
    const float lgC = sLg[C - 1];
    sco[blk * 2 * C + tid] = exp(lgC - sLg[tid]) * sRk[tid];
    if (tid == 0) sco[blk * 2 * C + C] = exp(lgC);
  }
  auto tK = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)kr + (ulong)t0 * ldk + hk * D,
                                                                    dextents<int32_t, 2>(D, C), array<int32_t, 2>{1, int(ldk)});
  auto tQ = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)qr + (ulong)t0 * ldk + hk * D,
                                                                    dextents<int32_t, 2>(D, C), array<int32_t, 2>{1, int(ldk)});
  auto tV = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)vr + (ulong)t0 * ldv + h * D,
                                                                    dextents<int32_t, 2>(D, C), array<int32_t, 2>{1, int(ldv)});
  auto mK = tK.slice(0, 0);
  auto mQ = tQ.slice(0, 0);
  auto mV = tV.slice(0, 0);
  constexpr auto d1 = matmul2d_descriptor(C, C, D, false, true, false);
  matmul2d<d1, execution_simdgroups<NSG>> op1;
  device half* a2 = A2 + blk * C * 192;
  {
    auto cK = op1.template get_destination_cooperative_tensor<decltype(mK), decltype(mK), float>();
    op1.run(mK, mK, cK);
#pragma unroll
    for (uint16_t i = 0; i < cK.get_capacity(); ++i)
      if (cK.is_valid_element(i)) {
        const auto ix = cK.get_multidimensional_index(i);
        const uint j = uint(ix[0]), r = uint(ix[1]);
        sX[r * C + j] = j < r ? sBe[r] * exp(sLg[r] - sLg[j]) * sRk[r] * sRk[j] * cK[i] : 0.0f;
      }
    auto cQ = op1.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    op1.run(mQ, mK, cQ);
#pragma unroll
    for (uint16_t i = 0; i < cQ.get_capacity(); ++i)
      if (cQ.is_valid_element(i)) {
        const auto ix = cQ.get_multidimensional_index(i);
        const uint j = uint(ix[0]), r = uint(ix[1]);
        a2[r * 192 + 128 + j] = half(j <= r ? exp(sLg[r] - sLg[j]) * sRq[r] * sRk[j] * cQ[i] : 0.0f);
      }
  }
  for (uint e = tid; e < C * D / 4; e += 256) {
    const uint r = e / (D / 4), k = (e % (D / 4)) * 4;
    const float4 v = float4(*(device const half4*)(qr + (ulong)(t0 + r) * ldk + hk * D + k));
    *(device half4*)(a2 + r * 192 + k) = half4(exp(sLg[r]) * sRq[r] * v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // X = (I + A)^-1 by forward substitution, X_ic = d_ic - sum_{j < i} A_ij X_jc: simdgroup sg owns columns 8 sg..8 sg+7,
  // lane l rows l and l + 32 (registers), one simd_sum a row and column, no barriers.
  float x0[8], x1[8];
#pragma unroll
  for (uint m = 0; m < 8; ++m) x0[m] = x1[m] = 0.0f;
  for (uint i = 0; i < C; ++i) {
    const float a0 = sX[i * C + lane], a1 = sX[i * C + lane + 32];
#pragma unroll
    for (uint m = 0; m < 8; ++m) {
      const float v = (i == sg * 8 + m ? 1.0f : 0.0f) - simd_sum(a0 * x0[m] + a1 * x1[m]);
      if (i == lane) x0[m] = v;
      if (i == lane + 32) x1[m] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup half* sH = (threadgroup half*)sX;
  auto tHi = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sH, dextents<int32_t, 2>(C, C), array<int32_t, 2>{1, int(C)});
  auto tLo = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sH + C * C, dextents<int32_t, 2>(C, C), array<int32_t, 2>{1, int(C)});
  constexpr auto d2 = matmul2d_descriptor(C, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d2, execution_simdgroups<NSG>> op2;
  for (uint pass = 0; pass < 2; ++pass) {  // 0: Uv = X diag(beta) V, 1: W = X diag(beta G / |k|) K
#pragma unroll
    for (uint m = 0; m < 8; ++m) {
      const uint j = sg * 8 + m;
      const float fj = sBe[j] * (pass ? exp(sLg[j]) * sRk[j] : 1.0f);
      const float f0 = x0[m] * fj, f1 = x1[m] * fj;
      const half h0 = half(f0), h1 = half(f1);
      sH[lane * C + j] = h0;
      sH[(lane + 32) * C + j] = h1;
      sH[C * C + lane * C + j] = half(f0 - float(h0));
      sH[C * C + (lane + 32) * C + j] = half(f1 - float(h1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto cU = op2.template get_destination_cooperative_tensor<decltype(tHi), decltype(mV), float>();
#pragma unroll
    for (uint16_t i = 0; i < cU.get_capacity(); ++i)
      if (cU.is_valid_element(i)) cU[i] = 0.0f;
    if (pass == 0) {
      op2.run(tHi, mV, cU);
      op2.run(tLo, mV, cU);
    } else {
      op2.run(tHi, mK, cU);
      op2.run(tLo, mK, cU);
    }
    device half* dst = (pass ? W : Uv) + blk * C * D;
#pragma unroll
    for (uint16_t i = 0; i < cU.get_capacity(); ++i)
      if (cU.is_valid_element(i)) {
        const auto ix = cU.get_multidimensional_index(i);
        dst[uint(ix[1]) * D + uint(ix[0])] = half(cU[i]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// grid (Dv / 32, Hv), 128 threads: one head's 32 state rows through every sub-chunk; y [T][Hv * Dv] f32.
kernel void gdn_chunk_seq(device const half* kr [[buffer(0)]], device const half* W [[buffer(1)]],
                          device const half* Uv [[buffer(2)]], device const half* A2 [[buffer(3)]],
                          device const float* sc [[buffer(4)]], device float* S [[buffer(5)]], device float* y [[buffer(6)]],
                          constant GdnChunkArgs& a [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr uint C = 64, D = 128, R = 32, NSG = 4, LB = D + C;
  threadgroup half sB[R * LB];  // row r: [half S[r][0..127] | U[0..63][r]]
  threadgroup half sUd[R * C];  // row r: U[t][r] G_C / G_t
  const uint rb = tg.x, h = tg.y, hk = h / (a.Hv / a.Hk), ldk = a.Hk * D, ldy = a.Hv * D;
  const uint nc = (a.T + C - 1) / C;
  auto tS = tensor<device float, dextents<int32_t, 2>, tensor_inline>(S + ((ulong)h * D + rb * R) * D,
                                                                     dextents<int32_t, 2>(D, R), array<int32_t, 2>{1, int(D)});
  auto tSh = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sB, dextents<int32_t, 2>(D, R), array<int32_t, 2>{1, int(LB)});
  auto tBa = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sB, dextents<int32_t, 2>(LB, R), array<int32_t, 2>{1, int(LB)});
  auto tUd = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sUd, dextents<int32_t, 2>(C, R), array<int32_t, 2>{1, int(C)});
  constexpr auto dA = matmul2d_descriptor(C, R, D, false, true, false);
  constexpr auto dO = matmul2d_descriptor(C, R, LB, false, true, false);
  constexpr auto dS = matmul2d_descriptor(R, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dA, execution_simdgroups<NSG>> opA;
  matmul2d<dO, execution_simdgroups<NSG>> opO;
  matmul2d<dS, execution_simdgroups<NSG>> opS;
  auto tK0 = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)kr + hk * D, dextents<int32_t, 2>(D, C),
                                                                     array<int32_t, 2>{1, int(ldk)});
  auto cS = opS.template get_destination_cooperative_tensor<decltype(tUd), decltype(tK0), float>();
  cS.load(tS);
  for (uint c = 0; c < nc; ++c) {
    const ulong blk = (ulong)c * a.Hv + h;
#pragma unroll
    for (uint16_t i = 0; i < cS.get_capacity(); ++i)
      if (cS.is_valid_element(i)) {
        const auto ix = cS.get_multidimensional_index(i);
        sB[uint(ix[1]) * LB + uint(ix[0])] = half(cS[i]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mW = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)W + blk * C * D, dextents<int32_t, 2>(D, C),
                                                                      array<int32_t, 2>{1, int(D)});
    auto cP = opA.template get_destination_cooperative_tensor<decltype(mW), decltype(tSh), float>();
    opA.run(mW, tSh, cP);
    device const float* scb = sc + blk * 2 * C;
#pragma unroll
    for (uint16_t i = 0; i < cP.get_capacity(); ++i)
      if (cP.is_valid_element(i)) {
        const auto ix = cP.get_multidimensional_index(i);
        const uint r = uint(ix[0]), t = uint(ix[1]);
        const float u = float(Uv[(blk * C + t) * D + rb * R + r]) - cP[i];
        sB[r * LB + D + t] = half(u);
        sUd[r * C + t] = half(u * scb[t]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mA2 = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)A2 + blk * C * LB, dextents<int32_t, 2>(LB, C),
                                                                       array<int32_t, 2>{1, int(LB)});
    auto cO = opO.template get_destination_cooperative_tensor<decltype(mA2), decltype(tBa), float>();
    opO.run(mA2, tBa, cO);
    auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(y + (ulong)c * C * ldy + h * D + rb * R,
                                                                       dextents<int32_t, 2>(R, int(min(C, a.T - c * C))),
                                                                       array<int32_t, 2>{1, int(ldy)});
    cO.store(tY);
    const float gC = scb[C];
#pragma unroll
    for (uint16_t i = 0; i < cS.get_capacity(); ++i)
      if (cS.is_valid_element(i)) cS[i] *= gC;
    auto mK = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)kr + (ulong)c * C * ldk + hk * D,
                                                                      dextents<int32_t, 2>(D, C), array<int32_t, 2>{1, int(ldk)});
    opS.run(tUd, mK, cS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  cS.store(tS);
}

// ---- greedy pick on the GPU (pipelined decode) ---------------------------------------------------------------
// One threadgroup of 1024 threads over the logits: the largest value (lowest index on ties, as the host loop) and the
// second largest of the multiset, skipping the banned ids. Writes the id to ids[0] (the next step's embedding input)
// and {id, top, second} to out.
struct Top2Args {
  uint n, nb;
  uint banned[8];
};
struct Top2 {
  float v, s;
  uint id;
};
inline Top2 top2_merge(Top2 a, Top2 b) {
  const bool bw = b.v > a.v || (b.v == a.v && b.id < a.id);  // ties: the lower index
  Top2 r;
  r.v = bw ? b.v : a.v;
  r.id = bw ? b.id : a.id;
  r.s = bw ? max(a.v, b.s) : max(a.s, b.v);
  return r;
}
kernel void top2_ban(device const float* x [[buffer(0)]], device uint* ids [[buffer(1)]],
                     device uint* out [[buffer(2)]], constant Top2Args& a [[buffer(3)]],
                     uint tid [[thread_index_in_threadgroup]]) {
  Top2 t{-INFINITY, -INFINITY, 0};
  const uint per = (a.n + 1023) / 1024, i0 = tid * per, i1 = min(i0 + per, a.n);  // a contiguous slice a thread
  for (uint i = i0; i < i1; ++i) {
    const float v = x[i];
    if (!(v > t.s)) continue;
    bool ban = false;
    for (uint b = 0; b < a.nb; ++b) ban = ban || a.banned[b] == i;
    if (ban) continue;
    if (v > t.v) {
      t.s = t.v;
      t.v = v;
      t.id = i;
    } else {
      t.s = v;
    }
  }
  threadgroup Top2 sh[1024];
  sh[tid] = t;
  for (uint w = 512; w > 0; w >>= 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < w) sh[tid] = top2_merge(sh[tid], sh[tid + w]);
  }
  if (tid == 0) {
    ids[0] = sh[0].id;
    out[0] = sh[0].id;
    out[1] = as_type<uint>(sh[0].v);
    out[2] = as_type<uint>(sh[0].s);
  }
}
