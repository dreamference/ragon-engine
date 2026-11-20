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
  uint rows;       // activation rows in this dispatch (<= kRows)
  float scale;     // global scale (NVFP4 scale_2, the FP8 per-tensor scale, 1 for BF16)
  uint add;        // 1: y += result (the residual adds)
};

inline void gemv_store(device float* y, constant GemvArgs& a, uint n, thread float* acc, float unscale, uint lane) {
  for (uint m = 0; m < kRows; ++m) {
    if (m >= a.rows) break;
    const float r = simd_sum(acc[m]) * a.scale * unscale;
    if (lane == 0) {
      device float* p = y + (ulong)m * a.ldy + n;
      *p = a.add ? *p + r : r;
    }
  }
}

// One simdgroup per output row n, 8 rows a threadgroup; each lane takes 16 weight bytes (32 values) a step.
kernel void gemv_nvfp4(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                       device const uchar* sc [[buffer(2)]], device float* y [[buffer(3)]],
                       constant GemvArgs& a [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg * 8 + sg;
  if (n >= a.N) return;
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
      if (m >= a.rows) break;
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
  gemv_store(y, a, n, acc, kFp4Unscale, lane);
}

// FP8 rows: each lane takes 16 weight bytes (16 values) a step.
kernel void gemv_fp8(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                     device float* y [[buffer(2)]], constant GemvArgs& a [[buffer(3)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg * 8 + sg;
  if (n >= a.N) return;
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
      if (m >= a.rows) break;
      device const float4* xr = (device const float4*)(x + (ulong)m * a.ldx + s * 16);
      float b = 0;
      for (uint i = 0; i < 4; ++i) {
        const float4 xa = xr[i];
        b += dot(v[2 * i], xa.xy) + dot(v[2 * i + 1], xa.zw);
      }
      acc[m] += b;
    }
  }
  gemv_store(y, a, n, acc, 1.0f / kFp8Unscale, lane);
}

// BF16 rows: each lane takes 16 bytes (8 values) a step.
kernel void gemv_bf16(device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
                      device float* y [[buffer(2)]], constant GemvArgs& a [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg * 8 + sg;
  if (n >= a.N) return;
  device const uint4* wr = (device const uint4*)(w + (ulong)n * a.K * 2);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const uint steps = a.K / 8;
  for (uint s = lane; s < steps; s += 32) {
    const uint4 q = wr[s];
    for (uint m = 0; m < kRows; ++m) {
      if (m >= a.rows) break;
      device const float4* xr = (device const float4*)(x + (ulong)m * a.ldx + s * 8);
      const float4 xa = xr[0], xb = xr[1];
      acc[m] += dot(bf16x2(q[0]), xa.xy) + dot(bf16x2(q[1]), xa.zw) + dot(bf16x2(q[2]), xb.xy) +
                dot(bf16x2(q[3]), xb.zw);
    }
  }
  gemv_store(y, a, n, acc, 1.0f, lane);
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
};

// One threadgroup (256 threads) a row; x and out may alias (in place).
kernel void rmsnorm_rows(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                         device const float* z [[buffer(2)]], device float* out [[buffer(3)]],
                         constant NormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const ulong xo = (ulong)(row / a.rpt) * a.ldt + (ulong)(row % a.rpt) * a.ldh;
  const ulong oo = (ulong)(row / a.rpt) * a.ldo_t + (ulong)(row % a.rpt) * a.ldo_h;
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
                         uint d [[thread_position_in_threadgroup]]) {
  const uint h = tg.x, t = tg.y;
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
