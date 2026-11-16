// The rows path's attention on HIP (gfx1151, wave32; BACKEND §7, W1): attention_body.cuh includes this file in place of
// its CUDA section (mma.sync, ldmatrix, cp.async, TMA). The contract is the CUDA one: the same blocks (ROWS query rows
// of one KV head, one fixed range of attention_chunk() keys), the same 32-key tiles in the same order, the same masks,
// the same online softmax, the same partials in `scratch` and the same combine. Inside a tile the products run on WMMA
// (16x16x16 BF16 into FP32) with the roles swapped: S^T = K Q^T and O^T += V^T P^T, so a lane owns one query row (lane
// & 15) and every per-row step (mask, max, exp, sum, rescale) is that lane's and its partner's (lane ^ 16), in a fixed
// order. A row's arithmetic depends only on its own query and keys: the WMMA columns are independent of each other and
// of the lane that holds them, keys past a row's position get p = 0, and keys past the range are zero-filled. So a row's
// bits are the same in every block, every verify width and every run (row invariance; the tests check it).
// Not the CUDA bits: another order of the same products (exp and the MMA's internal sums are the hardware's).
// LDS is 64 KB a block on gfx11, so the block holds one tile (K as [key][d], V transposed as [d][key]) and its rows'
// queries in BF16; no bulk copies (BULK is the plain path here) and no L2 prefetch (pf is ignored).
// Included inside attention_body.cuh's namespace, at its end (the CUDA objects' line tables stay put).
#pragma once

constexpr int kMmaD = 256;
constexpr int kHipQS = kMmaD + 8;  // padded row stride (bf16) of the Q and K tiles in LDS: conflict-free 32-byte reads
// The LDS of a block is static (attention_mma_body's own arrays): the launches pass no dynamic shared memory.
constexpr size_t mma_smem(int, int) { return 0; }
constexpr size_t mma_smem_q8(int, int) { return 0; }

typedef short attn_v16s __attribute__((ext_vector_type(16)));
typedef float attn_v8f __attribute__((ext_vector_type(8)));
typedef unsigned attn_v8u __attribute__((ext_vector_type(8)));

// 16 BF16 values (32 bytes, 16-byte aligned) from LDS as one WMMA operand.
__device__ __forceinline__ attn_v16s attn_lds16(const gpu::bf16* p) {
  const uint4 x = reinterpret_cast<const uint4*>(p)[0], y = reinterpret_cast<const uint4*>(p)[1];
  const attn_v8u w = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
  return __builtin_bit_cast(attn_v16s, w);
}

// A finite float to BF16 bits, rounded to nearest even (the rounding of __floats2bfloat162_rn).
__device__ __forceinline__ uint32_t attn_bf16_bits(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { return attn_bf16_bits(lo) | (attn_bf16_bits(hi) << 16); }
__device__ __forceinline__ float attn_bf16_float(uint32_t b) { return __uint_as_float(b << 16); }

// The other half of the wave's value at this lane's place (lane ^ 16).
__device__ __forceinline__ uint32_t attn_xswap(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}
__device__ __forceinline__ float attn_xswapf(float v) { return __uint_as_float(attn_xswap(__float_as_uint(v))); }

// ---- The FP8 KV cache (--kv-fp8, LOWBIT §8): the CUDA definitions' arithmetic (see attention_body.cuh). ----
// e = e4m3_satfinite(v * inv), two values a call, the first in the low byte (CUDA's fp8x2 storage).
__device__ __forceinline__ uint16_t kv_q8_encode2(float2 x, float inv) {
  return static_cast<uint16_t>(gpu::e4m3_from_float(gpu::mul_rn(x.x, inv)) |
                               (gpu::e4m3_from_float(gpu::mul_rn(x.y, inv)) << 8));
}
// float(e) for an E4M3 code (exact): its bits moved into an FP16, 2^8 restores the bias; the NaN codes give NaN.
__device__ __forceinline__ float kv_q8_code_float(uint32_t b) {
  const uint32_t r = ((b & 0x7fu) << 7) | ((b & 0x80u) << 8);
  const float f = __half2float(__builtin_bit_cast(__half, static_cast<unsigned short>(r))) * 256.f;
  return (b & 0x7fu) == 0x7fu ? __uint_as_float(0x7fc00000u) : f;
}
// R's read side: bf16(float(e) * s), two codes (low byte first) to a bf16 pair's bits (low first).
__device__ __forceinline__ uint32_t kv_q8_decode2_bits(uint32_t e2, float s) {
  return pack_bf16(gpu::mul_rn(kv_q8_code_float(e2 & 0xffu), s), gpu::mul_rn(kv_q8_code_float((e2 >> 8) & 0xffu), s));
}
// R(v) of a bf16 pair (bits, low first).
__device__ __forceinline__ uint32_t kv_q8_round2_bits(uint32_t v, float s, float inv) {
  return kv_q8_decode2_bits(kv_q8_encode2(make_float2(attn_bf16_float(v & 0xffffu), attn_bf16_float(v >> 16)), inv), s);
}
// The CUDA signatures (two codes in, a bf16x2 out), for callers outside the body (tests/kvq8_tests.cu).
// (HIP's bf16x2 is not trivially copyable, so the bits move by memcpy.)
__device__ __forceinline__ gpu::bf16x2 attn_bf16x2_of(uint32_t b) {
  gpu::bf16x2 r;
  __builtin_memcpy(&r, &b, 4);
  return r;
}
__device__ __forceinline__ gpu::bf16x2 kv_q8_decode2(uint16_t e, float s) {
  return attn_bf16x2_of(kv_q8_decode2_bits(e, s));
}
__device__ __forceinline__ gpu::bf16x2 kv_q8_round2(gpu::bf16x2 v, float s, float inv) {
  uint32_t b;
  __builtin_memcpy(&b, &v, 4);
  return attn_bf16x2_of(kv_q8_round2_bits(b, s, inv));
}

// One block's LDS: its rows' queries (BF16, scaled), one tile's K ([key][d]) and V^T ([d][key]).
template <int ROWS, int TK>
struct AttnLds {
  static constexpr int VS = TK + 8;  // V^T row stride (bf16)
  gpu::bf16 q[ROWS * kHipQS];
  gpu::bf16 k[TK * kHipQS];
  gpu::bf16 vt[kMmaD * VS];
};

// The tile of keys [k0, k0 + TK) into LDS, zero-filled from khi on; src(key) is the element offset of a key below khi.
// K row by row (16 bytes a thread a step); V four keys by eight dims a step, transposed on the way into V^T.
template <int ROWS, int TK, typename Src>
__device__ __forceinline__ void attn_load_tile(AttnLds<ROWS, TK>& sm, const gpu::bf16* __restrict__ kcache,
                                               const gpu::bf16* __restrict__ vcache, int k0, int khi, Src src) {
  constexpr int D = kMmaD, VS = AttnLds<ROWS, TK>::VS;
  for (int i = threadIdx.x; i < TK * (D / 8); i += blockDim.x) {
    const int j = i / (D / 8), c = i % (D / 8), key = k0 + j;
    uint4 w = make_uint4(0, 0, 0, 0);
    if (key < khi) w = *reinterpret_cast<const uint4*>(kcache + src(key) + 8 * c);
    *reinterpret_cast<uint4*>(sm.k + j * kHipQS + 8 * c) = w;
  }
  for (int i = threadIdx.x; i < (TK / 4) * (D / 8); i += blockDim.x) {
    const int jq = i / (D / 8), c = i % (D / 8);
    uint4 w[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int key = k0 + 4 * jq + u;
      w[u] = key < khi ? *reinterpret_cast<const uint4*>(vcache + src(key) + 8 * c) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      auto half = [&](const uint4& x) {
        const uint32_t d = e < 2 ? x.x : e < 4 ? x.y : e < 6 ? x.z : x.w;
        return (e & 1) ? d >> 16 : d & 0xffffu;
      };
      const uint2 o = make_uint2(half(w[0]) | (half(w[1]) << 16), half(w[2]) | (half(w[3]) << 16));
      *reinterpret_cast<uint2*>(sm.vt + (8 * c + e) * VS + 4 * jq) = o;
    }
  }
}

// M2: a sealed tile (keys [k0, k0 + TK), all below khi) from the layer's FP8 mirror, decoded into LDS: the values
// attn_round_tile leaves for the BF16 tile, bit for bit. The mirror holds a code at each pool element's offset.
template <int ROWS, int TK>
__device__ __forceinline__ void attn_load_tile_f8(AttnLds<ROWS, TK>& sm, const KvQ8& q8, int k0, int kvh, size_t hs,
                                                  size_t ps, const uint64_t* __restrict__ pages) {
  constexpr int D = kMmaD, VS = AttnLds<ROWS, TK>::VS;
  const size_t base = kv_off(pages, kvh, k0, hs, ps);
  for (int i = threadIdx.x; i < TK * (D / 16); i += blockDim.x) {
    const int j = i / (D / 16), c = i % (D / 16);
    const uint4 w = *reinterpret_cast<const uint4*>(q8.k8 + base + static_cast<size_t>(j) * ps + 16 * c);
    const uint32_t e[4] = {w.x, w.y, w.z, w.w};
    uint4 o[2];
    uint32_t* h = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      h[2 * u] = kv_q8_decode2_bits(e[u] & 0xffffu, q8.ks);
      h[2 * u + 1] = kv_q8_decode2_bits(e[u] >> 16, q8.ks);
    }
    uint4* dst = reinterpret_cast<uint4*>(sm.k + j * kHipQS + 16 * c);
    dst[0] = o[0];
    dst[1] = o[1];
  }
  for (int i = threadIdx.x; i < (TK / 4) * (D / 16); i += blockDim.x) {
    const int jq = i / (D / 16), c = i % (D / 16);
    uint4 w[4];
#pragma unroll
    for (int u = 0; u < 4; ++u)
      w[u] = *reinterpret_cast<const uint4*>(q8.v8 + base + static_cast<size_t>(4 * jq + u) * ps + 16 * c);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      auto code = [&](const uint4& x) {
        const uint32_t d = e < 4 ? x.x : e < 8 ? x.y : e < 12 ? x.z : x.w;
        return (d >> (8 * (e & 3))) & 0xffu;
      };
      const uint2 o = make_uint2(kv_q8_decode2_bits(code(w[0]) | (code(w[1]) << 8), q8.vs),
                                 kv_q8_decode2_bits(code(w[2]) | (code(w[3]) << 8), q8.vs));
      *reinterpret_cast<uint2*>(sm.vt + (16 * c + e) * VS + 4 * jq) = o;
    }
  }
}

// The staged tile replaced by R of its values, in place (K with the K scale, V^T with the V scale).
template <int ROWS, int TK>
__device__ __forceinline__ void attn_round_tile(AttnLds<ROWS, TK>& sm, const KvQ8& q8) {
  constexpr int D = kMmaD, VS = AttnLds<ROWS, TK>::VS;
  for (int i = threadIdx.x; i < TK * (D / 8); i += blockDim.x) {
    uint4* p = reinterpret_cast<uint4*>(sm.k + (i / (D / 8)) * kHipQS + 8 * (i % (D / 8)));
    uint4 w = *p;
    w.x = kv_q8_round2_bits(w.x, q8.ks, q8.kinv);
    w.y = kv_q8_round2_bits(w.y, q8.ks, q8.kinv);
    w.z = kv_q8_round2_bits(w.z, q8.ks, q8.kinv);
    w.w = kv_q8_round2_bits(w.w, q8.ks, q8.kinv);
    *p = w;
  }
  for (int i = threadIdx.x; i < D * (TK / 8); i += blockDim.x) {
    uint4* p = reinterpret_cast<uint4*>(sm.vt + (i / (TK / 8)) * VS + 8 * (i % (TK / 8)));
    uint4 w = *p;
    w.x = kv_q8_round2_bits(w.x, q8.vs, q8.vinv);
    w.y = kv_q8_round2_bits(w.y, q8.vs, q8.vinv);
    w.z = kv_q8_round2_bits(w.z, q8.vs, q8.vinv);
    w.w = kv_q8_round2_bits(w.w, q8.vs, q8.vinv);
    *p = w;
  }
}

// The CUDA template's parameters and arguments (attention_body.cuh): ROWS query rows a block, 16 a wave; TK keys a
// tile; NS, BULK, pf: the CUDA pipeline's, ignored here (one tile in LDS, plain loads); CQ: the queries are read plainly
// (no megakernel on HIP); CASC and Q8 as on CUDA.
template <int ROWS, int TK, int NS, bool BULK, bool CQ = false, int CASC = 0, bool Q8 = false>
__device__ __forceinline__ void attention_mma_body(const float* __restrict__ q, const gpu::bf16* __restrict__ kcache,
                                                   const gpu::bf16* __restrict__ vcache, int pos0, int M, int Hq,
                                                   int Hkv, int splits, int chunk, float* __restrict__ scratch, size_t hs,
                                                   size_t ps, int pf, int bx, int kvh, int split,
                                                   const TreeDesc* __restrict__ td = nullptr,
                                                   const uint64_t* __restrict__ pages = nullptr, int kcut = 0,
                                                   KvQ8 q8 = {}) {
  constexpr int D = kMmaD, G = 6, KT = TK / 16, VS = AttnLds<ROWS, TK>::VS;
  static_assert(kKvPageTokens % TK == 0, "a tile inside one KV page (P4)");
  static_assert(TK % 16 == 0 && ROWS % 16 == 0, "16-key and 16-row WMMA tiles");
  static_assert(!Q8 || TK == 32, "Q8: one 32-key unit per tile");
  static_assert(sizeof(AttnLds<ROWS, TK>) <= 65536, "LDS: 64 KB a block on gfx11");
  (void)pf;
  __shared__ __attribute__((aligned(16))) AttnLds<ROWS, TK> sm;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, col = lane & 15, half = lane >> 4;
  const int R = M * G, r0 = bx * ROWS, nrows = min(ROWS, R - r0);
  const bool active = warp * 16 < nrows;
  const int last_m = (r0 + nrows - 1) / G;
  const int kend = CASC == 1 ? kcut : pos0 + last_m + 1;  // keys [0, kend) are visible to some row of this block
  const int klo = split * chunk, khi = min(kend, klo + chunk);
  const int kfirst = CASC == 2 ? max(klo, kcut) : klo;  // this launch's first key of the range
  if (klo >= kend) {
    for (int r = tid; r < nrows; r += blockDim.x) {
      const int row = r0 + r;
      scratch[((static_cast<size_t>(row / G) * Hq + kvh * G + row % G) * splits + split) * (D + 2)] = -INFINITY;
    }
    return;
  }
  // This lane's row (the wave's row `col`; lanes col and col + 16 hold the same row) and its position.
  const int ra = r0 + warp * 16 + col;
  const bool va = ra < r0 + nrows;
  const int pa = pos0 + (td ? (va ? td->depth[ra / G] : 0) : ra / G);
  const int Pa = pa & ~31;
  int minP = 0, maxP = 0;
  if constexpr (Q8) {
    if (td) {
      minP = INT_MAX;
      for (int m = r0 / G; m <= last_m; ++m) {
        const int P = (pos0 + td->depth[m]) & ~31;
        minP = min(minP, P);
        maxP = max(maxP, P);
      }
    } else {
      minP = (pos0 + r0 / G) & ~31;
      maxP = (pos0 + last_m) & ~31;
    }
  }
  int pmin = INT_MAX;  // the block's first row position (tile_rows)
  for (int m = r0 / G; m <= last_m; ++m) pmin = min(pmin, pos0 + (td ? td->depth[m] : m));
  // The block's queries into LDS in BF16, scaled by 1 / sqrt(D) (a power of two: exact); rows past nrows are zeros.
  for (int i = tid; i < ROWS * (D / 4); i += blockDim.x) {
    const int r = i / (D / 4), c = i % (D / 4), row = r0 + r;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < nrows) v = *reinterpret_cast<const float4*>(q + (static_cast<size_t>(row / G) * Hq + kvh * G + row % G) * D + 4 * c);
    *reinterpret_cast<uint2*>(sm.q + r * kHipQS + 4 * c) =
        make_uint2(pack_bf16(v.x * 0.0625f, v.y * 0.0625f), pack_bf16(v.z * 0.0625f, v.w * 0.0625f));
  }
  // O^T for this lane's row: dim tile i holds dims 16 i + 2 j + half in o[i][j].
  attn_v8f o[D / 16];
#pragma unroll
  for (int i = 0; i < D / 16; ++i) o[i] = attn_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float ma = -INFINITY, la = 0.f;
  if constexpr (CASC == 2) {
    if (klo < kcut && va) {  // the partial state the first phase stored, back as it was
      const float* part = scratch + ((static_cast<size_t>(ra / G) * Hq + kvh * G + ra % G) * splits + split) * (D + 2);
      ma = part[0];
      la = part[1];
#pragma unroll
      for (int i = 0; i < D / 16; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) o[i][j] = part[2 + 16 * i + 2 * j + half];
    }
  }
  const gpu::bf16* qrow = sm.q + (warp * 16 + col) * kHipQS;
  // One tile's scores, online softmax and P V for this lane's row; ca: the row takes the tile.
  auto tile_compute = [&](int k0, bool ca) {
    attn_v8f sc[KT];
#pragma unroll
    for (int n = 0; n < KT; ++n) sc[n] = attn_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < D / 16; ++s) {
      const attn_v16s b = attn_lds16(qrow + 16 * s);
#pragma unroll
      for (int n = 0; n < KT; ++n)
        sc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(attn_lds16(sm.k + (16 * n + col) * kHipQS + 16 * s), b, sc[n]);
    }
    // sc[n][j] is the score of key k0 + 16 n + 2 j + half. Mask (causal, and the range end), then the online softmax.
    float ta = -INFINITY;
#pragma unroll
    for (int n = 0; n < KT; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int key = k0 + 16 * n + 2 * j + half;
        if (key > pa || key >= khi) sc[n][j] = -INFINITY;
        ta = fmaxf(ta, sc[n][j]);
      }
    ta = fmaxf(ta, attn_xswapf(ta));
    const float na = fmaxf(ma, ta);
    const float alpha = na == -INFINITY ? 1.f : __expf(ma - na);
    float sa = 0.f;
#pragma unroll
    for (int n = 0; n < KT; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sc[n][j] = sc[n][j] == -INFINITY ? 0.f : __expf(sc[n][j] - na);
        sa += sc[n][j];
      }
    sa += attn_xswapf(sa);  // even keys' sum + odd keys' sum, the same float on both lanes
    if (ca) {
      la = la * alpha + sa;
      ma = na;
    }
    // P^T as the B operand, 16 keys a step: key 2 j from the even lane's value, key 2 j + 1 from the odd one's, in BF16.
    attn_v16s pb[KT];
#pragma unroll
    for (int n = 0; n < KT; ++n) {
      uint32_t own[4], oth[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        own[u] = pack_bf16(sc[n][2 * u], sc[n][2 * u + 1]);
        oth[u] = attn_xswap(own[u]);
      }
      attn_v8u w;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint32_t ev = half ? oth[u] : own[u], od = half ? own[u] : oth[u];
        w[2 * u] = (ev & 0xffffu) | (od << 16);
        w[2 * u + 1] = (ev >> 16) | (od & 0xffff0000u);
      }
      pb[n] = __builtin_bit_cast(attn_v16s, w);
    }
    // The tile's P V apart (from zero), then o = o alpha + P V on the rows that take the tile; the others keep o bit
    // for bit. gfx11's WMMA is not exact for zero products (a zero P times a negative V moves the sum down an ulp), so a
    // row's o never meets products it does not take, and tile_rows zeroes the V its masked keys would multiply.
#pragma unroll
    for (int i = 0; i < D / 16; ++i) {
      attn_v8f t = attn_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int n = 0; n < KT; ++n)
        t = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(attn_lds16(sm.vt + (16 * i + col) * VS + 16 * n), pb[n], t);
      if (ca)
#pragma unroll
        for (int j = 0; j < 8; ++j) o[i][j] = __builtin_fmaf(o[i][j], alpha, t[j]);
    }
  };
  // A tile in LDS for the block's rows, `ca` the lane's flag. A row that sees every key the tile holds (to its last
  // key below khi) takes it as loaded. A row whose position pa ends inside it takes it with the V of the keys past pa
  // zeroed, as a launch whose last key is pa holds it (zero-filled from khi on): the row's bytes then do not depend on
  // the launch's other rows (the zero-product note above). parts: do the cut rows (not when every row taking the tile
  // sees all of it). Block-uniform (it meets barriers); returns whether it zeroed V.
  auto tile_rows = [&](int k0, bool ca, bool parts) {
    const int top = min(k0 + TK, khi) - 1;
    if (active) tile_compute(k0, ca && pa >= top);
    bool cut = false;
    if (!parts || pmin >= top) return cut;
    int zcur = top;  // the tile's keys past zcur hold zeros
    for (int pc = top - 1; pc >= max(k0, pmin); --pc) {
      bool any = false;
      for (int m = r0 / G; m <= last_m; ++m) any = any || pos0 + (td ? td->depth[m] : m) == pc;
      if (!any) continue;
      const int nz = zcur - pc;
      __syncthreads();
      for (int i = tid; i < D * nz; i += blockDim.x)
        reinterpret_cast<uint16_t*>(sm.vt)[(i / nz) * VS + pc + 1 - k0 + i % nz] = 0;
      __syncthreads();
      zcur = pc;
      cut = true;
      if (active) tile_compute(k0, ca && pa == pc);
    }
    return cut;
  };
  // M2 (Q8 with a mirror): a tile of a sealed page that every row of the block reads as R(v) comes from the FP8 mirror.
  auto sealed_tile = [&](int k0, int kctx) {
    if constexpr (!Q8) return false;
    else return q8.k8 != nullptr && pages != nullptr && k0 + TK <= minP && k0 + TK <= kctx &&
                q8.sealed[pages[k0 / kKvPageTokens] / q8.unit] != 0;
  };
  // Q8: a tile in the rows' two domains, as on CUDA: first as stored for the rows whose unit is not past the tile's,
  // then rounded in place for the rows past it. Block-uniform branches (every thread meets the barriers).
  // A tile whose V tile_rows cut is loaded again for the rows past its unit (every one of them sees the whole tile).
  auto tile_q8 = [&](int k0, bool ca, auto&& reload) {
    const bool raw = k0 >= minP, rnd = k0 < maxP;
    bool cut = false;
    if (raw) cut = tile_rows(k0, ca && Pa <= k0, true);
    if (rnd) {
      if (cut) {
        __syncthreads();
        reload();
      }
      if (raw) __syncthreads();
      attn_round_tile<ROWS, TK>(sm, q8);
      __syncthreads();
      tile_rows(k0, ca && Pa > k0, false);
    }
  };
  const int kctx = td ? min(khi, klo + max(0, (pos0 - klo) / TK) * TK) : khi;
  auto chain_src = [&](int key) { return kv_off(pages, kvh, key, hs, ps); };
  for (int k0 = kfirst; k0 < kctx; k0 += TK) {
    __syncthreads();  // the previous tile is consumed (and the queries are staged, the first time)
    bool f8 = false;
    if constexpr (Q8) f8 = sealed_tile(k0, kctx);
    if (f8) attn_load_tile_f8<ROWS, TK>(sm, q8, k0, kvh, hs, ps, pages);
    else attn_load_tile<ROWS, TK>(sm, kcache, vcache, k0, khi, chain_src);
    __syncthreads();
    if constexpr (Q8) {
      if (f8) {
        if (active) tile_compute(k0, Pa > k0);  // every row of the block is past the tile's unit
      } else {
        tile_q8(k0, true, [&] { attn_load_tile<ROWS, TK>(sm, kcache, vcache, k0, khi, chain_src); });
      }
    } else {
      tile_rows(k0, true, true);
    }
  }
  // The tail of a tree: per variant present in this block, its tile, taken by that variant's rows only.
  if (td != nullptr && kctx < khi) {
    const int first_m = r0 / G;
    for (int k0 = kctx; k0 < khi; k0 += TK) {
      for (int v = 0; v < td->nvar; ++v) {
        bool any = false;
        for (int m = first_m; m <= last_m; ++m) any = any || td->var[m] == v;
        if (!any) continue;
        const int* path = td->path[v];
        auto tree_src = [&](int key) {
          const int src = key < pos0 ? key : pos0 + path[min(key - pos0, kMaxSeqRows - 1)];
          return kv_off(pages, kvh, src, hs, ps);
        };
        __syncthreads();
        attn_load_tile<ROWS, TK>(sm, kcache, vcache, k0, khi, tree_src);
        __syncthreads();
        const bool cv = va && td->var[ra / G] == v;
        if constexpr (Q8) tile_q8(k0, cv, [&] { attn_load_tile<ROWS, TK>(sm, kcache, vcache, k0, khi, tree_src); });
        else tile_rows(k0, cv, true);
      }
    }
  }
  if (!active || !va) return;
  // Partial state per row, in attention_combine_kernel's layout.
  float* part = scratch + ((static_cast<size_t>(ra / G) * Hq + kvh * G + ra % G) * splits + split) * (D + 2);
  if (half == 0) {
    part[0] = ma;
    part[1] = la;
  }
#pragma unroll
  for (int i = 0; i < D / 16; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) part[2 + 16 * i + 2 * j + half] = o[i][j];
}

// attention_combine_kernel's output (row-head mh, dimension d): the CUDA combine (the partials of `splits` key ranges
// added in range order, the empty ones skipped). CG has no meaning without the megakernel: plain loads either way.
template <bool CG>
__device__ __forceinline__ float attention_combine_body(const float* __restrict__ scratch, int splits, int D, int mh,
                                                        int d) {
  const float* base = scratch + static_cast<size_t>(mh) * splits * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, base[s * (D + 2)]);
  float L = 0.f, A = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float ms = base[s * (D + 2)];
    if (ms == -INFINITY) continue;
    const float c = __expf(ms - M);
    L += base[s * (D + 2) + 1] * c;
    A += base[s * (D + 2) + 2 + d] * c;
  }
  return A / L;
}

