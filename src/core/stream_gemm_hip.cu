// The weight-streaming GEMM on HIP (gfx1151, wave32; BACKEND §7, W1): stream_gemm.cu's entry points for the HIP
// build, compiled in its place. The CUDA file's kernels are built around sm_121's mma.sync, ldmatrix, the coded
// scales and the persistent kernels; this one is the layout and kernel of the gfx1151 gate (bench/amd/vg.hip, AMD
// §6.5: K-major tiles, the E2M1/E4M3 decode in software, WMMA 16x16x16 FP16 into FP32, no persistent barrier), so it
// is a file of its own rather than guards through the CUDA one, whose objects stay as they are.
//
// Layout (the HIP tiled blob; tiled_bytes_* are the CUDA sizes): a tile is 16 rows x 64 K. Tiles are numbered
// K-major, tile = (k / 64) * (N / 16) + n / 16, so the waves of a launch read neighbouring memory at every step.
//   NVFP4: [tiles][512] codes (row r's 32 bytes at r * 32, the checkpoint's byte order: low nibble first), then
//          [tiles][64] E4M3 scales (row r's four 16-value scales at r * 4).
//   FP8:   [tiles][1024] E4M3 (row r's 64 bytes at r * 64).
// The retile only moves bytes (BACKEND §4: values and scales bit for bit), one output byte per thread, so the blob
// is the same on every load. No coded scales on HIP (scale_code_table is false: the raw blob is uploaded).
//
// Exactness: E2M1 x E4M3 and E4M3 decode to FP16 exactly; WMMA's FP16 products are exact in FP32 and its
// accumulation order is fixed, the K order is fixed (one wave walks all of K for its tile), and every output element
// is computed the same way whatever M (rows past M are zeros in LDS), the launch's other targets or the block size:
// a row's bytes do not depend on M or on grouping, and two runs give the same bytes.
#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "backend/device.hpp"

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

constexpr int kTileK = 64;                 // K per tile
constexpr int kKC = 256;                   // K per LDS chunk
constexpr int kXPad = 16;                  // LDS row padding (halves)
constexpr int kXSt = kKC + kXPad;          // LDS row stride
constexpr int kRows = 16;                  // rows per pass (one WMMA tile); more rows run as 16-row passes
constexpr int kMaxTargets = 4;
constexpr int kFp4Waves = 8;               // the gate's NVFP4 block: 256 threads
constexpr int kFp8Waves = 32;              // the gate's FP8 block: 1024 threads
constexpr int kSplitBelow = 256;           // weights narrower than this many 16-row tiles split K ...
constexpr int kSplitWaves = 320;           // ... into the fewest equal parts giving at least this many waves

struct Targets {
  const uint8_t* w[kMaxTargets];
  float scale[kMaxTargets];
  float* y[kMaxTargets];
  int N[kMaxTargets];
  int b0[kMaxTargets + 1];  // first block of each target; b0[n] = the grid
  int S[kMaxTargets];       // K splits (1: none)
  int cb[kMaxTargets];      // column blocks per split
  float* part[kMaxTargets];  // split partials [tile][S][256]
  u32* cnt[kMaxTargets];
  int n;
};

__device__ __forceinline__ u32 as_u(v2h x) { return __builtin_bit_cast(u32, x); }
__device__ __forceinline__ v2h as_h(u32 x) { return __builtin_bit_cast(v2h, x); }

// 8 E2M1 codes (one u32, low nibble first) -> 8 halves times the block scale sc, through a byte table: the FP16
// forms of the eight magnitudes have a zero low byte, so one v_perm over two table registers gives four high bytes.
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
// 4 E4M3 bytes -> 4 halves (exact: E4M3 is a subset of FP16, the bits move and 2^8 restores the bias).
__device__ __forceinline__ void dec_fp8(u32 c, v2h* o) {
  const v2h k8 = {(hf)256.0f, (hf)256.0f};
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const u32 d = __builtin_amdgcn_perm(0u, c, 0x0C010C00u + j * 0x00020002u);
    const u32 r = ((d & 0x007F007Fu) << 7) | ((d & 0x00800080u) << 8);
    o[j] = as_h(r) * k8;
  }
}
// One E4M3 scale byte -> half, in both halves (exact).
__device__ __forceinline__ v2h e4m3_h2(u32 b) {
  const u32 r = ((b & 0x7Fu) << 7) | ((b & 0x80u) << 8);
  const hf v = __builtin_bit_cast(hf, (unsigned short)r) * (hf)256.0f;
  const v2h o = {v, v};
  return o;
}
__device__ __forceinline__ u32 xswap(u32 v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}
__device__ __forceinline__ void bar() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// One wave: the 16 weight rows of tile column nt for all of K, against the pass's 16 activation rows. x is the pass's
// first row; rows at or past M are staged as zeros.
template <bool FP4, int BW>
__global__ void __launch_bounds__(BW * 32) stream_kernel(const gpu::half* __restrict__ xg,
                                                         const float* __restrict__ xinv, int M, int K, Targets T) {
  __shared__ __attribute__((aligned(16))) hf XS[2 * kRows * kXSt];
  const hf* x = reinterpret_cast<const hf*>(xg);
  const int b = blockIdx.x;
  int ti = 0;
#pragma unroll
  for (int i = 1; i < kMaxTargets; ++i)
    if (i < T.n && b >= T.b0[i]) ti = i;
  const uint8_t* __restrict__ w = T.w[ti];
  const int N = T.N[ti];
  const u32 tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, r16 = lane & 15;
  const u32 ntiles = N / 16;
  const int S = T.S[ti], rel = b - T.b0[ti], s = rel / T.cb[ti];
  const u32 nt = (rel - s * T.cb[ti]) * BW + wave;
  const bool valid = nt < ntiles;  // wave-uniform
  const int kbeg = s * (K / S), kend = kbeg + K / S;  // S > 1 only when K / S is a whole number of chunks
  const uint8_t* __restrict__ sc = w + static_cast<size_t>(N) * K / 2;

  constexpr u32 NV = kRows * kKC / 8;  // 16-byte loads per chunk
  constexpr u32 PER = (NV + BW * 32 - 1) / (BW * 32);
  v4u pre[PER];
  auto xload = [&](int kb) {
    const int kc = min(kKC, kend - kb);
#pragma unroll
    for (u32 p = 0; p < PER; p++) {
      const u32 i = tid + p * BW * 32, r = i / (kKC / 8), c = (i % (kKC / 8)) * 8;
      v4u v = {0, 0, 0, 0};
      if (i < NV && (int)r < M && (int)c < kc) v = *(const v4u*)(x + (u64)r * K + kb + c);
      pre[p] = v;
    }
  };
  auto xstore = [&](hf* buf) {
#pragma unroll
    for (u32 p = 0; p < PER; p++) {
      const u32 i = tid + p * BW * 32, r = i / (kKC / 8), c = (i % (kKC / 8)) * 8;
      if (i < NV) *(v4u*)(buf + r * kXSt + c) = pre[p];
    }
  };

  v8f acc = {};
  // One 64-K step of the wave's tile: decode, swap halves, four WMMAs.
  auto step = [&](const hf* xs, int k0, int kk) {
    const u64 tile = (u64)((k0 + kk) / kTileK) * ntiles + nt;
    v8u own0, own1;
    if constexpr (FP4) {
      const v4u c = __builtin_nontemporal_load((const v4u*)(w + tile * 512 + r16 * 32 + half * 16));
      const u32 s2 = *(const unsigned short*)(sc + tile * 64 + r16 * 4 + half * 2);
      const v2h sa = e4m3_h2(s2 & 0xFF), sb = e4m3_h2(s2 >> 8);
      v2h o[8];
      dec_fp4(c.x, sa, o);
      dec_fp4(c.y, sa, o + 4);
#pragma unroll
      for (int e = 0; e < 8; e++) own0[e] = as_u(o[e]);
      dec_fp4(c.z, sb, o);
      dec_fp4(c.w, sb, o + 4);
#pragma unroll
      for (int e = 0; e < 8; e++) own1[e] = as_u(o[e]);
    } else {
      const uint8_t* base = w + tile * 1024 + r16 * 64 + half * 32;
      const v4u c0 = __builtin_nontemporal_load((const v4u*)base);
      const v4u c1 = __builtin_nontemporal_load((const v4u*)(base + 16));
      v2h o[8];
      dec_fp8(c0.x, o); dec_fp8(c0.y, o + 2); dec_fp8(c0.z, o + 4); dec_fp8(c0.w, o + 6);
#pragma unroll
      for (int e = 0; e < 8; e++) own0[e] = as_u(o[e]);
      dec_fp8(c1.x, o); dec_fp8(c1.y, o + 2); dec_fp8(c1.z, o + 4); dec_fp8(c1.w, o + 6);
#pragma unroll
      for (int e = 0; e < 8; e++) own1[e] = as_u(o[e]);
    }
    v8u oth0, oth1;
#pragma unroll
    for (int e = 0; e < 8; e++) { oth0[e] = xswap(own0[e]); oth1[e] = xswap(own1[e]); }
    v16h bq[4];
    bq[0] = __builtin_bit_cast(v16h, half ? oth0 : own0);
    bq[1] = __builtin_bit_cast(v16h, half ? oth1 : own1);
    bq[2] = __builtin_bit_cast(v16h, half ? own0 : oth0);
    bq[3] = __builtin_bit_cast(v16h, half ? own1 : oth1);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const v16h a = *(const v16h*)(xs + r16 * kXSt + kk + q * 16);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bq[q], acc);
    }
  };

  xload(kbeg);
  xstore(XS);
  bar();
  for (int k0 = kbeg; k0 < kend; k0 += kKC) {
    const int c = (k0 - kbeg) / kKC;
    const hf* xs = XS + (c & 1) * (kRows * kXSt);
    const bool more = k0 + kKC < kend;
    if (more) xload(k0 + kKC);
    if (valid) {
      if (k0 + kKC <= kend) {
#pragma unroll 2
        for (int kk = 0; kk < kKC; kk += kTileK) step(xs, k0, kk);
      } else {
        for (int kk = 0; kk < kend - k0; kk += kTileK) step(xs, k0, kk);
      }
    }
    if (more) xstore(XS + ((c + 1) & 1) * (kRows * kXSt));
    bar();
  }
  if (!valid) return;
  if (S > 1) {
    // Split K: every split stores its partial tile and takes a ticket; the last one adds the S partials in split
    // order (its own from registers: the same values), so the sum is fixed whichever block finishes last.
    float* __restrict__ part = T.part[ti] + static_cast<u64>(nt) * S * 256 + lane * 8;
    *(v8f*)(part + s * 256) = acc;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    u32 old = 0;
    if (lane == 0) old = __hip_atomic_fetch_add(T.cnt[ti] + nt, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    old = __builtin_amdgcn_readfirstlane(old);
    if (old != static_cast<u32>(S - 1)) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    v8f sum = s == 0 ? acc : *(const v8f*)part;
    for (int j = 1; j < S; ++j) sum += j == s ? acc : *(const v8f*)(part + j * 256);
    acc = sum;
    if (lane == 0) __hip_atomic_store(T.cnt[ti] + nt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  const float scale = T.scale[ti];
  float* __restrict__ y = T.y[ti];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    const int m = 2 * r + half;
    if (m < M) y[(u64)m * N + nt * 16 + r16] = acc[r] * scale * xinv[m];
  }
}

// The sparse head's gathered launch (stream_gemm_gather_nvfp4_kmajor): one wave per 16 output columns j0 .. j0 + 15 of
// [0, cap), lane r16's weight row n = rows[j0 + r16] read from its own tile ((k / 64) * (N / 16) + n / 16, row n % 16)
// instead of the tile column's rows. The staging, decode, swap, WMMA order over K and epilogue are stream_kernel's
// with S = 1, so a selected row's logit is bit for bit the full head's (WMMA's columns are independent: a column's
// result depends only on its own B values and A). Columns at or past *nrows read row 0 and are written -inf; a wave
// whose 16 columns are all past it only stages x.
template <int BW>
__global__ void __launch_bounds__(BW * 32) gather_kernel(const gpu::half* __restrict__ xg, const float* __restrict__ xinv,
                                                         int M, int K, const uint8_t* __restrict__ w, int N, float scale,
                                                         const int* __restrict__ rows, const int* __restrict__ nrows,
                                                         int cap, float* __restrict__ y) {
  __shared__ __attribute__((aligned(16))) hf XS[2 * kRows * kXSt];
  const hf* x = reinterpret_cast<const hf*>(xg);
  const u32 tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = lane >> 4, r16 = lane & 15;
  const u32 ntiles = N / 16;
  const int nr = *nrows;
  const int j0 = (blockIdx.x * BW + wave) * 16, j = j0 + r16;
  const bool valid = j0 < nr && j0 < cap;  // wave-uniform
  const int n = (j < nr && j < cap) ? rows[j] : 0;
  const u64 rt = n >> 4, rr = n & 15;
  const uint8_t* __restrict__ sc = w + static_cast<size_t>(N) * K / 2;
  const int kbeg = 0, kend = K;

  constexpr u32 NV = kRows * kKC / 8;
  constexpr u32 PER = (NV + BW * 32 - 1) / (BW * 32);
  v4u pre[PER];
  auto xload = [&](int kb) {
    const int kc = min(kKC, kend - kb);
#pragma unroll
    for (u32 p = 0; p < PER; p++) {
      const u32 i = tid + p * BW * 32, r = i / (kKC / 8), c = (i % (kKC / 8)) * 8;
      v4u v = {0, 0, 0, 0};
      if (i < NV && (int)r < M && (int)c < kc) v = *(const v4u*)(x + (u64)r * K + kb + c);
      pre[p] = v;
    }
  };
  auto xstore = [&](hf* buf) {
#pragma unroll
    for (u32 p = 0; p < PER; p++) {
      const u32 i = tid + p * BW * 32, r = i / (kKC / 8), c = (i % (kKC / 8)) * 8;
      if (i < NV) *(v4u*)(buf + r * kXSt + c) = pre[p];
    }
  };

  v8f acc = {};
  auto step = [&](const hf* xs, int k0, int kk) {
    const u64 tile = (u64)((k0 + kk) / kTileK) * ntiles + rt;
    v8u own0, own1;
    const v4u c = __builtin_nontemporal_load((const v4u*)(w + tile * 512 + rr * 32 + half * 16));
    const u32 s2 = *(const unsigned short*)(sc + tile * 64 + rr * 4 + half * 2);
    const v2h sa = e4m3_h2(s2 & 0xFF), sb = e4m3_h2(s2 >> 8);
    v2h o[8];
    dec_fp4(c.x, sa, o);
    dec_fp4(c.y, sa, o + 4);
#pragma unroll
    for (int e = 0; e < 8; e++) own0[e] = as_u(o[e]);
    dec_fp4(c.z, sb, o);
    dec_fp4(c.w, sb, o + 4);
#pragma unroll
    for (int e = 0; e < 8; e++) own1[e] = as_u(o[e]);
    v8u oth0, oth1;
#pragma unroll
    for (int e = 0; e < 8; e++) { oth0[e] = xswap(own0[e]); oth1[e] = xswap(own1[e]); }
    v16h bq[4];
    bq[0] = __builtin_bit_cast(v16h, half ? oth0 : own0);
    bq[1] = __builtin_bit_cast(v16h, half ? oth1 : own1);
    bq[2] = __builtin_bit_cast(v16h, half ? own0 : oth0);
    bq[3] = __builtin_bit_cast(v16h, half ? own1 : oth1);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const v16h a = *(const v16h*)(xs + r16 * kXSt + kk + q * 16);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bq[q], acc);
    }
  };

  xload(kbeg);
  xstore(XS);
  bar();
  for (int k0 = kbeg; k0 < kend; k0 += kKC) {
    const int c = (k0 - kbeg) / kKC;
    const hf* xs = XS + (c & 1) * (kRows * kXSt);
    const bool more = k0 + kKC < kend;
    if (more) xload(k0 + kKC);
    if (valid) {
      if (k0 + kKC <= kend) {
#pragma unroll 2
        for (int kk = 0; kk < kKC; kk += kTileK) step(xs, k0, kk);
      } else {
        for (int kk = 0; kk < kend - k0; kk += kTileK) step(xs, k0, kk);
      }
    }
    if (more) xstore(XS + ((c + 1) & 1) * (kRows * kXSt));
    bar();
  }
  if (j0 >= cap) return;
  const bool live = valid && j < nr;
#pragma unroll
  for (int r = 0; r < 8; r++) {
    const int m = 2 * r + half;
    if (m < M) y[(u64)m * cap + j] = live ? acc[r] * scale * xinv[m] : -INFINITY;
  }
}

void check_shape(int N, int K, const char* what) {
  if (N <= 0 || K <= 0 || N % 16 != 0 || K % 128 != 0)
    throw std::runtime_error(std::string(what) + ": N % 16 and K % 128 must be 0");
}

// K splits of a marked weight (stream_gemm_split_k), from its shape: none from kSplitBelow tiles up or when K is not
// whole chunks; else the fewest equal whole-chunk parts reaching kSplitWaves waves (all chunks when none does).
int splits(int N, int K) {
  const int tiles = N / 16, chunks = K / kKC;
  if (tiles >= kSplitBelow || K % kKC || chunks < 2) return 1;
  for (int S = 2; S < chunks; ++S)
    if (chunks % S == 0 && tiles * S >= kSplitWaves) return S;
  return chunks;
}

// Split K is per blob, not per launch: only a blob the loader marks (stream_gemm_split_k) splits, alone or grouped,
// so its bytes are the same in every launch. Grouped targets fill the GPU without it (a split q|k|v ran 5% slower:
// its k and v blocks no longer filled the tail of q's), so only a matrix that runs alone with few tiles is marked.
// A marked blob's scratch (partials and tile counters) is made at the mark, outside any capture; retiling a blob at
// that address forgets it. The counters are per stream: kStreamSlots slots, each bound to the first stream that runs
// the blob through it, so the main, side, commit and fork streams may run one blob at once (a side prefill chunk
// beside the main step). One stream's launches are ordered, so a slot is never in use twice at a time. Binding
// allocates nothing and works under capture; a fifth stream throws.
constexpr int kStreamSlots = 4;
struct Scratch {
  int N = 0, K = 0, S = 0;
  float* part = nullptr;  // [kStreamSlots][tiles * S * 256]
  u32* cnt = nullptr;     // [kStreamSlots][tiles]
  gpu::Stream bound[kStreamSlots] = {};
  int nbound = 0;
};
struct Slot {
  float* part = nullptr;
  u32* cnt = nullptr;
};
std::mutex g_scratch_mu;
std::unordered_map<const void*, Scratch>& scratch_map() {
  static auto* m = new std::unordered_map<const void*, Scratch>();
  return *m;
}
void free_scratch(Scratch& e) {
  if (e.part) (void)gpu::device_free(e.part);
  if (e.cnt) (void)gpu::device_free(e.cnt);
  e = Scratch{};
}
// A new blob at `w` (retile): a mark there belonged to a freed blob.
void forget_split(const void* w) {
  std::lock_guard<std::mutex> lock(g_scratch_mu);
  auto it = scratch_map().find(w);
  if (it == scratch_map().end()) return;
  free_scratch(it->second);
  scratch_map().erase(it);
}
// The K splits of target w (1 when unmarked), and the launching stream's slot of its scratch.
int split_slot(const void* w, int N, int K, gpu::Stream st, Slot* slot) {
  std::lock_guard<std::mutex> lock(g_scratch_mu);
  auto it = scratch_map().find(w);
  if (it == scratch_map().end()) return 1;
  Scratch& e = it->second;
  if (e.N != N || e.K != K) return 1;  // the mark of a freed blob (a new one is retiled, which forgets it)
  int i = 0;
  while (i < e.nbound && e.bound[i] != st) ++i;
  if (i == e.nbound) {
    if (e.nbound == kStreamSlots) throw std::runtime_error("stream_gemm: a split-K blob ran on more than 4 streams");
    e.bound[e.nbound++] = st;
  }
  const size_t tiles = N / 16;
  slot->part = e.part + i * tiles * e.S * 256;
  slot->cnt = e.cnt + i * tiles;
  return e.S;
}

// The rows run in passes of 16 (each row's arithmetic is the same in any pass); the targets of one launch share x.
template <bool FP4>
void dispatch(const gpu::half* x, const float* xinv, int M, const StreamTarget* t, int n, int K, gpu::Stream s) {
  if (M < 1 || M > kMaxStreamRows) throw std::runtime_error("stream_gemm: M must be in [1, 64]");
  if (n < 1 || n > kMaxTargets) throw std::runtime_error("stream_gemm: 1 to 4 targets");
  constexpr int BW = FP4 ? kFp4Waves : kFp8Waves;
  Targets T{};
  T.n = n;
  int blocks = 0;
  for (int i = 0; i < n; ++i) {
    check_shape(t[i].N, K, "stream_gemm");
    if (FP4 && t[i].wscale) throw std::runtime_error("stream_gemm: coded NVFP4 blobs are not built on HIP");
    T.w[i] = t[i].w;
    T.scale[i] = t[i].scale;
    T.N[i] = t[i].N;
    T.b0[i] = blocks;
    T.cb[i] = (t[i].N / 16 + BW - 1) / BW;
    Slot sc;
    T.S[i] = split_slot(t[i].w, t[i].N, K, s, &sc);
    T.part[i] = sc.part, T.cnt[i] = sc.cnt;
    blocks += T.S[i] * T.cb[i];
  }
  T.b0[n] = blocks;
  for (int m0 = 0; m0 < M; m0 += kRows) {
    for (int i = 0; i < n; ++i) T.y[i] = t[i].y + static_cast<size_t>(m0) * t[i].N;
    launch_kernel("stream_gemm", stream_kernel<FP4, BW>, dim3(blocks), dim3(BW * 32), 0, s,
                  x + static_cast<size_t>(m0) * K, xinv + m0, std::min(kRows, M - m0), K, T);
  }
}

// Retile: one output byte per thread (bytes move, nothing is computed).
__global__ void retile_nvfp4_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ wscale,
                                    uint8_t* __restrict__ out, int N, int K) {
  const size_t ntiles = static_cast<size_t>(N / 16), tiles = ntiles * (K / kTileK);
  const size_t total = tiles * 576;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    if (i < tiles * 512) {
      const size_t tile = i / 512;
      const int byte = static_cast<int>(i % 512), r = byte / 32, j = byte % 32;
      const size_t kg = tile / ntiles, n = (tile % ntiles) * 16 + r;
      out[i] = w[n * (K / 2) + kg * 32 + j];
    } else {
      const size_t o = i - tiles * 512, tile = o / 64;
      const int byte = static_cast<int>(o % 64), r = byte / 4, q = byte % 4;
      const size_t kg = tile / ntiles, n = (tile % ntiles) * 16 + r;
      out[i] = wscale[n * (K / 16) + kg * 4 + q];
    }
  }
}

__global__ void retile_fp8_kernel(const uint8_t* __restrict__ w, uint8_t* __restrict__ out, int N, int K) {
  const size_t ntiles = static_cast<size_t>(N / 16), total = static_cast<size_t>(N) * K;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t tile = i / 1024;
    const int byte = static_cast<int>(i % 1024), r = byte / 64, j = byte % 64;
    const size_t kg = tile / ntiles, n = (tile % ntiles) * 16 + r;
    out[i] = w[n * K + kg * 64 + j];
  }
}

__constant__ float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// Tiled -> BF16 row-major (the prefill path's BF16 input), the CUDA kernels' arithmetic: E2M1 x (E4M3 x scale2).
__global__ void dequant_tiled_nvfp4_kernel(const uint8_t* __restrict__ tw, float scale2, gpu::bf16* __restrict__ out,
                                           int N, int K) {
  const size_t ntiles = static_cast<size_t>(N / 16), tiles = ntiles * (K / kTileK);
  const size_t pairs = static_cast<size_t>(N) * K / 2;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < pairs;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t n = i / (K / 2);
    const int k = static_cast<int>(i % (K / 2)) * 2;
    const size_t tile = static_cast<size_t>(k / kTileK) * ntiles + n / 16;
    const int r = static_cast<int>(n % 16), kk = k % kTileK;
    const uint8_t b = tw[tile * 512 + r * 32 + kk / 2];
    const float s = gpu::e4m3_to_float(tw[tiles * 512 + tile * 64 + r * 4 + kk / 16]) * scale2;
    out[n * K + k] = gpu::bf16_from_float(kE2M1[b & 0xf] * s);
    out[n * K + k + 1] = gpu::bf16_from_float(kE2M1[b >> 4] * s);
  }
}

__global__ void dequant_tiled_fp8_kernel(const uint8_t* __restrict__ tw, float wscale, gpu::bf16* __restrict__ out,
                                         int N, int K) {
  const size_t ntiles = static_cast<size_t>(N / 16), total = static_cast<size_t>(N) * K;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t n = i / K;
    const int k = static_cast<int>(i % K);
    const size_t tile = static_cast<size_t>(k / kTileK) * ntiles + n / 16;
    out[i] = gpu::bf16_from_float(gpu::e4m3_to_float(tw[tile * 1024 + (n % 16) * 64 + k % kTileK]) * wscale);
  }
}

int grid_of(size_t n) { return static_cast<int>(std::min<size_t>((n + 255) / 256, 65535 * 4)); }

// rmsnorm_half and to_half_rows: stream_gemm.cu's kernels (wave32, so the same warps and the same sums).
__global__ void __launch_bounds__(1024) rmsnorm_half_kernel(const float* __restrict__ x, const gpu::bf16* __restrict__ w,
                                                            float* __restrict__ out, gpu::half* __restrict__ xh,
                                                            float* __restrict__ xinv, int H, float eps, bool gemma) {
  __shared__ float red[32];
  const size_t r = blockIdx.x;
  const float* row = x + r * H;
  float ss = 0.f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) ss += row[h] * row[h];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(gpu::kWarpMask, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < (blockDim.x + 31) / 32; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / H + eps);
  float mx = 0.f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    const float wv = __bfloat162float(w[h]);
    const float v = row[h] * rstd * (gemma ? 1.f + wv : wv);
    out[r * H + h] = v;
    mx = fmaxf(mx, fabsf(v));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(gpu::kWarpMask, mx, o));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = 0.f;
  for (int i = 0; i < (blockDim.x + 31) / 32; ++i) mx = fmaxf(mx, red[i]);
  int e = 0;
  if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);
  const float s = ldexpf(1.f, 15 - e);
  for (int h = threadIdx.x; h < H; h += blockDim.x) xh[r * H + h] = __float2half_rn(out[r * H + h] * s);
  if (threadIdx.x == 0) xinv[r] = ldexpf(1.f, e - 15);
}

// Row blockIdx.x over gridDim.y blocks: each block finds the whole row's largest magnitude (exact in any order) and
// converts its slice, so the values are those of one block per row for any split.
__global__ void __launch_bounds__(256) to_half_rows_kernel(const float* __restrict__ x, int K, int slice,
                                                           gpu::half* __restrict__ out, float* __restrict__ xinv) {
  __shared__ float red[8];
  const size_t r = blockIdx.x;
  const float* row = x + r * K;
  const int tid = threadIdx.x;
  float mx = 0.f;
  for (int k = tid; k < K; k += 256) mx = fmaxf(mx, fabsf(row[k]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(gpu::kWarpMask, mx, o));
  if ((tid & 31) == 0) red[tid >> 5] = mx;
  __syncthreads();
  mx = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) mx = fmaxf(mx, red[i]);
  int e = 0;
  if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);  // mx < 2^e
  const float s = ldexpf(1.f, 15 - e);
  const int k0 = blockIdx.y * slice, k1 = min(K, k0 + slice);
  for (int k = k0 + tid; k < k1; k += 256) out[r * K + k] = __float2half_rn(row[k] * s);
  if (blockIdx.y == 0 && tid == 0) xinv[r] = ldexpf(1.f, e - 15);
}

}  // namespace

size_t tiled_bytes_nvfp4(int N, int K) { return static_cast<size_t>(N) * K / 2 + static_cast<size_t>(N) * K / 16; }
size_t tiled_bytes_fp8(int N, int K) { return static_cast<size_t>(N) * K; }
size_t tiled_bytes_nvfp4_coded(int, int) {
  throw std::runtime_error("tiled_bytes_nvfp4_coded: no coded NVFP4 blobs on HIP");
}

// No coded scales on HIP: every NVFP4 matrix is uploaded raw (model.cu's fallback).
bool scale_code_table(const uint8_t*, size_t, uint8_t (&)[kScaleCodes], uint8_t (&)[256]) { return false; }

void retile_nvfp4(const uint8_t* w, const uint8_t* wscale, uint8_t* out, int N, int K, gpu::Stream s) {
  check_shape(N, K, "retile_nvfp4");
  launch_kernel("retile_nvfp4", retile_nvfp4_kernel, dim3(grid_of(tiled_bytes_nvfp4(N, K))), dim3(256), 0, s, w,
                wscale, out, N, K);
  forget_split(out);
}

void retile_fp8(const uint8_t* w, uint8_t* out, int N, int K, gpu::Stream s) {
  check_shape(N, K, "retile_fp8");
  launch_kernel("retile_fp8", retile_fp8_kernel, dim3(grid_of(tiled_bytes_fp8(N, K))), dim3(256), 0, s, w, out, N, K);
  forget_split(out);
}

void retile_nvfp4_coded(const uint8_t*, const uint8_t*, const uint8_t*, uint8_t*, uint8_t*, int, int, gpu::Stream) {
  throw std::runtime_error("retile_nvfp4_coded: no coded NVFP4 blobs on HIP");
}

void dequant_tiled_nvfp4(const uint8_t* tw, float scale2, gpu::bf16* out, int N, int K, gpu::Stream s,
                         const uint8_t* raw_scales) {
  if (raw_scales) throw std::runtime_error("dequant_tiled_nvfp4: no coded NVFP4 blobs on HIP");
  check_shape(N, K, "dequant_tiled_nvfp4");
  launch_kernel("dequant_tiled_nvfp4", dequant_tiled_nvfp4_kernel, dim3(grid_of(static_cast<size_t>(N) * K / 2)),
                dim3(256), 0, s, tw, scale2, out, N, K);
}

void dequant_tiled_fp8(const uint8_t* tw, float wscale, gpu::bf16* out, int N, int K, gpu::Stream s) {
  check_shape(N, K, "dequant_tiled_fp8");
  launch_kernel("dequant_tiled_fp8", dequant_tiled_fp8_kernel, dim3(grid_of(static_cast<size_t>(N) * K)), dim3(256), 0,
                s, tw, wscale, out, N, K);
}

void rmsnorm_half(const float* x, const gpu::bf16* w, float* out, gpu::half* xh, float* xinv, int rows, int H, float eps,
                  bool gemma, gpu::Stream s) {
  launch_kernel("rmsnorm_half", rmsnorm_half_kernel, dim3(rows), dim3(1024), 0, s, x, w, out, xh, xinv, H, eps, gemma);
}

// No L2 run-ahead on HIP (the megakernel probe's; prefetch_l2 is a no-op here too).
void l2_prefetch_weights(const uint8_t*, bool, bool, int, int, size_t, gpu::Stream) {}

void to_half_rows(const float* x, int M, int K, gpu::half* out, float* xinv, gpu::Stream s) {
  static const int target = [] {
    const char* e = std::getenv("RAGON_HALF_BLOCKS");
    return e ? std::atoi(e) : 96;
  }();
  const int split = target <= 0 ? 1 : std::clamp((target + M - 1) / M, 1, std::max(1, K / 1024));
  const int slice = (K + split - 1) / split;
  launch_kernel("to_half_rows", to_half_rows_kernel, dim3(M, (K + slice - 1) / slice), dim3(256), 0, s, x, K, slice,
                out, xinv);
}

void stream_gemm_nvfp4(const gpu::half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
                       float scale2, float* y, int N, int K, gpu::Stream s) {
  const StreamTarget t{w, wscale, scale2, y, N};
  dispatch<true>(x, xinv, M, &t, 1, K, s);
}

void stream_gemm_fp8(const gpu::half* x, const float* xinv, int M, const uint8_t* w, float wscale, float* y, int N,
                     int K, gpu::Stream s) {
  const StreamTarget t{w, nullptr, wscale, y, N};
  dispatch<false>(x, xinv, M, &t, 1, K, s);
}

void stream_gemm_multi(bool nvfp4, const gpu::half* x, const float* xinv, int M, const StreamTarget* t, int n, int K,
                       gpu::Stream s) {
  if (nvfp4) dispatch<true>(x, xinv, M, t, n, K, s);
  else dispatch<false>(x, xinv, M, t, n, K, s);
}

void stream_gemm_gather_nvfp4_kmajor(const gpu::half* x, const float* xinv, int M, const uint8_t* w, float scale2,
                                     const int* rows, const int* nrows, int cap, float* y, int N, int K, gpu::Stream s) {
  check_shape(N, K, "stream_gemm_gather");
  if (cap < 16 || cap % 16 != 0) throw std::runtime_error("stream_gemm_gather: cap % 16 must be 0");
  if (M < 1 || M > kMaxStreamRows) throw std::runtime_error("stream_gemm_gather: M must be in [1, 64]");
  // The full head's bits need its K order: one wave over all of K. A blob that splits K (none at a head's width:
  // splits() is 1 from kSplitBelow tiles up) would sum its parts in another order.
  if (splits(N, K) != 1) throw std::runtime_error("stream_gemm_gather: a weight that splits K");
  constexpr int BW = kFp4Waves;
  const int blocks = (cap / 16 + BW - 1) / BW;
  for (int m0 = 0; m0 < M; m0 += kRows)
    launch_kernel("stream_gemm_gather", gather_kernel<BW>, dim3(blocks), dim3(BW * 32), 0, s,
                  x + static_cast<size_t>(m0) * K, xinv + m0, std::min(kRows, M - m0), K, w, N, scale2, rows, nrows,
                  cap, y + static_cast<size_t>(m0) * cap);
}

void stream_gemm_split_k(const uint8_t* w, int N, int K) {
  check_shape(N, K, "stream_gemm_split_k");
  const int S = splits(N, K);
  if (S == 1) return;
  std::lock_guard<std::mutex> lock(g_scratch_mu);
  Scratch& e = scratch_map()[w];
  free_scratch(e);
  const size_t tiles = N / 16;
  Scratch n{N, K, S};
  if (gpu::device_alloc(&n.part, kStreamSlots * tiles * S * 256 * sizeof(float)) != gpu::kSuccess ||
      gpu::device_alloc(&n.cnt, kStreamSlots * tiles * sizeof(u32)) != gpu::kSuccess ||
      gpu::memset_sync(n.cnt, 0, kStreamSlots * tiles * sizeof(u32)) != gpu::kSuccess) {
    free_scratch(n);
    scratch_map().erase(w);
    throw std::runtime_error("stream_gemm: split-K scratch allocation failed");
  }
  e = n;
}

}  // namespace ragon::kernels
