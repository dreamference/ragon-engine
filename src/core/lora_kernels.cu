// The LoRA kernels (lora_kernels.cuh). Both are bandwidth-bound on the adapter's weights: a decode step reads each
// slot's A and B once per site (rank 16: about 96 MB for every site of the 64 layers), the rows' inputs from L2.
#include "core/elem.cuh"
#include "core/launch.cuh"
#include "core/lora_kernels.cuh"


namespace ragon::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kTileRows = 8;  // rows per block: the shrink's accumulators, the expand's t tile
constexpr int kMaxRank = kLoraRankRows / 4;
constexpr int kPrefillRows = 16;  // rows per block of the prefill kernels: an MMA's m

static_assert(kTileRows < 32 && kPrefillRows < 32, "a tile's rows are one mask");

// The tile's rows of sequence q, as a mask (the same for every thread of the block).
template <int TILE_ROWS = kTileRows>
__device__ __forceinline__ unsigned tile_mask(const LoraRows& rows, int q, int r0, int M) {
  const int a = max(rows.row0[q], r0), b = min(q + 1 < rows.n ? rows.row0[q + 1] : M, min(r0 + TILE_ROWS, M));
  return a < b ? ((1u << (b - a)) - 1u) << (a - r0) : 0u;
}

// One A row j of sequence q's site against the tile's rows of q. Each thread runs over K in a fixed stride, then the
// warps' butterflies and the warps in order: a row's sum never depends on the tile it shares.
__global__ void __launch_bounds__(kThreads) lora_shrink_kernel(const float* __restrict__ x, int ldx, int M, LoraRows rows,
                                                                const LoraSiteDev* __restrict__ site, int slot_stride,
                                                                float* __restrict__ t) {
  pdl_begin();  // first statement: see launch.cuh (x and the slots are a previous kernel's output)
  const int q = blockIdx.z, j = blockIdx.x, r0 = blockIdx.y * kTileRows;
  const unsigned mask = tile_mask(rows, q, r0, M);
  if (!mask) return;
  const int s = lora_seq_slot(rows, q);
  if (s < 0) return;
  const LoraSiteDev& d = site[size_t(s) * slot_stride];
  if (!d.a || j >= d.blocks * d.rank) return;
  const __nv_bfloat16* a = d.a + size_t(j) * d.K;
  float acc[kTileRows];
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) acc[i] = 0.f;
  for (int k = threadIdx.x * 4; k < d.K; k += kThreads * 4) {
    const float2 a01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a + k));
    const float2 a23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a + k + 2));
#pragma unroll
    for (int i = 0; i < kTileRows; ++i) {
      if (!(mask >> i & 1u)) continue;
      const float4 v = *reinterpret_cast<const float4*>(x + size_t(r0 + i) * ldx + k);
      acc[i] = fmaf(v.x, a01.x, acc[i]);
      acc[i] = fmaf(v.y, a01.y, acc[i]);
      acc[i] = fmaf(v.z, a23.x, acc[i]);
      acc[i] = fmaf(v.w, a23.y, acc[i]);
    }
  }
  __shared__ float part[kThreads / 32][kTileRows];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) {
    float v = acc[i];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) part[warp][i] = v;
  }
  __syncthreads();
  const int i = threadIdx.x;
  if (i < kTileRows && (mask >> i & 1u)) {
    float v = 0.f;
    for (int w = 0; w < kThreads / 32; ++w) v += part[w][i];
    t[size_t(r0 + i) * kLoraRankRows + j] = v;
  }
}

// One output row n of one segment of sequence q's site, for the tile's rows of q: y += scale * (B[n] . t[block]). The
// segments of the outputs the pass computes lie end to end along blockIdx.x, so few blocks of a launch are empty.
__global__ void __launch_bounds__(kThreads) lora_expand_kernel(const float* __restrict__ t, int M, LoraRows rows,
                                                                const LoraSiteDev* __restrict__ site, int slot_stride,
                                                                LoraOuts outs, int tiles) {
  pdl_begin();  // first statement: see launch.cuh (y is the GEMM's output, read and written here)
  const int q = blockIdx.y / tiles, r0 = (blockIdx.y % tiles) * kTileRows;
  const unsigned mask = tile_mask(rows, q, r0, M);
  if (!mask) return;
  const int s = lora_seq_slot(rows, q);
  if (s < 0) return;
  const LoraSiteDev& d = site[size_t(s) * slot_stride];
  if (!d.a) return;
  int sg = 0, nb = blockIdx.x;
  for (; sg < d.nsegs; ++sg) {
    const LoraSegDev& g = d.segs[sg];
    if (!outs.y[g.out]) continue;  // an output the pass does not compute (the drafter's context rows have no q)
    const int blocks = (g.n1 - g.n0 + kThreads - 1) / kThreads;
    if (nb < blocks) break;
    nb -= blocks;
  }
  if (sg == d.nsegs) return;  // the grid is a bound
  const LoraSegDev seg = d.segs[sg];
  const int R = d.rank;
  __shared__ float ts[kTileRows][kMaxRank];
  for (int e = threadIdx.x; e < kTileRows * kMaxRank; e += kThreads) {
    const int i = e / kMaxRank, jj = e % kMaxRank;
    ts[i][jj] = (mask >> i & 1u) && jj < R ? t[size_t(r0 + i) * kLoraRankRows + seg.block * R + jj] : 0.f;
  }
  __syncthreads();
  const int n = seg.n0 + nb * kThreads + threadIdx.x;
  if (n >= seg.n1) return;
  const __nv_bfloat16* b = d.b + seg.b_off + size_t(n - seg.n0) * R;
  float bv[kMaxRank];
#pragma unroll
  for (int jj = 0; jj < kMaxRank; ++jj) bv[jj] = jj < R ? __bfloat162float(b[jj]) : 0.f;
  float* y = outs.y[seg.out];
  const int ld = outs.ld[seg.out];
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) {
    if (!(mask >> i & 1u)) continue;
    float acc = 0.f;
#pragma unroll
    for (int jj = 0; jj < kMaxRank; ++jj)
      if (jj < R) acc = fmaf(bv[jj], ts[i][jj], acc);
    float& out = y[size_t(r0 + i) * ld + n];
    out = lora_add(out, acc, d.scale);
  }
}

// The prefill path's shrink: tensor cores on BF16, as the reference computes an adapter. A chunk's x outgrows L2 and its
// rows make FP32 FMAs the prefill's largest cost, so this reads x once per tile and takes m16n8k16 MMAs: a block is 16
// rows, each MMA 16 rows x 8 outputs over 16 of K. A row's values do not depend on the launch's size, since the MMA's
// arithmetic is the same for every row and K is walked in the same steps. It gives a BF16 reference's answer, not the
// decode kernels' bits, and need not: a prompt's tokens take the prefill path whatever the chunk's size.
// t = x A^T for one tile of rows on slot s: a warp computes 16 of the A rows (a block is one warp, blockIdx.y of 4).
__global__ void __launch_bounds__(32)
    lora_shrink_prefill_kernel(const float* __restrict__ x, int ldx, int M, LoraRows rows, const LoraSiteDev* __restrict__ site,
                               int slot_stride, float* __restrict__ t) {
  pdl_begin();  // first statement: see launch.cuh (x and the slots are a previous kernel's output)
  constexpr int NF = 2;  // n8 fragments per warp: 4 warps x 2 x 8 = kLoraRankRows
  const int q = blockIdx.z, r0 = blockIdx.x * kPrefillRows, warp = blockIdx.y;
  const unsigned mask = tile_mask<kPrefillRows>(rows, q, r0, M);
  if (!mask) return;
  const int s = lora_seq_slot(rows, q);
  if (s < 0) return;
  const LoraSiteDev& d = site[size_t(s) * slot_stride];
  const int nj = d.a ? d.blocks * d.rank : 0;
  if (warp * NF * 8 >= nj) return;
  const int lane = threadIdx.x & 31, g = lane >> 2, tl = lane & 3;
  const bool on0 = mask >> g & 1u, on1 = mask >> (g + 8) & 1u;
  const float* xa = x + size_t(r0 + g) * ldx;
  const float* xb = x + size_t(r0 + g + 8) * ldx;
  float acc[NF][4] = {};
  for (int k = 0; k < d.K; k += 16) {
    const uint32_t a0 = on0 ? elem::pack_bf16x2(xa + k + 2 * tl) : 0u;
    const uint32_t a1 = on1 ? elem::pack_bf16x2(xb + k + 2 * tl) : 0u;
    const uint32_t a2 = on0 ? elem::pack_bf16x2(xa + k + 8 + 2 * tl) : 0u;
    const uint32_t a3 = on1 ? elem::pack_bf16x2(xb + k + 8 + 2 * tl) : 0u;
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      const int col = (warp * NF + f) * 8 + g;
      uint32_t b0 = 0u, b1 = 0u;
      if (col < nj) {
        const __nv_bfloat16* a = d.a + size_t(col) * d.K + k + 2 * tl;
        b0 = *reinterpret_cast<const uint32_t*>(a);
        b1 = *reinterpret_cast<const uint32_t*>(a + 8);
      }
      lora_mma(acc[f], a0, a1, a2, a3, b0, b1);
    }
  }
#pragma unroll
  for (int f = 0; f < NF; ++f)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int col = (warp * NF + f) * 8 + 2 * tl + e;
      if (col >= nj) continue;
      if (on0) t[size_t(r0 + g) * kLoraRankRows + col] = acc[f][e];
      if (on1) t[size_t(r0 + g + 8) * kLoraRankRows + col] = acc[f][2 + e];
    }
}

}  // namespace

void lora_shrink(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                 int a_rows, float* t, cudaStream_t stream) {
  if (M <= 0 || rows.n <= 0) return;
  const dim3 grid(a_rows, (M + kTileRows - 1) / kTileRows, rows.n);
  launch_kernel("lora_shrink", lora_shrink_kernel, grid, dim3(kThreads), 0, stream, x, ldx, M, rows, site,
                slot_stride, t);
}

void lora_expand(const float* t, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                 const LoraOuts& outs, cudaStream_t stream) {
  if (M <= 0 || rows.n <= 0) return;
  const int tiles = (M + kTileRows - 1) / kTileRows;
  int blocks = kLoraSegs;  // each segment's last block, however it falls
  for (int o = 0; o < kLoraOuts; ++o)
    if (outs.y[o]) blocks += outs.ld[o] / kThreads;
  const dim3 grid(blocks, rows.n * tiles);
  launch_kernel("lora_expand", lora_expand_kernel, grid, dim3(kThreads), 0, stream, t, M, rows, site, slot_stride,
                outs, tiles);
}

void lora_shrink_prefill(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                         float* t, cudaStream_t stream) {
  if (M <= 0 || rows.n <= 0) return;
  const dim3 grid((M + kPrefillRows - 1) / kPrefillRows, kLoraRankRows / 16, rows.n);
  launch_kernel("lora_shrink_prefill", lora_shrink_prefill_kernel, grid, dim3(32), 0, stream, x, ldx, M, rows, site,
                slot_stride, t);
}

void lora_gemm_parts(const LoraSiteDev& d, const float* t, LoraGemm (&parts)[kLoraOuts]) {
  for (LoraGemm& p : parts) p = LoraGemm{};
  if (!d.a) return;
  for (int i = 0; i < d.nsegs; ++i) {
    const LoraSegDev& g = d.segs[i];
    LoraGemm& p = parts[g.out];
    p.t = t;
    p.b = d.b;
    p.rank = d.rank;
    p.scale = d.scale;
    p.segs[p.nsegs++] = LoraGemm::Seg{g.n0, g.n1, g.block * d.rank, g.b_off};
  }
}

}  // namespace ragon::kernels
