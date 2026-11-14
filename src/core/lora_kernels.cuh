// The LoRA kernels (specs/DREAMFERENCE_RAGON_ENGINE_LORA.md §3): a site's shrink, t = x A^T over its stacked A blocks,
// and its expand, y += scale * B t, each B segment into its rows of its output. The base projection has run over the
// whole batch already; these touch only the rows whose sequence is on an adapter, each with its own slot's weights.
//
// Row-invariant like the rest of the engine: a row's t and its delta do not depend on how many rows the launch has,
// where the row sits or which other adapters share the batch (a fixed per-thread order over K and a fixed reduction
// tree, no atomics), so a verify's rows equal the decode's and speculation stays exact.
#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/backend.hpp"  // the interface types; the CUDA-only device helpers at the end
// test __CUDACC__ (inline PTX), the declarations above them are the backend interface (BACKEND §2).

#include "core/kernels.cuh"

namespace ragon::kernels {

constexpr int kLoraRankRows = 64;  // t's row stride: kLoraMaxBlocks (4) x kLoraMaxRank (16)
constexpr int kLoraSegs = 4;       // kLoraMaxSegs
constexpr int kLoraOuts = 3;       // kLoraMaxOuts

// One B segment: rows [n0, n1) of output `out`, from A block `block`; its B, [n1 - n0][rank], at b + b_off.
struct LoraSegDev {
  int out = 0, n0 = 0, n1 = 0, block = 0;
  long long b_off = 0;
};

// One site of one layer in one slot of the GPU pool. a == nullptr: the slot's adapter does not touch the site.
struct LoraSiteDev {
  const gpu::bf16* a = nullptr;  // [blocks * rank][K]
  const gpu::bf16* b = nullptr;
  int K = 0, rank = 0, blocks = 0, nsegs = 0;
  float scale = 0.f;
  LoraSegDev segs[kLoraSegs];
};

// Which slot each row is on: the batch's sequences (row0 as SeqBatch's), each one slot or -1 (the base). A captured
// graph reads the slots from slot_dev, written before every replay beside the positions; eager launches take slot.
struct LoraRows {
  int n = 0;
  int row0[kMaxSeqs + 1] = {};
  int slot[kMaxSeqs];
  const int* slot_dev = nullptr;
  LoraRows() {
    for (int& s : slot) s = -1;
  }
};

// The outputs a site's segments add into, FP32 [M][ld]; a null one is skipped.
struct LoraOuts {
  float* y[kLoraOuts] = {};
  int ld[kLoraOuts] = {};
};

// One output's part of a site on the prefill path, for the GEMM that writes that output: y[n] += scale * B t in the
// GEMM's own epilogue. Segments as LoraSegDev's, t_off the place of the segment's A block in a row of t.
struct LoraGemm {
  struct Seg {
    int n0 = 0, n1 = 0, t_off = 0;
    long long b_off = 0;
  };
  const float* t = nullptr;  // [M][kLoraRankRows]
  const gpu::bf16* b = nullptr;
  int rank = 0, nsegs = 0;
  float scale = 0.f;
  Seg segs[kLoraSegs];
};

// t[r][j] = sum_k x[r][k] A[j][k] for every row r on a slot, j < blocks * rank of that slot's site. `site` is the
// site in slot 0; slot s's is site[s * slot_stride]. x is FP32 [M][ldx]; K a multiple of 4.
void lora_shrink(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                 int a_rows, float* t, gpu::Stream stream);

// y_out[r][n] += scale * sum_j B[n - n0][j] t[r][block * rank + j] for every row r on a slot and every segment of its
// site.
void lora_expand(const float* t, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                 const LoraOuts& outs, gpu::Stream stream);

// The prefill path (a prompt's tokens, whatever the chunk's size): tensor cores on BF16, as the reference computes an
// adapter. The shrink, K a multiple of 16; the expand has no kernel of its own, its rows are added by the GEMM that
// writes the output (prefill_gemm_fp8_lora), so a chunk's y is written once and not read back.
void lora_shrink_prefill(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                         float* t, gpu::Stream stream);

// d's segments split by output, with t the shrink's output; an output with no segment gets nsegs == 0.
void lora_gemm_parts(const LoraSiteDev& d, const float* t, LoraGemm (&parts)[kLoraOuts]);

// prefill_gemm_fp8, every row of M on the adapter: y = [y +] alpha * W x + scale * B t. Segment bounds multiples of 8.
void prefill_gemm_fp8_lora(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                           bool accumulate, const LoraGemm& lora, gpu::Stream s);

#ifdef __CUDACC__
// Sequence q's slot (-1: the base), from slot_dev in a captured graph.
__device__ __forceinline__ int lora_seq_slot(const LoraRows& rows, int q) { return rows.slot_dev ? rows.slot_dev[q] : rows.slot[q]; }

// Row r's slot.
__device__ __forceinline__ int lora_row_slot(const LoraRows& rows, int r) {
  int i = 0;
  while (i + 1 < rows.n && r >= rows.row0[i + 1]) ++i;
  return lora_seq_slot(rows, i);
}

// One BF16 tensor-core step: c[0..3] += rows g and g + 8 of a 16-row a by column g of an 8-column b, over 16 of k.
__device__ __forceinline__ void lora_mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Elements j and j + 1 of a BF16 row of R elements, zero past its end.
__device__ __forceinline__ uint32_t lora_b_pair(const __nv_bfloat16* row, int j, int R) {
  if (j + 1 < R) return *reinterpret_cast<const uint32_t*>(row + j);
  if (j < R) return *reinterpret_cast<const uint16_t*>(row + j);
  return 0u;
}

// y + scale * acc, rounded as written: the expand kernel and the megakernels' epilogues (RAGON_LORA_MK) add an
// adapter's delta with this one formula, acc being the fmaf chain over j < rank in order, so neither can contract
// into a different fma and the fused and unfused paths stay bit for bit.
__device__ __forceinline__ float lora_add(float y, float acc, float scale) { return __fadd_rn(y, __fmul_rn(scale, acc)); }
#endif

}  // namespace ragon::kernels
