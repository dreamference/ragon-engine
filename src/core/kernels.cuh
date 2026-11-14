// The v0 kernels. Activations are FP32 between kernels, weights stay in the checkpoint's own encoding
// (NVFP4 for the MLP and the LM head, FP8 E4M3 for the attention projections, BF16 for the rest) and are
// dequantized in registers. Rows (M) are the tokens of one step: 1 for decode, up to a prefill chunk.
#pragma once

#include <cstdint>

#include "backend/backend.hpp"  // the interface types (gpu::Stream, gpu::bf16, gpu::half): one declaration for
// every backend, the CUDA one the .cu files of this directory (specs/DREAMFERENCE_RAGON_ENGINE_BACKEND.md).
// The device helpers below compile wherever the backend compiles device code (RAGON_GPU_DEVICE_CODE).

namespace ragon::kernels {

// A sequence position as a kernel argument. Eager launches pass the host's value (an int converts
// implicitly). A launch captured into a CUDA graph is replayed at every step, at a different position each
// time, so it passes `dev`: the kernel then reads the position from device memory (*dev + off), where the
// engine writes it before each replay. `host` is the host's view at launch time; it is never used for
// launch geometry inside a graph (kernels whose grid would depend on it take a fixed grid instead).
//
// The rope delta (images, specs/DREAMFERENCE_RAGON_ENGINE_VISION.md §20.1): after an image the target's RoPE position is
// the cache slot plus a negative delta. Only the target's attention prepare reads it (rope_value); the cache slot,
// the attention's key range and the drafter keep the slot (pos_value). `rdelta_dev` (a graph) or `rdelta` (eager).
struct DevPos {
  int host = 0;
  int off = 0;               // added to *dev
  const int* dev = nullptr;  // nullptr: the position is `host`
  int rdelta = 0;
  const int* rdelta_dev = nullptr;
  DevPos(int p) : host(p) {}  // NOLINT: implicit on purpose, so eager call sites pass a plain int
  DevPos(int p, const int* d, int o) : host(p), off(o), dev(d) {}
  DevPos plus(int k) const {
    DevPos q(host + k, dev, off + k);
    q.rdelta = rdelta;
    q.rdelta_dev = rdelta_dev;
    return q;
  }
  DevPos with_rope(int delta, const int* delta_dev) const {
    DevPos q = *this;
    q.rdelta = delta;
    q.rdelta_dev = delta_dev;
    return q;
  }
};
#ifdef RAGON_GPU_DEVICE_CODE
__device__ __forceinline__ int pos_value(const DevPos& p) { return p.dev ? *p.dev + p.off : p.host; }
__device__ __forceinline__ int rope_delta(const DevPos& p) { return p.rdelta_dev ? *p.rdelta_dev : p.rdelta; }
#endif

// Several sequences' rows in one launch (M4 stage 4, the batched verify and drafter): sequence i owns rows
// [row0[i], row0[i + 1]) of the launch, at positions pos(i) + (row - row0[i]). Passed by value, so a launch captured
// into a CUDA graph keeps the row split; the positions come from pos_dev (written before each replay) when set.
// Every row its own sequence at most (BATCHING P4d: a fan-out's forks decode one row each).
constexpr int kMaxSeqs = 32;
// P4 (BATCHING §5.3, --kv-pages): positions per KV page. A page holds them for every full-attention layer's K and V,
// [layer][K, V][kv head][kKvPageTokens][D]; a session's page table holds each page's element offset into the pool
// (uint64_t), one entry per kKvPageTokens positions, and a kernel given a null table reads the flat layout. A multiple
// of the attention's 32-key tile, so no tile crosses a page.
constexpr int kKvPageTokens = 64;
struct SeqBatch {
  int n = 0;
  int row0[kMaxSeqs + 1] = {};
  int pos[kMaxSeqs] = {};          // the host's positions (eager launches; also the launch geometry)
  const int* pos_dev = nullptr;    // a captured graph: sequence i's position is pos_dev[i]
  int rdelta[kMaxSeqs] = {};       // the sequences' rope deltas (DevPos::rdelta)
  const int* rdelta_dev = nullptr; // a captured graph: sequence i's rope delta is rdelta_dev[i]
  // The commit kernels only: sequence i's rows end at row1[i] when it is set (a wide step's accepted rows, which leave
  // gaps between the sequences), else at row0[i + 1].
  int row1[kMaxSeqs] = {};
};
template <typename T>
struct SeqPtrs {
  T* p[kMaxSeqs] = {};
};
// Runs of a batch that share pages (attention_rows_cascade): group g is sequences [seq0[g], seq1[g]), its keys below cut[g]
// are theirs alike. A group has at least two sequences.
inline constexpr int kMaxSeqGroups = kMaxSeqs / 2;
struct SeqGroups {
  int n = 0;
  int seq0[kMaxSeqGroups] = {}, seq1[kMaxSeqGroups] = {}, cut[kMaxSeqGroups] = {};
};
#ifdef RAGON_GPU_DEVICE_CODE
__device__ __forceinline__ int seq_of(const SeqBatch& b, int row) {
  int s = 0;
  while (s + 1 < b.n && row >= b.row0[s + 1]) ++s;
  return s;
}
__device__ __forceinline__ int seq_pos(const SeqBatch& b, int s) { return b.pos_dev ? b.pos_dev[s] : b.pos[s]; }
__device__ __forceinline__ int seq_rdelta(const SeqBatch& b, int s) { return b.rdelta_dev ? b.rdelta_dev[s] : b.rdelta[s]; }
#endif

// Programmatic dependent launch (launch.cuh) for the rows path's kernels (decode, verify, drafter, commit):
// off by default. Set before launching; it applies to every later launch, captured or not.
void set_pdl(bool on);
bool pdl_enabled();

// Largest M the weight-streaming GEMV kernels handle; larger M dequantizes to BF16 and uses gemm_bf16.
constexpr int kMaxGemvRows = 8;

// y[M][N] = scale2 * x[M][K] . W[N][K]^T, W NVFP4: packed [N][K/2] (low nibble first), E4M3 block
// scales [N][K/16]. K must be a multiple of 1024.
void gemv_nvfp4(const float* x, int M, const uint8_t* w, const uint8_t* wscale, float scale2, float* y,
                int N, int K, gpu::Stream s);
// y = wscale * x . W^T, W FP8 E4M3 [N][K]. K must be a multiple of 512.
void gemv_fp8(const float* x, int M, const uint8_t* w, float wscale, float* y, int N, int K,
              gpu::Stream s);
// y = x . W^T for narrow BF16 matrices and up to 64 rows (33 to 64 as two launches of the first 32 and the rest: the
// same arithmetic per row), row-invariant (the rows path). K % 1024 == 0.
// With w2/y2: a second matrix of the same shape and input in the same launch.
void bf16_rows(const float* x, int M, const gpu::bf16* w, float* y, int N, int K, gpu::Stream s,
               const gpu::bf16* w2 = nullptr, float* y2 = nullptr);
// The same with a BF16 input (rows of `x` K apart): the drafter's fc over the target features.
void bf16_rows_bf16in(const gpu::bf16* x, int M, const gpu::bf16* w, float* y, int N, int K, gpu::Stream s);
// y = x . W^T, W BF16 [N][K].
void gemv_bf16(const float* x, int M, const gpu::bf16* w, float* y, int N, int K, gpu::Stream s);

// ---- Weight-streaming GEMM on tensor cores for 1 to 64 rows (stream_gemm.cu): decode, verify, drafter. ----
// Row-invariant: a row's result is bit-for-bit the same whatever M is, so a token verified in a block of
// 16 gets exactly the numbers it would get decoded alone.
//
// Two limits (batching P1, specs/DREAMFERENCE_LING_ENGINE_BATCHING.md §5.1):
//   - kMaxStreamRows, the rows of one pass: a batched verify's sequences together (two trees of up to 24 rows each
//     since P1), and every buffer and pinned array sized by a pass's rows. 33 to 64 rows take the stream GEMMs' second
//     row group (stream_gemm.cu, GROUPS = 2: the weights read once for both groups, each row's arithmetic unchanged);
//   - kMaxSeqRows, the rows of one sequence: its draft block, its tree (TreeDesc), its commit, the drafter's ring slack,
//     and the megakernels' tiers (the persistent FFN, DeltaNet and attention kernels take one sequence's verify, or
//     a batch of two chains, up to 32 rows; a larger pass keeps the separate kernels, which compute the same bits).
constexpr int kMaxStreamRows = 64;
constexpr int kMaxSeqRows = 32;

// Tree verification (BACKLOG 26): one verify's rows form a tree under the anchor (row 0) instead of a chain. Row m
// sits at depth depth[m] (its position is pos0 + depth[m]) and its key and value at cache slot pos0 + m, as a chain's
// do; every ancestor of a row has a lower index. Each row is computed exactly as a chain holding its root-to-row path
// would compute it, bit for bit:
//   - attention: a row's keys from pos0 on are its path's (variant var[m], a root-to-leaf path through m, whose row
//     at depth j is path[var][j]), loaded at the positions the chain would hold them in;
//   - the conv: its taps are its ancestors' inputs (from the variant's path);
//   - the recurrence: walked along gdn paths, each a root-to-node path advanced row by row from the state, with a
//     leaf child computed from its parent's state without advancing it; gdn_step[p][i] = row | kLeaf | kWrite.
// Written by the host before each tree verify (a graph's memcpy node copies it from pinned memory).
constexpr int kMaxTreeVariants = 32;
struct TreeDesc {
  int n = 0;
  int depth[kMaxSeqRows] = {};
  int var[kMaxSeqRows] = {};
  int nvar = 0;
  int path[kMaxTreeVariants][kMaxSeqRows] = {};
  int ngdn = 0;
  int gdn_len[kMaxTreeVariants] = {};
  int gdn_step[kMaxTreeVariants][2 * kMaxSeqRows] = {};
  static constexpr int kLeaf = 1 << 8, kWrite = 1 << 9;
};
// The FP8 KV cache's read domain for one layer (--kv-fp8, LOWBIT §8; attention_body.cuh, kv_q8_*): the K and V scales
// and their reciprocals. ks == 0: the layer reads its KV as stored (BF16), the default path.
// k8/v8 (M2): the layer's FP8 mirror of the pool, E4M3 codes at the pool's element offsets (one byte each), valid for
// the pages whose sealed[id] is set (id: the page table's entry / unit); a sealed tile every row of the block reads as R(v)
// is loaded from it and decoded on chip (the same values as rounding the BF16 tile). Null: every tile from the pool.
struct KvQ8 {
  float ks = 0.f, vs = 0.f, kinv = 0.f, vinv = 0.f;
  const uint8_t* k8 = nullptr;
  const uint8_t* v8 = nullptr;
  const uint8_t* sealed = nullptr;
  unsigned long long unit = 0;
  bool on() const { return ks > 0.f; }
};
// M2 (LOWBIT §8): the FP8 mirror of the pages ids[0 .. n) (host list, copied as parameters, kKvSealMax a launch) for the
// full-attention slots [l0, l1): mirror = e4m3(pool * inv) with each slot's K and V reciprocal scales (kinv, vinv: device
// arrays per slot); then sealed[id] = 1 in a second launch, so a reader that sees the flag sees the whole mirror.
constexpr int kKvSealMax = 64;
constexpr int kMaxKvQ8Slots = 32;  // full-attention layers kv_seal takes
void kv_seal(const gpu::bf16* pool, uint8_t* mirror, uint8_t* sealed, const int* ids, int n, int l0, int l1,
             int Hkv, int D, size_t page_unit, const float* kinv, const float* vinv, gpu::Stream s);
// sealed[ids[i]] = 0 (host list): the pages' BF16 is about to change (a write, a copy into them, a new owner).
void kv_unseal(uint8_t* sealed, const int* ids, int n, gpu::Stream s);
// The verify's conv over a tree: post[m] from pre[m] and its ancestors' pre rows (or the window `state`, unwritten).
void gdn_conv_tree(const float* pre, float* post, const float* state, const gpu::bf16* w, int M, int C,
                   const TreeDesc* td, gpu::Stream s);
// The verify's recurrence over a tree, from state_in (unwritten).
void gdn_recurrent_tree(const float* mixed, const float* g, const float* beta, const float* state_in, float* out,
                        int H, int HV, const TreeDesc* td, int ngdn, gpu::Stream s);
// After a tree verify's accept: the accepted path's rows rows[j] (j < n, rows[j] >= j, rows[0] = 0) moved to row j in
// every buffer of `bufs` (row r of buffer i at base + (r + (pos_rel ? pos : row0)) * stride bytes, row_bytes long), in
// place, j ascending. row0: where a batched sequence's rows start in the buffers its batch shares (rows[] are its own).
// A paged KV buffer (P4): `pages` is its session's page table and row r is at base + pages[r / kKvPageTokens] *
// page_unit + (r % kKvPageTokens) * stride bytes (page_unit: the bytes of one table unit, an element).
struct GatherBuf {
  char* base;
  size_t stride;
  int row_bytes;
  int pos_rel;
  const uint64_t* pages = nullptr;
  int page_unit = 0;
};
void tree_gather(const GatherBuf* bufs, int nbufs, const int* rows, int n, DevPos pos, gpu::Stream s, int row0 = 0);
// P4: the KV of positions [from, to) between a session's pages (pool, its device page table) and a flat buffer
// [layer_kv][kv head][buf_len][D] whose first position is buf_start (the prefix KV store's layout); layer_kv counts
// every full-attention layer's K and V (2 x layers). to_buf: into the buffer. A copy: the bits are unchanged.
void kv_pages_copy(gpu::bf16* pool, const uint64_t* pages, int layer_kv, int Hkv, int D, gpu::bf16* buf,
                   int buf_start, int buf_len, int from, int to, bool to_buf, gpu::Stream s);
// P4b: table[idx[i]] = val[i] for i < n, on stream s, before the pass that reads the table. The entries travel as the
// kernel's parameters (kKvTableSetMax per launch, looped for more), never as a copy from a host buffer: a pinned
// mirror that the host changes again before an earlier queued copy has run would hand that copy the newer value. The
// table's pointer never changes, so a captured graph that baked it reads the new entries.
constexpr int kKvTableSetMax = 256;
void kv_table_set(uint64_t* table, const int* idx, const uint64_t* val, int n, gpu::Stream s);
// The weights stream_gemm reads are tiled (SPEC.md §8, stage 4): each (16-row tile, 128-value chunk) is
// one contiguous block in the order the warp's lanes read it, so every load covers 512 contiguous bytes
// (measured: the row-major pattern reads at 215 GB/s, contiguous blocks at 232). NVFP4 blocks carry
// their E4M3 scales after the values. N % 16 == 0 and K % 128 == 0.
size_t tiled_bytes_nvfp4(int N, int K);
size_t tiled_bytes_fp8(int N, int K);
void retile_nvfp4(const uint8_t* w, const uint8_t* wscale, uint8_t* out, int N, int K, gpu::Stream s);
void retile_fp8(const uint8_t* w, uint8_t* out, int N, int K, gpu::Stream s);
// B26, the NVFP4 scales as 6-bit codes: a tensor whose block scales take at most 64 distinct E4M3 values (every
// matrix of the target but one, and the drafter's) stores every (tile, chunk) block's 1,024 value bytes, then every
// block's 96 bytes of codes (a lane slot's two 6-bit codes are 12 bits, the first scale's low: lane L's slots L and
// L + 32 keep their low bytes at 2L and 2L + 1 and their high nibbles in byte 64 + L, so a lane reads one 16-bit
// word and one byte per tile, as many loads as the raw scales; the value blocks stay 128-byte aligned), 1,120 bytes
// per block instead of 1,152, with the tensor's 64-entry table (code -> E4M3 byte) beside it; the streaming GEMM
// looks each code up in shared memory, so the scales it multiplies by are bit for bit the checkpoint's (lossless).
// The prefill GEMM reads the raw scales from a separate copy (`scales`: 128 bytes per block, in the raw blob's
// order), which only the prefill path touches.
constexpr int kScaleCodes = 64;
// The code of every scale byte (0xff: unused) and the table, or false when the scales take more than 64 values.
bool scale_code_table(const uint8_t* scales, size_t n, uint8_t (&table)[kScaleCodes], uint8_t (&code_of)[256]);
size_t tiled_bytes_nvfp4_coded(int N, int K);
void retile_nvfp4_coded(const uint8_t* w, const uint8_t* wscale, const uint8_t* code_of_dev, uint8_t* out,
                        uint8_t* raw_scales, int N, int K, gpu::Stream s);
// Tiled weights -> BF16 row-major [N][K] (the prefill path's BF16 GEMM input).
void dequant_tiled_nvfp4(const uint8_t* tw, float scale2, gpu::bf16* out, int N, int K, gpu::Stream s,
                         const uint8_t* raw_scales = nullptr);  // raw_scales: the coded blob's (B26)
void dequant_tiled_fp8(const uint8_t* tw, float wscale, gpu::bf16* out, int N, int K, gpu::Stream s);

// ---- Prefill GEMMs for a prompt's chunks (prefill_gemm.cu): block-scaled tensor cores over the same tiled
// weights, with the activations quantized as production quantizes them (SPEC.md §11). ----
// x [M][K] -> NVFP4: q packed [M][K/2] (low nibble first), E4M3 scales sf [M][K/16], global scale
// 1 / input_scale. K % 16 == 0.
void quant_act_nvfp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s);
// The same for silu(g) * u (the FFN's activation, never written in FP32).
void silu_mul_quant_nvfp4(const float* g, const float* u, int M, int K, float input_scale, uint8_t* q, uint8_t* sf,
                          gpu::Stream s);
// x [M][K] -> FP8 E4M3 [M][K], x / input_scale saturated. K % 4 == 0.
void quant_act_fp8(const float* x, int M, int K, float input_scale, uint8_t* q, gpu::Stream s);
// Producers of the prefill GEMMs' inputs, fused with the quantization (the FP32 tensor is never written, except
// `out` of rmsnorm_quant when given): Gemma RMSNorm (H % 1024 == 0), the DeltaNet gated norm (D = 128) and the
// attention's sigmoid gate (n % 4 == 0).
void rmsnorm_quant(const float* x, const gpu::bf16* w, float* out, int M, int H, float eps, bool nvfp4,
                   float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s);
void gated_rmsnorm_fp8(const float* x, const float* z, const gpu::bf16* w, int rows, int D, float eps,
                       float input_scale, uint8_t* q, gpu::Stream s);
void sigmoid_mul_fp8(const float* x, const float* gate, size_t n, float input_scale, uint8_t* q, gpu::Stream s);
// y[M][0..N) (rows ldy apart) = alpha * xq . W^T, or += with `accumulate`; W tiled (retile_*), alpha =
// input_scale * the weight's global scale. N % 128 == 0, K % 128 == 0.
// wsf (B26): a coded blob's raw scales (retile_nvfp4_coded); nullptr when the blob carries them.
void prefill_gemm_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, float alpha, float* y, int ldy,
                        int N, int K, bool accumulate, gpu::Stream s, const uint8_t* wsf = nullptr);
void prefill_gemm_fp8(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                      bool accumulate, gpu::Stream s);
// The FFN's gate and up projections in one GEMM, straight into the down projection's input:
// NVFP4(silu(alpha_gate * xq . G^T) * (alpha_up * xq . U^T)) with global scale 1 / out_input_scale, into q_out
// [M][N/2] and sf_out [M][N/16]; the FP32 products are never written. N % 128 == 0, K % 128 == 0.
void prefill_swiglu_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* gate, float alpha_gate,
                          const uint8_t* up, float alpha_up, int N, int K, float out_input_scale, uint8_t* q_out,
                          uint8_t* sf_out, gpu::Stream s, const uint8_t* gate_sf = nullptr,
                          const uint8_t* up_sf = nullptr);

// ya = x . Wa^T, yb = x . Wb^T for two BF16 [48][K] matrices and any M, row-invariant (the prefill path's
// DeltaNet a and b projections).
void narrow_bf16(const float* x, int M, const gpu::bf16* wa, const gpu::bf16* wb, float* ya, float* yb, int N,
                 int K, gpu::Stream s);

// ---- --prefill-bf16 (EngineOptions::prefill_bf16, off by default; MILESTONES §14's exactness policy): the
// prefill's DeltaNet intermediates in BF16 where production keeps BF16 (M2 §8 item 3, cited as §7.3 in the
// backlog's B34 row). Each is a new instantiation of the kernel above or below it over the element type
// (elem.cuh); the FP32 kernels are untouched. ----
// rmsnorm_quant with the FP32 copy replaced by a BF16 one (the a/b projections' input; the same values
// narrow_bf16 rounds its FP32 input to).
void rmsnorm_quant_bf16(const float* x, const gpu::bf16* w, gpu::bf16* out, int M, int H, float eps, bool nvfp4,
                        float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s);
// quant_act_fp8 over the BF16 copy (the same bytes as from the FP32 values it was rounded from).
void quant_act_fp8_bf16in(const gpu::bf16* x, int M, int K, float input_scale, uint8_t* q, gpu::Stream s);
// prefill_gemm_fp8 with its FP32 output rounded to BF16 (the in_qkv and in_z projections; never accumulated).
void prefill_gemm_fp8_bf16out(const uint8_t* xq, int M, const uint8_t* w, float alpha, gpu::bf16* y, int ldy, int N,
                              int K, gpu::Stream s);
// narrow_bf16 over the BF16 copy of the norm (the same MMA fragments).
void narrow_bf16_bf16in(const gpu::bf16* x, int M, const gpu::bf16* wa, const gpu::bf16* wb, float* ya,
                        float* yb, int N, int K, gpu::Stream s);
// gated_rmsnorm_fp8 over the recurrence's BF16 output and the BF16 z projection.
void gated_rmsnorm_fp8_bf16(const gpu::bf16* x, const gpu::bf16* z, const gpu::bf16* w, int rows, int D,
                            float eps, float input_scale, uint8_t* q, gpu::Stream s);

// ---- The batched verify's state kernels (M4 stage 4): one launch over several sequences, each row computed
// exactly as the per-sequence launch computes it (ragon-stream-tests compares them bit for bit). ----
// attn_prepare over every row of b, each into its sequence's caches at its sequence's position.
void attn_prepare_seqs(const float* q_gate, const float* k, const float* v, const gpu::bf16* q_norm,
                       const gpu::bf16* k_norm, const SeqBatch& b, int Hq, int Hkv, int D, int rot, float theta,
                       float eps, float* q, float* gate, SeqPtrs<gpu::bf16> kc, SeqPtrs<gpu::bf16> vc,
                       gpu::Stream s, SeqPtrs<const uint64_t> pages = {}, uint8_t* sealed = nullptr,
                       unsigned long long unit = 0);
// attention_rows over every row of b (q and out [rows][Hq][D]), each sequence over its own caches.
void attention_rows_seqs(const float* q, SeqPtrs<const gpu::bf16> kc, SeqPtrs<const gpu::bf16> vc,
                         const SeqBatch& b, int Hq, int Hkv, int D, float* scratch, float* out, gpu::Stream s,
                         int fixed_ctx = 0, SeqPtrs<const uint64_t> pages = {}, KvQ8 q8 = {});
// The cascade (BATCHING §5.3): attention_rows_seqs with the keys below a group's cut computed once for every row of the
// group, whose sequences (a run of the batch) hold those keys on the same pages; cut on the tile grid, at or below each
// sequence's position. Every sequence's own keys follow from its group's cut, so each row's result is bit for bit its own
// launch's. Eager only: a captured graph's launches cannot follow which pages are shared.
void attention_rows_cascade(const float* q, SeqPtrs<const gpu::bf16> kc, SeqPtrs<const gpu::bf16> vc,
                            const SeqBatch& b, const SeqGroups& groups, int Hq, int Hkv, int D, float* scratch, float* out,
                            gpu::Stream s, SeqPtrs<const uint64_t> pages, KvQ8 q8 = {});
// The verify's conv: each sequence's rows of mixed from its own window (read, not written).
void gdn_conv_seqs(float* mixed, SeqPtrs<const float> state, const gpu::bf16* w, const SeqBatch& b, int C,
                   gpu::Stream s);
// The verify's recurrence: each sequence's rows from its own state (read, not written).
void gdn_recurrent_seqs(const float* mixed, const float* g, const float* beta, SeqPtrs<const float> state, float* out,
                        const SeqBatch& b, int H, int HV, gpu::Stream s);
// The verify's recurrence with the previous step's commit folded in (RAGON_MK_GDN_FUSED): sequence i first replays
// its pending rows, pend[i] .. pend[i] + pend[kMaxSeqs + i] - 1 of pmixed/pg/pbeta (outputs to the scratch pout), and
// writes the state they leave; then its verify rows from there, as gdn_recurrent_seqs. Bits are the commit's and the
// verify's: the same per-row arithmetic in the same order. pend lives on the device (a captured graph reads it).
void gdn_recurrent_fused_seqs(const float* mixed, const float* g, const float* beta, SeqPtrs<float> state, float* out,
                              const SeqBatch& b, int H, int HV, const float* pmixed, const float* pg, const float* pbeta,
                              const int* pend, float* pout, gpu::Stream s);
// A joint commit (BATCHING P4d, Engine::step_batch): gdn_conv_seqs and gdn_recurrent_seqs that also write each
// sequence's state, as gdn_conv and gdn_recurrent do for one. `mixed` of the conv is overwritten; `out` is scratch.
void gdn_conv_commit_seqs(float* mixed, SeqPtrs<float> state, const gpu::bf16* w, const SeqBatch& b, int C,
                          gpu::Stream s);
void gdn_recurrent_commit_seqs(const float* mixed, const float* g, const float* beta, SeqPtrs<float> state, float* out,
                               const SeqBatch& b, int H, int HV, gpu::Stream s);

// RMSNorm (as rmsnorm) that also writes the FP16 copy and scales to_half_rows would make from its output.
void rmsnorm_half(const float* x, const gpu::bf16* w, float* out, gpu::half* xh, float* xinv, int rows, int H, float eps,
                  bool gemma, gpu::Stream s);
// x [M][K] FP32 -> FP16 with a power-of-two scale per row (xinv[m] undoes it).
void to_half_rows(const float* x, int M, int K, gpu::half* out, float* xinv, gpu::Stream s);
// y[M][N] = scale2 * x . W^T, W NVFP4 in the tiled layout (wscale is unused: the scales are in the blob).
// wscale: the scale-code table of a coded blob (B26), nullptr when the blob carries its raw scales.
void stream_gemm_nvfp4(const gpu::half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
                       float scale2, float* y, int N, int K, gpu::Stream s);
// y[M][N] = wscale * x . W^T, W FP8 E4M3 in the tiled layout.
void stream_gemm_fp8(const gpu::half* x, const float* xinv, int M, const uint8_t* w, float wscale, float* y, int N,
                     int K, gpu::Stream s);

// Several matrices that share the input x, in one launch (fewer launch ramps and tails): each target
// gets y = scale * x . W^T. Row-invariant like the single-matrix calls, and bit-identical to them.
struct StreamTarget {
  const uint8_t* w;
  const uint8_t* wscale;  // NVFP4: the scale-code table of a coded blob (B26); nullptr for a raw blob and for FP8
  float scale;            // NVFP4 global scale, or the FP8 per-tensor scale
  float* y;
  int N;
};
void stream_gemm_multi(bool nvfp4, const gpu::half* x, const float* xinv, int M, const StreamTarget* t, int n, int K,
                       gpu::Stream s);
// The drafter's sparse head (BACKLOG 19): y[M][cap] = scale2 * x . W[rows[i]]^T for i < *nrows, -inf after, W the
// tiled NVFP4 blob the full head streams (raw, or coded with raw_scales its raw scales in block order: Fp4Weight's
// w and scales). Each element's arithmetic is the tiled stream_gemm's, so a selected row's logit is bit for bit the
// full head's. cap % 32 == 0.
void stream_gemm_gather_nvfp4(const gpu::half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* raw_scales,
                              float scale2, const int* rows, const int* nrows, int cap, float* y, int K, gpu::Stream s);
// The sparse head's selection (sparse_head.cu): approx[n] = max over the M rows of p[m] . us[n] (p [M][R] FP32, us
// [N][R] BF16, R = 256); then rows = every row any of the masks marks (the fixed list joined with each session's
// context vocabulary) and the `dynamic` best of the others (ties at the threshold all taken), in id order, *nrows
// their count (at most cap). `scratch` holds sparse_head_scratch_bytes(N).
struct SelMasks {
  const uint8_t* m[kMaxSeqs];
  int n;
};
size_t sparse_head_scratch_bytes(int N);
void sparse_head_select(const float* p, int M, const gpu::bf16* us, int N, int R, SelMasks masks, int dynamic,
                        int cap, void* scratch, int* rows, int* nrows, gpu::Stream s);
// ids[i] = rows[ids[i]] for the M x K top-K indices (the sparse head's columns back to token ids).
void sparse_head_ids(const int* rows, int* ids, int n, gpu::Stream s);
// mask[ids[i]] = 1 for i < n (a session's context vocabulary, BACKLOG 19).
void sparse_head_mark(uint8_t* mask, const int* ids, int n, gpu::Stream s);
// The drafter's head over n sequences' E drafted rows each (x [n * E][K] FP32), each sequence exactly as it is drafted
// alone (BACKLOG 19): sequence i's rows get the sparse head over masks[i] (its session's fixed list and context) or,
// with masks[i] null, the full head; vals and ids [n * E][K16] the top K16 logits and token ids. The full head's
// sequences go through one stream GEMM per up to 32 rows (copied together in xtmp); every sparse sequence has its own
// selection (its rank-256 scores from its own rows, its own mask), gathered GEMM and top K16. Every kernel's result
// for a row depends only on that row and the sequence's selection, so a sequence's values and ids are bit for bit what
// the same call with n = 1 on its rows gives (ragon-stream-tests).
struct SparseHeadBufs {
  const uint8_t* w = nullptr;      // the target head's tiled blob (Fp4Weight::w)
  const uint8_t* table = nullptr;  // a coded blob's code table (Fp4Weight::table), nullptr for a raw one
  const uint8_t* raw = nullptr;    // a coded blob's raw scales (Fp4Weight::scales)
  float scale2 = 0.f;
  int N = 0, K = 0;                // vocab x hidden
  const gpu::bf16 *us = nullptr, *vt = nullptr;
  int R = 256, dynamic = 0, cap = 0;
  float* p = nullptr;              // [32][R]
  void* scratch = nullptr;         // sparse_head_scratch_bytes(N)
  int *rows = nullptr, *nrows = nullptr;  // [N], [1]
  float* xtmp = nullptr;           // [32][K]
  gpu::half* xh = nullptr;            // [32][K]
  float* xinv = nullptr;           // [32]
  float* logits = nullptr;         // [32][N]
  float* topk_scratch = nullptr;   // topk_scratch_floats(32, 64)
};
void draft_head_rows(const SparseHeadBufs& b, const float* x, int n, int E, const uint8_t* const* masks, int K16,
                     float* vals, int* ids, gpu::Stream s);
// The megakernel's stage 1 in the step (RAGON_MK_FFN): the rows path's FFN (gate|up from xh, silu(g) * u into g, its FP16
// copy into xh2 with xinv2, down into down.y) as one persistent kernel, bit for bit the four kernels. `counters` is
// kMkFfnWords zeroed words, owned by this call site (the kernel leaves them zeroed). False: a shape or a stream GEMM
// override (RAGON_SG_*, RAGON_KSPLIT) the kernel does not reproduce; nothing was launched.
constexpr int kMkFfnWords = 5 + kMaxSeqRows;
// The megakernel's stage 4 (RAGON_MK_LAYERS, BACKLOG 28): a persistent kernel's last GEMM adds its output into the
// residual stream h itself (add_inplace's rounding), and, with norm_w, the next norm (rmsnorm_half's row, exactly)
// follows in the same launch into out (FP32), xh and xinv. Passed as a pointer: nullptr keeps the separate kernels.
struct MkTail {
  float* h;
  const gpu::bf16* norm_w = nullptr;  // nullptr: the residual add only
  float* out = nullptr;
  gpu::half* xh = nullptr;
  float* xinv = nullptr;
  float eps = 0.f;
};
// The rows the fused tails are built for (9 to 16: the verify's draft blocks of 12 and 16); other row counts keep the
// separate kernels (the persistent kernels then refuse a tail).
inline bool mk_tail_rows(int M) { return M > 8 && M <= 16; }
// The megakernel's stage 2 in the step (RAGON_MK_GDN): one verify sequence's DeltaNet mixer, from the input norm's output
// (xn in FP32, xh/xinv its FP16 copy) to the out-projection's output (out.y), as one persistent kernel: in_qkv's output
// into in_qkv.y (kept for the commit), in_z's into in_z.y, the a/b projections and the gating into a, b, g, beta, the
// conv's output into mixed (from conv_state, left unwritten), the recurrence's into core (from state, left unwritten),
// the gated norm's into normed and its FP16 copy into xh2/xinv2. Bit for bit the graph path's kernels. `counters` is
// kMkGdnWords zeroed words owned by the call site. False: a shape or override it does not reproduce; nothing launched.
constexpr int kMkGdnWords = 12 + kMaxSeqRows;
struct LoraRows;
struct LoraSiteDev;
struct MkGdnParams {
  const gpu::half* xh;
  const float* xinv;
  const float* xn;
  int M, K;  // rows; the hidden size (the projections' K and the out-projection's N)
  StreamTarget in_qkv, in_z, out;  // FP8, or NVFP4 with fp4 (their wscale the scale tables)
  bool fp4 = false;                // LOWBIT §8 a1: the projections in NVFP4
  const gpu::bf16 *wa, *wb;    // [HV][K] each
  float *a, *b, *g, *beta;
  const gpu::bf16 *A_log, *dt_bias;
  float* mixed;
  const float* conv_state;
  const gpu::bf16* conv_w;
  const float* state;
  float* core;
  const gpu::bf16* norm_w;
  float eps;
  float* normed;
  gpu::half* xh2;
  float* xinv2;
  int C, H, HV, Dk, Dv, Vs;
  unsigned* counters;
  unsigned long long* times = nullptr;  // RAGON_MK_TIMES: [blocks][kMkTimesPerBlock] stamps of the last launch
  const MkTail* tail = nullptr;  // stage 4: the out-projection into tail->h, then tail's norm
  const TreeDesc* td = nullptr;  // a tree verify (BACKLOG 26): the tree conv and the per-path recurrence
  // RAGON_LORA_MK: the Gdn site's delta on the in_qkv and in_z tiles, from lora_t (lora_shrink's of xn, run before),
  // as lora_expand adds it; lora_t == nullptr, none. The rows' slots and the site in slot 0 (slot s's at s * stride).
  const LoraRows* lora_rows = nullptr;
  const LoraSiteDev* lora_site = nullptr;
  int lora_stride = 0;
  const float* lora_t = nullptr;
};
bool gdn_persistent(const MkGdnParams& p, gpu::Stream s);
// The megakernel's stage 4 (RAGON_MK_SPAN): a verify's DeltaNet layer, its mixer (as gdn_persistent with gp.tail) and
// its FFN (as ffn_persistent with tail), in one persistent launch. gp.tail must carry the post-attention norm into its
// own FP16 buffer (not gp.xh or gp.xh2), and xh2/xinv2 (the down projection's input) must not be the mixer's. False:
// not built for these shapes or options (RAGON_MK_SPAN_PFG 2 or 3, RAGON_MK_SPAN_PFF 3 or 4); nothing launched.
bool gdn_ffn_persistent(const MkGdnParams& gp, const StreamTarget& gate, const StreamTarget& up, const StreamTarget& down,
                        int I, float* g, float* u, gpu::half* xh2, float* xinv2, unsigned* counters,
                        unsigned long long* times, const MkTail* tail, gpu::Stream s);
// tail (stage 4): the down projection into tail->h, then tail's norm (down.y unused).
bool ffn_persistent(const gpu::half* xh, const float* xinv, int M, const StreamTarget& gate, const StreamTarget& up,
                    const StreamTarget& down, int H, int I, float* g, float* u, gpu::half* xh2, float* xinv2,
                    unsigned* counters, gpu::Stream s, unsigned long long* times = nullptr,
                    const MkTail* tail = nullptr);
constexpr int kMkTimesPerBlock = 10;  // RAGON_MK_TIMES: globaltimer stamps per block of a persistent kernel
// The megakernel probe (BACKLOG 28): about `bytes` of a tiled weight blob (the first chunks of every tile, what a stream
// GEMM's blocks read first) prefetched into L2 by one PDL launch, so the latency-bound kernels before that GEMM run
// beside its first bytes. Writes nothing; the GEMM computes the same bits.
void l2_prefetch_weights(const uint8_t* w, bool fp4, bool coded, int N, int K, size_t bytes, gpu::Stream s);

// Top-K (K <= 64) of each row of x [rows][V]: values descending, ties to the lower index (so entry 0 is
// what std::max_element returns). `scratch` needs topk_scratch_floats(rows, K) floats.
size_t topk_scratch_floats(int rows, int K);
void topk_rows(const float* x, int rows, int V, int K, float* scratch, float* vals, int* ids, gpu::Stream s);
// The same under a token mask per row (structured output): mask [rows][words] int32, bit id%32 of word id/32 set
// when id is allowed. Disallowed ids count as -inf: they never appear. A row allowing m < K ids ends in K - m empty
// entries (-FLT_MAX, 0x7fffffff).
void topk_rows_masked(const float* x, int rows, int V, int K, const int32_t* mask, int words, float* scratch,
                      float* vals, int* ids, gpu::Stream s);

// Dequantize a whole matrix to BF16 (for the BF16 GEMM path).
void dequant_nvfp4(const uint8_t* w, const uint8_t* wscale, float scale2, gpu::bf16* out, int N, int K,
                   gpu::Stream s);
void dequant_fp8(const uint8_t* w, float wscale, gpu::bf16* out, int N, int K, gpu::Stream s);
void to_bf16(const float* x, gpu::bf16* out, int n, gpu::Stream s);
// y[M][N] = x[M][K] . W[N][K]^T with BF16 inputs and FP32 output, on tensor cores (gemm_bf16.cu; it
// replaced cuBLAS, BACKLOG §19.23). K % 64 == 0, N even, x and W 16-byte aligned; any M. A row's result
// does not depend on M. Graph-capturable.
void gemm_bf16(const gpu::bf16* x, int M, const gpu::bf16* w, float* y, int N, int K, gpu::Stream s);

void embed(const gpu::bf16* table, const int* ids, int M, int H, float* out, gpu::Stream s);
// out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * (gemma ? 1 + w : w), rows of length H.
void rmsnorm(const float* x, const gpu::bf16* w, float* out, int rows, int H, float eps, bool gemma,
             gpu::Stream s);
void add_inplace(float* h, const float* d, int n, gpu::Stream s);
// out = silu(g) * u
void silu_mul(const float* g, const float* u, float* out, int n, gpu::Stream s);
// x *= sigmoid(gate)
void sigmoid_mul(float* x, const float* gate, int n, gpu::Stream s);

// Gated DeltaNet. `mixed` is [M][C] (q | k | v, C = 2*Kd + Vd): causal depthwise conv (kernel 4) with
// state [C][3], then SiLU, in place.
void gdn_conv(float* mixed, float* conv_state, const gpu::bf16* w, int M, int C, gpu::Stream s);
// g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b); a, b, g, beta are [M][HV].
void gdn_gating(const float* a, const float* b, const gpu::bf16* A_log, const gpu::bf16* dt_bias,
                float* g, float* beta, int M, int HV, gpu::Stream s);
// The delta rule over M tokens with q/k L2 normalization: state [HV][Dk][Dv] (k-major), out [M][HV*Dv].
// Reads the state from state_in and writes the advanced state to state_out (the same pointer to update in
// place; nullptr to leave it unwritten, as a verify does). Each token's arithmetic is the same whatever M
// is, so replaying accepted rows reproduces sequential decoding bit for bit.
void gdn_recurrent(const float* mixed, const float* g, const float* beta, const float* state_in, float* state_out,
                   float* out, int M, int H, int HV, gpu::Stream s);
// The prefill path's conv (prefill_gdn.cu), parallel over tokens: out of place (in: the raw projections,
// out: after the conv and SiLU), and the window updated. Each token is computed the same way whatever M is.
// window_at (optional): also the window after the first `at` tokens (a state kept for a later prompt).
void gdn_conv_prefill(const float* in, float* out, float* conv_state, const gpu::bf16* w, int M, int C,
                      gpu::Stream s, float* window_at = nullptr, int at = -1);
// The prefill path's recurrence: the same math as gdn_recurrent in its chunked form (32-token chunks at absolute
// positions: pos0 is the first token's position) on tensor cores, the state updated in place. Its result for a
// prompt does not depend on how the prompt was split, as long as every split falls on a multiple of 32.
// state_at (optional): also the state after the first `at` tokens, which must end a chunk (pos0 + at % 32 == 0).
void gdn_recurrent_prefill(const float* mixed, const float* g, const float* beta, float* state, float* out, int M, int H,
                           int HV, int pos0, gpu::Stream s, float* state_at = nullptr, int at = -1);
// The two above over BF16 activations (--prefill-bf16; see the block after narrow_bf16): the conv reads the
// in_qkv projection in BF16 and writes its output in BF16 (the state window then holds BF16 values, as
// production's conv state does), the recurrence reads that and writes its output in BF16. The conv and
// recurrent state, g and beta stay FP32; the same chunk invariance holds.
void gdn_conv_prefill_bf16(const gpu::bf16* in, gpu::bf16* out, float* conv_state, const gpu::bf16* w, int M,
                           int C, gpu::Stream s, float* window_at = nullptr, int at = -1);
void gdn_recurrent_prefill_bf16(const gpu::bf16* mixed, const float* g, const float* beta, float* state,
                                gpu::bf16* out, int M, int H, int HV, int pos0, gpu::Stream s,
                                float* state_at = nullptr, int at = -1);
// out = rmsnorm(x) * w * silu(z), rows of length D.
void gated_rmsnorm(const float* x, const float* z, const gpu::bf16* w, float* out, int rows, int D,
                   float eps, gpu::Stream s);

// The KV caches' layout (elements): head h, position p at h * head_stride + p * pos_stride. The default
// is interleaved [pos][head][dim]; the engine sets head-major [head][pos][dim], so one head's keys are
// one contiguous stream for the attention kernels.
void set_kv_layout(size_t head_stride, size_t pos_stride);
// RAGON_KV_CALIB (LOWBIT §8): folds the largest |K| and |V| of positions [p0, p0 + n) of one layer's cache (pages:
// its page table, or nullptr for a flat slot) into amax[0] and amax[1].
void kv_absmax(const gpu::bf16* kcache, const gpu::bf16* vcache, const uint64_t* pages, int p0, int n,
               int Hkv, int D, float* amax, gpu::Stream s);

// Full attention: q_gate [M][Hq*2*D] (per head: q then gate), k/v [M][Hkv*D]. Applies the Gemma-style
// q/k norms and NeoX RoPE on the first `rot` dims, writes q [M][Hq][D] and gate [M][Hq][D], and appends
// k/v to the BF16 caches [pos][Hkv][D] at positions pos0 .. pos0+M-1.
void attn_prepare(const float* q_gate, const float* k, const float* v, const gpu::bf16* q_norm,
                  const gpu::bf16* k_norm, DevPos pos0, int M, int Hq, int Hkv, int D, int rot, float theta,
                  float eps, float* q, float* gate, gpu::bf16* kcache, gpu::bf16* vcache,
                  gpu::Stream s, const TreeDesc* td = nullptr, const int* mrope = nullptr,
                  const uint64_t* pages = nullptr, uint8_t* sealed = nullptr, unsigned long long unit = 0);
// mrope (a prefill chunk of a prompt with images): row m's RoPE positions are mrope[3m .. 3m + 2], (temporal, height,
// width), each frequency taking its component as the interleaved M-RoPE sections say (attention_body.cuh); null: the
// row's position is its slot plus pos0's rope delta for all three.
// Causal attention of M queries (positions pos0..) over the cache, on tensor cores over fixed 4096-key ranges at
// fixed positions (combined afterwards), so each row's result is independent of M: decode, verify and prefill
// (in slices of queries). `scratch` needs attention_rows_scratch_floats(M, Hq, D, ctx) floats.
// fixed_ctx > 0 (a captured graph, pos0 read on the device): launch the ranges covering fixed_ctx keys
// whatever the position; ranges past the last visible key write an empty partial that the combine skips,
// so the result is bit for bit the eager launch's.
size_t attention_rows_scratch_floats(int M, int Hq, int D, int ctx);
void attention_rows(const float* q, const gpu::bf16* kcache, const gpu::bf16* vcache, DevPos pos0, int M,
                    int Hq, int Hkv, int D, float* scratch, float* out, gpu::Stream s, int fixed_ctx = 0,
                    const TreeDesc* td = nullptr, const uint64_t* pages = nullptr, KvQ8 q8 = {});
// P4: kcache/vcache are a layer's K and V bases in the pool and `pages` the session's page table (null: flat).
// Sets the kernels' shared-memory attributes once, outside any graph capture.
void prepare_kernels();
// M3: attention_rows loads its key/value tiles with bulk copies (TMA, one 512-byte copy per key row) instead
// of 16-byte cp.async. Same tiles, same arithmetic: the output is bit for bit the same. Off by default.
void set_attention_bulk(bool on);
// M3: attention_rows prefetches its KV tiles into L2 this many tiles past its shared-memory ring (head-major KV
// only; 0 off). Moves data only: the output is bit for bit the same.
void set_attention_prefetch(int tiles);
// What attention_rows uses (for the megakernel's attention layer, which reproduces it): the KV strides, the L2
// prefetch depth, whether the bulk-copy tiles are on, and the key range size (attention_chunk()).
size_t kv_head_stride(int D);
size_t kv_pos_stride(int Hkv, int D);
int attention_prefetch_tiles();
bool attention_bulk_enabled();
int attention_chunk();
// The megakernel's stage 3 in the step (RAGON_MK_ATTN, BACKLOG 28): one verify sequence's full-attention layer, from the
// input norm's FP16 copy (xh/xinv) to the out-projection's output (o.y), as one persistent kernel: q|k|v into q.y,
// k.y, v.y, the prepare step (q/k norms, RoPE, the rows' K and V into the caches at pos0 ..) into q_out, gate and the
// caches, the attention over the session's real key ranges into scratch, the combine with the sigmoid gate into attn,
// its FP16 copy into xh2/xinv2, then the out-projection. Bit for bit the graph path's kernels (attn_prepare,
// attention_rows, sigmoid_mul, to_half_rows, the stream GEMMs); work is taken from one ordered ticket queue, so it
// completes with any number of resident blocks. `counters` is kMkAttnWords zeroed words owned by the call site (left
// zeroed). False: a shape or override it does not reproduce (bulk tiles, RAGON_SG_*, RAGON_KSPLIT); nothing launched.
constexpr int kMkAttnWords = 32;
constexpr int kMkAttnEvents = 16;  // RAGON_MK_TIMES: per launch, the globaltimer of each counter's completion
struct MkAttnParams {
  const gpu::half* xh;
  const float* xinv;
  int M, K;                     // rows; the hidden size (q|k|v's K, o's N)
  StreamTarget q, k, v, o;      // FP8 (NVFP4 with fp4); q is q|gate per head [Hq][2 D]
  bool fp4 = false;             // LOWBIT §8 a1: the projections in NVFP4
  int Hq, Hkv, D, rot;
  float theta, eps;
  const gpu::bf16 *q_norm, *k_norm;
  float *q_out, *gate;          // [M][Hq][D]
  gpu::bf16 *kcache, *vcache;
  const uint64_t* pages = nullptr;  // P4: the session's KV page table (kcache/vcache the layer's bases), or flat
  DevPos pos0 = DevPos(0);
  float* scratch;               // attention_rows_scratch_floats(M, Hq, D, the context) floats
  float* attn;                  // [M][Hq D]
  gpu::half* xh2;                  // the out-projection's input and row scales
  float* xinv2;
  unsigned* counters;
  unsigned long long* times = nullptr;  // RAGON_MK_TIMES: kMkAttnEvents stamps of the last launch
  const TreeDesc* td = nullptr;  // a tree verify (BACKLOG 26): its rows' depths and the attention's per-variant tail
  const MkTail* tail = nullptr;  // stage 4: the out-projection into tail->h (o.y unused), then tail's norm
};
bool attn_persistent(const MkAttnParams& p, gpu::Stream s);
void attn_persistent_prepare();  // its shared-memory attribute, once, outside any graph capture
// ---- The DFlash2 drafter (draft_kernels.cu). ----
// dst[r][0..H) = bf16(src[r][0..H)), rows dst_stride apart (the target features the drafter conditions on).
void copy_rows_bf16(const float* src, int rows, int H, gpu::bf16* dst, int dst_stride, gpu::Stream s);
// to_half_rows from BF16 rows [M][K] (BACKLOG 33: the captured features into the fc's NVFP4 copy).
void bf16_to_half_rows(const gpu::bf16* x, int M, int K, gpu::half* out, float* xinv, gpu::Stream s);
// Per-head RMSNorm (plain weight) and NeoX RoPE over the full 128-dim head at positions pos0 + row, in place.
void draft_qk_rope(float* x, int rows, int heads, int row_stride, const gpu::bf16* norm, DevPos pos0, float theta,
                   float eps, gpu::Stream s);
// k/v rows [rows][kv_size] into the BF16 rings [ring][kv_size] at positions pos0 ... (position p in row p % ring;
// M4 §1). rows <= ring.
void draft_store_kv(const float* k, const float* v, int rows, int kv_size, DevPos pos0, gpu::bf16* kc,
                    gpu::bf16* vc, int ring, gpu::Stream s);
// The block's B queries (positions L .. L + B - 1) over the cached context [L + j - window_left, L) and the
// whole block (bidirectional within it). q [B][Hq][128], kb/vb [B][Hkv][128]. The context is read from the
// rings kc/vc (position p in row p % ring, ring > window_left).
void draft_attention(const float* q, const gpu::bf16* kc, const gpu::bf16* vc, int ring, const float* kb,
                     const float* vb, DevPos L, int B, int window_left, int Hq, int Hkv, float* out, gpu::Stream s);
// How a warp of draft_attention walks the context (RAGON_DRAFT_ATTN_KEYS): 1, 2, 4 or 8 keys a load, or 0 (the
// default) eight keys with the dot products reduce-scattered across the warp. 1 is the one-key-at-a-time loop; every
// choice gives the same bits. For tests and timings.
void set_draft_attention_keys(int u);
// The drafter over several sequences' blocks at once (M4 stage 4): draft_qk_rope with each row at its sequence's
// position, and draft_attention with each sequence's queries over its own ring and its own block (b's rows).
void draft_qk_rope_seqs(float* x, int heads, int row_stride, const gpu::bf16* norm, const SeqBatch& b, float theta,
                        float eps, gpu::Stream s);
// draft_store_kv with each row at its sequence's position, into that sequence's rings (kc.p[i], vc.p[i]).
void draft_store_kv_seqs(const float* k, const float* v, int kv_size, SeqPtrs<gpu::bf16> kc,
                         SeqPtrs<gpu::bf16> vc, int ring, const SeqBatch& b, gpu::Stream s);
void draft_attention_seqs(const float* q, SeqPtrs<const gpu::bf16> kc, SeqPtrs<const gpu::bf16> vc, int ring,
                          const float* kb, const float* vb, const SeqBatch& b, int window_left, int Hq, int Hkv,
                          float* out, gpu::Stream s);
// DFlash2's two-tap grouped dynamic convolution over a block (side 0 wraps a sublayer's input, 1 its output).
void grouped_conv(const float* h, const float* coef, const gpu::bf16* base, int side, float* out, int rows,
                  int C, int groups, int block, gpu::Stream s);
// The candidate selector's 16 x 16 transition scores for E positions (hp: [E][256], cand/unary: [E][16]).
// anchor_dev (a captured graph): read the anchor from device memory instead of `anchor`.
void selector_lattice(const float* hp, const int* cand, const float* unary, const gpu::bf16* P,
                      const gpu::bf16* S, int anchor, int E, float* scores, gpu::Stream s,
                      const int* anchor_dev = nullptr);

}  // namespace ragon::kernels
