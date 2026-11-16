// W1 placeholder definitions (HIP builds only): the kernel entry points the host side links against whose CUDA files
// are not ported yet. Each one stops the process with "not ported: <entry>" on stderr, so a run shows how far the
// port carries it; the rows that port the files delete their entries here, and the last row deletes this file.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/kernels.cuh"
#include "core/lora_kernels.cuh"
#include "core/vision.hpp"

namespace {

[[noreturn]] void not_ported(const char* entry) {
  std::fprintf(stderr, "not ported: %s\n", entry);
  std::fflush(stderr);
#ifdef _WIN32
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);  // no CRT dialog, no Windows Error Reporting
#endif
  std::abort();
}

}  // namespace

namespace ragon::kernels {

int attention_chunk() { not_ported("attention_chunk"); }
void attention_rows(const float* q, const gpu::bf16* kcache, const gpu::bf16* vcache, DevPos pos0, int M, int Hq,
    int Hkv, int D, float* scratch, float* out, gpu::Stream s, int fixed_ctx, const TreeDesc* td,
    const uint64_t* pages, KvQ8 q8) { not_ported("attention_rows"); }
void attention_rows_cascade(const float* q, SeqPtrs<const gpu::bf16> kc, SeqPtrs<const gpu::bf16> vc,
    const SeqBatch& b, const SeqGroups& groups, int Hq, int Hkv, int D, float* scratch, float* out, gpu::Stream s,
    SeqPtrs<const uint64_t> pages, KvQ8 q8) { not_ported("attention_rows_cascade"); }
size_t attention_rows_scratch_floats(int M, int Hq, int D, int ctx) { not_ported("attention_rows_scratch_floats"); }
void attention_rows_seqs(const float* q, SeqPtrs<const gpu::bf16> kc, SeqPtrs<const gpu::bf16> vc, const SeqBatch& b,
    int Hq, int Hkv, int D, float* scratch, float* out, gpu::Stream s, int fixed_ctx, SeqPtrs<const uint64_t> pages,
    KvQ8 q8) { not_ported("attention_rows_seqs"); }
bool attn_persistent(const MkAttnParams& p, gpu::Stream s) { not_ported("attn_persistent"); }
void attn_persistent_prepare() { not_ported("attn_persistent_prepare"); }
void draft_head_rows(const SparseHeadBufs& b, const float* x, int n, int E, const uint8_t* const* masks, int K16,
    float* vals, int* ids, gpu::Stream s) { not_ported("draft_head_rows"); }
bool ffn_persistent(const gpu::half* xh, const float* xinv, int M, const StreamTarget& gate, const StreamTarget& up,
    const StreamTarget& down, int H, int I, float* g, float* u, gpu::half* xh2, float* xinv2, unsigned* counters,
    gpu::Stream s, unsigned long long* times, const MkTail* tail) { not_ported("ffn_persistent"); }
void gated_rmsnorm_fp8(const float* x, const float* z, const gpu::bf16* w, int rows, int D, float eps,
    float input_scale, uint8_t* q, gpu::Stream s) { not_ported("gated_rmsnorm_fp8"); }
void gated_rmsnorm_fp8_bf16(const gpu::bf16* x, const gpu::bf16* z, const gpu::bf16* w, int rows, int D, float eps,
    float input_scale, uint8_t* q, gpu::Stream s) { not_ported("gated_rmsnorm_fp8_bf16"); }
void gdn_conv_prefill(const float* in, float* out, float* conv_state, const gpu::bf16* w, int M, int C,
    gpu::Stream s, float* window_at, int at) { not_ported("gdn_conv_prefill"); }
void gdn_conv_prefill_bf16(const gpu::bf16* in, gpu::bf16* out, float* conv_state, const gpu::bf16* w, int M, int C,
    gpu::Stream s, float* window_at, int at) { not_ported("gdn_conv_prefill_bf16"); }
bool gdn_ffn_persistent(const MkGdnParams& gp, const StreamTarget& gate, const StreamTarget& up,
    const StreamTarget& down, int I, float* g, float* u, gpu::half* xh2, float* xinv2, unsigned* counters,
    unsigned long long* times, const MkTail* tail, gpu::Stream s) { not_ported("gdn_ffn_persistent"); }
bool gdn_persistent(const MkGdnParams& p, gpu::Stream s) { not_ported("gdn_persistent"); }
void gdn_recurrent_prefill(const float* mixed, const float* g, const float* beta, float* state, float* out, int M,
    int H, int HV, int pos0, gpu::Stream s, float* state_at, int at) { not_ported("gdn_recurrent_prefill"); }
void gdn_recurrent_prefill_bf16(const gpu::bf16* mixed, const float* g, const float* beta, float* state,
    gpu::bf16* out, int M, int H, int HV, int pos0, gpu::Stream s, float* state_at,
    int at) { not_ported("gdn_recurrent_prefill_bf16"); }
void gemm_bf16(const gpu::bf16* x, int M, const gpu::bf16* w, float* y, int N, int K,
    gpu::Stream s) { not_ported("gemm_bf16"); }
void kv_absmax(const gpu::bf16* kcache, const gpu::bf16* vcache, const uint64_t* pages, int p0, int n, int Hkv,
    int D, float* amax, gpu::Stream s) { not_ported("kv_absmax"); }
void kv_seal(const gpu::bf16* pool, uint8_t* mirror, uint8_t* sealed, const int* ids, int n, int l0, int l1, int Hkv,
    int D, size_t page_unit, const float* kinv, const float* vinv, gpu::Stream s) { not_ported("kv_seal"); }
void kv_unseal(uint8_t* sealed, const int* ids, int n, gpu::Stream s) { not_ported("kv_unseal"); }
void l2_prefetch_weights(const uint8_t* w, bool fp4, bool coded, int N, int K, size_t bytes,
    gpu::Stream s) { not_ported("l2_prefetch_weights"); }
void lora_expand(const float* t, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
    const LoraOuts& outs, gpu::Stream stream) { not_ported("lora_expand"); }
void lora_gemm_parts(const LoraSiteDev& d, const float* t,
    LoraGemm (&parts)[kLoraOuts]) { not_ported("lora_gemm_parts"); }
void lora_shrink(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
    int a_rows, float* t, gpu::Stream stream) { not_ported("lora_shrink"); }
void lora_shrink_prefill(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site,
    int slot_stride, float* t, gpu::Stream stream) { not_ported("lora_shrink_prefill"); }
void narrow_bf16(const float* x, int M, const gpu::bf16* wa, const gpu::bf16* wb, float* ya, float* yb, int N, int K,
    gpu::Stream s) { not_ported("narrow_bf16"); }
void narrow_bf16_bf16in(const gpu::bf16* x, int M, const gpu::bf16* wa, const gpu::bf16* wb, float* ya, float* yb,
    int N, int K, gpu::Stream s) { not_ported("narrow_bf16_bf16in"); }
void prefill_gemm_fp8(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
    bool accumulate, gpu::Stream s) { not_ported("prefill_gemm_fp8"); }
void prefill_gemm_fp8_bf16out(const uint8_t* xq, int M, const uint8_t* w, float alpha, gpu::bf16* y, int ldy, int N,
    int K, gpu::Stream s) { not_ported("prefill_gemm_fp8_bf16out"); }
void prefill_gemm_fp8_lora(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
    bool accumulate, const LoraGemm& lora, gpu::Stream s) { not_ported("prefill_gemm_fp8_lora"); }
void prefill_gemm_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, float alpha, float* y,
    int ldy, int N, int K, bool accumulate, gpu::Stream s, const uint8_t* wsf) { not_ported("prefill_gemm_nvfp4"); }
void prefill_swiglu_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* gate, float alpha_gate,
    const uint8_t* up, float alpha_up, int N, int K, float out_input_scale, uint8_t* q_out, uint8_t* sf_out,
    gpu::Stream s, const uint8_t* gate_sf, const uint8_t* up_sf) { not_ported("prefill_swiglu_nvfp4"); }
void prepare_kernels() { not_ported("prepare_kernels"); }
void quant_act_fp8(const float* x, int M, int K, float input_scale, uint8_t* q,
    gpu::Stream s) { not_ported("quant_act_fp8"); }
void quant_act_fp8_bf16in(const gpu::bf16* x, int M, int K, float input_scale, uint8_t* q,
    gpu::Stream s) { not_ported("quant_act_fp8_bf16in"); }
void quant_act_nvfp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf,
    gpu::Stream s) { not_ported("quant_act_nvfp4"); }
void retile_fp8(const uint8_t* w, uint8_t* out, int N, int K, gpu::Stream s) { not_ported("retile_fp8"); }
void retile_nvfp4(const uint8_t* w, const uint8_t* wscale, uint8_t* out, int N, int K,
    gpu::Stream s) { not_ported("retile_nvfp4"); }
void retile_nvfp4_coded(const uint8_t* w, const uint8_t* wscale, const uint8_t* code_of_dev, uint8_t* out,
    uint8_t* raw_scales, int N, int K, gpu::Stream s) { not_ported("retile_nvfp4_coded"); }
void rmsnorm_half(const float* x, const gpu::bf16* w, float* out, gpu::half* xh, float* xinv, int rows, int H,
    float eps, bool gemma, gpu::Stream s) { not_ported("rmsnorm_half"); }
void rmsnorm_quant(const float* x, const gpu::bf16* w, float* out, int M, int H, float eps, bool nvfp4,
    float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s) { not_ported("rmsnorm_quant"); }
void rmsnorm_quant_bf16(const float* x, const gpu::bf16* w, gpu::bf16* out, int M, int H, float eps, bool nvfp4,
    float input_scale, uint8_t* q, uint8_t* sf, gpu::Stream s) { not_ported("rmsnorm_quant_bf16"); }
bool scale_code_table(const uint8_t* scales, size_t n, uint8_t (&table)[kScaleCodes],
    uint8_t (&code_of)[256]) { not_ported("scale_code_table"); }
void sigmoid_mul_fp8(const float* x, const float* gate, size_t n, float input_scale, uint8_t* q,
    gpu::Stream s) { not_ported("sigmoid_mul_fp8"); }
void silu_mul_quant_nvfp4(const float* g, const float* u, int M, int K, float input_scale, uint8_t* q, uint8_t* sf,
    gpu::Stream s) { not_ported("silu_mul_quant_nvfp4"); }
void stream_gemm_fp8(const gpu::half* x, const float* xinv, int M, const uint8_t* w, float wscale, float* y, int N,
    int K, gpu::Stream s) { not_ported("stream_gemm_fp8"); }
void stream_gemm_multi(bool nvfp4, const gpu::half* x, const float* xinv, int M, const StreamTarget* t, int n, int K,
    gpu::Stream s) { not_ported("stream_gemm_multi"); }
void stream_gemm_nvfp4(const gpu::half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
    float scale2, float* y, int N, int K, gpu::Stream s) { not_ported("stream_gemm_nvfp4"); }
size_t tiled_bytes_fp8(int N, int K) { not_ported("tiled_bytes_fp8"); }
size_t tiled_bytes_nvfp4(int N, int K) { not_ported("tiled_bytes_nvfp4"); }
size_t tiled_bytes_nvfp4_coded(int N, int K) { not_ported("tiled_bytes_nvfp4_coded"); }
void to_half_rows(const float* x, int M, int K, gpu::half* out, float* xinv,
    gpu::Stream s) { not_ported("to_half_rows"); }

}  // namespace ragon::kernels

namespace ragon::vision_kernels {

void embed_rows(const gpu::bf16* src, int rows, int hidden, float* h, gpu::Stream s) { not_ported("embed_rows"); }

}  // namespace ragon::vision_kernels

namespace ragon {

// The vision tower (core/vision.cu); never constructed while its constructor is a stub.
VisionTower::VisionTower(const std::string&, int) { not_ported("VisionTower::VisionTower"); }
VisionTower::~VisionTower() { not_ported("VisionTower::~VisionTower"); }
void VisionTower::forward(const uint16_t*, int, int, gpu::bf16*, gpu::Stream, int, std::vector<uint16_t>*) {
  not_ported("VisionTower::forward");
}

}  // namespace ragon
