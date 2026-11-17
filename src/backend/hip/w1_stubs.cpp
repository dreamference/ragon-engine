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

void draft_head_rows(const SparseHeadBufs& b, const float* x, int n, int E, const uint8_t* const* masks, int K16,
    float* vals, int* ids, gpu::Stream s) { not_ported("draft_head_rows"); }
void lora_expand(const float* t, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
    const LoraOuts& outs, gpu::Stream stream) { not_ported("lora_expand"); }
void lora_gemm_parts(const LoraSiteDev& d, const float* t,
    LoraGemm (&parts)[kLoraOuts]) { not_ported("lora_gemm_parts"); }
void lora_shrink(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
    int a_rows, float* t, gpu::Stream stream) { not_ported("lora_shrink"); }
void lora_shrink_prefill(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site,
    int slot_stride, float* t, gpu::Stream stream) { not_ported("lora_shrink_prefill"); }

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
