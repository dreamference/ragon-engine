// Kernel launches with programmatic dependent launch (PDL; SPEC.md §16.7, M3). CUDA source files only.
//
// Normally a kernel starts only after the previous kernel in the stream has finished and its writes have
// been flushed: every launch pays the gap between the last block of one kernel and the first block of the
// next, and the next kernel's first weight loads start from an idle bus. With PDL (sm_90 and later; M0 §2
// measured it on this GB10), a kernel launched with the programmatic-serialization attribute may start while
// its predecessor is still running, once every block of the predecessor has executed
// griddepcontrol.launch_dependents (or exited). Its blocks then wait in griddepcontrol.wait until the
// predecessor has completed and its memory is visible.
//
// The rule every kernel launched through launch_kernel() follows, and the only thing that keeps PDL exact:
//   - pdl_begin() (wait, then trigger) is its first statement, before any read of memory a previous kernel
//     writes and before any write at all. A kernel whose prologue only reads immutable data (the weights) may
//     issue those reads before pdl_begin(), and nothing else: stream_gemm_kernel and bf16_rows_kernel do.
//   - Every block reaches pdl_begin() (no early return before it), so a kernel never completes before its
//     predecessor: the chain of waits makes completion transitive, and a kernel three launches later that
//     reads this one's predecessor's output is still ordered after it.
// With PDL off (the default) the attribute is not set, griddepcontrol.wait returns at once and the launch is
// an ordinary one, so the kernels compute exactly the same thing either way.
#pragma once

#include "backend/backend.hpp"  // the CUDA runtime under CUDA, the HIP runtime under HIP

#include <stdexcept>
#include <string>
#include <utility>

#include "core/kernels.cuh"

namespace ragon::kernels {

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Lets the next kernel in the stream launch as soon as every block of this one has started (its blocks
// then wait in pdl_wait() for this kernel to finish).
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The first statement of every kernel launched through launch_kernel().
__device__ __forceinline__ void pdl_begin() {
  pdl_wait();
  pdl_trigger();
}

// The megakernel probe (BACKLOG 28; a measurement, never for serving): RAGON_PROBE_SKIP is a comma-separated list of
// launch names (the `what` below; "*gemm" style prefixes end in '*') whose launches are replaced by
// probe_noop_kernel, one block that waits for its predecessor and triggers its successor. The output is then wrong;
// the step's time without those kernels is the bound on what hiding them beside the weight stream could take.
bool probe_skip(const char* what);
__global__ void probe_noop_kernel();

#if !defined(RAGON_BACKEND_HIP)
// kernel<<<grid, block, smem, s>>>(args...), with the PDL attribute when `pdl`.
template <typename... Params, typename... Args>
void launch_kernel_pdl(bool pdl, const char* what, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem,
                       cudaStream_t s, Args&&... args) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  if (probe_skip(what)) {  // timing probe only (RAGON_PROBE_SKIP): the launch becomes one block that waits and triggers
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(32);
    cfg.dynamicSmemBytes = 0;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, probe_noop_kernel);
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + " (probe): " + cudaGetErrorString(e));
    return;
  }
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// The same with the attribute when set_pdl(true).
template <typename... Params, typename... Args>
void launch_kernel(const char* what, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t s,
                   Args&&... args) {
  launch_kernel_pdl(pdl_enabled(), what, kernel, grid, block, smem, s, std::forward<Args>(args)...);
}
#else
// HIP (BACKEND §7): no programmatic dependent launch (gpu::kDependentLaunch is false) and no RAGON_PROBE_SKIP probe,
// so every launch is a plain one; pdl_begin() is then a no-op and the kernel computes exactly what it computes
// under PDL. `pdl` is accepted and ignored.
//
// RAGON_LAUNCH_LOG=1 (a measurement, never for serving; launch_log_hip.cu): every launch outside a graph capture is
// timed alone between two events and waited for, and at exit the slowest launch of each name is printed, so a run
// shows how far each kernel stays under the Windows display-timeout (TDR) limit.
bool launch_log_enabled();
void* launch_log_begin(gpu::Stream s);           // nullptr when the launch is not timed
void launch_log_end(void* h, const char* what, gpu::Stream s);
bool launch_log_wants(const char* what);  // RAGON_LAUNCH_LOG_ONLY (launch_log_hip.cu)
// RAGON_LAUNCH_DUP=a,b (a measurement, never for serving; launch_log_hip.cu): launches named exactly a or b run twice
// back to back, so a step's time grows by their true in-situ cost with no event in the stream. Only for kernels whose
// second run writes what the first wrote (stream_gemm, stream_gemm_gather, to_half_rows); a run's ids show it.
bool launch_dup_wants(const char* what);
// RAGON_LAUNCH_SKIP=a,b (a measurement that breaks the results; never for serving): launches named exactly a or b do
// not run, so a step's time shrinks by their in-situ cost (for in-place kernels, which RAGON_LAUNCH_DUP cannot time).
bool launch_skip_wants(const char* what);
// RAGON_LAUNCH_HOLLOW=1|2 (a measurement that breaks the results; never for serving; launch_log_hip.cu): inside the
// "draft" range every launch not named in RAGON_LAUNCH_HOLLOW_KEEP=a,b is replaced by an empty one-block kernel (=1,
// the same launch count) or dropped (=2), so the draft falls to the floor of its host issue, copies, the runtime's
// batching and the sync, plus the kept kernels.
bool launch_hollow_wants(const char* what);
void launch_hollow(gpu::Stream s);
// RAGON_LAUNCH_LOG=2: not waited for, keyed by grid size too (launch_log_hip.cu); otherwise launch_log_end.
void launch_log_end_grid(void* h, const char* what, gpu::Stream s, unsigned gx, unsigned gy);
// RAGON_HIP_KICK_EVERY=N (an A/B switch, 0 = off; launch_log_hip.cu): every N-th launch outside a capture queries
// its stream, so the runtime sends the batch it holds to the GPU at once (TheRock #8786: work reaches the GPU only
// at a sync or a query).
void launch_kick(gpu::Stream s);
template <typename... Params, typename... Args>
void launch_kernel_pdl(bool /*pdl*/, const char* what, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem,
                       gpu::Stream s, Args&&... args) {
  if (launch_skip_wants(what)) return;
  if (launch_hollow_wants(what)) return launch_hollow(s);
  void* h = launch_log_enabled() && launch_log_wants(what) ? launch_log_begin(s) : nullptr;
  kernel<<<grid, block, smem, s>>>(std::forward<Args>(args)...);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  if (h) launch_log_end_grid(h, what, s, grid.x, grid.y);
  if (launch_dup_wants(what)) kernel<<<grid, block, smem, s>>>(args...);
  launch_kick(s);
}

template <typename... Params, typename... Args>
void launch_kernel(const char* what, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, gpu::Stream s,
                   Args&&... args) {
  launch_kernel_pdl(false, what, kernel, grid, block, smem, s, std::forward<Args>(args)...);
}
#endif

// The attention kernels are left out of PDL unless RAGON_PDL_ATTN=1: llama.cpp found a race on DGX Spark in
// its flash-attention kernels launched with PDL (ggml-org/llama.cpp#23825, attributed to an internal bug, fixed
// there by launching those kernels without it, ~0.2% slower). Until a stress run here shows ours are safe,
// they launch as ordinary kernels; they still begin with pdl_begin(), which is then a no-op.
bool pdl_attention_enabled();

}  // namespace ragon::kernels
