// Named profiler ranges for the host files (BACKEND §5.2): NVTX under CUDA (Nsight Systems shows them). Under HIP no
// profiler hook is wired on the Windows node yet; with RAGON_LAUNCH_LOG=1 the innermost range names the phase of each
// timed launch instead ("draft/stream_gemm", launch_log_hip.cu), a no-op otherwise. Ranges never change what the
// engine computes.
#pragma once

#include "backend/backend.hpp"

#if defined(RAGON_BACKEND_HIP)
namespace ragon::kernels {
void launch_log_range_push(const char* name);  // launch_log_hip.cu: a no-op unless RAGON_LAUNCH_LOG=1
void launch_log_range_pop();
}  // namespace ragon::kernels
namespace ragon::gpu {
inline void range_push(const char* name) { kernels::launch_log_range_push(name); }
inline void range_pop() { kernels::launch_log_range_pop(); }
}  // namespace ragon::gpu
#else
#include <nvtx3/nvToolsExt.h>
namespace ragon::gpu {
inline void range_push(const char* name) { nvtxRangePushA(name); }
inline void range_pop() { nvtxRangePop(); }
}  // namespace ragon::gpu
#endif
