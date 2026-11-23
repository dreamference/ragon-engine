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
bool launch_hollow_io(unsigned bit);
void stream_kick(gpu::Stream s);
}  // namespace ragon::kernels
namespace ragon::gpu {
inline void range_push(const char* name) { kernels::launch_log_range_push(name); }
inline void range_pop() { kernels::launch_log_range_pop(); }
// RAGON_LAUNCH_HOLLOW_IO=bits (a measurement that breaks the results, with RAGON_LAUNCH_HOLLOW; launch_log_hip.cu):
// inside a hollow draft range the transfers named by the bits are skipped too: 1 the draft ids H2D, 2 the
// candidates and scores D2H.
inline bool hollow_io(unsigned bit) { return kernels::launch_hollow_io(bit); }
// One stream query outside a capture, so the runtime sends the batch it holds to the GPU now (TheRock #8786).
inline void kick(Stream s) { kernels::stream_kick(s); }
}  // namespace ragon::gpu
#else
#include <nvtx3/nvToolsExt.h>
namespace ragon::gpu {
inline void range_push(const char* name) { nvtxRangePushA(name); }
inline void range_pop() { nvtxRangePop(); }
inline bool hollow_io(unsigned) { return false; }
inline void kick(Stream) {}  // CUDA submits each launch at once
}  // namespace ragon::gpu
#endif
