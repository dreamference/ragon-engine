// Named profiler ranges for the host files (BACKEND §5.2): NVTX under CUDA (Nsight Systems shows them), nothing under
// HIP (no profiler hook is wired on the Windows node yet). Ranges never change what the engine computes.
#pragma once

#include "backend/backend.hpp"

#if defined(RAGON_BACKEND_HIP)
namespace ragon::gpu {
inline void range_push(const char*) {}
inline void range_pop() {}
}  // namespace ragon::gpu
#else
#include <nvtx3/nvToolsExt.h>
namespace ragon::gpu {
inline void range_push(const char* name) { nvtxRangePushA(name); }
inline void range_pop() { nvtxRangePop(); }
}  // namespace ragon::gpu
#endif
