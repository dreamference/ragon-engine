// The engine's compute backend (specs/DREAMFERENCE_RAGON_ENGINE_BACKEND.md). One interface, chosen at build time:
// the kernel entry points (core/kernels.cuh, core/lora_kernels.cuh, core/vision.hpp's vision_kernels) are declared
// once, in the neutral types below, and exactly one backend defines them. CMake's RAGON_BACKEND picks it (cuda, the
// default; metal, the Mac node, in its first lane; hip is the next row). No runtime choice: one binary serves one vendor.
#pragma once

#if defined(RAGON_BACKEND_HIP)
#error "RAGON_BACKEND=hip: the HIP backend is not built yet (BACKEND §7)"
#elif defined(RAGON_BACKEND_METAL)
#include "backend/metal/types.hpp"
#else
#include "backend/cuda/types.hpp"
#endif
