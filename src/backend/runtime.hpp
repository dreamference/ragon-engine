// The runtime half of the backend interface (BACKEND §5.2, tier 2): streams, events, graphs, allocation and copies
// for the host files (engine.cpp, speculate.cpp, engine_fanout.cpp, engine_lora.cpp, drafter.cpp, image_input.cpp).
// Each backend defines the same ragon::gpu names; the CUDA ones are one-line inline forwards to the calls the host
// files made before, so the CUDA objects keep the same code (§6). Kernel launches stay in the backend entry points.
#pragma once

#include "backend/backend.hpp"

#if defined(RAGON_BACKEND_HIP)
#include "backend/hip/runtime.hpp"
#else
#include "backend/cuda/runtime.hpp"
#endif
