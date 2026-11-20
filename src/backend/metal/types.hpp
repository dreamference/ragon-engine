// The Metal backend's neutral types (specs/DREAMFERENCE_RAGON_ENGINE_BACKEND.md §8). Host code is Objective-C++
// (.mm); the kernels are MSL in kernels.metal, compiled when the device opens (the Mac build needs no Xcode).
#pragma once

#include <cstdint>

namespace ragon::metal {
struct Queue;  // an engine-owned MTLCommandQueue that collects encoders into one command buffer (device.hpp)
}

namespace ragon::gpu {

using Stream = ragon::metal::Queue*;  // work is encoded in order and runs when the queue commits its buffer
using bf16 = uint16_t;                // bfloat16 storage on the host; the kernels read it as MSL bfloat
using half = uint16_t;                // IEEE binary16 storage on the host; MSL half on the device

inline constexpr const char* kName = "metal";
inline constexpr bool kDependentLaunch = false;  // no programmatic dependent launch on Metal
inline constexpr bool kGraphs = false;           // indirect command buffers come later (BACKEND §8)
inline constexpr bool kPersistentKernels = false;
inline constexpr bool kGreenContexts = false;
inline constexpr bool kBulkCopy = false;

}  // namespace ragon::gpu
