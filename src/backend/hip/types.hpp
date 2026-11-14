// The HIP backend's half of the interface types (BACKEND §3, §7): AMD GPUs through HIP (the Windows node, gfx1151,
// TheRock 10.0.0). The kernels' .cu files compile as HIP from the same source as the CUDA build; what the two
// spell differently goes through the names below (gpu::Stream, gpu::kWarpMask), never through #ifdefs in a kernel.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace ragon::gpu {

using Stream = hipStream_t;     // an ordered queue of device work; every entry point enqueues and returns
using bf16 = __hip_bfloat16;    // bfloat16 storage (weights, KV caches, BF16 activations)
using half = __half;            // IEEE binary16 storage (the stream GEMMs' activation copy)

// The all-lanes mask of the warp-synchronous intrinsics (__ballot_sync, __shfl_sync). HIP's take a 64-bit mask and
// refuse a narrower one at compile time; on a wave32 target (gfx11, the only one built) lanes 32-63 are ignored.
// The kernels assume 32-lane warps; the HIP build targets wave32 only (BACKEND §7).
inline constexpr unsigned long long kWarpMask = 0xffffffffull;

// What the backend offers to the engine's optional paths (BACKEND §4).
inline constexpr const char* kName = "hip";
inline constexpr bool kDependentLaunch = false;    // no programmatic dependent launch: plain launches (launch.cuh)
inline constexpr bool kGraphs = true;              // hipGraph capture of a step
inline constexpr bool kPersistentKernels = false;  // no co-residency guarantee (cooperativeLaunch 0 on Windows, AMD §5)
inline constexpr bool kGreenContexts = false;      // no SM partitions: one context, priority streams
inline constexpr bool kBulkCopy = false;           // no TMA: attention's plain loads (bit-identical)

}  // namespace ragon::gpu

// Device code compiled by hipcc/clang (both its host and its device pass, as __CUDACC__ for nvcc) sees the portable
// device helpers of the interface headers. Code that is CUDA-only (inline PTX) keeps testing __CUDACC__ itself.
#if defined(__HIPCC__)
#define RAGON_GPU_DEVICE_CODE 1
#endif
