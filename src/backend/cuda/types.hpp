// The CUDA backend's half of the interface types (BACKEND §3). Each alias names the very type the kernels were written
// against, so a declaration spelled with them mangles, links and compiles exactly as the CUDA spelling did.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ragon::gpu {

using Stream = cudaStream_t;  // an ordered queue of device work; every entry point enqueues and returns
using bf16 = __nv_bfloat16;   // bfloat16 storage (weights, KV caches, BF16 activations)
using half = __half;          // IEEE binary16 storage (the stream GEMMs' activation copy)

// What the backend offers to the engine's optional paths (BACKEND §4). A backend without one keeps the engine on
// the path that does not need it; the outputs do not change, only the speed.
inline constexpr const char* kName = "cuda";
inline constexpr bool kDependentLaunch = true;   // programmatic dependent launch (set_pdl, --pdl)
inline constexpr bool kGraphs = true;            // captured step graphs (--graphs, DevPos::dev)
inline constexpr bool kPersistentKernels = true; // the megakernels (gdn_persistent, ffn_persistent, attn_persistent)
inline constexpr bool kGreenContexts = true;     // SM partitions for the prefill/decode split
inline constexpr bool kBulkCopy = true;          // TMA tiles (set_attention_bulk)

}  // namespace ragon::gpu

// Device code compiled by this backend's compiler sees the portable device helpers of the interface headers
// (pos_value, seq_of, ...). Code that is CUDA-only (inline PTX) keeps testing __CUDACC__ itself.
#if defined(__CUDACC__)
#define RAGON_GPU_DEVICE_CODE 1
#endif
