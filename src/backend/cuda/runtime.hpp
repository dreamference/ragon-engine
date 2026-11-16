// The CUDA backend's runtime calls (BACKEND §5.2): one-line forwards with the CUDA argument order and types, inlined,
// so a host file spelled with them compiles to the code it compiled to before (§6). No defaults, no conversions.
// Each is a constexpr reference to the CUDA function itself (not a wrapper), so a call is the same call; a wrapper,
// even inlined, let GCC order the operands of a few compares differently (§6).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "backend/cuda/types.hpp"

namespace ragon::gpu {

using Error = cudaError_t;
using Event = cudaEvent_t;
using Graph = cudaGraph_t;
using GraphExec = cudaGraphExec_t;
using MemPool = cudaMemPool_t;
using MemPoolProps = cudaMemPoolProps;
using CopyKind = cudaMemcpyKind;
using CaptureStatus = cudaStreamCaptureStatus;
using CaptureMode = cudaStreamCaptureMode;
using PoolAttr = cudaMemPoolAttr;

inline constexpr Error kSuccess = cudaSuccess;
inline constexpr Error kNotReady = cudaErrorNotReady;  // an event query on work not yet done
inline constexpr CopyKind kH2D = cudaMemcpyHostToDevice;
inline constexpr CopyKind kD2H = cudaMemcpyDeviceToHost;
inline constexpr CopyKind kD2D = cudaMemcpyDeviceToDevice;
inline constexpr unsigned kEventDisableTiming = cudaEventDisableTiming;
inline constexpr unsigned kStreamNonBlocking = cudaStreamNonBlocking;
inline constexpr unsigned kHostAllocDefault = cudaHostAllocDefault;
inline constexpr CaptureMode kCaptureThreadLocal = cudaStreamCaptureModeThreadLocal;
inline constexpr CaptureStatus kCaptureNone = cudaStreamCaptureStatusNone;
inline constexpr unsigned long long kGraphNodePriority = cudaGraphInstantiateFlagUseNodePriority;
inline constexpr PoolAttr kPoolReleaseThreshold = cudaMemPoolAttrReleaseThreshold;
inline constexpr cudaMemAllocationType kAllocPinned = cudaMemAllocationTypePinned;
inline constexpr cudaMemLocationType kLocationDevice = cudaMemLocationTypeDevice;

// Errors
inline constexpr Error (&last_error)() = cudaGetLastError;
inline constexpr const char* (&error_string)(Error) = cudaGetErrorString;

// Device and memory
inline constexpr Error (&device_sync)() = cudaDeviceSynchronize;
inline constexpr Error (&get_device)(int*) = cudaGetDevice;
inline constexpr Error (&mem_info)(size_t*, size_t*) = cudaMemGetInfo;
template <typename T>
inline Error device_alloc(T** p, size_t bytes) { return cudaMalloc(p, bytes); }  // cuda_runtime.h's T** overload
inline constexpr Error (&device_free)(void*) = cudaFree;
inline constexpr Error (&host_alloc)(void**, size_t, unsigned) = cudaHostAlloc;
inline constexpr Error (&host_free)(void*) = cudaFreeHost;
inline constexpr Error (&free_async)(void*, Stream) = cudaFreeAsync;
inline constexpr Error (&pool_create)(MemPool*, const MemPoolProps*) = cudaMemPoolCreate;
inline constexpr Error (&pool_set_attribute)(MemPool, PoolAttr, void*) = cudaMemPoolSetAttribute;
inline constexpr Error (&pool_destroy)(MemPool) = cudaMemPoolDestroy;
inline constexpr Error (&pool_alloc_async)(void**, size_t, MemPool, Stream) = cudaMallocFromPoolAsync;

// Copies and fills
inline constexpr Error (&memcpy_sync)(void*, const void*, size_t, CopyKind) = cudaMemcpy;
inline constexpr Error (&memcpy_async)(void*, const void*, size_t, CopyKind, Stream) = cudaMemcpyAsync;
inline constexpr Error (&memcpy_2d_async)(void*, size_t, const void*, size_t, size_t, size_t, CopyKind,
                                          Stream) = cudaMemcpy2DAsync;
inline constexpr Error (&memset_sync)(void*, int, size_t) = cudaMemset;
inline constexpr Error (&memset_async)(void*, int, size_t, Stream) = cudaMemsetAsync;
inline Error copy_d2d_async(void* dst, const void* src, size_t bytes, Stream s) {
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
}

// Streams
inline constexpr Error (&stream_create)(Stream*, unsigned) = cudaStreamCreateWithFlags;
inline constexpr Error (&stream_create_priority)(Stream*, unsigned, int) = cudaStreamCreateWithPriority;
inline constexpr Error (&stream_priority_range)(int*, int*) = cudaDeviceGetStreamPriorityRange;
inline constexpr Error (&stream_destroy)(Stream) = cudaStreamDestroy;
inline constexpr Error (&stream_sync)(Stream) = cudaStreamSynchronize;
inline constexpr Error (&stream_wait_event)(Stream, Event, unsigned) = cudaStreamWaitEvent;

// Events
inline constexpr Error (&event_create)(Event*) = cudaEventCreate;
inline constexpr Error (&event_create_flags)(Event*, unsigned) = cudaEventCreateWithFlags;
inline constexpr Error (&event_destroy)(Event) = cudaEventDestroy;
inline constexpr Error (&event_record)(Event, Stream) = cudaEventRecord;
inline constexpr Error (&event_sync)(Event) = cudaEventSynchronize;
inline constexpr Error (&event_query)(Event) = cudaEventQuery;
inline constexpr Error (&event_elapsed)(float*, Event, Event) = cudaEventElapsedTime;

// Graphs (stream capture)
inline constexpr Error (&capture_begin)(Stream, CaptureMode) = cudaStreamBeginCapture;
inline constexpr Error (&capture_end)(Stream, Graph*) = cudaStreamEndCapture;
inline constexpr Error (&is_capturing)(Stream, CaptureStatus*) = cudaStreamIsCapturing;
inline constexpr Error (&graph_instantiate)(GraphExec*, Graph, unsigned long long) = cudaGraphInstantiate;
inline constexpr Error (&graph_launch)(GraphExec, Stream) = cudaGraphLaunch;
inline constexpr Error (&graph_destroy)(Graph) = cudaGraphDestroy;
inline constexpr Error (&graph_exec_destroy)(GraphExec) = cudaGraphExecDestroy;

}  // namespace ragon::gpu
