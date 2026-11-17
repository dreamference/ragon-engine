// The HIP backend's runtime calls (BACKEND §5.2, §7): the same ragon::gpu names as backend/cuda/runtime.hpp, each the
// HIP call of the same meaning (TheRock 10.0.0's hip_runtime_api.h has every one). Green contexts have no HIP twin
// and stay out of this layer: the engine keeps one context and priority streams on HIP (§4, kGreenContexts false).
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

#include "backend/hip/types.hpp"

namespace ragon::gpu {

using Error = hipError_t;
using Event = hipEvent_t;
using Graph = hipGraph_t;
using GraphExec = hipGraphExec_t;
using MemPool = hipMemPool_t;
using MemPoolProps = hipMemPoolProps;
using CopyKind = hipMemcpyKind;
using CaptureStatus = hipStreamCaptureStatus;
using CaptureMode = hipStreamCaptureMode;
using PoolAttr = hipMemPoolAttr;

inline constexpr Error kSuccess = hipSuccess;
inline constexpr Error kNotReady = hipErrorNotReady;  // an event query on work not yet done
inline constexpr CopyKind kH2D = hipMemcpyHostToDevice;
inline constexpr CopyKind kD2H = hipMemcpyDeviceToHost;
inline constexpr CopyKind kD2D = hipMemcpyDeviceToDevice;
inline constexpr unsigned kEventDisableTiming = hipEventDisableTiming;
inline constexpr unsigned kStreamNonBlocking = hipStreamNonBlocking;
inline constexpr unsigned kHostAllocDefault = hipHostMallocDefault;
inline constexpr CaptureMode kCaptureThreadLocal = hipStreamCaptureModeThreadLocal;
inline constexpr CaptureStatus kCaptureNone = hipStreamCaptureStatusNone;
// No node priorities on HIP (AMD inventory row 71): the step graphs instantiate without them; the fork stream keeps
// its stream priority.
inline constexpr unsigned long long kGraphNodePriority = 0;
inline constexpr PoolAttr kPoolReleaseThreshold = hipMemPoolAttrReleaseThreshold;
inline constexpr hipMemAllocationType kAllocPinned = hipMemAllocationTypePinned;
inline constexpr hipMemLocationType kLocationDevice = hipMemLocationTypeDevice;

// Errors
inline Error last_error() { return hipGetLastError(); }
inline const char* error_string(Error e) { return hipGetErrorString(e); }

// Device and memory
inline Error device_sync() { return hipDeviceSynchronize(); }
inline Error get_device(int* d) { return hipGetDevice(d); }
#ifdef _WIN32
// The device's free and total memory from the DXGI budget (AMD §1.1, :400): hipMemGetInfo over-reports under WDDM.
// free = the adapter's local budget less its current usage, total = the budget (backend/hip/mem_info.cpp).
Error mem_info(size_t* free_b, size_t* total_b);
#else
inline Error mem_info(size_t* free_b, size_t* total_b) { return hipMemGetInfo(free_b, total_b); }
#endif
template <typename T>
inline Error device_alloc(T** p, size_t bytes) { return hipMalloc(reinterpret_cast<void**>(p), bytes); }
inline Error device_free(void* p) { return hipFree(p); }
inline Error host_alloc(void** p, size_t bytes, unsigned flags) { return hipHostMalloc(p, bytes, flags); }
inline Error host_free(void* p) { return hipHostFree(p); }
inline Error free_async(void* p, Stream s) { return hipFreeAsync(p, s); }
inline Error pool_create(MemPool* pool, const MemPoolProps* props) { return hipMemPoolCreate(pool, props); }
inline Error pool_set_attribute(MemPool pool, PoolAttr a, void* v) { return hipMemPoolSetAttribute(pool, a, v); }
inline Error pool_destroy(MemPool pool) { return hipMemPoolDestroy(pool); }
inline Error pool_alloc_async(void** p, size_t bytes, MemPool pool, Stream s) {
  return hipMallocFromPoolAsync(p, bytes, pool, s);
}

// Copies and fills
inline Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind k) { return hipMemcpy(dst, src, bytes, k); }
inline Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind k, Stream s) {
  return hipMemcpyAsync(dst, src, bytes, k, s);
}
inline Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                              CopyKind k, Stream s) {
  return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, k, s);
}
inline Error memset_sync(void* p, int v, size_t bytes) { return hipMemset(p, v, bytes); }
inline Error memset_async(void* p, int v, size_t bytes, Stream s) { return hipMemsetAsync(p, v, bytes, s); }
inline Error copy_d2d_async(void* dst, const void* src, size_t bytes, Stream s) {
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, s);
}

// Streams
inline Error stream_create(Stream* s, unsigned flags) { return hipStreamCreateWithFlags(s, flags); }
inline Error stream_create_priority(Stream* s, unsigned flags, int priority) {
  return hipStreamCreateWithPriority(s, flags, priority);
}
inline Error stream_priority_range(int* least, int* greatest) {
  return hipDeviceGetStreamPriorityRange(least, greatest);
}
inline Error stream_destroy(Stream s) { return hipStreamDestroy(s); }
inline Error stream_sync(Stream s) { return hipStreamSynchronize(s); }
inline Error stream_wait_event(Stream s, Event e, unsigned flags) { return hipStreamWaitEvent(s, e, flags); }

// Events
inline Error event_create(Event* e) { return hipEventCreate(e); }
inline Error event_create_flags(Event* e, unsigned flags) { return hipEventCreateWithFlags(e, flags); }
inline Error event_destroy(Event e) { return hipEventDestroy(e); }
inline Error event_record(Event e, Stream s) { return hipEventRecord(e, s); }
inline Error event_sync(Event e) { return hipEventSynchronize(e); }
inline Error event_query(Event e) { return hipEventQuery(e); }
inline Error event_elapsed(float* ms, Event start, Event end) { return hipEventElapsedTime(ms, start, end); }

// Graphs (stream capture)
inline Error capture_begin(Stream s, CaptureMode mode) { return hipStreamBeginCapture(s, mode); }
inline Error capture_end(Stream s, Graph* g) { return hipStreamEndCapture(s, g); }
inline Error is_capturing(Stream s, CaptureStatus* status) { return hipStreamIsCapturing(s, status); }
inline Error graph_instantiate(GraphExec* exec, Graph g, unsigned long long flags) {
  return hipGraphInstantiateWithFlags(exec, g, flags);
}
inline Error graph_launch(GraphExec exec, Stream s) { return hipGraphLaunch(exec, s); }
inline Error graph_destroy(Graph g) { return hipGraphDestroy(g); }
inline Error graph_exec_destroy(GraphExec exec) { return hipGraphExecDestroy(exec); }

}  // namespace ragon::gpu
