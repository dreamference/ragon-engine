// The Metal device layer (BACKEND §8): one MTLDevice, one command queue, pipelines compiled at open from the
// embedded MSL (kernels.metal), and an encoder that collects every dispatch of a step into one command buffer.
// Objective-C++ only: include it from .mm files.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "backend/metal/types.hpp"

namespace ragon::metal {

// A device buffer and a byte offset into it (a tensor inside a mapped checkpoint shard, or a whole allocation).
struct View {
  id<MTLBuffer> buf = nil;
  size_t off = 0;
  bool empty() const { return buf == nil; }
  View at(size_t bytes) const { return {buf, off + bytes}; }
  template <class T>
  T* host() const {  // unified memory: the CPU reads and writes shared buffers directly
    return reinterpret_cast<T*>(static_cast<char*>([buf contents]) + off);
  }
};

struct Queue {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> q = nil;
  id<MTLLibrary> lib = nil;
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipes;

  // Opens the default device and compiles the kernels with safe math (no fast-math: correctness first).
  void open();
  // The pipeline of a kernel; GEMV kernels are specialized by the function constant kRows.
  id<MTLComputePipelineState> pipe(const char* name, uint32_t rows = 0);
  View alloc(size_t bytes);  // shared storage, zeroed
  id<MTLComputeCommandEncoder> encoder();
  // Ends the encoder, commits and waits; returns the GPU time of the buffer in seconds.
  double sync();
};

// Sets buffers 0..n-1 from views and the trailing argument struct (if any) with setBytes.
inline void bind(id<MTLComputeCommandEncoder> e, std::initializer_list<View> vs) {
  NSUInteger i = 0;
  for (const View& v : vs) [e setBuffer:v.buf offset:v.off atIndex:i++];
}
template <class A>
inline void bind_args(id<MTLComputeCommandEncoder> e, NSUInteger index, const A& a) {
  [e setBytes:&a length:sizeof(A) atIndex:index];
}

}  // namespace ragon::metal
