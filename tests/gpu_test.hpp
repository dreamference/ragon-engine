// The kernel tests' device memory and error calls, for either backend (BACKEND §3, §7), and the host's exact decode of
// the FP8 formats the kernels read.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

#include "backend/backend.hpp"

#if defined(RAGON_BACKEND_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace ragon::test {

#if defined(RAGON_BACKEND_HIP)
inline void* alloc_bytes(size_t n) {
  void* p = nullptr;
  if (hipMalloc(&p, n ? n : 1) != hipSuccess) return nullptr;
  return p;
}
inline void free_bytes(void* p) { (void)hipFree(p); }
inline void h2d(void* d, const void* h, size_t n) { (void)hipMemcpy(d, h, n, hipMemcpyHostToDevice); }
inline void d2h(void* h, const void* d, size_t n) { (void)hipMemcpy(h, d, n, hipMemcpyDeviceToHost); }
inline void zero(void* d, size_t n) { (void)hipMemset(d, 0, n); }
// nullptr when every launch so far succeeded, else the error's text.
inline const char* sync_error() {
  const hipError_t e = hipDeviceSynchronize();
  return e == hipSuccess ? nullptr : hipGetErrorString(e);
}
inline void d2d(void* d, const void* s, size_t n) { (void)hipMemcpy(d, s, n, hipMemcpyDeviceToDevice); }
inline void fill(void* d, int v, size_t n) { (void)hipMemset(d, v, n); }
// nullptr when no launch failed since the last call, else the error's text (cudaGetLastError's contract).
inline const char* last_error() {
  const hipError_t e = hipGetLastError();
  return e == hipSuccess ? nullptr : hipGetErrorString(e);
}
// Milliseconds between two points of the null stream (an event pair).
struct Timer {
  hipEvent_t a{}, b{};
  Timer() { (void)hipEventCreate(&a), (void)hipEventCreate(&b); }
  ~Timer() { (void)hipEventDestroy(a), (void)hipEventDestroy(b); }
  void start() { (void)hipEventRecord(a); }
  float stop() {
    float ms = 0;
    (void)hipEventRecord(b), (void)hipEventSynchronize(b), (void)hipEventElapsedTime(&ms, a, b);
    return ms;
  }
};
#else
inline void* alloc_bytes(size_t n) {
  void* p = nullptr;
  if (cudaMalloc(&p, n ? n : 1) != cudaSuccess) return nullptr;
  return p;
}
inline void free_bytes(void* p) { cudaFree(p); }
inline void h2d(void* d, const void* h, size_t n) { cudaMemcpy(d, h, n, cudaMemcpyHostToDevice); }
inline void d2h(void* h, const void* d, size_t n) { cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost); }
inline void zero(void* d, size_t n) { cudaMemset(d, 0, n); }
inline const char* sync_error() {
  const cudaError_t e = cudaDeviceSynchronize();
  return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
inline void d2d(void* d, const void* s, size_t n) { cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice); }
inline void fill(void* d, int v, size_t n) { cudaMemset(d, v, n); }
inline const char* last_error() {
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
struct Timer {
  cudaEvent_t a{}, b{};
  Timer() { cudaEventCreate(&a), cudaEventCreate(&b); }
  ~Timer() { cudaEventDestroy(a), cudaEventDestroy(b); }
  void start() { cudaEventRecord(a); }
  float stop() {
    float ms = 0;
    cudaEventRecord(b), cudaEventSynchronize(b), cudaEventElapsedTime(&ms, a, b);
    return ms;
  }
};
#endif

// cudaMalloc's form: *p gets n bytes of device memory.
template <typename T>
void alloc_into(T** p, size_t n) {
  *p = static_cast<T*>(alloc_bytes(n));
}

template <typename T>
T* alloc(size_t n) {
  return static_cast<T*>(alloc_bytes(n * sizeof(T)));
}
template <typename T>
T* upload(const std::vector<T>& h) {
  T* d = alloc<T>(h.size());
  h2d(d, h.data(), h.size() * sizeof(T));
  return d;
}
template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  d2h(h.data(), d, n * sizeof(T));
  return h;
}

// OCP FP8 E4M3 (bias 7, no infinities, S.1111.111 NaN), exactly.
inline float e4m3(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  float v;
  if (e == 15 && m == 7) v = __builtin_nanf("");
  else if (e == 0) v = std::ldexp(static_cast<float>(m), -9);  // m / 8 * 2^-6
  else v = std::ldexp(static_cast<float>(8 + m), e - 10);     // (1 + m / 8) * 2^(e - 7)
  return (b & 0x80) ? -v : v;
}

}  // namespace ragon::test
