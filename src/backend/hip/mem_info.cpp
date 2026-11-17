// gpu::mem_info on Windows (HIP builds): the DXGI budget of the adapter the HIP device runs on, found by its LUID.
// hipMemGetInfo reports the carve-out plus shared memory under WDDM and over-reports what the process may use (AMD
// §1.1); the budget (IDXGIAdapter3::QueryVideoMemoryInfo, local segment) is what the OS lets this process keep
// resident. No silent fallback: an adapter that cannot be found or queried is an error.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_4.h>

#include <cstdio>
#include <cstring>

#include "backend/hip/runtime.hpp"

namespace ragon::gpu {

Error mem_info(size_t* free_b, size_t* total_b) {
  int dev = 0;
  if (Error e = hipGetDevice(&dev); e != hipSuccess) return e;
  hipDeviceProp_t prop{};
  if (Error e = hipGetDeviceProperties(&prop, dev); e != hipSuccess) return e;
  LUID luid{};
  static_assert(sizeof(luid) == sizeof(prop.luid));
  std::memcpy(&luid, prop.luid, sizeof(luid));
  IDXGIFactory4* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) {
    std::fprintf(stderr, "memory info: no DXGI factory\n");
    return hipErrorUnknown;
  }
  IDXGIAdapter3* adapter = nullptr;
  const HRESULT found = factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter));
  factory->Release();
  if (FAILED(found)) {
    std::fprintf(stderr, "memory info: no DXGI adapter with the HIP device's LUID\n");
    return hipErrorUnknown;
  }
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  const HRESULT queried = adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
  adapter->Release();
  if (FAILED(queried)) {
    std::fprintf(stderr, "memory info: the DXGI budget query failed\n");
    return hipErrorUnknown;
  }
  *total_b = static_cast<size_t>(info.Budget);
  *free_b = info.Budget > info.CurrentUsage ? static_cast<size_t>(info.Budget - info.CurrentUsage) : 0;
  return hipSuccess;
}

}  // namespace ragon::gpu
#endif  // _WIN32
