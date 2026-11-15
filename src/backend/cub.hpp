// The device-wide primitives' library of the build's backend (BACKEND §3.1): gpu::cub is CUB on CUDA and hipCUB on
// HIP, the same interface. Included by .cu files only.
#pragma once

#include "backend/backend.hpp"

#if defined(RAGON_BACKEND_HIP)
#include <hipcub/hipcub.hpp>
namespace ragon::gpu {
namespace cub = ::hipcub;
}
#else
#include <cub/cub.cuh>
namespace ragon::gpu {
namespace cub = ::cub;
}
#endif
