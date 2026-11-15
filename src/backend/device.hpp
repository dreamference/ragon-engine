// The device half of the backend interface (BACKEND §3.1): what the kernels' .cu files spell differently between
// CUDA and HIP besides the types of types.hpp. Included by .cu files only (it declares __device__ functions).
#pragma once

#include "backend/backend.hpp"

#if defined(RAGON_BACKEND_HIP)
#include "backend/hip/device.hpp"
#else
#include "backend/cuda/device.hpp"
#endif
