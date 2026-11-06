// HIP device driver TU: the compute kernel (device_compute_kernel.inl) for AMD
// GPUs. hipcc compiles it to a code object, which the build embeds and
// device_compute_host.cpp loads through libamdhip64.
//
// The shared code picks its device paths on __CUDACC__ (atomicAdd, isfinite,
// __float_as_uint), which HIP provides under the same names, so this defines it
// after the HIP runtime header.

#define PATHTRACE_DEVICE_HIP 1

#include <hip/hip_runtime.h>

#if ! defined(__CUDACC__)
#define __CUDACC__ 1
#endif

#include "device_compute_kernel.inl"
