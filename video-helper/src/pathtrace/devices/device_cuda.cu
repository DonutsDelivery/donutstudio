// CUDA device driver TU: the compute kernel (device_compute_kernel.inl) for
// NVIDIA GPUs without OptiX. nvcc compiles it to PTX, which the build embeds and
// the driver compiles for the GPU on first load (device_compute_host.cpp).

#define PATHTRACE_DEVICE_CUDA 1

#if ! defined(__CUDACC_RELAXED_CONSTEXPR__)
#error "Compile this kernel with --expt-relaxed-constexpr"
#endif

#include "device_compute_kernel.inl"
