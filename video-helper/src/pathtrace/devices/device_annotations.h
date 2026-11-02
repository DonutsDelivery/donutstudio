#pragma once

// Function and data annotations of the code the path integrator shares between
// devices. They expand to nothing a host compiler treats differently: a host
// build sees `inline` where a function was inline and nothing where it was not.
//
//   PT_DEVICE_FN       device-only and inline: the body's helpers and device shims
//   PT_DEVICE_ENTRY    device-only, keeps host linkage: the body's entry points
//   PT_SHARED_FN       host and device, inline: header functions
//   PT_SHARED_ENTRY    host and device, keeps host linkage: out-of-line functions
//   PT_DEVICE_CONSTANT constant data read on both sides
//   PT_NO_UNROLL       before a loop: nvcc keeps one copy of the loop's body
//
// tests/pathtrace_device_portability_source_contract.py checks that the code
// these mark stays free of host-only constructs.

#if defined(__CUDACC__)
#define PT_DEVICE_FN __device__ __forceinline__
#define PT_DEVICE_ENTRY __device__
#define PT_SHARED_FN __host__ __device__ __forceinline__
#define PT_SHARED_ENTRY __host__ __device__
#define PT_DEVICE_CONSTANT static __constant__
#define PT_NO_UNROLL _Pragma ("unroll 1")
#else
#define PT_DEVICE_FN inline
#define PT_DEVICE_ENTRY
#define PT_SHARED_FN inline
#define PT_SHARED_ENTRY
#define PT_DEVICE_CONSTANT inline
#define PT_NO_UNROLL
#endif
