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
// Address spaces, for the Metal kernel (device_metal.metal), where every pointer
// and reference names one. They are empty for C++, CUDA, HIP and SYCL.
//
//   PT_GLOBAL          scene and film buffers the host uploads (Metal `device`)
//   PT_TABLE_SPACE     the lookup tables (Metal `constant`)
//   PT_VIEW_SPACE      the light set view a SceneView points at. The Metal kernel
//                      copies it out of its buffer into its own values (Metal
//                      `thread`), so its member functions run on it as on the host.
//   PT_GLOBAL_READ(T)  a name for a T in uploaded memory that code taking a
//                      function's own values reads: a const reference, and on
//                      Metal a const copy.
//   PT_THREAD          a function's own values (Metal `thread`). The Metal source
//                      assembler (cmake/metal_kernel_source.py) adds `thread` to
//                      any pointer or reference that names no space, so the code
//                      writes this only where a declarator needs it spelled out.
//
// PtCounterWord is the 64-bit word the device adds counters into: CUDA's
// atomicAdd takes unsigned long long, which Metal does not have.
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

#if defined(PATHTRACE_DEVICE_METAL)
#define PT_GLOBAL device
#define PT_TABLE_SPACE constant
#define PT_THREAD thread
#define PT_VIEW_SPACE thread
#define PT_GLOBAL_READ(T) const T
typedef ulong PtCounterWord;
#else
#define PT_GLOBAL
#define PT_TABLE_SPACE
#define PT_THREAD
#define PT_VIEW_SPACE
#define PT_GLOBAL_READ(T) const T&
typedef unsigned long long PtCounterWord;
static_assert (sizeof (PtCounterWord) == 8, "device counters are 64-bit words");
#endif
