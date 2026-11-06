#pragma once

// The oneAPI device library's entry points (Plan 3, Render engines and devices).
//
// device_oneapi.cpp is the library: icpx compiles it with SYCL into its own shared
// library, beside the helper, so the helper starts without the SYCL or Level Zero
// runtime. It exports the CUDA driver API's shape under these names, so
// device_compute_host.cpp drives it with the class that drives CUDA and HIP.
//
// Each entry returns 0 on success or one of the codes below; kLastError then
// returns the runtime's own words. The library picks an Intel GPU, through Level
// Zero first, then OpenCL. ARBIT_ONEAPI_DEVICE_TYPE=cpu lets it take an Intel
// CPU's OpenCL device instead, for the conformance test on machines without the GPU.

namespace videohelper::pathtrace::device::oneapi
{

#if defined(_WIN32)
inline constexpr const char* kLibraryFile = "arbit-pathtrace-oneapi.dll";
#else
inline constexpr const char* kLibraryFile = "libarbit-pathtrace-oneapi.so";
#endif

inline constexpr const char* kInit = "arbitOneApiInit";
inline constexpr const char* kDeviceGet = "arbitOneApiDeviceGet";
inline constexpr const char* kCtxCreate = "arbitOneApiCtxCreate";
inline constexpr const char* kCtxSetCurrent = "arbitOneApiCtxSetCurrent";
inline constexpr const char* kCtxDestroy = "arbitOneApiCtxDestroy";
inline constexpr const char* kMemAlloc = "arbitOneApiMemAlloc";
inline constexpr const char* kMemFree = "arbitOneApiMemFree";
inline constexpr const char* kMemcpyHtoD = "arbitOneApiMemcpyHtoD";
inline constexpr const char* kMemcpyDtoH = "arbitOneApiMemcpyDtoH";
inline constexpr const char* kMemsetD8 = "arbitOneApiMemsetD8";
inline constexpr const char* kModuleLoadData = "arbitOneApiModuleLoadData";
inline constexpr const char* kModuleUnload = "arbitOneApiModuleUnload";
inline constexpr const char* kModuleGetFunction = "arbitOneApiModuleGetFunction";
inline constexpr const char* kModuleGetGlobal = "arbitOneApiModuleGetGlobal";
inline constexpr const char* kLaunchKernel = "arbitOneApiLaunchKernel";
inline constexpr const char* kSynchronize = "arbitOneApiSynchronize";
inline constexpr const char* kMemGetInfo = "arbitOneApiMemGetInfo";
// const char* (): the last failure's message on this thread.
inline constexpr const char* kLastError = "arbitOneApiLastError";
// int (char* text, size_t size): 0 and the device's name when the library finds a
// device it would use, otherwise a code and why not.
inline constexpr const char* kProbe = "arbitOneApiProbe";

// The image moduleLoadData takes. The kernel lives in the library; the string
// tells a library from another build apart; it changes with the parameter block.
inline constexpr const char kModuleImage[] = "arbit-pathtrace-oneapi-2";

enum Code : int
{
    kOk = 0,
    kFailed = 1,
    kOutOfMemory = 2,
    kNotInitialised = 3,
    kWrongImage = 4,
    kUnknownName = 5,
    kNoDevice = 100
};

} // namespace videohelper::pathtrace::device::oneapi
