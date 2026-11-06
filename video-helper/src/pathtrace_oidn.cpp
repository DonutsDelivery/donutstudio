#include "pathtrace_oidn.h"

#include <cstddef>
#include <mutex>
#include <type_traits>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace videorender::fixture3d::pathtracedenoise
{
namespace
{

// The C API's values (OpenImageDenoise/oidn.h, stable since OIDN 2.0).
constexpr int kDeviceTypeCpu = 1;
constexpr int kDeviceTypeSycl = 2;
constexpr int kDeviceTypeCuda = 3;
constexpr int kDeviceTypeHip = 4;
constexpr int kErrorNone = 0;
constexpr int kFormatFloat3 = 3;

using Handle = void*;

struct OidnApi
{
    Handle (*newDevice) (int) = nullptr;
    void (*commitDevice) (Handle) = nullptr;
    int (*getDeviceError) (Handle, const char**) = nullptr;
    void (*releaseDevice) (Handle) = nullptr;
    Handle (*newBuffer) (Handle, std::size_t) = nullptr;
    void (*writeBuffer) (Handle, std::size_t, std::size_t, const void*) = nullptr;
    void (*readBuffer) (Handle, std::size_t, std::size_t, void*) = nullptr;
    void (*releaseBuffer) (Handle) = nullptr;
    Handle (*newFilter) (Handle, const char*) = nullptr;
    void (*setFilterImage) (Handle, const char*, Handle, int, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t) = nullptr;
    void (*setFilterBool) (Handle, const char*, bool) = nullptr;
    void (*commitFilter) (Handle) = nullptr;
    void (*executeFilter) (Handle) = nullptr;
    void (*releaseFilter) (Handle) = nullptr;
};

#if defined(_WIN32)
constexpr const char* kLibraries[] { "OpenImageDenoise.dll" };
#elif defined(__APPLE__)
constexpr const char* kLibraries[] { "libOpenImageDenoise.2.dylib", "libOpenImageDenoise.dylib" };
#else
constexpr const char* kLibraries[] { "libOpenImageDenoise.so.2", "libOpenImageDenoise.so" };
#endif

std::mutex& lock()
{
    static std::mutex mutex;
    return mutex;
}

const char* testLibrary = nullptr;
bool attempted = false;
std::string problem;
OidnApi api;

// One device per kind, made on first use and kept, as making one loads its
// kernels. A kind that failed keeps its message and is not tried again.
struct DeviceSlot
{
    Handle device = nullptr;
    bool tried = false;
    std::string failure;
};
DeviceSlot slots[4];

void load()
{
    if (attempted)
        return;
    attempted = true;
    void* library = nullptr;
    std::string tried;
    const auto open = [&] (const char* name) {
        if (! tried.empty())
            tried += ", ";
        tried += name;
#if defined(_WIN32)
        library = reinterpret_cast<void*> (LoadLibraryA (name));
#else
        library = dlopen (name, RTLD_NOW | RTLD_LOCAL);
#endif
    };
    if (testLibrary != nullptr)
        open (testLibrary);
    else
        for (const char* name : kLibraries)
            if (library == nullptr)
                open (name);
    if (library == nullptr)
    {
        problem = "the OIDN library is not installed (looked for " + tried + ")";
        return;
    }
    const auto find = [library] (const char* name) -> void* {
#if defined(_WIN32)
        return reinterpret_cast<void*> (GetProcAddress (static_cast<HMODULE> (library), name));
#else
        return dlsym (library, name);
#endif
    };
    const auto bind = [&] (auto& function, const char* name) {
        function = reinterpret_cast<std::remove_reference_t<decltype (function)>> (find (name));
        if (function == nullptr && problem.empty())
            problem = std::string ("the OIDN library has no ") + name;
    };
    bind (api.newDevice, "oidnNewDevice");
    bind (api.commitDevice, "oidnCommitDevice");
    bind (api.getDeviceError, "oidnGetDeviceError");
    bind (api.releaseDevice, "oidnReleaseDevice");
    bind (api.newBuffer, "oidnNewBuffer");
    bind (api.writeBuffer, "oidnWriteBuffer");
    bind (api.readBuffer, "oidnReadBuffer");
    bind (api.releaseBuffer, "oidnReleaseBuffer");
    bind (api.newFilter, "oidnNewFilter");
    bind (api.setFilterImage, "oidnSetFilterImage");
    bind (api.setFilterBool, "oidnSetFilterBool");
    bind (api.commitFilter, "oidnCommitFilter");
    bind (api.executeFilter, "oidnExecuteFilter");
    bind (api.releaseFilter, "oidnReleaseFilter");
}

// The device's pending error, cleared by reading it.
std::string takeError (Handle device)
{
    const char* message = nullptr;
    if (api.getDeviceError (device, &message) == kErrorNone)
        return {};
    return message != nullptr && *message != '\0' ? message : "an unnamed OIDN error";
}

Handle deviceFor (OidnDevice kind)
{
    DeviceSlot& slot = slots[static_cast<int> (kind)];
    if (slot.tried)
        return slot.device;
    slot.tried = true;
    const int type = kind == OidnDevice::Cuda ? kDeviceTypeCuda
                   : kind == OidnDevice::Hip  ? kDeviceTypeHip
                   : kind == OidnDevice::Sycl ? kDeviceTypeSycl
                                              : kDeviceTypeCpu;
    Handle device = api.newDevice (type);
    if (device == nullptr)
    {
        slot.failure = std::string ("OIDN has no ") + oidnDeviceName (kind) + " device here";
        // OIDN's SYCL device is a separate module that needs Intel's DPC++ runtime and GPU.
        if (kind == OidnDevice::Sycl)
            slot.failure += " (its libOpenImageDenoise_device_sycl module, Intel's oneAPI DPC++ runtime or an Intel GPU is missing)";
        return nullptr;
    }
    api.commitDevice (device);
    if (auto error = takeError (device); ! error.empty())
    {
        slot.failure = std::string ("OIDN's ") + oidnDeviceName (kind) + " device did not start: " + error;
        api.releaseDevice (device);
        return nullptr;
    }
    slot.device = device;
    return device;
}

bool run (Handle device, const DenoiseFrame& frame, std::string& error)
{
    const std::size_t bytes = static_cast<std::size_t> (frame.width) * frame.height * kGuideChannels * sizeof (float);
    Handle buffers[4] {};
    Handle filter = nullptr;
    const auto release = [&] {
        if (filter != nullptr)
            api.releaseFilter (filter);
        for (Handle buffer : buffers)
            if (buffer != nullptr)
                api.releaseBuffer (buffer);
    };
    const float* inputs[3] { frame.color, frame.albedo, frame.normal };
    for (int index = 0; index < 4; ++index)
    {
        buffers[index] = api.newBuffer (device, bytes);
        if (buffers[index] == nullptr)
        {
            error = "OIDN could not allocate a frame buffer: " + takeError (device);
            release();
            return false;
        }
        if (index < 3)
            api.writeBuffer (buffers[index], 0, bytes, inputs[index]);
    }
    filter = api.newFilter (device, "RT");
    if (filter == nullptr)
    {
        error = "OIDN has no RT filter: " + takeError (device);
        release();
        return false;
    }
    const char* names[4] { "color", "albedo", "normal", "output" };
    for (int index = 0; index < 4; ++index)
        api.setFilterImage (filter, names[index], buffers[index], kFormatFloat3, frame.width, frame.height, 0, 0, 0);
    api.setFilterBool (filter, "hdr", frame.hdr);
    api.commitFilter (filter);
    api.executeFilter (filter);
    if (auto message = takeError (device); ! message.empty())
    {
        error = "OIDN failed: " + message;
        release();
        return false;
    }
    api.readBuffer (buffers[3], 0, bytes, frame.output);
    error = takeError (device);
    release();
    if (! error.empty())
    {
        error = "OIDN failed reading its output: " + error;
        return false;
    }
    return true;
}

} // namespace

std::string oidnLoadProblem()
{
    const std::lock_guard<std::mutex> guard (lock());
    load();
    return problem;
}

void setOidnLibraryForTests (const char* path)
{
    const std::lock_guard<std::mutex> guard (lock());
    testLibrary = path;
    attempted = false;
    problem.clear();
    api = {};
    for (DeviceSlot& slot : slots)
        slot = {};
}

bool oidnDenoise (OidnDevice want, const DenoiseFrame& frame, OidnDevice& used, std::string& note, std::string& error)
{
    const std::lock_guard<std::mutex> guard (lock());
    load();
    note.clear();
    if (! problem.empty())
    {
        error = problem;
        return false;
    }
    if (frame.width == 0 || frame.height == 0 || frame.color == nullptr || frame.albedo == nullptr
        || frame.normal == nullptr || frame.output == nullptr)
    {
        error = "OIDN received an incomplete frame";
        return false;
    }
    used = want;
    Handle device = deviceFor (want);
    if (device == nullptr && want != OidnDevice::Cpu)
    {
        note = slots[static_cast<int> (want)].failure + ", so it ran on the CPU";
        used = OidnDevice::Cpu;
        device = deviceFor (OidnDevice::Cpu);
    }
    if (device == nullptr)
    {
        error = slots[static_cast<int> (OidnDevice::Cpu)].failure;
        return false;
    }
    return run (device, frame, error);
}

} // namespace videorender::fixture3d::pathtracedenoise
