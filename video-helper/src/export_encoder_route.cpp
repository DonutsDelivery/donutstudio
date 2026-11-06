#include "export_encoder_route.h"

namespace videohelper::exportroute
{

Platform currentPlatform()
{
#if defined(__APPLE__)
    return Platform::Apple;
#elif defined(_WIN32)
    return Platform::Windows;
#else
    return Platform::Linux;
#endif
}

bool isEncoderSetting (const std::string& setting)
{
    return setting == "auto" || setting == "software" || setting == "nvenc" || setting == "videotoolbox"
        || setting == "vaapi" || setting == "amf";
}

const char* backendSetting (Backend backend)
{
    switch (backend)
    {
        case Backend::Nvenc: return "nvenc";
        case Backend::VideoToolbox: return "videotoolbox";
        case Backend::Vaapi: return "vaapi";
        case Backend::Amf: return "amf";
        case Backend::Software: break;
    }
    return "software";
}

const char* backendName (Backend backend)
{
    switch (backend)
    {
        case Backend::Nvenc: return "NVENC";
        case Backend::VideoToolbox: return "VideoToolbox";
        case Backend::Vaapi: return "VA-API";
        case Backend::Amf: return "AMF";
        case Backend::Software: break;
    }
    return "software";
}

const char* frameRouteName (FrameRoute route)
{
    switch (route)
    {
        case FrameRoute::CudaFromGl: return "cuda-gl";
        case FrameRoute::VideoToolboxIOSurface: return "iosurface";
        case FrameRoute::NvencUpload: return "nvenc-upload";
        case FrameRoute::VaapiUpload: return "vaapi-upload";
        case FrameRoute::VaapiDmaBuf: return "vaapi-dmabuf";
        case FrameRoute::AmfUpload: return "amf-upload";
        case FrameRoute::VideoToolboxUpload: return "videotoolbox-upload";
        case FrameRoute::CpuFrames: break;
    }
    return "cpu";
}

bool framesStayOnGpu (FrameRoute route)
{
    return route == FrameRoute::CudaFromGl || route == FrameRoute::VideoToolboxIOSurface
        || route == FrameRoute::VaapiDmaBuf;
}

const char* hardwareEncoderName (const std::string& codec, Backend backend)
{
    if (codec != "h264" && codec != "h265")
        return nullptr;  // vp9, ProRes, FFV1 and DPX have no hardware path here
    const bool hevc = codec == "h265";
    switch (backend)
    {
        case Backend::Nvenc: return hevc ? "hevc_nvenc" : "h264_nvenc";
        case Backend::VideoToolbox: return hevc ? "hevc_videotoolbox" : "h264_videotoolbox";
        case Backend::Vaapi: return hevc ? "hevc_vaapi" : "h264_vaapi";
        case Backend::Amf: return hevc ? "hevc_amf" : "h264_amf";
        case Backend::Software: break;
    }
    return nullptr;
}

Backend backendOfEncoder (const std::string& encoderName)
{
    const auto endsWith = [&encoderName] (const std::string& suffix) {
        return encoderName.size() > suffix.size()
            && encoderName.compare (encoderName.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (endsWith ("_nvenc")) return Backend::Nvenc;
    if (endsWith ("_videotoolbox")) return Backend::VideoToolbox;
    if (endsWith ("_vaapi")) return Backend::Vaapi;
    if (endsWith ("_amf")) return Backend::Amf;
    return Backend::Software;
}

std::vector<Backend> autoOrder (Platform platform)
{
    switch (platform)
    {
        case Platform::Apple: return { Backend::VideoToolbox };
        case Platform::Windows: return { Backend::Nvenc, Backend::Amf };
        case Platform::Linux: break;
    }
    return { Backend::Nvenc, Backend::Vaapi };
}

bool backendOnPlatform (Backend backend, Platform platform)
{
    switch (backend)
    {
        case Backend::Software: return true;
        case Backend::Nvenc: return platform != Platform::Apple;
        case Backend::VideoToolbox: return platform == Platform::Apple;
        case Backend::Vaapi: return platform == Platform::Linux;
        case Backend::Amf: return platform == Platform::Windows;
    }
    return false;
}

namespace
{

Backend backendFromSetting (const std::string& setting)
{
    if (setting == "nvenc") return Backend::Nvenc;
    if (setting == "videotoolbox") return Backend::VideoToolbox;
    if (setting == "vaapi") return Backend::Vaapi;
    if (setting == "amf") return Backend::Amf;
    return Backend::Software;
}

const char* platformName (Platform platform)
{
    switch (platform)
    {
        case Platform::Apple: return "macOS";
        case Platform::Windows: return "Windows";
        case Platform::Linux: break;
    }
    return "Linux";
}

// Why the backend cannot take this export, or empty when it can.
std::string problem (Backend backend, const std::string& codec, Platform platform, bool hdr,
                     const std::function<bool (const char*)>& encoderExists)
{
    const std::string name = backendName (backend);
    if (! backendOnPlatform (backend, platform))
        return name + " does not run on " + platformName (platform);
    const char* encoder = hardwareEncoderName (codec, backend);
    if (encoder == nullptr)
        return name + " has no " + codec + " encoder; H.264 and H.265 only";
    // NVENC and VA-API take an HDR movie as P010 the GPU converts, and encode it
    // as H.265 Main10. AMF and VideoToolbox take 8-bit SDR frames here.
    const bool tenBit = backend == Backend::Nvenc || backend == Backend::Vaapi;
    if (hdr && tenBit && codec != "h265")
        return name + " encodes an HDR movie as H.265 Main10 only";
    if (hdr && ! tenBit)
        return name + " takes 8-bit SDR frames here, not an HDR movie";
    if (! encoderExists (encoder))
        return std::string (encoder) + " is not in this FFmpeg build";
    return {};
}

void appendReason (std::string& reasons, const std::string& reason)
{
    if (reason.empty())
        return;
    if (! reasons.empty())
        reasons += "; ";
    reasons += reason;
}

} // namespace

Choice choose (const std::string& setting, const std::string& codec, Platform platform, bool hdr,
               const std::function<bool (const char*)>& encoderExists)
{
    Choice choice;
    if (! isEncoderSetting (setting))
    {
        choice.error = "unknown encoder: " + setting;
        return choice;
    }
    if (setting == "software")
    {
        choice.ok = true;
        return choice;
    }
    if (setting != "auto")
    {
        const auto backend = backendFromSetting (setting);
        choice.error = problem (backend, codec, platform, hdr, encoderExists);
        if (! choice.error.empty())
            return choice;
        choice.ok = true;
        choice.backend = backend;
        choice.encoderName = hardwareEncoderName (codec, backend);
        return choice;
    }
    choice.ok = true;
    // Auto is hardware for H.264 and H.265 only; other codecs are software by
    // design, which is not a fallback.
    if (codec != "h264" && codec != "h265")
        return choice;
    for (const auto backend : autoOrder (platform))
    {
        const auto reason = problem (backend, codec, platform, hdr, encoderExists);
        if (reason.empty())
        {
            choice.backend = backend;
            choice.encoderName = hardwareEncoderName (codec, backend);
            return choice;
        }
        appendReason (choice.fallback, reason);
    }
    return choice;
}

FrameRoute plannedRoute (Backend backend, bool hdr)
{
    switch (backend)
    {
        // NVENC takes both SDR (NV12) and HDR (P010) planes the GPU converts.
        // VA-API surfaces, NV12 or P010, import into GL as dma-bufs. An HDR
        // movie is painted on the CPU for VideoToolbox.
        case Backend::Nvenc: return FrameRoute::CudaFromGl;
        case Backend::VideoToolbox: return hdr ? FrameRoute::VideoToolboxUpload : FrameRoute::VideoToolboxIOSurface;
        case Backend::Vaapi: return FrameRoute::VaapiDmaBuf;
        case Backend::Amf: return FrameRoute::AmfUpload;
        case Backend::Software: break;
    }
    return FrameRoute::CpuFrames;
}

FrameRoute uploadRoute (Backend backend)
{
    switch (backend)
    {
        case Backend::Nvenc: return FrameRoute::NvencUpload;
        case Backend::VideoToolbox: return FrameRoute::VideoToolboxUpload;
        case Backend::Vaapi: return FrameRoute::VaapiUpload;
        case Backend::Amf: return FrameRoute::AmfUpload;
        case Backend::Software: break;
    }
    return FrameRoute::CpuFrames;
}

void Route::fallBack (FrameRoute to, const std::string& reason)
{
    if (frames == to)
        return;
    frames = to;
    addFallback (reason);
}

void Route::addFallback (const std::string& reason)
{
    appendReason (fallback, reason);
}

std::string describe (const Route& route)
{
    std::string text = route.encoder;
    text += framesStayOnGpu (route.frames) ? ", frames stay on the GPU" : ", frames read back to the CPU";
    if (! route.fallback.empty())
        text += " (" + route.fallback + ")";
    return text;
}

namespace
{
// Per thread: an export and a render-cache build run on their own threads and
// each reads back its own route.
thread_local Route last;
thread_local bool haveLast = false;
} // namespace

void setLastRoute (const Route& route)
{
    last = route;
    haveLast = true;
}

Route lastRoute()
{
    return last;
}

void clearLastRoute()
{
    last = {};
    haveLast = false;
}

bool haveLastRoute()
{
    return haveLast;
}

} // namespace videohelper::exportroute
