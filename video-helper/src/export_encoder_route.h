#pragma once

// Which video encoder an export uses and how its frames reach it (Plan 3,
// export and encoders). Hardware encoders take frames from GPU memory where the
// platform allows it: NVENC through CUDA-GL interop, VideoToolbox through the
// IOSurface the Metal compositor renders into. VA-API and AMF take an upload.
// The CPU handoff happens only for a software encoder or a named fallback, and
// the export record says which.
//
// Nothing here touches FFmpeg: the caller passes whether an encoder exists, so
// the tests run without one.

#include <functional>
#include <string>
#include <vector>

namespace videohelper::exportroute
{

enum class Backend { Software, Nvenc, VideoToolbox, Vaapi, Amf };
enum class Platform { Linux, Windows, Apple };

// How each frame reaches the encoder.
enum class FrameRoute
{
    CpuFrames,              // software encoder: the readback is the input
    CudaFromGl,             // NVENC: GL composite -> CUDA frame, no readback
    VideoToolboxIOSurface,  // VideoToolbox: the IOSurface is the input, no readback
    NvencUpload,            // NVENC fed from the CPU after a named fallback
    VaapiUpload,            // readback, then upload into a VA-API surface
    VaapiDmaBuf,            // VA-API: the surface's planes, imported into GL, are rendered into
    AmfUpload,              // readback, then AMF uploads the frame
    VideoToolboxUpload      // VideoToolbox fed from the CPU after a named fallback
};

Platform currentPlatform();

// The stored setting: "auto", "software", "nvenc", "videotoolbox", "vaapi", "amf".
bool isEncoderSetting (const std::string& setting);
const char* backendSetting (Backend backend);
const char* backendName (Backend backend);  // "NVENC", "VA-API", ...
const char* frameRouteName (FrameRoute route);
bool framesStayOnGpu (FrameRoute route);

// h264_nvenc, hevc_vaapi, ... or nullptr when the codec has no hardware encoder
// on that backend. Software has no name here (softwareEncoderName in exporter.cpp).
const char* hardwareEncoderName (const std::string& codec, Backend backend);

// The backend of an FFmpeg encoder name: h264_nvenc -> Nvenc, libx264 -> Software.
Backend backendOfEncoder (const std::string& encoderName);

// The hardware encoders Auto tries, in order, on each platform.
std::vector<Backend> autoOrder (Platform platform);
// Whether the backend runs on the platform at all.
bool backendOnPlatform (Backend backend, Platform platform);

struct Choice
{
    bool ok = false;
    Backend backend = Backend::Software;
    std::string encoderName;  // empty for software: the caller names it
    std::string fallback;     // why Auto passed over each hardware encoder
    std::string error;        // an explicit request that cannot be met
};

// The encoder for a setting and codec. encoderExists answers whether FFmpeg has
// an encoder by that name. An explicit hardware request gets that encoder or an
// error naming why; Auto takes the first that exists and names those it passed.
Choice choose (const std::string& setting, const std::string& codec, Platform platform, bool hdr,
               const std::function<bool (const char*)>& encoderExists);

// The frame route a backend starts with, before the export knows whether the
// GPU path works. The export moves it to the upload route, by name, when not.
FrameRoute plannedRoute (Backend backend, bool hdr);
// The upload route a GPU route falls back to.
FrameRoute uploadRoute (Backend backend);

struct Route
{
    std::string encoder;  // the FFmpeg encoder that ran
    Backend backend = Backend::Software;
    FrameRoute frames = FrameRoute::CpuFrames;
    std::string fallback;  // empty when nothing fell back

    void fallBack (FrameRoute to, const std::string& reason);
    void addFallback (const std::string& reason);
};

// "h264_nvenc, frames stay on the GPU" or "... read back to the CPU (reason)".
std::string describe (const Route& route);

// The last export's route on this thread, for the export record.
void setLastRoute (const Route& route);
Route lastRoute();
void clearLastRoute();
bool haveLastRoute();

} // namespace videohelper::exportroute
