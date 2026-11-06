// Encoder choice and frame routes (Plan 3, export and encoders): the Auto order
// on each platform, explicit requests that cannot be met, the names of every
// fallback, and which routes keep frames on the GPU. FFmpeg is not linked: each
// test says which encoders "exist".

#include "export_encoder_route.h"

#include <iostream>
#include <set>
#include <string>

namespace
{

namespace route = videohelper::exportroute;
using route::Backend;
using route::FrameRoute;
using route::Platform;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

bool contains (const std::string& text, const std::string& part)
{
    return text.find (part) != std::string::npos;
}

std::function<bool (const char*)> having (std::set<std::string> names)
{
    return [names] (const char* name) { return names.count (name) != 0; };
}

const auto everything = having ({ "h264_nvenc", "hevc_nvenc", "h264_vaapi", "hevc_vaapi", "h264_amf", "hevc_amf",
                                  "h264_videotoolbox", "hevc_videotoolbox" });

void settingTests()
{
    for (const char* setting : { "auto", "software", "nvenc", "videotoolbox", "vaapi", "amf" })
        check (route::isEncoderSetting (setting), std::string ("accepts ") + setting);
    check (! route::isEncoderSetting ("qsv"), "refuses an unknown encoder");
    const auto unknown = route::choose ("qsv", "h264", Platform::Linux, false, everything);
    check (! unknown.ok && contains (unknown.error, "unknown encoder: qsv"), "an unknown setting names itself");

    for (const auto backend : { Backend::Nvenc, Backend::VideoToolbox, Backend::Vaapi, Backend::Amf })
    {
        const std::string name = route::hardwareEncoderName ("h265", backend);
        check (route::backendOfEncoder (name) == backend, "the encoder name reads back as its backend: " + name);
        check (std::string (route::backendSetting (backend)) != "software", "each backend has a setting");
    }
    check (route::backendOfEncoder ("libx264") == Backend::Software, "libx264 is software");
    check (route::backendOfEncoder ("_amf") == Backend::Software, "a bare suffix is not an encoder");
    check (route::hardwareEncoderName ("prores", Backend::Nvenc) == nullptr, "ProRes has no hardware encoder");
}

void autoOrderTests()
{
    auto choice = route::choose ("auto", "h264", Platform::Linux, false, everything);
    check (choice.ok && choice.encoderName == "h264_nvenc" && choice.fallback.empty(), "Linux Auto takes NVENC first");
    choice = route::choose ("auto", "h265", Platform::Linux, false, having ({ "hevc_vaapi" }));
    check (choice.ok && choice.backend == Backend::Vaapi && choice.encoderName == "hevc_vaapi",
           "Linux Auto takes VA-API on AMD and Intel");
    check (contains (choice.fallback, "hevc_nvenc is not in this FFmpeg build"), "and names NVENC: " + choice.fallback);

    choice = route::choose ("auto", "h264", Platform::Windows, false, having ({ "h264_amf" }));
    check (choice.ok && choice.encoderName == "h264_amf", "Windows Auto takes AMF after NVENC");
    choice = route::choose ("auto", "h264", Platform::Apple, false, everything);
    check (choice.ok && choice.encoderName == "h264_videotoolbox", "macOS Auto takes VideoToolbox");

    choice = route::choose ("auto", "h264", Platform::Linux, false, having ({}));
    check (choice.ok && choice.backend == Backend::Software && choice.encoderName.empty(),
           "Auto with no hardware encoder takes software");
    check (contains (choice.fallback, "h264_nvenc") && contains (choice.fallback, "h264_vaapi"),
           "and names every encoder it passed over: " + choice.fallback);

    choice = route::choose ("auto", "vp9", Platform::Linux, false, everything);
    check (choice.ok && choice.backend == Backend::Software && choice.fallback.empty(),
           "VP9 is software by design, not a fallback");

    choice = route::choose ("auto", "h265", Platform::Linux, true, having ({ "hevc_vaapi" }));
    check (choice.ok && choice.encoderName == "hevc_vaapi" && contains (choice.fallback, "hevc_nvenc"),
           "an HDR movie takes VA-API Main10 on AMD and Intel: " + choice.fallback);
    choice = route::choose ("vaapi", "h264", Platform::Linux, true, everything);
    check (! choice.ok && contains (choice.error, "H.265 Main10 only"), "HDR VA-API is H.265 only: " + choice.error);
    choice = route::choose ("vaapi", "h265", Platform::Linux, true, everything);
    check (choice.ok && choice.encoderName == "hevc_vaapi", "explicit VA-API takes an HDR movie");
    choice = route::choose ("auto", "h265", Platform::Windows, true, having ({ "hevc_amf" }));
    check (choice.ok && choice.backend == Backend::Software && contains (choice.fallback, "AMF takes 8-bit SDR"),
           "an HDR movie still passes AMF by name: " + choice.fallback);
    choice = route::choose ("auto", "h265", Platform::Linux, true, everything);
    check (choice.ok && choice.encoderName == "hevc_nvenc", "an HDR movie takes NVENC Main10 on Linux");
    choice = route::choose ("nvenc", "h264", Platform::Linux, true, everything);
    check (! choice.ok && contains (choice.error, "H.265 Main10 only"), "HDR NVENC is H.265 only: " + choice.error);
    choice = route::choose ("auto", "h265", Platform::Apple, true, everything);
    check (choice.ok && choice.backend == Backend::Software && contains (choice.fallback, "VideoToolbox takes 8-bit SDR"),
           "an HDR movie passes VideoToolbox by name: " + choice.fallback);
}

void explicitTests()
{
    auto choice = route::choose ("vaapi", "h264", Platform::Windows, false, everything);
    check (! choice.ok && contains (choice.error, "VA-API does not run on Windows"), "VA-API is Linux only: " + choice.error);
    choice = route::choose ("amf", "h264", Platform::Linux, false, everything);
    check (! choice.ok && contains (choice.error, "AMF does not run on Linux"), "AMF is Windows only: " + choice.error);
    choice = route::choose ("nvenc", "h264", Platform::Linux, false, having ({}));
    check (! choice.ok && contains (choice.error, "h264_nvenc is not in this FFmpeg build"),
           "a missing explicit encoder stops by name: " + choice.error);
    choice = route::choose ("nvenc", "prores", Platform::Linux, false, everything);
    check (! choice.ok && contains (choice.error, "H.264 and H.265 only"), "explicit NVENC for ProRes says why");
    choice = route::choose ("vaapi", "h265", Platform::Linux, false, everything);
    check (choice.ok && choice.encoderName == "hevc_vaapi" && choice.fallback.empty(), "explicit VA-API selects it");
    choice = route::choose ("software", "h264", Platform::Linux, false, everything);
    check (choice.ok && choice.backend == Backend::Software && choice.fallback.empty(), "software is not a fallback");
}

void routeTests()
{
    check (route::plannedRoute (Backend::Nvenc, false) == FrameRoute::CudaFromGl, "NVENC plans CUDA-GL frames");
    check (route::plannedRoute (Backend::Nvenc, true) == FrameRoute::CudaFromGl, "an HDR movie stays on the GPU for NVENC");
    check (route::plannedRoute (Backend::VideoToolbox, false) == FrameRoute::VideoToolboxIOSurface,
           "VideoToolbox plans IOSurface frames");
    check (route::plannedRoute (Backend::Vaapi, false) == FrameRoute::VaapiDmaBuf, "VA-API plans dma-buf frames");
    check (route::plannedRoute (Backend::Vaapi, true) == FrameRoute::VaapiDmaBuf,
           "an HDR movie goes to VA-API as P010 dma-bufs too");
    check (route::uploadRoute (Backend::Vaapi) == FrameRoute::VaapiUpload, "VA-API falls back to the upload");
    check (std::string (route::frameRouteName (FrameRoute::VaapiDmaBuf)) == "vaapi-dmabuf", "the dma-buf route has its name");
    check (route::plannedRoute (Backend::Amf, false) == FrameRoute::AmfUpload, "AMF uploads");
    check (route::plannedRoute (Backend::Software, false) == FrameRoute::CpuFrames, "software takes CPU frames");
    check (route::framesStayOnGpu (FrameRoute::CudaFromGl) && route::framesStayOnGpu (FrameRoute::VideoToolboxIOSurface)
               && route::framesStayOnGpu (FrameRoute::VaapiDmaBuf),
           "the zero-copy routes keep frames on the GPU");
    for (const auto frames : { FrameRoute::CpuFrames, FrameRoute::NvencUpload, FrameRoute::VaapiUpload,
                               FrameRoute::AmfUpload, FrameRoute::VideoToolboxUpload })
        check (! route::framesStayOnGpu (frames), std::string ("frames leave the GPU on ") + route::frameRouteName (frames));

    route::Route record;
    record.encoder = "h264_nvenc";
    record.backend = Backend::Nvenc;
    record.frames = route::plannedRoute (Backend::Nvenc, false);
    check (route::describe (record) == "h264_nvenc, frames stay on the GPU", "the GPU route reads plainly: " + route::describe (record));
    record.fallBack (route::uploadRoute (Backend::Nvenc), "CUDA-GL interop: registering the GL texture failed");
    check (record.frames == FrameRoute::NvencUpload, "a failed interop moves to the upload route");
    check (contains (route::describe (record), "read back to the CPU (CUDA-GL interop"), "and the record says why: "
           + route::describe (record));
    record.fallBack (FrameRoute::NvencUpload, "a second reason");
    check (! contains (record.fallback, "a second reason"), "falling back to the same route adds nothing");

    route::Route vaapi;
    vaapi.encoder = "h264_vaapi";
    vaapi.backend = Backend::Vaapi;
    vaapi.frames = route::plannedRoute (Backend::Vaapi, false);
    vaapi.fallBack (route::uploadRoute (Backend::Vaapi), "the export's GL context is not EGL-backed");
    check (vaapi.frames == FrameRoute::VaapiUpload && contains (route::describe (vaapi), "not EGL-backed"),
           "a VA-API export without EGL uploads and says why: " + route::describe (vaapi));

    route::clearLastRoute();
    check (! route::haveLastRoute(), "a cleared record has no route");
    route::setLastRoute (record);
    check (route::haveLastRoute() && route::lastRoute().encoder == "h264_nvenc", "the last route reads back");
}

} // namespace

int main()
{
    settingTests();
    autoOrderTests();
    explicitTests();
    routeTests();
    std::cout << (checks - failures) << "/" << checks << " export encoder route checks passed\n";
    return failures == 0 ? 0 : 1;
}
