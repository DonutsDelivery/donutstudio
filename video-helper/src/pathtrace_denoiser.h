#pragma once

// Which denoiser a Path Trace frame gets, what it reads and how the run is
// reported (Plan 3, Denoising). The Render node chooses Off, Auto, OptiX or
// OIDN. Auto takes the OptiX AI denoiser when the frame was traced on the
// OptiX device and OIDN otherwise. A choice that cannot run falls back to the
// next denoiser that can, and the native filter (pathtrace_preview_denoise.h)
// only when neither library is there. Every fallback carries a reason that the
// viewport header shows next to the denoiser's name, so none is silent.
//
// The AI denoisers read three float images of one layout, bottom row first like
// the engine's colour bytes: the noisy colour, the first-hit albedo and the
// world-space normal (EngineImage::albedo and normal). The temporal OptiX model
// also reads the flow, two floats per pixel: where each pixel's surface was on
// the previous frame, as the current position minus the previous one.

#include "pathtrace_denoiser_choice.h"
#include "pathtrace_preview_denoise.h"
#include "pathtrace/scene_ingest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace videorender::fixture3d::pathtracedenoise
{

// The device OIDN runs on.
enum class OidnDevice : std::uint8_t
{
    Cpu,
    Cuda,
    Hip,
    Sycl
};

// OIDN goes to the GPU the frame was traced on: CUDA for the OptiX and CUDA
// devices, HIP for HIP, SYCL for oneAPI. Every other device, the CPU included,
// gets CPU OIDN.
inline OidnDevice oidnDeviceFor (const char* renderDevice) noexcept
{
    if (renderDevice == nullptr)
        return OidnDevice::Cpu;
    if (std::strcmp (renderDevice, "OptiX") == 0 || std::strcmp (renderDevice, "CUDA") == 0)
        return OidnDevice::Cuda;
    if (std::strcmp (renderDevice, "HIP") == 0)
        return OidnDevice::Hip;
    if (std::strcmp (renderDevice, "oneAPI") == 0)
        return OidnDevice::Sycl;
    return OidnDevice::Cpu;
}

constexpr const char* oidnDeviceName (OidnDevice device) noexcept
{
    switch (device)
    {
        case OidnDevice::Cuda: return "CUDA";
        case OidnDevice::Hip: return "HIP";
        case OidnDevice::Sycl: return "SYCL";
        case OidnDevice::Cpu: break;
    }
    return "CPU";
}

// What can run for this frame. An empty problem means that denoiser can run.
struct DenoiserAvailability
{
    bool tracedOnOptix = false;
    std::string optixProblem;
    std::string oidnProblem;
};

struct DenoiserPlan
{
    Denoiser denoiser = Denoiser::None;
    // Why the choice did not get the denoiser it names, or Auto not its first
    // pick. Empty when it did.
    std::string reason;
};

inline DenoiserPlan planDenoiser (DenoiserChoice choice, const DenoiserAvailability& available)
{
    if (choice == DenoiserChoice::Off)
        return {};
    const bool optix = available.optixProblem.empty();
    const bool oidn = available.oidnProblem.empty();
    const std::string optixWhy = "OptiX AI did not run: " + available.optixProblem;
    const std::string oidnWhy = "OIDN did not run: " + available.oidnProblem;
    const auto withoutOptix = [&] (std::string why) -> DenoiserPlan {
        if (oidn)
            return { Denoiser::Oidn, std::move (why) };
        return { Denoiser::NativeFilter, why.empty() ? oidnWhy : why + "; " + oidnWhy };
    };
    switch (choice)
    {
        case DenoiserChoice::Auto:
            if (available.tracedOnOptix && optix)
                return { Denoiser::OptixAi, {} };
            return withoutOptix (available.tracedOnOptix ? optixWhy : std::string());
        case DenoiserChoice::OptiX:
            if (optix)
                return { Denoiser::OptixAi, {} };
            return withoutOptix (optixWhy);
        case DenoiserChoice::Oidn:
            if (oidn)
                return { Denoiser::Oidn, {} };
            return { Denoiser::NativeFilter, oidnWhy };
        case DenoiserChoice::Off:
            break;
    }
    return {};
}

// The name telemetry and the viewport header give the denoiser that ran.
constexpr const char* denoiserLabel (Denoiser denoiser, OidnDevice oidnDevice, bool temporal) noexcept
{
    switch (denoiser)
    {
        case Denoiser::OptixAi: return temporal ? "optix-ai-temporal" : "optix-ai";
        case Denoiser::Oidn:
            switch (oidnDevice)
            {
                case OidnDevice::Cuda: return "oidn-cuda";
                case OidnDevice::Hip: return "oidn-hip";
                case OidnDevice::Sycl: return "oidn-sycl";
                case OidnDevice::Cpu: break;
            }
            return "oidn-cpu";
        case Denoiser::NativeFilter: return "native-filter";
        case Denoiser::None: break;
    }
    return "none";
}

// Three floats per pixel in every guide and colour image the AI denoisers read.
inline constexpr std::size_t kGuideChannels = 3;
// Two floats per pixel in the flow.
inline constexpr std::size_t kFlowChannels = 2;

// Where pixel (x, row) of a guide starts, rows counted from the bottom.
constexpr std::size_t guideOffset (std::uint32_t x, std::uint32_t row, std::uint32_t width) noexcept
{
    return (static_cast<std::size_t> (row) * width + x) * kGuideChannels;
}

// The temporal model's flow from camera reprojection. depth is the engine's,
// (view depth - near) / (far - near), bottom row first. Each pixel's surface is
// put back in the world through the current camera and projected through the
// previous one. A point behind the previous camera gets no flow. Both cameras
// must have the image's size. False when the sizes do not match.
inline bool reprojectionFlow (const videohelper::pathtrace::PinholeCamera& current,
                              const videohelper::pathtrace::PinholeCamera& previous,
                              const std::vector<float>& depth, std::vector<float>& flow)
{
    const std::uint32_t width = current.width, height = current.height;
    const std::size_t pixels = static_cast<std::size_t> (width) * height;
    if (pixels == 0 || depth.size() != pixels || previous.width != width || previous.height != height
        || ! (current.nearPlane > 0.0f))
        return false;
    flow.assign (pixels * kFlowChannels, 0.0f);
    for (std::uint32_t row = 0; row < height; ++row)
    {
        const std::uint32_t top = height - 1u - row;
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::size_t index = static_cast<std::size_t> (row) * width + x;
            const auto ray = current.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (top) + 0.5f);
            // tMin is near times the length of the ray's view-space direction at depth 1.
            const float viewDepth = current.nearPlane + depth[index] * (current.farPlane - current.nearPlane);
            const float distance = viewDepth * ray.tMin / current.nearPlane;
            const auto world = ray.origin + ray.direction * distance;
            float previousX = 0.0f, previousTop = 0.0f, previousDepth = 0.0f;
            if (! previous.project (world, previousX, previousTop, previousDepth))
                continue;
            flow[index * kFlowChannels] = (static_cast<float> (x) + 0.5f) - previousX;
            flow[index * kFlowChannels + 1] = (static_cast<float> (row) + 0.5f) - (static_cast<float> (height) - previousTop);
        }
    }
    return true;
}

// Tight RGBA16F or RGBA8 rows as three floats per pixel, alpha dropped. False
// when the byte count does not match the pixel count.
inline bool colorToFloat3 (const std::vector<std::uint8_t>& bytes, bool linear, std::size_t pixels, std::vector<float>& rgb)
{
    if (bytes.size() != pixels * (linear ? 8u : 4u))
        return false;
    rgb.resize (pixels * kGuideChannels);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        for (std::size_t channel = 0; channel < kGuideChannels; ++channel)
        {
            if (linear)
            {
                std::uint16_t bits = 0;
                std::memcpy (&bits, bytes.data() + (pixel * 4 + channel) * 2, sizeof (bits));
                rgb[pixel * kGuideChannels + channel] = halfToFloat (bits);
            }
            else
                rgb[pixel * kGuideChannels + channel] = static_cast<float> (bytes[pixel * 4 + channel]) / 255.0f;
        }
    return true;
}

// The denoised floats back into the source's format, alpha copied from the
// source. A negative or non-finite value from a network becomes 0, since the
// view transform would turn a negative into light.
inline void float3ToColor (const std::vector<float>& rgb, bool linear, const std::vector<std::uint8_t>& source,
                           std::vector<std::uint8_t>& output)
{
    output = source;
    const std::size_t pixels = rgb.size() / kGuideChannels;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        for (std::size_t channel = 0; channel < kGuideChannels; ++channel)
        {
            float value = rgb[pixel * kGuideChannels + channel];
            if (! (value > 0.0f))
                value = 0.0f;
            if (linear)
            {
                const std::uint16_t bits = videohelper::pathtrace::halfBits (value);
                std::memcpy (output.data() + (pixel * 4 + channel) * 2, &bits, sizeof (bits));
            }
            else
                output[pixel * 4 + channel] = static_cast<std::uint8_t> (std::min (1.0f, value) * 255.0f + 0.5f);
        }
}

// One image for an AI denoiser. Every pointer is width * height pixels.
struct DenoiseFrame
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    const float* color = nullptr;   // three floats per pixel
    const float* albedo = nullptr;  // three floats per pixel
    const float* normal = nullptr;  // three floats per pixel
    float* output = nullptr;        // three floats per pixel
    bool hdr = true;                // linear radiance; false for display bytes
    // The OptiX temporal model: flow is two floats per pixel, and
    // usePrevious says the previous frame's output belongs to this sequence.
    bool temporal = false;
    bool usePrevious = false;
    const float* flow = nullptr;
};

} // namespace videorender::fixture3d::pathtracedenoise
