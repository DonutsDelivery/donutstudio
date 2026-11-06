#pragma once

// The Render node's Path Trace denoiser choice as the helper carries it from the
// request to the engine (Plan 3, Denoising). The values are the wire's
// (visualimportedscenerender::PathTraceDenoiser).

#include <cstdint>

namespace videorender::fixture3d::pathtracedenoise
{

enum class DenoiserChoice : std::uint8_t
{
    Off = 0,
    Auto = 1,
    OptiX = 2,
    Oidn = 3
};

constexpr const char* choiceName (DenoiserChoice choice) noexcept
{
    switch (choice)
    {
        case DenoiserChoice::Auto: return "auto";
        case DenoiserChoice::OptiX: return "optix";
        case DenoiserChoice::Oidn: return "oidn";
        case DenoiserChoice::Off: break;
    }
    return "off";
}

} // namespace videorender::fixture3d::pathtracedenoise
