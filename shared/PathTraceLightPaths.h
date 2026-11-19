#pragma once

// The Render node's Path Trace light paths, as in Cycles' Light Paths panel.
//
// Each kind limit counts the bounces of that kind after the camera hit:
// diffuse, glossy (mirror reflection included) and transmission, plus
// transparent crossings of alpha cut-outs, thin sheets and false interfaces.
// The kind limits share the overall bounce limit's range, so a kind at its top
// step never binds before the overall limit does. A path that reaches a limit
// ends there and keeps what it gathered; no limit refuses a frame.
//
// A clamp limits the brightest channel of one contribution: Direct for light
// seen directly or after one scatter, Indirect for light after two or more.
// 0 turns a clamp off.
//
// Filter Glossy blurs glossy lobes after a rough bounce, as Cycles' Filter
// Glossy does: the smaller the density of the bounces before, the rougher the
// lobe. 0 turns it off.
//
// The caustic switches drop glossy reflection (Reflective) or refraction
// (Refractive) after a diffuse bounce, and the light-traced caustics through a
// mirror or a refraction.
//
// The defaults are what Path Trace drew before these controls existed.

#include <cmath>
#include <cstdint>

namespace visualimportedscenerender
{
inline constexpr std::uint16_t kMaximumPathTraceKindBounces = 65535;
inline constexpr std::uint16_t kDefaultPathTraceTransparentBounces = 16;

struct PathTraceLightPaths final
{
    std::uint16_t diffuseBounces = kMaximumPathTraceKindBounces;
    std::uint16_t glossyBounces = kMaximumPathTraceKindBounces;
    std::uint16_t transmissionBounces = kMaximumPathTraceKindBounces;
    std::uint16_t transparentBounces = kDefaultPathTraceTransparentBounces;
    float directClamp = 0.0f;
    float indirectClamp = 0.0f;
    float filterGlossy = 0.0f;
    bool reflectiveCaustics = true;
    bool refractiveCaustics = true;
};

// A clamp or filter value: finite, and 0 or more with a positive sign, so one
// value has one encoding.
inline bool validLightPathAmount(float value) noexcept
{
    return std::isfinite(value) && value >= 0.0f && !std::signbit(value);
}

inline bool valid(const PathTraceLightPaths& paths) noexcept
{
    return validLightPathAmount(paths.directClamp) && validLightPathAmount(paths.indirectClamp)
        && validLightPathAmount(paths.filterGlossy);
}

inline bool sameLightPaths(const PathTraceLightPaths& left, const PathTraceLightPaths& right) noexcept
{
    return left.diffuseBounces == right.diffuseBounces && left.glossyBounces == right.glossyBounces
        && left.transmissionBounces == right.transmissionBounces
        && left.transparentBounces == right.transparentBounces
        && left.directClamp == right.directClamp && left.indirectClamp == right.indirectClamp
        && left.filterGlossy == right.filterGlossy
        && left.reflectiveCaustics == right.reflectiveCaustics
        && left.refractiveCaustics == right.refractiveCaustics;
}

inline bool defaultLightPaths(const PathTraceLightPaths& paths) noexcept
{
    return sameLightPaths(paths, PathTraceLightPaths {});
}

// The two caustic switches in one wire number: bit 0 Reflective, bit 1 Refractive.
inline constexpr std::uint32_t kLightPathReflectiveCaustics = 1u;
inline constexpr std::uint32_t kLightPathRefractiveCaustics = 2u;
inline std::uint32_t lightPathCausticFlags(const PathTraceLightPaths& paths) noexcept
{
    return (paths.reflectiveCaustics ? kLightPathReflectiveCaustics : 0u)
         | (paths.refractiveCaustics ? kLightPathRefractiveCaustics : 0u);
}
} // namespace visualimportedscenerender
