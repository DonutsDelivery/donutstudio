#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace basesurfacereflection
{
using Color = std::array<double, 3>;
using Direction = std::array<double, 3>;
inline constexpr double kPi = 3.14159265358979323846;
// Finite-width approximation to a delta lobe at authored roughness zero.
// Alpha is perceptual roughness squared, floored at 0.0025 in all backends.
inline constexpr double kMinimumAlpha = 0.0025;
inline constexpr double kDirectionLengthSquaredFloor = 1.0e-12;
inline double dot(Direction a, Direction b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Direction unit(Direction value) noexcept
{
    const auto lengthSquared = dot(value, value);
    if (!(lengthSquared > kDirectionLengthSquaredFloor) || !std::isfinite(lengthSquared))
        return {};
    const auto scale = 1.0 / std::sqrt(lengthSquared);
    return {value[0] * scale, value[1] * scale, value[2] * scale};
}
inline double dielectricF0(double ior, bool explicitIor = true) noexcept
{
    const auto ratio = (ior - 1.0) / (ior + 1.0);
    return explicitIor ? ratio * ratio : 0.04;
}
// Direct base response per unit incident radiance, including N.L. Inputs are
// admitted linear material values, with directions pointing away from surface.
// GGX/Trowbridge-Reitz D, separable Smith G1(L)G1(V), Schlick F, and Lambert
// diffuse weighted by (1-F)(1-metallic). See PBRT 4e, Roughness Using Microfacet
// Theory: https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
// Cancel G's cosine factors analytically rather than adding a grazing epsilon.
// Zero directions and nonpositive N.L/N.V return zero; no radiance clamp.
// Ambient, coat, thin alpha transmission and occlusion remain separate policies.
inline Color direct(Color base, double metallic, double roughness,
                    double dielectric, double transmission,
                    Direction normal, Direction light, Direction view) noexcept
{
    normal = unit(normal); light = unit(light); view = unit(view);
    const auto nl = std::clamp(dot(normal, light), 0.0, 1.0);
    const auto nv = std::clamp(dot(normal, view), 0.0, 1.0);
    if (!(nl > 0.0 && nv > 0.0)) return {};
    const auto half = unit({light[0] + view[0], light[1] + view[1], light[2] + view[2]});
    const auto nh = std::clamp(dot(normal, half), 0.0, 1.0);
    const auto vh = std::clamp(dot(view, half), 0.0, 1.0);
    const auto alpha = std::max(roughness * roughness, kMinimumAlpha);
    const auto a2 = alpha * alpha;
    const auto denominator = (1.0 - nh * nh) + a2 * nh * nh;
    const auto distribution = a2 / (kPi * denominator * denominator);
    const auto visibility = 1.0 / ((nl + std::sqrt(a2 + (1.0 - a2) * nl * nl))
                                * (nv + std::sqrt(a2 + (1.0 - a2) * nv * nv)));
    const auto schlick = std::pow(1.0 - vh, 5.0);
    Color result {};
    for (std::size_t channel = 0; channel < 3; ++channel)
    {
        const auto f0 = dielectric * (1.0 - metallic) + base[channel] * metallic;
        const auto fresnel = f0 + (1.0 - f0) * schlick;
        const auto diffuse = (1.0 - fresnel) * (1.0 - metallic) * base[channel]
                           * (1.0 - transmission) / kPi;
        result[channel] = (distribution * visibility * fresnel + diffuse) * nl;
    }
    return result;
}
} // namespace basesurfacereflection
