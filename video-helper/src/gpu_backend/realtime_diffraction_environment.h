#pragma once

// The Realtime engine lights a diffraction grating by the scene's environment through
// its orders. Path Trace finds that light by sampling. Realtime turns the grating
// equation around instead: for the pixel's view direction o and each propagating order
// m of one band, the light that order sends toward the camera arrives from the single
// direction whose tangent projection is m * wavelength / period * g - o.xy. The pixel
// reads the sky, HDR image or flat environment colour there and weights it by that
// order's power efficiency. Different wavelengths look in different directions, so a
// sky with any structure (a sun, a bright horizon, an image) splits into spectral
// streaks that follow the groove direction and pitch.
//
// The weights are the ones the raster transport's diffractionTriangleBsdf uses for a
// light: lattice efficiency, the roughness height loss, and cos(o)/cos(i). A single
// order's power efficiency cannot pass one, so each is bounded there. That is a
// physical bound on one term, not a budget.
//
// An RGB sky radiance becomes a band value by that band's own linear sRGB response,
// normalised so a white sky reads one in every band. The bands then go back to RGB
// through the same CIE weights as the rest of the grating's light.
//
// This header is the host form of the shader arithmetic in backend_opengl.cpp
// (realtimeDiffractionSky) and backend_sokol.mm (the Metal Realtime text). Tests use
// it. Raster and Path Trace never call it.

#include "../diffraction_order_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>

namespace arbitgpu::realtimediffraction
{
struct Grating
{
    diffractionmaterial::GrooveProfile profile = diffractionmaterial::GrooveProfile::Sinusoidal;
    float duty = 0.5f;
    float depthNanometres = 0.0f;
    float periodNanometres = 0.0f;
    // Unit groove reciprocal direction in the surface's tangent plane.
    std::array<float, 2> direction { 1.0f, 0.0f };
    int firstOrder = 1;
    int lastOrder = 1;
    float roughnessHeightNanometres = 0.0f;
};

// The share of an RGB radiance one band carries. bandXyz is the band's CIE response.
inline float bandShare (const std::array<float, 3>& bandXyz, const std::array<float, 3>& rgb) noexcept
{
    const auto& c = bandXyz;
    const std::array<float, 3> response {
        std::max (0.0f, 3.2406f * c[0] - 1.5372f * c[1] - 0.4986f * c[2]),
        std::max (0.0f, -0.9689f * c[0] + 1.8758f * c[1] + 0.0415f * c[2]),
        std::max (0.0f, 0.0557f * c[0] - 0.2040f * c[1] + 1.0570f * c[2]) };
    const float total = response[0] + response[1] + response[2];
    if (total <= 0.0f)
        return 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
    return (rgb[0] * response[0] + rgb[1] * response[1] + rgb[2] * response[2]) / total;
}

// The local incident direction that order m sends toward outgoing, if it propagates.
inline bool orderIncident (const Grating& grating, float wavelengthNanometres, int order,
                           const std::array<float, 3>& outgoing, std::array<float, 3>& incident) noexcept
{
    if (grating.periodNanometres <= 0.0f || wavelengthNanometres <= 0.0f || outgoing[2] <= 0.0f)
        return false;
    const float step = static_cast<float> (order) * wavelengthNanometres / grating.periodNanometres;
    const float x = step * grating.direction[0] - outgoing[0];
    const float y = step * grating.direction[1] - outgoing[1];
    const float square = x * x + y * y;
    if (square >= 1.0f)
        return false;
    const float z = std::sqrt (1.0f - square);
    if (z <= 1.0e-6f)
        return false;
    incident = { x, y, z };
    return true;
}

// The bounded power efficiency of order m from incident toward outgoing.
inline float orderWeight (const Grating& grating, float wavelengthNanometres, int order,
                          const std::array<float, 3>& incident, const std::array<float, 3>& outgoing) noexcept
{
    const float cosines = incident[2] + outgoing[2];
    const float phase = 2.0f * diffractionmaterial::ordercore::pi * grating.depthNanometres * cosines / wavelengthNanometres;
    const float lattice = diffractionmaterial::ordercore::latticeEfficiency (grating.profile, order, 0, false, false,
                                                                            grating.duty, phase);
    const float height = 2.0f * diffractionmaterial::ordercore::pi * grating.roughnessHeightNanometres * cosines
        / wavelengthNanometres;
    const float weight = lattice * std::exp (-height * height) * outgoing[2] / incident[2];
    return std::isfinite (weight) ? std::clamp (weight, 0.0f, 1.0f) : 0.0f;
}

// One band's radiance toward outgoing (local frame, z along the normal). sky takes a
// local incident direction and returns that band's radiance from it.
template <class Sky>
float bandRadiance (const Grating& grating, float wavelengthNanometres, const std::array<float, 3>& outgoing, Sky&& sky)
{
    float radiance = 0.0f;
    for (int order = -grating.lastOrder; order <= grating.lastOrder; ++order)
    {
        if (order != 0 && std::abs (order) < grating.firstOrder)
            continue;
        std::array<float, 3> incident {};
        if (! orderIncident (grating, wavelengthNanometres, order, outgoing, incident))
            continue;
        radiance += orderWeight (grating, wavelengthNanometres, order, incident, outgoing) * sky (incident);
    }
    return radiance;
}
} // namespace arbitgpu::realtimediffraction
