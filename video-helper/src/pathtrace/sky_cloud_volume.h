#pragma once

// Volumetric clouds of the analytic sky, as Path Trace (CPU and OptiX) and the
// raster's C++ twin evaluate them. The GL/Metal recipe in
// gpu_backend/fixture_analytic_sky.h (analyticSkyCloudVolume) runs the same
// arithmetic.
//
// The layer sits where the flat cloud layer does, one unit up, and is
// thickness units deep, with the flat layer's coverage, type, heading and drift.
// Eight steps march the view across it, and two more march toward the sun at
// each for self shadowing. Every step scatters the sun's irradiance through a
// two-lobe Henyey-Greenstein phase plus the ambient sky, at albedo 1. The view
// transmittance dims the sky and the sun behind the layer, so in Path Trace the
// sampled sun is shadowed by the clouds it passes through.

#include "devices/device_annotations.h"

#include <algorithm>
#include <cmath>

namespace videohelper::pathtrace::cloudvolume
{

struct Layer
{
    float coverage = 0.0f;         // 0 draws nothing
    float kind = 0.0f;             // 0 puffy, 1 streaky
    float headingRadians = 0.0f;
    float drift = 0.0f;
    float thickness = 0.0f;        // 0 keeps the flat layer
};

struct Result
{
    float r = 0.0f, g = 0.0f, b = 0.0f;  // radiance scattered toward the eye
    float transmittance = 1.0f;          // of the sky and sun behind
};

PT_SHARED_FN float fract (float x) noexcept { return x - std::floor (x); }
PT_SHARED_FN float mixf (float a, float b, float t) noexcept { return a + (b - a) * t; }
PT_SHARED_FN float smooth (float low, float high, float x) noexcept
{
    const float t = std::min (1.0f, std::max (0.0f, (x - low) / (high - low)));
    return t * t * (3.0f - 2.0f * t);
}

// The flat layer's hash, so both layers share their lattice.
PT_SHARED_FN float hash (float x, float y) noexcept
{
    float a = fract (x * 0.1031f), b = fract (y * 0.1031f), c = a;
    const float shift = a * (b + 33.33f) + b * (c + 33.33f) + c * (a + 33.33f);
    a += shift;
    b += shift;
    c += shift;
    return fract ((a + b) * c);
}

PT_SHARED_FN float noise (float x, float y, float z) noexcept
{
    const float cx = std::floor (x), cy = std::floor (y), cz = std::floor (z);
    float fx = x - cx, fy = y - cy, fz = z - cz;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    fz = fz * fz * (3.0f - 2.0f * fz);
    const float lx = cx + cy * 57.0f, lz = cz + cy * 113.0f;
    const float ux = lx + 57.0f, uz = lz + 113.0f;
    const float a = mixf (mixf (hash (lx, lz), hash (lx + 1.0f, lz), fx),
                          mixf (hash (lx, lz + 1.0f), hash (lx + 1.0f, lz + 1.0f), fx), fz);
    const float b = mixf (mixf (hash (ux, uz), hash (ux + 1.0f, uz), fx),
                          mixf (hash (ux, uz + 1.0f), hash (ux + 1.0f, uz + 1.0f), fx), fz);
    return mixf (a, b, fy);
}

// Density in [0, 1] at layer coordinates q (along, across the heading) and
// height fraction h.
PT_SHARED_FN float density (const Layer& layer, float qx, float qy, float h) noexcept
{
    const float px = qx * mixf (1.6f, 0.35f, layer.kind), py = h * layer.thickness * 2.0f,
                pz = qy * mixf (1.6f, 2.8f, layer.kind);
    const float value = 0.62f * noise (px, py, pz)
        + 0.38f * noise (px * 2.03f + 17.0f, py * 2.03f + 5.0f, pz * 2.03f + 9.0f);
    const float edge = mixf (0.12f, 0.3f, layer.kind);
    const float threshold = mixf (0.85f, -edge, layer.coverage);
    const float profile = smooth (0.0f, mixf (0.25f, 0.45f, layer.kind), h)
        * (1.0f - smooth (mixf (0.55f, 0.75f, layer.kind), 1.0f, h));
    return smooth (threshold, threshold + edge, value) * profile;
}

PT_SHARED_FN float phase (float mu, float g) noexcept
{
    const float t = std::max (1.0f + g * g - 2.0f * g * mu, 1.0e-6f);
    return (1.0f - g * g) / (12.566370614359172f * t * std::sqrt (t));
}

// d is a unit direction and sun the unit sun direction. sunIrradiance is the
// sun's irradiance (the cap's pi*(1-cos radius)*peak) and white the ambient sky.
PT_SHARED_FN Result march (const Layer& layer, float dx, float dy, float dz, float sx, float sy, float sz,
                           float sunR, float sunG, float sunB, float white) noexcept
{
    Result result;
    if (! (layer.thickness > 0.0f) || ! (layer.coverage > 0.0f) || ! (dy > 0.0f))
        return result;
    const float alongX = std::sin (layer.headingRadians), alongZ = std::cos (layer.headingRadians);
    const float fade = smooth (0.015f, 0.14f, dy);
    const float sigma = 6.0f / layer.thickness;
    const float stride = layer.thickness / (dy + 0.1f) / 8.0f;
    const float baseX = dx / (dy + 0.1f), baseZ = dz / (dy + 0.1f);
    const float sunAlong = sx * alongX + sz * alongZ, sunAcross = sx * alongZ - sz * alongX;
    const float sunUp = std::max (sy, 0.05f);
    const float mu = dx * sx + dy * sy + dz * sz;
    const float lobe = 0.7f * phase (mu, 0.6f) + 0.3f * phase (mu, -0.2f);
    const float lit = smooth (-0.05f, 0.05f, sy);
    float transmittance = 1.0f;
    for (int i = 0; i < 8; ++i)
    {
        const float h = (static_cast<float> (i) + 0.5f) / 8.0f;
        const float grow = 1.0f + layer.thickness * h;
        const float px = baseX * grow, pz = baseZ * grow;
        const float qx = px * alongX + pz * alongZ - layer.drift, qy = px * alongZ - pz * alongX;
        const float here = density (layer, qx, qy, h);
        if (! (here > 0.0f))
            continue;
        const float reach = std::min ((1.0f - h) * layer.thickness / sunUp, 4.0f * layer.thickness);
        float shade = 0.0f;
        for (int j = 0; j < 2; ++j)
        {
            const float s = (static_cast<float> (j) + 0.5f) * 0.5f * reach;
            const float hs = std::min (h + s * sunUp / layer.thickness, 1.0f);
            shade += density (layer, qx + sunAlong * s, qy + sunAcross * s, hs);
        }
        const float sunShadow = std::exp (-sigma * shade * 0.5f * reach);
        const float absorbed = 1.0f - std::exp (-sigma * here * stride);
        const float direct = transmittance * absorbed * sunShadow * lobe * lit;
        const float ambient = transmittance * absorbed * white * (0.5f + 0.5f * h);
        result.r += direct * sunR + ambient;
        result.g += direct * sunG + ambient;
        result.b += direct * sunB + ambient;
        transmittance *= 1.0f - absorbed;
    }
    result.r *= fade;
    result.g *= fade;
    result.b *= fade;
    result.transmittance = mixf (1.0f, transmittance, fade);
    return result;
}

} // namespace videohelper::pathtrace::cloudvolume
