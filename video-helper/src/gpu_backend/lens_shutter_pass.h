#pragma once

// The camera lens and Render node shutter passes over a finished Raster or
// Realtime Color image, shared by the OpenGL and Metal backends. Both run
// velocity motion blur first, then depth of field over the blurred frame.

#include "../../../shared/Visual3DScene.h"
#include "../../../shared/VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace arbitgpu::lensshutter
{
// Bokeh gather inputs. lens: x blur radius in pixels per unit of |d - focus| / d,
// y focus distance, z near, w far. aperture: x blades (0 round), y rotation,
// z ratio, w kernel radius in pixels.
struct BokehParameters
{
    std::array<float, 4> lens {};
    std::array<float, 4> aperture {};
};

// The blur radius in pixels is the aperture's blur disc on the focus plane over
// the pixels that plane spans. The kernel reaches the blur of the near plane,
// never past the image diagonal.
inline BokehParameters bokehParameters (const HarmonicMIDI::grid::SceneCameraRecord& camera,
                                        std::uint32_t width, std::uint32_t height) noexcept
{
    const float tanHalf = std::tan (0.5f * camera.verticalFovRadians);
    const float pixelsPerDepth = HarmonicMIDI::grid::sceneCameraApertureRadius (camera) * float (height)
        / (2.0f * tanHalf * camera.focusDistance);
    const float nearest = std::max (camera.nearPlane, 1.0e-6f);
    const float diagonal = std::sqrt (float (width) * float (width) + float (height) * float (height));
    const float kernel = std::min (diagonal, pixelsPerDepth
        * std::max (1.0f, std::fabs (camera.focusDistance - nearest) / nearest));
    BokehParameters result;
    result.lens = { pixelsPerDepth, camera.focusDistance, camera.nearPlane, camera.farPlane };
    result.aperture = { float (camera.apertureBlades), camera.bladeRotationRadians, camera.apertureRatio, kernel };
    return result;
}

// Taps the velocity gather takes for a pixel moving motion pixels per frame over
// the shutter: one per pixel crossed, at least one, never more than the image
// diagonal. Mirrors the shaders.
inline int velocityTaps (float motionX, float motionY,
                         visualimportedscenerender::ShutterInterval shutter,
                         std::uint32_t width, std::uint32_t height) noexcept
{
    const float sweep = std::sqrt (motionX * motionX + motionY * motionY) * (shutter.close - shutter.open);
    const float diagonal = std::sqrt (float (width) * float (width) + float (height) * float (height));
    return int (std::clamp (std::ceil (sweep), 1.0f, diagonal));
}

// Which passes a Color frame runs. Inspection images and AOV outputs keep their values.
struct Passes
{
    bool velocity = false;
    bool bokeh = false;
    bool any() const noexcept { return velocity || bokeh; }
};
inline Passes colorPasses (bool colorImage, visualimportedscenerender::ShutterInterval shutter,
                           const HarmonicMIDI::grid::SceneCameraRecord& camera) noexcept
{
    if (!colorImage) return {};
    return { shutter.active(), HarmonicMIDI::grid::sceneCameraHasLens (camera) };
}

// Metal: a fullscreen copy, the velocity gather and the bokeh gather, each a
// fragment entry over the kSdfVertexShader triangle. Positions are top-left
// origin with +Y down, so the +Y up Motion attachment and the aperture's
// offsets flip Y to match OpenGL's pictures.
inline const char* metalShaderSource() noexcept
{
    return R"metal(
#include <metal_stdlib>
using namespace metal;
struct LensShutterUniforms { float4 lens; float4 aperture; float4 shutter; };
fragment float4 _copy(float4 position [[position]], texture2d<float> color [[texture(0)]]) {
    return color.read(uint2(position.xy));
}
fragment float4 _velocity(float4 position [[position]],
                          texture2d<float> color [[texture(0)]],
                          texture2d<float> motionTexture [[texture(1)]],
                          constant LensShutterUniforms& u [[buffer(0)]]) {
    const float2 size = float2(color.get_width(), color.get_height());
    const float2 raw = motionTexture.read(uint2(position.xy)).rg;
    const float2 motion = float2(raw.x, -raw.y);
    const float sweep = length(motion) * (u.shutter.y - u.shutter.x);
    const int taps = int(clamp(ceil(sweep), 1.0f, length(size)));
    float4 sum = float4(0.0f);
    for (int i = 0; i < taps; ++i) {
        const float t = mix(u.shutter.x, u.shutter.y, (float(i) + 0.5f) / float(taps));
        const float2 at = clamp(position.xy - motion * t, float2(0.5f), size - float2(0.5f));
        sum += color.read(uint2(at));
    }
    return sum / float(taps);
}
float lensViewDepth(texture2d<float> depth, float2 pixel, constant LensShutterUniforms& u) {
    const float stored = depth.read(uint2(pixel)).r;
    return u.lens.z + clamp(stored, 0.0f, 1.0f) * (u.lens.w - u.lens.z);
}
float lensBlurRadius(float d, constant LensShutterUniforms& u) { return u.lens.x * abs(d - u.lens.y) / max(d, 1e-6f); }
bool lensInsideAperture(float2 offset, float radius, constant LensShutterUniforms& u) {
    if (radius <= 0.5f) return dot(offset, offset) <= 0.25f;
    const float c = cos(-u.aperture.y), s = sin(-u.aperture.y);
    float2 local = float2(c * offset.x - s * offset.y, s * offset.x + c * offset.y);
    local.x /= u.aperture.z;
    const float r = length(local) / radius;
    if (u.aperture.x < 3.0f) return r <= 1.0f;
    const float wedge = 6.28318531f / u.aperture.x;
    const float a = atan2(local.y, local.x);
    const float angle = (a - wedge * floor(a / wedge)) - 0.5f * wedge;
    return r * cos(angle) <= cos(0.5f * wedge);
}
float lensShapeArea(float radius, constant LensShutterUniforms& u) {
    const float r = max(radius, 0.5f);
    const float n = u.aperture.x;
    return u.aperture.z * (n < 3.0f ? 3.14159265f * r * r : 0.5f * n * r * r * sin(6.28318531f / n));
}
fragment float4 _bokeh(float4 position [[position]],
                       texture2d<float> color [[texture(0)]],
                       texture2d<float> depth [[texture(1)]],
                       constant LensShutterUniforms& u [[buffer(0)]]) {
    const float2 pixel = position.xy;
    const float2 size = float2(color.get_width(), color.get_height());
    const float centreDepth = lensViewDepth(depth, pixel, u);
    const float centreRadius = lensBlurRadius(centreDepth, u);
    float4 sum = color.read(uint2(pixel)) / lensShapeArea(centreRadius, u);
    float weight = 1.0f / lensShapeArea(centreRadius, u);
    const float2 upPixel = float2(pixel.x, size.y - pixel.y);
    const float spin = 6.28318531f * fract(52.9829189f * fract(dot(upPixel, float2(0.06711056f, 0.00583715f))));
    const float kernel = u.aperture.w * max(u.aperture.z, 1.0f);
    for (int i = 0; i < 64; ++i) {
        const float angle = spin + float(i) * 2.39996323f;
        const float2 offset = float2(cos(angle), sin(angle)) * kernel * sqrt((float(i) + 0.5f) / 64.0f);
        const float2 tap = clamp(pixel + float2(offset.x, -offset.y), float2(0.5f), size - float2(0.5f));
        const float tapDepth = lensViewDepth(depth, tap, u);
        float radius = lensBlurRadius(tapDepth, u);
        if (tapDepth > centreDepth) radius = min(radius, centreRadius);
        if (!lensInsideAperture(-offset, radius, u)) continue;
        const float w = 1.0f / lensShapeArea(radius, u);
        sum += color.read(uint2(tap)) * w;
        weight += w;
    }
    return sum / weight;
}
)metal";
}
} // namespace arbitgpu::lensshutter
