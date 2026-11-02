#pragma once

// The denoise step of the Path Trace engine (Plan 3, Denoising). The engine runs
// it on the resolved image when the Render node's "denoise" switch is on.
//
// The only denoiser in this build is "native-filter": the kernel of the native
// Denoise effect (EffectType::Denoise, mode 4 of kProductionFilterFragment in
// renderer.cpp) at that effect's default strength and radius from effect_defs.h.
// It runs here on the CPU because the effect pass cannot take these frames. A
// linear Scene3D layer refuses every effect (FrameRenderer::validate in
// renderer.h), and the OpenGL renderer compiles the production filter without
// drawing it. It is an edge-preserving blur, not an AI denoiser, and it takes
// no albedo or normal guides and keeps no temporal history.
//
// Linear radiance has no upper end, so the shader's range term cannot compare it
// directly. A spark of 300 on a face of 2 is outside any range sigma, so the
// kernel kept it whole and blended half of it back. On linear images the range
// term therefore reads display-like values x / (1 + x) and measures them from
// the 5 x 5 median, not from the centre pixel, and a centre far from that median
// is replaced in full. A straight edge keeps its side, because a pixel's median
// lies on the side holding most of the window.
//
// That kernel alone loses light. When a face's light arrives as sparse hits, the
// median sits below their mean and the hits are replaced by it, so the face darkens
// with fewer samples: a disc's sector read half its true radiance at 256 samples. On
// linear images each pixel's input therefore goes to the pixels around it, shared
// among those whose filtered values are close to its own. The filtered image has no
// sparks left, so the sharing stays on its side of an edge, and each pixel's shares
// add to 1, so the image keeps its energy exactly and adds no negatives of its own,
// which the view transform would clamp and turn into light (decodeTransfer in
// color_transform_gl.cpp).
//
// Shares must also add to 1 where a pixel gathers them, or even light gets a pattern.
// A pixel beside a hard edge has fewer close neighbours than one further in, so it gives
// each of them a larger share and gathers smaller ones: an 8 beside an edge read 6.75 with
// 8.72 two pixels on. A scale per pixel, balanced in kEnergyShareBalancePasses passes, makes
// sent and gathered shares agree, and steps past the image edge reflect so the border is no
// different. Sent shares stay exact, so the small gathered error left is the only one.
//
// Two kinds of pixel keep their input and share nothing. A channel at the largest
// half has lost its value, so that pixel is not light to pass on: sharing it wrote
// +/-65504 across its 5 x 5, which the view transform turned into white blocks. A lone
// spark, with nothing within kLoneSparkRatio of it in its 5 x 5, has no neighbour its
// light belongs with, so it stays one pixel of the same energy. Clusters still share.

// OptiX AI and OIDN have no implementation yet. Each one's compile definition
// below goes on only when its code lands: ARBIT_DENOISE_OPTIX_AI from the
// ARBIT_WITH_OPTIX branch of video-helper/CMakeLists.txt, and ARBIT_DENOISE_OIDN
// from a future OIDN dependency. See
// artifacts/planning/missing-features/evidence/video/plan3-pathtrace-denoise-20261030.md.

#include "pathtrace/engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace videorender::fixture3d::pathtracedenoise
{

enum class Denoiser
{
    None,
    NativeFilter,
    OptixAi,
    Oidn
};

// The names telemetry reports. Only a denoiser that ran may be named.
constexpr const char* denoiserName (Denoiser denoiser) noexcept
{
    switch (denoiser)
    {
        case Denoiser::NativeFilter: return "native-filter";
        case Denoiser::OptixAi: return "optix-ai";
        case Denoiser::Oidn: return "oidn";
        case Denoiser::None: break;
    }
    return "none";
}

// The plan's order: OptiX AI, then OIDN, then what every build has.
constexpr Denoiser selectDenoiser (bool optixAiLinked, bool oidnLinked) noexcept
{
    return optixAiLinked ? Denoiser::OptixAi : oidnLinked ? Denoiser::Oidn : Denoiser::NativeFilter;
}

#ifdef ARBIT_DENOISE_OPTIX_AI
inline constexpr bool kOptixAiLinked = true;
#else
inline constexpr bool kOptixAiLinked = false;
#endif
#ifdef ARBIT_DENOISE_OIDN
inline constexpr bool kOidnLinked = true;
#else
inline constexpr bool kOidnLinked = false;
#endif
inline constexpr Denoiser kBuildDenoiser = selectDenoiser (kOptixAiLinked, kOidnLinked);

// The Denoise effect's defaults in effect_defs.h.
inline constexpr float kNativeFilterStrength = 0.5f;
inline constexpr int kNativeFilterRadius = 2;
// Balancing passes behind the energy shares. Four leave a hard edge within 0.5% of its side.
inline constexpr int kEnergyShareBalancePasses = 4;
// A pixel is a lone spark when every other pixel in its 5 x 5 is below 1/8 of it. On the
// 16-sample sun 20 film, local peaks above 1 are under 1.25 times their neighbours or above
// 64 times, and peaks from 0.25 to 1 have none between 3 and 8 times.
inline constexpr float kLoneSparkRatio = 8.0f;
inline constexpr float kLargestHalf = 65504.0f;

inline float halfToFloat (std::uint16_t bits) noexcept
{
    const int exponent = (bits >> 10) & 0x1f;
    const int fraction = bits & 0x3ff;
    // SceneEngine writes no NaN or infinity; halfBits maps NaN to 0 and
    // infinity to the largest finite half. Read them back the same way.
    const float magnitude = exponent == 0 ? std::ldexp (static_cast<float> (fraction), -24)
        : exponent == 31 ? (fraction != 0 ? 0.0f : 65504.0f)
        : std::ldexp (static_cast<float> (1024 + fraction), exponent - 25);
    return (bits & 0x8000u) != 0 ? -magnitude : magnitude;
}

// Expands tight RGBA8 or RGBA16F rows to four floats per pixel. False when the
// byte count does not match the dimensions.
inline bool loadPixels (const std::vector<std::uint8_t>& color, bool linear, std::uint32_t width,
                        std::uint32_t height, std::vector<float>& pixels)
{
    const std::size_t count = static_cast<std::size_t> (width) * height;
    if (count == 0 || color.size() != count * (linear ? 8u : 4u))
        return false;
    pixels.resize (count * 4);
    if (linear)
    {
        for (std::size_t index = 0; index < count * 4; ++index)
        {
            std::uint16_t bits = 0;
            std::memcpy (&bits, color.data() + index * 2, sizeof (bits));
            pixels[index] = halfToFloat (bits);
        }
    }
    else
    {
        for (std::size_t index = 0; index < count * 4; ++index)
            pixels[index] = static_cast<float> (color[index]) / 255.0f;
    }
    return true;
}

inline float channelMagnitude (const float* pixel) noexcept
{
    return std::max ({ std::abs (pixel[0]), std::abs (pixel[1]), std::abs (pixel[2]) });
}

// A channel at the largest half has lost its value (halfBits).
inline bool atHalfLimit (const float* pixel) noexcept
{
    return channelMagnitude (pixel) >= kLargestHalf;
}

inline bool loneSpark (const float* pixels, std::uint32_t width, std::uint32_t height, std::uint32_t x, std::uint32_t y) noexcept
{
    const float magnitude = channelMagnitude (pixels + (static_cast<std::size_t> (y) * width + x) * 4);
    if (magnitude <= 0.0f)
        return false;
    for (int dy = -kNativeFilterRadius; dy <= kNativeFilterRadius; ++dy)
        for (int dx = -kNativeFilterRadius; dx <= kNativeFilterRadius; ++dx)
        {
            const int otherX = static_cast<int> (x) + dx;
            const int otherY = static_cast<int> (y) + dy;
            if ((dx == 0 && dy == 0) || otherX < 0 || otherY < 0 || otherX >= static_cast<int> (width) || otherY >= static_cast<int> (height))
                continue;
            if (channelMagnitude (pixels + (static_cast<std::size_t> (otherY) * width + static_cast<std::size_t> (otherX)) * 4) * kLoneSparkRatio >= magnitude)
                return false;
        }
    return true;
}

// The shader's kernel on the half-open rectangle, written into a tight output
// image of the input's format. Rows and columns clamp at the image edge like the
// shader's sampler. Disjoint rectangles may run at the same time.
inline void filterRegion (const float* pixels, std::uint32_t width, std::uint32_t height, bool linear,
                          std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1,
                          std::uint8_t* output) noexcept
{
    constexpr int radius = kNativeFilterRadius;
    constexpr int diameter = 2 * radius + 1;
    const float strength = kNativeFilterStrength;
    const float colorSigma = 0.02f + (0.30f - 0.02f) * strength;
    const float rangeScale = 1.0f / std::max (colorSigma * colorSigma, 1.0e-5f);
    float spatial[diameter * diameter];
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            spatial[(dy + radius) * diameter + dx + radius] =
                std::exp (-static_cast<float> (dx * dx + dy * dy) / static_cast<float> (radius * radius));

    const int lastX = static_cast<int> (width) - 1;
    const int lastY = static_cast<int> (height) - 1;
    for (std::uint32_t y = y0; y < y1; ++y)
    {
        for (std::uint32_t x = x0; x < x1; ++x)
        {
            const std::size_t centerIndex = (static_cast<std::size_t> (y) * width + x) * 4;
            const float* center = pixels + centerIndex;
            if (linear && atHalfLimit (center))
            {
                for (int channel = 0; channel < 4; ++channel)
                {
                    const std::uint16_t bits = videohelper::pathtrace::halfBits (center[channel]);
                    std::memcpy (output + centerIndex * 2 + static_cast<std::size_t> (channel) * 2, &bits, sizeof (bits));
                }
                continue;
            }
            const auto compressed = [] (float value) noexcept { const float v = std::max (value, 0.0f); return v / (1.0f + v); };
            float reference[3] { center[0], center[1], center[2] };
            float blend = strength;
            if (linear)
            {
                float rangeInput[3][diameter * diameter];
                for (int channel = 0; channel < 3; ++channel)
                {
                    int count = 0;
                    for (int dy = -radius; dy <= radius; ++dy)
                        for (int dx = -radius; dx <= radius; ++dx)
                        {
                            const int sampleY = std::min (lastY, std::max (0, static_cast<int> (y) + dy));
                            const int sampleX = std::min (lastX, std::max (0, static_cast<int> (x) + dx));
                            rangeInput[channel][count++] = compressed (
                                pixels[(static_cast<std::size_t> (sampleY) * width + static_cast<std::size_t> (sampleX)) * 4
                                       + static_cast<std::size_t> (channel)]);
                        }
                    std::nth_element (rangeInput[channel], rangeInput[channel] + count / 2, rangeInput[channel] + count);
                    reference[channel] = rangeInput[channel][count / 2];
                }
                const float dr = compressed (center[0]) - reference[0];
                const float dg = compressed (center[1]) - reference[1];
                const float db = compressed (center[2]) - reference[2];
                blend = strength + (1.0f - strength) * (1.0f - std::exp (-(dr * dr + dg * dg + db * db) * rangeScale));
            }
            float sum[4] {};
            float weightSum = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy)
            {
                const int sampleY = std::min (lastY, std::max (0, static_cast<int> (y) + dy));
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int sampleX = std::min (lastX, std::max (0, static_cast<int> (x) + dx));
                    const float* value = pixels + (static_cast<std::size_t> (sampleY) * width
                                                   + static_cast<std::size_t> (sampleX)) * 4;
                    if (linear && atHalfLimit (value))
                        continue;
                    const float dr = (linear ? compressed (value[0]) : value[0]) - reference[0];
                    const float dg = (linear ? compressed (value[1]) : value[1]) - reference[1];
                    const float db = (linear ? compressed (value[2]) : value[2]) - reference[2];
                    const float weight = spatial[(dy + radius) * diameter + dx + radius]
                        * std::exp (-(dr * dr + dg * dg + db * db) * rangeScale);
                    for (int channel = 0; channel < 4; ++channel)
                        sum[channel] += value[channel] * weight;
                    weightSum += weight;
                }
            }
            // Nothing in the window is near the median, for example beside pixels at the half limit.
            // Dividing by a floor here took the pixel and its alpha toward 0.
            const bool keepCenter = weightSum < 1.0e-5f;
            weightSum = std::max (weightSum, 1.0e-5f);
            for (int channel = 0; channel < 4; ++channel)
            {
                const float filtered = keepCenter ? center[channel]
                                                  : center[channel] + (sum[channel] / weightSum - center[channel]) * blend;
                if (linear)
                {
                    const std::uint16_t bits = videohelper::pathtrace::halfBits (filtered);
                    std::memcpy (output + centerIndex * 2 + static_cast<std::size_t> (channel) * 2,
                                 &bits, sizeof (bits));
                }
                else
                {
                    const float clamped = std::min (1.0f, std::max (0.0f, filtered));
                    output[centerIndex + static_cast<std::size_t> (channel)] =
                        static_cast<std::uint8_t> (clamped * 255.0f + 0.5f);
                }
            }
        }
    }
}

// The weight that shares a residual between two pixels. Symmetric, so a pixel can
// gather what its neighbours send.
inline float energyShareWeight (const float* a, const float* b, float spatial, float rangeScale) noexcept
{
    const auto compressed = [] (float value) noexcept { const float v = std::max (value, 0.0f); return v / (1.0f + v); };
    const float dr = compressed (a[0]) - compressed (b[0]);
    const float dg = compressed (a[1]) - compressed (b[1]);
    const float db = compressed (a[2]) - compressed (b[2]);
    return spatial * std::exp (-(dr * dr + dg * dg + db * db) * rangeScale);
}

inline float energyShareRangeScale() noexcept
{
    const float colorSigma = 0.02f + (0.30f - 0.02f) * kNativeFilterStrength;
    return 1.0f / std::max (colorSigma * colorSigma, 1.0e-5f);
}

// The row or column a step past the image edge reflects to. A pair of pixels then weighs the
// same in both directions, so every pixel gathers as much as it sends and a flat image stays
// flat up to its edges. Dropping those steps took a 4-pixel border 29% dark at the corners and
// 19% bright just inside them.
inline int mirroredIndex (int index, int size) noexcept
{
    while (index < 0 || index >= size)
        index = index < 0 ? -1 - index : 2 * size - 1 - index;
    return index;
}

// Each pixel's starting scale: 0 for a pixel that keeps its input and takes no part in sharing.
inline void energyShareScalesRegion (const float* pixels, std::uint32_t width, std::uint32_t height,
                                     std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1,
                                     float* scales) noexcept
{
    for (std::uint32_t y = y0; y < y1; ++y)
        for (std::uint32_t x = x0; x < x1; ++x)
        {
            const std::size_t selfIndex = static_cast<std::size_t> (y) * width + x;
            scales[selfIndex] = atHalfLimit (pixels + selfIndex * 4) || loneSpark (pixels, width, height, x, y) ? 0.0f : 1.0f;
        }
}

// Each pixel's total scaled weight toward the pixels that can take a share. filtered holds
// four floats per pixel, and scales and totals one float per pixel. A total of 0 marks a
// pixel that keeps its input and sends nothing.
inline void energyShareTotalsRegion (const float* filtered, const float* scales, std::uint32_t width, std::uint32_t height,
                                     std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1,
                                     float* totals) noexcept
{
    constexpr int radius = kNativeFilterRadius;
    const float rangeScale = energyShareRangeScale();
    for (std::uint32_t y = y0; y < y1; ++y)
        for (std::uint32_t x = x0; x < x1; ++x)
        {
            const std::size_t selfIndex = static_cast<std::size_t> (y) * width + x;
            if (scales[selfIndex] == 0.0f)
            {
                totals[selfIndex] = 0.0f;
                continue;
            }
            const float* self = filtered + selfIndex * 4;
            float total = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int otherX = mirroredIndex (static_cast<int> (x) + dx, static_cast<int> (width));
                    const int otherY = mirroredIndex (static_cast<int> (y) + dy, static_cast<int> (height));
                    const std::size_t otherIndex = static_cast<std::size_t> (otherY) * width + static_cast<std::size_t> (otherX);
                    if (scales[otherIndex] == 0.0f)
                        continue;
                    const float spatial = std::exp (-static_cast<float> (dx * dx + dy * dy) / static_cast<float> (radius * radius));
                    total += scales[otherIndex] * energyShareWeight (self, filtered + otherIndex * 4, spatial, rangeScale);
                }
            totals[selfIndex] = total;
        }
}

// One balancing step from the totals of the current scales.
inline void balanceEnergyShareScalesRegion (const float* totals, std::uint32_t width,
                                            std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1,
                                            float* scales) noexcept
{
    for (std::uint32_t y = y0; y < y1; ++y)
        for (std::uint32_t x = x0; x < x1; ++x)
        {
            const std::size_t selfIndex = static_cast<std::size_t> (y) * width + x;
            if (totals[selfIndex] > 0.0f)
                scales[selfIndex] = std::sqrt (scales[selfIndex] / totals[selfIndex]);
        }
}

// The last pass: each pixel gathers its share of its neighbours' input and its
// own, written over the filtered pixel in a tight RGBA16F output. Alpha stays filtered.
// A pixel at the half limit takes no shares, and its output is already its input.
inline void restoreEnergyRegion (const float* pixels, const float* filtered, const float* scales, const float* totals, std::uint32_t width,
                                 std::uint32_t height, std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1,
                                 std::uint8_t* output) noexcept
{
    constexpr int radius = kNativeFilterRadius;
    const float rangeScale = energyShareRangeScale();
    for (std::uint32_t y = y0; y < y1; ++y)
        for (std::uint32_t x = x0; x < x1; ++x)
        {
            const std::size_t selfIndex = static_cast<std::size_t> (y) * width + x;
            const float* self = filtered + selfIndex * 4;
            if (atHalfLimit (pixels + selfIndex * 4))
                continue;
            float restored[3] {};
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int otherX = mirroredIndex (static_cast<int> (x) + dx, static_cast<int> (width));
                    const int otherY = mirroredIndex (static_cast<int> (y) + dy, static_cast<int> (height));
                    const std::size_t otherIndex = static_cast<std::size_t> (otherY) * width + static_cast<std::size_t> (otherX);
                    if (totals[otherIndex] == 0.0f && (dx != 0 || dy != 0))
                        continue;
                    const float spatial = std::exp (-static_cast<float> (dx * dx + dy * dy) / static_cast<float> (radius * radius));
                    const float share = totals[otherIndex] == 0.0f
                        ? 1.0f
                        : scales[selfIndex] * energyShareWeight (self, filtered + otherIndex * 4, spatial, rangeScale) / totals[otherIndex];
                    for (int channel = 0; channel < 3; ++channel)
                        restored[channel] += share * pixels[otherIndex * 4 + static_cast<std::size_t> (channel)];
                }
            for (int channel = 0; channel < 3; ++channel)
            {
                const std::uint16_t bits = videohelper::pathtrace::halfBits (restored[channel]);
                std::memcpy (output + selfIndex * 8 + static_cast<std::size_t> (channel) * 2, &bits, sizeof (bits));
            }
        }
}

// Every energy pass over a filtered linear image. runRegions (fn) calls fn (x0, x1, y0, y1)
// over disjoint rectangles that cover the image, and returns when all are done, so one
// pass finishes before the next reads it.
template <typename RunRegions>
void restoreEnergy (const float* pixels, const float* filtered, std::uint32_t width, std::uint32_t height,
                    std::vector<float>& scales, std::vector<float>& totals, std::uint8_t* output, RunRegions&& runRegions)
{
    scales.resize (static_cast<std::size_t> (width) * height);
    totals.resize (scales.size());
    runRegions ([&] (std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1) {
        energyShareScalesRegion (pixels, width, height, x0, x1, y0, y1, scales.data());
    });
    const auto computeTotals = [&] (std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1) {
        energyShareTotalsRegion (filtered, scales.data(), width, height, x0, x1, y0, y1, totals.data());
    };
    for (int pass = 0; pass < kEnergyShareBalancePasses; ++pass)
    {
        runRegions (computeTotals);
        runRegions ([&] (std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1) {
            balanceEnergyShareScalesRegion (totals.data(), width, x0, x1, y0, y1, scales.data());
        });
    }
    runRegions (computeTotals);
    runRegions ([&] (std::uint32_t x0, std::uint32_t x1, std::uint32_t y0, std::uint32_t y1) {
        restoreEnergyRegion (pixels, filtered, scales.data(), totals.data(), width, height, x0, x1, y0, y1, output);
    });
}

// The whole image on the calling thread. False when the input is malformed.
inline bool applyNativeFilter (const std::vector<std::uint8_t>& color, bool linear, std::uint32_t width,
                               std::uint32_t height, std::vector<std::uint8_t>& output)
{
    std::vector<float> pixels;
    if (! loadPixels (color, linear, width, height, pixels))
        return false;
    output.resize (color.size());
    filterRegion (pixels.data(), width, height, linear, 0, width, 0, height, output.data());
    if (linear)
    {
        std::vector<float> filtered;
        if (! loadPixels (output, true, width, height, filtered))
            return false;
        std::vector<float> scales;
        std::vector<float> totals;
        restoreEnergy (pixels.data(), filtered.data(), width, height, scales, totals, output.data(), [&] (auto&& region) {
            region (0u, width, 0u, height);
        });
    }
    return true;
}

} // namespace videorender::fixture3d::pathtracedenoise
