#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// Authored fixtures and expected First-anchored motion come from known image
// construction, independently of the production patch matcher.
namespace opticalflowtest
{
constexpr std::uint32_t width = 36;
constexpr std::uint32_t height = 28;
struct Expected { int x, y, dx, dy; };
struct Pair
{
    const char* name;
    std::vector<float> first, second;
    std::vector<Expected> expected;
};

inline float textureValue(int x, int y, std::uint32_t seed)
{
    std::uint32_t bits = std::uint32_t(x) * 0x45d9f3bu
                       ^ std::uint32_t(y) * 0x119de1f3u ^ seed;
    bits ^= bits >> 16u;
    bits *= 0x45d9f3bu;
    bits ^= bits >> 16u;
    // Binary fractions remain exact in both RGBA16F source uploads.
    return float(bits & 255u) / 256.0f;
}

inline std::vector<float> image(int dx, int dy, bool foreground)
{
    std::vector<float> pixels(width * height * 4u);
    for (int y = 0; y < int(height); ++y)
        for (int x = 0; x < int(width); ++x)
        {
            const int localX = x - dx, localY = y - dy;
            const bool inside = foreground && localX >= 10 && localX <= 22
                                           && localY >= 8 && localY <= 19;
            const int tx = foreground && !inside ? x : localX;
            const int ty = foreground && !inside ? y : localY;
            const float base = foreground ? (inside ? 0.0625f : 0.5625f) : 0.125f;
            const float scale = foreground ? 0.25f : 0.75f;
            const auto offset = (std::size_t(y) * width + std::size_t(x)) * 4u;
            pixels[offset] = base + scale * textureValue(tx, ty, 0x3519u);
            pixels[offset + 1u] = base + scale * textureValue(tx, ty, 0x8ab7u);
            pixels[offset + 2u] = base + scale * textureValue(tx, ty, 0x16e3u);
            pixels[offset + 3u] = 1.0f;
        }
    return pixels;
}

inline std::vector<Pair> pairs()
{
    std::vector<Pair> result;
    for (const auto displacement : { Expected{0, 0, 3, -2}, Expected{0, 0, -2, 3},
                                    Expected{0, 0, 4, 0}, Expected{0, 0, 2, 0},
                                    Expected{0, 0, 1, 0} })
    {
        Pair pair { displacement.dx == 4 ? "textured-right-search-boundary"
                        : displacement.dx == 2 ? "textured-right-two"
                        : displacement.dx == 1 ? "textured-right-one"
                        : displacement.dx > 0 ? "textured-right-up" : "textured-left-down",
                    image(0, 0, false), image(displacement.dx, displacement.dy, false), {} };
        // The +4 case also verifies the inclusive search boundary in both
        // directions. These coordinates keep each translated 3x3 patch inside.
        for (int y = 6; y <= 21; ++y)
            for (int x = 6; x <= 29; ++x)
                pair.expected.push_back({x, y, displacement.dx, displacement.dy});
        result.push_back(pair);
    }
    Pair stationary { "textured-stationary", image(0, 0, false), image(0, 0, false), {} };
    for (int y = 1; y < int(height) - 1; ++y)
        for (int x = 1; x < int(width) - 1; ++x)
            stationary.expected.push_back({x, y, 0, 0});
    result.push_back(stationary);

    Pair layered { "foreground-stationary-background", image(0, 0, true), image(3, 2, true), {} };
    // First foreground remains visible at its translated Second coordinate.
    // This includes old coordinates newly revealed as background in Second.
    for (int y = 9; y <= 18; ++y)
        for (int x = 11; x <= 21; ++x)
            layered.expected.push_back({x, y, 3, 2});
    // Background visible in both images stays exactly stationary.
    for (int y = 5; y <= 22; ++y)
        for (const int x : { 6, 7, 28, 29 })
            layered.expected.push_back({x, y, 0, 0});
    // First background occluded in Second has no correspondence. Require the
    // explicit zero fallback rather than leaking the foreground displacement.
    // Second-only newly revealed content has no First sample or validity claim.
    for (int y = 12; y <= 17; ++y)
        layered.expected.push_back({24, y, 0, 0});
    result.push_back(layered);

    Pair ambiguous { "repeated-texture-ambiguous", image(0, 0, false), {}, {} };
    for (int y = 0; y < int(height); ++y)
        for (int x = 0; x < int(width); ++x)
        {
            const auto offset = (std::size_t(y) * width + std::size_t(x)) * 4u;
            for (int channel = 0; channel < 3; ++channel)
                ambiguous.first[offset + channel] = ((x + y) & 1) ? 0.75f : 0.25f;
        }
    ambiguous.second = ambiguous.first;
    // A one-pixel checkerboard phase change has many equally exact nonzero
    // matches. Deterministic tie breaking alone would invent motion here.
    for (std::size_t pixel = 0; pixel < width * height; ++pixel)
        for (int channel = 0; channel < 3; ++channel)
            ambiguous.second[pixel * 4u + channel] = 1.0f - ambiguous.first[pixel * 4u + channel];
    for (int y = 6; y <= 21; ++y)
        for (int x = 6; x <= 29; ++x)
            ambiguous.expected.push_back({x, y, 0, 0});
    result.push_back(ambiguous);

    Pair reverseAmbiguous { "reverse-match-ambiguous", image(0, 0, false), image(0, 0, false), {} };
    // First pixel 14,14 has a unique forward match at Second 17,14. The
    // reverse search sees identical First patches at 14,14 and 20,14.
    // A forward-only confidence check would publish +3,0 despite ambiguity.
    const auto original = reverseAmbiguous.first;
    for (int py = -1; py <= 1; ++py)
        for (int px = -1; px <= 1; ++px)
            for (int channel = 0; channel < 4; ++channel)
            {
                const auto source = (std::size_t(14 + py) * width + 20 + px) * 4u + channel;
                const auto firstTarget = (std::size_t(14 + py) * width + 14 + px) * 4u + channel;
                const auto secondTarget = (std::size_t(14 + py) * width + 17 + px) * 4u + channel;
                reverseAmbiguous.first[firstTarget] = original[source];
                reverseAmbiguous.second[secondTarget] = original[source];
            }
    reverseAmbiguous.expected.push_back({14, 14, 0, 0});
    result.push_back(reverseAmbiguous);
    return result;
}

inline bool verify(const Pair& pair, const std::vector<float>& field, const char* mode)
{
    if (field.size() != width * height * 2u) return false;
    for (const auto value : field)
        if (!std::isfinite(value)) return false;
    for (const auto expected : pair.expected)
    {
        const auto offset = (std::size_t(expected.y) * width + expected.x) * 2u;
        if (field[offset] != float(expected.dx) || field[offset + 1u] != float(expected.dy))
        {
            std::fprintf(stderr, "%s %s pixel=%d,%d expected=%d,%d actual=%g,%g\n",
                         pair.name, mode, expected.x, expected.y, expected.dx, expected.dy,
                         double(field[offset]), double(field[offset + 1u]));
            return false;
        }
    }
    return true;
}

inline bool verifyView(const Pair& pair, const std::vector<std::uint8_t>& rgba,
                       double range, const char* mode, int tolerance)
{
    if (rgba.size() != width * height * 4u) return false;
    for (const auto expected : pair.expected)
    {
        const auto offset = (std::size_t(expected.y) * width + expected.x) * 4u;
        // Known authored translations supply this oracle independently of the
        // backend shader and field readback. Range 16 is exact in binary16.
        const auto red = std::lround(255.0 * std::clamp(0.5 + expected.dx
            / (range > 0.0 ? range : double(width)), 0.0, 1.0));
        const auto green = std::lround(255.0 * std::clamp(0.5 + expected.dy
            / (range > 0.0 ? range : double(height)), 0.0, 1.0));
        if (std::abs(int(rgba[offset]) - int(red)) > tolerance
            || std::abs(int(rgba[offset + 1u]) - int(green)) > tolerance
            || std::abs(int(rgba[offset + 2u]) - 128) > tolerance || rgba[offset + 3u] != 255u)
        {
            std::fprintf(stderr, "%s %s view pixel=%d,%d range=%g expected=%ld,%ld,128,255 actual=%u,%u,%u,%u\n",
                pair.name, mode, expected.x, expected.y, range, red, green,
                unsigned(rgba[offset]), unsigned(rgba[offset + 1u]),
                unsigned(rgba[offset + 2u]), unsigned(rgba[offset + 3u]));
            return false;
        }
    }
    return true;
}
} // namespace opticalflowtest
