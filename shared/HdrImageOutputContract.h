#pragma once

#include "ColorTransformGpuMath.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

// Export interpretation is explicit: legacy composites have no uniform transfer
// declaration. These profiles do not infer or recover clipped source highlights.
namespace hdrimage
{
struct Profile
{
    const char* token;
    const char* label;
    colortransform::TransferFunction inputTransfer;
    colortransform::ColorSpace outputPrimaries;
};
inline constexpr Profile profiles[] {
    {"off", "Off (SDR video only)", colortransform::TransferFunction::Linear, colortransform::ColorSpace::LinearSRGB},
    {"linear-srgb", "Linear sRGB input -> linear sRGB", colortransform::TransferFunction::Linear, colortransform::ColorSpace::LinearSRGB},
    {"linear-p3", "Linear sRGB input -> linear Display P3", colortransform::TransferFunction::Linear, colortransform::ColorSpace::DisplayP3},
    {"linear-rec2020", "Linear sRGB input -> linear Rec.2020", colortransform::TransferFunction::Linear, colortransform::ColorSpace::Rec2020},
    {"srgb-linear-srgb", "sRGB input -> linear sRGB", colortransform::TransferFunction::SRGB, colortransform::ColorSpace::LinearSRGB},
    {"srgb-linear-p3", "sRGB input -> linear Display P3", colortransform::TransferFunction::SRGB, colortransform::ColorSpace::DisplayP3},
    {"srgb-linear-rec2020", "sRGB input -> linear Rec.2020", colortransform::TransferFunction::SRGB, colortransform::ColorSpace::Rec2020}
};
inline const Profile* find(const std::string& token) noexcept
{
    for (const auto& profile : profiles) if (token == profile.token) return &profile;
    return nullptr;
}
inline bool enabled(const std::string& token) noexcept { return token != "off"; }
inline const char* primaries(const Profile& profile) noexcept
{
    return profile.outputPrimaries == colortransform::ColorSpace::DisplayP3 ? "display-p3-d65"
        : profile.outputPrimaries == colortransform::ColorSpace::Rec2020 ? "rec2020-d65" : "srgb-bt709-d65";
}
inline float halfToFloat(std::uint16_t value) noexcept
{
    const int exponent = (value >> 10) & 31;
    const int fraction = value & 1023;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
        : exponent == 31 ? (fraction == 0 ? INFINITY : NAN)
        : std::ldexp(static_cast<float>(1024 + fraction), exponent - 25);
    return (value & 0x8000) != 0 ? -magnitude : magnitude;
}
inline double decode(double value, colortransform::TransferFunction transfer) noexcept
{
    if (transfer == colortransform::TransferFunction::Linear) return value;
    const double magnitude = std::abs(value);
    return std::copysign(magnitude <= 0.04045 ? magnitude / 12.92
        : std::pow((magnitude + 0.055) / 1.055, 2.4), value);
}
// Opaque final composition only. The HDR array retains signed, extended-range
// linear RGB. The companion video is an actual Reinhard + BT.709 OETF preview,
// not HDR metadata attached to SDR pixels. The existing video encoder stays SDR.
inline bool convert(const Profile& profile, const std::vector<float>& input,
                    std::vector<float>& linearOutput, std::vector<std::uint8_t>& preview,
                    std::string& error)
{
    linearOutput.clear(); preview.clear();
    if (std::string(profile.token) == "off" || input.empty() || input.size() % 4 != 0)
    { error = "HDR image export requires an explicit input interpretation and RGBA float pixels"; return false; }
    const auto matrix = profile.outputPrimaries == colortransform::ColorSpace::LinearSRGB
        ? colortransform::gpumath::Matrix3 {1, 0, 0, 0, 1, 0, 0, 0, 1}
        : colortransform::gpumath::multiply(colortransform::gpumath::xyzToRgb(profile.outputPrimaries),
            colortransform::gpumath::rgbToXyz(colortransform::ColorSpace::LinearSRGB));
    std::vector<float> converted(input.size());
    std::vector<std::uint8_t> display(input.size());
    for (std::size_t i = 0; i < input.size(); i += 4)
    {
        if (!std::isfinite(input[i]) || !std::isfinite(input[i + 1]) || !std::isfinite(input[i + 2])
            || !std::isfinite(input[i + 3]) || std::abs(input[i + 3] - 1.0f) > 0.001f)
        { error = "HDR image export requires finite pixels and an opaque canvas"; return false; }
        double linear[3];
        for (int c = 0; c < 3; ++c) linear[c] = decode(input[i + c], profile.inputTransfer);
        for (int c = 0; c < 3; ++c)
        {
            const double value = matrix[c * 3] * linear[0] + matrix[c * 3 + 1] * linear[1]
                + matrix[c * 3 + 2] * linear[2];
            converted[i + c] = static_cast<float>(value);
            if (!std::isfinite(converted[i + c]))
            { error = "HDR image conversion exceeded float32 range"; return false; }
            const double positive = std::max(0.0, linear[c]);
            const double mapped = positive / (1.0 + positive);
            const double encoded = mapped < 0.018 ? 4.5 * mapped : 1.099 * std::pow(mapped, 0.45) - 0.099;
            display[i + c] = static_cast<std::uint8_t>(std::lround(std::clamp(encoded, 0.0, 1.0) * 255.0));
        }
        converted[i + 3] = 1.0f; display[i + 3] = 255;
    }
    linearOutput.swap(converted); preview.swap(display); error.clear(); return true;
}
} // namespace hdrimage
