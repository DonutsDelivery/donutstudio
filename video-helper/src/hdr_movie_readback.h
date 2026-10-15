#pragma once
#include "../../shared/ColorTransformContract.h"
#include "../../shared/ColorTransformGpuMath.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// Encoder-boundary transforms share admission and pixel checks. The renderer
// supplies only its current owned linear float composite, never its SDR copy.
namespace videorender::hdrmovie
{
inline bool validPeak(float peakNits)
{
    return std::isfinite(peakNits) && peakNits >= colortransform::kMinimumHdrPeakNits
        && peakNits <= colortransform::kMaximumPeakNits;
}

template <typename ReadLinear, typename Transform>
bool readPq(float peakNits, colortransform::Extent extent, ReadLinear&& readLinear,
            Transform&& transform, std::vector<float>& rgba, std::string& error)
{
    rgba.clear();
    if (!validPeak(peakNits)) { error = "PQ movie peak is invalid"; return false; }
    std::vector<float> linear;
    if (!readLinear(linear, error)) return false;
    if (linear.size() != static_cast<std::size_t>(extent.width) * extent.height * 4 || linear.empty())
    { error = "PQ movie requires exact final float pixel dimensions"; return false; }
    constexpr double referenceWhiteNits = 203.0;
    for (std::size_t i = 0; i < linear.size(); i += 4)
    {
        if (!std::isfinite(linear[i + 3]) || linear[i + 3] != 1.0f)
        { error = "PQ movie output requires opaque final pixels"; return false; }
        for (int c = 0; c < 3; ++c)
            if (!std::isfinite(linear[i + c]) || linear[i + c] < 0.0f
                || static_cast<double>(linear[i + c]) * referenceWhiteNits > peakNits)
            { error = "PQ movie final pixel exceeds the declared luminance range"; return false; }
    }
    colortransform::Description description;
    description.input = {extent, colortransform::PixelFormat::RGBA16F,
        colortransform::ColorSpace::LinearSRGB, colortransform::TransferFunction::Linear,
        colortransform::AlphaMode::Opaque};
    description.output = {extent, colortransform::PixelFormat::RGBA16F,
        colortransform::ColorSpace::Rec2020, colortransform::TransferFunction::PQ,
        colortransform::AlphaMode::Opaque};
    description.workingColorSpace = colortransform::ColorSpace::Rec2020;
    description.outputIntent = colortransform::OutputIntent::HdrPq;
    description.luminance = {referenceWhiteNits, peakNits, referenceWhiteNits, peakNits};
    colortransform::AdmissionFailure failure;
    const auto admitted = colortransform::admit(
        description, colortransform::BackendCapability::NativeGpu, failure);
    if (!admitted)
    { error = "PQ movie color transform admission failed: "
        + std::string(colortransform::token(failure)); return false; }
    if (!transform(*admitted, rgba, error)) { rgba.clear(); return false; }
    if (rgba.size() != linear.size())
    { rgba.clear(); error = "PQ movie transform changed pixel dimensions"; return false; }
    const double m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const double c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0;
    const double c3 = 2392.0 / 128.0;
    const double p = std::pow(static_cast<double>(peakNits) / 10000.0, m1);
    const double encodedPeak = std::pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
    // PQ peaks admitted here lie in the binary16 [0.5, 1] binade. Its
    // representable spacing is 2^-11. Allow only the upper adjacent storage
    // value, since an in-range ideal value can round upward on RGBA16F write.
    // Input luminance validation above still rejects over-peak linear pixels.
    const double storedPeakBound = std::ceil(std::ldexp(encodedPeak, 11)) / 2048.0;
    for (std::size_t i = 0; i < rgba.size(); i += 4)
    {
        if (rgba[i + 3] != 1.0f)
        { error = "PQ movie transform returned non-opaque pixels"; rgba.clear(); return false; }
        for (int c = 0; c < 3; ++c)
            if (!std::isfinite(rgba[i + c]) || rgba[i + c] < 0.0f
                || rgba[i + c] > 1.0f || rgba[i + c] > storedPeakBound)
            { error = "PQ movie transform returned out-of-range pixels"; rgba.clear(); return false; }
    }
    error.clear(); return true;
}

template <typename ReadLinear>
bool readHlg(float peakNits, ReadLinear&& readLinear, std::vector<float>& rgba, std::string& error)
{
    rgba.clear();
    if (!validPeak(peakNits)) { error = "HLG movie peak is invalid"; return false; }
    if (!readLinear(rgba, error)) return false;
    if (rgba.empty() || rgba.size() % 4 != 0)
    { rgba.clear(); error = "HLG movie requires complete final float pixels"; return false; }
    const auto fail = [&](const char* message) {
        rgba.clear(); error = message; return false;
    };
    constexpr double referenceWhiteNits = 203.0;
    const auto matrix = colortransform::gpumath::multiply(
        colortransform::gpumath::xyzToRgb(colortransform::ColorSpace::Rec2020),
        colortransform::gpumath::rgbToXyz(colortransform::ColorSpace::LinearSRGB));
    // ITU-R BT.2100-3 Table 5, Notes 5f/5i: inverse display OOTF then OETF.
    // Table 10 uses PQ's existing display-linear 203-nit white normalization.
    // Lw is the declared nominal peak, with reference surround gamma and zero
    // black lift in the encoded signal.
    const double gamma = peakNits <= 2000.0f
        ? 1.2 + 0.42 * std::log10(peakNits / 1000.0)
        : 1.2 * std::pow(1.111, std::log2(peakNits / 1000.0));
    constexpr double a = 0.17883277, b = 1.0 - 4.0 * a;
    const double c = 0.5 - a * std::log(4.0 * a);
    for (std::size_t i = 0; i < rgba.size(); i += 4)
    {
        if (!std::isfinite(rgba[i + 3]) || rgba[i + 3] != 1.0f)
            return fail("HLG movie output requires opaque final pixels");
        for (int channel = 0; channel < 3; ++channel)
            if (!std::isfinite(rgba[i + channel]) || rgba[i + channel] < 0.0f
                || static_cast<double>(rgba[i + channel]) * referenceWhiteNits > peakNits)
                return fail("HLG movie final pixel exceeds the declared luminance range");
        double display[3] {};
        for (int channel = 0; channel < 3; ++channel)
            for (int input = 0; input < 3; ++input)
                display[channel] += matrix[channel * 3 + input] * rgba[i + input]
                    * referenceWhiteNits / peakNits;
        const double luminance = 0.2627 * display[0] + 0.6780 * display[1]
            + 0.0593 * display[2];
        const double inverseGain = luminance > 0.0
            ? std::pow(luminance, (1.0 - gamma) / gamma) : 0.0;
        for (int channel = 0; channel < 3; ++channel)
        {
            const double scene = display[channel] * inverseGain;
            // Permit only matrix arithmetic roundoff at unity, never clip color
            // that cannot be represented by the admitted normalized HLG signal.
            if (!std::isfinite(scene) || scene < 0.0 || scene > 1.0 + 1.0e-9)
                return fail("HLG movie pixel exceeds the normalized scene-light range");
            const double normalized = std::min(scene, 1.0);
            const double encoded = normalized <= 1.0 / 12.0
                ? std::sqrt(3.0 * normalized) : a * std::log(12.0 * normalized - b) + c;
            if (!std::isfinite(encoded) || encoded < 0.0 || encoded > 1.0)
                return fail("HLG movie transform returned out-of-range pixels");
            rgba[i + channel] = static_cast<float>(encoded);
        }
    }
    error.clear(); return true;
}
} // namespace videorender::hdrmovie
