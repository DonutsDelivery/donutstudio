#pragma once

#include "ColorTransformContract.h"
#include "PortableFloatingPointChars.h"

#include <array>
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace colortransformoperation
{
inline constexpr std::string_view kOperationKind = "visual.color.transform";
inline constexpr std::string_view kBackendCapability = "native-gpu";

// Image extent belongs to the concrete decoded frame, not the immutable graph.
// The helper binds it before admission and dispatch.
struct Payload
{
    std::uint32_t version = colortransform::kWireVersion;
    colortransform::PixelFormat inputFormat = colortransform::PixelFormat::Unspecified;
    colortransform::ColorSpace inputColorSpace = colortransform::ColorSpace::Unspecified;
    colortransform::TransferFunction inputTransfer = colortransform::TransferFunction::Unspecified;
    colortransform::AlphaMode inputAlpha = colortransform::AlphaMode::Unspecified;
    colortransform::PixelFormat outputFormat = colortransform::PixelFormat::Unspecified;
    colortransform::ColorSpace outputColorSpace = colortransform::ColorSpace::Unspecified;
    colortransform::TransferFunction outputTransfer = colortransform::TransferFunction::Unspecified;
    colortransform::AlphaMode outputAlpha = colortransform::AlphaMode::Unspecified;
    colortransform::ColorSpace workingColorSpace = colortransform::ColorSpace::Unspecified;
    colortransform::PixelFormat workingFormat = colortransform::PixelFormat::Unspecified;
    colortransform::OutputIntent outputIntent = colortransform::OutputIntent::SdrDisplay;
    colortransform::ToneMap toneMap = colortransform::ToneMap::None;
    colortransform::LuminanceParameters luminance;
};

inline colortransform::Description descriptionForExtent(
    const Payload& payload, colortransform::Extent extent) noexcept
{
    colortransform::Description description;
    description.version = payload.version;
    description.input = { extent, payload.inputFormat, payload.inputColorSpace,
                          payload.inputTransfer, payload.inputAlpha };
    description.output = { extent, payload.outputFormat, payload.outputColorSpace,
                           payload.outputTransfer, payload.outputAlpha };
    description.workingColorSpace = payload.workingColorSpace;
    description.workingFormat = payload.workingFormat;
    description.outputIntent = payload.outputIntent;
    description.toneMap = payload.toneMap;
    description.luminance = payload.luminance;
    return description;
}

namespace detail
{
inline void appendAttribute(std::string& wire, std::string_view name,
                            std::string_view value)
{
    wire += ' ';
    wire.append(name);
    wire += "=\"";
    wire.append(value);
    wire += '"';
}

inline void appendNumber(std::string& wire, std::string_view name, double value)
{
    std::array<char, 64> buffer {};
    const auto result = portablefloat::toChars(buffer.data(), buffer.data() + buffer.size(), value,
                                               std::chars_format::general,
                                               std::numeric_limits<double>::max_digits10);
    appendAttribute(wire, name,
                    { buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data()) });
}

inline bool takeAttribute(std::string_view wire, std::size_t& cursor,
                          std::string_view name, std::string_view& value) noexcept
{
    if (cursor >= wire.size() || wire[cursor] != ' ') return false;
    ++cursor;
    if (wire.substr(cursor, name.size()) != name) return false;
    cursor += name.size();
    if (wire.substr(cursor, 2) != "=\"") return false;
    cursor += 2;
    const auto end = wire.find('"', cursor);
    if (end == std::string_view::npos) return false;
    value = wire.substr(cursor, end - cursor);
    cursor = end + 1;
    return ! value.empty();
}

inline bool parseUnsigned(std::string_view value, std::uint32_t& result) noexcept
{
    std::uint32_t parsed = 0;
    const auto conversion = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (conversion.ec != std::errc() || conversion.ptr != value.data() + value.size()) return false;
    result = parsed;
    return true;
}

inline bool parseDouble(std::string_view value, double& result) noexcept
{
    std::string text(value);
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text.c_str(), &end);
    if (errno != 0 || end != text.c_str() + text.size() || ! std::isfinite(parsed)) return false;
    result = parsed;
    return true;
}

template <typename Enum, std::size_t N>
inline std::optional<Enum> enumForToken(std::string_view value,
                                        const std::array<Enum, N>& values) noexcept
{
    for (const auto candidate : values)
        if (colortransform::token(candidate) == value) return candidate;
    return std::nullopt;
}
} // namespace detail

inline std::string serialize(const Payload& payload)
{
    std::string wire = "<ColorTransform";
    detail::appendAttribute(wire, "version", std::to_string(payload.version));
    detail::appendAttribute(wire, "inputFormat", colortransform::token(payload.inputFormat));
    detail::appendAttribute(wire, "inputColorSpace", colortransform::token(payload.inputColorSpace));
    detail::appendAttribute(wire, "inputTransfer", colortransform::token(payload.inputTransfer));
    detail::appendAttribute(wire, "inputAlpha", colortransform::token(payload.inputAlpha));
    detail::appendAttribute(wire, "outputFormat", colortransform::token(payload.outputFormat));
    detail::appendAttribute(wire, "outputColorSpace", colortransform::token(payload.outputColorSpace));
    detail::appendAttribute(wire, "outputTransfer", colortransform::token(payload.outputTransfer));
    detail::appendAttribute(wire, "outputAlpha", colortransform::token(payload.outputAlpha));
    detail::appendAttribute(wire, "workingColorSpace", colortransform::token(payload.workingColorSpace));
    detail::appendAttribute(wire, "workingFormat", colortransform::token(payload.workingFormat));
    detail::appendAttribute(wire, "outputIntent", colortransform::token(payload.outputIntent));
    detail::appendAttribute(wire, "toneMap", colortransform::token(payload.toneMap));
    detail::appendNumber(wire, "sourceReferenceWhiteNits",
                         payload.luminance.sourceReferenceWhiteNits);
    detail::appendNumber(wire, "sourcePeakNits", payload.luminance.sourcePeakNits);
    detail::appendNumber(wire, "outputReferenceWhiteNits",
                         payload.luminance.outputReferenceWhiteNits);
    detail::appendNumber(wire, "outputPeakNits", payload.luminance.outputPeakNits);
    wire += "/>";
    return wire;
}

inline bool parse(std::string_view wire, Payload& result) noexcept
{
    constexpr std::string_view prefix = "<ColorTransform";
    if (wire.substr(0, prefix.size()) != prefix) return false;
    std::size_t cursor = prefix.size();
    std::string_view value;
    Payload parsed;
    const auto attribute = [&](std::string_view name)
    { return detail::takeAttribute(wire, cursor, name, value); };

    constexpr std::array formats { colortransform::PixelFormat::R8,
        colortransform::PixelFormat::R16, colortransform::PixelFormat::RGBA8,
        colortransform::PixelFormat::RGBA16F, colortransform::PixelFormat::RGBA32F };
    constexpr std::array spaces { colortransform::ColorSpace::LinearSRGB,
        colortransform::ColorSpace::SRGB, colortransform::ColorSpace::DisplayP3,
        colortransform::ColorSpace::Rec709, colortransform::ColorSpace::Rec2020 };
    constexpr std::array transfers { colortransform::TransferFunction::Linear,
        colortransform::TransferFunction::SRGB, colortransform::TransferFunction::Gamma22,
        colortransform::TransferFunction::PQ, colortransform::TransferFunction::HLG,
        colortransform::TransferFunction::Rec709 };
    constexpr std::array alphas { colortransform::AlphaMode::Opaque,
        colortransform::AlphaMode::Straight, colortransform::AlphaMode::Premultiplied };
    constexpr std::array intents { colortransform::OutputIntent::SceneLinear,
        colortransform::OutputIntent::SdrDisplay, colortransform::OutputIntent::HdrPq,
        colortransform::OutputIntent::HdrHlg };
    constexpr std::array toneMaps { colortransform::ToneMap::None,
        colortransform::ToneMap::Reinhard };

    if (! attribute("version") || ! detail::parseUnsigned(value, parsed.version)) return false;
    auto inputFormat = std::optional<colortransform::PixelFormat> {};
    auto inputSpace = std::optional<colortransform::ColorSpace> {};
    auto inputTransfer = std::optional<colortransform::TransferFunction> {};
    auto inputAlpha = std::optional<colortransform::AlphaMode> {};
    auto outputFormat = std::optional<colortransform::PixelFormat> {};
    auto outputSpace = std::optional<colortransform::ColorSpace> {};
    auto outputTransfer = std::optional<colortransform::TransferFunction> {};
    auto outputAlpha = std::optional<colortransform::AlphaMode> {};
    auto workingSpace = std::optional<colortransform::ColorSpace> {};
    auto workingFormat = std::optional<colortransform::PixelFormat> {};
    auto intent = std::optional<colortransform::OutputIntent> {};
    auto toneMap = std::optional<colortransform::ToneMap> {};
    if (! attribute("inputFormat") || ! (inputFormat = detail::enumForToken(value, formats))
        || ! attribute("inputColorSpace") || ! (inputSpace = detail::enumForToken(value, spaces))
        || ! attribute("inputTransfer") || ! (inputTransfer = detail::enumForToken(value, transfers))
        || ! attribute("inputAlpha") || ! (inputAlpha = detail::enumForToken(value, alphas))
        || ! attribute("outputFormat") || ! (outputFormat = detail::enumForToken(value, formats))
        || ! attribute("outputColorSpace") || ! (outputSpace = detail::enumForToken(value, spaces))
        || ! attribute("outputTransfer") || ! (outputTransfer = detail::enumForToken(value, transfers))
        || ! attribute("outputAlpha") || ! (outputAlpha = detail::enumForToken(value, alphas))
        || ! attribute("workingColorSpace") || ! (workingSpace = detail::enumForToken(value, spaces))
        || ! attribute("workingFormat") || ! (workingFormat = detail::enumForToken(value, formats))
        || ! attribute("outputIntent") || ! (intent = detail::enumForToken(value, intents))
        || ! attribute("toneMap") || ! (toneMap = detail::enumForToken(value, toneMaps)))
        return false;

    parsed.inputFormat = *inputFormat;
    parsed.inputColorSpace = *inputSpace;
    parsed.inputTransfer = *inputTransfer;
    parsed.inputAlpha = *inputAlpha;
    parsed.outputFormat = *outputFormat;
    parsed.outputColorSpace = *outputSpace;
    parsed.outputTransfer = *outputTransfer;
    parsed.outputAlpha = *outputAlpha;
    parsed.workingColorSpace = *workingSpace;
    parsed.workingFormat = *workingFormat;
    parsed.outputIntent = *intent;
    parsed.toneMap = *toneMap;
    if (! attribute("sourceReferenceWhiteNits")
        || ! detail::parseDouble(value, parsed.luminance.sourceReferenceWhiteNits)
        || ! attribute("sourcePeakNits")
        || ! detail::parseDouble(value, parsed.luminance.sourcePeakNits)
        || ! attribute("outputReferenceWhiteNits")
        || ! detail::parseDouble(value, parsed.luminance.outputReferenceWhiteNits)
        || ! attribute("outputPeakNits")
        || ! detail::parseDouble(value, parsed.luminance.outputPeakNits)
        || wire.substr(cursor) != "/>")
        return false;

    result = parsed;
    return true;
}

inline std::optional<colortransform::AdmittedTransform> parseAndAdmit(
    std::string_view wire, colortransform::Extent extent,
    colortransform::AdmissionFailure& failure,
    const colortransform::AdmissionLimits& limits = {}) noexcept
{
    Payload payload;
    if (! parse(wire, payload))
    {
        failure = colortransform::AdmissionFailure::UnsupportedVersion;
        return std::nullopt;
    }
    return colortransform::admit(descriptionForExtent(payload, extent),
                                 colortransform::BackendCapability::NativeGpu,
                                 failure, limits);
}
} // namespace colortransformoperation
