#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace commoneffect
{
inline constexpr std::uint32_t kSchemaVersion = 1;
inline constexpr std::size_t kMaximumParameterCount = 3;

inline constexpr const char* kSharpenWireName = "sharpen";
inline constexpr const char* kSaturationWireName = "saturation";
inline constexpr const char* kExposureWireName = "exposure";
inline constexpr const char* kBrightnessWireName = "brightness";
inline constexpr const char* kContrastWireName = "contrast";
inline constexpr const char* kHueWireName = "hue";
inline constexpr const char* kGammaWireName = "gamma";
inline constexpr const char* kBlurWireName = "blur";
inline constexpr const char* kVignetteWireName = "vignette";
inline constexpr const char* kGlowWireName = "glow";
inline constexpr const char* kBloomWireName = "bloom";
inline constexpr const char* kHalationWireName = "halation";
inline constexpr const char* kLensDistortionWireName = "lens_distortion";
inline constexpr const char* kDenoiseWireName = "denoise";
inline constexpr const char* kDirectionalBlurWireName = "directional_blur";
inline constexpr const char* kUnsharpMaskWireName = "unsharp_mask";
inline constexpr const char* kHighPassSharpenWireName = "high_pass_sharpen";

enum class Kind : std::uint8_t
{
    sharpen = 0,
    saturation,
    exposure,
    brightness,
    contrast,
    hue,
    gamma,
    blur,
    vignette,
    glow,
    bloom,
    halation,
    lensDistortion,
    denoise,
    directionalBlur,
    unsharpMask,
    highPassSharpen
};

inline constexpr std::size_t kKindCount = 17;

inline constexpr bool isKnownKind(Kind kind) noexcept
{
    return static_cast<std::size_t>(kind) < kKindCount;
}

inline constexpr const char* wireName(Kind kind) noexcept
{
    switch (kind)
    {
        case Kind::sharpen: return kSharpenWireName;
        case Kind::saturation: return kSaturationWireName;
        case Kind::exposure: return kExposureWireName;
        case Kind::brightness: return kBrightnessWireName;
        case Kind::contrast: return kContrastWireName;
        case Kind::hue: return kHueWireName;
        case Kind::gamma: return kGammaWireName;
        case Kind::blur: return kBlurWireName;
        case Kind::vignette: return kVignetteWireName;
        case Kind::glow: return kGlowWireName;
        case Kind::bloom: return kBloomWireName;
        case Kind::halation: return kHalationWireName;
        case Kind::lensDistortion: return kLensDistortionWireName;
        case Kind::denoise: return kDenoiseWireName;
        case Kind::directionalBlur: return kDirectionalBlurWireName;
        case Kind::unsharpMask: return kUnsharpMaskWireName;
        case Kind::highPassSharpen: return kHighPassSharpenWireName;
    }
    return "";
}

inline constexpr const char* graphKindName(Kind kind) noexcept
{
    switch (kind)
    {
        case Kind::sharpen: return "visual.effect.sharpen";
        case Kind::saturation: return "visual.color.saturation";
        case Kind::exposure: return "visual.color.exposure";
        case Kind::brightness: return "visual.color.brightness";
        case Kind::contrast: return "visual.color.contrast";
        case Kind::hue: return "visual.color.hue";
        case Kind::gamma: return "visual.color.gamma";
        case Kind::blur: return "visual.effect.blur";
        case Kind::vignette: return "visual.effect.vignette";
        case Kind::glow: return "visual.effect.glow";
        case Kind::bloom: return "visual.effect.bloom";
        case Kind::halation: return "visual.effect.halation";
        case Kind::lensDistortion: return "visual.effect.lens-distortion";
        case Kind::denoise: return "visual.effect.denoise";
        case Kind::directionalBlur: return "visual.effect.directional-blur";
        case Kind::unsharpMask: return "visual.effect.unsharp-mask";
        case Kind::highPassSharpen: return "visual.effect.high-pass-sharpen";
    }
    return "";
}

inline constexpr std::optional<Kind> kindForWireName(std::string_view name) noexcept
{
    for (std::size_t index = 0; index < kKindCount; ++index)
    {
        const auto kind = static_cast<Kind>(index);
        if (name == wireName(kind)) return kind;
    }
    return std::nullopt;
}

inline constexpr std::optional<Kind> kindForGraphName(std::string_view name) noexcept
{
    for (std::size_t index = 0; index < kKindCount; ++index)
    {
        const auto kind = static_cast<Kind>(index);
        if (name == graphKindName(kind)) return kind;
    }
    return std::nullopt;
}

inline constexpr std::size_t parameterCount(Kind kind) noexcept
{
    if (!isKnownKind(kind)) return 0;
    switch (kind)
    {
        case Kind::bloom:
        case Kind::halation:
        case Kind::unsharpMask: return 3;
        case Kind::vignette:
        case Kind::glow:
        case Kind::lensDistortion:
        case Kind::denoise:
        case Kind::directionalBlur:
        case Kind::highPassSharpen: return 2;
        case Kind::sharpen:
        case Kind::saturation:
        case Kind::exposure:
        case Kind::brightness:
        case Kind::contrast:
        case Kind::hue:
        case Kind::gamma:
        case Kind::blur:
        default: return 1;
    }
}

inline constexpr const char* parameterName(Kind kind, std::size_t index = 0) noexcept
{
    if (index >= parameterCount(kind)) return "";
    switch (kind)
    {
        case Kind::vignette: return index == 0 ? "amount" : "softness";
        case Kind::glow: return index == 0 ? "intensity" : "radius";
        case Kind::bloom:
            return index == 0 ? "threshold" : index == 1 ? "intensity" : "radius";
        case Kind::halation:
            return index == 0 ? "threshold" : index == 1 ? "radius" : "intensity";
        case Kind::lensDistortion:
            return index == 0 ? "distortion" : "chromaticAberration";
        case Kind::denoise: return index == 0 ? "strength" : "radius";
        case Kind::directionalBlur: return index == 0 ? "radius" : "angle";
        case Kind::unsharpMask:
            return index == 0 ? "amount" : index == 1 ? "radius" : "threshold";
        case Kind::highPassSharpen: return index == 0 ? "amount" : "radius";
        case Kind::exposure: return "stops";
        case Kind::hue: return "shift";
        case Kind::gamma: return "value";
        case Kind::blur: return "radius";
        case Kind::sharpen:
        case Kind::saturation:
        case Kind::brightness:
        case Kind::contrast:
        default: return "amount";
    }
}

inline constexpr double minimum(Kind kind, std::size_t index = 0) noexcept
{
    if (index >= parameterCount(kind)) return 0.0;
    switch (kind)
    {
        case Kind::lensDistortion: return index == 0 ? -1.0 : 0.0;
        case Kind::directionalBlur: return index == 0 ? 0.0 : -180.0;
        case Kind::denoise: return index == 0 ? 0.0 : 1.0;
        case Kind::unsharpMask:
        case Kind::highPassSharpen: return index == 0 ? 0.0 : index == 1 ? 0.5 : 0.0;
        case Kind::exposure: return -4.0;
        case Kind::brightness: return -1.0;
        case Kind::hue: return -180.0;
        case Kind::gamma: return 0.1;
        case Kind::sharpen:
        case Kind::saturation:
        case Kind::contrast:
        case Kind::blur:
        case Kind::vignette:
        case Kind::glow:
        case Kind::bloom:
        case Kind::halation:
        default: return 0.0;
    }
}

inline constexpr double maximum(Kind kind, std::size_t index = 0) noexcept
{
    if (index >= parameterCount(kind)) return 0.0;
    switch (kind)
    {
        case Kind::vignette: return 1.0;
        case Kind::glow: return index == 0 ? 3.0 : 20.0;
        case Kind::bloom:
            return index == 0 ? 2.0 : index == 1 ? 3.0 : 20.0;
        case Kind::halation:
            return index == 0 ? 2.0 : index == 1 ? 20.0 : 2.0;
        case Kind::lensDistortion: return index == 0 ? 1.0 : 0.05;
        case Kind::denoise: return index == 0 ? 1.0 : 4.0;
        case Kind::directionalBlur: return index == 0 ? 20.0 : 180.0;
        case Kind::unsharpMask:
        case Kind::highPassSharpen: return index == 0 ? 5.0 : index == 1 ? 10.0 : 1.0;
        case Kind::sharpen: return 5.0;
        case Kind::saturation:
        case Kind::contrast:
        case Kind::gamma: return 3.0;
        case Kind::exposure: return 4.0;
        case Kind::brightness: return 1.0;
        case Kind::hue: return 180.0;
        case Kind::blur: return 20.0;
    }
    return 0.0;
}

inline constexpr double defaultValue(Kind kind, std::size_t index = 0) noexcept
{
    if (index >= parameterCount(kind)) return 0.0;
    switch (kind)
    {
        case Kind::vignette: return index == 0 ? 0.0 : 0.5;
        case Kind::glow: return index == 0 ? 0.5 : 6.0;
        case Kind::bloom: return index == 0 ? 0.8 : index == 1 ? 0.5 : 6.0;
        case Kind::halation: return index == 0 ? 0.7 : index == 1 ? 8.0 : 0.35;
        case Kind::lensDistortion: return 0.0;
        case Kind::denoise: return index == 0 ? 0.5 : 2.0;
        case Kind::directionalBlur: return index == 0 ? 6.0 : 0.0;
        case Kind::unsharpMask:
            return index == 0 ? 1.0 : index == 1 ? 1.5 : 0.0;
        case Kind::highPassSharpen: return index == 0 ? 1.0 : 2.0;
        case Kind::sharpen:
        case Kind::saturation:
        case Kind::exposure:
        case Kind::brightness:
        case Kind::contrast:
        case Kind::hue:
        case Kind::gamma:
        case Kind::blur:
        default: break;
    }
    return kind == Kind::saturation || kind == Kind::contrast || kind == Kind::gamma
        ? 1.0 : 0.0;
}

// Stable EffectType wire values consumed by FrameRenderer's existing rack.
inline constexpr int rendererEffectType(Kind kind) noexcept
{
    switch (kind)
    {
        case Kind::brightness: return 0;
        case Kind::contrast: return 1;
        case Kind::saturation: return 2;
        case Kind::hue: return 3;
        case Kind::exposure: return 4;
        case Kind::gamma: return 5;
        case Kind::blur: return 6;
        case Kind::sharpen: return 7;
        case Kind::vignette: return 8;
        case Kind::glow: return 29;
        case Kind::bloom: return 30;
        case Kind::halation: return 31;
        case Kind::lensDistortion: return 32;
        case Kind::denoise: return 33;
        case Kind::directionalBlur: return 34;
        case Kind::unsharpMask: return 35;
        case Kind::highPassSharpen: return 36;
    }
    return -1;
}

struct Description
{
    std::uint32_t schemaVersion = kSchemaVersion;
    Kind kind = Kind::sharpen;
    std::array<double, kMaximumParameterCount> parameters {};
};

enum class AdmissionFailure : std::uint8_t
{
    none = 0,
    unsupportedSchema,
    invalidParameter,
    productionBackendUnavailable
};

struct BackendSupport
{
    bool sharpen = false;
    bool saturation = false;
    bool exposure = false;
    bool brightness = false;
    bool contrast = false;
    bool hue = false;
    bool gamma = false;
    bool blur = false;
    bool vignette = false;
    bool glow = false;
    bool bloom = false;
    bool halation = false;
    bool lensDistortion = false;
    bool denoise = false;
    bool directionalBlur = false;
    bool unsharpMask = false;
    bool highPassSharpen = false;
};

inline constexpr bool supports(BackendSupport backend, Kind kind) noexcept
{
    switch (kind)
    {
        case Kind::sharpen: return backend.sharpen;
        case Kind::saturation: return backend.saturation;
        case Kind::exposure: return backend.exposure;
        case Kind::brightness: return backend.brightness;
        case Kind::contrast: return backend.contrast;
        case Kind::hue: return backend.hue;
        case Kind::gamma: return backend.gamma;
        case Kind::blur: return backend.blur;
        case Kind::vignette: return backend.vignette;
        case Kind::glow: return backend.glow;
        case Kind::bloom: return backend.bloom;
        case Kind::halation: return backend.halation;
        case Kind::lensDistortion: return backend.lensDistortion;
        case Kind::denoise: return backend.denoise;
        case Kind::directionalBlur: return backend.directionalBlur;
        case Kind::unsharpMask: return backend.unsharpMask;
        case Kind::highPassSharpen: return backend.highPassSharpen;
    }
    return false;
}

class AdmittedCommonEffect final
{
public:
    const Description& description() const noexcept { return description_; }

private:
    friend std::optional<AdmittedCommonEffect> admitCommonEffect(
        const Description&, BackendSupport, AdmissionFailure&) noexcept;
    explicit AdmittedCommonEffect(Description description) noexcept : description_(description) {}
    Description description_;
};

inline std::optional<AdmittedCommonEffect> admitCommonEffect(
    const Description& description, BackendSupport backend,
    AdmissionFailure& failure) noexcept
{
    failure = AdmissionFailure::none;
    if (description.schemaVersion != kSchemaVersion)
    {
        failure = AdmissionFailure::unsupportedSchema;
        return std::nullopt;
    }
    const auto count = parameterCount(description.kind);
    if (count == 0)
    {
        failure = AdmissionFailure::invalidParameter;
        return std::nullopt;
    }
    for (std::size_t index = 0; index < count; ++index)
        if (!std::isfinite(description.parameters[index])
            || description.parameters[index] < minimum(description.kind, index)
            || description.parameters[index] > maximum(description.kind, index))
        {
            failure = AdmissionFailure::invalidParameter;
            return std::nullopt;
        }
    if (!supports(backend, description.kind))
    {
        failure = AdmissionFailure::productionBackendUnavailable;
        return std::nullopt;
    }
    return AdmittedCommonEffect(description);
}

inline constexpr BackendSupport currentProductionBackendSupport() noexcept
{
    // These are existing FrameRenderer operations on both native production
    // backends. This contract does not admit detached filters or imply broader
    // color-management support.
    return { true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true };
}
} // namespace commoneffect
