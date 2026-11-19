#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>

namespace videowire::keycleanup
{
inline constexpr std::uint32_t kSchemaVersion = 1;
inline constexpr double kDefaultTolerance = 0.18;
inline constexpr double kDefaultSoftness = 0.10;
inline constexpr double kDefaultDespill = 0.50;
inline constexpr double kMaximumFeatherPixels = 4.0;
inline constexpr double kMaximumAbsoluteChoke = 1.0;

struct FrameDescriptor final
{
    std::string_view carrier;
    std::string_view dataType;
    std::string_view pixelFormat;
    std::string_view colorSpace;
    std::string_view transfer;
    std::string_view alpha;

    constexpr bool operator==(const FrameDescriptor& other) const noexcept
    {
        return carrier == other.carrier && dataType == other.dataType
            && pixelFormat == other.pixelFormat && colorSpace == other.colorSpace
            && transfer == other.transfer && alpha == other.alpha;
    }
};

inline constexpr FrameDescriptor kFrameDescriptor {
    "frame", "image", "rgba8", "sRGB", "sRGB", "straight"
};

struct Description final
{
    std::uint32_t schemaVersion = kSchemaVersion;
    FrameDescriptor input = kFrameDescriptor;
    FrameDescriptor output = kFrameDescriptor;
    double keyRed = 0.0;
    double keyGreen = 1.0;
    double keyBlue = 0.0;
    double tolerance = kDefaultTolerance;
    double softness = kDefaultSoftness;
    double despill = kDefaultDespill;
    double choke = 0.0;
    double featherPixels = 0.0;
    double edgeRed = 1.0;
    double edgeGreen = 1.0;
    double edgeBlue = 1.0;
    double edgeAmount = 0.0;
    bool matteView = false;
};

struct BackendSupport final
{
    bool chromaKey = false;
    bool despill = false;
    bool choke = false;
    bool feather = false;
    bool edgeColor = false;
    bool matteView = false;
};

// The shared native-GPU key-cleanup pass implements the complete bounded v1
// contract on both production renderer backends. Admission remains explicit so
// a backend cannot silently approximate an authored cleanup control.
inline constexpr BackendSupport currentProductionBackendSupport() noexcept
{
    return { true, true, true, true, true, true };
}

enum class AdmissionFailure : std::uint8_t
{
    none = 0,
    unsupportedSchema,
    inexactFrameDescriptor,
    nonFiniteParameter,
    parameterOutsideBounds,
    productionBackendUnavailable
};

inline constexpr std::string_view token(AdmissionFailure value) noexcept
{
    switch (value)
    {
        case AdmissionFailure::none: return "none";
        case AdmissionFailure::unsupportedSchema: return "unsupported-schema";
        case AdmissionFailure::inexactFrameDescriptor: return "inexact-frame-descriptor";
        case AdmissionFailure::nonFiniteParameter: return "non-finite-parameter";
        case AdmissionFailure::parameterOutsideBounds: return "parameter-outside-bounds";
        case AdmissionFailure::productionBackendUnavailable: return "production-backend-unavailable";
    }
    return "unknown";
}

class AdmittedKeyCleanup final
{
public:
    static constexpr bool allowsCpuProductionFallback = false;

    const Description& description() const noexcept { return description_; }
    const BackendSupport& backendSupport() const noexcept { return backendSupport_; }

private:
    friend std::optional<AdmittedKeyCleanup> admit(
        const Description&, const BackendSupport&, AdmissionFailure&) noexcept;

    AdmittedKeyCleanup(Description description, BackendSupport backendSupport) noexcept
        : description_(description), backendSupport_(backendSupport)
    {
    }

    Description description_;
    BackendSupport backendSupport_;
};

inline std::optional<AdmittedKeyCleanup> admit(
    const Description& source, const BackendSupport& backend,
    AdmissionFailure& failure) noexcept
{
    failure = AdmissionFailure::none;
    if (source.schemaVersion != kSchemaVersion)
    {
        failure = AdmissionFailure::unsupportedSchema;
        return std::nullopt;
    }
    if (!(source.input == kFrameDescriptor) || !(source.output == kFrameDescriptor))
    {
        failure = AdmissionFailure::inexactFrameDescriptor;
        return std::nullopt;
    }

    const double values[] {
        source.keyRed, source.keyGreen, source.keyBlue, source.tolerance,
        source.softness, source.despill, source.choke, source.featherPixels,
        source.edgeRed, source.edgeGreen, source.edgeBlue, source.edgeAmount
    };
    for (const double value : values)
        if (!std::isfinite(value))
        {
            failure = AdmissionFailure::nonFiniteParameter;
            return std::nullopt;
        }

    const auto unit = [](double value) { return value >= 0.0 && value <= 1.0; };
    if (!unit(source.keyRed) || !unit(source.keyGreen) || !unit(source.keyBlue)
        || !unit(source.tolerance) || !unit(source.softness) || !unit(source.despill)
        || source.choke < -kMaximumAbsoluteChoke || source.choke > kMaximumAbsoluteChoke
        || source.featherPixels < 0.0 || source.featherPixels > kMaximumFeatherPixels
        || !unit(source.edgeRed) || !unit(source.edgeGreen) || !unit(source.edgeBlue)
        || !unit(source.edgeAmount))
    {
        failure = AdmissionFailure::parameterOutsideBounds;
        return std::nullopt;
    }

    if (!backend.chromaKey || (source.despill != 0.0 && !backend.despill)
        || (source.choke != 0.0 && !backend.choke)
        || (source.featherPixels != 0.0 && !backend.feather)
        || (source.edgeAmount != 0.0 && !backend.edgeColor)
        || (source.matteView && !backend.matteView))
    {
        failure = AdmissionFailure::productionBackendUnavailable;
        return std::nullopt;
    }

    Description canonical = source;
    if (canonical.choke == 0.0) canonical.choke = 0.0;
    if (canonical.featherPixels == 0.0) canonical.featherPixels = 0.0;
    if (canonical.edgeAmount == 0.0) canonical.edgeAmount = 0.0;
    return AdmittedKeyCleanup(canonical, backend);
}
} // namespace videowire::keycleanup
