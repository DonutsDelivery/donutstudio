#include "../../shared/ColorTransformContract.h"
#include "../../shared/DecodedFrameColorContract.h"
#include "../../shared/ColorTransformOperationContract.h"
#include "../../shared/ManagedColorTransformOperationContract.h"
#include "../../shared/ManagedVideoSourceColorPolicy.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>

using namespace colortransform;

namespace
{
[[noreturn]] void fail(const char* message)
{
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const char* message)
{
    if (! condition)
        fail(message);
}

Description sdrTransform()
{
    Description value;
    value.input = { { 1920, 1080 }, PixelFormat::RGBA8, ColorSpace::SRGB,
                    TransferFunction::SRGB, AlphaMode::Premultiplied };
    value.output = { value.input.extent, PixelFormat::RGBA8, ColorSpace::DisplayP3,
                     TransferFunction::SRGB, AlphaMode::Premultiplied };
    value.workingColorSpace = ColorSpace::Rec2020;
    value.workingFormat = PixelFormat::RGBA16F;
    value.outputIntent = OutputIntent::SdrDisplay;
    value.toneMap = ToneMap::None;
    value.luminance = { 100.0, 100.0, 100.0, 100.0 };
    return value;
}

Description hdrTransform()
{
    Description value;
    value.input = { { 3840, 2160 }, PixelFormat::RGBA16F, ColorSpace::Rec2020,
                    TransferFunction::PQ, AlphaMode::Opaque };
    value.output = { value.input.extent, PixelFormat::RGBA16F, ColorSpace::Rec2020,
                     TransferFunction::HLG, AlphaMode::Opaque };
    value.workingColorSpace = ColorSpace::Rec2020;
    value.workingFormat = PixelFormat::RGBA16F;
    value.outputIntent = OutputIntent::HdrHlg;
    value.toneMap = ToneMap::None;
    value.luminance = { 203.0, 1000.0, 203.0, 1000.0 };
    return value;
}

Description hdrToSdrTransform()
{
    auto value = hdrTransform();
    value.output = { value.input.extent, PixelFormat::RGBA8, ColorSpace::Rec709,
                     TransferFunction::Gamma22, AlphaMode::Opaque };
    value.outputIntent = OutputIntent::SdrDisplay;
    value.toneMap = ToneMap::Reinhard;
    value.luminance.outputReferenceWhiteNits = 100.0;
    value.luminance.outputPeakNits = 100.0;
    return value;
}

AdmittedTransform requireAdmitted(const Description& value,
                                  const AdmissionLimits& limits = {})
{
    AdmissionFailure failure = AdmissionFailure::None;
    const auto admitted = admit(value, BackendCapability::NativeGpu, failure, limits);
    require(admitted.has_value() && failure == AdmissionFailure::None,
            "valid transform was rejected");
    return *admitted;
}

void requireRejected(const Description& value,
                     AdmissionFailure expected,
                     const char* message,
                     BackendCapability backend = BackendCapability::NativeGpu,
                     const AdmissionLimits& limits = {})
{
    AdmissionFailure failure = AdmissionFailure::None;
    const auto admitted = admit(value, backend, failure, limits);
    require(! admitted.has_value() && failure == expected, message);
}

// Test-only scalar reference oracle. Production admission never exposes these
// functions and never authorizes this code as a pixel-processing fallback.
double decodeSrgb(double encoded)
{
    return encoded <= 0.04045 ? encoded / 12.92
                              : std::pow((encoded + 0.055) / 1.055, 2.4);
}

double encodeSrgb(double linear)
{
    return linear <= 0.0031308 ? linear * 12.92
                               : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

double decodePqNits(double encoded)
{
    constexpr double m1 = 2610.0 / 16384.0;
    constexpr double m2 = 2523.0 / 32.0;
    constexpr double c1 = 3424.0 / 4096.0;
    constexpr double c2 = 2413.0 / 128.0;
    constexpr double c3 = 2392.0 / 128.0;
    const double power = std::pow(encoded, 1.0 / m2);
    return 10000.0 * std::pow(std::max(power - c1, 0.0) / (c2 - c3 * power),
                             1.0 / m1);
}

double encodePqNits(double nits)
{
    constexpr double m1 = 2610.0 / 16384.0;
    constexpr double m2 = 2523.0 / 32.0;
    constexpr double c1 = 3424.0 / 4096.0;
    constexpr double c2 = 2413.0 / 128.0;
    constexpr double c3 = 2392.0 / 128.0;
    const double power = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), m1);
    return std::pow((c1 + c2 * power) / (1.0 + c3 * power), m2);
}

double decodeHlg(double encoded)
{
    constexpr double a = 0.17883277;
    constexpr double b = 0.28466892;
    constexpr double c = 0.55991073;
    return encoded <= 0.5 ? encoded * encoded / 3.0
                          : (std::exp((encoded - c) / a) + b) / 12.0;
}

double encodeHlg(double linear)
{
    constexpr double a = 0.17883277;
    constexpr double b = 0.28466892;
    constexpr double c = 0.55991073;
    return linear <= 1.0 / 12.0 ? std::sqrt(3.0 * linear)
                                : a * std::log(12.0 * linear - b) + c;
}

std::array<double, 4> unassociate(std::array<double, 4> rgba)
{
    if (rgba[3] <= 0.0)
        return { 0.0, 0.0, 0.0, 0.0 };
    rgba[0] /= rgba[3];
    rgba[1] /= rgba[3];
    rgba[2] /= rgba[3];
    return rgba;
}

std::array<double, 4> associate(std::array<double, 4> rgba)
{
    rgba[0] *= rgba[3];
    rgba[1] *= rgba[3];
    rgba[2] *= rgba[3];
    return rgba;
}

bool near(double actual, double expected, double tolerance = 1.0e-10)
{
    return std::abs(actual - expected) <= tolerance;
}
namespace mct = managedcolortransform;
namespace mco = managedcolortransformoperation;
namespace mvs = managedvideosourcecolor;

constexpr std::string_view kManagedHlgScene = R"(<ColorTransform version="2" mathVersion="2" inputFormat="rgba32f" inputColorSpace="rec2020" inputTransfer="hlg" inputAlpha="opaque" outputFormat="rgba16f" outputColorSpace="linearSRGB" outputTransfer="linear" outputAlpha="premultiplied" workingColorSpace="linearSRGB" workingFormat="rgba16f" outputIntent="sceneLinear" toneMap="none" sourceReferenceWhiteNits="203" sourcePeakNits="1000" outputReferenceWhiteNits="203" outputPeakNits="1000" sourceHlgMode="reference-display" outputHlgMode="none" encodedInputPolicy="normalized-reject" terminalRangePolicy="preserve"/>)";
constexpr std::string_view kManagedSdr = R"(<ColorTransform version="2" mathVersion="2" inputFormat="rgba8" inputColorSpace="sRGB" inputTransfer="sRGB" inputAlpha="straight" outputFormat="rgba8" outputColorSpace="displayP3" outputTransfer="sRGB" outputAlpha="premultiplied" workingColorSpace="linearSRGB" workingFormat="rgba16f" outputIntent="sdrDisplay" toneMap="none" sourceReferenceWhiteNits="203" sourcePeakNits="203" outputReferenceWhiteNits="203" outputPeakNits="203" sourceHlgMode="none" outputHlgMode="none" encodedInputPolicy="normalized-reject" terminalRangePolicy="clip-target-sdr"/>)";
constexpr std::string_view kManagedHlgOutput = R"(<ColorTransform version="2" mathVersion="2" inputFormat="rgba32f" inputColorSpace="rec2020" inputTransfer="hlg" inputAlpha="opaque" outputFormat="rgba16f" outputColorSpace="rec2020" outputTransfer="hlg" outputAlpha="opaque" workingColorSpace="linearSRGB" workingFormat="rgba16f" outputIntent="hdrHlg" toneMap="none" sourceReferenceWhiteNits="203" sourcePeakNits="1000" outputReferenceWhiteNits="203" outputPeakNits="2000" sourceHlgMode="reference-display" outputHlgMode="reference-display-inverse" encodedInputPolicy="normalized-reject" terminalRangePolicy="half-cell-target-hdr"/>)";
constexpr std::string_view kManagedPqOutput = R"(<ColorTransform version="2" mathVersion="2" inputFormat="rgba16f" inputColorSpace="linearSRGB" inputTransfer="linear" inputAlpha="opaque" outputFormat="rgba32f" outputColorSpace="rec2020" outputTransfer="pq" outputAlpha="opaque" workingColorSpace="linearSRGB" workingFormat="rgba16f" outputIntent="hdrPq" toneMap="none" sourceReferenceWhiteNits="203" sourcePeakNits="10000" outputReferenceWhiteNits="203" outputPeakNits="4000" sourceHlgMode="none" outputHlgMode="none" encodedInputPolicy="normalized-reject" terminalRangePolicy="reject-target-hdr"/>)";
constexpr std::string_view kLegacySdrWire = R"(<ColorTransform version="1" inputFormat="rgba8" inputColorSpace="sRGB" inputTransfer="sRGB" inputAlpha="premultiplied" outputFormat="rgba8" outputColorSpace="displayP3" outputTransfer="sRGB" outputAlpha="premultiplied" workingColorSpace="rec2020" workingFormat="rgba16f" outputIntent="sdrDisplay" toneMap="none" sourceReferenceWhiteNits="100" sourcePeakNits="100" outputReferenceWhiteNits="100" outputPeakNits="100"/>)";

std::string replaceAttribute(std::string_view wire, std::string_view name, std::string_view value)
{
    std::string result(wire);
    const std::string prefix = std::string(name) + "=\"";
    const auto start = result.find(prefix);
    require(start != std::string::npos, "fixture attribute was not found");
    const auto first = start + prefix.size();
    const auto end = result.find('"', first);
    result.replace(first, end - first, value);
    return result;
}

mct::DescriptionV2 managedDescription(std::string_view wire = kManagedHlgScene)
{
    mct::ValidationFailure failure;
    const auto result = mco::parseForExtent(wire, {1920, 1080}, failure);
    require(result && failure == mct::ValidationFailure::None, "literal managed transform refused");
    return *result;
}

void checkManagedRefusal(const mct::DescriptionV2& value, mct::ValidationFailure expected)
{
    mct::ValidationFailure failure;
    require(! mct::validate(value, failure) && failure == expected,
            "managed semantic refusal reason changed");
    require(! mct::identityFor(value, failure) && failure == expected,
            "invalid or unbound managed description acquired identity");
}

void testDormantManagedTransform()
{
    static_assert(! std::is_convertible_v<mct::DescriptionV2, AdmittedTransform>);
    static_assert(! std::is_constructible_v<AdmittedTransform, mct::DescriptionV2>);
    static_assert(! std::is_convertible_v<mco::ParsedTransformV2, AdmittedTransform>);
    static_assert(! std::is_constructible_v<AdmittedTransform, mco::ParsedTransformV2>);
    static_assert(! std::is_convertible_v<mct::SemanticIdentity, CacheIdentity>);
    static_assert(sizeof(mct::SemanticIdentity::bytes) == 64);
    for (const auto wire : {kManagedHlgScene, kManagedSdr, kManagedHlgOutput, kManagedPqOutput})
    {
        mco::ParsedTransformV2 parsed;
        require(mco::parse(wire, parsed) && mco::serialize(parsed) == wire,
                "independent literal managed XML is not canonical");
    }
    const auto value = managedDescription();
    require(value.version == 2 && value.mathVersion == 2
        && value.input.extent == Extent{1920,1080} && value.output.extent == Extent{1920,1080}
        && value.input.format == PixelFormat::RGBA32F && value.input.colorSpace == ColorSpace::Rec2020
        && value.input.transfer == TransferFunction::HLG && value.input.alpha == AlphaMode::Opaque
        && value.output.format == PixelFormat::RGBA16F && value.output.colorSpace == ColorSpace::LinearSRGB
        && value.output.transfer == TransferFunction::Linear && value.output.alpha == AlphaMode::Premultiplied
        && value.workingColorSpace == ColorSpace::LinearSRGB && value.workingFormat == PixelFormat::RGBA16F
        && value.outputIntent == OutputIntent::SceneLinear && value.toneMap == ToneMap::None
        && value.luminance.sourceReferenceWhiteNits == 203.0 && value.luminance.sourcePeakNits == 1000.0
        && value.luminance.outputReferenceWhiteNits == 203.0 && value.luminance.outputPeakNits == 1000.0
        && value.sourceHlgMode == mct::SourceHlgMode::ReferenceDisplay
        && value.outputHlgMode == mct::OutputHlgMode::None
        && value.encodedInputPolicy == mct::EncodedInputPolicy::NormalizedReject
        && value.terminalRangePolicy == mct::TerminalRangePolicy::Preserve,
        "complete v2 projection lost a field");
    const std::array<std::uint8_t,64> literalKey {
        0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x00,0x00,0x07,0x80,0x00,0x00,0x04,0x38,
        0x05,0x05,0x05,0x01,0x04,0x01,0x01,0x03,0x01,0x04,0x00,0x00,
        0x40,0x69,0x60,0x00,0x00,0x00,0x00,0x00,0x40,0x8f,0x40,0x00,0x00,0x00,0x00,0x00,
        0x40,0x69,0x60,0x00,0x00,0x00,0x00,0x00,0x40,0x8f,0x40,0x00,0x00,0x00,0x00,0x00,
        0x01,0x00,0x00,0x00 };
    mct::ValidationFailure failure;
    const auto identity = mct::identityFor(value, failure);
    require(identity && identity->bytes == literalKey, "all 64 managed identity bytes changed");
    const auto delta = [&] (const mct::DescriptionV2& changed, std::size_t first, std::size_t length) {
        const auto key = mct::identityFor(changed, failure);
        require(key && *key != *identity, "changed managed semantics reused full identity");
        bool changedBytes = false;
        for (std::size_t i = first; i < first + length; ++i)
            changedBytes = changedBytes || key->bytes[i] != identity->bytes[i];
        require(changedBytes, "changed semantic field did not reach its identity bytes");
    };
    auto changed = value; changed.input.extent.width = changed.output.extent.width = 1919; delta(changed,8,4);
    changed = value; changed.input.extent.height = changed.output.extent.height = 1079; delta(changed,12,4);
    changed = value; changed.input.format = PixelFormat::RGBA16F; delta(changed,16,1);
    changed = value; changed.input.colorSpace = ColorSpace::DisplayP3;
    changed.input.transfer = TransferFunction::Linear; changed.sourceHlgMode = mct::SourceHlgMode::None;
    delta(changed,17,1); delta(changed,18,1); delta(changed,60,1);
    changed = value; changed.input.alpha = AlphaMode::Straight; delta(changed,19,1);
    changed = value; changed.output.format = PixelFormat::RGBA32F; delta(changed,20,1);
    changed = value; changed.output.colorSpace = ColorSpace::DisplayP3; delta(changed,21,1);
    changed = value; changed.output.alpha = AlphaMode::Straight; delta(changed,23,1);
    changed = value; changed.output = {{1920,1080},PixelFormat::RGBA16F,ColorSpace::Rec2020,
        TransferFunction::HLG,AlphaMode::Opaque}; changed.outputIntent = OutputIntent::HdrHlg;
    changed.outputHlgMode = mct::OutputHlgMode::ReferenceDisplayInverse;
    changed.terminalRangePolicy = mct::TerminalRangePolicy::HalfCellTargetHdr;
    delta(changed,22,1); delta(changed,26,1); delta(changed,61,1); delta(changed,63,1);
    const auto hdrKey = mct::identityFor(changed, failure);
    changed.terminalRangePolicy = mct::TerminalRangePolicy::RejectTargetHdr;
    const auto strictKey = mct::identityFor(changed, failure);
    require(hdrKey && strictKey && *hdrKey != *strictKey && hdrKey->bytes[63] == 3 && strictKey->bytes[63] == 2,
        "independent HDR range choices lack distinct identities");
    changed = value; changed.luminance.sourceReferenceWhiteNits = 100.0; delta(changed,28,8);
    changed = value; changed.luminance.sourcePeakNits = 2000.0; delta(changed,36,8);
    changed = value; changed.luminance.outputPeakNits = 2000.0; delta(changed,52,8);
    changed = value; changed.output = {{1920,1080},PixelFormat::RGBA8,ColorSpace::SRGB,
        TransferFunction::SRGB,AlphaMode::Opaque}; changed.outputIntent = OutputIntent::SdrDisplay;
    changed.toneMap = ToneMap::Reinhard; changed.terminalRangePolicy = mct::TerminalRangePolicy::ClipTargetSdr;
    changed.luminance.outputReferenceWhiteNits = changed.luminance.outputPeakNits = 100.0;
    delta(changed,27,1); delta(changed,44,8);
    auto differentLastByte = *identity;
    differentLastByte.bytes[63] ^= 1;
    require(differentLastByte != *identity, "identity equality omitted last policy byte");
    for (std::size_t i = 0; i < literalKey.size(); ++i)
    {
        auto distinct = *identity; distinct.bytes[i] ^= 1;
        require(distinct != *identity, "full identity equality omitted a byte");
    }

    for (const auto bad : {0u,16385u,std::numeric_limits<std::uint32_t>::max()})
    {
        changed = value; changed.input.extent.width = changed.output.extent.width = bad;
        checkManagedRefusal(changed,mct::ValidationFailure::InvalidExtent);
        changed = value; changed.input.extent.height = changed.output.extent.height = bad;
        checkManagedRefusal(changed,mct::ValidationFailure::InvalidExtent);
    }
    changed = value; changed.input.extent = changed.output.extent = {16384,16384};
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidExtent);
    changed = value; changed.output.extent.width = 1;
    checkManagedRefusal(changed,mct::ValidationFailure::ExtentMismatch);
    changed = value; changed.input.extent = changed.output.extent = {8192,8192};
    require(mct::identityFor(changed,failure).has_value(), "hard pixel boundary refused");
    changed = value; changed.input.extent = changed.output.extent = {16384,4096};
    require(mct::identityFor(changed,failure).has_value(), "hard dimension boundary refused");
    changed = value; changed.input.extent = changed.output.extent = {16384,4097};
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidExtent);
    for (const auto version : {0u,1u,3u})
    {
        changed = value; changed.version = version;
        checkManagedRefusal(changed,mct::ValidationFailure::UnsupportedVersion);
        changed = value; changed.mathVersion = version;
        checkManagedRefusal(changed,mct::ValidationFailure::UnsupportedVersion);
    }
    for (const auto format : {PixelFormat::Unspecified,PixelFormat::R8,PixelFormat::R16,static_cast<PixelFormat>(255)})
    {
        changed = value; changed.input.format = format;
        checkManagedRefusal(changed,mct::ValidationFailure::UnsupportedPixelFormat);
        changed = value; changed.output.format = format;
        checkManagedRefusal(changed,mct::ValidationFailure::UnsupportedPixelFormat);
    }
    changed = value; changed.input.format = PixelFormat::RGBA8;
    checkManagedRefusal(changed,mct::ValidationFailure::UnsupportedPixelFormat);
    changed = value; changed.workingColorSpace = ColorSpace::Rec2020;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidWorkingSpace);
    changed = value; changed.workingFormat = PixelFormat::RGBA32F;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidWorkingFormat);
    changed = value; changed.outputIntent = static_cast<OutputIntent>(255);
    checkManagedRefusal(changed,mct::ValidationFailure::UnknownOutputIntent);
    changed = value; changed.toneMap = static_cast<ToneMap>(255);
    checkManagedRefusal(changed,mct::ValidationFailure::UnknownToneMap);
    for (const auto alpha : {AlphaMode::Unspecified,static_cast<AlphaMode>(255)})
    {
        changed = value; changed.input.alpha = alpha;
        checkManagedRefusal(changed,mct::ValidationFailure::UnknownAlphaMode);
        changed = value; changed.output.alpha = alpha;
        checkManagedRefusal(changed,mct::ValidationFailure::UnknownAlphaMode);
    }
    changed = value; changed.input.alpha = AlphaMode::Straight; changed.output.alpha = AlphaMode::Opaque;
    checkManagedRefusal(changed,mct::ValidationFailure::AlphaWouldBeDiscarded);
    for (const auto bad : {0.0,-1.0,79.0,501.0,10001.0})
    {
        changed = value; changed.luminance.sourceReferenceWhiteNits = bad;
        checkManagedRefusal(changed,mct::ValidationFailure::LuminanceOutOfRange);
        changed = value; changed.luminance.outputReferenceWhiteNits = bad;
        checkManagedRefusal(changed,mct::ValidationFailure::LuminanceOutOfRange);
    }
    for (const auto bad : {0.0,-1.0,202.0,399.0,10001.0})
    {
        changed = value; changed.luminance.sourcePeakNits = bad;
        checkManagedRefusal(changed,mct::ValidationFailure::LuminanceOutOfRange);
    }
    for (const auto bad : {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()})
    {
        for (auto member : {&mct::LuminanceParametersV2::sourceReferenceWhiteNits,
                            &mct::LuminanceParametersV2::sourcePeakNits,
                            &mct::LuminanceParametersV2::outputReferenceWhiteNits,
                            &mct::LuminanceParametersV2::outputPeakNits})
        {
            changed = value; changed.luminance.*member = bad;
            checkManagedRefusal(changed,mct::ValidationFailure::NonFiniteLuminance);
        }
    }
    changed = value; changed.sourceHlgMode = mct::SourceHlgMode::None;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidHlgMode);
    changed = value; changed.outputHlgMode = mct::OutputHlgMode::ReferenceDisplayInverse;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidHlgMode);
    changed = value; changed.encodedInputPolicy = static_cast<mct::EncodedInputPolicy>(1);
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidEncodedInputPolicy);
    changed = value; changed.terminalRangePolicy = static_cast<mct::TerminalRangePolicy>(255);
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidTerminalRangePolicy);
    changed = value; changed.terminalRangePolicy = mct::TerminalRangePolicy::ClipTargetSdr;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidOutputIntent);
    changed = value; changed.toneMap = ToneMap::Reinhard;
    checkManagedRefusal(changed,mct::ValidationFailure::ToneMapNotAllowed);
    changed = managedDescription(kManagedSdr); changed.luminance.sourcePeakNits = 1000.0;
    checkManagedRefusal(changed,mct::ValidationFailure::ToneMapRequired);
    changed = managedDescription(kManagedSdr); changed.toneMap = ToneMap::Reinhard;
    checkManagedRefusal(changed,mct::ValidationFailure::ToneMapNotAllowed);
    changed = managedDescription(kManagedSdr); changed.output.transfer = TransferFunction::Linear;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidOutputIntent);
    changed = managedDescription(kManagedSdr); changed.luminance.outputPeakNits = 204.0;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidOutputIntent);
    changed = managedDescription(kManagedHlgOutput); changed.luminance.outputPeakNits = 399.0;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidOutputIntent);
    changed = managedDescription(kManagedHlgOutput); changed.terminalRangePolicy = mct::TerminalRangePolicy::Preserve;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidOutputIntent);
    changed = managedDescription(kManagedPqOutput); changed.outputHlgMode = mct::OutputHlgMode::ReferenceDisplayInverse;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidHlgMode);
    changed = managedDescription(kManagedHlgOutput); changed.outputHlgMode = mct::OutputHlgMode::None;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidHlgMode);
    changed = managedDescription(kManagedSdr); changed.sourceHlgMode = mct::SourceHlgMode::ReferenceDisplay;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidHlgMode);
    changed = value; changed.sourceHlgMode = static_cast<mct::SourceHlgMode>(255);
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidHlgMode);
    changed = value; changed.outputHlgMode = static_cast<mct::OutputHlgMode>(255);
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidHlgMode);
    changed = value; changed.output.colorSpace = ColorSpace::Unspecified;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidColorTransferPair);
    changed = value; changed.input.transfer = static_cast<TransferFunction>(255);
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidColorTransferPair);
    changed = managedDescription(kManagedHlgOutput); changed.output.alpha = AlphaMode::Straight;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidOutputIntent);
    changed = managedDescription(kManagedHlgOutput); changed.output.format = PixelFormat::RGBA8;
    checkManagedRefusal(changed,mct::ValidationFailure::UnsupportedPixelFormat);
    changed = managedDescription(kManagedHlgOutput); changed.toneMap = ToneMap::Reinhard;
    checkManagedRefusal(changed,mct::ValidationFailure::ToneMapNotAllowed);
    changed = value; changed.luminance.outputReferenceWhiteNits = 100.0;
    checkManagedRefusal(changed,mct::ValidationFailure::InvalidOutputIntent);
    for (const auto bad : {0.0,-1.0,202.0,10001.0})
    {
        changed = value; changed.luminance.outputPeakNits = bad;
        checkManagedRefusal(changed,mct::ValidationFailure::LuminanceOutOfRange);
    }
    const auto independentHlg = managedDescription(kManagedHlgOutput);
    require(independentHlg.luminance.sourcePeakNits == 1000.0
        && independentHlg.luminance.outputPeakNits == 2000.0
        && independentHlg.sourceHlgMode == mct::SourceHlgMode::ReferenceDisplay
        && independentHlg.outputHlgMode == mct::OutputHlgMode::ReferenceDisplayInverse,
        "source and output HLG policies were conflated");

    // Independent exhaustive pair roster. This expected table does not call v1 validity.
    constexpr std::array spaces {ColorSpace::LinearSRGB,ColorSpace::SRGB,ColorSpace::DisplayP3,
                                ColorSpace::Rec709,ColorSpace::Rec2020};
    constexpr std::array transfers {TransferFunction::Linear,TransferFunction::SRGB,TransferFunction::Gamma22,
                                   TransferFunction::PQ,TransferFunction::HLG,TransferFunction::Rec709};
    constexpr bool expected[5][6] {
        {true,false,false,false,false,false}, {false,true,false,false,false,false},
        {true,true,false,false,false,false}, {false,false,true,false,false,true},
        {true,false,false,true,true,false} };
    for (std::size_t s = 0; s < spaces.size(); ++s)
        for (std::size_t t = 0; t < transfers.size(); ++t)
        {
            require(mct::validColorTransfer(spaces[s],transfers[t]) == expected[s][t],
                "managed color/transfer pair roster changed");
            changed = value; changed.input.colorSpace = spaces[s]; changed.input.transfer = transfers[t];
            changed.sourceHlgMode = transfers[t] == TransferFunction::HLG
                ? mct::SourceHlgMode::ReferenceDisplay : mct::SourceHlgMode::None;
            if (expected[s][t]) require(mct::validate(changed,failure), "valid managed input pair refused");
            else checkManagedRefusal(changed,mct::ValidationFailure::InvalidColorTransferPair);
        }
    require(! mct::validColorTransfer(ColorSpace::Unspecified,TransferFunction::Linear)
        && ! mct::validColorTransfer(ColorSpace::SRGB,static_cast<TransferFunction>(255)),
        "unknown color or transfer admitted");

    mco::ParsedTransformV2 parsed;
    const auto refuses = [&] (std::string_view wire) {
        require(! mco::parse(wire,parsed), "noncanonical or invalid managed wire accepted");
    };
    constexpr std::array names {"version","mathVersion","inputFormat","inputColorSpace","inputTransfer","inputAlpha",
        "outputFormat","outputColorSpace","outputTransfer","outputAlpha","workingColorSpace","workingFormat",
        "outputIntent","toneMap","sourceReferenceWhiteNits","sourcePeakNits","outputReferenceWhiteNits",
        "outputPeakNits","sourceHlgMode","outputHlgMode","encodedInputPolicy","terminalRangePolicy"};
    for (const auto name : names)
    {
        const std::string start = " " + std::string(name) + "=\"";
        const auto begin = kManagedHlgScene.find(start);
        const auto end = kManagedHlgScene.find('"',begin + start.size()) + 1;
        std::string omitted(kManagedHlgScene); omitted.erase(begin,end-begin); refuses(omitted);
        std::string duplicate(kManagedHlgScene); duplicate.insert(end,kManagedHlgScene.substr(begin,end-begin)); refuses(duplicate);
        std::string reordered(kManagedHlgScene); const auto attribute = reordered.substr(begin,end-begin);
        reordered.erase(begin,end-begin); reordered.insert(15,attribute); // First field stays first only for version.
        if (name == std::string_view("version")) reordered.insert(15," ");
        refuses(reordered);
        refuses(replaceAttribute(kManagedHlgScene,name,"future"));
        refuses(replaceAttribute(kManagedHlgScene,name,""));
    }
    for (const auto name : {"sourceReferenceWhiteNits","sourcePeakNits","outputReferenceWhiteNits","outputPeakNits"})
        for (const auto spelling : {"nan","inf","-inf","-1","-0","0","203,0","203.0","0203","+203",
                                    " 203","203 ","2.03e2","1e9999"})
            refuses(replaceAttribute(kManagedHlgScene,name,spelling));
    for (const auto name : {"version","mathVersion"})
        for (const auto spelling : {"1","3","02","2.0","+2"," 2"})
            refuses(replaceAttribute(kManagedHlgScene,name,spelling));
    refuses(std::string(" ")+std::string(kManagedHlgScene));
    refuses(std::string(kManagedHlgScene)+"\n");
    refuses(std::string(kManagedHlgScene)+"trailing");
    std::string unknown(kManagedHlgScene); unknown.insert(unknown.size()-2," unknown=\"1\""); refuses(unknown);
    std::string spaced(kManagedHlgScene); spaced.insert(15," "); refuses(spaced);
    std::string tabbed(kManagedHlgScene); tabbed[15]='\t'; refuses(tabbed);
    std::string children(kManagedHlgScene); children.replace(children.size()-2,2,"><Child/></ColorTransform>"); refuses(children);
    refuses(std::string(2049,'x'));
    refuses(replaceAttribute(kManagedHlgScene,"inputFormat","r8"));
    refuses(replaceAttribute(kManagedHlgScene,"inputFormat","rgba8"));
    refuses(replaceAttribute(kManagedHlgScene,"inputColorSpace","sRGB"));
    refuses(replaceAttribute(kManagedHlgScene,"inputAlpha","unspecified"));
    refuses(replaceAttribute(kManagedHlgScene,"workingFormat","rgba32f"));
    refuses(replaceAttribute(kManagedHlgScene,"sourceHlgMode","none"));
    refuses(replaceAttribute(kManagedHlgScene,"outputHlgMode","reference-display-inverse"));
    refuses(replaceAttribute(kManagedHlgScene,"terminalRangePolicy","clip-target-sdr"));
    // The 17-digit spelling is selected independently, rather than serializer-derived.
    const auto precise = replaceAttribute(kManagedHlgScene,"sourceReferenceWhiteNits","203.09999999999999");
    require(mco::parse(precise,parsed) && parsed.luminance.sourceReferenceWhiteNits == 203.1
        && mco::serialize(parsed) == precise, "portable precision-17 spelling changed");
    refuses(replaceAttribute(kManagedHlgScene,"sourceReferenceWhiteNits","203.1"));
    mco::ParsedTransformV2 unbound;
    require(mco::parse(kManagedHlgScene,unbound), "unbound parsed representation refused");
    checkManagedRefusal(mco::descriptionForExtent(unbound,{}),mct::ValidationFailure::InvalidExtent);
    require(! mco::parseForExtent(kManagedHlgScene,{},failure), "empty extent acquired bound representation");
    AdmissionFailure legacyFailure;
    require(! colortransformoperation::parseAndAdmit(kManagedSdr,{1920,1080},legacyFailure),
        "old executable admission accepted canonical managed SDR");
    require(! colortransformoperation::parseAndAdmit(replaceAttribute(kLegacySdrWire,"version","2"),
        {1920,1080},legacyFailure) && legacyFailure == AdmissionFailure::UnsupportedVersion,
        "old executable admission accepted legacy-shaped version 2");
}

void testDormantSourcePolicy()
{
    constexpr std::string_view literal = "video-source-color-v2|primaries=rec2020|transfer=hlg|matrix=bt2020-ncl|range=limited|whiteNits=203|peakNits=1000|hlgPeak=1000";
    constexpr std::string_view transferOnly = "video-source-color-v2|primaries=-|transfer=pq|matrix=-|range=-|whiteNits=-|peakNits=-|hlgPeak=-";
    constexpr std::string_view hlgOnly = "video-source-color-v2|primaries=-|transfer=-|matrix=-|range=-|whiteNits=-|peakNits=-|hlgPeak=1000";
    mvs::Policy policy;
    require(mvs::decode(literal,policy) && mvs::encode(policy) == std::optional<std::string>(literal)
        && policy.kind == mvs::PolicyKind::V2 && policy.primaries == ColorSpace::Rec2020
        && policy.transfer == TransferFunction::HLG && policy.matrix == mvs::Matrix::Bt2020Ncl
        && policy.range == mvs::Range::Limited && policy.whiteNits == 203.0
        && policy.peakNits == 1000.0 && policy.hlgPeak == 1000.0,
        "literal seven-field policy projection or bytes changed");
    mvs::Policy absent;
    require(mvs::decode("",absent) && absent.kind == mvs::PolicyKind::Legacy
        && ! absent.legacy.hlgNominalDisplayPeakNits && mvs::encode(absent) == std::optional<std::string>(""),
        "empty policy did not retain old absence bytes");
    for (const auto bytes : {"video-source-color-v1|hlgPeak=400","video-source-color-v1|hlgPeak=1000",
                            "video-source-color-v1|hlgPeak=10000","video-source-color-v1|hlgPeak=1000.1"})
    {
        videosourcecolor::Policy oldPolicy;
        const bool oldAccepted = videosourcecolor::decode(bytes,oldPolicy);
        mvs::Policy passthrough;
        require(mvs::decode(bytes,passthrough) == oldAccepted, "legacy source decoder validity changed");
        if (oldAccepted)
            require(passthrough.kind == mvs::PolicyKind::Legacy
                && passthrough.legacy.hlgNominalDisplayPeakNits == oldPolicy.hlgNominalDisplayPeakNits
                && mvs::encode(passthrough) == std::optional<std::string>(bytes),
                "v1 source policy was rewritten");
    }
    mvs::Policy emptyV2; emptyV2.kind = mvs::PolicyKind::V2;
    require(mvs::encode(emptyV2) == std::optional<std::string>(""), "all-absent policy must persist empty");
    const auto refuses = [&] (std::string_view bytes) {
        mvs::Policy untouched = policy;
        require(! mvs::decode(bytes,untouched) && mvs::encode(untouched) == mvs::encode(policy),
            "invalid source policy accepted or replaced caller result");
    };
    for (const auto bytes : {"video-source-color-v3|hlgPeak=1000","video-source-color-v2",
        "video-source-color-v2|primaries=-|transfer=-|matrix=-|range=-|whiteNits=-|peakNits=-|hlgPeak=-",
        "video-source-color-v1|hlgPeak=1000.0","video-source-color-v1|hlgPeak=nan",
        "video-source-color-v1|hlgPeak=399","video-source-color-v1|hlgPeak=10001"}) refuses(bytes);
    refuses(std::string(513,'x')); refuses(std::string(literal)+" ");
    refuses(std::string(" ")+std::string(literal)); refuses(std::string(literal)+"|unknown=1");
    constexpr std::array names {"primaries","transfer","matrix","range","whiteNits","peakNits","hlgPeak"};
    const auto replaceField = [] (std::string_view bytes, std::string_view name, std::string_view newValue) {
        std::string result(bytes); const auto prefix = "|"+std::string(name)+"=";
        const auto first = result.find(prefix)+prefix.size(); const auto end = result.find('|',first);
        result.replace(first,(end == std::string::npos ? result.size() : end)-first,newValue);
        return result;
    };
    for (const auto name : names)
    {
        const auto prefix = "|"+std::string(name)+"=";
        const auto first = literal.find(prefix); const auto end = literal.find('|',first+1);
        const auto length = (end == std::string_view::npos ? literal.size() : end)-first;
        std::string omitted(literal); omitted.erase(first,length); refuses(omitted);
        std::string duplicate(literal); duplicate.insert(first,literal.substr(first,length)); refuses(duplicate);
        std::string reordered(literal); const auto field = reordered.substr(first,length);
        reordered.erase(first,length); reordered += field;
        if (name == std::string_view("hlgPeak")) reordered += "|hlgPeak=1000";
        refuses(reordered);
        refuses(replaceField(literal,name,"")); refuses(replaceField(literal,name,"future"));
    }
    for (const auto name : {"whiteNits","peakNits","hlgPeak"})
        for (const auto spelling : {"nan","inf","-inf","-1","-0","0","203,0","203.0","0203","+203",
                                    " 203","203 ","2.03e2","1e9999"})
            refuses(replaceField(literal,name,spelling));
    refuses(replaceField(literal,"whiteNits","79")); refuses(replaceField(literal,"whiteNits","501"));
    refuses(replaceField(literal,"peakNits","79")); refuses(replaceField(literal,"peakNits","10001"));
    refuses(replaceField(literal,"hlgPeak","399")); refuses(replaceField(literal,"hlgPeak","10001"));
    const auto precise = replaceField(literal,"whiteNits","203.09999999999999");
    mvs::Policy precisePolicy;
    require(mvs::decode(precise,precisePolicy) && precisePolicy.whiteNits == 203.1
        && mvs::encode(precisePolicy) == std::optional<std::string>(precise), "source portable17 spelling changed");
    refuses(replaceField(literal,"whiteNits","203.1"));

    const mvs::DecodedStorage rgb {mvs::StorageKind::Rgb,PixelFormat::RGBA32F,AlphaMode::Straight};
    const mvs::DecodedStorage yuv {mvs::StorageKind::Yuv,PixelFormat::RGBA32F,AlphaMode::Opaque};
    const mvs::DecodedFacts sdr {ColorSpace::SRGB,TransferFunction::SRGB,mvs::Matrix::Bt709,mvs::Range::Full};
    const mvs::DecodedFacts pq {ColorSpace::Rec2020,TransferFunction::PQ,mvs::Matrix::Bt2020Ncl,mvs::Range::Limited};
    const mvs::DecodedFacts hlg {ColorSpace::Rec2020,TransferFunction::HLG,mvs::Matrix::Bt2020Ncl,mvs::Range::Limited};
    mvs::ResolutionFailure failure;
    const auto rejected = [&] (const mvs::Policy& p, mvs::DecodedFacts f, mvs::DecodedFacts s,
                              mvs::DecodedStorage storage, mvs::ResolutionFailure expected) {
        require(! mvs::resolve(p,f,s,storage,failure) && failure == expected,
            "resolved source refusal reason changed");
    };
    auto resolved = mvs::resolve(absent,sdr,pq,rgb,failure);
    require(resolved && resolved->facts.primaries == ColorSpace::SRGB && resolved->facts.transfer == TransferFunction::SRGB
        && resolved->referenceWhiteNits == 203.0 && resolved->peakNits == 203.0
        && resolved->origins[0] == mvs::FactOrigin::Frame && resolved->origins[1] == mvs::FactOrigin::Frame
        && resolved->authoredBytes.empty(), "frame facts or SDR defaults changed");
    resolved = mvs::resolve(absent,{},pq,yuv,failure);
    require(resolved && resolved->facts.primaries == ColorSpace::Rec2020 && resolved->facts.transfer == TransferFunction::PQ
        && resolved->facts.matrix == mvs::Matrix::Bt2020Ncl && resolved->facts.range == mvs::Range::Limited
        && resolved->referenceWhiteNits == 203.0 && resolved->peakNits == 10000.0
        && resolved->origins == std::array<mvs::FactOrigin,4>{mvs::FactOrigin::Stream,mvs::FactOrigin::Stream,
            mvs::FactOrigin::Stream,mvs::FactOrigin::Stream}, "stream fallback or PQ defaults changed");
    resolved = mvs::resolve(policy,sdr,sdr,yuv,failure);
    require(resolved && resolved->facts.primaries == ColorSpace::Rec2020 && resolved->facts.transfer == TransferFunction::HLG
        && resolved->facts.matrix == mvs::Matrix::Bt2020Ncl && resolved->facts.range == mvs::Range::Limited
        && resolved->origins == std::array<mvs::FactOrigin,4>{mvs::FactOrigin::Interpretation,mvs::FactOrigin::Interpretation,
            mvs::FactOrigin::Interpretation,mvs::FactOrigin::Interpretation}
        && resolved->referenceWhiteNits == 203.0 && resolved->peakNits == 1000.0
        && resolved->hlgMode == mct::SourceHlgMode::ReferenceDisplay && resolved->authoredBytes == literal,
        "seven-field authored interpretation did not override corresponding facts");
    // Each unspecified frame field falls back independently, never as a whole record.
    auto frame = pq; frame.primaries = ColorSpace::Unspecified; frame.range = mvs::Range::Full;
    resolved = mvs::resolve(absent,frame,pq,yuv,failure);
    require(resolved && resolved->origins[0] == mvs::FactOrigin::Stream
        && resolved->origins[1] == mvs::FactOrigin::Frame && resolved->origins[2] == mvs::FactOrigin::Frame
        && resolved->origins[3] == mvs::FactOrigin::Frame && resolved->facts.range == mvs::Range::Full,
        "per-field fallback replaced specified frame facts");
    frame = pq; frame.transfer = TransferFunction::Unspecified; frame.matrix = mvs::Matrix::Unspecified;
    resolved = mvs::resolve(absent,frame,pq,yuv,failure);
    require(resolved && resolved->origins[1] == mvs::FactOrigin::Stream && resolved->origins[2] == mvs::FactOrigin::Stream,
        "transfer/matrix stream fallback failed");
    frame = pq; frame.range = mvs::Range::Unspecified;
    resolved = mvs::resolve(absent,frame,pq,yuv,failure);
    require(resolved && resolved->origins[3] == mvs::FactOrigin::Stream, "range stream fallback failed");
    const mvs::DecodedFacts linear {ColorSpace::DisplayP3,TransferFunction::Linear};
    resolved = mvs::resolve(absent,linear,{},rgb,failure);
    require(resolved && resolved->referenceWhiteNits == 203.0 && resolved->peakNits == 10000.0,
        "external linear white/peak defaults changed");
    mvs::Policy whiteOnly; whiteOnly.kind = mvs::PolicyKind::V2; whiteOnly.whiteNits = 100.0;
    resolved = mvs::resolve(whiteOnly,pq,{},yuv,failure);
    require(resolved && resolved->referenceWhiteNits == 100.0 && resolved->peakNits == 10000.0,
        "PQ reference reporting override changed absent planning peak");
    rejected(absent,{}, {},rgb,mvs::ResolutionFailure::MissingFact);
    frame = pq; frame.primaries = ColorSpace::Unspecified;
    rejected(absent,frame,{},yuv,mvs::ResolutionFailure::MissingFact);
    frame = pq; frame.transfer = TransferFunction::Unspecified;
    rejected(absent,frame,{},yuv,mvs::ResolutionFailure::MissingFact);
    frame = pq; frame.matrix = mvs::Matrix::Unspecified;
    rejected(absent,frame,{},yuv,mvs::ResolutionFailure::MissingFact);
    frame = pq; frame.range = mvs::Range::Unspecified;
    rejected(absent,frame,{},yuv,mvs::ResolutionFailure::MissingFact);
    frame = pq; frame.primaries = ColorSpace::SRGB;
    rejected(absent,frame,pq,yuv,mvs::ResolutionFailure::InvalidColorTransferPair);
    mvs::Policy transferOverride; require(mvs::decode(transferOnly,transferOverride), "partial override grammar refused");
    rejected(transferOverride,sdr,pq,yuv,mvs::ResolutionFailure::InvalidColorTransferPair);
    frame = pq; frame.matrix = mvs::Matrix::Unspecified;
    rejected(transferOverride,frame,{},yuv,mvs::ResolutionFailure::MissingFact);
    rejected(transferOverride,{}, {},rgb,mvs::ResolutionFailure::MissingFact);
    // Authored values may fill exactly their missing fact. Other facts stay required.
    auto fullOverride = policy; fullOverride.transfer = TransferFunction::PQ;
    fullOverride.hlgPeak.reset(); fullOverride.peakNits = 10000.0;
    resolved = mvs::resolve(fullOverride,{}, {},yuv,failure);
    require(resolved && resolved->facts.transfer == TransferFunction::PQ, "complete explicit interpretation refused");
    rejected(policy,sdr,{},rgb,mvs::ResolutionFailure::RgbMatrixRangeOverride);
    auto rangeOnly = whiteOnly; rangeOnly.range = mvs::Range::Full;
    rejected(rangeOnly,sdr,{},rgb,mvs::ResolutionFailure::RgbMatrixRangeOverride);
    frame = pq; frame.matrix = mvs::Matrix::Bt709;
    rejected(absent,frame,pq,yuv,mvs::ResolutionFailure::UnsupportedMatrixRange);
    frame = pq; frame.matrix = static_cast<mvs::Matrix>(255);
    rejected(absent,frame,pq,yuv,mvs::ResolutionFailure::UnsupportedMatrixRange);
    frame = pq; frame.range = static_cast<mvs::Range>(255);
    rejected(absent,frame,pq,yuv,mvs::ResolutionFailure::UnsupportedMatrixRange);
    for (const auto storage : {mvs::DecodedStorage{},
            mvs::DecodedStorage{mvs::StorageKind::Yuv,PixelFormat::R16,AlphaMode::Opaque},
            mvs::DecodedStorage{mvs::StorageKind::Yuv,PixelFormat::RGBA8,AlphaMode::Opaque},
            mvs::DecodedStorage{mvs::StorageKind::Yuv,PixelFormat::RGBA32F,AlphaMode::Unspecified}})
        rejected(policy,hlg,{},storage,mvs::ResolutionFailure::UnsupportedStorage);
    rejected(policy,hlg,{},mvs::DecodedStorage{mvs::StorageKind::Yuv,static_cast<PixelFormat>(255),AlphaMode::Opaque},
        mvs::ResolutionFailure::UnsupportedStorage);
    rejected(policy,hlg,{},mvs::DecodedStorage{mvs::StorageKind::Yuv,PixelFormat::RGBA32F,static_cast<AlphaMode>(255)},
        mvs::ResolutionFailure::UnsupportedStorage);
    rejected(absent,hlg,{},yuv,mvs::ResolutionFailure::MissingHlgPeak);
    mvs::Policy hlgOverride; require(mvs::decode(hlgOnly,hlgOverride), "HLG-only v2 grammar refused");
    resolved = mvs::resolve(hlgOverride,hlg,{},yuv,failure);
    require(resolved && resolved->peakNits == 1000.0 && resolved->referenceWhiteNits == 203.0,
        "HLG explicit peak or white default changed");
    rejected(hlgOverride,pq,{},yuv,mvs::ResolutionFailure::UnexpectedHlgPeak);
    auto conflict = policy; conflict.peakNits = 2000.0;
    rejected(conflict,hlg,{},yuv,mvs::ResolutionFailure::ConflictingHlgPeak);
    auto badPeak = whiteOnly; badPeak.peakNits = 399.0;
    rejected(badPeak,pq,{},yuv,mvs::ResolutionFailure::LuminanceOutOfRange);
    badPeak.peakNits = 80.0; badPeak.whiteNits = 203.0;
    rejected(badPeak,sdr,{},rgb,mvs::ResolutionFailure::LuminanceOutOfRange);
    auto invalid = policy; invalid.whiteNits = std::numeric_limits<double>::quiet_NaN();
    rejected(invalid,hlg,{},yuv,mvs::ResolutionFailure::InvalidPolicy);
    invalid = policy; invalid.kind = static_cast<mvs::PolicyKind>(3);
    rejected(invalid,hlg,{},yuv,mvs::ResolutionFailure::InvalidPolicy);
    invalid = policy; invalid.legacy.hlgNominalDisplayPeakNits = 1000.0;
    rejected(invalid,hlg,{},yuv,mvs::ResolutionFailure::InvalidPolicy);
    mvs::Policy legacyHlg; require(mvs::decode("video-source-color-v1|hlgPeak=1000",legacyHlg), "v1 HLG policy refused");
    resolved = mvs::resolve(legacyHlg,hlg,{},yuv,failure);
    require(resolved && resolved->peakNits == 1000.0
        && resolved->authoredBytes == "video-source-color-v1|hlgPeak=1000", "v1 HLG peak passthrough changed");
    resolved = mvs::resolve(legacyHlg,pq,{},yuv,failure);
    require(resolved && resolved->peakNits == 10000.0, "legacy HLG policy altered PQ planning defaults");
    for (const auto peak : {400.0,1000.0,2000.0,4000.0,10000.0})
    {
        hlgOverride.hlgPeak = peak;
        resolved = mvs::resolve(hlgOverride,hlg,{},yuv,failure);
        require(resolved && resolved->peakNits == peak, "authored HLG peak lost during resolution");
    }
}

void testDormantLegacyConservation()
{
    const Description defaults;
    require(defaults.version == 1 && kWireVersion == 1 && kColorMathVersion == 1
        && defaults.workingColorSpace == ColorSpace::LinearSRGB && defaults.workingFormat == PixelFormat::RGBA16F
        && defaults.outputIntent == OutputIntent::SdrDisplay && defaults.toneMap == ToneMap::None
        && defaults.luminance.sourceReferenceWhiteNits == 100.0 && defaults.luminance.sourcePeakNits == 100.0
        && defaults.luminance.outputReferenceWhiteNits == 100.0 && defaults.luminance.outputPeakNits == 100.0
        && ! defaults.hlgReferenceDisplayOotf, "v1 defaults changed");
    colortransformoperation::Payload oldPayload;
    require(colortransformoperation::parse(kLegacySdrWire,oldPayload)
        && colortransformoperation::serialize(oldPayload) == kLegacySdrWire,
        "literal v1 payload bytes changed");
    AdmissionFailure failure;
    const auto old = colortransformoperation::parseAndAdmit(kLegacySdrWire,{1920,1080},failure);
    const std::array<std::uint8_t,64> literalLegacy {
        0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x07,0x80,0x00,0x00,0x04,0x38,
        0x03,0x02,0x02,0x03,0x03,0x03,0x02,0x03,0x05,0x04,0x01,0x00,
        0x40,0x59,0x00,0x00,0x00,0x00,0x00,0x00,0x40,0x59,0x00,0x00,0x00,0x00,0x00,0x00,
        0x40,0x59,0x00,0x00,0x00,0x00,0x00,0x00,0x40,0x59,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00 };
    require(old && old->cacheIdentity().size == 60 && old->cacheIdentity().bytes == literalLegacy,
        "literal v1 60-byte identity or trailing storage changed");
    const auto hlg = decodedframecolor::hdrToMaterialLinear({ColorSpace::Rec2020,TransferFunction::HLG},
        {2,2},videosourcecolor::Policy{1000.0},failure);
    const std::array<std::uint8_t,64> literalHlg {
        0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,
        0x05,0x05,0x05,0x01,0x04,0x01,0x01,0x01,0x01,0x04,0x00,0x00,
        0x40,0x69,0x60,0x00,0x00,0x00,0x00,0x00,0x40,0x8f,0x40,0x00,0x00,0x00,0x00,0x00,
        0x40,0x69,0x60,0x00,0x00,0x00,0x00,0x00,0x40,0x8f,0x40,0x00,0x00,0x00,0x00,0x00,
        0x01,0x00,0x00,0x00 };
    require(hlg && hlg->cacheIdentity().size == 61 && hlg->cacheIdentity().bytes == literalHlg,
        "literal v1 HLG optional identity byte changed");
    const auto pq = decodedframecolor::pqToMaterialLinear({ColorSpace::Rec2020,TransferFunction::PQ},{2,2},failure);
    const auto sdr = decodedframecolor::toMaterialSrgb({ColorSpace::DisplayP3,TransferFunction::SRGB},{2,2},failure);
    require(sdr && pq && hlg, "legacy material declarations refused");
    const auto checkMaterial = [] (const Description& d, PixelFormat inputFormat, ColorSpace primaries,
        TransferFunction transfer, AlphaMode alpha, PixelFormat outputFormat, ColorSpace outputPrimaries,
        TransferFunction outputTransfer, double sourceWhite, double sourcePeak, double outputWhite,
        double outputPeak, OutputIntent intent, bool sourceHlg) {
        require(d.version == 1 && d.input.extent == Extent{2,2} && d.output.extent == Extent{2,2}
            && d.input.format == inputFormat && d.input.colorSpace == primaries && d.input.transfer == transfer
            && d.input.alpha == alpha && d.output.format == outputFormat && d.output.colorSpace == outputPrimaries
            && d.output.transfer == outputTransfer && d.output.alpha == alpha
            && d.workingColorSpace == ColorSpace::LinearSRGB && d.workingFormat == PixelFormat::RGBA16F
            && d.toneMap == ToneMap::None && d.outputIntent == intent
            && d.luminance.sourceReferenceWhiteNits == sourceWhite && d.luminance.sourcePeakNits == sourcePeak
            && d.luminance.outputReferenceWhiteNits == outputWhite && d.luminance.outputPeakNits == outputPeak
            && d.hlgReferenceDisplayOotf == sourceHlg, "complete legacy decoded material projection changed");
    };
    checkMaterial(sdr->description(),PixelFormat::RGBA8,ColorSpace::DisplayP3,TransferFunction::SRGB,AlphaMode::Straight,
        PixelFormat::RGBA8,ColorSpace::SRGB,TransferFunction::SRGB,100,100,100,100,OutputIntent::SdrDisplay,false);
    checkMaterial(pq->description(),PixelFormat::RGBA32F,ColorSpace::Rec2020,TransferFunction::PQ,AlphaMode::Opaque,
        PixelFormat::RGBA16F,ColorSpace::LinearSRGB,TransferFunction::Linear,203,10000,203,10000,OutputIntent::SceneLinear,false);
    checkMaterial(hlg->description(),PixelFormat::RGBA32F,ColorSpace::Rec2020,TransferFunction::HLG,AlphaMode::Opaque,
        PixelFormat::RGBA16F,ColorSpace::LinearSRGB,TransferFunction::Linear,203,1000,203,1000,OutputIntent::SceneLinear,true);
}

} // namespace

int main()
{
    testDormantManagedTransform();
    testDormantSourcePolicy();
    testDormantLegacyConservation();
    static_assert(! std::is_default_constructible_v<AdmittedTransform>);
    static_assert(! std::is_copy_assignable_v<AdmittedTransform>);
    static_assert(! AdmittedTransform::allowsCpuProductionFallback);
    static_assert(AdmittedTransform::zeroAlphaForcesZeroRgb);

    require(token(PixelFormat::R8) == "r8"
                && token(PixelFormat::R16) == "r16"
                && token(PixelFormat::RGBA8) == "rgba8"
                && token(PixelFormat::RGBA16F) == "rgba16f"
                && token(PixelFormat::RGBA32F) == "rgba32f",
            "pixel-format vocabulary changed");
    require(token(ColorSpace::LinearSRGB) == "linearSRGB"
                && token(ColorSpace::SRGB) == "sRGB"
                && token(ColorSpace::DisplayP3) == "displayP3"
                && token(ColorSpace::Rec709) == "rec709"
                && token(ColorSpace::Rec2020) == "rec2020",
            "color-space vocabulary changed");
    require(token(TransferFunction::Linear) == "linear"
                && token(TransferFunction::SRGB) == "sRGB"
                && token(TransferFunction::Gamma22) == "gamma22"
                && token(TransferFunction::PQ) == "pq"
                && token(TransferFunction::HLG) == "hlg"
                && token(TransferFunction::Rec709) == "rec709",
            "transfer-function vocabulary changed");
    require(token(AlphaMode::Opaque) == "opaque"
                && token(AlphaMode::Straight) == "straight"
                && token(AlphaMode::Premultiplied) == "premultiplied",
            "alpha vocabulary changed");

    const auto sdr = requireAdmitted(sdrTransform());
    require(sdr.backend() == BackendCapability::NativeGpu,
            "admission did not retain the native GPU requirement");
    const std::array<StageKind, 6> expectedSdrStages {
        StageKind::UnassociateAlpha,
        StageKind::DecodeTransfer,
        StageKind::ConvertToWorkingPrimaries,
        StageKind::ConvertFromWorkingPrimaries,
        StageKind::EncodeTransfer,
        StageKind::AssociateAlpha
    };
    require(sdr.stageCount() == expectedSdrStages.size(),
            "SDR transform stage count is wrong");
    for (std::size_t index = 0; index < expectedSdrStages.size(); ++index)
        require(sdr.stages()[index] == expectedSdrStages[index],
                "SDR transform stage order is wrong");

    const auto& sdrResources = sdr.resources();
    const auto pixels1080p = 1920ull * 1080ull;
    require(sdrResources.pixelCount == pixels1080p
                && sdrResources.inputBytes == pixels1080p * 4ull
                && sdrResources.outputBytes == pixels1080p * 4ull
                && sdrResources.workingBytes == pixels1080p * 8ull
                && sdrResources.peakResidentBytes == pixels1080p * 16ull
                && sdrResources.intermediateImages == 1
                && sdrResources.stageCount == expectedSdrStages.size(),
            "SDR resource accounting is not exact");

    const auto hdr = requireAdmitted(hdrTransform());
    require(hdr.stageCount() == 2
                && hdr.stages()[0] == StageKind::DecodeTransfer
                && hdr.stages()[1] == StageKind::EncodeTransfer,
            "HDR transfer did not decode into and encode from linear Rec.2020");

    auto pqDescription = hdrTransform();
    pqDescription.output.transfer = TransferFunction::PQ;
    pqDescription.outputIntent = OutputIntent::HdrPq;
    const auto pq = requireAdmitted(pqDescription);
    require(pq.stageCount() == 2
                && pq.stages()[0] == StageKind::DecodeTransfer
                && pq.stages()[1] == StageKind::EncodeTransfer,
            "PQ output intent did not preserve explicit linear working light");

    auto sceneDescription = hdrTransform();
    sceneDescription.output.transfer = TransferFunction::Linear;
    sceneDescription.outputIntent = OutputIntent::SceneLinear;
    const auto scene = requireAdmitted(sceneDescription);
    require(scene.stageCount() == 1
                && scene.stages()[0] == StageKind::DecodeTransfer,
            "scene-linear output retained an encoded output stage");

    const auto hdrToSdr = requireAdmitted(hdrToSdrTransform());
    const std::array<StageKind, 4> expectedHdrToSdrStages {
        StageKind::DecodeTransfer,
        StageKind::ToneMapLuminance,
        StageKind::ConvertFromWorkingPrimaries,
        StageKind::EncodeTransfer
    };
    require(hdrToSdr.stageCount() == expectedHdrToSdrStages.size(),
            "HDR-to-SDR stage count is wrong");
    for (std::size_t index = 0; index < expectedHdrToSdrStages.size(); ++index)
        require(hdrToSdr.stages()[index] == expectedHdrToSdrStages[index],
                "HDR-to-SDR stage order is wrong");

    const auto sdrAgain = requireAdmitted(sdrTransform());
    require(sdr.cacheIdentity() == sdrAgain.cacheIdentity()
                && sdr.cacheIdentity().fingerprint()
                    == sdrAgain.cacheIdentity().fingerprint()
                && sdr.cacheIdentity().size == 60,
            "equivalent transforms do not have deterministic cache identity");
    auto changedIdentity = sdrTransform();
    changedIdentity.luminance.sourceReferenceWhiteNits = 90.0;
    const auto changed = requireAdmitted(changedIdentity);
    require(changed.cacheIdentity() != sdr.cacheIdentity(),
            "an output-affecting parameter was omitted from cache identity");
    changedIdentity = sdrTransform();
    changedIdentity.output.alpha = AlphaMode::Straight;
    const auto changedAlpha = requireAdmitted(changedIdentity);
    require(changedAlpha.cacheIdentity() != sdr.cacheIdentity(),
            "alpha semantics were omitted from cache identity");

    auto invalid = sdrTransform();
    invalid.version += 1;
    requireRejected(invalid, AdmissionFailure::UnsupportedVersion,
                    "unknown wire version was accepted");
    requireRejected(sdrTransform(), AdmissionFailure::NativeGpuRequired,
                    "non-native backend was admitted for production",
                    static_cast<BackendCapability>(2));
    requireRejected(sdrTransform(), AdmissionFailure::NativeGpuRequired,
                    "unspecified backend was admitted for production",
                    BackendCapability::Unspecified);

    invalid = sdrTransform();
    invalid.input.extent.width = 0;
    requireRejected(invalid, AdmissionFailure::InvalidExtent,
                    "zero input extent was accepted");
    invalid = sdrTransform();
    invalid.output.extent.width += 1;
    requireRejected(invalid, AdmissionFailure::ExtentMismatch,
                    "implicit resize was admitted as a color transform");

    invalid = sdrTransform();
    invalid.input.format = static_cast<PixelFormat>(99);
    requireRejected(invalid, AdmissionFailure::UnknownPixelFormat,
                    "unknown pixel format was accepted");
    invalid = sdrTransform();
    invalid.input.colorSpace = static_cast<ColorSpace>(99);
    requireRejected(invalid, AdmissionFailure::UnknownColorSpace,
                    "unknown color space was accepted");
    invalid = sdrTransform();
    invalid.input.transfer = static_cast<TransferFunction>(99);
    requireRejected(invalid, AdmissionFailure::UnknownTransferFunction,
                    "unknown transfer function was accepted");
    invalid = sdrTransform();
    invalid.input.alpha = static_cast<AlphaMode>(99);
    requireRejected(invalid, AdmissionFailure::UnknownAlphaMode,
                    "unknown alpha mode was accepted");
    invalid = sdrTransform();
    invalid.outputIntent = static_cast<OutputIntent>(99);
    requireRejected(invalid, AdmissionFailure::UnknownOutputIntent,
                    "unknown output intent was accepted");
    invalid = sdrTransform();
    invalid.toneMap = static_cast<ToneMap>(99);
    requireRejected(invalid, AdmissionFailure::UnknownToneMap,
                    "unknown tone map was accepted");

    invalid = sdrTransform();
    invalid.input.format = PixelFormat::R8;
    requireRejected(invalid, AdmissionFailure::UnsupportedPixelFormat,
                    "single-channel data was admitted as RGB color");
    invalid = sdrTransform();
    invalid.input.colorSpace = ColorSpace::SRGB;
    invalid.input.transfer = TransferFunction::Linear;
    requireRejected(invalid, AdmissionFailure::InvalidColorTransferPair,
                    "an ambiguous sRGB-linear pair was accepted");
    invalid = sdrTransform();
    invalid.workingColorSpace = ColorSpace::DisplayP3;
    requireRejected(invalid, AdmissionFailure::InvalidWorkingSpace,
                    "encoded Display P3 was accepted as a linear working space");
    invalid = sdrTransform();
    invalid.workingFormat = PixelFormat::RGBA8;
    requireRejected(invalid, AdmissionFailure::InvalidWorkingFormat,
                    "8-bit encoded storage was accepted as the working format");

    invalid = hdrTransform();
    invalid.input.format = PixelFormat::RGBA8;
    requireRejected(invalid, AdmissionFailure::UnsupportedPixelFormat,
                    "8-bit RGBA was accepted for PQ input");
    invalid = hdrTransform();
    invalid.output.alpha = AlphaMode::Straight;
    requireRejected(invalid, AdmissionFailure::InvalidOutputIntent,
                    "HDR delivery with alpha was accepted");
    invalid = hdrTransform();
    invalid.outputIntent = OutputIntent::SdrDisplay;
    requireRejected(invalid, AdmissionFailure::InvalidOutputIntent,
                    "HLG output was accepted as SDR display intent");

    invalid = sdrTransform();
    invalid.input.alpha = AlphaMode::Premultiplied;
    invalid.output.alpha = AlphaMode::Opaque;
    requireRejected(invalid, AdmissionFailure::AlphaWouldBeDiscarded,
                    "color transform silently discarded source alpha");

    invalid = hdrToSdrTransform();
    invalid.toneMap = ToneMap::None;
    requireRejected(invalid, AdmissionFailure::ToneMapRequired,
                    "HDR-to-SDR range reduction omitted tone mapping");
    invalid = hdrTransform();
    invalid.toneMap = ToneMap::Reinhard;
    requireRejected(invalid, AdmissionFailure::ToneMapNotAllowed,
                    "display tone map was accepted for HDR output");
    invalid = sdrTransform();
    invalid.toneMap = ToneMap::Reinhard;
    requireRejected(invalid, AdmissionFailure::ToneMapNotAllowed,
                    "unneeded SDR tone mapping was accepted");

    invalid = sdrTransform();
    invalid.luminance.sourcePeakNits = std::numeric_limits<double>::infinity();
    requireRejected(invalid, AdmissionFailure::NonFiniteLuminance,
                    "infinite luminance was accepted");
    invalid = sdrTransform();
    invalid.luminance.sourceReferenceWhiteNits
        = std::numeric_limits<double>::quiet_NaN();
    requireRejected(invalid, AdmissionFailure::NonFiniteLuminance,
                    "NaN luminance was accepted");
    invalid = sdrTransform();
    invalid.luminance.sourceReferenceWhiteNits = 79.0;
    requireRejected(invalid, AdmissionFailure::LuminanceOutOfRange,
                    "reference white below the hard bound was accepted");
    invalid = sdrTransform();
    invalid.luminance.sourcePeakNits = 10001.0;
    requireRejected(invalid, AdmissionFailure::LuminanceOutOfRange,
                    "peak luminance above the hard bound was accepted");
    invalid = sdrTransform();
    invalid.luminance.sourcePeakNits = 99.0;
    invalid.luminance.sourceReferenceWhiteNits = 100.0;
    requireRejected(invalid, AdmissionFailure::LuminanceOutOfRange,
                    "peak luminance below reference white was accepted");

    AdmissionLimits tight;
    tight.stages = expectedSdrStages.size() - 1;
    requireRejected(sdrTransform(), AdmissionFailure::StageLimitExceeded,
                    "stage budget was not enforced", BackendCapability::NativeGpu, tight);
    tight = {};
    tight.residentBytes = sdr.resources().peakResidentBytes - 1;
    requireRejected(sdrTransform(), AdmissionFailure::ResidentByteLimitExceeded,
                    "resident byte budget was not enforced",
                    BackendCapability::NativeGpu, tight);
    tight = {};
    tight.pixels = pixels1080p - 1;
    requireRejected(sdrTransform(), AdmissionFailure::InvalidExtent,
                    "pixel budget was not enforced", BackendCapability::NativeGpu, tight);
    tight = {};
    tight.residentBytes = kMaximumResidentBytes + 1;
    requireRejected(sdrTransform(), AdmissionFailure::InvalidLimits,
                    "caller relaxed the resident-byte hard bound",
                    BackendCapability::NativeGpu, tight);
    tight = {};
    tight.dimension = kMaximumDimension + 1;
    requireRejected(sdrTransform(), AdmissionFailure::InvalidLimits,
                    "caller relaxed the dimension hard bound",
                    BackendCapability::NativeGpu, tight);

    AdmissionFailure materialFailure;
    const decodedframecolor::Declaration pqDeclaration {ColorSpace::Rec2020,TransferFunction::PQ};
    const auto materialPq = decodedframecolor::pqToMaterialLinear(pqDeclaration,{2,2},materialFailure);
    require(materialPq && materialPq->description().output.format == PixelFormat::RGBA16F
        && materialPq->description().output.colorSpace == ColorSpace::LinearSRGB
        && materialPq->description().output.transfer == TransferFunction::Linear
        && materialPq->description().toneMap == ToneMap::None
        && materialPq->description().outputIntent == OutputIntent::SceneLinear,
        "PQ material contract must preserve scene-linear highlights and signed gamut components");
    require(!decodedframecolor::pqToMaterialLinear({ColorSpace::Rec2020,TransferFunction::HLG},{2,2},materialFailure)
        && !decodedframecolor::pqToMaterialLinear({ColorSpace::SRGB,TransferFunction::PQ},{2,2},materialFailure),
        "HLG without a source luminance policy and non-Rec2020 PQ cannot enter the linear material route");
    require(near(decodeSrgb(0.5),0.21404114048223255,1.0e-12)
        && near(decodePqNits(0.5)/203.0,0.4544123595766762,1.0e-9)
        && near(decodePqNits(0.75)/203.0,4.844225892077,1.0e-8),
        "independent SDR/PQ code-value oracles use the common 203-nit scene white");

    const decodedframecolor::Declaration hlgDeclaration {ColorSpace::Rec2020,TransferFunction::HLG};
    require(!decodedframecolor::hdrToMaterialLinear(hlgDeclaration,{2,2},{},materialFailure)
        && materialFailure == AdmissionFailure::MissingHlgSourcePolicy,
        "absent HLG nominal display peak must fail explicitly");
    for (double peak : {400.0,1000.0,2000.0,4000.0,10000.0})
    {
        const auto admittedHlg = decodedframecolor::hdrToMaterialLinear(hlgDeclaration,{2,2},
            videosourcecolor::Policy{peak},materialFailure);
        require(admittedHlg && admittedHlg->description().hlgReferenceDisplayOotf
            && admittedHlg->description().luminance.sourcePeakNits == peak
            && admittedHlg->description().luminance.outputReferenceWhiteNits == 203.0
            && admittedHlg->description().toneMap == ToneMap::None,
            "HLG interpretation uses explicit source peak and shared scene white without tone mapping");
        auto legacy = admittedHlg->description(); legacy.hlgReferenceDisplayOotf = false;
        const auto legacyTransform = admit(legacy,BackendCapability::NativeGpu,materialFailure);
        require(legacyTransform && legacyTransform->cacheIdentity() != admittedHlg->cacheIdentity(),
            "OOTF interpretation must have distinct transform identity");
    }
    for (double invalid : {399.0,10001.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
        require(!decodedframecolor::hdrToMaterialLinear(hlgDeclaration,{2,2},videosourcecolor::Policy{invalid},materialFailure),
            "invalid HLG source interpretation cannot enter native materials");
    const auto genericPq = decodedframecolor::hdrToMaterialLinear(pqDeclaration,{2,2},{},materialFailure);
    require(genericPq && genericPq->cacheIdentity() == materialPq->cacheIdentity(),
        "PQ material transform identity remains byte identical");
    // Independent BT.2100 reference OOTF, neutral half-code at 1000 nits.
    require(near(1000.0*std::pow(decodeHlg(0.5),1.2)/203.0,0.24973905660640633,1.0e-12),
        "HLG material oracle includes display OOTF rather than inverse OETF alone");

    for (const double encoded : { 0.0, 0.018, 0.5, 1.0 })
        require(near(encodeSrgb(decodeSrgb(encoded)), encoded, 2.0e-12),
                "test-only sRGB decode/encode oracle did not round-trip");
    for (const double nits : { 0.0, 100.0, 1000.0, 10000.0 })
        require(near(decodePqNits(encodePqNits(nits)), nits, 2.0e-6),
                "test-only PQ encode/decode oracle did not round-trip nits");
    for (const double linear : { 0.0, 0.01, 0.18, 1.0 })
        require(near(decodeHlg(encodeHlg(linear)), linear, 2.0e-10),
                "test-only HLG encode/decode oracle did not round-trip");

    const auto transparent = unassociate({ 0.4, 0.2, 0.1, 0.0 });
    require(transparent == std::array<double, 4> { 0.0, 0.0, 0.0, 0.0 },
            "zero-alpha policy did not force RGB to zero");
    const auto straight = unassociate({ 0.2, 0.1, 0.05, 0.5 });
    const auto premultiplied = associate(straight);
    require(near(premultiplied[0], 0.2)
                && near(premultiplied[1], 0.1)
                && near(premultiplied[2], 0.05)
                && near(premultiplied[3], 0.5),
            "test-only alpha association oracle did not round-trip");

    std::cout << "ColorTransformContractTests: PASS\n";
    return EXIT_SUCCESS;
}
