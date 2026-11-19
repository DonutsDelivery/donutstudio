#pragma once

#include "ColorTransformContract.h"
#include "VideoSourceColorPolicy.h"

namespace decodedframecolor
{
struct Declaration
{
    colortransform::ColorSpace primaries = colortransform::ColorSpace::Unspecified;
    colortransform::TransferFunction transfer = colortransform::TransferFunction::Unspecified;
    bool operator==(const Declaration& other) const noexcept
    { return primaries == other.primaries && transfer == other.transfer; }
};

// Maps ITU-T H.273 (CICP) code points, which FFmpeg's AVCOL_PRI_* and AVCOL_TRC_*
// enums use. Unlisted codes stay Unspecified; they never default to sRGB.
inline Declaration fromCicp(int primariesCode, int transferCode) noexcept
{
    using Space = colortransform::ColorSpace;
    using Transfer = colortransform::TransferFunction;
    Declaration declaration;
    switch (primariesCode)
    {
        case 1: declaration.primaries = Space::Rec709; break;
        case 9: declaration.primaries = Space::Rec2020; break;
        case 12: declaration.primaries = Space::DisplayP3; break;
        default: break;
    }
    switch (transferCode)
    {
        case 13:
            declaration.transfer = Transfer::SRGB;
            if (primariesCode == 1) declaration.primaries = Space::SRGB;
            break;
        case 1: declaration.transfer = Transfer::Rec709; break;
        case 4: declaration.transfer = Transfer::Gamma22; break;
        case 8:
            declaration.transfer = Transfer::Linear;
            if (primariesCode == 1) declaration.primaries = Space::LinearSRGB;
            break;
        // BT.2020 10 and 12-bit SDR reuses the BT.709 transfer.
        case 14:
        case 15: declaration.transfer = Transfer::Rec709; break;
        case 16: declaration.transfer = Transfer::PQ; break;
        case 18: declaration.transfer = Transfer::HLG; break;
        default: break;
    }
    return declaration;
}

inline std::optional<colortransform::AdmittedTransform> toMaterialSrgb(
    Declaration declaration, colortransform::Extent extent, colortransform::AdmissionFailure& failure)
{
    colortransform::Description description;
    description.input = {extent,colortransform::PixelFormat::RGBA8,declaration.primaries,
        declaration.transfer,colortransform::AlphaMode::Straight};
    description.output = {extent,colortransform::PixelFormat::RGBA8,colortransform::ColorSpace::SRGB,
        colortransform::TransferFunction::SRGB,colortransform::AlphaMode::Straight};
    // RGBA8 decode cannot preserve HDR highlights. HDR must enter through a
    // separately admitted float decode/output path, not this SDR texture route.
    if (colortransform::isHdrTransfer(declaration.transfer))
    { failure = colortransform::AdmissionFailure::UnsupportedPixelFormat; return std::nullopt; }
    return colortransform::admit(description,colortransform::BackendCapability::NativeGpu,failure);
}

inline std::optional<colortransform::AdmittedTransform> hdrToMaterialSrgb(
    Declaration declaration, colortransform::Extent extent, colortransform::AdmissionFailure& failure)
{
    if (declaration.primaries != colortransform::ColorSpace::Rec2020
        || !colortransform::isHdrTransfer(declaration.transfer))
    { failure = colortransform::AdmissionFailure::InvalidColorTransferPair; return std::nullopt; }
    colortransform::Description description;
    description.input = {extent,colortransform::PixelFormat::RGBA32F,declaration.primaries,
        declaration.transfer,colortransform::AlphaMode::Opaque};
    description.output = {extent,colortransform::PixelFormat::RGBA8,colortransform::ColorSpace::SRGB,
        colortransform::TransferFunction::SRGB,colortransform::AlphaMode::Opaque};
    description.outputIntent = colortransform::OutputIntent::SdrDisplay;
    description.toneMap = colortransform::ToneMap::Reinhard;
    description.luminance = {203.0,
        declaration.transfer == colortransform::TransferFunction::PQ ? 10000.0 : 1000.0,
        100.0,100.0};
    return colortransform::admit(description,colortransform::BackendCapability::NativeGpu,failure);
}
// PQ uses absolute luminance, normalized to the existing 203-nit scene white.
// HLG joins through hdrToMaterialLinear only with an explicit source policy.
inline std::optional<colortransform::AdmittedTransform> pqToMaterialLinear(
    Declaration declaration, colortransform::Extent extent, colortransform::AdmissionFailure& failure)
{
    if (declaration.primaries != colortransform::ColorSpace::Rec2020
        || declaration.transfer != colortransform::TransferFunction::PQ)
    { failure = colortransform::AdmissionFailure::InvalidColorTransferPair; return std::nullopt; }
    colortransform::Description description;
    description.input = {extent,colortransform::PixelFormat::RGBA32F,declaration.primaries,
        declaration.transfer,colortransform::AlphaMode::Opaque};
    description.output = {extent,colortransform::PixelFormat::RGBA16F,colortransform::ColorSpace::LinearSRGB,
        colortransform::TransferFunction::Linear,colortransform::AlphaMode::Opaque};
    description.outputIntent = colortransform::OutputIntent::SceneLinear;
    description.luminance = {203.0,10000.0,203.0,10000.0};
    return colortransform::admit(description,colortransform::BackendCapability::NativeGpu,failure);
}

// BT.2100 fixed 5 cd/m2 reference surround and zero black lift. Normalize the display-linear
// result to the same 203-nit white as PQ materials. Input remains bounded to
// normalized HLG code values by the native lease; extended headroom is unverified.
inline std::optional<colortransform::AdmittedTransform> hdrToMaterialLinear(
    Declaration declaration, colortransform::Extent extent, const videosourcecolor::Policy& policy,
    colortransform::AdmissionFailure& failure)
{
    if (!videosourcecolor::valid(policy))
    { failure = colortransform::AdmissionFailure::LuminanceOutOfRange; return std::nullopt; }
    if (declaration.transfer == colortransform::TransferFunction::PQ)
        return pqToMaterialLinear(declaration,extent,failure);
    if (declaration.primaries != colortransform::ColorSpace::Rec2020
        || declaration.transfer != colortransform::TransferFunction::HLG)
    { failure = colortransform::AdmissionFailure::InvalidColorTransferPair; return std::nullopt; }
    if (!policy.hlgNominalDisplayPeakNits)
    { failure = colortransform::AdmissionFailure::MissingHlgSourcePolicy; return std::nullopt; }
    colortransform::Description description;
    description.input = {extent,colortransform::PixelFormat::RGBA32F,declaration.primaries,
        declaration.transfer,colortransform::AlphaMode::Opaque};
    description.output = {extent,colortransform::PixelFormat::RGBA16F,colortransform::ColorSpace::LinearSRGB,
        colortransform::TransferFunction::Linear,colortransform::AlphaMode::Opaque};
    description.outputIntent = colortransform::OutputIntent::SceneLinear;
    description.hlgReferenceDisplayOotf = true;
    description.luminance = {203.0,*policy.hlgNominalDisplayPeakNits,203.0,*policy.hlgNominalDisplayPeakNits};
    return colortransform::admit(description,colortransform::BackendCapability::NativeGpu,failure);
}

}
