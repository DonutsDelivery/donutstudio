#pragma once

#include "VisualModelAssetContract.h"
#include <array>
#include <cstdint>
#include <string>
#include <limits>
#include <vector>

namespace visualenvironmentimage
{
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(std::array<float, 3>) == 12);
inline constexpr std::uint32_t kContractVersion = 1;
// An environment image is as large as its file. The device's texture size
// query and allocation decide whether it uploads; these bounds only keep the
// byte arithmetic below inside 64 bits.
inline constexpr std::uint32_t kMaxDimension = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint64_t kMaxTexels = (std::uint64_t { 1 } << 62) / 12ull;
inline constexpr std::uint64_t kMaxSourceBytes = std::uint64_t { 1 } << 62;
inline constexpr std::uint64_t kMaxDecodedBytes = kMaxTexels * 3ull * sizeof(float);

struct Identity
{
    std::string id;
    std::uint64_t version = 0;
    std::string contentSha256;
};

enum class SourcePrimaries { RadianceEqualEnergy, Rec709D65 };

struct Metadata
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    SourcePrimaries sourcePrimaries = SourcePrimaries::RadianceEqualEnergy;
};

inline bool valid(const Metadata& metadata) noexcept
{
    return metadata.width > 0 && metadata.height > 0
        && metadata.width <= kMaxDimension && metadata.height <= kMaxDimension
        && metadata.width == 2ull * metadata.height
        && std::uint64_t(metadata.width) * metadata.height <= kMaxTexels
        && (metadata.sourcePrimaries == SourcePrimaries::RadianceEqualEnergy
            || metadata.sourcePrimaries == SourcePrimaries::Rec709D65);
}

inline std::uint64_t decodedBytes(const Metadata& metadata) noexcept
{
    return std::uint64_t(metadata.width) * metadata.height * 3ull * sizeof(float);
}

// Owned, top-first RGB32F in scene-linear Rec.709/D65. File exposure and color
// correction are undone before conversion through XYZ, without white adaptation.
// Samples are nonnegative and finite. Nonzero values
// must remain nonzero in RGB32F. The 2:1 image is
// interpreted as equirectangular, +Y up. For u,v in [0,1], longitude is
// 2*pi*(u-.5), polar angle pi*v, and direction is
// (sin(longitude)*sin(polar), cos(polar), cos(longitude)*sin(polar)).
// Orientation and user exposure belong to scene binding, never these asset bytes.
struct DecodedImage
{
    Metadata metadata;
    std::vector<std::array<float, 3>> pixels;
};
} // namespace visualenvironmentimage
