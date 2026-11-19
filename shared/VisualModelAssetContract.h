#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace visualmodelasset
{
constexpr std::size_t kMaximumStableAssetIdBytes = 128;
constexpr std::size_t kSha256HexBytes = 64;

struct Identity
{
    std::string stableAssetId;
    std::string contentSha256;

    friend bool operator==(const Identity& left, const Identity& right) noexcept
    {
        return left.stableAssetId == right.stableAssetId
            && left.contentSha256 == right.contentSha256;
    }

    friend bool operator!=(const Identity& left, const Identity& right) noexcept
    {
        return !(left == right);
    }
};

enum class IdentityFailure
{
    None = 0,
    InvalidStableAssetId,
    InvalidContentFingerprint
};

inline constexpr std::string_view token(IdentityFailure failure) noexcept
{
    switch (failure)
    {
        case IdentityFailure::None: return "none";
        case IdentityFailure::InvalidStableAssetId: return "invalidStableAssetId";
        case IdentityFailure::InvalidContentFingerprint: return "invalidContentFingerprint";
    }
    return {};
}

inline bool validStableAssetId(std::string_view id) noexcept
{
    if (id.empty() || id.size() > kMaximumStableAssetIdBytes)
        return false;

    const auto alphaNumeric = [](char character) noexcept
    {
        return (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9');
    };

    if (!alphaNumeric(id.front()))
        return false;
    for (const auto character : id)
        if (!alphaNumeric(character) && character != '-' && character != '_'
            && character != '.')
            return false;
    return true;
}

inline bool validSha256(std::string_view digest) noexcept
{
    if (digest.size() != kSha256HexBytes)
        return false;
    for (const auto character : digest)
        if (!((character >= '0' && character <= '9')
              || (character >= 'a' && character <= 'f')))
            return false;
    return true;
}

inline bool validate(const Identity& identity, IdentityFailure& failure) noexcept
{
    failure = IdentityFailure::None;
    if (!validStableAssetId(identity.stableAssetId))
    {
        failure = IdentityFailure::InvalidStableAssetId;
        return false;
    }
    if (!validSha256(identity.contentSha256))
    {
        failure = IdentityFailure::InvalidContentFingerprint;
        return false;
    }
    return true;
}

// This shared identity carries no path, source bytes, decoded payload, catalog
// authority, trust grant, or executable capability. The helper receives model
// content only through a separate strict admission and transfer boundary.
static_assert(kSha256HexBytes == 64);
} // namespace visualmodelasset
