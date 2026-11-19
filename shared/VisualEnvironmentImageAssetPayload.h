#pragma once

#include "VisualEnvironmentImageAssetContract.h"
#include <memory>
#include <utility>

namespace visualenvironmentimagepayload
{
inline constexpr std::uint64_t kMaxPayloadBytes = visualenvironmentimage::kMaxSourceBytes;
inline constexpr std::size_t kMaxChunkBytes = 256u * 1024u;
inline constexpr std::size_t kMaxConcurrentTransfers = 2;
inline constexpr std::uint64_t kMaxPendingBytes = std::uint64_t { 1 } << 62;
// The resident sizes are cache targets: unreferenced images are evicted to stay
// under them, and an image still in use is admitted past them.
inline constexpr std::size_t kMaxResidentPayloads = 8;
inline constexpr std::uint64_t kMaxResidentSourceBytes = 64ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kMaxResidentDecodedBytes = 96ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kMaxResidentCombinedBytes = 128ull * 1024ull * 1024ull;

// Transfer limits for the helper admission boundary. This owned container
// carries no decode authority. A receiver must verify the digest and interpretation
// before decoding, and must enforce aggregate pending/resident limits.

class ImmutablePayload final
{
public:
    static std::shared_ptr<const ImmutablePayload> create(
        visualenvironmentimage::Identity key, visualenvironmentimage::Metadata metadata,
        std::vector<std::uint8_t>&& bytes)
    {
        if (!visualmodelasset::validStableAssetId(key.id) || key.version == 0
            || key.version > std::uint64_t(std::numeric_limits<std::int64_t>::max())
            || !visualmodelasset::validSha256(key.contentSha256)
            || !visualenvironmentimage::valid(metadata) || bytes.empty()
            || bytes.size() > kMaxPayloadBytes) return {};
        return std::shared_ptr<const ImmutablePayload>(
            new ImmutablePayload(std::move(key), metadata, std::move(bytes)));
    }
    const visualenvironmentimage::Identity& key() const noexcept { return key_; }
    const visualenvironmentimage::Metadata& metadata() const noexcept { return metadata_; }
    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
private:
    ImmutablePayload(visualenvironmentimage::Identity key, visualenvironmentimage::Metadata metadata,
                     std::vector<std::uint8_t>&& bytes)
        : key_(std::move(key)), metadata_(metadata), bytes_(std::move(bytes)) {}
    const visualenvironmentimage::Identity key_;
    const visualenvironmentimage::Metadata metadata_;
    const std::vector<std::uint8_t> bytes_;
};
using PayloadPtr = std::shared_ptr<const ImmutablePayload>;
} // namespace visualenvironmentimagepayload
