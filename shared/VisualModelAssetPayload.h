#pragma once

#include "VisualImportedAnimationDeformationRequest.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace visualmodelassetpayload
{
// The JSON-RPC sidecar transport is line bounded. Model payloads are split into
// independently bounded chunks, and the helper checks this payload limit before
// allocating the destination buffer. Keep this equal to the native GLB product
// decoder's bound so the helper never retains a payload preview/export cannot
// consume.
// A GLB's header stores its total length in 32 bits, so that is the payload bound.
inline constexpr std::uint64_t kMaxPayloadBytes = 0xffffffffull;
inline constexpr std::size_t kMaxChunkBytes = 256u * 1024u;
inline constexpr std::size_t kMaxConcurrentTransfers = 4;
inline constexpr std::uint64_t kMaxPendingBytes = kMaxConcurrentTransfers * kMaxPayloadBytes;
// The resident sizes are cache targets: unreferenced payloads are evicted to
// stay under them, and a payload still in use is admitted past them.
inline constexpr std::size_t kMaxResidentPayloads = 16;
inline constexpr std::uint64_t kMaxResidentBytes = 64ull * 1024ull * 1024ull;

class ImmutablePayload final
{
public:
    static std::shared_ptr<const ImmutablePayload> create(
        visualanimationimport::ExactContentAssetKey key,
        std::vector<std::uint8_t>&& exactBytes)
    {
        return std::shared_ptr<const ImmutablePayload>(
            new ImmutablePayload(std::move(key), std::move(exactBytes)));
    }

    const visualanimationimport::ExactContentAssetKey& key() const noexcept { return key_; }
    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }

private:
    ImmutablePayload(visualanimationimport::ExactContentAssetKey key,
                     std::vector<std::uint8_t>&& exactBytes)
        : key_(std::move(key)), bytes_(std::move(exactBytes))
    {
    }

    const visualanimationimport::ExactContentAssetKey key_;
    const std::vector<std::uint8_t> bytes_;
};

using PayloadPtr = std::shared_ptr<const ImmutablePayload>;
} // namespace visualmodelassetpayload
