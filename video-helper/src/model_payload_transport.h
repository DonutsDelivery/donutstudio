#pragma once

#include "../../shared/VisualModelAssetPayload.h"
#include "../../shared/VisualModelAssetContract.h"
#include "sha256.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace videohelper::modelpayload
{
enum class Failure
{
    None = 0,
    InvalidTransferId,
    DuplicateTransfer,
    InvalidIdentity,
    PayloadLimitExceeded,
    ConcurrentTransferLimitExceeded,
    PendingByteLimitExceeded,
    ResidentPayloadLimitExceeded,
    ResidentByteLimitExceeded,
    AllocationFailed,
    UnknownTransfer,
    InvalidChunkEncoding,
    ChunkLimitExceeded,
    NonSequentialChunk,
    PayloadSizeMismatch,
    DigestMismatch
};

inline constexpr std::string_view token(Failure failure) noexcept
{
    switch (failure)
    {
        case Failure::None: return "none";
        case Failure::InvalidTransferId: return "invalidTransferId";
        case Failure::DuplicateTransfer: return "duplicateTransfer";
        case Failure::InvalidIdentity: return "invalidIdentity";
        case Failure::PayloadLimitExceeded: return "payloadLimitExceeded";
        case Failure::ConcurrentTransferLimitExceeded: return "concurrentTransferLimitExceeded";
        case Failure::PendingByteLimitExceeded: return "pendingByteLimitExceeded";
        case Failure::ResidentPayloadLimitExceeded: return "residentPayloadLimitExceeded";
        case Failure::ResidentByteLimitExceeded: return "residentByteLimitExceeded";
        case Failure::AllocationFailed: return "allocationFailed";
        case Failure::UnknownTransfer: return "unknownTransfer";
        case Failure::InvalidChunkEncoding: return "invalidChunkEncoding";
        case Failure::ChunkLimitExceeded: return "chunkLimitExceeded";
        case Failure::NonSequentialChunk: return "nonSequentialChunk";
        case Failure::PayloadSizeMismatch: return "payloadSizeMismatch";
        case Failure::DigestMismatch: return "digestMismatch";
    }
    return {};
}

struct Result
{
    Failure failure = Failure::None;
    std::uint64_t receivedBytes = 0;

    explicit operator bool() const noexcept { return failure == Failure::None; }
};

struct CommitResult : Result
{
    visualmodelassetpayload::PayloadPtr payload;
};

namespace detail
{
inline bool checkedAdd(std::uint64_t left,
                       std::uint64_t right,
                       std::uint64_t& result) noexcept
{
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

inline int base64Value(char value) noexcept
{
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

inline bool decodeBase64Bounded(const std::string& encoded,
                                std::size_t maxDecodedBytes,
                                std::vector<std::uint8_t>& decoded)
{
    decoded.clear();
    if (encoded.empty() || (encoded.size() % 4u) != 0u)
        return false;

    std::size_t padding = 0;
    if (encoded.back() == '=') ++padding;
    if (encoded.size() > 1 && encoded[encoded.size() - 2] == '=') ++padding;
    const auto groups = encoded.size() / 4u;
    if (groups > (std::numeric_limits<std::size_t>::max() / 3u))
        return false;
    const auto decodedSize = groups * 3u - padding;
    if (decodedSize == 0 || decodedSize > maxDecodedBytes)
        return false;

    // The decoded allocation happens only after the exact decoded size has
    // passed the per-chunk bound.
    try
    {
        decoded.resize(decodedSize);
    }
    catch (const std::bad_alloc&)
    {
        return false;
    }

    std::size_t output = 0;
    for (std::size_t i = 0; i < encoded.size(); i += 4u)
    {
        const bool last = i + 4u == encoded.size();
        const int a = base64Value(encoded[i]);
        const int b = base64Value(encoded[i + 1]);
        const int c = encoded[i + 2] == '=' ? 0 : base64Value(encoded[i + 2]);
        const int d = encoded[i + 3] == '=' ? 0 : base64Value(encoded[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0
            || (! last && (encoded[i + 2] == '=' || encoded[i + 3] == '='))
            || (encoded[i + 2] == '=' && encoded[i + 3] != '=')
            || (last && padding == 2u && (b & 0x0f) != 0)
            || (last && padding == 1u && (c & 0x03) != 0))
        {
            decoded.clear();
            return false;
        }
        const auto packed = (static_cast<std::uint32_t>(a) << 18u)
                          | (static_cast<std::uint32_t>(b) << 12u)
                          | (static_cast<std::uint32_t>(c) << 6u)
                          | static_cast<std::uint32_t>(d);
        if (output < decodedSize) decoded[output++] = static_cast<std::uint8_t>(packed >> 16u);
        if (output < decodedSize) decoded[output++] = static_cast<std::uint8_t>(packed >> 8u);
        if (output < decodedSize) decoded[output++] = static_cast<std::uint8_t>(packed);
    }
    return output == decodedSize;
}

inline std::string digestHex(const std::vector<std::uint8_t>& bytes)
{
    Sha256 digest;
    digest.update(bytes.data(), bytes.size());
    return digest.finishHex();
}
} // namespace detail

class Receiver final
{
public:
    Result begin(const std::string& transferId,
                 const visualanimationimport::ExactContentAssetKey& key)
    {
        if (! visualmodelasset::validStableAssetId(transferId))
            return { Failure::InvalidTransferId, 0 };
        if (pending_.find(transferId) != pending_.end())
            return { Failure::DuplicateTransfer, 0 };
        if (! visualmodelasset::validStableAssetId(key.id)
            || key.version == 0
            || key.version > static_cast<std::uint64_t>(
                   std::numeric_limits<std::int64_t>::max())
            || ! visualmodelasset::validSha256(key.contentSha256)
            || key.sourceMediaType != "model/gltf-binary")
            return { Failure::InvalidIdentity, 0 };
        if (key.sourceByteSize == 0
            || key.sourceByteSize > visualmodelassetpayload::kMaxPayloadBytes)
            return { Failure::PayloadLimitExceeded, 0 };
        if (pending_.size() >= visualmodelassetpayload::kMaxConcurrentTransfers)
            return { Failure::ConcurrentTransferLimitExceeded, 0 };

        std::uint64_t nextPendingBytes = 0;
        if (! detail::checkedAdd(pendingBytes_, key.sourceByteSize, nextPendingBytes)
            || nextPendingBytes > visualmodelassetpayload::kMaxPendingBytes)
            return { Failure::PendingByteLimitExceeded, 0 };

        try
        {
            // All identity, per-payload, transfer-count, overflow, and aggregate
            // pending-byte checks above happen before this single allocation.
            Pending pending;
            pending.key = key;
            pending.bytes.resize(static_cast<std::size_t>(key.sourceByteSize));
            pending_.emplace(transferId, std::move(pending));
        }
        catch (const std::bad_alloc&)
        {
            return { Failure::AllocationFailed, 0 };
        }
        catch (const std::length_error&)
        {
            return { Failure::AllocationFailed, 0 };
        }
        pendingBytes_ = nextPendingBytes;
        return {};
    }

    Result appendBase64(const std::string& transferId,
                        std::uint64_t offset,
                        const std::string& encoded)
    {
        const auto found = pending_.find(transferId);
        if (found == pending_.end())
            return { Failure::UnknownTransfer, 0 };
        auto& pending = found->second;
        if (offset != pending.receivedBytes)
            return { Failure::NonSequentialChunk, pending.receivedBytes };

        std::vector<std::uint8_t> chunk;
        if (! detail::decodeBase64Bounded(
                encoded, visualmodelassetpayload::kMaxChunkBytes, chunk))
            return { Failure::InvalidChunkEncoding, pending.receivedBytes };
        if (chunk.size() > visualmodelassetpayload::kMaxChunkBytes)
            return { Failure::ChunkLimitExceeded, pending.receivedBytes };

        std::uint64_t end = 0;
        if (! detail::checkedAdd(offset, chunk.size(), end)
            || end > pending.bytes.size())
            return { Failure::PayloadSizeMismatch, pending.receivedBytes };
        std::copy(chunk.begin(), chunk.end(), pending.bytes.begin()
                                                + static_cast<std::ptrdiff_t>(offset));
        pending.receivedBytes = end;
        return { Failure::None, pending.receivedBytes };
    }

    CommitResult commit(const std::string& transferId)
    {
        const auto found = pending_.find(transferId);
        if (found == pending_.end())
            return { { Failure::UnknownTransfer, 0 }, {} };
        auto& pending = found->second;
        if (pending.receivedBytes != pending.key.sourceByteSize)
            return { { Failure::PayloadSizeMismatch, pending.receivedBytes }, {} };
        if (detail::digestHex(pending.bytes) != pending.key.contentSha256)
        {
            const auto received = pending.receivedBytes;
            erase(found);
            return { { Failure::DigestMismatch, received }, {} };
        }

        auto key = pending.key;
        auto bytes = std::move(pending.bytes);
        const auto received = pending.receivedBytes;
        erase(found);
        try
        {
            return { { Failure::None, received },
                     visualmodelassetpayload::ImmutablePayload::create(
                         std::move(key), std::move(bytes)) };
        }
        catch (const std::bad_alloc&)
        {
            return { { Failure::AllocationFailed, received }, {} };
        }
    }

    Result abort(const std::string& transferId)
    {
        const auto found = pending_.find(transferId);
        if (found == pending_.end())
            return { Failure::UnknownTransfer, 0 };
        const auto received = found->second.receivedBytes;
        erase(found);
        return { Failure::None, received };
    }

    void reset() noexcept
    {
        pending_.clear();
        pendingBytes_ = 0;
    }

    std::size_t pendingTransferCount() const noexcept { return pending_.size(); }
    std::uint64_t pendingByteCount() const noexcept { return pendingBytes_; }

private:
    struct Pending
    {
        visualanimationimport::ExactContentAssetKey key;
        std::vector<std::uint8_t> bytes;
        std::uint64_t receivedBytes = 0;
    };

    using PendingMap = std::map<std::string, Pending>;

    void erase(PendingMap::const_iterator found) noexcept
    {
        pendingBytes_ -= found->second.key.sourceByteSize;
        pending_.erase(found);
    }

    PendingMap pending_;
    std::uint64_t pendingBytes_ = 0;
};

/** Bounded immutable payload ownership shared by preview and export lookup. */
class Store final
{
public:
    Result begin(const std::string& transferId,
                 const visualanimationimport::ExactContentAssetKey& key)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return receiver_.begin(transferId, key);
    }

    Result appendBase64(const std::string& transferId,
                        std::uint64_t offset,
                        const std::string& encoded)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return receiver_.appendBase64(transferId, offset, encoded);
    }

    CommitResult commit(const std::string& transferId)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        auto received = receiver_.commit(transferId);
        if (! received || ! received.payload)
            return received;

        for (const auto& existing : resident_)
        {
            if (visualanimationimport::sameAsset(existing.second->key(),
                                                  received.payload->key()))
            {
                received.payload = existing.second;
                return received;
            }
        }
        std::uint64_t nextResidentBytes = 0;
        while (resident_.size() >= visualmodelassetpayload::kMaxResidentPayloads
               || ! detail::checkedAdd(residentBytes_, received.payload->bytes().size(),
                                       nextResidentBytes)
               || nextResidentBytes > visualmodelassetpayload::kMaxResidentBytes)
        {
            if (! evictOneUnreferenced())
            {
                if (resident_.size() >= visualmodelassetpayload::kMaxResidentPayloads)
                    return { { Failure::ResidentPayloadLimitExceeded,
                               received.receivedBytes }, {} };
                return { { Failure::ResidentByteLimitExceeded,
                           received.receivedBytes }, {} };
            }
        }

        try
        {
            resident_.emplace(Key(received.payload->key()), received.payload);
        }
        catch (const std::bad_alloc&)
        {
            return { { Failure::AllocationFailed, received.receivedBytes }, {} };
        }
        residentBytes_ = nextResidentBytes;
        return received;
    }

    Result abort(const std::string& transferId)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return receiver_.abort(transferId);
    }

    visualmodelassetpayload::PayloadPtr resolvePreview(
        const visualanimationimport::ExactContentAssetKey& key) const
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return resolve(key);
    }

    visualmodelassetpayload::PayloadPtr resolveExport(
        const visualanimationimport::ExactContentAssetKey& key) const
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return resolve(key);
    }

    bool erase(const visualanimationimport::ExactContentAssetKey& key) noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        for (auto found = resident_.begin(); found != resident_.end(); ++found)
            if (visualanimationimport::sameAsset(found->second->key(), key))
            {
                residentBytes_ -= found->second->bytes().size();
                resident_.erase(found);
                return true;
            }
        return false;
    }

    void reset() noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        receiver_.reset();
        resident_.clear();
        residentBytes_ = 0;
    }

    std::size_t pendingTransferCount() const noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return receiver_.pendingTransferCount();
    }

    std::uint64_t pendingByteCount() const noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return receiver_.pendingByteCount();
    }

    std::size_t residentPayloadCount() const noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return resident_.size();
    }
    std::uint64_t residentByteCount() const noexcept
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return residentBytes_;
    }

private:
    struct Key
    {
        explicit Key(const visualanimationimport::ExactContentAssetKey& key)
            : id(key.id), version(key.version), contentSha256(key.contentSha256),
              sourceMediaType(key.sourceMediaType), sourceByteSize(key.sourceByteSize)
        {
        }

        friend bool operator<(const Key& left, const Key& right) noexcept
        {
            if (left.id != right.id) return left.id < right.id;
            if (left.version != right.version) return left.version < right.version;
            if (left.contentSha256 != right.contentSha256)
                return left.contentSha256 < right.contentSha256;
            if (left.sourceMediaType != right.sourceMediaType)
                return left.sourceMediaType < right.sourceMediaType;
            return left.sourceByteSize < right.sourceByteSize;
        }

        std::string id;
        std::uint64_t version = 0;
        std::string contentSha256;
        std::string sourceMediaType;
        std::uint64_t sourceByteSize = 0;
    };

    visualmodelassetpayload::PayloadPtr resolve(
        const visualanimationimport::ExactContentAssetKey& key) const
    {
        for (const auto& resident : resident_)
            if (visualanimationimport::sameAsset(resident.second->key(), key))
                return resident.second;
        return {};
    }

    bool evictOneUnreferenced() noexcept
    {
        for (auto found = resident_.begin(); found != resident_.end(); ++found)
        {
            if (found->second.use_count() != 1)
                continue;
            residentBytes_ -= found->second->bytes().size();
            resident_.erase(found);
            return true;
        }
        return false;
    }

    Receiver receiver_;
    std::map<Key, visualmodelassetpayload::PayloadPtr> resident_;
    std::uint64_t residentBytes_ = 0;
    mutable std::mutex mutex_;
};
} // namespace videohelper::modelpayload
