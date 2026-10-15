#pragma once

#include "../../shared/VisualModelAssetContract.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace videohelper::modelcache
{
constexpr bool kOpensSourcePaths = false;
constexpr bool kDecodesImages = false;
constexpr bool kPerformsGpuUpload = false;
constexpr bool kAllowsCpuProductionFallback = false;
constexpr bool kAcceptsCatalogAsModelPayload = false;

struct Limits
{
    std::size_t maxEntries = 128;
    std::size_t maxOwnedResourcesPerEntry = 32768;
    std::size_t maxBorrowedResourcesPerEntry = 4096;
    std::uint64_t maxDecodedBytesPerEntry = 512ull * 1024ull * 1024ull;
    std::uint64_t maxGpuBytesPerEntry = 1024ull * 1024ull * 1024ull;
    std::uint64_t maxResidentBytesPerEntry = 1536ull * 1024ull * 1024ull;
    std::uint64_t maxDecodedBytesTotal = 2ull * 1024ull * 1024ull * 1024ull;
    std::uint64_t maxGpuBytesTotal = 4ull * 1024ull * 1024ull * 1024ull;
    std::uint64_t maxResidentBytesTotal = 6ull * 1024ull * 1024ull * 1024ull;
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

inline Limits boundedLimits(const Limits& requested) noexcept
{
    const Limits hard;
    Limits result;
    result.maxEntries = std::min(requested.maxEntries, hard.maxEntries);
    result.maxOwnedResourcesPerEntry = std::min(requested.maxOwnedResourcesPerEntry,
                                                hard.maxOwnedResourcesPerEntry);
    result.maxBorrowedResourcesPerEntry = std::min(requested.maxBorrowedResourcesPerEntry,
                                                   hard.maxBorrowedResourcesPerEntry);
    result.maxDecodedBytesPerEntry = std::min(requested.maxDecodedBytesPerEntry,
                                              hard.maxDecodedBytesPerEntry);
    result.maxGpuBytesPerEntry = std::min(requested.maxGpuBytesPerEntry,
                                          hard.maxGpuBytesPerEntry);
    result.maxResidentBytesPerEntry = std::min(requested.maxResidentBytesPerEntry,
                                               hard.maxResidentBytesPerEntry);
    result.maxDecodedBytesTotal = std::min(requested.maxDecodedBytesTotal,
                                           hard.maxDecodedBytesTotal);
    result.maxGpuBytesTotal = std::min(requested.maxGpuBytesTotal,
                                       hard.maxGpuBytesTotal);
    result.maxResidentBytesTotal = std::min(requested.maxResidentBytesTotal,
                                            hard.maxResidentBytesTotal);
    return result;
}

template <std::size_t Size>
inline bool anyNonZero(const std::array<std::uint8_t, Size>& value) noexcept
{
    return std::any_of(value.begin(), value.end(), [](std::uint8_t byte)
    {
        return byte != 0;
    });
}
} // namespace detail

struct BackendIdentity
{
    std::array<std::uint8_t, 16> implementation {};
    std::uint64_t implementationGeneration = 0;

    friend bool operator==(const BackendIdentity& left,
                           const BackendIdentity& right) noexcept
    {
        return left.implementation == right.implementation
            && left.implementationGeneration == right.implementationGeneration;
    }

    friend bool operator!=(const BackendIdentity& left,
                           const BackendIdentity& right) noexcept
    {
        return !(left == right);
    }
};

struct Context
{
    std::uint64_t helperGeneration = 0;
    BackendIdentity backend;

    friend bool operator==(const Context& left, const Context& right) noexcept
    {
        return left.helperGeneration == right.helperGeneration
            && left.backend == right.backend;
    }

    friend bool operator!=(const Context& left, const Context& right) noexcept
    {
        return !(left == right);
    }
};

inline bool valid(const Context& context) noexcept
{
    return context.helperGeneration != 0
        && context.backend.implementationGeneration != 0
        && detail::anyNonZero(context.backend.implementation);
}

struct Key
{
    visualmodelasset::Identity asset;
    BackendIdentity backend;
    std::uint64_t helperGeneration = 0;
    std::uint64_t decodedBytes = 0;
    std::uint64_t gpuBytes = 0;

    friend bool operator==(const Key& left, const Key& right) noexcept
    {
        return left.asset == right.asset
            && left.backend == right.backend
            && left.helperGeneration == right.helperGeneration
            && left.decodedBytes == right.decodedBytes
            && left.gpuBytes == right.gpuBytes;
    }

    friend bool operator!=(const Key& left, const Key& right) noexcept
    {
        return !(left == right);
    }
};

enum class AdmissionAction
{
    Rejected = 0,
    ReserveDecodeAndUpload,
    PendingExactHit,
    ReuseExactHit
};

enum class Failure
{
    None = 0,
    InvalidContext,
    InvalidAssetIdentity,
    InvalidByteDeclaration,
    ByteCountOverflow,
    PerEntryLimitExceeded,
    GlobalEntryLimitExceeded,
    GlobalByteLimitExceeded,
    ReservationIdExhausted,
    UnknownReservation,
    ResourceCountExceeded,
    InvalidResource,
    OwnedResourceByteMismatch
};

inline constexpr std::string_view token(Failure failure) noexcept
{
    switch (failure)
    {
        case Failure::None: return "none";
        case Failure::InvalidContext: return "invalidContext";
        case Failure::InvalidAssetIdentity: return "invalidAssetIdentity";
        case Failure::InvalidByteDeclaration: return "invalidByteDeclaration";
        case Failure::ByteCountOverflow: return "byteCountOverflow";
        case Failure::PerEntryLimitExceeded: return "perEntryLimitExceeded";
        case Failure::GlobalEntryLimitExceeded: return "globalEntryLimitExceeded";
        case Failure::GlobalByteLimitExceeded: return "globalByteLimitExceeded";
        case Failure::ReservationIdExhausted: return "reservationIdExhausted";
        case Failure::UnknownReservation: return "unknownReservation";
        case Failure::ResourceCountExceeded: return "resourceCountExceeded";
        case Failure::InvalidResource: return "invalidResource";
        case Failure::OwnedResourceByteMismatch: return "ownedResourceByteMismatch";
    }
    return {};
}

struct Admission
{
    AdmissionAction action = AdmissionAction::Rejected;
    Failure failure = Failure::None;
    std::uint64_t reservationId = 0;
    Key key;
};

enum class ResourceKind
{
    DecodedModel = 1,
    GpuBufferOrTexture = 2
};

enum class ResourceOwnership
{
    HelperOwned = 1,
    BorrowedGraph = 2
};

enum class InvalidationReason
{
    Relink = 1,
    Reimport,
    HelperRestart,
    BackendChange,
    ExplicitClear,
    Shutdown
};

inline constexpr std::string_view token(InvalidationReason reason) noexcept
{
    switch (reason)
    {
        case InvalidationReason::Relink: return "relink";
        case InvalidationReason::Reimport: return "reimport";
        case InvalidationReason::HelperRestart: return "helperRestart";
        case InvalidationReason::BackendChange: return "backendChange";
        case InvalidationReason::ExplicitClear: return "explicitClear";
        case InvalidationReason::Shutdown: return "shutdown";
    }
    return {};
}

struct Resource
{
    ResourceKind kind = ResourceKind::DecodedModel;
    ResourceOwnership ownership = ResourceOwnership::HelperOwned;
    std::uint64_t id = 0;
    std::uint64_t byteCount = 0;
};

struct ResourceSet
{
    std::vector<Resource> resources;
};

class ResourceReleaser
{
public:
    virtual ~ResourceReleaser() = default;
    virtual void release(ResourceKind kind,
                         std::uint64_t id,
                         InvalidationReason reason) noexcept = 0;
};

struct InvalidationResult
{
    std::size_t installedEntries = 0;
    std::size_t pendingReservations = 0;
    std::uint64_t decodedBytes = 0;
    std::uint64_t gpuBytes = 0;
};

class Entry final
{
public:
    Entry(const Entry&) = delete;
    Entry& operator=(const Entry&) = delete;
    Entry(Entry&&) = delete;
    Entry& operator=(Entry&&) = delete;

    const Key& key() const noexcept { return key_; }
    const ResourceSet& resources() const noexcept { return resources_; }

private:
    friend class Cache;

    Entry(Key key, ResourceSet resources)
        : key_(std::move(key)), resources_(std::move(resources))
    {
    }

    const Key key_;
    const ResourceSet resources_;
};

class Cache final
{
public:
    explicit Cache(ResourceReleaser& releaser, const Limits& requestedLimits = {})
        : releaser_(releaser), limits_(detail::boundedLimits(requestedLimits))
    {
    }

    ~Cache()
    {
        clear(InvalidationReason::Shutdown);
    }

    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = delete;
    Cache& operator=(Cache&&) = delete;

    Failure setContext(const Context& next)
    {
        if (!valid(next))
            return Failure::InvalidContext;

        if (!context_)
        {
            context_ = next;
            return Failure::None;
        }
        if (*context_ == next)
            return Failure::None;

        const auto reason = context_->helperGeneration != next.helperGeneration
            ? InvalidationReason::HelperRestart
            : InvalidationReason::BackendChange;
        clear(reason);
        context_ = next;
        return Failure::None;
    }

    Admission request(const visualmodelasset::Identity& asset,
                      std::uint64_t decodedBytes,
                      std::uint64_t gpuBytes)
    {
        Admission result;
        if (!context_ || !valid(*context_))
            return reject(std::move(result), Failure::InvalidContext);

        visualmodelasset::IdentityFailure identityFailure = visualmodelasset::IdentityFailure::None;
        if (!visualmodelasset::validate(asset, identityFailure))
            return reject(std::move(result), Failure::InvalidAssetIdentity);
        if (decodedBytes == 0 || gpuBytes == 0)
            return reject(std::move(result), Failure::InvalidByteDeclaration);

        result.key = { asset, context_->backend, context_->helperGeneration,
                       decodedBytes, gpuBytes };
        if (findInstalled(result.key) != nullptr)
        {
            result.action = AdmissionAction::ReuseExactHit;
            return result;
        }
        if (const auto* pending = findPending(result.key))
        {
            result.action = AdmissionAction::PendingExactHit;
            result.reservationId = pending->id;
            return result;
        }

        std::uint64_t residentBytes = 0;
        if (!detail::checkedAdd(decodedBytes, gpuBytes, residentBytes))
            return reject(std::move(result), Failure::ByteCountOverflow);
        if (decodedBytes > limits_.maxDecodedBytesPerEntry
            || gpuBytes > limits_.maxGpuBytesPerEntry
            || residentBytes > limits_.maxResidentBytesPerEntry)
            return reject(std::move(result), Failure::PerEntryLimitExceeded);

        if (installed_.size() + pending_.size() >= limits_.maxEntries)
            return reject(std::move(result), Failure::GlobalEntryLimitExceeded);

        std::uint64_t nextDecoded = 0;
        std::uint64_t nextGpu = 0;
        std::uint64_t currentResident = 0;
        std::uint64_t nextResident = 0;
        if (!detail::checkedAdd(reservedDecodedBytes_, decodedBytes, nextDecoded)
            || !detail::checkedAdd(reservedGpuBytes_, gpuBytes, nextGpu)
            || !detail::checkedAdd(reservedDecodedBytes_, reservedGpuBytes_, currentResident)
            || !detail::checkedAdd(currentResident, residentBytes, nextResident))
            return reject(std::move(result), Failure::ByteCountOverflow);
        if (nextDecoded > limits_.maxDecodedBytesTotal
            || nextGpu > limits_.maxGpuBytesTotal
            || nextResident > limits_.maxResidentBytesTotal)
            return reject(std::move(result), Failure::GlobalByteLimitExceeded);

        if (nextReservationId_ == 0)
            return reject(std::move(result), Failure::ReservationIdExhausted);
        const auto reservationId = nextReservationId_++;
        pending_.push_back({ reservationId, result.key });
        reservedDecodedBytes_ = nextDecoded;
        reservedGpuBytes_ = nextGpu;
        result.action = AdmissionAction::ReserveDecodeAndUpload;
        result.reservationId = reservationId;
        return result;
    }

    Failure publish(std::uint64_t reservationId, ResourceSet resources)
    {
        const auto pending = findPendingById(reservationId);
        if (pending == pending_.end())
            return Failure::UnknownReservation;

        const auto validation = validateResources(pending->key, resources);
        if (validation != Failure::None)
            return validation;

        installed_.push_back(std::unique_ptr<const Entry>(
            new Entry(pending->key, std::move(resources))));
        pending_.erase(pending);
        return Failure::None;
    }

    bool cancel(std::uint64_t reservationId) noexcept
    {
        const auto pending = findPendingById(reservationId);
        if (pending == pending_.end())
            return false;
        releaseBudget(pending->key);
        pending_.erase(pending);
        return true;
    }

    InvalidationResult invalidateAsset(std::string_view stableAssetId,
                                       InvalidationReason reason) noexcept
    {
        InvalidationResult result;
        if (reason != InvalidationReason::Relink
            && reason != InvalidationReason::Reimport)
            return result;

        for (auto entry = installed_.begin(); entry != installed_.end();)
        {
            if ((*entry)->key().asset.stableAssetId != stableAssetId)
            {
                ++entry;
                continue;
            }
            accumulate(result, (*entry)->key());
            releaseResources((*entry)->resources(), reason);
            releaseBudget((*entry)->key());
            entry = installed_.erase(entry);
        }
        for (auto pending = pending_.begin(); pending != pending_.end();)
        {
            if (pending->key.asset.stableAssetId != stableAssetId)
            {
                ++pending;
                continue;
            }
            ++result.pendingReservations;
            result.decodedBytes += pending->key.decodedBytes;
            result.gpuBytes += pending->key.gpuBytes;
            releaseBudget(pending->key);
            pending = pending_.erase(pending);
        }
        return result;
    }

    InvalidationResult clear(InvalidationReason reason = InvalidationReason::ExplicitClear) noexcept
    {
        InvalidationResult result;
        for (const auto& entry : installed_)
        {
            accumulate(result, entry->key());
            releaseResources(entry->resources(), reason);
        }
        result.pendingReservations = pending_.size();
        for (const auto& pending : pending_)
        {
            result.decodedBytes += pending.key.decodedBytes;
            result.gpuBytes += pending.key.gpuBytes;
        }
        installed_.clear();
        pending_.clear();
        reservedDecodedBytes_ = 0;
        reservedGpuBytes_ = 0;
        return result;
    }

    const Entry* find(const Key& key) const noexcept
    {
        return findInstalled(key);
    }

    const std::optional<Context>& context() const noexcept { return context_; }
    const Limits& limits() const noexcept { return limits_; }
    std::size_t installedEntries() const noexcept { return installed_.size(); }
    std::size_t pendingReservations() const noexcept { return pending_.size(); }
    std::uint64_t reservedDecodedBytes() const noexcept { return reservedDecodedBytes_; }
    std::uint64_t reservedGpuBytes() const noexcept { return reservedGpuBytes_; }

private:
    struct Reservation
    {
        std::uint64_t id = 0;
        Key key;
    };

    using PendingIterator = std::vector<Reservation>::iterator;

    static Admission reject(Admission result, Failure failure) noexcept
    {
        result.action = AdmissionAction::Rejected;
        result.failure = failure;
        result.reservationId = 0;
        return result;
    }

    const Entry* findInstalled(const Key& key) const noexcept
    {
        const auto found = std::find_if(installed_.begin(), installed_.end(),
            [&key](const auto& entry) { return entry->key() == key; });
        return found == installed_.end() ? nullptr : found->get();
    }

    const Reservation* findPending(const Key& key) const noexcept
    {
        const auto found = std::find_if(pending_.begin(), pending_.end(),
            [&key](const auto& reservation) { return reservation.key == key; });
        return found == pending_.end() ? nullptr : &*found;
    }

    PendingIterator findPendingById(std::uint64_t id) noexcept
    {
        return std::find_if(pending_.begin(), pending_.end(),
            [id](const auto& reservation) { return reservation.id == id; });
    }

    Failure validateResources(const Key& key, const ResourceSet& resources) const
    {
        std::size_t ownedCount = 0;
        std::size_t borrowedCount = 0;
        std::uint64_t decodedBytes = 0;
        std::uint64_t gpuBytes = 0;
        std::unordered_set<std::uint64_t> decodedIds;
        std::unordered_set<std::uint64_t> gpuIds;

        for (const auto& resource : resources.resources)
        {
            if (resource.id == 0
                || (resource.kind != ResourceKind::DecodedModel
                    && resource.kind != ResourceKind::GpuBufferOrTexture)
                || (resource.ownership != ResourceOwnership::HelperOwned
                    && resource.ownership != ResourceOwnership::BorrowedGraph))
                return Failure::InvalidResource;

            auto& ids = resource.kind == ResourceKind::DecodedModel ? decodedIds : gpuIds;
            if (!ids.insert(resource.id).second)
                return Failure::InvalidResource;

            if (resource.ownership == ResourceOwnership::BorrowedGraph)
            {
                ++borrowedCount;
                if (resource.byteCount != 0)
                    return Failure::InvalidResource;
                continue;
            }

            ++ownedCount;
            if (resource.byteCount == 0)
                return Failure::InvalidResource;
            auto& total = resource.kind == ResourceKind::DecodedModel
                ? decodedBytes : gpuBytes;
            std::uint64_t next = 0;
            if (!detail::checkedAdd(total, resource.byteCount, next))
                return Failure::ByteCountOverflow;
            total = next;
        }

        if (ownedCount > limits_.maxOwnedResourcesPerEntry
            || borrowedCount > limits_.maxBorrowedResourcesPerEntry)
            return Failure::ResourceCountExceeded;
        if (decodedBytes != key.decodedBytes || gpuBytes != key.gpuBytes)
            return Failure::OwnedResourceByteMismatch;
        return Failure::None;
    }

    void releaseBudget(const Key& key) noexcept
    {
        reservedDecodedBytes_ -= key.decodedBytes;
        reservedGpuBytes_ -= key.gpuBytes;
    }

    void releaseResources(const ResourceSet& resources,
                          InvalidationReason reason) noexcept
    {
        // Release GPU objects before decoded storage. Borrowed graph resources
        // are references only and never enter the helper-owned delete path.
        for (auto resource = resources.resources.rbegin();
             resource != resources.resources.rend(); ++resource)
            if (resource->ownership == ResourceOwnership::HelperOwned
                && resource->kind == ResourceKind::GpuBufferOrTexture)
                releaser_.release(resource->kind, resource->id, reason);
        for (auto resource = resources.resources.rbegin();
             resource != resources.resources.rend(); ++resource)
            if (resource->ownership == ResourceOwnership::HelperOwned
                && resource->kind == ResourceKind::DecodedModel)
                releaser_.release(resource->kind, resource->id, reason);
    }

    static void accumulate(InvalidationResult& result, const Key& key) noexcept
    {
        ++result.installedEntries;
        result.decodedBytes += key.decodedBytes;
        result.gpuBytes += key.gpuBytes;
    }

    ResourceReleaser& releaser_;
    const Limits limits_;
    std::optional<Context> context_;
    std::vector<std::unique_ptr<const Entry>> installed_;
    std::vector<Reservation> pending_;
    std::uint64_t reservedDecodedBytes_ = 0;
    std::uint64_t reservedGpuBytes_ = 0;
    std::uint64_t nextReservationId_ = 1;
};

static_assert(!kOpensSourcePaths);
static_assert(!kDecodesImages);
static_assert(!kPerformsGpuUpload);
static_assert(!kAllowsCpuProductionFallback);
static_assert(!kAcceptsCatalogAsModelPayload);
} // namespace videohelper::modelcache
