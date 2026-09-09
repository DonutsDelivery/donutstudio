#include "../src/model_cache_contract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace videohelper::modelcache;

static_assert(!std::is_copy_constructible_v<Cache>);
static_assert(!std::is_move_constructible_v<Cache>);
static_assert(!std::is_copy_constructible_v<Entry>);
static_assert(std::is_same_v<decltype(std::declval<const Entry&>().key()), const Key&>);
static_assert(!kOpensSourcePaths && !kDecodesImages && !kPerformsGpuUpload);
static_assert(!kAllowsCpuProductionFallback && !kAcceptsCatalogAsModelPayload);

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::string digest(char value)
{
    return std::string(64, value);
}

visualmodelasset::Identity asset(std::string id = "model-a", char fingerprint = 'a')
{
    return { std::move(id), digest(fingerprint) };
}

template <std::size_t Size>
std::array<std::uint8_t, Size> filled(std::uint8_t value)
{
    std::array<std::uint8_t, Size> result;
    result.fill(value);
    return result;
}

Context context(std::uint64_t helperGeneration = 7,
                std::uint64_t backendGeneration = 3,
                std::uint8_t implementation = 0x51)
{
    Context value;
    value.helperGeneration = helperGeneration;
    value.backend.implementation = filled<16>(implementation);
    value.backend.implementationGeneration = backendGeneration;
    return value;
}

Resource ownedDecoded(std::uint64_t id, std::uint64_t bytes)
{
    return { ResourceKind::DecodedModel, ResourceOwnership::HelperOwned, id, bytes };
}

Resource ownedGpu(std::uint64_t id, std::uint64_t bytes)
{
    return { ResourceKind::GpuBufferOrTexture, ResourceOwnership::HelperOwned, id, bytes };
}

Resource borrowedGpu(std::uint64_t id)
{
    return { ResourceKind::GpuBufferOrTexture, ResourceOwnership::BorrowedGraph, id, 0 };
}

struct ReleaseRecord
{
    ResourceKind kind = ResourceKind::DecodedModel;
    std::uint64_t id = 0;
    InvalidationReason reason = InvalidationReason::ExplicitClear;
};

class RecordingReleaser final : public ResourceReleaser
{
public:
    void release(ResourceKind kind,
                 std::uint64_t id,
                 InvalidationReason reason) noexcept override
    {
        records.push_back({ kind, id, reason });
    }

    bool released(std::uint64_t id, InvalidationReason reason) const
    {
        return std::any_of(records.begin(), records.end(), [id, reason](const auto& record)
        {
            return record.id == id && record.reason == reason;
        });
    }

    bool released(std::uint64_t id) const
    {
        return std::any_of(records.begin(), records.end(), [id](const auto& record)
        {
            return record.id == id;
        });
    }

    std::vector<ReleaseRecord> records;
};

Limits smallLimits()
{
    Limits limits;
    limits.maxEntries = 4;
    limits.maxOwnedResourcesPerEntry = 8;
    limits.maxBorrowedResourcesPerEntry = 4;
    limits.maxDecodedBytesPerEntry = 100;
    limits.maxGpuBytesPerEntry = 200;
    limits.maxResidentBytesPerEntry = 300;
    limits.maxDecodedBytesTotal = 160;
    limits.maxGpuBytesTotal = 320;
    limits.maxResidentBytesTotal = 480;
    return limits;
}

Admission reserve(Cache& cache,
                  const visualmodelasset::Identity& identity,
                  std::uint64_t decodedBytes,
                  std::uint64_t gpuBytes)
{
    const auto admission = cache.request(identity, decodedBytes, gpuBytes);
    check(admission.action == AdmissionAction::ReserveDecodeAndUpload,
          "fixture request receives a decode/upload reservation");
    check(admission.failure == Failure::None && admission.reservationId != 0,
          "fixture reservation carries an ID and no failure");
    return admission;
}

ResourceSet resources(std::uint64_t decodedId,
                      std::uint64_t decodedBytes,
                      std::uint64_t gpuId,
                      std::uint64_t gpuBytes,
                      std::uint64_t borrowedId = 0)
{
    ResourceSet set;
    set.resources.push_back(ownedDecoded(decodedId, decodedBytes));
    set.resources.push_back(ownedGpu(gpuId, gpuBytes));
    if (borrowedId != 0)
        set.resources.push_back(borrowedGpu(borrowedId));
    return set;
}

void testPortableAssetIdentity()
{
    visualmodelasset::IdentityFailure failure = visualmodelasset::IdentityFailure::None;
    check(visualmodelasset::validate(asset(), failure),
          "stable model asset identity is admitted");
    check(failure == visualmodelasset::IdentityFailure::None,
          "valid identity clears its diagnostic");

    auto invalid = asset("../model.glb");
    check(!visualmodelasset::validate(invalid, failure)
              && failure == visualmodelasset::IdentityFailure::InvalidStableAssetId,
          "path-like model asset IDs are rejected");
    invalid = asset();
    invalid.contentSha256 = digest('A');
    check(!visualmodelasset::validate(invalid, failure)
              && failure == visualmodelasset::IdentityFailure::InvalidContentFingerprint,
          "noncanonical fingerprints are rejected");
    check(token(failure) == "invalidContentFingerprint",
          "asset identity failure has a stable token");
}

void testContextAndExactKey()
{
    RecordingReleaser releaser;
    Cache cache(releaser, smallLimits());
    auto invalidContext = context();
    invalidContext.helperGeneration = 0;
    check(cache.setContext(invalidContext) == Failure::InvalidContext,
          "zero helper generation is rejected");
    invalidContext = context();
    invalidContext.backend.implementation = {};
    check(cache.setContext(invalidContext) == Failure::InvalidContext,
          "anonymous backend implementation is rejected");
    check(cache.request(asset(), 10, 20).failure == Failure::InvalidContext,
          "cache request requires an admitted helper/backend context");

    const auto initialContext = context();
    check(cache.setContext(initialContext) == Failure::None,
          "valid helper/backend context is admitted");
    const auto admission = reserve(cache, asset(), 10, 20);
    check(admission.key.asset == asset()
              && admission.key.helperGeneration == initialContext.helperGeneration
              && admission.key.backend == initialContext.backend
              && admission.key.decodedBytes == 10
              && admission.key.gpuBytes == 20,
          "cache key freezes exact asset, helper, backend, and byte identities");

    const auto pendingHit = cache.request(asset(), 10, 20);
    check(pendingHit.action == AdmissionAction::PendingExactHit
              && pendingHit.reservationId == admission.reservationId
              && cache.pendingReservations() == 1,
          "exact pending requests deduplicate before allocation");

    const auto differentFootprint = cache.request(asset(), 11, 20);
    check(differentFootprint.action == AdmissionAction::ReserveDecodeAndUpload
              && differentFootprint.key != admission.key,
          "declared decoded and GPU bytes participate in the exact key");
    check(cache.cancel(differentFootprint.reservationId),
          "cancelled reservation releases its budget");
}

void testPreallocationBudgets()
{
    RecordingReleaser releaser;
    auto limits = smallLimits();
    limits.maxEntries = 2;
    limits.maxDecodedBytesTotal = 100;
    limits.maxGpuBytesTotal = 200;
    limits.maxResidentBytesTotal = 300;
    Cache cache(releaser, limits);
    check(cache.setContext(context()) == Failure::None, "budget fixture context admits");

    const auto first = reserve(cache, asset("model-a", 'a'), 60, 100);
    const auto exact = cache.request(asset("model-a", 'a'), 60, 100);
    check(exact.action == AdmissionAction::PendingExactHit,
          "exact hit bypasses full-entry and global-byte checks");

    auto rejected = cache.request(asset("model-b", 'b'), 41, 100);
    check(rejected.action == AdmissionAction::Rejected
              && rejected.failure == Failure::GlobalByteLimitExceeded,
          "pending decoded bytes count against the global limit before allocation");
    check(cache.pendingReservations() == 1
              && cache.reservedDecodedBytes() == 60
              && cache.reservedGpuBytes() == 100,
          "failed global admission does not mutate reservations");

    rejected = cache.request(asset("model-b", 'b'), 101, 1);
    check(rejected.failure == Failure::PerEntryLimitExceeded,
          "per-entry decoded limit is enforced before allocation");
    rejected = cache.request(asset("model-b", 'b'),
                             std::numeric_limits<std::uint64_t>::max(), 1);
    check(rejected.failure == Failure::ByteCountOverflow,
          "declared resident-byte overflow is rejected");
    rejected = cache.request(asset("model-b", 'b'), 0, 1);
    check(rejected.failure == Failure::InvalidByteDeclaration,
          "zero decoded byte declaration is rejected");

    check(cache.cancel(first.reservationId)
              && cache.reservedDecodedBytes() == 0
              && cache.reservedGpuBytes() == 0,
          "cancellation restores global budget before model allocation");
    check(!cache.cancel(first.reservationId),
          "reservation cannot be cancelled twice");

    Limits relaxed;
    relaxed.maxDecodedBytesPerEntry = std::numeric_limits<std::uint64_t>::max();
    Cache hardBounded(releaser, relaxed);
    check(hardBounded.limits().maxDecodedBytesPerEntry
              == Limits {}.maxDecodedBytesPerEntry,
          "caller cannot relax the hard decoded-model limit");

    std::uint64_t sum = 0;
    check(!detail::checkedAdd(std::numeric_limits<std::uint64_t>::max(), 1, sum),
          "checked addition rejects uint64 overflow");
}

void testEntryLimitAndMalformedIdentity()
{
    RecordingReleaser releaser;
    auto limits = smallLimits();
    limits.maxEntries = 1;
    Cache cache(releaser, limits);
    check(cache.setContext(context()) == Failure::None, "entry-limit fixture context admits");
    const auto first = reserve(cache, asset(), 10, 20);
    check(cache.request(asset("model-b", 'b'), 10, 20).failure
              == Failure::GlobalEntryLimitExceeded,
          "pending reservations count against the global entry limit");
    check(cache.request(asset("bad/id", 'b'), 10, 20).failure
              == Failure::InvalidAssetIdentity,
          "malformed stable asset ID is rejected before entry accounting");
    check(cache.cancel(first.reservationId), "entry-limit fixture cancels reservation");
}

void testPublishExactResourcesAndPerFrameHits()
{
    RecordingReleaser releaser;
    Cache cache(releaser, smallLimits());
    check(cache.setContext(context()) == Failure::None, "publish fixture context admits");
    const auto admission = reserve(cache, asset(), 60, 100);

    auto malformed = resources(10, 59, 20, 100);
    check(cache.publish(admission.reservationId, std::move(malformed))
              == Failure::OwnedResourceByteMismatch,
          "publication rejects a decoded byte receipt that differs from reservation");
    check(cache.pendingReservations() == 1 && cache.installedEntries() == 0
              && releaser.records.empty(),
          "failed publication keeps reservation and does not take resource ownership");

    ResourceSet duplicate;
    duplicate.resources = { ownedDecoded(10, 30), ownedDecoded(10, 30),
                            ownedGpu(20, 100) };
    check(cache.publish(admission.reservationId, std::move(duplicate))
              == Failure::InvalidResource,
          "publication rejects duplicate helper-owned resource IDs");

    auto exact = resources(10, 60, 20, 100, 30);
    check(cache.publish(admission.reservationId, std::move(exact)) == Failure::None,
          "exact helper-owned resource receipt publishes");
    check(cache.installedEntries() == 1 && cache.pendingReservations() == 0,
          "published reservation becomes one immutable installed entry");
    check(cache.reservedDecodedBytes() == 60 && cache.reservedGpuBytes() == 100,
          "installed entry retains its bounded global byte charge");

    for (int frame = 0; frame < 120; ++frame)
    {
        const auto hit = cache.request(asset(), 60, 100);
        check(hit.action == AdmissionAction::ReuseExactHit
                  && hit.reservationId == 0
                  && hit.key == admission.key,
              "per-frame request reuses exact cache hit without decode/upload reservation");
    }
    check(cache.installedEntries() == 1 && cache.pendingReservations() == 0
              && releaser.records.empty(),
          "frame loop neither reloads nor uploads nor releases cached model resources");
    const auto* entry = cache.find(admission.key);
    check(entry != nullptr && entry->resources().resources.size() == 3,
          "installed cache entry owns an immutable resource receipt");
}

void testResourcePublicationBounds()
{
    RecordingReleaser releaser;
    auto limits = smallLimits();
    limits.maxOwnedResourcesPerEntry = 2;
    limits.maxBorrowedResourcesPerEntry = 1;
    Cache cache(releaser, limits);
    check(cache.setContext(context()) == Failure::None,
          "resource-bound fixture context admits");
    auto admission = reserve(cache, asset(), 60, 100);

    ResourceSet tooMany;
    tooMany.resources = { ownedDecoded(1, 30), ownedDecoded(2, 30),
                          ownedGpu(3, 100) };
    check(cache.publish(admission.reservationId, std::move(tooMany))
              == Failure::ResourceCountExceeded,
          "helper-owned resource count is bounded before publication retention");

    ResourceSet chargedBorrow;
    chargedBorrow.resources = { ownedDecoded(1, 60), ownedGpu(2, 100), borrowedGpu(3) };
    chargedBorrow.resources.back().byteCount = 1;
    check(cache.publish(admission.reservationId, std::move(chargedBorrow))
              == Failure::InvalidResource,
          "borrowed graph resource cannot impersonate cache-owned GPU bytes");
    check(cache.cancel(admission.reservationId),
          "resource-bound fixture releases failed reservation");
}

void testRelinkAndReimportInvalidation()
{
    RecordingReleaser releaser;
    Cache cache(releaser, smallLimits());
    check(cache.setContext(context()) == Failure::None,
          "asset-invalidation fixture context admits");
    auto admission = reserve(cache, asset(), 60, 100);
    check(cache.publish(admission.reservationId, resources(100, 60, 200, 100, 300))
              == Failure::None,
          "relink fixture publishes");

    const auto relinked = cache.invalidateAsset("model-a", InvalidationReason::Relink);
    check(relinked.installedEntries == 1 && relinked.pendingReservations == 0
              && relinked.decodedBytes == 60 && relinked.gpuBytes == 100,
          "relink invalidates the exact installed model footprint");
    check(releaser.released(200, InvalidationReason::Relink)
              && releaser.released(100, InvalidationReason::Relink),
          "relink releases helper-owned GPU and decoded resources");
    check(!releaser.released(300),
          "relink never deletes a borrowed graph resource as helper-owned");
    check(cache.request(asset(), 60, 100).action
              == AdmissionAction::ReserveDecodeAndUpload,
          "same-content relink requires a fresh helper reservation");

    const auto reimported = cache.invalidateAsset("model-a", InvalidationReason::Reimport);
    check(reimported.pendingReservations == 1 && reimported.installedEntries == 0,
          "reimport cancels pending work for the stable asset ID");
    const auto replacement = cache.request(asset("model-a", 'b'), 50, 90);
    check(replacement.action == AdmissionAction::ReserveDecodeAndUpload
              && replacement.key.asset.contentSha256 == digest('b'),
          "reimported fingerprint receives a distinct immutable cache key");
    check(cache.publish(replacement.reservationId, resources(101, 50, 201, 90))
              == Failure::None,
          "replacement fingerprint publishes after reimport invalidation");
}

void testHelperAndBackendInvalidation()
{
    RecordingReleaser releaser;
    Cache cache(releaser, smallLimits());
    check(cache.setContext(context()) == Failure::None,
          "context-invalidation fixture context admits");
    auto admission = reserve(cache, asset(), 40, 80);
    check(cache.publish(admission.reservationId, resources(10, 40, 20, 80, 30))
              == Failure::None,
          "helper restart fixture publishes");

    check(cache.setContext(context(8, 3, 0x51)) == Failure::None,
          "new helper generation is admitted");
    check(cache.installedEntries() == 0 && cache.reservedDecodedBytes() == 0
              && releaser.released(10, InvalidationReason::HelperRestart)
              && releaser.released(20, InvalidationReason::HelperRestart)
              && !releaser.released(30),
          "helper restart drops owned resources and preserves borrowed graph ownership");

    admission = reserve(cache, asset(), 40, 80);
    check(cache.publish(admission.reservationId, resources(11, 40, 21, 80))
              == Failure::None,
          "backend generation fixture publishes");
    check(cache.setContext(context(8, 4, 0x51)) == Failure::None,
          "new backend implementation generation is admitted");
    check(cache.installedEntries() == 0
              && releaser.released(11, InvalidationReason::BackendChange)
              && releaser.released(21, InvalidationReason::BackendChange),
          "backend generation change invalidates helper-owned resources");

    admission = reserve(cache, asset(), 40, 80);
    check(cache.publish(admission.reservationId, resources(12, 40, 22, 80))
              == Failure::None,
          "backend implementation fixture publishes");
    check(cache.setContext(context(8, 4, 0x52)) == Failure::None,
          "different backend implementation is admitted");
    check(releaser.released(12, InvalidationReason::BackendChange)
              && releaser.released(22, InvalidationReason::BackendChange),
          "backend implementation change invalidates decoded and GPU caches");

    const auto stableCount = releaser.records.size();
    check(cache.setContext(context(8, 4, 0x52)) == Failure::None
              && releaser.records.size() == stableCount,
          "reapplying exact helper/backend context is a no-op");
}

void testStalePublishAndShutdown()
{
    RecordingReleaser releaser;
    {
        Cache cache(releaser, smallLimits());
        check(cache.setContext(context()) == Failure::None,
              "shutdown fixture context admits");
        auto admission = reserve(cache, asset(), 40, 80);
        check(cache.setContext(context(8, 3, 0x51)) == Failure::None,
              "helper restart cancels pending reservation");
        check(cache.publish(admission.reservationId, resources(1, 40, 2, 80))
                  == Failure::UnknownReservation,
              "stale allocation cannot publish after helper restart");

        admission = reserve(cache, asset(), 40, 80);
        check(cache.publish(admission.reservationId, resources(3, 40, 4, 80, 5))
                  == Failure::None,
              "shutdown fixture publishes current resources");
    }
    check(releaser.released(3, InvalidationReason::Shutdown)
              && releaser.released(4, InvalidationReason::Shutdown)
              && !releaser.released(5),
          "cache shutdown releases only helper-owned resources");
}
} // namespace

int main()
{
    testPortableAssetIdentity();
    testContextAndExactKey();
    testPreallocationBudgets();
    testEntryLimitAndMalformedIdentity();
    testPublishExactResourcesAndPerFrameHits();
    testResourcePublicationBounds();
    testRelinkAndReimportInvalidation();
    testHelperAndBackendInvalidation();
    testStalePublishAndShutdown();

    if (failures != 0)
    {
        std::cerr << failures << " model cache contract check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "model cache contract checks passed\n";
    return EXIT_SUCCESS;
}
