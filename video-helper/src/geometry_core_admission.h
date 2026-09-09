#pragma once

#include "../../shared/GeometryCoreTransport.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace videohelper::geometry {
struct BackendCapabilities final {
  std::string backendIdentity;
  std::string deviceIdentity;
  bool nativeGpuAvailable = false;
  bool geometryCoreExecution = false;
  bool immutableSourceBuffers = false;
  bool stableElementIds = false;
  bool typedFieldEvaluation = false;
  bool gpuInstancingWithoutMeshExpansion = false;
  std::uint32_t supportedCarriers = 0;
  videowire::geometry::ResourceLimits limits{};
};

inline constexpr std::uint32_t
carrierBit(videowire::geometry::CarrierKind carrier) noexcept {
  return carrier == videowire::geometry::CarrierKind::unspecified
             ? 0u
             : 1u << static_cast<unsigned>(carrier);
}

class BackendCapabilitySource {
public:
  virtual ~BackendCapabilitySource() = default;
  virtual BackendCapabilities geometryCoreCapabilities() const = 0;
};

enum class PlanUse { preview, exportRender };

struct ExecutionBudgetReceipt final {
  videowire::geometry::BudgetReceipt value;
  videowire::geometry::ResourceLimits effectiveLimits;
  std::uint64_t backendBinding = 0;
};

// A plan is owned by one presentation or export request.  Keeping every
// generation in the key prevents a stale helper/device plan from being reused
// after either side restarts.
struct PlanOwnerIdentity final {
  std::uint64_t projectGeneration = 0;
  std::uint64_t helperGeneration = 0;
  std::uint64_t deviceGeneration = 0;
  std::uint64_t clipIdentity = 0;
  std::uint64_t planGeneration = 0;
  PlanUse use = PlanUse::preview;

  bool valid() const noexcept {
    return projectGeneration != 0 && helperGeneration != 0 &&
           deviceGeneration != 0 && clipIdentity != 0 && planGeneration != 0;
  }
  bool operator==(const PlanOwnerIdentity &other) const noexcept {
    return projectGeneration == other.projectGeneration &&
           helperGeneration == other.helperGeneration &&
           deviceGeneration == other.deviceGeneration &&
           clipIdentity == other.clipIdentity &&
           planGeneration == other.planGeneration && use == other.use;
  }
  bool operator<(const PlanOwnerIdentity &other) const noexcept {
    return std::tie(projectGeneration, helperGeneration, deviceGeneration,
                    clipIdentity, planGeneration, use) <
           std::tie(other.projectGeneration, other.helperGeneration,
                    other.deviceGeneration, other.clipIdentity,
                    other.planGeneration, other.use);
  }
};

class AdmittedPlanValue final {
public:
  AdmittedPlanValue(const AdmittedPlanValue &) = default;
  AdmittedPlanValue(AdmittedPlanValue &&) noexcept = default;
  AdmittedPlanValue &operator=(const AdmittedPlanValue &) = delete;
  AdmittedPlanValue &operator=(AdmittedPlanValue &&) = delete;
  const videowire::geometry::AdmittedValue &value() const noexcept {
    return value_;
  }
  const BackendCapabilities &capabilities() const noexcept {
    return capabilities_;
  }
  const ExecutionBudgetReceipt &receipt() const noexcept { return receipt_; }

private:
  friend std::optional<AdmittedPlanValue>
  admitPlanValue(videowire::geometry::ValueDescriptor,
                 const videowire::geometry::PortContract &,
                 const videowire::geometry::ResourceLimits &,
                 const videowire::geometry::AdmissionContext &,
                 const BackendCapabilitySource &, std::string &);
  AdmittedPlanValue(videowire::geometry::AdmittedValue value,
                    BackendCapabilities capabilities,
                    ExecutionBudgetReceipt receipt)
      : value_(std::move(value)), capabilities_(std::move(capabilities)),
        receipt_(std::move(receipt)) {}
  const videowire::geometry::AdmittedValue value_;
  const BackendCapabilities capabilities_;
  const ExecutionBudgetReceipt receipt_;
};

inline std::optional<AdmittedPlanValue>
admitPlanValue(videowire::geometry::ValueDescriptor source,
               const videowire::geometry::PortContract &contract,
               const videowire::geometry::ResourceLimits &requested,
               const videowire::geometry::AdmissionContext &context,
               const BackendCapabilitySource &backend, std::string &error) {
  error.clear();
  auto capabilities = backend.geometryCoreCapabilities();
  if (capabilities.backendIdentity.empty() ||
      !capabilities.nativeGpuAvailable || !capabilities.geometryCoreExecution ||
      !capabilities.immutableSourceBuffers || !capabilities.stableElementIds) {
    error = "production backend does not admit the Geometry Core execution "
            "contract";
    return std::nullopt;
  }
  if ((contract.carrier == videowire::geometry::CarrierKind::field &&
       !capabilities.typedFieldEvaluation) ||
      (contract.carrier == videowire::geometry::CarrierKind::instances3D &&
       !capabilities.gpuInstancingWithoutMeshExpansion)) {
    error = "production backend lacks the required typed geometry operation";
    return std::nullopt;
  }
  if ((capabilities.supportedCarriers & carrierBit(contract.carrier)) == 0) {
    error = "production backend does not execute this Geometry Core carrier";
    return std::nullopt;
  }
  std::string ignored;
  if (!videowire::geometry::validateLimits(capabilities.limits, ignored)) {
    error = "production backend reported invalid Geometry Core limits";
    return std::nullopt;
  }
  auto effective = requested;
#define ARBIT_GEOMETRY_TIGHTEN(name)                                           \
  effective.name = std::min(effective.name, capabilities.limits.name)
  ARBIT_GEOMETRY_TIGHTEN(maxVertices);
  ARBIT_GEOMETRY_TIGHTEN(maxIndices);
  ARBIT_GEOMETRY_TIGHTEN(maxPoints);
  ARBIT_GEOMETRY_TIGHTEN(maxCurvePoints);
  ARBIT_GEOMETRY_TIGHTEN(maxSplines);
  ARBIT_GEOMETRY_TIGHTEN(maxInstances);
  ARBIT_GEOMETRY_TIGHTEN(maxFieldElements);
  ARBIT_GEOMETRY_TIGHTEN(maxAttributes);
  ARBIT_GEOMETRY_TIGHTEN(maxOperations);
  ARBIT_GEOMETRY_TIGHTEN(maxDispatches);
  ARBIT_GEOMETRY_TIGHTEN(maxBufferBytes);
#undef ARBIT_GEOMETRY_TIGHTEN
  auto admitted = videowire::geometry::admitValue(std::move(source), contract,
                                                  effective, context, error);
  if (!admitted)
    return std::nullopt;
  ExecutionBudgetReceipt receipt{admitted->receipt(), effective,
                                 admitted->receipt().deterministicDigest};
  const auto bind = [](std::uint64_t hash, std::uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
  };
  for (const auto value :
       {effective.maxVertices, effective.maxIndices, effective.maxPoints,
        effective.maxCurvePoints, effective.maxSplines, effective.maxInstances,
        effective.maxFieldElements, effective.maxAttributes,
        effective.maxOperations, effective.maxDispatches,
        effective.maxBufferBytes})
    receipt.backendBinding = bind(receipt.backendBinding, value);
  for (const unsigned char character : capabilities.backendIdentity)
    receipt.backendBinding = bind(receipt.backendBinding, character);
  return AdmittedPlanValue(std::move(*admitted), std::move(capabilities),
                           std::move(receipt));
}

struct RuntimeAdmission final {
  PlanUse use = PlanUse::preview;
  PlanOwnerIdentity owner;
  std::shared_ptr<const AdmittedPlanValue> plan;
  std::shared_ptr<const std::uint8_t> ownerLease;
  struct NativeResources final {
    std::mutex mutex;
    std::shared_ptr<const void> value;
  };
  std::shared_ptr<NativeResources> nativeResources;
};

class GeometryCorePlanRuntime final {
public:
  explicit GeometryCorePlanRuntime(const BackendCapabilitySource &backend,
                                   std::size_t maximumOwners = 32)
      : backend_(backend), maximumOwners_(maximumOwners) {}
  bool reconcileGeneration(std::uint64_t projectGeneration,
                           std::uint64_t helperGeneration,
                           std::uint64_t deviceGeneration,
                           std::string &error) {
    error.clear();
    if (projectGeneration == 0 || helperGeneration == 0 || deviceGeneration == 0) {
      error = "Geometry Core generation replacement identity is incomplete";
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const bool changed = activeProjectGeneration_ != projectGeneration ||
                         activeHelperGeneration_ != helperGeneration ||
                         activeDeviceGeneration_ != deviceGeneration;
    for (auto it = cache_.begin(); it != cache_.end();) {
      const auto matches = it->first.projectGeneration == projectGeneration &&
                           it->first.helperGeneration == helperGeneration &&
                           it->first.deviceGeneration == deviceGeneration;
      if (!matches && it->second.ownerLease.use_count() == 1)
        it = cache_.erase(it);
      else
        ++it;
    }
    if (changed) {
      activeProjectGeneration_ = projectGeneration;
      activeHelperGeneration_ = helperGeneration;
      activeDeviceGeneration_ = deviceGeneration;
      clock_ = 0;
      admissionsInGeneration_ = 0;
    }
    return true;
  }
  std::uint64_t admissionsInGeneration() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return admissionsInGeneration_;
  }
  std::optional<RuntimeAdmission>
  admitPreviewText(const std::string &payload,
                   const videowire::geometry::ResourceLimits &limits,
                   const videowire::geometry::AdmissionContext &context,
                   std::string &error) {
    auto bytes = videowire::geometry::decodeLoweredPlanText(payload, error);
    return bytes ? admitPreview(*bytes, limits, context, error) : std::nullopt;
  }
  std::optional<RuntimeAdmission>
  admitExportText(const std::string &payload,
                  const videowire::geometry::ResourceLimits &limits,
                  const videowire::geometry::AdmissionContext &context,
                  std::string &error) {
    auto bytes = videowire::geometry::decodeLoweredPlanText(payload, error);
    return bytes ? admitExport(*bytes, limits, context, error) : std::nullopt;
  }
  std::optional<RuntimeAdmission>
  admitPreview(const std::vector<std::uint8_t> &loweredPlan,
               const videowire::geometry::ResourceLimits &limits,
               const videowire::geometry::AdmissionContext &context,
               std::string &error) {
    return admitLowered(loweredPlan, limits, context, PlanUse::preview, error);
  }
  std::optional<RuntimeAdmission>
  admitPreview(const PlanOwnerIdentity &owner,
               const std::vector<std::uint8_t> &loweredPlan,
               const videowire::geometry::ResourceLimits &limits,
               const videowire::geometry::AdmissionContext &context,
               std::string &error) {
    return admitLowered(loweredPlan, limits, context, owner, error);
  }
  std::optional<RuntimeAdmission>
  admitExport(const std::vector<std::uint8_t> &loweredPlan,
              const videowire::geometry::ResourceLimits &limits,
              const videowire::geometry::AdmissionContext &context,
              std::string &error) {
    return admitLowered(loweredPlan, limits, context, PlanUse::exportRender,
                        error);
  }
  std::optional<RuntimeAdmission>
  admitExport(const PlanOwnerIdentity &owner,
              const std::vector<std::uint8_t> &loweredPlan,
              const videowire::geometry::ResourceLimits &limits,
              const videowire::geometry::AdmissionContext &context,
              std::string &error) {
    return admitLowered(loweredPlan, limits, context, owner, error);
  }

private:
  std::optional<RuntimeAdmission>
  admitLowered(const std::vector<std::uint8_t> &bytes,
               const videowire::geometry::ResourceLimits &limits,
               const videowire::geometry::AdmissionContext &context,
               PlanUse use, std::string &error) {
    auto lowered = videowire::geometry::decodeLoweredRuntimePlan(bytes, error);
    if (!lowered)
      return std::nullopt;
    return admit(lowered->runtimeValue, lowered->contract, limits, context,
                 PlanOwnerIdentity{1, 1, 1, 1, 1, use}, error);
  }
  std::optional<RuntimeAdmission>
  admitLowered(const std::vector<std::uint8_t> &bytes,
               const videowire::geometry::ResourceLimits &limits,
               const videowire::geometry::AdmissionContext &context,
               const PlanOwnerIdentity &owner, std::string &error) {
    if (!owner.valid()) {
      error = "Geometry Core plan owner identity is incomplete";
      return std::nullopt;
    }
    auto lowered = videowire::geometry::decodeLoweredRuntimePlan(bytes, error);
    if (!lowered)
      return std::nullopt;
    return admit(lowered->runtimeValue, lowered->contract, limits, context,
                 owner, error);
  }
  std::optional<RuntimeAdmission>
  admit(const std::vector<std::uint8_t> &bytes,
        const videowire::geometry::PortContract &contract,
        const videowire::geometry::ResourceLimits &limits,
        const videowire::geometry::AdmissionContext &context,
        const PlanOwnerIdentity &owner,
        std::string &error) {
    auto decoded = videowire::geometry::decodeRuntimeValue(
        bytes, contract, limits, context, error);
    if (!decoded)
      return std::nullopt;
    auto plan = admitPlanValue(decoded->descriptor(), contract, limits, context,
                               backend_, error);
    if (!plan)
      return std::nullopt;
    std::uint64_t key = plan->receipt().backendBinding;
    for (const auto byte : bytes) {
      key ^= byte;
      key *= 1099511628211ull;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (activeProjectGeneration_ != 0
        && (owner.projectGeneration != activeProjectGeneration_
            || owner.helperGeneration != activeHelperGeneration_
            || owner.deviceGeneration != activeDeviceGeneration_)) {
      error = "Geometry Core plan owner does not match the active generation";
      return std::nullopt;
    }
    const auto found = cache_.find(owner);
    if (found != cache_.end()) {
      if (found->second.planDigest != key) {
        error = "Geometry Core owner was already bound to a different immutable plan";
        return std::nullopt;
      }
      found->second.lastUse = ++clock_;
      return RuntimeAdmission{owner.use, owner, found->second.plan,
                              found->second.ownerLease,
                              found->second.nativeResources};
    }
    evictUnusedOwners();
    if (cache_.size() >= maximumOwners_) {
      error = "Geometry Core owner cache is full with live leases";
      return std::nullopt;
    }
    auto shared = std::make_shared<const AdmittedPlanValue>(std::move(*plan));
    auto ownerLease = std::make_shared<const std::uint8_t>(0);
    auto nativeResources = std::make_shared<RuntimeAdmission::NativeResources>();
    cache_.emplace(owner, Entry{key, ++clock_, shared, ownerLease, nativeResources});
    ++admissionsInGeneration_;
    return RuntimeAdmission{owner.use, owner, std::move(shared),
                            std::move(ownerLease), std::move(nativeResources)};
  }
  struct Entry final {
    std::uint64_t planDigest = 0;
    std::uint64_t lastUse = 0;
    std::shared_ptr<const AdmittedPlanValue> plan;
    std::shared_ptr<const std::uint8_t> ownerLease;
    std::shared_ptr<RuntimeAdmission::NativeResources> nativeResources;
  };
  void evictUnusedOwners() {
    while (cache_.size() >= maximumOwners_) {
      auto victim = cache_.end();
      for (auto it = cache_.begin(); it != cache_.end(); ++it)
        if (it->second.ownerLease.use_count() == 1 &&
            (victim == cache_.end() || it->second.lastUse < victim->second.lastUse))
          victim = it;
      if (victim == cache_.end()) return;
      cache_.erase(victim);
    }
  }
  const BackendCapabilitySource &backend_;
  const std::size_t maximumOwners_;
  mutable std::mutex mutex_;
  std::uint64_t clock_ = 0;
  std::uint64_t activeProjectGeneration_ = 0;
  std::uint64_t activeHelperGeneration_ = 0;
  std::uint64_t activeDeviceGeneration_ = 0;
  std::uint64_t admissionsInGeneration_ = 0;
  std::map<PlanOwnerIdentity, Entry> cache_;
};
} // namespace videohelper::geometry
