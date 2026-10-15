#pragma once
#include "../../shared/GeometrySurfaceMaterial.h"

#include "../../shared/GeometryCoreTransport.h"
#include "timeline_scene_family_budget.h"
#include "particle_body_caller_admission.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace videorender { class ParticlePointsBodySourceAdmission; class ParticleBodyCallerLease; }
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
  AdmittedPlanValue(AdmittedPlanValue &&) = default;
  AdmittedPlanValue &operator=(const AdmittedPlanValue &) = delete;
  AdmittedPlanValue &operator=(AdmittedPlanValue &&) = delete;
  const videowire::geometry::AdmittedValue &value() const noexcept {
    return value_;
  }
  const BackendCapabilities &capabilities() const noexcept {
    return capabilities_;
  }
  const ExecutionBudgetReceipt &receipt() const noexcept { return receipt_; }
  const videowire::geometry::PortContract &contract() const noexcept { return contract_; }

private:
  friend std::optional<AdmittedPlanValue>
  admitPlanValue(videowire::geometry::ValueDescriptor,
                 const videowire::geometry::PortContract &,
                 const videowire::geometry::ResourceLimits &,
                 const videowire::geometry::AdmissionContext &,
                 const BackendCapabilitySource &, std::string &);
  AdmittedPlanValue(videowire::geometry::AdmittedValue value,
                    BackendCapabilities capabilities,
                    ExecutionBudgetReceipt receipt, videowire::geometry::PortContract contract)
      : value_(std::move(value)), capabilities_(std::move(capabilities)),
        receipt_(std::move(receipt)), contract_(std::move(contract)) {}
  const videowire::geometry::AdmittedValue value_;
  const BackendCapabilities capabilities_;
  const ExecutionBudgetReceipt receipt_;
  const videowire::geometry::PortContract contract_;
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
  const bool usesFields = contract.carrier == videowire::geometry::CarrierKind::field ||
      !source.spectrumFields.empty() || !source.scoreFields.empty() ||
      std::any_of(source.operations.begin(), source.operations.end(), [](const auto &operation) {
        return videowire::geometry::isElementFieldQuery(operation.code) ||
               videowire::geometry::isConstructedFieldOperation(operation.code) ||
               operation.code == videowire::geometry::OperationCode::evaluateField ||
               operation.code == videowire::geometry::OperationCode::vectorDisplacement;
      });
  if ((usesFields &&
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
  for (const auto& deferred : context.deferredPoints)
    if (deferred.second.capacity > effective.maxPoints || deferred.second.capacity > effective.maxInstances
        || (contract.carrier==videowire::geometry::CarrierKind::instances3D && deferred.second.capacity>contract.maxInstances)) {
      error="Deferred Points3D capacity exceeds the effective resource limit"; return std::nullopt;
    }
  for (const auto& body : context.deferredBodyGeometry)
    if (!body.second.reference || body.second.reference->positions.size()>effective.maxVertices
        || body.second.reference->indices.size()>effective.maxIndices || effective.maxAttributes<3) {
      error="Deferred body Geometry3D prototype or paired attributes exceed the effective resource limit"; return std::nullopt;
    }
  for (const auto& sample : context.evaluatedPoints)
    if (sample.second.attributes.size() > effective.maxAttributes) {
      error="Evaluated Points3D attributes exceed the effective resource limit"; return std::nullopt;
    }
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
                           std::move(receipt), contract);
}

// This census belongs to one runtime, not to the process or native driver.
// Weak registrations survive cache eviction; owning aliases retain the token.
class RuntimeResidency final : public std::enable_shared_from_this<RuntimeResidency> {
  friend class GeometryCorePlanRuntime;
  struct Token final {
    std::shared_ptr<const videorender::ParticleBodyCallerLease> lease;
    const std::size_t bytes;
    std::weak_ptr<RuntimeResidency> residency;
    std::uint64_t identity=0;
    explicit Token(std::size_t value) : bytes(value) {}
    ~Token() {
      if (const auto owner=residency.lock()) owner->retire(identity);
    }
  };
  struct Record final {
    std::shared_ptr<const videorender::ParticleBodyCallerLease> lease;
    std::weak_ptr<Token> token;
  };
  template <typename T> struct Alias final {
    const std::shared_ptr<Token> token;
    const std::shared_ptr<T> value;
  };
  using Records=std::map<std::uint64_t,Record>;
  // Named logical control slots, not an allocator implementation size.
  static constexpr std::size_t controlBytes=4*sizeof(void*);
  static constexpr std::size_t recordBytes=sizeof(Records::value_type)+4*sizeof(void*)+controlBytes;
  static constexpr std::size_t tokenBytes=sizeof(Token)+2*controlBytes;
  static constexpr std::size_t aliasBytes=2*sizeof(std::shared_ptr<const void>)+controlBytes;
  RuntimeResidency()=default;
  std::shared_ptr<const videorender::ParticleBodyCallerLease> lease_;
  // A final alias may retire its token during a census visit on this thread.
  mutable std::recursive_mutex mutex_;
  Records records_;
  std::uint64_t next_=0;
  std::shared_ptr<videorender::ParticleBodyCallerLedger> caller_;
  bool attempt_=false;
  void retire(std::uint64_t identity) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    records_.erase(identity);
  }
  void prune() {
    for (auto it=records_.begin();it!=records_.end();)
      if (it->second.token.expired()) it=records_.erase(it); else ++it;
  }
  bool mutationAllowed(std::string& error,std::size_t byteSlots=0) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::size_t work=0;
    if (!videowire::geometry::checkedMul(byteSlots,8,work)
        || !videowire::geometry::checkedAdd(work,1,work)
        || (caller_ && !caller_->charge(work))) {
      error="Geometry Core residency mutation work is short"; return false;
    }
    if (attempt_ || (caller_ && !caller_->onCallerThread())) {
      error="Geometry Core residency mutation has no paired attempt schedule"; return false;
    }
    return true;
  }
  bool pending(const std::shared_ptr<videorender::ParticleBodyCallerLedger>& caller,
      std::size_t& bytes,std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (records_.size()==std::numeric_limits<std::size_t>::max()
        || !caller->charge(records_.size()+1) || (caller_ && caller_!=caller)) {
      error="Geometry Core residency requires its original caller and census work"; return false;
    }
    prune(); bytes=0;
    if (caller_) return true;
    using namespace videowire::geometry;
    if (!checkedAdd(sizeof(RuntimeResidency)+2*controlBytes,sizeof(videorender::ParticleBodyCallerLease),bytes)) return false;
    for (auto it=records_.begin();it!=records_.end();) {
      const auto record=it++;
      if (const auto token=record->second.token.lock())
        if (!checkedAdd(bytes,recordBytes+2*sizeof(videorender::ParticleBodyCallerLease),bytes)
            || !checkedAdd(bytes,token->bytes,bytes)) return false;
    }
    return true;
  }
  bool attach(const std::shared_ptr<videorender::ParticleBodyCallerLedger>& caller) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (caller_) return caller_==caller;
    auto lease=caller->reserve(sizeof(RuntimeResidency)+2*controlBytes,sizeof(RuntimeResidency));
    if (!lease) return false;
    for (auto it=records_.begin();it!=records_.end();) {
      const auto record=it++;
      if (auto token=record->second.token.lock()) {
        record->second.lease=caller->reserve(recordBytes,recordBytes);
        token->lease=caller->reserve(token->bytes,token->bytes);
        if (!record->second.lease || !token->lease) {
          for (auto undo=records_.begin();undo!=records_.end();) {
            const auto previous=undo++;
            previous->second.lease.reset();
            if (auto live=previous->second.token.lock()) live->lease.reset();
          }
          return false;
        }
      }
    }
    caller_=caller; lease_=std::move(lease); return true;
  }
  std::shared_ptr<Token> registerOwner(std::size_t bytes,std::size_t peak,std::size_t limit,std::string& error) {
    using namespace videowire::geometry;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (attempt_ || next_==std::numeric_limits<std::uint64_t>::max()) {
      error="Geometry Core residency insertion has no current schedule"; return {};
    }
    if (records_.size()==std::numeric_limits<std::size_t>::max()
        || (caller_ && !caller_->charge(records_.size()+1))) return {};
    prune();
    std::size_t owned=0,total=0;
    if (!checkedAdd(bytes,tokenBytes,owned)
        || !checkedAdd(peak,tokenBytes+recordBytes+2*sizeof(videorender::ParticleBodyCallerLease),total)) return {};
    std::shared_ptr<const videorender::ParticleBodyCallerLease> recordLease,tokenLease;
    if (caller_) {
      const auto usage=caller_->usage().logicalBytes;
      if (!checkedAdd(total,usage,total) || total>limit) {
        caller_->charge(peak); error="Geometry Core live owners exceed the prospective aggregate"; return {};
      }
      recordLease=caller_->reserve(recordBytes,recordBytes);
      tokenLease=caller_->reserve(owned,peak);
      if (!recordLease || !tokenLease) { error="Geometry Core residency caller admission is short"; return {}; }
    }
    auto token=std::make_shared<Token>(owned); token->lease=std::move(tokenLease);
    records_.emplace(++next_,Record{std::move(recordLease),token});
    token->residency=shared_from_this(); token->identity=next_;
    return token;
  }
  bool prepare(std::size_t peak,std::size_t work,std::size_t limit,std::string& error) {
    using namespace videowire::geometry;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!caller_) return true;
    std::size_t total=0;
    if (!caller_->charge(work) || attempt_
        || !checkedAdd(peak,tokenBytes+recordBytes+2*sizeof(videorender::ParticleBodyCallerLease),total)
        || !checkedAdd(total,caller_->usage().logicalBytes,total) || total>limit) {
      error="Geometry Core family preparation exceeds its live caller schedule"; return false;
    }
    return true;
  }
  template <typename T> static std::shared_ptr<T> alias(const std::shared_ptr<Token>& token,std::shared_ptr<T> value) {
    auto payload=std::make_shared<const Alias<T>>(Alias<T>{token,std::move(value)});
    return std::shared_ptr<T>(payload,payload->value.get());
  }
public:
  static constexpr std::size_t initialBytes() noexcept {
    return sizeof(RuntimeResidency)+2*controlBytes+sizeof(videorender::ParticleBodyCallerLease);
  }
  bool admitNativeHeader(std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!caller_ || caller_->charge(8)) return true;
    error="Geometry Core issued ownership header work is short"; return false;
  }
  bool admitNativeAttempt(std::string& error) { return mutationAllowed(error); }
  std::shared_ptr<const void> retainNative(std::size_t bytes,std::size_t limit,std::string& error) {
    return registerOwner(bytes,bytes,limit,error);
  }
  // The guard retains the issued census throughout callback and publication.
  class Attempt final {
    const std::shared_ptr<RuntimeResidency> residency_;
    bool active_=false;
  public:
    explicit Attempt(std::shared_ptr<RuntimeResidency> residency) : residency_(std::move(residency)) {
      if (!residency_) return;
      std::lock_guard<std::recursive_mutex> lock(residency_->mutex_);
      if (!residency_->caller_ || residency_->records_.size()==std::numeric_limits<std::size_t>::max()
          || !residency_->caller_->charge(residency_->records_.size()+1)
          || residency_->attempt_) return;
      residency_->prune(); residency_->attempt_=true; active_=true;
    }
    Attempt(const Attempt&)=delete;
    ~Attempt() {
      if (active_) { std::lock_guard<std::recursive_mutex> lock(residency_->mutex_); residency_->attempt_=false; }
    }
    bool active() const noexcept { return active_; }
  };
};

inline bool runtimeContextCopyBytes(const videowire::geometry::AdmissionContext& context,
    std::size_t& bytes) noexcept {
  using namespace videowire::geometry;
  bytes=sizeof(AdmissionContext);
  return addArrayBytes(context.admittedGeometryOrSceneSources.size(),sizeof(StableId)+4*sizeof(void*),bytes)
      && addArrayBytes(context.deferredPoints.size(),sizeof(decltype(context.deferredPoints)::value_type)+4*sizeof(void*),bytes);
}

struct RuntimeAdmission final {
private:
  // Only the issued copy is counted. Arbitrary external value copies are not
  // runtime allocations. Declare retention first so its fields retire first.
  std::shared_ptr<const void> issuedCopy_;
public:
  PlanUse use = PlanUse::preview;
  PlanOwnerIdentity owner;
  std::shared_ptr<const AdmittedPlanValue> plan;
  std::shared_ptr<const std::uint8_t> ownerLease;
  struct NativeResources final {
    std::mutex mutex;
    std::shared_ptr<const void> value;
  };
  std::shared_ptr<NativeResources> nativeResources;
  std::string surfaceMaterial;
  videowire::geometry::AdmissionContext sourceContext;
  std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission> bodySource;
  std::shared_ptr<const videorender::ParticleBodyCallerLease> bodyAdmissionLease;
  const std::shared_ptr<RuntimeResidency>& residency() const noexcept { return residency_; }
  bool matchesRuntimeResidency() const noexcept {
    return !residency_ || (owner==issuedOwner_ && plan.get()==issuedPlan_
        && ownerLease.get()==issuedLease_ && nativeResources.get()==issuedResources_
        && bodySource.get()==issuedSource_ && bodyAdmissionLease.get()==issuedBodyLease_);
  }
private:
  friend class GeometryCorePlanRuntime;
  std::shared_ptr<RuntimeResidency> residency_;
  PlanOwnerIdentity issuedOwner_;
  const AdmittedPlanValue* issuedPlan_=nullptr;
  const std::uint8_t* issuedLease_=nullptr;
  const NativeResources* issuedResources_=nullptr;
  const videorender::ParticlePointsBodySourceAdmission* issuedSource_=nullptr;
  const videorender::ParticleBodyCallerLease* issuedBodyLease_=nullptr;
  void issueResidency(const std::shared_ptr<RuntimeResidency>& residency,std::shared_ptr<const void> copy) noexcept {
    issuedCopy_=std::move(copy); residency_=residency; issuedOwner_=owner; issuedPlan_=plan.get(); issuedLease_=ownerLease.get();
    issuedResources_=nativeResources.get(); issuedSource_=bodySource.get(); issuedBodyLease_=bodyAdmissionLease.get();
  }
public:
  RuntimeAdmission()=default;
  RuntimeAdmission(const RuntimeAdmission&)=default;
  RuntimeAdmission(RuntimeAdmission&&)=default;
  RuntimeAdmission& operator=(const RuntimeAdmission&)=default;
  RuntimeAdmission& operator=(RuntimeAdmission&&)=default;
private:
  RuntimeAdmission(PlanUse useValue,PlanOwnerIdentity ownerValue,std::shared_ptr<const AdmittedPlanValue> planValue,
      std::shared_ptr<const std::uint8_t> ownerValueLease,std::shared_ptr<NativeResources> resources,
      std::string surface,videowire::geometry::AdmissionContext context,
      std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission> source,
      std::shared_ptr<const videorender::ParticleBodyCallerLease> lease)
      : use(useValue),owner(ownerValue),plan(std::move(planValue)),ownerLease(std::move(ownerValueLease)),
        nativeResources(std::move(resources)),surfaceMaterial(std::move(surface)),sourceContext(std::move(context)),
        bodySource(std::move(source)),bodyAdmissionLease(std::move(lease)) {}

};

class TimelineSceneChildAdmission final {
public:
  TimelineSceneChildAdmission(const TimelineSceneChildAdmission&) = default;
  TimelineSceneChildAdmission(TimelineSceneChildAdmission&&) noexcept = default;
  TimelineSceneChildAdmission& operator=(const TimelineSceneChildAdmission&) = delete;
  TimelineSceneChildAdmission& operator=(TimelineSceneChildAdmission&&) = delete;
  const PlanOwnerIdentity& owner() const noexcept { return owner_; }
  const HarmonicMIDI::grid::SceneObjectRecord& object() const noexcept { return object_; }
  const videowire::geometry::TimelineSceneBinding& binding() const noexcept { return binding_; }
  const AdmittedPlanValue& plan() const noexcept { return *plan_; }
  const HarmonicMIDI::grid::Visual3DScene& scene() const noexcept { return *scene_; }
  RuntimeAdmission::NativeResources& nativeResources() const noexcept { return *nativeResources_; }
private:
  friend class GeometryCorePlanRuntime;
  friend class AdmittedTimelineScenePlans;
  TimelineSceneChildAdmission(PlanOwnerIdentity owner,HarmonicMIDI::grid::SceneObjectRecord object,
      videowire::geometry::TimelineSceneBinding binding,std::shared_ptr<const AdmittedPlanValue> plan,
      std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene,std::shared_ptr<const void> residency)
      : residency_(std::move(residency)),owner_(owner),object_(object),binding_(std::move(binding)),plan_(std::move(plan)),
        scene_(std::move(scene)),lease_(std::make_shared<const std::uint8_t>(0)),
        nativeResources_(std::make_shared<RuntimeAdmission::NativeResources>()) {}
  const std::shared_ptr<const void> residency_;
  const PlanOwnerIdentity owner_;
  const HarmonicMIDI::grid::SceneObjectRecord object_;
  videowire::geometry::TimelineSceneBinding binding_;
  const std::shared_ptr<const AdmittedPlanValue> plan_;
  const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene_;
  std::shared_ptr<const std::uint8_t> lease_;
  std::shared_ptr<RuntimeAdmission::NativeResources> nativeResources_;
};

class AdmittedTimelineScenePlans final {
public:
  AdmittedTimelineScenePlans(const AdmittedTimelineScenePlans&) = delete;
  AdmittedTimelineScenePlans& operator=(const AdmittedTimelineScenePlans&) = delete;
  const PlanOwnerIdentity& owner() const noexcept { return owner_; }
  const HarmonicMIDI::grid::Visual3DScene& scene() const noexcept { return *scene_; }
  const std::vector<TimelineSceneChildAdmission>& children() const noexcept { return children_; }
  const timelinefamily::Envelope& envelope() const noexcept { return envelope_; }
  const videowire::geometry::BudgetReceipt& aggregateReceipt() const noexcept { return aggregate_; }
private:
  friend class GeometryCorePlanRuntime;
  AdmittedTimelineScenePlans(PlanOwnerIdentity owner,
      std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene,
      std::vector<TimelineSceneChildAdmission> children,timelinefamily::Envelope envelope,
      videowire::geometry::BudgetReceipt aggregate)
      : owner_(owner),scene_(std::move(scene)),children_(std::move(children)),
        envelope_(envelope),aggregate_(aggregate) {}
  bool childrenUnused() const noexcept {
    return std::all_of(children_.begin(),children_.end(),[](const auto& child) {
      return child.lease_.use_count()==1;
    });
  }
  const PlanOwnerIdentity owner_;
  const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene_;
  const std::vector<TimelineSceneChildAdmission> children_;
  const timelinefamily::Envelope envelope_;
  const videowire::geometry::BudgetReceipt aggregate_;
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
    if (!residency_->mutationAllowed(error)) return false;
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
      if (!matches && entryUnused(it->second))
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
    if (!residency_->mutationAllowed(error,payload.size())) return std::nullopt;
    auto bytes = videowire::geometry::decodeLoweredPlanText(payload, error);
    return bytes ? admitPreview(*bytes, limits, context, error) : std::nullopt;
  }
  std::optional<RuntimeAdmission>
  admitExportText(const std::string &payload,
                  const videowire::geometry::ResourceLimits &limits,
                  const videowire::geometry::AdmissionContext &context,
                  std::string &error) {
    if (!residency_->mutationAllowed(error,payload.size())) return std::nullopt;
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

  std::optional<RuntimeAdmission> admitPointsBody(const PlanOwnerIdentity&,
      const std::vector<std::uint8_t>&,const videowire::geometry::ResourceLimits&,
      const std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission>&,std::string&);

  std::shared_ptr<const AdmittedTimelineScenePlans> admitTimelineScenePreview(
      const PlanOwnerIdentity& owner,const HarmonicMIDI::grid::Visual3DScene& scene,
      const std::vector<videowire::geometry::TimelineSceneBinding>& bindings,
      const videowire::geometry::ResourceLimits& limits,std::string& error) {
    return admitTimelineScene(owner,scene,bindings,limits,PlanUse::preview,error);
  }
  std::shared_ptr<const AdmittedTimelineScenePlans> admitTimelineSceneExport(
      const PlanOwnerIdentity& owner,const HarmonicMIDI::grid::Visual3DScene& scene,
      const std::vector<videowire::geometry::TimelineSceneBinding>& bindings,
      const videowire::geometry::ResourceLimits& limits,std::string& error) {
    return admitTimelineScene(owner,scene,bindings,limits,PlanUse::exportRender,error);
  }

private:
  std::optional<RuntimeAdmission>
  admitLowered(const std::vector<std::uint8_t> &bytes,
               const videowire::geometry::ResourceLimits &limits,
               const videowire::geometry::AdmissionContext &context,
               PlanUse use, std::string &error) {
    if (!residency_->mutationAllowed(error,bytes.size())) return std::nullopt;
    auto lowered = videowire::geometry::decodeLoweredRuntimePlan(bytes, error);
    if (!lowered)
      return std::nullopt;
    return admit(lowered->runtimeValue, lowered->contract, limits, context,
                 PlanOwnerIdentity{1, 1, 1, 1, 1, use}, error, lowered->surfaceMaterial);
  }
  std::optional<RuntimeAdmission>
  admitLowered(const std::vector<std::uint8_t> &bytes,
               const videowire::geometry::ResourceLimits &limits,
               const videowire::geometry::AdmissionContext &context,
               const PlanOwnerIdentity &owner, std::string &error,
               const std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission>& bodySource = {},
               const std::shared_ptr<const videorender::ParticleBodyCallerLease>& bodyLease = {}) {
    if (!owner.valid()) {
      error = "Geometry Core plan owner identity is incomplete";
      return std::nullopt;
    }
    if (!residency_->mutationAllowed(error,bytes.size())) return std::nullopt;
    auto lowered = videowire::geometry::decodeLoweredRuntimePlan(bytes, error);
    if (!lowered)
      return std::nullopt;
    return admit(lowered->runtimeValue, lowered->contract, limits, context,
                 owner, error, lowered->surfaceMaterial, bodySource, bodyLease);
  }
  std::optional<RuntimeAdmission>
  admit(const std::vector<std::uint8_t> &bytes,
        const videowire::geometry::PortContract &contract,
        const videowire::geometry::ResourceLimits &limits,
        const videowire::geometry::AdmissionContext &context,
        const PlanOwnerIdentity &owner,
        std::string &error, const std::string& surfaceMaterial = {},
        const std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission>& bodySource = {},
        const std::shared_ptr<const videorender::ParticleBodyCallerLease>& bodyLease = {}) {
    if (!context.evaluatedBodyGeometry.empty() || (!context.deferredBodyGeometry.empty() && (!bodySource || !bodyLease))) {
      error="Immutable body Geometry3D admission requires a private source and no frame samples"; return std::nullopt;
    }
    for (const auto& source:context.deferredBodyGeometry)
      if (source.second.structuralRevision!=owner.planGeneration) {
        error="Deferred body Geometry3D structural revision differs from its plan owner"; return std::nullopt;
      }
    if (!context.evaluatedPoints.empty()) {
      error = "Immutable Geometry Core admission cannot contain frame point samples"; return std::nullopt;
    }
    for (const auto& source : context.deferredPoints)
      if (source.second.sourceRevision != owner.planGeneration) {
        error="Deferred Points3D revision does not match its immutable plan owner"; return std::nullopt;
      }
    if (!surfaceMaterial.empty()) {
      const auto material = geometrysurfacematerial::decodeSet(surfaceMaterial, error);
      if (!material) return std::nullopt;
      if (material->front().material.structuralRevision != owner.planGeneration) {
        error = "Geometry Surface revision does not match its immutable plan owner";
        return std::nullopt;
      }
    }
    auto decoded = videowire::geometry::decodeRuntimeValue(
        bytes, contract, limits, context, error);
    if (!decoded)
      return std::nullopt;
    if (!surfaceMaterial.empty()) {
      const auto programs=geometrysurfacematerial::decodeSet(surfaceMaterial,error);
      const auto& operations=decoded->descriptor().operations;
      const bool hasTable=std::any_of(operations.begin(),operations.end(),[](const auto& op) {
        return op.code==videowire::geometry::OperationCode::materialTableSlot;
      });
      if (!programs) return std::nullopt;
      for (const auto& program : *programs) {
        const bool referenced=std::any_of(operations.begin(),operations.end(),[&](const auto& op) {
          return op.code==videowire::geometry::OperationCode::materialTableSlot && op.secondaryInputStableId==program.sourceMaterial;
        });
        if ((program.sourceMaterial==0 && hasTable) || (program.sourceMaterial!=0 && !referenced)) {
          error="Geometry Surface program has no exact material-table owner"; return std::nullopt;
        }
      }
    }
    auto plan = admitPlanValue(decoded->descriptor(), contract, limits, context,
                               backend_, error);
    if (!plan)
      return std::nullopt;
    std::uint64_t key = plan->receipt().backendBinding;
    for (const auto& source : context.deferredPoints) {
      for (const auto part : {source.first,source.second.sourceStableId,source.second.sourceRevision,
                             static_cast<std::uint64_t>(source.second.capacity)}) {
        key ^= part; key *= 1099511628211ull;
      }
    }
    for (const auto byte : bytes) {
      key ^= byte;
      key *= 1099511628211ull;
    }
    for (const unsigned char byte : surfaceMaterial) {
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
      if (found->second.family || found->second.planDigest != key || found->second.bodySource != bodySource) {
        error = "Geometry Core owner was already bound to a different immutable plan";
        return std::nullopt;
      }
      auto copy=bodySource ? std::shared_ptr<const void>{}:admitRuntimeCopy(context,surfaceMaterial,
          found->second.plan->receipt().effectiveLimits.maxBufferBytes,error);
      if (!bodySource && !copy) return std::nullopt;
      found->second.lastUse = ++clock_;
      RuntimeAdmission result{owner.use, owner, found->second.plan,
                              found->second.ownerLease,
                              found->second.nativeResources, found->second.surfaceMaterial, context, found->second.bodySource, bodyLease};
      result.issueResidency(residency_,std::move(copy)); return result;
    }
    auto victim=cache_.end();
    if (cache_.size() >= maximumOwners_) {
      for (auto it=cache_.begin();it!=cache_.end();++it)
        if (entryUnused(it->second) && (victim==cache_.end() || it->second.lastUse<victim->second.lastUse)) victim=it;
      if (victim==cache_.end()) {
        error = "Geometry Core owner cache is full with live leases";
        return std::nullopt;
      }
    }
    std::shared_ptr<RuntimeResidency::Token> token;
    if (!bodySource) {
      std::size_t bytes=0;
      if (!legacyResidentBytes(*plan,surfaceMaterial,bytes)
          || !videowire::geometry::checkedAdd(bytes,sizeof(Entry)+4*sizeof(void*)+3*RuntimeResidency::controlBytes+3*RuntimeResidency::aliasBytes,bytes)) {
        error="Geometry Core resident owner storage overflow"; return std::nullopt;
      }
      token=residency_->registerOwner(bytes,bytes,plan->receipt().effectiveLimits.maxBufferBytes,error);
      if (!token) return std::nullopt;
    }
    auto shared = std::make_shared<const AdmittedPlanValue>(std::move(*plan));
    auto ownerLease = std::make_shared<const std::uint8_t>(0);
    auto nativeResources = std::make_shared<RuntimeAdmission::NativeResources>();
    if (token) {
      shared=RuntimeResidency::alias(token,std::move(shared));
      ownerLease=RuntimeResidency::alias(token,std::move(ownerLease));
      nativeResources=RuntimeResidency::alias(token,std::move(nativeResources));
    }
    auto copy=bodySource ? std::shared_ptr<const void>{}:admitRuntimeCopy(context,surfaceMaterial,
        shared->receipt().effectiveLimits.maxBufferBytes,error);
    if (!bodySource && !copy) return std::nullopt;
    cache_.emplace(owner, Entry{key, ++clock_, shared, ownerLease, nativeResources, surfaceMaterial, {}, bodySource, bodyLease});
    if (victim!=cache_.end()) cache_.erase(victim);
    ++admissionsInGeneration_;
    RuntimeAdmission result{owner.use, owner, std::move(shared),
                            std::move(ownerLease), std::move(nativeResources), surfaceMaterial, context, bodySource, bodyLease};
    result.issueResidency(residency_,std::move(copy)); return result;
  }
  std::shared_ptr<const void> admitRuntimeCopy(const videowire::geometry::AdmissionContext& context,
      const std::string& surface,std::size_t limit,std::string& error) {
    std::size_t bytes=0;
    if (!runtimeContextCopyBytes(context,bytes)
        || !videowire::geometry::checkedAdd(bytes,sizeof(RuntimeAdmission)+surface.capacity(),bytes)) {
      error="Runtime admission copy storage overflow"; return {};
    }
    return residency_->registerOwner(bytes,bytes,limit,error);
  }
  // Preserve the legacy typed envelope. The census adds its owning alias and
  // registration records once; native executions register their distinct copies.
  static bool legacyResidentBytes(const AdmittedPlanValue& plan,const std::string& surface,
      std::size_t& bytes) noexcept {
    using namespace videowire::geometry;
    const auto& value=plan.value().descriptor();
    if (!checkedMul(plan.receipt().value.bufferBytes,16,bytes)
        || !checkedAdd(bytes,sizeof(AdmittedPlanValue)+sizeof(RuntimeAdmission::NativeResources)+1,bytes)
        || !addArrayBytes(value.attributes.capacity(),sizeof(AttributeData)+kMaximumNameBytes+1,bytes)
        || !addArrayBytes(value.operations.capacity(),sizeof(OperationRecord)+sizeof(RetainedMeshData)+kMaximumNameBytes+1,bytes)
        || !checkedAdd(bytes,surface.capacity(),bytes)
        || !checkedAdd(bytes,plan.capabilities().backendIdentity.capacity(),bytes)
        || !checkedAdd(bytes,plan.capabilities().deviceIdentity.capacity(),bytes)) return false;
    return true;
  }
  std::shared_ptr<const AdmittedTimelineScenePlans> admitTimelineScene(
      const PlanOwnerIdentity& owner,const HarmonicMIDI::grid::Visual3DScene& scene,
      const std::vector<videowire::geometry::TimelineSceneBinding>& bindings,
      const videowire::geometry::ResourceLimits& requested,PlanUse use,std::string& error) {
    using namespace videowire::geometry;
    using Scene=HarmonicMIDI::grid::Visual3DScene;
    error.clear();
    if (!residency_->mutationAllowed(error)) return {};
    if (!owner.valid() || owner.use!=use) {
      error="Timeline scene family has an incomplete or mismatched owner use"; return {};
    }
    const auto capabilities=backend_.geometryCoreCapabilities();
    if (!validateLimits(requested,error) || !validateLimits(capabilities.limits,error)
        || capabilities.backendIdentity.empty() || !capabilities.nativeGpuAvailable
        || !capabilities.geometryCoreExecution || !capabilities.immutableSourceBuffers
        || !capabilities.stableElementIds || !capabilities.typedFieldEvaluation
        || !(capabilities.supportedCarriers & carrierBit(CarrierKind::geometry3D))) {
      error="Production backend does not admit a timeline scene family"; return {};
    }
    auto effective=requested;
#define ARBIT_TIMELINE_TIGHTEN(name) effective.name=std::min(effective.name,capabilities.limits.name)
    ARBIT_TIMELINE_TIGHTEN(maxVertices); ARBIT_TIMELINE_TIGHTEN(maxIndices);
    ARBIT_TIMELINE_TIGHTEN(maxPoints); ARBIT_TIMELINE_TIGHTEN(maxCurvePoints);
    ARBIT_TIMELINE_TIGHTEN(maxSplines); ARBIT_TIMELINE_TIGHTEN(maxInstances);
    ARBIT_TIMELINE_TIGHTEN(maxFieldElements); ARBIT_TIMELINE_TIGHTEN(maxAttributes);
    ARBIT_TIMELINE_TIGHTEN(maxOperations); ARBIT_TIMELINE_TIGHTEN(maxDispatches);
    ARBIT_TIMELINE_TIGHTEN(maxBufferBytes);
#undef ARBIT_TIMELINE_TIGHTEN
    effective.maxVertices=std::min(effective.maxVertices,Scene::kMaxVertices);
    effective.maxIndices=std::min(effective.maxIndices,Scene::kMaxIndices);
    effective.maxOperations=std::min<std::size_t>(effective.maxOperations,128);
    timelinefamily::Envelope envelope;
    std::size_t childBytes=sizeof(TimelineSceneChildAdmission)+sizeof(AdmittedPlanValue)
        +sizeof(RuntimeAdmission::NativeResources)+3;
    if (!checkedAdd(childBytes,capabilities.backendIdentity.capacity(),childBytes)
        || !checkedAdd(childBytes,capabilities.deviceIdentity.capacity(),childBytes)) {
      error="Timeline scene family backend identity storage overflow"; return {};
    }
    if (!timelinefamily::deriveEnvelope(scene,bindings,effective,childBytes,envelope,error)) return {};
    if (!checkedAdd(envelope.retainedBytes,sizeof(AdmittedTimelineScenePlans)+sizeof(Entry),envelope.retainedBytes)
        || !checkedAdd(envelope.attemptBytes,sizeof(AdmittedTimelineScenePlans)+sizeof(Entry),envelope.attemptBytes)) {
      error="Timeline scene family owner record storage overflow"; return {};
    }
    std::size_t residencyBytes=0;
    if (!checkedMul(bindings.size(),4*RuntimeResidency::aliasBytes,residencyBytes)
        || !addArrayBytes(bindings.size(),3*RuntimeResidency::controlBytes,residencyBytes)
        || !checkedAdd(residencyBytes,RuntimeResidency::aliasBytes+2*RuntimeResidency::controlBytes+4*sizeof(void*),residencyBytes)
        || !checkedAdd(envelope.retainedBytes,residencyBytes,envelope.retainedBytes)
        || !checkedAdd(envelope.attemptBytes,residencyBytes,envelope.attemptBytes)) {
      error="Timeline scene residency alias arithmetic overflow"; return {};
    }
    // Admit the complete checked scene-work schedule and logical storage before
    // the first validator, hierarchy lookup, exact comparison or range walk.
    // The API has no independent scalar-work limit; checked fixed capacities
    // bound the named visit schedules. Storage still shares the effective cap.
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t preflightResident=0,preflightAttempt=0;
    // Preflight/resident/victim scans plus map lookup/insertion, each visiting
    // at most all incumbent owners, and the new entry. These visits inspect
    // fixed records/capacities or six-field keys, never scene contents.
    if (!checkedMul(cache_.size(),5,envelope.residentOwnerVisits)
        || !checkedAdd(envelope.residentOwnerVisits,1,envelope.residentOwnerVisits)) {
      error="Timeline scene family resident work schedule overflow"; return {};
    }
    std::size_t familyWork=0;
    for (const auto work : {envelope.decodeByteVisits,envelope.rangeComparisons,envelope.replayIterations,
        envelope.admissionReplayIterations,envelope.planPairVisits,envelope.entryCountChecks,
        envelope.entryRecordVisits,envelope.residentOwnerVisits,envelope.sceneValidationCalls,
        envelope.hierarchyLookupVisits,envelope.hierarchyDepthIterations,envelope.matrixConstructions,
        envelope.matrixMultiplyTerms,envelope.matrixElementVisits,envelope.matrixPostcheckIterations,
        envelope.lightChannelVisits,envelope.sceneRecordVisits,envelope.sceneStringByteVisits,
        envelope.fieldElementVisits,envelope.normalIndexVisits,envelope.waterSampleElements,envelope.sceneCopyElements})
      if (!checkedAdd(familyWork,work,familyWork)) {
        error="Timeline scene caller work arithmetic overflow"; return {};
      }
    if (!residency_->prepare(envelope.attemptBytes,familyWork,effective.maxBufferBytes,error)) return {};
    for (const auto& entry : cache_) {
      std::size_t bytes=0;
      if (entry.second.family) bytes=entry.second.family->envelope().retainedBytes;
      else if (!legacyResidentBytes(*entry.second.plan,entry.second.surfaceMaterial,bytes)) {
        error="Legacy Geometry Core resident storage overflow"; return {};
      }
      if (!checkedAdd(bytes,sizeof(Entry),bytes) || !checkedAdd(preflightResident,bytes,preflightResident)) {
        error="Timeline scene family resident logical storage overflow"; return {};
      }
    }
    // Include every incumbent and escaped generation, even an eviction victim.
    if (!checkedAdd(preflightResident,envelope.attemptBytes,preflightAttempt)
        || preflightAttempt>effective.maxBufferBytes) {
      error="Timeline scene family and retained owners exceed the attempt budget"; return {};
    }
    if (!HarmonicMIDI::grid::validateVisual3DScene(scene).valid()) {
      error="Timeline scene family requires a valid scene and 1..64 exact bindings"; return {};
    }
    // Canonical input order avoids ambiguous object-to-plan tables. Validate
    // all ranges before decoding the first child or copying any authored data.
    for (std::size_t i=0;i<bindings.size();++i) {
      if (i && bindings[i-1].object.value>=bindings[i].object.value) {
        error="Timeline scene family bindings are duplicated or noncanonical"; return {};
      }
      const auto* object=HarmonicMIDI::grid::visual3d_detail::findById(scene.objects,scene.objectCount,bindings[i].object);
      if (!object || !object->vertexCount || !object->indexCount) {
        error="Timeline scene family binding has no exact mesh object"; return {};
      }
      for (std::size_t j=0;j<scene.objectCount;++j) {
        const auto& other=scene.objects[j];
        const auto overlaps=[](std::size_t a,std::size_t n,std::size_t b,std::size_t m) {
          return n && m && a<b+m && b<a+n;
        };
        if (other.id!=object->id && (overlaps(object->firstVertex,object->vertexCount,other.firstVertex,other.vertexCount)
            || overlaps(object->firstIndex,object->indexCount,other.firstIndex,other.indexCount))) {
          error="Timeline scene family mesh ranges are not exclusive"; return {};
        }
      }
    }
    if (activeProjectGeneration_ && (owner.projectGeneration!=activeProjectGeneration_
        || owner.helperGeneration!=activeHelperGeneration_ || owner.deviceGeneration!=activeDeviceGeneration_)) {
      error="Timeline scene family owner does not match the active generation"; return {};
    }
    const auto found=cache_.find(owner);
    if (found!=cache_.end()) {
      const auto& family=found->second.family;
      if (!family || !timelinefamily::sameScene(family->scene(),scene) || family->children().size()!=bindings.size()) {
        error="Geometry Core owner was already bound to a different immutable scene family"; return {};
      }
      for (std::size_t i=0;i<bindings.size();++i)
        if (family->children()[i].binding().object!=bindings[i].object
            || family->children()[i].binding().loweredPlan!=bindings[i].loweredPlan) {
          error="Timeline scene family owner was already bound to a different child plan"; return {};
        }
      for (const auto& child : family->children())
        if (child.plan().capabilities().backendIdentity!=capabilities.backendIdentity
            || child.plan().capabilities().deviceIdentity!=capabilities.deviceIdentity) {
          error="Timeline scene family backend identity changed without a new owner"; return {};
        }
      std::size_t resident=0;
      for (const auto& entry : cache_) {
        std::size_t bytes=0;
        if (entry.second.family) bytes=entry.second.family->envelope().retainedBytes;
        else if (!legacyResidentBytes(*entry.second.plan,entry.second.surfaceMaterial,bytes)) {
          error="Legacy Geometry Core resident storage overflow"; return {};
        }
        if (!checkedAdd(bytes,sizeof(Entry),bytes) || !checkedAdd(resident,bytes,resident)) {
          error="Cached timeline scene family resident storage overflow"; return {};
        }
      }
      if (!checkedAdd(resident,family->envelope().attemptBytes,resident) || resident>effective.maxBufferBytes) {
        error="Cached timeline scene family exceeds the retained/attempt budget"; return {};
      }
      const auto& originalLimits=family->children().front().plan().receipt().effectiveLimits;
#define ARBIT_TIMELINE_SAME_LIMIT(name) originalLimits.name==effective.name
      if (!(ARBIT_TIMELINE_SAME_LIMIT(maxVertices) && ARBIT_TIMELINE_SAME_LIMIT(maxIndices)
          && ARBIT_TIMELINE_SAME_LIMIT(maxPoints) && ARBIT_TIMELINE_SAME_LIMIT(maxCurvePoints)
          && ARBIT_TIMELINE_SAME_LIMIT(maxSplines) && ARBIT_TIMELINE_SAME_LIMIT(maxInstances)
          && ARBIT_TIMELINE_SAME_LIMIT(maxFieldElements) && ARBIT_TIMELINE_SAME_LIMIT(maxAttributes)
          && ARBIT_TIMELINE_SAME_LIMIT(maxOperations) && ARBIT_TIMELINE_SAME_LIMIT(maxDispatches)
          && ARBIT_TIMELINE_SAME_LIMIT(maxBufferBytes))) {
        error="Timeline scene family budget binding changed without a new owner"; return {};
      }
#undef ARBIT_TIMELINE_SAME_LIMIT
      const auto& receipt=family->aggregateReceipt();
      if (receipt.vertices>effective.maxVertices || receipt.indices>effective.maxIndices
          || receipt.fieldElements>effective.maxFieldElements || receipt.attributes>effective.maxAttributes
          || receipt.operations>effective.maxOperations || receipt.dispatches>effective.maxDispatches
          || receipt.bufferBytes>effective.maxBufferBytes) {
        error="Cached timeline scene family exceeds the requested aggregate budget"; return {};
      }
      found->second.lastUse=++clock_; return family;
    }
    auto victim=cache_.end();
    if (cache_.size()>=maximumOwners_) {
      for (auto it=cache_.begin();it!=cache_.end();++it)
        if (entryUnused(it->second) && (victim==cache_.end() || it->second.lastUse<victim->second.lastUse)) victim=it;
      if (victim==cache_.end()) {
        error="Geometry Core owner cache is full with live leases"; return {};
      }
    }
    std::size_t resident=0,attempt=0;
    for (const auto& entry : cache_) {
      std::size_t bytes=0;
      if (entry.second.family) bytes=entry.second.family->envelope().retainedBytes;
      else if (!legacyResidentBytes(*entry.second.plan,entry.second.surfaceMaterial,bytes)) {
        error="Legacy Geometry Core resident storage overflow"; return {};
      }
      if (!checkedAdd(bytes,sizeof(Entry),bytes) || !checkedAdd(resident,bytes,resident)) {
        error="Timeline scene family resident logical storage overflow"; return {};
      }
    }
    // Include the incumbent even when it is eligible for eviction. Publication
    // happens only after every child and the complete attempt have succeeded.
    if (!checkedAdd(resident,envelope.attemptBytes,attempt) || attempt>effective.maxBufferBytes) {
      error="Timeline scene family and retained owners exceed the attempt budget"; return {};
    }
    auto token=residency_->registerOwner(envelope.retainedBytes,envelope.attemptBytes,effective.maxBufferBytes,error);
    if (!token) return {};
    try {
      auto frozen=RuntimeResidency::alias(token,std::make_shared<const Scene>(scene));
      std::vector<TimelineSceneChildAdmission> children;
      children.reserve(bindings.size());
      BudgetReceipt aggregate;
      auto remaining=effective;
      for (const auto& binding : bindings) {
        if (!validateLimits(remaining,error)) {
          error="Timeline scene family exhausted its remaining aggregate budget before a sibling"; return {};
        }
        std::size_t retainedFields=0;
        const auto value=decodeTimelineScenePlan(binding,remaining,error,&retainedFields);
        if (!value || !validTimelineSceneBinding(binding,*frozen,*value,error)) {
          if (error.empty()) error="Timeline scene family child does not own the frozen object mesh";
          return {};
        }
        const auto bytes=decodeLoweredPlanText(binding.loweredPlan,error);
        const auto lowered=bytes ? decodeLoweredRuntimePlan(*bytes,error) : std::nullopt;
        auto admitted=lowered ? admitPlanValue(value->descriptor(),lowered->contract,remaining,{},backend_,error) : std::nullopt;
        if (!admitted) return {};
        const auto& receipt=admitted->receipt().value;
#define ARBIT_TIMELINE_CONSUME(count,limit) \
        if (receipt.count>remaining.limit || !checkedAdd(aggregate.count,receipt.count,aggregate.count)) { \
          error="Timeline scene family exceeds aggregate Geometry Core receipts"; return {}; \
        } \
        remaining.limit-=receipt.count
        ARBIT_TIMELINE_CONSUME(vertices,maxVertices); ARBIT_TIMELINE_CONSUME(indices,maxIndices);
        ARBIT_TIMELINE_CONSUME(points,maxPoints); ARBIT_TIMELINE_CONSUME(curvePoints,maxCurvePoints);
        ARBIT_TIMELINE_CONSUME(splines,maxSplines); ARBIT_TIMELINE_CONSUME(instances,maxInstances);
        if (retainedFields>remaining.maxFieldElements
            || !checkedAdd(aggregate.fieldElements,retainedFields,aggregate.fieldElements)) {
          error="Timeline scene family exceeds aggregate retained field elements"; return {};
        }
        remaining.maxFieldElements-=retainedFields;
        ARBIT_TIMELINE_CONSUME(attributes,maxAttributes);
        ARBIT_TIMELINE_CONSUME(operations,maxOperations); ARBIT_TIMELINE_CONSUME(dispatches,maxDispatches);
        ARBIT_TIMELINE_CONSUME(bufferBytes,maxBufferBytes);
#undef ARBIT_TIMELINE_CONSUME
        const auto* object=HarmonicMIDI::grid::visual3d_detail::findById(frozen->objects,frozen->objectCount,binding.object);
        children.push_back(TimelineSceneChildAdmission(owner,*object,binding,
            RuntimeResidency::alias(token,std::make_shared<const AdmittedPlanValue>(*admitted)),frozen,token));
        auto& child=children.back();
        // The public child copy and every owning field retain one registration.
        child.lease_=RuntimeResidency::alias(token,std::move(child.lease_));
        child.nativeResources_=RuntimeResidency::alias(token,std::move(child.nativeResources_));
      }
      std::shared_ptr<const AdmittedTimelineScenePlans> family(new AdmittedTimelineScenePlans(
          owner,std::move(frozen),std::move(children),envelope,aggregate));
      family=RuntimeResidency::alias(token,std::move(family));
      // Map insertion is the last potentially allocating step. Keep the old
      // entry until insertion succeeds, so refusal cannot erase the incumbent.
      cache_.emplace(owner,Entry{0,clock_+1,{},{},{},{},family});
      if (victim!=cache_.end()) cache_.erase(victim);
      ++clock_; ++admissionsInGeneration_;
      return family;
    } catch (const std::bad_alloc&) {
      // Existing admitted-value noexcept moves retain their legacy exception
      // semantics. This handles propagated failures, not universal allocator rollback.
      error="Timeline scene family allocation failed before atomic publication"; return {};
    }
  }
  struct Entry final {
    std::uint64_t planDigest = 0;
    std::uint64_t lastUse = 0;
    std::shared_ptr<const AdmittedPlanValue> plan;
    std::shared_ptr<const std::uint8_t> ownerLease;
    std::shared_ptr<RuntimeAdmission::NativeResources> nativeResources;
    std::string surfaceMaterial;
    std::shared_ptr<const AdmittedTimelineScenePlans> family {};
    std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission> bodySource {};
    std::shared_ptr<const videorender::ParticleBodyCallerLease> bodyLease {};
  };
  static bool entryUnused(const Entry& entry) noexcept {
    if (entry.family) return entry.family.use_count()==1 && entry.family->childrenUnused();
    // The cache's own alias retains the native owner. It is not an escaped
    // consumer; two aliases retained by callers still keep eviction refused.
    if (entry.ownerLease.use_count()==1) return true;
    if (!entry.nativeResources) return false;
    std::lock_guard<std::mutex> lock(entry.nativeResources->mutex);
    return entry.ownerLease.use_count()==2 && entry.nativeResources->value.use_count()==1;
  }
  const std::shared_ptr<RuntimeResidency> residency_{new RuntimeResidency};
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
