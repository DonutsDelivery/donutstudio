#include "geometry_core_backend.h"
#include "../../shared/GeometryAudioDeformer.h"
#include "../../shared/GeometryCoreScene.h"
#include "../../shared/GeometryMaterialTable.h"
#include "geometry_score_field.h"
#include "fixture_scene_renderer.h"
#include "water_surface_replay.h"
#include "particle_points_body_frame.h"

#include <cmath>

namespace videohelper::geometry {
namespace {
using namespace videowire::geometry;
// The first paired route has exactly one retained source, one deferred Points
// copy and one InstanceOnPoints. Twelve complete payload passes cover immutable
// decode/admission, both ordered replay/admission passes, materialization and
// publication copies. The native scene and its frozen backend copy are separate.
}
bool particlePointsBodyCoreEnvelope(const DeferredBodyGeometrySource& source,std::size_t wireBytes,
    bool frame,videorender::ParticlePointsBodyEnvelope& envelope) {
  if (!source.reference || source.reference->positions.size()>256 || source.reference->indices.size()>384
      || source.points.capacity==0 || source.points.capacity>64 || wireBytes>65536) return false;
  std::size_t mesh=sizeof(GeometryData), points=sizeof(ValueDescriptor), instances=sizeof(ValueDescriptor);
  if (!addArrayBytes(source.reference->positions.size(),sizeof(Vec3)+sizeof(StableId),mesh)
      || !addArrayBytes(source.reference->indices.size(),sizeof(std::uint32_t),mesh)
      || !addArrayBytes(source.points.capacity,sizeof(PointRecord)+3*sizeof(FieldElement),points)
      || !addArrayBytes(3,sizeof(AttributeData)+kMaximumNameBytes+1,points)
      || !addArrayBytes(source.points.capacity,sizeof(InstanceRecord)+3*sizeof(FieldElement)+sizeof(StableId),instances)
      || !addArrayBytes(3,sizeof(AttributeData)+kMaximumNameBytes+1,instances)) return false;
  std::size_t pass=sizeof(ValueDescriptor)+3*sizeof(OperationRecord)+sizeof(RetainedMeshData)
      +sizeof(MaterializedInstancePlan)+sizeof(AdmissionContext);
  if (!checkedAdd(pass,mesh,pass) || !checkedAdd(pass,points,pass) || !checkedAdd(pass,instances,pass)
      || !checkedMul(pass,12,envelope.bytes) || !addArrayBytes(wireBytes,8,envelope.bytes)
      || !checkedAdd(envelope.bytes,3*(sizeof(DeferredBodyGeometrySource)+sizeof(BodyGeometryFrameData)
          +sizeof(ValueDescriptor)+sizeof(StableId)+4*sizeof(void*)),envelope.bytes)
      || !checkedAdd(envelope.bytes,sizeof(RuntimeAdmission)+sizeof(AdmittedPlanValue)
          +sizeof(RuntimeAdmission::NativeResources)+sizeof(NativeGeometryExecution)+1,envelope.bytes)) return false;
  if (!frame && !checkedAdd(envelope.bytes,RuntimeResidency::initialBytes(),envelope.bytes)) return false;
  if (frame && !checkedAdd(envelope.bytes,2*sizeof(HarmonicMIDI::grid::Visual3DScene)
      +sizeof(arbitgpu::NativeFixtureSceneFrame)+sizeof(arbitgpu::NativeFixtureSceneResources)
      +7*sizeof(std::shared_ptr<const void>)+sizeof(std::optional<videorender::ParticleBodyCallerCeiling>)
      +sizeof(std::optional<RuntimeResidency::Attempt>),envelope.bytes)) return false;
  // Byte visits cover payload copies and all fixed record checks. Each replay
  // also reserves every identity-set comparison and topology/finite pass.
  envelope.work=envelope.bytes;
  return checkedAdd(envelope.work,12*(64*64+256*16+384*8+3*32),envelope.work);
}
namespace {
bool sameBodyDeclaration(const DeferredBodyGeometrySource& a,const DeferredBodyGeometrySource& b) {
  return a.outputStableId==b.outputStableId && a.pointsOutputStableId==b.pointsOutputStableId
      && a.points==b.points && a.terminalStableId==b.terminalStableId && a.sourceStableId==b.sourceStableId
      && a.sourceRevision==b.sourceRevision && a.structuralRevision==b.structuralRevision
      && a.source==b.source && a.reference==b.reference && a.maxVertices==b.maxVertices && a.maxIndices==b.maxIndices;
}
struct PointsBodyNativePayload final {
  const std::shared_ptr<const videorender::ParticleBodyCallerLease> lease;
  const std::shared_ptr<const videorender::ParticlePointsBodyFrame> body;
  const std::shared_ptr<const std::uint8_t> owner;
  const std::shared_ptr<const AdmittedPlanValue> plan;
  const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
  const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
  const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> resources;
};
struct OrdinaryNativePayload final {
  const std::shared_ptr<const void> residency;
  const std::shared_ptr<const std::uint8_t> owner;
  const std::shared_ptr<const AdmittedPlanValue> plan;
  const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
  const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
  const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> resources;
};
static_assert(sizeof(PointsBodyNativePayload)==7*sizeof(std::shared_ptr<const void>));
}

std::optional<RuntimeAdmission> GeometryCorePlanRuntime::admitPointsBody(const PlanOwnerIdentity& owner,
    const std::vector<std::uint8_t>& bytes,const ResourceLimits& limits,
    const std::shared_ptr<const videorender::ParticlePointsBodySourceAdmission>& source,std::string& error) {
  error.clear();
  try {
  if (source && !source->caller()->charge(videorender::particlePointsBodyCoreOwnerVisits)) {
    error="Paired Geometry Core owner comparison work is short"; return std::nullopt;
  }
  if (!source || !owner.valid() || owner.planGeneration!=source->declaration().structuralRevision) {
    error="Paired Geometry Core admission requires its private source and exact structural owner"; return std::nullopt;
  }
  const auto coreLimit=std::min(limits.maxBufferBytes,backend_.geometryCoreCapabilities().limits.maxBufferBytes);
  videorender::ParticlePointsBodyEnvelope envelope;
  if (!particlePointsBodyCoreEnvelope(source->declaration(),bytes.size(),false,envelope)
      || envelope.bytes>coreLimit) {
    error="Paired Geometry Core source admission exceeds its prospective Core budget"; return std::nullopt;
  }
  if (!residency_->mutationAllowed(error)) return std::nullopt;
  std::size_t pending=0,attempt=0;
  if (!residency_->pending(source->caller(),pending,error)) return std::nullopt;
  // The static source recipe includes first-use census metadata. Live owner
  // registrations are leased separately, once, rather than snapshot-copied.
  envelope.bytes-=RuntimeResidency::initialBytes();
  const auto resident=source->caller()->usage().logicalBytes;
  if (!checkedAdd(resident,pending,attempt) || !checkedAdd(attempt,envelope.bytes,attempt)
      || !checkedAdd(attempt,sizeof(videorender::ParticleBodyCallerLease),attempt) || attempt>coreLimit) {
    source->caller()->charge(envelope.work);
    error="Paired Geometry Core source plus live/escaped storage exceeds its Core budget"; return std::nullopt;
  }
  videorender::ParticleBodyCallerCeiling ceiling(source->caller(),coreLimit);
  if (!residency_->attach(source->caller())) {
    error="Paired Geometry Core live census caller admission is short"; return std::nullopt;
  }
  auto lease=source->caller()->reserve(envelope.bytes,envelope.work);
  if (!lease) { error="Paired Geometry Core source prospective caller admission is short"; return std::nullopt; }
  const auto& declaration=source->declaration();
  AdmissionContext context;
  context.deferredPoints.emplace(declaration.pointsOutputStableId,declaration.points);
  context.deferredBodyGeometry.emplace(declaration.outputStableId,declaration);
  const auto lowered=decodeLoweredRuntimePlan(bytes,error);
  if (!lowered || !lowered->surfaceMaterial.empty() || !lowered->contract.attributes.empty()
      || lowered->contract.carrier!=CarrierKind::instances3D || lowered->runtimeValue.size()<21) {
    if (error.empty()) {error="Paired Geometry Core requires its bounded instances-only source schedule";} return std::nullopt;
  }
  // Reconstruct the only admitted schedule before decoding any caller-authored
  // operations. Exact wire comparison refuses compact but expansive generators.
  transport_detail::Reader identity(lowered->runtimeValue.data()+13,lowered->runtimeValue.size()-13);
  StableId terminal=0;
  if (!identity.integer(terminal) || !validStableId(terminal)) {
    error="Paired Geometry Core terminal identity is malformed"; return std::nullopt;
  }
  ValueDescriptor expected;
  expected.carrier=CarrierKind::instances3D; expected.stableId=terminal;
  expected.sourceStableId=declaration.points.sourceStableId; expected.sourceRevision=declaration.structuralRevision;
  expected.data=InstancesData{}; expected.dispatchCount=3;
  auto retained=std::make_shared<RetainedMeshData>(); retained->geometry=*declaration.reference;
  OperationRecord body{declaration.outputStableId,OperationCode::retainedMesh,declaration.terminalStableId,declaration.outputStableId,{}};
  body.retainedMesh=std::move(retained);
  expected.operations={std::move(body),{declaration.pointsOutputStableId,OperationCode::copy,0,declaration.pointsOutputStableId,{}},
      {terminal,OperationCode::instanceOnPoints,declaration.pointsOutputStableId,terminal,{},declaration.outputStableId}};
  const auto admitted=admitValue(expected,lowered->contract,limits,context,error);
  if (!admitted || lowerRuntimePlan(lowered->contract,*admitted)!=bytes) {
    if (error.empty()) {error="Paired Geometry Core wire differs from its exact admitted source schedule";} return std::nullopt;
  }
  return admitLowered(bytes,limits,context,owner,error,source,std::move(lease));
  } catch (const std::bad_alloc&) { error="Paired Geometry Core admission allocation failed before publication"; return std::nullopt; }
}

BackendCapabilities
NativeGeometryCoreCapabilitySource::geometryCoreCapabilities() const {
  const auto native = backend_.info();
  const auto geometry = backend_.geometryCoreCapabilities();
  BackendCapabilities capabilities;
  capabilities.backendIdentity = native.backend;
  capabilities.deviceIdentity = native.device;
  capabilities.nativeGpuAvailable = native.available;
  capabilities.geometryCoreExecution = native.available &&
                                       geometry.immutableSourceBuffers &&
                                       geometry.stableElementIds;
  capabilities.immutableSourceBuffers = geometry.immutableSourceBuffers;
  capabilities.stableElementIds = geometry.stableElementIds;
  capabilities.typedFieldEvaluation = geometry.typedFieldEvaluation;
  capabilities.gpuInstancingWithoutMeshExpansion =
      geometry.gpuInstancingWithoutMeshExpansion;
  capabilities.supportedCarriers = geometry.supportedCarriers;
  capabilities.limits.maxVertices = geometry.maxVertices;
  capabilities.limits.maxIndices = geometry.maxIndices;
  capabilities.limits.maxPoints = geometry.maxPoints;
  capabilities.limits.maxCurvePoints = geometry.maxCurvePoints;
  capabilities.limits.maxSplines = geometry.maxSplines;
  capabilities.limits.maxInstances = geometry.maxInstances;
  capabilities.limits.maxFieldElements = geometry.maxFieldElements;
  capabilities.limits.maxAttributes = geometry.maxAttributes;
  capabilities.limits.maxOperations = geometry.maxOperations;
  capabilities.limits.maxDispatches = geometry.maxDispatches;
  capabilities.limits.maxBufferBytes = geometry.maxBufferBytes;
  return capabilities;
}

namespace {
thread_local std::uint64_t geometryCorePreviewPendingResults = 0;
} // namespace

std::uint64_t geometryCorePreviewPendingResultsOnThisThread() noexcept {
  return geometryCorePreviewPendingResults;
}

void noteGeometryCorePreviewPendingOnThisThread() noexcept {
  ++geometryCorePreviewPendingResults;
}

std::optional<NativeGeometryExecution>
executeNativeGeometry(arbitgpu::NativeFixtureSceneBackend &backend,
                      const RuntimeAdmission &admission, std::uint32_t width,
                      std::uint32_t height, std::string &error,
                      bool diagnosticInstanceIdentityColors,
                      arbitgpu::NativeFixtureSceneRuntimeInputs runtimeInputs,
                      SpectrumEvaluation* spectrumEvaluation,
                      const ImportedGeometryEvaluator& importedGeometry,
                      double timelineTimeSeconds, const PointsEvaluator& pointsEvaluator,
                      const videorender::ParticleHistorySource* waterHistory, double waterProjectSeconds,
                      const videorender::WaterCoupledFrame* coupledWater,
                      const std::shared_ptr<const videorender::ParticlePointsBodyRequest>& pointsBodyRequest,
                      const PointsBodyEvaluator& pointsBodyEvaluator,
                      arbitgpu::NativeFixturePrepareUse prepareUse) {
  using namespace HarmonicMIDI::grid;
  error.clear();
  try {
  if (admission.residency() && !admission.residency()->admitNativeHeader(error)) return std::nullopt;
  if (!admission.matchesRuntimeResidency()) {
    error="Native Geometry Core ownership differs from its issued residency"; return std::nullopt;
  }
  if (!admission.plan || width == 0 || height == 0) {
    error = "Geometry Core native execution requires a plan and non-zero extent";
    return std::nullopt;
  }
  if (!admission.sourceContext.deferredPoints.empty() && !admission.owner.valid()) {
    error="Deferred Points3D requires a valid native plan owner"; return std::nullopt;
  }
  if (!admission.sourceContext.evaluatedBodyGeometry.empty()) {
    error="Native immutable admission cannot retain body frame samples"; return std::nullopt;
  }
  if (!admission.sourceContext.evaluatedPoints.empty()) {
    error="Native admission cannot retain point frame samples"; return std::nullopt;
  }
  std::optional<RuntimeResidency::Attempt> residencyAttempt;
  if (admission.residency() && admission.sourceContext.deferredBodyGeometry.empty()
      && !admission.residency()->admitNativeAttempt(error)) return std::nullopt;
  std::optional<videorender::ParticleBodyCallerCeiling> bodyCallerCeiling;
  std::shared_ptr<const videorender::ParticleBodyCallerLease> bodyFrameLease;
  std::shared_ptr<const videorender::ParticlePointsBodyFrame> pairedFrame;
  if (!admission.sourceContext.deferredBodyGeometry.empty()) {
    if (admission.bodySource && !admission.bodySource->caller()->charge(videorender::particlePointsBodyNativeHeaderVisits)) {
      error="Native paired body current-header comparison work is short"; return std::nullopt;
    }
    if (!admission.bodySource || !admission.bodyAdmissionLease || !pointsBodyRequest || !pointsBodyEvaluator
        || pointsBodyRequest->source()!=admission.bodySource || !(pointsBodyRequest->owner()==admission.owner)
        || pointsBodyRequest->parameters().bodyReplayContext->width!=static_cast<int>(width)
        || pointsBodyRequest->parameters().bodyReplayContext->height!=static_cast<int>(height)
        || pointsBodyRequest->frame()!=runtimeInputs.canonicalBlockCFrame.get()
        || pointsBodyRequest->clipSeconds()!=timelineTimeSeconds
        || admission.sourceContext.deferredBodyGeometry.size()!=1 || admission.sourceContext.deferredPoints.size()!=1
        || !admission.sourceContext.admittedGeometryOrSceneSources.empty()
        || !admission.surfaceMaterial.empty() || !admission.plan->value().descriptor().spectrumFields.empty()
        || !admission.plan->value().descriptor().scoreFields.empty()) {
      error="Native paired body requires its independent current request, exact owner/frame/clocks and bounded source"; return std::nullopt;
    }
    if (!admission.residency()) {
      error="Native paired body requires its privately issued runtime census"; return std::nullopt;
    }
    residencyAttempt.emplace(admission.residency());
    if (!residencyAttempt->active()) {
      error="Native paired body runtime attempt is already active or unadmitted"; return std::nullopt;
    }
    videorender::ParticlePointsBodyEnvelope envelope;
    const auto& source=admission.bodySource->declaration();
    if (!particlePointsBodyCoreEnvelope(source,0,true,envelope)
        || envelope.bytes>admission.plan->receipt().effectiveLimits.maxBufferBytes) {
      error="Native paired body exceeds its prospective Core frame budget"; return std::nullopt;
    }
    // Lower the persistent caller before the frame lease or any callback. An
    // evaluator retaining extra genuine replays cannot exceed this Core cap.
    bodyCallerCeiling.emplace(admission.bodySource->caller(),admission.plan->receipt().effectiveLimits.maxBufferBytes);
    videorender::ParticlePointsBodyEnvelope conversion;
    std::size_t replayPeak=0,attempt=0;
    if (!videorender::ParticlePointsBodyFrame::replayEnvelope(*pointsBodyRequest,conversion,replayPeak,error)) return std::nullopt;
    if (!checkedAdd(admission.bodySource->caller()->usage().logicalBytes,envelope.bytes,attempt)
        || !checkedAdd(attempt,sizeof(videorender::ParticleBodyCallerLease),attempt)
        || !checkedAdd(attempt,replayPeak,attempt)
        || attempt>admission.plan->receipt().effectiveLimits.maxBufferBytes) {
      std::size_t attemptedWork=0;
      if (!checkedAdd(envelope.work,conversion.work,attemptedWork)) attemptedWork=std::numeric_limits<std::size_t>::max();
      admission.bodySource->caller()->charge(attemptedWork);
      error="Native paired body and incumbent/escaped owners exceed its Core frame budget"; return std::nullopt;
    }
    bodyFrameLease=admission.bodySource->caller()->reserve(envelope.bytes,envelope.work);
    if (!bodyFrameLease) { error="Native paired body prospective frame/copy admission is short"; return std::nullopt; }
    const auto body=admission.sourceContext.deferredBodyGeometry.find(source.outputStableId);
    const auto points=admission.sourceContext.deferredPoints.find(source.pointsOutputStableId);
    std::size_t visits=0;
    if (body==admission.sourceContext.deferredBodyGeometry.end() || points==admission.sourceContext.deferredPoints.end()
        || !sameBodyDeclaration(body->second,source) || !(points->second==source.points)
        || !admission.bodySource->body()->recipe().sourceComparison.total(visits)
        || !admission.bodySource->body()->matchesExactSource(source.source,visits)) {
      error="Native paired body declaration or admitted source was substituted"; return std::nullopt;
    }
    const auto& immutable=admission.plan->value().descriptor();
    if (immutable.operations.size()!=3 || immutable.operations[0].code!=OperationCode::retainedMesh
        || immutable.operations[1].code!=OperationCode::copy || immutable.operations[2].code!=OperationCode::instanceOnPoints
        || !immutable.attributes.empty() || immutable.sourceStableId!=source.points.sourceStableId
        || immutable.sourceRevision!=source.structuralRevision
        || immutable.operations[2].inputStableId!=source.pointsOutputStableId
        || immutable.operations[2].secondaryInputStableId!=source.outputStableId
        || !validateOperationPlan(immutable,admission.sourceContext,nullptr,error)) {
      if (error.empty()) error="Native paired body operation schedule differs from its admitted three-record source";
      return std::nullopt;
    }
    pairedFrame=pointsBodyEvaluator(pointsBodyRequest,error);
    if (!pairedFrame || !pairedFrame->matches(*pointsBodyRequest,error)) {
      if (error.empty()) {error="Native paired body evaluator did not return a current completed replay";} return std::nullopt;
    }
  } else if (pointsBodyRequest || pointsBodyEvaluator || admission.bodySource || admission.bodyAdmissionLease) {
    error="Native paired body authority has no declared body source"; return std::nullopt;
  }
  std::shared_ptr<const void> ordinaryNativeResidency;
  if (!pairedFrame && admission.residency()) {
    std::size_t bytes=0,contextBytes=0;
    // Each actual execution owns distinct replay/copy and native objects. Its
    // aliases retain this registration as well as the immutable owner's token.
    if (!runtimeContextCopyBytes(admission.sourceContext,contextBytes)
        || !checkedMul(contextBytes,16,contextBytes)
        || !checkedMul(admission.plan->receipt().value.bufferBytes,16,bytes)
        || !checkedAdd(bytes,contextBytes,bytes)
        || !checkedAdd(bytes,2*sizeof(Visual3DScene)+sizeof(arbitgpu::NativeFixtureSceneFrame)
            +sizeof(arbitgpu::NativeFixtureSceneResources)+sizeof(OrdinaryNativePayload)+6*4*sizeof(void*),bytes)
        || !addArrayBytes(admission.plan->receipt().effectiveLimits.maxInstances,2*sizeof(StableId),bytes)) {
      error="Ordinary native residency arithmetic overflow"; return std::nullopt;
    }
    ordinaryNativeResidency=admission.residency()->retainNative(bytes,
        admission.plan->receipt().effectiveLimits.maxBufferBytes,error);
    if (!ordinaryNativeResidency) return std::nullopt;
  }
  auto frameContext = admission.sourceContext;
  if (pairedFrame) {
    frameContext.evaluatedPoints.emplace(admission.bodySource->declaration().pointsOutputStableId,pairedFrame->points());
    frameContext.evaluatedBodyGeometry.emplace(admission.bodySource->declaration().outputStableId,pairedFrame->geometry());
  }
  for (const auto& source : frameContext.deferredPoints) {
    if (pairedFrame && source.first==admission.bodySource->declaration().pointsOutputStableId) continue;
    videowire::geometry::ValueDescriptor sampled;
    if (!pointsEvaluator || !pointsEvaluator(source.first,source.second,admission.owner,sampled,error)) {
      if (error.empty()) error = "Deferred Points3D requires its exact compiled frame provider";
      return std::nullopt;
    }
    if (source.second.capacity > admission.plan->receipt().effectiveLimits.maxPoints) {
      error = "Deferred Points3D capacity exceeds the admitted backend limit"; return std::nullopt;
    }
    frameContext.evaluatedPoints.emplace(source.first,std::move(sampled));
  }
  auto framePlan=admission.plan;
  const auto& authored = admission.plan->value().descriptor();
  std::optional<videowire::geometry::ValueDescriptor> animated;
  for (const auto& operation : authored.operations)
    if (operation.retainedMesh && !operation.retainedMesh->importedAnimation.empty()) {
      if (!importedGeometry) { error="Animated Geometry3D has no frame evaluator"; return std::nullopt; }
      if (!animated) animated=authored;
      const auto index=static_cast<std::size_t>(&operation-authored.operations.data());
      auto source=std::make_shared<videowire::geometry::RetainedMeshData>(*operation.retainedMesh);
      if (!importedGeometry(*source,error)) return std::nullopt;
      if (source->geometry.vertexIds!=operation.retainedMesh->geometry.vertexIds
          || source->geometry.indices!=operation.retainedMesh->geometry.indices
          || source->geometry.positions.size()!=operation.retainedMesh->geometry.positions.size()
          || source->importedAnimation!=operation.retainedMesh->importedAnimation) {
        error="Animated Geometry3D changed its admitted topology or source request"; return std::nullopt;
      }
      animated->operations[index].retainedMesh=std::move(source);
    }
  const bool orderedInstanceScore=!authored.scoreFields.empty()
      && videowire::geometry::hasOnlyOrderedScoreBindings(authored);
  // Ordered instance Score Fields retain authored samples as ownership anchors.
  // Their current positions are evaluated once below, after frame callbacks exist.
  if ((animated || !frameContext.deferredPoints.empty()) && !orderedInstanceScore) {
    if (!animated) animated=authored;
    videowire::geometry::ValueDescriptor replayed;
    if (!videowire::geometry::validateOperationPlan(*animated,frameContext,nullptr,error,&replayed)) return std::nullopt;
    const auto& limits=admission.plan->receipt().effectiveLimits;
    auto contract=admission.plan->contract();
    if (contract.carrier!=replayed.carrier) {
      error="Evaluated geometry changed its admitted terminal carrier"; return std::nullopt;
    }
    NativeGeometryCoreCapabilitySource frameCapabilities(backend);
    auto frameAdmission=admitPlanValue(replayed,
        videowire::geometry::withAttributeContract(contract,replayed),limits,frameContext,frameCapabilities,error);
    if (!frameAdmission) return std::nullopt;
    framePlan=std::make_shared<const AdmittedPlanValue>(std::move(*frameAdmission));
    animated=std::move(replayed);
  }
  const auto &descriptor = animated ? *animated : authored;
  const videowire::geometry::GeometryData *mesh = nullptr;
  const videowire::geometry::InstancesData *instances = nullptr;
  const std::vector<videowire::geometry::AttributeData>* meshAttributes=&descriptor.attributes;
  videowire::geometry::GeometryData deformedMesh;
  videowire::geometry::MaterializedInstancePlan materialized;
  const bool scorePrototype=descriptor.carrier==videowire::geometry::CarrierKind::instances3D
      && std::any_of(descriptor.scoreFields.begin(),descriptor.scoreFields.end(),[](const auto& binding) {
        return binding.consumerGeometryStableId!=0;
      });
  const bool runtimeFields=videowire::geometry::hasRuntimeFields(descriptor) || scorePrototype;
  auto evaluation = videorender::waterSurfaceFrameEvaluation(
      waterHistory, waterProjectSeconds, timelineTimeSeconds, runtimeInputs.canonicalBlockCFrame, coupledWater);
  evaluation.scoreAt=[&](const auto& operation,const auto& positions,auto& result,auto& diagnostic) {
    if (operation.code==videowire::geometry::OperationCode::evaluateField
        && (operation.inputStableId!=operation.outputStableId || scorePrototype)) {
      const auto binding=std::find_if(descriptor.scoreFields.begin(),descriptor.scoreFields.end(),[&](const auto& b) {
        return b.fieldStableId==operation.stableId
            && (scorePrototype ? b.consumerGeometryStableId!=0
                : b.sourceGeometryStableId==operation.inputStableId && b.consumerGeometryStableId==0);
      });
      if (binding==descriptor.scoreFields.end()) {
        diagnostic="Instance Score Field has no admitted canonical score binding"; return false;
      }
      return scorefield::evaluateInstanceField(*binding,positions,runtimeInputs.canonicalBlockCFrame,result,diagnostic,!scorePrototype);
    }
    return scorefield::evaluateSampleField(operation,positions,runtimeInputs.canonicalBlockCFrame,result,diagnostic);
  };
  // The scalar reader that matches the band reader a binding uses: project
  // analysis for master, the selected track or group stream otherwise. Each
  // scalar sample comes from the frame that supplies the same band sample.
  const auto scalarInputFor=[&](const videowire::geometry::spectrum::Binding& binding,bool history) {
    videowire::geometry::spectrum::ScalarInput input;
    if (!spectrumEvaluation) return input;
    input.current=&spectrumEvaluation->scalars;
    if (binding.source==videowire::geometry::spectrum::Source::master) {
      input.at=history ? spectrumEvaluation->historyScalarFeaturesAt : spectrumEvaluation->scalarFeaturesAt;
    } else {
      input.at=[sourceScalars=spectrumEvaluation->sourceScalarFeaturesAt,source=binding.source,track=binding.sourceTrackId]
          (double seconds,videowire::geometry::spectrum::Scalars& scalars) {
        return static_cast<bool>(sourceScalars) && sourceScalars(source,track,seconds,scalars);
      };
    }
    return input;
  };
  const auto spectrumInput=[&](const videowire::geometry::spectrum::Binding& binding,
      videowire::geometry::spectrum::Bands& bands,
      videowire::geometry::spectrum::FeaturesAt& featuresAt,
      videowire::geometry::spectrum::FeaturesAt& historyAt,std::string& failure) {
    if (!spectrumEvaluation) { failure="Spectrum geometry requires the shared frame analysis"; return false; }
    bands=spectrumEvaluation->bands;
    featuresAt=spectrumEvaluation->featuresAt;
    historyAt=spectrumEvaluation->historyFeaturesAt;
    if (binding.source==videowire::geometry::spectrum::Source::master) return true;
    const auto& sourceAt=spectrumEvaluation->sourceFeaturesAt;
    if (!sourceAt || !sourceAt(binding.source,binding.sourceTrackId,spectrumEvaluation->timeSeconds,bands)) {
      failure="Selected track/group spectrum is unavailable. Check the saved source ID, use the standalone app and refresh project audio analysis.";
      return false;
    }
    featuresAt=[sourceAt,source=binding.source,track=binding.sourceTrackId](double seconds) {
      videowire::geometry::spectrum::Bands selected{};
      sourceAt(source,track,seconds,selected);
      return selected;
    };
    historyAt=featuresAt;
    return true;
  };
  evaluation.spectrumAt=[&](const auto& binding,auto& followed,std::string& failure) {
    const auto found=std::find_if(descriptor.spectrumFields.begin(),descriptor.spectrumFields.end(),[&](const auto& b) {
      return b.fieldStableId==binding.fieldStableId && videowire::geometry::spectrum::sameBinding(b,binding);
    });
    if (found==descriptor.spectrumFields.end() || !spectrumEvaluation
        || (binding.source==videowire::geometry::spectrum::Source::master
            && !spectrumEvaluation->liveFrameAvailable && !spectrumEvaluation->featuresAt)) {
      failure="Audio Spectrum Field requires its admitted binding and shared frame analysis"; return false;
    }
    const auto index=static_cast<std::size_t>(found-descriptor.spectrumFields.begin());
    spectrumEvaluation->followers.resize(descriptor.spectrumFields.size());
    videowire::geometry::spectrum::Bands current{};
    videowire::geometry::spectrum::FeaturesAt featuresAt,historyAt;
    if (!spectrumInput(binding,current,featuresAt,historyAt,failure)) return false;
    bool missingHistory=false;
    if (binding.source!=videowire::geometry::spectrum::Source::master && featuresAt) {
      const auto sourceAt=spectrumEvaluation->sourceFeaturesAt;
      featuresAt=[&missingHistory,sourceAt,source=binding.source,track=binding.sourceTrackId](double seconds) {
        videowire::geometry::spectrum::Bands selected{};
        if (!sourceAt(source,track,seconds,selected)) missingHistory=true;
        return selected;
      };
    }
    if (!videowire::geometry::spectrum::evaluate(binding,current,spectrumEvaluation->timeSeconds,
        featuresAt,spectrumEvaluation->followers[index],followed,failure,scalarInputFor(binding,false))) return false;
    if (missingHistory) { failure="Selected spectrum source is missing required project-time history"; return false; }
    return true;
  };
  evaluation.spectrumHistoryAt=[&](const auto& binding,auto& rows,std::string& failure) {
    const auto found=std::find_if(descriptor.spectrumFields.begin(),descriptor.spectrumFields.end(),[&](const auto& b) {
      return b.fieldStableId==binding.fieldStableId && videowire::geometry::spectrum::sameBinding(b,binding);
    });
    if (found==descriptor.spectrumFields.end() || !spectrumEvaluation || !binding.history.enabled) {
      failure="Spectral History Field requires its admitted binding and shared frame analysis"; return false;
    }
    const auto index=static_cast<std::size_t>(found-descriptor.spectrumFields.begin());
    spectrumEvaluation->followers.resize(descriptor.spectrumFields.size());
    auto& follower=spectrumEvaluation->followers[index];
    const auto current=spectrumEvaluation->bands;
    auto historyAt=spectrumEvaluation->historyFeaturesAt;
    bool missingHistory=false;
    if (binding.source!=videowire::geometry::spectrum::Source::master) {
      const auto sourceAt=spectrumEvaluation->sourceFeaturesAt;
      if (!sourceAt) { follower={}; failure="Selected spectrum source is missing required project-time history"; return false; }
      historyAt=[&missingHistory,sourceAt,source=binding.source,track=binding.sourceTrackId](double seconds) {
        videowire::geometry::spectrum::Bands selected{};
        if (!sourceAt(source,track,seconds,selected)) missingHistory=true;
        return selected;
      };
    }
    if (!videowire::geometry::spectrum::evaluateHistory(binding,current,spectrumEvaluation->timeSeconds,
        historyAt,spectrumEvaluation->liveFrameAvailable,admission.use==PlanUse::exportRender,
        follower,rows,failure,spectrumEvaluation->seekGeneration,spectrumEvaluation->loopGeneration,
        scalarInputFor(binding,true))) return false;
    if (missingHistory) {
      follower={}; failure="Selected spectrum source is missing required project-time history"; return false;
    }
    return true;
  };
  const bool orderedAudio=std::any_of(descriptor.spectrumFields.begin(),descriptor.spectrumFields.end(),
      [](const auto& binding) { return binding.target==videowire::geometry::spectrum::Target::geometryDeformation; });
  std::function<bool(videowire::geometry::GeometryData&,const videowire::geometry::spectrum::Binding&,
                     std::string&)> deform;
  if (orderedAudio) {
    if (!spectrumEvaluation) { error="Audio geometry replay requires the shared frame analysis"; return std::nullopt; }
    spectrumEvaluation->followers.resize(descriptor.spectrumFields.size());
    deform=[&](auto& geometry,const auto& binding,std::string& failure) {
      if (binding.history.enabled) {
        // Each vertex reads the row selected by its retained history age.
        std::vector<videowire::geometry::spectrum::Bands> rows;
        if (!evaluation.spectrumHistoryAt(binding,rows,failure)) return false;
        if (rows.size()!=binding.history.depth || binding.history.coordinates.size()!=geometry.positions.size()) {
          failure="Spectral History Audio Deformer rows disagree with its retained vertices"; return false;
        }
        std::vector<double> responses(geometry.positions.size());
        for (std::size_t vertex=0;vertex<responses.size();++vertex)
          responses[vertex]=videowire::geometry::spectrum::deformerHistoryResponse(binding,rows,vertex);
        return videowire::geometry::applyAudioDeformerResponses(geometry,binding,responses,
            spectrumEvaluation->timeSeconds,failure);
      }
      const auto index=static_cast<std::size_t>(&binding-descriptor.spectrumFields.data());
      videowire::geometry::spectrum::Bands bands{},followed{};
      videowire::geometry::spectrum::FeaturesAt featuresAt,historyAt;
      return spectrumInput(binding,bands,featuresAt,historyAt,failure)
          && videowire::geometry::spectrum::evaluate(binding,bands,
          spectrumEvaluation->timeSeconds,featuresAt,
          spectrumEvaluation->followers[index],followed,failure,scalarInputFor(binding,false))
          && videowire::geometry::applyAudioDeformer(geometry,binding,followed,spectrumEvaluation->timeSeconds,failure);
    };
  }
  // One ordered replay combines timeline/score fields and audio deformation.
  // Distribution and queries therefore see the same evaluated source frame.
  if (descriptor.carrier == videowire::geometry::CarrierKind::geometry3D) {
    mesh = videowire::geometry::replayGeometryMesh(descriptor, materialized, error,
        runtimeFields ? &evaluation : nullptr, deform);
    if (!mesh) return std::nullopt;
    if (orderedAudio || runtimeFields) meshAttributes = &materialized.terminalAttributes;
  } else if (descriptor.carrier == videowire::geometry::CarrierKind::instances3D) {
    if (!videowire::geometry::validateOperationPlan(descriptor,frameContext,&materialized,error,nullptr,
        runtimeFields ? &evaluation : nullptr,deform,
        (orderedInstanceScore || scorePrototype) ? &admission.plan->receipt().effectiveLimits : nullptr)) return std::nullopt;
    if (!admission.plan->capabilities().gpuInstancingWithoutMeshExpansion) {
      if (error.empty())
        error = "native Geometry Core instancing lacks retained operation results";
      return std::nullopt;
    }
    instances = &materialized.instances;
    const auto instanceOperation = std::find_if(descriptor.operations.rbegin(),descriptor.operations.rend(),[](const auto& op) {
      return op.code == videowire::geometry::OperationCode::instanceOnPoints;
    });
    const auto sourceId = !instances->instances.empty() ? instances->instances.front().sourceStableId
        : instanceOperation != descriptor.operations.rend() ? instanceOperation->secondaryInputStableId : 0;
    if (!std::all_of(instances->instances.begin(), instances->instances.end(),
                     [sourceId](const auto &instance) {
                       return instance.sourceStableId == sourceId;
                     })) {
      error = "native Geometry Core instancing requires one exact retained source";
      return std::nullopt;
    }
    const auto source = materialized.geometries.find(sourceId);
    if (source == materialized.geometries.end()) {
      error = "native Geometry Core instancing source was not materialized";
      return std::nullopt;
    }
    mesh = &source->second;
    const auto attributes=materialized.geometryAttributes.find(sourceId);
    meshAttributes=attributes==materialized.geometryAttributes.end() ? nullptr : &attributes->second;
  } else {
    error = "native Geometry Core execution does not advertise this carrier";
    return std::nullopt;
  }
  if (!descriptor.spectrumFields.empty()) {
    if (mesh->positions.size() > Visual3DScene::kMaxVertices
        || mesh->indices.size() > Visual3DScene::kMaxIndices) {
      error = "Audio Spectrum Field mesh exceeds the native retained-scene bounds";
      return std::nullopt;
    }
    if (spectrumEvaluation == nullptr) {
      error = "Spectrum geometry requires the frame's shared master analysis";
      return std::nullopt;
    }
    if (instances == nullptr) deformedMesh = *mesh;
    spectrumEvaluation->followers.resize(descriptor.spectrumFields.size());
    const auto scorePlacementAncestors=videowire::geometry::orderedInstanceScorePlacementAncestors(descriptor);
    for (std::size_t index = 0; index < descriptor.spectrumFields.size(); ++index) {
      const auto& binding = descriptor.spectrumFields[index];
      if (binding.target==videowire::geometry::spectrum::Target::instanceTransform
          && scorePlacementAncestors.count(binding.consumerGeometryStableId)!=0) continue;
      if (std::any_of(descriptor.operations.begin(),descriptor.operations.end(),[&](const auto& op) {
          return op.stableId==binding.fieldStableId && op.code==videowire::geometry::OperationCode::fieldAudioSpectrum;
        })) continue;
      if (orderedAudio && binding.target==videowire::geometry::spectrum::Target::geometryDeformation) continue;
      videowire::geometry::spectrum::Bands currentBands{};
      videowire::geometry::spectrum::FeaturesAt featuresAt,historyAt;
      if (!spectrumInput(binding,currentBands,featuresAt,historyAt,error)) return std::nullopt;
      videowire::geometry::spectrum::Bands followed {};
      if (binding.target==videowire::geometry::spectrum::Target::instanceTransform) {
        if (instances == nullptr || binding.coordinates.size() != materialized.instances.instances.size()) {
          error = "Spectrum Instancer has no matching native instance set";
          return std::nullopt;
        }
        // One instance per selected band. History rows use each instance's
        // retained placement age, the same row rule as Spectral History vertices.
        std::vector<float> amounts(binding.coordinates.size());
        if (binding.history.enabled) {
          std::vector<videowire::geometry::spectrum::Bands> rows;
          if (!evaluation.spectrumHistoryAt(binding,rows,error)) return std::nullopt;
          if (rows.size()!=binding.history.depth || binding.history.coordinates.size()!=amounts.size()) {
            error = "Spectral History Spectrum Instancer rows disagree with its instances"; return std::nullopt;
          }
          for (std::size_t instance = 0; instance < amounts.size(); ++instance)
            amounts[instance] = videowire::geometry::spectrum::sampleHistory(binding, rows, instance);
        } else {
          if (!evaluation.spectrumAt(binding,followed,error)) return std::nullopt;
          for (std::size_t instance = 0; instance < amounts.size(); ++instance)
            amounts[instance] = videowire::geometry::spectrum::sample(binding, followed, binding.coordinates[instance]);
        }
        for (std::size_t index = 0; index < materialized.instances.instances.size(); ++index) {
          const auto amount = amounts[index];
          auto& transform = materialized.instances.instances[index].transform;
          transform.translation.x += amount * binding.direction[0];
          transform.translation.y += amount * binding.direction[1];
          transform.translation.z += amount * binding.direction[2];
          const auto scale = [&](std::size_t axis, float base) {
            return std::clamp(base * (1.0f + amount * binding.scaleResponse[axis]), 0.000001f, 1000.0f);
          };
          transform.scale = {scale(0, transform.scale.x), scale(1, transform.scale.y), scale(2, transform.scale.z)};
        }
        continue;
      }
      if (binding.history.enabled && binding.target==videowire::geometry::spectrum::Target::vertexDisplacement) {
        std::vector<videowire::geometry::spectrum::Bands> rows;
        if (instances != nullptr || !videowire::geometry::spectrum::evaluateHistory(binding,
              currentBands, spectrumEvaluation->timeSeconds, historyAt,
              spectrumEvaluation->liveFrameAvailable, admission.use == PlanUse::exportRender,
              spectrumEvaluation->followers[index], rows, error,
              spectrumEvaluation->seekGeneration, spectrumEvaluation->loopGeneration,
              scalarInputFor(binding,true))) return std::nullopt;
        for (std::size_t vertex = 0; vertex < deformedMesh.positions.size(); ++vertex) {
          const auto amount = videowire::geometry::spectrum::sampleHistory(binding, rows, vertex);
          auto& position = deformedMesh.positions[vertex];
          position.x += amount * binding.direction[0];
          position.y += amount * binding.direction[1];
          position.z += amount * binding.direction[2];
        }
        continue;
      }
      if (!videowire::geometry::spectrum::evaluate(binding, currentBands,
            spectrumEvaluation->timeSeconds, featuresAt,
            spectrumEvaluation->followers[index], followed, error, scalarInputFor(binding,false))) return std::nullopt;
      if (instances != nullptr) {
        error = "Audio Spectrum Field requires a Geometry3D draw";
        return std::nullopt;
      }
      if (binding.target == videowire::geometry::spectrum::Target::geometryDeformation) {
        if (!videowire::geometry::applyAudioDeformer(deformedMesh,binding,followed,
              spectrumEvaluation->timeSeconds,error)) return std::nullopt;
        continue;
      }
      for (std::size_t vertex = 0; vertex < deformedMesh.positions.size(); ++vertex) {
        const auto amount = videowire::geometry::spectrum::sample(binding, followed,
                                                                 binding.coordinates[vertex]);
        auto& position = deformedMesh.positions[vertex];
        position.x += amount * binding.direction[0];
        position.y += amount * binding.direction[1];
        position.z += amount * binding.direction[2];
      }
    }
    if (instances == nullptr) mesh = &deformedMesh;
  }
  for (const auto& binding : descriptor.scoreFields) {
    if (scorePrototype) continue;
    if (std::any_of(descriptor.operations.begin(),descriptor.operations.end(),[&](const auto& op) {
          return op.code==videowire::geometry::OperationCode::evaluateField
              && op.stableId==binding.fieldStableId && op.inputStableId!=op.outputStableId;
        })) continue;
    if (instances != nullptr) {
      error = "Score Field requires a Geometry3D draw"; return std::nullopt;
    }
    if (deformedMesh.positions.empty()) deformedMesh=*mesh;
    scorefield::Evaluation evaluated;
    if (!scorefield::evaluate(binding,runtimeInputs.canonicalBlockCFrame,evaluated,error))
      return std::nullopt;
    for (std::size_t index=0;index<deformedMesh.positions.size();++index) {
      auto& p=deformedMesh.positions[index]; const auto& offset=evaluated.offsets[index];
      p.x+=offset[0]; p.y+=offset[1]; p.z+=offset[2];
    }
    mesh=&deformedMesh;
  }
  if (mesh->positions.empty() || mesh->indices.empty() ||
      mesh->vertexIds.size() != mesh->positions.size() ||
      mesh->indices.size() % 3 != 0 ||
      mesh->positions.size() > Visual3DScene::kMaxVertices ||
      mesh->indices.size() > Visual3DScene::kMaxIndices ||
      mesh->positions.size() > std::numeric_limits<std::uint32_t>::max() ||
      mesh->indices.size() > std::numeric_limits<std::uint32_t>::max() ||
      (instances != nullptr && instances->instances.size() > Visual3DScene::kMaxObjects)) {
    error = "Geometry Core mesh exceeds the native retained-scene bounds";
    return std::nullopt;
  }
  // Validate all untrusted references before allocating the fixed retained
  // scene. A hostile index must not cause a large Visual3DScene allocation or
  // any partial copy into backend-owned storage.
  for (const auto index : mesh->indices) {
    if (index >= mesh->positions.size()) {
      error = "Geometry Core native draw contains an out-of-range index";
      return std::nullopt;
    }
  }
  for (const auto &position : mesh->positions) {
    if (!videowire::geometry::finite(position)) {
      error = "Geometry Core native draw contains a non-finite position";
      return std::nullopt;
    }
  }
  const auto materialId = instances != nullptr ? materialized.materialStableId
      : videowire::geometry::retainedMaterialIdentity(descriptor);
  auto scene = std::make_shared<Visual3DScene>();
  scene->id = {1};
  scene->activeCamera = {1};
  scene->ambientColor = {0.12f, 0.12f, 0.12f};
  scene->vertexCount = mesh->positions.size();
  scene->indexCount = mesh->indices.size();
  for (std::size_t index = 0; index < mesh->positions.size(); ++index) {
    const auto &p = mesh->positions[index];
    scene->vertices[index].position = {p.x, p.y, p.z};
  }
  for (std::size_t index = 0; index < mesh->indices.size(); ++index)
    scene->indices[index] = mesh->indices[index];
  for (std::size_t index = 0; index + 2 < mesh->indices.size(); index += 3) {
    const auto a = mesh->indices[index], b = mesh->indices[index + 1],
               c = mesh->indices[index + 2];

    const auto &pa = mesh->positions[a], &pb = mesh->positions[b], &pc = mesh->positions[c];
    const float ux = pb.x - pa.x, uy = pb.y - pa.y, uz = pb.z - pa.z;
    const float vx = pc.x - pa.x, vy = pc.y - pa.y, vz = pc.z - pa.z;
    const float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz,
                nz = ux * vy - uy * vx;
    for (const auto vertex : {a, b, c}) {
      scene->vertices[vertex].normal.x += nx;
      scene->vertices[vertex].normal.y += ny;
      scene->vertices[vertex].normal.z += nz;
    }
  }
  for (std::size_t index = 0; index < mesh->positions.size(); ++index) {
    auto &normal = scene->vertices[index].normal;
    const auto length = std::sqrt(normal.x * normal.x + normal.y * normal.y +
                                  normal.z * normal.z);
    if (length > 0.0f) {
      normal.x /= length; normal.y /= length; normal.z /= length;
    } else normal = {0.0f, 0.0f, 1.0f};
  }
  scene->materials[0].id = {1};
  if (meshAttributes!=nullptr)
    for (const auto* name : {"uv","color"})
      if (const auto* attribute=videowire::geometry::findNamedAttribute(*meshAttributes,name)) {
        const bool color=std::string(name)=="color";
        if (attribute->descriptor.domain!=videowire::geometry::Domain::vertex
            || attribute->descriptor.valueType!=(color ? videowire::geometry::ValueType::color
                                                     : videowire::geometry::ValueType::vector)
            || attribute->elements.size()!=mesh->positions.size()) {
          error="Imported vertex attributes do not match native geometry"; return std::nullopt;
        }
        for (std::size_t i=0;i<attribute->elements.size();++i) {
          const auto& value=attribute->elements[i].components;
          if (color) scene->vertices[i].color={static_cast<float>(value[0]),static_cast<float>(value[1]),
              static_cast<float>(value[2]),static_cast<float>(value[3])};
          else scene->vertices[i].uv={static_cast<float>(value[0]),static_cast<float>(value[1])};
        }
      }
  if (meshAttributes!=nullptr)
    if (const auto* authored=videowire::geometry::findNamedAttribute(*meshAttributes,"normal")) {
      if (authored->descriptor.domain!=videowire::geometry::Domain::vertex
          || authored->descriptor.valueType!=videowire::geometry::ValueType::vector
          || authored->elements.size()!=mesh->positions.size()) {
        error="Authored normals do not match native geometry"; return std::nullopt;
      }
      for (std::size_t i=0;i<authored->elements.size();++i) {
        const auto& n=authored->elements[i].components;
        const auto length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (!std::isfinite(length) || length==0) { error="Authored normal is invalid"; return std::nullopt; }
        scene->vertices[i].normal={static_cast<float>(n[0]/length),static_cast<float>(n[1]/length),static_cast<float>(n[2]/length)};
      }
    }
  // Set Material is part of the immutable lowered value. Map its stable identity
  // to a bounded native color so the operation changes both GL and Metal draw.
  const auto channel = [materialId](unsigned shift) {
    return 0.2f + 0.7f * static_cast<float>((materialId >> shift) & 0xffu) / 255.0f;
  };
  scene->materials[0].baseColor = {channel(0), channel(8), channel(16)};
  scene->materials[0].roughness = 0.55f;
  scene->materialCount = 1;
  const auto objectCount = instances == nullptr ? std::size_t{1}
                                                : instances->instances.size();
  for (std::size_t index = 0; index < objectCount; ++index) {
    auto &object = scene->objects[index];
    if (instances == nullptr) {
      object.id = {1};
    } else {
      const auto &instance = instances->instances[index];
      if (instance.stableId > std::numeric_limits<std::uint32_t>::max()) {
        error = "Geometry Core instance identity exceeds native scene identity bounds";
        return std::nullopt;
      }
      object.id = {static_cast<std::uint32_t>(instance.stableId)};
      object.transform.translation = {instance.transform.translation.x,
                                      instance.transform.translation.y,
                                      instance.transform.translation.z};
      object.transform.rotation = {instance.transform.rotation.x,
                                   instance.transform.rotation.y,
                                   instance.transform.rotation.z,
                                   instance.transform.rotation.w};
      object.transform.scale = {instance.transform.scale.x,
                                instance.transform.scale.y,
                                instance.transform.scale.z};
    }
    object.material = {1};
    object.vertexCount = static_cast<std::uint32_t>(mesh->positions.size());
    object.indexCount = static_cast<std::uint32_t>(mesh->indices.size());
  }
  scene->objectCount = objectCount;
  if (instances && instances->instances.empty()) {
    // A transform-only scene root gives the native renderer a valid blank
    // scene. It has no geometry, no particle identity and no instance draw.
    scene->objects[0].id={1}; scene->objects[0].material=scene->materials[0].id;
    scene->objects[0].vertexCount=scene->objects[0].indexCount=0;
    scene->objectCount=1;
  }
  scene->lights[0].id = {1};
  scene->lights[0].kind = SceneLightKind::Directional;
  scene->lights[0].transform.rotation = {-0.25f, 0.25f, 0.0f, 0.9354143f};
  scene->lights[0].intensity = 1.5f;
  scene->lightCount = 1;
  scene->cameras[0].id = {1};
  scene->cameras[0].transform.translation = {0.0f, 0.0f, 5.0f};
  scene->cameras[0].verticalFovRadians = 0.7853981634f;
  scene->cameras[0].nearPlane = 0.1f;
  scene->cameras[0].farPlane = 1000.0f;
  scene->cameraCount = 1;
  auto frameDescriptor=descriptor;
  if (orderedAudio || runtimeFields) frameDescriptor.attributes=materialized.terminalAttributes;
  geometrysurfacematerial::ProgramSet authoredSurfaces;
  if (!admission.surfaceMaterial.empty()) {
    const auto decoded=geometrysurfacematerial::decodeSet(admission.surfaceMaterial,error);
    if (!decoded) return std::nullopt;
    authoredSurfaces=*decoded;
  }
  const bool hasTable=videowire::geometry::hasMaterialTable(descriptor,materialized);
  const bool overrideSurface=!authoredSurfaces.empty() && authoredSurfaces.front().sourceMaterial==0;
  if (overrideSurface && hasTable) {
    error = "Reactive Surface overrides do not replace material-table bindings"; return std::nullopt;
  }
  if (!authoredSurfaces.empty() && !overrideSurface && !hasTable) {
    error = "Geometry Surface table programs require their material table"; return std::nullopt;
  }
  std::map<videowire::geometry::StableId,SceneMaterialId> surfaceIds;
  for (const auto& program : authoredSurfaces) if (program.sourceMaterial)
    surfaceIds.emplace(program.sourceMaterial,SceneMaterialId{geometrysurfacematerial::materialId(program.material)});
  if (overrideSurface) {
    const SceneMaterialId id{geometrysurfacematerial::materialId(authoredSurfaces.front().material)};
    scene->materials[0].id=id;
    for (std::size_t index=0;index<scene->objectCount;++index) scene->objects[index].material=id;
  }
  std::vector<videowire::geometry::MaterialTableDrawBinding> drawBindings;
  if (!videowire::geometry::applyGeometryMaterialTable(frameDescriptor,materialized,meshAttributes,
        *scene,0,objectCount,error,&surfaceIds,&drawBindings)) return std::nullopt;
  std::shared_ptr<arbitgpu::NativeFixtureSurfaceMaterialProgram> surfaceProgram;
  if (!authoredSurfaces.empty()) {
    const auto nativeBackend = backend.info().backend;
    const auto target = nativeBackend == "opengl" ? materialprogram::BackendTarget::OpenGl
        : nativeBackend == "metal" ? materialprogram::BackendTarget::Metal : materialprogram::BackendTarget::Invalid;
    for (std::size_t index = 0; index < scene->objectCount; ++index) {
      const auto& object = scene->objects[index];
      if (object.indexCount == 0) continue;
      const auto receipt=std::find_if(drawBindings.begin(),drawBindings.end(),
          [&](const auto& draw){return draw.object==object.id.value;});
      const auto source=overrideSurface ? 0 : receipt!=drawBindings.end() ? receipt->sourceMaterial : 0;
      const auto authoredSurface=std::find_if(authoredSurfaces.begin(),authoredSurfaces.end(),
          [source](const auto& program){return program.sourceMaterial==source;});
      if (authoredSurface==authoredSurfaces.end()) continue;
      const auto instanceIndex=receipt!=drawBindings.end() ? receipt->instanceIndex : index;
      auto request = authoredSurface->material;
      request.sceneSnapshot = scene;
      request.binding.object = object.id;
      const auto binding = videorender::fixture3d::admitSurfaceMaterialBinding(scene, request, target, error);
      if (!binding) return std::nullopt;
      auto native=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*binding->nativeProgram());
      for (const auto& field : authoredSurface->fields)
        if (!surfacematerialfield::evaluate(field,evaluation,instanceIndex,native->parameters,native->timeMixEndColor,error))
          return std::nullopt;
      // Table batching uses ordinary retained-scene draws; apply the same
      // canonical instance appearance once, after material Field modulation.
      if (hasTable && instances) {
        const auto appearance=videowire::geometry::instanceAppearance(frameDescriptor.attributes,instanceIndex);
        for (std::size_t channel=0;channel<3;++channel) {
          native->parameters.baseColorMetallic[channel]*=appearance.color[channel];
          native->timeMixEndColor[channel]*=appearance.color[channel];
          native->parameters.emissionRoughness[channel]+=appearance.emission[channel];
        }
        native->parameters.normalOpacity[3]*=appearance.color[3];
        if (appearance.metallic>=0) native->parameters.baseColorMetallic[3]=appearance.metallic;
        if (appearance.roughness>=0) native->parameters.emissionRoughness[3]=appearance.roughness;
      }
      if (!surfaceProgram) surfaceProgram=std::move(native);
      else surfaceProgram->objectPrograms.push_back(std::move(native));
    }
    if (!std::isfinite(timelineTimeSeconds) || std::abs(timelineTimeSeconds) > std::numeric_limits<float>::max()) {
      error = "Geometry Surface requires finite native frame time"; return std::nullopt;
    }
    runtimeInputs.timeSeconds = static_cast<float>(timelineTimeSeconds);
  }
  auto prepared = instances != nullptr && !instances->instances.empty() && !videowire::geometry::hasMaterialTable(descriptor,materialized)
      ? backend.prepareGeometryInstances(scene, surfaceProgram, framePlan,
                                         diagnosticInstanceIdentityColors,
                                         (orderedAudio || runtimeFields) ? &materialized.terminalAttributes : nullptr,
                                         prepareUse)
      : backend.prepare(scene, surfaceProgram, prepareUse);
  if (prepared.pending) {
    // A Preview prepare whose shader is still compiling defers without an error.
    // The counter lets the caller tell this apart from a failure.
    if (prepareUse != arbitgpu::NativeFixturePrepareUse::Preview) {
      error = "Geometry Core native resource preparation returned pending for a blocking prepare";
      return std::nullopt;
    }
    ++geometryCorePreviewPendingResults;
    error.clear();
    return std::nullopt;
  }
  if (!prepared.prepared || !prepared.resources) {
    error = prepared.error.empty() ? "Geometry Core native resource preparation failed"
                                   : prepared.error;
    return std::nullopt;
  }
  // Score fields consume the frame above. A note-instance mapping separately
  // opts the draw into polyphonic copies of the resulting mesh.
  if (!runtimeInputs.noteInstanceMapping)
    runtimeInputs.canonicalBlockCFrame.reset();
  auto rendered = backend.render(scene, prepared.resources, width, height,
                                 std::move(runtimeInputs));
  if (!rendered.rendered || !rendered.frame) {
    error = rendered.error.empty() ? "Geometry Core native draw failed" : rendered.error;
    return std::nullopt;
  }
  rendered.stats.staticUploadCount = prepared.stats.staticUploadCount;
  rendered.stats.vertexBytes = prepared.stats.vertexBytes;
  rendered.stats.indexBytes = prepared.stats.indexBytes;
  std::vector<videowire::geometry::StableId> drawnInstanceIds;
  if (instances != nullptr) {
    drawnInstanceIds.reserve(instances->instances.size());
    for (const auto &instance : instances->instances)
      drawnInstanceIds.push_back(instance.stableId);
  }
  if (pairedFrame) {
    auto payload=std::make_shared<const PointsBodyNativePayload>(PointsBodyNativePayload{
        std::move(bodyFrameLease),std::move(pairedFrame),admission.ownerLease,framePlan,scene,rendered.frame,prepared.resources});
    prepared.resources=std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>(payload,payload->resources.get());
    rendered.frame=std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>(payload,payload->frame.get());
    if (admission.nativeResources) {
      std::lock_guard<std::mutex> lock(admission.nativeResources->mutex);
      admission.nativeResources->value=prepared.resources;
    }
  }
  if (!pairedFrame && ordinaryNativeResidency) {
    auto payload=std::make_shared<const OrdinaryNativePayload>(OrdinaryNativePayload{
        std::move(ordinaryNativeResidency),admission.ownerLease,framePlan,scene,rendered.frame,prepared.resources});
    prepared.resources=std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>(payload,payload->resources.get());
    rendered.frame=std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>(payload,payload->frame.get());
  }
  if (!pairedFrame && admission.nativeResources) {
    std::lock_guard<std::mutex> lock(admission.nativeResources->mutex);
    admission.nativeResources->value=prepared.resources;
  }
  return NativeGeometryExecution{std::move(prepared.resources),
                                 std::move(rendered.frame),
                                 admission.ownerLease,
                                 std::move(drawnInstanceIds),
                                 rendered.stats,
                                 framePlan->receipt().backendBinding};
  } catch (const std::bad_alloc&) {
    if (admission.sourceContext.deferredBodyGeometry.empty()) throw;
    error="Native paired body allocation failed before publication"; return std::nullopt;
  }
}
} // namespace videohelper::geometry
