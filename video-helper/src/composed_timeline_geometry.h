#pragma once

#include <atomic>

#include "geometry_core_admission.h"
#include "water_surface_replay.h"
#include "water_replay_source_budget.h"
#include "timeline_coupled_water_replay.h"
#include "../../shared/GeometryTimelineSceneBinding.h"

namespace videohelper::geometry {
// Names every zero owner field, so a refused owner points at the producer that
// did not supply it.
inline std::string zeroPlanOwnerFields(const PlanOwnerIdentity& owner) {
  std::string names;
  const auto append = [&names](bool zero, const char* name) {
    if (!zero) return;
    if (!names.empty()) names += ", ";
    names += name;
  };
  append(owner.projectGeneration == 0, "projectGeneration");
  append(owner.helperGeneration == 0, "helperGeneration");
  append(owner.deviceGeneration == 0, "deviceGeneration");
  append(owner.clipIdentity == 0, "clipIdentity");
  append(owner.planGeneration == 0, "planGeneration");
  return names;
}
// Keep the admission lease with the evaluated snapshot until the scene renderer
// and its published native frame have released the owned source.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> evaluateComposedTimelineGeometry(
    GeometryCorePlanRuntime& runtime, const PlanOwnerIdentity& owner,
    const videowire::geometry::TimelineSceneBinding& binding,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored,
    double seconds, std::string& error,
    const videorender::ParticleHistorySource* waterHistory = nullptr,
    double projectSeconds = std::numeric_limits<double>::quiet_NaN(),
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalFrame = {},
    const videorender::WaterCoupledFrame* coupledWater = nullptr) {
  if (!authored) {
    error = "Timeline Scene3D requires an authored scene";
    return {};
  }
  if (!owner.valid()) {
    error = "Timeline Scene3D owner is invalid because it has zero " + zeroPlanOwnerFields(owner);
    return {};
  }
  if (!std::isfinite(seconds)) {
    error = "Timeline Scene3D requires a finite frame time";
    return {};
  }
  if (coupledWater && (seconds!=projectSeconds || coupledWater->projectSeconds!=projectSeconds)) {
    error = "Coupled Timeline Scene3D requires the completed frame at its exact project-zero time";
    return {};
  }
  const auto value = videowire::geometry::decodeTimelineScenePlan(binding, error);
  if (!value || !videowire::geometry::validTimelineSceneBinding(binding, *authored, *value, error)) {
    if (error.empty()) error = "Timeline Scene3D binding does not match its authored mesh";
    return {};
  }
  const auto bytes = videowire::geometry::decodeLoweredPlanText(binding.loweredPlan, error);
  if (!bytes || !runtime.reconcileGeneration(owner.projectGeneration, owner.helperGeneration,
          owner.deviceGeneration, error)) return {};
  const auto admitted = owner.use == PlanUse::preview
      ? runtime.admitPreview(owner, *bytes, {}, {}, error)
      : runtime.admitExport(owner, *bytes, {}, {}, error);
  if (!admitted) return {};
  struct OwnedScene final {
    HarmonicMIDI::grid::Visual3DScene scene;
    std::shared_ptr<const std::uint8_t> lease;
  };
  auto result = std::make_shared<OwnedScene>();
  result->scene = *authored;
  result->lease = admitted->ownerLease;
  auto evaluation = videorender::waterSurfaceFrameEvaluation(waterHistory, projectSeconds, seconds, std::move(canonicalFrame), coupledWater);
  if (!videowire::geometry::replaceTimelineSceneMesh(binding,
          admitted->plan->value(), seconds, result->scene, error, &evaluation)) return {};
  return std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>(result, &result->scene);
}
} // namespace videohelper::geometry

namespace videohelper::geometry::timelineevaluation {
using Scene = HarmonicMIDI::grid::Visual3DScene;
using namespace videowire::geometry;

// This is a separate incremental evaluator budget. Admission's resident and
// attempt envelopes remain charged by GeometryCorePlanRuntime. These bytes
// count new logical payloads, including escaped results, rather than allocator
// capacity, tree balancing, library math, shared_ptr control blocks or driver internals.
struct Limits final {
  std::size_t maxAdditionalBytes = kMaximumBufferBytes;
  videorender::watersourcebudget::Limits waterSource;
  // Caller allowances for source-derived passes, not elapsed CPU guarantees.
  std::size_t maxWaterIdentityVisits=std::numeric_limits<std::size_t>::max();
  std::size_t maxWaterRestPositionVisits=std::numeric_limits<std::size_t>::max();
  std::size_t maxWaterMemoCopyElements=std::numeric_limits<std::size_t>::max();
  std::size_t maxWaterColumnDiscoveryVisits=std::numeric_limits<std::size_t>::max();
};
struct Envelope final {
  std::size_t retainedBytes=0, scratchBytes=0, concurrentBytes=0;
  // Only work/logicalBytes aggregate sibling schedules. Recipe metadata is
  // not an identity receipt; each child keeps the actual immutable sources.
  videorender::watersourcebudget::Recipe water;
  decltype(water.work) attemptedWaterWork{};
  std::size_t waterChildren=0, waterHeightBytes=0, waterMemoBytes=0;
  std::size_t waterIdentityVisits=0, waterHeightCopies=0, waterHeightValidationVisits=0;
  std::size_t waterRestPositionVisits=0, waterMemoCopyElements=0, waterColumnDiscoveryVisits=0;
  std::size_t preflightChildVisits=0, preflightOperationVisits=0, cardinalityLookupVisits=0;
  std::size_t preflightCountInitializations=0, preflightLightVisits=0;
  std::size_t bindingCopyRecords=0, bindingCopyBytes=0, sceneCopyRecords=0, sceneCopyStringBytes=0;
  std::size_t ownerTransferRecords=0, timeAnchorWrites=0, receiptTransfers=0;
  std::size_t sceneValidationCalls=0, sceneRecordVisits=0, sceneStringByteVisits=0;
  std::size_t hierarchyLookupVisits=0, hierarchyDepthIterations=0;
  std::size_t matrixConstructions=0, matrixMultiplyTerms=0, matrixElementVisits=0;
  std::size_t matrixPostcheckIterations=0, lightChannelVisits=0, vertexChannelVisits=0;
  std::size_t rangeComparisons=0, topologyComparisons=0;
  std::size_t replayOperationVisits=0, replayOperationScans=0, replayInputVisits=0;
  std::size_t replayDependencyVisits=0, replayOutputVisits=0, replayParameterVisits=0;
  std::size_t gridVertexVisits=0, gridCellVisits=0;
  std::size_t fieldInitializations=0, fieldElementVisits=0, fieldComponentVisits=0;
  std::size_t meshCopyElements=0, displacementVisits=0;
  std::size_t appendVertexVisits=0, appendIndexVisits=0, appendTriangleVisits=0;
  std::size_t appendNormalVisits=0, replacementVertexVisits=0;
};

inline bool add(std::size_t count,std::size_t passes,std::size_t& out) noexcept {
  return addArrayBytes(count,passes,out);
}

// Only sizes, operation tags, IDs and grid/count parameters are read here.
// No decoder, mesh copy, validator or replay runs during this schedule census.
// The table is fixed stack storage and each lookup scans only prior records.
inline bool childSchedule(const TimelineSceneChildAdmission& child,Envelope& out,
    std::size_t& peakReplay,std::string& error,bool allowCoupled=false) {
  struct Count final { StableId id=0; std::size_t vertices=0,indices=0,fields=0; };
  std::array<Count,128> counts{};
  const auto& value=child.plan().value().descriptor();
  const auto operations=value.operations.size();
  if (operations==0 || operations>counts.size() || !value.attributes.empty()
      || !value.spectrumFields.empty() || !value.scoreFields.empty()) {
    error="Timeline family evaluation has no incremental attribute/reactive-source schedule"; return false;
  }
  std::size_t used=0,geometryBytes=0,fieldBytes=0,edges=0,meshScratch=0;
  std::size_t geometryOutputs=0,largestVertices=0,displacements=0;
  if (!checkedAdd(out.preflightCountInitializations,counts.size(),out.preflightCountInitializations)) return false;
  const auto lookup=[&](StableId id,Count& count) {
    for (std::size_t i=0;i<used;++i) {
      if (!checkedAdd(out.cardinalityLookupVisits,1,out.cardinalityLookupVisits)) return false;
      if (counts[i].id==id) { count=counts[i]; return true; }
    }
    return false;
  };
  for (const auto& op : value.operations) {
    if (!checkedAdd(out.preflightOperationVisits,1,out.preflightOperationVisits)) return false;
    if (op.retainedMesh || op.sdfSurface || !op.field.attributeName.empty()
        || op.field.auxiliaryInputStableId!=0) {
      error="Timeline family evaluation has no incremental retained-source or auxiliary-field schedule"; return false;
    }
    Count count; count.id=op.outputStableId;
    Count source;
    if (op.code==OperationCode::grid) {
      std::uint32_t columns=0,rows=0; std::size_t cells=0;
      if (!exactIntegerParameter(op.parameters[0],2,columns)
          || !exactIntegerParameter(op.parameters[1],2,rows)
          || !checkedMul(columns,rows,count.vertices)
          || !checkedMul(columns-1,rows-1,cells) || !checkedMul(cells,6,count.indices)
          || !add(count.vertices,3,out.gridVertexVisits) || !add(cells,3,out.gridCellVisits)) return false;
    } else if (op.code==OperationCode::fieldWaterHeight) {
      if (op.field.values[4]!=0 && !allowCoupled) {
        error="Timeline family coupled Water requires independently owned child frames and an incremental coupled-surface schedule"; return false;
      }
      if (!lookup(op.inputStableId,source) || !source.vertices) return false;
      count.fields=source.vertices;
      // Three materializers validate the grid/settings and build the field.
      // The fresh replay's own validation/preparation is counted separately.
      if (!add(source.vertices,3,out.waterIdentityVisits)
          || !add(source.indices,3,out.waterIdentityVisits)
          || !add(source.vertices,3,out.waterColumnDiscoveryVisits)
          || !add(12,3,out.replayParameterVisits)
          || !add(source.vertices,3,out.fieldInitializations)
          || !add(source.vertices,3,out.fieldElementVisits)
          || !add(source.vertices,3,out.waterHeightValidationVisits)) return false;
    } else if (op.code==OperationCode::positionField || op.code==OperationCode::indexField
        || op.code==OperationCode::stableIdentityField) {
      if (op.secondaryInputStableId || !lookup(op.inputStableId,source) || !source.vertices) return false;
      count.fields=source.vertices;
      // retainedData clones the exact mesh, then materializeElementField
      // initializes and visits every field element and inserts its stable ID.
      if (!add(source.vertices,2*3,out.meshCopyElements)
          || !add(source.indices,3,out.meshCopyElements)
          || !add(count.fields,3,out.fieldInitializations)
          || !add(count.fields,3,out.fieldElementVisits)
          || !add(count.fields,3*4,out.fieldComponentVisits)) return false;
    } else if (op.code==OperationCode::fieldConstant || op.code==OperationCode::constantVectorField) {
      const auto n=op.code==OperationCode::fieldConstant ? op.field.values[4] : double(op.parameters[3]);
      if (!std::isfinite(n) || n<1 || n>Scene::kMaxVertices || std::floor(n)!=n) return false;
      count.fields=static_cast<std::size_t>(n);
      if (!add(count.fields,3,out.fieldInitializations)
          || (op.code==OperationCode::fieldConstant && !add(4,3,out.fieldComponentVisits))) return false;
    } else if (op.code==OperationCode::fieldTimeline || op.code==OperationCode::fieldVectorMeasure
        || op.code==OperationCode::fieldComposeVector) {
      // This bounded lane uses one input; binary/broadcast composition follows
      // with its own exact cardinalities rather than an assumed field capacity.
      if (op.secondaryInputStableId) {
        error="Timeline family evaluation has no incremental binary-field broadcast schedule"; return false;
      }
      if (!lookup(op.inputStableId,source) || !source.fields) return false;
      count.fields=source.fields;
      if (!add(count.fields,3,out.fieldInitializations)
          || !add(count.fields,3,out.fieldElementVisits)
          // Input/output validFieldElement each reads all four components.
          // VectorMeasure also always runs its three-axis source loop.
          || !add(count.fields,3*(8+(op.code==OperationCode::fieldVectorMeasure ? 3 : 0)),out.fieldComponentVisits)) return false;
    } else if (op.code==OperationCode::vectorDisplacement) {
      Count field; ++displacements;
      if (!lookup(op.inputStableId,source) || !source.vertices
          || !lookup(op.secondaryInputStableId,field) || !field.fields) return false;
      count.vertices=source.vertices; count.indices=source.indices;
      // retainedData copies positions/IDs/indices. displaceElements stages one
      // Vec3 per vertex, visits inputs once and transfers staged positions once.
      if (!add(count.vertices,2*3,out.meshCopyElements)
          || !add(count.indices,3,out.meshCopyElements)
          || !add(count.vertices,2*3,out.displacementVisits)
          || !add(count.vertices,3*4,out.fieldComponentVisits)
          || !add(operations,3,out.replayOperationScans)) return false;
    } else {
      error="Timeline family evaluation has no incremental schedule for operation "
          +std::to_string(static_cast<unsigned>(op.code)); return false;
    }
    if (count.vertices>Scene::kMaxVertices || count.indices>Scene::kMaxIndices
        || count.fields>Scene::kMaxVertices) return false;
    if (op.inputStableId && op.inputStableId!=op.outputStableId && op.code!=OperationCode::grid)
      if (!checkedAdd(edges,1,edges)) return false;
    if (op.secondaryInputStableId && !checkedAdd(edges,1,edges)) return false;
    if (count.vertices) {
      std::size_t bytes=sizeof(GeometryData);
      if (!add(count.vertices,sizeof(Vec3)+sizeof(StableId),bytes)
          || !add(count.indices,sizeof(std::uint32_t),bytes)
          || !checkedAdd(geometryBytes,bytes,geometryBytes)) return false;
      meshScratch=std::max(meshScratch,bytes);
      largestVertices=std::max(largestVertices,count.vertices); ++geometryOutputs;
    }
    if (count.fields && (!checkedAdd(fieldBytes,sizeof(FieldData),fieldBytes)
        || !add(count.fields,sizeof(FieldElement),fieldBytes))) return false;
    counts[used++]=count;
  }
  // Three released validateOperationPlan calls. Each has an ancestor pass,
  // main pass, closure, generated-output pass and retained-attribute transfer.
  // Map/set lookup and balancing internals are not scalar-operation receipts.
  if (!add(operations,3,out.replayOperationVisits)
      || !add(operations,3,out.replayOperationScans)
      || !add(operations,3*3,out.replayInputVisits)
      // Ancestor edge inserts, generate dependency inserts, displacement
      // secondary edges, closure edge transfers and closure pop iterations.
      || !add(edges,3*3,out.replayDependencyVisits)
      || !add(operations,3,out.replayDependencyVisits)
      || !add(displacements,3,out.replayDependencyVisits)
      || !add(1,3,out.replayDependencyVisits)
      || !add(operations,3*2,out.replayOutputVisits)
      || !add(operations,3*4,out.replayParameterVisits)) return false;
  // Constructed field settings run in validateOperationPlan and its field
  // helper. The operation census above separates field-element work; settings
  // validation is bounded by two eight-value scans per constructed operation.
  for (const auto& op : value.operations) {
    if (!checkedAdd(out.preflightOperationVisits,1,out.preflightOperationVisits)) return false;
    if (isConstructedFieldOperation(op.code) && !add(8,2*3,out.replayParameterVisits)) return false;
  }
  Count terminal;
  if (!lookup(value.stableId,terminal) || !terminal.vertices
      || terminal.vertices!=child.object().vertexCount || terminal.indices!=child.object().indexCount) return false;
  // Replay compares IDs and indices twice with runtime and a complete mesh
  // once without runtime; replace compares IDs once more and scene indices once.
  if (!add(terminal.vertices,4,out.topologyComparisons)
      || !add(terminal.indices,4,out.topologyComparisons)
      || !add(terminal.vertices,1,out.meshCopyElements)
      || !add(terminal.vertices,2,out.appendVertexVisits)
      || !add(terminal.indices,2*2,out.appendIndexVisits)
      || !add(terminal.indices/3,2,out.appendTriangleVisits)
      || !add(terminal.indices,2,out.appendNormalVisits)
      || !add(terminal.vertices,2,out.appendNormalVisits)
      || !add(terminal.vertices,1,out.replacementVertexVisits)
      // Each append scans runtime fields, three supportsTimeline predicates,
      // animated sources and material indices. validTimeline scans its three
      // supportsTimeline predicates. Short-circuiting only reduces these visits.
      || !add(operations,2*6+3,out.replayOperationScans)
      || !add(terminal.vertices,1,out.sceneRecordVisits)
      || !add(terminal.indices,1,out.sceneRecordVisits)) return false;
  std::size_t replay=sizeof(MaterializedInstancePlan);
  // Every main replay owns these otherwise-empty containers too.
  using K=StableId;
  replay+=sizeof(std::map<K,CarrierKind>)+sizeof(std::map<K,K>)+sizeof(std::map<K,std::set<K>>)
      +4*sizeof(std::set<K>)+sizeof(std::map<K,InstancesData>)+sizeof(std::map<K,FieldData>)
      +sizeof(std::map<K,K>)+sizeof(std::map<K,std::set<K>>)
      +sizeof(std::map<K,std::pair<Domain,ValueType>>)+sizeof(std::map<K,std::vector<AttributeData>>)
      +sizeof(std::map<K,std::set<K>>)+2*sizeof(std::set<K>)+2*sizeof(std::vector<K>);
  // Logical upper bounds for produced/material/dependency/operation/generated
  // maps, ancestor parents, empty per-output material tables/attributes, field
  // types and field payloads. Opaque node links/padding remain allocator detail.
  if (!checkedAdd(replay,geometryBytes,replay) || !checkedAdd(replay,fieldBytes,replay)
      || !add(operations,sizeof(std::pair<const K,CarrierKind>)+sizeof(std::pair<const K,K>)
          +2*sizeof(std::pair<const K,std::set<K>>)+3*sizeof(K)
          +sizeof(std::pair<const K,std::map<std::uint8_t,OperationRecord>>)
          +sizeof(std::pair<const K,std::vector<AttributeData>>)
          +sizeof(std::pair<const K,std::pair<Domain,ValueType>>),replay)
      || !add(edges,2*sizeof(K),replay)
      || !add(edges+1,sizeof(K),replay)
      || !add(operations,sizeof(std::pair<const K,FieldData>),replay)
      || !add(geometryOutputs,sizeof(std::pair<const K,GeometryData>)
          +sizeof(std::pair<const K,std::vector<AttributeData>>),replay)
      || !checkedAdd(replay,meshScratch,replay)
      || !add(largestVertices,sizeof(Vec3)+sizeof(K),replay)) return false;
  // replace keeps its verification replay alive during evaluated append replay.
  std::size_t concurrentReplay=0;
  if (!checkedMul(replay,2,concurrentReplay)) return false;
  peakReplay=std::max(peakReplay,concurrentReplay);
  return true;
}

// Only the existing four-operation grid/Water/Z-displacement lowering is
// admitted by this new lane. The immutable time-zero terminal is its rest mesh.
// Other Water source recipes require their own retained-source schedule.
struct WaterChild final {
  const OperationRecord* operation=nullptr;
  const GeometryData* rest=nullptr;
  WaterSurfaceGrid grid;
  videorender::watersourcebudget::Recipe recipe;
};
inline bool waterChild(const TimelineSceneChildAdmission& child,WaterChild& out,Envelope& envelope,std::string& error) {
  const auto& value=child.plan().value().descriptor();
  const auto& ops=value.operations;
  if (!checkedAdd(envelope.preflightOperationVisits,ops.size(),envelope.preflightOperationVisits)) {
    error="Timeline family Water operation census overflow"; return false;
  }
  const auto found=std::find_if(ops.begin(),ops.end(),[](const auto& op) {
    return op.code==OperationCode::fieldWaterHeight;
  });
  if (found==ops.end()) return true;
  if (ops.size()!=4 || ops[0].code!=OperationCode::grid || found!=ops.begin()+1
      || ops[2].code!=OperationCode::fieldComposeVector || ops[3].code!=OperationCode::vectorDisplacement
      || found->inputStableId!=ops[0].outputStableId || found->secondaryInputStableId
      || ops[2].inputStableId!=found->outputStableId || ops[2].secondaryInputStableId
      || ops[2].field.mode!=1 || ops[2].field.valueType!=ValueType::vector
      || ops[2].field.values!=std::array<double,8>{0,0,1,0,0,0,0,0}
      || ops[3].inputStableId!=ops[0].outputStableId || ops[3].secondaryInputStableId!=ops[2].outputStableId
      || ops[3].parameters!=std::array<float,4>{1,0,0,0} || ops[3].outputStableId!=value.stableId
      || !std::holds_alternative<GeometryData>(value.data)) {
    error="Timeline family Water requires its exact static grid and released Z-displacement lowering"; return false;
  }
  std::uint32_t columns=0,rows=0;
  if (!exactIntegerParameter(ops[0].parameters[0],2,columns)
      || !exactIntegerParameter(ops[0].parameters[1],2,rows)) return false;
  out.operation=&*found; out.rest=&std::get<GeometryData>(value.data);
  out.grid={columns,rows,ops[0].parameters[2],ops[0].parameters[2]};
  return true;
}
inline bool sameWaterOperation(const OperationRecord& a,const OperationRecord& b) {
  return a.code==b.code && a.stableId==b.stableId && a.inputStableId==b.inputStableId
      && a.outputStableId==b.outputStableId && a.secondaryInputStableId==b.secondaryInputStableId
      && a.parameters==b.parameters && a.field==b.field && equal(a.transform,b.transform)
      && a.retainedMesh==b.retainedMesh && a.sdfSurface==b.sdfSurface;
}

inline bool scheduleWork(const Envelope& e,std::size_t& total) {
  if (!checkedAdd(total,e.waterIdentityVisits,total)) return false;
  if (!checkedAdd(total,e.waterHeightCopies,total)) return false;
  if (!checkedAdd(total,e.waterHeightValidationVisits,total)) return false;
  if (!checkedAdd(total,e.waterRestPositionVisits,total)) return false;
  if (!checkedAdd(total,e.waterMemoCopyElements,total)) return false;
  if (!checkedAdd(total,e.waterColumnDiscoveryVisits,total)) return false;
  if (!checkedAdd(total,e.preflightChildVisits,total)) return false;
  if (!checkedAdd(total,e.preflightOperationVisits,total)) return false;
  if (!checkedAdd(total,e.cardinalityLookupVisits,total)) return false;
  if (!checkedAdd(total,e.preflightCountInitializations,total)) return false;
  if (!checkedAdd(total,e.preflightLightVisits,total)) return false;
  if (!checkedAdd(total,e.bindingCopyRecords,total)) return false;
  if (!checkedAdd(total,e.bindingCopyBytes,total)) return false;
  if (!checkedAdd(total,e.sceneCopyRecords,total)) return false;
  if (!checkedAdd(total,e.sceneCopyStringBytes,total)) return false;
  if (!checkedAdd(total,e.ownerTransferRecords,total)) return false;
  if (!checkedAdd(total,e.timeAnchorWrites,total)) return false;
  if (!checkedAdd(total,e.receiptTransfers,total)) return false;
  if (!checkedAdd(total,e.sceneValidationCalls,total)) return false;
  if (!checkedAdd(total,e.sceneRecordVisits,total)) return false;
  if (!checkedAdd(total,e.sceneStringByteVisits,total)) return false;
  if (!checkedAdd(total,e.hierarchyLookupVisits,total)) return false;
  if (!checkedAdd(total,e.hierarchyDepthIterations,total)) return false;
  if (!checkedAdd(total,e.matrixConstructions,total)) return false;
  if (!checkedAdd(total,e.matrixMultiplyTerms,total)) return false;
  if (!checkedAdd(total,e.matrixElementVisits,total)) return false;
  if (!checkedAdd(total,e.matrixPostcheckIterations,total)) return false;
  if (!checkedAdd(total,e.lightChannelVisits,total)) return false;
  if (!checkedAdd(total,e.vertexChannelVisits,total)) return false;
  if (!checkedAdd(total,e.rangeComparisons,total)) return false;
  if (!checkedAdd(total,e.topologyComparisons,total)) return false;
  if (!checkedAdd(total,e.replayOperationVisits,total)) return false;
  if (!checkedAdd(total,e.replayOperationScans,total)) return false;
  if (!checkedAdd(total,e.replayInputVisits,total)) return false;
  if (!checkedAdd(total,e.replayDependencyVisits,total)) return false;
  if (!checkedAdd(total,e.replayOutputVisits,total)) return false;
  if (!checkedAdd(total,e.replayParameterVisits,total)) return false;
  if (!checkedAdd(total,e.gridVertexVisits,total)) return false;
  if (!checkedAdd(total,e.gridCellVisits,total)) return false;
  if (!checkedAdd(total,e.fieldInitializations,total)) return false;
  if (!checkedAdd(total,e.fieldElementVisits,total)) return false;
  if (!checkedAdd(total,e.fieldComponentVisits,total)) return false;
  if (!checkedAdd(total,e.meshCopyElements,total)) return false;
  if (!checkedAdd(total,e.displacementVisits,total)) return false;
  if (!checkedAdd(total,e.appendVertexVisits,total)) return false;
  if (!checkedAdd(total,e.appendIndexVisits,total)) return false;
  if (!checkedAdd(total,e.appendTriangleVisits,total)) return false;
  if (!checkedAdd(total,e.appendNormalVisits,total)) return false;
  if (!checkedAdd(total,e.replacementVertexVisits,total)) return false;
  return true;
}

inline bool sceneSchedule(const Scene& scene,std::size_t children,Envelope& out,
    std::size_t& sceneBytes,std::string& error) {
  const auto O=scene.objectCount,M=scene.materialCount,T=scene.textureCount;
  const auto L=scene.lightCount,C=scene.cameraCount,V=scene.vertexCount;
  std::size_t calls=0,depths=0,multiplies=0,lookups=0,records=0,strings=0;
  if (!checkedMul(children,2,calls) || !checkedMul(O,O+1,depths)
      || !checkedMul(O,O,multiplies) || !checkedAdd(depths,multiplies,lookups)
      || !checkedMul(lookups,O,lookups)
      || !checkedMul(calls,lookups,out.hierarchyLookupVisits)
      || !checkedAdd(depths,multiplies,out.hierarchyDepthIterations)
      || !checkedMul(calls,out.hierarchyDepthIterations,out.hierarchyDepthIterations)
      || !checkedAdd(O,multiplies,out.matrixConstructions)
      || !checkedMul(calls,out.matrixConstructions,out.matrixConstructions)
      || !checkedMul(calls,multiplies,out.matrixMultiplyTerms)
      || !checkedMul(out.matrixMultiplyTerms,64,out.matrixMultiplyTerms)
      || !add(out.matrixConstructions,32,out.matrixElementVisits)
      || !add(out.matrixMultiplyTerms,3,out.matrixElementVisits)
      || !add(calls,O*30,out.matrixElementVisits)
      || !add(calls,multiplies*32,out.matrixElementVisits)
      || !add(calls,O*6,out.matrixPostcheckIterations)) goto overflow;
  out.sceneValidationCalls=calls;
  for (const auto n : {O,M,T,L,C}) if (n && !add(n,n-1,records)) goto overflow;
  records/=2; // Exact duplicateIdAt earlier-record visits.
  if (!checkedAdd(records,lookups,records) || !add(O,O+M+scene.indexCount,records)
      || !add(M,5*T,records) || !checkedAdd(records,C,records)
      || !checkedAdd(records,T+M+2*O+V+2*L+C,records)
      || !checkedMul(calls,records,out.sceneRecordVisits)
      || !add(calls,L*(3+3*diffractionmaterial::kMaximumSpectralSamples),out.lightChannelVisits)
      || !add(calls,V*8,out.vertexChannelVisits)
      || !add(children,O,out.rangeComparisons)
      // valid binding plus replacement perform three object and two material
      // lookups, then the baseline equality walks vertices and local indices.
      || !add(children,3*O+2*M,out.sceneRecordVisits)) goto overflow;
  out.preflightLightVisits=scene.lights.size();
  for (const auto& light : scene.lights) {
    if (!checkedAdd(strings,light.environmentImage.asset.id.size(),strings)
        || !checkedAdd(strings,light.environmentImage.asset.contentSha256.size(),strings)) goto overflow;
  }
  if (!checkedMul(calls,strings,out.sceneStringByteVisits)) goto overflow;
  sceneBytes=sizeof(Scene);
  if (!add(scene.textureTexels.size(),sizeof(HarmonicMIDI::grid::SceneTexelRgba8),sceneBytes)
      || !checkedAdd(sceneBytes,strings,sceneBytes)
      || !add(Scene::kMaxLights,2,sceneBytes)) goto overflow;
  // One result default construction and full assignment. Each child creates
  // a baseline and evaluated scene, both default-constructed fixed arrays.
  records=scene.vertices.size()+scene.indices.size()+scene.objects.size()+scene.materials.size()
      +scene.textures.size()+scene.lights.size()+scene.cameras.size();
  if (!add(records,2+2*children,out.sceneCopyRecords)
      || !checkedAdd(out.sceneCopyRecords,scene.textureTexels.size(),out.sceneCopyRecords)) goto overflow;
  out.sceneCopyStringBytes=strings;
  error.clear(); return true;
overflow:
  error="Timeline family incremental scene schedule overflow"; return false;
}
} // namespace videohelper::geometry::timelineevaluation

namespace videohelper::geometry {
// Keep one evaluator beside its GeometryCorePlanRuntime. The caller establishes
// the initial active generation with reconcileGeneration before admission. This
// class does not invent an active-generation fact or rewind it on a stale call.
// Runtime re-admission is the existing authority for exact anchor/use/generation
// checks, and charges its existing admission work separately from this extension.
class TimelineSceneFamilyEvaluator final {
  struct Ledger final { std::atomic<std::size_t> retainedBytes{0}; };
  struct Reservation final {
    std::shared_ptr<Ledger> ledger;
    std::size_t bytes=0;
    ~Reservation() {
      if (ledger) ledger->retainedBytes.fetch_sub(bytes,std::memory_order_relaxed);
    }
  };
  struct WaterSample final {
    const TimelineSceneChildAdmission* child=nullptr;
    const videowire::geometry::OperationRecord* operation=nullptr;
    const videowire::geometry::GeometryData* rest=nullptr;
    videowire::geometry::WaterSurfaceGrid grid;
    double sourceSeconds=0,projectSeconds=0;
    std::vector<double> heights;
  };
  struct OwnedScene final {
    // Reverse member destruction releases every payload and owner before bytes.
    Reservation reservation;
    HarmonicMIDI::grid::Visual3DScene scene;
    std::shared_ptr<const AdmittedTimelineScenePlans> family;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalFrame;
    std::vector<WaterSample> waterSamples;
    double sourceSeconds=0, projectSeconds=std::numeric_limits<double>::quiet_NaN();
  };
public:
  explicit TimelineSceneFamilyEvaluator(GeometryCorePlanRuntime& runtime) noexcept : runtime_(runtime) {}
  TimelineSceneFamilyEvaluator(const TimelineSceneFamilyEvaluator&)=delete;
  TimelineSceneFamilyEvaluator& operator=(const TimelineSceneFamilyEvaluator&)=delete;

  std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> evaluate(
      const PlanOwnerIdentity& owner,const std::shared_ptr<const AdmittedTimelineScenePlans>& family,
      const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored,
      double sourceSeconds,const timelineevaluation::Limits& limits,std::string& error,
      timelineevaluation::Envelope* receipt=nullptr,
      double projectSeconds=std::numeric_limits<double>::quiet_NaN(),
      const videorender::WaterCoupledFrame* coupledWater=nullptr,
      const videorender::ParticleHistorySource* waterHistory=nullptr,
      std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalFrame={}) {
    using namespace timelineevaluation;
    error.clear(); if (receipt) *receipt={};
    if (!family || !authored || !owner.valid() || !(owner==family->owner()) || !std::isfinite(sourceSeconds)) {
      error="Timeline family evaluation requires its exact admitted owner, authored scene and finite source time"; return {};
    }
    if (coupledWater) {
      error="Timeline family coupled Water requires independently owned child frames and an incremental coupled-surface schedule"; return {};
    }
    // Global project time is independent of overlay/clip-local source time.
    if (!std::isnan(projectSeconds) && !std::isfinite(projectSeconds)) {
      error="Timeline family evaluation requires a finite explicit project time"; return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Envelope envelope; std::size_t sceneBytes=0,peakReplay=0,bindingsBytes=0,waterScratch=0;
    struct Receipt final {
      Envelope* destination; Envelope& envelope; const decltype(envelope.water.work)& attempts;
      ~Receipt() { if (destination) { envelope.attemptedWaterWork=attempts; *destination=envelope; } }
    } report{receipt,envelope,attemptedWaterWork_};
    std::array<WaterChild,Scene::kMaxObjects> waterChildren{};
    if (family->children().empty() || family->children().size()>waterChildren.size()) {
      error="Timeline family Water child schedule exceeds the admitted object capacity"; return {};
    }
    std::size_t childIndex=0;
    envelope.preflightChildVisits=family->children().size();
    envelope.bindingCopyRecords=family->children().size();
    envelope.ownerTransferRecords=3; // Family, ledger and final aliasing scene owner.
    envelope.timeAnchorWrites=2; envelope.receiptTransfers=receipt ? 1 : 0;
    if (!sceneSchedule(family->scene(),family->children().size(),envelope,sceneBytes,error)) return {};
    for (const auto& child : family->children()) {
      if (!childSchedule(child,envelope,peakReplay,error)) {
        if (error.empty()) error="Timeline family incremental operation/cardinality schedule overflow or mismatch";
        return {};
      }
      auto& water=waterChildren[childIndex++];
      if (!waterChild(child,water,envelope,error)) return {};
      if (water.operation) {
        using namespace videorender::watersourcebudget;
        if (!waterHistory || !canonicalFrame) {
          error="Timeline family Water requires actual borrowed history and its current canonical frame"; return {};
        }
        // Census visits remain charged even if this or a later sibling refuses.
        const auto census=static_cast<std::size_t>(Work::censusNotes);
        const auto notes=waterHistory->score ? waterHistory->score->notes.size() : 0;
        std::uint64_t attempted=0;
        if (!videorender::watersourcebudget::checkedAdd(attemptedWaterWork_[census],notes,attempted)
            || attempted>limits.waterSource.work[census]) {
          error="Timeline family Water exceeds its cumulative note census allowance"; return {};
        }
        attemptedWaterWork_[census]=attempted;
        auto remaining=limits.waterSource;
        for (std::size_t i=1;i<remaining.work.size();++i) {
          std::uint64_t committed=0;
          if (!videorender::watersourcebudget::checkedAdd(attemptedWaterWork_[i],envelope.water.work[i],committed)
              || committed>remaining.work[i]) {
            error="Timeline family Water exceeds its cumulative source work allowance"; return {};
          }
          remaining.work[i]-=committed;
        }
        if (!preflight(*waterHistory,*water.operation,*water.rest,water.grid,canonicalFrame,
                projectSeconds,sourceSeconds,sourceSeconds,remaining,water.recipe,error)) return {};
        for (std::size_t i=0;i<water.recipe.work.size();++i)
          if (!videorender::watersourcebudget::checkedAdd(envelope.water.work[i],water.recipe.work[i],envelope.water.work[i])) {
            error="Timeline family Water source work overflow"; return {};
          }
        std::size_t sourceBytes=0;
        for (std::size_t i=0;i<water.recipe.logicalBytes.size();++i) {
          if (!videorender::watersourcebudget::checkedAdd(envelope.water.logicalBytes[i],water.recipe.logicalBytes[i],envelope.water.logicalBytes[i])
              || envelope.water.logicalBytes[i]>limits.waterSource.logicalBytes[i]
              || water.recipe.logicalBytes[i]>std::numeric_limits<std::size_t>::max()
              || !videowire::geometry::checkedAdd(sourceBytes,static_cast<std::size_t>(water.recipe.logicalBytes[i]),sourceBytes)) {
            error="Timeline family Water exceeds its all-child source payload allowance"; return {};
          }
        }
        std::size_t heights=0,memo=sizeof(videowire::geometry::OperationRecord)+sizeof(GeometryData)+sizeof(double)+sizeof(std::vector<double>);
        if (!videowire::geometry::checkedMul(water.rest->positions.size(),sizeof(double),heights)
            || !videowire::geometry::checkedAdd(envelope.waterHeightBytes,heights,envelope.waterHeightBytes)
            || !add(water.rest->positions.size(),sizeof(Vec3)+sizeof(StableId),memo)
            || !add(water.rest->indices.size(),sizeof(std::uint32_t),memo)
            || !videowire::geometry::checkedAdd(memo,heights,memo)
            // Memo copy, callback transfer, field materialization and returned
            // immutable sample may coexist. Source arrays are separate above.
            || !add(heights,4,sourceBytes)
            || !videowire::geometry::checkedAdd(sourceBytes,memo,sourceBytes)
            || !videowire::geometry::checkedAdd(envelope.waterMemoBytes,memo,envelope.waterMemoBytes)
            || !add(water.rest->positions.size(),4,envelope.waterHeightCopies)
            || !add(water.rest->positions.size(),4,envelope.waterHeightValidationVisits)
            // The family callback and released memo both compare the complete rest mesh.
            || !add(water.rest->positions.size(),2,envelope.waterIdentityVisits)
            || !add(water.rest->indices.size(),2,envelope.waterIdentityVisits)
            || !add(water.rest->positions.size(),2,envelope.waterRestPositionVisits)
            // The released memo copies positions, stable IDs and indices once.
            || !add(water.rest->positions.size(),2,envelope.waterMemoCopyElements)
            || !add(water.rest->indices.size(),1,envelope.waterMemoCopyElements)) {
          error="Timeline family Water retained height/memo schedule overflow"; return {};
        }
        if (!add(1,1,envelope.ownerTransferRecords)) return {};
        waterScratch=std::max(waterScratch,sourceBytes); ++envelope.waterChildren;
      }
      std::size_t textBytes=0;
      if (!videowire::geometry::checkedAdd(child.binding().loweredPlan.size(),1,textBytes)
          || !videowire::geometry::addArrayBytes(1,sizeof(videowire::geometry::TimelineSceneBinding),bindingsBytes)
          || !videowire::geometry::checkedAdd(bindingsBytes,textBytes,bindingsBytes)
          || !videowire::geometry::checkedAdd(envelope.bindingCopyBytes,child.binding().loweredPlan.size(),envelope.bindingCopyBytes)) {
        error="Timeline family incremental binding-copy storage overflow"; return {};
      }
    }
    if (envelope.waterIdentityVisits>limits.maxWaterIdentityVisits
        || envelope.waterRestPositionVisits>limits.maxWaterRestPositionVisits
        || envelope.waterMemoCopyElements>limits.maxWaterMemoCopyElements
        || envelope.waterColumnDiscoveryVisits>limits.maxWaterColumnDiscoveryVisits) {
      error="Timeline family Water exceeds its prospective rest/memo work budget"; return {};
    }
    envelope.retainedBytes=sizeof(OwnedScene);
    if (!videowire::geometry::checkedAdd(envelope.retainedBytes,sceneBytes-sizeof(Scene),envelope.retainedBytes)
        || !videowire::geometry::checkedAdd(peakReplay,sizeof(Scene),envelope.scratchBytes)
        || !videowire::geometry::checkedAdd(envelope.scratchBytes,bindingsBytes,envelope.scratchBytes)
        || !videowire::geometry::checkedAdd(envelope.scratchBytes,waterScratch,envelope.scratchBytes)
        || !videowire::geometry::checkedAdd(envelope.scratchBytes,sizeof(waterChildren),envelope.scratchBytes)
        || !videowire::geometry::checkedAdd(envelope.scratchBytes,sizeof(videowire::geometry::RuntimeFieldEvaluation),envelope.scratchBytes)
        || !videowire::geometry::checkedAdd(envelope.retainedBytes,envelope.waterHeightBytes,envelope.retainedBytes)
        || !add(envelope.waterChildren,sizeof(WaterSample),envelope.retainedBytes)) {
      error="Timeline family incremental evaluator storage overflow"; return {};
    }
    // Runtime exact re-admission runs while this binding copy is alive, before
    // result allocation. Its accepted attempt is NOT counted as new replay work.
    // During evaluation, each child is sequential; verification and append are
    // the two concurrently retained replay maps. Escaped results stay charged.
    const auto retained=ledger_ ? ledger_->retainedBytes.load(std::memory_order_relaxed) : 0;
    const auto maximum=std::min(limits.maxAdditionalBytes,
        family->children().front().plan().receipt().effectiveLimits.maxBufferBytes);
    if (!videowire::geometry::checkedAdd(retained,envelope.retainedBytes,envelope.concurrentBytes)
        || !videowire::geometry::checkedAdd(envelope.concurrentBytes,envelope.scratchBytes,envelope.concurrentBytes)
        || !videowire::geometry::checkedAdd(envelope.concurrentBytes,sizeof(Ledger),envelope.concurrentBytes)
        || envelope.concurrentBytes>maximum) {
      error="Timeline family evaluator exceeds its incremental retained/scratch storage budget"; return {};
    }
    try {
      std::vector<videowire::geometry::TimelineSceneBinding> bindings;
      bindings.reserve(family->children().size());
      for (const auto& child : family->children()) bindings.push_back(child.binding());
      const auto& admissionLimits=family->children().front().plan().receipt().effectiveLimits;
      const auto current=owner.use==PlanUse::preview
          ? runtime_.admitTimelineScenePreview(owner,*authored,bindings,admissionLimits,error)
          : runtime_.admitTimelineSceneExport(owner,*authored,bindings,admissionLimits,error);
      if (!current || current!=family) {
        if (error.empty()) error="Timeline family evaluator did not recover its exact active runtime admission";
        return {};
      }
      if (!ledger_) ledger_=std::make_shared<Ledger>();
      auto result=std::make_shared<OwnedScene>();
      result->family=family; result->reservation.ledger=ledger_;
      result->reservation.bytes=envelope.retainedBytes;
      result->sourceSeconds=sourceSeconds; result->projectSeconds=projectSeconds;
      ledger_->retainedBytes.fetch_add(result->reservation.bytes,std::memory_order_relaxed);
      result->scene=family->scene();
      result->canonicalFrame=envelope.waterChildren ? canonicalFrame : nullptr;
      result->waterSamples.reserve(envelope.waterChildren);
      childIndex=0;
      for (const auto& child : family->children()) {
        const auto& water=waterChildren[childIndex++];
        videowire::geometry::RuntimeFieldEvaluation evaluation;
        bool waterIssued=false;
        if (water.operation) {
          // Charge the admitted attempt before the first validation/replay.
          // Failed source predicates and later siblings never refund work.
          for (std::size_t i=1;i<water.recipe.work.size();++i)
            attemptedWaterWork_[i]+=water.recipe.work[i];
          auto released=videorender::waterSurfaceFrameEvaluation(waterHistory,projectSeconds,sourceSeconds,canonicalFrame);
          evaluation.timelineSeconds=sourceSeconds;
          evaluation.waterHeightAt=[&,released](const auto& operation,const auto& rest,const auto& grid,
              double requested,auto& heights,std::string& diagnostic) {
            if (waterIssued || !sameWaterOperation(operation,*water.operation) || !equal(rest,*water.rest)
                || grid.columns!=water.grid.columns || grid.rows!=water.grid.rows
                || grid.spacingX!=water.grid.spacingX || grid.spacingY!=water.grid.spacingY
                || requested!=sourceSeconds || !(child.owner()==owner)
                || result->family!=family || result->canonicalFrame!=canonicalFrame) {
              diagnostic="Timeline family Water callback does not match its exact child/plan/rest/time/frame"; return false;
            }
            std::vector<double> produced;
            if (!released.waterHeightAt(operation,rest,grid,requested,produced,diagnostic)) return false;
            // Issued only after the actual released canonical/source/grid/
            // settings/replay guards succeed. Borrowed history is not retained.
            result->waterSamples.push_back({&child,water.operation,water.rest,grid,sourceSeconds,projectSeconds,std::move(produced)});
            waterIssued=true;
            heights=result->waterSamples.back().heights; diagnostic.clear(); return true;
          };
        }
        // Each replacement starts from its own exclusive authored range. Its
        // released local Water memo can never be lent to another child.
        if (!videowire::geometry::replaceTimelineSceneMesh(child.binding(),child.plan().value(),
                sourceSeconds,result->scene,error,water.operation ? &evaluation : nullptr)) return {};
        if (water.operation && !waterIssued) {
          error="Timeline family Water replacement did not issue its authenticated child sample"; return {};
        }
      }
      return std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>(result,&result->scene);
    } catch (const std::bad_alloc&) {
      error="Timeline family evaluation allocation failed before atomic scene publication"; return {};
    }
  }
  struct CoupledResult final {
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::shared_ptr<const TimelineCoupledWaterReplay> replay;
  };

  // Separate privately paired route. No supplied frame or body receipt enters
  // this issuer; complete all-child schedules precede the single solver call.
  CoupledResult evaluateCoupled(const PlanOwnerIdentity& owner,
      const std::shared_ptr<const AdmittedTimelineScenePlans>& family,
      const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored,
      const videorender::ParticleParams& parameters,double sourceSeconds,double bodySeconds,double projectSeconds,
      float aspect,const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
      const timelineevaluation::Limits& limits,const coupledwaterbudget::Limits& coupledLimits,
      std::string& error,timelineevaluation::Envelope* receipt=nullptr,coupledwaterbudget::Recipe* coupledReceipt=nullptr) {
    using namespace timelineevaluation;
    using namespace videowire::geometry;
    error.clear(); if (receipt) *receipt={}; if (coupledReceipt) *coupledReceipt={};
    if (!family || family.use_count()==0 || !authored || !owner.valid() || !(owner==family->owner()) || !parameters.history
        || !parameters.history->score || !frame || !parameters.geometryBinding || !parameters.geometryBinding->mesh
        || !std::isfinite(sourceSeconds) || sourceSeconds<0 || sourceSeconds>10 || sourceSeconds!=bodySeconds
        || sourceSeconds!=projectSeconds || parameters.historyProjectSeconds!=projectSeconds
        || !std::isfinite(aspect) || aspect<=0 || family->children().size()<2
        || family->children().size()>Scene::kMaxObjects) {
      error="Coupled family requires exact admitted ownership, one Water child, authored siblings and equal finite clocks"; return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto caller=videorender::particleBodyRenderThreadCaller(error);
    if (!caller || !caller->onCallerThread()) return {};
    // Fixed prospective metadata envelope precedes all child schedule scans.
    if (!caller->charge(sizeof(Scene)+Scene::kMaxObjects*128*128*8)) {
      error="Coupled family permanent schedule census admission is short"; return {};
    }
    Envelope e; std::size_t sceneBytes=0,peakReplay=0,bindingsBytes=0,familyWork=0;
    WaterChild water; std::size_t index=0,waterIndex=family->children().size(),waterCount=0;
    e.preflightChildVisits=family->children().size(); e.bindingCopyRecords=family->children().size();
    if (!sceneSchedule(family->scene(),family->children().size(),e,sceneBytes,error)) return {};
    for (const auto& child:family->children()) {
      WaterChild candidate;
      if (!childSchedule(child,e,peakReplay,error,true) || !waterChild(child,candidate,e,error)) return {};
      if (candidate.operation) {
        ++waterCount;
        if (!(candidate.operation->field.values[4]>0) || waterCount!=1) {
          error="Coupled family requires exactly one positive-mass Water child and no separate Water trajectory"; return {};
        }
        water=candidate; waterIndex=index;
      }
      if (!addArrayBytes(1,sizeof(TimelineSceneBinding),bindingsBytes)
          || !checkedAdd(bindingsBytes,child.binding().loweredPlan.capacity(),bindingsBytes)
          || !checkedAdd(bindingsBytes,1,bindingsBytes)
          || !checkedAdd(e.bindingCopyBytes,child.binding().loweredPlan.size(),e.bindingCopyBytes)) return {};
      ++index;
    }
    if (waterCount!=1 || waterIndex>=family->children().size()) {
      error="Coupled family has no unique positive-mass Water child"; return {};
    }
    const auto& child=family->children()[waterIndex];
    // Fresh coupled callback, its released memo, and the exact pair sample have
    // separate rest/identity/copy visits from the generic geometry schedule.
    if (!add(water.rest->positions.size(),2,e.waterRestPositionVisits)
        || !add(water.rest->positions.size(),2,e.waterMemoCopyElements)
        || !add(water.rest->indices.size(),1,e.waterMemoCopyElements)
        || !add(water.rest->positions.size()+water.rest->indices.size(),2,e.waterIdentityVisits)
        || !add(water.rest->positions.size(),4,e.waterHeightCopies)
        || !add(water.rest->positions.size(),4,e.waterHeightValidationVisits)) return {};
    if (e.waterIdentityVisits>limits.maxWaterIdentityVisits || e.waterRestPositionVisits>limits.maxWaterRestPositionVisits
        || e.waterMemoCopyElements>limits.maxWaterMemoCopyElements || e.waterColumnDiscoveryVisits>limits.maxWaterColumnDiscoveryVisits
        || !scheduleWork(e,familyWork)) {
      error="Coupled family complete prospective comparison/geometry schedule is short"; return {};
    }
    // Reserve source census work before nested note reads. It is permanent even
    // when a later sibling, recipe, capacity or current-admission check refuses.
    std::size_t notes=0,censusWork=0;
    if (!checkedMul(parameters.history->score->notes.size(),2,notes) || !checkedAdd(familyWork,notes,censusWork)) return {};
    const auto census=static_cast<std::size_t>(videorender::watersourcebudget::Work::censusNotes);
    if (!caller->charge(censusWork) || !videorender::watersourcebudget::checkedAdd(
        coupledSourceAttempts_[census],notes,coupledSourceAttempts_[census])
        || coupledSourceAttempts_[census]>coupledLimits.source.work[census]) {
      error="Coupled family permanent prospective source census allowance is short"; return {};
    }
    auto remaining=coupledLimits;
    for (std::size_t i=1;i<remaining.source.work.size();++i) {
      if (coupledSourceAttempts_[i]>remaining.source.work[i]) { error="Coupled family source epoch is exhausted"; return {}; }
      remaining.source.work[i]-=coupledSourceAttempts_[i];
    }
    for (std::size_t i=0;i<remaining.work.size();++i) {
      if (coupledAttempts_[i]>remaining.work[i]) { error="Coupled family work epoch is exhausted"; return {}; }
      remaining.work[i]-=coupledAttempts_[i];
    }
    const auto& descriptor=child.plan().value().descriptor();
    const auto& mesh=*parameters.geometryBinding->mesh;
    if (!std::holds_alternative<GeometryData>(mesh.data) || !mesh.attributes.empty()
        || !mesh.spectrumFields.empty() || !mesh.scoreFields.empty() || mesh.operations.size()!=4) {
      error="Coupled family actual source is outside its complete four-operation domain"; return {};
    }
    std::size_t descriptorBytes=0,actualDescriptorBytes=0;
    if (!coupledwaterbudget::descriptorBytes(descriptor,descriptorBytes)
        || !coupledwaterbudget::descriptorBytes(mesh,actualDescriptorBytes)) return {};
    descriptorBytes=std::max(descriptorBytes,actualDescriptorBytes);
    Envelope geometryEnvelope; std::size_t geometryScratch=0,geometryWork=0;
    if (!childSchedule(child,geometryEnvelope,geometryScratch,error,true) || !scheduleWork(geometryEnvelope,geometryWork)) return {};
    // Reserve the later exact sphere append's bounded source/memo/output and
    // inspector envelopes. This prerequisite publishes the family and pair;
    // production append/native association is still a separate owner handoff.
    const auto& anchor=family->scene();
    if (parameters.count<1 || parameters.count>63 || std::size_t(parameters.count)>Scene::kMaxObjects-anchor.objectCount
        || anchor.materialCount>=Scene::kMaxMaterials || anchor.vertexCount>Scene::kMaxVertices-6
        || anchor.indexCount>Scene::kMaxIndices-24) {
      error="Coupled family sphere append exceeds the current authored scene capacities"; return {};
    }
    const auto* ancestor=&child.object();
    for (std::size_t depth=0;ancestor && depth<anchor.objectCount;++depth) {
      const auto& scale=ancestor->transform.scale;
      if (scale.x<=0 || scale.x!=scale.y || scale.x!=scale.z) {
        error="Coupled family sphere append requires positive uniform scale through its exact hierarchy"; return {};
      }
      ancestor=ancestor->parent.isValid() ? HarmonicMIDI::grid::visual3d_detail::findById(
          anchor.objects,anchor.objectCount,ancestor->parent):nullptr;
    }
    Envelope appendEnvelope; std::size_t appendSceneBytes=0,appendWork=0;
    if (!sceneSchedule(anchor,2,appendEnvelope,appendSceneBytes,error) || !scheduleWork(appendEnvelope,appendWork)) return {};
    const auto finalObjects=anchor.objectCount+std::size_t(parameters.count);
    // Each released hierarchy pass has at most 2 O^3 + O^2 lookup visits;
    // 400 O^2 covers its matrix terms, postchecks and duplicate-record scans.
    // Twelve passes conservatively include append validation and inspection.
    const auto finalVisits=2*finalObjects*finalObjects*finalObjects+400*finalObjects*finalObjects
        +finalObjects*(anchor.indexCount+24+anchor.materialCount+1+finalObjects+64)+sizeof(Scene);
    if (!add(finalVisits,12,appendWork)
        || !add(std::size_t(parameters.count),Scene::kMaxObjects*Scene::kMaxObjects,appendWork)
        || !checkedAdd(appendWork,6+24+water.rest->positions.size()+water.rest->indices.size(),appendWork)) return {};
    coupledwaterbudget::Recipe recipe;
    if (!coupledwaterbudget::complete(parameters,*water.operation,*water.rest,water.grid,frame,projectSeconds,
        remaining,familyWork,geometryWork,geometryScratch,descriptorBytes,appendWork,appendSceneBytes,recipe,error)) return {};
    // Comparison and publication work is charged before source equality, copying
    // descriptors or scenes, current admission and the actual solver call.
    for (std::size_t i=1;i<recipe.source.work.size();++i) coupledSourceAttempts_[i]+=recipe.source.work[i];
    for (std::size_t i=0;i<recipe.work.size();++i) coupledAttempts_[i]+=recipe.work[i];
    struct CoupledOwnedScene final {
      std::shared_ptr<const void> reservation;
      std::shared_ptr<const TimelineCoupledWaterReplay> pair;
      Scene scene;
    };
    e.retainedBytes=sizeof(CoupledOwnedScene);
    if (!checkedAdd(recipe.comparisonBytes,bindingsBytes,recipe.comparisonBytes)
        || !checkedAdd(recipe.comparisonBytes,sizeof(Scene),recipe.comparisonBytes)
        || !checkedAdd(e.retainedBytes,sceneBytes-sizeof(Scene),e.retainedBytes)
        || !checkedAdd(e.retainedBytes,sizeof(TimelineCoupledWaterReplay)+sizeof(videorender::ParticleBodyCallerLease),e.retainedBytes)
        || !checkedAdd(e.retainedBytes,recipe.retainedBytes,e.retainedBytes)
        || !checkedAdd(e.retainedBytes,sizeof(Reservation),e.retainedBytes)
        || !checkedAdd(e.retainedBytes,appendSceneBytes,e.retainedBytes)
        || !checkedAdd(peakReplay,recipe.scratchBytes,e.scratchBytes)
        || !checkedAdd(e.scratchBytes,recipe.comparisonBytes,e.scratchBytes)
        || !checkedAdd(e.scratchBytes,bindingsBytes,e.scratchBytes)
        || !add(appendSceneBytes,2,e.scratchBytes)
        || !checkedAdd(e.scratchBytes,2*sizeof(Scene)+2*sizeof(videorender::ParticleBodyCallerLease),e.scratchBytes)
        || !checkedAdd(e.scratchBytes,sizeof(coupledSourceAttempts_)+sizeof(coupledAttempts_),e.scratchBytes)) return {};
    // sceneSchedule's old mass-zero receipt uses string sizes. This route
    // admits actual application capacities, inactive identities and each NUL.
    for (const auto& light:family->scene().lights) {
      const auto& asset=light.environmentImage.asset;
      std::size_t extra=0;
      if (!checkedAdd(asset.id.capacity()-asset.id.size(),asset.contentSha256.capacity()-asset.contentSha256.size(),extra)
          || !checkedAdd(extra,2,extra) || !checkedAdd(e.retainedBytes,extra,e.retainedBytes)
          || !checkedAdd(e.retainedBytes,extra,e.retainedBytes)
          || !checkedAdd(sceneBytes,extra,sceneBytes) || !add(extra,5,e.scratchBytes)) return {};
    }
    std::size_t textureExtra=0;
    if (!checkedMul(family->scene().textureTexels.capacity()-family->scene().textureTexels.size(),
        sizeof(HarmonicMIDI::grid::SceneTexelRgba8),textureExtra)) return {};
    if (!checkedAdd(e.retainedBytes,textureExtra,e.retainedBytes)
        || !checkedAdd(e.retainedBytes,textureExtra,e.retainedBytes)
        || !checkedAdd(sceneBytes,textureExtra,sceneBytes) || !add(textureExtra,5,e.scratchBytes)) return {};
    const auto retained=ledger_ ? ledger_->retainedBytes.load(std::memory_order_relaxed):0;
    const auto maximum=std::min(limits.maxAdditionalBytes,child.plan().receipt().effectiveLimits.maxBufferBytes);
    if (!checkedAdd(retained,e.retainedBytes,e.concurrentBytes) || !checkedAdd(e.concurrentBytes,e.scratchBytes,e.concurrentBytes)
        || !checkedAdd(e.concurrentBytes,sizeof(Ledger),e.concurrentBytes)) return {};
    if (recipe.comparisonBytes>coupledLimits.maxComparisonBytes || e.retainedBytes>coupledLimits.maxRetainedBytes
        || e.scratchBytes>coupledLimits.maxScratchBytes) {
      error="Coupled family full publication/comparison envelope is short"; return {};
    }
    if (receipt) *receipt=e;
    if (coupledReceipt) *coupledReceipt=recipe;
    // One simultaneous caller reservation covers all family/sampler scratch.
    // Its attempted work remains spent even if either owning storage cap refuses.
    const auto scratch=caller->reserve(e.scratchBytes,recipe.totalWork);
    if (!scratch || e.concurrentBytes>maximum) {
      error="Coupled family simultaneous source/replay/escaped-output storage admission is short"; return {};
    }
    const auto comparisonLease=caller->reserve(recipe.comparisonBytes,recipe.totalWork);
    if (!comparisonLease) { error="Coupled family complete simultaneous consumer admission is short"; return {}; }
    const auto pairLease=caller->reserve(e.retainedBytes,e.retainedBytes);
    if (!pairLease) { error="Coupled family retained pair/publication admission is short"; return {}; }
    try {
      // Independent source equality precedes replay. The four-operation plan
      // comparator replaces decoder work only within this restricted domain.
      if (!coupledwaterdetail::sameDescriptor(*parameters.geometryBinding->mesh,descriptor)
          || !timelinefamily::sameScene(*authored,family->scene())) {
        error="Coupled family request differs from its complete admitted mesh/authored source"; return {};
      }
      std::vector<TimelineSceneBinding> bindings; bindings.reserve(family->children().size());
      for (const auto& sibling:family->children()) bindings.push_back(sibling.binding());
      const auto& admissionLimits=child.plan().receipt().effectiveLimits;
      const auto current=owner.use==PlanUse::preview
          ? runtime_.admitTimelineScenePreview(owner,*authored,bindings,admissionLimits,error)
          : runtime_.admitTimelineSceneExport(owner,*authored,bindings,admissionLimits,error);
      if (current!=family || current.owner_before(family) || family.owner_before(current)) { if (error.empty()) error="Coupled family has no exact current admission"; return {}; }
      if (!ledger_) ledger_=std::make_shared<Ledger>();
      auto reservation=std::make_shared<Reservation>(); reservation->ledger=ledger_; reservation->bytes=e.retainedBytes;
      ledger_->retainedBytes.fetch_add(e.retainedBytes,std::memory_order_relaxed);
      auto pair=TimelineCoupledWaterReplay::issue(pairLease,reservation,owner,family,waterIndex,parameters,aspect,frame,recipe,error);
      if (!pair || !pair->matchesReserved(runtime_,owner,family,waterIndex,authored,parameters,
          sourceSeconds,bodySeconds,projectSeconds,aspect,frame,comparisonLease,error)) return {};
      auto result=std::make_shared<CoupledOwnedScene>(); result->reservation=reservation; result->pair=pair; result->scene=family->scene();
      std::size_t actualSceneBytes=sizeof(Scene);
      if (!addArrayBytes(result->scene.textureTexels.capacity(),sizeof(HarmonicMIDI::grid::SceneTexelRgba8),actualSceneBytes)) return {};
      for (const auto& light:result->scene.lights)
        if (!checkedAdd(actualSceneBytes,light.environmentImage.asset.id.capacity(),actualSceneBytes)
            || !checkedAdd(actualSceneBytes,light.environmentImage.asset.contentSha256.capacity(),actualSceneBytes)
            || !checkedAdd(actualSceneBytes,2,actualSceneBytes)) return {};
      if (actualSceneBytes>sceneBytes) {
        error="Coupled family actual copied scene capacities exceed their prospective envelope"; return {};
      }
      index=0;
      for (const auto& sibling:family->children()) {
        RuntimeFieldEvaluation evaluation; evaluation.timelineSeconds=sourceSeconds;
        if (index==waterIndex) evaluation.waterHeightAt=[&](const auto& operation,const auto& rest,const auto& grid,
            double seconds,auto& heights,auto& diagnostic) {
          return pair->water_.sample(operation,rest,grid,seconds,heights,diagnostic);
        };
        if (!replaceTimelineSceneMesh(sibling.binding(),sibling.plan().value(),sourceSeconds,result->scene,error,
            index==waterIndex ? &evaluation:nullptr)) return {};
        ++index;
      }
      error.clear(); return {{result,&result->scene},std::move(pair)};
    } catch (const std::bad_alloc&) {
      error="Coupled family allocation refused before atomic pair/scene publication"; return {};
    }
  }

private:
  GeometryCorePlanRuntime& runtime_;
  std::mutex mutex_;
  std::shared_ptr<Ledger> ledger_;
  decltype(videorender::watersourcebudget::Recipe{}.work) attemptedWaterWork_{};
  decltype(videorender::watersourcebudget::Recipe{}.work) coupledSourceAttempts_{};
  decltype(coupledwaterbudget::Recipe{}.work) coupledAttempts_{};
};
} // namespace videohelper::geometry
