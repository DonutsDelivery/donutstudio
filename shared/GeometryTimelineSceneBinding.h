#pragma once

#include "GeometryCoreScene.h"
#include "GeometryCoreTransport.h"

namespace videowire::geometry {
struct TimelineSceneBinding final {
  HarmonicMIDI::grid::SceneObjectId object;
  std::string loweredPlan;
};

// Check the consumer subset before Geometry Core admission materializes any
// intermediate. The scene grows for any generator. Shade Mesh is the one operation
// that can add vertices, and it replays on each frame's mesh. Smooth and Flat split
// the same vertices every frame, and By Angle on a moving mesh splits every corner.
inline bool admitTimelineSceneOperation(const OperationRecord& op, std::string& error) {
  if (op.code == OperationCode::grid) {
    std::uint32_t columns = 0, rows = 0;
    std::size_t vertices = 0, cells = 0, indices = 0;
    if (exactIntegerParameter(op.parameters[0],2,columns) && exactIntegerParameter(op.parameters[1],2,rows)
        && checkedMul(columns,rows,vertices) && checkedMul(columns-1,rows-1,cells)
        && checkedMul(cells,6,indices)) return true;
  } else if (op.code >= OperationCode::plane && op.code <= OperationCode::cylinder) {
    GeometryData mesh;
    return materializeMeshGenerator(op,mesh,kMaximumVertices,kMaximumIndices,error);
  } else if (op.code == OperationCode::retainedMesh) {
    if (op.retainedMesh && op.retainedMesh->importedAnimation.empty()) return true;
  } else if (op.code == OperationCode::constantVectorField) {
    if (op.parameters[3] >= 1) return true;
  } else if (op.code == OperationCode::fieldWaterHeight) {
    if (op.field.domain == Domain::vertex && op.field.valueType == ValueType::floatValue) return true;
  } else if (op.code == OperationCode::fieldTimeline || op.code == OperationCode::fieldScoreSample
      || (op.code >= OperationCode::fieldConstant && op.code <= OperationCode::fieldSelect)) {
    if (op.field.domain == Domain::vertex
        && (op.code != OperationCode::fieldConstant
            || op.field.values[4] >= 1)) return true;
  } else if ((op.code >= OperationCode::positionField && op.code <= OperationCode::stableIdentityField)
      || op.code == OperationCode::vectorDisplacement || op.code == OperationCode::transformTRS
      || op.code == OperationCode::assignMaterial || op.code == OperationCode::transform
      || op.code == OperationCode::shadeMesh || op.code == OperationCode::audioDeformer
      || op.code == OperationCode::fieldAudioSpectrum || op.code == OperationCode::evaluateField
      || (op.code >= OperationCode::boundsMinimumField && op.code <= OperationCode::boundsCenterField)
      // Curve Animate keeps every curve point slot, so a tube or ribbon keeps its topology each frame.
      || op.code == OperationCode::curveCircle || op.code == OperationCode::curveLine
      || op.code == OperationCode::curveToMesh || op.code == OperationCode::curveToTube
      || (op.code >= OperationCode::curveResample && op.code <= OperationCode::curveWave
          && op.secondaryInputStableId == 0)
      || op.code == OperationCode::curveAnimate) return true;
  error = "Timeline Scene3D supports mesh generators, curves, vertex fields, score samples, audio spectrum and topology-stable deformation only";
  return false;
}

inline std::optional<AdmittedValue> decodeTimelineScenePlan(
    const TimelineSceneBinding& binding, const ResourceLimits& remaining, std::string& error,
    std::size_t* retainedFieldElements = nullptr) {
  if (!binding.object.isValid() || binding.loweredPlan.empty()) {
    error = "Timeline Scene3D requires one exact object and a Geometry Core plan";
    return std::nullopt;
  }
  if (!validateLimits(remaining,error) || remaining.maxOperations == 0
      || binding.loweredPlan.size()/2 > remaining.maxBufferBytes) {
    error = "Timeline Scene3D has exhausted its remaining aggregate budget";
    return std::nullopt;
  }
  const auto& limits = remaining;
  const auto bytes = decodeLoweredPlanText(binding.loweredPlan, error);
  const auto plan = bytes ? decodeLoweredRuntimePlan(*bytes, error) : std::nullopt;
  if (!plan || !plan->surfaceMaterial.empty()
      || (plan->contract.carrier != CarrierKind::geometry3D && plan->contract.carrier != CarrierKind::instances3D))
    return std::nullopt;
  // A Spectrum Instancer replays its whole operation list through validateOperationPlan;
  // only the mesh path is held to the per-frame mesh operation list.
  const bool instancePlan = plan->contract.carrier == CarrierKind::instances3D;
  std::size_t fieldElements = 0;
  struct OutputCount final { StableId id=0; std::size_t elements=0; bool field=false; };
  std::vector<OutputCount> outputs;
  const auto countFor=[&](StableId id,bool field) {
    for (const auto& output : outputs)
      if (output.id==id && output.field==field) return output.elements;
    return std::size_t{0};
  };
  auto value = decodeRuntimeValue(plan->runtimeValue, plan->contract, limits, {}, error,
      [&](const OperationRecord& op, std::string& diagnostic) {
        if (outputs.size()>=limits.maxOperations) { diagnostic = "Timeline Scene3D exceeds its remaining retained operations"; return false; }
        std::size_t count=0,indices=0;
        bool field=false;
        if (op.code==OperationCode::grid) {
          std::uint32_t columns=0,rows=0; std::size_t cells=0;
          if (!exactIntegerParameter(op.parameters[0],2,columns) || !exactIntegerParameter(op.parameters[1],2,rows)
              || !checkedMul(columns,rows,count) || !checkedMul(columns-1,rows-1,cells)
              || !checkedMul(cells,6,indices)) return false;
        } else if (op.code>=OperationCode::plane && op.code<=OperationCode::cylinder) {
          GeometryData mesh;
          if (!materializeMeshGenerator(op,mesh,limits.maxVertices,limits.maxIndices,diagnostic)) return false;
          count=mesh.positions.size(); indices=mesh.indices.size();
        } else if (op.code==OperationCode::retainedMesh && op.retainedMesh) {
          count=op.retainedMesh->geometry.positions.size(); indices=op.retainedMesh->geometry.indices.size();
        } else if (op.code==OperationCode::constantVectorField) {
          if (!finite(op.parameters[3]) || op.parameters[3]<1) return false;
          count=static_cast<std::size_t>(op.parameters[3]); field=true;
        } else if (op.code==OperationCode::fieldConstant) {
          if (!finite(op.field.values[4]) || op.field.values[4]<1) return false;
          count=static_cast<std::size_t>(op.field.values[4]); field=true;
        } else if (isBoundsFieldQuery(op.code)) {
          count=1; field=true;
        } else if (op.code==OperationCode::fieldWaterHeight || op.code==OperationCode::fieldAudioSpectrum
            || (op.code>=OperationCode::positionField && op.code<=OperationCode::stableIdentityField)) {
          count=countFor(op.inputStableId,false); field=true;
        } else if (op.code==OperationCode::fieldTimeline || op.code==OperationCode::fieldScoreSample
            || (op.code>=OperationCode::fieldConstant && op.code<=OperationCode::fieldSelect)) {
          count=std::max({countFor(op.inputStableId,true),countFor(op.secondaryInputStableId,true),
              countFor(op.field.auxiliaryInputStableId,true)}); field=true;
        } else count=countFor(op.inputStableId,false);
        if ((field && (count>limits.maxFieldElements
                || !checkedAdd(fieldElements,count,fieldElements) || fieldElements>limits.maxFieldElements))
            || (!field && (count>limits.maxVertices || indices>limits.maxIndices))) {
          diagnostic="Timeline Scene3D operation exceeds its remaining mesh/field capacity"; return false;
        }
        if (!instancePlan && !admitTimelineSceneOperation(op,diagnostic)) return false;
        outputs.push_back({op.outputStableId,count,field}); return true;
      });
  if (!value) return std::nullopt;
  if (instancePlan ? !supportsTimelineSceneInstances(value->descriptor())
                   : !supportsTimelineSceneGeometry(value->descriptor())) {
    error = "Composed Scene3D supports timeline, score-sampled and audio spectrum Geometry3D and Spectrum Instancers; Score Field bindings, animated imports and other reactive instances are unavailable";
    return std::nullopt;
  }
  if (retainedFieldElements) *retainedFieldElements=fieldElements;
  return value;
}

inline std::optional<AdmittedValue> decodeTimelineScenePlan(
    const TimelineSceneBinding& binding, std::string& error) {
  return decodeTimelineScenePlan(binding, ResourceLimits{}, error);
}

inline bool sameSceneVertex(const HarmonicMIDI::grid::SceneVertex& a,
                            const HarmonicMIDI::grid::SceneVertex& b) noexcept {
  return a.position.x == b.position.x && a.position.y == b.position.y && a.position.z == b.position.z
      && a.normal.x == b.normal.x && a.normal.y == b.normal.y && a.normal.z == b.normal.z
      && a.uv.x == b.uv.x && a.uv.y == b.uv.y
      && a.color == b.color && a.tangent == b.tangent && a.roughness == b.roughness;
}

// A timeline mesh owns exclusive vertex/index ranges in one draw. Validate its
// admission-time mesh against the snapshot before evaluating another frame.
// A Spectrum Instancer owns its parent object and one child per admitted instance, each
// drawing the parent's mesh range.
inline bool validTimelineSceneInstances(const TimelineSceneBinding& binding,
    const HarmonicMIDI::grid::Visual3DScene& scene, const AdmittedValue& value, std::string& error) {
  using namespace HarmonicMIDI::grid;
  const auto& descriptor = value.descriptor();
  const auto* instances = std::get_if<InstancesData>(&descriptor.data);
  if (!validateVisual3DScene(scene).valid() || !supportsTimelineSceneInstances(descriptor) || !instances
      || instances->instances.empty()
      || !visual3d_detail::findById(scene.objects, scene.objectCount, binding.object)) {
    if (error.empty()) error = "Timeline Spectrum Instancer does not own its retained parent object";
    return false;
  }
  for (const auto& instance : instances->instances) {
    const auto* child = visual3d_detail::findById(scene.objects, scene.objectCount,
                                                  sceneInstanceId(binding.object, instance.stableId));
    if (!child || child->parent != binding.object) {
      error = "Timeline Spectrum Instancer lost one of its admitted instance objects"; return false;
    }
  }
  error.clear();
  return true;
}

// Rewrites each instance object's transform from this frame's replay and spectrum, with
// the same per-instance translation and scale response as the native Spectrum Instancer.
inline bool replaceTimelineSceneInstances(const TimelineSceneBinding& binding,
    const AdmittedValue& value, double seconds, HarmonicMIDI::grid::Visual3DScene& scene,
    std::string& error, const RuntimeFieldEvaluation* frameEvaluation, bool bindingVerified) {
  using namespace HarmonicMIDI::grid;
  const auto& descriptor = value.descriptor();
  if (!std::isfinite(seconds) || (!bindingVerified && !validTimelineSceneInstances(binding, scene, value, error))) {
    if (error.empty()) error = "Timeline Spectrum Instancer does not own its retained instance objects";
    return false;
  }
  if (!frameEvaluation || !frameEvaluation->spectrumAt || !frameEvaluation->spectrumHistoryAt) {
    error = "Timeline Spectrum Instancer requires the frame's spectrum readers"; return false;
  }
  RuntimeFieldEvaluation evaluation = *frameEvaluation;
  evaluation.timelineSeconds = seconds;
  MaterializedInstancePlan retained;
  if (!validateOperationPlan(descriptor, {}, &retained, error, nullptr, &evaluation, evaluation.audioDeformAt))
    return false;
  auto placed = retained.instances.instances;
  const auto& admitted = std::get<InstancesData>(descriptor.data).instances;
  if (placed.size() != admitted.size() || !std::equal(placed.begin(), placed.end(), admitted.begin(),
        [](const auto& a, const auto& b) { return a.stableId == b.stableId; })) {
    error = "Timeline Spectrum Instancer changed its admitted instances"; return false;
  }
  for (const auto& spectral : descriptor.spectrumFields) {
    if (spectral.coordinates.size() != placed.size()) {
      error = "Spectrum Instancer has no matching instance set"; return false;
    }
    std::vector<float> amounts(placed.size());
    if (spectral.history.enabled) {
      std::vector<spectrum::Bands> rows;
      if (!evaluation.spectrumHistoryAt(spectral, rows, error)) return false;
      if (rows.size() != spectral.history.depth || spectral.history.coordinates.size() != amounts.size()) {
        error = "Spectral History Spectrum Instancer rows disagree with its instances"; return false;
      }
      for (std::size_t i = 0; i < amounts.size(); ++i) amounts[i] = spectrum::sampleHistory(spectral, rows, i);
    } else {
      spectrum::Bands followed{};
      if (!evaluation.spectrumAt(spectral, followed, error)) return false;
      for (std::size_t i = 0; i < amounts.size(); ++i)
        amounts[i] = spectrum::sample(spectral, followed, spectral.coordinates[i]);
    }
    for (std::size_t i = 0; i < placed.size(); ++i) {
      auto& transform = placed[i].transform;
      const auto amount = amounts[i];
      transform.translation.x += amount * spectral.direction[0];
      transform.translation.y += amount * spectral.direction[1];
      transform.translation.z += amount * spectral.direction[2];
      const auto scale = [&](std::size_t axis, float base) {
        return std::clamp(base * (1.0f + amount * spectral.scaleResponse[axis]), 0.000001f, 1000.0f);
      };
      transform.scale = {scale(0, transform.scale.x), scale(1, transform.scale.y), scale(2, transform.scale.z)};
    }
  }
  for (const auto& instance : placed) {
    if (!finite(instance.transform)) { error = "Spectrum Instancer produced a non-finite transform"; return false; }
    const auto id = sceneInstanceId(binding.object, instance.stableId);
    SceneObjectRecord* child = nullptr;
    for (std::size_t index = 0; index < scene.objectCount && !child; ++index)
      if (scene.objects[index].id == id) child = &scene.objects[index];
    if (!child || child->parent != binding.object) {
      error = "Timeline Spectrum Instancer lost one of its admitted instance objects"; return false;
    }
    const auto& t = instance.transform;
    child->transform.translation = {t.translation.x, t.translation.y, t.translation.z};
    child->transform.rotation = {t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};
    child->transform.scale = {t.scale.x, t.scale.y, t.scale.z};
  }
  if (!validateVisual3DScene(scene).valid()) { error = "Timeline Spectrum Instancer produced an invalid retained scene"; return false; }
  error.clear();
  return true;
}

namespace timeline_scene_detail {
// The binding's object with its own vertex and index ranges and its material, else null.
inline const HarmonicMIDI::grid::SceneObjectRecord* ownedObject(const TimelineSceneBinding& binding,
    const HarmonicMIDI::grid::Visual3DScene& scene, const AdmittedValue& value) {
  using namespace HarmonicMIDI::grid;
  if (!supportsTimelineSceneGeometry(value.descriptor())) return nullptr;
  const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount, binding.object);
  if (!object || object->vertexCount == 0 || object->indexCount == 0) return nullptr;
  const auto overlaps = [](std::uint32_t a, std::uint32_t n, std::uint32_t b, std::uint32_t m) {
    return n && m && a < b + m && b < a + n;
  };
  for (std::size_t i = 0; i < scene.objectCount; ++i) {
    const auto& other = scene.objects[i];
    if (other.id != object->id && (overlaps(object->firstVertex, object->vertexCount, other.firstVertex, other.vertexCount)
        || overlaps(object->firstIndex, object->indexCount, other.firstIndex, other.indexCount))) return nullptr;
  }
  if (!visual3d_detail::findById(scene.materials, scene.materialCount, object->material)) return nullptr;
  return object;
}
} // namespace timeline_scene_detail

// validTimelineSceneBinding for a scene whose object ranges fillTimelineSceneMesh wrote from
// this same value: the mesh is the plan's by construction, and the fill validated the scene,
// so only the object's ownership of its ranges is checked. Any other scene needs the full check.
inline bool validFilledTimelineSceneBinding(const TimelineSceneBinding& binding,
    const HarmonicMIDI::grid::Visual3DScene& scene, const AdmittedValue& value) {
  return value.descriptor().carrier == CarrierKind::geometry3D
      && timeline_scene_detail::ownedObject(binding, scene, value) != nullptr;
}

inline bool validTimelineSceneBinding(const TimelineSceneBinding& binding,
    const HarmonicMIDI::grid::Visual3DScene& scene, const AdmittedValue& value,
    std::string& error) {
  using namespace HarmonicMIDI::grid;
  if (value.descriptor().carrier == CarrierKind::instances3D)
    return validTimelineSceneInstances(binding, scene, value, error);
  if (!validateVisual3DScene(scene).valid()) return false;
  const auto* object = timeline_scene_detail::ownedObject(binding, scene, value);
  if (!object) return false;
  const auto* material = visual3d_detail::findById(scene.materials, scene.materialCount, object->material);
  auto baseline = std::make_shared<Visual3DScene>();
  if (!appendGeometryToScene(value, *object, *material, *baseline, error)) return false;
  if (baseline->vertexCount != object->vertexCount || baseline->indexCount != object->indexCount) return false;
  for (std::size_t i = 0; i < baseline->vertexCount; ++i)
    if (!sameSceneVertex(scene.vertices[object->firstVertex + i], baseline->vertices[i])) return false;
  for (std::size_t i = 0; i < baseline->indexCount; ++i)
    if (scene.indices[object->firstIndex + i] != baseline->indices[i]) return false;
  error.clear();
  return true;
}

// True when the binding's plan yields a Geometry3D mesh, so a scene wire may leave
// that object's vertex and index ranges out and the receiver rebuilds them.
inline bool rebuildsTimelineSceneMesh(const TimelineSceneBinding& binding) {
  std::string error;
  const auto bytes = decodeLoweredPlanText(binding.loweredPlan, error);
  const auto plan = bytes ? decodeLoweredRuntimePlan(*bytes, error) : std::nullopt;
  return binding.object.isValid() && plan && plan->contract.carrier == CarrierKind::geometry3D;
}

// Rebuilds the vertex and index ranges a scene wire left out for the binding's
// object: the mesh appended from the admitted plan, which is the same mesh
// validTimelineSceneBinding requires the producer's scene to hold.
inline std::optional<AdmittedValue> fillTimelineSceneMesh(const TimelineSceneBinding& binding,
    HarmonicMIDI::grid::Visual3DScene& scene, std::string& error) {
  using namespace HarmonicMIDI::grid;
  auto value = decodeTimelineScenePlan(binding, error);
  if (!value) return std::nullopt;
  const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount, binding.object);
  const auto* material = object ? visual3d_detail::findById(scene.materials, scene.materialCount, object->material) : nullptr;
  auto rebuilt = std::make_shared<Visual3DScene>();
  if (value->descriptor().carrier != CarrierKind::geometry3D || !material
      || !appendGeometryToScene(*value, *object, *material, *rebuilt, error)
      || rebuilt->vertexCount != object->vertexCount || rebuilt->indexCount != object->indexCount) {
    if (error.empty()) error = "Timeline Geometry3D plan does not rebuild its object's left-out mesh";
    return std::nullopt;
  }
  for (std::size_t i = 0; i < rebuilt->vertexCount; ++i) scene.vertices[object->firstVertex + i] = rebuilt->vertices[i];
  for (std::size_t i = 0; i < rebuilt->indexCount; ++i) scene.indices[object->firstIndex + i] = rebuilt->indices[i];
  // The wire decoder leaves this check to the rebuild, since the left-out ranges were zeroed.
  if (!validateVisual3DScene(scene).valid()) {
    error = "Timeline Geometry3D plan rebuilt its left-out mesh into an invalid scene";
    return std::nullopt;
  }
  error.clear();
  return value;
}

inline bool replaceTimelineSceneMesh(const TimelineSceneBinding& binding,
    const AdmittedValue& value, double seconds, HarmonicMIDI::grid::Visual3DScene& scene,
    std::string& error, const RuntimeFieldEvaluation* frameEvaluation = nullptr,
    bool bindingVerified = false) {
  using namespace HarmonicMIDI::grid;
  if (value.descriptor().carrier == CarrierKind::instances3D)
    return replaceTimelineSceneInstances(binding, value, seconds, scene, error, frameEvaluation, bindingVerified);
  // bindingVerified: this plan has already passed the checks and the identity replay
  // below on this same immutable scene, so a later frame only evaluates its new time.
  if (!std::isfinite(seconds) || (!bindingVerified && !validTimelineSceneBinding(binding, scene, value, error))) {
    if (error.empty()) error = "Timeline Geometry3D does not own the exact retained object mesh";
    return false;
  }
  const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount, binding.object);
  const auto* material = visual3d_detail::findById(scene.materials, scene.materialCount, object->material);
  RuntimeFieldEvaluation evaluation = frameEvaluation ? *frameEvaluation : RuntimeFieldEvaluation{};
  evaluation.timelineSeconds = seconds;
  // Mesh verification and scene append replay the same plan. Share only this
  // frame's exact water result, never derived solver state across frames.
  struct WaterFrameSample final {
    OperationRecord operation;
    GeometryData restMesh;
    double seconds = 0;
    std::vector<double> heights;
  };
  std::vector<WaterFrameSample> samples;
  if (evaluation.waterHeightAt) {
    const auto producer = evaluation.waterHeightAt;
    evaluation.waterHeightAt = [&, producer](const OperationRecord& op, const GeometryData& restMesh,
        const WaterSurfaceGrid& grid, double requestedSeconds, std::vector<double>& heights,
        std::string& diagnostic) {
      for (const auto& sample : samples) {
        const auto& saved = sample.operation;
        if (sample.seconds == requestedSeconds && saved.stableId == op.stableId && saved.code == op.code
            && saved.inputStableId == op.inputStableId && saved.outputStableId == op.outputStableId
            && saved.secondaryInputStableId == op.secondaryInputStableId && saved.parameters == op.parameters
            && equal(saved.transform,op.transform) && saved.field == op.field
            && saved.retainedMesh == op.retainedMesh && saved.sdfSurface == op.sdfSurface
            && equal(sample.restMesh,restMesh)) {
          heights = sample.heights; diagnostic.clear(); return true;
        }
      }
      std::vector<double> produced;
      if (!producer(op,restMesh,grid,requestedSeconds,produced,diagnostic)) return false;
      if (produced.size() != restMesh.positions.size()
          || !std::all_of(produced.begin(),produced.end(),[](double h) { return finite(h); })) {
        diagnostic = "Water surface replay changed vertex count or returned nonfinite heights"; return false;
      }
      samples.push_back({op,restMesh,requestedSeconds,produced});
      heights = std::move(produced); diagnostic.clear(); return true;
    };
  }
  if (!bindingVerified) {
    MaterializedInstancePlan replay;
    const auto* mesh = replayGeometryMesh(value.descriptor(), replay, error, &evaluation);
    if (!mesh || mesh->vertexIds != std::get<GeometryData>(value.descriptor().data).vertexIds) {
      if (error.empty()) error = "Timeline Geometry3D changed its admitted vertex identities";
      return false;
    }
  }
  auto evaluated = std::make_shared<Visual3DScene>();
  if (!appendGeometryToScene(value, *object, *material, *evaluated, error, nullptr, nullptr, false, &evaluation)) return false;
  if (evaluated->vertexCount != object->vertexCount || evaluated->indexCount != object->indexCount) {
    error = "Timeline Geometry3D changed its admitted topology"; return false;
  }
  for (std::size_t i = 0; i < evaluated->indexCount; ++i)
    if (evaluated->indices[i] != scene.indices[object->firstIndex + i]) {
      error = "Timeline Geometry3D changed its admitted indices"; return false;
    }
  for (std::size_t i = 0; i < evaluated->vertexCount; ++i)
    scene.vertices[object->firstVertex + i] = evaluated->vertices[i];
  if (!validateVisual3DScene(scene).valid()) { error = "Timeline Geometry3D produced an invalid retained scene"; return false; }
  return true;
}
} // namespace videowire::geometry
