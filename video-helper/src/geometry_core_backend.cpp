#include "geometry_core_backend.h"
#include "../../shared/GeometryAudioDeformer.h"
#include "../../shared/GeometryMaterialTable.h"
#include "geometry_score_field.h"
#include "fixture_scene_renderer.h"

#include <cmath>

namespace videohelper::geometry {
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

std::optional<NativeGeometryExecution>
executeNativeGeometry(arbitgpu::NativeFixtureSceneBackend &backend,
                      const RuntimeAdmission &admission, std::uint32_t width,
                      std::uint32_t height, std::string &error,
                      bool diagnosticInstanceIdentityColors,
                      arbitgpu::NativeFixtureSceneRuntimeInputs runtimeInputs,
                      SpectrumEvaluation* spectrumEvaluation,
                      const ImportedGeometryEvaluator& importedGeometry,
                      double timelineTimeSeconds) {
  using namespace HarmonicMIDI::grid;
  error.clear();
  if (!admission.plan || width == 0 || height == 0) {
    error = "Geometry Core native execution requires a plan and non-zero extent";
    return std::nullopt;
  }
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
          || source->geometry.positions.size()!=operation.retainedMesh->geometry.positions.size()) {
        error="Animated Geometry3D changed its admitted topology"; return std::nullopt;
      }
      animated->operations[index].retainedMesh=std::move(source);
    }
  if (animated) {
    videowire::geometry::ValueDescriptor replayed;
    if (!videowire::geometry::validateOperationPlan(*animated,{},nullptr,error,&replayed)) return std::nullopt;
    const auto& limits=admission.plan->receipt().effectiveLimits;
    videowire::geometry::PortContract contract;
    contract.carrier=replayed.carrier;
    if (replayed.carrier==videowire::geometry::CarrierKind::geometry3D) {
      contract.maxVertices=limits.maxVertices; contract.maxIndices=limits.maxIndices;
    } else if (replayed.carrier==videowire::geometry::CarrierKind::instances3D) {
      contract.maxInstances=limits.maxInstances;
    } else { error="Animated geometry render requires Geometry3D or Instances3D"; return std::nullopt; }
    contract.maxAttributes=limits.maxAttributes;
    if (!videowire::geometry::admitValue(replayed,
        videowire::geometry::withAttributeContract(contract,replayed),limits,{},error)) return std::nullopt;
    animated=std::move(replayed);
  }
  const auto &descriptor = animated ? *animated : authored;
  const videowire::geometry::GeometryData *mesh = nullptr;
  const videowire::geometry::InstancesData *instances = nullptr;
  const std::vector<videowire::geometry::AttributeData>* meshAttributes=&descriptor.attributes;
  videowire::geometry::GeometryData deformedMesh;
  videowire::geometry::MaterializedInstancePlan materialized;
  const bool runtimeFields=videowire::geometry::hasRuntimeFields(descriptor);
  videowire::geometry::RuntimeFieldEvaluation evaluation;
  evaluation.timelineSeconds=timelineTimeSeconds;
  evaluation.scoreAt=[&](const auto& operation,const auto& positions,auto& result,auto& diagnostic) {
    return scorefield::evaluateSampleField(operation,positions,runtimeInputs.canonicalBlockCFrame,result,diagnostic);
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
  const bool orderedAudio=std::any_of(descriptor.spectrumFields.begin(),descriptor.spectrumFields.end(),
      [](const auto& binding) { return binding.target==videowire::geometry::spectrum::Target::geometryDeformation; });
  std::function<bool(videowire::geometry::GeometryData&,const videowire::geometry::spectrum::Binding&,
                     std::string&)> deform;
  if (orderedAudio) {
    if (!spectrumEvaluation) { error="Audio geometry replay requires the shared frame analysis"; return std::nullopt; }
    spectrumEvaluation->followers.resize(descriptor.spectrumFields.size());
    deform=[&](auto& geometry,const auto& binding,std::string& failure) {
      const auto index=static_cast<std::size_t>(&binding-descriptor.spectrumFields.data());
      videowire::geometry::spectrum::Bands bands{},followed{};
      videowire::geometry::spectrum::FeaturesAt featuresAt,historyAt;
      return spectrumInput(binding,bands,featuresAt,historyAt,failure)
          && videowire::geometry::spectrum::evaluate(binding,bands,
          spectrumEvaluation->timeSeconds,featuresAt,
          spectrumEvaluation->followers[index],followed,failure)
          && videowire::geometry::applyAudioDeformer(geometry,binding,followed,spectrumEvaluation->timeSeconds,failure);
    };
  }
  // One ordered replay combines timeline/score fields and audio deformation.
  // Distribution and queries therefore see the same evaluated source frame.
  if (!videowire::geometry::validateOperationPlan(descriptor,{},&materialized,error,nullptr,
      runtimeFields ? &evaluation : nullptr,deform)) return std::nullopt;
  if (descriptor.carrier == videowire::geometry::CarrierKind::geometry3D) {
    if (orderedAudio || runtimeFields) {
      const auto terminal=materialized.geometries.find(descriptor.stableId);
      if (terminal==materialized.geometries.end()) {
        error="Reactive geometry replay requires a retained terminal mesh; this operation combination is unavailable";
        return std::nullopt;
      }
      mesh=&terminal->second;
    } else mesh=&std::get<videowire::geometry::GeometryData>(descriptor.data);
    if (orderedAudio || runtimeFields) meshAttributes=&materialized.terminalAttributes;
  } else if (descriptor.carrier == videowire::geometry::CarrierKind::instances3D) {
    if (!admission.plan->capabilities().gpuInstancingWithoutMeshExpansion ||
        materialized.instances.instances.empty()) {
      if (error.empty())
        error = "native Geometry Core instancing lacks retained operation results";
      return std::nullopt;
    }
    instances = &materialized.instances;
    const auto sourceId = instances->instances.front().sourceStableId;
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
    for (std::size_t index = 0; index < descriptor.spectrumFields.size(); ++index) {
      const auto& binding = descriptor.spectrumFields[index];
      if (orderedAudio && binding.target==videowire::geometry::spectrum::Target::geometryDeformation) continue;
      videowire::geometry::spectrum::Bands currentBands{};
      videowire::geometry::spectrum::FeaturesAt featuresAt,historyAt;
      if (!spectrumInput(binding,currentBands,featuresAt,historyAt,error)) return std::nullopt;
      videowire::geometry::spectrum::Bands followed {};
      if (binding.history.enabled) {
        std::vector<videowire::geometry::spectrum::Bands> rows;
        if (instances != nullptr || !videowire::geometry::spectrum::evaluateHistory(binding,
              currentBands, spectrumEvaluation->timeSeconds, historyAt,
              spectrumEvaluation->liveFrameAvailable, admission.use == PlanUse::exportRender,
              spectrumEvaluation->followers[index], rows, error,
              spectrumEvaluation->seekGeneration, spectrumEvaluation->loopGeneration)) return std::nullopt;
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
            spectrumEvaluation->followers[index], followed, error)) return std::nullopt;
      if (binding.target == videowire::geometry::spectrum::Target::instanceTransform) {
        if (instances == nullptr || binding.coordinates.size() != materialized.instances.instances.size()) {
          error = "Spectrum Instancer has no matching native instance set";
          return std::nullopt;
        }
        for (std::size_t index = 0; index < materialized.instances.instances.size(); ++index) {
          const auto amount = videowire::geometry::spectrum::sample(binding, followed, binding.coordinates[index]);
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
  if (!descriptor.scoreFields.empty()) {
    if (instances != nullptr) {
      error = "Score Field requires a Geometry3D draw"; return std::nullopt;
    }
    if (deformedMesh.positions.empty()) deformedMesh=*mesh;
    for (const auto& binding : descriptor.scoreFields) {
      scorefield::Evaluation evaluated;
      if (!scorefield::evaluate(binding,runtimeInputs.canonicalBlockCFrame,evaluated,error))
        return std::nullopt;
      for (std::size_t index=0;index<deformedMesh.positions.size();++index) {
        auto& p=deformedMesh.positions[index]; const auto& offset=evaluated.offsets[index];
        p.x+=offset[0]; p.y+=offset[1]; p.z+=offset[2];
      }
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
  auto prepared = instances != nullptr && !videowire::geometry::hasMaterialTable(descriptor,materialized)
      ? backend.prepareGeometryInstances(scene, surfaceProgram, admission.plan,
                                         diagnosticInstanceIdentityColors,
                                         (orderedAudio || runtimeFields) ? &materialized.terminalAttributes : nullptr)
      : backend.prepare(scene, surfaceProgram);
  if (!prepared.prepared || !prepared.resources) {
    error = prepared.error.empty() ? "Geometry Core native resource preparation failed"
                                   : prepared.error;
    return std::nullopt;
  }
  if (admission.nativeResources) {
    std::lock_guard<std::mutex> lock(admission.nativeResources->mutex);
    admission.nativeResources->value = prepared.resources;
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
  return NativeGeometryExecution{std::move(prepared.resources),
                                 std::move(rendered.frame),
                                 admission.ownerLease,
                                 std::move(drawnInstanceIds),
                                 rendered.stats,
                                 admission.plan->receipt().backendBinding};
}
} // namespace videohelper::geometry
