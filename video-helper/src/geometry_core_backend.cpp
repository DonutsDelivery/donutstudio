#include "geometry_core_backend.h"

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
                      arbitgpu::NativeFixtureSceneRuntimeInputs runtimeInputs) {
  using namespace HarmonicMIDI::grid;
  error.clear();
  if (!admission.plan || width == 0 || height == 0) {
    error = "Geometry Core native execution requires a plan and non-zero extent";
    return std::nullopt;
  }
  const auto &descriptor = admission.plan->value().descriptor();
  const videowire::geometry::GeometryData *mesh = nullptr;
  const videowire::geometry::InstancesData *instances = nullptr;
  videowire::geometry::MaterializedInstancePlan materialized;
  if (descriptor.carrier == videowire::geometry::CarrierKind::geometry3D) {
    mesh = &std::get<videowire::geometry::GeometryData>(descriptor.data);
  } else if (descriptor.carrier == videowire::geometry::CarrierKind::instances3D) {
    if (!admission.plan->capabilities().gpuInstancingWithoutMeshExpansion ||
        !videowire::geometry::validateOperationPlan(
            descriptor, {}, &materialized, error) ||
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
  } else {
    error = "native Geometry Core execution does not advertise this carrier";
    return std::nullopt;
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
  videowire::geometry::StableId materialId = 1;
  for (const auto &operation : descriptor.operations)
    if (operation.code == videowire::geometry::OperationCode::setMaterial)
      materialId = operation.stableId;
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
  auto prepared = instances != nullptr
      ? backend.prepareGeometryInstances(scene, {}, admission.plan,
                                         diagnosticInstanceIdentityColors)
      : backend.prepare(scene);
  if (!prepared.prepared || !prepared.resources) {
    error = prepared.error.empty() ? "Geometry Core native resource preparation failed"
                                   : prepared.error;
    return std::nullopt;
  }
  if (admission.nativeResources) {
    std::lock_guard<std::mutex> lock(admission.nativeResources->mutex);
    admission.nativeResources->value = prepared.resources;
  }
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
