#include "../../shared/GeometryTimelineSceneBinding.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace {
using namespace videowire::geometry;
using namespace HarmonicMIDI::grid;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool nearly(float a, float b) { return std::fabs(a - b) <= 1.0e-5f; }

PortContract instanceContract() {
  PortContract contract;
  contract.carrier = CarrierKind::instances3D; contract.overflow = OverflowPolicy::reject;
  contract.maxInstances = 64;
  return contract;
}

std::optional<ValueDescriptor> instancer(std::string& error) {
  PortContract mesh;
  mesh.carrier = CarrierKind::geometry3D; mesh.overflow = OverflowPolicy::reject;
  mesh.maxVertices = 4; mesh.maxIndices = 6;
  const auto prototype = lowerGrid(9901, 9901, 1, 2, 2, 1, mesh, error);
  if (!prototype) return std::nullopt;
  spectrum::Binding binding;
  binding.firstBand = 8; binding.lastBand = 11; binding.baseScale = {0.35f, 0.35f, 0.35f};
  binding.scaleResponse = {0, 3, 0}; binding.direction = {0, 0.5f, 0};
  return lowerSpectrumInstancer(*prototype, 9902, binding, {0, 2, 2, 2}, instanceContract(), error);
}

spectrum::Bands bands(float level) {
  spectrum::Bands values{};
  for (std::size_t band = 8; band <= 11; ++band) values[band] = level * static_cast<float>(band - 7) / 4.0f;
  return values;
}

std::optional<Visual3DScene> sceneFor(const AdmittedValue& value, std::string& error) {
  Visual3DScene scene;
  scene.id = {7}; scene.activeCamera = {50};
  scene.cameras[0].id = {50}; scene.cameras[0].transform.translation.z = 4; scene.cameraCount = 1;
  SceneObjectRecord object; object.id = {10};
  SceneMaterialRecord material; material.id = {20}; material.baseColor = {1, 0.5f, 0.2f};
  if (!appendGeometryToScene(value, object, material, scene, error) || !validateVisual3DScene(scene).valid())
    return std::nullopt;
  return scene;
}

const SceneObjectRecord* child(const Visual3DScene& scene, StableId instance) {
  return visual3d_detail::findById(scene.objects, scene.objectCount, sceneInstanceId({10}, instance));
}

// A composed Spectrum Instancer moves and scales each instance object from this
// frame's bands, by the same rule as the native instancer, and follows a new frame.
void composedInstancerFollowsEachFrame() {
  std::string error;
  const auto value = instancer(error);
  if (!value) std::fprintf(stderr, "instancer: %s\n", error.c_str());
  check(value.has_value(), "the Spectrum Instancer lowers");
  if (!value) return;
  check(supportsTimelineSceneInstances(*value), "a Spectrum Instancer composes in Scene3D");
  check(!supportsTimelineSceneGeometry(*value), "the mesh route keeps refusing instances");
  const auto admitted = admitValue(*value, instanceContract(), error);
  auto scene = admitted ? sceneFor(*admitted, error) : std::nullopt;
  if (!scene) std::fprintf(stderr, "scene: %s\n", error.c_str());
  check(scene.has_value(), "the instancer appends to a composed scene");
  if (!admitted || !scene) return;
  TimelineSceneBinding binding{{10}, encodeLoweredPlanText(lowerRuntimePlan(instanceContract(), *admitted))};
  const auto decoded = decodeTimelineScenePlan(binding, error);
  if (!decoded) std::fprintf(stderr, "decode: %s\n", error.c_str());
  check(decoded.has_value(), "the timeline binding decodes an Instances3D plan");
  if (!decoded) return;
  check(validTimelineSceneBinding(binding, *scene, *decoded, error), "the binding owns its instance objects");
  const auto& rest = std::get<InstancesData>(admitted->descriptor().data).instances;
  const auto& spectral = admitted->descriptor().spectrumFields.front();
  check(rest.size() == spectral.coordinates.size() && !rest.empty(), "one instance per sampled band");

  float level = 0;
  RuntimeFieldEvaluation evaluation;
  evaluation.spectrumAt = [&](const spectrum::Binding&, spectrum::Bands& followed, std::string&) {
    followed = bands(level); return true;
  };
  evaluation.spectrumHistoryAt = [](const spectrum::Binding&, std::vector<spectrum::Bands>&, std::string& failure) {
    failure = "no history here"; return false;
  };
  const auto matches = [&](const Visual3DScene& frame) {
    for (std::size_t i = 0; i < rest.size(); ++i) {
      const auto* object = child(frame, rest[i].stableId);
      if (!object) return false;
      const auto amount = spectrum::sample(spectral, bands(level), spectral.coordinates[i]);
      const auto& base = rest[i].transform;
      const auto scaleY = std::clamp(base.scale.y * (1.0f + amount * 3.0f), 0.000001f, 1000.0f);
      if (!nearly(object->transform.translation.x, base.translation.x)
          || !nearly(object->transform.translation.y, base.translation.y + 0.5f * amount)
          || !nearly(object->transform.scale.x, base.scale.x) || !nearly(object->transform.scale.y, scaleY))
        return false;
    }
    return true;
  };
  auto quiet = *scene;
  check(replaceTimelineSceneMesh(binding, *decoded, 1.0, quiet, error, &evaluation, false) && matches(quiet),
        "silence leaves each instance at its placement");
  level = 1;
  auto loud = *scene;
  check(replaceTimelineSceneMesh(binding, *decoded, 1.5, loud, error, &evaluation, true) && matches(loud),
        "a loud frame lifts and stretches each instance by its band");
  const auto* loudest = child(loud, rest.back().stableId);
  check(loudest && loudest->transform.translation.y > rest.back().transform.translation.y,
        "the loudest band moves its instance");
  RuntimeFieldEvaluation unread;
  auto refused = *scene;
  check(!replaceTimelineSceneMesh(binding, *decoded, 1.0, refused, error, &unread, true) && !error.empty(),
        "a frame without spectrum readers refuses");
}

// The composed instancer refuses a scene that lost one of its instance objects.
void composedInstancerNeedsEachInstanceObject() {
  std::string error;
  const auto value = instancer(error);
  const auto admitted = value ? admitValue(*value, instanceContract(), error) : std::nullopt;
  auto scene = admitted ? sceneFor(*admitted, error) : std::nullopt;
  check(scene.has_value(), "the instancer scene builds for the ownership check");
  if (!scene) return;
  TimelineSceneBinding binding{{10}, encodeLoweredPlanText(lowerRuntimePlan(instanceContract(), *admitted))};
  --scene->objectCount;
  check(!validTimelineSceneBinding(binding, *scene, *admitted, error), "a missing instance object refuses");
}
} // namespace

int main() {
  composedInstancerFollowsEachFrame();
  composedInstancerNeedsEachInstanceObject();
  if (failures != 0) {
    std::fprintf(stderr, "%d composed spectrum instancer check(s) failed\n", failures);
    return 1;
  }
  std::printf("composed spectrum instancer tests passed\n");
  return 0;
}
