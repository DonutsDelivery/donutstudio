// The Lighthouse and Ocean sea across a second plan install. Every viewport_set_timeline
// bumps the viewport's timeline generation, which is the project generation of every
// Geometry Core owner, so the first frame after a reinstall of the same graph had a new
// owner in a new project generation. Adoption required the same project generation and
// refused with geometry:adopt-miss:generation, and that frame checked the whole sea again:
// live, the first frame after install paid timeline-verify twice (8.2 s + 6.0 s) and
// timeline-replay-checked twice (2.2 s + 3.1 s).
//
// Each case opens the sea in one engine under install 1, then installs again with the
// same plan text on the same authored scene and takes the first frame of install 2, with
// the stage totals the viewport prints for its first frame after install. Raster moves the
// sea on the GPU; Path Trace replays it on the CPU. A project generation bump with
// unchanged plan text takes install 1's verification. A device or helper generation change,
// changed plan bytes or another scene still verify again.
//
// Header only. From the repository root:
//   c++ -std=c++17 -O2 -Ivideo-helper/src -Ishared -Iplugin/Source
//       video-helper/tests/timeline_plan_install_adoption_tests.cpp -o install-adoption-tests
//   ./install-adoption-tests [side=96]
// The side is the sea's grid; 1024 is the starter's sea.
#include "../src/composed_timeline_geometry.h"
#include "../src/geometry_core_backend.h"
#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometrySurfaceWave.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace {
using namespace videohelper::geometry;
using namespace videowire::geometry;
using namespace HarmonicMIDI::grid;
using Scene = Visual3DScene;
using Clock = std::chrono::steady_clock;

int failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}

class Capabilities final : public BackendCapabilitySource {
public:
  BackendCapabilities geometryCoreCapabilities() const override {
    BackendCapabilities c; c.backendIdentity = "install-adoption-tests"; c.deviceIdentity = "test-device";
    c.nativeGpuAvailable = c.geometryCoreExecution = c.immutableSourceBuffers = c.stableElementIds
        = c.typedFieldEvaluation = true;
    c.supportedCarriers = carrierBit(CarrierKind::geometry3D);
    return c;
  }
};

struct Sea final {
  std::shared_ptr<const Scene> authored;
  TimelineSceneBinding binding;
};

bool makeSea(std::uint32_t side, Sea& sea) {
  std::string error;
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = kMaximumAttributes;
  const auto grid = lowerGrid(7, 7, 1, side, side, 0.25f, contract, error);
  const auto wave = grid ? lowerWindSpectrumSurfaceWave(*grid, 9, (StableId{1} << 40) + 9 * 16,
      3.5, 1.5, 110, contract, error) : std::nullopt;
  const auto shaded = wave ? lowerShadeEdit(*wave, 11, 1.0f, 30.0f, {}, error) : std::nullopt;
  const auto admittedContract = shaded ? withAttributeContract(contract, *shaded) : contract;
  const auto admitted = shaded ? admitValue(*shaded, admittedContract, error) : std::nullopt;
  if (!admitted) { std::fprintf(stderr, "FAIL lower: %s\n", error.c_str()); return false; }
  auto scene = std::make_shared<Scene>();
  scene->id = Scene3DId{7};
  scene->cameras[0].id = SceneCameraId{50};
  scene->cameraCount = 1;
  scene->activeCamera = scene->cameras[0].id;
  scene->cameras[0].transform.translation.z = 4;
  scene->lights[0].id = SceneLightId{40};
  scene->lights[0].kind = SceneLightKind::Directional;
  scene->lights[0].intensity = 1.5f;
  scene->lightCount = 1;
  SceneObjectRecord object;
  object.id = SceneObjectId{10};
  object.material = SceneMaterialId{20};
  SceneMaterialRecord material;
  material.id = object.material;
  material.roughness = 0.8f;
  if (!appendGeometryToScene(*admitted, object, material, *scene, error)) {
    std::fprintf(stderr, "FAIL append: %s\n", error.c_str()); return false;
  }
  sea.authored = scene;
  sea.binding.object = object.id;
  sea.binding.loweredPlan = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));
  return true;
}

// One frame of the sea as prepareVisualImportedSceneLayerAtTime takes it, with the
// viewport's per-label stage totals. Null cpuSeaReason is a Raster or Realtime frame.
struct Frame final {
  bool ok = false;
  bool gpu = false;
  double wallMs = 0;
  loopstep::StageTotals stages;
  const char* adoptMiss = nullptr;
  std::uint64_t adopted = 0;
  double stageMs(const char* label) const {
    for (int i = 0; i < stages.count; ++i)
      if (std::strcmp(stages.labels[i], label) == 0) return stages.ns[i] / 1e6;
    return 0;
  }
  bool has(const char* label) const {
    for (int i = 0; i < stages.count; ++i)
      if (std::strcmp(stages.labels[i], label) == 0) return true;
    return false;
  }
};

Frame frame(GeometryCorePlanRuntime& runtime, const PlanOwnerIdentity& owner, const Sea& sea,
    const std::shared_ptr<const Scene>& authored, const TimelineSceneBinding& binding,
    const std::shared_ptr<const void>& publication, double seconds, const char* cpuSeaReason) {
  Frame result;
  std::atomic<const char*> label{nullptr};
  std::atomic<std::int64_t> enter{0};
  loopstep::Sink sink;
  sink.label = &label;
  sink.enterNs = &enter;
  sink.stages = &result.stages;
  const loopstep::SinkInstall install(&sink);
  const auto adoptedBefore = runtime.adoptedTimelinePlans();
  std::string error;
  SurfaceWaveFrame out;
  const auto start = Clock::now();
  {
    loopstep::Scope scope("advance:imported-scene:timeline-geometry");
    result.ok = surfaceWaveFrame(runtime, owner, binding, authored, publication, seconds, cpuSeaReason, [&] {
      return evaluateComposedTimelineGeometry(runtime, owner, binding, authored, seconds, error,
          nullptr, seconds, {}, nullptr, publication);
    }, out);
  }
  result.wallMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  result.gpu = out.waves.has_value();
  result.adoptMiss = runtime.lastAdoptionMiss();
  result.adopted = runtime.adoptedTimelinePlans() - adoptedBefore;
  if (!result.ok) std::fprintf(stderr, "frame failed: %s\n", error.c_str());
  (void) sea;
  return result;
}

void print(const char* name, const Frame& f) {
  std::printf("%-34s wallMs=%.0f %s stagesMs:", name, f.wallMs, f.gpu ? "gpu" : "cpu-replay");
  for (int i = 0; i < f.stages.count; ++i)
    std::printf(" %s=%.0f", f.stages.labels[i], f.stages.ns[i] / 1e6);
  std::printf(" adopted=%llu lastMiss=%s\n", static_cast<unsigned long long>(f.adopted),
      f.adoptMiss ? f.adoptMiss : "none");
}

// Install 1 then install 2 of the same graph in one engine. The live viewport's first
// frame after install covers both, as the plan installed twice before it drew.
void reinstall(const Sea& sea, const char* engine, const char* cpuSeaReason) {
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const auto first = std::make_shared<int>(1), second = std::make_shared<int>(2);
  const PlanOwnerIdentity install1 {1, 4, 5, 6, 9, PlanUse::preview};
  const PlanOwnerIdentity install2 {2, 4, 5, 6, 9, PlanUse::preview};
  const auto a = frame(runtime, install1, sea, sea.authored, sea.binding, first, 1.0, cpuSeaReason);
  const auto b = frame(runtime, install2, sea, sea.authored, sea.binding, second, 1.0, cpuSeaReason);
  const std::string tag = std::string(engine) + " ";
  print((tag + "install 1 first frame").c_str(), a);
  print((tag + "install 2 first frame").c_str(), b);
  std::printf("%-34s verifyMs=%.0f replayCheckedMs=%.0f\n", (tag + "both installs").c_str(),
      a.stageMs("geometry:timeline-verify") + b.stageMs("geometry:timeline-verify"),
      a.stageMs("geometry:timeline-replay-checked") + b.stageMs("geometry:timeline-replay-checked"));
  check(a.ok && b.ok, tag + "a frame failed");
  check(a.has("geometry:timeline-verify"), tag + "install 1's first frame did not verify the sea");
  // A just-launched runtime has nothing verified, and says so instead of naming a generation.
  check(a.has("geometry:adopt-miss:no-verified-revision") && !a.has("geometry:adopt-miss:generation"),
      tag + "install 1's first frame did not name no-verified-revision");
  check(!b.has("geometry:timeline-verify") && !b.has("geometry:timeline-replay-checked"),
      tag + "install 2 of the same plan text verified the sea again");
  check(b.adopted == 1, tag + "install 2 did not take install 1's verification");
  if (!cpuSeaReason) check(b.gpu, tag + "install 2's first frame did not move the sea on the GPU");
  else check(!b.gpu && b.has("geometry:timeline-replay"), tag + "install 2's Path Trace frame did not replay the sea");
  // A later frame of install 2 reuses its own owner.
  const auto c = frame(runtime, install2, sea, sea.authored, sea.binding, second, 1.0 + 1 / 30.0, cpuSeaReason);
  check(c.ok && !c.has("geometry:timeline-verify") && c.adopted == 0, tag + "install 2's second frame verified again");
  // Install 3 of a new graph revision keeps the adoption.
  const PlanOwnerIdentity install3 {3, 4, 5, 6, 10, PlanUse::preview};
  const auto d = frame(runtime, install3, sea, sea.authored, sea.binding, std::make_shared<int>(3), 1.0, cpuSeaReason);
  check(d.ok && !d.has("geometry:timeline-verify") && d.adopted == 1,
      tag + "install 3 with a new plan revision verified the sea again");
}

// What still verifies after a reinstall: the label must be geometry:timeline-verify and the
// miss reason the one named.
void verifiesAgain(const Sea& sea, const PlanOwnerIdentity& next, const std::shared_ptr<const Scene>& authored,
    const TimelineSceneBinding& binding, const char* reason, const char* what) {
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const PlanOwnerIdentity install1 {1, 4, 5, 6, 9, PlanUse::preview};
  const auto a = frame(runtime, install1, sea, sea.authored, sea.binding, std::make_shared<int>(1), 1.0, nullptr);
  const auto b = frame(runtime, next, sea, authored, binding, std::make_shared<int>(2), 1.0, nullptr);
  print(what, b);
  check(a.ok && b.ok, std::string(what) + ": a frame failed");
  check(b.has("geometry:timeline-verify") && b.adopted == 0, std::string(what) + " did not verify again");
  check(b.adoptMiss && std::strcmp(b.adoptMiss, reason) == 0,
      std::string(what) + " named " + (b.adoptMiss ? b.adoptMiss : "none") + ", not " + reason);
  check(!b.gpu, std::string(what) + " moved the sea on the GPU from another verification");
}
} // namespace

int main(int argc, char** argv) {
  const std::uint32_t side = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 96;
  Sea sea;
  if (!makeSea(side, sea)) return 1;
  std::printf("sea %ux%u, plan text %zu chars\n", side, side, sea.binding.loweredPlan.size());
  reinstall(sea, "Raster", nullptr);
  reinstall(sea, "Path Trace", kPathTraceSeaReason);
  // A device or helper generation change is a new GPU context or helper process: it verifies.
  verifiesAgain(sea, {2, 4, 6, 6, 9, PlanUse::preview}, sea.authored, sea.binding,
      "geometry:adopt-miss:generation", "device generation change");
  verifiesAgain(sea, {2, 5, 5, 6, 9, PlanUse::preview}, sea.authored, sea.binding,
      "geometry:adopt-miss:generation", "helper generation change");
  // Changed plan text or another authored scene in the new install is refused and verifies.
  auto changed = sea.binding;
  changed.loweredPlan.back() = changed.loweredPlan.back() == '0' ? '1' : '0';
  {
    Capabilities capabilities;
    GeometryCorePlanRuntime runtime(capabilities);
    const auto a = frame(runtime, {1, 4, 5, 6, 9, PlanUse::preview}, sea, sea.authored, sea.binding,
        std::make_shared<int>(1), 1.0, nullptr);
    const bool took = runtime.adoptVerifiedTimelinePlan({2, 4, 5, 6, 9, PlanUse::preview}, changed, sea.authored, {});
    const char* reason = runtime.lastAdoptionMiss();
    std::printf("%-34s adopted=%d lastMiss=%s\n", "reinstall with changed plan bytes", took ? 1 : 0, reason ? reason : "none");
    check(a.ok && !took, "a reinstall with changed plan bytes took a verification");
    check(reason && std::strcmp(reason, "geometry:adopt-miss:plan-differs") == 0,
        "a reinstall with changed plan bytes was not named plan-differs");
  }
  const std::shared_ptr<const Scene> copy = std::make_shared<Scene>(*sea.authored);
  verifiesAgain(sea, {2, 4, 5, 6, 9, PlanUse::preview}, copy, sea.binding,
      "geometry:adopt-miss:scene-differs", "reinstall on another scene");
  // Another clip or use never takes the verification.
  verifiesAgain(sea, {2, 4, 5, 7, 9, PlanUse::preview}, sea.authored, sea.binding,
      "geometry:adopt-miss:no-verified-revision", "reinstall of another clip");
  if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
  std::printf("PASS\n");
  return 0;
}
