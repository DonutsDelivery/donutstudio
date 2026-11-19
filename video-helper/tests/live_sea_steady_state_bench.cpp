// Steady-state sea step of the live viewport for Lighthouse and Ocean: the 1024 by
// 1024 Grid, Wind Spectrum, Shade Smooth sea, evaluated per frame the way
// prepareVisualImportedSceneLayerAtTime does it, with the non-null particle history
// the viewport hands every layer. "old" is the gate before this change, where that
// history alone sent every frame to the CPU replay; "new" takes the GPU wave hold
// from frame 2 on, so the frame keeps one scene snapshot (the static owner, its seam
// and its materials survive) and only the wave table is built on the CPU.
// Prints per-frame times for frames 2..N and the distinct snapshots seen there.
// Exits 1 if the new route doesn't hold the sea or keep one snapshot.
// Not in CMake; build it alone from the repository root:
//   c++ -std=c++17 -O3 -DNDEBUG -Ivideo-helper/src -Ishared -Iplugin/Source -I<nlohmann include>
//       video-helper/tests/live_sea_steady_state_bench.cpp -o live-sea-bench
//   ./live-sea-bench [side=1024] [frames=8]
#include "../src/composed_timeline_geometry.h"
#include "../src/geometry_core_backend.h"
#include "../src/gpu_backend/backend.h"
#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometrySurfaceWave.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class Capabilities final : public videohelper::geometry::BackendCapabilitySource {
public:
  videohelper::geometry::BackendCapabilities geometryCoreCapabilities() const override {
    using namespace videohelper::geometry;
    using namespace videowire::geometry;
    BackendCapabilities c; c.backendIdentity = "live-sea-bench"; c.deviceIdentity = "bench-device";
    c.nativeGpuAvailable = c.geometryCoreExecution = c.immutableSourceBuffers = c.stableElementIds
        = c.typedFieldEvaluation = true;
    c.supportedCarriers = carrierBit(CarrierKind::geometry3D);
    return c;
  }
};

struct Route final {
  std::vector<double> steadyMs;
  std::set<const void*> steadySnapshots;
  int gpuFrames = 0;
};

// One run of the viewport's sea step. oldGate keeps the history check that sent the
// live viewport to the CPU replay on every frame.
bool run(bool oldGate, int frames, const videowire::geometry::TimelineSceneBinding& binding,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& authored, Route& route) {
  using namespace videohelper::geometry;
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const PlanOwnerIdentity owner {3, 4, 5, oldGate ? 8u : 9u, 9, PlanUse::preview};
  const auto history = std::make_shared<videorender::ParticleHistorySource>();
  const auto* waterHistory = history.get();
  const auto publication = std::make_shared<int>(1);
  for (int frame = 0; frame < frames; ++frame) {
    const double seconds = 1.0 + frame / 30.0;
    std::string error;
    const auto start = Clock::now();
    const bool gpuWaveFrame = !oldGate || waterHistory == nullptr;
    const auto* hold = gpuWaveFrame ? surfaceWaveHold(runtime, owner, binding, authored, publication) : nullptr;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> snapshot;
    if (hold != nullptr) {
      auto wave = std::make_shared<arbitgpu::NativeSurfaceWaveFrame>();
      wave->object = binding.object;
      wave->columns = hold->program->columns;
      wave->rows = hold->program->rows;
      wave->waves = videowire::geometry::surfaceWaveTable(*hold->program, seconds);
      snapshot = hold->scene;
    } else {
      snapshot = evaluateComposedTimelineGeometry(runtime, owner, binding, authored, seconds, error,
          waterHistory, seconds, {}, nullptr, publication);
      if (!snapshot) { std::fprintf(stderr, "FAIL frame %d: %s\n", frame, error.c_str()); return false; }
      if (gpuWaveFrame) rememberSurfaceWaveScene(runtime, owner, binding, authored, publication, snapshot);
    }
    const double ms = msSince(start);
    std::printf("%s frame %d %s %.2f ms\n", oldGate ? "old" : "new", frame, hold ? "gpu-wave" : "cpu-replay", ms);
    if (frame >= 1) {
      route.steadyMs.push_back(ms);
      route.steadySnapshots.insert(snapshot.get());
      if (hold) ++route.gpuFrames;
    }
  }
  return true;
}

double median(std::vector<double> values) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
} // namespace

int main(int argc, char** argv) {
  using namespace videowire::geometry;
  using namespace HarmonicMIDI::grid;
  const std::uint32_t side = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 1024;
  const int frames = argc > 2 ? std::max(3, std::atoi(argv[2])) : 8;
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
  if (!admitted) { std::fprintf(stderr, "FAIL lower: %s\n", error.c_str()); return 1; }
  std::printf("wave chain recognized: %s\n",
      recognizeSurfaceWaveProgram(admitted->descriptor()) ? "yes" : "no");
  auto scene = std::make_shared<Visual3DScene>();
  scene->id = Scene3DId{301};
  scene->cameras[0].id = SceneCameraId{401};
  scene->cameraCount = 1;
  scene->activeCamera = scene->cameras[0].id;
  SceneObjectRecord object;
  object.id = SceneObjectId{101};
  object.material = SceneMaterialId{201};
  SceneMaterialRecord material;
  material.id = object.material;
  if (!appendGeometryToScene(*admitted, object, material, *scene, error)) {
    std::fprintf(stderr, "FAIL append: %s\n", error.c_str()); return 1;
  }
  const std::shared_ptr<const Visual3DScene> authored = scene;
  TimelineSceneBinding binding;
  binding.object = object.id;
  binding.loweredPlan = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));

  Route before, after;
  if (!run(true, frames, binding, authored, before) || !run(false, frames, binding, authored, after)) return 1;
  std::printf("steady frames 2..%d: old median %.2f ms, %zu snapshots; new median %.2f ms, %zu snapshots, %d gpu-wave\n",
      frames, median(before.steadyMs), before.steadySnapshots.size(), median(after.steadyMs),
      after.steadySnapshots.size(), after.gpuFrames);
  const bool held = after.gpuFrames == frames - 1 && after.steadySnapshots.size() == 1;
  if (!held) std::fprintf(stderr, "FAIL the new route did not hold the sea on one snapshot\n");
  return held ? 0 : 1;
}
