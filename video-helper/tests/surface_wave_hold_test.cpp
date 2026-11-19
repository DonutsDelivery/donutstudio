// The open sea's hold and refusal states, on the frame step the viewport takes
// (geometry::surfaceWaveFrame, from prepareVisualImportedSceneLayerAtTime). Each
// frame's sea is checked against a separate CPU replay at the same time: on the GPU
// route the heights are the ones the wave pass writes (the table summed at each
// vertex's rest X and Y, as kSurfaceWaveComputeSource does), on the CPU route the
// replayed vertices. Fails if any frame keeps the held scene without a wave table,
// if a refused frame is not replayed when asked again, if the sea does not move
// between frames 2..N, or if the path counters skip a frame. Also checks the water
// range a Realtime frame (no glass query) admits. No GPU needed.
// Not in CMake; build it alone from the repository root:
//   c++ -std=c++17 -O2 -Ivideo-helper/src -Ishared -Iplugin/Source
//       video-helper/tests/surface_wave_hold_test.cpp -o surface-wave-hold-test
//   ./surface-wave-hold-test [side=96] [frames=8]
#include "../src/composed_timeline_geometry.h"
#include "../src/geometry_core_backend.h"
#include "../src/gpu_backend/surface_wave_water_range.h"
#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometrySurfaceWave.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace videohelper::geometry;
using Scene = HarmonicMIDI::grid::Visual3DScene;
using ScenePtr = std::shared_ptr<const Scene>;

// The wave pass is single precision; a millimetre on a sea about 3.5 m high.
constexpr double kHeightTolerance = 1.0e-3;
// Smallest largest height change that counts as the sea having moved in 1/30 s.
constexpr double kMovedAtLeast = 1.0e-3;

int failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}

class Capabilities final : public BackendCapabilitySource {
public:
  BackendCapabilities geometryCoreCapabilities() const override {
    using namespace videowire::geometry;
    BackendCapabilities c; c.backendIdentity = "surface-wave-hold-test"; c.deviceIdentity = "test-device";
    c.nativeGpuAvailable = c.geometryCoreExecution = c.immutableSourceBuffers = c.stableElementIds
        = c.typedFieldEvaluation = true;
    c.supportedCarriers = carrierBit(CarrierKind::geometry3D);
    return c;
  }
};

struct Sea final {
  videowire::geometry::TimelineSceneBinding binding;
  ScenePtr authored;
  std::shared_ptr<int> publication = std::make_shared<int>(1);
};

// The water heights this frame draws: the wave table at the held rest positions,
// or the replayed vertices.
std::vector<double> heights(const SurfaceWaveFrame& frame) {
  const auto& object = frame.scene->objects[0];
  std::vector<double> z(object.vertexCount);
  for (std::uint32_t v = 0; v < object.vertexCount; ++v) {
    const auto& p = frame.scene->vertices[object.firstVertex + v].position;
    if (!frame.waves) { z[v] = p.z; continue; }
    float h = 0;
    for (const auto& k : *frame.waves) h += k[2] * std::sin(k[0] * p.x + k[1] * p.y + k[3]);
    z[v] = h;
  }
  return z;
}

double largestDifference(const std::vector<double>& a, const std::vector<double>& b) {
  double most = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) most = std::max(most, std::abs(a[i] - b[i]));
  return a.size() == b.size() ? most : INFINITY;
}

// A CPU replay at this time on its own runtime, the reference for every frame.
std::vector<double> replayHeights(const Sea& sea, double seconds) {
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const PlanOwnerIdentity owner {3, 4, 5, 77, 9, PlanUse::preview};
  std::string error;
  const auto scene = evaluateComposedTimelineGeometry(runtime, owner, sea.binding, sea.authored, seconds, error,
      nullptr, seconds, {}, nullptr, sea.publication);
  if (!scene) { std::fprintf(stderr, "FAIL reference replay: %s\n", error.c_str()); std::exit(1); }
  SurfaceWaveFrame frame;
  frame.scene = scene;
  return heights(frame);
}

// Plays frames like the viewport: the frame step, then the renderer, which runs
// the wave pass or, when refuseAt names this frame, refuses it and has it asked
// again. Every drawn frame is compared with the replay reference.
void play(const Sea& sea, int frames, int refuseAt, const char* name) {
  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const PlanOwnerIdentity owner {3, 4, 5, std::uint64_t(refuseAt + 100), 9, PlanUse::preview};
  const auto history = std::make_shared<videorender::ParticleHistorySource>();
  {
    std::lock_guard<std::mutex> lock(surfaceWavePathMutex());
    lastSurfaceWavePath().reset();
  }
  ScenePtr held, lastScene;
  std::vector<double> previous;
  int gpuDrawn = 0, cpuDrawn = 0, steps = 0;
  for (int index = 0; index < frames; ++index) {
    const double seconds = 1.0 + index / 30.0;
    SurfaceWaveFrame frame;
    for (int attempt = 0;; ++attempt) {
      std::string error;
      const bool stepped = surfaceWaveFrame(runtime, owner, sea.binding, sea.authored, sea.publication, seconds,
          nullptr, [&] {
            return evaluateComposedTimelineGeometry(runtime, owner, sea.binding, sea.authored, seconds, error,
                history.get(), seconds, {}, nullptr, sea.publication);
          }, frame);
      check(stepped, std::string(name) + " frame step: " + error);
      if (!stepped) return;
      ++steps;
      noteSurfaceWavePath(1, frame.waves.has_value(), frame.cpuReason);
      const auto note = surfaceWavePath();
      check(note && note->gpuFrames + note->cpuFrames == std::uint64_t(steps),
          std::string(name) + " path counters skipped a frame step");
      if (!frame.waves) held = frame.scene;
      // The renderer: a frame with a wave table moves the held water, or refuses
      // and the viewport defers the frame and asks for it again.
      if (frame.waves && index == refuseAt && attempt == 0) {
        check(frame.refused != nullptr, std::string(name) + " wave frame carries no refusal flag");
        if (frame.refused) frame.refused->store(true);
        continue;
      }
      check(attempt == 0 || !frame.waves, std::string(name) + " refused frame was not replayed when asked again");
      if (attempt > 0)
        check(frame.cpuReason && std::string(frame.cpuReason) == "backend-refused",
            std::string(name) + " refused frame's reason is not backend-refused");
      break;
    }
    check(frame.scene != nullptr, std::string(name) + " frame has no scene");
    // A frame without a wave table is a fresh replay, never the last drawn scene;
    // otherwise the water would stand still while the clock runs.
    check(frame.waves || frame.scene != lastScene, std::string(name) + " frame " + std::to_string(index)
        + " kept the last scene without a wave table");
    lastScene = frame.scene;
    if (frame.waves) {
      check(frame.scene == held, std::string(name) + " wave frame is not on the held scene");
      check(frame.columns >= 2 && frame.rows >= 2, std::string(name) + " wave frame has no grid");
      ++gpuDrawn;
    } else {
      ++cpuDrawn;
    }
    const auto z = heights(frame);
    const auto reference = replayHeights(sea, seconds);
    const double off = largestDifference(z, reference);
    const double moved = previous.empty() ? 0 : largestDifference(z, previous);
    std::printf("%s frame %d %s%s%s max|z - replay| %.2e  moved %.3f\n", name, index,
        frame.waves ? "gpu" : "cpu-replay", frame.cpuReason ? " " : "", frame.cpuReason ? frame.cpuReason : "",
        off, moved);
    check(off <= kHeightTolerance, std::string(name) + " frame " + std::to_string(index)
        + " heights differ from the replay at its time");
    if (index >= 1) check(moved >= kMovedAtLeast, std::string(name) + " frame " + std::to_string(index)
        + " sea did not move");
    previous = z;
  }
  std::printf("%s: %d gpu frames, %d cpu frames, %d steps\n", name, gpuDrawn, cpuDrawn, steps);
  if (refuseAt < 0) check(gpuDrawn == frames - 1, std::string(name) + " frames 2..N did not take the GPU wave route");
  else check(gpuDrawn == refuseAt - 1 && cpuDrawn == frames - gpuDrawn,
      std::string(name) + " frames after the refusal did not all replay");
}

void waterRange(const Sea& sea, std::uint32_t side) {
  using namespace HarmonicMIDI::grid;
  arbitgpu::SurfaceWaveWaterRange range;
  const auto& scene = *sea.authored;
  const char* why = arbitgpu::surfaceWaveWaterRange(scene, sea.binding.object, side, side, {}, range);
  check(why == nullptr, std::string("Realtime water range refused: ") + (why ? why : ""));
  check(range.firstVertex == scene.objects[0].firstVertex && range.vertexCount == side * side,
      "Realtime water range is not the water object");
  check(arbitgpu::surfaceWaveWaterRange(scene, sea.binding.object, side, side, {-1}, range) == nullptr,
      "a water outside the opaque query was refused");
  check(arbitgpu::surfaceWaveWaterRange(scene, sea.binding.object, side, side, {0}, range) != nullptr,
      "a water in the opaque query was admitted");
  check(arbitgpu::surfaceWaveWaterRange(scene, sea.binding.object, side + 1, side, {}, range) != nullptr,
      "a grid that is not the water was admitted");
  check(arbitgpu::surfaceWaveWaterRange(scene, SceneObjectId{999}, side, side, {}, range) != nullptr,
      "a missing water object was admitted");
  auto shared = scene;
  shared.objects[1] = shared.objects[0];
  shared.objects[1].id = SceneObjectId{102};
  shared.objects[1].vertexCount = 3;
  shared.objectCount = 2;
  check(arbitgpu::surfaceWaveWaterRange(shared, sea.binding.object, side, side, {}, range) != nullptr,
      "a water sharing its vertex range was admitted");
}
} // namespace

int main(int argc, char** argv) {
  using namespace videowire::geometry;
  using namespace HarmonicMIDI::grid;
  const std::uint32_t side = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 96;
  const int frames = argc > 2 ? std::max(4, std::atoi(argv[2])) : 8;
  std::string error;
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = kMaximumAttributes;
  // Lighthouse and Ocean's sea at a smaller grid: Grid, Wind Spectrum, Shade Smooth.
  const auto grid = lowerGrid(7, 7, 1, side, side, 0.25f, contract, error);
  const auto wave = grid ? lowerWindSpectrumSurfaceWave(*grid, 9, (StableId{1} << 40) + 9 * 16,
      3.5, 1.5, 110, contract, error) : std::nullopt;
  const auto shaded = wave ? lowerShadeEdit(*wave, 11, 1.0f, 30.0f, {}, error) : std::nullopt;
  const auto admittedContract = shaded ? withAttributeContract(contract, *shaded) : contract;
  const auto admitted = shaded ? admitValue(*shaded, admittedContract, error) : std::nullopt;
  if (!admitted) { std::fprintf(stderr, "FAIL lower: %s\n", error.c_str()); return 1; }
  if (!recognizeSurfaceWaveProgram(admitted->descriptor())) { std::fprintf(stderr, "FAIL wave chain not recognized\n"); return 1; }
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
  Sea sea;
  sea.authored = scene;
  sea.binding.object = object.id;
  sea.binding.loweredPlan = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));

  play(sea, frames, -1, "held");
  play(sea, frames, 2, "refused");
  waterRange(sea, side);
  if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
  std::printf("PASS\n");
  return 0;
}
