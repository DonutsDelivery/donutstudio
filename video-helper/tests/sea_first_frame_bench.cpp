// What a graph edit costs the Lighthouse and Ocean sea before its next frame: the
// 1024 by 1024 Grid, Wind Spectrum, Shade Smooth sea, timed piece by piece.
//   encode   replaysExactly, which the app's lowering runs to send the plan as its
//            operations alone (shared/GeometryCoreTransport.h)
//   check    validTimelineSceneBinding on the filled scene, as the request decoder runs it
//   decode   one render request carrying the sea, read the way plan lowering reads it: the
//            first read rebuilds the left-out mesh, the second finds it kept
//   frame    the first Preview frame of a plan revision through surfaceWaveFrame, then
//            the next frame, then the first frame of a second revision of the same sea,
//            which takes the first revision's verification and hold (exits 1 if it replays)
//   engine   the Render node's engine switched while stopped, as the lighthouse does it: it opens
//            in Path Trace, which replays the sea on the CPU, then Path Trace to Realtime, Realtime
//            to Path Trace and Raster to Realtime, each a new plan revision of the same request
//            scene. Both engines' requests must decode to the same filled scene. Last, a
//            revision whose scene is an equal copy verifies again and names scene-differs.
// -DSEA_BENCH_BEFORE builds it against headers without the adoption and skips those checks.
// -DSEA_BENCH_ENGINE_BEFORE skips only the engine switch checks.
// Not in CMake; build it alone from the repository root:
//   c++ -std=c++17 -O3 -DNDEBUG -Ivideo-helper/src -Ishared -Iplugin/Source
//       video-helper/tests/sea_first_frame_bench.cpp -o sea-first-frame-bench
//   ./sea-first-frame-bench [side=1024]
#include "../src/composed_timeline_geometry.h"
#include "../src/geometry_core_backend.h"
#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometrySurfaceWave.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace {
using namespace videohelper::geometry;
using Clock = std::chrono::steady_clock;
using Scene = HarmonicMIDI::grid::Visual3DScene;

double msSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what); }
}

class Capabilities final : public BackendCapabilitySource {
public:
  BackendCapabilities geometryCoreCapabilities() const override {
    using namespace videowire::geometry;
    BackendCapabilities c; c.backendIdentity = "sea-first-frame-bench"; c.deviceIdentity = "bench-device";
    c.nativeGpuAvailable = c.geometryCoreExecution = c.immutableSourceBuffers = c.stableElementIds
        = c.typedFieldEvaluation = true;
    c.supportedCarriers = carrierBit(CarrierKind::geometry3D);
    return c;
  }
};

// One Preview frame of the sea the way prepareVisualImportedSceneLayerAtTime takes it.
bool frame(GeometryCorePlanRuntime& runtime, const PlanOwnerIdentity& owner,
    const videowire::geometry::TimelineSceneBinding& binding, const std::shared_ptr<const Scene>& authored,
    const std::shared_ptr<int>& publication, double seconds, const char* name, SurfaceWaveFrame& sea,
    const char* cpuSeaReason = nullptr) {
  std::string error;
  const auto start = Clock::now();
  const bool ok = surfaceWaveFrame(runtime, owner, binding, authored, publication, seconds, cpuSeaReason, [&] {
    return evaluateComposedTimelineGeometry(runtime, owner, binding, authored, seconds, error,
        nullptr, seconds, {}, nullptr, publication);
  }, sea);
  std::printf("frame %-28s %8.1f ms  %s%s%s\n", name, msSince(start), sea.waves ? "gpu" : "cpu-replay",
      sea.cpuReason ? " " : "", sea.cpuReason ? sea.cpuReason : "");
  if (!ok) std::fprintf(stderr, "FAIL %s: %s\n", name, error.c_str());
  return ok;
}
} // namespace

int main(int argc, char** argv) {
  using namespace videowire::geometry;
  using namespace HarmonicMIDI::grid;
  const std::uint32_t side = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 1024;
  std::string error;
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = kMaximumAttributes;
  auto start = Clock::now();
  const auto grid = lowerGrid(7, 7, 1, side, side, 0.25f, contract, error);
  const auto wave = grid ? lowerWindSpectrumSurfaceWave(*grid, 9, (StableId{1} << 40) + 9 * 16,
      3.5, 1.5, 110, contract, error) : std::nullopt;
  const auto shaded = wave ? lowerShadeEdit(*wave, 11, 1.0f, 30.0f, {}, error) : std::nullopt;
  const auto admittedContract = shaded ? withAttributeContract(contract, *shaded) : contract;
  const auto admitted = shaded ? admitValue(*shaded, admittedContract, error) : std::nullopt;
  if (!admitted) { std::fprintf(stderr, "FAIL lower: %s\n", error.c_str()); return 1; }
  std::printf("lower and admit the sea            %8.1f ms\n", msSince(start));

  start = Clock::now();
  const bool exact = transport_detail::replaysExactly(admitted->descriptor());
  std::printf("encode replaysExactly              %8.1f ms  %s\n", msSince(start), exact ? "exact" : "not exact");
  check(exact, "the sea does not replay exactly");
  start = Clock::now();
  const auto planText = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));
  std::printf("encode lowerRuntimePlan, first    %8.1f ms  %zu chars\n", msSince(start), planText.size());
  start = Clock::now();
  const auto again = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));
  std::printf("encode lowerRuntimePlan, same value%8.1f ms\n", msSince(start));
  check(again == planText, "the same value lowered to different bytes");
  {
    // One changed data byte is a new value: it replays, and its data no longer matches its plan.
    auto edited = *admitted;
    auto& mesh = std::get<GeometryData>(const_cast<ValueDescriptor&>(edited.descriptor()).data);
    mesh.positions[0].z += 1;
    start = Clock::now();
    const auto changed = lowerRuntimePlan(admittedContract, edited, {}, {}, true);
    std::printf("encode lowerRuntimePlan, new data  %8.1f ms\n", msSince(start));
    check(encodeLoweredPlanText(changed) != planText, "a value with changed data was sent as its operations alone");
  }

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
    std::fprintf(stderr, "FAIL append: %s\n", error.c_str()); return 1;
  }
  const std::shared_ptr<const Scene> authored = scene;
  TimelineSceneBinding binding;
  binding.object = object.id;
  binding.loweredPlan = planText;

  start = Clock::now();
  const auto decoded = decodeTimelineScenePlan(binding, error);
  std::printf("decodeTimelineScenePlan            %8.1f ms\n", msSince(start));
  if (!decoded) { std::fprintf(stderr, "FAIL decode: %s\n", error.c_str()); return 1; }
  start = Clock::now();
  check(validTimelineSceneBinding(binding, *authored, *decoded, error), "the sea binding is not valid");
  std::printf("check validTimelineSceneBinding    %8.1f ms\n", msSince(start));
  {
    Capabilities capabilities;
    GeometryCorePlanRuntime runtime(capabilities);
    const PlanOwnerIdentity owner {3, 4, 5, 6, 1, PlanUse::preview};
    const auto bytes = decodeLoweredPlanText(binding.loweredPlan, error);
    runtime.reconcileGeneration(3, 4, 5, error);
    start = Clock::now();
    const auto plan = bytes ? runtime.admitPreview(owner, *bytes, {}, {}, error) : std::nullopt;
    std::printf("admitPreview                       %8.1f ms\n", msSince(start));
    check(plan.has_value(), "admitPreview refused the sea");
    if (plan) {
      auto copy = std::make_shared<Scene>(*authored);
      start = Clock::now();
      check(replaceTimelineSceneMesh(binding, plan->plan->value(), 1.0, *copy, error, nullptr, false),
          "unverified replace failed");
      std::printf("replaceTimelineSceneMesh unverified%8.1f ms\n", msSince(start));
      start = Clock::now();
      check(replaceTimelineSceneMesh(binding, plan->plan->value(), 1.0, *copy, error, nullptr, true),
          "verified replace failed");
      std::printf("replaceTimelineSceneMesh verified  %8.1f ms\n", msSince(start));
    }
  }

  {
    namespace render = visualimportedscenerender;
    render::Request request;
    request.sourceStableId = 7;
    request.renderStableId = 12;
    request.structuralRevision = 9;
    request.evaluationRevision = 12;
    request.sceneSnapshot = authored;
    request.timelineGeometry = binding;
    check(render::valid(request), "the sea request is not valid");
    start = Clock::now();
    const auto wire = render::encode(request);
    std::printf("request encode                     %8.1f ms  %zu chars\n", msSince(start), wire.size());
    check(!wire.empty(), "the sea request did not encode");
    sceneaov::detail::sceneTokenMemoSeeding = true;
    for (const char* name : {"request decode, rebuilds mesh  ", "request decode, kept scene     "}) {
      const auto runs = render::detail::timelineValidationRuns;
      render::Request decoded;
      start = Clock::now();
      const bool ok = render::decode(wire, decoded);
      std::printf("%s    %8.1f ms  %zu check replays\n", name, msSince(start),
          render::detail::timelineValidationRuns - runs);
      check(ok, "the sea request did not decode");
    }
    sceneaov::detail::sceneTokenMemoSeeding = false;
    // A scene that does not hold the plan's mesh still fails, kept memo or not.
    auto wrong = std::make_shared<Scene>(*authored);
    wrong->vertices[wrong->objects[0].firstVertex].position.z += 1;
    auto bad = request;
    bad.sceneSnapshot = wrong;
    check(!render::valid(bad), "a scene with a moved sea vertex was valid");
  }

  Capabilities capabilities;
  GeometryCorePlanRuntime runtime(capabilities);
  const auto publication = std::make_shared<int>(1);
  const PlanOwnerIdentity first {3, 4, 5, 6, 1, PlanUse::preview};
  SurfaceWaveFrame sea;
  if (!frame(runtime, first, binding, authored, publication, 1.0, "revision 1 first", sea)) return 1;
  if (!frame(runtime, first, binding, authored, publication, 1.0 + 1 / 30.0, "revision 1 second", sea)) return 1;
  check(sea.waves.has_value(), "revision 1's second frame did not move the sea on the GPU");
  // A graph edit elsewhere: a new plan revision, the same scene and binding bytes.
  const PlanOwnerIdentity second {3, 4, 5, 6, 2, PlanUse::preview};
  const auto publication2 = std::make_shared<int>(2);
  if (!frame(runtime, second, binding, authored, publication2, 1.0 + 2 / 30.0, "revision 2 first", sea)) return 1;
#ifndef SEA_BENCH_BEFORE
  check(sea.waves.has_value(), "revision 2's first frame did not take revision 1's hold");
  check(runtime.adoptedTimelinePlans() == 1, "revision 2 did not take revision 1's verification");
#endif
  if (!frame(runtime, second, binding, authored, publication2, 1.0 + 3 / 30.0, "revision 2 second", sea)) return 1;
  check(sea.waves.has_value(), "revision 2's second frame did not move the sea on the GPU");
#ifndef SEA_BENCH_BEFORE
  // Revision 1 asked again takes the hold back.
  if (!frame(runtime, first, binding, authored, publication, 1.0 + 4 / 30.0, "revision 1 again", sea)) return 1;
  check(sea.waves.has_value(), "revision 1 asked again did not take the hold back");
  // Changed plan bytes, another binding object or another scene never take a verification.
  {
    auto changed = binding;
    changed.loweredPlan.back() = changed.loweredPlan.back() == '0' ? '1' : '0';
    check(!runtime.adoptVerifiedTimelinePlan({3, 4, 5, 6, 3, PlanUse::preview}, changed, authored, {}),
        "a plan with changed bytes took a verification");
    auto other = binding;
    other.object = SceneObjectId{11};
    check(!runtime.adoptVerifiedTimelinePlan({3, 4, 5, 6, 3, PlanUse::preview}, other, authored, {}),
        "another object took a verification");
    const std::shared_ptr<const Scene> copy = std::make_shared<Scene>(*authored);
    check(!runtime.adoptVerifiedTimelinePlan({3, 4, 5, 6, 3, PlanUse::preview}, binding, copy, {}),
        "another scene took a verification");
    check(!runtime.adoptVerifiedTimelinePlan({3, 4, 5, 7, 3, PlanUse::preview}, binding, authored, {}),
        "another clip took a verification");
    // A revision on another scene replays its first frame and makes its own hold.
    const PlanOwnerIdentity third {3, 4, 5, 6, 3, PlanUse::preview};
    if (!frame(runtime, third, binding, copy, publication2, 1.0 + 5 / 30.0, "revision 3 other scene", sea)) return 1;
    check(!sea.waves.has_value() && sea.cpuReason && std::string(sea.cpuReason) == "first-frame",
        "a revision on another scene did not replay its first frame");
  }
#endif
  {
    // The engine is a request field, so the Path Trace and Realtime requests of one graph carry the
    // same scene text and decode to the same filled scene.
    namespace render = visualimportedscenerender;
    render::Request pathTrace;
    pathTrace.sourceStableId = 7;
    pathTrace.renderStableId = 12;
    pathTrace.structuralRevision = 20;
    pathTrace.evaluationRevision = 20;
    pathTrace.sceneSnapshot = authored;
    pathTrace.timelineGeometry = binding;
    pathTrace.renderEngine = render::RenderEngine::PathTrace;
    auto realtime = pathTrace;
    realtime.renderEngine = render::RenderEngine::Realtime;
    realtime.structuralRevision = realtime.evaluationRevision = 21;
    check(render::valid(pathTrace) && render::valid(realtime), "the engine switch requests are not valid");
    const auto ptWire = render::encode(pathTrace), rtWire = render::encode(realtime);
    sceneaov::detail::sceneTokenMemoSeeding = true;
    render::Request ptDecoded, rtDecoded;
    start = Clock::now();
    check(render::decode(ptWire, ptDecoded), "the Path Trace request did not decode");
    std::printf("engine decode Path Trace request    %8.1f ms\n", msSince(start));
    start = Clock::now();
    check(render::decode(rtWire, rtDecoded), "the Realtime request did not decode");
    std::printf("engine decode Realtime request      %8.1f ms  %s scene\n", msSince(start),
        ptDecoded.sceneSnapshot == rtDecoded.sceneSnapshot ? "same" : "another");
    sceneaov::detail::sceneTokenMemoSeeding = false;
    check(ptDecoded.sceneSnapshot && ptDecoded.sceneSnapshot == rtDecoded.sceneSnapshot,
        "the two engines' requests decoded to different scenes");
    const auto requestScene = ptDecoded.sceneSnapshot;
    if (requestScene && ptDecoded.timelineGeometry) {
      const auto& requestBinding = *ptDecoded.timelineGeometry;
      Capabilities engineCapabilities;
      GeometryCorePlanRuntime engines(engineCapabilities);
      std::uint64_t revision = 20;
      const auto next = [&] { return PlanOwnerIdentity {3, 4, 5, 6, revision++, PlanUse::preview}; };
      const auto ptOpen = next();
      const auto openPublication = std::make_shared<int>(20);
      double at = 2.0;
      // The lighthouse opens in Path Trace: one sample after another at a stopped time.
      if (!frame(engines, ptOpen, requestBinding, requestScene, openPublication, at, "PT open first", sea, "path-trace")) return 1;
      if (!frame(engines, ptOpen, requestBinding, requestScene, openPublication, at, "PT open next sample", sea, "path-trace")) return 1;
      // A graph edit that keeps Path Trace: the kernel-warm first sample of a new revision.
      const auto ptEdit = next();
      const auto editPublication = std::make_shared<int>(21);
      if (!frame(engines, ptEdit, requestBinding, requestScene, editPublication, at, "PT new revision first", sea, "path-trace")) return 1;
      // Path Trace to Realtime while stopped.
      const auto rtSwitch = next();
      const auto rtPublication = std::make_shared<int>(22);
      if (!frame(engines, rtSwitch, requestBinding, requestScene, rtPublication, at, "PT->RT first", sea)) return 1;
#ifndef SEA_BENCH_ENGINE_BEFORE
      check(sea.waves.has_value(), "Path Trace to Realtime replayed the sea on its first frame");
#endif
      if (!frame(engines, rtSwitch, requestBinding, requestScene, rtPublication, at + 1 / 30.0, "PT->RT second", sea)) return 1;
      check(sea.waves.has_value(), "the Realtime revision's second frame did not move the sea on the GPU");
      // Realtime back to Path Trace.
      const auto ptSwitch = next();
      const auto ptPublication = std::make_shared<int>(23);
      if (!frame(engines, ptSwitch, requestBinding, requestScene, ptPublication, at, "RT->PT first", sea, "path-trace")) return 1;
      // Raster to Realtime, the switch that was already quick.
      const auto raster = next();
      const auto rasterPublication = std::make_shared<int>(24);
      if (!frame(engines, raster, requestBinding, requestScene, rasterPublication, at, "Raster first", sea)) return 1;
      const auto rtAgain = next();
      const auto rtAgainPublication = std::make_shared<int>(25);
      if (!frame(engines, rtAgain, requestBinding, requestScene, rtAgainPublication, at, "Raster->RT first", sea)) return 1;
      check(sea.waves.has_value(), "Raster to Realtime replayed the sea on its first frame");
      std::printf("engine verifications adopted %llu, full checks %s\n",
          static_cast<unsigned long long>(engines.adoptedTimelinePlans()),
          engines.adoptedTimelinePlans() >= 5 ? "once" : "more than once");
#ifndef SEA_BENCH_ENGINE_BEFORE
      // An equal scene under a new pointer is not the scene a revision verified: it checks again
      // and names why, as the first-frame stage table reports it live.
      const auto copied = std::make_shared<const Scene>(*requestScene);
      const auto rtCopy = next();
      const auto copyPublication = std::make_shared<int>(26);
      if (!frame(engines, rtCopy, requestBinding, copied, copyPublication, at, "RT copied scene first", sea)) return 1;
      const char* reason = engines.lastAdoptionMiss();
      std::printf("adoption miss named: %s\n", reason ? reason : "none");
      check(reason != nullptr && std::strcmp(reason, "geometry:adopt-miss:scene-differs") == 0,
          "a copied scene's adoption miss was not named scene-differs");
#endif
    }
  }
  if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
  std::printf("PASS\n");
  return 0;
}
