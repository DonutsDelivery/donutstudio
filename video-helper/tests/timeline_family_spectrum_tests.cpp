#include "timeline_scene_family_evaluation_fixture.h"
#include "../src/block_b_defs.h"

#include <cstdio>

namespace {
using namespace timelinefamilyfixture;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

PortContract meshContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D; contract.maxVertices = 256; contract.maxIndices = 384;
  contract.maxAttributes = 32;
  return contract;
}

PortContract vectorContract() {
  PortContract contract;
  contract.carrier = CarrierKind::field; contract.fieldDomain = Domain::vertex;
  contract.fieldValueType = ValueType::vector; contract.fieldInterpolation = Interpolation::constant;
  contract.maxFieldElements = 256; contract.maxAttributes = 32;
  return contract;
}

// An Audio Spectrum Field on the source's vertices drives a Set Position Z offset.
std::optional<ValueDescriptor> spectrumTail(const ValueDescriptor& source, StableId id, std::uint32_t depth,
    std::string& error) {
  FieldOperationSettings settings;
  settings.domain = Domain::vertex; settings.valueType = ValueType::vector; settings.mode = 0;
  settings.values = {0, 1, arbitblockb::kBandLoHz, arbitblockb::kBandHiHz, 0, 0, 0, 0};
  spectrum::Binding binding;
  binding.firstBand = 2; binding.lastBand = 5; binding.direction = {0, 0, 0.35f};
  if (depth > 1) {
    binding.history.enabled = true; binding.history.depth = depth; binding.history.spacingSeconds = 0.25f;
    settings.values[4] = 0; settings.values[5] = -1; settings.values[6] = 1;
  }
  auto field = lowerAudioSpectrumField(source, id, binding, settings, vectorContract(), error);
  return field ? lowerVectorDisplacement(source, *field, id + 1, 1, withAttributeContract(meshContract(), source), error)
               : std::nullopt;
}

std::shared_ptr<const AdmittedTimelineScenePlans> admit(GeometryCorePlanRuntime& runtime, const ValueDescriptor& value,
    std::shared_ptr<Visual3DScene>& scene, std::vector<TimelineSceneBinding>& bindings, std::string& error) {
  const auto contract = withAttributeContract(meshContract(), value);
  const auto admitted = admitValue(value, contract, error);
  if (!admitted) return nullptr;
  scene = std::make_shared<Visual3DScene>();
  scene->id = {7}; scene->activeCamera = {50};
  scene->cameras[0].id = {50}; scene->cameras[0].transform.translation.z = 4; scene->cameraCount = 1;
  SceneObjectRecord object; object.id = {10};
  SceneMaterialRecord material; material.id = {20}; material.baseColor = {0.2f, 0.4f, 1};
  if (!appendGeometryToScene(*admitted, object, material, *scene, error) || !validateVisual3DScene(*scene).valid())
    return nullptr;
  bindings = {{{10}, encodeLoweredPlanText(lowerRuntimePlan(contract, *admitted))}};
  const PlanOwnerIdentity owner{41, 42, 43, 44, 45, PlanUse::preview};
  runtime.reconcileGeneration(41, 42, 43, error);
  return runtime.admitTimelineScenePreview(owner, *scene, bindings, {}, error);
}

// The family census counts each Audio Spectrum Field operation from sizes alone:
// one source mesh copy, one field initialization per history row and fixed
// element and component visits. A history of four rows adds three rows of
// initializations and nothing else.
void censusCountsSpectrumWork() {
  std::string error;
  const auto grid = lowerGrid(19000, 19000, 1, 3, 3, 1, meshContract(), error);
  const auto single = grid ? spectrumTail(*grid, 19010, 1, error) : std::nullopt;
  const auto rows = grid ? spectrumTail(*grid, 19010, 4, error) : std::nullopt;
  check(single && rows, "a 3 by 3 grid lowers an Audio Spectrum Field displacement with and without history: " + error);
  if (!single || !rows) return;
  Backend backend;
  GeometryCorePlanRuntime singleRuntime(backend, 16), rowsRuntime(backend, 16);
  std::shared_ptr<Visual3DScene> singleScene, rowsScene;
  std::vector<TimelineSceneBinding> singleBindings, rowsBindings;
  const auto singleFamily = admit(singleRuntime, *single, singleScene, singleBindings, error);
  const auto rowsFamily = admit(rowsRuntime, *rows, rowsScene, rowsBindings, error);
  check(singleFamily && rowsFamily, "both spectrum meshes admit as timeline families: " + error);
  if (!singleFamily || !rowsFamily) return;

  timelineevaluation::Envelope refused; std::size_t peak = 0;
  check(!timelineevaluation::childSchedule(singleFamily->children().front(), refused, peak, error, false, false)
          && error.find("reactive-source") != std::string::npos,
      "without a frame spectrum hook the census refuses spectrum bindings");

  timelineevaluation::Envelope one, four; std::size_t onePeak = 0, fourPeak = 0;
  check(timelineevaluation::childSchedule(singleFamily->children().front(), one, onePeak, error, false, true),
      "the census schedules a single-row Audio Spectrum Field: " + error);
  check(timelineevaluation::childSchedule(rowsFamily->children().front(), four, fourPeak, error, false, true),
      "the census schedules a four-row Spectral History Field: " + error);
  constexpr std::size_t vertices = 9;
  check(four.fieldInitializations - one.fieldInitializations == vertices * 3 * 3,
      "three more history rows add three initialization passes per vertex: "
          + std::to_string(one.fieldInitializations) + " -> " + std::to_string(four.fieldInitializations));
  check(four.fieldElementVisits == one.fieldElementVisits && four.fieldComponentVisits == one.fieldComponentVisits
          && four.meshCopyElements == one.meshCopyElements && four.displacementVisits == one.displacementVisits,
      "history depth changes only the field initialization count");
  check(one.fieldComponentVisits >= vertices * 3 * 8 && one.fieldElementVisits >= vertices * 3 * 2
          && one.meshCopyElements >= vertices * 2 * 3 + 24 * 3,
      "the spectrum operation's own element, component and mesh copy work is in the census");
}

// The same 9 by 9 body-coupled Water, optionally followed by an audio spectrum
// Z offset, with the parent record the coupled fixtures use.
std::optional<Source> waterSource(bool withSpectrum, std::string& error) {
  auto baseline = source(error);
  if (!baseline) return std::nullopt;
  const auto geometry = meshContract();
  auto grid = lowerGrid(18000, 18000, 1, 9, 9, .25f, geometry, error);
  auto water = grid ? lowerWaterSurfaceSimulation(*grid, 18001, 18002, 0, 0, 0, 0, geometry, error, 100) : std::nullopt;
  if (water && withSpectrum) water = spectrumTail(*water, 18010, 1, error);
  if (!water) return std::nullopt;
  const auto contract = withAttributeContract(geometry, *water);
  auto admitted = admitValue(*water, contract, error);
  if (!admitted) return std::nullopt;
  auto scene = std::make_shared<Visual3DScene>(*baseline->scene);
  scene->objectCount = scene->materialCount = scene->vertexCount = scene->indexCount = 0;
  if (!appendGeometryToScene(*admitted, baseline->scene->objects[0], baseline->scene->materials[0], *scene, error))
    return std::nullopt;
  scene->objects[scene->objectCount++] = baseline->scene->objects[3];
  if (!validateVisual3DScene(*scene).valid()) return std::nullopt;
  return Source{scene, {{{10}, encodeLoweredPlanText(lowerRuntimePlan(contract, *admitted))}}};
}

SpectrumFrameBinder bands(float level) {
  return [level](RuntimeFieldEvaluation& evaluation, const ValueDescriptor&, std::string&) {
    evaluation.spectrumAt = [level](const spectrum::Binding&, spectrum::Bands& values, std::string&) {
      values.fill(level); return true;
    };
    return true;
  };
}

// The app's caller passes the clip-local frame time as the source clock, project
// time for the bodies and the water, and the layer's frame spectrum hook. A Water
// child with an audio spectrum tail solves the same surface as plain Water, and the
// hook moves only the drawn mesh.
void callerPathCarriesSpectrumOverCoupledWater() {
  using namespace videorender;
  std::string error;
  auto plain = waterSource(false, error);
  auto tail = plain ? waterSource(true, error) : std::nullopt;
  check(plain && tail, "plain and spectrum-tail coupled Water sources author: " + error);
  if (!plain || !tail) return;
  Backend backend;
  GeometryCorePlanRuntime plainRuntime(backend, 16), tailRuntime(backend, 16);
  const PlanOwnerIdentity owner{31, 32, 33, 34, 35, PlanUse::preview};
  plainRuntime.reconcileGeneration(31, 32, 33, error);
  tailRuntime.reconcileGeneration(31, 32, 33, error);
  auto plainFamily = plainRuntime.admitTimelineScenePreview(owner, *plain->scene, plain->bindings, {}, error);
  auto tailFamily = tailRuntime.admitTimelineScenePreview(owner, *tail->scene, tail->bindings, {}, error);
  check(plainFamily && tailFamily, "both coupled Water families admit: " + error);
  if (!plainFamily || !tailFamily) return;

  auto audio = waterreplayfixture::audio(2);
  auto history = std::make_shared<ParticleHistorySource>(waterreplayfixture::source());
  auto score = std::make_shared<arbitmod::Score>(*history->score);
  score->notes.push_back(waterreplayfixture::note(81, 100)); history->score = score;
  history->audioFrames = &audio; history->audioSampleRate = 61440;
  constexpr double dt = 1.0 / 120, project = 3 * dt, clip = dt;
  constexpr float aspect = 640.0f / 360;
  auto plainParameters = coupledParameters(*plain, history, error);
  auto tailParameters = coupledParameters(*tail, history, error);
  plainParameters.historyProjectSeconds = tailParameters.historyProjectSeconds = project;
  const auto frame = waterreplayfixture::frame(*history, project, 30, false);

  TimelineSceneFamilyEvaluator plainEvaluator(plainRuntime), quietEvaluator(tailRuntime), loudEvaluator(tailRuntime);
  const auto reference = plainEvaluator.evaluateCoupled(owner, plainFamily, plain->scene, plainParameters,
      clip, project, project, aspect, frame, {}, {}, error);
  check(reference.scene && reference.replay, "plain coupled Water evaluates at a clip time apart from project time: " + error);
  const auto quiet = quietEvaluator.evaluateCoupled(owner, tailFamily, tail->scene, tailParameters,
      clip, project, project, aspect, frame, {}, {}, error, nullptr, nullptr, bands(0));
  check(quiet.scene && quiet.replay, "the caller path evaluates spectrum-tail coupled Water with its frame hook: " + error);
  const auto loud = loudEvaluator.evaluateCoupled(owner, tailFamily, tail->scene, tailParameters,
      clip, project, project, aspect, frame, {}, {}, error, nullptr, nullptr, bands(0.8f));
  check(loud.scene && loud.replay, "a louder frame evaluates on the same path: " + error);
  TimelineSceneFamilyEvaluator silentEvaluator(tailRuntime);
  check(!silentEvaluator.evaluateCoupled(owner, tailFamily, tail->scene, tailParameters,
            clip, project, project, aspect, frame, {}, {}, error).scene,
      "spectrum-tail Water without the frame hook is refused");
  if (!reference.scene || !quiet.scene || !loud.scene) return;

  check(reference.replay->water()->heights == quiet.replay->water()->heights
          && quiet.replay->water()->heights == loud.replay->water()->heights,
      "bodies and water solve the same surface whatever the spectrum tail reads");
  check(std::any_of(reference.replay->water()->heights.begin(), reference.replay->water()->heights.end(),
            [](double h) { return h != 0; }),
      "the body has already moved the water at project frame three");
  const auto vertices = reference.scene->vertexCount;
  bool quietMatches = vertices == quiet.scene->vertexCount && vertices == loud.scene->vertexCount;
  bool loudRaised = quietMatches;
  for (std::size_t i = 0; quietMatches && i < vertices; ++i) {
    const auto& a = reference.scene->vertices[i].position;
    const auto& b = quiet.scene->vertices[i].position;
    const auto& c = loud.scene->vertices[i].position;
    quietMatches = a.x == b.x && a.y == b.y && a.z == b.z;
    loudRaised = loudRaised && c.x == a.x && c.y == a.y && c.z > a.z;
  }
  check(quietMatches, "silent bands leave the spectrum-tail surface exactly on the plain coupled surface");
  check(loudRaised, "loud bands lift every drawn Water vertex along Z without moving it in X or Y");
}
}  // namespace

int main() {
  censusCountsSpectrumWork();
  callerPathCarriesSpectrumOverCoupledWater();
  if (failures != 0) return 1;
  std::puts("timeline family spectrum tests passed");
  return 0;
}
