#include "../../shared/GeometryCoreScene.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

// Subdivision Surface dicing in a composed scene (Render node, Path Trace):
// the cage is measured in the active camera's space, object transform
// included, and the scene gets the same replayed mesh as the native frame.
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

bool approx(double a, double b) { return std::abs(a - b) < 1e-5; }

PortContract meshContract() {
  PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 200000; contract.maxIndices = 1200000; contract.maxAttributes = 8;
  return contract;
}

void cameraSpace() {
  auto scene = std::make_unique<Visual3DScene>();
  SceneObjectRecord object; object.id = {1};
  SceneCameraRecord camera; camera.id = {1};
  camera.transform.translation = {0, 0, 5};
  const auto plain = sceneLocalToCamera(*scene, object, camera);
  const std::array<double, 12> expected{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -5}};
  bool same = true;
  for (std::size_t i = 0; i < 12; ++i) same = same && approx(plain[i], expected[i]);
  check(same, "a camera at +5 z looking down -z moves the object 5 m ahead");

  // A camera turned 90 degrees about +Y looks down world -X. From (5, 0, 0) the
  // object origin sits 5 m ahead, and object scale 2 doubles its x axis.
  const double half = std::sqrt(0.5);
  camera.transform.translation = {5, 0, 0};
  camera.transform.rotation = {0, static_cast<float>(half), 0, static_cast<float>(half)};
  object.transform.scale = {2, 2, 2};
  const auto turned = sceneLocalToCamera(*scene, object, camera);
  check(approx(turned[3], 0) && approx(turned[7], 0) && approx(turned[11], -5), "the turned camera sees the origin 5 m ahead");
  check(approx(turned[0], 0) && approx(turned[2], -2) && approx(turned[8], 2),
        "the turned camera maps object x to camera depth with the object's scale");
}

void composed() {
  std::string error;
  const auto contract = meshContract();
  auto grid = lowerGrid(60201, 60201, 1, 3, 3, 1.0f, contract, error);
  check(bool(grid), ("subdivision cage lowers: " + error).c_str());
  if (!grid) return;
  auto adaptive = lowerSubdivisionSurface(*grid, 60202, 1, 1, {}, false, contract, error, 16.0f);
  auto fine = lowerSubdivisionSurface(*grid, 60203, 4, 4, {}, false, contract, error);
  check(adaptive && fine, ("the adaptive and four-level meshes lower: " + error).c_str());
  if (!adaptive || !fine) return;
  const auto fineIndices = std::get<GeometryData>(fine->data).indices.size();
  auto admitted = admitValue(std::move(*adaptive), contract, {}, error);
  check(bool(admitted), ("the adaptive mesh admits: " + error).c_str());
  if (!admitted) return;

  const auto compose = [&](float distance, float fovRadians) -> std::size_t {
    auto scene = std::make_unique<Visual3DScene>();
    scene->cameras[0].id = {7};
    scene->cameras[0].transform.translation = {0, 0, distance};
    scene->cameras[0].verticalFovRadians = fovRadians;
    scene->cameraCount = 1;
    scene->activeCamera = {7};
    SceneObjectRecord object; object.id = {1}; object.material = {1};
    SceneMaterialRecord material; material.id = {1};
    std::string composeError;
    if (!appendGeometryToScene(*admitted, object, material, *scene, composeError)) {
      std::fprintf(stderr, "compose: %s\n", composeError.c_str());
      return 0;
    }
    return scene->indexCount;
  };
  // At 1080 lines with a 45 degree view, 10 m away covers the same pixels as
  // 5 m on the 540-line native frame: four levels.
  check(compose(10, 0.7853981634f) == fineIndices, "the composed scene dices from its camera");
  check(compose(40, 0.7853981634f) < fineIndices, "a farther camera dices coarser");
}
} // namespace

int main() {
  cameraSpace();
  composed();
  if (failures != 0) return 1;
  std::puts("geometry composed dicing tests passed");
  return 0;
}
