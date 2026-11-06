#include "SceneAovOperationContract.h"
#include "fixture_scene.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>

// The vertex roughness lane: SceneVertex grows to 18 floats, a scene with no
// lane weight encodes exactly as before, and a weighted lane survives the
// scene wire so Raster and Path Trace both read it.
namespace {
using namespace HarmonicMIDI::grid;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

void layout() {
  static_assert(sizeof(SceneVertex) == 18 * sizeof(float), "SceneVertex is 18 floats");
  const SceneVertex vertex;
  check(vertex.roughness[0] == 0.0f && vertex.roughness[1] == 0.0f, "a new vertex has no roughness weight");
}

void wire() {
  auto scene = std::make_unique<Visual3DScene>();
  *scene = videohelper::fixture3d::makeScene();
  const auto plain = sceneaov::detail::encodeScene(*scene);
  check(!plain.empty(), "the fixture scene encodes");
  const auto magic = sceneaov::detail::kVertexRoughnessExtensionMagic;
  bool tagged = false;
  for (std::size_t i = 0; i + 4 <= plain.size(); ++i)
    tagged |= plain[i] == (magic & 0xff) && plain[i + 1] == ((magic >> 8) & 0xff)
        && plain[i + 2] == ((magic >> 16) & 0xff) && plain[i + 3] == (magic >> 24);
  check(!tagged, "a scene without lane weight writes no roughness extension");

  for (std::size_t i = 0; i < scene->vertexCount; ++i) scene->vertices[i].roughness = {0.125f * float(i % 8), 1.0f};
  const auto weighted = sceneaov::detail::encodeScene(*scene);
  check(weighted.size() == plain.size() + 8 + 8 * scene->vertexCount, "the lane adds one tagged record per vertex");
  auto decoded = std::make_unique<Visual3DScene>();
  check(sceneaov::detail::decodeScene(weighted, *decoded), "the weighted scene decodes");
  bool same = decoded->vertexCount == scene->vertexCount;
  for (std::size_t i = 0; same && i < scene->vertexCount; ++i)
    same = decoded->vertices[i].roughness == scene->vertices[i].roughness;
  check(same, "the roughness lane round trips");

  auto old = std::make_unique<Visual3DScene>();
  check(sceneaov::detail::decodeScene(plain, *old) && old->vertexCount > 0 && old->vertices[0].roughness[1] == 0.0f,
        "an old scene decodes with no roughness weight");

  // Extension 17 follows the orthographic camera's extension 12 on the wire.
  check(scene->cameraCount > 0, "the fixture scene has a camera");
  if (scene->cameraCount > 0) {
    scene->cameras[0].orthographicHeight = 4.0f;
    const auto both = sceneaov::detail::encodeScene(*scene);
    const auto find = [&](std::uint32_t tag) {
      for (std::size_t i = 0; i + 4 <= both.size(); ++i)
        if (both[i] == (tag & 0xff) && both[i + 1] == ((tag >> 8) & 0xff) && both[i + 2] == ((tag >> 16) & 0xff)
            && both[i + 3] == (tag >> 24))
          return i;
      return both.size();
    };
    check(find(sceneaov::detail::kOrthographicCameraExtensionMagic) < find(magic) && find(magic) < both.size(),
          "the roughness trailer comes after the orthographic trailer");
    auto decodedBoth = std::make_unique<Visual3DScene>();
    check(sceneaov::detail::decodeScene(both, *decodedBoth) && decodedBoth->cameras[0].orthographicHeight == 4.0f
              && decodedBoth->vertices[1].roughness == scene->vertices[1].roughness,
          "an orthographic scene with a roughness lane round trips");
    scene->cameras[0].orthographicHeight = 0.0f;
  }

  scene->vertices[0].roughness = {0.5f, std::numeric_limits<float>::quiet_NaN()};
  check(sceneaov::detail::encodeScene(*scene).empty(), "a non-finite lane fails validation");
}
} // namespace

int main() {
  layout();
  wire();
  if (failures != 0) return 1;
  std::puts("scene vertex roughness tests passed");
  return 0;
}
