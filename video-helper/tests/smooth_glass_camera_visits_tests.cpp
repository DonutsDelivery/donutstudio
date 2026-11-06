#include "../src/gpu_backend/fixture_smooth_glass.h"
#include "SceneAovOperationContract.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// Smooth-glass camera walk visits authored per scene: scene wire extension 19
// carries the count, cameraVisitsFor reads it, and the transport text and the
// program key follow it. A scene on the default keeps its exact bytes.
namespace {
using namespace HarmonicMIDI::grid;
namespace sg = arbitgpu::smoothglass;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

std::unique_ptr<Visual3DScene> emptyScene() {
  auto scene = std::make_unique<Visual3DScene>();
  scene->id.value = 7;
  return scene;
}

bool tagged(const std::vector<std::uint8_t>& bytes) {
  const auto magic = sceneaov::detail::kSmoothGlassCameraVisitsExtensionMagic;
  for (std::size_t i = 0; i + 4 <= bytes.size(); ++i)
    if (bytes[i] == (magic & 0xff) && bytes[i + 1] == ((magic >> 8) & 0xff)
        && bytes[i + 2] == ((magic >> 16) & 0xff) && bytes[i + 3] == (magic >> 24))
      return true;
  return false;
}

void defaults() {
  const auto scene = emptyScene();
  check(sg::cameraVisitsFor(*scene) == sg::kDefaultCameraVisits, "an unauthored scene walks the default visits");
  check(sg::kDefaultCameraVisits == 64, "the default walk is the earlier fixed 64 visits");
  check(!tagged(sceneaov::detail::encodeScene(*scene)), "a scene on the default writes no visits extension");
}

void wire() {
  auto scene = emptyScene();
  const auto plain = sceneaov::detail::encodeScene(*scene);
  scene->smoothGlassCameraVisits = 24;
  const auto authored = sceneaov::detail::encodeScene(*scene);
  check(authored.size() == plain.size() + 12, "the visits extension is one magic, one version and one count");
  check(std::vector<std::uint8_t>(authored.begin(), authored.begin() + plain.size()) == plain,
        "the extension appends after the unchanged scene bytes");
  auto decoded = std::make_unique<Visual3DScene>();
  check(sceneaov::detail::decodeScene(authored, *decoded) && decoded->smoothGlassCameraVisits == 24,
        "the authored visits survive the scene wire");
  check(sg::cameraVisitsFor(*decoded) == 24, "the decoded scene walks its authored visits");
  auto old = std::make_unique<Visual3DScene>();
  check(sceneaov::detail::decodeScene(plain, *old) && old->smoothGlassCameraVisits == 0,
        "an old scene without the extension decodes to the default");
  auto zero = plain;
  for (const std::uint32_t word : {sceneaov::detail::kSmoothGlassCameraVisitsExtensionMagic, 1u, 0u})
    for (unsigned shift = 0; shift != 32; shift += 8) zero.push_back(std::uint8_t(word >> shift));
  check(!sceneaov::detail::decodeScene(zero, *decoded), "a written visits extension must carry a count");
}

void program() {
  auto scene = emptyScene();
  scene->smoothGlassCameraVisits = 200;
  const auto text = sg::transportShader(false, true, false, false, sg::cameraVisitsFor(*scene));
  check(text.find("#define GLASS_CAMERA_VISITS 200\n") != std::string::npos,
        "the transport text walks the scene's visits");
  check(sg::transportShader(false, true, false, false, sg::cameraVisitsFor(*emptyScene()))
            .find("#define GLASS_CAMERA_VISITS 64\n") != std::string::npos,
        "a default scene keeps the 64-visit text");
}
}  // namespace

int main() {
  defaults();
  wire();
  program();
  if (failures != 0) return 1;
  std::puts("smooth glass camera visits tests passed");
  return 0;
}
