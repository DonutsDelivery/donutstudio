#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"
#include "../../shared/GeometryTimelineSceneBinding.h"
#include "gpu_backend/fixture_analytic_sky.h"
#include "gpu_backend/fixture_realtime_opengl.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>

// The Lighthouse and Ocean sea, lowered as the starter lowers it (a 1024 by 1024
// Grid at spacing 0.25, Wind Spectrum at amplitude 3.5, speed 1.5, heading 110, then
// Shade Smooth) and replayed on timeline frames. Every vertex normal must be finite,
// unit length and on the water's upper side. A water shading sample at each vertex,
// seen from the starter's camera under its sky, must be finite and non-negative:
// Realtime's viewer-facing normal from the vertex normal and its grid facet, which
// must face the camera from above the water, then Fresnel, the mirror direction,
// the analytic sky's radiance there, and Realtime's footprint roughness from the
// normal's change to its grid neighbours.
namespace {
using namespace videowire::geometry;
using Vec3 = std::array<float, 3>;

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool finite(const Vec3& v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }
float dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The starter turns the grid -90 degrees about X, so the grid's +Z is world up.
Vec3 world(float x, float y, float z) { return {x, z, -y}; }

PortContract seaContract() {
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = kMaximumVertices;
  contract.maxIndices = kMaximumIndices;
  contract.maxAttributes = kMaximumAttributes;
  return contract;
}

arbitgpu::analyticsky::Uniforms starterSky() {
  HarmonicMIDI::grid::SceneLightRecord light;
  light.kind = HarmonicMIDI::grid::SceneLightKind::Environment;
  light.intensity = 1.0f;
  auto& s = light.sky;
  s.version = 1;
  s.sunAzimuthDegrees = 172;
  s.sunElevationDegrees = 12;
  s.sunRadiusDegrees = 2;
  s.sunIntensity = 40;
  s.zenith = {0.03f, 0.1f, 1.76f};
  s.horizon = {0.71f, 1.11f, 2.47f};
  s.ground = {0.02f, 0.12f, 0.12f};
  s.sunColor = {1.0f, 0.38f, 0.06f};
  s.cloudCoverage = 0.4f;
  s.cloudType = 0.35f;
  s.cloudDrift = 1.45f;
  s.cloudHeadingDegrees = 350;
  return arbitgpu::analyticsky::uniforms(light);
}

void footprintRoughnessBounds() {
  using arbitgpu::realtimeengine::footprintRoughness;
  const Vec3 still {0, 0, 0}, turn {0.3f, 0, 0.1f};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  check(footprintRoughness(0.0f, still, still) == 0.0f, "still water keeps roughness 0");
  check(std::abs(footprintRoughness(0.4f, still, still) - 0.4f) < 1e-6f, "still water keeps its own roughness");
  const float widened = footprintRoughness(0.0f, turn, turn);
  check(widened > 0.0f && widened < 1.0f, "a turning normal widens the lobe");
  const float bound = footprintRoughness(0.0f, {nan, 0, 0}, still);
  check(std::isfinite(bound) && bound > 0.0f && bound <= 1.0f, "a NaN derivative takes the bound");
  check(footprintRoughness(0.0f, {1e30f, 0, 0}, still) == bound, "a huge derivative takes the same bound");
}

void viewerNormalCases() {
  using arbitgpu::realtimeengine::viewerNormal;
  const float r = std::sqrt(0.5f);
  // A grazing view of a far wave slope: the facet faces the camera, the normal does not.
  const Vec3 view {0, 0.05f / std::sqrt(1.0025f), 1 / std::sqrt(1.0025f)};
  const Vec3 away {0, r, -r};
  const Vec3 tilted = viewerNormal(away, {0, 1, -0.02f}, view);
  check(dot(tilted, view) > 0.0f && dot(tilted, view) < 0.01f, "a turned-away normal tilts just past the view");
  check(tilted[1] > 0.0f, "the tilted normal stays above the water");
  check(dot(viewerNormal(away, {0, -1, 0.02f}, view), Vec3 {0, 1, 0}) > 0.0f, "the facet's sign does not matter");
  const Vec3 facing {0, 1, 0};
  check(viewerNormal(facing, {0, 1, 0}, view) == facing, "a facing normal is kept");
  // Seen from under the water, the side flips with the facet.
  const Vec3 below {0, -0.05f / std::sqrt(1.0025f), 1 / std::sqrt(1.0025f)};
  check(viewerNormal(facing, {0, 1, 0}, below)[1] < 0.0f, "from below the normal turns down");
  check(viewerNormal(facing, {0, 0, 0}, below)[1] < 0.0f, "an unknown facet takes the view's side");
}

void seaFrames() {
  constexpr std::uint32_t side = 1024;
  std::string error;
  const auto contract = seaContract();
  const auto grid = lowerGrid(7, 7, 1, side, side, 0.25f, contract, error);
  check(grid.has_value(), "grid lowers");
  if (!grid) return;
  const auto wave = lowerWindSpectrumSurfaceWave(*grid, 9, (StableId{1} << 40) + 9 * 16, 3.5, 1.5, 110, contract, error);
  check(wave.has_value(), "wind spectrum lowers");
  if (!wave) return;
  const auto shaded = lowerShadeEdit(*wave, 11, 1.0f, 30.0f, {}, error);
  check(shaded.has_value(), "shade smooth lowers");
  if (!shaded) return;
  const auto admittedContract = withAttributeContract(contract, *shaded);
  const auto admitted = admitValue(*shaded, admittedContract, error);
  check(admitted.has_value(), "sea admits");
  if (!admitted) return;

  using namespace HarmonicMIDI::grid;
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
  check(appendGeometryToScene(*admitted, object, material, *scene, error), "sea appends");
  TimelineSceneBinding binding;
  binding.object = object.id;
  binding.loweredPlan = encodeLoweredPlanText(lowerRuntimePlan(admittedContract, *admitted, {}, {}, true));
  const auto decoded = decodeTimelineScenePlan(binding, error);
  check(decoded.has_value(), "plan decodes");
  if (!decoded || !validTimelineSceneBinding(binding, *scene, *decoded, error)) {
    check(false, "binding validates");
    return;
  }
  check(scene->vertexCount >= std::size_t(side) * side, "grid vertices are in the scene");
  if (scene->vertexCount < std::size_t(side) * side) return;

  const auto sky = starterSky();
  const Vec3 camera {0.0f, 0.7f, 5.1f};
  // Frame 30, the probe's frame, then a later and an earlier one, each replayed.
  const double times[] {1.0, 7.3, 0.0};
  for (int f = 0; f < 3; ++f) {
    check(replaceTimelineSceneMesh(binding, *decoded, times[f], *scene, error, nullptr, f > 0), "frame replays");
    std::size_t badNormal = 0, downward = 0, badShade = 0, backFacing = 0, intoSea = 0, badFacing = 0, seen = 0;
    for (std::uint32_t y = 0; y < side; ++y)
      for (std::uint32_t x = 0; x < side; ++x) {
        const auto& v = scene->vertices[std::size_t(y) * side + x];
        const Vec3 n = world(v.normal.x, v.normal.y, v.normal.z);
        const float length = std::sqrt(dot(n, n));
        if (!finite(n) || !(std::abs(length - 1.0f) < 1e-3f)) { ++badNormal; continue; }
        if (!(n[1] > 0.0f)) ++downward;
        // Shade one vertex in 16 by 16 against the camera and the sky.
        if (x % 16 || y % 16 || x + 1 >= side || y + 1 >= side) continue;
        const Vec3 p = world(v.position.x, v.position.y, v.position.z);
        Vec3 view {camera[0] - p[0], camera[1] - p[1], camera[2] - p[2]};
        const float distance = std::sqrt(dot(view, view));
        for (auto& c : view) c /= distance;
        // The facet a-b-c the grid splits from this vertex's cell, b at x+1 and c at y+1.
        const auto& b = scene->vertices[std::size_t(y) * side + x + 1].position;
        const auto& c = scene->vertices[std::size_t(y + 1) * side + x].position;
        const Vec3 e1 = world(b.x - v.position.x, b.y - v.position.y, b.z - v.position.z);
        const Vec3 e2 = world(c.x - v.position.x, c.y - v.position.y, c.z - v.position.z);
        const Vec3 facet {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        // A camera above a height field sees a facet only from its upper side, so a
        // facet whose upper side turns away is hidden behind a nearer crest.
        if (!(dot(facet, view) * facet[1] > 0.0f)) continue;
        ++seen;
        if (dot(n, view) < 0.0f) ++backFacing;
        // The old choice negated a normal that turned away, into the sea.
        if (dot(n, view) < 0.0f && -n[1] < 0.0f) ++intoSea;
        const Vec3 shading = arbitgpu::realtimeengine::viewerNormal(n, facet, view);
        const float facing = dot(shading, view);
        if (!finite(shading) || !(facing > 0.0f) || !(shading[1] > 0.0f)
            || !(std::abs(std::sqrt(dot(shading, shading)) - 1.0f) < 1e-3f))
          ++badFacing;
        const float cosine = std::clamp(facing, 0.0f, 1.0f);
        const float fresnel = 0.02f + 0.98f * std::pow(1.0f - cosine, 5.0f);
        const Vec3 mirror {2 * facing * shading[0] - view[0], 2 * facing * shading[1] - view[1],
            2 * facing * shading[2] - view[2]};
        const auto radiance = arbitgpu::analyticsky::radiance(mirror, sky);
        const auto& right = scene->vertices[std::size_t(y) * side + x + 1].normal;
        const auto& up = scene->vertices[std::size_t(y + 1) * side + x].normal;
        const Vec3 dx = world(right.x - v.normal.x, right.y - v.normal.y, right.z - v.normal.z);
        const Vec3 dy = world(up.x - v.normal.x, up.y - v.normal.y, up.z - v.normal.z);
        const float rough = arbitgpu::realtimeengine::footprintRoughness(0.0f, dx, dy);
        const Vec3 shade {fresnel * radiance[0], fresnel * radiance[1], fresnel * radiance[2]};
        if (!finite(shade) || !finite(mirror) || shade[0] < 0 || shade[1] < 0 || shade[2] < 0
            || !std::isfinite(rough) || rough < 0.0f || rough > 1.0f)
          ++badShade;
      }
    std::printf("t %.1f: %zu bad normals, %zu downward; of %zu shading samples on seen facets %zu bad,"
        " %zu with the normal turned away (%zu the old choice pointed into the sea), %zu bad shading normals\n",
        times[f], badNormal, downward, seen, badShade, backFacing, intoSea, badFacing);
    check(badNormal == 0, "every sea normal is finite and unit length");
    check(downward == 0, "every sea normal points up out of the water");
    check(badShade == 0, "every water shading sample is finite and non-negative");
    check(badFacing == 0, "every shading normal faces the camera from the water's upper side");
  }
}
} // namespace

int main() {
  footprintRoughnessBounds();
  viewerNormalCases();
  seaFrames();
  if (failures == 0) std::printf("sea water normal shading tests passed\n");
  return failures == 0 ? 0 : 1;
}
