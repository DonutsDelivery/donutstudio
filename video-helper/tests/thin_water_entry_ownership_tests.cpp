// Thin water entry ownership: raster/query precision ties at the camera hit.
//
// The Lighthouse and Ocean starter refused whole export frames because one
// water pixel's camera query did not report the water first. The raster pass
// had covered that pixel with water, but the float32 query either passed the
// water by a precision sliver (shared edge or silhouette) or ranked the rock
// face first by less than one 24-bit depth step where the two surfaces cross.
// These cases run the production CPU query twin and the C++ twin of the
// shader's thinEntryOwned rule. They need no GPU.

#include "../src/gpu_backend/fixture_smooth_glass.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
namespace ov = arbitgpu::opaquevisibility;
namespace sg = arbitgpu::smoothglass;
using Scene = HarmonicMIDI::grid::Visual3DScene;
using D3 = std::array<double, 3>;

int failures = 0, checks = 0;
void check(bool condition, const std::string& text)
{
    ++checks;
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", text.c_str()); }
}

constexpr std::uint32_t kWater = 0x1001u, kRock = 0x2002u;
// Starter camera translation. The rock front face sits where the starter's
// Nearby Rock box (centre z -1.2, depth 1.7) meets the water plane.
const ov::Point kCamera{0.0f, 2.4f, 7.0f};
constexpr float kRockFace = -0.35f;

// Water: 8x8 plane at y=0, two triangles, normal +y. Rock: vertical face in
// z=kRockFace that crosses the water along y=0, like the starter's rock side.
std::shared_ptr<Scene> scene()
{
    auto s = std::make_shared<Scene>();
    const std::array<ov::Point, 8> points{{
        {-4, 0, -4}, {4, 0, -4}, {4, 0, 4}, {-4, 0, 4},
        {-2, -1, kRockFace}, {2, -1, kRockFace}, {2, 1, kRockFace}, {-2, 1, kRockFace}}};
    for (std::size_t i = 0; i < points.size(); ++i) {
        s->vertices[i].position = {points[i][0], points[i][1], points[i][2]};
        s->vertices[i].normal = i < 4 ? HarmonicMIDI::grid::SceneVec3{0, 1, 0} : HarmonicMIDI::grid::SceneVec3{0, 0, 1};
        s->vertices[i].color = {1, 1, 1, 1};
    }
    const std::array<std::uint32_t, 12> indices{{0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3}};
    for (std::size_t i = 0; i < indices.size(); ++i) s->indices[i] = indices[i];
    s->vertexCount = 8; s->indexCount = 12;
    s->objects[0].id = {kWater}; s->objects[0].material = {7};
    s->objects[0].firstVertex = 0; s->objects[0].vertexCount = 4;
    s->objects[0].firstIndex = 0; s->objects[0].indexCount = 6;
    s->objects[1].id = {kRock}; s->objects[1].material = {8};
    s->objects[1].firstVertex = 4; s->objects[1].vertexCount = 4;
    s->objects[1].firstIndex = 6; s->objects[1].indexCount = 6;
    s->objectCount = 2;
    s->materials[0].id = {7}; s->materials[1].id = {8}; s->materialCount = 2;
    return s;
}

struct Query
{
    ov::Prepared prepared;
    ov::Frame frame;
};
bool prepare(const Scene& value, Query& query)
{
    std::string error;
    const ov::Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const bool ok = ov::prepareScene(value, nullptr, {}, query.prepared, error, ov::AdmissionMode::allSurfaces)
        && ov::frame(query.prepared, {{identity, 0}, {identity, 1}}, query.frame, error);
    check(ok, "thin water ownership scene query prepares: " + error);
    return ok;
}
ov::ClosestHit cameraHit(const Query& query, const ov::Point& entry)
{
    const ov::Point direction{entry[0]-kCamera[0], entry[1]-kCamera[1], entry[2]-kCamera[2]};
    return ov::closestHitCpu(query.prepared, query.frame, kCamera, direction, 0.0f, 3.402823e38f);
}
float cameraDistance(const ov::Point& entry)
{
    const float dx = entry[0]-kCamera[0], dy = entry[1]-kCamera[1], dz = entry[2]-kCamera[2];
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}

D3 unit(D3 v) { const double n = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return {v[0]/n, v[1]/n, v[2]/n}; }
double dot(D3 a, D3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

// Binary64 one-interface water result from an owned entry: unpolarized
// Fresnel, mirror reflection and Snell transmission, each traced once with the
// CPU query. Misses take a constant sky and rock hits a constant emission. A
// secondary water hit is the shader's THIN_REENTRY refusal. This matches the
// smooth thinInterfaceTransport structure, not its lighting.
struct WaterResult { D3 rgb{}; bool complete = false; };
WaterResult waterFrom(const Query& query, const ov::Point& entry, double ior)
{
    WaterResult result;
    const D3 incoming = unit({double(entry[0])-kCamera[0], double(entry[1])-kCamera[1], double(entry[2])-kCamera[2]});
    const D3 normal{0, 1, 0};
    const double cosine = -dot(incoming, normal), ratio = 1.0/ior;
    const double sineSquared = ratio*ratio*(1-cosine*cosine);
    if (!(cosine > 0) || sineSquared > 1) return result;
    const double exitCosine = std::sqrt(1-sineSquared);
    const double s = (cosine-ior*exitCosine)/(cosine+ior*exitCosine);
    const double p = (ior*cosine-exitCosine)/(ior*cosine+exitCosine);
    const double reflectance = 0.5*(s*s+p*p);
    D3 reflected{}, transmitted{};
    for (int c = 0; c < 3; ++c) {
        reflected[c] = incoming[c]+2*cosine*normal[c];
        transmitted[c] = ratio*incoming[c]+(ratio*cosine-exitCosine)*normal[c];
    }
    const float bias = std::max(1.0e-5f, 2.0e-6f*std::max(std::max(std::abs(entry[0]), std::abs(entry[1])), std::abs(entry[2])));
    const D3 sky{0.2, 0.4, 0.8}, rock{1.0, 0.5, 0.25};
    bool complete = true;
    const auto radiance = [&](const D3& direction, float side) {
        const ov::Point origin{entry[0], entry[1]+side*bias, entry[2]};
        const ov::Point ray{float(direction[0]), float(direction[1]), float(direction[2])};
        const auto hit = ov::closestHitCpu(query.prepared, query.frame, origin, ray, 0.0f, 3.402823e38f);
        if (hit.status == 0) return sky;
        if (hit.status != 1 || hit.objectId == kWater) { complete = false; return D3{}; }
        return rock;
    };
    const auto up = radiance(reflected, 1.0f), down = radiance(transmitted, -1.0f);
    for (int c = 0; c < 3; ++c) result.rgb[c] = reflectance*up[c]+(1-reflectance)*down[c];
    result.complete = complete && std::isfinite(result.rgb[0]) && std::isfinite(result.rgb[1]) && std::isfinite(result.rgb[2]);
    return result;
}

// Water just behind the rock face where they cross. The face is nearer by
// about 1e-5, under one 24-bit depth step at this range (about 3.6e-5), so
// the raster can cover the pixel with water while the query reports rock.
void rockDepthTie()
{
    auto value = scene();
    Query query;
    if (!prepare(*value, query)) return;
    const ov::Point entry{0.3f, 0.0f, kRockFace-1.0e-5f};
    const auto first = cameraHit(query, entry);
    // This is the released refusal: the strict identity test rejected it.
    check(first.status == 1 && first.objectId == kRock && first.distance < cameraDistance(entry),
          "depth-tie camera query ranks the rock face before the water entry");
    check(sg::thinEntryOwned(first, kWater, kCamera, entry),
          "depth-tie water entry keeps raster ownership within the tolerance");
    const auto water = waterFrom(query, entry, 1.333);
    check(water.complete, "depth-tie water pixel completes one-interface transport");
    check(water.rgb[0] > 0 && water.rgb[1] > 0 && water.rgb[2] > 0, "depth-tie water pixel has finite nonzero radiance");
}

// Grazing view past the water's far edge by 2e-5. At this range and 11 degree
// grazing angle one 720-row pixel spans about 0.09 along z, so the sliver is
// under 1/4000 of a pixel: the fixed-point raster can cover it while the float
// query misses everything.
void silhouetteMiss()
{
    auto value = scene();
    Query query;
    if (!prepare(*value, query)) return;
    const ov::Point entry{3.5f, 0.0f, -4.0f-2.0e-5f};
    const auto first = cameraHit(query, entry);
    check(first.status == 0, "grazing camera query misses the water silhouette sliver");
    check(sg::thinEntryOwned(first, kWater, kCamera, entry), "silhouette sliver keeps raster ownership");
    // The entry lies on the water plane, so the interface result is defined.
    const auto water = waterFrom(query, entry, 1.333);
    check(water.complete, "silhouette sliver completes one-interface transport");
}

// A query that passed between two water triangles and reached a farther
// surface leaves the raster entry in front of it, so the entry stands.
void sharedEdgeLeak()
{
    const ov::Point entry{0.0f, 0.0f, 0.0f};
    ov::ClosestHit behind;
    behind.status = 1; behind.objectId = kRock; behind.distance = cameraDistance(entry)+0.5f;
    check(sg::thinEntryOwned(behind, kWater, kCamera, entry), "shared-edge leak to a farther surface keeps raster ownership");
}

// A surface clearly in front of the entry is a real ownership mismatch.
void realOccluderRefuses()
{
    auto value = scene();
    Query query;
    if (!prepare(*value, query)) return;
    const ov::Point entry{0.3f, 0.0f, kRockFace-0.05f};
    const auto first = cameraHit(query, entry);
    check(first.status == 1 && first.objectId == kRock, "occluded water entry is behind the rock face");
    check(!sg::thinEntryOwned(first, kWater, kCamera, entry), "rock clearly in front of the water still refuses");
    ov::ClosestHit refused; refused.status = -2;
    check(!sg::thinEntryOwned(refused, kWater, kCamera, entry), "refused camera query never grants ownership");
    ov::ClosestHit water; water.status = 1; water.objectId = kWater; water.distance = cameraDistance(entry);
    check(sg::thinEntryOwned(water, kWater, kCamera, entry), "a confirmed water hit keeps ownership");
    ov::ClosestHit nonfinite; nonfinite.status = 1; nonfinite.objectId = kRock;
    nonfinite.distance = std::numeric_limits<float>::quiet_NaN();
    check(!sg::thinEntryOwned(nonfinite, kWater, kCamera, entry), "nonfinite foreign hit distance refuses");
}

// A forced refusal names its reason, pixel count and first texel.
void forcedRefusalNamesReasonAndTexel()
{
    arbitgpu::NativeRawPassPixels normal{arbitgpu::NativeTexturePixelFormat::Rgba16Float, 4, 3,
        std::vector<std::uint8_t>(4*3*8)};
    const std::uint16_t complete = 0x3c00u;
    for (std::size_t texel = 0; texel < 12; ++texel) std::memcpy(normal.bytes.data()+texel*8+6, &complete, 2);
    std::string error;
    check(sg::completeFrame(normal, error), "complete water normal readback is accepted: " + error);
    const std::uint16_t firstHit = sg::negativeHalfBits(int(sg::SmoothTransportStatus::ThinFirstHit));
    const std::uint16_t reentry = sg::negativeHalfBits(int(sg::SmoothTransportStatus::ThinReentry));
    std::memcpy(normal.bytes.data()+(1*4+2)*8+6, &firstHit, 2);
    std::memcpy(normal.bytes.data()+(2*4+3)*8+6, &reentry, 2);
    const bool accepted = sg::completeFrame(normal, error);
    check(!accepted, "forced thin water refusal is not published");
    check(error.find("(first hit is not the water surface; 1 pixel; also water receiver ray re-entered the water surface; 1 pixel)")
              != std::string::npos, "forced refusal names every reason and count: " + error);
    check(error.find("first at texel x=2 y=1 of the 4x3 readback") != std::string::npos,
          "forced refusal names the first failing texel: " + error);
    check(error.find("no reason recorded") == std::string::npos, "forced refusal records a reason: " + error);
}
} // namespace

int main()
{
    rockDepthTie();
    silhouetteMiss();
    sharedEdgeLeak();
    realOccluderRefuses();
    forcedRefusalNamesReasonAndTexel();
    std::printf("thin water entry ownership: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
