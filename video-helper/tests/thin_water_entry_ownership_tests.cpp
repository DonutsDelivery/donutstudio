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
#include "../src/gpu_backend/fixture_analytic_sky.h"
#include "../../shared/SurfaceMaterialPreviewLayout.h"
#include <algorithm>
#include <array>
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

// A surface clearly in front of the entry fails the strict rule. The shader
// then probes past it (probedOccluderStillRefuses covers that for a closed
// occluder; this open face has no thickness for the probe to measure).
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

// The starter's Nearby Rock as a closed box (centre -1.2,0.05,-1.2, size
// 1.9x0.65x1.7) over a 12x12 grid of spacing 0.7, seen from the starter camera.
// Every face is wound so the query's cross(e1,e2) points outward.
constexpr float kRockLow[3]{-2.15f, -0.275f, -2.05f}, kRockHigh[3]{-0.25f, 0.375f, -0.35f};
std::shared_ptr<Scene> rockBoxScene()
{
    auto s = std::make_shared<Scene>();
    std::vector<ov::Point> points{{-4.2f, 0, -4.2f}, {4.2f, 0, -4.2f}, {4.2f, 0, 4.2f}, {-4.2f, 0, 4.2f}};
    std::vector<std::uint32_t> indices{0, 2, 1, 0, 3, 2};
    for (int axis = 0; axis < 3; ++axis)
        for (int side = 0; side < 2; ++side) {
            const int u = (axis+1)%3, v = (axis+2)%3;
            std::array<ov::Point, 4> quad{};
            const float uv[4][2]{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            for (int corner = 0; corner < 4; ++corner) {
                quad[corner][axis] = side ? kRockHigh[axis] : kRockLow[axis];
                quad[corner][u] = uv[corner][0] > 0 ? kRockHigh[u] : kRockLow[u];
                quad[corner][v] = uv[corner][1] > 0 ? kRockHigh[v] : kRockLow[v];
            }
            // cross(u,v) is +axis for this corner order; the low side flips.
            if (!side) std::swap(quad[1], quad[3]);
            // Mesh indices are local to the object's firstVertex (the rock starts
            // at vertex 4), so the face's local base is its global base minus 4.
            const auto base = std::uint32_t(points.size()) - 4u;
            points.insert(points.end(), quad.begin(), quad.end());
            for (std::uint32_t index : {0u, 1u, 2u, 0u, 2u, 3u}) indices.push_back(base+index);
        }
    for (std::size_t i = 0; i < points.size(); ++i) {
        s->vertices[i].position = {points[i][0], points[i][1], points[i][2]};
        s->vertices[i].normal = {0, 1, 0};
        s->vertices[i].color = {1, 1, 1, 1};
    }
    for (std::size_t i = 0; i < indices.size(); ++i) s->indices[i] = indices[i];
    s->vertexCount = points.size(); s->indexCount = indices.size();
    s->objects[0].id = {kWater}; s->objects[0].material = {7};
    s->objects[0].firstVertex = 0; s->objects[0].vertexCount = 4;
    s->objects[0].firstIndex = 0; s->objects[0].indexCount = 6;
    s->objects[1].id = {kRock}; s->objects[1].material = {8};
    s->objects[1].firstVertex = 4; s->objects[1].vertexCount = std::uint32_t(points.size()-4);
    s->objects[1].firstIndex = 6; s->objects[1].indexCount = std::uint32_t(indices.size()-6);
    s->objectCount = 2;
    s->materials[0].id = {7}; s->materials[1].id = {8}; s->materialCount = 2;
    return s;
}

// Water entry whose camera ray crosses the rock's top face offset inside the
// x=-2.15 silhouette edge by inset, at the height where the Metal export's
// failing texel 654,560 (1920x1080, 52 degree field of view) grazes it. One
// pixel there spans about 0.0076 world units, and that texel's centre lies
// about 1e-4 inside the edge.
ov::Point grazingEntry(double inset)
{
    const double edge[3]{kRockLow[0]+inset, kRockHigh[1], -0.5255};
    const double scale = double(kCamera[1])/(double(kCamera[1])-edge[1]);
    return {float(kCamera[0]+(edge[0]-kCamera[0])*scale), 0.0f, float(kCamera[2]+(edge[2]-kCamera[2])*scale)};
}

struct Probe { bool owned = false, strict = false, probed = false; ov::ClosestHit first; };
Probe probe(const Query& query, const ov::Point& entry, const ov::Point& direction)
{
    Probe result;
    result.first = ov::closestHitCpu(query.prepared, query.frame, kCamera, direction, 0.0f, 3.402823e38f);
    result.strict = sg::thinEntryOwned(result.first, kWater, kCamera, entry);
    result.owned = sg::thinEntryOwnedPastSliver(result.first, kWater, kCamera, entry, [&](float minimum) {
        result.probed = true;
        return ov::closestHitCpu(query.prepared, query.frame, kCamera, direction, minimum, 3.402823e38f);
    });
    return result;
}
ov::Point toward(const ov::Point& entry)
{
    return {entry[0]-kCamera[0], entry[1]-kCamera[1], entry[2]-kCamera[2]};
}

// The Metal refusal: the raster left the pixel to the water while the query
// clipped a sliver of the rock's top edge 1.5 units in front of it. The strict
// rule refuses; the probe one tolerance past the rock reaches the water.
void occluderSilhouetteSliver()
{
    auto value = rockBoxScene();
    Query query;
    if (!prepare(*value, query)) return;
    const auto entry = grazingEntry(1.0e-4);
    check(entry[0] < kRockLow[0] && entry[2] > kRockLow[2] && entry[2] < kRockHigh[2],
          "grazing water entry lies beside the rock, not under it");
    const auto result = probe(query, entry, toward(entry));
    check(result.first.status == 1 && result.first.objectId == kRock && result.first.entering
              && result.first.distance < cameraDistance(entry)-1.0f,
          "grazing camera query clips the rock's top edge well in front of the water");
    check(!result.strict, "the strict rule refuses the rock silhouette sliver (released Metal refusal)");
    check(result.probed && result.owned, "the probe past the sliver keeps raster ownership");
    const auto water = waterFrom(query, entry, 1.333);
    check(water.complete, "the sliver water pixel completes one-interface transport");
}

// Contraction or fast math moves the query's ray and triangle arithmetic by a
// few ulps, so the side of the edge the pixel centre lands on is a coin toss
// near the tie. Every inset from 2e-4 outside to 1.5e-4 inside the edge, with
// each direction component nudged by up to four ulps either way, keeps
// ownership. At least one of them is the strict rule's refusal. In a float32
// replay one nudged ray misses the rock's top face and catches only its exit
// face, so the probe must not depend on which way the foreign hit faces.
void perturbedRoundingAtTheEdge()
{
    auto value = rockBoxScene();
    Query query;
    if (!prepare(*value, query)) return;
    int strictRefusals = 0, cases = 0, refused = 0;
    for (double inset : {-2.0e-4, -1.0e-4, -1.0e-5, -1.0e-6, 0.0, 1.0e-6, 1.0e-5, 5.0e-5, 1.0e-4, 1.5e-4}) {
        const auto entry = grazingEntry(inset);
        for (int axis = 0; axis < 3; ++axis)
            for (int ulps = -4; ulps <= 4; ++ulps) {
                auto direction = toward(entry);
                for (int step = 0; step < std::abs(ulps); ++step)
                    direction[axis] = std::nextafter(direction[axis], ulps < 0 ? -1.0e9f : 1.0e9f);
                const auto result = probe(query, entry, direction);
                ++cases;
                if (!result.strict) ++strictRefusals;
                if (!result.owned) {
                    ++refused;
                    std::fprintf(stderr, "refused inset %.1e axis %d ulps %d: status %d object %u distance %.7g\n",
                                 inset, axis, ulps, result.first.status, unsigned(result.first.objectId),
                                 double(result.first.distance));
                }
            }
    }
    check(refused == 0, "every perturbed edge tie keeps ownership (" + std::to_string(refused) + " of "
              + std::to_string(cases) + " refused)");
    check(strictRefusals > 0, "the strict rule refuses some perturbed edge ties");
}

// The probe never hides a closed occluder. Inside the rock by more than the
// tolerance it meets the rock's own exit face in front of the water. A
// refused probe refuses.
void probedOccluderStillRefuses()
{
    auto value = rockBoxScene();
    Query query;
    if (!prepare(*value, query)) return;
    for (double inset : {1.0e-3, 0.05, 0.2}) {
        const auto entry = grazingEntry(inset);
        const auto result = probe(query, entry, toward(entry));
        check(result.first.status == 1 && result.first.objectId == kRock && result.probed && !result.owned,
              "rock " + std::to_string(inset) + " inside its silhouette still refuses after the probe");
    }
    const auto entry = grazingEntry(1.0e-4);
    ov::ClosestHit front;
    front.status = 1; front.objectId = kRock; front.entering = true;
    front.distance = cameraDistance(entry)-1.0f;
    check(!sg::thinEntryOwnedPastSliver(front, kWater, kCamera, entry,
              [](float) { ov::ClosestHit budget; budget.status = -2; return budget; }),
          "a refused probe refuses ownership");
    ov::ClosestHit refused; refused.status = -1;
    bool probed = false;
    check(!sg::thinEntryOwnedPastSliver(refused, kWater, kCamera, entry,
              [&](float) { probed = true; ov::ClosestHit miss; miss.status = 0; return miss; }) && !probed,
          "a refused camera query refuses without a probe");
    check(sg::kMaximumThinPixelRayQueries/sg::kThinPixelRayQueries >= std::uint64_t(3840)*2160,
          "the probe's query charge still admits a 3840x2160 thin water frame");
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

// Smooth Dispersive Glass on the default 3840x2160 canvas. The closed volume
// frame cap admits 4K and still refuses a larger frame with its limit.
void closedGlassFrameBudget()
{
    sg::Prepared glass;
    glass.enabled = true;
    glass.transport = surfacematerial::TransportMode::ClosedVolume;
    std::string error;
    check(sg::admitFrame(glass, 3840, 2160, false, error),
          "a smooth closed glass frame at 3840x2160 is admitted: " + error);
    check(std::uint64_t(3840)*2160*(3*sg::kMaximumRayVisits+1) <= sg::kMaximumClosedGlassFrameRayQueries
              && sg::kMaximumClosedGlassFrameRayQueries/(3*sg::kMaximumRayVisits+1) == 11126858u,
          "the closed glass frame cap names its pixel limit");
    check(!sg::admitFrame(glass, 4096, 4096, false, error)
              && error == "Closed glass transport pixel ray-query budget: frame 4096x4096 is 16777216 pixels and the limit is 11126858",
          "a closed glass frame above the cap still refuses: " + error);
    check(!sg::admitFrame(glass, 1920, 1080, true, error)
              && error == "Closed glass transport refuses dynamic instances",
          "closed glass still refuses dynamic instances: " + error);
}

// Bright finite radiance is stored at the RGBA16F ceiling by the shader, so a
// readback of 65504 passes. Infinity and NaN still refuse the frame, and the
// message names the value kind, the count and the first texel.
void finiteColorCeiling()
{
    const auto colour = [](std::uint16_t bits) {
        arbitgpu::NativeRawPassPixels pixels{arbitgpu::NativeTexturePixelFormat::Rgba16Float, 2, 1,
            std::vector<std::uint8_t>(2*8)};
        const std::uint16_t one = 0x3c00u;
        for (std::size_t half = 0; half < 8; ++half) std::memcpy(pixels.bytes.data()+half*2, &one, 2);
        std::memcpy(pixels.bytes.data()+8, &bits, 2);
        return pixels;
    };
    const std::string prefix = "Closed glass scene-linear Color exceeds finite RGBA16F representation";
    std::string error;
    check(sg::finiteColorFrame(colour(0x7bffu), error), "a Color at the 65504 ceiling is accepted: " + error);
    check(!sg::finiteColorFrame(colour(0x7c00u), error)
              && error == prefix + " (infinity in channel R; 1 value) first at texel x=1 y=0 of the 2x1 readback",
          "an infinite Color still refuses: " + error);
    check(!sg::finiteColorFrame(colour(0x7e00u), error)
              && error == prefix + " (NaN in channel R; 1 value) first at texel x=1 y=0 of the 2x1 readback",
          "a NaN Color still refuses: " + error);
    arbitgpu::NativeRawPassPixels normal{arbitgpu::NativeTexturePixelFormat::Rgba16Float, 3, 1,
        std::vector<std::uint8_t>(3*8)};
    const std::uint16_t unrepresentable = 0xc000u, refused = sg::negativeHalfBits(-5);
    std::memcpy(normal.bytes.data()+6, &refused, 2);
    std::memcpy(normal.bytes.data()+8+6, &unrepresentable, 2);
    std::memcpy(normal.bytes.data()+16+6, &unrepresentable, 2);
    check(!sg::completeFrame(normal, error)
              && error == prefix + " (NaN or infinite glass radiance; 2 pixels) first at texel x=1 y=0 of the 3x1 readback",
          "nonfinite glass pixels marked -2 still refuse with a count and texel: " + error);
}

// Smooth Dispersive Glass refused every export on NVIDIA GL. Its refracted
// rays land on the starter's diffuse floor and backdrop (roughness 1), which
// glassReceiver shades with analyticSkyBase. That integral's GGX sample 3 has
// a projected disk that crosses zero at n.v = 2/(1+sqrt(3.5/64))-1, about 0.62,
// so its half vector is the view itself. In float, v.h then rounds a few ulps
// above one, and pow(1.0-v.h,5.0) of a negative base is undefined in GLSL and
// returns NaN on NVIDIA. One NaN leaf marks the pixel -2. This mirrors the
// shader's sample construction for alpha = 1.
template<class T> T skySampleViewDotHalf(T nv, T azimuth, int sample)
{
    using V = std::array<T, 3>;
    const auto dotv = [](const V& a, const V& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; };
    const auto unit = [&](V a) { const T l = std::sqrt(dotv(a, a)); for (auto& c : a) c /= l; return a; };
    const auto cross = [](const V& a, const V& b) {
        return V{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    };
    const V n{T(0), T(1), T(0)};
    const V tangent = unit(cross(n, V{T(0), T(0), T(1)}));
    const V bitangent = cross(n, tangent);
    const T side = std::sqrt(T(1)-nv*nv);
    V v{};
    for (int c = 0; c < 3; ++c)
        v[c] = tangent[c]*side*std::cos(azimuth)+bitangent[c]*side*std::sin(azimuth)+n[c]*nv;
    v = unit(v);
    const T vn = std::clamp(dotv(n, v), T(0), T(1));
    const V stretched = unit(V{dotv(v, tangent), dotv(v, bitangent), vn});
    const V t1 = stretched[2] < T(0.99999) ? unit(V{-stretched[1], stretched[0], T(0)}) : V{T(1), T(0), T(0)};
    const V t2 = cross(stretched, t1);
    T inverse = 0, scale = T(0.5);
    for (int bit = 0, i = sample; bit < 6; ++bit) { inverse += T(i%2)*scale; i /= 2; scale *= T(0.5); }
    const T u = (T(sample)+T(0.5))/T(64);
    const T phi = T(6.283185307179586)*inverse;
    T dx = std::cos(phi)*std::sqrt(u), dy = std::sin(phi)*std::sqrt(u);
    const T w = (T(1)+stretched[2])*T(0.5);
    dy = std::sqrt(std::max(T(0), T(1)-dx*dx))*(T(1)-w)+dy*w;
    V projected{};
    const T lift = std::sqrt(std::max(T(0), T(1)-(dx*dx+dy*dy)));
    for (int c = 0; c < 3; ++c) projected[c] = t1[c]*dx+t2[c]*dy+stretched[c]*lift;
    const V local = unit(V{projected[0], projected[1], std::max(T(0.000001), projected[2])});
    V h{};
    for (int c = 0; c < 3; ++c) h[c] = tangent[c]*local[0]+bitangent[c]*local[1]+n[c]*local[2];
    return dotv(v, h);
}

void smoothDispersiveGlassSkyNaN()
{
    const double r = std::sqrt(3.5/64.0);
    const double onView = 2.0/(1.0+r)-1.0;
    check(std::abs(onView-0.6208) < 0.001, "sample 3 reaches the view direction near n.v 0.62");
    check(std::abs(skySampleViewDotHalf<double>(onView, 0.7, 3)-1.0) < 1.0e-9,
          "sample 3's half vector equals the view at that n.v, so v.h sits on one");
    // Sweep float32 views around that n.v, as the starter floor and backdrop
    // see them through the glass. The clamped Schlick base never goes negative.
    std::size_t above = 0, views = 0;
    bool finite = true;
    for (int a = 0; a < 64; ++a)
        for (int step = -500; step <= 500; ++step) {
            const float nv = float(onView)+float(step)*2.0e-7f;
            const float vh = skySampleViewDotHalf<float>(nv, float(a)*0.0981747704f, 3);
            ++views;
            if (vh > 1.0f) ++above;
            const float schlick = std::pow(1.0f-std::min(vh, 1.0f), 5.0f);
            finite = finite && std::isfinite(schlick) && schlick >= 0.0f && schlick <= 1.0f;
        }
    check(finite, "the clamped Schlick term stays finite in [0,1] for every float view");
    std::printf("analytic sky sample 3: %zu of %zu float32 views round v.h above one\n", above, views);
    const auto sky = arbitgpu::analyticsky::shader(false);
    check(sky.find("float vh=min(dot(v,h),1.0);") != std::string::npos
              && sky.find("float vh=dot(v,h);") == std::string::npos,
          "the analytic sky GGX loop clamps v.h before pow()");
}
// The material starter preview stage. Smooth Dispersive Glass is a closed
// volume, and a ray inside it that meets another object refuses its pixel.
// At translationY 0.2 the rotated cube's bottom corner sat at y -0.745, below
// the floor top at -0.70, and a 1080p export refused 75,439 pixels. A CPU walk
// of the starter at 480x270 counted 4,708 such pixels (x16 is 75,328) and none
// once the cube cleared the floor. These checks keep the cube clear of every
// receiver box with a separating axis.
using Box3 = std::array<std::array<double, 3>, 8>;
Box3 previewBoxCorners(const surfacematerialpreview::Box& box)
{
    // Same Euler composition as the Visual3DTransform node.
    constexpr double halfRadians = 0.0087266462599716478846;
    const double sx = std::sin(box.rotationXDegrees*halfRadians), cx = std::cos(box.rotationXDegrees*halfRadians);
    const double sy = std::sin(box.rotationYDegrees*halfRadians), cy = std::cos(box.rotationYDegrees*halfRadians);
    const double sz = std::sin(box.rotationZDegrees*halfRadians), cz = std::cos(box.rotationZDegrees*halfRadians);
    const double qx = sx*cy*cz-cx*sy*sz, qy = cx*sy*cz+sx*cy*sz, qz = cx*cy*sz-sx*sy*cz, qw = cx*cy*cz+sx*sy*sz;
    Box3 corners{};
    for (int i = 0; i < 8; ++i) {
        const double v[3]{(i&1 ? 0.5 : -0.5)*box.width, (i&2 ? 0.5 : -0.5)*box.height, (i&4 ? 0.5 : -0.5)*box.depth};
        const double t[3]{2*(qy*v[2]-qz*v[1]), 2*(qz*v[0]-qx*v[2]), 2*(qx*v[1]-qy*v[0])};
        corners[i] = {v[0]+qw*t[0]+(qy*t[2]-qz*t[1])+box.translationX,
                      v[1]+qw*t[1]+(qz*t[0]-qx*t[2])+box.translationY,
                      v[2]+qw*t[2]+(qx*t[1]-qy*t[0])+box.translationZ};
    }
    return corners;
}
// Largest gap between the two boxes along any separating axis candidate: the
// face normals of both boxes and their pairwise cross products. Negative
// means the boxes overlap.
double previewBoxGap(const Box3& a, const Box3& b)
{
    using V = std::array<double, 3>;
    const auto sub = [](const V& p, const V& q) { return V{p[0]-q[0], p[1]-q[1], p[2]-q[2]}; };
    const auto edges = [&](const Box3& c) { return std::array<V, 3>{sub(c[1], c[0]), sub(c[2], c[0]), sub(c[4], c[0])}; };
    std::vector<V> axes;
    for (const auto& e : edges(a)) axes.push_back(e);
    for (const auto& e : edges(b)) axes.push_back(e);
    for (const auto& p : edges(a))
        for (const auto& q : edges(b)) axes.push_back({p[1]*q[2]-p[2]*q[1], p[2]*q[0]-p[0]*q[2], p[0]*q[1]-p[1]*q[0]});
    double best = -std::numeric_limits<double>::infinity();
    for (const auto& axis : axes) {
        const double length = std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
        if (length < 1.0e-9) continue;
        double aMin = 1e300, aMax = -1e300, bMin = 1e300, bMax = -1e300;
        for (const auto& p : a) { const double d = (p[0]*axis[0]+p[1]*axis[1]+p[2]*axis[2])/length; aMin = std::min(aMin, d); aMax = std::max(aMax, d); }
        for (const auto& p : b) { const double d = (p[0]*axis[0]+p[1]*axis[1]+p[2]*axis[2])/length; bMin = std::min(bMin, d); bMax = std::max(bMax, d); }
        best = std::max(best, std::max(bMin-aMax, aMin-bMax));
    }
    return best;
}
void starterGlassClearsReceivers()
{
    namespace layout = surfacematerialpreview;
    const auto object = previewBoxCorners(layout::kObject);
    double lowest = 1e300;
    for (const auto& corner : object) lowest = std::min(lowest, corner[1]);
    const double floorTop = layout::kFloor.translationY+0.5*layout::kFloor.height;
    check(lowest-floorTop >= 0.01, "the starter glass bottom corner clears the floor top by 0.01: "
          + std::to_string(lowest) + " over " + std::to_string(floorTop));
    for (const auto* receiver : {&layout::kFloor, &layout::kBackdropLeft, &layout::kBackdropRight})
        check(previewBoxGap(object, previewBoxCorners(*receiver)) >= 0.01,
              "the starter glass is clear of every preview receiver box");
    auto released = layout::kObject;
    released.translationY = 0.2;
    check(previewBoxGap(previewBoxCorners(released), previewBoxCorners(layout::kFloor)) < 0.0,
          "the released translationY 0.2 is caught as overlapping the floor");
    // The refusal says in plain words what to change.
    arbitgpu::NativeRawPassPixels normal{arbitgpu::NativeTexturePixelFormat::Rgba16Float, 1, 1,
        std::vector<std::uint8_t>(8)};
    const std::uint16_t medium = sg::negativeHalfBits(int(sg::SmoothTransportStatus::MediumMismatch));
    std::memcpy(normal.bytes.data()+6, &medium, 2);
    std::string error;
    check(!sg::completeFrame(normal, error)
              && error.find("the glass object intersects another object") != std::string::npos
              && error.find("Move or resize the glass so no object passes into the glass solid") != std::string::npos,
          "an overlapping glass refusal names the intersection: " + error);
}

// A 4K closed glass frame drawn as one submission tripped the NVIDIA context
// switch timeout (Xid 109) and came back partly drawn. The GL backend draws
// it in row bands of at most kClosedGlassBandPixels pixels.
void closedGlassBands()
{
    check(sg::closedGlassBandRows(3840) == 68 && 68u*3840u <= sg::kClosedGlassBandPixels,
          "a 3840-wide glass frame draws 68-row bands");
    check(sg::closedGlassBandRows(1920) == 136, "a 1920-wide glass frame draws 136-row bands");
    check((2160+67)/68 == 32, "a 3840x2160 glass frame takes 32 bands");
    check(sg::closedGlassBandRows(1u<<20) == 1, "a very wide frame still draws one row per band");
    check(sg::closedGlassBandRows(0) == 0, "an empty frame has no bands");
}

// The starter's 1.5 cube, axis aligned so the cases can sit a float step from
// an edge, and optionally a floor plane at y=floorY. Faces follow the
// GeometryCore cube layout, so the winding gives outward normals.
constexpr std::uint32_t kGlass = 0x3003u, kFloorPlane = 0x4004u;
std::shared_ptr<Scene> seamScene(bool floor, float floorY)
{
    auto s = std::make_shared<Scene>();
    constexpr std::array<std::array<int, 4>, 6> faces{{{{0, 2, 3, 1}}, {{4, 5, 7, 6}}, {{0, 1, 5, 4}},
                                                       {{2, 6, 7, 3}}, {{0, 4, 6, 2}}, {{1, 3, 7, 5}}}};
    std::size_t v = 0, i = 0;
    const auto corner = [](int c) {
        return HarmonicMIDI::grid::SceneVec3{c & 1 ? 0.75f : -0.75f, c & 2 ? 0.75f : -0.75f, c & 4 ? 0.75f : -0.75f};
    };
    for (const auto& face : faces) {
        const auto first = std::uint32_t(v);
        for (const auto c : face) {
            s->vertices[v].position = corner(c);
            s->vertices[v].normal = {0, 1, 0};
            s->vertices[v].color = {1, 1, 1, 1};
            ++v;
        }
        for (const auto index : {first, first+1, first+2, first, first+2, first+3}) s->indices[i++] = index;
    }
    s->objects[0].id = {kGlass}; s->objects[0].material = {7};
    s->objects[0].firstVertex = 0; s->objects[0].vertexCount = 24;
    s->objects[0].firstIndex = 0; s->objects[0].indexCount = 36;
    s->objectCount = 1;
    if (floor) {
        const std::array<ov::Point, 4> points{{{-4, floorY, -4}, {4, floorY, -4}, {4, floorY, 4}, {-4, floorY, 4}}};
        for (std::size_t p = 0; p < points.size(); ++p) {
            s->vertices[24+p].position = {points[p][0], points[p][1], points[p][2]};
            s->vertices[24+p].normal = {0, 1, 0};
            s->vertices[24+p].color = {1, 1, 1, 1};
        }
        for (const std::uint32_t index : {24u, 26u, 25u, 24u, 27u, 26u}) s->indices[i++] = index;
        s->objects[1].id = {kFloorPlane}; s->objects[1].material = {8};
        s->objects[1].firstVertex = 24; s->objects[1].vertexCount = 4;
        s->objects[1].firstIndex = 36; s->objects[1].indexCount = 6;
        s->objectCount = 2;
    }
    s->vertexCount = std::uint32_t(v + (floor ? 4 : 0)); s->indexCount = std::uint32_t(i);
    s->materials[0].id = {7}; s->materials[1].id = {8}; s->materialCount = floor ? 2 : 1;
    return s;
}
bool prepareSeam(const Scene& value, Query& query)
{
    std::string error;
    const ov::Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<ov::Instance> instances{ov::Instance{identity, 0}};
    if (value.objectCount > 1) instances.push_back(ov::Instance{identity, 1});
    const bool ok = ov::prepareScene(value, nullptr, {}, query.prepared, error, ov::AdmissionMode::allSurfaces)
        && ov::frame(query.prepared, instances, query.frame, error);
    check(ok, "seam scene query prepares: " + error);
    return ok;
}
ov::Point unitPoint(float x, float y, float z)
{
    const float n = std::sqrt(x*x+y*y+z*z);
    return {x/n, y/n, z/n};
}
// C++ twin of one glassTransport (and glassPairWalk) visit with its seam
// retries: the hit that continues the branch, the side it continues on, and
// how many retries ran. consistent is false when the shader would refuse; hit
// is then the forward hit the refusal names. back is where retry 3 looks: the
// seamSpawnBack of a spawned branch, or -direction for a seed (seamVisit).
struct SeamVisit { ov::ClosestHit hit; bool inside = false, consistent = false; int retries = 0; };
SeamVisit seamVisitFrom(const Query& query, const ov::Point& origin, const ov::Point& direction, const ov::Point& back,
                        bool inside, std::uint32_t volume = kGlass)
{
    SeamVisit visit;
    visit.inside = inside;
    for (int retry = 0; retry <= sg::kSeamRetries; ++retry) {
        const auto ray = sg::seamQuery(retry, origin, direction, back);
        const auto hit = ov::closestHitCpu(query.prepared, query.frame, ray.origin, ray.direction, ray.minimum, ray.maximum);
        // The shader refuses a refused or over-budget query before its seam logic.
        if (hit.status < 0) { visit.hit = hit; visit.retries = retry; return visit; }
        if (retry == 3) {
            // Backwards from the origin: only an escaped origin continues, and
            // it leaves the glass through the face it met.
            visit.retries = 3;
            if (!sg::seamEscaped(hit, volume, origin, ray.direction)) return visit;
            visit.hit = hit;
            visit.hit.entering = false;
            visit.consistent = true;
            return visit;
        }
        visit.hit = hit;
        visit.retries = retry;
        if (!sg::seamInconsistent(hit, volume, inside, retry)) {
            if (retry == 1) visit.inside = !hit.entering;
            visit.consistent = true;
            return visit;
        }
        if (retry == 2) {
            const bool glass = hit.status == 1 && hit.objectId == volume;
            if (!inside) return visit;
            if (glass) {
                // Met from outside: continue outside only within the escape depth.
                if (sg::seamEscaped(hit, volume, origin, ray.direction)) { visit.inside = false; visit.consistent = true; }
                return visit;
            }
        }
    }
    return visit;
}
SeamVisit seamVisit(const Query& query, const ov::Point& origin, const ov::Point& direction, bool inside,
                    std::uint32_t volume = kGlass)
{
    return seamVisitFrom(query, origin, direction, ov::Point{-direction[0], -direction[1], -direction[2]}, inside, volume);
}

// Smooth Dispersive Glass refused 1080p frames for one pixel. A ray inside the
// closed, free-standing glass met the cube within a few float steps of an
// edge. The branch spawned there sat a float step past the neighbouring face,
// so its first query met no glass (or met that face from outside). The seam
// retries find the face within half a bias of the origin and let its side
// decide, while a floor that really cuts the glass still refuses.
void glassSeamRetries()
{
    Query glass;
    if (!prepareSeam(*seamScene(false, 0.0f), glass)) return;
    const float above = 0.75f+1.0e-6f;
    // Reflected off the +x face at its top edge: one bias inside +x, a float
    // step above the top face, heading up and out through the top face.
    {
        const ov::Point origin{0.75f-1.0e-5f, above, 0.1f};
        const auto direction = unitPoint(-0.5f, 0.8f, 0.1f);
        const auto first = ov::closestHitCpu(glass.prepared, glass.frame, origin, direction, 0.0f, 3.402823e38f);
        check(first.status == 0 && sg::seamInconsistent(first, kGlass, true, 0),
              "an inside branch past the top face meets no glass on its first query");
        const auto visit = seamVisit(glass, origin, direction, true);
        check(visit.consistent && visit.retries == 1 && visit.hit.objectId == kGlass && !visit.hit.entering
                  && visit.inside && std::abs(visit.hit.position[1]-0.75f) < 1.0e-6f,
              "retry 1 finds the top face exit within half a bias of the origin, not the +x face it left");
    }
    // The same origin heading down: it meets the top face from outside a float
    // step ahead, so the branch was already outside and now enters.
    {
        const ov::Point origin{0.75f-1.0e-5f, above, 0.1f};
        const auto direction = unitPoint(-0.5f, -0.8f, 0.1f);
        const auto first = ov::closestHitCpu(glass.prepared, glass.frame, origin, direction, 0.0f, 3.402823e38f);
        check(first.status == 1 && first.objectId == kGlass && first.entering
                  && sg::seamInconsistent(first, kGlass, true, 0),
              "an inside branch above the top face meets it from outside");
        const auto visit = seamVisit(glass, origin, direction, true);
        check(visit.consistent && visit.retries == 1 && visit.hit.entering && !visit.inside,
              "retry 1 lets the top face decide that the branch enters from outside");
    }
    // A ray from the centre exactly through the +x/+y edge exits the glass,
    // whether or not the float query finds a triangle on the edge itself.
    {
        const ov::Point origin{0.0f, 0.0f, 0.0f};
        const auto direction = unitPoint(0.75f, 0.75f, 0.3f);
        const auto visit = seamVisit(glass, origin, direction, true);
        check(visit.consistent && visit.hit.objectId == kGlass && !visit.hit.entering && visit.inside,
              "a ray through a cube edge leaves the glass");
        const auto tilted = sg::seamQuery(2, origin, direction);
        const float cosine = tilted.direction[0]*direction[0]+tilted.direction[1]*direction[1]+tilted.direction[2]*direction[2];
        const float length = std::sqrt(tilted.direction[0]*tilted.direction[0]+tilted.direction[1]*tilted.direction[1]
                                       +tilted.direction[2]*tilted.direction[2]);
        check(std::abs(length-1.0f) < 1.0e-6f && cosine < 1.0f && std::acos(std::min(1.0f, cosine)) < 2.0f*sg::kSeamTilt,
              "retry 2 tilts the ray by about kSeamTilt");
    }
    // A clean exit takes no retry.
    {
        const auto visit = seamVisit(glass, ov::Point{0.0f, 0.0f, 0.0f}, unitPoint(0.2f, 1.0f, 0.1f), true);
        check(visit.consistent && visit.retries == 0 && !visit.hit.entering, "a clean exit takes no retry");
    }
    // The starter sunk to translationY 0 cuts the floor: a ray inside the glass
    // meets the floor top at -0.70 before the glass bottom at -0.75. Retries 1
    // and 2 still meet the floor, and retry 3 meets the glass top from inside,
    // so the pixel refuses.
    Query cut;
    if (!prepareSeam(*seamScene(true, -0.70f), cut)) return;
    for (const auto& direction : {unitPoint(0.0f, -1.0f, 0.0f), unitPoint(0.3f, -1.0f, 0.2f), unitPoint(-0.4f, -1.0f, -0.1f)}) {
        const auto visit = seamVisit(cut, ov::Point{0.1f, 0.0f, 0.1f}, direction, true);
        check(!visit.consistent && visit.retries == sg::kSeamRetries && visit.hit.status == 1
                  && visit.hit.objectId == kFloorPlane,
              "glass that cuts the floor still refuses after every seam retry");
    }
    // A branch spawned on the floor-cut glass near the floor stays refused too:
    // its origin is a bias inside the glass, far from any glass face's window.
    {
        const auto visit = seamVisit(cut, ov::Point{0.5f, -0.65f, 0.0f}, unitPoint(0.1f, -1.0f, 0.0f), true);
        check(!visit.consistent && visit.hit.objectId == kFloorPlane, "a branch near the cut still meets the floor");
    }
}

// The Lighthouse and Ocean lens: a sphere of radius 0.15 flattened to 0.14 in
// y, 10 segments and 6 rings, here axis aligned at the origin. Its equator is
// a crease with a 55 degree wedge between the faces above and below it.
// Optionally a plane at y=floorY (axis 1) or z=floorY (axis 2). Triangles
// wind outward.
std::shared_ptr<Scene> lensScene(bool floor, float floorY, int axis = 1)
{
    constexpr double kPi = 3.141592653589793, radius = 0.15, flatten = 0.14;
    constexpr int segments = 10, rings = 6;
    auto s = std::make_shared<Scene>();
    std::size_t v = 0, i = 0;
    for (int k = 0; k <= rings; ++k) {
        const double theta = kPi*k/rings;
        for (int j = 0; j <= segments; ++j) {
            const double phi = 2*kPi*j/segments;
            s->vertices[v].position = {float(radius*std::sin(theta)*std::cos(phi)), float(flatten*radius*std::cos(theta)),
                                       float(radius*std::sin(theta)*std::sin(phi))};
            s->vertices[v].normal = {0, 1, 0};
            s->vertices[v].color = {1, 1, 1, 1};
            ++v;
        }
    }
    const auto at = [](int k, int j) { return std::uint32_t(k*(segments+1)+j); };
    for (int k = 0; k < rings; ++k)
        for (int j = 0; j < segments; ++j) {
            const auto a = at(k, j), b = at(k+1, j), c = at(k+1, j+1), d = at(k, j+1);
            // The pole fans drop the triangle whose two corners are the pole.
            if (k != rings-1) for (const auto index : {a, c, b}) s->indices[i++] = index;
            if (k != 0) for (const auto index : {a, d, c}) s->indices[i++] = index;
        }
    s->objects[0].id = {kGlass}; s->objects[0].material = {7};
    s->objects[0].firstVertex = 0; s->objects[0].vertexCount = std::uint32_t(v);
    s->objects[0].firstIndex = 0; s->objects[0].indexCount = std::uint32_t(i);
    s->objectCount = 1;
    if (floor) {
        const auto first = std::uint32_t(v), start = std::uint32_t(i);
        const std::array<ov::Point, 4> points = axis == 2
            ? std::array<ov::Point, 4>{{{-4, -4, floorY}, {4, -4, floorY}, {4, 4, floorY}, {-4, 4, floorY}}}
            : std::array<ov::Point, 4>{{{-4, floorY, -4}, {4, floorY, -4}, {4, floorY, 4}, {-4, floorY, 4}}};
        for (const auto& point : points) {
            s->vertices[v].position = {point[0], point[1], point[2]};
            s->vertices[v].normal = {0, 1, 0};
            s->vertices[v].color = {1, 1, 1, 1};
            ++v;
        }
        for (const auto index : {first, first+2, first+1, first, first+3, first+2}) s->indices[i++] = index;
        s->objects[1].id = {kFloorPlane}; s->objects[1].material = {8};
        s->objects[1].firstVertex = first; s->objects[1].vertexCount = 4;
        s->objects[1].firstIndex = start; s->objects[1].indexCount = 6;
        s->objectCount = 2;
    }
    s->vertexCount = std::uint32_t(v); s->indexCount = std::uint32_t(i);
    s->materials[0].id = {7}; s->materials[1].id = {8}; s->materialCount = floor ? 2 : 1;
    return s;
}

// The lens refused 1080p and 4K frames for one rim pixel each, on the paired
// walk that had no seam retries. A branch spawned on the face just above the
// equator, 0.3 bias from the crease, sits one bias inside along that face's
// normal, which puts it 0.28 bias below the face beneath the crease. Heading
// along the rim, it crosses that lower face 1.4 biases behind its origin, out
// of retry 1's half-bias window, and the tilt keeps the origin. The escape
// rules recover it. A floor that really cuts the lens still refuses, and a
// camera ray that passes the rim silhouette walks on in air.
void lensRimSeams()
{
    Query lens;
    if (!prepareSeam(*lensScene(false, 0.0f), lens)) return;
    // Equator segment 0 runs from (0.15, 0, 0) to (0.15 cos 36, 0, 0.15 sin 36).
    // The origin is P - nA*bias for P on the upper face 3e-6 from its midpoint.
    const ov::Point origin{1.3566920e-01f, -7.3199753e-06f, 4.4081591e-02f};
    // Along the rim, tipped 0.2 out through the lower face.
    {
        const ov::Point direction{-0.21320829f, -0.17188539f, 0.961768f};
        const auto first = ov::closestHitCpu(lens.prepared, lens.frame, origin, direction, 0.0f, 3.402823e38f);
        check(first.status == 0, "an inside branch at the lens rim meets nothing on its first query");
        const auto window = sg::seamQuery(1, origin, direction);
        check(ov::closestHitCpu(lens.prepared, lens.frame, window.origin, window.direction, window.minimum, window.maximum).status == 0,
              "retry 1 finds no face within half a bias of the rim origin");
        const auto visit = seamVisit(lens, origin, direction, true);
        check(visit.consistent && visit.retries == 3 && visit.inside && visit.hit.objectId == kGlass
                  && !visit.hit.entering && visit.hit.position[1] < 0.0f && visit.hit.outwardNormal[1] < -0.8f,
              "retry 3 finds the lower rim face behind the origin and the branch leaves through it");
        check(visit.hit.distance*std::abs(visit.hit.outwardNormal[0]*direction[0]+visit.hit.outwardNormal[1]*direction[1]
                                          +visit.hit.outwardNormal[2]*direction[2]) < 0.5f*sg::glassBiasCpu(origin),
              "the rim origin lies under half a bias past the lower face");
    }
    // Along the rim, tipped 0.2 in through the lower face: the origin was
    // already outside, and retry 2 meets that face from outside.
    {
        const ov::Point direction{-0.39282408f, 0.17188539f, 0.9034072f};
        const auto first = ov::closestHitCpu(lens.prepared, lens.frame, origin, direction, 0.0f, 3.402823e38f);
        check(first.status == 1 && first.objectId == kGlass && first.entering && first.distance > sg::glassBiasCpu(origin),
              "the rim branch meets the lower face from outside more than a bias ahead");
        const auto visit = seamVisit(lens, origin, direction, true);
        check(visit.consistent && visit.retries == 2 && !visit.inside && visit.hit.entering,
              "retry 2 lets the escape depth decide that the rim branch enters from outside");
    }
    // The same rim branch with a wall at z 0.3, clear of the lens (its far
    // edge is at z 0.15). The forward queries meet the wall, and retry 3 still
    // finds the lower face behind the origin. The wall is not dropped as an
    // overlap: from just outside that face the same ray meets it in air, so
    // the branch's exit and its children trace it as a receiver.
    {
        Query walled;
        if (!prepareSeam(*lensScene(true, 0.3f, 2), walled)) return;
        const ov::Point direction{-0.21320829f, -0.17188539f, 0.961768f};
        const auto first = ov::closestHitCpu(walled.prepared, walled.frame, origin, direction, 0.0f, 3.402823e38f);
        check(first.status == 1 && first.objectId == kFloorPlane, "the walled rim branch meets the wall first");
        const auto visit = seamVisit(walled, origin, direction, true);
        check(visit.consistent && visit.retries == 3 && visit.inside && visit.hit.objectId == kGlass && !visit.hit.entering,
              "retry 3 still finds the lower rim face when the forward hit is another object");
        const float bias = sg::glassBiasCpu(visit.hit.position);
        ov::Point outside = visit.hit.position;
        for (int a = 0; a < 3; ++a) outside[a] += visit.hit.outwardNormal[a]*bias;
        const auto air = ov::closestHitCpu(walled.prepared, walled.frame, outside, direction, 0.0f, 3.402823e38f);
        check(air.status == 1 && air.objectId == kFloorPlane && std::abs(air.position[2]-0.3f) < 1.0e-5f,
              "past the rim face the ray meets the wall in air, so no overlap is hidden");
    }
    // A clean exit from the lens centre takes no retry.
    {
        const auto visit = seamVisit(lens, ov::Point{0.0f, 0.0f, 0.0f}, unitPoint(1.0f, 0.05f, 0.3f), true);
        check(visit.consistent && visit.retries == 0 && !visit.hit.entering, "a clean lens exit takes no retry");
    }
    // A floor at y -0.01 cuts the lens, whose bottom is at -0.021. A branch
    // inside the lens meets the floor on every forward retry and the lens top
    // from inside on retry 3, so it still refuses.
    Query cut;
    if (!prepareSeam(*lensScene(true, -0.01f), cut)) return;
    for (const auto& direction : {unitPoint(0.1f, -1.0f, 0.0f), unitPoint(0.0f, -1.0f, 0.2f), unitPoint(0.6f, -0.2f, 0.1f)}) {
        const auto visit = seamVisit(cut, ov::Point{0.05f, -0.005f, 0.0f}, direction, true);
        check(!visit.consistent && visit.retries == sg::kSeamRetries && visit.hit.status == 1
                  && visit.hit.objectId == kFloorPlane,
              "a lens that cuts the floor still refuses after every seam retry");
    }
    // The camera rule. A ray that grazes 1e-6 outside the rim crease meets
    // nothing, though the raster covered the pixel; the entry stands and the
    // walk starts in air. A surface in front of the entry still refuses.
    {
        const ov::Point entry{1.3567723e-01f, 0.0f, 4.4084202e-02f};
        const ov::Point camera{1.6807626e+00f, 0.0f, -4.7111983e+00f};
        const auto incoming = unitPoint(entry[0]-camera[0], entry[1]-camera[1], entry[2]-camera[2]);
        const auto first = ov::closestHitCpu(lens.prepared, lens.frame, camera, incoming, 0.0f, 3.402823e38f);
        check(first.status == 0 && sg::thinEntryOwned(first, kGlass, camera, entry),
              "a camera ray past the lens rim misses it and the raster entry stands");
        const float distance = std::sqrt((entry[0]-camera[0])*(entry[0]-camera[0])+(entry[2]-camera[2])*(entry[2]-camera[2]));
        ov::ClosestHit occluder;
        occluder.status = 1; occluder.objectId = kFloorPlane; occluder.distance = distance-0.01f;
        check(!sg::thinEntryOwned(occluder, kGlass, camera, entry), "a surface in front of the lens entry still refuses");
        occluder.distance = distance+0.01f;
        check(sg::thinEntryOwned(occluder, kGlass, camera, entry), "a surface behind the lens entry lets the entry stand");
    }
}

// Water reflections of the lens refused 1080p frames for one pixel (texel
// 822,931 at beat 4.0 and 4.05), and this is the gap such a ray can reach. A
// ray off the water enters the lens face just
// above the rim crease and refracts steeply inward. Its transmitted child
// starts one bias inside that face, which puts it past the face beneath the
// crease, and it heads away from that face, so the forward and tilted queries
// meet nothing. Looking back along the ray, it crosses the upper face's plane
// beyond the crease and meets nothing either, and the branch refused MEDIUM.
// Retry 3 now looks back along the face normal toward the point the child was
// spawned from. It must enter the glass before reaching that point, here
// through the lower face.
void reflectedLensRimSeams()
{
    Query lens;
    if (!prepareSeam(*lensScene(false, 0.0f), lens)) return;
    // The lensRimSeams origin: P - nA*bias for P on the upper face of equator
    // segment 0, 0.3 bias from the crease. nA is that face's outward normal
    // and nB the lower face's, from the binary64 vertex positions.
    const ov::Point origin{1.3566920e-01f, -7.3199753e-06f, 4.4081591e-02f};
    const ov::Point upper{0.45793208f, 0.87644702f, 0.1487912f};
    const ov::Point lower{0.45793224f, -0.87644693f, 0.14879126f};
    // Refracted through the upper face 26.7 degrees off its inward normal
    // (inside the 41.8 degree critical cone for IOR 1.5): d.nA -0.893,
    // d.nB +0.099.
    const ov::Point direction{-0.78399605f, -0.56609178f, -0.25473583f};
    const auto dotted = [](const ov::Point& a, const ov::Point& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; };
    const float bias = sg::glassBiasCpu(origin);
    check(bias == 1.0e-5f, "the rim origin's bias is the 1e-5 floor");
    check(dotted(direction, upper) < -0.89f && dotted(direction, lower) > 0.09f && dotted(direction, lower) < 0.11f,
          "the rim child heads in through the upper face and slowly away from the lower one");
    const auto first = ov::closestHitCpu(lens.prepared, lens.frame, origin, direction, 0.0f, 3.402823e38f);
    check(first.status == 0, "the steep rim child meets nothing on its first query");
    const auto window = sg::seamQuery(1, origin, direction);
    check(ov::closestHitCpu(lens.prepared, lens.frame, window.origin, window.direction, window.minimum, window.maximum).status == 0,
          "retry 1 finds no face within half a bias of the steep rim child");
    const auto tilted = sg::seamQuery(2, origin, direction);
    check(ov::closestHitCpu(lens.prepared, lens.frame, tilted.origin, tilted.direction, 0.0f, 3.402823e38f).status == 0,
          "the tilted retry meets nothing either");
    // The earlier retry 3, back along the ray, is the seed form.
    const auto alongRay = sg::seamQuery(3, origin, direction);
    check(ov::closestHitCpu(lens.prepared, lens.frame, alongRay.origin, alongRay.direction, 0.0f, 3.402823e38f).status == 0,
          "back along the ray the rim child leaves the upper plane beyond the crease and meets nothing");
    check(!seamVisit(lens, origin, direction, true).consistent, "the seed form of retry 3 would refuse the rim child");
    // The transmitted child of an entry: the shader's normal there is the
    // outward normal, and the child looks back along it.
    const auto back = sg::seamSpawnBack(upper, true);
    check(back[0] == upper[0] && back[1] == upper[1] && back[2] == upper[2],
          "a transmitted child looks back along the entry normal");
    const auto reflectedBack = sg::seamSpawnBack(upper, false);
    check(reflectedBack[0] == -upper[0] && reflectedBack[1] == -upper[1] && reflectedBack[2] == -upper[2],
          "a reflected child looks back against the normal");
    const auto visit = seamVisitFrom(lens, origin, direction, back, true);
    check(visit.consistent && visit.retries == 3 && visit.inside && visit.hit.objectId == kGlass && !visit.hit.entering
              && visit.hit.outwardNormal[1] < -0.87f && visit.hit.position[1] < 0.0f,
          "retry 3 toward the spawning face finds the lower rim face and the child leaves through it");
    // The binary64 geometry puts the origin 0.283 bias outside the lower
    // face's plane, which the back ray reaches after 0.528 bias.
    check(std::abs(visit.hit.distance/bias-0.528f) < 0.05f, "the back ray meets the lower face 0.53 bias from the origin");
    check(std::abs(visit.hit.distance*std::abs(dotted(back, visit.hit.outwardNormal))/bias-0.283f) < 0.05f,
          "the rim child sits 0.28 bias past the lower face, inside the escape depth");
    // The earlier rim branch also resolves when it looks back to its face.
    {
        const ov::Point along{-0.21320829f, -0.17188539f, 0.961768f};
        const auto earlier = seamVisitFrom(lens, origin, along, back, true);
        check(earlier.consistent && earlier.retries == 3 && earlier.hit.objectId == kGlass && !earlier.hit.entering
                  && earlier.hit.outwardNormal[1] < -0.87f,
              "the rim branch along the crease also leaves through the lower face");
    }
    // A child that is really inside meets glass from inside when it looks
    // back, so a lens that cuts a floor still refuses. Here the child sits
    // deep in the lens and looks back up toward its top face, as a child
    // reflected off that face would.
    Query cut;
    if (!prepareSeam(*lensScene(true, -0.01f), cut)) return;
    for (const auto& towardFloor : {unitPoint(0.1f, -1.0f, 0.0f), unitPoint(0.0f, -1.0f, 0.2f)}) {
        const auto refused = seamVisitFrom(cut, ov::Point{0.05f, -0.005f, 0.0f}, towardFloor, unitPoint(0.0f, 1.0f, 0.0f), true);
        check(!refused.consistent && refused.retries == sg::kSeamRetries && refused.hit.status == 1
                  && refused.hit.objectId == kFloorPlane,
              "a lens that cuts the floor still refuses when retry 3 looks back to the spawning face");
    }
}
} // namespace

int main()
{
    rockDepthTie();
    silhouetteMiss();
    sharedEdgeLeak();
    realOccluderRefuses();
    occluderSilhouetteSliver();
    perturbedRoundingAtTheEdge();
    probedOccluderStillRefuses();
    forcedRefusalNamesReasonAndTexel();
    closedGlassFrameBudget();
    finiteColorCeiling();
    smoothDispersiveGlassSkyNaN();
    starterGlassClearsReceivers();
    closedGlassBands();
    glassSeamRetries();
    lensRimSeams();
    reflectedLensRimSeams();
    std::printf("thin water entry ownership: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
