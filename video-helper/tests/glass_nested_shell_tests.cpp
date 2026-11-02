// Nested glass shells and opaque objects beside them (stage 1 of nested
// dielectrics). One closed glass object may be several closed shells: the
// outer shell wound outward, each cavity shell wound inward inside it. Opaque
// objects may sit in air outside the glass or in a cavity, and may touch the
// glass, but must not pass into the glass solid. The single inside/outside
// parity of the walks then crosses outer glass, air cavity and inner glass.
// These cases run CPU admission (smoothglass::prepare), the CPU scene query
// and a binary64 branch walk that follows the shader's parity and interface
// rules. They need no GPU.

#include "../src/gpu_backend/fixture_smooth_glass.h"
#include "../src/gpu_backend/fixture_glass_overdraw_probe.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace ov = arbitgpu::opaquevisibility;
namespace sg = arbitgpu::smoothglass;
using namespace HarmonicMIDI::grid;
using Program = arbitgpu::NativeFixtureSurfaceMaterialProgram;

int failures = 0, checks = 0;
void check(bool condition, const std::string& text)
{
    ++checks;
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", text.c_str()); }
}

constexpr std::uint32_t kGlassId = 0x3003u, kGlassMaterial = 7;
constexpr double kIor = 1.5;

// Scenes are large, so they live on the heap.
std::shared_ptr<Visual3DScene> newScene()
{
    auto scene = std::make_shared<Visual3DScene>();
    scene->id = {7};
    return scene;
}

std::size_t beginObject(Visual3DScene& scene, std::uint32_t id, std::uint32_t material)
{
    const auto o = scene.objectCount++;
    auto& object = scene.objects[o];
    object = {};
    object.id = {id};
    object.material = {material};
    object.firstVertex = std::uint32_t(scene.vertexCount);
    object.firstIndex = std::uint32_t(scene.indexCount);
    auto& m = scene.materials[scene.materialCount++];
    m = {};
    m.id = {material};
    m.baseColor = {0.8f, 0.8f, 0.8f};
    m.metallic = 0;
    m.roughness = 1;
    m.opacity = 1;
    return o;
}
void endObject(Visual3DScene& scene, std::size_t o)
{
    auto& object = scene.objects[o];
    object.vertexCount = std::uint32_t(scene.vertexCount) - object.firstVertex;
    object.indexCount = std::uint32_t(scene.indexCount) - object.firstIndex;
}
std::uint32_t vertex(Visual3DScene& scene, std::size_t o, SceneVec3 p, SceneVec3 n = {0, 1, 0})
{
    auto& v = scene.vertices[scene.vertexCount++];
    v = {};
    v.position = p;
    v.normal = n;
    return std::uint32_t(scene.vertexCount - 1 - scene.objects[o].firstVertex);
}
void triangle(Visual3DScene& scene, std::uint32_t a, std::uint32_t b, std::uint32_t c)
{
    scene.indices[scene.indexCount++] = a;
    scene.indices[scene.indexCount++] = b;
    scene.indices[scene.indexCount++] = c;
}

// One closed cube shell of the current glass object: eight shared corners,
// wound outward, or inward for a cavity. Corner bit 0 is x, bit 1 y, bit 2 z.
void glassCube(Visual3DScene& scene, std::size_t o, float half, SceneVec3 centre, bool inward)
{
    std::array<std::uint32_t, 8> corners {};
    for (int c = 0; c < 8; ++c)
        corners[c] = vertex(scene, o, {centre.x + (c & 1 ? half : -half), centre.y + (c & 2 ? half : -half),
                                       centre.z + (c & 4 ? half : -half)});
    constexpr std::array<std::array<int, 4>, 6> faces {{{{0, 2, 3, 1}}, {{4, 5, 7, 6}}, {{0, 1, 5, 4}},
                                                        {{2, 6, 7, 3}}, {{0, 4, 6, 2}}, {{1, 3, 7, 5}}}};
    for (const auto& f : faces)
    {
        if (inward)
        {
            triangle(scene, corners[f[0]], corners[f[2]], corners[f[1]]);
            triangle(scene, corners[f[0]], corners[f[3]], corners[f[2]]);
        }
        else
        {
            triangle(scene, corners[f[0]], corners[f[1]], corners[f[2]]);
            triangle(scene, corners[f[0]], corners[f[2]], corners[f[3]]);
        }
    }
}

// An opaque box receiver with flat geometric normals (24 vertices).
void receiverBox(Visual3DScene& scene, std::uint32_t id, std::uint32_t material, SceneVec3 low, SceneVec3 high)
{
    const auto o = beginObject(scene, id, material);
    struct Face { float n[3], u[3], v[3]; };
    const Face faces[6] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}};
    const float centre[3] = {(low.x + high.x) / 2, (low.y + high.y) / 2, (low.z + high.z) / 2};
    const float half[3] = {(high.x - low.x) / 2, (high.y - low.y) / 2, (high.z - low.z) / 2};
    const float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (const auto& f : faces)
    {
        std::uint32_t at[4];
        for (int c = 0; c < 4; ++c)
        {
            float p[3];
            for (int a = 0; a < 3; ++a)
                p[a] = centre[a] + half[a] * (f.n[a] + corners[c][0] * f.u[a] + corners[c][1] * f.v[a]);
            at[c] = vertex(scene, o, {p[0], p[1], p[2]}, {f.n[0], f.n[1], f.n[2]});
        }
        triangle(scene, at[0], at[1], at[2]);
        triangle(scene, at[0], at[2], at[3]);
    }
    endObject(scene, o);
}

// An opaque quad over x0..x1 and z0..z1, facing up, with its height a linear
// function of x: y0 at x0 and y1 at x1. Every vertex carries the quad's flat
// geometric normal, which admission requires of a receiver, so a tilted quad
// reaches the interpenetration test instead of failing on its normals.
void opaqueQuad(Visual3DScene& scene, std::uint32_t id, std::uint32_t material, float x0, float x1, float z0, float z1,
                float y0, float y1)
{
    const auto o = beginObject(scene, id, material);
    // cross(b - a, c - a) with b - a = (0, 0, dz) and c - a = (dx, dy, dz).
    const double dx = double(x1) - x0, dy = double(y1) - y0, dz = double(z1) - z0;
    const double nx = -dz * dy, ny = dz * dx, length = std::sqrt(nx * nx + ny * ny);
    const SceneVec3 n {float(nx / length), float(ny / length), 0};
    const auto a = vertex(scene, o, {x0, y0, z0}, n), b = vertex(scene, o, {x0, y0, z1}, n);
    const auto c = vertex(scene, o, {x1, y1, z1}, n), d = vertex(scene, o, {x1, y1, z0}, n);
    triangle(scene, a, b, c);
    triangle(scene, a, c, d);
    endObject(scene, o);
}

// The glass Surface Material program, the scene's root.
std::shared_ptr<Program> glassProgram(const Visual3DScene& scene)
{
    auto p = std::make_shared<Program>();
    p->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    p->object = {kGlassId};
    p->bindingDigest = "nested-shell-binding";
    p->programIdentity = "nested-shell-program";
    p->authoredScene = scene.id;
    p->authoredMaterial = {kGlassMaterial};
    p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
    p->parameters.identifiers[0] = kGlassMaterial;
    p->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ClosedVolume);
    p->parameters.normalOpacity = {0, 0, 1, 1};
    p->parameters.baseColorMetallic = {1, 1, 1, 0};
    p->parameters.transmissionIorClearcoat = {1, float(kIor), 0, 0};
    p->parameters.emissionRoughness = {0, 0, 0, 0};
    return p;
}

enum class Glass { Solid, Hollow, HollowOutwardCavity, Intersecting, ThreeOutward, Siblings, Open, Inverted, Bowl };

// The glass object first, then the receivers the caller adds. Hollow is a
// cube of half 1 with a cube cavity of half 0.6, both at the origin.
std::size_t addGlass(Visual3DScene& scene, Glass shape)
{
    const auto o = beginObject(scene, kGlassId, kGlassMaterial);
    switch (shape)
    {
        case Glass::Solid: glassCube(scene, o, 1, {0, 0, 0}, false); break;
        case Glass::Hollow: glassCube(scene, o, 1, {0, 0, 0}, false); glassCube(scene, o, 0.6f, {0, 0, 0}, true); break;
        case Glass::HollowOutwardCavity:
            glassCube(scene, o, 1, {0, 0, 0}, false); glassCube(scene, o, 0.6f, {0, 0, 0}, false); break;
        case Glass::Intersecting:
            // The cavity reaches x 1.3, through the outer +x face.
            glassCube(scene, o, 1, {0, 0, 0}, false); glassCube(scene, o, 0.6f, {0.7f, 0, 0}, true); break;
        case Glass::ThreeOutward:
            glassCube(scene, o, 1, {0, 0, 0}, false); glassCube(scene, o, 0.7f, {0, 0, 0}, false);
            glassCube(scene, o, 0.4f, {0, 0, 0}, false); break;
        case Glass::Siblings:
            glassCube(scene, o, 0.5f, {-1, 0, 0}, false); glassCube(scene, o, 0.5f, {1, 0, 0}, false); break;
        case Glass::Open:
            glassCube(scene, o, 1, {0, 0, 0}, false); scene.indexCount -= 3; break;
        case Glass::Inverted: glassCube(scene, o, 1, {0, 0, 0}, true); break;
        case Glass::Bowl:
        {
            // A faceted dome shell: outer hemisphere radius 1, inner 0.8, eight
            // segments and three rings from the pole to the equator, closed by
            // a flat base ring at y 0 facing down. One connected shell.
            constexpr int segments = 8, rings = 3;
            constexpr double kPi = 3.14159265358979323846;
            std::array<std::array<std::uint32_t, segments>, rings + 1> outer {}, inner {};
            for (const bool isInner : {false, true})
            {
                const double radius = isInner ? 0.8 : 1.0;
                auto& ring = isInner ? inner : outer;
                const auto pole = vertex(scene, o, {0, float(radius), 0});
                for (int j = 0; j < segments; ++j) ring[0][j] = pole;
                for (int k = 1; k <= rings; ++k)
                    for (int j = 0; j < segments; ++j)
                    {
                        const double theta = 0.5 * kPi * k / rings, phi = 2 * kPi * j / segments;
                        const double y = k == rings ? 0.0 : radius * std::cos(theta);
                        ring[k][j] = vertex(scene, o, {float(radius * std::sin(theta) * std::cos(phi)), float(y),
                                                       float(radius * std::sin(theta) * std::sin(phi))});
                    }
                for (int k = 0; k < rings; ++k)
                    for (int j = 0; j < segments; ++j)
                    {
                        const int n = (j + 1) % segments;
                        const auto a = ring[k][j], b = ring[k + 1][j], c = ring[k + 1][n], d = ring[k][n];
                        if (isInner)
                        {
                            triangle(scene, a, b, c);
                            if (k != 0) triangle(scene, a, c, d);
                        }
                        else
                        {
                            triangle(scene, a, c, b);
                            if (k != 0) triangle(scene, a, d, c);
                        }
                    }
            }
            for (int j = 0; j < segments; ++j)
            {
                const int n = (j + 1) % segments;
                triangle(scene, outer[rings][j], outer[rings][n], inner[rings][j]);
                triangle(scene, outer[rings][n], inner[rings][n], inner[rings][j]);
            }
            break;
        }
    }
    endObject(scene, o);
    return o;
}

struct Admitted
{
    bool ok = false;
    std::string error;
    sg::Prepared prepared;
    ov::Frame frame;
};
// Admission sees the authored scene. moved, when set, is one object the frame
// places drop lower in y through its instance matrix, as a runtime transform
// would, after admission has run.
Admitted admit(const Visual3DScene& scene, std::size_t moved = ~std::size_t(0), float drop = 0)
{
    Admitted result;
    const auto root = glassProgram(scene);
    result.ok = sg::prepare(scene, root.get(), {}, result.prepared, result.error);
    if (!result.ok) return result;
    std::vector<ov::Instance> instances;
    const ov::Matrix identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        auto world = identity;
        if (o == moved) world[13] = -drop;
        instances.push_back({world, o});
    }
    result.ok = ov::frame(result.prepared.query, instances, result.frame, result.error);
    return result;
}

// Binary64 branch walk over the CPU scene query, with the shader's rules: the
// inside flag flips at each glass transmission, an inside branch that leaves
// without a glass hit refuses (MEDIUM), a glass hit whose orientation
// disagrees refuses, and GLASS_INTERFACE_DEPTH (32) bounds each path. camera
// applies the outcome of retries 4 and 5: an opaque front face met from inside
// ends the branch there, in air, only when the retry 5 query (sg::seamQuery,
// preferring the glass) finds the glass exit within the seam window. camera
// also applies the exit probe: the transmitted child of a glass exit whose hit
// is an opaque back face ends on the opaque front face the kSeamExitProbe
// query finds, avoiding the glass, if there is one. It skips retries 1 to 3
// and the visit cap; chain below models those. Every receiver must be met from
// its front: the scenes here are closed, so a back face in air means the
// branch started inside an opaque body, and the walk refuses it in both modes.
// The shader would shade such a face. The photon walk keeps the caustic rule
// this way too. inside starts the seed inside the glass.
struct Terminal { std::uint32_t object; double weight; int interfaces; bool inside; ov::Point normal; bool entering; };
struct Walk
{
    int status = 0;
    double environment = 0, unresolved = 0;
    int tir = 0, deepest = 0;
    std::vector<Terminal> terminals;
};
ov::Point unit(ov::Point v)
{
    const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return {v[0] / n, v[1] / n, v[2] / n};
}
float dot3(const ov::Point& a, const ov::Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Walk walk(const Admitted& admitted, ov::Point origin, ov::Point direction, bool camera, bool inside = false)
{
    using Status = sg::SmoothTransportStatus;
    // back is the branch's seamSpawnBack, exitChild marks a glass exit's transmitted child.
    struct Branch { ov::Point origin, direction; double weight; int interfaces; bool inside; ov::Point back; bool exitChild; };
    Walk result;
    const auto seed = unit(direction);
    std::vector<Branch> pending {{origin, seed, 1.0, 0, inside, {-seed[0], -seed[1], -seed[2]}, false}};
    for (int visit = 0; visit < 4096 && !pending.empty(); ++visit)
    {
        const auto ray = pending.back();
        pending.pop_back();
        auto hit = ov::closestHitCpu(admitted.prepared.query, admitted.frame, ray.origin, ray.direction, 0.0f, 3.402823e38f);
        if (hit.status < 0) { result.status = int(Status::QueryRefused); return result; }
        if (hit.status == 0)
        {
            if (ray.inside) { result.status = int(Status::MediumMismatch); return result; }
            result.environment += ray.weight;
            continue;
        }
        if (hit.objectId != kGlassId)
        {
            bool contact = false;
            if (ray.inside && camera && sg::seamContact(hit, kGlassId))
            {
                const auto probe = sg::seamQuery(sg::kSeamContactCheck, hit.position, ray.direction);
                const auto glassExit = ov::closestHitCpu(admitted.prepared.query, admitted.frame, probe.origin,
                                                         probe.direction, probe.minimum, probe.maximum, kGlassId);
                if (glassExit.status < 0) { result.status = int(Status::QueryRefused); return result; }
                contact = sg::seamContactVerified(glassExit, kGlassId);
            }
            if (ray.inside && !contact) { result.status = int(Status::MediumMismatch); return result; }
            if (camera && ray.exitChild && sg::seamExitBackFace(hit, kGlassId))
            {
                const auto probe = sg::seamQuery(sg::kSeamExitProbe, ray.origin, ray.direction, ray.back);
                const auto front = ov::closestHitCpu(admitted.prepared.query, admitted.frame, probe.origin,
                                                     probe.direction, probe.minimum, probe.maximum, 0, kGlassId);
                if (front.status < 0) { result.status = int(Status::QueryRefused); return result; }
                if (sg::seamContact(front, kGlassId)) hit = front;
            }
            if (!hit.entering) { result.status = int(Status::ReceiverMiss); return result; }
            // A contact ends the branch in air, as the shader sets inside=false.
            result.terminals.push_back({hit.objectId, ray.weight, ray.interfaces, ray.inside && !contact,
                                        hit.outwardNormal, hit.entering});
            continue;
        }
        if (ray.inside == hit.entering) { result.status = int(Status::OrientationMismatch); return result; }
        if (ray.interfaces >= int(sg::kMaximumInterfaceDepth)) { result.unresolved += ray.weight; continue; }
        auto normal = hit.outwardNormal;
        if (!hit.entering) for (auto& c : normal) c = -c;
        const double from = ray.inside ? kIor : 1.0, to = ray.inside ? 1.0 : kIor;
        const double cosine = std::clamp(-double(dot3(ray.direction, normal)), 0.0, 1.0);
        const double ratio = from / to, sineSquared = ratio * ratio * (1 - cosine * cosine);
        const bool tir = sineSquared > 1;
        double reflectance = 1;
        ov::Point transmitted {};
        if (!tir)
        {
            const double exitCosine = std::sqrt(1 - sineSquared);
            for (int c = 0; c < 3; ++c)
                transmitted[c] = float(ratio * ray.direction[c] + (ratio * cosine - exitCosine) * normal[c]);
            transmitted = unit(transmitted);
            const double s = (from * cosine - to * exitCosine) / (from * cosine + to * exitCosine);
            const double p = (to * cosine - from * exitCosine) / (to * cosine + from * exitCosine);
            reflectance = 0.5 * (s * s + p * p);
        }
        else ++result.tir;
        result.deepest = std::max(result.deepest, ray.interfaces + 1);
        const float bias = sg::glassBiasCpu(hit.position);
        ov::Point reflected {};
        for (int c = 0; c < 3; ++c) reflected[c] = ray.direction[c] + 2 * float(cosine) * normal[c];
        reflected = unit(reflected);
        const auto offset = [&](float side) {
            ov::Point p = hit.position;
            for (int c = 0; c < 3; ++c) p[c] += side * bias * normal[c];
            return p;
        };
        const double reflectedWeight = ray.weight * reflectance, transmittedWeight = ray.weight - reflectedWeight;
        if (reflectedWeight >= 1e-5)
            pending.push_back({offset(1), reflected, reflectedWeight, ray.interfaces + 1, ray.inside,
                               sg::seamSpawnBack(normal, false), false});
        else result.unresolved += reflectedWeight;
        if (!tir)
        {
            if (transmittedWeight >= 1e-5)
                pending.push_back({offset(-1), transmitted, transmittedWeight, ray.interfaces + 1, !ray.inside,
                                   sg::seamSpawnBack(normal, true), ray.inside});
            else result.unresolved += transmittedWeight;
        }
    }
    if (!pending.empty()) result.status = int(Status::RayVisits);
    return result;
}
double terminalWeight(const Walk& walk, std::uint32_t object, int interfaces = -1)
{
    double total = 0;
    for (const auto& t : walk.terminals)
        if (t.object == object && (interfaces < 0 || t.interfaces == interfaces)) total += t.weight;
    return total;
}
double total(const Walk& walk)
{
    double sum = walk.environment + walk.unresolved;
    for (const auto& t : walk.terminals) sum += t.weight;
    return sum;
}

// Visit-counting twin of one camera branch's seam chain in glassTransport:
// retries 0 to 5 and the exit child probe (kSeamExitChild, kSeamExitProbe),
// each query spending one visit of the walk's cap. glassPairWalk runs the same
// chain, except that its retry 4 and its exit probe also pass over the water,
// which none of these scenes has. It follows one branch until it ends,
// refuses, reaches a glass face it would split at, or runs out of visits
// (Trapped). start is the visit index the branch is popped at. seam is the
// state the branch was pushed with: 0, or kSeamExitChild for the transmitted
// child of a glass exit. A receiver end reports the face the shader would
// shade: its normal, and whether the branch met it from the front.
enum class ChainEnd { Receiver, Environment, Interface, Medium, Orientation, Trapped, Refused };
struct Chain
{
    ChainEnd end = ChainEnd::Refused;
    int visits = 0;
    std::uint32_t object = 0;
    ov::Point normal {};
    bool entering = false;
};
Chain chain(const Admitted& admitted, ov::Point origin, ov::Point direction, ov::Point back, bool inside,
            int start, int cap, int seam = 0)
{
    std::uint32_t kept = 0;
    for (int visit = start;; ++visit)
    {
        Chain out;
        out.visits = visit - start;
        if (visit >= cap) { out.end = ChainEnd::Trapped; return out; }
        out.visits = visit - start + 1;
        const auto query = sg::seamQuery(seam, origin, direction, back);
        auto hit = ov::closestHitCpu(admitted.prepared.query, admitted.frame, query.origin, query.direction, query.minimum,
                                     query.maximum, seam == sg::kSeamContactCheck ? kGlassId : 0,
                                     seam == sg::kSeamExitProbe ? kGlassId : 0);
        if (hit.status < 0) { out.end = ChainEnd::Refused; return out; }
        const bool glass = hit.status == 1 && hit.objectId == kGlassId;
        if (seam == sg::kSeamExitChild)
        {
            if (sg::seamExitBackFace(hit, kGlassId)) { seam = sg::kSeamExitProbe; continue; }
            seam = 0;
        }
        if (seam == 3)
        {
            if (sg::seamEscaped(hit, kGlassId, origin, query.direction)) hit.entering = false;
            else { seam = sg::kSeamContactRetry; continue; }
        }
        else if (seam == sg::kSeamContactRetry)
        {
            if (sg::seamContact(hit, kGlassId))
            {
                // The shader keeps the hit in the branch: origin, back and depth.
                origin = hit.position; back = hit.outwardNormal; kept = hit.objectId;
                seam = sg::kSeamContactCheck;
                continue;
            }
            hit.status = 0;
        }
        else if (seam == sg::kSeamContactCheck)
        {
            if (sg::seamContactVerified(hit, kGlassId))
            {
                hit.objectId = kept; hit.position = origin; hit.outwardNormal = back; hit.entering = true;
                inside = false;
            }
            else hit.status = 0;
        }
        else if (seam == sg::kSeamExitProbe)
        {
            // No opaque front face near the exit: query again as an ordinary branch.
            if (!sg::seamContact(hit, kGlassId)) { seam = 0; continue; }
        }
        else
        {
            bool seamed = seam == 1 ? !glass : inside ? (!glass || hit.entering) : (glass && !hit.entering);
            if (seamed && seam == 2 && inside && sg::seamEscaped(hit, kGlassId, origin, query.direction))
            { inside = false; seamed = false; }
            if (seamed && (seam < 2 || (inside && !glass))) { ++seam; continue; }
            if (seam == 1) inside = !hit.entering;
        }
        if (hit.status == 0) { out.end = inside ? ChainEnd::Medium : ChainEnd::Environment; return out; }
        if (hit.objectId != kGlassId)
        {
            out.end = inside ? ChainEnd::Medium : ChainEnd::Receiver;
            out.object = hit.objectId;
            out.normal = hit.outwardNormal;
            out.entering = hit.entering;
            return out;
        }
        out.end = inside == hit.entering ? ChainEnd::Orientation : ChainEnd::Interface;
        out.object = kGlassId;
        return out;
    }
}

constexpr std::uint32_t kCube = 0x5005u, kFloor = 0x6006u;
// True for the face a cube resting on the glass top shows the glass: its
// bottom, normal straight down, met from the front.
bool cubeBottom(const ov::Point& normal, bool entering)
{
    return entering && std::abs(normal[0]) < 1e-6f && std::abs(normal[1] + 1) < 1e-6f && std::abs(normal[2]) < 1e-6f;
}

// (a) A thick faceted dome closed by a base ring, standing on a floor box
// whose top is the base plane, with an opaque tower in its cavity.
void hollowDomeAdmitted()
{
    const auto owned = newScene();
    auto& scene = *owned;
    addGlass(scene, Glass::Bowl);
    check(scene.objects[0].indexCount / 3 == 96, "the dome shell has 96 triangles, inside the 128 entry triangle cap");
    receiverBox(scene, kFloor, 20, {-3, -0.5f, -3}, {3, 0, 3});
    receiverBox(scene, kCube, 21, {-0.2f, 0, -0.2f}, {0.2f, 0.6f, 0.2f});
    const auto admitted = admit(scene);
    check(admitted.ok, "a hollow dome on a floor with a tower in its cavity is admitted: " + admitted.error);
    if (!admitted.ok) return;
    // Straight down through the dome top: outer face, cavity ceiling, tower top.
    const auto down = walk(admitted, {0.02f, 3, 0.03f}, {0, -1, 0}, false);
    check(down.status == 0 && terminalWeight(down, kCube, 2) > 0.85,
          "a photon through the dome top deposits on the tower after two interfaces: weight "
              + std::to_string(terminalWeight(down, kCube, 2)));
    check(std::abs(total(down) - 1) < 1e-9 && down.unresolved < 1e-3, "the dome photon walk conserves its power");
}

// (b) Nested shells: an outer cube with an inward cube cavity, and an opaque
// cube in the cavity. The axial camera ray crosses outer glass, the cavity
// wall, and ends on the cube after two refractions.
void cavityCube()
{
    const auto owned = newScene();
    auto& scene = *owned;
    addGlass(scene, Glass::Hollow);
    receiverBox(scene, kCube, 20, {-0.2f, -0.2f, -0.2f}, {0.2f, 0.2f, 0.2f});
    const auto admitted = admit(scene);
    check(admitted.ok, "nested glass shells with an opaque cube in the cavity are admitted: " + admitted.error);
    if (!admitted.ok) return;
    const auto camera = walk(admitted, {0.05f, 0.03f, 5}, {0, 0, -1}, true);
    // Two normal-incidence transmissions keep (1-0.04)^2 of the power.
    check(camera.status == 0 && std::abs(terminalWeight(camera, kCube, 2) - 0.96 * 0.96) < 1e-3,
          "the camera ray meets the cube after outer glass and cavity wall: weight "
              + std::to_string(terminalWeight(camera, kCube, 2)));
    check(terminalWeight(camera, kCube, 0) == 0 && terminalWeight(camera, kCube, 1) == 0,
          "no path reaches the cube without crossing both glass faces");
    check(std::abs(total(camera) - 1) < 1e-9, "the camera walk conserves its power");

    // A photon from above deposits on the cube top from the cavity air.
    const auto photon = walk(admitted, {0.03f, 4, 0.05f}, {0, -1, 0}, false);
    check(photon.status == 0 && terminalWeight(photon, kCube, 2) > 0.9,
          "a caustic photon deposits on the cavity cube after two interfaces");
    for (const auto& t : photon.terminals)
        check(!t.inside, "every caustic deposit lands in air");

    // Total internal reflection in the shell wall. Entering the top face at
    // x 0.8 with 80 degrees incidence, the ray refracts to 41.0 degrees and
    // runs down the wall between x 0.6 and 1. It meets the outer +x face and
    // the cavity wall at 49 degrees, past the 41.8 degree critical angle, so
    // the wall guides it down. At the bottom face 39% reflects back up, so
    // part of the power is still in the wall at the 32 interface bound.
    const double incidence = 80.0 * 3.14159265358979323846 / 180.0;
    const ov::Point slant {float(std::sin(incidence)), float(-std::cos(incidence)), 0};
    const ov::Point start {0.8f - 2 * slant[0], 1 - 2 * slant[1], 0.01f};
    const auto guided = walk(admitted, start, slant, true);
    check(guided.status == 0 && guided.tir >= 3 && guided.deepest <= int(sg::kMaximumInterfaceDepth),
          "a ray guided down the shell wall by total internal reflection completes: status "
              + std::to_string(guided.status) + ", " + std::to_string(guided.tir) + " reflections");
    check(std::abs(total(guided) - 1) < 1e-9 && guided.environment > 0.5,
          "the guided walk conserves its power and most of it leaves the wall");
}

// (c) Opaque objects touching the outer shell are admitted; one that passes
// into the glass solid is refused by name.
void touchingAndInterpenetrating()
{
    for (const float overlap : {0.0f, 2e-6f})
    {
        // The cube comes first in the scene and rests on the glass top, at
        // exactly 0 overlap or 2e-6 into it, inside the clearance of 5e-6
        // there. At 0 overlap the cube's bottom ties the glass exit and the
        // tie goes to the cube, the lower object index; retry 5 prefers the
        // glass, so it still finds that exit within the seam window. The
        // cube's corners are exact in float but for the bottom at 2e-6.
        const std::string label = overlap == 0 ? "at exactly 0 overlap" : "2e-6 into the glass";
        const auto owned = newScene();
        auto& scene = *owned;
        receiverBox(scene, kCube, 20, {-0.25f, 1 - overlap, -0.25f}, {0.25f, 1.5f, 0.25f});
        addGlass(scene, Glass::Hollow);
        const auto admitted = admit(scene);
        check(admitted.ok, "an opaque cube resting on the glass top " + label + " is admitted: " + admitted.error);
        if (!admitted.ok) continue;
        // From inside the top wall straight up into the shared face.
        const auto up = walk(admitted, {0.03f, 0.8f, 0.05f}, {0, 1, 0}, true, true);
        check(up.status == 0 && up.terminals.size() == 1 && up.terminals[0].object == kCube
                  && std::abs(up.terminals[0].weight - 1) < 1e-9 && !up.terminals[0].inside && up.environment == 0,
              "an inside ray at the shared face " + label + " ends on the opaque cube, in air, with all its weight: status "
                  + std::to_string(up.status) + ", " + std::to_string(up.terminals.size()) + " terminals");
        check(up.terminals.size() == 1 && cubeBottom(up.terminals[0].normal, up.terminals[0].entering),
              "that terminal " + label + " is the cube's bottom face, met from the front");
        // The shader's seam chain for that branch: retries 0 to 5, one visit
        // each. Retry 1 meets nothing, retries 0, 2 and 4 the cube, retry 3
        // the cavity ceiling from inside, and retry 5 the glass exit.
        const auto first = chain(admitted, {0.03f, 0.8f, 0.05f}, {0, 1, 0}, {0, -1, 0}, true, 0, 64);
        check(first.end == ChainEnd::Receiver && first.object == kCube && first.visits == 6
                  && cubeBottom(first.normal, first.entering),
              "the contact chain " + label + " ends on the cube's bottom face after 6 visits: end "
                  + std::to_string(int(first.end)) + ", " + std::to_string(first.visits) + " visits");
        // Preview caps a walk at kPreviewCameraVisits visits, an export pair walk at 48.
        for (const int cap : {int(sg::kPreviewCameraVisits), 48})
        {
            std::string trapped;
            bool split = true;
            for (int start = 0; start < cap; ++start)
            {
                const auto c = chain(admitted, {0.03f, 0.8f, 0.05f}, {0, 1, 0}, {0, -1, 0}, true, start, cap);
                const bool complete = c.end == ChainEnd::Receiver && c.object == kCube;
                if (c.end == ChainEnd::Trapped) trapped += (trapped.empty() ? "" : ",") + std::to_string(start);
                if (start <= cap - 6 ? !complete : c.end != ChainEnd::Trapped) split = false;
            }
            std::printf("contact chain %s, cap %d: Trapped at start indices %s\n", label.c_str(), cap, trapped.c_str());
            check(split, "at cap " + std::to_string(cap) + " a contact branch popped at index " + std::to_string(cap - 6)
                             + " or earlier completes and a later one ends Trapped: Trapped at " + trapped);
        }
    }
    for (const float gap : {0.0f, 5e-6f})
    {
        // The glass exit wins the split: at exactly 0 gap with the glass first
        // in the scene (the tie goes to the lower object index), and at half a
        // bias above the glass top with the cube first (the exit is nearer).
        // The inside ray splits at the exit, and its transmitted child starts
        // one bias out, inside the cube. Its first hit is the cube's top face
        // from behind. The exit probe finds the cube's bottom face from the
        // front, and the child ends there in air.
        const std::string label = gap == 0 ? "with the glass first at 0 gap" : "with the cube first half a bias above";
        const auto owned = newScene();
        auto& scene = *owned;
        if (gap != 0) receiverBox(scene, kCube, 20, {-0.25f, 1 + gap, -0.25f}, {0.25f, 1.5f, 0.25f});
        addGlass(scene, Glass::Hollow);
        if (gap == 0) receiverBox(scene, kCube, 20, {-0.25f, 1, -0.25f}, {0.25f, 1.5f, 0.25f});
        const auto admitted = admit(scene);
        check(admitted.ok, "an opaque cube resting on the glass top " + label + " is admitted: " + admitted.error);
        if (!admitted.ok) continue;
        const ov::Point seed {0.03f, 0.8f, 0.05f}, up {0, 1, 0};
        const auto walked = walk(admitted, seed, up, true, true);
        check(walked.status == 0 && terminalWeight(walked, kCube, 1) > 0.95,
              "an inside ray at the shared face " + label + " transmits and ends on the cube: status "
                  + std::to_string(walked.status) + ", weight " + std::to_string(terminalWeight(walked, kCube, 1)));
        bool bottom = true;
        for (const auto& t : walked.terminals)
            if (t.object == kCube) bottom = bottom && !t.inside && cubeBottom(t.normal, t.entering);
        check(bottom, "every cube terminal " + label + " lies in air on the cube's bottom face, met from the front");
        const auto c = chain(admitted, seed, up, {0, -1, 0}, true, 0, 64);
        check(c.end == ChainEnd::Interface && c.visits == 1,
              "the chain " + label + " splits at the glass exit after 1 visit: end " + std::to_string(int(c.end))
                  + ", " + std::to_string(c.visits) + " visits");
        // The transmitted child, spawned as the shader spawns it.
        const auto split = ov::closestHitCpu(admitted.prepared.query, admitted.frame, seed, up, 0.0f, 3.402823e38f);
        check(split.status == 1 && split.objectId == kGlassId && !split.entering,
              "the inside ray " + label + " meets the glass exit first");
        ov::Point normal {-split.outwardNormal[0], -split.outwardNormal[1], -split.outwardNormal[2]};
        const float bias = sg::glassBiasCpu(split.position);
        ov::Point child = split.position;
        for (int a = 0; a < 3; ++a) child[a] -= normal[a] * bias;
        const auto back = sg::seamSpawnBack(normal, true);
        const auto bare = chain(admitted, child, up, back, false, 0, 64);
        check(bare.end == ChainEnd::Receiver && bare.object == kCube && !bare.entering,
              "without the probe the child " + label + " meets the cube's top face from behind");
        const auto probed = chain(admitted, child, up, back, false, 0, 64, sg::kSeamExitChild);
        check(probed.end == ChainEnd::Receiver && probed.object == kCube && probed.visits == 2
                  && cubeBottom(probed.normal, probed.entering),
              "the exit child " + label + " ends on the cube's bottom face after 2 visits: end "
                  + std::to_string(int(probed.end)) + ", " + std::to_string(probed.visits) + " visits");
        for (const int cap : {int(sg::kPreviewCameraVisits), 48})
        {
            const auto last = chain(admitted, child, up, back, false, cap - 2, cap, sg::kSeamExitChild);
            const auto late = chain(admitted, child, up, back, false, cap - 1, cap, sg::kSeamExitChild);
            check(last.end == ChainEnd::Receiver && cubeBottom(last.normal, last.entering) && late.end == ChainEnd::Trapped,
                  "at cap " + std::to_string(cap) + " an exit child " + label + " popped at index "
                      + std::to_string(cap - 2) + " completes on the cube and one popped later ends Trapped");
        }
    }
    {
        // (4a) A cube admitted in air above the glass, then moved 0.5 down by
        // its instance matrix so its bottom sits 0.25 inside the top wall. The
        // inside ray meets its face with no glass exit near, and refuses.
        const auto owned = newScene();
        auto& scene = *owned;
        receiverBox(scene, kCube, 20, {-0.25f, 1.25f, -0.25f}, {0.25f, 1.75f, 0.25f});
        addGlass(scene, Glass::Hollow);
        const auto inAir = admit(scene);
        check(inAir.ok, "a cube in air above the glass is admitted: " + inAir.error);
        const auto moved = admit(scene, 0, 0.5f);
        check(moved.ok, "the moved frame builds: " + moved.error);
        if (moved.ok)
        {
            const auto up = walk(moved, {0.03f, 0.7f, 0.05f}, {0, 1, 0}, true, true);
            check(up.status == int(sg::SmoothTransportStatus::MediumMismatch),
                  "a runtime transform that moves the cube into the glass refuses as MEDIUM: status "
                      + std::to_string(up.status));
        }
    }
    {
        // An opaque floor box whose top is the glass bottom face.
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Hollow);
        receiverBox(scene, kFloor, 20, {-3, -1.5f, -3}, {3, -1, 3});
        const auto admitted = admit(scene);
        check(admitted.ok, "glass standing on an opaque floor is admitted: " + admitted.error);
    }
    {
        // The cube sinks 0.1 into the glass top.
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Hollow);
        receiverBox(scene, kCube, 20, {-0.3f, 0.9f, -0.3f}, {0.3f, 1.3f, 0.3f});
        const auto refused = admit(scene);
        check(!refused.ok && refused.error.rfind(sg::kOpaqueInterpenetrationReason, 0) == 0
                  && refused.error.find("opaque object " + std::to_string(kCube)) != std::string::npos,
              "an opaque cube sunk into the glass is refused as interpenetration: " + refused.error);
    }
    {
        // A cube wholly inside the glass wall, crossing no glass face.
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Hollow);
        receiverBox(scene, kCube, 20, {0.7f, -0.1f, -0.1f}, {0.9f, 0.1f, 0.1f});
        const auto refused = admit(scene);
        check(!refused.ok && refused.error.rfind(sg::kOpaqueInterpenetrationReason, 0) == 0,
              "an opaque cube embedded in the glass wall is refused: " + refused.error);
    }
    {
        // The cavity cube pokes through the cavity wall into the glass.
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Hollow);
        receiverBox(scene, kCube, 20, {-0.2f, -0.2f, -0.2f}, {0.7f, 0.2f, 0.2f});
        const auto refused = admit(scene);
        check(!refused.ok && refused.error.rfind(sg::kOpaqueInterpenetrationReason, 0) == 0,
              "a cavity cube passing into the cavity wall is refused: " + refused.error);
    }
    // Retry 4 accepts only an opaque front face.
    ov::ClosestHit hit;
    hit.status = 1; hit.objectId = kCube; hit.entering = true;
    check(sg::seamContact(hit, kGlassId), "an opaque front face ends an inside branch");
    hit.entering = false;
    check(!sg::seamContact(hit, kGlassId), "an opaque back face still refuses");
    hit.entering = true; hit.objectId = kGlassId;
    check(!sg::seamContact(hit, kGlassId), "a glass face is never a contact");
    hit.objectId = kFloor;
    check(!sg::seamContact(hit, kGlassId, kFloor), "the paired walk's water is never a contact");
    hit.status = 0;
    check(!sg::seamContact(hit, kGlassId), "a miss still refuses");
}

// (4b, 4c) Admission tests crossings exactly. A flat quad tilted 0.2 degrees
// that dips 2e-5 under the solid cube's top face, four times the clearance
// there (kSeamWindow times the 1e-5 bias), crosses the face's interior along
// x 0.3. Probes in the quad's plane would sit only about 1e-6 under the face.
// Quads on the face, or within the clearance of it, touch and are admitted.
void grazingAndCoplanar()
{
    const double slope = std::tan(0.2 * 3.14159265358979323846 / 180.0);
    {
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Solid);
        const float end = float(0.3 + 2e-5 / slope);
        opaqueQuad(scene, kCube, 20, -0.5f, end, -0.2f, 0.2f, float(1 + 0.8 * slope), float(1 - 2e-5));
        const auto refused = admit(scene);
        check(!refused.ok && refused.error.rfind(sg::kOpaqueInterpenetrationReason, 0) == 0,
              "a quad pressed 2e-5 into the glass at 0.2 degrees is refused: " + refused.error);
    }
    {
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Solid);
        opaqueQuad(scene, kCube, 20, -0.5f, 0.5f, -0.5f, 0.5f, 1, 1);
        const auto admitted = admit(scene);
        check(admitted.ok, "a quad lying on the glass top face is admitted: " + admitted.error);
    }
    {
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Solid);
        opaqueQuad(scene, kCube, 20, -0.5f, 0.5f, -0.5f, 0.5f, 1 - 2e-6f, 1 - 2e-6f);
        const auto admitted = admit(scene);
        check(admitted.ok, "a quad 2e-6 under the glass top face, inside the clearance, is admitted: " + admitted.error);
    }
    {
        // Tilted across the top face's edge at x 1, so the crossing runs along
        // that edge only: the quad touches the cube's edge from outside.
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Solid);
        opaqueQuad(scene, kCube, 20, 0.5f, 1.5f, -0.5f, 0.5f, 1.5f, 0.5f);
        const auto admitted = admit(scene);
        check(admitted.ok, "a quad touching only the cube's top edge is admitted: " + admitted.error);
    }
}

// (d) Shells that intersect, are wound wrongly for their depth, or overlap
// are refused by name.
void shellRefusals()
{
    const auto reason = [](Glass shape) {
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, shape);
        return admit(scene).error;
    };
    check(reason(Glass::Intersecting) == sg::kShellsIntersectReason,
          "a cavity crossing the outer shell is refused as intersecting shells: " + reason(Glass::Intersecting));
    check(reason(Glass::HollowOutwardCavity) == sg::kShellOrientationReason,
          "a cavity wound outward is refused for its orientation: " + reason(Glass::HollowOutwardCavity));
    check(reason(Glass::ThreeOutward) == sg::kShellOverlapReason,
          "three nested outward shells are refused as overlapping: " + reason(Glass::ThreeOutward));
}

// (e) The single-shell case is unchanged.
// A closed glass sphere of 12000 triangles, past the old 2048 volume and 8192
// scene counts: 100 segments and 61 rings make 2*100*60 triangles, with one
// vertex at each pole, so it is one closed manifold shell.
void largeClosedGlassVolume()
{
    constexpr std::uint32_t segments = 100, rings = 61;
    constexpr double kPi = 3.14159265358979323846;
    const auto owned = newScene();
    auto& scene = *owned;
    Visual3DSceneGrowth growth;
    growth.vertices = 2 + segments * (rings - 1) + 24;
    growth.indices = 3 * 2 * std::size_t(segments) * (rings - 1) + 36;
    std::string error;
    const bool grown = growVisual3DScene(scene, growth, error);
    check(grown, "the 12000-triangle scene allocates: " + error);
    if (!grown) return;
    const auto o = beginObject(scene, kGlassId, kGlassMaterial);
    const auto north = vertex(scene, o, {0, 1, 0});
    std::vector<std::uint32_t> band;
    for (std::uint32_t k = 1; k < rings; ++k)
        for (std::uint32_t j = 0; j < segments; ++j)
        {
            const double theta = kPi * k / rings, phi = 2 * kPi * j / segments;
            band.push_back(vertex(scene, o, {float(std::sin(theta) * std::cos(phi)), float(std::cos(theta)),
                                             float(std::sin(theta) * std::sin(phi))}));
        }
    const auto south = vertex(scene, o, {0, -1, 0});
    const auto at = [&](std::uint32_t k, std::uint32_t j) { return band[(k - 1) * segments + j % segments]; };
    for (std::uint32_t j = 0; j < segments; ++j)
    {
        triangle(scene, north, at(1, j + 1), at(1, j));
        triangle(scene, south, at(rings - 1, j), at(rings - 1, j + 1));
        for (std::uint32_t k = 1; k + 1 < rings; ++k)
        {
            triangle(scene, at(k, j), at(k, j + 1), at(k + 1, j + 1));
            triangle(scene, at(k, j), at(k + 1, j + 1), at(k + 1, j));
        }
    }
    endObject(scene, o);
    check(scene.objects[o].indexCount / 3 == 12000, "the test sphere has 12000 triangles");
    receiverBox(scene, kFloor, 20, {-3, -1.5f, -3}, {3, -1.01f, 3});
    const auto admitted = admit(scene);
    check(admitted.ok && admitted.prepared.enabled
              && admitted.prepared.transport == surfacematerial::TransportMode::ClosedVolume,
          "a closed glass sphere of 12000 triangles is admitted: " + admitted.error);
    if (!admitted.ok) return;
    const auto camera = walk(admitted, {0.05f, 0.03f, 5}, {0, 0, -1}, true);
    check(camera.status == 0 && std::abs(total(camera) - 1) < 1e-9 && camera.environment > 0.9,
          "a ray through the 12000-triangle sphere conserves power and transmits to the environment");
}

void singleShell()
{
    {
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, Glass::Solid);
        receiverBox(scene, kFloor, 20, {-3, -1.5f, -3}, {3, -1.01f, 3});
        const auto admitted = admit(scene);
        check(admitted.ok && admitted.prepared.enabled
                  && admitted.prepared.transport == surfacematerial::TransportMode::ClosedVolume
                  && admitted.prepared.object.value == kGlassId,
              "a solid glass cube over a floor is admitted as before: " + admitted.error);
        if (admitted.ok)
        {
            const auto camera = walk(admitted, {0.05f, 0.03f, 5}, {0, 0, -1}, true);
            // The walk moves any branch under 1e-5 to unresolved, and at normal
            // incidence (R = 0.04) that drops about 2.5e-6 of the power.
            check(camera.status == 0 && camera.terminals.empty() && std::abs(total(camera) - 1) < 1e-9
                      && camera.environment > 1 - 1e-5,
                  "the axial ray through the solid cube leaves to the environment");
        }
    }
    const auto reason = [](Glass shape) {
        const auto owned = newScene();
        auto& scene = *owned;
        addGlass(scene, shape);
        return admit(scene).error;
    };
    check(reason(Glass::Open) == "Closed glass volume has an open edge or inconsistent winding",
          "an open cube keeps its refusal: " + reason(Glass::Open));
    check(reason(Glass::Inverted) == "Closed glass volume winding must point outward and enclose nonzero volume",
          "an inward cube keeps its refusal: " + reason(Glass::Inverted));
    const auto siblings = newScene();
    addGlass(*siblings, Glass::Siblings);
    const auto apart = admit(*siblings);
    check(apart.ok, "two separate outward shells are two glass solids in air: " + apart.error);
}
} // namespace

// The closed-glass overdraw probe (ARBIT_GL_GLASS_OVERDRAW_PROBE=1) is GPU-only, so this
// pins its GL-free half: the switch, the walk-mode tags, the per-object merge across
// bands, the walk and per-mode totals, the factors, the times and the exact log line.
void overdrawProbeLine()
{
    namespace go = arbitgpu::glassoverdraw;
    check(go::enabledValue("1") && !go::enabledValue("0") && !go::enabledValue(nullptr)
        && !go::enabledValue(""), "overdraw probe turns on only for 1");
    // Same test as the shader: exactly 2 is the water walk, anything else above 0.5 the
    // volume walk, the rest ordinary lighting.
    check(go::walkMode(0.0f) == go::kOrdinary && go::walkMode(0.4f) == go::kOrdinary
        && go::walkMode(1.0f) == go::kVolume && go::walkMode(2.0f) == go::kWater
        && go::walkMode(3.0f) == go::kVolume, "overdraw probe walk mode follows uGlassTransport.x");
    check(std::string(go::walkTag(go::kWater)) == "water" && std::string(go::walkTag(go::kVolume)) == "volume"
        && std::string(go::walkTag(go::kOrdinary)) == "obj", "overdraw probe walk tags");
    go::FrameCounts counts;
    counts.width = 672; counts.height = 378; counts.bands = 2; counts.covered = 1000;
    // Two bands draw the water, the dome and an ordinary object. Each object keeps one entry.
    go::accumulate(counts, 0x3004u, go::kWater, 600, 500, 1500000);
    go::accumulate(counts, kGlassId, go::kVolume, 900, 400, 2000000);
    go::accumulate(counts, 0x3005u, go::kOrdinary, 100, 100, 250000);
    go::accumulate(counts, 0x3004u, go::kWater, 300, 200, 1000000);
    go::accumulate(counts, kGlassId, go::kVolume, 700, 300, 1250000);
    // The coverage pass looks entries up as ordinary. That must not drop a walk tag.
    go::entryFor(counts, 0x3004u, go::kOrdinary).visible = 600;
    go::entryFor(counts, kGlassId, go::kOrdinary).visible = 400;
    go::entryFor(counts, 0x3005u, go::kOrdinary).visible = 50;
    check(counts.objects.size() == 3 && counts.objects[0].object == 0x3004u
        && counts.objects[0].invocations == 900 && counts.objects[0].nanoseconds == 2500000
        && counts.objects[1].samplesPassed == 700 && counts.objects[0].mode == go::kWater
        && counts.objects[1].mode == go::kVolume && counts.objects[2].mode == go::kOrdinary,
        "overdraw probe merges an object's draws across bands in first-draw order");
    const auto walk = go::walkTotals(counts), water = go::modeTotals(counts, go::kWater),
        volume = go::modeTotals(counts, go::kVolume);
    check(go::totalInvocations(counts) == 2600 && walk.invocations == 2500 && walk.visible == 1000
        && walk.nanoseconds == 5750000 && water.invocations == 900 && water.visible == 600
        && volume.invocations == 1600 && volume.visible == 400 && volume.nanoseconds == 3250000,
        "overdraw probe totals the walk objects and each walk mode");
    check(go::factor(go::totalInvocations(counts), counts.covered) == 2.6,
        "overdraw factor is invocations over covered pixels");
    check(go::factor(walk.invocations, walk.visible) == 2.5,
        "walk factor is walk invocations over visible walk pixels");
    check(go::factor(300, 1000) == 0.3, "an overdraw factor below one is not clamped");
    check(go::factor(5, 0) == 0.0, "overdraw factor is zero when nothing is covered");
    check(go::formatLine(counts) == "[gl] glass overdraw: invocations=2600 covered=1000 factor=2.600 "
        "walk-invocations=2500 walk-visible=1000 walk-factor=2.500 walk-ms=5.75 "
        "water-invocations=900 water-visible=600 water-ms=2.50 "
        "volume-invocations=1600 volume-visible=400 volume-ms=3.25 "
        "per-object=[water12292:900/700/600:ms=2.50,volume12291:1600/700/400:ms=3.25,"
        "obj12293:100/100/50:ms=0.25] frame=672x378 bands=2\n",
        "overdraw probe log line format");
    check(go::shouldLog(1) && !go::shouldLog(2) && !go::shouldLog(30) && go::shouldLog(31)
        && !go::shouldLog(0), "overdraw probe logs the first and every 30th measured frame");
}

// ARBIT_GL_GLASS_PREVIEW_VISITS: only a whole number from 1 to 64 sets the Fast
// Preview walk cap, and the cap reaches the Preview text's define.
void previewVisitsOverride()
{
    for (const char* ignored : { (const char*) nullptr, "", "0", "65", "-3", "+8", "8x", " 8", "8 ", "1.5", "abc", "99999999999" })
        check(!sg::parsePreviewCameraVisitsOverride(ignored).has_value(),
              std::string("preview visits text is ignored: ") + (ignored ? ignored : "(unset)"));
    const std::pair<const char*, std::uint32_t> accepted[] { {"1", 1u}, {"12", 12u}, {"24", 24u}, {"64", 64u} };
    for (const auto& [text, value] : accepted)
    {
        const auto parsed = sg::parsePreviewCameraVisitsOverride(text);
        check(parsed && *parsed == value, std::string("preview visits text parses: ") + text);
    }
    const std::string anchor = "        for(int visit=0;visit<64;++visit) {\n";
    const std::string text = "    for(int channel=0;channel<3;++channel) {\n        float wavelength=channel==0?0.6563:channel==1?0.5461:0.4358;\n"
        "        float ior=max(1.0,uSurfaceCoating.y+uGlassTransport.y*(1.0/(wavelength*wavelength)-1.0/(0.58756*0.58756)));\n"
        "        // backs: where retry 3 looks (seamSpawnBack), -direction for the seed.\n"
        "        vec3 origins[16],directions[16],backs[16];float weights[16];int depths[16];bool media[16];int seams[16];\n" + anchor
        + "if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}"
        + "if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}"
        + "radiance[channel]+=weight*glassEnvironment(direction GX_PASS)[channel];continue;"
        + "radiance[channel]+=weight*receiver[channel];continue;";
    const auto standard = sg::previewCameraTransportShader(text);
    const auto overridden = sg::previewCameraTransportShader(text, 24);
    check(standard.find("#define GLASS_PREVIEW_VISITS 6\n") != std::string::npos, "the default Preview cap stays 6");
    check(overridden.find("#define GLASS_PREVIEW_VISITS 24\n") != std::string::npos
              && overridden.find("#define GLASS_PREVIEW_VISITS 6") == std::string::npos,
          "an override cap replaces the Preview define");
    // A walk of v visits from s seeds holds at most s + v branches, so Preview
    // sizes its arrays to that and never past the 16 Export slots.
    check(standard.find("#define GLASS_PREVIEW_PENDING 7\n#define GLASS_PREVIEW_PAIR_PENDING 8\n") != std::string::npos,
          "6 visits size the single walk to 7 slots and the paired walk to 8");
    check(overridden.find("#define GLASS_PREVIEW_PENDING 16\n#define GLASS_PREVIEW_PAIR_PENDING 16\n") != std::string::npos,
          "a cap past 14 visits keeps all 16 slots");
    check(sg::previewCameraTransportShader(text, 1).find("#define GLASS_PREVIEW_PENDING 2\n#define GLASS_PREVIEW_PAIR_PENDING 3\n")
              != std::string::npos, "one visit sizes the walks to 2 and 3 slots");
    std::string oneCheck = text;
    oneCheck.erase(oneCheck.find("if(pending>=16) {"), std::char_traits<char>::length("if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}"));
    check(sg::previewCameraTransportShader(oneCheck).empty(), "a single walk without both slot checks builds no Preview text");
    for (const bool paired : { false, true })
    {
        const auto exported = sg::transportShader(false, true, false, paired);
        const auto preview = sg::previewCameraTransportShader(exported);
        const std::string exportArrays = "vec3 origins[16],directions[16],backs[16];float weights[16];int depths[16];bool media[16];int seams[16];";
        const auto count = [] (const std::string& haystack, const std::string& needle)
        {
            std::size_t found = 0;
            for (auto at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + needle.size())) ++found;
            return found;
        };
        check(!preview.empty() && count(exported, exportArrays) == (paired ? 2u : 1u) && count(preview, exportArrays) == 0
                  && count(exported, "if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}") == 2
                  && count(preview, "if(pending>=GLASS_PREVIEW_PENDING) {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}") == 2
                  && preview.find("int seams[GLASS_PREVIEW_PENDING];") != std::string::npos
                  && exported.find("GLASS_PREVIEW") == std::string::npos,
              std::string("the Preview walks use the compact slots and Export keeps 16: ") + (paired ? "paired" : "single"));
        if (paired)
            check(preview.find("int seams[GLASS_PREVIEW_PAIR_PENDING];") != std::string::npos
                      && preview.find("if(pending>=GLASS_PREVIEW_PAIR_PENDING) {") != std::string::npos
                      && preview.find("for(int slot=1;slot<GLASS_PREVIEW_PAIR_PENDING;++slot)") != std::string::npos
                      && exported.find("for(int slot=1;slot<16;++slot)") != std::string::npos,
                  "the paired walk spills at its compact slot count in Preview and at 16 in Export");
    }
    // The Preview query text hands BVH children back by value; the shared query
    // text, which Accurate and Export compile, keeps the inout stack.
    const auto query = sg::queryShader(false);
    const auto previewQuery = sg::previewQueryShader(query);
    check(!previewQuery.empty() && previewQuery != query
              && previewQuery.find("SC_STACK_ARGS ST_ARGS) {") == std::string::npos
              && previewQuery.find("out int pushFirst,out int pushSecond ST_ARGS) {") != std::string::npos
              && previewQuery.find("if(pushed>0) nodes[pending++]=pushFirst;") != std::string::npos
              && previewQuery.find("if(pushed>1) meshNodes[meshPending++]=pushSecond;") != std::string::npos
              && previewQuery.find(",nodes,pending,capacity ST_PASS)") == std::string::npos
              && query.find("int glassSceneChildren(SV3 origin,SV3 direction,float limit,int node,int escape SC_STACK_ARGS ST_ARGS) {")
                     != std::string::npos
              && query.find("escape,nodes,pending,capacity ST_PASS);") != std::string::npos
              && query.find("pushFirst") == std::string::npos,
          "only the Preview query text drops the inout BVH stack");
    check(sg::previewQueryShader("vec3 unrelated();").empty(), "a query text without the anchors builds no Preview query");
}

// A scene whose glass is all roughness 0 compiles the GL fixture program from the
// smooth-only transport text. That text has no rough walk definitions and keeps
// every smooth walk byte for byte. The full text keeps the rough walks.
void smoothOnlyTransportText()
{
    const auto count = [] (const std::string& haystack, const std::string& needle)
    {
        std::size_t found = 0;
        for (auto at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + needle.size())) ++found;
        return found;
    };
    const auto body = [] (const std::string& text, const std::string& signature)
    {
        const auto at = text.find(signature);
        if (at == std::string::npos) return std::string();
        const auto close = text.find("\n}\n", at);
        return close == std::string::npos ? std::string() : text.substr(at, close + 3 - at);
    };
    const char* rough[] { "vec3 glassRoughNormal(", "float glassRoughG1(", "void glassRoughInterface(",
        "vec3 glassDirectSpecular(", "vec3 roughThinInterfaceTransport(", "vec3 roughGlassTransport(",
        "uint glassMsChildHistory(", "uint glassMsEntryKey(", "uint glassMsSeed(", "float glassMsRandom(",
        "float glassMsLambda(", "float glassMsEscape(", "vec3 glassMsNormal(",
        "int glassMsWalk(", "vec3 roughGlassTransportV2(", "uint glassCameraPixelKey(", "vec3 roughCameraTransport(" };
    for (const bool paired : { false, true })
        for (const bool preview : { false, true })
        {
            const std::string variant = std::string(paired ? "paired" : "single") + (preview ? " Preview" : " export");
            auto full = sg::transportShader(false, true, false, paired);
            if (preview) full = sg::previewCameraTransportShader(full);
            const auto smooth = sg::smoothOnlyCameraTransportShader(full);
            check(!full.empty() && !smooth.empty() && smooth.size() < full.size(),
                  "the smooth-only transport text is built and shorter: " + variant);
            for (const char* definition : rough)
                check(count(full, definition) == 1 && count(smooth, definition) == 0,
                      "only the full transport text defines " + std::string(definition) + " " + variant);
            std::vector<std::string> kept { "float glassFresnel(", "vec3 glassEnvironment(", "vec3 glassReceiver(",
                "vec3 thinInterfaceReceiver(", "int glassRoughReason(", "bool thinEntryOwned(" };
            if (paired)
                for (const char* name : { "vec3 thinSingleInterfaceTransport(", "vec3 glassSingleTransport(",
                         "vec3 glassPairWalk(", "vec3 glassPairLeftover(" })
                    kept.emplace_back(name);
            kept.emplace_back("vec3 thinInterfaceTransport(vec3 camera,vec3 entry,vec3 evaluatedNormal,out bool complete,out int status GX_ARGS) {");
            kept.emplace_back("vec3 glassTransport(vec3 camera,vec3 entry,out bool complete,out int status GX_ARGS) {");
            for (const auto& signature : kept)
            {
                const auto before = body(full, signature);
                check(!before.empty() && before == body(smooth, signature),
                      "the smooth-only text keeps " + signature + " byte for byte: " + variant);
            }
        }
    const auto once = sg::smoothOnlyCameraTransportShader(sg::transportShader(false, true));
    check(sg::smoothOnlyCameraTransportShader(once).empty(), "a text without the rough anchors builds no smooth-only text");
    check(sg::smoothOnlyCameraTransportShader("vec3 unrelated();").empty(), "an unrelated text builds no smooth-only text");

    const auto program = [] (std::uint32_t mode, float roughness)
    {
        auto p = std::make_shared<Program>();
        p->parameters.identifiers[1] = mode;
        p->parameters.emissionRoughness = {0, 0, 0, roughness};
        return p;
    };
    const auto closed = std::uint32_t(surfacematerial::TransportMode::ClosedVolume);
    const auto thin = std::uint32_t(surfacematerial::TransportMode::ThinInterface);
    check(!sg::smoothOnlyCameraGlass(nullptr), "no program is not smooth-only glass");
    check(!sg::smoothOnlyCameraGlass(program(0, 0).get()), "a scene without glass is not smooth-only glass");
    check(sg::smoothOnlyCameraGlass(program(closed, 0).get()), "smooth closed glass is smooth-only glass");
    check(!sg::smoothOnlyCameraGlass(program(closed, .05f).get()), "rough closed glass keeps the rough text");
    check(!sg::smoothOnlyCameraGlass(program(closed, std::nanf("")).get()), "NaN roughness keeps the rough text");
    auto root = program(0, 1);
    root->objectPrograms.push_back(program(thin, 0));
    root->objectPrograms.push_back(program(closed, 0));
    check(sg::smoothOnlyCameraGlass(root.get()), "smooth water beside smooth closed glass is smooth-only glass");
    root->objectPrograms.push_back(program(closed, .35f));
    check(!sg::smoothOnlyCameraGlass(root.get()), "one rough glass child keeps the rough text");
}

// The lighthouse roof is a smooth sphere the water reflects. A receiver's camera
// walk hit carries its interpolated normal, the crossing keeps the face normal,
// and a flat receiver stores no normals, so its query and its shading stay as they were.
void smoothReceiverNormals()
{
    const auto scene = [](bool smooth) {
        const auto owned = newScene();
        addGlass(*owned, Glass::Solid);
        receiverBox(*owned, kFloor, 20, {-3, -1.5f, -3}, {3, -1.01f, 3});
        if (smooth)
            for (std::size_t v = 0; v < owned->vertexCount; ++v)
                if (auto& vertex = owned->vertices[v]; vertex.normal.y == 1 && vertex.position.y == -1.01f && vertex.position.x > 0)
                    vertex.normal = {0.6f, 0.8f, 0};
        return admit(*owned);
    };
    const auto down = [](const auto& admitted) {
        return ov::closestHitCpu(admitted.prepared.query, admitted.frame, {2.5f, 0, 2.5f}, {0, -1, 0}, 0.0f, 3.402823e38f);
    };
    const auto flat = scene(false);
    check(flat.ok && ! flat.prepared.smoothReceiverNormals, "a flat receiver is admitted without normals: " + flat.error);
    if (flat.ok)
    {
        const auto hit = down(flat);
        check(hit.status == 1 && hit.objectId == kFloor && hit.shadingNormal == hit.outwardNormal,
              "a flat receiver shades with exactly its face normal");
    }
    const auto smooth = scene(true);
    check(smooth.ok && smooth.prepared.smoothReceiverNormals, "a smooth receiver is admitted: " + smooth.error);
    if (smooth.ok)
    {
        const auto hit = down(smooth);
        const auto length = std::sqrt(hit.shadingNormal[0] * hit.shadingNormal[0]
            + hit.shadingNormal[1] * hit.shadingNormal[1] + hit.shadingNormal[2] * hit.shadingNormal[2]);
        check(hit.status == 1 && hit.objectId == kFloor && hit.shadingNormal[0] > 0 && std::abs(length - 1) < 1e-5f,
              "a smooth receiver's hit interpolates its vertex normals");
        check(hit.outwardNormal[0] == 0 && hit.outwardNormal[1] == 1 && hit.outwardNormal[2] == 0,
              "the same hit keeps its face normal for crossings");
        check(smooth.prepared.query.texels.size() > flat.prepared.query.texels.size(),
              "only the smooth receiver's faces add normal blocks");
    }
}

int main()
{
    smoothOnlyTransportText();
    smoothReceiverNormals();
    previewVisitsOverride();
    hollowDomeAdmitted();
    cavityCube();
    touchingAndInterpenetrating();
    grazingAndCoplanar();
    shellRefusals();
    singleShell();
    largeClosedGlassVolume();
    overdrawProbeLine();
    std::printf("glass nested shell: %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
