// Lighthouse Glass Caustics (visual.lighthouse-glass-caustics) as the starter
// ships it: water, tower, lantern, the glass dome (8-segment, 4-ring sphere
// shell joined with an inward box cavity, 60 triangles) and the rock, with the
// Render node's caustic policy (48 interfaces, 64 visits, 16 pending). The
// case runs the CPU half of the OpenGL caustic frame at 960x540, the size the
// live probe used, prints every host and device buffer the frame sizes, and
// checks each one follows from the scene: work from the dome's entries, deposit
// slots from that work and the policy's visits, spectra from the pixels. It
// needs no GPU.
//
// On 6 Nov this frame admitted 348 MB on the CPU and 169 MB of GL buffers. The
// same scene rendered on the RTX 4070 through the GL backend peaked at 1.85 GB
// resident, all of it in the driver's shader compiles; no single host
// allocation reached 32 MiB. The 10.5 GB the live helper reached for this
// starter is outside the caustic frame.

#include "support/glass_caustic_oracle.h"
#include "../src/gpu_backend/fixture_glass_caustics_opengl.h"
#include "../src/gpu_backend/fixture_smooth_glass.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>


namespace
{
namespace gt = glasscaustictests;
namespace gc = arbitgpu::glasscaustics;
namespace sg = arbitgpu::smoothglass;
using namespace HarmonicMIDI::grid;

int failures = 0, checks = 0;
void check(bool condition, const std::string& text)
{
    ++checks;
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", text.c_str()); }
}

std::vector<arbitgpu::opaquevisibility::Instance> instances(const Visual3DScene& scene)
{
    std::vector<arbitgpu::opaquevisibility::Instance> output;
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        auto m = videohelper::tests::sceneclosest::identity();
        const auto& p = scene.objects[o].transform.translation;
        m[12] = p.x; m[13] = p.y; m[14] = p.z;
        output.push_back({m, o});
    }
    return output;
}

// An axis-aligned box centred on the object origin plus offset, every face
// wound outward, or inward when inward is set (the mirrored cavity).
void box(Visual3DScene& scene, float w, float h, float d, SceneVec3 offset, bool inward)
{
    struct Face { float n[3], u[3], v[3]; };
    const Face faces[6] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}};
    const float half[3] = {w / 2, h / 2, d / 2}, at[3] = {offset.x, offset.y, offset.z};
    const float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    const float sign = inward ? -1.0f : 1.0f;
    for (std::size_t f = 0; f < 6; ++f)
    {
        const auto base = std::uint32_t(scene.vertexCount);
        for (std::size_t c = 0; c < 4; ++c)
        {
            auto& vertex = scene.vertices[scene.vertexCount++];
            float p[3];
            for (std::size_t a = 0; a < 3; ++a)
                p[a] = at[a] + half[a] * (faces[f].n[a] + corners[c][0] * faces[f].u[a] + corners[c][1] * faces[f].v[a]);
            vertex = {};
            vertex.position = {p[0], p[1], p[2]};
            vertex.normal = {sign * faces[f].n[0], sign * faces[f].n[1], sign * faces[f].n[2]};
        }
        const std::uint32_t outward[6] = {0, 1, 2, 0, 2, 3}, reversed[6] = {0, 2, 1, 0, 3, 2};
        for (const auto i : inward ? reversed : outward) scene.indices[scene.indexCount++] = base + i;
    }
}

void object(Visual3DScene& scene, std::size_t o, SceneVec3 at)
{
    auto& item = scene.objects[o];
    item.firstVertex = std::uint32_t(scene.vertexCount);
    item.firstIndex = std::uint32_t(scene.indexCount);
    item.transform.translation = at;
}
void close(Visual3DScene& scene, std::size_t o)
{
    auto& item = scene.objects[o];
    item.vertexCount = std::uint32_t(scene.vertexCount) - item.firstVertex;
    item.indexCount = std::uint32_t(scene.indexCount) - item.firstIndex;
    // Indices are relative to the object's first vertex.
    for (std::size_t i = item.firstIndex; i < scene.indexCount; ++i) scene.indices[i] -= item.firstVertex;
}

// The dome: the Geometry Core sphere (radius 1.68, 8 segments, 4 rings) turned
// 22.5 degrees about y, joined with the 0.99 x 2.44 x 0.99 cavity mirrored in x
// and lowered 0.05, at (-1.2, 1.85, -1.2), as BuiltInVisualModules.h authors it.
bool dome(Visual3DScene& scene, std::size_t o)
{
    namespace geometry = videowire::geometry;
    geometry::OperationRecord operation;
    operation.code = geometry::OperationCode::sphere;
    operation.parameters = {1.68f, 8, 4, 0};
    geometry::GeometryData mesh;
    std::string error;
    if (!geometry::materializeMeshGenerator(operation, mesh, geometry::kMaximumVertices, geometry::kMaximumIndices, error))
    { check(false, "dome sphere generator: " + error); return false; }
    object(scene, o, {-1.2f, 1.85f, -1.2f});
    const double turn = 22.5 * 3.14159265358979323846 / 180;
    const float c = float(std::cos(turn)), s = float(std::sin(turn));
    for (const auto& p : mesh.positions)
    {
        auto& vertex = scene.vertices[scene.vertexCount++];
        vertex = {};
        vertex.position = {c * p.x + s * p.z, p.y, -s * p.x + c * p.z};
        vertex.normal = {vertex.position.x / 1.68f, vertex.position.y / 1.68f, vertex.position.z / 1.68f};
    }
    const auto first = std::uint32_t(scene.vertexCount - mesh.positions.size());
    for (const auto index : mesh.indices) scene.indices[scene.indexCount++] = first + index;
    box(scene, 0.99f, 2.44f, 0.99f, {0, -0.05f, 0}, true);
    close(scene, o);
    return true;
}

struct Starter
{
    std::shared_ptr<Visual3DScene> scene;
    std::shared_ptr<gt::Program> root;
    SceneObjectId glass;
    SceneMaterialId glassMaterial;
    SceneLightId sun;
};

Starter starter()
{
    Starter s;
    s.scene = std::make_shared<Visual3DScene>();
    auto& scene = *s.scene;
    scene.id = {7};
    const std::uint32_t objectIds[5] = {101, 102, 103, 104, 105};
    const std::uint32_t materialIds[5] = {1001, 1003, 1004, 1002, 1005};
    for (std::size_t o = 0; o < 5; ++o)
    {
        scene.objects[o].id = {objectIds[o]};
        scene.objects[o].material = {materialIds[o]};
        scene.materials[o].id = {materialIds[o]};
        scene.materials[o].baseColor = {1, 1, 1};
        scene.materials[o].roughness = 1;
        scene.materials[o].opacity = 1;
    }
    scene.objectCount = scene.materialCount = 5;
    // Ocean Grid 12 x 12 at spacing 0.7 with the starter's wave at time zero.
    object(scene, 0, {0, 0, 0});
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
        {
            const float x = (float(i) - 5.5f) * 0.7f, z = (float(j) - 5.5f) * 0.7f;
            const float phase = 2.0f * 3.14159265f * (x * 0.94f + z * 0.34f) / 2.8f;
            const float slope = 0.09f * 2.0f * 3.14159265f / 2.8f * std::cos(phase);
            const float nx = -slope * 0.94f, nz = -slope * 0.34f, length = std::sqrt(nx * nx + 1 + nz * nz);
            auto& vertex = scene.vertices[scene.vertexCount++];
            vertex = {};
            vertex.position = {x, 0.09f * std::sin(phase), z};
            vertex.normal = {nx / length, 1 / length, nz / length};
        }
    for (std::uint32_t i = 0; i < 11; ++i)
        for (std::uint32_t j = 0; j < 11; ++j)
        {
            const std::uint32_t v00 = i * 12 + j, v01 = v00 + 1, v10 = v00 + 12, v11 = v10 + 1;
            for (const auto v : {v00, v01, v11, v00, v11, v10}) scene.indices[scene.indexCount++] = v;
        }
    close(scene, 0);
    object(scene, 1, {-1.2f, 1.6f, -1.2f}); box(scene, 0.95f, 2.0f, 0.95f, {}, false); close(scene, 1);
    object(scene, 2, {-1.2f, 2.8f, -1.2f}); box(scene, 0.95f, 0.4f, 0.95f, {}, false); close(scene, 2);
    dome(scene, 3);
    object(scene, 4, {-1.2f, -0.205f, -1.2f}); box(scene, 1.9f, 0.65f, 1.7f, {}, false); close(scene, 4);
    s.glass = scene.objects[3].id;
    s.glassMaterial = scene.objects[3].material;

    scene.lightCount = 2;
    scene.lights[0].id = {301};
    scene.lights[0].kind = SceneLightKind::Environment;
    scene.lights[0].color = {.125f, .25f, .5f};
    auto& sun = scene.lights[1];
    sun.id = {302};
    sun.kind = SceneLightKind::Directional;
    sun.color = {1, 0.9f, 0.75f};
    sun.intensity = 3.5f;
    const float ax = -35.0f * 3.14159265f / 180, ay = -25.0f * 3.14159265f / 180;
    const float sx = std::sin(ax / 2), cx = std::cos(ax / 2), sy = std::sin(ay / 2), cy = std::cos(ay / 2);
    sun.transform.rotation = {cy * sx, sy * cx, -sy * sx, cy * cx};
    sun.spectralSource.version = 1;
    sun.spectralSource.radiance.fill(1.0f);
    sun.spectralSource.traceMaximumDistance = 1000;
    s.sun = sun.id;
    scene.cameraCount = 1;
    scene.cameras[0].id = {401};
    scene.activeCamera = {401};
    scene.cameras[0].transform.translation = {0, 2.4f, 7};
    const float tilt = -14.0f * 3.14159265f / 180;
    scene.cameras[0].transform.rotation = {std::sin(tilt / 2), 0, 0, std::cos(tilt / 2)};
    scene.cameras[0].verticalFovRadians = 52.0f * 3.14159265f / 180;
    scene.cameras[0].nearPlane = 0.1f;
    scene.cameras[0].farPlane = 100;

    const auto program = [&](std::size_t o) {
        auto p = std::make_shared<gt::Program>();
        p->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
        p->object = scene.objects[o].id;
        p->bindingDigest = "dome-binding-" + std::to_string(o);
        p->programIdentity = "dome-program-" + std::to_string(o);
        p->authoredScene = scene.id;
        p->authoredMaterial = scene.objects[o].material;
        p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
        p->parameters.identifiers[0] = p->authoredMaterial.value;
        p->parameters.normalOpacity = {0, 0, 1, 1};
        return p;
    };
    const auto receiver = [&](std::size_t o, double r, double g, double b) {
        auto p = program(o);
        p->parameters.baseColorMetallic = {float(r), float(g), float(b), 0};
        p->parameters.transmissionIorClearcoat = {0, 1.5f, 0, 0};
        p->parameters.emissionRoughness = {0, 0, 0, 1};
        p->spectralReflectance = surfacematerial::SpectralReflectanceDescription{};
        p->spectralReflectance->coefficients = {float(b), float(b), float(b), float(0.5 * (g + b)), float(g),
                                                float(0.5 * (r + g)), float(r), float(r)};
        return p;
    };
    auto water = program(0);
    water->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ThinInterface);
    water->parameters.baseColorMetallic = {1, 1, 1, 0};
    water->parameters.transmissionIorClearcoat = {1, 1.33f, 0, 0};
    water->parameters.emissionRoughness = {0, 0, 0, 0};
    auto glass = program(3);
    glass->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ClosedVolume);
    glass->parameters.baseColorMetallic = {1, 1, 1, 0};
    glass->parameters.transmissionIorClearcoat = {1, 1.5f, 0, 0};
    glass->parameters.emissionRoughness = {0, 0, 0, 0};
    glass->objectPrograms = {water, receiver(1, 0.8, 0.72, 0.58), receiver(2, 0.95, 0.62, 0.2),
                             receiver(4, 0.12, 0.15, 0.18)};
    s.root = glass;
    return s;
}

void print(const char* name, std::uint64_t value)
{
    std::printf("  %-34s %llu\n", name, static_cast<unsigned long long>(value));
}

void domeSizing()
{
    auto s = starter();
    const std::shared_ptr<const Visual3DScene> scene = s.scene;
    const std::shared_ptr<const gt::Program> root = s.root;
    check(validateVisual3DScene(*scene).valid(), "the dome starter scene is a valid snapshot");
    check(scene->objects[3].indexCount / 3 == 60, "the dome is 60 triangles: "
          + std::to_string(scene->objects[3].indexCount / 3));
    const sg::CausticVolumeSelection selection {s.glass, s.glassMaterial};
    std::string error;
    sg::Prepared prepared;
    if (!sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &selection))
    { check(false, "dome starter smooth glass prepare: " + error); return; }
    check(sg::admitFrame(prepared, 960, 540, false, error), "960x540 closed glass frame: " + error);

    gc::BranchPolicyDescription description;
    description.version = gc::kBranchPolicyVersion;
    description.maximumUnresolvedFraction = 1e-4f;
    description.minimumBranchFraction = 1e-6f;
    description.maximumInterfaces = 48;
    description.maximumVisits = 64;
    description.maximumPending = 16;
    const auto policy = gc::AdmittedBranchPolicy::admit(description, error);
    if (!policy) { check(false, "starter policy: " + error); return; }
    gc::Prepared caustic;
    if (!gc::prepare(prepared, s.sun, *policy, caustic, error))
    { check(false, "dome starter caustic prepare: " + error); return; }
    gc::Frame frame;
    if (!gc::buildScheduledFrame(caustic, instances(*scene), nullptr, 960, 540, {}, frame, error))
    { check(false, "dome starter scheduled frame: " + error); return; }
    std::string splatError;
    const bool splat = gc::enableSplat(caustic, frame, splatError);
    gc::opengl::SplatScratchLayout layout;
    const bool laidOut = splat && gc::opengl::splatScratchLayout(frame, layout);
    std::uint64_t admitted = 0;
    check(gc::admitFrameMemory(frame, 0, admitted, error), "dome frame memory: " + error);

    const std::uint64_t pixels = 960ull * 540ull;
    const auto keyCount = splat ? frame.splat.paddedKeyCount : frame.bounds.deposition.paddedKeyCount;
    const std::uint64_t glBytes[7] {
        (frame.work.size() * gc::kPathReceiptStride + frame.bounds.historyTexels) * sizeof(gc::Texel),
        frame.bounds.maximumDeposits * gc::kDepositReceiptStride * sizeof(gc::Texel),
        pixels * 4 * sizeof(float), frame.workTexels.size() * sizeof(gc::Texel),
        frame.triangleTexels.size() * sizeof(gc::Texel), keyCount * sizeof(gc::DepositionKey), pixels * 4 * sizeof(float)};
    std::printf("Lighthouse Glass Caustics dome starter at 960x540 (splat %s%s)\n", splat ? "on" : "off: ",
                splat ? "" : splatError.c_str());
    print("triangles", frame.triangles.size());
    print("entries", frame.entries.size());
    print("work paths", frame.work.size());
    print("work texels", frame.workTexels.size());
    print("triangle texels", frame.triangleTexels.size());
    print("query texels", frame.query.texels.size());
    print("bounds.maximumDeposits", frame.bounds.maximumDeposits);
    print("bounds.historyExtraWords", frame.bounds.historyExtraWords);
    print("bounds.historyTexels", frame.bounds.historyTexels);
    print("bounds.ancestryExtraWords", frame.bounds.ancestryExtraWords);
    print("bounds.deposition.paddedKeyCount", frame.bounds.deposition.paddedKeyCount);
    print("bounds.deposition.scratchBytes", frame.bounds.deposition.scratchBytes);
    print("bounds.transientBytes", frame.bounds.transientBytes);
    print("splat.paddedKeyCount", frame.splat.paddedKeyCount);
    print("splat.scratchBytes", frame.splat.scratchBytes);
    print("splat scratch words", laidOut ? layout.words : 0);
    print("admitted frame bytes", admitted);
    // Every buffer follows from the scene. Each dome entry traces one path per
    // interface sample and spectral band, each path keeps one deposit slot per
    // policy visit, and the two spectra hold one RGBA32F texel per pixel.
    check(frame.work.size() == frame.entries.size() * frame.interfaceSamples * 8,
          "dome work is its entries times samples times 8 bands: " + std::to_string(frame.work.size()));
    check(frame.bounds.maximumDeposits == frame.work.size() * description.maximumVisits,
          "dome deposit slots are its paths times the policy's visits: " + std::to_string(frame.bounds.maximumDeposits));
    // Each deposit slot charges at most 2 KiB: its 512 byte receipt, its sort keys
    // and its splat scratch. Each pixel charges at most 64 bytes of spectra.
    // Measured: 348197092 bytes against this bound of 536494080.
    const std::uint64_t bound = frame.bounds.maximumDeposits * 2048 + pixels * 64;
    check(admitted <= bound, "dome frame memory " + std::to_string(admitted)
          + " stays within its deposit and pixel need " + std::to_string(bound));
    static constexpr const char* names[7] {"gl path receipts", "gl deposit receipts", "gl low spectrum",
        "gl work table", "gl triangle table", "gl keys", "gl high spectrum"};
    std::uint64_t deviceBytes = 0;
    for (unsigned i = 0; i < 7; ++i) { print(names[i], glBytes[i]); deviceBytes += glBytes[i]; }
    print("gl caustic device total", deviceBytes);
    if (laidOut)
    {
        try
        {
            const auto words = gc::opengl::splatScratchWords(frame, layout);
            check(words.size() == layout.words, "splat scratch words match the layout");
        }
        catch (const std::bad_alloc&) { check(false, "splat scratch words allocation failed"); }
    }
}

} // namespace

int main()
{
    try { domeSizing(); }
    catch (const std::exception& failure) { check(false, std::string("uncaught: ") + failure.what()); }
    std::printf("lighthouse_glass_dome_sizing: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
