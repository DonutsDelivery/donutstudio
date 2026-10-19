#pragma once

// The Smooth Dispersive Glass starter scene, built in C++ for the Path Trace
// engine tests and the arbit-pathtrace-timing tool: a glass cube resting on a
// floor, two backdrops behind it, a sun and an analytic sky.
//
// The glass is the starter's material (plugin/Source/graph/
// SurfaceMaterialStarterModules.h, smoothDispersiveGlassStarter: base colour 1,
// metallic 0, roughness 0, transmission 1, IOR 1.5, closed volume, Cauchy B
// 0.004) after the helper's Surface program mapping
// (src/pathtrace_fixture_engine.cpp): a smooth dielectric with the d-line IOR.
// Dispersion is not traced until spectral transport lands.

#include "pathtrace/engine.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace pathtraceglassscene
{
namespace grid = HarmonicMIDI::grid;
namespace pt = videohelper::pathtrace;

inline constexpr std::uint32_t kFloorObject = 10;
inline constexpr std::uint32_t kWarmBackdropObject = 11;
inline constexpr std::uint32_t kCoolBackdropObject = 12;
inline constexpr std::uint32_t kGlassObject = 13;
inline constexpr std::uint32_t kGlassMaterialId = 6;  // the starter's Material ID

inline grid::SceneQuaternion axisAngle (float x, float y, float z, float radians) noexcept
{
    const float s = std::sin (radians * 0.5f);
    return { x * s, y * s, z * s, std::cos (radians * 0.5f) };
}

inline grid::SceneQuaternion multiply (grid::SceneQuaternion a, grid::SceneQuaternion b) noexcept
{
    return { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
             a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
             a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
             a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

// Appends one quad face. Corners run counter-clockwise seen from the normal side.
inline void appendFace (grid::Visual3DScene& scene, grid::SceneObjectRecord& object, pt::Vec3 centre,
                        pt::Vec3 halfU, pt::Vec3 halfV)
{
    const pt::Vec3 n = pt::normalizeOrZero (pt::cross (halfU, halfV));
    const auto first = static_cast<std::uint32_t> (scene.vertexCount) - object.firstVertex;
    const pt::Vec3 corners[4] { centre - halfU - halfV, centre + halfU - halfV, centre + halfU + halfV,
                                centre - halfU + halfV };
    for (const pt::Vec3& corner : corners)
    {
        grid::SceneVertex& vertex = scene.vertices[scene.vertexCount++];
        vertex.position = { corner.x, corner.y, corner.z };
        vertex.normal = { n.x, n.y, n.z };
    }
    for (const std::uint32_t index : { 0u, 1u, 2u, 0u, 2u, 3u })
        scene.indices[scene.indexCount++] = first + index;
    object.vertexCount += 4;
    object.indexCount += 6;
}

inline grid::SceneObjectRecord& beginObject (grid::Visual3DScene& scene, std::uint32_t id, std::uint32_t material)
{
    grid::SceneObjectRecord& object = scene.objects[scene.objectCount++];
    object.id.value = id;
    object.material.value = material;
    object.firstVertex = static_cast<std::uint32_t> (scene.vertexCount);
    object.firstIndex = static_cast<std::uint32_t> (scene.indexCount);
    return object;
}

inline void addMaterial (grid::Visual3DScene& scene, std::uint32_t id, pt::Vec3 base, float roughness)
{
    grid::SceneMaterialRecord& record = scene.materials[scene.materialCount++];
    record.id.value = id;
    record.baseColor = { base.x, base.y, base.z };
    record.roughness = roughness;
}

inline std::shared_ptr<const grid::Visual3DScene> makeScene()
{
    auto scene = std::make_shared<grid::Visual3DScene>();
    scene->id.value = 4001;
    addMaterial (*scene, 1, { 0.8f, 0.8f, 0.78f }, 0.6f);   // floor
    addMaterial (*scene, 2, { 0.8f, 0.42f, 0.25f }, 0.8f);  // warm backdrop
    addMaterial (*scene, 3, { 0.22f, 0.42f, 0.8f }, 0.8f);  // cool backdrop
    addMaterial (*scene, kGlassMaterialId, { 1.0f, 1.0f, 1.0f }, 0.0f);

    auto& floor = beginObject (*scene, kFloorObject, 1);
    appendFace (*scene, floor, { 0.0f, 0.0f, 0.0f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -3.0f });
    auto& warm = beginObject (*scene, kWarmBackdropObject, 2);
    appendFace (*scene, warm, { 0.0f, 1.5f, -2.5f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 1.5f, 0.0f });
    auto& cool = beginObject (*scene, kCoolBackdropObject, 3);
    appendFace (*scene, cool, { -2.5f, 1.5f, 0.0f }, { 0.0f, 0.0f, -3.0f }, { 0.0f, 1.5f, 0.0f });

    // A closed unit cube, faces outward, turned 30 degrees about +Y.
    auto& glass = beginObject (*scene, kGlassObject, kGlassMaterialId);
    const float h = 0.5f;
    appendFace (*scene, glass, { h, 0.0f, 0.0f }, { 0.0f, 0.0f, -h }, { 0.0f, h, 0.0f });   // +X
    appendFace (*scene, glass, { -h, 0.0f, 0.0f }, { 0.0f, 0.0f, h }, { 0.0f, h, 0.0f });   // -X
    appendFace (*scene, glass, { 0.0f, h, 0.0f }, { h, 0.0f, 0.0f }, { 0.0f, 0.0f, -h });   // +Y
    appendFace (*scene, glass, { 0.0f, -h, 0.0f }, { h, 0.0f, 0.0f }, { 0.0f, 0.0f, h });   // -Y
    appendFace (*scene, glass, { 0.0f, 0.0f, h }, { h, 0.0f, 0.0f }, { 0.0f, h, 0.0f });    // +Z
    appendFace (*scene, glass, { 0.0f, 0.0f, -h }, { -h, 0.0f, 0.0f }, { 0.0f, h, 0.0f });  // -Z
    glass.transform.translation = { 0.0f, h, 0.0f };
    glass.transform.rotation = axisAngle (0.0f, 1.0f, 0.0f, 0.5235988f);

    // Sun: shines along its local -Z, pitched 50 degrees down and turned 30 degrees.
    grid::SceneLightRecord& sun = scene->lights[scene->lightCount++];
    sun.id.value = 20;
    sun.kind = grid::SceneLightKind::Directional;
    sun.color = { 1.0f, 0.95f, 0.88f };
    sun.intensity = 3.0f;
    sun.transform.rotation = multiply (axisAngle (0.0f, 1.0f, 0.0f, 0.5235988f),
                                       axisAngle (1.0f, 0.0f, 0.0f, -0.8726646f));
    grid::SceneLightRecord& sky = scene->lights[scene->lightCount++];
    sky.id.value = 21;
    sky.kind = grid::SceneLightKind::Environment;
    sky.intensity = 1.0f;
    sky.sky.version = 1;

    grid::SceneCameraRecord& camera = scene->cameras[scene->cameraCount++];
    camera.id.value = 9;
    camera.transform.translation = { 0.0f, 1.2f, 4.0f };
    camera.transform.rotation = axisAngle (1.0f, 0.0f, 0.0f, -0.1732674f);  // looks at the cube
    camera.verticalFovRadians = 0.75f;
    camera.nearPlane = 0.05f;
    camera.farPlane = 50.0f;
    scene->activeCamera.value = 9;
    return scene;
}

// The helper's mapping of the starter's Surface block (closed volume).
inline pt::Material glassMaterial() noexcept
{
    pt::Material material;
    material.baseColor = { 1.0f, 1.0f, 1.0f };
    material.metallic = 0.0f;
    material.roughness = 0.0f;
    material.opacity = 1.0f;
    material.transmission = 1.0f;
    material.ior = 1.5f;
    return material;
}

inline pt::EngineSceneSetup makeSetup (const std::shared_ptr<const grid::Visual3DScene>& scene)
{
    pt::EngineSceneSetup setup;
    setup.scene = scene;
    setup.objectMaterials.push_back ({ kGlassObject, glassMaterial() });
    setup.lights.assign (scene->lights.begin(), scene->lights.begin() + static_cast<std::ptrdiff_t> (scene->lightCount));
    return setup;
}

inline pt::EngineView makeView (const grid::Visual3DScene& scene, std::uint32_t width, std::uint32_t height)
{
    pt::EngineView view;
    view.camera = scene.cameras[0];
    view.width = width;
    view.height = height;
    return view;
}

} // namespace pathtraceglassscene
