#pragma once

// The three starters that wire Geometry straight into Render 3D (slice
// geolight2), as scenes the helper can draw without the plugin: Spectral Ring
// and Grid Instancer, Score-Deformed Imported Mesh and Audio Surface Instances.
// Sizes, counts and colors follow plugin/Source/graph/BuiltInVisualModules.h.
// Each scene stands in for the starter's lowered plan, and the rig below stands
// in for what the starter's Camera and Light nodes lower to over the default
// lighting. The real plans render through ARBIT_GEOLIGHT_PLANS.

#include "../constructed_field_fixture.h"
#include "../score_field_fixture.h"
#include "../../../shared/GeometryCoreRenderSettings.h"
#include "../../src/block_b_defs.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>

namespace videohelper::starterscenes
{
using namespace videowire::geometry;

struct Scene
{
    std::string name;
    PortContract contract;
    ValueDescriptor value;
    bool spectral = false;
    bool score = false;
};

inline PortContract instanceContract (const ValueDescriptor& value)
{
    PortContract contract;
    contract.carrier = CarrierKind::instances3D;
    contract.overflow = OverflowPolicy::reject;
    contract.maxInstances = 64;
    contract.maxAttributes = 32;
    return withAttributeContract (contract, value);
}

// Instance Color and Instance Emission as constant fields over every instance.
inline std::optional<ValueDescriptor> appearance (const ValueDescriptor& instances, std::array<double, 3> color,
                                                  std::array<double, 3> emission, StableId firstId, std::string& error)
{
    const auto fieldContract = constructedFieldContract (Domain::instance, ValueType::color);
    FieldOperationSettings settings;
    settings.domain = Domain::instance;
    settings.valueType = ValueType::color;
    settings.values = { color[0], color[1], color[2], 1, 1, 0, 0, 0 };
    const auto tint = lowerConstructedField (firstId, OperationCode::fieldConstant, settings, nullptr, nullptr, nullptr,
                                             fieldContract, error);
    settings.values = { emission[0], emission[1], emission[2], 1, 1, 0, 0, 0 };
    const auto glow = lowerConstructedField (firstId + 1, OperationCode::fieldConstant, settings, nullptr, nullptr,
                                             nullptr, fieldContract, error);
    if (! tint || ! glow) return std::nullopt;
    const auto base = instanceContract (instances);
    auto colored = lowerInstanceAppearance (instances, *tint, nullptr, firstId + 2, "color", base, error);
    return colored ? lowerInstanceAppearance (*colored, *glow, nullptr, firstId + 3, "emission", base, error)
                   : std::nullopt;
}

// An Audio Spectrum Field color on the instances: each is sampled by its index over the 64 bands,
// and every channel is (level * gain + bias) * its multiplier.
struct SpectrumColor
{
    float gain, bias;
    std::array<float, 3> multiplier;
};

// The Band Color and Band Glow of the Spectral Ring starter.
inline SpectrumColor ringTint() { return { 1.6f, 0.3f, { 0.2f, 1.0f, 0.9f } }; }
inline SpectrumColor ringGlow() { return { 1.0f, 0.2f, { 1.0f, 1.4f, 1.5f } }; }

inline std::optional<ValueDescriptor> spectralAppearance (const ValueDescriptor& instances, const SpectrumColor& color,
                                                          const SpectrumColor& emission, StableId firstId, std::string& error)
{
    const auto fieldContract = constructedFieldContract (Domain::instance, ValueType::color);
    FieldOperationSettings settings;
    settings.domain = Domain::instance;
    settings.valueType = ValueType::color;
    settings.mode = 0;
    settings.values = { 0, 1, arbitblockb::kBandLoHz, arbitblockb::kBandHiHz, 0, 0, 0, 0 };
    const auto field = [&] (const SpectrumColor& c, StableId id) {
        spectrum::Binding binding;
        binding.firstBand = 0;
        binding.lastBand = 63;
        binding.gain = c.gain;
        binding.bias = c.bias;
        binding.direction = c.multiplier;
        return lowerAudioSpectrumField (instances, id, binding, settings, fieldContract, error);
    };
    const auto tint = field (color, firstId);
    const auto glow = field (emission, firstId + 1);
    if (! tint || ! glow) return std::nullopt;
    const auto base = instanceContract (instances);
    auto colored = lowerInstanceAppearance (instances, *tint, nullptr, firstId + 2, "color", base, error);
    return colored ? lowerInstanceAppearance (*colored, *glow, nullptr, firstId + 3, "emission", base, error)
                   : std::nullopt;
}

// 64 bars on the ring, one per band, over a unit cube scaled to 0.12.
inline std::optional<Scene> spectralRing (std::string& error)
{
    PortContract meshContract;
    meshContract.carrier = CarrierKind::geometry3D;
    meshContract.overflow = OverflowPolicy::reject;
    meshContract.maxVertices = 24;
    meshContract.maxIndices = 36;
    Scene scene;
    scene.name = "spectral-ring";
    scene.spectral = true;
    scene.contract.carrier = CarrierKind::instances3D;
    scene.contract.overflow = OverflowPolicy::reject;
    scene.contract.maxInstances = 64;
    const auto cube = lowerMeshGenerator (921, 921, 1, OperationCode::cube, { 1, 1, 1, 0 }, meshContract, error);
    if (! cube) return std::nullopt;
    spectrum::Binding binding;
    binding.firstBand = 0;
    binding.lastBand = 63;
    binding.baseScale = { 0.12f, 0.12f, 0.24f };
    binding.scaleResponse = { 0, 3, 0 };
    binding.direction = { 0, 0, 0 };
    binding.attackSeconds = 0.02f;
    binding.releaseSeconds = 0.2f;
    auto value = lowerSpectrumInstancer (*cube, 922, binding, { 1, 3.2f, 3.2f, 8 }, scene.contract, error);
    if (! value) return std::nullopt;
    // Band Color and Band Glow are Audio Spectrum Fields on the instances, as in the starter:
    // each bar samples its own band, so its color and glow follow that band's level.
    if (const auto tinted = spectralAppearance (*value, ringTint(), ringGlow(), 930, error))
        value = tinted;
    else
        error.clear();
    scene.value = std::move (*value);
    scene.contract = instanceContract (scene.value);
    return scene;
}

// The 63 x 88 mm card at the starter's x20 scale: a 24-vertex box, bent by the score.
inline std::optional<Scene> scoreCard (std::string& error)
{
    Scene scene;
    scene.name = "score-card";
    scene.score = true;
    scene.contract.carrier = CarrierKind::geometry3D;
    scene.contract.overflow = OverflowPolicy::reject;
    scene.contract.maxVertices = 4096;
    scene.contract.maxIndices = 24576;
    const auto mesh = lowerMeshGenerator (951, 951, 1, OperationCode::cube, { 1.26f, 1.76f, 0.014f, 0 },
                                          scene.contract, error);
    if (! mesh) return std::nullopt;
    videowire::geometry::scorefield::Binding binding;
    binding.fieldStableId = 952;
    binding.sourceGeometryStableId = 951;
    binding.consumerGeometryStableId = 953;
    binding.scoreSourceStableId = 954;
    binding.mode = videowire::geometry::scorefield::Mode::velocityInfluence;
    binding.radius = 2;
    binding.mapping.xScale = 0.25f;
    binding.mapping.yScale = 0.15f;
    binding.mapping.zScale = 0.1f;
    binding.direction = { 0, 0, 1 };
    binding.gain = 0.4f;
    for (const auto& p : std::get<GeometryData> (mesh->data).positions)
        binding.positions.push_back ({ p.x, p.y, p.z });
    binding.elementIds = std::get<GeometryData> (mesh->data).vertexIds;
    scene.value = *mesh;
    scene.value.stableId = 953;
    scene.value.scoreFields.push_back (binding);
    scene.value.operations.push_back ({ 952, OperationCode::evaluateField, 952, 952, {} });
    scene.value.operations.push_back ({ 953, OperationCode::transformTRS, 951, 953, {} });
    scene.value.dispatchCount += 2;
    return scene;
}

// 36 cubes of 0.16 over a 2.4 unit surface, tinted and glowing at the resting emission.
inline std::optional<Scene> audioSurface (std::string& error)
{
    PortContract meshContract = constructedFieldMeshContract();
    Scene scene;
    scene.name = "audio-surface";
    scene.spectral = true;
    auto surface = lowerGrid (941, 941, 1, 6, 6, 0.4f, meshContract, error);
    auto shape = lowerMeshGenerator (942, 942, 1, OperationCode::cube, { 0.16f, 0.16f, 0.16f, 0 }, meshContract, error);
    PortContract pointsContract;
    pointsContract.carrier = CarrierKind::points3D;
    pointsContract.maxPoints = 64;
    auto points = surface ? lowerPointsFromVertices (*surface, 943, pointsContract, error) : std::nullopt;
    PortContract contract;
    contract.carrier = CarrierKind::instances3D;
    contract.maxInstances = 64;
    contract.maxAttributes = 32;
    auto instances = points && shape ? lowerInstanceOnPoints (*points, shape->stableId, 944, contract, error) : std::nullopt;
    if (! instances || ! appendUniqueOperations (instances->operations, shape->operations, error)) return std::nullopt;
    std::rotate (instances->operations.begin(), instances->operations.end() - 1, instances->operations.end());
    instances->dispatchCount = instances->operations.size();
    const auto lit = appearance (*instances, { 0.05, 0.8, 0.9 }, { 0.40, 0.55, 0.60 }, 950, error);
    if (! lit) return std::nullopt;
    scene.value = *lit;
    scene.contract = instanceContract (scene.value);
    return scene;
}

struct Rig
{
    std::string name = "fill";
    std::array<float, 3> eye { 0, 0, 5 };
    float fovDegrees = 45;
    std::array<float, 3> environment { 0, 0, 0 };
    float environmentIntensity = 0;
    // Zero keeps the default key light. Above zero it replaces it, as a Directional Light on the Light input does.
    float key = 0;
    // An analytic sky with a sun, in place of the uniform tint.
    float sunIntensity = 0;
    bool cameraVisible = true;
};

// What the starter's Camera and Environment Light lower to, over the default lighting.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> lowered (const Rig& rig)
{
    using namespace HarmonicMIDI::grid;
    auto lighting = std::make_shared<Visual3DScene>();
    lighting->id = { 1 };
    videowire::geometry::defaultLighting (*lighting);
    auto& camera = lighting->cameras[0];
    camera.id = { 78 };
    camera.transform.translation = { rig.eye[0], rig.eye[1], rig.eye[2] };
    camera.verticalFovRadians = rig.fovDegrees * 3.14159265f / 180.0f;
    lighting->activeCamera = camera.id;
    if (rig.key > 0)
        lighting->lights[0].intensity = rig.key;
    if (rig.key < 0)
        lighting->lightCount = 0; // no key at all
    if (rig.environmentIntensity > 0)
    {
        Visual3DSceneGrowth growth;
        growth.lights = 1;
        std::string error;
        if (! growVisual3DScene (*lighting, growth, error)) return nullptr;
        auto& sky = lighting->lights[lighting->lightCount++];
        sky = {};
        sky.id = { 79 };
        sky.kind = SceneLightKind::Environment;
        sky.color = { rig.environment[0], rig.environment[1], rig.environment[2] };
        sky.intensity = rig.environmentIntensity;
        sky.environmentCameraVisible = rig.cameraVisible;
        if (rig.sunIntensity > 0)
        {
            sky.sky.version = 1;
            sky.sky.zenith = { 0.1f, 0.3f, 0.9f };
            sky.sky.horizon = { 0.6f, 0.7f, 0.95f };
            sky.sky.ground = { 0.08f, 0.08f, 0.1f };
            sky.sky.sunColor = { 1.0f, 0.9f, 0.75f };
            sky.sky.sunIntensity = rig.sunIntensity;
            sky.sky.sunAzimuthDegrees = 30;
            sky.sky.sunElevationDegrees = 40;
            sky.sky.sunRadiusDegrees = 3;
        }
    }
    if (! validateVisual3DScene (*lighting).valid()) return nullptr;
    return lighting;
}
// What a floor starter's stage lowers to (slice geofloor): Scene Compose of the
// Geometry Scene and a 0.1 thick grey slab whose top is floorY, the Environment
// fill on the slab's object and a Directional key on Render 3D, with the camera
// raised and tilted down. Rotations compose as Transform 3D lowers them: X, then Y.
struct Stage
{
    std::array<float, 3> eye { 0, 0, 5 };
    // The camera turns yawDegrees about Y after it tilts, so it looks at the origin from the +X side.
    float yawDegrees = 0;
    float tiltDegrees = 0, fovDegrees = 45, floorY = -1, floorSize = 14;
    float keyElevation = 30, keyAzimuth = 100, keyIntensity = 3.5f, keyAngle = 4;
    std::array<float, 3> fill { 0.55f, 0.62f, 0.75f };
    float fillIntensity = 0.5f;
};

inline HarmonicMIDI::grid::SceneQuaternion stageRotation (float xDegrees, float yDegrees)
{
    const double half = 3.14159265358979 / 360.0;
    const double sx = std::sin (xDegrees * half), cx = std::cos (xDegrees * half);
    const double sy = std::sin (yDegrees * half), cy = std::cos (yDegrees * half);
    return { static_cast<float> (sx * cy), static_cast<float> (cx * sy),
             static_cast<float> (-sx * sy), static_cast<float> (cx * cy) };
}

inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> staged (const Stage& stage)
{
    using namespace HarmonicMIDI::grid;
    auto scene = std::make_shared<Visual3DScene>();
    scene->id = { 1 };
    Visual3DSceneGrowth growth;
    growth.vertices = 24;
    growth.indices = 36;
    growth.objects = 1;
    growth.materials = 1;
    growth.lights = 2;
    growth.cameras = 1;
    std::string error;
    if (! growVisual3DScene (*scene, growth, error)) return nullptr;
    auto& camera = scene->cameras[scene->cameraCount++];
    camera.id = { 78 };
    camera.transform.translation = { stage.eye[0], stage.eye[1], stage.eye[2] };
    camera.transform.rotation = stageRotation (-stage.tiltDegrees, stage.yawDegrees);
    camera.verticalFovRadians = stage.fovDegrees * 3.14159265f / 180.0f;
    camera.nearPlane = 0.1f;
    camera.farPlane = 100.0f;
    scene->activeCamera = camera.id;
    auto& fill = scene->lights[scene->lightCount++];
    fill.id = { 79 };
    fill.kind = SceneLightKind::Environment;
    fill.color = { stage.fill[0], stage.fill[1], stage.fill[2] };
    fill.intensity = stage.fillIntensity;
    auto& key = scene->lights[scene->lightCount++];
    key.id = { 80 };
    key.kind = SceneLightKind::Directional;
    key.transform.rotation = stageRotation (-stage.keyElevation, stage.keyAzimuth);
    key.intensity = stage.keyIntensity;
    key.angularRadius = stage.keyAngle * 3.14159265f / 360.0f;
    auto& material = scene->materials[scene->materialCount++];
    material = {};
    material.id = { 1 };
    material.baseColor = { 0.6f, 0.6f, 0.6f };
    material.roughness = 0.85f;
    auto& object = scene->objects[scene->objectCount++];
    object.id = { 1 };
    object.material = material.id;
    object.transform.translation = { 0, stage.floorY - 0.05f, 0 };
    object.vertexCount = 24;
    object.indexCount = 36;
    const float h = stage.floorSize * 0.5f;
    const std::array<std::array<float, 3>, 3> extent {{ { h, 0, 0 }, { 0, 0.05f, 0 }, { 0, 0, h } }};
    for (int axis = 0; axis < 3; ++axis)
        for (int sign = 1; sign >= -1; sign -= 2)
        {
            const int u = (axis + 1) % 3, v = (axis + 2) % 3;
            const auto base = static_cast<std::uint32_t> (scene->vertexCount);
            for (int corner = 0; corner < 4; ++corner)
            {
                const float a = (corner == 1 || corner == 2) ? 1.0f : -1.0f, b = corner >= 2 ? 1.0f : -1.0f;
                auto& vertex = scene->vertices[scene->vertexCount++];
                float p[3] {};
                for (int c = 0; c < 3; ++c)
                    p[c] = sign * extent[axis][c] + a * extent[u][c] + sign * b * extent[v][c];
                vertex.position = { p[0], p[1], p[2] };
                float n[3] {};
                n[axis] = static_cast<float> (sign);
                vertex.normal = { n[0], n[1], n[2] };
            }
            for (const std::uint32_t corner : { 0u, 1u, 2u, 0u, 2u, 3u })
                scene->indices[scene->indexCount++] = base + corner;
        }
    if (! validateVisual3DScene (*scene).valid()) return nullptr;
    return scene;
}
} // namespace videohelper::starterscenes
