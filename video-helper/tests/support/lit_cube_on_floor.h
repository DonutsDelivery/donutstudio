#pragma once

// A unit cube resting on a floor slab, with the lighting rig Render 3D's
// Camera and a sun on its Light input lower to for Geometry wired straight
// into Render 3D (slice georoute). The sun comes from the right and above, so
// the top is sunlit, the front face is not, and the shadow falls to the left.

#include "../../../shared/GeometryCore.h"
#include "../../../shared/GeometryCoreRenderSettings.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>

namespace videohelper::litcube
{
using namespace HarmonicMIDI::grid;

// The camera: up and in front, looking down at the origin.
inline constexpr SceneVec3 kEye { 0.0f, 2.0f, 5.0f };
inline constexpr float kPitch = -0.3805063771f; // atan (2 / 5), downward
inline constexpr float kFov = 0.7853981634f;

struct Pixel { int x, y; };

// Where a world point lands in a width x height frame, top row first.
inline Pixel project (std::uint32_t width, std::uint32_t height, SceneVec3 p)
{
    const float dx = p.x - kEye.x, dy = p.y - kEye.y, dz = p.z - kEye.z;
    // The inverse of the camera's pitch about X.
    const float c = std::cos (-kPitch), s = std::sin (-kPitch);
    const float vx = dx, vy = dy * c - dz * s, vz = dy * s + dz * c;
    const float t = std::tan (kFov * 0.5f), aspect = static_cast<float> (width) / static_cast<float> (height);
    const float nx = vx / (-vz * t * aspect), ny = vy / (-vz * t);
    return { static_cast<int> ((nx * 0.5f + 0.5f) * static_cast<float> (width)),
             static_cast<int> ((0.5f - ny * 0.5f) * static_cast<float> (height)) };
}

// The rig as the plugin lowers it over the default lighting: the sun replaces
// the default key light and the camera replaces the default camera. With
// environment, an Environment Light joins the sun. Null if the rig is invalid.
inline std::shared_ptr<const Visual3DScene> rig (bool environment)
{
    auto lighting = std::make_shared<Visual3DScene>();
    lighting->id = { 1 };
    videowire::geometry::defaultLighting (*lighting);
    auto& sun = lighting->lights[0];
    sun.id = { 77 };
    // toLight = (0.6, 0.8, 0).
    sun.transform.rotation = { -0.5656854f, 0.4242641f, 0.0f, 0.7071068f };
    sun.intensity = 3.0f;
    auto& camera = lighting->cameras[0];
    camera.id = { 78 };
    camera.transform.translation = kEye;
    camera.transform.rotation = { std::sin (kPitch * 0.5f), 0.0f, 0.0f, std::cos (kPitch * 0.5f) };
    lighting->activeCamera = camera.id;
    if (environment)
    {
        Visual3DSceneGrowth growth;
        growth.lights = 1;
        std::string error;
        if (! growVisual3DScene (*lighting, growth, error)) return nullptr;
        auto& sky = lighting->lights[lighting->lightCount++];
        sky = {};
        sky.id = { 79 };
        sky.kind = SceneLightKind::Environment;
        sky.color = { 0.2f, 0.35f, 0.7f };
        sky.intensity = 1.0f;
        sky.environmentCameraVisible = true;
    }
    if (! validateVisual3DScene (*lighting).valid()) return nullptr;
    return lighting;
}

// A unit cube on a wide, thin floor slab whose top is the cube's base.
inline std::optional<videowire::geometry::ValueDescriptor> cubeOnFloor (videowire::geometry::PortContract& contract,
                                                                        std::string& error)
{
    using namespace videowire::geometry;
    contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
    contract.maxVertices = 48; contract.maxIndices = 72;
    PortContract one = contract;
    one.maxVertices = 24; one.maxIndices = 36;
    const auto cube = lowerMeshGenerator (601, 601, 1, OperationCode::cube, {1, 1, 1, 0}, one, error);
    const auto slab = cube ? lowerMeshGenerator (602, 602, 1, OperationCode::cube, {8, 0.1f, 8, 0}, one, error) : std::nullopt;
    Transform down;
    down.translation = {0, -0.55f, 0};
    const auto floor = slab ? lowerTransformGeometry (*slab, 603, down, one, error) : std::nullopt;
    return floor ? lowerJoinGeometry (*cube, *floor, 604, contract, error) : std::nullopt;
}
} // namespace videohelper::litcube
