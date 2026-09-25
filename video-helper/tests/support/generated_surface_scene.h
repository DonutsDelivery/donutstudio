#pragma once

#include "../../../shared/GeometryCoreScene.h"
#include "../../../shared/GeometryCoreTransport.h"

namespace videohelper::tests
{
// Feed the exact generated cube operation through Geometry Core transport and
// admission, then the same retained Scene3D adapter used by Geometry Object.
// Lighting, transforms and material identities come from the independent fixture.
inline std::optional<HarmonicMIDI::grid::Visual3DScene> generatedSurfaceScene(
    const HarmonicMIDI::grid::Visual3DScene& lighting, std::string& error)
{
    using namespace videowire::geometry;
    PortContract contract;
    contract.carrier = CarrierKind::geometry3D;
    contract.maxVertices = 24;
    contract.maxIndices = 36;
    contract.overflow = OverflowPolicy::reject;
    auto cube = lowerMeshGenerator(9001, 9001, 1, OperationCode::cube,
                                   {2, 2, 2, 0}, contract, error);
    if (!cube) return std::nullopt;
    auto admitted = admitValue(std::move(*cube), contract, {}, error);
    if (!admitted) return std::nullopt;
    const auto encoded = lowerRuntimePlan(contract, *admitted);
    const auto decoded = decodeLoweredRuntimePlan(encoded, error);
    if (!decoded) return std::nullopt;
    const auto restored = decodeRuntimeValue(decoded->runtimeValue, decoded->contract, {}, {}, error);
    if (!restored) return std::nullopt;
    auto scene = lighting;
    const auto object = scene.objects[0];
    const auto material = scene.materials[0];
    scene.vertexCount = scene.indexCount = scene.objectCount = scene.materialCount = 0;
    if (!appendGeometryToScene(*restored, object, material, scene, error)
        || !HarmonicMIDI::grid::validateVisual3DScene(scene).valid()) return std::nullopt;
    return scene;
}
}
