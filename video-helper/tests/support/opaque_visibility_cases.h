#pragma once

#include "../../../shared/BaseSurfaceReflection.h"
#include "../../../shared/HdrImageOutputContract.h"
#include "../../../shared/generated/SurfaceMaterialStarterPrograms.h"
#include "../../src/fixture_scene_renderer.h"
#include "../../src/gpu_backend/backend.h"
#include "fixture_scene.h"
#include "vertex_modifier_fixture.h"
#include <cstring>
#include <limits>

namespace videohelper::tests
{
// Independent double-precision brute-force oracle. It has no acceleration
// payload, shader code, renderer intersection helper, or visibility texture.
inline bool opaqueTriangleOracle(const std::array<double, 3>& origin, const std::array<double, 3>& direction,
                                 const std::array<std::array<double, 3>, 3>& triangle, double limit)
{
    const auto sub = [](auto a, auto b)
    {
        for (int i = 0; i < 3; ++i)
            a[i] -= b[i];
        return a;
    };
    const auto dot = [](auto a, auto b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const auto cross = [](auto a, auto b)
    {
        return std::array<double, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                     a[0] * b[1] - a[1] * b[0]};
    };
    const auto e1 = sub(triangle[1], triangle[0]), e2 = sub(triangle[2], triangle[0]);
    const auto normal = cross(e1, e2);
    const auto denominator = dot(normal, direction);
    if (std::abs(denominator) < 1e-14)
        return false;
    const auto distance = dot(normal, sub(triangle[0], origin)) / denominator;
    if (distance <= 0 || distance >= limit)
        return false;
    auto hit = origin;
    for (int a = 0; a < 3; ++a)
        hit[a] += distance * direction[a];
    for (int edge = 0; edge < 3; ++edge)
        if (dot(normal, cross(sub(triangle[(edge + 1) % 3], triangle[edge]), sub(hit, triangle[edge]))) <
            -1e-12)
            return false;
    return true;
}
inline std::array<double, 3> opaqueOracleTransform(std::array<double, 3> p,
                                                   const HarmonicMIDI::grid::SceneTransform3D& transform)
{
    for (int axis = 0; axis < 3; ++axis)
        p[axis] *= axis == 0 ? transform.scale.x : axis == 1 ? transform.scale.y : transform.scale.z;
    const auto& q = transform.rotation;
    const std::array<double, 3> first{q.y * p[2] - q.z * p[1], q.z * p[0] - q.x * p[2],
                                      q.x * p[1] - q.y * p[0]};
    const std::array<double, 3> second{q.y * first[2] - q.z * first[1], q.z * first[0] - q.x * first[2],
                                       q.x * first[1] - q.y * first[0]};
    for (int axis = 0; axis < 3; ++axis)
        p[axis] += 2 * (second[axis] + q.w * first[axis]);
    p[0] += transform.translation.x;
    p[1] += transform.translation.y;
    p[2] += transform.translation.z;
    return p;
}
inline bool opaqueSceneOracle(const HarmonicMIDI::grid::Visual3DScene& scene,
                              const std::array<double, 3>& origin, const std::array<double, 3>& direction,
                              double limit, bool notes, bool moved)
{
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        const auto& object = scene.objects[o];
        for (std::size_t index = 0; index < object.indexCount; index += 3)
        {
            std::array<std::array<double, 3>, 3> triangle;
            for (int corner = 0; corner < 3; ++corner)
            {
                const auto& vertex =
                    scene.vertices[object.firstVertex + scene.indices[object.firstIndex + index + corner]]
                        .position;
                std::array<double, 3> p{vertex.x, vertex.y, vertex.z};
                if (notes)
                {
                    for (auto& component : p)
                        component *= .5;
                    p[0] += .25;
                }
                p = opaqueOracleTransform(p, object.transform);
                auto parent = object.parent;
                for (std::size_t depth = 0; parent.isValid() && depth < scene.objectCount; ++depth)
                    for (std::size_t candidate = 0; candidate < scene.objectCount; ++candidate)
                        if (scene.objects[candidate].id == parent)
                        {
                            p = opaqueOracleTransform(p, scene.objects[candidate].transform);
                            parent = scene.objects[candidate].parent;
                            break;
                        }
                if (moved && object.id.value == 65537)
                    p[0] += 2.5;
                triangle[corner] = p;
            }
            if (opaqueTriangleOracle(origin, direction, triangle, limit))
                return true;
        }
    }
    return false;
}
inline bool opaqueUnsupportedVertexCase(arbitgpu::NativeFixtureSceneBackend& backend,
                                        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
                                        std::string& error)
{
    surfacematerialbinding::ImportedSceneMaterialRequest request;
    request.scene = scene->id;
    request.sceneSnapshot = scene;
    request.sceneRevision = request.structuralRevision = request.evaluationRevision =
        request.programRevision = 1;
    request.program = surfacematerialstarterfixture::kPrograms[0].program();
    for (auto& operation : request.program.operations)
    {
        if (operation.id == request.program.outputs[0])
            operation.literal = {0, 0, 0, 0};
        if (operation.id == request.program.outputs[2])
            operation.literal[0] = .5f;
        if (operation.id == request.program.outputs[7])
            operation.literal[0] = 1.5f;
        if (operation.id == request.program.outputs[9])
            operation.unsignedLiteral = scene->objects[0].material.value;
    }
    request.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    request.binding.object = scene->objects[0].id;
    request.binding.surfaceMaterialRevision = 1;
    const auto admitted = surfacematerial::admit(request.program, error);
    if (!admitted)
        return false;
    request.binding.surfaceMaterialDigest = admitted->structuralDigest();
    request.vertexModifier = videohelper::test::audioNormalDisplacement();
    const auto binding = videorender::fixture3d::admitSurfaceMaterialBinding(
        scene, request,
        backend.info().backend == "metal" ? videohelper::materialprogram::BackendTarget::Metal
                                          : videohelper::materialprogram::BackendTarget::OpenGl,
        error);
    if (!binding)
        return false;
    const auto prepared = backend.prepare(scene, binding->nativeProgram());
    if (!prepared.prepared)
    {
        error = prepared.error;
        return false;
    }
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    inputs.vertexSpectrum[6] = .5f;
    const auto frame = backend.render(scene, prepared.resources, 65, 65, inputs);
    if (!frame.rendered || !frame.frame || frame.stats.opaqueVisibilityApplied ||
        !frame.stats.opaqueVisibilityUnsupportedVertexDeformation)
    {
        error = "Unsupported vertex shadows must report their boundary while retaining native color: " +
                frame.error;
        return false;
    }
    arbitgpu::NativeRawPassPixels pixels;
    if (!frame.frame->readRawPass(renderpassoutput::Output::Color, pixels, error) ||
        pixels.bytes.size() != 65 * 65 * 8)
        return false;
    std::uint16_t half;
    std::memcpy(&half, pixels.bytes.data() + (32 * 65 + 32) * 8, 2);
    if (!std::isfinite(hdrimage::halfToFloat(half)) || hdrimage::halfToFloat(half) <= 0)
    {
        error = "Unsupported vertex shadow gate blanked the existing color render";
        return false;
    }
    return true;
}
inline bool opaqueVisibilityCases(arbitgpu::NativeFixtureSceneBackend& backend, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    using Direction = std::array<double, 3>;
    const double inverseRoot2 = 1 / std::sqrt(2.0);
    const Direction ray{inverseRoot2, 0, inverseRoot2};
    const std::array<Direction, 3> oracleTriangle{Direction{.75, -.25, 2}, Direction{1.25, -.25, 2},
                                                  Direction{1.25, .25, 2}};
    if (!opaqueTriangleOracle({0, 0, 1}, ray, oracleTriangle, 10) ||
        opaqueTriangleOracle({0, 0, 1}, ray, oracleTriangle, 1) ||
        opaqueTriangleOracle({0, 0, 1}, {0, 0, 1}, oracleTriangle, 10))
    {
        error = "Independent opaque triangle oracle failed authored hit/distance/miss cases";
        return false;
    }
    for (const auto kind : {SceneLightKind::Directional, SceneLightKind::Point, SceneLightKind::Spot})
        for (int state = 0; state < 9; ++state)
        {
            auto sceneValue = fixture3d::makeScene();
            sceneValue.vertexCount = 9;
            sceneValue.indexCount = 24;
            sceneValue.objectCount = state == 0 || state == 5 ? 1 : 3;
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < 3; ++x)
                {
                    auto& vertex = sceneValue.vertices[y * 3 + x];
                    vertex.position = {float(x - 1), float(y - 1), 0};
                    vertex.normal = {0, 0, 1};
                    vertex.color = {1, 1, 1, 1};
                }
            std::size_t at = 0;
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x)
                {
                    const auto a = std::uint32_t(y * 3 + x), b = a + 1, c = a + 4, d = a + 3;
                    for (auto index : {a, b, c, a, c, d})
                        sceneValue.indices[at++] = index;
                }
            sceneValue.objects[0].firstVertex = 0;
            sceneValue.objects[0].vertexCount = 9;
            sceneValue.objects[0].firstIndex = 0;
            sceneValue.objects[0].indexCount = 24;
            sceneValue.objects[0].transform = {};
            sceneValue.objects[0].transform.translation = {0, 0, 1};
            // Blocker and receiver instance the same local indexed geometry. Parent
            // translation and nonuniform scale must reach both raster and rays.
            sceneValue.objects[1] = sceneValue.objects[0];
            sceneValue.objects[1].id = SceneObjectId{65537};
            sceneValue.objects[1].parent = SceneObjectId{65538};
            sceneValue.objects[1].transform = {};
            sceneValue.objects[1].transform.translation = {0, 0,
                                                           state == 3   ? 4.0f
                                                           : state == 4 ? -1.0f
                                                                        : 2.0f};
            sceneValue.objects[1].transform.scale = {.25f, .35f, .5f};
            sceneValue.objects[2] = sceneValue.objects[0];
            sceneValue.objects[2].id = SceneObjectId{65538};
            sceneValue.objects[2].indexCount = 0;
            sceneValue.objects[2].vertexCount = 0;
            sceneValue.objects[2].transform = {};
            sceneValue.objects[2].transform.translation = {state == 2 || state == 7 ? 2.5f
                                                           : state == 3             ? 3.0f
                                                           : state == 4             ? -2.0f
                                                                                    : 1.0f,
                                                           0, 0};
            sceneValue.materials[0].baseColor = {0, 0, 0};
            sceneValue.materials[0].baseColorTexture = {};
            sceneValue.materials[0].metallicRoughnessTexture = {};
            sceneValue.materials[0].normalTexture = {};
            sceneValue.materials[0].occlusionTexture = {};
            sceneValue.materials[0].emissiveTexture = {};
            sceneValue.materials[0].opacity = 1;
            sceneValue.materials[0].alphaMode = SceneAlphaMode::Opaque;
            sceneValue.materials[0].metallic = 0;
            sceneValue.materials[0].roughness = .5f;
            sceneValue.materials[0].emissive = {};
            sceneValue.ambientColor = {};
            sceneValue.lightCount = 1;
            auto& light = sceneValue.lights[0];
            light.kind = kind;
            light.intensity = 1;
            light.color = {1, 1, 1};
            light.range = 0;
            light.transform = {};
            light.transform.translation = {2, 0, 3};
            light.transform.rotation = {0, float(std::sin(basesurfacereflection::kPi / 8)), 0,
                                        float(std::cos(basesurfacereflection::kPi / 8))};
            light.innerConeAngle = .2f;
            light.outerConeAngle = .6f;
            // The same quaternion aims spot -Z toward the receiver and
            // directional +Z toward the light source.
            sceneValue.cameras[0].transform = {};
            sceneValue.cameras[0].transform.translation = {state >= 5 && state <= 7 ? .25f : 0, 0, 5};
            const auto scene = std::make_shared<const Visual3DScene>(std::move(sceneValue));
            if (state == 0 && !opaqueUnsupportedVertexCase(backend, scene, error))
                return false;
            const auto prepared = backend.prepare(scene);
            if (!prepared.prepared)
            {
                error = prepared.error;
                return false;
            }
            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
            inputs.linearColor = true;
            if (state >= 5 && state <= 7)
            {
                auto score = std::make_shared<arbitmod::Score>();
                score->rootFreq = 440;
                score->notationVersion = score->scoreRevision = 1;
                arbitmod::Note note;
                note.id = 1;
                note.midiNote = 69;
                note.velocity = 63.5f;
                note.freqHz = 440;
                note.startBeat = 0;
                note.lengthBeats = 4;
                note.trackId = 0;
                score->notes.push_back(note);
                canonicalblockc::FrameKey key;
                key.projectGeneration = key.sourceGeneration = key.helperGeneration = key.backendGeneration =
                    key.deviceGeneration = 1;
                key.scoreGeneration = key.beatMapGeneration = key.fpsGeneration = key.loopGeneration =
                    key.seekGeneration = 1;
                key.fps = 60;
                canonicalblockc::FrameProducer producer;
                inputs.canonicalBlockCFrame = producer.evaluate(key, score, 0);
                visualnoteinstancing::Mapping mapping;
                mapping.x = visualnoteinstancing::MappingAxis::Velocity;
                mapping.xScale = .5f;
                mapping.yScale = mapping.zScale = 0;
                mapping.meshScale = .5f;
                inputs.noteInstanceMapping = mapping;
                const auto notes = arbitgpu::prepareNativeNoteInstances(inputs);
                if (!canonicalblockc::valid(inputs.canonicalBlockCFrame) || notes.count != 1 ||
                    notes.transforms[0][0] != .25f)
                {
                    error = "Opaque visibility fixture canonical note translation differs from authored "
                            "quarter unit";
                    return false;
                }
            }
            std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> retainedBlocked;
            if (state == 8)
            {
                const auto blockedFrame = backend.render(scene, prepared.resources, 65, 65, inputs);
                if (!blockedFrame.rendered || !blockedFrame.stats.opaqueVisibilityApplied)
                {
                    error = "Opaque visibility runtime move baseline missing: " + blockedFrame.error;
                    return false;
                }
                retainedBlocked = blockedFrame.frame;
                inputs.objectNodeStableId = 2;
                inputs.objectTranslationOffset = {2.5f, 0, 0};
            }
            const auto submitted = backend.render(scene, prepared.resources, 65, 65, inputs);
            arbitgpu::NativeRawPassPixels pixels;
            if (!submitted.rendered || !submitted.stats.opaqueVisibilityApplied ||
                submitted.stats.opaqueVisibilityUnsupportedVertexDeformation || !submitted.frame ||
                !submitted.frame->readRawPass(renderpassoutput::Output::Color, pixels, error) ||
                pixels.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float ||
                pixels.bytes.size() != 65 * 65 * 8)
            {
                error = "Opaque visibility native float frame missing: " + submitted.error + error;
                return false;
            }
            arbitgpu::NativeRawPassPixels normalPixels;
            if (!submitted.frame->readRawPass(renderpassoutput::Output::Normal, normalPixels, error) ||
                normalPixels.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float ||
                normalPixels.bytes.size() != 65 * 65 * 8)
            {
                error = "Opaque visibility receiver normal pass missing: " + error;
                return false;
            }
            for (int channel = 0; channel < 4; ++channel)
            {
                std::uint16_t half;
                std::memcpy(&half, normalPixels.bytes.data() + (32 * 65 + 32) * 8 + channel * 2, 2);
                if (hdrimage::halfToFloat(half) != (channel < 2 ? 0 : 1))
                {
                    error = "Opaque visibility receiver is not the independently authored front normal";
                    return false;
                }
            }
            if (retainedBlocked)
            {
                arbitgpu::NativeRawPassPixels retained;
                if (!retainedBlocked->readRawPass(renderpassoutput::Output::Color, retained, error))
                    return false;
                for (int channel = 0; channel < 3; ++channel)
                {
                    std::uint16_t half;
                    std::memcpy(&half, retained.bytes.data() + (32 * 65 + 32) * 8 + channel * 2, 2);
                    if (hdrimage::halfToFloat(half) != 0)
                    {
                        error = "Opaque visibility retained blocked frame changed after moving its instance";
                        return false;
                    }
                }
            }
            Direction lightRay = ray;
            double distanceSquared = 8;
            if (state >= 5 && state <= 7 && kind != SceneLightKind::Directional)
            {
                distanceSquared = 1.75 * 1.75 + 4;
                lightRay = {1.75 / std::sqrt(distanceSquared), 0, 2 / std::sqrt(distanceSquared)};
            }
            const bool notes = state >= 5 && state <= 7;
            const bool blocked = opaqueSceneOracle(*scene, {notes ? .25 : 0, 0, 1}, lightRay,
                                                   kind == SceneLightKind::Directional
                                                       ? std::numeric_limits<double>::infinity()
                                                       : std::sqrt(distanceSquared),
                                                   notes, state == 8);
            const double expected =
                blocked ? 0
                        : basesurfacereflection::direct({0, 0, 0}, 0, .5, .04, 0, {0, 0, 1}, lightRay,
                                                        {0, 0, 1})[0] *
                              (kind == SceneLightKind::Directional ? 1 : 1.0 / distanceSquared);
            const double predicted = expected;
            const double ulp = expected < std::ldexp(1.0, -14) ? std::ldexp(1.0, -24)
                                                               : std::ldexp(1.0, std::ilogb(expected) - 10);
            const double tolerance = .5 * ulp + 64 * std::numeric_limits<float>::epsilon() * expected;
            for (int channel = 0; channel < 4; ++channel)
            {
                std::uint16_t half;
                std::memcpy(&half, pixels.bytes.data() + (32 * 65 + 32) * 8 + channel * 2, 2);
                const double actual = hdrimage::halfToFloat(half), target = channel == 3 ? 1 : predicted;
                if (!std::isfinite(actual) || std::abs(actual - target) > (channel == 3 ? 0 : tolerance))
                {
                    error = "Opaque visibility indexed instance mismatch: kind " + std::to_string(int(kind)) +
                            ", state " + std::to_string(state) + ", expected " + std::to_string(target) +
                            ", actual " + std::to_string(actual);
                    return false;
                }
            }
        }
    return true;
}
} // namespace videohelper::tests
