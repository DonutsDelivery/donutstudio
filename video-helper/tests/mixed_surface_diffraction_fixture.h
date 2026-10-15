#pragma once

#include "reactive_frame_fixture.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../src/gpu_backend/fixture_opaque_visibility.h"

// Two separate authored patches, with the grating deliberately at array index 1
// and with its own vertex/index ranges. No generated image or composite is used.
inline std::optional<visualimportedscenerender::Request> mixedSurfaceDiffractionFixture(std::string& error)
{
    using namespace HarmonicMIDI::grid;
    auto scene = std::make_shared<Visual3DScene>();
    scene->id = {701}; scene->activeCamera = {501};
    scene->cameraCount = scene->lightCount = 1;
    scene->cameras[0].id = scene->activeCamera;
    scene->cameras[0].transform.translation.z = 4;
    scene->lights[0].id = {601}; scene->lights[0].kind = SceneLightKind::Directional;
    scene->lights[0].intensity = 1;
    scene->lights[0].color = {1, 1, 1};
    scene->ambientColor = {0, 0, 0};
    scene->objectCount = scene->materialCount = 2;
    scene->vertexCount = 8; scene->indexCount = 12;
    const std::array<SceneVertex, 4> vertices {{
        {{-0.55f,-0.55f,0}, {0,0,1}, {1,0,0,1}, {0,0}},
        {{ 0.55f,-0.55f,0}, {0,0,1}, {1,0,0,1}, {1,0}},
        {{ 0.55f, 0.55f,0}, {0,0,1}, {1,0,0,1}, {1,1}},
        {{-0.55f, 0.55f,0}, {0,0,1}, {1,0,0,1}, {0,1}}
    }};
    const std::array<std::uint32_t, 6> indices {0,1,2,0,2,3};
    for (std::size_t i = 0; i < 2; ++i)
    {
        std::copy(vertices.begin(), vertices.end(), scene->vertices.begin() + i * 4);
        std::copy(indices.begin(), indices.end(), scene->indices.begin() + i * 6);
        auto& object = scene->objects[i];
        object.id = {i == 0 ? 197u : 991u};
        object.material = {i == 0 ? 123u : 456u};
        object.firstVertex = static_cast<std::uint32_t>(i * 4); object.vertexCount = 4;
        object.firstIndex = static_cast<std::uint32_t>(i * 6); object.indexCount = 6;
        object.transform.translation.x = i == 0 ? -0.7f : 0.7f;
        scene->materials[i].id = object.material;
        scene->materials[i].doubleSided = true;
    }
    visualimportedscenerender::Request request;
    request.sceneSnapshot = scene; request.sourceStableId = 701; request.renderStableId = 702;
    request.structuralRevision = 3; request.evaluationRevision = 7;
    geometrysurfacematerial::ObjectProgram surface;
    surface.program.sourceMaterial = 801;
    surface.program.material = tableSurfaceRequest(123, false);
    auto& material = surface.program.material;
    material.scene = scene->id; material.sceneSnapshot = scene;
    material.sceneRevision = material.structuralRevision = material.programRevision = 3;
    material.evaluationRevision = 7; material.binding.surfaceMaterialRevision = 3;
    material.binding.object = {197};
    for (auto& operation : material.program.operations)
    {
        if (operation.id == material.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::Transmission)]
            || operation.id == material.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::Clearcoat)])
            operation.literal = {};
        if (operation.id == material.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::Emission)])
            operation.literal = {0.4f, 0.02f, 0.01f, 0};
    }
    material.version = surfacematerialbinding::kReflectanceWireVersion;
    material.program.spectralReflectance = surfacematerial::SpectralReflectanceDescription {};
    material.program.spectralReflectance->coefficients = {0.03f, 0.11f, 0.19f, 0.31f, 0.43f, 0.59f, 0.71f, 0.89f};
    const auto admitted = surfacematerial::admit(material.program, error);
    if (!admitted) return {};
    material.binding.surfaceMaterialDigest = admitted->structuralDigest();
    request.surfacePrograms.push_back(std::move(surface));
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest grating;
    grating.scene = scene->id; grating.object = {991};
    grating.sceneRevision = grating.structuralRevision = grating.materialRevision = 3;
    grating.evaluationRevision = 7;
    grating.material = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    grating.lighting = diffractionmaterial::makeReferenceLighting();
    const auto physical = diffractionmaterial::admit(grating.material, error);
    if (!physical) return {};
    grating.structuralDigest = physical->structuralDigest();
    request.diffractionMaterial = std::move(grating);
    if (!visualimportedscenerender::valid(request)) { error = "mixed two-patch fixture is invalid"; return {}; }
    return request;
}

inline void mixedSurfaceFrame(visualimportedscenerender::Request& request, int node = 91)
{
    auto& object = request.surfacePrograms.front();
    const auto previous = object.program.material;
    object.program.material = reactiveFrameRequest(node);
    auto& material = object.program.material;
    material.scene = previous.scene; material.sceneSnapshot = previous.sceneSnapshot;
    material.binding.object = previous.binding.object;
    material.sceneRevision = material.structuralRevision = material.programRevision = previous.programRevision;
    material.binding.surfaceMaterialRevision = previous.programRevision;
    material.evaluationRevision = previous.evaluationRevision;
    material.program.spectralReflectance = previous.program.spectralReflectance;
    material.version = surfacematerialbinding::requestVersion(true, material.program.spectralReflectance.has_value());
    for (auto& operation : material.program.operations)
        if (operation.id == material.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)])
            operation.unsignedLiteral = 123;
    std::string error;
    const auto admitted = surfacematerial::admit(material.program, error);
    material.binding.surfaceMaterialDigest = admitted ? admitted->structuralDigest() : std::string{};
}

template <class ReadPixels>
bool verifyNativeMixedSurfaceDiffraction(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error)
{
    using namespace videorender::fixture3d;
    const auto original = mixedSurfaceDiffractionFixture(error);
    if (!original) return false;
    visualimportedscenerender::Request request;
    if (!visualimportedscenerender::decode(visualimportedscenerender::encode(*original), request))
    { error = "mixed material native fixture did not reopen"; return false; }
    const auto target = backend.info().backend == "metal" ? videohelper::materialprogram::BackendTarget::Metal
        : videohelper::materialprogram::BackendTarget::OpenGl;
    const auto draw = [&](const visualimportedscenerender::Request& value, RenderUse use, RenderedFrame& result,
                          FixtureSceneRenderer& renderer) {
        const auto surfaces = admitSurfaceMaterialCollection(value.sceneSnapshot, value.surfacePrograms, {}, target, error);
        const auto grating = admitDiffractionMaterialBinding(value.sceneSnapshot, *value.diffractionMaterial, target, error, true);
        return surfaces && grating && renderer.renderMixed(value.sceneSnapshot, surfaces, grating, {}, {128,128},
            use, kNativeGpuCapability, result, error);
    };
    const auto identities = [&](const RenderedFrame& frame, std::uint32_t a, std::uint32_t b) {
        arbitgpu::NativeRawPassPixels pixels;
        if (!frame.nativeFrame->readRawPass(renderpassoutput::Output::ObjectId, pixels, error)
            || pixels.format != arbitgpu::NativeTexturePixelFormat::R32Uint || pixels.bytes.size() != 128u * 128u * 4u)
            return std::array<std::size_t, 2> {};
        std::array<std::size_t, 2> counts {};
        for (std::size_t offset = 0; offset < pixels.bytes.size(); offset += 4) {
            std::uint32_t id = 0; std::memcpy(&id, pixels.bytes.data() + offset, 4);
            counts[0] += id == a; counts[1] += id == b;
        }
        return counts;
    };
    FixtureSceneRenderer renderer(backend);
    RenderedFrame preview, exported;
    if (!draw(request, RenderUse::Preview, preview, renderer) || !draw(request, RenderUse::Export, exported, renderer)) return false;
    const auto ids = identities(preview, 197, 991);
    if (ids[0] < 40 || ids[1] < 40 || preview.stats.ordinaryDrawCount != 4
        || readPixels(preview.nativeFrame).empty() || readPixels(preview.nativeFrame) != readPixels(exported.nativeFrame))
    { error = "one native mixed scene lost either exact draw, linear accumulation or preview/export parity"; return false; }
    // Changing only the Surface emission must change that exact object while
    // retaining every grating pixel from the same two-object scene.
    auto edited = request;
    auto& editedSurface = edited.surfacePrograms.front().program.material;
    for (auto& operation : editedSurface.program.operations)
        if (operation.id == editedSurface.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::Emission)])
            operation.literal = {0.01f, 0.02f, 0.4f, 0};
    const auto editedAdmission = surfacematerial::admit(editedSurface.program, error);
    if (!editedAdmission) return false;
    editedSurface.binding.surfaceMaterialDigest = editedAdmission->structuralDigest();
    edited.structuralRevision = editedSurface.sceneRevision = editedSurface.structuralRevision
        = editedSurface.programRevision = editedSurface.binding.surfaceMaterialRevision = 4;
    edited.evaluationRevision = editedSurface.evaluationRevision = 8;
    edited.diffractionMaterial->sceneRevision = edited.diffractionMaterial->structuralRevision
        = edited.diffractionMaterial->materialRevision = 4;
    edited.diffractionMaterial->evaluationRevision = 8;
    FixtureSceneRenderer editedRenderer(backend);
    RenderedFrame editedFrame;
    if (!draw(edited, RenderUse::Preview, editedFrame, editedRenderer)) return false;
    const auto before = readPixels(preview.nativeFrame), after = readPixels(editedFrame.nativeFrame);
    arbitgpu::NativeRawPassPixels objectIds;
    if (before.size() != 128u * 128u * 4u || after.size() != before.size()
        || !preview.nativeFrame->readRawPass(renderpassoutput::Output::ObjectId, objectIds, error)
        || objectIds.bytes.size() != before.size()) return false;
    bool changedSurface = false;
    for (std::size_t offset = 0; offset < before.size(); offset += 4) {
        std::uint32_t id = 0; std::memcpy(&id, objectIds.bytes.data() + offset, 4);
        const bool changed = !std::equal(before.begin() + offset, before.begin() + offset + 4, after.begin() + offset);
        if (id == 197) changedSurface |= changed;
        if (id == 991 && changed) { error = "Surface program edit leaked into the exact grating draw"; return false; }
    }
    if (!changedSurface) { error = "mixed native rendering did not dispatch the authored Surface emission edit"; return false; }
    // An independent geometric expectation: coincident quads at different z
    // are completely covered by the nearer quad, regardless of material kind.
    for (bool gratingInFront : {false, true}) {
        auto overlap = request;
        auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*request.sceneSnapshot);
        scene->objects[0].transform.translation = {0,0,gratingInFront ? 0.0f : 0.2f};
        scene->objects[1].transform.translation = {0,0,gratingInFront ? 0.2f : 0.0f};
        overlap.sceneSnapshot = scene;
        overlap.surfacePrograms.front().program.material.sceneSnapshot = scene;
        FixtureSceneRenderer overlapRenderer(backend);
        RenderedFrame frame;
        if (!draw(overlap, RenderUse::Preview, frame, overlapRenderer)) return false;
        const auto count = identities(frame, 197, 991);
        if ((gratingInFront ? count[0] : count[1]) != 0 || (gratingInFront ? count[1] : count[0]) < 40)
        { error = "mixed Surface/grating draws do not share exact front-to-back depth visibility"; return false; }
    }
    error.clear(); return true;
}
