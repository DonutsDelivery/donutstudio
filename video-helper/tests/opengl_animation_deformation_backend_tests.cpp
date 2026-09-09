#include "VisualAnimationDeformationEvaluation.h"
#include "../src/gl_loader.h"
#include "../src/gpu_backend/backend.h"
#include "support/fixture_scene.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace visualdeformation;
namespace animation = visualanimation;

bool expect(bool condition, const char* message)
{
    if (! condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

struct Fixture final
{
    HarmonicMIDI::grid::Visual3DScene sceneValue = videohelper::fixture3d::makeScene();
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::array<float, 24 * 3> morphPositions {};
    MorphTargetView morphTarget;
    MeshView mesh;
    AssetView assetView;
    std::shared_ptr<const DeformationAsset> asset;
    std::array<double, 2> keyTimes { 0.0, 1.0 };
    std::array<float, 2> morphValues { 0.0f, 0.75f };
    animation::TrackView track;
    animation::ClipView clipView;
    std::shared_ptr<const animation::Clip> clip;
    std::array<MorphTargetId, 1> targetIds { MorphTargetId { 601 } };
    MorphAnimationBindingView morphBinding;
    AnimationDeformationBindingView bindings;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;

    Fixture()
    {
        for (std::size_t vertex = 0; vertex < 24; ++vertex)
            morphPositions[vertex * 3] = 1.0f;
        morphTarget = { targetIds[0], morphPositions.data(), morphPositions.size(),
                        nullptr, 0, nullptr, 0 };
        mesh = { MeshId { 501 }, 24, {}, nullptr, 0, &morphTarget, 1 };
        assetView = { nullptr, 0, &mesh, 1 };
        std::string error;
        asset = DeformationAsset::create(assetView, {}, error);
        if (asset == nullptr)
            std::cerr << "fixture deformation asset error: " << error << '\n';

        track = { animation::TrackId { 12 }, animation::TargetId { 1002 },
                  animation::Channel::MorphWeights, animation::Interpolation::Linear,
                  keyTimes.data(), morphValues.data(), keyTimes.size(),
                  morphValues.size(), 1 };
        clipView = { animation::ClipId { 701 }, 1.0, &track, 1 };
        clip = animation::Clip::create(clipView, {}, error);
        if (clip == nullptr)
            std::cerr << "fixture animation clip error: " << error << '\n';

        morphBinding = { animation::TargetId { 1002 }, mesh.id,
                         targetIds.data(), targetIds.size() };
        bindings = { nullptr, 0, &morphBinding, 1 };
        scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(sceneValue);
        auto mutableSource = std::make_shared<arbitgpu::NativeDeformationScene>();
        mutableSource->sourceStableId = 801;
        mutableSource->deformationStableId = 802;
        mutableSource->structuralRevision = 77;
        mutableSource->clip = animation::ClipId { 701 };
        mutableSource->mesh = mesh.id;
        mutableSource->object = videohelper::fixture3d::kCubeObjectId;
        mutableSource->scene = scene;
        mutableSource->deformation = asset;
        mutableSource->morphBaseWeights = { 0.0f };
        source = std::move(mutableSource);
    }

    std::shared_ptr<const AnimationDeformationSnapshot> snapshot(std::int64_t frame) const
    {
        std::string error;
        const AnimationDeformationRequest request {
            RationalFrameTime { frame, 30, 1 }, 77,
            animation::Playback::Clamp, std::nullopt
        };
        auto result = evaluateAnimationDeformation(*asset, *clip, bindings,
                                                   request, {}, error);
        if (result == nullptr)
            std::cerr << "fixture snapshot error: " << error << '\n';
        return result;
    }
};

std::vector<std::uint8_t> readRgba8(
    const arbitgl::GlFuncs& gl,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    GLint previousFramebuffer = 0;
    GLint previousPackAlignment = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFramebuffer);
    glGetIntegerv(GL_PACK_ALIGNMENT, &previousPackAlignment);
    GLuint framebuffer = 0;
    gl.GenFramebuffers(1, &framebuffer);
    gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                            static_cast<GLuint>(frame->colorImageHandle()), 0);
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(frame->width()) * frame->height() * 4u);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
    {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, static_cast<GLsizei>(frame->width()),
                     static_cast<GLsizei>(frame->height()),
                     GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }
    glPixelStorei(GL_PACK_ALIGNMENT, previousPackAlignment);
    gl.BindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previousFramebuffer));
    gl.DeleteFramebuffers(1, &framebuffer);
    return pixels;
}

std::vector<float> readDepth32f(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    GLint previousTexture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture);
    glBindTexture(GL_TEXTURE_2D,
                  static_cast<GLuint>(frame->depthTextureViewHandle()));
    std::vector<float> pixels(
        static_cast<std::size_t>(frame->width()) * frame->height());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, pixels.data());
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previousTexture));
    return pixels;
}

std::array<int, 8> shaderStorageBindings(const arbitgl::GlFuncs& gl)
{
    std::array<int, 8> bindings {};
    for (unsigned index = 0; index < bindings.size(); ++index)
        gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, index, &bindings[index]);
    return bindings;
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t value = 14695981039346656037ull;
    for (const auto byte : bytes)
    {
        value ^= byte;
        value *= 1099511628211ull;
    }
    return value;
}
} // namespace

int main()
{
    if (! glfwInit())
    {
        std::cerr << "SKIP: GLFW initialization failed\n";
        return 77;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    auto* window = glfwCreateWindow(96, 96, "opengl-deformation-test", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "SKIP: hidden OpenGL 4.3 core context creation failed\n";
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent(window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions(gl, missing) || ! arbitgl::loadGl43Functions(gl))
    {
        std::cerr << "SKIP: OpenGL 4.3 function loading failed: " << missing << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 77;
    }

    bool ok = true;
    Fixture fixture;
    ok &= expect(fixture.asset != nullptr && fixture.clip != nullptr && fixture.source != nullptr,
                 "the physical deformation fixture must be admitted");
    auto firstSnapshot = fixture.snapshot(0);
    auto movedSnapshot = fixture.snapshot(30);
    ok &= expect(firstSnapshot != nullptr && movedSnapshot != nullptr,
                 "both exact deformation snapshots must evaluate");

    std::array<GLuint, 8> callerBuffers {};
    gl.GenBuffers(static_cast<int>(callerBuffers.size()), callerBuffers.data());
    std::array<std::uint32_t, 4> sentinel { 1, 2, 3, 4 };
    for (unsigned index = 0; index < callerBuffers.size(); ++index)
    {
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, callerBuffers[index]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,
                      static_cast<std::ptrdiff_t>(sizeof(sentinel)),
                      sentinel.data(), GL_STATIC_DRAW);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, index, callerBuffers[index]);
    }
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, callerBuffers[3]);
    const auto callerBindings = shaderStorageBindings(gl);
    GLint callerGenericBinding = 0;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &callerGenericBinding);

    auto& backend = arbitgpu::nativeDeformationBackend();
    const auto info = backend.info();
    ok &= expect(info.available && info.compute && info.backend == "opengl",
                 "the production deformation backend must report live OpenGL compute");
    auto preparation = backend.prepare(fixture.source);
    if (! preparation.prepared)
        std::cerr << "OpenGL deformation preparation error: " << preparation.error << '\n';
    ok &= expect(preparation.prepared && preparation.resources != nullptr,
                 "the exact textured morph scene must prepare on OpenGL");
    ok &= expect(shaderStorageBindings(gl) == callerBindings,
                 "deformation preparation must preserve caller indexed SSBO bindings");
    GLint afterPrepareGeneric = 0;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &afterPrepareGeneric);
    ok &= expect(afterPrepareGeneric == callerGenericBinding,
                 "deformation preparation must preserve caller generic SSBO binding");

    auto first = backend.render(fixture.source, firstSnapshot,
                                preparation.resources, 96, 96);
    auto repeated = backend.render(fixture.source, firstSnapshot,
                                   preparation.resources, 96, 96);
    auto moved = backend.render(fixture.source, movedSnapshot,
                                preparation.resources, 96, 96);
    arbitgpu::NativeDeformationRuntimeInputs mutedMorphInputs;
    mutedMorphInputs.timeSeconds = 0.0f;
    mutedMorphInputs.morphWeight = 0.0f;
    auto mutedMorph = backend.render(fixture.source, movedSnapshot,
                                     preparation.resources, 96, 96, mutedMorphInputs);
    arbitgpu::NativeDeformationRuntimeInputs sceneRuntime;
    sceneRuntime.objectTranslationOffset[0] = 0.5f;
    sceneRuntime.cameraTranslationOffset[2] = 0.25f;
    auto movedScene = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, sceneRuntime);
    arbitgpu::NativeDeformationRuntimeInputs rotatedScene;
    rotatedScene.objectRotationDegrees[1] = 35.0f;
    auto rotated = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, rotatedScene);
    arbitgpu::NativeDeformationRuntimeInputs scaledScene;
    scaledScene.objectScale = 0.7f;
    auto scaled = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, scaledScene);
    arbitgpu::NativeDeformationRuntimeInputs graphCamera;
    graphCamera.cameraOverride = fixture.source->scene->cameras.front();
    graphCamera.cameraOverride->transform.translation.x += 0.75f;
    auto cameraOverride = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, graphCamera);
    auto invalidGraphCamera = graphCamera;
    invalidGraphCamera.cameraOverride->nearPlane = -1.0f;
    auto rejectedCameraOverride = backend.render(
        fixture.source, firstSnapshot, preparation.resources, 96, 96, invalidGraphCamera);
    if (! first.rendered) std::cerr << "first render error: " << first.error << '\n';
    if (! repeated.rendered) std::cerr << "repeated render error: " << repeated.error << '\n';
    if (! moved.rendered) std::cerr << "moved render error: " << moved.error << '\n';
    ok &= expect(first.rendered && repeated.rendered && moved.rendered
                     && mutedMorph.rendered && movedScene.rendered
                     && rotated.rendered && scaled.rendered && cameraOverride.rendered,
                 "native OpenGL deformation must render exact snapshots");
    ok &= expect(!rejectedCameraOverride.rendered
                     && rejectedCameraOverride.error
                          == "native fixture scene runtime inputs are non-finite or out of bounds",
                 "invalid graph camera override must fail closed");
    ok &= expect(first.frame && first.frame->depthImageHandle() != 0
                     && first.frame->depthTextureViewHandle() != 0,
                 "Render 3D must publish its native depth resource");
    ok &= expect(first.stats.dispatchCount == 1 && first.stats.drawCount == 1
                     && first.stats.reusedStaticResources,
                 "each render must report one GPU deformation dispatch and draw");
    ok &= expect(shaderStorageBindings(gl) == callerBindings,
                 "deformation rendering must restore caller indexed SSBO bindings");
    GLint afterRenderGeneric = 0;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &afterRenderGeneric);
    ok &= expect(afterRenderGeneric == callerGenericBinding,
                 "deformation rendering must restore caller generic SSBO binding");

    std::uint64_t firstHash = 0;
    std::uint64_t movedHash = 0;
    if (first.frame && repeated.frame && moved.frame && rotated.frame && scaled.frame
        && cameraOverride.frame)
    {
        const auto firstPixels = readRgba8(gl, first.frame);
        const auto repeatedPixels = readRgba8(gl, repeated.frame);
        const auto movedPixels = readRgba8(gl, moved.frame);
        const auto mutedMorphPixels = readRgba8(gl, mutedMorph.frame);
        const auto movedScenePixels = readRgba8(gl, movedScene.frame);
        const auto rotatedPixels = readRgba8(gl, rotated.frame);
        const auto scaledPixels = readRgba8(gl, scaled.frame);
        const auto cameraOverridePixels = readRgba8(gl, cameraOverride.frame);
        firstHash = fnv1a64(firstPixels);
        movedHash = fnv1a64(movedPixels);
        ok &= expect(firstPixels == repeatedPixels,
                     "equal immutable snapshots must produce identical pixels");
        ok &= expect(firstPixels != movedPixels,
                     "a changed morph weight must change rendered pixels");
        ok &= expect(firstPixels == mutedMorphPixels,
                     "zero runtime morph weight must restore the imported base shape");
        ok &= expect(firstPixels != movedScenePixels,
                     "object and camera runtime offsets must change rendered pixels");
        ok &= expect(firstPixels != rotatedPixels,
                     "runtime object rotation must change rendered pixels");
        ok &= expect(firstPixels != scaledPixels,
                     "runtime object scale must change rendered pixels");
        ok &= expect(firstPixels != cameraOverridePixels,
                     "graph camera override must change rendered pixels");
        const auto firstDepth = readDepth32f(first.frame);
        const auto repeatedDepth = readDepth32f(repeated.frame);
        const auto movedDepth = readDepth32f(moved.frame);
        ok &= expect(firstDepth == repeatedDepth,
                     "equal immutable snapshots must produce identical depth");
        ok &= expect(firstDepth != movedDepth,
                     "a changed morph weight must change published depth");
    }

    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object = videohelper::fixture3d::kCubeObjectId;
    material->bindingDigest = "opengl-deformed-time-material-binding";
    material->programIdentity = "opengl-deformed-time-material-program";
    material->baseColorSource
        = arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix;
    material->parameters.baseColorMetallic = { 1.0f, 0.0f, 0.0f, 0.0f };
    material->parameters.emissionRoughness = { 0.1f, 0.0f, 0.0f, 0.5f };
    material->parameters.normalOpacity = { 0.0f, 0.0f, 1.0f, 1.0f };
    material->parameters.transmissionIorClearcoat = { 0.0f, 1.5f, 0.0f, 0.0f };
    material->parameters.identifiers[0] = videohelper::fixture3d::kCubeMaterialId.value;
    material->timeMixEndColor = { 0.0f, 0.0f, 1.0f };
    auto materialPreparation = backend.prepare(fixture.source, material);
    if (! materialPreparation.prepared)
        std::cerr << "OpenGL deformed material preparation error: "
                  << materialPreparation.error << '\n';
    ok &= expect(materialPreparation.prepared && materialPreparation.resources != nullptr,
                 "authored Surface Material must prepare with deformation resources");
    arbitgpu::NativeDeformationRuntimeInputs materialStartInputs;
    materialStartInputs.timeSeconds = 0.0f;
    auto materialStart = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96,
        materialStartInputs);
    arbitgpu::NativeDeformationRuntimeInputs materialEndInputs;
    materialEndInputs.timeSeconds = 1.0f;
    auto materialEnd = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96,
        materialEndInputs);
    arbitgpu::NativeDeformationRuntimeInputs darkEmission;
    darkEmission.emissionGain = 0.0f;
    auto materialDark = backend.render(
        fixture.source, firstSnapshot, materialPreparation.resources, 96, 96, darkEmission);
    if (! materialStart.rendered)
        std::cerr << "deformed material start error: " << materialStart.error << '\n';
    if (! materialEnd.rendered)
        std::cerr << "deformed material end error: " << materialEnd.error << '\n';
    ok &= expect(materialStart.rendered && materialEnd.rendered && materialDark.rendered,
                 "deformed geometry must render the authored material at exact times");
    if (materialStart.frame && materialEnd.frame)
    {
        const auto startPixels = readRgba8(gl, materialStart.frame);
        const auto endPixels = readRgba8(gl, materialEnd.frame);
        ok &= expect(startPixels != endPixels,
                     "the authored time-dependent material must change deformed pixels");
        ok &= expect(startPixels != readRgba8(gl, materialDark.frame),
                     "runtime emission gain must change deformed material pixels");
    }

    auto wrongOwner = std::make_shared<arbitgpu::NativeDeformationScene>(*fixture.source);
    wrongOwner->deformationStableId += 1;
    const auto rejected = backend.render(wrongOwner, firstSnapshot,
                                         preparation.resources, 96, 96);
    ok &= expect(! rejected.rendered && ! rejected.error.empty(),
                 "prepared resources must reject a different immutable owner");

    first.frame.reset();
    repeated.frame.reset();
    moved.frame.reset();
    mutedMorph.frame.reset();
    movedScene.frame.reset();
    rotated.frame.reset();
    scaled.frame.reset();
    materialStart.frame.reset();
    materialEnd.frame.reset();
    materialDark.frame.reset();
    materialPreparation.resources.reset();
    preparation.resources.reset();
    gl.DeleteBuffers(static_cast<int>(callerBuffers.size()), callerBuffers.data());
    if (ok)
    {
        std::cout << "OpenGL animation deformation backend: PASS; device="
                  << info.device << "; base-rgba8-fnv1a64=" << std::hex << firstHash
                  << "; morphed-rgba8-fnv1a64=" << movedHash << std::dec << '\n';
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return ok ? 0 : 1;
}
