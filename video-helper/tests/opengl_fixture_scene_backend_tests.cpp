#include "../src/gpu_backend/backend.h"
#include "../src/diffraction_material_admission.h"
#include "../src/diffraction_material_execution.h"
#include "../src/fixture_scene_renderer.h"
#include "../src/diffractive_foil_admission.h"
#include "../src/gl_loader.h"
#include "support/fixture_scene.h"
#include "support/surface_material_starter_oracle.h"
#include "../src/sha256.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

bool expect (bool condition, const char* message)
{
    if (! condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

struct CallerState final
{
    GLint framebuffer = 0;
    GLint activeTexture = 0;
    GLint texture = 0;
    GLint texture0 = 0;
    GLint sampler0 = 0;
    GLint pixelUnpackBuffer = 0;
    GLint unpackAlignment = 0;
    GLint unpackRowLength = 0;
    GLint unpackSkipPixels = 0;
    GLint unpackSkipRows = 0;
    std::array<GLint, 4> viewport {};
    std::array<GLboolean, 4> colorMask {};
    std::array<GLfloat, 4> clearColor {};
    GLdouble clearDepth = 0.0;
    GLboolean blend = GL_FALSE;
    GLboolean scissor = GL_FALSE;
};

CallerState captureCallerState (const arbitgl::GlFuncs& gl)
{
    CallerState state;
    glGetIntegerv (GL_FRAMEBUFFER_BINDING, &state.framebuffer);
    glGetIntegerv (GL_ACTIVE_TEXTURE, &state.activeTexture);
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &state.texture);
    gl.ActiveTexture (GL_TEXTURE0);
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &state.texture0);
    glGetIntegerv (GL_SAMPLER_BINDING, &state.sampler0);
    gl.ActiveTexture (static_cast<GLenum> (state.activeTexture));
    glGetIntegerv (GL_PIXEL_UNPACK_BUFFER_BINDING, &state.pixelUnpackBuffer);
    glGetIntegerv (GL_UNPACK_ALIGNMENT, &state.unpackAlignment);
    glGetIntegerv (GL_UNPACK_ROW_LENGTH, &state.unpackRowLength);
    glGetIntegerv (GL_UNPACK_SKIP_PIXELS, &state.unpackSkipPixels);
    glGetIntegerv (GL_UNPACK_SKIP_ROWS, &state.unpackSkipRows);
    glGetIntegerv (GL_VIEWPORT, state.viewport.data());
    glGetBooleanv (GL_COLOR_WRITEMASK, state.colorMask.data());
    glGetFloatv (GL_COLOR_CLEAR_VALUE, state.clearColor.data());
    glGetDoublev (GL_DEPTH_CLEAR_VALUE, &state.clearDepth);
    state.blend = glIsEnabled (GL_BLEND);
    state.scissor = glIsEnabled (GL_SCISSOR_TEST);
    return state;
}

bool sameCallerState (const arbitgl::GlFuncs& gl, const CallerState& expected)
{
    const auto actual = captureCallerState (gl);
    return actual.framebuffer == expected.framebuffer
        && actual.activeTexture == expected.activeTexture
        && actual.texture == expected.texture
        && actual.texture0 == expected.texture0
        && actual.sampler0 == expected.sampler0
        && actual.pixelUnpackBuffer == expected.pixelUnpackBuffer
        && actual.unpackAlignment == expected.unpackAlignment
        && actual.unpackRowLength == expected.unpackRowLength
        && actual.unpackSkipPixels == expected.unpackSkipPixels
        && actual.unpackSkipRows == expected.unpackSkipRows
        && actual.viewport == expected.viewport
        && actual.colorMask == expected.colorMask
        && actual.clearColor == expected.clearColor
        && actual.clearDepth == expected.clearDepth
        && actual.blend == expected.blend
        && actual.scissor == expected.scissor;
}

std::vector<std::uint8_t> readRgba8 (
    const arbitgl::GlFuncs& gl,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    GLint previousFramebuffer = 0;
    GLint previousPackAlignment = 0;
    glGetIntegerv (GL_FRAMEBUFFER_BINDING, &previousFramebuffer);
    glGetIntegerv (GL_PACK_ALIGNMENT, &previousPackAlignment);

    GLuint framebuffer = 0;
    gl.GenFramebuffers (1, &framebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D (
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        static_cast<GLuint> (frame->colorImageHandle()), 0);

    std::vector<std::uint8_t> pixels (
        static_cast<std::size_t> (frame->width()) * frame->height() * 4u);
    if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
    {
        glPixelStorei (GL_PACK_ALIGNMENT, 1);
        glReadPixels (0, 0, static_cast<GLsizei> (frame->width()),
                      static_cast<GLsizei> (frame->height()),
                      GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    glPixelStorei (GL_PACK_ALIGNMENT, previousPackAlignment);
    gl.BindFramebuffer (GL_FRAMEBUFFER, static_cast<GLuint> (previousFramebuffer));
    gl.DeleteFramebuffers (1, &framebuffer);
    return pixels;
}

std::uint64_t fnv1a64 (const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t value = 14695981039346656037ull;
    for (const auto byte : bytes)
    {
        value ^= byte;
        value *= 1099511628211ull;
    }
    return value;
}

void reportCoverage (const char* name, const std::vector<std::uint8_t>& pixels)
{
    std::size_t background = 0;
    std::size_t nonBackground = 0;
    std::array<std::uint8_t, 4> firstDrawn {};
    bool captured = false;
    for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
    {
        const bool clear = pixels[index] == 7u && pixels[index + 1] == 10u
            && pixels[index + 2] == 18u && pixels[index + 3] == 255u;
        clear ? ++background : ++nonBackground;
        if (! clear && ! captured)
        {
            std::copy_n (pixels.begin() + static_cast<std::ptrdiff_t> (index), 4,
                         firstDrawn.begin());
            captured = true;
        }
    }
    std::cerr << "coverage " << name << " background=" << background
              << " nonBackground=" << nonBackground
              << " first=" << static_cast<unsigned> (firstDrawn[0]) << ','
              << static_cast<unsigned> (firstDrawn[1]) << ','
              << static_cast<unsigned> (firstDrawn[2]) << ','
              << static_cast<unsigned> (firstDrawn[3])
              << " hash=" << std::hex << fnv1a64 (pixels) << std::dec << '\n';
}

} // namespace

int main()
{
    if (! glfwInit())
    {
        std::cerr << "SKIP: GLFW initialization failed\n";
        return 77;
    }

    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    auto* window = glfwCreateWindow (64, 64, "fixture-scene-backend-test", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "SKIP: hidden OpenGL 3.3 core context creation failed\n";
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::cerr << "FAIL: OpenGL function loading failed: " << missing << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    bool ok = true;
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    const auto info = backend.info();
    ok &= expect (info.available && info.backend == "opengl",
                  "the production fixture backend must report the live OpenGL context");

    auto sceneValue = videohelper::fixture3d::makeScene();
    sceneValue.lightCount = 0;
    sceneValue.ambientColor = { 0.0f, 0.0f, 0.0f };
#if defined(ARBIT_STRICT_PACKAGED_ACCEPTANCE)
    sceneValue.objectCount = 2;
    sceneValue.objects[1] = sceneValue.objects[0];
    sceneValue.objects[1].id = HarmonicMIDI::grid::SceneObjectId { 2 };
    sceneValue.objects[1].transform.translation.x += 1.75f;
#endif
    auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (sceneValue);

    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object = videohelper::fixture3d::kCubeObjectId;
    material->bindingDigest = "opengl-fixture-emission-binding";
    material->programIdentity = "opengl-fixture-emission-program";
    material->baseColorSource =
        arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear;
    material->parameters.baseColorMetallic = { 0.0f, 0.0f, 0.0f, 0.0f };
    material->parameters.emissionRoughness = { 0.8f, 0.08f, 0.02f, 0.5f };
    material->parameters.normalOpacity = { 0.0f, 0.0f, 1.0f, 1.0f };
    material->parameters.transmissionIorClearcoat = { 0.0f, 1.5f, 0.0f, 0.0f };
    material->parameters.identifiers = { videohelper::fixture3d::kCubeMaterialId.value, 0u, 0u, 0u };

    GLuint callerPixelUnpackBuffer = 0;
    std::array<std::uint8_t, 4096> callerPixelUnpackBytes {};
    gl.GenBuffers (1, &callerPixelUnpackBuffer);
    gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, callerPixelUnpackBuffer);
    gl.BufferData (GL_PIXEL_UNPACK_BUFFER,
                   static_cast<std::ptrdiff_t> (callerPixelUnpackBytes.size()),
                   callerPixelUnpackBytes.data(), GL_STATIC_DRAW);
    glPixelStorei (GL_UNPACK_ALIGNMENT, 8);
    glPixelStorei (GL_UNPACK_ROW_LENGTH, 7);
    glPixelStorei (GL_UNPACK_SKIP_PIXELS, 2);
    glPixelStorei (GL_UNPACK_SKIP_ROWS, 3);
    const auto dirtyUploadState = captureCallerState (gl);

    auto preparation = backend.prepare (scene, material);
    if (! preparation.prepared)
        std::cerr << "OpenGL fixture preparation error: " << preparation.error << '\n';
    ok &= expect (preparation.prepared && preparation.resources != nullptr,
                  "the exact bounded no-light emission scene must prepare on OpenGL");
    ok &= expect (sameCallerState (gl, dirtyUploadState),
                  "fixture preparation must isolate and restore caller pixel-unpack state");
    ok &= expect (preparation.stats.staticUploadCount == 1
                      && preparation.stats.materialProgramUploadCount == 1
                      && preparation.stats.vertexBytes == sceneValue.vertexCount
                                                         * sizeof (HarmonicMIDI::grid::SceneVertex)
                      && preparation.stats.indexBytes == sceneValue.indexCount
                                                        * sizeof (std::uint32_t),
                  "preparation must report exact bounded resource uploads");
    if (! preparation.prepared || preparation.resources == nullptr)
    {
        gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    GLuint callerTexture = 0;
    GLuint callerSampler = 0;
    GLuint callerFramebuffer = 0;
    glGenTextures (1, &callerTexture);
    gl.GenSamplers (1, &callerSampler);
    gl.SamplerParameteri (callerSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.SamplerParameteri (callerSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.ActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, callerTexture);
    gl.BindSampler (0, callerSampler);
    gl.ActiveTexture (GL_TEXTURE1);
    glBindTexture (GL_TEXTURE_2D, callerTexture);
    gl.GenFramebuffers (1, &callerFramebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, callerFramebuffer);
    glViewport (3, 5, 17, 19);
    glEnable (GL_BLEND);
    glEnable (GL_SCISSOR_TEST);
    glScissor (1, 2, 3, 4);
    glColorMask (GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
    glClearColor (0.6f, 0.4f, 0.2f, 0.8f);
    glClearDepth (0.25);
    while (glGetError() != GL_NO_ERROR) {}
    const auto dirtyState = captureCallerState (gl);

    arbitgpu::NativeFixtureSceneRuntimeInputs noteRuntime;
    auto first = backend.render (scene, preparation.resources, 64, 64, noteRuntime);
    if (! first.rendered)
        std::cerr << "OpenGL fixture render error: " << first.error << '\n';
    ok &= expect (first.rendered && first.frame != nullptr,
                  "the exact bounded scene must produce a native OpenGL frame");
    ok &= expect (first.stats.drawCount == sceneValue.objectCount
                      && first.stats.reusedStaticResources
                      && first.stats.reusedMaterialProgram,
                  "the draw must report every object submission and prepared-resource reuse");
    ok &= expect (sameCallerState (gl, dirtyState),
                  "fixture rendering must restore caller OpenGL state exactly");
    if (! first.rendered || first.frame == nullptr)
    {
        preparation.resources.reset();
        gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
        gl.DeleteFramebuffers (1, &callerFramebuffer);
        glDeleteTextures (1, &callerTexture);
        gl.DeleteSamplers (1, &callerSampler);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    glDisable (GL_SCISSOR_TEST);
    glDisable (GL_BLEND);
    glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.ActiveTexture (GL_TEXTURE0);
    const auto firstPixels = readRgba8 (gl, first.frame);
    reportCoverage ("constant-emission", firstPixels);

    auto second = backend.render (scene, preparation.resources, 64, 64, noteRuntime);
    if (! second.rendered)
        std::cerr << "OpenGL second fixture render error: " << second.error << '\n';
    ok &= expect (second.rendered && second.frame != nullptr,
                  "a second owner-local submission must produce a frame");

    arbitgpu::NativeFixtureSceneRuntimeInputs explicitLight;
    HarmonicMIDI::grid::SceneLightRecord lightOverride;
    lightOverride.id.value = 9001;
    lightOverride.kind = HarmonicMIDI::grid::SceneLightKind::Environment;
    lightOverride.color = { 0.2f, 0.4f, 0.6f };
    lightOverride.intensity = 2.0f;
    explicitLight.lightOverride = lightOverride;
    auto lightBound = backend.render (
        scene, preparation.resources, 64, 64, explicitLight);
    ok &= expect (lightBound.rendered && lightBound.frame != nullptr
                      && lightBound.frame->backend() == "opengl",
                  "an explicit immutable Light binding executes on OpenGL");

    explicitLight.lightOverride->id = {};
    const auto invalidLight = backend.render (
        scene, preparation.resources, 64, 64, explicitLight);
    ok &= expect (! invalidLight.rendered && invalidLight.frame == nullptr,
                  "OpenGL rejects a malformed Light binding before drawing");
    if (! second.rendered || second.frame == nullptr)
    {
        first.frame.reset();
        preparation.resources.reset();
        gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
        gl.DeleteFramebuffers (1, &callerFramebuffer);
        glDeleteTextures (1, &callerTexture);
        gl.DeleteSamplers (1, &callerSampler);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }
    const auto secondPixels = readRgba8 (gl, second.frame);
    ok &= expect (first.frame->colorImageHandle() != second.frame->colorImageHandle(),
                  "simultaneously live submissions must own distinct fresh color textures");
    ok &= expect (firstPixels == secondPixels,
                  "instanced preview/export-equivalent submissions must match exactly");

    const auto center = (32u * 64u + 32u) * 4u;
    std::cerr << "constant-emission center="
              << static_cast<unsigned> (firstPixels[center]) << ','
              << static_cast<unsigned> (firstPixels[center + 1u]) << ','
              << static_cast<unsigned> (firstPixels[center + 2u]) << ','
              << static_cast<unsigned> (firstPixels[center + 3u]) << '\n';
    ok &= expect (firstPixels.size() > center + 3u
                      && firstPixels[center] > firstPixels[center + 1u] * 4u
                      && firstPixels[center] > 100u
                      && firstPixels[center + 3u] == 255u,
                  "the no-light scene must contain the native constant-emission pixels");

    const auto foregroundPixels = [] (const std::vector<std::uint8_t>& pixels)
    {
        const std::array<std::uint8_t, 4> background { 7, 10, 18, 255 };
        std::size_t count = 0;
        for (std::size_t offset = 0; offset + 3 < pixels.size(); offset += 4)
            if (!std::equal (pixels.begin() + static_cast<std::ptrdiff_t> (offset),
                             pixels.begin() + static_cast<std::ptrdiff_t> (offset + 4),
                             background.begin()))
                ++count;
        return count;
    };
    auto frontTriangleValue = sceneValue;
    frontTriangleValue.objectCount = 1;
    frontTriangleValue.objects[0].vertexCount = 3;
    frontTriangleValue.objects[0].indexCount = 3;
    frontTriangleValue.vertexCount = 3;
    frontTriangleValue.indexCount = 3;
    const auto frontTriangle = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        frontTriangleValue);
    const auto frontPreparation = backend.prepare (frontTriangle, material);
    const auto frontFrame = backend.render (
        frontTriangle, frontPreparation.resources, 64, 64, {});
    const auto frontPixels = frontFrame.rendered
        ? readRgba8 (gl, frontFrame.frame) : std::vector<std::uint8_t> {};
    auto backTriangleValue = frontTriangleValue;
    std::swap (backTriangleValue.indices[1], backTriangleValue.indices[2]);
    const auto backTriangle = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        backTriangleValue);
    const auto backPreparation = backend.prepare (backTriangle, material);
    const auto backFrame = backend.render (
        backTriangle, backPreparation.resources, 64, 64, {});
    const auto backPixels = backFrame.rendered
        ? readRgba8 (gl, backFrame.frame) : std::vector<std::uint8_t> {};
    auto mirroredTriangleValue = frontTriangleValue;
    mirroredTriangleValue.objects[0].transform.scale.x = -1.0f;
    const auto mirroredTriangle
        = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (mirroredTriangleValue);
    const auto mirroredPreparation = backend.prepare (mirroredTriangle, material);
    const auto mirroredFrame = backend.render (
        mirroredTriangle, mirroredPreparation.resources, 64, 64, {});
    const auto mirroredPixels = mirroredFrame.rendered
        ? readRgba8 (gl, mirroredFrame.frame) : std::vector<std::uint8_t> {};
    ok &= expect (frontFrame.rendered && foregroundPixels (frontPixels) > 0
                      && backFrame.rendered && foregroundPixels (backPixels) == 0
                      && mirroredFrame.rendered && foregroundPixels (mirroredPixels) > 0,
                  "single-sided OpenGL rendering rejects backfaces and preserves mirrored frontfaces");

    const auto oracle = surfacematerialstarteroracle::load (
        SURFACE_MATERIAL_STARTER_ORACLE_PATH);
    auto starterLightingSceneValue = sceneValue;
    starterLightingSceneValue.lightCount = 1;
    starterLightingSceneValue.objects[0].transform.rotation
        = { -0.16773126f, 0.25488700f, 0.04494346f, 0.95125124f };
    starterLightingSceneValue.ambientColor = { 0.12f, 0.12f, 0.12f };
    starterLightingSceneValue.lights[0].kind
        = HarmonicMIDI::grid::SceneLightKind::Directional;
    starterLightingSceneValue.lights[0].transform = {};
    starterLightingSceneValue.lights[0].transform.rotation
        = starterLightingSceneValue.objects[0].transform.rotation;
    starterLightingSceneValue.lights[0].color = { 1.0f, 1.0f, 1.0f };
    starterLightingSceneValue.lights[0].intensity = 0.88f;
    starterLightingSceneValue.materials[0].baseColorTexture = {};
    std::vector<std::uint8_t> sampledStarterPixels;
    for (std::size_t starterIndex = 0;
         starterIndex < surfacematerialstarterfixture::kPrograms.size(); ++starterIndex)
    {
        const auto& starter = surfacematerialstarterfixture::kPrograms[starterIndex];
        const auto& expected = oracle.materials[starterIndex];
        ok &= expect (expected.id == starter.id,
                      "generated starter programs and independent JSON oracle must share stable IDs");

        auto starterSceneValue = starterLightingSceneValue;
        starterSceneValue.materials[0].id = { starter.materialIdValue() };
        starterSceneValue.objects[0].material = { starter.materialIdValue() };
        const auto starterScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
                std::move (starterSceneValue));

        surfacematerialbinding::ImportedSceneMaterialRequest request;
        request.scene = starterScene->id;
        request.sceneRevision = 1;
        request.structuralRevision = 1;
        request.evaluationRevision = 1;
        request.programRevision = 1;
        request.sceneSnapshot = starterScene;
        request.program = starter.program();
        request.binding.targetKind
            = surfacematerialbinding::BindingTargetKind::ObjectOverride;
        request.binding.object = videohelper::fixture3d::kCubeObjectId;
        request.binding.surfaceMaterialRevision = 1;
        std::string admissionError;
        const auto admittedProgram = surfacematerial::admit (request.program, admissionError);
        if (admittedProgram.has_value())
            request.binding.surfaceMaterialDigest = admittedProgram->structuralDigest();
        const auto binding = videorender::fixture3d::admitSurfaceMaterialBinding (
            starterScene, request, videohelper::materialprogram::BackendTarget::OpenGl,
            admissionError);
        ok &= expect (binding != nullptr,
                      "each generated exact starter program must pass OpenGL product admission");
        if (binding == nullptr)
            continue;

        const auto& block = binding->nativeProgram()->parameters;
        const std::array<std::array<float, 4>, 4> actualFloatLanes {{
            block.baseColorMetallic, block.emissionRoughness,
            block.normalOpacity, block.transmissionIorClearcoat
        }};
        bool exactBlock = true;
        for (std::size_t lane = 0; lane < actualFloatLanes.size(); ++lane)
            for (std::size_t channel = 0; channel < 4; ++channel)
                exactBlock &= std::abs (actualFloatLanes[lane][channel]
                    - expected.pbr[lane * 4 + channel]) < 0.000001f;
        exactBlock &= block.identifiers[0]
            == static_cast<std::uint32_t> (expected.pbr[16]);
        ok &= expect (exactBlock,
                      "native OpenGL lowering must match the independent JSON PBR oracle");

        videorender::fixture3d::FixtureSceneRenderer previewOwner (backend);
        videorender::fixture3d::FixtureSceneRenderer exportOwner (backend);
        videorender::fixture3d::RenderedFrame preview;
        videorender::fixture3d::RenderedFrame exportFrame;
        const bool previewRendered = previewOwner.renderPreview (
            starterScene, binding, { 64, 64 },
            videorender::fixture3d::kNativeGpuCapability, preview, admissionError);
        const bool exportRendered = exportOwner.renderExport (
            starterScene, binding, { 64, 64 },
            videorender::fixture3d::kNativeGpuCapability, exportFrame, admissionError);
        const auto previewPixels = previewRendered
            ? readRgba8 (gl, preview.nativeFrame) : std::vector<std::uint8_t> {};
        const auto exportPixels = exportRendered
            ? readRgba8 (gl, exportFrame.nativeFrame) : std::vector<std::uint8_t> {};
        ok &= expect (previewRendered && exportRendered
                          && preview.use == videorender::fixture3d::RenderUse::Preview
                          && exportFrame.use == videorender::fixture3d::RenderUse::Export
                          && preview.nativeFrame != nullptr && exportFrame.nativeFrame != nullptr
                          && preview.nativeFrame->colorImageHandle()
                              != exportFrame.nativeFrame->colorImageHandle(),
                      "distinct production preview and export owners must render fresh OpenGL frames");
        ok &= expect (! previewPixels.empty() && previewPixels == exportPixels,
                      "preview and export owners must produce exact matching OpenGL pixels");
        if (previewPixels.size() > center + 3u)
        {
            const std::array<std::uint8_t, 4> actualCenter {{
                previewPixels[center], previewPixels[center + 1u],
                previewPixels[center + 2u], previewPixels[center + 3u]
            }};
            ok &= expect (actualCenter == expected.centerRgba8,
                          "OpenGL center pixels must match the pinned independent JSON oracle");
            ok &= expect (std::find (expected.rejectedAlternatives.begin(),
                                    expected.rejectedAlternatives.end(), actualCenter)
                              == expected.rejectedAlternatives.end(),
                          "OpenGL pixels must reject black-output and wrong-transfer shading alternatives");
            sampledStarterPixels.insert (sampledStarterPixels.end(),
                                         actualCenter.begin(), actualCenter.end());

            const auto rejectsControlledMaterialAlternative = [&] (const char* label,
                                                                    auto mutate)
            {
                auto alternative = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
                    *binding->nativeProgram());
                alternative->bindingDigest += label;
                alternative->programIdentity += label;
                mutate (alternative->parameters);
                const auto prepared = backend.prepare (starterScene, alternative);
                if (! prepared.prepared) return false;
                const auto rendered = backend.render (starterScene, prepared.resources, 64, 64, {});
                const auto pixels = rendered.rendered ? readRgba8 (gl, rendered.frame)
                                                      : std::vector<std::uint8_t> {};
                if (pixels.size() <= center + 3u) return false;
                const std::array<std::uint8_t, 4> value {{ pixels[center], pixels[center + 1u],
                                                           pixels[center + 2u], pixels[center + 3u] }};
                return value != expected.centerRgba8;
            };
            if (starterIndex == 0)
            {
                ok &= expect (rejectsControlledMaterialAlternative ("/base-color", [] (auto& p)
                    { p.baseColorMetallic[0] = 0.9f; }),
                    "OpenGL lit pixels must detect a wrong base-color binding");
                ok &= expect (rejectsControlledMaterialAlternative ("/roughness", [] (auto& p)
                    { p.emissionRoughness[3] = 0.0f; }),
                    "OpenGL lit pixels must detect a wrong roughness binding");
                ok &= expect (rejectsControlledMaterialAlternative ("/emission", [] (auto& p)
                    { p.emissionRoughness[0] = 0.25f; }),
                    "OpenGL lit pixels must detect a wrong emission binding");
            }
            if (starterIndex == 1)
                ok &= expect (rejectsControlledMaterialAlternative ("/metallic", [] (auto& p)
                    { p.baseColorMetallic[3] = 0.0f; }),
                    "OpenGL lit pixels must detect a wrong metallic binding");
            if (starterIndex == 2)
                ok &= expect (rejectsControlledMaterialAlternative ("/opacity", [] (auto& p)
                    { p.normalOpacity[3] = 1.0f; }),
                    "OpenGL lit pixels must detect a wrong opacity binding");

            if (starterIndex != 0)
                continue;
            auto wrongNormalSceneValue = *starterScene;
            for (std::size_t vertex = 0; vertex < wrongNormalSceneValue.vertexCount; ++vertex)
                wrongNormalSceneValue.vertices[vertex].normal
                    = { -wrongNormalSceneValue.vertices[vertex].normal.x,
                        -wrongNormalSceneValue.vertices[vertex].normal.y,
                        -wrongNormalSceneValue.vertices[vertex].normal.z };
            const auto wrongNormalScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
                std::move (wrongNormalSceneValue));
            const auto wrongNormalPreparation = backend.prepare (wrongNormalScene, binding->nativeProgram());
            std::vector<std::uint8_t> wrongNormalPixels;
            if (wrongNormalPreparation.prepared)
            {
                const auto wrongNormalRender = backend.render (
                    wrongNormalScene, wrongNormalPreparation.resources, 64, 64, {});
                if (wrongNormalRender.rendered)
                    wrongNormalPixels = readRgba8 (gl, wrongNormalRender.frame);
            }
            ok &= expect (wrongNormalPixels.size() > center + 3u
                              && ! std::equal (actualCenter.begin(), actualCenter.end(),
                                               wrongNormalPixels.begin() + center),
                          "OpenGL lit pixels must detect a wrong transformed normal");
        }
    }
    const auto sampledDigest = fnv1a64 (sampledStarterPixels);
    char sampledDigestText[17] {};
    std::snprintf (sampledDigestText, sizeof (sampledDigestText), "%016llx",
                   static_cast<unsigned long long> (sampledDigest));
    ok &= expect (sampledStarterPixels.size() == 20
                      && oracle.sampledRgba8Fnv1a64 == sampledDigestText,
                  "all five OpenGL samples must match the pinned JSON digest");
    videohelper::Sha256 sampledSha256;
    sampledSha256.update (sampledStarterPixels.data(), sampledStarterPixels.size());
    ok &= expect (oracle.sampledRgba8Sha256 == sampledSha256.finishHex(),
                  "all five OpenGL samples must match the pinned SHA-256 digest");

    auto malformedStarter
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    malformedStarter->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    malformedStarter->object = videohelper::fixture3d::kCubeObjectId;
    malformedStarter->bindingDigest = "visual.material.malformed/binding/v1";
    malformedStarter->programIdentity = "visual.material.malformed/program/v1";
    malformedStarter->parameters = material->parameters;
    malformedStarter->parameters.emissionRoughness[0]
        = std::numeric_limits<float>::quiet_NaN();
    const auto malformedStarterPreparation = backend.prepare (scene, malformedStarter);
    ok &= expect (! malformedStarterPreparation.prepared
                      && malformedStarterPreparation.resources == nullptr
                      && malformedStarterPreparation.stats.staticUploadCount == 0
                      && malformedStarterPreparation.stats.materialProgramUploadCount == 0,
                  "malformed starter values must fail before native OpenGL allocation");

    auto wrongSceneValue = sceneValue;
    wrongSceneValue.objects[0].material = HarmonicMIDI::grid::SceneMaterialId { 999 };
    const auto wrongScene =
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (wrongSceneValue);
    const auto wrongPreparation = backend.prepare (wrongScene, material);
    ok &= expect (! wrongPreparation.prepared && wrongPreparation.resources == nullptr,
                  "a mismatched object/material identity must fail closed");

    auto composedSceneValue = sceneValue;
    composedSceneValue.objectCount = 2;
    composedSceneValue.materialCount = 2;
    composedSceneValue.objects[1] = composedSceneValue.objects[0];
    composedSceneValue.objects[1].id.value = 999;
    composedSceneValue.objects[1].parent = composedSceneValue.objects[0].id;
    // The local offset is the parent's inverse rotation of world +X, so the
    // child lands in a separate right-side draw region while exercising hierarchy.
    composedSceneValue.objects[1].transform.translation
        = { 2.5980762f, -0.5130302f, 1.4095389f };
    composedSceneValue.materials[1] = composedSceneValue.materials[0];
    composedSceneValue.materials[1].id.value = 998;
    composedSceneValue.materials[1].emissive = { 0.01f, 0.8f, 0.02f };
    composedSceneValue.objects[1].material = composedSceneValue.materials[1].id;
    auto objectZeroSceneValue = composedSceneValue;
    objectZeroSceneValue.objectCount = 1;
    auto objectZeroScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        std::move (objectZeroSceneValue));
    const auto objectZeroPreparation = backend.prepare (objectZeroScene);
    const auto objectZeroFrame = backend.render (
        objectZeroScene, objectZeroPreparation.resources, 64, 64, {});
    const auto objectZeroPixels = objectZeroFrame.rendered
        ? readRgba8 (gl, objectZeroFrame.frame) : std::vector<std::uint8_t> {};
    const auto composedScene
        = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (composedSceneValue);
    const auto composedPreparation = backend.prepare (composedScene);
    auto composed = backend.render (
        composedScene, composedPreparation.resources, 64, 64, {});
    const auto composedPixels = composed.rendered
        ? readRgba8 (gl, composed.frame) : std::vector<std::uint8_t> {};
    const auto composedExport = backend.render (
        composedScene, composedPreparation.resources, 64, 64, {});
    const auto composedExportPixels = composedExport.rendered
        ? readRgba8 (gl, composedExport.frame) : std::vector<std::uint8_t> {};
    const std::array<std::uint8_t, 4> background { 7, 10, 18, 255 };
    std::size_t secondObjectPixels = 0;
    std::size_t changedObjectZeroPixels = 0;
    bool objectZeroUnchanged = composedPixels.size() == objectZeroPixels.size();
    for (std::uint32_t y = 0; composedPixels.size() == objectZeroPixels.size() && y < 64u; ++y)
        for (std::uint32_t x = 0; x < 64u; ++x)
        {
            const auto offset = (y * 64u + x) * 4u;
            if (! std::equal (composedPixels.begin() + offset,
                              composedPixels.begin() + offset + 4u,
                              objectZeroPixels.begin() + offset))
            {
                const auto wasBackground = std::equal (
                    objectZeroPixels.begin() + offset, objectZeroPixels.begin() + offset + 4u,
                    background.begin());
                objectZeroUnchanged &= wasBackground;
                changedObjectZeroPixels += wasBackground ? 0u : 1u;
                secondObjectPixels += composedPixels[offset + 1u]
                    > composedPixels[offset] * 4u ? 1u : 0u;
            }
        }
    const auto composedContractHolds = objectZeroPreparation.prepared && objectZeroFrame.rendered
        && objectZeroFrame.stats.drawCount == 1
        && composedPreparation.prepared && composed.rendered
        && composedExport.rendered
        && composedPreparation.stats.staticUploadCount == 2
        && composed.stats.drawCount == 2
        && composed.stats.reusedStaticResources
        && composedExport.stats.reusedStaticResources
        && composedPixels.size() == 64u * 64u * 4u
        && objectZeroUnchanged && secondObjectPixels > 0
        && composedPixels == composedExportPixels;
    if (! composedContractHolds)
        std::cerr << "OpenGL composed diagnostics: prepared=" << composedPreparation.prepared
                  << " rendered=" << composed.rendered
                  << " uploads=" << composedPreparation.stats.staticUploadCount
                  << " draws=" << composed.stats.drawCount
                  << " object0=" << objectZeroUnchanged
                  << " changedObject0=" << changedObjectZeroPixels
                  << " green=" << secondObjectPixels << '\n';
    ok &= expect (composedContractHolds,
                  "only the second OpenGL object must add distinct green pixels in its own region");

    auto transparentSceneValue = sceneValue;
    transparentSceneValue.materials[0].opacity = 0.5f;
    const auto transparentPreparation = backend.prepare (
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (transparentSceneValue),
        material);
    ok &= expect (! transparentPreparation.prepared
                      && transparentPreparation.resources == nullptr
                      && transparentPreparation.stats.staticUploadCount == 0,
                  "OpenGL rejects unsupported opacity before GPU upload");

    auto wrongMaterial = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (*material);
    wrongMaterial->parameters.identifiers[0] += 1u;
    const auto wrongMaterialPreparation = backend.prepare (scene, wrongMaterial);
    ok &= expect (! wrongMaterialPreparation.prepared
                      && wrongMaterialPreparation.resources == nullptr
                      && wrongMaterialPreparation.stats.staticUploadCount == 0
                      && wrongMaterialPreparation.stats.materialProgramUploadCount == 0,
                  "a malformed Surface PBR identity must fail before OpenGL allocation");

    auto clippedSceneValue = sceneValue;
    clippedSceneValue.cameras[0].nearPlane = 7.0f;
    auto clippedScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        clippedSceneValue);
    auto clippedPreparation = backend.prepare (clippedScene, material);
    auto clipped = backend.render (
        clippedScene, clippedPreparation.resources, 64, 64, {});
    const auto clippedPixels = clipped.rendered
        ? readRgba8 (gl, clipped.frame) : std::vector<std::uint8_t> {};
    ok &= expect (clippedPreparation.prepared && clipped.rendered
                      && clippedPixels.size() > center + 3u
                      && clippedPixels[center] == 7u
                      && clippedPixels[center + 1u] == 10u
                      && clippedPixels[center + 2u] == 18u
                      && clippedPixels[center + 3u] == 255u,
                  "the configured OpenGL near plane must clip the whole fixture cube");

    auto texturedSceneValue = videohelper::fixture3d::makeScene();
    texturedSceneValue.lightCount = 0;
    texturedSceneValue.ambientColor = { 1.0f, 1.0f, 1.0f };
    texturedSceneValue.vertexCount = 4;
    texturedSceneValue.indexCount = 6;
    texturedSceneValue.vertices[0] = { { -1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 0.0f, 0.0f } };
    texturedSceneValue.vertices[1] = { { 1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 1.0f, 0.0f } };
    texturedSceneValue.vertices[2] = { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 0.0f, 1.0f } };
    texturedSceneValue.vertices[3] = { { 1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 1.0f, 1.0f } };
    texturedSceneValue.indices[0] = 0;
    texturedSceneValue.indices[1] = 2;
    texturedSceneValue.indices[2] = 1;
    texturedSceneValue.indices[3] = 1;
    texturedSceneValue.indices[4] = 2;
    texturedSceneValue.indices[5] = 3;
    texturedSceneValue.objects[0].transform = {};
    texturedSceneValue.objects[0].vertexCount = 4;
    texturedSceneValue.objects[0].indexCount = 6;
    texturedSceneValue.cameras[0].transform = {};
    texturedSceneValue.cameras[0].transform.translation.z = 2.0f;
    texturedSceneValue.textureTexels[0] = { 128, 0, 0, 255 };
    texturedSceneValue.textureTexels[1] = { 0, 128, 0, 255 };
    texturedSceneValue.textureTexels[2] = { 0, 0, 128, 255 };
    texturedSceneValue.textureTexels[3] = { 128, 128, 128, 255 };
    texturedSceneValue.textures[0].minFilter = GL_NEAREST;
    texturedSceneValue.textures[0].magFilter = GL_NEAREST;
    auto texturedScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        texturedSceneValue);
    auto texturedMaterial = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
        *material);
    texturedMaterial->bindingDigest = "opengl-fixture-texture-binding";
    texturedMaterial->programIdentity = "opengl-fixture-texture-program";
    texturedMaterial->baseColorSource =
        arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture;
    arbitgpu::NativeFixtureSurfaceMaterialProgram::ImportedSrgbTexture importedTexture;
    importedTexture.width = 2;
    importedTexture.height = 2;
    importedTexture.texelCount = 4;
    std::copy_n (texturedSceneValue.textureTexels.begin(), 4,
                 importedTexture.texels.begin());
    texturedMaterial->importedBaseColorTexture = std::move (importedTexture);
    texturedMaterial->parameters.baseColorMetallic = { 1.0f, 1.0f, 1.0f, 0.0f };
    texturedMaterial->parameters.emissionRoughness = { 0.0f, 0.0f, 0.0f, 0.0f };
    auto texturedPreparation = backend.prepare (texturedScene, texturedMaterial);
    auto textured = backend.render (
        texturedScene, texturedPreparation.resources, 64, 64, {});
    if (! texturedPreparation.prepared)
        std::cerr << "OpenGL textured preparation error: "
                  << texturedPreparation.error << '\n';
    if (! textured.rendered)
        std::cerr << "OpenGL textured render error: " << textured.error << '\n';
    const auto texturedPixels = textured.rendered
        ? readRgba8 (gl, textured.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("authored-quadrants", texturedPixels);
    const auto channel = [&] (std::uint32_t x, std::uint32_t y, std::uint32_t lane)
    {
        return texturedPixels[(y * 64u + x) * 4u + lane];
    };
    const auto decodedMidtone = [] (std::uint8_t value)
    {
        return value >= 40u && value <= 80u;
    };
    const auto texturedPixelsOk = texturedPreparation.prepared && textured.rendered
        && texturedPixels.size() == 64u * 64u * 4u
        && decodedMidtone (channel (16, 48, 0))
        && channel (16, 48, 1) < 5u && channel (16, 48, 2) < 5u
        && decodedMidtone (channel (48, 48, 1))
        && channel (48, 48, 0) < 5u && channel (48, 48, 2) < 5u
        && decodedMidtone (channel (16, 16, 2))
        && channel (16, 16, 0) < 5u && channel (16, 16, 1) < 5u
        && decodedMidtone (channel (48, 16, 0))
        && decodedMidtone (channel (48, 16, 1))
        && decodedMidtone (channel (48, 16, 2));
    if (! texturedPixelsOk && texturedPixels.size() == 64u * 64u * 4u)
    {
        for (const auto y : { 48u, 16u })
            for (const auto x : { 16u, 48u })
                std::cerr << "quadrant " << x << ',' << y << " = "
                          << static_cast<unsigned> (channel (x, y, 0)) << ','
                          << static_cast<unsigned> (channel (x, y, 1)) << ','
                          << static_cast<unsigned> (channel (x, y, 2)) << '\n';
    }
    ok &= expect (texturedPixelsOk,
                  "the top-left texture rows and sRGB samples must reach their authored quadrants");

    const auto renderStaticPixels = [&] (const HarmonicMIDI::grid::Visual3DScene& value)
    {
        auto snapshot = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (value);
        auto prepared = backend.prepare (snapshot, nullptr);
        if (! prepared.prepared)
        {
            std::cerr << "OpenGL static GLB semantic preparation error: "
                      << prepared.error << '\n';
            return std::vector<std::uint8_t> {};
        }
        auto rendered = backend.render (snapshot, prepared.resources, 64, 64, {});
        if (! rendered.rendered)
        {
            std::cerr << "OpenGL static GLB semantic render error: "
                      << rendered.error << '\n';
            return std::vector<std::uint8_t> {};
        }
        return readRgba8 (gl, rendered.frame);
    };
    const auto nonBackgroundCount = [] (const std::vector<std::uint8_t>& pixels)
    {
        std::size_t count = 0;
        for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
            if (pixels[index] != 7u || pixels[index + 1] != 10u
                || pixels[index + 2] != 18u || pixels[index + 3] != 255u)
                ++count;
        return count;
    };

    auto semanticScene = texturedSceneValue;
    semanticScene.textureCount = 0;
    semanticScene.textureTexelCount = 0;
    semanticScene.materials[0].baseColorTexture = {};
    semanticScene.materials[0].baseColor = { 0.8f, 0.1f, 0.05f };
    semanticScene.materials[0].roughness = 0.2f;
    semanticScene.ambientColor = { 0.3f, 0.3f, 0.3f };
    semanticScene.objectCount = 2;
    semanticScene.vertexCount = 8;
    semanticScene.indexCount = 12;
    semanticScene.materialCount = 2;
    semanticScene.materials[1] = semanticScene.materials[0];
    semanticScene.materials[1].id = HarmonicMIDI::grid::SceneMaterialId { 2 };
    semanticScene.materials[1].baseColor = { 0.05f, 0.7f, 0.2f };
    semanticScene.objects[0].transform.translation.x = -0.55f;
    semanticScene.objects[0].transform.scale = { 0.45f, 0.8f, 1.0f };
    semanticScene.objects[1] = semanticScene.objects[0];
    semanticScene.objects[1].id = HarmonicMIDI::grid::SceneObjectId { 2 };
    semanticScene.objects[1].material = HarmonicMIDI::grid::SceneMaterialId { 2 };
    semanticScene.objects[1].firstVertex = 4;
    semanticScene.objects[1].firstIndex = 6;
    semanticScene.objects[1].transform.translation.x = 0.55f;
    std::copy_n (semanticScene.vertices.begin(), 4, semanticScene.vertices.begin() + 4);
    std::copy_n (semanticScene.indices.begin(), 6, semanticScene.indices.begin() + 6);
    const auto multiPrimitivePixels = renderStaticPixels (semanticScene);
    auto shiftedCameraScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    shiftedCameraScene->cameras[0].transform.translation.x += 0.5f;
    const auto shiftedCameraPixels = renderStaticPixels (*shiftedCameraScene);

    auto onePrimitiveScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    onePrimitiveScene->objectCount = 1;
    onePrimitiveScene->vertexCount = 4;
    onePrimitiveScene->indexCount = 6;
    const auto onePrimitivePixels = renderStaticPixels (*onePrimitiveScene);

    auto changedMaterialScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    changedMaterialScene->materials[1].baseColor = { 0.05f, 0.1f, 0.9f };
    changedMaterialScene->materials[1].metallic = 1.0f;
    changedMaterialScene->materials[1].roughness = 1.0f;
    const auto changedMaterialPixels = renderStaticPixels (*changedMaterialScene);

    auto hierarchyScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    hierarchyScene->objects[0].transform.translation.x = -0.8f;
    hierarchyScene->objects[1].parent = hierarchyScene->objects[0].id;
    hierarchyScene->objects[1].transform.translation.x = 1.1f;
    const auto hierarchyPixels = renderStaticPixels (*hierarchyScene);
    auto localOnlyScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*hierarchyScene);
    localOnlyScene->objects[1].parent = {};
    const auto localOnlyPixels = renderStaticPixels (*localOnlyScene);

    auto triangleTopologyScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    triangleTopologyScene->objects[1].indexCount = 3;
    triangleTopologyScene->indexCount = 9;
    const auto triangleTopologyPixels = renderStaticPixels (*triangleTopologyScene);

    auto directionalScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    directionalScene->ambientColor = {};
    directionalScene->lightCount = 1;
    directionalScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    directionalScene->lights[0].transform = {};
    directionalScene->lights[0].intensity = 1.0f;
    const auto directionalPixels = renderStaticPixels (*directionalScene);
    auto pointScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*directionalScene);
    pointScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Point;
    pointScene->lights[0].transform.translation = { 0.0f, 0.0f, 1.0f };
    pointScene->lights[0].range = 5.0f;
    const auto pointPixels = renderStaticPixels (*pointScene);

    auto spotScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*pointScene);
    spotScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Spot;
    spotScene->lights[0].innerConeAngle = 0.15f;
    spotScene->lights[0].outerConeAngle = 0.35f;
    const auto spotInsidePixels = renderStaticPixels (*spotScene);
    auto spotOutsideScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*spotScene);
    // Rotate the cone away while holding position, range, and point-to-surface distance fixed.
    spotOutsideScene->lights[0].transform.rotation = { 0.0f, 1.0f, 0.0f, 0.0f };
    const auto spotOutsidePixels = renderStaticPixels (*spotOutsideScene);
    auto infiniteRangeScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*pointScene);
    infiniteRangeScene->lights[0].range = 0.0f;
    const auto infiniteRangePixels = renderStaticPixels (*infiniteRangeScene);
    auto finiteRangeScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*infiniteRangeScene);
    finiteRangeScene->lights[0].range = 0.25f; // Range is the pair's only changed field.
    const auto finiteRangePixels = renderStaticPixels (*finiteRangeScene);
    auto noLightScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*finiteRangeScene);
    noLightScene->lightCount = 0;
    const auto noLightPixels = renderStaticPixels (*noLightScene);

    auto backFaceScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    std::swap (backFaceScene->indices[0], backFaceScene->indices[1]);
    std::swap (backFaceScene->indices[3], backFaceScene->indices[4]);
    backFaceScene->materials[0].doubleSided = false;
    const auto culledPixels = renderStaticPixels (*backFaceScene);
    backFaceScene->materials[0].doubleSided = true;
    const auto doubleSidedPixels = renderStaticPixels (*backFaceScene);
    auto reflectedScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    reflectedScene->objects[0].transform.scale.x = -1.0f;
    reflectedScene->materials[0].doubleSided = false;
    const auto reflectedPixels = renderStaticPixels (*reflectedScene);

    auto maskedScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    maskedScene->materials[0].alphaMode = HarmonicMIDI::grid::SceneAlphaMode::Mask;
    maskedScene->materials[0].opacity = 0.25f;
    maskedScene->materials[0].alphaCutoff = 0.5f;
    const auto maskedPixels = renderStaticPixels (*maskedScene);
    auto blendedScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    blendedScene->materials[0].alphaMode = HarmonicMIDI::grid::SceneAlphaMode::Blend;
    blendedScene->materials[0].opacity = 0.25f;
    const auto blendedPixels = renderStaticPixels (*blendedScene);
    auto overlappingBlendScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    overlappingBlendScene->objectCount = 3;
    overlappingBlendScene->materialCount = 3;
    overlappingBlendScene->vertexCount = 12;
    overlappingBlendScene->indexCount = 18;
    for (std::size_t index = 0; index < 3; ++index)
    {
        overlappingBlendScene->objects[index] = overlappingBlendScene->objects[0];
        overlappingBlendScene->objects[index].id
            = HarmonicMIDI::grid::SceneObjectId { static_cast<std::uint32_t>(index + 1) };
        overlappingBlendScene->objects[index].material
            = HarmonicMIDI::grid::SceneMaterialId { static_cast<std::uint32_t>(index + 1) };
        overlappingBlendScene->objects[index].firstVertex = static_cast<std::uint32_t>(index * 4);
        overlappingBlendScene->objects[index].firstIndex = static_cast<std::uint32_t>(index * 6);
        overlappingBlendScene->materials[index] = overlappingBlendScene->materials[0];
        overlappingBlendScene->materials[index].id = overlappingBlendScene->objects[index].material;
        overlappingBlendScene->materials[index].baseColor = {
            index == 0 ? 1.0f : 0.0f, index == 1 ? 1.0f : 0.0f,
            index == 2 ? 1.0f : 0.0f };
        overlappingBlendScene->materials[index].opacity = 0.5f;
        overlappingBlendScene->materials[index].alphaMode
            = HarmonicMIDI::grid::SceneAlphaMode::Blend;
        std::copy_n (overlappingBlendScene->vertices.begin(), 4,
                     overlappingBlendScene->vertices.begin() + index * 4);
        std::copy_n (overlappingBlendScene->indices.begin(), 6,
                     overlappingBlendScene->indices.begin() + index * 6);
    }
    const auto overlappingBlendPixels = renderStaticPixels (*overlappingBlendScene);
    const auto repeatedBlendPixels = renderStaticPixels (*overlappingBlendScene);
    auto reversedBlendScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*overlappingBlendScene);
    std::swap (reversedBlendScene->objects[0], reversedBlendScene->objects[2]);
    const auto reversedBlendPixels = renderStaticPixels (*reversedBlendScene);

    auto packedTextureScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    packedTextureScene->ambientColor = { 0.05f, 0.05f, 0.05f };
    packedTextureScene->lightCount = 1;
    packedTextureScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    packedTextureScene->lights[0].transform = {};
    packedTextureScene->lights[0].intensity = 2.0f;
    packedTextureScene->materials[0].metallic = 0.8f;
    packedTextureScene->materials[0].roughness = 1.0f;
    packedTextureScene->textureCount = 5;
    packedTextureScene->textureTexelCount = 5;
    for (std::size_t index = 0; index < packedTextureScene->textureCount; ++index)
    {
        packedTextureScene->textures[index].id
            = HarmonicMIDI::grid::SceneTextureId { static_cast<std::uint32_t> (index + 1) };
        packedTextureScene->textures[index].firstTexel = static_cast<std::uint32_t> (index);
        packedTextureScene->textures[index].width = 1;
        packedTextureScene->textures[index].height = 1;
        packedTextureScene->textures[index].minFilter = GL_NEAREST;
        packedTextureScene->textures[index].magFilter = GL_NEAREST;
    }
    packedTextureScene->textureTexels[0] = { 128, 96, 192, 255 };
    packedTextureScene->textureTexels[1] = { 255, 128, 128, 255 };
    packedTextureScene->textureTexels[2] = { 64, 255, 255, 255 };
    packedTextureScene->textureTexels[3] = { 32, 192, 255, 255 };
    packedTextureScene->textureTexels[4] = packedTextureScene->textureTexels[0];
    packedTextureScene->materials[0].metallicRoughnessTexture
        = packedTextureScene->textures[0].id;
    packedTextureScene->materials[0].normalTexture = packedTextureScene->textures[1].id;
    packedTextureScene->materials[0].occlusionTexture = packedTextureScene->textures[2].id;
    packedTextureScene->materials[0].emissiveTexture = packedTextureScene->textures[3].id;
    packedTextureScene->materials[0].emissive = { 0.2f, 0.2f, 0.2f };
    const auto packedTexturePixels = renderStaticPixels (*packedTextureScene);
    auto reflectedNormalScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    reflectedNormalScene->objects[0].transform.scale.x = -1.0f;
    const auto reflectedNormalPixels = renderStaticPixels (*reflectedNormalScene);
    auto reflectedWithoutNormal = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*reflectedNormalScene);
    reflectedWithoutNormal->materials[0].normalTexture = {};
    const auto reflectedWithoutNormalPixels = renderStaticPixels (*reflectedWithoutNormal);
    // Independent analytic reflection. It bakes the reflection into geometry and tangent
    // handedness, so it has a positive object determinant but the same mapped surface.
    auto analyticReflectedNormal = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    for (std::size_t index = 0; index < analyticReflectedNormal->vertexCount; ++index)
    {
        analyticReflectedNormal->vertices[index].position.x *= -1.0f;
        analyticReflectedNormal->vertices[index].normal.x *= -1.0f;
        analyticReflectedNormal->vertices[index].tangent[0] *= -1.0f;
        analyticReflectedNormal->vertices[index].tangent[3] *= -1.0f;
    }
    for (std::size_t index = 0; index + 2 < analyticReflectedNormal->indexCount; index += 3)
        std::swap (analyticReflectedNormal->indices[index], analyticReflectedNormal->indices[index + 1]);
    const auto analyticReflectedNormalPixels = renderStaticPixels (*analyticReflectedNormal);
    auto sharedRoleScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    sharedRoleScene->materials[0].baseColorTexture = sharedRoleScene->textures[0].id;
    sharedRoleScene->materials[0].metallicRoughnessTexture = sharedRoleScene->textures[0].id;
    const auto sharedRolePixels = renderStaticPixels (*sharedRoleScene);
    // The duplicate binding is an independent reference for the required role split.
    auto splitRoleReference = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*sharedRoleScene);
    splitRoleReference->materials[0].baseColorTexture = splitRoleReference->textures[4].id;
    const auto splitRoleReferencePixels = renderStaticPixels (*splitRoleReference);
    auto wronglyLinearBase = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*splitRoleReference);
    // These are sRGB encodings of the original numeric channel values. A correct
    // sRGB base-color view decodes them to the forbidden linear-sampling result.
    wronglyLinearBase->textureTexels[4] = { 188, 165, 225, 255 };
    const auto wronglyLinearBasePixels = renderStaticPixels (*wronglyLinearBase);
    auto wronglySrgbMetallicRoughness = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*splitRoleReference);
    // sRGB decoding maps 128 to about 55. Using those bytes in the linear role models
    // the forbidden reuse of an sRGB image view without depending on backend code.
    wronglySrgbMetallicRoughness->textureTexels[0] = { 55, 30, 134, 255 };
    const auto wronglySrgbMetallicRoughnessPixels = renderStaticPixels (*wronglySrgbMetallicRoughness);
    std::cerr << "semantic reflected=" << std::hex << fnv1a64 (reflectedNormalPixels)
              << " analytic-reflected=" << fnv1a64 (analyticReflectedNormalPixels)
              << " shared-role=" << fnv1a64 (sharedRolePixels)
              << " split-role=" << fnv1a64 (splitRoleReferencePixels)
              << " wrong-linear-base=" << fnv1a64 (wronglyLinearBasePixels)
              << " wrong-srgb-data=" << fnv1a64 (wronglySrgbMetallicRoughnessPixels)
              << std::dec << '\n';
    auto noMetallicRoughnessTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noMetallicRoughnessTexture->materials[0].metallicRoughnessTexture = {};
    auto noNormalTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noNormalTexture->materials[0].normalTexture = {};
    auto noOcclusionTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noOcclusionTexture->materials[0].occlusionTexture = {};
    auto noEmissiveTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noEmissiveTexture->materials[0].emissiveTexture = {};
    auto semanticSnapshot =
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    auto semanticPreparation = backend.prepare (semanticSnapshot, nullptr);
    const auto generationFrameA = backend.render (
        semanticSnapshot, semanticPreparation.resources, 64, 64, {});
    const auto generationFrameB = backend.render (
        semanticSnapshot, semanticPreparation.resources, 64, 64, {});
    const auto reprepared = backend.prepare (semanticSnapshot, nullptr);
    const auto generationFrameC = backend.render (
        semanticSnapshot, reprepared.resources, 64, 64, {});
    const auto invalidExtent = backend.render (
        semanticSnapshot, semanticPreparation.resources, 0, 64, {});

    const std::array<std::pair<const char*, bool>, 22> semanticChecks {{
        { "primitive count", multiPrimitivePixels != onePrimitivePixels },
        { "camera", shiftedCameraPixels != multiPrimitivePixels },
        { "material", changedMaterialPixels != multiPrimitivePixels },
        { "hierarchy", hierarchyPixels != localOnlyPixels },
        { "topology", triangleTopologyPixels != multiPrimitivePixels },
        { "light kind", directionalPixels != pointPixels },
        { "spot cone", spotInsidePixels != spotOutsidePixels },
        { "finite range", infiniteRangePixels != finiteRangePixels
            && finiteRangePixels == noLightPixels },
        { "single sided", nonBackgroundCount (culledPixels) == 0 },
        { "double sided", nonBackgroundCount (doubleSidedPixels) > 0 },
        { "reflected culling", nonBackgroundCount (reflectedPixels) > 0 },
        { "mask", nonBackgroundCount (maskedPixels) == 0 },
        { "blend", blendedPixels != onePrimitivePixels && nonBackgroundCount (blendedPixels) > 0 },
        { "stable overlapping blend order", overlappingBlendPixels == repeatedBlendPixels
            && overlappingBlendPixels != reversedBlendPixels },
        { "metallic roughness", packedTexturePixels != renderStaticPixels (*noMetallicRoughnessTexture) },
        { "normal", packedTexturePixels != renderStaticPixels (*noNormalTexture) },
        { "reflected tangent", reflectedNormalPixels == analyticReflectedNormalPixels
            && reflectedNormalPixels != reflectedWithoutNormalPixels },
        { "occlusion", packedTexturePixels != renderStaticPixels (*noOcclusionTexture) },
        { "emissive", packedTexturePixels != renderStaticPixels (*noEmissiveTexture) },
        { "shared sRGB and linear image", sharedRolePixels == splitRoleReferencePixels
            && sharedRolePixels != wronglyLinearBasePixels
            && sharedRolePixels != wronglySrgbMetallicRoughnessPixels },
        { "stable generation", generationFrameA.rendered && generationFrameB.rendered
            && generationFrameC.rendered
            && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                == generationFrameB.frame->colorTextureDescriptor().rendererGeneration
            && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                != generationFrameC.frame->colorTextureDescriptor().rendererGeneration },
        { "bounded prepare", semanticPreparation.prepared && ! invalidExtent.rendered }
    }};
    for (const auto& semanticCheck : semanticChecks)
        if (! semanticCheck.second)
            std::cerr << "static GLB semantic check failed: " << semanticCheck.first << '\n';

    ok &= expect (! multiPrimitivePixels.empty()
                      && multiPrimitivePixels != onePrimitivePixels
                      && shiftedCameraPixels != multiPrimitivePixels
                      && changedMaterialPixels != multiPrimitivePixels
                      && hierarchyPixels != localOnlyPixels
                      && triangleTopologyPixels != multiPrimitivePixels
                      && directionalPixels != pointPixels
                      && spotInsidePixels != spotOutsidePixels
                      && infiniteRangePixels != finiteRangePixels
                      && finiteRangePixels == noLightPixels
                      && nonBackgroundCount (culledPixels) == 0
                      && nonBackgroundCount (doubleSidedPixels) > 0
                      && nonBackgroundCount (reflectedPixels) > 0
                      && nonBackgroundCount (maskedPixels) == 0
                      && blendedPixels != onePrimitivePixels
                      && nonBackgroundCount (blendedPixels) > 0
                      && overlappingBlendPixels == repeatedBlendPixels
                      && overlappingBlendPixels != reversedBlendPixels
                      && packedTexturePixels != renderStaticPixels (*noMetallicRoughnessTexture)
                      && packedTexturePixels != renderStaticPixels (*noNormalTexture)
                      && reflectedNormalPixels == analyticReflectedNormalPixels
                      && reflectedNormalPixels != reflectedWithoutNormalPixels
                      && packedTexturePixels != renderStaticPixels (*noOcclusionTexture)
                      && packedTexturePixels != renderStaticPixels (*noEmissiveTexture)
                      && sharedRolePixels == splitRoleReferencePixels
                      && sharedRolePixels != wronglyLinearBasePixels
                      && sharedRolePixels != wronglySrgbMetallicRoughnessPixels
                      && generationFrameA.rendered && generationFrameB.rendered
                      && generationFrameC.rendered
                      && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                          == generationFrameB.frame->colorTextureDescriptor().rendererGeneration
                      && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                          != generationFrameC.frame->colorTextureDescriptor().rendererGeneration
                      && semanticPreparation.prepared
                      && ! invalidExtent.rendered && invalidExtent.frame == nullptr
                      && invalidExtent.error
                           == "OpenGL fixture render dimensions exceed backend limits",
                  "production OpenGL pixels must independently consume primitive topology, materials, hierarchy, alpha mode, sidedness, and directional or point lights");

    std::string cardLoadError;
    const auto cardAsset = videohelper::fixture3d::loadHolographicTradingCardScene(
        cardLoadError);
    ok &= expect(cardAsset.has_value(),
                 "the exact checked-in card GLB must pass production decode and scene adaptation");
    if (!cardAsset)
    {
        std::cerr << "Card GLB load error: " << cardLoadError << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    const auto& diffractionScene = cardAsset->scene;
    ok &= expect(cardAsset->assetId == "builtin.visual-model.holographic-trading-card"
                     && cardAsset->contentSha256
                         == "5e3433aa19cc4636c33a1f6dd53a4a05fdffa39d788ad565b530d8dfdcce0287"
                     && cardAsset->sceneName == "Holographic Trading Card Scene"
                     && cardAsset->objectName == "Trading Card"
                     && cardAsset->materialName == "Physical Diffraction Target"
                     && cardAsset->cameraName == "Card Camera"
                     && cardAsset->lightName == "Card Key"
                     && diffractionScene->vertexCount == 24u
                     && diffractionScene->indexCount == 36u,
                 "card asset, content, names, and decoded geometry must survive adaptation");
    auto diffractionDescription = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    std::string diffractionError;
    auto admittedDiffraction = diffractionmaterial::admit (
        diffractionDescription, diffractionError);
    diffractionmaterial::SpectralIncidentLight incident;
    incident.radiance.fill (0.01f);
    diffractionmaterial::SpectralLightingPath lightingPath;
    lightingPath.kind = diffractionmaterial::LightingPathKind::Direct;
    lightingPath.incident = incident;
    auto diffractionProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    diffractionProgram->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    diffractionProgram->kind
        = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    diffractionProgram->object = diffractionScene->objects[0].id;
    diffractionProgram->bindingDigest = admittedDiffraction
        ? admittedDiffraction->structuralDigest() : std::string {};

    diffractionProgram->diffractionPathCount = 1;
    diffractionmaterial::LightingDescription lightingDescription;
    lightingDescription.pathCount = 1;
    lightingDescription.paths[0] = lightingPath;
    const auto admittedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        lightingDescription, diffractionError);
    if (admittedLighting)
    {
        diffractionProgram->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *admittedLighting);
        diffractionProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                diffractionProgram->bindingDigest, *admittedLighting);
    }
    if (admittedDiffraction)
        diffractionProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedDiffraction, incident), lightingPath);


    auto diffractionPreparation = backend.prepare (
        diffractionScene, diffractionProgram);
    auto replayedLightingProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *diffractionProgram);
    auto replayedLightingDescription = lightingDescription;
    replayedLightingDescription.paths[0].incident.radiance[1]
        = std::nextafter(
            replayedLightingDescription.paths[0].incident.radiance[1], 0.0f);
    const auto replayedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        replayedLightingDescription, diffractionError);
    if (replayedLighting && admittedDiffraction)
    {
        replayedLightingProgram->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *replayedLighting);
        replayedLightingProgram->diffractionPaths[0]
            = diffractionmaterial::makeGpuLightingPath(
                diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                    *admittedDiffraction,
                    replayedLightingDescription.paths[0].incident),
                replayedLightingDescription.paths[0]);
    }
    const auto replayedLightingPreparation = backend.prepare(
        diffractionScene, replayedLightingProgram);
    auto malformedDiffractionProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    malformedDiffractionProgram->programIdentity
        = "opengl-fixture-malformed-diffraction-program";
    malformedDiffractionProgram->diffractionPaths[0].material.spectralZ[7].w
        = std::numeric_limits<float>::quiet_NaN();
    const auto malformedDiffractionPreparation = backend.prepare (
        diffractionScene, malformedDiffractionProgram);
    auto layoutSentinelProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    layoutSentinelProgram->programIdentity
        = "opengl-fixture-layout-sentinel-diffraction-program";
    layoutSentinelProgram->parameters.identifiers[3] = 1u;
    const auto layoutSentinelPreparation = backend.prepare (
        diffractionScene, layoutSentinelProgram);
    auto divergentBlazeProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    divergentBlazeProgram->programIdentity
        = "opengl-fixture-divergent-local-blaze-program";
    divergentBlazeProgram->diffractionPaths[0].material.control.y
        = static_cast<float> (diffractionmaterial::GrooveProfile::BlazedSawtooth);
    divergentBlazeProgram->diffractionPaths[0].material.grooveField = {
        static_cast<float> (diffractionmaterial::GrooveFieldMode::Linear),
        0.5f, 0.5f, 0.0f
    };
    divergentBlazeProgram->diffractionPaths[0].material.grooveVariation
        = { 1.0f, 0.0f, 100.0f, 0.0f };
    const auto divergentBlazePreparation = backend.prepare (
        diffractionScene, divergentBlazeProgram);
    diffractionProgram->diffractionPaths[0].material.roughness.y = 0.0f;
    auto diffractionFrame = backend.render (
        diffractionScene, diffractionPreparation.resources, 96, 96, {});
    if (! diffractionPreparation.prepared)
        std::cerr << "OpenGL diffraction preparation error: "
                  << diffractionPreparation.error << '\n';
    if (! diffractionFrame.rendered)
        std::cerr << "OpenGL diffraction render error: "
                  << diffractionFrame.error << '\n';
    const auto diffractionPixels = diffractionFrame.rendered
        ? readRgba8 (gl, diffractionFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction", diffractionPixels);

    const auto foilEvaluationCount = 320u * 180u;
    const auto& foilRequest = *cardAsset->operation.diffractionMaterial;

    const auto foilPreviewBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, foilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    const auto foilExportBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, foilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    auto wrongCardObjectRequest = foilRequest;
    wrongCardObjectRequest.object = HarmonicMIDI::grid::SceneObjectId { 2 };
    const auto wrongCardObjectBinding
        = videorender::fixture3d::admitDiffractionMaterialBinding(
            diffractionScene, wrongCardObjectRequest,
            videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    if (foilPreviewBinding == nullptr || foilExportBinding == nullptr)
        std::cerr << "Diffractive foil binding error: " << diffractionError << '\n';
    videorender::fixture3d::FixtureSceneRenderer foilPreviewRenderer(backend);
    videorender::fixture3d::FixtureSceneRenderer foilExportRenderer(backend);
    videorender::fixture3d::RenderedFrame foilPreviewFrame;
    videorender::fixture3d::RenderedFrame foilExportFrame;
    const bool foilRendered = foilPreviewBinding != nullptr && foilExportBinding != nullptr
        && foilPreviewBinding != foilExportBinding
        && foilPreviewBinding->nativeProgram() != foilExportBinding->nativeProgram()
        && foilPreviewBinding->bindingDigest() == foilExportBinding->bindingDigest()
        && wrongCardObjectBinding == nullptr
        && foilPreviewRenderer.renderPreview(diffractionScene, foilPreviewBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            foilPreviewFrame, diffractionError)
        && foilExportRenderer.renderExport(diffractionScene, foilExportBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            foilExportFrame, diffractionError);
    if (!foilRendered)
        std::cerr << "Diffractive foil render error: " << diffractionError << '\n';
    ok &= expect(foilRendered
                     && foilPreviewFrame.stats.diffractionEvaluationBudget
                         == foilEvaluationCount
                     && foilPreviewFrame.stats.diffractionEvaluationCount
                         == foilEvaluationCount
                     && foilExportFrame.stats.diffractionEvaluationBudget
                         == foilEvaluationCount
                     && foilExportFrame.stats.diffractionEvaluationCount
                         == foilEvaluationCount,
                 "OpenGL reports the exact deterministic pixel-slot and lighting-path evaluation count");

    auto belowCeilingFoilRequest = foilRequest;
    belowCeilingFoilRequest.spatialFoil->workBudget.maximumEvaluations
        = foilEvaluationCount + 1u;
    const auto belowCeilingAdmission = diffractivefoil::admit(
        *belowCeilingFoilRequest.spatialFoil, diffractionError);
    belowCeilingFoilRequest.structuralDigest = belowCeilingAdmission
        ? belowCeilingAdmission->structuralDigest() : std::string {};
    const auto belowCeilingBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, belowCeilingFoilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    videorender::fixture3d::FixtureSceneRenderer belowCeilingRenderer(backend);
    videorender::fixture3d::RenderedFrame belowCeilingFrame;
    const bool belowCeilingRendered = belowCeilingBinding != nullptr
        && belowCeilingRenderer.renderPreview(diffractionScene, belowCeilingBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            belowCeilingFrame, diffractionError);
    ok &= expect(belowCeilingRendered
                     && belowCeilingFrame.stats.diffractionEvaluationBudget
                         == foilEvaluationCount + 1u
                     && belowCeilingFrame.stats.diffractionEvaluationCount
                         == foilEvaluationCount,
                 "OpenGL admits exact work below the immutable evaluation ceiling");

    auto exhaustedFoilRequest = foilRequest;
    exhaustedFoilRequest.spatialFoil->workBudget.maximumEvaluations
        = foilEvaluationCount - 1u;
    const auto exhaustedAdmission = diffractivefoil::admit(
        *exhaustedFoilRequest.spatialFoil, diffractionError);
    exhaustedFoilRequest.structuralDigest = exhaustedAdmission
        ? exhaustedAdmission->structuralDigest() : std::string {};
    const auto exhaustedBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, exhaustedFoilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    videorender::fixture3d::FixtureSceneRenderer exhaustedRenderer(backend);
    videorender::fixture3d::RenderedFrame exhaustedFrame;
    const bool exhaustedRendered = exhaustedBinding != nullptr
        && exhaustedRenderer.renderPreview(diffractionScene, exhaustedBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            exhaustedFrame, diffractionError);
    ok &= expect(!exhaustedRendered && exhaustedFrame.nativeFrame == nullptr
                     && diffractionError
                         == "OpenGL fixture diffraction workload exceeds backend limits",
                 "OpenGL rejects a spatial foil one evaluation over budget before frame allocation");
    const auto foilPreviewPixels = foilRendered
        ? readRgba8(gl, foilPreviewFrame.nativeFrame) : std::vector<std::uint8_t> {};
    const auto foilExportPixels = foilRendered
        ? readRgba8(gl, foilExportFrame.nativeFrame) : std::vector<std::uint8_t> {};
    const auto belowCeilingPixels = belowCeilingRendered
        ? readRgba8(gl, belowCeilingFrame.nativeFrame) : std::vector<std::uint8_t> {};
    const auto disabledFoilPreparation = backend.prepare(diffractionScene, nullptr);
    const auto disabledFoilFrame = backend.render(
        diffractionScene, disabledFoilPreparation.resources, 320, 180, {});
    const auto disabledFoilPixels = disabledFoilFrame.rendered
        ? readRgba8(gl, disabledFoilFrame.frame) : std::vector<std::uint8_t> {};
    std::size_t disabledFoilDifferingPixels = 0;
    if (foilPreviewPixels.size() == disabledFoilPixels.size())
    {
        for (std::size_t offset = 0; offset < foilPreviewPixels.size(); offset += 4)
        {
            if (!std::equal(foilPreviewPixels.begin() + static_cast<std::ptrdiff_t>(offset),
                            foilPreviewPixels.begin() + static_cast<std::ptrdiff_t>(offset + 3),
                            disabledFoilPixels.begin() + static_cast<std::ptrdiff_t>(offset)))
                ++disabledFoilDifferingPixels;
        }
    }
    ok &= expect(disabledFoilPreparation.prepared && disabledFoilFrame.rendered
                     && disabledFoilPixels.size() == 320u * 180u * 4u
                     && fnv1a64(foilPreviewPixels) != fnv1a64(disabledFoilPixels)
                     && disabledFoilDifferingPixels > 100u,
                 "OpenGL card foil must differ from a same-size disabled-diffraction render of the decoded card scene");

    bool everyFoilCornerPerturbsPixels = foilRendered && !foilPreviewPixels.empty()
        && fnv1a64(foilPreviewPixels) == fnv1a64(foilExportPixels)
        && fnv1a64(foilPreviewPixels) == fnv1a64(belowCeilingPixels);
    for (std::size_t corner = 0; corner < diffractivefoil::kFieldSampleCount; ++corner)
    {
        auto changed = foilRequest;
        changed.spatialFoil->grooveField[corner].grooveSpacingNanometres += 113.0f;
        if (corner == 0)
        {
            changed.spatialFoil->physicalBsdf.geometry.grooveSpacingNanometres
                = changed.spatialFoil->grooveField[0].grooveSpacingNanometres;
            changed.material = changed.spatialFoil->physicalBsdf;
        }
        const auto changedAdmission = diffractivefoil::admit(*changed.spatialFoil, diffractionError);
        changed.structuralDigest = changedAdmission
            ? changedAdmission->structuralDigest() : std::string {};
        const auto changedBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            diffractionScene, changed,
            videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
        videorender::fixture3d::FixtureSceneRenderer changedRenderer(backend);
        videorender::fixture3d::RenderedFrame changedFrame;
        const bool changedRendered = changedBinding != nullptr
            && changedRenderer.renderPreview(diffractionScene, changedBinding, {}, { 320, 180 },
                videorender::fixture3d::kNativeGpuCapability, changedFrame, diffractionError);
        const auto changedPixels = changedRendered
            ? readRgba8(gl, changedFrame.nativeFrame) : std::vector<std::uint8_t> {};
        const auto changedHash = fnv1a64(changedPixels);
        const auto baselineHash = fnv1a64(foilPreviewPixels);
        const bool grooveChanged = changedRendered && changedHash != baselineHash;
        if (!grooveChanged)
            std::cerr << "Diffractive foil groove corner " << corner
                      << " did not change pixels: " << std::hex << baselineHash
                      << " vs " << changedHash << std::dec << '\n';
        everyFoilCornerPerturbsPixels = everyFoilCornerPerturbsPixels && grooveChanged;

        changed = foilRequest;
        changed.spatialFoil->grooveField[corner].diffractionCoverage = 0.0f;
        const auto occupancyAdmission = diffractivefoil::admit(
            *changed.spatialFoil, diffractionError);
        changed.structuralDigest = occupancyAdmission
            ? occupancyAdmission->structuralDigest() : std::string {};
        const auto occupancyBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            diffractionScene, changed,
            videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
        videorender::fixture3d::FixtureSceneRenderer occupancyRenderer(backend);
        videorender::fixture3d::RenderedFrame occupancyFrame;
        const bool occupancyRendered = occupancyBinding != nullptr
            && occupancyRenderer.renderPreview(diffractionScene, occupancyBinding, {}, { 320, 180 },
                videorender::fixture3d::kNativeGpuCapability, occupancyFrame, diffractionError);
        const auto occupancyPixels = occupancyRendered
            ? readRgba8(gl, occupancyFrame.nativeFrame) : std::vector<std::uint8_t> {};
        const auto occupancyHash = fnv1a64(occupancyPixels);
        const bool occupancyChanged = occupancyRendered && occupancyHash != baselineHash;
        if (!occupancyChanged)
            std::cerr << "Diffractive foil occupancy corner " << corner
                      << " did not change pixels: " << std::hex << baselineHash
                      << " vs " << occupancyHash << std::dec << '\n';
        everyFoilCornerPerturbsPixels = everyFoilCornerPerturbsPixels
            && occupancyChanged;
    }
    ok &= expect(everyFoilCornerPerturbsPixels,
                 "graph-authored Diffractive Foil preview and export must share exact immutable semantics and pixels must depend on all four groove and occupancy corners");

    auto malformedFoil = foilRequest;
    malformedFoil.spatialFoil->grooveField[2].reciprocalDirectionUv[0]
        = std::numeric_limits<float>::quiet_NaN();
    const auto malformedFoilBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, malformedFoil,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    auto staleFoil = foilRequest;
    staleFoil.structuralDigest.push_back('0');
    const auto staleFoilBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, staleFoil,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    const auto unsupportedFoilBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, foilRequest,
        videohelper::materialprogram::BackendTarget::Invalid, diffractionError);
    videorender::fixture3d::RenderedFrame rejectedFoilFrame;
    const bool cpuFallbackRendered = foilPreviewBinding != nullptr
        && foilPreviewRenderer.renderPreview(diffractionScene, foilPreviewBinding, {}, { 320, 180 },
            "cpu-rgb", rejectedFoilFrame, diffractionError);
    ok &= expect(malformedFoilBinding == nullptr && staleFoilBinding == nullptr
                     && unsupportedFoilBinding == nullptr && !cpuFallbackRendered
                     && rejectedFoilFrame.nativeFrame == nullptr
                     && rejectedFoilFrame.stats.vertexBytes == 0
                     && rejectedFoilFrame.stats.indexBytes == 0
                     && rejectedFoilFrame.stats.materialBytes == 0
                     && rejectedFoilFrame.stats.textureBytes == 0,
                 "malformed fields, stale digests, unsupported backends, and CPU/RGB fallback fail closed before OpenGL allocation");

    auto transportProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *diffractionProgram);

    transportProgram->diffractionPathCount = 3;
    transportProgram->diffractionMaximumBounceDepth = 1;
    diffractionmaterial::LightingDescription transportLighting;
    transportLighting.pathCount = 3;
    const std::array<diffractionmaterial::LightingPathKind, 3> transportKinds {
        diffractionmaterial::LightingPathKind::Direct,
        diffractionmaterial::LightingPathKind::Environment,
        diffractionmaterial::LightingPathKind::Indirect };
    const std::array<std::array<float, 3>, 3> transportDirections {{
        {{ 0.0f, 0.0f, 1.0f }},
        {{ 0.36f, 0.0f, 0.9329523f }},
        {{ -0.28f, 0.28f, 0.918259f }} }};
    for (std::size_t pathIndex = 0; pathIndex < transportKinds.size(); ++pathIndex)
    {
        auto path = lightingPath;
        path.kind = transportKinds[pathIndex];
        path.bounceDepth = path.kind == diffractionmaterial::LightingPathKind::Indirect ? 1 : 0;
        path.incident.direction = transportDirections[pathIndex];
        transportLighting.paths[pathIndex] = path;
        transportProgram->diffractionPaths[pathIndex]
            = diffractionmaterial::makeGpuLightingPath(
                diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                    *admittedDiffraction, path.incident), path);
    }
    const auto admittedTransportLighting
        = diffractionmaterial::AdmittedLightingPlan::admit(
            transportLighting, diffractionError);
    if (admittedTransportLighting)
    {
        transportProgram->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *admittedTransportLighting);
        transportProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                transportProgram->bindingDigest, *admittedTransportLighting);
    }
    const auto transportPreparation = backend.prepare(diffractionScene, transportProgram);
    const auto transportFrame = backend.render(
        diffractionScene, transportPreparation.resources, 96, 96, {});
    const auto transportPixels = transportFrame.rendered
        ? readRgba8(gl, transportFrame.frame) : std::vector<std::uint8_t> {};
    std::array<std::vector<std::uint8_t>, 3> omittedPathPixels;
    bool everyPathPerturbsPixels = transportPreparation.prepared && transportFrame.rendered;
    for (std::size_t pathIndex = 0; pathIndex < omittedPathPixels.size(); ++pathIndex)
    {
        auto omitted = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *transportProgram);

        auto omittedLighting = transportLighting;
        omittedLighting.paths[pathIndex].incident.radiance.fill(0.0f);
        const auto admittedOmitted = diffractionmaterial::AdmittedLightingPlan::admit(
            omittedLighting, diffractionError);
        if (admittedOmitted)
        {
            omitted->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *admittedOmitted);
            omitted->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                omitted->bindingDigest, *admittedOmitted);
            omitted->diffractionPaths[pathIndex]
                = diffractionmaterial::makeGpuLightingPath(
                    diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                        *admittedDiffraction,
                        omittedLighting.paths[pathIndex].incident),
                    omittedLighting.paths[pathIndex]);
        }
        const auto omittedPreparation = backend.prepare(diffractionScene, omitted);
        const auto omittedFrame = backend.render(
            diffractionScene, omittedPreparation.resources, 96, 96, {});
        if (omittedFrame.rendered)
            omittedPathPixels[pathIndex] = readRgba8(gl, omittedFrame.frame);
        everyPathPerturbsPixels = everyPathPerturbsPixels && omittedPreparation.prepared
            && omittedFrame.rendered && omittedPathPixels[pathIndex] != transportPixels;
    }

    arbitgpu::NativeFixtureSceneRuntimeInputs shiftedCamera;
    shiftedCamera.cameraTranslationOffset[0] = 0.75f;
    auto shiftedDiffractionFrame = backend.render (
        diffractionScene, diffractionPreparation.resources, 96, 96, shiftedCamera);
    const auto shiftedDiffractionPixels = shiftedDiffractionFrame.rendered
        ? readRgba8 (gl, shiftedDiffractionFrame.frame)
        : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-shifted-camera", shiftedDiffractionPixels);

    auto rotatedDescription = diffractionDescription;
    rotatedDescription.geometry.directionUv = { 0.0f, 1.0f };
    auto admittedRotated = diffractionmaterial::admit (
        rotatedDescription, diffractionError);
    auto rotatedProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    rotatedProgram->bindingDigest = admittedRotated
        ? admittedRotated->structuralDigest() : std::string {};
    if (admittedLighting)
        rotatedProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            rotatedProgram->bindingDigest, *admittedLighting);
    if (admittedRotated)
        rotatedProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedRotated, incident), lightingPath);
    auto rotatedPreparation = backend.prepare (diffractionScene, rotatedProgram);
    auto rotatedFrame = backend.render (
        diffractionScene, rotatedPreparation.resources, 96, 96, {});
    const auto rotatedPixels = rotatedFrame.rendered
        ? readRgba8 (gl, rotatedFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-rotated", rotatedPixels);

    auto spatialDescription = diffractionDescription;
    spatialDescription.grooveField.mode = diffractionmaterial::GrooveFieldMode::Linear;
    spatialDescription.grooveField.originUv = { 0.5f, 0.5f };
    spatialDescription.grooveField.axisUv = { 1.0f, 0.0f };
    spatialDescription.grooveField.grooveSpacingDeltaNanometresPerUnit = 500.0f;
    spatialDescription.grooveField.orientationDegreesPerUnit = 90.0f;
    auto admittedSpatial = diffractionmaterial::admit (
        spatialDescription, diffractionError);
    auto spatialProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    spatialProgram->bindingDigest = admittedSpatial
        ? admittedSpatial->structuralDigest() : std::string {};
    if (admittedLighting)
        spatialProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            spatialProgram->bindingDigest, *admittedLighting);
    if (admittedSpatial)
        spatialProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedSpatial, incident), lightingPath);
    auto spatialPreparation = backend.prepare (diffractionScene, spatialProgram);
    auto spatialFrame = backend.render (
        diffractionScene, spatialPreparation.resources, 96, 96, {});
    const auto spatialPixels = spatialFrame.rendered
        ? readRgba8 (gl, spatialFrame.frame) : std::vector<std::uint8_t> {};


    auto crossedDescription
        = diffractionmaterial::makeRealtimeCrossedTwoDimensionalGratingPreset();
    auto admittedCrossed = diffractionmaterial::admit (
        crossedDescription, diffractionError);
    auto crossedProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    crossedProgram->bindingDigest = admittedCrossed
        ? admittedCrossed->structuralDigest() : std::string {};
    if (admittedLighting)
        crossedProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            crossedProgram->bindingDigest, *admittedLighting);
    if (admittedCrossed)
        crossedProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedCrossed, incident), lightingPath);
    auto unequalCrossedBlazedProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *crossedProgram);
    unequalCrossedBlazedProgram->programIdentity
        = "opengl-fixture-unequal-crossed-blazed-program";
    unequalCrossedBlazedProgram->diffractionPaths[0].material.control.y
        = static_cast<float> (diffractionmaterial::GrooveProfile::BlazedSawtooth);
    unequalCrossedBlazedProgram->diffractionPaths[0].material.secondaryGeometry.z
        = unequalCrossedBlazedProgram->diffractionPaths[0].material.geometry.z + 1.0f;
    const auto unequalCrossedBlazedPreparation = backend.prepare (
        diffractionScene, unequalCrossedBlazedProgram);
    auto crossedPreparation = backend.prepare (diffractionScene, crossedProgram);
    auto crossedFrame = backend.render (
        diffractionScene, crossedPreparation.resources, 96, 96, {});
    const auto crossedPixels = crossedFrame.rendered
        ? readRgba8 (gl, crossedFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-crossed", crossedPixels);

    auto crossedDirectionDescription = crossedDescription;
    crossedDirectionDescription.geometry.directionUv
        = { 0.70710678f, 0.70710678f };
    crossedDirectionDescription.geometry.secondaryDirectionUv
        = { -0.70710678f, 0.70710678f };
    auto admittedCrossedDirection = diffractionmaterial::admit (
        crossedDirectionDescription, diffractionError);
    auto crossedDirectionProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *crossedProgram);
    crossedDirectionProgram->bindingDigest = admittedCrossedDirection
        ? admittedCrossedDirection->structuralDigest() : std::string {};
    if (admittedLighting)
        crossedDirectionProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                crossedDirectionProgram->bindingDigest, *admittedLighting);
    if (admittedCrossedDirection)
        crossedDirectionProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedCrossedDirection, incident), lightingPath);
    auto crossedDirectionPreparation = backend.prepare (
        diffractionScene, crossedDirectionProgram);
    auto crossedDirectionFrame = backend.render (
        diffractionScene, crossedDirectionPreparation.resources, 96, 96, {});
    const auto crossedDirectionPixels = crossedDirectionFrame.rendered
        ? readRgba8 (gl, crossedDirectionFrame.frame)
        : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-crossed-direction", crossedDirectionPixels);

    auto crossedPeriodDescription = crossedDescription;
    crossedPeriodDescription.geometry.secondaryGrooveSpacingNanometres = 900.0f;
    auto admittedCrossedPeriod = diffractionmaterial::admit (
        crossedPeriodDescription, diffractionError);
    auto crossedPeriodProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *crossedProgram);
    crossedPeriodProgram->bindingDigest = admittedCrossedPeriod
        ? admittedCrossedPeriod->structuralDigest() : std::string {};
    if (admittedLighting)
        crossedPeriodProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                crossedPeriodProgram->bindingDigest, *admittedLighting);
    if (admittedCrossedPeriod)
        crossedPeriodProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedCrossedPeriod, incident), lightingPath);
    auto crossedPeriodPreparation = backend.prepare (
        diffractionScene, crossedPeriodProgram);
    auto crossedPeriodFrame = backend.render (
        diffractionScene, crossedPeriodPreparation.resources, 96, 96, {});
    const auto crossedPeriodPixels = crossedPeriodFrame.rendered
        ? readRgba8 (gl, crossedPeriodFrame.frame)
        : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-crossed-period", crossedPeriodPixels);

    auto primaryPeriodDescription = crossedDescription;
    primaryPeriodDescription.geometry.grooveSpacingNanometres = 1350.0f;
    auto admittedPrimaryPeriod = diffractionmaterial::admit (
        primaryPeriodDescription, diffractionError);
    auto primaryPeriodProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (*crossedProgram);
    primaryPeriodProgram->programIdentity
        = "opengl-fixture-diffraction-primary-period-program";
    primaryPeriodProgram->bindingDigest = admittedPrimaryPeriod
        ? admittedPrimaryPeriod->structuralDigest() : std::string {};
    if (admittedLighting)
        primaryPeriodProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                primaryPeriodProgram->bindingDigest, *admittedLighting);
    if (admittedPrimaryPeriod)
        primaryPeriodProgram->diffractionPaths[0]
            = diffractionmaterial::makeGpuLightingPath(
                diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                    *admittedPrimaryPeriod, incident), lightingPath);
    auto primaryPeriodPreparation = backend.prepare (diffractionScene, primaryPeriodProgram);
    auto primaryPeriodFrame = backend.render (
        diffractionScene, primaryPeriodPreparation.resources, 96, 96, {});
    const auto primaryPeriodPixels = primaryPeriodFrame.rendered
        ? readRgba8 (gl, primaryPeriodFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-primary-period", primaryPeriodPixels);

    const auto overBudgetCrossedFrame = backend.render (
        diffractionScene, crossedPreparation.resources, 3840, 2160, {});

    const auto backgroundOnly = [] (const std::vector<std::uint8_t>& pixels)
    {
        for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
            if (pixels[index] != 7u || pixels[index + 1] != 10u
                || pixels[index + 2] != 18u || pixels[index + 3] != 255u)
                return false;
        return true;
    };

    ok &= expect (admittedDiffraction.has_value()
                      && everyPathPerturbsPixels
                      && diffractionPreparation.prepared
                      && diffractionFrame.rendered && shiftedDiffractionFrame.rendered
                      && rotatedPreparation.prepared && rotatedFrame.rendered
                      && admittedSpatial.has_value()
                      && spatialPreparation.prepared && spatialFrame.rendered
                      && admittedCrossed.has_value()
                      && crossedPreparation.prepared && crossedFrame.rendered
                      && admittedCrossedDirection.has_value()
                      && crossedDirectionPreparation.prepared
                      && crossedDirectionFrame.rendered
                      && admittedCrossedPeriod.has_value()
                      && crossedPeriodPreparation.prepared
                      && crossedPeriodFrame.rendered
                      && admittedPrimaryPeriod.has_value()
                      && primaryPeriodPreparation.prepared
                      && primaryPeriodFrame.rendered
                      && diffractionPixels.size() == 96u * 96u * 4u
                      && ! backgroundOnly (diffractionPixels)
                      && diffractionPixels != shiftedDiffractionPixels
                      && diffractionPixels != rotatedPixels
                      && spatialPixels != diffractionPixels
                      && spatialPixels != rotatedPixels

                      && ! backgroundOnly (crossedPixels)
                      && crossedPixels != diffractionPixels
                      && crossedDirectionPixels != crossedPixels
                      && primaryPeriodPixels != crossedPixels
                      && crossedPeriodPixels != crossedPixels,
                  "physical OpenGL diffraction pixels must execute every admitted lighting path and depend on camera, local UV groove fields, crossed-lattice orientation, both periods, and topology");
    ok &= expect (! overBudgetCrossedFrame.rendered
                      && overBudgetCrossedFrame.frame == nullptr
                      && overBudgetCrossedFrame.error
                           == "OpenGL fixture diffraction workload exceeds backend limits",
                  "OpenGL rejects over-budget crossed diffraction before frame allocation");
    ok &= expect (! malformedDiffractionPreparation.prepared
                      && malformedDiffractionPreparation.resources == nullptr
                      && malformedDiffractionPreparation.stats.vertexBytes == 0
                      && malformedDiffractionPreparation.stats.indexBytes == 0
                      && malformedDiffractionPreparation.stats.materialBytes == 0
                      && malformedDiffractionPreparation.stats.textureBytes == 0
                      && malformedDiffractionPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects malformed diffraction GPU parameters before resource allocation");
    ok &= expect (! replayedLightingPreparation.prepared
                      && replayedLightingPreparation.resources == nullptr
                      && replayedLightingPreparation.stats.vertexBytes == 0
                      && replayedLightingPreparation.stats.indexBytes == 0
                      && replayedLightingPreparation.stats.materialBytes == 0
                      && replayedLightingPreparation.stats.textureBytes == 0
                      && replayedLightingPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects a replayed spectral receipt before resource allocation");
    ok &= expect (! layoutSentinelPreparation.prepared
                      && layoutSentinelPreparation.resources == nullptr
                      && layoutSentinelPreparation.stats.vertexBytes == 0
                      && layoutSentinelPreparation.stats.indexBytes == 0
                      && layoutSentinelPreparation.stats.materialBytes == 0
                      && layoutSentinelPreparation.stats.textureBytes == 0
                      && layoutSentinelPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects reused layout-sentinel data before GPU allocation");
    ok &= expect (! divergentBlazePreparation.prepared
                      && divergentBlazePreparation.resources == nullptr
                      && divergentBlazePreparation.stats.vertexBytes == 0
                      && divergentBlazePreparation.stats.indexBytes == 0
                      && divergentBlazePreparation.stats.materialBytes == 0
                      && divergentBlazePreparation.stats.textureBytes == 0
                      && divergentBlazePreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects local period/blaze divergence before GPU allocation");
    ok &= expect (! unequalCrossedBlazedPreparation.prepared
                      && unequalCrossedBlazedPreparation.resources == nullptr
                      && unequalCrossedBlazedPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL preflight rejects unequal crossed blazed periods before GPU allocation");
    ok &= expect (diffractionFrame.rendered,
                  "OpenGL preparation owns an immutable material snapshot after caller alias mutation");

    auto* alternateContext = glfwCreateWindow (
        16, 16, "fixture-scene-alternate-context", nullptr, nullptr);
    ok &= expect (alternateContext != nullptr,
                  "a second context must be available for cleanup ownership coverage");
    if (alternateContext != nullptr)
        glfwMakeContextCurrent (alternateContext);

    first.frame.reset();
    second.frame.reset();
    preparation.resources.reset();
    clipped.frame.reset();
    clippedPreparation.resources.reset();
    textured.frame.reset();
    texturedPreparation.resources.reset();
    diffractionFrame.frame.reset();
    shiftedDiffractionFrame.frame.reset();
    diffractionPreparation.resources.reset();
    rotatedFrame.frame.reset();
    rotatedPreparation.resources.reset();
    spatialFrame.frame.reset();
    spatialPreparation.resources.reset();
    crossedFrame.frame.reset();
    crossedPreparation.resources.reset();
    crossedDirectionFrame.frame.reset();
    crossedDirectionPreparation.resources.reset();
    crossedPeriodFrame.frame.reset();
    crossedPeriodPreparation.resources.reset();
    ok &= expect (alternateContext == nullptr || glfwGetCurrentContext() == alternateContext,
                  "fixture resource cleanup must restore the caller's OpenGL context");

    glfwMakeContextCurrent (window);
    gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
    gl.DeleteFramebuffers (1, &callerFramebuffer);
    glDeleteTextures (1, &callerTexture);
    gl.DeleteSamplers (1, &callerSampler);
    if (alternateContext != nullptr)
        glfwDestroyWindow (alternateContext);
    glfwDestroyWindow (window);
    glfwTerminate();

    if (! ok)
        return 1;
    std::cout << "OpenGL fixture-scene backend: PASS; device="
              << info.device << "; rgba8-fnv1a64=" << std::hex
              << fnv1a64 (firstPixels) << '\n';
    return 0;
}
