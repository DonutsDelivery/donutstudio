#include "../src/sdf_native_renderer.h"
#include "../src/gl_loader.h"
#include "support/sdf_native_backend_parity.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
std::shared_ptr<const videohelper::sdf::AdmittedSdfIr> admittedPrimitive (
    videowire::SdfOperation operation, videowire::SdfStableId stableId,
    std::initializer_list<double> parameters, const char* name,
    std::string& digest)
{
    videowire::SdfIr source;
    source.rootId = stableId;
    videowire::SdfRecord primitive;
    primitive.stableId = source.rootId;
    primitive.operation = operation;
    primitive.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (parameters.begin(), parameters.end(), primitive.parameters.begin());
    source.records.push_back (primitive);

    std::string error;
    auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    if (! admitted)
    {
        std::fprintf (stderr, "%s IR admission failed: %s\n", name, error.c_str());
        return {};
    }
    digest = admitted->structuralDigest();
    return std::make_shared<const videohelper::sdf::AdmittedSdfIr> (std::move (*admitted));
}

std::shared_ptr<const videohelper::sdf::AdmittedSdfIr> admittedSphere (
    double radius, std::string& digest)
{
    return admittedPrimitive (videowire::SdfOperation::sphere, 100,
                              { radius }, "sphere", digest);
}

std::shared_ptr<const videohelper::sdf::AdmittedSdfIr> admittedBox (
    double x, double y, double z, std::string& digest)
{
    return admittedPrimitive (videowire::SdfOperation::box, 101,
                              { x, y, z }, "box", digest);
}

videowire::SdfRecord record (videowire::SdfStableId id,
                             videowire::SdfOperation operation,
                             std::initializer_list<videowire::SdfStableId> inputs,
                             std::initializer_list<double> parameters)
{
    videowire::SdfRecord result;
    result.stableId = id;
    result.operation = operation;
    result.inputCount = static_cast<std::uint8_t> (inputs.size());
    result.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), result.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), result.parameters.begin());
    return result;
}

std::shared_ptr<const videohelper::sdf::AdmittedSdfIr> admittedComposite()
{
    videowire::SdfIr source;
    source.rootId = 5;
    source.records = {
        record (1, videowire::SdfOperation::sphere, {}, { 0.55 }),
        record (2, videowire::SdfOperation::translate, { 1 }, { -0.7, 0.0, 0.0 }),
        record (3, videowire::SdfOperation::box, {}, { 0.38, 0.5, 0.45 }),
        record (4, videowire::SdfOperation::translate, { 3 }, { 0.7, 0.0, 0.0 }),
        record (5, videowire::SdfOperation::smoothUnion, { 2, 4 }, { 0.18 })
    };
    std::string error;
    auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    if (! admitted)
    {
        std::fprintf (stderr, "composite IR admission failed: %s\n", error.c_str());
        return {};
    }
    return std::make_shared<const videohelper::sdf::AdmittedSdfIr> (std::move (*admitted));
}

std::vector<std::uint8_t> readPixels (const arbitgpu::NativeSdfSceneFrame& frame,
                                      const arbitgl::GlFuncs& gl)
{
    unsigned framebuffer = 0;
    gl.GenFramebuffers (1, &framebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                             static_cast<unsigned> (frame.colorImageHandle()), 0);
    std::vector<std::uint8_t> pixels (
        static_cast<std::size_t> (frame.width()) * frame.height() * 4u);
    glReadPixels (0, 0, static_cast<int> (frame.width()), static_cast<int> (frame.height()),
                  GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &framebuffer);

    return pixels;
}

std::array<std::uint8_t, 4> pixel (
    const arbitgpu::NativeSdfSceneFrame& frame, int x, int y,
    const arbitgl::GlFuncs& gl, std::uint32_t& checksum)
{
    const auto pixels = readPixels (frame, gl);

    checksum = 2166136261u;
    for (const auto value : pixels)
    {
        checksum ^= value;
        checksum *= 16777619u;
    }
    const auto offset = (static_cast<std::size_t> (y) * frame.width()
                         + static_cast<std::size_t> (x)) * 4u;
    return { pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3] };
}
} // namespace

int main()
{
    if (glfwInit() != GLFW_TRUE) return 77;
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    auto* window = glfwCreateWindow (96, 64, "native OpenGL SDF", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::fprintf (stderr, "OpenGL loader failed: %s\n", missing.c_str());
        return 1;
    }

    renderpassoutput::Description outputDescription;
    outputDescription.extent = { 64, 32 };
    for (const auto output : { renderpassoutput::Output::Color,
                               renderpassoutput::Output::Depth })
    {
        const auto required = renderpassoutput::requirements (output);
        outputDescription.attachments.push_back (
            { output, required.format, required.colorSpace, outputDescription.extent });
    }
    renderpassoutput::AdmissionFailure outputFailure = renderpassoutput::AdmissionFailure::None;
    auto admittedOutputs = renderpassoutput::admit (outputDescription, outputFailure);
    auto& outputBackend = arbitgpu::nativeRenderPassOutputBackend();
    const auto outputCapabilities = outputBackend.renderPassOutputCapabilities();
    auto outputAdmission = admittedOutputs.has_value()
        ? outputBackend.admitRenderPassOutputs (*admittedOutputs)
        : arbitgpu::RenderPassOutputAdmission {};
    if (! outputCapabilities.available || ! outputAdmission.lifecycle
        || outputAdmission.resources.size() != 2
        || outputAdmission.frameMemory.failure
               != arbitgpu::FrameMemoryAdmissionFailure::none
        || outputAdmission.frameMemory.requestedBytes != 64ull * 32ull * 12ull
        || outputAdmission.frameMemory.allocatedSlots != 2)
    {
        std::fprintf (stderr, "OpenGL frame-memory admission failed: %s\n",
                      outputAdmission.error.c_str());
        return 20;
    }
    outputBackend.releaseRenderPassOutputs (outputAdmission.lifecycle);

    const auto capabilities = arbitgpu::queryNativeSdfExecution();
    if (! capabilities.available || capabilities.backend != "opengl"
        || ! capabilities.supports (videowire::SdfOperation::sphere)
        || ! capabilities.supports (videowire::SdfOperation::box)
        || ! capabilities.supports (videowire::SdfOperation::roundedBox)
        || ! capabilities.supports (videowire::SdfOperation::plane)
        || ! capabilities.supports (videowire::SdfOperation::torus)
        || ! capabilities.supports (videowire::SdfOperation::capsule)
        || ! capabilities.supports (videowire::SdfOperation::cylinder)
        || ! capabilities.supports (videowire::SdfOperation::cone)
        || ! capabilities.supports (videowire::SdfOperation::gyroid)
        || ! capabilities.supports (videowire::SdfOperation::translate)
        || ! capabilities.supports (videowire::SdfOperation::smoothUnion)
        || ! capabilities.supports (arbitgpu::NativeSdfOutput::color)
        || ! capabilities.supports (arbitgpu::NativeSdfOutput::depth)
        || ! capabilities.supports (arbitgpu::NativeSdfOutput::normal)
        || capabilities.maxOperations != arbitgpu::kNativeSdfMaximumRecords
        || capabilities.maxDepth != arbitgpu::kNativeSdfMaximumDepth)
    {
        std::fprintf (stderr, "strict OpenGL SDF capabilities failed: %s\n",
                      capabilities.error.c_str());
        return 2;
    }

    std::string digest;
    const auto geometry = admittedSphere (1.0, digest);
    if (geometry == nullptr) return 3;

    videohelper::sdf::NativeSdfRenderControls controls;
    controls.maximumSteps = 128;
    controls.epsilon = 0.001;
    controls.maximumDistance = 100.0;
    auto& renderer = videohelper::sdf::nativeSdfRenderer();
    videohelper::sdf::NativeSdfRenderedFrame rendered;
    std::string error;
    glEnable (GL_BLEND);
    glEnable (GL_DEPTH_TEST);
    glEnable (GL_CULL_FACE);
    glEnable (GL_SCISSOR_TEST);
    glEnable (GL_RASTERIZER_DISCARD);
    glEnable (GL_COLOR_LOGIC_OP);
    glEnable (GL_FRAMEBUFFER_SRGB);
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glColorMask (GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
    if (! renderer.renderPreview (geometry, { 96, 64 }, controls,
                                  videohelper::sdf::kNativeGpuCapability,
                                  rendered, error))
    {
        std::fprintf (stderr, "native renderer API failed: %s\n", error.c_str());
        return 4;
    }
    const auto& frameMemory = rendered.nativeFrame->frameMemoryAdmission();
    if (frameMemory.requestedBytes != 96ull * 64ull * 8ull
        || frameMemory.allocatedSlots != 1
        || frameMemory.failure != arbitgpu::FrameMemoryAdmissionFailure::none)
    {
        std::fprintf (stderr, "native SDF frame-memory receipt mismatch bytes=%llu slots=%llu\n",
            static_cast<unsigned long long> (frameMemory.requestedBytes),
            static_cast<unsigned long long> (frameMemory.allocatedSlots));
        return 4;
    }
    const auto& receipt = rendered.nativeFrame->sdfResourceReceipt();
    if (receipt.compiledRecordCount != 1
        || receipt.compiledRecordBytes != sizeof (arbitgpu::NativeSdfCompiledRecord)
        || receipt.geometryCacheBytes < receipt.compiledRecordBytes
        || receipt.backendProgramBytes == 0 || receipt.uniformBytes == 0
        || receipt.attachmentBytes != frameMemory.requestedBytes
        || receipt.totalBytes != receipt.compiledRecordBytes + receipt.geometryCacheBytes
            + receipt.backendProgramBytes + receipt.uniformBytes + receipt.attachmentBytes)
    {
        std::fprintf (stderr, "OpenGL native SDF resource receipt mismatch\n");
        return 4;
    }

    int polygonMode = 0;
    unsigned char colorMask[4] = {};
    glGetIntegerv (GL_POLYGON_MODE, &polygonMode);
    glGetBooleanv (GL_COLOR_WRITEMASK, colorMask);
    if (glIsEnabled (GL_BLEND) != GL_TRUE
        || glIsEnabled (GL_DEPTH_TEST) != GL_TRUE
        || glIsEnabled (GL_CULL_FACE) != GL_TRUE
        || glIsEnabled (GL_SCISSOR_TEST) != GL_TRUE
        || glIsEnabled (GL_RASTERIZER_DISCARD) != GL_TRUE
        || glIsEnabled (GL_COLOR_LOGIC_OP) != GL_TRUE
        || glIsEnabled (GL_FRAMEBUFFER_SRGB) != GL_TRUE
        || polygonMode != GL_LINE
        || colorMask[0] != GL_FALSE || colorMask[1] != GL_TRUE
        || colorMask[2] != GL_FALSE || colorMask[3] != GL_TRUE)
    {
        std::fprintf (stderr,
            "OpenGL SDF state mismatch blend=%d depth=%d cull=%d scissor=%d discard=%d "
            "logic=%d srgb=%d polygon=%d mask=%u,%u,%u,%u\n",
            glIsEnabled (GL_BLEND), glIsEnabled (GL_DEPTH_TEST),
            glIsEnabled (GL_CULL_FACE), glIsEnabled (GL_SCISSOR_TEST),
            glIsEnabled (GL_RASTERIZER_DISCARD), glIsEnabled (GL_COLOR_LOGIC_OP),
            glIsEnabled (GL_FRAMEBUFFER_SRGB), polygonMode,
            colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        return 5;
    }
    glDisable (GL_CULL_FACE);
    glDisable (GL_SCISSOR_TEST);
    glDisable (GL_RASTERIZER_DISCARD);
    glDisable (GL_COLOR_LOGIC_OP);
    glDisable (GL_FRAMEBUFFER_SRGB);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    std::uint32_t pixelHash = 0;
    const auto center = pixel (*rendered.nativeFrame, 48, 32, gl, pixelHash);
    const auto background = pixel (*rendered.nativeFrame, 0, 0, gl, pixelHash);
    if (background != std::array<std::uint8_t, 4> { 7, 10, 18, 255 }
        || center == background || center[3] != 255)
    {
        std::fprintf (stderr,
            "OpenGL SDF pixel mismatch center=%u,%u,%u,%u background=%u,%u,%u,%u hash=%u\n",
            center[0], center[1], center[2], center[3],
            background[0], background[1], background[2], background[3],
            pixelHash);
        return 5;
    }

    std::string boxDigest;
    const auto boxGeometry = admittedBox (0.75, 0.5, 0.25, boxDigest);
    if (boxGeometry == nullptr) return 15;
    videohelper::sdf::NativeSdfRenderedFrame boxRendered;
    if (! renderer.renderPreview (boxGeometry, { 96, 64 }, controls,
                                  videohelper::sdf::kNativeGpuCapability,
                                  boxRendered, error))
    {
        std::fprintf (stderr, "native SDF box output failed: %s\n", error.c_str());
        return 16;
    }
    std::uint32_t boxHash = 0;
    const auto boxCenter = pixel (*boxRendered.nativeFrame, 48, 32, gl, boxHash);
    const auto boxBackground = pixel (*boxRendered.nativeFrame, 0, 0, gl, boxHash);
    if (boxBackground != background || boxCenter == boxBackground
        || boxCenter[3] != 255 || boxHash == pixelHash)
    {
        std::fprintf (stderr,
            "OpenGL SDF box mismatch center=%u,%u,%u,%u hash=%u sphereHash=%u\n",
            boxCenter[0], boxCenter[1], boxCenter[2], boxCenter[3], boxHash, pixelHash);
        return 17;
    }
    boxRendered = {};

    const auto compositeGeometry = admittedComposite();
    videohelper::sdf::NativeSdfRenderedFrame compositeRendered;
    if (compositeGeometry == nullptr
        || ! renderer.renderPreview (compositeGeometry, { 96, 64 }, controls,
                                     videohelper::sdf::kNativeGpuCapability,
                                     compositeRendered, error))
    {
        std::fprintf (stderr, "native SDF composite output failed: %s\n", error.c_str());
        return 21;
    }
    std::uint32_t compositeHash = 0;
    const auto compositeLeft = pixel (*compositeRendered.nativeFrame, 34, 32, gl, compositeHash);
    const auto compositeRight = pixel (*compositeRendered.nativeFrame, 62, 32, gl, compositeHash);
    if (compositeHash == pixelHash || compositeLeft == background || compositeRight == background)
    {
        std::fprintf (stderr,
            "OpenGL transformed composite mismatch left=%u,%u,%u right=%u,%u,%u hash=%u\n",
            compositeLeft[0], compositeLeft[1], compositeLeft[2],
            compositeRight[0], compositeRight[1], compositeRight[2], compositeHash);
        return 22;
    }

    arbitgpu::NativeSdfDrawRequest malformed;
    malformed.geometry.rootId = 2;
    malformed.geometry.records = {
        record (1, videowire::SdfOperation::translate, { 2 }, { 0.0, 0.0, 0.0 }),
        record (2, videowire::SdfOperation::translate, { 1 }, { 0.0, 0.0, 0.0 })
    };
    malformed.width = 96;
    malformed.height = 64;
    malformed.maximumSteps = 128;
    malformed.epsilon = 0.001;
    malformed.maximumDistance = 100.0;
    const auto rejectedCycle = arbitgpu::nativeSdfExecutionBackend().render (malformed);
    malformed.geometry.records[0].inputs[0] = 999;
    const auto rejectedReference = arbitgpu::nativeSdfExecutionBackend().render (malformed);

    auto overDepth = malformed;
    overDepth.geometry = {};
    overDepth.geometry.rootId = 34;
    overDepth.geometry.records.push_back (
        record (1, videowire::SdfOperation::sphere, {}, { 0.5 }));
    for (videowire::SdfStableId id = 2; id <= 34; ++id)
        overDepth.geometry.records.push_back (
            record (id, videowire::SdfOperation::translate, { id - 1 }, { 0.0, 0.0, 0.0 }));
    const auto rejectedDepth = arbitgpu::nativeSdfExecutionBackend().render (overDepth);

    auto oversized = malformed;
    oversized.geometry = {};
    oversized.geometry.rootId = 1;
    for (videowire::SdfStableId id = 1; id <= 65; ++id)
        oversized.geometry.records.push_back (
            record (id, videowire::SdfOperation::sphere, {}, { 0.5 }));
    const auto rejectedCount = arbitgpu::nativeSdfExecutionBackend().render (oversized);

    auto nonFinite = malformed;
    nonFinite.geometry = {};
    nonFinite.geometry.rootId = 1;
    nonFinite.geometry.records.push_back (record (
        1, videowire::SdfOperation::sphere, {},
        { std::numeric_limits<double>::quiet_NaN() }));
    const auto rejectedNonFinite = arbitgpu::nativeSdfExecutionBackend().render (nonFinite);
    if (rejectedCycle.rendered || rejectedCycle.frame != nullptr
        || rejectedReference.rendered || rejectedReference.frame != nullptr
        || rejectedDepth.rendered || rejectedDepth.frame != nullptr
        || rejectedCount.rendered || rejectedCount.frame != nullptr
        || rejectedNonFinite.rendered || rejectedNonFinite.frame != nullptr)
    {
        std::fprintf (stderr, "OpenGL malformed SDF request allocated a frame\n");
        return 23;
    }

    const auto verifyPrimitive = [&] (videowire::SdfOperation operation,
                                      videowire::SdfStableId stableId,
                                      std::initializer_list<double> parameters,
                                      const char* name)
    {
        std::string primitiveDigest;
        const auto primitive = admittedPrimitive (
            operation, stableId, parameters, name, primitiveDigest);
        videohelper::sdf::NativeSdfRenderedFrame primitiveFrame;
        std::string primitiveError;
        if (primitive == nullptr
            || ! renderer.renderPreview (primitive, { 96, 64 }, controls,
                                         videohelper::sdf::kNativeGpuCapability,
                                         primitiveFrame, primitiveError)
            || primitiveFrame.nativeFrame == nullptr
            || primitiveFrame.nativeFrame->colorImageHandle() == 0)
        {
            std::fprintf (stderr, "native SDF %s output failed: %s\n",
                          name, primitiveError.c_str());
            return false;
        }
        return true;
    };
    if (! verifyPrimitive (videowire::SdfOperation::roundedBox, 102,
                           { 0.75, 0.5, 0.25, 0.1 }, "rounded box")
        || ! verifyPrimitive (videowire::SdfOperation::plane, 103,
                              { 0.0, 1.0, 0.0, 0.5 }, "plane")
        || ! verifyPrimitive (videowire::SdfOperation::torus, 104,
                              { 0.7, 0.2 }, "torus")
        || ! verifyPrimitive (videowire::SdfOperation::capsule, 105,
                              { 0.0, -0.7, 0.0, 0.0, 0.7, 0.0, 0.2 }, "capsule")
        || ! verifyPrimitive (videowire::SdfOperation::cylinder, 106,
                              { 0.6, 0.8 }, "cylinder")
        || ! verifyPrimitive (videowire::SdfOperation::cone, 107,
                              { 0.7, 0.9 }, "cone")
        || ! verifyPrimitive (videowire::SdfOperation::gyroid, 108,
                              { 2.0, 0.1 }, "gyroid"))
        return 18;

    auto parityControls = controls;
    parityControls.output = arbitgpu::NativeSdfOutput::depth;
    auto parityFixtures = videohelper::sdf::test::nativeOperationFixtures (error);
    auto polarFixtures = videohelper::sdf::test::largeOffsetPolarRepeatFixtures (error);
    parityFixtures.insert (parityFixtures.end(),
                           std::make_move_iterator (polarFixtures.begin()),
                           std::make_move_iterator (polarFixtures.end()));
    if (parityFixtures.size() != 28)
    {
        std::fprintf (stderr, "OpenGL SDF parity fixtures failed: %s\n", error.c_str());
        return 24;
    }
    for (const auto& fixture : parityFixtures)
    {
        videohelper::sdf::NativeSdfRenderedFrame parityFrame;
        if (! renderer.renderPreview (fixture.geometry, { 17, 17 }, parityControls,
                                      videohelper::sdf::kNativeGpuCapability,
                                      parityFrame, error)
            || parityFrame.nativeFrame == nullptr
            || ! videohelper::sdf::test::verifyNativeDepthParity (
                   fixture, parityControls, readPixels (*parityFrame.nativeFrame, gl),
                   17, 17, error))
        {
            std::fprintf (stderr, "OpenGL SDF operation parity failed: %s\n", error.c_str());
            return 25;
        }
    }

    controls.output = arbitgpu::NativeSdfOutput::depth;
    videohelper::sdf::NativeSdfRenderedFrame depthRendered;
    if (! renderer.renderPreview (geometry, { 96, 64 }, controls,
                                  videohelper::sdf::kNativeGpuCapability,
                                  depthRendered, error))
    {
        std::fprintf (stderr, "native SDF depth output failed: %s\n", error.c_str());
        return 11;
    }
    std::uint32_t depthHash = 0;
    const auto depthCenter = pixel (*depthRendered.nativeFrame, 48, 32, gl, depthHash);
    const auto depthBackground = pixel (*depthRendered.nativeFrame, 0, 0, gl, depthHash);
    if (depthCenter[0] < 4 || depthCenter[0] > 16
        || depthCenter[0] != depthCenter[1] || depthCenter[1] != depthCenter[2]
        || depthCenter[3] != 255
        || depthBackground != std::array<std::uint8_t, 4> { 255, 255, 255, 255 }
        || depthHash == pixelHash)
    {
        std::fprintf (stderr,
            "OpenGL SDF depth mismatch center=%u,%u,%u,%u hash=%u\n",
            depthCenter[0], depthCenter[1], depthCenter[2], depthCenter[3], depthHash);
        return 12;
    }

    controls.output = arbitgpu::NativeSdfOutput::normal;
    videohelper::sdf::NativeSdfRenderedFrame normalRendered;
    if (! renderer.renderPreview (geometry, { 96, 64 }, controls,
                                  videohelper::sdf::kNativeGpuCapability,
                                  normalRendered, error))
    {
        std::fprintf (stderr, "native SDF normal output failed: %s\n", error.c_str());
        return 13;
    }
    std::uint32_t normalHash = 0;
    const auto normalCenter = pixel (*normalRendered.nativeFrame, 48, 32, gl, normalHash);
    if (normalCenter[0] < 120 || normalCenter[0] > 136
        || normalCenter[1] < 120 || normalCenter[1] > 136
        || normalCenter[2] < 240 || normalCenter[3] != 255
        || normalHash == pixelHash)
    {
        std::fprintf (stderr,
            "OpenGL SDF normal mismatch center=%u,%u,%u,%u hash=%u\n",
            normalCenter[0], normalCenter[1], normalCenter[2], normalCenter[3], normalHash);
        return 14;
    }

    auto unsupported = videowire::SdfIr {};
    unsupported.rootId = 902;
    videowire::SdfRecord unionRecord;
    unionRecord.stableId = unsupported.rootId;
    unionRecord.operation = videowire::SdfOperation::unionOp;
    unionRecord.inputCount = 2;
    unionRecord.inputs = { 903, 904 };
    unsupported.records.push_back (unionRecord);
    auto direct = arbitgpu::NativeSdfDrawRequest {};
    direct.geometry = unsupported;
    direct.width = 96;
    direct.height = 64;
    direct.maximumSteps = 128;
    direct.epsilon = 0.001;
    direct.maximumDistance = 100.0;
    const auto rejected = arbitgpu::nativeSdfExecutionBackend().render (direct);
    if (rejected.rendered || rejected.frame != nullptr || rejected.error.empty())
    {
        std::fprintf (stderr, "missing-reference SDF did not fail closed\n");
        return 6;
    }

    direct.geometry = {};
    direct.geometry.rootId = 903;
    videowire::SdfRecord invalidSphere;
    invalidSphere.stableId = direct.geometry.rootId;
    invalidSphere.operation = videowire::SdfOperation::sphere;
    invalidSphere.parameterCount = 1;
    invalidSphere.parameters[0] = 0.0;
    direct.geometry.records.push_back (invalidSphere);
    const auto invalidParameter = arbitgpu::nativeSdfExecutionBackend().render (direct);
    direct.geometry.records[0].parameters[0] = 1.0;
    auto secondSphere = direct.geometry.records[0];
    secondSphere.stableId = 904;
    direct.geometry.records.push_back (secondSphere);
    const auto invalidTopology = arbitgpu::nativeSdfExecutionBackend().render (direct);
    if (invalidParameter.rendered || invalidParameter.frame != nullptr
        || invalidTopology.rendered || invalidTopology.frame != nullptr
        || invalidParameter.error.empty() || invalidTopology.error.empty())
    {
        std::fprintf (stderr, "unsupported SDF parameter or topology did not fail closed\n");
        return 7;
    }

    const auto texture = static_cast<unsigned> (rendered.nativeFrame->colorImageHandle());
    if (texture == 0 || glIsTexture (texture) != GL_TRUE)
    {
        std::fprintf (stderr, "OpenGL SDF frame did not own a live texture\n");
        return 8;
    }
    glfwMakeContextCurrent (nullptr);
    rendered = {};
    if (glfwGetCurrentContext() != nullptr)
    {
        std::fprintf (stderr, "OpenGL SDF frame deletion did not restore the caller context\n");
        return 9;
    }
    glfwMakeContextCurrent (window);
    if (glIsTexture (texture) != GL_FALSE)
    {
        std::fprintf (stderr, "OpenGL SDF frame leaked its texture\n");
        return 10;
    }
    compositeRendered = {};
    depthRendered = {};
    normalRendered = {};
    arbitgpu::invalidateNativeSdfExecutionContext (reinterpret_cast<std::uintptr_t> (window));
    glfwDestroyWindow (window);
    window = glfwCreateWindow (96, 64, "native OpenGL SDF replacement", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);
    videohelper::sdf::NativeSdfRenderedFrame replacementFrame;
    if (! renderer.renderPreview (geometry, { 96, 64 }, controls,
                                  videohelper::sdf::kNativeGpuCapability,
                                  replacementFrame, error)
        || replacementFrame.nativeFrame == nullptr)
    {
        std::fprintf (stderr, "OpenGL SDF context replacement reused a stale program: %s\n",
                      error.c_str());
        return 11;
    }
    replacementFrame = {};

    std::printf (
        "opengl-sdf PASS device=%s ir=root:901,ops:[sphere(1.0)] digest=%s "
        "center=%u,%u,%u,%u background=%u,%u,%u,%u hash=%u "
        "compositeLeft=%u,%u,%u compositeRight=%u,%u,%u compositeHash=%u texture_deleted=yes\n",
        capabilities.device.c_str(), digest.c_str(),
        center[0], center[1], center[2], center[3],
        background[0], background[1], background[2], background[3], pixelHash,
        compositeLeft[0], compositeLeft[1], compositeLeft[2],
        compositeRight[0], compositeRight[1], compositeRight[2], compositeHash);

    arbitgpu::invalidateNativeSdfExecutionContext (reinterpret_cast<std::uintptr_t> (window));
    glfwMakeContextCurrent (nullptr);
    glfwDestroyWindow (window);
    glfwTerminate();
    return 0;
}
