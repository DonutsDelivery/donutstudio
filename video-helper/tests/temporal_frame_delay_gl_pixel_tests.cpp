#include "gl_loader.h"
#include "renderer.h"
#include "visual_plan_executor.h"
#include <GLFW/glfw3.h>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{
using Pixel = std::array<uint8_t, 4>;
constexpr int kTolerance = 2;

bool closeEnough(const std::vector<uint8_t>& actual, const Pixel& expected)
{
    if (actual.size() != 4) return false;
    for (size_t channel = 0; channel < expected.size(); ++channel)
        if (std::abs(static_cast<int>(actual[channel])
                     - static_cast<int>(expected[channel])) > kTolerance)
            return false;
    return true;
}

bool render(videorender::FrameRenderer& renderer, videorender::LayerDesc& layer,
            unsigned texture, const Pixel& input, bool reset, bool hold,
            const Pixel& expected, std::string& error, bool publish = true)
{
    static std::uint64_t generation = 1;
    auto candidate = renderer.beginFramePublicationCandidate(
        1, generation++, {}, error);
    if (! candidate) return false;
    if (renderer.uploadRgba(input.data(), 1, 1, 4, texture) == 0)
        return false;
    layer.feedbackHistoryReset = reset;
    layer.feedbackHistoryHold = hold;
    std::vector<uint8_t> pixels;
    const bool rendered = renderer.renderToPixels(&layer, 1, pixels, error);
    const bool published = publish
        ? renderer.commitFramePublicationCandidate(candidate, error)
        : (renderer.discardFramePublicationCandidate(candidate), true);
    const bool ok = rendered && published
        && closeEnough(pixels, expected);
    if (! ok)
    {
        std::cerr << "input=" << static_cast<int>(input[0]) << ','
                  << static_cast<int>(input[1]) << ',' << static_cast<int>(input[2])
                  << " expected=" << static_cast<int>(expected[0]) << ','
                  << static_cast<int>(expected[1]) << ',' << static_cast<int>(expected[2]);
        if (pixels.size() == 4)
            std::cerr << " actual=" << static_cast<int>(pixels[0]) << ','
                      << static_cast<int>(pixels[1]) << ',' << static_cast<int>(pixels[2]);
        std::cerr << " error=" << error << '\n';
    }
    return ok;
}
}

int main()
{
#if !defined(__linux__)
    return 77;
#else
    glfwSetErrorCallback([](int code, const char* text) {
        std::cerr << "GLFW " << code << ": " << (text != nullptr ? text : "unknown") << '\n';
    });
    if (glfwInit() != GLFW_TRUE) return 77;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(16, 16, "Frame Delay pixel acceptance", nullptr, nullptr);
    if (window == nullptr)
    {
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent(window);

    arbitgl::GlFuncs gl;
    std::string error;
    if (! arbitgl::loadGlFunctions(gl, error))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 77;
    }

    videorender::FrameRenderer renderer;
    if (! renderer.initialize(&gl, 1, 1, error))
    {
        std::cerr << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    renderer.setBackgroundColor(0.0f, 0.0f, 0.0f, 1.0f);

    const Pixel black { 0, 0, 0, 255 };
    const Pixel red { 255, 0, 0, 255 };
    const Pixel green { 0, 255, 0, 255 };
    const Pixel blue { 0, 0, 255, 255 };
    const Pixel yellow { 255, 255, 0, 255 };
    const Pixel magenta { 255, 0, 255, 255 };
    const Pixel cyan { 0, 255, 255, 255 };
    const Pixel white { 255, 255, 255, 255 };

    unsigned texture = renderer.uploadRgba(red.data(), 1, 1, 4, 0);
    videorender::LayerDesc layer;
    layer.clipId = 7;
    layer.texture = texture;
    layer.texWidth = 1;
    layer.texHeight = 1;
    layer.graphTemporalActive = true;
    layer.graphTemporalNodeId = 30;
    layer.visualPlanStructuralRevision = 11;
    layer.graphTemporalPayload.mode = visualtemporaloperation::Mode::frameDelay;
    layer.graphTemporalPayload.historyLength = 2;
    layer.graphTemporalPayload.holdFrames = 1;
    layer.graphTemporalPayload.mix = 1.0f;
    layer.graphTemporalPayload.decay = 0.0f;
    layer.graphTemporalPayload.zoom = 1.0f;
    layer.graphTemporalPayload.swirl = 0.0f;

    const videotemporal::ResourceContract contract(
        videotemporal::Mode::frameDelay, videotemporal::PixelFormat::rgba16f,
        { 1, 1 }, 2);
    videotemporal::ResourceLimits limits;
    videotemporal::ResourceFootprint footprint;
    if (! videotemporal::admitResource(contract, limits, false, footprint, error))
        return 1;
    videowire::TemporalFeedbackPass pass;
    pass.clipId = layer.clipId;
    pass.nodeId = layer.graphTemporalNodeId;
    pass.structuralRevision = layer.visualPlanStructuralRevision;
    pass.payload = layer.graphTemporalPayload;
    pass.extent = { 1, 1 };
    pass.format = videotemporal::PixelFormat::rgba16f;
    pass.historyLength = 2;
    pass.footprint = footprint;

    bool ok = renderer.prepareTemporalFeedbackPass(pass, error)
        && footprint.retainedImages == 2 && footprint.transientImages == 1
        && footprint.historyBytes == 16 && footprint.transientBytes == 8;
    if (! ok) std::cerr << "admission failed: " << error << '\n';
    auto rejected = pass;
    rejected.footprint.transientBytes = 0;
    const bool budgetRejected = ! renderer.prepareTemporalFeedbackPass(rejected, error);
    if (! budgetRejected)
        std::cerr << "zero-byte transient budget was accepted\n";
    ok = ok && budgetRejected;
    const bool readmitted = renderer.prepareTemporalFeedbackPass(pass, error);
    if (! readmitted) std::cerr << "readmission failed: " << error << '\n';
    ok = ok && readmitted;

    ok = ok && render(renderer, layer, texture, red, true, false, black, error);
    ok = ok && render(renderer, layer, texture, green, false, false, black, error);
    ok = ok && render(renderer, layer, texture, blue, false, false, red, error);
    ok = ok && render(renderer, layer, texture, yellow, false, false, green, error);
    ok = ok && render(renderer, layer, texture, white, false, true, green, error);
    ok = ok && render(renderer, layer, texture, magenta, true, false, black, error, false);
    ok = ok && render(renderer, layer, texture, cyan, false, false, blue, error);

    // Explicit reset covers loop discontinuity. Changing the structural revision
    // proves the renderer also rejects stale history even without the reset bit.
    ok = ok && render(renderer, layer, texture, magenta, true, false, black, error);
    ok = ok && render(renderer, layer, texture, cyan, false, false, black, error);
    ok = ok && render(renderer, layer, texture, white, false, false, magenta, error);
    layer.visualPlanStructuralRevision = 12;
    ok = ok && render(renderer, layer, texture, red, false, false, black, error);

    // History ownership is bounded across removed clips. Filling the admitted
    // owner ceiling evicts the least-recently-used stale owner rather than
    // retaining its textures for the renderer lifetime.
    layer.visualPlanStructuralRevision = 13;
    layer.clipId = 1000;
    ok = ok && render(renderer, layer, texture, red, true, false, black, error);
    ok = ok && render(renderer, layer, texture, green, false, false, black, error);
    for (int clipId = 1001; ok && clipId <= 1064; ++clipId)
    {
        layer.clipId = clipId;
        ok = ok && render(renderer, layer, texture, black, true, false, black, error);
    }
    layer.clipId = 1000;
    ok = ok && render(renderer, layer, texture, blue, false, false, black, error);

    auto competingA = renderer.beginFramePublicationCandidate(55, 1, {}, error);
    auto competingB = renderer.beginFramePublicationCandidate(55, 1, {}, error);
    const auto generationBeforeCompetition = renderer.publishedFrameGeneration();
    const bool competitionRejected = competingA && competingB
        && ! renderer.commitFramePublicationCandidate(competingA, error)
        && renderer.publishedFrameGeneration() == generationBeforeCompetition;
    renderer.discardFramePublicationCandidate(competingA);
    renderer.discardFramePublicationCandidate(competingB);
    ok = ok && competitionRejected;

    auto teardownCandidate = renderer.beginFramePublicationCandidate(77, 1, {}, error);
    ok = ok && teardownCandidate && renderer.liveFramePublicationCandidates() != 0;

    videorender::FrameRenderer metalOnly;
    std::string metalError;
    const bool metalInitialized = metalOnly.initialize(&gl, 1, 1, metalError, true);
    const bool metalRejected = ! metalInitialized
        || (! metalOnly.prepareTemporalFeedbackPass(pass, metalError)
            && metalError == "native OpenGL temporal resource backend is unavailable");
    metalOnly.shutdown();
    ok = ok && metalRejected;

    if (texture != 0) glDeleteTextures(1, &texture);
    renderer.shutdown();
    teardownCandidate.reset();
    ok = ok && renderer.liveFramePublicationCandidates() == 0;
    glfwDestroyWindow(window);
    glfwTerminate();
    if (! ok) return 1;
    std::cout << "visual.frame-delay OpenGL PASS\n";
    return 0;
#endif
}
