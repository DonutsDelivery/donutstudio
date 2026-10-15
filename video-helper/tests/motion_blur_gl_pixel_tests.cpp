#include "gl_loader.h"
#include "renderer.h"
#include "visual_plan_executor.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{
constexpr int kWidth = 9;
constexpr int kHeight = 1;
constexpr int kTolerance = 2;

renderpassoutput::Description motionDescription()
{
    renderpassoutput::Description description;
    description.extent = { kWidth, kHeight };
    const auto requirements = renderpassoutput::requirements(renderpassoutput::Output::Motion);
    description.attachments.push_back({ renderpassoutput::Output::Motion,
                                        requirements.format,
                                        requirements.colorSpace,
                                        description.extent });
    return description;
}

bool channelClose(std::uint8_t actual, int expected)
{
    return std::abs(static_cast<int>(actual) - expected) <= kTolerance;
}
}

int main()
{
#if !defined(__linux__)
    return 77;
#else
    glfwSetErrorCallback([](int code, const char* text) {
        std::cerr << "GLFW " << code << ": "
                  << (text != nullptr ? text : "unknown") << '\n';
    });
    if (glfwInit() != GLFW_TRUE) return 77;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(32, 16, "Motion blur pixel acceptance",
                                          nullptr, nullptr);
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
    if (! renderer.initialize(&gl, kWidth, kHeight, error))
    {
        std::cerr << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    renderer.setBackgroundColor(0.0f, 0.0f, 0.0f, 1.0f);

    std::array<std::uint8_t, kWidth * 4> source {};
    for (int x = 0; x < kWidth; ++x)
        source[static_cast<std::size_t>(x) * 4u + 3u] = 255;
    source[4u * 4u] = 255;

    const unsigned texture = renderer.uploadRgba(source.data(), kWidth, kHeight,
                                                  kWidth * 4, 0);
    if (texture == 0)
    {
        std::cerr << "source texture upload failed\n";
        renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const arbitgpu::RenderPassMotionAovClear motionClear { { 4.0f, 0.0f } };
    bool ok = renderer.replaceMotionAovPass(motionDescription(), motionClear, error);
    if (! ok)
        std::cerr << "Motion AOV publication failed: " << error << '\n';

    videorender::LayerDesc layer;
    layer.clipId = 7;
    layer.texture = texture;
    layer.texWidth = kWidth;
    layer.texHeight = kHeight;

    videowire::TemporalSamplingExecution execution;
    execution.pass.clipId = layer.clipId;
    execution.pass.nodeId = 30;
    execution.pass.structuralRevision = 11;
    execution.pass.payload.mode = visualtemporalsampling::Mode::motionBlur;
    execution.pass.payload.sampleCount = 5;
    execution.pass.payload.historyWeight = 0.0f;
    execution.pass.payload.jitterSpread = 0.0f;
    execution.pass.payload.shutterAngleDegrees = 360.0f;
    execution.pass.payload.motionScale = 1.0f;
    execution.pass.width = kWidth;
    execution.pass.height = kHeight;
    execution.point.ownerId = layer.clipId;
    execution.point.structuralRevision = execution.pass.structuralRevision;
    execution.point.width = kWidth;
    execution.point.height = kHeight;
    execution.point.helperGeneration = 1;
    execution.point.timeSec = 0.0;
    execution.transition.action = visualtemporalsampling::TransitionAction::resetAndAdvance;
    execution.transition.resetCause = visualtemporalsampling::ResetCause::firstEvaluation;

    ok = ok && renderer.prepareMotionBlurPass(execution, layer, error);
    if (! ok)
        std::cerr << "Motion blur admission failed: " << error << '\n';

    std::vector<std::uint8_t> pixels;
    ok = ok && renderer.renderToPixels(&layer, 1, pixels, error);
    if (! ok)
        std::cerr << "Motion blur render failed: " << error << '\n';

    if (pixels.size() != static_cast<std::size_t>(kWidth * 4))
    {
        std::cerr << "Unexpected output size: " << pixels.size() << '\n';
        ok = false;
    }
    else
    {
        for (int x = 0; x < kWidth; ++x)
        {
            const int expectedRed = x >= 2 && x <= 6 ? 51 : 0;
            const auto offset = static_cast<std::size_t>(x) * 4u;
            const bool pixelOk = channelClose(pixels[offset], expectedRed)
                && channelClose(pixels[offset + 1u], 0)
                && channelClose(pixels[offset + 2u], 0)
                && channelClose(pixels[offset + 3u], 255);
            if (! pixelOk)
            {
                std::cerr << "pixel " << x << " expected=" << expectedRed
                          << ",0,0,255 actual="
                          << static_cast<int>(pixels[offset]) << ','
                          << static_cast<int>(pixels[offset + 1u]) << ','
                          << static_cast<int>(pixels[offset + 2u]) << ','
                          << static_cast<int>(pixels[offset + 3u]) << '\n';
                ok = false;
            }
        }
    }

    auto stale = execution;
    stale.point.structuralRevision += 1;
    videorender::LayerDesc rejectedLayer = layer;
    rejectedLayer.graphMotionBlurActive = false;
    const bool staleRejected = ! renderer.prepareMotionBlurPass(stale, rejectedLayer, error)
        && error == "motion-blur temporal lifecycle does not match the admitted pass"
        && ! rejectedLayer.graphMotionBlurActive;
    if (! staleRejected)
    {
        std::cerr << "stale lifecycle was not rejected transactionally: " << error << '\n';
        ok = false;
    }

    renderer.deleteTexture(texture);
    renderer.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    if (! ok) return 1;
    std::cout << "visual.motion-blur OpenGL PASS\n";
    return 0;
#endif
}
