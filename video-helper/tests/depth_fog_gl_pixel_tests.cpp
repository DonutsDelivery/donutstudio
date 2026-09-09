#include "gl_loader.h"
#include "renderer.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{
constexpr int kWidth = 3;
constexpr int kHeight = 1;
constexpr int kTolerance = 3;
constexpr std::array<uint8_t, 4> kSource { 32, 96, 192, 255 };
constexpr std::array<uint8_t, 4> kFog { 224, 48, 16, 128 };
constexpr std::array<uint16_t, 3> kDepth { 0, 32768, 65535 };
constexpr double kNear = 0.0;
constexpr double kFar = 1.0;
constexpr double kDensity = 2.0;

int colorTransformOracle(uint8_t encoded)
{
    const double value = static_cast<double>(encoded) / 255.0;
    const double linear = value <= 0.04045 ? value / 12.92
        : std::pow((value + 0.055) / 1.055, 2.4);
    const double inputNits = linear * 100.0;
    const double shoulder = 1.0 / 100.0 - 1.0 / 1000.0;
    const double mapped = inputNits / (1.0 + shoulder * inputNits) / 100.0;
    const double output = mapped <= 0.0031308 ? mapped * 12.92
        : 1.055 * std::pow(mapped, 1.0 / 2.4) - 0.055;
    return static_cast<int>(std::lround(std::clamp(output, 0.0, 1.0) * 255.0));
}

std::array<int, 4> oracle(uint16_t rawDepth)
{
    // Independent specification oracle: normalized UNORM16 depth, Beer-Lambert
    // transmittance, then straight RGB interpolation. This intentionally does
    // not call any renderer/fog helper or reproduce shader source text.
    const double normalizedDepth = static_cast<double>(rawDepth) / 65535.0;
    const double progress = std::max(0.0, std::min(1.0,
        (normalizedDepth - kNear) / (kFar - kNear)));
    const double fogOpacity = (static_cast<double>(kFog[3]) / 255.0)
        * (1.0 - std::exp(-kDensity * progress));
    std::array<int, 4> expected {};
    for (int channel = 0; channel < 3; ++channel)
        expected[channel] = static_cast<int>(std::lround(
            kSource[channel] * (1.0 - fogOpacity) + kFog[channel] * fogOpacity));
    expected[3] = kSource[3];
    return expected;
}

bool closeEnough(const uint8_t* actual, const std::array<int, 4>& expected)
{
    for (int channel = 0; channel < 4; ++channel)
        if (std::abs(static_cast<int>(actual[channel]) - expected[channel]) > kTolerance)
            return false;
    return true;
}

void printPixel(const char* name, const uint8_t* actual, const std::array<int, 4>& expected)
{
    std::cout << name << " actual=(" << static_cast<int>(actual[0]) << ','
              << static_cast<int>(actual[1]) << ',' << static_cast<int>(actual[2]) << ','
              << static_cast<int>(actual[3]) << ") oracle=(" << expected[0] << ','
              << expected[1] << ',' << expected[2] << ',' << expected[3] << ")\n";
}
}

int main()
{
#if !defined(__linux__)
    std::cerr << "This acceptance target requires native Linux OpenGL\n";
    return 1;
#else
    glfwSetErrorCallback([](int code, const char* text) {
        std::cerr << "GLFW " << code << ": " << (text != nullptr ? text : "unknown") << '\n';
    });
    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "BLOCKED: glfwInit failed; no native EGL/GLX display/context available\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(16, 16, "Depth Fog pixel acceptance", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "BLOCKED: GLFW could not create a native Linux GL 4.3 offscreen context\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    arbitgl::GlFuncs gl;
    std::string error;
    if (!arbitgl::loadGlFunctions(gl, error))
    {
        std::cerr << "BLOCKED: production GL loader missing: " << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Force production FrameRenderer shader admission to fail, and prove its
    // owner unwinds to a non-ready state without leaving a GL error behind.
    auto rejectedGl = gl;
    rejectedGl.CreateShader = [](GLenum) -> GLuint { return 0; };
    videorender::FrameRenderer rejected;
    std::string rejection;
    const bool rejectedAdmission = !rejected.initialize(&rejectedGl, kWidth, kHeight, rejection)
        && !rejected.ready() && !rejection.empty();
    rejected.shutdown();
    // The deliberately invalid CreateShader handle can set GL_INVALID_VALUE;
    // consume that injected diagnostic before proving the same context remains
    // usable by a fresh production owner.
    while (glGetError() != GL_NO_ERROR) {}

    videorender::FrameRenderer renderer;
    if (!renderer.initialize(&gl, kWidth, kHeight, error))
    {
        std::cerr << "FrameRenderer initialization failed: " << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::array<uint8_t, kWidth * kHeight * 4> source {};
    for (int x = 0; x < kWidth; ++x)
        std::copy(kSource.begin(), kSource.end(), source.begin() + x * 4);
    const unsigned sourceTexture = renderer.uploadRgba(source.data(), kWidth, kHeight,
                                                        kWidth * 4, 0);
    const unsigned depthTexture = renderer.uploadR16(kDepth.data(), kWidth, kHeight, 0);

    videorender::LayerDesc layer;
    layer.texture = sourceTexture;
    layer.texWidth = kWidth;
    layer.texHeight = kHeight;
    layer.depthTexture = depthTexture;
    layer.depthWidth = kWidth;
    layer.depthHeight = kHeight;
    layer.depthFog = true;
    layer.fogNear = static_cast<float>(kNear);
    layer.fogFar = static_cast<float>(kFar);
    layer.fogDensity = static_cast<float>(kDensity);
    layer.fogRed = kFog[0] / 255.0f;
    layer.fogGreen = kFog[1] / 255.0f;
    layer.fogBlue = kFog[2] / 255.0f;
    layer.fogAlpha = kFog[3] / 255.0f;

    std::vector<uint8_t> pixels;
    const bool rendered = sourceTexture != 0 && depthTexture != 0
        && renderer.renderToPixels(&layer, 1, pixels, error)
        && pixels.size() == static_cast<size_t>(kWidth * kHeight * 4);
    bool pixelsPass = rendered;
    for (int x = 0; x < kWidth && rendered; ++x)
    {
        const auto expected = oracle(kDepth[x]);
        printPixel(x == 0 ? "near" : (x == 1 ? "mid" : "far"), pixels.data() + x * 4, expected);
        pixelsPass = pixelsPass && closeEnough(pixels.data() + x * 4, expected);
    }
    const bool depthChangesOutput = rendered
        && !std::equal(pixels.begin(), pixels.begin() + 4, pixels.begin() + 4)
        && !std::equal(pixels.begin() + 4, pixels.begin() + 8, pixels.begin() + 8);

    std::array<uint8_t, kWidth * kHeight * 4> gray {};
    for (int x = 0; x < kWidth; ++x)
    {
        gray[static_cast<size_t>(x) * 4 + 0] = 188;
        gray[static_cast<size_t>(x) * 4 + 1] = 188;
        gray[static_cast<size_t>(x) * 4 + 2] = 188;
        gray[static_cast<size_t>(x) * 4 + 3] = 255;
    }
    renderer.uploadRgba(gray.data(), kWidth, kHeight, kWidth * 4, sourceTexture);
    layer.depthFog = false;
    layer.depthTexture = 0;
    layer.graphColorTransformActive = true;
    layer.graphColorTransform.version = colortransform::kWireVersion;
    layer.graphColorTransform.input = {
        { static_cast<uint32_t>(kWidth), static_cast<uint32_t>(kHeight) },
        colortransform::PixelFormat::RGBA8, colortransform::ColorSpace::SRGB,
        colortransform::TransferFunction::SRGB, colortransform::AlphaMode::Straight };
    layer.graphColorTransform.output = layer.graphColorTransform.input;
    layer.graphColorTransform.workingColorSpace = colortransform::ColorSpace::LinearSRGB;
    layer.graphColorTransform.workingFormat = colortransform::PixelFormat::RGBA16F;
    layer.graphColorTransform.outputIntent = colortransform::OutputIntent::SdrDisplay;
    layer.graphColorTransform.toneMap = colortransform::ToneMap::None;
    layer.graphColorTransform.luminance = { 100.0, 100.0, 100.0, 100.0 };
    std::vector<uint8_t> identityPixels;
    const bool identityRendered = renderer.renderToPixels(
        &layer, 1, identityPixels, error);
    layer.graphColorTransform.toneMap = colortransform::ToneMap::Reinhard;
    layer.graphColorTransform.luminance.sourcePeakNits = 1000.0;
    std::vector<uint8_t> mappedPixels;
    const bool mappedRendered = renderer.renderToPixels(
        &layer, 1, mappedPixels, error);
    const int mappedOracle = colorTransformOracle(188);
    const bool colorTransformPixels = identityRendered && mappedRendered
        && identityPixels.size() >= 4 && mappedPixels.size() >= 4
        && std::abs(static_cast<int>(identityPixels[0]) - 188) <= kTolerance
        && std::abs(static_cast<int>(identityPixels[1]) - 188) <= kTolerance
        && std::abs(static_cast<int>(identityPixels[2]) - 188) <= kTolerance
        && std::abs(static_cast<int>(mappedPixels[0]) - mappedOracle) <= kTolerance
        && std::abs(static_cast<int>(mappedPixels[1]) - mappedOracle) <= kTolerance
        && std::abs(static_cast<int>(mappedPixels[2]) - mappedOracle) <= kTolerance
        && mappedPixels[3] == 255 && mappedPixels[0] < identityPixels[0];
    std::cout << "color-transform actual="
              << (mappedPixels.empty() ? -1 : static_cast<int>(mappedPixels[0]))
              << " oracle=" << mappedOracle << '\n';

    videorender::ColorTransformGl stateProbe;
    std::string stateError;
    GLuint stateTarget = 0;
    GLuint priorReadFramebuffer = 0;
    GLuint priorDrawFramebuffer = 0;
    glGenTextures(1, &stateTarget);
    glBindTexture(GL_TEXTURE_2D, stateTarget);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kWidth, kHeight, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    gl.GenFramebuffers(1, &priorReadFramebuffer);
    gl.GenFramebuffers(1, &priorDrawFramebuffer);
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, priorReadFramebuffer);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, priorDrawFramebuffer);
    colortransform::AdmissionFailure stateFailure = colortransform::AdmissionFailure::None;
    const auto stateTransform = colortransform::admit(
        layer.graphColorTransform, colortransform::BackendCapability::NativeGpu, stateFailure);
    const bool stateRendered = stateProbe.initialize(&gl, stateError) && stateTransform
        && stateProbe.render(sourceTexture, stateTarget, kWidth, kHeight,
                             *stateTransform, stateError);
    GLint restoredReadFramebuffer = 0;
    GLint restoredDrawFramebuffer = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &restoredReadFramebuffer);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &restoredDrawFramebuffer);
    const bool framebufferStatePreserved = stateRendered
        && restoredReadFramebuffer == static_cast<GLint>(priorReadFramebuffer)
        && restoredDrawFramebuffer == static_cast<GLint>(priorDrawFramebuffer);
    gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
    stateProbe.shutdown();
    gl.DeleteFramebuffers(1, &priorReadFramebuffer);
    gl.DeleteFramebuffers(1, &priorDrawFramebuffer);
    glDeleteTextures(1, &stateTarget);

    layer.graphColorTransform.toneMap = colortransform::ToneMap::None;
    std::vector<uint8_t> rejectedPixels { 1, 2, 3, 4 };
    std::string rejectedRenderError;
    const bool zeroTextureReadbackRejected = !renderer.renderToPixels(
        &layer, 1, rejectedPixels, rejectedRenderError)
        && rejectedPixels.empty()
        && rejectedRenderError == "color transform admission failed: toneMapRequired";

    renderer.deleteTexture(sourceTexture);
    renderer.deleteTexture(depthTexture);
    const bool externalTexturesDeleted = glIsTexture(sourceTexture) == GL_FALSE
        && glIsTexture(depthTexture) == GL_FALSE;
    renderer.shutdown();
    const bool ownerClean = !renderer.ready() && glGetError() == GL_NO_ERROR;

    glfwDestroyWindow(window);
    glfwTerminate();
    if (!rejectedAdmission || !pixelsPass || !depthChangesOutput || !colorTransformPixels
        || !framebufferStatePreserved
        || !zeroTextureReadbackRejected
        || !externalTexturesDeleted || !ownerClean)
    {
        std::cerr << "Depth Fog native pixel acceptance FAIL: rendered=" << rendered
                  << " rejectedAdmission=" << rejectedAdmission
                  << " depthChangesOutput=" << depthChangesOutput
                  << " colorTransformPixels=" << colorTransformPixels
                  << " framebufferStatePreserved=" << framebufferStatePreserved
                  << " zeroTextureReadbackRejected=" << zeroTextureReadbackRejected
                  << " texturesDeleted=" << externalTexturesDeleted
                  << " ownerClean=" << ownerClean << " error=" << error
                  << " rejection=" << rejection << '\n';
        return 1;
    }
    std::cout << "Native OpenGL frame pixels PASS tolerance=+/-" << kTolerance
              << " channels; Depth Fog and graph color transform production readback, "
                 "shader-failure unwind, and texture cleanup verified\n";
    return 0;
#endif
}
