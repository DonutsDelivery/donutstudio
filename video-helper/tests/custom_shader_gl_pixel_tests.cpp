#include "flat_shader_bridge.h"
#include "gl_loader.h"
#include "renderer.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr int kWidth = 4;
constexpr int kHeight = 4;
constexpr int kTempleWidth = 32;
constexpr int kTempleHeight = 32;

std::string readFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

programmableruntime::SessionSecret secret()
{
    programmableruntime::SessionSecret value {};
    for (std::size_t i = 0; i < value.size(); ++i)
        value[i] = static_cast<std::uint8_t>(i * 7u + 3u);
    return value;
}

programmableruntime::Grant grantFor(const std::string& source,
                                    programmableruntime::PayloadKind kind,
                                    std::uint64_t nonce)
{
    programmableruntime::Grant grant;
    grant.kind = kind;
    grant.fingerprint = programmableruntime::fingerprint(kind, source);
    grant.sessionGeneration = 8128;
    grant.nonce = nonce;
    grant.issuedAtMs = programmableadmission::nowMs();
    grant.approved = true;
    grant.cpuMs = programmableruntime::defaultCpuMs;
    grant.gpuMs = programmableruntime::defaultGpuMs;
    grant.memoryMiB = programmableruntime::defaultMemoryMiB;
    grant.mac = programmableruntime::sign(secret(), grant, kind);
    return grant;
}

videowire::ShaderOperation operation(videowire::ShaderOperationKind kind, int nodeId,
                                     int outputNodeId, const std::string& source,
                                     std::uint64_t nonce)
{
    videowire::ShaderOperation value;
    value.kind = kind;
    value.nodeId = nodeId;
    value.outputNodeId = outputNodeId;
    value.payload.schemaVersion = 2;
    value.payload.role = kind == videowire::ShaderOperationKind::filter
        ? videowire::FlatShaderRole::filter : videowire::FlatShaderRole::generator;
    value.payload.language = videowire::FlatShaderLanguage::glsl;
    value.payload.source = source;
    value.payload.sourceSha256 = videohelper::sha256Text(source);
    value.customGrant = grantFor(source, programmableruntime::PayloadKind::shader, nonce);
    return value;
}

videowire::ShaderOperation curatedGenerator(int nodeId, const std::string& source)
{
    auto value = operation(videowire::ShaderOperationKind::generator, nodeId, nodeId,
                           source, 0);
    value.customGrant.reset();
    value.payload.catalogPackId = "donutstudio-3d-raymarch-v1";
    value.payload.catalogProgramId = "fractal-bass-temple";
    return value;
}

std::shared_ptr<const videowire::ShaderOperationPlan> finishPlan(
    std::vector<videowire::ShaderOperation> operations, std::uint64_t revision)
{
    auto plan = std::make_shared<videowire::ShaderOperationPlan>();
    plan->operations = std::move(operations);
    plan->revision = revision;
    for (const auto& item : plan->operations)
        plan->passTargetCount += item.payload.passResources.passes.size();
    plan->digest = videowire::shaderOperationPlanDigest(*plan, revision);
    return plan;
}

videorender::LayerDesc layerFor(
    int clipId, const std::shared_ptr<const videowire::ShaderOperationPlan>& plan,
    unsigned texture = 0)
{
    videorender::LayerDesc layer;
    layer.clipId = clipId;
    layer.visualPlanStructuralRevision = plan->revision;
    layer.flatShaderBridge = true;
    layer.shaderOperationPlan = plan;
    layer.texture = texture;
    layer.texWidth = kWidth;
    layer.texHeight = kHeight;
    for (const auto& item : plan->operations)
    {
        auto values = item.generatedParameters;
        if (item.transitionPayload)
            values["progress"] = shadertransition::evaluateProgress(
                item.transitionPayload->progress, item.transitionPayload->direction,
                item.transitionPayload->easing);
        layer.shaderOperationParameters.emplace(item.nodeId, std::move(values));
    }
    return layer;
}

bool pixelNear(const std::vector<std::uint8_t>& pixels,
               std::array<int, 4> expected, int tolerance = 3)
{
    if (pixels.size() != kWidth * kHeight * 4u)
    {
        std::cerr << "pixel size " << pixels.size() << '\n';
        return false;
    }
    for (std::size_t offset = 0; offset < pixels.size(); offset += 4u)
        for (std::size_t channel = 0; channel < expected.size(); ++channel)
            if (std::abs(static_cast<int>(pixels[offset + channel]) - expected[channel]) > tolerance)
            {
                std::cerr << "pixel " << offset / 4u << " is "
                          << static_cast<int>(pixels[offset]) << ','
                          << static_cast<int>(pixels[offset + 1]) << ','
                          << static_cast<int>(pixels[offset + 2]) << ','
                          << static_cast<int>(pixels[offset + 3]) << '\n';
                return false;
            }
    return true;
}

bool render(videorender::FrameRenderer& renderer, const videorender::LayerDesc& layer,
            std::vector<std::uint8_t>& pixels, std::string& error)
{
    return renderer.prepareFlatShaderBridge(layer, error)
        && renderer.renderToPixels(&layer, 1, pixels, error);
}

bool checkRender(videorender::FrameRenderer& renderer, const videorender::LayerDesc& layer,
                 std::vector<std::uint8_t>& pixels, std::string& error, const char* message)
{
    if (render(renderer, layer, pixels, error)) return true;
    std::cerr << "FAIL: " << message << ": " << error << '\n';
    return false;
}

bool check(bool condition, const char* message)
{
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

std::string pixelSha256(const std::vector<std::uint8_t>& pixels)
{
    videohelper::Sha256 hash;
    hash.update(pixels.data(), pixels.size());
    return hash.finishHex();
}

bool templePixelsHaveContent(const std::vector<std::uint8_t>& pixels)
{
    if (pixels.size() != kTempleWidth * kTempleHeight * 4u) return false;
    std::size_t nonBlack = 0;
    std::array<std::uint8_t, 3> minimum { 255, 255, 255 };
    std::array<std::uint8_t, 3> maximum {};
    for (std::size_t offset = 0; offset < pixels.size(); offset += 4u)
    {
        if (pixels[offset] != 0 || pixels[offset + 1] != 0 || pixels[offset + 2] != 0)
            ++nonBlack;
        for (std::size_t channel = 0; channel < 3; ++channel)
        {
            minimum[channel] = std::min(minimum[channel], pixels[offset + channel]);
            maximum[channel] = std::max(maximum[channel], pixels[offset + channel]);
        }
        if (pixels[offset + 3] < 250) return false;
    }
    const bool spatiallyNonuniform = maximum[0] - minimum[0] >= 20
        || maximum[1] - minimum[1] >= 20 || maximum[2] - minimum[2] >= 20;
    const auto minimumCoveredPixels =
        static_cast<std::size_t>(kTempleWidth * kTempleHeight * 3 / 4);
    if (nonBlack < minimumCoveredPixels || !spatiallyNonuniform)
        std::cerr << "Temple coverage=" << nonBlack << " ranges="
                  << static_cast<int>(maximum[0] - minimum[0]) << ','
                  << static_cast<int>(maximum[1] - minimum[1]) << ','
                  << static_cast<int>(maximum[2] - minimum[2]) << '\n';
    return nonBlack >= minimumCoveredPixels && spatiallyNonuniform;
}

std::string readTextFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    std::ostringstream bytes;
    bytes << input.rdbuf();
    return input ? bytes.str() : std::string {};
}

std::shared_ptr<const videowire::ShaderOperationPlan> curatedTransitionPlan(
    const std::string& programId, const std::string& source, double progress, double strength,
    std::uint64_t revision, std::string& error)
{
    const auto* catalog = shadercatalog::find("vidvox-isf", programId);
    if (catalog == nullptr)
    {
        error = "missing curated transition catalog entry";
        return {};
    }
    shadertransition::Payload contract;
    contract.catalogPackId = "vidvox-isf";
    contract.catalogProgramId = programId;
    contract.sourceSha256 = videohelper::sha256Text(source);
    contract.source = source;
    contract.progress = progress;
    contract.direction = shadertransition::Direction::forward;
    contract.easing = shadertransition::Easing::linear;
    contract.parameters = shadertransition::parametersForProgress(*catalog, progress, error);
    if (programId == "crosszoom" && error.empty())
    {
        std::map<std::string, double> values;
        if (!shadercatalog::validateParameterWire(*catalog, contract.parameters, values, error))
            return {};
        values["strength"] = strength;
        contract.parameters = shadercatalog::parameterWire(*catalog, values, error);
    }
    if (!error.empty() || !shadertransition::validate(contract, error)) return {};

    videowire::ShaderOperation operation;
    operation.kind = videowire::ShaderOperationKind::transition;
    operation.nodeId = 120;
    operation.inputCount = 2;
    operation.inputNodeIds = { 100, 110 };
    operation.outputNodeId = 130;
    operation.payload.schemaVersion = 2;
    operation.payload.role = videowire::FlatShaderRole::filter;
    operation.payload.language = videowire::FlatShaderLanguage::isf;
    operation.payload.source = contract.source;
    operation.payload.sourceSha256 = contract.sourceSha256;
    operation.payload.catalogPackId = contract.catalogPackId;
    operation.payload.catalogProgramId = contract.catalogProgramId;
    operation.payload.parameters = contract.parameters;
    if (!shadercatalog::validateParameterWire(
            *catalog, contract.parameters, operation.payload.parameterValues, error))
        return {};
    operation.generatedParameters = operation.payload.parameterValues;
    operation.transitionPayload = contract;
    return finishPlan({ std::move(operation) }, revision);
}

std::uint8_t channelByte(double value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0, 1.0) * 255.0));
}

std::array<double, 4> sampleLinearClamp(
    const std::array<std::uint8_t, kWidth * kHeight * 4>& image, double u, double displayV)
{
    // Production uploads RGBA rows directly. IMG_NORM_PIXEL then maps the
    // display-space ISF coordinate to GL texture storage with 1.0 - y.
    const double v = 1.0 - displayV;
    const double px = std::clamp(u * kWidth - 0.5, 0.0, kWidth - 1.0);
    const double py = std::clamp(v * kHeight - 0.5, 0.0, kHeight - 1.0);
    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = std::min(x0 + 1, kWidth - 1);
    const int y1 = std::min(y0 + 1, kHeight - 1);
    const double fx = px - x0;
    const double fy = py - y0;
    std::array<double, 4> result {};
    for (std::size_t channel = 0; channel < result.size(); ++channel)
    {
        const auto at = [&](int x, int y)
        {
            return image[(static_cast<std::size_t>(y) * kWidth + x) * 4u + channel] / 255.0;
        };
        result[channel] = (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy)
                        + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy;
    }
    return result;
}

double shaderRand(double u, double v)
{
    const double value = std::sin(u * 12.9898 + v * 78.233) * 43758.5453;
    return value - std::floor(value);
}

std::vector<std::uint8_t> transitionOracle(
    const std::string& programId, double progress, double strength,
    const std::array<std::uint8_t, kWidth * kHeight * 4>& from,
    const std::array<std::uint8_t, kWidth * kHeight * 4>& to)
{
    std::vector<std::uint8_t> pixels;
    pixels.reserve(kWidth * kHeight * 4u);
    const auto smoothstep = [](double edge0, double edge1, double value)
    {
        const double t = std::clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x)
        {
            const double u = (static_cast<double>(x) + 0.5) / kWidth;
            // renderToPixels returns top-row-first compositor pixels. The ISF
            // pass runs in OpenGL's bottom-origin fragment space, then the
            // ordinary layer pass maps that texture back into compositor space.
            const double v = 1.0 - (static_cast<double>(y) + 0.5) / kHeight;
            double mixWeight = progress;
            if (programId == "directional-wipe")
            {
                const double projected = 0.5 * u - 0.5 * v - (-0.5 + progress * 1.5);
                mixWeight = 1.0 - smoothstep(-0.5, 0.0, projected);
            }
            else if (programId == "crosszoom")
            {
                if (progress == 0.0) mixWeight = 0.0;
                else if (progress == 1.0) mixWeight = 1.0;
                else if (progress < 0.5)
                    mixWeight = 0.5 * std::pow(2.0, 20.0 * progress - 10.0);
                else
                    mixWeight = 0.5 * (-std::pow(2.0, -20.0 * progress + 10.0) + 2.0);
            }
            if (programId != "crosszoom")
            {
                const auto a = sampleLinearClamp(from, u, v);
                const auto b = sampleLinearClamp(to, u, v);
                std::array<double, 4> transitionPixel {};
                for (std::size_t channel = 0; channel < 4; ++channel)
                    transitionPixel[channel] = a[channel] * (1.0 - mixWeight)
                                             + b[channel] * mixWeight;
                const double alpha = transitionPixel[3];
                constexpr std::array<double, 3> canvas {{ 0.04, 0.04, 0.05 }};
                for (std::size_t channel = 0; channel < 3; ++channel)
                    pixels.push_back(channelByte(canvas[channel] * (1.0 - alpha)
                                               + transitionPixel[channel] * alpha));
                pixels.push_back(255);
                continue;
            }

            const double centerX = 0.25 + 0.5 * progress;
            const double easedStrength = -strength * 0.5
                * (std::cos(3.141592653589793 * progress / 0.5) - 1.0);
            const double offset = shaderRand(u, v);
            std::array<double, 4> accumulated {};
            double total = 0.0;
            for (int sample = 0; sample <= 40; ++sample)
            {
                const double percent = (sample + offset) / 40.0;
                const double weight = 4.0 * (percent - percent * percent);
                const double sampleU = u + (centerX - u) * percent * easedStrength;
                const double sampleV = v + (0.5 - v) * percent * easedStrength;
                const auto a = sampleLinearClamp(from, sampleU, sampleV);
                const auto b = sampleLinearClamp(to, sampleU, sampleV);
                for (std::size_t channel = 0; channel < 3; ++channel)
                    accumulated[channel] += (a[channel] * (1.0 - mixWeight)
                                           + b[channel] * mixWeight) * weight;
                total += weight;
            }
            for (std::size_t channel = 0; channel < 3; ++channel)
                pixels.push_back(channelByte(accumulated[channel] / total));
            pixels.push_back(255);
        }
    return pixels;
}

bool pixelsNear(const std::vector<std::uint8_t>& actual,
                const std::vector<std::uint8_t>& expected, int tolerance = 4)
{
    if (actual.size() != expected.size())
    {
        std::cerr << "pixel buffer size expected=" << expected.size()
                  << " actual=" << actual.size() << '\n';
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index)
        if (std::abs(static_cast<int>(actual[index]) - static_cast<int>(expected[index])) > tolerance)
        {
            const auto pixel = index / 4u;
            std::cerr << "pixel mismatch x=" << pixel % kWidth << " y=" << pixel / kWidth
                      << " channel=" << index % 4u
                      << " expected=" << static_cast<int>(expected[index])
                      << " actual=" << static_cast<int>(actual[index])
                      << " tolerance=" << tolerance << '\n';
            return false;
        }
    return true;
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& pixels)
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : pixels)
    {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

bool pinnedPixelsNear(const std::vector<std::uint8_t>& actual,
                      const std::array<std::array<int, 2>, 4>& coordinates,
                      const std::array<std::array<std::uint8_t, 4>, 4>& expected,
                      int tolerance)
{
    if (actual.size() != kWidth * kHeight * 4u) return false;
    for (std::size_t pixel = 0; pixel < coordinates.size(); ++pixel)
    {
        const auto x = static_cast<std::size_t>(coordinates[pixel][0]);
        const auto y = static_cast<std::size_t>(coordinates[pixel][1]);
        const auto offset = (y * kWidth + x) * 4u;
        for (std::size_t channel = 0; channel < 4; ++channel)
            if (std::abs(static_cast<int>(actual[offset + channel])
                         - static_cast<int>(expected[pixel][channel])) > tolerance)
            {
                std::cerr << "pinned pixel mismatch x=" << x << " y=" << y
                          << " channel=" << channel
                          << " expected=" << static_cast<int>(expected[pixel][channel])
                          << " actual=" << static_cast<int>(actual[offset + channel])
                          << " tolerance=" << tolerance << '\n';
                return false;
            }
    }
    return true;
}
}

int main()
{
#if !defined(__linux__) && !defined(ARBIT_STRICT_METAL_FIXTURE)
    return 77;
#else
#if defined(ARBIT_STRICT_METAL_FIXTURE)
    constexpr int initializationFailure = 1;
#else
    constexpr int initializationFailure = 77;
#endif
    programmableadmission::verifier().reset(secret(), 8128);
    glfwSetErrorCallback([](int code, const char* text) {
        std::cerr << "GLFW " << code << ": " << (text != nullptr ? text : "unknown") << '\n';
    });
    if (glfwInit() != GLFW_TRUE) return initializationFailure;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
#if defined(ARBIT_STRICT_METAL_FIXTURE)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight,
                                          "Ordered shader pixel acceptance", nullptr, nullptr);
    if (window == nullptr) { glfwTerminate(); return initializationFailure; }
    glfwMakeContextCurrent(window);

    std::string error;
    arbitgl::GlFuncs gl;
    if (!arbitgl::loadGlFunctions(gl, error))
    { glfwDestroyWindow(window); glfwTerminate(); return initializationFailure; }

    bool ok = true;
    videorender::FrameRenderer preview;
    ok &= check(preview.initialize(&gl, kWidth, kHeight, error,
#if defined(ARBIT_STRICT_METAL_FIXTURE)
                                   true
#else
                                   false
#endif
                                   ), "preview renderer initializes");

    const std::string generatorSource =
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(0.10, 0.20, 0.30, 1.0); }";
    const std::string transitionSource =
        "uniform sampler2D startImage; uniform sampler2D endImage;"
        "uniform float progress;"
        "void mainImage(out vec4 c, in vec2 p) {"
        " vec2 uv=p/uResolution; c=mix(texture(startImage,uv),texture(endImage,uv),progress); }";
    const std::string filterSource =
        "uniform sampler2D inputImage; void mainImage(out vec4 c, in vec2 p) {"
        " vec4 v=texture(inputImage,p/uResolution); c=vec4(v.g*0.5+0.1,v.r*0.25,v.b*0.75,1.0); }";

    auto generator = operation(videowire::ShaderOperationKind::generator, 10, 10,
                               generatorSource, 1);
    auto transition = operation(videowire::ShaderOperationKind::transition, 20, 20,
                                transitionSource, 2);
    transition.inputCount = 2;
    transition.inputNodeIds = { 10, 99 };
    shadertransition::Payload transitionContract;
    transitionContract.direction = shadertransition::Direction::reverse;
    transitionContract.easing = shadertransition::Easing::easeInOut;
    transitionContract.progress = 0.25;
    transitionContract.source = transitionSource;
    transitionContract.sourceSha256 = videohelper::sha256Text(transitionSource);
    transition.transitionPayload = transitionContract;
    auto filter = operation(videowire::ShaderOperationKind::filter, 30, 30,
                            filterSource, 3);
    filter.inputCount = 1;
    filter.inputNodeIds = { 20, 0 };
    const auto chainPlan = finishPlan({ generator, transition, filter }, 7);
    std::weak_ptr<const videowire::ShaderOperationPlan> destroyedPlan;
    {
        auto prior = finishPlan({ generator, transition, filter }, 7);
        destroyedPlan = prior;
        ok &= check(prior->digest == chainPlan->digest,
                    "equal immutable plans have one deterministic digest");
    }
    ok &= check(destroyedPlan.expired(),
                "prior plan is destroyed before the digest-keyed replacement is used");

    std::array<std::uint8_t, kWidth * kHeight * 4> sourcePixels {};
    for (std::size_t offset = 0; offset < sourcePixels.size(); offset += 4u)
    {
        sourcePixels[offset] = 204;
        sourcePixels[offset + 1] = 26;
        sourcePixels[offset + 2] = 13;
        sourcePixels[offset + 3] = 255;
    }
    const unsigned sourceTexture = preview.uploadRgba(sourcePixels.data(), kWidth, kHeight,
                                                       kWidth * 4, 0);
    auto chainLayer = layerFor(31, chainPlan, sourceTexture);
    std::vector<std::uint8_t> chainPixels;
    ok &= checkRender(preview, chainLayer, chainPixels, error, "ordered chain renders");
#if defined(ARBIT_STRICT_METAL_FIXTURE)
    ok &= check(preview.compositorBackend() == "metal",
                "strict fixture reports the Metal compositor backend");
#else
    ok &= check(preview.compositorBackend() == "opengl",
                "OpenGL fixture reports the OpenGL compositor backend");
#endif
    ok &= check(pixelNear(chainPixels, { 25, 0, 0, 255 }),
                "generator, two-source reverse ease-in-out transition, and filter execute once in order");

    const auto templeSource = readFile(
        std::string(ARBIT_SHADER_PACK_ROOT)
        + "/arbit-3d-raymarch/shaders/fractal_bass_temple.fs");
    auto templeOperation = curatedGenerator(80, templeSource);
    const auto templePlan = finishPlan({ templeOperation }, 15);
    auto templePreviewLayer = layerFor(80, templePlan);
    templePreviewLayer.shaderClock.timeSec = 0.5;
    std::vector<std::uint8_t> templePreviewPixels;
    ok &= check(!templeSource.empty() && templeSource.size() <= 1024u * 1024u
                    && templePlan->operations.size() == 1u
                    && templePlan->passTargetCount == 0u,
                "Raymarched Temple stays inside source, operation, and pass budgets");
    videorender::FrameRenderer templeRenderer;
    ok &= check(templeRenderer.initialize(&gl, kTempleWidth, kTempleHeight, error,
#if defined(ARBIT_STRICT_METAL_FIXTURE)
                                         true
#else
                                         false
#endif
                                         ), "first renderer lifecycle owner initializes");
    ok &= checkRender(templeRenderer, templePreviewLayer, templePreviewPixels, error,
                      "Raymarched Temple first renderer lifecycle owner renders");
    ok &= check(templePreviewPixels.size() == kTempleWidth * kTempleHeight * 4u,
                "Raymarched Temple returns one complete bounded RGBA frame");
    ok &= check(templePixelsHaveContent(templePreviewPixels),
                "Raymarched Temple has opaque nonblack coverage and spatially nonuniform RGB");
    const auto templePixelDigest = pixelSha256(templePreviewPixels);
    if (templePixelDigest != "c062d7635faec31f88fad2617eb4f27a8ea6faa419bcc1addd1c437b7a3dd60c")
        std::cerr << "Temple pixel digest: " << templePixelDigest << '\n';
    ok &= check(templePixelDigest == "c062d7635faec31f88fad2617eb4f27a8ea6faa419bcc1addd1c437b7a3dd60c",
                "Raymarched Temple matches the independent native pixel fixture digest");

    videorender::FrameRenderer secondRenderer;
    ok &= check(secondRenderer.initialize(&gl, kTempleWidth, kTempleHeight, error,
#if defined(ARBIT_STRICT_METAL_FIXTURE)
                                         true
#else
                                         false
#endif
                                         ), "second renderer lifecycle owner initializes");
    auto templeSecondLayer = layerFor(81, templePlan);
    templeSecondLayer.shaderClock.timeSec = 0.5;
    std::vector<std::uint8_t> templeSecondPixels;
    ok &= checkRender(secondRenderer, templeSecondLayer, templeSecondPixels, error,
                      "Raymarched Temple second renderer lifecycle owner renders");
    ok &= check(templeSecondPixels == templePreviewPixels,
                "Raymarched Temple distinct renderer lifecycle owners produce identical pixels");

    videorender::FrameRenderer restartedRenderer;
    ok &= check(restartedRenderer.initialize(&gl, kTempleWidth, kTempleHeight, error,
#if defined(ARBIT_STRICT_METAL_FIXTURE)
                                            true
#else
                                            false
#endif
                                            ), "restarted renderer initializes");
    auto templeRestartLayer = layerFor(82, templePlan);
    templeRestartLayer.shaderClock.timeSec = 0.5;
    std::vector<std::uint8_t> templeRestartPixels;
    ok &= checkRender(restartedRenderer, templeRestartLayer, templeRestartPixels, error,
                      "Raymarched Temple reconstructs after renderer restart");
    ok &= check(templeRestartPixels == templePreviewPixels,
                "Raymarched Temple restart replay preserves exact pixels");

    auto templeFilter = operation(videowire::ShaderOperationKind::filter, 81, 81,
        "uniform sampler2D inputImage; void mainImage(out vec4 c,in vec2 p){"
        "vec4 v=texture(inputImage,p/uResolution);c=vec4(1.0-v.rgb,v.a);}", 16);
    templeFilter.inputCount = 1;
    templeFilter.inputNodeIds = { 80, 0 };
    const auto filteredTemplePlan = finishPlan({ templeOperation, templeFilter }, 16);
    auto filteredTempleLayer = layerFor(83, filteredTemplePlan);
    filteredTempleLayer.shaderClock.timeSec = 0.5;
    std::vector<std::uint8_t> filteredTemplePixels;
    ok &= checkRender(restartedRenderer, filteredTempleLayer, filteredTemplePixels, error,
                      "Raymarched Temple graph mutation renders");
    ok &= check(filteredTemplePixels != templeRestartPixels,
                "Raymarched Temple graph mutation changes native output");

    auto alteredTemplePlan = std::make_shared<videowire::ShaderOperationPlan>(*templePlan);
    alteredTemplePlan->operations[0].payload.source += "\n// altered";
    alteredTemplePlan->operations[0].payload.sourceSha256 = videohelper::sha256Text(
        alteredTemplePlan->operations[0].payload.source);
    alteredTemplePlan->digest = videowire::shaderOperationPlanDigest(
        *alteredTemplePlan, alteredTemplePlan->revision);
    std::string alteredTempleError;
    ok &= check(!restartedRenderer.prepareFlatShaderBridge(
                    layerFor(84, alteredTemplePlan), alteredTempleError)
                    && !restartedRenderer.hasPreparedShaderPlan(84),
                "altered Raymarched Temple source fails exact catalog admission");
    ok &= check(alteredTempleError == "shader operation source differs from catalog authority",
                "altered Raymarched Temple rejection comes from catalog authority");

    auto forgedCatalogPlan = std::make_shared<videowire::ShaderOperationPlan>(*chainPlan);
    forgedCatalogPlan->operations[0].payload.catalogPackId = "vidvox-isf";
    forgedCatalogPlan->operations[0].payload.catalogProgramId = "rgb-invert";
    forgedCatalogPlan->digest = videowire::shaderOperationPlanDigest(
        *forgedCatalogPlan, forgedCatalogPlan->revision);
    std::string forgedCatalogError;
    ok &= check(!preview.prepareFlatShaderBridge(layerFor(32, forgedCatalogPlan),
                                                 forgedCatalogError)
                    && !preview.hasPreparedShaderPlan(32),
                "custom payload with forged curated identity is rejected before preparation");

    auto skippedPlan = finishPlan({ generator, filter }, 8);
    auto skippedFilter = std::const_pointer_cast<videowire::ShaderOperationPlan>(skippedPlan);
    skippedFilter->operations[1].inputNodeIds = { 10, 0 };
    skippedFilter->digest = videowire::shaderOperationPlanDigest(*skippedFilter, skippedFilter->revision);
    auto skippedLayer = layerFor(32, skippedFilter, sourceTexture);
    std::vector<std::uint8_t> skippedPixels;
    ok &= checkRender(preview, skippedLayer, skippedPixels, error, "skipped control renders");
    ok &= check(pixelNear(skippedPixels, { 31, 3, 10, 255 }),
                "control fixture distinguishes a skipped transition");

    auto duplicate = filter;
    duplicate.nodeId = 40;
    duplicate.outputNodeId = 40;
    duplicate.inputNodeIds = { 30, 0 };
    const auto duplicatePlan = finishPlan({ generator, transition, filter, duplicate }, 9);
    auto duplicateLayer = layerFor(33, duplicatePlan, sourceTexture);
    std::vector<std::uint8_t> duplicatePixels;
    ok &= checkRender(preview, duplicateLayer, duplicatePixels, error, "duplicate control renders");
    ok &= check(duplicatePixels != chainPixels,
                "control fixture distinguishes a duplicated filter");

    auto firstOrder = operation(videowire::ShaderOperationKind::filter, 50, 50,
        "uniform sampler2D inputImage; void mainImage(out vec4 c,in vec2 p){"
        "vec4 v=texture(inputImage,p/uResolution);c=vec4(v.r+0.2,v.g*0.5,v.b,1.0);}", 8);
    firstOrder.inputCount = 1; firstOrder.inputNodeIds = { 10, 0 };
    auto secondOrder = operation(videowire::ShaderOperationKind::filter, 51, 51,
        "uniform sampler2D inputImage; void mainImage(out vec4 c,in vec2 p){"
        "vec4 v=texture(inputImage,p/uResolution);c=vec4(v.r*0.5,v.g+0.2,v.b,1.0);}", 9);
    secondOrder.inputCount = 1; secondOrder.inputNodeIds = { 50, 0 };
    const auto orderedControl = finishPlan({ generator, firstOrder, secondOrder }, 12);
    auto reorderedFirst = secondOrder;
    reorderedFirst.inputNodeIds = { 10, 0 };
    auto reorderedSecond = firstOrder;
    reorderedSecond.inputNodeIds = { 51, 0 };
    const auto reorderedControl = finishPlan({ generator, reorderedFirst, reorderedSecond }, 13);
    std::vector<std::uint8_t> orderedControlPixels, reorderedControlPixels;
    ok &= checkRender(preview, layerFor(34, orderedControl, sourceTexture),
                      orderedControlPixels, error, "ordered control renders");
    ok &= checkRender(preview, layerFor(36, reorderedControl, sourceTexture),
                      reorderedControlPixels, error, "actually reordered control renders");
    ok &= check(orderedControlPixels != reorderedControlPixels,
                "fixture pixels fail when two noncommutative operations are reordered");

    ok &= check(chainLayer.fromLayer == nullptr && !chainPixels.empty(),
                "single-layer fallback keeps the two exact operation sources");

    const std::string multipassSource = R"isf(/*{
 "PASSES":[{"TARGET":"first"},{"TARGET":"second"}]
}*/
void main() {
 vec2 uv=isf_FragNormCoord;
 if (PASSINDEX==0) gl_FragColor=vec4(0.10,0.20,0.30,1.0);
 else gl_FragColor=texture(first,uv)+vec4(0.20,0.10,0.05,0.0);
})isf";
    auto multipassOperation = operation(videowire::ShaderOperationKind::generator, 4, 40,
                                        multipassSource, 4);
    multipassOperation.payload.language = videowire::FlatShaderLanguage::isf;
    multipassOperation.customGrant = grantFor(
        multipassSource, programmableruntime::PayloadKind::isf, 4);
    std::string passAdmissionError;
    ok &= check(videowire::admitCuratedIsfPassResources(
                    multipassSource, multipassOperation.payload.passResources,
                    passAdmissionError),
                "multipass resources come from the exact source schedule");
    const auto multipassPlan = finishPlan({ multipassOperation }, 10);
    auto multipassLayer = layerFor(35, multipassPlan);
    std::vector<std::uint8_t> multipassPixels;
    std::string rejectedError;
    ok &= checkRender(preview, multipassLayer, multipassPixels, error, "multipass fixture renders");
    ok &= check(pixelNear(multipassPixels, { 77, 77, 89, 255 }),
                "multipass output proves both passes execute once and in order");
    auto missingPass = std::make_shared<videowire::ShaderOperationPlan>(*multipassPlan);
    missingPass->operations[0].payload.passResources.passes.pop_back();
    missingPass->passTargetCount = 1;
    missingPass->digest = videowire::shaderOperationPlanDigest(*missingPass, missingPass->revision);
    ok &= check(!preview.prepareFlatShaderBridge(layerFor(37, missingPass), passAdmissionError),
                "missing multipass schedule is rejected before pixels can pass");
    auto repeatedPass = std::make_shared<videowire::ShaderOperationPlan>(*multipassPlan);
    repeatedPass->operations[0].payload.passResources.passes.push_back(
        repeatedPass->operations[0].payload.passResources.passes.front());
    repeatedPass->passTargetCount = 3;
    repeatedPass->digest = videowire::shaderOperationPlanDigest(*repeatedPass, repeatedPass->revision);
    ok &= check(!preview.prepareFlatShaderBridge(layerFor(38, repeatedPass), passAdmissionError),
                "repeated multipass schedule is rejected before pixels can pass");

    auto singletonTransition = transition;
    singletonTransition.inputNodeIds = { 10, 10 };
    const auto illicitSingleton = finishPlan({ singletonTransition }, 14);
    ok &= check(!preview.prepareFlatShaderBridge(layerFor(39, illicitSingleton), rejectedError),
                "illicit singleton transition is rejected instead of producing a pixel");

    const std::string invalidCompileSource =
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(1.0) this_is_not_glsl; }";
    auto invalid = operation(videowire::ShaderOperationKind::generator, 60, 60,
                             invalidCompileSource, 5);
    const auto invalidPlan = finishPlan({ invalid }, 11);
    auto invalidLayer = layerFor(31, invalidPlan);
    std::vector<std::uint8_t> priorPixels;
    ok &= checkRender(preview, chainLayer, priorPixels, error,
                      "prior program is active before failed replacement");
    ok &= check(!preview.prepareFlatShaderBridge(invalidLayer, rejectedError)
                    && !rejectedError.empty(),
                "semantically admitted replacement fails actual staged shader compilation");
    std::vector<std::uint8_t> retainedPixels;
    ok &= check(preview.renderToPixels(&chainLayer, 1, retainedPixels, error)
                    && retainedPixels == priorPixels,
                "failed staged compilation leaves the prior program rendering");

    const std::string oldDigest = chainPlan->digest;
    auto sameContentPlan = finishPlan({ generator, transition, filter }, 7);
    ok &= check(sameContentPlan->digest == oldDigest,
                "equal immutable plans have an exact shared digest independent of address");
    auto collisionMutation = std::make_shared<videowire::ShaderOperationPlan>(*sameContentPlan);
    collisionMutation->operations[0].outputNodeId = 1234;
    ok &= check(videowire::shaderOperationPlanDigest(*collisionMutation,
                    collisionMutation->revision) != sameContentPlan->digest,
                "resource mutation cannot collide with the prior plan identity");

    videorender::FrameRenderer exported;
    ok &= check(exported.initialize(&gl, kWidth, kHeight, error,
#if defined(ARBIT_STRICT_METAL_FIXTURE)
                                    true
#else
                                    false
#endif
                                    ), "export renderer initializes");

    videorender::FrameRenderer reconstructed;
    ok &= check(reconstructed.initialize(&gl, kWidth, kHeight, error,
#if defined(ARBIT_STRICT_METAL_FIXTURE)
                                     true
#else
                                     false
#endif
                                     ), "isolated reconstructed renderer initializes");
    std::array<std::uint8_t, kWidth * kHeight * 4> fromPixels {};
    std::array<std::uint8_t, kWidth * kHeight * 4> toPixels {};
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x)
        {
            const auto offset = (static_cast<std::size_t>(y) * kWidth + x) * 4u;
            fromPixels[offset] = static_cast<std::uint8_t>(24 + x * 43 + y * 11);
            fromPixels[offset + 1] = static_cast<std::uint8_t>(210 - x * 17 - y * 29);
            fromPixels[offset + 2] = static_cast<std::uint8_t>(18 + x * 13 + y * 47);
            fromPixels[offset + 3] = static_cast<std::uint8_t>(45 + x * 37 + y * 19);
            toPixels[offset] = static_cast<std::uint8_t>(225 - x * 31 - y * 19);
            toPixels[offset + 1] = static_cast<std::uint8_t>(20 + x * 21 + y * 37);
            toPixels[offset + 2] = static_cast<std::uint8_t>(196 - x * 39 + y * 7);
            toPixels[offset + 3] = static_cast<std::uint8_t>(220 - x * 23 - y * 31);
        }
    const auto uploadPair = [&](videorender::FrameRenderer& owner)
    {
        return std::array<unsigned, 2> {
            owner.uploadRgba(fromPixels.data(), kWidth, kHeight, kWidth * 4, 0),
            owner.uploadRgba(toPixels.data(), kWidth, kHeight, kWidth * 4, 0)
        };
    };
    const auto previewPair = uploadPair(preview);
    const auto exportPair = uploadPair(exported);
    const auto reconstructedPair = uploadPair(reconstructed);
    struct StarterTransitionFixture
    {
        const char* id;
        const char* file;
    };
    constexpr std::array<StarterTransitionFixture, 3> starterTransitions {{
        { "fade", "Fade.fs" },
        { "directional-wipe", "Directional Wipe.fs" },
        { "crosszoom", "CrossZoom.fs" }
    }};
    constexpr std::array<std::array<int, 2>, 4> pinnedCoordinates {{
        { 0, 0 }, { 3, 0 }, { 1, 2 }, { 3, 3 }
    }};
    // Independently calculated from the fixture equations and bundled shader
    // formulas through the complete production path: direct top-row-first upload,
    // bottom-origin ISF coordinates, IMG_NORM_PIXEL's y flip, the layer-space flip,
    // straight-alpha composition over the default opaque canvas, and UNORM8 rounding.
    constexpr std::array<std::array<std::array<std::uint8_t, 4>, 4>, 3> pinnedRgba {{
        {{{ 70, 65, 62, 255 }, { 90, 77, 46, 255 },
          { 66, 68, 80, 255 }, { 74, 75, 85, 255 }}},
        {{{ 196, 19, 171, 255 }, { 90, 77, 46, 255 },
          { 66, 68, 80, 255 }, { 157, 62, 167, 255 }}},
        {{{ 57, 179, 54, 255 }, { 136, 150, 73, 255 },
          { 96, 136, 123, 255 }, { 155, 99, 167, 255 }}}
    }};
    constexpr std::array<std::uint64_t, 3> oracleHashes {{
        0x0d45263ad8339facull, 0x7879688a231b0c11ull, 0x54311baeeebb8516ull
    }};
    const auto renderTransition = [&](videorender::FrameRenderer& owner,
                                      const std::array<unsigned, 2>& textures,
                                      const std::shared_ptr<const videowire::ShaderOperationPlan>& plan,
                                      int clipId, std::vector<std::uint8_t>& pixels)
    {
        videorender::LayerDesc fromLayer;
        fromLayer.texture = textures[0];
        fromLayer.texWidth = kWidth;
        fromLayer.texHeight = kHeight;
        auto layer = layerFor(clipId, plan, textures[1]);
        layer.fromLayer = &fromLayer;
        return checkRender(owner, layer, pixels, error, "curated starter transition renders");
    };
    for (std::size_t index = 0; index < starterTransitions.size(); ++index)
    {
        const auto& fixture = starterTransitions[index];
        const auto source = readTextFile(
            std::string(ARBIT_SHADER_PACK_ROOT) + "/vidvox-isf/ISF/" + fixture.file);
        ok &= check(!source.empty(), "exact bundled transition source loads");
        const bool crossZoom = fixture.id == std::string("crosszoom");
        const double progress = crossZoom ? 0.37 : 0.5;
        const double strength = crossZoom ? 0.68 : 0.0;
        const auto plan = curatedTransitionPlan(
            fixture.id, source, progress, strength, 200 + index, error);
        ok &= check(plan != nullptr, "exact bundled transition creates an immutable plan");
        if (plan == nullptr) continue;
        const auto expected = transitionOracle(
            fixture.id, progress, strength, fromPixels, toPixels);
        ok &= check(fnv1a64(expected) == oracleHashes[index],
                    "transition oracle matches its independently pinned full-frame hash");
        std::vector<std::uint8_t> previewPixels, exportPixelsForStarter, reconstructedPixels;
        ok &= renderTransition(preview, previewPair, plan, 200 + static_cast<int>(index),
                               previewPixels);
        ok &= renderTransition(exported, exportPair, plan, 300 + static_cast<int>(index),
                               exportPixelsForStarter);
        ok &= renderTransition(reconstructed, reconstructedPair, plan,
                               400 + static_cast<int>(index), reconstructedPixels);
        ok &= check(pixelsNear(previewPixels, expected, crossZoom ? 7 : 4),
                    "starter transition OpenGL pixels match the independent CPU oracle");
        ok &= check(pinnedPixelsNear(previewPixels, pinnedCoordinates, pinnedRgba[index],
                                     crossZoom ? 7 : 4),
                    "starter transition matches independently pinned multi-pixel RGBA values");
        ok &= check(previewPixels.size() >= 16u
                        && !std::equal(previewPixels.begin(), previewPixels.begin() + 4,
                                       previewPixels.begin() + 12),
                    "starter transition keeps spatially nonuniform output pixels");
        ok &= check(exportPixelsForStarter == previewPixels,
                    "starter transition export owner matches preview pixels");
        ok &= check(reconstructedPixels == previewPixels,
                    "isolated reconstructed owner reproduces immutable transition state");
        if (crossZoom)
        {
            const auto ignoredStrength = transitionOracle(
                fixture.id, progress, 0.25, fromPixels, toPixels);
            const auto noDisplacement = transitionOracle(
                fixture.id, progress, 0.0, fromPixels, toPixels);
            const auto plainDissolve = transitionOracle(
                "fade", progress, 0.0, fromPixels, toPixels);
            ok &= check(!pixelsNear(expected, ignoredStrength, 7),
                        "corrected oracle distinguishes the authored strength");
            ok &= check(!pixelsNear(expected, noDisplacement, 7),
                        "corrected oracle distinguishes spatial displacement");
            ok &= check(!pixelsNear(expected, plainDissolve, 7),
                        "corrected oracle distinguishes weighted Cross Zoom sampling");
            ok &= check(!pixelsNear(previewPixels, ignoredStrength, 7),
                        "Cross Zoom pixels reject an ignored-strength implementation");
            ok &= check(!pixelsNear(previewPixels, noDisplacement, 7),
                        "Cross Zoom pixels reject a no-displacement implementation");
            ok &= check(!pixelsNear(previewPixels, plainDissolve, 7),
                        "Cross Zoom pixels reject a plain dissolve implementation");
        }
    }

    const unsigned exportTexture = exported.uploadRgba(sourcePixels.data(), kWidth, kHeight,
                                                        kWidth * 4, 0);
    auto exportLayer = layerFor(31, sameContentPlan, exportTexture);
    std::vector<std::uint8_t> previewParityPixels;
    ok &= checkRender(preview, chainLayer, previewParityPixels, error,
                      "preview owner renders parity frame");
    std::vector<std::uint8_t> exportPixels;
    ok &= checkRender(exported, exportLayer, exportPixels, error, "export owner renders");
    ok &= check(sameContentPlan->digest == chainPlan->digest
                    && exportPixels == previewParityPixels,
                "isolated preview and export owners share the exact digest and pixels");

    preview.clearClipShader(31);
    std::vector<std::uint8_t> clearedPixels;
    ok &= check(!preview.hasPreparedShaderPlan(31),
                "clip removal clears the prepared operation program");
    ok &= check(render(preview, chainLayer, clearedPixels, error),
                "clip replacement can prepare the exact plan again");
    auto replacementOwnerLayer = layerFor(41, sameContentPlan, sourceTexture);
    ok &= check(render(preview, replacementOwnerLayer, clearedPixels, error)
                    && preview.hasPreparedShaderPlan(31)
                    && preview.hasPreparedShaderPlan(41),
                "publication replacement fixture starts with both native shader owners");
    preview.retainShaderPlanClips({ 41 });
    ok &= check(!preview.hasPreparedShaderPlan(31)
                    && preview.hasPreparedShaderPlan(41),
                "publication replacement retires a stale clip owner without a timeline change");
    auto revisedLayer = chainLayer;
    revisedLayer.visualPlanStructuralRevision = 99;
    ok &= check(!preview.prepareFlatShaderBridge(revisedLayer, rejectedError),
                "revision change fails closed and cannot reuse prepared programs");

    preview.deleteTexture(sourceTexture);
    exported.deleteTexture(exportTexture);
    for (const auto texture : previewPair) preview.deleteTexture(texture);
    for (const auto texture : exportPair) exported.deleteTexture(texture);
    for (const auto texture : reconstructedPair) reconstructed.deleteTexture(texture);
    preview.shutdown();
    exported.shutdown();
    reconstructed.shutdown();
    preview.shutdown();
    exported.shutdown();
    reconstructed.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();

    if (!ok) return 1;
    std::cout << "ordered shader "
#if defined(ARBIT_STRICT_METAL_FIXTURE)
              << "strict Metal"
#else
              << "OpenGL"
#endif
              << " behavioral fixtures PASS\n";
    return 0;
#endif
}
