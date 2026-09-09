#include "diffraction_material_gl.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "support/diffraction_reference_oracle.h"
#include "support/diffraction_wavelength_validation_scenes.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
bool near(float actual, double expected, double tolerance = 2.0e-4)
{
    return std::abs(static_cast<double>(actual) - expected) <= tolerance;
}

std::array<double, 3> expectedDirectionMoment(
    const diffractionmaterial::reference::EvaluationResult& result)
{
    std::array<double, 3> moment {};
    for (std::size_t index = 0; index < result.eventCount; ++index)
    {
        const auto& event = result.events[index];
        const auto broadeningMoment = std::exp(
            -0.5 * event.angularStandardDeviationRadians
                * event.angularStandardDeviationRadians);
        moment[0] += broadeningMoment * event.energy * event.outgoingDirection.x;
        moment[1] += broadeningMoment * event.energy * event.outgoingDirection.y;
        moment[2] += broadeningMoment * event.energy * event.outgoingDirection.z;
    }
    return moment;
}

std::array<float, 4> readAttachment(arbitgl::GlFuncs& gl,
                                    const diffractionmaterial::OpenGlPhysicalDiffractionFrame& frame,
                                    unsigned attachment)
{
    int previousReadFramebuffer = 0;
    int previousReadBuffer = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
    glGetIntegerv(GL_READ_BUFFER, &previousReadBuffer);

    std::array<float, 4> values {};
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, frame.framebuffer);
    glReadBuffer(attachment);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, values.data());

    gl.BindFramebuffer(GL_READ_FRAMEBUFFER,
                       static_cast<unsigned>(previousReadFramebuffer));
    glReadBuffer(static_cast<unsigned>(previousReadBuffer));
    return values;
}
} // namespace

int main()
{
#if !defined(__linux__)
    std::cerr << "This acceptance target requires native Linux OpenGL\n";
    return 77;
#else
    using namespace diffractionmaterial;
    using namespace diffractionmaterial::reference;

    auto description = makeAluminiumBinaryGratingPreset();
    description.roughness = {};
    std::string admissionError;
    auto alternateDescription = description;
    alternateDescription.spectrum.wavelengthsNanometres[1] = 410.0f;
    const auto alternate = admit(alternateDescription, admissionError);
    const bool alternateAdmissionRejected = !alternate
        && admissionError
            == "diffraction material production tier requires the canonical wavelength schedule";

    glfwSetErrorCallback([](int code, const char* text) {
        std::cerr << "GLFW " << code << ": "
                  << (text != nullptr ? text : "unknown") << '\n';
    });
    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "BLOCKED: no native GLX/EGL display for diffraction execution\n";
        return 77;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(8, 8, "Physical diffraction GL", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "BLOCKED: native OpenGL 3.3 context creation failed\n";
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent(window);

    arbitgl::GlFuncs gl;
    std::string error;
    if (!arbitgl::loadGlFunctions(gl, error))
    {
        std::cerr << "BLOCKED: GL loader failed: " << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return 77;
    }

    auto rejectedGl = gl;
    rejectedGl.CreateShader = [](unsigned) -> unsigned { return 0; };
    OpenGlPhysicalDiffractionExecutor rejectedExecutor;
    const bool shaderFailureClosed = !rejectedExecutor.initialize(&rejectedGl, error)
        && !rejectedExecutor.ready()
        && error == "physical diffraction OpenGL shader allocation failed";
    while (glGetError() != GL_NO_ERROR) {}

    unsigned retainedTexture = 0;
    glGenTextures(1, &retainedTexture);
    glBindTexture(GL_TEXTURE_2D, retainedTexture);
    OpenGlPhysicalDiffractionExecutor executor;
    if (!executor.initialize(&gl, error))
    {
        std::cerr << "physical diffraction executor initialization failed: " << error << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    int textureAfterInitialization = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &textureAfterInitialization);

    const auto admitted = admit(description, admissionError);
    if (!admitted)
    {
        std::cerr << "physical diffraction fixture admission failed: " << admissionError << '\n';
        return EXIT_FAILURE;
    }

    SpectralIncidentLight light;
    OpenGlPhysicalDiffractionFrame frame;
    unsigned retainedPixelPackBuffer = 0;
    gl.GenBuffers(1, &retainedPixelPackBuffer);
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER, retainedPixelPackBuffer);
    gl.BufferData(GL_PIXEL_PACK_BUFFER, 128, nullptr, GL_STREAM_READ);
    glPixelStorei(GL_PACK_ALIGNMENT, 8);
    glPixelStorei(GL_PACK_ROW_LENGTH, 2);
    glPixelStorei(GL_PACK_SKIP_ROWS, 1);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 1);
    glPixelStorei(GL_PACK_SWAP_BYTES, GL_TRUE);
    glPixelStorei(GL_PACK_LSB_FIRST, GL_TRUE);
    glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_RASTERIZER_DISCARD);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glCullFace(GL_FRONT_AND_BACK);
    glDepthFunc(GL_NEVER);
    glScissor(0, 0, 0, 0);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    const bool executed = executor.execute(*admitted, light, frame, error);
    std::array<unsigned char, 4> restoredColorMask {};
    glGetBooleanv(GL_COLOR_WRITEMASK, restoredColorMask.data());
    int restoredPixelPackBuffer = 0;
    int restoredPackAlignment = 0;
    int restoredPackRowLength = 0;
    int restoredPackSkipRows = 0;
    int restoredPackSkipPixels = 0;
    int restoredPackSwapBytes = 0;
    int restoredPackLsbFirst = 0;
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &restoredPixelPackBuffer);
    glGetIntegerv(GL_PACK_ALIGNMENT, &restoredPackAlignment);
    glGetIntegerv(GL_PACK_ROW_LENGTH, &restoredPackRowLength);
    glGetIntegerv(GL_PACK_SKIP_ROWS, &restoredPackSkipRows);
    glGetIntegerv(GL_PACK_SKIP_PIXELS, &restoredPackSkipPixels);
    glGetIntegerv(GL_PACK_SWAP_BYTES, &restoredPackSwapBytes);
    glGetIntegerv(GL_PACK_LSB_FIRST, &restoredPackLsbFirst);
    const bool stateRestored = glIsEnabled(GL_BLEND) == GL_TRUE
        && glIsEnabled(GL_CULL_FACE) == GL_TRUE
        && glIsEnabled(GL_DEPTH_TEST) == GL_TRUE
        && glIsEnabled(GL_SCISSOR_TEST) == GL_TRUE
        && glIsEnabled(GL_RASTERIZER_DISCARD) == GL_TRUE
        && glIsEnabled(GL_FRAMEBUFFER_SRGB) == GL_TRUE
        && restoredPixelPackBuffer == static_cast<int>(retainedPixelPackBuffer)
        && restoredPackAlignment == 8
        && restoredPackRowLength == 2
        && restoredPackSkipRows == 1
        && restoredPackSkipPixels == 1
        && restoredPackSwapBytes == GL_TRUE
        && restoredPackLsbFirst == GL_TRUE
        && std::all_of(restoredColorMask.begin(), restoredColorMask.end(),
                       [](unsigned char value) { return value == GL_FALSE; });
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SWAP_BYTES, GL_FALSE);
    glPixelStorei(GL_PACK_LSB_FIRST, GL_FALSE);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_RASTERIZER_DISCARD);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glCullFace(GL_BACK);
    glDepthFunc(GL_LESS);

    EvaluationInput oracleInput;
    EvaluationFailure oracleFailure = EvaluationFailure::None;
    const auto oracle = evaluate(*admitted, oracleInput, oracleFailure);
    if (!executed || !oracle)
    {
        std::cerr << "physical diffraction execution failed: " << error << '\n';
        return EXIT_FAILURE;
    }

    const auto color = readAttachment(gl, frame, GL_COLOR_ATTACHMENT0);
    const auto energy = readAttachment(gl, frame, GL_COLOR_ATTACHMENT1);
    const auto orders = readAttachment(gl, frame, GL_COLOR_ATTACHMENT2);
    const auto direction = readAttachment(gl, frame, GL_COLOR_ATTACHMENT3);

    bool pass = alternateAdmissionRejected && shaderFailureClosed && stateRestored
        && textureAfterInitialization == static_cast<int>(retainedTexture)
        && frame.generation == 1
        && frame.linearSrgbTexture != 0 && frame.energyLedgerTexture != 0
        && frame.orderLedgerTexture != 0 && frame.directionMomentTexture != 0
        && near(color[0], oracle->outputLinearSrgb.red)
        && near(color[1], oracle->outputLinearSrgb.green)
        && near(color[2], oracle->outputLinearSrgb.blue)
        && near(color[3], 1.0)
        && near(energy[0], oracle->incidentEnergy)
        && near(energy[1], oracle->substrateReflectedEnergy)
        && near(energy[2], oracle->zeroOrderEnergy)
        && near(energy[3], oracle->higherOrderEnergy)
        && near(orders[0], oracle->unresolvedReflectedEnergy)
        && near(orders[1], oracle->absorbedEnergy)
        && near(orders[2], static_cast<double>(oracle->eventCount
                                               - oracle->spectralSampleCount), 0.01)
        && near(orders[3], static_cast<double>(oracle->evanescentOrderCount
                                               + oracle->grazingOrderCount), 0.01)
        && orders[2] > 0.0f && orders[3] > 0.0f
        && near(direction[3], oracle->resolvedReflectedEnergy);

    auto sinusoidalDescription = makeSinusoidalGratingPreset();
    sinusoidalDescription.roughness = {};
    const auto sinusoidal = admit(sinusoidalDescription, admissionError);
    OpenGlPhysicalDiffractionFrame sinusoidalFrame;
    const bool sinusoidalExecuted = sinusoidal
        && executor.execute(*sinusoidal, light, sinusoidalFrame, error);
    const auto sinusoidalOracle = sinusoidal
        ? evaluate(*sinusoidal, oracleInput, oracleFailure) : std::nullopt;
    std::array<float, 4> sinusoidalColor {};
    std::array<float, 4> sinusoidalEnergy {};
    std::array<float, 4> sinusoidalOrders {};
    std::array<float, 4> sinusoidalDirection {};
    if (sinusoidalExecuted)
    {
        sinusoidalColor = readAttachment(gl, sinusoidalFrame, GL_COLOR_ATTACHMENT0);
        sinusoidalEnergy = readAttachment(gl, sinusoidalFrame, GL_COLOR_ATTACHMENT1);
        sinusoidalOrders = readAttachment(gl, sinusoidalFrame, GL_COLOR_ATTACHMENT2);
        sinusoidalDirection = readAttachment(
            gl, sinusoidalFrame, GL_COLOR_ATTACHMENT3);
    }
    pass = pass && sinusoidalExecuted && sinusoidalOracle
        && sinusoidalFrame.generation == 2
        && sinusoidalFrame.linearSrgbTexture == frame.linearSrgbTexture
        && sinusoidalFrame.energyLedgerTexture == frame.energyLedgerTexture
        && near(sinusoidalColor[0], sinusoidalOracle->outputLinearSrgb.red)
        && near(sinusoidalColor[1], sinusoidalOracle->outputLinearSrgb.green)
        && near(sinusoidalColor[2], sinusoidalOracle->outputLinearSrgb.blue)
        && near(sinusoidalColor[3], 1.0)
        && near(sinusoidalEnergy[0], sinusoidalOracle->incidentEnergy)
        && near(sinusoidalEnergy[1], sinusoidalOracle->substrateReflectedEnergy)
        && near(sinusoidalEnergy[2], sinusoidalOracle->zeroOrderEnergy)
        && near(sinusoidalEnergy[3], sinusoidalOracle->higherOrderEnergy)
        && near(sinusoidalOrders[0], sinusoidalOracle->unresolvedReflectedEnergy)
        && near(sinusoidalOrders[1], sinusoidalOracle->absorbedEnergy)
        && near(sinusoidalOrders[2], static_cast<double>(
            sinusoidalOracle->eventCount - sinusoidalOracle->spectralSampleCount), 0.01)
        && near(sinusoidalOrders[3], static_cast<double>(
            sinusoidalOracle->evanescentOrderCount
                + sinusoidalOracle->grazingOrderCount), 0.01)
        && near(sinusoidalDirection[3],
                sinusoidalOracle->resolvedReflectedEnergy);

    auto blazedDescription = makeBlazedGratingPreset();
    blazedDescription.coating = { CoatingModel::Uncoated, 0.0f, {} };
    blazedDescription.roughness = {};
    const auto blazed = admit(blazedDescription, admissionError);
    OpenGlPhysicalDiffractionFrame blazedFrame;
    const bool blazedExecuted = blazed
        && executor.execute(*blazed, light, blazedFrame, error);
    const auto blazedOracle = blazed
        ? evaluate(*blazed, oracleInput, oracleFailure) : std::nullopt;
    std::array<float, 4> blazedColor {};
    std::array<float, 4> blazedEnergy {};
    std::array<float, 4> blazedOrders {};
    std::array<float, 4> blazedDirection {};
    if (blazedExecuted)
    {
        blazedColor = readAttachment(gl, blazedFrame, GL_COLOR_ATTACHMENT0);
        blazedEnergy = readAttachment(gl, blazedFrame, GL_COLOR_ATTACHMENT1);
        blazedOrders = readAttachment(gl, blazedFrame, GL_COLOR_ATTACHMENT2);
        blazedDirection = readAttachment(gl, blazedFrame, GL_COLOR_ATTACHMENT3);
    }
    const auto blazedMoment = blazedOracle
        ? expectedDirectionMoment(*blazedOracle) : std::array<double, 3> {};
    pass = pass && blazedExecuted && blazedOracle
        && blazedFrame.generation == 3
        && blazedFrame.linearSrgbTexture == frame.linearSrgbTexture
        && blazedFrame.energyLedgerTexture == frame.energyLedgerTexture
        && blazedFrame.orderLedgerTexture == frame.orderLedgerTexture
        && blazedFrame.directionMomentTexture == frame.directionMomentTexture
        && near(blazedColor[0], blazedOracle->outputLinearSrgb.red)
        && near(blazedColor[1], blazedOracle->outputLinearSrgb.green)
        && near(blazedColor[2], blazedOracle->outputLinearSrgb.blue)
        && near(blazedColor[3], 1.0)
        && near(blazedEnergy[0], blazedOracle->incidentEnergy)
        && near(blazedEnergy[1], blazedOracle->substrateReflectedEnergy)
        && near(blazedEnergy[2], blazedOracle->zeroOrderEnergy)
        && near(blazedEnergy[3], blazedOracle->higherOrderEnergy)
        && near(blazedOrders[0], blazedOracle->unresolvedReflectedEnergy)
        && near(blazedOrders[1], blazedOracle->absorbedEnergy)
        && near(blazedOrders[2], static_cast<double>(
            blazedOracle->eventCount - blazedOracle->spectralSampleCount), 0.01)
        && near(blazedOrders[3], static_cast<double>(
            blazedOracle->evanescentOrderCount
                + blazedOracle->grazingOrderCount), 0.01)
        && near(blazedDirection[0], blazedMoment[0])
        && near(blazedDirection[1], blazedMoment[1])
        && near(blazedDirection[2], blazedMoment[2])
        && near(blazedDirection[3], blazedOracle->resolvedReflectedEnergy);

    auto crossedDescription = makeCrossedTwoDimensionalGratingPreset();
    const auto crossed = admit(crossedDescription, admissionError);
    OpenGlPhysicalDiffractionFrame crossedFrame;
    const bool crossedExecuted = crossed
        && executor.execute(*crossed, light, crossedFrame, error);
    const auto crossedOracle = crossed
        ? evaluate(*crossed, oracleInput, oracleFailure) : std::nullopt;
    std::array<float, 4> crossedColor {};
    std::array<float, 4> crossedEnergy {};
    std::array<float, 4> crossedOrders {};
    std::array<float, 4> crossedDirection {};
    if (crossedExecuted)
    {
        crossedColor = readAttachment(gl, crossedFrame, GL_COLOR_ATTACHMENT0);
        crossedEnergy = readAttachment(gl, crossedFrame, GL_COLOR_ATTACHMENT1);
        crossedOrders = readAttachment(gl, crossedFrame, GL_COLOR_ATTACHMENT2);
        crossedDirection = readAttachment(gl, crossedFrame, GL_COLOR_ATTACHMENT3);
    }
    const auto crossedMoment = crossedOracle
        ? expectedDirectionMoment(*crossedOracle) : std::array<double, 3> {};
    pass = pass && crossedExecuted && crossedOracle
        && crossedFrame.generation == 4
        && crossedFrame.linearSrgbTexture == frame.linearSrgbTexture
        && crossedFrame.energyLedgerTexture == frame.energyLedgerTexture
        && crossedFrame.orderLedgerTexture == frame.orderLedgerTexture
        && crossedFrame.directionMomentTexture == frame.directionMomentTexture
        && near(crossedColor[0], crossedOracle->outputLinearSrgb.red)
        && near(crossedColor[1], crossedOracle->outputLinearSrgb.green)
        && near(crossedColor[2], crossedOracle->outputLinearSrgb.blue)
        && near(crossedColor[3], 1.0)
        && near(crossedEnergy[0], crossedOracle->incidentEnergy)
        && near(crossedEnergy[1], crossedOracle->substrateReflectedEnergy)
        && near(crossedEnergy[2], crossedOracle->zeroOrderEnergy)
        && near(crossedEnergy[3], crossedOracle->higherOrderEnergy)
        && near(crossedOrders[0], crossedOracle->unresolvedReflectedEnergy)
        && near(crossedOrders[1], crossedOracle->absorbedEnergy)
        && near(crossedOrders[2], static_cast<double>(
            crossedOracle->eventCount - crossedOracle->spectralSampleCount), 0.01)
        && near(crossedOrders[3], static_cast<double>(
            crossedOracle->evanescentOrderCount
                + crossedOracle->grazingOrderCount), 0.01)
        && near(crossedDirection[0], crossedMoment[0])
        && near(crossedDirection[1], crossedMoment[1])
        && near(crossedDirection[2], crossedMoment[2])
        && near(crossedDirection[3], crossedOracle->resolvedReflectedEnergy);

    auto coatedDescription = makeBlazedGratingPreset();
    coatedDescription.roughness = {};
    const auto coatedBlazed = admit(coatedDescription, admissionError);
    OpenGlPhysicalDiffractionFrame unsupportedFrame;
    OpenGlPhysicalDiffractionFrame coatedFrame;
    const bool coatedExecuted = coatedBlazed
        && executor.execute(*coatedBlazed, light, coatedFrame, error);
    const auto coatedOracle = coatedBlazed
        ? evaluate(*coatedBlazed, oracleInput, oracleFailure) : std::nullopt;
    std::array<float, 4> coatedEnergy {};
    std::array<float, 4> coatedOrders {};
    if (coatedExecuted)
    {
        coatedEnergy = readAttachment(gl, coatedFrame, GL_COLOR_ATTACHMENT1);
        coatedOrders = readAttachment(gl, coatedFrame, GL_COLOR_ATTACHMENT2);
    }
    pass = pass && coatedExecuted && coatedOracle
        && near(coatedEnergy[0], coatedOracle->incidentEnergy)
        && near(coatedEnergy[1], coatedOracle->substrateReflectedEnergy)
        && near(coatedEnergy[2], coatedOracle->zeroOrderEnergy)
        && near(coatedEnergy[3], coatedOracle->higherOrderEnergy)
        && near(coatedOrders[0], coatedOracle->unresolvedReflectedEnergy)
        && near(coatedOrders[1], coatedOracle->absorbedEnergy);

    auto roughDescription = description;
    roughDescription.roughness.rmsHeightNanometres = 8.0f;
    const auto roughHeight = admit(roughDescription, admissionError);
    OpenGlPhysicalDiffractionFrame roughFrame;
    const bool roughExecuted = roughHeight
        && executor.execute(*roughHeight, light, roughFrame, error);
    const auto roughOracle = roughHeight
        ? evaluate(*roughHeight, oracleInput, oracleFailure) : std::nullopt;
    std::array<float, 4> roughEnergy {};
    std::array<float, 4> roughOrders {};
    std::array<float, 4> roughDirection {};
    if (roughExecuted)
    {
        roughEnergy = readAttachment(gl, roughFrame, GL_COLOR_ATTACHMENT1);
        roughOrders = readAttachment(gl, roughFrame, GL_COLOR_ATTACHMENT2);
        roughDirection = readAttachment(
            gl, roughFrame, GL_COLOR_ATTACHMENT3);
    }
    pass = pass && roughExecuted && roughOracle
        && near(roughEnergy[1], roughOracle->substrateReflectedEnergy)
        && near(roughEnergy[2], roughOracle->zeroOrderEnergy)
        && near(roughEnergy[3], roughOracle->higherOrderEnergy)
        && near(roughOrders[0], roughOracle->unresolvedReflectedEnergy)
        && roughOrders[0] > orders[0];

    roughDescription.roughness.rmsSlope = 0.08f;
    const auto roughSlope = admit(roughDescription, admissionError);
    OpenGlPhysicalDiffractionFrame roughSlopeFrame;
    const bool roughSlopeExecuted = roughSlope
        && executor.execute(*roughSlope, light, roughSlopeFrame, error);
    const auto roughSlopeOracle = roughSlope
        ? evaluate(*roughSlope, oracleInput, oracleFailure) : std::nullopt;
    std::array<float, 4> roughSlopeEnergy {};
    std::array<float, 4> roughSlopeDirection {};
    if (roughSlopeExecuted)
    {
        roughSlopeEnergy = readAttachment(
            gl, roughSlopeFrame, GL_COLOR_ATTACHMENT1);
        roughSlopeDirection = readAttachment(
            gl, roughSlopeFrame, GL_COLOR_ATTACHMENT3);
    }
    const auto roughSlopeMoment = roughSlopeOracle
        ? expectedDirectionMoment(*roughSlopeOracle) : std::array<double, 3> {};
    const auto smoothMomentLength = std::sqrt(
        roughDirection[0] * roughDirection[0]
            + roughDirection[1] * roughDirection[1]
            + roughDirection[2] * roughDirection[2]);
    const auto roughMomentLength = std::sqrt(
        roughSlopeDirection[0] * roughSlopeDirection[0]
            + roughSlopeDirection[1] * roughSlopeDirection[1]
            + roughSlopeDirection[2] * roughSlopeDirection[2]);
    pass = pass && roughSlopeExecuted && roughSlopeOracle
        && near(roughSlopeEnergy[0], roughSlopeOracle->incidentEnergy)
        && near(roughSlopeEnergy[1], roughSlopeOracle->substrateReflectedEnergy)
        && near(roughSlopeEnergy[2], roughSlopeOracle->zeroOrderEnergy)
        && near(roughSlopeEnergy[3], roughSlopeOracle->higherOrderEnergy)
        && near(roughSlopeDirection[0], roughSlopeMoment[0])
        && near(roughSlopeDirection[1], roughSlopeMoment[1])
        && near(roughSlopeDirection[2], roughSlopeMoment[2])
        && near(roughSlopeDirection[3], roughSlopeOracle->resolvedReflectedEnergy)
        && roughMomentLength < smoothMomentLength;

    auto invalidLight = light;
    invalidLight.radiance[3] = -1.0f;
    pass = pass
        && !executor.execute(*admitted, invalidLight, unsupportedFrame, error)
        && error == "physical diffraction OpenGL incident spectrum is invalid";

    for (const auto& sample : spatialParitySamples())
    {
        const auto spatialMaterial = admit(sample.description, admissionError);
        EvaluationFailure spatialFailure = EvaluationFailure::None;
        const auto spatialOracle = spatialMaterial
            ? evaluate(*spatialMaterial, sample.input, spatialFailure) : std::nullopt;
        OpenGlPhysicalDiffractionFrame spatialFrame;
        const std::array<float, 2> materialUv {
            static_cast<float>(sample.input.materialUv[0]),
            static_cast<float>(sample.input.materialUv[1])
        };
        const bool spatialExecuted = spatialMaterial
            && executor.executeAtMaterialUv(
                *spatialMaterial, light, materialUv, spatialFrame, error);
        std::array<float, 4> spatialColor {};
        std::array<float, 4> spatialEnergy {};
        std::array<float, 4> spatialOrders {};
        std::array<float, 4> spatialDirection {};
        if (spatialExecuted)
        {
            spatialColor = readAttachment(gl, spatialFrame, GL_COLOR_ATTACHMENT0);
            spatialEnergy = readAttachment(gl, spatialFrame, GL_COLOR_ATTACHMENT1);
            spatialOrders = readAttachment(gl, spatialFrame, GL_COLOR_ATTACHMENT2);
            spatialDirection = readAttachment(
                gl, spatialFrame, GL_COLOR_ATTACHMENT3);
        }
        const auto moment = spatialOracle
            ? expectedDirectionMoment(*spatialOracle) : std::array<double, 3> {};
        const bool sampleMatches = spatialExecuted && spatialOracle
            && near(spatialColor[0], spatialOracle->outputLinearSrgb.red)
            && near(spatialColor[1], spatialOracle->outputLinearSrgb.green)
            && near(spatialColor[2], spatialOracle->outputLinearSrgb.blue)
            && near(spatialEnergy[0], spatialOracle->incidentEnergy)
            && near(spatialEnergy[1], spatialOracle->substrateReflectedEnergy)
            && near(spatialEnergy[2], spatialOracle->zeroOrderEnergy)
            && near(spatialEnergy[3], spatialOracle->higherOrderEnergy)
            && near(spatialOrders[0], spatialOracle->unresolvedReflectedEnergy)
            && near(spatialOrders[1], spatialOracle->absorbedEnergy)
            && near(spatialDirection[0], moment[0])
            && near(spatialDirection[1], moment[1])
            && near(spatialDirection[2], moment[2])
            && near(spatialDirection[3], spatialOracle->resolvedReflectedEnergy);
        if (!sampleMatches)
            std::cerr << "OpenGL spatial parity failed: " << sample.name
                      << " oracle=" << token(spatialFailure) << '\n';
        pass = pass && sampleMatches;
    }

    for (const auto& scene : validation::wavelengthValidationScenes())
    {
        const auto material = admit(scene.description, admissionError);
        SpectralIncidentLight sceneLight;
        sceneLight.direction = {
            static_cast<float>(scene.input.incidentDirection.x),
            static_cast<float>(scene.input.incidentDirection.y),
            static_cast<float>(scene.input.incidentDirection.z)
        };
        for (std::size_t index = 0; index < sceneLight.radiance.size(); ++index)
            sceneLight.radiance[index] = static_cast<float>(scene.input.incidentSpectrum[index]);
        OpenGlPhysicalDiffractionFrame validationFrame;
        const std::array<float, 2> materialUv {
            static_cast<float>(scene.input.materialUv[0]),
            static_cast<float>(scene.input.materialUv[1])
        };
        const bool sceneExecuted = material
            && executor.executeAtMaterialUv(
                *material, sceneLight, materialUv, validationFrame, error);
        EvaluationFailure sceneFailure = EvaluationFailure::None;
        const auto sceneOracle = material
            ? evaluate(*material, scene.input, sceneFailure) : std::nullopt;
        std::array<float, 4> sceneColor {};
        std::array<float, 4> sceneEnergy {};
        std::array<float, 4> sceneOrders {};
        std::array<float, 4> sceneDirection {};
        if (sceneExecuted)
        {
            sceneColor = readAttachment(gl, validationFrame, GL_COLOR_ATTACHMENT0);
            sceneEnergy = readAttachment(gl, validationFrame, GL_COLOR_ATTACHMENT1);
            sceneOrders = readAttachment(gl, validationFrame, GL_COLOR_ATTACHMENT2);
            sceneDirection = readAttachment(
                gl, validationFrame, GL_COLOR_ATTACHMENT3);
        }
        const auto sceneMoment = sceneOracle
            ? expectedDirectionMoment(*sceneOracle) : std::array<double, 3> {};
        const bool sceneMatches = sceneExecuted && sceneOracle
            && near(sceneColor[0], scene.expectedZeroClampedLinearSrgb[0],
                    validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError)
            && near(sceneColor[1], scene.expectedZeroClampedLinearSrgb[1],
                    validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError)
            && near(sceneColor[2], scene.expectedZeroClampedLinearSrgb[2],
                    validation::kFiniteSceneMaximumZeroClampedLinearSrgbChannelError)
            && near(sceneEnergy[0], sceneOracle->incidentEnergy)
            && near(sceneEnergy[1], sceneOracle->substrateReflectedEnergy)
            && near(sceneEnergy[2], sceneOracle->zeroOrderEnergy)
            && near(sceneEnergy[3], sceneOracle->higherOrderEnergy)
            && near(sceneOrders[0], sceneOracle->unresolvedReflectedEnergy)
            && near(sceneOrders[1], sceneOracle->absorbedEnergy)
            && near(sceneOrders[2], static_cast<double>(
                sceneOracle->eventCount - sceneOracle->spectralSampleCount), 0.01)
            && near(sceneOrders[3], static_cast<double>(
                sceneOracle->evanescentOrderCount
                    + sceneOracle->grazingOrderCount), 0.01)
            && near(sceneDirection[0], sceneMoment[0])
            && near(sceneDirection[1], sceneMoment[1])
            && near(sceneDirection[2], sceneMoment[2])
            && near(sceneDirection[3], sceneOracle->resolvedReflectedEnergy);
        if (!sceneMatches)
            std::cerr << "OpenGL wavelength validation failed: " << scene.name
                      << " color=" << sceneColor[0] << ',' << sceneColor[1] << ','
                      << sceneColor[2] << " energy=" << sceneEnergy[0] << ','
                      << sceneEnergy[1] << ',' << sceneEnergy[2] << ','
                      << sceneEnergy[3] << " orders=" << sceneOrders[2] << '/'
                      << sceneOrders[3] << " direction=" << sceneDirection[0] << ','
                      << sceneDirection[1] << ',' << sceneDirection[2]
                      << " oracle=" << token(sceneFailure)
                      << " error=" << error << '\n';
        pass = pass && sceneMatches;
    }

    const std::array<unsigned, 4> textures {
        frame.linearSrgbTexture, frame.energyLedgerTexture,
        frame.orderLedgerTexture, frame.directionMomentTexture
    };
    executor.shutdown();
    gl.DeleteBuffers(1, &retainedPixelPackBuffer);
    glDeleteTextures(1, &retainedTexture);
    for (const auto texture : textures)
        pass = pass && glIsTexture(texture) == GL_FALSE;
    pass = pass && glGetError() == GL_NO_ERROR;

    glfwDestroyWindow(window);
    glfwTerminate();
    if (!pass)
    {
        std::cerr << "physical diffraction native OpenGL FAIL\n"
                  << "color=" << color[0] << ',' << color[1] << ',' << color[2]
                  << " energy=" << energy[0] << ',' << energy[1] << ','
                  << energy[2] << ',' << energy[3]
                  << " orders=" << orders[2] << '/' << orders[3]
                  << " error=" << error << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "physical diffraction native OpenGL PASS: finite wavelength scenes, binary, sinusoidal, and "
                 "blazed Fourier orders, incoherent coating, RMS-height attenuation, "
                 "RMS-slope broadening, "
                 "eight wavelengths, energy ledgers, and CIE output verified\n";
    return EXIT_SUCCESS;
#endif
}
