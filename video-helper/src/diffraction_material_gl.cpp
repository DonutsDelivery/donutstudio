#include "diffraction_material_gl.h"
#include "../../shared/DiffractionMaterialGpuLayout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace diffractionmaterial
{
namespace
{

constexpr const char* kVertexShader = R"GLSL(#version 330 core
void main()
{
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

constexpr const char* kFragmentShader = R"GLSL(#version 330 core
layout(location = 0) out vec4 outLinearSrgb;
layout(location = 1) out vec4 outEnergy;
layout(location = 2) out vec4 outOrders;
layout(location = 3) out vec4 outDirectionMoment;

uniform vec4 uGeometry;       // reciprocal direction xy, spacing nm, depth nm
uniform vec4 uSecondaryGeometry; // reciprocal direction xy, spacing nm, lattice
uniform vec4 uMicrostructure; // duty cycle, substrate n, substrate k, first order
uniform vec4 uControl;        // last order, profile, reserved, coating model
uniform vec3 uIncident;
uniform vec4 uCoating;        // thickness nm, coating n/k, reserved
uniform vec4 uRoughness;      // RMS height nm, RMS slope, reserved
uniform vec4 uGrooveField;    // mode, origin uv, orientation degrees/unit
uniform vec4 uGrooveVariation; // axis uv, primary/secondary period delta
uniform vec4 uSpectral[8];    // wavelength nm, spectral radiance, CIE x, CIE y
uniform vec4 uSpectralZ[8];   // CIE z, normalized quadrature weight, reserved

const float pi = 3.14159265358979323846;
const int profileBinaryRectangular = 1;
const int profileSinusoidal = 2;
const int profileBlazedSawtooth = 3;
const int coatingUncoated = 1;
const int coatingIncoherentDielectric = 2;

float interfaceReflectance(vec2 first, vec2 second)
{
    vec2 difference = first - second;
    vec2 sum = first + second;
    return clamp(dot(difference, difference) / dot(sum, sum), 0.0, 1.0);
}

float spectralReflectance(float wavelength, int coatingModel,
                          float substrateN, float substrateK)
{
    if (coatingModel == coatingUncoated)
        return interfaceReflectance(vec2(1.0, 0.0), vec2(substrateN, substrateK));
    if (coatingModel != coatingIncoherentDielectric)
        return -1.0;

    vec2 layer = vec2(uCoating.y, uCoating.z);
    float r01 = interfaceReflectance(vec2(1.0, 0.0), layer);
    float r12 = interfaceReflectance(layer, vec2(substrateN, substrateK));
    float roundTrip = exp(-8.0 * pi * uCoating.z * uCoating.x / wavelength);
    float denominator = 1.0 - r01 * r12 * roundTrip;
    return clamp(r01 + (1.0 - r01) * (1.0 - r01)
                    * r12 * roundTrip / denominator, 0.0, 1.0);
}

float coherentRoughnessFraction(float wavelength, float outgoingCosine)
{
    float argument = 2.0 * pi * uRoughness.x
        * (uIncident.z + outgoingCosine) / wavelength;
    return exp(-(argument * argument));
}

vec2 complexMultiply(vec2 a, vec2 b)
{
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 complexExponentialIntegral(float frequency, float begin, float end)
{
    if (abs(frequency) < 1.0e-6)
        return vec2(end - begin, 0.0);
    float beginPhase = frequency * begin;
    float endPhase = frequency * end;
    return vec2(
        (sin(endPhase) - sin(beginPhase)) / frequency,
        -(cos(endPhase) - cos(beginPhase)) / frequency);
}

// Integer-order Bessel J_n power series. Native admission bounds its argument
// to four, where 24 terms keep the float result inside the readback tolerance.
float besselJ(int order, float argument)
{
    int magnitude = order < 0 ? -order : order;
    float halfArgument = 0.5 * argument;
    float term = 1.0;
    for (int factor = 1; factor <= 8; ++factor)
    {
        if (factor <= magnitude)
            term *= halfArgument / float(factor);
    }
    float sum = term;
    for (int seriesIndex = 1; seriesIndex <= 24; ++seriesIndex)
    {
        term *= -(halfArgument * halfArgument)
              / (float(seriesIndex) * float(magnitude + seriesIndex));
        sum += term;
    }
    return sum;
}

float profileEfficiency(int profile, int signedOrder, float duty,
                        float phase, vec2 terracePhase)
{
    if (profile == profileBinaryRectangular)
    {
        if (signedOrder == 0)
        {
            vec2 amplitude = vec2(1.0 - duty, 0.0) + duty * terracePhase;
            return dot(amplitude, amplitude);
        }
        float order = float(signedOrder);
        float aperture = sin(pi * order * duty) / (pi * order);
        float rotation = -pi * order * duty;
        vec2 amplitude = aperture * complexMultiply(
            terracePhase - vec2(1.0, 0.0),
            vec2(cos(rotation), sin(rotation)));
        return dot(amplitude, amplitude);
    }
    if (profile == profileSinusoidal)
    {
        float amplitude = besselJ(signedOrder, 0.5 * phase);
        return amplitude * amplitude;
    }
    if (profile == profileBlazedSawtooth)
    {
        float orderFrequency = -2.0 * pi * float(signedOrder);
        vec2 ramp = complexExponentialIntegral(
            phase / duty + orderFrequency, 0.0, duty);
        vec2 land = complexExponentialIntegral(orderFrequency, duty, 1.0);
        vec2 amplitude = ramp + land;
        return dot(amplitude, amplitude);
    }
    return -1.0;
}

void main()
{
    float duty = uMicrostructure.x;
    float n = uMicrostructure.y;
    float k = uMicrostructure.z;
    int firstOrder = int(uMicrostructure.w + 0.5);
    int lastOrder = int(uControl.x + 0.5);
    int profile = int(uControl.y + 0.5);
    int coatingModel = int(uControl.w + 0.5);
    bool crossedTwoDimensional = int(uSecondaryGeometry.w + 0.5) == 2;
    vec4 localGeometry = uGeometry;
    vec4 localSecondaryGeometry = uSecondaryGeometry;
    int fieldMode = int(uGrooveField.x + 0.5);
    if (fieldMode != 1)
    {
        vec2 offset = vec2(0.5) - uGrooveField.yz;
        float coordinate = fieldMode == 2
            ? dot(offset, uGrooveVariation.xy) : length(offset);
        localGeometry.z += coordinate * uGrooveVariation.z;
        localSecondaryGeometry.z += coordinate * uGrooveVariation.w;
        float angle = radians(coordinate * uGrooveField.w);
        mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
        localGeometry.xy = rotation * localGeometry.xy;
        localSecondaryGeometry.xy = rotation * localSecondaryGeometry.xy;
    }

    vec3 xyz = vec3(0.0);
    vec3 directionMoment = vec3(0.0);
    float referenceWhiteY = 0.0;
    float incidentTotal = 0.0;
    float reflectedTotal = 0.0;
    float zeroTotal = 0.0;
    float higherTotal = 0.0;
    float unresolvedTotal = 0.0;
    float absorbedTotal = 0.0;
    float propagatingCount = 0.0;
    float rejectedCount = 0.0;
    // A Gaussian angular deviation with sigma=2*rmsSlope has a first
    // directional moment of exp(-sigma^2/2). Energy remains in w unchanged.
    float broadeningMoment = exp(-2.0 * uRoughness.y * uRoughness.y);

    for (int wavelengthIndex = 0; wavelengthIndex < 8; ++wavelengthIndex)
    {
        vec4 spectral = uSpectral[wavelengthIndex];
        vec4 spectralZ = uSpectralZ[wavelengthIndex];
        float wavelength = spectral.x;
        float quadrature = spectralZ.y;
        float reflectance = spectralReflectance(wavelength, coatingModel, n, k);
        float zeroCoherent = coherentRoughnessFraction(wavelength, uIncident.z);
        float zeroPhase = 4.0 * pi * uGeometry.w * uIncident.z / wavelength;
        vec2 zeroTerracePhase = vec2(cos(zeroPhase), sin(zeroPhase));

        // Binary terraces, sinusoidal phase, and the blazed ramp use their
        // analytic complex Fourier coefficients for every integer order.
        float zeroProfileEfficiency = profileEfficiency(
            profile, 0, duty, zeroPhase, zeroTerracePhase);
        float zeroEfficiency = zeroCoherent * zeroProfileEfficiency
            * (crossedTwoDimensional ? zeroProfileEfficiency : 1.0);
        float higherEfficiency = 0.0;
        vec3 wavelengthDirectionMoment = zeroEfficiency
            * vec3(-uIncident.xy, uIncident.z);

        for (int primaryOrder = -8; primaryOrder <= 8; ++primaryOrder)
        {
            int primaryMagnitude = primaryOrder < 0 ? -primaryOrder : primaryOrder;
            bool primaryAdmitted = primaryOrder == 0
                || (primaryMagnitude >= firstOrder && primaryMagnitude <= lastOrder);
            if (!primaryAdmitted)
                continue;
            for (int secondaryOrder = -8; secondaryOrder <= 8; ++secondaryOrder)
            {
                int secondaryMagnitude = secondaryOrder < 0
                    ? -secondaryOrder : secondaryOrder;
                bool secondaryAdmitted = secondaryOrder == 0
                    || (secondaryMagnitude >= firstOrder && secondaryMagnitude <= lastOrder);
                if (!secondaryAdmitted || (!crossedTwoDimensional && secondaryOrder != 0)
                    || (primaryOrder == 0 && secondaryOrder == 0))
                    continue;

                vec2 tangent = -uIncident.xy
                    + float(primaryOrder) * wavelength
                        / localGeometry.z * localGeometry.xy;
                if (crossedTwoDimensional)
                    tangent += float(secondaryOrder) * wavelength
                        / localSecondaryGeometry.z * localSecondaryGeometry.xy;
                float tangentSquared = dot(tangent, tangent);
                if (tangentSquared >= 1.0 - 1.0e-12)
                {
                    rejectedCount += 1.0;
                    continue;
                }

                vec3 outgoing = vec3(tangent, sqrt(max(0.0, 1.0 - tangentSquared)));
                float phase = 2.0 * pi * uGeometry.w
                    * (uIncident.z + outgoing.z) / wavelength;
                vec2 terracePhase = vec2(cos(phase), sin(phase));
                float efficiency = coherentRoughnessFraction(wavelength, outgoing.z)
                    * profileEfficiency(
                        profile, primaryOrder, duty, phase, terracePhase);
                if (crossedTwoDimensional)
                    efficiency *= profileEfficiency(
                        profile, secondaryOrder, duty, phase, terracePhase);
                efficiency *= outgoing.z / uIncident.z;
                higherEfficiency += efficiency;
                wavelengthDirectionMoment += efficiency * outgoing;
                propagatingCount += 1.0;
            }
        }

        float resolvedEfficiency = zeroEfficiency + higherEfficiency;
        if (!(reflectance >= 0.0) || reflectance > 1.0
            || !(zeroCoherent >= 0.0) || zeroCoherent > 1.0
            || !(resolvedEfficiency >= 0.0) || resolvedEfficiency > 1.0001)
        {
            outLinearSrgb = vec4(0.0, 0.0, 0.0, -1.0);
            outEnergy = vec4(-1.0);
            outOrders = vec4(-1.0);
            outDirectionMoment = vec4(-1.0);
            return;
        }

        float incidentEnergy = spectral.y * quadrature;
        float reflectedEnergy = incidentEnergy * reflectance;
        float zeroEnergy = reflectedEnergy * zeroEfficiency;
        float higherEnergy = reflectedEnergy * higherEfficiency;
        float unresolvedEnergy = reflectedEnergy * max(0.0, 1.0 - resolvedEfficiency);
        float absorbedEnergy = incidentEnergy - reflectedEnergy;
        float resolvedEnergy = zeroEnergy + higherEnergy;

        incidentTotal += incidentEnergy;
        reflectedTotal += reflectedEnergy;
        zeroTotal += zeroEnergy;
        higherTotal += higherEnergy;
        unresolvedTotal += unresolvedEnergy;
        absorbedTotal += absorbedEnergy;
        directionMoment += broadeningMoment
            * reflectedEnergy * wavelengthDirectionMoment;
        xyz += resolvedEnergy * vec3(spectral.z, spectral.w, spectralZ.x);
        referenceWhiteY += quadrature * spectral.w;
    }

    xyz /= referenceWhiteY;
    // Exactly one output conversion follows the complete spectral accumulation.
    vec3 linearSrgb = max(vec3(0.0), mat3(
         3.2406, -0.9689,  0.0557,
        -1.5372,  1.8758, -0.2040,
        -0.4986,  0.0415,  1.0570) * xyz);

    outLinearSrgb = vec4(linearSrgb, 1.0);
    outEnergy = vec4(incidentTotal, reflectedTotal, zeroTotal, higherTotal);
    outOrders = vec4(unresolvedTotal, absorbedTotal, propagatingCount, rejectedCount);
    outDirectionMoment = vec4(directionMoment, zeroTotal + higherTotal);
}
)GLSL";

unsigned compileShader(arbitgl::GlFuncs& gl, unsigned type,
                       const char* source, std::string& error)
{
    const auto shader = gl.CreateShader(type);
    if (shader == 0)
    {
        error = "physical diffraction OpenGL shader allocation failed";
        return 0;
    }
    gl.ShaderSource(shader, 1, &source, nullptr);
    gl.CompileShader(shader);
    int compiled = 0;
    gl.GetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE)
        return shader;

    std::array<char, 2048> log {};
    int length = 0;
    gl.GetShaderInfoLog(shader, static_cast<int>(log.size()), &length, log.data());
    error = "physical diffraction OpenGL shader compile failed: ";
    error.append(log.data(), static_cast<std::size_t>(std::max(length, 0)));
    gl.DeleteShader(shader);
    return 0;
}

} // namespace

bool OpenGlPhysicalDiffractionExecutor::initialize(arbitgl::GlFuncs* gl,
                                                    std::string& error)
{
    shutdown();
    error.clear();
    if (gl == nullptr || gl->Uniform4fv == nullptr)
    {
        error = "physical diffraction OpenGL requires a complete GL 3.3 function table";
        return false;
    }

    const auto vertex = compileShader(*gl, GL_VERTEX_SHADER, kVertexShader, error);
    if (vertex == 0)
        return false;
    const auto fragment = compileShader(*gl, GL_FRAGMENT_SHADER, kFragmentShader, error);
    if (fragment == 0)
    {
        gl->DeleteShader(vertex);
        return false;
    }

    const auto program = gl->CreateProgram();
    gl->AttachShader(program, vertex);
    gl->AttachShader(program, fragment);
    gl->LinkProgram(program);
    gl->DeleteShader(vertex);
    gl->DeleteShader(fragment);
    int linked = 0;
    gl->GetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE)
    {
        std::array<char, 2048> log {};
        int length = 0;
        gl->GetProgramInfoLog(program, static_cast<int>(log.size()), &length, log.data());
        error = "physical diffraction OpenGL program link failed: ";
        error.append(log.data(), static_cast<std::size_t>(std::max(length, 0)));
        gl->DeleteProgram(program);
        return false;
    }

    unsigned vao = 0;
    unsigned framebuffer = 0;
    std::array<unsigned, 4> textures {};
    int previousDrawFramebuffer = 0;
    int previousReadFramebuffer = 0;
    int previousTexture = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture);
    gl->GenVertexArrays(1, &vao);
    gl->GenFramebuffers(1, &framebuffer);
    glGenTextures(static_cast<int>(textures.size()), textures.data());
    gl->BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    const std::array<unsigned, 4> attachments {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3
    };
    for (std::size_t index = 0; index < textures.size(); ++index)
    {
        glBindTexture(GL_TEXTURE_2D, textures[index]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, nullptr);
        gl->FramebufferTexture2D(GL_FRAMEBUFFER, attachments[index], GL_TEXTURE_2D,
                                 textures[index], 0);
    }
    gl->DrawBuffers(static_cast<int>(attachments.size()), attachments.data());
    if (gl->CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        glDeleteTextures(static_cast<int>(textures.size()), textures.data());
        gl->DeleteFramebuffers(1, &framebuffer);
        gl->DeleteVertexArrays(1, &vao);
        gl->DeleteProgram(program);
        glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(previousTexture));
        gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER,
                            static_cast<unsigned>(previousDrawFramebuffer));
        gl->BindFramebuffer(GL_READ_FRAMEBUFFER,
                            static_cast<unsigned>(previousReadFramebuffer));
        error = "physical diffraction OpenGL framebuffer is incomplete";
        return false;
    }
    glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(previousTexture));
    gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER,
                        static_cast<unsigned>(previousDrawFramebuffer));
    gl->BindFramebuffer(GL_READ_FRAMEBUFFER,
                        static_cast<unsigned>(previousReadFramebuffer));

    gl_ = gl;
    program_ = program;
    vao_ = vao;
    framebuffer_ = framebuffer;
    textures_ = textures;
    return true;
}

bool OpenGlPhysicalDiffractionExecutor::validateCheckpoint(
    const AdmittedDiffractionMaterialIR& material,
    const SpectralIncidentLight& light,
    std::string& error) const noexcept
{
    return physicalcheckpoint::validate(material, light, "OpenGL", error);
}

bool OpenGlPhysicalDiffractionExecutor::execute(
    const AdmittedDiffractionMaterialIR& material,
    const SpectralIncidentLight& light,
    OpenGlPhysicalDiffractionFrame& output,
    std::string& error)
{
    return executeAtMaterialUv(material, light, { 0.5f, 0.5f }, output, error);
}

bool OpenGlPhysicalDiffractionExecutor::executeAtMaterialUv(
    const AdmittedDiffractionMaterialIR& material,
    const SpectralIncidentLight& light,
    const std::array<float, 2>& materialUv,
    OpenGlPhysicalDiffractionFrame& output,
    std::string& error)
{
    output = {};
    error.clear();
    if (!ready())
    {
        error = "physical diffraction native OpenGL backend is unavailable";
        return false;
    }
    if (!validateCheckpoint(material, light, error))
        return false;

    auto parameters = physicalcheckpoint::makeGpuParameters(material, light);
    if (!std::isfinite(materialUv[0]) || !std::isfinite(materialUv[1])
        || materialUv[0] < 0.0f || materialUv[0] > 1.0f
        || materialUv[1] < 0.0f || materialUv[1] > 1.0f)
    {
        error = "physical diffraction OpenGL material UV is invalid";
        return false;
    }
    parameters.grooveField.y += 0.5f - materialUv[0];
    parameters.grooveField.z += 0.5f - materialUv[1];

    int previousProgram = 0;
    int previousVao = 0;
    int previousDrawFramebuffer = 0;
    int previousReadFramebuffer = 0;
    int previousPixelPackBuffer = 0;
    int previousPackAlignment = 0;
    int previousPackRowLength = 0;
    int previousPackSkipRows = 0;
    int previousPackSkipPixels = 0;
    int previousPackSwapBytes = 0;
    int previousPackLsbFirst = 0;
    int previousViewport[4] {};
    bool previousBlend = glIsEnabled(GL_BLEND) == GL_TRUE;
    bool previousCullFace = glIsEnabled(GL_CULL_FACE) == GL_TRUE;
    bool previousDepthTest = glIsEnabled(GL_DEPTH_TEST) == GL_TRUE;
    bool previousScissorTest = glIsEnabled(GL_SCISSOR_TEST) == GL_TRUE;
    bool previousRasterizerDiscard = glIsEnabled(GL_RASTERIZER_DISCARD) == GL_TRUE;
    bool previousFramebufferSrgb = glIsEnabled(GL_FRAMEBUFFER_SRGB) == GL_TRUE;
    std::array<unsigned char, 4> previousColorMask {};
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previousVao);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &previousPixelPackBuffer);
    glGetIntegerv(GL_PACK_ALIGNMENT, &previousPackAlignment);
    glGetIntegerv(GL_PACK_ROW_LENGTH, &previousPackRowLength);
    glGetIntegerv(GL_PACK_SKIP_ROWS, &previousPackSkipRows);
    glGetIntegerv(GL_PACK_SKIP_PIXELS, &previousPackSkipPixels);
    glGetIntegerv(GL_PACK_SWAP_BYTES, &previousPackSwapBytes);
    glGetIntegerv(GL_PACK_LSB_FIRST, &previousPackLsbFirst);
    glGetIntegerv(GL_VIEWPORT, previousViewport);
    glGetBooleanv(GL_COLOR_WRITEMASK, previousColorMask.data());

    gl_->BindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    const std::array<unsigned, 4> attachments {
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3
    };
    gl_->DrawBuffers(static_cast<int>(attachments.size()), attachments.data());
    glViewport(0, 0, 1, 1);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_RASTERIZER_DISCARD);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    gl_->UseProgram(program_);
    gl_->BindVertexArray(vao_);

    const auto geometryLocation = gl_->GetUniformLocation(program_, "uGeometry");
    const auto secondaryGeometryLocation
        = gl_->GetUniformLocation(program_, "uSecondaryGeometry");
    const auto microstructureLocation = gl_->GetUniformLocation(program_, "uMicrostructure");
    const auto controlLocation = gl_->GetUniformLocation(program_, "uControl");
    const auto incidentLocation = gl_->GetUniformLocation(program_, "uIncident");
    const auto coatingLocation = gl_->GetUniformLocation(program_, "uCoating");
    const auto roughnessLocation = gl_->GetUniformLocation(program_, "uRoughness");
    const auto grooveFieldLocation = gl_->GetUniformLocation(program_, "uGrooveField");
    const auto grooveVariationLocation
        = gl_->GetUniformLocation(program_, "uGrooveVariation");
    const auto spectralLocation = gl_->GetUniformLocation(program_, "uSpectral[0]");
    const auto spectralZLocation = gl_->GetUniformLocation(program_, "uSpectralZ[0]");
    gl_->Uniform4f(geometryLocation,
                  parameters.geometry.x, parameters.geometry.y,
                  parameters.geometry.z, parameters.geometry.w);
    gl_->Uniform4f(secondaryGeometryLocation,
                  parameters.secondaryGeometry.x, parameters.secondaryGeometry.y,
                  parameters.secondaryGeometry.z, parameters.secondaryGeometry.w);
    gl_->Uniform4f(microstructureLocation,
                  parameters.microstructure.x, parameters.microstructure.y,
                  parameters.microstructure.z, parameters.microstructure.w);
    gl_->Uniform4f(controlLocation,
                  parameters.control.x, parameters.control.y,
                  parameters.control.z, parameters.control.w);
    gl_->Uniform3f(incidentLocation,
                  parameters.incident.x, parameters.incident.y,
                  parameters.incident.z);
    gl_->Uniform4f(coatingLocation,
                  parameters.coating.x, parameters.coating.y,
                  parameters.coating.z, parameters.coating.w);
    gl_->Uniform4f(roughnessLocation,
                  parameters.roughness.x, parameters.roughness.y,
                  parameters.roughness.z, parameters.roughness.w);
    gl_->Uniform4f(grooveFieldLocation,
                  parameters.grooveField.x, parameters.grooveField.y,
                  parameters.grooveField.z, parameters.grooveField.w);
    gl_->Uniform4f(grooveVariationLocation,
                  parameters.grooveVariation.x, parameters.grooveVariation.y,
                  parameters.grooveVariation.z, parameters.grooveVariation.w);
    gl_->Uniform4fv(spectralLocation, static_cast<int>(kMaximumSpectralSamples),
                    &parameters.spectral[0].x);
    gl_->Uniform4fv(spectralZLocation, static_cast<int>(kMaximumSpectralSamples),
                    &parameters.spectralZ[0].x);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    const auto drawError = glGetError();
    std::array<std::array<float, 4>, 4> validationPixels {};
    gl_->BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SWAP_BYTES, GL_FALSE);
    glPixelStorei(GL_PACK_LSB_FIRST, GL_FALSE);
    bool validationReadbackOk = true;
    for (std::size_t index = 0; index < validationPixels.size(); ++index)
    {
        glReadBuffer(attachments[index]);
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT,
                     validationPixels[index].data());
        validationReadbackOk = validationReadbackOk && glGetError() == GL_NO_ERROR;
    }
    const bool validatedShaderOutput = validationReadbackOk
        && physicalcheckpoint::validateGpuReadback(
            validationPixels[0], validationPixels[1], validationPixels[2],
            validationPixels[3],
            physicalcheckpoint::expectedSignedOrderEvaluations(material));

    gl_->BindVertexArray(static_cast<unsigned>(previousVao));
    gl_->UseProgram(static_cast<unsigned>(previousProgram));
    gl_->BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<unsigned>(previousPixelPackBuffer));
    glPixelStorei(GL_PACK_ALIGNMENT, previousPackAlignment);
    glPixelStorei(GL_PACK_ROW_LENGTH, previousPackRowLength);
    glPixelStorei(GL_PACK_SKIP_ROWS, previousPackSkipRows);
    glPixelStorei(GL_PACK_SKIP_PIXELS, previousPackSkipPixels);
    glPixelStorei(GL_PACK_SWAP_BYTES, previousPackSwapBytes);
    glPixelStorei(GL_PACK_LSB_FIRST, previousPackLsbFirst);
    gl_->BindFramebuffer(GL_DRAW_FRAMEBUFFER,
                         static_cast<unsigned>(previousDrawFramebuffer));
    gl_->BindFramebuffer(GL_READ_FRAMEBUFFER,
                         static_cast<unsigned>(previousReadFramebuffer));
    glViewport(previousViewport[0], previousViewport[1],
               previousViewport[2], previousViewport[3]);
    const auto restoreEnable = [] (unsigned capability, bool enabled)
    {
        if (enabled) glEnable(capability);
        else glDisable(capability);
    };
    restoreEnable(GL_BLEND, previousBlend);
    restoreEnable(GL_CULL_FACE, previousCullFace);
    restoreEnable(GL_DEPTH_TEST, previousDepthTest);
    restoreEnable(GL_SCISSOR_TEST, previousScissorTest);
    restoreEnable(GL_RASTERIZER_DISCARD, previousRasterizerDiscard);
    restoreEnable(GL_FRAMEBUFFER_SRGB, previousFramebufferSrgb);
    glColorMask(previousColorMask[0], previousColorMask[1],
                previousColorMask[2], previousColorMask[3]);

    if (drawError != GL_NO_ERROR)
    {
        error = "physical diffraction OpenGL draw failed";
        return false;
    }
    if (!validatedShaderOutput)
    {
        error = "physical diffraction OpenGL validation readback rejected shader output";
        return false;
    }

    OpenGlPhysicalDiffractionFrame frame;
    frame.generation = ++generation_;
    frame.framebuffer = framebuffer_;
    frame.linearSrgbTexture = textures_[0];
    frame.energyLedgerTexture = textures_[1];
    frame.orderLedgerTexture = textures_[2];
    frame.directionMomentTexture = textures_[3];
    frame.linearSrgb = validationPixels[0];
    frame.energyLedger = validationPixels[1];
    frame.orderLedger = validationPixels[2];
    frame.directionMoment = validationPixels[3];
    output = frame;
    return true;
}

bool OpenGlPhysicalDiffractionExecutor::executeTransport(
    const AdmittedDiffractionMaterialIR& material,
    const AdmittedLightingPlan& lighting,
    OpenGlPhysicalDiffractionFrame& output,
    TransportEnergyReceipt& receipt,
    std::string& error)
{
    output = {};
    receipt = {};
    const auto& description = lighting.description();
    for (std::size_t pathIndex = 0; pathIndex < description.pathCount; ++pathIndex)
    {
        OpenGlPhysicalDiffractionFrame pathFrame;
        if (!execute(material, description.paths[pathIndex].incident, pathFrame, error))
            return false;
        output = pathFrame;
        for (std::size_t component = 0; component < 4; ++component)
        {
            receipt.energyLedger[component] += pathFrame.energyLedger[component];
            receipt.orderLedger[component] += pathFrame.orderLedger[component];
            receipt.directionMoment[component] += pathFrame.directionMoment[component];
        }
        receipt.maximumBounceDepth = std::max(
            receipt.maximumBounceDepth, description.paths[pathIndex].bounceDepth);

        ++receipt.pathCount;
    }
    return true;
}

void OpenGlPhysicalDiffractionExecutor::shutdown() noexcept
{
    if (gl_ != nullptr)
    {
        if (textures_[0] != 0)
            glDeleteTextures(static_cast<int>(textures_.size()), textures_.data());
        if (framebuffer_ != 0)
            gl_->DeleteFramebuffers(1, &framebuffer_);
        if (vao_ != 0)
            gl_->DeleteVertexArrays(1, &vao_);
        if (program_ != 0)
            gl_->DeleteProgram(program_);
    }
    gl_ = nullptr;
    program_ = 0;
    vao_ = 0;
    framebuffer_ = 0;
    textures_.fill(0);
    generation_ = 0;
}
} // namespace diffractionmaterial
