#include "color_transform_gl.h"
#include "../../shared/ColorTransformGpuMath.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace videorender
{
namespace
{
constexpr const char* kVertexShader = R"GLSL(#version 330 core
out vec2 uv;
void main()
{
    vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

constexpr const char* kFragmentShader = R"GLSL(#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D sourceTexture;
uniform int inputTransfer;
uniform int outputTransfer;
uniform int inputAlpha;
uniform int outputAlpha;
uniform int toneMap;
uniform int clampOutput;
uniform mat4 inputToWorking;
uniform mat4 workingToOutput;
uniform vec3 workingLuma;
uniform vec4 luminance; // source reference, source peak, output reference, output peak

vec3 decodeTransfer(vec3 value)
{
    value = max(value, vec3(0.0));
    if (inputTransfer == 1) return value;
    if (inputTransfer == 2)
    {
        bvec3 low = lessThanEqual(value, vec3(0.04045));
        vec3 linearLow = value / 12.92;
        vec3 linearHigh = pow((value + 0.055) / 1.055, vec3(2.4));
        return mix(linearHigh, linearLow, low);
    }
    if (inputTransfer == 3) return pow(value, vec3(2.2));
    if (inputTransfer == 4)
    {
        const float m1 = 2610.0 / 16384.0;
        const float m2 = 2523.0 / 32.0;
        const float c1 = 3424.0 / 4096.0;
        const float c2 = 2413.0 / 128.0;
        const float c3 = 2392.0 / 128.0;
        vec3 p = pow(value, vec3(1.0 / m2));
        vec3 absoluteNits = 10000.0 * pow(max(p - c1, vec3(0.0))
            / max(c2 - c3 * p, vec3(1.0e-6)), vec3(1.0 / m1));
        return absoluteNits / luminance.x;
    }
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    bvec3 low = lessThanEqual(value, vec3(0.5));
    vec3 linearLow = value * value / 3.0;
    vec3 linearHigh = (exp((value - c) / a) + b) / 12.0;
    return mix(linearHigh, linearLow, low);
}

vec3 encodeTransfer(vec3 value)
{
    value = max(value, vec3(0.0));
    if (outputTransfer == 1) return value;
    if (outputTransfer == 2)
    {
        bvec3 low = lessThanEqual(value, vec3(0.0031308));
        vec3 encodedLow = 12.92 * value;
        vec3 encodedHigh = 1.055 * pow(value, vec3(1.0 / 2.4)) - 0.055;
        return mix(encodedHigh, encodedLow, low);
    }
    if (outputTransfer == 3) return pow(value, vec3(1.0 / 2.2));
    if (outputTransfer == 4)
    {
        const float m1 = 2610.0 / 16384.0;
        const float m2 = 2523.0 / 32.0;
        const float c1 = 3424.0 / 4096.0;
        const float c2 = 2413.0 / 128.0;
        const float c3 = 2392.0 / 128.0;
        vec3 normalizedNits = value * luminance.z / 10000.0;
        vec3 p = pow(normalizedNits, vec3(m1));
        return pow((c1 + c2 * p) / (1.0 + c3 * p), vec3(m2));
    }
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    bvec3 low = lessThanEqual(value, vec3(1.0 / 12.0));
    vec3 encodedLow = sqrt(3.0 * value);
    vec3 encodedHigh = a * log(12.0 * value - b) + c;
    return mix(encodedHigh, encodedLow, low);
}

void main()
{
    vec4 sampled = texture(sourceTexture, uv);
    float alpha = inputAlpha == 1 ? 1.0 : sampled.a;
    vec3 encoded = sampled.rgb;
    if (inputAlpha == 3)
        encoded = alpha > 0.0 ? encoded / alpha : vec3(0.0);

    vec3 working = (inputToWorking * vec4(decodeTransfer(encoded), 1.0)).rgb;
    vec3 absoluteNits = working * luminance.x;
    if (toneMap == 1)
    {
        float y = max(dot(max(absoluteNits, vec3(0.0)), workingLuma), 0.0);
        float shoulder = max(0.0, 1.0 / luminance.w - 1.0 / luminance.y);
        float mappedY = y / (1.0 + shoulder * y);
        absoluteNits *= y > 0.0 ? mappedY / y : 0.0;
    }
    working = absoluteNits / luminance.z;
    vec3 outputLinear = (workingToOutput * vec4(working, 1.0)).rgb;
    vec3 outputEncoded = encodeTransfer(outputLinear);

    if (outputAlpha == 1) alpha = 1.0;
    if (outputAlpha == 3) outputEncoded *= alpha;
    if (alpha == 0.0 && outputAlpha == 3) outputEncoded = vec3(0.0);
    if (clampOutput != 0) outputEncoded = clamp(outputEncoded, vec3(0.0), vec3(1.0));
    fragColor = vec4(outputEncoded, alpha);
}
)GLSL";

unsigned compileShader(arbitgl::GlFuncs& gl, unsigned type,
                       const char* source, std::string& error)
{
    const auto shader = gl.CreateShader(type);
    if (shader == 0)
    {
        error = "color transform OpenGL shader allocation failed";
        return 0;
    }
    gl.ShaderSource(shader, 1, &source, nullptr);
    gl.CompileShader(shader);
    int compiled = 0;
    gl.GetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) return shader;
    std::array<char, 2048> log {};
    int length = 0;
    gl.GetShaderInfoLog(shader, static_cast<int>(log.size()), &length, log.data());
    error = "color transform OpenGL shader compile failed: ";
    error.append(log.data(), static_cast<std::size_t>(std::max(length, 0)));
    gl.DeleteShader(shader);
    return 0;
}
} // namespace

bool ColorTransformGl::initialize(arbitgl::GlFuncs* gl, std::string& error)
{
    shutdown();
    error.clear();
    if (gl == nullptr || gl->UniformMatrix4fv == nullptr)
    {
        error = "color transform OpenGL requires a complete GL 3.3 function table";
        return false;
    }
    const auto vertex = compileShader(*gl, GL_VERTEX_SHADER, kVertexShader, error);
    if (vertex == 0) return false;
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
        error = "color transform OpenGL program link failed: ";
        error.append(log.data(), static_cast<std::size_t>(std::max(length, 0)));
        gl->DeleteProgram(program);
        return false;
    }
    unsigned vao = 0;
    unsigned framebuffer = 0;
    gl->GenVertexArrays(1, &vao);
    gl->GenFramebuffers(1, &framebuffer);
    if (vao == 0 || framebuffer == 0)
    {
        if (framebuffer != 0) gl->DeleteFramebuffers(1, &framebuffer);
        if (vao != 0) gl->DeleteVertexArrays(1, &vao);
        gl->DeleteProgram(program);
        error = "color transform OpenGL resource allocation failed";
        return false;
    }
    gl_ = gl;
    program_ = program;
    vao_ = vao;
    framebuffer_ = framebuffer;
    return true;
}

bool ColorTransformGl::render(unsigned sourceTexture, unsigned targetTexture,
                              int width, int height,
                              const colortransform::AdmittedTransform& transform,
                              std::string& error)
{
    error.clear();
    if (! ready() || sourceTexture == 0 || targetTexture == 0 || width <= 0 || height <= 0)
    {
        error = "color transform OpenGL dispatch has invalid resources";
        return false;
    }
    const auto& description = transform.description();
    if (description.input.extent.width != static_cast<std::uint32_t>(width)
        || description.input.extent.height != static_cast<std::uint32_t>(height))
    {
        error = "color transform OpenGL extent does not match the admitted frame";
        return false;
    }

    int previousProgram = 0;
    int previousVao = 0;
    int previousDrawFramebuffer = 0;
    int previousActiveTexture = 0;
    int previousTexture = 0;
    int previousViewport[4] {};
    std::array<unsigned char, 4> previousColorMask {};
    const bool previousBlend = glIsEnabled(GL_BLEND) == GL_TRUE;
    const bool previousCull = glIsEnabled(GL_CULL_FACE) == GL_TRUE;
    const bool previousDepth = glIsEnabled(GL_DEPTH_TEST) == GL_TRUE;
    const bool previousScissor = glIsEnabled(GL_SCISSOR_TEST) == GL_TRUE;
    const bool previousSrgb = glIsEnabled(GL_FRAMEBUFFER_SRGB) == GL_TRUE;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previousVao);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActiveTexture);
    glGetIntegerv(GL_VIEWPORT, previousViewport);
    glGetBooleanv(GL_COLOR_WRITEMASK, previousColorMask.data());
    gl_->ActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture);

    gl_->BindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer_);
    gl_->FramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, targetTexture, 0);
    if (gl_->CheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        gl_->BindFramebuffer(GL_DRAW_FRAMEBUFFER,
                             static_cast<unsigned>(previousDrawFramebuffer));
        gl_->ActiveTexture(static_cast<unsigned>(previousActiveTexture));
        error = "color transform OpenGL framebuffer is incomplete";
        return false;
    }
    glViewport(0, 0, width, height);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    gl_->UseProgram(program_);
    gl_->BindVertexArray(vao_);
    glBindTexture(GL_TEXTURE_2D, sourceTexture);
    gl_->Uniform1i(gl_->GetUniformLocation(program_, "sourceTexture"), 0);
    gl_->Uniform1i(gl_->GetUniformLocation(program_, "inputTransfer"),
                   static_cast<int>(description.input.transfer));
    gl_->Uniform1i(gl_->GetUniformLocation(program_, "outputTransfer"),
                   static_cast<int>(description.output.transfer));
    gl_->Uniform1i(gl_->GetUniformLocation(program_, "inputAlpha"),
                   static_cast<int>(description.input.alpha));
    gl_->Uniform1i(gl_->GetUniformLocation(program_, "outputAlpha"),
                   static_cast<int>(description.output.alpha));
    gl_->Uniform1i(gl_->GetUniformLocation(program_, "toneMap"),
                   static_cast<int>(description.toneMap));
    const bool clamp = description.output.format == colortransform::PixelFormat::R8
        || description.output.format == colortransform::PixelFormat::R16
        || description.output.format == colortransform::PixelFormat::RGBA8;
    gl_->Uniform1i(gl_->GetUniformLocation(program_, "clampOutput"), clamp ? 1 : 0);
    const auto toWorking = colortransform::gpumath::inputToWorking(description);
    const auto fromWorking = colortransform::gpumath::workingToOutput(description);
    gl_->UniformMatrix4fv(gl_->GetUniformLocation(program_, "inputToWorking"),
                          1, GL_FALSE, toWorking.data());
    gl_->UniformMatrix4fv(gl_->GetUniformLocation(program_, "workingToOutput"),
                          1, GL_FALSE, fromWorking.data());
    const auto workingLuma = colortransform::gpumath::workingLuma(description);
    gl_->Uniform3f(gl_->GetUniformLocation(program_, "workingLuma"),
                   workingLuma[0], workingLuma[1], workingLuma[2]);
    gl_->Uniform4f(gl_->GetUniformLocation(program_, "luminance"),
                   static_cast<float>(description.luminance.sourceReferenceWhiteNits),
                   static_cast<float>(description.luminance.sourcePeakNits),
                   static_cast<float>(description.luminance.outputReferenceWhiteNits),
                   static_cast<float>(description.luminance.outputPeakNits));
    glDrawArrays(GL_TRIANGLES, 0, 3);
    const auto drawError = glGetError();

    glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(previousTexture));
    gl_->ActiveTexture(static_cast<unsigned>(previousActiveTexture));
    gl_->BindVertexArray(static_cast<unsigned>(previousVao));
    gl_->UseProgram(static_cast<unsigned>(previousProgram));
    gl_->BindFramebuffer(GL_DRAW_FRAMEBUFFER,
                         static_cast<unsigned>(previousDrawFramebuffer));
    glViewport(previousViewport[0], previousViewport[1],
               previousViewport[2], previousViewport[3]);
    const auto restore = [](unsigned capability, bool enabled)
    { if (enabled) glEnable(capability); else glDisable(capability); };
    restore(GL_BLEND, previousBlend);
    restore(GL_CULL_FACE, previousCull);
    restore(GL_DEPTH_TEST, previousDepth);
    restore(GL_SCISSOR_TEST, previousScissor);
    restore(GL_FRAMEBUFFER_SRGB, previousSrgb);
    glColorMask(previousColorMask[0], previousColorMask[1],
                previousColorMask[2], previousColorMask[3]);
    if (drawError != GL_NO_ERROR)
    {
        error = "color transform OpenGL draw failed";
        return false;
    }
    return true;
}

void ColorTransformGl::shutdown() noexcept
{
    if (gl_ != nullptr)
    {
        if (framebuffer_ != 0) gl_->DeleteFramebuffers(1, &framebuffer_);
        if (vao_ != 0) gl_->DeleteVertexArrays(1, &vao_);
        if (program_ != 0) gl_->DeleteProgram(program_);
    }
    gl_ = nullptr;
    program_ = 0;
    vao_ = 0;
    framebuffer_ = 0;
}
} // namespace videorender
