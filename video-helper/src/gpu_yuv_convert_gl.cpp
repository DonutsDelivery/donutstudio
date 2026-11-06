#include "gpu_yuv_convert_gl.h"

#include <algorithm>
#include <array>

namespace videorender
{
namespace
{
constexpr const char* kVertexShader = R"GLSL(#version 330 core
void main()
{
    vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

// yuv_encode_math.h in GLSL: encodedRgb, toYCbCr and quantize. Plane 0 writes a
// luma code per pixel; plane 1 writes the chroma pair for the even column on
// two rows (AVCHROMA_LOC_LEFT).
constexpr const char* kFragmentShader = R"GLSL(#version 330 core
out vec4 fragColor;
uniform sampler2D sourceTexture;
uniform int plane;
uniform int transfer;   // 0 SDR, 1 PQ, 2 HLG
uniform vec3 toBt2020Row0;
uniform vec3 toBt2020Row1;
uniform vec3 toBt2020Row2;
uniform vec2 weights;   // kr, kb
uniform vec4 codes;     // luma offset, luma scale, chroma mid, chroma scale
uniform vec2 storage;   // maximum code, a code's value in the plane texture
uniform vec3 hdr;       // peak nits, reference white nits, HLG gamma

float pqEncode(float nits)
{
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    float y = pow(clamp(nits / 10000.0, 0.0, 1.0), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

float hlgOetf(float scene)
{
    const float a = 0.17883277;
    const float b = 1.0 - 4.0 * a;
    float c = 0.5 - a * log(4.0 * a);
    scene = clamp(scene, 0.0, 1.0);
    return scene <= 1.0 / 12.0 ? sqrt(3.0 * scene) : a * log(12.0 * scene - b) + c;
}

vec3 encodedRgb(vec3 rgb)
{
    if (transfer == 0) return clamp(rgb, 0.0, 1.0);
    rgb = max(rgb, vec3(0.0));
    vec3 display = vec3(dot(toBt2020Row0, rgb), dot(toBt2020Row1, rgb), dot(toBt2020Row2, rgb));
    display = min(display * hdr.y, vec3(hdr.x));
    if (transfer == 1)
        return vec3(pqEncode(display.r), pqEncode(display.g), pqEncode(display.b));
    display /= hdr.x;
    float luminance = dot(display, vec3(0.2627, 0.6780, 0.0593));
    float inverseGain = luminance > 0.0 ? pow(luminance, (1.0 - hdr.z) / hdr.z) : 0.0;
    display *= inverseGain;
    return vec3(hlgOetf(display.r), hlgOetf(display.g), hlgOetf(display.b));
}

vec3 toYCbCr(vec3 e)
{
    float y = weights.x * e.r + (1.0 - weights.x - weights.y) * e.g + weights.y * e.b;
    return vec3(y, (e.b - y) / (2.0 * (1.0 - weights.y)), (e.r - y) / (2.0 * (1.0 - weights.x)));
}

float quantize(float value, float offset, float scale)
{
    return clamp(floor(offset + scale * value + 0.5), 0.0, storage.x) * storage.y;
}

vec3 sampleAt(ivec2 p)
{
    ivec2 size = textureSize(sourceTexture, 0);
    return texelFetch(sourceTexture, clamp(p, ivec2(0), size - 1), 0).rgb;
}

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (plane == 0)
    {
        float y = toYCbCr(encodedRgb(sampleAt(p))).x;
        fragColor = vec4(quantize(y, codes.x, codes.y), 0.0, 0.0, 1.0);
        return;
    }
    vec3 top = toYCbCr(encodedRgb(sampleAt(p * 2)));
    vec3 bottom = toYCbCr(encodedRgb(sampleAt(p * 2 + ivec2(0, 1))));
    fragColor = vec4(quantize((top.y + bottom.y) * 0.5, codes.z, codes.w),
                     quantize((top.z + bottom.z) * 0.5, codes.z, codes.w), 0.0, 1.0);
}
)GLSL";

unsigned compileShader (arbitgl::GlFuncs& gl, unsigned type, const char* source, std::string& error)
{
    const auto shader = gl.CreateShader (type);
    if (shader == 0)
    {
        error = "GPU YUV conversion shader allocation failed";
        return 0;
    }
    gl.ShaderSource (shader, 1, &source, nullptr);
    gl.CompileShader (shader);
    int compiled = 0;
    gl.GetShaderiv (shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE)
        return shader;
    std::array<char, 2048> log {};
    int length = 0;
    gl.GetShaderInfoLog (shader, static_cast<int> (log.size()), &length, log.data());
    error = "GPU YUV conversion shader compile failed: ";
    error.append (log.data(), static_cast<std::size_t> (std::max (length, 0)));
    gl.DeleteShader (shader);
    return 0;
}

unsigned makePlane (int width, int height, int bits, bool chroma)
{
    unsigned texture = 0;
    glGenTextures (1, &texture);
    if (texture == 0)
        return 0;
    GLint previous = 0;
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &previous);
    glBindTexture (GL_TEXTURE_2D, texture);
    const GLint internal = chroma ? (bits == 10 ? GL_RG16 : GL_RG8) : (bits == 10 ? GL_R16 : GL_R8);
    glTexImage2D (GL_TEXTURE_2D, 0, internal, width, height, 0, chroma ? GL_RG : GL_RED,
                  bits == 10 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previous));
    return texture;
}
} // namespace

bool YuvPlanesGl::initialize (arbitgl::GlFuncs* gl, std::string& error)
{
    shutdown();
    error.clear();
    if (gl == nullptr || gl->Uniform3f == nullptr)
    {
        error = "GPU YUV conversion requires a complete GL 3.3 function table";
        return false;
    }
    const auto vertex = compileShader (*gl, GL_VERTEX_SHADER, kVertexShader, error);
    if (vertex == 0)
        return false;
    const auto fragment = compileShader (*gl, GL_FRAGMENT_SHADER, kFragmentShader, error);
    if (fragment == 0)
    {
        gl->DeleteShader (vertex);
        return false;
    }
    const auto program = gl->CreateProgram();
    gl->AttachShader (program, vertex);
    gl->AttachShader (program, fragment);
    gl->LinkProgram (program);
    gl->DeleteShader (vertex);
    gl->DeleteShader (fragment);
    int linked = 0;
    gl->GetProgramiv (program, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE)
    {
        std::array<char, 2048> log {};
        int length = 0;
        gl->GetProgramInfoLog (program, static_cast<int> (log.size()), &length, log.data());
        error = "GPU YUV conversion program link failed: ";
        error.append (log.data(), static_cast<std::size_t> (std::max (length, 0)));
        gl->DeleteProgram (program);
        return false;
    }
    unsigned vao = 0, framebuffer = 0;
    gl->GenVertexArrays (1, &vao);
    gl->GenFramebuffers (1, &framebuffer);
    if (vao == 0 || framebuffer == 0)
    {
        if (framebuffer != 0) gl->DeleteFramebuffers (1, &framebuffer);
        if (vao != 0) gl->DeleteVertexArrays (1, &vao);
        gl->DeleteProgram (program);
        error = "GPU YUV conversion resource allocation failed";
        return false;
    }
    gl_ = gl;
    program_ = program;
    vao_ = vao;
    framebuffer_ = framebuffer;
    return true;
}

bool YuvPlanesGl::render (unsigned sourceTexture, int width, int height,
                          const videohelper::yuvencode::Conversion& conversion, std::string& error)
{
    if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0)
    {
        error = "GPU YUV conversion needs an even frame size";
        return false;
    }
    if (luma_ == 0 || planesWidth_ != width || planesHeight_ != height || planesBits_ != conversion.bits)
    {
        if (luma_ != 0) glDeleteTextures (1, &luma_);
        if (chroma_ != 0) glDeleteTextures (1, &chroma_);
        luma_ = makePlane (width, height, conversion.bits, false);
        chroma_ = makePlane (width / 2, height / 2, conversion.bits, true);
        planesWidth_ = width;
        planesHeight_ = height;
        planesBits_ = conversion.bits;
        if (luma_ == 0 || chroma_ == 0)
        {
            error = "GPU YUV conversion could not allocate its planes";
            return false;
        }
    }
    return renderInto (sourceTexture, luma_, chroma_, width, height, conversion, error);
}

bool YuvPlanesGl::renderInto (unsigned sourceTexture, unsigned lumaTexture, unsigned chromaTexture, int width,
                              int height, const videohelper::yuvencode::Conversion& conversion, std::string& error)
{
    error.clear();
    if (! ready() || sourceTexture == 0 || lumaTexture == 0 || chromaTexture == 0 || width <= 0 || height <= 0
        || (width & 1) != 0 || (height & 1) != 0)
    {
        error = "GPU YUV conversion has invalid resources";
        return false;
    }
    const auto constants = videohelper::yuvencode::shaderConstants (conversion);
    return draw (sourceTexture, lumaTexture, width, height, 0, constants, error)
        && draw (sourceTexture, chromaTexture, width / 2, height / 2, 1, constants, error);
}

bool YuvPlanesGl::draw (unsigned sourceTexture, unsigned target, int width, int height, int plane,
                        const videohelper::yuvencode::ShaderConstants& constants, std::string& error)
{
    int previousProgram = 0, previousVao = 0, previousDrawFramebuffer = 0, previousActiveTexture = 0,
        previousTexture = 0;
    int previousViewport[4] {};
    const bool previousBlend = glIsEnabled (GL_BLEND) == GL_TRUE;
    const bool previousScissor = glIsEnabled (GL_SCISSOR_TEST) == GL_TRUE;
    const bool previousSrgb = glIsEnabled (GL_FRAMEBUFFER_SRGB) == GL_TRUE;
    glGetIntegerv (GL_CURRENT_PROGRAM, &previousProgram);
    glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &previousVao);
    glGetIntegerv (GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
    glGetIntegerv (GL_ACTIVE_TEXTURE, &previousActiveTexture);
    glGetIntegerv (GL_VIEWPORT, previousViewport);
    gl_->ActiveTexture (GL_TEXTURE0);
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &previousTexture);

    const auto restore = [&] {
        glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTexture));
        gl_->ActiveTexture (static_cast<unsigned> (previousActiveTexture));
        gl_->BindVertexArray (static_cast<unsigned> (previousVao));
        gl_->UseProgram (static_cast<unsigned> (previousProgram));
        gl_->BindFramebuffer (GL_DRAW_FRAMEBUFFER, static_cast<unsigned> (previousDrawFramebuffer));
        glViewport (previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
        if (previousBlend) glEnable (GL_BLEND);
        if (previousScissor) glEnable (GL_SCISSOR_TEST);
        if (previousSrgb) glEnable (GL_FRAMEBUFFER_SRGB);
    };

    gl_->BindFramebuffer (GL_DRAW_FRAMEBUFFER, framebuffer_);
    gl_->FramebufferTexture2D (GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target, 0);
    if (gl_->CheckFramebufferStatus (GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        restore();
        error = "GPU YUV conversion framebuffer is incomplete";
        return false;
    }
    glViewport (0, 0, width, height);
    glDisable (GL_BLEND);
    glDisable (GL_SCISSOR_TEST);
    glDisable (GL_FRAMEBUFFER_SRGB);
    gl_->UseProgram (program_);
    gl_->BindVertexArray (vao_);
    glBindTexture (GL_TEXTURE_2D, sourceTexture);
    const auto at = [&] (const char* name) { return gl_->GetUniformLocation (program_, name); };
    const auto& m = constants.toBt2020;
    gl_->Uniform1i (at ("sourceTexture"), 0);
    gl_->Uniform1i (at ("plane"), plane);
    gl_->Uniform1i (at ("transfer"), constants.transfer);
    gl_->Uniform3f (at ("toBt2020Row0"), m[0], m[1], m[2]);
    gl_->Uniform3f (at ("toBt2020Row1"), m[3], m[4], m[5]);
    gl_->Uniform3f (at ("toBt2020Row2"), m[6], m[7], m[8]);
    gl_->Uniform2f (at ("weights"), constants.kr, constants.kb);
    gl_->Uniform4f (at ("codes"), constants.lumaOffset, constants.lumaScale, constants.chromaMid, constants.chromaScale);
    gl_->Uniform2f (at ("storage"), constants.maximum, constants.storageUnit);
    gl_->Uniform3f (at ("hdr"), constants.peakNits, constants.referenceWhite, constants.hlgGamma);
    glDrawArrays (GL_TRIANGLES, 0, 3);
    const auto drawError = glGetError();
    restore();
    if (drawError != GL_NO_ERROR)
    {
        error = "GPU YUV conversion draw failed";
        return false;
    }
    return true;
}

void YuvPlanesGl::shutdown() noexcept
{
    if (gl_ != nullptr)
    {
        if (luma_ != 0) glDeleteTextures (1, &luma_);
        if (chroma_ != 0) glDeleteTextures (1, &chroma_);
        if (framebuffer_ != 0) gl_->DeleteFramebuffers (1, &framebuffer_);
        if (vao_ != 0) gl_->DeleteVertexArrays (1, &vao_);
        if (program_ != 0) gl_->DeleteProgram (program_);
    }
    gl_ = nullptr;
    program_ = vao_ = framebuffer_ = luma_ = chroma_ = 0;
    planesWidth_ = planesHeight_ = planesBits_ = 0;
}
} // namespace videorender
