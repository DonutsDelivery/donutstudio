#include "backend.h"
#include "../geometry_core_diagnostic_colors.h"
#include "../geometry_core_admission.h"
#include "../gl_loader.h"
#include "../sdf_native_program.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace arbitgpu
{
namespace
{
constexpr const char* kUnavailable = "native OpenGL SDF execution requires a current OpenGL 3.3 context";

const char* kVertexShader = R"glsl(#version 330 core
const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0); }
)glsl";

const char* kFragmentShader = R"glsl(#version 330 core
layout(location = 0) out vec4 outColor;
uniform vec4 uRecords[64];
uniform vec4 uParameters0[64];
uniform vec4 uParameters1[64];
uniform int uRootIndex;
uniform float uEpsilon;
uniform float uMaximumDistance;
uniform vec2 uExtent;
uniform int uMaximumSteps;
uniform int uAdaptiveQuality;
uniform int uNormalQuality;
uniform int uShadowQuality;
uniform int uOutputPass;

float componentAt(vec3 value, int axis) { return axis == 0 ? value.x : (axis == 1 ? value.y : value.z); }
vec3 withComponent(vec3 value, int axis, float replacement) {
    if (axis == 0) value.x = replacement;
    else if (axis == 1) value.y = replacement;
    else value.z = replacement;
    return value;
}
vec3 rotateAxis(vec3 value, vec3 axis, float radians) {
    axis = normalize(axis);
    float c = cos(radians), s = sin(radians);
    return value * c + cross(axis, value) * s + axis * dot(axis, value) * (1.0 - c);
}
vec3 rotatePair(vec3 value, int first, int second, float radians) {
    float a = componentAt(value, first), b = componentAt(value, second);
    float c = cos(radians), s = sin(radians);
    value = withComponent(value, first, c * a - s * b);
    return withComponent(value, second, s * a + c * b);
}
float boxDistance(vec3 point, vec3 halfExtent) {
    vec3 outside = abs(point) - halfExtent;
    return length(max(outside, vec3(0.0)))
         + min(max(outside.x, max(outside.y, outside.z)), 0.0);
}
float primitiveDistance(int operation, vec3 point, vec4 p0, vec4 p1) {
    if (operation == 2) return boxDistance(point, p0.xyz);
    if (operation == 3) return boxDistance(point, p0.xyz - vec3(p0.w)) - p0.w;
    if (operation == 4) return dot(point, normalize(p0.xyz)) + p0.w;
    if (operation == 5) return length(vec2(length(point.xz) - p0.x, point.y)) - p0.y;
    if (operation == 6) {
        vec3 relative = point - p0.xyz, segment = vec3(p0.w, p1.x, p1.y) - p0.xyz;
        float along = clamp(dot(relative, segment) / dot(segment, segment), 0.0, 1.0);
        return length(relative - segment * along) - p1.z;
    }
    if (operation == 7) {
        vec2 cylinder = abs(vec2(length(point.xz), point.y)) - p0.xy;
        return min(max(cylinder.x, cylinder.y), 0.0) + length(max(cylinder, vec2(0.0)));
    }
    if (operation == 8) {
        vec2 q = vec2(length(point.xz), point.y), base = vec2(p0.x, -p0.y);
        vec2 side = vec2(p0.x, -2.0 * p0.y);
        vec2 cap = vec2(q.x - min(q.x, q.y < 0.0 ? p0.x : 0.0), abs(q.y) - p0.y);
        vec2 slope = q - base + side * clamp(dot(base - q, side) / dot(side, side), 0.0, 1.0);
        return (slope.x < 0.0 && cap.y < 0.0 ? -1.0 : 1.0)
             * sqrt(min(dot(cap, cap), dot(slope, slope)));
    }
    if (operation == 9) {
        vec3 cell = point * p0.x;
        return abs(dot(sin(cell), cos(cell.zxy))) / p0.x - p0.y;
    }
    return length(point) - p0.x;
}
float smoothUnionDistance(float a, float b, float radius) {
    float h = clamp(0.5 + 0.5 * (b - a) / radius, 0.0, 1.0);
    return mix(b, a, h) - radius * h * (1.0 - h);
}
vec3 childPoint(int operation, vec3 point, vec4 p0, vec4 p1) {
    if (operation == 16) return point - p0.xyz;
    if (operation == 17) return rotateAxis(point, p0.xyz, -p0.w);
    if (operation == 18) return point / p0.xyz;
    if (operation == 19) return mod(point + p0.xyz * 0.5, p0.xyz) - p0.xyz * 0.5;
    if (operation == 20) {
        int axis = int(p0.x), first = (axis + 1) % 3, second = (axis + 2) % 3;
        float a = componentAt(point, first), b = componentAt(point, second);
        float radius = length(vec2(a, b)), sector = 6.283185307179586 / p0.y;
        float angle = mod(atan(b, a) - p0.z + sector * 0.5, sector) - sector * 0.5;
        point = withComponent(point, first, radius * cos(angle));
        return withComponent(point, second, radius * sin(angle));
    }
    if (operation == 21) return mix(point, abs(point), p0.xyz);
    if (operation == 22) {
        int axis = int(p0.x);
        return rotatePair(point, (axis + 1) % 3, (axis + 2) % 3,
                          -p0.y * componentAt(point, axis));
    }
    if (operation == 23) {
        int axis = int(p0.x);
        return rotatePair(point, (axis + 1) % 3, axis,
                          -p0.y * componentAt(point, axis));
    }
    if (operation == 24) {
        int axis = int(p0.x); float factor = 1.0 - p0.y * componentAt(point, axis);
        point = withComponent(point, (axis + 1) % 3,
                              componentAt(point, (axis + 1) % 3) * factor);
        return withComponent(point, (axis + 2) % 3,
                             componentAt(point, (axis + 2) % 3) * factor);
    }
    if (operation == 26)
        return point + vec3(p0.x * sin(p0.w * point.y),
                            p0.y * sin(p1.x * point.z),
                            p0.z * sin(p1.y * point.x));
    return point;
}
float sceneDistance(vec3 point) {
    int indices[32], stages[32]; vec3 points[32]; float firstValues[32];
    int top = 0; indices[0] = uRootIndex; stages[0] = 0; points[0] = point;
    float value = 0.0;
    for (int iteration = 0; iteration < 768; ++iteration) {
        int index = indices[top]; ivec4 record = ivec4(uRecords[index]);
        int operation = record.x; vec4 p0 = uParameters0[index], p1 = uParameters1[index];
        if (operation <= 9) {
            value = primitiveDistance(operation, points[top], p0, p1);
            if (top == 0) return value;
            --top; continue;
        }
        bool binary = operation >= 10 && operation <= 15;
        if (binary) {
            if (stages[top] == 0) {
                stages[top] = 1; ++top; indices[top] = record.y; stages[top] = 0;
                points[top] = points[top - 1]; continue;
            }
            if (stages[top] == 1) {
                firstValues[top] = value; stages[top] = 2; ++top;
                indices[top] = record.z; stages[top] = 0; points[top] = points[top - 1]; continue;
            }
            float a = firstValues[top], b = value;
            if (operation == 10) value = min(a, b);
            else if (operation == 11) value = max(a, b);
            else if (operation == 12) value = max(a, -b);
            else if (operation == 13) value = smoothUnionDistance(a, b, p0.x);
            else if (operation == 14) value = -smoothUnionDistance(-a, -b, p0.x);
            else value = -smoothUnionDistance(-a, b, p0.x);
        } else if (stages[top] == 0) {
            stages[top] = 1; ++top; indices[top] = record.y; stages[top] = 0;
            points[top] = childPoint(operation, points[top - 1], p0, p1); continue;
        } else {
            if (operation == 18) value *= min(abs(p0.x), min(abs(p0.y), abs(p0.z)));
            else if (operation == 25) {
                vec3 q = points[top];
                value += p0.x * sin(p0.y * q.x) * sin(p0.y * q.y) * sin(p0.y * q.z);
            }
        }
        if (top == 0) return value;
        --top;
    }
    return uMaximumDistance;
}
float qualityScale(int quality) {
    if (quality <= 0) return 4.0; if (quality == 1) return 2.0;
    if (quality == 2) return 1.0; return 0.5;
}
vec3 sceneNormal(vec3 point) {
    float e = max(uEpsilon * qualityScale(uNormalQuality), 0.000001);
    return normalize(vec3(
        sceneDistance(point + vec3(e,0,0)) - sceneDistance(point - vec3(e,0,0)),
        sceneDistance(point + vec3(0,e,0)) - sceneDistance(point - vec3(0,e,0)),
        sceneDistance(point + vec3(0,0,e)) - sceneDistance(point - vec3(0,0,e))));
}
float sceneShadow(vec3 origin, vec3 direction) {
    int limit = uShadowQuality <= 0 ? 8 : (uShadowQuality == 1 ? 16 : (uShadowQuality == 2 ? 32 : 64));
    float travel = uEpsilon * 4.0, visibility = 1.0;
    for (int step = 0; step < 64; ++step) {
        if (step >= limit) break; float field = sceneDistance(origin + direction * travel);
        if (field < uEpsilon) return 0.0;
        visibility = min(visibility, 12.0 * field / max(travel, uEpsilon));
        travel += clamp(field, uEpsilon * 2.0, 0.25);
        if (travel > min(uMaximumDistance, 8.0)) break;
    }
    return clamp(visibility, 0.0, 1.0);
}
void main() {
    vec2 uv = (2.0 * gl_FragCoord.xy - uExtent) / uExtent.y;
    vec3 origin = vec3(0.0,0.0,3.0), direction = normalize(vec3(uv,-1.8));
    float travel = 0.0; bool hit = false;
    for (int step = 0; step < 512; ++step) {
        if (step >= uMaximumSteps || travel > uMaximumDistance) break;
        float field = sceneDistance(origin + direction * travel);
        float threshold = max(uEpsilon * qualityScale(uAdaptiveQuality) * max(1.0, travel * 0.05), 0.000001);
        if (field <= threshold) { hit = true; break; } travel += field;
    }
    if (!hit) { outColor = uOutputPass == 1 ? vec4(1.0) : vec4(0.02745,0.03922,0.07059,1.0); return; }
    vec3 point = origin + direction * travel, normal = sceneNormal(point);
    if (uOutputPass == 1) { outColor = vec4(vec3(clamp(travel / uMaximumDistance,0.0,1.0)),1.0); return; }
    if (uOutputPass == 2) { outColor = vec4(normal * 0.5 + 0.5,1.0); return; }
    vec3 light = normalize(vec3(-0.45,0.75,0.6));
    float diffuse = max(dot(normal,light),0.0), shadow = sceneShadow(point + normal*uEpsilon*4.0,light);
    float rim = pow(1.0 - max(dot(normal,-direction),0.0),3.0);
    outColor = vec4(vec3(0.12,0.42,0.88)*(0.12+0.88*diffuse*shadow)+vec3(0.18,0.35,0.65)*rim,1.0);
}
)glsl";

bool validQuality (NativeSdfQuality quality) noexcept
{
    return static_cast<std::uint8_t> (quality)
        < static_cast<std::uint8_t> (NativeSdfQuality::count);
}

std::shared_ptr<const NativeSdfCompiledProgram> resolveProgram (
    const NativeSdfDrawRequest& request, std::string& error)
{
    if (request.compiledProgram != nullptr)
    {
        if (! videohelper::sdf::validateNativeSdfProgram (
                *request.compiledProgram, request.geometry, error))
            return {};
        return request.compiledProgram;
    }
    return videohelper::sdf::compileNativeSdfProgram (request.geometry, error);
}

unsigned compileShader (const arbitgl::GlFuncs& gl, unsigned type,
                        const char* source, std::string& error)
{
    const auto shader = gl.CreateShader (type);
    gl.ShaderSource (shader, 1, &source, nullptr);
    gl.CompileShader (shader);
    int compiled = 0;
    gl.GetShaderiv (shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE)
    {
        char log[1024] = {};
        int length = 0;
        gl.GetShaderInfoLog (shader, static_cast<int> (sizeof (log)), &length, log);
        error = std::string ("OpenGL native SDF shader compilation failed: ")
              + std::string (log, static_cast<std::size_t> (std::max (length, 0)));
        gl.DeleteShader (shader);
        return 0;
    }
    return shader;
}

class OpenGlSdfSceneFrame final : public NativeSdfSceneFrame
{
public:
    ~OpenGlSdfSceneFrame() override
    {
        if (outputBackend_ != nullptr && lifecycle_.value != 0)
            outputBackend_->releaseRenderPassOutputs (lifecycle_);
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return texture_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return texture_; }
    const FrameMemoryAdmission& frameMemoryAdmission() const noexcept override
    {
        return frameMemory_;
    }
    const NativeSdfResourceReceipt& sdfResourceReceipt() const noexcept override
    {
        return receipt_;
    }

    std::string backend_ = "opengl";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    unsigned texture_ = 0;
    RenderPassOutputBackend* outputBackend_ = nullptr;
    RenderPassOutputLifecycleHandle lifecycle_ {};
    FrameMemoryAdmission frameMemory_ {};
    NativeSdfResourceReceipt receipt_ {};
};

class OpenGlSdfExecutionBackend final : public NativeSdfExecutionBackend
{
public:
    void invalidateContext (std::uintptr_t identity) noexcept
    {
        std::lock_guard<std::mutex> lock (programMutex_);
        const auto context = reinterpret_cast<GLFWwindow*> (identity);
        const auto generation = contextGenerations_[context];
        const ContextKey key { context, generation };
        const auto found = programs_.find (key);
        if (found != programs_.end())
        {
            if (glfwGetCurrentContext() == context)
            {
                arbitgl::GlFuncs gl;
                std::string missing;
                if (arbitgl::loadGlFunctions (gl, missing))
                    gl.DeleteProgram (found->second);
            }
            programs_.erase (found);
        }
        ++contextGenerations_[context];
    }
    NativeSdfExecutionCapabilities capabilities() const override
    {
        NativeSdfExecutionCapabilities result;
        if (glfwGetCurrentContext() == nullptr)
        {
            result.error = kUnavailable;
            return result;
        }

        int major = 0;
        int minor = 0;
        int maximumTextureSize = 0;
        glGetIntegerv (GL_MAJOR_VERSION, &major);
        glGetIntegerv (GL_MINOR_VERSION, &minor);
        glGetIntegerv (GL_MAX_TEXTURE_SIZE, &maximumTextureSize);
        if (major < 3 || (major == 3 && minor < 3) || maximumTextureSize <= 0)
        {
            result.error = "native OpenGL SDF execution requires OpenGL 3.3";
            return result;
        }

        const auto* renderer = reinterpret_cast<const char*> (glGetString (GL_RENDERER));
        result.available = true;
        result.backend = "opengl";
        result.device = renderer != nullptr ? renderer : "OpenGL device";
        for (std::uint32_t operation = static_cast<std::uint32_t> (
                 videowire::SdfOperation::sphere);
             operation <= static_cast<std::uint32_t> (videowire::SdfOperation::domainWarp);
             ++operation)
            result.supportedOperations[operation] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::color)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::depth)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::normal)] = true;
        result.maxOperations = kNativeSdfMaximumRecords;
        result.maxDepth = kNativeSdfMaximumDepth;
        result.maxExtent = static_cast<std::uint32_t> (std::min (maximumTextureSize, 4096));
        result.maxPixels = std::min<std::uint64_t> (
            static_cast<std::uint64_t> (result.maxExtent) * result.maxExtent,
            static_cast<std::uint64_t> (4096) * 4096);
        result.maxSteps = 512;
        result.minEpsilon = 0.000001;
        result.maxEpsilon = 0.1;
        result.maxDistance = 1000.0;
        return result;
    }

    NativeSdfSceneSubmission render (const NativeSdfDrawRequest& request) override
    {
        NativeSdfSceneSubmission result;
        std::string programError;
        const auto compiledProgram = resolveProgram (request, programError);
        if (! compiledProgram)
        {
            result.error = programError.empty()
                ? "OpenGL native SDF program admission failed" : std::move (programError);
            return result;
        }
        const auto available = capabilities();
        if (! available.available)
        {
            result.error = available.error;
            return result;
        }
        if (request.width == 0 || request.height == 0
            || request.width > available.maxExtent || request.height > available.maxExtent
            || static_cast<std::uint64_t> (request.width) * request.height > available.maxPixels)
        {
            result.error = "OpenGL native SDF render dimensions exceed backend limits";
            return result;
        }
        if (request.maximumSteps == 0 || request.maximumSteps > available.maxSteps
            || ! std::isfinite (request.epsilon)
            || request.epsilon < available.minEpsilon || request.epsilon > available.maxEpsilon
            || ! std::isfinite (request.maximumDistance)
            || request.maximumDistance < request.epsilon
            || request.maximumDistance > available.maxDistance
            || ! validQuality (request.adaptiveQuality)
            || ! validQuality (request.normalQuality)
            || ! validQuality (request.shadowQuality)
            || (request.output != NativeSdfOutput::color
                && request.output != NativeSdfOutput::depth
                && request.output != NativeSdfOutput::normal))
        {
            result.error = "OpenGL native SDF raymarch controls exceed backend limits";
            return result;
        }

        arbitgl::GlFuncs gl;
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing))
        {
            result.error = "OpenGL native SDF loader failed: " + missing;
            return result;
        }

        int previousDrawFramebuffer = 0;
        int previousReadFramebuffer = 0;
        int previousProgram = 0;
        int previousVertexArray = 0;
        int previousTexture = 0;
        int previousViewport[4] = {};
        int previousPolygonMode = 0;
        unsigned char previousColorMask[4] = {};
        const bool blendWasEnabled = glIsEnabled (GL_BLEND) == GL_TRUE;
        const bool depthWasEnabled = glIsEnabled (GL_DEPTH_TEST) == GL_TRUE;
        const bool cullWasEnabled = glIsEnabled (GL_CULL_FACE) == GL_TRUE;
        const bool scissorWasEnabled = glIsEnabled (GL_SCISSOR_TEST) == GL_TRUE;
        const bool rasterizerDiscardWasEnabled = glIsEnabled (GL_RASTERIZER_DISCARD) == GL_TRUE;
        const bool colorLogicOpWasEnabled = glIsEnabled (GL_COLOR_LOGIC_OP) == GL_TRUE;
        const bool framebufferSrgbWasEnabled = glIsEnabled (GL_FRAMEBUFFER_SRGB) == GL_TRUE;
        glGetIntegerv (GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
        glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
        glGetIntegerv (GL_CURRENT_PROGRAM, &previousProgram);
        glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &previousVertexArray);
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &previousTexture);
        glGetIntegerv (GL_VIEWPORT, previousViewport);
        glGetIntegerv (GL_POLYGON_MODE, &previousPolygonMode);
        glGetBooleanv (GL_COLOR_WRITEMASK, previousColorMask);

        unsigned vertexArray = 0;
        auto frame = std::make_shared<OpenGlSdfSceneFrame>();
        frame->width_ = request.width;
        frame->height_ = request.height;

        renderpassoutput::Description outputDescription;
        outputDescription.extent = { request.width, request.height };
        outputDescription.attachments.push_back ({
            renderpassoutput::Output::Color,
            renderpassoutput::PixelFormat::RGBA16Float,
            renderpassoutput::ColorSpace::LinearSRGB,
            outputDescription.extent });
        renderpassoutput::AdmissionFailure outputFailure;
        const auto admittedOutputs = renderpassoutput::admit (outputDescription, outputFailure);
        if (! admittedOutputs)
        {
            result.error = "OpenGL native SDF output admission failed: "
                + std::string (renderpassoutput::token (outputFailure));
            return result;
        }
        auto& outputBackend = nativeRenderPassOutputBackend();
        auto outputAdmission = outputBackend.admitRenderPassOutputs (*admittedOutputs);
        if (outputAdmission.lifecycle.value == 0 || outputAdmission.resources.size() != 1)
        {
            if (outputAdmission.lifecycle.value != 0)
                outputBackend.releaseRenderPassOutputs (outputAdmission.lifecycle);
            result.error = outputAdmission.error.empty()
                ? "OpenGL native SDF output allocation failed"
                : std::move (outputAdmission.error);
            return result;
        }
        frame->outputBackend_ = &outputBackend;
        frame->lifecycle_ = outputAdmission.lifecycle;
        frame->frameMemory_ = outputAdmission.frameMemory;
        frame->receipt_.compiledRecordCount = static_cast<std::uint32_t> (
            compiledProgram->records().size());
        frame->receipt_.compiledRecordBytes = frame->receipt_.compiledRecordCount
            * sizeof (NativeSdfCompiledRecord);
        frame->receipt_.geometryCacheBytes = request.geometryCacheBytes;
        frame->receipt_.backendProgramBytes = std::strlen (kVertexShader) + std::strlen (kFragmentShader);
        frame->receipt_.uniformBytes = (kNativeSdfMaximumRecords * 12u * sizeof (float))
            + 8u * sizeof (float);
        frame->receipt_.attachmentBytes = outputAdmission.frameMemory.requestedBytes;
        frame->receipt_.totalBytes = frame->receipt_.compiledRecordBytes
            + frame->receipt_.geometryCacheBytes + frame->receipt_.backendProgramBytes
            + frame->receipt_.uniformBytes + frame->receipt_.attachmentBytes;
        frame->texture_ = static_cast<unsigned> (outputAdmission.resources[0].image);
        const auto framebuffer = static_cast<unsigned> (
            outputAdmission.resources[0].attachmentView);

        auto cleanup = [&]
        {
            gl.BindFramebuffer (GL_DRAW_FRAMEBUFFER,
                                static_cast<unsigned> (previousDrawFramebuffer));
            gl.BindFramebuffer (GL_READ_FRAMEBUFFER,
                                static_cast<unsigned> (previousReadFramebuffer));
            gl.UseProgram (static_cast<unsigned> (previousProgram));
            gl.BindVertexArray (static_cast<unsigned> (previousVertexArray));
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTexture));
            glViewport (previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
            glPolygonMode (GL_FRONT_AND_BACK,
                           static_cast<unsigned> (previousPolygonMode));
            glColorMask (previousColorMask[0], previousColorMask[1],
                         previousColorMask[2], previousColorMask[3]);
            if (blendWasEnabled) glEnable (GL_BLEND); else glDisable (GL_BLEND);
            if (depthWasEnabled) glEnable (GL_DEPTH_TEST); else glDisable (GL_DEPTH_TEST);
            if (cullWasEnabled) glEnable (GL_CULL_FACE); else glDisable (GL_CULL_FACE);
            if (scissorWasEnabled) glEnable (GL_SCISSOR_TEST); else glDisable (GL_SCISSOR_TEST);
            if (rasterizerDiscardWasEnabled) glEnable (GL_RASTERIZER_DISCARD); else glDisable (GL_RASTERIZER_DISCARD);
            if (colorLogicOpWasEnabled) glEnable (GL_COLOR_LOGIC_OP); else glDisable (GL_COLOR_LOGIC_OP);
            if (framebufferSrgbWasEnabled) glEnable (GL_FRAMEBUFFER_SRGB); else glDisable (GL_FRAMEBUFFER_SRGB);
            if (vertexArray != 0) gl.DeleteVertexArrays (1, &vertexArray);
        };
        auto fail = [&] (std::string diagnostic)
        {
            cleanup();
            result.error = std::move (diagnostic);
            return result;
        };

        gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
        if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            return fail ("OpenGL native SDF framebuffer is incomplete");

        const auto program = programForCurrentContext (gl, result.error);
        if (program == 0) return fail (result.error);

        gl.GenVertexArrays (1, &vertexArray);
        gl.BindVertexArray (vertexArray);
        gl.UseProgram (program);
        std::array<float, kNativeSdfMaximumRecords * 4> gpuRecords {};
        std::array<float, kNativeSdfMaximumRecords * 4> gpuParameters0 {};
        std::array<float, kNativeSdfMaximumRecords * 4> gpuParameters1 {};
        for (std::size_t index = 0; index < compiledProgram->records().size(); ++index)
        {
            const auto& record = compiledProgram->records()[index];
            const auto offset = index * 4;
            gpuRecords[offset] = static_cast<float> (record.operation);
            gpuRecords[offset + 1] = record.input0 == std::numeric_limits<std::uint32_t>::max()
                ? -1.0f : static_cast<float> (record.input0);
            gpuRecords[offset + 2] = record.input1 == std::numeric_limits<std::uint32_t>::max()
                ? -1.0f : static_cast<float> (record.input1);
            gpuRecords[offset + 3] = static_cast<float> (record.parameterCount);
            std::copy_n (record.parameters.data(), 4, gpuParameters0.data() + offset);
            std::copy_n (record.parameters.data() + 4, 4, gpuParameters1.data() + offset);
        }
        gl.Uniform4fv (gl.GetUniformLocation (program, "uRecords"),
                       static_cast<int> (kNativeSdfMaximumRecords), gpuRecords.data());
        gl.Uniform4fv (gl.GetUniformLocation (program, "uParameters0"),
                       static_cast<int> (kNativeSdfMaximumRecords), gpuParameters0.data());
        gl.Uniform4fv (gl.GetUniformLocation (program, "uParameters1"),
                       static_cast<int> (kNativeSdfMaximumRecords), gpuParameters1.data());
        gl.Uniform1i (gl.GetUniformLocation (program, "uRootIndex"),
                      static_cast<int> (compiledProgram->rootIndex()));
        gl.Uniform1f (gl.GetUniformLocation (program, "uEpsilon"),
                      static_cast<float> (request.epsilon));
        gl.Uniform1f (gl.GetUniformLocation (program, "uMaximumDistance"),
                      static_cast<float> (request.maximumDistance));
        gl.Uniform2f (gl.GetUniformLocation (program, "uExtent"),
                      static_cast<float> (request.width), static_cast<float> (request.height));
        gl.Uniform1i (gl.GetUniformLocation (program, "uMaximumSteps"),
                      static_cast<int> (request.maximumSteps));
        gl.Uniform1i (gl.GetUniformLocation (program, "uAdaptiveQuality"),
                      static_cast<int> (request.adaptiveQuality));
        gl.Uniform1i (gl.GetUniformLocation (program, "uNormalQuality"),
                      static_cast<int> (request.normalQuality));
        gl.Uniform1i (gl.GetUniformLocation (program, "uShadowQuality"),
                      static_cast<int> (request.shadowQuality));
        gl.Uniform1i (gl.GetUniformLocation (program, "uOutputPass"),
                      static_cast<int> (request.output));
        glViewport (0, 0, static_cast<int> (request.width), static_cast<int> (request.height));
        glDisable (GL_BLEND);
        glDisable (GL_DEPTH_TEST);
        glDisable (GL_CULL_FACE);
        glDisable (GL_SCISSOR_TEST);
        glDisable (GL_RASTERIZER_DISCARD);
        glDisable (GL_COLOR_LOGIC_OP);
        glDisable (GL_FRAMEBUFFER_SRGB);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDrawArrays (GL_TRIANGLES, 0, 3);
        glFinish();
        if (glGetError() != GL_NO_ERROR)
            return fail ("OpenGL native SDF draw reported a GPU error");

        cleanup();
        result.rendered = true;
        result.frame = std::move (frame);
        return result;
    }

private:
    struct ContextKey final
    {
        GLFWwindow* pointer = nullptr;
        std::uint64_t generation = 0;
        bool operator== (const ContextKey& other) const noexcept
        {
            return pointer == other.pointer && generation == other.generation;
        }
    };
    struct ContextKeyHash final
    {
        std::size_t operator() (const ContextKey& key) const noexcept
        {
            return std::hash<GLFWwindow*> {} (key.pointer)
                ^ (std::hash<std::uint64_t> {} (key.generation) << 1u);
        }
    };
    unsigned programForCurrentContext (const arbitgl::GlFuncs& gl, std::string& error)
    {
        std::lock_guard<std::mutex> lock (programMutex_);
        const auto context = glfwGetCurrentContext();
        const ContextKey key { context, contextGenerations_[context] };
        const auto found = programs_.find (key);
        if (found != programs_.end()) return found->second;

        const auto vertex = compileShader (gl, GL_VERTEX_SHADER, kVertexShader, error);
        const auto fragment = compileShader (gl, GL_FRAGMENT_SHADER, kFragmentShader, error);
        if (vertex == 0 || fragment == 0)
        {
            if (vertex != 0) gl.DeleteShader (vertex);
            if (fragment != 0) gl.DeleteShader (fragment);
            return 0;
        }
        const auto program = gl.CreateProgram();
        gl.AttachShader (program, vertex);
        gl.AttachShader (program, fragment);
        gl.LinkProgram (program);
        gl.DeleteShader (vertex);
        gl.DeleteShader (fragment);
        int linked = 0;
        gl.GetProgramiv (program, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE)
        {
            char log[1024] = {};
            int length = 0;
            gl.GetProgramInfoLog (program, static_cast<int> (sizeof (log)), &length, log);
            gl.DeleteProgram (program);
            error = std::string ("OpenGL native SDF program link failed: ")
                + std::string (log, static_cast<std::size_t> (std::max (length, 0)));
            return 0;
        }
        programs_.emplace (key, program);
        return program;
    }

    std::mutex programMutex_;
    std::unordered_map<ContextKey, unsigned, ContextKeyHash> programs_;
    std::unordered_map<GLFWwindow*, std::uint64_t> contextGenerations_;
};

class OpenGlRenderPassOutputBackend final : public RenderPassOutputBackend
{
    struct Allocation final
    {
        explicit Allocation (const renderpassoutput::AdmittedOutputs& admitted)
            : outputs (admitted) {}
        renderpassoutput::AdmittedOutputs outputs;
        std::vector<unsigned> textures;
        std::vector<unsigned> framebuffers;
        GLFWwindow* ownerContext = nullptr;
    };

    static bool currentGl (arbitgl::GlFuncs& gl, std::string& error)
    {
        if (glfwGetCurrentContext() == nullptr)
        {
            error = "native OpenGL render-pass outputs require a current OpenGL 3.3 context";
            return false;
        }
        int major = 0, minor = 0;
        glGetIntegerv (GL_MAJOR_VERSION, &major);
        glGetIntegerv (GL_MINOR_VERSION, &minor);
        if (major < 3 || (major == 3 && minor < 3)
            || ! arbitgl::loadGlFunctions (gl, error))
        {
            if (error.empty()) error = "native OpenGL render-pass outputs require OpenGL 3.3";
            return false;
        }
        return true;
    }

    static bool textureFormat (renderpassoutput::PixelFormat format,
                               int& internal, unsigned& external, unsigned& type)
    {
        using renderpassoutput::PixelFormat;
        switch (format)
        {
            case PixelFormat::R8Unorm:
                internal = GL_R8; external = GL_RED; type = GL_UNSIGNED_BYTE; return true;
            case PixelFormat::RG16Float:
                internal = GL_RG16F; external = GL_RG; type = GL_HALF_FLOAT; return true;
            case PixelFormat::RGBA16Float:
                internal = GL_RGBA16F; external = GL_RGBA; type = GL_HALF_FLOAT; return true;
            case PixelFormat::R32Float:
                internal = GL_R32F; external = GL_RED; type = GL_FLOAT; return true;
            case PixelFormat::R32Uint:
                internal = GL_R32UI; external = GL_RED_INTEGER; type = GL_UNSIGNED_INT; return true;
            case PixelFormat::Invalid: break;
        }
        return false;
    }

    Allocation* find (RenderPassOutputLifecycleHandle lifecycle)
    {
        const auto found = allocations_.find (lifecycle.value);
        return found == allocations_.end() ? nullptr : &found->second;
    }

    static std::string fragmentSource (renderpassoutput::Output output)
    {
        const char* declaration = "layout(location=0) out vec4 outAov;";
        const char* expression = "outAov=vec4(0.0);";
        switch (output)
        {
            case renderpassoutput::Output::Depth:
                declaration = "layout(location=0) out float outAov;";
                expression = "outAov=vDepth;";
                break;
            case renderpassoutput::Output::Normal:
                expression = "outAov=vec4(normalize(vNormal),1.0);";
                break;
            case renderpassoutput::Output::Emission:
                expression = "outAov=vec4(uEmission,1.0);";
                break;
            case renderpassoutput::Output::Mask:
                declaration = "layout(location=0) out float outAov;";
                expression = "outAov=1.0;";
                break;
            case renderpassoutput::Output::MaterialId:
                declaration = "layout(location=0) out uint outAov;";
                expression = "outAov=uMaterialId;";
                break;
            case renderpassoutput::Output::ObjectId:
                declaration = "layout(location=0) out uint outAov;";
                expression = "outAov=uObjectId;";
                break;
            default: break;
        }
        return std::string ("#version 330 core\n") + declaration
            + "\nin vec3 vNormal; in float vDepth; uniform vec3 uEmission;"
              " uniform uint uMaterialId; uniform uint uObjectId; void main(){"
            + expression + "}\n";
    }

    static const char* sceneVertexSource()
    {
        return R"glsl(#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform vec4 uObjectRotation;
uniform vec3 uObjectTranslation;
uniform vec3 uObjectScale;
uniform vec4 uCameraRotation;
uniform vec3 uCameraTranslation;
uniform vec4 uProjection;
out vec3 vNormal;
out float vDepth;
vec3 rotateQ(vec4 q, vec3 v) { return v + 2.0*cross(q.xyz, cross(q.xyz,v)+q.w*v); }
void main() {
    vec3 world=rotateQ(uObjectRotation,aPosition*uObjectScale)+uObjectTranslation;
    vNormal=normalize(rotateQ(uObjectRotation,aNormal/uObjectScale));
    vec4 iq=vec4(-uCameraRotation.xyz,uCameraRotation.w);
    vec3 camera=rotateQ(iq,world-uCameraTranslation);
    float distance=-camera.z;
    vDepth=distance;
    gl_Position=vec4(camera.x/(uProjection.x*uProjection.y),camera.y/uProjection.x,
        distance*uProjection.w/(uProjection.w-uProjection.z)
          -uProjection.z*uProjection.w/(uProjection.w-uProjection.z),distance);
}
)glsl";
    }

public:
    RenderPassOutputCapabilities renderPassOutputCapabilities() const override
    {
        RenderPassOutputCapabilities result;
        arbitgl::GlFuncs gl;
        if (! currentGl (gl, result.error)) return result;
        result.available = true;
        result.supportedOutputs.fill (true);
        return result;
    }

    RenderPassOutputAdmission admitRenderPassOutputs (
        const renderpassoutput::AdmittedOutputs& outputs) override
    {
        std::lock_guard<std::mutex> lock (mutex_);
        RenderPassOutputAdmission result;
        arbitgl::GlFuncs gl;
        if (! currentGl (gl, result.error)) return result;
        const RenderPassOutputCapabilities capabilities;
        if (! admitRenderPassOutputFrameMemory (
                outputs, capabilities.limits.totalBytes,
                result.frameMemory, result.error))
            return result;
        Allocation allocation (outputs);
        allocation.ownerContext = glfwGetCurrentContext();
        auto cleanup = [&]
        {
            if (! allocation.framebuffers.empty())
                gl.DeleteFramebuffers (static_cast<int> (allocation.framebuffers.size()),
                                       allocation.framebuffers.data());
            if (! allocation.textures.empty())
                glDeleteTextures (static_cast<int> (allocation.textures.size()),
                                  allocation.textures.data());
        };
        for (const auto& attachment : outputs.attachments())
        {
            int internal = 0; unsigned external = 0, type = 0;
            if (! textureFormat (attachment.format, internal, external, type))
            {
                result.error = "OpenGL render-pass attachment format is unsupported";
                cleanup(); return result;
            }
            unsigned texture = 0, framebuffer = 0;
            glGenTextures (1, &texture);
            glBindTexture (GL_TEXTURE_2D, texture);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D (GL_TEXTURE_2D, 0, internal,
                          static_cast<int> (attachment.extent.width),
                          static_cast<int> (attachment.extent.height), 0,
                          external, type, nullptr);
            gl.GenFramebuffers (1, &framebuffer);
            gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
            gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                     GL_TEXTURE_2D, texture, 0);
            if (texture == 0 || framebuffer == 0
                || gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            {
                if (framebuffer != 0) gl.DeleteFramebuffers (1, &framebuffer);
                if (texture != 0) glDeleteTextures (1, &texture);
                result.error = "OpenGL render-pass resource allocation failed";
                cleanup(); gl.BindFramebuffer (GL_FRAMEBUFFER, 0); return result;
            }
            allocation.textures.push_back (texture);
            allocation.framebuffers.push_back (framebuffer);
            result.resources.push_back ({ attachment.output, texture, framebuffer, texture });
        }
        gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
        const RenderPassOutputLifecycleHandle lifecycle { nextLifecycle_++ };
        allocations_.emplace (lifecycle.value, std::move (allocation));
        result.lifecycle = lifecycle;
        return result;
    }

    RenderPassColorAovExecution executeColorAovClear (
        RenderPassOutputLifecycleHandle lifecycle,
        const RenderPassColorAovClear& clear) override
    {
        std::lock_guard<std::mutex> lock (mutex_);
        RenderPassColorAovExecution result;
        arbitgl::GlFuncs gl;
        if (! currentGl (gl, result.error)) return result;
        auto* allocation = find (lifecycle);
        if (allocation == nullptr || allocation->outputs.attachments().size() != 1
            || allocation->outputs.attachments()[0].output != renderpassoutput::Output::Color)
        { result.error = "OpenGL Color AOV lifecycle is stale or incompatible"; return result; }
        gl.BindFramebuffer (GL_FRAMEBUFFER, allocation->framebuffers[0]);
        gl.ClearBufferfv (GL_COLOR, 0, clear.linearRgba.data());
        gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
        result.submitted = true; result.submission = nextSubmission_++;
        return result;
    }

    RenderPassMotionAovExecution executeMotionAovClear (
        RenderPassOutputLifecycleHandle lifecycle,
        const RenderPassMotionAovClear& clear) override
    {
        std::lock_guard<std::mutex> lock (mutex_);
        RenderPassMotionAovExecution result;
        arbitgl::GlFuncs gl;
        if (! currentGl (gl, result.error)) return result;
        auto* allocation = find (lifecycle);
        if (allocation == nullptr || allocation->outputs.attachments().size() != 1
            || allocation->outputs.attachments()[0].output != renderpassoutput::Output::Motion)
        { result.error = "OpenGL Motion AOV lifecycle is stale or incompatible"; return result; }
        const float values[4] = { clear.pixelDisplacement[0], clear.pixelDisplacement[1], 0, 0 };
        gl.BindFramebuffer (GL_FRAMEBUFFER, allocation->framebuffers[0]);
        gl.ClearBufferfv (GL_COLOR, 0, values);
        gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
        result.submitted = true; result.submission = nextSubmission_++;
        return result;
    }
    RenderPassAovInspectionExecution executeAovInspection (
        RenderPassOutputLifecycleHandle lifecycle,
        const aovinspection::Payload& payload) override
    {
        std::lock_guard<std::mutex> lock (mutex_);
        RenderPassAovInspectionExecution result;
        arbitgl::GlFuncs gl;
        if (! currentGl (gl, result.error)) return result;
        auto* allocation = find (lifecycle);
        const auto sourceOutput = aovinspection::output (payload.source);
        std::size_t colorIndex = std::numeric_limits<std::size_t>::max();
        std::size_t sourceIndex = std::numeric_limits<std::size_t>::max();
        if (allocation != nullptr)
        {
            const auto& attachments = allocation->outputs.attachments();
            for (std::size_t i = 0; i < attachments.size(); ++i)
            {
                if (attachments[i].output == renderpassoutput::Output::Color) colorIndex = i;
                if (attachments[i].output == sourceOutput) sourceIndex = i;
            }
        }
        if (allocation == nullptr || allocation->ownerContext != glfwGetCurrentContext()
            || allocation->outputs.attachments().size() != 2
            || allocation->outputs.extent() != payload.extent
            || colorIndex == std::numeric_limits<std::size_t>::max()
            || sourceIndex == std::numeric_limits<std::size_t>::max()
            || ! aovinspection::finiteAndOrdered (payload))
        {
            result.error = "OpenGL AOV inspection lifecycle or payload is incompatible";
            return result;
        }

        static constexpr const char* vertexSource = R"glsl(#version 330 core
out vec2 vUv;
const vec2 positions[3] = vec2[3](vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
void main(){ vec2 p=positions[gl_VertexID]; vUv=p*0.5+0.5; gl_Position=vec4(p,0.0,1.0); }
)glsl";
        static constexpr const char* depthFragmentSource = R"glsl(#version 330 core
in vec2 vUv; layout(location=0) out vec4 outColor;
uniform sampler2D uSource; uniform vec2 uDepthRange; uniform int uInvert;
void main(){ float d=texture(uSource,vUv).r;
float value=clamp((d-uDepthRange.x)/(uDepthRange.y-uDepthRange.x),0.0,1.0);
if(uInvert!=0) value=1.0-value; outColor=vec4(value,value,value,1.0); }
)glsl";
        static constexpr const char* normalFragmentSource = R"glsl(#version 330 core
in vec2 vUv; layout(location=0) out vec4 outColor; uniform sampler2D uSource;
void main(){ vec3 n=normalize(texture(uSource,vUv).xyz); outColor=vec4(n*0.5+0.5,1.0); }
)glsl";

        int previousProgram = 0, previousVao = 0, previousActiveTexture = 0;
        int previousTexture0 = 0, previousDrawFramebuffer = 0, previousReadFramebuffer = 0;
        int previousPolygonMode[2] {}, previousViewport[4] {};
        unsigned char previousColorMask[4] {};
        glGetIntegerv (GL_CURRENT_PROGRAM, &previousProgram);
        glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &previousVao);
        glGetIntegerv (GL_ACTIVE_TEXTURE, &previousActiveTexture);
        glGetIntegerv (GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
        glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
        glGetIntegerv (GL_POLYGON_MODE, previousPolygonMode);
        glGetIntegerv (GL_VIEWPORT, previousViewport);
        glGetBooleanv (GL_COLOR_WRITEMASK, previousColorMask);
        const bool previousBlend = glIsEnabled (GL_BLEND) == GL_TRUE;
        const bool previousDepthTest = glIsEnabled (GL_DEPTH_TEST) == GL_TRUE;
        const bool previousCullFace = glIsEnabled (GL_CULL_FACE) == GL_TRUE;
        const bool previousScissor = glIsEnabled (GL_SCISSOR_TEST) == GL_TRUE;
        const bool previousFramebufferSrgb = glIsEnabled (GL_FRAMEBUFFER_SRGB) == GL_TRUE;
        const bool previousRasterizerDiscard = glIsEnabled (GL_RASTERIZER_DISCARD) == GL_TRUE;
        gl.ActiveTexture (GL_TEXTURE0);
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &previousTexture0);
        gl.ActiveTexture (static_cast<unsigned> (previousActiveTexture));

        unsigned vertexShader = 0, fragmentShader = 0, program = 0, vao = 0;
        auto cleanup = [&]
        {
            gl.ActiveTexture (GL_TEXTURE0);
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTexture0));
            gl.ActiveTexture (static_cast<unsigned> (previousActiveTexture));
            gl.BindFramebuffer (GL_DRAW_FRAMEBUFFER,
                                static_cast<unsigned> (previousDrawFramebuffer));
            gl.BindFramebuffer (GL_READ_FRAMEBUFFER,
                                static_cast<unsigned> (previousReadFramebuffer));
            glViewport (previousViewport[0], previousViewport[1],
                        previousViewport[2], previousViewport[3]);
            const auto restoreEnable = [] (unsigned capability, bool enabled)
            { enabled ? glEnable (capability) : glDisable (capability); };
            restoreEnable (GL_BLEND, previousBlend);
            restoreEnable (GL_DEPTH_TEST, previousDepthTest);
            restoreEnable (GL_CULL_FACE, previousCullFace);
            restoreEnable (GL_SCISSOR_TEST, previousScissor);
            restoreEnable (GL_FRAMEBUFFER_SRGB, previousFramebufferSrgb);
            restoreEnable (GL_RASTERIZER_DISCARD, previousRasterizerDiscard);
            glColorMask (previousColorMask[0], previousColorMask[1],
                         previousColorMask[2], previousColorMask[3]);
            glPolygonMode (GL_FRONT, static_cast<unsigned> (previousPolygonMode[0]));
            glPolygonMode (GL_BACK, static_cast<unsigned> (previousPolygonMode[1]));
            gl.UseProgram (static_cast<unsigned> (previousProgram));
            gl.BindVertexArray (static_cast<unsigned> (previousVao));
            if (vao != 0) gl.DeleteVertexArrays (1, &vao);
            if (program != 0) gl.DeleteProgram (program);
            if (fragmentShader != 0) gl.DeleteShader (fragmentShader);
            if (vertexShader != 0) gl.DeleteShader (vertexShader);
        };

        vertexShader = compileShader (gl, GL_VERTEX_SHADER, vertexSource, result.error);
        if (vertexShader != 0)
            fragmentShader = compileShader (gl, GL_FRAGMENT_SHADER,
                payload.source == aovinspection::Source::Depth
                    ? depthFragmentSource : normalFragmentSource,
                result.error);
        if (vertexShader == 0 || fragmentShader == 0) { cleanup(); return result; }
        program = gl.CreateProgram();
        gl.AttachShader (program, vertexShader);
        gl.AttachShader (program, fragmentShader);
        gl.LinkProgram (program);
        int linked = 0;
        gl.GetProgramiv (program, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE)
        {
            result.error = "OpenGL AOV inspection shader link failed";
            cleanup();
            return result;
        }

        gl.BindFramebuffer (GL_FRAMEBUFFER, allocation->framebuffers[colorIndex]);
        glViewport (0, 0, static_cast<int> (payload.extent.width),
                    static_cast<int> (payload.extent.height));
        glDisable (GL_BLEND); glDisable (GL_DEPTH_TEST); glDisable (GL_CULL_FACE);
        glDisable (GL_SCISSOR_TEST); glDisable (GL_FRAMEBUFFER_SRGB);
        glDisable (GL_RASTERIZER_DISCARD);
        glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        gl.UseProgram (program);
        gl.ActiveTexture (GL_TEXTURE0);
        glBindTexture (GL_TEXTURE_2D, allocation->textures[sourceIndex]);
        gl.Uniform1i (gl.GetUniformLocation (program, "uSource"), 0);
        if (payload.source == aovinspection::Source::Depth)
        {
            gl.Uniform2f (gl.GetUniformLocation (program, "uDepthRange"),
                          payload.depthNear, payload.depthFar);
            gl.Uniform1i (gl.GetUniformLocation (program, "uInvert"),
                          payload.invertDepth ? 1 : 0);
        }
        gl.GenVertexArrays (1, &vao);
        gl.BindVertexArray (vao);
        glDrawArrays (GL_TRIANGLES, 0, 3);
        const auto gpuError = glGetError();
        cleanup();
        if (gpuError != GL_NO_ERROR)
        {
            result.error = "OpenGL AOV inspection draw reported a GPU error";
            return result;
        }
        result.submitted = true;
        result.submission = nextSubmission_++;
        return result;
    }

    RenderPassSceneAovExecution executeSceneAov (
        RenderPassOutputLifecycleHandle lifecycle,
        const sceneaov::Payload& payload) override
    {
        using namespace HarmonicMIDI::grid;
        std::lock_guard<std::mutex> lock (mutex_);
        RenderPassSceneAovExecution result;
        arbitgl::GlFuncs gl;
        if (! currentGl (gl, result.error)) return result;
        auto* allocation = find (lifecycle);
        std::size_t outputIndex = std::numeric_limits<std::size_t>::max();
        if (allocation != nullptr)
        {
            const auto& attachments = allocation->outputs.attachments();
            for (std::size_t i = 0; i < attachments.size(); ++i)
                if (attachments[i].output == payload.output) outputIndex = i;
        }
        if (! sceneaov::valid (payload) || allocation == nullptr
            || allocation->ownerContext != glfwGetCurrentContext()
            || outputIndex == std::numeric_limits<std::size_t>::max()
            || allocation->outputs.extent().width != payload.extent.width
            || allocation->outputs.extent().height != payload.extent.height
            || payload.scene->objects.size() != 1)
        { result.error = "OpenGL scene AOV lifecycle or payload is incompatible"; return result; }

        const auto& scene = *payload.scene;
        const auto& object = scene.objects[0];
        const auto materialIt = std::find_if (scene.materials.begin(), scene.materials.end(),
            [&] (const auto& candidate) { return candidate.id == object.material; });
        const auto cameraIt = std::find_if (scene.cameras.begin(), scene.cameras.end(),
            [&] (const auto& candidate) { return candidate.id == scene.activeCamera; });
        if (materialIt == scene.materials.end() || cameraIt == scene.cameras.end())
        { result.error = "OpenGL scene AOV references unavailable material or camera"; return result; }
        const auto& material = *materialIt;
        const auto& camera = *cameraIt;
        int previousProgram = 0, previousVao = 0, previousArrayBuffer = 0;
        int previousTexture = 0, previousDrawFramebuffer = 0, previousReadFramebuffer = 0;
        int previousDepthFunction = 0, previousPolygonMode[2] {}, previousViewport[4] {};
        unsigned char previousColorMask[4] {};
        double previousDepthClear = 1.0;
        glGetIntegerv (GL_CURRENT_PROGRAM, &previousProgram);
        glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &previousVao);
        glGetIntegerv (GL_ARRAY_BUFFER_BINDING, &previousArrayBuffer);
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &previousTexture);
        glGetIntegerv (GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFramebuffer);
        glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &previousReadFramebuffer);
        glGetIntegerv (GL_DEPTH_FUNC, &previousDepthFunction);
        glGetIntegerv (GL_POLYGON_MODE, previousPolygonMode);
        glGetIntegerv (GL_VIEWPORT, previousViewport);
        glGetDoublev (GL_DEPTH_CLEAR_VALUE, &previousDepthClear);
        glGetBooleanv (GL_COLOR_WRITEMASK, previousColorMask);
        const bool previousDepthTest = glIsEnabled (GL_DEPTH_TEST) == GL_TRUE;
        const bool previousBlend = glIsEnabled (GL_BLEND) == GL_TRUE;
        const bool previousCullFace = glIsEnabled (GL_CULL_FACE) == GL_TRUE;
        const bool previousScissor = glIsEnabled (GL_SCISSOR_TEST) == GL_TRUE;
        const bool previousFramebufferSrgb = glIsEnabled (GL_FRAMEBUFFER_SRGB) == GL_TRUE;
        const bool previousRasterizerDiscard = glIsEnabled (GL_RASTERIZER_DISCARD) == GL_TRUE;
        unsigned depth = 0, vao = 0, vertices = 0, indices = 0;
        unsigned vertexShader = 0, fragmentShader = 0, program = 0;
        auto cleanup = [&]
        {
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTexture));
            gl.BindFramebuffer (GL_DRAW_FRAMEBUFFER,
                                static_cast<unsigned> (previousDrawFramebuffer));
            gl.BindFramebuffer (GL_READ_FRAMEBUFFER,
                                static_cast<unsigned> (previousReadFramebuffer));
            glViewport (previousViewport[0], previousViewport[1],
                        previousViewport[2], previousViewport[3]);
            previousDepthTest ? glEnable (GL_DEPTH_TEST) : glDisable (GL_DEPTH_TEST);
            previousBlend ? glEnable (GL_BLEND) : glDisable (GL_BLEND);
            previousCullFace ? glEnable (GL_CULL_FACE) : glDisable (GL_CULL_FACE);
            previousScissor ? glEnable (GL_SCISSOR_TEST) : glDisable (GL_SCISSOR_TEST);
            previousFramebufferSrgb ? glEnable (GL_FRAMEBUFFER_SRGB)
                                    : glDisable (GL_FRAMEBUFFER_SRGB);
            previousRasterizerDiscard ? glEnable (GL_RASTERIZER_DISCARD)
                                      : glDisable (GL_RASTERIZER_DISCARD);
            glDepthFunc (static_cast<unsigned> (previousDepthFunction));
            glClearDepth (previousDepthClear);
            glColorMask (previousColorMask[0], previousColorMask[1],
                         previousColorMask[2], previousColorMask[3]);
            glPolygonMode (GL_FRONT, static_cast<unsigned> (previousPolygonMode[0]));
            glPolygonMode (GL_BACK, static_cast<unsigned> (previousPolygonMode[1]));
            gl.UseProgram (static_cast<unsigned> (previousProgram));
            gl.BindVertexArray (static_cast<unsigned> (previousVao));
            gl.BindBuffer (GL_ARRAY_BUFFER, static_cast<unsigned> (previousArrayBuffer));
            if (program) gl.DeleteProgram (program);
            if (fragmentShader) gl.DeleteShader (fragmentShader);
            if (vertexShader) gl.DeleteShader (vertexShader);
            if (indices) gl.DeleteBuffers (1, &indices);
            if (vertices) gl.DeleteBuffers (1, &vertices);
            if (vao) gl.DeleteVertexArrays (1, &vao);
            if (depth) glDeleteTextures (1, &depth);
        };

        glGenTextures (1, &depth);
        glBindTexture (GL_TEXTURE_2D, depth);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
                      static_cast<int> (payload.extent.width),
                      static_cast<int> (payload.extent.height), 0,
                      GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        gl.BindFramebuffer (GL_FRAMEBUFFER, allocation->framebuffers[outputIndex]);
        gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                 GL_TEXTURE_2D, depth, 0);
        if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { result.error = "OpenGL scene AOV framebuffer is incomplete"; cleanup(); return result; }

        vertexShader = compileShader (gl, GL_VERTEX_SHADER, sceneVertexSource(), result.error);
        const auto fragment = fragmentSource (payload.output);
        if (vertexShader != 0)
            fragmentShader = compileShader (gl, GL_FRAGMENT_SHADER, fragment.c_str(), result.error);
        if (vertexShader == 0 || fragmentShader == 0) { cleanup(); return result; }
        program = gl.CreateProgram();
        gl.AttachShader (program, vertexShader); gl.AttachShader (program, fragmentShader);
        gl.LinkProgram (program);
        int linked = 0; gl.GetProgramiv (program, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE)
        { result.error = "OpenGL scene AOV shader link failed"; cleanup(); return result; }

        gl.GenVertexArrays (1, &vao); gl.BindVertexArray (vao);
        gl.GenBuffers (1, &vertices); gl.BindBuffer (GL_ARRAY_BUFFER, vertices);
        gl.BufferData (GL_ARRAY_BUFFER,
                       static_cast<std::ptrdiff_t> (object.vertexCount * sizeof (SceneVertex)),
                       scene.vertices.data() + object.firstVertex, GL_STATIC_DRAW);
        gl.GenBuffers (1, &indices); gl.BindBuffer (GL_ELEMENT_ARRAY_BUFFER, indices);
        gl.BufferData (GL_ELEMENT_ARRAY_BUFFER,
                       static_cast<std::ptrdiff_t> (object.indexCount * sizeof (std::uint32_t)),
                       scene.indices.data() + object.firstIndex, GL_STATIC_DRAW);
        gl.VertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, sizeof (SceneVertex),
                                reinterpret_cast<const void*> (offsetof (SceneVertex, position)));
        gl.VertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE, sizeof (SceneVertex),
                                reinterpret_cast<const void*> (offsetof (SceneVertex, normal)));
        gl.EnableVertexAttribArray (0); gl.EnableVertexAttribArray (1);

        glViewport (0, 0, static_cast<int> (payload.extent.width),
                    static_cast<int> (payload.extent.height));
        glDisable (GL_BLEND); glDisable (GL_CULL_FACE); glDisable (GL_SCISSOR_TEST);
        glDisable (GL_FRAMEBUFFER_SRGB);
        glDisable (GL_RASTERIZER_DISCARD);
        glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        if (payload.output == renderpassoutput::Output::MaterialId
            || payload.output == renderpassoutput::Output::ObjectId)
        { const unsigned zero = 0; gl.ClearBufferuiv (GL_COLOR, 0, &zero); }
        else
        { const float zero[4] = {}; gl.ClearBufferfv (GL_COLOR, 0, zero); }
        glClearDepth (1.0); glClear (GL_DEPTH_BUFFER_BIT);
        glEnable (GL_DEPTH_TEST); glDepthFunc (GL_LESS);
        gl.UseProgram (program);
        const auto set4 = [&] (const char* name, const SceneQuaternion& q)
        { gl.Uniform4f (gl.GetUniformLocation (program, name), q.x, q.y, q.z, q.w); };
        const auto set3 = [&] (const char* name, const SceneVec3& v)
        { gl.Uniform3f (gl.GetUniformLocation (program, name), v.x, v.y, v.z); };
        set4 ("uObjectRotation", object.transform.rotation);
        set3 ("uObjectTranslation", object.transform.translation);
        set3 ("uObjectScale", object.transform.scale);
        set4 ("uCameraRotation", camera.transform.rotation);
        set3 ("uCameraTranslation", camera.transform.translation);
        gl.Uniform4f (gl.GetUniformLocation (program, "uProjection"),
                      std::tan (camera.verticalFovRadians * 0.5f),
                      static_cast<float> (payload.extent.width) / payload.extent.height,
                      camera.nearPlane, camera.farPlane);
        set3 ("uEmission", material.emissive);
        gl.Uniform1ui (gl.GetUniformLocation (program, "uMaterialId"), material.id.value);
        gl.Uniform1ui (gl.GetUniformLocation (program, "uObjectId"), object.id.value);
        glDrawElements (GL_TRIANGLES, static_cast<int> (object.indexCount),
                        GL_UNSIGNED_INT, nullptr);
        glDisable (GL_DEPTH_TEST);
        cleanup();
        result.submitted = true; result.submission = nextSubmission_++;
        return result;
    }

    void releaseRenderPassOutputs (RenderPassOutputLifecycleHandle lifecycle) noexcept override
    {
        std::lock_guard<std::mutex> lock (mutex_);
        const auto found = allocations_.find (lifecycle.value);
        if (found == allocations_.end()) return;
        auto& allocation = found->second;
        auto* previous = glfwGetCurrentContext();
        if (previous != allocation.ownerContext) glfwMakeContextCurrent (allocation.ownerContext);
        arbitgl::GlFuncs gl; std::string ignored;
        if (currentGl (gl, ignored))
        {
            if (! allocation.framebuffers.empty())
                gl.DeleteFramebuffers (static_cast<int> (allocation.framebuffers.size()),
                                       allocation.framebuffers.data());
            if (! allocation.textures.empty())
                glDeleteTextures (static_cast<int> (allocation.textures.size()),
                                  allocation.textures.data());
        }
        if (previous != allocation.ownerContext) glfwMakeContextCurrent (previous);
        allocations_.erase (found);
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::uint64_t, Allocation> allocations_;
    std::uint64_t nextLifecycle_ = 1;
    std::uint64_t nextSubmission_ = 1;
};

class UnavailableOpticalFlowBackend final : public NativeOpticalFlowExecutionBackend
{
public:
    videoopticalflow::BackendCapabilities opticalFlowCapabilities() const override
    {
        return {};
    }
    NativeOpticalFlowSubmission executeOpticalFlow (
        const videoopticalflow::AdmittedRequest&,
        const NativeOpticalFlowInputResource&,
        const NativeOpticalFlowInputResource&,
        bool) override
    {
        NativeOpticalFlowSubmission result;
        result.error = "native optical-flow execution requires the Metal compute backend";
        return result;
    }
    void releaseOpticalFlowOutput (NativeOpticalFlowOutputLifecycleHandle) noexcept override {}
};

constexpr const char* kFixtureUnavailable
    = "native OpenGL fixture execution requires a current OpenGL 3.3 context";

std::mutex& fixtureMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::uint64_t nextFixtureRendererGeneration() noexcept
{
    static std::uint64_t generation = 0;
    return ++generation;
}

struct OpenGlFixtureState final
{
    explicit OpenGlFixtureState (const arbitgl::GlFuncs& functions) : gl (functions)
    {
        glGetIntegerv (GL_DRAW_FRAMEBUFFER_BINDING, &drawFramebuffer);
        glGetIntegerv (GL_READ_FRAMEBUFFER_BINDING, &readFramebuffer);
        glGetIntegerv (GL_CURRENT_PROGRAM, &program);
        glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &vertexArray);
        glGetIntegerv (GL_ARRAY_BUFFER_BINDING, &arrayBuffer);
        glGetIntegerv (GL_ELEMENT_ARRAY_BUFFER_BINDING, &elementArrayBuffer);
        glGetIntegerv (GL_ACTIVE_TEXTURE, &activeTexture);
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &activeTextureBinding);
        for (std::size_t unit = 0; unit < textureBindings.size(); ++unit)
        {
            gl.ActiveTexture (GL_TEXTURE0 + static_cast<unsigned> (unit));
            glGetIntegerv (GL_TEXTURE_BINDING_2D, &textureBindings[unit]);
            glGetIntegerv (GL_SAMPLER_BINDING, &samplerBindings[unit]);
        }
        gl.ActiveTexture (static_cast<unsigned> (activeTexture));
        glGetIntegerv (GL_VIEWPORT, viewport.data());
        glGetIntegerv (GL_POLYGON_MODE, polygonMode.data());
        glGetIntegerv (GL_DEPTH_FUNC, &depthFunction);
        glGetIntegerv (GL_FRONT_FACE, &frontFace);
        glGetIntegerv (GL_PIXEL_UNPACK_BUFFER_BINDING, &pixelUnpackBuffer);
        glGetIntegerv (GL_UNPACK_ALIGNMENT, &unpackAlignment);
        glGetIntegerv (GL_UNPACK_ROW_LENGTH, &unpackRowLength);
        glGetIntegerv (GL_UNPACK_SKIP_PIXELS, &unpackSkipPixels);
        glGetIntegerv (GL_UNPACK_SKIP_ROWS, &unpackSkipRows);
        glGetBooleanv (GL_COLOR_WRITEMASK, colorMask.data());
        glGetBooleanv (GL_DEPTH_WRITEMASK, &depthMask);
        glGetFloatv (GL_COLOR_CLEAR_VALUE, clearColor.data());
        glGetDoublev (GL_DEPTH_CLEAR_VALUE, &clearDepth);
        blend = glIsEnabled (GL_BLEND) == GL_TRUE;
        depthTest = glIsEnabled (GL_DEPTH_TEST) == GL_TRUE;
        cullFace = glIsEnabled (GL_CULL_FACE) == GL_TRUE;
        scissor = glIsEnabled (GL_SCISSOR_TEST) == GL_TRUE;
        rasterizerDiscard = glIsEnabled (GL_RASTERIZER_DISCARD) == GL_TRUE;
        colorLogicOp = glIsEnabled (GL_COLOR_LOGIC_OP) == GL_TRUE;
        framebufferSrgb = glIsEnabled (GL_FRAMEBUFFER_SRGB) == GL_TRUE;
    }

    void restore() const noexcept
    {
        for (std::size_t unit = 0; unit < textureBindings.size(); ++unit)
        {
            gl.ActiveTexture (GL_TEXTURE0 + static_cast<unsigned> (unit));
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (textureBindings[unit]));
            gl.BindSampler (static_cast<unsigned> (unit),
                            static_cast<unsigned> (samplerBindings[unit]));
        }
        gl.ActiveTexture (static_cast<unsigned> (activeTexture));
        glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (activeTextureBinding));
        gl.BindFramebuffer (GL_DRAW_FRAMEBUFFER, static_cast<unsigned> (drawFramebuffer));
        gl.BindFramebuffer (GL_READ_FRAMEBUFFER, static_cast<unsigned> (readFramebuffer));
        glViewport (viewport[0], viewport[1], viewport[2], viewport[3]);
        glPolygonMode (GL_FRONT_AND_BACK, static_cast<unsigned> (polygonMode[0]));
        glColorMask (colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
        glDepthMask (depthMask);
        glDepthFunc (static_cast<unsigned> (depthFunction));
        glFrontFace (static_cast<unsigned> (frontFace));
        glClearColor (clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
        glClearDepth (clearDepth);
        gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER,
                       static_cast<unsigned> (pixelUnpackBuffer));
        glPixelStorei (GL_UNPACK_ALIGNMENT, unpackAlignment);
        glPixelStorei (GL_UNPACK_ROW_LENGTH, unpackRowLength);
        glPixelStorei (GL_UNPACK_SKIP_PIXELS, unpackSkipPixels);
        glPixelStorei (GL_UNPACK_SKIP_ROWS, unpackSkipRows);
        restoreEnable (GL_BLEND, blend);
        restoreEnable (GL_DEPTH_TEST, depthTest);
        restoreEnable (GL_CULL_FACE, cullFace);
        restoreEnable (GL_SCISSOR_TEST, scissor);
        restoreEnable (GL_RASTERIZER_DISCARD, rasterizerDiscard);
        restoreEnable (GL_COLOR_LOGIC_OP, colorLogicOp);
        restoreEnable (GL_FRAMEBUFFER_SRGB, framebufferSrgb);
        gl.UseProgram (static_cast<unsigned> (program));
        gl.BindVertexArray (static_cast<unsigned> (vertexArray));
        gl.BindBuffer (GL_ARRAY_BUFFER, static_cast<unsigned> (arrayBuffer));
        gl.BindBuffer (GL_ELEMENT_ARRAY_BUFFER,
                       static_cast<unsigned> (elementArrayBuffer));
    }

    static void restoreEnable (unsigned capability, bool enabled) noexcept
    {
        enabled ? glEnable (capability) : glDisable (capability);
    }

    const arbitgl::GlFuncs& gl;
    int drawFramebuffer = 0;
    int readFramebuffer = 0;
    int program = 0;
    int vertexArray = 0;
    int arrayBuffer = 0;
    int elementArrayBuffer = 0;
    int activeTexture = GL_TEXTURE0;
    int activeTextureBinding = 0;
    std::array<int, 5> textureBindings {};
    std::array<int, 5> samplerBindings {};
    std::array<int, 4> viewport {};
    std::array<int, 2> polygonMode {};
    int depthFunction = GL_LESS;
    int frontFace = GL_CCW;
    int pixelUnpackBuffer = 0;
    int unpackAlignment = 4;
    int unpackRowLength = 0;
    int unpackSkipPixels = 0;
    int unpackSkipRows = 0;
    std::array<unsigned char, 4> colorMask {};
    unsigned char depthMask = GL_TRUE;
    std::array<float, 4> clearColor {};
    double clearDepth = 1.0;
    bool blend = false;
    bool depthTest = false;
    bool cullFace = false;
    bool scissor = false;
    bool rasterizerDiscard = false;
    bool colorLogicOp = false;
    bool framebufferSrgb = false;
};

const char* kFixtureVertexShader = R"glsl(#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUv;
layout(location=3) in vec4 aColor;
layout(location=4) in vec4 aTangent;
layout(location=5) in vec4 aInstanceMatrix0;
layout(location=6) in vec4 aInstanceMatrix1;
layout(location=7) in vec4 aInstanceMatrix2;
layout(location=8) in vec4 aInstanceMatrix3;
layout(location=9) in vec4 aInstanceIdentityColor;
uniform mat4 uObjectMatrix;
uniform float uTangentHandednessSign;
uniform int uUseInstanceMatrix;
uniform vec4 uCameraRotation;
uniform vec3 uCameraTranslation;
uniform vec4 uProjection;
uniform vec4 uBaseColor;
uniform vec4 uMaterialParams;
uniform vec4 uTimeMixEndColorAndTime;
uniform vec3 uEmissive;
uniform vec3 uAmbient;
uniform vec4 uNoteInstanceTransforms[128];
uniform int uNoteInstanceIndex;
out vec2 vUv;
out vec3 vBaseColor;
out vec3 vLitBase;
out vec3 vEmissive;
out vec3 vWorldPosition;
out vec3 vWorldNormal;
out vec3 vWorldTangent;
flat out float vBitangentSign;
out float vLinearDepth;
out float vOpacity;
out vec2 vMetallicRoughness;
flat out vec4 vInstanceIdentityColor;
vec3 rotateQ(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
void main() {
    mat4 objectMatrix = uUseInstanceMatrix != 0
        ? mat4(aInstanceMatrix0, aInstanceMatrix1,
               aInstanceMatrix2, aInstanceMatrix3)
        : uObjectMatrix;
    int noteIndex = uNoteInstanceIndex >= 0 ? uNoteInstanceIndex : gl_InstanceID;
    vec4 noteInstance = uNoteInstanceTransforms[noteIndex];
    vec3 world = (objectMatrix
        * vec4(aPosition * noteInstance.w + noteInstance.xyz, 1.0)).xyz;
    vec3 worldNormal = normalize(transpose(inverse(mat3(objectMatrix))) * aNormal);
    vec4 inverseCamera = vec4(-uCameraRotation.xyz, uCameraRotation.w);
    vec3 camera = rotateQ(inverseCamera, world - uCameraTranslation);
    float distance = -camera.z;
    gl_Position = vec4(camera.x / (uProjection.x * uProjection.y),
                       camera.y / uProjection.x,
                       distance * (uProjection.w + uProjection.z)
                                   / (uProjection.w - uProjection.z)
                           - 2.0 * uProjection.z * uProjection.w
                               / (uProjection.w - uProjection.z),
                       distance);
    float metallic = uMaterialParams.x;
    float roughness = uMaterialParams.y;
    vec3 resolvedBaseColor = uMaterialParams.w > 0.5
        ? mix(uBaseColor.xyz, uTimeMixEndColorAndTime.xyz,
              clamp(uTimeMixEndColorAndTime.w, 0.0, 1.0))
        : uBaseColor.xyz;
    vUv = aUv;
    vBaseColor = resolvedBaseColor * aColor.rgb
        * (uUseInstanceMatrix != 0
            ? mix(vec3(1.0), aInstanceIdentityColor.rgb,
                  step(0.5, aInstanceIdentityColor.a))
            : vec3(1.0));
    vLitBase = vBaseColor * uAmbient;
    vEmissive = uEmissive;
    vWorldPosition = world;
    vWorldNormal = worldNormal;
    vWorldTangent = normalize(mat3(objectMatrix) * aTangent.xyz);
    vBitangentSign = aTangent.w * uTangentHandednessSign
        * (uUseInstanceMatrix != 0 ? sign(determinant(mat3(objectMatrix))) : 1.0);
    vMetallicRoughness = vec2(metallic, roughness);
    vLinearDepth = clamp((distance - uProjection.z)
        / max(uProjection.w - uProjection.z, 0.000001), 0.0, 1.0);
    vOpacity = uBaseColor.w * aColor.a;
    vInstanceIdentityColor = uUseInstanceMatrix != 0
        ? aInstanceIdentityColor : vec4(0.0);
}
)glsl";

const char* kFixtureFragmentShader = R"glsl(#version 330 core
in vec2 vUv;
in vec3 vBaseColor;
in vec3 vLitBase;
in vec3 vEmissive;
in vec3 vWorldPosition;
in vec3 vWorldNormal;
in vec3 vWorldTangent;
flat in float vBitangentSign;
in vec2 vMetallicRoughness;
in float vLinearDepth;
in float vOpacity;
flat in vec4 vInstanceIdentityColor;
uniform sampler2D uBaseTexture;
uniform sampler2D uMetallicRoughnessTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uOcclusionTexture;
uniform sampler2D uEmissiveTexture;
uniform vec4 uTexturePresence;
uniform float uEmissiveTexturePresence;
uniform vec2 uAlphaModeCutoff;
uniform vec4 uMaterialParams;
uniform vec3 uAmbient;
uniform int uMaterialKind;
uniform vec3 uCameraTranslation;
uniform vec4 uLightRotation;
uniform int uLightCount;
uniform vec4 uLightRotations[16];
uniform vec4 uLightColors[16];
uniform vec4 uLightPositions[16];
uniform vec4 uLightCones[16];
uniform int uLightKinds[16];
uniform vec4 uDiffractionGeometry;
uniform vec4 uDiffractionSecondaryGeometry;
uniform vec4 uDiffractionMicrostructure;
uniform vec4 uDiffractionControl;
uniform vec4 uDiffractionCoating;
uniform vec4 uDiffractionRoughness;
uniform vec4 uDiffractionGrooveField;
uniform vec4 uDiffractionGrooveVariation;
uniform vec4 uDiffractionSpectral[8];
uniform vec4 uDiffractionSpectralZ[8];
uniform vec4 uDiffractionIncidentDirectionAndIntensity;
uniform vec4 uDiffractionPathKindAndBounce;
uniform vec4 uDiffractionFoilField[4];
uniform vec4 uDiffractionOccupancyRectangles[5];
uniform vec4 uDiffractionSpatialCounts;
uniform vec4 uDiffractionEvaluationSchedule;
layout(location=0) out vec4 outColor;
layout(location=1) out float outLinearDepth;

const float pi = 3.14159265358979323846;
const int materialDiffractionReflective = 1;
const int profileBinaryRectangular = 1;
const int profileSinusoidal = 2;
const int profileBlazedSawtooth = 3;
const int coatingUncoated = 1;
const int coatingIncoherentDielectric = 2;

vec3 srgbToLinear(vec3 value) {
    bvec3 low = lessThanEqual(value, vec3(0.04045));
    vec3 linearLow = value / 12.92;
    vec3 linearHigh = pow((value + 0.055) / 1.055, vec3(2.4));
    return mix(linearHigh, linearLow, low);
}

vec3 rotateQ(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

float interfaceReflectance(vec2 first, vec2 second) {
    vec2 difference = first - second;
    vec2 sum = first + second;
    return clamp(dot(difference, difference) / dot(sum, sum), 0.0, 1.0);
}

float spectralReflectance(float wavelength, int coatingModel,
                          float substrateN, float substrateK) {
    if (coatingModel == coatingUncoated)
        return interfaceReflectance(vec2(1.0, 0.0), vec2(substrateN, substrateK));
    if (coatingModel != coatingIncoherentDielectric)
        return -1.0;
    vec2 layer = vec2(uDiffractionCoating.y, uDiffractionCoating.z);
    float r01 = interfaceReflectance(vec2(1.0, 0.0), layer);
    float r12 = interfaceReflectance(layer, vec2(substrateN, substrateK));
    float roundTrip = exp(-8.0 * pi * uDiffractionCoating.z
                          * uDiffractionCoating.x / wavelength);
    float denominator = 1.0 - r01 * r12 * roundTrip;
    return clamp(r01 + (1.0 - r01) * (1.0 - r01)
                    * r12 * roundTrip / denominator, 0.0, 1.0);
}

vec2 complexMultiply(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 complexExponentialIntegral(float frequency, float begin, float end) {
    if (abs(frequency) < 1.0e-6)
        return vec2(end - begin, 0.0);
    float beginPhase = frequency * begin;
    float endPhase = frequency * end;
    return vec2((sin(endPhase) - sin(beginPhase)) / frequency,
                -(cos(endPhase) - cos(beginPhase)) / frequency);
}

float besselJ(int order, float argument) {
    int magnitude = order < 0 ? -order : order;
    float halfArgument = 0.5 * argument;
    float term = 1.0;
    for (int factor = 1; factor <= 8; ++factor)
        if (factor <= magnitude)
            term *= halfArgument / float(factor);
    float sum = term;
    for (int seriesIndex = 1; seriesIndex <= 24; ++seriesIndex) {
        term *= -(halfArgument * halfArgument)
              / (float(seriesIndex) * float(magnitude + seriesIndex));
        sum += term;
    }
    return sum;
}

float profileEfficiency(int profile, int signedOrder, float duty,
                        float phase, vec2 terracePhase) {
    if (profile == profileBinaryRectangular) {
        if (signedOrder == 0) {
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
    if (profile == profileSinusoidal) {
        float amplitude = besselJ(signedOrder, 0.5 * phase);
        return amplitude * amplitude;
    }
    if (profile == profileBlazedSawtooth) {
        float orderFrequency = -2.0 * pi * float(signedOrder);
        vec2 ramp = complexExponentialIntegral(
            phase / duty + orderFrequency, 0.0, duty);
        vec2 land = complexExponentialIntegral(orderFrequency, duty, 1.0);
        vec2 amplitude = ramp + land;
        return dot(amplitude, amplitude);
    }
    return 0.0;
}

void resolveGrooveGeometry(vec2 uv, out vec4 geometry,
                           out vec4 secondaryGeometry) {
    geometry = uDiffractionGeometry;
    secondaryGeometry = uDiffractionSecondaryGeometry;
    if (uDiffractionSpatialCounts.x > 0.0) {
        vec4 lower = mix(uDiffractionFoilField[0], uDiffractionFoilField[1], clamp(uv.x, 0.0, 1.0));
        vec4 upper = mix(uDiffractionFoilField[2], uDiffractionFoilField[3], clamp(uv.x, 0.0, 1.0));
        vec4 local = mix(lower, upper, clamp(uv.y, 0.0, 1.0));
        geometry.xyz = vec3(normalize(local.xy), local.z);
        return;
    }
    int mode = int(uDiffractionGrooveField.x + 0.5);
    if (mode == 1)
        return;
    vec2 offset = uv - uDiffractionGrooveField.yz;
    float coordinate = mode == 2
        ? dot(offset, uDiffractionGrooveVariation.xy) : length(offset);
    geometry.z += coordinate * uDiffractionGrooveVariation.z;
    secondaryGeometry.z += coordinate * uDiffractionGrooveVariation.w;
    float angle = radians(coordinate * uDiffractionGrooveField.w);
    mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    geometry.xy = rotation * geometry.xy;
    secondaryGeometry.xy = rotation * secondaryGeometry.xy;
}

float diffractionOrderContribution(int primaryOrder,
                                   int secondaryOrder,
                                   bool crossedTwoDimensional,
                                   float wavelength,
                                   vec3 incident,
                                   vec3 outgoing,
                                   int profile,
                                   float duty,
                                   float sigma) {
    int firstOrder = int(round(uDiffractionMicrostructure.w));
    int lastOrder = int(uDiffractionControl.x + 0.5);
    int primaryMagnitude = primaryOrder < 0 ? -primaryOrder : primaryOrder;
    int secondaryMagnitude = secondaryOrder < 0 ? -secondaryOrder : secondaryOrder;
    if ((primaryOrder != 0
            && (primaryMagnitude < firstOrder || primaryMagnitude > lastOrder))
        || (crossedTwoDimensional && secondaryOrder != 0
            && (secondaryMagnitude < firstOrder || secondaryMagnitude > lastOrder)))
        return 0.0;

    vec4 localGeometry;
    vec4 localSecondaryGeometry;
    resolveGrooveGeometry(vUv, localGeometry, localSecondaryGeometry);
    vec2 expectedTangent = -incident.xy
        + float(primaryOrder) * wavelength / localGeometry.z
            * localGeometry.xy;
    if (crossedTwoDimensional)
        expectedTangent += float(secondaryOrder) * wavelength
            / localSecondaryGeometry.z
            * localSecondaryGeometry.xy;
    if (dot(expectedTangent, expectedTangent) >= 1.0)
        return 0.0;

    float expectedOutgoingCosine = sqrt(max(
        0.0, 1.0 - dot(expectedTangent, expectedTangent)));
    vec2 deviation = outgoing.xy - expectedTangent;
    float gaussianNormalization = 1.0 / (2.0 * pi * sigma * sigma);
    float lobe = gaussianNormalization
        * exp(-dot(deviation, deviation) / (2.0 * sigma * sigma));
    float roughnessArgument = 2.0 * pi * uDiffractionRoughness.x
        * (incident.z + expectedOutgoingCosine) / wavelength;
    float coherent = exp(-(roughnessArgument * roughnessArgument));
    float phase = 2.0 * pi * uDiffractionGeometry.w
        * (incident.z + expectedOutgoingCosine) / wavelength;
    vec2 terracePhase = vec2(cos(phase), sin(phase));
    float efficiency = coherent * profileEfficiency(
        profile, primaryOrder, duty, phase, terracePhase);
    if (crossedTwoDimensional)
        efficiency *= profileEfficiency(
            profile, secondaryOrder, duty, phase, terracePhase);
    efficiency *= expectedOutgoingCosine / incident.z;
    return efficiency * lobe;
}

vec3 evaluateDiffraction() {
    vec3 normal = normalize(vWorldNormal);
    vec3 dpdx = dFdx(vWorldPosition);
    vec3 dpdy = dFdy(vWorldPosition);
    vec2 duvdx = dFdx(vUv);
    vec2 duvdy = dFdy(vUv);
    float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    if (abs(determinant) <= 1.0e-8)
        return vec3(0.0);
    vec3 tangent = normalize((dpdx * duvdy.y - dpdy * duvdx.y) / determinant);
    tangent = normalize(tangent - normal * dot(normal, tangent));
    vec3 bitangent = normalize(cross(normal, tangent));
    if (dot(bitangent, (-dpdx * duvdy.x + dpdy * duvdx.x) / determinant) < 0.0)
        bitangent = -bitangent;

    vec3 toViewWorld = normalize(uCameraTranslation - vWorldPosition);
    vec3 incident = uDiffractionIncidentDirectionAndIntensity.xyz;
    vec3 outgoing = vec3(dot(toViewWorld, tangent), dot(toViewWorld, bitangent),
                         dot(toViewWorld, normal));
    if (incident.z <= 0.0 || outgoing.z <= 0.0)
        return vec3(0.0);

    float duty = uDiffractionMicrostructure.x;
    float substrateN = uDiffractionMicrostructure.y;
    float substrateK = uDiffractionMicrostructure.z;
    int profile = int(uDiffractionControl.y + 0.5);
    int coatingModel = int(uDiffractionControl.w + 0.5);
    int firstOrder = int(round(uDiffractionMicrostructure.w));
    int lastOrder = int(uDiffractionControl.x + 0.5);
    bool crossedTwoDimensional = int(uDiffractionSecondaryGeometry.w + 0.5) == 2;
    float pixelAngularFootprint = 0.5 * max(length(dFdx(outgoing.xy)),
                                            length(dFdy(outgoing.xy)));
    float sigma = max(2.0 * uDiffractionRoughness.y,
                      max(pixelAngularFootprint, 1.0e-4));
    vec3 xyz = vec3(0.0);
    float referenceWhiteY = 0.0;

    for (int wavelengthIndex = 0; wavelengthIndex < 8; ++wavelengthIndex) {
        vec4 spectral = uDiffractionSpectral[wavelengthIndex];
        vec4 spectralZ = uDiffractionSpectralZ[wavelengthIndex];
        float wavelength = spectral.x;
        float quadrature = spectralZ.y;
        float reflectance = spectralReflectance(
            wavelength, coatingModel, substrateN, substrateK);
        float wavelengthRadiance = diffractionOrderContribution(
            0, 0, crossedTwoDimensional, wavelength,
            incident, outgoing, profile, duty, sigma);
        if (crossedTwoDimensional) {
            for (int primaryOrder = -lastOrder; primaryOrder <= -firstOrder; ++primaryOrder) {
                for (int secondaryOrder = -lastOrder; secondaryOrder <= -firstOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma);
                for (int secondaryOrder = firstOrder; secondaryOrder <= lastOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma);
            }
            for (int primaryOrder = firstOrder; primaryOrder <= lastOrder; ++primaryOrder) {
                for (int secondaryOrder = -lastOrder; secondaryOrder <= -firstOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma);
                for (int secondaryOrder = firstOrder; secondaryOrder <= lastOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma);
            }
            for (int primaryOrder = -lastOrder; primaryOrder <= -firstOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, true, wavelength,
                    incident, outgoing, profile, duty, sigma);
            for (int primaryOrder = firstOrder; primaryOrder <= lastOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, true, wavelength,
                    incident, outgoing, profile, duty, sigma);
            for (int secondaryOrder = -lastOrder; secondaryOrder <= -firstOrder; ++secondaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    0, secondaryOrder, true, wavelength,
                    incident, outgoing, profile, duty, sigma);
            for (int secondaryOrder = firstOrder; secondaryOrder <= lastOrder; ++secondaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    0, secondaryOrder, true, wavelength,
                    incident, outgoing, profile, duty, sigma);
        } else {
            for (int primaryOrder = -lastOrder; primaryOrder <= -firstOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, false, wavelength,
                    incident, outgoing, profile, duty, sigma);
            for (int primaryOrder = firstOrder; primaryOrder <= lastOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, false, wavelength,
                    incident, outgoing, profile, duty, sigma);
        }
        float pathTransport = uDiffractionPathKindAndBounce.x == 3.0
            ? 1.0 / float(1 + uDiffractionPathKindAndBounce.y) : 1.0;
        float energy = uDiffractionIncidentDirectionAndIntensity.w
                     * spectral.y * pathTransport * quadrature * reflectance
                     * wavelengthRadiance * incident.z;
        xyz += energy * vec3(spectral.z, spectral.w, spectralZ.x);
        referenceWhiteY += quadrature * spectral.w;
    }
    xyz /= max(referenceWhiteY, 1.0e-8);
    return vec3(3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
                -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
                0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z);
}

void main() {
    outLinearDepth = vLinearDepth;
    if (uMaterialKind == materialDiffractionReflective) {
        float coverage = uDiffractionSpatialCounts.y == 0.0
            ? 1.0 : uintBitsToFloat(floatBitsToUint(uDiffractionPathKindAndBounce.w));
        float occupancy = uDiffractionSpatialCounts.y == 0.0 ? 1.0 : 0.0;
        for (int i = 0; i < 5; ++i) {
            if (float(i) >= uDiffractionSpatialCounts.y) break;
            vec4 rectangle = uDiffractionOccupancyRectangles[i];
            occupancy = max(occupancy, float(vUv.x >= rectangle.x && vUv.y >= rectangle.y
                && vUv.x <= rectangle.z && vUv.y <= rectangle.w));
        }
        if (uDiffractionSpatialCounts.x > 0.0) {
            vec4 lower = mix(uDiffractionFoilField[0], uDiffractionFoilField[1], clamp(vUv.x, 0.0, 1.0));
            vec4 upper = mix(uDiffractionFoilField[2], uDiffractionFoilField[3], clamp(vUv.x, 0.0, 1.0));
            occupancy *= mix(lower.w, upper.w, clamp(vUv.y, 0.0, 1.0));
        }
        occupancy *= coverage;
        uvec4 evaluationSchedule = floatBitsToUint(uDiffractionEvaluationSchedule);
        uint evaluationIndex = evaluationSchedule.w
            * (evaluationSchedule.x * evaluationSchedule.y)
            + uint(gl_FragCoord.y) * evaluationSchedule.x
            + uint(gl_FragCoord.x);
        vec3 diffraction = (uDiffractionSpatialCounts.x == 0.0
                || evaluationIndex < evaluationSchedule.z)
            ? max(evaluateDiffraction(), vec3(0.0)) : vLitBase;
        outColor = vec4(mix(vLitBase, diffraction,
            clamp(occupancy, 0.0, 1.0)), 1.0);
        return;
    }
    vec4 texel = texture(uBaseTexture, vUv);
    texel.rgb = srgbToLinear(texel.rgb);
    float alpha = vOpacity * texel.a;
    if (uAlphaModeCutoff.x == 1.0 && alpha < uAlphaModeCutoff.y)
        discard;
    vec4 metallicRoughnessTexel = texture(uMetallicRoughnessTexture, vUv);
    float roughness = vMetallicRoughness.y * (uTexturePresence.x > 0.5 ? metallicRoughnessTexel.g : 1.0);
    float metallic = vMetallicRoughness.x * (uTexturePresence.x > 0.5 ? metallicRoughnessTexel.b : 1.0);
    vec3 normal = normalize(vWorldNormal);
    if (uTexturePresence.y > 0.5) {
        vec3 tangent = normalize(vWorldTangent - normal * dot(normal, vWorldTangent));
        vec3 bitangent = normalize(cross(normal, tangent)) * vBitangentSign;
        vec3 tangentNormal = texture(uNormalTexture, vUv).xyz * 2.0 - 1.0;
        tangentNormal.xy *= uMaterialParams.z;
        normal = normalize(mat3(tangent, bitangent, normal) * tangentNormal);
    }
    vec3 lighting = uAmbient;
    for (int lightIndex = 0; lightIndex < 16; ++lightIndex) {
        if (lightIndex >= uLightCount) break;
        int kind = uLightKinds[lightIndex];
        if (kind == 2) { lighting += uLightColors[lightIndex].rgb * uLightColors[lightIndex].a; continue; }
        vec3 delta = uLightPositions[lightIndex].xyz - vWorldPosition;
        vec3 toLight = kind == 0 ? rotateQ(uLightRotations[lightIndex], vec3(0.0, 0.0, 1.0)) : normalize(delta);
        float attenuation = 1.0;
        if (kind == 1 || kind == 3) {
            float lightDistance = length(delta);
            attenuation = 1.0 / max(lightDistance * lightDistance, 1.0);
            float range = uLightPositions[lightIndex].w;
            if (range > 0.0) attenuation *= pow(clamp(1.0 - lightDistance / range, 0.0, 1.0), 2.0);
            if (kind == 3) {
                vec3 spotDirection = rotateQ(uLightRotations[lightIndex], vec3(0.0, 0.0, -1.0));
                float coneCosine = dot(spotDirection, normalize(vWorldPosition - uLightPositions[lightIndex].xyz));
                attenuation *= smoothstep(uLightCones[lightIndex].y, uLightCones[lightIndex].x, coneCosine);
            }
        }
        lighting += uLightColors[lightIndex].rgb * uLightColors[lightIndex].a
                  * max(dot(normal, normalize(toLight)), 0.0) * attenuation;
    }
    float diffuseWeight = (1.0 - metallic) * (1.0 - 0.5 * roughness);
    float specularWeight = mix(0.04, 1.0, metallic) * (1.0 - roughness);
    float occlusion = uTexturePresence.z > 0.5
        ? texture(uOcclusionTexture, vUv).r : 1.0;
    vec3 emissive = vEmissive * (uEmissiveTexturePresence > 0.5
        ? srgbToLinear(texture(uEmissiveTexture, vUv).rgb) : vec3(1.0));
    outColor = vec4(vBaseColor * texel.rgb * lighting
                    * (diffuseWeight + specularWeight) * occlusion + emissive, alpha);
    if (vInstanceIdentityColor.a > 0.5)
        outColor = vec4(vInstanceIdentityColor.rgb, 1.0);
}
)glsl";

unsigned compileFixtureShader (const arbitgl::GlFuncs& gl, unsigned type,
                               const char* source, std::string& error)
{
    const auto shader = gl.CreateShader (type);
    if (shader == 0)
    {
        error = "OpenGL fixture shader allocation failed";
        return 0;
    }
    gl.ShaderSource (shader, 1, &source, nullptr);
    gl.CompileShader (shader);
    int compiled = 0;
    gl.GetShaderiv (shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE)
        return shader;

    char log[1024] = {};
    int length = 0;
    gl.GetShaderInfoLog (shader, static_cast<int> (sizeof (log)), &length, log);
    error = std::string ("OpenGL fixture shader compilation failed: ")
        + std::string (log, static_cast<std::size_t> (std::max (length, 0)));
    gl.DeleteShader (shader);
    return 0;
}

struct OpenGlFixtureUniformLocations final
{
    int objectMatrix = -1;
    int tangentHandednessSign = -1;
    int useInstanceMatrix = -1;
    int cameraRotation = -1;
    int cameraTranslation = -1;
    int projection = -1;
    int baseColor = -1;
    int materialParams = -1;
    int timeMixEndColorAndTime = -1;
    int emissive = -1;
    int ambient = -1;
    int lightRotation = -1;
    int lightColorIntensity = -1;
    int lightPositionRange = -1;
    int lightKind = -1;
    int lightCount = -1;
    std::array<int, 16> lightRotations {};
    std::array<int, 16> lightColors {};
    std::array<int, 16> lightPositions {};
    std::array<int, 16> lightCones {};
    std::array<int, 16> lightKinds {};
    int baseTexture = -1;
    int metallicRoughnessTexture = -1;
    int normalTexture = -1;
    int occlusionTexture = -1;
    int emissiveTexture = -1;
    int texturePresence = -1;
    int emissiveTexturePresence = -1;
    int alphaModeCutoff = -1;
    int materialKind = -1;
    int diffractionGeometry = -1;
    int diffractionSecondaryGeometry = -1;
    int diffractionMicrostructure = -1;
    int diffractionControl = -1;
    int diffractionCoating = -1;
    int diffractionRoughness = -1;
    int diffractionGrooveField = -1;
    int diffractionGrooveVariation = -1;
    int diffractionSpectral = -1;
    int diffractionSpectralZ = -1;
    int diffractionIncidentDirectionAndIntensity = -1;
    int diffractionPathKindAndBounce = -1;
    int diffractionFoilField = -1;
    int diffractionOccupancyRectangles = -1;
    int diffractionSpatialCounts = -1;
    int diffractionEvaluationSchedule = -1;
    int noteInstanceTransforms = -1;
    int noteInstanceIndex = -1;

    bool complete() const noexcept
    {
        return objectMatrix >= 0 && tangentHandednessSign >= 0
            && useInstanceMatrix >= 0
            && cameraRotation >= 0 && cameraTranslation >= 0 && projection >= 0
            && baseColor >= 0 && materialParams >= 0
            && timeMixEndColorAndTime >= 0 && emissive >= 0 && ambient >= 0
            && baseTexture >= 0 && lightCount >= 0
            && metallicRoughnessTexture >= 0 && normalTexture >= 0
            && occlusionTexture >= 0 && emissiveTexture >= 0
            && texturePresence >= 0 && emissiveTexturePresence >= 0
            && alphaModeCutoff >= 0
            && materialKind >= 0 && diffractionGeometry >= 0
            && diffractionSecondaryGeometry >= 0
            && diffractionMicrostructure >= 0 && diffractionControl >= 0
            && diffractionCoating >= 0 && diffractionRoughness >= 0
            && diffractionGrooveField >= 0 && diffractionGrooveVariation >= 0
            && diffractionSpectral >= 0 && diffractionSpectralZ >= 0
            && diffractionIncidentDirectionAndIntensity >= 0
            && diffractionPathKindAndBounce >= 0
            && diffractionFoilField >= 0 && diffractionOccupancyRectangles >= 0
            && diffractionSpatialCounts >= 0
            && diffractionEvaluationSchedule >= 0 && noteInstanceTransforms >= 0
            && noteInstanceIndex >= 0;
    }
};

class OpenGlFixtureSceneResources final : public NativeFixtureSceneResources
{
public:
    ~OpenGlFixtureSceneResources() override
    {
        if (! hasResources() || ownerContext == nullptr)
            return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        auto* previous = glfwGetCurrentContext();
        if (previous != ownerContext)
            glfwMakeContextCurrent (ownerContext);
        if (glfwGetCurrentContext() == ownerContext)
            destroyUnlocked();
        if (previous != ownerContext)
            glfwMakeContextCurrent (previous);
    }

    const std::string& backend() const noexcept override { return backend_; }

    bool hasResources() const noexcept
    {
        return vertexArray != 0 || vertexBuffer != 0 || indexBuffer != 0
            || instanceBuffer != 0
            || ! textures.empty() || program != 0;
    }

    void destroyUnlocked() noexcept
    {
        if (program != 0) gl.DeleteProgram (program);
        if (! textures.empty())
            glDeleteTextures (static_cast<int> (textures.size()), textures.data());
        if (indexBuffer != 0) gl.DeleteBuffers (1, &indexBuffer);
        if (instanceBuffer != 0) gl.DeleteBuffers (1, &instanceBuffer);
        if (vertexBuffer != 0) gl.DeleteBuffers (1, &vertexBuffer);
        if (vertexArray != 0) gl.DeleteVertexArrays (1, &vertexArray);
        vertexArray = 0;
        vertexBuffer = 0;
        indexBuffer = 0;
        instanceBuffer = 0;
        textures.clear();
        program = 0;
    }

    std::string backend_ = "opengl";
    GLFWwindow* ownerContext = nullptr;
    arbitgl::GlFuncs gl {};
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> snapshot;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> materialProgram;
    // Identifies one prepared static-resource lifetime. Frames rendered from the
    // same preparation retain this value; preparing again allocates a new one.
    std::uint64_t rendererGeneration = 0;
    unsigned vertexArray = 0;
    unsigned vertexBuffer = 0;
    unsigned indexBuffer = 0;
    unsigned instanceBuffer = 0;
    bool instancedSharedGeometry = false;
    bool diagnosticInstanceIdentityColors = false;
    std::vector<unsigned> textures;
    unsigned program = 0;
    OpenGlFixtureUniformLocations uniforms {};
};

class OpenGlFixtureSceneFrame final : public NativeFixtureSceneFrame
{
public:
    ~OpenGlFixtureSceneFrame() override
    {
        if ((texture_ == 0 && depthTexture_ == 0) || ownerContext == nullptr)
            return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        auto* previous = glfwGetCurrentContext();
        if (previous != ownerContext)
            glfwMakeContextCurrent (ownerContext);
        if (glfwGetCurrentContext() == ownerContext)
        {
            if (depthTexture_ != 0) glDeleteTextures (1, &depthTexture_);
            if (texture_ != 0) glDeleteTextures (1, &texture_);
        }
        if (previous != ownerContext)
            glfwMakeContextCurrent (previous);
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return texture_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return texture_; }
    std::uintptr_t depthImageHandle() const noexcept override { return depthTexture_; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return depthTexture_; }
    std::uintptr_t nativeResourceCacheIdentity() const noexcept override
    {
        return reinterpret_cast<std::uintptr_t> (staticResources.get());
    }
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::Rgba8Unorm, texture_, texture_, width_, height_, 1,
                 true, reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_ };
    }
    NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::R32Float, depthTexture_, depthTexture_, width_, height_, 1,
                 true, reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_ };
    }

    std::string backend_ = "opengl";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    GLFWwindow* ownerContext = nullptr;
    std::shared_ptr<const OpenGlFixtureSceneResources> staticResources;
    std::uint64_t rendererGeneration_ = 0;
    unsigned texture_ = 0;
    unsigned depthTexture_ = 0;
};

struct OpenGlInstanceGpuRecord final
{
    std::array<float, 16> matrix {};
    std::array<float, 4> identityColor {};
};

struct OpenGlFixtureUniforms final
{
    std::array<float, 16> objectMatrix {};
    std::array<float, 4> cameraRotation {};
    std::array<float, 4> cameraTranslation {};
    std::array<float, 4> projection {};
    std::array<float, 4> baseColor {};
    std::array<float, 4> materialParams {};
    std::array<float, 4> timeMixEndColorAndTime {};
    std::array<float, 4> emissive {};
    std::array<float, 4> ambient {};
    std::array<float, 4> lightRotation {};
    std::array<float, 4> lightColorIntensity {};
    std::array<float, 4> lightPositionRange {};
};
static_assert (sizeof (OpenGlFixtureUniforms) == 240,
               "OpenGL fixture uniform receipt changed");

void storeFixtureVec3 (std::array<float, 4>& destination,
                       const HarmonicMIDI::grid::SceneVec3& source,
                       float fourth = 0.0f) noexcept
{
    destination = { source.x, source.y, source.z, fourth };
}

void storeFixtureQuaternion (
    std::array<float, 4>& destination,
    const HarmonicMIDI::grid::SceneQuaternion& source) noexcept
{
    destination = { source.x, source.y, source.z, source.w };
}

std::array<float, 16> fixtureTransformMatrix (
    const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept
{
    const auto x = transform.rotation.x, y = transform.rotation.y;
    const auto z = transform.rotation.z, w = transform.rotation.w;
    const auto sx = transform.scale.x, sy = transform.scale.y, sz = transform.scale.z;
    return {
        (1.0f - 2.0f * (y * y + z * z)) * sx,
        (2.0f * (x * y + w * z)) * sx,
        (2.0f * (x * z - w * y)) * sx, 0.0f,
        (2.0f * (x * y - w * z)) * sy,
        (1.0f - 2.0f * (x * x + z * z)) * sy,
        (2.0f * (y * z + w * x)) * sy, 0.0f,
        (2.0f * (x * z + w * y)) * sz,
        (2.0f * (y * z - w * x)) * sz,
        (1.0f - 2.0f * (x * x + y * y)) * sz, 0.0f,
        transform.translation.x, transform.translation.y, transform.translation.z, 1.0f
    };
}

std::array<float, 16> multiplyFixtureMatrices (
    const std::array<float, 16>& left,
    const std::array<float, 16>& right) noexcept
{
    std::array<float, 16> result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row] += left[inner * 4 + row]
                    * right[column * 4 + inner];
    return result;
}

float fixtureMatrixDeterminant3x3 (const std::array<float, 16>& matrix) noexcept
{
    return matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6])
         - matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2])
         + matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2]);
}

std::array<float, 16> fixtureWorldMatrix (
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const HarmonicMIDI::grid::SceneObjectRecord& object) noexcept
{
    auto result = fixtureTransformMatrix (object.transform);
    auto parent = object.parent;
    for (std::size_t depth = 0; parent.isValid() && depth < scene.objectCount; ++depth)
    {
        const auto* parentObject = HarmonicMIDI::grid::visual3d_detail::findById (
            scene.objects, scene.objectCount, parent);
        if (parentObject == nullptr)
            break;
        result = multiplyFixtureMatrices (
            fixtureTransformMatrix (parentObject->transform), result);
        parent = parentObject->parent;
    }
    return result;
}

class OpenGlFixtureSceneBackend final : public NativeFixtureSceneBackend
{
public:
    BackendInfo info() const override
    {
        BackendInfo result;
        result.backend = "opengl";
        if (glfwGetCurrentContext() == nullptr)
        {
            result.error = kFixtureUnavailable;
            return result;
        }

        arbitgl::GlFuncs gl;
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing))
        {
            result.error = "native OpenGL fixture execution is missing required functions: "
                         + missing;
            return result;
        }

        const auto* renderer = glGetString (GL_RENDERER);
        result.available = true;
        result.device = renderer != nullptr
            ? reinterpret_cast<const char*> (renderer) : "OpenGL 3.3 context";
        return result;
    }

    GeometryCoreExecutionCapabilities geometryCoreCapabilities() const override
    {
        GeometryCoreExecutionCapabilities capabilities;
        const auto available = info();
        if (! available.available) return capabilities;
        capabilities.immutableSourceBuffers = true;
        capabilities.stableElementIds = true;
        capabilities.gpuInstancingWithoutMeshExpansion = true;
        capabilities.supportedCarriers =
            (1u << static_cast<unsigned> (
                videowire::geometry::CarrierKind::geometry3D))
            | (1u << static_cast<unsigned> (
                videowire::geometry::CarrierKind::instances3D));
        capabilities.maxVertices = HarmonicMIDI::grid::Visual3DScene::kMaxVertices;
        capabilities.maxIndices = HarmonicMIDI::grid::Visual3DScene::kMaxIndices;
        capabilities.maxPoints = 1;
        capabilities.maxCurvePoints = 1;
        capabilities.maxSplines = 1;
        capabilities.maxInstances = HarmonicMIDI::grid::Visual3DScene::kMaxObjects;
        capabilities.maxFieldElements = 1;
        capabilities.maxAttributes = videowire::geometry::kMaximumAttributes;
        capabilities.maxOperations = videowire::geometry::kMaximumOperations;
        capabilities.maxDispatches = videowire::geometry::kMaximumDispatches;
        capabilities.maxBufferBytes = 64u * 1024u * 1024u;
        return capabilities;
    }

    NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram) override
    {
        return prepareGeometryInstances(scene, requestedMaterialProgram, {}, false);
    }

    NativeFixtureScenePreparation prepareGeometryInstances (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>& geometryAdmission,
        bool diagnosticInstanceIdentityColors) override
    {
        using namespace HarmonicMIDI::grid;

        NativeFixtureScenePreparation result;
        const auto materialProgram = requestedMaterialProgram == nullptr
            ? std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> {}
            : std::make_shared<const NativeFixtureSurfaceMaterialProgram> (
                *requestedMaterialProgram);
        std::lock_guard<std::mutex> lock (fixtureMutex());
        if (scene == nullptr || ! validateVisual3DScene (*scene).valid()
            || scene->objectCount == 0 || scene->materialCount == 0
            || scene->lightCount > Visual3DScene::kMaxLights
            || scene->cameraCount == 0 || scene->cameraCount > Visual3DScene::kMaxCameras
            || visual3d_detail::findById (scene->cameras, scene->cameraCount,
                                          scene->activeCamera) == nullptr)
        {
            result.error = "OpenGL fixture preparation requires a bounded renderable scene";
            return result;
        }
        const auto& sceneMaterial = scene->materials[0];
        const bool materialKindValid = materialProgram == nullptr
            || (materialProgram->kind == NativeFixtureMaterialKind::SurfacePbr
                && materialProgram->parameters.identifiers[0] == sceneMaterial.id.value
                && validNativeFixtureSurfaceParameters (materialProgram->parameters))
            || validNativeFixtureDiffractionProgram (
                *materialProgram, NativeFixtureMaterialBackend::OpenGl,
                scene->objects[0].id);
        if (materialProgram != nullptr
            && (materialProgram->layoutVersion
                    != NativeFixtureSurfaceMaterialProgram::kLayoutVersion
                || materialProgram->backend != NativeFixtureMaterialBackend::OpenGl
                || materialProgram->programIdentity.empty()
                || materialProgram->bindingDigest.empty()
                || materialProgram->object != scene->objects[0].id
                || ! materialKindValid))
        {
            result.error = "OpenGL fixture preparation requires an exact bounded material binding";
            return result;
        }

        const auto available = info();
        if (! available.available)
        {
            result.error = available.error;
            return result;
        }

        arbitgl::GlFuncs gl;
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing))
        {
            result.error = "OpenGL fixture loader failed: " + missing;
            return result;
        }
        OpenGlFixtureState previous (gl);
        auto resources = std::make_shared<OpenGlFixtureSceneResources>();
        resources->ownerContext = glfwGetCurrentContext();
        resources->rendererGeneration = nextFixtureRendererGeneration();
        resources->gl = gl;
        resources->snapshot = scene;
        resources->materialProgram = materialProgram;
        const auto& firstObject = scene->objects[0];
        const auto admittedInstanceSetMatchesScene = [&]
        {
            if (geometryAdmission == nullptr
                || geometryAdmission->value().descriptor().carrier
                    != videowire::geometry::CarrierKind::instances3D)
                return false;
            const auto& admittedInstances = std::get<videowire::geometry::InstancesData>(
                geometryAdmission->value().descriptor().data).instances;
            if (admittedInstances.size() != scene->objectCount)
                return false;
            for (std::size_t index = 0; index < scene->objectCount; ++index)
                if (admittedInstances[index].stableId != scene->objects[index].id.value)
                    return false;
            return true;
        }();
        resources->instancedSharedGeometry = admittedInstanceSetMatchesScene
            && scene->objectCount > 1
            && materialProgram == nullptr
            && !firstObject.parent.isValid()
            && firstObject.material == sceneMaterial.id
            && sceneMaterial.alphaMode != SceneAlphaMode::Blend
            && std::all_of(scene->objects.begin() + 1,
                           scene->objects.begin() + scene->objectCount,
                           [&](const auto& object)
            {
                return !object.parent.isValid()
                    && object.material == firstObject.material
                    && object.firstVertex == firstObject.firstVertex
                    && object.vertexCount == firstObject.vertexCount
                    && object.firstIndex == firstObject.firstIndex
                    && object.indexCount == firstObject.indexCount;
            });
            resources->diagnosticInstanceIdentityColors = diagnosticInstanceIdentityColors;
        unsigned vertexShader = 0;
        unsigned fragmentShader = 0;
        auto cleanupShaders = [&]
        {
            if (fragmentShader != 0) gl.DeleteShader (fragmentShader);
            if (vertexShader != 0) gl.DeleteShader (vertexShader);
            fragmentShader = 0;
            vertexShader = 0;
        };
        auto fail = [&] (std::string error)
        {
            cleanupShaders();
            resources->destroyUnlocked();
            previous.restore();
            result.error = std::move (error);
            return result;
        };

        static_assert (sizeof (SceneVertex) == 16 * sizeof (float),
                       "fixture vertex upload layout changed");
        static_assert (sizeof (SceneTexelRgba8) == 4,
                       "fixture texel upload layout changed");
        const auto vertexBytes = scene->vertexCount * sizeof (SceneVertex);
        const auto indexBytes = scene->indexCount * sizeof (std::uint32_t);

        gl.GenVertexArrays (1, &resources->vertexArray);
        gl.BindVertexArray (resources->vertexArray);
        gl.GenBuffers (1, &resources->vertexBuffer);
        gl.BindBuffer (GL_ARRAY_BUFFER, resources->vertexBuffer);
        gl.BufferData (GL_ARRAY_BUFFER, static_cast<std::ptrdiff_t> (vertexBytes),
                       scene->vertices.data(), GL_STATIC_DRAW);
        gl.GenBuffers (1, &resources->indexBuffer);
        gl.BindBuffer (GL_ELEMENT_ARRAY_BUFFER, resources->indexBuffer);
        std::vector<std::uint32_t> globalIndices (scene->indices.begin(),
                                                  scene->indices.begin() + scene->indexCount);
        const auto indexOwnerCount = resources->instancedSharedGeometry
            ? std::size_t { 1 } : scene->objectCount;
        for (std::size_t objectIndex = 0; objectIndex < indexOwnerCount; ++objectIndex)
        {
            const auto& drawObject = scene->objects[objectIndex];
            for (std::size_t index = 0; index < drawObject.indexCount; ++index)
                globalIndices[drawObject.firstIndex + index] += drawObject.firstVertex;
        }
        gl.BufferData (GL_ELEMENT_ARRAY_BUFFER, static_cast<std::ptrdiff_t> (indexBytes),
                       globalIndices.data(), GL_STATIC_DRAW);
        gl.VertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, sizeof (SceneVertex),
                                reinterpret_cast<const void*> (offsetof (SceneVertex, position)));
        gl.VertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE, sizeof (SceneVertex),
                                reinterpret_cast<const void*> (offsetof (SceneVertex, normal)));
        gl.VertexAttribPointer (2, 2, GL_FLOAT, GL_FALSE, sizeof (SceneVertex),
                                reinterpret_cast<const void*> (offsetof (SceneVertex, uv)));
        gl.VertexAttribPointer (3, 4, GL_FLOAT, GL_FALSE, sizeof (SceneVertex),
                                reinterpret_cast<const void*> (offsetof (SceneVertex, color)));
        gl.VertexAttribPointer (4, 4, GL_FLOAT, GL_FALSE, sizeof (SceneVertex),
                                reinterpret_cast<const void*> (offsetof (SceneVertex, tangent)));
        gl.EnableVertexAttribArray (0);
        gl.EnableVertexAttribArray (1);
        gl.EnableVertexAttribArray (2);
        gl.EnableVertexAttribArray (3);
        gl.EnableVertexAttribArray (4);
        if (resources->instancedSharedGeometry)
        {
            gl.GenBuffers(1, &resources->instanceBuffer);
            gl.BindBuffer(GL_ARRAY_BUFFER, resources->instanceBuffer);
            gl.BufferData(GL_ARRAY_BUFFER,
                static_cast<std::ptrdiff_t>(scene->objectCount
                    * sizeof(OpenGlInstanceGpuRecord)), nullptr, GL_DYNAMIC_DRAW);
            for (unsigned column = 0; column < 4; ++column)
            {
                const auto attribute = 5u + column;
                gl.VertexAttribPointer(attribute, 4, GL_FLOAT, GL_FALSE,
                    sizeof(OpenGlInstanceGpuRecord),
                    reinterpret_cast<const void*>(column * 4u * sizeof(float)));
                gl.EnableVertexAttribArray(attribute);
                gl.VertexAttribDivisor(attribute, 1);
            }
            gl.VertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE,
                sizeof(OpenGlInstanceGpuRecord), reinterpret_cast<const void*>(
                    offsetof(OpenGlInstanceGpuRecord, identityColor)));
            gl.EnableVertexAttribArray(9);
            gl.VertexAttribDivisor(9, 1);
        }

        const SceneTexelRgba8 whiteTexel { 255, 255, 255, 255 };
        // Upload each image once. The shader applies the sRGB transfer only at
        // base-color and emissive sampling sites, while data roles read the same
        // immutable RGBA8 storage without a color transform.
        resources->textures.resize (1 + scene->textureCount);
        glGenTextures (static_cast<int> (resources->textures.size()), resources->textures.data());
        gl.ActiveTexture (GL_TEXTURE0);
        gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei (GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei (GL_UNPACK_SKIP_PIXELS, 0);
        glPixelStorei (GL_UNPACK_SKIP_ROWS, 0);
        std::size_t textureBytes = sizeof (whiteTexel);
        for (std::size_t slot = 0; slot < resources->textures.size(); ++slot)
        {
            const auto* data = static_cast<const void*> (&whiteTexel);
            std::uint32_t textureWidth = 1, textureHeight = 1;
            if (slot > 0)
            {
                const auto& texture = scene->textures[slot - 1];
                data = scene->textureTexels.data() + texture.firstTexel;
                textureWidth = texture.width;
                textureHeight = texture.height;
                textureBytes += static_cast<std::size_t> (textureWidth) * textureHeight
                              * sizeof (SceneTexelRgba8);
            }
            glBindTexture (GL_TEXTURE_2D, resources->textures[slot]);
            const auto* source = slot > 0 ? &scene->textures[slot - 1] : nullptr;
            const auto minFilter = source != nullptr
                ? static_cast<int> (source->minFilter) : GL_NEAREST;
            const auto magFilter = source != nullptr
                ? static_cast<int> (source->magFilter) : GL_NEAREST;
            const auto wrapS = source != nullptr
                ? static_cast<int> (source->wrapS) : GL_REPEAT;
            const auto wrapT = source != nullptr
                ? static_cast<int> (source->wrapT) : GL_REPEAT;
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
            glTexImage2D (GL_TEXTURE_2D, 0,
                          GL_RGBA8,
                          static_cast<int> (textureWidth), static_cast<int> (textureHeight),
                          0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            if (minFilter == GL_NEAREST_MIPMAP_NEAREST
                || minFilter == GL_LINEAR_MIPMAP_NEAREST
                || minFilter == GL_NEAREST_MIPMAP_LINEAR
                || minFilter == GL_LINEAR_MIPMAP_LINEAR)
            {
                const auto generateMipmap = reinterpret_cast<PFNGLGENERATEMIPMAPPROC> (
                    glfwGetProcAddress ("glGenerateMipmap"));
                generateMipmap (GL_TEXTURE_2D);
            }
        }

        vertexShader = compileFixtureShader (
            gl, GL_VERTEX_SHADER, kFixtureVertexShader, result.error);
        if (vertexShader == 0)
            return fail (result.error);
        fragmentShader = compileFixtureShader (
            gl, GL_FRAGMENT_SHADER, kFixtureFragmentShader, result.error);
        if (fragmentShader == 0)
            return fail (result.error);
        resources->program = gl.CreateProgram();
        gl.AttachShader (resources->program, vertexShader);
        gl.AttachShader (resources->program, fragmentShader);
        gl.LinkProgram (resources->program);
        int linked = 0;
        gl.GetProgramiv (resources->program, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE)
        {
            char log[1024] = {};
            int length = 0;
            gl.GetProgramInfoLog (resources->program,
                                  static_cast<int> (sizeof (log)), &length, log);
            return fail (std::string ("OpenGL fixture program link failed: ")
                + std::string (log, static_cast<std::size_t> (std::max (length, 0))));
        }
        cleanupShaders();

        const auto location = [&] (const char* name)
        {
            return gl.GetUniformLocation (resources->program, name);
        };
        auto& uniforms = resources->uniforms;
        uniforms.objectMatrix = location ("uObjectMatrix");
        uniforms.tangentHandednessSign = location ("uTangentHandednessSign");
        uniforms.useInstanceMatrix = location("uUseInstanceMatrix");
        uniforms.cameraRotation = location ("uCameraRotation");
        uniforms.cameraTranslation = location ("uCameraTranslation");
        uniforms.projection = location ("uProjection");
        uniforms.baseColor = location ("uBaseColor");
        uniforms.materialParams = location ("uMaterialParams");
        uniforms.timeMixEndColorAndTime = location ("uTimeMixEndColorAndTime");
        uniforms.emissive = location ("uEmissive");
        uniforms.ambient = location ("uAmbient");
        uniforms.lightRotation = location ("uLightRotation");
        uniforms.lightColorIntensity = location ("uLightColorIntensity");
        uniforms.lightPositionRange = location ("uLightPositionRange");
        uniforms.lightKind = location ("uLightKind");
        uniforms.lightCount = location ("uLightCount");
        for (std::size_t lightIndex = 0; lightIndex < 16; ++lightIndex)
        {
            const auto suffix = "[" + std::to_string (lightIndex) + "]";
            uniforms.lightRotations[lightIndex] = location (("uLightRotations" + suffix).c_str());
            uniforms.lightColors[lightIndex] = location (("uLightColors" + suffix).c_str());
            uniforms.lightPositions[lightIndex] = location (("uLightPositions" + suffix).c_str());
            uniforms.lightCones[lightIndex] = location (("uLightCones" + suffix).c_str());
            uniforms.lightKinds[lightIndex] = location (("uLightKinds" + suffix).c_str());
        }
        uniforms.baseTexture = location ("uBaseTexture");
        uniforms.metallicRoughnessTexture = location ("uMetallicRoughnessTexture");
        uniforms.normalTexture = location ("uNormalTexture");
        uniforms.occlusionTexture = location ("uOcclusionTexture");
        uniforms.emissiveTexture = location ("uEmissiveTexture");
        uniforms.texturePresence = location ("uTexturePresence");
        uniforms.emissiveTexturePresence = location ("uEmissiveTexturePresence");
        uniforms.alphaModeCutoff = location ("uAlphaModeCutoff");
        uniforms.materialKind = location ("uMaterialKind");
        uniforms.diffractionGeometry = location ("uDiffractionGeometry");
        uniforms.diffractionSecondaryGeometry
            = location ("uDiffractionSecondaryGeometry");
        uniforms.diffractionMicrostructure = location ("uDiffractionMicrostructure");
        uniforms.diffractionControl = location ("uDiffractionControl");
        uniforms.diffractionCoating = location ("uDiffractionCoating");
        uniforms.diffractionRoughness = location ("uDiffractionRoughness");
        uniforms.diffractionGrooveField = location ("uDiffractionGrooveField");
        uniforms.diffractionGrooveVariation = location ("uDiffractionGrooveVariation");
        uniforms.diffractionSpectral = location ("uDiffractionSpectral[0]");
        uniforms.diffractionSpectralZ = location ("uDiffractionSpectralZ[0]");
        uniforms.diffractionIncidentDirectionAndIntensity
            = location ("uDiffractionIncidentDirectionAndIntensity");
        uniforms.diffractionPathKindAndBounce
            = location ("uDiffractionPathKindAndBounce");
        uniforms.diffractionFoilField = location ("uDiffractionFoilField[0]");
        uniforms.diffractionOccupancyRectangles
            = location ("uDiffractionOccupancyRectangles[0]");
        uniforms.diffractionSpatialCounts = location ("uDiffractionSpatialCounts");
        uniforms.diffractionEvaluationSchedule
            = location ("uDiffractionEvaluationSchedule");
        uniforms.noteInstanceTransforms = location ("uNoteInstanceTransforms[0]");
        uniforms.noteInstanceIndex = location ("uNoteInstanceIndex");
        if (resources->vertexArray == 0 || resources->vertexBuffer == 0
            || resources->indexBuffer == 0 || resources->textures.empty()
            || (resources->instancedSharedGeometry && resources->instanceBuffer == 0)
            || resources->program == 0 || ! uniforms.complete())
        {
            return fail ("OpenGL fixture static GPU resource creation failed");
        }

        previous.restore();
        result.prepared = true;
        result.resources = std::move (resources);
        result.stats.staticUploadCount = 1;
        result.stats.vertexBytes = vertexBytes;
        result.stats.indexBytes = indexBytes;
        result.stats.textureBytes = textureBytes;
        result.stats.materialBytes = materialProgram != nullptr
            ? sizeof (NativeFixtureSurfaceMaterialProgram) : sizeof (SceneMaterialRecord);
        result.stats.materialProgramUploadCount = materialProgram != nullptr ? 1u : 0u;
        return result;
    }

    NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSceneResources>& nativeResources,
        std::uint32_t width,
        std::uint32_t height,
        NativeFixtureSceneRuntimeInputs runtimeInputs) override
    {
        using namespace HarmonicMIDI::grid;

        NativeFixtureSceneSubmission result;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        const auto available = info();
        if (! available.available)
        {
            result.error = available.error;
            return result;
        }
        const auto resources = std::dynamic_pointer_cast<const OpenGlFixtureSceneResources> (
            nativeResources);
        if (scene == nullptr || resources == nullptr || resources->snapshot != scene
            || resources->ownerContext != glfwGetCurrentContext())
        {
            result.error = "OpenGL fixture render requires exact prepared scene resources";
            return result;
        }
        int maximumTextureSize = 0;
        glGetIntegerv (GL_MAX_TEXTURE_SIZE, &maximumTextureSize);
        const auto maximumExtent = static_cast<std::uint32_t> (std::max (
            0, std::min (maximumTextureSize, static_cast<int> (kMaximumNativeFixtureExtent))));
        if (! nativeFixtureDimensionsWithinBounds (width, height)
            || width > maximumExtent || height > maximumExtent)
        {
            result.error = "OpenGL fixture render dimensions exceed backend limits";
            return result;
        }
        if (resources->materialProgram != nullptr
            && ! nativeFixtureDiffractionWorkWithinBudget (
                *resources->materialProgram, width, height))
        {
            result.error = "OpenGL fixture diffraction workload exceeds backend limits";
            return result;
        }
        if (!validFixtureRuntimeInputs(runtimeInputs))
        {
            result.error = "OpenGL fixture scene modulation is non-finite or out of bounds";
            return result;
        }
        if (resources->materialProgram != nullptr
            && resources->materialProgram->baseColorSource
                == NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix
            && (! std::isfinite (runtimeInputs.timeSeconds)
                || std::abs (runtimeInputs.timeSeconds)
                    > surfacematerial::kMaximumEvaluationMagnitude))
        {
            result.error = "OpenGL fixture material time input is non-finite or out of bounds";
            return result;
        }

        const auto& gl = resources->gl;
        const auto noteInstances = prepareNativeNoteInstances(runtimeInputs);
        OpenGlFixtureState previous (gl);
        auto frame = std::make_shared<OpenGlFixtureSceneFrame>();
        frame->width_ = width;
        frame->height_ = height;
        frame->ownerContext = glfwGetCurrentContext();
        frame->rendererGeneration_ = resources->rendererGeneration;
        frame->staticResources = resources;
        unsigned framebuffer = 0;
        unsigned depthAttachment = 0;
        auto cleanup = [&]
        {
            if (framebuffer != 0) gl.DeleteFramebuffers (1, &framebuffer);
            if (depthAttachment != 0) glDeleteTextures (1, &depthAttachment);
            framebuffer = 0;
            depthAttachment = 0;
            previous.restore();
        };
        auto fail = [&] (std::string error)
        {
            cleanup();
            if (frame->texture_ != 0)
            {
                glDeleteTextures (1, &frame->texture_);
                frame->texture_ = 0;
            }
            result.error = std::move (error);
            return result;
        };

        gl.ActiveTexture (GL_TEXTURE0);
        gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei (GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei (GL_UNPACK_SKIP_PIXELS, 0);
        glPixelStorei (GL_UNPACK_SKIP_ROWS, 0);
        glGenTextures (1, &frame->texture_);
        glBindTexture (GL_TEXTURE_2D, frame->texture_);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<int> (width),
                      static_cast<int> (height), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glGenTextures (1, &frame->depthTexture_);
        glBindTexture (GL_TEXTURE_2D, frame->depthTexture_);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_R32F,
                      static_cast<int> (width), static_cast<int> (height), 0,
                      GL_RED, GL_FLOAT, nullptr);
        glGenTextures (1, &depthAttachment);
        glBindTexture (GL_TEXTURE_2D, depthAttachment);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
                      static_cast<int> (width), static_cast<int> (height), 0,
                      GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        gl.GenFramebuffers (1, &framebuffer);
        gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
        gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                 GL_TEXTURE_2D, frame->texture_, 0);
        gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                                 GL_TEXTURE_2D, frame->depthTexture_, 0);
        gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                 GL_TEXTURE_2D, depthAttachment, 0);
        const unsigned drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        gl.DrawBuffers (2, drawBuffers);
        if (frame->texture_ == 0 || frame->depthTexture_ == 0
            || depthAttachment == 0 || framebuffer == 0
            || gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            return fail ("OpenGL fixture GPU resource creation failed");
        }

        const auto& object = scene->objects[0];
        const auto& material = scene->materials[0];
        const auto* selectedCamera = visual3d_detail::findById (
            scene->cameras, scene->cameraCount, scene->activeCamera);
        if (! runtimeInputs.cameraOverride.has_value() && selectedCamera == nullptr)
        {
            result.error = "OpenGL fixture render cannot resolve the active camera";
            return result;
        }
        const auto& camera = runtimeInputs.cameraOverride.has_value()
            ? *runtimeInputs.cameraOverride : *selectedCamera;
        OpenGlFixtureUniforms values;
        values.objectMatrix = fixtureWorldMatrix (*scene, object);
        storeFixtureQuaternion (values.cameraRotation, camera.transform.rotation);
        storeFixtureVec3 (values.cameraTranslation, camera.transform.translation);
        for (std::size_t axis = 0; axis < 3; ++axis)
            values.cameraTranslation[axis] += runtimeInputs.cameraTranslationOffset[axis];
        values.projection = {
            std::tan (camera.verticalFovRadians * 0.5f),
            static_cast<float> (width) / static_cast<float> (height),
            camera.nearPlane, camera.farPlane
        };
        if (resources->materialProgram != nullptr)
        {
            const auto& program = *resources->materialProgram;
            const auto& pbr = program.parameters;
            values.baseColor = {
                pbr.baseColorMetallic[0], pbr.baseColorMetallic[1],
                pbr.baseColorMetallic[2], pbr.normalOpacity[3]
            };
            values.materialParams = {
                pbr.baseColorMetallic[3], pbr.emissionRoughness[3], 1.0f, 0.0f
            };
            if (program.baseColorSource
                == NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix)
            {
                values.timeMixEndColorAndTime = {
                    program.timeMixEndColor[0], program.timeMixEndColor[1],
                    program.timeMixEndColor[2], runtimeInputs.timeSeconds
                };
                values.materialParams[3] = 1.0f;
            }
            values.emissive = {
                pbr.emissionRoughness[0], pbr.emissionRoughness[1],
                pbr.emissionRoughness[2], 0.0f
            };
        }
        else
        {
            storeFixtureVec3 (values.baseColor, material.baseColor, material.opacity);
            storeFixtureVec3 (values.emissive, material.emissive);
        }
        for (std::size_t channel = 0; channel < 3; ++channel)
            values.emissive[channel] *= runtimeInputs.emissionGain;
        storeFixtureVec3 (values.ambient, scene->ambientColor);
        values.lightRotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        if (scene->lightCount == 1)
        {
            const auto& light = scene->lights[0];
            if (light.kind == SceneLightKind::Environment)
            {
                values.ambient[0] += light.color.x * light.intensity;
                values.ambient[1] += light.color.y * light.intensity;
                values.ambient[2] += light.color.z * light.intensity;
            }
            else
            {
                storeFixtureQuaternion (values.lightRotation, light.transform.rotation);
                storeFixtureVec3 (values.lightColorIntensity, light.color, light.intensity);
                if (light.kind == SceneLightKind::Point)
                    storeFixtureVec3 (values.lightPositionRange,
                                      light.transform.translation, light.range);
            }
        }

        glViewport (0, 0, static_cast<int> (width), static_cast<int> (height));
        glDisable (GL_BLEND);
        glEnable (GL_DEPTH_TEST);
        glDisable (GL_CULL_FACE);
        glDisable (GL_SCISSOR_TEST);
        glDisable (GL_RASTERIZER_DISCARD);
        glDisable (GL_COLOR_LOGIC_OP);
        glDisable (GL_FRAMEBUFFER_SRGB);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask (GL_TRUE);
        glDepthFunc (GL_LESS);
        glClearColor (7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f, 1.0f);
        glClearDepth (1.0);
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gl.UseProgram (resources->program);
        gl.BindVertexArray (resources->vertexArray);
        gl.ActiveTexture (GL_TEXTURE0);
        glBindTexture (GL_TEXTURE_2D, resources->textures[0]);
        gl.BindSampler (0, 0);
        const auto& locations = resources->uniforms;
        gl.UniformMatrix4fv (locations.objectMatrix, 1, GL_FALSE,
                             values.objectMatrix.data());
        gl.Uniform1f (locations.tangentHandednessSign,
                      fixtureMatrixDeterminant3x3 (values.objectMatrix) < 0.0f
                          ? -1.0f : 1.0f);
        gl.Uniform4fv (locations.cameraRotation, 1, values.cameraRotation.data());
        gl.Uniform3f (locations.cameraTranslation, values.cameraTranslation[0],
                      values.cameraTranslation[1], values.cameraTranslation[2]);
        gl.Uniform4fv (locations.projection, 1, values.projection.data());
        gl.Uniform4fv (locations.baseColor, 1, values.baseColor.data());
        gl.Uniform4fv (locations.materialParams, 1, values.materialParams.data());
        gl.Uniform4fv (locations.timeMixEndColorAndTime, 1,
                       values.timeMixEndColorAndTime.data());
        gl.Uniform3f (locations.emissive, values.emissive[0],
                      values.emissive[1], values.emissive[2]);
        gl.Uniform3f (locations.ambient, values.ambient[0],
                      values.ambient[1], values.ambient[2]);
        gl.Uniform4fv (locations.noteInstanceTransforms,
                       static_cast<int>(visualnoteinstancing::kMaximumInstances),
                       noteInstances.transforms[0].data());
        gl.Uniform1i (locations.noteInstanceIndex, -1);
        gl.Uniform4fv (locations.lightRotation, 1, values.lightRotation.data());
        gl.Uniform4fv (locations.lightColorIntensity, 1,
                       values.lightColorIntensity.data());
        gl.Uniform4fv (locations.lightPositionRange, 1,
                       values.lightPositionRange.data());
        gl.Uniform1i (locations.lightKind,
                      scene->lightCount == 1
                          && scene->lights[0].kind == SceneLightKind::Point ? 1 : 0);
        gl.Uniform1i (locations.lightCount, static_cast<int> (scene->lightCount));
        for (std::size_t lightIndex = 0; lightIndex < scene->lightCount; ++lightIndex)
        {
            const auto& light = scene->lights[lightIndex];
            std::array<float, 4> rotation {}, color {}, position {}, cones {};
            storeFixtureQuaternion (rotation, light.transform.rotation);
            storeFixtureVec3 (color, light.color, light.intensity);
            storeFixtureVec3 (position, light.transform.translation, light.range);
            cones = { std::cos (light.innerConeAngle),
                      std::cos (light.outerConeAngle), 0.0f, 0.0f };
            gl.Uniform4fv (locations.lightRotations[lightIndex], 1, rotation.data());
            gl.Uniform4fv (locations.lightColors[lightIndex], 1, color.data());
            gl.Uniform4fv (locations.lightPositions[lightIndex], 1, position.data());
            gl.Uniform4fv (locations.lightCones[lightIndex], 1, cones.data());
            gl.Uniform1i (locations.lightKinds[lightIndex], static_cast<int> (light.kind));
        }
        gl.Uniform1i (locations.baseTexture, 0);
        gl.Uniform1i (locations.metallicRoughnessTexture, 1);
        gl.Uniform1i (locations.normalTexture, 2);
        gl.Uniform1i (locations.occlusionTexture, 3);
        gl.Uniform1i (locations.emissiveTexture, 4);
        const auto materialKind = resources->materialProgram != nullptr
            ? resources->materialProgram->kind : NativeFixtureMaterialKind::SurfacePbr;
        gl.Uniform1i (locations.materialKind, static_cast<int> (materialKind));
        diffractivefoil::EvaluationSchedule spatialSchedule;
        std::uint32_t instanceBufferUploadCount = 0;
        std::uint32_t instancedDrawCount = 0;
        std::uint32_t submittedInstanceCount = 0;
        std::uint32_t noteInstanceDrawCount = 0;
        std::uint32_t submittedNoteInstanceCount = 0;
        if (materialKind == NativeFixtureMaterialKind::DiffractionReflective)
        {
            const diffractionmaterial::PhysicalDiffractionGpuLightingPath emptyPath {};
            const auto diffractionPathCount = materialKind
                    == NativeFixtureMaterialKind::DiffractionReflective
                ? resources->materialProgram->diffractionPathCount : 1u;

            if (resources->materialProgram != nullptr
                && resources->materialProgram->diffractionFoilMaximumEvaluations != 0
                && !nativeFixtureSpatialFoilEvaluationSchedule(
                    *resources->materialProgram, width, height, spatialSchedule))
            {
                result.error = "OpenGL fixture diffraction workload exceeds backend limits";
                return result;
            }
            const auto uintAsFloat = [] (std::uint32_t value) noexcept
            {
                float encoded = 0.0f;
                std::memcpy(&encoded, &value, sizeof(value));
                return encoded;
            };
            for (std::uint8_t pathIndex = 0; pathIndex < diffractionPathCount; ++pathIndex)
            {
                const auto& lightingPath = materialKind
                        == NativeFixtureMaterialKind::DiffractionReflective
                    ? resources->materialProgram->diffractionPaths[pathIndex]
                    : emptyPath;
                const auto& diffraction = lightingPath.material;
                gl.Uniform4fv (locations.diffractionGeometry, 1, &diffraction.geometry.x);
                gl.Uniform4fv (locations.diffractionSecondaryGeometry, 1,
                               &diffraction.secondaryGeometry.x);
                gl.Uniform4fv (locations.diffractionMicrostructure, 1,
                               &diffraction.microstructure.x);
                gl.Uniform4fv (locations.diffractionControl, 1, &diffraction.control.x);
                gl.Uniform4fv (locations.diffractionCoating, 1, &diffraction.coating.x);
                gl.Uniform4fv (locations.diffractionRoughness, 1, &diffraction.roughness.x);
                gl.Uniform4fv (locations.diffractionGrooveField, 1, &diffraction.grooveField.x);
                gl.Uniform4fv (locations.diffractionGrooveVariation, 1,
                               &diffraction.grooveVariation.x);
                gl.Uniform4fv (locations.diffractionSpectral, 8, &diffraction.spectral[0].x);
                gl.Uniform4fv (locations.diffractionSpectralZ, 8, &diffraction.spectralZ[0].x);
                gl.Uniform4fv (locations.diffractionIncidentDirectionAndIntensity, 1,
                               &lightingPath.incidentDirectionAndIntensity.x);
                float maskCoverage = 0.0f;
                std::memcpy(&maskCoverage, &lightingPath.kindBounceAndReserved[3],
                            sizeof(maskCoverage));
                gl.Uniform4f (locations.diffractionPathKindAndBounce,
                              static_cast<float>(lightingPath.kindBounceAndReserved[0]),
                              static_cast<float>(lightingPath.kindBounceAndReserved[1]),
                              static_cast<float>(lightingPath.kindBounceAndReserved[2]),
                              maskCoverage);
                gl.Uniform4fv(locations.diffractionFoilField, 4,
                              &resources->materialProgram->diffractionFoilField[0].x);
                gl.Uniform4fv(locations.diffractionOccupancyRectangles, 5,
                              &resources->materialProgram->diffractionOccupancyRectangles[0].x);
                gl.Uniform4f(locations.diffractionSpatialCounts,
                             resources->materialProgram->diffractionFoilMaximumEvaluations > 0 ? 4.0f : 0.0f,
                             static_cast<float>(resources->materialProgram->diffractionOccupancyRectangleCount),
                             0.0f, 0.0f);
                gl.Uniform4f(locations.diffractionEvaluationSchedule,
                             uintAsFloat(spatialSchedule.width),
                             uintAsFloat(spatialSchedule.height),
                             uintAsFloat(spatialSchedule.maximumEvaluations),
                             uintAsFloat(pathIndex));
                if (pathIndex != 0)
                {
                    glEnable(GL_BLEND);
                    glBlendFunc(GL_ONE, GL_ONE);
                    glDepthFunc(GL_EQUAL);
                    glDepthMask(GL_FALSE);
                }
                gl.DrawElementsInstanced (GL_TRIANGLES,
                                          static_cast<int> (object.indexCount),
                                          GL_UNSIGNED_INT, nullptr,
                                          static_cast<int>(noteInstances.count));
                if (noteInstances.admitted)
                {
                    ++noteInstanceDrawCount;
                    submittedNoteInstanceCount += static_cast<std::uint32_t>(noteInstances.count);
                }
            }
        }
        else
        {
            const diffractionmaterial::PhysicalDiffractionGpuParameters emptyDiffraction {};
            const auto& diffraction = emptyDiffraction;
            gl.Uniform4fv (locations.diffractionGeometry, 1, &diffraction.geometry.x);
            gl.Uniform4fv (locations.diffractionSecondaryGeometry, 1,
                           &diffraction.secondaryGeometry.x);
            gl.Uniform4fv (locations.diffractionMicrostructure, 1,
                           &diffraction.microstructure.x);
            gl.Uniform4fv (locations.diffractionControl, 1, &diffraction.control.x);
            gl.Uniform4fv (locations.diffractionCoating, 1, &diffraction.coating.x);
            gl.Uniform4fv (locations.diffractionRoughness, 1, &diffraction.roughness.x);
            gl.Uniform4fv (locations.diffractionSpectral, 8, &diffraction.spectral[0].x);
            gl.Uniform4fv (locations.diffractionSpectralZ, 8, &diffraction.spectralZ[0].x);
            std::vector<std::size_t> drawOrder;
            std::vector<std::pair<float, std::size_t>> blendedDraws;
            drawOrder.reserve (scene->objectCount);
            for (std::size_t objectIndex = 0; objectIndex < scene->objectCount; ++objectIndex)
            {
                const auto* queuedMaterial = visual3d_detail::findById (
                    scene->materials, scene->materialCount, scene->objects[objectIndex].material);
                if (queuedMaterial == nullptr || queuedMaterial->alphaMode != SceneAlphaMode::Blend)
                    drawOrder.push_back (objectIndex);
                else
                {
                    const auto matrix = fixtureWorldMatrix (*scene, scene->objects[objectIndex]);
                    const auto dx = matrix[12] - values.cameraTranslation[0];
                    const auto dy = matrix[13] - values.cameraTranslation[1];
                    const auto dz = matrix[14] - values.cameraTranslation[2];
                    blendedDraws.emplace_back (dx * dx + dy * dy + dz * dz, objectIndex);
                }
            }
            std::stable_sort (blendedDraws.begin(), blendedDraws.end(),
                              [] (const auto& left, const auto& right)
                              { return left.first > right.first; });
            for (const auto& blend : blendedDraws) drawOrder.push_back (blend.second);
            std::array<OpenGlInstanceGpuRecord, Visual3DScene::kMaxObjects>
                instanceRecords {};
            if (resources->instancedSharedGeometry)
            {
                std::vector<videowire::geometry::StableId> diagnosticStableIdentities;
                diagnosticStableIdentities.reserve(scene->objectCount);
                for (std::size_t index = 0; index < scene->objectCount; ++index)
                    diagnosticStableIdentities.push_back(scene->objects[index].id.value);
                std::sort(diagnosticStableIdentities.begin(), diagnosticStableIdentities.end());
                diagnosticStableIdentities.erase(
                    std::unique(diagnosticStableIdentities.begin(), diagnosticStableIdentities.end()),
                    diagnosticStableIdentities.end());
                auto runtimeTransform = SceneTransform3D {};
                runtimeTransform.translation = {
                    runtimeInputs.objectTranslationOffset[0],
                    runtimeInputs.objectTranslationOffset[1],
                    runtimeInputs.objectTranslationOffset[2]
                };
                runtimeTransform = applyRuntimeObjectTransform(
                    runtimeTransform, runtimeInputs.objectRotationDegrees,
                    runtimeInputs.objectScale);
                const auto runtimeMatrix = fixtureTransformMatrix(runtimeTransform);
                for (std::size_t index = 0; index < scene->objectCount; ++index)
                {
                    instanceRecords[index].matrix = multiplyFixtureMatrices(
                        runtimeMatrix, fixtureWorldMatrix(*scene, scene->objects[index]));
                    const auto stableIdentity = scene->objects[index].id.value;
                    const auto colorSlot = videohelper::geometry::diagnosticColorSlot(
                        diagnosticStableIdentities, stableIdentity);
                    instanceRecords[index].identityColor = {
                        static_cast<float>((colorSlot >> 16) & 0xffu) / 255.0f,
                        static_cast<float>((colorSlot >> 8) & 0xffu) / 255.0f,
                        static_cast<float>(colorSlot & 0xffu) / 255.0f,
                        resources->diagnosticInstanceIdentityColors ? 1.0f : 0.0f };
                }
                gl.BindBuffer(GL_ARRAY_BUFFER, resources->instanceBuffer);
                gl.BufferData(GL_ARRAY_BUFFER,
                    static_cast<std::ptrdiff_t>(scene->objectCount
                        * sizeof(OpenGlInstanceGpuRecord)),
                    instanceRecords.data(), GL_DYNAMIC_DRAW);
                ++instanceBufferUploadCount;
            }
            for (const auto objectIndex : drawOrder)
            {
                const auto& drawObject = scene->objects[objectIndex];
                const auto* drawMaterial = visual3d_detail::findById (
                    scene->materials, scene->materialCount, drawObject.material);
                auto runtimeTransform = HarmonicMIDI::grid::SceneTransform3D {};
                runtimeTransform.translation = {
                    runtimeInputs.objectTranslationOffset[0],
                    runtimeInputs.objectTranslationOffset[1],
                    runtimeInputs.objectTranslationOffset[2]
                };
                runtimeTransform = applyRuntimeObjectTransform (
                    runtimeTransform, runtimeInputs.objectRotationDegrees,
                    runtimeInputs.objectScale);
                values.objectMatrix = multiplyFixtureMatrices (
                    fixtureTransformMatrix (runtimeTransform),
                    fixtureWorldMatrix (*scene, drawObject));
                if (resources->materialProgram == nullptr && drawMaterial != nullptr)
                {
                    storeFixtureVec3 (values.baseColor, drawMaterial->baseColor,
                                      drawMaterial->opacity);
                    values.materialParams = {
                        drawMaterial->metallic, drawMaterial->roughness,
                        drawMaterial->normalScale, 0.0f
                    };
                    storeFixtureVec3 (values.emissive, drawMaterial->emissive);
                    for (std::size_t channel = 0; channel < 3; ++channel)
                        values.emissive[channel] *= runtimeInputs.emissionGain;
                    gl.Uniform4fv (locations.baseColor, 1, values.baseColor.data());
                    gl.Uniform4fv (locations.materialParams, 1,
                                   values.materialParams.data());
                    gl.Uniform3f (locations.emissive, values.emissive[0],
                                  values.emissive[1], values.emissive[2]);
                }
                gl.UniformMatrix4fv (locations.objectMatrix, 1, GL_FALSE,
                                     values.objectMatrix.data());
                gl.Uniform1i(locations.useInstanceMatrix,
                             resources->instancedSharedGeometry ? 1 : 0);
                const auto textureSlot = [&] (SceneTextureId id)
                {
                    if (id.isValid())
                        for (std::size_t index = 0; index < scene->textureCount; ++index)
                            if (scene->textures[index].id == id)
                                return 1 + index;
                    return std::size_t { 0 };
                };
                const std::array<SceneTextureId, 5> textureIds {
                    drawMaterial != nullptr ? drawMaterial->baseColorTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->metallicRoughnessTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->normalTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->occlusionTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->emissiveTexture : SceneTextureId {}
                };
                for (std::size_t unit = 0; unit < textureIds.size(); ++unit)
                {
                    gl.ActiveTexture (GL_TEXTURE0 + static_cast<unsigned> (unit));
                    glBindTexture (GL_TEXTURE_2D,
                                   resources->textures[textureSlot (textureIds[unit])]);
                    gl.BindSampler (static_cast<unsigned> (unit), 0);
                }
                gl.Uniform4f (locations.texturePresence,
                              textureIds[1].isValid() ? 1.0f : 0.0f,
                              textureIds[2].isValid() ? 1.0f : 0.0f,
                              textureIds[3].isValid() ? 1.0f : 0.0f, 0.0f);
                gl.Uniform1f (locations.emissiveTexturePresence,
                              textureIds[4].isValid() ? 1.0f : 0.0f);
                const auto alphaMode = drawMaterial != nullptr
                    ? drawMaterial->alphaMode : SceneAlphaMode::Opaque;
                gl.Uniform2f (locations.alphaModeCutoff,
                              alphaMode == SceneAlphaMode::Mask ? 1.0f : 0.0f,
                              drawMaterial != nullptr ? drawMaterial->alphaCutoff : 0.5f);
                gl.Uniform1f (locations.tangentHandednessSign,
                              resources->instancedSharedGeometry ? 1.0f
                              : fixtureMatrixDeterminant3x3 (values.objectMatrix) < 0.0f
                                  ? -1.0f : 1.0f);
                if (alphaMode == SceneAlphaMode::Blend)
                {
                    glEnable (GL_BLEND);
                    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    glDepthMask (GL_FALSE);
                }
                else
                {
                    glDisable (GL_BLEND);
                    glDepthMask (GL_TRUE);
                }
                if (drawObject.indexCount == 0)
                    continue;
                if (drawMaterial != nullptr && drawMaterial->doubleSided)
                    glDisable (GL_CULL_FACE);
                else
                {
                    glEnable (GL_CULL_FACE);
                    glCullFace (GL_BACK);
                }
                glFrontFace (fixtureMatrixDeterminant3x3 (values.objectMatrix) < 0.0f
                                 ? GL_CW : GL_CCW);
                const auto indexOffset = reinterpret_cast<const void*> (
                    static_cast<std::uintptr_t> (drawObject.firstIndex)
                    * sizeof (std::uint32_t));
                if (resources->instancedSharedGeometry)
                {
                    for (std::size_t noteIndex = 0; noteIndex < noteInstances.count; ++noteIndex)
                    {
                        gl.Uniform1i(locations.noteInstanceIndex, static_cast<int>(noteIndex));
                        gl.DrawElementsInstanced(GL_TRIANGLES,
                            static_cast<int>(drawObject.indexCount), GL_UNSIGNED_INT,
                            indexOffset, static_cast<int>(scene->objectCount));
                        ++instancedDrawCount;
                        submittedInstanceCount += static_cast<std::uint32_t>(scene->objectCount);
                        if (noteInstances.admitted)
                        {
                            ++noteInstanceDrawCount;
                            submittedNoteInstanceCount += static_cast<std::uint32_t>(scene->objectCount);
                        }
                    }
                    break;
                }
                gl.Uniform1i(locations.noteInstanceIndex, -1);
                gl.DrawElementsInstanced(GL_TRIANGLES,
                    static_cast<int>(drawObject.indexCount), GL_UNSIGNED_INT,
                    indexOffset, static_cast<int>(noteInstances.count));
                if (noteInstances.admitted)
                {
                    ++noteInstanceDrawCount;
                    submittedNoteInstanceCount += static_cast<std::uint32_t>(noteInstances.count);
                }
        }
        }
        glFinish();
        const auto gpuError = glGetError();
        cleanup();
        if (gpuError != GL_NO_ERROR)
        {
            if (frame->texture_ != 0)
            {
                glDeleteTextures (1, &frame->texture_);
                frame->texture_ = 0;
            }
            result.error = "OpenGL fixture draw reported GPU error "
                         + std::to_string (static_cast<unsigned> (gpuError));
            return result;
        }

        result.rendered = true;
        result.frame = std::move (frame);
        result.stats.drawCount = resources->instancedSharedGeometry ? instancedDrawCount
            : materialKind == NativeFixtureMaterialKind::DiffractionReflective
            ? resources->materialProgram->diffractionPathCount
            : static_cast<std::size_t>(std::count_if(
                scene->objects.begin(), scene->objects.begin() + scene->objectCount,
                [] (const auto& drawObject) { return drawObject.indexCount != 0; }));
        result.stats.ordinaryDrawCount = resources->instancedSharedGeometry
            ? 0 : result.stats.drawCount;
        result.stats.instancedDrawCount = instancedDrawCount;
        result.stats.instanceBufferUploadCount = instanceBufferUploadCount;
        result.stats.submittedInstanceCount = submittedInstanceCount;
        result.stats.noteInstanceDrawCount = noteInstanceDrawCount;
        result.stats.noteInstanceTransformUploadCount = noteInstances.admitted ? 1u : 0u;
        result.stats.submittedNoteInstanceCount = submittedNoteInstanceCount;
        result.stats.materialBytes = sizeof (values) * result.stats.drawCount;
        result.stats.reusedStaticResources = true;
        result.stats.reusedMaterialProgram = resources->materialProgram != nullptr;
        result.stats.diffractionEvaluationBudget = spatialSchedule.maximumEvaluations;
        result.stats.diffractionEvaluationCount = spatialSchedule.requiredEvaluations;
        return result;
    }
};

const char* kDeformationComputeShader = R"glsl(#version 430 core
layout(local_size_x = 64) in;
layout(std430, binding=0) readonly buffer BaseVertices { float baseVertices[]; };
layout(std430, binding=1) writeonly buffer OutputVertices { float outputVertices[]; };
layout(std430, binding=2) readonly buffer JointIndices { uint jointIndices[]; };
layout(std430, binding=3) readonly buffer JointWeights { float jointWeights[]; };
layout(std430, binding=4) readonly buffer MorphPositions { float morphPositions[]; };
layout(std430, binding=5) readonly buffer MorphNormals { float morphNormals[]; };
layout(std430, binding=6) readonly buffer JointPalette { float jointPalette[]; };
layout(std430, binding=7) readonly buffer MorphWeights { float morphWeights[]; };
uniform uint uVertexCount;
uniform uint uJointCount;
uniform uint uMorphTargetCount;
mat4 jointMatrix(uint index)
{
    uint first = index * 16u;
    return mat4(
        jointPalette[first + 0u], jointPalette[first + 1u],
        jointPalette[first + 2u], jointPalette[first + 3u],
        jointPalette[first + 4u], jointPalette[first + 5u],
        jointPalette[first + 6u], jointPalette[first + 7u],
        jointPalette[first + 8u], jointPalette[first + 9u],
        jointPalette[first + 10u], jointPalette[first + 11u],
        jointPalette[first + 12u], jointPalette[first + 13u],
        jointPalette[first + 14u], jointPalette[first + 15u]);
}
void main()
{
    uint vertex = gl_GlobalInvocationID.x;
    if (vertex >= uVertexCount) return;
    uint base = vertex * 8u;
    vec3 position = vec3(baseVertices[base], baseVertices[base + 1u],
                         baseVertices[base + 2u]);
    vec3 normal = vec3(baseVertices[base + 3u], baseVertices[base + 4u],
                       baseVertices[base + 5u]);
    for (uint target = 0u; target < uMorphTargetCount; ++target)
    {
        uint delta = (target * uVertexCount + vertex) * 3u;
        float weight = morphWeights[target];
        position += vec3(morphPositions[delta], morphPositions[delta + 1u],
                         morphPositions[delta + 2u]) * weight;
        normal += vec3(morphNormals[delta], morphNormals[delta + 1u],
                       morphNormals[delta + 2u]) * weight;
    }
    if (uJointCount > 0u)
    {
        mat4 skin = mat4(0.0);
        uint influence = vertex * 4u;
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            uint joint = jointIndices[influence + slot];
            float weight = jointWeights[influence + slot];
            if (joint < uJointCount && weight != 0.0)
                skin += jointMatrix(joint) * weight;
        }
        position = (skin * vec4(position, 1.0)).xyz;
        normal = mat3(skin) * normal;
    }
    float normalLength = length(normal);
    normal = normalLength > 1.0e-8 ? normal / normalLength : vec3(0.0, 0.0, 1.0);
    outputVertices[base] = position.x;
    outputVertices[base + 1u] = position.y;
    outputVertices[base + 2u] = position.z;
    outputVertices[base + 3u] = normal.x;
    outputVertices[base + 4u] = normal.y;
    outputVertices[base + 5u] = normal.z;
    outputVertices[base + 6u] = baseVertices[base + 6u];
    outputVertices[base + 7u] = baseVertices[base + 7u];
}
)glsl";

class OpenGlDeformationResources final : public NativeDeformationResources
{
public:
    ~OpenGlDeformationResources() override
    {
        if (ownerContext == nullptr)
            return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        auto* previous = glfwGetCurrentContext();
        if (previous != ownerContext)
            glfwMakeContextCurrent (ownerContext);
        if (glfwGetCurrentContext() == ownerContext)
        {
            if (computeProgram != 0) gl.DeleteProgram (computeProgram);
            for (auto& buffer : buffers)
                if (buffer != 0) gl.DeleteBuffers (1, &buffer);
        }
        if (previous != ownerContext)
            glfwMakeContextCurrent (previous);
    }

    const std::string& backend() const noexcept override { return backend_; }

    std::string backend_ = "opengl";
    GLFWwindow* ownerContext = nullptr;
    arbitgl::GlFuncs gl {};
    std::shared_ptr<const NativeDeformationScene> source;
    std::shared_ptr<const OpenGlFixtureSceneResources> fixture;
    std::array<unsigned, 7> buffers {};
    unsigned computeProgram = 0;
    int vertexCountLocation = -1;
    int jointCountLocation = -1;
    int morphTargetCountLocation = -1;
    mutable std::mutex submissionMutex;
};

class OpenGlDeformationBackend final : public NativeDeformationBackend
{
public:
    BackendInfo info() const override
    {
        BackendInfo result;
        result.backend = "opengl";
        if (glfwGetCurrentContext() == nullptr)
        {
            result.error = "native OpenGL deformation requires a current OpenGL 4.3 context";
            return result;
        }
        int major = 0;
        int minor = 0;
        glGetIntegerv (GL_MAJOR_VERSION, &major);
        glGetIntegerv (GL_MINOR_VERSION, &minor);
        if (major < 4 || (major == 4 && minor < 3))
        {
            result.error = "native OpenGL deformation requires OpenGL 4.3 compute shaders";
            return result;
        }
        arbitgl::GlFuncs gl;
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing)
            || ! arbitgl::loadGl43Functions (gl))
        {
            result.error = "native OpenGL deformation is missing required functions: " + missing;
            return result;
        }
        const auto* renderer = glGetString (GL_RENDERER);
        result.available = true;
        result.compute = true;
        result.device = renderer != nullptr
            ? reinterpret_cast<const char*> (renderer) : "OpenGL 4.3 context";
        return result;
    }

    NativeDeformationPreparation prepare (
        const std::shared_ptr<const NativeDeformationScene>& source,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram) override
    {
        NativeDeformationPreparation result;
        const auto available = info();
        if (! available.available)
        {
            result.error = available.error;
            return result;
        }
        if (source == nullptr || source->sourceStableId == 0
            || source->deformationStableId == 0 || source->structuralRevision == 0
            || ! source->clip.isValid() || ! source->mesh.isValid()
            || ! source->object.isValid() || ! source->scene || ! source->deformation
            || source->scene->objectCount != 1 || source->scene->materialCount != 1
            || source->scene->lightCount != 1 || source->scene->cameraCount != 1
            || source->scene->objects[0].id != source->object
            || source->scene->objects[0].firstVertex != 0
            || source->scene->objects[0].vertexCount != source->scene->vertexCount)
        {
            result.error = "native OpenGL deformation requires one exact bounded scene owner";
            return result;
        }
        const auto* mesh = source->deformation->findMesh (source->mesh);
        if (mesh == nullptr || mesh->vertexCount() != source->scene->vertexCount
            || mesh->vertexCount() == 0
            || mesh->vertexCount() > kNativeDeformationMaxVertices
            || mesh->jointWeightSets().size() > 1
            || mesh->morphTargets().size() > kNativeDeformationMaxMorphTargets)
        {
            result.error = "native OpenGL deformation mesh exceeds the bounded GPU subset";
            return result;
        }
        for (const auto& target : mesh->morphTargets())
            if (target.hasTangentDeltas())
            {
                result.error = "native OpenGL deformation does not admit tangent morph deltas";
                return result;
            }

        OpenGlFixtureSceneBackend fixtureBackend;
        auto fixturePreparation = fixtureBackend.prepare (source->scene, materialProgram);
        if (! fixturePreparation.prepared)
        {
            result.error = fixturePreparation.error;
            return result;
        }
        auto fixture = std::dynamic_pointer_cast<const OpenGlFixtureSceneResources> (
            fixturePreparation.resources);
        if (fixture == nullptr)
        {
            result.error = "native OpenGL deformation did not receive fixture GPU resources";
            return result;
        }

        std::lock_guard<std::mutex> lock (fixtureMutex());
        arbitgl::GlFuncs gl;
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing)
            || ! arbitgl::loadGl43Functions (gl))
        {
            result.error = "native OpenGL deformation loader failed: " + missing;
            return result;
        }
        OpenGlFixtureState previous (gl);
        int previousShaderStorageBuffer = 0;
        glGetIntegerv (GL_SHADER_STORAGE_BUFFER_BINDING, &previousShaderStorageBuffer);
        auto resources = std::make_shared<OpenGlDeformationResources>();
        resources->ownerContext = glfwGetCurrentContext();
        resources->gl = gl;
        resources->source = source;
        resources->fixture = std::move (fixture);
        auto fail = [&] (std::string error)
        {
            if (resources->computeProgram != 0)
                gl.DeleteProgram (resources->computeProgram);
            resources->computeProgram = 0;
            for (auto& buffer : resources->buffers)
            {
                if (buffer != 0) gl.DeleteBuffers (1, &buffer);
                buffer = 0;
            }
            gl.BindBuffer (GL_SHADER_STORAGE_BUFFER,
                           static_cast<unsigned> (previousShaderStorageBuffer));
            previous.restore();
            result.error = std::move (error);
            return result;
        };

        const auto vertexCount = mesh->vertexCount();
        const auto vertexBytes = vertexCount * sizeof (HarmonicMIDI::grid::SceneVertex);
        std::vector<std::uint32_t> jointIndices (vertexCount * 4u, 0u);
        std::vector<float> jointWeights (vertexCount * 4u, 0.0f);
        if (! mesh->jointWeightSets().empty())
        {
            jointIndices = mesh->jointWeightSets()[0].jointIndices();
            jointWeights = mesh->jointWeightSets()[0].weights();
        }
        const auto morphValueCount = std::max<std::size_t> (
            1u, mesh->morphTargets().size() * vertexCount * 3u);
        std::vector<float> morphPositions (morphValueCount, 0.0f);
        std::vector<float> morphNormals (morphValueCount, 0.0f);
        for (std::size_t target = 0; target < mesh->morphTargets().size(); ++target)
        {
            const auto first = target * vertexCount * 3u;
            const auto& position = mesh->morphTargets()[target].positionDeltas();
            std::copy (position.begin(), position.end(), morphPositions.begin() + first);
            const auto& normal = mesh->morphTargets()[target].normalDeltas();
            if (! normal.empty())
                std::copy (normal.begin(), normal.end(), morphNormals.begin() + first);
        }
        std::array<float, kNativeDeformationMaxJoints * 16> jointPalette {};
        std::array<float, kNativeDeformationMaxMorphTargets> morphWeights {};
        gl.GenBuffers (static_cast<int> (resources->buffers.size()),
                       resources->buffers.data());
        const auto upload = [&] (std::size_t index, const void* bytes,
                                 std::size_t size, unsigned usage)
        {
            gl.BindBuffer (GL_SHADER_STORAGE_BUFFER, resources->buffers[index]);
            gl.BufferData (GL_SHADER_STORAGE_BUFFER, static_cast<std::ptrdiff_t> (size),
                           bytes, usage);
        };
        upload (0, source->scene->vertices.data(), vertexBytes, GL_STATIC_DRAW);
        upload (1, jointIndices.data(), jointIndices.size() * sizeof (std::uint32_t),
                GL_STATIC_DRAW);
        upload (2, jointWeights.data(), jointWeights.size() * sizeof (float), GL_STATIC_DRAW);
        upload (3, morphPositions.data(), morphPositions.size() * sizeof (float), GL_STATIC_DRAW);
        upload (4, morphNormals.data(), morphNormals.size() * sizeof (float), GL_STATIC_DRAW);
        upload (5, jointPalette.data(), jointPalette.size() * sizeof (float), GL_DYNAMIC_DRAW);
        upload (6, morphWeights.data(), morphWeights.size() * sizeof (float), GL_DYNAMIC_DRAW);

        auto shader = compileFixtureShader (
            gl, GL_COMPUTE_SHADER, kDeformationComputeShader, result.error);
        if (shader == 0)
            return fail (result.error);
        resources->computeProgram = gl.CreateProgram();
        gl.AttachShader (resources->computeProgram, shader);
        gl.LinkProgram (resources->computeProgram);
        gl.DeleteShader (shader);
        int linked = 0;
        gl.GetProgramiv (resources->computeProgram, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE)
        {
            char log[1024] = {};
            int length = 0;
            gl.GetProgramInfoLog (resources->computeProgram,
                                  static_cast<int> (sizeof (log)), &length, log);
            return fail (std::string ("native OpenGL deformation program link failed: ")
                + std::string (log, static_cast<std::size_t> (std::max (length, 0))));
        }
        resources->vertexCountLocation = gl.GetUniformLocation (
            resources->computeProgram, "uVertexCount");
        resources->jointCountLocation = gl.GetUniformLocation (
            resources->computeProgram, "uJointCount");
        resources->morphTargetCountLocation = gl.GetUniformLocation (
            resources->computeProgram, "uMorphTargetCount");
        if (resources->computeProgram == 0 || resources->vertexCountLocation < 0
            || resources->jointCountLocation < 0
            || resources->morphTargetCountLocation < 0
            || std::any_of (resources->buffers.begin(), resources->buffers.end(),
                            [] (unsigned buffer) { return buffer == 0; }))
        {
            return fail ("native OpenGL deformation static GPU resource creation failed");
        }

        gl.BindBuffer (GL_SHADER_STORAGE_BUFFER,
                       static_cast<unsigned> (previousShaderStorageBuffer));
        previous.restore();
        result.prepared = true;
        result.resources = std::move (resources);
        result.stats.staticUploadCount = 1;
        result.stats.staticVertexBytes = vertexBytes * 2u;
        result.stats.staticDeformationBytes =
            jointIndices.size() * sizeof (std::uint32_t)
            + jointWeights.size() * sizeof (float)
            + morphPositions.size() * sizeof (float)
            + morphNormals.size() * sizeof (float);
        result.stats.sourceStableId = source->sourceStableId;
        result.stats.deformationStableId = source->deformationStableId;
        result.stats.clipId = source->clip.value;
        result.stats.meshId = source->mesh.value;
        result.stats.skinId = mesh->skin().value;
        result.stats.revision = source->structuralRevision;
        return result;
    }

    NativeDeformationSubmission render (
        const std::shared_ptr<const NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& snapshot,
        const std::shared_ptr<const NativeDeformationResources>& nativeResources,
        std::uint32_t width,
        std::uint32_t height,
        NativeDeformationRuntimeInputs runtimeInputs) override
    {
        NativeDeformationSubmission result;
        const auto resources = std::dynamic_pointer_cast<const OpenGlDeformationResources> (
            nativeResources);
        if (source == nullptr || snapshot == nullptr || resources == nullptr
            || resources->source != source || resources->ownerContext != glfwGetCurrentContext())
        {
            result.error = "native OpenGL deformation requires exact prepared owner resources";
            return result;
        }
        std::lock_guard<std::mutex> submissionLock (resources->submissionMutex);
        NativeDeformationFrameData frameData;
        if (! prepareNativeDeformationFrame (*source, *snapshot, frameData, result.error))
            return result;
        for (std::size_t index = 0; index < frameData.morphTargetCount; ++index)
        {
            const auto base = source->morphBaseWeights.empty()
                ? 0.0f : source->morphBaseWeights[index];
            frameData.morphWeights[index]
                = base + (frameData.morphWeights[index] - base) * runtimeInputs.morphWeight;
        }

        {
            std::lock_guard<std::mutex> lock (fixtureMutex());
            const auto& gl = resources->gl;
            OpenGlFixtureState previous (gl);
            int previousShaderStorageBuffer = 0;
            std::array<int, 8> previousBindings {};
            glGetIntegerv (GL_SHADER_STORAGE_BUFFER_BINDING, &previousShaderStorageBuffer);
            for (unsigned binding = 0; binding < previousBindings.size(); ++binding)
                gl.GetIntegeri_v (GL_SHADER_STORAGE_BUFFER_BINDING, binding,
                                  &previousBindings[binding]);
            const auto restore = [&]
            {
                previous.restore();
                for (unsigned binding = 0; binding < previousBindings.size(); ++binding)
                    gl.BindBufferBase (GL_SHADER_STORAGE_BUFFER, binding,
                                       static_cast<unsigned> (previousBindings[binding]));
                gl.BindBuffer (GL_SHADER_STORAGE_BUFFER,
                               static_cast<unsigned> (previousShaderStorageBuffer));
            };
            gl.BindBuffer (GL_SHADER_STORAGE_BUFFER, resources->buffers[5]);
            gl.BufferData (GL_SHADER_STORAGE_BUFFER,
                           static_cast<std::ptrdiff_t> (frameData.jointPalette.size()
                                                       * sizeof (float)),
                           frameData.jointPalette.data(), GL_DYNAMIC_DRAW);
            gl.BindBuffer (GL_SHADER_STORAGE_BUFFER, resources->buffers[6]);
            gl.BufferData (GL_SHADER_STORAGE_BUFFER,
                           static_cast<std::ptrdiff_t> (frameData.morphWeights.size()
                                                       * sizeof (float)),
                           frameData.morphWeights.data(), GL_DYNAMIC_DRAW);
            gl.BindBufferBase (GL_SHADER_STORAGE_BUFFER, 0, resources->buffers[0]);
            gl.BindBufferBase (GL_SHADER_STORAGE_BUFFER, 1,
                               resources->fixture->vertexBuffer);
            for (unsigned binding = 2; binding < 8; ++binding)
                gl.BindBufferBase (GL_SHADER_STORAGE_BUFFER, binding,
                                   resources->buffers[binding - 1u]);
            gl.UseProgram (resources->computeProgram);
            gl.Uniform1ui (resources->vertexCountLocation, frameData.vertexCount);
            gl.Uniform1ui (resources->jointCountLocation, frameData.jointCount);
            gl.Uniform1ui (resources->morphTargetCountLocation,
                           frameData.morphTargetCount);
            gl.DispatchCompute ((frameData.vertexCount + 63u) / 64u, 1, 1);
            gl.MemoryBarrier (GL_SHADER_STORAGE_BARRIER_BIT
                              | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
            glFinish();
            const auto gpuError = glGetError();
            restore();
            if (gpuError != GL_NO_ERROR)
            {
                result.error = "native OpenGL deformation dispatch reported GPU error "
                             + std::to_string (static_cast<unsigned> (gpuError));
                return result;
            }
        }

        auto fixtureSubmission = nativeFixtureSceneBackend().render (
            source->scene, resources->fixture, width, height, runtimeInputs);
        if (! fixtureSubmission.rendered)
        {
            result.error = fixtureSubmission.error;
            return result;
        }
        const auto* mesh = source->deformation->findMesh (source->mesh);
        result.rendered = true;
        result.frame = std::move (fixtureSubmission.frame);
        result.stats.drawCount = fixtureSubmission.stats.drawCount;
        result.stats.dispatchCount = 1;
        result.stats.reusedStaticResources = true;
        result.stats.dynamicUniformBytes =
            static_cast<std::uint64_t> (frameData.jointCount) * 16u * sizeof (float)
            + static_cast<std::uint64_t> (frameData.morphTargetCount) * sizeof (float);
        result.stats.sourceStableId = source->sourceStableId;
        result.stats.deformationStableId = source->deformationStableId;
        result.stats.clipId = source->clip.value;
        result.stats.meshId = source->mesh.value;
        result.stats.skinId = mesh != nullptr ? mesh->skin().value : 0;
        result.stats.revision = source->structuralRevision;
        result.stats.time = snapshot->time();
        return result;
    }
};
} // namespace

BackendInfo queryNativeBackend()
{
    BackendInfo result;
    result.error = "general native GPU backend not compiled in; OpenGL SDF execution is separate";
    return result;
}

BackendSelfTest runNativeBackendSelfTest()
{
    BackendSelfTest result;
    result.error = "general native GPU backend not compiled in; OpenGL SDF execution is separate";
    return result;
}

NativeSdfExecutionBackend& nativeSdfExecutionBackend()
{
    static OpenGlSdfExecutionBackend backend;
    return backend;
}

void invalidateNativeSdfExecutionContext (std::uintptr_t contextIdentity) noexcept
{
    static_cast<OpenGlSdfExecutionBackend&> (nativeSdfExecutionBackend())
        .invalidateContext (contextIdentity);
}

NativeSdfExecutionCapabilities queryNativeSdfExecution()
{
    return nativeSdfExecutionBackend().capabilities();
}

RenderPassOutputBackend& nativeRenderPassOutputBackend()
{
    static OpenGlRenderPassOutputBackend backend;
    return backend;
}

NativeOpticalFlowExecutionBackend& nativeOpticalFlowExecutionBackend()
{
    static UnavailableOpticalFlowBackend backend;
    return backend;
}

NativeFixtureSceneBackend& nativeFixtureSceneBackend()
{
    static OpenGlFixtureSceneBackend backend;
    return backend;
}

NativeDeformationBackend& nativeDeformationBackend()
{
    static OpenGlDeformationBackend backend;
    return backend;
}
} // namespace arbitgpu
