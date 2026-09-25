#include "backend.h"
#include "fixture_vertex_modifier_shader.h"
#include "fixture_texture_upload.h"
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
float sceneDistanceWithMaterial(vec3 point, out int contributor) {
    int indices[32], stages[32], firstContributors[32]; vec3 points[32]; float firstValues[32];
    int top = 0; indices[0] = uRootIndex; stages[0] = 0; points[0] = point;
    float value = 0.0; contributor = -1;
    for (int iteration = 0; iteration < 768; ++iteration) {
        int index = indices[top]; ivec4 record = ivec4(uRecords[index]);
        int operation = record.x; vec4 p0 = uParameters0[index], p1 = uParameters1[index];
        if (operation <= 9) {
            value = primitiveDistance(operation, points[top], p0, p1);
            contributor = index;
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
                firstContributors[top] = contributor;
                firstValues[top] = value; stages[top] = 2; ++top;
                indices[top] = record.z; stages[top] = 0; points[top] = points[top - 1]; continue;
            }
            float a = firstValues[top], b = value;
            // The dominant smooth weight has the same ordering as its hard
            // Boolean. Cut surfaces belong to B; ordered input A wins ties.
            bool takeFirst = (operation == 10 || operation == 13) ? a <= b
                : a >= ((operation == 12 || operation == 15) ? -b : b);
            if (takeFirst) contributor = firstContributors[top];
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
float sceneDistance(vec3 point) {
    int contributor;
    return sceneDistanceWithMaterial(point, contributor);
}
float qualityScale(int quality) {
    if (quality <= 0) return 4.0; if (quality == 1) return 2.0;
    if (quality == 2) return 1.0; return 0.5;
}
vec3 sceneNormal(vec3 point) {
    float e = max(uEpsilon * qualityScale(uNormalQuality), 0.000001);
    vec3 gradient = vec3(0.0);
    for (int sampleIndex = 0; sampleIndex < 6; ++sampleIndex) {
        int axis = sampleIndex / 2;
        float sign = (sampleIndex % 2) == 0 ? 1.0 : -1.0;
        vec3 offset = vec3(0.0); offset[axis] = sign * e;
        gradient[axis] += sign * sceneDistance(point + offset);
    }
    float squaredLength = dot(gradient, gradient);
    return squaredLength > 0.0 && !isinf(squaredLength) && !isnan(squaredLength)
        ? normalize(gradient) : vec3(0.0,0.0,1.0);
}
float sceneShadow(vec3 origin, vec3 direction) {
    int limit = uShadowQuality <= 0 ? 8 : (uShadowQuality == 1 ? 16 : (uShadowQuality == 2 ? 32 : 64));
    float travel = uEpsilon * 4.0, visibility = 1.0;
    for (int step = 0; step < 64; ++step) {
        if (step >= limit) break; float field = sceneDistance(origin + direction * travel);
        if (isinf(field) || isnan(field)) return 0.0;
        if (field < uEpsilon) return 0.0;
        visibility = min(visibility, 12.0 * field / max(travel, uEpsilon));
        travel += clamp(field, uEpsilon * 2.0, 0.25);
        if (travel > min(uMaximumDistance, 8.0)) break;
    }
    return clamp(visibility, 0.0, 1.0);
}
float sceneCurvature(vec3 point) {
    // Mean curvature of an implicit surface from its gradient and Hessian.
    // One distance-call site avoids inlining 36 complete traversal stacks.
    float h = max(max(uEpsilon*qualityScale(uNormalQuality)*4.0,0.002),length(point)*0.0001);
    const vec3 offsets[19] = vec3[19](vec3(0),
        vec3(1,0,0),vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1),
        vec3(1,1,0),vec3(1,-1,0),vec3(-1,1,0),vec3(-1,-1,0),
        vec3(1,0,1),vec3(1,0,-1),vec3(-1,0,1),vec3(-1,0,-1),
        vec3(0,1,1),vec3(0,1,-1),vec3(0,-1,1),vec3(0,-1,-1));
    float values[19];
    for (int i = 0; i < 19; ++i) values[i] = sceneDistance(point + offsets[i]*h);
    vec3 g = vec3(values[1]-values[2],values[3]-values[4],values[5]-values[6])/(2.0*h);
    vec3 diagonal = (vec3(values[1]+values[2],values[3]+values[4],values[5]+values[6])-2.0*values[0])/(h*h);
    vec3 crossTerms = vec3(values[7]-values[8]-values[9]+values[10],
        values[11]-values[12]-values[13]+values[14],values[15]-values[16]-values[17]+values[18])/(4.0*h*h);
    float g2 = dot(g,g);
    if (g2 <= 0.000000000001 || isinf(g2) || isnan(g2)) return 0.5;
    float directional = dot(g*g,diagonal)
        + 2.0*dot(vec3(g.x*g.y,g.x*g.z,g.y*g.z),crossTerms);
    float curvature = (g2*(diagonal.x+diagonal.y+diagonal.z)-directional)/(2.0*g2*sqrt(g2));
    if (isinf(curvature) || isnan(curvature)) return 0.5;
    curvature = clamp(curvature,-1.0/h,1.0/h);
    return 0.5+0.5*curvature/(1.0+abs(curvature));
}
float sceneAmbientVisibility(vec3 point, vec3 normal) {
    int count = 4+4*uNormalQuality;
    float radius = min(uMaximumDistance,max(0.5,32.0*uEpsilon));
    float occlusion = 0.0, total = 0.0, weight = 1.0;
    for (int i=0;i<16;++i) {
        if (i>=count) break;
        float reach = radius*float(i+1)/float(count);
        float field = sceneDistance(point+normal*reach);
        if (isinf(field) || isnan(field)) return 0.0;
        occlusion += weight*clamp(1.0-field/reach,0.0,1.0);
        total += weight; weight *= 0.75;
    }
    return clamp(1.0-occlusion/total,0.0,1.0);
}
bool traceSdf(vec3 direction, out float travel) {
    vec3 origin = vec3(0.0,0.0,3.0);
    travel = 0.0;
    for (int step = 0; step < 512; ++step) {
        if (step >= uMaximumSteps || travel > uMaximumDistance) break;
        float field = sceneDistance(origin + direction * travel);
        if (isinf(field) || isnan(field)) break;
        float threshold = max(uEpsilon * qualityScale(uAdaptiveQuality) * max(1.0, travel * 0.05), 0.000001);
        if (field <= threshold) return true; travel += field;
    }
    return false;
}
float sceneEdgeDistance(vec2 uv) {
    // Bounded eight-direction silhouette distance in output pixels, capped at
    // eight pixels. Four bisections resolve a hit/miss bracket to half a pixel.
    const vec2 axes[8] = vec2[8](vec2(1,0),vec2(-1,0),vec2(0,1),vec2(0,-1),
        vec2(0.70710678,0.70710678),vec2(-0.70710678,0.70710678),
        vec2(0.70710678,-0.70710678),vec2(-0.70710678,-0.70710678));
    float distance = 8.0;
    for (int i=0;i<8;++i) {
        float low = 0.0, high = 8.0, travel;
        vec2 axis = axes[i]*2.0/uExtent.y;
        if (traceSdf(normalize(vec3(uv+axis*high,-1.8)),travel)) continue;
        for (int j=0;j<4;++j) {
            float middle = (low+high)*0.5;
            if (traceSdf(normalize(vec3(uv+axis*middle,-1.8)),travel)) low = middle;
            else high = middle;
        }
        distance = min(distance,high);
    }
    return distance/8.0;
}
void main() {
    vec2 uv = (2.0 * gl_FragCoord.xy - uExtent) / uExtent.y;
    vec3 origin = vec3(0.0,0.0,3.0), direction = normalize(vec3(uv,-1.8));
    float travel;
    bool hit = traceSdf(direction,travel);
    if (!hit) {
        outColor = uOutputPass >= 3 ? vec4(0.0,0.0,0.0,1.0)
            : (uOutputPass == 1 ? vec4(1.0) : vec4(0.02745,0.03922,0.07059,1.0));
        return;
    }
    if (uOutputPass == 1) { outColor = vec4(vec3(clamp(travel / uMaximumDistance,0.0,1.0)),1.0); return; }
    if (uOutputPass == 7) { outColor = vec4(vec3(sceneEdgeDistance(uv)),1.0); return; }
    vec3 point = origin + direction * travel;
    if (uOutputPass == 3) {
        int contributor;
        sceneDistanceWithMaterial(point,contributor);
        int code = contributor < 0 ? 0 : int(uRecords[contributor].w);
        outColor = vec4(vec3(code&255,(code>>8)&255,(code>>16)&255)/255.0,1.0); return;
    }
    if (uOutputPass == 4) { outColor = vec4(vec3(sceneCurvature(point)),1.0); return; }
    vec3 normal = sceneNormal(point);
    if (uOutputPass == 2) { outColor = vec4(normal * 0.5 + 0.5,1.0); return; }
    if (uOutputPass == 5) { outColor = vec4(vec3(sceneAmbientVisibility(point,normal)),1.0); return; }
    vec3 light = normalize(vec3(-0.45,0.75,0.6));
    if (uOutputPass == 6) {
        float visibility = dot(normal,light) <= 0.0 ? 0.0 : sceneShadow(point+normal*uEpsilon*4.0,light);
        outColor = vec4(vec3(visibility),1.0); return;
    }
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
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::Rgba16Float, texture_, texture_, width_, height_,
                 1, true, contextIdentity_, lifecycle_.value,
                 colortransform::ColorSpace::LinearSRGB, colortransform::TransferFunction::Linear,
                 NativeTextureRowOrder::BottomFirst };
    }
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
    std::uintptr_t contextIdentity_ = 0;
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
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::materialId)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::curvature)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::ambientOcclusion)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::softShadow)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::edgeDistance)] = true;
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
            || ! available.supports (request.output))
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
        int previousPolygonMode[2] = {}; // GL_POLYGON_MODE returns front and back modes.
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
        glGetIntegerv (GL_POLYGON_MODE, previousPolygonMode);
        glGetBooleanv (GL_COLOR_WRITEMASK, previousColorMask);

        unsigned vertexArray = 0;
        auto frame = std::make_shared<OpenGlSdfSceneFrame>();
        frame->width_ = request.width;
        frame->height_ = request.height;
        frame->contextIdentity_ = reinterpret_cast<std::uintptr_t>(glfwGetCurrentContext());

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
            + 4u * sizeof (float) + 6u * sizeof (int);
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
                           static_cast<unsigned> (previousPolygonMode[0]));
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
            // All 24 color bits fit exactly in a float. The shader does not
            // consume parameterCount; admission already checked the schema.
            gpuRecords[offset + 3] = static_cast<float> (
                videohelper::sdf::nativeSdfMaterialColorCode (record.stableId));
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

const char* kOpticalFlowFragment = R"glsl(#version 330 core
layout(location=0) out vec2 outFlow;
uniform sampler2D uFirst;
uniform sampler2D uSecond;
uniform vec2 uExtent;
float luma(vec4 c){ return dot(c.rgb,vec3(0.2126,0.7152,0.0722)); }
ivec2 bounded(ivec2 p){ return clamp(p,ivec2(0),ivec2(uExtent)-ivec2(1)); }
void main(){
    ivec2 g=ivec2(gl_FragCoord.xy); float best=3.402823466e38; ivec2 bestD=ivec2(0); int bestMag=0;
    for(int dy=-4;dy<=4;++dy) for(int dx=-4;dx<=4;++dx){
        float score=0.0;
        for(int py=-1;py<=1;++py) for(int px=-1;px<=1;++px){
            ivec2 a=bounded(g+ivec2(px,py)); ivec2 b=bounded(g+ivec2(px+dx,py+dy));
            float d=luma(texelFetch(uFirst,a,0))-luma(texelFetch(uSecond,b,0)); score+=d*d;
        }
        int mag=dx*dx+dy*dy; bool tie=score==best && (mag<bestMag || (mag==bestMag && (dy<bestD.y || (dy==bestD.y && dx<bestD.x))));
        if(score<best || tie){best=score;bestD=ivec2(dx,dy);bestMag=mag;}
    }
    outFlow=vec2(bestD);
})glsl";

class OpenGlOpticalFlowBackend final : public NativeOpticalFlowExecutionBackend
{
public:
    videoopticalflow::BackendCapabilities opticalFlowCapabilities() const override
    {
        videoopticalflow::BackendCapabilities result;
        if (glfwGetCurrentContext()==nullptr) return result;
        arbitgl::GlFuncs gl; std::string ignored; if(!loadCurrentGl(gl,ignored)) return result;
        result.kind=videoopticalflow::BackendKind::NativeGpu; result.supportsOpticalFlow=true;
        result.implementation={'G','L','B','l','o','c','k','F','l','o','w','0','0','0','0','1'};
        result.implementationRevision=1; result.temporaryBytesPerPixel=0; return result;
    }
    NativeOpticalFlowSubmission executeOpticalFlow (
        const videoopticalflow::AdmittedRequest& request,
        const NativeOpticalFlowInputResource& first,
        const NativeOpticalFlowInputResource& second,
        bool) override
    {
        NativeOpticalFlowSubmission result;
        std::lock_guard<std::mutex> lock(mutex_); auto* context=glfwGetCurrentContext();
        if(context==nullptr){result.error="OpenGL optical-flow requires a current context";return result;}
        const auto caps=opticalFlowCapabilities();
        if(request.backendImplementation()!=caps.implementation || request.backendImplementationRevision()!=caps.implementationRevision){result.error="OpenGL optical-flow request was admitted for another backend revision";return result;}
        const auto& extent=request.output().extent;
        const bool exact=first.immutable&&second.immutable&&first.identity!=second.identity&&first.imageHandle&&second.imageHandle&&first.imageHandle!=second.imageHandle
            && first.imageHandle==first.textureViewHandle&&second.imageHandle==second.textureViewHandle&&first.descriptor.extent==extent&&second.descriptor.extent==extent
            && first.descriptor.format==renderpassoutput::PixelFormat::RGBA16Float&&second.descriptor.format==renderpassoutput::PixelFormat::RGBA16Float
            && first.imageHandle <= std::numeric_limits<unsigned>::max()
            && second.imageHandle <= std::numeric_limits<unsigned>::max()
            && first.helperGeneration != 0 && first.helperGeneration == second.helperGeneration
            && first.structuralRevision != 0 && first.structuralRevision == second.structuralRevision;
        if(!exact){result.error="OpenGL optical-flow inputs violate exact immutable RGBA16F bindings";return result;}
        arbitgl::GlFuncs gl; if(!loadCurrentGl(gl,result.error)) return result; if(!ensureProgram(context,gl,result.error)) return result;
        collectRetired(context);
        const State saved(gl);
        gl.ActiveTexture(GL_TEXTURE0);
        for (const auto* source : {&first, &second})
        {
            const auto texture = static_cast<unsigned>(source->textureViewHandle);
            if (glIsTexture(texture) != GL_TRUE)
            { result.error = "OpenGL optical-flow source texture is unavailable in this context"; return result; }
            glBindTexture(GL_TEXTURE_2D, texture);
            int width = 0, height = 0, format = 0;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
            if (width != static_cast<int>(extent.width) || height != static_cast<int>(extent.height)
                || format != GL_RGBA16F)
            { result.error = "OpenGL optical-flow source storage does not match its admitted descriptor"; return result; }
        }
        unsigned output=0,fbo=0; glGenTextures(1,&output); glBindTexture(GL_TEXTURE_2D,output); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RG16F,(int)extent.width,(int)extent.height,0,GL_RG,GL_HALF_FLOAT,nullptr);
        gl.GenFramebuffers(1,&fbo); gl.BindFramebuffer(GL_FRAMEBUFFER,fbo); gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,output,0);
        if(gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){gl.DeleteFramebuffers(1,&fbo);glDeleteTextures(1,&output);result.error="OpenGL optical-flow output framebuffer is incomplete";return result;}
        auto& p=programs_[context]; gl.UseProgram(p.program); gl.ActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,(unsigned)first.textureViewHandle); gl.Uniform1i(p.first,0); gl.ActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,(unsigned)second.textureViewHandle); gl.Uniform1i(p.second,1); gl.Uniform2f(p.extent,(float)extent.width,(float)extent.height);
        for (const auto capability : State::capabilities) glDisable(capability);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glViewport(0,0,(int)extent.width,(int)extent.height); gl.BindVertexArray(p.vao); glDrawArrays(GL_TRIANGLES,0,3); glFinish(); gl.DeleteFramebuffers(1,&fbo);
        do result.lifecycle.value=nextLifecycle_++; while(!result.lifecycle||outputs_.count(result.lifecycle.value)); outputs_[result.lifecycle.value]={context,output}; do result.submission=nextSubmission_++; while(result.submission==0);
        result.result.requestCacheKey=request.cacheKey(); result.result.backendImplementation=request.backendImplementation(); result.result.backendImplementationRevision=request.backendImplementationRevision(); result.result.motionVectors.helperGeneration=first.helperGeneration; result.result.motionVectors.descriptor=request.output(); result.result.motionVectors.byteCount=request.footprint().outputBytes; std::memcpy(result.result.motionVectors.identity.data(),&result.lifecycle.value,sizeof(result.lifecycle.value)); std::memcpy(result.result.motionVectors.identity.data()+sizeof(result.lifecycle.value),&result.submission,sizeof(result.submission)); result.imageHandle=output; result.textureViewHandle=output; result.completed=true;
        return result;
    }
    void releaseOpticalFlowOutput (NativeOpticalFlowOutputLifecycleHandle lifecycle) noexcept override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = outputs_.find(lifecycle.value);
        if (found == outputs_.end()) return;
        auto& output = found->second;
        if (output.texture == 0 || output.context == glfwGetCurrentContext())
        {
            if (output.texture != 0) glDeleteTextures(1, &output.texture);
            outputs_.erase(found);
        }
        else
            output.retired = true; // Drain on its render thread, never steal a live context.
    }
    void invalidateContext(std::uintptr_t identity) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* context = reinterpret_cast<GLFWwindow*>(identity);
        if (context == nullptr || glfwGetCurrentContext() != context) return;
        arbitgl::GlFuncs gl;
        std::string ignored;
        if (!loadCurrentGl(gl, ignored)) return;
        for (auto entry = outputs_.begin(); entry != outputs_.end();)
        {
            auto& output = entry->second;
            if (output.context != context) { ++entry; continue; }
            if (output.texture != 0) glDeleteTextures(1, &output.texture);
            output.texture = 0;
            output.context = nullptr;
            if (output.retired) entry = outputs_.erase(entry);
            else ++entry;
        }
        const auto program = programs_.find(context);
        if (program != programs_.end())
        {
            if (program->second.vao) gl.DeleteVertexArrays(1, &program->second.vao);
            if (program->second.program) gl.DeleteProgram(program->second.program);
            programs_.erase(program);
        }
    }
private:
    struct Program{unsigned program=0,vao=0;int first=-1,second=-1,extent=-1;};
    struct Output { GLFWwindow* context = nullptr; unsigned texture = 0; bool retired = false; };
    struct State final
    {
        inline static constexpr std::array<GLenum, 8> capabilities {
            GL_BLEND, GL_DEPTH_TEST, GL_STENCIL_TEST, GL_CULL_FACE, GL_SCISSOR_TEST,
            GL_RASTERIZER_DISCARD, GL_COLOR_LOGIC_OP, GL_FRAMEBUFFER_SRGB };
        const arbitgl::GlFuncs& gl;
        int drawFramebuffer = 0, readFramebuffer = 0, program = 0, vao = 0, activeTexture = 0;
        int texture[2] {}, viewport[4] {}, polygonMode[2] {};
        GLboolean colorMask[4] {};
        std::array<GLboolean, capabilities.size()> enabled {};
        explicit State(const arbitgl::GlFuncs& functions) : gl(functions)
        {
            glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFramebuffer);
            glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFramebuffer);
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
            for (int unit = 0; unit < 2; ++unit)
            {
                gl.ActiveTexture(GL_TEXTURE0 + unit);
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture[unit]);
            }
            glGetIntegerv(GL_VIEWPORT, viewport);
            glGetIntegerv(GL_POLYGON_MODE, polygonMode);
            glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
            for (std::size_t i = 0; i < capabilities.size(); ++i) enabled[i] = glIsEnabled(capabilities[i]);
        }
        ~State()
        {
            gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<unsigned>(drawFramebuffer));
            gl.BindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<unsigned>(readFramebuffer));
            gl.UseProgram(static_cast<unsigned>(program));
            gl.BindVertexArray(static_cast<unsigned>(vao));
            for (int unit = 0; unit < 2; ++unit)
            {
                gl.ActiveTexture(GL_TEXTURE0 + unit);
                glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(texture[unit]));
            }
            gl.ActiveTexture(static_cast<unsigned>(activeTexture));
            glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            glPolygonMode(GL_FRONT_AND_BACK, static_cast<unsigned>(polygonMode[0]));
            glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
            for (std::size_t i = 0; i < capabilities.size(); ++i)
                if (enabled[i]) glEnable(capabilities[i]); else glDisable(capabilities[i]);
        }
    };
    void collectRetired(GLFWwindow* context)
    {
        for (auto entry = outputs_.begin(); entry != outputs_.end();)
            if (entry->second.context == context && entry->second.retired)
            {
                if (entry->second.texture != 0) glDeleteTextures(1, &entry->second.texture);
                entry = outputs_.erase(entry);
            }
            else ++entry;
    }
    static bool loadCurrentGl(arbitgl::GlFuncs& gl,std::string& error){if(glfwGetCurrentContext()==nullptr){error="OpenGL optical-flow requires a current OpenGL 3.3 context";return false;}int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);if(major<3||(major==3&&minor<3)){error="OpenGL optical-flow requires OpenGL 3.3";return false;}return arbitgl::loadGlFunctions(gl,error);}
    bool ensureProgram(GLFWwindow* c,const arbitgl::GlFuncs& gl,std::string& error){if(programs_.count(c))return true; auto vs=compileShader(gl,GL_VERTEX_SHADER,kVertexShader,error);if(!vs)return false;auto fs=compileShader(gl,GL_FRAGMENT_SHADER,kOpticalFlowFragment,error);if(!fs){gl.DeleteShader(vs);return false;}Program p;p.program=gl.CreateProgram();gl.AttachShader(p.program,vs);gl.AttachShader(p.program,fs);gl.LinkProgram(p.program);gl.DeleteShader(vs);gl.DeleteShader(fs);int ok=0;gl.GetProgramiv(p.program,GL_LINK_STATUS,&ok);if(ok!=GL_TRUE){gl.DeleteProgram(p.program);error="OpenGL optical-flow program link failed";return false;}gl.GenVertexArrays(1,&p.vao);p.first=gl.GetUniformLocation(p.program,"uFirst");p.second=gl.GetUniformLocation(p.program,"uSecond");p.extent=gl.GetUniformLocation(p.program,"uExtent");programs_[c]=p;return true;}
    std::mutex mutex_;std::unordered_map<GLFWwindow*,Program> programs_;std::unordered_map<std::uint64_t,Output> outputs_;std::uint64_t nextLifecycle_=1,nextSubmission_=1;
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
        glGetIntegerv(GL_BLEND_SRC_RGB,&blendFactors[0]);
        glGetIntegerv(GL_BLEND_DST_RGB,&blendFactors[1]);
        glGetIntegerv(GL_BLEND_SRC_ALPHA,&blendFactors[2]);
        glGetIntegerv(GL_BLEND_DST_ALPHA,&blendFactors[3]);
        for (unsigned i = 0; i < attachmentBlends.size(); ++i)
            attachmentBlends[i] = gl.IsEnabledi(GL_BLEND, i) == GL_TRUE;
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
        gl.BlendFuncSeparate(blendFactors[0],blendFactors[1],blendFactors[2],blendFactors[3]);
        for (unsigned i = 0; i < attachmentBlends.size(); ++i)
            attachmentBlends[i] ? gl.Enablei(GL_BLEND, i) : gl.Disablei(GL_BLEND, i);
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
    std::array<bool, 8> attachmentBlends {};
    std::array<int, 4> blendFactors {};
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
layout(location=10) in vec4 aInstanceColor;
layout(location=11) in vec4 aInstanceEmission;
uniform mat4 uObjectMatrix;
uniform float uTangentHandednessSign;
uniform int uUseInstanceMatrix;
uniform vec4 uRawIdentifiers[64];
uniform int uRawObjectIndex;
flat out vec4 vRawIdentifiers;
uniform vec4 uCameraRotation;
uniform vec3 uCameraTranslation;
uniform vec4 uProjection;
uniform vec4 uBaseColor;
uniform vec4 uMotionRotation;
uniform vec4 uMotionTranslationScale;
uniform vec4 uPreviousCameraRotation;
uniform vec4 uPreviousCameraTranslation;
uniform vec4 uPreviousProjection;
out vec4 vMotionCurrentClip;
out vec4 vMotionPreviousClip;
uniform vec4 uMaterialParams;
uniform vec4 uTimeMixEndColorAndTime;
uniform vec4 uVertexTime;
uniform vec4 uVertexSpectrum[16];
uniform vec3 uEmissive;
uniform vec3 uAmbient;
uniform vec4 uNoteInstanceTransforms[128];
uniform int uNoteInstanceIndex;
uniform float uNoteMeshScale;
uniform vec4 uNoteAppearanceLow;
uniform vec4 uNoteAppearanceHigh;
flat out vec4 vNoteAppearance;
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
    vRawIdentifiers = uRawIdentifiers[uUseInstanceMatrix != 0 ? gl_InstanceID : uRawObjectIndex];
    mat4 objectMatrix = uUseInstanceMatrix != 0
        ? mat4(aInstanceMatrix0, aInstanceMatrix1,
               aInstanceMatrix2, aInstanceMatrix3)
        : uObjectMatrix;
    int noteIndex = uNoteInstanceIndex >= 0 ? uNoteInstanceIndex : gl_InstanceID;
    vec4 noteInstance = uNoteInstanceTransforms[noteIndex];
    vNoteAppearance = mix(uNoteAppearanceLow, uNoteAppearanceHigh, noteInstance.w);
    // ARBIT_VERTEX_MODIFIER
    vec3 world = (objectMatrix
        * vec4(modifiedPosition * uNoteMeshScale + noteInstance.xyz, 1.0)).xyz;
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
    vMotionCurrentClip = gl_Position;
    vec3 previousWorld = rotateQ(uMotionRotation, world) * uMotionTranslationScale.w
        + uMotionTranslationScale.xyz;
    vec3 previousCamera = rotateQ(vec4(-uPreviousCameraRotation.xyz, uPreviousCameraRotation.w),
        previousWorld - uPreviousCameraTranslation.xyz);
    vMotionPreviousClip = uPreviousProjection.w > 0.5 && -previousCamera.z > uPreviousProjection.z
        ? vec4(previousCamera.x / (uPreviousProjection.x * uPreviousProjection.y),
               previousCamera.y / uPreviousProjection.x, 0.0, -previousCamera.z)
        : vMotionCurrentClip;
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
    if (uUseInstanceMatrix != 0) vBaseColor *= aInstanceColor.rgb;
    vLitBase = vBaseColor * uAmbient;
    vEmissive = uEmissive + (uUseInstanceMatrix != 0 ? aInstanceEmission.rgb : vec3(0.0));
    vWorldPosition = world;
    vWorldNormal = worldNormal;
    vWorldTangent = normalize(mat3(objectMatrix) * aTangent.xyz);
    vBitangentSign = aTangent.w * uTangentHandednessSign
        * (uUseInstanceMatrix != 0 ? sign(determinant(mat3(objectMatrix))) : 1.0);
    vMetallicRoughness = vec2(metallic, roughness);
    vLinearDepth = clamp((distance - uProjection.z)
        / max(uProjection.w - uProjection.z, 0.000001), 0.0, 1.0);
    vOpacity = uBaseColor.w * aColor.a * (uUseInstanceMatrix != 0 ? aInstanceColor.a : 1.0);
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
flat in vec4 vNoteAppearance;
flat in vec4 vRawIdentifiers;
in vec4 vMotionCurrentClip;
in vec4 vMotionPreviousClip;
uniform vec4 uPassProgramControl;
uniform vec4 uPassModes[8];
uniform vec4 uPassColors[8];
uniform vec4 uPassReferences[8];
uniform vec4 uPassTransforms[8];
uniform vec4 uMotionViewport;
uniform sampler2D uBaseTexture;
uniform sampler2D uMetallicRoughnessTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uOcclusionTexture;
uniform sampler2D uEmissiveTexture;
uniform vec4 uTexturePresence;
uniform float uEmissiveTexturePresence;
uniform vec2 uAlphaModeCutoff;
uniform vec4 uImageOutput;
uniform vec4 uMaterialParams;
uniform vec3 uAmbient;
uniform vec4 uSurfaceCoating;
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
uniform vec4 uDiffractionEnvironment;
layout(location=0) out vec4 outColor;
layout(location=1) out float outLinearDepth;
layout(location=2) out vec4 outRawNormal;
layout(location=3) out vec4 outRawEmission;
layout(location=4) out float outRawMask;
layout(location=5) out uint outRawMaterialId;
layout(location=6) out uint outRawObjectId;
layout(location=7) out vec2 outRawMotion;

vec4 compositePass(vec4 color) {
    int count = int(uPassProgramControl.x);
    if (count == 0) return color;
    vec4 values[8];
    for (int i = 0; i < 8; ++i) {
        if (i >= count) break;
        vec4 p = uPassModes[i];
        int mode = int(p.x);
        float depth = clamp((outLinearDepth - p.y) / (p.z - p.y), 0.0, 1.0);
        vec3 fog = uPassColors[i].rgb;
        if (uPassTransforms[i].y > 0.5)
            fog = mix(pow((fog + 0.055) / 1.055, vec3(2.4)), fog / 12.92,
                lessThanEqual(fog, vec3(0.04045)));
        uint identity = uint(uPassReferences[i].x) | (uint(uPassReferences[i].y) << 16u);
        vec4 value = vec4(color.rgb, outRawMask);
        if (mode == 1) value = vec4(mix(color.rgb, fog, depth * p.w), outRawMask);
        float inspectionAlpha = uPassTransforms[i].w > 0.5 ? 1.0 : outRawMask;
        if (mode == 2) value = vec4(vec3(uPassTransforms[i].w > 1.5 ? 1.0 - depth : depth), inspectionAlpha);
        if (mode == 3) value = vec4(outRawNormal.xyz * 0.5 + 0.5, inspectionAlpha);
        if (mode == 4) value = vec4(outRawEmission.rgb, outRawMask);
        if (mode >= 5 && mode <= 7) {
            float matte = outRawMask;
            if (mode == 6) matte *= float(outRawMaterialId == identity);
            if (mode == 7) matte *= float(outRawObjectId == identity);
            value = vec4(vec3(matte), 1.0);
        }
        if (mode == 8) value = vec4(clamp(vec2(0.5) + outRawMotion / uMotionViewport.xy
            * (1.0 + 63.0 * p.w), 0.0, 1.0), 0.5, outRawMask);
        if (mode == 9 || mode == 10) {
            vec4 a = values[int(uPassReferences[i].z) - 1];
            vec4 b = values[int(uPassReferences[i].w) - 1];
            value = mode == 9 ? mix(a, b, p.w)
                : vec4(a.rgb + b.rgb * p.w, max(a.a, b.a));
        }
        value.rgb *= exp2(uPassTransforms[i].x);
        values[i] = value;
    }
    int selected = int(uPassProgramControl.y);
    vec4 value = values[selected];
    int transform = int(uPassTransforms[selected].z);
    if (transform > 0 && uPassProgramControl.z < 0.5) {
        vec3 rgb = max(value.rgb, vec3(0.0));
        if (transform == 2) rgb /= vec3(1.0) + rgb;
        value.rgb = mix(1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055, rgb * 12.92,
            lessThanEqual(rgb, vec3(0.0031308)));
    }
    return uPassProgramControl.z > 0.5 ? value : clamp(value, 0.0, 1.0);
}

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

// Integrate discrete reciprocal orders. The environment is affine on the unit
// sphere. A five-point angular rule follows the admitted roughness width.
float diffractionEnvironmentSample(vec3 direction, vec3 position) {
    bool hit = false;
    if (direction.y < -1.0e-6 && position.y > uDiffractionEnvironment.y) {
        float distance = (uDiffractionEnvironment.y - position.y) / direction.y;
        hit = distance > 0.0 && distance <= uDiffractionEnvironment.z;
    }
    vec3 axis = uDiffractionIncidentDirectionAndIntensity.xyz;
    if (uDiffractionPathKindAndBounce.x == 3.0)
        return hit ? 0.625 + 0.25 * axis.y : 0.0;
    return hit ? 0.0 : 0.625 + 0.375 * dot(axis, direction);
}

vec3 integrateDiffractionEnvironment(vec3 outgoing, vec3 tangent,
                                    vec3 bitangent, vec3 normal) {
    vec4 geometry;
    vec4 secondaryGeometry;
    resolveGrooveGeometry(vUv, geometry, secondaryGeometry);
    bool crossed = int(secondaryGeometry.w + 0.5) == 2;
    int firstOrder = int(round(uDiffractionMicrostructure.w));
    int lastOrder = int(uDiffractionControl.x + 0.5);
    int profile = int(uDiffractionControl.y + 0.5);
    float duty = uDiffractionMicrostructure.x;
    float sigma = max(2.0 * uDiffractionRoughness.y, 1.0e-4);
    vec3 xyz = vec3(0.0);
    float whiteY = 0.0;
    for (int wavelengthIndex = 0; wavelengthIndex < 8; ++wavelengthIndex) {
        vec4 spectral = uDiffractionSpectral[wavelengthIndex];
        vec4 spectralZ = uDiffractionSpectralZ[wavelengthIndex];
        float wavelength = spectral.x;
        float radiance = 0.0;
        float totalEfficiency = 0.0;
        for (int primary = -lastOrder; primary <= lastOrder; ++primary) {
            if (primary != 0 && abs(primary) < firstOrder) continue;
            int secondaryLimit = crossed ? lastOrder : 0;
            for (int secondary = -secondaryLimit; secondary <= secondaryLimit; ++secondary) {
                if (secondary != 0 && abs(secondary) < firstOrder) continue;
                vec2 xy = -outgoing.xy + float(primary) * wavelength / geometry.z * geometry.xy;
                if (crossed)
                    xy += float(secondary) * wavelength / secondaryGeometry.z * secondaryGeometry.xy;
                if (dot(xy, xy) >= 1.0) continue;
                float z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
                float phase = 2.0 * pi * geometry.w * (outgoing.z + z) / wavelength;
                vec2 terrace = vec2(cos(phase), sin(phase));
                float efficiency = profileEfficiency(profile, primary, duty, phase, terrace);
                if (crossed) efficiency *= profileEfficiency(profile, secondary, duty, phase, terrace);
                if (primary != 0 || secondary != 0) efficiency *= z / outgoing.z;
                float height = 2.0 * pi * uDiffractionRoughness.x * (outgoing.z + z) / wavelength;
                efficiency *= exp(-height * height);
                vec3 direction = xy.x * tangent + xy.y * bitangent + z * normal;
                vec3 side = normalize(cross(abs(direction.y) < 0.9
                    ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0), direction));
                vec3 up = cross(direction, side);
                float lighting = 0.5 * diffractionEnvironmentSample(direction, vWorldPosition);
                lighting += 0.125 * diffractionEnvironmentSample(normalize(direction + sigma * side), vWorldPosition);
                lighting += 0.125 * diffractionEnvironmentSample(normalize(direction - sigma * side), vWorldPosition);
                lighting += 0.125 * diffractionEnvironmentSample(normalize(direction + sigma * up), vWorldPosition);
                lighting += 0.125 * diffractionEnvironmentSample(normalize(direction - sigma * up), vWorldPosition);
                radiance += efficiency * lighting;
                totalEfficiency += efficiency;
            }
        }
        float reflectance = spectralReflectance(wavelength, int(uDiffractionControl.w + 0.5),
            uDiffractionMicrostructure.y, uDiffractionMicrostructure.z);
        float energy = uDiffractionIncidentDirectionAndIntensity.w * spectral.y
            * reflectance * spectralZ.y * radiance / max(1.0, totalEfficiency);
        xyz += energy * vec3(spectral.z, spectral.w, spectralZ.x);
        whiteY += spectralZ.y * spectral.w;
    }
    xyz /= max(whiteY, 1.0e-8);
    return vec3(3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
                -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
                0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z);
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
    if (outgoing.z <= 0.0)
        return vec3(0.0);
    if (uDiffractionEnvironment.x == 2.0 && uDiffractionPathKindAndBounce.x != 1.0)
        return integrateDiffractionEnvironment(outgoing, tangent, bitangent, normal);
    if (uDiffractionEnvironment.x == 2.0) {
        if (uLightCount == 0) return vec3(0.0);
        vec3 toLight = normalize(rotateQ(uLightRotations[0], vec3(0.0, 0.0, 1.0)));
        incident = vec3(dot(toLight, tangent), dot(toLight, bitangent), dot(toLight, normal));
    }
    if (incident.z <= 1.0e-6) return vec3(0.0);

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
        if (uDiffractionEnvironment.x == 2.0) energy *= uLightColors[0].w;
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
    // Only graph Frames may be native scene attachments. Embedded glTF images
    // keep their authored UVs and raster storage; w=2 declares a bottom-first Frame.
    vec2 baseUv = uTexturePresence.w > 1.5 ? vec2(vUv.x, 1.0 - vUv.y) : vUv;
    outRawNormal = vec4(normalize(vWorldNormal), 1.0);
    outRawEmission = vec4(0.0);
    outRawMask = 1.0;
    outRawMaterialId = uint(vRawIdentifiers.x) | (uint(vRawIdentifiers.y) << 16u);
    outRawObjectId = uint(vRawIdentifiers.z) | (uint(vRawIdentifiers.w) << 16u);
    outRawMotion = clamp((vMotionCurrentClip.xy / vMotionCurrentClip.w
        - vMotionPreviousClip.xy / vMotionPreviousClip.w) * 0.5 * uMotionViewport.xy,
        vec2(-65504.0), vec2(65504.0));
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
        if (uTexturePresence.w > 0.5) {
            vec4 imagery = texture(uBaseTexture, baseUv);
            if (imagery.a <= 0.0) discard;
            outColor *= vec4(srgbToLinear(imagery.rgb), imagery.a);
        }
        outColor.rgb = vNoteAppearance.rgb * (outColor.rgb + vec3(vNoteAppearance.a));
        outColor = compositePass(outColor);
        return;
    }
    vec4 texel = texture(uBaseTexture, baseUv);
    texel.rgb = srgbToLinear(texel.rgb);
    // Thin-surface, normal-incidence dielectric model. Transmission reveals
    // the composited scene behind the surface; it is not volume refraction.
    float interfaceRatio = (uSurfaceCoating.y - 1.0) / (uSurfaceCoating.y + 1.0);
    float dielectric = uSurfaceCoating.w > 0.5 ? interfaceRatio * interfaceRatio : 0.04;
    float transmission = uSurfaceCoating.w > 0.5 ? uSurfaceCoating.x : 0.0;
    float coat = uSurfaceCoating.w > 0.5 ? 0.04 * uSurfaceCoating.z : 0.0;
    float coverage = 1.0 - transmission * (1.0 - dielectric);
    float alpha = vOpacity * texel.a * coverage;
    if (alpha <= 0.0 || (uAlphaModeCutoff.x == 1.0 && alpha < uAlphaModeCutoff.y))
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
    vec3 emissive = vEmissive * (uEmissiveTexturePresence > 0.5
        ? srgbToLinear(texture(uEmissiveTexture, vUv).rgb) : vec3(1.0));
    outRawNormal = vec4(normal, 1.0);
    outRawEmission = vec4(vNoteAppearance.rgb * (emissive + vec3(vNoteAppearance.a)), 1.0);
    outRawMask = clamp(alpha, 0.0, 1.0);
    int imageOutput = int(uImageOutput.x);
    if (imageOutput != 0) {
        vec3 image = uImageOutput.yzw;
        if (imageOutput == 1) image = vec3(vLinearDepth);
        if (imageOutput == 2) image = normal * 0.5 + 0.5;
        if (imageOutput == 3) image = vec3(clamp(vec2(0.5) + outRawMotion / uMotionViewport.xy, 0.0, 1.0), 0.5);
        if (imageOutput == 4) image = vNoteAppearance.rgb * (emissive + vec3(vNoteAppearance.a));
        if (imageOutput == 5) image = vec3(1.0);
        if (imageOutput == 6 || imageOutput == 7) {
            uint hash = (imageOutput == 6 ? outRawMaterialId : outRawObjectId) * 0x9e3779b9u;
            hash ^= hash >> 16u;
            image = vec3(32u + (hash & 191u), 32u + ((hash >> 8u) & 191u),
                         32u + ((hash >> 16u) & 191u)) / 255.0;
        }
        outColor = vec4(image, alpha);
        return;
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
    float diffuseWeight = (1.0 - metallic) * (1.0 - 0.5 * roughness) * (1.0 - transmission);
    float specularWeight = mix(dielectric, 1.0, metallic) * (1.0 - roughness);
    float occlusion = uTexturePresence.z > 0.5
        ? texture(uOcclusionTexture, vUv).r : 1.0;
    outColor = vec4((vBaseColor * texel.rgb * lighting
                    * (diffuseWeight + specularWeight) * (1.0 - coat) * occlusion
                    + lighting * coat + emissive) / max(coverage, 0.000001), alpha);
    outColor.rgb = vNoteAppearance.rgb * (outColor.rgb + vec3(vNoteAppearance.a));
    if (vInstanceIdentityColor.a > 0.5)
        outColor = vec4(vInstanceIdentityColor.rgb, 1.0);
    outColor = compositePass(outColor);
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
    int surfaceCoating = -1;
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
    int imageOutput = -1;
    int rawIdentifiers = -1, rawObjectIndex = -1;
    int passProgramControl = -1, passModes = -1, passColors = -1, passReferences = -1, passTransforms = -1;
    int motionRotation = -1, motionTranslationScale = -1;
    int previousCameraRotation = -1, previousCameraTranslation = -1, previousProjection = -1;
    int motionViewport = -1;
    int vertexTime = -1, vertexSpectrum = -1;
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
    int diffractionEnvironment = -1;
    int noteInstanceTransforms = -1;
    int noteInstanceIndex = -1;
    int noteMeshScale = -1;
    int noteAppearanceLow = -1;
    int noteAppearanceHigh = -1;

    bool complete() const noexcept
    {
        return objectMatrix >= 0 && tangentHandednessSign >= 0
            && useInstanceMatrix >= 0
            && cameraRotation >= 0 && cameraTranslation >= 0 && projection >= 0
            && baseColor >= 0 && materialParams >= 0 && surfaceCoating >= 0
            && timeMixEndColorAndTime >= 0 && emissive >= 0 && ambient >= 0
            && baseTexture >= 0 && lightCount >= 0
            && metallicRoughnessTexture >= 0 && normalTexture >= 0
            && occlusionTexture >= 0 && emissiveTexture >= 0
            && texturePresence >= 0 && emissiveTexturePresence >= 0
            && alphaModeCutoff >= 0 && imageOutput >= 0
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
            && diffractionEvaluationSchedule >= 0 && diffractionEnvironment >= 0
            && noteInstanceTransforms >= 0
            && noteInstanceIndex >= 0 && noteMeshScale >= 0
            && noteAppearanceLow >= 0 && noteAppearanceHigh >= 0;
    }
};

class OpenGlFixtureSceneResources final : public NativeFixtureSceneResources
{
public:
    struct SharedProgram final
    {
        GLFWwindow* context = nullptr;
        arbitgl::GlFuncs gl {};
        unsigned id = 0;
        ~SharedProgram()
        {
            // Scene resources and the context cache release their program leases
            // under fixtureMutex while this program's context is current.
            if (id != 0 && context != nullptr && glfwGetCurrentContext()==context)
                gl.DeleteProgram(id);
        }
    };
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
        sharedProgram.reset();
    }

    std::string backend_ = "opengl";
    GLFWwindow* ownerContext = nullptr;
    arbitgl::GlFuncs gl {};
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> snapshot;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> materialProgram;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> materialProgramSource;
    // Identifies one prepared static-resource lifetime. Frames rendered from the
    // same preparation retain this value; preparing again allocates a new one.
    std::uint64_t rendererGeneration = 0;
    unsigned vertexArray = 0;
    unsigned vertexBuffer = 0;
    unsigned indexBuffer = 0;
    unsigned instanceBuffer = 0;
    bool instancedSharedGeometry = false;
    bool diagnosticInstanceIdentityColors = false;
    std::vector<videowire::geometry::InstanceAppearance> instanceAppearances;
    std::vector<unsigned> textures;
    unsigned program = 0;
    std::shared_ptr<SharedProgram> sharedProgram;
    OpenGlFixtureUniformLocations uniforms {};
};

class OpenGlFixtureSceneFrame final : public NativeFixtureSceneFrame
{
public:
    ~OpenGlFixtureSceneFrame() override
    {
        if (ownerContext == nullptr)
            return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        auto* previous = glfwGetCurrentContext();
        if (previous != ownerContext)
            glfwMakeContextCurrent (ownerContext);
        if (glfwGetCurrentContext() == ownerContext)
        {
            if (depthTexture_ != 0) glDeleteTextures (1, &depthTexture_);
            glDeleteTextures(static_cast<int>(rawTextures.size()), rawTextures.data());
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
                 linearColor ? NativeTexturePixelFormat::Rgba16Float : NativeTexturePixelFormat::Rgba8Unorm,
                 texture_, texture_, width_, height_, 1,
                 true, reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_,
                 linearColor ? colortransform::ColorSpace::LinearSRGB : colortransform::ColorSpace::Unspecified,
                 linearColor ? colortransform::TransferFunction::Linear : colortransform::TransferFunction::Unspecified,
                 NativeTextureRowOrder::BottomFirst };
    }
    NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::R32Float, depthTexture_, depthTexture_, width_, height_, 1,
                 true, reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_,
                 colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
                 NativeTextureRowOrder::BottomFirst };
    }

    NativeTextureViewDescriptor passTextureDescriptor(renderpassoutput::Output output) const noexcept override
    {
        for (std::size_t i = 2; i < renderpasscomposite::kInputs.size(); ++i)
            if (renderpasscomposite::kInputs[i] == output)
                return { backend_, NativeTextureViewKind::Texture2D, rawFormats[i - 2],
                    rawTextures[i - 2], rawTextures[i - 2], width_, height_, 1, true,
                    reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_,
                    colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
                    NativeTextureRowOrder::BottomFirst };
        return NativeFixtureSceneFrame::passTextureDescriptor(output);
    }

    std::string backend_ = "opengl";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    GLFWwindow* ownerContext = nullptr;
    std::shared_ptr<const OpenGlFixtureSceneResources> staticResources;
    bool linearColor = false;
    std::shared_ptr<const NativeFixtureSceneFrame> materialFrameTexture;
    std::map<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint,
        std::shared_ptr<const NativeFixtureSceneFrame>> materialFrameTextures;
    std::uint64_t rendererGeneration_ = 0;
    unsigned texture_ = 0;
    unsigned depthTexture_ = 0;
    bool readRawPass(renderpassoutput::Output output, NativeRawPassPixels& pixels,
                     std::string& error) const override
    {
        pixels = {};
        std::lock_guard<std::mutex> lock(fixtureMutex());
        const auto descriptor = passTextureDescriptor(output);
        if (!descriptor.complete() || !rawPassFormatMatches(output, descriptor.format)
            || !nativeFixtureDimensionsWithinBounds(width_, height_)
            || !staticResources || glfwGetCurrentContext() != ownerContext)
        { error = "OpenGL raw pass requires its live owning context and supported attachment"; return false; }
        GLenum layout = GL_RED, type = GL_UNSIGNED_BYTE;
        switch (descriptor.format)
        {
            case NativeTexturePixelFormat::Rgba8Unorm: layout = GL_RGBA; break;
            case NativeTexturePixelFormat::Rgba16Float: layout = GL_RGBA; type = GL_HALF_FLOAT; break;
            case NativeTexturePixelFormat::Rg16Float: layout = GL_RG; type = GL_HALF_FLOAT; break;
            case NativeTexturePixelFormat::R32Float: type = GL_FLOAT; break;
            case NativeTexturePixelFormat::R32Uint: layout = GL_RED_INTEGER; type = GL_UNSIGNED_INT; break;
            case NativeTexturePixelFormat::R8Unorm: break;
            default: error = "OpenGL raw attachment format is unsupported"; return false;
        }
        if (glGetError() != GL_NO_ERROR)
        { error = "OpenGL raw pass encountered an earlier GPU error"; return false; }
        const auto rowBytes = static_cast<std::size_t>(width_) * rawPassChannels(descriptor.format)
            * rawPassScalarBytes(descriptor.format);
        NativeRawPassPixels candidate { descriptor.format, width_, height_,
            std::vector<std::uint8_t>(rowBytes * height_) };
        GLint texture = 0, buffer = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture);
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &buffer);
        constexpr std::array<GLenum, 6> packNames {
            GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS,
            GL_PACK_SKIP_ROWS, GL_PACK_SWAP_BYTES, GL_PACK_LSB_FIRST };
        std::array<GLint, 6> packValues {};
        for (std::size_t i = 0; i < packNames.size(); ++i)
        {
            glGetIntegerv(packNames[i], &packValues[i]);
            glPixelStorei(packNames[i], i == 0 ? 1 : 0);
        }
        staticResources->gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(descriptor.imageHandle));
        glGetTexImage(GL_TEXTURE_2D, 0, layout, type, candidate.bytes.data());
        const auto result = glGetError();
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
        staticResources->gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(buffer));
        for (std::size_t i = 0; i < packNames.size(); ++i) glPixelStorei(packNames[i], packValues[i]);
        if (result != GL_NO_ERROR)
        { error = "OpenGL raw pass readback failed"; return false; }
        for (std::uint32_t y = 0; y < height_ / 2; ++y)
            std::swap_ranges(candidate.bytes.data() + y * rowBytes,
                candidate.bytes.data() + (y + 1) * rowBytes,
                candidate.bytes.data() + (height_ - 1 - y) * rowBytes);
        pixels = std::move(candidate);
        error.clear();
        return true;
    }

    std::array<unsigned, 6> rawTextures {};
    inline static constexpr std::array<NativeTexturePixelFormat, 6> rawFormats {
        NativeTexturePixelFormat::Rgba16Float, NativeTexturePixelFormat::Rgba16Float,
        NativeTexturePixelFormat::R8Unorm, NativeTexturePixelFormat::R32Uint,
        NativeTexturePixelFormat::R32Uint, NativeTexturePixelFormat::Rg16Float };
};

struct OpenGlInstanceGpuRecord final
{
    std::array<float, 16> matrix {};
    std::array<float, 4> identityColor {};
    std::array<float, 4> color {1,1,1,1};
    std::array<float, 4> emission {};
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
    void releaseCachedProgramsForCurrentContext() noexcept override
    {
        auto* context = glfwGetCurrentContext();
        if (context == nullptr) return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        programCache_.erase(context);
    }

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
        capabilities.typedFieldEvaluation = true;
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
        capabilities.maxFieldElements = HarmonicMIDI::grid::Visual3DScene::kMaxVertices;
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
        bool diagnosticInstanceIdentityColors,
        const std::vector<videowire::geometry::AttributeData>* frameAttributes = nullptr) override
    {
        using namespace HarmonicMIDI::grid;

        NativeFixtureScenePreparation result;
        const auto materialProgram = snapshotNativeSurfaceProgram(requestedMaterialProgram);
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
                && validNativeSurfaceObjectPrograms(*materialProgram, *scene, NativeFixtureMaterialBackend::OpenGl))
            || validNativeFixtureDiffractionProgram (
                *materialProgram, NativeFixtureMaterialBackend::OpenGl,
                scene->objects[0].id);
        if (materialProgram != nullptr
            && (materialProgram->layoutVersion
                    != NativeFixtureSurfaceMaterialProgram::kLayoutVersion
                || materialProgram->backend != NativeFixtureMaterialBackend::OpenGl
                || materialProgram->programIdentity.empty()
                || materialProgram->bindingDigest.empty()
                || (materialProgram->kind != NativeFixtureMaterialKind::SurfacePbr
                    && (materialProgram->object != scene->objects[0].id || !materialProgram->objectPrograms.empty()))
                || (materialProgram->vertexProgram
                    && (materialProgram->kind != NativeFixtureMaterialKind::SurfacePbr
                        || scene->objectCount != 1
                        || materialProgram->vertexProgram->programIdentity() != materialProgram->programIdentity))
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
        int maximumTextureExtent = 0;
        glGetIntegerv (GL_MAX_TEXTURE_SIZE, &maximumTextureExtent);
        std::size_t textureBytes = sizeof (SceneTexelRgba8);
        fixturetexture::SceneMipLayouts textureLayouts;
        if (maximumTextureExtent <= 0
            || ! fixturetexture::sceneMipLayouts (*scene, 1,
                static_cast<std::uint32_t> (maximumTextureExtent),
                fixturetexture::kMaximumTextureBytes, textureBytes, textureLayouts))
        {
            result.error = "OpenGL fixture texture bounds, sampler settings or allocation budget are invalid";
            return result;
        }
        const auto generateMipmap = reinterpret_cast<PFNGLGENERATEMIPMAPPROC> (
            glfwGetProcAddress ("glGenerateMipmap"));
        if (generateMipmap == nullptr && std::any_of (
                textureLayouts.begin(), textureLayouts.begin() + scene->textureCount,
                [] (const auto& layout) { return layout.levelCount > 1; }))
        {
            result.error = "OpenGL fixture textures require mipmap generation support";
            return result;
        }
        OpenGlFixtureState previous (gl);
        auto resources = std::make_shared<OpenGlFixtureSceneResources>();
        resources->ownerContext = glfwGetCurrentContext();
        resources->rendererGeneration = nextFixtureRendererGeneration();
        resources->gl = gl;
        resources->snapshot = scene;
        resources->materialProgram = materialProgram;
        resources->materialProgramSource = requestedMaterialProgram;
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
        if (geometryAdmission != nullptr && !admittedInstanceSetMatchesScene) {
            result.error = "OpenGL Geometry Core authority does not match the retained instance scene";
            return result;
        }
        if (frameAttributes && (!admittedInstanceSetMatchesScene
            || !videowire::geometry::validInstanceAppearanceAttributes(*frameAttributes,scene->objectCount,result.error)))
            return result;
        if (admittedInstanceSetMatchesScene)
            for (std::size_t index=0;index<scene->objectCount;++index)
                resources->instanceAppearances.push_back(videowire::geometry::instanceAppearance(
                    frameAttributes ? *frameAttributes : geometryAdmission->value().descriptor().attributes,index));
        resources->instancedSharedGeometry = admittedInstanceSetMatchesScene
            && scene->objectCount > 1
            && std::all_of(resources->instanceAppearances.begin(),resources->instanceAppearances.end(),
                [](const auto& appearance) {
                    return appearance.color[3]>=1.0f && appearance.metallic<0 && appearance.roughness<0;
                })
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
            for (unsigned attribute=10;attribute<=11;++attribute) {
                gl.VertexAttribPointer(attribute,4,GL_FLOAT,GL_FALSE,sizeof(OpenGlInstanceGpuRecord),
                    reinterpret_cast<const void*>(attribute==10 ? offsetof(OpenGlInstanceGpuRecord,color)
                                                               : offsetof(OpenGlInstanceGpuRecord,emission)));
                gl.EnableVertexAttribArray(attribute);
                gl.VertexAttribDivisor(attribute,1);
            }
        }

        const SceneTexelRgba8 whiteTexel { 255, 255, 255, 255 };
        // Upload each image once. The shader applies the sRGB transfer only at
        // base-color and emissive sampling sites, while data roles read the same
        // immutable RGBA8 storage without a color transform.
        try { resources->textures.resize (1 + scene->textureCount); }
        catch (const std::bad_alloc&) { return fail ("OpenGL fixture texture handle allocation failed"); }
        glGenTextures (static_cast<int> (resources->textures.size()), resources->textures.data());
        if (glGetError() != GL_NO_ERROR || std::any_of (
                resources->textures.begin(), resources->textures.end(),
                [] (unsigned texture) { return texture == 0; }))
            return fail ("OpenGL fixture texture allocation failed");
        gl.ActiveTexture (GL_TEXTURE0);
        gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei (GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei (GL_UNPACK_SKIP_PIXELS, 0);
        glPixelStorei (GL_UNPACK_SKIP_ROWS, 0);
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
            const auto levelCount = slot > 0 ? textureLayouts[slot - 1].levelCount : 1u;
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, static_cast<int> (levelCount - 1));
            glTexImage2D (GL_TEXTURE_2D, 0,
                          GL_RGBA8,
                          static_cast<int> (textureWidth), static_cast<int> (textureHeight),
                          0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            if (glGetError() != GL_NO_ERROR)
                return fail ("OpenGL fixture texture upload failed");
            if (levelCount > 1)
                generateMipmap (GL_TEXTURE_2D);
            if (glGetError() != GL_NO_ERROR)
                return fail ("OpenGL fixture mipmap allocation failed");
        }

        const auto vertexSource = fixtureVertexModifierShader (kFixtureVertexShader,
            materialProgram != nullptr ? materialProgram->vertexProgram.get() : nullptr,
            videohelper::materialprogram::BackendTarget::OpenGl, result.error);
        if (vertexSource.empty()) return fail (result.error);
        // Solid-body and harmonic-geometry frames can release every scene lease
        // before the next preparation. Retain only the immutable linked program
        // for the exact context and vertex source; the fragment source is fixed.
        auto& contextPrograms = programCache_[glfwGetCurrentContext()];
        const auto cached = contextPrograms.find(vertexSource);
        if (cached != contextPrograms.end())
        {
            resources->sharedProgram = cached->second.program;
            cached->second.lastUse = ++programUseSerial_;
        }
        if (!resources->sharedProgram)
        {
            vertexShader=compileFixtureShader(gl,GL_VERTEX_SHADER,vertexSource.c_str(),result.error);
            if(vertexShader==0) return fail(result.error);
            fragmentShader=compileFixtureShader(gl,GL_FRAGMENT_SHADER,kFixtureFragmentShader,result.error);
            if(fragmentShader==0) return fail(result.error);
            auto shared=std::make_shared<OpenGlFixtureSceneResources::SharedProgram>();
            shared->context=glfwGetCurrentContext(); shared->gl=gl; shared->id=gl.CreateProgram();
            gl.AttachShader(shared->id,vertexShader); gl.AttachShader(shared->id,fragmentShader); gl.LinkProgram(shared->id);
            int linked=0; gl.GetProgramiv(shared->id,GL_LINK_STATUS,&linked);
            if(linked!=GL_TRUE) {
                char log[1024]={}; int length=0; gl.GetProgramInfoLog(shared->id,(int)sizeof(log),&length,log);
                return fail(std::string("OpenGL fixture program link failed: ")+std::string(log,(size_t)std::max(length,0)));
            }
            resources->sharedProgram=shared;
            result.stats.shaderProgramBuildCount = 1;
        }
        resources->program=resources->sharedProgram->id;
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
        uniforms.surfaceCoating = location ("uSurfaceCoating");
        uniforms.timeMixEndColorAndTime = location ("uTimeMixEndColorAndTime");
        uniforms.vertexTime = location ("uVertexTime");
        uniforms.vertexSpectrum = location ("uVertexSpectrum[0]");
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
        uniforms.imageOutput = location ("uImageOutput");
        uniforms.rawIdentifiers = location("uRawIdentifiers[0]");
        uniforms.rawObjectIndex = location("uRawObjectIndex");
        uniforms.passProgramControl = location("uPassProgramControl");
        uniforms.passModes = location("uPassModes[0]");
        uniforms.passColors = location("uPassColors[0]");
        uniforms.passReferences = location("uPassReferences[0]");
        uniforms.passTransforms = location("uPassTransforms[0]");
        uniforms.motionRotation = location("uMotionRotation");
        uniforms.motionTranslationScale = location("uMotionTranslationScale");
        uniforms.previousCameraRotation = location("uPreviousCameraRotation");
        uniforms.previousCameraTranslation = location("uPreviousCameraTranslation");
        uniforms.previousProjection = location("uPreviousProjection");
        uniforms.motionViewport = location("uMotionViewport");
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
        uniforms.diffractionEnvironment = location ("uDiffractionEnvironment");
        uniforms.diffractionEvaluationSchedule
            = location ("uDiffractionEvaluationSchedule");
        uniforms.noteInstanceTransforms = location ("uNoteInstanceTransforms[0]");
        uniforms.noteInstanceIndex = location ("uNoteInstanceIndex");
        uniforms.noteMeshScale = location ("uNoteMeshScale");
        uniforms.noteAppearanceLow = location ("uNoteAppearanceLow");
        uniforms.noteAppearanceHigh = location ("uNoteAppearanceHigh");
        if (resources->vertexArray == 0 || resources->vertexBuffer == 0
            || resources->indexBuffer == 0 || resources->textures.empty()
            || (resources->instancedSharedGeometry && resources->instanceBuffer == 0)
            || resources->program == 0 || ! uniforms.complete())
        {
            return fail ("OpenGL fixture static GPU resource creation failed");
        }

        if (cached == contextPrograms.end())
        {
            constexpr std::size_t kMaximumFixtureProgramCacheEntries = 64;
            if (contextPrograms.size() >= kMaximumFixtureProgramCacheEntries)
            {
                const auto oldest = std::min_element(contextPrograms.begin(), contextPrograms.end(),
                    [](const auto& a, const auto& b) { return a.second.lastUse < b.second.lastUse; });
                contextPrograms.erase(oldest);
            }
            contextPrograms.emplace(vertexSource, CachedProgram{++programUseSerial_, resources->sharedProgram});
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
        std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> materialProgram;
        if (!resolveDiffractionRuntimeProgram(resources->materialProgram, resources->materialProgramSource, runtimeInputs,
                materialProgram, result.error)) return result;
        if (materialProgram != nullptr
            && ! nativeFixtureDiffractionWorkWithinBudget (
                *materialProgram, width, height))
        {
            result.error = "OpenGL fixture diffraction workload exceeds backend limits";
            return result;
        }
        if (!validateMaterialFrameForDraw(materialProgram, runtimeInputs, "opengl",
                reinterpret_cast<std::uintptr_t>(glfwGetCurrentContext()), result.error))
            return result;
        unsigned materialFrameTexture = 0;
        std::vector<std::shared_ptr<const NativeFixtureSceneFrame>> inputFrames;
        if (runtimeInputs.materialFrameTexture) inputFrames.push_back(runtimeInputs.materialFrameTexture);
        for (const auto& entry : runtimeInputs.materialFrameTextures) inputFrames.push_back(entry.second);
        for (const auto& inputFrame : inputFrames)
        {
            const auto descriptor = inputFrame->colorTextureDescriptor();
            if (descriptor.textureViewHandle > std::numeric_limits<unsigned>::max()
                || descriptor.imageHandle != descriptor.textureViewHandle
                || !glIsTexture(static_cast<unsigned>(descriptor.textureViewHandle)))
            { result.error = "OpenGL material Frame texture is stale"; return result; }
            materialFrameTexture = static_cast<unsigned>(descriptor.textureViewHandle);
            GLint previous = 0, actualWidth = 0, actualHeight = 0, format = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
            glBindTexture(GL_TEXTURE_2D, materialFrameTexture);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &actualWidth);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &actualHeight);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
            glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(previous));
            if (actualWidth != static_cast<int>(descriptor.width)
                || actualHeight != static_cast<int>(descriptor.height) || format != GL_RGBA8)
            { result.error = "OpenGL material Frame texture no longer matches its descriptor"; return result; }
        }
        const auto rootFrame = materialFrameForProgram(materialProgram.get(),runtimeInputs);
        materialFrameTexture = rootFrame ? static_cast<unsigned>(rootFrame->colorTextureViewHandle()) : 0;
        if (!validFixtureRuntimeInputs(runtimeInputs))
        {
            result.error = !render3dimage::supported(runtimeInputs.imageOutput)
                ? "Render 3D requested image output is unavailable on this backend"
                : "OpenGL fixture scene modulation is non-finite or out of bounds";
            return result;
        }
        if ((runtimeInputs.imageOutput != renderpassoutput::Output::Color || runtimeInputs.passComposite)
            && materialProgram != nullptr
            && materialProgram->kind == NativeFixtureMaterialKind::DiffractionReflective)
        {
            result.error = "Render 3D pass inspection/compositing is unavailable for diffraction materials";
            return result;
        }
        if (runtimeInputs.imageOutput == renderpassoutput::Output::ObjectId
            && resources->instancedSharedGeometry)
        {
            result.error = "Render 3D Object ID inspection requires individual scene draws";
            return result;
        }
        if (materialProgram != nullptr
            && nativeSurfaceUsesTime(*materialProgram)
            && (! std::isfinite (runtimeInputs.timeSeconds)
                || std::abs (runtimeInputs.timeSeconds)
                    > surfacematerial::kMaximumEvaluationMagnitude))
        {
            result.error = "OpenGL fixture material time input is non-finite or out of bounds";
            return result;
        }

        if (materialProgram != nullptr && materialProgram->vertexProgram)
        {
            if (runtimeInputs.imageOutput == renderpassoutput::Output::Motion
                || sceneUsesMotionPass(runtimeInputs))
            {
                result.error = "Motion output is unavailable for graph vertex deformation";
                return result;
            }
            runtimeInputs.previousMotion.reset();
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
        frame->materialFrameTexture = runtimeInputs.materialFrameTexture;
        frame->materialFrameTextures = runtimeInputs.materialFrameTextures;
        frame->linearColor = runtimeInputs.linearColor;
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
        glTexImage2D (GL_TEXTURE_2D, 0, runtimeInputs.linearColor ? GL_RGBA16F : GL_RGBA8, static_cast<int> (width),
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
        const std::array<int, 6> rawFormats { GL_RGBA16F, GL_RGBA16F, GL_R8, GL_R32UI, GL_R32UI, GL_RG16F };
        glGenTextures(6, frame->rawTextures.data());
        for (std::size_t i = 0; i < frame->rawTextures.size(); ++i)
        {
            glBindTexture(GL_TEXTURE_2D, frame->rawTextures[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, rawFormats[i], static_cast<int>(width),
                static_cast<int>(height), 0, i == 5 ? GL_RG : i < 2 ? GL_RGBA : i == 2 ? GL_RED : GL_RED_INTEGER,
                i == 5 || i < 2 ? GL_FLOAT : i == 2 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_INT, nullptr);
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 2 + static_cast<unsigned>(i),
                GL_TEXTURE_2D, frame->rawTextures[i], 0);
        }
        const unsigned drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4,
            GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7 };
        gl.DrawBuffers (8, drawBuffers);
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
        if (!runtimeInputs.cameraOverride && !runtimeInputs.animatedCamera && selectedCamera == nullptr)
        {
            result.error = "OpenGL fixture render cannot resolve the active camera";
            return result;
        }
        const auto& camera = runtimeInputs.cameraOverride ? *runtimeInputs.cameraOverride
            : runtimeInputs.animatedCamera ? *runtimeInputs.animatedCamera : *selectedCamera;
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
        if (materialProgram != nullptr)
        {
            const auto& program = *materialProgram;
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
        const auto lightCount = runtimeInputs.lightOverride ? std::size_t { 1 }
            : !runtimeInputs.animatedLights.empty() ? runtimeInputs.animatedLights.size() : scene->lightCount;
        const auto lightAt = [&](std::size_t index) -> const SceneLightRecord&
            { return runtimeInputs.lightOverride ? *runtimeInputs.lightOverride
                : !runtimeInputs.animatedLights.empty() ? runtimeInputs.animatedLights[index] : scene->lights[index]; };
        if (lightCount == 1)
        {
            const auto& light = lightAt(0);
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
        if (runtimeInputs.imageOutput == renderpassoutput::Output::Color)
            glClearColor (7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f, 1.0f);
        else
            glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth (1.0);
        glClear (GL_DEPTH_BUFFER_BIT);
        const float background[4] { 7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f, 1.0f };
        const float emptyPass[4] { 0, 0, 0, 0 };
        const float farDepth[4] { 1, 0, 0, 0 };
        const unsigned emptyIdentity[4] { 0, 0, 0, 0 };
        gl.ClearBufferfv(GL_COLOR, 0, runtimeInputs.imageOutput == renderpassoutput::Output::Color
            && !runtimeInputs.passComposite ? background : emptyPass);
        gl.ClearBufferfv(GL_COLOR, 1, farDepth);
        for (int attachment = 2; attachment < 5; ++attachment) gl.ClearBufferfv(GL_COLOR, attachment, emptyPass);
        gl.ClearBufferuiv(GL_COLOR, 5, emptyIdentity);
        gl.ClearBufferuiv(GL_COLOR, 6, emptyIdentity);
        gl.ClearBufferfv(GL_COLOR, 7, emptyPass);
        if (runtimeInputs.passComposite && runtimeInputs.passComposite->inspectionImage)
        {
            const auto clear = renderpasscomposite::inspectionClear(*runtimeInputs.passComposite);
            gl.ClearBufferfv(GL_COLOR, 0, clear.data());
        }
        gl.UseProgram (resources->program);
        gl.BindVertexArray (resources->vertexArray);
        gl.ActiveTexture (GL_TEXTURE0);
        glBindTexture (GL_TEXTURE_2D, resources->textures[0]);
        gl.BindSampler (0, 0);
        const auto& locations = resources->uniforms;
        const auto motion = sceneMotionUniforms(runtimeInputs, camera, width, height);
        gl.Uniform4fv(locations.motionRotation, 1, motion.rotation.data());
        gl.Uniform4fv(locations.motionTranslationScale, 1, motion.translationScale.data());
        gl.Uniform4fv(locations.previousCameraRotation, 1, motion.cameraRotation.data());
        gl.Uniform4fv(locations.previousCameraTranslation, 1, motion.cameraTranslation.data());
        gl.Uniform4fv(locations.previousProjection, 1, motion.projection.data());
        gl.Uniform4fv(locations.motionViewport, 1, motion.viewport.data());
        gl.Uniform4f(locations.vertexTime, runtimeInputs.timeSeconds, 0.0f, 0.0f, 0.0f);
        if (materialProgram != nullptr && materialProgram->vertexProgram)
            gl.Uniform4fv(locations.vertexSpectrum,
                (materialProgram->vertexProgram->resources().audioParameterSlots + 3) / 4,
                runtimeInputs.vertexSpectrum.data());
        std::array<std::array<float, 4>, Visual3DScene::kMaxObjects> rawIdentifiers {};
        for (std::size_t i = 0; i < scene->objectCount; ++i)
        {
            const auto& object = scene->objects[i];
            rawIdentifiers[i] = { static_cast<float>(object.material.value & 65535u),
                static_cast<float>(object.material.value >> 16u), static_cast<float>(object.id.value & 65535u),
                static_cast<float>(object.id.value >> 16u) };
        }
        gl.Uniform4fv(locations.rawIdentifiers, static_cast<int>(rawIdentifiers.size()), rawIdentifiers[0].data());
        gl.Uniform1i(locations.rawObjectIndex, 0);
        auto passProgram = renderpasscomposite::gpuProgram(scenePassProgram(runtimeInputs));
        passProgram.control[2] = runtimeInputs.linearColor ? 1.0f : 0.0f;
        gl.Uniform4fv(locations.passProgramControl, 1, passProgram.control.data());
        gl.Uniform4fv(locations.passModes, 8, passProgram.modes[0].data());
        gl.Uniform4fv(locations.passColors, 8, passProgram.colors[0].data());
        gl.Uniform4fv(locations.passReferences, 8, passProgram.references[0].data());
        gl.Uniform4fv(locations.passTransforms, 8, passProgram.transforms[0].data());
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
        const bool surfacePbr = materialProgram != nullptr
            && materialProgram->kind == NativeFixtureMaterialKind::SurfacePbr;
        const auto coating = surfacePbr ? materialProgram->parameters.transmissionIorClearcoat
            : std::array<float, 4> {0.0f, 1.5f, 0.0f, 0.0f};
        gl.Uniform4f(locations.surfaceCoating, coating[0], coating[1], coating[2], surfacePbr ? 1.0f : 0.0f);
        gl.Uniform4fv (locations.timeMixEndColorAndTime, 1,
                       values.timeMixEndColorAndTime.data());
        gl.Uniform3f (locations.emissive, values.emissive[0],
                      values.emissive[1], values.emissive[2]);
        gl.Uniform3f (locations.ambient, values.ambient[0],
                      values.ambient[1], values.ambient[2]);
        const auto noteTransforms = noteInstanceShaderTransforms(noteInstances);
        gl.Uniform4fv (locations.noteInstanceTransforms,
                       static_cast<int>(visualnoteinstancing::kMaximumInstances),
                       noteTransforms[0].data());
        gl.Uniform1i (locations.noteInstanceIndex, -1);
        gl.Uniform1f (locations.noteMeshScale, noteInstances.meshScale);
        gl.Uniform4fv (locations.noteAppearanceLow, 1, noteInstances.appearanceLow.data());
        gl.Uniform4fv (locations.noteAppearanceHigh, 1, noteInstances.appearanceHigh.data());
        gl.Uniform4fv (locations.lightRotation, 1, values.lightRotation.data());
        gl.Uniform4fv (locations.lightColorIntensity, 1,
                       values.lightColorIntensity.data());
        gl.Uniform4fv (locations.lightPositionRange, 1,
                       values.lightPositionRange.data());
        gl.Uniform1i (locations.lightKind,
                      lightCount == 1 && lightAt(0).kind == SceneLightKind::Point ? 1 : 0);
        gl.Uniform1i (locations.lightCount, static_cast<int> (lightCount));
        for (std::size_t lightIndex = 0; lightIndex < lightCount; ++lightIndex)
        {
            const auto& light = lightAt(lightIndex);
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
        const auto materialKind = materialProgram != nullptr
            ? materialProgram->kind : NativeFixtureMaterialKind::SurfacePbr;
        gl.Uniform1i (locations.materialKind, static_cast<int> (materialKind));
        gl.Uniform4f (locations.imageOutput, static_cast<float> (runtimeInputs.imageOutput),
                      0.0f, 0.0f, 0.0f);
        diffractivefoil::EvaluationSchedule spatialSchedule;
        std::uint32_t instanceBufferUploadCount = 0;
        std::uint32_t instancedDrawCount = 0;
        std::uint32_t submittedInstanceCount = 0;
        std::uint32_t noteInstanceDrawCount = 0;
        std::uint32_t submittedNoteInstanceCount = 0;
        if (materialKind == NativeFixtureMaterialKind::DiffractionReflective)
        {
            gl.ActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, materialFrameTexture != 0
                ? materialFrameTexture : resources->textures[0]);
            gl.BindSampler(0, 0);
            gl.Uniform4f(locations.texturePresence, 0, 0, 0, materialFrameTexture != 0
                ? (materialFrameIsBottomFirst(rootFrame) ? 2.0f : 1.0f) : 0.0f);
            const auto& lighting = materialProgram->diffractionLightingAdmission->description();
            gl.Uniform4f(locations.diffractionEnvironment,
                         static_cast<float>(lighting.version), lighting.bouncePlaneHeight,
                         lighting.bounceMaximumDistance, 0.0f);
            const diffractionmaterial::PhysicalDiffractionGpuLightingPath emptyPath {};
            const auto diffractionPathCount = materialKind
                    == NativeFixtureMaterialKind::DiffractionReflective
                ? materialProgram->diffractionPathCount : 1u;

            if (materialProgram != nullptr
                && materialProgram->diffractionFoilMaximumEvaluations != 0
                && !nativeFixtureSpatialFoilEvaluationSchedule(
                    *materialProgram, width, height, spatialSchedule))
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
                    ? materialProgram->diffractionPaths[pathIndex]
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
                              &materialProgram->diffractionFoilField[0].x);
                gl.Uniform4fv(locations.diffractionOccupancyRectangles, 5,
                              &materialProgram->diffractionOccupancyRectangles[0].x);
                gl.Uniform4f(locations.diffractionSpatialCounts,
                             materialProgram->diffractionFoilMaximumEvaluations > 0 ? 4.0f : 0.0f,
                             static_cast<float>(materialProgram->diffractionOccupancyRectangleCount),
                             0.0f, 0.0f);
                gl.Uniform4f(locations.diffractionEvaluationSchedule,
                             uintAsFloat(spatialSchedule.width),
                             uintAsFloat(spatialSchedule.height),
                             uintAsFloat(spatialSchedule.maximumEvaluations),
                             uintAsFloat(pathIndex));
                if (pathIndex != 0)
                {
                    glEnable(GL_BLEND);
                    for (unsigned attachment = 1; attachment < 8; ++attachment) gl.Disablei(GL_BLEND, attachment);
                    gl.BlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE);
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
                const bool transparentInstance=!resources->instanceAppearances.empty()
                    && resources->instanceAppearances[objectIndex].color[3]<1.0f;
                if (!nativeSurfaceRequiresBlend(nativeSurfaceForObject(materialProgram.get(), scene->objects[objectIndex].id)) && !transparentInstance && (queuedMaterial == nullptr || queuedMaterial->alphaMode != SceneAlphaMode::Blend))
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
                    if (!runtimeTargetsObject(runtimeInputs, scene->objects[index]))
                        instanceRecords[index].matrix = fixtureWorldMatrix(*scene, scene->objects[index]);
                    const auto stableIdentity = scene->objects[index].id.value;
                    const auto colorSlot = videohelper::geometry::diagnosticColorSlot(
                        diagnosticStableIdentities, stableIdentity);
                    instanceRecords[index].identityColor = {
                        static_cast<float>((colorSlot >> 16) & 0xffu) / 255.0f,
                        static_cast<float>((colorSlot >> 8) & 0xffu) / 255.0f,
                        static_cast<float>(colorSlot & 0xffu) / 255.0f,
                        resources->diagnosticInstanceIdentityColors ? 1.0f : 0.0f };
                    instanceRecords[index].color=resources->instanceAppearances[index].color;
                    instanceRecords[index].emission=resources->instanceAppearances[index].emission;
                    for (std::size_t channel=0;channel<3;++channel)
                        instanceRecords[index].emission[channel]*=runtimeInputs.emissionGain;
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
                gl.Uniform1i(locations.rawObjectIndex, static_cast<int>(objectIndex));
                const auto identityColor = render3dimage::identityColor (
                    runtimeInputs.imageOutput == renderpassoutput::Output::MaterialId
                        ? drawObject.material.value : drawObject.id.value);
                gl.Uniform4f (locations.imageOutput, static_cast<float> (runtimeInputs.imageOutput),
                              identityColor[0], identityColor[1], identityColor[2]);
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
                if (!runtimeTargetsObject(runtimeInputs, drawObject)) runtimeTransform = {};
                values.objectMatrix = multiplyFixtureMatrices (
                    fixtureTransformMatrix (runtimeTransform),
                    fixtureWorldMatrix (*scene, drawObject));
                const auto* drawProgram = nativeSurfaceForObject(materialProgram.get(), drawObject.id);
                if (drawMaterial != nullptr)
                {
                    storeFixtureVec3 (values.baseColor, drawMaterial->baseColor,
                                      drawMaterial->opacity);
                    values.materialParams = {
                        drawMaterial->metallic, drawMaterial->roughness,
                        drawMaterial->normalScale, 0.0f
                    };
                    storeFixtureVec3 (values.emissive, drawMaterial->emissive);
                    values.timeMixEndColorAndTime = {};
                    if (drawProgram != nullptr)
                    {
                        const auto& pbr = drawProgram->parameters;
                        values.baseColor = {pbr.baseColorMetallic[0], pbr.baseColorMetallic[1],
                            pbr.baseColorMetallic[2], pbr.normalOpacity[3]};
                        values.materialParams = {pbr.baseColorMetallic[3], pbr.emissionRoughness[3], 1.0f, 0.0f};
                        values.emissive = {pbr.emissionRoughness[0], pbr.emissionRoughness[1], pbr.emissionRoughness[2], 0.0f};
                        if (drawProgram->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix)
                        {
                            values.timeMixEndColorAndTime = {drawProgram->timeMixEndColor[0], drawProgram->timeMixEndColor[1],
                                drawProgram->timeMixEndColor[2], runtimeInputs.timeSeconds};
                            values.materialParams[3] = 1.0f;
                        }
                    }
                    const auto drawCoating = drawProgram ? drawProgram->parameters.transmissionIorClearcoat
                        : std::array<float, 4> {0, 1.5f, 0, 0};
                    gl.Uniform4f(locations.surfaceCoating, drawCoating[0], drawCoating[1], drawCoating[2], drawProgram ? 1.0f : 0.0f);
                    if (!resources->instancedSharedGeometry && !resources->instanceAppearances.empty()) {
                        const auto& appearance=resources->instanceAppearances[objectIndex];
                        for (std::size_t channel=0;channel<3;++channel) {
                            values.baseColor[channel]*=appearance.color[channel];
                            values.timeMixEndColorAndTime[channel]*=appearance.color[channel];
                            values.emissive[channel]+=appearance.emission[channel];
                        }
                        values.baseColor[3]*=appearance.color[3];
                        if (appearance.metallic>=0) values.materialParams[0]=appearance.metallic;
                        if (appearance.roughness>=0) values.materialParams[1]=appearance.roughness;
                    }
                    for (std::size_t channel = 0; channel < 3; ++channel)
                        values.emissive[channel] *= runtimeInputs.emissionGain;
                    gl.Uniform4fv (locations.baseColor, 1, values.baseColor.data());
                    gl.Uniform4fv(locations.timeMixEndColorAndTime, 1, values.timeMixEndColorAndTime.data());
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
                std::array<SceneTextureId, 5> textureIds {
                    drawMaterial != nullptr ? drawMaterial->baseColorTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->metallicRoughnessTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->normalTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->occlusionTexture : SceneTextureId {},
                    drawMaterial != nullptr ? drawMaterial->emissiveTexture : SceneTextureId {}
                };
                if (drawProgram != nullptr) {
                    const auto baseTextureId = textureIds[0];
                    textureIds = {};
                    if (drawProgram->baseColorSource == NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture)
                        textureIds[0] = baseTextureId;
                }
                const bool drawFrame = drawProgram && drawProgram->baseColorSource
                    == NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture;
                const auto selectedFrame = materialFrameForProgram(drawProgram,runtimeInputs);
                const auto selectedTexture = selectedFrame ? static_cast<unsigned>(selectedFrame->colorTextureViewHandle()) : 0;
                for (std::size_t unit = 0; unit < textureIds.size(); ++unit)
                {
                    gl.ActiveTexture (GL_TEXTURE0 + static_cast<unsigned> (unit));
                    glBindTexture (GL_TEXTURE_2D,
                                   unit == 0 && drawFrame && selectedTexture != 0
                                       ? selectedTexture : resources->textures[textureSlot (textureIds[unit])]);
                    gl.BindSampler (static_cast<unsigned> (unit), 0);
                }
                gl.Uniform4f (locations.texturePresence,
                              textureIds[1].isValid() ? 1.0f : 0.0f,
                              textureIds[2].isValid() ? 1.0f : 0.0f,
                              textureIds[3].isValid() ? 1.0f : 0.0f,
                              drawFrame ? (materialFrameIsBottomFirst(selectedFrame) ? 2.0f : 1.0f) : 0.0f);
                gl.Uniform1f (locations.emissiveTexturePresence,
                              textureIds[4].isValid() ? 1.0f : 0.0f);
                const auto alphaMode = nativeSurfaceRequiresBlend(drawProgram) || (!resources->instanceAppearances.empty()
                    && resources->instanceAppearances[objectIndex].color[3]<1.0f) ? SceneAlphaMode::Blend
                    : drawMaterial != nullptr ? drawMaterial->alphaMode : SceneAlphaMode::Opaque;
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
                    for (unsigned attachment = 1; attachment < 8; ++attachment) gl.Disablei(GL_BLEND, attachment);
                    gl.BlendFuncSeparate (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
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
            ? materialProgram->diffractionPathCount
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
        result.stats.reusedMaterialProgram = materialProgram != nullptr;
        result.stats.diffractionEvaluationBudget = spatialSchedule.maximumEvaluations;
        result.stats.diffractionEvaluationCount = spatialSchedule.requiredEvaluations;
        return result;
    }

private:
    struct CachedProgram
    {
        std::uint64_t lastUse = 0;
        std::shared_ptr<OpenGlFixtureSceneResources::SharedProgram> program;
    };
    // Access and eviction hold fixtureMutex. Teardown removes only the current
    // context so an export/probe cannot retire a viewport's cached programs.
    std::map<GLFWwindow*, std::map<std::string, CachedProgram>> programCache_;
    std::uint64_t programUseSerial_ = 0;
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
uniform uint uOutputOffset;
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
    uint base = vertex * 16u;
    uint outputBase = (vertex + uOutputOffset) * 16u;
    for (uint component = 0u; component < 16u; ++component)
        outputVertices[outputBase + component] = baseVertices[base + component];
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
    outputVertices[outputBase] = position.x;
    outputVertices[outputBase + 1u] = position.y;
    outputVertices[outputBase + 2u] = position.z;
    outputVertices[outputBase + 3u] = normal.x;
    outputVertices[outputBase + 4u] = normal.y;
    outputVertices[outputBase + 5u] = normal.z;
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
    int outputOffsetLocation = -1;
    std::vector<std::shared_ptr<const OpenGlDeformationResources>> draws;
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
        if (!source || source->draws.empty()) return prepareDraw(source, materialProgram, {});
        NativeDeformationPreparation result;
        if (source->draws.size() > HarmonicMIDI::grid::Visual3DScene::kMaxObjects)
        { result.error = "native OpenGL deformation draw capacity exceeded"; return result; }
        OpenGlFixtureSceneBackend fixtureBackend;
        const auto fixture = fixtureBackend.prepare(source->scene, materialProgram);
        if (!fixture.prepared) { result.error = fixture.error; return result; }
        auto resources = std::make_shared<OpenGlDeformationResources>();
        resources->source = source;
        resources->ownerContext = glfwGetCurrentContext();
        resources->fixture = std::dynamic_pointer_cast<const OpenGlFixtureSceneResources>(fixture.resources);
        if (!resources->fixture) { result.error = "native OpenGL scene batch has no vertex buffer"; return result; }
        for (const auto& draw : source->draws)
        {
            if (!draw || !draw->draws.empty() || !draw->batchMember || draw->scene != source->scene)
            { result.error = "native OpenGL scene batch has an invalid draw owner"; return result; }
            auto prepared = prepareDraw(draw, {}, resources->fixture);
            if (!prepared.prepared) { result.error = prepared.error; return result; }
            resources->draws.push_back(std::dynamic_pointer_cast<const OpenGlDeformationResources>(prepared.resources));
            result.stats.staticVertexBytes += prepared.stats.staticVertexBytes;
            result.stats.staticDeformationBytes += prepared.stats.staticDeformationBytes;
        }
        result.prepared = true;
        result.resources = std::move(resources);
        result.stats.staticUploadCount = 1;
        return result;
    }

    NativeDeformationPreparation prepareDraw (
        const std::shared_ptr<const NativeDeformationScene>& source,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram,
        std::shared_ptr<const OpenGlFixtureSceneResources> fixture)
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
            || !HarmonicMIDI::grid::validateVisual3DScene(*source->scene).valid()
            || (!source->batchMember && (source->scene->objectCount != 1 || source->scene->materialCount != 1
            || source->scene->lightCount != 1 || source->scene->cameraCount != 1
            || source->scene->objects[0].id != source->object
            || source->scene->objects[0].firstVertex != 0
            || source->scene->objects[0].vertexCount != source->scene->vertexCount)))
        {
            result.error = "native OpenGL deformation requires one exact bounded scene owner";
            return result;
        }
        const auto* mesh = source->deformation->findMesh (source->mesh);
        const auto object = std::find_if(source->scene->objects.begin(),
            source->scene->objects.begin() + source->scene->objectCount,
            [&](const auto& value) { return value.id == source->object; });
        if (mesh == nullptr || object == source->scene->objects.begin() + source->scene->objectCount
            || object->firstVertex > source->scene->vertexCount
            || mesh->vertexCount() > source->scene->vertexCount - object->firstVertex
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

        if (!fixture)
        {
            OpenGlFixtureSceneBackend fixtureBackend;
            auto fixturePreparation = fixtureBackend.prepare (source->scene, materialProgram);
            if (! fixturePreparation.prepared)
            {
                result.error = fixturePreparation.error;
                return result;
            }
            fixture = std::dynamic_pointer_cast<const OpenGlFixtureSceneResources> (
                fixturePreparation.resources);
            if (fixture == nullptr)
            {
                result.error = "native OpenGL deformation did not receive fixture GPU resources";
                return result;
            }
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
        upload (0, source->scene->vertices.data() + object->firstVertex, vertexBytes, GL_STATIC_DRAW);
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
        resources->outputOffsetLocation = gl.GetUniformLocation(resources->computeProgram, "uOutputOffset");
        if (resources->computeProgram == 0 || resources->vertexCountLocation < 0
            || resources->jointCountLocation < 0
            || resources->morphTargetCountLocation < 0
            || resources->outputOffsetLocation < 0
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
        result.stats.skinId = source->skin.value_or(mesh->skin()).value;
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
        if (!source->draws.empty())
        {
            if (resources->draws.size() != source->draws.size())
            { result.error = "native OpenGL scene batch topology changed"; return result; }
            if (source->retainDeformedGeometry)
                for (std::size_t index=0;index<source->scene->vertexCount;++index) {
                    const auto& vertex=source->scene->vertices[index];
                    result.deformedVertices.insert(result.deformedVertices.end(),{
                        vertex.position.x,vertex.position.y,vertex.position.z,
                        vertex.normal.x,vertex.normal.y,vertex.normal.z,vertex.uv.x,vertex.uv.y});
                }
            for (std::size_t index = 0; index < source->draws.size(); ++index)
            {
                auto drawInputs = runtimeInputs;
                if (snapshot->pose().morphEnabled && snapshot->pose().nodeStableId != source->draws[index]->animationNodeStableId
                    && snapshot->combinationMode() != visualanimation::CombinationMode::WeightedBlend)
                    drawInputs.morphWeight = static_cast<float>(snapshot->combinationWeight());
                auto draw = render(source->draws[index], snapshot, resources->draws[index], width, height, drawInputs);
                if (!draw.rendered) { result.error = draw.error; return result; }
                if (source->retainDeformedGeometry) {
                    const auto object=std::find_if(source->scene->objects.begin(),
                        source->scene->objects.begin()+source->scene->objectCount,
                        [&](const auto& value) { return value.id==source->draws[index]->object; });
                    const auto* mesh=source->deformation->findMesh(source->draws[index]->mesh);
                    if (draw.deformedVertices.size()!=mesh->vertexCount()*8u) {
                        result.error="native OpenGL scene geometry readback changed topology"; return result;
                    }
                    std::copy(draw.deformedVertices.begin(),draw.deformedVertices.end(),
                        result.deformedVertices.begin()+object->firstVertex*8u);
                }
                result.stats.dispatchCount += draw.stats.dispatchCount;
                result.stats.dynamicUniformBytes += draw.stats.dynamicUniformBytes;
            }
            auto frame = nativeFixtureSceneBackend().render(source->scene, resources->fixture, width, height, runtimeInputs);
            if (!frame.rendered) { result.error = frame.error; return result; }
            result.rendered = true; result.frame = std::move(frame.frame);
            result.stats.drawCount = frame.stats.drawCount;
            result.stats.reusedStaticResources = true;
            result.stats.sourceStableId = source->sourceStableId;
            result.stats.deformationStableId = source->deformationStableId;
            result.stats.clipId = source->clip.value;
            result.stats.revision = source->structuralRevision;
            result.stats.time = snapshot->time();
            return result;
        }
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
            const auto object = std::find_if(source->scene->objects.begin(),
                source->scene->objects.begin() + source->scene->objectCount,
                [&](const auto& value) { return value.id == source->object; });
            gl.Uniform1ui(resources->outputOffsetLocation, object->firstVertex);
            gl.DispatchCompute ((frameData.vertexCount + 63u) / 64u, 1, 1);
            gl.MemoryBarrier (GL_SHADER_STORAGE_BARRIER_BIT
                              | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
            glFinish();
            if (source->retainDeformedGeometry)
            {
                const auto count = static_cast<std::size_t>(frameData.vertexCount);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, resources->fixture->vertexBuffer);
                const auto* mapped = static_cast<const HarmonicMIDI::grid::SceneVertex*>(gl.MapBufferRange(
                    GL_SHADER_STORAGE_BUFFER, static_cast<std::ptrdiff_t>(object->firstVertex * sizeof(HarmonicMIDI::grid::SceneVertex)),
                    static_cast<std::ptrdiff_t>(count * sizeof(HarmonicMIDI::grid::SceneVertex)),
                    GL_MAP_READ_BIT));
                if (mapped == nullptr)
                {
                    restore();
                    result.error = "native OpenGL deformed geometry readback failed";
                    return result;
                }
                result.deformedVertices.reserve(count * 8u);
                for (std::size_t index=0;index<count;++index) {
                    const auto& vertex=mapped[index];
                    result.deformedVertices.insert(result.deformedVertices.end(),{
                        vertex.position.x,vertex.position.y,vertex.position.z,
                        vertex.normal.x,vertex.normal.y,vertex.normal.z,vertex.uv.x,vertex.uv.y});
                }
                if (gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER) != GL_TRUE)
                {
                    restore();
                    result.deformedVertices.clear();
                    result.error = "native OpenGL deformed geometry buffer became invalid";
                    return result;
                }
            }
            const auto gpuError = glGetError();
            restore();
            if (gpuError != GL_NO_ERROR)
            {
                result.error = "native OpenGL deformation dispatch reported GPU error "
                             + std::to_string (static_cast<unsigned> (gpuError));
                return result;
            }
        }

        if (source->batchMember)
        {
            result.rendered = true;
            result.stats.dispatchCount = 1;
            result.stats.dynamicUniformBytes = frameData.jointCount * 16u * sizeof(float)
                + frameData.morphTargetCount * sizeof(float);
            return result;
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
        result.stats.skinId = mesh != nullptr ? source->skin.value_or(mesh->skin()).value : 0;
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
    static OpenGlOpticalFlowBackend backend;
    return backend;
}

void invalidateNativeOpticalFlowExecutionContext (std::uintptr_t contextIdentity) noexcept
{
    static_cast<OpenGlOpticalFlowBackend&>(nativeOpticalFlowExecutionBackend()).invalidateContext(contextIdentity);
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
