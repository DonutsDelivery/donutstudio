#include "fixture_analytic_sky.h"
#include "fixture_environment_image.h"
#include "gpu_residency_ledger.h"
#include "backend.h"
#include "fixture_vertex_modifier_shader.h"
#include "fixture_texture_upload.h"
#include "fixture_opaque_visibility.h"
#include "fixture_smooth_glass.h"
#include "fixture_glass_caustics_opengl.h"
#include "fixture_rough_glass_tiles_opengl.h"
#include "fixture_diffraction_transport.h"
#include "fixture_primary_object.h"
#include "gl_program_binary_cache.h"
#include "../geometry_core_diagnostic_colors.h"
#include "../geometry_core_admission.h"
#include "../gl_loader.h"
#include "../sdf_native_program.h"
#include "../sdf_native_render_admission.h"
#include "../optical_flow_native_executor.h"
#include "../loop_step_probe.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace arbitgpu
{
namespace
{
constexpr const char* kUnavailable = "native OpenGL SDF execution requires a current OpenGL 3.3 context";

// Opt-in probe for deformation and Motion investigations. ARBIT_DEBUG_DEFORMATION_MOTION=1
// writes the [deformation-motion] and [scene-motion] lines to stderr. The variable is read once.
bool deformationMotionProbeEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("ARBIT_DEBUG_DEFORMATION_MOTION");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// Batch member index for the probe. The recursive member render receives no index.
thread_local std::size_t deformationMotionProbeMember = 0;

std::string deformationMotionProbeFloats(const float* values, std::size_t count)
{
    std::string text;
    for (std::size_t index = 0; index < count; ++index)
    {
        if (index != 0) text += ',';
        char buffer[32];
        std::snprintf(buffer, sizeof buffer, "%.9g", static_cast<double>(values[index]));
        text += buffer;
    }
    return text;
}

const char* kVertexShader = R"glsl(#version 330 core
const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0); }
)glsl";

const char* kFragmentShader = R"glsl(#version 330 core
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out float outDepth;
uniform int uNormalInspection;
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
uniform usampler2D uTextReferences;
uniform usampler2D uTextResources;
uniform usampler2D uTextContours;
uniform sampler2D uTextSegments;
uniform sampler2D uTextIndexBounds;
uniform usampler2D uTextIndexNodes;
uniform usampler2D uTextSegmentIndices;
uniform usampler2D uTextIndexRoots;

ivec2 tableCoordinate(int index, int width) { return ivec2(index % width, index / width); }
float textDistance(int recordIndex, vec3 point, float depth) {
    int resourceIndex = int(texelFetch(uTextReferences,
        tableCoordinate(recordIndex, textureSize(uTextReferences,0).x),0).x);
    float nearest = 3.402823466e38;
    int winding = 0;
    int node = int(texelFetch(uTextIndexRoots,
        tableCoordinate(resourceIndex,textureSize(uTextIndexRoots,0).x),0).x);
    int end = int(texelFetch(uTextIndexNodes,
        tableCoordinate(node,textureSize(uTextIndexNodes,0).x),0).z);
    while (node < end) {
        vec4 bounds = texelFetch(uTextIndexBounds,
            tableCoordinate(node,textureSize(uTextIndexBounds,0).x),0);
        uvec4 index = texelFetch(uTextIndexNodes,
            tableCoordinate(node,textureSize(uTextIndexNodes,0).x),0);
        // These outward boxes prove impossible half-open y-crossings. Nearest
        // distance still visits every segment until its float error is certified.
        bool crossing = point.y >= bounds.y && point.y < bounds.w;
        if (index.y == 0u) { ++node; continue; }
        for (int i = 0; i < int(index.y); ++i) {
            int segment = int(texelFetch(uTextSegmentIndices,
                tableCoordinate(int(index.x)+i,textureSize(uTextSegmentIndices,0).x),0).x);
            vec4 edge = texelFetch(uTextSegments,
                tableCoordinate(segment, textureSize(uTextSegments,0).x),0);
            vec2 delta = edge.zw-edge.xy, relative = point.xy-edge.xy;
            float along = clamp(dot(relative,delta)/dot(delta,delta),0.0,1.0);
            nearest = min(nearest,length(relative-delta*along));
            if (crossing) {
                float side = delta.x*relative.y-delta.y*relative.x;
                // Half-open crossings keep shared vertices from counting twice.
                if (edge.y <= point.y && edge.w > point.y && side > 0.0) ++winding;
                if (edge.y > point.y && edge.w <= point.y && side < 0.0) --winding;
            }
        }
        node = int(index.z);
    }
    float d2 = winding == 0 ? nearest : -nearest;
    vec2 field = vec2(d2,abs(point.z)-depth*0.5);
    return length(max(field,vec2(0.0)))+min(max(field.x,field.y),0.0);
}

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
        if (operation <= 9 || operation == 27) {
            value = operation == 27 ? textDistance(index, points[top], p0.x)
                                    : primitiveDistance(operation, points[top], p0, p1);
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
    outDepth = hit ? clamp(travel / uMaximumDistance,0.0,1.0) : 1.0;
    vec3 point = origin + direction * travel;
    vec3 normal = hit ? sceneNormal(point) : vec3(0.0);
    outNormal = vec4(normal, hit ? 1.0 : 0.0);
    if (!hit) {
        outColor = uNormalInspection != 0 ? vec4(0.5,0.5,0.5,1.0)
            : uOutputPass >= 3 ? vec4(0.0,0.0,0.0,1.0)
            : (uOutputPass == 1 ? vec4(1.0) : vec4(0.02745,0.03922,0.07059,1.0));
        return;
    }
    if (uOutputPass == 1) { outColor = vec4(vec3(clamp(travel / uMaximumDistance,0.0,1.0)),1.0); return; }
    if (uOutputPass == 7) { outColor = vec4(vec3(sceneEdgeDistance(uv)),1.0); return; }
    if (uOutputPass == 3) {
        int contributor;
        sceneDistanceWithMaterial(point,contributor);
        int code = contributor < 0 ? 0 : int(uRecords[contributor].w);
        outColor = vec4(vec3(code&255,(code>>8)&255,(code>>16)&255)/255.0,1.0); return;
    }
    if (uOutputPass == 4) { outColor = vec4(vec3(sceneCurvature(point)),1.0); return; }
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
                *request.compiledProgram, request.geometry, error, true))
            return {};
        return request.compiledProgram;
    }
    return videohelper::sdf::compileNativeSdfProgram (request.geometry, error, true);
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
        if (textTextures_[0] != 0 && contextIdentity_ != 0)
        {
            auto* previous = glfwGetCurrentContext();
            auto* owner = reinterpret_cast<GLFWwindow*> (contextIdentity_);
            if (previous != owner) glfwMakeContextCurrent (owner);
            if (glfwGetCurrentContext() == owner)
                glDeleteTextures (static_cast<int> (textTextures_.size()), textTextures_.data());
            if (previous != owner) glfwMakeContextCurrent (previous);
        }
        if (outputBackend_ != nullptr && lifecycle_.value != 0)
            outputBackend_->releaseRenderPassOutputs (lifecycle_);
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return texture_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return texture_; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return depthTexture_; }
    std::uintptr_t normalTextureViewHandle() const noexcept override { return normalTexture_; }
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::Rgba16Float, texture_, texture_, width_, height_,
                 1, true, contextIdentity_, lifecycle_.value,
                 colortransform::ColorSpace::LinearSRGB, colortransform::TransferFunction::Linear,
                 NativeTextureRowOrder::BottomFirst };
    }
    NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::R32Float, depthTexture_, depthTexture_, width_, height_,
                 1, true, contextIdentity_, lifecycle_.value,
                 colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
                 NativeTextureRowOrder::BottomFirst };
    }
    NativeTextureViewDescriptor normalTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::Rgba16Float, normalTexture_, normalTexture_, width_, height_,
                 1, true, contextIdentity_, lifecycle_.value,
                 colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
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
    bool readDepthFloatPixels (std::vector<float>& output) const override
    {
        output.clear();
        if (depthTexture_ == 0
            || reinterpret_cast<std::uintptr_t>(glfwGetCurrentContext()) != contextIdentity_)
            return false;
        arbitgl::GlFuncs gl;
        std::string missing;
        if (! arbitgl::loadGlFunctions (gl, missing)) return false;
        int previousTexture = 0, previousPackBuffer = 0;
        glGetIntegerv (GL_TEXTURE_BINDING_2D, &previousTexture);
        glGetIntegerv (GL_PIXEL_PACK_BUFFER_BINDING, &previousPackBuffer);
        constexpr std::array<unsigned, 6> packNames {
            GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_PIXELS,
            GL_PACK_SKIP_ROWS, GL_PACK_SWAP_BYTES, GL_PACK_LSB_FIRST };
        std::array<int, 6> previousPack {};
        for (std::size_t i = 0; i < packNames.size(); ++i)
        {
            glGetIntegerv (packNames[i], &previousPack[i]);
            glPixelStorei (packNames[i], i == 0 ? 1 : 0);
        }
        output.resize (static_cast<std::size_t> (width_) * height_);
        gl.BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
        glBindTexture (GL_TEXTURE_2D, depthTexture_);
        glGetTexImage (GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, output.data());
        const auto success = glGetError() == GL_NO_ERROR;
        glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTexture));
        gl.BindBuffer (GL_PIXEL_PACK_BUFFER, static_cast<unsigned> (previousPackBuffer));
        for (std::size_t i = 0; i < packNames.size(); ++i)
            glPixelStorei (packNames[i], previousPack[i]);
        if (! success) { output.clear(); return false; }
        const auto row = static_cast<std::size_t> (width_);
        for (std::uint32_t y = 0; y < height_ / 2; ++y)
            std::swap_ranges (output.data() + y * row,
                output.data() + (y + 1) * row,
                output.data() + (height_ - 1 - y) * row);
        return true;
    }

    std::string backend_ = "opengl";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    unsigned texture_ = 0;
    unsigned depthTexture_ = 0;
    unsigned normalTexture_ = 0;
    std::array<unsigned, 8> textTextures_ {};
    std::shared_ptr<const NativeSdfCompiledProgram> textProgram_;
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
        result.reusesPrimaryTrace = true;
        result.backend = "opengl";
        result.device = renderer != nullptr ? renderer : "OpenGL device";
        for (std::uint32_t operation = static_cast<std::uint32_t> (
                 videowire::SdfOperation::sphere);
             operation <= static_cast<std::uint32_t> (videowire::SdfOperation::domainWarp);
             ++operation)
            result.supportedOperations[operation] = true;
        int textureUnits = 0, combinedUnits = 0, uniformComponents = 0;
        glGetIntegerv (GL_MAX_TEXTURE_IMAGE_UNITS, &textureUnits);
        glGetIntegerv (GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &combinedUnits);
        glGetIntegerv (GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &uniformComponents);
        arbitgl::GlFuncs gl;
        std::string consumerError;
        if (textureUnits >= 8 && combinedUnits >= 8
            && uniformComponents >= static_cast<int> (kNativeSdfMaximumRecords * 12 + 16)
            && arbitgl::loadGlFunctions (gl, consumerError))
        {
            const auto program = programForCurrentContext (gl, consumerError);
            if (program != 0
                && gl.GetUniformLocation (program, "uTextReferences") >= 0
                && gl.GetUniformLocation (program, "uTextSegments") >= 0
                && gl.GetUniformLocation (program, "uTextIndexBounds") >= 0
                && gl.GetUniformLocation (program, "uTextIndexNodes") >= 0
                && gl.GetUniformLocation (program, "uTextSegmentIndices") >= 0
                && gl.GetUniformLocation (program, "uTextIndexRoots") >= 0)
                result.supportedOperations[static_cast<std::size_t> (
                    videowire::SdfOperation::textExtrusion)] = true;
        }
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
        NativeSdfWorkReceipt textWork;
        if (! videohelper::sdf::admitNativeSdfStagedTextWork (request, textWork, result.error))
            return result;
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
        if ((request.normalInspection && request.output != NativeSdfOutput::normal)
            || request.maximumSteps == 0 || request.maximumSteps > available.maxSteps
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

        const bool hasText = ! compiledProgram->textResources().empty();
        if (hasText && ! available.supports (videowire::SdfOperation::textExtrusion))
        {
            result.error = "OpenGL native SDF text table consumer is unavailable";
            return result;
        }
        int maximumTextureSize = 0, maximumAttachments = 0, maximumDrawBuffers = 0;
        glGetIntegerv (GL_MAX_TEXTURE_SIZE, &maximumTextureSize);
        glGetIntegerv (GL_MAX_COLOR_ATTACHMENTS, &maximumAttachments);
        glGetIntegerv (GL_MAX_DRAW_BUFFERS, &maximumDrawBuffers);
        if (maximumAttachments < 3 || maximumDrawBuffers < 3)
        {
            result.error = "OpenGL native SDF staged attachments exceed device capacity";
            return result;
        }
        const std::array<std::size_t, 8> tableCounts {
            compiledProgram->textResourceIndices().size(), compiledProgram->textResources().size(),
            compiledProgram->textContours().size(), compiledProgram->textSegments().size(),
            compiledProgram->textIndexNodes().size(), compiledProgram->textIndexNodes().size(),
            compiledProgram->textSegmentIndices().size(), compiledProgram->textIndexRoots().size() };
        std::array<int, 8> tableWidths {}, tableHeights {};
        std::uint64_t nativeTextBytes = 0, uploadStagingBytes = 0;
        for (std::size_t i = 0; hasText && i < tableCounts.size(); ++i)
        {
            if (maximumTextureSize <= 0 || tableCounts[i] == 0
                || tableCounts[i] > static_cast<std::uint64_t> (maximumTextureSize) * maximumTextureSize)
            {
                result.error = "OpenGL native SDF text tables exceed texture capacity";
                return result;
            }
            tableWidths[i] = static_cast<int> (std::min<std::size_t> (tableCounts[i], maximumTextureSize));
            tableHeights[i] = static_cast<int> ((tableCounts[i] + tableWidths[i] - 1) / tableWidths[i]);
            const auto tableBytes = static_cast<std::uint64_t> (tableWidths[i]) * tableHeights[i]
                * (i == 0 || i >= 6 ? 4u : 16u);
            if (! renderpassoutput::detail::checkedAdd (nativeTextBytes, tableBytes, nativeTextBytes))
            {
                result.error = "OpenGL native SDF text table allocation overflows";
                return result;
            }
            uploadStagingBytes = std::max (uploadStagingBytes, tableBytes);
        }

        const std::uint64_t backendProgramBytes = std::strlen (kVertexShader) + std::strlen (kFragmentShader);
        const std::uint64_t uniformBytes = (kNativeSdfMaximumRecords * 12u * sizeof (float))
            + 4u * sizeof (float) + 12u * sizeof (int);
        const std::uint64_t attachmentBytes = static_cast<std::uint64_t> (request.width) * request.height * 20u;
        std::uint64_t admittedTextBytes = 0;
        if (hasText)
        {
            if (request.geometryCacheBytes != 0
                && request.geometryCacheBytes < compiledProgram->storageBytes())
            {
                result.error = "OpenGL native SDF text cache receipt is smaller than its compiled program";
                return result;
            }
            const auto uncachedProgramBytes = request.geometryCacheBytes == 0
                ? compiledProgram->storageBytes() : 0;
            for (const auto bytes : { request.geometryCacheBytes, uncachedProgramBytes,
                    backendProgramBytes, uniformBytes, attachmentBytes, nativeTextBytes })
                if (! renderpassoutput::detail::checkedAdd (admittedTextBytes, bytes, admittedTextBytes)
                    || admittedTextBytes > renderpassoutput::kMaximumTotalBytes)
                {
                    result.error = "OpenGL native SDF text aggregate exceeds resource capacity";
                    return result;
                }
        }

        // Upload tables one at a time. The host staging vector is transient,
        // but its peak overlaps every retained allocation and must fit too.
        std::uint64_t peakTextBytes = admittedTextBytes;
        if (hasText
            && (! renderpassoutput::detail::checkedAdd (peakTextBytes, uploadStagingBytes, peakTextBytes)
                || peakTextBytes > renderpassoutput::kMaximumTotalBytes))
        {
            result.error = "OpenGL native SDF text upload exceeds aggregate resource capacity";
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
        int previousTexture = 0, previousActiveTexture = 0, previousUnpackBuffer = 0;
        std::array<int, 8> previousTableTextures {}, previousSamplers {};
        constexpr std::array<unsigned, 6> unpackNames { GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH,
            GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS, GL_UNPACK_SWAP_BYTES, GL_UNPACK_LSB_FIRST };
        std::array<int, 6> previousUnpack {};
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
        glGetIntegerv (GL_ACTIVE_TEXTURE, &previousActiveTexture);
        glGetIntegerv (GL_PIXEL_UNPACK_BUFFER_BINDING, &previousUnpackBuffer);
        for (std::size_t i = 0; i < unpackNames.size(); ++i)
            glGetIntegerv (unpackNames[i], &previousUnpack[i]);
        for (unsigned i = 0; i < previousTableTextures.size(); ++i)
        {
            gl.ActiveTexture (GL_TEXTURE0 + i);
            glGetIntegerv (GL_TEXTURE_BINDING_2D, &previousTableTextures[i]);
            glGetIntegerv (GL_SAMPLER_BINDING, &previousSamplers[i]);
        }
        gl.ActiveTexture (static_cast<unsigned> (previousActiveTexture));
        glGetIntegerv (GL_VIEWPORT, previousViewport);
        glGetIntegerv (GL_POLYGON_MODE, previousPolygonMode);
        glGetBooleanv (GL_COLOR_WRITEMASK, previousColorMask);

        unsigned vertexArray = 0;
        auto frame = std::make_shared<OpenGlSdfSceneFrame>();
        frame->width_ = request.width;
        frame->height_ = request.height;
        frame->contextIdentity_ = reinterpret_cast<std::uintptr_t>(glfwGetCurrentContext());
        if (hasText) frame->textProgram_ = compiledProgram;

        auto cleanup = [&]
        {
            gl.BindFramebuffer (GL_DRAW_FRAMEBUFFER,
                                static_cast<unsigned> (previousDrawFramebuffer));
            gl.BindFramebuffer (GL_READ_FRAMEBUFFER,
                                static_cast<unsigned> (previousReadFramebuffer));
            gl.UseProgram (static_cast<unsigned> (previousProgram));
            gl.BindVertexArray (static_cast<unsigned> (previousVertexArray));
            for (unsigned i = 0; i < previousTableTextures.size(); ++i)
            {
                gl.ActiveTexture (GL_TEXTURE0 + i);
                glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTableTextures[i]));
                gl.BindSampler (i, static_cast<unsigned> (previousSamplers[i]));
            }
            gl.ActiveTexture (static_cast<unsigned> (previousActiveTexture));
            glBindTexture (GL_TEXTURE_2D, static_cast<unsigned> (previousTexture));
            gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, static_cast<unsigned> (previousUnpackBuffer));
            for (std::size_t i = 0; i < unpackNames.size(); ++i)
                glPixelStorei (unpackNames[i], previousUnpack[i]);
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

        renderpassoutput::Description outputDescription;
        outputDescription.extent = { request.width, request.height };
        outputDescription.attachments.push_back ({
            renderpassoutput::Output::Color,
            renderpassoutput::PixelFormat::RGBA16Float,
            renderpassoutput::ColorSpace::LinearSRGB,
            outputDescription.extent });
        outputDescription.attachments.push_back ({
            renderpassoutput::Output::Depth,
            renderpassoutput::PixelFormat::R32Float,
            renderpassoutput::ColorSpace::Data,
            outputDescription.extent });
        outputDescription.attachments.push_back ({
            renderpassoutput::Output::Normal,
            renderpassoutput::PixelFormat::RGBA16Float,
            renderpassoutput::ColorSpace::Data,
            outputDescription.extent });
        renderpassoutput::AdmissionFailure outputFailure;
        const auto admittedOutputs = renderpassoutput::admit (outputDescription, outputFailure);
        if (! admittedOutputs)
        {
            return fail ("OpenGL native SDF output admission failed: "
                + std::string (renderpassoutput::token (outputFailure)));
        }
        gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, 0);
        for (std::size_t i = 0; i < unpackNames.size(); ++i)
            glPixelStorei (unpackNames[i], i == 0 ? 1 : 0);
        auto& outputBackend = nativeRenderPassOutputBackend();
        auto outputAdmission = outputBackend.admitRenderPassOutputs (*admittedOutputs);
        if (outputAdmission.lifecycle.value == 0 || outputAdmission.resources.size() != 3)
        {
            if (outputAdmission.lifecycle.value != 0)
                outputBackend.releaseRenderPassOutputs (outputAdmission.lifecycle);
            return fail (outputAdmission.error.empty()
                ? "OpenGL native SDF output allocation failed"
                : std::move (outputAdmission.error));
        }
        frame->outputBackend_ = &outputBackend;
        frame->lifecycle_ = outputAdmission.lifecycle;
        frame->frameMemory_ = outputAdmission.frameMemory;
        frame->receipt_.compiledRecordCount = static_cast<std::uint32_t> (
            compiledProgram->records().size());
        frame->receipt_.compiledRecordBytes = frame->receipt_.compiledRecordCount
            * sizeof (NativeSdfCompiledRecord);
        frame->receipt_.geometryCacheBytes = request.geometryCacheBytes;
        frame->receipt_.backendProgramBytes = backendProgramBytes;
        frame->receipt_.uniformBytes = uniformBytes;
        frame->receipt_.attachmentBytes = outputAdmission.frameMemory.requestedBytes;
        frame->receipt_.compiledTextTableBytes = compiledProgram->textTableBytes();
        frame->receipt_.nativeTextTableBytes = nativeTextBytes;
        frame->receipt_.distanceQueryCount = textWork.distanceQueryCount;
        frame->receipt_.recordVisits = textWork.recordVisits;
        frame->receipt_.segmentVisits = textWork.segmentVisits;
        frame->receipt_.textIndexNodeVisits = textWork.textIndexNodeVisits;
        // A renderer cache already owns this program and its compiled tables.
        const auto uncachedProgramBytes = ! hasText ? frame->receipt_.compiledRecordBytes
            : request.geometryCacheBytes == 0 ? compiledProgram->storageBytes() : 0;
        for (const auto bytes : { uncachedProgramBytes, frame->receipt_.geometryCacheBytes,
                frame->receipt_.backendProgramBytes, frame->receipt_.uniformBytes,
                frame->receipt_.attachmentBytes, nativeTextBytes })
            if (! renderpassoutput::detail::checkedAdd (frame->receipt_.totalBytes, bytes,
                                                       frame->receipt_.totalBytes))
                return fail ("OpenGL native SDF resource receipt overflows");
        frame->receipt_.transientUploadBytes = uploadStagingBytes;
        if (! renderpassoutput::detail::checkedAdd (frame->receipt_.totalBytes, uploadStagingBytes,
                                                   frame->receipt_.peakBytes))
            return fail ("OpenGL native SDF peak allocation receipt overflows");
        if (hasText && (frame->receipt_.totalBytes != admittedTextBytes
                       || frame->receipt_.peakBytes != peakTextBytes))
            return fail ("OpenGL native SDF text allocation receipt differs from admitted resources");
        frame->texture_ = static_cast<unsigned> (outputAdmission.resources[0].image);
        const auto framebuffer = static_cast<unsigned> (
            outputAdmission.resources[0].attachmentView);
        frame->depthTexture_ = static_cast<unsigned> (outputAdmission.resources[1].image);
        frame->normalTexture_ = static_cast<unsigned> (outputAdmission.resources[2].image);

        gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
        gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                                 GL_TEXTURE_2D, frame->normalTexture_, 0);
        gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,
                                 GL_TEXTURE_2D, frame->depthTexture_, 0);
        const unsigned sdfDrawBuffers[3] { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        gl.DrawBuffers (3, sdfDrawBuffers);
        if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            return fail ("OpenGL native SDF framebuffer is incomplete");

        const auto program = programForCurrentContext (gl, result.error);
        if (program == 0) return fail (result.error);

        gl.GenVertexArrays (1, &vertexArray);
        gl.BindVertexArray (vertexArray);
        gl.UseProgram (program);
        const std::array<const char*, 8> tableUniforms {
            "uTextReferences", "uTextResources", "uTextContours", "uTextSegments",
            "uTextIndexBounds", "uTextIndexNodes", "uTextSegmentIndices", "uTextIndexRoots" };
        if (hasText)
            glGenTextures (static_cast<int> (frame->textTextures_.size()), frame->textTextures_.data());
        for (unsigned i = 0; i < tableUniforms.size(); ++i)
        {
            gl.Uniform1i (gl.GetUniformLocation (program, tableUniforms[i]), static_cast<int> (i));
            if (! hasText) continue;
            gl.ActiveTexture (GL_TEXTURE0 + i);
            gl.BindSampler (i, 0);
            glBindTexture (GL_TEXTURE_2D, frame->textTextures_[i]);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
            glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
            const auto texelBytes = i == 0 || i >= 6 ? 4u : 16u;
            std::vector<std::uint32_t> padded (static_cast<std::size_t> (tableWidths[i])
                * tableHeights[i] * texelBytes / sizeof (std::uint32_t));
            if (i == 4 || i == 5)
            {
                for (std::size_t node = 0; node < compiledProgram->textIndexNodes().size(); ++node)
                {
                    const auto& entry = compiledProgram->textIndexNodes()[node];
                    if (i == 4)
                    {
                        const std::array<float,4> bounds { entry.minX,entry.minY,entry.maxX,entry.maxY };
                        std::memcpy (padded.data()+node*4, bounds.data(), sizeof (bounds));
                    }
                    else
                    {
                        const std::array<std::uint32_t,4> indices {
                            entry.firstIndex,entry.indexCount,entry.escape,entry.reserved };
                        std::copy (indices.begin(),indices.end(),padded.begin()+node*4);
                    }
                }
            }
            else
            {
                const void* source = i == 0 ? static_cast<const void*> (compiledProgram->textResourceIndices().data())
                    : i == 1 ? static_cast<const void*> (compiledProgram->textResources().data())
                    : i == 2 ? static_cast<const void*> (compiledProgram->textContours().data())
                    : i == 3 ? static_cast<const void*> (compiledProgram->textSegments().data())
                    : i == 6 ? static_cast<const void*> (compiledProgram->textSegmentIndices().data())
                             : static_cast<const void*> (compiledProgram->textIndexRoots().data());
                std::memcpy (padded.data(), source, tableCounts[i] * texelBytes);
            }
            const bool floating = i == 3 || i == 4;
            const bool scalar = i == 0 || i >= 6;
            const auto internal = scalar ? GL_R32UI : floating ? GL_RGBA32F : GL_RGBA32UI;
            glTexImage2D (GL_TEXTURE_2D, 0, internal, tableWidths[i], tableHeights[i], 0,
                scalar ? GL_RED_INTEGER : floating ? GL_RGBA : GL_RGBA_INTEGER,
                floating ? GL_FLOAT : GL_UNSIGNED_INT, padded.data());
            int actualWidth = 0, actualHeight = 0, actualFormat = 0;
            glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &actualWidth);
            glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &actualHeight);
            glGetTexLevelParameteriv (GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &actualFormat);
            if (frame->textTextures_[i] == 0 || actualWidth != tableWidths[i] || actualHeight != tableHeights[i]
                || actualFormat != internal
                || glGetError() != GL_NO_ERROR)
                return fail ("OpenGL native SDF text table allocation failed");
        }
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
        gl.Uniform1i (gl.GetUniformLocation (program, "uNormalInspection"),
                      request.normalInspection ? 1 : 0);
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
    unsigned programForCurrentContext (const arbitgl::GlFuncs& gl, std::string& error) const
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

    mutable std::mutex programMutex_;
    mutable std::unordered_map<ContextKey, unsigned, ContextKeyHash> programs_;
    mutable std::unordered_map<GLFWwindow*, std::uint64_t> contextGenerations_;
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
// Bounded integer search only: 81 candidates with a 3x3 luminance patch.
// A match needs <=0.05 linear-luma RMS error and a unique runner-up margin.
// Both directions must pass confidence and return to the exact First pixel.
// Rejection writes finite zero, which is not a validity-mask API.
ivec2 matchPatch(sampler2D first,sampler2D second,ivec2 g,out bool confident){
    float best=3.402823466e38; float runnerUp=best; ivec2 bestD=ivec2(0); int bestMag=0;
    for(int dy=-4;dy<=4;++dy) for(int dx=-4;dx<=4;++dx){
        float score=0.0;
        for(int py=-1;py<=1;++py) for(int px=-1;px<=1;++px){
            ivec2 a=bounded(g+ivec2(px,py)); ivec2 b=bounded(g+ivec2(px+dx,py+dy));
            float d=luma(texelFetch(first,a,0))-luma(texelFetch(second,b,0)); score+=d*d;
        }
        int mag=dx*dx+dy*dy; bool tie=score==best && (mag<bestMag || (mag==bestMag && (dy<bestD.y || (dy==bestD.y && dx<bestD.x))));
        if(score<best || tie){runnerUp=best;best=score;bestD=ivec2(dx,dy);bestMag=mag;}
        else {runnerUp=min(runnerUp,score);}
    }
    confident=!isnan(best) && !isinf(best) && best<=9.0*0.0025
        && runnerUp>best*1.1+0.000001;
    return bestD;
}
void main(){
    ivec2 g=ivec2(gl_FragCoord.xy); bool forwardConfidence=false;
    ivec2 forward=matchPatch(uFirst,uSecond,g,forwardConfidence);
    ivec2 destination=g+forward;
    outFlow=vec2(0.0);
    if(!forwardConfidence || any(lessThan(destination,ivec2(0)))
        || any(greaterThanEqual(destination,ivec2(uExtent)))) return;
    // At most one more 81-candidate search; no pyramid or large-motion claim.
    bool reverseConfidence=false;
    ivec2 reverse=matchPatch(uSecond,uFirst,destination,reverseConfidence);
    if(reverseConfidence && all(equal(reverse,-forward))) outFlow=vec2(forward);
})glsl";

const char* kOpticalMotionViewFragment = R"glsl(#version 330 core
layout(location=0) out vec4 outImage;
uniform sampler2D uField;
uniform vec2 uExtent;
uniform float uDisplayRangePixels;
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    // The uploaded first and second frames keep top-first rows at texture
    // coordinate V=0. The flow kernel and this pass preserve that row order.
    vec2 flow = texelFetch(uField, pixel, 0).rg;
    outImage = vec4(clamp(vec2(0.5) + flow/(uDisplayRangePixels > 0.0 ? vec2(uDisplayRangePixels) : uExtent), 0.0, 1.0), 0.5, 1.0);
})glsl";

const char* kOpticalMotionBlurFragment = R"glsl(#version 330 core
layout(location=0) out vec4 outImage;
uniform sampler2D uImage;
uniform sampler2D uField;
uniform vec2 uExtent;
uniform int uSamples;
uniform float uScale;
vec3 toSrgb(vec3 linearColor) {
    vec3 low = 12.92 * linearColor;
    vec3 high = 1.055 * pow(max(linearColor, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return clamp(mix(low, high, step(vec3(0.0031308), linearColor)), 0.0, 1.0);
}
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec2 motion = texelFetch(uField, pixel, 0).rg * uScale;
    vec4 sum = vec4(0.0);
    for (int i = 0; i < 64; ++i) {
        if (i >= uSamples) break;
        float position = uSamples == 1 ? 0.0 : float(i) / float(uSamples - 1) - 0.5;
        vec2 uv = clamp((vec2(pixel) + vec2(0.5) + motion * position) / uExtent,
                        vec2(0.0), vec2(1.0));
        sum += texture(uImage, uv);
    }
    vec4 linearColor = sum / float(uSamples);
    outImage = vec4(toSrgb(linearColor.rgb), clamp(linearColor.a, 0.0, 1.0));
})glsl";

class OpenGlOpticalFlowBackend;
class OpenGlMotionViewFrame final : public NativeFixtureSceneFrame
{
public:
    OpenGlMotionViewFrame(OpenGlOpticalFlowBackend& backend, GLFWwindow* context,
        unsigned texture, std::uint32_t width, std::uint32_t height,
        std::uint64_t generation,
        std::shared_ptr<const videoopticalflow::PublishedNativeResult> field,
        std::shared_ptr<const videoopticalflow::NativeSourceFrame> image = {})
        : backend_(backend), context_(context), texture_(texture), width_(width),
          height_(height), generation_(generation), field_(std::move(field)),
          image_(std::move(image)) {}
    ~OpenGlMotionViewFrame() override;
    const std::string& backend() const noexcept override { return backendName_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return texture_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return texture_; }
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backendName_, NativeTextureViewKind::Texture2D,
            NativeTexturePixelFormat::Rgba8Unorm, texture_, texture_, width_, height_,
            1, true, reinterpret_cast<std::uintptr_t>(context_), generation_,
            colortransform::ColorSpace::SRGB, colortransform::TransferFunction::SRGB,
            NativeTextureRowOrder::TopFirst };
    }
private:
    OpenGlOpticalFlowBackend& backend_;
    GLFWwindow* context_ = nullptr;
    unsigned texture_ = 0;
    std::uint32_t width_ = 0, height_ = 0;
    std::uint64_t generation_ = 0;
    std::shared_ptr<const videoopticalflow::PublishedNativeResult> field_;
    std::shared_ptr<const videoopticalflow::NativeSourceFrame> image_;
    const std::string backendName_ = "opengl";
};

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
        const auto* pair=request.sourcePairV2() ? &*request.sourcePairV2() : nullptr;
        const bool v2Receipts=pair==nullptr
            || (exactOpticalFlowV2InputReceipt(request,first,pair->first)
                && exactOpticalFlowV2InputReceipt(request,second,pair->second)
                && first.structuralRevision==pair->structuralRevision
                && first.nativeContextIdentity==reinterpret_cast<std::uintptr_t>(context)
                && second.nativeContextIdentity==first.nativeContextIdentity);
        const bool exact=v2Receipts&&first.immutable&&second.immutable&&first.identity!=second.identity&&first.imageHandle&&second.imageHandle&&first.imageHandle!=second.imageHandle
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
    std::shared_ptr<const NativeFixtureSceneFrame> renderMotionView (
        const std::shared_ptr<const videoopticalflow::PublishedNativeResult>& field,
        const motionviewoperation::Payload& presentation, std::string& error) override
    {
        if (! motionviewoperation::valid (presentation)
            || presentation.convention != motionviewoperation::Convention::OpticalFirstToSecondTopLeft)
        { error = "OpenGL Motion View requires a valid optical display range"; return {}; }
        std::lock_guard<std::mutex> lock(mutex_);
        auto* context = glfwGetCurrentContext();
        if (!field || !context || field->nativeContextIdentity()
                != reinterpret_cast<std::uintptr_t>(context)
            || field->structuralRevision() == 0 || field->helperGeneration() == 0
            || field->submissionSerial() == 0 || field->imageHandle() == 0
            || field->imageHandle() != field->textureViewHandle()
            || field->textureViewHandle() > std::numeric_limits<unsigned>::max()
            || field->receipt().backendImplementation() != opticalFlowCapabilities().implementation
            || field->receipt().backendImplementationRevision()
                != opticalFlowCapabilities().implementationRevision)
        { error = "OpenGL Motion View field belongs to another context or backend"; return {}; }
        const auto& descriptor = field->receipt().motionVectors().descriptor;
        const auto width = descriptor.extent.width, height = descriptor.extent.height;
        if (descriptor.format != renderpassoutput::PixelFormat::RG16Float
            || width == 0 || height == 0 || field->receipt().motionVectors().helperGeneration
                != field->helperGeneration())
        { error = "OpenGL Motion View requires an exact RG16F field receipt"; return {}; }
        arbitgl::GlFuncs gl;
        if (!loadCurrentGl(gl,error)) return {};
        collectRetired(context);
        const State saved(gl);
        const auto source = static_cast<unsigned>(field->textureViewHandle());
        if (glIsTexture(source) != GL_TRUE)
        { error = "OpenGL Motion View field expired"; return {}; }
        gl.ActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,source);
        int actualWidth=0,actualHeight=0,actualFormat=0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&actualWidth);
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&actualHeight);
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&actualFormat);
        if (actualWidth != static_cast<int>(width) || actualHeight != static_cast<int>(height)
            || actualFormat != GL_RG16F)
        { error = "OpenGL Motion View field storage changed"; return {}; }
        if (!ensureViewProgram(context,gl,error)) return {};
        unsigned output=0,fbo=0;
        glGenTextures(1,&output); glBindTexture(GL_TEXTURE_2D,output);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,static_cast<int>(width),static_cast<int>(height),
                     0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
        gl.GenFramebuffers(1,&fbo); gl.BindFramebuffer(GL_FRAMEBUFFER,fbo);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,output,0);
        if (!output || !fbo || gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
        {
            if(fbo) gl.DeleteFramebuffers(1,&fbo);
            if(output) glDeleteTextures(1,&output);
            error = "OpenGL Motion View Image target is incomplete"; return {};
        }
        const auto& program=viewPrograms_.at(context);
        gl.UseProgram(program.program); gl.BindVertexArray(program.vao);
        gl.ActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,source);
        gl.Uniform1i(program.field,0); gl.Uniform2f(program.extent,float(width),float(height));
        gl.Uniform1f(program.displayRangePixels,static_cast<float>(presentation.displayRangePixels));
        for (const auto capability : State::capabilities) glDisable(capability);
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        glViewport(0,0,static_cast<int>(width),static_cast<int>(height));
        glDrawArrays(GL_TRIANGLES,0,3); glFinish();
        const auto drawError=glGetError();
        gl.DeleteFramebuffers(1,&fbo);
        if(drawError!=GL_NO_ERROR)
        { glDeleteTextures(1,&output); error="OpenGL Motion View Image draw failed"; return {}; }
        do ++nextViewGeneration_; while(nextViewGeneration_==0);
        views_.insert({ {context,output}, true });
        error.clear();
        return std::make_shared<OpenGlMotionViewFrame>(
            *this,context,output,width,height,nextViewGeneration_,field);
    }
    std::shared_ptr<const NativeFixtureSceneFrame> renderMotionBlur (
        const std::shared_ptr<const videoopticalflow::NativeSourceFrame>& image,
        const std::shared_ptr<const videoopticalflow::PublishedNativeResult>& field,
        const visualtemporalsampling::Payload& payload, std::string& error) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* context = glfwGetCurrentContext();
        const auto caps = opticalFlowCapabilities();
        if (!image || !field || !context || !visualtemporalsampling::validate(payload)
            || payload.mode != visualtemporalsampling::Mode::motionBlur
            || image->nativeContextIdentity() != reinterpret_cast<std::uintptr_t>(context)
            || field->nativeContextIdentity() != image->nativeContextIdentity()
            || image->helperGeneration() == 0
            || image->helperGeneration() != field->helperGeneration()
            || image->structuralRevision() == 0
            || image->structuralRevision() != field->structuralRevision()
            || image->backendImplementation() != caps.implementation
            || image->backendImplementationRevision() != caps.implementationRevision
            || field->receipt().backendImplementation() != caps.implementation
            || field->receipt().backendImplementationRevision() != caps.implementationRevision
            || image->frame().extent != field->receipt().motionVectors().descriptor.extent
            || image->frame().format != renderpassoutput::PixelFormat::RGBA16Float
            || field->receipt().motionVectors().descriptor.format
                != renderpassoutput::PixelFormat::RG16Float
            || image->textureViewHandle() == 0 || field->textureViewHandle() == 0
            || image->textureViewHandle() > std::numeric_limits<unsigned>::max()
            || field->textureViewHandle() > std::numeric_limits<unsigned>::max())
        { error = "OpenGL Motion Blur requires the exact retained image and optical field"; return {}; }
        arbitgl::GlFuncs gl;
        if (!loadCurrentGl(gl,error)) return {};
        collectRetired(context);
        const State saved(gl);
        const auto width=image->frame().extent.width, height=image->frame().extent.height;
        const unsigned sources[] = { static_cast<unsigned>(image->textureViewHandle()),
                                     static_cast<unsigned>(field->textureViewHandle()) };
        const int formats[] = { GL_RGBA16F, GL_RG16F };
        for (int index=0; index<2; ++index)
        {
            if (glIsTexture(sources[index]) != GL_TRUE)
            { error="OpenGL Motion Blur source storage expired"; return {}; }
            gl.ActiveTexture(GL_TEXTURE0+index); glBindTexture(GL_TEXTURE_2D,sources[index]);
            int actualWidth=0,actualHeight=0,actualFormat=0;
            glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&actualWidth);
            glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&actualHeight);
            glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&actualFormat);
            if (actualWidth!=static_cast<int>(width) || actualHeight!=static_cast<int>(height)
                || actualFormat!=formats[index])
            { error="OpenGL Motion Blur source storage changed"; return {}; }
        }
        if (!ensureBlurProgram(context,gl,error)) return {};
        unsigned output=0,fbo=0;
        glGenTextures(1,&output); glBindTexture(GL_TEXTURE_2D,output);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,static_cast<int>(width),static_cast<int>(height),
                     0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
        gl.GenFramebuffers(1,&fbo); gl.BindFramebuffer(GL_FRAMEBUFFER,fbo);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,output,0);
        if (!output || !fbo || gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
        {
            if(fbo) gl.DeleteFramebuffers(1,&fbo);
            if(output) glDeleteTextures(1,&output);
            error="OpenGL Motion Blur Image target is incomplete"; return {};
        }
        const auto& program=blurPrograms_.at(context);
        gl.UseProgram(program.program); gl.BindVertexArray(program.vao);
        gl.ActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,sources[0]);
        gl.ActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,sources[1]);
        gl.Uniform1i(program.image,0); gl.Uniform1i(program.field,1);
        gl.Uniform2f(program.extent,float(width),float(height));
        gl.Uniform1i(program.samples,static_cast<int>(payload.sampleCount));
        gl.Uniform1f(program.scale,payload.motionScale*payload.shutterAngleDegrees/360.0f);
        for (const auto capability : State::capabilities) glDisable(capability);
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        glViewport(0,0,static_cast<int>(width),static_cast<int>(height));
        glDrawArrays(GL_TRIANGLES,0,3); glFinish();
        const auto drawError=glGetError();
        gl.DeleteFramebuffers(1,&fbo);
        if(drawError!=GL_NO_ERROR)
        { glDeleteTextures(1,&output); error="OpenGL Motion Blur Image draw failed"; return {}; }
        do ++nextViewGeneration_; while(nextViewGeneration_==0);
        views_.insert({ {context,output}, true });
        error.clear();
        return std::make_shared<OpenGlMotionViewFrame>(
            *this,context,output,width,height,nextViewGeneration_,field,image);
    }
    void releaseView(unsigned texture, GLFWwindow* context) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto key=std::make_pair(context,texture);
        const auto found=views_.find(key);
        if(found==views_.end()) return;
        if(glfwGetCurrentContext()==context) { glDeleteTextures(1,&texture); views_.erase(found); }
        else retiredViews_.insert(key);
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
        for(auto view=views_.begin();view!=views_.end();)
        {
            if(view->first.first!=context) { ++view; continue; }
            unsigned texture=view->first.second;
            glDeleteTextures(1,&texture);
            retiredViews_.erase(view->first);
            view=views_.erase(view);
        }
        const auto viewProgram=viewPrograms_.find(context);
        if(viewProgram!=viewPrograms_.end())
        {
            gl.DeleteVertexArrays(1,&viewProgram->second.vao);
            gl.DeleteProgram(viewProgram->second.program);
            viewPrograms_.erase(viewProgram);
        }
        const auto blurProgram=blurPrograms_.find(context);
        if(blurProgram!=blurPrograms_.end())
        {
            if(blurProgram->second.vao) gl.DeleteVertexArrays(1,&blurProgram->second.vao);
            if(blurProgram->second.program) gl.DeleteProgram(blurProgram->second.program);
            blurPrograms_.erase(blurProgram);
        }
    }
private:
    struct Program{unsigned program=0,vao=0;int first=-1,second=-1,extent=-1;};
    struct ViewProgram{unsigned program=0,vao=0;int field=-1,extent=-1,displayRangePixels=-1;};
    struct BlurProgram{unsigned program=0,vao=0;int image=-1,field=-1,extent=-1,samples=-1,scale=-1;};
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
        for(auto item=retiredViews_.begin();item!=retiredViews_.end();)
        {
            const auto found=views_.find(*item);
            if(found==views_.end()) { item=retiredViews_.erase(item); continue; }
            if(found->first.first!=context) { ++item; continue; }
            unsigned texture=item->second;
            glDeleteTextures(1,&texture); views_.erase(found);
            item=retiredViews_.erase(item);
        }
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
    bool ensureViewProgram(GLFWwindow* c,const arbitgl::GlFuncs& gl,std::string& error)
    {
        if(viewPrograms_.count(c)) return true;
        auto vs=compileShader(gl,GL_VERTEX_SHADER,kVertexShader,error);
        if(!vs) return false;
        auto fs=compileShader(gl,GL_FRAGMENT_SHADER,kOpticalMotionViewFragment,error);
        if(!fs) { gl.DeleteShader(vs); return false; }
        ViewProgram p; p.program=gl.CreateProgram(); gl.AttachShader(p.program,vs);
        gl.AttachShader(p.program,fs); gl.LinkProgram(p.program);
        gl.DeleteShader(vs); gl.DeleteShader(fs);
        int linked=0; gl.GetProgramiv(p.program,GL_LINK_STATUS,&linked);
        if(linked!=GL_TRUE) { gl.DeleteProgram(p.program); error="OpenGL Motion View program link failed"; return false; }
        gl.GenVertexArrays(1,&p.vao);
        p.field=gl.GetUniformLocation(p.program,"uField");
        p.extent=gl.GetUniformLocation(p.program,"uExtent");
        p.displayRangePixels=gl.GetUniformLocation(p.program,"uDisplayRangePixels");
        viewPrograms_[c]=p; return true;
    }
    bool ensureBlurProgram(GLFWwindow* c,const arbitgl::GlFuncs& gl,std::string& error)
    {
        if(blurPrograms_.count(c)) return true;
        auto vs=compileShader(gl,GL_VERTEX_SHADER,kVertexShader,error);
        if(!vs) return false;
        auto fs=compileShader(gl,GL_FRAGMENT_SHADER,kOpticalMotionBlurFragment,error);
        if(!fs) { gl.DeleteShader(vs); return false; }
        BlurProgram p; p.program=gl.CreateProgram(); gl.AttachShader(p.program,vs);
        gl.AttachShader(p.program,fs); gl.LinkProgram(p.program);
        gl.DeleteShader(vs); gl.DeleteShader(fs);
        int linked=0; gl.GetProgramiv(p.program,GL_LINK_STATUS,&linked);
        if(linked!=GL_TRUE) { gl.DeleteProgram(p.program); error="OpenGL Motion Blur program link failed"; return false; }
        gl.GenVertexArrays(1,&p.vao);
        p.image=gl.GetUniformLocation(p.program,"uImage");
        p.field=gl.GetUniformLocation(p.program,"uField");
        p.extent=gl.GetUniformLocation(p.program,"uExtent");
        p.samples=gl.GetUniformLocation(p.program,"uSamples");
        p.scale=gl.GetUniformLocation(p.program,"uScale");
        blurPrograms_[c]=p; return true;
    }
    std::mutex mutex_;std::unordered_map<GLFWwindow*,Program> programs_;std::unordered_map<std::uint64_t,Output> outputs_;std::uint64_t nextLifecycle_=1,nextSubmission_=1;
    std::unordered_map<GLFWwindow*,ViewProgram> viewPrograms_;
    std::unordered_map<GLFWwindow*,BlurProgram> blurPrograms_;
    std::map<std::pair<GLFWwindow*,unsigned>,bool> views_;
    std::set<std::pair<GLFWwindow*,unsigned>> retiredViews_;
    std::uint64_t nextViewGeneration_=0;
};

OpenGlMotionViewFrame::~OpenGlMotionViewFrame()
{
    backend_.releaseView(texture_,context_);
}

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
        glGetIntegerv (GL_SCISSOR_BOX, scissorBox.data());
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
        glScissor (scissorBox[0], scissorBox[1], scissorBox[2], scissorBox[3]);
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
    std::array<int, 11> textureBindings {};
    std::array<int, 11> samplerBindings {};
    std::array<int, 4> viewport {}, scissorBox {};
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

constexpr diffractiontransportnative::ConsumerContract kOpenGlDiffractionConsumer {
    NativeFixtureMaterialBackend::OpenGl, diffractiontransportnative::kTableVersion,
    diffractiontransportnative::kSampleStride, 3, true, true };

std::string diffractionQueryShader()
{
    auto text = opaquevisibility::glslShader();
    smoothglass::replace(text, "opaque", "diffractionQuery");
    smoothglass::replace(text, "Opaque", "DiffractionQuery");
    smoothglass::replace(text, "SceneClosestHit", "DiffractionClosestHit");
    smoothglass::replace(text, "scene", "diffractionScene");
    return "\n#undef SV3\n#undef SV4\n#undef SC_TEXELS\n#undef SC_OUT\n#undef SC_STACK_ARGS\n#undef ST_ARGS\n#undef ST_PASS\n#undef ST_READ\n" + text;
}

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
layout(location=12) in vec3 aPreviousPosition;
uniform int uUsePreviousDeformation;
uniform int uNoteMotion;
uniform sampler2D uPreviousNoteTransforms;
uniform float uSkyBackground;
uniform mat4 uObjectMatrix;
uniform float uTangentHandednessSign;
uniform int uUseInstanceMatrix;
uniform vec4 uRawIdentifiers[64];
uniform int uRawObjectIndex;
flat out vec4 vRawIdentifiers;
flat out uint vDiffractionInstance;
uniform int uDiffractionInstanceCount;
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
uniform vec4 uPreviousVertexSpectrum[16];
uniform vec4 uVertexDisplacementLimits[16];
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
    if (uSkyBackground>0.5) {
        vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2)*2.0-1.0;
        gl_Position=vec4(p,0.0,1.0);
        vWorldPosition=rotateQ(uCameraRotation,vec3(p.x*uProjection.x*uProjection.y,p.y*uProjection.x,-1.0));
        return;
    }
    vDiffractionInstance = uint(uRawObjectIndex * uDiffractionInstanceCount + gl_InstanceID);
    // Mode 1 is one Geometry instance per scene object; mode 2 is one scene
    // object drawn once per instance transform.
    vRawIdentifiers = uRawIdentifiers[uUseInstanceMatrix == 1 ? gl_InstanceID : uRawObjectIndex];
    mat4 objectMatrix = uUseInstanceMatrix != 0
        ? mat4(aInstanceMatrix0, aInstanceMatrix1,
               aInstanceMatrix2, aInstanceMatrix3)
        : uObjectMatrix;
    // Mode 2 has no per-instance appearance. Its identity, color and emission
    // vectors carry the instance's prior world matrix for Motion.
    mat4 previousObjectMatrix = uUseInstanceMatrix == 2
        ? mat4(vec4(aInstanceIdentityColor.xyz, 0.0),
               vec4(aInstanceIdentityColor.w, aInstanceColor.xy, 0.0),
               vec4(aInstanceColor.zw, aInstanceEmission.x, 0.0),
               vec4(aInstanceEmission.yzw, 1.0))
        : objectMatrix;
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
    vec4 previousNote = uNoteMotion != 0
        ? texelFetch(uPreviousNoteTransforms,ivec2(noteIndex,0),0)
        : vec4(noteInstance.xyz,uNoteMeshScale);
    // The modifier block starts the prior pose at aPreviousPosition when a
    // prior deformation exists and applies any vertex program to it.
    vec3 previousLocalWorld = (previousObjectMatrix
        * vec4(previousModifiedPosition * previousNote.w + previousNote.xyz,1.0)).xyz;
    vec3 previousWorld = rotateQ(uMotionRotation, previousLocalWorld) * uMotionTranslationScale.w
        + uMotionTranslationScale.xyz;
    vec3 previousCamera = rotateQ(vec4(-uPreviousCameraRotation.xyz, uPreviousCameraRotation.w),
        previousWorld - uPreviousCameraTranslation.xyz);
    vMotionPreviousClip = uPreviousProjection.w > 0.5 && previousNote.w > 0.0
        && -previousCamera.z > uPreviousProjection.z
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
        * (uUseInstanceMatrix == 1
            ? mix(vec3(1.0), aInstanceIdentityColor.rgb,
                  step(0.5, aInstanceIdentityColor.a))
            : vec3(1.0));
    if (uUseInstanceMatrix == 1) vBaseColor *= aInstanceColor.rgb;
    vLitBase = vBaseColor * uAmbient;
    vEmissive = uEmissive + (uUseInstanceMatrix == 1 ? aInstanceEmission.rgb : vec3(0.0));
    vWorldPosition = world;
    vWorldNormal = worldNormal;
    vWorldTangent = normalize(mat3(objectMatrix) * aTangent.xyz);
    vBitangentSign = aTangent.w * uTangentHandednessSign
        * (uUseInstanceMatrix != 0 ? sign(determinant(mat3(objectMatrix))) : 1.0);
    vMetallicRoughness = vec2(metallic, roughness);
    vLinearDepth = clamp((distance - uProjection.z)
        / max(uProjection.w - uProjection.z, 0.000001), 0.0, 1.0);
    vOpacity = uBaseColor.w * aColor.a * (uUseInstanceMatrix == 1 ? aInstanceColor.a : 1.0);
    vInstanceIdentityColor = uUseInstanceMatrix == 1
        ? aInstanceIdentityColor : vec4(0.0);
}
)glsl";

const char* kFixtureFragmentShaderHead = R"glsl(#version 330 core
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
flat in uint vDiffractionInstance;
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
uniform sampler2D uSelectedRoughResult;
uniform vec4 uSelectedRoughTile;
uniform float uSelectedRoughResolve;
uniform uint uSelectedRoughVolume;
uniform vec4 uMaterialParams;
uniform vec3 uAmbient;
uniform vec4 uSurfaceCoating;
uniform int uMaterialKind;
uniform vec3 uCameraTranslation;
uniform vec4 uSky[5];
uniform float uSkyBackground;
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
uniform int uDiffractionPathIndex;
uniform int uDiffractionTriangleCount;
uniform sampler2D uDiffractionSamples;
uniform vec4 uDiffractionSampleControl;
uniform vec4 uDiffractionSourceDirectionTrace;
uniform vec4 uDiffractionSourceRadiance[2];
uniform float uDiffractionRayMinimum;
layout(location=0) out vec4 outColor;
layout(location=1) out float outLinearDepth;
layout(location=2) out vec4 outRawNormal;
layout(location=3) out vec4 outRawEmission;
layout(location=4) out float outRawMask;
layout(location=5) out uint outRawMaterialId;
layout(location=6) out uint outRawObjectId;
layout(location=7) out vec2 outRawMotion;
)glsl";

const char* kFixturePassCompositeShader = R"glsl(
vec4 compositePass(vec4 color) {
    int count = int(uPassProgramControl.x);
    if (count == 0) return color;
    vec4 values[8];
    #pragma optionNV(unroll none)
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
)glsl";

const char* kFixtureFragmentShaderTail = R"glsl(
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

#if FIXTURE_DIFFRACTION
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

ivec2 propagatingOneDimensionalOrders(vec2 incidentTangent, vec2 grooveDirection,
                                     float period, float wavelength, int lastOrder) {
    float grooveLength = length(grooveDirection);
    if (grooveLength <= 1.0e-8 || abs(period) <= 1.0e-8 || wavelength <= 0.0)
        return ivec2(-lastOrder, lastOrder);
    float step = wavelength * grooveLength / period;
    if (abs(step) <= 1.0e-8)
        return ivec2(-lastOrder, lastOrder);
    // A propagating order needs |m * step - projectedIncident| < 1.
    float projectedIncident = dot(incidentTangent, grooveDirection) / grooveLength;
    float lower = (projectedIncident - 1.0) / step;
    float upper = (projectedIncident + 1.0) / step;
    return ivec2(int(max(-float(lastOrder), floor(min(lower, upper)) - 1.0)),
                 int(min(float(lastOrder), ceil(max(lower, upper)) + 1.0)));
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
        int firstPrimary = -lastOrder;
        int lastPrimary = lastOrder;
        if (!crossed) {
            ivec2 range = propagatingOneDimensionalOrders(
                outgoing.xy, geometry.xy, geometry.z, wavelength, lastOrder);
            firstPrimary = range.x;
            lastPrimary = range.y;
        }
        for (int primary = firstPrimary; primary <= lastPrimary; ++primary) {
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

bool diffractionTableValid() {
    return uDiffractionSampleControl.z==8.0 && uDiffractionSampleControl.w==1.0
        && uDiffractionSampleControl.x>=1.0 && uDiffractionSampleControl.x<=32.0
        && uDiffractionSampleControl.y>=1.0 && uDiffractionSampleControl.y<=32.0
        && floor(uDiffractionSampleControl.x)==uDiffractionSampleControl.x
        && floor(uDiffractionSampleControl.y)==uDiffractionSampleControl.y
        && textureSize(uDiffractionSamples,0).x==512
        && (uDiffractionSampleControl.x+uDiffractionSampleControl.y)*8.0
            <=float(textureSize(uDiffractionSamples,0).x*textureSize(uDiffractionSamples,0).y);
}
vec3 diffractionDiagnostic() { return vec3(uintBitsToFloat(0x7fc00000u)); }
vec4 diffractionSample(int sampleIndex, int lane) {
    int index = sampleIndex * 8 + lane;
    return texelFetch(uDiffractionSamples, ivec2(index % 512, index / 512), 0);
}
uint diffractionId(vec2 words) { return uint(words.x) | (uint(words.y) << 16u); }
vec3 diffractionGeometricNormal() {
    vec3 n = normalize(cross(dFdx(vWorldPosition), dFdy(vWorldPosition)));
    return dot(n, vWorldNormal) < 0.0 ? -n : n;
}
vec3 diffractionLocal(vec3 direction, vec3 tangent, vec3 bitangent, vec3 normal) {
    return vec3(dot(direction,tangent), dot(direction,bitangent), dot(direction,normal));
}
// One directional scalar-wave BSDF per physical wavelength, including zero order.
// The raw authored source and rho are multiplied only after evaluating this BSDF.
float diffractionTriangleBsdf(int band, vec2 uv, vec3 incident, vec3 outgoing) {
    if (incident.z <= 1.0e-6 || outgoing.z <= 0.0) return 0.0;
    vec4 geometry, secondaryGeometry;
    resolveGrooveGeometry(uv, geometry, secondaryGeometry);
    bool crossed = int(secondaryGeometry.w + 0.5) == 2;
    int firstOrder = int(round(uDiffractionMicrostructure.w));
    int lastOrder = int(uDiffractionControl.x + 0.5);
    int profile = int(uDiffractionControl.y + 0.5);
    float wavelength = uDiffractionSpectral[band].x;
    float sigma = max(2.0*uDiffractionRoughness.y, 1.0e-4);
    float radiance = 0.0, efficiencySum = 0.0;
    int firstPrimary = -lastOrder, lastPrimary = lastOrder;
    if (!crossed) {
        ivec2 range = propagatingOneDimensionalOrders(incident.xy, geometry.xy,
            geometry.z, wavelength, lastOrder);
        firstPrimary = range.x; lastPrimary = range.y;
    }
    for (int primary=firstPrimary; primary<=lastPrimary; ++primary) {
        if (primary != 0 && abs(primary)<firstOrder) continue;
        int limit = crossed ? lastOrder : 0;
        for (int secondary=-limit; secondary<=limit; ++secondary) {
            if (secondary != 0 && abs(secondary)<firstOrder) continue;
            vec2 centre = -incident.xy + float(primary)*wavelength/geometry.z*geometry.xy;
            if (crossed) centre += float(secondary)*wavelength/secondaryGeometry.z*secondaryGeometry.xy;
            float square = dot(centre,centre);
            if (square >= 1.0) continue;
            float z = sqrt(max(0.0,1.0-square));
            float phase = 2.0*pi*geometry.w*(incident.z+z)/wavelength;
            vec2 terrace = vec2(cos(phase),sin(phase));
            float efficiency = profileEfficiency(profile,primary,uDiffractionMicrostructure.x,phase,terrace);
            if (crossed) efficiency *= profileEfficiency(profile,secondary,uDiffractionMicrostructure.x,phase,terrace);
            float height = 2.0*pi*uDiffractionRoughness.x*(incident.z+z)/wavelength;
            efficiency *= exp(-height*height)*z/incident.z;
            vec2 delta = outgoing.xy-centre;
            radiance += efficiency*exp(-dot(delta,delta)/(2.0*sigma*sigma))/(2.0*pi*sigma*sigma);
            efficiencySum += efficiency;
        }
    }
    if (!diffractionSceneFinite(efficiencySum) || efficiencySum > 1.0002)
        return uintBitsToFloat(0x7fc00000u);
    return radiance * spectralReflectance(wavelength,int(uDiffractionControl.w+0.5),
        uDiffractionMicrostructure.y,uDiffractionMicrostructure.z);
}
vec3 diffractionBandsToRgb(float bands[8]) {
    vec3 xyz = vec3(0.0); float whiteY = 0.0;
    for (int b=0; b<8; ++b) {
        vec4 spectral = uDiffractionSpectral[b], z = uDiffractionSpectralZ[b];
        xyz += bands[b]*z.y*vec3(spectral.z,spectral.w,z.x);
        whiteY += z.y*spectral.w;
    }
    xyz /= max(whiteY,1.0e-8);
    return max(vec3(3.2406*xyz.x-1.5372*xyz.y-0.4986*xyz.z,
        -0.9689*xyz.x+1.8758*xyz.y+0.0415*xyz.z,
        0.0557*xyz.x-0.2040*xyz.y+1.0570*xyz.z),vec3(0.0));
}
// Exact endpoint checks distinguish an admitted nearer blocker from a stale or
// missing endpoint. Refusal is propagated as diagnostic Color for the whole frame.
bool diffractionEndpoint(DiffractionClosestHit hit, uint objectId, uint materialId,
    uint instanceId, uint triangleId, vec3 normal, float distance, out bool complete) {
    complete = hit.status == 1;
    if (!complete) return false;
    if (hit.distance < distance-uDiffractionRayMinimum) return false;
    complete = hit.objectId==objectId && hit.materialId==materialId
        && hit.instanceId==instanceId && hit.triangleId==triangleId
        && abs(hit.distance-distance)<=uDiffractionRayMinimum
        && dot(hit.outwardNormal,normal)>=1.0-1.0e-5;
    return complete;
}
bool diffractionSegment(int sampleIndex, vec3 pixelNormal, out vec3 toSample,
    out float distance, out bool complete) {
    vec4 p = diffractionSample(sampleIndex,0), ids = diffractionSample(sampleIndex,2);
    vec4 identity = diffractionSample(sampleIndex,3);
    vec3 delta = p.xyz-vWorldPosition;
    distance = length(delta); toSample = delta/distance;
    complete = diffractionSceneFinite(distance) && distance>2.0*uDiffractionRayMinimum
        && distance+uDiffractionRayMinimum<=uDiffractionSourceDirectionTrace.w;
    if (!complete) return false;
    DiffractionClosestHit forward = diffractionSceneClosestHit(vWorldPosition,toSample,
        uDiffractionRayMinimum,distance+uDiffractionRayMinimum,uDiffractionQueryControl);
    DiffractionClosestHit backward = diffractionSceneClosestHit(p.xyz,-toSample,
        uDiffractionRayMinimum,distance+uDiffractionRayMinimum,uDiffractionQueryControl);
    if (forward.status<0 || backward.status<0 || forward.status>1 || backward.status>1) {
        complete=false; return false;
    }
    bool forwardComplete, backwardComplete;
    bool forwardVisible = diffractionEndpoint(forward,diffractionId(ids.xy),diffractionId(ids.zw),
        uint(identity.x),uint(identity.y),diffractionSample(sampleIndex,1).xyz,distance,forwardComplete);
    bool backwardVisible = diffractionEndpoint(backward,outRawObjectId,outRawMaterialId,
        vDiffractionInstance,uint(gl_PrimitiveID % uDiffractionTriangleCount),pixelNormal,distance,backwardComplete);
    complete = forwardComplete && backwardComplete;
    return forwardVisible && backwardVisible;
}
float diffractionSourceVisibility(vec3 position, out bool complete) {
    DiffractionClosestHit hit = diffractionSceneClosestHit(position,uDiffractionSourceDirectionTrace.xyz,
        uDiffractionRayMinimum,uDiffractionSourceDirectionTrace.w,uDiffractionQueryControl);
    complete=hit.status==0 || hit.status==1;
    return hit.status==0 ? 1.0 : 0.0;
}
vec3 evaluateTriangleDiffraction(vec3 tangent, vec3 bitangent, vec3 normal, vec3 outgoing) {
    if (!diffractionTableValid()) return diffractionDiagnostic();
    bool complete;
    float direct = diffractionSourceVisibility(vWorldPosition,complete);
    if (!complete) return diffractionDiagnostic();
    vec3 incident = diffractionLocal(uDiffractionSourceDirectionTrace.xyz,tangent,bitangent,normal);
    float bands[8];
    #pragma optionNV(unroll none)
    for (int b=0; b<8; ++b) bands[b] = direct*uDiffractionSourceRadiance[b/4][b%4]
        *max(incident.z,0.0)*diffractionTriangleBsdf(b,vUv,incident,outgoing);
    #pragma optionNV(unroll none)
    for (int r=0; r<32; ++r) {
        if (float(r)>=uDiffractionSampleControl.x) break;
        vec3 direction; float distance;
        bool visible = diffractionSegment(r,normal,direction,distance,complete);
        if (!complete) return diffractionDiagnostic();
        float source = diffractionSourceVisibility(diffractionSample(r,0).xyz,complete);
        if (!complete) return diffractionDiagnostic();
        if (!visible) continue;
        vec3 receiverNormal = diffractionSample(r,1).xyz;
        float receiverCosine = max(dot(receiverNormal,-direction),0.0);
        float lightCosine = max(dot(receiverNormal,uDiffractionSourceDirectionTrace.xyz),0.0)*source;
        vec3 local = diffractionLocal(direction,tangent,bitangent,normal);
        float weight = diffractionSample(r,0).w*receiverCosine*max(local.z,0.0)/(pi*distance*distance);
        #pragma optionNV(unroll none)
        for (int b=0; b<8; ++b) bands[b] += weight*lightCosine*uDiffractionSourceRadiance[b/4][b%4]
            *diffractionSample(r,4+b/4)[b%4]*diffractionTriangleBsdf(b,vUv,local,outgoing);
    }
    for (int b=0; b<8; ++b) if (!diffractionSceneFinite(bands[b])) return diffractionDiagnostic();
    return diffractionBandsToRgb(bands);
}
float diffractionScatterCoverage(vec2 uv) {
    float coverage = uDiffractionSpatialCounts.y == 0.0
        ? 1.0 : uintBitsToFloat(floatBitsToUint(uDiffractionPathKindAndBounce.w));
    float occupancy = uDiffractionSpatialCounts.y == 0.0 ? 1.0 : 0.0;
    for (int i=0; i<5; ++i) {
        if (float(i)>=uDiffractionSpatialCounts.y) break;
        vec4 rectangle=uDiffractionOccupancyRectangles[i];
        occupancy=max(occupancy,float(uv.x>=rectangle.x && uv.y>=rectangle.y
            && uv.x<=rectangle.z && uv.y<=rectangle.w));
    }
    if (uDiffractionSpatialCounts.x>0.0) {
        vec4 lower=mix(uDiffractionFoilField[0],uDiffractionFoilField[1],clamp(uv.x,0.0,1.0));
        vec4 upper=mix(uDiffractionFoilField[2],uDiffractionFoilField[3],clamp(uv.x,0.0,1.0));
        occupancy*=mix(lower.w,upper.w,clamp(uv.y,0.0,1.0));
    }
    return clamp(coverage*occupancy,0.0,1.0);
}
vec3 evaluateSpectralReceiver(vec3 normal, out bool receiver) {
    receiver=true;
    if (!diffractionTableValid()) return diffractionDiagnostic();
    receiver=false;
    int receiverIndex=-1;
    #pragma optionNV(unroll none)
    for (int r=0; r<32; ++r) {
        if (float(r)>=uDiffractionSampleControl.x) break;
        vec4 ids=diffractionSample(r,2);
        if (diffractionId(ids.xy)==outRawObjectId && diffractionId(ids.zw)==outRawMaterialId) {
            receiverIndex=r; receiver=true; break;
        }
    }
    if (!receiver) return vec3(0.0);
    bool complete;
    float source=diffractionSourceVisibility(vWorldPosition,complete);
    if (!complete) return diffractionDiagnostic();
    float bands[8];
    #pragma optionNV(unroll none)
    for (int b=0; b<8; ++b) bands[b]=source*max(dot(normal,uDiffractionSourceDirectionTrace.xyz),0.0)
        *uDiffractionSourceRadiance[b/4][b%4];
    #pragma optionNV(unroll none)
    for (int g=0; g<32; ++g) {
        if (float(g)>=uDiffractionSampleControl.y) break;
        int index=int(uDiffractionSampleControl.x)+g;
        vec3 direction; float distance;
        bool visible=diffractionSegment(index,normal,direction,distance,complete);
        if (!complete) return diffractionDiagnostic();
        float sourceVisible=diffractionSourceVisibility(diffractionSample(index,0).xyz,complete);
        if (!complete) return diffractionDiagnostic();
        if (!visible) continue;
        vec3 n=diffractionSample(index,1).xyz;
        vec4 t=diffractionSample(index,6),bt=diffractionSample(index,7);
        vec3 localLight=diffractionLocal(uDiffractionSourceDirectionTrace.xyz,t.xyz,bt.xyz,n);
        vec3 outgoing=diffractionLocal(-direction,t.xyz,bt.xyz,n);
        float weight=diffractionSample(index,0).w*max(dot(normal,direction),0.0)
            *max(outgoing.z,0.0)/(distance*distance)*diffractionScatterCoverage(vec2(t.w,bt.w));
        #pragma optionNV(unroll none)
        for (int b=0; b<8; ++b) bands[b]+=weight*sourceVisible*uDiffractionSourceRadiance[b/4][b%4]
            *max(localLight.z,0.0)*diffractionTriangleBsdf(b,vec2(t.w,bt.w),localLight,outgoing);
    }
    for (int b=0; b<8; ++b) {
        bands[b]*=diffractionSample(receiverIndex,4+b/4)[b%4]/pi;
        if (!diffractionSceneFinite(bands[b])) return diffractionDiagnostic();
    }
    return diffractionBandsToRgb(bands);
}

vec3 evaluateDiffraction() {
    vec3 normal = uDiffractionSampleControl.w == 1.0
        ? diffractionGeometricNormal() : normalize(vWorldNormal);
    vec3 dpdx = dFdx(vWorldPosition);
    vec3 dpdy = dFdy(vWorldPosition);
    vec2 duvdx = dFdx(vUv);
    vec2 duvdy = dFdy(vUv);
    float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    if (abs(determinant) <= 1.0e-8)
        return uDiffractionSampleControl.w == 1.0 ? diffractionDiagnostic() : vec3(0.0);
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
    if (uDiffractionSampleControl.w == 1.0)
        return evaluateTriangleDiffraction(tangent,bitangent,normal,outgoing);
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
    vec4 oneDimensionalGeometry;
    vec4 unusedSecondaryGeometry;
    if (!crossedTwoDimensional)
        resolveGrooveGeometry(vUv, oneDimensionalGeometry, unusedSecondaryGeometry);
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
            ivec2 range = propagatingOneDimensionalOrders(
                incident.xy, oneDimensionalGeometry.xy, oneDimensionalGeometry.z,
                wavelength, lastOrder);
            for (int primaryOrder = range.x; primaryOrder <= min(-firstOrder, range.y); ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, false, wavelength,
                    incident, outgoing, profile, duty, sigma);
            for (int primaryOrder = max(firstOrder, range.x); primaryOrder <= range.y; ++primaryOrder)
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

#endif

// Direct base GGX contract: shared/BaseSurfaceReflection.h. Ambient and coat
// retain their existing independent approximations. No radiance/output clamp.
vec3 baseSurfaceUnit(vec3 value) {
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12 ? value * inversesqrt(lengthSquared) : vec3(0.0);
}
vec3 baseSurfaceDirect(vec3 base, float metallic, float roughness,
                     float dielectric, float transmission,
                     vec3 normal, vec3 toLight, vec3 toView) {
    vec3 n = baseSurfaceUnit(normal);
    vec3 l = baseSurfaceUnit(toLight);
    vec3 v = baseSurfaceUnit(toView);
    float nl = clamp(dot(n, l), 0.0, 1.0);
    float nv = clamp(dot(n, v), 0.0, 1.0);
    if (nl <= 0.0 || nv <= 0.0) return vec3(0.0);
    vec3 h = baseSurfaceUnit(l + v);
    float nh = clamp(dot(n, h), 0.0, 1.0);
    float vh = clamp(dot(v, h), 0.0, 1.0);
    float alpha = max(roughness * roughness, 0.0025);
    float a2 = alpha * alpha;
    float denominator = (1.0 - nh * nh) + a2 * nh * nh;
    float distribution = a2 / (3.14159265358979323846 * denominator * denominator);
    float visibility = 1.0 / ((nl + sqrt(a2 + (1.0 - a2) * nl * nl))
                            * (nv + sqrt(a2 + (1.0 - a2) * nv * nv)));
    vec3 f0 = mix(vec3(dielectric), base, metallic);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    vec3 diffuse = (1.0 - fresnel) * (1.0 - metallic) * base
                  * (1.0 - transmission) / 3.14159265358979323846;
    return (distribution * visibility * fresnel + diffuse) * nl;
}
void main() {
    if(uSelectedRoughResolve>0.5) {
        ivec2 nativePixel=ivec2(floor(gl_FragCoord.xy));
        ivec2 pixel=ivec2(nativePixel.x,int(uRoughCameraControl.w)-1-nativePixel.y);
        ivec2 local=pixel-ivec2(uSelectedRoughTile.xy);
        if(any(lessThan(local,ivec2(0))) || any(greaterThanEqual(local,ivec2(uSelectedRoughTile.zw)))) discard;
        vec4 result=texelFetch(uSelectedRoughResult,ivec2(local.x*3,local.y),0);
        uvec4 owners=uvec4(texelFetch(uSelectedRoughResult,ivec2(local.x*3+1,local.y),0));
        uint object=owners.x|(owners.y<<16u);
        if(object!=uSelectedRoughVolume) discard;
        // This Color-only submission samples the completed native AOV images.
        // Shared-edge raster coverage cannot choose a different transport seed.
        outLinearDepth=texelFetch(uMetallicRoughnessTexture,nativePixel,0).r;
        outRawNormal=texelFetch(uBaseTexture,nativePixel,0);
        outRawMask=texelFetch(uNormalTexture,nativePixel,0).r;
        outRawEmission=texelFetch(uOcclusionTexture,nativePixel,0);
        outRawMotion=texelFetch(uEmissiveTexture,nativePixel,0).xy;
        outRawObjectId=object;outRawMaterialId=owners.z|(owners.w<<16u);
        outColor=compositePass(vec4(result.rgb,1.0));
        return;
    }
    if (uSkyBackground>0.5) {
        outLinearDepth=1.0; outRawNormal=vec4(0.0); outRawEmission=vec4(0.0);
        outRawMask=0.0; outRawMaterialId=0u; outRawObjectId=0u; outRawMotion=vec2(0.0);
        vec3 radiance=uSky[0].w>1.5 ? environmentImageRadiance(vWorldPosition,uSky[0],uSky[1])
            : analyticSkyRadiance(vWorldPosition,uSky[0],uSky[1],uSky[2],uSky[3],uSky[4]);
        outColor=compositePass(vec4(radiance,1.0));
        return;
    }
    outLinearDepth = vLinearDepth;
    // Only graph Frames may be native scene attachments. Embedded glTF images
    // keep their authored UVs and raster storage; w=2 declares a bottom-first Frame.
    vec2 baseUv = uTexturePresence.w == 2.0 ? vec2(vUv.x, 1.0 - vUv.y) : vUv;
    outRawNormal = vec4(normalize(vWorldNormal), 1.0);
    outRawEmission = vec4(0.0);
    outRawMask = 1.0;
    outRawMaterialId = uint(vRawIdentifiers.x) | (uint(vRawIdentifiers.y) << 16u);
    outRawObjectId = uint(vRawIdentifiers.z) | (uint(vRawIdentifiers.w) << 16u);
    outRawMotion = clamp((vMotionCurrentClip.xy / vMotionCurrentClip.w
        - vMotionPreviousClip.xy / vMotionPreviousClip.w) * 0.5 * uMotionViewport.xy,
        vec2(-65504.0), vec2(65504.0));
#if FIXTURE_DIFFRACTION
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
        vec3 diffraction = uDiffractionSampleControl.w == 1.0 ? evaluateDiffraction()
            : (uDiffractionSpatialCounts.x == 0.0
                || evaluationIndex < evaluationSchedule.z)
            ? max(evaluateDiffraction(), vec3(0.0)) : vLitBase;
        if (any(isnan(diffraction)) || any(isinf(diffraction))) {
            outColor = vec4(diffraction, 1.0); return;
        }
        outColor = vec4((uDiffractionPathIndex == 0 ? vLitBase * (1.0 - clamp(occupancy, 0.0, 1.0))
            : vec3(0.0)) + diffraction * clamp(occupancy, 0.0, 1.0), 1.0);
        if (uTexturePresence.w > 0.5) {
            vec4 imagery = texture(uBaseTexture, baseUv);
            if (imagery.a <= 0.0) discard;
            outColor *= vec4(uTexturePresence.w >= 3.0 ? imagery.rgb : srgbToLinear(imagery.rgb), imagery.a);
        }
        outColor.rgb = vNoteAppearance.rgb * (outColor.rgb
            + (uDiffractionPathIndex == 0 ? vec3(vNoteAppearance.a) : vec3(0.0)));
        return;
    }
    if (uDiffractionSampleControl.w == 1.0) {
        bool receiver;
        vec3 color = evaluateSpectralReceiver(diffractionGeometricNormal(),receiver);
        if (receiver) { outColor=vec4(color,1.0); return; }
    }
#endif
    vec4 texel = texture(uBaseTexture, baseUv);
    if (uTexturePresence.w < 3.0) texel.rgb = srgbToLinear(texel.rgb);
    // Thin coverage retains normal-incidence F0; direct base reflection is
    // GGX below. Coat remains the existing constant layer, not a coat BRDF.
    // Transmission reveals the composite behind the surface, not volume refraction.
    float interfaceRatio = (uSurfaceCoating.y - 1.0) / (uSurfaceCoating.y + 1.0);
    float dielectric = uSurfaceCoating.w > 0.5 ? interfaceRatio * interfaceRatio : 0.04;
    float transmission = uSurfaceCoating.w > 0.5 ? uSurfaceCoating.x : 0.0;
    float coat = uSurfaceCoating.w > 0.5 ? 0.04 * uSurfaceCoating.z : 0.0;
    float coverage = uGlassTransport.x > 0.5 ? 1.0 : 1.0 - transmission * (1.0 - dielectric);
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
    // Private receipt pass reuses this exact linked program and draw schedule.
    // Locations 0, 2 and 3 are routed to frame-owned RGBA32F scratch targets.
    if (imageOutput == -1) {
#if FIXTURE_SMOOTH_GLASS
        if(uGlassTransport.x==1.0 && uRoughCameraControl.x==2.0) discard;
        if ((uGlassTransport.x != 1.0 && uGlassTransport.x != 2.0) || roughness <= 0.0) discard;
        vec3 terminal, masked, tail;
        int status, queryCount, microEvents=0;
        if (uGlassTransport.x == 2.0)
            roughThinInterfaceTransport(uCameraTranslation,vWorldPosition,normal,roughness,true,
                terminal,masked,tail,status,queryCount);
        else
            roughCameraTransport(uCameraTranslation,vWorldPosition,roughness,true,
                vec2(gl_FragCoord.x,uRoughCameraControl.w-gl_FragCoord.y)+uRoughCameraControl.yz,
                terminal,masked,tail,status,queryCount,microEvents);
        outColor = vec4(terminal,float(status));
        outRawNormal = vec4(masked,float(queryCount));
        outRawEmission = vec4(tail,uGlassTransport.x==1.0 && uRoughCameraControl.x==2.0?float(microEvents):float(status));
        return;
#else
        // Without smooth glass, uGlassTransport.x is 0 for every draw and this block always discards.
        discard;
#endif
    }
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
#if FIXTURE_SMOOTH_GLASS
    if (uGlassTransport.x > 0.5) {
        if(uGlassTransport.x==1.0 && roughness>0.0 && uRoughCameraControl.x==2.0) {
            ivec2 pixel=ivec2(floor(vec2(gl_FragCoord.x,uRoughCameraControl.w-gl_FragCoord.y)));
            ivec2 local=pixel-ivec2(uSelectedRoughTile.xy);
            if(any(lessThan(local,ivec2(0))) || any(greaterThanEqual(local,ivec2(uSelectedRoughTile.zw)))) discard;
            vec4 result=texelFetch(uSelectedRoughResult,ivec2(local.x*3,local.y),0);
            uvec4 owners=uvec4(texelFetch(uSelectedRoughResult,ivec2(local.x*3+1,local.y),0));
            uvec2 primary=uvec2(owners.x|(owners.y<<16u),owners.z|(owners.w<<16u));
            if(primary.x!=outRawObjectId || primary.y!=outRawMaterialId) discard;
            bool complete=result.w==33.0;
            bool representable=glassSceneFinite3(result.rgb) && max(max(abs(result.x),abs(result.y)),abs(result.z))<=65504.0;
            outRawMask=complete && representable?1.0:0.0;
            outRawNormal.w=complete && representable?1.0:representable?float(glassRoughReason(int(result.w))):-2.0;
            // Raster retains the released AOV shading. Only the selected
            // fullscreen resolve consumes transport radiance, once per pixel.
            outColor=vec4(0.0,0.0,0.0,1.0);
            return;
        }
        bool complete;
        int glassStatus=GLASS_SMOOTH_STATUS_UNCLASSIFIED;
        vec3 glass;
        if (uGlassTransport.x == 2.0 && roughness == 0.0)
            glass = thinInterfaceTransport(uCameraTranslation,vWorldPosition,normal,complete,glassStatus);
        else if (uGlassTransport.x == 2.0) {
            vec3 terminal, masked, tail;
            int status, queryCount;
            glass = roughThinInterfaceTransport(uCameraTranslation,vWorldPosition,normal,roughness,false,
                terminal,masked,tail,status,queryCount);
            complete = status == 1;
            glassStatus = glassRoughReason(status);
        }
        else if (roughness == 0.0)
            glass = glassTransport(uCameraTranslation,vWorldPosition,complete,glassStatus);
        else {
            vec3 terminal, masked, tail;
            int status, queryCount, microEvents;
            glass = roughCameraTransport(uCameraTranslation,vWorldPosition,roughness,false,
                vec2(gl_FragCoord.x,uRoughCameraControl.w-gl_FragCoord.y)+uRoughCameraControl.yz,
                terminal,masked,tail,status,queryCount,microEvents);
            complete = status == (uRoughCameraControl.x==2.0?33:1);
            glassStatus = glassRoughReason(status);
        }
        bool representable = glassSceneFinite3(glass)
            && max(max(abs(glass.x),abs(glass.y)),abs(glass.z)) <= 65504.0;
        outRawMask = complete && representable ? 1.0 : 0.0;
        outRawNormal.w = complete && representable ? 1.0 : representable ? float(glassStatus) : -2.0;
        outColor = compositePass(vec4(complete && representable ? glass : vec3(0.0),1.0));
        return;
    }
#endif
    vec3 lighting = uAmbient;
    vec3 directBase = vec3(0.0);
    vec3 directCoat = vec3(0.0);
    vec3 skyBase = vec3(0.0);
    vec3 toView = uCameraTranslation - vWorldPosition;
    #pragma optionNV(unroll none)
    for (int lightIndex = 0; lightIndex < 16; ++lightIndex) {
        if (lightIndex >= uLightCount) break;
        int kind = uLightKinds[lightIndex];
        if (kind == 2) {
            if (uSky[0].w>0.5) {
                if (transmission==0.0 && uAlphaModeCutoff.x==0.0 && alpha>=1.0)
                    skyBase+=uSky[0].w>1.5
                        ? environmentImageBase(vBaseColor*texel.rgb,metallic,roughness,dielectric,normal,toView,uSky[0],uSky[1])
                        : analyticSkyBase(vBaseColor*texel.rgb,metallic,roughness,dielectric,
                            normal,toView,uSky[0],uSky[1],uSky[2],uSky[3],uSky[4]);
            } else lighting += uLightColors[lightIndex].rgb * uLightColors[lightIndex].a;
            continue;
        }
        vec3 delta = uLightPositions[lightIndex].xyz - vWorldPosition;
        vec3 toLight = kind == 0 ? rotateQ(uLightRotations[lightIndex], vec3(0.0, 0.0, 1.0)) : baseSurfaceUnit(delta);
        float attenuation = 1.0;
        if (kind == 1 || kind == 3) {
            float lightDistance = length(delta);
            attenuation = 1.0 / max(lightDistance * lightDistance, 1.0);
            float range = uLightPositions[lightIndex].w;
            if (range > 0.0) attenuation *= pow(clamp(1.0 - lightDistance / range, 0.0, 1.0), 2.0);
            if (kind == 3) {
                vec3 spotDirection = rotateQ(uLightRotations[lightIndex], vec3(0.0, 0.0, -1.0));
                float coneCosine = dot(spotDirection, baseSurfaceUnit(vWorldPosition - uLightPositions[lightIndex].xyz));
                attenuation *= smoothstep(uLightCones[lightIndex].y, uLightCones[lightIndex].x, coneCosine);
            }
        }
        vec3 radiance = uLightColors[lightIndex].rgb * uLightColors[lightIndex].a * attenuation;
        float visibility = uAlphaModeCutoff.x == 0.0 && transmission == 0.0 && alpha >= 1.0
            ? opaqueVisibility(vWorldPosition,cross(dFdx(vWorldPosition),dFdy(vWorldPosition)),
                toLight,kind == 0 ? 3.402823e38 : length(delta),uOpaqueControl) : 1.0;
        directBase += radiance * baseSurfaceDirect(vBaseColor * texel.rgb, metallic, roughness,
            dielectric, transmission, normal, toLight, toView) * visibility;
        directCoat += radiance * max(dot(normal, baseSurfaceUnit(toLight)), 0.0);
    }
    float diffuseWeight = (1.0 - metallic) * (1.0 - 0.5 * roughness) * (1.0 - transmission);
    float specularWeight = mix(dielectric, 1.0, metallic) * (1.0 - roughness);
    float occlusion = uTexturePresence.z > 0.5
        ? texture(uOcclusionTexture, vUv).r : 1.0;
    outColor = vec4((vBaseColor * texel.rgb * lighting
                    * (diffuseWeight + specularWeight) * (1.0 - coat) * occlusion
                    + (directBase + skyBase) * (1.0 - coat) * occlusion
                    + (lighting + directCoat) * coat + emissive) / max(coverage, 0.000001), alpha);
    outColor.rgb = vNoteAppearance.rgb * (outColor.rgb + vec3(vNoteAppearance.a));
    if (vInstanceIdentityColor.a > 0.5)
        outColor = vec4(vInstanceIdentityColor.rgb, 1.0);
    outColor = compositePass(outColor);
}
)glsl";

const char* kDiffractionResolveVertexShader = R"glsl(#version 330 core
void main() {
    vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

const char* kDiffractionResolveFragmentShaderHead = R"glsl(#version 330 core
uniform sampler2D uAccumulatedColor;
uniform sampler2D uLinearDepth;
uniform sampler2D uRawNormal;
uniform sampler2D uRawEmission;
uniform sampler2D uRawMask;
uniform usampler2D uRawMaterialId;
uniform usampler2D uRawObjectId;
uniform sampler2D uRawMotion;
uniform vec4 uPassProgramControl;
uniform vec4 uPassModes[8];
uniform vec4 uPassColors[8];
uniform vec4 uPassReferences[8];
uniform vec4 uPassTransforms[8];
uniform vec4 uMotionViewport;
uniform int uImageOutput;
layout(location=0) out vec4 outColor;
float outLinearDepth;
vec4 outRawNormal;
vec4 outRawEmission;
float outRawMask;
uint outRawMaterialId;
uint outRawObjectId;
vec2 outRawMotion;
)glsl";

const char* kDiffractionResolveFragmentShaderTail = R"glsl(
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    outRawMask = texelFetch(uRawMask, pixel, 0).r;
    if (outRawMask <= 0.0) discard;
    outLinearDepth = texelFetch(uLinearDepth, pixel, 0).r;
    outRawNormal = texelFetch(uRawNormal, pixel, 0);
    outRawEmission = texelFetch(uRawEmission, pixel, 0);
    outRawMaterialId = texelFetch(uRawMaterialId, pixel, 0).r;
    outRawObjectId = texelFetch(uRawObjectId, pixel, 0).r;
    outRawMotion = texelFetch(uRawMotion, pixel, 0).rg;
    vec4 color = texelFetch(uAccumulatedColor, pixel, 0);
    if (any(isnan(color)) || any(isinf(color))) { outColor = color; return; }
    if (uImageOutput != 0) {
        vec3 image = vec3(0.0);
        if (uImageOutput == 1) image = vec3(outLinearDepth);
        if (uImageOutput == 2) image = outRawNormal.xyz * 0.5 + 0.5;
        if (uImageOutput == 3) image = vec3(clamp(vec2(0.5) + outRawMotion
            / uMotionViewport.xy, 0.0, 1.0), 0.5);
        if (uImageOutput == 4) image = outRawEmission.rgb;
        if (uImageOutput == 5) image = vec3(outRawMask);
        if (uImageOutput == 6 || uImageOutput == 7) {
            uint hash = (uImageOutput == 6 ? outRawMaterialId : outRawObjectId) * 0x9e3779b9u;
            hash ^= hash >> 16u;
            image = vec3(32u + (hash & 191u), 32u + ((hash >> 8u) & 191u),
                32u + ((hash >> 16u) & 191u)) / 255.0;
        }
        outColor = vec4(image, color.a);
    } else {
        outColor = compositePass(color);
    }
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
    gl.DeleteShader (shader);
    error = std::string ("OpenGL fixture shader compilation failed: ")
        + std::string (log, static_cast<std::size_t> (std::max (length, 0)));
    return 0;
}

// GL seam for the fixture program binary cache. The entry points are optional.
// A missing pointer reports zero binary formats, which disables the cache for
// that preparation. The binary calls are compiled out on macOS, where the
// entry points are not declared.
class FixtureProgramBinaryDriver final : public arbitglprogramcache::Driver
{
public:
    explicit FixtureProgramBinaryDriver (const arbitgl::GlFuncs& gl) : gl_ (gl) {}

    int binaryFormatCount() const override
    {
#if defined(__APPLE__)
        (void) gl_;
        return 0;
#else
        if (gl_.ProgramBinary == nullptr || gl_.GetProgramBinary == nullptr
            || gl_.ProgramParameteri == nullptr)
            return 0;
        GLint count = 0;
        glGetIntegerv (GL_NUM_PROGRAM_BINARY_FORMATS, &count);
        return static_cast<int> (count);
#endif
    }

    arbitglprogramcache::DriverIdentity identity() const override
    {
        arbitglprogramcache::DriverIdentity result;
        result.vendor = text (GL_VENDOR);
        result.renderer = text (GL_RENDERER);
        result.version = text (GL_VERSION);
        return result;
    }

    std::uint32_t createProgram() override { return gl_.CreateProgram(); }

    void deleteProgram (std::uint32_t program) override { gl_.DeleteProgram (program); }

    void setRetrievableHint (std::uint32_t program) override
    {
#if defined(__APPLE__)
        (void) program;
#else
        gl_.ProgramParameteri (program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
#endif
    }

    bool loadBinary (std::uint32_t program, std::uint32_t binaryFormat,
                     const std::uint8_t* data, std::size_t size) override
    {
#if defined(__APPLE__)
        (void) program; (void) binaryFormat; (void) data; (void) size;
        return false;
#else
        gl_.ProgramBinary (program, binaryFormat, data, static_cast<GLsizei> (size));
        GLint linked = GL_FALSE;
        gl_.GetProgramiv (program, GL_LINK_STATUS, &linked);
        if (linked == GL_TRUE)
            return true;
        // A rejected binary can leave errors queued. Drain them so the compile
        // fallback and later render checks start clean.
        for (int i = 0; i < 8 && glGetError() != GL_NO_ERROR; ++i) {}
        return false;
#endif
    }

    bool retrieveBinary (std::uint32_t program, std::uint32_t& binaryFormat,
                         std::vector<std::uint8_t>& payload) override
    {
#if defined(__APPLE__)
        (void) program; (void) binaryFormat; (void) payload;
        return false;
#else
        GLint length = 0;
        gl_.GetProgramiv (program, GL_PROGRAM_BINARY_LENGTH, &length);
        if (length <= 0)
            return false;
        std::vector<std::uint8_t> buffer (static_cast<std::size_t> (length));
        GLsizei written = 0;
        GLenum format = 0;
        gl_.GetProgramBinary (program, length, &written, &format, buffer.data());
        if (written <= 0 || written > length)
            return false;
        buffer.resize (static_cast<std::size_t> (written));
        payload.swap (buffer);
        binaryFormat = static_cast<std::uint32_t> (format);
        return true;
#endif
    }

private:
    static std::string text (GLenum name)
    {
        const auto* value = glGetString (name);
        return value != nullptr ? std::string (reinterpret_cast<const char*> (value)) : std::string();
    }

    const arbitgl::GlFuncs& gl_;
};

// Prints the fixture cache state once per process, on the first preparation
// that constructs the cache. The format count is queried once, here, and the
// caller holds fixtureMutex(), so the flag needs no further locking.
void reportFixtureProgramCacheOnce (const arbitglprogramcache::ProgramBinaryCache& binaryCache,
                                    const FixtureProgramBinaryDriver& binaryDriver,
                                    const arbitgl::GlFuncs& gl)
{
    static std::once_flag reported;
    std::call_once (reported, [&]
    {
        arbitglprogramcache::EntryPointReport entries;
       #if ! defined (__APPLE__)
        entries.programBinary = gl.ProgramBinary != nullptr;
        entries.getProgramBinary = gl.GetProgramBinary != nullptr;
        entries.programParameteri = gl.ProgramParameteri != nullptr;
       #else
        (void) gl;
       #endif
        const auto line = arbitglprogramcache::stateLine (binaryCache.directory(),
            binaryDriver.binaryFormatCount(), entries);
        std::fprintf (stderr, "%s\n", line.c_str());
    });
}

// Ordinary camera V2 has its own capability. The GL3.3 consumer remains valid
// for v1 and exact zero; compute availability never changes that baseline.
bool roughSelectedComputeCapability(arbitgl::GlFuncs& gl)
{
#if defined(__APPLE__)
    (void)gl;return false;
#else
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4 || (major==4 && minor<3)) return false;
    arbitgl::loadGl43Functions(gl);
    if(!gl.DispatchCompute || !gl.MemoryBarrier || !gl.BindBufferBase || !gl.GetIntegeri_v
        || !gl.GetInteger64v || !gl.GetInteger64i_v || !gl.BindBufferRange || !gl.GetBufferParameteri64v
        || !gl.MapBufferRange || !gl.UnmapBuffer || !gl.FenceSync || !gl.ClientWaitSync || !gl.DeleteSync) return false;
    int blocks=0,bindings=0,groups=0,threads=0,local=0,units=0,texture=0,computeTextures=0;
    GLint64 bytes=0;gl.GetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&bytes);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,&blocks);glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
    gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&groups);gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0,&local);
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&threads);glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS,&units);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&texture);glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS,&computeTextures);
    return blocks>=3 && bindings>=3 && computeTextures>=8 && groups>=int(roughglasstiles::kMaximumMultipleScatterTilePixels)
        && threads>=1 && local>=1 && units>10 && texture>=int(3*roughglasstiles::kMaximumMultipleScatterTilePixels)
        && bytes>=GLint64(roughglasstiles::kMaximumMultipleScatterTilePixels*96) && glGetError()==GL_NO_ERROR;
#endif
}

void storeFixtureQuaternion (
    std::array<float, 4>& destination,
    const HarmonicMIDI::grid::SceneQuaternion& source) noexcept;

struct OpenGlQueryProgram;
struct OpenGlQueryContext final
{
    GLFWwindow* context=nullptr;
    std::uint64_t generation=0;
    int major=0,minor=0,profile=0;
    bool active=true;
    OpenGlQueryProgram* programs=nullptr;
    std::atomic<std::uint64_t> deferredBytes{0};
    void drainDeferredUnlocked() noexcept;
    void retireUnlocked() noexcept;
};
// Access, unlink and retirement all hold fixtureMutex with this context current.
// The existing context cache owns the lifetime; intrusive links allocate no
// additional registry and let teardown retire every still-leased native ID.
struct OpenGlQueryProgram final
{
    arbitgl::GlFuncs gl{};
    std::shared_ptr<OpenGlQueryContext> context;
    std::shared_ptr<const OpenGlQueryArithmeticProgramIdentity> identity;
    std::array<std::shared_ptr<OpenGlQueryProgram>,2> parents{};
    std::shared_ptr<OpenGlQueryProgram> deferredLease;
    OpenGlQueryProgram* previous=nullptr;
    OpenGlQueryProgram* next=nullptr;
    unsigned program=0;
    std::array<unsigned,3> directBuffers{};
    unsigned directionalInputBuffer=0; // Cohort-only immutable source input.
    GLsync directFence=nullptr;
    std::array<char,96> label{};
    std::uint64_t retainedBytes=0;
    bool registered=false;
    bool preciseSceneQuery=true;
    bool liveUnlocked() const noexcept {
#if defined(__APPLE__)
        return false;
#else
        if(!program||!context||!context->active||context->context!=glfwGetCurrentContext())return false;
        const auto isProgram=reinterpret_cast<PFNGLISPROGRAMPROC>(glfwGetProcAddress("glIsProgram"));
        const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
        if(!isProgram||!getLabel||!isProgram(program))return false;
        int linked=0,deleted=0;gl.GetProgramiv(program,GL_LINK_STATUS,&linked);gl.GetProgramiv(program,GL_DELETE_STATUS,&deleted);
        std::array<char,96> actual{};int length=0;getLabel(GL_PROGRAM,program,int(actual.size()),&length,actual.data());
        return linked==GL_TRUE&&deleted==GL_FALSE&&actual==label&&glGetError()==GL_NO_ERROR;
#endif
    }
    ~OpenGlQueryProgram() {
        if(registered) {
            if(previous)previous->next=next;else if(context)context->programs=next;
            if(next)next->previous=previous;
        }
        releaseUnlocked();
    }
    void releaseUnlocked() noexcept {
        if(context&&context->active&&glfwGetCurrentContext()==context->context) {
#if !defined(__APPLE__)
            const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
            const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
            for(auto buffer:{directBuffers[0],directBuffers[1],directBuffers[2],directionalInputBuffer})if(buffer&&isBuffer&&getLabel&&isBuffer(buffer)) {
                std::array<char,96> actual{};int length=0;
                getLabel(GL_BUFFER,buffer,int(actual.size()),&length,actual.data());
                if(actual==label)gl.DeleteBuffers(1,&buffer);
            }
            const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
            if(directFence&&isSync&&isSync(directFence))gl.DeleteSync(directFence);
#endif
            directBuffers={};directionalInputBuffer=0;directFence=nullptr;
        }
        if(program&&context&&context->active&&glfwGetCurrentContext()==context->context) {
#if !defined(__APPLE__)
            const auto isProgram=reinterpret_cast<PFNGLISPROGRAMPROC>(glfwGetProcAddress("glIsProgram"));
            const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
            // A deleted/reused numeric handle must never delete another owner.
            if(isProgram&&getLabel&&isProgram(program)) {
                std::array<char,96> actual{};int length=0;
                getLabel(GL_PROGRAM,program,int(actual.size()),&length,actual.data());
                if(actual==label)gl.DeleteProgram(program);
            }
#endif
        }
        program=0;
    }
};
void deferQueryProgramUnlocked(const std::shared_ptr<OpenGlQueryProgram>& owner) noexcept
{
    // Failed context switching must not drop the final native ID or unlink
    // after the frame/resource destructor's mutex guard ends. The existing
    // context lifecycle drains this charged lease once its context is current.
    if(!owner||!owner->context||!owner->context->active||owner->deferredLease)return;
    owner->deferredLease=owner;
    auto bytes=owner->context->deferredBytes.load(std::memory_order_relaxed);
    if(!renderpassoutput::detail::checkedAdd(bytes,owner->retainedBytes,bytes))bytes=std::numeric_limits<std::uint64_t>::max();
    owner->context->deferredBytes.store(bytes,std::memory_order_relaxed);
}
void OpenGlQueryContext::drainDeferredUnlocked() noexcept
{
    if(glfwGetCurrentContext()!=context)return;
    for(;;) {
        bool drained=false;
        for(auto* owner=programs;owner;owner=owner->next)if(owner->deferredLease) {
            // Destruction can unlink several parent nodes; restart at the head.
            const auto lease=std::move(owner->deferredLease);
            drained=true;break;
        }
        if(!drained)break;
    }
    deferredBytes.store(0,std::memory_order_relaxed);
}
void OpenGlQueryContext::retireUnlocked() noexcept
{
    for(auto* owner=programs;owner;owner=owner->next)owner->releaseUnlocked();
    active=false;
    drainDeferredUnlocked();
}
inline constexpr std::uint64_t kQueryProgramSourceReservation=4ull*1024ull*1024ull;
std::uint64_t nextQueryOwnerGeneration() noexcept
{
    static std::uint64_t serial=0;
    return serial==std::numeric_limits<std::uint64_t>::max()?0:++serial;
}
std::uint64_t querySourceFingerprint(const std::string& source) noexcept
{
    // Exact source bytes are retained too; this fingerprint is not authority.
    std::uint64_t value=14695981039346656037ull;
    for(unsigned char byte:source)value=(value^byte)*1099511628211ull;
    return value;
}
std::shared_ptr<const OpenGlQueryArithmeticEvidence> completedQueryArithmeticEvidence(
    const std::shared_ptr<OpenGlQueryProgram>& owner,OpenGlQueryArithmeticRole role,
    const glasscaustics::SubmissionIdentity& submission,
    std::array<std::shared_ptr<OpenGlQueryProgram>,2> parentOwners={},
    std::array<std::shared_ptr<const OpenGlQueryArithmeticEvidence>,2> parentEvidence={})
{
    // Called only after the owning producer's actual completed native fence.
    auto evidence=std::make_shared<OpenGlQueryArithmeticEvidence>();
    evidence->role=role;evidence->program=owner->identity;evidence->submission=submission;evidence->parents=std::move(parentEvidence);
    owner->parents=std::move(parentOwners);
    for(unsigned parent=0;parent<owner->parents.size();++parent)if(owner->parents[parent])
        owner->retainedBytes+=owner->parents[parent]->retainedBytes+sizeof(OpenGlQueryArithmeticEvidence)+128
            +evidence->parents[parent]->submission.frameIdentity.capacity()+1;
    return evidence;
}
#if !defined(__APPLE__)
unsigned linkCausticProgram(const arbitgl::GlFuncs&,const std::vector<std::pair<unsigned,std::string>>&,std::string&);
std::shared_ptr<OpenGlQueryProgram> linkRegisteredOpenGlProgram(const arbitgl::GlFuncs& gl,
    const std::shared_ptr<OpenGlQueryContext>& context,std::uint64_t rendererGeneration,
    const std::vector<std::pair<unsigned,std::string>>& sources,std::string& error,bool preciseSceneQuery)
{
    const auto label=reinterpret_cast<PFNGLOBJECTLABELPROC>(glfwGetProcAddress("glObjectLabel"));
    const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
    if(!context||!context->active||context->context!=glfwGetCurrentContext()||!rendererGeneration
        ||(context->major<4||(context->major==4&&context->minor<3))||!label||!getLabel||sources.empty()||sources.size()>2)
    {error="OpenGL query contract has no live GL4.3 context/program identity authority";return {};}
    context->drainDeferredUnlocked();
    std::uint64_t bytes=sizeof(OpenGlQueryProgram)+sizeof(OpenGlQueryArithmeticProgramIdentity)+256;
    for(const auto& source:sources) {
        if(source.second.rfind("#version 430 core\n",0)!=0||source.second.find('\0')!=std::string::npos
            ||source.second.empty()||source.second.capacity()>kQueryProgramSourceReservation/8
            ||!renderpassoutput::detail::checkedAdd(bytes,2*(source.second.capacity()+1),bytes))
        {error="OpenGL query contract assembled source exceeds its admitted profile/storage";return {};}
    }
    if(preciseSceneQuery&&std::none_of(sources.begin(),sources.end(),[](const auto& source) {
        return source.second.find("precise SV3 e1=b-a,e2=c-a,p=cross(direction,e2);precise float det=dot(e1,p);")!=std::string::npos
            &&source.second.find("precise float barycentricSum=u+v;")!=std::string::npos
            &&source.second.find("precise float enteringDot=dot(direction,outwardNormal);")!=std::string::npos;
    })) {error="OpenGL query contract assembled source omitted its precise query stages";return {};}
    if(bytes>kQueryProgramSourceReservation) {error="OpenGL query contract identity exceeds its reservation";return {};}
    const auto generation=nextQueryOwnerGeneration();
    if(!generation) {error="OpenGL query contract owner generation exhausted";return {};}
    auto owner=std::make_shared<OpenGlQueryProgram>();owner->gl=gl;owner->context=context;owner->preciseSceneQuery=preciseSceneQuery;
    auto identity=std::make_shared<OpenGlQueryArithmeticProgramIdentity>();
    identity->context=reinterpret_cast<std::uintptr_t>(context->context);identity->contextGeneration=context->generation;
    identity->contextMajor=context->major;identity->contextMinor=context->minor;identity->contextProfile=context->profile;
    identity->ownerGeneration=generation;identity->rendererGeneration=rendererGeneration;identity->stageCount=unsigned(sources.size());
    for(std::size_t index=0;index<sources.size();++index) {
        identity->stages[index]={sources[index].first,querySourceFingerprint(sources[index].second),sources[index].second};
    }
    std::uint64_t retained=sizeof(OpenGlQueryProgram)+sizeof(OpenGlQueryArithmeticProgramIdentity)+256;
    for(unsigned stage=0;stage<identity->stageCount;++stage) {
        const auto capacity=identity->stages[stage].source.capacity();
        if(capacity>kQueryProgramSourceReservation/8
            ||!renderpassoutput::detail::checkedAdd(retained,capacity+1,retained))
        {error="OpenGL retained query source capacity exceeds its reservation";return {};}
    }
    bytes=std::max(bytes,retained);
    if(bytes>kQueryProgramSourceReservation) {error="OpenGL retained query identity exceeds its reservation";return {};}
    owner->program=linkCausticProgram(gl,sources,error);if(!owner->program)return {};
    identity->linkedProgram=owner->program;
    std::snprintf(owner->label.data(),owner->label.size(),"DonutStudio/query430/%llu/%llu/%llu",
        static_cast<unsigned long long>(context->generation),static_cast<unsigned long long>(rendererGeneration),static_cast<unsigned long long>(generation));
    label(GL_PROGRAM,owner->program,-1,owner->label.data());
    std::array<char,96> actual{};int length=0;getLabel(GL_PROGRAM,owner->program,int(actual.size()),&length,actual.data());
    if(glGetError()!=GL_NO_ERROR||actual!=owner->label)
    {gl.DeleteProgram(owner->program);owner->program=0;error="OpenGL query contract linked native owner label differs";return {};}
    owner->identity=std::move(identity);owner->retainedBytes=bytes;
    owner->next=context->programs;if(owner->next)owner->next->previous=owner.get();context->programs=owner.get();owner->registered=true;
    return owner;
}
std::shared_ptr<OpenGlQueryProgram> linkQueryArithmeticProgram(const arbitgl::GlFuncs& gl,
    const std::shared_ptr<OpenGlQueryContext>& context,std::uint64_t rendererGeneration,
    const std::vector<std::pair<unsigned,std::string>>& sources,std::string& error)
{return linkRegisteredOpenGlProgram(gl,context,rendererGeneration,sources,error,true);}
#endif

struct RoughSelectedCamera final
{
    // std430 records consist only of 16-byte lanes. No host-dependent vec3 ABI.
    struct Selection { std::array<float,4> ray,position,normal;std::array<std::uint32_t,4> ids;std::array<std::int32_t,4> state; };
    struct Result { std::array<float,4> color,terminal,masked,tail;std::array<std::uint32_t,4> ids,work; };
    static_assert(sizeof(Selection)==80 && sizeof(Result)==96);
    // GPU selection/result, their completed native readback, shared Color
    // texture and its bounded upload copy. Existing receipt accounting is separate.
    static constexpr std::uint64_t residentBytesPerPixel=2*(sizeof(Selection)+sizeof(Result))+2*48+16+16+16
        +sizeof(roughglasstiles::NativeTileReceipt::Work);
    explicit RoughSelectedCamera(const arbitgl::GlFuncs& functions,std::shared_ptr<OpenGlQueryContext> context={},std::uint64_t generation=0)
        :gl(functions),queryContext(std::move(context)),rendererGeneration(generation) {}
    ~RoughSelectedCamera() {
#if !defined(__APPLE__)
        for(unsigned i=0;i<3;++i) if(saved) {
            if(previousBuffers[i] && previousSizes[i]>0)
                gl.BindBufferRange(GL_SHADER_STORAGE_BUFFER,i,unsigned(previousBuffers[i]),previousStarts[i],previousSizes[i]);
            else gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,i,unsigned(previousBuffers[i]));
        }
        if(saved)gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(previousGeneric));
        gl.DeleteBuffers(3,buffers.data());
#endif
        for(unsigned index=0;index<programs.size();++index)if(programs[index]&&!queryOwners[index])gl.DeleteProgram(programs[index]);
        if(texture)glDeleteTextures(1,&texture);
    }
    bool initialize(std::uint64_t pixels,std::string& error) {
#if defined(__APPLE__)
        (void)pixels;error="OpenGL v2 selected camera compute is unavailable on this backend";return false;
#else
        if(!pixels || pixels>roughglasstiles::kMaximumMultipleScatterTilePixels)
        {error="OpenGL selected camera storage exceeds admitted tile pixels";return false;}
        capacity=pixels;
        glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&previousGeneric);
        for(unsigned i=0;i<3;++i) {
            gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,i,&previousBuffers[i]);
            gl.GetInteger64i_v(GL_SHADER_STORAGE_BUFFER_START,i,&previousStarts[i]);
            gl.GetInteger64i_v(GL_SHADER_STORAGE_BUFFER_SIZE,i,&previousSizes[i]);
        }
        saved=true;gl.GenBuffers(3,buffers.data());
        const std::uint64_t strides[]{sizeof(Selection),sizeof(Result),sizeof(roughglasstiles::NativeTileReceipt::Work)};
        for(unsigned i=0;i<3;++i) {
            if(!buffers[i]){error="OpenGL selected camera buffer allocation failed";return false;}
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[i]);
            gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(pixels*strides[i]),nullptr,GL_DYNAMIC_READ);
            GLint64 actual=0;gl.GetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(actual!=GLint64(pixels*strides[i]) || glGetError()!=GL_NO_ERROR)
            {error="OpenGL selected camera buffer has incomplete native storage";return false;}
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,i,buffers[i]);
        }
        glGenTextures(1,&texture);
        if(!texture){error="OpenGL selected camera shared Color texture allocation failed";return false;}
        for(unsigned i=0;i<2;++i) {
            const auto source=shader(i==1);
            queryOwners[i]=linkQueryArithmeticProgram(gl,queryContext,rendererGeneration,{{GL_COMPUTE_SHADER,source}},error);
            if(!queryOwners[i]){return false;}programs[i]=queryOwners[i]->program;
        }
        return true;
#endif
    }
    bool run(const roughglasstiles::Tile& tile,const roughglasstiles::FrameKey& key,
        const smoothglass::Prepared& prepared,unsigned fragmentProgram,
        const opaquevisibility::Frame& query,const std::array<std::array<float,4>,64>& emission,
        std::array<NativeRawPassPixels,3>& receipts,std::vector<roughglasstiles::NativeTileReceipt::Work>& work,
        std::vector<std::array<std::uint32_t,4>>& identities,std::uint64_t completedWork,std::string& error) {
#if defined(__APPLE__)
        (void)tile;(void)key;(void)prepared;(void)fragmentProgram;(void)query;(void)emission;
        (void)receipts;(void)work;(void)identities;(void)completedWork;
        error="OpenGL v2 selected camera compute is unavailable on this backend";return false;
#else
        if(!tile.pixels() || tile.pixels()>capacity || !prepared.roughConsumer.selectedComputeV2)
        {error="OpenGL selected camera submission exceeds its immutable tile owner";return false;}
        const auto* volume=nativeSurfaceForObject(key.program.get(),prepared.object);
        if(!volume) {error="OpenGL selected camera has no retained volume material owner";return false;}
        const auto getFloat=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
        const auto getInt=reinterpret_cast<PFNGLGETUNIFORMIVPROC>(glfwGetProcAddress("glGetUniformiv"));
        if(!getFloat || !getInt){error="OpenGL selected camera frame uniform authority is unavailable";return false;}
        std::vector<Selection> selected(tile.pixels());std::vector<Result> results(tile.pixels());
        work.assign(tile.pixels(),{});
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[2]);
        const auto counterBytes=tile.pixels()*sizeof(roughglasstiles::NativeTileReceipt::Work);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(counterBytes),work.data(),GL_DYNAMIC_READ);
        GLint64 stored=0;gl.GetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&stored);
        if(stored!=GLint64(counterBytes) || glGetError()!=GL_NO_ERROR)
        {error="OpenGL selected camera native counter initialization failed";return false;}
        const auto read=[&](unsigned binding,void* destination,std::size_t bytes,std::size_t offset=0) {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[binding]);
            const auto* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(offset),std::ptrdiff_t(bytes),GL_MAP_READ_BIT);
            if(!mapped)return false;
            std::memcpy(destination,mapped,bytes);
            return gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)==GL_TRUE && glGetError()==GL_NO_ERROR;
        };
        for(unsigned stage=0;stage<2;++stage) {
            gl.UseProgram(programs[stage]);
            const auto loc=[&](const char* name){return gl.GetUniformLocation(programs[stage],name);};
            const auto copy=[&](const char* name,unsigned components) {
                const auto from=gl.GetUniformLocation(fragmentProgram,name),to=loc(name);
                if(from<0 || to<0)return;
                std::array<float,4> value{};getFloat(fragmentProgram,from,value.data());
                if(components==3)gl.Uniform3f(to,value[0],value[1],value[2]);else gl.Uniform4fv(to,1,value.data());
            };
            gl.Uniform2f(loc("uExtent"),float(key.width),float(key.height));
            gl.Uniform4f(loc("uTile"),float(tile.x),float(tile.y),float(tile.width),float(tile.height));
            std::array<float,4> rotation{};storeFixtureQuaternion(rotation,key.camera.transform.rotation);
            gl.Uniform4fv(loc("uCameraRotation"),1,rotation.data());
            gl.Uniform4f(loc("uProjection"),std::tan(key.camera.verticalFovRadians*.5f),float(key.width)/key.height,0,0);
            const auto& camera=key.camera.transform.translation;
            gl.Uniform3f(loc("uCameraTranslation"),camera.x,camera.y,camera.z);
            gl.Uniform1i(loc("uGlassVisibility"),6);gl.Uniform4fv(loc("uGlassControl"),1,query.control.data());
            if(stage==1) {
                const auto& p=volume->parameters;
                gl.Uniform4f(loc("uGlassTransport"),1,p.transmissionIorClearcoat[3],float(prepared.object.value&65535u),float(prepared.object.value>>16));
                gl.Uniform4fv(loc("uSurfaceCoating"),1,p.transmissionIorClearcoat.data());
                gl.Uniform1f(loc("uRoughness"),prepared.roughness);
                gl.Uniform4fv(loc("uGlassReceivers[0]"),64,prepared.receivers[0].data());
                gl.Uniform4fv(loc("uGlassEmission[0]"),64,emission[0].data());
                gl.Uniform1i(loc("uOpaqueVisibility"),5);gl.Uniform1i(loc("uEnvironmentImage"),7);
                copy("uOpaqueControl",4);copy("uAmbient",3);
                int lightCount=0;getInt(fragmentProgram,gl.GetUniformLocation(fragmentProgram,"uLightCount"),&lightCount);
                gl.Uniform1i(loc("uLightCount"),lightCount);
                for(int i=0;i<16;++i) {
                    for(const char* array:{"uLightRotations","uLightColors","uLightPositions","uLightCones"})
                        copy((std::string(array)+"["+std::to_string(i)+"]").c_str(),4);
                    const auto name="uLightKinds["+std::to_string(i)+"]";
                    const auto from=gl.GetUniformLocation(fragmentProgram,name.c_str()),to=loc(name.c_str());
                    if(from>=0 && to>=0) {int kind=0;getInt(fragmentProgram,from,&kind);gl.Uniform1i(to,kind);}
                }
                for(int i=0;i<5;++i)copy(("uSky["+std::to_string(i)+"]").c_str(),4);
            }
            // local_size_x=1 gives precisely one invocation per admitted pixel.
            // Compute has no fragment helper invocations or raster overdraw.
            if(stage==0) {
                gl.Uniform1ui(loc("uDispatchOffset"),0);
                gl.DispatchCompute(unsigned(tile.pixels()),1,1);
                gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
                if(!roughglasstiles::opengl::completeSubmission(gl,error)
                    || !read(0,selected.data(),selected.size()*sizeof(Selection)))
                {if(error.empty())error="OpenGL selected camera primary readback failed";return false;}
                for(const auto& primary:selected)if(primary.state[0]<0 || primary.state[0]>1 || primary.state[1]!=1)
                {error="OpenGL selected camera closest-hit selection did not complete";return false;}
            } else {
                std::size_t next=0;
                while(next<selected.size()) {
                    if(completedWork>smoothglass::kMaximumCameraMicroWork)
                    {error="OpenGL selected camera whole-frame executed work exceeds its bound";return false;}
                    const auto reserve=(smoothglass::kMaximumCameraMicroWork-completedWork)/smoothglass::kMaximumCameraPixelMicroWork;
                    std::size_t end=next;std::uint64_t starts=0;
                    while(end<selected.size()) {
                        const bool glass=selected[end].state[0]==1 && selected[end].ids[0]==prepared.object.value;
                        if(glass && starts==reserve)break;
                        starts+=glass?1u:0u;++end;
                    }
                    if(end==next)
                    {error="OpenGL selected camera whole-frame work reserve exhausted before transport";return false;}
                    // Disjoint pixel ranges consume completed selection records.
                    // Reserve their worst case before execution, then charge the
                    // native events before admitting the next range.
                    gl.Uniform1ui(loc("uDispatchOffset"),unsigned(next));
                    gl.DispatchCompute(unsigned(end-next),1,1);
                    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
                    if(!roughglasstiles::opengl::completeSubmission(gl,error)
                        || !read(1,results.data()+next,(end-next)*sizeof(Result),next*sizeof(Result)))
                    {if(error.empty())error="OpenGL selected camera completed transport readback failed";return false;}
                    for(std::size_t pixel=next;pixel<end;++pixel) {
                        const auto& result=results[pixel];
                        const bool glass=selected[pixel].state[0]==1 && selected[pixel].ids[0]==prepared.object.value;
                        if(result.ids!=selected[pixel].ids || result.work[0]!=1 || result.work[1]!=(glass?1u:0u)
                            || result.work[3]>smoothglass::kMaximumCameraMicroEvents
                            || (glass && result.color[3]!=smoothglass::kMultipleScatterCameraComplete)
                            || (!glass && result.work[3]!=0))
                        {error="OpenGL selected camera native invocation did not complete within its work reserve";return false;}
                        completedWork+=std::uint64_t(result.work[3])*smoothglass::kCameraMicroWorkPerEvent;
                    }
                    next=end;
                }
            }
        }
        if(!read(2,work.data(),work.size()*sizeof(roughglasstiles::NativeTileReceipt::Work)))
        {error="OpenGL selected camera completed atomic counter readback failed";return false;}
        identities.resize(tile.pixels());
        for(auto& receipt:receipts)receipt={NativeTexturePixelFormat::Rgba32Float,tile.width,tile.height,std::vector<std::uint8_t>(tile.pixels()*16)};
        std::vector<std::array<float,4>> sharedColor(tile.pixels()*3);
        for(std::size_t i=0;i<results.size();++i) {
            const auto& primary=selected[i];const auto& result=results[i];
            const bool glass=primary.state[0]==1 && primary.ids[0]==prepared.object.value;
            if(std::memcmp(&work[i],result.work.data(),sizeof(work[i]))!=0
                || primary.state[0]<0 || primary.state[0]>1 || primary.state[1]!=1 || result.ids!=primary.ids
                || result.work[0]!=1 || result.work[1]!=(glass?1u:0u)
                || (glass && (result.color[3]!=smoothglass::kMultipleScatterCameraComplete
                    || result.work[2]!=result.masked[3] || result.work[3]!=result.tail[3]))
                || (!glass && (result.work[2]!=1 || result.work[3]!=0)))
            {error="OpenGL selected camera native primary or executed-work record is incomplete";return false;}
            identities[i]=result.ids;
            std::memcpy(receipts[0].bytes.data()+i*16,result.terminal.data(),16);
            std::memcpy(receipts[1].bytes.data()+i*16,result.masked.data(),16);
            std::memcpy(receipts[2].bytes.data()+i*16,result.tail.data(),16);
            sharedColor[i*3]=result.color;
            for(unsigned id=0;id<4;++id) {
                sharedColor[i*3+1+id/2][(id%2)*2]=float(result.ids[id]&65535u);
                sharedColor[i*3+1+id/2][(id%2)*2+1]=float(result.ids[id]>>16);
            }
        }
        gl.ActiveTexture(GL_TEXTURE10);gl.BindSampler(10,0);glBindTexture(GL_TEXTURE_2D,texture);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_BASE_LEVEL,0);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,int(tile.width*3),int(tile.height),0,GL_RGBA,GL_FLOAT,sharedColor.data());
        gl.UseProgram(fragmentProgram);
        gl.Uniform4f(gl.GetUniformLocation(fragmentProgram,"uSelectedRoughTile"),float(tile.x),float(tile.y),float(tile.width),float(tile.height));
        gl.ActiveTexture(GL_TEXTURE0);
        if(glGetError()!=GL_NO_ERROR){error="OpenGL selected camera shared native result upload failed";return false;}
        return true;
#endif
    }
    static std::string shader(bool transport) {
        std::string source=R"glsl(#version 430 core
layout(local_size_x=1) in;
uniform vec2 uExtent;
uniform uint uDispatchOffset;
uniform vec4 uTile,uCameraRotation,uProjection;
uniform vec3 uCameraTranslation;
struct Selection {vec4 ray,position,normal;uvec4 ids;ivec4 state;};
struct Result {vec4 color,terminal,masked,tail;uvec4 ids,work;};
layout(std430,binding=2) buffer ActualWork {uvec4 executed[];};
)glsl";
        source+=transport?"layout(std430,binding=0) readonly buffer Primary {Selection primary[];};\nlayout(std430,binding=1) writeonly buffer Results {Result results[];};\n"
            :"layout(std430,binding=0) writeonly buffer Primary {Selection primary[];};\n";
        // The glass query resets the shared query macros before its own namespace.
        if(transport)source+=opaquevisibility::glslShader(true);
        source+=smoothglass::queryShader(false,true);
        source+=R"glsl(
vec3 rotateQ(vec4 q,vec3 p) {return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
vec2 cameraPixel(uint index) {return uTile.xy+vec2(index%uint(uTile.z),index/uint(uTile.z))+vec2(0.5);}
)glsl";
        const auto cameraMath=glasscaustics::nativeMathSource();
        const auto cameraBegin=cameraMath.find("vec3 causticRotate(");
        source+=cameraMath.substr(cameraBegin,cameraMath.find("bool causticPixelCorner(",cameraBegin)-cameraBegin);
        if(!transport)return source+R"glsl(
void main() {
    uint index=gl_GlobalInvocationID.x+uDispatchOffset;
    vec3 ray=causticCameraRay(cameraPixel(index),uCameraRotation,uProjection,uExtent);
    atomicAdd(executed[index].x,1u);
    GlassClosestHit hit=glassSceneClosestHit(uCameraTranslation,ray,0.0,3.402823e38,uGlassControl);
    atomicAdd(executed[index].z,1u);
    primary[index].ray=vec4(ray,0.0);primary[index].position=vec4(hit.position,hit.distance);
    primary[index].normal=vec4(hit.outwardNormal,hit.entering?1.0:0.0);
    primary[index].ids=uvec4(hit.objectId,hit.materialId,hit.instanceId,hit.triangleId);
    primary[index].state=ivec4(hit.status,1,0,0);
}
)glsl";
        source+="uniform float uRoughness;\nuniform vec4 uSurfaceCoating,uSky[5];\nuniform vec3 uAmbient;\nuniform int uLightCount,uLightKinds[16];\nuniform vec4 uLightRotations[16],uLightColors[16],uLightPositions[16],uLightCones[16];\n";
        source+=analyticsky::shader(false)+environmentimage::shader(false)
            +smoothglass::transportShader(false,true,true);
        const std::string material=kFixtureFragmentShaderTail;
        const auto begin=material.find("vec3 baseSurfaceUnit(");
        source+=material.substr(begin,material.find("void main()",begin)-begin);
        return source+R"glsl(
void main() {
    uint index=gl_GlobalInvocationID.x+uDispatchOffset;
    Selection selection=primary[index];
    Result result;result.color=vec4(0.0);result.terminal=vec4(0.0);result.masked=vec4(0.0);result.tail=vec4(0.0);
    result.ids=selection.ids;result.work=uvec4(0u);
    uint volume=uint(uGlassTransport.z)|(uint(uGlassTransport.w)<<16u);
    if(selection.state.x==1 && selection.ids.x==volume) {
        GlassClosestHit first;first.status=selection.state.x;first.distance=selection.position.w;
        first.position=selection.position.xyz;first.outwardNormal=selection.normal.xyz;
        first.objectId=selection.ids.x;first.materialId=selection.ids.y;first.instanceId=selection.ids.z;first.triangleId=selection.ids.w;
        first.entering=selection.normal.w==1.0;
        vec3 terminal,masked,tail;int status,queries,events;
        atomicAdd(executed[index].y,1u); // Count the actual expensive entry before any event walk.
        vec3 color=roughGlassTransportV2Selected(uCameraTranslation,selection.ray.xyz,first,uRoughness,false,
            glassCameraPixelKey(cameraPixel(index)),terminal,masked,tail,status,queries,events);
        result.color=vec4(color,float(status));result.terminal=vec4(terminal,float(status));
        result.masked=vec4(masked,float(queries));result.tail=vec4(tail,float(events));
        atomicAdd(executed[index].z,uint(queries-1));atomicAdd(executed[index].w,uint(events));
    }
    memoryBarrierBuffer();
    result.work=executed[index];
    results[index]=result;
}
)glsl";
    }
    arbitgl::GlFuncs gl;
    std::shared_ptr<OpenGlQueryContext> queryContext;
    std::uint64_t rendererGeneration=0;
    std::array<std::shared_ptr<OpenGlQueryProgram>,2> queryOwners{};
    std::array<unsigned,2> programs{};std::array<unsigned,3> buffers{};
    unsigned texture=0;std::uint64_t capacity=0;bool saved=false;
    int previousGeneric=0;std::array<int,3> previousBuffers{};
#if !defined(__APPLE__)
    std::array<GLint64,3> previousStarts{},previousSizes{};
#endif
};

unsigned compileDiffractionResolveProgram(const arbitgl::GlFuncs& gl, std::string& error)
{
    const auto fragmentSource = std::string(kDiffractionResolveFragmentShaderHead)
        + kFixturePassCompositeShader + kDiffractionResolveFragmentShaderTail;
    const auto vertex = compileFixtureShader(gl, GL_VERTEX_SHADER,
        kDiffractionResolveVertexShader, error);
    if (vertex == 0) return 0;
    const auto fragment = compileFixtureShader(gl, GL_FRAGMENT_SHADER,
        fragmentSource.c_str(), error);
    if (fragment == 0) { gl.DeleteShader(vertex); return 0; }
    const auto program = gl.CreateProgram();
    gl.AttachShader(program, vertex);
    gl.AttachShader(program, fragment);
    gl.LinkProgram(program);
    gl.DeleteShader(vertex);
    gl.DeleteShader(fragment);
    int linked = 0;
    gl.GetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) return program;
    char log[1024] = {};
    int length = 0;
    gl.GetProgramInfoLog(program, static_cast<int>(sizeof(log)), &length, log);
    error = "OpenGL diffraction resolve program link failed: "
        + std::string(log, static_cast<std::size_t>(std::max(length, 0)));
    gl.DeleteProgram(program);
    return 0;
}

// Feature bits of the scene fragment program. A set bit keeps the GLSL that the feature
// needs. A clear bit removes that GLSL. A uniform that only the removed GLSL reads is not
// used by the linked program, so its location is -1 and complete() does not require it.
enum : unsigned
{
    kFixtureFragmentSmoothGlass = 1u << 0,
    kFixtureFragmentDiffraction = 1u << 1,
    kFixtureFragmentAllFeatures = kFixtureFragmentSmoothGlass | kFixtureFragmentDiffraction,
};

struct OpenGlFixtureUniformLocations final
{
    int sky = -1, skyBackground = -1, environmentImage = -1;
    int opaqueTexture = -1, opaqueControl = -1;
    int glassTexture = -1, glassControl = -1, glassTransport = -1, glassReceivers = -1, glassEmission = -1;
    int roughCameraControl = -1;
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
    int usePreviousDeformation = -1, noteMotion = -1, previousNoteTransforms = -1;
    int vertexTime = -1, vertexSpectrum = -1, previousVertexSpectrum = -1, vertexDisplacementLimits = -1;
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
    int diffractionPathIndex = -1, diffractionTriangleCount = -1;
    int diffractionQueryTexture = -1, diffractionQueryControl = -1;
    int diffractionSamples = -1, diffractionSampleControl = -1, diffractionInstanceCount = -1;
    int diffractionSourceDirectionTrace = -1, diffractionSourceRadiance = -1, diffractionRayMinimum = -1;
    int noteInstanceTransforms = -1;
    int noteInstanceIndex = -1;
    int noteMeshScale = -1;
    int noteAppearanceLow = -1;
    int noteAppearanceHigh = -1;

    // The requirement depends on the feature mask the program was composed with. Each group
    // follows the GLSL that reads it in fixtureFragmentShaderSource. The common group is read
    // by ungated composer or vertex GLSL in every variant. glassTransport is in it because the
    // ungated tail reads it for coverage, and the composer declares it when smooth glass is off.
    bool complete (unsigned features) const noexcept
    {
        const bool smoothGlass = (features & kFixtureFragmentSmoothGlass) != 0;
        const bool diffraction = (features & kFixtureFragmentDiffraction) != 0;
        const bool common = usePreviousDeformation >= 0 && noteMotion >= 0 && previousNoteTransforms >= 0
            && sky >= 0 && skyBackground >= 0 && objectMatrix >= 0 && tangentHandednessSign >= 0
            && useInstanceMatrix >= 0
            && cameraRotation >= 0 && cameraTranslation >= 0 && projection >= 0
            && baseColor >= 0 && materialParams >= 0 && surfaceCoating >= 0
            && timeMixEndColorAndTime >= 0 && emissive >= 0 && ambient >= 0
            && baseTexture >= 0 && lightCount >= 0
            && metallicRoughnessTexture >= 0 && normalTexture >= 0
            && occlusionTexture >= 0 && emissiveTexture >= 0
            && texturePresence >= 0 && emissiveTexturePresence >= 0
            && alphaModeCutoff >= 0 && imageOutput >= 0
            && glassTransport >= 0
            && noteInstanceTransforms >= 0
            && noteInstanceIndex >= 0 && noteMeshScale >= 0
            && noteAppearanceLow >= 0 && noteAppearanceHigh >= 0;
        // Smooth glass: the transport GLSL (fixture_smooth_glass.h) declares these and reads
        // GU_GlassReceivers and GU_GlassEmission. The glass query reads glassTexture and glassControl.
        const bool glass = ! smoothGlass
            || (glassTexture >= 0 && glassControl >= 0 && glassReceivers >= 0 && glassEmission >= 0);
        // Diffraction: the #if FIXTURE_DIFFRACTION blocks and diffractionQueryShader. materialKind is
        // read only by the diffraction block in main. diffractionInstanceCount reaches the fragment
        // only through vDiffractionInstance, which only that block reads.
        const bool diffractionUniforms = ! diffraction
            || (materialKind >= 0 && diffractionGeometry >= 0
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
            && diffractionPathIndex >= 0 && diffractionTriangleCount >= 0
            && diffractionQueryTexture >= 0 && diffractionQueryControl >= 0
            && diffractionSamples >= 0 && diffractionSampleControl >= 0 && diffractionInstanceCount >= 0
            && diffractionSourceDirectionTrace >= 0 && diffractionSourceRadiance >= 0 && diffractionRayMinimum >= 0);
        return common && glass && diffractionUniforms;
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
        unsigned diffractionResolveId = 0;
        unsigned diffractionResolveVao = 0;
        ~SharedProgram()
        {
            // Scene resources and the context cache release their program leases
            // under fixtureMutex while this program's context is current.
            if (context != nullptr && glfwGetCurrentContext()==context)
            {
                if (id != 0) gl.DeleteProgram(id);
                if (diffractionResolveId != 0) gl.DeleteProgram(diffractionResolveId);
                if (diffractionResolveVao != 0) gl.DeleteVertexArrays(1, &diffractionResolveVao);
            }
        }
    };
    ~OpenGlFixtureSceneResources() override
    {
        if((!hasResources()||ownerContext==nullptr)&&!causticQueryProgram)return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        if(queryContext&&!queryContext->active) {
            // The teardown hook retired native query IDs before the window
            // could disappear/reuse its pointer. Late leases need no GL call.
            causticQueryProgram.reset();queryContext.reset();
            if(sharedProgram)sharedProgram->context=nullptr;
            sharedProgram.reset();ownerContext=nullptr;return;
        }
        if (! hasResources() || ownerContext == nullptr) {
            causticQueryProgram.reset();queryContext.reset();return;
        }
        auto* previous = glfwGetCurrentContext();
        if (previous != ownerContext)
            glfwMakeContextCurrent (ownerContext);
        if (glfwGetCurrentContext() == ownerContext)
            destroyUnlocked();
        else {
            // The owner context is not current here, whether another context or none is.
            // No GL call is made in this branch that deletes the fixture's direct members,
            // so they die with ownerContext. causticQueryProgram is deferred instead and
            // is deleted at the next drain on ownerContext.
            // Their residency charges release when the members are destroyed after this
            // body, so the ledger can lead the context's own teardown.
            deferQueryProgramUnlocked(causticQueryProgram);causticQueryProgram.reset();queryContext.reset();
            sharedProgram.reset();
        }
        if (previous != ownerContext)
            glfwMakeContextCurrent (previous);
    }

    const std::string& backend() const noexcept override { return backend_; }

    bool hasResources() const noexcept
    {
        return vertexArray != 0 || vertexBuffer != 0 || previousVertexBuffer != 0 || indexBuffer != 0
            || instanceBuffer != 0
            || ! textures.empty() || environmentTexture!=0 || program != 0;
    }

    void destroyUnlocked() noexcept
    {
        if (environmentTexture) glDeleteTextures(1,&environmentTexture);
        environmentTexture=0; environmentResident.reset(); environmentBudget.reset();
        if (! textures.empty())
            glDeleteTextures (static_cast<int> (textures.size()), textures.data());
        textureCharges.clear();
        if (indexBuffer != 0) gl.DeleteBuffers (1, &indexBuffer);
        indexBufferCharge.reset();
        if (instanceBuffer != 0) gl.DeleteBuffers (1, &instanceBuffer);
        instanceBufferCharge.reset();
        if (previousVertexBuffer != 0) gl.DeleteBuffers (1, &previousVertexBuffer);
        previousVertexBufferCharge.reset();
        previousVertexBuffer = 0;
        if (vertexBuffer != 0) gl.DeleteBuffers (1, &vertexBuffer);
        vertexBufferCharge.reset();
        if (vertexArray != 0) gl.DeleteVertexArrays (1, &vertexArray);
        vertexArray = 0;
        vertexBuffer = 0;
        indexBuffer = 0;
        instanceBuffer = 0;
        textures.clear();
        program = 0;
        for (std::size_t index=0;index<causticPrograms.size();++index)
            if(causticPrograms[index]&&(index!=0||!causticQueryProgram))gl.DeleteProgram(causticPrograms[index]);
        causticPrograms = {};
        causticQueryProgram.reset();queryContext.reset();
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
    opaquevisibility::Prepared opaqueVisibility;
    smoothglass::Prepared smoothGlass;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> gratingLeaf;
    std::uint64_t gratingStaticBytes=0;
    diffractiontransportnative::Prepared diffractionTransport;
    // On the destroyUnlocked path, each residency charge releases with the GL object
    // beside it. The destructor's else branch runs when ownerContext is not current. It
    // skips destroyUnlocked, so its charges release with the members while the GL objects
    // survive until ownerContext dies. The environment texture is charged through
    // environmentBudget, which is the same decoded byte count.
    unsigned vertexArray = 0;
    unsigned vertexBuffer = 0;
    residency::ResidencyCharge vertexBufferCharge;
    mutable unsigned previousVertexBuffer = 0;
    mutable residency::ResidencyCharge previousVertexBufferCharge;
    mutable bool usePreviousDeformation = false;
    // Joint and composition mirror state of each deformed member, keyed by its first output
    // vertex. The compute dispatch writes it, and the draw combines it with objectMatrix.
    struct DeformedWinding final
    {
        bool jointsMixed = false;
        bool skinOrPointMirrored = false;
    };
    mutable std::unordered_map<std::uint32_t, DeformedWinding> deformedWindings;
    const DeformedWinding* deformedWindingFor (std::uint32_t firstVertex) const
    {
        const auto found = deformedWindings.find (firstVertex);
        return found == deformedWindings.end() ? nullptr : &found->second;
    }
    unsigned indexBuffer = 0;
    residency::ResidencyCharge indexBufferCharge;
    unsigned instanceBuffer = 0;
    residency::ResidencyCharge instanceBufferCharge;
    bool instancedSharedGeometry = false;
    bool diagnosticInstanceIdentityColors = false;
    std::vector<videowire::geometry::InstanceAppearance> instanceAppearances;
    std::vector<unsigned> textures;
    // One charge per entry in textures, in the same order.
    std::vector<residency::ResidencyCharge> textureCharges;
    unsigned environmentTexture=0;
    videohelper::environmentimage::ResidentPtr environmentResident;
    std::shared_ptr<environmentimage::GpuBudget> environmentBudget;
    unsigned program = 0;
    std::uint64_t retainedTextureBytes = 0;
    mutable std::array<unsigned, 6> causticPrograms {};
    mutable std::uint32_t causticProgramMultipleScatterVersion = 0;
    mutable std::uint64_t causticSubmissionSerial=0;
    std::shared_ptr<OpenGlQueryContext> queryContext;
    mutable std::shared_ptr<OpenGlQueryProgram> causticQueryProgram;
    std::shared_ptr<SharedProgram> sharedProgram;
    OpenGlFixtureUniformLocations uniforms {};
};

struct OpenGlDirectGratingBatch
{
    std::shared_ptr<OpenGlQueryProgram> program;
    std::shared_ptr<const OpenGlQueryArithmeticEvidence> evidence;
    glasscaustics::NativeDirectGratingEndpointUpload upload;
    glasscaustics::NativeDirectGratingContributionReadback readback;
    glasscaustics::DirectGratingContributionBounds bounds;
    std::vector<glasscaustics::opengl::DirectGratingPacked> packed;
    unsigned queryTexture=0,sourceBuffer=0;
    std::array<float,4> queryControl{};
    std::uint64_t ownedBytes=0,retainedWork=0,nativeBytes=0;
    std::unique_ptr<OpenGlDirectGratingBatch> previous;
};

struct OpenGlCameraGlassGratingBatch
{
    std::shared_ptr<OpenGlQueryProgram> program;
    std::shared_ptr<const OpenGlQueryArithmeticEvidence> evidence;
    glasscaustics::NativeGlassCameraGratingTraceUpload upload;
    glasscaustics::NativeGlassCameraGratingTraceReadback readback;
    glasscaustics::GlassCameraGratingTraceBounds bounds;
    std::vector<glasscaustics::opengl::CameraGlassGratingPacked> packed;
    unsigned queryTexture=0,sceneBuffer=0;
    std::array<float,4> queryControl{};
    std::uint64_t ownedBytes=0,retainedWork=0,nativeBytes=0;
    std::unique_ptr<OpenGlCameraGlassGratingBatch> previous;
};

class OpenGlFixtureSceneFrame;
class OpenGlDirectTerminalLease;
class OpenGlDirectTerminalReceiverLease;
class OpenGlCameraCoverageLease;
thread_local const OpenGlCameraCoverageLease* cameraCoverageThreadLease=nullptr;
enum class OpenGlNativeAttemptKind { Direct,Cohort,Camera,Adjunct,Terminal,Receiver,CameraCoverage };
struct OpenGlNativeAttempt {
    OpenGlNativeAttemptKind kind=OpenGlNativeAttemptKind::Direct;
    std::shared_ptr<const glasscaustics::GratingIncidentBinding> binding;
    std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort> cohort;
    const OpenGlFixtureSceneFrame* owner=nullptr;
    GLFWwindow* context=nullptr;
    std::uint64_t renderer=0,contextGeneration=0,ordinal=0,stagingGeneration=0;
    std::size_t count=0,leaves=0;unsigned adjuncts=0;
    std::array<std::uint64_t,3> adjunctStorage{};
    std::array<const void*,3> adjunctOwners{};
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult> terminal;
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult> receiver;
    glasscaustics::CameraCellCoverageBounds cameraCoverageBounds;
    // Arithmetic units never enter the released geometric-work counter.
    std::array<std::uint64_t,8> cameraCoverageWork{};
    bool cameraCoverageReserved=false;
};
struct OpenGlNativeAttemptTotals {
    std::uint64_t direct=0,camera=0,native=0,cameraAdjunct=0,cell=0,coverage=0,terminal=0,terminalDeposition=0,cohortDeposition=0,bytes=0,nativeDeposition=0;
    std::array<std::uint64_t,8> cameraCoverageWork{};
};
bool openGlAttemptTotals(const OpenGlFixtureSceneFrame&,OpenGlNativeAttemptTotals&,std::string&);
bool openGlTerminalNativeBudget(const OpenGlFixtureSceneFrame&,glasscaustics::DirectGratingTerminalNativeBudget&,std::string&);
thread_local const OpenGlFixtureSceneFrame* directTerminalEntryFrame=nullptr;
thread_local std::unique_lock<std::mutex>* directTerminalEntryLock=nullptr;
thread_local const OpenGlDirectTerminalLease* directTerminalThreadLease=nullptr;
struct OpenGlDirectTerminalRecord
{
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult> result;
    std::uint64_t ownedBytes=0,work=0,depositionWork=0;
    std::unique_ptr<OpenGlDirectTerminalRecord> previous;
};
thread_local const OpenGlDirectTerminalReceiverLease* directReceiverThreadLease=nullptr;
thread_local OpenGlTerminalReceiverStopForFixture directReceiverStop=OpenGlTerminalReceiverStopForFixture::None;
struct OpenGlDirectTerminalReceiverRecord
{
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult> result;
    std::shared_ptr<OpenGlQueryProgram> program;
    std::vector<glasscaustics::NativeDirectGratingTerminalReceiverInput> inputs;
    std::vector<glasscaustics::opengl::DirectTerminalReceiverPacked> packed;
    std::uint64_t ownedBytes=0,nativeBytes=0,attemptOrdinal=0,inputGeneration=0;
    glasscaustics::DirectGratingTerminalReceiverBounds reservation;
    std::unique_ptr<OpenGlDirectTerminalReceiverRecord> previous;
};
struct OpenGlCameraCoverageRecord
{
    std::shared_ptr<const glasscaustics::NativeCameraCellCoverageResult> result;
    std::shared_ptr<OpenGlQueryProgram> program;
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult> parent;
    std::vector<std::uint32_t> inputs,certificate;
    glasscaustics::CameraCellCoverageBounds reservation;
    std::uint64_t ownedBytes=0,nativeBytes=0,attemptOrdinal=0,generation=0;
    std::unique_ptr<OpenGlCameraCoverageRecord> previous;
};
const OpenGlCameraCoverageRecord* openGlActiveCameraCoverageRecord(const OpenGlFixtureSceneFrame&);
bool openGlCameraCoverageReadbackBytes(const glasscaustics::CameraCellCoverageReadback& value,std::uint64_t& bytes)
{
    const auto add=[&](std::uint64_t count,std::uint64_t stride=1){
        std::uint64_t amount=0;return renderpassoutput::detail::checkedMultiply(count,stride,amount)
            &&renderpassoutput::detail::checkedAdd(bytes,amount,bytes)&&bytes<=renderpassoutput::kMaximumTotalBytes;
    };
    return add(sizeof(value))&&add(value.submission.frameIdentity.capacity()+1)
        &&add(value.preparation.pixels.capacity(),sizeof(glasscaustics::CameraCellCoverageInput))
        &&add(value.preparation.associations.capacity(),sizeof(glasscaustics::CameraCellCoverageAssociation))
        &&add(value.preparation.candidates.capacity(),sizeof(glasscaustics::CameraCellCoverageCandidate))
        &&add(value.proof.regions.capacity(),sizeof(glasscaustics::CameraCellCoverageRegion))
        &&add(value.proof.vertices.capacity(),sizeof(std::array<double,2>));
}
struct OpenGlDirectGratingCohort;
// Inspection views borrow one actual frame. They own no independent scene,
// raster, query or Color attachments and live in that frame's retained cohort.
class OpenGlDirectGratingSource final : public NativeFixtureSceneFrame,
    public glasscaustics::NativeDirectionalSourceBinding, public glasscaustics::NativeDirectGratingIrradianceBinding
{
public:
    const OpenGlFixtureSceneFrame* frame=nullptr;
    OpenGlDirectGratingCohort* cohort=nullptr;
    std::shared_ptr<const glasscaustics::GratingIncidentBinding> binding;
    glasscaustics::NativeDirectionalSourceUpload sourceUpload;
    glasscaustics::DirectGratingContributionBounds admitted;
    std::uint64_t otherBytes=0,otherWork=0;
    std::unique_ptr<OpenGlDirectGratingBatch> batch;
    const std::string& backend() const noexcept override;
    std::uint32_t width() const noexcept override;
    std::uint32_t height() const noexcept override;
    std::uintptr_t nativeResourceCacheIdentity() const noexcept override;
    std::uintptr_t colorImageHandle() const noexcept override {return 0;}
    std::uintptr_t colorTextureViewHandle() const noexcept override {return 0;}
    const NativeFixtureSceneFrame& owner() const noexcept override {return *this;}
    const glasscaustics::NativeDirectionalSourceUpload& upload() const noexcept override {return sourceUpload;}
    const glasscaustics::NativeDirectionalSourceBinding& directSourceBinding() const noexcept override {return *this;}
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& directGratingBinding() const noexcept override {return binding;}
    const glasscaustics::NativeDirectGratingEndpointUpload& directEndpointUpload() const noexcept override
    {
        static const glasscaustics::NativeDirectGratingEndpointUpload empty;
        return batch?batch->upload:empty;
    }
    bool validateDirectProductionQuery(const glasscaustics::SubmissionIdentity&,
        const glasscaustics::NativeDirectGratingContributionReadback&,std::string&) const override;
};
struct OpenGlDirectGratingCohort
{
    std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort> binding;
    std::vector<std::unique_ptr<OpenGlDirectGratingSource>> sources;
    glasscaustics::SubmissionIdentity submission;
    std::uint64_t ownedBytes=0,reservedWork=0,chargedWork=0,reservedDepositionWork=0;
    std::unique_ptr<OpenGlDirectGratingCohort> previous;
    ~OpenGlDirectGratingCohort() {
        for(const auto& source:sources)if(source->batch&&source->batch->program
            &&source->batch->program->context&&glfwGetCurrentContext()!=source->batch->program->context->context)
            deferQueryProgramUnlocked(source->batch->program);
    }
};

class OpenGlFixtureSceneFrame final : public NativeFixtureSceneFrame, public glasscaustics::NativeDirectionalSourceBinding,
    public glasscaustics::NativeGratingCellSourceBinding, public glasscaustics::NativeGratingCellCoverageSource,
    public glasscaustics::NativeDirectGratingIrradianceBinding, public glasscaustics::NativeGlassCameraGratingTraceBinding,
    public glasscaustics::NativeDirectGratingIrradianceCohortBinding, public glasscaustics::NativeCameraCellCoverageBinding
{
public:
    ~OpenGlFixtureSceneFrame() override
    {
        if(ownerContext==nullptr&&!rasterCaptureProgram&&!cameraCoverageResults&&!failedCameraCoverageResources
            &&std::none_of(queryPrograms.begin(),queryPrograms.end(),[](const auto& owner){return bool(owner);}))return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        if(ownerContext==nullptr||(staticResources&&staticResources->queryContext&&!staticResources->queryContext->active)) {
            for(auto* receiver=directReceiverResults.get();receiver;receiver=receiver->previous.get())deferQueryProgramUnlocked(receiver->program);
            for(auto* receiver=failedReceiverResources.get();receiver;receiver=receiver->previous.get())deferQueryProgramUnlocked(receiver->program);
            for(auto* coverage=cameraCoverageResults.get();coverage;coverage=coverage->previous.get())deferQueryProgramUnlocked(coverage->program);
            for(auto* coverage=failedCameraCoverageResources.get();coverage;coverage=coverage->previous.get())deferQueryProgramUnlocked(coverage->program);
            deferQueryProgramUnlocked(rasterCaptureProgram);
            cameraCoverageResults.reset();failedCameraCoverageResources.reset();rasterCaptureProgram.reset();
            directReceiverResults.reset();failedReceiverResources.reset();
            directGratingCohort.reset();directGratingBatch.reset();cameraGlassTraceBatch.reset();
            queryEvidence={};queryPrograms={};queryRetainedBytes=0;ownerContext=nullptr;return;
        }
        auto* previous = glfwGetCurrentContext();
        if (previous != ownerContext)
            glfwMakeContextCurrent (ownerContext);
        if (glfwGetCurrentContext() == ownerContext)
            releaseTexturesUnlocked();
        else {
            for(const auto& owner:queryPrograms)deferQueryProgramUnlocked(owner);
            for(auto* batch=directGratingBatch.get();batch;batch=batch->previous.get())deferQueryProgramUnlocked(batch->program);
            for(auto* batch=cameraGlassTraceBatch.get();batch;batch=batch->previous.get())deferQueryProgramUnlocked(batch->program);
            for(auto* cohort=directGratingCohort.get();cohort;cohort=cohort->previous.get())
                for(const auto& source:cohort->sources)if(source->batch)deferQueryProgramUnlocked(source->batch->program);
            for(auto* receiver=directReceiverResults.get();receiver;receiver=receiver->previous.get())deferQueryProgramUnlocked(receiver->program);
            for(auto* receiver=failedReceiverResources.get();receiver;receiver=receiver->previous.get())deferQueryProgramUnlocked(receiver->program);
            for(auto* coverage=cameraCoverageResults.get();coverage;coverage=coverage->previous.get())deferQueryProgramUnlocked(coverage->program);
            for(auto* coverage=failedCameraCoverageResources.get();coverage;coverage=coverage->previous.get())deferQueryProgramUnlocked(coverage->program);
            deferQueryProgramUnlocked(rasterCaptureProgram);
            cameraCoverageResults.reset();failedCameraCoverageResources.reset();rasterCaptureProgram.reset();
            directReceiverResults.reset();failedReceiverResources.reset();
            directGratingCohort.reset();directGratingBatch.reset();cameraGlassTraceBatch.reset();
            queryEvidence={};queryPrograms={};queryRetainedBytes=0;
        }
        if (previous != ownerContext)
            glfwMakeContextCurrent (previous);
    }

    void releaseTexturesUnlocked() noexcept
    {
        nativeAttempts={};nativeAttemptCount=0;nativeAttemptBytes=gratingNativeWork=gratingNativeDepositionWork=0;
        directReceiverResults.reset();failedReceiverResources.reset();activeReceiverStorage=0;
        cameraCoverageResults.reset();failedCameraCoverageResources.reset();activeCameraCoverageStorage=0;
        rasterCaptureProgram.reset();rasterCaptureWords={};rasterCaptureViewport={};rasterCaptureGeneration=0;
        cameraCoverageWork={};
        directTerminalResults.reset();terminalAttemptWork=terminalAttemptDepositionWork=directCohortDepositionWork=0;
        directGratingCohort.reset();directGratingBatch.reset();cameraGlassTraceBatch.reset();directGratingBytes=directGratingWork=0;cameraGlassTraceBytes=cameraGlassTraceWork=0;
        if (previousNoteTexture != 0) glDeleteTextures(1,&previousNoteTexture);
        previousNoteTexture = 0;
        if (diffractionQueryTexture != 0) glDeleteTextures(1, &diffractionQueryTexture);
        if (diffractionSamplesTexture != 0) glDeleteTextures(1, &diffractionSamplesTexture);
        diffractionQueryTexture = diffractionSamplesTexture = 0;
        glDeleteTextures(static_cast<int>(roughReceiptTextures.size()), roughReceiptTextures.data());
        roughReceiptTextures = {};
        roughReceiptComplete = false;
        roughTileSchedule.reset(); roughTileAggregate = {};
        if (staticResources) staticResources->gl.DeleteBuffers(static_cast<int>(causticBuffers.size()), causticBuffers.data());
        glDeleteTextures(static_cast<int>(causticTextures.size()), causticTextures.data());
        causticBuffers = {}; causticTextures = {}; causticReadback.reset();
        gratingStagingBinding.reset();gratingStagingSubmission={};
        if(directionalSourceBuffer&&staticResources)staticResources->gl.DeleteBuffers(1,&directionalSourceBuffer);
        directionalSourceBuffer=0;directionalSourceUpload={};
        if (gratingCameraDepthAttachment) glDeleteTextures(1,&gratingCameraDepthAttachment);
        gratingCameraDepthAttachment=0;gratingCameraRuntime.reset();
        if(staticResources)staticResources->gl.DeleteBuffers(2,gratingCameraSpectrumBuffers.data());
        gratingCameraSpectrumBuffers={};gratingCameraSpectrum.reset();gratingCameraSpectrumBytes=0;
        gratingCellSourceUpload.reset();gratingCellSourceBytes=0;gratingCoverageEvidence={};gratingCoverageBytes=0;
        queryEvidence={};queryPrograms={};queryRetainedBytes=0;
        if(gratingCameraSpectrumTexture)glDeleteTextures(1,&gratingCameraSpectrumTexture);
        gratingCameraSpectrumTexture=0;
        if (glassTexture != 0) glDeleteTextures(1, &glassTexture);
        if (opaqueTexture != 0) glDeleteTextures(1, &opaqueTexture);
        if (depthTexture_ != 0) glDeleteTextures(1, &depthTexture_);
        if (accumulationTexture_ != 0) glDeleteTextures(1, &accumulationTexture_);
        glDeleteTextures(static_cast<int>(rawTextures.size()), rawTextures.data());
        if (texture_ != 0) glDeleteTextures(1, &texture_);
        glassTexture = opaqueTexture = depthTexture_ = accumulationTexture_ = texture_ = 0;
        rawTextures = {};
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return privateCameraAttachments ? 0 : texture_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return privateCameraAttachments ? 0 : texture_; }
    std::uintptr_t depthImageHandle() const noexcept override { return privateCameraAttachments ? 0 : depthTexture_; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return privateCameraAttachments ? 0 : depthTexture_; }
    std::uintptr_t nativeResourceCacheIdentity() const noexcept override
    {
        return reinterpret_cast<std::uintptr_t> (staticResources.get());
    }
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        if (privateCameraAttachments) return {};
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
        if (privateCameraAttachments) return {};
        return { backend_, NativeTextureViewKind::Texture2D,
                 NativeTexturePixelFormat::R32Float, depthTexture_, depthTexture_, width_, height_, 1,
                 true, reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_,
                 colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
                 NativeTextureRowOrder::BottomFirst };
    }

    NativeTextureViewDescriptor passTextureDescriptor(renderpassoutput::Output output) const noexcept override
    {
        if (privateCameraAttachments) return {};
        for (std::size_t i = 2; i < renderpasscomposite::kInputs.size(); ++i)
            if (renderpasscomposite::kInputs[i] == output)
                return { backend_, NativeTextureViewKind::Texture2D, rawFormats[i - 2],
                    rawTextures[i - 2], rawTextures[i - 2], width_, height_, 1, true,
                    reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_,
                    colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
                    NativeTextureRowOrder::BottomFirst };
        return NativeFixtureSceneFrame::passTextureDescriptor(output);
    }

    const NativeFixtureSceneFrame& owner() const noexcept override {return *this;}
    const glasscaustics::NativeDirectionalSourceUpload& upload() const noexcept override {
        static const glasscaustics::NativeDirectionalSourceUpload empty;
        return nativeAccountingAccessible()?directionalSourceUpload:empty;
    }
    const glasscaustics::NativeGratingCellSourceUpload& cellUpload() const noexcept override
    {
        static const glasscaustics::NativeGratingCellSourceUpload empty;
        return gratingCellSourceUpload&&nativeAccountingAccessible()?*gratingCellSourceUpload:empty;
    }
    const glasscaustics::NativeGratingCellCoverageEvidence& coverageEvidence() const noexcept override {
        static const glasscaustics::NativeGratingCellCoverageEvidence empty;
        return nativeAccountingAccessible()?gratingCoverageEvidence:empty;
    }
    const glasscaustics::NativeDirectionalSourceBinding& directSourceBinding() const noexcept override {return *this;}
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& directGratingBinding() const noexcept override {return gratingStagingBinding;}
    const glasscaustics::NativeDirectGratingEndpointUpload& directEndpointUpload() const noexcept override
    {
        static const glasscaustics::NativeDirectGratingEndpointUpload empty;
        return directGratingBatch?directGratingBatch->upload:empty;
    }
    bool validateDirectProductionQuery(const glasscaustics::SubmissionIdentity&,
        const glasscaustics::NativeDirectGratingContributionReadback&,std::string&) const override;
    bool validateDirectBatch(const glasscaustics::GratingIncidentBinding&,const OpenGlDirectGratingBatch&,
        const glasscaustics::NativeDirectionalSourceUpload&,const glasscaustics::SubmissionIdentity&,
        const glasscaustics::NativeDirectGratingContributionReadback&,bool,std::string&) const;
    const NativeFixtureSceneFrame& cohortOwner() const noexcept override {return *this;}
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& directCohort() const noexcept override
    {
        static const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort> empty;
        return directGratingCohort?directGratingCohort->binding:empty;
    }
    std::size_t directSourceCount() const noexcept override {return directGratingCohort?directGratingCohort->sources.size():0;}
    const glasscaustics::NativeDirectGratingIrradianceBinding* directSourceBatch(std::size_t index) const noexcept override
    {return directGratingCohort&&index<directGratingCohort->sources.size()?directGratingCohort->sources[index].get():nullptr;}
    bool validateDirectCohortProductionQueries(const glasscaustics::SubmissionIdentity&,
        const std::vector<glasscaustics::NativeDirectGratingContributionReadback>&,std::string&) const override;
    bool beginDirectCohortTerminalLease(const glasscaustics::SubmissionIdentity&,double,
        const std::vector<glasscaustics::NativeDirectGratingContributionReadback>&,
        std::unique_ptr<glasscaustics::NativeDirectGratingTerminalLease>&,
        glasscaustics::DirectGratingTerminalNativeBudget&,std::string&) const override;
    bool reserveDirectCohortTerminalLease(glasscaustics::NativeDirectGratingTerminalLease&,
        const glasscaustics::DirectGratingTerminalOrderBounds&,const glasscaustics::DirectGratingTerminalOrderLedger&,std::string&) const override;
    bool finishDirectCohortTerminalLease(glasscaustics::NativeDirectGratingTerminalLease&,
        const glasscaustics::DirectGratingTerminalOrderLedger&,
        std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>&,std::string&) const override;
    bool validateDirectCohortTerminalResult(const glasscaustics::NativeDirectGratingTerminalLease&,
        const glasscaustics::DirectGratingTerminalOrderLedger&,std::string&) const override;
    bool beginDirectTerminalReceiverLease(const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>&,
        const glasscaustics::SubmissionIdentity&,double,std::unique_ptr<glasscaustics::NativeDirectGratingTerminalReceiverLease>&,
        glasscaustics::DirectGratingTerminalReceiverNativeBudget&,std::string&) const override;
    bool reserveDirectTerminalReceiverLease(glasscaustics::NativeDirectGratingTerminalReceiverLease&,
        const glasscaustics::DirectGratingTerminalReceiverBounds&,const glasscaustics::DirectGratingTerminalReceiverReadback&,std::string&) const override;
    bool dispatchDirectTerminalReceivers(glasscaustics::NativeDirectGratingTerminalReceiverLease&,
        const std::vector<glasscaustics::NativeDirectGratingTerminalReceiverInput>&,
        glasscaustics::DirectGratingTerminalReceiverReadback&,std::string&) const override;
    bool finishDirectTerminalReceiverLease(glasscaustics::NativeDirectGratingTerminalReceiverLease&,
        const glasscaustics::DirectGratingTerminalReceiverReadback&,
        std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>&,std::string&) const override;
    bool validateDirectTerminalReceiverResult(const glasscaustics::NativeDirectGratingTerminalReceiverLease&,
        const glasscaustics::DirectGratingTerminalReceiverReadback&,std::string&) const override;
    const NativeFixtureSceneFrame& coverageOwner() const noexcept override{return *this;}
    bool beginCameraCellCoverageLease(const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>&,
        const glasscaustics::SubmissionIdentity&,double,std::unique_ptr<glasscaustics::NativeCameraCellCoverageLease>&,
        glasscaustics::NativeCameraCellCoverageBudget&,std::string&) const override;
    bool reserveCameraCellCoverageLease(glasscaustics::NativeCameraCellCoverageLease&,const glasscaustics::CameraCellCoverageBounds&,
        const glasscaustics::CameraCellCoverageReadback&,std::string&) const override;
    bool dispatchCameraCellCoverage(glasscaustics::NativeCameraCellCoverageLease&,const glasscaustics::CameraCellCoveragePreparation&,
        glasscaustics::CameraCellCoverageProof&,std::string&) const override;
    bool validateCameraCellCoverageProof(const glasscaustics::NativeCameraCellCoverageLease&,
        const glasscaustics::CameraCellCoverageReadback&,std::string&) const override;
    bool finishCameraCellCoverageLease(glasscaustics::NativeCameraCellCoverageLease&,const glasscaustics::CameraCellCoverageReadback&,
        std::shared_ptr<const glasscaustics::NativeCameraCellCoverageResult>&,std::string&) const override;
    bool directDepositionBudget(std::uint64_t& total) const noexcept
    {
        if(!gratingStagingBinding||terminalAttemptWork>directGratingWork)return false;
        total=gratingStagingBinding->frame.bounds.deposition.depositionWork;
        for(auto work:{gratingStagingBinding->bounds.orderWork,directCohortDepositionWork,terminalAttemptDepositionWork,gratingNativeDepositionWork,
            directGratingWork-terminalAttemptWork,cameraGlassTraceWork,gratingCameraWork,gratingCellWork,gratingCoverageWork})
            if(!renderpassoutput::detail::checkedAdd(total,work,total))return false;
        return total<=glasscaustics::kMaximumDepositionWork;
    }
    bool reconcileNativeAccounting(std::string& error) const;
    bool nativeAccountingAccessible() const noexcept {
        try {std::string error;return reconcileNativeAccounting(error);}catch(...){return false;}
    }
    bool reserveNativeAttempt(OpenGlNativeAttemptKind,const std::shared_ptr<const glasscaustics::GratingIncidentBinding>&,
        const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>&,std::size_t,unsigned,std::size_t,
        std::uint64_t,std::string&,const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& = {}) const;
    const NativeFixtureSceneFrame& cameraTraceOwner() const noexcept override {return *this;}
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& cameraTraceGratingBinding() const noexcept override {return gratingStagingBinding;}
    const glasscaustics::NativeGlassCameraGratingTraceUpload& cameraTraceInputUpload() const noexcept override
    {
        static const glasscaustics::NativeGlassCameraGratingTraceUpload empty;
        return cameraGlassTraceBatch?cameraGlassTraceBatch->upload:empty;
    }
    bool validateCameraTraceProductionQuery(const glasscaustics::SubmissionIdentity&,
        const glasscaustics::NativeGlassCameraGratingTraceReadback&,std::string&) const override;
    void retainQueryArithmetic(OpenGlQueryArithmeticRole role,std::shared_ptr<OpenGlQueryProgram> program,
        std::shared_ptr<const OpenGlQueryArithmeticEvidence> evidence) const noexcept
    {
        const auto index=std::size_t(role);queryPrograms[index]=std::move(program);queryEvidence[index]=std::move(evidence);
        queryRetainedBytes=0;
        for(std::size_t at=0;at<queryPrograms.size();++at)if(queryPrograms[at]&&queryEvidence[at])
            queryRetainedBytes+=queryPrograms[at]->retainedBytes+sizeof(OpenGlQueryArithmeticEvidence)+128
                +queryEvidence[at]->submission.frameIdentity.capacity()+1;
    }
    unsigned directionalSourceBuffer=0;
    glasscaustics::NativeDirectionalSourceUpload directionalSourceUpload;

    std::string backend_ = "opengl";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    GLFWwindow* ownerContext = nullptr;
    std::shared_ptr<const OpenGlFixtureSceneResources> staticResources;
    bool linearColor = false;
    bool privateCameraAttachments=false;
    bool gratingCameraBaseComplete=false;
    unsigned gratingCameraDepthAttachment=0;
    std::optional<NativeFixtureSceneRuntimeInputs> gratingCameraRuntime;
    double gratingCameraEvaluationTime=0;
    mutable std::array<unsigned,2> gratingCameraSpectrumBuffers{};
    mutable unsigned gratingCameraSpectrumTexture=0;
    mutable std::uint64_t gratingCameraSpectrumBytes=0;
    mutable std::shared_ptr<const OpenGlGratingCameraSpectrum> gratingCameraSpectrum;
    mutable std::shared_ptr<const glasscaustics::NativeGratingCellSourceUpload> gratingCellSourceUpload;
    mutable std::uint64_t gratingCellSourceBytes=0;
    mutable glasscaustics::NativeGratingCellCoverageEvidence gratingCoverageEvidence;
    mutable std::uint64_t gratingCoverageBytes=0;
    mutable std::array<std::shared_ptr<OpenGlQueryProgram>,std::size_t(OpenGlQueryArithmeticRole::Count)> queryPrograms{};
    mutable std::array<std::shared_ptr<const OpenGlQueryArithmeticEvidence>,std::size_t(OpenGlQueryArithmeticRole::Count)> queryEvidence{};
    mutable std::uint64_t queryRetainedBytes=0;
    mutable std::unique_ptr<OpenGlDirectGratingCohort> directGratingCohort;
    mutable std::unique_ptr<OpenGlDirectGratingBatch> directGratingBatch;
    mutable std::uint64_t directGratingBytes=0,directGratingWork=0;
    // Every accepted owner stays in this charged chain, even if callers retain
    // earlier returned handles. Registration uses one staged node, no vector growth.
    mutable std::unique_ptr<OpenGlDirectTerminalRecord> directTerminalResults;
    mutable const OpenGlDirectTerminalLease* activeDirectTerminalLease=nullptr;
    mutable std::uint64_t terminalAttemptWork=0,terminalAttemptDepositionWork=0,directCohortDepositionWork=0;
    // Fixed registration storage is admitted with sizeof(frame). Frozen owners
    // survive failed allocation/dispatch/finish; no allocation publishes a record.
    static constexpr std::size_t kMaximumNativeAttempts=1024;
    mutable std::array<OpenGlNativeAttempt,kMaximumNativeAttempts> nativeAttempts{};
    mutable std::size_t nativeAttemptCount=0;
    mutable std::uint64_t nativeAttemptBytes=0,gratingNativeWork=0,gratingNativeDepositionWork=0;
    mutable std::uint64_t activeTerminalStorage=0,activeReceiverStorage=0;
    mutable std::unique_ptr<OpenGlDirectTerminalReceiverRecord> directReceiverResults,failedReceiverResources;
    mutable const OpenGlDirectTerminalReceiverLease* activeDirectReceiverLease=nullptr;
    mutable std::unique_ptr<OpenGlCameraCoverageRecord> cameraCoverageResults,failedCameraCoverageResources;
    mutable const OpenGlCameraCoverageLease* activeCameraCoverageLease=nullptr;
    mutable std::uint64_t activeCameraCoverageStorage=0;
    mutable std::array<std::uint64_t,8> cameraCoverageWork{};
    std::shared_ptr<OpenGlQueryProgram> rasterCaptureProgram;
    std::array<std::uint32_t,13> rasterCaptureWords{};
    std::array<int,4> rasterCaptureViewport{};
    std::uint64_t rasterCaptureGeneration=0;
    mutable std::unique_ptr<OpenGlCameraGlassGratingBatch> cameraGlassTraceBatch;
    mutable std::uint64_t cameraGlassTraceBytes=0,cameraGlassTraceWork=0;
    mutable std::uint64_t gratingCameraWork=0,gratingCellWork=0,gratingCoverageWork=0;
    std::shared_ptr<const NativeFixtureSceneFrame> materialFrameTexture;
    std::map<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint,
        std::shared_ptr<const NativeFixtureSceneFrame>> materialFrameTextures;
    std::uint64_t rendererGeneration_ = 0;
    unsigned texture_ = 0;
    unsigned opaqueTexture = 0;
    unsigned glassTexture = 0;
    unsigned diffractionQueryTexture = 0, diffractionSamplesTexture = 0;
    unsigned previousNoteTexture = 0;
    unsigned accumulationTexture_ = 0;
    unsigned depthTexture_ = 0;
    std::array<unsigned, 3> roughReceiptTextures {};
    bool roughReceiptComplete = false;
    std::shared_ptr<const roughglasstiles::Schedule> roughTileSchedule;
    roughglasstiles::Aggregate roughTileAggregate;
    std::array<unsigned, 7> causticBuffers {};
    OpenGlGlassCausticDiagnostics causticDiagnostics;
    std::array<unsigned, 4> causticTextures {};
    std::shared_ptr<const glasscaustics::NativeReadback> causticReadback;
    glasscaustics::EnergyLedger causticEnergy;
    std::shared_ptr<const glasscaustics::GratingIncidentBinding> gratingStagingBinding;
    glasscaustics::SubmissionIdentity gratingStagingSubmission;
    bool readRoughReceiptOwned(std::size_t index, NativeRawPassPixels& pixels,
                              std::string& error) const
    {
        pixels = {};
        if (!roughReceiptComplete || (roughTileSchedule && roughTileSchedule->tiles().size()!=1)
            || index >= roughReceiptTextures.size()
            || !roughReceiptTextures[index] || !staticResources
            || !staticResources->smoothGlass.enabled || staticResources->smoothGlass.roughness <= 0
            || staticResources->rendererGeneration != rendererGeneration_
            || glfwGetCurrentContext() != ownerContext)
        { error = "OpenGL rough glass receipt requires its completed frame owner"; return false; }
        const NativeTextureViewDescriptor descriptor { backend_, NativeTextureViewKind::Texture2D,
            NativeTexturePixelFormat::Rgba32Float, roughReceiptTextures[index], roughReceiptTextures[index],
            width_, height_, 1, true, reinterpret_cast<std::uintptr_t>(ownerContext), rendererGeneration_,
            colortransform::ColorSpace::Unspecified, colortransform::TransferFunction::Unspecified,
            NativeTextureRowOrder::BottomFirst };
        return readTextureOwned(descriptor, pixels, error);
    }
    bool readRawPass(renderpassoutput::Output output, NativeRawPassPixels& pixels,
                     std::string& error) const override
    {
        pixels = {};
        std::lock_guard<std::mutex> lock(fixtureMutex());
        return readRawPassOwned(output, pixels, error);
    }
    // The renderer already holds the owning context/device mutex.
    bool readRawPassOwned(renderpassoutput::Output output, arbitgpu::NativeRawPassPixels& pixels,
                          std::string& error) const
    {
        pixels = {};
        const auto descriptor = passTextureDescriptor(output);
        if (!descriptor.complete() || !rawPassFormatMatches(output, descriptor.format)
            || !nativeFixtureDimensionsWithinBounds(width_, height_)
            || !staticResources || glfwGetCurrentContext() != ownerContext)
        { error = "OpenGL raw pass requires its live owning context and supported attachment"; return false; }
        return readTextureOwned(descriptor, pixels, error);
    }
    bool readTextureOwned(const NativeTextureViewDescriptor& descriptor, NativeRawPassPixels& pixels,
                          std::string& error) const
    {
        if (!descriptor.complete() || !nativeFixtureDimensionsWithinBounds(width_, height_)
            || !staticResources || glfwGetCurrentContext() != ownerContext)
        { error = "OpenGL texture readback requires its live owning context"; return false; }
        GLenum layout = GL_RED, type = GL_UNSIGNED_BYTE;
        switch (descriptor.format)
        {
            case NativeTexturePixelFormat::Rgba8Unorm: layout = GL_RGBA; break;
            case NativeTexturePixelFormat::Rgba16Float: layout = GL_RGBA; type = GL_HALF_FLOAT; break;
            case NativeTexturePixelFormat::Rgba32Float: layout = GL_RGBA; type = GL_FLOAT; break;
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

// Shared normal camera attachments. Private grating staging reserves these on
// its existing owner, but does not mark them rendered or publish their views.
static bool allocateFixtureCameraAttachments(
    const arbitgl::GlFuncs& gl, OpenGlFixtureSceneFrame* frame, bool diffractionFrame,
    unsigned& framebuffer, unsigned& depthAttachment, std::string& error)
{
    glGenTextures (1, &frame->texture_);
    glBindTexture (GL_TEXTURE_2D, frame->texture_);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexImage2D (GL_TEXTURE_2D, 0, frame->linearColor ? GL_RGBA16F : GL_RGBA8, static_cast<int> (frame->width_),
                  static_cast<int> (frame->height_), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    if (diffractionFrame)
    {
        glGenTextures(1, &frame->accumulationTexture_);
        glBindTexture(GL_TEXTURE_2D, frame->accumulationTexture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, static_cast<int>(frame->width_),
            static_cast<int>(frame->height_), 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
    }
    glGenTextures (1, &frame->depthTexture_);
    glBindTexture (GL_TEXTURE_2D, frame->depthTexture_);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_R32F,
                  static_cast<int> (frame->width_), static_cast<int> (frame->height_), 0,
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
                  static_cast<int> (frame->width_), static_cast<int> (frame->height_), 0,
                  GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
    gl.GenFramebuffers (1, &framebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, diffractionFrame
                                 ? frame->accumulationTexture_ : frame->texture_, 0);
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
        glTexImage2D(GL_TEXTURE_2D, 0, rawFormats[i], static_cast<int>(frame->width_),
            static_cast<int>(frame->height_), 0, i == 5 ? GL_RG : i < 2 ? GL_RGBA : i == 2 ? GL_RED : GL_RED_INTEGER,
            i == 5 || i < 2 ? GL_FLOAT : i == 2 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_INT, nullptr);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 2 + static_cast<unsigned>(i),
            GL_TEXTURE_2D, frame->rawTextures[i], 0);
    }
    const unsigned drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4,
        GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7 };
    gl.DrawBuffers (8, drawBuffers);
    if (frame->texture_ == 0 || frame->depthTexture_ == 0
        || (diffractionFrame && frame->accumulationTexture_ == 0)
        || depthAttachment == 0 || framebuffer == 0
        || gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        error = "OpenGL fixture GPU resource creation failed";
        return false;
    }

    return true;
}

static bool fixtureCameraAttachmentsMatch(const OpenGlFixtureSceneFrame& frame,
                                          std::string& error)
{
    const std::array<unsigned,10> textures {frame.texture_,frame.accumulationTexture_,
        frame.depthTexture_,frame.gratingCameraDepthAttachment,frame.rawTextures[0],frame.rawTextures[1],
        frame.rawTextures[2],frame.rawTextures[3],frame.rawTextures[4],frame.rawTextures[5]};
    const std::array<int,10> formats {GL_RGBA16F,GL_RGBA16F,GL_R32F,GL_DEPTH_COMPONENT24,
        GL_RGBA16F,GL_RGBA16F,GL_R8,GL_R32UI,GL_R32UI,GL_RG16F};
    for(std::size_t i=0;i<textures.size();++i) {
        if(!textures[i]||!glIsTexture(textures[i])) {error="OpenGL private camera attachment is absent";return false;}
        glBindTexture(GL_TEXTURE_2D,textures[i]);
        int width=0,height=0,format=0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&height);
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&format);
        if(width!=int(frame.width_)||height!=int(frame.height_)||format!=formats[i])
        {error="OpenGL private camera attachment differs from admitted storage";return false;}
    }
    if(glGetError()!=GL_NO_ERROR) {error="OpenGL private camera attachment allocation failed";return false;}
    return true;
}

static std::uint64_t fixtureLinearCameraStorageStride()
{
    std::uint64_t bytes = 8 + 8 + 4 + 4; // Color, accumulation, linear depth, raster depth.
    for (const auto format : OpenGlFixtureSceneFrame::rawFormats)
        bytes += rawPassChannels(format) * rawPassScalarBytes(format);
    return bytes;
}

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

// Saves back-face culling and front-face state before a member with a mixed joint palette
// disables culling. The destructor restores the saved state when the enclosing scope ends.
class OpenGlCullStateScope final
{
public:
    explicit OpenGlCullStateScope (bool engaged) noexcept : engaged_ (engaged)
    {
        if (!engaged_) return;
        cullEnabled_ = glIsEnabled (GL_CULL_FACE) == GL_TRUE;
        glGetIntegerv (GL_FRONT_FACE, &frontFace_);
    }
    ~OpenGlCullStateScope()
    {
        if (!engaged_) return;
        if (cullEnabled_) glEnable (GL_CULL_FACE);
        else glDisable (GL_CULL_FACE);
        glFrontFace (static_cast<unsigned> (frontFace_));
    }
    OpenGlCullStateScope (const OpenGlCullStateScope&) = delete;
    OpenGlCullStateScope& operator= (const OpenGlCullStateScope&) = delete;

private:
    bool engaged_ = false;
    bool cullEnabled_ = false;
    GLint frontFace_ = GL_CCW;
};

// Front-face choice for one draw. The vertex shader applies objectMatrix after the palette skin
// and the composition point, so a deformed member's joint and point mirror state flips the
// objectMatrix sign.
bool fixtureDrawMirrored (const std::array<float, 16>& objectMatrix,
    const OpenGlFixtureSceneResources::DeformedWinding* deformed) noexcept
{
    const bool objectMirrored = fixtureMatrixDeterminant3x3 (objectMatrix) < 0.0f;
    return deformed != nullptr ? objectMirrored != deformed->skinOrPointMirrored : objectMirrored;
}

// Mirror state of one deformed member from its palette and composition post. Every palette
// entry contributes the sign of det(upper 3x3), because the joint use of each vertex is not
// cheap to read here. Mixed signs cannot share one front face, so the draw disables culling.
OpenGlFixtureSceneResources::DeformedWinding deformedMemberWinding (const NativeDeformationFrameData& data) noexcept
{
    const auto jointCount = std::min<std::size_t> (data.jointCount, data.jointPalette.size() / 16u);
    std::size_t mirroredJoints = 0;
    for (std::size_t joint = 0; joint < jointCount; ++joint)
    {
        std::array<float, 16> matrix {};
        std::copy_n (data.jointPalette.begin() + static_cast<std::ptrdiff_t> (joint * 16u), 16, matrix.begin());
        if (fixtureMatrixDeterminant3x3 (matrix) < 0.0f) ++mirroredJoints;
    }
    const bool skinMirrored = jointCount != 0 && mirroredJoints == jointCount;
    const bool pointMirrored = data.compositionPost.control()[1] == 1.0f
        && fixtureMatrixDeterminant3x3 (data.compositionPost.point()) < 0.0f;
    OpenGlFixtureSceneResources::DeformedWinding winding;
    winding.jointsMixed = mirroredJoints != 0 && mirroredJoints != jointCount;
    winding.skinOrPointMirrored = skinMirrored != pointMirrored;
    return winding;
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

// Most unfinished submissions one context keeps. A later submission evicts the
// oldest entry and deletes its GL objects.
constexpr std::size_t kMaximumInFlightFixturePrograms = 4;

// A key submitted again within this window after it was evicted or finished compiles
// synchronously instead of going back to the async path.
constexpr auto kFixtureResubmitWindow = std::chrono::seconds (2);

// A scene program whose shaders and link were submitted without waiting for their
// results. Its GL names belong to this entry until finishFixtureProgram succeeds or
// the entry is dropped. Context release deletes them through deleteInFlightObjects.
struct OpenGlFixtureInFlightProgram
{
    arbitgl::GlFuncs gl {};
    unsigned program = 0;
    unsigned vertexShader = 0;
    unsigned fragmentShader = 0;
    // The feature mask the fragment text was built from. Logged with the key.
    unsigned fragmentFeatures = 0;
    std::chrono::steady_clock::time_point submitted {};
};

// Decided from the prepared material programs and grating admission, before the program
// lookup. Smooth glass is on when any program has a nonzero transport mode, the value that
// each draw uploads to uGlassTransport.x. Diffraction is on when any program is a diffraction
// material. The grating path turns on both. Each test errs toward on.
unsigned fixtureFragmentFeatures (const NativeFixtureSurfaceMaterialProgram* root, bool distinctGrating) noexcept
{
    if (distinctGrating) return kFixtureFragmentAllFeatures;
    unsigned features = 0;
    const auto visit = [&features] (const NativeFixtureSurfaceMaterialProgram& program)
    {
        if (program.parameters.identifiers[1] != 0) features |= kFixtureFragmentSmoothGlass;
        if (program.kind == NativeFixtureMaterialKind::DiffractionReflective) features |= kFixtureFragmentDiffraction;
    };
    if (root != nullptr)
    {
        visit (*root);
        for (const auto& object : root->objectPrograms)
            if (object) visit (*object);
    }
    return features;
}

// In-memory key of one scene program. The disk cache keys on the final fragment text,
// which already contains the feature defines.
std::string fixtureProgramCacheKey (const std::string& vertexSource, unsigned features)
{
    return vertexSource + "\nfragmentFeatures=" + std::to_string (features);
}

std::string fixtureFragmentShaderSource (unsigned features)
{
    const bool smoothGlass = (features & kFixtureFragmentSmoothGlass) != 0;
    const bool diffraction = (features & kFixtureFragmentDiffraction) != 0;
    // The head opens with the version line. The defines go directly after it.
    const std::string version = "#version 330 core\n";
    const std::string head = kFixtureFragmentShaderHead;
    std::string text = version
        + std::string ("#define FIXTURE_SMOOTH_GLASS ") + (smoothGlass ? "1" : "0") + "\n"
        + std::string ("#define FIXTURE_DIFFRACTION ") + (diffraction ? "1" : "0") + "\n"
        + head.substr (version.size());
    // The smooth-glass transport text declares these two uniforms. Without it, declare them here.
    if (! smoothGlass) text += "uniform vec4 uRoughCameraControl;\nuniform vec4 uGlassTransport;\n";
    // The direct opaque BVH stays on. Its visibility result can be needed without a feature bit.
    text += opaquevisibility::glslShader() + analyticsky::shader (false) + environmentimage::shader (false) + kFixturePassCompositeShader;
    if (smoothGlass) text += smoothglass::queryShader (false) + smoothglass::transportShader (false, true);
    if (diffraction) text += diffractionQueryShader();
    text += kFixtureFragmentShaderTail;
    return text;
}

void deleteInFlightShaders (OpenGlFixtureInFlightProgram& entry) noexcept
{
    if (entry.fragmentShader != 0) entry.gl.DeleteShader (entry.fragmentShader);
    if (entry.vertexShader != 0) entry.gl.DeleteShader (entry.vertexShader);
    entry.fragmentShader = 0;
    entry.vertexShader = 0;
}

// The program goes first, so the shaders it still holds are released with it.
void deleteInFlightObjects (OpenGlFixtureInFlightProgram& entry) noexcept
{
    if (entry.program != 0) entry.gl.DeleteProgram (entry.program);
    entry.program = 0;
    deleteInFlightShaders (entry);
}

std::string fixtureShaderFailure (const arbitgl::GlFuncs& gl, unsigned shader)
{
    char log[1024] = {};
    int length = 0;
    gl.GetShaderInfoLog (shader, static_cast<int> (sizeof (log)), &length, log);
    return std::string ("OpenGL fixture shader compilation failed: ")
        + std::string (log, static_cast<std::size_t> (std::max (length, 0)));
}

// Creates both shaders, starts both compiles, and starts the link. It reads no
// compile or link status, so it never waits on the driver. The caller polls
// GL_COMPLETION_STATUS_KHR on the program before it calls finishFixtureProgram.
std::string submitFixtureProgram (const arbitgl::GlFuncs& gl,
                                  arbitglprogramcache::ProgramBinaryCache& binaryCache,
                                  const std::string& vertexSource,
                                  const std::string& fragmentSource,
                                  unsigned fragmentFeatures,
                                  OpenGlFixtureInFlightProgram& entry)
{
    entry = OpenGlFixtureInFlightProgram {};
    entry.gl = gl;
    entry.fragmentFeatures = fragmentFeatures;
    entry.submitted = std::chrono::steady_clock::now();
    entry.vertexShader = gl.CreateShader (GL_VERTEX_SHADER);
    entry.fragmentShader = gl.CreateShader (GL_FRAGMENT_SHADER);
    entry.program = gl.CreateProgram();
    if (entry.vertexShader == 0 || entry.fragmentShader == 0 || entry.program == 0)
    {
        deleteInFlightObjects (entry);
        return "OpenGL fixture shader allocation failed";
    }
    const char* vertex = vertexSource.c_str();
    const char* fragment = fragmentSource.c_str();
    gl.ShaderSource (entry.vertexShader, 1, &vertex, nullptr);
    gl.CompileShader (entry.vertexShader);
    gl.ShaderSource (entry.fragmentShader, 1, &fragment, nullptr);
    gl.CompileShader (entry.fragmentShader);
    binaryCache.prepareLink (entry.program);
    gl.AttachShader (entry.program, entry.vertexShader);
    gl.AttachShader (entry.program, entry.fragmentShader);
    gl.LinkProgram (entry.program);
    return {};
}

// Reads the compile and link results. Call it only after GL_COMPLETION_STATUS_KHR
// reports completion, so these queries do not wait on the driver. Returns an empty
// string when the program linked.
std::string finishFixtureProgram (const OpenGlFixtureInFlightProgram& entry)
{
    for (const unsigned shader : { entry.vertexShader, entry.fragmentShader })
    {
        int compiled = GL_FALSE;
        entry.gl.GetShaderiv (shader, GL_COMPILE_STATUS, &compiled);
        if (compiled != GL_TRUE) return fixtureShaderFailure (entry.gl, shader);
    }
    int linked = GL_FALSE;
    entry.gl.GetProgramiv (entry.program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) return {};
    char log[1024] = {};
    int length = 0;
    entry.gl.GetProgramInfoLog (entry.program, static_cast<int> (sizeof (log)), &length, log);
    return std::string ("OpenGL fixture program link failed: ")
        + std::string (log, static_cast<std::size_t> (std::max (length, 0)));
}

class OpenGlFixtureSceneBackend final : public NativeFixtureSceneBackend
{
public:
    bool supportsRetainedNoteMotion() const noexcept override { return true; }

    void releaseCachedProgramsForCurrentContext() noexcept override
    {
        auto* context = glfwGetCurrentContext();
        if (context == nullptr) return;
        std::lock_guard<std::mutex> lock (fixtureMutex());
        const auto found=programCache_.find(context);
        if(found!=programCache_.end()&&found->second.queryContext)found->second.queryContext->retireUnlocked();
        // In-flight programs hold GL names that no SharedProgram releases. Delete them
        // while their context is current, before the entry goes away.
        if(found!=programCache_.end())
            for(auto& flying:found->second.inFlight)deleteInFlightObjects(flying.second);
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

    bool supportsAuthoredSpectralReceiverTransport() const noexcept override
    {
        return diffractiontransportnative::consumerCanRender(kOpenGlDiffractionConsumer,
            NativeFixtureMaterialBackend::OpenGl);
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
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram,
        NativeFixturePrepareUse = NativeFixturePrepareUse::Blocking) override;

    NativeFixtureScenePreparation prepareGeometryInstances (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>& geometryAdmission,
        bool diagnosticInstanceIdentityColors,
        const std::vector<videowire::geometry::AttributeData>* frameAttributes = nullptr,
        NativeFixturePrepareUse use = NativeFixturePrepareUse::Blocking) override
    {
        return prepareGeometryInstancesImpl(scene,requestedMaterialProgram,geometryAdmission,
            diagnosticInstanceIdentityColors,frameAttributes,nullptr,{},0,use);
    }

    NativeFixtureScenePreparation prepareGeometryInstancesImpl(
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>& geometryAdmission,
        bool diagnosticInstanceIdentityColors,const std::vector<videowire::geometry::AttributeData>* frameAttributes,
        const smoothglass::DistinctGratingAdmission* distinctGrating=nullptr,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& gratingLeaf={},
        std::uint64_t privateStaticBytes=0, NativeFixturePrepareUse use=NativeFixturePrepareUse::Blocking)
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
            || validNativeFixtureMaterialPrograms(*materialProgram, *scene, NativeFixtureMaterialBackend::OpenGl,
                diffractiontransportnative::consumerCanRender(kOpenGlDiffractionConsumer, NativeFixtureMaterialBackend::OpenGl),
                distinctGrating?NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders:NativeFixtureDiffractionConsumer::FiniteLobe);
        if (materialProgram != nullptr
            && (materialProgram->layoutVersion
                    != NativeFixtureSurfaceMaterialProgram::kLayoutVersion
                || materialProgram->backend != NativeFixtureMaterialBackend::OpenGl
                || materialProgram->programIdentity.empty()
                || materialProgram->bindingDigest.empty()
                || (materialProgram->vertexProgram
                    && (materialProgram->kind != NativeFixtureMaterialKind::SurfacePbr
                        || scene->objectCount != 1
                        || materialProgram->vertexProgram->programIdentity() != materialProgram->programIdentity))
                || ! materialKindValid))
        {
            result.error = "OpenGL fixture preparation requires an exact bounded material binding";
            return result;
        }
        if (!nativeObjectInstancesSupported(*scene, materialProgram.get(),
                geometryAdmission != nullptr || frameAttributes != nullptr || distinctGrating != nullptr,
                result.error))
            return result;

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
        // Soft-loaded. A driver without these entry points still runs the fixture
        // program. It just compiles it again in each session.
        arbitgl::loadGlProgramBinaryFunctions (gl);
        // Soft-loaded. Without these, Preview compiles the scene program synchronously.
        arbitgl::loadGlParallelShaderCompileFunctions (gl);
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
        // Extend the existing context cache before any query-program identity
        // is allocated. Its epoch survives ordinary entry eviction and is
        // retired by the existing current-context teardown hook.
        const auto existingContext=programCache_.find(glfwGetCurrentContext());
        if(existingContext!=programCache_.end()&&existingContext->second.queryContext)
            existingContext->second.queryContext->drainDeferredUnlocked();
        std::uint64_t staticOwned=0;
        if(!renderpassoutput::detail::checkedAdd(privateStaticBytes,textureBytes,staticOwned)
            ||!renderpassoutput::detail::checkedAdd(staticOwned,sizeof(OpenGlQueryContext)+512,staticOwned)
            ||staticOwned>renderpassoutput::kMaximumTotalBytes)
        {result.error="OpenGL query context ownership exceeds static admission";return result;}
        OpenGlFixtureState previous (gl);
        auto resources = std::make_shared<OpenGlFixtureSceneResources>();
        resources->ownerContext = glfwGetCurrentContext();
        resources->rendererGeneration = nextFixtureRendererGeneration();
        resources->gl = gl;
        auto& contextCache=programCache_[resources->ownerContext];
        if(!contextCache.queryContext) {
            const auto generation=nextQueryOwnerGeneration();
            if(!generation) {resources->ownerContext=nullptr;result.error="OpenGL query context generation exhausted";return result;}
            auto context=std::make_shared<OpenGlQueryContext>();context->context=resources->ownerContext;context->generation=generation;
            glGetIntegerv(GL_MAJOR_VERSION,&context->major);glGetIntegerv(GL_MINOR_VERSION,&context->minor);
            glGetIntegerv(GL_CONTEXT_PROFILE_MASK,&context->profile);contextCache.queryContext=std::move(context);
        }
        resources->queryContext=contextCache.queryContext;
        bool privatePrepared=false;int privateToken=0;
        auto privateCleanup=[&](int*) {
            if(distinctGrating&&!privatePrepared&&resources) {
                resources->destroyUnlocked();resources->ownerContext=nullptr;previous.restore();
            }
        };
        const std::unique_ptr<int,decltype(privateCleanup)> privateGuard(&privateToken,privateCleanup);

        resources->snapshot = scene;
        resources->materialProgram = materialProgram;
        resources->materialProgramSource = requestedMaterialProgram;
        resources->gratingLeaf=gratingLeaf;resources->gratingStaticBytes=privateStaticBytes+sizeof(OpenGlQueryContext)+512;
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

        // Vertex source keys every cached and in-flight program of this context. It is
        // CPU-only, so it is computed before the uploads. Its failure is still reported
        // at the original position, after the uploads, as before.
        std::string vertexError;
        const auto vertexSource = fixtureVertexModifierShader (kFixtureVertexShader,
            materialProgram != nullptr ? materialProgram->vertexProgram.get() : nullptr,
            videohelper::materialprogram::BackendTarget::OpenGl, vertexError);
        // The fragment feature mask is known here, before any compile. It is part of the key.
        const auto fragmentFeatures = fixtureFragmentFeatures (materialProgram.get(), distinctGrating != nullptr);
        const auto programCacheKey = fixtureProgramCacheKey (vertexSource, fragmentFeatures);
        // Solid-body and harmonic-geometry frames can release every scene lease
        // before the next preparation. Retain only the immutable linked program
        // for the exact context and vertex source; the fragment source is fixed.
        auto& contextPrograms = contextCache.programs;
        const auto cached = contextPrograms.find(programCacheKey);
        if (use == NativeFixturePrepareUse::Preview && contextCache.parallelCompile < 0)
        {
            contextCache.parallelCompile = arbitgl::hasParallelShaderCompile (gl) ? 1 : 0;
            if (contextCache.parallelCompile == 1)
                gl.MaxShaderCompilerThreads (0xFFFFFFFFu);
        }
        // Preview submits the scene program and returns pending. A later Preview polls
        // it, and Blocking finishes it. Every return in this block comes before the
        // first upload, so a pending result leaves the scene without GL objects.
        auto& inFlight = contextCache.inFlight;
        // Diagnostics only. One stderr line on each pending state change, and otherwise
        // at most one per second for this context. Nothing is logged when nothing is pending.
        const auto logFixturePending = [&] (const std::string& key, unsigned features,
            std::chrono::steady_clock::time_point submitted, bool forced)
        {
            const auto now = std::chrono::steady_clock::now();
            if (! forced && now - contextCache.lastPendingLog < std::chrono::seconds (1)) return;
            contextCache.lastPendingLog = now;
            const auto ageMs = std::chrono::duration_cast<std::chrono::milliseconds> (now - submitted).count();
            std::fprintf (stderr,
                "[gl] fixture program pending: polls=%llu inFlight=%zu submits=%llu evictions=%llu ageMs=%lld features=0x%x key=%08x\n",
                static_cast<unsigned long long> (contextCache.pendingPolls), inFlight.size(),
                static_cast<unsigned long long> (contextCache.submitCount),
                static_cast<unsigned long long> (contextCache.evictionCount),
                static_cast<long long> (ageMs),
                features,
                static_cast<unsigned> (std::hash<std::string> {}(key)));
        };
        const auto flying = inFlight.find (programCacheKey);
        const bool asyncSubmit = use == NativeFixturePrepareUse::Preview
            && contextCache.parallelCompile == 1;
        if (! distinctGrating && ! vertexSource.empty() && cached == contextPrograms.end()
            && (flying != inFlight.end() || asyncSubmit))
        {
            FixtureProgramBinaryDriver binaryDriver (gl);
            arbitglprogramcache::ProgramBinaryCache binaryCache (binaryDriver,
                arbitglprogramcache::cacheDirectory());
            const auto fragmentSource = fixtureFragmentShaderSource (fragmentFeatures);
            // Records when a key left inFlight. Entries older than the window are dropped
            // here, so the map holds only recent keys.
            const auto recordRetired = [&] (const std::string& key)
            {
                const auto now = std::chrono::steady_clock::now();
                for (auto it = contextCache.retiredAt.begin(); it != contextCache.retiredAt.end();)
                {
                    if (now - it->second > kFixtureResubmitWindow) it = contextCache.retiredAt.erase (it);
                    else ++it;
                }
                contextCache.retiredAt[key] = now;
            };
            const auto recentlyRetired = [&] (const std::string& key)
            {
                const auto found = contextCache.retiredAt.find (key);
                return found != contextCache.retiredAt.end()
                    && std::chrono::steady_clock::now() - found->second <= kFixtureResubmitWindow;
            };
            // Finishes a submitted program, stores its binary, and installs it as the scene
            // program. On a returned error the entry's GL objects are already deleted.
            const auto installFinished = [&] (OpenGlFixtureInFlightProgram& entry) -> std::string
            {
                std::string finishError;
                {
                    loopstep::Scope finishScope ("gl:fixture-scene-program-compile");
                    finishError = finishFixtureProgram (entry);
                }
                deleteInFlightShaders (entry);
                recordRetired (programCacheKey);
                if (! finishError.empty())
                {
                    deleteInFlightObjects (entry);
                    return finishError;
                }
                binaryCache.store (vertexSource, fragmentSource, entry.program);
                auto shared = std::make_shared<OpenGlFixtureSceneResources::SharedProgram>();
                shared->context = glfwGetCurrentContext();
                shared->gl = gl;
                shared->id = entry.program;
                resources->sharedProgram = shared;
                result.stats.shaderProgramBuildCount = 1;
                return {};
            };
            if (flying != inFlight.end())
            {
                // A Preview prepare only polls. It defers until the driver reports the link
                // complete, however old the submission is, so the render loop never waits on
                // the driver. Only Blocking finishes a link that is still in progress.
                if (use == NativeFixturePrepareUse::Preview)
                {
                    int complete = GL_FALSE;
                    {
                        loopstep::Scope pollScope ("gl:fixture-scene-program-poll");
                        gl.GetProgramiv (flying->second.program, GL_COMPLETION_STATUS_KHR, &complete);
                    }
                    if (complete != GL_TRUE)
                    {
                        ++contextCache.pendingPolls;
                        logFixturePending (programCacheKey, flying->second.fragmentFeatures,
                            flying->second.submitted, ! contextCache.pendingActive);
                        contextCache.pendingActive = true;
                        previous.restore();
                        result.pending = true;
                        return result;
                    }
                }
                auto entry = std::move (flying->second);
                inFlight.erase (flying);
                if (contextCache.pendingActive)
                {
                    contextCache.pendingActive = false;
                    logFixturePending (programCacheKey, entry.fragmentFeatures, entry.submitted, true);
                }
                if (auto error = installFinished (entry); ! error.empty()) return fail (error);
                const auto readyMs = std::chrono::duration_cast<std::chrono::milliseconds> (
                    std::chrono::steady_clock::now() - entry.submitted).count();
                std::fprintf (stderr, "[gl] fixture program ready after %lld ms (async) mode=async-finish vertexBytes=%llu fragmentBytes=%llu features=0x%x key=%08x\n",
                    static_cast<long long> (readyMs),
                    static_cast<unsigned long long> (vertexSource.size()),
                    static_cast<unsigned long long> (fragmentSource.size()),
                    entry.fragmentFeatures,
                    static_cast<unsigned> (std::hash<std::string> {}(programCacheKey)));
            }
            else
            {
                auto shared = std::make_shared<OpenGlFixtureSceneResources::SharedProgram>();
                shared->context = glfwGetCurrentContext();
                shared->gl = gl;
                {
                    loopstep::Scope binaryLoadScope ("gl:fixture-scene-program-binary-load");
                    shared->id = binaryCache.load (vertexSource, fragmentSource);
                }
                if (shared->id != 0)
                {
                    resources->sharedProgram = shared;
                }
                else
                {
                    OpenGlFixtureInFlightProgram entry;
                    std::string submitError;
                    {
                        loopstep::Scope submitScope ("gl:fixture-scene-program-submit");
                        submitError = submitFixtureProgram (gl, binaryCache, vertexSource,
                            fragmentSource, fragmentFeatures, entry);
                    }
                    if (! submitError.empty()) return fail (submitError);
                    ++contextCache.submitCount;
                    if (recentlyRetired (programCacheKey))
                    {
                        // This key was evicted or finished moments ago. Compile it now rather
                        // than queueing it async again, so the scene is never left pending.
                        std::fprintf (stderr, "[gl] fixture program resubmitted within 2 s of retirement; compiling synchronously\n");
                        if (auto error = installFinished (entry); ! error.empty()) return fail (error);
                    }
                    else
                    {
                        while (inFlight.size() >= kMaximumInFlightFixturePrograms)
                        {
                            const auto oldest = std::min_element (inFlight.begin(), inFlight.end(),
                                [] (const auto& a, const auto& b)
                                { return a.second.submitted < b.second.submitted; });
                            const auto evictedKey = oldest->first;
                            const auto evictedAt = oldest->second.submitted;
                            const auto evictedFeatures = oldest->second.fragmentFeatures;
                            deleteInFlightObjects (oldest->second);
                            inFlight.erase (oldest);
                            recordRetired (evictedKey);
                            ++contextCache.evictionCount;
                            logFixturePending (evictedKey, evictedFeatures, evictedAt, true);
                        }
                        const auto submittedAt = entry.submitted;
                        const auto submittedFeatures = entry.fragmentFeatures;
                        inFlight.emplace (programCacheKey, std::move (entry));
                        contextCache.pendingActive = true;
                        logFixturePending (programCacheKey, submittedFeatures, submittedAt, true);
                        previous.restore();
                        result.pending = true;
                        return result;
                    }
                }
            }
        }
        const bool selectedCompute=roughSelectedComputeCapability(resources->gl);
        if (!smoothglass::prepare(scene, materialProgram, resources->instanceAppearances,
                resources->smoothGlass, result.error,
                {NativeFixtureMaterialBackend::OpenGl, true, true, true, smoothglass::kMultipleScatterVersion,selectedCompute},distinctGrating)) return fail(result.error);
        if(distinctGrating)resources->opaqueVisibility=resources->smoothGlass.query;
        else {
        environmentimage::Prepared environment;
        if (!environmentimage::prepare(*scene, materialProgram.get(), environment, result.error,
                resources->smoothGlass.enabled ? environmentimage::SecondaryRayConsumer::SmoothTransport
                                               : environmentimage::SecondaryRayConsumer::None,
                resources->smoothGlass.enabled ? &resources->smoothGlass : nullptr)) return fail(result.error);
        if (environment.resident)
        {
            if (std::any_of(resources->instanceAppearances.begin(),resources->instanceAppearances.end(),
                [](const auto& a){return a.color[3]<1;})) return fail("HDR environment requires opaque instance appearance");
            resources->environmentResident=environment.resident;
            resources->environmentBudget=environment.gpuBudget;
            int imageLimit=0; glGetIntegerv(GL_MAX_TEXTURE_SIZE,&imageLimit);
            if (imageLimit<static_cast<int>(environment.resident->image.metadata.width)
                || imageLimit<static_cast<int>(environment.resident->image.metadata.height))
                return fail("HDR environment exceeds native texture dimensions");
            gl.ActiveTexture(GL_TEXTURE0); gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
            glPixelStorei(GL_UNPACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ROW_LENGTH,0);
            glPixelStorei(GL_UNPACK_SKIP_PIXELS,0); glPixelStorei(GL_UNPACK_SKIP_ROWS,0);
            // Bytes are charged by environmentBudget, the lease acquired above.
            glGenTextures(1,&resources->environmentTexture);
            glBindTexture(GL_TEXTURE_2D,resources->environmentTexture);
            const auto& metadata=environment.resident->image.metadata;
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,metadata.width,metadata.height,0,GL_RGBA,GL_FLOAT,environment.texels.data());
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
            if (!resources->environmentTexture || glGetError()!=GL_NO_ERROR) return fail("HDR environment float upload failed");
            std::vector<std::array<float,4>>().swap(environment.texels);
        }
        if (!opaquevisibility::prepareScene(*scene, materialProgram.get(),
                resources->instanceAppearances, resources->opaqueVisibility, result.error,
                opaquevisibility::AdmissionMode::opaqueOnly, false, supportsAuthoredSpectralReceiverTransport()))
            return fail(result.error);
        if (materialProgram && materialProgram->diffractionLightingAdmission
            && materialProgram->diffractionLightingAdmission->description().version == 3
            && !diffractiontransportnative::prepareScene(*scene, *materialProgram,
                materialProgram->sceneRevision, resources->diffractionTransport, result.error)) return fail(result.error);

        }

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
        resources->vertexBufferCharge = residency::ResidencyCharge (
            residency::BackingClass::fixtureScene, residency::nominalBufferBytes (vertexBytes));
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
        resources->indexBufferCharge = residency::ResidencyCharge (
            residency::BackingClass::fixtureScene, residency::nominalBufferBytes (indexBytes));
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
        if (resources->instancedSharedGeometry || scene->instancedObject.isValid())
        {
            const auto instanceRecordCount = resources->instancedSharedGeometry
                ? scene->objectCount : scene->objectInstanceCount;
            gl.GenBuffers(1, &resources->instanceBuffer);
            gl.BindBuffer(GL_ARRAY_BUFFER, resources->instanceBuffer);
            gl.BufferData(GL_ARRAY_BUFFER,
                static_cast<std::ptrdiff_t>(instanceRecordCount
                    * sizeof(OpenGlInstanceGpuRecord)), nullptr, GL_DYNAMIC_DRAW);
            resources->instanceBufferCharge = residency::ResidencyCharge(
                residency::BackingClass::fixtureScene,
                residency::nominalBufferBytes(instanceRecordCount * sizeof(OpenGlInstanceGpuRecord)));
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
        try
        {
            resources->textures.resize (1 + scene->textureCount);
            resources->textureCharges.reserve (resources->textures.size());
        }
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
            // Slots upload in order, so charges stay index-aligned with textures.
            resources->textureCharges.emplace_back (residency::BackingClass::fixtureScene,
                residency::nominalTextureBytes (textureWidth, textureHeight, 1, levelCount,
                                                sizeof (SceneTexelRgba8), 1));
            if (glGetError() != GL_NO_ERROR)
                return fail ("OpenGL fixture texture upload failed");
            if (levelCount > 1)
                generateMipmap (GL_TEXTURE_2D);
            if (glGetError() != GL_NO_ERROR)
                return fail ("OpenGL fixture mipmap allocation failed");
        }

        if(distinctGrating) {
            if(!resources->vertexArray||!resources->vertexBuffer||!resources->indexBuffer||glGetError()!=GL_NO_ERROR)
                return fail("OpenGL private mixed static allocation failed");
            resources->retainedTextureBytes=textureBytes;
            previous.restore();result.prepared=true;result.stats.staticUploadCount=1;
            result.stats.vertexBytes=vertexBytes;result.stats.indexBytes=indexBytes;result.stats.textureBytes=textureBytes;
            result.stats.materialBytes=sizeof(NativeFixtureSurfaceMaterialProgram)*(1+materialProgram->objectPrograms.size());
            result.stats.materialProgramUploadCount=1;privatePrepared=true;result.resources=std::move(resources);return result;
        }

        if (vertexSource.empty()) return fail (vertexError);
        if (cached != contextPrograms.end())
        {
            resources->sharedProgram = cached->second.program;
            cached->second.lastUse = ++programUseSerial_;
        }
        if (!resources->sharedProgram)
        {
            const auto fragmentSource = fixtureFragmentShaderSource (fragmentFeatures);
            // A program linked by an earlier session on this driver is loaded from
            // the disk cache. A hit skips both compiles and the link, so the build
            // count for this preparation stays at zero.
            FixtureProgramBinaryDriver binaryDriver (gl);
            arbitglprogramcache::ProgramBinaryCache binaryCache (binaryDriver, arbitglprogramcache::cacheDirectory());
            reportFixtureProgramCacheOnce (binaryCache, binaryDriver, gl);
            auto shared=std::make_shared<OpenGlFixtureSceneResources::SharedProgram>();
            shared->context=glfwGetCurrentContext(); shared->gl=gl;
            {
                loopstep::Scope binaryLoadScope ("gl:fixture-scene-program-binary-load");
                shared->id=binaryCache.load(vertexSource,fragmentSource);
            }
            if(shared->id==0)
            {
                const auto syncCompileStarted=std::chrono::steady_clock::now();
                // Compiles and links under the fixtureMutex held by this function.
                loopstep::Scope sceneCompileScope ("gl:fixture-scene-program-compile");
                vertexShader=compileFixtureShader(gl,GL_VERTEX_SHADER,vertexSource.c_str(),result.error);
                if(vertexShader==0) return fail(result.error);
                fragmentShader=compileFixtureShader(gl,GL_FRAGMENT_SHADER,fragmentSource.c_str(),result.error);
                if(fragmentShader==0) return fail(result.error);
                shared->id=gl.CreateProgram();
                binaryCache.prepareLink(shared->id);
                gl.AttachShader(shared->id,vertexShader); gl.AttachShader(shared->id,fragmentShader); gl.LinkProgram(shared->id);
                int linked=0; gl.GetProgramiv(shared->id,GL_LINK_STATUS,&linked);
                if(linked!=GL_TRUE) {
                    char log[1024]={}; int length=0; gl.GetProgramInfoLog(shared->id,(int)sizeof(log),&length,log);
                    return fail(std::string("OpenGL fixture program link failed: ")+std::string(log,(size_t)std::max(length,0)));
                }
                binaryCache.store(vertexSource,fragmentSource,shared->id);
                result.stats.shaderProgramBuildCount = 1;
                const auto syncCompileMs=std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now()-syncCompileStarted).count();
                std::fprintf(stderr,"[gl] fixture program compile: mode=sync ms=%lld vertexBytes=%llu fragmentBytes=%llu features=0x%x key=%08x\n",
                    static_cast<long long>(syncCompileMs),static_cast<unsigned long long>(vertexSource.size()),
                    static_cast<unsigned long long>(fragmentSource.size()),static_cast<unsigned>(fragmentFeatures),
                    static_cast<unsigned>(std::hash<std::string>{}(programCacheKey)));
            }
            resources->sharedProgram=shared;
        }
        resources->program=resources->sharedProgram->id;
        cleanupShaders();
        if (materialProgram != nullptr
            && materialProgram->kind == NativeFixtureMaterialKind::DiffractionReflective
            && resources->sharedProgram->diffractionResolveId == 0)
        {
            auto& shared = *resources->sharedProgram;
            shared.diffractionResolveId = compileDiffractionResolveProgram(gl, result.error);
            if (shared.diffractionResolveId == 0) return fail(result.error);
            gl.GenVertexArrays(1, &shared.diffractionResolveVao);
            if (shared.diffractionResolveVao == 0)
            {
                gl.DeleteProgram(shared.diffractionResolveId);
                shared.diffractionResolveId = 0;
                return fail("OpenGL diffraction resolve vertex array allocation failed");
            }
        }

        const auto location = [&] (const char* name)
        {
            return gl.GetUniformLocation (resources->program, name);
        };
        auto& uniforms = resources->uniforms;
        uniforms.opaqueTexture = location("uOpaqueVisibility");
        uniforms.opaqueControl = location("uOpaqueControl");
        uniforms.glassTexture = location("uGlassVisibility");
        uniforms.glassControl = location("uGlassControl");
        uniforms.glassTransport = location("uGlassTransport");
        uniforms.roughCameraControl = location("uRoughCameraControl");
        uniforms.glassReceivers = location("uGlassReceivers[0]");
        uniforms.glassEmission = location("uGlassEmission[0]");
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
        uniforms.previousVertexSpectrum = location ("uPreviousVertexSpectrum[0]");
        uniforms.vertexDisplacementLimits = location ("uVertexDisplacementLimits[0]");
        uniforms.emissive = location ("uEmissive");
        uniforms.ambient = location ("uAmbient");
        uniforms.sky = location("uSky[0]");
        uniforms.skyBackground = location("uSkyBackground");
        uniforms.environmentImage=location("uEnvironmentImage");
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
        uniforms.usePreviousDeformation = location("uUsePreviousDeformation");
        uniforms.noteMotion = location("uNoteMotion");
        uniforms.previousNoteTransforms = location("uPreviousNoteTransforms");
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
        uniforms.diffractionTriangleCount = location("uDiffractionTriangleCount");
        uniforms.diffractionPathIndex = location ("uDiffractionPathIndex");
        uniforms.diffractionQueryTexture = location("uDiffractionQueryVisibility");
        uniforms.diffractionQueryControl = location("uDiffractionQueryControl");
        uniforms.diffractionInstanceCount = location("uDiffractionInstanceCount");
        uniforms.diffractionSamples = location("uDiffractionSamples");
        uniforms.diffractionSampleControl = location("uDiffractionSampleControl");
        uniforms.diffractionSourceDirectionTrace = location("uDiffractionSourceDirectionTrace");
        uniforms.diffractionSourceRadiance = location("uDiffractionSourceRadiance[0]");
        uniforms.diffractionRayMinimum = location("uDiffractionRayMinimum");
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
            || resources->program == 0 || ! uniforms.complete (fragmentFeatures))
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
            contextPrograms.emplace(programCacheKey, CachedProgram{++programUseSerial_, resources->sharedProgram});
        }

        previous.restore();
        result.prepared = true;
        resources->retainedTextureBytes = textureBytes;
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
        if (runtimeInputs.noteInstanceMapping && runtimeInputs.noteInstanceMapping->schemaVersion
                == visualnoteinstancing::kVelocityScaleMappingSchemaVersion)
        {
            result.error = "Native per-note velocity scale transport is unavailable";
            return result;
        }
        std::lock_guard<std::mutex> lock (fixtureMutex());
        const auto available = info();
        if (! available.available)
        {
            result.error = available.error;
            return result;
        }
        const auto resources = std::dynamic_pointer_cast<const OpenGlFixtureSceneResources> (
            nativeResources);
        if (scene == nullptr || resources == nullptr || resources->snapshot != scene || resources->gratingLeaf
            || resources->ownerContext != glfwGetCurrentContext()||!resources->queryContext||!resources->queryContext->active)
        {
            result.error = "OpenGL fixture render requires exact prepared scene resources";
            return result;
        }
        resources->queryContext->drainDeferredUnlocked();
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
                materialProgram, result.error,
                diffractiontransportnative::consumerCanRender(kOpenGlDiffractionConsumer, NativeFixtureMaterialBackend::OpenGl))) return result;
        if (materialProgram != nullptr
            && (!materialProgram->diffractionLightingAdmission
                || materialProgram->diffractionLightingAdmission->description().version != 3)
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
                || actualHeight != static_cast<int>(descriptor.height)
                || format != (materialFrameIsLinear(descriptor) ? GL_RGBA16F : GL_RGBA8))
            { result.error = "OpenGL material Frame texture no longer matches its descriptor"; return result; }
        }
        const auto rootFrame = materialFrameForProgram(materialProgram.get(),runtimeInputs);
        materialFrameTexture = rootFrame ? static_cast<unsigned>(rootFrame->colorTextureViewHandle()) : 0;
        if (runtimeInputs.objectWorlds && !validNativeImportedObjectWorlds(*runtimeInputs.objectWorlds))
        {
            result.error = "OpenGL static local object worlds are not finite affine worlds with distinct object ids";
            return result;
        }
        if (!validFixtureRuntimeInputs(runtimeInputs))
        {
            result.error = !render3dimage::supported(runtimeInputs.imageOutput)
                ? "Render 3D requested image output is unavailable on this backend"
                : "OpenGL fixture scene modulation is non-finite or out of bounds";
            return result;
        }
        if (runtimeInputs.objectWorlds && !nativeImportedObjectWorldsCover(*scene, *runtimeInputs.objectWorlds))
        {
            result.error = "OpenGL static local object worlds do not match the drawn scene objects";
            return result;
        }
        const bool diffractionFrame = materialProgram != nullptr
            && materialProgram->kind == NativeFixtureMaterialKind::DiffractionReflective;
        if (runtimeInputs.objectWorlds && diffractionFrame)
        {
            result.error = "OpenGL static local object worlds do not yet admit diffraction gratings";
            return result;
        }
        if (runtimeInputs.objectWorlds
            && (resources->instancedSharedGeometry || scene->instancedObject.isValid()
                || runtimeInputs.noteInstanceMapping))
        {
            result.error = "OpenGL static local object worlds do not yet admit instanced draws";
            return result;
        }
        if (runtimeInputs.imageOutput == renderpassoutput::Output::ObjectId
            && (resources->instancedSharedGeometry || scene->instancedObject.isValid()))
        {
            result.error = "Render 3D Object ID inspection requires individual scene draws";
            return result;
        }
        if (!nativeObjectInstancesSupported(*scene, materialProgram.get(), false, result.error)
            || !nativeObjectInstanceRuntimeSupported(*scene, runtimeInputs, result.error)
            || !nativeObjectPriorTransformsSupported(*scene, runtimeInputs,
                resources->instancedSharedGeometry, result.error))
            return result;
        if (materialProgram != nullptr
            && nativeSurfaceUsesTime(*materialProgram)
            && (! std::isfinite (runtimeInputs.timeSeconds)
                || std::abs (runtimeInputs.timeSeconds)
                    > surfacematerial::kMaximumEvaluationMagnitude))
        {
            result.error = "OpenGL fixture material time input is non-finite or out of bounds";
            return result;
        }

        // Retained prior positions of a topology-stable deformed mesh feed the
        // same attribute that skinned meshes use for their prior pose.
        const bool motionPass = sceneUsesMotionPass(runtimeInputs);
        if (runtimeInputs.objectWorlds && motionPass
            && !nativeObjectWorldMotionSupported(*scene, runtimeInputs, result.error))
            return result;
        std::shared_ptr<const std::vector<SceneVec3>> priorScenePositions;
        if (motionPass && runtimeInputs.previousMotion)
            priorScenePositions = runtimeInputs.previousMotion->scenePositions;
        if (priorScenePositions && (!sceneMotionPositionsMatch(*scene, priorScenePositions->size())
            || resources->instancedSharedGeometry || resources->usePreviousDeformation
            || runtimeInputs.previousMotion->deformation))
        {
            result.error = "Motion output prior Geometry3D positions do not match the scene vertices";
            return result;
        }
        if (materialProgram != nullptr && materialProgram->vertexProgram)
        {
            // The program runs a second time with the prior time, spectrum and
            // prior deformed position from attribute 12. Other imported inputs
            // come from the current vertex, so refuse programs that need them.
            const auto prior = !motionPass ? videowire::VertexModifierPriorGeometry::none
                : priorScenePositions ? videowire::VertexModifierPriorGeometry::retainedPositions
                : resources->usePreviousDeformation ? videowire::VertexModifierPriorGeometry::skinnedPose
                : videowire::VertexModifierPriorGeometry::none;
            if (const auto refusal = videowire::vertexModifierPriorGeometryRefusal(
                    materialProgram->vertexProgram->vertexInstructions(), prior))
            {
                result.error = refusal;
                return result;
            }
            if (!motionPass) runtimeInputs.previousMotion.reset();
        }

        const auto& gl = resources->gl;
        const auto noteInstances = prepareNativeNoteInstances(runtimeInputs);
        const bool roughGlassFrame = resources->smoothGlass.enabled && resources->smoothGlass.roughness > 0;
        std::uint64_t roughReceiptBytes = 0;
        std::shared_ptr<const roughglasstiles::Schedule> roughSchedule;
        roughglasstiles::FrameKey roughKey;
        if (roughGlassFrame)
        {
            const auto* selected = visual3d_detail::findById(scene->cameras, scene->cameraCount, scene->activeCamera);
            if (!runtimeInputs.cameraOverride && !runtimeInputs.animatedCamera && !selected)
            { result.error = "OpenGL rough glass cannot resolve its immutable camera"; return result; }
            roughKey.scene = scene; roughKey.program = materialProgram;
            try { roughKey.runtime = std::make_shared<const NativeFixtureSceneRuntimeInputs>(runtimeInputs); }
            catch (const std::bad_alloc&) { result.error = "OpenGL rough immutable runtime allocation failed"; return result; }
            roughKey.owner = resources;
            roughKey.camera = runtimeCameraRecord(runtimeInputs.cameraOverride ? *runtimeInputs.cameraOverride
                : runtimeInputs.animatedCamera ? *runtimeInputs.animatedCamera : *selected,
                runtimeInputs.cameraTranslationOffset, runtimeInputs.cameraRotationDegrees,
                runtimeInputs.cameraVerticalFovDegrees);
            roughKey.sceneRevision = materialProgram->sceneRevision;
            roughKey.rendererGeneration = resources->rendererGeneration;
            roughKey.width = width; roughKey.height = height;
            try {
                if (!roughglasstiles::Schedule::admit(resources->smoothGlass, roughKey,
                        noteInstances.admitted || resources->instancedSharedGeometry, roughSchedule, result.error)) return result;
            } catch (const std::bad_alloc&) { result.error = "OpenGL rough tile schedule allocation failed"; return result; }
            roughReceiptBytes = roughSchedule->scratchBytes();
            std::array<int,2> maximumViewport{};
            glGetIntegerv(GL_MAX_VIEWPORT_DIMS, maximumViewport.data());
            if (int(width)>maximumViewport[0] || int(height)>maximumViewport[1])
            { result.error = "OpenGL rough tiled frame exceeds the exact native viewport extent"; return result; }
            const auto colorFormat = runtimeInputs.linearColor
                ? NativeTexturePixelFormat::Rgba16Float : NativeTexturePixelFormat::Rgba8Unorm;
            std::uint64_t stride = rawPassChannels(colorFormat) * rawPassScalarBytes(colorFormat)
                + sizeof(float) + sizeof(std::uint32_t);
            for (const auto format : OpenGlFixtureSceneFrame::rawFormats)
                stride += rawPassChannels(format) * rawPassScalarBytes(format);
            std::uint64_t resident = 0;
            const auto charge = [&](std::uint64_t count, std::uint64_t bytes) {
                std::uint64_t amount = 0;
                return renderpassoutput::detail::checkedMultiply(count, bytes, amount)
                    && renderpassoutput::detail::checkedAdd(resident, amount, resident);
            };
            // Query frame tables are each bounded by the existing 1024x2048
            // authority. Charge both GPU tables and both host staging copies,
            // prepared tables, static owners, runtime leases and tile depth.
            bool memory = charge(4 * std::uint64_t(opaquevisibility::kTextureWidth)
                    * opaquevisibility::kTextureHeight, sizeof(opaquevisibility::Texel))
                && charge(resources->opaqueVisibility.texels.capacity()
                    + resources->smoothGlass.query.texels.capacity(), sizeof(opaquevisibility::Texel))
                && charge(resources->opaqueVisibility.meshes.capacity()
                    + resources->smoothGlass.query.meshes.capacity(), sizeof(opaquevisibility::Mesh))
                && charge(1, resources->retainedTextureBytes)
                && charge(3, resources->environmentBudget ? resources->environmentBudget->ownedBytes : 0)
                && charge(2, sizeof(*scene)) && charge(scene->textureTexels.capacity(), sizeof(SceneTexelRgba8)*2)
                && charge(scene->vertexCount, sizeof(SceneVertex)*(resources->previousVertexBuffer ? 3u : 2u))
                && charge(noteInstances.admitted ? visualnoteinstancing::kMaximumInstances : 0, sizeof(float)*8u)
                && charge(scene->indexCount, sizeof(std::uint32_t))
                && charge(1, sizeof(*resources) + sizeof(NativeFixtureSceneRuntimeInputs) + sizeof(OpenGlFixtureSceneFrame))
                && charge(2*runtimeInputs.animatedLights.capacity(), sizeof(SceneLightRecord))
                && charge(resources->textures.capacity(), sizeof(unsigned))
                && charge(resources->instanceAppearances.capacity(), sizeof(videowire::geometry::InstanceAppearance))
                && charge(scene->objectCount, sizeof(opaquevisibility::Matrix))
                && charge(1 + materialProgram->objectPrograms.size(), sizeof(NativeFixtureSurfaceMaterialProgram))
                && charge(roughSchedule->tiles().size()>1 ? roughSchedule->maximumTilePixels() : 0,
                    sizeof(std::uint32_t))
                && charge(smoothglass::multipleScatterCamera(resources->smoothGlass)?roughSchedule->maximumTilePixels():0,
                    RoughSelectedCamera::residentBytesPerPixel);
            const auto chargeProgram = [&](const NativeFixtureSurfaceMaterialProgram& program) {
                return charge(program.bindingDigest.capacity()+program.programIdentity.capacity()
                    +program.diffractionProductDigest.capacity(), 3)
                    && charge(program.objectPrograms.capacity(), sizeof(std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>))
                    && (!program.importedBaseColorTexture
                        || charge(program.importedBaseColorTexture->texels.capacity(), sizeof(SceneTexelRgba8)*3));
            };
            memory = memory && chargeProgram(*materialProgram);
            for (const auto& program : materialProgram->objectPrograms)
                memory = memory && program && chargeProgram(*program);
            for (const auto& input : inputFrames)
                memory = memory && charge(std::uint64_t(input->width())*input->height(), 16);
            if(smoothglass::multipleScatterCamera(resources->smoothGlass))
                memory=memory && charge(1,sizeof(RoughSelectedCamera)+sizeof(OpenGlQueryContext)+512)
                    && charge(2,kQueryProgramSourceReservation+1024)
                    && charge(1,resources->queryContext->deferredBytes.load(std::memory_order_relaxed))
                    && charge(1,resources->causticQueryProgram?resources->causticQueryProgram->retainedBytes:0);
            FrameMemoryAdmission admission;
            if (!memory || !roughglasstiles::admitMemory(*roughSchedule, stride, resident, admission, result.error))
            { if (result.error.empty()) result.error = "OpenGL rough glass full resource accounting overflow"; return result; }
        }
        else if (!smoothglass::admitFrame(resources->smoothGlass, width, height,
                noteInstances.admitted || resources->instancedSharedGeometry, result.error,
                &roughReceiptBytes)) return result;
        std::vector<opaquevisibility::Matrix> visibilityMatrices;
        for (std::size_t o = 0; o < scene->objectCount; ++o)
        {
            SceneTransform3D transform{};
            if (runtimeTargetsObject(runtimeInputs, scene->objects[o]))
            {
                transform.translation = {runtimeInputs.objectTranslationOffset[0],
                                         runtimeInputs.objectTranslationOffset[1],
                                         runtimeInputs.objectTranslationOffset[2]};
                transform = applyRuntimeObjectTransform(transform, runtimeInputs.objectRotationDegrees,
                                                        runtimeInputs.objectScale);
            }
            visibilityMatrices.push_back(multiplyFixtureMatrices(fixtureTransformMatrix(transform),
                nativeObjectWorldMatrix(runtimeInputs.objectWorlds.get(), scene->objects[o],
                                        fixtureWorldMatrix(*scene, scene->objects[o]))));
        }
        const bool authoredDiffraction = diffractionFrame
            && materialProgram->diffractionLightingAdmission->description().version == 3;
        diffractiontransportnative::Frame diffractionTransportFrame;
        if (authoredDiffraction)
        {
            diffractiontransportnative::Prepared evaluatedTransport;
            const auto* prepared = &resources->diffractionTransport;
            // Preparation retained an immutable snapshot of this exact resource
            // owner. An evaluated runtime program needs its own admitted table.
            if (materialProgram != resources->materialProgram) {
                if (!diffractiontransportnative::prepareScene(*scene, *materialProgram,
                        materialProgram->sceneRevision, evaluatedTransport, result.error)) return result;
                prepared = &evaluatedTransport;
            }
            if (!diffractiontransportnative::buildFrame(*prepared,
                    opaquevisibility::noteInstances(visibilityMatrices, noteInstances), width, height,
                    diffractionTransportFrame, result.error)) return result;
            if (maximumTextureSize < static_cast<int>(diffractiontransportnative::kTextureWidth)
                || maximumTextureSize < static_cast<int>(diffractionTransportFrame.height)
                || maximumTextureSize < static_cast<int>(opaquevisibility::kTextureWidth)
                || maximumTextureSize < static_cast<int>(diffractionTransportFrame.query.height))
            { result.error = "OpenGL diffraction table/query exceeds native texture dimensions"; return result; }
        }
        if (noteInstances.admitted)
        {
            constexpr auto noteBytes = visualnoteinstancing::kMaximumInstances * 4u * sizeof(float);
            const std::uint64_t motionResident = std::uint64_t(scene->vertexCount)*sizeof(SceneVertex)*3
                + resources->retainedTextureBytes + std::uint64_t(width)*height*56 + noteBytes*2;
            if (motionResident > renderpassoutput::kMaximumTotalBytes)
            { result.error = "OpenGL preceding note data exceeds resource capacity"; return result; }
        }
        std::unique_ptr<roughglasstiles::Completion> roughCompletion;
        if (roughGlassFrame)
        {
            try { roughCompletion = std::make_unique<roughglasstiles::Completion>(roughSchedule); }
            catch (const std::bad_alloc&) { result.error = "OpenGL rough tile completion allocation failed"; return result; }
        }
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
        unsigned resolveFramebuffer = 0;
        unsigned roughReceiptFramebuffer = 0;
        unsigned roughReceiptDepth = 0;
        unsigned depthAttachment = 0;
        auto cleanup = [&]
        {
            if (framebuffer != 0) gl.DeleteFramebuffers (1, &framebuffer);
            if (resolveFramebuffer != 0) gl.DeleteFramebuffers(1, &resolveFramebuffer);
            if (roughReceiptFramebuffer != 0) gl.DeleteFramebuffers(1, &roughReceiptFramebuffer);
            if (depthAttachment != 0) glDeleteTextures (1, &depthAttachment);
            if (roughReceiptDepth != 0) glDeleteTextures(1, &roughReceiptDepth);
            roughReceiptDepth = 0;
            framebuffer = resolveFramebuffer = roughReceiptFramebuffer = 0;
            depthAttachment = 0;
            previous.restore();
        };
        auto fail = [&] (std::string error)
        {
            cleanup();
            frame->releaseTexturesUnlocked();
            // Failure releases under the existing fixture lock. Its destructor
            // must not attempt to acquire that lock a second time.
            frame->ownerContext = nullptr;
            result.error = std::move (error);
            return result;
        };

        gl.ActiveTexture (GL_TEXTURE0);
        gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei (GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei (GL_UNPACK_SKIP_PIXELS, 0);
        glPixelStorei (GL_UNPACK_SKIP_ROWS, 0);
        if (!allocateFixtureCameraAttachments(gl, frame.get(), diffractionFrame,
                framebuffer, depthAttachment, result.error)) return fail(result.error);

        if (roughGlassFrame)
        {
            const auto& scratch = roughSchedule->tiles().front();
            if (roughSchedule->tiles().size()>1)
            {
                glGenTextures(1, &roughReceiptDepth);
                glBindTexture(GL_TEXTURE_2D, roughReceiptDepth);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, int(scratch.width),
                    int(scratch.height), 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
            }
            gl.GenFramebuffers(1, &roughReceiptFramebuffer);
            gl.BindFramebuffer(GL_FRAMEBUFFER, roughReceiptFramebuffer);
            glGenTextures(static_cast<int>(frame->roughReceiptTextures.size()), frame->roughReceiptTextures.data());
            for (std::size_t i = 0; i < frame->roughReceiptTextures.size(); ++i)
            {
                glBindTexture(GL_TEXTURE_2D, frame->roughReceiptTextures[i]);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<int>(scratch.width),
                    static_cast<int>(scratch.height), 0, GL_RGBA, GL_FLOAT, nullptr);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + static_cast<unsigned>(i),
                    GL_TEXTURE_2D, frame->roughReceiptTextures[i], 0);
            }
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, roughReceiptDepth ? roughReceiptDepth : depthAttachment, 0);
            const unsigned receiptBuffers[] { GL_COLOR_ATTACHMENT0, GL_NONE, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
            gl.DrawBuffers(4, receiptBuffers);
            if (!roughReceiptFramebuffer
                || std::any_of(frame->roughReceiptTextures.begin(), frame->roughReceiptTextures.end(),
                    [](unsigned texture) { return texture == 0; })
                || gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE
                || glGetError() != GL_NO_ERROR)
                return fail("OpenGL rough glass requires three renderable RGBA32F receipt attachments");
            result.stats.textureBytes += roughReceiptBytes
                + (roughReceiptDepth ? roughSchedule->maximumTilePixels()*sizeof(std::uint32_t) : 0);
            gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        }

        // A diffraction program names its object. Other renders draw from the
        // first object with geometry, since transform-only ancestors come first.
        const auto* primaryObject = materialProgram && materialProgram->kind == NativeFixtureMaterialKind::DiffractionReflective
            ? visual3d_detail::findById(scene->objects, scene->objectCount, materialProgram->object)
            : arbitgpu::fixtureprimary::firstDrawableObject(*scene);
        const auto* primaryMaterial = arbitgpu::fixtureprimary::materialOf(*scene, primaryObject);
        if (primaryObject == nullptr || primaryMaterial == nullptr)
            return fail("OpenGL fixture render requires a drawable object with a material");
        const auto& object = *primaryObject;
        const auto& material = *primaryMaterial;
        const auto* selectedCamera = visual3d_detail::findById (
            scene->cameras, scene->cameraCount, scene->activeCamera);
        if (!runtimeInputs.cameraOverride && !runtimeInputs.animatedCamera && selectedCamera == nullptr)
            return fail ("OpenGL fixture render cannot resolve the active camera");
        const auto& authoredCamera = runtimeInputs.cameraOverride ? *runtimeInputs.cameraOverride
            : runtimeInputs.animatedCamera ? *runtimeInputs.animatedCamera : *selectedCamera;
        const auto camera = runtimeCameraRecord (authoredCamera, runtimeInputs.cameraTranslationOffset,
            runtimeInputs.cameraRotationDegrees, runtimeInputs.cameraVerticalFovDegrees);
        OpenGlFixtureUniforms values;
        values.objectMatrix = nativeObjectWorldMatrix (runtimeInputs.objectWorlds.get(), object,
            fixtureWorldMatrix (*scene, object));
        storeFixtureQuaternion (values.cameraRotation, camera.transform.rotation);
        storeFixtureVec3 (values.cameraTranslation, camera.transform.translation);
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
        std::string lightTargetError;
        if (!validNativeImportedLightOverrideTargets (*scene, runtimeInputs, lightTargetError))
            return fail (lightTargetError);
        const auto lightCount = runtimeInputs.lightOverride ? std::size_t { 1 }
            : effectiveNativeLightCount (*scene, runtimeInputs);
        // Overrides apply here, once, to the record each light index reads.
        const auto lightAt = [&](std::size_t index) -> SceneLightRecord
            { return effectiveNativeLightRecord (*scene, runtimeInputs, index); };
        if (lightCount == 1)
        {
            const auto& light = lightAt(0);
            if (light.kind == SceneLightKind::Environment)
            {
                // Constant environment fill is added once in the surface shader.
                // Diffraction's released vLitBase route keeps its prior CPU fill.
                if (materialProgram && materialProgram->kind==NativeFixtureMaterialKind::DiffractionReflective)
                {
                    values.ambient[0] += light.color.x * light.intensity;
                    values.ambient[1] += light.color.y * light.intensity;
                    values.ambient[2] += light.color.z * light.intensity;
                }
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
        gl.ClearBufferfv(GL_COLOR, 0, diffractionFrame ? emptyPass
            : runtimeInputs.imageOutput == renderpassoutput::Output::Color
                && !runtimeInputs.passComposite ? background : emptyPass);
        gl.ClearBufferfv(GL_COLOR, 1, farDepth);
        for (int attachment = 2; attachment < 5; ++attachment) gl.ClearBufferfv(GL_COLOR, attachment, emptyPass);
        gl.ClearBufferuiv(GL_COLOR, 5, emptyIdentity);
        gl.ClearBufferuiv(GL_COLOR, 6, emptyIdentity);
        gl.ClearBufferfv(GL_COLOR, 7, emptyPass);
        if (!diffractionFrame && runtimeInputs.passComposite
            && runtimeInputs.passComposite->inspectionImage)
        {
            const auto clear = renderpasscomposite::inspectionClear(*runtimeInputs.passComposite);
            gl.ClearBufferfv(GL_COLOR, 0, clear.data());
        }
        gl.UseProgram (resources->program);
        gl.BindVertexArray (resources->vertexArray);
        if (priorScenePositions)
        {
            // Interleave like the scene vertex buffer so attribute 12 keeps the
            // layout and residency charge of the skinned prior-pose buffer.
            // Vertices past the retained prefix keep their current position.
            std::vector<SceneVertex> prior;
            try { prior.assign(scene->vertices.begin(), scene->vertices.begin() + scene->vertexCount); }
            catch (const std::bad_alloc&) { return fail("OpenGL prior Geometry3D position allocation failed"); }
            for (std::size_t index = 0; index < priorScenePositions->size(); ++index)
                prior[index].position = (*priorScenePositions)[index];
            const auto bytes = prior.size() * sizeof(SceneVertex);
            if (resources->previousVertexBuffer == 0)
            {
                gl.GenBuffers(1, &resources->previousVertexBuffer);
                resources->previousVertexBufferCharge = residency::ResidencyCharge(
                    residency::BackingClass::fixtureScene, residency::nominalBufferBytes(bytes));
            }
            gl.BindBuffer(GL_ARRAY_BUFFER, resources->previousVertexBuffer);
            gl.BufferData(GL_ARRAY_BUFFER, static_cast<std::ptrdiff_t>(bytes), prior.data(), GL_DYNAMIC_DRAW);
            gl.VertexAttribPointer(12, 3, GL_FLOAT, GL_FALSE, sizeof(SceneVertex),
                reinterpret_cast<const void*>(offsetof(SceneVertex, position)));
            gl.EnableVertexAttribArray(12);
            if (!resources->previousVertexBuffer || glGetError() != GL_NO_ERROR)
                return fail("OpenGL prior Geometry3D position buffer allocation failed");
        }
        gl.ActiveTexture (GL_TEXTURE0);
        glBindTexture (GL_TEXTURE_2D, resources->textures[0]);
        gl.BindSampler (0, 0);
        const auto& locations = resources->uniforms;
        opaquevisibility::Frame visibilityFrame;
        bool hasDirectLight = false;
        for (std::size_t i = 0; i < lightCount; ++i)
            hasDirectLight |= lightAt(i).kind != SceneLightKind::Environment && lightAt(i).intensity > 0;
        bool directVisibility = (!diffractionFrame || !materialProgram->objectPrograms.empty()) &&
                                runtimeInputs.imageOutput == renderpassoutput::Output::Color &&
                                !resources->opaqueVisibility.meshes.empty() && hasDirectLight;
        if (directVisibility && opaquevisibility::vertexDeformation(materialProgram.get()))
        {
            result.stats.opaqueVisibilityUnsupportedVertexDeformation = true;
            directVisibility = false;
        }
        if (directVisibility &&
            !opaquevisibility::frame(resources->opaqueVisibility,
                                     opaquevisibility::sceneInstances(*scene, visibilityMatrices, noteInstances),
                                     visibilityFrame, result.error))
            return fail(result.error);
        if (!directVisibility)
            visibilityFrame.texels.resize(opaquevisibility::kTextureWidth);
        result.stats.opaqueVisibilityApplied = directVisibility;
        result.stats.opaqueVisibilityBytes = visibilityFrame.texels.size() * sizeof(opaquevisibility::Texel);
        if (maximumTextureSize < static_cast<int>(opaquevisibility::kTextureWidth) ||
            maximumTextureSize < static_cast<int>(visibilityFrame.height))
            return fail("OpenGL opaque visibility RGBA32F payload exceeds native texture dimensions");
        gl.ActiveTexture(GL_TEXTURE5);
        glGenTextures(1, &frame->opaqueTexture);
        glBindTexture(GL_TEXTURE_2D, frame->opaqueTexture);
        gl.BindSampler(5, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<int>(opaquevisibility::kTextureWidth),
                     static_cast<int>(visibilityFrame.height), 0, GL_RGBA, GL_FLOAT,
                     visibilityFrame.texels.data());
        if (!frame->opaqueTexture || glGetError() != GL_NO_ERROR)
            return fail("OpenGL opaque visibility RGBA32F texture admission failed");
        gl.Uniform1i(locations.opaqueTexture, 5);
        gl.Uniform4fv(locations.opaqueControl, 1, visibilityFrame.control.data());
        opaquevisibility::Frame glassFrame;
        if (resources->smoothGlass.enabled) {
            if (!opaquevisibility::frame(resources->smoothGlass.query,
                    opaquevisibility::noteInstances(visibilityMatrices, noteInstances), glassFrame, result.error)) return fail(result.error);
        } else glassFrame.texels.resize(opaquevisibility::kTextureWidth);
        if (maximumTextureSize < static_cast<int>(glassFrame.height)) return fail("OpenGL glass query exceeds native texture dimensions");
        gl.ActiveTexture(GL_TEXTURE6);
        glGenTextures(1, &frame->glassTexture);
        glBindTexture(GL_TEXTURE_2D, frame->glassTexture);
        gl.BindSampler(6, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<int>(opaquevisibility::kTextureWidth),
                    static_cast<int>(glassFrame.height), 0, GL_RGBA, GL_FLOAT, glassFrame.texels.data());
        if (!frame->glassTexture || glGetError() != GL_NO_ERROR) return fail("OpenGL glass query RGBA32F texture admission failed");
        gl.Uniform1i(locations.glassTexture, 6);
        gl.Uniform1i(gl.GetUniformLocation(resources->program,"uSelectedRoughResult"),10);
        gl.Uniform1f(gl.GetUniformLocation(resources->program,"uSelectedRoughResolve"),0);
        gl.Uniform4fv(locations.glassControl, 1, glassFrame.control.data());
        gl.Uniform4fv(locations.glassReceivers, 64, resources->smoothGlass.receivers[0].data());
        auto glassEmission = resources->smoothGlass.emission;
        for (auto& emission : glassEmission) for (std::size_t channel=0; channel<3; ++channel) emission[channel]*=runtimeInputs.emissionGain;
        gl.Uniform4fv(locations.glassEmission, 64, glassEmission[0].data());
        result.stats.smoothGlassTransportApplied = resources->smoothGlass.enabled;
        result.stats.smoothGlassQueryBytes = resources->smoothGlass.enabled ? glassFrame.texels.size()*sizeof(opaquevisibility::Texel) : 0;
        // Separate bindings from opaque shadow unit 5, glass unit 6 and HDR unit 7.
        const auto uploadDiffraction = [&](unsigned unit, unsigned& texture, std::size_t textureWidth,
            std::size_t textureHeight, const auto& texels) {
            gl.ActiveTexture(GL_TEXTURE0 + unit); gl.BindSampler(unit, 0);
            glGenTextures(1, &texture); glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<int>(textureWidth),
                static_cast<int>(textureHeight), 0, GL_RGBA, GL_FLOAT, texels.data());
            return texture != 0 && glGetError() == GL_NO_ERROR;
        };
        if (authoredDiffraction) {
            if (!uploadDiffraction(8, frame->diffractionQueryTexture, opaquevisibility::kTextureWidth,
                    diffractionTransportFrame.query.height, diffractionTransportFrame.query.texels)
                || !uploadDiffraction(9, frame->diffractionSamplesTexture, diffractiontransportnative::kTextureWidth,
                    diffractionTransportFrame.height, diffractionTransportFrame.texels))
                return fail("OpenGL diffraction RGBA32F table/query upload failed");
        }
        gl.Uniform1i(locations.diffractionQueryTexture, 8);
        gl.Uniform1i(locations.diffractionSamples, 9);
        gl.Uniform1i(locations.diffractionInstanceCount, static_cast<int>(noteInstances.count));
        gl.Uniform4fv(locations.diffractionQueryControl, 1, diffractionTransportFrame.query.control.data());
        gl.Uniform4fv(locations.diffractionSampleControl, 1, diffractionTransportFrame.sampleControl.data());
        gl.Uniform4fv(locations.diffractionSourceDirectionTrace, 1, diffractionTransportFrame.sourceDirectionTrace.data());
        gl.Uniform4fv(locations.diffractionSourceRadiance, 2, diffractionTransportFrame.sourceRadiance[0].data());
        gl.Uniform1f(locations.diffractionRayMinimum, diffractionTransportFrame.rayMinimumDistance);
        gl.ActiveTexture(GL_TEXTURE0);

        // Motion rebuilds the prior camera from the authored record and its prior adjustments.
        const auto motion = sceneMotionUniforms(runtimeInputs, authoredCamera, width, height);
        if (deformationMotionProbeEnabled())
            std::fprintf (stderr,
                "[scene-motion] previousMotion=%d projectionW=%s usePrevious=%d rotation=[%s] translationScale=[%s] cameraRotation=[%s] cameraTranslation=[%s]\n",
                runtimeInputs.previousMotion.has_value() ? 1 : 0,
                deformationMotionProbeFloats (motion.projection.data() + 3, 1).c_str(),
                resources->usePreviousDeformation || priorScenePositions ? 1 : 0,
                deformationMotionProbeFloats (motion.rotation.data(), motion.rotation.size()).c_str(),
                deformationMotionProbeFloats (motion.translationScale.data(), motion.translationScale.size()).c_str(),
                deformationMotionProbeFloats (motion.cameraRotation.data(), motion.cameraRotation.size()).c_str(),
                deformationMotionProbeFloats (motion.cameraTranslation.data(), motion.cameraTranslation.size()).c_str());
        gl.Uniform4fv(locations.motionRotation, 1, motion.rotation.data());
        gl.Uniform4fv(locations.motionTranslationScale, 1, motion.translationScale.data());
        gl.Uniform4fv(locations.previousCameraRotation, 1, motion.cameraRotation.data());
        gl.Uniform4fv(locations.previousCameraTranslation, 1, motion.cameraTranslation.data());
        gl.Uniform4fv(locations.previousProjection, 1, motion.projection.data());
        gl.Uniform4fv(locations.motionViewport, 1, motion.viewport.data());
        gl.Uniform4f(locations.roughCameraControl,float(resources->smoothGlass.roughConsumer.version),0,0,float(height));
        gl.Uniform1i(locations.usePreviousDeformation,
            resources->usePreviousDeformation || priorScenePositions ? 1 : 0);
        gl.Uniform1i(locations.noteMotion, noteInstances.admitted ? 1 : 0);
        gl.Uniform1i(locations.previousNoteTransforms,10);
        if (noteInstances.admitted)
        {
            std::array<std::array<float,4>,visualnoteinstancing::kMaximumInstances> previousNotes {};
            if (runtimeInputs.previousMotion && runtimeInputs.previousMotion->noteTransforms)
                for (std::size_t index=0;index<noteInstances.count;++index)
                    if (const auto transform = runtimeInputs.previousMotion->noteTransforms->find(noteInstances.identities[index]))
                        previousNotes[index] = *transform;
            gl.ActiveTexture(GL_TEXTURE10); gl.BindSampler(10,0);
            glGenTextures(1,&frame->previousNoteTexture);
            glBindTexture(GL_TEXTURE_2D,frame->previousNoteTexture);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_BASE_LEVEL,0);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,visualnoteinstancing::kMaximumInstances,1,0,
                         GL_RGBA,GL_FLOAT,previousNotes.data());
            if (!frame->previousNoteTexture || glGetError()!=GL_NO_ERROR)
                return fail("OpenGL preceding note transform allocation failed");
            result.stats.textureBytes += sizeof(previousNotes);
            gl.ActiveTexture(GL_TEXTURE0);
        }
        // Without a predecessor the prior inputs equal the current ones, so the
        // second program evaluation lands on the same vertex.
        const float priorVertexTime = runtimeInputs.previousMotion
            ? runtimeInputs.previousMotion->timeSeconds : runtimeInputs.timeSeconds;
        const auto& priorVertexSpectrum = runtimeInputs.previousMotion
            ? runtimeInputs.previousMotion->vertexSpectrum : runtimeInputs.vertexSpectrum;
        gl.Uniform4f(locations.vertexTime, runtimeInputs.timeSeconds, priorVertexTime, 0.0f, 0.0f);
        if (materialProgram != nullptr && materialProgram->vertexProgram)
        {
            const auto spectrumVectors = (materialProgram->vertexProgram->resources().audioParameterSlots + 3) / 4;
            gl.Uniform4fv(locations.vertexSpectrum, spectrumVectors, runtimeInputs.vertexSpectrum.data());
            gl.Uniform4fv(locations.previousVertexSpectrum, spectrumVectors, priorVertexSpectrum.data());
            const auto displacementLimits = fixtureVertexDisplacementLimits (materialProgram->vertexProgram.get());
            gl.Uniform4fv(locations.vertexDisplacementLimits, static_cast<int> (kVertexDisplacementLimitCapacity / 4), displacementLimits.data());
        }
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
        gl.Uniform4f(locations.glassTransport, surfacePbr ? float(materialProgram->parameters.identifiers[1]) : 0.0f,
            coating[3], float(resources->smoothGlass.object.value & 65535u), float(resources->smoothGlass.object.value >> 16u));
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
        analyticsky::Uniforms sky {};
        for (std::size_t i=0;i<lightCount;++i)
            if (lightAt(i).kind==SceneLightKind::Environment && lightAt(i).sky.version!=0)
                sky=analyticsky::uniforms(lightAt(i));
        for (std::size_t i=0;i<lightCount;++i)
            if (lightAt(i).environmentImage.version)
            {
                if (!resources->environmentResident || !environmentimage::validBinding(*resources->environmentResident,lightAt(i))) return fail("HDR environment runtime identity differs from prepared scene");
                sky=environmentimage::uniforms(lightAt(i));
            }
        gl.ActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D,resources->environmentTexture ? resources->environmentTexture : resources->textures[0]);
        gl.BindSampler(7,0);
        gl.Uniform1i(locations.environmentImage,7);
        gl.Uniform4fv(locations.sky,5,sky[0].data());
        gl.Uniform1f(locations.skyBackground,0.0f);
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
        std::uint32_t objectInstanceDrawCount = 0;
        std::uint32_t submittedObjectInstanceCount = 0;
        if (sky[0][3]>0.5f && materialKind!=NativeFixtureMaterialKind::DiffractionReflective
            && runtimeInputs.imageOutput==renderpassoutput::Output::Color
            && (!runtimeInputs.passComposite || !runtimeInputs.passComposite->inspectionImage))
        {
            gl.Uniform1f(locations.skyBackground,1.0f);
            glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE);
            glDrawArrays(GL_TRIANGLES,0,3);
            glDepthMask(GL_TRUE); glEnable(GL_DEPTH_TEST);
            gl.Uniform1f(locations.skyBackground,0.0f);
        }
        const bool mixedMaterials = diffractionFrame && !materialProgram->objectPrograms.empty();
        std::uint32_t surfaceDrawCount = 0;
        const auto gratingValues = values;
        bool diffractionComplete = true;
        const auto validateDiffractionDraw = [&](unsigned texture = 0) {
            if (!authoredDiffraction) return true;
            GLint savedTexture = 0, savedBuffer = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTexture);
            glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &savedBuffer);
            constexpr std::array<GLenum, 6> names { GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH,
                GL_PACK_SKIP_PIXELS, GL_PACK_SKIP_ROWS, GL_PACK_SWAP_BYTES, GL_PACK_LSB_FIRST };
            std::array<GLint, 6> saved {};
            for (std::size_t i=0; i<names.size(); ++i) {
                glGetIntegerv(names[i], &saved[i]); glPixelStorei(names[i], i == 0 ? 1 : 0);
            }
            gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, texture ? texture : frame->accumulationTexture_);
            std::vector<float> completion(std::size_t(width)*height*4);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, completion.data());
            const auto readError = glGetError();
            glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(savedTexture));
            gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<unsigned>(savedBuffer));
            for (std::size_t i=0; i<names.size(); ++i) glPixelStorei(names[i], saved[i]);
            if (readError != GL_NO_ERROR) { result.error="OpenGL diffraction whole-output readback failed"; return false; }
            if (!diffractiontransportnative::validateWholeOutput(completion, width, height, result.error))
                return false;
            return true;
        };
        bool roughReceiptPass = false;
        const auto drawSurfaces = [&](int stage) {
            gl.Uniform1i(locations.materialKind, static_cast<int>(NativeFixtureMaterialKind::SurfacePbr));
            if (diffractionFrame) {
                auto linearPass = passProgram;
                linearPass.control = {0.0f, 0.0f, 1.0f, 0.0f};
                gl.Uniform4fv(locations.passProgramControl, 1, linearPass.control.data());
                gl.Uniform3f(locations.ambient, scene->ambientColor.x, scene->ambientColor.y, scene->ambientColor.z);
            }
            glDepthFunc(roughReceiptPass ? GL_EQUAL : GL_LESS);

            const diffractionmaterial::PhysicalDiffractionGpuParameters emptyDiffraction {};
            const auto& diffraction = authoredDiffraction
                ? materialProgram->diffractionPaths[0].material : emptyDiffraction;
            gl.Uniform4fv (locations.diffractionGeometry, 1, &diffraction.geometry.x);
            gl.Uniform4fv (locations.diffractionSecondaryGeometry, 1,
                           &diffraction.secondaryGeometry.x);
            gl.Uniform4fv (locations.diffractionMicrostructure, 1,
                           &diffraction.microstructure.x);
            gl.Uniform4fv (locations.diffractionControl, 1, &diffraction.control.x);
            gl.Uniform4fv (locations.diffractionCoating, 1, &diffraction.coating.x);
            gl.Uniform4fv (locations.diffractionRoughness, 1, &diffraction.roughness.x);
            gl.Uniform4fv(locations.diffractionGrooveField, 1, &diffraction.grooveField.x);
            gl.Uniform4fv(locations.diffractionGrooveVariation, 1, &diffraction.grooveVariation.x);
            gl.Uniform4fv (locations.diffractionSpectral, 8, &diffraction.spectral[0].x);
            gl.Uniform4fv (locations.diffractionSpectralZ, 8, &diffraction.spectralZ[0].x);
            if (authoredDiffraction) {
                gl.Uniform4fv(locations.diffractionFoilField, 4, &materialProgram->diffractionFoilField[0].x);
                gl.Uniform4fv(locations.diffractionOccupancyRectangles, 5, &materialProgram->diffractionOccupancyRectangles[0].x);
                const auto& path=materialProgram->diffractionPaths[0];
                float coverage=0.0f;
                std::memcpy(&coverage,&path.kindBounceAndReserved[3],sizeof(coverage));
                gl.Uniform4f(locations.diffractionPathKindAndBounce,
                    static_cast<float>(path.kindBounceAndReserved[0]),
                    static_cast<float>(path.kindBounceAndReserved[1]),
                    static_cast<float>(path.kindBounceAndReserved[2]),coverage);
                gl.Uniform4f(locations.diffractionSpatialCounts,
                    materialProgram->diffractionFoilMaximumEvaluations > 0 ? 4.0f : 0.0f,
                    static_cast<float>(materialProgram->diffractionOccupancyRectangleCount), 0.0f, 0.0f);
            }
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
                    const auto matrix = nativeObjectWorldMatrix (runtimeInputs.objectWorlds.get(),
                        scene->objects[objectIndex], fixtureWorldMatrix (*scene, scene->objects[objectIndex]));
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
            else if (scene->instancedObject.isValid())
            {
                const auto* instanced = visual3d_detail::findById(
                    scene->objects, scene->objectCount, scene->instancedObject);
                auto runtimeTransform = SceneTransform3D {};
                if (runtimeTargetsObject(runtimeInputs, *instanced))
                {
                    runtimeTransform.translation = {runtimeInputs.objectTranslationOffset[0],
                        runtimeInputs.objectTranslationOffset[1], runtimeInputs.objectTranslationOffset[2]};
                    runtimeTransform = applyRuntimeObjectTransform(runtimeTransform,
                        runtimeInputs.objectRotationDegrees, runtimeInputs.objectScale);
                }
                const auto world = multiplyFixtureMatrices(fixtureTransformMatrix(runtimeTransform),
                    fixtureWorldMatrix(*scene, *instanced));
                std::array<OpenGlInstanceGpuRecord, Visual3DScene::kMaxObjectInstances> objectRecords {};
                std::array<std::pair<float, std::size_t>, Visual3DScene::kMaxObjectInstances> order {};
                for (std::size_t index = 0; index < scene->objectInstanceCount; ++index)
                {
                    const auto matrix = nativeObjectInstanceMatrix(world, scene->objectInstances[index]);
                    const auto dx = matrix[12] - values.cameraTranslation[0];
                    const auto dy = matrix[13] - values.cameraTranslation[1];
                    const auto dz = matrix[14] - values.cameraTranslation[2];
                    order[index] = {dx * dx + dy * dy + dz * dz, index};
                }
                // Blended instances draw back to front in one call. Equal
                // distances keep the scene's instance order.
                const auto* instancedMaterial = visual3d_detail::findById(
                    scene->materials, scene->materialCount, instanced->material);
                if (instancedMaterial != nullptr && instancedMaterial->alphaMode == SceneAlphaMode::Blend)
                    std::stable_sort(order.begin(), order.begin() + scene->objectInstanceCount,
                        [] (const auto& left, const auto& right) { return left.first > right.first; });
                for (std::size_t index = 0; index < scene->objectInstanceCount; ++index)
                {
                    const auto source = order[index].second;
                    auto& record = objectRecords[index];
                    record.matrix = nativeObjectInstanceMatrix(world, scene->objectInstances[source]);
                    // Mode 2 reads these three vectors as the prior world matrix.
                    const auto prior = nativeObjectInstancePreviousColumns(world, *scene, source);
                    record.identityColor = prior[0]; record.color = prior[1]; record.emission = prior[2];
                }
                gl.BindBuffer(GL_ARRAY_BUFFER, resources->instanceBuffer);
                gl.BufferData(GL_ARRAY_BUFFER,
                    static_cast<std::ptrdiff_t>(scene->objectInstanceCount * sizeof(OpenGlInstanceGpuRecord)),
                    objectRecords.data(), GL_DYNAMIC_DRAW);
                ++instanceBufferUploadCount;
            }
            for (const auto objectIndex : drawOrder)
            {
                const auto& drawObject = scene->objects[objectIndex];
                if (roughReceiptPass && drawObject.id != resources->smoothGlass.object) continue;
                if (diffractionFrame && drawObject.id == materialProgram->object) continue;
                const auto* queued = visual3d_detail::findById(scene->materials, scene->materialCount, drawObject.material);
                const bool blendDraw = nativeSurfaceRequiresBlend(nativeSurfaceForObject(materialProgram.get(), drawObject.id))
                    || (!resources->instanceAppearances.empty() && resources->instanceAppearances[objectIndex].color[3] < 1.0f)
                    || (queued && queued->alphaMode == SceneAlphaMode::Blend);
                if (stage >= 0 && blendDraw != (stage == 1)) continue;
                gl.Uniform1i(locations.rawObjectIndex, static_cast<int>(objectIndex));
                gl.Uniform1i(locations.diffractionTriangleCount, static_cast<int>((drawObject.indexCount ? drawObject.indexCount : drawObject.vertexCount)/3));
                const auto identityColor = render3dimage::identityColor (
                    runtimeInputs.imageOutput == renderpassoutput::Output::MaterialId
                        ? drawObject.material.value : drawObject.id.value);
                gl.Uniform4f (locations.imageOutput, roughReceiptPass ? -1.0f
                    : static_cast<float> (diffractionFrame ? renderpassoutput::Output::Color : runtimeInputs.imageOutput),
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
                const bool objectTargeted = runtimeTargetsObject(runtimeInputs, drawObject);
                if (!objectTargeted) runtimeTransform = {};
                values.objectMatrix = multiplyFixtureMatrices (
                    fixtureTransformMatrix (runtimeTransform),
                    nativeObjectWorldMatrix (runtimeInputs.objectWorlds.get(), drawObject,
                        fixtureWorldMatrix (*scene, drawObject)));
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
                    gl.Uniform4f(locations.glassTransport, drawProgram ? float(drawProgram->parameters.identifiers[1]) : 0.0f,
                        drawCoating[3], float(resources->smoothGlass.object.value & 65535u), float(resources->smoothGlass.object.value >> 16u));
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
                const auto objectMotion = runtimeInputs.objectWorlds
                    ? sceneObjectWorldMotionUniforms(motion, runtimeTransform, objectTargeted,
                        runtimeInputs.previousMotion ? runtimeInputs.previousMotion->objectWorlds.get() : nullptr,
                        runtimeInputs.objectWorlds.get(), drawObject.id)
                    : sceneObjectMotionUniforms(motion, *scene, objectIndex, runtimeTransform, objectTargeted);
                gl.Uniform4fv(locations.motionRotation, 1, objectMotion.rotation.data());
                gl.Uniform4fv(locations.motionTranslationScale, 1, objectMotion.translationScale.data());
                if (deformationMotionProbeEnabled())
                {
                    // Inputs behind this draw's Motion: the prior and current checked worlds,
                    // the per-object step, and whether the prior clip path is live (projectionW).
                    const auto* priorEntry = runtimeInputs.previousMotion && runtimeInputs.objectWorlds
                        ? nativeObjectWorldEntry (runtimeInputs.previousMotion->objectWorlds.get(), drawObject.id) : nullptr;
                    const auto* currentEntry = runtimeInputs.objectWorlds
                        ? nativeObjectWorldEntry (runtimeInputs.objectWorlds.get(), drawObject.id) : nullptr;
                    const auto worldText = [](const auto* entry) {
                        return entry != nullptr
                            ? deformationMotionProbeFloats (entry->world.data(), entry->world.size())
                            : std::string ("none");
                    };
                    std::fprintf (stderr,
                        "[motion-inputs] object=%u previousMotion=%d projectionW=%s priorWorldPresent=%d currentWorldPresent=%d worldsEqual=%d objectRotation=[%s] objectTranslationScale=[%s] priorWorld=[%s] currentWorld=[%s]\n",
                        static_cast<unsigned> (drawObject.id.value),
                        runtimeInputs.previousMotion.has_value () ? 1 : 0,
                        deformationMotionProbeFloats (motion.projection.data () + 3, 1).c_str (),
                        priorEntry != nullptr ? 1 : 0,
                        currentEntry != nullptr ? 1 : 0,
                        priorEntry != nullptr && currentEntry != nullptr && priorEntry->world == currentEntry->world ? 1 : 0,
                        deformationMotionProbeFloats (objectMotion.rotation.data (), objectMotion.rotation.size ()).c_str (),
                        deformationMotionProbeFloats (objectMotion.translationScale.data (), objectMotion.translationScale.size ()).c_str (),
                        worldText (priorEntry).c_str (),
                        worldText (currentEntry).c_str ());
                }
                const bool objectInstanceDraw = drawObject.id == scene->instancedObject;
                gl.Uniform1i(locations.useInstanceMatrix,
                             resources->instancedSharedGeometry ? 1 : objectInstanceDraw ? 2 : 0);
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
                              drawFrame ? (selectedFrame && materialFrameIsLinear(selectedFrame->colorTextureDescriptor()) ? 3.0f
                                  : materialFrameIsBottomFirst(selectedFrame) ? 2.0f : 1.0f) : 0.0f);
                gl.Uniform1f (locations.emissiveTexturePresence,
                              textureIds[4].isValid() ? 1.0f : 0.0f);
                const auto alphaMode = nativeSurfaceRequiresBlend(drawProgram) || (!resources->instanceAppearances.empty()
                    && resources->instanceAppearances[objectIndex].color[3]<1.0f) ? SceneAlphaMode::Blend
                    : drawMaterial != nullptr ? drawMaterial->alphaMode : SceneAlphaMode::Opaque;
                gl.Uniform2f (locations.alphaModeCutoff,
                              alphaMode == SceneAlphaMode::Mask ? 1.0f : 0.0f,
                              drawMaterial != nullptr ? drawMaterial->alphaCutoff : 0.5f);
                gl.Uniform1f (locations.tangentHandednessSign,
                              resources->instancedSharedGeometry || objectInstanceDraw ? 1.0f
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
                    glDepthMask (roughReceiptPass ? GL_FALSE : GL_TRUE);
                }
                if (drawObject.indexCount == 0)
                    continue;
                const auto* deformedWinding = resources->deformedWindingFor (drawObject.firstVertex);
                const bool jointsMixed = deformedWinding != nullptr && deformedWinding->jointsMixed;
                const bool mirrored = fixtureDrawMirrored (values.objectMatrix, deformedWinding);
                // A mixed palette has no single front face, so this draw keeps back faces. The scope
                // restores the previous cull and front-face state when this member's draw ends.
                const OpenGlCullStateScope cullScope (jointsMixed);
                if ((drawMaterial != nullptr && drawMaterial->doubleSided)
                    || (drawProgram && drawProgram->parameters.identifiers[1] != 0) || jointsMixed)
                    glDisable (GL_CULL_FACE);
                else
                {
                    glEnable (GL_CULL_FACE);
                    glCullFace (GL_BACK);
                }
                glFrontFace (mirrored ? GL_CW : GL_CCW);
                if (deformationMotionProbeEnabled ())
                {
                    GLint vertexArray = 0;
                    GLint frontFace = 0;
                    GLint attributeEnabled = 0;
                    GLint attributeStride = 0;
                    GLint attributeBuffer = 0;
                    void* attributePointer = nullptr;
                    glGetIntegerv (GL_VERTEX_ARRAY_BINDING, &vertexArray);
                    glGetIntegerv (GL_FRONT_FACE, &frontFace);
                    gl.GetVertexAttribiv (12, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &attributeEnabled);
                    gl.GetVertexAttribiv (12, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &attributeStride);
                    gl.GetVertexAttribiv (12, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &attributeBuffer);
                    gl.GetVertexAttribPointerv (12, GL_VERTEX_ATTRIB_ARRAY_POINTER, &attributePointer);
                    std::fprintf (stderr,
                        "[deformation-draw] member=%zu object=%u vao=%d expectedVao=%u mirrored=%s frontFace=%s cull=%d objectDeterminant=%g attribute12Enabled=%d attribute12Buffer=%d attribute12Offset=%zu attribute12Stride=%d usePrevious=%d\n",
                        deformationMotionProbeMember,
                        static_cast<unsigned> (drawObject.id.value),
                        vertexArray,
                        resources->vertexArray,
                        jointsMixed ? "mixed" : mirrored ? "1" : "0",
                        glIsEnabled (GL_CULL_FACE) != GL_TRUE ? "nocull"
                            : static_cast<unsigned> (frontFace) == GL_CW ? "cw" : "ccw",
                        glIsEnabled (GL_CULL_FACE) == GL_TRUE ? 1 : 0,
                        static_cast<double> (fixtureMatrixDeterminant3x3 (values.objectMatrix)),
                        attributeEnabled,
                        attributeBuffer,
                        static_cast<std::size_t> (reinterpret_cast<std::uintptr_t> (attributePointer)),
                        attributeStride,
                        resources->usePreviousDeformation || priorScenePositions ? 1 : 0);
                    // The opaque visibility BVH is prepared from the rest-pose scene vertices, so a
                    // palette-deformed member casts shadows from its authored geometry, not the skinned
                    // buffer drawn here. vertexDeformation() sees only a vertex program, not the palette.
                    const auto& occluderMatrix = visibilityMatrices[objectIndex];
                    std::fprintf (stderr,
                        "[visibility-draw] member=%zu object=%u visibilityApplied=%d occluderMeshes=%zu occluderSource=rest deformed=%d objectWorldsSupplied=%d occluderTranslation=%.9g,%.9g,%.9g occluderDeterminant=%.9g vertexProgramDeformation=%d\n",
                        deformationMotionProbeMember,
                        static_cast<unsigned> (drawObject.id.value),
                        directVisibility ? 1 : 0,
                        resources->opaqueVisibility.meshes.size(),
                        deformedWinding != nullptr ? 1 : 0,
                        runtimeInputs.objectWorlds != nullptr ? 1 : 0,
                        static_cast<double> (occluderMatrix[12]),
                        static_cast<double> (occluderMatrix[13]),
                        static_cast<double> (occluderMatrix[14]),
                        static_cast<double> (fixtureMatrixDeterminant3x3 (occluderMatrix)),
                        opaquevisibility::vertexDeformation (materialProgram.get()) ? 1 : 0);
                }
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
                if (objectInstanceDraw)
                {
                    // Instance scale is positive, so the object matrix sets the winding.
                    gl.Uniform1i(locations.noteInstanceIndex, 0);
                    gl.DrawElementsInstanced(GL_TRIANGLES,
                        static_cast<int>(drawObject.indexCount), GL_UNSIGNED_INT,
                        indexOffset, static_cast<int>(scene->objectInstanceCount));
                    if (!roughReceiptPass) ++surfaceDrawCount;
                    ++objectInstanceDrawCount;
                    submittedObjectInstanceCount += static_cast<std::uint32_t>(scene->objectInstanceCount);
                    continue;
                }
                gl.Uniform1i(locations.noteInstanceIndex, -1);
                if (!roughReceiptPass) ++surfaceDrawCount;
                gl.DrawElementsInstanced(GL_TRIANGLES,
                    static_cast<int>(drawObject.indexCount), GL_UNSIGNED_INT,
                    indexOffset, static_cast<int>(noteInstances.count));
                if (authoredDiffraction && drawProgram && drawProgram->spectralReflectance
                    && !validateDiffractionDraw()) { diffractionComplete=false; return; }
                if (noteInstances.admitted)
                {
                    ++noteInstanceDrawCount;
                    submittedNoteInstanceCount += static_cast<std::uint32_t>(noteInstances.count);
                }
            }
            // Later draws use the frame Motion uniforms again.
            gl.Uniform4fv(locations.motionRotation, 1, motion.rotation.data());
            gl.Uniform4fv(locations.motionTranslationScale, 1, motion.translationScale.data());
        };
        if (materialKind == NativeFixtureMaterialKind::DiffractionReflective)
        {
            if (mixedMaterials) drawSurfaces(0);
            if (!diffractionComplete) return fail(result.error);
            values = gratingValues;
            const auto* gratingDeformed = resources->deformedWindingFor (object.firstVertex);
            const bool gratingJointsMixed = mixedMaterials && gratingDeformed != nullptr
                && gratingDeformed->jointsMixed;
            const OpenGlCullStateScope gratingCullScope (gratingJointsMixed);
            if (mixedMaterials) {
                SceneTransform3D transform {};
                transform.translation = {runtimeInputs.objectTranslationOffset[0], runtimeInputs.objectTranslationOffset[1], runtimeInputs.objectTranslationOffset[2]};
                transform = applyRuntimeObjectTransform(transform, runtimeInputs.objectRotationDegrees, runtimeInputs.objectScale);
                if (!runtimeTargetsObject(runtimeInputs, object)) transform = {};
                values.objectMatrix = multiplyFixtureMatrices(fixtureTransformMatrix(transform), fixtureWorldMatrix(*scene, object));
            }
            gl.Uniform1i(locations.materialKind, static_cast<int>(materialKind));
            gl.Uniform1i(locations.rawObjectIndex, static_cast<int>(&object - scene->objects.data()));
            gl.Uniform1i(locations.diffractionTriangleCount, static_cast<int>(object.indexCount/3));
            gl.UniformMatrix4fv(locations.objectMatrix, 1, GL_FALSE, values.objectMatrix.data());
            gl.Uniform1f(locations.tangentHandednessSign, fixtureMatrixDeterminant3x3(values.objectMatrix) < 0.0f ? -1.0f : 1.0f);
            gl.Uniform4fv(locations.baseColor, 1, values.baseColor.data());
            gl.Uniform4fv(locations.materialParams, 1, values.materialParams.data());
            gl.Uniform4fv(locations.timeMixEndColorAndTime, 1, values.timeMixEndColorAndTime.data());
            gl.Uniform3f(locations.emissive, values.emissive[0], values.emissive[1], values.emissive[2]);
            gl.Uniform3f(locations.ambient, values.ambient[0], values.ambient[1], values.ambient[2]);
            if (mixedMaterials) {
                gl.Uniform4f(locations.imageOutput, 0.0f, 0.0f, 0.0f, 0.0f);
                if (material.doubleSided || gratingJointsMixed) glDisable(GL_CULL_FACE);
                else { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
                glFrontFace(fixtureDrawMirrored(values.objectMatrix, gratingDeformed) ? GL_CW : GL_CCW);
            }
            glDisable(GL_BLEND); glDepthMask(GL_TRUE); glDepthFunc(GL_LESS);

            gl.ActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, materialFrameTexture != 0
                ? materialFrameTexture : resources->textures[0]);
            gl.BindSampler(0, 0);
            gl.Uniform4f(locations.texturePresence, 0, 0, 0, materialFrameTexture != 0
                ? (materialFrameIsLinear(rootFrame->colorTextureDescriptor()) ? 3.0f
                    : materialFrameIsBottomFirst(rootFrame) ? 2.0f : 1.0f) : 0.0f);
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
                return fail ("OpenGL fixture diffraction workload exceeds backend limits");
            const auto uintAsFloat = [] (std::uint32_t value) noexcept
            {
                float encoded = 0.0f;
                std::memcpy(&encoded, &value, sizeof(value));
                return encoded;
            };
            for (std::uint8_t pathIndex = 0; pathIndex < diffractionPathCount; ++pathIndex)
            {
                gl.Uniform1i(locations.diffractionPathIndex, static_cast<int>(pathIndex));
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
                    if (pathIndex == 1)
                    {
                        const unsigned colorOnly[] { GL_COLOR_ATTACHMENT0,
                            GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE };
                        gl.DrawBuffers(8, colorOnly);
                    }
                    glEnable(GL_BLEND);
                    for (unsigned attachment = 1; attachment < 8; ++attachment) gl.Disablei(GL_BLEND, attachment);
                    gl.BlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE);
                    glDepthFunc(GL_EQUAL);
                    glDepthMask(GL_FALSE);
                }
                gl.DrawElementsInstanced (GL_TRIANGLES,
                                          static_cast<int> (object.indexCount),
                                          GL_UNSIGNED_INT, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(object.firstIndex) * sizeof(std::uint32_t)),
                                          static_cast<int>(noteInstances.count));
                if (authoredDiffraction && !validateDiffractionDraw()) return fail(result.error);
                if (noteInstances.admitted)
                {
                    ++noteInstanceDrawCount;
                    submittedNoteInstanceCount += static_cast<std::uint32_t>(noteInstances.count);
                }
            }
            if (mixedMaterials) {
                const unsigned attachments[] {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
                    GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
                gl.DrawBuffers(8, attachments);
                drawSurfaces(1);
                if (!diffractionComplete) return fail(result.error);
            }
            if (authoredDiffraction && !validateDiffractionDraw()) return fail(result.error);
            gl.GenFramebuffers(1, &resolveFramebuffer);
            gl.BindFramebuffer(GL_FRAMEBUFFER, resolveFramebuffer);
            gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D, frame->texture_, 0);
            const unsigned colorAttachment = GL_COLOR_ATTACHMENT0;
            gl.DrawBuffers(1, &colorAttachment);
            if (resolveFramebuffer == 0
                || gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                return fail("OpenGL diffraction resolve target creation failed");
            glDisable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            const auto clearColor = runtimeInputs.imageOutput == renderpassoutput::Output::Color
                && !runtimeInputs.passComposite ? background : emptyPass;
            gl.ClearBufferfv(GL_COLOR, 0, clearColor);
            if (runtimeInputs.passComposite && runtimeInputs.passComposite->inspectionImage)
            {
                const auto clear = renderpasscomposite::inspectionClear(*runtimeInputs.passComposite);
                gl.ClearBufferfv(GL_COLOR, 0, clear.data());
            }
            const auto& resolve = *resources->sharedProgram;
            gl.UseProgram(resolve.diffractionResolveId);
            gl.BindVertexArray(resolve.diffractionResolveVao);
            const std::array<unsigned, 8> inputs { frame->accumulationTexture_,
                frame->depthTexture_, frame->rawTextures[0], frame->rawTextures[1],
                frame->rawTextures[2], frame->rawTextures[3], frame->rawTextures[4],
                frame->rawTextures[5] };
            const std::array<const char*, 8> samplers { "uAccumulatedColor", "uLinearDepth",
                "uRawNormal", "uRawEmission", "uRawMask", "uRawMaterialId",
                "uRawObjectId", "uRawMotion" };
            for (std::size_t unit = 0; unit < inputs.size(); ++unit)
            {
                gl.ActiveTexture(GL_TEXTURE0 + static_cast<unsigned>(unit));
                glBindTexture(GL_TEXTURE_2D, inputs[unit]);
                gl.BindSampler(static_cast<unsigned>(unit), 0);
                gl.Uniform1i(gl.GetUniformLocation(resolve.diffractionResolveId,
                    samplers[unit]), static_cast<int>(unit));
            }
            const auto resolveLocation = [&](const char* name)
                { return gl.GetUniformLocation(resolve.diffractionResolveId, name); };
            gl.Uniform4fv(resolveLocation("uPassProgramControl"), 1, passProgram.control.data());
            gl.Uniform4fv(resolveLocation("uPassModes[0]"), 8, passProgram.modes[0].data());
            gl.Uniform4fv(resolveLocation("uPassColors[0]"), 8, passProgram.colors[0].data());
            gl.Uniform4fv(resolveLocation("uPassReferences[0]"), 8,
                passProgram.references[0].data());
            gl.Uniform4fv(resolveLocation("uPassTransforms[0]"), 8,
                passProgram.transforms[0].data());
            gl.Uniform4fv(resolveLocation("uMotionViewport"), 1, motion.viewport.data());
            gl.Uniform1i(resolveLocation("uImageOutput"),
                static_cast<int>(runtimeInputs.imageOutput));
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        else if (roughGlassFrame)
        {
            auto& completion = *roughCompletion;
            const auto ownerValid = [&] {
                return roughglasstiles::sameOwner(frame->staticResources, resources)
                    && roughglasstiles::sameOwner(resources->snapshot, scene)
                    && roughglasstiles::sameOwner(resources->smoothGlass.sourceScene, roughKey.scene)
                    && roughglasstiles::sameOwner(resources->smoothGlass.sourceMaterial, materialProgram)
                    && frame->rendererGeneration_ == roughKey.rendererGeneration
                    && resources->rendererGeneration == roughKey.rendererGeneration
                    && glfwGetCurrentContext() == frame->ownerContext
                    && frame->width_ == roughKey.width && frame->height_ == roughKey.height;
            };
            const bool selected=smoothglass::multipleScatterCamera(resources->smoothGlass);
            RoughSelectedCamera selectedCamera(gl,resources->queryContext,resources->rendererGeneration);
            if(selected && !selectedCamera.initialize(roughSchedule->maximumTilePixels(),result.error))return fail(result.error);
            for (const auto& tile : roughSchedule->tiles())
            {
                if (!ownerValid()) return fail("OpenGL rough tile GPU frame owner changed before submission");
                const auto bottom = height - tile.y - tile.height;
                std::array<NativeRawPassPixels,3> receipts;
                std::vector<roughglasstiles::NativeTileReceipt::Work> nativeWork;
                std::vector<std::array<std::uint32_t,4>> selectedIds;
                if(selected && !selectedCamera.run(tile,roughKey,resources->smoothGlass,resources->program,
                    glassFrame,glassEmission,receipts,nativeWork,selectedIds,completion.aggregate().microWork,result.error))return fail(result.error);
                // Product attachments retain the full-frame viewport/camera.
                // Scissor bounds Color and receipt work to this admitted job.
                gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
                glViewport(0, 0, int(width), int(height));
                glEnable(GL_SCISSOR_TEST);
                glScissor(int(tile.x), int(bottom), int(tile.width), int(tile.height));
                gl.Uniform4f(locations.roughCameraControl,float(resources->smoothGlass.roughConsumer.version),0,0,float(height));
                drawSurfaces(-1);
                glDisable(GL_SCISSOR_TEST);
                if(!selected) {
                    if (roughReceiptDepth)
                    {
                        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
                        gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, roughReceiptFramebuffer);
                        gl.BlitFramebuffer(int(tile.x), int(bottom), int(tile.x+tile.width), int(bottom+tile.height),
                            0, 0, int(tile.width), int(tile.height), GL_DEPTH_BUFFER_BIT, GL_NEAREST);
                    }
                    gl.BindFramebuffer(GL_FRAMEBUFFER, roughReceiptFramebuffer);
                    // Integer translation preserves clip interpolation and sample
                    // centers. Rough transport uses world/camera coordinates and
                    // v1 keeps its fixed strata and the full-frame camera.
                    glViewport(-int(tile.x), -int(bottom), int(width), int(height));
                    glEnable(GL_SCISSOR_TEST);
                    glScissor(0, 0, int(tile.width), int(tile.height));
                    gl.ClearBufferfv(GL_COLOR, 0, emptyPass);
                    gl.ClearBufferfv(GL_COLOR, 2, emptyPass);
                    gl.ClearBufferfv(GL_COLOR, 3, emptyPass);
                    glEnable(GL_DEPTH_TEST); glDisable(GL_BLEND);
                    roughReceiptPass = true;
                    gl.Uniform4f(locations.roughCameraControl,float(resources->smoothGlass.roughConsumer.version),float(tile.x),-float(bottom),float(height));
                    drawSurfaces(-1);
                    roughReceiptPass = false;
                    glDisable(GL_SCISSOR_TEST);
                    if (!roughglasstiles::opengl::completeSubmission(gl, result.error)) return fail(result.error);
                    for (std::size_t i=0; i<receipts.size(); ++i)
                        if (!roughglasstiles::opengl::readRegion(gl, roughReceiptFramebuffer,
                            GL_COLOR_ATTACHMENT0+unsigned(i), 0, 0, tile.width, tile.height,
                            NativeTexturePixelFormat::Rgba32Float, receipts[i], result.error)) return fail(result.error);
                } else {
                    // Private diagnostic textures contain the same completed
                    // result as Color. No private raster transport is submitted.
                    for(std::size_t attachment=0;attachment<3;++attachment) {
                        auto upload=receipts[attachment].bytes;
                        const auto rowBytes=std::size_t(tile.width)*16;
                        for(std::uint32_t row=0;row<tile.height/2;++row)
                            std::swap_ranges(upload.data()+row*rowBytes,upload.data()+(row+1)*rowBytes,
                                upload.data()+(tile.height-1-row)*rowBytes);
                        gl.ActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,frame->roughReceiptTextures[attachment]);
                        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,int(tile.width),int(tile.height),0,GL_RGBA,GL_FLOAT,upload.data());
                    }
                    if(runtimeInputs.imageOutput==renderpassoutput::Output::Color) {
                        if(!resolveFramebuffer) {
                            gl.GenFramebuffers(1,&resolveFramebuffer);
                            gl.BindFramebuffer(GL_FRAMEBUFFER,resolveFramebuffer);
                            gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,frame->texture_,0);
                            const unsigned colorOnly=GL_COLOR_ATTACHMENT0;gl.DrawBuffers(1,&colorOnly);
                            if(!resolveFramebuffer || gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
                                return fail("OpenGL selected camera Color resolve framebuffer is incomplete");
                        } else gl.BindFramebuffer(GL_FRAMEBUFFER,resolveFramebuffer);
                        const unsigned images[]{frame->rawTextures[0],frame->depthTexture_,frame->rawTextures[2],
                            frame->rawTextures[1],frame->rawTextures[5]};
                        for(unsigned unit=0;unit<5;++unit) {
                            gl.ActiveTexture(GL_TEXTURE0+unit);gl.BindSampler(unit,0);glBindTexture(GL_TEXTURE_2D,images[unit]);
                        }
                        gl.ActiveTexture(GL_TEXTURE0);glViewport(0,0,int(width),int(height));
                        glEnable(GL_SCISSOR_TEST);glScissor(int(tile.x),int(bottom),int(tile.width),int(tile.height));
                        glDisable(GL_DEPTH_TEST);glDepthMask(GL_FALSE);glDisable(GL_BLEND);glDisable(GL_CULL_FACE);
                        gl.Uniform1f(locations.skyBackground,1);
                        gl.Uniform1f(gl.GetUniformLocation(resources->program,"uSelectedRoughResolve"),1);
                        gl.Uniform1ui(gl.GetUniformLocation(resources->program,"uSelectedRoughVolume"),resources->smoothGlass.object.value);
                        glDrawArrays(GL_TRIANGLES,0,3);
                        gl.Uniform1f(gl.GetUniformLocation(resources->program,"uSelectedRoughResolve"),0);
                        gl.Uniform1f(locations.skyBackground,0);
                        glDisable(GL_SCISSOR_TEST);glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);
                    }
                    if(!roughglasstiles::opengl::completeSubmission(gl,result.error))return fail(result.error);
                }
                NativeRawPassPixels objects, materials, normal, color;
                const auto readProduct = [&](unsigned attachment, NativeTexturePixelFormat format, NativeRawPassPixels& pixels) {
                    return roughglasstiles::opengl::readRegion(gl, framebuffer, attachment, tile.x, bottom,
                        tile.width, tile.height, format, pixels, result.error);
                };
                if (!readProduct(GL_COLOR_ATTACHMENT6, NativeTexturePixelFormat::R32Uint, objects)
                    || !readProduct(GL_COLOR_ATTACHMENT5, NativeTexturePixelFormat::R32Uint, materials)
                    || !readProduct(GL_COLOR_ATTACHMENT2, NativeTexturePixelFormat::Rgba16Float, normal)
                    || !readProduct(GL_COLOR_ATTACHMENT0, runtimeInputs.linearColor
                        ? NativeTexturePixelFormat::Rgba16Float : NativeTexturePixelFormat::Rgba8Unorm, color))
                    return fail(result.error);
                if (!ownerValid()) return fail("OpenGL rough tile GPU frame owner changed during readback");
                if(selected)for(std::size_t pixel=0;pixel<tile.pixels();++pixel) {
                    std::uint32_t objectId=0,materialId=0;
                    std::memcpy(&objectId,objects.bytes.data()+pixel*4,4);std::memcpy(&materialId,materials.bytes.data()+pixel*4,4);
                    if(objectId!=selectedIds[pixel][0] || materialId!=selectedIds[pixel][1])
                        return fail("OpenGL selected camera result differs from completed primary raster ownership");
                }
                const roughglasstiles::NativeTileReceipt receipt {true, &receipts[0], &receipts[1], &receipts[2],
                    &objects, &materials, &normal, &color,selected?&nativeWork:nullptr};
                if (!completion.accept(roughKey, tile, receipt, result.error)) return fail(result.error);
            }
            if (!ownerValid() || !completion.publish(roughKey, result.stats, frame->roughTileAggregate, result.error))
                return fail(result.error.empty() ? "OpenGL rough tiled frame lost its publication owner" : result.error);
            if(selected) {
#if !defined(__APPLE__)
                if(resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max()
                    ||!roughglasstiles::opengl::completeSubmission(gl,result.error))return fail(result.error.empty()?"OpenGL selected query contract fence exhausted":result.error);
                const auto serial=++resources->causticSubmissionSerial;
                try {
                const glasscaustics::SubmissionIdentity submission{"selected-camera/"+std::to_string(resources->rendererGeneration)+"/"+std::to_string(serial),
                    resources->rendererGeneration,serial,serial,serial,serial,serial};
                for(unsigned index=0;index<2;++index) {
                    const auto role=index?OpenGlQueryArithmeticRole::SelectedCameraTransport:OpenGlQueryArithmeticRole::SelectedCameraPrimary;
                    const auto evidence=completedQueryArithmeticEvidence(selectedCamera.queryOwners[index],role,submission,
                        index?std::array<std::shared_ptr<OpenGlQueryProgram>,2>{selectedCamera.queryOwners[0],{}}:std::array<std::shared_ptr<OpenGlQueryProgram>,2>{},
                        index?std::array<std::shared_ptr<const OpenGlQueryArithmeticEvidence>,2>{frame->queryEvidence[std::size_t(OpenGlQueryArithmeticRole::SelectedCameraPrimary)],{}}
                            :std::array<std::shared_ptr<const OpenGlQueryArithmeticEvidence>,2>{});
                    frame->retainQueryArithmetic(role,selectedCamera.queryOwners[index],evidence);
                }
                } catch(const std::bad_alloc&) {return fail("OpenGL selected query evidence allocation failed");}
#endif
            }
            frame->roughTileSchedule = roughSchedule;
            frame->roughReceiptComplete = true;
            gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glViewport(0, 0, int(width), int(height));
        }
        else
        { drawSurfaces(-1); }
        {
            loopstep::Scope drawFinishScope ("gl:fixture-draw-finish");
            glFinish();
        }
        const auto gpuError = glGetError();
        if (gpuError != GL_NO_ERROR)
            return fail ("OpenGL fixture draw reported GPU error "
                         + std::to_string (static_cast<unsigned> (gpuError)));
        if (authoredDiffraction && runtimeInputs.linearColor
            && !validateDiffractionDraw(frame->texture_)) return fail(result.error);
        if (resources->smoothGlass.enabled && !roughGlassFrame) {
            NativeRawPassPixels completion;
            if (!frame->readRawPassOwned(renderpassoutput::Output::Normal, completion, result.error)
                || !smoothglass::completeFrame(completion, result.error)) return fail(result.error);
            if (runtimeInputs.linearColor
                && (!frame->readRawPassOwned(renderpassoutput::Output::Color, completion, result.error)
                    || !smoothglass::finiteColorFrame(completion, result.error))) return fail(result.error);
        }
        cleanup();

        result.rendered = true;
        result.frame = std::move (frame);
        result.stats.drawCount = resources->instancedSharedGeometry ? instancedDrawCount
            : materialKind == NativeFixtureMaterialKind::DiffractionReflective
            ? materialProgram->diffractionPathCount + surfaceDrawCount
            : static_cast<std::size_t>(std::count_if(
                scene->objects.begin(), scene->objects.begin() + scene->objectCount,
                [] (const auto& drawObject) { return drawObject.indexCount != 0; }));
        if (roughGlassFrame) result.stats.drawCount = surfaceDrawCount + roughSchedule->tiles().size();
        result.stats.ordinaryDrawCount = resources->instancedSharedGeometry
            ? 0 : result.stats.drawCount;
        result.stats.instancedDrawCount = instancedDrawCount;
        result.stats.instanceBufferUploadCount = instanceBufferUploadCount;
        result.stats.submittedInstanceCount = submittedInstanceCount;
        result.stats.noteInstanceDrawCount = noteInstanceDrawCount;
        result.stats.noteInstanceTransformUploadCount = noteInstances.admitted ? 1u : 0u;
        result.stats.submittedNoteInstanceCount = submittedNoteInstanceCount;
        result.stats.objectInstanceDrawCount = objectInstanceDrawCount;
        result.stats.submittedObjectInstanceCount = submittedObjectInstanceCount;
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
    struct ContextPrograms {
        std::map<std::string,CachedProgram> programs;
        // Submitted scene programs that have not been finished yet, keyed by vertex source.
        std::map<std::string,OpenGlFixtureInFlightProgram> inFlight;
        // -1 until a Preview prepare asks. 1 when the parallel compile extension is usable.
        int parallelCompile = -1;
        std::shared_ptr<OpenGlQueryContext> queryContext;
        // Diagnostics for a Preview scene program that stays pending. Counts only, plus
        // the rate limit of the stderr line and whether a pending state was reported.
        std::uint64_t pendingPolls = 0;
        std::uint64_t submitCount = 0;
        std::uint64_t evictionCount = 0;
        bool pendingActive = false;
        std::chrono::steady_clock::time_point lastPendingLog {};
        // When each key last left inFlight, by finish or eviction. Read by the resubmit window.
        std::map<std::string,std::chrono::steady_clock::time_point> retiredAt;
    };
    std::map<GLFWwindow*,ContextPrograms> programCache_;
    std::uint64_t programUseSerial_ = 0;
};

NativeFixtureScenePreparation OpenGlFixtureSceneBackend::prepare (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram,
    NativeFixturePrepareUse use)
{
    return prepareGeometryInstancesImpl (scene, requestedMaterialProgram, {}, false, nullptr,
        nullptr, {}, 0, use);
}

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
uniform vec4 uCompositionPost[9];
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
    if (uCompositionPost[0].y == 1.0)
    {
        vec3 tangent = vec3(baseVertices[base+6u],baseVertices[base+7u],baseVertices[base+8u]);
        float handedness = baseVertices[base+9u];
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
            vec3 cofactor0 = cross(skin[1].xyz,skin[2].xyz);
            vec3 cofactor1 = cross(skin[2].xyz,skin[0].xyz);
            vec3 cofactor2 = cross(skin[0].xyz,skin[1].xyz);
            float skinDeterminant = dot(skin[0].xyz,cofactor0);
            normal = (cofactor0*normal.x+cofactor1*normal.y+cofactor2*normal.z)/skinDeterminant;
            tangent = mat3(skin) * tangent;
            handedness *= skinDeterminant < 0.0 ? -1.0 : 1.0;
        }
        mat4 point = mat4(uCompositionPost[1],uCompositionPost[2],uCompositionPost[3],uCompositionPost[4]);
        mat4 normalMatrix = mat4(uCompositionPost[5],uCompositionPost[6],uCompositionPost[7],uCompositionPost[8]);
        position = (point * vec4(position,1.0)).xyz;
        normal = mat3(normalMatrix) * normal;
        tangent = mat3(point) * tangent;
        float tangentLength = length(tangent);
        tangent = tangentLength > 1.0e-8 ? tangent/tangentLength : vec3(1.0,0.0,0.0);
        outputVertices[outputBase+6u] = tangent.x;
        outputVertices[outputBase+7u] = tangent.y;
        outputVertices[outputBase+8u] = tangent.z;
        outputVertices[outputBase+9u] = handedness * normalMatrix[3].w;
    }
    else
    {
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
    int compositionPostLocation = -1;
    std::vector<std::shared_ptr<const OpenGlDeformationResources>> draws;
    mutable std::mutex submissionMutex;
};

class OpenGlDeformationBackend final : public NativeDeformationBackend
{
public:
    bool supportsRetainedDeformationMotion() const noexcept override { return true; }
    bool supportsRetainedImportedObjectMotion() const noexcept override { return true; }
    bool supportsRetainedNoteMotion() const noexcept override { return true; }

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
        if (smoothglass::hasTransport(materialProgram.get())) {
            NativeDeformationPreparation refused;
            refused.error = "Closed glass transport refuses imported deformation";
            return refused;
        }
        if (!source || source->draws.empty()) return prepareDraw(source, materialProgram, {});
        NativeDeformationPreparation result;
        if (source->compositionOwner && !source->compositionOwner->matches(*source))
        { result.error = "native OpenGL composition batch has a foreign owner"; return result; }
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
            || (source->compositionOwner && !source->compositionOwner->matches(*source))
            || !HarmonicMIDI::grid::validateVisual3DScene(*source->scene).valid()
            || (!source->batchMember && (source->scene->objectCount != 1 || source->scene->materialCount != 1
            || source->scene->lightCount > 1 || source->scene->cameraCount != 1
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
        resources->fixture = fixture;
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
            resources->ownerContext = nullptr;
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
        const auto fullVertexBytes = source->scene->vertexCount * sizeof(HarmonicMIDI::grid::SceneVertex);
        const std::uint64_t retained = std::uint64_t(fullVertexBytes)*3
            + jointIndices.size()*sizeof(std::uint32_t) + jointWeights.size()*sizeof(float)
            + (morphPositions.size()+morphNormals.size())*sizeof(float)
            + resources->fixture->retainedTextureBytes;
        if (fullVertexBytes > 64u*1024u*1024u || retained > renderpassoutput::kMaximumTotalBytes)
            return fail("OpenGL preceding deformation buffer exceeds resource capacity");
        const auto previousBufferBytes = resources->fixture->previousVertexBuffer == 0 ? fullVertexBytes : 0;
        if (previousBufferBytes != 0)
        {
            gl.GenBuffers(1,&resources->fixture->previousVertexBuffer);
            gl.BindBuffer(GL_ARRAY_BUFFER,resources->fixture->previousVertexBuffer);
            gl.BufferData(GL_ARRAY_BUFFER,static_cast<std::ptrdiff_t>(fullVertexBytes),
                          source->scene->vertices.data(),GL_DYNAMIC_DRAW);
            resources->fixture->previousVertexBufferCharge=residency::ResidencyCharge(
                residency::BackingClass::fixtureScene,residency::nominalBufferBytes(fullVertexBytes));
            gl.BindVertexArray(resources->fixture->vertexArray);
            gl.VertexAttribPointer(12,3,GL_FLOAT,GL_FALSE,sizeof(HarmonicMIDI::grid::SceneVertex),
                reinterpret_cast<const void*>(offsetof(HarmonicMIDI::grid::SceneVertex,position)));
            gl.EnableVertexAttribArray(12);
            if (!resources->fixture->previousVertexBuffer || glGetError()!=GL_NO_ERROR)
                return fail("OpenGL preceding deformation buffer allocation failed");
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

        {
            // Compute compile and link only; the lease below is unchanged.
            loopstep::Scope computeCompileScope ("gl:fixture-deformation-compute-compile");
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
        }
        resources->vertexCountLocation = gl.GetUniformLocation (
            resources->computeProgram, "uVertexCount");
        resources->jointCountLocation = gl.GetUniformLocation (
            resources->computeProgram, "uJointCount");
        resources->morphTargetCountLocation = gl.GetUniformLocation (
            resources->computeProgram, "uMorphTargetCount");
        resources->outputOffsetLocation = gl.GetUniformLocation(resources->computeProgram, "uOutputOffset");
        resources->compositionPostLocation = gl.GetUniformLocation(resources->computeProgram, "uCompositionPost[0]");
        if (resources->computeProgram == 0 || resources->vertexCountLocation < 0
            || resources->jointCountLocation < 0
            || resources->morphTargetCountLocation < 0
            || resources->outputOffsetLocation < 0
            || resources->compositionPostLocation < 0
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
        result.stats.staticVertexBytes = vertexBytes * 2u + previousBufferBytes;
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
        // Checked worlds reach a deformed draw only for one rigid animated leaf, which
        // the executor admits. Scene Compose carries its own frame and is still refused.
        if (runtimeInputs.objectWorlds
            && (source == nullptr || runtimeInputs.compositionFrame || source->compositionOwner
                || !validNativeImportedObjectWorlds(*runtimeInputs.objectWorlds)))
        {
            result.error = "OpenGL static local object worlds require one rigid imported deformation leaf";
            return result;
        }
        const auto resources = std::dynamic_pointer_cast<const OpenGlDeformationResources> (
            nativeResources);
        if (source == nullptr || snapshot == nullptr || resources == nullptr
            || resources->source != source || resources->ownerContext != glfwGetCurrentContext())
        {
            result.error = "native OpenGL deformation requires exact prepared owner resources";
            return result;
        }
        if (runtimeInputs.previousMotion && !validFixtureRuntimeInputs(runtimeInputs))
        { result.error = "OpenGL preceding deformation inputs are invalid"; return result; }
        std::lock_guard<std::mutex> submissionLock (resources->submissionMutex);
        NativeDeformationFrameData frameData;
        if (! prepareNativeDeformationFrame (*source, *snapshot, runtimeInputs.compositionFrame, frameData, result.error))
            return result;
        std::optional<NativeDeformationFrameData> precedingData;
        if (runtimeInputs.previousMotion)
        {
            if (!runtimeInputs.previousMotion->deformation)
            {
                if (runtimeInputs.previousMotion->importedNodeWorldMatrices)
                { result.error = "OpenGL imported Motion requires the retained deformation snapshot"; return result; }
                runtimeInputs.previousMotion.reset();
            }
            else
            {
                NativeDeformationFrameData prior;
                NativeSceneMotionSample::ImportedNodeWorldMatrices sampledWorlds;
                const auto& retainedWorlds = runtimeInputs.previousMotion->importedNodeWorldMatrices;
                if (!prepareNativeDeformationFrame(*source,*runtimeInputs.previousMotion->deformation,
                        runtimeInputs.previousMotion->compositionFrame,prior,result.error,retainedWorlds ? &sampledWorlds : nullptr)) return result;
                if (retainedWorlds)
                    for (const auto& retained : *retainedWorlds)
                    {
                        const auto sampled = std::find_if(sampledWorlds.begin(), sampledWorlds.end(),
                            [&](const auto& entry) { return entry.nodeStableId == retained.nodeStableId; });
                        // The imported owner checks complete drawn ancestry and static
                        // worlds. Authored skin joints outside that ancestry are not retained.
                        if (sampled != sampledWorlds.end() && retained.world != sampled->world)
                        { result.error = "OpenGL imported Motion worlds disagree with the retained deformation"; return result; }
                    }
                const auto currentTime = snapshot->time();
                const auto previousTime = runtimeInputs.previousMotion->deformation->time();
                if (currentTime.frame == std::numeric_limits<std::int64_t>::min()
                    || previousTime.frame != currentTime.frame-1
                    || previousTime.rateNumerator != currentTime.rateNumerator
                    || previousTime.rateDenominator != currentTime.rateDenominator)
                    runtimeInputs.previousMotion.reset();
                else
                {
                    const auto& priorSnapshot = *runtimeInputs.previousMotion->deformation;
                    const auto gain = source->batchMember && priorSnapshot.pose().morphEnabled
                        && priorSnapshot.pose().nodeStableId != source->animationNodeStableId
                        && priorSnapshot.combinationMode() != visualanimation::CombinationMode::WeightedBlend
                        ? static_cast<float>(priorSnapshot.combinationWeight()) : runtimeInputs.previousMotion->morphWeight;
                    for (std::size_t index=0;index<prior.morphTargetCount;++index)
                    {
                        const auto base = source->morphBaseWeights.empty() ? 0.0f : source->morphBaseWeights[index];
                        prior.morphWeights[index] = base+(prior.morphWeights[index]-base)*gain;
                    }
                    precedingData = prior;
                }
            }
        }
        if (!source->batchMember) resources->fixture->usePreviousDeformation = false;
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
                if (deformationMotionProbeEnabled()) deformationMotionProbeMember = index;
                auto draw = render(source->draws[index], snapshot, resources->draws[index], width, height, drawInputs);
                if (!draw.rendered) { result.error = draw.error; return result; }
                if (source->retainDeformedGeometry) {
                    const auto& member=source->draws[index];
                    const auto object=std::find_if(source->scene->objects.begin(),
                        source->scene->objects.begin()+source->scene->objectCount,
                        [&](const auto& value) { return member && value.id==member->object; });
                    const auto* mesh=member && member->deformation ? member->deformation->findMesh(member->mesh) : nullptr;
                    const auto objectEnd=source->scene->objects.begin()+source->scene->objectCount;
                    const auto number=[](unsigned long long value) { return std::to_string(value); };
                    const std::string topology="native OpenGL scene geometry readback changed topology";
                    // Same conditions as before, checked in the same order, each with its own message.
                    if (!mesh) {
                        result.error=topology+" (mesh "+(member ? number(member->mesh.value) : std::string("none"))+" not found)"; return result;
                    }
                    if (object==objectEnd) {
                        result.error=topology+" (object "+(member ? number(member->object.value) : std::string("none"))+" not found)"; return result;
                    }
                    if (mesh->vertexCount()!=object->vertexCount) {
                        result.error=topology+" (mesh vertices "+number(mesh->vertexCount())+" vs object "+number(object->vertexCount)+")"; return result;
                    }
                    if (object->firstVertex>source->scene->vertexCount) {
                        result.error=topology+" (object first vertex "+number(object->firstVertex)+" beyond scene vertices "+number(source->scene->vertexCount)+")"; return result;
                    }
                    if (object->vertexCount>source->scene->vertexCount-object->firstVertex) {
                        result.error=topology+" (object vertices "+number(object->vertexCount)+" overrun scene vertices "+number(source->scene->vertexCount)+" from first vertex "+number(object->firstVertex)+")"; return result;
                    }
                    if (draw.deformedVertices.size()!=mesh->vertexCount()*8u) {
                        result.error=topology+" (deformed floats "+number(draw.deformedVertices.size())+" vs expected "+number(mesh->vertexCount()*8u)+")"; return result;
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
            const auto object = std::find_if(source->scene->objects.begin(),
                source->scene->objects.begin() + source->scene->objectCount,
                [&](const auto& value) { return value.id == source->object; });
            const auto dispatch = [&](const NativeDeformationFrameData& data,unsigned outputBuffer)
            {
                gl.BindBuffer (GL_SHADER_STORAGE_BUFFER, resources->buffers[5]);
                gl.BufferData (GL_SHADER_STORAGE_BUFFER,
                               static_cast<std::ptrdiff_t> (data.jointPalette.size()
                                                           * sizeof (float)),
                               data.jointPalette.data(), GL_DYNAMIC_DRAW);
                gl.BindBuffer (GL_SHADER_STORAGE_BUFFER, resources->buffers[6]);
                gl.BufferData (GL_SHADER_STORAGE_BUFFER,
                               static_cast<std::ptrdiff_t> (data.morphWeights.size()
                                                           * sizeof (float)),
                               data.morphWeights.data(), GL_DYNAMIC_DRAW);
                gl.BindBufferBase (GL_SHADER_STORAGE_BUFFER, 0, resources->buffers[0]);
                gl.BindBufferBase (GL_SHADER_STORAGE_BUFFER, 1,
                                   outputBuffer);
                for (unsigned binding = 2; binding < 8; ++binding)
                    gl.BindBufferBase (GL_SHADER_STORAGE_BUFFER, binding,
                                       resources->buffers[binding - 1u]);
                gl.UseProgram (resources->computeProgram);
                gl.Uniform1ui (resources->vertexCountLocation, data.vertexCount);
                gl.Uniform1ui (resources->jointCountLocation, data.jointCount);
                gl.Uniform1ui (resources->morphTargetCountLocation,
                               data.morphTargetCount);
                gl.Uniform1ui(resources->outputOffsetLocation, object->firstVertex);
                std::array<float,36> postWords;
                std::copy(data.compositionPost.control().begin(),data.compositionPost.control().end(),postWords.begin());
                std::copy(data.compositionPost.point().begin(),data.compositionPost.point().end(),postWords.begin()+4);
                std::copy(data.compositionPost.normal().begin(),data.compositionPost.normal().end(),postWords.begin()+20);
                gl.Uniform4fv(resources->compositionPostLocation,9,postWords.data());
                gl.DispatchCompute ((data.vertexCount + 63u) / 64u, 1, 1);
            };
            resources->fixture->deformedWindings[object->firstVertex] = deformedMemberWinding (frameData);
            dispatch(frameData,resources->fixture->vertexBuffer);
            if (precedingData)
            {
                dispatch(*precedingData,resources->fixture->previousVertexBuffer);
                resources->fixture->usePreviousDeformation = true;
            }
            gl.MemoryBarrier (GL_SHADER_STORAGE_BARRIER_BIT
                              | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
            glFinish();
            if (deformationMotionProbeEnabled())
            {
                // Reads one member range back. The loader has no glGetBufferSubData entry, so this maps
                // the range the same way the retained readback below does. The restore lambda rebinds the target.
                const auto readMemberRange = [&](unsigned buffer, std::vector<HarmonicMIDI::grid::SceneVertex>& out)
                {
                    gl.BindBuffer (GL_SHADER_STORAGE_BUFFER, buffer);
                    const auto* mapped = static_cast<const HarmonicMIDI::grid::SceneVertex*> (gl.MapBufferRange (
                        GL_SHADER_STORAGE_BUFFER,
                        static_cast<std::ptrdiff_t> (object->firstVertex * sizeof (HarmonicMIDI::grid::SceneVertex)),
                        static_cast<std::ptrdiff_t> (frameData.vertexCount * sizeof (HarmonicMIDI::grid::SceneVertex)),
                        GL_MAP_READ_BIT));
                    if (mapped == nullptr) return false;
                    out.assign (mapped, mapped + frameData.vertexCount);
                    return gl.UnmapBuffer (GL_SHADER_STORAGE_BUFFER) == GL_TRUE;
                };
                const auto precedingText = precedingData && runtimeInputs.previousMotion && runtimeInputs.previousMotion->deformation
                    ? std::to_string (static_cast<long long> (runtimeInputs.previousMotion->deformation->time().frame))
                    : std::string ("none");
                const auto memberText = source->batchMember
                    ? std::to_string (deformationMotionProbeMember) : std::string ("none");
                std::string paletteText = "none", bufferText = "none";
                if (precedingData)
                {
                    float paletteDiff = 0.0f;
                    for (std::size_t index = 0; index < frameData.jointPalette.size(); ++index)
                    {
                        const float diff = std::fabs (frameData.jointPalette[index] - precedingData->jointPalette[index]);
                        if (std::isnan (diff) || diff > paletteDiff) paletteDiff = diff;
                    }
                    paletteText = deformationMotionProbeFloats (&paletteDiff, 1);
                    std::vector<HarmonicMIDI::grid::SceneVertex> liveMember, precedingMember;
                    const bool inRange = static_cast<std::uint64_t> (object->firstVertex) + frameData.vertexCount
                        <= source->scene->vertexCount;
                    if (inRange && readMemberRange (resources->fixture->vertexBuffer, liveMember)
                        && readMemberRange (resources->fixture->previousVertexBuffer, precedingMember))
                    {
                        float bufferDiff = 0.0f;
                        for (std::size_t index = 0; index < liveMember.size(); ++index)
                        {
                            const auto& live = liveMember[index].position;
                            const auto& preceding = precedingMember[index].position;
                            const float diffs[3] = { std::fabs (live.x - preceding.x), std::fabs (live.y - preceding.y),
                                std::fabs (live.z - preceding.z) };
                            for (const float diff : diffs)
                                if (std::isnan (diff) || diff > bufferDiff) bufferDiff = diff;
                        }
                        bufferText = deformationMotionProbeFloats (&bufferDiff, 1);
                    }
                    else bufferText = "unreadable";
                }
                // usePrevious is the member's flag at this point. A batch fixture render reads the flag
                // after the whole member loop, and [scene-motion] prints the value it uploads.
                std::fprintf (stderr,
                    "[deformation-motion] frame=%lld prev=%s member=%s source=%llu deformation=%llu clip=%llu object=%llu node=%llu usePrevious=%d vertices=%u paletteMaxDiff=%s bufferMaxDiff=%s\n",
                    static_cast<long long> (snapshot->time().frame), precedingText.c_str(), memberText.c_str(),
                    static_cast<unsigned long long> (source->sourceStableId),
                    static_cast<unsigned long long> (source->deformationStableId),
                    static_cast<unsigned long long> (source->clip.value),
                    static_cast<unsigned long long> (source->object.value),
                    static_cast<unsigned long long> (source->animationNodeStableId),
                    resources->fixture->usePreviousDeformation ? 1 : 0,
                    static_cast<unsigned> (frameData.vertexCount), paletteText.c_str(), bufferText.c_str());
            }
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
            result.stats.dispatchCount = precedingData ? 2 : 1;
            result.stats.dynamicUniformBytes = (frameData.jointCount * 16u * sizeof(float)
                + frameData.morphTargetCount * sizeof(float) + sizeof(NativeDeformationComputePostBlock)) * (precedingData ? 2u : 1u);
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
        result.stats.dispatchCount = precedingData ? 2 : 1;
        result.stats.reusedStaticResources = true;
        result.stats.dynamicUniformBytes =
            (static_cast<std::uint64_t> (frameData.jointCount) * 16u * sizeof (float)
            + static_cast<std::uint64_t> (frameData.morphTargetCount) * sizeof (float)
            + sizeof(NativeDeformationComputePostBlock))
            * (precedingData ? 2u : 1u);
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

std::string openGlSelectedRoughCameraShaderForFixture(bool transport)
{
    return RoughSelectedCamera::shader(transport);
}

namespace
{
bool queryArithmeticEvidenceMatches(const OpenGlFixtureSceneFrame* frame,
    const std::shared_ptr<const OpenGlQueryArithmeticEvidence>& evidence,
    const glasscaustics::SubmissionIdentity& expected,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    if(frame&&!frame->reconcileNativeAccounting(error))return false;
    if(!frame||!frame->staticResources||!evidence||!evidence->program||!(evidence->submission==expected)
        ||!expected.submissionSerial||expected.completedFence!=expected.submissionSerial
        ||expected.rendererGeneration!=frame->rendererGeneration_||expected.frameIdentity.empty()
        ||std::size_t(evidence->role)>=frame->queryPrograms.size())return refuse("OpenGL query contract frame/submission is invalid");
    const auto findOwner=[&](const std::shared_ptr<OpenGlQueryProgram>& program,
        const std::shared_ptr<const OpenGlQueryArithmeticEvidence>& record,unsigned depth,const auto& self)->std::shared_ptr<OpenGlQueryProgram> {
        if(!program||!record||depth>2)return {};
        if(record==evidence&&program->identity==evidence->program)return program;
        for(unsigned parent=0;parent<2;++parent)if(const auto found=self(program->parents[parent],record->parents[parent],depth+1,self))return found;
        return {};
    };
    std::shared_ptr<OpenGlQueryProgram> owner;
    for(std::size_t index=0;index<frame->queryPrograms.size()&&!owner;++index)
        owner=findOwner(frame->queryPrograms[index],frame->queryEvidence[index],0,findOwner);
    for(auto* cohort=frame->directGratingCohort.get();cohort&&!owner;cohort=cohort->previous.get())
        for(const auto& source:cohort->sources)if(source->batch&&!owner)
            owner=findOwner(source->batch->program,source->batch->evidence,0,findOwner);
    if(!owner||!owner->preciseSceneQuery||owner->identity!=evidence->program||!owner->context
        ||owner->context!=frame->staticResources->queryContext||!owner->context->active
        ||owner->context->context!=glfwGetCurrentContext()||frame->ownerContext!=glfwGetCurrentContext()
        ||frame->rendererGeneration_!=frame->staticResources->rendererGeneration)
        return refuse("OpenGL query contract producer lease or current context expired");
    const auto& identity=*owner->identity;
    if(identity.contractVersion!=1||identity.glslVersion!=430||identity.context!=reinterpret_cast<std::uintptr_t>(frame->ownerContext)
        ||identity.contextGeneration!=owner->context->generation||!identity.contextGeneration||!identity.ownerGeneration
        ||identity.contextMajor!=std::uint32_t(owner->context->major)||identity.contextMinor!=std::uint32_t(owner->context->minor)
        ||identity.contextProfile!=std::uint32_t(owner->context->profile)||identity.rendererGeneration!=frame->rendererGeneration_
        ||!owner->program||identity.linkedProgram!=owner->program||!identity.stageCount||identity.stageCount>identity.stages.size())
        return refuse("OpenGL query contract linked profile/generation changed");
    for(unsigned stage=0;stage<identity.stageCount;++stage) {
        const auto& source=identity.stages[stage];
        if(source.source.rfind("#version 430 core\n",0)!=0||source.fingerprint!=querySourceFingerprint(source.source))
            return refuse("OpenGL query contract retained assembled stage changed");
    }
#if defined(__APPLE__)
    return refuse("OpenGL query contract requires native GL4.3 ownership");
#else
    const auto isProgram=reinterpret_cast<PFNGLISPROGRAMPROC>(glfwGetProcAddress("glIsProgram"));
    const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
    if(!isProgram||!getLabel||!isProgram(owner->program))return refuse("OpenGL query contract native linked owner was deleted");
    int linked=0,deleted=0;owner->gl.GetProgramiv(owner->program,GL_LINK_STATUS,&linked);owner->gl.GetProgramiv(owner->program,GL_DELETE_STATUS,&deleted);
    std::array<char,96> label{};int length=0;getLabel(GL_PROGRAM,owner->program,int(label.size()),&length,label.data());
    if(linked!=GL_TRUE||deleted!=GL_FALSE||label!=owner->label||glGetError()!=GL_NO_ERROR)
        return refuse("OpenGL query contract native program label/link/lifetime changed");
    for(const auto& parent:evidence->parents)if(parent
        &&!queryArithmeticEvidenceMatches(frame,parent,parent->submission,error))return false;
    error.clear();return true;
#endif
}
}
bool OpenGlFixtureSceneFrame::validateDirectProductionQuery(const glasscaustics::SubmissionIdentity& expected,
    const glasscaustics::NativeDirectGratingContributionReadback& supplied,std::string& error) const
{
    if(!directGratingBatch||!gratingStagingBinding){error="OpenGL direct contribution has no retained batch";return false;}
    return validateDirectBatch(*gratingStagingBinding,*directGratingBatch,directionalSourceUpload,expected,supplied,false,error);
}
const std::string& OpenGlDirectGratingSource::backend() const noexcept {return frame->backend();}
std::uint32_t OpenGlDirectGratingSource::width() const noexcept {return frame->width();}
std::uint32_t OpenGlDirectGratingSource::height() const noexcept {return frame->height();}
std::uintptr_t OpenGlDirectGratingSource::nativeResourceCacheIdentity() const noexcept {return frame->nativeResourceCacheIdentity();}
bool OpenGlDirectGratingSource::validateDirectProductionQuery(const glasscaustics::SubmissionIdentity& expected,
    const glasscaustics::NativeDirectGratingContributionReadback& supplied,std::string& error) const
{
    if(!frame||frame->directGratingCohort.get()!=cohort||!binding||!batch
        ||std::none_of(cohort->sources.begin(),cohort->sources.end(),[&](const auto& source){return source.get()==this;}))
    {error="OpenGL directional cohort source view is no longer current";return false;}
    return frame->validateDirectBatch(*binding,*batch,sourceUpload,expected,supplied,true,error);
}
bool OpenGlFixtureSceneFrame::validateDirectBatch(const glasscaustics::GratingIncidentBinding& binding,
    const OpenGlDirectGratingBatch& retained,const glasscaustics::NativeDirectionalSourceUpload& sourceUpload,
    const glasscaustics::SubmissionIdentity& expected,const glasscaustics::NativeDirectGratingContributionReadback& supplied,
    bool cohort,std::string& error) const
{
#if defined(__APPLE__)
    (void)binding;(void)retained;(void)sourceUpload;(void)cohort;(void)expected;(void)supplied;error="OpenGL direct contribution requires native GL4.3 ownership";return false;
#else
    namespace gc=glasscaustics;
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto* batch=&retained;
    if(!batch||!batch->program||!batch->evidence||!staticResources||!gratingStagingBinding
        ||!privateCameraAttachments||gratingCameraBaseComplete||!gratingCameraRuntime
        ||gratingCameraEvaluationTime!=binding.evaluationTime
        ||ownerContext!=glfwGetCurrentContext()||staticResources->ownerContext!=ownerContext
        ||batch->sourceBuffer!=(cohort?batch->program->directionalInputBuffer:directionalSourceBuffer)||batch->queryTexture!=causticTextures[0]
        ||batch->queryControl!=binding.frame.query.control
        ||batch->evidence->role!=OpenGlQueryArithmeticRole::GratingDirect
        ||(!cohort&&(batch->evidence!=queryEvidence[std::size_t(OpenGlQueryArithmeticRole::GratingDirect)]
            ||batch->program!=queryPrograms[std::size_t(OpenGlQueryArithmeticRole::GratingDirect)]))
        ||!(batch->upload.submission==expected)||!batch->upload.complete
        ||batch->upload.generation!=expected.submissionSerial
        ||expected.rasterGeneration!=gratingStagingSubmission.rasterGeneration
        ||expected.queryGeneration!=gratingStagingSubmission.queryGeneration
        ||!queryArithmeticEvidenceMatches(this,batch->evidence,expected,error))
        return refuse("OpenGL direct contribution lost its completed native dispatch owner");
    const auto& gl=staticResources->gl;
    const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
    const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
    const auto syncState=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
    const auto bufferSize=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    const auto uniformFloat=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
    const auto uniformInt=reinterpret_cast<PFNGLGETUNIFORMIVPROC>(glfwGetProcAddress("glGetUniformiv"));
    const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
    if(!isBuffer||!isSync||!syncState||!bufferSize||!uniformFloat||!uniformInt||!getLabel
        ||!batch->program->directFence||!isSync(batch->program->directFence))
        return refuse("OpenGL direct contribution has no retained native fence or buffer inspection");
    int fence=0;syncState(batch->program->directFence,GL_SYNC_STATUS,1,nullptr,&fence);
    if(fence!=GL_SIGNALED)return refuse("OpenGL direct contribution fence is incomplete");
    const auto location=[&](const char* name){return gl.GetUniformLocation(batch->program->program,name);};
    const auto matchesFloat=[&](const char* name,const float* values,std::size_t count) {
        const int at=location(name);std::array<float,8> actual{};
        if(at<0||count>actual.size())return false;
        // Array elements have independent locations and are queried separately.
        uniformFloat(batch->program->program,at,actual.data());
        return std::memcmp(actual.data(),values,count*sizeof(float))==0;
    };
    const auto matchesInt=[&](const char* name,int value) {
        int actual=0;const int at=location(name);if(at<0)return false;
        uniformInt(batch->program->program,at,&actual);return actual==value;
    };
    if(!matchesInt("uEndpointCount",int(batch->upload.inputs.size()))||!matchesInt("uGratingCount",int(binding.triangles.size()))
        ||!matchesInt("uOpaqueVisibility",0)||!matchesFloat("uOpaqueControl",batch->queryControl.data(),4)
        ||!matchesFloat("uSource",binding.frame.sourceDirectionTrace.data(),4)
        ||!matchesFloat("uLi[0]",binding.frame.sourceRadiance[0].data(),4)
        ||!matchesFloat("uLi[1]",binding.frame.sourceRadiance[1].data(),4))
        return refuse("OpenGL direct contribution native query/source uniforms changed");
    int generic=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&generic);
    int token=0;auto restore=[&](int*){gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(generic));};
    std::unique_ptr<int,decltype(restore)> guard(&token,restore);
    const auto matchesBuffer=[&](unsigned buffer,const void* bytes,std::size_t size) {
        if(!buffer||!isBuffer(buffer))return false;
        if(cohort||buffer!=directionalSourceBuffer) {
            std::array<char,96> actual{};int length=0;
            getLabel(GL_BUFFER,buffer,int(actual.size()),&length,actual.data());
            if(actual!=batch->program->label)return false;
        }
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffer);GLint64 actual=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
        if(actual!=GLint64(size))return false;
        const auto* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(size),GL_MAP_READ_BIT);
        if(!mapped)return false;
        const bool same=std::memcmp(mapped,bytes,size)==0;
        return gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)==GL_TRUE&&same;
    };
    if(!matchesBuffer(batch->program->directBuffers[0],batch->upload.inputs.data(),batch->upload.inputs.size()*sizeof(gc::NativeDirectGratingEndpointInput))
        ||!matchesBuffer(batch->program->directBuffers[1],batch->packed.data(),batch->packed.size()*sizeof(gc::opengl::DirectGratingPacked))
        ||!matchesBuffer(batch->program->directBuffers[2],binding.witnessTexels.data(),binding.witnessTexels.size()*sizeof(gc::Texel))
        ||!matchesBuffer(batch->sourceBuffer,&sourceUpload.input,sizeof(gc::NativeDirectionalSourceInput)))
        return refuse("OpenGL direct contribution actual endpoint/source/result buffers changed");
    if(!glIsTexture(batch->queryTexture))return refuse("OpenGL direct contribution production query texture expired");
    int texture=0,packBuffer=0;glGetIntegerv(GL_TEXTURE_BINDING_2D,&texture);glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING,&packBuffer);
    constexpr std::array<GLenum,6> packNames{GL_PACK_ALIGNMENT,GL_PACK_ROW_LENGTH,GL_PACK_SKIP_PIXELS,GL_PACK_SKIP_ROWS,GL_PACK_SWAP_BYTES,GL_PACK_LSB_FIRST};
    std::array<int,6> pack{};for(std::size_t at=0;at<pack.size();++at)glGetIntegerv(packNames[at],&pack[at]);
    int textureToken=0;auto restoreTexture=[&](int*) {
        glBindTexture(GL_TEXTURE_2D,unsigned(texture));gl.BindBuffer(GL_PIXEL_PACK_BUFFER,unsigned(packBuffer));
        for(std::size_t at=0;at<pack.size();++at)glPixelStorei(packNames[at],pack[at]);
    };std::unique_ptr<int,decltype(restoreTexture)> textureGuard(&textureToken,restoreTexture);
    glBindTexture(GL_TEXTURE_2D,batch->queryTexture);int width=0,height=0,format=0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&width);glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&height);
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&format);
    if(width!=int(opaquevisibility::kTextureWidth)||height!=int(binding.frame.query.height)||format!=GL_RGBA32F)
        return refuse("OpenGL direct contribution production query allocation changed");
    std::vector<gc::Texel> query(binding.frame.query.texels.size());
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER,0);for(std::size_t at=0;at<pack.size();++at)glPixelStorei(packNames[at],at?0:1);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,query.data());
    if(query.capacity()>2*query.size()||query!=binding.frame.query.texels||glGetError()!=GL_NO_ERROR)
        return refuse("OpenGL direct contribution production query bytes changed");
    std::uint64_t bytes=causticDiagnostics.admittedFrameBytes,
        work=cohort?gratingStagingBinding->bounds.combinedWork:binding.bounds.combinedWork;
    for(auto owned:{directGratingBytes,cameraGlassTraceBytes,queryRetainedBytes,nativeAttemptBytes,gratingCameraSpectrumBytes,gratingCellSourceBytes,gratingCoverageBytes,
        staticResources->queryContext->deferredBytes.load(std::memory_order_relaxed)})
        if(!renderpassoutput::detail::checkedAdd(bytes,owned,bytes))return refuse("OpenGL direct contribution aggregate storage overflow");
    for(auto reserved:{directGratingWork,cameraGlassTraceWork,gratingNativeWork,gratingCameraWork,gratingCellWork,gratingCoverageWork})
        if(!renderpassoutput::detail::checkedAdd(work,reserved,work))return refuse("OpenGL direct contribution aggregate work overflow");
    std::uint64_t deposition=0;
    if(bytes>renderpassoutput::kMaximumTotalBytes||work>gc::kMaximumQueryWork||!directDepositionBudget(deposition)
        ||!batch->nativeBytes||batch->nativeBytes>batch->ownedBytes)
        return refuse("OpenGL direct contribution aggregate admission changed");
    // This retained result was decoded from the completed native buffer whose
    // exact packed bytes were checked above. Caller edits cannot replace it.
    if(!gc::sameDirectGratingContributionReadback(supplied,batch->readback))
        return refuse("OpenGL direct contribution differs from its actual retained native result");
    error.clear();return true;
#endif
}
bool OpenGlFixtureSceneFrame::validateDirectCohortProductionQueries(const glasscaustics::SubmissionIdentity& expected,
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>& supplied,std::string& error) const
{
#if defined(__APPLE__)
    (void)expected;(void)supplied;error="OpenGL directional cohort requires native GL4.3 ownership";return false;
#else
    namespace gc=glasscaustics;const auto* cohort=directGratingCohort.get();
    if(!reconcileNativeAccounting(error))return false;
    const auto refuse=[&](const char* message){error=message;return false;};
    if(!cohort||!cohort->binding||!(cohort->submission==expected)||!staticResources
        ||!staticResources->queryContext||!staticResources->queryContext->active
        ||ownerContext!=glfwGetCurrentContext()||staticResources->ownerContext!=ownerContext
        ||rendererGeneration_!=staticResources->rendererGeneration||!privateCameraAttachments||gratingCameraBaseComplete
        ||!gratingCameraRuntime||!gratingStagingBinding||supplied.size()!=cohort->sources.size()
        ||cohort->sources.empty()||cohort->sources.size()!=cohort->binding->sources().size()
        ||!cohort->sources.back()->batch||cohort->chargedWork!=cohort->reservedWork||!cohort->chargedWork
        ||directGratingBytes<cohort->ownedBytes||directGratingWork<cohort->chargedWork
        ||expected.rendererGeneration!=rendererGeneration_
        ||expected.rasterGeneration!=gratingStagingSubmission.rasterGeneration
        ||expected.queryGeneration!=gratingStagingSubmission.queryGeneration
        ||expected.submissionSerial!=cohort->sources.back()->batch->readback.submission.submissionSerial
        ||expected.completedFence!=expected.submissionSerial)
        return refuse("OpenGL directional cohort lost its common completed frame or cumulative reservation");
    int texture=0;glGetIntegerv(GL_TEXTURE_BINDING_2D,&texture);
    int textureToken=0;auto restoreTexture=[&](int*){glBindTexture(GL_TEXTURE_2D,unsigned(texture));};
    std::unique_ptr<int,decltype(restoreTexture)> textureGuard(&textureToken,restoreTexture);
    if(!fixtureCameraAttachmentsMatch(*this,error))return false;
    for(unsigned slot:{1u,2u,3u}) {
        if(!causticTextures[slot]||!glIsTexture(causticTextures[slot]))return refuse("OpenGL directional cohort common raster expired");
        glBindTexture(GL_TEXTURE_2D,causticTextures[slot]);int width=0,height=0,format=0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&width);glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&height);
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&format);
        if(width!=int(width_)||height!=int(height_)||format!=(slot==3?GL_DEPTH_COMPONENT32F:GL_RGBA32F))
            return refuse("OpenGL directional cohort common raster allocation changed");
    }
    std::string common,retained;
    if(!gc::directGratingCommonIdentity(*gratingStagingBinding,common,error))return false;
    for(std::size_t i=0;i<cohort->sources.size();++i) {
        const auto& source=*cohort->sources[i];
        if(source.frame!=this||source.cohort!=cohort||source.binding!=cohort->binding->sources()[i]||!source.batch||!source.batch->program
            ||source.batch->queryTexture!=causticTextures[0]
            ||source.batch->program->context!=staticResources->queryContext
            ||!gc::directGratingCommonIdentity(*source.binding,retained,error)||retained!=common
            ||!source.validateDirectProductionQuery(supplied[i].submission,supplied[i],error))
            return refuse("OpenGL directional cohort source is outside its exact common native frame");
        for(std::size_t j=0;j<i;++j) {
            const auto& previous=*cohort->sources[j];
            if(source.batch->sourceBuffer==previous.batch->sourceBuffer||source.batch->program==previous.batch->program
                ||source.batch->program->directBuffers[0]==previous.batch->program->directBuffers[0]
                ||source.batch->program->directBuffers[1]==previous.batch->program->directBuffers[1]
                ||source.batch->program->directFence==previous.batch->program->directFence)
                return refuse("OpenGL directional cohort reused another source input, result, program or fence");
        }
    }
    // Inspect the actual common world-triangle staging buffer as well as the
    // production query texture checked independently by every source callback.
    const auto& gl=staticResources->gl;
    const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
    const auto sizeOf=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    if(!isBuffer||!sizeOf||!causticBuffers[4]||!isBuffer(causticBuffers[4]))
        return refuse("OpenGL directional cohort common geometry storage expired");
    int previous=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&previous);
    int token=0;auto restore=[&](int*){gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(previous));};
    std::unique_ptr<int,decltype(restore)> guard(&token,restore);
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,causticBuffers[4]);GLint64 actual=0;sizeOf(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
    const auto& triangles=gratingStagingBinding->frame.triangleTexels;const auto bytes=triangles.size()*sizeof(gc::Texel);
    if(actual!=GLint64(bytes))return refuse("OpenGL directional cohort common geometry allocation changed");
    const auto* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes),GL_MAP_READ_BIT);
    if(!mapped)return refuse("OpenGL directional cohort common geometry mapping failed");
    const bool same=std::memcmp(mapped,triangles.data(),bytes)==0;
    if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!same||glGetError()!=GL_NO_ERROR)
        return refuse("OpenGL directional cohort common geometry bytes changed");
    error.clear();return true;
#endif
}
namespace
{
namespace terminalgc=glasscaustics;
bool openGlCameraCoverageCosts(const OpenGlFixtureSceneFrame& frame,const OpenGlNativeAttempt& attempt,
    std::array<std::uint64_t,8>& output,std::string& error)
{
    namespace cc=cameraraycoefficients;namespace cp=camerapixelsigns;
    const auto mul=[](std::uint64_t a,std::uint64_t b,std::uint64_t& out){return renderpassoutput::detail::checkedMultiply(a,b,out);};
    const auto add=[](std::uint64_t& a,std::uint64_t b){return renderpassoutput::detail::checkedAdd(a,b,a);};
    if(!attempt.receiver||attempt.receiver->owner()!=&frame||!attempt.receiver->retained().terminal
        ||!attempt.cohort||attempt.cohort->sources().empty())return false;
    const auto& frozen=attempt.cohort->sources()[0]->frame;
    const auto t=std::uint64_t(frozen.triangles.size());if(t<3)return false;
    std::array<std::uint64_t,8> w{};
    // Thirty-two lifecycle/inspection/reconciliation passes cover begin,
    // reservation, dispatch, the three guarded proof inspections and
    // their nested accounting checks; each may visit the common source
    // lanes, parent program lanes and every fixed attempt. The byte gate stops
    // an oversized walk, so even a refused setup has a finite host envelope.
    // Local/world association additionally scans one bounded BLAS per triangle.
    const auto passes=32ull*(HarmonicMIDI::grid::Visual3DScene::kMaxLights+std::size_t(OpenGlQueryArithmeticRole::Count)+1)
        *(OpenGlFixtureSceneFrame::kMaximumNativeAttempts+1);
    std::uint64_t association=0,source=0;
    if(!mul(renderpassoutput::kMaximumTotalBytes/4,passes,w[7])
        ||!mul(t,frozen.query.texels.size(),association)||!mul(association,8,association)
        ||!mul(kQueryProgramSourceReservation/4,8,source)||!add(w[7],association)||!add(w[7],source))return false;
    if(attempt.cameraCoverageReserved) {
        std::uint64_t p=0;for(const auto& row:attempt.receiver->retained().rows)
            if(row.disposition==terminalgc::DirectGratingTerminalReceiverRow::Disposition::Pixel)++p;
        const auto& b=attempt.cameraCoverageBounds;
        terminalgc::NativeCameraCellCoverageBudget minimal;
        minimal.retainedBytes=attempt.receiver->retained().bounds.bytes;
        minimal.combinedWork=attempt.receiver->retained().bounds.combinedWork;minimal.fixedBytes=1;
        terminalgc::CameraCellCoverageBounds geometry;terminalgc::CameraCellCoverageReadback empty;
        if(!terminalgc::calculateCameraCellCoverageBounds(*attempt.receiver,frozen,minimal,empty,geometry,error)
            ||b.triangles!=t||b.maximumPixels!=p||b.operationWork!=geometry.operationWork
            ||!mul(p,t,w[0])||!mul(p,t-1,w[1])||!mul(p,5*t+3,w[2]))return false;
        std::uint64_t extra=0;
        if(!mul(w[1],cc::maximumVisits(256),w[3])||!mul(w[2],cp::maximumVisits(),extra)||!add(w[3],extra)
            ||!mul(w[1],cc::maximumWordMoves(256),w[4])||!mul(w[2],cp::maximumWordMoves(),extra)||!add(w[4],extra)
            ||!mul(w[1],cc::maximumScalarOperations(256),w[5])||!mul(w[2],cp::maximumScalarOperations(),extra)||!add(w[5],extra))return false;
        // Native wrapper clears/copies, request assembly, retained primary,
        // all endpoint comparisons and classification loops, separate from EL.
        // corners() adds its own array arguments, local request, certificate
        // stores and at most three uniform-sign argument copies per call.
        constexpr std::uint64_t wrapperWords=1506+16+2*57+34+23+557
            +13*(57+557+2*73+65+73+3*73)+3*13*73+2*557+168+10+512;
        if(!mul(w[0],wrapperWords,extra)||!add(w[4],extra)||!mul(extra,64,extra)||!add(w[5],extra)
            // Dispatch decode plus the three protocol inspections. Standalone
            // fixture diagnostics do not dispatch or grant production authority.
            ||!mul(w[1],4*camerawideline::kHostValidationScalarOperations,extra)||!add(w[6],extra)
            ||!mul(w[2],4*cp::kHostValidationScalarOperations,extra)||!add(w[6],extra)
            ||!mul(w[1],4*(4*557+4*57),extra)||!add(w[7],extra)
            ||!mul(w[2],4*(3*73+65),extra)||!add(w[7],extra))return false;
    }
    std::uint64_t hostWordScalar=0;
    if(!mul(w[7],64,hostWordScalar)||!add(w[6],hostWordScalar))return false;
    output=w;return true;
}
bool openGlAddAttempt(const OpenGlFixtureSceneFrame& frame,const OpenGlNativeAttempt& attempt,
    OpenGlNativeAttemptTotals& total,std::string& error)
{
    const auto add=[](std::uint64_t& value,std::uint64_t amount){return renderpassoutput::detail::checkedAdd(value,amount,value);};
    if(attempt.owner!=&frame||attempt.context!=frame.ownerContext||!frame.staticResources||!frame.staticResources->queryContext
        ||attempt.renderer!=frame.rendererGeneration_||attempt.contextGeneration!=frame.staticResources->queryContext->generation
        ||!attempt.binding||(frame.gratingStagingBinding&&attempt.binding->evaluationTime!=frame.gratingCameraEvaluationTime))return false;
    const auto stagingGeneration=frame.gratingStagingBinding?frame.gratingStagingSubmission.submissionSerial
        :frame.causticReadback?frame.causticReadback->deposition.submission.submissionSerial:0;
    if(!stagingGeneration||attempt.stagingGeneration!=stagingGeneration)return false;
    const auto& binding=*attempt.binding;
    if(attempt.kind==OpenGlNativeAttemptKind::CameraCoverage) {
        std::array<std::uint64_t,8> exact{};
        if(!attempt.receiver||!attempt.cohort||attempt.receiver->retained().terminal!=attempt.terminal
            ||attempt.terminal->retained().incident.cohort!=attempt.cohort
            ||!openGlCameraCoverageCosts(frame,attempt,exact,error)||exact!=attempt.cameraCoverageWork)return false;
        for(unsigned k=0;k<exact.size();++k)if(!add(total.cameraCoverageWork[k],exact[k]))return false;
        constexpr auto maximumProducts=terminalgc::kMaximumQueryWork/96;
        if(total.cameraCoverageWork[0]>maximumProducts||total.cameraCoverageWork[1]>maximumProducts
            ||total.cameraCoverageWork[2]>6*maximumProducts)return false;
        return !attempt.cameraCoverageReserved||add(total.coverage,attempt.cameraCoverageBounds.operationWork);
    }
    if(attempt.kind==OpenGlNativeAttemptKind::Receiver) {
        std::uint64_t allowance=0;
        if(!attempt.terminal||attempt.terminal->owner()!=&frame||!attempt.cohort
            ||attempt.terminal->retained().incident.cohort!=attempt.cohort
            ||attempt.cohort->sources().empty()||attempt.count!=attempt.terminal->retained().bounds.maximumOrders
            ||!terminalgc::directGratingTerminalReceiverAllowance(attempt.cohort->sources()[0]->frame,allowance))return false;
        return add(total.bytes,attempt.terminal->retained().bounds.bytes)
            &&add(total.native,attempt.count*(allowance+attempt.cohort->sources()[0]->frame.triangles.size()));
    }
    if(attempt.terminal)return false;
    if(attempt.kind==OpenGlNativeAttemptKind::Cohort||attempt.kind==OpenGlNativeAttemptKind::Terminal) {
        if(!attempt.cohort||attempt.cohort->sources().empty()||attempt.cohort->sources().size()>16
            ||attempt.cohort->inputs().empty()||attempt.cohort->inputs().size()>terminalgc::kMaximumEntrySamples
            ||!add(total.bytes,attempt.cohort->bounds().bytes))return false;
        if(attempt.kind==OpenGlNativeAttemptKind::Terminal) {
            const auto bands=attempt.cohort->sources().size()*attempt.cohort->inputs().size()*8,orders=bands*terminalgc::kGratingOrdersPerIncident;
            return add(total.direct,bands+orders)&&add(total.terminal,bands+orders)&&add(total.terminalDeposition,orders);
        }
        std::uint64_t work=0,deposition=0;
        for(const auto& source:attempt.cohort->sources()) {
            terminalgc::DirectGratingContributionBounds bounds;
            if(!source||!terminalgc::calculateDirectGratingContributionBounds(*source,attempt.cohort->inputs().size(),0,0,bounds,error)
                ||!add(work,bounds.combinedWork)||!add(deposition,source->frame.bounds.deposition.depositionWork+source->bounds.orderWork))return false;
        }
        const auto baseWork=frame.gratingStagingBinding->bounds.combinedWork;
        const auto baseDeposition=frame.gratingStagingBinding->frame.bounds.deposition.depositionWork+frame.gratingStagingBinding->bounds.orderWork;
        return work>=baseWork&&deposition>=baseDeposition&&add(total.direct,work-baseWork)&&add(total.cohortDeposition,deposition-baseDeposition);
    }
    if(attempt.cohort||!add(total.bytes,binding.bounds.bytes+binding.frame.bounds.transientBytes))return false;
    if(attempt.kind==OpenGlNativeAttemptKind::Direct) {
        terminalgc::DirectGratingContributionBounds bounds;
        return terminalgc::calculateDirectGratingContributionBounds(binding,attempt.count,0,0,bounds,error)&&add(total.direct,bounds.queryWork+bounds.bandWork);
    }
    if(attempt.kind==OpenGlNativeAttemptKind::Camera) {
        terminalgc::GlassCameraGratingTraceBounds bounds;
        return terminalgc::calculateGlassCameraGratingTraceBounds(binding,attempt.count,0,0,bounds,error)&&add(total.camera,bounds.queryWork+bounds.branchWork);
    }
    if(attempt.kind!=OpenGlNativeAttemptKind::Adjunct||attempt.adjuncts>15||!binding.prepared.branchPolicy)return false;
    terminalgc::GratingIncidentBounds original;
    if(!terminalgc::calculateGratingIncidentBounds(binding.frame,binding.triangles.size(),original,error)
        ||original.combinedWork!=binding.bounds.combinedWork||!add(total.native,original.combinedWork)
        ||!add(total.nativeDeposition,binding.frame.bounds.deposition.depositionWork+original.orderWork))return false;
    if(attempt.adjuncts&1) {
        terminalgc::GratingCameraBounds camera;terminalgc::DepositionSchedule deposition;
        if(!terminalgc::calculateGratingCameraBounds(binding,camera,error)
            ||!terminalgc::calculateDepositionSchedule(binding.frame.width,binding.frame.height,binding.bounds.maximumOrders,deposition,error)
            ||!add(total.cameraAdjunct,camera.queryWork)||!add(total.nativeDeposition,deposition.depositionWork+deposition.resolveWork))return false;
    }
    const auto interfaces=binding.prepared.branchPolicy->description().maximumInterfaces;
    const auto query=terminalgc::gratingCellQueryAllowance(binding.frame);
    if((attempt.adjuncts&2)&&!add(total.cell,binding.bounds.maximumIncidents*terminalgc::kGratingCellTracePaths*(interfaces+1)*query))return false;
    return !(attempt.adjuncts&4)||(attempt.leaves&&attempt.leaves<=terminalgc::kMaximumGratingCellCoverageLeaves
        &&add(total.coverage,binding.bounds.maximumIncidents*attempt.leaves*(interfaces+1)*query));
}
bool openGlAttemptTotals(const OpenGlFixtureSceneFrame& frame,OpenGlNativeAttemptTotals& total,std::string& error)
{
    if(frame.nativeAttemptCount>frame.nativeAttempts.size())return false;
    std::size_t found=0;
    for(std::size_t index=0;index<frame.nativeAttempts.size();++index)if(frame.nativeAttempts[index].ordinal) {
        if(index!=found||frame.nativeAttempts[index].ordinal!=index+1||!openGlAddAttempt(frame,frame.nativeAttempts[index],total,error))return false;
        ++found;
    }
    if(found!=frame.nativeAttemptCount)return false;
    return true;
}
bool openGlTerminalLedgerBytes(const terminalgc::DirectGratingTerminalOrderLedger& value,std::uint64_t& bytes)
{
    const auto add=[&](std::uint64_t count,std::uint64_t stride=1) {
        std::uint64_t amount=0;return renderpassoutput::detail::checkedMultiply(count,stride,amount)
            &&renderpassoutput::detail::checkedAdd(bytes,amount,bytes)&&bytes<=renderpassoutput::kMaximumTotalBytes;
    };
    if(!add(sizeof(value))||!add(value.sources.capacity(),sizeof(terminalgc::NativeDirectGratingContributionReadback))
        ||!add(value.bands.capacity(),sizeof(terminalgc::DirectGratingTerminalBand))
        ||!add(value.orders.capacity(),sizeof(terminalgc::DirectGratingTerminalOrder))
        ||!add(value.incident.endpoints.capacity(),sizeof(terminalgc::DirectGratingEndpointIrradianceSum))
        ||!add(value.incident.submission.frameIdentity.capacity()+1))return false;
    for(const auto& source:value.sources)if(!add(source.samples.capacity(),sizeof(terminalgc::NativeDirectGratingContributionSample))
        ||!add(source.submission.frameIdentity.capacity()+1))return false;
    // Each result retains its frozen owner. Charging aliases conservatively
    // avoids losing an earlier distinct owner when the frame changes cohort.
    return !value.incident.cohort||add(value.incident.cohort->bounds().bytes);
}
bool openGlSameTerminalLedger(const terminalgc::DirectGratingTerminalOrderLedger& a,
    const terminalgc::DirectGratingTerminalOrderLedger& b)
{
    if(a.incident.cohort!=b.incident.cohort||!(a.incident.submission==b.incident.submission)
        ||a.incident.sourceMeasure!=b.incident.sourceMeasure||a.incident.endpoints.size()!=b.incident.endpoints.size()
        ||a.sources.size()!=b.sources.size()||a.bands.size()!=b.bands.size()||a.orders.size()!=b.orders.size()
        ||std::tie(a.bounds.bytes,a.bounds.combinedWork,a.bounds.depositionWork,a.bounds.maximumBands,a.bounds.maximumOrders)
            !=std::tie(b.bounds.bytes,b.bounds.combinedWork,b.bounds.depositionWork,b.bounds.maximumBands,b.bounds.maximumOrders))return false;
    const auto same=[](const auto& x,const auto& y){return std::memcmp(&x,&y,sizeof(x))==0;};
    for(std::size_t i=0;i<a.incident.endpoints.size();++i) {
        const auto& x=a.incident.endpoints[i];const auto& y=b.incident.endpoints[i];const auto& h=x.endpoint;const auto& k=y.endpoint;
        if(!terminalgc::sameDirectGratingEndpointInput(x.input,y.input)||!same(x.barycentric,y.barycentric)
            ||!same(x.surfaceIrradiance,y.surfaceIrradiance)||!same(h.distance,k.distance)||!same(h.position,k.position)
            ||!same(h.outwardNormal,k.outwardNormal)
            ||std::tie(h.status,h.objectId,h.materialId,h.instanceId,h.triangleId,h.entering,h.work)
                !=std::tie(k.status,k.objectId,k.materialId,k.instanceId,k.triangleId,k.entering,k.work))return false;
    }
    for(std::size_t i=0;i<a.sources.size();++i)if(!terminalgc::sameDirectGratingContributionReadback(a.sources[i],b.sources[i]))return false;
    for(std::size_t i=0;i<a.bands.size();++i) {
        const auto& x=a.bands[i];const auto& y=b.bands[i];
        if(std::tie(x.source,x.endpoint,x.band,x.witness,x.firstOrder,x.orderCount,x.responseEvaluated)
            !=std::tie(y.source,y.endpoint,y.band,y.witness,y.firstOrder,y.orderCount,y.responseEvaluated)
            ||!same(x.frame.tangent,y.frame.tangent)||!same(x.frame.bitangent,y.frame.bitangent)||!same(x.frame.normal,y.frame.normal)
            ||!same(x.toSource,y.toSource)||!same(x.uv,y.uv)||!same(x.wavelengthNanometres,y.wavelengthNanometres)
            ||!same(x.incidentIrradiance,y.incidentIrradiance)||!same(x.substrateReflectedPower,y.substrateReflectedPower)
            ||!same(x.resolvedReflectedPower,y.resolvedReflectedPower)||!same(x.unresolvedReflectedPower,y.unresolvedReflectedPower)
            ||!same(x.substrateReflectedIrradiance,y.substrateReflectedIrradiance)||!same(x.resolvedReflectedIrradiance,y.resolvedReflectedIrradiance)
            ||!same(x.unresolvedReflectedIrradiance,y.unresolvedReflectedIrradiance)||!same(x.absorbedIrradiance,y.absorbedIrradiance))return false;
    }
    for(std::size_t i=0;i<a.orders.size();++i) {
        const auto& x=a.orders[i];const auto& y=b.orders[i];
        if(std::tie(x.source,x.endpoint,x.band,x.order.primary,x.order.secondary)
            !=std::tie(y.source,y.endpoint,y.band,y.order.primary,y.order.secondary)
            ||!same(x.order.direction,y.order.direction)||!same(x.order.reflectedPower,y.order.reflectedPower)
            ||!same(x.order.selectionProbability,y.order.selectionProbability)
            ||!same(x.order.angularStandardDeviationRadians,y.order.angularStandardDeviationRadians)||!same(x.orderIrradiance,y.orderIrradiance))return false;
    }
    return true;
}
// Reconstruct capacities without allocating or copying a frozen source. The
// original reservations remain conservative ceilings for existing adjuncts;
// actual native allocations and retained host owners must fit those ceilings.
bool openGlTerminalNativeBudget(const OpenGlFixtureSceneFrame& frame,terminalgc::DirectGratingTerminalNativeBudget& budget,std::string& error)
{
#if defined(__APPLE__)
    (void)frame;(void)budget;error="OpenGL terminal orders require native GL4.3 storage";return false;
#else
    const auto refuse=[&](){error="OpenGL terminal native ownership, capacity or cumulative bounds changed";return false;};
    const auto resources=frame.staticResources;
    if(!resources||!resources->snapshot||!resources->materialProgram||!resources->queryContext||!resources->queryContext->active
        ||resources->ownerContext!=frame.ownerContext||frame.ownerContext!=glfwGetCurrentContext()
        ||resources->rendererGeneration!=frame.rendererGeneration_||!frame.gratingStagingBinding
        ||!frame.gratingCameraRuntime||!frame.privateCameraAttachments||frame.gratingCameraBaseComplete
        ||!frame.reconcileNativeAccounting(error))return refuse();
    const auto isRasterSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
    const auto rasterSyncStatus=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
    if(!frame.rasterCaptureProgram||frame.rasterCaptureProgram->preciseSceneQuery
        ||!frame.rasterCaptureProgram->registered||!frame.rasterCaptureProgram->liveUnlocked()
        ||frame.rasterCaptureProgram->context!=resources->queryContext
        ||frame.rasterCaptureGeneration!=frame.gratingStagingSubmission.rasterGeneration
        ||!isRasterSync||!rasterSyncStatus||!frame.rasterCaptureProgram->directFence
        ||!isRasterSync(frame.rasterCaptureProgram->directFence))return refuse();
    int rasterDone=0;rasterSyncStatus(frame.rasterCaptureProgram->directFence,GL_SYNC_STATUS,1,nullptr,&rasterDone);
    if(rasterDone!=GL_SIGNALED)return refuse();
    terminalgc::DirectGratingTerminalNativeBudget result;
    const auto add=[](std::uint64_t& total,std::uint64_t count,std::uint64_t stride=1) {
        std::uint64_t amount=0;return renderpassoutput::detail::checkedMultiply(count,stride,amount)
            &&renderpassoutput::detail::checkedAdd(total,amount,total)&&total<=renderpassoutput::kMaximumTotalBytes;
    };
    result.retainedBytes=frame.causticDiagnostics.admittedFrameBytes;
    for(auto owned:{frame.directGratingBytes,frame.cameraGlassTraceBytes,frame.queryRetainedBytes,frame.nativeAttemptBytes,frame.gratingCameraSpectrumBytes,
        frame.gratingCellSourceBytes,frame.gratingCoverageBytes,resources->queryContext->deferredBytes.load(std::memory_order_relaxed)})
        if(!add(result.retainedBytes,owned))return refuse();
    result.combinedWork=frame.gratingStagingBinding->bounds.combinedWork;
    for(auto work:{frame.directGratingWork,frame.cameraGlassTraceWork,frame.gratingNativeWork,frame.gratingCameraWork,frame.gratingCellWork,frame.gratingCoverageWork})
        if(!add(result.combinedWork,work))return refuse();
    if(result.combinedWork>terminalgc::kMaximumQueryWork||!frame.directDepositionBudget(result.depositionWork))return refuse();
    std::uint64_t actual=sizeof(frame)+sizeof(*resources)+sizeof(OpenGlQueryContext)+384,native=0;
    const auto vector=[&](const auto& values){return add(actual,values.capacity(),sizeof(typename std::decay_t<decltype(values)>::value_type));};
    const auto material=[&](const NativeFixtureSurfaceMaterialProgram& value) {
        return add(actual,sizeof(value)+128)&&add(actual,value.bindingDigest.capacity()+value.programIdentity.capacity()+value.diffractionProductDigest.capacity()+3)
            &&vector(value.objectPrograms)&&(!value.importedBaseColorTexture||vector(value.importedBaseColorTexture->texels))
            &&(!value.vertexProgram||(add(actual,sizeof(*value.vertexProgram)+128+value.vertexProgram->sourceIdentity().capacity()
                +value.vertexProgram->programIdentity().capacity()+2)&&vector(value.vertexProgram->surfaceInstructions())&&vector(value.vertexProgram->vertexInstructions())))
            &&(!value.diffractionLightingAdmission||add(actual,sizeof(*value.diffractionLightingAdmission)+128));
    };
    const auto materialOwner=[&](const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& value) {
        if(!value||!material(*value))return false;
        for(const auto& child:value->objectPrograms)if(!child||!material(*child))return false;
        return true;
    };
    if(!add(actual,sizeof(*resources->snapshot)+128+resources->backend_.capacity()+1)||!vector(resources->snapshot->textureTexels)
        ||!materialOwner(resources->materialProgram)||!vector(resources->instanceAppearances)||!vector(resources->textures)
        ||!vector(resources->opaqueVisibility.texels)||!vector(resources->opaqueVisibility.meshes)
        ||!vector(resources->smoothGlass.query.texels)||!vector(resources->smoothGlass.query.meshes)
        ||!vector(resources->diffractionTransport.query.texels)||!vector(resources->diffractionTransport.query.meshes)
        ||!add(actual,resources->diffractionTransport.identity.capacity()+1)
        ||!vector(frame.gratingCameraRuntime->animatedLights)||!add(actual,frame.gratingStagingSubmission.frameIdentity.capacity()+1)
        ||(resources->sharedProgram&&!add(actual,sizeof(*resources->sharedProgram)+128)))return refuse();
    if(resources->materialProgramSource&&resources->materialProgramSource!=resources->materialProgram
        &&!materialOwner(resources->materialProgramSource))return refuse();
    if(resources->environmentResident) {
        const auto& resident=*resources->environmentResident;
        if(!resident.payload||!add(actual,sizeof(resident)+sizeof(*resident.payload)+256+resident.payload->key().id.capacity()
            +resident.payload->key().contentSha256.capacity()+2)||!vector(resident.image.pixels)||!vector(resident.payload->bytes()))return refuse();
    }
    if(resources->environmentBudget&&!add(actual,sizeof(*resources->environmentBudget)+128))return refuse();
    const auto evidence=[&](const auto& self,const std::shared_ptr<const OpenGlQueryArithmeticEvidence>& value,unsigned depth)->bool {
        if(!value)return true;
        if(depth>std::size_t(OpenGlQueryArithmeticRole::Count)||!add(actual,sizeof(*value)+128+value->submission.frameIdentity.capacity()+1))return false;
        for(const auto& parent:value->parents)if(!self(self,parent,depth+1))return false;
        return true;
    };
    for(const auto& value:frame.queryEvidence)if(!evidence(evidence,value,0))return refuse();
    const auto frozen=[&](const terminalgc::GratingIncidentBinding& binding) {
        const auto& f=binding.frame;
        return add(actual,sizeof(binding))&&vector(f.query.texels)&&vector(f.instances)&&vector(f.triangles)&&vector(f.entries)
            &&vector(f.work)&&vector(f.workTexels)&&vector(f.triangleTexels)&&vector(binding.triangles)&&vector(binding.witnessTexels)
            &&vector(binding.prepared.glass.query.texels)&&vector(binding.prepared.glass.query.meshes)
            &&add(actual,binding.identity.capacity()+binding.prepared.identity.capacity()+f.identity.capacity()+f.sourceIdentity.capacity()
                +f.branchPolicyIdentity.capacity()+f.cameraIdentity.capacity()+6)
            &&binding.prepared.glass.sourceScene&&add(actual,sizeof(*binding.prepared.glass.sourceScene))
            &&vector(binding.prepared.glass.sourceScene->textureTexels)&&binding.gratingOwner&&material(*binding.gratingOwner)
            &&materialOwner(binding.prepared.glass.sourceMaterial)&&(!f.activeSource||add(actual,sizeof(*f.activeSource)))
            &&(!binding.prepared.glass.distinctGrating||add(actual,sizeof(*binding.prepared.glass.distinctGrating)+128))
            &&binding.material&&add(actual,sizeof(*binding.material)+binding.material->structuralDigest().capacity()+1);
    };
    if(!frozen(*frame.gratingStagingBinding))return refuse();
    for(std::size_t index=0;index<frame.nativeAttemptCount;++index) {
        const auto& attempt=frame.nativeAttempts[index];
        // Coverage setup aliases the permanently retained published receiver;
        // its frozen cohort is already counted through that receiver's owners.
        if(attempt.kind==OpenGlNativeAttemptKind::CameraCoverage)continue;
        if(attempt.cohort) {
            if(!add(actual,sizeof(*attempt.cohort)+attempt.cohort->identity().capacity()+1)
                ||!vector(attempt.cohort->sources())||!vector(attempt.cohort->inputs()))return refuse();
            for(const auto& source:attempt.cohort->sources())if(!source||!frozen(*source))return refuse();
        } else if(!frozen(*attempt.binding))return refuse();
    }
    const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
    const auto sizeOf=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    if(!isBuffer||!sizeOf)return refuse();
    const auto& gl=resources->gl;int previous=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&previous);
    int token=0;auto restore=[&](int*){gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(previous));};
    std::unique_ptr<int,decltype(restore)> guard(&token,restore);
    const auto buffer=[&](unsigned name,std::uint64_t expected) {
        if(!name||!isBuffer(name))return false;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,name);GLint64 bytes=0;sizeOf(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&bytes);
        return bytes>=0&&std::uint64_t(bytes)==expected&&add(native,std::uint64_t(bytes));
    };
    if(!buffer(frame.causticBuffers[3],frame.gratingStagingBinding->frame.workTexels.size()*sizeof(terminalgc::Texel))
        ||!buffer(frame.causticBuffers[4],frame.gratingStagingBinding->frame.triangleTexels.size()*sizeof(terminalgc::Texel))
        ||!buffer(frame.directionalSourceBuffer,sizeof(terminalgc::NativeDirectionalSourceInput)))return refuse();
    const auto retainedBuffer=[&](unsigned name) {
        if(!name){return true;}if(!isBuffer(name))return false;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,name);GLint64 bytes=0;sizeOf(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&bytes);
        return bytes>=0&&add(native,std::uint64_t(bytes));
    };
    for(unsigned slot:{0u,1u,2u,5u,6u})if(!retainedBuffer(frame.causticBuffers[slot]))return refuse();
    for(auto name:frame.gratingCameraSpectrumBuffers)if(!retainedBuffer(name))return refuse();
    for(auto name:{resources->vertexBuffer,resources->indexBuffer,resources->instanceBuffer,resources->previousVertexBuffer})
        if(!retainedBuffer(name))return refuse();
    int previousTexture=0;glGetIntegerv(GL_TEXTURE_BINDING_2D,&previousTexture);
    int textureToken=0;auto restoreTexture=[&](int*){glBindTexture(GL_TEXTURE_2D,unsigned(previousTexture));};
    std::unique_ptr<int,decltype(restoreTexture)> textureGuard(&textureToken,restoreTexture);
    const auto retainedTexture=[&](unsigned name) {
        if(!name){return true;}if(!glIsTexture(name))return false;
        glBindTexture(GL_TEXTURE_2D,name);int width=0,height=0,format=0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&width);glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&height);
        glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&format);
        std::uint64_t stride=0;
        switch(format) {
            case GL_RGBA32F:stride=16;break;
            case GL_RGB32F:stride=12;break;
            case GL_RGBA16F:case GL_RG32F:stride=8;break;
            case GL_RGB16F:stride=6;break;
            case GL_RGBA8:case GL_SRGB8_ALPHA8:case GL_RG16F:case GL_R32F:case GL_R32UI:
            case GL_DEPTH_COMPONENT24:case GL_DEPTH_COMPONENT32F:stride=4;break;
            case GL_R16F:stride=2;break;
            case GL_R8:stride=1;break;
            default:return false;
        }
        return width>0&&height>0&&add(native,std::uint64_t(width)*std::uint64_t(height),stride);
    };
    for(auto name:frame.causticTextures)if(!retainedTexture(name))return refuse();
    for(auto name:{frame.texture_,frame.accumulationTexture_,frame.depthTexture_,frame.gratingCameraDepthAttachment,
        frame.gratingCameraSpectrumTexture,frame.glassTexture,frame.opaqueTexture,frame.diffractionQueryTexture,frame.diffractionSamplesTexture,
        frame.previousNoteTexture,resources->environmentTexture})if(!retainedTexture(name))return refuse();
    for(auto name:frame.rawTextures)if(!retainedTexture(name))return refuse();
    for(auto name:resources->textures)if(!retainedTexture(name))return refuse();
    const auto direct=[&](const OpenGlDirectGratingBatch& batch) {
        const auto inputs=batch.upload.inputs.size()*sizeof(terminalgc::NativeDirectGratingEndpointInput);
        const auto packed=batch.packed.size()*sizeof(terminalgc::opengl::DirectGratingPacked);
        const auto source=batch.program&&batch.program->directionalInputBuffer?sizeof(terminalgc::NativeDirectionalSourceInput):0;
        if(!batch.program||batch.nativeBytes<inputs+packed+source)return false;
        return batch.program&&evidence(evidence,batch.evidence,0)&&add(actual,sizeof(batch))&&vector(batch.upload.inputs)&&vector(batch.packed)&&vector(batch.readback.samples)
            &&add(actual,batch.upload.submission.frameIdentity.capacity()+batch.readback.submission.frameIdentity.capacity()+2)
            &&buffer(batch.program->directBuffers[0],batch.upload.inputs.size()*sizeof(terminalgc::NativeDirectGratingEndpointInput))
            &&buffer(batch.program->directBuffers[1],batch.packed.size()*sizeof(terminalgc::opengl::DirectGratingPacked))
            &&batch.program->retainedBytes>=batch.nativeBytes&&buffer(batch.program->directBuffers[2],batch.nativeBytes-inputs-packed-source)
            &&(!batch.program->directionalInputBuffer||buffer(batch.sourceBuffer,sizeof(terminalgc::NativeDirectionalSourceInput)));
    };
    std::uint64_t nativeWork=0,cohortDeposition=0,cohortBytes=0;
    for(auto* cohort=frame.directGratingCohort.get();cohort;cohort=cohort->previous.get()) {
        if(!cohort->binding||cohort->sources.size()!=cohort->binding->sources().size()||!vector(cohort->sources)
            ||!add(actual,sizeof(*cohort)+cohort->submission.frameIdentity.capacity()+1)
            ||!add(actual,cohort->binding->identity().capacity()+sizeof(terminalgc::DirectGratingIrradianceCohort))
            ||!vector(cohort->binding->inputs())||!vector(cohort->binding->sources())
            ||!add(nativeWork,cohort->chargedWork)||!add(cohortDeposition,cohort->reservedDepositionWork)||!add(cohortBytes,cohort->ownedBytes))return refuse();
        for(const auto& source:cohort->sources)if(!source||source->frame!=&frame||source->cohort!=cohort||!source->binding
            ||!source->batch||!add(actual,sizeof(*source)+source->sourceUpload.submission.frameIdentity.capacity()+1)
            ||!frozen(*source->binding)||!direct(*source->batch))return refuse();
    }
    for(auto* batch=frame.directGratingBatch.get();batch;batch=batch->previous.get())
        if(!direct(*batch)||!add(nativeWork,batch->retainedWork)||!add(cohortBytes,batch->ownedBytes))return refuse();
    for(auto* batch=frame.cameraGlassTraceBatch.get();batch;batch=batch->previous.get()) {
        if(!batch->program||!evidence(evidence,batch->evidence,0)||!add(actual,sizeof(*batch))||!vector(batch->upload.inputs)||!vector(batch->packed)
            ||!vector(batch->readback.bands)||!add(actual,batch->upload.submission.frameIdentity.capacity()+batch->readback.submission.frameIdentity.capacity()+2))return refuse();
        for(const auto& band:batch->readback.bands)if(!vector(band.visits))return refuse();
        for(unsigned slot=0;slot<3;++slot)if(!buffer(batch->program->directBuffers[slot],slot==0?batch->upload.inputs.size()*sizeof(terminalgc::NativeGlassCameraGratingTraceInput)
            :slot==1?batch->packed.size()*sizeof(terminalgc::opengl::CameraGlassGratingPacked):frame.gratingStagingBinding->witnessTexels.size()*sizeof(terminalgc::Texel)))return refuse();
    }
    // Program identities and all their sources are retained by this resource
    // context's intrusive native lease list, including deferred owners.
    for(auto* program=resources->queryContext->programs;program;program=program->next) {
        if(!program->identity||!add(actual,sizeof(*program)+sizeof(*program->identity)))return refuse();
        for(unsigned stage=0;stage<program->identity->stageCount;++stage)
            if(stage>=program->identity->stages.size()||!add(actual,program->identity->stages[stage].source.capacity()+1))return refuse();
    }
    std::uint64_t terminalWork=0,terminalDeposition=0;
    for(auto* record=frame.directTerminalResults.get();record;record=record->previous.get()) {
        std::uint64_t owned=sizeof(*record)+sizeof(terminalgc::NativeDirectGratingTerminalResult)+128;
        if(!record->result||record->result->owner()!=&frame||record->result->retained().nativeResult
            ||!openGlTerminalLedgerBytes(record->result->retained(),owned)||owned>record->ownedBytes
            ||!add(actual,owned)||!add(cohortBytes,record->ownedBytes)||!add(terminalWork,record->work)
            ||!add(terminalDeposition,record->depositionWork))return refuse();
    }
    for(auto* failed=frame.failedReceiverResources.get();failed;failed=failed->previous.get()) {
        if(!failed->program||!add(actual,sizeof(*failed))||!vector(failed->inputs)||!vector(failed->packed)
            ||!add(actual,failed->program->retainedBytes)||!add(cohortBytes,failed->ownedBytes)
            ||!buffer(failed->program->directBuffers[0],failed->inputs.size()*sizeof(terminalgc::NativeDirectGratingTerminalReceiverInput))
            ||!buffer(failed->program->directBuffers[1],failed->packed.size()*sizeof(terminalgc::opengl::DirectTerminalReceiverPacked)))return refuse();
        if(failed->result&&(failed->result->owner()!=&frame||failed->result->retained().nativeResult
            ||!add(actual,sizeof(terminalgc::NativeDirectGratingTerminalReceiverResult)+128)
            ||!vector(failed->result->retained().rows)||!add(actual,failed->result->retained().submission.frameIdentity.capacity()+1)))return refuse();
    }
    for(auto* record=frame.directReceiverResults.get();record;record=record->previous.get()) {
        const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
        const auto syncStatus=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
        if(!record->program||!isSync||!syncStatus||!record->program->directFence||!isSync(record->program->directFence))return refuse();
        int signalled=0;syncStatus(record->program->directFence,GL_SYNC_STATUS,1,nullptr,&signalled);
        if(signalled!=GL_SIGNALED||!record->attemptOrdinal||record->attemptOrdinal>frame.nativeAttemptCount
            ||frame.nativeAttempts[record->attemptOrdinal-1].kind!=OpenGlNativeAttemptKind::Receiver
            ||!record->result||frame.nativeAttempts[record->attemptOrdinal-1].terminal!=record->result->retained().terminal
            ||record->inputGeneration!=record->result->retained().inputGeneration)return refuse();
        std::uint64_t owned=sizeof(*record)+sizeof(terminalgc::NativeDirectGratingTerminalReceiverResult)+128;
        if(!record->result||record->result->owner()!=&frame||record->result->retained().nativeResult||!record->program
            ||!record->program->liveUnlocked()||!vector(record->inputs)||!vector(record->packed)
            ||!add(owned,record->inputs.capacity(),sizeof(terminalgc::NativeDirectGratingTerminalReceiverInput))
            ||!add(owned,record->packed.capacity(),sizeof(terminalgc::opengl::DirectTerminalReceiverPacked))
            ||!add(owned,record->result->retained().rows.capacity(),sizeof(terminalgc::DirectGratingTerminalReceiverRow))
            ||!add(owned,record->result->retained().submission.frameIdentity.capacity()+1)
            ||!add(owned,record->program->retainedBytes)||owned>record->ownedBytes
            ||!add(actual,sizeof(*record)+sizeof(terminalgc::NativeDirectGratingTerminalReceiverResult)+128)
            ||!vector(record->result->retained().rows)||!add(actual,record->result->retained().submission.frameIdentity.capacity()+1)
            ||!buffer(record->program->directBuffers[0],record->inputs.size()*sizeof(terminalgc::NativeDirectGratingTerminalReceiverInput))
            ||!buffer(record->program->directBuffers[1],record->packed.size()*sizeof(terminalgc::opengl::DirectTerminalReceiverPacked))
            ||!add(cohortBytes,record->ownedBytes))return refuse();
    }
    const auto coverage=[&](const OpenGlCameraCoverageRecord& record) {
        if(!record.parent||!record.attemptOrdinal||record.attemptOrdinal>frame.nativeAttemptCount
            ||frame.nativeAttempts[record.attemptOrdinal-1].receiver!=record.parent
            ||!frame.nativeAttempts[record.attemptOrdinal-1].cameraCoverageReserved
            ||!vector(record.inputs)||!vector(record.certificate))return false;
        if(!record.program)return record.inputs.empty()&&record.certificate.empty()&&!record.result&&add(actual,sizeof(record));
        if(!record.program->liveUnlocked()||!buffer(record.program->directBuffers[0],record.inputs.size()*4)
            ||!buffer(record.program->directBuffers[1],record.certificate.size()*4))return false;
        const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
        const auto status=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
        if(!isSync||!status||!record.program->directFence||!isSync(record.program->directFence))return false;
        int done=0;status(record.program->directFence,GL_SYNC_STATUS,1,nullptr,&done);if(done!=GL_SIGNALED)return false;
        std::uint64_t owned=sizeof(record)+record.program->retainedBytes;
        if(!add(owned,record.inputs.capacity(),4)||!add(owned,record.certificate.capacity(),4)
            ||!add(actual,sizeof(record))||!add(cohortBytes,record.ownedBytes))return false;
        if(record.result&&(record.result->owner()!=&frame||record.result->retained().nativeResult
            ||!add(owned,sizeof(glasscaustics::NativeCameraCellCoverageResult)+128)
            ||!openGlCameraCoverageReadbackBytes(record.result->retained(),owned)
            ||!add(actual,sizeof(glasscaustics::NativeCameraCellCoverageResult)+128)
            ||!openGlCameraCoverageReadbackBytes(record.result->retained(),actual)))return false;
        return owned<=record.ownedBytes;
    };
    for(auto* record=frame.cameraCoverageResults.get();record;record=record->previous.get())if(!coverage(*record))return refuse();
    for(auto* record=frame.failedCameraCoverageResources.get();record;record=record->previous.get())if(!coverage(*record))return refuse();
    if(const auto* active=openGlActiveCameraCoverageRecord(frame);active&&!coverage(*active))return refuse();
    if(frame.gratingCameraSpectrum&&(!add(actual,sizeof(*frame.gratingCameraSpectrum))||!vector(frame.gratingCameraSpectrum->keys)
        ||!vector(frame.gratingCameraSpectrum->spectralRadiance)||!vector(frame.gratingCameraSpectrum->resolvedColor)))return refuse();
    if(frame.gratingCellSourceUpload&&(!add(actual,sizeof(*frame.gratingCellSourceUpload))||!vector(frame.gratingCellSourceUpload->cellTexels)))return refuse();
    const auto directional=[&](const std::shared_ptr<const terminalgc::NativeDirectionalSourceReceipt>& value) {
        return !value||(add(actual,sizeof(*value)+value->submission.frameIdentity.capacity()+1)&&vector(value->potential));
    };
    const auto grating=[&](const std::shared_ptr<const terminalgc::NativeGratingReadback>& value) {
        return !value||(add(actual,sizeof(*value)+value->submission.frameIdentity.capacity()+1)&&vector(value->incidents)
            &&vector(value->paths)&&vector(value->children)&&directional(value->directionalSource));
    };
    const auto camera=[&](const std::shared_ptr<const terminalgc::NativeGratingCameraReadback>& value) {
        return !value||(add(actual,sizeof(*value)+value->submission.frameIdentity.capacity()+1)&&vector(value->orders)&&grating(value->incidents));
    };
    if(frame.gratingCameraSpectrum&&!camera(frame.gratingCameraSpectrum->camera))return refuse();
    if(frame.gratingCellSourceUpload) {
        const auto& upload=*frame.gratingCellSourceUpload;
        if(!add(actual,upload.submission.frameIdentity.capacity()+upload.pointSubmission.frameIdentity.capacity()+2)
            ||!grating(upload.pointReadback)||!upload.sourceCells||!add(actual,sizeof(*upload.sourceCells)+upload.sourceCells->identity.capacity()+1)
            ||!vector(upload.sourceCells->cells)||!vector(upload.sourceCells->cellTexels))return refuse();
    }
    if(frame.gratingCoverageEvidence.payload) {
        const auto& payload=*frame.gratingCoverageEvidence.payload;
        if(!add(actual,sizeof(payload)+frame.gratingCoverageEvidence.submission.frameIdentity.capacity()
            +frame.gratingCoverageEvidence.boundarySubmission.frameIdentity.capacity()+2)||!vector(payload.records))return refuse();
        for(const auto& record:payload.records) {
            if(!vector(record.leaves))return refuse();
            for(const auto& leaf:record.leaves) {
                if(!vector(leaf.sweeps))return refuse();
                for(const auto& sweep:leaf.sweeps)if(!vector(sweep.blockers))return refuse();
            }
        }
    }
    if(frame.causticReadback&&(!add(actual,sizeof(*frame.causticReadback))||!vector(frame.causticReadback->paths)||!vector(frame.causticReadback->deposits)
        ||!vector(frame.causticReadback->spectralRadiance)||!vector(frame.causticReadback->deposition.keys)
        ||!add(actual,frame.causticReadback->frameIdentity.capacity()+frame.causticReadback->deposition.submission.frameIdentity.capacity()+2)
        ||!directional(frame.causticReadback->directionalSource)||!grating(frame.causticReadback->gratingIncidents)||!camera(frame.causticReadback->gratingCamera)))return refuse();
    if(!add(actual,native)||actual>result.retainedBytes||cohortBytes>frame.directGratingBytes
        ||nativeWork>frame.directGratingWork||cohortDeposition>frame.directCohortDepositionWork
        ||terminalWork>frame.terminalAttemptWork||terminalDeposition>frame.terminalAttemptDepositionWork
        ||(frame.directGratingCohort&&(result.retainedBytes<frame.directGratingCohort->binding->bounds().bytes
            ||result.combinedWork<frame.directGratingCohort->binding->bounds().combinedWork
            ||result.depositionWork<frame.directGratingCohort->binding->bounds().depositionWork))||glGetError()!=GL_NO_ERROR)return refuse();
    budget=result;error.clear();return true;
#endif
}

class OpenGlDirectTerminalLease final : public glasscaustics::NativeDirectGratingTerminalLease
{
public:
    struct SourceSeal {
        const OpenGlDirectGratingSource* source=nullptr;const OpenGlDirectGratingBatch* batch=nullptr;
        const OpenGlQueryArithmeticEvidence* evidence=nullptr;
        std::array<unsigned,5> buffers{};unsigned program=0;GLsync fence=nullptr;
        std::uint64_t endpointGeneration=0,sourceGeneration=0;
    };
    // Declared first so the guard outlives staged-owner destruction and rollback.
    std::unique_lock<std::mutex> lock;
    const OpenGlFixtureSceneFrame& frame;
    const OpenGlDirectGratingCohort* cohort;
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sources;
    GLFWwindow* context;
    std::uint64_t contextGeneration,rendererGeneration,oldBytes,workAfter=0,depositionAfter=0;
    double time;
    std::array<SourceSeal,HarmonicMIDI::grid::Visual3DScene::kMaxLights> seals{};
    glasscaustics::DirectGratingTerminalNativeBudget budget;
    glasscaustics::DirectGratingTerminalOrderBounds reservation;
    const glasscaustics::DirectGratingTerminalOrderLedger* candidate=nullptr;
    std::unique_ptr<OpenGlDirectTerminalRecord> staged;
    bool reserved=false,published=false;
    OpenGlDirectTerminalLease(const OpenGlFixtureSceneFrame& owner,std::unique_lock<std::mutex>&& guard,
        const std::vector<glasscaustics::NativeDirectGratingContributionReadback>& values,double evaluationTime)
        :lock(std::move(guard)),frame(owner),cohort(owner.directGratingCohort.get()),sources(values),context(owner.ownerContext),
         contextGeneration(owner.staticResources->queryContext->generation),rendererGeneration(owner.rendererGeneration_),oldBytes(owner.directGratingBytes),time(evaluationTime){}
    ~OpenGlDirectTerminalLease() override {
        staged.reset();
        if(reserved&&!published)frame.directGratingBytes=oldBytes;
        frame.activeTerminalStorage=0;
        if(frame.activeDirectTerminalLease==this)frame.activeDirectTerminalLease=nullptr;
        if(directTerminalThreadLease==this)directTerminalThreadLease=nullptr;
    }
    const NativeFixtureSceneFrame& owner() const noexcept override{return frame;}
    bool live(std::string& error) const {
#if defined(__APPLE__)
        error="OpenGL terminal orders require native GL4.3 fences";return false;
#else
        const auto resources=frame.staticResources;
        if(!lock.owns_lock()||directTerminalThreadLease!=this||frame.activeDirectTerminalLease!=this||frame.directGratingCohort.get()!=cohort||!cohort
            ||sources.size()!=cohort->sources.size()||sources.size()>seals.size()||!resources||!resources->queryContext
            ||!resources->queryContext->active||resources->queryContext->generation!=contextGeneration
            ||resources->queryContext->context!=context||glfwGetCurrentContext()!=context||frame.ownerContext!=context
            ||resources->ownerContext!=context||frame.rendererGeneration_!=rendererGeneration||resources->rendererGeneration!=rendererGeneration
            ||frame.gratingCameraEvaluationTime!=time||!frame.privateCameraAttachments||frame.gratingCameraBaseComplete)
        {error="OpenGL terminal lease lost its guarded common native frame";return false;}
        const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
        const auto status=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
        for(std::size_t i=0;i<sources.size();++i) {
            const auto& seal=seals[i];const auto* source=cohort->sources[i].get();const auto* batch=source?source->batch.get():nullptr;
            if(!source||source!=seal.source||!batch||batch!=seal.batch||!batch->program||batch->evidence.get()!=seal.evidence
                ||source->frame!=&frame||source->cohort!=cohort||source->binding!=cohort->binding->sources()[i]
                ||batch->program->context!=resources->queryContext||batch->program->program!=seal.program||batch->program->directFence!=seal.fence
                ||std::array<unsigned,5>{batch->program->directBuffers[0],batch->program->directBuffers[1],batch->program->directBuffers[2],batch->sourceBuffer,batch->queryTexture}!=seal.buffers
                ||batch->upload.generation!=seal.endpointGeneration||source->sourceUpload.bindingGeneration!=seal.sourceGeneration
                ||!isSync||!status||!seal.fence||!isSync(seal.fence))
            {error="OpenGL terminal lease source registration or completed fence changed";return false;}
            int complete=0;status(seal.fence,GL_SYNC_STATUS,1,nullptr,&complete);
            if(complete!=GL_SIGNALED){error="OpenGL terminal lease native fence is incomplete";return false;}
        }
        if(glGetError()!=GL_NO_ERROR){error="OpenGL terminal lease native inspection failed";return false;}
        error.clear();return true;
#endif
    }
protected:
    void commit() noexcept override {
        staged->previous=std::move(frame.directTerminalResults);
        frame.directTerminalResults=std::move(staged);published=true;
    }
};
class OpenGlDirectTerminalReceiverLease final : public glasscaustics::NativeDirectGratingTerminalReceiverLease
{
public:
    std::unique_lock<std::mutex> lock;
    const OpenGlFixtureSceneFrame& frame;
    const OpenGlDirectGratingCohort* cohort;
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult> terminal;
    GLFWwindow* context;
    std::uint64_t contextGeneration,rendererGeneration,generation,oldBytes,workAfter=0;
    double time;
    std::array<OpenGlDirectTerminalLease::SourceSeal,HarmonicMIDI::grid::Visual3DScene::kMaxLights> seals{};
    std::array<unsigned,5> common{}; // query, raster owners/triangle/depth, world triangles
    glasscaustics::DirectGratingTerminalReceiverNativeBudget budget;
    glasscaustics::DirectGratingTerminalReceiverBounds reservation;
    std::unique_ptr<OpenGlDirectTerminalReceiverRecord> staged;
    bool reserved=false,published=false,dispatched=false,complete=false;
    OpenGlDirectTerminalReceiverLease(const OpenGlFixtureSceneFrame& owner,std::unique_lock<std::mutex>&& guard,
        const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& value,double evaluationTime,std::uint64_t serial)
        :lock(std::move(guard)),frame(owner),cohort(owner.directGratingCohort.get()),terminal(value),context(owner.ownerContext),
         contextGeneration(owner.staticResources->queryContext->generation),rendererGeneration(owner.rendererGeneration_),
         generation(serial),oldBytes(owner.directGratingBytes),time(evaluationTime),
         common{owner.causticTextures[0],owner.causticTextures[1],owner.causticTextures[2],owner.causticTextures[3],owner.causticBuffers[4]} {}
    ~OpenGlDirectTerminalReceiverLease() override {
        if(staged&&!published&&((dispatched&&!complete)||(staged->result&&staged->result.use_count()>1))) {
            // An incomplete wait or an escaped uncommitted result does not
            // prove storage was freed. Keep its charged owner until teardown.
            staged->previous=std::move(frame.failedReceiverResources);
            frame.failedReceiverResources=std::move(staged);
        } else {
            if(staged&&staged->program&&glfwGetCurrentContext()!=context)deferQueryProgramUnlocked(staged->program);
            staged.reset();
            if(reserved&&!published)frame.directGratingBytes=oldBytes;
        }
        frame.activeReceiverStorage=0;
        if(frame.activeDirectReceiverLease==this)frame.activeDirectReceiverLease=nullptr;
        if(directReceiverThreadLease==this)directReceiverThreadLease=nullptr;
    }
    const NativeFixtureSceneFrame& owner() const noexcept override{return frame;}
    const glasscaustics::Frame& frozenFrame() const noexcept override{return cohort->binding->sources()[0]->frame;}
    std::uint64_t inputGeneration() const noexcept override{return generation;}
    bool live(std::string& error) const {
#if defined(__APPLE__)
        error="OpenGL receiver lease requires GL4.3 native storage";return false;
#else
        const auto resources=frame.staticResources;
        if(!lock.owns_lock()||directReceiverThreadLease!=this||frame.activeDirectReceiverLease!=this
            ||frame.activeDirectTerminalLease||frame.directGratingCohort.get()!=cohort||!cohort||!cohort->binding
            ||cohort->sources.empty()||cohort->sources.size()>seals.size()||terminal->owner()!=&frame
            ||terminal->retained().incident.cohort!=cohort->binding||!(terminal->retained().incident.submission==cohort->submission)
            ||!resources||!resources->queryContext||!resources->queryContext->active
            ||resources->queryContext->generation!=contextGeneration||resources->queryContext->context!=context
            ||glfwGetCurrentContext()!=context||frame.ownerContext!=context||resources->ownerContext!=context
            ||frame.rendererGeneration_!=rendererGeneration||resources->rendererGeneration!=rendererGeneration
            ||frame.gratingCameraEvaluationTime!=time||!frame.privateCameraAttachments||frame.gratingCameraBaseComplete
            ||common!=std::array<unsigned,5>{frame.causticTextures[0],frame.causticTextures[1],frame.causticTextures[2],frame.causticTextures[3],frame.causticBuffers[4]})
        {error="OpenGL receiver lease lost its exact guarded native frame";return false;}
        const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
        const auto status=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
        for(std::size_t i=0;i<cohort->sources.size();++i) {
            const auto& seal=seals[i];const auto* source=cohort->sources[i].get();const auto* batch=source?source->batch.get():nullptr;
            if(!source||source!=seal.source||!batch||batch!=seal.batch||!batch->program||batch->evidence.get()!=seal.evidence
                ||batch->program->context!=resources->queryContext||batch->program->program!=seal.program||batch->program->directFence!=seal.fence
                ||std::array<unsigned,5>{batch->program->directBuffers[0],batch->program->directBuffers[1],batch->program->directBuffers[2],batch->sourceBuffer,batch->queryTexture}!=seal.buffers
                ||batch->upload.generation!=seal.endpointGeneration||source->sourceUpload.bindingGeneration!=seal.sourceGeneration
                ||!isSync||!status||!seal.fence||!isSync(seal.fence))
            {error="OpenGL receiver source registration changed";return false;}
            int signalled=0;status(seal.fence,GL_SYNC_STATUS,1,nullptr,&signalled);
            if(signalled!=GL_SIGNALED){error="OpenGL receiver source fence is incomplete";return false;}
        }
        bool registered=false;
        for(auto* record=frame.directTerminalResults.get();record;record=record->previous.get())
            if(record->result==terminal)registered=true;
        if(!registered||glGetError()!=GL_NO_ERROR){error="OpenGL receiver scattering registration expired";return false;}
        if(reserved) {
            if(!staged||!staged->attemptOrdinal||staged->attemptOrdinal>frame.nativeAttemptCount)return false;
            const auto& attempt=frame.nativeAttempts[staged->attemptOrdinal-1];
            if(attempt.kind!=OpenGlNativeAttemptKind::Receiver||attempt.terminal!=terminal||attempt.cohort!=cohort->binding
                ||attempt.count!=reservation.maximumRows||attempt.ordinal!=staged->attemptOrdinal
                ||frame.gratingNativeWork!=workAfter){error="OpenGL receiver persistent admission changed";return false;}
        }
        error.clear();return true;
#endif
    }
protected:
    void commit() noexcept override {
        staged->previous=std::move(frame.directReceiverResults);
        frame.directReceiverResults=std::move(staged);published=true;
    }
};
bool openGlReceiverBudget(const OpenGlFixtureSceneFrame& frame,
    const glasscaustics::NativeDirectGratingTerminalResult& terminal,
    glasscaustics::DirectGratingTerminalReceiverNativeBudget& output,std::string& error)
{
    glasscaustics::DirectGratingTerminalNativeBudget native;
    if(!openGlTerminalNativeBudget(frame,native,error))return false;
    const auto maximum=renderpassoutput::kMaximumTotalBytes;
    const auto floor=terminal.retained().bounds.bytes;
    if(floor>maximum||native.retainedBytes>maximum)return false;
    glasscaustics::DirectGratingTerminalReceiverNativeBudget budget;
    // Preserve the full scattering admission floor, including temporary shared
    // copies, and reserve the next journal's retained terminal before allocation.
    budget.retainedBytes=std::max(native.retainedBytes,floor);
    if(floor>maximum-budget.retainedBytes||sizeof(OpenGlDirectTerminalReceiverLease)>maximum-budget.retainedBytes-floor)return false;
    budget.retainedBytes+=floor+sizeof(OpenGlDirectTerminalReceiverLease);budget.combinedWork=native.combinedWork;
    const auto& frozen=terminal.retained().incident.cohort->sources()[0]->frame;
    std::uint64_t classification=0;
    if(!renderpassoutput::detail::checkedMultiply(terminal.retained().bounds.maximumOrders,frozen.triangles.size(),classification)
        ||classification>glasscaustics::kMaximumQueryWork-budget.combinedWork)return false;
    // Receiver trait lookup is a native all-triangle scan in addition to the
    // released closest-hit work. Charge it independently before dispatch.
    budget.combinedWork+=classification;
    budget.nativeInputBytesPerRow=4*sizeof(glasscaustics::NativeDirectGratingTerminalReceiverInput);
    budget.nativeOutputBytesPerRow=4*sizeof(glasscaustics::opengl::DirectTerminalReceiverPacked);
    // GPU plus retained packed/input copies allow worst doubled capacity.
    // Shared bounds separately reserve all three doubled decoded row copies.
    budget.nativeFixedBytes=4*kQueryProgramSourceReservation+sizeof(OpenGlDirectTerminalReceiverRecord)+512
        +4ull*frame.width_*frame.height_*sizeof(glasscaustics::Texel);
    output=budget;return true;
}
bool openGlReceiverReadbackBytes(const glasscaustics::DirectGratingTerminalReceiverReadback& value,std::uint64_t& bytes)
{
    std::uint64_t rows=0;
    return renderpassoutput::detail::checkedMultiply(value.rows.capacity(),sizeof(glasscaustics::DirectGratingTerminalReceiverRow),rows)
        &&renderpassoutput::detail::checkedAdd(bytes,sizeof(value)+rows+value.submission.frameIdentity.capacity()+1,bytes)
        &&bytes<=renderpassoutput::kMaximumTotalBytes;
}

}

bool OpenGlFixtureSceneFrame::reconcileNativeAccounting(std::string& error) const
{
    if(!gratingStagingBinding&&!nativeAttemptCount
        &&std::none_of(nativeAttempts.begin(),nativeAttempts.end(),[](const auto& attempt){return attempt.ordinal!=0;}))return true;
    OpenGlNativeAttemptTotals measured;
    if(!openGlAttemptTotals(*this,measured,error)
        ||std::tie(measured.direct,measured.camera,measured.native,measured.cameraAdjunct,measured.cell,measured.coverage,
            measured.terminal,measured.terminalDeposition,measured.cohortDeposition,measured.bytes,measured.nativeDeposition)
          !=std::tie(directGratingWork,cameraGlassTraceWork,gratingNativeWork,gratingCameraWork,gratingCellWork,gratingCoverageWork,
            terminalAttemptWork,terminalAttemptDepositionWork,directCohortDepositionWork,nativeAttemptBytes,gratingNativeDepositionWork))
    {error="OpenGL native cumulative counters differ from retained frozen attempts";return false;}
    if(measured.cameraCoverageWork!=cameraCoverageWork)
    {error="OpenGL camera coverage arithmetic counters differ from permanent attempts";return false;}
    std::uint64_t directBytes=activeTerminalStorage+activeReceiverStorage+activeCameraCoverageStorage,cameraBytes=0,queryBytes=0;
    const auto add=[](std::uint64_t& value,std::uint64_t amount){return renderpassoutput::detail::checkedAdd(value,amount,value);};
    if(activeTerminalStorage&&!activeDirectTerminalLease){error="OpenGL terminal storage has no live registration";return false;}
    if(activeReceiverStorage&&!activeDirectReceiverLease){error="OpenGL receiver storage has no live registration";return false;}
    if(activeCameraCoverageStorage&&!activeCameraCoverageLease){error="OpenGL coverage storage has no live registration";return false;}
    for(auto* coverage=cameraCoverageResults.get();coverage;coverage=coverage->previous.get())
        if(!coverage->result||coverage->result->owner()!=this||!add(directBytes,coverage->ownedBytes))return false;
    for(auto* coverage=failedCameraCoverageResources.get();coverage;coverage=coverage->previous.get())
        if(!add(directBytes,coverage->ownedBytes))return false;
    for(auto* receiver=directReceiverResults.get();receiver;receiver=receiver->previous.get())
        if(!receiver->result||receiver->result->owner()!=this||!add(directBytes,receiver->ownedBytes))return false;
    for(auto* receiver=failedReceiverResources.get();receiver;receiver=receiver->previous.get())
        if(!receiver->program||(receiver->result&&(receiver->result->owner()!=this||receiver->result->retained().nativeResult))
            ||!add(directBytes,receiver->ownedBytes))return false;
    for(auto* cohort=directGratingCohort.get();cohort;cohort=cohort->previous.get())if(!add(directBytes,cohort->ownedBytes))return false;
    for(auto* batch=directGratingBatch.get();batch;batch=batch->previous.get())if(!add(directBytes,batch->ownedBytes))return false;
    for(auto* owner=directTerminalResults.get();owner;owner=owner->previous.get())if(!owner->result||owner->result->owner()!=this||!add(directBytes,owner->ownedBytes))return false;
    for(auto* batch=cameraGlassTraceBatch.get();batch;batch=batch->previous.get())if(!add(cameraBytes,batch->ownedBytes))return false;
    for(std::size_t i=0;i<queryPrograms.size();++i)if(queryPrograms[i]&&queryEvidence[i])
        if(!add(queryBytes,queryPrograms[i]->retainedBytes+sizeof(OpenGlQueryArithmeticEvidence)+128
            +queryEvidence[i]->submission.frameIdentity.capacity()+1))return false;
    if(directBytes!=directGratingBytes||cameraBytes!=cameraGlassTraceBytes||queryBytes!=queryRetainedBytes)
    {error="OpenGL native retained owner storage differs from its cumulative counters";return false;}
    std::array<std::uint64_t,3> adjunctStorage{};std::array<const void*,3> adjunctOwners{};
    for(std::size_t i=0;i<nativeAttemptCount;++i)for(unsigned slot=0;slot<3;++slot)if(nativeAttempts[i].adjunctOwners[slot]) {
        adjunctOwners[slot]=nativeAttempts[i].adjunctOwners[slot];adjunctStorage[slot]=nativeAttempts[i].adjunctStorage[slot];
    }
    if(adjunctStorage!=std::array<std::uint64_t,3>{gratingCameraSpectrumBytes,gratingCellSourceBytes,gratingCoverageBytes}
        ||adjunctOwners!=std::array<const void*,3>{gratingCameraSpectrum.get(),gratingCellSourceUpload.get(),gratingCoverageEvidence.payload.get()})
    {error="OpenGL completed native adjunct storage lost its registered owner reservation";return false;}
    error.clear();return true;
}
bool OpenGlFixtureSceneFrame::reserveNativeAttempt(OpenGlNativeAttemptKind kind,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,std::size_t count,unsigned adjuncts,std::size_t leaves,
    std::uint64_t maximumBytes,std::string& error,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& terminal) const
{
    OpenGlNativeAttemptTotals before,after;
    if(nativeAttemptCount==nativeAttempts.size()||!reconcileNativeAccounting(error)||!openGlAttemptTotals(*this,before,error))
    {error="OpenGL native attempt registry or retained accounting refuses admission";return false;}
    after=before;
    OpenGlNativeAttempt attempt{kind,binding,cohort,this,ownerContext,rendererGeneration_,staticResources->queryContext->generation,
        nativeAttemptCount+1,gratingStagingBinding?gratingStagingSubmission.submissionSerial
            :causticReadback?causticReadback->deposition.submission.submissionSerial:0,count,leaves,adjuncts};
    attempt.terminal=terminal;
    if(!openGlAddAttempt(*this,attempt,after,error)||after.bytes<before.bytes
        ||maximumBytes>renderpassoutput::kMaximumTotalBytes||after.bytes-before.bytes>renderpassoutput::kMaximumTotalBytes-maximumBytes)
    {error="OpenGL retained native attempt owner exceeds storage admission";return false;}
    const auto& base=gratingStagingBinding?*gratingStagingBinding:*binding;
    std::uint64_t work=base.bounds.combinedWork,deposition=base.frame.bounds.deposition.depositionWork;
    const auto add=[](std::uint64_t& value,std::uint64_t amount){return renderpassoutput::detail::checkedAdd(value,amount,value);};
    for(auto amount:{after.direct,after.camera,after.native,after.cameraAdjunct,after.cell,after.coverage})if(!add(work,amount))return false;
    if(after.terminal>after.direct)return false;
    for(auto amount:{base.bounds.orderWork,after.cohortDeposition,after.terminalDeposition,after.nativeDeposition,
        after.direct-after.terminal,after.camera,after.cameraAdjunct,after.cell,after.coverage})if(!add(deposition,amount))return false;
    if(work>glasscaustics::kMaximumQueryWork||deposition>glasscaustics::kMaximumDepositionWork)
    {error="OpenGL retained native attempts exceed cumulative work or deposition cap";return false;}
    // Fixed-slot publication cannot allocate. Work and frozen owner retention
    // never unwind, including failure before the first dispatch.
    nativeAttempts[nativeAttemptCount++]=std::move(attempt);
    directGratingWork=after.direct;cameraGlassTraceWork=after.camera;gratingNativeWork=after.native;
    gratingCameraWork=after.cameraAdjunct;gratingCellWork=after.cell;gratingCoverageWork=after.coverage;
    terminalAttemptWork=after.terminal;terminalAttemptDepositionWork=after.terminalDeposition;
    directCohortDepositionWork=after.cohortDeposition;nativeAttemptBytes=after.bytes;gratingNativeDepositionWork=after.nativeDeposition;
    error.clear();return true;
}

bool OpenGlFixtureSceneFrame::beginDirectCohortTerminalLease(const glasscaustics::SubmissionIdentity& expected,double time,
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sources,
    std::unique_ptr<glasscaustics::NativeDirectGratingTerminalLease>& output,
    glasscaustics::DirectGratingTerminalNativeBudget& budget,std::string& error) const
{
    if(directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease){error="OpenGL terminal operation is already leased on this thread";return false;}
    std::unique_lock<std::mutex> guard(fixtureMutex(),std::defer_lock);
    if(directTerminalEntryFrame==this&&directTerminalEntryLock&&directTerminalEntryLock->owns_lock())guard=std::move(*directTerminalEntryLock);
    else guard.lock();
    if(activeDirectTerminalLease||activeDirectReceiverLease||!directGratingCohort||!directGratingCohort->binding
        ||!(directGratingCohort->submission==expected)||!std::isfinite(time)||time!=gratingCameraEvaluationTime
        ||sources.size()!=directGratingCohort->sources.size()||sources.empty()||sources.size()>HarmonicMIDI::grid::Visual3DScene::kMaxLights
        ||!staticResources||!staticResources->queryContext||ownerContext!=glfwGetCurrentContext())
    {error="OpenGL terminal begin requires its exact current common cohort";return false;}
    glasscaustics::DirectGratingTerminalNativeBudget native;
    if(!openGlTerminalNativeBudget(*this,native,error))return false;
    const auto overhead=sizeof(OpenGlDirectTerminalLease)+sizeof(OpenGlDirectTerminalRecord)+directGratingCohort->binding->bounds().bytes;
    if(overhead>renderpassoutput::kMaximumTotalBytes||native.retainedBytes>renderpassoutput::kMaximumTotalBytes-overhead)
    {error="OpenGL terminal lease registration exceeds storage cap";return false;}
    auto lease=std::make_unique<OpenGlDirectTerminalLease>(*this,std::move(guard),sources,time);lease->budget=native;
    lease->budget.retainedBytes+=overhead;
    for(std::size_t i=0;i<sources.size();++i) {
        const auto* source=directGratingCohort->sources[i].get();const auto* batch=source?source->batch.get():nullptr;
        if(!batch||!batch->program||!batch->evidence){error="OpenGL terminal source has no completed native batch";return false;}
        lease->seals[i]={source,batch,batch->evidence.get(),{batch->program->directBuffers[0],batch->program->directBuffers[1],
            batch->program->directBuffers[2],batch->sourceBuffer,batch->queryTexture},batch->program->program,batch->program->directFence,
            batch->upload.generation,source->sourceUpload.bindingGeneration};
    }
    activeDirectTerminalLease=lease.get();
    directTerminalThreadLease=lease.get();
    if(!lease->live(error))return false;
    budget=lease->budget;output=std::move(lease);error.clear();return true;
}
bool OpenGlFixtureSceneFrame::reserveDirectCohortTerminalLease(glasscaustics::NativeDirectGratingTerminalLease& token,
    const glasscaustics::DirectGratingTerminalOrderBounds& bounds,const glasscaustics::DirectGratingTerminalOrderLedger& incumbent,std::string& error) const
{
    auto* lease=dynamic_cast<OpenGlDirectTerminalLease*>(&token);
    if(!lease||&lease->frame!=this||activeDirectTerminalLease!=lease||lease->reserved||!lease->live(error))
    {error="OpenGL terminal reserve has no registered current lease";return false;}
    glasscaustics::DirectGratingTerminalNativeBudget native;glasscaustics::DirectGratingTerminalOrderBounds exact;
    if(!openGlTerminalNativeBudget(*this,native,error))return false;
    const auto ownerBytes=lease->cohort->binding->bounds().bytes;
    native.retainedBytes+=sizeof(OpenGlDirectTerminalLease)+sizeof(OpenGlDirectTerminalRecord)+ownerBytes;
    if(std::tie(native.retainedBytes,native.combinedWork,native.depositionWork)
            !=std::tie(lease->budget.retainedBytes,lease->budget.combinedWork,lease->budget.depositionWork)
        ||!glasscaustics::calculateDirectGratingTerminalOrderBounds(*lease->cohort->binding,lease->sources,native,incumbent,exact,error)
        ||std::tie(bounds.bytes,bounds.combinedWork,bounds.depositionWork,bounds.maximumBands,bounds.maximumOrders)
            !=std::tie(exact.bytes,exact.combinedWork,exact.depositionWork,exact.maximumBands,exact.maximumOrders))
    {error="OpenGL terminal reservation differs from reconstructed native capacities";return false;}
    const auto work=bounds.maximumBands+bounds.maximumOrders;
    const auto bytes=bounds.bytes-native.retainedBytes+sizeof(OpenGlDirectTerminalLease)+sizeof(OpenGlDirectTerminalRecord);
    if(!reserveNativeAttempt(OpenGlNativeAttemptKind::Terminal,gratingStagingBinding,lease->cohort->binding,0,0,0,bounds.bytes-ownerBytes,error))return false;
    // No native/source/result copy or optical evaluation precedes this charge.
    lease->reservation=bounds;lease->reserved=true;directGratingBytes+=bytes;activeTerminalStorage=bytes;
    lease->workAfter=directGratingWork;lease->depositionAfter=terminalAttemptDepositionWork;
    lease->staged=std::make_unique<OpenGlDirectTerminalRecord>();
    lease->staged->ownedBytes=bytes;lease->staged->work=work;lease->staged->depositionWork=bounds.maximumOrders;
    error.clear();return true;
}
bool OpenGlFixtureSceneFrame::finishDirectCohortTerminalLease(glasscaustics::NativeDirectGratingTerminalLease& token,
    const glasscaustics::DirectGratingTerminalOrderLedger& candidate,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& output,std::string& error) const
{
    auto* lease=dynamic_cast<OpenGlDirectTerminalLease*>(&token);
    if(!lease||&lease->frame!=this||activeDirectTerminalLease!=lease||!lease->reserved||!lease->staged||lease->staged->result
        ||!lease->live(error)||candidate.nativeResult||candidate.incident.cohort!=lease->cohort->binding
        ||!(candidate.incident.submission==lease->cohort->submission)
        ||std::tie(candidate.bounds.bytes,candidate.bounds.combinedWork,candidate.bounds.depositionWork,candidate.bounds.maximumBands,candidate.bounds.maximumOrders)
            !=std::tie(lease->reservation.bytes,lease->reservation.combinedWork,lease->reservation.depositionWork,lease->reservation.maximumBands,lease->reservation.maximumOrders))
    {error="OpenGL terminal finish has no exact reserved candidate";return false;}
    std::uint64_t bytes=sizeof(OpenGlDirectTerminalRecord)+sizeof(glasscaustics::NativeDirectGratingTerminalResult)+128;
    if(!openGlTerminalLedgerBytes(candidate,bytes)||!openGlTerminalLedgerBytes(candidate,bytes)||bytes>lease->staged->ownedBytes)
    {error="OpenGL terminal returned and staged capacities exceed native reservation";return false;}
    lease->candidate=&candidate;lease->staged->result=makeDirectCohortTerminalResult(candidate);
    bytes=sizeof(OpenGlDirectTerminalRecord)+sizeof(glasscaustics::NativeDirectGratingTerminalResult)+128;
    if(!openGlTerminalLedgerBytes(candidate,bytes)||!openGlTerminalLedgerBytes(lease->staged->result->retained(),bytes)
        ||bytes>lease->staged->ownedBytes||!openGlSameTerminalLedger(candidate,lease->staged->result->retained()))
    {error="OpenGL terminal actual sealed storage or payload changed";return false;}
    output=lease->staged->result;error.clear();return true;
}
bool OpenGlFixtureSceneFrame::validateDirectCohortTerminalResult(const glasscaustics::NativeDirectGratingTerminalLease& token,
    const glasscaustics::DirectGratingTerminalOrderLedger& candidate,std::string& error) const
{
    const auto* lease=dynamic_cast<const OpenGlDirectTerminalLease*>(&token);
    if(!lease||&lease->frame!=this||activeDirectTerminalLease!=lease||!lease->reserved||!lease->staged
        ||lease->candidate!=&candidate||!candidate.nativeResult||candidate.nativeResult!=lease->staged->result
        ||candidate.nativeResult->owner()!=this||candidate.nativeResult->retained().nativeResult||!lease->live(error)
        ||directGratingWork!=lease->workAfter||terminalAttemptDepositionWork!=lease->depositionAfter
        ||!openGlSameTerminalLedger(candidate,candidate.nativeResult->retained()))
    {error="OpenGL terminal result is outside its registered staged native operation";return false;}
    for(auto* old=directTerminalResults.get();old;old=old->previous.get())if(old->result==candidate.nativeResult)
    {error="OpenGL terminal result reused an incumbent sealed owner";return false;}
    if(!validateDirectCohortProductionQueries(lease->cohort->submission,candidate.sources,error))return false;
    glasscaustics::DirectGratingTerminalNativeBudget actual;
    std::uint64_t bytes=sizeof(OpenGlDirectTerminalRecord)+sizeof(glasscaustics::NativeDirectGratingTerminalResult)+128;
    if(!openGlTerminalNativeBudget(*this,actual,error)||actual.retainedBytes!=lease->reservation.bytes
        ||actual.combinedWork!=lease->reservation.combinedWork||actual.depositionWork!=lease->reservation.depositionWork
        ||!openGlTerminalLedgerBytes(candidate,bytes)||!openGlTerminalLedgerBytes(candidate.nativeResult->retained(),bytes)
        ||bytes>lease->staged->ownedBytes)
    {error="OpenGL terminal registered native counters or actual result capacities changed";return false;}
    error.clear();return true;
}

bool OpenGlFixtureSceneFrame::beginDirectTerminalReceiverLease(
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& terminal,
    const glasscaustics::SubmissionIdentity& expected,double time,
    std::unique_ptr<glasscaustics::NativeDirectGratingTerminalReceiverLease>& output,
    glasscaustics::DirectGratingTerminalReceiverNativeBudget& budget,std::string& error) const
{
    if(directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease){error="OpenGL receiver cannot nest a native lease";return false;}
    std::unique_lock<std::mutex> guard(fixtureMutex(),std::defer_lock);
    if(directTerminalEntryFrame==this&&directTerminalEntryLock&&directTerminalEntryLock->owns_lock())guard=std::move(*directTerminalEntryLock);
    else guard.lock();
    if(activeDirectTerminalLease||activeDirectReceiverLease||!terminal||terminal->owner()!=this
        ||!directGratingCohort||!directGratingCohort->binding||directGratingCohort->binding->sources().empty()
        ||terminal->retained().incident.cohort!=directGratingCohort->binding
        ||!(expected==directGratingCohort->submission)||!(expected==terminal->retained().incident.submission)
        ||!std::isfinite(time)||time!=gratingCameraEvaluationTime||!staticResources||!staticResources->queryContext
        ||ownerContext!=glfwGetCurrentContext()||nativeAttemptCount==nativeAttempts.size())
    {error="OpenGL receiver begin requires its exact current scattering frame";return false;}
    bool registered=false;for(auto* record=directTerminalResults.get();record;record=record->previous.get())if(record->result==terminal)registered=true;
    if(!registered||!validateDirectCohortProductionQueries(expected,terminal->retained().sources,error))return false;
    glasscaustics::DirectGratingTerminalReceiverNativeBudget native;
    if(!openGlReceiverBudget(*this,*terminal,native,error))return false;
    const auto generation=nextQueryOwnerGeneration();if(!generation)return false;
    auto lease=std::make_unique<OpenGlDirectTerminalReceiverLease>(*this,std::move(guard),terminal,time,generation);
    lease->budget=native;
    for(std::size_t i=0;i<directGratingCohort->sources.size();++i) {
        const auto* source=directGratingCohort->sources[i].get();const auto* batch=source?source->batch.get():nullptr;
        if(!batch||!batch->program||!batch->evidence)return false;
        lease->seals[i]={source,batch,batch->evidence.get(),{batch->program->directBuffers[0],batch->program->directBuffers[1],
            batch->program->directBuffers[2],batch->sourceBuffer,batch->queryTexture},batch->program->program,batch->program->directFence,
            batch->upload.generation,source->sourceUpload.bindingGeneration};
    }
    activeDirectReceiverLease=lease.get();directReceiverThreadLease=lease.get();
    if(!lease->live(error))return false;
    budget=native;output=std::move(lease);error.clear();return true;
}
bool OpenGlFixtureSceneFrame::reserveDirectTerminalReceiverLease(glasscaustics::NativeDirectGratingTerminalReceiverLease& token,
    const glasscaustics::DirectGratingTerminalReceiverBounds& bounds,
    const glasscaustics::DirectGratingTerminalReceiverReadback& incumbent,std::string& error) const
{
    auto* lease=dynamic_cast<OpenGlDirectTerminalReceiverLease*>(&token);
    if(!lease||&lease->frame!=this||lease->reserved||!lease->live(error))return false;
    glasscaustics::DirectGratingTerminalReceiverNativeBudget native;
    glasscaustics::DirectGratingTerminalReceiverBounds exact;
    if(!openGlReceiverBudget(*this,*lease->terminal,native,error)
        ||std::tie(native.retainedBytes,native.combinedWork,native.nativeInputBytesPerRow,native.nativeOutputBytesPerRow,native.nativeFixedBytes)
          !=std::tie(lease->budget.retainedBytes,lease->budget.combinedWork,lease->budget.nativeInputBytesPerRow,
            lease->budget.nativeOutputBytesPerRow,lease->budget.nativeFixedBytes)
        ||!glasscaustics::calculateDirectGratingTerminalReceiverBounds(*lease->terminal,lease->frozenFrame(),native,incumbent,exact,error)
        ||std::tie(bounds.bytes,bounds.combinedWork,bounds.queryWork,bounds.maximumRows,bounds.perRayWork,
            bounds.nativeInputBytes,bounds.nativeOutputBytes,bounds.nativeFixedBytes)
          !=std::tie(exact.bytes,exact.combinedWork,exact.queryWork,exact.maximumRows,exact.perRayWork,
            exact.nativeInputBytes,exact.nativeOutputBytes,exact.nativeFixedBytes))
    {error="OpenGL receiver reservation differs from its native capacities";return false;}
    glasscaustics::DirectGratingTerminalNativeBudget before;
    if(!openGlTerminalNativeBudget(*this,before,error))return false;
    const auto journal=lease->terminal->retained().bounds.bytes;
    if(bounds.bytes<before.retainedBytes+journal
        ||!reserveNativeAttempt(OpenGlNativeAttemptKind::Receiver,gratingStagingBinding,lease->cohort->binding,
            bounds.maximumRows,0,0,bounds.bytes-journal,error,lease->terminal))return false;
    // No input, frozen-cohort copy, result, shader or native allocation precedes
    // this permanent fixed-slot publication and full worst-query charge.
    lease->reservation=bounds;lease->reserved=true;lease->workAfter=gratingNativeWork;
    activeReceiverStorage=bounds.bytes-before.retainedBytes-journal;
    directGratingBytes+=activeReceiverStorage;
    lease->staged=std::make_unique<OpenGlDirectTerminalReceiverRecord>();
    lease->staged->ownedBytes=activeReceiverStorage;lease->staged->attemptOrdinal=nativeAttemptCount;
    lease->staged->inputGeneration=lease->generation;lease->staged->reservation=bounds;
    if(directReceiverStop==OpenGlTerminalReceiverStopForFixture::Reserved)
    {error="OpenGL receiver fixture refused after persistent reservation";return false;}
    error.clear();return true;
}
namespace
{
#if !defined(__APPLE__)
struct OpenGlReceiverState
{
    const arbitgl::GlFuncs& gl;
    OpenGlFixtureState fixture;
    std::array<int,3> buffers{};std::array<GLint64,3> starts{},sizes{};
    int generic=0,packBuffer=0;std::array<int,6> pack{};
    const std::array<GLenum,6> names{GL_PACK_ALIGNMENT,GL_PACK_ROW_LENGTH,GL_PACK_SKIP_PIXELS,GL_PACK_SKIP_ROWS,GL_PACK_SWAP_BYTES,GL_PACK_LSB_FIRST};
    PFNGLBINDBUFFERRANGEPROC bindRange=nullptr;
    OpenGlReceiverState(const arbitgl::GlFuncs& functions):gl(functions),fixture(functions) {
        bindRange=reinterpret_cast<PFNGLBINDBUFFERRANGEPROC>(glfwGetProcAddress("glBindBufferRange"));
        const auto indexed=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
        glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&generic);glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING,&packBuffer);
        for(unsigned i=0;i<3;++i) {
            gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,i,&buffers[i]);
            if(indexed){indexed(GL_SHADER_STORAGE_BUFFER_START,i,&starts[i]);indexed(GL_SHADER_STORAGE_BUFFER_SIZE,i,&sizes[i]);}
        }
        for(unsigned i=0;i<pack.size();++i)glGetIntegerv(names[i],&pack[i]);
    }
    ~OpenGlReceiverState() {
        for(unsigned i=0;i<3;++i) {
            if(buffers[i]&&sizes[i]>0&&bindRange)bindRange(GL_SHADER_STORAGE_BUFFER,i,unsigned(buffers[i]),starts[i],sizes[i]);
            else gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,i,unsigned(buffers[i]));
        }
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(generic));gl.BindBuffer(GL_PIXEL_PACK_BUFFER,unsigned(packBuffer));
        for(unsigned i=0;i<pack.size();++i){glPixelStorei(names[i],pack[i]);}fixture.restore();
    }
    void packHost() const {
        gl.BindBuffer(GL_PIXEL_PACK_BUFFER,0);for(unsigned i=0;i<pack.size();++i)glPixelStorei(names[i],i?0:1);
    }
};
bool openGlInspectReceiver(const OpenGlFixtureSceneFrame& frame,const OpenGlDirectTerminalReceiverRecord& record,
    const glasscaustics::DirectGratingTerminalReceiverReadback& candidate,std::string& error)
{
    namespace gc=glasscaustics;
    const auto refuse=[&](){error="OpenGL receiver native buffer, fence, payload or raster changed";return false;};
    const auto& gl=frame.staticResources->gl;OpenGlReceiverState state(gl);
    const auto* program=record.program.get();
    const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
    const auto syncStatus=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
    const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
    const auto bufferSize=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    const auto label=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
    const auto uniformI=reinterpret_cast<PFNGLGETUNIFORMIVPROC>(glfwGetProcAddress("glGetUniformiv"));
    const auto uniformF=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
    if(!program||!program->liveUnlocked()||program->context!=frame.staticResources->queryContext||!program->identity
        ||program->identity->stageCount!=1||program->identity->stages[0].type!=GL_COMPUTE_SHADER
        ||program->identity->stages[0].source!=gc::opengl::directTerminalReceiverComputeSource()
        ||!isSync||!syncStatus||!isBuffer||!bufferSize||!label||!uniformI||!uniformF
        ||!program->directFence||!isSync(program->directFence)||!candidate.terminal
        ||candidate.terminal->owner()!=&frame||!candidate.inputGeneration||!record.attemptOrdinal
        ||record.attemptOrdinal>frame.nativeAttemptCount)return refuse();
    const auto& terminal=candidate.terminal->retained();const auto& attempt=frame.nativeAttempts[record.attemptOrdinal-1];
    const auto& a=candidate.bounds;const auto& b=record.reservation;
    if(candidate.inputGeneration!=record.inputGeneration||!(candidate.submission==terminal.incident.submission)
        ||candidate.evaluationTime!=terminal.incident.cohort->sources()[0]->evaluationTime
        ||std::tie(a.bytes,a.combinedWork,a.queryWork,a.maximumRows,a.perRayWork,a.nativeInputBytes,a.nativeOutputBytes,a.nativeFixedBytes)
          !=std::tie(b.bytes,b.combinedWork,b.queryWork,b.maximumRows,b.perRayWork,b.nativeInputBytes,b.nativeOutputBytes,b.nativeFixedBytes))return refuse();
    if(attempt.kind!=OpenGlNativeAttemptKind::Receiver||attempt.terminal!=candidate.terminal
        ||attempt.cohort!=terminal.incident.cohort||attempt.count!=candidate.bounds.maximumRows
        ||attempt.ordinal!=record.attemptOrdinal)return refuse();
    int signalled=0;syncStatus(program->directFence,GL_SYNC_STATUS,1,nullptr,&signalled);if(signalled!=GL_SIGNALED)return refuse();
    const auto& frozen=terminal.incident.cohort->sources()[0]->frame;
    const auto location=[&](const char* name){return gl.GetUniformLocation(program->program,name);};
    const auto integer=[&](const char* name,int expected) {
        int actual=-1;const auto loc=location(name);if(loc<0)return false;uniformI(program->program,loc,&actual);return actual==expected;
    };
    const auto floats=[&](const char* name,const float* expected,unsigned count) {
        std::array<float,4> actual{};const auto loc=location(name);if(loc<0)return false;uniformF(program->program,loc,actual.data());
        return std::equal(actual.begin(),actual.begin()+count,expected);
    };
    const auto camera=gc::cameraPosition(frozen);const auto& q=frozen.camera.transform.rotation;
    const gc::Texel position{camera[0],camera[1],camera[2],0},rotation{q.x,q.y,q.z,q.w};
    const gc::Texel projection{std::tan(frozen.camera.verticalFovRadians*.5f),float(frozen.width)/frozen.height,frozen.camera.nearPlane,frozen.camera.farPlane};
    const gc::Texel extent{float(frozen.width),float(frozen.height),0,0};
    if(!integer("uRowCount",int(terminal.orders.size()))||!integer("uTriangleCount",int(frozen.triangles.size()))
        ||!integer("uOpaqueVisibility",0)||!integer("uRasterOwners",1)||!integer("uRasterTriangle",2)
        ||!floats("uOpaqueControl",frozen.query.control.data(),4)||!floats("uCamera",position.data(),4)
        ||!floats("uRotation",rotation.data(),4)||!floats("uProjection",projection.data(),4)||!floats("uExtent",extent.data(),2))return refuse();
    const auto inspectBuffer=[&](unsigned slot,const void* expected,std::size_t bytes) {
        const auto id=program->directBuffers[slot];if(!id||!isBuffer(id))return false;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,id);GLint64 actual=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
        std::array<char,96> name{};int length=0;label(GL_BUFFER,id,int(name.size()),&length,name.data());
        if(actual!=GLint64(bytes)||name!=program->label)return false;
        const void* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes),GL_MAP_READ_BIT);
        if(!mapped){return false;}const bool same=std::memcmp(mapped,expected,bytes)==0;
        return gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)==GL_TRUE&&same;
    };
    const auto count=std::max<std::size_t>(1,terminal.orders.size());
    if(record.inputs.size()!=count||record.packed.size()!=count||record.inputs.capacity()>2*candidate.bounds.maximumRows
        ||record.packed.capacity()>2*candidate.bounds.maximumRows
        ||!inspectBuffer(0,record.inputs.data(),count*sizeof(record.inputs[0]))
        ||!inspectBuffer(1,record.packed.data(),count*sizeof(record.packed[0]))
        ||record.nativeBytes!=count*(sizeof(record.inputs[0])+sizeof(record.packed[0])))return refuse();
    gc::DirectGratingTerminalReceiverReadback decoded;
    decoded.terminal=candidate.terminal;decoded.submission=candidate.submission;decoded.evaluationTime=candidate.evaluationTime;
    decoded.inputGeneration=candidate.inputGeneration;decoded.bounds=candidate.bounds;decoded.rows.reserve(candidate.bounds.maximumRows);
    for(std::size_t i=0;i<terminal.orders.size();++i) {
        gc::NativeDirectGratingTerminalReceiverInput exact;gc::DirectGratingTerminalReceiverRow row;
        if(!gc::makeDirectGratingTerminalReceiverInput(terminal,frozen,i,exact,error)
            ||!gc::sameDirectGratingTerminalReceiverInput(exact,record.inputs[i])
            ||!gc::opengl::decodeDirectTerminalReceiver(record.packed[i],record.inputs[i],terminal.orders[i],row))return refuse();
        decoded.queryWork+=row.queryWork;decoded.rows.push_back(row);
    }
    if(!gc::sameDirectGratingTerminalReceiverReadback(decoded,candidate,false)
        ||!gc::validateDirectGratingTerminalReceiverRows(terminal,frozen,decoded,error))return refuse();
    // Read the actual private raster, independent of the outgoing native query.
    // This comparison never substitutes a CPU ray for native nearest ownership.
    std::array<std::vector<gc::Texel>,2> raster;
    const auto pixels=std::size_t(frame.width_)*frame.height_;state.packHost();
    for(unsigned slot=0;slot<2;++slot) {
        raster[slot].resize(pixels);if(raster[slot].capacity()>2*pixels)return refuse();
        glBindTexture(GL_TEXTURE_2D,frame.causticTextures[slot+1]);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,raster[slot].data());
    }
    for(std::size_t i=0;i<decoded.rows.size();++i) {
        const auto& row=decoded.rows[i];if(!row.pixel.projected)continue;
        const auto index=(std::size_t(frame.height_)-1-row.pixel.y)*frame.width_+row.pixel.x;
        if(row.rasterSample==gc::DirectGratingTerminalReceiverRow::RasterSample::Background) {
            if(raster[0][index]!=gc::Texel{}||raster[1][index]!=gc::Texel{})return refuse();
        } else if(raster[0][index]!=record.packed[i].data[6]||raster[1][index]!=record.packed[i].data[7])return refuse();
    }
    if(!frame.validateDirectCohortProductionQueries(candidate.submission,terminal.sources,error)||glGetError()!=GL_NO_ERROR)return refuse();
    std::uint64_t bytes=sizeof(record)+sizeof(gc::NativeDirectGratingTerminalReceiverResult)+128+program->retainedBytes
        +record.inputs.capacity()*sizeof(record.inputs[0])+record.packed.capacity()*sizeof(record.packed[0]);
    if(!openGlReceiverReadbackBytes(candidate,bytes)||!openGlReceiverReadbackBytes(decoded,bytes)
        ||(record.result&&!openGlReceiverReadbackBytes(record.result->retained(),bytes))||bytes>record.ownedBytes)return refuse();
    error.clear();return true;
}
#endif
}
bool OpenGlFixtureSceneFrame::dispatchDirectTerminalReceivers(glasscaustics::NativeDirectGratingTerminalReceiverLease& token,
    const std::vector<glasscaustics::NativeDirectGratingTerminalReceiverInput>& inputs,
    glasscaustics::DirectGratingTerminalReceiverReadback& candidate,std::string& error) const
{
#if defined(__APPLE__)
    (void)token;(void)inputs;(void)candidate;error="OpenGL receiver dispatch requires GL4.3";return false;
#else
    namespace gc=glasscaustics;auto* lease=dynamic_cast<OpenGlDirectTerminalReceiverLease*>(&token);
    if(!lease||&lease->frame!=this||!lease->reserved||!lease->staged||lease->staged->program||!lease->live(error)
        ||inputs.size()!=lease->terminal->retained().orders.size()||inputs.capacity()>2*lease->reservation.maximumRows
        ||candidate.terminal!=lease->terminal||candidate.inputGeneration!=lease->generation||candidate.nativeResult)return false;
    const auto& frozen=lease->frozenFrame();const auto& gl=staticResources->gl;
    const auto bufferSize=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    const auto label=reinterpret_cast<PFNGLOBJECTLABELPROC>(glfwGetProcAddress("glObjectLabel"));
    const auto indexedSize=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
    if(!bufferSize||!label||!indexedSize)return false;
    auto& record=*lease->staged;const auto count=std::max<std::size_t>(1,inputs.size());
    record.inputs.resize(count);record.packed.resize(count);
    if(record.inputs.capacity()>2*lease->reservation.maximumRows||record.packed.capacity()>2*lease->reservation.maximumRows)return false;
    std::copy(inputs.begin(),inputs.end(),record.inputs.begin());
    for(std::size_t i=0;i<inputs.size();++i) {
        gc::NativeDirectGratingTerminalReceiverInput exact;
        if(!gc::makeDirectGratingTerminalReceiverInput(lease->terminal->retained(),frozen,i,exact,error)
            ||!gc::sameDirectGratingTerminalReceiverInput(exact,inputs[i]))return false;
    }
    OpenGlReceiverState state(gl);const auto source=gc::opengl::directTerminalReceiverComputeSource();
    if(source.capacity()>kQueryProgramSourceReservation/8)return false;
    record.program=linkQueryArithmeticProgram(gl,staticResources->queryContext,rendererGeneration_,{{GL_COMPUTE_SHADER,source}},error);
    if(!record.program)return false;
    gl.GenBuffers(2,record.program->directBuffers.data());
    const std::array<std::uint64_t,2> sizes{count*sizeof(record.inputs[0]),count*sizeof(record.packed[0])};
    const void* payloads[]{record.inputs.data(),record.packed.data()};
    for(unsigned slot=0;slot<2;++slot) {
        const auto id=record.program->directBuffers[slot];if(!id)return false;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,id);label(GL_BUFFER,id,-1,record.program->label.data());
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(sizes[slot]),payloads[slot],slot?GL_DYNAMIC_READ:GL_STATIC_DRAW);
        GLint64 actual=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
        if(actual!=GLint64(sizes[slot])||glGetError()!=GL_NO_ERROR)return false;
        record.nativeBytes+=sizes[slot];record.program->retainedBytes+=sizes[slot];
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,id);
    }
    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,causticBuffers[4]);gl.UseProgram(record.program->program);
    const auto location=[&](const char* name){return gl.GetUniformLocation(record.program->program,name);};
    for(const auto* name:{"uRowCount","uTriangleCount","uOpaqueVisibility","uOpaqueControl","uRasterOwners","uRasterTriangle",
        "uCamera","uRotation","uProjection","uExtent"})if(location(name)<0)return false;
    gl.Uniform1i(location("uRowCount"),int(inputs.size()));gl.Uniform1i(location("uTriangleCount"),int(frozen.triangles.size()));
    gl.Uniform4fv(location("uOpaqueControl"),1,frozen.query.control.data());
    const auto camera=gc::cameraPosition(frozen);const auto& q=frozen.camera.transform.rotation;
    gl.Uniform4f(location("uCamera"),camera[0],camera[1],camera[2],0);gl.Uniform4f(location("uRotation"),q.x,q.y,q.z,q.w);
    gl.Uniform4f(location("uProjection"),std::tan(frozen.camera.verticalFovRadians*.5f),float(frozen.width)/frozen.height,frozen.camera.nearPlane,frozen.camera.farPlane);
    gl.Uniform2f(location("uExtent"),float(frozen.width),float(frozen.height));
    const char* samplers[]{"uOpaqueVisibility","uRasterOwners","uRasterTriangle"};
    for(unsigned slot=0;slot<3;++slot) {
        gl.ActiveTexture(GL_TEXTURE0+slot);gl.BindSampler(slot,0);glBindTexture(GL_TEXTURE_2D,causticTextures[slot]);gl.Uniform1i(location(samplers[slot]),int(slot));
        int actual=0;gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,slot,&actual);GLint64 start=0,size=0;
        indexedSize(GL_SHADER_STORAGE_BUFFER_START,slot,&start);indexedSize(GL_SHADER_STORAGE_BUFFER_SIZE,slot,&size);
        if(unsigned(actual)!=(slot<2?record.program->directBuffers[slot]:causticBuffers[4])||start!=0
            ||size!=GLint64(slot<2?sizes[slot]:frozen.triangleTexels.size()*sizeof(gc::Texel)))return false;
    }
    if(record.program->retainedBytes>record.ownedBytes)return false;
    lease->dispatched=true;
    gl.DispatchCompute(unsigned(std::max<std::size_t>(1,(inputs.size()+31)/32)),1,1);gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
    record.program->directFence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);if(!record.program->directFence)return false;
    const auto wait=gl.ClientWaitSync(record.program->directFence,GL_SYNC_FLUSH_COMMANDS_BIT,30000000000ull);
    if((wait!=GL_ALREADY_SIGNALED&&wait!=GL_CONDITION_SATISFIED)||glGetError()!=GL_NO_ERROR)return false;
    lease->complete=true;
    if(directReceiverStop==OpenGlTerminalReceiverStopForFixture::Fence)
    {error="OpenGL receiver fixture refused after completed native fence";return false;}
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,record.program->directBuffers[1]);
    const auto* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(sizes[1]),GL_MAP_READ_BIT);if(!mapped)return false;
    std::memcpy(record.packed.data(),mapped,sizes[1]);if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return false;
    candidate.rows.reserve(lease->reservation.maximumRows);
    for(std::size_t i=0;i<inputs.size();++i) {
        gc::DirectGratingTerminalReceiverRow row;
        if(!gc::opengl::decodeDirectTerminalReceiver(record.packed[i],inputs[i],lease->terminal->retained().orders[i],row))return false;
        candidate.queryWork+=row.queryWork;candidate.rows.push_back(row);
    }
    if(glGetError()!=GL_NO_ERROR)return false;
    return openGlInspectReceiver(*this,record,candidate,error);
#endif
}
bool OpenGlFixtureSceneFrame::finishDirectTerminalReceiverLease(glasscaustics::NativeDirectGratingTerminalReceiverLease& token,
    const glasscaustics::DirectGratingTerminalReceiverReadback& candidate,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>& output,std::string& error) const
{
    auto* lease=dynamic_cast<OpenGlDirectTerminalReceiverLease*>(&token);
    if(!lease||&lease->frame!=this||!lease->reserved||!lease->staged||lease->staged->result||candidate.nativeResult
        ||!validateDirectTerminalReceiverResult(token,candidate,error))return false;
    lease->staged->result=makeDirectTerminalReceiverResult(candidate);
    std::uint64_t bytes=sizeof(OpenGlDirectTerminalReceiverRecord)+sizeof(glasscaustics::NativeDirectGratingTerminalReceiverResult)+128
        +lease->staged->program->retainedBytes+lease->staged->inputs.capacity()*sizeof(lease->staged->inputs[0])
        +lease->staged->packed.capacity()*sizeof(lease->staged->packed[0]);
    if(!openGlReceiverReadbackBytes(candidate,bytes)||!openGlReceiverReadbackBytes(lease->staged->result->retained(),bytes)
        ||bytes>lease->staged->ownedBytes)return false;
    output=lease->staged->result;error.clear();return true;
}
bool OpenGlFixtureSceneFrame::validateDirectTerminalReceiverResult(const glasscaustics::NativeDirectGratingTerminalReceiverLease& token,
    const glasscaustics::DirectGratingTerminalReceiverReadback& candidate,std::string& error) const
{
#if defined(__APPLE__)
    (void)token;(void)candidate;error="OpenGL receiver inspection requires GL4.3";return false;
#else
    auto* lease=dynamic_cast<const OpenGlDirectTerminalReceiverLease*>(&token);
    if(!lease||&lease->frame!=this||!lease->reserved||!lease->complete||!lease->staged||!lease->live(error)
        ||candidate.terminal!=lease->terminal||!(candidate.submission==lease->cohort->submission)
        ||candidate.evaluationTime!=lease->time||candidate.inputGeneration!=lease->generation
        ||std::tie(candidate.bounds.bytes,candidate.bounds.combinedWork,candidate.bounds.queryWork,candidate.bounds.maximumRows,candidate.bounds.perRayWork,
            candidate.bounds.nativeInputBytes,candidate.bounds.nativeOutputBytes,candidate.bounds.nativeFixedBytes)
          !=std::tie(lease->reservation.bytes,lease->reservation.combinedWork,lease->reservation.queryWork,lease->reservation.maximumRows,lease->reservation.perRayWork,
            lease->reservation.nativeInputBytes,lease->reservation.nativeOutputBytes,lease->reservation.nativeFixedBytes)
        ||candidate.rows.capacity()>2*lease->reservation.maximumRows)return false;
    if(lease->staged->result) {
        if(candidate.nativeResult!=lease->staged->result||candidate.nativeResult->owner()!=this||candidate.nativeResult->retained().nativeResult
            ||!glasscaustics::sameDirectGratingTerminalReceiverReadback(candidate,candidate.nativeResult->retained(),false))return false;
    } else if(candidate.nativeResult)return false;
    glasscaustics::DirectGratingTerminalNativeBudget actual;
    if(!openGlTerminalNativeBudget(*this,actual,error)||actual.retainedBytes!=lease->reservation.bytes
        ||actual.combinedWork!=lease->reservation.combinedWork)return false;
    return openGlInspectReceiver(*this,*lease->staged,candidate,error);
#endif
}

namespace
{
thread_local OpenGlCameraCoverageStopForFixture cameraCoverageStop=OpenGlCameraCoverageStopForFixture::None;
class OpenGlCameraCoverageLease final : public glasscaustics::NativeCameraCellCoverageLease
{
public:
    const OpenGlFixtureSceneFrame& frame;
    std::unique_lock<std::mutex> guard;
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult> parent;
    const OpenGlDirectTerminalReceiverRecord* receiver=nullptr;
    glasscaustics::NativeCameraCellCoverageBudget budget;
    glasscaustics::CameraCellCoverageBounds reservation;
    std::unique_ptr<OpenGlCameraCoverageRecord> staged;
    std::uint64_t generation=0,ordinal=0;
    bool reserveAttempted=false,dispatched=false,finishAttempted=false,reserved=false,complete=false,published=false;
    mutable unsigned inspections=0;
    OpenGlCameraCoverageLease(const OpenGlFixtureSceneFrame& f,std::unique_lock<std::mutex>&& lock,
        std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult> p,
        const OpenGlDirectTerminalReceiverRecord* r,std::uint64_t g,std::uint64_t attempt)
        :frame(f),guard(std::move(lock)),parent(std::move(p)),receiver(r),generation(g),ordinal(attempt){}
    ~OpenGlCameraCoverageLease() override {
        if(!published&&!staged) {
            if(frame.directGratingBytes>=frame.activeCameraCoverageStorage)frame.directGratingBytes-=frame.activeCameraCoverageStorage;
        }
        if(!published&&staged) {
            if(staged->result) {
                staged->previous=std::move(frame.failedCameraCoverageResources);
                frame.failedCameraCoverageResources=std::move(staged);
            } else {
                // No result escaped. Only uncommitted storage unwinds; the
                // fixed attempt and every admitted work unit remain permanent.
                if(frame.directGratingBytes>=staged->ownedBytes)frame.directGratingBytes-=staged->ownedBytes;
                staged.reset();
            }
        }
        frame.activeCameraCoverageStorage=0;frame.activeCameraCoverageLease=nullptr;
        cameraCoverageThreadLease=nullptr;
    }
    const NativeFixtureSceneFrame& owner() const noexcept override{return frame;}
    const glasscaustics::Frame& frozenFrame() const noexcept override{return parent->retained().terminal->retained().incident.cohort->sources()[0]->frame;}
    std::uint64_t inputGeneration() const noexcept override{return parent->retained().inputGeneration;}
    std::uint64_t proofGeneration() const noexcept override{return generation;}
    bool live(std::string& error) const {
        if(!guard.owns_lock()||frame.activeCameraCoverageLease!=this||cameraCoverageThreadLease!=this
            ||!frame.staticResources||!frame.staticResources->queryContext||!frame.staticResources->queryContext->active
            ||frame.ownerContext!=glfwGetCurrentContext()||frame.staticResources->ownerContext!=frame.ownerContext
            ||frame.rendererGeneration_!=frame.staticResources->rendererGeneration||!parent||parent->owner()!=&frame
            ||!receiver||receiver->result!=parent||!frame.directGratingCohort
            ||parent->retained().terminal->retained().incident.cohort!=frame.directGratingCohort->binding
            ||parent->retained().evaluationTime!=frame.gratingCameraEvaluationTime
            ||!ordinal||ordinal>frame.nativeAttemptCount||frame.nativeAttempts[ordinal-1].receiver!=parent
            ||frame.nativeAttempts[ordinal-1].cameraCoverageReserved!=reserved||!generation||generation>UINT32_MAX)
        {error="OpenGL camera coverage lost its registered native lease";return false;}
        return frame.reconcileNativeAccounting(error);
    }
protected:
    void commit() noexcept override {
        staged->previous=std::move(frame.cameraCoverageResults);
        frame.cameraCoverageResults=std::move(staged);published=true;
    }
};
const OpenGlCameraCoverageRecord* openGlActiveCameraCoverageRecord(const OpenGlFixtureSceneFrame& frame)
{return frame.activeCameraCoverageLease?frame.activeCameraCoverageLease->staged.get():nullptr;}
#if !defined(__APPLE__)
bool openGlCoverageProgramIdentity(const OpenGlFixtureSceneFrame& frame,const std::shared_ptr<OpenGlQueryProgram>& program)
{
    if(!program||program->preciseSceneQuery||!program->registered||!program->identity||!program->liveUnlocked()
        ||program->context!=frame.staticResources->queryContext)return false;
    const auto& i=*program->identity;const auto& context=*program->context;
    if(i.contractVersion!=1||i.glslVersion!=430||i.context!=reinterpret_cast<std::uintptr_t>(frame.ownerContext)
        ||i.contextGeneration!=context.generation||!i.ownerGeneration||i.rendererGeneration!=frame.rendererGeneration_
        ||i.contextMajor!=std::uint32_t(context.major)||i.contextMinor!=std::uint32_t(context.minor)||i.contextProfile!=std::uint32_t(context.profile)
        ||i.linkedProgram!=program->program||!i.stageCount||i.stageCount>2)return false;
    bool found=false;for(auto* p=context.programs;p;p=p->next)if(p==program.get())found=true;
    if(!found)return false;
    for(unsigned stage=0;stage<i.stageCount;++stage)if(i.stages[stage].fingerprint!=querySourceFingerprint(i.stages[stage].source))return false;
    return true;
}
bool openGlRasterCaptureMatches(const OpenGlFixtureSceneFrame& frame,std::string& error)
{
    const auto refuse=[&](){error="OpenGL camera coverage lacks its exact historical raster capture";return false;};
    const auto& program=frame.rasterCaptureProgram;
    const auto get=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
    const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
    const auto status=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
    if(!openGlCoverageProgramIdentity(frame,program)
        ||program->context!=frame.staticResources->queryContext||program->identity->stageCount!=2
        ||program->identity->stages[0].type!=GL_VERTEX_SHADER||program->identity->stages[1].type!=GL_FRAGMENT_SHADER
        ||program->identity->stages[0].source!=glasscaustics::opengl::rasterVertexSource()
        ||program->identity->stages[1].source!=glasscaustics::opengl::rasterFragmentSource
        ||frame.rasterCaptureGeneration!=frame.gratingStagingSubmission.rasterGeneration
        ||frame.rasterCaptureViewport!=std::array<int,4>{0,0,int(frame.width_),int(frame.height_)}
        ||!get||!isSync||!status||!program->directFence||!isSync(program->directFence))return refuse();
    int done=0;status(program->directFence,GL_SYNC_STATUS,1,nullptr,&done);if(done!=GL_SIGNALED)return refuse();
    std::array<std::uint32_t,11> words{};std::array<float,4> values{};
    for(unsigned item=0;item<3;++item) {
        const char* name=item==0?"uCamera":item==1?"uRotation":"uProjection";
        const auto at=program->gl.GetUniformLocation(program->program,name);if(at<0)return refuse();
        get(program->program,at,values.data());
        const unsigned start=item==0?0:item==1?3:7,count=item==0?3:4;
        for(unsigned k=0;k<count;++k)std::memcpy(&words[start+k],&values[k],4);
    }
    if(!std::equal(words.begin(),words.end(),frame.rasterCaptureWords.begin())||glGetError()!=GL_NO_ERROR)return refuse();
    return true;
}
bool openGlCameraCoverageTriangle(const OpenGlFixtureSceneFrame& frame,const glasscaustics::Frame& frozen,
    std::size_t index,glasscaustics::opengl::CameraDiagonalTriangle& output)
{
    const auto& t=frozen.triangles[index];const auto& source=*frame.gratingStagingBinding->prepared.glass.sourceScene;
    if(t.instance>=frozen.instances.size())return false;
    const auto& placement=frozen.instances[t.instance];if(placement.object>=source.objectCount)return false;
    const auto& object=source.objects[placement.object];
    const opaquevisibility::Matrix identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    if(placement.world!=identity||t.object!=object.id.value||t.material!=object.material.value
        ||t.triangle>=object.indexCount/3)return false;
    const auto& prepared=frame.gratingStagingBinding->prepared.glass.query;
    if(placement.object>=prepared.objectMesh.size()||prepared.objectMesh[placement.object]<0
        ||std::size_t(prepared.objectMesh[placement.object])>=prepared.meshes.size())return false;
    const auto& mesh=prepared.meshes[std::size_t(prepared.objectMesh[placement.object])];
    // Inspect the actual uploaded inverse rows and the released metadata-table
    // index/instance key. The whole texture is bit-compared by the inspector.
    opaquevisibility::Matrix inverse;
    if(!opaquevisibility::inverse(placement.world,inverse))return false;
    bool instanceFound=false;
    const auto root=std::size_t(frozen.query.control[0]),limit=std::size_t(frozen.query.control[1]);
    if(root>=limit||limit>frozen.query.texels.size())return false;
    for(std::size_t node=root;node<limit;) {
        if(node+1>=limit)return false;
        std::uint32_t escape=0,leaf=0;
        if(!glasscaustics::exactReceiptInteger(frozen.query.texels[node][3],std::uint32_t(limit),escape)
            ||!glasscaustics::exactReceiptInteger(frozen.query.texels[node+1][3],UINT32_MAX,leaf)||escape<=node+1)return false;
        if(!leaf){node+=2;continue;}
        const auto at=std::size_t(leaf-1);if(at+3>=root)return false;
        const auto& metadata=frozen.query.texels[at+3];std::uint32_t instance=0;
        if(!glasscaustics::exactReceiptInteger(metadata[3],UINT32_MAX,instance))return false;
        if(instance==t.instance) {
            if(instanceFound||metadata[0]!=float(mesh.root)||metadata[1]!=prepared.texels[mesh.root][3]
                ||metadata[2]!=float(prepared.texels.size()+placement.object))return false;
            instanceFound=true;
            for(unsigned row=0;row<3;++row) {
                const std::array<float,4> expected{inverse[row],inverse[4+row],inverse[8+row],inverse[12+row]};
                if(std::memcmp(expected.data(),frozen.query.texels[at+row].data(),sizeof(expected))!=0)return false;
            }
        }
        node=escape;
    }
    if(!instanceFound)return false;
    if(mesh.root+1>=frozen.query.texels.size())return false;
    const auto end=std::size_t(frozen.query.texels[mesh.root][3]);if(end>frozen.query.texels.size())return false;
    bool found=false;
    for(std::size_t node=mesh.root;node<end;) {
        if(node+1>=end)return false;
        const auto& low=frozen.query.texels[node];const auto& high=frozen.query.texels[node+1];
        std::uint32_t escape=0,leaf=0;
        if(!glasscaustics::exactReceiptInteger(low[3],std::uint32_t(end),escape)
            ||!glasscaustics::exactReceiptInteger(high[3],std::uint32_t(end),leaf)||escape<=node+1)return false;
        if(!leaf){node+=2;continue;}
        const auto start=std::size_t(leaf-1);if(start<node+2||start>=escape)return false;
        std::uint32_t count=0;
        if(!glasscaustics::exactReceiptInteger(frozen.query.texels[start][3],4,count)||!count||start+3*count>escape)return false;
        for(unsigned k=0;k<count;++k) {
            std::uint32_t id=0;const auto at=start+3*k;
            if(!glasscaustics::exactReceiptInteger(frozen.query.texels[at+1][3],UINT32_MAX,id))return false;
            if(id!=t.triangle)continue;
            if(found)return false;
            found=true;
            for(unsigned v=0;v<3;++v)if(std::memcmp(frozen.query.texels[at+v].data(),t.points[v].data(),12)!=0)return false;
        }
        node=escape;
    }
    if(!found)return false;
    output={};output[0]=std::uint32_t(index);output[1]=t.object;output[2]=t.material;output[3]=t.instance;output[4]=t.triangle;
    for(unsigned v=0;v<3;++v) {
        const auto local=source.indices[object.firstIndex+t.triangle*3+v];if(local>=object.vertexCount)return false;
        output[5+v]=object.firstVertex+local;
        const auto& p=source.vertices[output[5+v]].position;
        const std::array<float,3> localPoint{p.x,p.y,p.z};
        if(std::memcmp(localPoint.data(),t.points[v].data(),12)!=0)return false;
        std::memcpy(output.data()+8+3*v,t.points[v].data(),12);
    }
    output[17]=std::uint32_t(placement.object);output[18]=t.reflectance?1:0;output[19]=t.glass?1:0;return true;
}
bool openGlCameraCoverageInputs(const OpenGlFixtureSceneFrame& frame,const OpenGlCameraCoverageRecord& record,
    const glasscaustics::CameraCellCoveragePreparation& preparation,std::vector<std::uint32_t>* output,std::string& error)
{
    namespace gc=glasscaustics;const auto& frozen=record.parent->retained().terminal->retained().incident.cohort->sources()[0]->frame;
    const auto size=23+8*preparation.pixels.size()+gc::opengl::kCameraDiagonalTriangleWords*frozen.triangles.size();
    if(output)output->resize(size);else if(record.inputs.size()!=size)return false;
    const auto put=[&](std::size_t at,std::uint32_t word){if(output){(*output)[at]=word;return true;}return record.inputs[at]==word;};
    const std::array<std::uint32_t,8> head{1,256,cameraraycoefficients::maximumVisits(256),0,
        std::uint32_t(frame.rendererGeneration_),std::uint32_t(frame.rasterCaptureGeneration),std::uint32_t(record.parent->retained().inputGeneration),std::uint32_t(record.generation)};
    for(unsigned k=0;k<head.size();++k)if(!put(k,head[k]))return false;
    for(unsigned k=0;k<13;++k)if(!put(10+k,frame.rasterCaptureWords[k]))return false;
    for(unsigned k=8;k<10;++k)if(!put(k,0))return false;
    for(std::size_t p=0;p<preparation.pixels.size();++p) {
        const auto& pixel=preparation.pixels[p];
        if(!pixel.associationCount||pixel.firstAssociation>=preparation.associations.size())return false;
        const auto& association=preparation.associations[pixel.firstAssociation];
        if(association.pixel!=p||association.parentRow>=record.parent->retained().rows.size())return false;
        const auto& row=record.parent->retained().rows[association.parentRow];
        if(row.disposition!=gc::DirectGratingTerminalReceiverRow::Disposition::Pixel||row.pixel.x!=pixel.x||row.pixel.y!=pixel.y
            ||std::tie(row.hit.objectId,row.hit.materialId,row.hit.instanceId)!=std::tie(pixel.object,pixel.material,pixel.instance))return false;
        std::size_t primary=frozen.triangles.size();
        for(std::size_t t=0;t<frozen.triangles.size();++t)if(frozen.triangles[t].triangle==row.hit.triangleId
            &&std::tie(frozen.triangles[t].object,frozen.triangles[t].material,frozen.triangles[t].instance)
                ==std::tie(pixel.object,pixel.material,pixel.instance)){
            if(primary!=frozen.triangles.size())return false;
            primary=t;
        }
        if(primary==frozen.triangles.size())return false;
        const gc::opengl::CameraDiagonalPixel values{pixel.x,pixel.y,pixel.object,pixel.material,pixel.instance,
            association.parentRow,std::uint32_t(primary),std::uint32_t(record.generation)};
        for(unsigned k=0;k<values.size();++k)if(!put(23+p*8+k,values[k]))return false;
    }
    for(std::size_t t=0;t<frozen.triangles.size();++t) {
        gc::opengl::CameraDiagonalTriangle values{};
        if(!openGlCameraCoverageTriangle(frame,frozen,t,values)){error="OpenGL coverage refuses nonidentical native local/world geometry";return false;}
        for(unsigned k=0;k<values.size();++k)if(!put(23+8*preparation.pixels.size()+20*t+k,values[k]))return false;
    }
    return true;
}
bool openGlDecodeCameraCoverage(const OpenGlCameraCoverageRecord& record,const glasscaustics::CameraCellCoveragePreparation& preparation,
    glasscaustics::CameraCellCoverageProof& proof,std::string& error)
{
    namespace gc=glasscaustics;namespace cc=cameraraycoefficients;namespace cp=camerapixelsigns;
    const auto refuse=[&](){error="OpenGL coverage refused an incomplete diagonal/competitor certificate";return false;};
    const auto t=record.reservation.triangles,p=preparation.pixels.size(),stride=16+t*gc::opengl::kCameraDiagonalPairWords;
    if(record.certificate.size()!=std::max<std::uint64_t>(1,p*stride))return refuse();
    gc::CameraCellCoverageProof decoded;decoded.generation=record.generation;decoded.work=record.reservation.operationWork;
    decoded.regions.reserve(2*p);decoded.vertices.reserve(6*p);
    for(std::size_t pixel=0;pixel<p;++pixel) {
        const auto* h=record.certificate.data()+pixel*stride;
        if(h[0]!=1||h[1]>=t||h[2]>=t||h[1]==h[2]||h[3]>=3||h[4]>=3||h[5]>1||h[6]<1||h[6]>2||h[7]>=t)return refuse();
        for(unsigned k=0;k<8;++k)if(h[8+k]!=record.inputs[23+pixel*8+k])return refuse();
        if(h[1]!=h[14]||h[15]!=record.generation)return refuse();
        const auto a=h[1],b=h[2],ea=h[3],eb=h[4];
        const auto tri=[&](std::uint32_t index,unsigned word){return record.inputs[23+8*p+20*index+word];};
        if(tri(a,17)!=tri(b,17)||tri(a,3)!=tri(b,3)||tri(a,1)!=tri(b,1)||tri(a,2)!=tri(b,2)
            ||tri(a,18)!=1||tri(b,18)!=1||tri(a,19)||tri(b,19)
            ||h[7]!=(tri(a,4)<tri(b,4)?a:b))return refuse();
        for(unsigned k=0;k<2;++k) {
            const unsigned va=(ea+k)%3,vb=(eb+1-k)%3;
            if(tri(a,5+va)!=tri(b,5+vb))return refuse();
            for(unsigned axis=0;axis<3;++axis)if(tri(a,8+3*va+axis)!=tri(b,8+3*vb+axis))return refuse();
        }
        std::uint32_t horizon=0;const auto* primary=h+16+b*gc::opengl::kCameraDiagonalPairWords;
        for(std::uint32_t index=0;index<t;++index) {
            const auto* pair=h+16+index*gc::opengl::kCameraDiagonalPairWords;
            if(index==a){
                for(unsigned k=0;k<gc::opengl::kCameraDiagonalPairWords;++k)if(pair[k])return refuse();
                continue;
            }
            cc::Request expected{};std::copy_n(record.inputs.begin(),23,expected.begin());
            expected[8]=preparation.pixels[pixel].x;expected[9]=preparation.pixels[pixel].y;
            for(unsigned k=0;k<17;++k){expected[23+k]=tri(a,k);expected[40+k]=tri(index,k);}
            cc::Result constructed{};std::copy_n(pair,557,constructed.begin());
            if(!camerawideline::validateRecord(constructed,expected))return refuse();
            const auto signs=[&](unsigned form,std::array<std::uint32_t,4>& output) {
                const auto* r=pair+557+(form-4)*73;
                if(r[0]!=1||r[1]!=cp::maximumScalarOperations()||r[2]!=cp::kWireTag||!r[3]||r[3]>cp::maximumVisits())return false;
                for(unsigned k=0;k<57;++k)if(r[8+k]!=expected[k])return false;
                const std::array<std::uint32_t,8> tail{1,cp::kRequestTag,0,cp::maximumScalarOperations(),form,cp::kWireTag,0,0};
                for(unsigned k=0;k<8;++k)if(r[65+k]!=tail[k])return false;
                for(unsigned k=0;k<4;++k){
                    if(r[4+k]>2)return false;
                    output[k]=r[4+k];
                }
                return true;
            };
            const auto uniform=[](const std::array<std::uint32_t,4>& s,std::uint32_t value){return std::all_of(s.begin(),s.end(),[&](auto x){return x==value;});};
            if(index==b) {
                std::array<std::array<std::uint32_t,4>,13> s{};
                for(unsigned form=4;form<=16;++form)if(!signs(form,s[form-4]))return refuse();
                horizon=s[3][0];if(!horizon||!uniform(s[3],horizon)||constructed[538]!=horizon||constructed[548]!=horizon)return refuse();
                for(unsigned form=4;form<16;++form)if(form!=4+ea&&form!=10+eb&&!uniform(s[form-4],horizon))return refuse();
                if(!uniform(s[12],0))return refuse();
                const auto& left=s[ea];const auto& right=s[6+eb];
                if(h[5]==0?(left[0]!=0||left[3]!=0||!left[1]||left[2]!=3-left[1])
                    :(left[1]!=0||left[2]!=0||!left[0]||left[3]!=3-left[0]))return refuse();
                for(unsigned k=0;k<4;++k)if(right[k]!=(left[k]?3-left[k]:0))return refuse();
                if(h[6]!=((h[5]==0?left[1]:left[0])==horizon?1u:2u))return refuse();
                const unsigned x=61+(4+ea)*28,y=61+(10+eb)*28;
                if(constructed[x]!=constructed[y])return refuse();
                for(unsigned k=0;k<3;++k) {
                    if(constructed[y+1+9*k]!=(constructed[x+1+9*k]?3-constructed[x+1+9*k]:0))return refuse();
                    for(unsigned limb=1;limb<9;++limb)if(constructed[x+1+9*k+limb]!=constructed[y+1+9*k+limb])return refuse();
                }
            } else {
                // Inspect the native primary form identity, not an independently
                // rounded reconstruction, before accepting competitor signs.
                for(unsigned k=0;k<168;++k)if(pair[61+4*28+k]!=primary[61+4*28+k])return refuse();
                for(unsigned k=0;k<10;++k)if(pair[537+k]!=primary[537+k])return refuse();
                std::array<std::array<std::uint32_t,4>,5> s{};
                const unsigned forms[]{10,11,12,13,16};for(unsigned k=0;k<5;++k)if(!signs(forms[k],s[k]))return refuse();
                for(unsigned form=4;form<=16;++form)if(form<10||form==14||form==15)
                    for(unsigned k=0;k<73;++k)if(pair[557+(form-4)*73+k])return refuse();
                const auto other=uniform(s[3],1)?1u:uniform(s[3],2)?2u:0u;
                std::array<std::uint32_t,3> strict{};bool excluded=false;
                for(unsigned k=0;k<3;++k)strict[k]=uniform(s[k],1)?1u:uniform(s[k],2)?2u:0u;
                for(unsigned x=0;x<3;++x)for(unsigned y=x+1;y<3;++y)if(strict[x]&&strict[y]==3-strict[x])excluded=true;
                if(other) {
                    if(pair[548]==3-other)excluded=true;
                    for(unsigned k=0;k<3;++k)if(strict[k]==3-other)excluded=true;
                }
                // The nominated pair may occur after this raw table row.
                const auto receiverHorizon=primary[557+3*73+4];
                const bool behind=other&&pair[548]==other&&uniform(s[4],other==receiverHorizon?2u:1u);
                if(!excluded&&!behind)return refuse();
            }
        }
        // Exact fixed vertices, closed complementary regions. The native wire
        // records the unique seam owner; measure-zero incidence grants no rival.
        const std::array<std::array<double,2>,4> corners{{{0,0},{1,0},{0,1},{1,1}}};
        const std::array<unsigned,3> first=h[5]==0?std::array<unsigned,3>{0,1,3}:std::array<unsigned,3>{0,1,2};
        const std::array<unsigned,3> second=h[5]==0?std::array<unsigned,3>{0,3,2}:std::array<unsigned,3>{1,3,2};
        for(unsigned region=0;region<2;++region) {
            const auto triangle=(region==0)==(h[6]==1)?a:b;
            decoded.regions.push_back({std::uint32_t(pixel),triangle,decoded.vertices.size(),3,.5});
            for(auto corner:region==0?first:second)decoded.vertices.push_back(corners[corner]);
        }
    }
    proof=std::move(decoded);return true;
}
bool openGlInspectCameraCoverage(const OpenGlFixtureSceneFrame& frame,const OpenGlCameraCoverageRecord& record,
    const glasscaustics::CameraCellCoverageReadback& candidate,std::string& error)
{
    namespace gc=glasscaustics;const auto refuse=[&](){error="OpenGL camera coverage native ownership or certificate changed";return false;};
    const auto& program=record.program;const auto& gl=frame.staticResources->gl;OpenGlReceiverState state(gl);
    const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
    const auto bufferSize=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
    const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
    const auto syncStatus=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
    const auto getUniform=reinterpret_cast<PFNGLGETUNIFORMUIVPROC>(glfwGetProcAddress("glGetUniformuiv"));
    if(!record.parent||candidate.parent!=record.parent||candidate.inputGeneration!=record.parent->retained().inputGeneration
        ||candidate.proof.generation!=record.generation||!gc::sameCameraCellCoverageBounds(candidate.bounds,record.reservation)
        ||!(candidate.submission==record.parent->retained().submission)||candidate.evaluationTime!=record.parent->retained().evaluationTime
        ||!openGlCoverageProgramIdentity(frame,program)||program->identity->stageCount!=1
        ||program->identity->stages[0].type!=GL_COMPUTE_SHADER
        ||program->identity->stages[0].source!=gc::opengl::cameraDiagonalComputeSource()
        ||!isBuffer||!bufferSize||!getLabel||!isSync||!syncStatus||!getUniform||!program->directFence||!isSync(program->directFence)
        ||!record.attemptOrdinal||record.attemptOrdinal>frame.nativeAttemptCount)return refuse();
    const auto& attempt=frame.nativeAttempts[record.attemptOrdinal-1];
    if(attempt.kind!=OpenGlNativeAttemptKind::CameraCoverage||attempt.receiver!=record.parent||!attempt.cameraCoverageReserved
        ||!gc::sameCameraCellCoverageBounds(attempt.cameraCoverageBounds,record.reservation))return refuse();
    int done=0;syncStatus(program->directFence,GL_SYNC_STATUS,1,nullptr,&done);if(done!=GL_SIGNALED)return refuse();
    const char* names[]{"uPixelCount","uTriangleCount","uGeneration"};
    const std::uint32_t expected[]{std::uint32_t(candidate.preparation.pixels.size()),std::uint32_t(record.reservation.triangles),std::uint32_t(record.generation)};
    for(unsigned k=0;k<3;++k){std::uint32_t value=0;const auto at=gl.GetUniformLocation(program->program,names[k]);
        if(at<0)return refuse();
        getUniform(program->program,at,&value);if(value!=expected[k])return refuse();
    }
    const auto inspect=[&](unsigned slot,const std::vector<std::uint32_t>& values) {
        const auto id=program->directBuffers[slot];if(!id||!isBuffer(id))return false;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,id);GLint64 bytes=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&bytes);
        std::array<char,96> label{};int length=0;getLabel(GL_BUFFER,id,int(label.size()),&length,label.data());
        if(bytes!=GLint64(values.size()*4)||label!=program->label)return false;
        const void* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,bytes,GL_MAP_READ_BIT);if(!mapped)return false;
        const bool same=std::memcmp(mapped,values.data(),values.size()*4)==0;
        return gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)==GL_TRUE&&same;
    };
    if(!inspect(0,record.inputs)||!inspect(1,record.certificate)||!openGlRasterCaptureMatches(frame,error))return refuse();
    const OpenGlDirectTerminalReceiverRecord* receiver=nullptr;
    for(auto* r=frame.directReceiverResults.get();r;r=r->previous.get())if(r->result==record.parent)receiver=r;
    if(!receiver||!openGlInspectReceiver(frame,*receiver,record.parent->retained(),error))return refuse();
    const auto getFloat=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
    if(!getFloat)return refuse();
    std::array<std::uint32_t,13> camera{};std::array<float,4> values{};
    const char* cameraNames[]{"uCamera","uRotation","uProjection","uExtent"};
    const unsigned starts[]{0,3,7,11},counts[]{3,4,4,2};
    for(unsigned k=0;k<4;++k) {
        const auto at=gl.GetUniformLocation(receiver->program->program,cameraNames[k]);if(at<0)return refuse();
        getFloat(receiver->program->program,at,values.data());std::memcpy(camera.data()+starts[k],values.data(),4*counts[k]);
    }
    if(camera!=frame.rasterCaptureWords)return refuse();
    // Capture and compare actual local query texture words, including zero
    // signs. This is source association only, never a CPU nearest-hit query.
    const auto& frozen=record.parent->retained().terminal->retained().incident.cohort->sources()[0]->frame;
    std::vector<gc::Texel> query(frozen.query.texels.size());state.packHost();
    if(query.capacity()>2*frozen.query.texels.size())return refuse();
    glBindTexture(GL_TEXTURE_2D,frame.causticTextures[0]);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,query.data());
    if(std::memcmp(query.data(),frozen.query.texels.data(),query.size()*sizeof(gc::Texel))!=0)return refuse();
    gc::CameraCellCoverageReadback decoded;decoded.parent=candidate.parent;decoded.submission=candidate.submission;
    decoded.evaluationTime=candidate.evaluationTime;decoded.inputGeneration=candidate.inputGeneration;decoded.bounds=candidate.bounds;
    if(!gc::prepareCameraCellCoverage(*record.parent,frozen,record.reservation,decoded.preparation,error)
        ||!openGlCameraCoverageInputs(frame,record,decoded.preparation,nullptr,error)
        ||!openGlDecodeCameraCoverage(record,decoded.preparation,decoded.proof,error)
        ||!gc::sameCameraCellCoverageReadback(decoded,candidate,false))return refuse();
    std::uint64_t bytes=sizeof(record)+program->retainedBytes+4*(record.inputs.capacity()+record.certificate.capacity())
        +query.capacity()*sizeof(gc::Texel)+sizeof(gc::NativeCameraCellCoverageResult)+128;
    if(!openGlCameraCoverageReadbackBytes(candidate,bytes)||!openGlCameraCoverageReadbackBytes(decoded,bytes)
        ||(record.result&&!openGlCameraCoverageReadbackBytes(record.result->retained(),bytes))||bytes>record.ownedBytes)return refuse();
    if(glGetError()!=GL_NO_ERROR||!frame.reconcileNativeAccounting(error))return refuse();
    return true;
}
#endif
bool openGlCameraCoverageBudget(const OpenGlFixtureSceneFrame& frame,
    const glasscaustics::NativeDirectGratingTerminalReceiverResult& parent,
    glasscaustics::NativeCameraCellCoverageBudget& output,std::string& error)
{
    glasscaustics::DirectGratingTerminalNativeBudget native;
    if(!openGlTerminalNativeBudget(frame,native,error))return false;
    const auto& frozen=parent.retained().terminal->retained().incident.cohort->sources()[0]->frame;
    auto budget=glasscaustics::NativeCameraCellCoverageBudget{};
    budget.retainedBytes=std::max(native.retainedBytes,parent.retained().bounds.bytes);
    if(!renderpassoutput::detail::checkedAdd(budget.retainedBytes,sizeof(OpenGlCameraCoverageLease),budget.retainedBytes)
        ||budget.retainedBytes>renderpassoutput::kMaximumTotalBytes)return false;
    budget.combinedWork=native.combinedWork;
    // Separate logical native live locals, not a driver spill measurement.
    constexpr std::uint64_t logical=cameraraycoefficients::kLogicalScratchBytes+camerapixelsigns::kLogicalScratchBytes
        +3*557*4+4*73*4+2*65*4+256;
    budget.pixelBytes=4*8*4+logical;
    budget.candidateBytes=4*glasscaustics::opengl::kCameraDiagonalPairWords*4;
    constexpr std::uint64_t hostScratch=4*557*4+3*57*4+2*13*4*4+4*20*4+512;
    budget.fixedBytes=2*kQueryProgramSourceReservation+sizeof(OpenGlCameraCoverageRecord)+sizeof(OpenGlCameraCoverageLease)
        +sizeof(OpenGlNativeAttempt)+3*sizeof(OpenGlNativeAttemptTotals)+hostScratch
        +4*23*4+4ull*frozen.triangles.size()*glasscaustics::opengl::kCameraDiagonalTriangleWords*4
        +2*frozen.query.texels.size()*sizeof(glasscaustics::Texel)+4ull*frame.width_*frame.height_*sizeof(glasscaustics::Texel);
    // Header is separately allocated for each admitted pre-dedup Pixel row.
    budget.pixelBytes+=4*glasscaustics::opengl::kCameraDiagonalHeaderWords*4;
    if(budget.fixedBytes>renderpassoutput::kMaximumTotalBytes)return false;
    output=budget;return true;
}
}

bool OpenGlFixtureSceneFrame::beginCameraCellCoverageLease(
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>& parent,
    const glasscaustics::SubmissionIdentity& expected,double time,
    std::unique_ptr<glasscaustics::NativeCameraCellCoverageLease>& output,
    glasscaustics::NativeCameraCellCoverageBudget& budget,std::string& error) const
{
#if defined(__APPLE__)
    (void)parent;(void)expected;(void)time;(void)output;(void)budget;error="OpenGL camera coverage requires GL4.3";return false;
#else
    if(cameraCoverageThreadLease||directTerminalThreadLease||directReceiverThreadLease)return false;
    std::unique_lock<std::mutex> guard(fixtureMutex());
    if(activeCameraCoverageLease||activeDirectTerminalLease||activeDirectReceiverLease||!parent||!directReceiverResults
        ||directReceiverResults->result!=parent||parent->owner()!=this||!parent->retained().terminal||!directGratingCohort
        ||parent->retained().terminal->retained().incident.cohort!=directGratingCohort->binding
        ||!(parent->retained().submission==expected)||!std::isfinite(time)||time!=parent->retained().evaluationTime
        ||time!=gratingCameraEvaluationTime||!staticResources||!staticResources->queryContext
        ||ownerContext!=glfwGetCurrentContext()||nativeAttemptCount==nativeAttempts.size()
        ||rendererGeneration_>UINT32_MAX||gratingStagingSubmission.rasterGeneration>UINT32_MAX
        ||parent->retained().inputGeneration>UINT32_MAX
        ||parent->retained().terminal->retained().incident.cohort->sources()[0]->frame.triangles.size()<3)
    {error="OpenGL camera coverage requires the current published receiver";return false;}
    // Setup/link/provenance are not bounded by P*T. Charge a fixed existing
    // native slot before reconstruction, source assembly, copies or allocation,
    // including every zero-P and failed begin. Parent aliases already belong
    // to the permanent receiver chain; fixed registry storage is in sizeof(frame).
    OpenGlNativeAttempt attempt;
    attempt.kind=OpenGlNativeAttemptKind::CameraCoverage;attempt.binding=gratingStagingBinding;attempt.cohort=directGratingCohort->binding;
    attempt.owner=this;attempt.context=ownerContext;attempt.renderer=rendererGeneration_;
    attempt.contextGeneration=staticResources->queryContext->generation;attempt.ordinal=nativeAttemptCount+1;
    attempt.stagingGeneration=gratingStagingSubmission.submissionSerial;attempt.receiver=parent;attempt.terminal=parent->retained().terminal;
    if(!openGlCameraCoverageCosts(*this,attempt,attempt.cameraCoverageWork,error))return false;
    auto work=cameraCoverageWork;
    for(unsigned k=0;k<work.size();++k)if(!renderpassoutput::detail::checkedAdd(work[k],attempt.cameraCoverageWork[k],work[k]))return false;
    nativeAttempts[nativeAttemptCount++]=std::move(attempt);cameraCoverageWork=work;
    glasscaustics::NativeCameraCellCoverageBudget native;
    if(!reconcileNativeAccounting(error)||!openGlCameraCoverageBudget(*this,*parent,native,error)
        ||!openGlRasterCaptureMatches(*this,error)
        ||!openGlInspectReceiver(*this,*directReceiverResults,parent->retained(),error))return false;
    const auto generation=nextQueryOwnerGeneration();if(!generation||generation>UINT32_MAX)return false;
    auto lease=std::make_unique<OpenGlCameraCoverageLease>(*this,std::move(guard),parent,directReceiverResults.get(),generation,nativeAttemptCount);
    lease->budget=native;activeCameraCoverageLease=lease.get();cameraCoverageThreadLease=lease.get();
    if(!lease->live(error))return false;
    budget=native;output=std::move(lease);return true;
#endif
}
bool OpenGlFixtureSceneFrame::reserveCameraCellCoverageLease(glasscaustics::NativeCameraCellCoverageLease& token,
    const glasscaustics::CameraCellCoverageBounds& bounds,const glasscaustics::CameraCellCoverageReadback& incumbent,std::string& error) const
{
    namespace gc=glasscaustics;auto* lease=dynamic_cast<OpenGlCameraCoverageLease*>(&token);
    if(!lease||&lease->frame!=this||lease->reserveAttempted||lease->reserved)return false;
    lease->reserveAttempted=true;if(!lease->live(error))return false;
    gc::NativeCameraCellCoverageBudget native;gc::CameraCellCoverageBounds exact;
    if(!openGlCameraCoverageBudget(*this,*lease->parent,native,error)
        ||std::tie(native.retainedBytes,native.combinedWork,native.pixelBytes,native.candidateBytes,native.regionBytes,native.vertexBytes,native.fixedBytes)
            !=std::tie(lease->budget.retainedBytes,lease->budget.combinedWork,lease->budget.pixelBytes,lease->budget.candidateBytes,lease->budget.regionBytes,lease->budget.vertexBytes,lease->budget.fixedBytes)
        ||!gc::calculateCameraCellCoverageBounds(*lease->parent,lease->frozenFrame(),native,incumbent,exact,error)
        ||!gc::sameCameraCellCoverageBounds(exact,bounds))return false;
    auto prospective=nativeAttempts[lease->ordinal-1];prospective.cameraCoverageReserved=true;prospective.cameraCoverageBounds=bounds;
    if(!openGlCameraCoverageCosts(*this,prospective,prospective.cameraCoverageWork,error))return false;
    OpenGlNativeAttemptTotals before,after;
    if(!openGlAttemptTotals(*this,before,error))return false;
    after=before;
    for(unsigned k=0;k<after.cameraCoverageWork.size();++k)after.cameraCoverageWork[k]-=nativeAttempts[lease->ordinal-1].cameraCoverageWork[k];
    if(!openGlAddAttempt(*this,prospective,after,error))return false;
    gc::DirectGratingTerminalNativeBudget base;if(!openGlTerminalNativeBudget(*this,base,error)||bounds.bytes<base.retainedBytes)return false;
    const auto storage=bounds.bytes-base.retainedBytes;
    if(bounds.operationWork>gc::kMaximumQueryWork-base.combinedWork
        ||bounds.operationWork>gc::kMaximumDepositionWork-base.depositionWork
        ||storage>renderpassoutput::kMaximumTotalBytes-directGratingBytes)return false;
    nativeAttempts[lease->ordinal-1]=std::move(prospective);cameraCoverageWork=after.cameraCoverageWork;
    gratingCoverageWork+=bounds.operationWork;directGratingBytes+=storage;activeCameraCoverageStorage=storage;
    lease->reservation=bounds;lease->reserved=true;
    lease->staged=std::make_unique<OpenGlCameraCoverageRecord>();lease->staged->ownedBytes=storage;
    lease->staged->parent=lease->parent;lease->staged->attemptOrdinal=lease->ordinal;lease->staged->generation=lease->generation;lease->staged->reservation=bounds;
    if(cameraCoverageStop==OpenGlCameraCoverageStopForFixture::Reserved){error="OpenGL coverage fixture refused after full reservation";return false;}
    return true;
}
bool OpenGlFixtureSceneFrame::dispatchCameraCellCoverage(glasscaustics::NativeCameraCellCoverageLease& token,
    const glasscaustics::CameraCellCoveragePreparation& preparation,glasscaustics::CameraCellCoverageProof& proof,std::string& error) const
{
#if defined(__APPLE__)
    (void)token;(void)preparation;(void)proof;error="OpenGL camera coverage requires GL4.3";return false;
#else
    namespace gc=glasscaustics;auto* lease=dynamic_cast<OpenGlCameraCoverageLease*>(&token);
    if(!lease||&lease->frame!=this||!lease->reserved||!lease->staged||lease->dispatched||lease->staged->program)return false;
    lease->dispatched=true;if(!lease->live(error))return false;
    auto& record=*lease->staged;const auto& gl=staticResources->gl;OpenGlReceiverState state(gl);
    if(preparation.pixels.size()>record.reservation.maximumPixels
        ||!openGlCameraCoverageInputs(*this,record,preparation,&record.inputs,error))return false;
    const auto words=std::max<std::uint64_t>(1,preparation.pixels.size()*(16+record.reservation.triangles*gc::opengl::kCameraDiagonalPairWords));
    if(words>renderpassoutput::kMaximumTotalBytes/4)return false;
    record.certificate.resize(words);
    if(record.inputs.capacity()>2*record.inputs.size()||record.certificate.capacity()>2*words)return false;
    const auto source=gc::opengl::cameraDiagonalComputeSource();
    record.program=linkRegisteredOpenGlProgram(gl,staticResources->queryContext,rendererGeneration_,{{GL_COMPUTE_SHADER,source}},error,false);
    if(!record.program)return false;
    const auto sizeOf=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    const auto label=reinterpret_cast<PFNGLOBJECTLABELPROC>(glfwGetProcAddress("glObjectLabel"));
    const auto indexedSize=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
    const auto integer64=reinterpret_cast<PFNGLGETINTEGER64VPROC>(glfwGetProcAddress("glGetInteger64v"));
    if(!sizeOf||!label||!indexedSize||!integer64)return false;
    GLint64 block=0;integer64(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&block);
    const std::array<std::uint64_t,2> sizes{record.inputs.size()*4,record.certificate.size()*4};
    if(block<=0||sizes[0]>std::uint64_t(block)||sizes[1]>std::uint64_t(block))return false;
    int groups=0;gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&groups);
    if(groups<=0||preparation.pixels.size()>std::uint32_t(groups))return false;
    gl.GenBuffers(2,record.program->directBuffers.data());const void* payloads[]{record.inputs.data(),record.certificate.data()};
    for(unsigned slot=0;slot<2;++slot) {
        const auto id=record.program->directBuffers[slot];if(!id)return false;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,id);label(GL_BUFFER,id,-1,record.program->label.data());
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(sizes[slot]),payloads[slot],slot?GL_DYNAMIC_READ:GL_STATIC_DRAW);
        GLint64 actual=0;sizeOf(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);if(actual!=GLint64(sizes[slot])||glGetError()!=GL_NO_ERROR)return false;
        record.nativeBytes+=sizes[slot];record.program->retainedBytes+=sizes[slot];gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,id);
        int binding=0;gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,slot,&binding);GLint64 start=0,size=0;
        indexedSize(GL_SHADER_STORAGE_BUFFER_START,slot,&start);indexedSize(GL_SHADER_STORAGE_BUFFER_SIZE,slot,&size);
        if(unsigned(binding)!=id||start||size!=GLint64(sizes[slot]))return false;
    }
    gl.UseProgram(record.program->program);const char* names[]{"uPixelCount","uTriangleCount","uGeneration"};
    const std::uint32_t values[]{std::uint32_t(preparation.pixels.size()),std::uint32_t(record.reservation.triangles),std::uint32_t(record.generation)};
    for(unsigned k=0;k<3;++k){const auto at=gl.GetUniformLocation(record.program->program,names[k]);
        if(at<0)return false;
        gl.Uniform1ui(at,values[k]);
    }
    gl.DispatchCompute(unsigned(std::max<std::size_t>(1,preparation.pixels.size())),1,1);
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
    record.program->directFence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);if(!record.program->directFence)return false;
    const auto wait=[&]
    {
        loopstep::Scope fenceScope ("gl:caustic-coverage-fence-wait");
        return gl.ClientWaitSync(record.program->directFence,GL_SYNC_FLUSH_COMMANDS_BIT,30000000000ull);
    }();
    if((wait!=GL_ALREADY_SIGNALED&&wait!=GL_CONDITION_SATISFIED)||glGetError()!=GL_NO_ERROR)return false;
    lease->complete=true;
    if(cameraCoverageStop==OpenGlCameraCoverageStopForFixture::Fence){error="OpenGL coverage fixture refused after completed native fence";return false;}
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,record.program->directBuffers[1]);
    const void* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(sizes[1]),GL_MAP_READ_BIT);if(!mapped)return false;
    std::memcpy(record.certificate.data(),mapped,sizes[1]);if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return false;
    return openGlDecodeCameraCoverage(record,preparation,proof,error);
#endif
}
bool OpenGlFixtureSceneFrame::validateCameraCellCoverageProof(const glasscaustics::NativeCameraCellCoverageLease& token,
    const glasscaustics::CameraCellCoverageReadback& candidate,std::string& error) const
{
#if defined(__APPLE__)
    (void)token;(void)candidate;error="OpenGL camera coverage requires GL4.3";return false;
#else
    const auto* lease=dynamic_cast<const OpenGlCameraCoverageLease*>(&token);
    if(!lease||&lease->frame!=this||!lease->reserved||!lease->complete||!lease->staged||lease->inspections==3)return false;
    ++lease->inspections;if(!lease->live(error))return false;
    if(lease->staged->result) {
        if(candidate.nativeResult!=lease->staged->result||candidate.nativeResult->owner()!=this
            ||!glasscaustics::sameCameraCellCoverageReadback(candidate,candidate.nativeResult->retained(),false))return false;
    } else if(candidate.nativeResult)return false;
    glasscaustics::DirectGratingTerminalNativeBudget actual;
    if(!openGlTerminalNativeBudget(*this,actual,error)||actual.retainedBytes!=lease->reservation.bytes
        ||actual.combinedWork!=lease->reservation.combinedWork)return false;
    return openGlInspectCameraCoverage(*this,*lease->staged,candidate,error);
#endif
}
bool OpenGlFixtureSceneFrame::finishCameraCellCoverageLease(glasscaustics::NativeCameraCellCoverageLease& token,
    const glasscaustics::CameraCellCoverageReadback& candidate,
    std::shared_ptr<const glasscaustics::NativeCameraCellCoverageResult>& output,std::string& error) const
{
    auto* lease=dynamic_cast<OpenGlCameraCoverageLease*>(&token);
    if(!lease||&lease->frame!=this||!lease->staged||lease->finishAttempted||lease->staged->result||candidate.nativeResult)return false;
    lease->finishAttempted=true;if(!validateCameraCellCoverageProof(token,candidate,error))return false;
    lease->staged->result=makeCameraCellCoverageResult(candidate);
    std::uint64_t bytes=sizeof(OpenGlCameraCoverageRecord)+sizeof(glasscaustics::NativeCameraCellCoverageResult)+128
        +lease->staged->program->retainedBytes+4*(lease->staged->inputs.capacity()+lease->staged->certificate.capacity());
    if(!openGlCameraCoverageReadbackBytes(candidate,bytes)||!openGlCameraCoverageReadbackBytes(lease->staged->result->retained(),bytes)
        ||bytes>lease->staged->ownedBytes||!glasscaustics::sameCameraCellCoverageReadback(candidate,lease->staged->result->retained(),false))return false;
    output=lease->staged->result;return true;
}

bool produceOpenGlCameraCellCoverageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>& parent,
    const glasscaustics::SubmissionIdentity& expected,double time,glasscaustics::CameraCellCoverageReadback& output,
    std::string& error,OpenGlCameraCoverageStopForFixture stop)
{
    if(!frame||cameraCoverageThreadLease||directTerminalThreadLease||directReceiverThreadLease)return false;
    const auto prior=cameraCoverageStop;cameraCoverageStop=stop;
    int token=0;auto restore=[&](int*){cameraCoverageStop=prior;};std::unique_ptr<int,decltype(restore)> guard(&token,restore);
    try{return glasscaustics::proveCameraCellCoverage(parent,*frame,expected,time,output,error);}
    catch(const std::bad_alloc&){error="OpenGL camera coverage allocation failed after permanent work admission";return false;}
}
bool inspectOpenGlCameraCellCoverageForFixture(const NativeFixtureSceneFrame& owner,
    const glasscaustics::CameraCellCoverageReadback& candidate,std::string& error)
{
#if defined(__APPLE__)
    (void)owner;(void)candidate;error="OpenGL camera coverage requires GL4.3";return false;
#else
    if(cameraCoverageThreadLease||directTerminalThreadLease||directReceiverThreadLease)return false;
    std::lock_guard<std::mutex> guard(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&owner);
    if(!frame||!frame->staticResources||frame->ownerContext!=glfwGetCurrentContext()||!candidate.nativeResult
        ||candidate.nativeResult->owner()!=frame||!glasscaustics::sameCameraCellCoverageReadback(candidate,candidate.nativeResult->retained(),false))return false;
    const OpenGlCameraCoverageRecord* record=nullptr;
    for(auto* p=frame->cameraCoverageResults.get();p;p=p->previous.get())if(p->result==candidate.nativeResult)record=p;
    return record&&openGlInspectCameraCoverage(*frame,*record,candidate,error);
#endif
}
bool readOpenGlCameraCellCoverageStorageForFixture(const NativeFixtureSceneFrame& owner,
    std::array<unsigned,2>& buffers,std::uintptr_t& fence,std::array<std::uint64_t,8>& counters,std::string& error)
{
    if(cameraCoverageThreadLease||directTerminalThreadLease||directReceiverThreadLease)return false;
    std::lock_guard<std::mutex> guard(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&owner);
    if(!frame||!frame->staticResources||frame->ownerContext!=glfwGetCurrentContext()||!frame->reconcileNativeAccounting(error))return false;
    glasscaustics::DirectGratingTerminalNativeBudget budget;
    if(!openGlTerminalNativeBudget(*frame,budget,error))return false;
    buffers={};fence=0;counters=frame->cameraCoverageWork;
    if(const auto* record=frame->cameraCoverageResults.get()) {
        if(!record->program)return false;
        buffers={record->program->directBuffers[0],record->program->directBuffers[1]};
        fence=reinterpret_cast<std::uintptr_t>(record->program->directFence);
    }
    return true;
}

bool OpenGlFixtureSceneFrame::validateCameraTraceProductionQuery(const glasscaustics::SubmissionIdentity& expected,
    const glasscaustics::NativeGlassCameraGratingTraceReadback& supplied,std::string& error) const
{
#if defined(__APPLE__)
    (void)expected;(void)supplied;error="OpenGL camera glass grating trace requires native GL4.3 ownership";return false;
#else
    namespace gc=glasscaustics;
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto* batch=cameraGlassTraceBatch.get();
    if(!batch||!batch->program||!batch->evidence||!staticResources||!gratingStagingBinding
        ||!privateCameraAttachments||gratingCameraBaseComplete||!gratingCameraRuntime
        ||gratingCameraEvaluationTime!=gratingStagingBinding->evaluationTime
        ||ownerContext!=glfwGetCurrentContext()||staticResources->ownerContext!=ownerContext
        ||batch->sceneBuffer!=causticBuffers[4]||batch->queryTexture!=causticTextures[0]
        ||batch->queryControl!=gratingStagingBinding->frame.query.control
        ||batch->evidence!=queryEvidence[std::size_t(OpenGlQueryArithmeticRole::GlassCameraGratingTrace)]
        ||batch->program!=queryPrograms[std::size_t(OpenGlQueryArithmeticRole::GlassCameraGratingTrace)]
        ||!(batch->upload.submission==expected)||!batch->upload.complete
        ||batch->upload.generation!=expected.submissionSerial
        ||expected.rasterGeneration!=gratingStagingSubmission.rasterGeneration
        ||expected.queryGeneration!=gratingStagingSubmission.queryGeneration
        ||!queryArithmeticEvidenceMatches(this,batch->evidence,expected,error))
        return refuse("OpenGL camera glass grating trace lost its completed native dispatch owner");
    const auto& binding=*gratingStagingBinding;const auto& gl=staticResources->gl;
    if(!gc::admitGlassCameraGratingTraceCamera(binding.frame,error))return false;
    const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
    const auto isSync=reinterpret_cast<PFNGLISSYNCPROC>(glfwGetProcAddress("glIsSync"));
    const auto syncState=reinterpret_cast<PFNGLGETSYNCIVPROC>(glfwGetProcAddress("glGetSynciv"));
    const auto bufferSize=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
    const auto uniformFloat=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
    const auto uniformInt=reinterpret_cast<PFNGLGETUNIFORMIVPROC>(glfwGetProcAddress("glGetUniformiv"));
    const auto getLabel=reinterpret_cast<PFNGLGETOBJECTLABELPROC>(glfwGetProcAddress("glGetObjectLabel"));
    if(!isBuffer||!isSync||!syncState||!bufferSize||!uniformFloat||!uniformInt||!getLabel
        ||!batch->program->directFence||!isSync(batch->program->directFence))
        return refuse("OpenGL camera glass grating trace has no retained native fence or buffer inspection");
    int fence=0;syncState(batch->program->directFence,GL_SYNC_STATUS,1,nullptr,&fence);
    if(fence!=GL_SIGNALED)return refuse("OpenGL camera glass grating trace fence is incomplete");
    const auto location=[&](const char* name){return gl.GetUniformLocation(batch->program->program,name);};
    const auto matchesFloat=[&](const char* name,const float* values,std::size_t count) {
        const int at=location(name);std::array<float,8> actual{};
        if(at<0||count>actual.size())return false;
        // Array elements have independent locations and are queried separately.
        uniformFloat(batch->program->program,at,actual.data());
        return std::memcmp(actual.data(),values,count*sizeof(float))==0;
    };
    const auto matchesInt=[&](const char* name,int value) {
        int actual=0;const int at=location(name);if(at<0)return false;
        uniformInt(batch->program->program,at,&actual);return actual==value;
    };
    const auto camera=gc::cameraPosition(binding.frame);const auto& rotation=binding.frame.camera.transform.rotation;
    const gc::Texel cameraUniform{camera[0],camera[1],camera[2],0},rotationUniform{rotation.x,rotation.y,rotation.z,rotation.w};
    const gc::Texel projection{std::tan(binding.frame.camera.verticalFovRadians*.5f),float(width_)/height_,binding.frame.camera.nearPlane,binding.frame.camera.farPlane};
    const gc::Texel glass{float(binding.prepared.glass.object.value&65535u),float(binding.prepared.glass.object.value>>16),
        float(binding.prepared.glass.material.value&65535u),float(binding.prepared.glass.material.value>>16)};
    const std::array<float,2> extent{float(width_),float(height_)};
    if(!matchesInt("uSampleCount",int(batch->upload.inputs.size()))||!matchesInt("uSceneTriangleCount",int(binding.frame.triangles.size()))
        ||!matchesInt("uGratingCount",int(binding.triangles.size()))||!matchesInt("uOpaqueVisibility",0)
        ||!matchesFloat("uOpaqueControl",batch->queryControl.data(),4)||!matchesFloat("uCamera",cameraUniform.data(),4)
        ||!matchesFloat("uRotation",rotationUniform.data(),4)||!matchesFloat("uProjection",projection.data(),4)
        ||!matchesFloat("uExtent",extent.data(),2)||!matchesFloat("uGlass",glass.data(),4)
        ||!matchesFloat("uIor[0]",binding.frame.ior.data(),4)||!matchesFloat("uIor[1]",binding.frame.ior.data()+4,4)
        ||!matchesFloat("uWavelength[0]",diffractionmaterial::kProductionWavelengthsNanometres.data(),4)
        ||!matchesFloat("uWavelength[1]",diffractionmaterial::kProductionWavelengthsNanometres.data()+4,4))
        return refuse("OpenGL camera trace native camera, IOR or query uniforms changed");
    int generic=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&generic);
    int token=0;auto restore=[&](int*){gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(generic));};
    std::unique_ptr<int,decltype(restore)> guard(&token,restore);
    const auto matchesBuffer=[&](unsigned buffer,const void* bytes,std::size_t size) {
        if(!buffer||!isBuffer(buffer))return false;
        if(buffer!=batch->sceneBuffer) {
            std::array<char,96> actual{};int length=0;
            getLabel(GL_BUFFER,buffer,int(actual.size()),&length,actual.data());
            if(actual!=batch->program->label)return false;
        }
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffer);GLint64 actual=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
        if(actual!=GLint64(size))return false;
        const auto* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(size),GL_MAP_READ_BIT);
        if(!mapped)return false;
        const bool same=std::memcmp(mapped,bytes,size)==0;
        return gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)==GL_TRUE&&same;
    };
    if(!matchesBuffer(batch->program->directBuffers[0],batch->upload.inputs.data(),batch->upload.inputs.size()*sizeof(gc::NativeGlassCameraGratingTraceInput))
        ||!matchesBuffer(batch->program->directBuffers[1],batch->packed.data(),batch->packed.size()*sizeof(gc::opengl::CameraGlassGratingPacked))
        ||!matchesBuffer(batch->program->directBuffers[2],binding.witnessTexels.data(),binding.witnessTexels.size()*sizeof(gc::Texel))
        ||!matchesBuffer(batch->sceneBuffer,binding.frame.triangleTexels.data(),binding.frame.triangleTexels.size()*sizeof(gc::Texel)))
        return refuse("OpenGL camera glass grating trace actual input/scene/result buffers changed");
    if(!glIsTexture(batch->queryTexture))return refuse("OpenGL camera glass grating trace production query texture expired");
    int texture=0,packBuffer=0;glGetIntegerv(GL_TEXTURE_BINDING_2D,&texture);glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING,&packBuffer);
    constexpr std::array<GLenum,6> packNames{GL_PACK_ALIGNMENT,GL_PACK_ROW_LENGTH,GL_PACK_SKIP_PIXELS,GL_PACK_SKIP_ROWS,GL_PACK_SWAP_BYTES,GL_PACK_LSB_FIRST};
    std::array<int,6> pack{};for(std::size_t at=0;at<pack.size();++at)glGetIntegerv(packNames[at],&pack[at]);
    int textureToken=0;auto restoreTexture=[&](int*) {
        glBindTexture(GL_TEXTURE_2D,unsigned(texture));gl.BindBuffer(GL_PIXEL_PACK_BUFFER,unsigned(packBuffer));
        for(std::size_t at=0;at<pack.size();++at)glPixelStorei(packNames[at],pack[at]);
    };std::unique_ptr<int,decltype(restoreTexture)> textureGuard(&textureToken,restoreTexture);
    glBindTexture(GL_TEXTURE_2D,batch->queryTexture);int width=0,height=0,format=0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&width);glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&height);
    glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_INTERNAL_FORMAT,&format);
    if(width!=int(opaquevisibility::kTextureWidth)||height!=int(binding.frame.query.height)||format!=GL_RGBA32F)
        return refuse("OpenGL camera glass grating trace production query allocation changed");
    std::vector<gc::Texel> query(binding.frame.query.texels.size());
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER,0);for(std::size_t at=0;at<pack.size();++at)glPixelStorei(packNames[at],at?0:1);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,query.data());
    if(query.capacity()>2*query.size()||query!=binding.frame.query.texels||glGetError()!=GL_NO_ERROR)
        return refuse("OpenGL camera glass grating trace production query bytes changed");
    std::uint64_t bytes=causticDiagnostics.admittedFrameBytes,work=binding.bounds.combinedWork;
    for(auto owned:{directGratingBytes,cameraGlassTraceBytes,queryRetainedBytes,nativeAttemptBytes,gratingCameraSpectrumBytes,gratingCellSourceBytes,gratingCoverageBytes,
        staticResources->queryContext->deferredBytes.load(std::memory_order_relaxed)})
        if(!renderpassoutput::detail::checkedAdd(bytes,owned,bytes))return refuse("OpenGL camera glass grating trace aggregate storage overflow");
    for(auto reserved:{directGratingWork,cameraGlassTraceWork,gratingNativeWork,gratingCameraWork,gratingCellWork,gratingCoverageWork})
        if(!renderpassoutput::detail::checkedAdd(work,reserved,work))return refuse("OpenGL camera glass grating trace aggregate work overflow");
    if(bytes>renderpassoutput::kMaximumTotalBytes||work>gc::kMaximumQueryWork||!batch->nativeBytes||batch->nativeBytes>batch->ownedBytes)
        return refuse("OpenGL camera glass grating trace aggregate admission changed");
    // This retained result was decoded from the completed native buffer whose
    // exact packed bytes were checked above. Caller edits cannot replace it.
    if(!gc::sameGlassCameraGratingTraceReadback(supplied,batch->readback))
        return refuse("OpenGL camera glass grating trace differs from its actual retained native result");
    error.clear();return true;
#endif
}
bool readOpenGlGlassCameraGratingTraceStorageForFixture(const NativeFixtureSceneFrame& nativeFrame,
    std::array<unsigned,5>& buffers,std::uintptr_t& fence,std::string& error)
{
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    const auto* batch=frame?frame->cameraGlassTraceBatch.get():nullptr;
    if(!batch||!frame->validateCameraTraceProductionQuery(batch->readback.submission,batch->readback,error))
    {if(error.empty())error="OpenGL camera fixture has no actual completed storage owner";return false;}
    buffers={batch->program->directBuffers[0],batch->program->directBuffers[1],batch->program->directBuffers[2],batch->sceneBuffer,batch->queryTexture};fence=reinterpret_cast<std::uintptr_t>(batch->program->directFence);
    error.clear();return true;
}
bool readOpenGlQueryArithmeticEvidenceForFixture(const NativeFixtureSceneFrame& nativeFrame,OpenGlQueryArithmeticRole role,
    std::shared_ptr<const OpenGlQueryArithmeticEvidence>& output,std::string& error)
{
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||std::size_t(role)>=frame->queryEvidence.size()||!frame->queryEvidence[std::size_t(role)])
    {error="OpenGL query contract has no completed producer-owned role";return false;}
    const auto& evidence=frame->queryEvidence[std::size_t(role)];
    if(!queryArithmeticEvidenceMatches(frame,evidence,evidence->submission,error))return false;
    output=evidence;error.clear();return true;
}
bool validateOpenGlQueryArithmeticEvidenceForFixture(const NativeFixtureSceneFrame& nativeFrame,
    const std::shared_ptr<const OpenGlQueryArithmeticEvidence>& evidence,const glasscaustics::SubmissionIdentity& expected,std::string& error)
{
    std::lock_guard<std::mutex> lock(fixtureMutex());
    return queryArithmeticEvidenceMatches(dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame),evidence,expected,error);
}

// GL-only read-only diagnostic declared by the native fixture support. Private
// receipts never become public Color/AOV views or cross the helper wire.
bool readOpenGlRoughGlassReceiptForFixture(const NativeFixtureSceneFrame& nativeFrame,
    std::array<NativeRawPassPixels, 3>& receipt, NativeRawPassPixels& objectIds,
    NativeRawPassPixels& materialIds, std::string& error)
{
    receipt = {};
    objectIds = materialIds = {};
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame = dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if (!frame || !frame->roughReceiptComplete || glfwGetCurrentContext() != frame->ownerContext)
    { error = "OpenGL rough diagnostic requires a completed frame and its current owning context"; return false; }
    std::array<NativeRawPassPixels, 3> candidate;
    NativeRawPassPixels objects, materials;
    for (std::size_t i = 0; i < candidate.size(); ++i)
        if (!frame->readRoughReceiptOwned(i, candidate[i], error)) return false;
    if (!frame->readRawPassOwned(renderpassoutput::Output::ObjectId, objects, error)
        || !frame->readRawPassOwned(renderpassoutput::Output::MaterialId, materials, error)) return false;
    receipt = std::move(candidate);
    objectIds = std::move(objects);
    materialIds = std::move(materials);
    return true;
}

bool readOpenGlRoughGlassTileSummaryForFixture(const NativeFixtureSceneFrame& nativeFrame,
    std::shared_ptr<const roughglasstiles::Schedule>& schedule, roughglasstiles::Aggregate& aggregate,
    std::string& error)
{
    schedule.reset(); aggregate = {};
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame = dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if (!frame || !frame->roughReceiptComplete || !frame->roughTileSchedule || !frame->staticResources
        || glfwGetCurrentContext()!=frame->ownerContext
        || frame->rendererGeneration_!=frame->staticResources->rendererGeneration
        || frame->width_!=frame->roughTileSchedule->key().width
        || frame->height_!=frame->roughTileSchedule->key().height
        || frame->roughTileAggregate.completedTiles!=frame->roughTileSchedule->tiles().size()
        || frame->roughTileAggregate.completedPixels!=frame->roughTileSchedule->pixels()
        || !glIsTexture(frame->texture_)
        || !roughglasstiles::sameOwner(frame->staticResources->snapshot,frame->roughTileSchedule->key().scene)
        || !roughglasstiles::sameOwner(frame->staticResources->smoothGlass.sourceMaterial,frame->roughTileSchedule->key().program)
        || !roughglasstiles::sameOwner(frame->roughTileSchedule->key().owner,
            std::shared_ptr<const void>(frame->staticResources)))
    { error = "OpenGL rough tile summary requires a complete frame and exact current GPU owner"; return false; }
    schedule = frame->roughTileSchedule; aggregate = frame->roughTileAggregate;
    return true;
}

namespace
{
#if !defined(__APPLE__)
unsigned linkCausticProgram(const arbitgl::GlFuncs& gl,
    const std::vector<std::pair<unsigned,std::string>>& sources,std::string& error)
{
    const auto program=gl.CreateProgram();
    if(!program) {error="OpenGL caustic program allocation failed";return 0;}
    bool retained=false;int token=0;
    auto release=[&](int*){if(!retained)gl.DeleteProgram(program);};
    const std::unique_ptr<int,decltype(release)> guard(&token,release);
    for(const auto& source:sources)
    {
        const auto shader=compileFixtureShader(gl,source.first,source.second.c_str(),error);
        if(!shader)return 0;
        gl.AttachShader(program,shader);gl.DeleteShader(shader);
    }
    gl.LinkProgram(program);int linked=0;gl.GetProgramiv(program,GL_LINK_STATUS,&linked);
    if(linked==GL_TRUE){retained=true;return program;}
    char log[2048]{};int length=0;gl.GetProgramInfoLog(program,sizeof(log),&length,log);
    error="OpenGL caustic program link failed: "+std::string(log,std::size_t(std::max(0,length)));
    return 0;
}
bool causticFence(const arbitgl::GlFuncs& gl,std::string& error)
{
    const auto fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
    if(!fence) {error="OpenGL caustic submission fence allocation failed";return false;}
    const auto status=gl.ClientWaitSync(fence,GL_SYNC_FLUSH_COMMANDS_BIT,30000000000ull);
    gl.DeleteSync(fence);
    if(status!=GL_ALREADY_SIGNALED&&status!=GL_CONDITION_SATISFIED)
    {error="OpenGL caustic submission did not complete before readback";return false;}
    if(glGetError()!=GL_NO_ERROR) {error="OpenGL caustic submission or fence failed";return false;}
    return true;
}
#endif
} // namespace

NativeFixtureSceneSubmission renderOpenGlGlassCausticScene(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSceneResources>& nativeResources,
    HarmonicMIDI::grid::SceneLightId source, HarmonicMIDI::grid::SceneObjectId glassObject,
    HarmonicMIDI::grid::SceneMaterialId glassMaterial,const glasscaustics::AdmittedBranchPolicy& policy,
    glasscaustics::SchedulingDescription scheduling,std::uint32_t width,std::uint32_t height,
    NativeFixtureSceneRuntimeInputs runtime,const glasscaustics::MultipleScatterPolicy* multipleScatterPolicy)
{
    NativeFixtureSceneSubmission refused;
    if(runtime.objectWorlds){refused.error="OpenGL static local object worlds do not yet admit glass caustics";return refused;}
#if defined(__APPLE__)
    (void)scene;(void)nativeResources;(void)source;(void)glassObject;(void)glassMaterial;(void)policy;(void)scheduling;(void)width;(void)height;(void)runtime;(void)multipleScatterPolicy;
    refused.error="OpenGL glass caustics require native OpenGL 4.3 compute storage";return refused;
#else
    const auto started=std::chrono::steady_clock::now();
    namespace gc=glasscaustics;
    using namespace HarmonicMIDI::grid;
    const auto resources=std::dynamic_pointer_cast<const OpenGlFixtureSceneResources>(nativeResources);
    if(!resources||!resources->queryContext||!resources->queryContext->active||resources->gratingLeaf||!gc::consumerCanRender(gc::opengl::kConsumer,NativeFixtureMaterialBackend::OpenGl,resources->smoothGlass.roughness,
            multipleScatterPolicy?multipleScatterPolicy->version:0)
        ||!scene||resources->snapshot!=scene||resources->ownerContext!=glfwGetCurrentContext()
        ||resources->smoothGlass.sourceScene!=scene||resources->smoothGlass.sourceMaterial!=resources->materialProgram
        ||!validFixtureRuntimeInputs(runtime)||!runtime.linearColor
        ||runtime.imageOutput!=renderpassoutput::Output::Color||runtime.rawExportMask||runtime.passProgram||runtime.passComposite
        ||runtime.diffractionEvaluatedProgram||runtime.diffractionSourceProgram||!runtime.diffractionParameters.empty()
        ||prepareNativeNoteInstances(runtime).admitted||resources->instancedSharedGeometry||scene->instancedObject.isValid())
    {refused.error="OpenGL caustics require exact prepared glass scene and runtime owners";return refused;}
    const auto* glassOwner=nativeSurfaceForObject(resources->materialProgram.get(),glassObject);
    if(resources->smoothGlass.object!=glassObject||!glassOwner||glassOwner->authoredMaterial!=glassMaterial
        ||glassOwner->parameters.identifiers[0]!=glassMaterial.value)
    {refused.error="OpenGL caustic selected glass object/material differs from its exact prepared owner";return refused;}
    const auto& gl=resources->gl;
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if((major<4||(major==4&&minor<3))||!gl.DispatchCompute||!gl.MemoryBarrier||!gl.BindBufferBase||!gl.GetIntegeri_v)
    {refused.error="OpenGL glass caustics require native OpenGL 4.3 compute storage";return refused;}
    gc::Prepared prepared;gc::Frame scheduled;
    if(!gc::prepare(resources->smoothGlass,source,policy,prepared,refused.error))return refused;
    std::vector<opaquevisibility::Instance> instances;
    for(std::size_t o=0;o<scene->objectCount;++o)
    {
        SceneTransform3D transform{};
        if(runtimeTargetsObject(runtime,scene->objects[o]))
        {
            transform.translation={runtime.objectTranslationOffset[0],runtime.objectTranslationOffset[1],runtime.objectTranslationOffset[2]};
            transform=applyRuntimeObjectTransform(transform,runtime.objectRotationDegrees,runtime.objectScale);
        }
        instances.push_back({multiplyFixtureMatrices(fixtureTransformMatrix(transform),fixtureWorldMatrix(*scene,scene->objects[o])),o});
    }
    if(multipleScatterPolicy) {
        if(!gc::buildMultipleScatterFrame(prepared,instances,&runtime,width,height,scheduling,*multipleScatterPolicy,scheduled,refused.error))return refused;
    } else if(!gc::buildScheduledFrame(prepared,instances,&runtime,width,height,scheduling,scheduled,refused.error))return refused;
    if(scheduled.sourceDirectionalIrradiance.version) {
        refused.error="OpenGL authored directional irradiance Color requires the completed paired camera consumer";
        return refused;
    }
    const auto pixels=std::uint64_t(width)*height;
    std::shared_ptr<const roughglasstiles::Schedule> cameraRoughSchedule;
    if(scheduled.roughness>0)
    {
        roughglasstiles::FrameKey key;
        key.scene=scene;key.program=resources->materialProgram;
        try {key.runtime=std::make_shared<const NativeFixtureSceneRuntimeInputs>(runtime);}
        catch(const std::bad_alloc&) {refused.error="OpenGL caustic rough camera runtime allocation failed";return refused;}
        key.owner=resources;key.camera=scheduled.camera;
        key.sceneRevision=resources->materialProgram->sceneRevision;key.rendererGeneration=resources->rendererGeneration;
        key.width=width;key.height=height;
        try {
            if(!roughglasstiles::Schedule::admit(resources->smoothGlass,key,false,cameraRoughSchedule,refused.error))return refused;
        } catch(const std::bad_alloc&) {refused.error="OpenGL caustic rough camera schedule allocation failed";return refused;}
    }
    const auto& deposition=scheduled.bounds.deposition;
    const auto elapsed=[](const auto& since) {return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-since).count();};
    OpenGlGlassCausticDiagnostics diagnostics;diagnostics.deposition=deposition;
    // The shared reservation includes native and completed host receipts and
    // eight-band storage. Add the original frame, parallel raster, resolve,
    // host Color check, retained scene/material/geometry, and validation scratch.
    std::uint64_t stride=8+4+4+36+8+16;
    for(const auto format:OpenGlFixtureSceneFrame::rawFormats)stride+=rawPassChannels(format)*rawPassScalarBytes(format);
    std::uint64_t other=0,total=0;
    const auto charge=[&](std::uint64_t count,std::uint64_t size) {
        std::uint64_t amount=0;return renderpassoutput::detail::checkedMultiply(count,size,amount)
            &&renderpassoutput::detail::checkedAdd(other,amount,other);
    };
    // Fixed GLSL arrays and local copies exist even under smaller authored
    // pending limits. Subtract only the shared explicit micro scratch charge.
    bool nativeScratchMemory=true;
    if(scheduled.multipleScatterPolicy) {
        nativeScratchMemory=renderpassoutput::detail::checkedMultiply(scheduled.work.size(),
                gc::opengl::kMultipleScatterNativeTemporaryBytesPerWork,diagnostics.nativeTemporaryBytes)
            &&diagnostics.nativeTemporaryBytes>=scheduled.bounds.microfacetTemporaryBytes
            &&charge(1,diagnostics.nativeTemporaryBytes-scheduled.bounds.microfacetTemporaryBytes);
    }
    const auto chargeMaterial=[&](const NativeFixtureSurfaceMaterialProgram& material) {
        return charge(material.bindingDigest.size(),3)&&charge(material.programIdentity.size(),3)
            &&charge(material.diffractionProductDigest.size(),3)
            &&charge(material.objectPrograms.size(),3*sizeof(std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>))
            &&(!material.importedBaseColorTexture||charge(material.importedBaseColorTexture->texels.size(),3*sizeof(SceneTexelRgba8)));
    };
    bool materialMemory=chargeMaterial(*resources->materialProgram);
    for(const auto& material:resources->materialProgram->objectPrograms)
        materialMemory=materialMemory&&material&&chargeMaterial(*material);
    // The base rough camera pass keeps its native tile receipts and schedule.
    // Admit them together with caustic scratch before either frame can publish.
    bool cameraRoughMemory=true;
    if(cameraRoughSchedule)
        cameraRoughMemory=charge(cameraRoughSchedule->maximumTilePixels(),
                smoothglass::kRoughReceiptBytesPerPixel+roughglasstiles::kTileReadbackBytesPerPixel+sizeof(std::uint32_t))
            &&charge(1,cameraRoughSchedule->retainedBytes()+sizeof(roughglasstiles::Completion)+sizeof(NativeFixtureSceneRuntimeInputs))
            &&charge(cameraRoughSchedule->tiles().size(),roughglasstiles::Completion::metadataBytesPerTile())
            &&charge(2*runtime.animatedLights.capacity(),sizeof(SceneLightRecord))
            &&charge(1,sizeof(*resources)+sizeof(OpenGlFixtureSceneFrame))
            &&charge(4*std::uint64_t(opaquevisibility::kTextureWidth)*opaquevisibility::kTextureHeight,sizeof(opaquevisibility::Texel))
            &&charge(resources->opaqueVisibility.texels.capacity()+resources->smoothGlass.query.texels.capacity(),sizeof(opaquevisibility::Texel))
            &&charge(resources->opaqueVisibility.meshes.capacity()+resources->smoothGlass.query.meshes.capacity(),sizeof(opaquevisibility::Mesh));
    if(cameraRoughSchedule&&smoothglass::multipleScatterCamera(resources->smoothGlass))
        cameraRoughMemory=cameraRoughMemory&&charge(2,kQueryProgramSourceReservation+1024)
            &&charge(1,sizeof(RoughSelectedCamera));
    if(!nativeScratchMemory||!materialMemory||!cameraRoughMemory
        ||!charge(1,kQueryProgramSourceReservation+sizeof(OpenGlQueryContext)+512)
        ||!charge(1,resources->queryContext->deferredBytes.load(std::memory_order_relaxed))
        ||!charge(2,scheduled.identity.capacity()+sizeof(OpenGlQueryArithmeticEvidence)+128)
        ||!charge(1,resources->causticQueryProgram?resources->causticQueryProgram->retainedBytes:0)
        ||!charge(pixels,stride)||!charge(1,resources->retainedTextureBytes)
        ||!charge(3,resources->environmentBudget?resources->environmentBudget->ownedBytes:0)
        ||!charge(3,sizeof(*scene))
        ||!charge(scene->textureTexelCount,sizeof(SceneTexelRgba8)*3)
        ||!charge(prepared.identity.size()+scheduled.identity.size(),4)||!charge(scene->vertexCount,sizeof(SceneVertex)*2)
        ||!charge(scene->indexCount,sizeof(std::uint32_t))||!charge(scene->objectCount,256)
        ||!charge(1+resources->materialProgram->objectPrograms.size(),sizeof(NativeFixtureSurfaceMaterialProgram))
        ||!charge(scheduled.triangles.size(),sizeof(gc::Triangle)*3)
        ||!charge(scheduled.workTexels.size()+scheduled.triangleTexels.size()+scheduled.query.texels.size(),sizeof(gc::Texel)*3)
        ||!charge(scheduled.entries.size(),sizeof(gc::EntrySample)*3)
        ||!charge(scheduled.work.size(),sizeof(gc::EntryWork)*3+256)
        ||!charge(scheduled.bounds.maximumDeposits,256)
        ||!gc::admitFrameMemory(scheduled,other,total,refused.error))
    {if(refused.error.empty())refused.error="OpenGL caustic full resource accounting overflow";return refused;}
    diagnostics.admittedFrameBytes=total;
    const std::array<std::uint64_t,7> bytes{
        scheduled.work.size()*gc::kPathReceiptStride*sizeof(gc::Texel),
        scheduled.bounds.maximumDeposits*gc::kDepositReceiptStride*sizeof(gc::Texel),
        pixels*4*sizeof(float),
        scheduled.workTexels.size()*sizeof(gc::Texel),scheduled.triangleTexels.size()*sizeof(gc::Texel),
        deposition.paddedKeyCount*sizeof(gc::DepositionKey),pixels*4*sizeof(float)};
    const auto integer64=reinterpret_cast<PFNGLGETINTEGER64VPROC>(glfwGetProcAddress("glGetInteger64v"));
    const auto indexed64=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
    if(!integer64||!indexed64) {refused.error="OpenGL caustic storage limit query unavailable";return refused;}
    GLint64 maximumBlock=0;integer64(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&maximumBlock);
    int maximumTexture=0,bindings=0,computeBlocks=0,groups=0,maximumThreads=0,maximumLocalX=0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maximumTexture);glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,&computeBlocks);gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&groups);
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&maximumThreads);
    gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0,&maximumLocalX);
    if(maximumThreads<32||maximumLocalX<32||maximumBlock<=0||bindings<7||computeBlocks<5||groups<=0||scheduled.work.size()>std::uint64_t(groups)*32
        ||maximumTexture<int(width)||maximumTexture<int(height)||maximumTexture<int(opaquevisibility::kTextureWidth)
        ||maximumTexture<int(scheduled.query.height)
        ||std::any_of(bytes.begin(),bytes.end(),[&](auto size){return !size||size>std::uint64_t(maximumBlock);}))
    {refused.error="OpenGL caustic admitted storage exceeds native buffer, dispatch or texture limits";return refused;}
    if(!gc::admitDepositionDevice(deposition,{std::uint64_t(maximumBlock),std::uint64_t(groups),32,4},refused.error))return refused;
    cameraRoughSchedule.reset();
    // Base Color stays private and scene-linear until the caustic readback gate.
    auto baseRuntime=runtime;baseRuntime.linearColor=true;baseRuntime.imageOutput=renderpassoutput::Output::Color;
    baseRuntime.passProgram.reset();baseRuntime.passComposite.reset();
    const auto baseStarted=std::chrono::steady_clock::now();
    auto result=nativeFixtureSceneBackend().render(scene,nativeResources,width,height,baseRuntime);
    if(!result.rendered)return result;
    diagnostics.baseMilliseconds=elapsed(baseStarted);
    auto frame=std::const_pointer_cast<OpenGlFixtureSceneFrame>(std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(result.frame));
    if(!frame) {refused.error="OpenGL caustic base frame has a foreign owner";return refused;}
    std::lock_guard<std::mutex> lock(fixtureMutex());
    OpenGlFixtureState previous(gl);
    if(resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max())
    {previous.restore();frame->releaseTexturesUnlocked();frame->ownerContext=nullptr;
        NativeFixtureSceneSubmission failure;failure.error="OpenGL caustic submission identity exhausted";return failure;}
    const auto serial=++resources->causticSubmissionSerial;
    const gc::SubmissionIdentity submission{scheduled.identity,resources->rendererGeneration,serial,serial,serial,serial,serial};
    std::array<int,7> savedBindings{};
    std::array<GLint64,7> savedStarts{},savedSizes{};
    // Indexed range bindings are restored with their original offsets/sizes.
    for(unsigned i=0;i<7;++i)
    {
        gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,i,&savedBindings[i]);
        indexed64(GL_SHADER_STORAGE_BUFFER_START,i,&savedStarts[i]);
        indexed64(GL_SHADER_STORAGE_BUFFER_SIZE,i,&savedSizes[i]);
    }
    int savedStorage=0,savedPack=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&savedStorage);
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING,&savedPack);
    const auto bindRange=reinterpret_cast<PFNGLBINDBUFFERRANGEPROC>(glfwGetProcAddress("glBindBufferRange"));
    if(!bindRange) {previous.restore();frame->releaseTexturesUnlocked();frame->ownerContext=nullptr;
        refused.error="OpenGL caustic indexed binding restoration unavailable";return refused;}
    unsigned rasterFbo=0,resolveFbo=0,vao=0,resolveTexture=0;
    const auto transportVersion=scheduled.multipleScatterPolicy?gc::kPrivateMultipleScatterVersion:0u;
    const bool samePrograms=resources->causticProgramMultipleScatterVersion==transportVersion&&resources->causticQueryProgram
        &&resources->causticQueryProgram->program==resources->causticPrograms[0]&&resources->causticQueryProgram->liveUnlocked();
    std::array<unsigned,6> programs=samePrograms?resources->causticPrograms:std::array<unsigned,6>{};
    auto queryProgram=samePrograms?resources->causticQueryProgram:std::shared_ptr<OpenGlQueryProgram>{};
    bool cachedPrograms=programs[5]!=0;
    const auto cleanup=[&] {
        if(!cachedPrograms)for(std::size_t index=0;index<programs.size();++index)
            if(programs[index]&&(index!=0||!queryProgram))gl.DeleteProgram(programs[index]);
        if(rasterFbo)gl.DeleteFramebuffers(1,&rasterFbo);
        if(resolveFbo)gl.DeleteFramebuffers(1,&resolveFbo);
        if(vao)gl.DeleteVertexArrays(1,&vao);
        if(resolveTexture)glDeleteTextures(1,&resolveTexture);
        for(unsigned i=0;i<7;++i)
            if(savedBindings[i]&&savedSizes[i]>0)bindRange(GL_SHADER_STORAGE_BUFFER,i,unsigned(savedBindings[i]),savedStarts[i],savedSizes[i]);
            else gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,i,unsigned(savedBindings[i]));
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(savedStorage));gl.BindBuffer(GL_PIXEL_PACK_BUFFER,unsigned(savedPack));
        previous.restore();
    };
    const auto fail=[&](const std::string& error) {
        cleanup();frame->releaseTexturesUnlocked();frame->ownerContext=nullptr;
        NativeFixtureSceneSubmission failure;failure.error=error;return failure;
    };
    const auto baseOwnerValid=[&] {
        if(frame->staticResources!=resources||frame->rendererGeneration_!=resources->rendererGeneration
            ||frame->ownerContext!=glfwGetCurrentContext()||frame->width_!=width||frame->height_!=height
            ||!roughglasstiles::sameOwner(resources->snapshot,scene))return false;
        if(scheduled.roughness==0)return true;
        const auto& schedule=frame->roughTileSchedule;
        return frame->roughReceiptComplete&&schedule&&result.stats.roughGlassReceiptComplete
            &&roughglasstiles::sameOwner(schedule->key().scene,scene)
            &&roughglasstiles::sameOwner(schedule->key().program,resources->materialProgram)
            &&roughglasstiles::sameOwner(schedule->key().owner,std::static_pointer_cast<const void>(resources))
            &&schedule->key().rendererGeneration==resources->rendererGeneration
            &&schedule->key().sceneRevision==resources->materialProgram->sceneRevision
            &&schedule->key().width==width&&schedule->key().height==height
            &&roughglasstiles::cameraValues(schedule->key().camera)==roughglasstiles::cameraValues(scheduled.camera)
            &&schedule->prepared().roughness==scheduled.roughness
            &&frame->roughTileAggregate.completedTiles==schedule->tiles().size()
            &&frame->roughTileAggregate.completedPixels==pixels
            &&frame->roughTileAggregate.reservedQueries==schedule->reservedQueries();
    };
    if(!baseOwnerValid())return fail("OpenGL caustic base Color lacks its complete exact rough camera owner");
    auto readback=std::make_shared<gc::NativeReadback>();readback->frameIdentity=scheduled.identity;
    readback->spectralRadiance.assign(std::size_t(pixels)*8,0.0f);
    if(!cachedPrograms)
    {
        try {
        queryProgram=linkQueryArithmeticProgram(gl,resources->queryContext,resources->rendererGeneration,
            {{GL_COMPUTE_SHADER,scheduled.multipleScatterPolicy?gc::opengl::multipleScatterComputeSource():
            scheduled.roughness>0?gc::opengl::roughComputeSource():gc::opengl::computeSource()}},result.error);
        programs[0]=queryProgram?queryProgram->program:0;
        programs[1]=programs[0]?linkCausticProgram(gl,{{GL_VERTEX_SHADER,gc::opengl::rasterVertexSource()},
            {GL_FRAGMENT_SHADER,gc::opengl::rasterFragmentSource}},result.error):0;
        std::string resolveHead=kDiffractionResolveFragmentShaderHead;
        smoothglass::replace(resolveHead,"#version 330 core","#version 430 core");
        std::string resolveTail=kDiffractionResolveFragmentShaderTail;
        smoothglass::replace(resolveTail,"if (outRawMask <= 0.0) discard;","");
        smoothglass::replace(resolveTail,"vec4 color = texelFetch(uAccumulatedColor, pixel, 0);",
            "vec4 color = texelFetch(uAccumulatedColor,pixel,0); vec3 xyz=vec3(0.0); "
            "uint first=(uint(uExtent.y-1.0-float(pixel.y))*uint(uExtent.x)+uint(pixel.x))*4u; "
            "for(int b=0;b<8;++b) xyz+=causticBandXyz(uintBitsToFloat(b<4?bandsLow[first+uint(b)]:bandsHigh[first+uint(b-4)]),uCie[b]); "
            "color.rgb+=causticXyzToSceneLinear(xyz);");
        programs[2]=programs[1]?linkCausticProgram(gl,{{GL_VERTEX_SHADER,kDiffractionResolveVertexShader},
            {GL_FRAGMENT_SHADER,resolveHead+gc::nativeMathSource()+
                "\nlayout(std430,binding=2) readonly buffer SpectrumLow {uint bandsLow[];};\n"
                "layout(std430,binding=6) readonly buffer SpectrumHigh {uint bandsHigh[];};\nuniform vec4 uCie[8];\nuniform vec2 uExtent;\n"
                +kFixturePassCompositeShader+resolveTail}},result.error):0;
        if(!programs[2])return fail(result.error);
        programs[3]=linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::depositionKeySource}},result.error);
        programs[4]=programs[3]?linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::depositionSortSource}},result.error):0;
        programs[5]=programs[4]?linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::depositionReduceSource()}},result.error):0;
        if(!programs[5])return fail(result.error);
        if(!samePrograms)for(std::size_t index=0;index<resources->causticPrograms.size();++index)
            if(resources->causticPrograms[index]&&(index!=0||!resources->causticQueryProgram))gl.DeleteProgram(resources->causticPrograms[index]);
        resources->causticQueryProgram=queryProgram;
        resources->causticPrograms=programs;resources->causticProgramMultipleScatterVersion=transportVersion;
        cachedPrograms=true;result.stats.shaderProgramBuildCount+=6;
        } catch(const std::bad_alloc&) {return fail("OpenGL query program source/identity allocation failed");}
    }
    gl.GenBuffers(int(frame->causticBuffers.size()),frame->causticBuffers.data());
    for(unsigned i=0;i<7;++i)
    {
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->causticBuffers[i]);
        const void* input=(i==2||i==6)?static_cast<const void*>(readback->spectralRadiance.data())
            :i==3?static_cast<const void*>(scheduled.workTexels.data()):i==4?static_cast<const void*>(scheduled.triangleTexels.data()):nullptr;
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(bytes[i]),input,(i<3||i>=5)?GL_DYNAMIC_COPY:GL_STATIC_DRAW);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,i,frame->causticBuffers[i]);
    }
    gl.ActiveTexture(GL_TEXTURE0);gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);glPixelStorei(GL_UNPACK_ROW_LENGTH,0);glPixelStorei(GL_UNPACK_SKIP_PIXELS,0);glPixelStorei(GL_UNPACK_SKIP_ROWS,0);
    glGenTextures(int(frame->causticTextures.size()),frame->causticTextures.data());
    for(unsigned i=0;i<4;++i)
    {
        glBindTexture(GL_TEXTURE_2D,frame->causticTextures[i]);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        if(i==0)glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,opaquevisibility::kTextureWidth,scheduled.query.height,0,GL_RGBA,GL_FLOAT,scheduled.query.texels.data());
        else if(i==3)glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32F,width,height,0,GL_DEPTH_COMPONENT,GL_FLOAT,nullptr);
        else glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,width,height,0,GL_RGBA,GL_FLOAT,nullptr);
    }
    if(glGetError()!=GL_NO_ERROR||std::any_of(frame->causticTextures.begin(),frame->causticTextures.end(),[](auto t){return !t;})
        ||std::any_of(frame->causticBuffers.begin(),frame->causticBuffers.end(),[](auto b){return !b;}))return fail("OpenGL caustic native allocation failed");
    gl.GenVertexArrays(1,&vao);gl.BindVertexArray(vao);
    gl.GenFramebuffers(1,&rasterFbo);gl.BindFramebuffer(GL_FRAMEBUFFER,rasterFbo);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,frame->causticTextures[1],0);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,frame->causticTextures[2],0);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,frame->causticTextures[3],0);
    const unsigned drawBuffers[]{GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1};gl.DrawBuffers(2,drawBuffers);
    if(!rasterFbo||!vao||gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)return fail("OpenGL caustic private raster attachments unavailable");
    glViewport(0,0,width,height);glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);glDepthMask(GL_TRUE);glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);glDisable(GL_BLEND);for(unsigned i=0;i<8;++i)gl.Disablei(GL_BLEND,i);
    glDisable(GL_SCISSOR_TEST);glDisable(GL_CULL_FACE);glDisable(GL_RASTERIZER_DISCARD);glDisable(GL_COLOR_LOGIC_OP);glDisable(GL_FRAMEBUFFER_SRGB);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);glClearDepth(1);glClear(GL_DEPTH_BUFFER_BIT);
    const float zero[4]{};gl.ClearBufferfv(GL_COLOR,0,zero);gl.ClearBufferfv(GL_COLOR,1,zero);
    const auto camera=gc::cameraPosition(scheduled);const auto& rotation=scheduled.camera.transform.rotation;
    const auto uniforms=[&](unsigned program) {
        gl.UseProgram(program);const auto loc=[&](const char* n){return gl.GetUniformLocation(program,n);};
        gl.Uniform4f(loc("uCamera"),camera[0],camera[1],camera[2],0);
        gl.Uniform4f(loc("uRotation"),rotation.x,rotation.y,rotation.z,rotation.w);
        gl.Uniform4f(loc("uProjection"),std::tan(scheduled.camera.verticalFovRadians*.5f),float(width)/height,scheduled.camera.nearPlane,scheduled.camera.farPlane);
    };
    const auto rasterStarted=std::chrono::steady_clock::now();
    uniforms(programs[1]);
    // Replay the exact admitted world triangles, in the query's owner order.
    // Their private raster IDs are sampled by the GPU, never supplied by a CPU hit.
    for(std::size_t t=0;t<scheduled.triangles.size();++t)
    {gl.Uniform1i(gl.GetUniformLocation(programs[1],"uTriangle"),int(t));glDrawArrays(GL_TRIANGLES,0,3);}
    gl.MemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT);
    if(!causticFence(gl,result.error))return fail(result.error);
    diagnostics.rasterMilliseconds=elapsed(rasterStarted);
    const auto traceStarted=std::chrono::steady_clock::now();
    uniforms(programs[0]);const auto loc=[&](const char* n){return gl.GetUniformLocation(programs[0],n);};
    gl.Uniform1i(loc("uWorkCount"),int(scheduled.work.size()));gl.Uniform1i(loc("uTriangleCount"),int(scheduled.triangles.size()));
    gl.Uniform1i(loc("uVisits"),int(policy.description().maximumVisits));gl.Uniform1i(loc("uInterfaces"),int(policy.description().maximumInterfaces));
    gl.Uniform1i(loc("uPending"),int(policy.description().maximumPending));gl.Uniform2f(loc("uPolicy"),policy.description().maximumUnresolvedFraction,policy.description().minimumBranchFraction);
    gl.Uniform1i(loc("uInterfaceSamples"),int(scheduled.interfaceSamples));gl.Uniform1f(loc("uRoughness"),scheduled.roughness);
    if(scheduled.multipleScatterPolicy) {
        gl.Uniform1i(loc("uMicroEvents"),int(scheduled.multipleScatterPolicy->maximumEvents));
        gl.Uniform1i(loc("uMicroExits"),int(scheduled.multipleScatterPolicy->maximumExits));
        gl.Uniform1f(loc("uMicroResidual"),scheduled.multipleScatterPolicy->maximumResidual);
    }
    gl.Uniform2f(loc("uExtent"),float(width),float(height));gl.Uniform4fv(loc("uSource"),1,scheduled.sourceDirectionTrace.data());
    gl.Uniform4fv(loc("uOpaqueControl"),1,scheduled.query.control.data());gl.Uniform4fv(loc("uIor[0]"),2,scheduled.ior.data());
    gl.Uniform4fv(loc("uLi[0]"),2,scheduled.sourceRadiance[0].data());
    const char* samplers[]{"uOpaqueVisibility","uRasterOwners","uRasterTriangle"};
    for(unsigned i=0;i<3;++i) {gl.ActiveTexture(GL_TEXTURE0+i);gl.BindSampler(i,0);glBindTexture(GL_TEXTURE_2D,frame->causticTextures[i]);gl.Uniform1i(loc(samplers[i]),i);}
    gl.DispatchCompute(unsigned((scheduled.work.size()+31)/32),1,1);
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if(!causticFence(gl,result.error))return fail(result.error);
    diagnostics.traceMilliseconds=elapsed(traceStarted);
    const auto sortStarted=std::chrono::steady_clock::now();
    gl.UseProgram(programs[3]);
    gl.Uniform1i(gl.GetUniformLocation(programs[3],"uWorkCount"),int(scheduled.work.size()));
    gl.Uniform1i(gl.GetUniformLocation(programs[3],"uVisits"),int(policy.description().maximumVisits));
    gl.Uniform1i(gl.GetUniformLocation(programs[3],"uKeyCount"),int(deposition.paddedKeyCount));
    gl.Uniform2f(gl.GetUniformLocation(programs[3],"uExtent"),float(width),float(height));
    gl.DispatchCompute(unsigned((deposition.paddedKeyCount+31)/32),1,1);
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    gl.UseProgram(programs[4]);
    gl.Uniform1i(gl.GetUniformLocation(programs[4],"uKeyCount"),int(deposition.paddedKeyCount));
    for(std::uint64_t span=2;span<=deposition.paddedKeyCount;span*=2)
        for(auto distance=span/2;distance;distance/=2)
        {
            gl.Uniform1i(gl.GetUniformLocation(programs[4],"uSortSpan"),int(span));
            gl.Uniform1i(gl.GetUniformLocation(programs[4],"uSortDistance"),int(distance));
            gl.DispatchCompute(unsigned((deposition.paddedKeyCount/2+31)/32),1,1);
            gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    if(!causticFence(gl,result.error))return fail(result.error);
    diagnostics.sortMilliseconds=elapsed(sortStarted);
    const auto reduceStarted=std::chrono::steady_clock::now();
    gl.UseProgram(programs[5]);
    gl.Uniform1i(gl.GetUniformLocation(programs[5],"uKeyCount"),int(deposition.paddedKeyCount));
    gl.DispatchCompute(unsigned((deposition.paddedKeyCount+31)/32),1,1);
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
    if(!causticFence(gl,result.error))return fail(result.error);
    diagnostics.reduceMilliseconds=elapsed(reduceStarted);
    const auto readbackStarted=std::chrono::steady_clock::now();
    // Shared entry binding relies on this exact frozen table validation.
    if(!gc::validateTables(prepared,scheduled,result.error))return fail(result.error);
    readback->paths.resize(scheduled.work.size());
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->causticBuffers[0]);
    const auto* packedPaths=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[0]),GL_MAP_READ_BIT));
    if(!packedPaths)return fail("OpenGL caustic completed path mapping failed");
    bool decoded=true;std::size_t depositCount=0;
    for(std::size_t w=0;w<readback->paths.size();++w)
    {
        if(!gc::opengl::causticDecodePath(packedPaths+w*gc::kPathReceiptStride,scheduled,w,readback->paths[w])
            ||(scheduled.multipleScatterPolicy&&!gc::bindOpenGlMultipleScatterEntry(scheduled,readback->paths[w],result.error))
            ||readback->paths[w].depositCount>policy.description().maximumVisits
            ||readback->paths[w].depositCount>scheduled.bounds.maximumDeposits-depositCount)
        {decoded=false;break;}
        depositCount+=readback->paths[w].depositCount;
    }
    if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!decoded)return fail("OpenGL caustic path is incomplete or its native receipt is malformed");
    readback->deposits.reserve(depositCount);gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->causticBuffers[1]);
    const auto* packedDeposits=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[1]),GL_MAP_READ_BIT));
    if(!packedDeposits)return fail("OpenGL caustic completed deposit mapping failed");
    for(std::size_t w=0;w<readback->paths.size()&&decoded;++w)for(std::size_t d=0;d<readback->paths[w].depositCount;++d)
    {
        gc::NativeDeposit deposit;
        if(!gc::opengl::causticDecodeDeposit(packedDeposits+(w*policy.description().maximumVisits+d)*gc::kDepositReceiptStride,
                readback->paths[w].multipleScatterVersion,deposit)) {decoded=false;break;}
        readback->deposits.push_back(deposit);
    }
    if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!decoded)return fail("OpenGL caustic deposit receipt is malformed");
    for(unsigned half=0;half<2;++half)
    {
        const unsigned buffer=half?6:2;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->causticBuffers[buffer]);
        const auto* spectrum=static_cast<const float*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[buffer]),GL_MAP_READ_BIT));
        if(!spectrum)return fail("OpenGL caustic completed spectrum mapping failed");
        for(std::size_t pixel=0;pixel<std::size_t(pixels);++pixel)
            std::memcpy(readback->spectralRadiance.data()+pixel*8+half*4,spectrum+pixel*4,4*sizeof(float));
        if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||glGetError()!=GL_NO_ERROR)
            return fail("OpenGL caustic completed native readback failed");
    }
    readback->deposition.keys.resize(std::size_t(deposition.paddedKeyCount));
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->causticBuffers[5]);
    const auto* keys=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[5]),GL_MAP_READ_BIT);
    if(!keys)return fail("OpenGL caustic completed deposition key mapping failed");
    std::memcpy(readback->deposition.keys.data(),keys,std::size_t(bytes[5]));
    if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||glGetError()!=GL_NO_ERROR)
        return fail("OpenGL caustic completed deposition key readback failed");
    for(const auto& key:readback->deposition.keys)if(key.pixel!=std::numeric_limits<std::uint32_t>::max())++readback->deposition.visibleCount;
    readback->deposition.version=gc::kDepositionVersion;readback->deposition.submission=submission;readback->deposition.complete=true;
    for(const auto& path:readback->paths)
    {
        if(path.queryWork>gc::kMaximumQueryWork||diagnostics.queryWork>gc::kMaximumQueryWork-path.queryWork)
            return fail("OpenGL caustic completed query work exceeded admission");
        diagnostics.queryWork+=path.queryWork;
    }
    diagnostics.readbackMilliseconds=elapsed(readbackStarted);
    gc::EnergyLedger energy;
    if(!baseOwnerValid()||!gc::validateNativeReadback(prepared,scheduled,*readback,energy,result.error,&submission))
        return fail(result.error.empty()?"OpenGL caustic frame owner changed before Color publication":result.error);
    // Only validated GPU spectral storage is resolved into product Color.
    const auto resolveStarted=std::chrono::steady_clock::now();
    glGenTextures(1,&resolveTexture);glBindTexture(GL_TEXTURE_2D,resolveTexture);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,0,runtime.linearColor?GL_RGBA16F:GL_RGBA8,width,height,0,GL_RGBA,runtime.linearColor?GL_HALF_FLOAT:GL_UNSIGNED_BYTE,nullptr);
    gl.GenFramebuffers(1,&resolveFbo);gl.BindFramebuffer(GL_FRAMEBUFFER,resolveFbo);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,resolveTexture,0);gl.DrawBuffers(1,drawBuffers);
    if(!resolveTexture||!resolveFbo||gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)return fail("OpenGL caustic Color resolve allocation failed");
    glDisable(GL_DEPTH_TEST);glDepthMask(GL_FALSE);gl.UseProgram(programs[2]);
    const auto resolveLocation=[&](const char* n){return gl.GetUniformLocation(programs[2],n);};
    gl.Uniform2f(resolveLocation("uExtent"),width,height);const auto weights=gc::resolveWeights();gl.Uniform4fv(resolveLocation("uCie[0]"),8,weights[0].data());
    const auto pass=renderpasscomposite::gpuProgram(scenePassProgram(runtime));auto control=pass.control;control[2]=runtime.linearColor?1.0f:0.0f;
    gl.Uniform4fv(resolveLocation("uPassProgramControl"),1,control.data());gl.Uniform4fv(resolveLocation("uPassModes[0]"),8,pass.modes[0].data());
    gl.Uniform4fv(resolveLocation("uPassColors[0]"),8,pass.colors[0].data());gl.Uniform4fv(resolveLocation("uPassReferences[0]"),8,pass.references[0].data());
    gl.Uniform4fv(resolveLocation("uPassTransforms[0]"),8,pass.transforms[0].data());gl.Uniform1i(resolveLocation("uImageOutput"),int(runtime.imageOutput));
    gl.Uniform4f(resolveLocation("uMotionViewport"),width,height,0,0);
    const unsigned textures[]{frame->texture_,frame->depthTexture_,frame->rawTextures[0],frame->rawTextures[1],frame->rawTextures[2],frame->rawTextures[3],frame->rawTextures[4],frame->rawTextures[5]};
    const char* names[]{"uAccumulatedColor","uLinearDepth","uRawNormal","uRawEmission","uRawMask","uRawMaterialId","uRawObjectId","uRawMotion"};
    for(unsigned i=0;i<8;++i) {gl.ActiveTexture(GL_TEXTURE0+i);gl.BindSampler(i,0);glBindTexture(GL_TEXTURE_2D,textures[i]);gl.Uniform1i(resolveLocation(names[i]),i);}
    glDrawArrays(GL_TRIANGLES,0,3);
    if(!causticFence(gl,result.error))return fail(result.error);
    // Read the candidate without replacing the base frame's published view.
    NativeRawPassPixels color;
    auto descriptor=frame->colorTextureDescriptor();descriptor.imageHandle=resolveTexture;descriptor.textureViewHandle=resolveTexture;
    descriptor.format=runtime.linearColor?NativeTexturePixelFormat::Rgba16Float:NativeTexturePixelFormat::Rgba8Unorm;
    if(!frame->readTextureOwned(descriptor,color,result.error)
        ||(runtime.linearColor&&!smoothglass::finiteColorFrame(color,result.error)))return fail(result.error);
    if(!baseOwnerValid())return fail("OpenGL caustic frame lost its complete base owner during Color resolve");
    std::shared_ptr<const OpenGlQueryArithmeticEvidence> queryEvidence;
    try {queryEvidence=completedQueryArithmeticEvidence(queryProgram,OpenGlQueryArithmeticRole::CausticPath,submission);}
    catch(const std::bad_alloc&) {return fail("OpenGL caustic query evidence allocation failed");}
    diagnostics.resolveMilliseconds=elapsed(resolveStarted);
    glDeleteTextures(1,&frame->texture_);frame->texture_=resolveTexture;resolveTexture=0;frame->linearColor=runtime.linearColor;
    frame->causticEnergy=energy;frame->causticReadback=std::move(readback);
    frame->retainQueryArithmetic(OpenGlQueryArithmeticRole::CausticPath,queryProgram,queryEvidence);
    // Existing frame attachment accounting covers the final Color. Report all
    // additional frame-owned native caustic storage once, including SSBOs.
    std::uint64_t owned=scheduled.query.texels.size()*sizeof(gc::Texel)+pixels*36;
    for(auto size:bytes)owned+=size;
    result.stats.textureBytes+=owned;
    diagnostics.nativeStorageBytes=owned;diagnostics.frameMilliseconds=elapsed(started);frame->causticDiagnostics=diagnostics;
    result.stats.drawCount+=scheduled.triangles.size()+1;
    cleanup();return result;
#endif
}

bool readOpenGlGlassCausticReceiptForFixture(const NativeFixtureSceneFrame& nativeFrame,
    std::shared_ptr<const glasscaustics::NativeReadback>& receipt,glasscaustics::EnergyLedger& energy,std::string& error,
    OpenGlGlassCausticDiagnostics* diagnostics)
{
    receipt.reset();std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||!frame->causticReadback||!frame->staticResources||frame->ownerContext!=glfwGetCurrentContext()
        ||frame->rendererGeneration_!=frame->staticResources->rendererGeneration)
    {error="OpenGL caustic diagnostic requires its completed exact native frame owner";return false;}
    receipt=frame->causticReadback;energy=frame->causticEnergy;
    if(diagnostics)*diagnostics=frame->causticDiagnostics;
    error.clear();return true;
}

NativeFixtureScenePreparation prepareOpenGlGlassGratingResources(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& mixedRoot,
    const smoothglass::DistinctGratingAdmission& selection,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& compiledLeaf)
{
    NativeFixtureScenePreparation refused;
    using namespace HarmonicMIDI::grid;
    try {
        const auto mode=NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders;
        if(!scene||!mixedRoot||!compiledLeaf||!selection.material||!validateVisual3DScene(*scene).valid()
            ||mixedRoot->kind!=NativeFixtureMaterialKind::DiffractionReflective||mixedRoot->object!=selection.object
            ||!compiledLeaf->objectPrograms.empty()||!smoothglass::validDistinctGrating(*mixedRoot,selection)
            ||!smoothglass::validDistinctGrating(*compiledLeaf,selection)
            ||!validNativeFixtureMaterialPrograms(*mixedRoot,*scene,NativeFixtureMaterialBackend::OpenGl,true,mode)
            ||!validNativeFixtureDiffractionProgram(*compiledLeaf,NativeFixtureMaterialBackend::OpenGl,selection.object,true,mode)
            ||std::tie(mixedRoot->object,mixedRoot->authoredScene,mixedRoot->authoredMaterial,mixedRoot->sceneRevision,
                mixedRoot->structuralRevision,mixedRoot->evaluationRevision,mixedRoot->materialRevision,mixedRoot->bindingDigest,mixedRoot->programIdentity)
                !=std::tie(compiledLeaf->object,compiledLeaf->authoredScene,compiledLeaf->authoredMaterial,compiledLeaf->sceneRevision,
                compiledLeaf->structuralRevision,compiledLeaf->evaluationRevision,compiledLeaf->materialRevision,compiledLeaf->bindingDigest,compiledLeaf->programIdentity)
            ||std::memcmp(&mixedRoot->diffractionPaths[0],&compiledLeaf->diffractionPaths[0],sizeof(mixedRoot->diffractionPaths[0]))!=0)
        {refused.error="OpenGL private mixed preparation requires the exact compiled grating root and leaf";return refused;}
        std::uint64_t total=0;
        const auto charge=[&](std::uint64_t count,std::uint64_t size) {
            std::uint64_t amount=0;return renderpassoutput::detail::checkedMultiply(count,size,amount)
                &&renderpassoutput::detail::checkedAdd(total,amount,total);
        };
        const auto materialCharge=[&](const NativeFixtureSurfaceMaterialProgram& owner) {
            return charge(4,sizeof(owner))&&charge(owner.bindingDigest.size()+owner.programIdentity.size()+owner.diffractionProductDigest.size(),4)
                &&charge(owner.objectPrograms.size(),4*sizeof(std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>))
                &&(!owner.importedBaseColorTexture||charge(owner.importedBaseColorTexture->texels.size(),4*sizeof(SceneTexelRgba8)));
        };
        bool materialMemory=materialCharge(*mixedRoot)&&materialCharge(*compiledLeaf);
        for(const auto& owner:mixedRoot->objectPrograms)materialMemory=materialMemory&&owner&&materialCharge(*owner);
        // Reserve immutable snapshots, query preparation copies and static native
        // uploads before cloning a Scene or material or allocating GPU storage.
        if(!materialMemory||!charge(4,sizeof(*scene))||!charge(scene->textureTexels.size(),4*sizeof(SceneTexelRgba8))
            ||!charge(4*opaquevisibility::kTextureWidth*opaquevisibility::kTextureHeight,sizeof(opaquevisibility::Texel))
            ||!charge(8,sceneaov::detail::kMaximumSceneBytes)
            ||!charge(scene->vertexCount,2*sizeof(SceneVertex))||!charge(scene->indexCount,2*sizeof(std::uint32_t))
            ||!charge(1,sizeof(OpenGlFixtureSceneResources))||total>renderpassoutput::kMaximumTotalBytes)
        {refused.error="OpenGL private mixed static storage exceeds the whole-frame cap";return refused;}
        const auto snapshot=std::make_shared<const Visual3DScene>(*scene);
        const auto leaf=snapshotNativeSurfaceProgram(compiledLeaf);
        auto* backend=dynamic_cast<OpenGlFixtureSceneBackend*>(&nativeFixtureSceneBackend());
        if(!backend){refused.error="OpenGL private mixed preparation has a foreign backend owner";return refused;}
        return backend->prepareGeometryInstancesImpl(snapshot,mixedRoot,{},false,nullptr,&selection,leaf,total);
    } catch(const std::bad_alloc&) {refused.error="OpenGL private mixed static allocation failed";return refused;}
}

bool stageOpenGlGlassGratingFrame(
    const std::shared_ptr<const NativeFixtureSceneResources>& nativeResources,HarmonicMIDI::grid::SceneLightId source,
    const glasscaustics::AdmittedBranchPolicy& policy,std::uint32_t width,std::uint32_t height,
    NativeFixtureSceneRuntimeInputs runtime,double evaluationTime,
    std::shared_ptr<const NativeFixtureSceneFrame>& frameOut,
    std::shared_ptr<const glasscaustics::GratingIncidentBinding>& bindingOut,std::string& error,
    glasscaustics::SchedulingDescription scheduling)
{
#if defined(__APPLE__)
    (void)nativeResources;(void)source;(void)policy;(void)width;(void)height;(void)runtime;(void)evaluationTime;(void)frameOut;(void)bindingOut;(void)scheduling;
    error="OpenGL private grating staging requires native OpenGL 4.3 storage";return false;
#else
    namespace gc=glasscaustics;using namespace HarmonicMIDI::grid;
    if(runtime.objectWorlds){error="OpenGL static local object worlds do not yet admit diffraction gratings";return false;}
    try {
        const auto resources=std::dynamic_pointer_cast<const OpenGlFixtureSceneResources>(nativeResources);
        if(!resources||!resources->gratingLeaf||!resources->smoothGlass.distinctGrating||!resources->snapshot
            ||resources->ownerContext!=glfwGetCurrentContext()||!resources->gratingStaticBytes
            ||resources->smoothGlass.sourceScene!=resources->snapshot||resources->smoothGlass.sourceMaterial!=resources->materialProgram
            ||!validFixtureRuntimeInputs(runtime)||!runtime.linearColor||!std::isfinite(evaluationTime)
            ||runtime.imageOutput!=renderpassoutput::Output::Color||runtime.rawExportMask||runtime.passProgram||runtime.passComposite
            ||runtime.diffractionEvaluatedProgram||runtime.diffractionSourceProgram||!runtime.diffractionParameters.empty()
            ||runtime.materialFrameTexture||!runtime.materialFrameTextures.empty()||runtime.canonicalBlockCFrame
            ||runtime.previousMotion||runtime.noteInstanceMapping
            ||prepareNativeNoteInstances(runtime).admitted||resources->instancedSharedGeometry
            ||resources->snapshot->instancedObject.isValid())
        {error="OpenGL grating staging requires exact private mixed owners and static runtime";return false;}
        const auto incumbent=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(frameOut);
        gc::DirectGratingTerminalNativeBudget incumbentBudget;
        if(incumbent&&(incumbent->gratingStagingBinding?!openGlTerminalNativeBudget(*incumbent,incumbentBudget,error)
            :!incumbent->reconcileNativeAccounting(error)))return false;
        gc::Prepared prepared;gc::Frame scheduled;
        if(!gc::prepare(resources->smoothGlass,source,policy,prepared,error))return false;
        std::vector<opaquevisibility::Instance> instances;
        for(std::size_t o=0;o<resources->snapshot->objectCount;++o) {
            SceneTransform3D transform{};
            if(runtimeTargetsObject(runtime,resources->snapshot->objects[o])) {
                transform.translation={runtime.objectTranslationOffset[0],runtime.objectTranslationOffset[1],runtime.objectTranslationOffset[2]};
                transform=applyRuntimeObjectTransform(transform,runtime.objectRotationDegrees,runtime.objectScale);
            }
            instances.push_back({multiplyFixtureMatrices(fixtureTransformMatrix(transform),fixtureWorldMatrix(*resources->snapshot,resources->snapshot->objects[o])),o});
        }
        if(!gc::buildGratingStagingFrame(prepared,instances,&runtime,width,height,scheduling,scheduled,error))return false;
        const bool directional=scheduled.sourceDirectionalIrradiance.version!=0;
        std::uint64_t other=resources->gratingStaticBytes,total=0;const auto pixels=std::uint64_t(width)*height;
        const auto charge=[&](std::uint64_t count,std::uint64_t size) {
            std::uint64_t amount=0;return renderpassoutput::detail::checkedMultiply(count,size,amount)
                &&renderpassoutput::detail::checkedAdd(other,amount,other);
        };
        if((frameOut&&(!incumbent||!incumbent->causticDiagnostics.admittedFrameBytes))
            ||(incumbent&&(!charge(1,incumbent->causticDiagnostics.admittedFrameBytes)
                ||!charge(1,incumbent->gratingCameraSpectrumBytes)||!charge(1,incumbent->gratingCellSourceBytes)||!charge(1,incumbent->gratingCoverageBytes)
                ||!charge(1,incumbent->queryRetainedBytes)||!charge(1,incumbent->directGratingBytes)||!charge(1,incumbent->nativeAttemptBytes)
                ||!charge(1,incumbent->cameraGlassTraceBytes)))
            ||(bindingOut&&(!charge(1,bindingOut->bounds.bytes)||!charge(1,bindingOut->frame.bounds.transientBytes))))
        {error="OpenGL private staging incumbent ownership or memory is invalid";return false;}
        gc::GratingIncidentBounds adjunct;
        // Binding preparation itself copies its frozen Frame and query authority.
        if(!gc::calculateGratingIncidentBounds(scheduled,kMaximumNativeDiffractionTriangles,adjunct,error)
            ||!resources->queryContext||!resources->queryContext->active
            ||!charge(1,resources->queryContext->deferredBytes.load(std::memory_order_relaxed))
            ||!charge(3,adjunct.bytes)||!charge(pixels,36+fixtureLinearCameraStorageStride())
            ||!charge(runtime.animatedLights.capacity(),sizeof(SceneLightRecord))
            ||!charge(1,sizeof(OpenGlFixtureSceneFrame))
            // Historical raster source/identity and actual uniform words stay
            // owned by this frame. Driver-private executable memory is unknown.
            ||!charge(2,kQueryProgramSourceReservation+128)
            ||!charge(prepared.identity.size()+scheduled.identity.size(),8)
            ||(directional&&(!charge(1,sizeof(gc::NativeDirectionalSourceInput)+sizeof(gc::NativeDirectionalSourceUpload))
                ||!charge(2,scheduled.identity.size())))
            ||!gc::admitFrameMemory(scheduled,other,total,error))return false;
        std::shared_ptr<const gc::GratingIncidentBinding> binding;
        if(!gc::prepareGratingIncidentBinding(prepared,scheduled,resources->gratingLeaf,
            resources->smoothGlass.distinctGrating->material,evaluationTime,binding,error,
            directional?gc::SourceMeasureUse::DirectionalIrradiancePreparation:gc::SourceMeasureUse::LegacyNative))return false;
        // The exact admitted binding can require more material/identity storage
        // than the conservative geometry-only bound. Charge that delta too.
        if(binding->bounds.bytes>adjunct.bytes&&(!charge(3,binding->bounds.bytes-adjunct.bytes)
            ||!gc::admitFrameMemory(scheduled,other,total,error)))return false;
        if(directional&&binding->identity.size()>scheduled.identity.size()
            &&(!charge(2,binding->identity.size()-scheduled.identity.size())
                ||!gc::admitFrameMemory(scheduled,other,total,error)))return false;

        const auto& gl=resources->gl;
        const auto buffer64=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
        const auto indexed64=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
        const auto bindRange=reinterpret_cast<PFNGLBINDBUFFERRANGEPROC>(glfwGetProcAddress("glBindBufferRange"));
        const auto integer64=reinterpret_cast<PFNGLGETINTEGER64VPROC>(glfwGetProcAddress("glGetInteger64v"));
        int major=0,minor=0,limit=0,bindings=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
        glGetIntegerv(GL_MAX_TEXTURE_SIZE,&limit);glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
        GLint64 block=0;if(integer64)integer64(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&block);
        if((major<4||(major==4&&minor<3))||!buffer64||!indexed64||!bindRange||!integer64||!gl.GetIntegeri_v
            ||bindings<7||block<=0||limit<int(std::max(width,height))||limit<int(opaquevisibility::kTextureWidth)
            ||limit<int(scheduled.query.height)||scheduled.workTexels.size()*sizeof(gc::Texel)>std::uint64_t(block)
            ||scheduled.triangleTexels.size()*sizeof(gc::Texel)>std::uint64_t(block))
        {error="OpenGL grating staging exceeds native query, raster or storage limits";return false;}
        std::unique_lock<std::mutex> lock(fixtureMutex());OpenGlFixtureState previous(gl);
        auto frame=std::make_shared<OpenGlFixtureSceneFrame>();frame->ownerContext=resources->ownerContext;frame->staticResources=resources;
        frame->rendererGeneration_=resources->rendererGeneration;frame->width_=width;frame->height_=height;frame->linearColor=true;
        frame->causticDiagnostics.admittedFrameBytes=total;frame->causticDiagnostics.deposition=scheduled.bounds.deposition;
        int storage=0,indexed=0;GLint64 start=0,size=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&storage);
        gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,4,&indexed);indexed64(GL_SHADER_STORAGE_BUFFER_START,4,&start);indexed64(GL_SHADER_STORAGE_BUFFER_SIZE,4,&size);
        unsigned raster=0,fbo=0,vao=0,cameraFbo=0;bool cleaned=false;
        const auto cleanup=[&] {
            if(cleaned){return;}cleaned=true;
            if(raster&&!frame->rasterCaptureProgram)gl.DeleteProgram(raster);
            if(fbo)gl.DeleteFramebuffers(1,&fbo);
            if(vao)gl.DeleteVertexArrays(1,&vao);
            if(cameraFbo)gl.DeleteFramebuffers(1,&cameraFbo);
            if(indexed&&size>0)bindRange(GL_SHADER_STORAGE_BUFFER,4,unsigned(indexed),start,size);else gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,4,unsigned(indexed));
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(storage));previous.restore();
        };
        bool published=false;
        int token=0;auto release=[&](int*){cleanup();if(!published){frame->releaseTexturesUnlocked();frame->ownerContext=nullptr;}};const std::unique_ptr<int,decltype(release)> guard(&token,release);
        const auto fail=[&](const char* message) {error=message;cleanup();frame->releaseTexturesUnlocked();frame->ownerContext=nullptr;return false;};
        if(resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max())return fail("OpenGL private staging submission serial exhausted");
        frame->rasterCaptureProgram=linkRegisteredOpenGlProgram(gl,resources->queryContext,resources->rendererGeneration,
            {{GL_VERTEX_SHADER,gc::opengl::rasterVertexSource()},{GL_FRAGMENT_SHADER,gc::opengl::rasterFragmentSource}},error,false);
        raster=frame->rasterCaptureProgram?frame->rasterCaptureProgram->program:0;
        if(!raster) {const auto message=error;return fail(message.c_str());}
        if(directional) {
            const auto input=gc::directionalSourceInput(prepared,scheduled);
            gl.GenBuffers(1,&frame->directionalSourceBuffer);gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->directionalSourceBuffer);
            gl.BufferData(GL_SHADER_STORAGE_BUFFER,sizeof(input),&input,GL_STATIC_DRAW);
            GLint64 actual=0;buffer64(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(!frame->directionalSourceBuffer||actual!=sizeof(input))return fail("OpenGL directional source native input allocation changed");
            frame->causticDiagnostics.nativeStorageBytes+=sizeof(input);
        }
        for(unsigned slot:{3u,4u}) {
            const auto& table=slot==3?scheduled.workTexels:scheduled.triangleTexels;
            gl.GenBuffers(1,&frame->causticBuffers[slot]);gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->causticBuffers[slot]);
            gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(table.size()*sizeof(gc::Texel)),table.data(),GL_STATIC_DRAW);
            GLint64 actual=0;buffer64(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(!frame->causticBuffers[slot]||actual<0||std::uint64_t(actual)!=table.size()*sizeof(gc::Texel))return fail("OpenGL private staging table allocation differs from admission");
            if(slot==4)gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,4,frame->causticBuffers[slot]);
            frame->causticDiagnostics.nativeStorageBytes+=std::uint64_t(actual);
        }
        gl.ActiveTexture(GL_TEXTURE0);gl.BindSampler(0,0);gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);glPixelStorei(GL_UNPACK_ROW_LENGTH,0);glPixelStorei(GL_UNPACK_SKIP_PIXELS,0);glPixelStorei(GL_UNPACK_SKIP_ROWS,0);
        frame->privateCameraAttachments=true;
        frame->gratingCameraEvaluationTime=evaluationTime;
        if(!allocateFixtureCameraAttachments(gl,frame.get(),true,cameraFbo,
            frame->gratingCameraDepthAttachment,error)
            ||!fixtureCameraAttachmentsMatch(*frame,error)) {const auto message=error;return fail(message.c_str());}
        // No camera draw has completed. Keep every view private until the exact
        // discrete camera evaluator and the combined finalizer validate it.
        frame->causticDiagnostics.nativeStorageBytes+=pixels*fixtureLinearCameraStorageStride();
        glGenTextures(int(frame->causticTextures.size()),frame->causticTextures.data());
        for(unsigned i=0;i<4;++i) {
            glBindTexture(GL_TEXTURE_2D,frame->causticTextures[i]);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            const int tw=i?int(width):int(opaquevisibility::kTextureWidth),th=i?int(height):int(scheduled.query.height);
            glTexImage2D(GL_TEXTURE_2D,0,i==3?GL_DEPTH_COMPONENT32F:GL_RGBA32F,tw,th,0,i==3?GL_DEPTH_COMPONENT:GL_RGBA,GL_FLOAT,
                i==0?static_cast<const void*>(scheduled.query.texels.data()):nullptr);
            int aw=0,ah=0;glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&aw);glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&ah);
            if(!frame->causticTextures[i]||aw!=tw||ah!=th)return fail("OpenGL private staging raster/query allocation differs from admission");
            frame->causticDiagnostics.nativeStorageBytes+=std::uint64_t(tw)*th*(i==3?4:16);
        }
        gl.GenVertexArrays(1,&vao);gl.BindVertexArray(vao);gl.GenFramebuffers(1,&fbo);gl.BindFramebuffer(GL_FRAMEBUFFER,fbo);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,frame->causticTextures[1],0);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,frame->causticTextures[2],0);
        gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,frame->causticTextures[3],0);
        const unsigned draws[]{GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1};gl.DrawBuffers(2,draws);
        if(!vao||!fbo||gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE||glGetError()!=GL_NO_ERROR)
            return fail("OpenGL private staging raster attachments unavailable");
        glViewport(0,0,width,height);glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);glDepthMask(GL_TRUE);glDepthFunc(GL_LESS);
        glEnable(GL_DEPTH_TEST);glDisable(GL_BLEND);for(unsigned i=0;i<8;++i)gl.Disablei(GL_BLEND,i);
        glDisable(GL_SCISSOR_TEST);glDisable(GL_CULL_FACE);glDisable(GL_RASTERIZER_DISCARD);glDisable(GL_COLOR_LOGIC_OP);glDisable(GL_FRAMEBUFFER_SRGB);
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);glClearDepth(1);glClear(GL_DEPTH_BUFFER_BIT);
        const float zero[4]{};gl.ClearBufferfv(GL_COLOR,0,zero);gl.ClearBufferfv(GL_COLOR,1,zero);
        gl.UseProgram(raster);const auto location=[&](const char* name){return gl.GetUniformLocation(raster,name);};
        const auto camera=gc::cameraPosition(scheduled);const auto& rotation=scheduled.camera.transform.rotation;
        gl.Uniform4f(location("uCamera"),camera[0],camera[1],camera[2],0);gl.Uniform4f(location("uRotation"),rotation.x,rotation.y,rotation.z,rotation.w);
        gl.Uniform4f(location("uProjection"),std::tan(scheduled.camera.verticalFovRadians*.5f),float(width)/height,scheduled.camera.nearPlane,scheduled.camera.farPlane);
        for(std::size_t t=0;t<scheduled.triangles.size();++t) {gl.Uniform1i(location("uTriangle"),int(t));glDrawArrays(GL_TRIANGLES,0,3);}
        gl.MemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT);
        const auto getUniform=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
        if(!getUniform)return fail("OpenGL historical raster uniform capture is unavailable");
        std::array<float,4> actual{};
        for(unsigned item=0;item<3;++item) {
            const char* name=item==0?"uCamera":item==1?"uRotation":"uProjection";
            const auto at=location(name);if(at<0)return fail("OpenGL historical raster uniform is absent");
            getUniform(raster,at,actual.data());
            const unsigned start=item==0?0:item==1?3:7,count=item==0?3:4;
            for(unsigned k=0;k<count;++k)std::memcpy(&frame->rasterCaptureWords[start+k],&actual[k],4);
        }
        glGetIntegerv(GL_VIEWPORT,frame->rasterCaptureViewport.data());
        const float extent[]{float(width),float(height)};
        std::memcpy(frame->rasterCaptureWords.data()+11,extent,sizeof(extent));
        if(frame->rasterCaptureViewport!=std::array<int,4>{0,0,int(width),int(height)})
            return fail("OpenGL historical raster viewport changed");
        frame->rasterCaptureProgram->directFence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!frame->rasterCaptureProgram->directFence)return fail("OpenGL historical raster fence is unavailable");
        const auto rasterWait=gl.ClientWaitSync(frame->rasterCaptureProgram->directFence,GL_SYNC_FLUSH_COMMANDS_BIT,30000000000ull);
        if((rasterWait!=GL_ALREADY_SIGNALED&&rasterWait!=GL_CONDITION_SATISFIED)||glGetError()!=GL_NO_ERROR)
            return fail("OpenGL historical raster capture did not complete");
        if(!directional&&!gc::validateTables(prepared,scheduled,error,true)) {const auto message=error;return fail(message.c_str());}
        const auto serial=++resources->causticSubmissionSerial;
        frame->rasterCaptureGeneration=serial;
        frame->gratingCameraRuntime.emplace(std::move(runtime));
        frame->gratingStagingBinding=binding;
        frame->gratingStagingSubmission={binding->identity,resources->rendererGeneration,serial,serial,serial,serial,serial};
        if(directional) {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->directionalSourceBuffer);
            const auto* nativeInput=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,sizeof(gc::NativeDirectionalSourceInput),GL_MAP_READ_BIT);
            if(!nativeInput)return fail("OpenGL completed directional source input mapping failed");
            std::memcpy(&frame->directionalSourceUpload.input,nativeInput,sizeof(gc::NativeDirectionalSourceInput));
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||glGetError()!=GL_NO_ERROR)
                return fail("OpenGL completed directional source input readback failed");
            frame->directionalSourceUpload.backend=NativeFixtureMaterialBackend::OpenGl;
            frame->directionalSourceUpload.submission=frame->gratingStagingSubmission;
            frame->directionalSourceUpload.bindingGeneration=serial;frame->directionalSourceUpload.complete=true;
            auto bound=scheduled;bound.identity=binding->identity;bound.gratingIncidentBinding=binding;
            if(!gc::validateDirectionalSourceTables(prepared,bound,*frame,error,true)) {const auto message=error;return fail(message.c_str());}
        }

        // Only query/raster staging completed. No path, physical readback or
        // Color/depth publication is synthesized for this private owner.
        cleanup();published=true;lock.unlock();frameOut=std::move(frame);bindingOut=std::move(binding);error.clear();return true;
    } catch(const std::bad_alloc&) {error="OpenGL private mixed staging allocation failed";return false;}
#endif
}

static bool produceOpenGlDirectGratingContributionImpl(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeDirectGratingEndpointInput>& inputs,
    double evaluationTime,glasscaustics::NativeDirectGratingContributionReadback& output,std::string& error,
    OpenGlDirectGratingSource* target)
{
#if defined(__APPLE__)
    (void)nativeFrame;(void)binding;(void)inputs;(void)evaluationTime;(void)output;(void)target;
    error="OpenGL direct grating contribution requires native GL4.3 storage";return false;
#else
    namespace gc=glasscaustics;
    try {
        std::unique_lock<std::mutex> lock(fixtureMutex(),std::defer_lock);if(!target)lock.lock();
        const auto frame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(nativeFrame);
        const auto resources=frame?frame->staticResources:nullptr;
        if(!frame||!resources||!resources->queryContext||!resources->queryContext->active||!binding
            ||(!target&&frame->gratingStagingBinding!=binding)||!resources->gratingLeaf||!resources->smoothGlass.distinctGrating
            ||frame->ownerContext!=glfwGetCurrentContext()||resources->ownerContext!=frame->ownerContext
            ||frame->rendererGeneration_!=resources->rendererGeneration||!frame->privateCameraAttachments||frame->gratingCameraBaseComplete
            ||!frame->gratingCameraRuntime||!std::isfinite(evaluationTime)||evaluationTime!=binding->evaluationTime
            ||frame->gratingCameraEvaluationTime!=evaluationTime||!frame->causticDiagnostics.admittedFrameBytes
            ||frame->width_!=binding->frame.width||frame->height_!=binding->frame.height
            ||!gc::validSourceSubmission(frame->gratingStagingSubmission,frame->gratingStagingBinding->frame,frame->gratingStagingBinding.get())
            ||(target&&(target->frame!=frame.get()||target->binding!=binding||target->cohort!=frame->directGratingCohort.get()))
            ||frame->gratingStagingSubmission.rendererGeneration!=frame->rendererGeneration_
            ||binding->frame.sourceDirectionalIrradiance.version!=1||!frame->directionalSourceUpload.complete
            ||inputs.empty()||inputs.size()>gc::kMaximumEntrySamples||inputs.capacity()>2*inputs.size())
        {if(error.empty())error="OpenGL direct contribution requires exact authored D staging and bounded endpoint inputs";return false;}
        gc::DirectGratingTerminalNativeBudget nativeBudget;
        if(!target&&!openGlTerminalNativeBudget(*frame,nativeBudget,error))return false;
        for(const auto& input:inputs)if(!gc::validDirectGratingEndpointInput(*binding,input))
        {error="OpenGL direct contribution endpoint query is outside the admitted grating";return false;}
        const auto& gl=resources->gl;
        const auto bufferSize=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
        const auto indexedSize=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
        const auto bindRange=reinterpret_cast<PFNGLBINDBUFFERRANGEPROC>(glfwGetProcAddress("glBindBufferRange"));
        const auto integerSize=reinterpret_cast<PFNGLGETINTEGER64VPROC>(glfwGetProcAddress("glGetInteger64v"));
        const auto uniformFloat=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
        const auto uniformInt=reinterpret_cast<PFNGLGETUNIFORMIVPROC>(glfwGetProcAddress("glGetUniformiv"));
        const auto label=reinterpret_cast<PFNGLOBJECTLABELPROC>(glfwGetProcAddress("glObjectLabel"));
        int major=0,minor=0,slots=0,blocks=0,groups=0,threads=0,local=0,textures=0;GLint64 maximum=0;
        glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&slots);glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,&blocks);
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&threads);
        if(gl.GetIntegeri_v) {
            gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&groups);
            gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0,&local);
        }
        glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS,&textures);
        if(integerSize)integerSize(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&maximum);
        const std::array<std::uint64_t,3> nativeSizes{inputs.size()*sizeof(gc::NativeDirectGratingEndpointInput),
            inputs.size()*sizeof(gc::opengl::DirectGratingPacked),binding->witnessTexels.size()*sizeof(gc::Texel)};
        if((major<4||(major==4&&minor<3))||slots<4||blocks<4||groups<int((inputs.size()+31)/32)||threads<32||local<32||textures<1
            ||!bufferSize||!indexedSize||!bindRange||!integerSize||!uniformFloat||!uniformInt||!label||!gl.GetIntegeri_v
            ||!gl.DispatchCompute||!gl.MapBufferRange||!gl.UnmapBuffer||!gl.MemoryBarrier||!gl.FenceSync||!gl.ClientWaitSync||!gl.DeleteSync
            ||maximum<=0||*std::max_element(nativeSizes.begin(),nativeSizes.end())>std::uint64_t(maximum))
        {error="OpenGL direct contribution exceeds native compute, storage or inspection limits";return false;}
        const auto baseBytes=binding->frame.bounds.transientBytes+binding->bounds.bytes;
        if(frame->causticDiagnostics.admittedFrameBytes<baseBytes)
        {error="OpenGL direct contribution original frame accounting is incomplete";return false;}
        std::uint64_t otherBytes=target?target->otherBytes:frame->causticDiagnostics.admittedFrameBytes-baseBytes,
            otherWork=target?target->otherWork:frame->directGratingWork;
        const auto charge=[&](std::uint64_t bytes){return renderpassoutput::detail::checkedAdd(otherBytes,bytes,otherBytes);};
        // Include current adjuncts, earlier retained direct batches, actual
        // program/source construction, query/witness inspection and native locals.
        const auto batchExtra=2*kQueryProgramSourceReservation+2*binding->frame.query.texels.size()*sizeof(gc::Texel)
            +2*nativeSizes[2]+inputs.size()*2048+sizeof(OpenGlDirectGratingBatch)+512
            +2*binding->frame.bounds.transientBytes+2*binding->bounds.bytes+4*binding->identity.capacity();
        // The query-role ledger also retains the new program and native
        // buffers. Admit that independent ledger charge before construction.
        const auto queryReservation=kQueryProgramSourceReservation+nativeSizes[0]+nativeSizes[1]+nativeSizes[2]
            +sizeof(OpenGlQueryArithmeticEvidence)+128+2*binding->identity.capacity()+1;
        bool admitted=true;
        if(!target) {
            admitted=charge(batchExtra)&&charge(queryReservation);
            for(auto bytes:{frame->directGratingBytes,frame->cameraGlassTraceBytes,frame->queryRetainedBytes,frame->nativeAttemptBytes,frame->gratingCameraSpectrumBytes,frame->gratingCellSourceBytes,
                frame->gratingCoverageBytes,resources->queryContext->deferredBytes.load(std::memory_order_relaxed),
                output.samples.capacity()*sizeof(gc::NativeDirectGratingContributionSample),output.submission.frameIdentity.capacity()})
                admitted=admitted&&charge(bytes);
            for(auto work:{frame->cameraGlassTraceWork,frame->gratingNativeWork,frame->gratingCameraWork,frame->gratingCellWork,frame->gratingCoverageWork})
                admitted=admitted&&renderpassoutput::detail::checkedAdd(otherWork,work,otherWork);
        }
        gc::DirectGratingContributionBounds bounds;
        if(!admitted||!gc::calculateDirectGratingContributionBounds(*binding,inputs.size(),otherBytes,otherWork,bounds,error))
        {if(error.empty())error="OpenGL direct contribution whole-frame accounting overflow";return false;}
        std::uint64_t deposition=0;
        if(!frame->directDepositionBudget(deposition)||(!target&&bounds.queryWork+bounds.bandWork>gc::kMaximumDepositionWork-deposition))
        {error="OpenGL direct contribution cumulative deposition cap exceeded";return false;}
        if(target) {
            if(bounds.bytes!=target->admitted.bytes||bounds.combinedWork!=target->admitted.combinedWork
                ||bounds.queryWork!=target->admitted.queryWork||bounds.bandWork!=target->admitted.bandWork)
            {error="OpenGL directional source batch drifted from whole-cohort admission";return false;}
        } else if(!gc::validateDirectionalSourceUpload(binding->prepared,binding->frame,*frame,
            frame->directionalSourceUpload.submission,error,true,binding.get()))return false;
        if(resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max())
        {error="OpenGL direct contribution submission serial exhausted";return false;}
        if(!target&&!frame->reserveNativeAttempt(OpenGlNativeAttemptKind::Direct,binding,{},inputs.size(),0,0,
            binding->frame.bounds.transientBytes+binding->bounds.bytes+otherBytes+bounds.bytes,error))return false;
        auto candidate=std::make_unique<OpenGlDirectGratingBatch>();candidate->bounds=bounds;
        int lifetimeToken=0;auto preserveFailedContext=[&](int*) {
            if(target&&candidate&&candidate->program&&candidate->program->context
                &&glfwGetCurrentContext()!=candidate->program->context->context)deferQueryProgramUnlocked(candidate->program);
        };std::unique_ptr<int,decltype(preserveFailedContext)> lifetimeGuard(&lifetimeToken,preserveFailedContext);
        candidate->ownedBytes=bounds.bytes+batchExtra;candidate->retainedWork=bounds.queryWork+bounds.bandWork;
        candidate->queryTexture=frame->causticTextures[0];candidate->sourceBuffer=target?0:frame->directionalSourceBuffer;
        candidate->queryControl=binding->frame.query.control;
        candidate->upload.otherOwnedBytes=otherBytes;candidate->upload.otherWork=otherWork;
        candidate->upload.inputs.resize(inputs.size());candidate->packed.resize(inputs.size());candidate->readback.samples.reserve(inputs.size());
        if(candidate->upload.inputs.capacity()>2*inputs.size()||candidate->packed.capacity()>2*inputs.size()
            ||candidate->readback.samples.capacity()>2*inputs.size())
        {error="OpenGL direct contribution decoded vector capacity exceeded reservation";return false;}
        const auto source=gc::opengl::directGratingComputeSource();
        if(source.capacity()>kQueryProgramSourceReservation/8)
        {error="OpenGL direct contribution generated source exceeds its reservation";return false;}
        OpenGlFixtureState previous(gl);int generic=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&generic);
        std::array<int,4> indexed{};std::array<GLint64,4> starts{},sizes{};
        for(unsigned slot=0;slot<4;++slot) {
            gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,slot,&indexed[slot]);
            indexedSize(GL_SHADER_STORAGE_BUFFER_START,slot,&starts[slot]);indexedSize(GL_SHADER_STORAGE_BUFFER_SIZE,slot,&sizes[slot]);
        }
        int stateToken=0;auto restoreState=[&](int*) {
            for(unsigned slot=0;slot<4;++slot) {
                if(indexed[slot]&&sizes[slot]>0)bindRange(GL_SHADER_STORAGE_BUFFER,slot,unsigned(indexed[slot]),starts[slot],sizes[slot]);
                else gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,unsigned(indexed[slot]));
            }
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(generic));previous.restore();
        };std::unique_ptr<int,decltype(restoreState)> stateGuard(&stateToken,restoreState);
        candidate->program=linkQueryArithmeticProgram(gl,resources->queryContext,resources->rendererGeneration,{{GL_COMPUTE_SHADER,source}},error);
        if(!candidate->program)return false;
        gl.GenBuffers(int(candidate->program->directBuffers.size()),candidate->program->directBuffers.data());
        const void* data[]{inputs.data(),nullptr,binding->witnessTexels.data()};
        for(unsigned buffer=0;buffer<3;++buffer) {
            const auto id=candidate->program->directBuffers[buffer];
            if(!id){error="OpenGL direct contribution native buffer allocation failed";return false;}
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,id);
            label(GL_BUFFER,id,-1,candidate->program->label.data());
            gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(nativeSizes[buffer]),data[buffer],GL_DYNAMIC_READ);
            GLint64 actual=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(actual<0||std::uint64_t(actual)!=nativeSizes[buffer]||glGetError()!=GL_NO_ERROR)
            {error="OpenGL direct contribution actual storage differs from its reservation";return false;}
            candidate->nativeBytes+=std::uint64_t(actual);
            candidate->program->retainedBytes+=std::uint64_t(actual);
        }
        if(target) {
            const auto input=gc::directionalSourceInput(binding->prepared,binding->frame);
            gl.GenBuffers(1,&candidate->program->directionalInputBuffer);
            candidate->sourceBuffer=candidate->program->directionalInputBuffer;
            if(!candidate->sourceBuffer){error="OpenGL cohort source buffer allocation failed";return false;}
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,candidate->sourceBuffer);
            label(GL_BUFFER,candidate->sourceBuffer,-1,candidate->program->label.data());
            gl.BufferData(GL_SHADER_STORAGE_BUFFER,sizeof(input),&input,GL_STATIC_DRAW);
            GLint64 actual=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(actual!=sizeof(input)||glGetError()!=GL_NO_ERROR)
            {error="OpenGL cohort immutable source input allocation changed";return false;}
            candidate->nativeBytes+=sizeof(input);candidate->program->retainedBytes+=sizeof(input);
        }
        if(candidate->program->retainedBytes>candidate->ownedBytes||candidate->nativeBytes>candidate->ownedBytes)
        {error="OpenGL direct contribution linked native ownership exceeds admission";return false;}
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,candidate->program->directBuffers[0]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,candidate->program->directBuffers[1]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,candidate->sourceBuffer);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,3,candidate->program->directBuffers[2]);
        gl.ActiveTexture(GL_TEXTURE0);gl.BindSampler(0,0);glBindTexture(GL_TEXTURE_2D,candidate->queryTexture);
        gl.UseProgram(candidate->program->program);
        const auto location=[&](const char* name){return gl.GetUniformLocation(candidate->program->program,name);};
        for(const auto* name:{"uEndpointCount","uGratingCount","uOpaqueVisibility","uOpaqueControl","uSource","uLi[0]","uLi[1]"})
            if(location(name)<0){error="OpenGL direct contribution required uniform is absent";return false;}
        gl.Uniform1i(location("uEndpointCount"),int(inputs.size()));gl.Uniform1i(location("uGratingCount"),int(binding->triangles.size()));
        gl.Uniform1i(location("uOpaqueVisibility"),0);gl.Uniform4fv(location("uOpaqueControl"),1,candidate->queryControl.data());
        gl.Uniform4fv(location("uSource"),1,binding->frame.sourceDirectionTrace.data());gl.Uniform4fv(location("uLi[0]"),2,binding->frame.sourceRadiance[0].data());
        // Bind the concrete query legs to their actual input buffers before
        // dispatch. The completed hook also checks their retained native bytes.
        const std::array<unsigned,4> expectedBuffers{candidate->program->directBuffers[0],candidate->program->directBuffers[1],
            candidate->sourceBuffer,candidate->program->directBuffers[2]};
        for(unsigned slot=0;slot<4;++slot) {
            int actual=0;gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,slot,&actual);
            GLint64 start=0,size=0;indexedSize(GL_SHADER_STORAGE_BUFFER_START,slot,&start);indexedSize(GL_SHADER_STORAGE_BUFFER_SIZE,slot,&size);
            const auto required=slot==2?sizeof(gc::NativeDirectionalSourceInput):nativeSizes[slot==3?2:slot];
            if(unsigned(actual)!=expectedBuffers[slot]||start!=0||size!=GLint64(required))
            {error="OpenGL direct contribution dispatch input binding changed";return false;}
        }
        // Executed unsupported queries and failed fences still consume their
        // admitted work. Refusing publication must not make a retry free.
        if(target) {
            if(!target->cohort->chargedWork) {
                target->cohort->chargedWork=target->cohort->reservedWork;
            }
        }
        gl.DispatchCompute(unsigned((inputs.size()+31)/32),1,1);gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
        candidate->program->directFence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!candidate->program->directFence){error="OpenGL direct contribution native fence allocation failed";return false;}
        const auto completed=gl.ClientWaitSync(candidate->program->directFence,GL_SYNC_FLUSH_COMMANDS_BIT,30000000000ull);
        if((completed!=GL_ALREADY_SIGNALED&&completed!=GL_CONDITION_SATISFIED)||glGetError()!=GL_NO_ERROR)
        {error="OpenGL direct contribution native dispatch or fence did not complete";return false;}
        const auto readBuffer=[&](unsigned buffer,void* target,std::size_t bytes) {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffer);
            const auto* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes),GL_MAP_READ_BIT);
            if(!mapped){return false;}std::memcpy(target,mapped,bytes);
            return gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)==GL_TRUE&&glGetError()==GL_NO_ERROR;
        };
        if(!readBuffer(candidate->program->directBuffers[0],candidate->upload.inputs.data(),nativeSizes[0])
            ||!readBuffer(candidate->program->directBuffers[1],candidate->packed.data(),nativeSizes[1])
            ||std::memcmp(candidate->upload.inputs.data(),inputs.data(),nativeSizes[0])!=0)
        {error="OpenGL direct contribution completed input/output readback failed";return false;}
        for(const auto& packed:candidate->packed) {
            gc::NativeDirectGratingContributionSample sample;
            if(!gc::opengl::decodeDirectGratingContribution(packed,inputs.size(),sample)||!sample.complete
                ||sample.outcome==gc::NativeDirectGratingContributionSample::Outcome::Unsupported)
            {error="OpenGL direct contribution endpoint/source evaluation is unsupported";return false;}
            candidate->readback.queryWork+=sample.queryWork;candidate->readback.bandWork+=packed[26][0];
            candidate->readback.samples.push_back(sample);
        }
        if(target&&!readBuffer(candidate->sourceBuffer,&target->sourceUpload.input,sizeof(gc::NativeDirectionalSourceInput)))
        {error="OpenGL cohort completed source input readback failed";return false;}
        const auto serial=++resources->causticSubmissionSerial;
        gc::SubmissionIdentity submission{binding->identity,resources->rendererGeneration,serial,
            frame->gratingStagingSubmission.rasterGeneration,frame->gratingStagingSubmission.queryGeneration,serial,serial};
        candidate->upload.submission=submission;candidate->upload.generation=serial;candidate->upload.complete=true;
        candidate->readback.version=gc::kDirectGratingContributionVersion;candidate->readback.submission=submission;
        candidate->readback.endpointGeneration=serial;
        if(target) {
            target->sourceUpload.backend=NativeFixtureMaterialBackend::OpenGl;target->sourceUpload.submission=submission;
            target->sourceUpload.bindingGeneration=serial;target->sourceUpload.complete=true;
        }
        candidate->readback.sourceGeneration=target?target->sourceUpload.bindingGeneration:frame->directionalSourceUpload.bindingGeneration;
        candidate->readback.complete=true;
        candidate->evidence=completedQueryArithmeticEvidence(candidate->program,OpenGlQueryArithmeticRole::GratingDirect,submission);
        auto returned=candidate->readback;
        if(target) {
            target->batch=std::move(candidate);
            if(!gc::validateDirectGratingContributionReadback(*binding,*target,submission,evaluationTime,returned,error))return false;
            output=std::move(returned);error.clear();return true;
        }
        const auto role=std::size_t(OpenGlQueryArithmeticRole::GratingDirect);
        const auto oldProgram=frame->queryPrograms[role];const auto oldEvidence=frame->queryEvidence[role];
        candidate->previous=std::move(frame->directGratingBatch);
        const auto oldBytes=frame->directGratingBytes,oldWork=frame->directGratingWork;
        int publicationToken=0;auto rollback=[&](int*) {
            auto previous=std::move(frame->directGratingBatch->previous);
            frame->directGratingBatch=std::move(previous);frame->directGratingBytes=oldBytes;frame->directGratingWork=oldWork;
            frame->retainQueryArithmetic(OpenGlQueryArithmeticRole::GratingDirect,oldProgram,oldEvidence);
        };std::unique_ptr<int,decltype(rollback)> publicationGuard(&publicationToken,rollback);
        frame->directGratingBytes+=candidate->ownedBytes;
        frame->retainQueryArithmetic(OpenGlQueryArithmeticRole::GratingDirect,candidate->program,candidate->evidence);
        frame->directGratingBatch=std::move(candidate);
        if(!gc::validateDirectGratingContributionReadback(*binding,*frame,submission,evaluationTime,returned,error))return false;
        publicationGuard.release();output=std::move(returned);error.clear();return true;
    } catch(const std::bad_alloc&) {error="OpenGL direct contribution admitted allocation failed";return false;}
#endif
}

bool produceOpenGlDirectGratingContribution(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeDirectGratingEndpointInput>& inputs,
    double evaluationTime,glasscaustics::NativeDirectGratingContributionReadback& output,std::string& error)
{return produceOpenGlDirectGratingContributionImpl(nativeFrame,binding,inputs,evaluationTime,output,error,nullptr);}

namespace
{
// Both arguments own snapshots admitted by the shared constructors. Compare
// the canonical prepare() scene/material identity around its eight-digit
// selected-light word, without rebuilding the scene codec or copying a Frame.
// The remaining fields are precisely the source-independent frame/grating
// fields included by directGratingCommonIdentity().
bool openGlFrozenDirectionalCommonMatches(const glasscaustics::GratingIncidentBinding& source,
    const glasscaustics::GratingIncidentBinding& staged)
{
    const auto sourceWord=[](const glasscaustics::Prepared& prepared,std::size_t& offset) {
        if(!prepared.glass.sourceScene||!prepared.glass.sourceMaterial||!prepared.branchPolicy)return false;
        std::size_t tail=prepared.branchPolicy->identity().size();
        const auto& scene=*prepared.glass.sourceScene;
        for(std::size_t i=0;i<scene.objectCount;++i) {
            const auto* surface=nativeSurfaceForObject(prepared.glass.sourceMaterial.get(),scene.objects[i].id);
            if(surface)tail+=48+surface->bindingDigest.size()+8*surface->parameters.identifiers.size();
        }
        if(prepared.identity.size()<tail+8)return false;
        offset=prepared.identity.size()-tail-8;
        constexpr char digits[]="0123456789abcdef";
        for(unsigned i=0;i<8;++i)if(prepared.identity[offset+i]!=digits[(prepared.sceneLight.value>>((7-i)*4))&15])return false;
        return true;
    };
    std::size_t a=0,b=0;
    if(!sourceWord(source.prepared,a)||!sourceWord(staged.prepared,b)||a!=b
        ||source.prepared.identity.compare(0,a,staged.prepared.identity,0,b)!=0
        ||source.prepared.identity.compare(a+8,std::string::npos,staged.prepared.identity,b+8,std::string::npos)!=0
        ||!source.gratingOwner||!staged.gratingOwner||!source.material||!staged.material)return false;
    const auto& x=*source.gratingOwner;const auto& y=*staged.gratingOwner;
    const auto same=[](const auto& lhs,const auto& rhs) {
        return lhs.size()==rhs.size()&&(lhs.empty()||std::memcmp(lhs.data(),rhs.data(),lhs.size()*sizeof(lhs[0]))==0);
    };
    return source.prepared.glass.object==staged.prepared.glass.object
        &&source.prepared.glass.sourceMaterial->backend==staged.prepared.glass.sourceMaterial->backend
        &&source.prepared.glass.sourceMaterial->layoutVersion==staged.prepared.glass.sourceMaterial->layoutVersion
        &&x.object==y.object&&x.authoredMaterial==y.authoredMaterial&&x.sceneRevision==y.sceneRevision
        &&x.structuralRevision==y.structuralRevision&&x.evaluationRevision==y.evaluationRevision&&x.materialRevision==y.materialRevision
        &&x.parameters.identifiers==y.parameters.identifiers
        &&same(x.parameters.baseColorMetallic,y.parameters.baseColorMetallic)
        &&same(x.parameters.emissionRoughness,y.parameters.emissionRoughness)
        &&same(x.parameters.normalOpacity,y.parameters.normalOpacity)
        &&same(x.parameters.transmissionIorClearcoat,y.parameters.transmissionIorClearcoat)
        &&source.material->structuralDigest()==staged.material->structuralDigest()
        &&x.bindingDigest==y.bindingDigest&&x.programIdentity==y.programIdentity
        &&std::memcmp(&source.evaluationTime,&staged.evaluationTime,sizeof(source.evaluationTime))==0
        &&source.frame.cameraIdentity==staged.frame.cameraIdentity&&source.frame.branchPolicyIdentity==staged.frame.branchPolicyIdentity
        &&source.frame.width==staged.frame.width&&source.frame.height==staged.frame.height
        &&source.frame.query.height==staged.frame.query.height&&source.frame.scheduling.version==staged.frame.scheduling.version
        &&source.frame.scheduling.entrySubdivisions==staged.frame.scheduling.entrySubdivisions
        &&source.frame.interfaceSamples==staged.frame.interfaceSamples
        &&std::memcmp(&source.frame.roughness,&staged.frame.roughness,sizeof(source.frame.roughness))==0
        &&source.frame.gratingStaging==staged.frame.gratingStaging
        &&same(source.frame.query.control,staged.frame.query.control)&&same(source.frame.query.texels,staged.frame.query.texels)
        &&same(source.frame.triangleTexels,staged.frame.triangleTexels)&&same(source.witnessTexels,staged.witnessTexels)
        &&same(source.frame.ior,staged.frame.ior);
}

// Mirror the shared cohort's pure capacity phase using its already frozen
// identity length. No candidate, source snapshot, string or table is copied.
bool openGlFrozenCohortBounds(const glasscaustics::DirectGratingIrradianceCohort& cohort,
    std::uint64_t otherBytes,std::uint64_t otherWork,glasscaustics::DirectGratingIrradianceCohortBounds& output,std::string& error)
{
    namespace gc=glasscaustics;
    const auto add=[](std::uint64_t& value,std::uint64_t amount){return renderpassoutput::detail::checkedAdd(value,amount,value);};
    gc::DirectGratingIrradianceCohortBounds bounds;bounds.bytes=otherBytes;bounds.combinedWork=otherWork;bounds.depositionWork=otherWork;
    for(const auto& source:cohort.sources()) {
        gc::DirectGratingContributionBounds batch;
        if(!gc::calculateDirectGratingContributionBounds(*source,cohort.inputs().size(),0,0,batch,error)
            ||!add(bounds.bytes,2*(source->frame.bounds.transientBytes+source->bounds.bytes)+batch.bytes
                +2*sizeof(gc::NativeDirectionalSourceUpload)+4*sizeof(gc::NativeDirectionalSourceInput)
                +2*sizeof(gc::NativeDirectionalSourceEcho)+4*source->identity.size())
            ||!add(bounds.combinedWork,batch.combinedWork)||!add(bounds.queryWork,batch.queryWork)||!add(bounds.bandWork,batch.bandWork)
            ||!add(bounds.depositionWork,source->frame.bounds.deposition.depositionWork+source->bounds.orderWork+batch.bandWork))return false;
    }
    if(!add(bounds.bytes,sizeof(gc::DirectGratingIrradianceCohort)+2*cohort.sources().size()*sizeof(cohort.sources().front())
        +4*cohort.identity().size()+2*cohort.inputs().size()*(sizeof(gc::NativeDirectGratingEndpointInput)+sizeof(gc::DirectGratingEndpointIrradianceSum))
        +2*sizeof(gc::DirectGratingIrradianceCohortReadback))
        ||bounds.bytes>renderpassoutput::kMaximumTotalBytes||bounds.combinedWork>gc::kMaximumQueryWork
        ||bounds.depositionWork>gc::kMaximumDepositionWork)
    {error="OpenGL directional cohort exceeds cumulative frame bounds";return false;}
    output=bounds;return true;
}
}

static bool produceOpenGlDirectGratingIrradianceCohortImpl(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& binding,double evaluationTime,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sourceReadbacks,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error,bool stopAfterReservation)
{
#if defined(__APPLE__)
    (void)nativeFrame;(void)binding;(void)evaluationTime;(void)sourceReadbacks;(void)output;(void)stopAfterReservation;
    error="OpenGL directional cohort requires native GL4.3 storage";return false;
#else
    namespace gc=glasscaustics;
    try {
        std::lock_guard<std::mutex> lock(fixtureMutex());
        const auto frame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(nativeFrame);
        const auto resources=frame?frame->staticResources:nullptr;
        if(!frame||!resources||!binding||!frame->gratingStagingBinding||!resources->queryContext||!resources->queryContext->active
            ||frame->ownerContext!=glfwGetCurrentContext()||resources->ownerContext!=frame->ownerContext
            ||frame->rendererGeneration_!=resources->rendererGeneration||!frame->privateCameraAttachments||frame->gratingCameraBaseComplete
            ||!frame->gratingCameraRuntime||!std::isfinite(evaluationTime)||evaluationTime!=frame->gratingCameraEvaluationTime
            ||binding->sources().empty()||binding->sources().size()>HarmonicMIDI::grid::Visual3DScene::kMaxLights
            ||binding->inputs().empty()||binding->inputs().size()>gc::kMaximumEntrySamples
            ||binding->inputs().capacity()>2*binding->inputs().size())
        {error="OpenGL directional cohort requires an exact current private common frame";return false;}
        gc::DirectGratingTerminalNativeBudget nativeBudget;
        if(!openGlTerminalNativeBudget(*frame,nativeBudget,error))return false;
        const auto& staged=*frame->gratingStagingBinding;
        const auto& scene=*staged.prepared.glass.sourceScene;
        std::size_t sourceIndex=0;
        for(std::size_t i=0;i<scene.lightCount;++i) {
            const auto& light=scene.lights[i];
            if(light.kind!=HarmonicMIDI::grid::SceneLightKind::Directional||light.spectralSource.version!=1
                ||light.spectralSource.directionalIrradiance.version!=1)continue;
            if(sourceIndex>=binding->sources().size()||!binding->sources()[sourceIndex])
            {error="OpenGL directional cohort omits the actual frozen source roster";return false;}
            const auto& source=*binding->sources()[sourceIndex++];
            if(source.prepared.sceneLight!=light.id||evaluationTime!=source.evaluationTime
                ||!source.prepared.branchPolicy||source.frame.gratingIncidentBinding||source.frame.sourceDirectionalIrradiance.version!=1
                ||!openGlFrozenDirectionalCommonMatches(source,staged))
            {error="OpenGL directional cohort differs from the actual frozen scene, camera, query or material";return false;}
            for(const auto& input:binding->inputs())if(!gc::validDirectGratingEndpointInput(source,input))
            {error="OpenGL directional cohort endpoint is outside the frozen source";return false;}
        }
        if(sourceIndex!=binding->sources().size())
        {error="OpenGL directional cohort adds an owner to the actual frozen source roster";return false;}
        if(output.endpoints.capacity()>renderpassoutput::kMaximumTotalBytes/sizeof(gc::DirectGratingEndpointIrradianceSum)
            ||output.submission.frameIdentity.capacity()>renderpassoutput::kMaximumTotalBytes
            ||sourceReadbacks.size()>HarmonicMIDI::grid::Visual3DScene::kMaxLights
            ||sourceReadbacks.capacity()>renderpassoutput::kMaximumTotalBytes/sizeof(gc::NativeDirectGratingContributionReadback))
        {error="OpenGL directional cohort incumbent storage exceeds the original cap";return false;}
        for(const auto& old:sourceReadbacks)if(old.samples.capacity()>renderpassoutput::kMaximumTotalBytes/sizeof(gc::NativeDirectGratingContributionSample)
            ||old.submission.frameIdentity.capacity()>renderpassoutput::kMaximumTotalBytes)
        {error="OpenGL directional cohort incumbent source storage exceeds the original cap";return false;}
        std::uint64_t priorBytes=frame->causticDiagnostics.admittedFrameBytes,priorWork=0;
        const auto add=[](std::uint64_t& total,std::uint64_t amount){return renderpassoutput::detail::checkedAdd(total,amount,total);};
        bool admitted=true;
        for(auto bytes:{frame->directGratingBytes,frame->cameraGlassTraceBytes,frame->queryRetainedBytes,frame->nativeAttemptBytes,frame->gratingCameraSpectrumBytes,
            frame->gratingCellSourceBytes,frame->gratingCoverageBytes,resources->queryContext->deferredBytes.load(std::memory_order_relaxed)})
            admitted=admitted&&add(priorBytes,bytes);
        for(auto work:{frame->directGratingWork,frame->cameraGlassTraceWork,frame->gratingNativeWork,frame->gratingCameraWork,frame->gratingCellWork,frame->gratingCoverageWork})
            admitted=admitted&&add(priorWork,work);
        std::uint64_t otherBytes=priorBytes;
        admitted=admitted&&add(otherBytes,sizeof(OpenGlDirectGratingCohort)+2*binding->identity().capacity()
            +2*binding->sources().size()*sizeof(std::unique_ptr<OpenGlDirectGratingSource>)
            +output.endpoints.capacity()*sizeof(gc::DirectGratingEndpointIrradianceSum)+output.submission.frameIdentity.capacity()
            +sourceReadbacks.capacity()*sizeof(gc::NativeDirectGratingContributionReadback));
        bool incumbentRetained=!output.cohort;
        for(auto* retained=frame->directGratingCohort.get();retained;retained=retained->previous.get())
            incumbentRetained=incumbentRetained||retained->binding==output.cohort;
        if(!incumbentRetained)admitted=admitted&&add(otherBytes,output.cohort->bounds().bytes);
        for(const auto& old:sourceReadbacks)admitted=admitted&&add(otherBytes,
            old.samples.capacity()*sizeof(gc::NativeDirectGratingContributionSample)+old.submission.frameIdentity.capacity());
        for(const auto& source:binding->sources()) {
            const auto count=binding->inputs().size();
            const auto native=count*(sizeof(gc::NativeDirectGratingEndpointInput)+sizeof(gc::opengl::DirectGratingPacked))
                +source->witnessTexels.size()*sizeof(gc::Texel)+sizeof(gc::NativeDirectionalSourceInput);
            // Per-source linked source/program, native/decoded copies and actual
            // callback inspection coexist with the shared cohort/sum reservation.
            admitted=admitted&&add(otherBytes,3*kQueryProgramSourceReservation
                +2*source->frame.query.texels.size()*sizeof(gc::Texel)+2*source->witnessTexels.size()*sizeof(gc::Texel)
                +count*2048+sizeof(OpenGlDirectGratingBatch)+sizeof(OpenGlDirectGratingSource)+1024
                +2*source->frame.bounds.transientBytes+2*source->bounds.bytes+8*source->identity.capacity()
                +native+sizeof(OpenGlQueryArithmeticEvidence)+2*sizeof(gc::NativeDirectionalSourceUpload));
        }
        gc::DirectGratingIrradianceCohortBounds forecast;
        if(!admitted||!openGlFrozenCohortBounds(*binding,otherBytes,priorWork,forecast,error))
        {if(error.empty())error="OpenGL directional cohort whole-frame admission changed";return false;}
        const auto totalBytes=forecast.bytes,totalWork=forecast.combinedWork;
        const auto baseWork=frame->gratingStagingBinding->bounds.combinedWork;
        std::uint64_t cohortDeposition=0,deposition=0;
        for(const auto& source:binding->sources()) {
            if(!add(cohortDeposition,source->frame.bounds.deposition.depositionWork)||!add(cohortDeposition,source->bounds.orderWork))
            {error="OpenGL directional cohort deposition reservation overflow";return false;}
        }
        const auto baseDeposition=frame->gratingStagingBinding->frame.bounds.deposition.depositionWork+frame->gratingStagingBinding->bounds.orderWork;
        if(totalBytes<priorBytes||totalWork<priorWork||totalWork-priorWork<baseWork
            ||cohortDeposition<baseDeposition||!frame->directDepositionBudget(deposition)
            ||cohortDeposition-baseDeposition>gc::kMaximumDepositionWork-deposition
            ||totalWork-priorWork-baseWork>gc::kMaximumDepositionWork-deposition-(cohortDeposition-baseDeposition)
            ||resources->causticSubmissionSerial>std::numeric_limits<std::uint64_t>::max()-binding->sources().size())
        {error="OpenGL directional cohort reservation or serial overflow";return false;}
        if(!frame->reserveNativeAttempt(OpenGlNativeAttemptKind::Cohort,frame->gratingStagingBinding,binding,0,0,0,totalBytes,error))return false;
        // From here every allocation/common-identity refusal keeps the complete
        // authenticated attempt, including its frozen owner fee and work.
        if(stopAfterReservation)throw std::bad_alloc();
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> readmitted;
        if(!gc::prepareDirectGratingIrradianceCohort(binding->sources(),binding->inputs(),otherBytes,priorWork,readmitted,error)
            ||readmitted->identity()!=binding->identity()||readmitted->bounds().bytes!=forecast.bytes
            ||readmitted->bounds().combinedWork!=forecast.combinedWork||readmitted->bounds().queryWork!=forecast.queryWork
            ||readmitted->bounds().bandWork!=forecast.bandWork||readmitted->bounds().depositionWork!=forecast.depositionWork)
        {if(error.empty())error="OpenGL directional cohort frozen admission changed during preparation";return false;}
        std::string common,sourceIdentity;
        if(!gc::directGratingCommonIdentity(staged,common,error))return false;
        for(const auto& source:binding->sources())if(!gc::directGratingCommonIdentity(*source,sourceIdentity,error)||sourceIdentity!=common)
        {error="OpenGL directional cohort common identity changed during preparation";return false;}
        auto candidate=std::make_unique<OpenGlDirectGratingCohort>();candidate->binding=binding;
        candidate->ownedBytes=totalBytes-priorBytes;candidate->reservedWork=totalWork-priorWork-baseWork;
        candidate->reservedDepositionWork=cohortDeposition-baseDeposition;
        candidate->sources.reserve(binding->sources().size());
        if(candidate->sources.capacity()>2*binding->sources().size())
        {error="OpenGL directional cohort source capacity exceeds admission";return false;}
        for(const auto& source:binding->sources()) {
            auto view=std::make_unique<OpenGlDirectGratingSource>();view->frame=frame.get();view->cohort=candidate.get();view->binding=source;
            gc::DirectGratingContributionBounds batch;
            if(!gc::calculateDirectGratingContributionBounds(*source,binding->inputs().size(),0,0,batch,error))return false;
            view->otherBytes=totalBytes-source->frame.bounds.transientBytes-source->bounds.bytes-batch.bytes;
            view->otherWork=totalWork-source->bounds.combinedWork-batch.queryWork-batch.bandWork;
            if(!gc::calculateDirectGratingContributionBounds(*source,binding->inputs().size(),view->otherBytes,view->otherWork,view->admitted,error))return false;
            candidate->sources.push_back(std::move(view));
        }
        std::vector<gc::NativeDirectGratingContributionReadback> returned;returned.resize(binding->sources().size());
        if(returned.capacity()>2*returned.size()){error="OpenGL directional cohort readback capacity exceeds admission";return false;}
        candidate->previous=std::move(frame->directGratingCohort);const auto oldBytes=frame->directGratingBytes;
        int token=0;auto rollback=[&](int*) {
            auto previous=std::move(frame->directGratingCohort->previous);
            frame->directGratingCohort=std::move(previous);frame->directGratingBytes=oldBytes;
            // The dispatched attempt's reservation remains in directGratingWork.
        };std::unique_ptr<int,decltype(rollback)> guard(&token,rollback);
        frame->directGratingBytes+=candidate->ownedBytes;frame->directGratingCohort=std::move(candidate);
        for(std::size_t i=0;i<binding->sources().size();++i)
            if(!produceOpenGlDirectGratingContributionImpl(nativeFrame,binding->sources()[i],binding->inputs(),evaluationTime,
                returned[i],error,frame->directGratingCohort->sources[i].get()))return false;
        auto& retained=*frame->directGratingCohort;retained.submission=returned.back().submission;
        retained.submission.frameIdentity=binding->identity();
        gc::DirectGratingIrradianceCohortReadback summed;
        if(!gc::accumulateDirectGratingIrradianceCohort(binding,*frame,retained.submission,evaluationTime,returned,summed,error))return false;
        guard.release();sourceReadbacks=std::move(returned);output=std::move(summed);error.clear();return true;
    } catch(const std::bad_alloc&) {error="OpenGL directional cohort admitted allocation failed";return false;}
#endif
}

bool produceOpenGlDirectGratingIrradianceCohort(
    const std::shared_ptr<const NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,double time,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sources,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error)
{return produceOpenGlDirectGratingIrradianceCohortImpl(frame,cohort,time,sources,output,error,false);}

bool refuseOpenGlDirectGratingCohortPreparationForFixture(
    const std::shared_ptr<const NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,double time,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sources,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error)
{return produceOpenGlDirectGratingIrradianceCohortImpl(frame,cohort,time,sources,output,error,true);}

bool readOpenGlDirectGratingCohortStorageForFixture(const NativeFixtureSceneFrame& nativeFrame,std::size_t source,
    std::array<unsigned,4>& buffers,unsigned& program,std::uintptr_t& fence,std::string& error,
    std::array<std::uint64_t,2>* reservation)
{
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||!frame->directGratingCohort||source>=frame->directGratingCohort->sources.size()
        ||!frame->directGratingCohort->sources[source]->batch||frame->ownerContext!=glfwGetCurrentContext())
    {error="OpenGL directional cohort storage is unavailable";return false;}
    const auto& batch=*frame->directGratingCohort->sources[source]->batch;
    if(!batch.program){error="OpenGL directional cohort program is unavailable";return false;}
    buffers={batch.program->directBuffers[0],batch.program->directBuffers[1],batch.sourceBuffer,batch.queryTexture};
    program=batch.program->program;fence=reinterpret_cast<std::uintptr_t>(batch.program->directFence);
    if(reservation)*reservation={frame->directGratingBytes,frame->directGratingWork};
    error.clear();return true;
}

bool produceOpenGlDirectGratingTerminalOrders(const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,
    const glasscaustics::SubmissionIdentity& submission,double time,
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sources,
    glasscaustics::DirectGratingTerminalOrderLedger& output,std::string& error)
{
    try {
        if(directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease||directTerminalEntryLock)
        {error="OpenGL terminal entry cannot nest an active native operation";return false;}
        std::unique_lock<std::mutex> lock(fixtureMutex());
        const auto frame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(nativeFrame);
        if(!frame||!cohort||frame->directCohort()!=cohort||frame->ownerContext!=glfwGetCurrentContext())
        {error="OpenGL terminal entry requires the exact current cohort frame";return false;}
        // Begin transfers this existing guard into its lease. Shared prechecks
        // and every subsequent native callback therefore run under one lock.
        directTerminalEntryFrame=frame.get();directTerminalEntryLock=&lock;
        int token=0;auto restore=[&](int*){directTerminalEntryFrame=nullptr;directTerminalEntryLock=nullptr;};
        std::unique_ptr<int,decltype(restore)> guard(&token,restore);
        return glasscaustics::evaluateDirectGratingTerminalOrders(cohort,*frame,submission,time,sources,output,error);
    } catch(const std::bad_alloc&) {error="OpenGL terminal admitted allocation failed";return false;}
}
bool produceOpenGlDirectGratingTerminalReceivers(const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& terminal,
    const glasscaustics::SubmissionIdentity& expected,double time,glasscaustics::DirectGratingTerminalReceiverReadback& output,
    std::string& error,OpenGlTerminalReceiverStopForFixture stop)
{
    try {
        if(directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease||directTerminalEntryLock)
        {error="OpenGL receiver entry cannot nest a native operation";return false;}
        std::unique_lock<std::mutex> lock(fixtureMutex());
        const auto frame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(nativeFrame);
        if(!frame||frame->ownerContext!=glfwGetCurrentContext())
        {error="OpenGL receiver entry requires its exact current native frame";return false;}
        directTerminalEntryFrame=frame.get();directTerminalEntryLock=&lock;directReceiverStop=stop;
        int token=0;auto restore=[&](int*) {
            directTerminalEntryFrame=nullptr;directTerminalEntryLock=nullptr;directReceiverStop=OpenGlTerminalReceiverStopForFixture::None;
        };std::unique_ptr<int,decltype(restore)> guard(&token,restore);
        return glasscaustics::bindDirectGratingTerminalReceivers(terminal,*frame,expected,time,output,error);
    } catch(const std::bad_alloc&) {error="OpenGL receiver admitted allocation failed";return false;}
}
bool readOpenGlDirectGratingTerminalReceiverStateForFixture(const NativeFixtureSceneFrame& nativeFrame,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>& output,
    std::array<unsigned,2>& buffers,std::uintptr_t& fence,std::array<std::uint64_t,3>& counters,std::string& error)
{
    if(directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease||directTerminalEntryLock)return false;
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||frame->ownerContext!=glfwGetCurrentContext()||!frame->reconcileNativeAccounting(error))return false;
    const auto* record=frame->directReceiverResults.get();output=record?record->result:nullptr;
    buffers=record&&record->program?std::array<unsigned,2>{record->program->directBuffers[0],record->program->directBuffers[1]}:std::array<unsigned,2>{};
    fence=record&&record->program?reinterpret_cast<std::uintptr_t>(record->program->directFence):0;
    std::uint64_t count=0;for(auto* owner=record;owner;owner=owner->previous.get())++count;
    counters={frame->nativeAttemptCount,frame->gratingNativeWork,count};error.clear();return true;
}
bool inspectOpenGlDirectGratingTerminalReceiverForFixture(const NativeFixtureSceneFrame& nativeFrame,
    const glasscaustics::DirectGratingTerminalReceiverReadback& value,std::string& error)
{
#if defined(__APPLE__)
    (void)nativeFrame;(void)value;error="OpenGL receiver inspection requires GL4.3";return false;
#else
    if(directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease||directTerminalEntryLock)return false;
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||frame->ownerContext!=glfwGetCurrentContext()||!value.nativeResult||value.nativeResult->owner()!=frame)return false;
    for(auto* record=frame->directReceiverResults.get();record;record=record->previous.get())if(record->result==value.nativeResult) {
        if(!glasscaustics::sameDirectGratingTerminalReceiverReadback(value,record->result->retained(),false))return false;
        glasscaustics::DirectGratingTerminalNativeBudget native;
        if(!openGlTerminalNativeBudget(*frame,native,error)||native.retainedBytes<value.bounds.bytes||native.combinedWork<value.bounds.combinedWork)return false;
        return openGlInspectReceiver(*frame,*record,value,error);
    }
    error="OpenGL receiver result is outside the retained native registrations";return false;
#endif
}

bool readOpenGlDirectGratingTerminalStateForFixture(const NativeFixtureSceneFrame& nativeFrame,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>& output,
    std::array<std::uint64_t,5>& counters,std::string& error)
{
    if(directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease||directTerminalEntryLock){error="OpenGL terminal fixture cannot nest an active native operation";return false;}
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||frame->ownerContext!=glfwGetCurrentContext())
    {error="OpenGL terminal fixture requires the current native frame";return false;}
    output=frame->directTerminalResults?frame->directTerminalResults->result:nullptr;
    std::uint64_t count=0;for(auto* record=frame->directTerminalResults.get();record;record=record->previous.get())++count;
    counters={frame->directGratingBytes,frame->directGratingWork,frame->terminalAttemptWork,frame->terminalAttemptDepositionWork,count};
    error.clear();return true;
}
bool readOpenGlNativeAccountingForFixture(const NativeFixtureSceneFrame& nativeFrame,std::array<std::uint64_t,14>& counters,std::string& error,
    glasscaustics::DirectGratingTerminalNativeBudget* budget)
{
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||frame->ownerContext!=glfwGetCurrentContext()||!frame->reconcileNativeAccounting(error))return false;
    if(budget&&!openGlTerminalNativeBudget(*frame,*budget,error))return false;
    counters={frame->directGratingBytes,frame->directGratingWork,frame->terminalAttemptWork,frame->terminalAttemptDepositionWork,
        frame->nativeAttemptBytes,frame->gratingNativeWork,frame->gratingNativeDepositionWork,frame->cameraGlassTraceWork,
        frame->gratingCameraWork,frame->gratingCellWork,frame->gratingCoverageWork,frame->cameraGlassTraceBytes,frame->queryRetainedBytes,frame->nativeAttemptCount};
    error.clear();return true;
}
bool exchangeOpenGlNativeAccountingForFixture(const NativeFixtureSceneFrame& nativeFrame,std::array<std::uint64_t,14>& counters,std::string& error)
{
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    if(!frame||frame->ownerContext!=glfwGetCurrentContext()||directTerminalThreadLease||directReceiverThreadLease||cameraCoverageThreadLease||frame->activeDirectTerminalLease||frame->activeDirectReceiverLease
        ||counters[13]!=frame->nativeAttemptCount){error="OpenGL accounting fault requires its inactive native frame";return false;}
    const std::array<std::uint64_t*,13> values{&frame->directGratingBytes,&frame->directGratingWork,&frame->terminalAttemptWork,&frame->terminalAttemptDepositionWork,
        &frame->nativeAttemptBytes,&frame->gratingNativeWork,&frame->gratingNativeDepositionWork,&frame->cameraGlassTraceWork,
        &frame->gratingCameraWork,&frame->gratingCellWork,&frame->gratingCoverageWork,&frame->cameraGlassTraceBytes,&frame->queryRetainedBytes};
    for(std::size_t i=0;i<values.size();++i)std::swap(*values[i],counters[i]);
    error.clear();return true;
}

bool produceOpenGlGlassCameraGratingTrace(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeGlassCameraGratingTraceInput>& inputs,
    double evaluationTime,glasscaustics::NativeGlassCameraGratingTraceReadback& output,std::string& error)
{
#if defined(__APPLE__)
    (void)nativeFrame;(void)binding;(void)inputs;(void)evaluationTime;(void)output;
    error="OpenGL camera glass grating trace requires native GL4.3 storage";return false;
#else
    namespace gc=glasscaustics;
    try {
        std::lock_guard<std::mutex> lock(fixtureMutex());
        const auto frame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(nativeFrame);
        const auto resources=frame?frame->staticResources:nullptr;
        if(!frame||!resources||!resources->queryContext||!resources->queryContext->active||!binding
            ||frame->gratingStagingBinding!=binding||!resources->gratingLeaf||!resources->smoothGlass.distinctGrating
            ||frame->ownerContext!=glfwGetCurrentContext()||resources->ownerContext!=frame->ownerContext
            ||frame->rendererGeneration_!=resources->rendererGeneration||!frame->privateCameraAttachments||frame->gratingCameraBaseComplete
            ||!frame->gratingCameraRuntime||!std::isfinite(evaluationTime)||evaluationTime!=binding->evaluationTime
            ||frame->gratingCameraEvaluationTime!=evaluationTime||!frame->causticDiagnostics.admittedFrameBytes
            ||frame->width_!=binding->frame.width||frame->height_!=binding->frame.height
            ||!gc::validSourceSubmission(frame->gratingStagingSubmission,binding->frame,binding.get())
            ||frame->gratingStagingSubmission.rendererGeneration!=frame->rendererGeneration_
            ||binding->frame.roughness!=0||binding->frame.multipleScatterPolicy||!frame->causticBuffers[4]
            ||inputs.empty()||inputs.size()>gc::kMaximumEntrySamples||inputs.capacity()>2*inputs.size())
        {if(error.empty())error="OpenGL camera glass grating trace requires exact private smooth staging and bounded camera samples";return false;}
        gc::DirectGratingTerminalNativeBudget nativeBudget;
        if(!openGlTerminalNativeBudget(*frame,nativeBudget,error))return false;
        if(!gc::admitGlassCameraGratingTraceCamera(binding->frame,error))return false;
        for(const auto& input:inputs) {
            gc::NativeGlassCameraGratingTraceInput canonical;
            if(!gc::makeGlassCameraGratingTraceInput(binding->frame,input.pixel[0],input.pixel[1],input.pixel[2],canonical,error)
                ||!gc::sameGlassCameraGratingTraceInput(input,canonical))
            {error="OpenGL camera trace input differs from its actual frame camera sample";return false;}
        }
        const auto& gl=resources->gl;
        const auto bufferSize=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
        const auto indexedSize=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
        const auto bindRange=reinterpret_cast<PFNGLBINDBUFFERRANGEPROC>(glfwGetProcAddress("glBindBufferRange"));
        const auto integerSize=reinterpret_cast<PFNGLGETINTEGER64VPROC>(glfwGetProcAddress("glGetInteger64v"));
        const auto uniformFloat=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
        const auto uniformInt=reinterpret_cast<PFNGLGETUNIFORMIVPROC>(glfwGetProcAddress("glGetUniformiv"));
        const auto label=reinterpret_cast<PFNGLOBJECTLABELPROC>(glfwGetProcAddress("glObjectLabel"));
        int major=0,minor=0,slots=0,blocks=0,groups=0,threads=0,local=0,textures=0;GLint64 maximum=0;
        glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&slots);glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,&blocks);
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&threads);
        if(gl.GetIntegeri_v) {
            gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&groups);
            gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0,&local);
        }
        glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS,&textures);
        if(integerSize)integerSize(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&maximum);
        const std::array<std::uint64_t,3> nativeSizes{inputs.size()*sizeof(gc::NativeGlassCameraGratingTraceInput),
            inputs.size()*8*sizeof(gc::opengl::CameraGlassGratingPacked),binding->witnessTexels.size()*sizeof(gc::Texel)};
        if((major<4||(major==4&&minor<3))||slots<4||blocks<4||groups<int((inputs.size()*8+31)/32)||threads<32||local<32||textures<1
            ||!bufferSize||!indexedSize||!bindRange||!integerSize||!uniformFloat||!uniformInt||!label||!gl.GetIntegeri_v
            ||!gl.DispatchCompute||!gl.MapBufferRange||!gl.UnmapBuffer||!gl.MemoryBarrier||!gl.FenceSync||!gl.ClientWaitSync||!gl.DeleteSync
            ||maximum<=0||*std::max_element(nativeSizes.begin(),nativeSizes.end())>std::uint64_t(maximum))
        {error="OpenGL camera glass grating trace exceeds native compute, storage or inspection limits";return false;}
        const auto baseBytes=binding->frame.bounds.transientBytes+binding->bounds.bytes;
        if(frame->causticDiagnostics.admittedFrameBytes<baseBytes)
        {error="OpenGL camera glass grating trace original frame accounting is incomplete";return false;}
        std::uint64_t otherBytes=frame->causticDiagnostics.admittedFrameBytes-baseBytes,otherWork=frame->cameraGlassTraceWork;
        const auto charge=[&](std::uint64_t bytes){return renderpassoutput::detail::checkedAdd(otherBytes,bytes,otherBytes);};
        // Include current adjuncts, earlier retained camera batches, actual
        // program/source construction, query/witness inspection and native locals.
        const auto batchExtra=2*kQueryProgramSourceReservation+2*binding->frame.query.texels.size()*sizeof(gc::Texel)
            +2*nativeSizes[2]+2*binding->frame.triangleTexels.size()*sizeof(gc::Texel)+inputs.size()*8*2048+sizeof(OpenGlCameraGlassGratingBatch)+512
            +2*binding->frame.bounds.transientBytes+2*binding->bounds.bytes+4*binding->identity.capacity();
        // The query-role ledger also retains the new program and native
        // buffers. Admit that independent ledger charge before construction.
        const auto queryReservation=kQueryProgramSourceReservation+nativeSizes[0]+nativeSizes[1]+nativeSizes[2]
            +sizeof(OpenGlQueryArithmeticEvidence)+128+2*binding->identity.capacity()+1;
        bool admitted=charge(batchExtra)&&charge(queryReservation);
        for(auto bytes:{frame->directGratingBytes,frame->cameraGlassTraceBytes,frame->queryRetainedBytes,frame->nativeAttemptBytes,frame->gratingCameraSpectrumBytes,frame->gratingCellSourceBytes,
            frame->gratingCoverageBytes,resources->queryContext->deferredBytes.load(std::memory_order_relaxed),
            output.bands.capacity()*sizeof(gc::NativeGlassCameraGratingTraceBand),output.submission.frameIdentity.capacity()})
            admitted=admitted&&charge(bytes);
        for(const auto& band:output.bands)admitted=admitted&&charge(band.visits.capacity()*sizeof(gc::NativeGlassCameraGratingTraceVisit));
        for(auto work:{frame->directGratingWork,frame->gratingNativeWork,frame->gratingCameraWork,frame->gratingCellWork,frame->gratingCoverageWork})
            admitted=admitted&&renderpassoutput::detail::checkedAdd(otherWork,work,otherWork);
        gc::GlassCameraGratingTraceBounds bounds;
        if(!admitted||!gc::calculateGlassCameraGratingTraceBounds(*binding,inputs.size(),otherBytes,otherWork,bounds,error))
        {if(error.empty())error="OpenGL camera glass grating trace whole-frame accounting overflow";return false;}
        std::uint64_t deposition=0;
        if(!frame->directDepositionBudget(deposition)||bounds.queryWork+bounds.branchWork>gc::kMaximumDepositionWork-deposition)
        {error="OpenGL camera trace cumulative deposition cap exceeded";return false;}
        if(resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max())
        {error="OpenGL camera glass grating trace submission serial exhausted";return false;}
        if(!frame->reserveNativeAttempt(OpenGlNativeAttemptKind::Camera,binding,{},inputs.size(),0,0,
            binding->frame.bounds.transientBytes+binding->bounds.bytes+otherBytes+bounds.bytes,error))return false;
        auto candidate=std::make_unique<OpenGlCameraGlassGratingBatch>();candidate->bounds=bounds;
        candidate->ownedBytes=bounds.bytes+batchExtra;candidate->retainedWork=bounds.queryWork+bounds.branchWork;
        candidate->queryTexture=frame->causticTextures[0];candidate->sceneBuffer=frame->causticBuffers[4];
        candidate->queryControl=binding->frame.query.control;
        candidate->upload.otherOwnedBytes=otherBytes;candidate->upload.otherWork=otherWork;
        candidate->upload.inputs.resize(inputs.size());candidate->packed.resize(bounds.maximumBands);candidate->readback.bands.reserve(bounds.maximumBands);
        if(candidate->upload.inputs.capacity()>2*inputs.size()||candidate->packed.capacity()>2*bounds.maximumBands
            ||candidate->readback.bands.capacity()>2*bounds.maximumBands)
        {error="OpenGL camera glass grating trace decoded vector capacity exceeded reservation";return false;}
        const auto source=gc::opengl::cameraGlassGratingComputeSource();
        if(source.capacity()>kQueryProgramSourceReservation/8)
        {error="OpenGL camera glass grating trace generated source exceeds its reservation";return false;}
        OpenGlFixtureState previous(gl);int generic=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&generic);
        std::array<int,4> indexed{};std::array<GLint64,4> starts{},sizes{};
        for(unsigned slot=0;slot<4;++slot) {
            gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,slot,&indexed[slot]);
            indexedSize(GL_SHADER_STORAGE_BUFFER_START,slot,&starts[slot]);indexedSize(GL_SHADER_STORAGE_BUFFER_SIZE,slot,&sizes[slot]);
        }
        int stateToken=0;auto restoreState=[&](int*) {
            for(unsigned slot=0;slot<4;++slot) {
                if(indexed[slot]&&sizes[slot]>0)bindRange(GL_SHADER_STORAGE_BUFFER,slot,unsigned(indexed[slot]),starts[slot],sizes[slot]);
                else gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,unsigned(indexed[slot]));
            }
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(generic));previous.restore();
        };std::unique_ptr<int,decltype(restoreState)> stateGuard(&stateToken,restoreState);
        candidate->program=linkQueryArithmeticProgram(gl,resources->queryContext,resources->rendererGeneration,{{GL_COMPUTE_SHADER,source}},error);
        if(!candidate->program)return false;
        gl.GenBuffers(int(candidate->program->directBuffers.size()),candidate->program->directBuffers.data());
        const void* data[]{inputs.data(),nullptr,binding->witnessTexels.data()};
        for(unsigned buffer=0;buffer<3;++buffer) {
            const auto id=candidate->program->directBuffers[buffer];
            if(!id){error="OpenGL camera glass grating trace native buffer allocation failed";return false;}
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,id);
            label(GL_BUFFER,id,-1,candidate->program->label.data());
            gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(nativeSizes[buffer]),data[buffer],GL_DYNAMIC_READ);
            GLint64 actual=0;bufferSize(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(actual<0||std::uint64_t(actual)!=nativeSizes[buffer]||glGetError()!=GL_NO_ERROR)
            {error="OpenGL camera glass grating trace actual storage differs from its reservation";return false;}
            candidate->nativeBytes+=std::uint64_t(actual);
            candidate->program->retainedBytes+=std::uint64_t(actual);
        }
        if(candidate->program->retainedBytes>candidate->ownedBytes||candidate->nativeBytes>candidate->ownedBytes)
        {error="OpenGL camera glass grating trace linked native ownership exceeds admission";return false;}
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,candidate->program->directBuffers[0]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,candidate->program->directBuffers[1]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,candidate->sceneBuffer);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,3,candidate->program->directBuffers[2]);
        gl.ActiveTexture(GL_TEXTURE0);gl.BindSampler(0,0);glBindTexture(GL_TEXTURE_2D,candidate->queryTexture);
        gl.UseProgram(candidate->program->program);
        const auto location=[&](const char* name){return gl.GetUniformLocation(candidate->program->program,name);};
        for(const auto* name:{"uSampleCount","uSceneTriangleCount","uGratingCount","uOpaqueVisibility","uOpaqueControl",
            "uCamera","uRotation","uProjection","uExtent","uGlass","uIor[0]","uIor[1]","uWavelength[0]","uWavelength[1]"})
            if(location(name)<0){error="OpenGL camera trace required native uniform is absent";return false;}
        gl.Uniform1i(location("uSampleCount"),int(inputs.size()));gl.Uniform1i(location("uSceneTriangleCount"),int(binding->frame.triangles.size()));
        gl.Uniform1i(location("uGratingCount"),int(binding->triangles.size()));
        gl.Uniform1i(location("uOpaqueVisibility"),0);gl.Uniform4fv(location("uOpaqueControl"),1,candidate->queryControl.data());
        const auto camera=gc::cameraPosition(binding->frame);const auto& rotation=binding->frame.camera.transform.rotation;
        gl.Uniform4f(location("uCamera"),camera[0],camera[1],camera[2],0);
        gl.Uniform4f(location("uRotation"),rotation.x,rotation.y,rotation.z,rotation.w);
        gl.Uniform4f(location("uProjection"),std::tan(binding->frame.camera.verticalFovRadians*.5f),float(frame->width_)/frame->height_,
            binding->frame.camera.nearPlane,binding->frame.camera.farPlane);
        gl.Uniform2f(location("uExtent"),float(frame->width_),float(frame->height_));
        gl.Uniform4f(location("uGlass"),float(binding->prepared.glass.object.value&65535u),float(binding->prepared.glass.object.value>>16),
            float(binding->prepared.glass.material.value&65535u),float(binding->prepared.glass.material.value>>16));
        gl.Uniform4fv(location("uIor[0]"),2,binding->frame.ior.data());
        gl.Uniform4fv(location("uWavelength[0]"),2,diffractionmaterial::kProductionWavelengthsNanometres.data());
        // Bind the concrete query legs to their actual input buffers before
        // dispatch. The completed hook also checks their retained native bytes.
        const std::array<unsigned,4> expectedBuffers{candidate->program->directBuffers[0],candidate->program->directBuffers[1],
            candidate->sceneBuffer,candidate->program->directBuffers[2]};
        for(unsigned slot=0;slot<4;++slot) {
            int actual=0;gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,slot,&actual);
            GLint64 start=0,size=0;indexedSize(GL_SHADER_STORAGE_BUFFER_START,slot,&start);indexedSize(GL_SHADER_STORAGE_BUFFER_SIZE,slot,&size);
            const auto required=slot==2?binding->frame.triangleTexels.size()*sizeof(gc::Texel):nativeSizes[slot==3?2:slot];
            if(unsigned(actual)!=expectedBuffers[slot]||start!=0||size!=GLint64(required))
            {error="OpenGL camera glass grating trace dispatch input binding changed";return false;}
        }
        // Executed unsupported queries and failed fences still consume their
        // admitted work. Refusing publication must not make a retry free.
        gl.DispatchCompute(unsigned((inputs.size()*8+31)/32),1,1);gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
        candidate->program->directFence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!candidate->program->directFence){error="OpenGL camera glass grating trace native fence allocation failed";return false;}
        const auto completed=gl.ClientWaitSync(candidate->program->directFence,GL_SYNC_FLUSH_COMMANDS_BIT,30000000000ull);
        if((completed!=GL_ALREADY_SIGNALED&&completed!=GL_CONDITION_SATISFIED)||glGetError()!=GL_NO_ERROR)
        {error="OpenGL camera glass grating trace native dispatch or fence did not complete";return false;}
        const auto readBuffer=[&](unsigned buffer,void* target,std::size_t bytes) {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffer);
            const auto* mapped=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes),GL_MAP_READ_BIT);
            if(!mapped){return false;}std::memcpy(target,mapped,bytes);
            return gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)==GL_TRUE&&glGetError()==GL_NO_ERROR;
        };
        if(!readBuffer(candidate->program->directBuffers[0],candidate->upload.inputs.data(),nativeSizes[0])
            ||!readBuffer(candidate->program->directBuffers[1],candidate->packed.data(),nativeSizes[1])
            ||std::memcmp(candidate->upload.inputs.data(),inputs.data(),nativeSizes[0])!=0)
        {error="OpenGL camera glass grating trace completed input/output readback failed";return false;}
        for(const auto& packed:candidate->packed) {
            gc::NativeGlassCameraGratingTraceBand band;
            if(!gc::decodeGlassCameraGratingTraceBand(packed.header,packed.visits.data(),packed.visits.size(),band,error))return false;
            for(std::size_t unused=band.visits.size();unused<packed.visits.size();++unused)
                if(packed.visits[unused]!=gc::NativeGlassCameraGratingTracePackedVisit{})
                {error="OpenGL camera trace unused native output slots changed";return false;}
            candidate->readback.queryWork+=band.queryWork;candidate->readback.bands.push_back(std::move(band));
        }
        const auto serial=++resources->causticSubmissionSerial;
        gc::SubmissionIdentity submission{binding->identity,resources->rendererGeneration,serial,
            frame->gratingStagingSubmission.rasterGeneration,frame->gratingStagingSubmission.queryGeneration,serial,serial};
        candidate->upload.submission=submission;candidate->upload.generation=serial;candidate->upload.complete=true;
        candidate->readback.version=gc::kGlassCameraGratingTraceVersion;candidate->readback.submission=submission;
        candidate->readback.inputGeneration=serial;candidate->readback.dispatchComplete=true;
        candidate->evidence=completedQueryArithmeticEvidence(candidate->program,OpenGlQueryArithmeticRole::GlassCameraGratingTrace,submission);
        auto returned=candidate->readback;
        const auto role=std::size_t(OpenGlQueryArithmeticRole::GlassCameraGratingTrace);
        const auto oldProgram=frame->queryPrograms[role];const auto oldEvidence=frame->queryEvidence[role];
        candidate->previous=std::move(frame->cameraGlassTraceBatch);
        const auto oldBytes=frame->cameraGlassTraceBytes,oldWork=frame->cameraGlassTraceWork;
        int publicationToken=0;auto rollback=[&](int*) {
            auto previous=std::move(frame->cameraGlassTraceBatch->previous);
            frame->cameraGlassTraceBatch=std::move(previous);frame->cameraGlassTraceBytes=oldBytes;frame->cameraGlassTraceWork=oldWork;
            frame->retainQueryArithmetic(OpenGlQueryArithmeticRole::GlassCameraGratingTrace,oldProgram,oldEvidence);
        };std::unique_ptr<int,decltype(rollback)> publicationGuard(&publicationToken,rollback);
        frame->cameraGlassTraceBytes+=candidate->ownedBytes;
        frame->retainQueryArithmetic(OpenGlQueryArithmeticRole::GlassCameraGratingTrace,candidate->program,candidate->evidence);
        frame->cameraGlassTraceBatch=std::move(candidate);
        if(!gc::validateGlassCameraGratingTraceReadback(*binding,*frame,submission,evaluationTime,returned,error))return false;
        publicationGuard.release();output=std::move(returned);error.clear();return true;
    } catch(const std::bad_alloc&) {error="OpenGL camera glass grating trace admitted allocation failed";return false;}
#endif
}

static bool produceOpenGlGlassGratingReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeGratingReadback& output,std::string& error,
    OpenGlGratingIncidentDiagnostics* diagnostics,glasscaustics::NativeReadback* terminalOutput,
    glasscaustics::EnergyLedger* terminalEnergy,OpenGlGratingCameraDiagnostics* cameraDiagnostics=nullptr,
    std::shared_ptr<const glasscaustics::GratingEntryCellBinding> sourceCells={},
    glasscaustics::NativeGratingCellTransportReadback* cellOutput=nullptr,
    OpenGlGratingCellCoverageReadback* coverageOutput=nullptr,std::size_t maximumLeaves=4,
    OpenGlGratingAttemptStopForFixture stopForFixture=OpenGlGratingAttemptStopForFixture::None)
{
#if defined(__APPLE__)
    (void)nativeFrame;(void)binding;(void)output;(void)diagnostics;(void)terminalOutput;(void)terminalEnergy;(void)cameraDiagnostics;(void)sourceCells;(void)cellOutput;(void)coverageOutput;(void)maximumLeaves;(void)stopForFixture;
    error="OpenGL grating incidents require native OpenGL 4.3 compute storage";return false;
#else
    try
    {
        namespace gc=glasscaustics;
        const bool terminal=terminalOutput&&terminalEnergy;
        const bool cameraPass=cameraDiagnostics!=nullptr;
        const bool cellPass=bool(sourceCells)&&cellOutput;
        const bool coveragePass=coverageOutput!=nullptr;
        if(coveragePass&&(!cellPass||!maximumLeaves||maximumLeaves>gc::kMaximumGratingCellCoverageLeaves))
            {error="OpenGL cell interval coverage requires its bounded optional boundary producer";return false;}
        if(bool(sourceCells)!=bool(cellOutput)) {error="OpenGL cell transport requires both optional source and output";return false;}
        // The old witness can be its frame's last owner. Declaration order
        // releases its binding after unlocking, even on refusal or exception.
        const auto previousCellBinding=cellOutput?cellOutput->binding:nullptr;
        const auto previousCoverageCandidate=coverageOutput?coverageOutput->candidate:nullptr;
        const auto previousCoveragePayload=coverageOutput?coverageOutput->payload:nullptr;
        std::lock_guard<std::mutex> lock(fixtureMutex());
        const auto frame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(nativeFrame);
        const auto resources=frame?frame->staticResources:nullptr;
        const bool privateStaging=frame&&bool(frame->gratingStagingBinding);
        const bool completedOwner=privateStaging
            ?binding&&resources&&terminal&&resources->gratingLeaf&&resources->smoothGlass.distinctGrating
                &&frame->gratingStagingBinding==binding&&binding->frame.gratingStaging&&!frame->causticReadback
                &&frame->privateCameraAttachments&&frame->gratingCameraRuntime
                &&frame->gratingCameraEvaluationTime==binding->evaluationTime
                &&frame->texture_&&frame->depthTexture_&&frame->gratingCameraDepthAttachment
                &&!frame->colorImageHandle()&&!frame->depthImageHandle()
                &&frame->gratingStagingSubmission.frameIdentity==binding->identity
                &&frame->gratingStagingSubmission.rendererGeneration==resources->rendererGeneration
                &&frame->gratingStagingSubmission.submissionSerial>0
                &&frame->gratingStagingSubmission.submissionSerial<=resources->causticSubmissionSerial
                &&frame->gratingStagingSubmission.rasterGeneration==frame->gratingStagingSubmission.submissionSerial
                &&frame->gratingStagingSubmission.queryGeneration==frame->gratingStagingSubmission.submissionSerial
                &&frame->gratingStagingSubmission.scratchGeneration==frame->gratingStagingSubmission.submissionSerial
                &&frame->gratingStagingSubmission.completedFence==frame->gratingStagingSubmission.submissionSerial
            :binding&&frame&&resources&&!resources->gratingLeaf&&frame->causticReadback
                &&frame->causticReadback->frameIdentity==binding->frame.identity
                &&!frame->causticReadback->gratingIncidentVersion&&!frame->causticReadback->gratingIncidents
                &&frame->causticReadback->deposition.complete&&!frame->causticReadback->deposition.overflow
                &&frame->causticReadback->deposition.submission.completedFence
                &&frame->causticReadback->deposition.submission.rendererGeneration==resources->rendererGeneration;
        if(!binding||!frame||!resources||!completedOwner
            ||frame->ownerContext!=glfwGetCurrentContext()||resources->ownerContext!=frame->ownerContext
            ||frame->rendererGeneration_!=resources->rendererGeneration
            ||!binding->prepared.glass.sourceScene||!binding->prepared.glass.sourceMaterial
            ||frame->width_!=binding->frame.width||frame->height_!=binding->frame.height
            ||binding->frame.roughness!=0||binding->frame.multipleScatterPolicy||!frame->causticDiagnostics.admittedFrameBytes
            ||!frame->causticBuffers[3]||!frame->causticBuffers[4]||!glIsTexture(frame->causticTextures[0]))
        {error="OpenGL grating incidents require their completed exact smooth staging frame owner";return false;}
        gc::DirectGratingTerminalNativeBudget nativeBudget;
        if(privateStaging?!openGlTerminalNativeBudget(*frame,nativeBudget,error):!frame->reconcileNativeAccounting(error))return false;
        if(cellPass&&(!privateStaging||!terminal||cameraPass
            ||!glasscaustics::validateGratingEntryCellBinding(*sourceCells,binding,error)))
        {if(error.empty())error="OpenGL cell transport requires its exact retained private point parent";return false;}
        const auto& gl=resources->gl;
        const auto integer64=reinterpret_cast<PFNGLGETINTEGER64VPROC>(glfwGetProcAddress("glGetInteger64v"));
        const auto indexed64=reinterpret_cast<PFNGLGETINTEGER64I_VPROC>(glfwGetProcAddress("glGetInteger64i_v"));
        const auto buffer64=reinterpret_cast<PFNGLGETBUFFERPARAMETERI64VPROC>(glfwGetProcAddress("glGetBufferParameteri64v"));
        const auto bindRange=reinterpret_cast<PFNGLBINDBUFFERRANGEPROC>(glfwGetProcAddress("glBindBufferRange"));
        if(!integer64||!indexed64||!buffer64||!bindRange||!gl.DispatchCompute||!gl.GetIntegeri_v)
        {error="OpenGL grating incident storage and binding authority is unavailable";return false;}
        const auto& scheduled=binding->frame;
        const bool directional=scheduled.sourceDirectionalIrradiance.version!=0;
        gc::GratingCameraBounds cameraBounds;
        gc::DepositionSchedule cameraDeposition;
        if(cameraPass&&(!privateStaging||!terminal||!directional
            ||!gc::calculateGratingCameraBounds(*binding,cameraBounds,error)
            ||!gc::calculateDepositionSchedule(scheduled.width,scheduled.height,
                binding->bounds.maximumOrders,cameraDeposition,error)
            ||cameraDeposition.depositionWork>gc::kMaximumDepositionWork-scheduled.bounds.deposition.depositionWork
            ||cameraDeposition.resolveWork>gc::kMaximumDepositionWork-scheduled.bounds.deposition.resolveWork))
        {if(error.empty())error="OpenGL private camera primitives require retained authored D staging";return false;}
        const auto perQuery=gc::gratingCellQueryAllowance(scheduled);
        const auto interfaces=binding->prepared.branchPolicy->description().maximumInterfaces;
        const auto cameraReservation=cameraPass?cameraBounds.queryWork:0;
        const auto cellReservation=cellPass
            ?binding->bounds.maximumIncidents*gc::kGratingCellTracePaths*(interfaces+1)*perQuery:std::uint64_t(0);
        const auto coverageReservation=coveragePass
            ?binding->bounds.maximumIncidents*maximumLeaves*(interfaces+1)*perQuery:std::uint64_t(0);
        std::uint64_t cumulativeWork=binding->bounds.combinedWork;
        for(auto work:{frame->directGratingWork,frame->cameraGlassTraceWork,frame->gratingNativeWork,frame->gratingCameraWork,
            frame->gratingCellWork,frame->gratingCoverageWork,binding->bounds.combinedWork,cameraReservation,cellReservation,coverageReservation})
            if(!renderpassoutput::detail::checkedAdd(cumulativeWork,work,cumulativeWork))
            {error="OpenGL grating cumulative work accounting overflow";return false;}
        if(cumulativeWork>gc::kMaximumQueryWork)
        {error="OpenGL grating adjuncts exceed the original cumulative direct contribution work cap";return false;}
        std::uint64_t retainedDeposition=0;
        if(privateStaging&&(!frame->directDepositionBudget(retainedDeposition)
            ||cameraReservation+cellReservation+coverageReservation
                >gc::kMaximumDepositionWork-retainedDeposition))
        {error="OpenGL grating adjuncts exceed the cumulative terminal deposition cap";return false;}
        const auto& deposition=scheduled.bounds.deposition;
        const auto pixels=std::uint64_t(scheduled.width)*scheduled.height;
        const auto workBase=binding->bounds.maximumIncidents*gc::kGratingIncidentStride;
        const auto childBase=binding->bounds.maximumOrders*gc::kGratingOrderStride;
        std::array<std::uint64_t,13> bytes{{scheduled.work.size()*gc::kPathReceiptStride*sizeof(gc::Texel),
            (workBase+(terminal?scheduled.work.size()*2:0))*sizeof(gc::Texel),
            (childBase+(terminal?binding->bounds.maximumOrders*8:0))*sizeof(gc::Texel),binding->witnessTexels.size()*sizeof(gc::Texel),
            terminal?scheduled.bounds.maximumDeposits*gc::kDepositReceiptStride*sizeof(gc::Texel):0,
            terminal?deposition.paddedKeyCount*sizeof(gc::DepositionKey):0,
            terminal?pixels*4*sizeof(float):0,terminal?pixels*4*sizeof(float):0,
            directional?sizeof(gc::NativeDirectionalSourceInput)+sizeof(gc::NativeDirectionalSourceEcho)
                +scheduled.work.size()*sizeof(float):0,
            cameraPass?binding->bounds.maximumOrders*gc::opengl::kGratingCameraPrimitiveStride*sizeof(gc::Texel):0,
            cameraPass?cameraDeposition.paddedKeyCount*sizeof(gc::DepositionKey):0,
            cameraPass?pixels*4*sizeof(float):0,cameraPass?pixels*4*sizeof(float):0}};
        // Binding admission includes adjunct GPU storage and typed host copies.
        // Retain the completed base frame, a second canonical binding during the
        // ownership check, private path output, and all fixed native stack copies.
        std::uint64_t total=frame->causticDiagnostics.admittedFrameBytes;
        const auto charge=[&](std::uint64_t count,std::uint64_t size) {
            std::uint64_t amount=0;return renderpassoutput::detail::checkedMultiply(count,size,amount)
                &&renderpassoutput::detail::checkedAdd(total,amount,total);
        };
        const auto chargeMaterial=[&](const NativeFixtureSurfaceMaterialProgram& owner) {
            return charge(1,sizeof(owner))&&charge(owner.bindingDigest.size()+owner.programIdentity.size()+owner.diffractionProductDigest.size(),3)
                &&charge(owner.objectPrograms.size(),sizeof(std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>))
                &&(!owner.importedBaseColorTexture||charge(owner.importedBaseColorTexture->texels.size(),2*sizeof(HarmonicMIDI::grid::SceneTexelRgba8)));
        };
        bool materialMemory=chargeMaterial(*resources->materialProgram);
        for(const auto& owner:resources->materialProgram->objectPrograms)
            materialMemory=materialMemory&&owner&&chargeMaterial(*owner);
        // Caller cell storage and a distinct previous frame remain live while
        // the point phase allocates. Admit them at that first memory gate.
        bool cellMemory=true;
        gc::GratingCellTransportBounds cellWorstBounds;
        gc::GratingCellCoverageBounds coverageWorstBounds;
        if(cellPass) {
            if(!gc::calculateGratingCellTransportBounds(*sourceCells,binding->bounds.maximumIncidents,cellWorstBounds,error))return false;
            cellMemory=charge(2,cellWorstBounds.bytes)&&charge(cellWorstBounds.maximumRecords,2048);
            if(coveragePass) {
                gc::GratingCellTransportBinding boundary;boundary.sourceCells=sourceCells;boundary.bounds=cellWorstBounds;
                if(!gc::calculateGratingCellCoverageBounds(boundary,maximumLeaves,coverageWorstBounds,error))return false;
                // The real boundary identity is formed after point completion.
                // Its complete source/parent identities and fixed lineage words
                // are reserved now with the worst record/sweep/native extents.
                coverageWorstBounds.hostBytes+=4*sourceCells->identity.capacity()+8*binding->identity.capacity()+4096;
                coverageWorstBounds.bytes=coverageWorstBounds.hostBytes+coverageWorstBounds.nativeBytes;
                cellMemory=cellMemory&&charge(2,coverageWorstBounds.bytes)
                    &&charge(cellWorstBounds.maximumRecords*maximumLeaves,2048);
            }
            bool oldMemory=charge(cellOutput->records.capacity(),sizeof(gc::NativeGratingCellTransportRecord));
            for(const auto& record:cellOutput->records)for(const auto& path:record.paths)
                oldMemory=oldMemory&&charge(path.capacity(),sizeof(gc::NativeGratingCellTraceStep));
            if(previousCellBinding) {
                if(!gc::validateGratingCellTransportBinding(*previousCellBinding,error))return false;
                const auto& previous=*previousCellBinding;
                oldMemory=oldMemory&&previous.sourceCells&&previous.sourceCells->incidentBinding
                    &&charge(1,previous.bounds.bytes)&&charge(1,previous.sourceCells->bounds.bytes)
                    &&charge(1,previous.sourceCells->incidentBinding->bounds.bytes);
                if(previous.nativeOwner!=nativeFrame) {
                    const auto previousFrame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(previous.nativeOwner);
                    oldMemory=oldMemory&&previousFrame&&charge(1,previousFrame->causticDiagnostics.admittedFrameBytes)
                        &&previousFrame->reconcileNativeAccounting(error)&&charge(1,previousFrame->nativeAttemptBytes)
                        &&charge(1,previousFrame->gratingCameraSpectrumBytes)&&charge(1,previousFrame->gratingCellSourceBytes)
                        &&charge(1,previousFrame->gratingCoverageBytes)&&charge(1,previousFrame->queryRetainedBytes)
                        &&charge(1,previousFrame->directGratingBytes)&&charge(1,previousFrame->cameraGlassTraceBytes);
                }
            }
            cellMemory=cellMemory&&oldMemory&&charge(1,sourceCells->bounds.bytes)
                &&charge(cellOutput->submission.frameIdentity.capacity(),1);
        }
        if(coveragePass) {
            if(previousCoverageCandidate) {
                const auto& previous=*previousCoverageCandidate;
                const auto& boundary=previous.boundary;
                if(!boundary||!boundary->binding||!gc::validateGratingCellCoverageCandidate(previous,
                    boundary->binding->nativeOwner,previous.submission,error))return false;
                cellMemory=cellMemory&&charge(1,previous.bounds.bytes)
                    &&charge(1,previous.identity.capacity())&&charge(1,previous.submission.frameIdentity.capacity())
                    &&charge(1,boundary->binding->bounds.bytes)
                    &&charge(1,boundary->binding->sourceCells->bounds.bytes)
                    &&charge(1,boundary->binding->sourceCells->incidentBinding->bounds.bytes)
                    &&charge(boundary->records.capacity(),sizeof(gc::NativeGratingCellTransportRecord));
                for(const auto& record:boundary->records)for(const auto& path:record.paths)
                    cellMemory=cellMemory&&charge(path.capacity(),sizeof(gc::NativeGratingCellTraceStep));
                if(boundary->binding->nativeOwner!=nativeFrame) {
                    const auto previousFrame=std::dynamic_pointer_cast<const OpenGlFixtureSceneFrame>(boundary->binding->nativeOwner);
                    cellMemory=cellMemory&&previousFrame&&charge(1,previousFrame->causticDiagnostics.admittedFrameBytes)
                        &&previousFrame->reconcileNativeAccounting(error)&&charge(1,previousFrame->nativeAttemptBytes)
                        &&charge(1,previousFrame->gratingCameraSpectrumBytes)&&charge(1,previousFrame->gratingCellSourceBytes)
                        &&charge(1,previousFrame->gratingCoverageBytes)&&charge(1,previousFrame->queryRetainedBytes)
                        &&charge(1,previousFrame->directGratingBytes)&&charge(1,previousFrame->cameraGlassTraceBytes);
                }
            }
            if(previousCoveragePayload) {
                cellMemory=cellMemory&&charge(previousCoveragePayload->records.capacity(),sizeof(gc::GratingCellCoverageRecord));
                for(const auto& record:previousCoveragePayload->records) {
                    cellMemory=cellMemory&&charge(record.leaves.capacity(),sizeof(gc::GratingCellCoverageLeaf));
                    for(const auto& leaf:record.leaves) {
                        cellMemory=cellMemory&&charge(leaf.sweeps.capacity(),sizeof(gc::GratingCellSweepEvidence));
                        for(const auto& sweep:leaf.sweeps)
                            cellMemory=cellMemory&&charge(sweep.blockers.capacity(),sizeof(gc::GratingCellBlockerSeparation));
                    }
                }
            }
        }
        // Reserve codec writer growth, returned payloads and hexadecimal
        // identity copies, plus the re-admitted Scene's dynamic texture copy.
        const auto queryProgramCount=1u+(cameraPass?1u:0u)+(cellPass?1u:0u)+(coveragePass?1u:0u);
        if(!resources->queryContext||!resources->queryContext->active||!materialMemory||!cellMemory
            ||!charge(1,resources->queryContext->deferredBytes.load(std::memory_order_relaxed))
            ||!charge(queryProgramCount+2,kQueryProgramSourceReservation)
            ||!charge(queryProgramCount*2,binding->identity.capacity()+sizeof(OpenGlQueryArithmeticEvidence)+256)
            ||!charge(1,frame->queryRetainedBytes)
            ||!charge(1,resources->causticQueryProgram?resources->causticQueryProgram->retainedBytes:0)
            ||!charge(1,frame->gratingCameraSpectrumBytes)||!charge(1,frame->gratingCellSourceBytes)||!charge(1,frame->gratingCoverageBytes)
            ||!charge(1,frame->directGratingBytes)
            ||!charge(1,frame->nativeAttemptBytes)
            ||!charge(1,frame->cameraGlassTraceBytes)
            ||!charge(1,sizeof(HarmonicMIDI::grid::Visual3DScene))
            ||!charge(8,sceneaov::detail::kMaximumSceneBytes)
            ||!charge(resources->snapshot->textureTexels.size(),sizeof(HarmonicMIDI::grid::SceneTexelRgba8))
            ||!charge(resources->smoothGlass.query.texels.size(),2*sizeof(gc::Texel))
            ||!charge(resources->smoothGlass.query.meshes.size(),2*sizeof(opaquevisibility::Mesh))
            ||!charge(2,binding->bounds.bytes)||!charge(2,bytes[0])
            ||(directional&&(!charge(2,bytes[8])||!charge(3,sizeof(gc::NativeDirectionalSourceReceipt))
                ||!charge(3,binding->identity.size())
                ||(output.directionalSource&&(!charge(output.directionalSource->potential.capacity(),sizeof(float))
                    ||!charge(1,output.directionalSource->submission.frameIdentity.capacity())))
                ||(terminal&&terminalOutput->directionalSource
                    &&(!charge(terminalOutput->directionalSource->potential.capacity(),sizeof(float))
                        ||!charge(1,terminalOutput->directionalSource->submission.frameIdentity.capacity())))))
            ||!charge(scheduled.work.size(),(16+2)*64+2048+64)
            ||(cameraPass&&(!charge(1,cameraBounds.bytes)
                ||!charge(binding->bounds.maximumOrders,gc::opengl::kGratingCameraPrimitiveStride*sizeof(gc::Texel)-gc::kGratingCameraOrderNativeBytes)
                ||!charge(binding->bounds.maximumOrders,2048) // Native query/angular temporary storage.
                ||!charge(3,bytes[10])||!charge(6,bytes[11])||!charge(pixels,8+sizeof(gc::Texel))
                ||!charge(1,sizeof(OpenGlGratingCameraSpectrum))
                ||!charge(1,sizeof(gc::NativeGratingCameraReadback))||!charge(1,binding->identity.size())
                ||(terminalOutput->gratingCamera&&(!charge(terminalOutput->gratingCamera->orders.capacity(),sizeof(gc::NativeGratingCameraReadback::OrderReceipt))
                    ||!charge(1,terminalOutput->gratingCamera->submission.frameIdentity.capacity())))
                ||!charge(2,kQueryProgramSourceReservation)))
            ||!charge(output.incidents.capacity(),sizeof(gc::NativeGratingIncident))||!charge(output.submission.frameIdentity.capacity(),1)
            ||!charge(output.paths.capacity(),sizeof(gc::NativeGratingReadback::WorkReceipt))
            ||!charge(output.children.capacity(),sizeof(gc::NativeGratingReadback::ChildReceipt))
            || (terminal&&(!charge(1,scheduled.bounds.transientBytes)
                ||!charge(1,bytes[6])
                ||!charge(terminalOutput->paths.capacity(),sizeof(gc::NativePathReceipt))
                ||!charge(terminalOutput->deposits.capacity(),sizeof(gc::NativeDeposit))
                ||!charge(terminalOutput->spectralRadiance.capacity(),sizeof(float))
                ||!charge(terminalOutput->deposition.keys.capacity(),sizeof(gc::DepositionKey))
                ||!charge(terminalOutput->frameIdentity.capacity()+terminalOutput->deposition.submission.frameIdentity.capacity(),1)
                ||(terminalOutput->gratingIncidents&&(
                    !charge(terminalOutput->gratingIncidents->incidents.capacity(),sizeof(gc::NativeGratingIncident))
                    ||!charge(terminalOutput->gratingIncidents->paths.capacity(),sizeof(gc::NativeGratingReadback::WorkReceipt))
                    ||!charge(terminalOutput->gratingIncidents->children.capacity(),sizeof(gc::NativeGratingReadback::ChildReceipt))
                    ||!charge(terminalOutput->gratingIncidents->submission.frameIdentity.capacity(),1)
                    ||(terminalOutput->gratingIncidents->directionalSource
                        &&(!charge(terminalOutput->gratingIncidents->directionalSource->potential.capacity(),sizeof(float))
                            ||!charge(1,terminalOutput->gratingIncidents->directionalSource->submission.frameIdentity.capacity())))))))
            ||total>renderpassoutput::kMaximumTotalBytes)
        {error="OpenGL grating incident storage exceeds the existing whole-frame memory cap";return false;}
        const auto attemptIndex=frame->nativeAttemptCount;
        if(!frame->reserveNativeAttempt(OpenGlNativeAttemptKind::Adjunct,binding,{},0,
            (cameraPass?1u:0u)|(cellPass?2u:0u)|(coveragePass?4u:0u)|(terminal?8u:0u),maximumLeaves,total,error))return false;
        // Generated strings, link copies and retained source identities were
        // reserved above, including all supplemental query roles.
        const std::string materialSource=kFixtureFragmentShaderTail;
        const auto constantsEnd=materialSource.find("vec3 srgbToLinear");
        const auto mathBegin=materialSource.find("float interfaceReflectance");
        const auto mathEnd=materialSource.find("void resolveGrooveGeometry",mathBegin);
        const auto bsdfBegin=materialSource.find("float diffractionTriangleBsdf(");
        const auto orderBegin=materialSource.find("vec2 centre = -incident.xy",bsdfBegin);
        const auto orderEnd=materialSource.find("vec2 delta = outgoing.xy-centre;",orderBegin);
        if(constantsEnd==std::string::npos||mathBegin==std::string::npos||mathEnd==std::string::npos
            ||bsdfBegin==std::string::npos||orderBegin==std::string::npos||orderEnd==std::string::npos)
        {error="OpenGL released diffraction material functions are incomplete";return false;}
        auto source=(terminal?gc::opengl::gratingTerminalComputeSource:gc::opengl::gratingIncidentComputeSource)(materialSource.substr(0,constantsEnd)
            +materialSource.substr(mathBegin,mathEnd-mathBegin),
            "vec4 geometry=uDiffractionGeometry,secondaryGeometry=vec4(0.0);bool crossed=false;int secondary=0;\n"
            "int profile=int(uDiffractionControl.y+0.5);\n"+materialSource.substr(orderBegin,orderEnd-orderBegin));
        if(directional)source=gc::opengl::directionalIrradianceComputeSource(std::move(source));
        if(source.empty()||source.capacity()>kQueryProgramSourceReservation/8
            ||materialSource.capacity()>kQueryProgramSourceReservation/8)
        {error="OpenGL query source construction exceeds its fixed admission";return false;}
        std::shared_ptr<const gc::GratingIncidentBinding> canonical;
        if(!gc::prepareGratingIncidentBinding(binding->prepared,scheduled,binding->gratingOwner,binding->material,
                binding->evaluationTime,canonical,error,
                directional?gc::SourceMeasureUse::DirectionalIrradiancePreparation:gc::SourceMeasureUse::LegacyNative))return false;
        if(canonical->identity!=binding->identity||canonical->witnessTexels!=binding->witnessTexels
            ||canonical->bounds.maximumIncidents!=binding->bounds.maximumIncidents
            ||canonical->bounds.maximumOrders!=binding->bounds.maximumOrders
            ||canonical->bounds.queryWork!=binding->bounds.queryWork||canonical->bounds.orderWork!=binding->bounds.orderWork
            ||canonical->bounds.combinedWork!=binding->bounds.combinedWork||canonical->bounds.bytes!=binding->bounds.bytes)
        {error="OpenGL grating incident binding differs from its frozen authored authority";return false;}
        canonical.reset();
        // prepareScene and backend preparation snapshot independent owners.
        // Bind their immutable values to the retained native staging authority.
        if(sceneaov::detail::encodeScene(*binding->prepared.glass.sourceScene)!=sceneaov::detail::encodeScene(*resources->snapshot))
        {error="OpenGL grating incident scene differs from its retained staging snapshot";return false;}
        const auto sameProgram=[](const NativeFixtureSurfaceMaterialProgram& a,const NativeFixtureSurfaceMaterialProgram& b) {
            if(bool(a.importedBaseColorTexture)!=bool(b.importedBaseColorTexture)||bool(a.spectralReflectance)!=bool(b.spectralReflectance))return false;
            if(a.importedBaseColorTexture) {
                const auto& x=*a.importedBaseColorTexture;const auto& y=*b.importedBaseColorTexture;
                if(std::tie(x.width,x.height,x.texelCount)!=std::tie(y.width,y.height,y.texelCount)||x.texels.size()!=y.texels.size()
                    ||(!x.texels.empty()&&std::memcmp(x.texels.data(),y.texels.data(),x.texels.size()*sizeof(x.texels[0]))!=0))return false;
            }
            if(a.spectralReflectance&&(a.spectralReflectance->version!=b.spectralReflectance->version
                ||a.spectralReflectance->coefficients!=b.spectralReflectance->coefficients))return false;
            return std::tie(a.layoutVersion,a.backend,a.kind,a.object,a.authoredScene,a.authoredMaterial,
                    a.sceneRevision,a.structuralRevision,a.evaluationRevision,a.materialRevision,a.bindingDigest,a.programIdentity,
                    a.baseColorSource,a.frameEndpoint,a.vertexProgram,a.timeMixEndColor)
                ==std::tie(b.layoutVersion,b.backend,b.kind,b.object,b.authoredScene,b.authoredMaterial,
                    b.sceneRevision,b.structuralRevision,b.evaluationRevision,b.materialRevision,b.bindingDigest,b.programIdentity,
                    b.baseColorSource,b.frameEndpoint,b.vertexProgram,b.timeMixEndColor)
                &&std::memcmp(&a.parameters,&b.parameters,sizeof(a.parameters))==0;
        };
        const auto& requested=*binding->prepared.glass.sourceMaterial;
        if(!sameProgram(requested,*resources->materialProgram)||requested.objectPrograms.size()!=resources->materialProgram->objectPrograms.size())
        {error="OpenGL grating incident root material differs from its retained staging snapshot";return false;}
        for(std::size_t o=0;o<resources->snapshot->objectCount;++o) {
            const auto object=resources->snapshot->objects[o].id;
            const auto* expected=nativeSurfaceForObject(resources->materialProgram.get(),object);
            const auto* received=nativeSurfaceForObject(&requested,object);
            if(bool(expected)!=bool(received)||(expected&&!sameProgram(*expected,*received)))
            {error="OpenGL grating incident object material differs from its retained staging snapshot";return false;}
        }
        gc::Prepared retained;
        if(!gc::prepare(resources->smoothGlass,binding->prepared.sceneLight,*binding->prepared.branchPolicy,retained,error)
            ||retained.identity!=binding->prepared.identity
            ||!(directional?gc::validateDirectionalSourceUpload(retained,scheduled,*frame,frame->upload().submission,error,
                privateStaging,binding.get()):gc::validateTables(retained,scheduled,error,privateStaging)))
        {if(error.empty())error="OpenGL grating incident schedule differs from its retained staging authority";return false;}
        gc::EnergyLedger verified;
        if(!privateStaging&&!gc::validateNativeReadback(retained,scheduled,*frame->causticReadback,verified,error,
                &frame->causticReadback->deposition.submission))return false;
        const auto copyBuffer=reinterpret_cast<PFNGLCOPYBUFFERSUBDATAPROC>(glfwGetProcAddress("glCopyBufferSubData"));
        if(directional&&(!privateStaging||!terminal||!copyBuffer||!frame->directionalSourceBuffer))
        {error="OpenGL directional source requires its real retained native input and private terminal owner";return false;}
        GLint64 maximumBlock=0;integer64(GL_MAX_SHADER_STORAGE_BLOCK_SIZE,&maximumBlock);
        int bindings=0,blocks=0,groups=0,threads=0,local=0;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,&blocks);
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&threads);
        gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&groups);gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0,&local);
        if(maximumBlock<=0||bindings<(directional?8:terminal?7:6)||blocks<(directional?8:terminal?7:6)||groups<=0||threads<32||local<32
            ||scheduled.work.size()>std::uint64_t(groups)*32
            ||(cameraPass&&(!gc::admitDepositionDevice(cameraDeposition,{std::uint64_t(maximumBlock),std::uint64_t(groups),32,4},error)
                ||binding->bounds.maximumOrders>std::uint64_t(groups)*32))
            ||std::any_of(bytes.begin(),bytes.end(),[&](auto size){return size>std::uint64_t(maximumBlock);})
            ||(terminal&&(deposition.paddedKeyCount>std::uint64_t(groups)*32||!glIsTexture(frame->causticTextures[1])
                ||!glIsTexture(frame->causticTextures[2])))
            ||resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max())
        {error="OpenGL grating incident native allocation, dispatch or serial bound is exceeded";return false;}
        OpenGlFixtureState previous(gl);
        std::array<int,8> savedBindings{};std::array<GLint64,8> starts{},sizes{};const unsigned slotCount=directional?8:terminal?7:6;
        for(unsigned i=0;i<slotCount;++i) {
            gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,i,&savedBindings[i]);
            indexed64(GL_SHADER_STORAGE_BUFFER_START,i,&starts[i]);indexed64(GL_SHADER_STORAGE_BUFFER_SIZE,i,&sizes[i]);
        }
        int savedStorage=0,savedCopyRead=0,savedCopyWrite=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&savedStorage);
        if(directional) {glGetIntegerv(GL_COPY_READ_BUFFER_BINDING,&savedCopyRead);glGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING,&savedCopyWrite);}
        std::array<unsigned,13> buffers{};std::array<unsigned,3> depositionPrograms{};std::array<unsigned,5> cameraPrograms{};
        std::array<unsigned,3> cellBuffers{};
        std::array<std::shared_ptr<OpenGlQueryProgram>,4> queryOwners{};
        std::array<std::shared_ptr<const OpenGlQueryArithmeticEvidence>,4> queryEvidence{};
        unsigned program=0,cameraFbo=0,cameraVao=0,cameraSpectrumTexture=0,cellProgram=0,coverageProgram=0,coverageBuffer=0;bool cleaned=false;
        const auto cleanup=[&] {
            if(cleaned)return;
            cleaned=true;
            if(program&&!queryOwners[0])gl.DeleteProgram(program);
            for(auto owned:depositionPrograms)if(owned)gl.DeleteProgram(owned);
            for(std::size_t index=0;index<cameraPrograms.size();++index)
                if(cameraPrograms[index]&&(index!=0||!queryOwners[1]))gl.DeleteProgram(cameraPrograms[index]);
            if(cameraFbo)gl.DeleteFramebuffers(1,&cameraFbo);
            if(cameraVao)gl.DeleteVertexArrays(1,&cameraVao);
            if(cameraSpectrumTexture)glDeleteTextures(1,&cameraSpectrumTexture);
            if(cellProgram&&!queryOwners[2])gl.DeleteProgram(cellProgram);
            if(coverageProgram&&!queryOwners[3]){gl.DeleteProgram(coverageProgram);}if(coverageBuffer)gl.DeleteBuffers(1,&coverageBuffer);
            gl.DeleteBuffers(int(cellBuffers.size()),cellBuffers.data());
            gl.DeleteBuffers(int(buffers.size()),buffers.data());
            for(unsigned i=0;i<slotCount;++i)
                if(savedBindings[i]&&sizes[i]>0)bindRange(GL_SHADER_STORAGE_BUFFER,i,unsigned(savedBindings[i]),starts[i],sizes[i]);
                else gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,i,unsigned(savedBindings[i]));
            if(directional) {gl.BindBuffer(GL_COPY_READ_BUFFER,unsigned(savedCopyRead));gl.BindBuffer(GL_COPY_WRITE_BUFFER,unsigned(savedCopyWrite));}
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(savedStorage));previous.restore();
        };
        int cleanupToken=0;auto release=[&](int*){cleanup();};
        const std::unique_ptr<int,decltype(release)> cleanupGuard(&cleanupToken,release);
        const auto fail=[&](const char* message){error=message;cleanup();return false;};
        if(cameraPass&&!fixtureCameraAttachmentsMatch(*frame,error)) {cleanup();return false;}
        queryOwners[0]=linkQueryArithmeticProgram(gl,resources->queryContext,resources->rendererGeneration,{{GL_COMPUTE_SHADER,source}},error);
        program=queryOwners[0]?queryOwners[0]->program:0;
        if(!program){cleanup();return false;}
        gl.GenBuffers(cameraPass?13:directional?9:8,buffers.data());
        std::uint64_t nativeBytes=0;
        const unsigned slots[]{0,1,2,5,6,5,2,6,7};
        const std::vector<float> zeroSpectrum(terminal?std::size_t(pixels*4):0,0.0f);
        for(unsigned i=0;i<(cameraPass?13u:directional?9u:terminal?8u:4u);++i) {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[i]);
            gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(bytes[i]),i==3?static_cast<const void*>(binding->witnessTexels.data()):(i==6||i==7||i==11||i==12)?static_cast<const void*>(zeroSpectrum.data()):nullptr,i==3?GL_STATIC_DRAW:GL_DYNAMIC_COPY);
            GLint64 actual=0;buffer64(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(!buffers[i]||actual<0||std::uint64_t(actual)!=bytes[i])return fail("OpenGL grating incident native storage differs from admission");
            nativeBytes+=std::uint64_t(actual);if(i<5||i==8)gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,slots[i],buffers[i]);
        }
        if(directional) {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->directionalSourceBuffer);GLint64 actual=0;
            buffer64(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            if(actual!=sizeof(gc::NativeDirectionalSourceInput))return fail("OpenGL retained directional source allocation changed");
            const auto* input=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,sizeof(gc::NativeDirectionalSourceInput),GL_MAP_READ_BIT);
            if(!input)return fail("OpenGL retained directional source mapping failed");
            const bool exact=std::memcmp(input,&frame->upload().input,sizeof(gc::NativeDirectionalSourceInput))==0;
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!exact)return fail("OpenGL retained directional source native input changed");
            gl.BindBuffer(GL_COPY_READ_BUFFER,frame->directionalSourceBuffer);gl.BindBuffer(GL_COPY_WRITE_BUFFER,buffers[8]);
            copyBuffer(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,0,0,sizeof(gc::NativeDirectionalSourceInput));
        }
        for(unsigned slot:{3u,4u}) {
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,frame->causticBuffers[slot]);GLint64 actual=0;buffer64(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
            const auto expected=(slot==3?scheduled.workTexels.size():scheduled.triangleTexels.size())*sizeof(gc::Texel);
            if(actual<0||std::uint64_t(actual)!=expected)return fail("OpenGL grating incident retained query tables differ from the staging frame");
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,slot,frame->causticBuffers[slot]);
        }
        if(terminal) {
            depositionPrograms[0]=linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::depositionKeySource}},error);
            depositionPrograms[1]=linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::depositionSortSource}},error);
            depositionPrograms[2]=linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::depositionReduceSource()}},error);
            if(std::any_of(depositionPrograms.begin(),depositionPrograms.end(),[](auto value){return !value;})) {cleanup();return false;}
        }
        if(glGetError()!=GL_NO_ERROR)return fail("OpenGL grating incident native allocation failed");
        const auto serial=++resources->causticSubmissionSerial;
        gc::NativeGratingReadback candidate;
        candidate.submission={binding->identity,resources->rendererGeneration,serial,serial,serial,serial,serial};
        if(directional) {
            candidate.submission.rasterGeneration=frame->upload().submission.rasterGeneration;
            candidate.submission.queryGeneration=frame->upload().submission.queryGeneration;
        }

        gl.UseProgram(program);const auto loc=[&](const char* name){return gl.GetUniformLocation(program,name);};
        const auto& policy=binding->prepared.branchPolicy->description();
        gl.Uniform1i(loc("uWorkCount"),int(scheduled.work.size()));gl.Uniform1i(loc("uTriangleCount"),int(scheduled.triangles.size()));
        if(terminal) {
            gl.Uniform1i(loc("uGratingWorkBase"),int(workBase));gl.Uniform1i(loc("uGratingChildBase"),int(childBase));
            const auto camera=gc::cameraPosition(scheduled);const auto& rotation=scheduled.camera.transform.rotation;
            gl.Uniform4f(loc("uCamera"),camera[0],camera[1],camera[2],0);
            gl.Uniform4f(loc("uRotation"),rotation.x,rotation.y,rotation.z,rotation.w);
            gl.Uniform4f(loc("uProjection"),std::tan(scheduled.camera.verticalFovRadians*.5f),float(scheduled.width)/scheduled.height,
                scheduled.camera.nearPlane,scheduled.camera.farPlane);
            gl.Uniform2f(loc("uExtent"),float(scheduled.width),float(scheduled.height));
            for(unsigned i=1;i<3;++i) {
                gl.ActiveTexture(GL_TEXTURE0+i);gl.BindSampler(i,0);glBindTexture(GL_TEXTURE_2D,frame->causticTextures[i]);
                gl.Uniform1i(loc(i==1?"uRasterOwners":"uRasterTriangle"),int(i));
            }
        }
        gl.Uniform1i(loc("uWitnessCount"),int(binding->triangles.size()));gl.Uniform1i(loc("uVisits"),int(policy.maximumVisits));
        gl.Uniform1i(loc("uInterfaces"),int(policy.maximumInterfaces));gl.Uniform1i(loc("uPending"),int(policy.maximumPending));
        gl.Uniform2f(loc("uPolicy"),policy.maximumUnresolvedFraction,policy.minimumBranchFraction);
        gl.Uniform4fv(loc("uSource"),1,scheduled.sourceDirectionTrace.data());gl.Uniform4fv(loc("uOpaqueControl"),1,scheduled.query.control.data());
        gl.Uniform4fv(loc("uIor[0]"),2,scheduled.ior.data());gl.Uniform4fv(loc("uLi[0]"),2,scheduled.sourceRadiance[0].data());
        const auto& material=binding->gratingOwner->diffractionPaths[0].material;
        gl.Uniform4fv(loc("uDiffractionGeometry"),1,&material.geometry.x);
        gl.Uniform4fv(loc("uDiffractionMicrostructure"),1,&material.microstructure.x);
        gl.Uniform4fv(loc("uDiffractionRoughness"),1,&material.roughness.x);
        gl.Uniform4fv(loc("uDiffractionControl"),1,&material.control.x);gl.Uniform4fv(loc("uDiffractionCoating"),1,&material.coating.x);
        gl.Uniform4fv(loc("uDiffractionSpectral[0]"),8,&material.spectral[0].x);
        gl.ActiveTexture(GL_TEXTURE0);gl.BindSampler(0,0);glBindTexture(GL_TEXTURE_2D,frame->causticTextures[0]);gl.Uniform1i(loc("uOpaqueVisibility"),0);
        gl.DispatchCompute(unsigned((scheduled.work.size()+31)/32),1,1);gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        if(directional)gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        if(!causticFence(gl,error)){cleanup();return false;}
        if(stopForFixture==OpenGlGratingAttemptStopForFixture::PointFence)
            {cleanup();error="OpenGL fixture refusal after real point dispatch and completed fence";return false;}
        gc::NativeReadback full;
        if(directional) {
            auto receipt=std::make_shared<gc::NativeDirectionalSourceReceipt>();
            receipt->version=gc::kDirectionalSourceReceiptVersion;receipt->submission=candidate.submission;
            receipt->bindingGeneration=frame->upload().bindingGeneration;receipt->potential.resize(scheduled.work.size());
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[8]);
            const auto* native=static_cast<const std::uint8_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,
                sizeof(gc::NativeDirectionalSourceInput),std::ptrdiff_t(bytes[8]-sizeof(gc::NativeDirectionalSourceInput)),GL_MAP_READ_BIT));
            if(!native)return fail("OpenGL directional source shader echo mapping failed");
            std::memcpy(&receipt->echo,native,sizeof(receipt->echo));
            std::memcpy(receipt->potential.data(),native+sizeof(receipt->echo),receipt->potential.size()*sizeof(float));
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||glGetError()!=GL_NO_ERROR)
                return fail("OpenGL directional source shader echo readback failed");
            receipt->complete=true;
            if(!gc::validateDirectionalSourceReadback(retained,scheduled,*frame,candidate.submission,*receipt,error,true,binding.get()))
                {cleanup();return false;}
            candidate.directionalSource=receipt;full.directionalSource=std::move(receipt);
        }
        if(terminal) {
            full.frameIdentity=binding->identity;full.paths.resize(scheduled.work.size());
            candidate.paths.resize(scheduled.work.size());
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[1]);
            const auto* packed=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[1]),GL_MAP_READ_BIT));
            if(!packed)return fail("OpenGL grating dense work mapping failed");
            bool valid=true;
            for(std::size_t w=0;w<scheduled.work.size();++w)
                valid=valid&&gc::opengl::decodeGratingWork(packed+workBase+w*2,candidate.paths[w]);
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!valid)return fail("OpenGL grating dense work receipt is malformed");
        }
        std::vector<std::uint32_t> counts(scheduled.work.size());std::vector<double> tails(scheduled.work.size());
        std::uint64_t totalQuery=0,baseQuery=0;bool decoded=true;
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[0]);
        const auto* paths=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[0]),GL_MAP_READ_BIT));
        if(!paths)return fail("OpenGL grating incident path mapping failed");
        for(std::size_t w=0;w<scheduled.work.size()&&decoded;++w) {
            const auto* p=paths+w*gc::kPathReceiptStride;std::uint32_t word=0,visits=0,work=0,queries=0,extra=0,orders=0;
            for(unsigned t=0;t<gc::kPathReceiptStride;++t)for(float value:p[t])if(!std::isfinite(value))decoded=false;
            decoded=decoded&&gc::opengl::causticId(p[0][0],p[0][1],work)&&work==w
                &&gc::opengl::causticWord(p[0][2],policy.maximumVisits,counts[w])
                &&gc::opengl::causticWord(p[0][3],policy.maximumVisits,visits)&&p[1][0]==1&&(p[1][1]==0||p[1][1]==1)
                &&gc::opengl::causticId(p[2][0],p[2][1],queries)&&p[2][2]==0&&p[2][3]==0
                &&gc::opengl::causticId(p[1][2],p[1][3],extra)&&queries>=extra
                &&gc::opengl::causticWord(p[4][1],policy.maximumVisits*3,orders)&&(terminal||orders==counts[w]*3)
                &&p[4][0]>=0&&p[4][2]==0&&p[4][3]==0&&p[15]==gc::Texel{}
                &&gc::opengl::causticWord(p[1][1],1,word);
            if(!decoded)break;
            if(terminal) {
                auto legacy=std::array<gc::Texel,gc::kPathReceiptStride>{};std::copy_n(p,legacy.size(),legacy.begin());
                legacy[1][2]=legacy[1][3]=legacy[4][1]=0;
                if(!gc::opengl::causticDecodePath(legacy.data(),scheduled,w,full.paths[w])
                    ||candidate.paths[w].queryWork!=extra||candidate.paths[w].orderWork!=orders) {decoded=false;break;}
                counts[w]=candidate.paths[w].incidentCount;
                if(orders!=counts[w]*3) {decoded=false;break;}
            }
            tails[w]=p[4][0];totalQuery+=queries;baseQuery+=queries-extra;candidate.queryWork+=extra;candidate.orderWork+=orders;
        }
        if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!decoded)return fail("OpenGL grating incident path is incomplete or malformed");
        if(baseQuery>scheduled.bounds.queryWork||totalQuery>binding->bounds.queryWork
            ||totalQuery+candidate.orderWork>binding->bounds.combinedWork)return fail("OpenGL grating incident measured work exceeds shared admission");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[1]);
        const auto* events=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[1]),GL_MAP_READ_BIT));
        if(!events)return fail("OpenGL grating incident completed mapping failed");
        candidate.incidents.reserve(std::size_t(binding->bounds.maximumIncidents));
        for(std::size_t w=0;w<counts.size()&&decoded;++w)for(std::size_t i=0;i<policy.maximumVisits;++i) {
            const auto* p=events+(w*policy.maximumVisits+i)*gc::kGratingIncidentStride;
            if(i>=counts[w]) {for(unsigned t=0;t<gc::kGratingIncidentStride;++t)if(p[t]!=gc::Texel{})decoded=false;continue;}
            gc::NativeGratingIncident incident;
            if(!gc::opengl::decodeGratingIncident(p,incident)||incident.work!=w){decoded=false;break;}
            if(!terminal){tails[w]+=incident.unresolvedPower;}candidate.incidents.push_back(incident);
        }
        if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!decoded)return fail("OpenGL grating incident witness is malformed");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[2]);
        const auto* orders=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[2]),GL_MAP_READ_BIT));
        if(!orders)return fail("OpenGL grating order completed mapping failed");
        // GPU segments use a sparse serialization locator. Compact it only after
        // exact parent work and signed order proof; never replace branch identity.
        std::vector<std::uint32_t> locators(std::size_t(binding->bounds.maximumIncidents),UINT32_MAX);
        std::size_t incidentIndex=0;
        for(std::size_t w=0;w<counts.size()&&decoded;++w)for(std::size_t i=0;i<policy.maximumVisits;++i) {
            auto* incident=i<counts[w]?&candidate.incidents[incidentIndex++]:nullptr;
            const auto locator=w*policy.maximumVisits+i;
            if(incident)locators[locator]=std::uint32_t(incidentIndex-1);
            for(unsigned o=0;o<gc::kGratingOrdersPerIncident;++o) {
                const auto* p=orders+(locator*3+o)*gc::kGratingOrderStride;
                if(!incident||o>=incident->orderCount) {for(unsigned t=0;t<gc::kGratingOrderStride;++t)if(p[t]!=gc::Texel{})decoded=false;}
                else if(!gc::opengl::decodeGratingOrder(p,std::uint32_t(locator),incident->orders[o]))decoded=false;
                if(terminal) {
                    const auto* childWords=orders+childBase+(locator*3+o)*8;
                    if(!incident||o>=incident->orderCount) {
                        for(unsigned t=0;t<8;++t)if(childWords[t]!=gc::Texel{})decoded=false;
                    } else {
                        gc::NativeGratingReadback::ChildReceipt child;
                        if(!gc::opengl::decodeGratingChild(childWords,child)||child.incident!=locator
                            ||child.order!=incident->orders[o].order)decoded=false;
                        else {child.incident=locators[locator];candidate.children.push_back(child);}
                    }
                }
            }
        }
        if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!decoded)return fail("OpenGL signed grating order receipt is malformed");
        for(std::size_t w=0;w<tails.size();++w)
            if(tails[w]>scheduled.work[w].incidentPower*policy.maximumUnresolvedFraction*(1.0+2e-4))
                return fail("OpenGL combined glass and grating omitted flux exceeds the original unresolved fraction");
        candidate.complete=true;candidate.childrenComplete=terminal;
        if(!gc::validateGratingIncidentReadback(*binding,candidate,candidate.submission,error,directional?frame.get():nullptr)){cleanup();return false;}
        queryEvidence[0]=completedQueryArithmeticEvidence(queryOwners[0],OpenGlQueryArithmeticRole::GratingPoint,candidate.submission);
        if(terminal) {
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,buffers[4]);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,5,buffers[5]);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,buffers[6]);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffers[7]);
            gl.UseProgram(depositionPrograms[0]);
            const auto uniform=[&](unsigned index,const char* name){return gl.GetUniformLocation(depositionPrograms[index],name);};
            gl.Uniform1i(uniform(0,"uWorkCount"),int(scheduled.work.size()));gl.Uniform1i(uniform(0,"uVisits"),int(policy.maximumVisits));
            gl.Uniform1i(uniform(0,"uKeyCount"),int(deposition.paddedKeyCount));gl.Uniform2f(uniform(0,"uExtent"),scheduled.width,scheduled.height);
            gl.DispatchCompute(unsigned((deposition.paddedKeyCount+31)/32),1,1);gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            gl.UseProgram(depositionPrograms[1]);gl.Uniform1i(uniform(1,"uKeyCount"),int(deposition.paddedKeyCount));
            for(std::uint64_t span=2;span<=deposition.paddedKeyCount;span*=2)for(auto distance=span/2;distance;distance/=2) {
                gl.Uniform1i(uniform(1,"uSortSpan"),int(span));gl.Uniform1i(uniform(1,"uSortDistance"),int(distance));
                gl.DispatchCompute(unsigned((deposition.paddedKeyCount/2+31)/32),1,1);gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
            gl.UseProgram(depositionPrograms[2]);gl.Uniform1i(uniform(2,"uKeyCount"),int(deposition.paddedKeyCount));
            gl.DispatchCompute(unsigned((deposition.paddedKeyCount+31)/32),1,1);gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
            if(!causticFence(gl,error)) {cleanup();return false;}
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[4]);
            const auto* deposits=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[4]),GL_MAP_READ_BIT));
            if(!deposits)return fail("OpenGL grating completed receiver deposit mapping failed");
            full.deposits.reserve(std::size_t(scheduled.bounds.maximumDeposits));
            for(std::size_t w=0;w<full.paths.size()&&decoded;++w)for(std::size_t d=0;d<policy.maximumVisits;++d) {
                const auto* packed=deposits+(w*policy.maximumVisits+d)*gc::kDepositReceiptStride;
                if(d>=full.paths[w].depositCount) {
                    for(unsigned t=0;t<gc::kDepositReceiptStride;++t)if(packed[t]!=gc::Texel{})decoded=false;
                    continue;
                }
                gc::NativeDeposit deposit;
                if(!gc::opengl::decodeGratingDeposit(packed,deposit)||deposit.work!=w) {decoded=false;break;}
                if(deposit.gratingVersion) {
                    if(deposit.gratingIncident>=locators.size()||locators[deposit.gratingIncident]==UINT32_MAX) {decoded=false;break;}
                    deposit.gratingIncident=locators[deposit.gratingIncident];
                }
                full.deposits.push_back(deposit);
            }
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!decoded)return fail("OpenGL grating completed receiver deposit is malformed");
            full.spectralRadiance.resize(std::size_t(pixels*8));
            for(unsigned half=0;half<2;++half) {
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[6+half]);
                const auto* spectrum=static_cast<const float*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[6+half]),GL_MAP_READ_BIT));
                if(!spectrum)return fail("OpenGL grating completed spectrum mapping failed");
                for(std::size_t pixel=0;pixel<pixels;++pixel)
                    std::memcpy(full.spectralRadiance.data()+pixel*8+half*4,spectrum+pixel*4,4*sizeof(float));
                if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return fail("OpenGL grating completed spectrum readback failed");
            }
            full.deposition.keys.resize(std::size_t(deposition.paddedKeyCount));
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[5]);
            const auto* keys=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[5]),GL_MAP_READ_BIT);
            if(!keys)return fail("OpenGL grating completed stable keys mapping failed");
            std::memcpy(full.deposition.keys.data(),keys,std::size_t(bytes[5]));
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||glGetError()!=GL_NO_ERROR)return fail("OpenGL grating completed stable readback failed");
            for(const auto& key:full.deposition.keys)if(key.pixel!=UINT32_MAX)++full.deposition.visibleCount;
            full.deposition.version=gc::kDepositionVersion;full.deposition.submission=candidate.submission;full.deposition.complete=true;
            full.gratingIncidentVersion=gc::kGratingIncidentVersion;full.gratingIncidents=std::make_shared<const gc::NativeGratingReadback>(candidate);
            auto bound=scheduled;bound.identity=binding->identity;bound.gratingIncidentBinding=binding;
            gc::EnergyLedger ledger;
            if(!gc::validateNativeReadback(retained,bound,full,ledger,error,&candidate.submission,true,directional?frame.get():nullptr)) {cleanup();return false;}
            if(cameraPass) {
                // Opt-in only: terminal callers retain their original queries.
                queryOwners[1]=linkQueryArithmeticProgram(gl,resources->queryContext,resources->rendererGeneration,
                    {{GL_COMPUTE_SHADER,gc::opengl::gratingCameraComputeSource()}},error);
                cameraPrograms[0]=queryOwners[1]?queryOwners[1]->program:0;
                cameraPrograms[1]=linkCausticProgram(gl,{{GL_VERTEX_SHADER,gc::opengl::rasterVertexSource()},
                    {GL_FRAGMENT_SHADER,gc::opengl::gratingCameraGeometryFragmentSource}},error);
                cameraPrograms[2]=linkCausticProgram(gl,{{GL_VERTEX_SHADER,kDiffractionResolveVertexShader},
                    {GL_FRAGMENT_SHADER,gc::opengl::gratingReceiverResolveSource()}},error);
                cameraPrograms[3]=linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::gratingCameraKeySource}},error);
                cameraPrograms[4]=linkCausticProgram(gl,{{GL_COMPUTE_SHADER,gc::opengl::gratingCameraReduceSource}},error);
                if(std::any_of(cameraPrograms.begin(),cameraPrograms.end(),[](auto p){return !p;})) {cleanup();return false;}
                gl.GenVertexArrays(1,&cameraVao);gl.BindVertexArray(cameraVao);
                gl.GenFramebuffers(1,&cameraFbo);gl.BindFramebuffer(GL_FRAMEBUFFER,cameraFbo);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,frame->depthTexture_,0);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,frame->rawTextures[0],0);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT5,GL_TEXTURE_2D,frame->rawTextures[3],0);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT6,GL_TEXTURE_2D,frame->rawTextures[4],0);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,frame->gratingCameraDepthAttachment,0);
                const unsigned geometryDraws[]{GL_NONE,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_NONE,GL_NONE,GL_COLOR_ATTACHMENT5,GL_COLOR_ATTACHMENT6};
                gl.DrawBuffers(7,geometryDraws);
                if(!cameraVao||!cameraFbo||gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
                    return fail("OpenGL private camera geometry framebuffer is incomplete");
                glViewport(0,0,scheduled.width,scheduled.height);glDisable(GL_BLEND);
                for(unsigned i=0;i<8;++i)gl.Disablei(GL_BLEND,i);
                glDisable(GL_SCISSOR_TEST);glDisable(GL_CULL_FACE);glDisable(GL_RASTERIZER_DISCARD);
                glDisable(GL_COLOR_LOGIC_OP);glDisable(GL_FRAMEBUFFER_SRGB);glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
                glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);glDepthMask(GL_TRUE);glDepthFunc(GL_LESS);glEnable(GL_DEPTH_TEST);
                glClearDepth(1);glClear(GL_DEPTH_BUFFER_BIT);
                const float zero[4]{};const unsigned zeroId[4]{};
                gl.ClearBufferfv(GL_COLOR,1,zero);gl.ClearBufferfv(GL_COLOR,2,zero);
                gl.ClearBufferuiv(GL_COLOR,5,zeroId);gl.ClearBufferuiv(GL_COLOR,6,zeroId);
                const auto camera=gc::cameraPosition(scheduled);const auto& rotation=scheduled.camera.transform.rotation;
                const auto configureCamera=[&](unsigned shader) {
                    const auto location=[&](const char* name){return gl.GetUniformLocation(shader,name);};
                    gl.Uniform4f(location("uCamera"),camera[0],camera[1],camera[2],0);
                    gl.Uniform4f(location("uRotation"),rotation.x,rotation.y,rotation.z,rotation.w);
                    gl.Uniform4f(location("uProjection"),std::tan(scheduled.camera.verticalFovRadians*.5f),
                        float(scheduled.width)/scheduled.height,scheduled.camera.nearPlane,scheduled.camera.farPlane);
                };
                gl.UseProgram(cameraPrograms[1]);configureCamera(cameraPrograms[1]);
                for(std::size_t t=0;t<scheduled.triangles.size();++t) {
                    gl.Uniform1i(gl.GetUniformLocation(cameraPrograms[1],"uTriangle"),int(t));glDrawArrays(GL_TRIANGLES,0,3);
                }
                gl.MemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffers[9]);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,buffers[1]);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,buffers[2]);
                gl.UseProgram(cameraPrograms[0]);configureCamera(cameraPrograms[0]);
                const auto location=[&](const char* name){return gl.GetUniformLocation(cameraPrograms[0],name);};
                gl.Uniform1i(location("uOrderCount"),int(binding->bounds.maximumOrders));
                gl.Uniform1i(location("uEntrySubdivisions"),int(scheduled.scheduling.entrySubdivisions));
                gl.Uniform2f(location("uExtent"),scheduled.width,scheduled.height);
                gl.Uniform4fv(location("uOpaqueControl"),1,scheduled.query.control.data());
                for(unsigned i=0;i<3;++i) {
                    gl.ActiveTexture(GL_TEXTURE0+i);gl.BindSampler(i,0);glBindTexture(GL_TEXTURE_2D,frame->causticTextures[i]);
                    gl.Uniform1i(location(i==0?"uOpaqueVisibility":i==1?"uRasterOwners":"uRasterTriangle"),int(i));
                }
                gl.DispatchCompute(unsigned((binding->bounds.maximumOrders+31)/32),1,1);
                gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
                // Resolve only the real receiver spectrum. Base surfaces and
                // finite grating-cell radiance have no complete Color grant.
                gl.BindFramebuffer(GL_FRAMEBUFFER,cameraFbo);
                gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,frame->accumulationTexture_,0);
                const unsigned resolveDraw=GL_COLOR_ATTACHMENT0;gl.DrawBuffers(1,&resolveDraw);
                if(gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
                    return fail("OpenGL private receiver spectrum framebuffer is incomplete");
                glDisable(GL_DEPTH_TEST);glDepthMask(GL_FALSE);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,buffers[6]);gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffers[7]);
                gl.UseProgram(cameraPrograms[2]);const auto weights=gc::resolveWeights();
                gl.Uniform2f(gl.GetUniformLocation(cameraPrograms[2],"uExtent"),scheduled.width,scheduled.height);
                gl.Uniform4fv(gl.GetUniformLocation(cameraPrograms[2],"uCie[0]"),8,weights[0].data());
                glDrawArrays(GL_TRIANGLES,0,3);gl.MemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT);
                if(glGetError()!=GL_NO_ERROR||!causticFence(gl,error)) {if(error.empty())error="OpenGL private camera draw failed";cleanup();return false;}
                queryEvidence[1]=completedQueryArithmeticEvidence(queryOwners[1],OpenGlQueryArithmeticRole::GratingCamera,candidate.submission,
                    {queryOwners[0],{}},{queryEvidence[0],{}});
                auto cameraReceipt=std::make_shared<gc::NativeGratingCameraReadback>();
                cameraReceipt->version=gc::kGratingCameraReceiptVersion;cameraReceipt->submission=candidate.submission;
                cameraReceipt->incidents=full.gratingIncidents;cameraReceipt->complete=true;
                cameraReceipt->orders.reserve(candidate.children.size());
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[9]);
                const auto* native=static_cast<const gc::Texel*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[9]),GL_MAP_READ_BIT));
                if(!native)return fail("OpenGL private camera receipt mapping failed");
                bool cameraDecoded=true;std::size_t nativeLocator=0;
                for(std::size_t i=0;i<candidate.incidents.size()&&cameraDecoded;++i) {
                    const auto& event=candidate.incidents[i];
                    while(nativeLocator<locators.size()&&locators[nativeLocator]!=i)++nativeLocator;
                    if(nativeLocator>=locators.size()) {cameraDecoded=false;break;}
                    const auto locator=nativeLocator++;
                    for(int signedOrder=-1;signedOrder<=1;++signedOrder) {
                        unsigned slot=0;while(slot<event.orderCount&&event.orders[slot].order!=signedOrder)++slot;
                        if(slot==event.orderCount)continue;
                        const auto* p=native+(locator*3+slot)*gc::opengl::kGratingCameraPrimitiveStride;
                        gc::NativeGratingCameraReadback::OrderReceipt r;std::uint32_t stored=0,band=0,order=0,outcome=0,queries=0;
                        for(unsigned t=0;t<gc::opengl::kGratingCameraPrimitiveStride;++t)for(float value:p[t])if(!std::isfinite(value))cameraDecoded=false;
                        cameraDecoded=cameraDecoded&&gc::opengl::causticWord(p[0][0],std::uint32_t(locators.size()-1),stored)&&stored==locator
                            &&gc::opengl::causticWord(p[0][1],7,band)&&band==event.band
                            &&gc::opengl::causticWord(p[0][2],2,order)&&int(order)-1==signedOrder
                            &&gc::opengl::causticWord(p[0][3],3,outcome)&&gc::opengl::causticId(p[2][0],p[2][1],queries)
                            &&(p[1][3]==0||p[1][3]==1)&&p[1][2]>=0&&p[2][2]==0&&p[2][3]==0
                            &&p[10]==gc::Texel{}&&p[11]==gc::Texel{};
                        r.incident=std::uint32_t(i);r.band=band;r.order=signedOrder;r.outcome=gc::NativeGratingCameraReadback::Outcome(outcome);
                        r.complete=p[1][3]==1;r.spectralRadiance=p[1][2];
                        if(r.outcome==gc::NativeGratingCameraReadback::Outcome::OutsideCamera) {
                            cameraDecoded=cameraDecoded&&p[1][0]==-1&&p[1][1]==-1&&queries==0;
                        } else {
                            cameraDecoded=cameraDecoded&&gc::opengl::causticWord(p[1][0],scheduled.width-1,r.x)
                                &&gc::opengl::causticWord(p[1][1],scheduled.height-1,r.y);
                            if(p[3][0]>=0)cameraDecoded=cameraDecoded&&gc::opengl::causticHit(p+3,r.cameraHit);
                            cameraDecoded=cameraDecoded&&gc::opengl::causticId(p[8][0],p[8][1],r.raster.object)
                                &&gc::opengl::causticId(p[8][2],p[8][3],r.raster.material)
                                &&gc::opengl::causticWord(p[9][0],UINT32_MAX,r.raster.instance)
                                &&gc::opengl::causticWord(p[9][1],UINT32_MAX,r.raster.triangle);
                            r.raster.linearDepth=p[9][2];
                        }
                        cameraReceipt->queryWork+=queries;cameraReceipt->complete=cameraReceipt->complete&&r.complete;
                        cameraReceipt->orders.push_back(r);
                    }
                }
                if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||!cameraDecoded||glGetError()!=GL_NO_ERROR
                    ||cameraReceipt->queryWork>cameraBounds.queryWork
                    ||totalQuery+candidate.orderWork+cameraReceipt->queryWork>cameraBounds.combinedWork)
                    return fail("OpenGL private camera receipt is malformed or exceeds native admission");
                full.gratingCameraVersion=gc::kGratingCameraReceiptVersion;full.gratingCamera=std::move(cameraReceipt);
                if(full.gratingCamera->complete&&!gc::validateNativeReadback(retained,bound,full,ledger,error,&candidate.submission,true,frame.get())) {cleanup();return false;}
                *cameraDiagnostics={full.gratingCamera->queryWork,bytes[9],true,true};
                if(full.gratingCamera->complete) {
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffers[9]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,5,buffers[10]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,buffers[11]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffers[12]);
                    gl.UseProgram(cameraPrograms[3]);
                    const auto keyLocation=[&](const char* name){return gl.GetUniformLocation(cameraPrograms[3],name);};
                    gl.Uniform1i(keyLocation("uOrderCount"),int(binding->bounds.maximumOrders));
                    gl.Uniform1i(keyLocation("uKeyCount"),int(cameraDeposition.paddedKeyCount));
                    gl.Uniform2f(keyLocation("uExtent"),scheduled.width,scheduled.height);
                    gl.DispatchCompute(unsigned((cameraDeposition.paddedKeyCount+31)/32),1,1);
                    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    // Same bitonic authority as the released receiver deposits.
                    gl.UseProgram(depositionPrograms[1]);
                    gl.Uniform1i(gl.GetUniformLocation(depositionPrograms[1],"uKeyCount"),int(cameraDeposition.paddedKeyCount));
                    for(std::uint64_t span=2;span<=cameraDeposition.paddedKeyCount;span*=2)for(auto distance=span/2;distance;distance/=2) {
                        gl.Uniform1i(gl.GetUniformLocation(depositionPrograms[1],"uSortSpan"),int(span));
                        gl.Uniform1i(gl.GetUniformLocation(depositionPrograms[1],"uSortDistance"),int(distance));
                        gl.DispatchCompute(unsigned((cameraDeposition.paddedKeyCount/2+31)/32),1,1);
                        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    }
                    gl.UseProgram(cameraPrograms[4]);
                    gl.Uniform1i(gl.GetUniformLocation(cameraPrograms[4],"uKeyCount"),int(cameraDeposition.paddedKeyCount));
                    gl.DispatchCompute(unsigned((cameraDeposition.paddedKeyCount+31)/32),1,1);
                    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
                    gl.ActiveTexture(GL_TEXTURE0);gl.BindSampler(0,0);gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
                    glGenTextures(1,&cameraSpectrumTexture);glBindTexture(GL_TEXTURE_2D,cameraSpectrumTexture);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
                    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,scheduled.width,scheduled.height,0,GL_RGBA,GL_HALF_FLOAT,nullptr);
                    gl.BindFramebuffer(GL_FRAMEBUFFER,cameraFbo);
                    gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,cameraSpectrumTexture,0);
                    const unsigned resolveDraw=GL_COLOR_ATTACHMENT0;gl.DrawBuffers(1,&resolveDraw);
                    if(!cameraSpectrumTexture||gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
                        return fail("OpenGL private grating spectrum resolve framebuffer is incomplete");
                    gl.UseProgram(cameraPrograms[2]);
                    gl.Uniform2f(gl.GetUniformLocation(cameraPrograms[2],"uExtent"),scheduled.width,scheduled.height);
                    gl.Uniform4fv(gl.GetUniformLocation(cameraPrograms[2],"uCie[0]"),8,weights[0].data());
                    glDrawArrays(GL_TRIANGLES,0,3);gl.MemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT);
                    if(glGetError()!=GL_NO_ERROR||!causticFence(gl,error)) {if(error.empty())error="OpenGL private grating spectrum resolve failed";cleanup();return false;}
                    auto spectrum=std::make_shared<OpenGlGratingCameraSpectrum>();
                    spectrum->camera=full.gratingCamera;spectrum->width=scheduled.width;spectrum->height=scheduled.height;
                    spectrum->keys.resize(std::size_t(cameraDeposition.paddedKeyCount));
                    spectrum->spectralRadiance.resize(std::size_t(pixels*8));spectrum->resolvedColor.resize(std::size_t(pixels));
                    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[10]);
                    const auto* nativeKeys=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[10]),GL_MAP_READ_BIT);
                    if(!nativeKeys)return fail("OpenGL private grating sorted keys mapping failed");
                    std::memcpy(spectrum->keys.data(),nativeKeys,std::size_t(bytes[10]));
                    if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return fail("OpenGL private grating sorted keys readback failed");
                    for(unsigned half=0;half<2;++half) {
                        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[11+half]);
                        const auto* nativeBands=static_cast<const float*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(bytes[11+half]),GL_MAP_READ_BIT));
                        if(!nativeBands)return fail("OpenGL private grating spectral buffer mapping failed");
                        for(std::size_t pixel=0;pixel<pixels;++pixel)
                            std::memcpy(spectrum->spectralRadiance.data()+pixel*8+half*4,nativeBands+pixel*4,4*sizeof(float));
                        if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return fail("OpenGL private grating spectral buffer readback failed");
                    }
                    int savedPack=0,alignment=0,rowLength=0,skipPixels=0,skipRows=0,swapBytes=0;
                    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING,&savedPack);glGetIntegerv(GL_PACK_ALIGNMENT,&alignment);
                    glGetIntegerv(GL_PACK_ROW_LENGTH,&rowLength);glGetIntegerv(GL_PACK_SKIP_PIXELS,&skipPixels);glGetIntegerv(GL_PACK_SKIP_ROWS,&skipRows);
                    glGetIntegerv(GL_PACK_SWAP_BYTES,&swapBytes);
                    gl.BindBuffer(GL_PIXEL_PACK_BUFFER,0);glPixelStorei(GL_PACK_ALIGNMENT,1);glPixelStorei(GL_PACK_ROW_LENGTH,0);
                    glPixelStorei(GL_PACK_SKIP_PIXELS,0);glPixelStorei(GL_PACK_SKIP_ROWS,0);glBindTexture(GL_TEXTURE_2D,cameraSpectrumTexture);
                    glPixelStorei(GL_PACK_SWAP_BYTES,GL_FALSE);
                    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,spectrum->resolvedColor.data());
                    gl.BindBuffer(GL_PIXEL_PACK_BUFFER,unsigned(savedPack));glPixelStorei(GL_PACK_ALIGNMENT,alignment);
                    glPixelStorei(GL_PACK_ROW_LENGTH,rowLength);glPixelStorei(GL_PACK_SKIP_PIXELS,skipPixels);glPixelStorei(GL_PACK_SKIP_ROWS,skipRows);
                    glPixelStorei(GL_PACK_SWAP_BYTES,swapBytes);
                    if(glGetError()!=GL_NO_ERROR)return fail("OpenGL private grating resolved texture readback failed");
                    for(std::uint32_t y=0;y<scheduled.height/2;++y)for(std::uint32_t x=0;x<scheduled.width;++x)
                        std::swap(spectrum->resolvedColor[std::size_t(y)*scheduled.width+x],
                            spectrum->resolvedColor[std::size_t(scheduled.height-1-y)*scheduled.width+x]);
                    // Compare native echoes only. This does not construct or
                    // upload a CPU image or evaluate another material/scene.
                    std::vector<gc::DepositionKey> expectedKeys;expectedKeys.reserve(std::size_t(cameraDeposition.paddedKeyCount));
                    std::vector<float> expectedBands(std::size_t(pixels*8),0);std::size_t locator=0;
                    for(const auto& order:full.gratingCamera->orders) {
                        if(order.outcome!=gc::NativeGratingCameraReadback::Outcome::Visible||order.spectralRadiance==0)continue;
                        while(locator<locators.size()&&locators[locator]!=order.incident)++locator;
                        if(locator==locators.size())return fail("OpenGL private grating spectrum lost a native incident locator");
                        const auto& event=candidate.incidents[order.incident];unsigned slot=0;
                        while(slot<event.orderCount&&event.orders[slot].order!=order.order)++slot;
                        if(slot==event.orderCount)return fail("OpenGL private grating spectrum lost a native signed order");
                        const auto pixel=order.y*scheduled.width+order.x;
                        expectedKeys.push_back({pixel,std::uint32_t(locator),std::uint32_t(order.order+1),std::uint32_t(locator*3+slot)});
                        expectedBands[std::size_t(pixel)*8+order.band]+=order.spectralRadiance;
                    }
                    std::sort(expectedKeys.begin(),expectedKeys.end());expectedKeys.resize(spectrum->keys.size());
                    if(expectedKeys!=spectrum->keys)return fail("OpenGL private grating native stable keys differ from their order echoes");
                    for(std::size_t pixel=0;pixel<pixels;++pixel) {
                        std::array<float,3> xyz{};
                        for(unsigned band=0;band<8;++band) {
                            const auto actual=spectrum->spectralRadiance[pixel*8+band],expected=expectedBands[pixel*8+band];
                            if(!std::isfinite(actual)||actual<0||actual>surfacematerial::kMaximumLinearColorComponent
                                ||(actual!=expected&&(expected==0||std::abs(double(actual)-expected)>double(expected)*4e-6)))
                                return fail("OpenGL private grating native spectral reduction differs from its ordered echoes");
                            for(unsigned axis=0;axis<3;++axis)xyz[axis]+=actual*weights[band][3]*weights[band][axis];
                        }
                        const std::array<float,3> rgb{3.2406f*xyz[0]-1.5372f*xyz[1]-.4986f*xyz[2],
                            -.9689f*xyz[0]+1.8758f*xyz[1]+.0415f*xyz[2],.0557f*xyz[0]-.2040f*xyz[1]+1.0570f*xyz[2]};
                        for(unsigned axis=0;axis<3;++axis) {
                            const float actual=spectrum->resolvedColor[pixel][axis];
                            const float tolerance=std::max(1e-6f,std::abs(rgb[axis])*1e-3f)
                                +2e-6f*(std::abs(xyz[0])+std::abs(xyz[1])+std::abs(xyz[2]));
                            if(!std::isfinite(actual)||std::abs(actual)>surfacematerial::kMaximumLinearColorComponent
                                ||std::abs(actual-rgb[axis])>tolerance)
                                return fail("OpenGL private grating native CIE resolve differs from its spectral echo");
                        }
                        if(spectrum->resolvedColor[pixel][3]!=1)return fail("OpenGL private grating resolve alpha is incomplete");
                    }
                    std::uint64_t retainedBytes=binding->bounds.bytes+cameraBounds.bytes+bytes[10]*3+bytes[11]*6
                        +pixels*(8+sizeof(gc::Texel))+bytes[8]*2+sizeof(OpenGlGratingCameraSpectrum);
                    // No fallible native work follows this private candidate
                    // swap. A refusal above leaves the previous owner intact.
                    spectrum->complete=true;
                    const auto oldBuffers=frame->gratingCameraSpectrumBuffers;const auto oldTexture=frame->gratingCameraSpectrumTexture;
                    frame->gratingCameraSpectrumBuffers={buffers[11],buffers[12]};buffers[11]=buffers[12]=0;
                    frame->gratingCameraSpectrumTexture=cameraSpectrumTexture;cameraSpectrumTexture=0;
                    frame->gratingCameraSpectrum=spectrum;frame->gratingCameraSpectrumBytes=retainedBytes;
                    gl.DeleteBuffers(2,oldBuffers.data());if(oldTexture)glDeleteTextures(1,&oldTexture);
                    cameraDiagnostics->nativeBytes+=bytes[10]+bytes[11]+bytes[12]+pixels*8;
                    cameraDiagnostics->gratingSpectrumComplete=true;
                }
            }
            if(cellPass) {
                // Supplemental boundary samples retain the original native point
                // buffer and counters. They confer no finite-domain authority.
                if(resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max())
                    return fail("OpenGL cell transport supplemental serial is exhausted");
                const auto cellSerial=++resources->causticSubmissionSerial;
                auto submission=candidate.submission;
                submission.submissionSerial=submission.scratchGeneration=submission.completedFence=cellSerial;
                std::shared_ptr<const gc::GratingCellTransportBinding> cellBinding;
                if(!gc::prepareGratingCellTransportBinding(sourceCells,full.gratingIncidents,nativeFrame,
                    candidate.submission,submission,cellBinding,error)) {cleanup();return false;}
                const auto& bounds=cellBinding->bounds;
                const std::array<std::uint64_t,3> cellBytes{sourceCells->cellTexels.size()*sizeof(gc::Texel),
                    bounds.maximumRecords*bounds.recordStride*sizeof(gc::Texel),bounds.maximumRecords*sizeof(std::uint32_t)};
                const auto cellSource=gc::opengl::gratingCellTransportComputeSource();
                if(bounds.bytes>cellWorstBounds.bytes||bounds.maximumRecords>cellWorstBounds.maximumRecords
                    ||cellSource.capacity()>kQueryProgramSourceReservation/8
                    ||bounds.maximumRecords>std::uint64_t(groups)*32
                    ||std::any_of(cellBytes.begin(),cellBytes.end(),[&](auto size){return size>std::uint64_t(maximumBlock);}))
                    return fail("OpenGL cell transport native storage exceeds whole-frame admission");
                std::vector<std::uint32_t> denseLocators(std::size_t(bounds.maximumRecords),UINT32_MAX);
                for(std::size_t sparse=0;sparse<locators.size();++sparse)
                    if(locators[sparse]!=UINT32_MAX)denseLocators[locators[sparse]]=std::uint32_t(sparse);
                if(std::find(denseLocators.begin(),denseLocators.end(),UINT32_MAX)!=denseLocators.end())
                    return fail("OpenGL cell transport lost an original native incident locator");
                queryOwners[2]=linkQueryArithmeticProgram(gl,resources->queryContext,resources->rendererGeneration,{{GL_COMPUTE_SHADER,cellSource}},error);
                cellProgram=queryOwners[2]?queryOwners[2]->program:0;
                if(!cellProgram) {cleanup();return false;}
                gl.GenBuffers(int(cellBuffers.size()),cellBuffers.data());
                for(unsigned i=0;i<cellBuffers.size();++i) {
                    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,cellBuffers[i]);
                    const void* input=i==0?static_cast<const void*>(sourceCells->cellTexels.data())
                        :i==2?static_cast<const void*>(denseLocators.data()):nullptr;
                    gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(cellBytes[i]),input,i==1?GL_DYNAMIC_COPY:GL_STATIC_DRAW);
                    GLint64 actual=0;buffer64(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
                    if(!cellBuffers[i]||actual<0||std::uint64_t(actual)!=cellBytes[i])
                        return fail("OpenGL cell transport allocation differs from its reservation");
                    nativeBytes+=std::uint64_t(actual);
                }
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffers[1]);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,cellBuffers[0]);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,cellBuffers[1]);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,3,frame->causticBuffers[3]);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,4,frame->causticBuffers[4]);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,5,cellBuffers[2]);
                gl.UseProgram(cellProgram);const auto location=[&](const char* name){return gl.GetUniformLocation(cellProgram,name);};
                gl.Uniform1i(location("uRecordCount"),int(bounds.maximumRecords));
                gl.Uniform1i(location("uMaximumInterfaces"),int(bounds.maximumInterfaces));
                gl.Uniform1i(location("uRecordStride"),int(bounds.recordStride));
                gl.Uniform1i(location("uEntryCount"),int(sourceCells->cells.size()));
                gl.Uniform1i(location("uTriangleCount"),int(scheduled.triangles.size()));
                gl.Uniform4fv(location("uSource"),1,scheduled.sourceDirectionTrace.data());
                gl.Uniform4fv(location("uIor[0]"),2,scheduled.ior.data());
                gl.Uniform4fv(location("uOpaqueControl"),1,scheduled.query.control.data());
                gl.ActiveTexture(GL_TEXTURE0);gl.BindSampler(0,0);glBindTexture(GL_TEXTURE_2D,frame->causticTextures[0]);
                gl.Uniform1i(location("uOpaqueVisibility"),0);
                if(bounds.maximumRecords)gl.DispatchCompute(unsigned((bounds.maximumRecords+31)/32),1,1);
                gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
                if(!causticFence(gl,error)) {cleanup();return false;}
                if(stopForFixture==OpenGlGratingAttemptStopForFixture::CellFence)
                    {cleanup();error="OpenGL fixture refusal after real cell dispatch and completed fence";return false;}
                queryEvidence[2]=completedQueryArithmeticEvidence(queryOwners[2],OpenGlQueryArithmeticRole::CellBoundary,submission,
                    {queryOwners[0],{}},{queryEvidence[0],{}});
                auto upload=std::make_shared<gc::NativeGratingCellSourceUpload>();
                upload->version=gc::kGratingCellTransportVersion;upload->sourceCells=sourceCells;
                upload->pointReadback=full.gratingIncidents;upload->pointSubmission=candidate.submission;
                upload->submission=submission;upload->bindingGeneration=cellSerial;
                upload->cellTexels.resize(sourceCells->cellTexels.size());
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,cellBuffers[0]);
                const auto* input=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(cellBytes[0]),GL_MAP_READ_BIT);
                if(!input)return fail("OpenGL cell transport completed source input mapping failed");
                std::memcpy(upload->cellTexels.data(),input,std::size_t(cellBytes[0]));
                if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return fail("OpenGL cell transport completed source input is invalid");
                upload->complete=true;
                std::vector<gc::Texel> rows(std::size_t(bounds.maximumRecords*bounds.recordStride));
                if(cellBytes[1]) {
                    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,cellBuffers[1]);
                    const auto* native=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(cellBytes[1]),GL_MAP_READ_BIT);
                    if(!native)return fail("OpenGL cell transport completed trace mapping failed");
                    std::memcpy(rows.data(),native,std::size_t(cellBytes[1]));
                    if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return fail("OpenGL cell transport completed traces are invalid");
                }
                gc::NativeGratingCellTransportReadback witness;
                witness.version=gc::kGratingCellTransportVersion;witness.binding=cellBinding;
                witness.submission=submission;witness.bindingGeneration=cellSerial;
                witness.records.resize(std::size_t(bounds.maximumRecords));
                for(std::size_t i=0;i<witness.records.size();++i) {
                    if(!gc::decodeGratingCellTransportRecord(rows.data()+i*bounds.recordStride,*cellBinding,witness.records[i],error))
                        {cleanup();return false;}
                    if(witness.records[i].queryWork>bounds.queryWork-witness.queryWork)
                        return fail("OpenGL cell transport native query work exceeds supplemental admission");
                    witness.queryWork+=witness.records[i].queryWork;
                }
                if(totalQuery+candidate.orderWork+witness.queryWork>gc::kMaximumQueryWork||glGetError()!=GL_NO_ERROR)
                    return fail("OpenGL cell transport native completion exceeds whole-frame work");
                witness.complete=true;
                const auto previousUpload=frame->gratingCellSourceUpload;
                const auto previousBytes=frame->gratingCellSourceBytes;
                frame->gratingCellSourceUpload=upload;
                frame->gratingCellSourceBytes=sourceCells->bounds.bytes+binding->bounds.bytes+bounds.hostBytes;
                if(full.gratingIncidents->directionalSource) {
                    const auto& retainedSource=*full.gratingIncidents->directionalSource;
                    frame->gratingCellSourceBytes+=sizeof(gc::NativeDirectionalSourceReceipt)
                        +retainedSource.potential.capacity()*sizeof(float)+retainedSource.submission.frameIdentity.capacity();
                }
                frame->nativeAttempts[attemptIndex].adjunctStorage[1]=frame->gratingCellSourceBytes;
                frame->nativeAttempts[attemptIndex].adjunctOwners[1]=upload.get();
                int rollbackToken=0;auto rollback=[&](int*) {
                    frame->gratingCellSourceUpload=previousUpload;frame->gratingCellSourceBytes=previousBytes;
                    frame->nativeAttempts[attemptIndex].adjunctStorage[1]=0;frame->nativeAttempts[attemptIndex].adjunctOwners[1]=nullptr;
                };
                std::unique_ptr<int,decltype(rollback)> rollbackGuard(&rollbackToken,rollback);
                if(!gc::validateGratingCellTransportReadback(*cellBinding,witness,nativeFrame,submission,error))
                    {cleanup();return false;}
                std::shared_ptr<const gc::GratingCellCoverageCandidate> coverageCandidate;
                std::shared_ptr<const gc::GratingCellCoveragePayload> coveragePayload;
                gc::NativeGratingCellCoverageEvidence evidence;
                if(coveragePass) {
                    if(resources->causticSubmissionSerial==std::numeric_limits<std::uint64_t>::max())
                        return fail("OpenGL native interval coverage serial is exhausted");
                    const auto coverageSerial=++resources->causticSubmissionSerial;
                    auto coverageSubmission=submission;
                    coverageSubmission.submissionSerial=coverageSubmission.scratchGeneration=coverageSubmission.completedFence=coverageSerial;
                    auto boundary=std::make_shared<const gc::NativeGratingCellTransportReadback>(std::move(witness));
                    if(!gc::prepareGratingCellCoverageCandidate(boundary,nativeFrame,submission,coverageSubmission,
                        maximumLeaves,coverageCandidate,error)) {cleanup();return false;}
                    unsigned level=0,leaves=1;
                    while(leaves*4<=maximumLeaves) {leaves*=4;++level;}
                    const auto sweepStride=16+coverageCandidate->bounds.maximumBlockers*2;
                    const auto leafStride=3+(bounds.maximumInterfaces+1)*sweepStride;
                    const auto invocations=bounds.maximumRecords*leaves;
                    const auto coverageBytes=invocations*leafStride*sizeof(gc::Texel);
                    const auto intervalSource=gc::opengl::gratingCellCoverageComputeSource();
                    if(coverageCandidate->bounds.bytes>coverageWorstBounds.bytes||invocations>cellWorstBounds.maximumRecords*maximumLeaves
                        ||intervalSource.capacity()>kQueryProgramSourceReservation/8||coverageBytes>coverageCandidate->bounds.nativeBytes
                        ||coverageBytes>std::uint64_t(maximumBlock)||invocations>std::uint64_t(groups)*32)
                        return fail("OpenGL native interval coverage exceeds whole-frame storage or driver admission");
                    queryOwners[3]=linkQueryArithmeticProgram(gl,resources->queryContext,resources->rendererGeneration,{{GL_COMPUTE_SHADER,intervalSource}},error);
                    coverageProgram=queryOwners[3]?queryOwners[3]->program:0;
                    if(!coverageProgram) {cleanup();return false;}
                    gl.GenBuffers(1,&coverageBuffer);gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,coverageBuffer);
                    gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(coverageBytes),nullptr,GL_DYNAMIC_COPY);
                    GLint64 actual=0;buffer64(GL_SHADER_STORAGE_BUFFER,GL_BUFFER_SIZE,&actual);
                    if(!coverageBuffer||actual<0||std::uint64_t(actual)!=coverageBytes)
                        return fail("OpenGL native interval proof allocation differs from admission");
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffers[1]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,cellBuffers[0]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,2,cellBuffers[1]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,3,coverageBuffer);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,4,frame->causticBuffers[3]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,5,frame->causticBuffers[4]);
                    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,6,cellBuffers[2]);
                    gl.UseProgram(coverageProgram);const auto location=[&](const char* name){return gl.GetUniformLocation(coverageProgram,name);};
                    gl.Uniform1i(location("uRecordCount"),int(bounds.maximumRecords));gl.Uniform1i(location("uLeaves"),int(leaves));
                    gl.Uniform1i(location("uLevel"),int(level));gl.Uniform1i(location("uLeafStride"),int(leafStride));
                    gl.Uniform1i(location("uSweepStride"),int(sweepStride));gl.Uniform1i(location("uBoundaryStride"),int(bounds.recordStride));
                    gl.Uniform1i(location("uMaximumInterfaces"),int(bounds.maximumInterfaces));
                    gl.Uniform1i(location("uTriangleCount"),int(scheduled.triangles.size()));
                    gl.Uniform1i(location("uQueryAllowance"),int(gc::gratingCellQueryAllowance(scheduled)));
                    gl.Uniform4fv(location("uSource"),1,scheduled.sourceDirectionTrace.data());
                    gl.Uniform4fv(location("uIor[0]"),2,scheduled.ior.data());
                    if(invocations)gl.DispatchCompute(unsigned((invocations+31)/32),1,1);
                    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
                    if(!causticFence(gl,error)) {cleanup();return false;}
                    queryEvidence[3]=completedQueryArithmeticEvidence(queryOwners[3],OpenGlQueryArithmeticRole::CellCoverage,coverageSubmission,
                        {queryOwners[0],queryOwners[2]},{queryEvidence[0],queryEvidence[2]});
                    std::vector<gc::Texel> proofRows(std::size_t(coverageBytes/sizeof(gc::Texel)));
                    if(coverageBytes) {
                        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,coverageBuffer);
                        const auto* native=gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,std::ptrdiff_t(coverageBytes),GL_MAP_READ_BIT);
                        if(!native)return fail("OpenGL completed native interval proof mapping failed");
                        std::memcpy(proofRows.data(),native,std::size_t(coverageBytes));
                        if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return fail("OpenGL completed native interval proof is invalid");
                    }
                    auto payload=std::make_shared<gc::GratingCellCoveragePayload>();
                    if(!gc::opengl::decodeGratingCellCoverage(proofRows.data(),proofRows.size(),*coverageCandidate,leaves,level,*payload,error))
                        {cleanup();return false;}
                    if(payload->queryWork>coverageCandidate->bounds.queryWork
                        ||totalQuery+candidate.orderWork+boundary->queryWork+payload->queryWork>gc::kMaximumQueryWork
                        ||glGetError()!=GL_NO_ERROR)return fail("OpenGL completed interval sweep work exceeds exact native admission");
                    coveragePayload=payload;
                    evidence.version=gc::kGratingCellCoverageVersion;evidence.rule=gc::kGratingCellCoverageRule;
                    evidence.candidate=coverageCandidate;evidence.boundary=boundary;evidence.payload=coveragePayload;
                    evidence.boundarySubmission=submission;evidence.submission=coverageSubmission;
                    evidence.bindingGeneration=coverageSerial;evidence.nativeBytes=coverageBytes;evidence.complete=true;
                    const auto previousEvidence=frame->gratingCoverageEvidence;
                    const auto previousCoverageBytes=frame->gratingCoverageBytes;
                    frame->gratingCoverageEvidence=evidence;
                    frame->gratingCoverageBytes=coverageCandidate->bounds.hostBytes;
                    frame->nativeAttempts[attemptIndex].adjunctStorage[2]=frame->gratingCoverageBytes;
                    frame->nativeAttempts[attemptIndex].adjunctOwners[2]=coveragePayload.get();
                    int evidenceToken=0;auto restoreEvidence=[&](int*) {
                        frame->gratingCoverageEvidence=previousEvidence;frame->gratingCoverageBytes=previousCoverageBytes;
                        frame->nativeAttempts[attemptIndex].adjunctStorage[2]=0;frame->nativeAttempts[attemptIndex].adjunctOwners[2]=nullptr;
                    };
                    std::unique_ptr<int,decltype(restoreEvidence)> evidenceGuard(&evidenceToken,restoreEvidence);
                    if(!gc::validateNativeGratingCellCoverage(coverageCandidate,coveragePayload,nativeFrame,coverageSubmission,error))
                        {cleanup();return false;}
                    evidenceGuard.release();
                }
                // Publish only native echoes validated against this exact owner.
                if(coveragePass) {
                    coverageOutput->candidate=std::move(coverageCandidate);coverageOutput->payload=std::move(coveragePayload);
                } else *cellOutput=std::move(witness);
                rollbackGuard.release();
            }
            *terminalOutput=std::move(full);*terminalEnergy=ledger;
        }
        for(std::size_t index=0;index<queryOwners.size();++index)if(queryOwners[index]&&queryEvidence[index])
            frame->retainQueryArithmetic(queryEvidence[index]->role,queryOwners[index],queryEvidence[index]);
        // Register exact completed owners only after every failing native and
        // capacity check. Failed prefixes keep their immutable work entry.
        frame->nativeAttempts[attemptIndex].adjunctStorage={frame->gratingCameraSpectrumBytes,frame->gratingCellSourceBytes,frame->gratingCoverageBytes};
        frame->nativeAttempts[attemptIndex].adjunctOwners={frame->gratingCameraSpectrum.get(),frame->gratingCellSourceUpload.get(),frame->gratingCoverageEvidence.payload.get()};
        cleanup();output=std::move(candidate);
        if(diagnostics)*diagnostics={total,nativeBytes,baseQuery,output.queryWork,output.orderWork};
        error.clear();return true;
    }
    catch(const std::bad_alloc&)
    {error="OpenGL grating incident admitted allocation failed";return false;}
#endif
}

bool produceOpenGlGlassGratingIncidents(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeGratingReadback& output,std::string& error,OpenGlGratingIncidentDiagnostics* diagnostics)
{
    return produceOpenGlGlassGratingReadback(nativeFrame,binding,output,error,diagnostics,nullptr,nullptr);
}
bool produceOpenGlGlassGratingAttemptForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& cells,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,glasscaustics::NativeGratingCellTransportReadback& cellOutput,
    OpenGlGratingAttemptStopForFixture stop,std::string& error)
{
    if(stop==OpenGlGratingAttemptStopForFixture::None||(stop==OpenGlGratingAttemptStopForFixture::CellFence&&!cells))
    {error="OpenGL fixture requires a real point or cell refusal seam";return false;}
    glasscaustics::NativeGratingReadback adjunct;
    return produceOpenGlGlassGratingReadback(frame,binding,adjunct,error,nullptr,&output,&energy,nullptr,cells,cells?&cellOutput:nullptr,nullptr,4,stop);
}
bool produceOpenGlGlassGratingTerminalReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error,
    OpenGlGratingIncidentDiagnostics* diagnostics)
{
    glasscaustics::NativeGratingReadback adjunct;
    return produceOpenGlGlassGratingReadback(nativeFrame,binding,adjunct,error,diagnostics,&output,&energy);
}

bool produceOpenGlGlassGratingCameraReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error,
    OpenGlGratingCameraDiagnostics* diagnostics)
{
    glasscaustics::NativeGratingReadback adjunct;OpenGlGratingCameraDiagnostics measured;
    if(!produceOpenGlGlassGratingReadback(nativeFrame,binding,adjunct,error,nullptr,&output,&energy,&measured))return false;
    if(diagnostics)*diagnostics=measured;
    return true;
}

bool produceOpenGlGlassGratingCellTransportReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    glasscaustics::NativeReadback& pointOutput,glasscaustics::EnergyLedger& energy,
    glasscaustics::NativeGratingCellTransportReadback& output,std::string& error)
{
    glasscaustics::NativeGratingReadback adjunct;
    return produceOpenGlGlassGratingReadback(nativeFrame,sourceCells?sourceCells->incidentBinding:nullptr,
        adjunct,error,nullptr,&pointOutput,&energy,nullptr,sourceCells,&output);
}

bool produceOpenGlGlassGratingCellCoverageReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    glasscaustics::NativeReadback& pointOutput,glasscaustics::EnergyLedger& energy,
    OpenGlGratingCellCoverageReadback& output,std::string& error,std::size_t maximumLeaves)
{
    glasscaustics::NativeGratingReadback adjunct;glasscaustics::NativeGratingCellTransportReadback boundary;
    return produceOpenGlGlassGratingReadback(nativeFrame,sourceCells?sourceCells->incidentBinding:nullptr,
        adjunct,error,nullptr,&pointOutput,&energy,nullptr,sourceCells,&boundary,&output,maximumLeaves);
}

bool readOpenGlGratingCameraSpectrumForFixture(const NativeFixtureSceneFrame& nativeFrame,
    std::shared_ptr<const OpenGlGratingCameraSpectrum>& output,std::string& error)
{
#if defined(__APPLE__)
    (void)nativeFrame;(void)output;
    error="OpenGL private grating spectrum requires native OpenGL 4.3 storage";return false;
#else
    std::lock_guard<std::mutex> lock(fixtureMutex());
    const auto* frame=dynamic_cast<const OpenGlFixtureSceneFrame*>(&nativeFrame);
    const auto isBuffer=reinterpret_cast<PFNGLISBUFFERPROC>(glfwGetProcAddress("glIsBuffer"));
    const auto spectrum=frame?frame->gratingCameraSpectrum:nullptr;
    if(!frame||!frame->staticResources||frame->ownerContext!=glfwGetCurrentContext()
        ||!frame->privateCameraAttachments||frame->gratingCameraBaseComplete||!frame->gratingStagingBinding
        ||!spectrum||!spectrum->complete||!spectrum->camera||!spectrum->camera->complete||spectrum->camera->physicalColor
        ||spectrum->width!=frame->width_||spectrum->height!=frame->height_
        ||spectrum->camera->submission.frameIdentity!=frame->gratingStagingBinding->identity
        ||spectrum->camera->submission.rendererGeneration!=frame->rendererGeneration_
        ||!spectrum->camera->submission.completedFence||!frame->gratingCameraSpectrumBytes
        ||!glIsTexture(frame->gratingCameraSpectrumTexture)||!isBuffer
        ||!isBuffer(frame->gratingCameraSpectrumBuffers[0])||!isBuffer(frame->gratingCameraSpectrumBuffers[1]))
    {error="OpenGL private grating spectrum lacks its retained completed native owner";return false;}
    output=spectrum;error.clear();return true;
#endif
}

NativeDeformationBackend& nativeDeformationBackend()
{
    static OpenGlDeformationBackend backend;
    return backend;
}
} // namespace arbitgpu
