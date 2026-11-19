#pragma once

// The Realtime engine's fragment text and the scope that selects its program.
// Realtime rasterizes the same scene and materials with no ray walks through the
// glass BVH. A frame draws in phases: the opaque image, screen-space reflections
// over it, then each glass object over a copy of what is behind it. Every step
// count here is a quality constant of one approximation. None of them depends on
// the scene's size or content, and the marches scale with the camera's depth span.
// Preview and Export draw the same phases, so Accurate viewports match their export.
// A Render Mode Fast preview skips the reflection phase (render_mode_policy.h).
// A diffraction grating keeps the raster order sum between phases 0 and 1, so its
// highlights and spectral lines are real orders that reflections and glass can see.
// Only the receiver transport approximates, in backend_opengl.cpp, with a gather
// whose size is the square root of the scene's sample rows.

#include <algorithm>
#include <array>
#include <cmath>

namespace arbitgpu::realtimeengine
{
inline bool& currentlyActive() noexcept
{
    thread_local bool value = false;
    return value;
}

// True while the renderer prepares and draws a Realtime frame on this thread.
inline bool active() noexcept { return currentlyActive(); }

class Scope final
{
public:
    Scope() noexcept : previous_ (currentlyActive()) { currentlyActive() = true; }
    ~Scope() { currentlyActive() = previous_; }
    Scope (const Scope&) = delete;
    Scope& operator= (const Scope&) = delete;

private:
    bool previous_;
};

// The phases a frame draws: 0 opaque, 1 reflections over it, 2 each glass object over
// a snapshot of what is behind it. -1 is one draw with no snapshots.
enum Phase : int { kSingle = -1, kOpaque = 0, kReflections = 1, kGlass = 2 };

// Which surfaces a phase draws. Glass waits for phase 2, and reflections skip blended
// surfaces, whose depth they cannot match.
inline bool drawsInPhase (int phase, bool glass, bool blended) noexcept
{
    if (phase == kGlass) return glass;
    if (phase == kOpaque) return ! glass;
    if (phase == kReflections) return ! glass && ! blended;
    return true;
}

// How a Realtime frame draws. A phased frame (Color output, no inspection image) runs
// phases 0, 1 and 2. A grating frame with other Surface objects draws them in phase 0,
// then its grating, then reflections and glass over the orders. A lone grating has no
// surfaces to phase and draws once.
enum class FramePhases { kSingle, kSurfaces, kAroundGrating };
inline FramePhases framePhases (bool phased, bool grating, bool gratingWithSurfaces) noexcept
{
    if (! phased) return FramePhases::kSingle;
    if (! grating) return FramePhases::kSurfaces;
    return gratingWithSurfaces ? FramePhases::kAroundGrating : FramePhases::kSingle;
}

// A thin water interface's roughness for one Realtime shading sample. Far off, the
// open sea's triangles shrink below a pixel and its normal turns across the pixel,
// but one sample reads a single mirror direction from that spread, sky in one pixel
// and cloud or horizon in the next: a speckle Path Trace does not show, because its
// samples average the spread. dx and dy are the normal's screen derivatives. Their
// variance widens the GGX lobe (alpha squared, Kaplanyan, Hill, Patney and Lefohn
// 2016, "Filtering Distributions of Normals for Shading Antialiasing", with their
// 0.18 bound), so the sample reads the pixel's average. Near water, whose normal
// barely turns over a pixel, keeps its own roughness. A non-finite derivative takes
// the bound instead of reaching the shading.
inline float footprintRoughness (float roughness, const std::array<float, 3>& dx, const std::array<float, 3>& dy) noexcept
{
    const float variance = 0.25f * (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
                                    + dy[0] * dy[0] + dy[1] * dy[1] + dy[2] * dy[2]);
    const float kernel = variance == variance ? std::min (2.0f * variance, 0.18f) : 0.18f;
    const float alpha = roughness * roughness;
    return std::sqrt (std::sqrt (std::clamp (alpha * alpha + kernel, 0.0f, 1.0f)));
}

// A glass sample's shading normal, as Path Trace's faceViewer chooses it. An
// interpolated normal can turn away from the camera on a facet the camera sees: on
// the far slope of each wave the open sea's normals do so, for about half the water
// in view. Negating it there pointed the normal into the sea, so its Fresnel fell
// to almost nothing and its reflection read the ground under the horizon, while the
// next pixel read the bright grazing sky. The facet the camera sees decides the
// side instead, and a normal that still turns away tilts just far enough to face
// the camera. facet is the triangle's normal with either sign (zero when unknown,
// which falls back to the view's side), and view the unit direction to the camera.
inline std::array<float, 3> viewerNormal (std::array<float, 3> shaded, std::array<float, 3> facet,
                                          const std::array<float, 3>& view) noexcept
{
    const auto dot = [] (const std::array<float, 3>& a, const std::array<float, 3>& b)
        { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const float facetLength = std::sqrt (dot (facet, facet));
    std::array<float, 3> seen {};
    if (facetLength > 0.0f)
        for (int c = 0; c < 3; ++c) seen[c] = facet[c] / facetLength;
    if (dot (seen, view) < 0.0f)
        for (auto& c : seen) c = -c;
    if (facetLength > 0.0f ? dot (shaded, seen) < 0.0f : dot (shaded, view) < 0.0f)
        for (auto& c : shaded) c = -c;
    if (dot (shaded, view) > 1.0e-4f) return shaded;
    const float lift = 1.0e-3f - dot (shaded, view);
    std::array<float, 3> tilted {shaded[0] + view[0] * lift, shaded[1] + view[1] * lift, shaded[2] + view[2] * lift};
    const float tiltedLength = std::sqrt (dot (tilted, tilted));
    if (tiltedLength > 0.0f && dot (tilted, view) > 1.0e-4f * tiltedLength)
    {
        for (auto& c : tilted) c /= tiltedLength;
        return tilted;
    }
    return facetLength > 0.0f ? seen : view;
}

// Follows the opaque visibility, sky, environment and pass composite text, and
// comes before the tail, which defines rotateQ and calls these from main.
inline const char* glslShader() noexcept
{
    return R"glsl(
uniform vec4 uRealtimeControl;
uniform sampler2D uRealtimeColor;
uniform sampler2D uRealtimeDepth;
// The planar mirror target (realtime_planar_mirror.h), read in phase 1 when
// uRealtimeControl.y is 1: a grating's mirror image, alpha 1 where it has one.
uniform sampler2D uRealtimeMirror;
// A closed glass object's back faces, drawn just before it in phase 2 when
// uRealtimeControl.z is 1: world normal in xyz, linear depth in w, zero where none.
uniform sampler2D uRealtimeBack;
// uCameraRotation, uCameraTranslation and uProjection come from the fragment head.
// A second declaration here is an error on NVIDIA (C1038), though Mesa accepts it.
vec3 rotateQ(vec4 q, vec3 v);

// A light without an authored size is treated as this angular radius.
const float kRealtimeLightRadius = 0.03;
const float kRealtimeGoldenAngle = 2.39996323;

// Interleaved gradient noise. It is a function of the pixel only, so a repeated
// frame repeats exactly.
float realtimeNoise() {
    return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
}

// Texture position in xy and linear depth in z, by the vertex shader's projection.
// A point at or behind the camera has a negative position.
vec3 realtimeProject(vec3 world) {
    vec3 camera = rotateQ(vec4(-uCameraRotation.xyz, uCameraRotation.w), world - uCameraTranslation);
    float distance = -camera.z;
    if (distance <= 0.0) return vec3(-1.0);
    // uProjection.x below zero is an orthographic camera, minus half the view height.
    vec2 ndc = uProjection.x < 0.0
        ? vec2(camera.x / (-uProjection.x * uProjection.y), camera.y / -uProjection.x)
        : vec2(camera.x / (uProjection.x * uProjection.y), camera.y / uProjection.x) / distance;
    return vec3(ndc * 0.5 + 0.5, (distance - uProjection.z) / max(uProjection.w - uProjection.z, 1e-6));
}

bool realtimeOnScreen(vec3 position) {
    return all(greaterThanEqual(position, vec3(0.0))) && all(lessThanEqual(position, vec3(1.0)));
}

bool realtimeSurfaceDepth(float depth) { return depth > 0.0 && depth < 1.0; }

// The world point the depth snapshot holds at texture position uv and linear depth.
vec3 realtimeUnproject(vec2 uv, float depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    float distance = uProjection.z + depth * (uProjection.w - uProjection.z);
    vec3 camera = uProjection.x < 0.0
        ? vec3(ndc.x * -uProjection.x * uProjection.y, ndc.y * -uProjection.x, -distance)
        : vec3(ndc.x * uProjection.x * uProjection.y, ndc.y * uProjection.x, -1.0) * distance;
    return rotateQ(uCameraRotation, camera) + uCameraTranslation;
}

// A planar mirror draw keeps a grating fragment only where the frame's visible surface
// lies on the mirror plane and the grating point is on the camera's side of it.
bool realtimeMirrorKeeps(vec3 world) {
    float span = max(uProjection.w - uProjection.z, 1e-6);
    if (dot(uViewerMirror.xyz, world) + uViewerMirror.w < -1e-4 * span) return false;
    vec2 size = vec2(textureSize(uRealtimeDepth, 0));
    float depth = texelFetch(uRealtimeDepth, ivec2(gl_FragCoord.xy), 0).r;
    if (!realtimeSurfaceDepth(depth)) return false;
    vec3 surface = realtimeUnproject(gl_FragCoord.xy / size, depth);
    return abs(dot(uViewerMirror.xyz, surface) + uViewerMirror.w) <= 1e-3 * span;
}

// Marches the depth snapshot in steps that grow with distance. On a hit, true and the
// hit's texture position and depth.
bool realtimeTraceHit(vec3 origin, vec3 direction, out vec3 hit) {
    hit = vec3(-1.0);
    float span = max(uProjection.w - uProjection.z, 1e-6);
    float previous = 0.0;
    float travel = span * 0.001 * (0.5 + realtimeNoise());
    for (int step = 0; step < 40; ++step) {
        vec3 position = realtimeProject(origin + direction * travel);
        if (!realtimeOnScreen(position)) return false;
        float depth = texture(uRealtimeDepth, position.xy).r;
        float thickness = 2.0 * (travel - previous) / span;
        if (realtimeSurfaceDepth(depth) && position.z > depth + 0.0001 && position.z - depth <= thickness) {
            float low = previous;
            float high = travel;
            for (int refine = 0; refine < 5; ++refine) {
                float middle = 0.5 * (low + high);
                vec3 candidate = realtimeProject(origin + direction * middle);
                if (candidate.z > texture(uRealtimeDepth, candidate.xy).r) high = middle;
                else low = middle;
            }
            hit = realtimeProject(origin + direction * high);
            return true;
        }
        previous = travel;
        travel *= 1.25;
    }
    return false;
}

// Confidence in a hit that fades at the frame's edge.
float realtimeEdgeConfidence(vec2 position) {
    vec2 edge = min(position, 1.0 - position);
    return clamp(min(edge.x, edge.y) * 12.0, 0.0, 1.0);
}

// On a hit, the snapshot's color and the hit's edge confidence come back.
bool realtimeTrace(vec3 origin, vec3 direction, out vec3 color, out float confidence) {
    color = vec3(0.0);
    confidence = 0.0;
    vec3 hit;
    if (!realtimeTraceHit(origin, direction, hit)) return false;
    color = texture(uRealtimeColor, hit.xy).rgb;
    confidence = realtimeEdgeConfidence(hit.xy);
    return true;
}

vec3 realtimeFresnel(vec3 f0, vec3 normal, vec3 view) {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - clamp(dot(normal, view), 0.0, 1.0), 5.0);
}

// viewerNormal in C++ is the same choice.
vec3 realtimeViewerNormal(vec3 shadedNormal, vec3 facet, vec3 view) {
    float facetLength = length(facet);
    vec3 seen = facetLength > 0.0 ? facet / facetLength : vec3(0.0);
    if (dot(seen, view) < 0.0) seen = -seen;
    vec3 normal = shadedNormal;
    if (facetLength > 0.0 ? dot(normal, seen) < 0.0 : dot(normal, view) < 0.0) normal = -normal;
    if (dot(normal, view) > 1.0e-4) return normal;
    vec3 tilted = normal + view * (1.0e-3 - dot(normal, view));
    float tiltedLength = length(tilted);
    if (tiltedLength > 0.0 && dot(tilted, view) > 1.0e-4 * tiltedLength) return tilted / tiltedLength;
    return facetLength > 0.0 ? seen : view;
}

// footprintRoughness in C++ is the same formula.
float realtimeFootprintRoughness(float roughness, vec3 dx, vec3 dy) {
    float variance = 0.25 * (dot(dx, dx) + dot(dy, dy));
    float kernel = variance == variance ? min(2.0 * variance, 0.18) : 0.18;
    float alpha = roughness * roughness;
    return sqrt(sqrt(clamp(alpha * alpha + kernel, 0.0, 1.0)));
}

// Rays from position toward points spread over area light i: record 3's w holds
// half its X extent and record 2's w half its Y extent, negative for a disk or
// ellipse. A long rectangle's shadow therefore softens more along its length.
// Four rays per pixel, with per-pixel rotation, stand in for the integral.
float realtimeAreaVisibility(int i, vec3 position, vec3 normal) {
    vec4 rotation = fixtureLightRotation(i);
    vec3 centre = fixtureLightPosition(i).xyz;
    float halfY = fixtureLightPosition(i).w;
    bool ellipse = halfY < 0.0;
    halfY = abs(halfY);
    float halfX = fixtureLightSize(i);
    vec3 axisX = rotateQ(rotation, vec3(1.0, 0.0, 0.0));
    vec3 axisY = rotateQ(rotation, vec3(0.0, 1.0, 0.0));
    float angle = realtimeNoise() * 6.28318531;
    float sum = 0.0;
    for (int ray = 0; ray < 4; ++ray) {
        angle += kRealtimeGoldenAngle;
        vec2 unit = vec2(cos(angle), sin(angle)) * sqrt((float(ray) + 0.5) / 4.0);
        // Stretch the disc to the square, so the samples cover a rectangle's corners.
        if (!ellipse) unit /= max(max(abs(cos(angle)), abs(sin(angle))), 1e-4);
        vec3 delta = centre + axisX * (unit.x * halfX) + axisY * (unit.y * halfY) - position;
        float distance = max(length(delta), 1e-6);
        sum += opaqueVisibility(position, normal, delta / distance, distance, uOpaqueControl);
    }
    return sum * 0.25;
}

// Several rays across the light's disc, for directional, point and spot lights.
// angularRadius is the angle the light subtends from position; 0 uses
// kRealtimeLightRadius. Area lights use realtimeAreaVisibility instead.
float realtimeSoftVisibility(vec3 position, vec3 normal, vec3 toLight, float maxDistance, float angularRadius) {
    float lightRadius = angularRadius > 0.0 ? tan(min(angularRadius, 1.5)) : kRealtimeLightRadius;
    vec3 axis = normalize(toLight);
    vec3 tangent = normalize(abs(axis.z) < 0.9 ? cross(axis, vec3(0.0, 0.0, 1.0)) : cross(axis, vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(axis, tangent);
    float angle = realtimeNoise() * 6.28318531;
    float sum = 0.0;
    for (int ray = 0; ray < 4; ++ray) {
        float radius = lightRadius * sqrt((float(ray) + 0.5) / 4.0);
        angle += kRealtimeGoldenAngle;
        vec3 direction = normalize(axis + (tangent * cos(angle) + bitangent * sin(angle)) * radius);
        sum += opaqueVisibility(position, normal, direction, maxDistance, uOpaqueControl);
    }
    return sum * 0.25;
}

// The sky is a light from every direction, so its shadow is a hemisphere of rays.
float realtimeSkyVisibility(vec3 position, vec3 normal) {
#ifdef REALTIME_ABLATE_SKYVIS
    return 1.0;
#endif
    vec3 axis = normalize(normal);
    vec3 tangent = normalize(abs(axis.z) < 0.9 ? cross(axis, vec3(0.0, 0.0, 1.0)) : cross(axis, vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(axis, tangent);
    float angle = realtimeNoise() * 6.28318531;
    float sum = 0.0;
    for (int ray = 0; ray < 4; ++ray) {
        float radial = (float(ray) + 0.5) / 4.0;
        angle += kRealtimeGoldenAngle;
        vec3 direction = axis * sqrt(1.0 - radial) + (tangent * cos(angle) + bitangent * sin(angle)) * sqrt(radial);
        sum += opaqueVisibility(position, normal, normalize(direction), 3.402823e38, uOpaqueControl);
    }
    return sum * 0.25;
}

// Reflected scene color over the opaque image. The alpha takes out the environment
// reflection the base shading already holds where the march found an object.
vec4 realtimeReflection(vec3 position, vec3 normal, vec3 toView, vec3 base, float metallic, float roughness, float dielectric) {
    vec3 view = normalize(toView);
    vec3 direction = reflect(-view, normal);
    float gloss = (1.0 - roughness) * (1.0 - roughness);
    vec3 color;
    float confidence = 1.0;
    // A grating's planar mirror image, where the mirror target holds one, else the march.
    vec4 mirrored = uRealtimeControl.y > 0.5 ? texelFetch(uRealtimeMirror, ivec2(gl_FragCoord.xy), 0) : vec4(0.0);
    if (gloss <= 0.0) return vec4(0.0);
    if (mirrored.a > 0.0) color = mirrored.rgb;
    else if (!realtimeTrace(position + normal * 0.0005 * (uProjection.w - uProjection.z), direction, color, confidence))
        return vec4(0.0);
    vec3 fresnel = realtimeFresnel(mix(vec3(dielectric), base, metallic), normal, view) * gloss * confidence;
    return vec4(fresnel * color, max(max(fresnel.r, fresnel.g), fresnel.b));
}

// The environment in a mirror direction, weighted only by the caller's Fresnel. The
// scene's ambient colour is environment radiance from every direction, as Path Trace
// adds it (pathtrace/lights.h).
vec3 realtimeEnvironment(vec3 normal, vec3 toView, float roughness, float dielectric) {
    if (uSky[0].w <= 0.5) return uAmbient;
    return uAmbient + (uSky[0].w > 1.5
        ? environmentImageBase(vec3(1.0), 1.0, roughness, dielectric, normal, toView, uSky[0], uSky[1])
        : analyticSkyBase(vec3(1.0), 1.0, roughness, dielectric, normal, toView, uSky[0], uSky[1], uSky[2], uSky[3], uSky[4]));
}

// Where a ray bent into a closed glass object at position leaves it: the march follows
// it over the object's back-face depth (uRealtimeBack) to the first texel it passes,
// with a first step from the object's own depth there. True with the exit point and
// the back face's outward normal; false when the pixel has no back face.
bool realtimeBackExit(vec3 position, vec3 inside, out vec3 exitPoint, out vec3 exitNormal) {
    exitPoint = position;
    exitNormal = vec3(0.0);
    float span = max(uProjection.w - uProjection.z, 1e-6);
    vec4 here = texelFetch(uRealtimeBack, ivec2(gl_FragCoord.xy), 0);
    vec3 own = realtimeProject(position);
    if (here.w <= 0.0 || own.x < 0.0 || here.w <= own.z) return false;
    // How fast the ray deepens, from the camera's forward axis.
    vec3 forward = uProjection.x < 0.0 ? rotateQ(uCameraRotation, vec3(0.0, 0.0, -1.0))
                                       : normalize(position - uCameraTranslation);
    float estimate = (here.w - own.z) * span / max(dot(inside, forward), 0.2);
    vec2 size = vec2(textureSize(uRealtimeBack, 0));
    float low = 0.0;
    float high = -1.0;
    exitNormal = here.xyz;
    for (int step = 1; step <= 12; ++step) {
        float travel = estimate * 3.0 * float(step) / 12.0;
        vec3 at = realtimeProject(position + inside * travel);
        if (!realtimeOnScreen(at)) { high = travel; break; }
        vec4 back = texelFetch(uRealtimeBack, ivec2(at.xy * size), 0);
        // Past the object's outline the ray has left it.
        if (back.w <= 0.0 || at.z >= back.w) { high = travel; break; }
        low = travel;
        exitNormal = back.xyz;
    }
    if (high < 0.0) high = estimate * 3.0;
    for (int refine = 0; refine < 5; ++refine) {
        float middle = 0.5 * (low + high);
        vec3 at = realtimeProject(position + inside * middle);
        vec4 back = realtimeOnScreen(at) ? texelFetch(uRealtimeBack, ivec2(at.xy * size), 0) : vec4(0.0);
        if (back.w <= 0.0 || at.z >= back.w) high = middle;
        else { low = middle; exitNormal = back.xyz; }
    }
    exitPoint = position + inside * high;
    exitNormal = normalize(exitNormal);
    return true;
}

// What lies along a ray leaving glass: the snapshot where the march meets a surface,
// else the environment in its direction. The taps blur a rough surface's hit.
vec3 realtimeBeyond(vec3 origin, vec3 direction, float blur, inout float angle) {
    vec3 environment = uAmbient;
    if (uSky[0].w > 0.5)
        environment += uSky[0].w > 1.5 ? environmentImageRadiance(direction, uSky[0], uSky[1])
            : analyticSkyRadiance(direction, uSky[0], uSky[1], uSky[2], uSky[3], uSky[4]);
    vec3 hit;
    if (!realtimeTraceHit(origin, direction, hit)) return environment;
    vec3 sum = vec3(0.0);
    for (int tap = 0; tap < 4; ++tap) {
        angle += kRealtimeGoldenAngle;
        float radius = blur * sqrt((float(tap) + 0.5) / 4.0);
        sum += texture(uRealtimeColor, hit.xy + vec2(cos(angle), sin(angle)) * radius).rgb;
    }
    return mix(environment, sum * 0.25, realtimeEdgeConfidence(hit.xy));
}

// One refracted direction per wavelength, each landing on the snapshot where the
// scene behind the glass lies. The depth gap sets how far the bent ray travels.
// A closed object with its back faces drawn refracts through both surfaces instead:
// in at this point, out where the ray meets the back face, and a ray the back face
// reflects totally leaves along the reflection.
vec3 realtimeRefraction(vec3 position, vec3 normal, vec3 view, float roughness) {
    float span = max(uProjection.w - uProjection.z, 1e-6);
    vec3 own = realtimeProject(position);
    if (own.x < 0.0) return vec3(0.0);
    if (uRealtimeControl.z > 0.5 && uGlassTransport.x <= 1.5
        && texelFetch(uRealtimeBack, ivec2(gl_FragCoord.xy), 0).w > own.z) {
        const vec3 thickWavelengths = vec3(0.6563, 0.5461, 0.4358);
        float thickBlur = roughness * roughness * 0.05;
        float thickAngle = realtimeNoise() * 6.28318531;
        vec3 thick = vec3(0.0);
        for (int channel = 0; channel < 3; ++channel) {
            float wavelength = thickWavelengths[channel];
            float ior = max(uSurfaceCoating.y + uGlassTransport.y * (1.0 / (wavelength * wavelength) - 1.0 / (0.58756 * 0.58756)), 1e-3);
            vec3 inside = refract(-view, normal, 1.0 / ior);
            if (dot(inside, inside) == 0.0) inside = -view;
            vec3 exitPoint, exitNormal;
            vec3 leaving = inside;
            if (realtimeBackExit(position, inside, exitPoint, exitNormal)) {
                vec3 facing = dot(inside, exitNormal) > 0.0 ? -exitNormal : exitNormal;
                leaving = refract(inside, facing, ior);
                if (dot(leaving, leaving) == 0.0) leaving = reflect(inside, facing);
            }
            thick[channel] = realtimeBeyond(exitPoint + leaving * 0.0005 * span, leaving, thickBlur, thickAngle)[channel];
        }
        return thick;
    }
    float ownDepth = texture(uRealtimeDepth, own.xy).r;
    bool surface = realtimeSurfaceDepth(ownDepth) && ownDepth > own.z;
    // Over the sky, nothing gives a gap, so the bend is judged at the camera's distance.
    float gap = surface ? (ownDepth - own.z) * span : own.z * span + uProjection.z;
    const vec3 wavelengths = vec3(0.6563, 0.5461, 0.4358);
    float blur = roughness * roughness * 0.05;
    float angle = realtimeNoise() * 6.28318531;
    vec3 result = vec3(0.0);
    for (int channel = 0; channel < 3; ++channel) {
        float wavelength = wavelengths[channel];
        float ior = uSurfaceCoating.y + uGlassTransport.y * (1.0 / (wavelength * wavelength) - 1.0 / (0.58756 * 0.58756));
        vec3 bent = uGlassTransport.x > 1.5 ? -view : refract(-view, normal, 1.0 / max(ior, 1e-3));
        if (dot(bent, bent) == 0.0) bent = reflect(-view, normal);
        vec3 target = realtimeProject(position + bent * gap);
        // A second read follows the ray to the depth where it lands.
        if (target.x >= 0.0) {
            float landed = texture(uRealtimeDepth, target.xy).r;
            if (realtimeSurfaceDepth(landed) && landed > own.z) {
                vec3 second = realtimeProject(position + bent * ((landed - own.z) * span));
                if (second.x >= 0.0) target = second;
            }
        } else target = own;
        float sum = 0.0;
        for (int tap = 0; tap < 4; ++tap) {
            angle += kRealtimeGoldenAngle;
            float radius = blur * sqrt((float(tap) + 0.5) / 4.0);
            sum += texture(uRealtimeColor, target.xy + vec2(cos(angle), sin(angle)) * radius)[channel];
        }
        result[channel] = sum * 0.25;
    }
    return result;
}

// What a thin water interface passes up from below. Path Trace gives the interface a
// medium under it that absorbs by the base colour per scene unit (kTintDistance in
// pathtrace_fixture_engine.cpp) and by the material's own medium, so the scene behind
// fades over its depth, and an open sea with nothing behind passes none of the sky
// below the horizon. What it shows instead is the light its body scatters back up:
// the sky above it, cosine weighted, times the share of the phase function that turns
// back, fading as both the light and the view ray go deeper. Lights other than the
// sky reach a viewer above the water only by the same back-scatter, and are left out.
vec3 realtimeWaterBody(vec3 position, vec3 normal, vec3 view, vec3 base, vec3 behind) {
    float span = max(uProjection.w - uProjection.z, 1e-6);
    vec3 own = realtimeProject(position);
    float ownDepth = own.x >= 0.0 ? texture(uRealtimeDepth, own.xy).r : 0.0;
    float depth = own.x >= 0.0 && realtimeSurfaceDepth(ownDepth) && ownDepth > own.z
        ? (ownDepth - own.z) * span : 1.0e30;
    vec3 sigmaT = -log(clamp(base, vec3(1.0e-4), vec3(1.0)));
    vec3 sigmaS = vec3(0.0);
    float g = 0.0;
    if (int(uMedium[0].w) == 1) {
        sigmaT += uMedium[0].xyz + uMedium[1].xyz;
        sigmaS = uMedium[1].xyz;
        g = uMedium[1].w;
    }
    vec3 transmitted = behind * exp(-sigmaT * depth);
    if (uSky[0].w <= 0.5 || dot(sigmaS, sigmaS) <= 0.0) return transmitted;
    const vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 sky = uSky[0].w > 1.5
        ? environmentImageBase(vec3(1.0), 0.0, 1.0, 0.0, up, up, uSky[0], uSky[1])
        : analyticSkyBase(vec3(1.0), 0.0, 1.0, 0.0, up, up, uSky[0], uSky[1], uSky[2], uSky[3], uSky[4]);
    // Henyey-Greenstein's mass in the backward hemisphere.
    float back = abs(g) < 1.0e-3 ? 0.5
        : (1.0 - g * g) / (2.0 * g) * (1.0 / sqrt(1.0 + g * g) - 1.0 / (1.0 + g));
    vec3 inside = refract(-view, normal, 1.0 / max(uSurfaceCoating.y, 1.0));
    float deepening = dot(inside, inside) > 0.0 ? max(-inside.y, 0.0) : 1.0;
    vec3 sigmaPath = sigmaT * (1.0 + deepening);
    vec3 body = sigmaS * back * sky * (vec3(1.0) - exp(-sigmaPath * depth)) / max(sigmaPath, vec3(1.0e-6));
    return transmitted + body;
}

// What a glass surface shows: the scene behind it, its own shaded layer where it
// does not transmit, and a Fresnel reflection of nearby objects and the environment.
vec3 realtimeGlass(vec3 position, vec3 shadedNormal, vec3 facet, vec3 toView, vec3 base, float roughness, float dielectric, vec3 shaded, vec3 surface) {
    vec3 view = normalize(toView);
    // Closed glass keeps the view's side; its silhouettes have the same turn.
    vec3 normal = uGlassTransport.x > 1.5 ? realtimeViewerNormal(shadedNormal, facet, view)
        : dot(shadedNormal, view) < 0.0 ? -shadedNormal : shadedNormal;
    float transmitted = uSurfaceCoating.w > 0.5 ? uSurfaceCoating.x : 1.0;
    vec3 fresnel = realtimeFresnel(vec3(dielectric), normal, view);
    vec3 reflected = realtimeEnvironment(normal, toView, roughness, dielectric);
    vec3 hit;
    float confidence;
    if (realtimeTrace(position + normal * 0.0005 * (uProjection.w - uProjection.z), reflect(-view, normal), hit, confidence))
        reflected = mix(reflected, hit, confidence);
    // A medium inside tints what it transmits over the refracted path through its bounds.
    // A thin water interface has its body below it instead.
    vec3 refracted = realtimeRefraction(position, normal, view, roughness);
    vec3 behind = uGlassTransport.x > 1.5 ? realtimeWaterBody(position, normal, view, base, refracted)
        : refracted * base * mediumBoxTint(position, refract(-view, normal, 1.0 / max(uSurfaceCoating.y, 1.0)));
    return mix(shaded, (vec3(1.0) - fresnel) * behind + surface, transmitted) + fresnel * reflected;
}

// How much light reaches the air between the camera and this surface, as a factor
// on the Depth Fog color: shadowed air goes dim, so objects cast visible shafts.
vec3 realtimeFogLight(vec3 position) {
    bool fogPass = false;
    for (int i = 0; i < int(uPassProgramControl.x); ++i)
        if (int(uPassModes[i].x) == 1) fogPass = true;
    if (!fogPass) return vec3(1.0);
    vec3 lit = vec3(0.0);
    float total = 0.0;
    float noise = realtimeNoise();
    for (int lightIndex = 0; lightIndex < uLightCount; ++lightIndex) {
        int kind = fixtureLightKind(lightIndex);
        if (kind == 2 || !fixtureLightInVolumes(lightIndex)) continue;
        vec4 light = fixtureLightColor(lightIndex);
        float weight = max(max(light.r, light.g), light.b) * light.a;
        total += weight * 8.0;
        for (int step = 0; step < 8; ++step) {
            vec3 air = mix(uCameraTranslation, position, (float(step) + noise) / 8.0);
            vec3 delta = fixtureLightPosition(lightIndex).xyz - air;
            vec3 toLight = kind == 0 ? rotateQ(fixtureLightRotation(lightIndex), vec3(0.0, 0.0, 1.0)) : normalize(delta);
            float cone = 1.0;
            if (kind == 3) {
                vec3 spot = rotateQ(fixtureLightRotation(lightIndex), vec3(0.0, 0.0, -1.0));
                cone = smoothstep(fixtureLightCone(lightIndex).y, fixtureLightCone(lightIndex).x, dot(spot, -toLight));
            }
            // An area light lights the air as a point with its emission profile.
            if (kind == 4) cone = fixtureAreaEmission(lightIndex, -toLight);
            if (cone <= 0.0) continue;
            float visible = !fixtureLightShadows(lightIndex) ? 1.0
                : opaqueVisibility(air, toLight, toLight, kind == 0 ? 3.402823e38 : length(delta), uOpaqueControl);
            lit += light.rgb / max(max(max(light.r, light.g), light.b), 1e-6) * weight * cone * visible;
        }
    }
    return total > 0.0 ? vec3(0.35) + 0.65 * lit / total : vec3(1.0);
}
)glsl";
}

// Depth of field for a camera with a lens: a gather over the finished colour and
// linear depth snapshots. Each tap counts when the centre lies inside the tap's
// own blur shape (disc or blade polygon, stretched by the aperture ratio), weighed
// by one over that shape's area, so out-of-focus points spread their light as a
// bokeh of the aperture's shape. Taps behind the centre use at most its blur, so a
// sharp foreground keeps its edges. The approximation sees only the visible
// surface: it cannot reveal what a blurred foreground hides.
// uLens: x blur radius in pixels per unit of |d - focus| / d, y focus distance,
// z near, w far. uAperture: x blades (0 round), y rotation, z ratio, w kernel
// radius in pixels.
inline const char* bokehFragmentShader() noexcept
{
    return R"glsl(#version 330 core
layout(location = 0) out vec4 outColor;
uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform vec4 uLens;
uniform vec4 uAperture;
const float kGoldenAngle = 2.39996323;
float viewDepth(vec2 pixel) {
    float stored = texelFetch(uDepth, ivec2(pixel), 0).r;
    return uLens.z + clamp(stored, 0.0, 1.0) * (uLens.w - uLens.z);
}
float blurRadius(float depth) { return uLens.x * abs(depth - uLens.y) / max(depth, 1e-6); }
// True when offset lies inside the aperture shape of circumradius radius.
bool insideAperture(vec2 offset, float radius) {
    if (radius <= 0.5) return dot(offset, offset) <= 0.25;
    float c = cos(-uAperture.y), s = sin(-uAperture.y);
    vec2 local = vec2(c * offset.x - s * offset.y, s * offset.x + c * offset.y);
    local.x /= uAperture.z;
    float r = length(local) / radius;
    if (uAperture.x < 3.0) return r <= 1.0;
    float wedge = 6.28318531 / uAperture.x;
    float angle = mod(atan(local.y, local.x), wedge) - 0.5 * wedge;
    return r * cos(angle) <= cos(0.5 * wedge);
}
float shapeArea(float radius) {
    float r = max(radius, 0.5);
    float n = uAperture.x;
    return uAperture.z * (n < 3.0 ? 3.14159265 * r * r : 0.5 * n * r * r * sin(6.28318531 / n));
}
void main() {
    vec2 pixel = gl_FragCoord.xy;
    vec2 size = vec2(textureSize(uColor, 0));
    float centreDepth = viewDepth(pixel);
    float centreRadius = blurRadius(centreDepth);
    vec4 centre = texelFetch(uColor, ivec2(pixel), 0);
    vec4 sum = centre / shapeArea(centreRadius);
    float weight = 1.0 / shapeArea(centreRadius);
    float spin = 6.28318531 * fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
    float kernel = uAperture.w * max(uAperture.z, 1.0);
    for (int i = 0; i < 64; ++i) {
        float angle = spin + float(i) * kGoldenAngle;
        vec2 offset = vec2(cos(angle), sin(angle)) * kernel * sqrt((float(i) + 0.5) / 64.0);
        vec2 tap = clamp(pixel + offset, vec2(0.5), size - 0.5);
        float tapDepth = viewDepth(tap);
        float radius = blurRadius(tapDepth);
        if (tapDepth > centreDepth) radius = min(radius, centreRadius);
        if (!insideAperture(-offset, radius)) continue;
        float w = 1.0 / shapeArea(radius);
        sum += texelFetch(uColor, ivec2(tap), 0) * w;
        weight += w;
    }
    outColor = sum / weight;
}
)glsl";
}

// Velocity motion blur for Raster and Realtime: a gather over the finished colour
// along the pixel's own screen motion. uMotion holds current minus previous
// position in pixels per frame, +Y up as gl_FragCoord. What a pixel shows at time
// t (frames from now) is what sits at pixel - motion * t now, so the shutter from
// uShutter.x to uShutter.y averages those taps. One tap per pixel crossed, never
// more than the image diagonal. The gather sees only each pixel's own motion: a
// moving object smears over itself, not over the still background behind it.
inline const char* velocityBlurFragmentShader() noexcept
{
    return R"glsl(#version 330 core
layout(location = 0) out vec4 outColor;
uniform sampler2D uColor;
uniform sampler2D uMotion;
uniform vec4 uShutter;
void main() {
    vec2 size = vec2(textureSize(uColor, 0));
    vec2 motion = texelFetch(uMotion, ivec2(gl_FragCoord.xy), 0).rg;
    float sweep = length(motion) * (uShutter.y - uShutter.x);
    int taps = int(clamp(ceil(sweep), 1.0, length(size)));
    vec4 sum = vec4(0.0);
    for (int i = 0; i < taps; ++i) {
        float t = mix(uShutter.x, uShutter.y, (float(i) + 0.5) / float(taps));
        vec2 at = clamp(gl_FragCoord.xy - motion * t, vec2(0.5), size - vec2(0.5));
        sum += texture(uColor, at / size);
    }
    outColor = sum / float(taps);
}
)glsl";
}
} // namespace arbitgpu::realtimeengine
