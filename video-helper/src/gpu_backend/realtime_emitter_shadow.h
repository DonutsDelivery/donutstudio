#pragma once

// Realtime's shadow on the room's light from its glowing triangles. Realtime lights a
// receiver from the emitter triangles by their vector irradiance (realtimeEmitterDirect),
// which has no occlusion, so the Diffraction Orb floated over its plate where Path Trace
// shows a contact shadow. Shadow rays compiled into the transport shader slowed every
// gather band about fivefold through register pressure, even with the ray itself compiled
// out, and the orb's gather bands, already 150-330 ms at 960x540, then ran past the
// driver's context switch timeout (Xid 109).
//
// So the rays run in a pass of their own over the reduced-resolution gather target
// (realtime_diffraction_gather.h), as the grating mirror does (realtime_diffraction_mirror.h).
// The gather round's receivers write their unshadowed emitter light, and in alpha the
// view depth of the point it was computed at, to the target's emission attachment
// instead of adding the light to the gather term. Each such texel rebuilds its world
// point from that depth, traces kRays rays to points on the triangles,
// each triangle picked in proportion to the irradiance it gives there, and adds the light
// times the share of rays that reach it to the texel's gather term. The full pass reads
// it back through the depth/normal-aware upsample, which blends neighbouring texels'
// points into a penumbra. A pixel that gathers itself in the full pass, as at a
// silhouette, keeps the unshadowed light.

#include <algorithm>
#include <cstdint>

namespace arbitgpu::realtimeemittershadow
{
// Rays per gather texel, at stratified picks over the triangles' irradiance.
inline constexpr int kRays = 4;
// Gather target rows per submission: at most 640 x 16 texels, 41k rays.
inline constexpr std::uint32_t kBandRows = 16;

inline std::uint32_t bandCount (std::uint32_t rows) noexcept
{
    return std::max<std::uint32_t> (1, (rows + kBandRows - 1) / kBandRows);
}

// The pass's fragment shader body. It follows "#version 330 core" and
// opaquevisibility::glslShader(), whose uOpaqueVisibility and uOpaqueControl the host
// sets to the frame's opaque visibility query, the one Realtime's direct light shadows
// with. The diagnostic REALTIME_ABLATE_EMITTERSHADOWQUERY traces the transport's
// all-surface closest-hit query instead, and with REALTIME_ABLATE_EMITTERSHADOWHIT
// writes the first ray's hit (object id mod 1000, status, distance) as the term;
// REALTIME_ABLATE_EMITTERSHADOWWORLD writes the rebuilt world point and
// REALTIME_ABLATE_EMITTERSHADOWNORMAL the texel's normal.
inline const char* fragmentShader() noexcept
{
    return R"glsl(
layout(location = 0) out vec4 outColor;
uniform sampler2D uShadowNormal;
uniform usampler2D uShadowObjectIds;
uniform sampler2D uShadowLight;
uniform vec2 uShadowExtent;
uniform vec4 uCameraRotation;
uniform vec3 uCameraTranslation;
uniform vec4 uProjection;
uniform sampler2D uSceneTable;
uniform int uShadowEmitterFirst;
uniform int uShadowEmitterTriangles;
const int kShadowRays = SHADOW_RAYS;
vec3 shadowRotate(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
vec4 shadowTable(int record) { return texelFetch(uSceneTable, ivec2(record % 1024, record / 1024), 0); }
vec3 shadowEdge(vec3 p, vec3 q) {
    vec3 n = cross(p, q);
    float length2 = dot(n, n);
    return length2 > 1.0e-24 ? 0.5 * acos(clamp(dot(p, q), -1.0, 1.0)) * n * inversesqrt(length2) : vec3(0.0);
}
// Triangle t's luminance-weighted irradiance at position on the side normal faces, by
// its vector irradiance as realtimeEmitterDirect has it; 0 where it does not light it.
float shadowWeight(int t, vec3 position, vec3 normal) {
    int record = uShadowEmitterFirst + 3 * t;
    vec4 a = shadowTable(record), b = shadowTable(record + 1), c = shadowTable(record + 2);
    vec3 A = a.xyz - position, B = b.xyz - position, C = c.xyz - position;
    vec3 plane = cross(B - A, C - A);
    float planeLength = length(plane);
    if (!(planeLength > 0.0) || abs(dot(plane, A)) <= 1.0e-6 * planeLength) return 0.0;
    vec3 va = normalize(A), vb = normalize(B), vc = normalize(C);
    vec3 vector = shadowEdge(va, vb) + shadowEdge(vb, vc) + shadowEdge(vc, va);
    if (dot(vector, va + vb + vc) < 0.0) vector = -vector;
    return max(dot(normal, vector), 0.0) * dot(vec3(a.w, b.w, c.w), vec3(0.2126, 0.7152, 0.0722));
}
// Interleaved gradient noise at a pixel shifted by offset.
float shadowNoise(vec2 offset) {
    return fract(52.9829189 * fract(dot(gl_FragCoord.xy + offset, vec2(0.06711056, 0.00583715))));
}
void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    vec4 light = texelFetch(uShadowLight, texel, 0);
    if (!(light.a > 0.0) || !(max(light.r, max(light.g, light.b)) > 0.0)) discard;
    // A texel the pass cannot trace from keeps its light unshadowed.
    outColor = vec4(light.rgb, 0.0);
    uint self = texelFetch(uShadowObjectIds, texel, 0).r;
    vec3 normal = texelFetch(uShadowNormal, texel, 0).xyz;
    if (self == 0u || !(dot(normal, normal) > 1.0e-12)) return;
    normal = normalize(normal);
    // The texel's world point on its camera ray, at the view depth the receiver wrote
    // in alpha beside its light (realtime_diffraction_mirror.h gatherWorld has the ray).
    vec2 ndc = gl_FragCoord.xy / uShadowExtent * 2.0 - 1.0;
    float distance = light.a;
    vec3 camera = uProjection.x < 0.0
        ? vec3(ndc.x * -uProjection.x * uProjection.y, ndc.y * -uProjection.x, -distance)
        : vec3(ndc.x * uProjection.x * uProjection.y * distance, ndc.y * uProjection.x * distance, -distance);
    vec3 world = shadowRotate(uCameraRotation, camera) + uCameraTranslation;
#ifdef REALTIME_ABLATE_EMITTERSHADOWWORLD
    outColor = vec4(world, 0.0);
    return;
#endif
#ifdef REALTIME_ABLATE_EMITTERSHADOWNORMAL
    outColor = vec4(normal, 0.0);
    return;
#endif
    float total = 0.0;
    #pragma optionNV(unroll none)
    for (int t = 0; t < uShadowEmitterTriangles; ++t) total += shadowWeight(t, world, normal);
    if (!(total > 0.0)) return;
    float first = shadowNoise(vec2(0.0)), across = shadowNoise(vec2(17.0, 31.0));
    float seen = 0.0;
#ifdef REALTIME_ABLATE_EMITTERSHADOW
    seen = float(kShadowRays);
#else
#ifdef REALTIME_ABLATE_EMITTERSHADOWQUERY
    float bias = max(1.0e-4, 1.0e-4 * max(max(abs(world.x), abs(world.y)), abs(world.z)));
    vec3 origin = world + normal * bias;
#ifdef REALTIME_ABLATE_EMITTERSHADOWHIT
    vec3 probe = vec3(-1.0);
#endif
#endif
    #pragma optionNV(unroll none)
    for (int ray = 0; ray < kShadowRays; ++ray) {
        // Stratified picks over the triangles' summed weight.
        float pick = (float(ray) + first) / float(kShadowRays) * total, running = 0.0;
        int chosen = -1;
        float u = 0.0;
        #pragma optionNV(unroll none)
        for (int t = 0; t < uShadowEmitterTriangles; ++t) {
            float weight = shadowWeight(t, world, normal);
            if (!(weight > 0.0)) continue;
            chosen = t;
            u = clamp((pick - running) / weight, 0.0, 1.0);
            running += weight;
            if (running >= pick) break;
        }
        if (chosen < 0) { seen += 1.0; continue; }
        // A point on the triangle: where the pick fell in its share, and a second number
        // that moves with the ray, folded into the triangle.
        vec2 r = vec2(u, fract(across + 0.6180340 * float(ray)));
        if (r.x + r.y > 1.0) r = 1.0 - r;
        int record = uShadowEmitterFirst + 3 * chosen;
        vec3 a = shadowTable(record).xyz, b = shadowTable(record + 1).xyz, c = shadowTable(record + 2).xyz;
        vec3 target = a + r.x * (b - a) + r.y * (c - a);
#ifdef REALTIME_ABLATE_EMITTERSHADOWQUERY
        vec3 delta = target - origin;
        float reach = length(delta);
        if (!(reach > 0.0)) { seen += 1.0; continue; }
        SceneClosestHit hit = sceneClosestHit(origin, delta / reach, 0.0, reach * (1.0 - 1.0e-3), uOpaqueControl,
            uint(0), self);
#ifdef REALTIME_ABLATE_EMITTERSHADOWHIT
        if (ray == 0) probe = vec3(float(hit.objectId % 1000u), float(hit.status), hit.distance);
#endif
        seen += hit.status == 1 ? 0.0 : 1.0;
#else
        // Any opaque surface short of the emitter shadows it; transmissive ones are not
        // in the opaque frame, so their light passes as Path Trace's does.
        vec3 delta = target - world;
        float reach = length(delta);
        if (!(reach > 0.0)) { seen += 1.0; continue; }
        seen += opaqueVisibility(world, normal, delta, reach * (1.0 - 1.0e-3), uOpaqueControl);
#endif
    }
#ifdef REALTIME_ABLATE_EMITTERSHADOWHIT
    outColor = vec4(probe, 0.0);
    return;
#endif
#endif
    vec3 added = light.rgb * (seen / float(kShadowRays));
    if (!any(isnan(added)) && !any(isinf(added))) outColor = vec4(added, 0.0);
}
)glsl";
}
} // namespace arbitgpu::realtimeemittershadow
