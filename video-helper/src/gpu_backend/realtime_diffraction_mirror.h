#pragma once

// Realtime's grating mirror: the zero order's reflection of the lit room. Realtime's
// orders read only the sky and the glowing triangles (realtimeDiffractionSky), so a
// mirror ray that meets the floor, the plate or a wall read the dark ambient sky, and
// the orb's lower hemisphere and rim stayed black where Path Trace shows the room.
//
// A pass of its own runs over the reduced-resolution gather target after the grating's
// gather bands (realtime_diffraction_gather.h). Each grating texel rebuilds its world
// point from the gather depth and normal, traces the mirror ray to the closest opaque
// hit in the transport's scene query, shades the hit with its base colour under the
// glowing triangles' unshadowed light and the ambient, and adds it, times the zero
// order's weight, to the texel's gather term. The full pass then reads it back through
// the existing depth/normal-aware upsample. A screen-space march compiled into the
// transport shader cost the 1080p orb about 250 ms through register pressure alone.
//
// The zero order's efficiency depends only on the incident polar angle: the order
// leaves at the incident z, so the lattice phase is 2 pi depth 2 z / lambda for any
// azimuth. The weight per band is efficiency * exp(-h^2) / max(sum, 1) * reflectance,
// as diffractionTriangleBsdf has it, tabulated here over cos(theta) from the grating's
// efficiency table. The band sum is read along the tangent axis; it is the one part
// that depends on azimuth, and it only normalises.

#include "realtime_diffraction_efficiency_table.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace arbitgpu::realtimediffractionmirror
{
// Samples of the weight over cos(theta) in [0, 1], interpolated linearly.
inline constexpr int kWeightSamples = 33;
using Texel = std::array<float, 4>;

// The table's bilinear read at incident tangent xy, as realtimeOrderEfficiency reads it.
inline float tableRead (const realtimediffractionefficiency::Table& table, int tile, float x, float y) noexcept
{
    namespace eff = realtimediffractionefficiency;
    const float grid = float (eff::kGrid - 1);
    const float nx = std::clamp ((x * 0.5f + 0.5f) * grid, 0.0f, grid);
    const float ny = std::clamp ((y * 0.5f + 0.5f) * grid, 0.0f, grid);
    const int x0 = std::min (int (nx), eff::kGrid - 2), y0 = std::min (int (ny), eff::kGrid - 2);
    const float fx = nx - float (x0), fy = ny - float (y0);
    const int originX = (tile % eff::kTilesPerRow) * eff::kGrid, originY = (tile / eff::kTilesPerRow) * eff::kGrid;
    const auto at = [&] (int i, int j) {
        const std::size_t index = std::size_t (originY + j) * std::size_t (table.width) + std::size_t (originX + i);
        return index < table.texels.size() ? table.texels[index] : 0.0f;
    };
    return (1.0f - fy) * ((1.0f - fx) * at (x0, y0) + fx * at (x0 + 1, y0))
         + fy * ((1.0f - fx) * at (x0, y0 + 1) + fx * at (x0 + 1, y0 + 1));
}

// The shader's spectralReflectance (backend_opengl.cpp), -1 for an unknown coating model.
inline float spectralReflectance (const diffractionmaterial::PhysicalDiffractionGpuParameters& m, float wavelength) noexcept
{
    const auto fresnel = [] (float n1, float k1, float n2, float k2) {
        const float dn = n1 - n2, dk = k1 - k2, sn = n1 + n2, sk = k1 + k2;
        const float denominator = sn * sn + sk * sk;
        return denominator > 0.0f ? std::clamp ((dn * dn + dk * dk) / denominator, 0.0f, 1.0f) : 0.0f;
    };
    const int model = int (m.control.w + 0.5f);
    const float substrateN = m.microstructure.y, substrateK = m.microstructure.z;
    if (model == 1) return fresnel (1.0f, 0.0f, substrateN, substrateK);
    if (model != 2) return -1.0f;
    const float r01 = fresnel (1.0f, 0.0f, m.coating.y, m.coating.z);
    const float r12 = fresnel (m.coating.y, m.coating.z, substrateN, substrateK);
    const float roundTrip = std::exp (-8.0f * 3.14159265358979f * m.coating.z * m.coating.x / wavelength);
    const float denominator = 1.0f - r01 * r12 * roundTrip;
    return std::clamp (r01 + (1.0f - r01) * (1.0f - r01) * r12 * roundTrip / denominator, 0.0f, 1.0f);
}

// The shader's diffractionBandsToRgb.
inline std::array<float, 3> bandsToRgb (const diffractionmaterial::PhysicalDiffractionGpuParameters& m,
                                        const std::array<float, 8>& bands) noexcept
{
    float x = 0.0f, y = 0.0f, z = 0.0f, whiteY = 0.0f;
    for (std::size_t b = 0; b < 8; ++b)
    {
        const auto& spectral = m.spectral[b];
        const auto& lane = m.spectralZ[b];
        x += bands[b] * lane.y * spectral.z;
        y += bands[b] * lane.y * spectral.w;
        z += bands[b] * lane.y * lane.x;
        whiteY += lane.y * spectral.w;
    }
    const float scale = 1.0f / std::max (whiteY, 1.0e-8f);
    x *= scale; y *= scale; z *= scale;
    return { std::max (3.2406f * x - 1.5372f * y - 0.4986f * z, 0.0f),
             std::max (-0.9689f * x + 1.8758f * y + 0.0415f * z, 0.0f),
             std::max (0.0557f * x - 0.2040f * y + 1.0570f * z, 0.0f) };
}

// The zero order's RGB weight at kWeightSamples values of cos(theta) from 0 to 1.
inline std::vector<Texel> weights (const realtimediffractionefficiency::Table& table,
                                   const diffractionmaterial::PhysicalDiffractionGpuParameters& m)
{
    namespace eff = realtimediffractionefficiency;
    std::vector<Texel> out (std::size_t (kWeightSamples), Texel {});
    const int zero = table.crossed ? table.lastOrder * (2 * table.lastOrder + 1) + table.lastOrder : table.lastOrder;
    for (int s = 0; s < kWeightSamples; ++s)
    {
        const float c = float (s) / float (kWeightSamples - 1);
        const float x = std::sqrt (std::max (0.0f, 1.0f - c * c));
        std::array<float, 8> bands {};
        for (int b = 0; b < 8; ++b)
        {
            const float wavelength = m.spectral[std::size_t (b)].x;
            const float reflectance = spectralReflectance (m, wavelength);
            if (! (wavelength > 0.0f) || ! (reflectance > 0.0f)) continue;
            const float height = 2.0f * 3.14159265358979f * m.roughness.x * 2.0f * c / wavelength;
            const float total = tableRead (table, eff::tileIndex (b, table.orders, table.orders), x, 0.0f);
            const float weight = tableRead (table, eff::tileIndex (b, zero, table.orders), x, 0.0f)
                * std::exp (-height * height) / std::max (total, 1.0f) * reflectance;
            bands[std::size_t (b)] = std::isfinite (weight) ? std::clamp (weight, 0.0f, 1.0f) : 0.0f;
        }
        const auto rgb = bandsToRgb (m, bands);
        out[std::size_t (s)] = { rgb[0], rgb[1], rgb[2], 0.0f };
    }
    return out;
}

// One scene object as the mirror pass shades it: its id, metallic and whether it glows,
// then its base colour. A glowing object is already in the orders' emitter light.
inline void appendObject (std::vector<Texel>& records, std::uint32_t objectId,
                          const std::array<float, 3>& base, float metallic, bool glows)
{
    records.push_back ({ float (objectId & 65535u), float (objectId >> 16u),
                         std::clamp (std::isfinite (metallic) ? metallic : 0.0f, 0.0f, 1.0f), glows ? 1.0f : 0.0f });
    const auto lane = [] (float v) { return std::isfinite (v) ? std::max (v, 0.0f) : 0.0f; };
    records.push_back ({ lane (base[0]), lane (base[1]), lane (base[2]), 0.0f });
}

// The world point of a gather texel: its linear depth (vLinearDepth over the authored
// planes) along the camera ray through the texel's NDC. rotation is the camera's
// quaternion (camera to world), projection uProjection.
inline std::array<float, 3> gatherWorld (const std::array<float, 4>& rotation, const std::array<float, 3>& translation,
                                         const std::array<float, 4>& projection, float ndcX, float ndcY, float linearDepth) noexcept
{
    const float distance = projection[2] + linearDepth * (projection[3] - projection[2]);
    std::array<float, 3> v = projection[0] < 0.0f
        ? std::array<float, 3> { ndcX * -projection[0] * projection[1], ndcY * -projection[0], -distance }
        : std::array<float, 3> { ndcX * projection[0] * projection[1] * distance, ndcY * projection[0] * distance, -distance };
    const auto cross = [] (const std::array<float, 3>& a, const std::array<float, 3>& b) {
        return std::array<float, 3> { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
    };
    const std::array<float, 3> q { rotation[0], rotation[1], rotation[2] };
    auto inner = cross (q, v);
    for (int i = 0; i < 3; ++i) inner[std::size_t (i)] += rotation[3] * v[std::size_t (i)];
    const auto outer = cross (q, inner);
    return { v[0] + 2.0f * outer[0] + translation[0], v[1] + 2.0f * outer[1] + translation[1],
             v[2] + 2.0f * outer[2] + translation[2] };
}

// The pass's fragment shader body. It follows "#version 330 core" and
// opaquevisibility::glslShader(), whose uOpaqueVisibility and uOpaqueControl the host
// sets to the transport's scene query, which holds closest-hit records.
inline const char* fragmentShader() noexcept
{
    return R"glsl(
layout(location = 0) out vec4 outColor;
uniform sampler2D uMirrorDepth;
uniform sampler2D uMirrorNormal;
uniform usampler2D uMirrorObjectIds;
uniform uint uMirrorGrating;
uniform vec2 uMirrorExtent;
uniform vec4 uCameraRotation;
uniform vec3 uCameraTranslation;
uniform vec4 uProjection;
uniform vec4 uMirrorWeight[33];
uniform sampler2D uMirrorObjects;
uniform int uMirrorObjectCount;
uniform sampler2D uSceneTable;
uniform int uMirrorEmitterFirst;
uniform int uMirrorEmitterTriangles;
uniform vec3 uAmbient;
vec3 mirrorRotate(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
vec4 mirrorTable(sampler2D table, int record) { return texelFetch(table, ivec2(record % 1024, record / 1024), 0); }
vec3 mirrorEdge(vec3 p, vec3 q) {
    vec3 n = cross(p, q);
    float length2 = dot(n, n);
    return length2 > 1.0e-24 ? 0.5 * acos(clamp(dot(p, q), -1.0, 1.0)) * n * inversesqrt(length2) : vec3(0.0);
}
// The glowing triangles' unshadowed irradiance at a point, by each triangle's vector
// irradiance, as realtimeEmitterDirect lights the room's surfaces.
vec3 mirrorEmitterIrradiance(vec3 position, vec3 normal) {
    vec3 sum = vec3(0.0);
    #pragma optionNV(unroll none)
    for (int t = 0; t < uMirrorEmitterTriangles; ++t) {
        int record = uMirrorEmitterFirst + 3 * t;
        vec4 a = mirrorTable(uSceneTable, record), b = mirrorTable(uSceneTable, record + 1), c = mirrorTable(uSceneTable, record + 2);
        vec3 A = a.xyz - position, B = b.xyz - position, C = c.xyz - position;
        vec3 plane = cross(B - A, C - A);
        float planeLength = length(plane);
        if (!(planeLength > 0.0) || abs(dot(plane, A)) <= 1.0e-6 * planeLength) continue;
        vec3 va = normalize(A), vb = normalize(B), vc = normalize(C);
        vec3 vector = mirrorEdge(va, vb) + mirrorEdge(vb, vc) + mirrorEdge(vc, va);
        if (dot(vector, va + vb + vc) < 0.0) vector = -vector;
        sum += vec3(a.w, b.w, c.w) * max(dot(normal, vector), 0.0);
    }
    return sum;
}
void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    if (texelFetch(uMirrorObjectIds, texel, 0).r != uMirrorGrating) discard;
    vec3 normal = texelFetch(uMirrorNormal, texel, 0).xyz;
    if (!(dot(normal, normal) > 1.0e-12)) discard;
    normal = normalize(normal);
    // The texel's world point, as realtime_diffraction_mirror.h gatherWorld rebuilds it.
    vec2 ndc = gl_FragCoord.xy / uMirrorExtent * 2.0 - 1.0;
    float distance = uProjection.z + texelFetch(uMirrorDepth, texel, 0).r * (uProjection.w - uProjection.z);
    vec3 camera = uProjection.x < 0.0
        ? vec3(ndc.x * -uProjection.x * uProjection.y, ndc.y * -uProjection.x, -distance)
        : vec3(ndc.x * uProjection.x * uProjection.y * distance, ndc.y * uProjection.x * distance, -distance);
    vec3 world = mirrorRotate(uCameraRotation, camera) + uCameraTranslation;
    vec3 back = mirrorRotate(uCameraRotation, vec3(0.0, 0.0, 1.0));
    vec3 toView = uProjection.x >= 0.0 ? normalize(uCameraTranslation - world) : back;
    float cosine = dot(normal, toView);
    if (!(cosine > 0.0)) discard;
    float at = clamp(cosine, 0.0, 1.0) * 32.0;
    int low = min(int(at), 31);
    vec3 weight = mix(uMirrorWeight[low].rgb, uMirrorWeight[low + 1].rgb, at - float(low));
    if (!(max(weight.r, max(weight.g, weight.b)) > 1.0e-4)) discard;
    vec3 direction = reflect(-toView, normal);
    float bias = max(1.0e-4, 1.0e-4 * max(max(abs(world.x), abs(world.y)), abs(world.z)));
    SceneClosestHit hit = sceneClosestHit(world + normal * bias, direction, 0.0, 3.402823e38, uOpaqueControl,
        uint(0), uMirrorGrating);
    // A miss sees the sky and a glowing hit an emitter, which the orders already read.
    if (hit.status != 1) discard;
    int found = -1;
    #pragma optionNV(unroll none)
    for (int i = 0; i < uMirrorObjectCount; ++i) {
        vec4 id = mirrorTable(uMirrorObjects, 2 * i);
        if (uint(id.x) + uint(id.y) * 65536u == hit.objectId) { found = i; break; }
    }
    if (found < 0) discard;
    vec4 record = mirrorTable(uMirrorObjects, 2 * found);
    if (record.w > 0.5) discard;
    vec3 base = mirrorTable(uMirrorObjects, 2 * found + 1).rgb;
    vec3 surfaceNormal = dot(hit.shadingNormal, direction) > 0.0 ? -hit.shadingNormal : hit.shadingNormal;
    vec3 radiance = base * ((1.0 - record.z) * mirrorEmitterIrradiance(hit.position, surfaceNormal)
        / 3.14159265358979 + uAmbient);
    vec3 added = weight * radiance;
    if (any(isnan(added)) || any(isinf(added))) discard;
    outColor = vec4(added, 0.0);
}
)glsl";
}
} // namespace arbitgpu::realtimediffractionmirror
