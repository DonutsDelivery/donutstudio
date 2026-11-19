#pragma once

// Material media in Raster and Realtime, which have no media walk.
//
// A glass or water medium tints the light it transmits by Beer-Lambert
// absorption over the refracted path length: the exit of the refracted camera
// ray, from the glass query where the raster has one, else from the object's
// world bounds. A Volume Material object draws no surface. Its draw raymarches
// a homogeneous or noise volume inside the object's world bounds, single
// scattering the scene lights with the raster's shadow visibility, plus the
// ambient light and the medium's emission. The coefficients are Path Trace's:
// scattering density * colour and absorption density * (1 - colour) *
// (1 - absorption colour), both scaled by the density field, and the noise
// field is Path Trace's volumeFieldAt.
//
// The draw's medium is uMedium[10]:
//   [0] absorption rgb, mode (0 none, 1 tint, 2 volume)
//   [1] scattering rgb, anisotropy
//   [2] emission rgb, field (0 uniform, 1 noise)
//   [3] field offset xyz, field scale
//   [4] bounds minimum xyz, field detail
//   [5] bounds maximum xyz, 0
//   [6] the paired walk's partner dielectric absorption rgb, 0
//   [7] the paired walk's water body up axis xyz, its floor along up
//   [8] the water body's first lateral axis xyz, its surface along up
//   [9] the water body's extent along the first and second lateral axes:
//       minimum, maximum, minimum, maximum. A zero up axis means no body.
//
// World fog. Raster and Realtime march the scene fog (the Render node's or the
// scene's world volume) from the camera to each surface, and to the far plane
// behind the sky, single scattering every light with the same shadow
// visibility, so lit fog shows light shafts where the scene shadows it. It
// replaces the Depth Fog approximation. Its rows are uWorldFog[5], laid out as
// rows 0 to 4 of a medium: [0].w is 1 when the fog is on and [4].x is the far
// distance a sky pixel marches.

#include "Visual3DScene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>

namespace rastermedium
{
inline constexpr int kMarchSteps = 32;

enum class Mode : int
{
    None = 0,
    Tint = 1,
    Volume = 2
};

using Uniforms = std::array<std::array<float, 4>, 10>;
using Rgb = std::array<float, 3>;

inline float unit (float x) noexcept { return std::isfinite (x) ? std::min (1.0f, std::max (0.0f, x)) : 0.0f; }

inline Mode modeOf (const HarmonicMIDI::grid::SceneVolume& volume) noexcept
{
    if (! HarmonicMIDI::grid::sceneVolumeActive (volume)) return Mode::None;
    return volume.volumeOnly ? Mode::Volume : Mode::Tint;
}

inline Rgb absorption (const HarmonicMIDI::grid::SceneVolume& v) noexcept
{
    if (! HarmonicMIDI::grid::sceneVolumeActive (v)) return {};
    return { v.density * (1.0f - unit (v.color.x)) * (1.0f - unit (v.absorptionColor.x)),
             v.density * (1.0f - unit (v.color.y)) * (1.0f - unit (v.absorptionColor.y)),
             v.density * (1.0f - unit (v.color.z)) * (1.0f - unit (v.absorptionColor.z)) };
}

inline Rgb scattering (const HarmonicMIDI::grid::SceneVolume& v) noexcept
{
    if (! HarmonicMIDI::grid::sceneVolumeActive (v)) return {};
    return { v.density * unit (v.color.x), v.density * unit (v.color.y), v.density * unit (v.color.z) };
}

// Bounds are the object's world box; mode None packs zeros.
inline Uniforms pack (const HarmonicMIDI::grid::SceneVolume& v, HarmonicMIDI::grid::SceneVec3 low,
                      HarmonicMIDI::grid::SceneVec3 high) noexcept
{
    Uniforms u {};
    const auto mode = modeOf (v);
    if (mode == Mode::None) return u;
    const auto a = absorption (v), s = scattering (v);
    const auto finite = [] (float x) { return std::isfinite (x) ? x : 0.0f; };
    u[0] = { a[0], a[1], a[2], float (static_cast<int> (mode)) };
    u[1] = { s[0], s[1], s[2], std::min (0.99f, std::max (-0.99f, finite (v.anisotropy))) };
    u[2] = { std::max (0.0f, finite (v.emission.x)), std::max (0.0f, finite (v.emission.y)),
             std::max (0.0f, finite (v.emission.z)), v.field == HarmonicMIDI::grid::SceneVolumeField::Noise ? 1.0f : 0.0f };
    u[3] = { finite (v.fieldOffset.x), finite (v.fieldOffset.y), finite (v.fieldOffset.z),
             std::isfinite (v.fieldScale) && v.fieldScale > 0.0f ? v.fieldScale : 1.0f };
    u[4] = { low.x, low.y, low.z, float (std::min<std::uint32_t> (8u, std::max<std::uint32_t> (1u, v.fieldDetail))) };
    u[5] = { high.x, high.y, high.z, 0.0f };
    return u;
}

// The world box of an object's vertices under its column-major world matrix.
inline void worldBounds (const HarmonicMIDI::grid::Visual3DScene& scene, const HarmonicMIDI::grid::SceneObjectRecord& object,
                         const std::array<float, 16>& m, HarmonicMIDI::grid::SceneVec3& low,
                         HarmonicMIDI::grid::SceneVec3& high) noexcept
{
    low = { 3.402823e38f, 3.402823e38f, 3.402823e38f };
    high = { -3.402823e38f, -3.402823e38f, -3.402823e38f };
    const std::size_t first = object.firstVertex;
    const std::size_t end = std::min<std::size_t> (std::min<std::size_t> (scene.vertexCount, scene.vertices.size()),
                                                   first + object.vertexCount);
    for (std::size_t index = first; index < end; ++index)
    {
        const auto& p = scene.vertices[index].position;
        const float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
        const float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
        const float z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
        low = { std::min (low.x, x), std::min (low.y, y), std::min (low.z, z) };
        high = { std::max (high.x, x), std::max (high.y, y), std::max (high.z, z) };
    }
    if (! (low.x <= high.x)) low = high = {};
}

// Grows low and high to hold another box; an empty box (low above high) adds nothing.
inline void unite (HarmonicMIDI::grid::SceneVec3& low, HarmonicMIDI::grid::SceneVec3& high,
                   HarmonicMIDI::grid::SceneVec3 otherLow, HarmonicMIDI::grid::SceneVec3 otherHigh) noexcept
{
    if (! (otherLow.x <= otherHigh.x)) return;
    low = { std::min (low.x, otherLow.x), std::min (low.y, otherLow.y), std::min (low.z, otherLow.z) };
    high = { std::max (high.x, otherHigh.x), std::max (high.y, otherHigh.y), std::max (high.z, otherHigh.z) };
}

// Row 6: the absorption of the paired walk's partner dielectric (the water of a
// caustic volume draw, or the volume of a water draw), named by its object id.
// The paired walk tints each segment in either body by that body's own medium.
inline void packPartner (Uniforms& u, const HarmonicMIDI::grid::Visual3DScene& scene,
                         HarmonicMIDI::grid::SceneObjectId partner) noexcept
{
    u[6] = {};
    if (! partner.isValid()) return;
    for (std::size_t object = 0; object < scene.objectCount; ++object)
    {
        if (scene.objects[object].id != partner) continue;
        for (std::size_t material = 0; material < scene.materialCount; ++material)
            if (scene.materials[material].id == scene.objects[object].material
                && modeOf (scene.materials[material].volume) == Mode::Tint)
            {
                const auto a = absorption (scene.materials[material].volume);
                u[6] = { a[0], a[1], a[2], 0.0f };
            }
        return;
    }
}
// A dielectric's base colour as absorption per scene unit (smoothglass::baseAbsorption),
// added to the draw's own medium (row 0, which turns the tint on) or to its partner's
// (row 6). A Volume Material draw marches its own medium and takes none.
inline void addDielectricTint (Uniforms& u, const Rgb& own, const Rgb& partner) noexcept
{
    if ((own[0] != 0.0f || own[1] != 0.0f || own[2] != 0.0f)
        && u[0][3] != float (static_cast<int> (Mode::Volume)))
    {
        for (std::size_t c = 0; c < 3; ++c) u[0][c] += own[c];
        u[0][3] = float (static_cast<int> (Mode::Tint));
    }
    for (std::size_t c = 0; c < 3; ++c) u[6][c] += partner[c];
}
// Rows 7 to 9: the body of the paired walk's water, for a branch that leaves
// the scene below the water surface. The water is a thin surface, so its body
// is an oriented box: the water's footprint across its up axis, from its
// highest point down to the scene's lowest point along that axis. Up is the
// water's mean authored normal in the world, else its local +Y.
// worldMatrix (object) is the object's column-major world matrix, with any
// runtime transform the frame draws it under.
template <class WorldMatrix>
inline void packWaterBody (Uniforms& u, const HarmonicMIDI::grid::Visual3DScene& scene,
                           HarmonicMIDI::grid::SceneObjectId water, WorldMatrix&& worldMatrix)
{
    u[7] = {};
    u[8] = {};
    u[9] = {};
    if (! water.isValid()) return;
    const HarmonicMIDI::grid::SceneObjectRecord* body = nullptr;
    for (std::size_t index = 0; index < scene.objectCount; ++index)
        if (scene.objects[index].id == water && scene.objects[index].vertexCount != 0)
            body = &scene.objects[index];
    if (body == nullptr) return;
    const std::size_t vertexEnd = std::min<std::size_t> (scene.vertexCount, scene.vertices.size());
    const auto range = [&] (const HarmonicMIDI::grid::SceneObjectRecord& object, std::size_t& first, std::size_t& end) {
        first = object.firstVertex;
        end = std::min<std::size_t> (vertexEnd, first + object.vertexCount);
    };
    using V = std::array<float, 3>;
    const auto dot = [] (const V& a, const V& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const auto cross = [] (const V& a, const V& b) {
        return V { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
    };
    const auto normalized = [&] (V v, V& out) {
        const float length = std::sqrt (dot (v, v));
        if (! (length > 1.0e-8f) || ! std::isfinite (length)) return false;
        out = { v[0] / length, v[1] / length, v[2] / length };
        return true;
    };
    const std::array<float, 16> m = worldMatrix (*body);
    const V column0 { m[0], m[1], m[2] }, column1 { m[4], m[5], m[6] }, column2 { m[8], m[9], m[10] };
    // Normals take the cofactor matrix, which keeps their side under any scale.
    const V row0 = cross (column1, column2), row1 = cross (column2, column0), row2 = cross (column0, column1);
    const float handedness = dot (column0, row0) < 0.0f ? -1.0f : 1.0f;
    V normalSum {};
    std::size_t first = 0, end = 0;
    range (*body, first, end);
    for (std::size_t index = first; index < end; ++index)
    {
        const auto& n = scene.vertices[index].normal;
        for (int c = 0; c < 3; ++c)
            normalSum[std::size_t (c)] += handedness * (row0[std::size_t (c)] * n.x + row1[std::size_t (c)] * n.y + row2[std::size_t (c)] * n.z);
    }
    V up {}, lateral {};
    if (! normalized (normalSum, up) && ! normalized (column1, up)) up = { 0.0f, 1.0f, 0.0f };
    // The first lateral axis is the water's local +X across up, else a world axis.
    const auto across = [&] (const V& v) {
        const float along = dot (v, up);
        return V { v[0] - along * up[0], v[1] - along * up[1], v[2] - along * up[2] };
    };
    if (! normalized (across (column0), lateral) && ! normalized (across (V { 1.0f, 0.0f, 0.0f }), lateral))
        normalized (across (V { 0.0f, 0.0f, 1.0f }), lateral);
    const V second = cross (up, lateral);
    const auto world = [] (const std::array<float, 16>& w, const HarmonicMIDI::grid::SceneVec3& p) {
        return V { w[0] * p.x + w[4] * p.y + w[8] * p.z + w[12],
                   w[1] * p.x + w[5] * p.y + w[9] * p.z + w[13],
                   w[2] * p.x + w[6] * p.y + w[10] * p.z + w[14] };
    };
    constexpr float big = 3.402823e38f;
    float surface = -big, low1 = big, high1 = -big, low2 = big, high2 = -big;
    for (std::size_t index = first; index < end; ++index)
    {
        const auto p = world (m, scene.vertices[index].position);
        surface = std::max (surface, dot (p, up));
        low1 = std::min (low1, dot (p, lateral));
        high1 = std::max (high1, dot (p, lateral));
        low2 = std::min (low2, dot (p, second));
        high2 = std::max (high2, dot (p, second));
    }
    float floor = surface;
    for (std::size_t object = 0; object < scene.objectCount; ++object)
    {
        if (scene.objects[object].vertexCount == 0) continue;
        const std::array<float, 16> w = worldMatrix (scene.objects[object]);
        range (scene.objects[object], first, end);
        for (std::size_t index = first; index < end; ++index)
            floor = std::min (floor, dot (world (w, scene.vertices[index].position), up));
    }
    if (! (low1 <= high1) || ! std::isfinite (floor) || ! std::isfinite (surface)) return;
    u[7] = { up[0], up[1], up[2], floor };
    u[8] = { lateral[0], lateral[1], lateral[2], surface };
    u[9] = { low1, high1, low2, high2 };
}
// The partner id that smoothglass::cameraPartner packs as two 16-bit halves.
inline HarmonicMIDI::grid::SceneObjectId partnerId (const std::array<float, 4>& cameraPartner) noexcept
{
    return HarmonicMIDI::grid::SceneObjectId { std::uint32_t (cameraPartner[2]) | (std::uint32_t (cameraPartner[3]) << 16u) };
}

// The world fog rows of a fog volume, marched to farDistance behind the sky.
// An inactive volume packs zero, which leaves every pixel unfogged.
inline Uniforms packWorldFog (const HarmonicMIDI::grid::SceneVolume& volume, float farDistance) noexcept
{
    if (! HarmonicMIDI::grid::sceneVolumeActive (volume)) return {};
    auto u = pack (volume, {}, {});
    u[0][3] = 1.0f;
    u[4] = { std::isfinite (farDistance) && farDistance > 0.0f ? farDistance : 0.0f, 0.0f, 0.0f, u[4][3] };
    u[5] = {};
    return u;
}

// The world fog of a Raster or Realtime frame: the render's own fog over the
// scene's world volume, marched to the active camera's far plane.
inline Uniforms worldFogForScene (const HarmonicMIDI::grid::Visual3DScene& scene,
                                  const HarmonicMIDI::grid::SceneVolume* renderFog) noexcept
{
    const auto* camera = HarmonicMIDI::grid::visual3d_detail::findById (scene.cameras, scene.cameraCount, scene.activeCamera);
    if (camera == nullptr) return {};
    return packWorldFog (renderFog != nullptr ? *renderFog : scene.worldVolume, camera->farPlane);
}

inline bool worldFogOn (const Uniforms& fog) noexcept { return fog[0][3] > 0.5f; }

// The scene-linear backdrop a frame with no sky draw clears to, which the world
// fog's backdrop pass fogs: the stage colour, or with a camera-hidden
// environment its Colour, opaque black or zero-alpha Transparent.
inline std::array<float, 4> backdropRow (bool cameraHiddenBackdrop,
                                         const HarmonicMIDI::grid::SceneCameraBackground& background) noexcept
{
    using Mode = HarmonicMIDI::grid::SceneCameraBackgroundMode;
    if (! cameraHiddenBackdrop) return { 7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f, 1.0f };
    const bool colour = background.mode == Mode::Colour;
    const auto channel = [colour] (float c) { return colour && std::isfinite (c) ? std::max (0.0f, c) : 0.0f; };
    return { channel (background.color.x), channel (background.color.y), channel (background.color.z),
             background.mode == Mode::Transparent ? 0.0f : 1.0f };
}

// The world fog of the frame on this thread. The scene renderer opens a
// WorldFogScope around a Raster or Realtime frame; the backends read it when
// they build the program and fill the uniforms.
inline Uniforms& currentWorldFog() noexcept
{
    thread_local Uniforms fog {};
    return fog;
}
class WorldFogScope final
{
public:
    explicit WorldFogScope (const Uniforms& fog) noexcept : previous_ (currentWorldFog()) { currentWorldFog() = fog; }
    ~WorldFogScope() { currentWorldFog() = previous_; }
    WorldFogScope (const WorldFogScope&) = delete;
    WorldFogScope& operator= (const WorldFogScope&) = delete;

private:
    Uniforms previous_;
};

// Beer-Lambert transmittance of the absorption over distance.
inline Rgb tint (const Uniforms& u, float distance) noexcept
{
    const float d = std::isfinite (distance) ? std::max (0.0f, distance) : 0.0f;
    return { std::exp (-u[0][0] * d), std::exp (-u[0][1] * d), std::exp (-u[0][2] * d) };
}

// Path Trace's lattice hash and fBm (pathtrace/media.h volumeFieldAt).
inline std::uint32_t hash (std::uint32_t input) noexcept
{
    const std::uint32_t state = input * 747796405u + 2891336453u;
    const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline std::uint32_t wrapped (float x, float period) noexcept
{
    const float w = x - period * std::floor (x / period);
    return std::isfinite (w) && w >= 0.0f && w < period ? static_cast<std::uint32_t> (w) : 0u;
}
inline float lattice (float x, float y, float z) noexcept
{
    constexpr float kPeriod = 1048576.0f;
    std::uint32_t h = hash (wrapped (x, kPeriod));
    h = hash (h ^ (wrapped (y, kPeriod) * 0x9e3779b9u));
    h = hash (h ^ (wrapped (z, kPeriod) * 0x85ebca6bu));
    return static_cast<float> (h >> 8) * (1.0f / 16777216.0f);
}
inline float valueNoise (float x, float y, float z) noexcept
{
    const auto fade = [] (float t) { return t * t * (3.0f - 2.0f * t); };
    const auto lerp = [] (float a, float b, float t) { return a + (b - a) * t; };
    const float fx = std::floor (x), fy = std::floor (y), fz = std::floor (z);
    const float tx = fade (x - fx), ty = fade (y - fy), tz = fade (z - fz);
    const float c00 = lerp (lattice (fx, fy, fz), lattice (fx + 1.0f, fy, fz), tx);
    const float c10 = lerp (lattice (fx, fy + 1.0f, fz), lattice (fx + 1.0f, fy + 1.0f, fz), tx);
    const float c01 = lerp (lattice (fx, fy, fz + 1.0f), lattice (fx + 1.0f, fy, fz + 1.0f), tx);
    const float c11 = lerp (lattice (fx, fy + 1.0f, fz + 1.0f), lattice (fx + 1.0f, fy + 1.0f, fz + 1.0f), tx);
    return lerp (lerp (c00, c10, ty), lerp (c01, c11, ty), tz);
}
inline float field (const Uniforms& u, float x, float y, float z) noexcept
{
    if (u[2][3] < 0.5f) return 1.0f;
    const float scale = u[3][3];
    float qx = (x + u[3][0]) * scale, qy = (y + u[3][1]) * scale, qz = (z + u[3][2]) * scale;
    const int octaves = static_cast<int> (u[4][3]);
    float sum = 0.0f, amplitude = 1.0f, total = 0.0f;
    for (int o = 0; o < octaves; ++o)
    {
        sum += amplitude * valueNoise (qx, qy, qz);
        total += amplitude;
        amplitude *= 0.5f;
        qx = qx * 2.0f + 17.1f;
        qy = qy * 2.0f + 31.7f;
        qz = qz * 2.0f + 5.3f;
    }
    return std::min (1.0f, std::max (0.0f, sum / total));
}

inline float phase (float cosine, float g) noexcept
{
    const float denominator = 1.0f + g * g - 2.0f * g * cosine;
    return denominator > 0.0f ? (1.0f - g * g) / (12.566370614359172f * denominator * std::sqrt (denominator)) : 0.0f;
}

// Entry and exit distances of the ray o + t d through the bounds, if it meets
// them: rows 4 and 5, or the water body in rows 7 and 8.
inline bool boxSpan (const Uniforms& u, const float o[3], const float d[3], float& t0, float& t1, int row = 4) noexcept
{
    t0 = -3.402823e38f;
    t1 = 3.402823e38f;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float low = u[row][axis], high = u[row + 1][axis];
        if (std::abs (d[axis]) < 1.0e-12f)
        {
            if (o[axis] < low || o[axis] > high) return false;
            continue;
        }
        float a = (low - o[axis]) / d[axis], b = (high - o[axis]) / d[axis];
        if (a > b) std::swap (a, b);
        t0 = std::max (t0, a);
        t1 = std::min (t1, b);
    }
    return t1 >= std::max (t0, 0.0f);
}

// The path a paired-walk branch that left the scene travels inside the water
// body, from its origin (or where it enters, a bias away) to where it leaves:
// zero without a body or when the branch misses it.
inline float waterExit (const Uniforms& u, const float o[3], const float d[3]) noexcept
{
    const float* up = u[7].data();
    const float* lateral = u[8].data();
    if (up[0] * up[0] + up[1] * up[1] + up[2] * up[2] < 0.5f) return 0.0f;
    const float second[3] = { up[1] * lateral[2] - up[2] * lateral[1], up[2] * lateral[0] - up[0] * lateral[2],
                              up[0] * lateral[1] - up[1] * lateral[0] };
    const float* axes[3] = { up, lateral, second };
    const float lows[3] = { u[7][3], u[9][0], u[9][2] }, highs[3] = { u[8][3], u[9][1], u[9][3] };
    float t0 = -3.402823e38f, t1 = 3.402823e38f;
    for (int axis = 0; axis < 3; ++axis)
    {
        const float* a = axes[axis];
        const float oa = a[0] * o[0] + a[1] * o[1] + a[2] * o[2], da = a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
        if (std::fabs (da) < 1.0e-12f)
        {
            if (oa < lows[axis] || oa > highs[axis]) return 0.0f;
            continue;
        }
        float enter = (lows[axis] - oa) / da, leave = (highs[axis] - oa) / da;
        if (enter > leave) std::swap (enter, leave);
        t0 = std::max (t0, enter);
        t1 = std::min (t1, leave);
    }
    if (! (t1 >= std::max (t0, 0.0f))) return 0.0f;
    return std::max (0.0f, t1 - std::max (0.0f, t0));
}

struct Light
{
    Rgb toLight {};      // unit direction from the sample toward the light
    Rgb radiance {};     // after distance and cone attenuation
};
struct March
{
    Rgb radiance {};
    Rgb transmittance { 1.0f, 1.0f, 1.0f };
};

// Single scattering along o + t d for t in [start, end]. lights gives the
// lights at a sample with their visibility applied; ambient is isotropic.
inline March march (const Uniforms& u, const float o[3], const float d[3], float start, float end, Rgb ambient,
                    const std::function<std::array<Light, 8> (const float p[3], int& count)>& lights)
{
    March result;
    if (! (end > start)) return result;
    const float step = (end - start) / float (kMarchSteps);
    for (int i = 0; i < kMarchSteps; ++i)
    {
        const float t = start + (float (i) + 0.5f) * step;
        const float p[3] = { o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t };
        const float f = field (u, p[0], p[1], p[2]);
        int count = 0;
        const auto sampled = lights (p, count);
        for (int c = 0; c < 3; ++c)
        {
            const float sigmaS = f * u[1][c], sigmaT = f * (u[0][c] + u[1][c]);
            float source = sigmaS * ambient[c] + f * u[2][c];
            for (int l = 0; l < count; ++l)
            {
                const auto& light = sampled[l];
                const float cosine = light.toLight[0] * d[0] + light.toLight[1] * d[1] + light.toLight[2] * d[2];
                source += sigmaS * phase (cosine, u[1][3]) * light.radiance[c];
            }
            const float keep = std::exp (-sigmaT * step);
            const float span = sigmaT > 0.0f ? (1.0f - keep) / sigmaT : step;
            result.radiance[c] += result.transmittance[c] * source * span;
            result.transmittance[c] *= keep;
        }
    }
    return result;
}

// The world fog's backdrop pass on the host: the backdrop row seen along the
// unit d from o through the fog to its far distance, the opacity a transparent
// backdrop takes from the fog's mean transmittance.
inline std::array<float, 4> fogBackdrop (const Uniforms& fog, const std::array<float, 4>& backdrop,
                                         const float o[3], const float d[3], Rgb ambient,
                                         const std::function<std::array<Light, 8> (const float p[3], int& count)>& lights)
{
    if (! worldFogOn (fog)) return backdrop;
    const auto fogged = march (fog, o, d, 0.0f, fog[4][0], ambient, lights);
    const float mean = (fogged.transmittance[0] + fogged.transmittance[1] + fogged.transmittance[2]) / 3.0f;
    return { backdrop[0] * fogged.transmittance[0] + fogged.radiance[0],
             backdrop[1] * fogged.transmittance[1] + fogged.radiance[1],
             backdrop[2] * fogged.transmittance[2] + fogged.radiance[2],
             1.0f - (1.0f - backdrop[3]) * mean };
}

// The same arithmetic for the GL fixture fragment program. It reads
// uCameraTranslation, uAmbient, uLightCount and the light table from the
// fixture head, and the shadow visibility of Raster (opaqueVisibility) or
// Realtime (realtimeSoftVisibility).
inline std::string glslShader()
{
    return R"glsl(
uniform vec4 uMedium[10];
uniform vec4 uWorldFog[5];
// The glass walks' medium hook (fixture_smooth_glass.h): absorption over a
// segment inside the shaded dielectric, and inside the paired walk's partner.
// Branch weight a medium absorbs is not counted as discarded walk power.
#define GLASS_MEDIUM_TINT(distance) mediumTint(distance)
#define GLASS_PARTNER_MEDIUM_TINT(distance) exp(-uMedium[6].xyz * max(distance, 0.0))
#define GLASS_MEDIUM_DISCARDS (int(uMedium[0].w) == 1 || any(greaterThan(uMedium[6].xyz, vec3(0.0))) ? 0.0 : 1.0)
#define GLASS_WATER_EXIT(origin, direction) mediumWaterExit(origin, direction)
#ifdef FIXTURE_REALTIME
float realtimeSoftVisibility(vec3 position, vec3 normal, vec3 toLight, float maxDistance, float angularRadius);
#endif
uint mediumHash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint mediumWrapped(float x, float period) {
    float w = x - period * floor(x / period);
    return (!isnan(w) && !isinf(w) && w >= 0.0 && w < period) ? uint(w) : 0u;
}
float mediumLattice(float x, float y, float z) {
    uint h = mediumHash(mediumWrapped(x, 1048576.0));
    h = mediumHash(h ^ (mediumWrapped(y, 1048576.0) * 0x9e3779b9u));
    h = mediumHash(h ^ (mediumWrapped(z, 1048576.0) * 0x85ebca6bu));
    return float(h >> 8u) * (1.0 / 16777216.0);
}
float mediumNoise(vec3 p) {
    vec3 c = floor(p);
    vec3 t = p - c;
    t = t * t * (3.0 - 2.0 * t);
    float c00 = mix(mediumLattice(c.x, c.y, c.z), mediumLattice(c.x + 1.0, c.y, c.z), t.x);
    float c10 = mix(mediumLattice(c.x, c.y + 1.0, c.z), mediumLattice(c.x + 1.0, c.y + 1.0, c.z), t.x);
    float c01 = mix(mediumLattice(c.x, c.y, c.z + 1.0), mediumLattice(c.x + 1.0, c.y, c.z + 1.0), t.x);
    float c11 = mix(mediumLattice(c.x, c.y + 1.0, c.z + 1.0), mediumLattice(c.x + 1.0, c.y + 1.0, c.z + 1.0), t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}
// Rows of the draw's medium, or of the world fog.
vec4 mediumRow(bool world, int row) { return world ? uWorldFog[row] : uMedium[row]; }
float mediumFieldOf(bool world, vec3 p) {
    if (mediumRow(world, 2).w < 0.5) return 1.0;
    vec3 q = (p + mediumRow(world, 3).xyz) * mediumRow(world, 3).w;
    int octaves = int(mediumRow(world, 4).w);
    float sum = 0.0, amplitude = 1.0, total = 0.0;
    for (int o = 0; o < 8; ++o) {
        if (o >= octaves) break;
        sum += amplitude * mediumNoise(q);
        total += amplitude;
        amplitude *= 0.5;
        q = q * 2.0 + vec3(17.1, 31.7, 5.3);
    }
    return clamp(sum / max(total, 1.0e-6), 0.0, 1.0);
}
float mediumField(vec3 p) { return mediumFieldOf(false, p); }
float mediumPhase(float cosine, float g) {
    float denominator = 1.0 + g * g - 2.0 * g * cosine;
    return denominator > 0.0 ? (1.0 - g * g) / (12.566370614359172 * denominator * sqrt(denominator)) : 0.0;
}
bool mediumRowsSpan(int row, vec3 o, vec3 d, out float t0, out float t1) {
    t0 = -3.402823e38;
    t1 = 3.402823e38;
    for (int axis = 0; axis < 3; ++axis) {
        float low = uMedium[row][axis], high = uMedium[row + 1][axis];
        if (abs(d[axis]) < 1.0e-12) {
            if (o[axis] < low || o[axis] > high) return false;
            continue;
        }
        float a = (low - o[axis]) / d[axis], b = (high - o[axis]) / d[axis];
        t0 = max(t0, min(a, b));
        t1 = min(t1, max(a, b));
    }
    return t1 >= max(t0, 0.0);
}
bool mediumBoxSpan(vec3 o, vec3 d, out float t0, out float t1) {
    return mediumRowsSpan(4, o, d, t0, t1);
}
// The oriented water body of rows 7 to 9 (rastermedium::waterExit).
float mediumWaterExit(vec3 o, vec3 d) {
    vec3 up = uMedium[7].xyz, lateral = uMedium[8].xyz;
    if (dot(up, up) < 0.5) return 0.0;
    mat3 axes = mat3(up, lateral, cross(up, lateral));
    vec3 oa = o * axes, da = d * axes;
    vec3 lows = vec3(uMedium[7].w, uMedium[9].x, uMedium[9].z), highs = vec3(uMedium[8].w, uMedium[9].y, uMedium[9].w);
    float t0 = -3.402823e38, t1 = 3.402823e38;
    for (int axis = 0; axis < 3; ++axis) {
        if (abs(da[axis]) < 1.0e-12) {
            if (oa[axis] < lows[axis] || oa[axis] > highs[axis]) return 0.0;
            continue;
        }
        float a = (lows[axis] - oa[axis]) / da[axis], b = (highs[axis] - oa[axis]) / da[axis];
        t0 = max(t0, min(a, b));
        t1 = min(t1, max(a, b));
    }
    if (!(t1 >= max(t0, 0.0))) return 0.0;
    return max(0.0, t1 - max(t0, 0.0));
}
vec3 mediumTint(float distance) {
    return exp(-uMedium[0].xyz * max(distance, 0.0));
}
// The tint of a refracted ray that leaves the bounds after their exit distance.
vec3 mediumBoxTint(vec3 position, vec3 direction) {
    float t0, t1;
    if (int(uMedium[0].w) != 1 || dot(direction, direction) < 0.25
        || !mediumBoxSpan(position, direction, t0, t1)) return vec3(1.0);
    return mediumTint(t1);
}
vec3 mediumRotate(vec4 q, vec3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
// The light scattered toward the camera at p, with the raster's shadows.
vec3 mediumSourceOf(bool world, vec3 p, vec3 d, float f) {
    vec3 sigmaS = f * mediumRow(world, 1).xyz;
    vec3 source = sigmaS * uAmbient + f * mediumRow(world, 2).xyz;
    for (int i = 0; i < uLightCount; ++i) {
        // A light hidden from volume scatter rays adds no in-scatter.
        if (!fixtureLightInVolumes(i)) continue;
        int kind = fixtureLightKind(i);
        vec3 radiance = fixtureLightColor(i).rgb * fixtureLightColor(i).a;
        if (kind == 2) { source += sigmaS * radiance; continue; }
        vec3 delta = fixtureLightPosition(i).xyz - p;
        float lightDistance = length(delta);
        vec3 toLight = kind == 0 ? mediumRotate(fixtureLightRotation(i), vec3(0.0, 0.0, 1.0))
                                 : delta / max(lightDistance, 1.0e-6);
        if (kind == 1 || kind == 3 || kind == 4) {
            radiance *= 1.0 / max(lightDistance * lightDistance, 1.0);
            float range = kind == 4 ? 0.0 : fixtureLightPosition(i).w;
            if (range > 0.0) radiance *= pow(clamp(1.0 - lightDistance / range, 0.0, 1.0), 2.0);
            if (kind == 3) {
                vec3 spot = mediumRotate(fixtureLightRotation(i), vec3(0.0, 0.0, -1.0));
                radiance *= smoothstep(fixtureLightCone(i).y, fixtureLightCone(i).x, dot(spot, -toLight));
            }
            // An area light shades as a point at its centre, as the raster surface does.
            if (kind == 4) radiance *= fixtureAreaEmission(i, -toLight);
        }
        float limit = kind == 0 ? 3.402823e38 : lightDistance;
#ifdef FIXTURE_REALTIME
        // A sized light's shadow softens by the angle it subtends from p, as on surfaces.
        float lightSize = fixtureLightSize(i);
        float subtended = kind == 0 ? lightSize : atan(lightSize, max(lightDistance, 1.0e-4));
        float visibility = realtimeSoftVisibility(p, toLight, toLight, limit, subtended);
#else
        float visibility = opaqueVisibility(p, toLight, toLight, limit, uOpaqueControl);
#endif
        source += sigmaS * mediumPhase(dot(toLight, d), mediumRow(world, 1).w) * radiance * visibility;
    }
    return source;
}
vec3 mediumSource(vec3 p, vec3 d, float f) { return mediumSourceOf(false, p, d, f); }
// The world fog from the camera along the unit d for distance: in-scattered
// radiance, with transmittance out.
vec3 worldFogMarch(vec3 camera, vec3 d, float distance, out vec3 transmittance) {
    float stride = max(distance, 0.0) / 32.0;
    vec3 radiance = vec3(0.0);
    transmittance = vec3(1.0);
    for (int i = 0; i < 32; ++i) {
        vec3 p = camera + d * ((float(i) + 0.5) * stride);
        float f = mediumFieldOf(true, p);
        vec3 sigmaT = f * (uWorldFog[0].xyz + uWorldFog[1].xyz);
        vec3 keep = exp(-sigmaT * stride);
        vec3 span = mix(vec3(stride), (vec3(1.0) - keep) / max(sigmaT, vec3(1.0e-12)), greaterThan(sigmaT, vec3(0.0)));
        radiance += transmittance * mediumSourceOf(true, p, d, f) * span;
        transmittance *= keep;
    }
    return radiance;
}
// compositePass's world fog: a surface fogs over its distance from the camera,
// and the sky over the far distance along its direction. The backdrop pass
// (uSkyBackground 2) fogs a frame with no sky draw the same way, and the fog
// covers a transparent backdrop by its mean opacity.
vec4 worldFogApply(vec4 color) {
    if (uWorldFog[0].w < 0.5) return color;
    vec3 camera = uCameraTranslation;
    bool sky = uSkyBackground > 0.5;
    vec3 d = sky ? normalize(vWorldPosition) : normalize(vWorldPosition - camera);
    float distance = sky ? uWorldFog[4].x : length(vWorldPosition - camera);
    vec3 transmittance;
    vec3 inscattered = worldFogMarch(camera, d, distance, transmittance);
    float alpha = uSkyBackground > 1.5
        ? 1.0 - (1.0 - color.a) * dot(transmittance, vec3(1.0 / 3.0)) : color.a;
    return vec4(color.rgb * transmittance + inscattered, alpha);
}
// A Volume Material draw. Front faces march from the surface to the bounds'
// exit; with the camera inside the bounds, back faces march from the camera.
// The march stops at the first surface of the opaque scene past its start,
// which is also the far side of the volume's own mesh. Returns radiance over
// the mean opacity and that opacity, for the draw's source-alpha blend.
vec4 mediumVolume(vec3 position, bool front) {
    vec3 camera = uCameraTranslation;
    vec3 d = normalize(position - camera);
    float t0, t1;
    if (!mediumBoxSpan(camera, d, t0, t1)) return vec4(0.0);
    float surface = length(position - camera);
    float start = front ? surface : 0.0;
    float end = front ? max(t1, surface) : surface;
    if (!front && t0 > 0.0) return vec4(0.0);
    // The opaque frame answers only blocked or clear, so a blocked span is
    // bisected down to the wall to a 1/256 of its length.
    vec3 from = camera + d * start;
    if (end > start && opaqueVisibility(from, d, d, end - start, uOpaqueControl) < 0.5) {
        float clear = 0.0, blocked = end - start;
        for (int i = 0; i < 8; ++i) {
            float middle = 0.5 * (clear + blocked);
            if (opaqueVisibility(from, d, d, middle, uOpaqueControl) < 0.5) blocked = middle;
            else clear = middle;
        }
        end = start + blocked;
    }
    float stride = (end - start) / 32.0;
    vec3 radiance = vec3(0.0), transmittance = vec3(1.0);
    for (int i = 0; i < 32; ++i) {
        vec3 p = camera + d * (start + (float(i) + 0.5) * stride);
        float f = mediumField(p);
        vec3 sigmaT = f * (uMedium[0].xyz + uMedium[1].xyz);
        vec3 keep = exp(-sigmaT * stride);
        vec3 span = mix(vec3(stride), (vec3(1.0) - keep) / max(sigmaT, vec3(1.0e-12)), greaterThan(sigmaT, vec3(0.0)));
        radiance += transmittance * mediumSource(p, d, f) * span;
        transmittance *= keep;
    }
    float opacity = clamp(1.0 - (transmittance.r + transmittance.g + transmittance.b) / 3.0, 0.0, 1.0);
    return vec4(radiance / max(opacity, 1.0e-4), opacity);
}
)glsl";
}

// The same arithmetic in MSL for the Metal fixture fragment, whose Uniforms
// carry u.medium[10] after realtimeControl and whose lights follow the Metal
// surface shading (lightKinds, lightCones spread for area lights). It needs
// opaqueVisibility from opaquevisibility::metalShader() above it, and it
// defines the glass walks' medium hook, so it comes before the transport text.
inline std::string metalShader()
{
    return R"metal(
#define FIXTURE_MEDIUM 1
#define GLASS_MEDIUM_TINT(distance) mediumTint(u, distance)
#define GLASS_PARTNER_MEDIUM_TINT(distance) exp(-u.medium[6].xyz * max(distance, 0.0f))
#define GLASS_MEDIUM_DISCARDS (int(u.medium[0].w) == 1 || any(u.medium[6].xyz > float3(0.0f)) ? 0.0f : 1.0f)
#define GLASS_WATER_EXIT(origin, direction) mediumWaterExit(u, origin, direction)
uint mediumHash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint mediumWrapped(float x, float period) {
    float w = x - period * floor(x / period);
    return (!isnan(w) && !isinf(w) && w >= 0.0f && w < period) ? uint(w) : 0u;
}
float mediumLattice(float x, float y, float z) {
    uint h = mediumHash(mediumWrapped(x, 1048576.0f));
    h = mediumHash(h ^ (mediumWrapped(y, 1048576.0f) * 0x9e3779b9u));
    h = mediumHash(h ^ (mediumWrapped(z, 1048576.0f) * 0x85ebca6bu));
    return float(h >> 8u) * (1.0f / 16777216.0f);
}
float mediumNoise(float3 p) {
    float3 c = floor(p);
    float3 t = p - c;
    t = t * t * (3.0f - 2.0f * t);
    float c00 = mix(mediumLattice(c.x, c.y, c.z), mediumLattice(c.x + 1.0f, c.y, c.z), t.x);
    float c10 = mix(mediumLattice(c.x, c.y + 1.0f, c.z), mediumLattice(c.x + 1.0f, c.y + 1.0f, c.z), t.x);
    float c01 = mix(mediumLattice(c.x, c.y, c.z + 1.0f), mediumLattice(c.x + 1.0f, c.y, c.z + 1.0f), t.x);
    float c11 = mix(mediumLattice(c.x, c.y + 1.0f, c.z + 1.0f), mediumLattice(c.x + 1.0f, c.y + 1.0f, c.z + 1.0f), t.x);
    return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}
// Rows of the draw's medium, or of the world fog.
float4 mediumRow(constant Uniforms& u, bool world, int row) { return world ? u.worldFog[row] : u.medium[row]; }
float mediumFieldOf(constant Uniforms& u, bool world, float3 p) {
    if (mediumRow(u, world, 2).w < 0.5f) return 1.0f;
    float3 q = (p + mediumRow(u, world, 3).xyz) * mediumRow(u, world, 3).w;
    int octaves = int(mediumRow(u, world, 4).w);
    float sum = 0.0f, amplitude = 1.0f, total = 0.0f;
    for (int o = 0; o < 8; ++o) {
        if (o >= octaves) break;
        sum += amplitude * mediumNoise(q);
        total += amplitude;
        amplitude *= 0.5f;
        q = q * 2.0f + float3(17.1f, 31.7f, 5.3f);
    }
    return clamp(sum / max(total, 1.0e-6f), 0.0f, 1.0f);
}
float mediumField(constant Uniforms& u, float3 p) { return mediumFieldOf(u, false, p); }
float mediumPhase(float cosine, float g) {
    float denominator = 1.0f + g * g - 2.0f * g * cosine;
    return denominator > 0.0f ? (1.0f - g * g) / (12.566370614359172f * denominator * sqrt(denominator)) : 0.0f;
}
bool mediumRowsSpan(constant Uniforms& u, int row, float3 o, float3 d, thread float& t0, thread float& t1) {
    t0 = -3.402823e38f;
    t1 = 3.402823e38f;
    for (int axis = 0; axis < 3; ++axis) {
        float low = u.medium[row][axis], high = u.medium[row + 1][axis];
        if (abs(d[axis]) < 1.0e-12f) {
            if (o[axis] < low || o[axis] > high) return false;
            continue;
        }
        float a = (low - o[axis]) / d[axis], b = (high - o[axis]) / d[axis];
        t0 = max(t0, min(a, b));
        t1 = min(t1, max(a, b));
    }
    return t1 >= max(t0, 0.0f);
}
bool mediumBoxSpan(constant Uniforms& u, float3 o, float3 d, thread float& t0, thread float& t1) {
    return mediumRowsSpan(u, 4, o, d, t0, t1);
}
// The oriented water body of rows 7 to 9, as the GL mediumWaterExit.
float mediumWaterExit(constant Uniforms& u, float3 o, float3 d) {
    float3 up = u.medium[7].xyz, lateral = u.medium[8].xyz;
    if (dot(up, up) < 0.5f) return 0.0f;
    float3x3 axes = float3x3(up, lateral, cross(up, lateral));
    float3 oa = o * axes, da = d * axes;
    float3 lows = float3(u.medium[7].w, u.medium[9].x, u.medium[9].z);
    float3 highs = float3(u.medium[8].w, u.medium[9].y, u.medium[9].w);
    float t0 = -3.402823e38f, t1 = 3.402823e38f;
    for (int axis = 0; axis < 3; ++axis) {
        if (abs(da[axis]) < 1.0e-12f) {
            if (oa[axis] < lows[axis] || oa[axis] > highs[axis]) return 0.0f;
            continue;
        }
        float a = (lows[axis] - oa[axis]) / da[axis], b = (highs[axis] - oa[axis]) / da[axis];
        t0 = max(t0, min(a, b));
        t1 = min(t1, max(a, b));
    }
    if (!(t1 >= max(t0, 0.0f))) return 0.0f;
    return max(0.0f, t1 - max(t0, 0.0f));
}
float3 mediumTint(constant Uniforms& u, float distance) {
    return exp(-u.medium[0].xyz * max(distance, 0.0f));
}
float3 mediumRotate(float4 q, float3 v) { return v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
// The light scattered toward the camera at p, with the raster's shadows.
float3 mediumSourceOf(constant Uniforms& u, texture2d<float> opaqueTexture, bool world, float3 p, float3 d, float f) {
    float3 sigmaS = f * mediumRow(u, world, 1).xyz;
    float3 source = sigmaS * u.ambient.xyz + f * mediumRow(u, world, 2).xyz;
    for (uint i = 0u; i < 16u; ++i) {
        if (i >= uint(u.ambient.w)) break;
        // A light hidden from volume scatter rays adds no in-scatter.
        if (u.lightKinds[i].y != 0u) continue;
        uint kind = u.lightKinds[i].x;
        float3 radiance = u.lightColors[i].rgb * u.lightColors[i].a;
        if (kind == 2u) { source += sigmaS * radiance; continue; }
        float3 delta = u.lightPositions[i].xyz - p;
        float lightDistance = length(delta);
        float3 toLight = kind == 0u ? mediumRotate(u.lightRotations[i], float3(0.0f, 0.0f, 1.0f))
                                    : delta / max(lightDistance, 1.0e-6f);
        if (kind == 1u || kind == 3u || kind == 4u) {
            radiance *= 1.0f / max(lightDistance * lightDistance, 1.0f);
            float range = u.lightPositions[i].w;
            if (range > 0.0f) radiance *= pow(clamp(1.0f - lightDistance / range, 0.0f, 1.0f), 2.0f);
            if (kind == 3u) {
                float3 spot = mediumRotate(u.lightRotations[i], float3(0.0f, 0.0f, -1.0f));
                radiance *= smoothstep(u.lightCones[i].y, u.lightCones[i].x, dot(spot, -toLight));
            }
            // An area light shades as a point at its centre, as the raster surface does.
            if (kind == 4u) {
                float3 areaNormal = mediumRotate(u.lightRotations[i], float3(0.0f, 0.0f, -1.0f));
                float cosine = dot(areaNormal, -toLight);
                float4 spread = u.lightCones[i];
                radiance *= cosine <= 0.0f ? 0.0f : spread.y <= 0.0f ? cosine
                    : cosine * max(0.0f, (spread.x - sqrt(max(0.0f, 1.0f - cosine * cosine)) / cosine) * spread.y);
            }
        }
        float limit = kind == 0u ? 3.402823e38f : lightDistance;
        float visibility = u.lightCones[i].w < 0.0f ? 1.0f
            : opaqueVisibility(p, toLight, toLight, limit, u.opaqueControl, opaqueTexture);
        source += sigmaS * mediumPhase(dot(toLight, d), mediumRow(u, world, 1).w) * radiance * visibility;
    }
    return source;
}
float3 mediumSource(constant Uniforms& u, texture2d<float> opaqueTexture, float3 p, float3 d, float f) {
    return mediumSourceOf(u, opaqueTexture, false, p, d, f);
}
// The world fog from the camera along the unit d for distance, as the GL
// worldFogMarch: in-scattered radiance, with transmittance out.
float3 worldFogMarch(constant Uniforms& u, texture2d<float> opaqueTexture, float3 camera, float3 d, float distance,
                     thread float3& transmittance) {
    float stride = max(distance, 0.0f) / 32.0f;
    float3 radiance = float3(0.0f);
    transmittance = float3(1.0f);
    for (int i = 0; i < 32; ++i) {
        float3 p = camera + d * ((float(i) + 0.5f) * stride);
        float f = mediumFieldOf(u, true, p);
        float3 sigmaT = f * (u.worldFog[0].xyz + u.worldFog[1].xyz);
        float3 keep = exp(-sigmaT * stride);
        float3 span = select(float3(stride), (float3(1.0f) - keep) / max(sigmaT, float3(1.0e-12f)), sigmaT > float3(0.0f));
        radiance += transmittance * mediumSourceOf(u, opaqueTexture, true, p, d, f) * span;
        transmittance *= keep;
    }
    return radiance;
}
// The fragment's colour under the world fog, before compositePass: a surface
// fogs over its distance from the camera, and the sky or the backdrop pass
// (skyBackground 2) over the far distance.
FragmentOut worldFogged(FragmentOut value, constant Uniforms& u, texture2d<float> opaqueTexture, float3 position) {
    if (u.worldFog[0].w < 0.5f) return value;
    float3 camera = u.cameraTranslation.xyz;
    bool sky = u.skyBackground.x > 0.5f;
    float3 d = sky ? normalize(position) : normalize(position - camera);
    float distance = sky ? u.worldFog[4].x : length(position - camera);
    float3 transmittance;
    float3 inscattered = worldFogMarch(u, opaqueTexture, camera, d, distance, transmittance);
    value.color.rgb = value.color.rgb * transmittance + inscattered;
    if (u.skyBackground.x > 1.5f)
        value.color.a = 1.0f - (1.0f - value.color.a) * dot(transmittance, float3(1.0f / 3.0f));
    return value;
}
// A Volume Material draw, as the GL mediumVolume: the march runs inside the
// bounds and stops at the first opaque surface past its start.
float4 mediumVolume(constant Uniforms& u, texture2d<float> opaqueTexture, float3 position, bool front) {
    float3 camera = u.cameraTranslation.xyz;
    float3 d = normalize(position - camera);
    float t0, t1;
    if (!mediumBoxSpan(u, camera, d, t0, t1)) return float4(0.0f);
    float surface = length(position - camera);
    float start = front ? surface : 0.0f;
    float end = front ? max(t1, surface) : surface;
    if (!front && t0 > 0.0f) return float4(0.0f);
    float3 from = camera + d * start;
    if (end > start && opaqueVisibility(from, d, d, end - start, u.opaqueControl, opaqueTexture) < 0.5f) {
        float clear = 0.0f, blocked = end - start;
        for (int i = 0; i < 8; ++i) {
            float middle = 0.5f * (clear + blocked);
            if (opaqueVisibility(from, d, d, middle, u.opaqueControl, opaqueTexture) < 0.5f) blocked = middle;
            else clear = middle;
        }
        end = start + blocked;
    }
    float stride = (end - start) / 32.0f;
    float3 radiance = float3(0.0f), transmittance = float3(1.0f);
    for (int i = 0; i < 32; ++i) {
        float3 p = camera + d * (start + (float(i) + 0.5f) * stride);
        float f = mediumField(u, p);
        float3 sigmaT = f * (u.medium[0].xyz + u.medium[1].xyz);
        float3 keep = exp(-sigmaT * stride);
        float3 span = select(float3(stride), (float3(1.0f) - keep) / max(sigmaT, float3(1.0e-12f)), sigmaT > float3(0.0f));
        radiance += transmittance * mediumSource(u, opaqueTexture, p, d, f) * span;
        transmittance *= keep;
    }
    float opacity = clamp(1.0f - (transmittance.r + transmittance.g + transmittance.b) / 3.0f, 0.0f, 1.0f);
    return float4(radiance / max(opacity, 1.0e-4f), opacity);
}
)metal";
}
} // namespace rastermedium
