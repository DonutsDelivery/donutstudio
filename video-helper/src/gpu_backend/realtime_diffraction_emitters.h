#pragma once

// Realtime lights a grating's orders by the scene's glowing objects as well as its sky.
// Path Trace finds a studio panel or a lamp through a delta order by tracing the order's
// ray into the scene. Realtime has no closest-hit structure for ordinary surfaces, so the
// frame lists every triangle of every emissive object in world space, and the grating
// shader casts each order's ray against that list (realtimeDiffractionEmitter in
// backend_opengl.cpp). The nearest emitter the ray reaches lights that order with its
// emission, unless something opaque stands in front of it. A ray that reaches no emitter
// reads the sky as before.
//
// The list is the scene's own emissive geometry, so its length follows the scene. Each
// triangle is three RGBA32F texels: the corners in xyz and the emission's red, green and
// blue in the three w components. The texels are appended to the frame's scene table.
//
// Tests use the host forms here. Raster and Path Trace never read the list.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace arbitgpu::realtimediffraction
{
using EmitterTexel = std::array<float, 4>;
// Column-major 4x4, translation in elements 12 to 14, as the fixture matrices are.
using EmitterMatrix = std::array<float, 16>;

inline std::array<float, 3> emitterTransform (const EmitterMatrix& m, const std::array<float, 3>& p) noexcept
{
    return { m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
             m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
             m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14] };
}

// Something glows when any channel of its emission is finite and above zero.
inline bool emits (const std::array<float, 3>& emission) noexcept
{
    bool any = false;
    for (const auto value : emission)
    {
        if (! std::isfinite (value) || value < 0.0f) return false;
        any = any || value > 0.0f;
    }
    return any;
}

// Appends one object's triangles. indices are relative to the object's first position.
// An index past the positions is malformed geometry; the object is left out and the
// call says so.
inline bool appendEmitter (std::vector<EmitterTexel>& texels, const EmitterMatrix& world,
                           const std::vector<std::array<float, 3>>& positions,
                           const std::vector<std::uint32_t>& indices,
                           const std::array<float, 3>& emission)
{
    if (! emits (emission)) return true;
    for (const auto index : indices)
        if (index >= positions.size()) return false;
    for (std::size_t t = 0; t + 2 < indices.size(); t += 3)
        for (std::size_t corner = 0; corner < 3; ++corner)
        {
            const auto p = emitterTransform (world, positions[indices[t + corner]]);
            texels.push_back ({ p[0], p[1], p[2], emission[corner] });
        }
    return true;
}

struct EmitterHit
{
    bool hit = false;
    float distance = 0.0f;
    std::array<float, 3> emission {};
};

// The shader's cast: the nearest triangle the ray crosses from either side, past a small
// start distance so the grating does not find itself.
inline EmitterHit nearestEmitter (const std::vector<EmitterTexel>& texels,
                                  const std::array<float, 3>& origin,
                                  const std::array<float, 3>& direction,
                                  float minimum = 1.0e-4f) noexcept
{
    const auto sub = [] (const std::array<float, 3>& a, const std::array<float, 3>& b)
        { return std::array<float, 3> { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; };
    const auto cross = [] (const std::array<float, 3>& a, const std::array<float, 3>& b)
        { return std::array<float, 3> { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] }; };
    const auto dot = [] (const std::array<float, 3>& a, const std::array<float, 3>& b)
        { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    EmitterHit best;
    for (std::size_t t = 0; t + 2 < texels.size(); t += 3)
    {
        const std::array<float, 3> a { texels[t][0], texels[t][1], texels[t][2] };
        const std::array<float, 3> b { texels[t + 1][0], texels[t + 1][1], texels[t + 1][2] };
        const std::array<float, 3> c { texels[t + 2][0], texels[t + 2][1], texels[t + 2][2] };
        const auto e1 = sub (b, a), e2 = sub (c, a);
        const auto p = cross (direction, e2);
        const float det = dot (e1, p);
        if (std::abs (det) <= 1.0e-12f) continue;
        const float inverse = 1.0f / det;
        const auto s = sub (origin, a);
        const float u = dot (s, p) * inverse;
        if (u < 0.0f || u > 1.0f) continue;
        const auto q = cross (s, e1);
        const float v = dot (direction, q) * inverse;
        if (v < 0.0f || u + v > 1.0f) continue;
        const float distance = dot (e2, q) * inverse;
        if (! (distance > minimum) || (best.hit && distance >= best.distance)) continue;
        best.hit = true;
        best.distance = distance;
        best.emission = { texels[t][3], texels[t + 1][3], texels[t + 2][3] };
    }
    return best;
}

// Each band stands for the wavelengths between the midpoints to its neighbours. An order
// sweeps an arc of directions across that interval, so the shader reads the band at
// sub-wavelengths spread over it rather than at the band's centre alone. A single
// wavelength per band lights a band fully or not at all, which splits a lit order into
// eight flat wedges with hard edges and leaves gaps between them.
inline std::array<float, 2> bandInterval (const float* wavelengths, int count, int band) noexcept
{
    const float centre = wavelengths[band];
    const float below = band > 0 ? wavelengths[band - 1] : 2.0f * centre - wavelengths[band + 1];
    const float above = band + 1 < count ? wavelengths[band + 1] : 2.0f * centre - wavelengths[band - 1];
    return { 0.5f * (below + centre), 0.5f * (centre + above) };
}

// A sub-wavelength of band's interval lands on the band centres either side of it,
// weighted by distance, so an order's colour follows its own wavelength rather than
// stepping through eight band hues. Past the first or last centre it all lands on that
// band. realtimeDiffractionDeposit in backend_opengl.cpp is the shader form.
struct TentBands
{
    int lower = 0;
    int upper = 0;
    float upperWeight = 0.0f;
};
inline TentBands tentBands (const float* centres, int count, int band, float wavelength) noexcept
{
    TentBands tent { band, band, 0.0f };
    if (wavelength < centres[band] && band > 0) tent.lower = band - 1;
    else if (wavelength > centres[band] && band + 1 < count) tent.upper = band + 1;
    const float span = centres[tent.upper] - centres[tent.lower];
    if (span > 0.0f) tent.upperWeight = std::fmin (std::fmax ((wavelength - centres[tent.lower]) / span, 0.0f), 1.0f);
    return tent;
}

// The share of the straight piece from d0 to d1 (unit directions a short arc apart) that
// lies inside a sun disc around the unit direction sun, whose edge is at cosine
// cosRadius, and the piece's parameter at the middle of that share. A delta order meets
// a small sun only where its arc crosses the disc, so the sun is integrated over the
// arc in closed form instead of being sampled at a few wavelengths that step over it.
struct SunArc
{
    float share = 0.0f;
    float middle = 0.0f;
};
inline SunArc sunArc (const std::array<float, 3>& d0, const std::array<float, 3>& d1,
                      const std::array<float, 3>& sun, float cosRadius) noexcept
{
    // |d0 - sun + t (d1 - d0)|^2 < 2 (1 - cosRadius) on unit vectors is the angle test.
    const std::array<float, 3> e { d1[0] - d0[0], d1[1] - d0[1], d1[2] - d0[2] };
    const std::array<float, 3> f { d0[0] - sun[0], d0[1] - sun[1], d0[2] - sun[2] };
    const float a = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    const float b = e[0] * f[0] + e[1] * f[1] + e[2] * f[2];
    const float c = f[0] * f[0] + f[1] * f[1] + f[2] * f[2] - 2.0f * (1.0f - cosRadius);
    SunArc arc;
    if (! (a > 1.0e-12f))
    {
        if (c < 0.0f) arc = { 1.0f, 0.5f };
        return arc;
    }
    const float discriminant = b * b - a * c;
    if (! (discriminant > 0.0f)) return arc;
    const float root = std::sqrt (discriminant);
    const float low = std::fmax ((-b - root) / a, 0.0f), high = std::fmin ((-b + root) / a, 1.0f);
    if (! (high > low)) return arc;
    return { high - low, 0.5f * (low + high) };
}

// A rough grating spreads each order over a Gaussian lobe of per-axis width sigma, so a
// small source of edge cosine cosRadius seen through an order is a disc widened to the
// radius whose per-axis spread matches, sqrt(r^2 + (2 sigma)^2), at the same power: the
// host form of realtimeDiffractionLobeDisc. Path Trace spreads it by sampling the lobe.
// Without the spread, a rough sphere's sun streak was a saturated hairline.
struct LobeDisc
{
    float cosRadius = 1.0f;
    float radianceScale = 1.0f;
};
inline LobeDisc lobeDisc (float cosRadius, float sigma) noexcept
{
    if (! (sigma > 0.0f)) return { cosRadius, 1.0f };
    const float radius = std::acos (std::fmin (std::fmax (cosRadius, -1.0f), 1.0f));
    const float cosWide = std::cos (std::fmin (std::sqrt (radius * radius + 4.0f * sigma * sigma), 3.14159265f));
    return { cosWide, (1.0f - cosRadius) / std::fmax (1.0f - cosWide, 1.0e-12f) };
}

// A pressed disc's track (PitStatistics::PressedDisc) in Realtime, the host forms of
// realtimePressedDiscEfficiency and realtimePitScatter in backend_opengl.cpp, which
// follow Path Trace's pressedDiscEfficiency and pit lobe. A pit stripe of
// stripeFraction per track pitch, pit or land along the track with mean one half.
inline float pressedDiscOrderEfficiency (int order, float stripeFraction, float phase) noexcept
{
    const float stepRe = std::cos (phase) - 1.0f, stepIm = std::sin (phase);
    const float meanStripe = 0.5f * stripeFraction;
    if (order == 0)
    {
        const float re = 1.0f + meanStripe * stepRe, im = meanStripe * stepIm;
        return re * re + im * im;
    }
    const float argument = 3.14159265358979f * static_cast<float> (order) * stripeFraction;
    const float amplitude = meanStripe * std::sin (argument) / argument;
    return (stepRe * stepRe + stepIm * stepIm) * amplitude * amplitude;
}
// What the random runs leave out of the orders, stripeFraction q (1 - q) |e^(i phase) - 1|^2
// with q = 1/2. With the orders it sums to the whole reflection (Parseval).
inline float pitScatterPower (float stripeFraction, float phase) noexcept
{
    const float halfSine = std::sin (0.5f * phase);
    return stripeFraction * halfSine * halfSine;
}
// A directional light of irradiance E seen through an order as a disc of angular
// radius r and radiance E / (pi r^2), so the arc that crosses it carries E.
inline float directionalDiscRadiance (float irradiance, float radius) noexcept
{
    return irradiance / (3.14159265358979f * radius * radius);
}
// The share of the straight piece of directions d0 + t (d1 - d0), t in [0, 1], from
// position that reaches triangle abc. Such a direction lies in the cone of the corners
// when its three barycentric triple products keep the cone's sign; each is linear in t,
// so the share is one interval.
inline float chordTriangleShare (const std::array<float, 3>& position,
    const std::array<float, 3>& d0, const std::array<float, 3>& d1,
    const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c) noexcept
{
    const auto sub = [] (const std::array<float, 3>& x, const std::array<float, 3>& y)
    { return std::array<float, 3> { x[0] - y[0], x[1] - y[1], x[2] - y[2] }; };
    const auto cross = [] (const std::array<float, 3>& x, const std::array<float, 3>& y)
    { return std::array<float, 3> { x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0] }; };
    const auto dot = [] (const std::array<float, 3>& x, const std::array<float, 3>& y)
    { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
    const auto A = sub (a, position), B = sub (b, position), C = sub (c, position), e = sub (d1, d0);
    const float det = dot (A, cross (B, C));
    if (std::abs (det) <= 1.0e-12f) return 0.0f;
    const float orientation = det > 0.0f ? 1.0f : -1.0f;
    float low = 0.0f, high = 1.0f;
    for (const auto& plane : { cross (B, C), cross (C, A), cross (A, B) })
    {
        const float p = orientation * dot (plane, d0), q = orientation * dot (plane, e);
        if (std::abs (q) <= 1.0e-12f) { if (p < 0.0f) return 0.0f; }
        else if (q > 0.0f) low = std::max (low, -p / q);
        else high = std::min (high, -p / q);
    }
    return high > low ? high - low : 0.0f;
}
// The normal distribution's CDF in logistic form, within 0.01 everywhere: the shader's
// realtimeSoftEdge.
inline float softEdge (float x) noexcept { return 1.0f / (1.0f + std::exp (-1.702f * x)); }
// The share of a short piece of directions around d0 + (d1 - d0) / 2 that reaches triangle
// abc once a rough grating's lobe, Gaussian with per-axis width sigma, blurs its image, as
// Path Trace's lobe sampling does. Each edge of the corners' cone is a soft step over sigma
// in the sine of the angle to it. A piece is a sub-wavelength's step along an order's arc,
// far shorter than sigma, so its middle stands for it. sigma 0 is the exact chord share.
// Two triangles sharing an edge sum to one across it, so a panel shows no seam.
inline float lobeTriangleShare (const std::array<float, 3>& position,
    const std::array<float, 3>& d0, const std::array<float, 3>& d1,
    const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c, float sigma) noexcept
{
    if (! (sigma > 0.0f)) return chordTriangleShare (position, d0, d1, a, b, c);
    using V = std::array<float, 3>;
    const auto sub = [] (const V& x, const V& y) { return V { x[0] - y[0], x[1] - y[1], x[2] - y[2] }; };
    const auto cross = [] (const V& x, const V& y)
    { return V { x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0] }; };
    const auto dot = [] (const V& x, const V& y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
    const auto A = sub (a, position), B = sub (b, position), C = sub (c, position);
    const float det = dot (A, cross (B, C));
    if (std::abs (det) <= 1.0e-12f) return 0.0f;
    const float orientation = det > 0.0f ? 1.0f : -1.0f;
    V middle { 0.5f * (d0[0] + d1[0]), 0.5f * (d0[1] + d1[1]), 0.5f * (d0[2] + d1[2]) };
    const float length = std::sqrt (dot (middle, middle));
    if (! (length > 0.0f)) return 0.0f;
    for (auto& value : middle) value /= length;
    float share = 1.0f;
    for (const auto& plane : { cross (B, C), cross (C, A), cross (A, B) })
        share *= softEdge (orientation * dot (plane, middle) / std::max (std::sqrt (dot (plane, plane)), 1.0e-30f) / sigma);
    return share;
}
// A triangle's vector irradiance at position for unit radiance, Lambert's polygon form:
// half the sum over its edges of the edge's angle times the unit normal of the plane
// through it and position, turned toward the triangle. The irradiance on a surface with
// unit normal n, where the triangle is wholly above its horizon, is the radiance times
// dot (n, vector). The shader's realtimeEmitterDirect lights ordinary surfaces with it.
inline std::array<float, 3> emitterIrradianceVector (const std::array<float, 3>& position,
    const std::array<float, 3>& a, const std::array<float, 3>& b, const std::array<float, 3>& c) noexcept
{
    using V = std::array<double, 3>;
    const auto unit = [&] (const std::array<float, 3>& corner)
    {
        V v { double (corner[0]) - position[0], double (corner[1]) - position[1], double (corner[2]) - position[2] };
        const double length = std::sqrt (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (auto& x : v) x = length > 0.0 ? x / length : 0.0;
        return v;
    };
    const std::array<V, 3> corners { unit (a), unit (b), unit (c) };
    V sum {};
    for (std::size_t i = 0; i < 3; ++i)
    {
        const auto& p = corners[i];
        const auto& q = corners[(i + 1) % 3];
        const V n { p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0] };
        const double length = std::sqrt (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (! (length > 1.0e-12)) continue;
        const double angle = std::acos (std::clamp (p[0] * q[0] + p[1] * q[1] + p[2] * q[2], -1.0, 1.0));
        for (int k = 0; k < 3; ++k) sum[std::size_t (k)] += 0.5 * angle * n[std::size_t (k)] / length;
    }
    double toward = 0.0;
    for (std::size_t k = 0; k < 3; ++k) toward += sum[k] * (corners[0][k] + corners[1][k] + corners[2][k]);
    const double sign = toward < 0.0 ? -1.0 : 1.0;
    return { float (sign * sum[0]), float (sign * sum[1]), float (sign * sum[2]) };
}
// The smooth groove direction at each vertex, Path Trace's smoothGrooveTangents
// (pathtrace/scene_ingest.cpp) in the object's own space. Each triangle's dP/du is
// constant, so a polar cap read through it is a fan of straight gratings stepping by a
// whole wedge. A vertex sums the unit dP/du of its constant-v edges, which measure dP/du
// at its radius, else of its triangles, each made right-handed against the triangle's
// dP/dv. Returns a unit direction per vertex, or zero where none is defined, such as at
// a cap's hub where the directions cancel.
inline std::vector<std::array<float, 3>> smoothGrooveTangents (const std::vector<std::array<float, 3>>& positions,
    const std::vector<std::array<float, 2>>& uvs, const std::vector<std::uint32_t>& indices)
{
    using V = std::array<float, 3>;
    const auto sub = [] (const V& x, const V& y) { return V { x[0] - y[0], x[1] - y[1], x[2] - y[2] }; };
    const auto cross = [] (const V& x, const V& y)
    { return V { x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0] }; };
    const auto dot = [] (const V& x, const V& y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
    const auto add = [] (V& sum, const V& x, float scale) { for (int i = 0; i < 3; ++i) sum[i] += x[i] * scale; };
    const std::size_t count = std::min (positions.size(), uvs.size());
    std::vector<V> directions (count, V {}), edgeDirections (count, V {});
    std::vector<char> onEdge (count, 0);
    for (std::size_t tri = 0; tri + 3 <= indices.size(); tri += 3)
    {
        const std::array<std::uint32_t, 3> local { indices[tri], indices[tri + 1], indices[tri + 2] };
        if (local[0] >= count || local[1] >= count || local[2] >= count) continue;
        const V edge1 = sub (positions[local[1]], positions[local[0]]), edge2 = sub (positions[local[2]], positions[local[0]]);
        const auto& uv0 = uvs[local[0]]; const auto& uv1 = uvs[local[1]]; const auto& uv2 = uvs[local[2]];
        const float du1 = uv1[0] - uv0[0], dv1 = uv1[1] - uv0[1], du2 = uv2[0] - uv0[0], dv2 = uv2[1] - uv0[1];
        const double area = static_cast<double> (du1) * dv2 - static_cast<double> (du2) * dv1;
        const V normal = cross (edge1, edge2);
        if (! (std::abs (area) > 1.0e-14) || ! (dot (normal, normal) > 0.0f)) continue;
        const float inverse = static_cast<float> (1.0 / area);
        V tangent {}, bitangent {};
        for (int i = 0; i < 3; ++i)
        {
            tangent[i] = (edge1[i] * dv2 - edge2[i] * dv1) * inverse;
            bitangent[i] = (edge2[i] * du1 - edge1[i] * du2) * inverse;
        }
        const float size = std::sqrt (dot (tangent, tangent));
        if (! std::isfinite (size) || ! (size > 0.0f)) continue;
        const float handed = dot (bitangent, cross (normal, tangent)) < 0.0f ? -1.0f : 1.0f;
        for (std::size_t k = 0; k < 3; ++k)
        {
            add (directions[local[k]], tangent, handed / size);
            const std::size_t next = (k + 1) % 3;
            const auto& a = uvs[local[k]]; const auto& b = uvs[local[next]];
            if (std::abs (b[0] - a[0]) > 1.0e-6f && std::abs (b[1] - a[1]) <= 1.0e-6f)
            {
                add (edgeDirections[local[k]], tangent, handed / size);
                add (edgeDirections[local[next]], tangent, handed / size);
                onEdge[local[k]] = onEdge[local[next]] = 1;
            }
        }
    }
    for (std::size_t v = 0; v < count; ++v)
    {
        V sum = onEdge[v] ? edgeDirections[v] : directions[v];
        const float size = std::sqrt (dot (sum, sum));
        // Around a cap's hub the unit directions cancel to rounding noise.
        if (size > 1.0e-3f) for (auto& c : sum) c /= size;
        else sum = V {};
        directions[v] = sum;
    }
    return directions;
}
} // namespace arbitgpu::realtimediffraction
