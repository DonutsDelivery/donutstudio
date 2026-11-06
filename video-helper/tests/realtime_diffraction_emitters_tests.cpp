// The Realtime grating's emitter list (realtime_diffraction_emitters.h): which objects
// it takes, where their triangles land, and the order ray's nearest-emitter cast that
// realtimeDiffractionEmitter in backend_opengl.cpp runs per order.

#include "gpu_backend/realtime_diffraction_emitters.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{
namespace rd = arbitgpu::realtimediffraction;
int failures = 0;
void check (bool condition, const std::string& text)
{
    if (condition) return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", text.c_str());
}
bool near (float a, float b) { return std::abs (a - b) <= 1.0e-5f; }

rd::EmitterMatrix translation (float x, float y, float z)
{
    return { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1 };
}

// A unit square in its local xz plane, facing up y, two triangles. The panels below sit
// above the grating, so their plane is horizontal.
const std::vector<std::array<float, 3>> square { { -0.5f, 0, -0.5f }, { 0.5f, 0, -0.5f }, { 0.5f, 0, 0.5f }, { -0.5f, 0, 0.5f } };
const std::vector<std::uint32_t> squareIndices { 0, 1, 2, 0, 2, 3 };
} // namespace

int main()
{
    // Only glowing objects join, and negative or nonfinite emission is not a glow.
    std::vector<rd::EmitterTexel> texels;
    check (rd::appendEmitter (texels, translation (0, 0, 0), square, squareIndices, { 0, 0, 0 }) && texels.empty(),
           "a dark object adds no triangles");
    check (rd::appendEmitter (texels, translation (0, 0, 0), square, squareIndices, { -1, 2, 2 }) && texels.empty(),
           "negative emission adds no triangles");
    check (rd::appendEmitter (texels, translation (0, 0, 0), square, squareIndices, { NAN, 2, 2 }) && texels.empty(),
           "nonfinite emission adds no triangles");

    // A panel 3 above the grating, glowing warm: six texels, corners in world space,
    // red, green and blue in the three w components.
    check (rd::appendEmitter (texels, translation (0, 3, 0), square, squareIndices, { 4, 2, 1 }), "a panel is admitted");
    check (texels.size() == 6, "two triangles are six texels");
    check (near (texels[0][1], 3.0f) && near (texels[1][0], 0.5f), "corners are moved into world space");
    check (near (texels[0][3], 4.0f) && near (texels[1][3], 2.0f) && near (texels[2][3], 1.0f),
           "the emission rides in the w components");

    // An index past the vertices is malformed geometry and is refused.
    auto refused = texels;
    check (! rd::appendEmitter (refused, translation (0, 0, 0), square, { 0, 1, 7 }, { 1, 1, 1 }) && refused.size() == texels.size(),
           "an index past the vertices is refused without adding triangles");

    // A grating at the origin looking straight up finds the panel 3 away, from below.
    auto hit = rd::nearestEmitter (texels, { 0.1f, 0, 0.1f }, { 0, 1, 0 });
    check (hit.hit && near (hit.distance, 3.0f) && near (hit.emission[0], 4.0f), "an order ray reaches the panel above");
    // An order ray tilted past the panel's edge reads the sky instead.
    const float tilt = 1.0f / std::sqrt (2.0f);
    check (! rd::nearestEmitter (texels, { 0, 0, 0 }, { tilt, tilt, 0 }).hit, "a ray past the panel's edge misses");
    // A ray pointing away misses: the cast does not run backwards.
    check (! rd::nearestEmitter (texels, { 0, 0, 0 }, { 0, -1, 0 }).hit, "a ray away from the panel misses");

    // The nearer of two panels wins, whichever was listed first.
    check (rd::appendEmitter (texels, translation (0, 1.5f, 0), square, squareIndices, { 0.5f, 1, 3 }), "a second panel");
    hit = rd::nearestEmitter (texels, { 0, 0, 0.2f }, { 0, 1, 0 });
    check (hit.hit && near (hit.distance, 1.5f) && near (hit.emission[2], 3.0f), "the nearer panel lights the order");
    // A panel the ray starts on is not found again.
    hit = rd::nearestEmitter (texels, { 0, 1.5f, 0.2f }, { 0, 1, 0 });
    check (hit.hit && near (hit.distance, 1.5f) && near (hit.emission[0], 4.0f), "the start surface is skipped");

    // Band intervals meet at midpoints and the end bands are as wide as their neighbour.
    const float wavelengths[3] { 400.0f, 450.0f, 520.0f };
    const auto first = rd::bandInterval (wavelengths, 3, 0), middle = rd::bandInterval (wavelengths, 3, 1),
               last = rd::bandInterval (wavelengths, 3, 2);
    check (near (first[0], 375.0f) && near (first[1], 425.0f), "the first band reaches as far below as above");
    check (near (middle[0], 425.0f) && near (middle[1], 485.0f), "a middle band runs between midpoints");
    check (near (last[0], 485.0f) && near (last[1], 555.0f), "the last band reaches as far above as below");

    // A sub-wavelength between two centres splits between them by distance; one past the
    // last centre stays on the last band; one on a centre is all that band's.
    auto tent = rd::tentBands (wavelengths, 3, 1, 470.0f);
    check (tent.lower == 1 && tent.upper == 2 && near (tent.upperWeight, 20.0f / 70.0f), "above a centre leans to the next band");
    tent = rd::tentBands (wavelengths, 3, 1, 440.0f);
    check (tent.lower == 0 && tent.upper == 1 && near (tent.upperWeight, 0.8f), "below a centre leans to the previous band");
    tent = rd::tentBands (wavelengths, 3, 2, 540.0f);
    check (tent.lower == 2 && tent.upper == 2 && tent.upperWeight == 0.0f, "past the last centre stays on the last band");
    tent = rd::tentBands (wavelengths, 3, 0, 400.0f);
    check (tent.lower == 0 && tent.upper == 0 && tent.upperWeight == 0.0f, "a centre is its own band's");

    // A sun of radius 2 degrees straight up. An arc of 8 degrees across its centre is
    // inside it for half its length, centred on the middle.
    const float radius = 2.0f * 3.14159265f / 180.0f, cosRadius = std::cos (radius);
    const auto at = [] (float angle) { return std::array<float, 3> { std::sin (angle), std::cos (angle), 0.0f }; };
    auto arc = rd::sunArc (at (-2 * radius), at (2 * radius), { 0, 1, 0 }, cosRadius);
    check (std::abs (arc.share - 0.5f) < 2.0e-3f && std::abs (arc.middle - 0.5f) < 1.0e-3f, "an arc through the sun is lit for its chord");
    // An arc starting at the centre is inside for its first quarter.
    arc = rd::sunArc (at (0), at (4 * radius), { 0, 1, 0 }, cosRadius);
    check (std::abs (arc.share - 0.25f) < 2.0e-3f && std::abs (arc.middle - 0.125f) < 2.0e-3f, "an arc from the centre is lit to the edge");
    // An arc wholly beside the sun gets nothing, and an arc wholly inside gets all of it.
    check (rd::sunArc (at (3 * radius), at (5 * radius), { 0, 1, 0 }, cosRadius).share == 0.0f, "an arc beside the sun misses");
    check (std::abs (rd::sunArc (at (-0.5f * radius), at (0.5f * radius), { 0, 1, 0 }, cosRadius).share - 1.0f) < 1.0e-6f,
           "an arc inside the sun is wholly lit");
    // A point arc inside the sun counts whole and one outside counts nothing.
    check (rd::sunArc (at (0.1f * radius), at (0.1f * radius), { 0, 1, 0 }, cosRadius).share == 1.0f, "a point inside is lit");
    check (rd::sunArc (at (3 * radius), at (3 * radius), { 0, 1, 0 }, cosRadius).share == 0.0f, "a point outside is not");

    // Through a rough grating's order the sun is a wider disc of the same power: the
    // Grooved Sphere's 2 degree sun under an rms slope of 0.08 (sigma 0.16).
    const auto smooth = rd::lobeDisc (cosRadius, 0.0f);
    check (smooth.cosRadius == cosRadius && smooth.radianceScale == 1.0f, "a smooth grating keeps the sun's disc");
    const auto rough = rd::lobeDisc (cosRadius, 0.16f);
    const float wide = std::acos (rough.cosRadius);
    check (std::abs (wide - std::sqrt (radius * radius + 4.0f * 0.16f * 0.16f)) < 1.0e-4f, "the lobe widens the disc");
    check (std::abs (rough.radianceScale * (1.0f - rough.cosRadius) - (1.0f - cosRadius)) < 1.0e-7f,
           "the widened disc keeps the sun's power");
    // An order's arc across the sun's edge region is now lit, so its wavelengths spread
    // over the width the lobe gives instead of the bare disc.
    check (rd::sunArc (at (3 * radius), at (5 * radius), { 0, 1, 0 }, rough.cosRadius).share == 1.0f,
           "an arc beside the bare disc lies inside the lobe's disc");

    // A pressed track's orders and its pit scatter make up the whole reflection, at any
    // pit depth; the scatter vanishes at a whole-wave pit and peaks at a half-wave one.
    for (const float phase : { 0.4f, 1.7f, 3.14159265f, 5.0f })
    {
        double sum = rd::pitScatterPower (0.3125f, phase);
        for (int order = -20000; order <= 20000; ++order) sum += rd::pressedDiscOrderEfficiency (order, 0.3125f, phase);
        check (std::abs (sum - 1.0) < 2.0e-3, "pressed-disc orders and pit scatter sum to the reflection");
    }
    check (rd::pitScatterPower (0.3125f, 2.0f * 3.14159265f) < 1.0e-10f, "a whole-wave pit scatters nothing");
    check (near (rd::pitScatterPower (0.5f, 3.14159265f), 0.5f), "a half-wave pit at half duty scatters half");
    // A directional light's disc integrates back to its irradiance over its solid angle.
    const float lightRadius = 0.03f;
    check (std::abs (rd::directionalDiscRadiance (2.0f, lightRadius) * 3.14159265f * lightRadius * lightRadius - 2.0f) < 1.0e-5f,
           "a directional light's disc carries its irradiance");

    // An arc piece sweeping across a panel covers it for exactly the share between the
    // panel's edges: the piece runs from x = -1 to x = 1 at distance 1, the triangle spans
    // x in [-0.5, 0.5] where the piece passes, and a piece behind the origin sees nothing.
    {
        const std::array<float, 3> origin { 0, 0, 0 }, a { -0.5f, -1, 1 }, b { 0.5f, -1, 1 }, c { 0.5f, 1, 1 };
        const std::array<float, 3> a2 { -0.5f, -1, 1 }, c2 { -0.5f, 1, 1 };
        const float lower = rd::chordTriangleShare (origin, { -1, 0, 1 }, { 1, 0, 1 }, a, b, c);
        const float upper = rd::chordTriangleShare (origin, { -1, 0, 1 }, { 1, 0, 1 }, a2, c, c2);
        check (near (lower + upper, 0.5f), "an arc piece crosses a panel for the share between its edges");
        check (rd::chordTriangleShare (origin, { -1, 0, -1 }, { 1, 0, -1 }, a, b, c) == 0.0f,
               "a panel behind the arc is not seen");
        check (near (rd::chordTriangleShare (origin, { 0, 0, 1 }, { 0.25f, 0, 1 }, a, b, c)
                + rd::chordTriangleShare (origin, { 0, 0, 1 }, { 0.25f, 0, 1 }, a2, c, c2), 1.0f),
               "a piece inside the panel is wholly covered");
    }

    // A 64-segment polar annulus, the Grooved Disc's cap: each vertex's groove direction
    // is the circle's tangent there, which a triangle's own dP/du misses by half a wedge.
    {
        const int segments = 64;
        std::vector<std::array<float, 3>> positions;
        std::vector<std::array<float, 2>> uvs;
        std::vector<std::uint32_t> indices;
        for (int ring = 0; ring < 2; ++ring)
            for (int k = 0; k <= segments; ++k)
            {
                const float u = float (k) / float (segments), r = ring ? 0.6f : 0.3f, angle = 6.2831853f * u;
                positions.push_back ({ r * std::cos (angle), r * std::sin (angle), 0.0f });
                uvs.push_back ({ u, r });
            }
        for (std::uint32_t k = 0; k < std::uint32_t (segments); ++k)
        {
            const std::uint32_t a = k, b = k + 1, c = std::uint32_t (segments) + 1 + k, d = c + 1;
            indices.insert (indices.end(), { a, b, d, a, d, c });
        }
        const auto grooves = rd::smoothGrooveTangents (positions, uvs, indices);
        float worst = 1.0f;
        for (int k = 1; k < segments; ++k)
            for (int ring = 0; ring < 2; ++ring)
            {
                const auto& t = grooves[std::size_t (ring * (segments + 1) + k)];
                const float angle = 6.2831853f * float (k) / float (segments);
                worst = std::min (worst, std::abs (-std::sin (angle) * t[0] + std::cos (angle) * t[1]));
            }
        check (worst > 0.9999f, "a polar cap's groove directions follow the circles at every vertex");
    }

    // A rough grating's lobe blurs a panel's image: the share is a soft step at each edge,
    // a half at the edge, the hard share deep inside or far outside, and a panel's two
    // triangles sum to one across their shared diagonal, so the image has no seam.
    {
        const std::array<float, 3> origin { 0, 0, 0 }, a { -0.5f, -0.5f, 1 }, b { 0.5f, -0.5f, 1 }, c { 0.5f, 0.5f, 1 },
            d { -0.5f, 0.5f, 1 };
        const float sigma = 0.04f;
        const auto share = [&] (float x, float y)
        {
            const std::array<float, 3> d0 { x - 1.0e-4f, y, 1 }, d1 { x + 1.0e-4f, y, 1 };
            return rd::lobeTriangleShare (origin, d0, d1, a, b, c, sigma) + rd::lobeTriangleShare (origin, d0, d1, a, c, d, sigma);
        };
        check (share (0.0f, 0.0f) > 0.999f, "the middle of a panel is wholly covered through the lobe");
        check (share (2.0f, 0.0f) < 1.0e-4f, "a direction far off a panel sees none of it through the lobe");
        check (std::abs (share (0.5f, 0.0f) - 0.5f) < 0.02f,
               "a panel's edge is half covered through the lobe");
        check (std::abs (share (0.1f, 0.1f) - share (0.1001f, 0.0999f)) < 1.0e-3f
                   && share (0.1f, 0.1f) > 0.999f,
               "a panel's two triangles meet across their diagonal with no seam");
        check (share (0.5f + 0.02f, 0.0f) > 0.05f && share (0.5f + 0.02f, 0.0f) < 0.45f,
               "the edge softens over the lobe's width");
        const std::array<float, 3> d0 { 0.3f - 1.0e-4f, 0.1f, 1 }, d1 { 0.3f + 1.0e-4f, 0.1f, 1 };
        check (rd::lobeTriangleShare (origin, d0, d1, a, b, c, 0.0f) == rd::chordTriangleShare (origin, d0, d1, a, b, c),
               "a smooth grating keeps the exact chord share");
    }

    // A panel's vector irradiance: a small square of side s facing a point at distance 1
    // gives about s^2, its two triangles' vectors add, and the winding does not matter.
    {
        const float s = 0.02f;
        const std::array<float, 3> origin { 0, 0, 0 }, a { -s / 2, -s / 2, 1 }, b { s / 2, -s / 2, 1 }, c { s / 2, s / 2, 1 },
            d { -s / 2, s / 2, 1 };
        const auto first = rd::emitterIrradianceVector (origin, a, b, c), second = rd::emitterIrradianceVector (origin, a, c, d);
        const auto flipped = rd::emitterIrradianceVector (origin, a, c, b);
        check (std::abs ((first[2] + second[2]) / (s * s) - 1.0f) < 1.0e-3f, "a small facing panel gives its area over the distance squared");
        check (std::abs (first[0] + second[0]) < 1.0e-7f && std::abs (first[1] + second[1]) < 1.0e-7f,
               "a centred panel's irradiance points at it");
        check (near (flipped[2], first[2]), "a panel lights from either winding");
        // A large square of half side 1 at height 1 on its axis: the form factor of a
        // parallel square, 4 x the corner rectangle's, times pi.
        const std::array<float, 3> A { -1, -1, 1 }, B { 1, -1, 1 }, C { 1, 1, 1 }, D { -1, 1, 1 };
        const float large = rd::emitterIrradianceVector (origin, A, B, C)[2] + rd::emitterIrradianceVector (origin, A, C, D)[2];
        const double x = 1.0, root = std::sqrt (1.0 + x * x);
        const double corner = (x / root * std::atan (x / root) + x / root * std::atan (x / root)) / (2.0 * 3.14159265358979);
        check (std::abs (large - 4.0 * corner * 3.14159265358979) < 1.0e-4, "a large parallel panel matches the form factor");
    }

    if (failures != 0) return 1;
    std::puts ("realtime diffraction emitters: ok");
    return 0;
}
