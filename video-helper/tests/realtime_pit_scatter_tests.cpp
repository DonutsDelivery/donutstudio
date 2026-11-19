// The Realtime pressed-disc pit lobe on a glowing triangle (pitTriangleMass in
// realtime_diffraction_emitters.h, realtimePitTriangleMass in backend_opengl.cpp and
// backend_sokol.mm), against a brute-force integral of Path Trace's pit density over the
// directions that reach the triangle. The scene is Grooved Disc Diffraction's: a far
// camera 46 degrees up, and one long lamp above and behind the disc.

#include "gpu_backend/realtime_diffraction_emitters.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{
namespace rd = arbitgpu::realtimediffraction;
using V = std::array<float, 3>;
constexpr float pi = 3.14159265358979f;
int failures = 0;
void check (bool condition, const std::string& text)
{
    if (condition) return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", text.c_str());
}

double dot (const V& a, const V& b) { return double (a[0]) * b[0] + double (a[1]) * b[1] + double (a[2]) * b[2]; }

// Moller-Trumbore, either face.
bool hits (const V& origin, const V& d, const V& a, const V& b, const V& c)
{
    const double e1[3] { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    const double p[3] { d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0] };
    const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (std::abs (det) < 1.0e-12) return false;
    const double s[3] { origin[0] - a[0], origin[1] - a[1], origin[2] - a[2] };
    const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
    if (u < 0.0 || u > 1.0) return false;
    const double q[3] { s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0] };
    const double v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
    if (v < 0.0 || u + v > 1.0) return false;
    return (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det > 0.0;
}

// The lobe's mass on the triangle on a fine grid of the plane of tangents: sinc^2 over
// the reach across, normalised as the sky nodes are, times the Lorentzian along.
double bruteForce (const V& position, const rd::PitLobe& lobe, const V& a, const V& b, const V& c,
                   double yLow, double yHigh)
{
    const double along[2] { -lobe.across[1], lobe.across[0] };
    const int uSteps = 3000, ySteps = 6000;
    const double du = 2.0 * rd::kPitScatterReach / uSteps, dy = (yHigh - yLow) / ySteps;
    double sum = 0.0;
    for (int i = 0; i < uSteps; ++i)
    {
        const double u = -rd::kPitScatterReach + (i + 0.5) * du;
        const double sinc = std::abs (u) > 1.0e-9 ? std::sin (pi * u) / (pi * u) : 1.0;
        const double f = sinc * sinc / rd::kPitScatterAcrossMass;
        for (int j = 0; j < ySteps; ++j)
        {
            const double y = yLow + (j + 0.5) * dy;
            const double x0 = lobe.mirror[0] + lobe.across[0] * lobe.acrossWidth * u + along[0] * y;
            const double x1 = lobe.mirror[1] + lobe.across[1] * lobe.acrossWidth * u + along[1] * y;
            const double z2 = 1.0 - x0 * x0 - x1 * x1;
            if (! (z2 > 0.0)) continue;
            const double z = std::sqrt (z2);
            V d {};
            for (int k = 0; k < 3; ++k) d[k] = float (x0 * lobe.tangent[k] + x1 * lobe.bitangent[k] + z * lobe.normal[k]);
            if (! hits (position, d, a, b, c)) continue;
            const double g = lobe.alongWidth / (pi * (y * y + double (lobe.alongWidth) * lobe.alongWidth));
            sum += f * g * du * dy;
        }
    }
    return sum;
}

// The sky nodes' old emitter read: three lines along the tracks at the Lorentzian's
// sixths, each cut across into twelve pieces, a piece counting the share of it the
// triangle covers (chordTriangleShare).
double threeLines (const V& position, const rd::PitLobe& lobe, const V& a, const V& b, const V& c)
{
    const float along[2] { -lobe.across[1], lobe.across[0] };
    double light = 0.0, total = 0.0;
    for (int line = 0; line < 3; ++line)
    {
        const float offset = lobe.alongWidth * 0.57735027f * float (line - 1);
        const float width = 2.0f * rd::kPitScatterReach / 12.0f;
        for (int k = 0; k < 12; ++k)
        {
            const float x0 = -rd::kPitScatterReach + float (k) * width, x1 = x0 + width, middle = 0.5f * (x0 + x1);
            const float sinc = std::sin (pi * middle) / (pi * middle);
            const auto direction = [&] (float x)
            {
                const float t0 = lobe.mirror[0] + along[0] * offset + lobe.across[0] * lobe.acrossWidth * x;
                const float t1 = lobe.mirror[1] + along[1] * offset + lobe.across[1] * lobe.acrossWidth * x;
                const float z = std::sqrt (std::max (0.0f, 1.0f - t0 * t0 - t1 * t1));
                V d {};
                for (int i = 0; i < 3; ++i) d[i] = t0 * lobe.tangent[i] + t1 * lobe.bitangent[i] + z * lobe.normal[i];
                return d;
            };
            light += sinc * sinc * width * rd::chordTriangleShare (position, direction (x0), direction (x1), a, b, c);
            total += sinc * sinc * width;
        }
    }
    return light / total;
}

struct Case
{
    const char* name;
    V tangent, bitangent;
    V position;
};
} // namespace

int main()
{
    // Grooved Disc Diffraction's lamp: a panel 80 wide along x and 3 deep, flat at height
    // 90.9 and centred 41.6 behind the hub. Two triangles.
    const V p0 { -40.0f, 90.9f, -43.1f }, p1 { 40.0f, 90.9f, -43.1f }, p2 { 40.0f, 90.9f, -40.1f }, p3 { -40.0f, 90.9f, -40.1f };
    const V normal { 0.0f, 1.0f, 0.0f };
    // The far camera 57.5 up and 55 out.
    const double viewLength = std::sqrt (57.5 * 57.5 + 55.0 * 55.0);
    const V toView { 0.0f, float (57.5 / viewLength), float (55.0 / viewLength) };
    // 550 nm on the 1600 nm track with 500 nm pits: across width 550 / 500 in the plane of
    // tangents, along width 550 / (pi 1354 nm mean run).
    const float acrossWidth = 550.0f / 500.0f, alongWidth = 550.0f / (pi * 1354.0f);
    // The tangent is across the tracks (radial) and the bitangent n x t. On the disc's left
    // the tracks run toward the camera, so the lamp sits along them from the mirror; at
    // its far side they run across the view and the lamp lies across them.
    const Case cases[] {
        { "left of the hub", { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, { -0.5f, 0.0025f, 0.0f } },
        { "beyond the hub", { 0.0f, 0.0f, -1.0f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 0.0025f, -0.5f } },
    };
    for (const auto& test : cases)
    {
        rd::PitLobe lobe;
        lobe.tangent = test.tangent;
        lobe.bitangent = test.bitangent;
        lobe.normal = normal;
        lobe.mirror = { -float (dot (toView, test.tangent)), -float (dot (toView, test.bitangent)) };
        lobe.across = { 1.0f, 0.0f };
        lobe.acrossWidth = acrossWidth;
        lobe.alongWidth = alongWidth;
        const auto first = rd::pitTriangleMass (test.position, lobe, p0, p1, p2);
        const auto second = rd::pitTriangleMass (test.position, lobe, p0, p2, p3);
        const double mass = double (first.mass) + second.mass;
        const double reference = bruteForce (test.position, lobe, p0, p1, p2, -1.0, 1.0)
            + bruteForce (test.position, lobe, p0, p2, p3, -1.0, 1.0);
        const double old = threeLines (test.position, lobe, p0, p1, p2) + threeLines (test.position, lobe, p0, p2, p3);
        std::printf ("%s: triangle mass %.5f, brute force %.5f, three lines %.5f\n", test.name, mass, reference, old);
        check (reference > 1.0e-3, std::string (test.name) + ": the lamp holds a real share of the lobe");
        check (std::abs (mass - reference) <= 0.05 * reference, std::string (test.name) + ": the triangle mass matches the integral");
        const auto aimed = first.mass >= second.mass ? first.direction : second.direction;
        const auto& [a, b, c] = first.mass >= second.mass ? std::array<V, 3> { p0, p1, p2 } : std::array<V, 3> { p0, p2, p3 };
        check (hits (test.position, aimed, a, b, c), std::string (test.name) + ": the shadow ray aims into the triangle");
        if (std::string (test.name) == "left of the hub")
            check (old < 0.1 * reference, "left of the hub: three lines along the tracks miss the lamp");
    }
    // A triangle behind the surface takes none of the lobe.
    {
        rd::PitLobe lobe;
        lobe.tangent = { 1.0f, 0.0f, 0.0f };
        lobe.bitangent = { 0.0f, 0.0f, -1.0f };
        lobe.normal = normal;
        lobe.mirror = { 0.0f, -float (dot (toView, lobe.bitangent)) };
        lobe.across = { 1.0f, 0.0f };
        lobe.acrossWidth = acrossWidth;
        lobe.alongWidth = alongWidth;
        const V below0 { -40.0f, -90.9f, -43.1f }, below1 { 40.0f, -90.9f, -43.1f }, below2 { 40.0f, -90.9f, -40.1f };
        check (rd::pitTriangleMass ({ 0.0f, 0.0f, 0.0f }, lobe, below0, below1, below2).mass == 0.0f,
               "a lamp below the horizon takes none of the lobe");
    }
    if (failures == 0) std::printf ("realtime pit scatter tests passed\n");
    return failures == 0 ? 0 : 1;
}
