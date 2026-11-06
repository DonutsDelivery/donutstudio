// Realtime's grating mirror (realtime_diffraction_mirror.h): the zero order's weight the
// host tabulates from the efficiency table, the object records the pass shades its hits
// with, and the world point it rebuilds from a gather texel's linear depth.

#include "gpu_backend/realtime_diffraction_mirror.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{
namespace mirror = arbitgpu::realtimediffractionmirror;
namespace eff = arbitgpu::realtimediffractionefficiency;
int failures = 0;
void check (bool condition, const std::string& text)
{
    if (condition) return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", text.c_str());
}
bool near (float a, float b, float tolerance = 1.0e-5f) { return std::abs (a - b) <= tolerance; }

// A one-dimensional grating to order 1 (three orders) whose zero order tile holds
// `zero` and whose band sums hold `sum` at every node, over eight bands.
eff::Table flatTable (float zero, float sum)
{
    eff::Table table;
    table.lastOrder = 1;
    table.crossed = false;
    table.orders = 3;
    table.width = eff::kTilesPerRow * eff::kGrid;
    const int tiles = 8 * (table.orders + 1);
    table.height = ((tiles + eff::kTilesPerRow - 1) / eff::kTilesPerRow) * eff::kGrid;
    table.texels.assign (std::size_t (table.width) * std::size_t (table.height), 0.0f);
    for (int band = 0; band < 8; ++band)
        for (int order : { table.lastOrder, table.orders })
        {
            const int tile = eff::tileIndex (band, order, table.orders);
            for (int j = 0; j < eff::kGrid; ++j)
                for (int i = 0; i < eff::kGrid; ++i)
                    table.texels[std::size_t ((tile / eff::kTilesPerRow) * eff::kGrid + j) * std::size_t (table.width)
                        + std::size_t ((tile % eff::kTilesPerRow) * eff::kGrid + i)] = order == table.orders ? sum : zero;
        }
    return table;
}

// An uncoated metal with a flat spectrum whose bands each map to equal X, Y and Z.
diffractionmaterial::PhysicalDiffractionGpuParameters metal (float roughness)
{
    diffractionmaterial::PhysicalDiffractionGpuParameters m {};
    m.control = { 1.0f, 0.0f, 0.0f, 1.0f };
    m.microstructure = { 0.5f, 0.2f, 3.0f, 0.0f };
    m.roughness = { roughness, 0.0f, 0.0f, 0.0f };
    for (std::size_t b = 0; b < 8; ++b)
    {
        m.spectral[b] = { 400.0f + 40.0f * float (b), 0.0f, 1.0f, 1.0f };
        m.spectralZ[b] = { 1.0f, 1.0f, 0.0f, 0.0f };
    }
    return m;
}
} // namespace

int main()
{
    // The bilinear read lands on a node's value and halfway between two nodes.
    {
        auto table = flatTable (0.0f, 0.0f);
        const int tile = eff::tileIndex (2, table.lastOrder, table.orders);
        const auto at = [&] (int i, int j) -> float& {
            return table.texels[std::size_t ((tile / eff::kTilesPerRow) * eff::kGrid + j) * std::size_t (table.width)
                + std::size_t ((tile % eff::kTilesPerRow) * eff::kGrid + i)];
        };
        at (31, 15) = 1.0f; at (31, 16) = 3.0f;
        check (near (mirror::tableRead (table, tile, 1.0f, 0.0f), 2.0f), "the read at y 0 is the mean of the two rows around it");
        at (0, 0) = 5.0f;
        check (near (mirror::tableRead (table, tile, -1.0f, -1.0f), 5.0f), "the corner node reads its own value");
    }

    // An uncoated metal reflects ((n-1)^2 + k^2) / ((n+1)^2 + k^2).
    const auto smooth = metal (0.0f);
    const float reflectance = (0.64f + 9.0f) / (1.44f + 9.0f);
    check (near (mirror::spectralReflectance (smooth, 500.0f), reflectance), "the uncoated reflectance is the Fresnel normal incidence");
    auto unknown = smooth;
    unknown.control.w = 7.0f;
    check (mirror::spectralReflectance (unknown, 500.0f) < 0.0f, "an unknown coating model has no reflectance");

    // A flat band spectrum maps to grey in the shader's XYZ to linear sRGB matrix.
    {
        std::array<float, 8> bands {};
        bands.fill (0.5f);
        const auto rgb = mirror::bandsToRgb (smooth, bands);
        check (near (rgb[0], 0.5f * (3.2406f - 1.5372f - 0.4986f), 1.0e-4f)
               && near (rgb[1], 0.5f * (-0.9689f + 1.8758f + 0.0415f), 1.0e-4f), "bands go to RGB through the shader's matrix");
    }

    // The weight is the zero order's efficiency over the band sum where the sum passes 1,
    // times the reflectance, at every incidence.
    {
        const auto weights = mirror::weights (flatTable (0.6f, 1.5f), smooth);
        check (weights.size() == std::size_t (mirror::kWeightSamples), "one weight per cosine sample");
        std::array<float, 8> bands {};
        bands.fill (0.6f / 1.5f * reflectance);
        const auto expected = mirror::bandsToRgb (smooth, bands);
        check (near (weights.front()[1], expected[1], 1.0e-4f) && near (weights.back()[1], expected[1], 1.0e-4f),
               "a normalised zero order weighs efficiency / sum * reflectance");
        const auto unnormalised = mirror::weights (flatTable (0.6f, 0.5f), smooth);
        bands.fill (0.6f * reflectance);
        check (near (unnormalised.back()[1], mirror::bandsToRgb (smooth, bands)[1], 1.0e-4f),
               "a band sum under 1 does not raise the zero order");
    }

    // Roughness takes exp(-(4 pi sigma cos / lambda)^2) at normal view and nothing at grazing.
    {
        const auto rough = metal (20.0f);
        const auto weights = mirror::weights (flatTable (1.0f, 1.0f), rough);
        std::array<float, 8> bands {};
        for (std::size_t b = 0; b < 8; ++b)
        {
            const float h = 4.0f * 3.14159265f * 20.0f / rough.spectral[b].x;
            bands[b] = std::exp (-h * h) * reflectance;
        }
        check (near (weights.back()[1], mirror::bandsToRgb (rough, bands)[1], 1.0e-4f), "roughness dims the mirror at normal view");
        bands.fill (reflectance);
        check (near (weights.front()[1], mirror::bandsToRgb (rough, bands)[1], 1.0e-4f), "roughness leaves the grazing mirror whole");
        const auto none = mirror::weights (flatTable (1.0f, 1.0f), unknown);
        check (none.back()[0] == 0.0f && none.back()[1] == 0.0f && none.back()[2] == 0.0f, "an unknown coating has no mirror");
    }

    // Each object is two texels: its id in 16-bit halves, metallic and glow, then base colour.
    {
        std::vector<mirror::Texel> records;
        mirror::appendObject (records, 1070032u, { 0.12f, 0.35f, 0.1f }, 0.0f, false);
        mirror::appendObject (records, 1070079u, { -1.0f, NAN, 2.0f }, 4.0f, true);
        check (records.size() == 4, "two objects are four texels");
        check (records[0][0] + records[0][1] * 65536.0f == 1070032.0f, "the id splits into exact 16-bit halves");
        check (near (records[1][1], 0.35f) && records[0][3] == 0.0f, "the base colour and no glow");
        check (records[2][2] == 1.0f && records[2][3] == 1.0f, "metallic is clamped and a glowing object is marked");
        check (records[3][0] == 0.0f && records[3][1] == 0.0f && near (records[3][2], 2.0f), "negative and nonfinite colour lanes are zero");
    }

    // The world point comes back from realtimeProject's NDC and linear depth, for a
    // perspective and an orthographic camera.
    {
        const float half = 0.3f;
        const std::array<float, 4> rotation { std::sin (half) * 0.6f, std::sin (half) * 0.8f, 0.0f, std::cos (half) };
        const std::array<float, 3> translation { 1.0f, 2.0f, 8.0f };
        const std::array<float, 3> world { 0.4f, 0.7f, -1.5f };
        const auto toCamera = [&] (const std::array<float, 3>& p) {
            const std::array<float, 3> v { p[0] - translation[0], p[1] - translation[1], p[2] - translation[2] };
            const std::array<float, 3> q { -rotation[0], -rotation[1], -rotation[2] };
            const auto cross = [] (const std::array<float, 3>& a, const std::array<float, 3>& b) {
                return std::array<float, 3> { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
            };
            auto inner = cross (q, v);
            for (int i = 0; i < 3; ++i) inner[std::size_t (i)] += rotation[3] * v[std::size_t (i)];
            const auto outer = cross (q, inner);
            return std::array<float, 3> { v[0] + 2.0f * outer[0], v[1] + 2.0f * outer[1], v[2] + 2.0f * outer[2] };
        };
        const auto camera = toCamera (world);
        const float distance = -camera[2];
        for (const std::array<float, 4> projection : { std::array<float, 4> { 0.7f, 1.7778f, 0.1f, 50.0f },
                                                       std::array<float, 4> { -3.0f, 1.7778f, 0.1f, 50.0f } })
        {
            const bool ortho = projection[0] < 0.0f;
            const float ndcX = ortho ? camera[0] / (-projection[0] * projection[1]) : camera[0] / (projection[0] * projection[1]) / distance;
            const float ndcY = ortho ? camera[1] / -projection[0] : camera[1] / projection[0] / distance;
            const float depth = (distance - projection[2]) / (projection[3] - projection[2]);
            const auto back = mirror::gatherWorld (rotation, translation, projection, ndcX, ndcY, depth);
            check (near (back[0], world[0], 1.0e-4f) && near (back[1], world[1], 1.0e-4f) && near (back[2], world[2], 1.0e-4f),
                   ortho ? "an orthographic gather texel rebuilds its world point" : "a perspective gather texel rebuilds its world point");
        }
    }

    if (failures != 0)
    {
        std::fprintf (stderr, "realtime diffraction mirror: %d failures\n", failures);
        return 1;
    }
    std::printf ("realtime diffraction mirror: ok\n");
    return 0;
}
