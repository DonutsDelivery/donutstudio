#pragma once

// Realtime's lattice order efficiencies, tabulated once per grating. With a uniform
// groove field an order's efficiency depends only on the band and the incident
// direction's tangent xy, so each (band, order) is one kGrid x kGrid tile over xy in
// [-1, 1]^2, read with bilinear filtering. Every band ends with one more tile: the
// sum over its propagating orders of efficiency * exp(-h^2) * z / incident z, the
// normalisation diffractionTriangleBsdf divides by. Values come from Path Trace's
// latticeEfficiency (diffraction_order_core.h). Evaluating the third family's 27
// profile amplitudes per order in every receiver gather row and every sky
// sub-wavelength cost the hexagonal orb's frame tens of seconds.

#include "../diffraction_order_core.h"
#include "../../../shared/DiffractionMaterialGpuLayout.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace arbitgpu::realtimediffractionefficiency
{
inline constexpr int kGrid = 32;
inline constexpr int kTilesPerRow = 64;
// A node past the unit disc stands for the incident direction 0.999 of the way out,
// so filtering near the horizon reads the same smooth field.
inline constexpr float kRimRadius = 0.999f;

struct Table
{
    int lastOrder = 0;
    bool crossed = false;
    int orders = 0;
    int width = 0, height = 0;
    std::vector<float> texels;
};

// The shader's resolveGrooveGeometry returns the material's own geometry at every uv.
inline bool uniformGrooveField (const diffractionmaterial::PhysicalDiffractionGpuParameters& m, bool foilField) noexcept
{
    if (foilField) return false;
    if (static_cast<int> (m.grooveField.x + 0.5f) == 1) return true;
    return m.grooveVariation.z == 0.0f && m.grooveVariation.w == 0.0f && m.grooveField.w == 0.0f;
}

inline int orderCount (bool crossed, int lastOrder) noexcept
{
    const int side = 2 * lastOrder + 1;
    return crossed ? side * side : side;
}

// The shader's tile for band b and order k (k == orders is the band's sum).
inline int tileIndex (int band, int order, int orders) noexcept { return band * (orders + 1) + order; }

inline std::shared_ptr<const Table> build (const diffractionmaterial::PhysicalDiffractionGpuParameters& m)
{
    using namespace diffractionmaterial;
    using namespace diffractionmaterial::ordercore;
    auto table = std::make_shared<Table>();
    const bool crossed = static_cast<int> (m.secondaryGeometry.w + 0.5f) == 2;
    const int firstOrder = static_cast<int> (std::lround (m.microstructure.w));
    const int lastOrder = static_cast<int> (m.control.x + 0.5f);
    const auto profile = static_cast<GrooveProfile> (static_cast<int> (m.control.y + 0.5f));
    const bool thirdFamily = m.control.z > 0.5f;
    const int orders = orderCount (crossed, lastOrder);
    const int tiles = 8 * (orders + 1);
    table->lastOrder = lastOrder;
    table->crossed = crossed;
    table->orders = orders;
    table->width = kTilesPerRow * kGrid;
    table->height = ((tiles + kTilesPerRow - 1) / kTilesPerRow) * kGrid;
    table->texels.assign (std::size_t (table->width) * std::size_t (table->height), 0.0f);
    constexpr float pi = 3.14159265358979f;
    // The amplitudes a third-family order reads reach lastOrder + the convolution radius.
    const int reach = lastOrder + kThirdFamilyConvolutionRadius;
    const auto bandWork = [&] (int band) {
        const float wavelength = m.spectral[std::size_t (band)].x;
        std::vector<Complex> amplitudes (std::size_t (2 * reach + 1));
        for (int j = 0; j < kGrid; ++j)
            for (int i = 0; i < kGrid; ++i)
            {
                float x = -1.0f + 2.0f * float (i) / float (kGrid - 1), y = -1.0f + 2.0f * float (j) / float (kGrid - 1);
                const float radius = std::sqrt (x * x + y * y);
                if (radius > kRimRadius) { x *= kRimRadius / radius; y *= kRimRadius / radius; }
                const float incidentZ = std::sqrt (std::max (0.0f, 1.0f - x * x - y * y));
                float sum = 0.0f;
                for (int primary = -lastOrder; primary <= lastOrder; ++primary)
                {
                    const int limit = crossed ? lastOrder : 0;
                    for (int secondary = -limit; secondary <= limit; ++secondary)
                    {
                        const int k = crossed ? (primary + lastOrder) * (2 * lastOrder + 1) + secondary + lastOrder
                                              : primary + lastOrder;
                        if ((primary != 0 && std::abs (primary) < firstOrder)
                            || (secondary != 0 && std::abs (secondary) < firstOrder)) continue;
                        float cx = -x + float (primary) * wavelength / m.geometry.z * m.geometry.x;
                        float cy = -y + float (primary) * wavelength / m.geometry.z * m.geometry.y;
                        if (crossed)
                        {
                            cx += float (secondary) * wavelength / m.secondaryGeometry.z * m.secondaryGeometry.x;
                            cy += float (secondary) * wavelength / m.secondaryGeometry.z * m.secondaryGeometry.y;
                        }
                        const float square = cx * cx + cy * cy;
                        // An evanescent order keeps its efficiency at z = 0, so the
                        // field stays continuous across the cut-off a texel may straddle.
                        const float z = square < 1.0f ? std::sqrt (1.0f - square) : 0.0f;
                        const float phase = 2.0f * pi * m.geometry.w * (incidentZ + z) / wavelength;
                        float efficiency = 0.0f;
                        if (crossed && thirdFamily && std::abs (primary) + kThirdFamilyConvolutionRadius <= reach
                            && std::abs (secondary) + kThirdFamilyConvolutionRadius <= reach)
                        {
                            for (int a = -reach; a <= reach; ++a)
                                amplitudes[std::size_t (a + reach)] = profileAmplitude (profile, a, m.microstructure.x, phase);
                            Complex total {};
                            for (int r = -kThirdFamilyConvolutionRadius; r <= kThirdFamilyConvolutionRadius; ++r)
                                total = total + amplitudes[std::size_t (primary - r + reach)]
                                    * amplitudes[std::size_t (secondary + r + reach)]
                                    * amplitudes[std::size_t (r + reach)];
                            efficiency = normSquared (total);
                        }
                        else
                            efficiency = latticeEfficiency (profile, primary, secondary, crossed, thirdFamily,
                                                            m.microstructure.x, phase);
                        const int tile = tileIndex (band, k, orders);
                        const std::size_t at = std::size_t ((tile / kTilesPerRow) * kGrid + j) * std::size_t (table->width)
                            + std::size_t ((tile % kTilesPerRow) * kGrid + i);
                        table->texels[at] = efficiency;
                        if (square < 1.0f && incidentZ > 1.0e-6f)
                        {
                            const float height = 2.0f * pi * m.roughness.x * (incidentZ + z) / wavelength;
                            sum += efficiency * std::exp (-height * height) * z / incidentZ;
                        }
                    }
                }
                const int tile = tileIndex (band, orders, orders);
                table->texels[std::size_t ((tile / kTilesPerRow) * kGrid + j) * std::size_t (table->width)
                    + std::size_t ((tile % kTilesPerRow) * kGrid + i)] = sum;
            }
    };
    std::vector<std::future<void>> work;
    for (int band = 0; band < 8; ++band) work.push_back (std::async (std::launch::async, bandWork, band));
    for (auto& w : work) w.get();
    return table;
}

// The parameter bytes a table reads. The incident lane is the authored source, which
// the table never reads.
inline std::string key (const diffractionmaterial::PhysicalDiffractionGpuParameters& m)
{
    std::string bytes (reinterpret_cast<const char*> (&m), sizeof (m));
    std::memset (&bytes[offsetof (diffractionmaterial::PhysicalDiffractionGpuParameters, incident)], 0,
                 sizeof (m.incident));
    return bytes;
}

inline bool sameGrating (const diffractionmaterial::PhysicalDiffractionGpuParameters& a,
                         const diffractionmaterial::PhysicalDiffractionGpuParameters& b)
{
    return key (a) == key (b);
}

// One table per grating, so an unchanged material reuses it from frame to frame.
inline std::shared_ptr<const Table> cached (const diffractionmaterial::PhysicalDiffractionGpuParameters& m)
{
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const Table>> tables;
    const auto bytes = key (m);
    std::lock_guard<std::mutex> lock (mutex);
    if (const auto found = tables.find (bytes); found != tables.end()) return found->second;
    if (tables.size() >= 16) tables.clear();
    return tables[bytes] = build (m);
}
} // namespace arbitgpu::realtimediffractionefficiency
