// Opt-in overdraw probe for the closed-glass camera walk (ARBIT_GL_GLASS_OVERDRAW_PROBE=1).
//
// The OpenGL backend wraps every object draw of the closed-glass bands in a
// GL_FRAGMENT_SHADER_INVOCATIONS_ARB query, a GL_SAMPLES_PASSED query and a
// GL_TIME_ELAPSED query. Each object is tagged by the camera walk its draws run.
// After the bands, full-screen passes over the frame's raw object-id image count, inside GL_SAMPLES_PASSED,
// the pixels whose final surface belongs to each drawn object, and the pixels any band
// draw left an id in. Only band-loop draws write object ids in that frame, so the
// invocations and the visible pixels count the same draws. The results are read one or
// more frames late, only once GL_QUERY_RESULT_AVAILABLE says so, so the probe never waits
// on the GPU. This header holds the GL-free parts: the switch, the walk-mode test, the
// log cadence and the log line, so a CPU test can pin the format.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace arbitgpu::glassoverdraw
{
inline constexpr const char* kEnvironment = "ARBIT_GL_GLASS_OVERDRAW_PROBE";
// One line per this many measured frames. The first measured frame always logs.
inline constexpr std::uint32_t kLogEveryMeasuredFrames = 30;

// Depth prepass draws of walked glass issued in this process (backend_opengl.cpp). It
// counts with the probe off too, so a test can tell whether the prepass ran.
inline std::atomic<std::uint64_t>& depthPrepassDraws() noexcept
{
    static std::atomic<std::uint64_t> draws {0};
    return draws;
}

inline bool enabledValue(const char* value) noexcept
{
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// Camera walk mode of one draw, from the uGlassTransport.x value it shades with: the
// per-object program's identifiers[1] when the draw writes the uniform, otherwise the last
// value written (the frame default from the root program, or 0). The shader runs the thin
// water walk for exactly 2, the closed-volume walk for any other value above 0.5, and
// ordinary lighting otherwise. This mirrors that test.
enum WalkMode : std::uint8_t { kOrdinary = 0, kVolume = 1, kWater = 2 };

inline std::uint8_t walkMode(float transportMode) noexcept
{
    if (!(transportMode > 0.5f)) return kOrdinary;
    return transportMode == 2.0f ? kWater : kVolume;
}

inline const char* walkTag(std::uint8_t mode) noexcept
{
    return mode == kWater ? "water" : mode == kVolume ? "volume" : "obj";
}

struct ObjectCount
{
    std::uint32_t object = 0;          // SceneObjectId value
    std::uint8_t mode = kOrdinary;     // WalkMode of its draws
    std::uint64_t invocations = 0;     // fragment shader invocations, all bands
    std::uint64_t samplesPassed = 0;   // samples that passed the depth test at draw time
    std::uint64_t visible = 0;         // pixels whose final object id is this object
    std::uint64_t nanoseconds = 0;     // GL_TIME_ELAPSED of its draws, all bands
};

struct FrameCounts
{
    std::uint32_t width = 0, height = 0, bands = 0;
    std::uint64_t covered = 0;         // pixels any band draw left an object id in
    std::vector<ObjectCount> objects;  // first-draw order
};

struct Totals
{
    std::uint64_t invocations = 0, visible = 0, nanoseconds = 0;
};

inline std::uint64_t totalInvocations(const FrameCounts& counts) noexcept
{
    std::uint64_t total = 0;
    for (const auto& object : counts.objects) total += object.invocations;
    return total;
}

// Sums the objects of one walk mode.
inline Totals modeTotals(const FrameCounts& counts, std::uint8_t mode) noexcept
{
    Totals totals;
    for (const auto& object : counts.objects)
        if (object.mode == mode)
        {
            totals.invocations += object.invocations;
            totals.visible += object.visible;
            totals.nanoseconds += object.nanoseconds;
        }
    return totals;
}

// Sums every object that runs a camera walk (water and volume).
inline Totals walkTotals(const FrameCounts& counts) noexcept
{
    const auto water = modeTotals(counts, kWater), volume = modeTotals(counts, kVolume);
    return {water.invocations + volume.invocations, water.visible + volume.visible,
            water.nanoseconds + volume.nanoseconds};
}

// Invocations per covered pixel, unclamped. Zero when nothing is covered.
inline double factor(std::uint64_t invocations, std::uint64_t covered) noexcept
{
    return covered == 0 ? 0.0 : static_cast<double>(invocations) / static_cast<double>(covered);
}

inline double milliseconds(std::uint64_t nanoseconds) noexcept
{
    return static_cast<double>(nanoseconds) * 1e-6;
}

// measuredIndex counts frames whose results were read, starting at 1.
inline bool shouldLog(std::uint64_t measuredIndex) noexcept
{
    return measuredIndex != 0 && (measuredIndex - 1) % kLogEveryMeasuredFrames == 0;
}

// An object keeps the highest walk mode any of its draws used.
inline ObjectCount& entryFor(FrameCounts& counts, std::uint32_t object, std::uint8_t mode)
{
    for (auto& entry : counts.objects)
        if (entry.object == object) { if (mode > entry.mode) entry.mode = mode; return entry; }
    counts.objects.push_back({object, mode, 0, 0, 0, 0});
    return counts.objects.back();
}

// Adds one draw's counts to its object's entry, keeping first-draw order.
inline void accumulate(FrameCounts& counts, std::uint32_t object, std::uint8_t mode,
                       std::uint64_t invocations, std::uint64_t samplesPassed, std::uint64_t nanoseconds)
{
    auto& entry = entryFor(counts, object, mode);
    entry.invocations += invocations;
    entry.samplesPassed += samplesPassed;
    entry.nanoseconds += nanoseconds;
}

// [gl] glass overdraw: invocations=<n> covered=<m> factor=<n/m> walk-invocations=<w> walk-visible=<v> walk-factor=<w/v> walk-ms=<t> water-invocations=<wi> water-visible=<wv> water-ms=<wt> volume-invocations=<vi> volume-visible=<vv> volume-ms=<vt> per-object=[<water|volume|obj><id>:<invocations>/<samplesPassed>/<visible>:ms=<x.xx>,...] frame=<w>x<h> bands=<b>
inline std::string formatLine(const FrameCounts& counts)
{
    const auto invocations = totalInvocations(counts);
    const auto walk = walkTotals(counts), water = modeTotals(counts, kWater), volume = modeTotals(counts, kVolume);
    char head[768];
    std::snprintf(head, sizeof head,
                  "[gl] glass overdraw: invocations=%llu covered=%llu factor=%.3f walk-invocations=%llu "
                  "walk-visible=%llu walk-factor=%.3f walk-ms=%.2f water-invocations=%llu water-visible=%llu "
                  "water-ms=%.2f volume-invocations=%llu volume-visible=%llu volume-ms=%.2f per-object=[",
                  static_cast<unsigned long long>(invocations),
                  static_cast<unsigned long long>(counts.covered),
                  factor(invocations, counts.covered),
                  static_cast<unsigned long long>(walk.invocations),
                  static_cast<unsigned long long>(walk.visible),
                  factor(walk.invocations, walk.visible),
                  milliseconds(walk.nanoseconds),
                  static_cast<unsigned long long>(water.invocations),
                  static_cast<unsigned long long>(water.visible),
                  milliseconds(water.nanoseconds),
                  static_cast<unsigned long long>(volume.invocations),
                  static_cast<unsigned long long>(volume.visible),
                  milliseconds(volume.nanoseconds));
    std::string line = head;
    for (std::size_t index = 0; index < counts.objects.size(); ++index)
    {
        const auto& object = counts.objects[index];
        char entry[160];
        std::snprintf(entry, sizeof entry, "%s%s%u:%llu/%llu/%llu:ms=%.2f", index == 0 ? "" : ",",
                      walkTag(object.mode), static_cast<unsigned>(object.object),
                      static_cast<unsigned long long>(object.invocations),
                      static_cast<unsigned long long>(object.samplesPassed),
                      static_cast<unsigned long long>(object.visible),
                      milliseconds(object.nanoseconds));
        line += entry;
    }
    char tail[64];
    std::snprintf(tail, sizeof tail, "] frame=%ux%u bands=%u\n", counts.width, counts.height, counts.bands);
    line += tail;
    return line;
}
} // namespace arbitgpu::glassoverdraw
