// A selected rough tile's dispatch slices can't change a bit of its result.
//
// This runs the production slicer over a float32 stand-in for the walk: order-
// sensitive additions per lane, a per-channel step at each channel's first lane and
// its closure at the last. Every slicing must match the whole-pixel result bit for
// bit. The GLSL gates are checked as text. A GPU run of the same comparison is owed
// while no GPU run is allowed. These need no GPU.

#include "../src/gpu_backend/fixture_gpu_test_pacing.h"
#include "../src/gpu_backend/fixture_rough_lane_slices.h"
#include "../src/gpu_backend/fixture_smooth_glass.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <initializer_list>
#include <tuple>
#include <vector>

namespace
{
namespace rt = arbitgpu::roughglasstiles;
namespace sg = arbitgpu::smoothglass;
int failures = 0;
void check(bool condition, const std::string& label)
{
    std::printf("%s: %s\n", condition ? "ok" : "FAILED", label.c_str());
    if (!condition) ++failures;
}
struct Pixel { std::array<float, 3> radiance {}, terminal {}; std::uint32_t next = 0; bool ordered = true; };
void runLanes(Pixel& p, std::uint32_t pixel, std::uint32_t begin, std::uint32_t end)
{
    if (p.next != begin) p.ordered = false;
    p.next = end;
    for (auto lane = begin; lane < end; ++lane)
    {
        const auto channel = lane / 16, sample = lane % 16;
        if (sample == 0) p.radiance[channel] += std::sin(float(pixel + channel) * .37f) * 1e-3f;
        const auto seed = float(pixel * 48 + lane + 1);
        p.radiance[channel] = p.radiance[channel] * .999f + std::sqrt(seed) * 1e-5f / seed;
        p.terminal[channel] += 1.f / 16.f * (1.f - 1e-7f * seed);
        if (sample == 15) p.terminal[channel] = std::min(p.terminal[channel], 1.f);
    }
}
std::vector<std::uint8_t> flags(std::uint32_t pixels, std::uint32_t stride)
{
    std::vector<std::uint8_t> glass(pixels);
    for (std::uint32_t i = 0; i < pixels; ++i) glass[i] = stride && i % stride != 0;
    return glass;
}
// Runs a tile, feeding the budget from a stand-in clock.
template<class Clock> std::vector<Pixel> sliced(const std::vector<std::uint8_t>& glass, Clock clock,
    std::uint64_t& dispatches, std::uint64_t* largest = nullptr)
{
    std::vector<Pixel> pixels(glass.size());
    rt::SliceBudget budget;
    rt::LaneSlicer slicer(glass);
    rt::LaneSlice slice;
    dispatches = 0;
    while (slicer.next(budget, slice))
    {
        ++dispatches;
        for (std::uint32_t i = slice.firstPixel; i < slice.firstPixel + slice.pixelCount; ++i)
            if (glass[i]) runLanes(pixels[i], i, slice.laneBegin, slice.laneEnd);
        if (largest) *largest = std::max(*largest, slice.units);
        budget.observe(slice.units, clock(slice, dispatches));
    }
    return pixels;
}
bool same(const std::vector<Pixel>& a, const std::vector<Pixel>& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(a[i].radiance.data(), b[i].radiance.data(), 12) || std::memcmp(a[i].terminal.data(), b[i].terminal.data(), 12)
            || !a[i].ordered || a[i].next != b[i].next) return false;
    return true;
}
}

int main()
{
    using namespace std::chrono;
    const auto glass = flags(136, 5);
    std::vector<Pixel> whole(glass.size());
    for (std::uint32_t i = 0; i < glass.size(); ++i)
        if (glass[i]) runLanes(whole[i], i, 0, rt::kRoughPixelLanes);

    std::uint64_t dispatches = 0;
    const auto per = [](nanoseconds unit) { return [unit](const rt::LaneSlice& s, std::uint64_t) { return unit * std::int64_t(s.units); }; };
    check(same(sliced(glass, per(milliseconds(1)), dispatches), whole), "1 ms a unit matches the whole pixel bit for bit");
    check(same(sliced(glass, per(microseconds(1)), dispatches), whole), "1 us a unit matches, with whole-tile dispatches");
    const auto fast = dispatches;
    check(same(sliced(glass, per(milliseconds(200)), dispatches), whole), "200 ms a unit matches, at one pixel's one lane");
    check(dispatches > fast && dispatches >= 108 * 48, "a slow walk takes one dispatch per glass pixel lane");
    check(same(sliced(glass, [](const rt::LaneSlice&, std::uint64_t n) { return milliseconds(n % 7 ? 1 : 400); }, dispatches), whole),
          "spiky timing matches");

    // The tile the 640x360 frame averaged: 6.8 ms a glass pixel, 48 lanes. Derived, not timed.
    const auto measured = nanoseconds(std::int64_t(6.82e6 / 48));
    std::uint64_t largest = 0;
    const auto full = flags(136, 0x7fffffff);
    sliced(full, per(measured), dispatches, &largest);
    check(std::int64_t(largest) * measured.count() <= rt::kMaximumDispatchTime.count(), "at that average no dispatch passes 33 ms");
    std::printf("average-rate tile: %llu dispatches, largest %llu pixel lanes, %.1f ms derived\n",
        (unsigned long long) dispatches, (unsigned long long) largest, double(largest) * double(measured.count()) * 1e-6);

    const std::vector<std::uint8_t> none(136, 0);
    rt::SliceBudget budget;
    rt::LaneSlicer empty(none);
    rt::LaneSlice slice;
    check(empty.next(budget, slice) && slice.pixelCount == 136 && slice.laneEnd == 48 && !slice.units, "a tile with no glass writes all its records once");
    check(!empty.next(budget, slice), "and takes one dispatch");

    rt::SliceBudget grows;
    for (int i = 0; i < 4; ++i) grows.observe(grows.units(), microseconds(1));
    check(grows.units() == 16, "the budget at most doubles a dispatch");
    grows.observe(grows.units(), seconds(10));
    check(grows.units() == 1, "and drops at once, to no less than one unit");

    const auto text = sg::transportShader(false, true, true);
    for (const auto* gate : {"int laneBegin,int laneEnd,vec3 radianceSoFar", "laneBegin<=channel*GLASS_ROUGH_SAMPLES",
             "max(laneBegin-channel*GLASS_ROUGH_SAMPLES,0)", "if(laneEnd<(channel+1)*GLASS_ROUGH_SAMPLES) {status=GLASS_CAMERA_V2_RUNNING;return radiance;}",
             "inout int queryCount", "vec3 radiance=radianceSoFar;",
             "status=laneEnd>=3*GLASS_ROUGH_SAMPLES?GLASS_CAMERA_V2_COMPLETE:GLASS_CAMERA_V2_RUNNING;"})
        check(text.find(gate) != std::string::npos, std::string("the GL walk has ") + gate);
    check(sg::transportShader(true, true, true).find("laneBegin") == std::string::npos, "Metal's walk stays unsliced");
    // Raster bands: every pixel once, in order, under any timing, so a frame's pixels
    // shade what they shade unbanded.
    const auto bands = [](std::uint32_t width, std::uint32_t height, nanoseconds perPixel, std::uint64_t* over = nullptr) {
        rt::GlassBandBudget budget { rt::kFirstGlassBandPixels, sg::kClosedGlassBandPixels };
        std::vector<std::uint8_t> drawn(std::size_t(width) * height);
        std::uint32_t count = 0, x = 0, y = 0;
        std::size_t order = 0;
        bool once = true;
        while (y < height)
        {
            const auto band = rt::nextGlassBand(budget, width, height, x, y);
            for (auto row = band.y; row < band.y + band.rows; ++row)
                for (auto column = band.x; column < band.x + band.columns; ++column)
                {
                    const auto at = std::size_t(row) * width + column;
                    once = once && at == order && !drawn[at];
                    drawn[at] = 1; ++order;
                }
            const auto elapsed = perPixel * std::int64_t(band.pixels());
            if (over && elapsed > rt::kMaximumDispatchTime) ++*over;
            budget.observe(band.pixels(), elapsed);
            x += band.columns;
            if (x >= width) { x = 0; y += band.rows; }
            ++count;
        }
        return std::tuple(once && order == drawn.size(), count);
    };
    for (const auto rate : std::initializer_list<nanoseconds>{nanoseconds(1), nanoseconds(540), microseconds(32), milliseconds(3)})
        check(std::get<0>(bands(1920, 1080, rate)) && std::get<0>(bands(3840, 2160, rate)) && std::get<0>(bands(1u << 20, 3, rate)),
              "every pixel draws once in order at " + std::to_string(rate.count()) + " ns a pixel");
    std::uint64_t over = 0;
    bands(1920, 1080, microseconds(32), &over);
    check(over == 0, "at the measured 32 us a pixel no band passes 33 ms, the first included");
    over = 0;
    bands(3840, 2160, nanoseconds(540), &over);
    check(over == 0, "nor at 0.54 us a pixel");
    {
        // The 2 Nov frame: 108 empty rows, then a sphere at 32 us a pixel.
        rt::GlassBandBudget budget { rt::kFirstGlassBandPixels, sg::kClosedGlassBandPixels };
        std::uint32_t x = 0, y = 0;
        nanoseconds longest{0};
        while (y < 400)
        {
            const auto band = rt::nextGlassBand(budget, 1920, 400, x, y);
            const auto elapsed = band.y < 108 ? nanoseconds(100) : microseconds(32) * std::int64_t(band.pixels());
            longest = std::max(longest, elapsed);
            budget.observe(band.pixels(), elapsed);
            x += band.columns;
            if (x >= 1920) { x = 0; y += band.rows; }
        }
        check(longest <= rt::kMaximumDispatchTime, "empty rows don't grow the band that meets the glass");
    }
    rt::GlassBandBudget capped { rt::kFirstGlassBandPixels, sg::kClosedGlassBandPixels };
    for (int i = 0; i < 40; ++i) capped.observe(capped.units(), milliseconds(2));
    check(rt::nextGlassBand(capped, 3840, 2160, 0, 0).rows == sg::closedGlassBandRows(3840), "a band never passes the old band size");
    check(rt::nextGlassBand(capped, 3840, 10, 0, 0).rows == 10, "a band stops at the frame's last row");
    capped.observe(capped.units(), seconds(100));
    {
        const auto band = rt::nextGlassBand(capped, 3840, 2160, 0, 0);
        check(band.rows == 1 && band.columns == 1, "a very slow band drops to one pixel");
        check(rt::nextGlassBand(capped, 3840, 2160, 3839, 5).x == 3839, "and a partial row goes on from its column");
    }
    {
        rt::GlassBandBudget cheap { rt::kFirstGlassBandPixels, sg::kClosedGlassBandPixels };
        for (std::uint64_t seen = 0; seen < rt::kUnmeasuredPixelsBeforeGrowth; seen += cheap.units())
            cheap.observe(cheap.units(), nanoseconds(1));
        check(cheap.units() == rt::kFirstGlassBandPixels * 2, "a frame's worth of unmeasured pixels lets a band grow");
    }

    namespace tp = arbitgpu::testpacing;
    check(tp::parseEnabled("1") && !tp::parseEnabled("0") && !tp::parseEnabled("") && !tp::parseEnabled(nullptr), "pacing takes only an exact 1");
    check(tp::yieldFor(milliseconds(5)) == milliseconds(5), "a test submission yields as long as it held the GPU");
    check(tp::yieldFor(seconds(30)) == tp::kLongestYield, "for at most a second");
    check(tp::yieldFor(nanoseconds(-3)) == nanoseconds(0), "and never a negative time");
    {
        // The yield changes when work reaches the GPU, not the slices' partition.
        rt::SliceBudget budget;
        budget.observe(1, milliseconds(1));
        check(budget.units() == 2, "the budget reads the GPU's time, with no yield in it");
    }

    check(sg::kRoughInterfaceSamples * 3 == rt::kRoughPixelLanes, "a pixel has 48 lanes");
    return failures == 0 ? 0 : 1;
}
