#pragma once

#include "fixture_rough_lane_slices.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

// Test runs share the GPU with the desktop. With ARBIT_TEST_GPU_YIELD=1, which
// tools/run-capped.sh sets, every fenced glass submission gives the GPU back for as
// long as it held it, and one past the 33 ms limit says so. The product never sets it.
namespace arbitgpu::testpacing
{
inline bool parseEnabled(const char* value) noexcept {return value && !std::strcmp(value,"1");}
inline bool enabled() noexcept
{
    static const bool on=parseEnabled(std::getenv("ARBIT_TEST_GPU_YIELD"));
    return on;
}
inline constexpr std::chrono::nanoseconds kLongestYield = std::chrono::seconds(1);
inline std::chrono::nanoseconds yieldFor(std::chrono::nanoseconds held) noexcept
{
    return std::clamp(held,std::chrono::nanoseconds(0),kLongestYield);
}
// Splits one raster band's hold, so its fixed cost can be named: encode is the CPU's draw
// calls, gpu the two timestamps around them, and the rest of held is waiting in the queue.
inline void logBand(std::uint32_t columns,std::uint32_t rows,std::uint64_t pixels,double encodeMilliseconds,double gpuMilliseconds,std::chrono::nanoseconds held)
{
    std::fprintf(stderr,"[gl] test pacing: band columns=%u rows=%u pixels=%llu encode %.1f ms gpu %.1f ms held %.1f ms\n",columns,rows,
        (unsigned long long)pixels,encodeMilliseconds,gpuMilliseconds,double(held.count())*1e-6);
}
inline void afterSubmission(std::chrono::nanoseconds held)
{
    if(!enabled())return;
    if(held>roughglasstiles::kMaximumDispatchTime)
        std::fprintf(stderr,"[gl] test pacing: submission held the GPU %.1f ms\n",double(held.count())*1e-6);
    std::this_thread::sleep_for(yieldFor(held));
}
}
