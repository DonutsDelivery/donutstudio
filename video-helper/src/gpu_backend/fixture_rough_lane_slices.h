#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

// Slices glass work into short submissions so a walk never holds the GPU for long.
// A selected rough tile goes by lane, a closed glass frame by rows. A lane is one rough sample of one channel, 48 per pixel. A pixel
// keeps its accumulators between dispatches and runs its lanes in order, so the
// slice sizes change when work reaches the GPU and not one bit of the result.
namespace arbitgpu::roughglasstiles
{
inline constexpr std::uint32_t kRoughPixelLanes = 48;
// The longest dispatch a slice aims for without yielding.
inline constexpr std::chrono::nanoseconds kMaximumDispatchTime = std::chrono::milliseconds(33);

// How much one submission may carry, from the last one's time: glass pixel lanes for
// a tile, frame pixels for a raster band. It aims at half the limit because one pixel can cost far more than the last
// one did. The smallest slice is one pixel's one lane, since a lane holds its path
// stack in shader locals.
class SliceBudget final
{
public:
    explicit SliceBudget(std::uint64_t first=1,std::uint64_t most=std::uint64_t(1)<<20) noexcept:units_(first),most_(most) {}
    std::uint64_t units() const noexcept { return units_; }
    void observe(std::uint64_t units,std::chrono::nanoseconds elapsed) noexcept
    {
        if(!units)return;
        const auto nanoseconds=std::max<std::int64_t>(1,elapsed.count());
        const auto fit=double(kMaximumDispatchTime.count())*.5*double(units)/double(nanoseconds);
        const auto most=std::min<double>(double(units_)*2,double(most_));
        units_=std::uint64_t(std::max(1.0,std::min(fit,most)));
    }
private:
    std::uint64_t units_,most_;
};

struct LaneSlice final
{
    std::uint32_t firstPixel=0,pixelCount=0,laneBegin=0,laneEnd=0;
    std::uint64_t units=0;
};

class LaneSlicer final
{
public:
    explicit LaneSlicer(std::vector<std::uint8_t> glass):glass_(std::move(glass))
    {
        for(const auto flag:glass_)count_+=flag?1u:0u;
    }
    // A tile with no glass still writes its pixels' records, once.
    bool next(const SliceBudget& budget,LaneSlice& slice)
    {
        const auto pixels=std::uint32_t(glass_.size());
        if(lane_>=kRoughPixelLanes)return false;
        if(!count_) {
            slice={0,pixels,0,kRoughPixelLanes,0};
            lane_=kRoughPixelLanes;
            return true;
        }
        const auto capacity=std::max<std::uint64_t>(1,budget.units());
        if(pixel_==0 && capacity>=count_) {
            const auto lanes=std::uint32_t(std::min<std::uint64_t>(capacity/count_,kRoughPixelLanes-lane_));
            slice={0,pixels,lane_,lane_+lanes,count_*lanes};
            lane_+=lanes;
            return true;
        }
        auto end=pixel_;
        std::uint64_t starts=0;
        while(end<pixels) {
            if(glass_[end] && starts==capacity)break;
            starts+=glass_[end]?1u:0u;
            ++end;
        }
        slice={pixel_,end-pixel_,lane_,lane_+1,starts};
        pixel_=end;
        if(pixel_==pixels) {pixel_=0;++lane_;}
        return true;
    }
private:
    std::vector<std::uint8_t> glass_;
    std::uint64_t count_=0;
    std::uint32_t pixel_=0,lane_=0;
};

// A raster band's first pixels, before any frame has been timed: 16.5 ms at the 32 us a
// glass pixel a 1080p frame measured on 2 Nov.
inline constexpr std::uint64_t kFirstGlassBandPixels = 512;
// A band this short says nothing about glass: 108 empty rows measured 0 ms, then the
// doubled band that met the sphere held 789 ms. So short bands don't grow the next one
// until a frame's worth of pixels has shown the glass is gone or cheap.
inline constexpr std::chrono::nanoseconds kUnmeasuredBandTime = std::chrono::milliseconds(1);
inline constexpr std::uint64_t kUnmeasuredPixelsBeforeGrowth = std::uint64_t(1)<<24;
// Time of a frame's hardest band against its mean, for a flushed frame that only has the
// mean: one row near the sphere's edge measured 92 ms against a mean of 20.
inline constexpr double kFlushedBandPeak = 5.0;
class GlassBandBudget final
{
public:
    GlassBandBudget(std::uint64_t first,std::uint64_t most) noexcept:slices_(first,most) {}
    std::uint64_t units() const noexcept { return slices_.units(); }
    void observe(std::uint64_t units,std::chrono::nanoseconds elapsed) noexcept
    {
        if(!units)return;
        if(elapsed>=kUnmeasuredBandTime)unmeasured_=0;
        else if((unmeasured_+=units)<kUnmeasuredPixelsBeforeGrowth)return;
        slices_.observe(units,elapsed);
    }
private:
    SliceBudget slices_;
    std::uint64_t unmeasured_=0;
};
// One scissor rectangle. Whole rows while the budget covers a row, and part of one row
// when one row is more than a submission can hold: a 1920 pixel row measured 92 ms.
// Rectangles stay disjoint and in order, so their sizes change when each pixel reaches
// the GPU and not what it draws.
struct GlassBand final
{
    std::uint32_t x=0,y=0,columns=0,rows=0;
    std::uint64_t pixels() const noexcept {return std::uint64_t(columns)*rows;}
};
inline GlassBand nextBand(std::uint64_t pixels,std::uint32_t width,std::uint32_t height,std::uint32_t x,std::uint32_t y) noexcept
{
    const auto units=std::max<std::uint64_t>(1,pixels);
    if(x==0 && units>=width)
        return {0,y,width,std::uint32_t(std::min<std::uint64_t>(units/std::max<std::uint32_t>(1,width),height-y))};
    return {x,y,std::uint32_t(std::min<std::uint64_t>(units,width-x)),1};
}
inline GlassBand nextGlassBand(const GlassBandBudget& budget,std::uint32_t width,std::uint32_t height,std::uint32_t x,std::uint32_t y) noexcept
{
    return nextBand(budget.units(),width,height,x,y);
}
}
