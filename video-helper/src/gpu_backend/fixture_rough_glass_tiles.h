#pragma once

#include "fixture_smooth_glass.h"
#include "frame_memory_budget.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace arbitgpu::roughglasstiles
{
inline constexpr std::uint64_t kMaximumTilePixels = smoothglass::kMaximumPixelRayQueries
    / smoothglass::kMaximumRoughFrameRayQueries;
inline constexpr std::uint64_t kMaximumFrameRayQueries = renderpassoutput::kMaximumPixels
    * smoothglass::kMaximumRoughFrameRayQueries;
// Three GPU RGBA32F receipts and one live owned readback of those receipts,
// raw IDs, Normal and worst-case RGBA16F Color. No full-frame CPU pixel assembly.
inline constexpr std::uint64_t kTileReadbackBytesPerPixel = 48+4+4+8+8;
inline constexpr std::uint64_t kMaximumMultipleScatterTilePixels = smoothglass::kMaximumCameraMicroWork
    / smoothglass::kMaximumCameraPixelMicroWork;
static_assert(kMaximumTilePixels==9708 && kMaximumMultipleScatterTilePixels==136);

struct FrameKey final
{
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> program;
    // Native callers create one const snapshot after resolving overrides and
    // retain it for every tile. Snapshot identity covers all runtime values and
    // subordinate Frame leases, including time, transforms and animated inputs.
    std::shared_ptr<const NativeFixtureSceneRuntimeInputs> runtime;
    std::shared_ptr<const void> owner;
    HarmonicMIDI::grid::SceneCameraRecord camera;
    std::uint64_t sceneRevision = 0, rendererGeneration = 0;
    std::uint32_t width = 0, height = 0;
};
template<class T> inline bool sameOwner(const std::shared_ptr<const T>& a,
                                       const std::shared_ptr<const T>& b) noexcept
{
    return a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);
}
inline std::array<float,13> cameraValues(const HarmonicMIDI::grid::SceneCameraRecord& camera) noexcept
{
    const auto& t=camera.transform;
    return {t.translation.x,t.translation.y,t.translation.z,t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w,
        t.scale.x,t.scale.y,t.scale.z,camera.verticalFovRadians,camera.nearPlane,camera.farPlane};
}
inline bool sameKey(const FrameKey& a,const FrameKey& b) noexcept
{
    const auto av=cameraValues(a.camera),bv=cameraValues(b.camera);
    return sameOwner(a.scene,b.scene) && sameOwner(a.program,b.program)
        && sameOwner(a.runtime,b.runtime) && sameOwner(a.owner,b.owner)
        && a.camera.id==b.camera.id && std::memcmp(av.data(),bv.data(),av.size()*sizeof(float))==0
        && a.sceneRevision==b.sceneRevision && a.rendererGeneration==b.rendererGeneration
        && a.width==b.width && a.height==b.height;
}
struct Tile final
{
    // Full-frame coordinates, with top row first. Viewport/projection and camera
    // remain those of FrameKey; only scratch allocation/readback use tile extent.
    std::uint32_t index = 0, x = 0, y = 0, width = 0, height = 0;
    std::uint64_t pixels() const noexcept { return std::uint64_t(width)*height; }
    std::uint64_t receiptBytes() const noexcept { return pixels()*smoothglass::kRoughReceiptBytesPerPixel; }
    std::uint64_t reservedQueries() const noexcept { return pixels()*smoothglass::kMaximumRoughFrameRayQueries; }
};
inline bool sameTile(const Tile& a,const Tile& b) noexcept
{
    return a.index==b.index && a.x==b.x && a.y==b.y && a.width==b.width && a.height==b.height;
}
class Schedule final
{
public:
    static bool admit(const smoothglass::Prepared& prepared,const FrameKey& key,bool instanced,
                      std::shared_ptr<const Schedule>& output,std::string& error)
    {
        if (!prepared.enabled || !(prepared.roughness>0) || !key.scene || !key.program || !key.runtime || !key.owner
            || !sameOwner(prepared.sourceScene,key.scene) || !sameOwner(prepared.sourceMaterial,key.program)
            || key.sceneRevision!=key.program->sceneRevision || key.rendererGeneration==0
            || !validFixtureRuntimeInputs(*key.runtime)
            || !validFixtureCameraOverride(std::optional<HarmonicMIDI::grid::SceneCameraRecord>{key.camera}))
        { error="Rough glass tiles require exact immutable prepared, runtime, camera and native owner inputs"; return false; }
        const auto pixels=std::uint64_t(key.width)*key.height;
        if (!key.width || !key.height || key.width>renderpassoutput::kMaximumWidth
            || key.height>renderpassoutput::kMaximumHeight || pixels>renderpassoutput::kMaximumPixels
            || pixels>kMaximumFrameRayQueries/smoothglass::kMaximumRoughFrameRayQueries)
        { error="Rough glass tiled frame extent or total ray-query budget is outside its bound"; return false; }
        const auto tilePixelLimit=smoothglass::multipleScatterCamera(prepared)?kMaximumMultipleScatterTilePixels:kMaximumTilePixels;
        std::uint32_t tileWidth=0,tileHeight=0;
        std::uint64_t jobs=std::numeric_limits<std::uint64_t>::max(),area=0;
        // Minimize rectangular jobs, then prefer larger area and wider tiles.
        // This choice is independent of visible glass, GPU timing and completion.
        for (std::uint32_t w=1;w<=std::min<std::uint64_t>(key.width,tilePixelLimit);++w) {
            const auto h=static_cast<std::uint32_t>(std::min<std::uint64_t>(key.height,tilePixelLimit/w));
            const auto count=((std::uint64_t(key.width)+w-1)/w)*((std::uint64_t(key.height)+h-1)/h);
            const auto size=std::uint64_t(w)*h;
            if (count<jobs || (count==jobs && (size>area || (size==area && w>tileWidth))))
            { jobs=count;area=size;tileWidth=w;tileHeight=h; }
        }
        std::uint64_t scratch=0;
        if (!smoothglass::admitFrame(prepared,tileWidth,tileHeight,instanced,error,&scratch)) return false;
        auto candidate=std::shared_ptr<Schedule>(new Schedule);
        candidate->key_=key; candidate->prepared_=prepared;
        candidate->maximumTilePixels_=area;
        candidate->tiles_.reserve(static_cast<std::size_t>(jobs));
        for (std::uint32_t y=0;y<key.height;y+=tileHeight)
            for (std::uint32_t x=0;x<key.width;x+=tileWidth)
                candidate->tiles_.push_back({static_cast<std::uint32_t>(candidate->tiles_.size()),x,y,
                    std::min(tileWidth,key.width-x),std::min(tileHeight,key.height-y)});
        output=std::move(candidate);
        return true;
    }
    const FrameKey& key() const noexcept { return key_; }
    const smoothglass::Prepared& prepared() const noexcept { return prepared_; }
    const std::vector<Tile>& tiles() const noexcept { return tiles_; }
    std::uint64_t pixels() const noexcept { return std::uint64_t(key_.width)*key_.height; }
    std::uint64_t reservedQueries() const noexcept { return pixels()*smoothglass::kMaximumRoughFrameRayQueries; }
    std::uint64_t maximumTilePixels() const noexcept { return maximumTilePixels_; }
    std::uint64_t scratchBytes() const noexcept { return maximumTilePixels_*smoothglass::kRoughReceiptBytesPerPixel; }
    std::uint64_t retainedBytes() const noexcept {
        return sizeof(Schedule)+tiles_.capacity()*sizeof(Tile)
            +prepared_.query.texels.capacity()*sizeof(opaquevisibility::Texel)
            +prepared_.query.meshes.capacity()*sizeof(opaquevisibility::Mesh);
    }
private:
    Schedule()=default;
    FrameKey key_;
    smoothglass::Prepared prepared_;
    std::vector<Tile> tiles_;
    std::uint64_t maximumTilePixels_=0;
};
// Raster and the float32 camera ray can disagree on the one pixel where a silhouette
// edge crosses a pixel center. Where the walk ran, its result owns the pixel, because
// the resolve already wrote its Color there. Elsewhere raster keeps the pixel, which
// has no walk. A disagreement away from any neighbor that explains it is a real
// mismatch, such as a stale or shifted selection.
//
// The frame keeps both owners of every pixel it has reconciled. A neighbor in a tile
// that hasn't run yet can't explain anything, so that pixel is checked again once all
// tiles have run, before the frame publishes. It keeps its promotion until then, since
// its tile's receipt needs it, and a failed check loses the whole frame.
//
// Each frame's receipt reports its promoted pixels and its silhouette: selected glass
// touching anything that either owner doesn't call glass, a raster crack's edge included.
// A promoted pixel is always on it. The counts measure a frame and never refuse one.
class SelectedOwnerFrame final
{
public:
    bool prepare(std::uint32_t width,std::uint32_t height,std::string& error)
    {
        width_=width;height_=height;promoted_=0;deferred_.clear();deferredCount_=0;silhouette_=0;
        try {
            const auto pixels=std::size_t(width)*height;
            owners_.assign(pixels,{});
            filled_.assign(pixels,0);
        } catch(const std::bad_alloc&) {error="Rough glass selected owner frame allocation failed";return false;}
        return true;
    }
    // x and y place the tile in the frame, counting rows from the top like its images.
    bool reconcile(std::uint32_t x,std::uint32_t y,std::uint32_t width,std::uint32_t height,std::uint32_t volume,
        const std::vector<std::array<std::uint32_t,4>>& selected,NativeRawPassPixels& objects,NativeRawPassPixels& materials,
        std::vector<std::size_t>& promoted,std::string& error)
    {
        promoted.clear();
        const auto pixels=std::size_t(width)*height;
        if(selected.size()!=pixels || objects.bytes.size()!=pixels*4 || materials.bytes.size()!=pixels*4
            || std::uint64_t(x)+width>width_ || std::uint64_t(y)+height>height_)
        {error="Rough glass selected owners lack their completed tile attachments";return false;}
        const auto word=[](const NativeRawPassPixels& image,std::size_t pixel) {
            std::uint32_t value=0;std::memcpy(&value,image.bytes.data()+pixel*4,4);return value;
        };
        for(std::size_t pixel=0;pixel<pixels;++pixel) {
            const auto at=framePixel(x+pixel%width,y+pixel/width);
            owners_[at]={selected[pixel][0],selected[pixel][1],word(objects,pixel),word(materials,pixel)};
            filled_[at]=1;
        }
        for(std::size_t pixel=0;pixel<pixels;++pixel) {
            const auto at=framePixel(x+pixel%width,y+pixel/width);
            const auto& owner=owners_[at];
            if(owner[2]==owner[0] && owner[3]==owner[1])continue;
            // The raster fragment discards itself where selection differs, so raster glass here has no walk behind it.
            if(owner[2]==volume && owner[0]!=volume)
            {error=kMismatch;return false;}
            const auto found=explained(at);
            if(found==Explained::No)
            {error=kMismatch;return false;}
            if(found==Explained::Later) {deferred_.push_back(at);++deferredCount_;}
            if(owner[0]==volume)promoted.push_back(pixel);
        }
        for(const auto pixel:promoted) {
            std::memcpy(objects.bytes.data()+pixel*4,&selected[pixel][0],4);
            std::memcpy(materials.bytes.data()+pixel*4,&selected[pixel][1],4);
        }
        promoted_+=promoted.size();
        return true;
    }
    bool finish(std::uint32_t volume,std::string& error)
    {
        for(const auto at:deferred_)
            if(explained(at)!=Explained::Yes) {error=kMismatch;return false;}
        deferred_.clear();
        for(std::size_t y=0;y<height_;++y)for(std::size_t x=0;x<width_;++x) {
            if(owners_[y*width_+x][0]!=volume)continue;
            const auto outside=[&](std::size_t other) {return owners_[other][0]!=volume || owners_[other][2]!=volume;};
            if(outside(y*width_+x) || (x>0 && outside(y*width_+x-1)) || (x+1<width_ && outside(y*width_+x+1))
                || (y>0 && outside((y-1)*width_+x)) || (y+1<height_ && outside((y+1)*width_+x)))++silhouette_;
        }
        return true;
    }
    std::uint64_t promotedPixels() const noexcept {return promoted_;}
    std::uint64_t silhouettePixels() const noexcept {return silhouette_;}
    std::uint64_t deferredPixels() const noexcept {return deferredCount_;}
private:
    enum class Explained {Yes,No,Later};
    static constexpr const char* kMismatch="OpenGL selected camera result differs from completed primary raster ownership";
    std::size_t framePixel(std::size_t x,std::size_t y) const noexcept {return y*width_+x;}
    Explained explained(std::size_t at) const
    {
        const auto column=std::int64_t(at%width_),row=std::int64_t(at/width_);
        const auto& owner=owners_[at];
        bool waiting=false;
        for(auto y=row-1;y<=row+1;++y)for(auto x=column-1;x<=column+1;++x) {
            if((x==column && y==row) || x<0 || y<0 || x>=std::int64_t(width_) || y>=std::int64_t(height_))continue;
            const auto other=framePixel(std::size_t(x),std::size_t(y));
            if(!filled_[other]) {waiting=true;continue;}
            const auto& neighbor=owners_[other];
            if((neighbor[0]==owner[2] && neighbor[1]==owner[3]) || (neighbor[2]==owner[0] && neighbor[3]==owner[1]))return Explained::Yes;
        }
        return waiting?Explained::Later:Explained::No;
    }
    std::uint32_t width_=0,height_=0;
    std::vector<std::array<std::uint32_t,4>> owners_; // Selected object and material, then raster's.
    std::vector<std::uint8_t> filled_;
    std::vector<std::size_t> deferred_;
    std::uint64_t promoted_=0,silhouette_=0,deferredCount_=0;
};
// A promoted pixel's linear depth, as the raster vertex writes it, from where the
// walk's first hit sits. rotation is the camera's quaternion, xyzw.
inline float promotedLinearDepth(const std::array<float,3>& position,const std::array<float,3>& translation,
    const std::array<float,4>& rotation,float nearPlane,float farPlane) noexcept
{
    const auto [x,y,z,w]=rotation;
    const float forward[3]{-2.0f*(x*z+w*y),-2.0f*(y*z-w*x),-(1.0f-2.0f*(x*x+y*y))};
    float distance=0;
    for(int axis=0;axis<3;++axis)distance+=(position[axis]-translation[axis])*forward[axis];
    return std::clamp((distance-nearPlane)/std::max(farPlane-nearPlane,0.000001f),0.0f,1.0f);
}
struct NativeTileReceipt final
{
    // The backend proves GPU completion and that every attachment belongs to
    // this exact job/key before invoking accept. Values alone prove no owner.
    bool gpuCompleted = false;
    struct Work final { std::uint32_t selections=0,starts=0,queries=0,events=0; };
    static_assert(sizeof(Work)==16);
    const NativeRawPassPixels *terminal=nullptr,*masked=nullptr,*tail=nullptr;
    const NativeRawPassPixels *objectIds=nullptr,*materialIds=nullptr,*normal=nullptr,*color=nullptr;
    const std::vector<Work>* nativeWork=nullptr;
};
struct Aggregate final
{
    std::uint32_t completedTiles = 0;
    std::uint64_t completedPixels = 0,coveredPixels = 0;
    std::uint64_t reservedQueries = 0,receiptClosestHitQueries = 0;
    std::uint64_t microEvents = 0,microWork = 0,selectionQueries = 0,transportStarts = 0;
    std::uint64_t peakReceiptBytes = 0,processedReceiptBytes = 0;
    // Silhouette pixels the walk took from raster, the silhouette itself, and pixels checked at the frame's end.
    std::uint64_t promotedPixels = 0,silhouettePixels = 0,deferredOwnerPixels = 0;
    // Sums of normalized source power over visible glass pixels, independently
    // for each wavelength. These are not radiance or convergence estimates.
    std::array<double,3> source {},terminal {},masked {},tail {};
    double maximumEnergyResidual = 0,maximumUnresolvedPower = 0;
    double absorbedTotalPower = 0,maximumAbsorbedPower = 0;
    std::uint64_t absorbedPixelsOverLimit = 0;
};
class Completion final
{
public:
    explicit Completion(std::shared_ptr<const Schedule> schedule) : schedule_(std::move(schedule))
    { if (schedule_) receipts_=std::make_unique<TilePower[]>(schedule_->tiles().size()); }
    const Aggregate& aggregate() const noexcept { return aggregate_; }
    bool invalidated() const noexcept { return invalidated_; }
    static constexpr std::size_t metadataBytesPerTile() noexcept { return sizeof(TilePower); }
    bool accept(const FrameKey& key,const Tile& tile,const NativeTileReceipt& receipt,std::string& error)
    {
        const auto fail=[&](const char* message) { invalidated_=true;error=message;return false; };
        if (invalidated_ || !schedule_ || !sameKey(key,schedule_->key()))
            return fail("Rough glass tile belongs to a stale or invalidated frame owner");
        if (tile.index>=schedule_->tiles().size() || !sameTile(tile,schedule_->tiles()[tile.index])
            || receipts_[tile.index].complete)
            return fail("Rough glass tile is duplicate, overlapping or outside its immutable schedule");
        if (!receipt.gpuCompleted || !receipt.terminal || !receipt.masked || !receipt.tail
            || !receipt.objectIds || !receipt.materialIds || !receipt.normal || !receipt.color)
            return fail("Rough glass tile lacks completed owned native attachments");
        const auto extent=[&](const NativeRawPassPixels& pixels) {
            return pixels.width==tile.width && pixels.height==tile.height;
        };
        if (!extent(*receipt.normal) || !extent(*receipt.color))
            return fail("Rough glass tile Color or Normal extent differs from its scheduled scratch extent");
        NativeFixtureSceneStats stats;
        if (!smoothglass::completeRoughFrame(schedule_->prepared(),tile.width,tile.height,
                *receipt.terminal,*receipt.masked,*receipt.tail,*receipt.objectIds,*receipt.materialIds,stats,error)
            || !smoothglass::completeFrame(*receipt.normal,error))
        { invalidated_=true;return false; }
        if (key.runtime->linearColor) {
            if (!smoothglass::finiteColorFrame(*receipt.color,error)) { invalidated_=true;return false; }
        } else if (receipt.color->format!=NativeTexturePixelFormat::Rgba8Unorm
            || receipt.color->bytes.size()!=tile.pixels()*4)
            return fail("Rough glass tile requires complete owned unorm preview Color");
        const bool selected=smoothglass::multipleScatterCamera(schedule_->prepared());
        if(selected && (!receipt.nativeWork || receipt.nativeWork->size()!=tile.pixels()))
            return fail("Rough glass v2 tile lacks completed native invocation counters");
        TilePower power;
        power.complete=true;power.coveredPixels=stats.roughGlassCoveredPixels;
        power.maximumEnergyResidual=stats.roughGlassMaximumEnergyResidual;
        power.maximumUnresolvedPower=stats.roughGlassMaximumUnresolvedPower;
        power.absorbedTotalPower=stats.roughGlassAbsorbedTotalPower;
        power.maximumAbsorbedPower=stats.roughGlassMaximumAbsorbedPower;
        power.absorbedPixelsOverLimit=stats.roughGlassAbsorbedPixelsOverLimit;
        for (std::size_t pixel=0;pixel<tile.pixels();++pixel) {
            std::uint32_t object=0;
            std::memcpy(&object,receipt.objectIds->bytes.data()+pixel*4,4);
            if(selected) {
                const auto& work=(*receipt.nativeWork)[pixel];
                const bool glass=object==schedule_->prepared().object.value;
                if(work.selections!=1 || work.starts!=(glass?1u:0u) || work.queries<1
                    || work.queries>smoothglass::kMaximumRoughClosestHitQueries
                    || work.events>smoothglass::kMaximumCameraMicroEvents
                    || (!glass && (work.queries!=1 || work.events!=0)))
                    return fail("Rough glass v2 native selection or transport invocation count is invalid");
                ++power.selections;power.starts+=work.starts;
            }
            if (object!=schedule_->prepared().object.value) continue;
            std::array<float,4> t{},m{},r{};
            std::memcpy(t.data(),receipt.terminal->bytes.data()+pixel*16,16);
            std::memcpy(m.data(),receipt.masked->bytes.data()+pixel*16,16);
            std::memcpy(r.data(),receipt.tail->bytes.data()+pixel*16,16);
            if(selected && ((*receipt.nativeWork)[pixel].queries!=m[3]
                || (*receipt.nativeWork)[pixel].events!=r[3]))
                return fail("Rough glass v2 native executed work differs from its shared result receipt");
            power.queries+=static_cast<std::uint64_t>(m[3]);
            if(smoothglass::multipleScatterCamera(schedule_->prepared()))power.events+=static_cast<std::uint64_t>(r[3]);
            for (std::size_t c=0;c<3;++c) { power.terminal[c]+=t[c];power.masked[c]+=m[c];power.tail[c]+=r[c]; }
        }
        // Store only bounded scalar receipts. Recompute in schedule order so
        // completion order cannot change floating-point aggregation or replay.
        receipts_[tile.index]=power;
        Aggregate next;
        for (std::size_t i=0;i<schedule_->tiles().size();++i) if (receipts_[i].complete) {
            const auto& value=receipts_[i];const auto& job=schedule_->tiles()[i];
            ++next.completedTiles;next.completedPixels+=job.pixels();
            next.coveredPixels+=value.coveredPixels;
            next.reservedQueries+=job.reservedQueries();next.receiptClosestHitQueries+=value.queries;
            next.microEvents+=value.events;next.selectionQueries+=value.selections;next.transportStarts+=value.starts;
            next.peakReceiptBytes=std::max(next.peakReceiptBytes,job.receiptBytes());
            next.processedReceiptBytes+=job.receiptBytes();
            next.maximumEnergyResidual=std::max(next.maximumEnergyResidual,value.maximumEnergyResidual);
            next.maximumUnresolvedPower=std::max(next.maximumUnresolvedPower,value.maximumUnresolvedPower);
            next.absorbedTotalPower+=value.absorbedTotalPower;
            next.maximumAbsorbedPower=std::max(next.maximumAbsorbedPower,value.maximumAbsorbedPower);
            next.absorbedPixelsOverLimit+=value.absorbedPixelsOverLimit;
            for (std::size_t c=0;c<3;++c) { next.terminal[c]+=value.terminal[c];next.masked[c]+=value.masked[c];next.tail[c]+=value.tail[c]; }
        }
        next.microWork=next.microEvents*smoothglass::kCameraMicroWorkPerEvent;
        next.source.fill(static_cast<double>(next.coveredPixels));
        aggregate_=next;
        return true;
    }
    // Backends publish their GPU-owned full-size frame only after this returns
    // true. No texture is assembled here and partial aggregates certify no frame.
    bool publish(const FrameKey& key,NativeFixtureSceneStats& stats,Aggregate& output,std::string& error)
    {
        if (invalidated_ || !schedule_ || !sameKey(key,schedule_->key()))
        { invalidated_=true;error="Rough glass cannot publish a stale or invalidated tiled frame";return false; }
        if (aggregate_.completedTiles!=schedule_->tiles().size() || aggregate_.completedPixels!=schedule_->pixels()
            || aggregate_.reservedQueries!=schedule_->reservedQueries())
        { error="Rough glass tiled frame is incomplete";return false; }
        auto candidate=stats;
        candidate.roughGlassReceiptComplete=true;
        candidate.roughGlassCoveredPixels=aggregate_.coveredPixels;
        // This is resident GPU scratch, not the sum of sequential allocations.
        candidate.roughGlassReceiptBytes=aggregate_.peakReceiptBytes;
        candidate.roughGlassMaximumEnergyResidual=aggregate_.maximumEnergyResidual;
        candidate.roughGlassMaximumUnresolvedPower=aggregate_.maximumUnresolvedPower;
        candidate.roughGlassAbsorbedTotalPower=aggregate_.absorbedTotalPower;
        candidate.roughGlassMaximumAbsorbedPower=aggregate_.maximumAbsorbedPower;
        candidate.roughGlassAbsorbedPixelsOverLimit=aggregate_.absorbedPixelsOverLimit;
        if (aggregate_.absorbedTotalPower>0)
            std::fprintf(stderr,"[rough-glass] absorbed at bounce limit: total=%.6g max=%.6g pixels-over-0.0001=%llu\n",
                aggregate_.absorbedTotalPower,aggregate_.maximumAbsorbedPower,
                static_cast<unsigned long long>(aggregate_.absorbedPixelsOverLimit));
        stats=candidate;output=aggregate_;
        return true;
    }
private:
    struct TilePower {
        bool complete=false;
        std::uint64_t coveredPixels=0;
        double maximumEnergyResidual=0,maximumUnresolvedPower=0,absorbedTotalPower=0,maximumAbsorbedPower=0;
        std::uint64_t absorbedPixelsOverLimit=0;
        std::array<double,3> terminal {},masked {},tail {};
        std::uint64_t queries=0,events=0,selections=0,starts=0;
    };
    std::shared_ptr<const Schedule> schedule_;
    std::unique_ptr<TilePower[]> receipts_;
    Aggregate aggregate_;
    bool invalidated_=false;
};
// The backend supplies the complete resident attachment stride and all other
// retained bytes, including any tile depth/identity staging targets. This is
// admission, not proof that resources were allocated.
inline bool admitMemory(const Schedule& schedule,std::uint64_t fullFrameBytesPerPixel,
                        std::uint64_t additionalResidentBytes,FrameMemoryAdmission& admission,std::string& error)
{
    FrameMemoryAdmissionRequest request;
    request.byteBudget=renderpassoutput::kMaximumTotalBytes;
    request.classes[request.classCount++]={schedule.key().width,schedule.key().height,fullFrameBytesPerPixel,1};
    request.classes[request.classCount++]={schedule.maximumTilePixels(),1,
        smoothglass::kRoughReceiptBytesPerPixel+kTileReadbackBytesPerPixel,1};
    // Account schedule descriptors, completion flags and per-tile power totals.
    const auto metadata=schedule.retainedBytes()+sizeof(Completion)
        +std::uint64_t(schedule.tiles().size())*Completion::metadataBytesPerTile();
    if (additionalResidentBytes>std::numeric_limits<std::uint64_t>::max()-metadata)
    { admission={}; admission.failure=FrameMemoryAdmissionFailure::byteCountOverflow;
      error="Rough glass retained-resource memory accounting overflow"; return false; }
    request.classes[request.classCount++]={1,1,additionalResidentBytes+metadata,1};
    return admitFrameMemory(request,admission,error);
}
} // namespace arbitgpu::roughglasstiles
