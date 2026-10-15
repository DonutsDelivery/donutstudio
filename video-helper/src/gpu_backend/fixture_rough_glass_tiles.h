#pragma once

#include "fixture_smooth_glass.h"
#include "frame_memory_budget.h"

namespace arbitgpu::roughglasstiles
{
inline constexpr std::uint64_t kMaximumTilePixels = smoothglass::kMaximumPixelRayQueries
    / smoothglass::kMaximumRoughFrameRayQueries;
inline constexpr std::uint32_t kMaximumFrameJobs = 8192;
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
        std::uint64_t jobs=kMaximumFrameJobs+1,area=0;
        // Minimize rectangular jobs, then prefer larger area and wider tiles.
        // This choice is independent of visible glass, GPU timing and completion.
        for (std::uint32_t w=1;w<=std::min<std::uint64_t>(key.width,tilePixelLimit);++w) {
            const auto h=static_cast<std::uint32_t>(std::min<std::uint64_t>(key.height,tilePixelLimit/w));
            const auto count=((std::uint64_t(key.width)+w-1)/w)*((std::uint64_t(key.height)+h-1)/h);
            const auto size=std::uint64_t(w)*h;
            if (count<jobs || (count==jobs && (size>area || (size==area && w>tileWidth))))
            { jobs=count;area=size;tileWidth=w;tileHeight=h; }
        }
        if (jobs>kMaximumFrameJobs)
        { error="Rough glass tiled frame exceeds its finite native job bound"; return false; }
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
    // Sums of normalized source power over visible glass pixels, independently
    // for each wavelength. These are not radiance or convergence estimates.
    std::array<double,3> source {},terminal {},masked {},tail {};
    double maximumEnergyResidual = 0,maximumUnresolvedPower = 0;
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
            for (std::size_t c=0;c<3;++c) { next.terminal[c]+=value.terminal[c];next.masked[c]+=value.masked[c];next.tail[c]+=value.tail[c]; }
        }
        next.microWork=next.microEvents*smoothglass::kCameraMicroWorkPerEvent;
        if(next.microWork>smoothglass::kMaximumCameraMicroWork)
            return fail("Rough glass v2 whole-frame microfacet work budget exhausted");
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
        stats=candidate;output=aggregate_;
        return true;
    }
private:
    struct TilePower {
        bool complete=false;
        std::uint64_t coveredPixels=0;
        double maximumEnergyResidual=0,maximumUnresolvedPower=0;
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
