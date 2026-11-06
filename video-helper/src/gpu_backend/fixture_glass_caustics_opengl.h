#pragma once

#include "fixture_glass_caustics.h"
#include "fixture_preview_budget.h"
#include "fixture_glass_grating_incident.h"
#include "fixture_glass_grating_cell_transport.h"
#include "fixture_glass_grating_cell_coverage.h"
#include "fixture_glass_camera_cell_coverage.h"
#include "fixture_native_camera_pixel_signs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace arbitgpu
{
enum class OpenGlQueryArithmeticRole : std::uint32_t
{ CausticPath,GratingPoint,GratingCamera,CellBoundary,CellCoverage,SelectedCameraPrimary,SelectedCameraTransport,GratingDirect,GlassCameraGratingTrace,Count };
struct OpenGlQueryArithmeticProgramIdentity
{
    struct Stage {unsigned type=0;std::uint64_t fingerprint=0;std::string source;};
    std::uint32_t contractVersion=1,glslVersion=430,contextMajor=0,contextMinor=0,contextProfile=0;
    std::uintptr_t context=0;
    std::uint64_t contextGeneration=0,ownerGeneration=0,rendererGeneration=0;
    unsigned linkedProgram=0,stageCount=0;
    std::array<Stage,2> stages{};
};
struct OpenGlQueryArithmeticEvidence
{
    OpenGlQueryArithmeticRole role=OpenGlQueryArithmeticRole::CausticPath;
    std::shared_ptr<const OpenGlQueryArithmeticProgramIdentity> program;
    glasscaustics::SubmissionIdentity submission;
    std::array<std::shared_ptr<const OpenGlQueryArithmeticEvidence>,2> parents{};
};
// Producer-owned linked source and completed submission for sceneClosestHit
// and its existing renamed glass copy. This does not qualify the legacy shadow
// query or all shader arithmetic, and grants no interval/camera/Color authority.
bool readOpenGlQueryArithmeticEvidenceForFixture(const NativeFixtureSceneFrame& frame,OpenGlQueryArithmeticRole role,
    std::shared_ptr<const OpenGlQueryArithmeticEvidence>& output,std::string& error);
bool validateOpenGlQueryArithmeticEvidenceForFixture(const NativeFixtureSceneFrame& frame,
    const std::shared_ptr<const OpenGlQueryArithmeticEvidence>& evidence,const glasscaustics::SubmissionIdentity& expected,std::string& error);

struct OpenGlGlassCausticDiagnostics
{
    std::uint64_t admittedFrameBytes=0,nativeStorageBytes=0,queryWork=0,nativeTemporaryBytes=0;
    glasscaustics::DepositionSchedule deposition;
    double baseMilliseconds=0,rasterMilliseconds=0,traceMilliseconds=0,sortMilliseconds=0;
    double reduceMilliseconds=0,readbackMilliseconds=0,resolveMilliseconds=0,frameMilliseconds=0;
    // Splat frames: sortMilliseconds covers tile keys, sort and segments, and
    // reduceMilliseconds the gather. splatVersion 0 is single-pixel deposition.
    std::uint32_t splatVersion=0;
    std::uint64_t splatGatherWork=0;
    double normaliserMilliseconds=0;
    // The splat caps overflowed and the frame re-ran as single-pixel deposits.
    bool splatFallback=false;
    // Preview frames: no host receipt, and whether the spectrum or query program was reused.
    // Preview stages do not wait, so their host times are encode times. The GPU times come
    // from timestamps; deposition covers the splat normaliser, sort and reduce or gather.
    bool preview=false,spectrumReused=false,queryProgramReused=false;
    std::uint64_t workCount=0;
    double gpuRasterMilliseconds=0,gpuTraceMilliseconds=0,gpuDepositionMilliseconds=0,gpuResolveMilliseconds=0;
};
// Who consumes a caustic render. Export keeps the full entry schedule, a fence after
// every stage and band, and the host receipt: paths, deposits, keys and spectrum mapped,
// decoded and validated, and the resolved Color read back and checked finite. Preview
// only draws Color. It traces a coarser entry schedule, flushes between stages, and
// reuses the spectrum of an unchanged frame. Up to kPreviewCausticUnfencedPixels it never
// waits on the caustic: the submission is fenced and polled on later frames, the splat
// counters are read only after that fence signals, and the resolve is not fenced. Larger
// extents still wait on each gather band. Nothing else is read back.
enum class OpenGlCausticRenderUse : std::uint8_t { Export=0, Preview };
// Preview entry subdivisions. Each entry carries subdivisions^-2 of its triangle's
// area, so a coarser schedule keeps the projected power and traces 1/4 of the paths.
inline constexpr std::size_t kPreviewCausticEntrySubdivisions=2;
// Preview gather bands still wait one by one above this extent, where a band nears
// the per-submission work the export bands bound.
inline constexpr std::uint64_t kPreviewCausticUnfencedPixels=2560ull*1440ull;
// The retained preview spectrum is two RGBA32F buffers, 29.5 MB at this extent.
inline constexpr std::uint64_t kPreviewCausticReusePixels=1280ull*720ull;
// Preview frames slower than this log their per-stage timings.
inline constexpr double kPreviewCausticSlowFrameMilliseconds=100;
struct OpenGlCausticRenderPlan
{
    glasscaustics::SchedulingDescription scheduling;
    bool hostReceipt=true,stageFences=true,bandFences=true,reuseSpectrum=false;
};
// exportSchedule is the Accurate render mode (previewbudget::exportBudgets). Its Preview
// keeps the export entry schedule, so the caustic it resolves is the export caustic, and
// still never waits or reads back: the fences, receipt and reuse below stay Preview's.
inline OpenGlCausticRenderPlan openGlCausticRenderPlan(OpenGlCausticRenderUse use,std::uint32_t width,std::uint32_t height,
    bool exportSchedule=false)
{
    OpenGlCausticRenderPlan plan;
    // Tier K S4: a Fast export (Preview budget) takes the Preview plan's
    // reductions; the Export draw under Export budgets - the default path -
    // takes this default plan exactly as before.
    if(use!=OpenGlCausticRenderUse::Preview && previewbudget::exportBudgets())return plan;
    const auto pixels=std::uint64_t(width)*height;
    if(!exportSchedule)plan.scheduling.entrySubdivisions=kPreviewCausticEntrySubdivisions;
    plan.hostReceipt=false;plan.stageFences=false;
    plan.bandFences=pixels>kPreviewCausticUnfencedPixels;plan.reuseSpectrum=pixels<=kPreviewCausticReusePixels;
    return plan;
}
// A preview frame splats only when the export schedule of the same frame would, so
// the smaller entry budget never turns a single-pixel caustic into a splat one.
inline bool openGlCausticPreviewSplatAllowed(const glasscaustics::Frame& frame)
{
    const auto subdivisions=std::uint64_t(frame.scheduling.entrySubdivisions);
    if(frame.work.empty()||!subdivisions||glasscaustics::kEntrySubdivisions%subdivisions
        ||frame.bounds.maximumDeposits%frame.work.size())return false;
    const auto ratio=(glasscaustics::kEntrySubdivisions/subdivisions)*(glasscaustics::kEntrySubdivisions/subdivisions);
    glasscaustics::SplatSchedule full;std::string error;
    return glasscaustics::calculateSplatSchedule(frame.width,frame.height,frame.work.size()*ratio,
        frame.bounds.maximumDeposits/frame.work.size(),glasscaustics::splatKeySlots(frame),full,error);
}
// The native GL limits a caustic frame must fit. Software GL such as llvmpipe caps one
// storage block at 128 MiB, which a full deposit receipt buffer can pass.
struct OpenGlCausticDeviceLimits
{
    std::uint64_t maximumStorageBlockBytes=0,maximumWorkGroupsX=0,maximumTextureSize=0;
    std::uint64_t maximumInvocations=0,maximumLocalSizeX=0,storageBindings=0,computeStorageBlocks=0;
    std::uint64_t maximumWorkGroupsY=0,maximumLocalSizeY=0;
};
inline bool operator==(const OpenGlCausticDeviceLimits& a,const OpenGlCausticDeviceLimits& b)
{
    return a.maximumStorageBlockBytes==b.maximumStorageBlockBytes&&a.maximumWorkGroupsX==b.maximumWorkGroupsX
        &&a.maximumTextureSize==b.maximumTextureSize&&a.maximumInvocations==b.maximumInvocations
        &&a.maximumLocalSizeX==b.maximumLocalSizeX&&a.storageBindings==b.storageBindings
        &&a.computeStorageBlocks==b.computeStorageBlocks&&a.maximumWorkGroupsY==b.maximumWorkGroupsY
        &&a.maximumLocalSizeY==b.maximumLocalSizeY;
}
inline bool operator!=(const OpenGlCausticDeviceLimits& a,const OpenGlCausticDeviceLimits& b){return !(a==b);}
// Test only. ARBIT_TEST_GL_CAUSTIC_LIMIT_SCALE scales the size limits a caustic frame is
// checked against, so a candidate on a large GPU can force a limit refusal. The value is a
// number in (0,1). Text that is not a number, zero, a negative value, NaN, or 1 or more is
// ignored. Ignored means no override: the queried limits stay as GL reported them.
inline std::optional<double> parseOpenGlCausticLimitScale(const char* text)
{
    if(!text||!*text)return std::nullopt;
    char* end=nullptr;const double value=std::strtod(text,&end);
    if(end==text||*end!='\0'||!std::isfinite(value)||value<=0||value>=1)return std::nullopt;
    return value;
}
// The limits with every size a demand is compared against scaled down: the storage block
// bytes, the X and Y work group counts and the texture size. Capability counts (bindings,
// blocks, invocations, local sizes) keep their values, so the refusal names a size. A
// nonzero limit stays at least 1, and a scale outside (0,1) returns the limits unchanged.
inline OpenGlCausticDeviceLimits scaleOpenGlCausticDeviceLimits(OpenGlCausticDeviceLimits limits,double scale)
{
    if(!(scale>0&&scale<1))return limits;
    const auto scaled=[scale](std::uint64_t value)
    {return value?std::max<std::uint64_t>(1,std::uint64_t(double(value)*scale)):value;};
    limits.maximumStorageBlockBytes=scaled(limits.maximumStorageBlockBytes);
    limits.maximumWorkGroupsX=scaled(limits.maximumWorkGroupsX);
    limits.maximumWorkGroupsY=scaled(limits.maximumWorkGroupsY);
    limits.maximumTextureSize=scaled(limits.maximumTextureSize);
    return limits;
}
// An Accurate Preview caustic over a GL limit is refused, as its export would be; it is
// never reduced or dropped. The viewport asks for the frame again every tick, so the
// refusal is latched: a frame with the same key returns the same named error before the
// CPU caustic prepare, the schedule build or any GPU work. The key holds every input the
// schedule and the limit checks read: the scene content fingerprint, prepared resources,
// light, glass, policy, schedule, extent, mode and GL limits, and from the runtime inputs
// the evaluated camera (its schedule identity), the evaluated source light record and the
// object instance transforms. Time is not in the key: the schedule reads it only through
// those three, so a still frame of an animated scene keeps its latch, and a frame whose
// camera, light or objects moved retries. A frame whose runtime inputs cannot be keyed
// is never latched.
// The scene enters the key as a content fingerprint, not as the snapshot address: the
// viewport alternates between live Visual3DScene snapshot objects across ticks, so an
// address key can never match the next frame (verify81: every held-key miss named scene
// and nothing else). The fingerprint covers the scene content the schedule and its limit
// checks read. Motion-only previous-frame arrays and the texture texel values are
// excluded on purpose: neither changes the storage demand a limit refusal names, so a
// latch that ignores them still refuses the same frame. The scene's light records are
// excluded for the same reason: the source light enters the key as its evaluated record
// (below), and no other light can change the storage demand a limit refusal names.
// verify83 named the lights array as the only segment that churns per tick in the
// lighthouse starter, so including it would keep every animated scene unlatched while
// nothing the refusal depends on moves. The content splits into named segment hashes so
// a held-key miss can print which family moved (verify82: the fingerprint alternated
// every tick and no segment was named).
inline std::uint64_t causticFnvFold(std::uint64_t hash,const void* data,std::size_t bytes)
{
    const auto* p=static_cast<const unsigned char*>(data);
    for(std::size_t i=0;i<bytes;++i){hash^=p[i];hash*=1099511628211ull;}
    return hash;
}
struct OpenGlCausticSceneSegments
{
    std::uint64_t scalars=0,vertices=0,indices=0,objects=0,materials=0,textures=0,cameras=0,instances=0;
};
inline OpenGlCausticSceneSegments causticSceneContentSegments(const HarmonicMIDI::grid::Visual3DScene& scene)
{
    constexpr std::uint64_t kOffset=14695981039346656037ull;
    OpenGlCausticSceneSegments out;
    out.scalars=causticFnvFold(kOffset,&scene.schemaVersion,sizeof(scene.schemaVersion));
    out.scalars=causticFnvFold(out.scalars,&scene.id,sizeof(scene.id));
    out.scalars=causticFnvFold(out.scalars,&scene.activeCamera,sizeof(scene.activeCamera));
    out.scalars=causticFnvFold(out.scalars,&scene.ambientColor,sizeof(scene.ambientColor));
    out.scalars=causticFnvFold(out.scalars,&scene.instancedObject,sizeof(scene.instancedObject));
    const auto counts=std::array<std::size_t,9>{scene.vertexCount,scene.indexCount,scene.objectCount,
        scene.materialCount,scene.textureCount,scene.textureTexelCount,scene.lightCount,
        scene.cameraCount,scene.objectInstanceCount};
    out.scalars=causticFnvFold(out.scalars,counts.data(),counts.size()*sizeof(counts[0]));
    out.vertices=causticFnvFold(kOffset,scene.vertices.data(),scene.vertexCount*sizeof(scene.vertices[0]));
    out.indices=causticFnvFold(kOffset,scene.indices.data(),scene.indexCount*sizeof(scene.indices[0]));
    out.objects=causticFnvFold(kOffset,scene.objects.data(),scene.objectCount*sizeof(scene.objects[0]));
    out.materials=causticFnvFold(kOffset,scene.materials.data(),scene.materialCount*sizeof(scene.materials[0]));
    out.textures=causticFnvFold(kOffset,scene.textures.data(),scene.textureCount*sizeof(scene.textures[0]));
    out.cameras=causticFnvFold(kOffset,scene.cameras.data(),scene.cameraCount*sizeof(scene.cameras[0]));
    out.instances=causticFnvFold(kOffset,scene.objectInstances.data(),scene.objectInstanceCount*sizeof(scene.objectInstances[0]));
    return out;
}
inline std::uint64_t causticSceneFingerprintOf(const OpenGlCausticSceneSegments& segments)
{
    const std::array<std::uint64_t,8> parts={segments.scalars,segments.vertices,segments.indices,
        segments.objects,segments.materials,segments.textures,segments.cameras,segments.instances};
    return causticFnvFold(14695981039346656037ull,parts.data(),parts.size()*sizeof(parts[0]));
}
inline std::uint64_t causticSceneContentFingerprint(const HarmonicMIDI::grid::Visual3DScene& scene)
{return causticSceneFingerprintOf(causticSceneContentSegments(scene));}
struct OpenGlCausticRefusalKey
{
    std::uint64_t sceneContent=0;
    std::uintptr_t resources=0,context=0;std::uint64_t rendererGeneration=0;
    std::uint32_t source=0,glassObject=0,glassMaterial=0,multipleScatter=0;
    std::string policy;
    glasscaustics::SchedulingDescription scheduling;
    std::uint32_t width=0,height=0;
    bool exportBudgets=false;
    OpenGlCausticDeviceLimits limits;
    // Runtime inputs, as the schedule build evaluates them.
    std::string cameraIdentity;
    HarmonicMIDI::grid::SceneLightRecord light{};
    std::vector<opaquevisibility::Instance> instances;
};
// The source record fields a caustic schedule reads. Sky and environment image belong to
// other light kinds; a caustic source is a directional spectral source.
inline bool sameCausticRefusalLight(const HarmonicMIDI::grid::SceneLightRecord& a,const HarmonicMIDI::grid::SceneLightRecord& b)
{
    const auto bits=[](const auto& x,const auto& y){return std::memcmp(&x,&y,sizeof(x))==0;};
    return a.id==b.id&&a.kind==b.kind&&bits(a.transform.translation,b.transform.translation)
        &&bits(a.transform.rotation,b.transform.rotation)&&bits(a.transform.scale,b.transform.scale)
        &&bits(a.color,b.color)&&bits(a.intensity,b.intensity)&&bits(a.range,b.range)
        &&bits(a.innerConeAngle,b.innerConeAngle)&&bits(a.outerConeAngle,b.outerConeAngle)
        &&sameSpectralSource(a.spectralSource,b.spectralSource)
        &&HarmonicMIDI::grid::sameLightShapeAndVisibility(a,b);
}
inline bool sameCausticRefusalInstances(const std::vector<opaquevisibility::Instance>& a,
    const std::vector<opaquevisibility::Instance>& b)
{
    return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin(),[](const auto& x,const auto& y) {
        return x.object==y.object&&std::memcmp(x.world.data(),y.world.data(),sizeof(x.world))==0;});
}
inline bool operator==(const OpenGlCausticRefusalKey& a,const OpenGlCausticRefusalKey& b)
{
    return a.sceneContent==b.sceneContent&&a.resources==b.resources&&a.context==b.context&&a.rendererGeneration==b.rendererGeneration
        &&a.source==b.source&&a.glassObject==b.glassObject&&a.glassMaterial==b.glassMaterial
        &&a.multipleScatter==b.multipleScatter&&a.policy==b.policy&&a.scheduling.version==b.scheduling.version
        &&a.scheduling.entrySubdivisions==b.scheduling.entrySubdivisions&&a.width==b.width&&a.height==b.height
        &&a.exportBudgets==b.exportBudgets&&a.limits==b.limits&&a.cameraIdentity==b.cameraIdentity
        &&sameCausticRefusalLight(a.light,b.light)&&sameCausticRefusalInstances(a.instances,b.instances);
}
inline bool operator!=(const OpenGlCausticRefusalKey& a,const OpenGlCausticRefusalKey& b){return !(a==b);}
struct OpenGlCausticRefusalLatch
{
    bool latched=false;
    OpenGlCausticRefusalKey key;
    std::string error;
    // The latched error for this key, or null. A different key clears the latch, so the
    // frame retries in full.
    const std::string* find(const OpenGlCausticRefusalKey& candidate)
    {
        if(latched&&key==candidate)return &error;
        clear();return nullptr;
    }
    void latch(OpenGlCausticRefusalKey candidate,std::string reason)
    {latched=true;key=std::move(candidate);error=std::move(reason);}
    void clear(){latched=false;key={};error.clear();}
};
// The refusal an Accurate Preview shows. It names the limit and the way out.
inline std::string openGlAccurateCausticRefusal(const std::string& reason)
{
    return reason+". Accurate Preview matches export, so it does not reduce the caustic;"
        " switch Render Mode to Fast to preview this scene";
}
// The storage, dispatch and texture sizes of one caustic frame. bytes[5] holds keyCount
// keys: the deposition key count, or the splat key count once a frame splats.
struct OpenGlCausticStorageDemand
{
    std::uint64_t workCount=0,textureExtent=0,depositionKeys=0;
    std::array<std::uint64_t,7> bytes{};
};
inline constexpr std::array<const char*,7> kOpenGlCausticStorageNames{
    "path receipts","deposit receipts","spectrum","work texels","triangle texels","deposition keys","resolve spectrum"};
inline constexpr std::uint64_t kOpenGlCausticMaximumPathTexels=2147483647u;
inline OpenGlCausticStorageDemand openGlCausticStorageDemand(const glasscaustics::Frame& frame,std::uint64_t keyCount)
{
    namespace gc=glasscaustics;
    OpenGlCausticStorageDemand demand;
    const auto pixels=std::uint64_t(frame.width)*frame.height;
    demand.workCount=frame.work.size();demand.depositionKeys=frame.bounds.deposition.paddedKeyCount;
    demand.textureExtent=std::max({std::uint64_t(frame.width),std::uint64_t(frame.height),
        std::uint64_t(opaquevisibility::kTextureWidth),std::uint64_t(frame.query.height)});
    demand.bytes={(frame.work.size()*gc::kPathReceiptStride+frame.bounds.historyTexels)*sizeof(gc::Texel),
        frame.bounds.maximumDeposits*gc::kDepositReceiptStride*sizeof(gc::Texel),pixels*4*sizeof(float),
        frame.workTexels.size()*sizeof(gc::Texel),frame.triangleTexels.size()*sizeof(gc::Texel),
        keyCount*sizeof(gc::DepositionKey),pixels*4*sizeof(float)};
    return demand;
}
inline std::string openGlCausticMebibytes(std::uint64_t bytes)
{
    char text[32];std::snprintf(text,sizeof(text),"%.1f MiB",double(bytes)/(1024.0*1024.0));return text;
}
// Names the first limit the demand exceeds, with the needed size and the limit, or returns
// an empty string when the frame fits. The deposition key buffer and its dispatch repeat
// admitDepositionDevice at 32 threads and four bands per buffer.
inline std::string openGlCausticLimitFailure(const OpenGlCausticDeviceLimits& limits,const OpenGlCausticStorageDemand& demand)
{
    const auto count=[](std::uint64_t value){return std::to_string(value);};
    if(limits.maximumInvocations<32||limits.maximumLocalSizeX<32)
        return "compute work group of 32 threads exceeds max invocations "+count(limits.maximumInvocations)+" or local size "+count(limits.maximumLocalSizeX);
    if(limits.storageBindings<7||limits.computeStorageBlocks<5)
        return "7 storage bindings and 5 compute blocks exceed max bindings "+count(limits.storageBindings)+" or compute blocks "+count(limits.computeStorageBlocks);
    if(!limits.maximumStorageBlockBytes||!limits.maximumWorkGroupsX)return "GL reports no storage block size or work group count";
    for(std::size_t i=0;i<demand.bytes.size();++i)
        if(demand.bytes[i]>limits.maximumStorageBlockBytes)
            return std::string(kOpenGlCausticStorageNames[i])+" need "+openGlCausticMebibytes(demand.bytes[i])
                +", max SSBO "+openGlCausticMebibytes(limits.maximumStorageBlockBytes);
    // The walk indexes path texels with a GLSL int, a representability bound
    // of the shading language rather than of the scene.
    if(demand.bytes[0]/sizeof(glasscaustics::Texel)>kOpenGlCausticMaximumPathTexels)
        return "path receipts need "+count(demand.bytes[0]/sizeof(glasscaustics::Texel))+" texels, past the "
            +count(kOpenGlCausticMaximumPathTexels)+" a GLSL int index reaches";
    const auto keyBytes=demand.depositionKeys*sizeof(glasscaustics::DepositionKey);
    if(keyBytes>limits.maximumStorageBlockBytes)
        return "deposition keys need "+openGlCausticMebibytes(keyBytes)+", max SSBO "+openGlCausticMebibytes(limits.maximumStorageBlockBytes);
    if(demand.workCount>limits.maximumWorkGroupsX*32)
        return "trace needs "+count((demand.workCount+31)/32)+" work groups, max work group count "+count(limits.maximumWorkGroupsX);
    if((demand.depositionKeys+31)/32>limits.maximumWorkGroupsX)
        return "deposition needs "+count((demand.depositionKeys+31)/32)+" work groups, max work group count "+count(limits.maximumWorkGroupsX);
    if(demand.textureExtent>limits.maximumTextureSize)
        return "texture extent "+count(demand.textureExtent)+" exceeds max texture size "+count(limits.maximumTextureSize);
    return {};
}
// The extra terms of a splat frame: binding 7, 16x16 gather groups (four per tile), and
// dispatches over the splat key count and every trace slot.
struct OpenGlCausticSplatDemand
{
    std::uint64_t scratchWords=0,keyCount=0,slots=0,maximumDeposits=0,tilesX=0,tilesY=0;
};
// Names the first splat term past its limit, with the need and the limit, or returns empty.
inline std::string openGlCausticSplatLimitFailure(const OpenGlCausticDeviceLimits& limits,const OpenGlCausticSplatDemand& demand)
{
    const auto count=[](std::uint64_t value){return std::to_string(value);};
    const auto side=std::uint64_t(glasscaustics::kSplatGatherGroup);
    const auto groupsPerTile=std::uint64_t(glasscaustics::kSplatTile/glasscaustics::kSplatGatherGroup);
    const auto groups=[&](std::uint64_t work){return (work+31)/32;};
    if(limits.storageBindings<8)return "splat needs 8 storage bindings, max "+count(limits.storageBindings);
    if(limits.computeStorageBlocks<7)return "splat needs 7 compute storage blocks, max "+count(limits.computeStorageBlocks);
    if(limits.maximumInvocations<side*side)
        return "splat needs "+count(side*side)+" invocations per work group, max "+count(limits.maximumInvocations);
    if(limits.maximumLocalSizeX<side)return "splat needs local size X "+count(side)+", max "+count(limits.maximumLocalSizeX);
    if(limits.maximumLocalSizeY<side)return "splat needs local size Y "+count(side)+", max "+count(limits.maximumLocalSizeY);
    if(demand.scratchWords*sizeof(std::uint32_t)>limits.maximumStorageBlockBytes)
        return "splat scratch needs "+openGlCausticMebibytes(demand.scratchWords*sizeof(std::uint32_t))
            +", max SSBO "+openGlCausticMebibytes(limits.maximumStorageBlockBytes);
    if(demand.keyCount>std::uint64_t(std::numeric_limits<int>::max()))
        return "splat needs "+count(demand.keyCount)+" keys, max "+count(std::uint64_t(std::numeric_limits<int>::max()));
    if(demand.slots>demand.maximumDeposits)
        return "splat needs "+count(demand.slots)+" trace slots, max "+count(demand.maximumDeposits)+" deposit slots";
    if(groups(demand.keyCount)>limits.maximumWorkGroupsX)
        return "splat keys need "+count(groups(demand.keyCount))+" work groups, max work group count "+count(limits.maximumWorkGroupsX);
    if(groups(demand.slots)>limits.maximumWorkGroupsX)
        return "splat slots need "+count(groups(demand.slots))+" work groups, max work group count "+count(limits.maximumWorkGroupsX);
    if(demand.tilesX*groupsPerTile>limits.maximumWorkGroupsX)
        return "splat gather needs "+count(demand.tilesX*groupsPerTile)+" X work groups, max "+count(limits.maximumWorkGroupsX);
    if(demand.tilesY*groupsPerTile>limits.maximumWorkGroupsY)
        return "splat gather needs "+count(demand.tilesY*groupsPerTile)+" Y work groups, max "+count(limits.maximumWorkGroupsY);
    return {};
}
// Preview steps down one entry subdivision level when the plan does not fit: 4 to 2, 2 to 1.
// 0 means no smaller valid schedule.
inline std::size_t openGlCausticReducedEntrySubdivisions(std::size_t subdivisions)
{return subdivisions>1&&glasscaustics::validScheduling({glasscaustics::kScheduleVersion,subdivisions/2})?subdivisions/2:0;}
// Full renders the planned schedule. Reduced renders entrySubdivisions instead. Off renders
// the scene without the caustic and carries reason as a non-fatal warning. Refused is the
// Export answer when the plan does not fit, so an export never silently drops the caustic.
enum class OpenGlCausticAdmission : std::uint8_t { Full, Reduced, Off, Refused };
struct OpenGlCausticAdmissionDecision
{
    OpenGlCausticAdmission admission=OpenGlCausticAdmission::Refused;
    std::size_t entrySubdivisions=0;
    std::string reason;
};
// planned is the demand at planSubdivisions. reduced is the demand at
// openGlCausticReducedEntrySubdivisions(planSubdivisions), or null when none was built.
inline OpenGlCausticAdmissionDecision decideOpenGlCausticAdmission(OpenGlCausticRenderUse use,
    const OpenGlCausticDeviceLimits& limits,std::size_t planSubdivisions,
    const OpenGlCausticStorageDemand& planned,const OpenGlCausticStorageDemand* reduced)
{
    OpenGlCausticAdmissionDecision decision;
    const auto failure=openGlCausticLimitFailure(limits,planned);
    if(failure.empty()) {decision.admission=OpenGlCausticAdmission::Full;decision.entrySubdivisions=planSubdivisions;return decision;}
    if(use!=OpenGlCausticRenderUse::Preview)
    {decision.reason="OpenGL caustic admitted storage exceeds native GL limits: "+failure;return decision;}
    const auto smaller=openGlCausticReducedEntrySubdivisions(planSubdivisions);
    if(reduced&&smaller)
    {
        const auto reducedFailure=openGlCausticLimitFailure(limits,*reduced);
        if(reducedFailure.empty())
        {
            decision.admission=OpenGlCausticAdmission::Reduced;decision.entrySubdivisions=smaller;
            decision.reason="caustic reduced to "+std::to_string(smaller)+" entry subdivisions: "+failure;return decision;
        }
        decision.admission=OpenGlCausticAdmission::Off;decision.reason="caustic off: exceeds GL limits ("+reducedFailure+")";return decision;
    }
    decision.admission=OpenGlCausticAdmission::Off;decision.reason="caustic off: exceeds GL limits ("+failure+")";return decision;
}
// The Preview caustic runs across frames. A frame submits the raster, trace and deposition
// with a fence and never waits on it. Later frames poll the fence, and the spectrum is
// resolved once it signals. The time checked against this budget is the GPU span from the
// raster start to the deposition end, read from timestamps after the fence signals. Without
// timestamps, the submission's age at the poll that found it signalled stands in. A submission
// still running at four times the budget is checked by its age at once. The admission above
// runs from static limits; this ladder adds measured time.
inline constexpr double kPreviewCausticSubmitBudgetMilliseconds=250;
// Consecutive submissions over budget before the ladder steps down one level.
inline constexpr unsigned kPreviewCausticOverBudgetSteps=2;
// One submission past this multiple of the budget steps down at once.
inline constexpr double kPreviewCausticSevereBudgetFactor=4;
// Consecutive submissions within budget before Reduced retries Full.
inline constexpr unsigned kPreviewCausticRetryWithinSteps=120;
// Off retries Reduced after this long. Each retry that falls back to Off doubles it, up to the cap.
inline constexpr double kPreviewCausticOffRetryMilliseconds=30000,kPreviewCausticOffRetryCapMilliseconds=480000;
// True when a submission took longer than the budget. A non-finite time counts as over.
inline bool decidePreviewCausticOverBudget(double elapsedMs,double budgetMs)
{return budgetMs>0&&!(elapsedMs<=budgetMs);}
// Whole milliseconds for a reason string. A negative or non-finite time reads 0.
inline std::string previewCausticMilliseconds(double elapsedMs)
{
    const auto ms=std::isfinite(elapsedMs)&&elapsedMs>0?static_cast<long long>(std::min(elapsedMs,1e15)+.5):0LL;
    return std::to_string(ms)+" ms";
}
// The HUD reason of a Preview frame drawn without its caustic while the submission runs.
inline std::string previewCausticOverBudgetReason(double elapsedMs)
{return "caustic submit over budget ("+previewCausticMilliseconds(elapsedMs)+")";}
enum class OpenGlPreviewCausticLevel : std::uint8_t { Full=0, Reduced, Off };
struct OpenGlPreviewCausticBudget
{
    OpenGlPreviewCausticLevel level=OpenGlPreviewCausticLevel::Full;
    unsigned over=0,within=0;
    // Off: when it was entered and how long it waits before retrying Reduced.
    double offSinceMs=0,offRetryMs=kPreviewCausticOffRetryMilliseconds;
    // A Reduced level entered by an Off retry. Falling back to Off doubles the wait.
    bool retrying=false;
    std::string reason;
};
// Records one submission's elapsed time and steps the level. Full steps to Reduced and
// Reduced to Off after kPreviewCausticOverBudgetSteps consecutive submissions over budget,
// or after one past kPreviewCausticSevereBudgetFactor times the budget. Reduced returns to
// Full after kPreviewCausticRetryWithinSteps within budget. Returns true when the level changed.
inline bool stepPreviewCausticBudget(OpenGlPreviewCausticBudget& state,double elapsedMs,double nowMs,
    double budgetMs=kPreviewCausticSubmitBudgetMilliseconds)
{
    if(state.level==OpenGlPreviewCausticLevel::Off)return false;
    if(!decidePreviewCausticOverBudget(elapsedMs,budgetMs))
    {
        state.over=0;
        if(state.level!=OpenGlPreviewCausticLevel::Reduced||++state.within<kPreviewCausticRetryWithinSteps)return false;
        state.level=OpenGlPreviewCausticLevel::Full;state.within=0;state.retrying=false;
        state.offRetryMs=kPreviewCausticOffRetryMilliseconds;state.reason.clear();return true;
    }
    state.within=0;
    const bool severe=!(elapsedMs<kPreviewCausticSevereBudgetFactor*budgetMs);
    if(++state.over<kPreviewCausticOverBudgetSteps&&!severe)return false;
    state.over=0;
    const auto over="submit over budget ("+previewCausticMilliseconds(elapsedMs)+")";
    if(state.level==OpenGlPreviewCausticLevel::Full)
    {state.level=OpenGlPreviewCausticLevel::Reduced;state.reason="caustic reduced: "+over;return true;}
    if(state.retrying)state.offRetryMs=std::min(2*state.offRetryMs,kPreviewCausticOffRetryCapMilliseconds);
    state.level=OpenGlPreviewCausticLevel::Off;state.offSinceMs=nowMs;state.retrying=false;
    state.reason="caustic off: "+over;return true;
}
// Off retries Reduced once its wait has passed. Returns true when the level changed.
inline bool retryPreviewCausticBudget(OpenGlPreviewCausticBudget& state,double nowMs)
{
    if(state.level!=OpenGlPreviewCausticLevel::Off||nowMs-state.offSinceMs<state.offRetryMs)return false;
    state.level=OpenGlPreviewCausticLevel::Reduced;state.over=0;state.within=0;state.retrying=true;
    state.reason="caustic reduced: retrying after submit over budget";return true;
}
// A completed Preview submission is resolved only onto a frame with the same caustic
// schedule: context, prepared owner generation, extent and the schedule identity, which
// covers the camera, source, instances, triangles, schedule and branch policy. The resident
// water revision is left out on purpose, so the spectrum of the previous water step can be
// drawn beside the current water. With animated water the caustic therefore lags the water by
// one or more frames. That lag is accepted: otherwise each new water step would drop the
// spectrum, and a slow device would never show the caustic.
struct OpenGlCausticScheduleKey
{
    std::uintptr_t context=0;std::uint64_t rendererGeneration=0;
    std::uint32_t width=0,height=0;
    std::string scheduleIdentity;
};
inline bool operator==(const OpenGlCausticScheduleKey& a,const OpenGlCausticScheduleKey& b)
{
    return a.context==b.context&&a.rendererGeneration==b.rendererGeneration&&a.width==b.width&&a.height==b.height
        &&a.scheduleIdentity==b.scheduleIdentity;
}
inline bool operator!=(const OpenGlCausticScheduleKey& a,const OpenGlCausticScheduleKey& b){return !(a==b);}
// A preview spectrum is reused only for the same context, prepared owner generation,
// extent, evaluation time and frame identity. The identity covers the camera, source,
// instance transforms, triangles, schedule and branch policy.
struct OpenGlCausticSpectrumReuseKey
{
    std::uintptr_t context=0;std::uint64_t rendererGeneration=0;
    std::uint32_t width=0,height=0;double timeSeconds=0;
    std::string frameIdentity;
};
inline bool operator==(const OpenGlCausticSpectrumReuseKey& a,const OpenGlCausticSpectrumReuseKey& b)
{
    return a.context==b.context&&a.rendererGeneration==b.rendererGeneration&&a.width==b.width&&a.height==b.height
        &&a.timeSeconds==b.timeSeconds&&a.frameIdentity==b.frameIdentity;
}
inline OpenGlCausticSpectrumReuseKey openGlCausticSpectrumReuseKey(const glasscaustics::Frame& frame,double timeSeconds,
    std::uintptr_t context,std::uint64_t rendererGeneration)
{return {context,rendererGeneration,frame.width,frame.height,timeSeconds,frame.identity};}
enum class OpenGlCausticProgramReadiness : std::uint8_t { Ready, Pending, Failed };
// Starts or finishes the caustic programs of the current context before the caustic
// render needs them. Preview submits them without waiting and returns Pending until the
// driver reports every link complete. Blocking finishes them before it returns. Ready
// programs stay with the context, so a scene re-prepare does not compile them again.
// multipleScatter and rough select the query variant the render will schedule.
OpenGlCausticProgramReadiness prepareOpenGlGlassCausticPrograms(bool multipleScatter,bool rough,
    NativeFixturePrepareUse use,std::string& error);
// The viewport's readiness seam: true once when every submitted, still-unadopted
// caustic program set reads complete, so the idle gate can wake for exactly one
// adopting frame instead of spending its wake-frame cap first on a slow link.
// Report-only -- the next acquire still finishes and publishes the set -- and it
// follows the acquire's extension rule. Needs a current context on the caller's
// thread; false without one, with nothing in flight, or once the completion has
// already been reported for the set in flight.
bool takeOpenGlCausticProgramCompletion();
// Backend-private native route used by the scene-linear Color dispatcher. A Preview
// frame whose caustic exceeds the GL limits renders reduced or without the caustic, and
// degradation receives the reason. Empty means the caustic rendered as planned.
NativeFixtureSceneSubmission renderOpenGlGlassCausticScene(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSceneResources>& resources,
    HarmonicMIDI::grid::SceneLightId source,
    HarmonicMIDI::grid::SceneObjectId glassObject, HarmonicMIDI::grid::SceneMaterialId glassMaterial,
    const glasscaustics::AdmittedBranchPolicy& policy,
    glasscaustics::SchedulingDescription scheduling,
    std::uint32_t width, std::uint32_t height, NativeFixtureSceneRuntimeInputs runtime,
    const glasscaustics::MultipleScatterPolicy* multipleScatterPolicy=nullptr,
    OpenGlCausticRenderUse use=OpenGlCausticRenderUse::Export,
    std::string* degradation=nullptr);
bool readOpenGlGlassCausticReceiptForFixture(const NativeFixtureSceneFrame& frame,
    std::shared_ptr<const glasscaustics::NativeReadback>& receipt,
    glasscaustics::EnergyLedger& energy, std::string& error,
    OpenGlGlassCausticDiagnostics* diagnostics=nullptr);

struct OpenGlGratingIncidentDiagnostics
{
    std::uint64_t admittedBytes=0,nativeBytes=0,baseQueryWork=0,extraQueryWork=0,orderWork=0;
};
// Produces actual-hit/order adjunct receipts only. No combined Color authority.
bool produceOpenGlGlassGratingIncidents(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeGratingReadback& output,std::string& error,
    OpenGlGratingIncidentDiagnostics* diagnostics=nullptr);

// Private eight-band center/corner ray traces only. No footprint or Color grant.
bool produceOpenGlGlassCameraGratingTrace(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeGlassCameraGratingTraceInput>& inputs,
    double evaluationTime,glasscaustics::NativeGlassCameraGratingTraceReadback& output,std::string& error);
// Native fixture inspection only; these handles cannot supply a trace receipt.
bool readOpenGlGlassCameraGratingTraceStorageForFixture(const NativeFixtureSceneFrame& frame,
    std::array<unsigned,5>& buffers,std::uintptr_t& fence,std::string& error);
// One selected directional light at native queried endpoints. No base/Color grant.
bool produceOpenGlDirectGratingContribution(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    const std::vector<glasscaustics::NativeDirectGratingEndpointInput>& inputs,
    double evaluationTime,glasscaustics::NativeDirectGratingContributionReadback& output,std::string& error);

// All frozen directional D8 owners at the same ordered native endpoints.
// Private irradiance sums only; no camera/base/radiance/Color grant.
bool produceOpenGlDirectGratingIrradianceCohort(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,double evaluationTime,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sourceReadbacks,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error);
// Actual admission/registration followed by allocation refusal before shared
// preparation or dispatch. Invalid or over-cap requests remain unadmitted.
bool refuseOpenGlDirectGratingCohortPreparationForFixture(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,double evaluationTime,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>& sourceReadbacks,
    glasscaustics::DirectGratingIrradianceCohortReadback& output,std::string& error);
bool readOpenGlDirectGratingCohortStorageForFixture(const NativeFixtureSceneFrame& frame,std::size_t source,
    std::array<unsigned,4>& buffers,unsigned& program,std::uintptr_t& fence,std::string& error,
    std::array<std::uint64_t,2>* reservation=nullptr);
// Registered same-context terminal order ledger. Keeps irradiance per area;
// this private result supplies no finite footprint, radiance or public Color.
bool produceOpenGlDirectGratingTerminalOrders(const std::shared_ptr<const NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>& cohort,
    const glasscaustics::SubmissionIdentity&,double,
    const std::vector<glasscaustics::NativeDirectGratingContributionReadback>&,
    glasscaustics::DirectGratingTerminalOrderLedger&,std::string&);
bool readOpenGlDirectGratingTerminalStateForFixture(const NativeFixtureSceneFrame&,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>&,
    // Direct bytes/work, terminal attempted work/deposition, retained owner count.
    std::array<std::uint64_t,5>& counters,std::string&);
// Native outgoing order hits and independent private pixel-center raster rows.
enum class OpenGlTerminalReceiverStopForFixture { None,Reserved,Fence };
bool produceOpenGlDirectGratingTerminalReceivers(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalResult>&,
    const glasscaustics::SubmissionIdentity&,double,glasscaustics::DirectGratingTerminalReceiverReadback&,
    std::string&,OpenGlTerminalReceiverStopForFixture=OpenGlTerminalReceiverStopForFixture::None);
bool readOpenGlDirectGratingTerminalReceiverStateForFixture(const NativeFixtureSceneFrame&,
    std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>&,
    std::array<unsigned,2>&,std::uintptr_t&,std::array<std::uint64_t,3>&,std::string&);
bool inspectOpenGlDirectGratingTerminalReceiverForFixture(const NativeFixtureSceneFrame&,
    const glasscaustics::DirectGratingTerminalReceiverReadback&,std::string&);
// Counter fault seam does not expose or alter the frozen attempt registry.
// Direct bytes/work, terminal work/deposition, attempt-owner bytes, native
// work/deposition, camera trace work, camera/cell/coverage work, camera bytes,
// query-owner bytes, independently registered attempt count.
bool readOpenGlNativeAccountingForFixture(const NativeFixtureSceneFrame&,std::array<std::uint64_t,14>&,std::string&,
    glasscaustics::DirectGratingTerminalNativeBudget* reconstructedBudget=nullptr);
bool exchangeOpenGlNativeAccountingForFixture(const NativeFixtureSceneFrame&,std::array<std::uint64_t,14>&,std::string&);
enum class OpenGlCameraCoverageStopForFixture { None,Reserved,Fence };
bool produceOpenGlCameraCellCoverageForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::NativeDirectGratingTerminalReceiverResult>&,
    const glasscaustics::SubmissionIdentity&,double,glasscaustics::CameraCellCoverageReadback&,std::string&,
    OpenGlCameraCoverageStopForFixture=OpenGlCameraCoverageStopForFixture::None);
bool inspectOpenGlCameraCellCoverageForFixture(const NativeFixtureSceneFrame&,
    const glasscaustics::CameraCellCoverageReadback&,std::string&);
bool readOpenGlCameraCellCoverageStorageForFixture(const NativeFixtureSceneFrame&,
    std::array<unsigned,2>&,std::uintptr_t&,std::array<std::uint64_t,8>&,std::string&);
enum class OpenGlGratingAttemptStopForFixture { None,PointFence,CellFence };
// Real dispatch and completed fence precede the requested refusal seam.
bool produceOpenGlGlassGratingAttemptForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>&,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>&,
    glasscaustics::NativeReadback&,glasscaustics::EnergyLedger&,glasscaustics::NativeGratingCellTransportReadback&,
    OpenGlGratingAttemptStopForFixture,std::string&);

// Completed private child/deposition receipts. The caller must keep combined
// Color and ordinary dispatch guarded until the paired consumers are admitted.
bool produceOpenGlGlassGratingTerminalReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error,
    OpenGlGratingIncidentDiagnostics* diagnostics=nullptr);

struct OpenGlGratingCameraDiagnostics
{
    std::uint64_t queryWork=0,nativeBytes=0;
    bool geometryDrawComplete=false,receiverSpectrumResolveComplete=false;
    bool gratingSpectrumComplete=false;
};
struct OpenGlGratingCameraSpectrum
{
    std::shared_ptr<const glasscaustics::NativeGratingCameraReadback> camera;
    std::uint32_t width=0,height=0;
    std::vector<glasscaustics::DepositionKey> keys;
    std::vector<float> spectralRadiance; // Native diagnostic only, top-first.
    std::vector<glasscaustics::Texel> resolvedColor; // Native RGBA16F echo, top-first.
    bool complete=false;
};
bool readOpenGlGratingCameraSpectrumForFixture(const NativeFixtureSceneFrame& frame,
    std::shared_ptr<const OpenGlGratingCameraSpectrum>& output,std::string& error);
// Optional native center/corner boundary samples, never finite coverage.
bool produceOpenGlGlassGratingCellTransportReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    glasscaustics::NativeReadback& pointOutput,glasscaustics::EnergyLedger& energy,
    glasscaustics::NativeGratingCellTransportReadback& output,std::string& error);
// Private angular/visibility primitives and separate eight-band diagnostics.
// Finite receiving-cell coverage and complete base Color are still required.
bool produceOpenGlGlassGratingCameraReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>& binding,
    glasscaustics::NativeReadback& output,glasscaustics::EnergyLedger& energy,std::string& error,
    OpenGlGratingCameraDiagnostics* diagnostics=nullptr);

// Private caustic preparation for a selected closed volume beside one thin
// camera dielectric. The volume lands in smoothglass::Prepared::causticVolume.
// A camera-only selection prepares the same slot for the paired camera walk.
NativeFixtureScenePreparation prepareOpenGlGlassCausticVolumeResources(
    NativeFixtureSceneBackend& backend,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& material,
    const smoothglass::CausticVolumeSelection& selection,NativeFixturePrepareUse use);
// Private real mixed owners. Staging is fenced geometry/query ownership only.
NativeFixtureScenePreparation prepareOpenGlGlassGratingResources(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& mixedRoot,
    const smoothglass::DistinctGratingAdmission& selection,
    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& compiledLeaf);
bool stageOpenGlGlassGratingFrame(
    const std::shared_ptr<const NativeFixtureSceneResources>& resources,HarmonicMIDI::grid::SceneLightId source,
    const glasscaustics::AdmittedBranchPolicy& policy,std::uint32_t width,std::uint32_t height,
    NativeFixtureSceneRuntimeInputs runtime,double evaluationTime,
    std::shared_ptr<const NativeFixtureSceneFrame>& frameOut,
    std::shared_ptr<const glasscaustics::GratingIncidentBinding>& bindingOut,std::string& error,
    glasscaustics::SchedulingDescription scheduling={});

struct OpenGlGratingCellCoverageReadback
{
    std::shared_ptr<const glasscaustics::GratingCellCoverageCandidate> candidate;
    std::shared_ptr<const glasscaustics::GratingCellCoveragePayload> payload;
};
// Optional whole-leaf native evidence. The caller keeps the strong candidate.
bool produceOpenGlGlassGratingCellCoverageReadback(
    const std::shared_ptr<const NativeFixtureSceneFrame>& nativeFrame,
    const std::shared_ptr<const glasscaustics::GratingEntryCellBinding>& sourceCells,
    glasscaustics::NativeReadback& pointOutput,glasscaustics::EnergyLedger& energy,
    OpenGlGratingCellCoverageReadback& output,std::string& error,std::size_t maximumLeaves=4);

namespace glasscaustics::opengl
{
static_assert(kTableVersion==2&&kScheduleVersion==1&&kTriangleStride==8
    &&kPathReceiptStride==16&&kDepositReceiptStride==32);
inline constexpr ConsumerContract kConsumer {NativeFixtureMaterialBackend::OpenGl,
    kTableVersion,3,kBranchPolicyVersion,true,true,true,kScheduleVersion,true,true,true,kDepositionVersion,kRoughTransportVersion,kPrivateMultipleScatterVersion};
// All packed integer words are checked before casts. A malformed native buffer
// must fail atomically rather than enter the shared validator with rounded IDs.
inline bool causticWord(float value,std::uint32_t maximum,std::uint32_t& result)
{
    if(!std::isfinite(value)||value<0||value>float(maximum)||std::floor(value)!=value)return false;
    result=std::uint32_t(value);return true;
}
inline bool causticId(float low,float high,std::uint32_t& result)
{
    std::uint32_t a=0,b=0;if(!causticWord(low,65535,a)||!causticWord(high,65535,b))return false;
    result=a|(b<<16);return true;
}
inline bool causticHit(const glasscaustics::Texel* words,opaquevisibility::ClosestHit& hit,
    bool retainIndices=true)
{
    const auto& h=words[0];
    if(h[0]!=0&&h[0]!=1)return false;
    if(h[2]!=0&&h[2]!=1)return false;
    hit.status=int(h[0]);hit.distance=h[1];hit.entering=h[2]!=0;
    hit.position={words[1][0],words[1][1],words[1][2]};
    hit.outwardNormal={words[2][0],words[2][1],words[2][2]};
    return words[2][3]==0&&causticId(h[3],words[1][3],hit.work)
        &&causticId(words[3][0],words[3][1],hit.objectId)&&causticId(words[3][2],words[3][3],hit.materialId)
        &&(!retainIndices||(causticId(words[4][0],words[4][1],hit.instanceId)
            &&causticId(words[4][2],words[4][3],hit.triangleId)));
}
// Texel 4.z of an incomplete path receipt: the first condition that stopped the
// photon walk in computeSource. Complete receipts keep the word zero. None is
// left by an invocation that returned before its walk, and VariantCheck by a
// rough, multiple-scatter or grating check that sets no code of its own.
enum class CausticWalkRefusal : std::uint32_t
{
    None,SourceQuery,EntryMismatch,BranchQuery,InsideEscaped,UnknownTriangle,InsideWater,
    ReceiverFacing,ReceiverFootprint,CameraQuery,RasterMismatch,Radiance,MediumMismatch,
    UnresolvedTail,VariantCheck,Count
};
inline const char* causticWalkRefusalText(float word)
{
    static constexpr const char* text[]{
        "the shader walk did not complete and recorded no reason",
        "the shader walk did not complete: the source visibility query was refused",
        "the shader walk did not complete: the entry query did not return the scheduled glass triangle",
        "the shader walk did not complete: a branch query was refused",
        "the shader walk did not complete: a branch inside the glass met the environment",
        "the shader walk did not complete: a branch hit a triangle missing from the frame table",
        "the shader walk did not complete: a branch inside the glass met the water",
        "the shader walk did not complete: a receiver was hit from inside, from behind, with no reflectance or before any glass interface",
        "the shader walk did not complete: a receiver pixel footprint met the horizon or overflowed",
        "the shader walk did not complete: a camera endpoint or pixel query was refused or missed",
        "the shader walk did not complete: the raster owner or depth at a visible receiver pixel does not match its pixel-centre hit",
        "the shader walk did not complete: a receiver pixel radiance is outside the half-float range",
        "the shader walk did not complete: a branch medium disagrees with the facing of its glass hit",
        "the shader walk did not complete: the unresolved tail is above the policy fraction",
        "the shader walk did not complete: a rough, multiple-scatter or grating transport check failed"};
    static_assert(std::size(text)==std::size_t(CausticWalkRefusal::Count));
    std::uint32_t code=0;
    if(!causticWord(word,std::uint32_t(CausticWalkRefusal::Count)-1,code))
        return "the shader walk did not complete and its reason word is malformed";
    return text[code];
}
// reason, when given, names the first failed condition for the refusal text.
inline bool causticDecodePath(const glasscaustics::Texel* p,const glasscaustics::Frame& frame,
    std::size_t work,glasscaustics::NativePathReceipt& output,const char** reason=nullptr)
{
    const auto refuse=[&](const char* text){if(reason)*reason=text;return false;};
    glasscaustics::NativePathReceipt r;
    for(unsigned t=0;t<glasscaustics::kPathReceiptStride;++t)
        for(float value:p[t])if(!std::isfinite(value))return refuse("a receipt word is not finite");
    // Inspect the explicit marker before interpreting overlapping texel14.
    std::uint32_t version=0;
    if(!causticWord(p[15][3],kPrivateMultipleScatterVersion,version)
        ||version!=(frame.multipleScatterPolicy?kPrivateMultipleScatterVersion:0u))return refuse("multiple-scatter version mismatch");
    // The shader writes 0 here when its walk did not complete, and texel 4.z
    // names the first failed condition.
    if(p[1][0]!=1)return refuse(causticWalkRefusalText(p[4][2]));
    std::uint32_t low=0,high=0;
    if(!causticId(p[0][0],p[0][1],r.work))return refuse("work id word");
    if(!causticWord(p[0][2],glasscaustics::kMaximumBranchVisits,r.depositCount))return refuse("deposit count word");
    if(!causticWord(p[0][3],glasscaustics::kMaximumBranchVisits,r.visits))return refuse("visit count word");
    if(p[1][1]!=0&&p[1][1]!=1)return refuse("source-blocked flag word");
    if(!causticId(p[2][0],p[2][1],low)||!causticId(p[2][2],p[2][3],high))return refuse("query work word");
    if(r.work!=work||work>=frame.work.size()||frame.work[work].entry>=frame.entries.size())return refuse("work id does not match its scheduled slot");
    if(p[1][2]!=0||p[1][3]!=0||p[4][1]<0||p[4][2]!=0||p[4][3]!=0)return refuse("reserved or thin-absorbed word");
    r.complete=true;r.sourceBlocked=p[1][1]==1;r.queryWork=low|(std::uint64_t(high)<<32);
    r.launchedPower=p[3][0];r.depositedPower=p[3][1];r.absorbedPower=p[3][2];r.escapedPower=p[3][3];r.unresolvedPower=p[4][0];
    r.thinAbsorbedPower=p[4][1];
    if(!causticHit(p+5,r.sourceHit))return refuse("source hit decode");
    // Blocked work never queries the entry. Its unused sentinel is retained.
    if(!r.sourceBlocked&&!causticHit(p+10,r.entryHit,version==0))return refuse("entry hit decode");
    if(!decodePathExtensions(p[14],p[15],PathExtensionLayout::OpenGl,r))return refuse("path extension decode");
    output=r;return true;
}
inline bool causticDecodeDeposit(const glasscaustics::Texel* p,std::uint32_t owningMultipleScatterVersion,
    glasscaustics::NativeDeposit& output)
{
    glasscaustics::NativeDeposit r;std::uint32_t low=0,high=0,wideLow=0,wideHigh=0;
    for(unsigned t=0;t<glasscaustics::kDepositReceiptStride;++t)
        for(float value:p[t])if(!std::isfinite(value))return false;
    if(p[4][2]!=0||p[4][3]!=0||p[31]!=glasscaustics::Texel{})return false;
    std::array<glasscaustics::Texel,8> ancestry;
    std::copy_n(p+22,ancestry.size(),ancestry.begin());
    if(!glasscaustics::decodeMacroAncestry(ancestry,owningMultipleScatterVersion,r.ancestry))return false;
    if(!causticId(p[0][0],p[0][1],r.work)||!causticWord(p[0][2],glasscaustics::kMaximumInterfaces,r.interfaces)
        ||(p[0][3]!=0&&p[0][3]!=1)||!causticId(p[1][0],p[1][1],low)||!causticId(p[1][2],p[1][3],high)
        ||!causticId(p[30][0],p[30][1],wideLow)||!causticId(p[30][2],p[30][3],wideHigh)||(p[3][0]!=0&&p[3][0]!=1)||!causticWord(p[3][1],kMaximumNativeFixtureExtent,r.footprint.x)
        ||!causticWord(p[3][2],kMaximumNativeFixtureExtent,r.footprint.y))return false;
    r.reflectionHistory=glasscaustics::CausticHistory::fromWords(low,high,wideLow,wideHigh);r.cameraVisible=p[0][3]==1;
    if(!r.reflectionHistory.clearFrom(r.interfaces))return false;
    r.fraction=p[2][0];r.incidentPower=p[2][1];r.outgoingPower=p[2][2];r.absorbedPower=p[2][3];
    r.footprint.projected=p[3][0]==1;r.footprint.linearDepth=p[3][3];r.footprint.worldArea=p[4][0];r.raster.linearDepth=p[4][1];
    if(!causticId(p[5][0],p[5][1],r.raster.object)||!causticId(p[5][2],p[5][3],r.raster.material)
        ||!causticId(p[6][0],p[6][1],r.raster.instance)||!causticId(p[6][2],p[6][3],r.raster.triangle)
        ||!causticHit(p+7,r.receiver)||!causticHit(p+12,r.cameraHit)||!causticHit(p+17,r.pixelHit))return false;
    output=r;return true;
}
inline std::string querySource()
{
    auto source=opaquevisibility::glslShader(true);
    // Instrument the released query's existing work units, without changing
    // its intersection, traversal, tie breaking or unsupported status rules.
    smoothglass::replace(source,"++work;","++work; ++causticLastWork;");
    return "uint causticLastWork=0u;\n"+source;
}
// Native camera trace helpers. The typed record writer is assembled separately
// from the pinned shared trace contract; these functions publish no receipt.
inline std::string cameraGlassGratingMathSource()
{
    return R"GLSL(
bool cameraTraceFinite3(vec3 value) {return all(not(isnan(value)))&&all(not(isinf(value)));}
float cameraTraceBias(vec3 p) {return max(1e-5,2e-6*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
bool cameraTraceInterior(SceneClosestHit hit,vec3 a,vec3 b,vec3 c,vec3 normal,out vec3 barycentric) {
    precise vec3 e1=b-a,e2=c-a,delta=hit.position-a;
    precise float q00=dot(e1,e1),q01=dot(e1,e2),q11=dot(e2,e2),det=q00*q11-q01*q01;
    barycentric=vec3(0);
    if(!causticFinite(det)||det<=0.0)return false;
    precise float b1=(q11*dot(delta,e1)-q01*dot(delta,e2))/det;
    precise float b2=(q00*dot(delta,e2)-q01*dot(delta,e1))/det;
    barycentric=vec3(1.0-b1-b2,b1,b2);
    return cameraTraceFinite3(barycentric)&&all(greaterThan(barycentric,vec3(1e-5)))
        &&all(lessThan(barycentric,vec3(1.0-1e-5)))
        &&length(a+e1*b1+e2*b2-hit.position)<=8.0*cameraTraceBias(hit.position)
        &&length(hit.outwardNormal-normal)<=2e-4;
}
bool cameraTraceInterface(vec3 direction,bool inside,SceneClosestHit hit,float ior,
    out vec3 normal,out vec3 transmitted,out vec3 reflected,out float fresnel,out bool tir) {
    normal=inside?-hit.outwardNormal:hit.outwardNormal;
    transmitted=reflected=vec3(0);fresnel=0.0;tir=false;
    if(inside==hit.entering||!causticFinite(ior)||ior<1.0||dot(direction,normal)>=0.0)return false;
    fresnel=causticFresnel(direction,normal,inside?ior:1.0,inside?1.0:ior,transmitted,reflected,tir);
    return causticFinite(fresnel)&&fresnel>=0.0&&fresnel<=1.0&&cameraTraceFinite3(reflected)
        &&abs(dot(reflected,reflected)-1.0)<=1e-4
        &&cameraTraceFinite3(transmitted)&&(tir||abs(dot(transmitted,transmitted)-1.0)<=1e-4);
}
struct CameraTraceNativeBranch {
    vec3 origin,direction;float fraction;uint parent,token,interfaces;uvec2 history;bool inside;uvec4 ancestry[2];
};
struct CameraTraceNativeVisit {
    CameraTraceNativeBranch path;SceneClosestHit hit;uint work,kind;
    vec3 normal,transmitted,reflected,barycentric;float fresnel,transmittedFraction,reflectedFraction,ior;bool tir;
};
// The contract-specific writer consumes this native local after its query and
// interface calculation. No record can supply these values to the traversal.
void cameraTraceWriteVisit(int invocation,int ordinal,CameraTraceNativeVisit visit);
uint cameraTraceId(vec2 words) {return uint(words.x)|(uint(words.y)<<16u);}
bool cameraTraceTuple(SceneClosestHit hit,vec4 owners,vec4 indices) {
    return hit.objectId==cameraTraceId(owners.xy)&&hit.materialId==cameraTraceId(owners.zw)
        &&hit.instanceId==uint(indices.x)&&hit.triangleId==uint(indices.y);
}
bool cameraTraceOutside(vec3 origin,uint glassObject,uint glassMaterial) {
    vec3 lower=vec3(3.402823e38),upper=vec3(-3.402823e38);bool found=false;
    for(int triangle=0;triangle<uSceneTriangleCount;++triangle) {
        int at=triangle*8;vec4 owners=sceneTriangles[at+4];
        if(cameraTraceId(owners.xy)!=glassObject||cameraTraceId(owners.zw)!=glassMaterial)continue;
        for(int vertex=0;vertex<3;++vertex) {
            vec3 point=sceneTriangles[at+vertex].xyz;
            if(!cameraTraceFinite3(point))return false;
            lower=min(lower,point);upper=max(upper,point);found=true;
        }
    }
    double margin=4.0*double(cameraTraceBias(origin));
    // An outward closed shell need not be convex. Outside its complete native
    // vertex AABB is sufficient; all other initial media are unsupported.
    if(found)for(int axis=0;axis<3;++axis)
        if(double(lower[axis])-double(origin[axis])>margin||double(origin[axis])-double(upper[axis])>margin)return true;
    return false;
}
// The pending stack lives in CameraPending, uCameraPending branches of six
// lanes per invocation. A depth-first walk holds at most visits+1 branches,
// so the backend sizes it from the visit setting and it never fills.
void cameraPendingStore(int base,int at,CameraTraceNativeBranch b) {
    int lane=(base+at)*6;
    cameraPending[lane]=uvec4(floatBitsToUint(b.origin),floatBitsToUint(b.fraction));
    cameraPending[lane+1]=uvec4(floatBitsToUint(b.direction),b.parent);
    cameraPending[lane+2]=uvec4(b.token,b.interfaces,b.history);
    cameraPending[lane+3]=b.ancestry[0];cameraPending[lane+4]=b.ancestry[1];
    cameraPending[lane+5]=uvec4(b.inside?1u:0u,0u,0u,0u);
}
CameraTraceNativeBranch cameraPendingLoad(int base,int at) {
    int lane=(base+at)*6;CameraTraceNativeBranch b;
    b.origin=uintBitsToFloat(cameraPending[lane].xyz);b.fraction=uintBitsToFloat(cameraPending[lane].w);
    b.direction=uintBitsToFloat(cameraPending[lane+1].xyz);b.parent=cameraPending[lane+1].w;
    b.token=cameraPending[lane+2].x;b.interfaces=cameraPending[lane+2].y;b.history=cameraPending[lane+2].zw;
    b.ancestry[0]=cameraPending[lane+3];b.ancestry[1]=cameraPending[lane+4];b.inside=cameraPending[lane+5].x!=0u;
    return b;
}
void cameraTraceBranches(int invocation,vec3 origin,vec3 direction,float minimum,float maximum,float ior,
    uint glassObject,uint glassMaterial,out int count,out uint work,out uint reasons,out float unresolved) {
    int stackBase=invocation*uCameraPending,pending=1;
    CameraTraceNativeBranch root;root.origin=origin;root.direction=direction;root.fraction=1.0;
    root.parent=0xffffffffu;root.token=0u;root.interfaces=0u;root.history=uvec2(0u);root.inside=false;
    root.ancestry[0]=root.ancestry[1]=uvec4(0);cameraPendingStore(stackBase,0,root);
    count=0;work=0u;reasons=0u;unresolved=0.0;
    if(!cameraTraceOutside(origin,glassObject,glassMaterial)) {reasons=2u;unresolved=1.0;return;}
    while(pending>0) {
        if(count>=uCameraVisits) {reasons=16u;break;}
        CameraTraceNativeBranch branch=cameraPendingLoad(stackBase,--pending);
        causticLastWork=0u;
        SceneClosestHit hit=sceneClosestHit(branch.origin,branch.direction,minimum,maximum,uOpaqueControl);
        work+=causticLastWork;
        CameraTraceNativeVisit visit;visit.path=branch;visit.hit=hit;visit.work=causticLastWork;visit.kind=4u;
        visit.normal=visit.transmitted=visit.reflected=visit.barycentric=vec3(0);
        visit.fresnel=visit.transmittedFraction=visit.reflectedFraction=0.0;visit.ior=ior;visit.tir=false;
        bool valid=(hit.status==0||hit.status==1)&&causticLastWork>0u;
        if(!valid)reasons=1u;
        else if(hit.status==0) {
            valid=!branch.inside;visit.kind=valid?3u:4u;if(!valid)reasons=2u;
        } else {
            int sceneTriangle=-1;
            for(int triangle=0;triangle<uSceneTriangleCount;++triangle) {
                int at=triangle*8;
                if(cameraTraceTuple(hit,sceneTriangles[at+4],sceneTriangles[at+5])) {sceneTriangle=at;break;}
            }
            valid=sceneTriangle>=0;
            if(valid)valid=length(hit.outwardNormal-sceneTriangles[sceneTriangle+3].xyz)<=2e-4;
            if(valid&&hit.objectId==glassObject) {
                if(branch.interfaces>=uint(uCameraInterfaces)) {valid=false;reasons=4u;}
                else {
                    valid=hit.materialId==glassMaterial&&cameraTraceInterface(branch.direction,branch.inside,hit,ior,
                        visit.normal,visit.transmitted,visit.reflected,visit.fresnel,visit.tir);
                    if(!valid)reasons=64u;
                    else {
                        visit.kind=0u;
                        precise float reflectedFraction=branch.fraction*visit.fresnel;
                        precise float transmittedFraction=branch.fraction*(1.0-visit.fresnel);
                        visit.reflectedFraction=reflectedFraction;visit.transmittedFraction=transmittedFraction;
                        if(!causticFinite(reflectedFraction)||!causticFinite(transmittedFraction)) {valid=false;reasons=64u;}
                        else if((visit.fresnel>0.0&&reflectedFraction<=0.0)
                            ||(!visit.tir&&visit.fresnel<1.0&&transmittedFraction<=0.0)) {valid=false;reasons=32u;}
                    }
                }
            } else if(valid) {
                valid=!branch.inside;
                int target=-1;
                for(int triangle=0;triangle<uGratingCount;++triangle) {
                    int at=triangle*8;if(cameraTraceTuple(hit,witness[at+4],witness[at+5])) {target=at;break;}
                }
                visit.kind=target<0?2u:1u;
                if(valid&&target>=0)valid=hit.entering&&cameraTraceInterior(hit,witness[target].xyz,
                    witness[target+1].xyz,witness[target+2].xyz,witness[target+7].xyz,visit.barycentric);
                // A hit on another primitive of the selected grating is never
                // an ordinary terminal: its seam/ownership remains unresolved.
                if(target<0&&hit.objectId==cameraTraceId(witness[4].xy))valid=false;
            }
            if(!valid&&reasons==0u)reasons=2u;
        }
        if(!valid) {
            visit.kind=4u;visit.normal=visit.transmitted=visit.reflected=visit.barycentric=vec3(0);
            visit.fresnel=visit.transmittedFraction=visit.reflectedFraction=0.0;visit.tir=false;
        }
        cameraTraceWriteVisit(invocation,count,visit);uint parent=uint(count++);
        if(reasons!=0u) {unresolved+=branch.fraction;break;}
        if(visit.kind!=0u)continue;
        int children=(visit.reflectedFraction>0.0?1:0)+(!visit.tir&&visit.transmittedFraction>0.0?1:0);
        if(pending+children>uCameraPending) {reasons=8u;unresolved+=visit.reflectedFraction+visit.transmittedFraction;break;}
        float offset=cameraTraceBias(hit.position);
        CameraTraceNativeBranch child=branch;child.parent=parent;child.interfaces=branch.interfaces+1u;
        // The uvec2 history holds interfaces below 64; the visit writer copies
        // words 2 and up from the parent's record. Inline ancestry keeps the
        // first 32 tokens; later ones follow from the history.
        bool inlineToken=branch.interfaces<32u;
        int ancestryRow=int(min(branch.interfaces,31u)/16u),ancestryLane=int((min(branch.interfaces,31u)%16u)/4u);
        uint ancestryShift=(branch.interfaces%4u)*8u;
        uvec2 reflectedHistory=branch.history;
        if(branch.interfaces<32u)reflectedHistory.x|=1u<<branch.interfaces;
        else if(branch.interfaces<64u)reflectedHistory.y|=1u<<(branch.interfaces-32u);
        // Push reflection first so transmission is visited first. Only exact
        // zero factors are omitted; every positive optical branch is retained.
        if(visit.reflectedFraction>0.0) {
            child.origin=hit.position+visit.normal*offset;child.direction=visit.reflected;
            child.fraction=visit.reflectedFraction;child.token=1u;child.history=reflectedHistory;
            child.inside=branch.inside;child.ancestry=branch.ancestry;
            if(inlineToken)child.ancestry[ancestryRow][ancestryLane]|=1u<<ancestryShift;
            cameraPendingStore(stackBase,pending++,child);
        }
        if(!visit.tir&&visit.transmittedFraction>0.0) {
            child.origin=hit.position-visit.normal*offset;child.direction=visit.transmitted;
            child.fraction=visit.transmittedFraction;child.token=65u;child.history=branch.history;
            child.inside=!branch.inside;child.ancestry=branch.ancestry;
            if(inlineToken)child.ancestry[ancestryRow][ancestryLane]|=65u<<ancestryShift;
            cameraPendingStore(stackBase,pending++,child);
        }
    }
    for(int branch=0;branch<pending;++branch)unresolved+=uintBitsToFloat(cameraPending[(stackBase+branch)*6].w);
}
)GLSL";
}
// Fixed shared typed lanes; no CPU optics, visibility or terminal replay.
inline std::string cameraGlassGratingComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
struct CameraTraceInput {uvec4 pixel,control;vec4 originMinimum,directionMaximum;};
layout(std430,binding=0) readonly buffer Inputs {CameraTraceInput inputs[];};
layout(std430,binding=1) buffer Traces {uvec4 result[];};
layout(std430,binding=2) readonly buffer SceneTriangles {vec4 sceneTriangles[];};
layout(std430,binding=3) readonly buffer Grating {vec4 witness[];};
uniform int uSampleCount,uSceneTriangleCount,uGratingCount;
// The branch policy's interface count; a branch at it is unresolved (reason 4).
uniform int uCameraInterfaces;
// The branch policy's visits per band and lanes per visit (16 plus history
// words 2 and up, four per lane; glassCameraTraceVisitLanes).
uniform int uCameraVisits,uCameraVisitLanes;
// Pending branches per invocation, visits+1; six lanes each.
layout(std430,binding=4) buffer CameraPending {uvec4 cameraPending[];};
uniform int uCameraPending;
uniform vec4 uGlass;
uniform vec4 uCamera,uRotation,uProjection,uIor[2],uWavelength[2];
uniform vec2 uExtent;
)GLSL"+cameraGlassGratingMathSource()+R"GLSL(
void cameraTraceWriteVisit(int invocation,int ordinal,CameraTraceNativeVisit visit) {
    int band=invocation*(10+uCameraVisits*uCameraVisitLanes)+10,base=band+ordinal*uCameraVisitLanes;
    CameraTraceNativeBranch path=visit.path;SceneClosestHit hit=visit.hit;
    // History words 2 and up: the parent's words plus this branch's reflection.
    for(int lane=16;lane<uCameraVisitLanes;++lane) {
        uvec4 words=path.parent==0xffffffffu?uvec4(0u):result[band+int(path.parent)*uCameraVisitLanes+lane];
        uint reflected=path.interfaces-1u,extra=reflected/32u-2u;
        if(path.parent!=0xffffffffu&&path.token==1u&&reflected>=64u&&int(extra/4u)==lane-16)
            words[extra%4u]|=1u<<(reflected%32u);
        result[base+lane]=words;
    }
    result[base]=uvec4(uint(ordinal),path.parent,path.token,path.interfaces);
    result[base+1]=uvec4(uint(path.inside),path.history.x,visit.kind,uint(visit.tir));
    result[base+2]=floatBitsToUint(vec4(path.origin,1e-5));
    result[base+3]=floatBitsToUint(vec4(path.direction,path.fraction));
    result[base+4]=uvec4(uint(hit.status),hit.objectId,hit.materialId,hit.instanceId);
    result[base+5]=uvec4(hit.triangleId,visit.work,uint(hit.entering),0);
    result[base+6]=floatBitsToUint(vec4(hit.position,hit.distance));
    result[base+7]=floatBitsToUint(vec4(hit.outwardNormal,0));
    result[base+8]=floatBitsToUint(vec4(visit.normal,visit.fresnel));
    result[base+9]=floatBitsToUint(vec4(visit.transmitted,visit.transmittedFraction));
    result[base+10]=floatBitsToUint(vec4(visit.reflected,visit.reflectedFraction));
    result[base+11]=floatBitsToUint(vec4(visit.barycentric,visit.ior));
    result[base+12]=path.ancestry[0];result[base+13]=path.ancestry[1];
    result[base+14]=visit.kind==0u?floatBitsToUint(vec4(path.inside?visit.ior:1.0,path.inside?1.0:visit.ior,0,0)):uvec4(0);
    result[base+15]=uvec4(path.history.y,0u,0u,0u);
}
void main() {
    uint invocation=gl_GlobalInvocationID.x;if(invocation>=uint(uSampleCount)*8u)return;
    int sampleIndex=int(invocation/8u),band=int(invocation%8u),base=int(invocation)*(10+uCameraVisits*uCameraVisitLanes);
    for(int lane=0;lane<10+uCameraVisits*uCameraVisitLanes;++lane)result[base+lane]=uvec4(0);
    CameraTraceInput cameraInput=inputs[sampleIndex];
    float dx=cameraInput.pixel.z==0u?0.5:(cameraInput.pixel.z==2u||cameraInput.pixel.z==3u?1.0:0.0);
    float dy=cameraInput.pixel.z==0u?0.5:(cameraInput.pixel.z==3u||cameraInput.pixel.z==4u?1.0:0.0);
    vec3 ray=causticCameraRay(vec2(cameraInput.pixel.xy)+vec2(dx,dy),uRotation,uProjection,uExtent);
    vec3 cameraOrigin=causticCameraOrigin(vec2(cameraInput.pixel.xy)+vec2(dx,dy),uCamera.xyz,uRotation,uProjection,uExtent);
    float ior=uIor[band/4][band%4];
    result[base]=cameraInput.pixel;result[base+1]=cameraInput.control;
    result[base+2]=floatBitsToUint(cameraInput.originMinimum);result[base+3]=floatBitsToUint(cameraInput.directionMaximum);
    result[base+4]=floatBitsToUint(vec4(cameraOrigin,1e-5));result[base+5]=floatBitsToUint(vec4(ray,uProjection.w));
    int count;uint work,reasons;float unresolved;
    cameraTraceBranches(int(invocation),cameraOrigin,ray,1e-5,uProjection.w,ior,cameraTraceId(uGlass.xy),cameraTraceId(uGlass.zw),count,work,reasons,unresolved);
    result[base+6]=uvec4(1u,uint(sampleIndex),uint(band),uint(count));
    result[base+7]=uvec4(reasons==0u?1u:0u,reasons,work,0u);
    result[base+8]=uvec4(floatBitsToUint(ior),floatBitsToUint(uWavelength[band/4][band%4]),floatBitsToUint(unresolved),1u);
}
)GLSL";
}
inline std::string directGratingComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+R"GLSL(
layout(local_size_x=32) in;
struct EndpointInput {uvec4 primitive,control;vec4 originMinimum,directionMaximum;};
layout(std430,binding=0) readonly buffer Endpoints {EndpointInput endpoints[];};
layout(std430,binding=1) writeonly buffer Contributions {uvec4 result[];};
layout(std430,binding=2) readonly buffer Source {uvec4 sourceIds;vec4 sourceD[2];};
layout(std430,binding=3) readonly buffer Grating {vec4 witness[];};
uniform int uEndpointCount,uGratingCount;
uniform vec4 uSource,uLi[2];
bool directFinite(float value) {return !isnan(value)&&!isinf(value);}
bool directFinite3(vec3 value) {return all(not(isnan(value)))&&all(not(isinf(value)));}
uint directId(vec2 words) {return uint(words.x)|(uint(words.y)<<16u);}
uvec4 directPrimitive(SceneClosestHit hit) {return uvec4(hit.objectId,hit.materialId,hit.instanceId,hit.triangleId);}
void directHit(int base,SceneClosestHit hit,uint work) {
    result[base]=uvec4(uint(hit.status),floatBitsToUint(hit.distance),uint(hit.entering),work);
    result[base+1]=floatBitsToUint(vec4(hit.position,0));
    result[base+2]=floatBitsToUint(vec4(hit.outwardNormal,0));
    result[base+3]=directPrimitive(hit);result[base+4]=uvec4(0);
}
void main() {
    uint index=gl_GlobalInvocationID.x;if(index>=uint(uEndpointCount))return;
    int base=int(index)*27;
    for(int lane=0;lane<27;++lane)result[base+lane]=uvec4(0);
    EndpointInput endpointInput=endpoints[index];
    result[base]=endpointInput.primitive;result[base+1]=endpointInput.control;
    result[base+2]=floatBitsToUint(endpointInput.originMinimum);result[base+3]=floatBitsToUint(endpointInput.directionMaximum);
    result[base+4]=sourceIds;result[base+5]=floatBitsToUint(sourceD[0]);result[base+6]=floatBitsToUint(sourceD[1]);
    result[base+7]=floatBitsToUint(uLi[0]);result[base+8]=floatBitsToUint(uLi[1]);result[base+9]=floatBitsToUint(uSource);
    result[base+25]=uvec4(1,2,0,0);result[base+26]=uvec4(0,uint(uEndpointCount),0,0);
    if(endpointInput.control!=uvec4(1,0,0,0)||sourceIds.xyz!=uvec3(1,1,1))return;
    causticLastWork=0u;
    SceneClosestHit endpoint=sceneClosestHit(endpointInput.originMinimum.xyz,endpointInput.directionMaximum.xyz,
        endpointInput.originMinimum.w,endpointInput.directionMaximum.w,uOpaqueControl);
    uint endpointWork=causticLastWork;directHit(base+10,endpoint,endpointWork);
    if(endpoint.status!=1||directPrimitive(endpoint)!=endpointInput.primitive||endpointWork==0u)return;
    int target=-1;
    for(int triangle=0;triangle<uGratingCount;++triangle) {
        int at=triangle*8;vec4 owners=witness[at+4],indices=witness[at+5];
        if(uvec4(directId(owners.xy),directId(owners.zw),uint(indices.x),uint(indices.y))==endpointInput.primitive) {
            target=at;break;
        }
    }
    if(target<0)return;
    vec3 a=witness[target].xyz,e1=witness[target+1].xyz-a,e2=witness[target+2].xyz-a;
    vec3 delta=endpoint.position-a,normal=witness[target+7].xyz;
    precise float q00=dot(e1,e1),q01=dot(e1,e2),q11=dot(e2,e2),det=q00*q11-q01*q01;
    if(!directFinite(det)||det<=0.0)return;
    precise float b1=(q11*dot(delta,e1)-q01*dot(delta,e2))/det;
    precise float b2=(q00*dot(delta,e2)-q01*dot(delta,e1))/det;
    precise vec3 barycentric=vec3(1.0-b1-b2,b1,b2);
    result[base+20]=floatBitsToUint(vec4(barycentric,0));
    float tolerance=8.0*max(1e-5,2e-6*max(max(abs(endpoint.position.x),abs(endpoint.position.y)),abs(endpoint.position.z)));
    if(!directFinite3(barycentric)||any(lessThanEqual(barycentric,vec3(1e-5)))
        ||any(greaterThanEqual(barycentric,vec3(1.0-1e-5)))
        ||length(a+e1*b1+e2*b2-endpoint.position)>tolerance
        ||length(endpoint.outwardNormal-normal)>2e-4)return;
    vec4 sourceOrigin=vec4(endpoint.position,1e-5);
    result[base+21]=floatBitsToUint(sourceOrigin);result[base+22]=floatBitsToUint(uSource);
    causticLastWork=0u;
    SceneClosestHit source=sceneClosestHit(endpoint.position,uSource.xyz,sourceOrigin.w,uSource.w,uOpaqueControl);
    uint sourceWork=causticLastWork;directHit(base+15,source,sourceWork);
    result[base+25].w=endpointWork+sourceWork;
    if((source.status!=0&&source.status!=1)||sourceWork==0u)return;
    precise float cosine=max(0.0,dot(normal,uSource.xyz));
    bool positive=false;
    for(int bandGroup=0;bandGroup<2;++bandGroup) {
        precise vec4 bands=source.status==0?sourceD[bandGroup]*cosine:vec4(0);
        if(any(isnan(bands))||any(isinf(bands))||any(lessThan(bands,vec4(0))))return;
        result[base+23+bandGroup]=floatBitsToUint(bands);positive=positive||any(greaterThan(bands,vec4(0)));
    }
    result[base+25]=uvec4(1,positive?0u:1u,1,endpointWork+sourceWork);
    result[base+26].x=8u;
}
)GLSL";
}
using DirectGratingPacked=std::array<std::array<std::uint32_t,4>,27>;
static_assert(sizeof(DirectGratingPacked)==kDirectGratingContributionNativeBytes);
inline bool decodeDirectGratingContribution(const DirectGratingPacked& packed,
    std::size_t endpoints,NativeDirectGratingContributionSample& output)
{
    if(packed[25][0]!=kDirectGratingContributionVersion||packed[25][1]>2||packed[25][2]>1
        ||packed[26]!=std::array<std::uint32_t,4>{8,std::uint32_t(endpoints),0,0})return false;
    NativeDirectGratingContributionSample sample;
    std::memcpy(static_cast<void*>(&sample.input),packed.data(),sizeof(sample.input));
    std::memcpy(static_cast<void*>(&sample.source),packed.data()+4,sizeof(sample.source));
    const auto hit=[&](std::size_t at,opaquevisibility::ClosestHit& value) {
        const auto status=packed[at][0];if(status!=0&&status!=1&&status!=UINT32_MAX)return false;
        value.status=status==UINT32_MAX?-1:int(status);std::memcpy(&value.distance,&packed[at][1],sizeof(float));
        if(packed[at][2]>1||packed[at+1][3]||packed[at+2][3]||packed[at+4]!=std::array<std::uint32_t,4>{})return false;
        value.entering=packed[at][2]!=0;value.work=packed[at][3];
        std::memcpy(value.position.data(),packed[at+1].data(),3*sizeof(float));
        std::memcpy(value.outwardNormal.data(),packed[at+2].data(),3*sizeof(float));
        value.objectId=packed[at+3][0];value.materialId=packed[at+3][1];value.instanceId=packed[at+3][2];value.triangleId=packed[at+3][3];
        return true;
    };
    if(!hit(10,sample.endpoint)||!hit(15,sample.sourceHit)||packed[20][3])return false;
    std::memcpy(sample.barycentric.data(),packed[20].data(),3*sizeof(float));
    std::memcpy(sample.sourceOriginMinimum.data(),packed[21].data(),sizeof(Texel));
    std::memcpy(sample.sourceDirectionMaximum.data(),packed[22].data(),sizeof(Texel));
    std::memcpy(sample.surfaceIrradiance.data(),packed[23].data(),8*sizeof(float));
    sample.queryWork=packed[25][3];sample.complete=packed[25][2]!=0;
    sample.outcome=static_cast<NativeDirectGratingContributionSample::Outcome>(packed[25][1]);
    output=sample;return true;
}
inline std::string computeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) buffer Paths {vec4 paths[];};
layout(std430,binding=1) buffer Deposits {vec4 deposits[];};
layout(std430,binding=3) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform sampler2D uRasterOwners,uRasterTriangle;
uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending;
uniform vec4 uSource,uCamera,uRotation,uProjection,uIor[2],uLi[2];
uniform vec2 uExtent,uPolicy;
// Reflection history words 4 and up (interfaces 128 and up), uHistoryExtra words
// per branch as 16-bit halves like deposit texel 1. They sit in the Paths buffer
// after the path receipts: one block per deposit slot, then uPending+1
// branch blocks per path, the last holding the branch being walked. Zero extra
// words make every helper a no-op. ResourceBounds::historyTexels sizes the tail.
// Multiple-scatter ancestry tokens for interfaces 32 and up follow in the same
// block, uAncestryExtra words of four tokens.
uniform int uHistoryExtra,uAncestryExtra;
int historyTexels() {return (uHistoryExtra+uAncestryExtra+1)/2;}
// Tail offsets are summed in uint, so no product of the work, visit and pending
// counts wraps. The host admits a path buffer only when every texel index fits
// an int (openGlCausticLimitFailure), so the final offset converts exactly.
int pathTailIndex(uint value) {return int(value);}
int historyDepositBlock(int w,int slot) {
    return pathTailIndex(uint(uWorkCount)*uint(CAUSTIC_PATH_RECEIPT_STRIDE)+(uint(w)*uint(uVisits)+uint(slot))*uint(historyTexels()));
}
int historyBranchBlock(int w,int slot) {
    return pathTailIndex(uint(uWorkCount)*uint(CAUSTIC_PATH_RECEIPT_STRIDE)
        +(uint(uWorkCount)*uint(uVisits)+uint(w)*uint(uPending+1)+uint(slot))*uint(historyTexels()));
}
void historyCopy(int from,int to) {for(int i=0;i<historyTexels();++i)paths[to+i]=paths[from+i];}
void historyClear(int at) {for(int i=0;i<historyTexels();++i)paths[at+i]=vec4(0.0);}
void historySet(int at,int interfaces) {
    int bit=interfaces-128;if(bit<0||bit>=32*uHistoryExtra)return;
    int word=bit/32,component=(word%2)*2+(bit%32)/16;
    paths[at+word/2][component]=float(uint(paths[at+word/2][component])|(1u<<uint(bit%16)));
}
void ancestrySet(int at,int interfaces,uint token) {
    int b=interfaces-32;if(b<0||b>=4*uAncestryExtra)return;
    int word=uHistoryExtra+b/4,component=(word%2)*2+(b%4)/2;
    paths[at+word/2][component]=float(uint(paths[at+word/2][component])|(token<<uint(8*(b%2))));
}
uint queryWork=0u;
// First failed walk condition, a CausticWalkRefusal code. Written to texel 4.z
// of an incomplete path receipt only; complete receipts keep that word zero.
uint causticFailReason=0u;
uint id(vec2 words) {return uint(words.x)|(uint(words.y)<<16u);}
vec2 words(uint value) {return vec2(value&65535u,value>>16u);}
vec4 owners(SceneClosestHit hit) {return vec4(words(hit.objectId),words(hit.materialId));}
vec4 indices(SceneClosestHit hit) {return vec4(words(hit.instanceId),words(hit.triangleId));}
float rayBias(vec3 p) {return max(1e-5,2e-6*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
SceneClosestHit query(vec3 origin,vec3 direction,float minimum,float maximum) {
    causticLastWork=0u;
    SceneClosestHit hit=sceneClosestHit(origin,direction,minimum,maximum,uOpaqueControl);
    queryWork+=causticLastWork;return hit;
}
void pathHit(int offset,SceneClosestHit hit,uint work) {
    paths[offset]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    paths[offset+1]=vec4(hit.position,float(work>>16u));paths[offset+2]=vec4(hit.outwardNormal,0.0);
    paths[offset+3]=owners(hit);paths[offset+4]=indices(hit);
}
void depositHit(int offset,SceneClosestHit hit,uint work) {
    deposits[offset]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    deposits[offset+1]=vec4(hit.position,float(work>>16u));deposits[offset+2]=vec4(hit.outwardNormal,0.0);
    deposits[offset+3]=owners(hit);deposits[offset+4]=indices(hit);
}
int findTriangle(SceneClosestHit hit) {
    for(int t=0;t<uTriangleCount;++t) {
        vec4 o=triangles[t*8+4],i=triangles[t*8+5];
        if(id(o.xy)==hit.objectId&&id(o.zw)==hit.materialId&&uint(i.x)==hit.instanceId&&uint(i.y)==hit.triangleId) return t;
    }return -1;
}
bool sameTriangle(SceneClosestHit a,SceneClosestHit b) {
    return a.objectId==b.objectId&&a.materialId==b.materialId&&a.instanceId==b.instanceId&&a.triangleId==b.triangleId;
}
// A pixel centre on an edge shared by two triangles of one receiver belongs to
// both, and the raster and the closest-hit query can each name a different
// one. The raster triangle must then hold the pixel-centre hit within the
// 8-bias tolerance of the camera checks. rasterOwnsPixelHit mirrors this.
bool causticRasterOwns(vec4 raster,SceneClosestHit hit,SceneClosestHit pixelHit) {
    if(uint(raster.y)==hit.triangleId) return true;
    float tolerance=8.0*rayBias(pixelHit.position);
    for(int t=0;t<uTriangleCount;++t) {
        vec4 o=triangles[t*8+4],i=triangles[t*8+5];
        if(id(o.xy)!=hit.objectId||id(o.zw)!=hit.materialId||uint(i.x)!=hit.instanceId||uint(i.y)!=uint(raster.y)) continue;
        vec3 v[3]=vec3[3](triangles[t*8].xyz,triangles[t*8+1].xyz,triangles[t*8+2].xyz);
        vec3 n=triangles[t*8+3].xyz;
        if(abs(dot(pixelHit.position-v[0],n))>tolerance) return false;
        for(int e=0;e<3;++e) {
            vec3 edge=v[(e+1)%3]-v[e];
            if(dot(cross(edge,pixelHit.position-v[e]),n)<-tolerance*length(edge)) return false;
        }
        return true;
    }
    return false;
}
struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;};
// Pending branches live in the Paths tail after the branch history blocks, uPending
// entries of CAUSTIC_BRANCH_TEXELS per path, so the policy's maximumPending sizes
// the stack. Words are stored as exact 16-bit halves.
int pendingBlock(int w,int slot) {
    return pathTailIndex(uint(uWorkCount)*uint(CAUSTIC_PATH_RECEIPT_STRIDE)
        +uint(uWorkCount)*(uint(uVisits)+uint(uPending)+1u)*uint(historyTexels())
        +(uint(w)*uint(uPending)+uint(slot))*uint(CAUSTIC_BRANCH_TEXELS));
}
void pendingStore(int at,Branch b,uvec2 high) {
    paths[at]=vec4(b.origin,b.fraction);paths[at+1]=vec4(b.direction,b.inside?1.0:0.0);
    paths[at+2]=vec4(words(b.history.x),words(b.history.y));paths[at+3]=vec4(words(high.x),words(high.y));
    paths[at+4].xy=words(uint(b.interfaces));
}
Branch pendingLoad(int at,out uvec2 high) {
    Branch b;b.origin=paths[at].xyz;b.fraction=paths[at].w;b.direction=paths[at+1].xyz;b.inside=paths[at+1].w!=0.0;
    b.history=uvec2(id(paths[at+2].xy),id(paths[at+2].zw));high=uvec2(id(paths[at+3].xy),id(paths[at+3].zw));
    b.interfaces=int(id(paths[at+4].xy));
    return b;
}
void main() {
    uint invocation=gl_GlobalInvocationID.x;if(invocation>=uint(uWorkCount)) return;
    int w=int(invocation),base=w*CAUSTIC_PATH_RECEIPT_STRIDE;
    for(int i=0;i<CAUSTIC_PATH_RECEIPT_STRIDE;++i)paths[base+i]=vec4(0.0);
    // Only this invocation writes its fixed receipt segment.
    int depositBase=w*uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;
    for(int i=0;i<uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;++i)deposits[depositBase+i]=vec4(0.0);
    vec4 entry=workTable[w*4],normalPower=workTable[w*4+1],entryOwners=workTable[w*4+2],entryIndices=workTable[w*4+3];
    int band=int(entryIndices.w);float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);
    paths[base]=vec4(words(invocation),0.0,0.0);
    bool complete=true;int count=0,visits=0,pending=0;float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;
    SceneClosestHit source=query(entry.xyz,uSource.xyz,1e-5,uSource.w);uint sourceWork=causticLastWork;
    pathHit(base+5,source,sourceWork);
    bool blocked=source.status==1;complete=source.status>=0;
    if(!complete)causticFailReason=1u;
    SceneClosestHit entryHit=sceneEmptyHit(-1,0.0);uint entryWork=0u;
    // pendingStore keeps interfaces 64..127 of each branch's reflection history (glass_caustic_history.h).
    float thinAbsorbed=0.0;
    if(complete&&!blocked) {
        float offset=rayBias(entry.xyz);
        entryHit=query(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,0.0,uSource.w);entryWork=causticLastWork;
        complete=entryHit.status==1&&entryHit.entering&&entryHit.objectId==id(entryOwners.xy)
            &&entryHit.materialId==id(entryOwners.zw)&&entryHit.instanceId==uint(entryIndices.x)&&entryHit.triangleId==uint(entryIndices.y);
        if(!complete)causticFailReason=2u;
        if(complete)pendingStore(pendingBlock(w,pending++),Branch(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,1.0,uvec2(0u),0,false),uvec2(0u));
        historyClear(historyBranchBlock(w,0));
    }
    pathHit(base+10,entryHit,entryWork);
    while(complete&&pending>0&&visits<uVisits) {
        uvec2 branchHigh;Branch branch=pendingLoad(pendingBlock(w,--pending),branchHigh);++visits;
        // A child push reuses this slot, so the walked branch keeps its own copy.
        historyCopy(historyBranchBlock(w,pending),historyBranchBlock(w,uPending));
        SceneClosestHit hit=query(branch.origin,branch.direction,branch.interfaces==0?0.0:1e-5,uSource.w);uint hitWork=causticLastWork;
        if(hit.status<0) {causticFailReason=3u;complete=false;break;}
        if(hit.status==0) {
            if(branch.inside){causticFailReason=4u;complete=false;break;}escaped+=potential*branch.fraction;continue;
        }
        int triangle=findTriangle(hit);if(triangle<0) {causticFailReason=5u;complete=false;break;}
        vec4 shape=triangles[triangle*8+5];
        // Owner flag 2 is the thin camera dielectric beside the caustic volume.
        // The photon is absorbed: no deposit, no refusal, no unresolved charge.
        if(shape.z==2.0) {
            if(branch.inside) {causticFailReason=6u;complete=false;break;}
            thinAbsorbed+=potential*branch.fraction;continue;
        }
        if(shape.z==0.0) {
            if(branch.inside||!hit.entering||shape.w!=1.0||branch.interfaces==0) {causticFailReason=7u;complete=false;break;}
            float incident=potential*branch.fraction;
            vec2 power=causticReceiverPower(incident,triangles[triangle*8+6+band/4][band%4]);
            outgoing+=power.x;absorbed+=power.y;
            vec4 footprint;int projected=causticProjectReceiver(hit.position,hit.outwardNormal,uCamera.xyz,uRotation,uProjection,uExtent,footprint);
            if(projected<0) {causticFailReason=8u;complete=false;break;}
            SceneClosestHit cameraHit=sceneEmptyHit(0,0.0),pixelHit=sceneEmptyHit(0,0.0);
            uint cameraWork=0u,pixelWork=0u;bool visible=false;vec4 rasterOwners=vec4(0.0),raster=vec4(0.0);
            if(projected>0) {
                vec3 endpointOrigin=causticEndpointOrigin(hit.position,uCamera.xyz,uRotation,uProjection);
                vec3 delta=hit.position-endpointOrigin;float distance=length(delta);
                cameraHit=query(endpointOrigin,delta,0.0,distance+8.0*rayBias(hit.position));cameraWork=causticLastWork;
                pixelHit=query(causticCameraOrigin(footprint.xy+vec2(0.5),uCamera.xyz,uRotation,uProjection,uExtent),causticCameraRay(footprint.xy+vec2(0.5),uRotation,uProjection,uExtent),0.0,uProjection.w);pixelWork=causticLastWork;
                if(cameraHit.status!=1||pixelHit.status<0) {causticFailReason=9u;complete=false;break;}
                visible=sameTriangle(hit,cameraHit)&&abs(cameraHit.distance-distance)<=8.0*rayBias(hit.position)
                    &&length(cameraHit.position-hit.position)<=8.0*rayBias(hit.position)&&pixelHit.status==1&&sameTriangle(hit,pixelHit);
                ivec2 pixel=ivec2(footprint.x,uExtent.y-1.0-footprint.y);
                rasterOwners=texelFetch(uRasterOwners,pixel,0);raster=texelFetch(uRasterTriangle,pixel,0);
                if(visible) {
                    vec3 local=causticRotate(vec4(-uRotation.xyz,uRotation.w),pixelHit.position-uCamera.xyz);
                    if(id(rasterOwners.xy)!=hit.objectId||id(rasterOwners.zw)!=hit.materialId
                        ||uint(raster.x)!=hit.instanceId||!causticRasterOwns(raster,hit,pixelHit)
                        ||abs(raster.z+local.z)>2e-4*max(abs(local.z),1e-20)) {causticFailReason=10u;complete=false;break;}
                    float radiance=causticPixelRadiance(power.x,footprint.w);
                    if(!causticFinite(radiance)||radiance<0.0||radiance>65504.0) {causticFailReason=11u;complete=false;break;}
                }
            }
            int d=depositBase+count*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;
            deposits[d]=vec4(words(invocation),float(branch.interfaces),float(visible));
            deposits[d+1]=vec4(words(branch.history.x),words(branch.history.y));
            deposits[d+30]=vec4(words(branchHigh.x),words(branchHigh.y));
            historyCopy(historyBranchBlock(w,uPending),historyDepositBlock(w,count));
            deposits[d+2]=vec4(branch.fraction,incident,power);
            deposits[d+3]=vec4(float(projected>0),footprint.xy,footprint.z);
            deposits[d+4]=vec4(footprint.w,raster.z,0.0,0.0);
            deposits[d+5]=rasterOwners;deposits[d+6]=vec4(words(uint(raster.x)),words(uint(raster.y)));
            depositHit(d+7,hit,hitWork);depositHit(d+12,cameraHit,cameraWork);depositHit(d+17,pixelHit,pixelWork);
            ++count;continue;
        }
        if(branch.inside==hit.entering) {causticFailReason=12u;complete=false;break;}
        if(branch.interfaces>=uInterfaces) {tail+=potential*branch.fraction;continue;}
        vec3 normal=branch.inside?-hit.outwardNormal:hit.outwardNormal;
        vec3 transmitted,reflected;bool tir;float ior=uIor[band/4][band%4];
        float reflection=causticFresnel(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,transmitted,reflected,tir);
        float reflectedFraction=branch.fraction*reflection,transmittedFraction=branch.fraction-reflectedFraction;
        if(!causticFinite(reflection)||(reflection>0.0&&reflectedFraction==0.0)||(reflection<1.0&&transmittedFraction==0.0)) {
            tail+=potential*branch.fraction;continue;
        }
        for(int child=0;child<2;++child) {
            bool reflect=child==0;float fraction=reflect?reflectedFraction:transmittedFraction;
            if(fraction==0.0||(tir&&!reflect))continue;
            if(fraction<uPolicy.y||pending>=uPending) {tail+=potential*fraction;continue;}
            uvec2 history=branch.history,high=branchHigh;
            if(reflect) {
                if(branch.interfaces<32)history.x|=1u<<uint(branch.interfaces);
                else if(branch.interfaces<64)history.y|=1u<<uint(branch.interfaces-32);
                else if(branch.interfaces<96)high.x|=1u<<uint(branch.interfaces-64);
                else if(branch.interfaces<128)high.y|=1u<<uint(branch.interfaces-96);
            }
            historyCopy(historyBranchBlock(w,uPending),historyBranchBlock(w,pending));
            if(reflect)historySet(historyBranchBlock(w,pending),branch.interfaces);
            pendingStore(pendingBlock(w,pending++),Branch(hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position)),
                reflect?reflected:transmitted,fraction,history,branch.interfaces+1,reflect?branch.inside:!branch.inside),high);
        }
    }
    for(int i=0;i<pending;++i)tail+=potential*paths[pendingBlock(w,i)].w;
    float launched=blocked?0.0:potential;
    bool walked=complete;
    complete=complete&&tail<=launched*uPolicy.x*(1.0+2e-4);
    if(!complete&&causticFailReason==0u)causticFailReason=walked?13u:14u;
    paths[base].zw=vec2(float(count),float(visits));
    paths[base+1]=vec4(float(complete),float(blocked),0.0,0.0);
    paths[base+2]=vec4(words(queryWork),0.0,0.0);
    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);
    paths[base+4].y=thinAbsorbed;
    if(!complete)paths[base+4].z=float(causticFailReason);
}
)GLSL";
}
// Keep the released smooth shader bytes. Add the shared VNDF kernel and
// stratum receipts only for a resource prepared with nonzero roughness.
inline std::string roughComputeSource()
{
    auto source=computeSource();
    smoothglass::replace(source,nativeMathSource(),nativeRoughMathSource());
    smoothglass::replace(source,R"GLSL(uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending;
)GLSL",R"GLSL(uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending,uInterfaceSamples;
uniform float uRoughness;
)GLSL");
    smoothglass::replace(source,R"GLSL(    int band=int(entryIndices.w);float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);
)GLSL",R"GLSL(    int band=int(entryIndices.w);uint interfaceSample=(invocation/8u)%uint(uInterfaceSamples);
    float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);
    if(uRoughness>0.0)potential/=float(uInterfaceSamples);
)GLSL");
    smoothglass::replace(source,R"GLSL(    bool complete=true;int count=0,visits=0,pending=0;float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;
)GLSL",R"GLSL(    bool complete=true;int count=0,visits=0,pending=0;float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0,masked=0.0;
)GLSL");
    smoothglass::replace(source,R"GLSL(        float reflection=causticFresnel(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,transmitted,reflected,tir);
        float reflectedFraction=branch.fraction*reflection,transmittedFraction=branch.fraction-reflectedFraction;
        if(!causticFinite(reflection)||(reflection>0.0&&reflectedFraction==0.0)||(reflection<1.0&&transmittedFraction==0.0)) {
            tail+=potential*branch.fraction;continue;
)GLSL",R"GLSL(        float reflectedFraction,transmittedFraction;
        if(uRoughness==0.0) {
            float reflection=causticFresnel(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,transmitted,reflected,tir);
            reflectedFraction=branch.fraction*reflection;transmittedFraction=branch.fraction-reflectedFraction;
            if(!causticFinite(reflection)||(reflection>0.0&&reflectedFraction==0.0)||(reflection<1.0&&transmittedFraction==0.0)) {
                tail+=potential*branch.fraction;continue;
            }
        } else {
            vec2 sampleUv=causticRoughSample(interfaceSample,uint(branch.interfaces));
            vec3 power;bool interfaceComplete;
            causticRoughInterface(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,
                uRoughness,sampleUv.x,sampleUv.y,reflected,transmitted,power,interfaceComplete);
            if(!interfaceComplete||power.x<0.0||power.y<0.0||power.z<0.0
                ||abs(power.x+power.y+power.z-1.0)>2e-5) {complete=false;break;}
            masked+=potential*branch.fraction*power.z;
            reflectedFraction=branch.fraction*power.x;transmittedFraction=branch.fraction*power.y;tir=power.y==0.0;
            if((power.x>0.0&&reflectedFraction==0.0)||(power.y>0.0&&transmittedFraction==0.0)) {
                tail+=potential*branch.fraction*(power.x+power.y);continue;
            }
)GLSL");
    smoothglass::replace(source,R"GLSL(    complete=complete&&tail<=launched*uPolicy.x*(1.0+2e-4);
)GLSL",R"GLSL(    complete=complete&&masked+tail<=launched*uPolicy.x*(1.0+2e-4);
)GLSL");
    smoothglass::replace(source,R"GLSL(    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);
)GLSL",R"GLSL(    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);
    if(uRoughness>0.0)paths[base+15]=vec4(masked,float(CAUSTIC_ROUGH_VERSION),float(interfaceSample),0.0);
)GLSL");
    return source;
}
// V2 emits every physical Smith-height exit through the same macro traversal.
// The released smooth and single-interface generators retain their bytes.
// The pending stack lives in the path tail. Reserve the popped/pushed branch
// copies. A branch
// storage model occupies96 bytes at16-byte alignment; reserve112 per
// branch for conservative native padding. Both exit/order argument
// copies consume320 bytes each. Reserve2048 for retained hit/entry values and
// nested query/walk scalar and vector locals without relying on lifetime reuse.
struct alignas(16) MultipleScatterBranchStorageShape
{
    alignas(16) std::array<float,3> origin,direction;
    float fraction;
    alignas(8) std::array<std::uint32_t,2> history;
    std::uint32_t interfaces,inside;
    alignas(16) std::array<std::uint32_t,4> ancestryLow,ancestryHigh;
    std::uint32_t samplingHistory;
};
static_assert(sizeof(MultipleScatterBranchStorageShape)==96);
inline constexpr std::uint64_t kMultipleScatterBranchBytes=112;
static_assert(sizeof(MultipleScatterBranchStorageShape)<=kMultipleScatterBranchBytes);
inline constexpr std::uint64_t kMultipleScatterNativeTemporaryBytesPerWork=
    2*kMultipleScatterBranchBytes+2*16*(16+4)+3*16+8+2048;
static_assert(kMultipleScatterPendingExtensionBytes>=48);
static_assert(smoothglass::kMaximumMicrofacetExits==16);
inline std::string multipleScatterComputeSource()
{
    auto source=computeSource();
    smoothglass::replace(source,nativeMathSource(),nativeMultipleScatterMathSource());
    smoothglass::replace(source,"uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending;",
        "uniform int uWorkCount,uTriangleCount,uVisits,uInterfaces,uPending,uInterfaceSamples,uMicroEvents,uMicroExits;\nuniform float uRoughness,uMicroResidual;");
    smoothglass::replace(source,"struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;};",
        "struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;uvec4 ancestryLow,ancestryHigh;uint samplingHistory;};");
    smoothglass::replace(source,"    paths[at+4].xy=words(uint(b.interfaces));\n",
        "    paths[at+4]=vec4(words(uint(b.interfaces)),words(b.samplingHistory));\n"
        "    for(int i=0;i<2;++i)paths[at+5+i]=vec4(words(b.ancestryLow[2*i]),words(b.ancestryLow[2*i+1]));\n"
        "    for(int i=0;i<2;++i)paths[at+7+i]=vec4(words(b.ancestryHigh[2*i]),words(b.ancestryHigh[2*i+1]));\n");
    smoothglass::replace(source,"    b.interfaces=int(id(paths[at+4].xy));\n",
        "    b.interfaces=int(id(paths[at+4].xy));b.samplingHistory=id(paths[at+4].zw);\n"
        "    for(int i=0;i<2;++i) {vec4 t=paths[at+5+i];b.ancestryLow[2*i]=id(t.xy);b.ancestryLow[2*i+1]=id(t.zw);}\n"
        "    for(int i=0;i<2;++i) {vec4 t=paths[at+7+i];b.ancestryHigh[2*i]=id(t.xy);b.ancestryHigh[2*i+1]=id(t.zw);}\n");
    smoothglass::replace(source,"int band=int(entryIndices.w);float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w);",
        "int band=int(entryIndices.w);uint interfaceSample=(invocation/8u)%uint(uInterfaceSamples);\n"
        "    uint entryKey=causticMsEntryKey(id(entryOwners.xy),uint(entryIndices.x),uint(entryIndices.y),uint(entryIndices.z));\n"
        "    float potential=causticEntryPower(uLi[band/4][band%4],normalPower.xyz,uSource.xyz,entry.w)/float(uInterfaceSamples);");
    smoothglass::replace(source,"float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;",
        "float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0,microResidualPower=0.0;uint microEvents=0u,microDraws=0u;");
    smoothglass::replace(source,"1.0,uvec2(0u),0,false),",
        "1.0,uvec2(0u),0,false,uvec4(0u),uvec4(0u),2166136261u),");
    smoothglass::replace(source,"            ++count;continue;",R"GLSL(            for(int token=0;token<32;++token) {
                uint ancestryWord=token<16?branch.ancestryLow[token/4]:branch.ancestryHigh[(token-16)/4];
                deposits[d+22+token/4][token%4]=float((ancestryWord>>uint(8*(token%4)))&255u);
            }
            ++count;continue;)GLSL");
    const auto begin=source.find("        vec3 transmitted,reflected;bool tir;float ior=");
    const auto end=source.find("\n    }\n    for(int i=0;i<pending;",begin);
    if(begin==std::string::npos||end==std::string::npos)return {};
    source.replace(begin,end-begin,R"GLSL(        float ior=uIor[band/4][band%4];
        vec4 exits[16];uint orders[16];uint exitCount,events,draws;float residual;
        uint seed=causticMsSeed(entryKey,interfaceSample,uint(branch.interfaces),branch.samplingHistory);
        int status=causticMsWalk(branch.direction,normal,branch.inside?ior:1.0,branch.inside?1.0:ior,
            uRoughness,seed,uint(uMicroEvents),uint(uMicroExits),uMicroResidual,
            exits,orders,exitCount,events,draws,residual);
        microEvents+=events;microDraws+=draws;
        if(status!=1||exitCount>uint(uMicroExits)||events>uint(uMicroEvents)||draws!=4u*events
            ||!causticFinite(residual)||residual<0.0||residual>uMicroResidual) {complete=false;break;}
        float sum=residual;
        for(uint child=0u;child<exitCount;++child)sum+=abs(exits[child].w);
        if(!causticFinite(sum)||abs(sum-1.0)>2e-5) {complete=false;break;}
        float unresolved=potential*branch.fraction*residual;
        microResidualPower+=unresolved;tail+=unresolved;
        for(uint child=0u;child<exitCount;++child) {
            vec4 emitted=exits[child];bool transmitted=emitted.w<0.0;
            float power=abs(emitted.w),fraction=branch.fraction*power;
            if(!causticFinite3(emitted.xyz)||!causticFinite(power)||power<=0.0||fraction<=0.0
                ||abs(dot(emitted.xyz,emitted.xyz)-1.0)>1e-4||orders[child]>uint(uMicroEvents)
                ||(ior==1.0?(orders[child]!=0u||!transmitted):orders[child]==0u)
                ||(transmitted?dot(emitted.xyz,normal)>=0.0:dot(emitted.xyz,normal)<=0.0)) {complete=false;break;}
            // Cutoffs and pending overflow account for each omitted exit once.
            if(fraction<uPolicy.y||pending>=uPending) {tail+=potential*fraction;continue;}
            uvec2 history=branch.history,high=branchHigh;
            if(!transmitted) {
                if(branch.interfaces<32)history.x|=1u<<uint(branch.interfaces);
                else if(branch.interfaces<64)history.y|=1u<<uint(branch.interfaces-32);
                else if(branch.interfaces<96)high.x|=1u<<uint(branch.interfaces-64);
                else if(branch.interfaces<128)high.y|=1u<<uint(branch.interfaces-96);
            }
            uvec4 ancestryLow=branch.ancestryLow,ancestryHigh=branch.ancestryHigh;
            uint token=orders[child]|(transmitted?64u:0u),word=uint(branch.interfaces)/4u;
            if(word<4u)ancestryLow[word]|=token<<uint(8*(branch.interfaces%4));
            else if(word<8u)ancestryHigh[word-4u]|=token<<uint(8*(branch.interfaces%4));
            // Interfaces 32 and up keep their tokens, and 128 and up their history, in the tail.
            historyCopy(historyBranchBlock(w,uPending),historyBranchBlock(w,pending));
            if(!transmitted)historySet(historyBranchBlock(w,pending),branch.interfaces);
            ancestrySet(historyBranchBlock(w,pending),branch.interfaces,token);
            pendingStore(pendingBlock(w,pending++),Branch(hit.position+normal*(transmitted?-rayBias(hit.position):rayBias(hit.position)),
                emitted.xyz,fraction,history,branch.interfaces+1,transmitted?!branch.inside:branch.inside,
                ancestryLow,ancestryHigh,causticMsChildHistory(branch.samplingHistory,orders[child],transmitted)),high);
        })GLSL");
    smoothglass::replace(source,"    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);",
        "    paths[base+3]=vec4(launched,outgoing,absorbed,escaped);paths[base+4]=vec4(tail,0.0,0.0,0.0);\n"
        "    paths[base+14]=vec4(float(microEvents),float(microDraws),microResidualPower,entryIndices.z);\n"
        "    paths[base+15]=vec4(0.0,float(CAUSTIC_ROUGH_VERSION),float(interfaceSample),float(GLASS_MS_VERSION));");
    return source;
}
inline bool decodeGratingIncident(const Texel* p,NativeGratingIncident& output)
{
    NativeGratingIncident result;
    for(unsigned lane=0;lane<kGratingIncidentStride;++lane)
        for(float value:p[lane])if(!std::isfinite(value))return false;
    for(unsigned lane:{16u,18u,19u,20u,21u})if(p[lane][3]!=0)return false;
    if(p[17][2]!=0||p[17][3]!=0||p[22][1]!=1)return false;
    std::uint32_t low=0,high=0,entering=0;
    if(!causticWord(p[0][0],kMaximumEntrySamples*8-1,result.work)||!causticWord(p[0][1],7,result.band)
        ||!causticWord(p[0][2],std::uint32_t(kMaximumInterfaces),result.interfaces)
        ||!causticWord(p[0][3],kGratingOrdersPerIncident,result.orderCount)
        ||!causticId(p[1][0],p[1][1],low)||!causticId(p[1][2],p[1][3],high)
        ||!causticId(p[13][0],p[13][1],result.hit.objectId)||!causticId(p[13][2],p[13][3],result.hit.materialId)
        // Glass schedules admit one instance per authored scene object.
        ||!causticWord(p[14][0],std::uint32_t(opaquevisibility::kExactFloatIndexRange-1),result.hit.instanceId)
        ||!causticWord(p[14][1],std::uint32_t(opaquevisibility::kExactFloatIndexRange-1),result.hit.triangleId)
        ||!causticId(p[14][2],p[14][3],result.hit.work)||!causticWord(p[15][3],1,entering)
        ||!causticWord(p[22][0],kGratingIncidentVersion,result.version)||result.version!=kGratingIncidentVersion)return false;
    // Words 2 and up follow from the incident tail (decodeGratingIncidentHistoryTail).
    result.reflectionHistory=CausticHistory(low|(std::uint64_t(high)<<32));
    for(unsigned index=0;index<32;++index) {
        std::uint32_t token=0;if(!causticWord(p[2+index/4][index%4],127,token))return false;
        result.ancestry[index]=std::uint8_t(token);
    }
    const auto point=[&](unsigned lane){return Point{p[lane][0],p[lane][1],p[lane][2]};};
    result.origin=point(10);result.fraction=p[10][3];result.direction=point(11);result.power=p[11][3];
    result.hit.status=1;result.hit.position=point(12);result.hit.distance=p[12][3];
    result.hit.outwardNormal=point(15);result.hit.entering=entering!=0;result.barycentric=point(16);
    result.uv={p[17][0],p[17][1]};result.tangent=point(18);result.bitangent=point(19);result.normal=point(20);
    result.dHdu={p[22][2],p[22][3],p[23][0]};result.dHdv={p[23][1],p[23][2],p[23][3]};
    result.substrateReflectedPower=p[21][0];result.absorbedPower=p[21][1];result.unresolvedPower=p[21][2];result.complete=true;
    output=result;return true;
}
inline bool decodeGratingOrder(const Texel* p,std::uint32_t incident,NativeGratingOrder& output)
{
    for(unsigned lane=0;lane<kGratingOrderStride;++lane)
        for(float value:p[lane])if(!std::isfinite(value))return false;
    std::uint32_t locator=0,signedWord=0;
    if(!causticWord(p[0][0],kMaximumExactReceiptCount-1,locator)||locator!=incident
        ||!causticWord(p[0][1],2,signedWord)||p[0][2]!=0||p[0][3]!=0||p[2]!=Texel{}||p[3]!=Texel{})return false;
    NativeGratingOrder result;result.order=std::int32_t(signedWord)-1;
    result.direction={p[1][0],p[1][1],p[1][2]};result.power=p[1][3];output=result;return true;
}
// The caller supplies exact function bodies from the existing native material
// shader. No order efficiencies or substrate constants come from a CPU oracle.
inline std::string gratingIncidentComputeSource(const std::string& materialMath,const std::string& orderEquation)
{
    if(materialMath.empty()||orderEquation.empty())return {};
    auto source=computeSource();
    smoothglass::replace(source,"layout(std430,binding=1) buffer Deposits {vec4 deposits[];};",
        "layout(std430,binding=1) buffer Incidents {vec4 incidents[];};\nlayout(std430,binding=2) buffer Orders {vec4 orders[];};\n"
        "layout(std430,binding=5) readonly buffer Witness {vec4 witness[];};\nuniform int uWitnessCount;\n"
        // History words 2 and up per incident (gratingIncidentHistoryTailWords)
        // start at incident texel uGratingHistoryBase, as 16-bit halves.
        "uniform int uGratingHistoryBase,uGratingHistoryWords;\n"
        "uniform vec4 uDiffractionGeometry,uDiffractionMicrostructure,uDiffractionControl,uDiffractionCoating,uDiffractionRoughness,uDiffractionSpectral[8];\n");
    const auto helpers=source.find("uint queryWork=0u;");
    if(helpers==std::string::npos)return {};
    source.insert(helpers,"#define CAUSTIC_GRATING_VERSION "+std::to_string(kGratingIncidentVersion)+"\n"+materialMath+"\nbool gratingFinite3(vec3 p) {return causticFinite(p.x)&&causticFinite(p.y)&&causticFinite(p.z);}\n");
    const auto hitBegin=source.find("void depositHit(");const auto hitEnd=source.find("int findTriangle(",hitBegin);
    if(hitBegin==std::string::npos||hitEnd==std::string::npos)return {};
    source.erase(hitBegin,hitEnd-hitBegin);
    smoothglass::replace(source,"    int depositBase=w*uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;\n","");
    smoothglass::replace(source,"    for(int i=0;i<uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;++i)deposits[depositBase+i]=vec4(0.0);",
        "    for(int i=0;i<uVisits*24;++i)incidents[w*uVisits*24+i]=vec4(0.0);\n"
        "    for(int i=0;i<uVisits*((uGratingHistoryWords+1)/2);++i)incidents[uGratingHistoryBase+w*uVisits*((uGratingHistoryWords+1)/2)+i]=vec4(0.0);\n"
        "    for(int i=0;i<uVisits*3*4;++i)orders[w*uVisits*3*4+i]=vec4(0.0);");
    smoothglass::replace(source,"float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;",
        "float outgoing=0.0,absorbed=0.0,escaped=0.0,tail=0.0;uint extraQueryWork=0u,orderWork=0u;");
    smoothglass::replace(source,"struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;};",
        "struct Branch {vec3 origin,direction;float fraction;uvec2 history;int interfaces;bool inside;vec3 du,dv;};\n"+nativeGratingDifferentialMathSource());
    smoothglass::replace(source,"    paths[at+4].xy=words(uint(b.interfaces));\n",
        "    paths[at+4].xy=words(uint(b.interfaces));paths[at+5]=vec4(b.du,0.0);paths[at+6]=vec4(b.dv,0.0);\n");
    smoothglass::replace(source,"    b.interfaces=int(id(paths[at+4].xy));\n",
        "    b.interfaces=int(id(paths[at+4].xy));b.du=paths[at+5].xyz;b.dv=paths[at+6].xyz;\n");
    smoothglass::replace(source,"    // pendingStore keeps interfaces 64..127 of each branch's reflection history (glass_caustic_history.h).\n",
        "    // pendingStore keeps interfaces 64..127 of each branch's reflection history (glass_caustic_history.h).\n"
        "    vec3 entryE1=vec3(0.0),entryE2=vec3(0.0);\n");
    smoothglass::replace(source,"        if(complete)pendingStore(pendingBlock(w,pending++),Branch(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,1.0,uvec2(0u),0,false),uvec2(0u));",R"GLSL(        if(complete) {
            int sourceTriangle=findTriangle(entryHit);float dbu,dbv;
            if(sourceTriangle<0) {complete=false;}
            else {
                int t=sourceTriangle*8;entryE1=triangles[t+1].xyz-triangles[t].xyz;entryE2=triangles[t+2].xyz-triangles[t].xyz;
                complete=triangles[t+5].z==1.0&&gratingStrictHit(triangles[t].xyz,triangles[t+1].xyz,triangles[t+2].xyz,entryHit.position,entryHit.outwardNormal)
                    &&gratingBiasDerivative(entry.xyz,entryE1,dbu)&&gratingBiasDerivative(entry.xyz,entryE2,dbv);
                if(complete)pendingStore(pendingBlock(w,pending++),Branch(entry.xyz+uSource.xyz*(2.0*offset),-uSource.xyz,1.0,uvec2(0u),0,false,
                    entryE1+2.0*uSource.xyz*dbu,entryE2+2.0*uSource.xyz*dbv),uvec2(0u));
            }
        })GLSL");
    smoothglass::replace(source,"        vec4 shape=triangles[triangle*8+5];",R"GLSL(        vec4 shape=triangles[triangle*8+5];vec3 hitDu,hitDv;
        int plane=triangle*8;
        if(!gratingStrictHit(triangles[plane].xyz,triangles[plane+1].xyz,triangles[plane+2].xyz,hit.position,hit.outwardNormal)
            ||!gratingPlaneDerivative(branch.du,branch.direction,hit.outwardNormal,hitDu)
            ||!gratingPlaneDerivative(branch.dv,branch.direction,hit.outwardNormal,hitDv)
            ||!gratingArea(entryE1,entryE2,entry.w,hitDu,hitDv)) {complete=false;break;})GLSL");
    smoothglass::replace(source,"            pendingStore(pendingBlock(w,pending++),Branch(hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position)),\n                reflect?reflected:transmitted,fraction,history,branch.interfaces+1,reflect?branch.inside:!branch.inside),high);",R"GLSL(            float dbu,dbv;
            if(!gratingBiasDerivative(hit.position,hitDu,dbu)||!gratingBiasDerivative(hit.position,hitDv,dbv)) {complete=false;break;}
            float side=reflect?1.0:-1.0;
            pendingStore(pendingBlock(w,pending++),Branch(hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position)),
                reflect?reflected:transmitted,fraction,history,branch.interfaces+1,reflect?branch.inside:!branch.inside,
                hitDu+side*normal*dbu,hitDv+side*normal*dbv),high);)GLSL");
    const auto receiverBegin=source.find("        if(shape.z==0.0) {");
    const auto receiverEnd=source.find("        if(branch.inside==hit.entering)",receiverBegin);
    if(receiverBegin==std::string::npos||receiverEnd==std::string::npos)return {};
    source.replace(receiverBegin,receiverEnd-receiverBegin,R"GLSL(        if(shape.z==0.0) {
            if(branch.inside||!hit.entering||branch.interfaces==0) {complete=false;break;}
            int target=-1;
            for(int t=0;t<uWitnessCount;++t) {
                vec4 o=witness[t*8+4],i=witness[t*8+5];
                if(id(o.xy)==hit.objectId&&id(o.zw)==hit.materialId&&uint(i.x)==hit.instanceId&&uint(i.y)==hit.triangleId) {target=t;break;}
            }
            // Other staging receivers terminate their actual macro branch.
            // Their ordinary power is not relabeled as a grating event.
            if(target<0)continue;
            int t=target*8;
            vec3 a=witness[t].xyz,e1=witness[t+1].xyz-a,e2=witness[t+2].xyz-a,delta=hit.position-a;
            float q00=dot(e1,e1),q01=dot(e1,e2),q11=dot(e2,e2),det=q00*q11-q01*q01;
            if(!causticFinite(det)||det<=0.0) {complete=false;break;}
            float b1=(q11*dot(delta,e1)-q01*dot(delta,e2))/det,b2=(q00*dot(delta,e2)-q01*dot(delta,e1))/det;
            vec3 barycentric=vec3(1.0-b1-b2,b1,b2);
            vec2 uv=vec2(dot(barycentric,vec3(witness[t].w,witness[t+1].w,witness[t+2].w)),dot(barycentric,witness[t+3].xyz));
            vec3 tangent=witness[t+6].xyz,normal=witness[t+7].xyz,bitangent=cross(normal,tangent);
            vec3 incident=-vec3(dot(branch.direction,tangent),dot(branch.direction,bitangent),dot(branch.direction,normal));
            float tolerance=8.0*rayBias(hit.position);
            if(!gratingFinite3(barycentric)||any(lessThanEqual(barycentric,vec3(1e-5)))||any(greaterThanEqual(barycentric,vec3(1.0-1e-5)))
                ||length(a+e1*b1+e2*b2-hit.position)>tolerance||length(normal-hit.outwardNormal)>2e-4
                ||incident.z<=1e-6||!gratingArea(entryE1,entryE2,entry.w,hitDu,hitDv)) {complete=false;break;}
            float biasDu,biasDv;
            if(!gratingBiasDerivative(hit.position,hitDu,biasDu)||!gratingBiasDerivative(hit.position,hitDv,biasDv)) {complete=false;break;}
            float incidentPower=potential*branch.fraction;
            float wavelength=uDiffractionSpectral[band].x;
            float substrate=spectralReflectance(wavelength,int(uDiffractionControl.w+0.5),uDiffractionMicrostructure.y,uDiffractionMicrostructure.z);
            if(!causticFinite(substrate)||substrate<0.0||substrate>1.0) {complete=false;break;}
            int locator=w*uVisits+count,d=locator*24,orderCount=0;float resolved=0.0,efficiencySum=0.0;
            orderWork+=3u;
            for(int primary=-1;primary<=1;++primary) {
                /*GRATING_NATIVE_ORDER_EQUATION*/
                if(!causticFinite(efficiency)||efficiency<0.0) {complete=false;break;}
                efficiencySum+=efficiency;float power=incidentPower*substrate*efficiency;
                if(power==0.0)continue;
                vec3 direction=tangent*centre.x+bitangent*centre.y+normal*z;
                if(!gratingFinite3(direction)||!causticFinite(power)||power<0.0||abs(dot(direction,direction)-1.0)>1e-4
                    ||visits>=uVisits) {complete=false;break;}
                ++visits;
                SceneClosestHit childHit=query(hit.position+normal*rayBias(hit.position),direction,1e-5,uSource.w);
                extraQueryWork+=causticLastWork;
                if(childHit.status<0) {complete=false;break;}
                int orderBase=(locator*3+orderCount)*4;
                orders[orderBase]=vec4(float(locator),float(primary+1),0.0,0.0);
                orders[orderBase+1]=vec4(direction,power);++orderCount;resolved+=power;
            }
            if(!complete||!causticFinite(efficiencySum)||efficiencySum>1.0002) {complete=false;break;}
            incidents[d]=vec4(float(w),float(band),float(branch.interfaces),float(orderCount));
            incidents[d+1]=vec4(words(branch.history.x),words(branch.history.y));
            // Inline tokens cover the first 32 interfaces; the tail holds the rest of the history.
            for(int depth=0;depth<min(branch.interfaces,32);++depth)
                incidents[d+2+depth/4][depth%4]=((branch.history.x>>uint(depth))&1u)!=0u?1.0:65.0;
            int historyAt=uGratingHistoryBase+locator*((uGratingHistoryWords+1)/2);
            for(int j=0;j<uGratingHistoryWords;++j) {
                vec2 halves;
                if(j<2)halves=words(j==0?branchHigh.x:branchHigh.y);
                else {int k=j-2,at=historyBranchBlock(w,uPending)+k/2;halves=(k%2)==0?paths[at].xy:paths[at].zw;}
                if((j%2)==0)incidents[historyAt+j/2].xy=halves;else incidents[historyAt+j/2].zw=halves;
            }
            incidents[d+10]=vec4(branch.origin,branch.fraction);incidents[d+11]=vec4(branch.direction,incidentPower);
            incidents[d+12]=vec4(hit.position,hit.distance);incidents[d+13]=owners(hit);
            incidents[d+14]=vec4(float(hit.instanceId),float(hit.triangleId),words(hitWork));
            incidents[d+15]=vec4(hit.outwardNormal,float(hit.entering));incidents[d+16]=vec4(barycentric,0.0);
            incidents[d+17]=vec4(uv,0.0,0.0);incidents[d+18]=vec4(tangent,0.0);incidents[d+19]=vec4(bitangent,0.0);
            incidents[d+20]=vec4(normal,0.0);incidents[d+21]=vec4(incidentPower*substrate,incidentPower*(1.0-substrate),max(0.0,incidentPower*substrate-resolved),0.0);
            incidents[d+22]=vec4(float(CAUSTIC_GRATING_VERSION),1.0,hitDu.xy);incidents[d+23]=vec4(hitDu.z,hitDv);++count;continue;
        }
)GLSL");
    smoothglass::replace(source,"    paths[base+1]=vec4(float(complete),float(blocked),0.0,0.0);",
        "    paths[base+1]=vec4(float(complete),float(blocked),words(extraQueryWork));");
    // Texel 4.y carries orderWork here. Grating frames have no thin absorber.
    smoothglass::replace(source,"\n    paths[base+4].y=thinAbsorbed;","");
    smoothglass::replace(source,"paths[base+4]=vec4(tail,0.0,0.0,0.0);",
        "paths[base+4]=vec4(tail,float(orderWork),0.0,0.0);");
    smoothglass::replace(source,"/*GRATING_NATIVE_ORDER_EQUATION*/",orderEquation);
    return source;
}

// Explicit dimensional source route. Legacy generators retain their bytes.
// std430 input48 + echo96 + one pre-blocking potential per scheduled work item.
inline std::string directionalIrradianceComputeSource(std::string source)
{
    const auto declarations=source.find("uint queryWork=0u;");
    const auto potential=source.find("uLi[band/4][band%4]");
    const auto receipt=source.find("    paths[base]=vec4(words(invocation),0.0,0.0);");
    if(declarations==std::string::npos||potential==std::string::npos||receipt==std::string::npos
        ||source.find("uLi[band/4][band%4]",potential+1)!=std::string::npos)return {};
    source.replace(receipt,0,R"GLSL(    directionalPotentials[w]=potential;
    if(invocation==0u) {
        directionalEchoIdentifiers=directionalIdentifiers;
        directionalEchoCoefficients[0]=directionalCoefficients[0];
        directionalEchoCoefficients[1]=directionalCoefficients[1];
        directionalEchoRawRadiance[0]=uLi[0];directionalEchoRawRadiance[1]=uLi[1];
        directionalEchoDirectionTrace=uSource;
    }
)GLSL");
    source.replace(potential,std::string("uLi[band/4][band%4]").size(),"directionalCoefficients[band/4][band%4]");
    // Guard before any dimensional power is computed. Invalid inputs leave
    // incomplete path receipts and cannot pass the whole-batch validation.
    const auto entry=source.find("    int band=int(entryIndices.w);");
    if(entry==std::string::npos)return {};
    source.insert(entry,R"GLSL(    if(directionalIdentifiers.x!=1u||directionalIdentifiers.y!=1u
        ||directionalIdentifiers.z!=1u||directionalIdentifiers.w==0u)return;
    for(int b=0;b<8;++b) {
        float coefficient=directionalCoefficients[b/4][b%4];
        if(!causticFinite(coefficient)||coefficient<0.0||coefficient>16.0)return;
    }
)GLSL");
    source.insert(declarations,R"GLSL(layout(std430,binding=7) buffer DirectionalSource {
    uvec4 directionalIdentifiers;
    vec4 directionalCoefficients[2];
    uvec4 directionalEchoIdentifiers;
    vec4 directionalEchoCoefficients[2];
    vec4 directionalEchoRawRadiance[2];
    vec4 directionalEchoDirectionTrace;
    float directionalPotentials[];
};
)GLSL");
    return source;
}

inline bool decodeGratingWork(const Texel* p,NativeGratingReadback::WorkReceipt& output)
{
    NativeGratingReadback::WorkReceipt r;std::uint32_t queries=0;
    for(unsigned t=0;t<kGratingWorkReceiptStride;++t)for(float value:p[t])if(!std::isfinite(value))return false;
    if(!causticWord(p[0][0],kMaximumBranchVisits,r.incidentCount)
        ||!causticWord(p[0][1],kMaximumExactReceiptCount,r.orderWork)||r.orderWork!=r.incidentCount*3
        ||!causticId(p[0][2],p[0][3],queries)||p[1][0]<0||p[1][1]<0||p[1][2]!=1||p[1][3]!=0)return false;
    r.queryWork=queries;r.absorbedPower=p[1][0];r.unresolvedPower=p[1][1];r.complete=true;output=r;return true;
}
inline bool decodeGratingChild(const Texel* p,NativeGratingReadback::ChildReceipt& output)
{
    NativeGratingReadback::ChildReceipt r;std::uint32_t order=0,disposition=0,queries=0;
    for(unsigned t=0;t<kGratingChildReceiptStride;++t)for(float value:p[t])if(!std::isfinite(value))return false;
    if(!causticWord(p[0][0],kMaximumExactReceiptCount-1,r.incident)
        ||!causticWord(p[0][1],2,order)||p[0][2]!=1||!causticWord(p[0][3],2,disposition)
        ||p[1][0]<0||p[1][1]<0||!causticId(p[1][2],p[1][3],queries)||p[7]!=Texel{}||!causticHit(p+2,r.hit))return false;
    r.order=int(order)-1;r.disposition=NativeGratingReadback::Disposition(disposition);r.queryWork=queries;
    r.escapedPower=p[1][0];r.unresolvedPower=p[1][1];r.complete=true;output=r;return true;
}
inline bool decodeGratingDeposit(const Texel* p,NativeDeposit& output)
{
    std::array<Texel,kDepositReceiptStride> legacy;std::copy_n(p,legacy.size(),legacy.begin());
    std::fill(legacy.begin()+22,legacy.end(),Texel{});NativeDeposit r;
    // Texel 31 holds history words 2 and 3 for every grating terminal deposit;
    // texel 30 is the grating extension. Words 4 and up are the Paths tail.
    std::uint32_t wideLow=0,wideHigh=0;
    if(!causticDecodeDeposit(legacy.data(),0,r)||!causticId(p[31][0],p[31][1],wideLow)||!causticId(p[31][2],p[31][3],wideHigh)
        ||!decodeGratingDepositExtension(p[30],Texel{},r))return false;
    r.reflectionHistory=CausticHistory::fromWords(r.reflectionHistory.word(0),r.reflectionHistory.word(1),wideLow,wideHigh);
    if(!r.reflectionHistory.clearFrom(r.interfaces))return false;
    std::array<Texel,8> ancestry;std::copy_n(p+22,ancestry.size(),ancestry.begin());
    if(r.gratingVersion) {if(!decodeGratingDepositAncestry(ancestry,r.ancestry))return false;}
    else if(!decodeMacroAncestry(ancestry,0,r.ancestry))return false;
    output=r;return true;
}
inline std::string gratingTerminalComputeSource(const std::string& materialMath,const std::string& orderEquation)
{
    auto source=gratingIncidentComputeSource(materialMath,orderEquation);if(source.empty())return {};
    const auto original=computeSource();
    const auto receiverBegin=original.find("        if(shape.z==0.0) {");
    const auto receiverEnd=original.find("        if(branch.inside==hit.entering)",receiverBegin);
    const auto hitBegin=original.find("void depositHit(");const auto hitEnd=original.find("int findTriangle(",hitBegin);
    if(receiverBegin==std::string::npos||receiverEnd==std::string::npos||hitBegin==std::string::npos||hitEnd==std::string::npos)return {};
    auto receiver=original.substr(receiverBegin,receiverEnd-receiverBegin);
    smoothglass::replace(receiver,"        if(shape.z==0.0) {",R"GLSL(bool gratingReceiver(int w,int band,float potential,Branch branch,SceneClosestHit hit,uint hitWork,int triangle,
    uvec2 branchHigh,bool coupled,int locator,int signedOrder,inout int depositCount,inout float outgoing,inout float absorbed,out uint cameraQueries) {
    cameraQueries=0u;vec4 shape=triangles[triangle*8+5];
    if(depositCount>=uVisits)return false;
)GLSL");
    smoothglass::replace(receiver,"complete=false;break;","return false;");
    smoothglass::replace(receiver,"int d=depositBase+count*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;","int d=(w*uVisits+depositCount)*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;");
    smoothglass::replace(receiver,"words(invocation)","words(uint(w))");
    // Texel 30 is the grating extension here, so history words 2 and 3 move to 31.
    smoothglass::replace(receiver,"deposits[d+30]=vec4(words(branchHigh.x),words(branchHigh.y));",
        "deposits[d+31]=vec4(words(branchHigh.x),words(branchHigh.y));");
    smoothglass::replace(receiver,"historyDepositBlock(w,count)","historyDepositBlock(w,depositCount)");
    smoothglass::replace(receiver,"++count;continue;",R"GLSL(if(coupled) {
                for(int depth=0;depth<min(branch.interfaces,32);++depth)
                    deposits[d+22+depth/4][depth%4]=((branch.history.x>>uint(depth))&1u)!=0u?1.0:65.0;
                deposits[d+30]=vec4(float(CAUSTIC_GRATING_VERSION),float(locator),float(signedOrder+1),0.0);
            }
            cameraQueries=cameraWork+pixelWork;++depositCount;return true;)GLSL");
    const auto main=source.find("void main() {");if(main==std::string::npos)return {};
    source.insert(main,original.substr(hitBegin,hitEnd-hitBegin)+R"GLSL(
void gratingChildHit(int offset,SceneClosestHit hit,uint work) {
    orders[offset]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    orders[offset+1]=vec4(hit.position,float(work>>16u));orders[offset+2]=vec4(hit.outwardNormal,0.0);
    orders[offset+3]=owners(hit);orders[offset+4]=indices(hit);
}
)GLSL"+receiver);
    smoothglass::replace(source,"uniform int uWitnessCount;","layout(std430,binding=6) buffer Deposits {vec4 deposits[];};\nuniform int uWitnessCount,uGratingWorkBase,uGratingChildBase;");
    smoothglass::replace(source,"uint extraQueryWork=0u,orderWork=0u;","uint extraQueryWork=0u,orderWork=0u;int depositCount=0;float gratingAbsorbed=0.0,gratingTail=0.0;");
    smoothglass::replace(source,"    // Only this invocation writes its fixed receipt segment.",R"GLSL(    // Only this invocation writes its fixed receipt segment.
    for(int i=0;i<uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE;++i)deposits[w*uVisits*CAUSTIC_DEPOSIT_RECEIPT_STRIDE+i]=vec4(0.0);
    for(int i=0;i<uVisits*3*8;++i)orders[uGratingChildBase+w*uVisits*3*8+i]=vec4(0.0);
    incidents[uGratingWorkBase+w*2]=vec4(0.0);incidents[uGratingWorkBase+w*2+1]=vec4(0.0);
)GLSL");
    smoothglass::replace(source,"            if(target<0)continue;",R"GLSL(            if(target<0) {
                uint cameraQueries;
                if(!gratingReceiver(w,band,potential,branch,hit,hitWork,triangle,branchHigh,false,0,0,depositCount,outgoing,absorbed,cameraQueries)) {complete=false;break;}
                continue;
            })GLSL");
    smoothglass::replace(source,"                if(power==0.0)continue;",R"GLSL(                if(power==0.0) {
                    if(incidentPower>0.0&&substrate>0.0&&efficiency>0.0) {complete=false;break;}
                    continue;
                })GLSL");
    const auto orderBegin=source.find("                ++visits;\n                SceneClosestHit childHit=");
    const auto orderEnd=source.find("                int orderBase=(locator*3+orderCount)*4;",orderBegin);
    if(orderBegin==std::string::npos||orderEnd==std::string::npos)return {};
    source.replace(orderBegin,orderEnd-orderBegin,R"GLSL(                ++visits;
                SceneClosestHit childHit=query(hit.position+normal*rayBias(hit.position),direction,1e-5,uSource.w);
                uint childWork=causticLastWork;extraQueryWork+=childWork;
                if(childHit.status<0) {complete=false;break;}
                int childBase=uGratingChildBase+(locator*3+orderCount)*8;uint cameraQueries=0u;
                if(childHit.status==0) {
                    escaped+=power;orders[childBase]=vec4(float(locator),float(primary+1),1.0,0.0);
                    orders[childBase+1]=vec4(power,0.0,words(childWork));
                } else {
                    int childTriangle=findTriangle(childHit);
                    if(childTriangle<0||childHit.objectId==hit.objectId||triangles[childTriangle*8+5].z!=0.0) {complete=false;break;}
                    float fraction=power/potential;
                    if(!causticFinite(fraction)||fraction<=0.0) {complete=false;break;}
                    Branch child=branch;child.origin=hit.position+normal*rayBias(hit.position);child.direction=direction;child.fraction=fraction;child.inside=false;
                    if(!gratingReceiver(w,band,potential,child,childHit,childWork,childTriangle,branchHigh,true,locator,primary,
                        depositCount,outgoing,absorbed,cameraQueries)) {complete=false;break;}
                    extraQueryWork+=cameraQueries;
                    orders[childBase]=vec4(float(locator),float(primary+1),1.0,1.0);
                    orders[childBase+1]=vec4(0.0,0.0,words(childWork+cameraQueries));
                }
                gratingChildHit(childBase+2,childHit,childWork);
)GLSL");
    smoothglass::replace(source,"            incidents[d+20]=vec4(normal,0.0);incidents[d+21]=",R"GLSL(            float substrateAbsorption=incidentPower*(1.0-substrate),profileTail=max(0.0,incidentPower*substrate-resolved);
            absorbed+=substrateAbsorption;tail+=profileTail;gratingAbsorbed+=substrateAbsorption;gratingTail+=profileTail;
            incidents[d+20]=vec4(normal,0.0);incidents[d+21]=)GLSL");
    smoothglass::replace(source,"    paths[base].zw=vec2(float(count),float(visits));", "    paths[base].zw=vec2(float(depositCount),float(visits));");
    smoothglass::replace(source,"    paths[base+1]=vec4(float(complete),float(blocked),words(extraQueryWork));",R"GLSL(    incidents[uGratingWorkBase+w*2]=vec4(float(count),float(orderWork),words(extraQueryWork));
    incidents[uGratingWorkBase+w*2+1]=vec4(gratingAbsorbed,gratingTail,float(complete),0.0);
    paths[base+1]=vec4(float(complete),float(blocked),words(extraQueryWork));)GLSL");
    return source;
}

inline constexpr const char* depositionKeySource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Paths {vec4 paths[];};
layout(std430,binding=1) readonly buffer Deposits {vec4 deposits[];};
layout(std430,binding=5) buffer Keys {uvec4 keys[];};
uniform int uWorkCount,uVisits,uKeyCount;
uniform vec2 uExtent;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uvec4 key=uvec4(0xffffffffu);
    uint work=at/uint(uVisits),slot=at%uint(uVisits);
    if(work<uint(uWorkCount)&&slot<uint(paths[work*16u].z)) {
        uint deposit=at*32u;
        vec4 target=deposits[deposit+3u];
        if(deposits[deposit].w==1.0&&target.x==1.0&&target.y>=0.0&&target.z>=0.0
            &&target.y<uExtent.x&&target.z<uExtent.y)
            key=uvec4(uint(target.z)*uint(uExtent.x)+uint(target.y),work,slot,at);
    }
    keys[at]=key;
}
)GLSL";
inline constexpr const char* depositionSortSource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=5) buffer Keys {uvec4 keys[];};
uniform int uKeyCount,uSortSpan,uSortDistance;
bool lessKey(uvec4 a,uvec4 b) {
    if(a.x!=b.x)return a.x<b.x;
    if(a.y!=b.y)return a.y<b.y;
    if(a.z!=b.z)return a.z<b.z;
    return a.w<b.w;
}
void main() {
    uint pair=gl_GlobalInvocationID.x;if(pair>=uint(uKeyCount)/2u)return;
    uint distance=uint(uSortDistance);
    uint first=(pair/distance)*(2u*distance)+pair%distance,second=first+distance;
    uvec4 a=keys[first],b=keys[second];
    bool ascending=(first&uint(uSortSpan))==0u;
    if(ascending?lessKey(b,a):lessKey(a,b)) {keys[first]=b;keys[second]=a;}
}
)GLSL";
inline std::string depositionReduceSource()
{
    return std::string("#version 430 core\n")+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=1) readonly buffer Deposits {vec4 deposits[];};
layout(std430,binding=2) buffer SpectrumLow {uint bandsLow[];};
layout(std430,binding=6) buffer SpectrumHigh {uint bandsHigh[];};
layout(std430,binding=3) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=5) readonly buffer Keys {uvec4 keys[];};
uniform int uKeyCount;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uint pixel=keys[at].x;
    if(pixel==0xffffffffu||(at>0u&&keys[at-1u].x==pixel))return;
    float sum[8];for(int b=0;b<8;++b)sum[b]=0.0;
    // One segment leader writes each pixel. Lexicographic keys preserve the
    // native work/deposit order without any unordered floating-point atomics.
    for(uint next=at;next<uint(uKeyCount)&&keys[next].x==pixel;++next) {
        uvec4 key=keys[next];uint deposit=key.w*uint(CAUSTIC_DEPOSIT_RECEIPT_STRIDE);
        int band=int(workTable[key.y*4u+3u].w);
        sum[band]+=causticPixelRadiance(deposits[deposit+2u].z,deposits[deposit+4u].x);
    }
    for(int b=0;b<4;++b) {bandsLow[pixel*4u+uint(b)]=floatBitsToUint(sum[b]);
        bandsHigh[pixel*4u+uint(b)]=floatBitsToUint(sum[b+4]);}
}
)GLSL";
}
// Kernel splat deposition (Frame::splatVersion 1) for ordinary caustic frames.
// Grating paths never link these programs. After the trace the render runs the
// normaliser, the tile keys, the shared bitonic sort over the tile keys, the
// tile segment table and one gather thread per pixel. Tile keys use binding 5,
// sized frame.splat.paddedKeyCount. Binding 7 is one uint scratch array, laid
// out by SplatScratchLayout: a float normaliser per trace slot, a begin/end
// pair per tile, the sorted triangle owner rows and two counters.
inline constexpr unsigned kSplatScratchBinding=7;
inline constexpr std::uint64_t kSplatTriangleRowWords=5,kSplatCounterWords=2;
// Counter word 0 collects these overflow bits. Word 1 is the measured gather work.
// A path with more deposits than its keyed slots sets kSplatOverflowKeySlots.
// Like an out-of-image tile it is a defect, so the frame refuses.
inline constexpr std::uint32_t kSplatOverflowTile=1,kSplatOverflowTileCap=2,kSplatOverflowGatherWork=4,kSplatOverflowKeySlots=8;
struct SplatScratchLayout
{
    std::uint64_t normaliserBase=0,rangeBase=0,lookupBase=0,lookupCount=0,counterBase=0,words=0;
};
inline bool splatScratchLayout(const Frame& frame,SplatScratchLayout& output)
{
    const auto tiles=frame.splat.tilesX*frame.splat.tilesY;
    if(frame.splatVersion!=kSplatVersion || !tiles || frame.work.empty() || frame.triangles.empty())return false;
    SplatScratchLayout layout;
    layout.rangeBase=frame.bounds.maximumDeposits;layout.lookupBase=layout.rangeBase+2*tiles;
    layout.lookupCount=frame.triangles.size();
    layout.counterBase=layout.lookupBase+kSplatTriangleRowWords*layout.lookupCount;
    layout.words=layout.counterBase+kSplatCounterWords;
    if(layout.words>UINT32_MAX)return false;
    output=layout;return true;
}
// Why splatScratchLayout refuses a frame. limit is true only for a word count past the
// index range; a missing precondition is an invalid frame, not a GL limit.
inline std::string splatScratchLayoutFailure(const Frame& frame,bool& limit)
{
    limit=false;
    const auto tiles=frame.splat.tilesX*frame.splat.tilesY;
    if(frame.splatVersion!=kSplatVersion)return "the frame does not splat";
    if(!tiles||frame.work.empty()||frame.triangles.empty())
        return "the frame has "+std::to_string(tiles)+" tiles, "
            +std::to_string(frame.work.size())+" paths, "+std::to_string(frame.triangles.size())+" triangles";
    const auto words=frame.bounds.maximumDeposits+2*tiles+kSplatTriangleRowWords*frame.triangles.size()+kSplatCounterWords;
    limit=true;return "splat scratch needs "+std::to_string(words)+" words, max "+std::to_string(std::uint64_t(UINT32_MAX));
}
// Initial scratch words. Normalisers, tile ranges and counters start at zero.
// Triangle rows are (instance, triangle, object, material, table index), sorted
// so a lower bound finds the lowest table index among equal owners, the row
// findTriangle and the CPU hitTriangle pick.
inline std::vector<std::uint32_t> splatScratchWords(const Frame& frame,const SplatScratchLayout& layout)
{
    std::vector<std::uint32_t> words(std::size_t(layout.words),0u);
    std::vector<std::uint32_t> order(frame.triangles.size());
    for(std::size_t i=0;i<order.size();++i)order[i]=std::uint32_t(i);
    const auto owner=[&](std::uint32_t i) {
        const auto& t=frame.triangles[i];return std::make_tuple(t.instance,t.triangle,t.object,t.material,i);
    };
    std::sort(order.begin(),order.end(),[&](std::uint32_t a,std::uint32_t b){return owner(a)<owner(b);});
    for(std::size_t i=0;i<order.size();++i) {
        const auto& t=frame.triangles[order[i]];auto* row=words.data()+layout.lookupBase+i*kSplatTriangleRowWords;
        row[0]=t.instance;row[1]=t.triangle;row[2]=t.object;row[3]=t.material;row[4]=order[i];
    }
    return words;
}
// The host dispatches the normaliser and gather passes in bands and waits for
// each band. One 4K gather is tens of millions of key-pixel evaluations, and a
// single long submission can trip the NVIDIA context switch timeout (Xid 109),
// which drops the rest of the dispatch. A gather band is whole 16-row groups
// of at most kSplatGatherBandPixels pixels, the bound the closed glass draw
// uses. A normaliser band is kSplatNormaliserBandSlots trace slots.
inline constexpr std::uint64_t kSplatGatherBandPixels=262144;
inline constexpr std::uint64_t kSplatNormaliserBandSlots=16384;
inline std::uint32_t splatGatherBandRows(std::uint32_t width)
{
    const std::uint64_t group=kSplatGatherGroup,rows=width?kSplatGatherBandPixels/width/group*group:group;
    return std::uint32_t(std::max<std::uint64_t>(group,rows));
}
// Shared by the four splat programs. A window pixel is accepted by the CPU rule
// in buildSplatReference: the private raster's admitted triangle at the pixel is
// a reflective non-glass receiver, its receiver-plane footprint through the pixel
// centre projects back to the pixel, and its normal agrees with the deposit's.
static_assert(kSplatNormalAgreement==.9f,"splatCommonSource spells the normal agreement literally");
static_assert(kSplatTilesPerDeposit==9,"splatKeySource emits at most nine tile keys per deposit");
static_assert(kSplatGatherGroup==16,"splatGatherSource spells its 16x16 work group literally");
inline std::string splatCommonSource()
{
    return std::string("#version 430 core\n")+nativeMathSource()+nativeSplatMathSource()
        +"#define CAUSTIC_SPLAT_GATHER_CAP "+std::to_string(kMaximumSplatGatherWork)+"u\n"+R"GLSL(
layout(std430,binding=1) readonly buffer Deposits {vec4 deposits[];};
layout(std430,binding=3) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
layout(std430,binding=7) buffer Splat {uint splat[];};
uniform sampler2D uRasterOwners,uRasterTriangle;
uniform vec4 uCamera,uRotation,uProjection;
uniform vec2 uExtent;
uniform int uWorkCount,uVisits,uKeyCount,uKeySlots;
uniform uint uNormaliserBase,uRangeBase,uLookupBase,uLookupCount,uCounterBase,uTilesX,uTilesY;
// First slot of a normaliser band and first row of a gather band.
uniform uint uSlotOffset,uRowOffset;
uint causticSplatId(vec2 words) {return uint(words.x)|(uint(words.y)<<16u);}
bool causticSplatOwnerLess(uvec4 a,uvec4 b) {
    if(a.x!=b.x)return a.x<b.x;
    if(a.y!=b.y)return a.y<b.y;
    if(a.z!=b.z)return a.z<b.z;
    return a.w<b.w;
}
uvec4 causticSplatOwnerRow(uint row) {
    uint at=uLookupBase+row*5u;return uvec4(splat[at],splat[at+1u],splat[at+2u],splat[at+3u]);
}
// Admitted triangle table index with these raster owners, or -1.
int causticSplatTriangle(uvec4 owner) {
    uint low=0u,high=uLookupCount;
    while(low<high) {
        uint middle=low+(high-low)/2u;
        if(causticSplatOwnerLess(causticSplatOwnerRow(middle),owner))low=middle+1u;else high=middle;
    }
    if(low>=uLookupCount||causticSplatOwnerRow(low)!=owner)return -1;
    return int(splat[uLookupBase+low*5u+4u]);
}
// The receiver seen through the centre of top-first pixel p. position is on the
// centre ray and the admitted triangle's plane; area is its footprint area.
bool causticSplatSurface(uvec2 pixel,out vec3 position,out vec3 normal,out float area) {
    position=vec3(0.0);normal=vec3(0.0);area=0.0;
    ivec2 texel=ivec2(int(pixel.x),int(uExtent.y)-1-int(pixel.y));
    vec4 raster=texelFetch(uRasterTriangle,texel,0);
    if(raster.w!=1.0)return false;
    vec4 owners=texelFetch(uRasterOwners,texel,0);
    int t=causticSplatTriangle(uvec4(uint(raster.x),uint(raster.y),causticSplatId(owners.xy),causticSplatId(owners.zw)));
    if(t<0)return false;
    vec4 shape=triangles[t*8+5];
    if(shape.z!=0.0||shape.w!=1.0)return false;
    normal=triangles[t*8+3].xyz;
    if(!causticPixelCorner(vec2(pixel)+vec2(0.5),triangles[t*8].xyz,normal,uCamera.xyz,uRotation,uProjection,uExtent,position))
        return false;
    vec4 footprint;
    if(causticProjectReceiver(position,normal,uCamera.xyz,uRotation,uProjection,uExtent,footprint)!=1
        ||uint(footprint.x)!=pixel.x||uint(footprint.y)!=pixel.y)return false;
    area=footprint.w;return causticFinite(area)&&area>0.0;
}
// A camera-visible deposit slot with a projected in-image footprint pixel.
bool causticSplatVisible(uint index,out uvec2 pixel) {
    uint d=index*uint(CAUSTIC_DEPOSIT_RECEIPT_STRIDE);vec4 target=deposits[d+3u];pixel=uvec2(0u);
    if(deposits[d].w!=1.0||target.x!=1.0||!(target.y>=0.0&&target.z>=0.0&&target.y<uExtent.x&&target.z<uExtent.y))
        return false;
    pixel=uvec2(target.yz);return true;
}
struct CausticSplatDeposit {uvec2 pixel;vec3 position,normal;float outgoing,radius,spread;int band;};
bool causticSplatDepositAt(uint index,out CausticSplatDeposit deposit) {
    deposit.position=vec3(0.0);deposit.normal=vec3(0.0);deposit.outgoing=0.0;deposit.radius=0.0;deposit.spread=0.0;deposit.band=0;
    if(!causticSplatVisible(index,deposit.pixel))return false;
    uint d=index*uint(CAUSTIC_DEPOSIT_RECEIPT_STRIDE),work=index/uint(uVisits);
    deposit.position=deposits[d+8u].xyz;deposit.normal=deposits[d+9u].xyz;deposit.outgoing=deposits[d+2u].z;
    deposit.band=int(workTable[work*4u+3u].w);
    deposit.radius=causticSplatRadiusFromScale(workTable[work*4u].w,
        causticSplatPixelScale(deposit.pixel,deposit.position,deposit.normal,uCamera.xyz,uRotation,uProjection,uExtent));
    deposit.spread=causticSplatSpread(deposit.position,deposit.normal,
        causticEndpointOrigin(deposit.position,uCamera.xyz,uRotation,uProjection));
    return causticFinite(deposit.radius)&&deposit.radius>0.0&&causticFinite(deposit.spread)&&deposit.band>=0&&deposit.band<8;
}
// Inclusive window (x0,y0,x1,y1) of a deposit pixel, clipped to the image.
uvec4 causticSplatWindow(uvec2 pixel) {
    uint r=uint(CAUSTIC_SPLAT_RADIUS);
    return uvec4(pixel.x>=r?pixel.x-r:0u,pixel.y>=r?pixel.y-r:0u,
        min(pixel.x+r,uint(uExtent.x)-1u),min(pixel.y+r,uint(uExtent.y)-1u));
}
// K times the grazing spread for an accepted pixel. The deposit's own pixel
// always weighs 1.
float causticSplatWeight(CausticSplatDeposit deposit,uvec2 pixel,vec3 position,vec3 normal) {
    if(!(dot(normal,deposit.normal)>=0.9))return 0.0;
    if(pixel==deposit.pixel)return 1.0;
    return causticSplatKernel(length(position-deposit.position),deposit.radius)*deposit.spread;
}
)GLSL";
}
// One thread per trace slot (work*visits+slot). N is the sum of K*area over the
// accepted window pixels, written as float bits, and 0 for a hidden slot.
inline std::string splatNormaliserSource()
{
    return splatCommonSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Paths {vec4 paths[];};
void main() {
    uint index=gl_GlobalInvocationID.x+uSlotOffset;if(index>=uint(uWorkCount)*uint(uVisits))return;
    uint work=index/uint(uVisits),slot=index%uint(uVisits);
    float normaliser=0.0;CausticSplatDeposit deposit;
    if(slot<uint(paths[work*uint(CAUSTIC_PATH_RECEIPT_STRIDE)].z)&&causticSplatDepositAt(index,deposit)) {
        uvec4 window=causticSplatWindow(deposit.pixel);
        for(uint y=window.y;y<=window.w;++y)for(uint x=window.x;x<=window.z;++x) {
            vec3 position,normal;float area;
            if(!causticSplatSurface(uvec2(x,y),position,normal,area))continue;
            normaliser+=causticSplatWeight(deposit,uvec2(x,y),position,normal)*area;
        }
    }
    splat[uNormaliserBase+index]=floatBitsToUint(normaliser);
}
)GLSL";
}
// One thread per padded key. Keyed slot k = work*uKeySlots+slot owns keys
// 9k..9k+8: one (tile, work, slot, slot index) key per 32x32 tile its window
// touches, then all-ones sentinels. The slot index is work*uVisits+slot.
inline std::string splatKeySource()
{
    return splatCommonSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Paths {vec4 paths[];};
layout(std430,binding=5) writeonly buffer Keys {uvec4 keys[];};
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uvec4 key=uvec4(0xffffffffu);
    uint per=uint(CAUSTIC_SPLAT_TILES_PER_DEPOSIT),keyed=at/per,which=at%per,work=keyed/uint(uKeySlots),slot=keyed%uint(uKeySlots);
    uint index=work*uint(uVisits)+slot;uvec2 pixel;
    // The last keyed slot of a path flags deposits the keys would skip.
    if(which==0u&&work<uint(uWorkCount)&&slot+1u==uint(uKeySlots)
        &&uint(paths[work*uint(CAUSTIC_PATH_RECEIPT_STRIDE)].z)>uint(uKeySlots))atomicOr(splat[uCounterBase],8u);
    if(work<uint(uWorkCount)&&slot<uint(paths[work*uint(CAUSTIC_PATH_RECEIPT_STRIDE)].z)&&causticSplatVisible(index,pixel)) {
        uvec4 window=causticSplatWindow(pixel);uint tile=uint(CAUSTIC_SPLAT_TILE);
        uint x0=window.x/tile,y0=window.y/tile,columns=window.z/tile-x0+1u,count=columns*(window.w/tile-y0+1u);
        if(which<count)key=uvec4((y0+which/columns)*uTilesX+x0+which%columns,work,slot,index);
    }
    keys[at]=key;
}
)GLSL";
}
// One thread per sorted key. Each run leader records its tile's [begin,end),
// adds tile pixels x run length to the gather work and flags overflow. It scans
// at most one key past the per-tile cap.
inline std::string splatSegmentSource()
{
    return splatCommonSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=5) readonly buffer Keys {uvec4 keys[];};
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uint tile=keys[at].x;
    if(tile==0xffffffffu||(at>0u&&keys[at-1u].x==tile))return;
    if(tile>=uTilesX*uTilesY) {atomicOr(splat[uCounterBase],1u);return;}
    uint cap=uint(CAUSTIC_SPLAT_TILE_CAP),end=at+1u;
    while(end<uint(uKeyCount)&&end-at<=cap&&keys[end].x==tile)++end;
    uint run=end-at;
    if(run>cap)atomicOr(splat[uCounterBase],2u);
    splat[uRangeBase+tile*2u]=at;splat[uRangeBase+tile*2u+1u]=end;
    uint side=uint(CAUSTIC_SPLAT_TILE),tx=tile%uTilesX,ty=tile/uTilesX;
    uint amount=(min(uint(uExtent.x),(tx+1u)*side)-tx*side)*(min(uint(uExtent.y),(ty+1u)*side)-ty*side)*run;
    uint previous=atomicAdd(splat[uCounterBase+1u],amount);
    if(previous>CAUSTIC_SPLAT_GATHER_CAP-amount)atomicOr(splat[uCounterBase],4u);
}
)GLSL";
}
// One thread per pixel, 16x16 work groups, four per 32x32 tile. A pixel walks its tile's
// keys in (work, slot) order, the canonical deposit order, and adds
// outgoing*K/(pi*N) to its deposit's band. Pixels nobody splats write zero.
inline std::string splatGatherSource()
{
    return splatCommonSource()+R"GLSL(
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=2) buffer SpectrumLow {uint bandsLow[];};
layout(std430,binding=6) buffer SpectrumHigh {uint bandsHigh[];};
layout(std430,binding=5) readonly buffer Keys {uvec4 keys[];};
void main() {
    uvec2 pixel=gl_GlobalInvocationID.xy+uvec2(0u,uRowOffset);
    if(pixel.x>=uint(uExtent.x)||pixel.y>=uint(uExtent.y))return;
    float sum[8];for(int b=0;b<8;++b)sum[b]=0.0;
    vec3 position,normal;float area;
    if(causticSplatSurface(pixel,position,normal,area)) {
        uint side=uint(CAUSTIC_SPLAT_TILE),tile=(pixel.y/side)*uTilesX+pixel.x/side;
        uint begin=splat[uRangeBase+tile*2u],end=splat[uRangeBase+tile*2u+1u];
        for(uint at=begin;at<end;++at) {
            uint index=keys[at].w;CausticSplatDeposit deposit;
            if(!causticSplatDepositAt(index,deposit))continue;
            uvec4 window=causticSplatWindow(deposit.pixel);
            if(pixel.x<window.x||pixel.x>window.z||pixel.y<window.y||pixel.y>window.w)continue;
            float normaliser=uintBitsToFloat(splat[uNormaliserBase+index]);
            float weight=causticSplatWeight(deposit,pixel,position,normal);
            if(weight>0.0&&normaliser>0.0)sum[deposit.band]+=deposit.outgoing*weight/(3.14159265358979323846*normaliser);
        }
    }
    uint first=(pixel.y*uint(uExtent.x)+pixel.x)*4u;
    for(int b=0;b<4;++b) {bandsLow[first+uint(b)]=floatBitsToUint(sum[b]);bandsHigh[first+uint(b)]=floatBitsToUint(sum[b+4]);}
}
)GLSL";
}
inline std::string rasterVertexSource()
{
    return std::string("#version 430 core\n")+nativeMathSource()+R"GLSL(
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform int uTriangle;
uniform vec4 uCamera,uRotation,uProjection;
out float linearDepth;
void main() {
    vec3 world=triangles[uTriangle*8+gl_VertexID].xyz;
    vec3 local=causticRotate(vec4(-uRotation.xyz,uRotation.w),world-uCamera.xyz);
    float depth=-local.z,n=uProjection.z,f=uProjection.w;linearDepth=depth;
    // uProjection.x below zero is an orthographic camera, minus half its view height.
    gl_Position=uProjection.x<0.0
        ? vec4(local.x/(-uProjection.x*uProjection.y),local.y/-uProjection.x,(2.0*depth-f-n)/(f-n),1.0)
        : vec4(local.x/(uProjection.x*uProjection.y),local.y/uProjection.x,
            ((f+n)/(f-n))*depth-2.0*f*n/(f-n),depth);
}
)GLSL";
}
inline constexpr const char* rasterFragmentSource=R"GLSL(#version 430 core
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform int uTriangle;
in float linearDepth;
layout(location=0) out vec4 owners;
layout(location=1) out vec4 identityDepth;
void main() {
    owners=triangles[uTriangle*8+4];identityDepth=vec4(triangles[uTriangle*8+5].xy,linearDepth,1.0);
}
)GLSL";

inline constexpr unsigned kGratingCameraPrimitiveStride=12;
struct alignas(16) DirectTerminalReceiverPacked
{
    std::array<std::uint32_t,4> key{};
    std::array<std::int32_t,4> order{};
    std::array<Texel,10> data{};
};
static_assert(sizeof(DirectTerminalReceiverPacked)==12*sizeof(Texel));
inline bool decodeDirectTerminalReceiver(const DirectTerminalReceiverPacked& packed,
    const NativeDirectGratingTerminalReceiverInput& input,const DirectGratingTerminalOrder& emitted,
    DirectGratingTerminalReceiverRow& output)
{
    if(packed.key!=input.key||packed.order!=input.order)return false;
    for(const auto& t:packed.data)for(float v:t)if(!std::isfinite(v))return false;
    const auto* p=packed.data.data();DirectGratingTerminalReceiverRow row;row.input=input;row.emitted=emitted;
    std::uint32_t disposition=0,sample=0,projected=0;
    if(!causticHit(p,row.hit)||!exactReceiptInteger(p[5][0],UINT32_MAX,row.pixel.x)
        ||!exactReceiptInteger(p[5][1],UINT32_MAX,row.pixel.y)||!causticWord(p[5][3],3,disposition)
        ||!causticWord(p[8][0],1,projected)||!causticWord(p[8][1],2,sample)
        ||p[8][2]!=0||p[8][3]!=0||p[9]!=Texel{})return false;
    row.pixel.linearDepth=p[5][2];row.pixel.projected=projected!=0;
    row.disposition=DirectGratingTerminalReceiverRow::Disposition(disposition);
    row.rasterSample=DirectGratingTerminalReceiverRow::RasterSample(sample);row.queryWork=row.hit.work;
    if(sample==2) {
        if(p[7][3]!=1||!causticId(p[6][0],p[6][1],row.raster.object)
            ||!causticId(p[6][2],p[6][3],row.raster.material)
            ||!exactReceiptInteger(p[7][0],UINT32_MAX,row.raster.instance)
            ||!exactReceiptInteger(p[7][1],UINT32_MAX,row.raster.triangle))return false;
        row.raster.linearDepth=p[7][2];
    } else if(p[6]!=Texel{}||p[7]!=Texel{})return false;
    output=row;return true;
}
inline std::string directTerminalReceiverComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
struct Input {uvec4 key;ivec4 order;vec4 originMinimum;vec4 directionMaximum;};
struct Row {uvec4 key;ivec4 order;vec4 data[10];};
layout(std430,binding=0) readonly buffer Inputs {Input inputs[];};
layout(std430,binding=1) buffer Results {Row rows[];};
layout(std430,binding=2) readonly buffer Triangles {vec4 triangles[];};
uniform sampler2D uRasterOwners,uRasterTriangle;
uniform int uRowCount,uTriangleCount;
uniform vec4 uCamera,uRotation,uProjection;
uniform vec2 uExtent;
vec2 words(uint n) {return vec2(n&65535u,n>>16u);}
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
void main() {
    uint index=gl_GlobalInvocationID.x;if(index>=uint(uRowCount))return;
    Input rayInput=inputs[index];rows[index].key=rayInput.key;rows[index].order=rayInput.order;
    for(int i=0;i<10;++i)rows[index].data[i]=vec4(0);
    causticLastWork=0u;
    SceneClosestHit hit=sceneClosestHit(rayInput.originMinimum.xyz,rayInput.directionMaximum.xyz,
        rayInput.originMinimum.w,rayInput.directionMaximum.w,uOpaqueControl);
    rows[index].data[0]=vec4(float(hit.status),hit.distance,float(hit.entering),float(causticLastWork&65535u));
    rows[index].data[1]=vec4(hit.position,float(causticLastWork>>16u));
    rows[index].data[2]=vec4(hit.outwardNormal,0);
    rows[index].data[3]=vec4(words(hit.objectId),words(hit.materialId));
    rows[index].data[4]=vec4(words(hit.instanceId),words(hit.triangleId));
    if(hit.status!=1)return;
    rows[index].data[5].w=1.0;
    int triangle=-1;
    for(int t=0;t<uTriangleCount;++t) {
        vec4 owner=triangles[t*8+4],shape=triangles[t*8+5];
        if(id(owner.xy)==hit.objectId&&id(owner.zw)==hit.materialId
            &&uint(shape.x)==hit.instanceId&&uint(shape.y)==hit.triangleId) {triangle=t;break;}
    }
    if(triangle<0) {rows[index].data[0].x=-1.0;return;}
    vec4 shape=triangles[triangle*8+5];
    if(shape.z!=0.0||shape.w!=1.0||!hit.entering)return;
    rows[index].data[5].w=2.0;
    vec3 local=causticRotate(vec4(-uRotation.xyz,uRotation.w),hit.position-uCamera.xyz);
    float depth=-local.z;rows[index].data[5].z=depth;
    if(!causticFinite(depth)||depth<uProjection.z||depth>uProjection.w)return;
    float spread=uProjection.x<0.0?-uProjection.x:depth*uProjection.x;
    vec2 projected=vec2((local.x/(spread*uProjection.y)+1.0)*.5*uExtent.x,
        (1.0-local.y/spread)*.5*uExtent.y);
    if(any(isnan(projected))||any(isinf(projected))) {rows[index].data[0].x=-1.0;return;}
    if(any(lessThan(projected,vec2(0)))||any(greaterThanEqual(projected,uExtent)))return;
    uvec2 pixel=uvec2(projected);rows[index].data[5].xy=vec2(pixel);
    vec3 centreRay=causticCameraRay(vec2(pixel)+vec2(.5),uRotation,uProjection,uExtent);
    if(dot(hit.outwardNormal,centreRay)>=-1e-6)return;
    rows[index].data[5].w=3.0;rows[index].data[8]=vec4(1,1,0,0);
    ivec2 rasterPixel=ivec2(pixel.x,uint(uExtent.y)-1u-pixel.y);
    vec4 owner=texelFetch(uRasterOwners,rasterPixel,0),raster=texelFetch(uRasterTriangle,rasterPixel,0);
    // A background sample stays a truthful background. A different foreground
    // owner is retained without asserting visibility of the outgoing hit.
    if(raster.w==0.0&&owner==vec4(0))return;
    rows[index].data[6]=owner;rows[index].data[7]=raster;rows[index].data[8].y=2.0;
}
)GLSL";
}

inline std::string gratingCameraComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) buffer CameraReceipts {vec4 receipts[];};
layout(std430,binding=1) readonly buffer Incidents {vec4 incidents[];};
layout(std430,binding=2) readonly buffer Orders {vec4 orders[];};
uniform sampler2D uRasterOwners,uRasterTriangle;
uniform int uOrderCount,uEntrySubdivisions;
uniform vec4 uCamera,uRotation,uProjection;
uniform vec2 uExtent;
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
vec2 words(uint v) {return vec2(v&65535u,v>>16u);}
double angle(dvec3 a,dvec3 b,dvec3 c) {
    return 2.0*double(atan(float(abs(dot(a,cross(b,c)))),float(1.0+dot(a,b)+dot(b,c)+dot(c,a))));
}
double solidAngle(uvec2 pixel) {
    double halfY=double(uProjection.x),halfX=halfY*double(uProjection.y);
    double l=halfX*(2.0*double(pixel.x)/double(uExtent.x)-1.0);
    double r=halfX*(2.0*double(pixel.x+1u)/double(uExtent.x)-1.0);
    double t=halfY*(1.0-2.0*double(pixel.y)/double(uExtent.y));
    double b=halfY*(1.0-2.0*double(pixel.y+1u)/double(uExtent.y));
    dvec3 a=normalize(dvec3(l,b,1)),bb=normalize(dvec3(r,b,1));
    dvec3 c=normalize(dvec3(r,t,1)),d=normalize(dvec3(l,t,1));
    return angle(a,bb,c)+angle(a,c,d);
}
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uOrderCount))return;
    int dst=int(at)*12;
    for(int i=0;i<12;++i)receipts[dst+i]=vec4(0);
    int o=int(at)*4;float power=orders[o+1].w;if(power<=0.0)return;
    int locator=int(orders[o].x),event=locator*24;
    vec3 outgoing=orders[o+1].xyz,normal=incidents[event+20].xyz;
    receipts[dst]=vec4(float(locator),incidents[event].y,orders[o].y,3.0);
    receipts[dst+1]=vec4(-1,-1,0,0);receipts[dst+3].x=-1.0;
    vec3 ray=-outgoing,local=causticRotate(vec4(-uRotation.xyz,uRotation.w),ray);
    float backward=-local.z,halfY=uProjection.x,halfX=halfY*uProjection.y;
    // A parallel (orthographic) camera receives no discrete order direction.
    if(uProjection.x<0.0||backward<=0.0||local.x < -halfX*backward||local.x>=halfX*backward
        ||local.y>halfY*backward||local.y<=-halfY*backward) {
        receipts[dst].w=2.0;receipts[dst+1].w=1.0;return;
    }
    uvec2 pixel=uvec2((local.x/(backward*halfX)+1.0)*0.5*uExtent.x,
        (1.0-local.y/(backward*halfY))*0.5*uExtent.y);
    receipts[dst+1].xy=vec2(pixel);
    causticLastWork=0u;
    SceneClosestHit hit=sceneClosestHit(uCamera.xyz,ray,1e-5,uProjection.w,uOpaqueControl);
    receipts[dst+2]=vec4(words(causticLastWork),0,0);
    receipts[dst+3]=vec4(float(hit.status),hit.distance,float(hit.entering),float(causticLastWork&65535u));
    receipts[dst+4]=vec4(hit.position,float(causticLastWork>>16u));receipts[dst+5]=vec4(hit.outwardNormal,0);
    receipts[dst+6]=vec4(words(hit.objectId),words(hit.materialId));
    receipts[dst+7]=vec4(words(hit.instanceId),words(hit.triangleId));
    ivec2 rasterPixel=ivec2(pixel.x,uint(uExtent.y)-1u-pixel.y);
    vec4 owner=texelFetch(uRasterOwners,rasterPixel,0),identity=texelFetch(uRasterTriangle,rasterPixel,0);
    receipts[dst+8]=owner;receipts[dst+9]=vec4(identity.xy,identity.z,identity.w);
    if(hit.status!=1||!hit.entering||identity.w!=1.0||owner!=receipts[dst+6]
        ||identity.x!=float(hit.instanceId)||identity.y!=float(hit.triangleId))return;
    float depth=-causticRotate(vec4(-uRotation.xyz,uRotation.w),hit.position-uCamera.xyz).z;
    vec3 centreRay=causticCameraRay(vec2(pixel)+vec2(0.5),uRotation,uProjection,uExtent);
    float centreDenominator=dot(centreRay,hit.outwardNormal);
    float centreDistance=dot(hit.position-uCamera.xyz,hit.outwardNormal)/centreDenominator;
    float centreDepth=-causticRotate(vec4(-uRotation.xyz,uRotation.w),centreRay*centreDistance).z;
    if(depth<uProjection.z||depth>uProjection.w||centreDenominator>=-1e-6||!causticFinite(centreDepth)
        ||centreDepth<uProjection.z||centreDepth>uProjection.w||abs(centreDepth-identity.z)>max(1e-5,abs(centreDepth)*2e-4))return;
    bool visible=owner==incidents[event+13]&&identity.x==incidents[event+14].x&&identity.y==incidents[event+14].y;
    float denominator=dot(ray,normal),target=dot(incidents[event+12].xyz-uCamera.xyz,normal)/denominator;
    float hitBias=max(1e-5,2e-6*max(max(abs(hit.position.x),abs(hit.position.y)),abs(hit.position.z)));
    if(denominator>=-1e-6||!causticFinite(target)||target<=0.0||hit.distance>target+8.0*hitBias)return;
    double radiance=0.0;
    if(visible) {
        vec3 du=vec3(incidents[event+22].zw,incidents[event+23].x),dv=incidents[event+23].yzw;
        // The admitted barycentric cells have entryArea/|e1 cross e2|
        // equal to 1/(2*n*n). This is the same receiving Jacobian, without
        // a second triangle search or an additional visibility query.
        double n=double(uEntrySubdivisions);
        double area=length(cross(dvec3(du),dvec3(dv)))/(2.0*n*n);
        double cosine=dot(dvec3(outgoing),dvec3(normal)),omega=solidAngle(pixel);
        if(area<=0.0||cosine<=0.0||omega<=0.0)return;
        // Native order power already includes substrate and profile fractions.
        radiance=double(power)/area/cosine/omega;
        if(isnan(radiance)||isinf(radiance)||radiance<0.0)return;
    }
    receipts[dst].w=visible?0.0:1.0;receipts[dst+1].zw=vec2(float(radiance),1.0);
}
)GLSL";
}
inline constexpr const char* gratingCameraGeometryFragmentSource=R"GLSL(#version 430 core
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
uniform int uTriangle;
in float linearDepth;
layout(location=1) out float depth;
layout(location=2) out vec4 normal;
layout(location=5) out uint materialId;
layout(location=6) out uint objectId;
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
void main() {
    depth=linearDepth;normal=vec4(triangles[uTriangle*8+3].xyz,1);
    objectId=id(triangles[uTriangle*8+4].xy);materialId=id(triangles[uTriangle*8+4].zw);
}
)GLSL";
inline std::string gratingReceiverResolveSource()
{
    return std::string("#version 430 core\n")+nativeMathSource()+R"GLSL(
layout(std430,binding=2) readonly buffer SpectrumLow {uint low[];};
layout(std430,binding=6) readonly buffer SpectrumHigh {uint high[];};
uniform vec2 uExtent;
uniform vec4 uCie[8];
layout(location=0) out vec4 color;
void main() {
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint first=((uint(uExtent.y)-1u-pixel.y)*uint(uExtent.x)+pixel.x)*4u;
    vec3 xyz=vec3(0);
    for(int band=0;band<8;++band)xyz+=causticBandXyz(uintBitsToFloat(band<4?low[first+uint(band)]:high[first+uint(band-4)]),uCie[band]);
    color=vec4(causticXyzToSceneLinear(xyz),1);
}
)GLSL";
}
inline constexpr const char* gratingCameraKeySource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer CameraReceipts {vec4 receipts[];};
layout(std430,binding=5) buffer Keys {uvec4 keys[];};
uniform int uOrderCount,uKeyCount;
uniform vec2 uExtent;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uvec4 key=uvec4(0xffffffffu);
    if(at<uint(uOrderCount)) {
        uint r=at*12u;vec4 header=receipts[r],target=receipts[r+1u];
        if(header.w==0.0&&target.w==1.0&&target.z>0.0)
            key=uvec4(uint(target.y)*uint(uExtent.x)+uint(target.x),uint(header.x),uint(header.z),at);
    }
    keys[at]=key;
}
)GLSL";
inline constexpr const char* gratingCameraReduceSource=R"GLSL(#version 430 core
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer CameraReceipts {vec4 receipts[];};
layout(std430,binding=2) buffer SpectrumLow {uint low[];};
layout(std430,binding=6) buffer SpectrumHigh {uint high[];};
layout(std430,binding=5) readonly buffer Keys {uvec4 keys[];};
uniform int uKeyCount;
void main() {
    uint at=gl_GlobalInvocationID.x;if(at>=uint(uKeyCount))return;
    uint pixel=keys[at].x;
    if(pixel==0xffffffffu||(at>0u&&keys[at-1u].x==pixel))return;
    float sum[8];for(int band=0;band<8;++band)sum[band]=0.0;
    // Reuse the released sorted segment-leader reduction order. Each pixel
    // has one writer and each band follows native incident/signed-order keys.
    for(uint next=at;next<uint(uKeyCount)&&keys[next].x==pixel;++next) {
        uint r=keys[next].w*12u;
        sum[int(receipts[r].y)]+=receipts[r+1u].z;
    }
    for(int band=0;band<4;++band) {
        low[pixel*4u+uint(band)]=floatBitsToUint(sum[band]);
        high[pixel*4u+uint(band)]=floatBitsToUint(sum[band+4]);
    }
}
)GLSL";
inline std::string gratingCellTransportComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Incidents {vec4 incidents[];};
layout(std430,binding=1) readonly buffer SourceCells {vec4 cells[];};
layout(std430,binding=2) buffer CellTraces {vec4 traces[];};
layout(std430,binding=3) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=4) readonly buffer Triangles {vec4 triangles[];};
layout(std430,binding=5) readonly buffer Locators {uint locators[];};
uniform int uRecordCount,uMaximumInterfaces,uRecordStride,uEntryCount,uTriangleCount;
uniform int uGratingHistoryBase,uGratingHistoryWords;
uniform vec4 uSource,uIor[2];
uint id(vec2 v) {return uint(v.x)|(uint(v.y)<<16u);}
// Interface tokens: 32 inline bytes, then the incident's reflection history
// (texel 1 words 0-1, then the per-incident tail at uGratingHistoryBase).
int cellToken(int event,int locator,int depth,int interfaces) {
    if(depth>=interfaces)return 0;
    if(depth<32)return int(incidents[event+2+depth/4][depth%4]);
    int w=depth/32;vec4 t;bool low;
    if(w<2) {t=incidents[event+1];low=w==0;}
    else {int j=w-2;t=incidents[uGratingHistoryBase+locator*((uGratingHistoryWords+1)/2)+j/2];low=(j%2)==0;}
    uint word=low?id(t.xy):id(t.zw);
    return ((word>>uint(depth%32))&1u)!=0u?1:65;
}
vec2 words(uint v) {return vec2(v&65535u,v>>16u);}
float rayBias(vec3 p) {return max(1e-5,2e-6*max(max(abs(p.x),abs(p.y)),abs(p.z)));}
void stepHit(int at,SceneClosestHit hit,uint work) {
    traces[at+2]=vec4(float(hit.status),hit.distance,float(hit.entering),float(work&65535u));
    traces[at+3]=vec4(hit.position,float(work>>16u));traces[at+4]=vec4(hit.outwardNormal,0);
    traces[at+5]=vec4(words(hit.objectId),words(hit.materialId));
    traces[at+6]=vec4(words(hit.instanceId),words(hit.triangleId));
}
bool sameOwner(int at,SceneClosestHit hit) {
    return traces[at+5]==vec4(words(hit.objectId),words(hit.materialId))
        &&traces[at+6]==vec4(words(hit.instanceId),words(hit.triangleId));
}
bool glassHit(SceneClosestHit hit) {
    for(int t=0;t<uTriangleCount;++t) {
        vec4 owners=triangles[t*8+4],indices=triangles[t*8+5];
        if(owners==vec4(words(hit.objectId),words(hit.materialId))
            &&indices.x==float(hit.instanceId)&&indices.y==float(hit.triangleId))return indices.z==1.0;
    }return false;
}
void main() {
    uint invocation=gl_GlobalInvocationID.x;if(invocation>=uint(uRecordCount))return;
    int locator=int(locators[invocation]),base=int(invocation)*uRecordStride,event=locator*24;
    for(int i=0;i<uRecordStride;++i)traces[base+i]=vec4(0);
    int work=int(incidents[event].x),band=int(incidents[event].y),interfaces=int(incidents[event].z);
    vec4 entry=workTable[work*4],entryOwners=workTable[work*4+2],entryIndices=workTable[work*4+3];
    int entryIndex=int(entryIndices.z),cell=entryIndex*7;
    if(entryIndex<0||entryIndex>=uEntryCount||interfaces<1||interfaces>uMaximumInterfaces
        ||cells[cell+6].x!=1.0||cells[cell+4]!=entryOwners||cells[cell+5].xy!=entryIndices.xy
        ||cells[cell+5].w!=float(entryIndex)||cells[cell+3].w!=entry.w)return;
    traces[base]=vec4(1,float(invocation),float(work),float(entryIndex));
    traces[base+1]=vec4(float(band),float(interfaces),2,0);
    uint totalWork=0u;bool supported=true;
    for(int path=0;path<4;++path) {
        vec3 start=path==0?entry.xyz:cells[cell+path-1].xyz;
        vec3 origin=start+uSource.xyz*(2.0*rayBias(start)),direction=-uSource.xyz;
        bool inside=false,pathSupported=true;
        for(int depth=0;depth<=interfaces;++depth) {
            int at=base+4+(path*(uMaximumInterfaces+1)+depth)*7;
            int token=cellToken(event,locator,depth,interfaces);
            causticLastWork=0u;
            SceneClosestHit hit=sceneClosestHit(origin,direction,depth==0?0.0:1e-5,uSource.w,uOpaqueControl);
            uint workUnits=causticLastWork;totalWork+=workUnits;
            traces[at]=vec4(origin,float(depth));traces[at+1]=vec4(direction,float(token));stepHit(at,hit,workUnits);
            traces[base+3][path]=float(depth+1);
            if(hit.status<0) {pathSupported=false;break;}
            int center=base+4+depth*7;
            bool ownerMatches=path==0||sameOwner(center,hit);
            if(depth==interfaces) {
                bool endpoint=hit.status==1&&hit.entering&&!inside&&ownerMatches
                    &&vec4(words(hit.objectId),words(hit.materialId))==incidents[event+13]
                    &&float(hit.instanceId)==incidents[event+14].x&&float(hit.triangleId)==incidents[event+14].y;
                if(path==0)endpoint=endpoint&&length(hit.position-incidents[event+12].xyz)<=8.0*rayBias(hit.position)
                    &&length(origin-incidents[event+10].xyz)<=8.0*rayBias(origin)
                    &&length(direction-incidents[event+11].xyz)<=2e-4;
                pathSupported=pathSupported&&endpoint;break;
            }
            bool matches=hit.status==1&&glassHit(hit)&&ownerMatches&&inside!=hit.entering&&(token==1||token==65);
            if(depth==0)matches=matches&&hit.objectId==id(entryOwners.xy)&&hit.materialId==id(entryOwners.zw)
                &&hit.instanceId==uint(entryIndices.x)&&hit.triangleId==uint(entryIndices.y);
            if(!matches) {pathSupported=false;break;}
            vec3 normal=inside?-hit.outwardNormal:hit.outwardNormal,transmitted,reflected;bool tir;
            float ior=uIor[band/4][band%4];
            float reflection=causticFresnel(direction,normal,inside?ior:1.0,inside?1.0:ior,transmitted,reflected,tir);
            bool reflect=token==1;
            if(!causticFinite(reflection)||(reflect?reflection<=0.0:tir||reflection>=1.0)) {pathSupported=false;break;}
            float signedBias=(inside?-1.0:1.0)*(reflect?1.0:-1.0)*rayBias(hit.position);
            traces[at+4].w=signedBias;
            // Match the original producer's oriented-normal operation order.
            origin=hit.position+normal*(reflect?rayBias(hit.position):-rayBias(hit.position));
            direction=reflect?reflected:transmitted;if(!reflect)inside=!inside;
        }
        supported=supported&&pathSupported;
    }
    traces[base+1].zw=vec2(supported?1.0:2.0,1.0);
    traces[base+2]=vec4(words(totalWork),0,0);
}
)GLSL";
}
inline bool decodeGratingCellCoverage(const Texel* rows,std::size_t rowCount,
    const GratingCellCoverageCandidate& candidate,std::uint32_t leaves,std::uint32_t level,
    GratingCellCoveragePayload& output,std::string& error)
{
    const auto refuse=[&](){error="OpenGL completed native interval proof is malformed";return false;};
    if(!candidate.boundary||!candidate.boundary->binding||!candidate.boundary->binding->sourceCells
        ||!candidate.boundary->binding->sourceCells->incidentBinding||!leaves||leaves>candidate.bounds.maximumLeaves
        ||level>kGratingCellCoverageDepth||leaves!=(1u<<(2*level)))return refuse();
    const auto& binding=*candidate.boundary->binding;
    const auto& frame=binding.sourceCells->incidentBinding->frame;
    const auto sweepStride=16+candidate.bounds.maximumBlockers*2;
    const auto leafStride=3+(binding.bounds.maximumInterfaces+1)*sweepStride;
    if(rowCount!=candidate.boundary->records.size()*leaves*leafStride||(rowCount&&!rows))return refuse();
    for(std::size_t i=0;i<rowCount;++i)for(float value:rows[i])if(!std::isfinite(value))return refuse();
    GratingCellCoveragePayload payload;payload.records.resize(candidate.boundary->records.size());
    bool all=!payload.records.empty();
    for(std::size_t incident=0;incident<payload.records.size();++incident) {
        auto& record=payload.records[incident];record.incident=std::uint32_t(incident);
        record.entry=candidate.boundary->records[incident].entry;record.leaves.reserve(leaves);bool recordClear=true;
        const auto interfaces=binding.pointReadback->incidents[incident].interfaces;
        for(std::uint32_t index=0;index<leaves;++index) {
            const auto* h=rows+(incident*leaves+index)*leafStride;
            if(h[0]!=Texel{float(kGratingCellCoverageVersion),float(incident),float(record.entry),float(index)}
                ||h[1][0]!=level||h[1][1]!=interfaces+1||h[1][3]!=1||(h[1][2]!=1&&h[1][2]!=2)
                ||h[2][2]!=0||h[2][3]!=0)return refuse();
            std::uint32_t lo=0,hi=0;
            if(!exactReceiptInteger(h[2][0],65535,lo)||!exactReceiptInteger(h[2][1],65535,hi))return refuse();
            const auto leafWork=lo|(std::uint64_t(hi)<<16);std::uint64_t work=0;bool leafClear=true;
            GratingCellCoverageLeaf leaf;
            if(!gratingCellDyadicTriangle(level,index,leaf.domain))return refuse();
            leaf.sweeps.reserve(interfaces+1);
            for(std::size_t depth=0;depth<=binding.bounds.maximumInterfaces;++depth) {
                const auto* t=h+3+depth*sweepStride;
                if(depth>interfaces) {for(std::size_t i=0;i<sweepStride;++i){if(t[i]!=Texel{})return refuse();}continue;}
                GratingCellSweepEvidence sweep;std::uint32_t token=0,rule=0,disposition=0,blockers=0;
                if(t[0][0]!=depth||!exactReceiptInteger(t[0][1],65,token)||!exactReceiptInteger(t[0][2],1,rule)||rule!=1
                    ||!exactReceiptInteger(t[0][3],2,disposition)||!disposition||(t[3][3]!=0&&t[3][3]!=1)
                    ||(t[4][3]!=-1&&t[4][3]!=0&&t[4][3]!=1)||t[8][3]!=0||t[12][1]!=1||t[12][3]!=0
                    ||!exactReceiptInteger(t[12][0],std::uint32_t(candidate.bounds.maximumBlockers),blockers)
                    ||t[13]!=Texel{}||t[14]!=Texel{}||t[15]!=Texel{})return refuse();
                sweep.depth=std::uint32_t(depth);sweep.token=token;sweep.rule=rule;
                sweep.disposition=GratingCellCoverageDisposition(disposition);sweep.entering=t[3][3]==1;sweep.biasSign=int(t[4][3]);
                if(!causticId(t[1][0],t[1][1],sweep.object)||!causticId(t[1][2],t[1][3],sweep.material)
                    ||!causticId(t[2][0],t[2][1],sweep.instance)||!causticId(t[2][2],t[2][3],sweep.triangle)
                    ||!exactReceiptInteger(t[6][3],65535,lo)||!exactReceiptInteger(t[7][3],65535,hi))return refuse();
                sweep.queryWork=lo|(std::uint64_t(hi)<<16);work+=sweep.queryWork;
                sweep.direction={t[3][0],t[3][1],t[3][2]};sweep.normal={t[4][0],t[4][1],t[4][2]};
                for(unsigned a=0;a<3;++a) {sweep.originBounds[a]={t[5][a],t[6][a]};sweep.hitBounds[a]={t[7][a],t[8][a]};}
                sweep.roundingExpansion=t[5][3];sweep.distance={t[9][0],t[9][1]};sweep.biasMagnitude={t[9][2],t[9][3]};
                sweep.signedBias={t[10][0],t[10][1]};sweep.barycentricBounds={{{t[10][2],t[10][3]},{t[11][0],t[11][1]},{t[11][2],t[11][3]}}};
                if(disposition==1) {
                    std::uint32_t target=0;
                    if(!exactReceiptInteger(t[12][2],std::uint32_t(frame.triangles.size()),target)||target>=frame.triangles.size()
                        ||blockers!=candidate.bounds.maximumBlockers)return refuse();
                    sweep.blockers.reserve(blockers);
                    for(std::uint32_t b=0;b<blockers;++b) {
                        const auto* p=t+16+b*2;GratingCellBlockerSeparation separation;
                        if(!exactReceiptInteger(p[0][0],std::uint32_t(frame.triangles.size()),separation.triangle)
                            ||separation.triangle>=frame.triangles.size()||separation.triangle==target)return refuse();
                        separation.axis={p[0][1],p[0][2],p[0][3]};
                        separation.sweptProjection={p[1][0],p[1][1]};separation.triangleProjection={p[1][2],p[1][3]};
                        sweep.blockers.push_back(separation);
                    }
                } else {
                    leafClear=false;
                    if(blockers||t[12][2]!=-1)return refuse();
                    for(std::size_t b=16;b<sweepStride;++b)if(t[b]!=Texel{})return refuse();
                }
                leaf.sweeps.push_back(std::move(sweep));
            }
            if(work!=leafWork||(h[1][2]==1)!=leafClear)return refuse();
            payload.queryWork+=work;recordClear=recordClear&&leafClear;record.leaves.push_back(std::move(leaf));
        }
        record.disposition=recordClear?GratingCellCoverageDisposition::InteriorCertified:GratingCellCoverageDisposition::Unsupported;
        all=all&&recordClear;
    }
    payload.disposition=all?GratingCellCoverageDisposition::InteriorCertified:GratingCellCoverageDisposition::Unsupported;
    if(!validateGratingCellCoverageShape(candidate,payload,error))return false;
    output=std::move(payload);error.clear();return true;
}

// Each leaf has a 48-byte header, then fixed 256-byte sweep headers and
// 32-byte all-triangle separation rows. These are private proof records.
inline std::string gratingCellCoverageComputeSource()
{
    return std::string("#version 430 core\n")+querySource()+nativeMathSource()+R"GLSL(
layout(local_size_x=32) in;
layout(std430,binding=0) readonly buffer Incidents {vec4 incidents[];};
layout(std430,binding=1) readonly buffer SourceCells {vec4 cells[];};
layout(std430,binding=2) readonly buffer Boundaries {vec4 boundaries[];};
layout(std430,binding=3) buffer Coverage {vec4 coverage[];};
layout(std430,binding=4) readonly buffer Work {vec4 workTable[];};
layout(std430,binding=5) readonly buffer Triangles {vec4 triangles[];};
layout(std430,binding=6) readonly buffer Locators {uint locators[];};
uniform int uRecordCount,uLeaves,uLevel,uLeafStride,uSweepStride,uBoundaryStride,uMaximumInterfaces,uTriangleCount,uQueryAllowance;
uniform vec4 uSource,uIor[2];
bool arithmetic=true;
vec2 words(uint n) {return vec2(n&65535u,n>>16u);}
bool normalFloat(float x) {return !isnan(x)&&!isinf(x)&&(x==0.0||abs(x)>=uintBitsToFloat(0x00800000u));}
float outward(float x,bool upper) {
    if(!normalFloat(x)) {arithmetic=false;return 0.0;}
    if(x==0.0)return upper?uintBitsToFloat(0x00800000u):-uintBitsToFloat(0x00800000u);
    uint bits=floatBitsToUint(x);bits+=((x>0.0)==upper)?1u:0xffffffffu;
    float y=uintBitsToFloat(bits);if(!normalFloat(y))arithmetic=false;return normalFloat(y)?y:0.0;
}
vec2 addI(vec2 a,vec2 b) {
    if(a==vec2(0))return b;if(b==vec2(0))return a;
    if(a.x==a.y&&b.x==b.y&&a.x==-b.x)return vec2(0);
    precise float lo=a.x+b.x;precise float hi=a.y+b.y;
    return vec2(outward(lo,false),outward(hi,true));
}
vec2 negI(vec2 a) {return -a.yx;}
vec2 subI(vec2 a,vec2 b) {return addI(a,negI(b));}
vec2 mulI(vec2 a,vec2 b) {
    if(a==vec2(0)||b==vec2(0))return vec2(0);
    if(a==vec2(1))return b;if(b==vec2(1))return a;
    if(a==vec2(-1))return negI(b);if(b==vec2(-1))return negI(a);
    precise float p0=a.x*b.x;precise float p1=a.x*b.y;precise float p2=a.y*b.x;precise float p3=a.y*b.y;
    if((p0==0.0&&a.x!=0.0&&b.x!=0.0)||(p1==0.0&&a.x!=0.0&&b.y!=0.0)
        ||(p2==0.0&&a.y!=0.0&&b.x!=0.0)||(p3==0.0&&a.y!=0.0&&b.y!=0.0))arithmetic=false;
    if(!normalFloat(p0)||!normalFloat(p1)||!normalFloat(p2)||!normalFloat(p3))arithmetic=false;
    return vec2(outward(min(min(p0,p1),min(p2,p3)),false),outward(max(max(p0,p1),max(p2,p3)),true));
}
vec2 divI(vec2 a,vec2 b) {
    // GLSL 4.30 section 4.7.1 bounds reciprocal error only for positive
    // normal denominators. Exact interval negation preserves a/b when both
    // operands change sign, including entering-plane negative denominators.
    if(b.x<=0.0&&b.y>=0.0) {arithmetic=false;return vec2(0);}
    if(b.y<0.0) {a=negI(a);b=negI(b);}
    // Six outward neighbors cover 2.5 ULP, also across a binade boundary.
    if(b.x<uintBitsToFloat(0x00800000u)||b.y>uintBitsToFloat(0x7e800000u))
        {arithmetic=false;return vec2(0);}
    precise float lo=1.0/b.y;precise float hi=1.0/b.x;
    for(int step=0;step<6;++step) {lo=outward(lo,false);hi=outward(hi,true);}
    return mulI(a,vec2(lo,hi));
}
struct Box {vec3 lo;vec3 hi;};
Box pointB(vec3 p) {for(int k=0;k<3;++k)if(!normalFloat(p[k]))arithmetic=false;return Box(p,p);}
Box addB(Box a,Box b) {
    Box c;for(int k=0;k<3;++k) {vec2 v=addI(vec2(a.lo[k],a.hi[k]),vec2(b.lo[k],b.hi[k]));c.lo[k]=v.x;c.hi[k]=v.y;}return c;
}
Box negB(Box a) {return Box(-a.hi,-a.lo);}
Box subB(Box a,Box b) {return addB(a,negB(b));}
Box scaleB(Box a,vec2 factor) {
    Box c;for(int k=0;k<3;++k) {vec2 v=mulI(vec2(a.lo[k],a.hi[k]),factor);c.lo[k]=v.x;c.hi[k]=v.y;}return c;
}
Box hullB(Box a,Box b) {return Box(min(a.lo,b.lo),max(a.hi,b.hi));}
vec2 dotB(Box a,Box b) {
    vec2 v=vec2(0);for(int k=0;k<3;++k)v=addI(v,mulI(vec2(a.lo[k],a.hi[k]),vec2(b.lo[k],b.hi[k])));return v;
}
Box crossB(Box a,Box b) {
    Box c;for(int k=0;k<3;++k) {int p=(k+1)%3,q=(k+2)%3;
        vec2 v=subI(mulI(vec2(a.lo[p],a.hi[p]),vec2(b.lo[q],b.hi[q])),mulI(vec2(a.lo[q],a.hi[q]),vec2(b.lo[p],b.hi[p])));
        c.lo[k]=v.x;c.hi[k]=v.y;}return c;
}
vec2 biasI(Box p) {
    float lower=0.0,upper=0.0;
    for(int k=0;k<3;++k) {
        float lo=p.lo[k]<=0.0&&p.hi[k]>=0.0?0.0:min(abs(p.lo[k]),abs(p.hi[k]));
        lower=max(lower,lo);upper=max(upper,max(abs(p.lo[k]),abs(p.hi[k])));
    }
    vec2 v=mulI(vec2(lower,upper),vec2(2e-6));return max(vec2(1e-5),v);
}
vec2 triangleProjection(int triangle,vec3 axis) {
    vec2 p=dotB(pointB(triangles[triangle*8].xyz),pointB(axis));
    for(int c=1;c<3;++c) {vec2 q=dotB(pointB(triangles[triangle*8+c].xyz),pointB(axis));p=vec2(min(p.x,q.x),max(p.y,q.y));}return p;
}
void domain(int level,uint code,out ivec2 a,out ivec2 b,out ivec2 c) {
    a=ivec2(0);b=ivec2(256,0);c=ivec2(0,256);
    for(int depth=0;depth<level;++depth) {
        ivec2 ab=(a+b)/2,bc=(b+c)/2,ca=(c+a)/2;
        uint child=(code>>uint(2*(level-depth-1)))&3u;
        if(child==0u) {b=ab;c=ca;}else if(child==1u) {a=ab;c=bc;}
        else if(child==2u) {a=ca;b=bc;}else {a=ab;b=bc;c=ca;}
    }
}
Box sourceVertex(int cell,ivec2 q) {
    return addB(addB(pointB(cells[cell].xyz),scaleB(subB(pointB(cells[cell+1].xyz),pointB(cells[cell].xyz)),vec2(float(q.x)/256.0))),
        scaleB(subB(pointB(cells[cell+2].xyz),pointB(cells[cell].xyz)),vec2(float(q.y)/256.0)));
}
void main() {
    uint invocation=gl_GlobalInvocationID.x;
    if(invocation>=uint(uRecordCount*uLeaves))return;
    int incident=int(invocation)/uLeaves,leaf=int(invocation)%uLeaves,base=int(invocation)*uLeafStride;
    for(int i=0;i<uLeafStride;++i)coverage[base+i]=vec4(0);
    int event=int(locators[incident])*24,boundary=incident*uBoundaryStride;
    int work=int(incidents[event].x),band=int(incidents[event].y),interfaces=int(incidents[event].z);
    int entry=int(workTable[work*4+3].z),cell=entry*7;
    if(boundaries[boundary]!=vec4(1,float(incident),float(work),float(entry))
        ||boundaries[boundary+1].xy!=vec2(float(band),float(interfaces))||boundaries[boundary+1].w!=1.0
        ||cells[cell+6].x!=1.0||cells[cell+5].w!=float(entry))return;
    coverage[base]=vec4(1,float(incident),float(entry),float(leaf));
    coverage[base+1]=vec4(float(uLevel),float(interfaces+1),2,0);
    ivec2 a,b,c;domain(uLevel,uint(leaf),a,b,c);
    arithmetic=true;
    Box start=hullB(hullB(sourceVertex(cell,a),sourceVertex(cell,b)),sourceVertex(cell,c));
    Box origin=addB(start,scaleB(pointB(uSource.xyz),mulI(biasI(start),vec2(2))));
    bool continuing=arithmetic&&boundaries[boundary+1].z==1.0,recordClear=continuing;
    uint totalWork=0u;
    for(int depth=0;depth<=interfaces;++depth) {
        int center=boundary+4+depth*7,at=base+3+depth*uSweepStride;
        vec4 owners=boundaries[center+5],indices=boundaries[center+6];
        vec3 direction=boundaries[center+1].xyz,normal=boundaries[center+4].xyz;
        int token=int(boundaries[center+1].w);float sign=depth<interfaces?(boundaries[center+4].w>0.0?1.0:-1.0):0.0;
        coverage[at]=vec4(float(depth),float(token),1,2);
        coverage[at+1]=owners;coverage[at+2]=indices;
        coverage[at+3]=vec4(direction,boundaries[center+2].z);coverage[at+4]=vec4(normal,sign);
        coverage[at+12]=vec4(0,1,-1,0);
        if(!continuing) {recordClear=false;continue;}
        arithmetic=true;int target=-1;uint workUnits=0u;
        for(int t=0;t<uTriangleCount;++t) {
            ++workUnits;vec4 ids=triangles[t*8+5];
            if(triangles[t*8+4]==owners&&vec4(words(uint(ids.x)),words(uint(ids.y)))==indices) {target=t;break;}
        }
        bool clear=target>=0&&workUnits+uint(uTriangleCount-1)<=uint(uQueryAllowance);
        Box hit=pointB(vec3(0));vec2 distance=vec2(0),bias=vec2(0),signedBias=vec2(0),bary[3];
        bary[0]=bary[1]=bary[2]=vec2(0);float expansion=0.0;
        if(clear) {
            Box p=pointB(triangles[target*8].xyz),e1=subB(pointB(triangles[target*8+1].xyz),p),e2=subB(pointB(triangles[target*8+2].xyz),p);
            Box n=crossB(e1,e2),ray=pointB(direction);vec2 denominator=dotB(ray,n);
            bool entering=coverage[at+3].w==1.0;
            clear=!(denominator.x<=0.0&&denominator.y>=0.0)
                &&(entering?denominator.y<0.0:denominator.x>0.0);
            // Enclose both the true retained plane and the released triangle
            // arithmetic, including its independent barycentric/distance path.
            vec2 plane=divI(dotB(subB(p,origin),n),denominator);
            Box t=subB(origin,p),crossRay=crossB(ray,e2),q=crossB(t,e1);vec2 det=dotB(e1,crossRay);
            vec2 nativeDistance=divI(dotB(e2,q),det);
            distance=vec2(min(plane.x,nativeDistance.x),max(plane.y,nativeDistance.y));
            hit=addB(origin,scaleB(ray,distance));
            bary[1]=divI(dotB(t,crossRay),det);bary[2]=divI(dotB(ray,q),det);
            bary[0]=subI(subI(vec2(1),bary[1]),bary[2]);
            clear=clear&&distance.x>=(depth==0?0.0:1e-5)&&distance.y<=uSource.w;
            for(int k=0;k<3;++k)clear=clear&&bary[k].x>0.0&&bary[k].y<1.0;
            if(depth<interfaces) {
                // Entering state fixes the outward-normal orientation.
                bool inside=!entering;vec3 oriented=inside?-normal:normal;
                vec3 transmitted,reflected;bool tir;float ior=uIor[band/4][band%4];
                float reflection=causticFresnel(direction,oriented,inside?ior:1.0,inside?1.0:ior,transmitted,reflected,tir);
                vec3 next=token==1?reflected:transmitted;
                clear=clear&&triangles[target*8+5].z==1.0&&(token==1||token==65)
                    &&causticFinite(reflection)&&(token==1?reflection>0.0:!tir&&reflection<1.0)
                    &&next==boundaries[center+8].xyz;
                bias=biasI(hit);signedBias=sign>0.0?bias:negI(bias);
            }
            Box swept=hullB(origin,hit);
            precise float magnitude=max(1.0,max(max(max(abs(swept.lo.x),abs(swept.hi.x)),max(abs(swept.lo.y),abs(swept.hi.y))),max(abs(swept.lo.z),abs(swept.hi.z))));
            precise float margin=magnitude*4.76837158203125e-7;
            expansion=outward(margin,true);
            int blocker=0;
            for(int triangle=0;triangle<uTriangleCount;++triangle) {
                if(triangle==target)continue;++workUnits;bool separated=false;
                vec3 axis=vec3(0);vec2 sweptProjection=vec2(0),projection=vec2(0);
                for(int choice=0;choice<4;++choice) {
                    vec3 trial=choice==3?triangles[triangle*8+3].xyz:vec3(choice==0?1.0:0.0,choice==1?1.0:0.0,choice==2?1.0:0.0);
                    if(abs(dot(trial,trial)-1.0)>1e-4)continue;
                    vec2 bundle=dotB(swept,pointB(trial)),surface=triangleProjection(triangle,trial);
                    precise float gap0=surface.x-bundle.y;precise float gap1=bundle.x-surface.y;
                    if(arithmetic&&max(outward(gap0,false),outward(gap1,false))>expansion) {
                        axis=trial;sweptProjection=bundle;projection=surface;separated=true;break;
                    }
                }
                int row=at+16+blocker*2;
                coverage[row]=vec4(float(triangle),axis);coverage[row+1]=vec4(sweptProjection,projection);
                ++blocker;clear=clear&&separated;
            }
            clear=clear&&arithmetic;
            if(clear) {
                coverage[at].w=1.0;
                coverage[at+5]=vec4(origin.lo,expansion);coverage[at+6]=vec4(origin.hi,float(workUnits&65535u));
                coverage[at+7]=vec4(hit.lo,float(workUnits>>16u));coverage[at+8]=vec4(hit.hi,0);
                coverage[at+9]=vec4(distance,bias);coverage[at+10]=vec4(signedBias,bary[0]);coverage[at+11]=vec4(bary[1],bary[2]);
                coverage[at+12]=vec4(float(uTriangleCount-1),1,float(target),0);
                origin=addB(hit,scaleB(pointB(normal),signedBias));continuing=arithmetic;
            }
        }
        if(!clear) {
            for(int k=16;k<uSweepStride;++k)coverage[at+k]=vec4(0);
            coverage[at+6].w=float(workUnits&65535u);coverage[at+7].w=float(workUnits>>16u);
            continuing=false;recordClear=false;
        }
        totalWork+=workUnits;
    }
    coverage[base+1].zw=vec2(recordClear?1.0:2.0,1.0);coverage[base+2]=vec4(words(totalWork),0,0);
}
)GLSL";
}

// Each native pair retains its full constructor and the fixed-corner records.
// Unused corner slots are zero. Wide coefficients are constructed natively.
static_assert(cameraraycoefficients::kRequestWords==57&&cameraraycoefficients::kResultWords==557
    &&camerapixelsigns::kRequestWords==65&&camerapixelsigns::kResultWords==73);
inline constexpr std::uint32_t kCameraDiagonalPairWords=557+13*73;
inline constexpr std::uint32_t kCameraDiagonalHeaderWords=16;
inline constexpr std::uint32_t kCameraDiagonalTriangleWords=20;
using CameraDiagonalTriangle=std::array<std::uint32_t,kCameraDiagonalTriangleWords>;
using CameraDiagonalPixel=std::array<std::uint32_t,8>;
inline std::string cameraDiagonalComputeSource()
{
    return std::string("#version 430 core\nlayout(local_size_x=1) in;\n")
        +camerapixelsigns::languageSource(false)+R"DIAGONAL(
layout(std430,binding=0) readonly buffer Inputs {uint inputs[];};
layout(std430,binding=1) buffer Certificates {uint certificates[];};
uniform uint uPixelCount,uTriangleCount,uGeneration;
const uint pairWords=1506u;
uint tri(uint t,uint w){return inputs[23u+8u*uPixelCount+t*20u+w];}
uint pix(uint p,uint w){return inputs[23u+p*8u+w];}
bool endpoint(uint t,uint a,uint s,uint b){
    if(tri(t,5u+a)!=tri(s,5u+b))return false;
    for(uint k=0u;k<3u;k++)if(tri(t,8u+a*3u+k)!=tri(s,8u+b*3u+k))return false;
    return true;
}
void request(uint p,uint a,uint b,out uint q[57]){
    for(uint k=0u;k<23u;k++)q[k]=inputs[k];q[8]=pix(p,0u);q[9]=pix(p,1u);
    for(uint k=0u;k<17u;k++){q[23u+k]=tri(a,k);q[40u+k]=tri(b,k);}
}
bool uniformSign(uint r[73],uint s){
    return r[0]==1u&&r[4]==s&&r[5]==s&&r[6]==s&&r[7]==s;
}
bool corners(uint q[57],uint record[557],uint form,uint base,out uint r[73]){
    uint a[65];for(int k=0;k<57;k++)a[k]=q[k];
    a[57]=1u;a[58]=CP_REQUEST_TAG;a[59]=0u;a[60]=CP_CHARGE;
    a[61]=form;a[62]=CP_WIRE;a[63]=0u;a[64]=0u;
    cpEvaluate(record,a,r);
    for(uint k=0u;k<73u;k++)certificates[base+557u+(form-4u)*73u+k]=r[k];
    return r[0]==1u;
}
bool reversedForm(uint record[557],uint a,uint b){
    uint x=61u+a*28u,y=61u+b*28u;
    if(record[x]!=record[y])return false;
    for(uint k=0u;k<3u;k++){
        uint sx=record[x+1u+k*9u],sy=record[y+1u+k*9u];
        if(sy!=(sx==0u?0u:3u-sx))return false;
        for(uint l=1u;l<9u;l++)if(record[x+1u+k*9u+l]!=record[y+1u+k*9u+l])return false;
    }
    return true;
}
void main(){
    uint p=gl_GlobalInvocationID.x;if(p>=uPixelCount)return;
    uint first=p*(16u+uTriangleCount*pairWords);
    for(uint k=0u;k<16u+uTriangleCount*pairWords;k++)certificates[first+k]=0u;
    for(uint k=0u;k<8u;k++)certificates[first+8u+k]=pix(p,k);
    uint a=pix(p,6u);if(pix(p,7u)!=uGeneration||a>=uTriangleCount||tri(a,18u)!=1u||tri(a,19u)!=0u)return;
    uint b=uTriangleCount,edgeA=0u,edgeB=0u;bool ambiguous=false;
    for(uint t=0u;t<uTriangleCount;t++){
        if(t==a||tri(t,1u)!=pix(p,2u)||tri(t,2u)!=pix(p,3u)||tri(t,3u)!=pix(p,4u)
            ||tri(t,18u)!=1u||tri(t,19u)!=0u)continue;
        for(uint ea=0u;ea<3u;ea++)for(uint eb=0u;eb<3u;eb++)
            if(endpoint(a,ea,t,(eb+1u)%3u)&&endpoint(a,(ea+1u)%3u,t,eb)){
                if(b!=uTriangleCount)ambiguous=true;
                else{b=t;edgeA=ea;edgeB=eb;}
            }
    }
    if(b==uTriangleCount||ambiguous||tri(a,17u)!=tri(b,17u)||tri(a,4u)==tri(b,4u))return;
    uint q[57],record[557],r[73],s[73];request(p,a,b,q);ccEvaluate(q,record);
    uint base=first+16u+b*pairWords;
    for(uint k=0u;k<557u;k++)certificates[base+k]=record[k];
    if(record[0]!=1u)return;
    uint horizon=0u;
    for(uint form=4u;form<=16u;form++)if(!corners(q,record,form,base,r))return;
    for(uint form=4u;form<=16u;form++){
        for(uint k=0u;k<73u;k++)r[k]=certificates[base+557u+(form-4u)*73u+k];
        if(form==7u){horizon=r[4];if(horizon==0u||!uniformSign(r,horizon))return;}
    }
    if(record[538]!=horizon||record[548]!=horizon)return;
    for(uint form=4u;form<16u;form++){
        for(uint k=0u;k<73u;k++)r[k]=certificates[base+557u+(form-4u)*73u+k];
        if(form==4u+edgeA||form==10u+edgeB)continue;
        if(!uniformSign(r,horizon))return;
    }
    for(uint k=0u;k<73u;k++)r[k]=certificates[base+557u+12u*73u+k];
    if(!uniformSign(r,0u)||!reversedForm(record,4u+edgeA,10u+edgeB))return;
    for(uint k=0u;k<73u;k++){
        r[k]=certificates[base+557u+edgeA*73u+k];
        s[k]=certificates[base+557u+(6u+edgeB)*73u+k];
    }
    uint diagonal=2u;
    if(r[4]==0u&&r[7]==0u&&r[5]!=0u&&r[6]==3u-r[5])diagonal=0u;
    if(r[5]==0u&&r[6]==0u&&r[4]!=0u&&r[7]==3u-r[4])diagonal=1u;
    if(diagonal==2u)return;
    for(uint k=4u;k<8u;k++)if(s[k]!=(r[k]==0u?0u:3u-r[k]))return;
    uint side=diagonal==0u?r[5]:r[4];
    uint primary[557];for(uint k=0u;k<557u;k++)primary[k]=record[k];
    for(uint t=0u;t<uTriangleCount;t++){
        if(t==a||t==b)continue;
        request(p,a,t,q);ccEvaluate(q,record);base=first+16u+t*pairWords;
        for(uint k=0u;k<557u;k++)certificates[base+k]=record[k];
        if(record[0]!=1u)return;
        for(uint k=0u;k<10u;k++)if(record[537u+k]!=primary[537u+k])return;
        for(uint k=0u;k<6u*28u;k++)if(record[61u+4u*28u+k]!=primary[61u+4u*28u+k])return;
        bool excluded=false;uint strict[3];
        for(uint e=0u;e<3u;e++){
            if(!corners(q,record,10u+e,base,r))return;
            strict[e]=uniformSign(r,1u)?1u:uniformSign(r,2u)?2u:0u;
        }
        if(!corners(q,record,13u,base,r))return;
        uint other=uniformSign(r,1u)?1u:uniformSign(r,2u)?2u:0u;
        // Two strict opposite edges exclude a finite hit even on an exactly
        // known horizon crossing. No division or inferred horizon orientation.
        for(uint x=0u;x<3u;x++)for(uint y=x+1u;y<3u;y++)
            if(strict[x]!=0u&&strict[y]==3u-strict[x])excluded=true;
        if(other!=0u){
            if(record[548]==3u-other)excluded=true; // Entire plane is backward.
            for(uint e=0u;e<3u;e++)if(strict[e]==3u-other)excluded=true;
        }
        if(!corners(q,record,16u,base,r))return;
        bool behind=other!=0u&&record[548]==other&&uniformSign(r,other==horizon?2u:1u);
        if(!excluded&&!behind)return;
    }
    // The exact equal-depth seam belongs to the released lexicographic key.
    // Both triangles share object-table and instance; triangle id decides it.
    certificates[first]=1u;certificates[first+1u]=a;certificates[first+2u]=b;
    certificates[first+3u]=edgeA;certificates[first+4u]=edgeB;certificates[first+5u]=diagonal;
    certificates[first+6u]=side==horizon?1u:2u;
    certificates[first+7u]=tri(a,4u)<tri(b,4u)?a:b;
}
)DIAGONAL";
}

} // namespace glasscaustics::opengl
} // namespace arbitgpu
