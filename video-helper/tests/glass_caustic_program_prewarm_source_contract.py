#!/usr/bin/env python3
# Contract for the OpenGL glass caustic programs. The renderer starts them before the
# scene preparation, and Preview frames stay pending until the driver reports every link
# complete, so no compile or link wait runs inside fixture-caustic-submit. The programs
# live in a per-context cache with the disk program binary cache behind it, so a scene
# re-prepare reuses them.
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
header = (root / "src/gpu_backend/fixture_glass_caustics_opengl.h").read_text()
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
cmake = (root / "CMakeLists.txt").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
metal_header = (root / "src/gpu_backend/metal_glass_caustic_consumer.h").read_text()
grating = (root / "src/gpu_backend/fixture_glass_grating_incident.h").read_text()

errors = []


def require(text, needle, label):
    if needle not in text:
        errors.append(label + " is missing " + needle)


def forbid(text, needle, label):
    if needle in text:
        errors.append(label + " must not contain " + needle)


def region(text, begin, end, label):
    start = text.find(begin)
    if start < 0:
        errors.append(label + " is missing " + begin)
        return ""
    stop = text.find(end, start + len(begin))
    if stop < 0:
        errors.append(label + " is missing " + end)
        return ""
    return text[start:stop]


# The caustic render takes its programs from the context set. It links nothing itself.
render = region(backend, "NativeFixtureSceneSubmission renderOpenGlGlassCausticScene(",
                "bool readOpenGlGlassCausticReceiptForFixture(", "caustic render")
for needle in ("linkCausticProgram(", "GL_LINK_STATUS", "GL_COMPILE_STATUS", "CompileShader", "LinkProgram"):
    forbid(render, needle, "caustic render")
for needle in ("acquireOpenGlCausticProgramsUnlocked(multipleScatterQuery,roughQuery,true,programSet,programError)",
               "openGlCausticQueryProgramForOwnerUnlocked(*programSet,querySources)",
               "querySources,result.error,linkedQuery)",
               "resources->causticProgramSet=programSet;",
               "resources->causticProgramSet->key==causticProgramKey"):
    require(render, needle, "caustic render")

# Resources lease the set. They never own or delete the shared programs.
forbid(backend, "causticPrograms[", "backend_opengl.cpp")
forbid(backend, "causticProgramMultipleScatterVersion", "backend_opengl.cpp")
require(backend, "mutable std::shared_ptr<OpenGlCausticProgramSet> causticProgramSet;", "scene resources")

# The cache is per GL context and released with the context's other programs.
require(backend, "std::map<GLFWwindow*, OpenGlCausticContextPrograms>& openGlCausticProgramCache()",
        "caustic program cache")
release = region(backend, "void releaseCachedProgramsForCurrentContext() noexcept override",
                 "BackendInfo info() const override", "context release")
require(release, "releaseOpenGlCausticProgramsForCurrentContextUnlocked();", "context release")

# Submission starts the compiles and the link without reading any status, after a disk
# binary lookup, and marks the link for the binary cache.
submit = region(backend, "std::string submitCausticStage (", "std::string finishCausticStage (",
                "caustic submit")
for needle in ("binaryCache.load (openGlCausticStageDiskKey (sources), kCausticDiskKeySecond)",
               "gl.CompileShader (shader);", "binaryCache.prepareLink (stage.program);",
               "gl.LinkProgram (stage.program);"):
    require(submit, needle, "caustic submit")
for needle in ("GetProgramiv", "GetShaderiv", "GL_LINK_STATUS", "GL_COMPILE_STATUS"):
    forbid(submit, needle, "caustic submit")

# Without a wait the acquire polls GL_COMPLETION_STATUS_KHR and returns Pending before
# it reads any compile or link result. Finished programs go to the disk binary cache.
acquire = region(backend, "OpenGlCausticProgramAcquire acquireOpenGlCausticProgramsUnlocked (",
                 "unsigned openGlCausticQueryProgramForOwnerUnlocked (", "caustic acquire")
poll_at = acquire.find("GL_COMPLETION_STATUS_KHR")
pending_at = acquire.find("return OpenGlCausticProgramAcquire::Pending;")
finish_at = acquire.find("finishCausticStage (gl, stage)")
if poll_at < 0 or pending_at < 0 or finish_at < 0 or not (poll_at < pending_at < finish_at):
    errors.append("caustic acquire must poll completion and return Pending before finishing")
for needle in ("arbitgl::hasParallelShaderCompile (gl)", "if (cache.parallelCompile != 1) wait = true;",
               "cache.inFlight.emplace (key, std::move (entry))",
               "binaryCache.store (openGlCausticStageDiskKey ((*sources)[index]), kCausticDiskKeySecond, program);",
               "contextEntry->second.ready[key] = set;"):
    require(acquire, needle, "caustic acquire")
ready_at = acquire.find("cache.ready.find (key)")
build_at = acquire.find("openGlCausticProgramSources (multipleScatter, rough)")
if ready_at < 0 or build_at < 0 or ready_at > build_at:
    errors.append("caustic acquire must reuse a ready set before it builds any source text")

# Every program is marked retrievable before its link, not only the query program.
require(acquire, "retrievable, entry.stages[index]);", "caustic acquire")
forbid(acquire, "index == 0 && retrievable", "caustic acquire")

# GL names of an unpublished entry are deleted on errors and exceptions. The submit
# guard is disarmed only after the entry is in the cache.
emplace_at = acquire.find("flying = cache.inFlight.emplace (key, std::move (entry)).first;")
disarm_at = acquire.find("submitGuard.entry = nullptr;")
if emplace_at < 0 or disarm_at < 0 or disarm_at < emplace_at:
    errors.append("caustic acquire must disarm the submit guard only after the entry is cached")
for needle in ("OpenGlCausticInFlightGuard submitGuard {gl, &entry};",
               "OpenGlCausticInFlightGuard finishGuard {gl, &entry};"):
    require(acquire, needle, "caustic acquire")

# A blocking wait releases fixtureMutex when the caller hands over its lock, and the set
# is published after the mutex is taken again.
unlock_at = acquire.find("if (wait && lock != nullptr) lock->unlock();")
finish_loop_at = acquire.find("finishCausticStage (gl, stage)")
publish_at = acquire.find("contextEntry->second.ready[key] = set;")
if min(unlock_at, finish_loop_at, publish_at) < 0 or not (unlock_at < finish_loop_at < publish_at):
    errors.append("caustic acquire must finish a blocking wait with fixtureMutex released and publish after")
require(acquire, "OpenGlCausticRelock relock {lock};", "caustic acquire")

# The query owner adopts a linked program instead of compiling when one is available.
require(backend, "owner->program=prelinked?prelinked:linkCausticProgram(gl,sources,error);", "query owner")

# The shader text did not change when it moved out of the render.
for needle in ('smoothglass::replace(resolveTail,"if (outRawMask <= 0.0) discard;","");',
               '"color.rgb+=causticXyzToSceneLinear(xyz);");',
               "+kFixturePassCompositeShader+resolveTail}};"):
    if backend.count(needle) != 1:
        errors.append("caustic resolve source must appear exactly once: " + needle)

# Public entry point.
require(header, "enum class OpenGlCausticProgramReadiness : std::uint8_t { Ready, Pending, Failed };", "header")
require(header, "OpenGlCausticProgramReadiness prepareOpenGlGlassCausticPrograms(bool multipleScatter,bool rough,",
        "header")
entry = region(backend, "OpenGlCausticProgramReadiness prepareOpenGlGlassCausticPrograms(",
               "NativeFixtureSceneSubmission renderOpenGlGlassCausticScene(", "entry point")
require(entry, "use==NativeFixturePrepareUse::Blocking", "entry point")
require(entry, "std::unique_lock<std::mutex> lock(fixtureMutex());", "entry point")
require(entry, "programs,error,&lock);", "entry point")
require(entry, "if(glfwGetCurrentContext()==nullptr)return OpenGlCausticProgramReadiness::Ready;", "entry point")
lock_at = entry.find("std::unique_lock<std::mutex> lock(fixtureMutex());")
no_context_at = entry.find("if(glfwGetCurrentContext()==nullptr)")
if lock_at < 0 or no_context_at < 0 or no_context_at > lock_at:
    errors.append("entry point must return before locking when no context is current")

# Renderer: programs start before the preparation, and a pending result returns before
# the caustic submit scope opens.
start_at = renderer.find("arbitgpu::prepareOpenGlGlassCausticPrograms (")
prepare_at = renderer.find('prepareScope.emplace ("advance:imported-scene:fixture-prepare");')
pending_at = renderer.find("if (causticProgramsPending)")
submit_at = renderer.find('"advance:imported-scene:fixture-caustic-submit"')
if min(start_at, prepare_at, pending_at, submit_at) < 0 or not (start_at < prepare_at < pending_at < submit_at):
    errors.append("renderer must start caustic programs before prepare and return pending before the caustic submit")
require(renderer, 'loopstep::Scope causticProgramScope ("advance:imported-scene:fixture-caustic-programs");',
        "renderer")
require(renderer, "use == RenderUse::Preview ? arbitgpu::NativeFixturePrepareUse::Preview", "renderer")
pending_block = renderer[pending_at:submit_at] if pending_at >= 0 and submit_at > pending_at else ""
for needle in ("++previewPendingResults;", "if (use != RenderUse::Preview)", "output.pending = true;",
               "return true;"):
    require(pending_block, needle, "renderer caustic pending")

# The caustic refusals come before the programs start, so a backend without a caustic
# consumer still reports its own refusal.
refusal_at = renderer.find('error = "Glass caustics have no native consumer in this renderer backend";')
if refusal_at < 0 or start_at < 0 or refusal_at > start_at:
    errors.append("renderer must refuse a backend without a caustic consumer before starting caustic programs")
# One roughness test selects both the multiple scatter transport and the rough query.
require(renderer, "causticRoughQuery = selectedMaterial->parameters.emissionRoughness[3] > 0.0f;\n        if (causticRoughQuery)\n            multipleScatterPolicy.emplace();", "renderer")

# Kernel splat programs are part of the prewarmed set, so the async compile and the
# disk binary cache cover them like the trace, sort and reduce programs.
require(backend, "constexpr std::size_t kCausticProgramCount = 10;", "caustic program set")
for needle in ("sources[kCausticSplatNormaliserProgram] = {{GL_COMPUTE_SHADER,gc::opengl::splatNormaliserSource()}};",
               "sources[kCausticSplatKeyProgram] = {{GL_COMPUTE_SHADER,gc::opengl::splatKeySource()}};",
               "sources[kCausticSplatSegmentProgram] = {{GL_COMPUTE_SHADER,gc::opengl::splatSegmentSource()}};",
               "sources[kCausticSplatGatherProgram] = {{GL_COMPUTE_SHADER,gc::opengl::splatGatherSource()}};"):
    require(backend, needle, "caustic program set")

# Only the GL consumer opts in, after the frame is scheduled, and it charges the
# schedule's scratch bytes in frame admission.
require(render, "gc::enableSplat(prepared,scheduled,splatError)", "caustic render")
require(render, "charge(1,scheduled.splat.scratchBytes)", "caustic render")
require(render, "keyCount=splat?scheduled.splat.paddedKeyCount:deposition.paddedKeyCount", "caustic render")
enable_at = render.find("gc::enableSplat(prepared,scheduled,splatError)")
charge_at = render.find("charge(1,scheduled.splat.scratchBytes)")
if enable_at < 0 or charge_at < 0 or enable_at > charge_at:
    errors.append("caustic render must enable splat before it admits frame memory")

# Pass order: trace, normaliser, keys, bitonic sort, segments, overflow counters,
# gather, readback, then validation.
order = ("splatUniforms(programs[kCausticSplatNormaliserProgram]);",
         "splatUniforms(programs[kCausticSplatKeyProgram]);",
         "gl.UseProgram(programs[4]);",
         "splatUniforms(programs[kCausticSplatSegmentProgram]);",
         "splatLayout.counterBase",
         "splatUniforms(programs[kCausticSplatGatherProgram]);",
         "readback->deposition.splatNormalisers.push_back(",
         "gc::validateNativeReadback(prepared,splatOwner?*splatOwner:scheduled,*readback,energy,result.error,&submission)")
positions = []
cursor = 0
for needle in order:
    at = render.find(needle, cursor)
    if at < 0:
        errors.append("caustic render splat pass order is missing " + needle)
        break
    positions.append(at)
    cursor = at + len(needle)
require(render, "readback->deposition.overflow=true", "caustic render")

# Key layout: one uvec4 (tile, work, slot, deposit) per touched tile with an all-ones
# sentinel that sorts after every valid key.
keys = region(header, "inline std::string splatKeySource()", ")GLSL\";", "splat key source")
for needle in ("uvec4 key=uvec4(0xffffffffu);",
               "key=uvec4((y0+which/columns)*uTilesX+x0+which%columns,work,slot,index);",
               "uint per=uint(CAUSTIC_SPLAT_TILES_PER_DEPOSIT),keyed=at/per,which=at%per,work=keyed/uint(uKeySlots),slot=keyed%uint(uKeySlots);",
               "uint index=work*uint(uVisits)+slot;uvec2 pixel;"):
    require(keys, needle, "splat key source")
require(header, "static_assert(kSplatTilesPerDeposit==9", "splat key source")
# Keys cover only the keyed slots of each path. Both hosts pass the schedule's count.
require(render, 'gl.Uniform1i(at("uKeySlots"),int(scheduled.splat.keySlots));', "caustic render")
# A path with deposits past its keyed slots sets overflow bit 8, and both hosts refuse.
require(keys, "atomicOr(splat[uCounterBase],8u);", "splat key source")
require(header, "kSplatOverflowKeySlots=8;", "GL splat overflow bits")
require(render, "if(overflow&gc::opengl::kSplatOverflowKeySlots)", "caustic render")
require(metal_header, "kSplatOverflowKeySlots=8;", "Metal splat overflow bits")
require(metal, "if(counters[0]&metalcaustic::kSplatOverflowKeySlots)return refuse(", "Metal caustic schedule")
require((root / "src/gpu_backend/fixture_glass_caustics.h").read_text(), "+\"\\n#define CAUSTIC_SPLAT_TILES_PER_DEPOSIT \"+std::to_string(kSplatTilesPerDeposit)", "splat math source")
require(header, "layout(local_size_x=16,local_size_y=16) in;", "splat gather source")
require(render, "gl.DispatchCompute(unsigned(scheduled.splat.tilesX*groupsPerTile),unsigned((rows+gc::kSplatGatherGroup-1)/gc::kSplatGatherGroup),1);",
        "caustic render")
require(header, "constexpr unsigned kSplatScratchBinding=7", "splat scratch binding")

# Grating bindings own the single-pixel frame. The GL producer accepts that identity
# and validates the ordinary receipt against the splat copy.
require(backend, "||frame->causticSinglePixelIdentity==binding->frame.identity", "GL grating producer")
require(backend, "if(!gc::enableSplat(retained,*splatOwner,error))return false;", "GL grating producer")
require(grating, "if(!disableSplat(prepared,single,error))", "grating binding")

# Metal mirrors the GL splat. The consumer opts in after the frame is scheduled and
# charges the scratch bytes in frame admission. Metal has no caustic prewarm.
require(metal, "if(!glasscaustics::enableSplat(*causticPrepared,causticSchedule,splatError)", "Metal caustic schedule")
require(metal, "causticSchedule.splat.scratchBytes", "Metal caustic schedule")
require(metal, "+causticSchedule.bounds.maximumDeposits*128+splatExtra;", "Metal caustic schedule")
require(metal, "causticKeyCount=causticSplat?causticSchedule.splat.paddedKeyCount", "Metal caustic schedule")
metal_enable = metal.find("glasscaustics::enableSplat(*causticPrepared,causticSchedule,splatError)")
metal_admit = metal.find("glasscaustics::admitFrameMemory(causticSchedule,extra,causticBytes,result.error)")
if metal_enable < 0 or metal_admit < 0 or metal_enable > metal_admit:
    errors.append("Metal caustic schedule must enable splat before it admits frame memory")
require(metal, "state.keys=buffer(nullptr,keyCount*sizeof(glasscaustics::DepositionKey));", "Metal caustic storage")
require(metal, "||(frame->caustic->schedule.identity!=binding->frame.identity&&frame->caustic->singlePixelIdentity!=binding->frame.identity)",
        "Metal grating producer")

# Metal pass order: normaliser, keys, bitonic sort, segments, host overflow check,
# gather, normaliser readback in w*maximumVisits+d order, then validation.
complete = region(metal, "bool completeMetalCaustic(MetalFixtureSceneFrame& frame,std::string& error)",
                  "glasscaustics::validateNativeReadback(state.prepared,splatOwner?*splatOwner:schedule,receipt,ledger,error,&state.identity)",
                  "Metal caustic completion")
cursor = 0
for needle in ("bindSplat(encoder,state.splatNormaliser);",
               "bindSplat(encoder,state.splatKeys);",
               "bindSplat(encoder,state.sort);",
               "bindSplat(encoder,state.splatSegments);",
               "metalcaustic::kSplatOverflowTile",
               "state.metrics.splatGatherWork=splatGatherWork;",
               "bindSplat(encoder,state.splatGather);",
               "std::memcpy(&normaliser,words+w*policy.maximumVisits+d,sizeof(normaliser));",
               "receipt.deposition.splatNormalisers.push_back(normaliser);"):
    at = complete.find(needle, cursor)
    if at < 0:
        errors.append("Metal caustic splat pass order is missing " + needle)
        break
    cursor = at + len(needle)

# Metal key layout matches the GL one: a uint4 (tile, work, slot, deposit) per
# touched tile and an all-ones sentinel.
metal_keys = region(metal_header, "kernel void causticSplatKeys(", "kernel void causticSplatSegments(", "Metal splat key kernel")
for needle in ("uint4 key=uint4(0xffffffffu);",
               "uint per=uint(CAUSTIC_SPLAT_TILES_PER_DEPOSIT),keyed=at/per,which=at%per,work=keyed/s.tiles.w,slot=keyed%s.tiles.w;",
               "uint index=work*u.policy.y+slot;uint2 pixel;",
               "key=uint4((y0+which/columns)*s.tiles.y+x0+which%columns,work,slot,index);"):
    require(metal_keys, needle, "Metal splat key kernel")
require(metal_header, 'static_assert(glasscaustics::kSplatTilesPerDeposit==9', "Metal splat key kernel")
require(metal_keys, "atomic_fetch_or_explicit((device atomic_uint*)(splat+s.tiles.x),8u,memory_order_relaxed);", "Metal splat key kernel")
require(metal, "std::uint32_t(schedule.splat.keySlots)};", "Metal caustic schedule")
for needle in ("constexpr std::uint32_t kSplatOverflowTile=1", "kSplatOverflowTileCap=2", "kSplatOverflowGatherWork=4"):
    require(metal_header, needle, "Metal splat overflow bits")
# The scratch layout is the GL one: normalisers at 0, ranges after maximumDeposits,
# triangle rows, then two counter words.
gl_layout = region(header, "inline bool splatScratchLayout(const Frame& frame,SplatScratchLayout& output)", "output=layout;", "GL splat layout")
metal_layout = region(metal_header, "inline bool splatScratchLayout(const glasscaustics::Frame& frame,SplatScratchLayout& output)",
                      "output=layout;", "Metal splat layout")
def layout_body(text):
    return text[text.find("SplatScratchLayout layout;"):]
if not gl_layout or layout_body(gl_layout) != layout_body(metal_layout):
    errors.append("Metal splat scratch layout differs from the GL layout")

# Both consumers cap the radius by the footprint Jacobian's smallest singular value
# and use the NaN-safe normal agreement test of the CPU reference.
require(header, "deposit.radius=causticSplatRadiusFromScale(workTable[work*4u].w,", "GL splat radius")
require(header, "causticSplatPixelScale(deposit.pixel,deposit.position,deposit.normal,uCamera.xyz,uRotation,uProjection,uExtent)", "GL splat radius")
require(header, "if(!(dot(normal,deposit.normal)>=0.9))return 0.0;", "GL splat weight")
require(metal_header, "deposit.radius=causticSplatRadiusFromScale(work[w*4u].w,", "Metal splat radius")
require(metal_header, "causticSplatPixelScale(deposit.pixel,deposit.position,deposit.normal,u.camera.xyz,u.rotation,u.projection,float2(u.extent.xy))",
        "Metal splat radius")
require(metal_header, "if(!(dot(normal,deposit.normal)>=0.9f))return 0.0f;", "Metal splat weight")
require(metal_header, 'replace("uvec2","uint2");replace("vec2","float2");', "Metal splat translation")
for text, label in ((header, "GL splat source"), (metal_header, "Metal splat source")):
    forbid(text.replace("causticSplatRadiusFromScale", ""), "causticSplatRadius(", label)

# Crowded tiles fall back to single-pixel deposits instead of failing the frame.
# Only a tile key outside the image stays fatal. The consumer re-keys the frame and
# submission to the single-pixel identity and validates against the splat frame,
# whose CPU reference must overflow the same caps.
caustics = (root / "src/gpu_backend/fixture_glass_caustics.h").read_text()
for needle in ("candidate.capOverflow=true;",
               "if(reference.capOverflow)return refuse(",
               "if(!reference.capOverflow)return refuse(",
               "inline bool disableSplat(const Prepared& prepared,Frame& frame,std::string& error)",
               "if (placement.object!=i)return refuse("):
    require(caustics, needle, "shared splat fallback")
fallback = region(render, "if(overflow&gc::opengl::kSplatOverflowTile)", "readback->deposition.splatGatherWork=gatherWork;", "GL splat fallback")
for needle in ("return fail(\"OpenGL caustic splat tile key is outside the image\");",
               "splatOwner.emplace(scheduled);",
               "gc::disableSplat(prepared,scheduled,result.error)",
               "if(scheduled.identity!=singlePixelIdentity)",
               "submission.frameIdentity=scheduled.identity;readback->frameIdentity=scheduled.identity;",
               "splatGathered=false;readbackKeyCount=deposition.paddedKeyCount;"):
    require(fallback, needle, "GL splat fallback")
require(render, "if(!splatGathered)", "GL splat fallback")
require(render, "gc::validateNativeReadback(prepared,splatOwner?*splatOwner:scheduled,*readback,energy,result.error,&submission)",
        "GL splat fallback")
metal_fallback = region(metal, "if(counters[0]&metalcaustic::kSplatOverflowTile)return refuse(", "splatGatherWork=counters[1];",
                        "Metal splat fallback")
for needle in ("splatOwner.emplace(state.schedule);",
               "glasscaustics::disableSplat(state.prepared,state.schedule,error)",
               "if(state.schedule.identity!=state.singlePixelIdentity)",
               "state.identity.frameIdentity=state.schedule.identity;",
               "splatGathered=false;readbackKeyCount=deposition.paddedKeyCount;"):
    require(metal_fallback, needle, "Metal splat fallback")
for needle in ("if(!splatGathered) {", "receipt.deposition.keys.assign(nativeKeys,nativeKeys+readbackKeyCount);",
               "glasscaustics::validateNativeReadback(state.prepared,splatOwner?*splatOwner:schedule,receipt,ledger,error,&state.identity)"):
    require(metal, needle, "Metal splat fallback")
# Both splat surface predicates accept the trace's receivers only: shape.w == 1.
require(header, "if(shape.z!=0.0||shape.w!=1.0)return false;", "GL splat surface")
require(metal_header, "if(shape.z!=0.0f||shape.w!=1.0f)return false;", "Metal splat surface")

# smoothglass::replace leaves the source unchanged when its needle is missing, so
# a stale needle in the grating terminal rewrite of computeSource would pass
# silently. Every needle must still be in the text it rewrites.
terminal = region(header, "inline std::string gratingTerminalComputeSource(", "    return source;\n}", "grating terminal rewrite")
plain = "\n".join(line for line in header.split("\n") if "smoothglass::replace(" not in line)
receiver_text = region(plain, "        if(shape.z==0.0) {", "        if(branch.inside==hit.entering)", "grating terminal receiver")
needles = re.findall(r'smoothglass::replace\((receiver|source),"((?:[^"\\]|\\.)*)"', terminal)
if len(needles) < 8:
    errors.append("grating terminal rewrite needles were not found")
for target, needle in needles:
    text = needle.encode().decode("unicode_escape")
    if text not in (receiver_text if target == "receiver" else plain):
        errors.append("grating terminal rewrite needle is stale: " + text[:80])
for needle in ('smoothglass::replace(receiver,"complete=false;break;","return false;");',
               'smoothglass::replace(source,"    paths[base].zw=vec2(float(count),float(visits));"'):
    require(terminal, needle, "grating terminal rewrite")

# Preview cost. Export keeps the default schedule, a fence after every stage and band,
# the frozen table check, the host receipt with validateNativeReadback, and the finite
# Color readback. Preview skips all of these, flushes between stages and waits once on
# the resolve. The plan, not the call site, decides each of them.
plan = region(header, "inline OpenGlCausticRenderPlan openGlCausticRenderPlan(", "\n}\n", "caustic render plan")
for needle in ("if(use!=OpenGlCausticRenderUse::Preview)return plan;",
               "plan.scheduling.entrySubdivisions=kPreviewCausticEntrySubdivisions;",
               "plan.hostReceipt=false;plan.stageFences=false;"):
    require(plan, needle, "caustic render plan")
require(header, "bool hostReceipt=true,stageFences=true,bandFences=true,reuseSpectrum=false;", "caustic render plan")
require(header, "OpenGlCausticRenderUse use=OpenGlCausticRenderUse::Export,", "caustic render declaration")
require(render, "const auto plan=openGlCausticRenderPlan(use,width,height);", "caustic render")
require(render, "const bool trySplat=!preview||openGlCausticPreviewSplatAllowed(scheduled);", "caustic render")
stage = region(render, "const auto stageDone=[&](bool band) {", "};", "caustic stage sync")
require(stage, "if(plan.stageFences||(band&&plan.bandFences))return causticFence(gl,result.error);", "caustic stage sync")
require(stage, "glFlush();return true;", "caustic stage sync")
passes = region(render, "const auto rasterStarted=std::chrono::steady_clock::now();",
                "// Only validated GPU spectral storage is resolved into product Color.", "caustic passes")
if passes.count("causticFence(") != 0:
    errors.append("caustic passes must wait through stageDone, so Preview can flush instead")
if passes.count("stageDone(") != 7:
    errors.append("caustic passes changed their stage waits: %d" % passes.count("stageDone("))
for name in ("uSortSpan", "uSortDistance"):
    if passes.count('GetUniformLocation(programs[4],"' + name + '")') != 1:
        errors.append("the bitonic sort must look up " + name + " once, outside its loops")
if passes.count('GetUniformLocation(programs[1],"uTriangle")') != 1:
    errors.append("the raster must look up uTriangle once, outside its triangle loop")
receipt = region(render, "if(plan.hostReceipt)\n", "// Only validated GPU spectral storage", "caustic host receipt")
for needle in ("if(!gc::validateTables(prepared,scheduled,result.error))return fail(result.error);",
               "gc::opengl::causticDecodePath(",
               "gc::opengl::causticDecodeDeposit(",
               "gc::validateNativeReadback(prepared,splatOwner?*splatOwner:scheduled,*readback,energy,result.error,&submission)"):
    require(receipt.split("else\n")[0], needle, "Export caustic host receipt")
preview_branch = receipt.split("        else\n")[-1]
for needle in ("validateNativeReadback", "validateTables", "MapBufferRange", "causticDecode"):
    forbid(preview_branch, needle, "Preview caustic branch")
require(render, "if(plan.hostReceipt&&(!frame->readTextureOwned(descriptor,color,result.error)", "caustic Color check")
require(render, "if(plan.hostReceipt) {frame->causticEnergy=energy;frame->causticReadback=std::move(readback);}", "caustic receipt")
# Export waits on the resolve fence. Preview flushes it and never waits.
resolve = region(render, "// Only validated GPU spectral storage is resolved into product Color.", "NativeRawPassPixels color;",
                 "caustic resolve")
require(resolve, "glDrawArrays(GL_TRIANGLES,0,3);", "caustic resolve")
require(resolve, "if(!preview)\n        {\n            loopstep::Scope resolveWaitScope(\"gl:caustic-resolve-wait\");\n            if(!causticFence(gl,result.error))return fail(result.error);\n        }\n        else glFlush();",
        "caustic resolve")
# Spectrum reuse is Preview only and keyed on the frame identity, extent, time and owner.
require(render, "const auto reuseKey=plan.reuseSpectrum?openGlCausticSpectrumReuseKey(scheduled,runtime.timeSeconds,", "caustic reuse")
require(render, "const bool trace=!preview||(!spectrumReused&&!previewPoll.inFlight&&!previewPoll.declined&&!resources->previewCausticJob);", "caustic reuse")
require(render, "const bool resolveCaustic=!preview||spectrumReused||previewLagged;", "caustic reuse")
require(render, "if(plan.reuseSpectrum)job.exact=reuseKey;", "caustic reuse")
reuse_key = region(header, "inline bool operator==(const OpenGlCausticSpectrumReuseKey& a,", "\n}\n", "caustic reuse key")
for field in ("context", "rendererGeneration", "width", "height", "timeSeconds", "frameIdentity"):
    require(reuse_key, "a." + field + "==b." + field, "caustic reuse key")
require(backend, "for(auto& buffer:previewSpectrumBuffers)if(buffer){gl.DeleteBuffers(1,&buffer);buffer=0;}", "scene resources")
# The renderer passes the plan's schedule and its use. Metal is unchanged.
require(renderer, "causticPlan.scheduling, dimensions.width, dimensions.height, sceneInputs,", "renderer")
require(renderer, "multipleScatterPolicy ? &*multipleScatterPolicy : nullptr, causticUse, &causticDegradation);", "renderer")
require(renderer, "use == RenderUse::Preview ? arbitgpu::OpenGlCausticRenderUse::Preview", "renderer")
# Slow Preview frames log their stage timings.
require(render, "const bool slowLog=preview&&diagnostics.frameMilliseconds>kPreviewCausticSlowFrameMilliseconds&&[] {", "caustic timing log")
require(render, "if(slowLogged&&now-*slowLogged<std::chrono::seconds(1))return false;", "caustic timing log throttle")
require(render, "if(slowLog)\n        std::fprintf(stderr,", "caustic timing log throttle")
# A failed frame drops the retained Preview spectrum; names are deleted only on a live context.
require(render, "resources->previewSpectrumKey.reset();\n        if(std::string resetError;causticContextIntact(resetError))", "caustic fail drops reuse")
require(render, "resources->previewSpectrumBuffers={};", "caustic fail drops reuse")
# A Preview caller must pass the Preview plan's schedule.
require(render, "const auto expected=openGlCausticRenderPlan(use,width,height).scheduling;", "preview schedule check")
require(render, "scheduling.entrySubdivisions!=expected.entrySubdivisions", "preview schedule check")
require(render, '"[gl] caustic preview %ux%u %.1f ms: cpu base', "caustic timing log")

# GL limits. Preview steps one entry subdivision coarser, then renders the scene without
# the caustic and carries the reason as a non-fatal warning. Export keeps its schedule and
# refuses with the named limit, so it never silently drops the caustic.
decide = region(header, "inline OpenGlCausticAdmissionDecision decideOpenGlCausticAdmission(", "\n}\n",
                "caustic limit decision")
for needle in ("if(failure.empty()) {decision.admission=OpenGlCausticAdmission::Full;",
               "if(use!=OpenGlCausticRenderUse::Preview)\n    {decision.reason=\"OpenGL caustic admitted storage exceeds native GL limits: \"+failure;return decision;}",
               "decision.admission=OpenGlCausticAdmission::Reduced;",
               "decision.admission=OpenGlCausticAdmission::Off;decision.reason=\"caustic off: exceeds GL limits (\"+failure+\")\";"):
    require(decide, needle, "caustic limit decision")
export_at = decide.find("if(use!=OpenGlCausticRenderUse::Preview)")
reduced_at = decide.find("OpenGlCausticAdmission::Reduced;")
if export_at < 0 or reduced_at < 0 or export_at > reduced_at:
    errors.append("caustic limit decision must refuse Export before any reduction")
require(header, '+", max SSBO "+openGlCausticMebibytes(limits.maximumStorageBlockBytes);', "caustic limit failure")
require(header, "std::string* degradation=nullptr);", "caustic render declaration")
forbid(render, "OpenGL caustic admitted storage exceeds native buffer, dispatch or texture limits", "caustic render")
forbid(render, "exceeds native bindings, buffer or dispatch limits", "caustic render")
forbid(render, "(max SSBO \"", "caustic render")
require(header, '"%.1f MiB"', "caustic limit failure")
require(header, '"splat needs 8 storage bindings, max "+count(limits.storageBindings);', "caustic splat limit failure")
for needle in ("if(preview)\n        {\n            const auto planned=openGlCausticStorageDemand(scheduled,scheduled.bounds.deposition.paddedKeyCount);",
               "if(decision.admission!=OpenGlCausticAdmission::Reduced)return renderWithoutCaustic(decision.reason);",
               "scheduled=std::move(reducedFrame);reducedReason=decision.reason;",
               "if(preview&&!failure.empty()) {cameraRoughSchedule.reset();return renderWithoutCaustic(",
               '} else refused.error="OpenGL caustic admitted storage exceeds native GL limits: "+failure;',
               'refused.error="OpenGL caustic splat storage exceeds native GL limits: "+failure;return refused;',
               'if(preview&&limit)return renderWithoutCaustic("caustic off: exceeds GL limits ("+failure+")");',
               'if(preview)return renderWithoutCaustic("caustic off: GL storage limit query unavailable");',
               "const auto failure=openGlCausticLimitFailure(limits,openGlCausticStorageDemand(scheduled,deposition.paddedKeyCount));\n        if(failure.empty())return refused;",
               "const auto failure=openGlCausticSplatLimitFailure(limits,{splatLayout.words,keyCount,",
               "if(base.rendered&&degradation)*degradation=std::move(reason);",
               "if(!reducedReason.empty()) {reportOpenGlCausticDegradation(reducedReason);if(degradation)*degradation=std::move(reducedReason);}"):
    require(render, needle, "caustic limit degrade")
# Every degrade is Preview only.
for line in render.splitlines():
    if "return renderWithoutCaustic(" in line and "preview" not in line and "decision.admission" not in line:
        errors.append("caustic render degrades outside a Preview branch: " + line.strip())
without = region(render, "const auto renderWithoutCaustic=[&](std::string reason) {", "\n    };", "caustic degrade render")
for needle in ("reportOpenGlCausticDegradation(reason);", "baseRuntime.linearColor=true;",
               "baseRuntime.imageOutput=renderpassoutput::Output::Color;",
               "nativeFixtureSceneBackend().render(scene,nativeResources,width,height,baseRuntime);"):
    require(without, needle, "caustic degrade render")
require(backend, "if(reported.size()>=64||!reported.insert(reason).second)return;", "caustic degrade log")
require(renderer, "rendered.causticDegradation = std::move (causticDegradation);", "renderer")
renderer_header = (root / "src/fixture_scene_renderer.h").read_text()
require(renderer_header, "std::string causticDegradation;", "RenderedFrame")
tests = (root / "tests/glass_caustic_contract_tests.cpp").read_text()
require(tests, "void causticLimitAdmission()", "glass caustic tests")
require(tests, "causticLimitAdmission();previewCausticPlan();", "glass caustic tests")

require(cmake, "tests/glass_caustic_program_prewarm_source_contract.py", "CMakeLists.txt")

# Preview camera walk budget. Export keeps the released transport text, the 48/16/64
# visits, every rough sample and a finish per glass band. Only the Preview variant of a
# smooth-glass program, built without a wait, carries the Preview budget.
smooth = (root / "src/gpu_backend/fixture_smooth_glass.h").read_text()
recipe = smooth[smooth.find("template<class String> inline String transportShaderRecipe"):]
for needle in ("int walks=sharedPaths?1:3,visits=sharedPaths?48:16;", "bool sharedPaths=volumeDispersion==0.0;",
               "        for(int visit=0;visit<64;++visit) {\n", "+ \"\\n#define GLASS_ROUGH_SAMPLES \" + surfacesource::number<String>(kRoughInterfaceSamples)",
               "+ \"\\n#define GLASS_ROUGH_VISITS \" + surfacesource::number<String>(kMaximumRayVisits)"):
    require(recipe, needle, "export transport text")
forbid(recipe, "GLASS_PREVIEW", "export transport text")
budget = region(smooth, "inline CameraWalkBudget cameraWalkBudget(", "inline std::string previewCameraTransportShader(", "walk budget")
require(budget, "if (!preview) return budget;", "walk budget")
require(smooth, "std::uint32_t pairVisits = 48, dispersivePairVisits = 16, singleVisits = 64, singleChannelWalks = 3;", "walk budget")
require(smooth, "inline constexpr std::uint64_t kPreviewUnfencedGlassPixels = 2560ull*1440ull;", "walk budget")
fragment = region(backend, "std::string fixtureFragmentShaderSource (unsigned features)", "void readFixtureUniformLocations (", "fragment source")
require(fragment, "if ((features & kFixtureFragmentGlassPreview) != 0)", "fragment source")
require(fragment, "transport = smoothglass::previewCameraTransportShader (std::move (transport));", "fragment source")
if backend.count("kFixtureFragmentGlassPreview") != 3:
    errors.append("kFixtureFragmentGlassPreview must appear only in the enum, the fragment source and the variant build")
variant = region(backend, "const OpenGlFixtureSceneResources::SharedProgram* advancePreviewGlassProgram (", "\n}\n", "preview variant")
for needle in ("binaryCache.load (shared->vertexSource, shared->previewFragmentSource);", "arbitgl::hasParallelShaderCompile (gl)",
               "submitFixtureProgram (gl, binaryCache, shared->vertexSource,", "GL_COMPLETION_STATUS_KHR, &linked);",
               "if (linked != GL_TRUE) return nullptr;", "binaryCache.store (shared->vertexSource, shared->previewFragmentSource, shared->previewId);"):
    require(variant, needle, "preview variant")
for needle in ("gl.CompileShader", "gl.LinkProgram", "glFinish"):
    forbid(variant, needle, "preview variant")
if variant.find("finishFixtureProgram (entry)") < variant.find("if (linked != GL_TRUE) return nullptr;"):
    errors.append("the preview variant reads its link status only after the completion poll")
draw = region(backend, "const bool previewGlassFrame = runtimeInputs.previewUse && resources->smoothGlass.enabled", "result.rendered = true;", "scene draw")
for needle in ("&& !roughGlassFrame && !diffractionFrame;", "const unsigned sceneProgram = previewGlassProgram ? previewGlassProgram->previewId : resources->program;",
               "gl.UseProgram (sceneProgram);",
               "const bool bandFinish = smoothglass::cameraWalkBudget(runtimeInputs.previewUse, width, height).bandFinish;",
               "                if (bandFinish)\n                {\n                    loopstep::Scope bandFinishScope (\"gl:closed-glass-band-finish\");\n                    glFinish();\n                }\n                else glFlush();",
               "queryAvailable(glassTimer->glassTimer[1], GL_QUERY_RESULT_AVAILABLE, &available);",
               "if (!(runtimeInputs.previewUse && runtimeInputs.previewCallerWaits))",
               "if (resources->smoothGlass.enabled && !roughGlassFrame && !runtimeInputs.previewUse) {",
               "if (!frame->readRawPassOwned(renderpassoutput::Output::Normal, completion, result.error)",
               "|| !smoothglass::finiteColorFrame(completion, result.error))) return fail(result.error);"):
    require(draw, needle, "scene draw")
forbid(draw, "GL_QUERY_RESULT_NO_WAIT", "scene draw")
require(backend, "if (!authoredDiffraction || runtimeInputs.previewUse) return true;", "diffraction validation")
require(render, "baseRuntime.previewUse=preview;baseRuntime.previewCallerWaits=preview;", "caustic render")
if render.count("baseRuntime.previewUse=preview;baseRuntime.previewCallerWaits=preview;") != 2:
    errors.append("caustic render: the caustic base and the fallback base must both skip the Preview wait")
require(renderer, "sceneInputs.previewUse = use == RenderUse::Preview;", "renderer")
require(tests, "previewCausticPlan();previewCameraWalkBudget();", "glass caustic tests")

if errors:
    for error in errors:
        print("FAIL: " + error)
    sys.exit(1)
print("glass caustic program prewarm source contract: PASS")
