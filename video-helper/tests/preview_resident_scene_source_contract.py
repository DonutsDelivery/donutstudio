#!/usr/bin/env python3
# Contract for the Preview resident scene. A Preview snapshot that changes only
# the deformed water vertices keeps every static resource (a structural cache
# hit). The backend uploads only the water vertex range with glBufferSubData,
# keeps the query BLAS texels resident, writes only changed TLAS tail texels
# with glTexSubImage2D and refits the water bounds with a compute pass. A
# Preview glass caustic beside the water keeps its prepared scene, schedule,
# work table, triangle table and query texture resident; a water frame rewrites
# only the water triangle rows and refits the water bounds, on the GPU. Export
# keeps the full prepare and the whole per-frame query and table upload.
import re
from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
interface = (root / "src/gpu_backend/backend.h").read_text()
visibility = (root / "src/gpu_backend/fixture_opaque_visibility.h").read_text()
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
renderer_header = (root / "src/fixture_scene_renderer.h").read_text()
cmake = (root / "CMakeLists.txt").read_text()

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


# The renderer takes the structural hit only for Preview frames without
# diffraction, with or without glass caustics. Every other frame keeps the full prepare.
require(renderer, "const bool residentFrame = use == RenderUse::Preview && diffractionMaterial == nullptr;", "renderer")
forbid(renderer, "use == RenderUse::Preview && ! sceneInputs.glassCaustic", "renderer")
require(renderer, "&& (residentFrame || ! cache_.residentRefreshed)\n", "renderer exact hit")
hit = region(renderer, "bool structuralCacheHit = false;", "const bool staticReuse", "renderer structural hit")
require(hit, "if (! exactCacheHit && residentFrame && cache_.native != nullptr", "renderer structural hit")
require(hit, "backend_.refreshPreviewDeformedVertices (cache_.native, snapshot, nativeMaterial)", "renderer structural hit")
require(hit, "cache_.residentRefreshed = true;", "renderer structural hit")
require(renderer, "const bool staticReuse = exactCacheHit || structuralCacheHit;", "renderer")
require(renderer, "if (! staticReuse)", "renderer")
require(renderer, "rendered.stats.reusedStaticResources = staticReuse;", "renderer stats")
require(renderer, "rendered.stats.staticUploadCount = staticReuse", "renderer stats")
# A full prepare clears the refreshed mark, and export always prepares on a new snapshot.
prepare = region(renderer, "if (! staticReuse)", "if (causticProgramsPending)", "renderer prepare")
require(prepare, "cache_.residentRefreshed = false;", "renderer prepare")
require(prepare, ": arbitgpu::NativeFixturePrepareUse::Blocking;", "renderer prepare")
require(renderer_header, "bool residentRefreshed = false;", "renderer cache")

# The interface default refuses, so backends without the path take the full prepare.
default = region(interface, "virtual bool refreshPreviewDeformedVertices (", "virtual NativeFixtureSceneSubmission render (", "interface")
require(default, "return false;", "interface default")

# The OpenGL refresh compares structure and sizes, uploads only the water range
# and touches no static texture or table.
refresh = region(backend, "    bool refreshPreviewDeformedVertices (", "    NativeFixtureSceneSubmission render (", "OpenGL refresh")
for needle in ("residentSurfaceProgramMatches (*resources.materialProgram, *materialProgram)",
               "? !residentSceneMatches (prepared, *scene, dynamic) || !validateVisual3DScene (*scene).valid()",
               ": !residentSceneStructureMatches (prepared, *scene, dynamic))",
               "glass.transport != surfacematerial::TransportMode::ThinInterface",
               "(glass.causticVolume.present && glass.causticVolume.roughness != 0.0f)",
               "resources.environmentTexture != 0",
               "resources.opaqueVisibility.objectMesh[dynamic] >= 0",
               "glfwGetProcAddress (\"glBufferSubData\")",
               "(GL_ARRAY_BUFFER,",
               "std::ptrdiff_t (std::size_t (water.firstVertex) * sizeof (SceneVertex))",
               "std::ptrdiff_t (std::size_t (water.vertexCount) * sizeof (SceneVertex))",
               "resources.snapshot = scene;",
               "++refit.vertexRevision;"):
    require(refresh, needle, "OpenGL refresh")
for needle in ("glTexImage2D", "glTexSubImage2D", "BufferData (", "BufferData(", "prepareGeometryInstancesImpl",
               "opaquevisibility::prepare", "smoothglass::prepare", "destroyUnlocked"):
    forbid(refresh, needle, "OpenGL refresh")
# Validation and the full compare run once, when the refit is first planned. Later
# water frames run only the structure check, which reads no mesh data.
if not re.search(r"if \(!refit\.planned\s*\? !residentSceneMatches", refresh):
    errors.append("OpenGL refresh must run the full scene compare only when the refit is unplanned")
if refresh.count("validateVisual3DScene") != 1:
    errors.append("OpenGL refresh must validate the scene only at establishment")
structure = region(backend, "bool residentSceneStructureMatches(", "bool residentSceneMatches(", "scene structure")
for needle in ("prepared.vertexCount != next.vertexCount", "prepared.indexCount != next.indexCount",
               "residentSameId(prepared.id, next.id)", "object.vertexCount != 0"):
    require(structure, needle, "scene structure")
for needle in ("memcmp", "vertices[", "vertices.data()", "indices[", "indices.data()", "textureTexels.data()", "isfinite"):
    forbid(structure, needle, "scene structure")
matches = region(backend, "bool residentSceneMatches(", "// GPU refit of the resident glass query.", "scene match")
for needle in ("if (!residentSceneStructureMatches(prepared, next, dynamic)) return false;",
               "next.indexCount * sizeof(std::uint32_t)", "next.textureTexelCount * sizeof(SceneTexelRgba8)",
               "(next.vertexCount - end) * sizeof(SceneVertex)", "object.firstVertex * sizeof(SceneVertex)"):
    require(matches, needle, "scene match")

# Preview renders keep both query textures resident. Only allocation or growth
# uses glTexImage2D; a frame writes the changed tail span with glTexSubImage2D.
upload = region(backend, "bool updateResidentQuery(", "bool runResidentRefit(", "resident upload")
require(upload, "glTexSubImage2D(GL_TEXTURE_2D, 0, int(column), int(row), int(run), 1, GL_RGBA, GL_FLOAT, data);", "resident upload")
require(upload, "glTexSubImage2D(GL_TEXTURE_2D, 0, 0, int(row), int(width), int(rows), GL_RGBA, GL_FLOAT, data);", "resident upload")
require(upload, "if (resident.texture == 0 || resident.height < tail.height || resident.staticTexels != prepared.texels.size())", "resident upload")
require(upload, "resident.tail = after;", "resident upload")
if upload.count("glTexImage2D(") != 1:
    errors.append("resident upload must allocate with exactly one glTexImage2D")
render = region(backend, "const bool residentQueries = runtimeInputs.previewUse && !roughGlassFrame && !diffractionFrame;",
                "gl.Uniform1i(locations.glassTexture, 6);", "OpenGL render queries")
resident = region(render, "if (residentQueries)", "        else\n        {", "resident render")
for needle in ("opaquevisibility::frameTail(resources->opaqueVisibility,",
               "opaquevisibility::frameTail(resources->smoothGlass.query,",
               "updateResidentQuery(gl, resources->residentOpaque,",
               "updateResidentQuery(gl, resources->residentGlass,",
               "runResidentRefit(gl, refit, resources->residentGlass.texture, resources->vertexBuffer,",
               "gl.UseProgram(sceneProgram);"):
    require(resident, needle, "resident render")
for needle in ("opaquevisibility::frame(", "glGenTextures(1, &frame->", "frame->opaqueTexture", "frame->glassTexture"):
    forbid(resident, needle, "resident render")
# Allocation of the one-row empty texture happens once per resource lifetime.
if resident.count("glTexImage2D(") != 1 or "if (resources->residentEmptyTexture == 0)" not in resident:
    errors.append("resident render may only allocate the empty query texture once")
require(render, "if (!residentQueries && resources->residentRefit.vertexRevision != 0)", "OpenGL render queries")
# Export keeps the released whole-frame upload.
export = render[render.find("        else\n        {"):]
for needle in ("!opaquevisibility::frame(resources->opaqueVisibility,",
               "if (!opaquevisibility::frame(resources->smoothGlass.query,",
               "glGenTextures(1, &frame->opaqueTexture);",
               "glGenTextures(1, &frame->glassTexture);"):
    require(export, needle, "export query upload")

# The GPU refit keeps topology: every write keeps the texel w.
refit = region(backend, "constexpr const char* kResidentRefitComputeSource =", "bool planResidentRefit(", "refit shader")
for needle in ("layout(rgba32f,binding=0) coherent uniform image2D uQuery;",
               "imageStore(uQuery,at(node),vec4(low,imageLoad(uQuery,at(node)).w));",
               "imageStore(uQuery,at(c.x),vec4(finitePoint(v),imageLoad(uQuery,at(c.x)).w));",
               "return any(isnan(p))||any(isinf(p))?vec3(0.0):p;}",
               "vec3 q=finitePoint(c[k]*16u);",
               "for(uint k=(uEnd-uRoot)/2u;k>0u;--k){"):
    require(refit, needle, "refit shader")
run = region(backend, "bool runResidentRefit(", "#endif", "refit dispatch")
for needle in ("gl.BindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);",
               "gl.MemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);",
               "gl.DispatchCompute("):
    require(run, needle, "refit dispatch")
forbid(run, "glTexImage2D", "refit dispatch")
forbid(run, "glReadPixels", "refit dispatch")
forbid(run, "glGetTexImage", "refit dispatch")

# The tail mirrors frame() without copying the static texels.
tail = region(visibility, "inline bool frameTail(", "// Float companion", "frame tail")
require(visibility, "inline std::size_t instanceTree(", "fixture_opaque_visibility.h")
require(tail, "const auto root = instanceTree(candidate.texels, base, instances, 0, instances.size());", "frame tail")
forbid(tail, "prepared.texels.begin()", "frame tail")

# Preview glass caustics beside the refreshed water keep their schedule resident.
refit_shader = refit
for needle in ("layout(std430,binding=4) writeonly buffer RefitTriangleTable{vec4 triangleTexels[];};",
               "layout(std430,binding=5) readonly buffer RefitTriangleCorners{uvec4 triangleCorners[];};",
               "if(uMode==3u){",
               "for(int k=0;k<3;++k)triangleTexels[c.w+uint(k)]=vec4(p[k],area);",
               "triangleTexels[c.w+3u]=vec4(l>0.0?n/l:vec3(0.0),0.0);"):
    require(refit_shader, needle, "refit shader caustic rows")
rows = region(backend, "bool runResidentCausticTriangles(", "#endif", "caustic triangle refresh")
for needle in ("glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);", "gl.UseProgram(unsigned(previousProgram));",
               "gl.Uniform1ui(plan.locations[0], 3u);", "gl.Uniform4fv(plan.locations[7], 4, world.data());",
               "gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);", "gl.DispatchCompute("):
    require(rows, needle, "caustic triangle refresh")
for needle in ("glTexImage2D", "BufferData", "MapBuffer", "glGetTexImage", "glReadPixels"):
    forbid(rows, needle, "caustic triangle refresh")
inputs = region(backend, "bool residentCausticInputsMatch(", "} // namespace", "resident caustic inputs")
for needle in ("gc::selectCamera(probe,&runtime,camera,cameraIdentity,error)", "cameraIdentity!=resident.frame.cameraIdentity",
               "effectiveNativeLightRecord(*scene,runtime,", "identity.compare(prefix,std::string::npos,light)==0"):
    require(inputs, needle, "resident caustic inputs")
# The match reads only the camera and lights: it copies no scene and builds nothing.
for needle in ("make_shared<Visual3DScene>", "selectActiveSource", "gc::sourceInputs(", "encodeScene", "buildScheduledFrame"):
    forbid(inputs, needle, "resident caustic inputs")
caustic = region(backend, "NativeFixtureSceneSubmission renderOpenGlGlassCausticScene(",
                 "bool readOpenGlGlassCausticReceiptForFixture(", "caustic render")
for needle in ("const bool residentEligible=preview&&!multipleScatterPolicy&&waterRefit.planned",
               "&&residentCausticInputsMatch(residentState,scene,runtime);",
               "gc::Prepared& prepared=residentCaustic?residentState.prepared:builtPrepared;",
               "gc::Frame& scheduled=residentCaustic?residentState.frame:builtScheduled;",
               "if(!residentCaustic&&!gc::prepare(resources->smoothGlass,source,policy,prepared,refused.error))return refused;",
               "const auto water=\":resident-water:\"+std::to_string(waterRefit.vertexRevision);",
               "residentTables=residentCaustic||(establishCaustic&&establishResidentCaustic());",
               "if(residentCaustic&&residentState.revision!=waterRefit.vertexRevision)",
               "runResidentCausticTriangles(gl,residentState.refit,resources->vertexBuffer,residentState.triangleBuffer,",
               "runResidentRefit(gl,residentState.refit,residentState.query.texture,resources->vertexBuffer,",
               "if(!residentCaustic||(i!=3&&i!=4))",
               "frame->causticTextures[0]=residentState.query.texture;",
               "copyBuffer(GL_COPY_READ_BUFFER,GL_SHADER_STORAGE_BUFFER,0,0,std::ptrdiff_t(wordBytes));",
               "if(establishCaustic)\n                residentState.copyBufferSubData=residentTables?glfwGetProcAddress(\"glCopyBufferSubData\"):nullptr;",
               "?reinterpret_cast<PFNGLCOPYBUFFERSUBDATAPROC>(residentState.copyBufferSubData):nullptr;",
               "if(residentCaustic&&scheduled.identity==residentState.singlePixelIdentity)",
               "residentState.prepared=std::move(builtPrepared);residentState.frame=std::move(builtScheduled);"):
    require(caustic, needle, "caustic render")
# A resident frame neither prepares nor builds: the build and the Preview fit run
# only in the branch that has no resident schedule.
fit = region(caustic, "if(residentCaustic)reducedReason=residentState.reducedReason;", "if(scheduled.sourceDirectionalIrradiance.version)", "caustic schedule")
if not fit.lstrip().startswith("if(residentCaustic)reducedReason=residentState.reducedReason;\n    else\n    {\n        if(!build(scheduling,scheduled,refused.error))return refused;"):
    errors.append("caustic schedule must build only when no resident schedule is reused")
# Lent resident names go back before every frame release, so the frame never deletes them.
lend = region(caustic, "const auto returnResidentTables=[&] {", "cleanup();frame->releaseTexturesUnlocked();", "caustic lend")
require(lend, "frame->causticBuffers[3]=0;frame->causticBuffers[4]=0;frame->causticTextures[0]=0;", "caustic lend")
require(lend, "returnResidentTables();", "caustic lend")
tail_end = caustic[caustic.rfind("result.stats.drawCount+="):]
require(tail_end, "returnResidentTables();", "caustic render end")
# Export keeps the whole per-frame table and query upload.
for needle in ("gl.BufferData(GL_SHADER_STORAGE_BUFFER,std::ptrdiff_t(bytes[i]),input,(i<3||i>=5)?GL_DYNAMIC_COPY:GL_STATIC_DRAW);",
               "if(i==0)glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,opaquevisibility::kTextureWidth,scheduled.query.height,0,GL_RGBA,GL_FLOAT,scheduled.query.texels.data());",
               "?gc::buildMultipleScatterFrame(prepared,instances,&runtime,width,height,description,*multipleScatterPolicy,output,error)",
               ":gc::buildScheduledFrame(prepared,instances,&runtime,width,height,description,output,error);"):
    require(caustic, needle, "caustic export upload")

require(cmake, "tests/preview_resident_scene_source_contract.py", "CMakeLists.txt")

if errors:
    for error in errors:
        print("FAIL: " + error)
    sys.exit(1)
print("preview resident scene source contract: PASS")
