#!/usr/bin/env python3
# Contract for the Geometry Core Preview prepare. Only the viewport asks for Preview.
# A pending Preview prepare defers the frame with no plan error, and the retained
# cache and the event cursor stay as they were. A pending from a blocking caller is
# still an error, and the Metal backend accepts the use and never defers.
# Preview callers: the viewport context, Harmonic Link Geometry and Solid 3D. Each of these
# passes its use through. Known gap, not pinned here: the deformation backend
# (NativeDeformationBackend::prepare) has no use argument, so imported animation preview
# still prepares its fixture program blocking. It needs a backend.h change to close.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend.h").read_text()
gl = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
sokol = (root / "src/gpu_backend/backend_sokol.mm").read_text()
particle_fixture = (root / "tests/particle_points_fixture.h").read_text()
core_header = (root / "src/geometry_core_backend.h").read_text()
core = (root / "src/geometry_core_backend.cpp").read_text()
executor = (root / "src/visual_plan_executor.h").read_text()
viewport = (root / "src/viewport.cpp").read_text()
exporter = (root / "src/exporter.cpp").read_text()
renderer = (root / "src/renderer.cpp").read_text()
cmake = (root / "CMakeLists.txt").read_text()
harmonic = (root / "src/harmonic_link_geometry.h").read_text()
particle_solid = (root / "src/particle_solid_renderer.h").read_text()

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


def ordered(text, needles, label):
    # Each needle must appear, and each must follow the one before it by text order.
    at = -1
    for needle in needles:
        found = text.find(needle, at + 1)
        if found < 0:
            errors.append(label + " is missing, or out of order, " + needle)
            return
        at = found


# Seam: the instanced prepare takes the use and forwards it. Metal accepts and ignores it.
require(backend, "return prepare (scene, materialProgram, use);", "backend.h prepareGeometryInstances")
require(gl, "NativeFixturePrepareUse use = NativeFixturePrepareUse::Blocking) override", "backend_opengl.cpp override")
require(gl, "diagnosticInstanceIdentityColors,frameAttributes,nullptr,{},0,use);", "backend_opengl.cpp forwarding")
require(sokol, "NativeFixturePrepareUse = NativeFixturePrepareUse::Blocking) override", "backend_sokol.mm override")
require(particle_fixture, "prepareGeometryInstances(scene,material,admission,diagnostic,attributes,use)",
        "particle_points_fixture.h wrapper")

# Geometry Core: the prepare takes the use, pending returns before render, and only
# Preview defers. A pending from a blocking prepare stays an error.
require(core_header, "arbitgpu::NativeFixturePrepareUse prepareUse = arbitgpu::NativeFixturePrepareUse::Blocking);",
        "geometry_core_backend.h executeNativeGeometry")
require(core_header, "std::uint64_t geometryCorePreviewPendingResultsOnThisThread() noexcept;",
        "geometry_core_backend.h accessor")
require(core, "thread_local std::uint64_t geometryCorePreviewPendingResults = 0;", "geometry_core_backend.cpp counter")
require(core, "std::uint64_t geometryCorePreviewPendingResultsOnThisThread() noexcept {",
        "geometry_core_backend.cpp accessor")
prepare = region(core, "auto prepared = instances != nullptr",
                 "if (!prepared.prepared || !prepared.resources)", "geometry_core_backend.cpp prepare")
require(prepare, "backend.prepare(scene, surfaceProgram, prepareUse);", "geometry_core_backend.cpp prepare")
forbid(prepare, "backend.prepare(scene, surfaceProgram);", "geometry_core_backend.cpp prepare")
require(prepare, "if (prepareUse != arbitgpu::NativeFixturePrepareUse::Preview) {",
        "geometry_core_backend.cpp pending guard")
require(prepare, "returned pending for a blocking prepare", "geometry_core_backend.cpp blocking pending error")
require(prepare, "++geometryCorePreviewPendingResults;", "geometry_core_backend.cpp pending count")
ordered(core, ["if (prepared.pending) {", "if (!prepared.prepared || !prepared.resources) {",
               "backend.render(scene, prepared.resources"], "geometry_core_backend.cpp pending before render")

# Admission: only the viewport context asks for Preview. Displaced retained entries
# come back on a pending result, and the event cursor is consumed after admission.
require(executor, "arbitgpu::NativeFixturePrepareUse geometryPrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking;",
        "visual_plan_executor.h context default")
forbid(executor, "geometryPrepareUse = arbitgpu::NativeFixturePrepareUse::Preview", "visual_plan_executor.h")
require(executor, "&materialContext, true, linearColor))", "visual_plan_executor.h context copy")
forbid(executor, "retained.owners.erase(existing);", "visual_plan_executor.h existing entry")
forbid(executor, "retained.owners.erase(victim);", "visual_plan_executor.h victim entry")
ordered(executor, [
    "std::vector<typename decltype(retained.owners)::node_type> displaced;",
    "displaced.push_back(retained.owners.extract(existing));",
    "displaced.push_back(retained.owners.extract(victim));",
    "const auto pendingBefore=videohelper::geometry::geometryCorePreviewPendingResultsOnThisThread();",
    "auto execution = videohelper::geometry::executeNativeGeometry(",
    "{},{},evaluationContext->geometryPrepareUse);",
    "const bool previewDeferred=",
    "retained.owners.erase(owner);",
    "if(previewDeferred)",
    "for(auto& node : displaced) (void) retained.owners.insert(std::move(node));",
], "visual_plan_executor.h admission pending order")
ordered(executor, [
    "if (! admitGeometryCoreOperations(*plan, geometryUse, evaluationContext",
    "if (eventSchedules != nullptr && eventCursor != nullptr)",
], "visual_plan_executor.h event consumption after admission")

# Viewport: only the viewport asks for Preview, and a pending result defers the frame.
require(viewport, "videowire::VisualPlanEvaluationContext visualEvaluationContext {", "viewport.cpp context")
forbid(viewport, "const videowire::VisualPlanEvaluationContext visualEvaluationContext", "viewport.cpp context")
ordered(viewport, [
    "videowire::VisualPlanEvaluationContext visualEvaluationContext {",
    "visualEvaluationContext.geometryPrepareUse = arbitgpu::NativeFixturePrepareUse::Preview;",
    "videowire::executeVisualLayerPlanForRenderer",
], "viewport.cpp Preview assignment before the first plan execution")
require(viewport, "const auto geometryPendingAtFrameStart = videohelper::geometry::geometryCorePreviewPendingResultsOnThisThread();",
        "viewport.cpp frame snapshot")
hold = region(viewport, "const auto holdFrameForPlanError = [&](const std::string& error)",
              "const auto prepareShaderClip", "viewport.cpp holdFrameForPlanError")
ordered(hold, [
    "if (geometryPendingThisFrame())",
    "deferredSource = \"geometry-core\";",
    "frameDeferred = true;",
    "return;",
    "planFrameError = error.empty()",
    "im.rendererError = planFrameError;",
], "viewport.cpp pending deferral before any plan error")
rejected = "visual plan execution rejected"
at = viewport.find(rejected)
logs = 0
while at >= 0:
    logs += 1
    guard = viewport.rfind("if (!geometryPendingThisFrame())", 0, at)
    if guard < 0 or at - guard > 120:
        errors.append("viewport.cpp rejected log must sit behind geometryPendingThisFrame")
    at = viewport.find(rejected, at + 1)
if logs != 2:
    errors.append("viewport.cpp must have two rejected logs, found " + str(logs))

# Other callers keep the blocking default. Only the viewport context sets Preview.
forbid(exporter, "geometryPrepareUse", "exporter.cpp")
forbid(renderer, "geometryPrepareUse", "renderer.cpp")

# Harmonic Link Geometry: the preview-time render takes the context's use. It has no default,
# so a new caller must choose. Its Geometry Core prepare receives that same use.
require(harmonic, "LayerDesc& layer, std::string& error, bool linearColor,\n                   arbitgpu::NativeFixturePrepareUse prepareUse)",
        "harmonic_link_geometry.h render signature")
require(harmonic, "nullptr, std::numeric_limits<double>::quiet_NaN(), nullptr, {}, {}, prepareUse);",
        "harmonic_link_geometry.h executeNativeGeometry use")
harmonic_call = region(executor, "videohelper::harmonicgeometry::render(", ");",
                       "visual_plan_executor.h harmonic render call")
require(harmonic_call, "evaluationContext->geometryPrepareUse", "visual_plan_executor.h harmonic render call")
forbid(harmonic, "error, linearColor = false)", "harmonic_link_geometry.h render")

# Solid 3D: an export blocks, a Preview defers. The prepare passes the use and never
# relies on the default. A pending Preview counts as a deferral; a pending export is an error.
require(particle_solid, "const bool exportUse=owner.use==videohelper::geometry::PlanUse::exportRender;",
        "particle_solid_renderer.h export owner")
ordered(particle_solid, [
    "const auto prepareUse=exportUse ? arbitgpu::NativeFixturePrepareUse::Blocking",
    ": arbitgpu::NativeFixturePrepareUse::Preview;",
    "auto prepared=backend.prepare(scene,{},prepareUse);",
    "if (prepared.pending) {",
    "if (exportUse) { error=\"Solid 3D export preparation returned pending\"; return false; }",
    "videohelper::geometry::noteGeometryCorePreviewPendingOnThisThread();",
    "if (!prepared.prepared || !prepared.resources) { error=prepared.error; return false; }",
], "particle_solid_renderer.h prepare use and pending order")
forbid(particle_solid, "backend.prepare(scene);", "particle_solid_renderer.h prepare")
require(core_header, "void noteGeometryCorePreviewPendingOnThisThread() noexcept;",
        "geometry_core_backend.h pending note")
require(core, "void noteGeometryCorePreviewPendingOnThisThread() noexcept {\n  ++geometryCorePreviewPendingResults;\n}",
        "geometry_core_backend.cpp pending note")

# Viewport: every frame deferral names its source, so "other" never hides a wait. The
# optical and motion-view deferrals and the spectrum bake wait are labelled; a plan error
# hold is "plan-error" and never a GL wait. The decode path inherits the spectrum label.
for label, count in (('deferredSource = "plan-error";', 1), ('deferredSource = "spectrum-bake";', 2),
                     ('deferredSource = "motion-view-rejected";', 1), ('deferredSource = "optical-field";', 1)):
    if viewport.count(label) != count:
        errors.append("viewport.cpp must have " + str(count) + " of " + label)
at = viewport.find("frameDeferred = true;")
while at >= 0:
    window = viewport[max(0, at - 260):at + 260]
    if ("deferredSource =" not in window and "spectrumBakePending || frameDeferred" not in window):
        errors.append("viewport.cpp frame deferral without a source label near offset " + str(at))
    at = viewport.find("frameDeferred = true;", at + 1)
require(viewport, 'std::fprintf (stderr, "[viewport] spectrum-bake cause: %s\\n"',
        "viewport.cpp spectrum bake cause log")
require(hold, "deferredSource = \"plan-error\";", "viewport.cpp holdFrameForPlanError plan-error label")

require(cmake, "add_test(NAME geometry_core_preview_prepare_source_contract", "CMakeLists.txt")
require(cmake, "tests/geometry_core_preview_prepare_source_contract.py", "CMakeLists.txt")

if errors:
    print("geometry core preview prepare contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("geometry core preview prepare source contract: PASS")
