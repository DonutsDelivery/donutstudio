#!/usr/bin/env python3
from pathlib import Path
import hashlib
import re
import sys


root = Path(sys.argv[1])
repo = root.parent
gl = (root / "src/renderer.cpp").read_text(encoding="utf-8")
renderer_header = (root / "src/renderer.h").read_text(encoding="utf-8")
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")
fixture = (root / "tests/custom_shader_gl_pixel_tests.cpp").read_text(encoding="utf-8")
starter_registry = (repo / "plugin/Source/graph/BuiltInVisualModules.h").read_text(encoding="utf-8")
transition_contract = (repo / "shared/CuratedShaderTransitionContract.h").read_text(encoding="utf-8")

# Flat generators and filters have one authority: the immutable ordered plan.
# The separate curated transition carrier remains for the transition fallback.
for stale_type in ("ImmutableCuratedFlatShader", "ImmutableCatalogIsfFilter"):
    assert stale_type not in renderer_header
assert "ImmutableShaderOperationPlan shaderOperationPlan" in renderer_header
assert "ImmutableCuratedShaderTransition curatedShaderTransition" in renderer_header
assert "FlatShaderBridge requires an immutable shader operation plan" in gl

gl_admission = gl[
    gl.index("bool FrameRenderer::prepareFlatShaderBridge"):
    gl.index("std::map<FlatShaderKey", gl.index("bool FrameRenderer::prepareFlatShaderBridge"))
]
metal_start = metal.index("bool MetalFrameRenderer::prepareShaderOperationPlan")
metal_allocation = metal.index(
    "std::map<std::string, std::unique_ptr<MetalShaderGenerator>> staged", metal_start
)
metal_admission = metal[metal_start:metal_allocation]

payload_guards = (
    "|| !operation.payload.catalogPackId.empty()",
    "|| !operation.payload.catalogProgramId.empty()",
)
for guard in payload_guards:
    assert guard in gl_admission
    assert guard in metal_admission

# Strict Metal must reject forged curated identity while it still validates the
# source-bound custom grant, before staging or compiling any renderer resource.
assert "programmableruntime::admits(*operation.customGrant, kind," in metal_admission
assert "operation.payload.source, errorOut)" in metal_admission
assert "Metal custom grant no longer owns the exact payload" in metal_admission
for allocation in (
    "std::make_unique<MetalShaderGenerator>()",
    "generator->setSource(operation.payload.source)",
    "shadercatalog::find(operation.payload.catalogPackId",
):
    assert metal.index(allocation, metal_start) > metal_allocation

# The shared behavioral fixture injects curated-looking fields into a valid
# custom payload, recomputes the complete digest, and runs under both backends.
for marker in (
    'payload.catalogPackId = "vidvox-isf"',
    'payload.catalogProgramId = "rgb-invert"',
    "forgedCatalogPlan->digest = videowire::shaderOperationPlanDigest(",
    "!preview.hasPreparedShaderPlan(32)",
    "ARBIT_STRICT_METAL_FIXTURE",
):
    assert marker in fixture

# The named collection has exactly three source-backed entries. Their canonical
# preview graph must preserve the helper's distinct start/end resource contract,
# and the native fixture must execute each exact bundled source against an
# independently coded CPU result under isolated preview, export, and reconstructed owners.
for module_id, program_id, file_name in (
    ("visual.starter.transition.fade", "fade", "Fade.fs"),
    ("visual.starter.transition.directional-wipe", "directional-wipe", "Directional Wipe.fs"),
    ("visual.starter.transition.cross-zoom", "crosszoom", "CrossZoom.fs"),
):
    assert starter_registry.count(f'"{module_id}"') == 1
    assert f'"{file_name}"' in fixture
    assert f'{{ "{program_id}",' in fixture
    source = repo / "content/shader-packs/vidvox-isf/ISF" / file_name
    assert source.is_file() and source.stat().st_size > 0
    identity = re.search(
        rf'\{{ "vidvox-isf", "{re.escape(program_id)}", "([0-9a-f]{{64}})" \}}',
        transition_contract,
    )
    assert identity is not None
    assert hashlib.sha256(source.read_bytes()).hexdigest() == identity.group(1)

for marker in (
    "NodeKindId { kinds::VideoLayerSource }",
    'port.name == "from"',
    "starter transition OpenGL pixels match the independent CPU oracle",
    "starter transition export owner matches preview pixels",
    "isolated reconstructed owner reproduces immutable transition state",
    "starter transition keeps spatially nonuniform output pixels",
    "Cross Zoom pixels reject an ignored-strength implementation",
    "Cross Zoom pixels reject a no-displacement implementation",
    "Cross Zoom pixels reject a plain dissolve implementation",
    'const double strength = crossZoom ? 0.68 : 0.0',
    "for (int sample = 0; sample <= 40; ++sample)",
    'programId == "directional-wipe"',
    'programId == "crosszoom"',
):
    assert marker in (starter_registry + fixture)
