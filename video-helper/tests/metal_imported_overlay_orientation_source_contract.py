#!/usr/bin/env python3
"""Pin the row order of the Metal imported particle overlay blend.

The overlay blends the particle image (front) and the retained Render 3D image
(back) before the ordinary layer pass, so both take one more inversion than a
normal layer. The particle image already lands where OpenGL has it with that
extra inversion: MetalParticleEngine draws pos.y = 0 into the last row, the
opposite of the OpenGL engine's row zero. The Render 3D image is a top-first
Metal texture, so it must be sampled with v flipped to undo the extra inversion.
Flipping the particle input, or changing the pass for both inputs, moves the
particles off the model.
"""

from pathlib import Path
import re
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
gl_particles = (root / "src/particle_engine.cpp").read_text()


def shader_source(name: str) -> str:
    match = re.search(rf'const char\* {name} = R"metal\((.*?)\)metal";', metal, re.S)
    assert match is not None, name
    return match.group(1)


# Particle rows: both engines map pos.y to NDC y = pos.y * 2 - 1 with no flip, so
# Metal row order for the particle image is fixed by the draw pass alone.
assert "out.position = float4(p.pos * 2.0f - 1.0f, 0.0f, 1.0f);" in metal
assert "gl_Position  = vec4(pt.pos.x * 2.0 - 1.0, pt.pos.y * 2.0 - 1.0, 0.0, 1.0);" in gl_particles

# Scene rows: the retained Metal Render 3D image is declared top-first.
assert re.search(r"descriptor\.rowOrder = arbitgpu::NativeTextureRowOrder::TopFirst;", metal)

# The overlay fragment leaves the front (particle) input at in.uv and samples the
# back (scene) input with v flipped.
fragment = shader_source("kMetalOverlayBlendFragment")
assert "frontTex.sample(imageSampler, in.uv)" in fragment
assert "backTex.sample(imageSampler, float2(in.uv.x, 1.0f - in.uv.y))" in fragment
assert fragment.count("1.0f - in.uv.y") == 1

# The ordinary blend still samples both inputs identically.
ordinary = shader_source("kMetalBlendFragment")
assert "frontTex.sample(imageSampler, in.uv)" in ordinary
assert "backTex.sample(imageSampler, in.uv)" in ordinary

# The overlay program uses the usual fullscreen vertex with that fragment.
program = re.search(
    r"blendDesc\.fragment_func\.source = kMetalOverlayBlendFragment;(.*?)"
    r"overlayBlendPipeline = makeCheckedPipeline", metal, re.S)
assert program is not None, "overlay blend program is missing"
assert "overlayBlendShader = makeCheckedShader (blendDesc)" in program.group(1)
assert "blendPipelineDesc.shader = overlayBlendShader" in program.group(1)
assert "kMetalDirectOutputVertex" not in program.group(1)
assert re.search(
    r"blendDesc\.vertex_func\.source = kMetalFullscreenVertex;\s*"
    r"blendDesc\.fragment_func\.source = kMetalBlendFragment;", metal)

# The overlay branch binds particles as view 0 and the retained scene as view 1,
# and draws with the overlay pipeline.
start = metal.index("if (layer.importedParticleOverlay)\n                    {")
branch = metal[start:]
branch = branch[:branch.index("sourceView = impl_->particleComposite.texture;")]
assert "sg_apply_pipeline(impl_->overlayBlendPipeline);" in branch
assert "impl_->blendPipeline" not in branch
assert "bindings.views[0] = sourceView;" in branch
assert "bindings.views[1].id = static_cast<std::uint32_t>(layer.nativeTextureView);" in branch

# The program lifecycle covers the new objects.
assert "overlayBlendPipeline = {}" in metal
assert "sg_destroy_pipeline (impl_->overlayBlendPipeline)" in metal
assert "sg_destroy_shader (impl_->overlayBlendShader)" in metal
assert "resourceValid (sg_query_pipeline_state (overlayBlendPipeline))" in metal
print("metal imported overlay orientation source contract ok")
