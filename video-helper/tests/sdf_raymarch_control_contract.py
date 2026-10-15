#!/usr/bin/env python3
# Source contract for the SDF raymarch renderer controls. It pins the shared
# ceilings, graph parameter bounds, wire admission, backend capabilities and
# uniform wiring. These are source checks, not pixel or runtime evidence.
from pathlib import Path
import re
import sys


root = Path(sys.argv[1]).resolve()
repo = root.parent
shared = (repo / "shared/SdfRaymarchOperationContract.h").read_text()
nodes = (repo / "plugin/Source/graph/SdfGraphNodes.h").read_text()
lowering = (repo / "plugin/Source/VideoVisualGraph.cpp").read_text()
execution = (root / "src/sdf_visual_plan_execution.h").read_text()
gl = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()

CONTROLS = ("maximumSteps", "epsilon", "maximumDistance", "adaptiveQuality",
            "normalQuality", "shadowQuality", "outputPass")


def number(text, pattern):
    match = re.search(pattern, text)
    assert match, pattern
    return float(match.group(1))


# 1. Shared ceilings, the single source for graph, wire and backends.
ceilings = {
    "kSdfRaymarchMaximumSteps": 512,
    "kSdfRaymarchMaximumQuality": 3,
    "kSdfRaymarchMaximumOutputPass": 7,
    "kSdfRaymarchMinimumEpsilon": 0.000001,
    "kSdfRaymarchMaximumEpsilon": 0.1,
    "kSdfRaymarchMaximumDistance": 1000.0,
}
for name, value in ceilings.items():
    assert number(shared, rf"inline constexpr \S+ {name} = ([0-9.]+);") == value, name

# 2. The shared wire contract refuses out-of-range controls on encode and decode.
assert "! sdfRaymarchControlsAdmissible (operation)" in shared, "encode refusal"
assert "! sdfRaymarchControlsAdmissible (candidate)" in shared, "decode refusal"

# 3. Graph lowering: parameter table, ParamSpec ceilings and wire copy.
parse_start = nodes.index("inline bool parseRaymarchParameters")
parse_region = nodes[parse_start:nodes.index("std::array<double, parameters.size()> values", parse_start)]
for control in CONTROLS:
    assert f'{{ "{control}", ' in parse_region, f"parser lacks {control}"
assert "videowire::kSdfRaymarchMaximumEpsilon" in parse_region, "epsilon ceiling"
assert "videowire::kSdfRaymarchMaximumDistance" in parse_region, "distance ceiling"
assert "videowire::kSdfRaymarchMaximumQuality" in parse_region, "quality ceiling"
assert "videowire::kSdfRaymarchMaximumOutputPass" in parse_region, "output ceiling"

terminal_start = nodes.index("terminal.params = {")
terminal_region = nodes[terminal_start:nodes.index("};", terminal_start)]
assert "static_cast<float> (videowire::kSdfRaymarchMaximumEpsilon)" in terminal_region, "epsilon ParamSpec"
assert "static_cast<float> (videowire::kSdfRaymarchMaximumDistance)" in terminal_region, "distance ParamSpec"
assert "1.0f" not in terminal_region.split('ParamSpec { "epsilon"')[1].split("\n")[0], "stale epsilon literal"

for control in CONTROLS:
    pattern = rf"operation\.{control} = [^\n;]*request\.{control}\b"
    assert re.search(pattern, lowering), f"lowering does not copy {control}"

# 4. Native execution copies every wire control into the renderer request.
for control in ("maximumSteps", "epsilon", "maximumDistance"):
    assert re.search(rf"controls\.{control} = operation\.{control};", execution), control
for control, enum in (("adaptiveQuality", "NativeSdfQuality"), ("normalQuality", "NativeSdfQuality"),
                      ("shadowQuality", "NativeSdfQuality"), ("outputPass", "NativeSdfOutput")):
    assert re.search(rf"static_cast<arbitgpu::{enum}>\s*\(operation\.{control}\)", execution), control

# 5. Both native backends advertise exactly the shared ceilings and refuse
#    over-limit controls before frame allocation.
for label, text in (("OpenGL", gl), ("Metal", metal)):
    assert number(text, r"result\.maxSteps = ([0-9]+);") == ceilings["kSdfRaymarchMaximumSteps"], label
    assert number(text, r"result\.minEpsilon = ([0-9.]+);") == ceilings["kSdfRaymarchMinimumEpsilon"], label
    assert number(text, r"result\.maxEpsilon = ([0-9.]+);") == ceilings["kSdfRaymarchMaximumEpsilon"], label
    assert number(text, r"result\.maxDistance = ([0-9.]+);") == ceilings["kSdfRaymarchMaximumDistance"], label
    for guard in ("request.maximumSteps > available.maxSteps",
                  "request.epsilon > available.maxEpsilon",
                  "request.maximumDistance > available.maxDistance",
                  "! validQuality (request.adaptiveQuality)",
                  "! validQuality (request.normalQuality)",
                  "! validQuality (request.shadowQuality)",
                  "! available.supports (request.output)"):
        assert guard in text, f"{label} lacks guard: {guard}"

# 6. Uniform wiring. OpenGL binds named uniforms that the shader declares.
gl_uniforms = {
    "uEpsilon": ("float", "static_cast<float> (request.epsilon)"),
    "uMaximumDistance": ("float", "static_cast<float> (request.maximumDistance)"),
    "uMaximumSteps": ("int", "static_cast<int> (request.maximumSteps)"),
    "uAdaptiveQuality": ("int", "static_cast<int> (request.adaptiveQuality)"),
    "uNormalQuality": ("int", "static_cast<int> (request.normalQuality)"),
    "uShadowQuality": ("int", "static_cast<int> (request.shadowQuality)"),
    "uOutputPass": ("int", "static_cast<int> (request.output)"),
}
for uniform, (kind, value) in gl_uniforms.items():
    assert re.search(rf'GetUniformLocation \(program, "{uniform}"\),\s*{re.escape(value)}', gl), uniform
    assert re.search(rf"^uniform {kind} {uniform};", gl, re.M), f"shader declaration {uniform}"

# Metal packs the same controls into its raymarch, quality and program blocks.
metal_slots = {
    "uniforms.raymarch[0] = static_cast<float> (request.epsilon);",
    "uniforms.raymarch[1] = static_cast<float> (request.maximumDistance);",
    "uniforms.quality[0] = request.maximumSteps;",
    "uniforms.quality[1] = static_cast<std::uint32_t> (request.normalQuality);",
    "uniforms.quality[2] = static_cast<std::uint32_t> (request.shadowQuality);",
    "uniforms.quality[3] = static_cast<std::uint32_t> (request.adaptiveQuality);",
    "uniforms.program[0] = static_cast<std::uint32_t> (request.output);",
}
for statement in metal_slots:
    assert statement in metal, statement

print("sdf raymarch control contract passed")
