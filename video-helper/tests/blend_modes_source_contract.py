#!/usr/bin/env python3
"""Check the persisted blend vocabulary and both GPU implementations."""

from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
defs = (root / "src/effect_defs.h").read_text()
gl = (root / "src/renderer.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
ui = (root.parent / "plugin/Source/VideoEditor/ClipPropertiesPanel.h").read_text()
protocol = (root / "PROTOCOL.md").read_text()

modes = [
    ("Normal", 0, "Normal"),
    ("Add", 1, "Add"),
    ("Multiply", 2, "Multiply"),
    ("Screen", 3, "Screen"),
    ("Overlay", 4, "Overlay"),
    ("Difference", 5, "Difference"),
    ("Exclusion", 6, "Exclusion"),
    ("Darken", 7, "Darken"),
    ("Lighten", 8, "Lighten"),
    ("ColorDodge", 9, "Color Dodge"),
    ("ColorBurn", 10, "Color Burn"),
    ("SoftLight", 11, "Soft Light"),
    ("HardLight", 12, "Hard Light"),
]

for symbol, ordinal, label in modes:
    assert re.search(rf"\b{symbol}\s*=\s*{ordinal}\b", defs), (symbol, ordinal)
    assert f'"{label}"' in defs, label
assert "Count = 13" in defs

# The clip inspector projects this append-only table rather than keeping a
# second UI vocabulary.
assert "videofx::BlendMode::Count" in ui
assert "videofx::kBlendModeNames[i]" in ui

# OpenGL owns one compositor implementation. Metal owns the ordinary and
# transition compositor programs, both of which must dispatch all modes.
def shader_source(name: str) -> str:
    match = re.search(rf'const char\* {name} = R"metal\((.*?)\)metal";', metal, re.S)
    assert match is not None, name
    return match.group(1)

metal_blend = shader_source("kMetalBlendFragment")
metal_transition = shader_source("kMetalTransitionFragment")
for ordinal in range(1, 13):
    token = f"if (mode == {ordinal})"
    assert gl.count(token) == 1, ("OpenGL", token, gl.count(token))
    assert metal_blend.count(token) == 1, ("Metal blend", token, metal_blend.count(token))
    assert metal_transition.count(token) == 1, (
        "Metal transition", token, metal_transition.count(token)
    )
assert gl.count("applyBlendMode(back.rgb, front.rgb, uBlendMode)") == 1
assert metal_blend.count("applyBlendMode(back.rgb, front.rgb, p.mode)") == 1
assert metal_transition.count("applyBlendMode(back.rgb, front.rgb, p.blendMode)") == 1

for helper in ("blendColorDodge", "blendColorBurn", "blendSoftLight", "blendHardLight"):
    assert helper in gl, helper
for helper in ("colorDodge", "colorBurn", "softLight", "hardLight"):
    assert helper in metal_blend and helper in metal_transition, helper

# Pin the divide-by-zero guards and the W3C soft-light transfer curve.
for source, suffix in ((gl, ""), (metal_blend, "f"), (metal_transition, "f")):
    assert f"base[i] <= 0.0{suffix}" in source
    assert f"blend[i] >= 1.0{suffix}" in source
    assert f"base[i] / (1.0{suffix} - blend[i])" in source
    assert f"base[i] >= 1.0{suffix}" in source
    assert f"blend[i] <= 0.0{suffix}" in source
    assert f"(1.0{suffix} - base[i]) / blend[i]" in source
    assert f"blend[i] <= 0.5{suffix}" in source
    assert f"base[i] <= 0.25{suffix}" in source
    assert "sqrt(base[i])" in source

for ordinal, label in ((5, "difference"), (6, "exclusion"), (7, "darken"),
                       (8, "lighten"), (9, "color dodge"), (10, "color burn"),
                       (11, "soft light"), (12, "hard light")):
    assert f"{ordinal} {label}" in protocol

print("blend mode source contract: 13 pinned modes, UI projection, OpenGL, and Metal parity")
