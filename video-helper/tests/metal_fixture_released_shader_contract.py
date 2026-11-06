#!/usr/bin/env python3
"""Check that Metal fixture programs without a diffraction layer or caustic volume
compile the pre-layer shader bytes.

The helper builds every fixture pipeline from source with no app-side cache. Only
Metal's own shader cache saves the compile, and it misses whenever the source bytes
change, so a cold build of the fixture pipelines took 12-13 s on the MacBook and
the Particle Halo row missed its 12 s present window. The diffraction layer and the
paired volume therefore live outside the released text: the layer's _main blocks
are inserted by composeLayeredFixtureMain, and the trailing stamLayer and
glassTransport2 uniforms by the layered and paired declarations. A Metal shader
may declare a prefix of the FixtureUniforms block, so the released fragment
Uniforms end at vertexDisplacementLimits. The vertex Uniforms also declare the
last member, the prior model matrix Motion reads.
"""

from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sokol = (root / "src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")

errors = []


def require(condition, label):
    if not condition:
        errors.append(label)


def raw_string(name):
    match = re.search(r"constexpr char " + name + r'\[\] = R"metal\((.*?)\)metal";', sokol, re.S)
    if match is None:
        errors.append(name + " is missing")
        return ""
    return match.group(1)


def function_body(signature):
    start = sokol.find(signature)
    if start < 0:
        errors.append(signature + " is missing")
        return ""
    end = sokol.find("\n}\n", start)
    return sokol[start:end]


def uniforms_end(text):
    uniforms = re.search(r"struct Uniforms \{\n(.*?)\n\};", text, re.S)
    if uniforms is None:
        return None
    sizes = {"float4x4": 64, "float4": 16, "uint4": 16}
    offset = 0
    for line in uniforms.group(1).splitlines():
        match = re.fullmatch(r"\s*(float4x4|float4|uint4) (.*);", line)
        if match is None:
            errors.append("unparsed Metal Uniforms member: " + line)
            continue
        for name in match.group(2).split(","):
            count = re.search(r"\[(\d+)\]", name)
            offset += sizes[match.group(1)] * (int(count.group(1)) if count else 1)
    return offset


vertex = raw_string("kFixtureVertexShader")
fragment = raw_string("kFixtureFragmentShader")

# 1. The released texts carry no layer or paired-volume token. The fragment
#    Uniforms end at vertexDisplacementLimits. The vertex Uniforms skip the
#    fragment-only tail and end at previousObjectMatrix, the last C++ member,
#    which only Motion reads.
for name, text in (("kFixtureVertexShader", vertex), ("kFixtureFragmentShader", fragment)):
    for token in ("stamLayer", "FIXTURE_STAM_LAYER", "StamLayer", "StamSurface", "glassTransport2"):
        require(token not in text, name + " must not contain " + token)
require(fragment.count("    float4 vertexDisplacementLimits[16];\n};") == 1,
        "kFixtureFragmentShader Uniforms must end at vertexDisplacementLimits")
end = uniforms_end(fragment)
require(end == 11536, "kFixtureFragmentShader Uniforms must end at 11536, found " + str(end))
require(vertex.count("    float4 vertexDisplacementLimits[16];\n    float4 fragmentTail[26];\n"
                     "    float4x4 previousObjectMatrix;\n};") == 1,
        "kFixtureVertexShader Uniforms must end at previousObjectMatrix after the fragment tail")
end = uniforms_end(vertex)
require(end == 12016, "kFixtureVertexShader Uniforms must end at 12016, found " + str(end))
require("&& offsetof (FixtureUniforms, previousObjectMatrix) == 11952" in sokol, "previousObjectMatrix offset")
require("&& sizeof (FixtureUniforms) == 12016," in sokol, "FixtureUniforms size")

# 2. That end is exactly where the C++ block's layer starts, and the layer and
#    partner keep their offsets, so a prefix declaration reads the same bytes.
require("&& offsetof (FixtureUniforms, vertexDisplacementLimits) == 11280" in sokol, "vertexDisplacementLimits offset")
require("&& offsetof (FixtureUniforms, stamLayer) == 11536" in sokol, "stamLayer offset")
require("&& offsetof (FixtureUniforms, glassTransport2) == 11648" in sokol, "glassTransport2 offset")

# 3. The restricted SurfacePbr archive never composes the layer.
recipe = function_body("template<class String> auto restrictedSurfaceSourceRecipe(bool pairedVolume=false,")
for token in ("composeLayeredFixtureMain", "declareLayeredFixtureUniform", "stamlayer"):
    require(token not in recipe, "restrictedSurfaceSourceRecipe must not use " + token)
require("SurfacePbr) && !layered && !media;" in sokol, "layered and media programs must leave the restricted archive")

# 4. Only the general path composes the layer, and only for a layered program,
#    before the modules go in (they could repeat an anchor).
general = ("            fragmentSource=kFixtureFragmentShader;\n"
           "            if(layered && !composeLayeredFixtureMain(fragmentSource)) {\n"
           "                result.error=\"Metal fixture diffraction layer anchor is not unique\";\n"
           "                return result;\n"
           "            }\n"
           "            if(viewportSolidProgram && !composeViewportSolidFixtureMain(fragmentSource)) {\n"
           "                result.error=\"Metal fixture solid viewport anchor is not unique\";\n"
           "                return result;\n"
           "            }\n"
           "            if(viewportSolidProgram && !composeRealtimeFixtureMain(fragmentSource)) {\n"
           "                result.error=\"Metal fixture Realtime anchor is not unique\";\n"
           "                return result;\n"
           "            }\n"
           "            if(media && !composeMediumFixtureMain(fragmentSource)) {\n"
           "                result.error=\"Metal fixture material medium anchor is not unique\";\n"
           "                return result;\n"
           "            }\n"
           "            if(viewportSolidProgram) declareViewportSolidFixtureUniform(fragmentSource);\n"
           "            else if(pairedVolume) declarePairedFixtureUniform(fragmentSource);\n"
           "            else if(layered) declareLayeredFixtureUniform(fragmentSource);\n"
           "            if(media) declareMediumFixtureUniform(fragmentSource);\n"
           "            fragmentSource.insert(")
require(sokol.count(general) == 1, "the general Metal fragment path must gate the layer on layered")
require(sokol.count("composeLayeredFixtureMain(") == 2, "composeLayeredFixtureMain has one caller")
require(sokol.count("declareLayeredFixtureUniform(") == 2, "declareLayeredFixtureUniform has one caller")
require("+(layered ? arbitgpu::stamlayer::metalShader() : std::string())\n" in sokol,
        "the layer module is appended only for a layered program")

# 5. Every layer block has a unique anchor in the released _main and is gated on
#    FIXTURE_STAM_LAYER, which only stamlayer::metalShader() defines.
compose = function_body("inline bool composeLayeredFixtureMain(std::string& fragment)")
pairs = re.findall(r'\{"(.*?)", (kFixtureStamLayer\w+)\}', compose)
require(len(pairs) == 4, "composeLayeredFixtureMain must place four blocks, found " + str(len(pairs)))
layered = fragment
for anchor, name in pairs:
    anchor = anchor.encode().decode("unicode_escape")
    block = raw_string(name)[1:]
    require(layered.count(anchor) == 1, "layer anchor must be unique in the released fragment: " + anchor.strip())
    require(block.startswith("#ifdef FIXTURE_STAM_LAYER\n") and block.endswith("#endif\n"),
            name + " must be gated on FIXTURE_STAM_LAYER")
    at = layered.find(anchor)
    if at >= 0:
        layered = layered[:at + len(anchor)] + block + layered[at + len(anchor):]
require(layered.count("#ifdef FIXTURE_STAM_LAYER") == 4, "the composed layered fragment has four gated blocks")
require("const bool stamLayerOn = u.stamLayer[0].w > 0.5f;" in layered, "the composed fragment reads the enable lane")

# 6. The declarations add the trailing members after the released end.
require('"    float4 vertexDisplacementLimits[16];\\n    float4 stamLayer[7];\\n};"' in sokol,
        "the layered declaration adds stamLayer")
require('"    float4 vertexDisplacementLimits[16];\\n    float4 stamLayer[7];\\n    float4 glassTransport2;\\n};"' in sokol,
        "the paired declaration adds stamLayer and glassTransport2")

# 7. Every Metal path that walks glass in a scene with media carries the medium
# text: the general fragment and the selected rough camera kernels. The
# restricted archive keeps its released bytes, so media scenes never take it.
selected = function_body("std::string metalSelectedRoughShader()")
require("declareMediumFixtureUniform(source);" in selected and "rastermedium::metalShader()" in selected,
        "the selected rough camera tints inside a medium")
require("+(media ? rastermedium::metalShader() : std::string())" in sokol, "the general fragment composes media")
require("rastermedium" not in function_body("template<class String> auto restrictedSurfaceSourceRecipe(bool pairedVolume=false,"),
        "the restricted archive keeps its released bytes")

if errors:
    for error in errors:
        print("FAIL: " + error, file=sys.stderr)
    raise SystemExit(1)
print("metal fixture released shader contract: PASS")
