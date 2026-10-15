#!/usr/bin/env python3
# Contract for the OpenGL fixture program completeness check. complete() takes the feature
# mask the program was composed with. A uniform that only a disabled #if FIXTURE_* block reads
# is not used by that variant, so its location is -1. complete() must not require such a
# uniform unconditionally. Each required uniform needs a read outside every FIXTURE gate.
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text(encoding="utf-8")
modifier = (root / "src/gpu_backend/fixture_vertex_modifier_shader.h").read_text(encoding="utf-8")

errors = []


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


# Call shape: the feature mask reaches complete() from the program lookup, and the enum that
# names the bits is declared before the struct that uses it.
if "bool complete (unsigned features) const noexcept" not in backend:
    errors.append("OpenGlFixtureUniformLocations::complete must take the unsigned feature mask")
if "uniforms.complete (fragmentFeatures)" not in backend:
    errors.append("the program lookup must pass fragmentFeatures to complete()")
if "uniforms.complete()" in backend:
    errors.append("complete() must not be called without the feature mask")
enum_at = backend.find("kFixtureFragmentSmoothGlass = 1u << 0")
struct_at = backend.find("struct OpenGlFixtureUniformLocations final")
if enum_at < 0 or struct_at < 0 or enum_at > struct_at:
    errors.append("the fixture feature bits must be declared before OpenGlFixtureUniformLocations")

# The glass-off declarations that the ungated tail needs are still composed here.
if 'if (! smoothGlass) text += "uniform vec4 uRoughCameraControl;\\nuniform vec4 uGlassTransport;\\n";' not in backend:
    errors.append("fixtureFragmentShaderSource must declare uRoughCameraControl and uGlassTransport when smooth glass is off")

# Split complete() into the common, smooth-glass and diffraction groups.
body = region(backend, "bool complete (unsigned features) const noexcept",
              "return common && glass && diffractionUniforms;", "OpenGlFixtureUniformLocations")
body += "return common && glass && diffractionUniforms;"
if "return common && glass && diffractionUniforms;" not in backend:
    errors.append("complete() must return the three groups together")
common = region(body, "const bool common =", "const bool glass =", "complete()")
glass = region(body, "const bool glass =", "const bool diffractionUniforms =", "complete()")
diffraction = region(body, "const bool diffractionUniforms =", "return", "complete()")


def fields(text):
    return set(re.findall(r"\b([a-z][A-Za-z0-9]*)\s*>=\s*0", text))


common_fields = fields(common)
glass_fields = fields(glass)
diffraction_fields = fields(diffraction)

if not {"glassTexture", "glassControl", "glassReceivers", "glassEmission"} <= glass_fields:
    errors.append("the smooth-glass group must require glassTexture, glassControl, glassReceivers and glassEmission")
if "glassTransport" in glass_fields:
    errors.append("glassTransport is read by the ungated tail and belongs to the common group")
if "materialKind" not in diffraction_fields or "diffractionQueryTexture" not in diffraction_fields:
    errors.append("the diffraction group must require materialKind and diffractionQueryTexture")
if "smoothGlass" not in glass or "diffraction" not in diffraction:
    errors.append("the glass and diffraction groups must be conditioned on their feature bits")
for name in sorted(common_fields & (glass_fields | diffraction_fields)):
    errors.append("field " + name + " is required both in the common group and a gated group")

# GLSL name for each uniform field, from the location lookups.
lookup = {}
for field, name in re.findall(r'uniforms\.(\w+)\s*=\s*location\s*\(\s*"([^"]+)"\s*\)', backend):
    lookup[field] = re.sub(r"\[\d+\]$", "", name)

# Composed GLSL text that is not inside any FIXTURE gate, plus the modifier header.
vertex = region(backend, 'const char* kFixtureVertexShader = R"glsl(', ')glsl";', "backend_opengl.cpp")
composite = region(backend, "const char* kFixturePassCompositeShader", ')glsl";', "backend_opengl.cpp")
tail = region(backend, 'const char* kFixtureFragmentShaderTail = R"glsl(', ')glsl";', "backend_opengl.cpp")


def has_ungated_read(text, name):
    # Walks the #if stack. A read counts only when no enclosing directive names a FIXTURE_ flag.
    # Declarations and // comments do not count.
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    stack = []
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if line.startswith("#"):
            directive = line.split()[0] if line.split() else ""
            if directive in ("#if", "#ifdef", "#ifndef"):
                stack.append(line)
            elif directive == "#endif" and stack:
                stack.pop()
            continue
        if line.startswith("uniform ") or not pattern.search(line):
            continue
        if not any("FIXTURE_" in entry for entry in stack):
            return True
    return False


sources = [vertex, composite, tail, modifier]
for field in sorted(common_fields):
    name = lookup.get(field)
    if name is None:
        errors.append("common field " + field + " has no location lookup")
        continue
    if not any(has_ungated_read(source, name) for source in sources):
        errors.append("common field " + field + " (" + name + ") is only read inside a FIXTURE_* gate")

if errors:
    for error in errors:
        print("FAIL: " + error, file=sys.stderr)
    raise SystemExit(1)
print("PASS: fixture program completeness requires only uniforms read outside FIXTURE_* gates")
