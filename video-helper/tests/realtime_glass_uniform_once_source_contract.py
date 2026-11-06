#!/usr/bin/env python3
"""The Realtime fixture program declares each uniform once.

fixtureFragmentShaderSource puts the Realtime text (fixture_realtime_opengl.h)
after the shared fragment head. Both used to declare uCameraRotation and
uProjection. Mesa accepted the second declaration, NVIDIA refused the program
(C1038), and every Realtime glass scene failed. This composes the head and the
Realtime text, finds every uniform name declared twice, and, when
glslangValidator is installed, compiles the pair with a stub tail.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
backend = (root / "video-helper/src/gpu_backend/backend_opengl.cpp").read_text()
realtime = (root / "video-helper/src/gpu_backend/fixture_realtime_opengl.h").read_text()

failures = []


def raw_glsl(text, anchor, what):
    start = text.find(anchor)
    if start < 0:
        raise SystemExit(f"missing: {what}: {anchor}")
    opening = text.find('R"glsl(', start)
    closing = text.find(')glsl"', opening)
    if opening < 0 or closing < 0:
        raise SystemExit(f"missing: {what}: raw glsl literal after {anchor}")
    return text[opening + len('R"glsl('):closing]


def uniform_names(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"//[^\n]*", "", source)
    names = []
    for match in re.finditer(r"\buniform\s+(?:(?:lowp|mediump|highp)\s+)?\w+\s+([^;]+);", source):
        for declarator in match.group(1).split(","):
            name = re.match(r"\s*(\w+)", declarator)
            if name:
                names.append(name.group(1))
    return names


head = raw_glsl(backend, "const char* kFixtureFragmentShaderHead =", "fixture fragment head")
realtime_text = raw_glsl(realtime, "inline const char* glslShader() noexcept", "Realtime fragment text")

# The composed order in fixtureFragmentShaderSource: the head, then (much later) the
# Realtime text. Any name seen twice is a redeclaration a strict compiler refuses.
seen = {}
for piece, text in (("head", head), ("realtime", realtime_text)):
    for name in uniform_names(text):
        if name in seen:
            failures.append(f"uniform {name} declared in {seen[name]} and again in {piece}")
        else:
            seen[name] = piece

for name in ("uCameraRotation", "uProjection", "uCameraTranslation"):
    if name not in uniform_names(head):
        failures.append(f"the fragment head no longer declares {name}; the Realtime text relies on it")
    if name in uniform_names(realtime_text):
        failures.append(f"the Realtime text declares {name}, which the fragment head already declares")

# The composer must still put the Realtime text after the head.
composer_start = backend.find("std::string fixtureFragmentShaderSource (unsigned features,")
composer_end = backend.find("void readFixtureUniformLocations (", composer_start)
composer = backend[composer_start:composer_end]
if "realtimeengine::glslShader()" not in composer or "kFixtureFragmentShaderHead" not in composer:
    failures.append("fixtureFragmentShaderSource no longer composes the head and the Realtime text")

# Compile the head, then every uniform declaration of the Realtime text, as one
# fragment shader with the Realtime program's defines. glslang refuses a redeclared
# uniform the way NVIDIA does. Only the declarations go in: the Realtime functions call
# helpers from pieces this test leaves out, and glslang stops at the first unknown call.
validator = shutil.which("glslangValidator")
if validator and not failures:
    version = "#version 330 core\n"
    defines = ("#define FIXTURE_SMOOTH_GLASS 0\n#define FIXTURE_DIFFRACTION 0\n#define FIXTURE_STAM_LAYER 0\n"
               "#define FIXTURE_ROUGH_GLASS 0\n#define FIXTURE_REALTIME 1\n#define FIXTURE_WORLD_FOG 1\n")
    declarations = "".join(line + "\n" for line in re.sub(r"//[^\n]*", "", realtime_text).splitlines()
                           if re.match(r"\s*uniform\b", line))
    body = head[len(version):] if head.startswith(version) else head
    program = version + defines + body + declarations + "void main() {}\n"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "realtime_fixture.frag"
        path.write_text(program)
        result = subprocess.run([validator, "-S", "frag", str(path)], capture_output=True, text=True)
        log = result.stdout + result.stderr
        for name in ("uCameraRotation", "uProjection"):
            if re.search(rf"redefinition[^\n]*{name}|{name}[^\n]*redefinition", log):
                failures.append(f"glslangValidator: {name} redefined in the composed Realtime program")
        if result.returncode != 0:
            failures.append("glslangValidator refuses the head plus the Realtime uniforms:\n" + log.strip())
elif not validator:
    print("glslangValidator not installed: compile step not run, declaration scan only")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print("realtime glass uniforms declared once")
