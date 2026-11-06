#!/usr/bin/env python3
"""Every call to a realtime* shader function matches a definition's parameter count.

The Realtime fixture program composes the fragment head, the material medium
(shared/RasterMedium.h), the glass query and transport text with its Preview and
smooth-only cuts (fixture_smooth_glass.h), and the Realtime text
(fixture_realtime_opengl.h). realtimeSoftVisibility gained an angularRadius
parameter, but the medium kept a four-parameter forward declaration and call.
Mesa never linked the unused path, NVIDIA did and refused every Realtime glass
program (C3002, undefined function). This scans the shader string literals of
every file the GL and Metal composers draw from, checks every realtime* call and
prototype against the definitions, and, when glslangValidator is installed,
compiles and links the medium's prototype and call against the definition.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
gpu = root / "video-helper/src/gpu_backend"
medium_header = (root / "shared/RasterMedium.h").read_text()
metal_split = medium_header.find("inline std::string metalShader()")
if metal_split < 0:
    raise SystemExit("missing: RasterMedium.h metalShader")

failures = []


def literals(source):
    """The contents of every C++ string literal, in file order, comments skipped."""
    out = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if source.startswith("//", i):
            i = source.find("\n", i)
            i = n if i < 0 else i
        elif source.startswith("/*", i):
            i = source.find("*/", i + 2)
            i = n if i < 0 else i + 2
        elif c == "R" and source.startswith('R"', i) and (i == 0 or not (source[i - 1].isalnum() or source[i - 1] == "_")):
            opening = source.find("(", i + 2)
            delimiter = source[i + 2:opening]
            closing = source.find(")" + delimiter + '"', opening)
            if closing < 0:
                raise SystemExit("unterminated raw literal")
            out.append(source[opening + 1:closing])
            i = closing + len(delimiter) + 2
        elif c == '"':
            j, piece = i + 1, []
            while j < n and source[j] != '"':
                if source[j] == "\\" and j + 1 < n:
                    piece.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(source[j + 1], ""))
                    j += 2
                else:
                    piece.append(source[j])
                    j += 1
            out.append("".join(piece))
            i = j + 1
        elif c == "'":
            j = i + 1
            while j < n and source[j] != "'":
                j += 2 if source[j] == "\\" else 1
            i = j + 1
        else:
            i += 1
    return "".join(out)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def argument_list(text, opening):
    """The top-level argument count of the parenthesis at opening, and the index after it."""
    depth, count, empty, i = 0, 1, True, opening
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return (0 if empty else count), i + 1
        elif c == "," and depth == 1:
            count += 1
        elif depth >= 1 and not c.isspace() and not (depth == 1 and c == "("):
            empty = False
        i += 1
    return None, len(text)


TYPES = r"(?:void|bool|int|uint|float|half|vec[234]|ivec[234]|float[234]|half[234]|float[234]x[234]|mat[234])"


def scan(text):
    text = strip_comments(text)
    definitions, prototypes, calls = {}, {}, []
    for match in re.finditer(r"\b(realtime[A-Z]\w*)\s*\(", text):
        name = match.group(1)
        count, after = argument_list(text, match.end() - 1)
        if count is None:
            continue
        before = text[:match.start()].rstrip()
        declared = re.search(rf"\b{TYPES}$", before) is not None
        line = text.count("\n", 0, match.start()) + 1
        if declared:
            follow = text[after:].lstrip()[:1]
            (definitions if follow == "{" else prototypes).setdefault(name, set()).add(count)
            if follow not in "{;":
                failures.append(f"unreadable declaration of {name}")
        elif not re.search(r"\bdefine\s*$", before):
            calls.append((name, count, line))
    return definitions, prototypes, calls


def check(label, text):
    definitions, prototypes, calls = scan(text)
    for name, counts in prototypes.items():
        for count in counts:
            if count not in definitions.get(name, set()):
                failures.append(f"{label}: prototype {name} takes {count} parameters, "
                                f"no definition does ({sorted(definitions.get(name, set()))})")
    for name, count, line in calls:
        if name not in definitions:
            continue  # host-side names (C++ helpers) and names from other languages
        if count not in definitions[name]:
            failures.append(f"{label}: call {name} with {count} arguments (literal line {line}), "
                            f"definitions take {sorted(definitions[name])}")
    return definitions, calls


def read(name):
    return (gpu / name).read_text()


# Every text the GL fixture composer (fixtureFragmentShaderSource) puts into a
# Realtime glass program, plus the replacement strings its variant cuts splice in.
gl_text = "\n".join([
    literals(read("backend_opengl.cpp")),
    literals(read("fixture_realtime_opengl.h")),
    literals(medium_header[:metal_split]),
    literals(read("fixture_smooth_glass.h")),
    literals(read("stam_diffraction_layer.h")),
    literals(read("fixture_opaque_visibility.h")),
])
gl_definitions, gl_calls = check("GL", gl_text)
if 5 not in gl_definitions.get("realtimeSoftVisibility", set()):
    failures.append("GL: realtimeSoftVisibility(position, normal, toLight, maxDistance, angularRadius) not found")
if not any(name == "realtimeSoftVisibility" for name, _, _ in scan(literals(medium_header[:metal_split]))[2]):
    failures.append("GL: the medium march no longer calls realtimeSoftVisibility; update this test")

# The Metal fixture: the Realtime text and its splices live in backend_sokol.mm.
metal_text = "\n".join([
    literals(read("backend_sokol.mm")),
    literals(medium_header[metal_split:]),
    literals(read("fixture_smooth_glass.h")),
    literals(read("fixture_opaque_visibility.h")),
])
metal_definitions, _ = check("Metal", metal_text)
if "realtimeSoftVisibility" not in metal_definitions:
    failures.append("Metal: realtimeSoftVisibility definition not found")

# Compile and link the medium's prototype and call against the definition's exact
# signature. A prototype without a matching definition fails the link the way
# NVIDIA does.
validator = shutil.which("glslangValidator")
if validator and not failures:
    medium = strip_comments(literals(medium_header[:metal_split]))
    realtime = strip_comments(literals(read("fixture_realtime_opengl.h")))
    prototype = re.search(r"float realtimeSoftVisibility\([^)]*\);", medium)
    definition = re.search(r"float realtimeSoftVisibility\([^)]*\)\s*\{", realtime)
    call = re.search(r"#ifdef FIXTURE_REALTIME\s*(.*?)#else", medium[medium.find("vec3 mediumSourceOf("):], re.S)
    if not (prototype and definition and call):
        failures.append("glslang step: medium prototype, Realtime definition or medium call not found")
    else:
        program = ("#version 330 core\n"
                   "float fixtureLightSize(int i) { return 0.1 * float(i); }\n"
                   + prototype.group(0) + "\n"
                   "out vec4 fragment;\n"
                   "uniform vec4 uProbe;\n"
                   "void main() {\n"
                   "    int i = int(uProbe.w); int kind = int(uProbe.z);\n"
                   "    vec3 p = uProbe.xyz; vec3 toLight = normalize(uProbe.yzx + 1.0);\n"
                   "    float lightDistance = uProbe.x; float limit = uProbe.y;\n"
                   "    " + call.group(1).strip() + "\n"
                   "    fragment = vec4(visibility);\n"
                   "}\n"
                   + definition.group(0) + " return maxDistance > 0.0 ? 1.0 : 0.0; }\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "realtime_medium.frag"
            path.write_text(program)
            result = subprocess.run([validator, "-l", "-S", "frag", str(path)], capture_output=True, text=True)
            if result.returncode != 0:
                failures.append("glslangValidator refuses the medium's realtimeSoftVisibility prototype and call:\n"
                                + (result.stdout + result.stderr).strip())
elif not validator:
    print("glslangValidator not installed: link step not run, arity scan only")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"realtime calls match their definitions (GL {len(gl_calls)} calls)")
