#!/usr/bin/env python3
"""The Metal Path Trace kernel's source, as the Mac's Metal compiler receives it.

This is a source check: there is no Metal compiler on Linux or Windows. It runs
cmake/metal_kernel_source.py's assemble() over device_metal.metal and rejects
the constructs the Metal Shading Language does not have, so they are caught on
every platform before a Mac reports "the kernel did not compile":

1. no lambda, no `long long`, no C++ standard or quoted include, no #pragma once;
2. no host runtime: heap, exceptions, virtual calls, RTTI, std containers, I/O;
3. none of the host-only headers or classes the shared code keeps behind
   `#if ! defined(PATHTRACE_DEVICE_METAL)`;
4. every pointer and reference declarator names an address space;
5. the pointers in the structs the host uploads name theirs in the source
   (PT_GLOBAL or PT_TABLE_SPACE), never the assembler's `thread` default;
6. the address-space macros are empty for C++, CUDA, HIP and SYCL;
7. the kernel copies every SceneView field the host uploads, and the host
   uploads the motion blur vertices and the world volume, so Metal renders the
   features the other devices do.

Usage: pathtrace_metal_kernel_source_contract.py <video-helper dir>
"""
import importlib.util
import pathlib
import re
import sys

HELPER = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("metal_kernel_source", HELPER / "cmake" / "metal_kernel_source.py")
assembler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembler)

failures = []
text, files = assembler.assemble(HELPER / "src" / "pathtrace" / "devices" / "device_metal.metal")


def code_lines():
    """(line number, code) with comments and string literals blanked."""
    body = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    for number, line in enumerate(body.split("\n"), 1):
        line = re.sub(r"//.*", "", line)
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        yield number, line


FORBIDDEN = {
    r"(?:^|[=(,]|\breturn)\s*\[[&=]?\s*[\w&, ]*\]\s*(?:\(|\{|mutable\b|->)": "lambda",
    r"\blong\s+long\b|\b\d+u?ll\b": "long long, which Metal does not have",
    r"\bnew\b|\bdelete\b|\b(?:malloc|free)\s*\(": "heap",
    r"\bthrow\b|\btry\b|\bcatch\b": "exception",
    r"\bvirtual\b|\bdynamic_cast\b|\btypeid\b": "virtual call or RTTI",
    r"\bgoto\b|\bthread_local\b": "goto or thread storage",
    r"\bstd::(?:string|string_view|vector|map|unordered_map|optional|function|unique_ptr|shared_ptr|mutex|atomic|thread)\b":
        "host container or runtime",
    r"\b(?:printf|fprintf|fopen|std::cout|std::cerr|memcpy)\b": "host I/O or memory call",
}
HOST_FILES = {"thread_pool.h", "Visual3DScene.h", "VisualModelAssetContract.h",
              "VisualEnvironmentImageAssetContract.h", "DiffractionLightingPlan.h"}
HOST_CLASSES = re.compile(r"\bclass\s+(?:PathTracer|Accumulator|FrameLayerFilm|LightSet|EnvironmentMap|Bvh|ThreadPool)\b(?!View)")
# Structs whose pointers hold uploaded addresses.
UPLOADED = ["BufferView", "SceneView", "Film", "LaunchTotals", "LaunchParams", "DeviceLookupTables",
            "LightSetView", "EnvironmentMapView", "EnvironmentField"]

for number, line in code_lines():
    if re.match(r"\s*#\s*include\s*\"", line):
        failures.append(f"line {number}: a quoted include is left: {line.strip()}")
    elif re.match(r"\s*#\s*include\s*<(?!metal_)", line):
        failures.append(f"line {number}: a C++ header is left: {line.strip()}")
    if re.match(r"\s*#\s*pragma\s+once", line):
        failures.append(f"line {number}: #pragma once is left")
    for pattern, what in FORBIDDEN.items():
        if re.search(pattern, line):
            failures.append(f"line {number}: {what}: {line.strip()}")
    if HOST_CLASSES.search(line):
        failures.append(f"line {number}: a host class is in the kernel: {line.strip()}")
    for match in assembler.DECLARATOR.finditer(line):
        if match.group(2) in assembler.NOT_TYPES:
            continue
        if not assembler.QUALIFIED_BEFORE.search(line[:match.start()]) and not re.match(assembler.SPACES, match.group(2)):
            failures.append(f"line {number}: a declarator names no address space: {line.strip()}")

for path in files:
    if path.name in HOST_FILES:
        failures.append(f"{path.name} is inlined into the kernel; it is host-only")

# 5. Each pointer member of an uploaded struct names its space in the source.
for path in files:
    source = path.read_text()
    for name in UPLOADED:
        for match in re.finditer(r"\b(?:struct|class)\s+" + name + r"\b[^;{]*\{", source):
            depth, i = 1, match.end()
            while depth and i < len(source):
                depth += {"{": 1, "}": -1}.get(source[i], 0)
                i += 1
            for member in source[match.end():i].split("\n"):
                code = re.sub(r"//.*", "", member)
                if re.search(r"\w\*+\s+\w+\s*(?:=|;|\[)", code) and "(" not in code \
                        and not re.search(r"\b(?:PT_GLOBAL|PT_TABLE_SPACE)\b", code):
                    failures.append(f"{path.name}: {name} keeps a pointer without its space: {code.strip()}")

# Metal has no double: the kernel reads it as float, and its host uploads double buffers as float.
metal = (HELPER / "src" / "pathtrace" / "devices" / "device_metal.metal").read_text()
if "#define double float" not in metal:
    failures.append("device_metal.metal must read double as float")
host = (HELPER / "src" / "pathtrace" / "devices" / "device_metal_host.mm").read_text()
if "BufferView<double>" not in host:
    failures.append("device_metal_host.mm must upload double buffers as float")

# 6. Empty off Metal.
annotations = (HELPER / "src" / "pathtrace" / "devices" / "device_annotations.h").read_text()
off_metal = annotations.split("#if defined(PATHTRACE_DEVICE_METAL)")[-1].split("#else")[1].split("#endif")[0]
for macro in ("PT_GLOBAL", "PT_TABLE_SPACE", "PT_THREAD", "PT_VIEW_SPACE"):
    if not re.search(r"#define " + macro + r"\s*\n", off_metal):
        failures.append(f"{macro} must be empty for C++, CUDA, HIP and SYCL")
if "#define PT_GLOBAL_READ(T) const T&\n" not in off_metal:
    failures.append("PT_GLOBAL_READ must be a const reference for C++, CUDA, HIP and SYCL")
if "typedef unsigned long long PtCounterWord;" not in off_metal:
    failures.append("PtCounterWord must stay CUDA's atomicAdd word off Metal")

# 7. Every SceneView field reaches the Metal kernel's view; accel is bound apart.
scene_view = (HELPER / "src" / "pathtrace" / "devices" / "scene_view.h").read_text()
body = scene_view[scene_view.index("struct SceneView"):]
body = body[body.index("{") + 1:body.index("\n};")]
fields = [m.group(1) for line in body.split("\n")
          for m in [re.search(r"(\w+)\s*(?:=[^;]*|\{\})?;", re.sub(r"//.*", "", line))] if m]
for field in fields:
    # The light set's member functions run on the kernel's own copy (PT_VIEW_SPACE).
    copied = ("lights = *params.view.lights;" in metal and "view.lights = &lights;" in metal) if field == "lights" \
        else f"view.{field} = params.view.{field};" in metal
    if field != "accel" and not copied:
        failures.append(f"device_metal.metal does not copy SceneView::{field} into the kernel's view")
for needed in ("view_.motionOpen", "view_.motionClose", "view_.worldMedium", "supportsPrimitiveMotionBlur", "MTLMotionKeyframeData"):
    if needed not in host:
        failures.append(f"device_metal_host.mm lacks {needed}")
if "ray.time" not in metal or "hit.time = ray.time;" not in metal:
    failures.append("the Metal trace must intersect at the ray's time and copy it to the hit")

for failure in failures:
    print("FAIL:", failure)
print(f"{len(failures)} problems in the Metal kernel source ({len(text.splitlines())} lines, {len(files)} files)")
sys.exit(1 if failures else 0)
