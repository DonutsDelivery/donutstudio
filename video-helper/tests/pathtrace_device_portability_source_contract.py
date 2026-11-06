#!/usr/bin/env python3
"""Device-portability boundary of the path integrator's shared code.

This is a source check. It does not run a CUDA compiler and does not prove the
OptiX kernel compiles. It proves three things about the sources:

1. Every function in the certified files carries a device annotation, unless it
   is named in HOST_ONLY.
2. An annotated function body uses no host-only construct (FORBIDDEN), with any
   `#if ! defined(__CUDACC__)` branch removed first.
3. Every function the kernel or an annotated body calls resolves to an annotated
   function, a standard function with a device overload (DEVICE_STD), or a name
   in KNOWN_BLOCKERS. That list and CAPPED_NAMES may only shrink.

Usage: pathtrace_device_portability_source_contract.py <video-helper dir>
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1]) / "src" / "pathtrace"

# file -> the text span to certify, as (start marker, end marker). None is the
# whole file.
CERTIFIED = {
    "vec_math.h": (None, None),
    "accumulator.h": ("std::uint32_t pcgHash", "class Accumulator"),
    "sampler.h": (None, None),
    "media.h": (None, None),
    "spectrum.h": (None, None),
    "spectrum.cpp": (None, None),
    "bsdf.h": (None, None),
    "bsdf.cpp": (None, None),
    "bvh.h": (None, "struct BvhNode"),
    "bvh_traversal.h": (None, None),
    "scene_ingest.h": ("PT_SHARED_FN Vec3 pointAtFrame", "bool makePinholeCamera"),
    "devices/device_tables.h": (None, None),
    "environment_map.h": ("class EnvironmentMapView", "class EnvironmentMap : public"),
    "environment_map.cpp": (None, None),
    "lights.h": ("struct EnvironmentField", "class LightSet : public"),
    "lights.cpp": (None, None),
    "sky_cloud_volume.h": (None, None),
    "devices/device_interface.h": (None, None),
    "devices/integrator_body.inl": (None, None),
    "../diffraction_order_core.h": (None, None),
}

# Functions in certified files that only the host runs, with what replaces them
# on a device. None of them removes a feature from device rendering.
HOST_ONLY = {
    "spectrum.h": {"exportSpectrumTables"},          # fills the upload
    "spectrum.cpp": {"exportSpectrumTables"},
    "bsdf.h": {"exportGgxTables"},
    "bsdf.cpp": {"exportGgxTables"},
    "scene_ingest.h": {"ingestScene", "objectWorldMatrix", "transformMatrix"},  # scene build, declared
                                                                            # beside the shutter helpers
    "bvh.h": {"motionTriangleBounds"},              # the BVH build; traversal reads the built bounds
    "environment_map.cpp": {"valid", "copyEnvironmentPixels", "environmentImageTint",
                            "environmentImageOrientationRadians", "build", "clear", "refreshView",
                            "EnvironmentMap", "operator="},  # scene build; the device reads the built view
    "lights.cpp": {"build", "clear", "refreshView", "sameAsset", "bindEnvironmentImage", "finiteProduct",
                   "LightSet", "operator="},         # scene build; the device reads the built view
}

# Called from device code and still host C++. May only shrink.
KNOWN_BLOCKERS = {}

# Existing fixed sizes in the certified files, none of them a scene size. May
# only shrink: a new name here needs Emil's say, under his no-limits rule.
CAPPED_NAMES = {"kMaxShadowCrossings", "kMaxLobes", "kMaxDepth"}

FORBIDDEN = {
    r"\bnew\b": "heap allocation",
    r"\bdelete\b": "heap release",
    r"\b(?:malloc|calloc|realloc|free)\s*\(": "heap allocation",
    r"\bthrow\b": "exception",
    r"\btry\b": "exception",
    r"\bcatch\b": "exception",
    r"\bstd::(?:string|vector|map|unordered_map|shared_ptr|unique_ptr|function|mutex|atomic)\b": "host container or runtime",
    r"\bstd::(?:upper_bound|lower_bound|sort|swap|memcpy)\b": "not constexpr under C++17",
    r"\bmemcpy\b": "host memory call",
    r"\.fill\s*\(": "std::array::fill is not constexpr under C++17",
    r"\bvirtual\b": "virtual dispatch",
    r"\bdynamic_cast\b": "RTTI",
    r"\btypeid\b": "RTTI",
    r"\bthread_local\b": "thread storage",
    r"\bstatic\s+(?!constexpr\b)[\w:<>]+\s+\w+\s*[;={(]": "function-local static",
    r"\b(?:printf|fprintf|fopen|std::cout|std::cerr)\b": "file or console IO",
}

# Standard functions with device overloads or constexpr definitions.
DEVICE_STD = {"sqrt", "fabs", "abs", "isfinite", "isinf", "isnan", "exp", "exp2", "log", "sin", "cos", "tan",
              "atan", "atan2", "acos", "asin", "floor", "ceil", "pow", "nextafter", "min", "max", "clamp",
              "hypot", "copysign", "fmin", "quiet_NaN", "infinity", "max_digits10", "epsilon", "lowest",
              "denorm_min"}
# Constexpr functions outside the certified files. Under relaxed constexpr nvcc
# compiles them for the device.
DEVICE_CONSTEXPR = {"aluminiumIndexAt"}
substrate = (ROOT / "../../../shared/DiffractionSubstrateData.h").read_text()
if "constexpr ComplexIndexRow aluminiumIndexAt" not in substrate:
    print("aluminiumIndexAt must stay constexpr")
    sys.exit(1)
# Device built-ins and shim-provided names the kernel may call.
DEVICE_BUILTINS = {"__float_as_uint", "__uint_as_float", "optixTrace", "optixGetLaunchIndex", "atomicAdd",
                   "make_float3", "OptixVisibilityMask", "optixGetPrimitiveIndex", "optixGetPayload_4",
                   "optixGetPayload_1", "optixIgnoreIntersection", "optixGetTriangleBarycentrics",
                   "optixSetPayload_0", "optixSetPayload_1", "optixSetPayload_2", "optixSetPayload_3",
                   "optixGetRayTmax"}
KEYWORDS = {"float", "if", "for", "while", "switch", "return", "sizeof", "static_cast", "reinterpret_cast", "const_cast",
            "noexcept", "decltype", "alignof", "defined", "catch", "constexpr", "operator"}
ANNOTATION = re.compile(r"\b(PT_DEVICE_FN|PT_DEVICE_ENTRY|PT_SHARED_FN|PT_SHARED_ENTRY)\b")
TYPE_NAME = re.compile(r"^[A-Z]")

failures = []


def strip_host_branches(text):
    """Blank comments, strings and the host branch of each __CUDACC__ conditional."""
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', text)
    out = []
    stack = []  # (is_cuda_conditional, keep_now)
    for line in text.split("\n"):
        stripped = line.strip()
        keep = all(k for _, k in stack)
        if re.match(r"#\s*if\b", stripped) or re.match(r"#\s*ifn?def\b", stripped):
            if re.match(r"#\s*if\s+defined\s*\(\s*__CUDACC__\s*\)\s*$", stripped):
                stack.append((True, True))
            elif re.match(r"#\s*if\s*!\s*defined\s*\(\s*__CUDACC__\s*\)\s*$", stripped):
                stack.append((True, False))
            else:
                stack.append((False, True))
            out.append("")
        elif re.match(r"#\s*else\b", stripped):
            if stack and stack[-1][0]:
                stack[-1] = (True, not stack[-1][1])
            out.append("")
        elif re.match(r"#\s*endif\b", stripped):
            stack.pop()
            out.append("")
        elif stripped.startswith("#"):
            out.append("")
        else:
            out.append(line if keep else "")
    return "\n".join(out)


def functions(text):
    """Yield (name, signature, body or None) for functions at namespace or class depth."""
    i, n = 0, len(text)
    stack, buf, paren = [], "", 0
    while i < n:
        c = text[i]
        if c == ":" and buf.strip() in ("public", "private", "protected"):
            buf = ""
        elif c == "(":
            paren += 1
            buf += c
        elif c == ")":
            paren -= 1
            buf += c
        elif paren == 0 and c in ";{}":
            sig = buf.strip()
            buf = ""
            if c == "}":
                if stack:
                    stack.pop()
                i += 1
                continue
            is_scope = c == "{" and (sig.startswith("namespace") or sig == "namespace"
                                     or (re.match(r"(template\s*<[^{]*>\s*)?(class|struct|union)\b", sig) and "(" not in sig))
            at_scope = all(stack)
            body = None
            if c == "{" and not is_scope:
                depth, j = 1, i + 1
                while j < n and depth:
                    depth += {"{": 1, "}": -1}.get(text[j], 0)
                    j += 1
                body = text[i + 1 : j - 1]
                if at_scope:
                    yield from emit(sig, body)
                i = j
                continue
            if at_scope and c == ";":
                yield from emit(sig, None)
            if c == "{":
                stack.append(is_scope)
            i += 1
            continue
        else:
            buf += c
        i += 1


def emit(sig, body):
    if "(" not in sig or re.search(r"\(\s*\*", sig.split("=")[0]):
        return  # not a function, or a data member holding a pointer to an array
    clean = re.sub(r"^template\s*<[^(]*?>\s*", "", sig)
    head = clean.split("(")[0]
    if "=" in head and "operator" not in head:
        return
    if re.match(r"(using|friend|typedef|static_assert|extern|enum|class |struct |return\b)", clean):
        return
    if re.search(r"=\s*(default|delete)\s*$", sig):
        return
    if clean.count("(") == 0 or not head.strip():
        return
    m = re.search(r"(operator\s*(?:\[\]|\(\)|[-+*/=<>!]+)|~?\w+)\s*$", head.strip())
    if m:
        yield re.sub(r"\s+", "", m.group(1)), sig, body


def certified_text(path, span):
    text = strip_host_branches((ROOT / path).read_text())
    start, end = span
    if start is not None:
        # Back up to the line start so a leading annotation stays in the span.
        text = text[text.rfind("\n", 0, text.index(start)) + 1 :]
    if end is not None:
        text = text[: text.index(end)]
    return text


annotated = {}
bodies = {}
for path, span in CERTIFIED.items():
    for name, sig, body in functions(certified_text(path, span)):
        has = ANNOTATION.search(sig)
        if not has:
            if name not in HOST_ONLY.get(path, set()):
                failures.append(f"{path}: {name} has no device annotation and is not listed as host-only")
            continue
        annotated[name] = (path, has.group(1))
        if body is not None:
            bodies.setdefault(path, []).append((name, body))

# A host-only name must not also be annotated in the same file: that would make
# the list say something false.
for path, names in HOST_ONLY.items():
    text = certified_text(path, CERTIFIED[path])
    for name, sig, _ in functions(text):
        if name in names and ANNOTATION.search(sig):
            failures.append(f"{path}: {name} is listed as host-only but is annotated")

# Types whose constructors and members are called by plain name.
type_like = set(re.findall(r"\b(?:struct|class|enum class|using)\s+(\w+)",
                           "\n".join((ROOT / p).read_text() for p in CERTIFIED)))

ALLOWED_CALLS = annotated.keys() | KNOWN_BLOCKERS.keys() | DEVICE_CONSTEXPR | DEVICE_STD | DEVICE_BUILTINS | KEYWORDS | type_like
# The body reaches its device through these shim members.
ALLOWED_CALLS |= {"makeSampler", "intersect", "occluded", "filmAdd", "filmSplat", "filmCountLightPath", "filmGuide"}
# A functor parameter of buildBsdf; its callers pass lambdas, which take the
# device context of the function that writes them.
ALLOWED_CALLS |= {"toColor"}

for path, entries in bodies.items():
    for name, body in entries:
        for pattern, why in FORBIDDEN.items():
            hit = re.search(pattern, body)
            if hit:
                failures.append(f"{path}: {name} uses {hit.group(0).strip()} ({why})")
        # Lambda parameters and locals called as functions are local names.
        local_callables = set(re.findall(r"\b(?:const\s+)?auto\s+(\w+)\s*=\s*\[", body))
        for callee in re.findall(r"(?<![\w.])(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)\s*(?:<[^<>()]*>)?\s*\(", body):
            if callee in ALLOWED_CALLS or callee in local_callables or TYPE_NAME.match(callee):
                continue
            failures.append(f"{path}: {name} calls {callee}, which is not annotated, device-legal or a known blocker")
        for callee in re.findall(r"[.>]\s*(\w+)\s*\(", body):
            if callee in ALLOWED_CALLS:
                continue
            failures.append(f"{path}: {name} calls member {callee}, which is not annotated")

# The kernel's own file: the host pass gets no new work, and the kernel stays honest.
kernel = (ROOT / "devices/device_optix.cu").read_text()
for include in ("spectrum.cpp", "bsdf.cpp", "environment_map.cpp", "lights.cpp"):
    if f'#include "../{include}"' not in kernel:
        failures.append(f"device_optix.cu no longer includes {include}")
if "__CUDACC_RELAXED_CONSTEXPR__" not in kernel:
    failures.append("device_optix.cu must refuse a build without relaxed constexpr")
if "params.tables" not in kernel:
    failures.append("device_optix.cu must read the uploaded lookup tables from params")
registry = (ROOT / "devices/device_registry.cpp").read_text()
if "#if defined(ARBIT_HAVE_OPTIX_KERNEL)\n    environment.optixKernelBuilt = true;" not in registry:
    failures.append("the registry must claim the kernel only in a build that holds it")
for name, why in KNOWN_BLOCKERS.items():
    if name not in (ROOT / "devices/integrator_body.inl").read_text():
        failures.append(f"{name} is gone: remove it from KNOWN_BLOCKERS ({why})")

# Files included into one TU must not define the same internal helper twice.
seen = {}
for path in ("spectrum.cpp", "bsdf.cpp", "environment_map.cpp", "lights.cpp"):
    for line in strip_host_branches((ROOT / path).read_text()).split("\n"):
        m = re.match(r"(?:PT_SHARED_ENTRY |PT_SHARED_FN )?(?:constexpr )?[\w:<>&*]+(?: [\w:<>&*]+)* (\w+)(?: \(| =)", line)
        if m and "::" not in line.split("(")[0]:
            symbol = m.group(1)
            if symbol in seen and seen[symbol] != path:
                failures.append(f"{symbol} is defined at namespace scope in both {seen[symbol]} and {path}")
            seen.setdefault(symbol, path)

# No limits: fixed-size names may not grow, and no refusal may key on scene size.
for path in CERTIFIED:
    text = strip_host_branches((ROOT / path).read_text())
    for name in re.findall(r"\bkMax[A-Z]\w*", text):
        if name not in CAPPED_NAMES:
            failures.append(f"{path}: new cap name {name}")
    if re.search(r"\b(?:unsupported|tooLarge|too_large|subset)\b", text, re.I):
        failures.append(f"{path}: refusal wording in shared device code")

for path in ("lights.h", "environment_map.h"):
    text = (ROOT / path).read_text()
    if "device::BufferView" not in text:
        failures.append(f"{path} must expose its buffers as device views")
if "std::size_t count" not in (ROOT / "devices/device_interface.h").read_text():
    failures.append("BufferView::count must stay wide enough for any allocation")

if failures:
    print("\n".join(sorted(set(failures))))
    sys.exit(1)
print(f"portable: {len(annotated)} annotated functions in {len(CERTIFIED)} files; "
      f"{len(KNOWN_BLOCKERS)} known blockers: {', '.join(sorted(KNOWN_BLOCKERS))}")
