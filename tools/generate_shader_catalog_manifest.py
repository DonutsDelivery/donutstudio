#!/usr/bin/env python3
"""Generate the immutable bundled shader catalog compiled into app and helper."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


MAX_PARAMETERS = 32
MAX_FLOAT = 3.4028234663852886e38
MIN_LONG = -(2 ** 31)
MAX_LONG = 2 ** 31 - 1
VALUE_TYPES = {"bool": 1, "long": 1, "float": 1, "point2D": 2, "color": 4}
NON_VALUE_TYPES = {"image", "audio", "audioFFT", "event"}


def cpp(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def cpp_number(value: float) -> str:
    return format(0.0 if value == 0.0 else value, ".17g")


def read_number(value, context: str) -> float:
    if isinstance(value, bool):
        result = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        raise SystemExit(f"{context} must be numeric")
    if not math.isfinite(result) or abs(result) > MAX_FLOAT:
        raise SystemExit(f"{context} must be representable as a finite renderer float")
    return result


def read_components(value, defaults: list[float], context: str) -> list[float]:
    if not isinstance(value, list) or len(value) > len(defaults):
        raise SystemExit(f"{context} must be an array of at most {len(defaults)} numbers")
    result = defaults.copy()
    for index, component in enumerate(value):
        result[index] = read_number(component, f"{context}[{index}]")
    return result


def isf_parameters(source: bytes, context: str) -> list[tuple]:
    text = source.decode("utf-8")
    header = re.match(r"\s*/\*\s*(\{.*?\})\s*\*/", text, re.DOTALL)
    if header is None:
        raise SystemExit(f"bundled ISF source has no JSON header: {context}")
    try:
        doc = json.loads(header.group(1))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid bundled ISF header {context}: {exc}") from exc

    parameters = []
    names = set()
    for index, item in enumerate(doc.get("INPUTS", [])):
        if not isinstance(item, dict):
            raise SystemExit(f"ISF INPUTS[{index}] must be an object: {context}")
        name = item.get("NAME")
        kind = item.get("TYPE")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None:
            raise SystemExit(f"invalid ISF INPUT name {name!r}: {context}")
        if name in names:
            raise SystemExit(f"duplicate ISF INPUT name {name!r}: {context}")
        names.add(name)
        if kind in NON_VALUE_TYPES:
            continue
        if kind not in VALUE_TYPES:
            raise SystemExit(f"unsupported ISF INPUT type {kind!r}: {context}")
        # The renderer imports enum metadata as a plain int and cannot preserve
        # its discrete VALUES/LABELS contract.  Do not authorize those inputs on
        # the curated bridge until the compiled schema represents that contract.
        if kind == "long" and ("VALUES" in item or "LABELS" in item):
            continue

        components = VALUE_TYPES[kind]
        defaults = [0.0] * components
        if kind == "color":
            defaults[3] = 1.0
        if "DEFAULT" in item:
            if components == 1:
                defaults[0] = read_number(item["DEFAULT"], f"{context}/{name} DEFAULT")
            else:
                defaults = read_components(item["DEFAULT"], defaults, f"{context}/{name} DEFAULT")

        has_min = "MIN" in item
        has_max = "MAX" in item
        minimum = [0.0] * components
        maximum = [0.0] * components
        if has_min:
            minimum = ([read_number(item["MIN"], f"{context}/{name} MIN")]
                       if components == 1 else
                       read_components(item["MIN"], minimum, f"{context}/{name} MIN"))
        if has_max:
            maximum = ([read_number(item["MAX"], f"{context}/{name} MAX")]
                       if components == 1 else
                       read_components(item["MAX"], maximum, f"{context}/{name} MAX"))
        coherent = True
        for component in range(components):
            if has_min and has_max and minimum[component] > maximum[component]:
                coherent = False
            if has_min and defaults[component] < minimum[component]:
                coherent = False
            if has_max and defaults[component] > maximum[component]:
                coherent = False
        if kind == "bool" and defaults[0] not in (0.0, 1.0):
            coherent = False
        if kind == "long":
            for value in defaults + minimum + maximum:
                if value != math.trunc(value) or value < MIN_LONG or value > MAX_LONG:
                    coherent = False
        # Malformed third-party metadata remains usable at its renderer default,
        # but cannot become an override authority in the compiled catalog.
        if not coherent:
            continue
        parameters.append((name, kind, components, defaults, minimum, maximum, has_min, has_max))

    if len(parameters) > MAX_PARAMETERS:
        raise SystemExit(f"bundled shader exceeds {MAX_PARAMETERS} value parameters: {context}")
    return parameters

def glsl_parameters(source: bytes, context: str) -> list[tuple]:
    text = source.decode("utf-8")
    parameters=[]; names=set()
    for line in text.splitlines():
        marker="// ARBIT_PARAM "
        if not line.startswith(marker): continue
        try: item=json.loads(line[len(marker):])
        except json.JSONDecodeError as exc: raise SystemExit(f"invalid ARBIT_PARAM JSON {context}: {exc}") from exc
        if not isinstance(item,dict) or any(key not in {"NAME","TYPE","DEFAULT","MIN","MAX"} for key in item): raise SystemExit(f"ARBIT_PARAM requires only NAME/TYPE/DEFAULT/MIN/MAX: {context}")
        name=item.get("NAME"); kind=item.get("TYPE")
        reserved={
            "uResolution","uTime","uTimeDelta","uFrame","uBeat","uBPM","uBeatPhase","uBarPhase","uBeatsPerBar","uClipBeat","uClipLength","uPlaying","uRMS","uPeak","uOnset","uOnsetAge","uAudioBands","uAudioBands2D","uNotes","uLinks","uNoteCount","uLinkCount","uRootFreq","uScoreHistoryBeats","uScoreLookaheadBeats","fragColor","uCamPos","uCamTarget","uCamUp","uCamFov","uCamNear","uCamFar",
            "arbitBand","hsv2rgb","beatPulse","arbitGuard","arbitCameraRay","arbitDisplayFragCoord","arbitTextureCoord","arbitNoteTexel","arbitNoteMidi","arbitNoteVel","arbitNoteAge","arbitNoteRemain","arbitNoteLength","arbitNoteStartOffset","arbitNoteEndOffset","arbitNoteFreq","arbitNoteCents","arbitNoteTrack","arbitNoteIsRoot","arbitNotePrimesLo","arbitNotePrimesHi","arbitNoteMasterRow","arbitNotePitch","arbitNoteBend","arbitNoteActive","arbitLink","arbitLinkRatio","arbitLinkTexel","arbitNote","lattice2","ArbitNote","ArbitLink",
            "TIME","TIMEDELTA","FRAMEINDEX","RENDERSIZE","DATE","isf_FragNormCoord","IMG_PIXEL","IMG_NORM_PIXEL","IMG_THIS_PIXEL","IMG_SIZE","iTime","iTimeDelta","iFrame","iResolution","iChannel0","iMouse","texture","texture2D","main",
            "void","float","int","uint","bool","true","false","const","uniform","attribute","varying","in","out","inout","layout","struct","precision","lowp","mediump","highp","if","else","for","while","do","switch","case","default","break","continue","return","discard","invariant","vec2","vec3","vec4","ivec2","ivec3","ivec4","uvec2","uvec3","uvec4","bvec2","bvec3","bvec4","mat2","mat3","mat4","sampler1D","sampler2D","sampler3D","samplerCube"}

        if not isinstance(name,str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}",name) is None or name.startswith("gl_") or name in reserved or name in names:
            raise SystemExit(f"invalid or duplicate ARBIT_PARAM name {name!r}: {context}")
        names.add(name)
        if kind != "float": raise SystemExit(f"ARBIT_PARAM currently requires TYPE float: {context}/{name}")
        default=read_number(item.get("DEFAULT",0),f"{context}/{name} DEFAULT")
        has_min="MIN" in item; has_max="MAX" in item
        minimum=read_number(item["MIN"],f"{context}/{name} MIN") if has_min else 0.0
        maximum=read_number(item["MAX"],f"{context}/{name} MAX") if has_max else 0.0
        if (has_min and has_max and minimum>maximum) or (has_min and default<minimum) or (has_max and default>maximum):
            raise SystemExit(f"incoherent ARBIT_PARAM range: {context}/{name}")
        parameters.append((name,kind,1,[default,0,0,0],[minimum,0,0,0],[maximum,0,0,0],has_min,has_max))
    if len(parameters)>MAX_PARAMETERS: raise SystemExit(f"bundled shader exceeds {MAX_PARAMETERS} value parameters: {context}")
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="fail if output is not the exact deterministic generated manifest")
    args = parser.parse_args()
    entries = []
    for manifest in sorted(args.catalog.glob("*/pack.json")):
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        pack = doc["pack"]
        pack_id = pack["id"]
        pack_format = pack.get("format", "").lower()
        if pack_format not in ("glsl", "isf", "raymarch"):
            raise SystemExit(f"unsupported bundled shader format {pack_format!r}: {manifest}")
        for program in doc.get("shaders", []):
            program_id = program["id"]
            value = program.get("format", pack_format)
            fmt = value.lower() if isinstance(value, str) else ""
            semantic_kind = {"glsl": "shader", "isf": "isf", "raymarch": "raymarch"}.get(fmt)
            if semantic_kind is None:
                raise SystemExit(f"unsupported bundled shader format {value!r}: {manifest}/{program_id}")
            source_path = manifest.parent / program["file"]
            source = source_path.read_bytes()
            source.decode("utf-8")
            parameters = isf_parameters(source, f"{pack_id}/{program_id}") if fmt == "isf" else glsl_parameters(source, f"{pack_id}/{program_id}")
            entries.append((pack_id, program_id, semantic_kind,
                            hashlib.sha256(source).hexdigest(), parameters))
            for index, digest in enumerate(program.get("legacySourceSha256", []), 1):
                digest = digest.strip().lower()
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise SystemExit(
                        f"invalid legacy shader SHA-256 for {pack_id}/{program_id}: {digest!r}")
                entries.append((pack_id, f"{program_id}@legacy-{index}", semantic_kind, digest, []))
    entries.sort(key=lambda entry: entry[:2])
    if len(entries) != len({(x[0], x[1]) for x in entries}):
        raise SystemExit("duplicate bundled pack/program identity")

    flat_parameters = []
    catalog_entries = []
    for pack_id, program_id, kind, digest, parameters in entries:
        offset = len(flat_parameters)
        flat_parameters.extend(parameters)
        catalog_entries.append((pack_id, program_id, kind, digest, offset, len(parameters)))

    lines = [
        "// Generated by tools/generate_shader_catalog_manifest.py; do not edit.",
        "#pragma once", "#include <cstddef>", "#include <string_view>",
        "namespace shadercatalog {",
        "inline constexpr std::size_t maxParameterCount = 32;",
        "struct Parameter { std::string_view name, type; std::size_t componentCount; double defaultValue[4], minValue[4], maxValue[4]; bool hasMin, hasMax; };",
        "struct Entry { std::string_view packId, programId, kind, sourceSha256; std::size_t parameterOffset, parameterCount; };",
        "inline constexpr Parameter parameters[] = {",
    ]
    for name, kind, components, defaults, minimum, maximum, has_min, has_max in flat_parameters:
        pad = lambda values: values + [0.0] * (4 - len(values))
        default_text = ", ".join(cpp_number(value) for value in pad(defaults))
        min_text = ", ".join(cpp_number(value) for value in pad(minimum))
        max_text = ", ".join(cpp_number(value) for value in pad(maximum))
        lines.append(f"    {{{cpp(name)}, {cpp(kind)}, {components}, "
                     f"{{{default_text}}}, {{{min_text}}}, {{{max_text}}}, "
                     f"{'true' if has_min else 'false'}, {'true' if has_max else 'false'}}},")
    lines += [
        "};",
        "inline constexpr Entry entries[] = {",
    ]
    lines += [f"    {{{cpp(a)}, {cpp(b)}, {cpp(c)}, {cpp(d)}, {offset}, {count}}},"
              for a, b, c, d, offset, count in catalog_entries]
    lines += [
        "};",
        "inline constexpr std::size_t entryCount = sizeof(entries) / sizeof(entries[0]);",
        "inline const Entry* find(std::string_view pack, std::string_view program)",
        "{ for (const auto& e : entries) if (e.packId == pack && e.programId == program) return &e; return nullptr; }",
        "} // namespace shadercatalog", "",
    ]
    generated = "\n".join(lines)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"stale shader catalog manifest: regenerate {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
