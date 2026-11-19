#!/usr/bin/env python3
"""Verify Geometry Core link closure for renderer-bearing CMake targets."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SOURCE_CALLS = {"add_executable", "add_library", "target_sources"}
RENDERER_BEARING_SOURCES = {"renderer.cpp", "exporter.cpp", "viewport.cpp"}
GEOMETRY_IMPLEMENTATION = "geometry_core_backend.cpp"
EXPECTED_TARGETS = {
    "arbit-video-helper",
    "arbit-depth-fog-gl-pixel-tests",
    "arbit-temporal-frame-delay-gl-pixel-tests",
    "arbit-motion-blur-gl-pixel-tests",
    "arbit-custom-shader-gl-pixel-tests",
    "arbit-native-scene-orientation-gl-tests",
    "arbit-metal-color-transform-runtime-tests",
    "arbit-custom-shader-metal-pixel-tests",
    "CustomShaderAuthorityPixelAcceptance",
}


def cmake_calls(text: str):
    call_start = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for match in call_start.finditer(text):
        name = match.group(1)
        if name not in SOURCE_CALLS:
            continue
        depth = 1
        cursor = match.end()
        in_quote = False
        escaped = False
        while cursor < len(text) and depth:
            char = text[cursor]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_quote = not in_quote
            elif not in_quote:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
            cursor += 1
        if depth:
            raise ValueError(f"unterminated {name}() call")
        yield name, text[match.end() : cursor - 1]


def target_sources(cmake_path: Path) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    for name, body in cmake_calls(cmake_path.read_text(encoding="utf-8")):
        tokens = re.findall(r'"[^"\n]*"|[^\s()]+', body)
        if not tokens:
            continue
        target = tokens[0].strip('"')
        sources = targets.setdefault(target, set())
        for token in tokens[1:]:
            source = token.strip('"').replace("\\", "/").rsplit("/", 1)[-1]
            if source.endswith((".cpp", ".cc", ".cxx", ".mm")):
                sources.add(source)
    return targets


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    sources: dict[str, set[str]] = {}
    for relative in ("video-helper/CMakeLists.txt", "plugin/CMakeLists.txt"):
        for target, target_source_set in target_sources(repo / relative).items():
            sources.setdefault(target, set()).update(target_source_set)

    renderer_targets = {
        target
        for target, target_source_set in sources.items()
        if target_source_set & RENDERER_BEARING_SOURCES
    }
    if renderer_targets != EXPECTED_TARGETS:
        missing = sorted(EXPECTED_TARGETS - renderer_targets)
        unexpected = sorted(renderer_targets - EXPECTED_TARGETS)
        print(f"renderer-bearing target inventory changed; missing={missing}, unexpected={unexpected}", file=sys.stderr)
        return 1

    missing_closure = sorted(
        target for target in renderer_targets if GEOMETRY_IMPLEMENTATION not in sources[target]
    )
    if missing_closure:
        print(
            "renderer-bearing targets lack geometry_core_backend.cpp: "
            + ", ".join(missing_closure),
            file=sys.stderr,
        )
        return 1

    print("renderer Geometry Core closure: " + ", ".join(sorted(renderer_targets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
