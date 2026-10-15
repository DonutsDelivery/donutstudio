#!/usr/bin/env python3
# Contract for the render-loop phase probe (diagnostic only). It pins the phase
# list, the marker call sites, the viewport_info passthrough, and the resize
# receipt, so a refactor cannot drop a probe without failing here.
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
header = (root / "src/viewport.h").read_text()
main = (root / "src/main.cpp").read_text()

expected = [
    "idle", "tick-top", "socket-pump", "linear-wait", "resize", "transport-read",
    "event-gate", "transport-store", "audio-drain", "should-produce", "advance",
    "composite", "render", "present", "publish", "frame-pacing",
]
errors = []


def body(text, start_pattern, end_pattern):
    match = re.search(start_pattern, text, re.S)
    if match is None:
        errors.append("missing: " + start_pattern)
        return ""
    end = text.find(end_pattern, match.end())
    return text[match.start():end + len(end_pattern)] if end >= 0 else text[match.start():]


# Name table matches the expected phase list, in order.
names_block = body(viewport, r"constexpr const char\* kLoopPhaseNames\[\]", "};")
names = re.findall(r'"([a-z-]+)"', names_block)
if names != expected:
    errors.append("kLoopPhaseNames out of step: " + ", ".join(names))

# The enum mirrors the name table one-for-one and ends with Count.
enum_block = body(viewport, r"enum class LoopPhase : int", "};")
members = [m for m in re.findall(r"\b([A-Z][A-Za-z]*)\b", re.sub(r"//[^\n]*", "", enum_block))
           if m not in ("LoopPhase", "int")]
kebab = [re.sub(r"(?<!^)([A-Z])", r"-\1", m).lower() for m in members[:-1]]
if members[-1] != "Count" or kebab != expected:
    errors.append("LoopPhase enum does not mirror kLoopPhaseNames: " + ", ".join(members))

# Every marker names a real phase, and every non-idle phase has a marker.
used = set(re.findall(r"enterLoopPhase \(LoopPhase::(\w+)\)", viewport))
unknown = sorted(used - set(members))
if unknown:
    errors.append("markers name unknown phases: " + ", ".join(unknown))
unmarked = [p for p in members if p not in ("Idle", "Count") and p not in used]
if unmarked:
    errors.append("phases with no marker: " + ", ".join(unmarked))
if "enterLoopPhase (LoopPhase::TickTop);" not in viewport:
    errors.append("loop head does not stamp TickTop")
if "im.loopTicks.fetch_add (1, std::memory_order_relaxed);" not in viewport:
    errors.append("loopTicks is not incremented once per tick")

# The render-thread watchdog logs the previous phase when it ran over 2 s.
if 'loop phase %s took %lld ms' not in viewport or "2000000000ll" not in viewport:
    errors.append("missing 2 s loop-phase watchdog line")

# info() exposes the probe; the header declares it; main.cpp serializes it.
info_body = body(viewport, r"ViewportInfo Viewport::info\(\) const", "\n}\n")
for needle in ("vi.loopPhase = kLoopPhaseNames[phase];",
               "vi.loopPhaseAgeMs =", "vi.loopTicks ="):
    if needle not in info_body:
        errors.append("info() does not set: " + needle)
for decl in ("std::string loopPhase", "double loopPhaseAgeMs", "uint64_t loopTicks"):
    if decl not in header:
        errors.append("viewport.h does not declare: " + decl)
for key in ("loopPhase", "loopPhaseAgeMs", "loopTicks"):
    if '{ "%s", vi.%s }' % (key, key) not in main:
        errors.append("viewport_info RPC reply does not serialize: " + key)

# Each shared resize request leaves one receipt line with its size and result.
resize_body = body(viewport, r"std::string Viewport::resizeShared \(int width, int height\)", "\n}\n")
if "[viewport] shared resize request %dx%d -> %s" not in resize_body:
    errors.append("resizeShared has no per-call receipt line")
if resize_body.count("receipt (") < 3:
    errors.append("resizeShared does not return every path through the receipt")

if errors:
    print("missing viewport loop-phase contracts: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport loop phase source contract: PASS")
