#!/usr/bin/env python3
from pathlib import Path
import re
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sidecar = (root / "../plugin/Source/VideoSidecar.cpp").read_text()
start = sidecar.index("bool VideoSidecar::viewportSetTimeline(")
body = sidecar[start:sidecar.index("\n}\n", start)]

# A timeline install runs inline and grows with the authored scene. A clock on it
# killed a healthy helper during R4's placement, so it waits like a composite probe.
assert re.search(
    r"ForegroundRequestScope\s+\w+\s*\(\s*processManager_\s*\)\s*;\s*"
    r'reply\s*=\s*sendSync\s*\(\s*"viewport_set_timeline"\s*,\s*juce::var\s*\(\s*obj\s*\)\s*,\s*errorOut\s*,\s*'
    r"SidecarProcessManager::noRequestTimeout\s*,\s*cancelRequested\s*\)\s*;",
    body), "viewport_set_timeline must wait unclocked inside a foreground scope"
print("viewport_timeline_no_kill_source_contract passed")
