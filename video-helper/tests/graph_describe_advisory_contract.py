#!/usr/bin/env python3
"""graph_describe is a read-only query, so its timeout must never kill the helper.

The helper answers requests in order and the frame loop can hold it for tens of
seconds on a large scene. A 1,048,576-vertex sea lost its helper to this timeout
twice at 11 and 14 of 16 Path Trace samples on 31 October 2026.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
text = (ROOT / "plugin/Source/VideoSidecar.h").read_text(encoding="utf-8")
body = re.search(r"juce::var graphDescribe \(juce::String& errorOut\)\s*\{(.*?)\n    \}", text, re.S).group(1)
call = re.search(r'sendRequestSync \("graph_describe".*?\);', body, re.S)
if call is None or not re.search(r",\s*false\s*\);$", call.group(0)):
    print("FAIL: graphDescribe must send through sendRequestSync with timeoutKillsHelper false")
    sys.exit(1)
# The frame loop waits on the viewport state lock, so the payload decode in the
# resource accounting and the payload copy into the reply run after the lock is gone.
viewport = (ROOT / "video-helper/src/viewport.cpp").read_text()
describe = viewport[viewport.index("std::string Viewport::describeGraph() const"):]
describe = describe[:describe.index("\n}\n")]
released = describe.index("state lock held")
assert "visualPlanPublication.shared()" in describe[:released], "plans are leased under the lock"
for work in ("accountVisualPlanResources", "operation.payloadXml"):
    assert describe.index(work) > released, f"{work} runs under the viewport state lock"

print("graph_describe advisory: PASS")
