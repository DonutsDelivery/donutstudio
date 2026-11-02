#!/usr/bin/env python3
"""A clip refused without a compile error says which export condition failed.

"The clip's visual plan is not exportable" covered five conditions, so the refusals
at placeModule on 31 October 2026 could not be traced from the reply or the logs.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / "plugin/Source/PluginProcessorVideoRender.cpp").read_text(encoding="utf-8")
start = source.find("if (! revision.compiledExportable)\n            {\n                VideoRenderSnapshot::RejectedClip rejected;")
block = source[start:start + 1800] if start >= 0 else ""
missing = [text for text in ("has no compiled plan", "failed to compile and gave no reason",
                             "was not validated by its producer", "identity mode does not match",
                             "plan revision", "plan->structuralRevision")
           if text not in block]
if missing or "TRANS (\"The clip's visual plan is not exportable\")" in block:
    print("FAIL: the refusal must name the failed condition; missing", missing)
    sys.exit(1)
print("visual plan refusal reason: PASS")
