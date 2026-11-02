#!/usr/bin/env python3
"""The video helper runs without a fixed address-space or CPU ceiling.

Only a real allocation failure stops a frame. A 16 GiB RLIMIT_AS turned a Path Trace
of the 1,048,576-vertex lighthouse sea into std::bad_alloc on a 62 GB host at 26 to
29 of 64 samples, three times on 31 October 2026. About 8 GiB of that address space
is reservation (0.5 GiB resident at idle), and the helper reached 7 GiB resident.
SidecarProcessManager no longer has the mechanism either, so no caller can bring it back
by setting a field.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "plugin/Source/VideoSidecar.h",
    "plugin/Source/SidecarProcessManager.h",
    "plugin/Source/SidecarProcessManager.cpp",
)
BANNED = re.compile(
    r"maxAddressSpaceBytes|maxCpuSeconds|requiresResourceLimits|"
    r"\bsetrlimit\b|\bRLIMIT_(AS|CPU|DATA|STACK)\b|\bSIGXCPU\b")

failed = False
for relative in SOURCES:
    text = (ROOT / relative).read_text(encoding="utf-8")
    for match in BANNED.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        print(f"FAIL: {relative}:{line} mentions {match.group(0)}")
        failed = True
if failed:
    sys.exit(1)
print("helper resource limits: PASS")
