#!/usr/bin/env python3
"""A publication of an unchanged scene does not encode the scene to hash it.

The scene is an immutable shared snapshot. Encoding the 1,048,576-vertex lighthouse
sea cost 276 MB of temporary string and 1.5 to 11 s per Path Trace sample on
31 October 2026, in the seam-identity step.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
seam = (ROOT / "video-helper/src/glb_native_render_seam.cpp").read_text(encoding="utf-8")
contract = (ROOT / "shared/VisualImportedSceneRenderOperationContract.h").read_text(encoding="utf-8")
guard = re.search(r"lastIdentityRequest_->sceneSnapshot != operation\.sceneSnapshot"
                  r".*?\{[^{}]*sha256Text\(visualimportedscenerender::encode\(operation\)\)", seam, re.S)
if guard is None:
    print("FAIL: the seam must hash the encoded operation only when the request changed")
    sys.exit(1)
if "left.sceneSnapshot == right.sceneSnapshot" not in contract:
    print("FAIL: sameRequest must not encode a scene to compare it with itself")
    sys.exit(1)
print("seam scene identity: PASS")
