#!/usr/bin/env python3
"""Check retained camera admission arithmetic without claiming native execution."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "src/gpu_backend/backend_sokol.mm"
metal = source.read_text()
expression = re.search(r"const auto cameraHost=(.*?);", metal, re.S).group(1)
maximum = re.search(r"kMaximumTotalBytes = (.*?);",
                    (root.parent / "shared/RenderPassOutputContract.h").read_text()).group(1)
maximum = eval(maximum.replace("ull", ""), {"__builtins__": {}})


def camera_bytes(camera_pass, retained):
    value = expression
    replacements = {
        "frame->caustic->cameraPrimitiveBytes": retained,
        "cameraBounds.bytes": 4096,
        "bytes[13]": 1024,
        "bytes[14]": 2048,
        "pixels": 64,
        "incumbentCamera": 256,
    }
    for token, amount in replacements.items():
        value = value.replace(token, str(amount))
    value = " ".join(value.split())
    conditional = re.search(r"\(cameraPass\?([^?:]+):0\)", value)
    if conditional:
        value = value.replace(conditional.group(0), conditional.group(1) if camera_pass else "0")
    else:
        conditional = re.fullmatch(r"cameraPass\?([^?:]+):0", value)
        assert conditional, "cameraHost expression changed; review admission coverage"
        value = conditional.group(1) if camera_pass else "0"
    return eval(value, {"__builtins__": {}})


# The producer feeds this sum to both preflight and actual-allocation checks.
assert "+cameraHost" in metal.split("const auto extra=", 1)[1].split(";", 1)[0]
assert "extra>renderpassoutput::kMaximumTotalBytes-retained-2*confirmed->bounds.bytes" in metal
assert "alignedExtra>renderpassoutput::kMaximumTotalBytes-retained-2*confirmed->bounds.bytes-extra" in metal

retained = 64 * 1024 * 1024
assert camera_bytes(False, 0) == 0
assert camera_bytes(True, retained) - camera_bytes(True, 0) == retained
# A prior camera publication survives both wrappers, including fresh outputs.
for ordinary_call in ("produceMetalGlassGratingTerminalReadback", "produceMetalGlassGratingIncidents"):
    charge = camera_bytes(False, retained)
    assert charge == retained, f"{ordinary_call} omits retained private camera storage"
    remaining = retained - 1
    assert charge > remaining, f"{ordinary_call} admits over the frame ceiling"
    assert maximum - remaining + charge > maximum
    assert charge <= retained, f"{ordinary_call} charges retained storage twice"

print("Metal retained camera source arithmetic passed; native behavior remains unverified")
