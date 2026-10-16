#!/usr/bin/env python3
# Guards the 15 Oct MacBook kernel panic ("IOGPUResource::free called for
# resource still owned by an IOGPUDevice" during sg_commit). Sokol's frame
# command buffers must retain what they encode, and a refused fixture frame
# must submit Sokol's open command buffer before releasing its objects.
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()

setup = metal[metal.index("bool ensureSokolMetal()"):metal.index("int metalIdPoolFreeSlots()")]
assert "desc.metal.use_command_buffer_with_retained_references = true;" in setup
assert setup.index("use_command_buffer_with_retained_references") < setup.index("sg_setup (&desc);")

settle = metal[
    metal.index("void settleOpenSokolCommandBufferUnlocked()"):
    metal.index("bool ensureMetalIdHeadroom (")
]
assert "_sg.cur_pass.in_pass" in settle
assert settle.index("sg_commit();") < settle.index("waitUntilCompleted")

frame = metal[metal.index("class MetalFixtureSceneFrame final"):]
destroy = frame[frame.index("void destroyUnlocked() noexcept"):]
destroy = destroy[:destroy.index("caustic.reset();")]
assert "settleOpenSokolCommandBufferUnlocked();" in destroy

# No backend path may opt back into unretained command buffers.
assert not re.search(r"commandBufferWithUnretainedReferences", metal)
assert not re.search(r"use_command_buffer_with_retained_references\s*=\s*false", metal)

print("metal command buffer lifetime source contract: PASS")
