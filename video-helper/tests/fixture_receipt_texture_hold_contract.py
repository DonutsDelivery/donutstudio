#!/usr/bin/env python3
# Contract for the pooled-attachment fixes behind the composite probe refusal:
# a fresh frame generates its six raw textures (the pooling commit dropped that
# generation, so every raw read failed on zero handles), and a frame published
# as a receipt's current read target never releases its textures mid-life -
# pool-return waits for the render's next publish to release the mark, and
# destruction clears it before the final release.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
base = (root / "src/gpu_backend/backend.h").read_text()
renderer = (root / "src/fixture_scene_renderer.cpp").read_text()
errors = []

if "const bool freshRawSet = frame->rawTextures[0] == 0;" not in backend:
    errors.append("a zero raw set must be detected on the fresh path")
if "glGenTextures (static_cast<int> (frame->rawTextures.size()), frame->rawTextures.data())" not in backend:
    errors.append("the fresh path must generate the six raw textures")
if "if (!pooledDepth || freshRawSet)" not in backend:
    errors.append("the storage pass must configure generated raw sets too")
if "if(receiptReadTarget_) return;" not in backend:
    errors.append("a receipt's current read target must never release its textures")
if "        receiptReadTarget_ = false;\n        // Destruction is the receipt's replacement" not in backend and \
   "receiptReadTarget_ = false;" not in backend.split("~OpenGlFixtureSceneFrame() override", 1)[1][:400]:
    errors.append("the destructor must clear the read-target mark before release")
if "void markReceiptReadTarget() const noexcept override" not in backend:
    errors.append("the OpenGl frame must implement the read-target mark")
if "virtual void markReceiptReadTarget() const noexcept {}" not in base:
    errors.append("the base frame must expose the read-target mark")
if renderer.count("->markReceiptReadTarget();") != 1:
    errors.append("the published native frame must take the read-target mark")
if renderer.count("->releaseReceiptReadTarget();") != 1:
    errors.append("the replaced publish must release its read-target mark")
# The release must name the superseded frame: `output` still holds the
# previous publish when the new one is built, while the local `rendered` is
# fresh and its nativeFrame is null, so a released-on-rendered call releases
# nothing and the superseded frame keeps its textures out of the pool.
if "if (output.nativeFrame) output.nativeFrame->releaseReceiptReadTarget();" not in renderer:
    errors.append("the superseded publish must release the mark on output's frame, not the fresh local")

if errors:
    print("fixture receipt texture hold contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture receipt texture hold contract: PASS")
