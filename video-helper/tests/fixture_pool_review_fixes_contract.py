#!/usr/bin/env python3
# Contract for the p3 render-pool review fixes: the early caustic stage runs
# under fixtureMutex like the rest of the function's shared state, the compute
# clear points indexed binding i at the slot's own buffer, a resize evicts the
# mismatched raster triple, the base/resolve pools key on dimensions as well as
# bytes, and the queue-idle reap drains its lot on the lot's own fence instead
# of glFinish-ing the whole queue.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
errors = []

# (a) The compute clear writes indexed binding i; binding only the generic
# target left the dispatch clearing the previous buffer at i.
bind = backend.find("gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,i,earlySpectrum[slot]);")
dispatch = backend.find("gl.DispatchCompute((unsigned(bytes[i]/4)+255u)/256u,1,1);")
if bind < 0:
    errors.append("the early clear must BindBufferBase the slot buffer at binding i")
elif dispatch < 0 or bind > dispatch:
    errors.append("the indexed bind must precede the clear dispatch")

# (b) A resize must evict the mismatched raster triple at consume.
if "resources->pooledRasterTriple.key!=rasterKey" not in backend:
    errors.append("a mismatched pooled raster triple must be evicted on consume")

# (c) The pools key on dimensions as well as byte count.
for needle in (
    "pooledBaseColorWidth == frame->width_",
    "pooledBaseColorHeight == frame->height_",
    "staticResources->pooledBaseColorWidth = width_;",
    "resources->pooledResolveColorWidth==width",
    "resources->pooledResolveColorWidth=width;resources->pooledResolveColorHeight=height;",
    "resources->pendingBaseColorWidth=frame->width_;",
):
    if needle not in backend:
        errors.append("pool keying lost " + needle)

# (d) The early stage shares fixtureMutex with the readers and the function's
# own frame-lock stage.
if "std::lock_guard<std::mutex> earlyLock(fixtureMutex());" not in backend:
    errors.append("the early caustic stage must take fixtureMutex")
lock_at = backend.find("std::lock_guard<std::mutex> earlyLock(fixtureMutex());")
stage_at = backend.find('loopstep::Scope earlyMoveScope ("gl:caustic-early-move");')
if lock_at < 0 or stage_at < 0 or lock_at > stage_at:
    errors.append("the early lock must precede the early-move stage")

# (e) The reap waits the lot's own fence; glFinish stays only as the fallback
# for a missing or unknown fence.
if "mutable GLsync pendingDeleteFence=nullptr;" not in backend:
    errors.append("the parked-delete lot must own a fence")
if "pendingDeleteFence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);" not in backend:
    errors.append("each park must refresh the lot fence")
if "gl.ClientWaitSync(resources->pendingDeleteFence,0,0)" not in backend:
    errors.append("the queue-idle reap must poll the lot fence")
reap_at = backend.find('loopstep::Scope reapFinishScope("gl:caustic-reap-finish");')
if reap_at < 0:
    errors.append("the queue-idle reap scope is missing")
else:
    reap = backend[reap_at:reap_at + 1400]
    finish = reap.find("glFinish();")
    wait = reap.find("gl.ClientWaitSync(resources->pendingDeleteFence,0,0)")
    if wait < 0:
        errors.append("the reap must poll the fence before touching the lot")
    elif finish < 0:
        errors.append("the reap must keep the queue-wide finish as its no-fence fallback")
    elif finish < wait:
        errors.append("the reap must poll the lot fence before falling back to glFinish")

if errors:
    print("fixture pool review fixes contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture pool review fixes contract: PASS")
