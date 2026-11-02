#!/usr/bin/env python3
# Contract for the raw-pass refusal fix: the grating staging marks the shared
# scene frame's attachments private and there is no clearing assignment, so an
# abandoned staging attempt left every later passTextureDescriptor empty and the
# composite probe refused valid frames. The ordinary fixture setup resets the
# flag, and staging keeps asserting it before a staged view exists.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
errors = []

if "frame->privateCameraAttachments = false;" not in backend:
    errors.append("the ordinary fixture setup must reset the private-attachments flag")
if "frame->privateCameraAttachments=true;" not in backend:
    errors.append("staging must keep asserting the flag before a staged view exists")
if "if(frame) frame->privateCameraAttachments=false;" not in backend:
    errors.append("the completed caustic pipeline must clear the staging privacy before returning its frame")
if backend.count("frame->privateCameraAttachments = false;") < 1:
    errors.append("the ordinary setup reset is missing")

# The staging re-assert must come AFTER staging's own allocate call: the
# allocator resets the flag for every frame, so a set that only precedes the
# call is already false when the staging gates read it and the staged frame
# refuses itself (the review's confirmed defect at the staging path).
anchor = backend.find("frame->gratingCameraEvaluationTime=evaluationTime;")
if anchor < 0:
    errors.append("staging setup anchor (gratingCameraEvaluationTime) is missing")
else:
    region = backend[anchor:anchor + 900]
    allocate = region.find("allocateFixtureCameraAttachments(gl,frame.get(),true,cameraFbo")
    reassert = region.find("frame->privateCameraAttachments=true;", 0 if allocate < 0 else allocate)
    if allocate < 0:
        errors.append("staging no longer allocates the shared camera attachments")
    elif reassert < 0:
        errors.append("staging must re-assert privateCameraAttachments AFTER its allocate call")
    elif not (allocate < reassert):
        errors.append("the staging privacy re-assert must follow the allocator that resets it")

if errors:
    print("fixture private camera reset contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture private camera reset contract: PASS")
