#!/usr/bin/env python3
# Contract for the identity-view coverage fix: in the material- and object-id
# views a closed glass volume draw is skipped so the receiver identities behind
# it stay visible - the lighthouse/lantern/rock coverage the acceptance proves.
# The skip is gated to identity views and excluded from the diffraction and
# rough-receipt submissions, so normal color rendering is untouched.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
errors = []

if "TransportMode::ClosedVolume))" not in backend:
    errors.append("the identity-view skip must key on the closed glass volume transport")
skip_at = backend.find("surfacematerial::TransportMode::ClosedVolume))")
if skip_at < 0:
    errors.append("transport skip missing")
else:
    window = backend[max(0, skip_at - 700):skip_at]
    if "runtimeInputs.imageOutput == renderpassoutput::Output::MaterialId" not in window:
        errors.append("the skip must be gated to the material-id view")
    if "runtimeInputs.imageOutput == renderpassoutput::Output::ObjectId" not in window:
        errors.append("the skip must be gated to the object-id view")
    if "!roughReceiptPass && !diffractionFrame" not in window:
        errors.append("the skip must stay out of the diffraction and rough-receipt submissions")
    if "continue;" not in backend[skip_at:skip_at + 220]:
        errors.append("the skip must not draw the volume in an identity view")

if errors:
    print("fixture identity view coverage contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture identity view coverage contract: PASS")
