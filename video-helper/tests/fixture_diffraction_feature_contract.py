#!/usr/bin/env python3
# Contract for the fixture fragment feature map. The holographic trading card's
# prepare logs features=0x2, and 0x2 is the diffraction bit: a DiffractionReflective
# material program turns kFixtureFragmentDiffraction on exactly when any program in
# the admitted collection is a diffraction material, so the foil reaches the prepare
# with its bit set. Dropping the mapping or renumbering the bit fails this contract.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
errors = []

if "kFixtureFragmentDiffraction = 1u << 1" not in backend:
    errors.append("diffraction bit must stay 1u << 1 (the observed features=0x2)")
if ("if (program.kind == NativeFixtureMaterialKind::DiffractionReflective) features |= kFixtureFragmentDiffraction;"
        not in backend):
    errors.append("diffraction material programs must set the diffraction feature bit")
if "Diffraction is on when any program is a diffraction" not in backend:
    errors.append("feature-map contract comment lost")
if "if (distinctGrating) return kFixtureFragmentAllFeatures;" not in backend:
    errors.append("the grating path must keep turning every feature on")

if errors:
    print("fixture diffraction feature contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture diffraction feature contract: PASS")
