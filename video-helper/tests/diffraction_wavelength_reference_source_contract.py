#!/usr/bin/env python3
"""Verify diffraction reference provenance and deterministic regeneration."""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "diffraction"
TRACKED_OUTPUTS = (
    FIXTURES / "wavelength_validation_expected.csv",
    ROOT / "tests" / "support" / "diffraction_cie1931_2deg_5nm.h",
    ROOT / "tests" / "support" / "diffraction_wavelength_validation_scenes.h",
)

cie = FIXTURES / "CIE_xyz_1931_2deg.csv"
assert hashlib.sha256(cie.read_bytes()).hexdigest() == (
    "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1"
)
subprocess.run(
    [sys.executable, str(FIXTURES / "regenerate_wavelength_references.py"), "--check"],
    check=True,
)
for path in TRACKED_OUTPUTS:
    assert path.is_file(), f"missing generated output: {path}"

oracle = (ROOT / "tests" / "support" / "diffraction_reference_oracle.cpp").read_text()
assert '#include "diffraction_cie1931_2deg_5nm.h"' in oracle
assert "constexpr std::array<CieEntry" not in oracle
print("diffraction wavelength reference provenance PASS")
