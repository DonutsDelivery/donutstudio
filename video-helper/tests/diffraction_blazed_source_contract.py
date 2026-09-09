#!/usr/bin/env python3
"""Pin bounded blazed, RMS-slope, and crossed-lattice parity."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
layout = (root.parent / "shared/DiffractionMaterialGpuLayout.h").read_text()
execution = (root / "src/diffraction_material_execution.h").read_text()
gl = (root / "src/diffraction_material_gl.cpp").read_text()
metal_header = (root / "src/diffraction_material_metal.h").read_text()
metal = (root / "src/diffraction_material_metal.mm").read_text()
execution_test = (root / "tests/diffraction_material_execution_tests.cpp").read_text()
gl_test = (root / "tests/diffraction_material_gl_tests.cpp").read_text()
metal_test = (root / "tests/diffraction_material_metal_tests.cpp").read_text()
oracle = (root / "tests/support/diffraction_reference_oracle.cpp").read_text()
oracle_test = (root / "tests/diffraction_reference_oracle_tests.cpp").read_text()

assert "static_cast<unsigned>(GrooveProfile::BlazedSawtooth) == 3u" in layout
assert "CrossedTwoDimensional" in layout
assert "description.microstructure.profile != GrooveProfile::BlazedSawtooth" in execution
assert "expectedSignedOrderEvaluations" in execution

for source in (gl, metal):
    compact = "".join(source.split())
    assert "profileBlazedSawtooth=3" in compact
    assert "complexExponentialIntegral" in source
    assert "phase/duty+orderFrequency" in compact
    assert "complexExponentialIntegral(orderFrequency,duty,1.0" in compact
    assert "if(profile==profileBlazedSawtooth)" in compact
    assert "broadeningMoment" in source
    assert "directionMoment+=broadeningMoment*reflectedEnergy" in compact
    assert "secondaryOrder" in source
    assert "secondaryGeometry" in source

assert "bool blazedSawtoothProfile = false" in metal_header
assert "bool crossedTwoDimensionalLattice = false" in metal_header
assert "result.blazedSawtoothProfile = true" in metal
assert "result.crossedTwoDimensionalLattice = true" in metal
assert "result.roughnessBroadening = true" in metal
assert "does not implement RMS-slope broadening" not in execution
assert "admits the bounded blazed-sawtooth profile" in execution_test
assert "admits bounded RMS-slope broadening" in execution_test
assert "blazedExecuted" in gl_test and "blazedMoment" in gl_test
assert "roughSlopeMoment" in gl_test
assert "executor, *blazed, whiteNormal, 4" in metal_test
assert "executor, *crossed, whiteNormal, 5" in metal_test
assert "executor, *roughSlope, whiteNormal, 8" in metal_test
assert "disjoint(firstWhite.imageHandles, isolatedWhite.imageHandles)" in metal_test
assert "profile == GrooveProfile::BlazedSawtooth" in oracle
assert "solveCrossedOrder" in oracle
assert "2.0 * description.roughness.rmsSlope" in oracle
assert "blazedParseval" in oracle_test

print("bounded physical diffraction source contract: "
      "OpenGL/Metal blazed, RMS-slope, and crossed-lattice parity PASS")