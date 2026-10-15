#!/usr/bin/env python3
"""Regenerate finite-scene diffraction validation fixtures.

This script is independent of the C++ test oracle and native shaders. It reads
only the checked-in CIE data and scene tuples, evaluates the scalar model in
Python, and writes the CSV consumed to generate the C++ fixture constants.
"""

import argparse
import cmath
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CIE = HERE / "CIE_xyz_1931_2deg.csv"
SCENES = HERE / "wavelength_validation_scenes.json"
OUTPUT = HERE / "wavelength_validation_expected.csv"
SCENE_HEADER = HERE.parent.parent / "support" / "diffraction_wavelength_validation_scenes.h"
CIE_HEADER = HERE.parent.parent / "support" / "diffraction_cie1931_2deg_5nm.h"
EXPECTED_CIE_SHA256 = "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1"
PI = math.pi


def bessel_j(order, value):
    order = abs(order)
    term = (value / 2.0) ** order / math.factorial(order)
    total = term
    for k in range(1, 200):
        term *= -(value * value / 4.0) / (k * (k + order))
        total += term
        if abs(term) <= 1e-18 * max(1.0, abs(total)):
            break
    return total


def exponential_integral(frequency, begin, end):
    if abs(frequency) < 1e-14:
        return complex(end - begin, 0.0)
    return (cmath.exp(1j * frequency * end) - cmath.exp(1j * frequency * begin)) / (1j * frequency)


def profile_efficiency(scene, wavelength, outgoing_cosine, order):
    phase = 2.0 * PI * scene["depth_nm"] * (1.0 + outgoing_cosine) / wavelength
    duty = scene["duty_cycle"]
    profile = scene["profile"]
    if profile == "binary-rectangular":
        if order == 0:
            value = 1.0 - duty + duty * cmath.exp(1j * phase)
            return abs(value) ** 2
        amplitude = math.sin(PI * order * duty) / (PI * order)
        return 4.0 * math.sin(phase / 2.0) ** 2 * amplitude * amplitude
    if profile == "sinusoidal":
        return bessel_j(order, phase / 2.0) ** 2
    if profile == "blazed-sawtooth":
        frequency = -2.0 * PI * order
        ramp = exponential_integral(phase / duty + frequency, 0.0, duty)
        land = exponential_integral(frequency, duty, 1.0)
        return abs(ramp + land) ** 2
    raise ValueError(profile)


def reflectance(scene, wavelength):
    n, k = scene["substrate_nk"]
    coating = scene["coating"]
    if coating["model"] == "uncoated":
        return ((n - 1.0) ** 2 + k * k) / ((n + 1.0) ** 2 + k * k)
    layer = complex(*coating["nk"])
    base = complex(n, k)
    r01 = abs((1.0 - layer) / (1.0 + layer)) ** 2
    r12 = abs((layer - base) / (layer + base)) ** 2
    attenuation = math.exp(-8.0 * PI * coating["nk"][1] * coating["thickness_nm"] / wavelength)
    return min(1.0, max(0.0, r01 + (1.0 - r01) ** 2 * r12 * attenuation / (1.0 - r01 * r12 * attenuation)))


def order_cosine(scene, wavelength, primary, secondary=0):
    x = primary * wavelength / scene["primary_period_nm"]
    y = 0.0
    if scene["lattice"] == "crossed-two-dimensional":
        y = secondary * wavelength / scene["secondary_period_nm"]
    tangent_squared = x * x + y * y
    return None if tangent_squared >= 1.0 else math.sqrt(1.0 - tangent_squared)


def resolved_fraction(scene, wavelength):
    roughness = scene["roughness"]["rms_height_nm"]
    zero_profile = profile_efficiency(scene, wavelength, 1.0, 0)
    zero_coherence = math.exp(-(4.0 * PI * roughness / wavelength) ** 2)
    crossed = scene["lattice"] == "crossed-two-dimensional"
    total = zero_profile * zero_coherence * (zero_profile if crossed else 1.0)
    last = scene["last_order"]
    secondary_orders = range(-last, last + 1) if crossed else (0,)
    for primary in range(-last, last + 1):
        for secondary in secondary_orders:
            if primary == 0 and secondary == 0:
                continue
            cosine = order_cosine(scene, wavelength, primary, secondary)
            if cosine is None:
                continue
            efficiency = profile_efficiency(scene, wavelength, cosine, primary)
            efficiency *= math.exp(-(2.0 * PI * roughness * (1.0 + cosine) / wavelength) ** 2)
            if crossed:
                efficiency *= profile_efficiency(scene, wavelength, cosine, secondary)
            total += efficiency * cosine
    return total


def load_cie():
    digest = hashlib.sha256(CIE.read_bytes()).hexdigest()
    if digest != EXPECTED_CIE_SHA256:
        raise SystemExit(f"CIE source SHA-256 mismatch: {digest}")
    rows = {}
    with CIE.open(newline="") as source:
        for wavelength, x, y, z in csv.reader(source):
            value = int(wavelength)
            if 380 <= value <= 700 and value % 5 == 0:
                rows[value] = (float(x), float(y), float(z))
    if sorted(rows) != list(range(380, 701, 5)):
        raise SystemExit("CIE source does not contain the required 380..700 nm samples")
    return rows


def evaluate(scene, cie):
    wavelengths = list(range(380, 701, 5))
    weights = [2.5 if i in (0, len(wavelengths) - 1) else 5.0 for i in range(len(wavelengths))]
    white_y = sum(weight * cie[wavelength][1] for weight, wavelength in zip(weights, wavelengths))
    xyz = [0.0, 0.0, 0.0]
    for weight, wavelength in zip(weights, wavelengths):
        energy = weight * reflectance(scene, wavelength) * resolved_fraction(scene, wavelength)
        for channel in range(3):
            xyz[channel] += energy * cie[wavelength][channel] / white_y
    x, y, z = xyz
    return (
        max(0.0, 3.2406 * x - 1.5372 * y - 0.4986 * z),
        max(0.0, -0.9689 * x + 1.8758 * y + 0.0415 * z),
        max(0.0, 0.0557 * x - 0.2040 * y + 1.0570 * z),
    )


def generated_outputs(source, cie, values):
    outputs = {}
    cie_lines = [
        "#pragma once", "", "#include <array>", "",
        "namespace diffractionmaterial::reference", "{",
        "struct CieEntry { double x; double y; double z; };",
        f"inline constexpr std::array<CieEntry, {len(cie)}> kCie1931 {{{{",
    ]
    for wavelength in sorted(cie):
        x, y, z = cie[wavelength]
        cie_lines.append(f"    {{ {x:.12g}, {y:.12g}, {z:.12g} }}, // {wavelength} nm")
    cie_lines.extend(("}};", "} // namespace diffractionmaterial::reference", ""))
    outputs[CIE_HEADER] = "\n".join(cie_lines)
    csv_lines = ["scene,zero_clamped_linear_srgb_r,zero_clamped_linear_srgb_g,zero_clamped_linear_srgb_b"]
    csv_lines.extend(
        ",".join((row[0], *(f"{value:.12f}" for value in row[1:])))
        for row in values
    )
    outputs[OUTPUT] = "\n".join(csv_lines) + "\n"

    lattice = {
        "one-dimensional": "GratingLattice::OneDimensional",
        "crossed-two-dimensional": "GratingLattice::CrossedTwoDimensional",
    }
    profile = {
        "binary-rectangular": "GrooveProfile::BinaryRectangular",
        "sinusoidal": "GrooveProfile::Sinusoidal",
        "blazed-sawtooth": "GrooveProfile::BlazedSawtooth",
    }
    coating = {
        "uncoated": "CoatingModel::Uncoated",
        "incoherent-dielectric": "CoatingModel::IncoherentDielectric",
    }
    field_mode = {"constant": "GrooveFieldMode::Constant"}
    integration = {"cie-1931-xyz": "SpectralIntegrationModel::Cie1931Xyz"}
    def f(value):
        text = f"{float(value):.9g}"
        if "." not in text and "e" not in text:
            text += ".0"
        return text + "f"
    def floats(values):
        return ", ".join(f(value) for value in values)

    validation = source["validation"]
    lines = [
        "// Generated by regenerate_wavelength_references.py. Do not edit.",
        "#pragma once", "", "#include \"diffraction_reference_oracle.h\"", "",
        "#include <array>", "#include <string_view>", "",
        "namespace diffractionmaterial::validation", "{",
        f"inline constexpr double kFiniteSceneReferenceStepNanometres = {float(validation['reference_step_nm']):.12g};",
        "inline constexpr double kFiniteSceneMaximumZeroClampedLinearSrgbChannelError",
        f"    = {float(validation['maximum_zero_clamped_linear_srgb_channel_error']):.12g};",
        "", "struct WavelengthValidationScene", "{",
        "    std::string_view name;", "    Description description;",
        "    reference::EvaluationInput input;",
        "    std::array<double, 3> expectedZeroClampedLinearSrgb;", "};", "",
        f"inline const std::array<WavelengthValidationScene, {len(values)}>& wavelengthValidationScenes()",
        "{", "    static const auto scenes = [] {",
        f"        std::array<WavelengthValidationScene, {len(values)}> result {{}};",
    ]
    wavelengths = source["production_wavelengths_nm"]
    incident = source["incident"]
    for index, (scene, expected) in enumerate(zip(source["scenes"], values)):
        if scene["name"] != expected[0]:
            raise ValueError("scene and expected-value order diverged")
        g = scene["groove_field"]
        lines.extend((
            "        {", "            Description description {};",
            f"            description.version = {int(scene['wire_version'])};",
            f"            description.geometry.directionUv = {{ {floats(scene['primary_direction_uv'])} }};",
            f"            description.geometry.grooveSpacingNanometres = {f(scene['primary_period_nm'])};",
            f"            description.geometry.lattice = {lattice[scene['lattice']]};",
            f"            description.geometry.secondaryDirectionUv = {{ {floats(scene['secondary_direction_uv'])} }};",
            f"            description.geometry.secondaryGrooveSpacingNanometres = {f(scene['secondary_period_nm'])};",
            f"            description.grooveField.mode = {field_mode[g['mode']]};",
            f"            description.grooveField.originUv = {{ {floats(g['origin_uv'])} }};",
            f"            description.grooveField.axisUv = {{ {floats(g['axis_uv'])} }};",
            f"            description.grooveField.grooveSpacingDeltaNanometresPerUnit = {f(g['primary_period_delta_nm_per_unit'])};",
            f"            description.grooveField.secondarySpacingDeltaNanometresPerUnit = {f(g['secondary_period_delta_nm_per_unit'])};",
            f"            description.grooveField.orientationDegreesPerUnit = {f(g['orientation_degrees_per_unit'])};",
            f"            description.microstructure = {{ {profile[scene['profile']]}, {f(scene['depth_nm'])}, {f(scene['duty_cycle'])}, {f(scene['blaze_angle_degrees'])} }};",
            f"            description.substrate = {{ {f(scene['substrate_nk'][0])}, {f(scene['substrate_nk'][1])} }};",
            f"            description.coating = {{ {coating[scene['coating']['model']]}, {f(scene['coating']['thickness_nm'])}, {{ {floats(scene['coating']['nk'])} }} }};",
            f"            description.roughness = {{ {f(scene['roughness']['rms_height_nm'])}, {f(scene['roughness']['rms_slope'])} }};",
            f"            description.spectrum.integrationModel = {integration[scene['integration_model']]};",
            f"            description.spectrum.wavelengthCount = {len(wavelengths)};",
            f"            description.spectrum.wavelengthsNanometres = {{ {floats(wavelengths)} }};",
            f"            description.spectrum.firstOrder = {int(scene['first_order'])};",
            f"            description.spectrum.lastOrder = {int(scene['last_order'])};",
            "            reference::EvaluationInput input {};",
            f"            input.incidentDirection = {{ {', '.join(f'{float(v):.12g}' for v in incident['direction'])} }};",
            f"            input.materialUv = {{ {', '.join(f'{float(v):.12g}' for v in incident['material_uv'])} }};",
            f"            input.incidentSpectrum = {{ {', '.join(f'{float(v):.12g}' for v in incident['radiance_at_production_wavelengths'])} }};",
            f'            result[{index}] = {{ "{scene["name"]}", description, input,',
            f"                {{ {expected[1]:.12f}, {expected[2]:.12f}, {expected[3]:.12f} }} }};",
            "        }",
        ))
    lines.extend(("        return result;", "    }();", "    return scenes;", "}",
                  "} // namespace diffractionmaterial::validation", ""))
    outputs[SCENE_HEADER] = "\n".join(lines)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail instead of writing when generated files are stale")
    args = parser.parse_args()
    source = json.loads(SCENES.read_text())
    cie = load_cie()
    evaluated = [(scene["name"], *evaluate(scene, cie)) for scene in source["scenes"]]
    values = [
        (scene["name"], *scene["expected_zero_clamped_linear_srgb"])
        for scene in source["scenes"]
    ]
    for actual, expected in zip(evaluated, values):
        if actual[0] != expected[0] or any(
            abs(left - right) > 5.0e-13
            for left, right in zip(actual[1:], expected[1:])
        ):
            raise SystemExit(
                f"JSON expected RGB is stale for scene {expected[0]}: {actual[1:]}"
            )
    outputs = generated_outputs(source, cie, values)
    stale = [path for path, content in outputs.items()
             if not path.exists() or path.read_text() != content]
    if args.check:
        if stale:
            raise SystemExit("stale generated diffraction fixture: "
                             + ", ".join(str(path) for path in stale))
        return
    for path, content in outputs.items():
        path.write_text(content)


if __name__ == "__main__":
    main()
