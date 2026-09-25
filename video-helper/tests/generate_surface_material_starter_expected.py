#!/usr/bin/env python3
"""Generate the independent Surface Material starter pixel oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "video-helper/tests/fixtures/surface_material_starter_expected.json"

OBJECT_ROTATION = (-0.16773126, 0.25488700, 0.04494346, 0.95125124)
LIGHT_ROTATION = OBJECT_ROTATION
AMBIENT = (0.12, 0.12, 0.12)
LIGHT_COLOR = (1.0, 1.0, 1.0)
LIGHT_INTENSITY = 0.88

# Authored independently from SurfaceMaterialStarterModules.h and the native
# generated-program fixture. Each tuple is base color, metallic, emission,
# roughness, normal slot, opacity, transmission, IOR, clearcoat, texture slot.
MATERIALS = (
    ("visual.material.matte-clay", (0.55, 0.18, 0.08), 0.0, (0.0, 0.0, 0.0), 0.82, (0.0, 0.0, 1.0), 1.0, 0.0, 1.45, 0.0, 0),
    ("visual.material.polished-metal", (0.72, 0.78, 0.86), 1.0, (0.0, 0.0, 0.0), 0.18, (0.0, 0.0, 1.0), 1.0, 0.0, 1.45, 0.0, 0),
    ("visual.material.frosted-glass", (0.88, 0.95, 1.0), 0.0, (0.0, 0.0, 0.0), 0.28, (0.0, 0.0, 1.0), 0.35, 0.92, 1.5, 0.0, 0),
    ("visual.material.clearcoat-plastic", (0.04, 0.18, 0.72), 0.0, (0.0, 0.0, 0.0), 0.32, (0.0, 0.0, 1.0), 1.0, 0.0, 1.46, 1.0, 0),
    ("visual.material.emissive-neon", (0.01, 0.08, 0.12), 0.0, (0.0, 8.0, 12.0), 0.4, (0.0, 0.0, 1.0), 1.0, 0.0, 1.45, 0.0, 0),
)


def rotate(q: tuple[float, float, float, float],
           v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = q
    qv = (x, y, z)
    cross1 = (qv[1] * v[2] - qv[2] * v[1], qv[2] * v[0] - qv[0] * v[2], qv[0] * v[1] - qv[1] * v[0])
    inner = tuple(cross1[i] + w * v[i] for i in range(3))
    cross2 = (qv[1] * inner[2] - qv[2] * inner[1], qv[2] * inner[0] - qv[0] * inner[2], qv[0] * inner[1] - qv[1] * inner[0])
    return (v[0] + 2.0 * cross2[0], v[1] + 2.0 * cross2[1],
            v[2] + 2.0 * cross2[2])


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in v))
    return (v[0] / length, v[1] / length, v[2] / length)


def unorm8(value: float) -> int:
    return int(math.floor(min(max(value, 0.0), 1.0) * 255.0 + 0.5))


def srgb(value: float) -> float:
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1.0 / 2.4) - 0.055


def fnv1a64(payload: bytes | bytearray) -> str:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = value * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def generate() -> bytes:
    normal = normalize(rotate(OBJECT_ROTATION, (0.0, 0.0, 1.0)))
    to_light = normalize(rotate(LIGHT_ROTATION, (0.0, 0.0, 1.0)))
    ndotl = max(sum(normal[i] * to_light[i] for i in range(3)), 0.0)
    lighting = tuple(AMBIENT[i] + LIGHT_COLOR[i] * LIGHT_INTENSITY * ndotl for i in range(3))
    rows = []
    samples = bytearray()
    for index, material in enumerate(MATERIALS, start=1):
        identifier, base, metallic, emission, roughness, normal_slot, opacity, transmission, ior, clearcoat, texture_slot = material
        dielectric = ((ior - 1.0) / (ior + 1.0)) ** 2
        coverage = 1.0 - transmission * (1.0 - dielectric)
        coat = 0.04 * clearcoat
        diffuse = (1.0 - metallic) * (1.0 - 0.5 * roughness) * (1.0 - transmission)
        specular = (dielectric + (1.0 - dielectric) * metallic) * (1.0 - roughness)
        weight = diffuse + specular
        linear = tuple((base[i] * lighting[i] * weight * (1.0 - coat)
                        + lighting[i] * coat + emission[i]) / max(coverage, 0.000001) for i in range(3))
        alpha = opacity * coverage
        if opacity < 1.0 or transmission > 0.0:
            background = (7 / 255, 10 / 255, 18 / 255)
            linear = tuple(min(max(linear[i], 0.0), 1.0) * alpha
                           + background[i] * (1.0 - alpha) for i in range(3))
        rgba = [*(unorm8(channel) for channel in linear), 255]
        wrong_transfer = [*(unorm8(srgb(min(max(channel, 0.0), 1.0))) for channel in linear), 255]
        pbr = [*base, metallic, *emission, roughness, *normal_slot, opacity,
               transmission, ior, clearcoat, float(texture_slot), index, 0, 0, 0]
        rows.append({
            "id": identifier,
            "pbrParameterBlock": pbr,
            "rejectedWrongShadingRgba8": [[0, 0, 0, unorm8(opacity)], wrong_transfer],
            "directionalLitCenterRgba8": rgba,
        })
        samples.extend(rgba)
    document = {
        "materials": rows,
        "provenance": {
            "constants": "Independent authored constants, separate from production module and generated-program tables",
            "independence": "This Python binary64 model computes JSON pixels without reading native output or production starter definitions.",
            "model": "Quaternion-rotated fixture normal and directional light, ambient, metallic/roughness weights, normal-incidence IOR Fresnel, thin transmission coverage, clearcoat energy layering, emission, straight-alpha blending over the opaque fixture background, and linear RGBA8 quantization",
            "fixture": {
                "objectRotationXyzw": list(OBJECT_ROTATION),
                "lightRotationXyzw": list(LIGHT_ROTATION),
                "ambientRgb": list(AMBIENT),
                "lightColorRgb": list(LIGHT_COLOR),
                "lightIntensity": LIGHT_INTENSITY,
            },
        },
        "sampledRgba8Fnv1a64": fnv1a64(samples),
        "sampledRgba8Sha256": hashlib.sha256(samples).hexdigest(),
        "schema": 3,
        "truthBoundary": "The independent CPU model covers the exact constant PBR block and one directional-lit center sample per starter, including metallic, roughness, emission, opacity, transmission, IOR and clearcoat. The native surface model uses normal-incidence Fresnel and thin alpha transmission, not volumetric refraction, screen-space distortion, absorption or multilayer path tracing.",
    }
    return (json.dumps(document, indent=2) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = generate()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != generated:
            raise SystemExit(f"stale independent Surface Material starter oracle: {OUTPUT}")
        print("Surface Material starter independent pixel oracle PASS")
        return 0
    OUTPUT.write_bytes(generated)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
