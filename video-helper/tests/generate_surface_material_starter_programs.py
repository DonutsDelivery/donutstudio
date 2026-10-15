#!/usr/bin/env python3
"""Generate/check starter programs by executing production C++ lowering."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "shared/generated/SurfaceMaterialStarterPrograms.h"
DEFAULT_GENERATOR = ROOT / "plugin/build/SurfaceMaterialGraphLoweringTests"


def generate(executable: Path) -> bytes:
    if not executable.is_file():
        raise SystemExit(
            f"missing native generator {executable}; build the "
            "SurfaceMaterialGraphLoweringTests target through ./build.sh first"
        )
    with tempfile.TemporaryDirectory(prefix="surface-material-programs-") as temporary:
        candidate = Path(temporary) / OUTPUT.name
        subprocess.run(
            [str(executable), "--generate-starter-programs", str(candidate)],
            check=True,
        )
        return candidate.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path(os.environ.get("SURFACE_MATERIAL_PROGRAM_GENERATOR", DEFAULT_GENERATOR)),
    )
    args = parser.parse_args()
    generated = generate(args.generator)
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != generated:
            raise SystemExit(f"stale generated Surface Material starter programs: {OUTPUT}")
        print("Surface Material starter program fixture PASS (native production lowering)")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(generated)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
