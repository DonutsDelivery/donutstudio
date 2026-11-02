#!/usr/bin/env python3
# Contract for the caustic splat key-budget clamp: keyed slots per path are
# limited to what the dispatch-pinned gather budget admits (keyed deposits x 9
# tiles x 1024 pixels under kMaximumSplatGatherWork, the bound that keeps one
# 2^20 key dispatch), while frames at or under the design envelope keep their
# full slot count. The lighthouse dome's 3840 paths keyed 122880 slots against
# a 116508 cap before this clamp, so every frame fell to single-pixel deposits.
from pathlib import Path
import sys

root = Path(sys.argv[1])
caustics = (root / "src/gpu_backend/fixture_glass_caustics.h").read_text()
opengl = (root / "src/gpu_backend/fixture_glass_caustics_opengl.h").read_text()
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
errors = []

if "const auto byBudget=keyedCap/frame.work.size();" not in caustics:
    errors.append("key slots must clamp to the gather budget per path count")
if "return byBudget && byBudget<slots ? byBudget : slots;" not in caustics:
    errors.append("frames at or under the design envelope must keep their full slot count")
if "kMaximumSplatGatherWork\n        /(std::uint64_t(kSplatTilesPerDeposit)*kSplatTile*kSplatTile)" not in caustics:
    errors.append("the clamp must use the same divisor as the schedule admission")
if caustics.count("frame.multipleScatterPolicy?visits:(visits+1)/2") != 1:
    errors.append("the physical deposit bound formula must stay")
if "splatKeySlots(frame)" not in opengl:
    errors.append("the preview splat gate must inherit the clamped key slots")
if "paths=%llu" not in backend:
    errors.append("the refusal arithmetic record must stay for frames past the clamp")

if errors:
    print("fixture caustic splat key budget contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture caustic splat key budget contract: PASS")
