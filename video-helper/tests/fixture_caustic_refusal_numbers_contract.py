#!/usr/bin/env python3
# Contract for the caustic splat refusal arithmetic record: the once-throttled
# refusal prints the frame inputs (paths, visits, keyed slots, keyed deposits)
# beside both budget caps, because the gather cap is dispatch-pinned - the
# admitted sort stays one 2^20 key dispatch - so any budget change must be
# decided against these numbers.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
caustics = (root / "src/gpu_backend/fixture_glass_caustics.h").read_text()
errors = []

if "[gl] caustic splat refused, single-pixel deposits: %s" not in backend:
    errors.append("the refusal must print its frame arithmetic")
for field in ("paths=%llu", "visits=%llu", "keySlots=%llu", "keyed=%llu",
              "gatherCap=%llu", "passWorkCap=%llu"):
    if field not in backend:
        errors.append("the refusal must print " + field.strip("%=llu").strip() )
if "gc::splatKeySlots(scheduled)" not in backend:
    errors.append("the refusal must print the schedule key slots, not a guess")
if "!splatRefusalReported.exchange(true)" not in backend:
    errors.append("the refusal must stay once-throttled while printing arithmetic")
if "kMaximumSplatGatherWork = 1024ull*1024ull*1024ull" not in caustics:
    errors.append("the gather cap is dispatch-pinned at one 2^20 key sort; changing it needs dispatch redesign")

if errors:
    print("fixture caustic refusal numbers contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture caustic refusal numbers contract: PASS")
