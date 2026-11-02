#!/usr/bin/env python3
# Contract for the splat radiance refusal arithmetic: the first native-vs-CPU
# mismatch carries check order, actual, expected, tolerance and the schedule's
# keyed-set numbers into the refused error itself, so the row result names the
# arithmetic that produced the divergence instead of only its verdict.
from pathlib import Path
import sys

root = Path(sys.argv[1])
caustics = (root / "src/gpu_backend/fixture_glass_caustics.h").read_text()
errors = []

if 'error+=" (check="+std::to_string(bandChecks)' not in caustics:
    errors.append("the radiance refusal must carry its check order")
if '" band="+std::to_string(firstFlat%8)' not in caustics:
    errors.append("the radiance refusal must carry the failing band")
if 'firstPixel%frame.width' not in caustics:
    errors.append("the radiance refusal must carry the failing pixel column")
if "checkSpectralBandsIndexed" not in caustics:
    errors.append("the splat walk must expose its flat band index")
for field in ('" actual="+std::to_string(firstActual)',
              '" expected="+std::to_string(firstExpected)',
              '" tolerance="+std::to_string(firstTolerance)',
              '" keys="+std::to_string(receipt.keys.size())',
              '" padded="+std::to_string(frame.splat.paddedKeyCount)',
              '" keySlots="+std::to_string(frame.splat.keySlots)',
              '" paths="+std::to_string(frame.work.size())'):
    if field not in caustics:
        errors.append("the radiance refusal must carry " + field.split("+std::to_string(")[-1].rstrip(")"))
if "firstFound=true;firstActual=actual;firstExpected=expected;firstTolerance=tolerance;" not in caustics:
    errors.append("only the first mismatch may be recorded")
if 'return refuse("Glass caustic splat native pixel radiance differs from the CPU reference");' not in caustics:
    errors.append("the canonical refusal text must stay")

if errors:
    print("fixture caustic radiance detail contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture caustic radiance detail contract: PASS")
