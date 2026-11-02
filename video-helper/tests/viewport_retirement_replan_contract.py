#!/usr/bin/env python3
# Contract for the card row's retirement fix: when an imported-frame execution is
# rejected with the lifecycle-retired reason, the plan cache is reset before the
# hold, so the next plan re-compiles and re-resolves against the new publication
# instead of holding plan-error on the retired source's state. The commit-time
# retry that drops and re-produces the stale frame must stay.
from pathlib import Path
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
errors = []

reset = ("if (planError == kPreviewLifecycleRetired)\n"
         "                        importedScenePlanCache.reset();")
if reset not in viewport:
    errors.append("a retirement must reset the imported-scene plan cache before the hold")
if viewport.count("importedScenePlanCache.reset();") < 1:
    errors.append("plan cache reset missing")
if "commitError == kPreviewLifecycleRetired && ! im.wantClose.load()" not in viewport:
    errors.append("commit-time retired-frame retry lost")

if errors:
    print("viewport retirement replan contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport retirement replan contract: PASS")
