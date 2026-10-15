#!/usr/bin/env python3
# Contract for the stale-frame retry on the preview commit path. A seek pushes a new
# lifecycle while a frame is in flight. The Imported Frame publication then fails with
# the retired-source reason. That must drop the frame and retry, not close the channel,
# which left the loop wedged after a seek.
from pathlib import Path
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
errors = []

RETIRED = "Imported Frame preview source or lifecycle was retired"
CONST = "constexpr const char* kPreviewLifecycleRetired ="


def region(text, start, end):
    begin = text.find(start)
    if begin < 0:
        errors.append("missing: " + start)
        return ""
    stop = text.find(end, begin + len(start))
    if stop < 0:
        errors.append("missing end: " + end)
        return text[begin:]
    return text[begin:stop + len(end)]


# The retired-source reason is one named constant, used by the activity check and
# by the commit-path comparison. A literal copy could drift from the comparison.
if viewport.count(CONST) != 1 or viewport.count(RETIRED) != 1:
    errors.append("retired-source reason must be defined once, as the named constant")
if "failure = kPreviewLifecycleRetired; return false;" not in viewport:
    errors.append("activity check does not return the named retired-source reason")

# The commit block: the retired case is handled before the device-close call.
commit = region(viewport, "if (framePublished)\n        {\n            std::string commitError;",
                "enterLoopPhase (LoopPhase::Publish);")
retired_branch = "if (commitError == kPreviewLifecycleRetired && ! im.wantClose.load())"
close_call = 'closeForDeviceFailure("temporal sampling could not be committed", commitError);'
if retired_branch not in commit:
    errors.append("commit block does not branch on the retired-source reason")
elif commit.index(retired_branch) > commit.index(close_call):
    errors.append("retired-source branch must come before the device-close call")
if close_call not in commit:
    errors.append("device-close call for other commit failures was removed")

# The retired branch drops the frame, tells the gate that nothing was published so it
# produces again, paces, and continues without closing. It never sets wantClose.
retry = region(commit, retired_branch, "closeForDeviceFailure(")
for needle in (
    "temporalSamplingTransaction.discard();",
    "if (offscreen) publishGate.finish (gateSnapshot, false, false, shm);",
    "std::this_thread::sleep_for (std::chrono::milliseconds (10));",
    "nextTickNs = nowNs();",
    "continue;",
):
    if needle not in retry:
        errors.append("retry branch missing: " + needle)
if "wantClose" in retry.replace(retired_branch, ""):
    errors.append("retry branch must not close the channel")

# The gate marks an unpublished render as pending, so the next tick produces even when
# a seek lands on the same displaySec and no other input changed.
gate = (root / "src/viewport_publish_gate.h").read_text()
if "if (! published)\n        {\n            pending_ = true;" not in gate:
    errors.append("publish gate no longer forces a produce after an unpublished render")

# Other commit failures keep closing the channel with their reason, so the device-close
# count in sdf_product_route_source_contract.py is unchanged.
if viewport.count('closeForDeviceFailure("temporal sampling could not be committed"') != 1:
    errors.append("temporal sampling commit must keep exactly one device-close call")

if errors:
    print("viewport retired-publication contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport retired-publication source contract: PASS")
