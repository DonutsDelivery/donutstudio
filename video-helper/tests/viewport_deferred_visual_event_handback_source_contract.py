#!/usr/bin/env python3
# Contract for visual-event handback on the frame paths that leave the tick without
# presenting. The tick swaps im.visualEvents into frameVisualEvents at the top. A frame
# that is deferred, or dropped on a retired-lifecycle commit, never reaches the consumer,
# so its events must return to im.visualEvents for the next produce. The idle gate-skip
# path already does this. The deferred and retired paths must do the same, exactly.
from pathlib import Path
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
errors = []


def normalized(text):
    return " ".join(text.split())


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


# The handback block as the gate-skip path writes it. Each path must carry this same
# statement, so the three copies cannot drift apart.
HANDBACK = normalized(
    """
    if (! frameVisualEvents.empty())
    {
        std::lock_guard<std::mutex> lock (im.mutex);
        im.visualEvents.insert (im.visualEvents.begin(),
                                frameVisualEvents.begin(), frameVisualEvents.end());
        frameVisualEvents.clear();
    }
    """
)

gate = region(viewport, "if (! publishGate.shouldProduce (gateSnapshot))",
              "enterLoopPhase (LoopPhase::FramePacing);")
if HANDBACK not in normalized(gate):
    errors.append("gate-skip path no longer hands back frameVisualEvents")

# The deferred branch hands the events back before it sleeps and continues.
deferred = region(viewport, "if (frameDeferred)\n        {\n            ++deferredConsecutive;",
                  "deferredConsecutive = 0;")
if HANDBACK not in normalized(deferred):
    errors.append("deferred-frame branch does not hand back frameVisualEvents")
else:
    handback_at = normalized(deferred).index(HANDBACK)
    sleep_at = normalized(deferred).index('enterLoopStep ("advance:frame-deferred-sleep");')
    if handback_at > sleep_at:
        errors.append("deferred-frame handback must come before the deferred sleep")
if "continue;" not in deferred:
    errors.append("deferred-frame branch no longer continues the tick")

# The retired-lifecycle retry hands the events back before it paces and continues.
retired_branch = "if (commitError == kPreviewLifecycleRetired && ! im.wantClose.load())"
retired = region(viewport, retired_branch, "closeForDeviceFailure(")
if HANDBACK not in normalized(retired):
    errors.append("retired-lifecycle retry does not hand back frameVisualEvents")
else:
    handback_at = normalized(retired).index(HANDBACK)
    pacing_at = normalized(retired).index("enterLoopPhase (LoopPhase::FramePacing);")
    if handback_at > pacing_at:
        errors.append("retired-lifecycle handback must come before the frame pacing")
if "continue;" not in retired:
    errors.append("retired-lifecycle retry no longer continues the tick")

# The retry branch logs its count and reason, at most once per second.
if viewport.count("uint64_t retiredPublishRetries = 0;") != 1:
    errors.append("retry counter must be declared once")
if "++retiredPublishRetries;" not in retired:
    errors.append("retry branch does not count the retry")
if "if (retiredNow - lastRetiredLog >= std::chrono::seconds (1))" not in retired:
    errors.append("retry log is not rate limited to one line per second")
if '"[viewport] publish retry: reason=%s count=%llu\\n"' not in retired:
    errors.append("retry branch does not log its reason and count")

if errors:
    print("viewport deferred visual event handback contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport deferred visual event handback source contract: PASS")
