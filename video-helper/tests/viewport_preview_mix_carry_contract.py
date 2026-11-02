#!/usr/bin/env python3
# Contract for the preview-mix carry across viewport opens: setAudioMix mirrors
# every publication (including the empty clear) outside Impl, which close()
# resets, and every open path reseeds the fresh Impl from that mirror one
# generation later so the render thread re-copies it. Without this, a
# close/reopen (device fallback, dock switch) starts the viewport empty while
# the plugin's baked/ready state says nothing more is owed, and the stopped
# spectrum gates wait forever for a re-bake alreadyReady never schedules.
from pathlib import Path
import sys

root = Path(sys.argv[1])
viewport = (root / "src/viewport.cpp").read_text()
header = (root / "src/viewport.h").read_text()
errors = []

if "void seedImplPreviewMix();" not in header:
    errors.append("seedImplPreviewMix declaration missing from viewport.h")
for member in ("carriedPreviewMixFeats_", "carriedPreviewSourceFeats_",
               "carriedPreviewMixSr_", "carriedPreviewMixGeneration_"):
    if member not in header:
        errors.append(f"carry member {member} missing from the Viewport (must survive close/reopen)")
if "void Viewport::seedImplPreviewMix()" not in viewport:
    errors.append("seedImplPreviewMix definition missing")
if "impl_->previewMixGeneration = carriedPreviewMixGeneration_ + 1;" not in viewport:
    errors.append("the reseed must advance one generation so the render thread re-copies")
if viewport.count("seedImplPreviewMix();") != 3:
    errors.append("every open path (window, shared, shm) must reseed the carried mix")
if "carriedPreviewMixGeneration_ = impl_->previewMixGeneration;" not in viewport:
    errors.append("setAudioMix must mirror the publication outside Impl")
if "std::lock_guard<std::mutex> carryLock (previewMixCarryMutex_);" not in viewport:
    errors.append("the carry mirror needs its own lock")

if errors:
    print("viewport preview mix carry contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport preview mix carry contract: PASS")
