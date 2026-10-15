#!/usr/bin/env python3
from pathlib import Path
import re
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sidecar = (root / "../plugin/Source/VideoSidecar.cpp").read_text()


def function_body(text, signature):
    start = text.index(signature)
    # Member functions in VideoSidecar.cpp close with a brace in column 0.
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


probe = function_body(sidecar, "juce::var VideoSidecar::compositeFrameProbe(")
revision_source = function_body(sidecar, "uint64_t VideoSidecar::authoringRevisionForHelper() noexcept")

# The probe takes its revision from the one helper and sends it as both the
# authoring and exportable revision, so a composite probe cannot disagree with
# the export request it mirrors.
sources = re.findall(r"const\s+auto\s+(\w+)\s*=\s*authoringRevisionForHelper\s*\(\s*\)\s*;", probe)
assert len(sources) == 1, "composite probe must read the authoring revision exactly once"
revision_variable = sources[0]

sent = dict(re.findall(
    r'setProperty\s*\(\s*"(authoringRevision|exportableRevision)"\s*,\s*'
    r"static_cast<juce::int64>\s*\(\s*(\w+)\s*\)\s*\)",
    probe))
assert sent.get("authoringRevision") == revision_variable, \
    "composite probe must send authoringRevision from authoringRevisionForHelper()"
assert sent.get("exportableRevision") == revision_variable, \
    "composite probe must send exportableRevision from authoringRevisionForHelper()"

# The lazy first-use initialization is one compare-exchange from 0 to 1, so
# concurrent bumps cannot be overwritten by a stale load-then-store.
assert "compare_exchange_strong" in revision_source
assert "expected = 0" in " ".join(revision_source.split())
assert "graphStructuralRevision_.store" not in revision_source
