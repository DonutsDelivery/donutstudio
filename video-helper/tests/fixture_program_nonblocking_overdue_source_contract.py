#!/usr/bin/env python3
# Contract for the non-blocking Preview poll of an in-flight fixture scene program. A Preview
# prepare polls GL_COMPLETION_STATUS_KHR on every call, however old the submission is, and
# returns pending until the driver reports the link complete. Only then does it finish the
# program. A Blocking prepare still finishes an unfinished link. No time bound may make a
# Preview prepare wait on the driver, because the render loop must never stall in a link.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
cmake = (root / "CMakeLists.txt").read_text()

errors = []


def require(text, needle, label):
    if needle not in text:
        errors.append(label + " is missing " + needle)


def forbid(text, needle, label):
    if needle in text:
        errors.append(label + " must not contain " + needle)


def region(text, begin, end, label):
    start = text.find(begin)
    if start < 0:
        errors.append(label + " is missing " + begin)
        return ""
    stop = text.find(end, start + len(begin))
    if stop < 0:
        errors.append(label + " is missing " + end)
        return ""
    return text[start:stop]


# The overdue bound that finished a link blocking is gone, with its log line and its name.
for stale in ("kFixtureProgramOverdue", "overdue", "async bound", "async-bound", "finished blocking"):
    forbid(backend, stale, "backend_opengl.cpp")

# The in-flight branch. Its Preview poll reads the program completion status before any finish.
flight = region(backend, "if (flying != inFlight.end())", "auto entry = std::move (flying->second);",
                "backend_opengl.cpp in-flight branch")
require(flight, "if (use == NativeFixturePrepareUse::Preview)", "in-flight branch")
require(flight, "loopstep::Scope pollScope (\"gl:fixture-scene-program-poll\");", "in-flight branch")
require(flight, "gl.GetProgramiv (flying->second.program, GL_COMPLETION_STATUS_KHR, &complete);",
        "in-flight branch")
require(flight, "if (complete != GL_TRUE)", "in-flight branch")
require(flight, "result.pending = true;", "in-flight branch")
require(flight, "return result;", "in-flight branch")
# The pending decision depends only on the completion status, never on how old the submission is.
forbid(flight, "steady_clock::now() - flying", "in-flight branch")
forbid(flight, "submitted >", "in-flight branch")
forbid(flight, "installFinished", "in-flight branch")
forbid(flight, "finishFixtureProgram", "in-flight branch")
forbid(flight, "std::chrono::milliseconds (", "in-flight branch")

# Order inside the branch: the poll, then the pending return, and nothing before the poll
# that could wait on the driver.
ordered_needles = [
    "if (use == NativeFixturePrepareUse::Preview)",
    "GL_COMPLETION_STATUS_KHR, &complete);",
    "if (complete != GL_TRUE)",
    "result.pending = true;",
]
at = -1
for needle in ordered_needles:
    found = flight.find(needle, at + 1)
    if found < 0:
        errors.append("in-flight branch is missing, or out of order, " + needle)
        break
    at = found

# The finish stays after the in-flight branch, so a pending program is never finished early.
after_flight = backend[backend.find("auto entry = std::move (flying->second);"):]
if "installFinished (entry)" not in after_flight:
    errors.append("the finished in-flight program must still be installed")

# The pending diagnostic and the finish diagnostic are still emitted.
require(backend, "[gl] fixture program pending:", "backend_opengl.cpp")
require(backend, "[gl] fixture program ready after", "backend_opengl.cpp")

# Registration: the contract runs under ctest next to the async compile contract.
require(cmake, "add_test(NAME fixture_program_nonblocking_overdue_source_contract", "CMakeLists.txt")
require(cmake, "tests/fixture_program_nonblocking_overdue_source_contract.py", "CMakeLists.txt")

if errors:
    print("fixture program nonblocking overdue contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture program nonblocking overdue source contract: PASS")
