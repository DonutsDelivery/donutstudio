#!/usr/bin/env python3
# Contract for the NVIDIA unroll pragmas on the constant-bound fixture shader loops and the
# feature mask on the async scene program log lines. The pragmas keep NVIDIA from unrolling
# the loops into a long link. The log lines carry the feature mask with the key, so a slow
# link can be matched to its program.
from pathlib import Path
import sys

root = Path(sys.argv[1])
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
smooth = (root / "src/gpu_backend/fixture_smooth_glass.h").read_text()
cmake = (root / "CMakeLists.txt").read_text()

PRAGMA = "#pragma optionNV(unroll none)"
errors = []


def require(text, needle, label):
    if needle not in text:
        errors.append(label + " is missing " + needle)


def pragma_before(text, loop, count, label):
    # The loop must follow the pragma on the line directly above it. The pattern must
    # occur exactly `count` times, so a loop added without its pragma is caught.
    found = text.count(PRAGMA + "\n" + loop)
    if found != count:
        errors.append(label + " has " + str(found) + " of " + str(count) + " pragma-guarded loops: " + loop)


# The five loops. Each carries the NVIDIA unroll pragma directly above it.
pragma_before(backend, "    for (int i = 0; i < 8; ++i) {\n        if (i >= count) break;", 1,
              "backend_opengl.cpp compositePass")
pragma_before(backend, "    for (int b=0; b<8; ++b) bands[b] = direct*", 1,
              "backend_opengl.cpp diffraction direct bands")
pragma_before(backend, "    for (int b=0; b<8; ++b) bands[b]=source*", 1,
              "backend_opengl.cpp spectral receiver bands")
pragma_before(smooth, "    for(int channel=0;channel<3;++channel) {", 2,
              "fixture_smooth_glass.h rough glass channel")

# The NVIDIA pragmas stay GLSL-only. The Metal recipe still strips them.
require(smooth, 'smoothglass::replace(source, "' + PRAGMA + '\\n", "");', "fixture_smooth_glass.h Metal recipe")

# The log lines print the feature mask with the key.
pending_line = next((line for line in backend.splitlines() if "[gl] fixture program pending:" in line), "")
ready_line = next((line for line in backend.splitlines() if "[gl] fixture program ready after" in line), "")
if "features=0x%x" not in pending_line or "key=%08x" not in pending_line:
    errors.append("the fixture program pending log line must print features=0x%x and key=%08x")
if "features=0x%x" not in ready_line or "key=%08x" not in ready_line:
    errors.append("the fixture program ready log line must print features=0x%x and key=%08x")

# Registration: the contract runs under ctest next to the nonblocking overdue contract.
require(cmake, "add_test(NAME fixture_program_unroll_and_log_source_contract", "CMakeLists.txt")
require(cmake, "tests/fixture_program_unroll_and_log_source_contract.py", "CMakeLists.txt")

if errors:
    print("fixture program unroll and log contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture program unroll and log source contract: PASS")
