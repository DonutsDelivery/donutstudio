#!/usr/bin/env python3
"""Pin the named refusals and the contact retry for nested glass shells."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
header = (root / "src/gpu_backend/fixture_smooth_glass.h").read_text()
tests = (root / "tests/glass_nested_shell_tests.cpp").read_text()
visibility = (root / "src/gpu_backend/fixture_opaque_visibility.h").read_text()
failures = []


def check(condition, text):
    if not condition:
        failures.append(text)


reasons = {
    "kShellsIntersectReason":
        "Closed glass volume shells intersect or touch. Separate the outer shell from each cavity shell",
    "kShellOrientationReason":
        "Closed glass volume shell orientation must alternate with nesting depth: "
        "the outer shell faces outward and each cavity shell faces inward",
    "kShellOverlapReason":
        "Closed glass volume shells overlap: a point lies inside more than one glass solid",
    "kOpaqueInterpenetrationReason":
        "Closed glass volume and an opaque object interpenetrate. Opaque objects may touch the glass "
        "or sit in an air cavity, but must not pass into the glass solid",
}
for name, text in reasons.items():
    check(f'{name} =\n    "{text}"' in header or f'{name} = "{text}"' in header,
          f"{name} keeps its text")
    check(f"sg::{name}" in tests, f"glass_nested_shell_tests.cpp checks {name}")

check("requires one connected shell" not in header, "the one-shell refusal is gone")
check("kSeamRetries = 3, kSeamContactRetry = 4, kSeamContactCheck = 5;" in header
      and "kSeamContactEnd" not in header, "the contact retries are seams 4 and 5")
check("seamContactProbe" not in header and "seamContactProbe" not in tests,
      "retry 5 looks back from the retry 4 hit itself")
for helper in ("inline bool seamContact(", "inline bool seamContactVerified("):
    check(helper in header, f"{helper} is shared by the walks and the tests")
check(header.count("} else if(seam==4) {") == 2,
      "the released and paired camera walks both meet an opaque contact at retry 4")
check(header.count("} else if(seam==5) {") == 2,
      "the released and paired camera walks both check the glass exit at retry 5")
check("if(seam==5) {float seamWindow=0.5*glassBias(origin);queryOrigin=origin+direction*seamWindow;"
      "queryDirection=-direction;farthest=2.0*0.5*glassBias(queryOrigin);}" in header
      and "if(seam==5) {float seamWindow=0.5*glassBias(rayOrigin);queryOrigin=rayOrigin+rayDirection*seamWindow;"
      "queryDirection=-rayDirection;farthest=2.0*0.5*glassBias(queryOrigin);}" in header,
      "retry 5 looks back from the contact over twice the seam window")
check(header.count("GU_GlassControl,seam==5?volume:uint(0),seam==7?volume:uint(0) GX_QUERY)") == 2,
      "both camera walks prefer the glass in the retry 5 query only and avoid it in the exit probe only")
check("float minimum,float maximum,SV4 control,uint preferred ST_ARGS) {" in visibility
      and "return sceneClosestHit(origin,rayDirection,minimum,maximum,control,uint(0) ST_PASS);" in visibility,
      "the shared scene query takes a preferred object and the old signature prefers none")
check("float minimum, float maximum, std::uint32_t preferred = 0,\n"
      "                                std::uint32_t avoid = 0)" in visibility,
      "the CPU scene query takes the same preferred and avoided objects")
check("float minimum,float maximum,SV4 control,uint preferred,uint avoid ST_ARGS) {" in visibility
      and "return sceneClosestHit(origin,rayDirection,minimum,maximum,control,preferred,uint(0) ST_PASS);" in visibility
      and visibility.count("if(avoid!=uint(0)&&objectId==avoid) continue;") == 1
      and visibility.count("if (avoid!=0 && objectId==avoid) return 0;") == 1,
      "the shader and CPU queries skip the avoided object at the same point")
# The exit child probe (kSeamExitChild 6, kSeamExitProbe 7) in both walks.
check("inline constexpr int kSeamExitChild = 6, kSeamExitProbe = 7;" in header
      and "inline constexpr float kExitProbeBack = 2.0f, kExitProbeReach = 2.5f;" in header,
      "the exit probe states and reach are named")
check("if(seam==7) {float probeBias=glassBias(origin);queryOrigin=origin+back*(2.0*probeBias);"
      "queryDirection=-back;farthest=2.5*probeBias;}" in header
      and "if(seam==7) {float probeBias=glassBias(rayOrigin);queryOrigin=rayOrigin+rayBack*(2.0*probeBias);"
      "queryDirection=-rayBack;farthest=2.5*probeBias;}" in header,
      "the exit probe looks out from one bias inside the exit past the child origin")
check(header.count("if(seam==6) {") == 2 and header.count("} else if(seam==7) {") == 2,
      "both camera walks run the exit probe")
check("media[pending]=!inside;seams[pending]=inside?6:0;++pending;" in header
      and "seams[pending]=transmission&&!thin&&rayInside?6:0;++pending;" in header,
      "only the transmitted child of a glass exit is an exit child")
check("sg::kSeamExitChild" in tests and "sg::kSeamExitProbe" in tests and "cubeBottom(" in tests,
      "the tests walk the exit probe and check the terminal face")
check("kSeamWindow = 0.5f" in header, "the shader's 0.5 is kSeamWindow")
check("sg::seamContactVerified(" in tests and "sg::kSeamContactCheck" in tests
      and "sg::seamQuery(sg::kSeamContactCheck, hit.position, ray.direction)" in tests,
      "the CPU walk in the tests uses the shared retry 5 rule")
check("Chain chain(const Admitted& admitted" in tests and "kPreviewCameraVisits), 48}" in tests,
      "the tests model the seam chain's visits at the preview and export caps")
check("for (const float overlap : {0.0f, 2e-6f})" in tests,
      "the tests cover a contact at exactly 0 overlap")
check("Move or resize the glass so no object passes into the glass solid. "
      "Objects may touch the glass or sit in an air cavity" in header,
      "the medium refusal tells the user what is allowed")
check("opaqueClearOfGlass(" in header, "prepare checks opaque objects against the glass solid")
check("const auto crossesFace = [&](const Face& face, const Face& glassFace)" in header,
      "admission tests each crossing of a glass face exactly")

if failures:
    for failure in failures:
        print("FAIL:", failure)
    sys.exit(1)
print("glass nested shell source contract: ok")
