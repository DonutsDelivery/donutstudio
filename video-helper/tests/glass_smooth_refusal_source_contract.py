#!/usr/bin/env python3
"""Check that every incomplete closed-glass refusal names its cause."""

from pathlib import Path
import re
import struct

root = Path(__file__).resolve().parents[1]
header = (root / "src/gpu_backend/fixture_smooth_glass.h").read_text()
opengl = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
sokol = (root / "src/gpu_backend/backend_sokol.mm").read_text()
cases = (root / "tests/support/smooth_glass_cases.h").read_text()


def negative_half_bits(value):
    # Same packing as negativeHalfBits in fixture_smooth_glass.h.
    magnitude = -value
    exponent = 0
    while (2 << exponent) <= magnitude:
        exponent += 1
    mantissa = ((magnitude << 10) >> exponent) & 0x3FF
    return 0x8000 | ((exponent + 15) << 10) | mantissa


def function_body(signature):
    start = header.index(signature)
    end = header.index("\n}\n", start)
    return header[start:end]


# The packing matches IEEE binary16 for every code the shader can write.
for value in range(-29, 0):
    assert negative_half_bits(value) == struct.unpack("<H", struct.pack("<e", value))[0], value

# Smooth codes are distinct, start at -3 and never collide with -1 or -2.
enum = re.search(r"enum class SmoothTransportStatus : int\s*\{(.*?)\};", header, re.S)
assert enum is not None, "SmoothTransportStatus enum is missing"
codes = {name: int(value) for name, value in re.findall(r"(\w+)=(-?\d+)", enum.group(1))}
# Trapped (+2) is the one non-refusal: a completed walk that absorbed the power
# it could not trace within its visits.
assert codes.pop("Trapped", None) == 2, "SmoothTransportStatus::Trapped must be +2"
assert "inline constexpr std::uint16_t kTrappedNormalBits = 0x4000u;" in header
assert struct.unpack("<H", struct.pack("<e", 2.0))[0] == 0x4000
assert len(codes) == 17, codes
assert sorted(codes.values()) == [c for c in range(-20, -2) if c != -10], codes
assert len({negative_half_bits(v) for v in codes.values()}) == len(codes)
assert negative_half_bits(-1) == 0xBC00 and negative_half_bits(-2) == 0xC000

# Rough refusals (RoughTransportStatus -1 to -10) map to -21 to -30, after the smooth range.
rough_enum = re.search(r"enum class RoughTransportStatus : int\s*\{(.*?)\};", header, re.S)
assert rough_enum is not None, "RoughTransportStatus enum is missing"
rough_refusals = {name: int(value) for name, value in re.findall(r"(\w+)=(-?\d+)", rough_enum.group(1)) if int(value) < 0}
assert sorted(rough_refusals.values()) == [c for c in range(-10, 0) if c != -5], rough_refusals
assert re.search(r"kRoughReasonBase=-20;", header), "rough reason base changed"
assert "constexpr int roughReasonCode(RoughTransportStatus status) { return kRoughReasonBase+int(status); }" in header

# The shader receives one define per code, and every shader use names a define.
defined = dict(re.findall(r'\{"([A-Z_]+)",SmoothTransportStatus::(\w+)\}', header))
assert set(defined.values()) == set(codes) | {"Trapped"}, (defined, codes)
for name in re.findall(r"GLASS_SMOOTH_STATUS_([A-Z_]+)", header):
    assert name in defined, name
assert '#define GLASS_ROUGH_REASON_BASE " + surfacesource' in header
assert "GLASS_ROUGH_REASON_BASE+status" in header

# The C++ reader maps every code to a reason with a distinct binary16 pattern.
table = re.search(r"kIncompleteNormalReasons\[\]\{(.*?)\n\};", header, re.S)
assert table is not None, "kIncompleteNormalReasons is missing"
reasons = dict(re.findall(r'\{negativeHalfBits\(int\(SmoothTransportStatus::(\w+)\)\),\s*"([^"]+)"\}', table.group(1)))
assert set(reasons) == set(codes), (reasons, codes)
rough_reasons = dict(re.findall(r'\{negativeHalfBits\(roughReasonCode\(RoughTransportStatus::(\w+)\)\),\s*"([^"]+)"\}', table.group(1)))
assert set(rough_reasons) == set(rough_refusals), (rough_reasons, rough_refusals)
assert '{negativeHalfBits(-1), "no reason recorded"}' in table.group(1)
assert reasons["QueryBudget"] == "query refused: work budget"
assert reasons["ThinFirstHit"] == "first hit is not the water surface"
assert reasons["ThinFirstQuery"] == "first water-entry query refused or nonfinite"
assert "{std::uint16_t(0), \"unrecognized normal value\"}" in table.group(1)

# completeFrame keeps the message prefix and reports the first reason and counts.
assert 'result (";' in header, "completeFrame prefix changed"
assert '"; also "' in header and '" pixel"' in header and '" pixels"' in header

# Every early return in glassTransport sets a code. The function has 12 such
# exits. Running out of visits is no longer one: the walk ends, absorbs the
# untraced branches and reports TRAPPED on a complete pixel.
start = header.index("vec3 glassTransport(vec3 camera,vec3 entry,out bool complete,out int status GX_ARGS) {")
body = header[start:header.index("// Native callers must retain", start)]
assert "complete=false;status=GLASS_SMOOTH_STATUS_UNCLASSIFIED;" in body
exits = [line for line in body.splitlines() if "return vec3(0.0);" in line]
assert len(exits) == 11, len(exits)
assert "if(pending!=0) trapped=true;" in body
# A path at the depth limit ends: the pixel keeps what it gathered.
assert "if(depth>=GLASS_INTERFACE_DEPTH) {trapped=true;continue;}" in body
assert "GLASS_SMOOTH_STATUS_DEPTH" not in header
# A rough path at the depth limit ends too. Both closed walks report the path's
# tracked power as absorbed, which closes the receipt but does not count as unresolved.
assert "if(depth>=GLASS_INTERFACE_DEPTH) {tail[channel]+=power;continue;}" not in header
assert header.count("if(depth>=GLASS_INTERFACE_DEPTH) {masked[channel]+=power;continue;}") == 2
assert header.count("if(tail[channel]>GLASS_ROUGH_UNRESOLVED)") == 2
assert "GLASS_ROUGH_STATUS_DEPTH" not in header
# A rough walk out of visits or stack ends its branches as absorbed, heaviest visited first.
assert "if(pending!=0) {status=GLASS_ROUGH_STATUS_VISITS;" not in header
assert "if(pending>=16) {status=GLASS_ROUGH_STATUS_PENDING;" not in header
assert header.count("for(int slot=0;slot<pending;++slot) masked[channel]+=powers[slot];") == 2
assert header.count("if(powers[free]<=branchPower) masked[channel]+=powers[free];") == 2
assert header.count("if(powers[slot]>powers[heaviest]) heaviest=slot;") == 2
assert header.count("if(visit+1>=GLASS_ROUGH_VISITS) {masked[channel]+=power;continue;}") == 4
assert header.count("if(masked[channel]+tail[channel]>GLASS_ROUGH_UNRESOLVED)") == 1
assert "inline constexpr float kMaximumRoughUnresolvedPower = 0.0001f;" in header
assert "inline constexpr float kRoughEnergyConservationTolerance = 0.00002f;" in header
assert "if(complete&&trapped) status=GLASS_SMOOTH_STATUS_TRAPPED;" in body
assert "GLASS_SMOOTH_STATUS_VISITS" not in body

# completeFrame admits trapped pixels at any count.
assert "if (alpha == kTrappedNormalBits) {" in header
assert "admitTrappedCameraPixels" not in header
assert "kMaximumTrappedCamera" not in header
# A zero-roughness rough delegation publishes a trapped smooth walk too.
assert header.count("status=smoothComplete?GLASS_ROUGH_STATUS_SMOOTH:GLASS_ROUGH_STATUS_QUERY;") == 3
for line in exits:
    assert re.search(r"status=[^;]*GLASS_SMOOTH_STATUS_", line), line

# Every early return in the thin water path sets a status. Receiver refusals
# are forwarded through the out status, so their caller returns only after the
# receiver has named the reason.
receiver = function_body("vec3 thinInterfaceReceiver(vec3 origin,vec3 direction,uint water,out bool complete,out int status GX_ARGS) {")
assert "complete=false;status=GLASS_SMOOTH_STATUS_UNCLASSIFIED;" in receiver
receiver_exits = [line for line in receiver.splitlines() if "return vec3(0.0);" in line]
assert len(receiver_exits) == 2, receiver_exits
for line in receiver_exits:
    assert re.search(r"status=[^;]*GLASS_SMOOTH_STATUS_", line), line
assert "if(!complete) status=GLASS_SMOOTH_STATUS_THIN_RECEIVER;" in receiver

thin = function_body("vec3 thinInterfaceTransport(vec3 camera,vec3 entry,vec3 evaluatedNormal,out bool complete,out int status GX_ARGS) {")
assert "complete=false;status=GLASS_SMOOTH_STATUS_UNCLASSIFIED;" in thin
thin_exits = [line for line in thin.splitlines() if "return vec3(0.0);" in line]
assert len(thin_exits) == 6, thin_exits
forwarded = 0
for line in thin_exits:
    if re.search(r"if\(!(reflected|transmitted)Complete\) return vec3\(0\.0\);", line):
        forwarded += 1
        continue
    if line.strip() == "if(entryStatus!=0) {status=entryStatus;return vec3(0.0);}":
        continue
    assert re.search(r"status=[^;]*GLASS_SMOOTH_STATUS_", line), line
assert forwarded == 2
assert "reflectedComplete,status GX_PASS" in thin and "transmittedComplete,status GX_PASS" in thin
assert "if(!complete) status=GLASS_SMOOTH_STATUS_THIN_RADIANCE;" in thin

# The camera query may miss or rank another surface first by a precision tie
# where the raster drew water. thinEntryOwned keeps raster ownership within a
# relative tolerance that the C++ twin shares, and still refuses a clear occluder.
owned = function_body("bool thinEntryOwned(GlassClosestHit first,uint water,vec3 camera,vec3 entry) {")
assert "if(first.status==1&&first.objectId==water) return true;" in owned
assert "if(first.status==0) return true;" in owned
assert "if(first.status!=1) return false;" in owned
assert "float tolerance=thinEntryTolerance(camera,entry);" in owned
assert "return glassSceneFinite(first.distance)&&first.distance>=distance-tolerance;" in owned
tolerance = function_body("float thinEntryTolerance(vec3 camera,vec3 entry) {")
assert "return max(4.0*glassBias(entry),1.0e-4*length(entry-camera));" in tolerance
assert "inline constexpr float kThinEntryRelativeTolerance = 1.0e-4f;" in header
assert "kThinEntryRelativeTolerance*distance" in header

# An occluder silhouette tie gets one probe along the same ray, one tolerance
# past the foreign hit. A closed occluder's exit or any other surface in front
# of the entry still refuses.
status = function_body("int thinEntryStatus(GlassClosestHit first,uint water,vec3 camera,vec3 incoming,vec3 entry GX_ARGS) {")
assert "if(thinEntryOwned(first,water,camera,entry)) return 0;" in status
assert "if(first.status!=1||!glassSceneFinite(first.distance)) return GLASS_SMOOTH_STATUS_THIN_FIRST_QUERY;" in status
assert "GlassClosestHit beyond=glassSceneClosestHit(camera,incoming,first.distance+thinEntryTolerance(camera,entry),3.402823e38,GU_GlassControl GX_QUERY);" in status
assert "if(beyond.status<0) return beyond.status==-2?GLASS_SMOOTH_STATUS_QUERY_BUDGET:GLASS_SMOOTH_STATUS_QUERY_REFUSED;" in status
assert "return thinEntryOwned(beyond,water,camera,entry)?0:GLASS_SMOOTH_STATUS_THIN_FIRST_HIT;" in status
assert status.count("glassSceneClosestHit(") == 1
assert "int entryStatus=thinEntryStatus(first,water,camera,incoming,entry GX_PASS);" in thin
assert "if(entryStatus!=0) {status=entryStatus;return vec3(0.0);}" in thin
# The admitted thin pixel budget counts the probe.
assert "inline constexpr std::uint64_t kThinPixelRayQueries = 4;" in header
assert "kMaximumThinPixelRayQueries/kThinPixelRayQueries" in header
assert "first.status!=1 || first.objectId!=water" not in thin

# completeFrame names the first failing readback texel after the reason list.
assert '" first at texel x="' in header

# The rough helper returns the shared reason range for refusals only.
helper = function_body("int glassRoughReason(int status) {")
assert "status>=GLASS_ROUGH_STATUS_CLOSURE&&status<=GLASS_ROUGH_STATUS_INPUT" in helper
assert "GLASS_ROUGH_REASON_BASE+status : GLASS_SMOOTH_STATUS_UNCLASSIFIED" in helper

# Both native writers start from an unclassified code and write the status
# of every branch. No refused pixel keeps the released -1 unless unknown.
assert opengl.count("int glassStatus=GLASS_SMOOTH_STATUS_UNCLASSIFIED;") == 1
assert sokol.count("int glassStatus=GLASS_SMOOTH_STATUS_UNCLASSIFIED;") == 1
assert "int glassStatus=-1;" not in opengl and "int glassStatus=-1;" not in sokol
assert (
    "outRawNormal.w = complete && representable ? (glassStatus == GLASS_SMOOTH_STATUS_TRAPPED ? 2.0 : 1.0) : representable ? float(glassStatus) : -2.0;"
    in opengl
)
assert (
    "out.rawNormal.w = complete && representable ? (glassStatus == GLASS_SMOOTH_STATUS_TRAPPED ? 2.0f : 1.0f) : representable ? float(glassStatus) : -2.0f;"
    in sokol
)
assert "glassTransport(uCameraTranslation,vWorldPosition,complete,glassStatus);" in opengl
assert (
    "glassTransport(u.cameraTranslation.xyz,in.worldPosition,complete,glassStatus,u,opaqueTexture,glassTexture,environmentTexture);"
    in sokol
)
assert "thinInterfaceTransport(uCameraTranslation,vWorldPosition,normal,complete,glassStatus);" in opengl
assert (
    "thinInterfaceTransport(u.cameraTranslation.xyz,in.worldPosition,normal,complete,glassStatus,u,opaqueTexture,glassTexture,environmentTexture)"
    in sokol
)
assert opengl.count("glassStatus = glassRoughReason(status);") == 2
assert sokol.count("glassStatus = glassRoughReason(status);") == 1
assert "outRawNormal.w=complete && representable?1.0:representable?float(glassRoughReason(int(result.w))):-2.0;" in opengl
assert "representable?-1.0:-2.0" not in opengl and "representable?-1.0:-2.0" not in sokol
assert "smoothComplete,smoothStatus GX_PASS" in header
assert header.count("smoothComplete,smoothStatus GX_PASS") == 3
assert "vec3 smoothColour=glassTransport(camera,entry,smoothComplete,smoothStatus GX_PASS);" in cases

# Bright finite radiance is held at the RGBA16F ceiling. Only NaN or Inf
# marks a glass pixel unrepresentable, and the Color funnel keeps NaN/Inf.
assert opengl.count("bool representable = glassSceneFinite3(glass);") == 1
assert sokol.count("bool representable = glassSceneFinite3(glass);") == 1
assert "glass = clamp(glass, vec3(-65504.0), vec3(65504.0));" in opengl
assert "glass = clamp(glass, float3(-65504.0f), float3(65504.0f));" in sokol
assert "vec4 fixtureHalfRange(vec4 color) {" in opengl
assert "float4 fixtureHalfRange(float4 color) {" in sokol
assert "isnan(color)" in opengl and "isinf(color)" in opengl
assert "isnan(color) || isinf(color)" in sokol
assert "if (count == 0) return fixtureHalfRange(color);" in opengl
assert "if (count == 0) return fixtureHalfRange(value.color);" in sokol
assert "uPassProgramControl.z > 0.5 ? fixtureHalfRange(value) : clamp(value, 0.0, 1.0);" in opengl
assert "u.passProgramControl.z > 0.5f ? fixtureHalfRange(result) : clamp(result, 0.0f, 1.0f);" in sokol
assert "kMaximumClosedGlassFrameRayQueries = 2048ull * 1024ull * 1024ull" in header
assert ": kMaximumClosedGlassFrameRayQueries/(3*kMaximumRayVisits+1);" in header

# Smooth Dispersive Glass shades its receivers with the analytic sky. A GGX
# sample on the view direction rounds v.h above one, and pow() of the negative
# Schlick base made NaN on NVIDIA GL, so every export refused. v.h is clamped.
sky = (root / "src/gpu_backend/fixture_analytic_sky.h").read_text()
assert "float vh=min(dot(v,h),1.0);" in sky
assert "float vh=dot(v,h);" not in sky
assert sky.count("pow(1.0-vh,5.0)") == 2
# -2 pixels and nonfinite Color values report a count and the first texel.
assert "(NaN or infinite glass radiance; " in header
assert '"NaN" : "infinity")+" in channel "' in header

print("glass smooth refusal source contract: ok")
