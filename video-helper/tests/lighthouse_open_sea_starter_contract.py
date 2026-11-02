#!/usr/bin/env python3
"""Pins what the open Lighthouse and Ocean starter authors for its R1 frame.

The builder writes the glass-dome branch first and the open branch second, so
the last authored value of a label is the open starter's. The glint section runs
glintPath's metric from open_sea_sky_tests.cpp on those values, with the same
thresholds. The C++ test measures a sea of its own and this one measures the
starter's, from the camera and sun the starter authors.
"""
from pathlib import Path
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
failures = []


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def check(condition, message):
    if not condition:
        failures.append(message)


def open_values():
    text = read("plugin/Source/graph/BuiltInVisualModules.h")
    begin = text.index("inline Build build (bool glassDome)")
    body = text[begin:text.index("\n}\n", begin)]
    values = {}
    for label, pairs in re.findall(r'TRANS\("([^"]+)"\),\s*\{(\{"[^;]*?\})\}\)', body):
        values[label] = {key: float(value)
                         for key, value in re.findall(r'\{"(\w+)", (-?[0-9.]+)\}', pairs)}
    return values


STARTER = open_values()
sky = STARTER["Ocean Sky or HDR Environment"]
wave = STARTER["Ocean Waves"]
grid = STARTER["Ocean Grid"]
camera = STARTER["Ocean Camera"]
camera_position = STARTER["Ocean Camera Position"]

# Ambient: Path Trace adds it from every direction, so the open starter authors none.
builder = read("plugin/Source/graph/BuiltInVisualModules.h")
check("const double sceneAmbient = glassDome ? 0.3 : 0.0;" in builder, "only the dome variant keeps scene ambient")
check('"ambientRed", 0.3' not in builder[builder.index("inline Build build (bool glassDome)"):builder.index("Ring Scene Object")],
      "the starter's objects take their ambient from sceneAmbient")
check(sky.get("red", 0) == 1 and sky.get("intensity", 0) == 1, "the sky is untinted, not the node's 0.05 default")
check(STARTER["Render Lighthouse and Ocean"].get("viewTransform") == 1, "the open starter renders through AgX")
check(STARTER["Render Lighthouse and Ocean"].get("renderEngine") == 1,
      "the open starter opens on Path Trace, since Raster refuses its glass receivers")

# Sea: absorption over a floor gives its colour.
floor_top = STARTER["Sea Floor Position"]["translationY"] + STARTER["Sea Floor"]["height"] / 2
check(0.92 ** (2 * -floor_top) < 1.0e-3, "the sea floor is deep enough that the water absorbs it, so the sea reads deep")
check("glassDome ? std::array<double, 3> {1.0, 1.0, 1.0} : std::array<double, 3> {0.45, 0.92, 0.80}" in builder,
      "the open water absorbs red first, and the dome's water stays clear")

# Clouds: the same four fields the cloud source contract follows through the wire.
node = read("plugin/Source/graph/Visual3DGraphNodes.h")
for field in ("cloudCoverage", "cloudType", "cloudDrift", "cloudHeadingDegrees"):
    check(f'"{field}"' in node and field in sky, f"the open starter authors {field} on the node that lowers it")
check(0 < sky.get("cloudCoverage", 0) < 1, "the layer is broken cloud, neither clear nor closed")
check(sky.get("cloudType", 0) > 0, "the layer mixes streaks into the puffs")
check(sky.get("cloudDrift", 0) != 0, "the layer has an authored position along its heading")
check(sky.get("cloudHeadingDegrees", 0) != 0, "the layer has an authored heading")
check(sky.get("skyZenithBlue", 0) > 2 * sky.get("skyZenithRed", 1), "the clear sky between clouds is blue")
check(sky.get("skyGroundBlue", 0) > 2 * sky.get("skyGroundRed", 1), "water tints transmitted rays toward sea colour")

# Clouds: Path Trace is the judged engine, so its sky has to read the authored layer.
check("cloudCoverage" in read("video-helper/src/pathtrace/lights.cpp"), "the Path Trace sky reads the cloud layer")

# Waves: the wind path, on a grid that clears the far plane test and admits the field.
lowering = read("shared/GeometryFieldLowering.h")
core = read("plugin/Source/graph/GeometryCoreGraphNodes.h")
check('"windSpectrum"' in core and '"windSpeed"' in core and '"windHeading"' in core,
      "the wave node lowers the wind settings")
check(wave.get("windSpectrum") == 1, "the open sea uses the wind spectrum")
check(0.1 <= wave.get("windSpeed", 0) <= 60, "validWindSpectrumSettings admits the wind speed")
check(abs(wave.get("windHeading", 1000)) <= 360, "validWindSpectrumSettings admits the wind heading")
check(wave.get("amplitude", 0) > 0, "the sea has height")
check("const auto shortest=4*std::min(grid.spacingX,grid.spacingY);" in lowering,
      "the wind band still starts at four grid cells")
columns, rows, spacing = grid["columns"], grid["rows"], grid["spacing"]
check(columns == rows and 2 <= columns <= 1024, "the Geometry Grid node admits the sea's size")
half_extent = (columns - 1) * spacing / 2
check(half_extent >= 120, f"the sea reaches toward the horizon, half extent {half_extent}")
corner = math.hypot(half_extent, half_extent + camera_position["translationZ"])
check(camera["farPlane"] > corner, f"the far plane {camera['farPlane']} clears the sea corner {corner}")
# A wave crest three deviations up stays under the rock top and the tower base.
rms = wave.get("amplitude", 0) * (wave.get("windSpeed", 0) / 10) ** 2
rock_top = STARTER["Nearby Rock Position"]["translationY"] + STARTER["Nearby Rock"]["height"] / 2
check(3 * rms < rock_top, f"a 3 sigma crest {3 * rms} stays under the rock top {rock_top}")

# Lantern room: gallery, closed glass room, lamp inside it, roof above it.
tower_top = STARTER["Lighthouse Position"]["translationY"] + STARTER["Lighthouse Tower"]["height"] / 2
span = lambda mesh, position: (STARTER[position]["translationY"] - STARTER[mesh]["height"] / 2,
                               STARTER[position]["translationY"] + STARTER[mesh]["height"] / 2)
gallery = span("Lantern Gallery", "Lantern Gallery Position")
glass = span("Lantern Glass Shell", "Lantern Glass Position")
lamp = span("Lighthouse Lantern", "Lantern Position")
cavity_half = STARTER["Lantern Glass Cavity"]["height"] / 2
glass_centre = STARTER["Lantern Glass Position"]["translationY"]
check(abs(gallery[0] - tower_top) < 1e-9, "the gallery sits on the tower top")
check(STARTER["Lantern Gallery"]["width"] > STARTER["Lighthouse Tower"]["width"], "the gallery overhangs the tower")
check(glass[0] > gallery[1], "the glass clears the gallery, so no paint sits inside its rays")
check(glass_centre - cavity_half < lamp[0] and lamp[1] < glass_centre + cavity_half, "the lamp fits in the air cavity")
for axis in ("width", "depth"):
    check(STARTER["Lighthouse Lantern"][axis] < STARTER["Lantern Glass Cavity"][axis]
          < STARTER["Lantern Glass Shell"][axis], f"lamp, cavity and shell nest in {axis}")
roof_bottom = (STARTER["Lantern Roof Position"]["translationY"]
               - STARTER["Lantern Roof Shell"]["radius"] * STARTER["Lantern Roof Shape"]["scaleY"])
check(roof_bottom > glass[1], "the roof clears the glass")
check(STARTER["Lantern Glass Cavity Mirror"].get("scaleX") == -1, "the cavity winds inward, like the dome's")


# Glint: ports of stableRandom and windSpectrumWaves, then the metric.
MASK = (1 << 64) - 1
fields = read("shared/GeometryFieldOperations.h")
check("kWindSpectrumRings = 12" in fields and "kWindSpectrumDirections = 8" in fields,
      "this port still matches the spectrum's size")
metric = read("video-helper/tests/open_sea_sky_tests.cpp")
for threshold in ("glint.on>.01", "glint.on>10*glint.off", "std::abs(glint.peakAzimuth-azimuth)<=5", "glint.pathRows>=60"):
    check(threshold in metric, f"this port still applies {threshold}")


def stable_random(identity, seed):
    value = identity ^ ((seed + 0x9e3779b97f4a7c15) & MASK)
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK
    value ^= value >> 31
    return (value >> 11) * 2.0 ** -53


def wind_waves(speed, heading, rms_height, shortest, longest):
    tau = 2 * math.pi
    scale = speed * speed / 9.81
    waves, energy = [], 0.0
    for ring in range(12):
        k = tau / (shortest * (longest / shortest) ** ((ring + 0.5) / 12))
        for direction in range(8):
            identity = ring * 8 + direction
            offset = ((direction + 0.5 + 0.8 * (stable_random(identity, 1) - 0.5)) / 8 - 0.5) * tau / 2
            angle = math.radians(heading) + offset
            amplitude = math.exp(-0.5 / (k * scale) ** 2) * math.cos(offset) / k
            waves.append([k * math.cos(angle), k * math.sin(angle), amplitude, stable_random(identity, 2) * tau])
            energy += amplitude * amplitude
    gain = rms_height / math.sqrt(energy / 2) if energy > 0 else 0.0
    return [(a, b, c * gain, d) for a, b, c, d in waves]


def height(waves, x, y):
    return sum(amplitude * math.sin(kx * x + ky * y + phase) for kx, ky, amplitude, phase in waves)


def cloud_noise(x, y):
    def hash2(p, q):
        a = (p * 0.1031) % 1.0
        b = (q * 0.1031) % 1.0
        c = a
        shift = a * (b + 33.33) + b * (c + 33.33) + c * (a + 33.33)
        return (((a + shift) + (b + shift)) * (c + shift)) % 1.0
    cell_x, cell_y = math.floor(x), math.floor(y)
    fx, fy = x - cell_x, y - cell_y
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    low = hash2(cell_x, cell_y) * (1 - fx) + hash2(cell_x + 1, cell_y) * fx
    high = hash2(cell_x, cell_y + 1) * (1 - fx) + hash2(cell_x + 1, cell_y + 1) * fx
    return low * (1 - fy) + high * fy


def smoothstep(low, high, x):
    t = min(max((x - low) / (high - low), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# analyticsky::cloudOpacity in doubles. Its floats differ by rounding only.
def cloud_opacity(d):
    if sky.get("cloudCoverage", 0) <= 0 or d[1] <= 0:
        return 0.0
    kind, coverage = sky["cloudType"], sky["cloudCoverage"]
    angle = math.radians(sky["cloudHeadingDegrees"])
    along_x, along_z = math.sin(angle), math.cos(angle)
    px, pz = d[0] / (d[1] + 0.1), d[2] / (d[1] + 0.1)
    qx = (px * along_x + pz * along_z - sky["cloudDrift"]) * (1.6 + (0.35 - 1.6) * kind)
    qy = (px * along_z - pz * along_x) * (1.6 + (2.8 - 1.6) * kind)
    value, amplitude = 0.0, 0.5
    for _ in range(5):
        value += amplitude * cloud_noise(qx, qy)
        qx, qy = qx * 2.03 + 17, qy * 2.03 + 9
        amplitude *= 0.5
    edge = 0.12 + (0.3 - 0.12) * kind
    threshold = 0.85 + (-edge - 0.85) * coverage
    return smoothstep(threshold, threshold + edge, value) * (0.96 + (0.7 - 0.96) * kind) * smoothstep(0.015, 0.14, d[1])


AZIMUTH = sky["sunAzimuthDegrees"]
ELEVATION = sky["sunElevationDegrees"]
sun = (math.sin(math.radians(AZIMUTH)) * math.cos(math.radians(ELEVATION)), math.sin(math.radians(ELEVATION)),
       math.cos(math.radians(AZIMUTH)) * math.cos(math.radians(ELEVATION)))
check(cloud_opacity(sun) < 0.25, f"clouds leave the sun open, opacity {cloud_opacity(sun)}")

# The cap in the sun's red channel over the starter's sky. A dark sky isolates the
# sun as in sunOverDarkSky, and the lit test below keeps the real cloud layer.
peak = sky["sunRed"] * sky["sunIntensity"] * sky["intensity"]
cosine = math.cos(math.radians(sky["sunRadiusDegrees"]))
cloud_lit = peak * (1 - cosine) * max(sun[1], 0.0)
cam_y, cam_z = camera_position["translationY"], camera_position["translationZ"]
step = spacing
waves = wind_waves(wave.get("windSpeed", 10), wave.get("windHeading", 0), rms, 4 * spacing, (columns - 1) * spacing)

# Water sits at rotationX -90, so world z is minus grid y. Bearings are measured
# around the camera with the starter's sun in the middle, out to 60 degrees: the
# off lobe of the C++ metric starts at 20, and the far flanks hold no sun at all.
AZIMUTHS, RANGES = 240, 300
profile = [0.0] * AZIMUTHS
path = [False] * RANGES
hidden = lit = 0.0
for a in range(AZIMUTHS):
    relative = -60 + a * 0.5
    bearing = math.radians(AZIMUTH + relative)
    for r in range(RANGES):
        distance = 15 * 40 ** (r / (RANGES - 1))
        x, z = distance * math.sin(bearing), cam_z + distance * math.cos(bearing)
        here = height(waves, x, -z)
        slope_x = (height(waves, x + step, -z) - here) / step
        slope_z = (height(waves, x, -(z + step)) - here) / step
        length = math.sqrt(slope_x * slope_x + slope_z * slope_z + 1)
        n = (-slope_x / length, 1 / length, -slope_z / length)
        v = (-x, cam_y - here, cam_z - z)
        view = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        v = (v[0] / view, v[1] / view, v[2] / view)
        vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2]
        if vn <= 0:
            continue
        reflected = (2 * vn * n[0] - v[0], 2 * vn * n[1] - v[1], 2 * vn * n[2] - v[2])
        cap = min(max((reflected[0] * sun[0] + reflected[1] * sun[1] + reflected[2] * sun[2] - cosine) / (1 - cosine), 0.0), 1.0)
        if cap <= 0:
            continue
        cloud = cloud_opacity(reflected)
        radiance = peak * cap * cap * (3 - 2 * cap)
        radiance += (cloud_lit - radiance) * cloud
        radiance *= 0.02 + 0.98 * (1 - vn) ** 5
        profile[a] += radiance / RANGES
        lit += 1
        hidden += cloud
        if abs(relative) <= 5:
            path[r] = True
on_values = [value for a, value in enumerate(profile) if abs(-60 + a * 0.5) <= 5]
off_values = [value for a, value in enumerate(profile) if abs(-60 + a * 0.5) >= 20]
on, off = sum(on_values) / len(on_values), sum(off_values) / len(off_values)
peak_azimuth = -60 + max(range(AZIMUTHS), key=profile.__getitem__) * 0.5
path_rows = sum(path)
check(on > 0.01, f"the path carries sun radiance, {on}")
check(on > 10 * off, f"glintRatio is at least 10, on {on} off {off}")
check(abs(peak_azimuth) <= 5, f"the brightest azimuth follows the sun, {peak_azimuth}")
check(path_rows >= 60, f"the glint spreads into a path, rows {path_rows}")
check(lit > 0 and hidden / lit < 0.5, f"clouds do not hide the mirrored sun, mean opacity {hidden / max(lit, 1.0)}")

# The sun's bearing has to miss the tower, or the tower stands on its own glint path.
tower = STARTER["Lighthouse Position"]
tower_bearing = math.degrees(math.atan2(tower["translationX"], tower["translationZ"] - cam_z)) % 360
half_width = math.degrees(math.atan2(STARTER["Nearby Rock"]["width"] / 2, cam_z - tower["translationZ"]))
check(abs(AZIMUTH - tower_bearing) > half_width + 2,
      f"the sun at {AZIMUTH} clears the rock at {tower_bearing} plus or minus {half_width}")
vertical = camera["verticalFovDegrees"] / 2
pitch = camera_position["rotationX"]
check(pitch + vertical > 0 > pitch - vertical, "the horizon is in frame")

# The reference frame: horizon 72% down, the camera just over the water, the sun in shot.
tangent = math.tan(math.radians(vertical))
aspect = 16 / 9


def frame_row(elevation):
    return 0.5 - math.tan(math.radians(elevation - pitch)) / tangent / 2


horizon_row = frame_row(0)
check(0.69 <= horizon_row <= 0.75, f"the horizon sits at the reference's 0.72 of the frame, {horizon_row}")
check(cam_y <= 1, f"the camera sits just over the water, height {cam_y}")
tower_top = (STARTER["Lantern Roof Position"]["translationY"]
             + STARTER["Lantern Roof Shell"]["radius"] * STARTER["Lantern Roof Shape"]["scaleY"])
tower_range = cam_z - tower["translationZ"] - STARTER["Lantern Gallery"]["depth"] / 2
tower_row = frame_row(math.degrees(math.atan2(tower_top - cam_y, tower_range)))
check(0.05 <= tower_row <= 0.35, f"the tower rises well above the horizon and stays in frame, top row {tower_row}")
crest = max(height(waves, 0.25 * i - 10, 0.25 * j - cam_z - 10) for i in range(81) for j in range(81))
check(cam_y - crest > camera["nearPlane"], f"crests {crest} clear the camera at {cam_y} by the near plane")
sun_row = frame_row(ELEVATION)
sun_column = 0.5 + math.tan(math.radians(180 - AZIMUTH)) / (tangent * aspect) / 2
radius_rows = math.tan(math.radians(sky["sunRadiusDegrees"])) / tangent / 2
check(radius_rows < sun_row < horizon_row - radius_rows and 0 < sun_column < 1,
      f"the whole sun disc is in shot above the horizon, row {sun_row} column {sun_column}")

# Why the glint lands where it does. Water seen g degrees below the horizon mirrors the
# sun at elevation e when the wave tilts by (e - g) / 2 toward the camera. The frame's
# lowest water is g = vertical - pitch, so the whole water band needs tilts up to
# max(e, vertical - pitch - e) / 2. The sea has to carry that slope within two deviations,
# or the path breaks up before it reaches the camera.
slope_sum = count = 0
for i in range(120):
    for j in range(120):
        x, y = 0.5 * i - 30, 0.5 * j - 30
        here = height(waves, x, y)
        slope_sum += ((height(waves, x + 0.05, y) - here) / 0.05) ** 2 + ((height(waves, x, y + 0.05) - here) / 0.05) ** 2
        count += 1
slope_rms = math.sqrt(slope_sum / count)
lowest_water = vertical - pitch
needed_tilt = max(ELEVATION, abs(lowest_water - ELEVATION)) / 2
check(math.tan(math.radians(needed_tilt)) < 2 * slope_rms,
      f"tilt {needed_tilt} degrees is within two slope deviations of {slope_rms}")
check(slope_rms >= 0.1, f"the waves are steep enough to break the sun's reflection up, slope {slope_rms}")

# The path as the camera sees it: mirrored sun in the on-axis water, per frame row.
lit_rows = set()
for a in range(-24, 25):
    bearing = math.radians(AZIMUTH + a * 0.25)
    for r in range(400):
        distance = 2.5 * 160 ** (r / 399)
        x, z = distance * math.sin(bearing), cam_z + distance * math.cos(bearing)
        here = height(waves, x, -z)
        slope_x = (height(waves, x + 0.05, -z) - here) / 0.05
        slope_z = (height(waves, x, -(z + 0.05)) - here) / 0.05
        length = math.sqrt(slope_x * slope_x + slope_z * slope_z + 1)
        n = (-slope_x / length, 1 / length, -slope_z / length)
        v = (-x, cam_y - here, cam_z - z)
        view = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        v = (v[0] / view, v[1] / view, v[2] / view)
        vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2]
        reflected = (2 * vn * n[0] - v[0], 2 * vn * n[1] - v[1], 2 * vn * n[2] - v[2])
        if vn > 0 and reflected[0] * sun[0] + reflected[1] * sun[1] + reflected[2] * sun[2] > cosine:
            lit_rows.add(frame_row(-math.degrees(math.atan2(cam_y - here, distance))))
path_top = min(lit_rows, default=1)
path_bottom = max(lit_rows, default=0)
check(path_top < horizon_row + 0.04, f"the glint starts at the horizon, row {path_top} against {horizon_row}")
check(path_bottom - horizon_row > 0.12, f"the glint runs down toward the camera, rows {path_top} to {path_bottom}")

if failures:
    for failure in failures:
        print("FAIL:", failure)
    sys.exit(1)
print(f"open sea starter: on {on:.3f} off {off:.5f} rows {path_rows} peak {peak_azimuth} sun cloud {cloud_opacity(sun):.3f}")
print(f"frame: horizon {horizon_row:.3f} sun row {sun_row:.3f} column {sun_column:.3f} tower top {tower_row:.3f} "
      f"crest {crest:.3f} slope {slope_rms:.3f} needed tilt {needed_tilt:.1f} glint rows {path_top:.3f}-{path_bottom:.3f}")
