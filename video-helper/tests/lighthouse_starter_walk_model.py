#!/usr/bin/env python3
"""CPU model of the Lighthouse and Ocean camera transport and caustic at 4K.

The scene comes from the built-in starter (plugin/Source/graph/
BuiltInVisualModules.h, lighthouseAndOceanStarter): the water grid and wave at
time zero, the tower, lantern and rock cubes, the sphere lens with its
Transform 3D node, the camera and the sun direction are read from its authored
values. Pixel (x, y) maps to a ray as in glass_caustics::cameraRay: y runs down
from the top row.

The lens is the closed sphere GeometryCore.h generates (a top pole, latitude
rings of `segments` vertices, a bottom pole, every triangle wound outward),
squashed along its pole by the scale, rotated by the quaternion of rotationX, Y
and Z (R = Rz Ry Rx), then moved. Its walk hits are triangle hits with the
outward geometric face normal.

glass_pair_walk mirrors glassPairWalk in fixture_smooth_glass.h: the 1e-5
branch and 1e-4 discarded power thresholds, GLASS_INTERFACE_DEPTH, the medium
and orientation refusals, heaviest-first visits, the lightest-branch spill
when the slots are full, the truncation tolerance, and one shared 48-visit
walk for nondispersive glass. caustic_walk mirrors traceBranches in
fixture_glass_caustics.h with the starter's branch policy. The query mirrors
sceneClosestHit: the nearest hit wins, and an exact distance tie goes to the
lower scene object index. The lens hangs in front of the lantern, so it shares no
plane with it; the old lantern-top tie only matters for the earlier cube
geometry kept in TOUCHING.

Run without arguments for the contract: every glass pixel and a stride of
water pixels at 3840x2160, and a grid of caustic entries on each sunlit glass
face, must complete, and the caustic must focus on the tower front face below
the lantern. --compare also runs the earlier per-channel 16-visit walk
on the earlier geometry, and the glass with dispersion 0.004.
"""

import math
import random
import re
import sys
from pathlib import Path

WIDTH, HEIGHT = 3840, 2160
VISITS_PER_CHANNEL = 16
SLOTS = 16
TOLERANCE = 0.02
INTERFACE_DEPTH = 32
WATER_IOR = 1.333
GLASS_IOR = 1.5
WAVELENGTHS = (0.6563, 0.5461, 0.4358)
# The starter's caustic branch policy (glassCausticMaximum*, Minimum*).
CAUSTIC_VISITS, CAUSTIC_PENDING, CAUSTIC_INTERFACES = 64, 16, 32
CAUSTIC_UNRESOLVED, CAUSTIC_MINIMUM_BRANCH, CAUSTIC_RAY_MINIMUM = 1.0e-4, 1.0e-6, 1.0e-5

WATER, TOWER, LANTERN, GLASS, ROCK = range(5)
NAMES = ("water", "tower", "lantern", "glass", "rock")


def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def normalize(a):
    length = math.sqrt(dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length)


def reflect(d, n): return sub(d, mul(n, 2.0 * dot(d, n)))


def starter_values():
    """Authored node values of lighthouseAndOceanStarter, keyed by node label."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "plugin/Source/graph/BuiltInVisualModules.h").read_text(encoding="utf-8")
    begin = text.index("inline GraphModulePackage lighthouseAndOceanStarter()")
    body = text[begin:text.index("\n}\n", begin)]
    values = {}
    for label, pairs in re.findall(r'TRANS\("([^"]+)"\),\s*\{(\{"[^;]*?\})\}\)', body):
        values[label] = {key: float(value) for key, value in re.findall(r'\{"(\w+)", (-?[0-9.]+)\}', pairs)}
    lens = re.search(r'\{\{"value", ([0-9.]+)\}\}, "Lens IOR"', body)
    water = re.search(r'\{0\.0, 0\.0, 0\.0\},\s*[0-9.]+, [0-9.]+, ([0-9.]+), ([0-9.]+), 1001,', body)
    if lens is None or water is None:
        raise SystemExit("FAIL: the starter IORs were not found")
    values["iors"] = {"glass": float(lens.group(1)), "water": float(water.group(1)), "waterDispersion": float(water.group(2))}
    return values


STARTER = starter_values()


def rotation(degrees_x, degrees_y):
    """Quaternion of a Visual3DTransform with rotationX then rotationY, as in the C++ test's sun."""
    ax, ay = math.radians(degrees_x), math.radians(degrees_y)
    sx, cx, sy, cy = math.sin(ax / 2), math.cos(ax / 2), math.sin(ay / 2), math.cos(ay / 2)
    return (cy * sx, sy * cx, -sy * sx, cy * cx)


def rotate(q, v):
    """v + 2 cross(q.xyz, cross(q.xyz, v) + q.w v), as rotateSceneVector and glassRotateQ."""
    axis = q[:3]
    inner = add(cross(axis, v), mul(v, q[3]))
    return add(v, mul(cross(axis, inner), 2.0))


# A barycentric slack that keeps a ray through a shared mesh edge from slipping between its
# two triangles. The box slabs have no such seam.
EDGE_SLACK = 1.0e-9


class Box:
    """An axis-aligned cube of the starter: the tower, lantern and rock."""

    entry_samples = 48

    def __init__(self, index, centre, half):
        self.index, self.centre, self.half = index, centre, half

    def hit(self, origin, direction, minimum=0.0):
        near, far, near_axis, far_axis = -math.inf, math.inf, -1, -1
        for axis in range(3):
            low, high = self.centre[axis] - self.half[axis], self.centre[axis] + self.half[axis]
            if direction[axis] == 0.0:
                if origin[axis] < low or origin[axis] > high:
                    return None
                continue
            a = (low - origin[axis]) / direction[axis]
            b = (high - origin[axis]) / direction[axis]
            if a > b:
                a, b = b, a
            if a > near:
                near, near_axis = a, axis
            if b < far:
                far, far_axis = b, axis
        if near > far or far < minimum:
            return None
        if near >= minimum:
            normal = [0.0, 0.0, 0.0]
            normal[near_axis] = -math.copysign(1.0, direction[near_axis])
            return near, tuple(normal), self.index
        normal = [0.0, 0.0, 0.0]
        normal[far_axis] = math.copysign(1.0, direction[far_axis])
        return far, tuple(normal), self.index

    def vertices(self):
        return [tuple(self.centre[a] + self.half[a] * (1 if corner >> a & 1 else -1) for a in range(3))
                for corner in range(8)]

    def bounds(self):
        return (tuple(self.centre[a] - self.half[a] for a in range(3)),
                tuple(self.centre[a] + self.half[a] for a in range(3)))

    def entry_points(self, toward, samples):
        """A samples-by-samples grid of points on every face the sun lights."""
        for axis in range(3):
            for sign in (-1.0, 1.0):
                if sign * toward[axis] <= 0.0:
                    continue
                u, v = [a for a in range(3) if a != axis]
                for i in range(samples):
                    for j in range(samples):
                        point = list(self.centre)
                        point[axis] += sign * self.half[axis]
                        point[u] += self.half[u] * (2 * (i + 0.5) / samples - 1)
                        point[v] += self.half[v] * (2 * (j + 0.5) / samples - 1)
                        yield tuple(point)


def euler_quaternion(degrees_x, degrees_y, degrees_z):
    """Quaternion of a Visual3DTransform, as lowerTransform: R = Rz Ry Rx. Returns (x, y, z, w)."""
    sx, cx = math.sin(math.radians(degrees_x) / 2), math.cos(math.radians(degrees_x) / 2)
    sy, cy = math.sin(math.radians(degrees_y) / 2), math.cos(math.radians(degrees_y) / 2)
    sz, cz = math.sin(math.radians(degrees_z) / 2), math.cos(math.radians(degrees_z) / 2)
    return (sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz)


def sphere_mesh(radius, segments, rings):
    """Positions and triangles of GeometryCore.h's sphere generator, in local space.

    The top pole (0, radius, 0), then rings - 1 latitude rings of `segments`
    vertices from the top down, then the bottom pole. Each column has a top fan
    triangle, two triangles per band and a bottom fan triangle, all wound outward.
    """
    positions = [(0.0, radius, 0.0)]
    for row in range(1, rings):
        latitude = math.pi * row / rings
        for column in range(segments):
            longitude = 2 * math.pi * column / segments
            positions.append((radius * math.sin(latitude) * math.cos(longitude), radius * math.cos(latitude),
                              radius * math.sin(latitude) * math.sin(longitude)))
    bottom = len(positions)
    positions.append((0.0, -radius, 0.0))
    triangles = []
    for column in range(segments):
        following = (column + 1) % segments
        triangles.append((0, 1 + following, 1 + column))
        for row in range(rings - 2):
            a, b = 1 + row * segments + column, 1 + row * segments + following
            triangles.append((a, b, a + segments))
            triangles.append((b, b + segments, a + segments))
        triangles.append((bottom, 1 + (rings - 2) * segments + column, 1 + (rings - 2) * segments + following))
    return positions, triangles


class Lens:
    """The starter's sphere lens as a world-space triangle mesh."""

    entry_samples = 12

    def __init__(self, index, shape, placement):
        self.index = index
        self.segments, self.rings = int(shape["segments"]), int(shape["rings"])
        self.local_radius = shape["radius"]
        scale = tuple(placement.get("scale" + axis, 1.0) for axis in "XYZ")
        self.centre = tuple(placement.get("translation" + axis, 0.0) for axis in "XYZ")
        self.rotation = euler_quaternion(*(placement.get("rotation" + axis, 0.0) for axis in "XYZ"))
        self.scale = scale
        positions, triangles = sphere_mesh(self.local_radius, self.segments, self.rings)
        self.points = [add(rotate(self.rotation, tuple(p[a] * scale[a] for a in range(3))), self.centre)
                       for p in positions]
        self.indices = triangles
        self.triangles = []
        for i, j, k in triangles:
            a, b, c = self.points[i], self.points[j], self.points[k]
            e1, e2 = sub(b, a), sub(c, a)
            self.triangles.append((a, e1, e2, normalize(cross(e1, e2))))
        low, high = self.bounds()
        self.low, self.high = low, high

    def vertices(self):
        return list(self.points)

    def bounds(self):
        return (tuple(min(p[a] for p in self.points) for a in range(3)),
                tuple(max(p[a] for p in self.points) for a in range(3)))

    def reaches(self, origin, direction, minimum):
        near, far = minimum, math.inf
        for axis in range(3):
            if direction[axis] == 0.0:
                if origin[axis] < self.low[axis] or origin[axis] > self.high[axis]:
                    return False
                continue
            a = (self.low[axis] - origin[axis]) / direction[axis]
            b = (self.high[axis] - origin[axis]) / direction[axis]
            if a > b:
                a, b = b, a
            near, far = max(near, a), min(far, b)
            if near > far:
                return False
        return True

    def hit(self, origin, direction, minimum=0.0):
        """Nearest triangle crossing at or past `minimum`, with the outward face normal."""
        if not self.reaches(origin, direction, minimum):
            return None
        best = None
        for a, e1, e2, normal in self.triangles:
            p = cross(direction, e2)
            det = dot(e1, p)
            if abs(det) < 1.0e-14:
                continue
            offset = sub(origin, a)
            u = dot(offset, p) / det
            if u < -EDGE_SLACK or u > 1.0 + EDGE_SLACK:
                continue
            q = cross(offset, e1)
            v = dot(direction, q) / det
            if v < -EDGE_SLACK or u + v > 1.0 + EDGE_SLACK:
                continue
            distance = dot(e2, q) / det
            if distance < minimum:
                continue
            if best is None or distance < best[0]:
                best = (distance, normal, self.index)
        return best

    def entry_points(self, toward, samples, weighted=False):
        """Centroids of a `samples`-per-edge subdivision of every triangle the sun lights.

        weighted also yields each centroid's share of the sun power the lens takes in:
        its sub-triangle area times the cosine to the sun.
        """
        for a, e1, e2, normal in self.triangles:
            cosine = dot(normal, toward)
            if cosine <= 0.0:
                continue
            power = 0.5 * math.sqrt(dot(cross(e1, e2), cross(e1, e2))) / (samples * samples) * cosine
            for i in range(samples):
                for j in range(samples - i):
                    points = [add(a, add(mul(e1, (i + 1.0 / 3.0) / samples), mul(e2, (j + 1.0 / 3.0) / samples)))]
                    if i + j < samples - 1:
                        points.append(add(a, add(mul(e1, (i + 2.0 / 3.0) / samples), mul(e2, (j + 2.0 / 3.0) / samples))))
                    for point in points:
                        yield (point, power) if weighted else point

    def volume(self):
        """Signed volume of the mesh. Positive means every triangle is wound outward."""
        return sum(dot(self.points[i], cross(self.points[j], self.points[k])) for i, j, k in self.indices) / 6.0

    def closed(self):
        """True when welded positions give every directed edge exactly one reverse partner."""
        weld = {}
        ids = [weld.setdefault(tuple(round(c, 9) + 0.0 for c in p), len(weld)) for p in self.points]
        edges = {}
        for i, j, k in self.indices:
            for a, b in ((ids[i], ids[j]), (ids[j], ids[k]), (ids[k], ids[i])):
                edges[a, b] = edges.get((a, b), 0) + 1
        return all(count == 1 and edges.get((b, a)) == 1 for (a, b), count in edges.items())


def starter_box(index, mesh, position):
    size, at = STARTER[mesh], STARTER[position]
    centre = tuple(at.get("translation" + axis, 0.0) for axis in "XYZ")
    return Box(index, centre, (size["width"] / 2, size["height"] / 2, size["depth"] / 2))


def starter_lens(index, mesh, position):
    return Lens(index, STARTER[mesh], STARTER[position])


BOXES = (
    starter_box(TOWER, "Lighthouse Tower", "Lighthouse Position"),
    starter_box(LANTERN, "Lighthouse Lantern", "Lantern Position"),
    starter_lens(GLASS, "Lantern Glass Lens", "Lantern Glass Position"),
    starter_box(ROCK, "Nearby Rock", "Nearby Rock Position"),
)
WATER_IOR = STARTER["iors"]["water"]
GLASS_IOR = STARTER["iors"]["glass"]
# The shared walk relies on a water IOR with no dispersion.
assert STARTER["iors"]["waterDispersion"] == 0.0


def water_grid(time=0.0):
    """Vertices and evaluated normals of the wave grid, as in lighthouse_caustic_volume_tests.cpp."""
    grid, wave = STARTER["Ocean Grid"], STARTER["Ocean Waves"]
    columns, rows, spacing = int(grid["columns"]), int(grid["rows"]), grid["spacing"]
    angle = math.radians(wave["direction"])
    direction = (math.cos(angle), math.sin(angle))
    amplitude, wavelength, speed = wave["amplitude"], wave["wavelength"], wave["speed"]
    points, normals = {}, {}
    for i in range(columns):
        for j in range(rows):
            x, z = (i - (columns - 1) / 2) * spacing, (j - (rows - 1) / 2) * spacing
            phase = 2 * math.pi * (x * direction[0] + z * direction[1] - speed * time) / wavelength
            slope = amplitude * 2 * math.pi / wavelength * math.cos(phase)
            points[i, j] = (x, amplitude * math.sin(phase), z)
            normals[i, j] = normalize((-slope * direction[0], 1.0, -slope * direction[1]))
    return points, normals


class Scene:
    def __init__(self, time=0.0, glass_wins_ties=False, boxes=None):
        self.points, self.normals = water_grid(time)
        self.boxes = boxes or BOXES
        # Exact distance ties go to the lower object index (the lantern, 2,
        # before the glass, 3). glass_wins_ties models the other float outcome.
        # Boxes and the lens both answer hit(origin, direction, minimum).
        self.glass_wins_ties = glass_wins_ties
        self.shared_plane_hits = 0

    def water_hit(self, origin, direction, best, minimum=0.0):
        # Only the slab |y| <= amplitude holds water triangles.
        slab = STARTER["Ocean Waves"]["amplitude"] * 1.001
        if direction[1] == 0.0:
            if abs(origin[1]) > slab:
                return None
            t0, t1 = 0.0, best
        else:
            a, b = (-slab - origin[1]) / direction[1], (slab - origin[1]) / direction[1]
            t0, t1 = max(0.0, min(a, b)), min(best, max(a, b))
        if t0 > t1:
            return None
        p0, p1 = add(origin, mul(direction, t0)), add(origin, mul(direction, t1))
        grid = STARTER["Ocean Grid"]
        cell = lambda v: int(math.floor(v / grid["spacing"] + (grid["columns"] - 1) / 2))
        i0, i1 = sorted((cell(p0[0]), cell(p1[0])))
        j0, j1 = sorted((cell(p0[2]), cell(p1[2])))
        found = None
        last = int(grid["columns"]) - 2
        for i in range(max(0, i0 - 1), min(last, i1 + 1) + 1):
            for j in range(max(0, j0 - 1), min(last, j1 + 1) + 1):
                v00, v01, v10, v11 = (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)
                for tri in ((v00, v01, v11), (v00, v11, v10)):
                    hit = self.triangle(origin, direction, tri, minimum)
                    if hit is not None and hit[0] <= (found[0] if found else best):
                        found = hit
        return found

    def triangle(self, origin, direction, tri, minimum=0.0):
        a, b, c = (self.points[v] for v in tri)
        e1, e2 = sub(b, a), sub(c, a)
        p = cross(direction, e2)
        det = dot(e1, p)
        if det == 0.0:
            return None
        t = sub(origin, a)
        u = dot(t, p) / det
        if u < 0.0 or u > 1.0:
            return None
        q = cross(t, e1)
        v = dot(direction, q) / det
        if v < 0.0 or u + v > 1.0:
            return None
        distance = dot(e2, q) / det
        if distance < minimum:
            return None
        na, nb, nc = (self.normals[w] for w in tri)
        evaluated = normalize(add(add(mul(na, 1 - u - v), mul(nb, u)), mul(nc, v)))
        return distance, normalize(cross(e1, e2)), WATER, evaluated

    def closest(self, origin, direction, minimum=0.0):
        """(distance, outward normal, object index, evaluated normal) or None for a miss."""
        best = None
        for box in self.boxes:
            hit = box.hit(origin, direction, minimum)
            if hit is None:
                continue
            if best is None or hit[0] < best[0] - 1e-9:
                best = hit
            elif abs(hit[0] - best[0]) <= 1e-9:
                # A shared plane: the lantern top and the glass bottom at y=3
                # (the earlier lantern-top geometry; the starter no longer has one).
                if {hit[2], best[2]} == {LANTERN, GLASS}:
                    self.shared_plane_hits += 1
                    winner = GLASS if self.glass_wins_ties else LANTERN
                    best = hit if hit[2] == winner else best
                elif hit[2] < best[2]:
                    best = hit
        water = self.water_hit(origin, direction, best[0] if best else math.inf, minimum)
        if water is not None and (best is None or water[0] < best[0]):
            return water
        if best is None:
            return None
        return best[0], best[1], best[2], best[1]


def fresnel(direction, normal, from_ior, to_ior):
    cosine = min(max(-dot(direction, normal), 0.0), 1.0)
    ratio = from_ior / to_ior
    sine_squared = ratio * ratio * max(0.0, 1.0 - cosine * cosine)
    if sine_squared > 1.0:
        return 1.0, None
    exit_cosine = math.sqrt(max(0.0, 1.0 - sine_squared))
    transmitted = normalize(add(mul(direction, ratio), mul(normal, ratio * cosine - exit_cosine)))
    if from_ior == to_ior:
        return 0.0, transmitted
    s = (from_ior * cosine - to_ior * exit_cosine) / (from_ior * cosine + to_ior * exit_cosine)
    p = (to_ior * cosine - from_ior * exit_cosine) / (to_ior * cosine + from_ior * exit_cosine)
    return min(max(0.5 * (s * s + p * p), 0.0), 1.0), transmitted


def bias(position):
    return max(1.0e-5, 2.0e-6 * max(abs(position[0]), abs(position[1]), abs(position[2])))


def glass_pair_walk(scene, seeds, iors, visits, depth_refuses=False, slots=SLOTS, tolerance=TOLERANCE):
    """Mirror of one glassPairWalk.

    seeds: (origin, direction, weight, inside). iors: one volume IOR per walk;
    a single IOR is the shared walk that carries all three channels.
    visits: closest-hit queries per walk. depth_refuses keeps the earlier
    DEPTH refusal. Returns (status, max truncated, queries).
    """
    queries = 0
    worst = 0.0
    for ior in iors:
        pending = [list(seed) + [0] for seed in seeds if seed[2] > 0.0]
        discarded = truncated = 0.0
        spilled = False

        def leftover(branch):
            return branch[2] if branch[3] else 0.0

        for _ in range(visits):
            if not pending:
                break
            heaviest = 0
            for slot in range(1, len(pending)):
                if pending[slot][2] > pending[heaviest][2]:
                    heaviest = slot
            origin, direction, weight, inside, depth = pending[heaviest]
            pending[heaviest] = pending[-1]
            pending.pop()
            queries += 1
            hit = scene.closest(origin, direction)
            if hit is None:
                if inside:
                    return "MEDIUM", worst, queries
                continue
            distance, outward, index, _ = hit
            thin = index == WATER
            if not thin and index != GLASS:
                if inside:
                    return "MEDIUM", worst, queries
                continue
            entering = dot(direction, outward) < 0.0
            if thin and inside:
                return "MEDIUM", worst, queries
            if not thin and inside == entering:
                return "ORIENTATION", worst, queries
            if depth >= INTERFACE_DEPTH:
                if depth_refuses:
                    return "DEPTH", worst, queries
                truncated += leftover((origin, direction, weight, inside))
                continue
            normal = outward if entering else mul(outward, -1.0)
            if thin:
                from_ior, to_ior = (1.0, WATER_IOR) if entering else (WATER_IOR, 1.0)
            else:
                from_ior, to_ior = (ior, 1.0) if inside else (1.0, ior)
            reflection, transmitted = fresnel(direction, normal, from_ior, to_ior)
            tir = transmitted is None
            position = add(origin, mul(direction, distance))
            offset = bias(position)
            if not tir and weight * (1.0 - reflection) < 0.00001:
                discarded += weight * (1.0 - reflection)
            if weight * reflection < 0.00001:
                discarded += weight * reflection
            if discarded > 0.0001:
                return "DISCARDED", worst, queries
            for transmission in (True, False):
                child = weight * ((1.0 - reflection) if transmission else reflection)
                if (transmission and tir) or child < 0.00001:
                    continue
                child_direction = transmitted if transmission else normalize(reflect(direction, normal))
                child_inside = (not inside) if (transmission and not thin) else inside
                child_origin = add(position, mul(normal, -offset if transmission else offset))
                branch = [child_origin, child_direction, child, child_inside, depth + 1]
                if len(pending) >= slots:
                    spilled = True
                    lightest = 0
                    for slot in range(1, slots):
                        if pending[slot][2] < pending[lightest][2]:
                            lightest = slot
                    if pending[lightest][2] >= child:
                        truncated += leftover(branch)
                        continue
                    truncated += leftover(pending[lightest])
                    pending[lightest] = pending[-1]
                    pending.pop()
                pending.append(branch)
        for branch in pending:
            truncated += leftover(branch)
        worst = max(worst, truncated)
        if truncated > tolerance:
            return ("PENDING" if spilled else "VISITS"), worst, queries
    return "COMPLETE", worst, queries


CAMERA = tuple(STARTER["Ocean Camera Position"].get("translation" + axis, 0.0) for axis in "XYZ")
CAMERA_ROTATION = rotation(STARTER["Ocean Camera Position"].get("rotationX", 0.0),
                           STARTER["Ocean Camera Position"].get("rotationY", 0.0))
TANGENT = math.tan(math.radians(STARTER["Ocean Camera"]["verticalFovDegrees"]) * 0.5)
TOWARD_SUN = rotate(rotation(STARTER["Ocean Key Direction"]["rotationX"], STARTER["Ocean Key Direction"]["rotationY"]),
                    (0.0, 0.0, 1.0))


def camera_ray(x, y):
    local = ((2 * x / WIDTH - 1) * TANGENT * WIDTH / HEIGHT, (1 - 2 * y / HEIGHT) * TANGENT, -1.0)
    return normalize(rotate(CAMERA_ROTATION, local))


def volume_iors(dispersion):
    return [GLASS_IOR + dispersion * (1.0 / (w * w) - 1.0 / (0.58756 * 0.58756)) for w in WAVELENGTHS]


def policy_walk(policy, dispersion):
    """(iors, visits per walk, depth refuses). 'shipped' walks once with 48
    visits for nondispersive glass and each channel with 16 otherwise; 'channel'
    is the earlier walk: each channel with 16 visits, and DEPTH refuses."""
    iors = volume_iors(dispersion)
    if policy == "shipped":
        return (iors[:1], 3 * VISITS_PER_CHANNEL, False) if dispersion == 0.0 else (iors, VISITS_PER_CHANNEL, False)
    return iors, VISITS_PER_CHANNEL, True


def run_walk(scene, seeds, policy, dispersion):
    iors, visits, depth_refuses = policy_walk(policy, dispersion)
    return glass_pair_walk(scene, seeds, iors, visits, depth_refuses)


def glass_pixel(scene, x, y, policy, dispersion):
    direction = camera_ray(x + 0.5, y + 0.5)
    first = scene.closest(CAMERA, direction)
    if first is None or first[2] != GLASS:
        return None
    status, truncated, queries = run_walk(scene, [(CAMERA, direction, 1.0, dot(direction, first[1]) >= 0.0)],
                                          policy, dispersion)
    return status, truncated, queries + 1


def water_pixel(scene, x, y, policy, dispersion):
    direction = camera_ray(x + 0.5, y + 0.5)
    first = scene.closest(CAMERA, direction)
    if first is None or first[2] != WATER:
        return None
    position = add(CAMERA, mul(direction, first[0]))
    normal = first[3]
    entering = dot(direction, normal) < 0.0
    if not entering:
        normal = mul(normal, -1.0)
    reflection, transmitted = fresnel(direction, normal, 1.0 if entering else WATER_IOR, WATER_IOR if entering else 1.0)
    offset = bias(position)
    seeds = [(add(position, mul(normal, offset)), normalize(reflect(direction, normal)), reflection, False)]
    if transmitted is not None:
        seeds.append((sub(position, mul(normal, offset)), transmitted, 1.0 - reflection, False))
    status, truncated, queries = run_walk(scene, seeds, policy, dispersion)
    return status, truncated, queries + 1


def glass_region(boxes):
    """Pixel rectangle that covers the projected glass vertices, plus a margin."""
    xs, ys = [], []
    inverse = (-CAMERA_ROTATION[0], -CAMERA_ROTATION[1], -CAMERA_ROTATION[2], CAMERA_ROTATION[3])
    for p in glass_solid(boxes).vertices():
        local = rotate(inverse, sub(p, CAMERA))
        xs.append((local[0] / -local[2] / (TANGENT * WIDTH / HEIGHT) + 1) * WIDTH / 2)
        ys.append((1 - local[1] / -local[2] / TANGENT) * HEIGHT / 2)
    return (max(0, int(min(xs)) - 4), min(WIDTH, int(max(xs)) + 5),
            max(0, int(min(ys)) - 4), min(HEIGHT, int(max(ys)) + 5))


def caustic_walk(scene, entry, ior, deposits=None):
    """Mirror of traceBranches for a smooth glass entry: (status, unresolved fraction).

    Last in, first out; a branch below the minimum fraction, a full pending
    list, an interface at the bound and the visit bound all add unresolved
    power, which may not pass CAUSTIC_UNRESOLVED. An inside branch that leaves
    the glass for anything else refuses, as does a receiver met from behind.
    Given a list, deposits collects (object index, hit position, fraction) for
    every receiver hit.
    """
    toward = TOWARD_SUN
    offset = max(CAUSTIC_RAY_MINIMUM, 2.0e-6 * max(abs(v) for v in entry))
    pending = [(add(entry, mul(toward, 2 * offset)), mul(toward, -1.0), 1.0, 0, False)]
    unresolved = 0.0
    visits = 0
    while pending and visits < CAUSTIC_VISITS:
        origin, direction, fraction, interfaces, inside = pending.pop()
        visits += 1
        hit = scene.closest(origin, direction, CAUSTIC_RAY_MINIMUM if interfaces else 0.0)
        if hit is None:
            if inside:
                return "UNCLOSED", unresolved + fraction
            continue
        distance, outward, index, _ = hit
        entering = dot(direction, outward) < 0.0
        if index != GLASS:
            if index == WATER:
                if inside:
                    return "INTERIOR", unresolved + fraction
                continue
            if inside or not entering:
                return "INTERIOR", unresolved + fraction
            if deposits is not None:
                deposits.append((index, add(origin, mul(direction, distance)), fraction))
            continue
        if inside == entering:
            return "ORDER", unresolved + fraction
        if interfaces >= CAUSTIC_INTERFACES:
            unresolved += fraction
            continue
        normal = mul(outward, -1.0) if inside else outward
        reflection, transmitted = fresnel(direction, normal, ior if inside else 1.0, 1.0 if inside else ior)
        position = add(origin, mul(direction, distance))
        step = max(CAUSTIC_RAY_MINIMUM, 2.0e-6 * max(abs(v) for v in position))
        reflected = fraction * reflection
        children = [(add(position, mul(normal, step)), normalize(reflect(direction, normal)), reflected, interfaces + 1, inside)]
        if transmitted is not None:
            children.append((add(position, mul(normal, -step)), transmitted, fraction - reflected, interfaces + 1, not inside))
        for child in children:
            if child[2] == 0.0:
                continue
            if child[2] < CAUSTIC_MINIMUM_BRANCH or len(pending) >= CAUSTIC_PENDING:
                unresolved += child[2]
                continue
            pending.append(child)
    unresolved += sum(path[2] for path in pending)
    return ("COMPLETE" if unresolved <= CAUSTIC_UNRESOLVED else "UNRESOLVED"), unresolved


def glass_solid(boxes):
    return [solid for solid in boxes if solid.index == GLASS][0]


# The lens focuses the sun on the tower front face 0.40 below the lantern. The spot
# radius is the one the focus contract measures power within.
SPOT_CENTRE, SPOT_RADIUS = (-1.2, 2.2), 0.05


def caustic_survey(boxes, dispersion=0.0, coarse=False):
    """Entries on every glass surface the sun lights: a grid per box face or a subdivided lit triangle.

    For the sphere lens the report also sums the sun power the lit triangles take
    in, the power deposited on receivers and the power inside the tower-front spot.
    """
    scene = Scene(0.0, False, boxes)
    glass = glass_solid(boxes)
    report = {"entries": 0, "refused": {}, "max_unresolved": 0.0, "power": 0.0, "deposited": 0.0, "spot": 0.0}
    iors = volume_iors(dispersion)
    samples = glass.entry_samples // (2 if coarse else 1)
    tower = [box for box in boxes if box.index == TOWER][0]
    front = tower.bounds()[1][2]
    if isinstance(glass, Lens):
        entries = glass.entry_points(TOWARD_SUN, samples, True)
    else:
        entries = ((point, 0.0) for point in glass.entry_points(TOWARD_SUN, samples))
    for point, power in entries:
        for ior in iors[:1] if dispersion == 0.0 else iors:
            deposits = []
            status, unresolved = caustic_walk(scene, point, ior, deposits)
            report["entries"] += 1
            report["max_unresolved"] = max(report["max_unresolved"], unresolved)
            if status != "COMPLETE":
                report["refused"][status] = report["refused"].get(status, 0) + 1
            report["power"] += power
            for index, position, fraction in deposits:
                report["deposited"] += power * fraction
                if index == TOWER and abs(position[2] - front) < 1.0e-6 and \
                        math.hypot(position[0] - SPOT_CENTRE[0], position[1] - SPOT_CENTRE[1]) <= SPOT_RADIUS:
                    report["spot"] += power * fraction
    return report


def survey(policy, dispersion=0.0, water_stride=24, glass_wins_ties=False, times=(0.0,), boxes=None):
    boxes = boxes or BOXES
    report = {"glass": 0, "water": 0, "refused": {}, "first": None, "max_truncated": 0.0,
              "max_queries": 0, "shared_plane_hits": 0}
    x0, x1, y0, y1 = glass_region(boxes)
    for time in times:
        scene = Scene(time, glass_wins_ties, boxes)
        jobs = [(glass_pixel, x, y) for y in range(y0, y1) for x in range(x0, x1)]
        jobs += [(water_pixel, x, y) for y in range(0, HEIGHT, water_stride) for x in range(0, WIDTH, water_stride)]
        for function, x, y in jobs:
            result = function(scene, x, y, policy, dispersion)
            if result is None:
                continue
            status, truncated, queries = result
            report["glass" if function is glass_pixel else "water"] += 1
            report["max_truncated"] = max(report["max_truncated"], truncated)
            report["max_queries"] = max(report["max_queries"], queries)
            if status != "COMPLETE":
                report["refused"][status] = report["refused"].get(status, 0) + 1
                if report["first"] is None or (y, x) < (report["first"][1], report["first"][0]):
                    report["first"] = (x, y, status)
        report["shared_plane_hits"] += scene.shared_plane_hits
    return report


def describe(label, report):
    refused = sum(report["refused"].values())
    print("%s: %d glass and %d sampled water pixels, %d refused %s, first %s, max truncated %.5f, max queries %d, "
          "shared-plane hits %d" % (label, report["glass"], report["water"], refused, report["refused"], report["first"],
                                    report["max_truncated"], report["max_queries"], report["shared_plane_hits"]))
    return refused


def describe_caustic(label, report):
    refused = sum(report["refused"].values())
    print("%s: %d caustic entries, %d refused %s, max unresolved %.3g"
          % (label, report["entries"], refused, report["refused"], report["max_unresolved"]))
    return refused


# The earlier geometry, kept so the model still reproduces the live VISITS
# refusal: the earlier 0.5 glass cube with its bottom exactly on the lantern top, at the
# lantern's x and z. The starter's lens is a sphere now, so the cube is fixed here.
TOUCHING = tuple(box if box.index != GLASS else Box(GLASS, (-1.2, 3.25, -1.2), (0.25, 0.25, 0.25)) for box in BOXES)


def overlap(a, b):
    """True when two bounding boxes share volume (touching faces do not count)."""
    (alow, ahigh), (blow, bhigh) = a.bounds(), b.bounds()
    return all(alow[axis] < bhigh[axis] and blow[axis] < ahigh[axis] for axis in range(3))


def lens_errors(boxes):
    """The authored lens is a closed, outward-wound sphere of 10 segments and 6 rings, scaled and turned."""
    lens = glass_solid(boxes)
    if not isinstance(lens, Lens):
        return []
    errors = []
    if (lens.segments, lens.rings) != (10, 6):
        errors.append("the lens must be a sphere of 10 segments and 6 rings (found %d and %d)"
                      % (lens.segments, lens.rings))
    positions, triangles = sphere_mesh(lens.local_radius, lens.segments, lens.rings)
    # GraphInstrumentTests pins the scene totals: 2 + segments * (rings - 1) vertices and
    # 6 * segments * (rings - 1) indices.
    if len(positions) != 2 + lens.segments * (lens.rings - 1) or \
            3 * len(triangles) != 6 * lens.segments * (lens.rings - 1):
        errors.append("the sphere mesh no longer has the generator's vertex and index counts")
    if not lens.closed():
        errors.append("the lens mesh is not a closed volume")
    # The mesh is convex around its centre, so every outward face normal points away from it.
    for a, e1, e2, normal in lens.triangles:
        centroid = add(a, mul(add(e1, e2), 1.0 / 3.0))
        if dot(normal, sub(centroid, lens.centre)) <= 0.0:
            errors.append("a lens triangle is wound inward")
            break
    ellipsoid = 4.0 / 3.0 * math.pi * lens.local_radius ** 3 * lens.scale[0] * lens.scale[1] * lens.scale[2]
    if not 0.0 < lens.volume() < ellipsoid:
        errors.append("the lens volume %.9f is not inside the scaled sphere's %.9f" % (lens.volume(), ellipsoid))
    # The mesh is convex, so an independent half-space clip must agree with the triangle hits.
    planes = [(triangle[3], dot(triangle[3], triangle[0])) for triangle in lens.triangles]

    def clip(origin, direction):
        near, far = -math.inf, math.inf
        for normal, offset in planes:
            slope, room = dot(normal, direction), offset - dot(normal, origin)
            if abs(slope) < 1.0e-14:
                if room < 0.0:
                    return None
                continue
            if slope < 0.0:
                near = max(near, room / slope)
            else:
                far = min(far, room / slope)
        if near > far:
            return None
        return near if near >= 0.0 else (far if far >= 0.0 else None)

    generator, hits = random.Random(7), 0
    for _ in range(4000):
        # Rays start anywhere in a 1-unit box but aim at a point near the lens, so
        # most of them meet it, while the rest still test the misses.
        origin = add(lens.centre, tuple(generator.uniform(-0.5, 0.5) for _ in range(3)))
        target = add(lens.centre, tuple(generator.uniform(-0.2, 0.2) for _ in range(3)))
        direction = normalize(sub(target, origin))
        found, expected_distance = lens.hit(origin, direction), clip(origin, direction)
        hits += found is not None
        if (found is None) != (expected_distance is None) or \
                (found is not None and abs(found[0] - expected_distance) > 1.0e-9):
            errors.append("a triangle hit disagrees with the convex half-space clip of the lens")
            break
    if hits < 200:
        errors.append("the lens hit cross-check found only %d hits, too few to mean anything" % hits)
    return errors


def placement_errors(boxes):
    """The lens hangs clear above the rock top and stands clear in front of the tower."""
    glass = glass_solid(boxes)
    rock = [box for box in boxes if box.index == ROCK][0]
    tower = [box for box in boxes if box.index == TOWER][0]
    errors = []
    (low_x, low_y, low_z), _ = glass.bounds()
    rock_top = rock.bounds()[1][1]
    if low_y <= rock_top:
        errors.append("the lens must clear the rock top (gap %.6f)" % (low_y - rock_top))
    # The camera looks from +z toward the tower, so in front means a larger z than the tower front face.
    if low_z <= tower.bounds()[1][2]:
        errors.append("the lens must stand in front of the tower front face (gap %.6f)" % (low_z - tower.bounds()[1][2]))
    for box in boxes:
        if box.index != GLASS and overlap(glass, box):
            errors.append("the lens overlaps the " + NAMES[box.index])
    return errors


# fixture_smooth_glass.h: kMaximumTrappedCameraPixels and kMaximumTrappedCameraFraction.
MAXIMUM_TRAPPED_PIXELS, MAXIMUM_TRAPPED_FRACTION = 4096, 0.005


def refusing(report, water_stride=24):
    """Pixels that refuse the frame. VISITS without a spill completes as Trapped
    (dark, power absorbed), which the frame admits while trapped pixels stay at
    or under 4096 and 0.5% of the drawn pixels. The sampled water pixels stand
    for water_stride squared drawn pixels each."""
    trapped = report["refused"].get("VISITS", 0)
    others = sum(report["refused"].values()) - trapped
    drawn = report["glass"] + report["water"] * water_stride * water_stride
    if trapped > MAXIMUM_TRAPPED_PIXELS or trapped > MAXIMUM_TRAPPED_FRACTION * drawn:
        return others + trapped
    return others


def main():
    errors = []
    errors.extend(lens_errors(BOXES))
    errors.extend(placement_errors(BOXES))
    shipped = survey("shipped")
    describe("shipped walk, starter", shipped)
    # The sphere lens is thin, so a few camera rays entering at its top rim
    # bounce inside it by total internal reflection past the visit budget. They
    # publish as Trapped pixels; anything else, or too many of them, refuses.
    if refusing(shipped) != 0:
        errors.append("the shipped walk refuses starter pixels")
    if shipped["max_queries"] > 2 + 3 * VISITS_PER_CHANNEL:
        errors.append("a pixel needs more than kPairPixelRayQueries queries")
    if shipped["shared_plane_hits"] != 0:
        errors.append("a branch meets two box faces at one distance")
    # Glass Caustics off: the renderer prepares a camera-only selection of the
    # glass beside the water, so the camera runs this same paired walk with the
    # same budget and no photon pass. Its refusals are the shipped walk's.
    caustic_off = refusing(shipped)
    print("caustic off, camera-only paired walk: %d refusing, %d trapped"
          % (caustic_off, shipped["refused"].get("VISITS", 0)))
    if caustic_off != 0:
        errors.append("the caustic-off camera walk refuses starter pixels")
    caustic = caustic_survey(BOXES)
    if describe_caustic("caustic, starter", caustic) != 0:
        errors.append("the caustic branch walk refuses starter entries")
    # The lens must show a focused spot: at least half of the sun power it takes in
    # lands within 5 cm of the spot centre on the tower front face.
    focused = caustic["spot"] / caustic["power"] if caustic["power"] > 0.0 else 0.0
    print("caustic focus: %.3f of the lens power inside the %.2f spot, %.3f deposited on receivers"
          % (focused, SPOT_RADIUS, caustic["deposited"] / max(caustic["power"], 1.0e-30)))
    if focused < 0.5:
        errors.append("the lens no longer focuses half its power into the tower-front spot (%.3f)" % focused)
    # The model reproduces the live refusal: 680 pixels, first at texel 1649,205.
    before = survey("channel", boxes=TOUCHING)
    if before["refused"].get("VISITS", 0) == 0 or before["first"][:2] != (1649, 205):
        errors.append("the earlier walk no longer reproduces the live VISITS refusal: " + str(before["first"]))
    if "--compare" in sys.argv:
        describe("earlier per-channel walk, touching glass", before)
        describe("shipped walk, touching glass, lantern wins the tie", survey("shipped", boxes=TOUCHING))
        describe("shipped walk, touching glass, glass wins the tie", survey("shipped", glass_wins_ties=True, boxes=TOUCHING))
        describe("earlier per-channel walk, starter", survey("channel"))
        describe("shipped walk, dispersion 0.004", survey("shipped", dispersion=0.004))
        describe("shipped walk, wave at 1 s and 2 s", survey("shipped", times=(1.0, 2.0)))
        describe_caustic("caustic, touching glass", caustic_survey(TOUCHING))
        describe_caustic("caustic, dispersion 0.004", caustic_survey(BOXES, dispersion=0.004, coarse=True))
    if errors:
        for error in errors:
            print("FAIL: " + error, file=sys.stderr)
        raise SystemExit(1)
    print("lighthouse starter walk model: PASS")


if __name__ == "__main__":
    main()
