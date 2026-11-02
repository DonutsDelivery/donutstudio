#!/usr/bin/env python3
"""CPU model of the Lighthouse and Ocean camera transport and caustic at 4K.

The scene comes from the built-in starter (plugin/Source/graph/
BuiltInVisualModules.h, lighthouseAndOceanStarter): the water grid and wave at
time zero, the tower, lantern and rock cubes, the glass dome, the camera and
the sun direction are read from its authored values. Pixel (x, y) maps to a ray
as in glass_caustics::cameraRay: y runs down from the top row.

The dome is one closed glass object of two shells, as Join Geometry builds it:
the outer shell is the closed sphere GeometryCore.h generates (a top pole,
latitude rings of `segments` vertices, a bottom pole, every triangle wound
outward), turned about y by Glass Dome Shell Turn, and the cavity is a
generated cube mirrored in x by Transform Geometry, which winds it inward.
Glass Dome Position moves both. Each shell is convex, so a walk query clips the
ray against the face planes of each shell (dome_errors checks that this agrees
with triangle hits). A hit carries the triangle's geometric normal: outward on
the outer shell and into the cavity on the cavity shell, so the walks'
inside/outside parity crosses outer glass, air cavity, inner glass and air as
fixture_smooth_glass.h admits it.

glass_pair_walk mirrors glassPairWalk in fixture_smooth_glass.h: the 1e-5
branch and 1e-4 discarded power thresholds, GLASS_INTERFACE_DEPTH, the medium
and orientation refusals, heaviest-first visits, the lightest-branch spill when
the slots are full, the truncation tolerance, and one shared 48-visit walk for
nondispersive glass. caustic_walk mirrors traceBranches in
fixture_glass_caustics.h with the branch policy read from the starter's render
node, and caustic entries mirror appendEntrySamples and the source leg of
tracePath: every glass triangle facing the sun takes 16 entries (4 subdivisions
per edge), and an entry whose ray to the sun meets anything (a cavity wall
behind the outer glass) is source-blocked. The query mirrors sceneClosestHit:
the nearest hit wins, and an exact distance tie goes to the lower scene object
index.

Run without arguments for the contract: every glass pixel and a stride of water
pixels at 3840x2160, and every caustic entry the renderer traces, must
complete; the dome must clear the water, the rock and the lighthouse it
encloses; its triangles and sunlit triangles must fit the caustic's entry and
frame query budgets; the renderer's entries, each moved by up to 1e-4 with four
seeds, must still complete; and the caustic's completed walks, sampled on a
finer grid of entries, must lay a readable pattern on the tower front. Refused
walks on that grid are reported, with the share of tower power they carry, and
left out of the pattern. The pixels run on LIGHTHOUSE_MODEL_WORKERS processes
(default: up to 16). --compare also runs the earlier per-channel 16-visit walk
on the earlier geometry, and the glass with dispersion 0.004.
"""

import math
import multiprocessing
import os
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
    """Authored node values of the domed lighthouse starter, keyed by node label.

    The shared builder serves two starters, so this model parses the glass-dome
    variant: its branch is written first, and the first authored value for a
    label wins, so the open starter's finer grid, low sun and lower tower do not
    replace the values the caustic walk is measured against.
    """
    root = Path(__file__).resolve().parents[2]
    text = (root / "plugin/Source/graph/BuiltInVisualModules.h").read_text(encoding="utf-8")
    begin = text.index("inline Build build (bool glassDome)")
    body = text[begin:text.index("\n}\n", begin)]
    values = {}
    for label, pairs in re.findall(r'TRANS\("([^"]+)"\),\s*\{(\{"[^;]*?\})\}\)', body):
        values.setdefault(label, {key: float(value)
                                  for key, value in re.findall(r'\{"(\w+)", (-?[0-9.]+)\}', pairs)})
    dome = re.search(r'\{\{"value", ([0-9.]+)\}\}, "Dome IOR"', body)
    water = re.search(r'\{0\.0, 0\.0, 0\.0\},\s*[0-9.]+, [0-9.]+, ([0-9.]+), ([0-9.]+), 1001,', body)
    if dome is None or water is None:
        raise SystemExit("FAIL: the starter IORs were not found")
    values["iors"] = {"glass": float(dome.group(1)), "water": float(water.group(1)), "waterDispersion": float(water.group(2))}
    return values


STARTER = starter_values()
# The starter's caustic branch policy (glassCausticMaximum*). The dome needs 48
# interfaces, so the starter raises the policy above the default 32.
RENDER = STARTER["Render Lighthouse and Ocean"]
CAUSTIC_VISITS, CAUSTIC_PENDING, CAUSTIC_INTERFACES = (int(RENDER["glassCausticMaximum" + name])
                                                       for name in ("Visits", "Pending", "Interfaces"))


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


class Shell:
    """One closed convex shell of the dome in world space.

    points and indices are the mesh as authored; each triangle keeps its
    geometric normal from its winding. sign is 1 for the outer shell (wound
    outward) and -1 for the cavity (wound inward), so sign * normal is the
    outward normal of the convex polyhedron the shell bounds.
    """

    def __init__(self, points, indices, sign):
        self.points, self.indices, self.sign = points, indices, sign
        self.triangles = []
        for i, j, k in indices:
            a, b, c = points[i], points[j], points[k]
            e1, e2 = sub(b, a), sub(c, a)
            self.triangles.append((a, e1, e2, normalize(cross(e1, e2))))
        # Clip planes (n, d) with n the polyhedron's outward normal: inside means n.p <= d.
        # A quad's two triangles share a plane, so each plane is kept once.
        planes = {}
        for a, _, _, normal in self.triangles:
            plane = (mul(normal, sign), dot(mul(normal, sign), a))
            planes.setdefault(tuple(round(v, 9) for v in plane[0] + (plane[1],)), plane)
        self.planes = list(planes.values())
        count = len(points)
        self.centre = tuple(sum(p[a] for p in points) / count for a in range(3))
        self.radius = max(math.sqrt(dot(sub(p, self.centre), sub(p, self.centre))) for p in points) * (1 + 1e-9)

    def clip(self, origin, direction):
        """(near, near plane normal, far, far plane normal) of the ray line inside the polyhedron, or None."""
        offset = sub(origin, self.centre)
        b = dot(offset, direction)
        if dot(offset, offset) - b * b > self.radius * self.radius:
            return None
        near, far, near_normal, far_normal = -math.inf, math.inf, None, None
        ox, oy, oz = origin
        dx, dy, dz = direction
        for (nx, ny, nz), d in self.planes:
            slope = nx * dx + ny * dy + nz * dz
            room = d - (nx * ox + ny * oy + nz * oz)
            if -1.0e-14 < slope < 1.0e-14:
                if room < 0.0:
                    return None
                continue
            t = room / slope
            if slope < 0.0:
                if t > near:
                    near, near_normal = t, (nx, ny, nz)
            elif t < far:
                far, far_normal = t, (nx, ny, nz)
        if near > far:
            return None
        return near, near_normal, far, far_normal

    def contains(self, point):
        """Signed distance inside the polyhedron (positive inside)."""
        return min(d - dot(n, point) for n, d in self.planes)

    def closed(self):
        """True when welded positions give every directed edge exactly one reverse partner."""
        weld = {}
        ids = [weld.setdefault(tuple(round(c, 9) + 0.0 for c in p), len(weld)) for p in self.points]
        edges = {}
        for i, j, k in self.indices:
            for a, b in ((ids[i], ids[j]), (ids[j], ids[k]), (ids[k], ids[i])):
                edges[a, b] = edges.get((a, b), 0) + 1
        return all(count == 1 and edges.get((b, a)) == 1 for (a, b), count in edges.items())

    def volume(self):
        """Signed volume: positive for an outward shell, negative for a cavity."""
        return sum(dot(self.points[i], cross(self.points[j], self.points[k])) for i, j, k in self.indices) / 6.0

    def triangle_hit(self, origin, direction, minimum=0.0):
        """Nearest triangle crossing at or past `minimum`, with the triangle's geometric normal."""
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
            if distance >= minimum and (best is None or distance < best[0]):
                best = (distance, normal)
        return best


def cube_mesh(width, height, depth):
    """Positions and triangles of GeometryCore.h's cube generator: four vertices per face, wound outward."""
    faces = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))
    positions, triangles = [], []
    for face in faces:
        first = len(positions)
        for corner in face:
            positions.append(((0.5 if corner & 1 else -0.5) * width, (0.5 if corner & 2 else -0.5) * height,
                              (0.5 if corner & 4 else -0.5) * depth))
        triangles += [(first, first + 1, first + 2), (first, first + 2, first + 3)]
    return positions, triangles


def cylinder_mesh(radius, height, segments):
    """Positions and triangles of GeometryCore.h's cylinder generator: a side ring pair, separate cap rings
    and the two cap centres, all wound outward."""
    positions = []
    for ring in range(4):
        for column in range(segments):
            angle = 2 * math.pi * column / segments
            positions.append((radius * math.cos(angle), (-0.5 if ring % 2 == 0 else 0.5) * height,
                              radius * math.sin(angle)))
    bottom, top = len(positions), len(positions) + 1
    positions += [(0.0, -0.5 * height, 0.0), (0.0, 0.5 * height, 0.0)]
    triangles = []
    for column in range(segments):
        following = (column + 1) % segments
        triangles.append((column, column + segments, following))
        triangles.append((following, column + segments, following + segments))
        triangles.append((bottom, 2 * segments + column, 2 * segments + following))
        triangles.append((top, 3 * segments + following, 3 * segments + column))
    return positions, triangles


def icosphere_mesh(radius, subdivisions):
    """Positions and triangles of GeometryCore.h's icosphere generator: an icosahedron whose triangles
    are split in four `subdivisions` times, with every new vertex pushed out to the radius."""
    t = (1.0 + math.sqrt(5.0)) * 0.5

    def project(v):
        length = math.sqrt(dot(v, v))
        return (v[0] / length * radius, v[1] / length * radius, v[2] / length * radius)

    positions = [project(v) for v in ((-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
                                      (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1))]
    flat = (0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1)
    triangles = [flat[i:i + 3] for i in range(0, len(flat), 3)]
    for _ in range(subdivisions):
        midpoints = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                midpoints[key] = len(positions)
                positions.append(project(mul(add(positions[a], positions[b]), 0.5)))
            return midpoints[key]

        previous, triangles = triangles, []
        for a, b, c in previous:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            triangles += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return positions, triangles


def shape_mesh(shape):
    if "subdivisions" in shape:
        return icosphere_mesh(shape["radius"], int(shape["subdivisions"]))
    if "rings" in shape:
        return sphere_mesh(shape["radius"], int(shape["segments"]), int(shape["rings"]))
    if "radius" in shape:
        return cylinder_mesh(shape["radius"], shape["height"], int(shape["segments"]))
    return cube_mesh(shape["width"], shape["height"], shape["depth"])


class Dome:
    """The starter's glass dome: an outward sphere shell joined with an inward cavity cube.

    Transform Geometry scales, then rotates, then translates, so the shell turn
    turns the sphere about y, and the cavity mirror flips the cube in x (winding
    it inward) and lifts it to the lighthouse.
    """

    def __init__(self, index, outer, cavity, mirror, placement, turn=None):
        turn = turn or {}
        self.index = index
        self.outer_shape, self.cavity_shape = outer, cavity
        self.centre = tuple(placement.get("translation" + axis, 0.0) for axis in "XYZ")
        if any(placement.get(key, default) != default for key, default in
               (("rotationX", 0.0), ("rotationY", 0.0), ("rotationZ", 0.0),
                ("scaleX", 1.0), ("scaleY", 1.0), ("scaleZ", 1.0))):
            raise SystemExit("FAIL: the model expects Glass Dome Position to only translate the dome")
        if any(mirror.get(key, 0.0) != 0.0 for key in ("rotateX", "rotateY", "rotateZ")):
            raise SystemExit("FAIL: the model expects the cavity mirror to only scale and translate")
        if any(turn.get(key, default) != default for key, default in
               (("rotateX", 0.0), ("rotateZ", 0.0), ("scaleX", 1.0), ("scaleY", 1.0), ("scaleZ", 1.0),
                ("translateX", 0.0), ("translateY", 0.0), ("translateZ", 0.0))):
            raise SystemExit("FAIL: the model expects Glass Dome Shell Turn to only turn the shell about y")
        self.turn = turn.get("rotateY", 0.0)
        # GeometryCore's rotateY: x' = cos x + sin z, z' = -sin x + cos z.
        cosine, sine = math.cos(math.radians(self.turn)), math.sin(math.radians(self.turn))
        turned = lambda p: (cosine * p[0] + sine * p[2], p[1], -sine * p[0] + cosine * p[2])
        self.mirror = tuple(mirror.get("scale" + axis, 1.0) for axis in "XYZ")
        lift = tuple(mirror.get("translate" + axis, 0.0) for axis in "XYZ")
        self.shells = []
        for shape, scale, shift, spin in ((outer, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), turned),
                                          (cavity, self.mirror, lift, lambda p: p)):
            positions, triangles = shape_mesh(shape)
            points = [add(add(spin(tuple(p[a] * scale[a] for a in range(3))), shift), self.centre) for p in positions]
            # A negative scale determinant reverses the winding without reordering the indices.
            determinant = scale[0] * scale[1] * scale[2]
            self.shells.append(Shell(points, triangles, 1 if determinant > 0 else -1))
        self.outer, self.cavity = self.shells
        self.cavity_centre = add(lift, self.centre)
        self.triangle_count = sum(len(shell.indices) for shell in self.shells)

    def vertices(self):
        return list(self.outer.points)

    def bounds(self):
        return (tuple(min(p[a] for p in self.outer.points) for a in range(3)),
                tuple(max(p[a] for p in self.outer.points) for a in range(3)))

    def hit(self, origin, direction, minimum=0.0):
        """Nearest shell crossing at or past `minimum`, with the triangle's geometric normal."""
        best = None
        for shell in self.shells:
            clipped = shell.clip(origin, direction)
            if clipped is None:
                continue
            near, near_normal, far, far_normal = clipped
            for distance, normal in ((near, near_normal), (far, far_normal)):
                if normal is not None and distance >= minimum and (best is None or distance < best[0]):
                    best = (distance, mul(normal, shell.sign), self.index)
                    break
        return best

    def entry_triangles(self, toward):
        """(triangle, cosine) of every glass triangle facing the sun, as appendEntrySamples selects them."""
        for shell in self.shells:
            for triangle in shell.triangles:
                cosine = dot(triangle[3], toward)
                if cosine > 0.0:
                    yield triangle, cosine

    def entry_points(self, toward, samples):
        """Centroids of a `samples`-per-edge subdivision of every sunlit triangle, with their sun power.

        The power is the sub-triangle area times the cosine to the sun, as tracePath's incident power.
        """
        for (a, e1, e2, normal), cosine in self.entry_triangles(toward):
            power = 0.5 * math.sqrt(dot(cross(e1, e2), cross(e1, e2))) / (samples * samples) * cosine
            for i in range(samples):
                for j in range(samples - i):
                    yield add(a, add(mul(e1, (i + 1.0 / 3.0) / samples), mul(e2, (j + 1.0 / 3.0) / samples))), power
                    if i + j < samples - 1:
                        yield add(a, add(mul(e1, (i + 2.0 / 3.0) / samples), mul(e2, (j + 2.0 / 3.0) / samples))), power


def starter_box(index, mesh, position):
    size, at = STARTER[mesh], STARTER[position]
    centre = tuple(at.get("translation" + axis, 0.0) for axis in "XYZ")
    return Box(index, centre, (size["width"] / 2, size["height"] / 2, size["depth"] / 2))


def starter_dome(index):
    return Dome(index, STARTER["Glass Dome Shell"], STARTER["Glass Dome Cavity"],
                STARTER["Glass Dome Cavity Mirror"], STARTER["Glass Dome Position"], STARTER["Glass Dome Shell Turn"])


BOXES = (
    starter_box(TOWER, "Lighthouse Tower", "Lighthouse Position"),
    starter_box(LANTERN, "Lighthouse Lantern", "Lantern Position"),
    starter_dome(GLASS),
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
        # Boxes and the dome both answer hit(origin, direction, minimum).
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


def thin_entry_diagnostic(x=1396, y=1063):
    """The runtime thin-band ownership check at one 4K texel, exit-capable.

    Live builds refuse the 4K Accurate export frame at texel 1396,1063 with
    "Closed glass ray transport did not produce a complete finite result
    (first hit is not the water surface; 1 pixel)". This mirrors
    fixture_smooth_glass.h thinEntryStatus there: the center ray enters the
    glass dome first, the past-tolerance second query starts inside the dome
    and returns the dome's exit face, and that exit lands before the water
    entry, so thinEntryOwned fails and the whole frame refuses. Scene.closest
    reports dome entries only (Dome.hit clips at the near face), so the exit
    comes from the shells' clip intervals directly. Returns the named verdict.
    """
    scene = Scene()
    direction = camera_ray(x + 0.5, y + 0.5)
    water = scene.water_hit(CAMERA, direction, math.inf)
    if water is None:
        return {"error": "no float water entry on the ray"}
    entry = add(CAMERA, mul(direction, water[0]))
    tolerance = max(4.0 * bias(entry), 1.0e-4 * water[0])

    def surfaces_at(minimum):
        best = None

        def offer(hit):
            nonlocal best
            if hit is None:
                return
            if best is None or hit[0] < best[0]:
                best = hit

        for box in scene.boxes:
            if isinstance(box, Dome):
                for shell in box.shells:
                    clip = shell.clip(CAMERA, direction)
                    if clip is None:
                        continue
                    near, _, far, _ = clip
                    for t in (near, far):
                        if math.isfinite(t) and t >= minimum:
                            offer((t, None, box.index, None))
            else:
                offer(box.hit(CAMERA, direction, minimum))
        offer(scene.water_hit(CAMERA, direction, best[0] if best else math.inf, minimum))
        return best

    first = surfaces_at(0.0)
    if first is None:
        return {"error": "the ray misses every surface"}
    beyond = surfaces_at(first[0] + tolerance)
    owned_first = first[2] == WATER or first[0] >= water[0] - tolerance
    owned_beyond = (beyond is None or beyond[2] == WATER
                    or beyond[0] >= water[0] - tolerance)
    result = {
        "first": (NAMES[first[2]], first[0]),
        "beyond": None if beyond is None else (NAMES[beyond[2]], beyond[0]),
        "water": water[0],
        "tolerance": tolerance,
        "refuses": not (owned_first or owned_beyond),
    }
    for box in scene.boxes:
        if isinstance(box, Dome) and box.index == first[2]:
            clip = box.shells[0].clip(CAMERA, direction)
            if clip:
                result["domeSliver"] = clip[2] - clip[0]
            break
    return result


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


def caustic_walk(scene, entry, ior, deposits=None, depth=None):
    """Mirror of traceBranches for a smooth glass entry: (status, unresolved fraction).

    Last in, first out; a branch below the minimum fraction, a full pending
    list, an interface at the bound and the visit bound all add unresolved
    power, which may not pass CAUSTIC_UNRESOLVED. An inside branch that leaves
    the glass for anything else refuses, as does a receiver met from behind.
    Given a list, deposits collects (object index, hit position, fraction) for
    every receiver hit. Given a dict, depth gets the visits the walk took and
    the most interfaces a visited branch had crossed, and for an ORDER refusal
    the origin of the branch that met the glass from the wrong side.
    """
    toward = TOWARD_SUN
    offset = max(CAUSTIC_RAY_MINIMUM, 2.0e-6 * max(abs(v) for v in entry))
    pending = [(add(entry, mul(toward, 2 * offset)), mul(toward, -1.0), 1.0, 0, False)]
    unresolved = 0.0
    visits = deepest = 0
    if depth is not None:
        depth.update(visits=0, interfaces=0)
    while pending and visits < CAUSTIC_VISITS:
        origin, direction, fraction, interfaces, inside = pending.pop()
        visits += 1
        deepest = max(deepest, interfaces)
        if depth is not None:
            depth.update(visits=visits, interfaces=deepest)
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
            if depth is not None:
                depth["order_origin"] = origin
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


def solid(boxes, index):
    return [box for box in boxes if box.index == index][0]


# The earlier geometry, kept so the model still reproduces the live VISITS
# refusal: the earlier 0.5 glass cube with its bottom exactly on the lantern top,
# at the lantern's x and z, beside the earlier 2.6 tower and the earlier rock.
# The starter has a dome now and a shorter tower on a lower rock, so all of it is
# fixed here.
TOUCHING = (
    Box(TOWER, (-1.2, 1.3, -1.2), (0.325, 1.3, 0.325)),
    Box(LANTERN, (-1.2, 2.8, -1.2), (0.475, 0.2, 0.475)),
    Box(GLASS, (-1.2, 3.25, -1.2), (0.25, 0.25, 0.25)),
    Box(ROCK, (-1.2, 0.05, -1.2), (0.95, 0.325, 0.85)),
)
BOX_SETS = {"starter": BOXES, "touching": TOUCHING}

# appendEntrySamples' entrySubdivisions: the renderer traces 16 entries per lit
# triangle. The pattern samples each lit triangle on a finer grid, so the cells
# on the tower front collect enough paths.
RENDERER_ENTRY_SAMPLES, DOME_ENTRY_SAMPLES = 4, 24
# The robustness check moves every entry by up to this much, with this many seeds.
JITTER, JITTER_SEEDS = 1.0e-4, 4
# Pattern cells on the tower front face, in metres.
CELL_WIDTH, CELL_HEIGHT = 0.065, 0.1


def caustic_chunk(task):
    """One worker's share of the caustic survey: entries[begin:end] of the box set.

    With a jitter scale, each entry moves by up to that much along each axis,
    from a generator seeded by the seed and the chunk's first entry, and then
    back onto the plane of its glass face. An entry left off the face would
    start inside the glass, where the source leg blocks it, or float above it.
    """
    key, dispersion, coarse, renderer, jitter, seed, begin, end = task
    boxes = BOX_SETS[key]
    scene = Scene(0.0, False, boxes)
    entries = caustic_entries(boxes, coarse, renderer)[begin:end]
    if jitter:
        generator = random.Random(seed * 1000003 + begin)
        glass = glass_solid(boxes)
        planes = [plane for shell in glass.shells for plane in shell.planes]
        moved = []
        for point, power in entries:
            normal, d = min(planes, key=lambda plane: abs(plane[1] - dot(plane[0], point)))
            shifted = tuple(v + generator.uniform(-jitter, jitter) for v in point)
            moved.append((add(shifted, mul(normal, d - dot(normal, shifted))), power))
        entries = moved
    iors = volume_iors(dispersion)
    tower = solid(boxes, TOWER)
    (low_x, low_y, _), (_, _, front) = tower.bounds()
    report = empty_caustic_report()
    for point, power in entries:
        offset = max(CAUSTIC_RAY_MINIMUM, 2.0e-6 * max(abs(v) for v in point))
        # The source leg: an entry with anything between it and the sun launches no power.
        if scene.closest(add(point, mul(TOWARD_SUN, 2 * offset)), TOWARD_SUN) is not None:
            report["blocked"] += power
            continue
        for ior in iors[:1] if dispersion == 0.0 else iors:
            deposits, depth = [], {}
            status, unresolved = caustic_walk(scene, point, ior, deposits, depth)
            report["entries"] += 1
            report["max_unresolved"] = max(report["max_unresolved"], unresolved)
            report["max_visits"] = max(report["max_visits"], depth["visits"])
            report["max_interfaces"] = max(report["max_interfaces"], depth["interfaces"])
            complete = status == "COMPLETE"
            if not complete:
                report["refused"][status] = report["refused"].get(status, 0) + 1
                report["refusals"].append((status, point, unresolved, depth["visits"], depth["interfaces"],
                                           depth.get("order_origin")))
            report["power"] += power
            # A refused walk's partial deposits are kept apart: the renderer would
            # refuse the frame, so they never belong in the pattern.
            cells = report["cells"] if complete else report["refused_cells"]
            for index, position, fraction in deposits:
                report["deposited" if complete else "refused_deposited"] += power * fraction
                if index == TOWER and abs(position[2] - front) < 1.0e-6:
                    cell = (int((position[0] - low_x) / CELL_WIDTH), int((position[1] - low_y) / CELL_HEIGHT))
                    cells[cell] = cells.get(cell, 0.0) + power * fraction
    return report


def empty_caustic_report():
    return {"entries": 0, "refused": {}, "refusals": [], "max_unresolved": 0.0, "max_visits": 0, "max_interfaces": 0,
            "power": 0.0, "blocked": 0.0, "deposited": 0.0, "refused_deposited": 0.0, "cells": {}, "refused_cells": {}}


def caustic_entries(boxes, coarse, renderer=False):
    glass = glass_solid(boxes)
    if isinstance(glass, Dome):
        if renderer:
            return list(glass.entry_points(TOWARD_SUN, RENDERER_ENTRY_SAMPLES))
        return list(glass.entry_points(TOWARD_SUN, DOME_ENTRY_SAMPLES // (2 if coarse else 1)))
    return [(point, 0.0) for point in glass.entry_points(TOWARD_SUN, glass.entry_samples // (2 if coarse else 1))]


def caustic_survey(key="starter", dispersion=0.0, coarse=False, renderer=False, jitter=0.0, seed=0):
    """Entries on every glass surface the sun lights: a grid per box face or a subdivided lit triangle.

    renderer takes the 16 entries per lit triangle the renderer traces. For the
    dome the report also sums the sun power the lit triangles take in, the
    power the source leg blocks, the power deposited on receivers and the power
    deposited in each cell of the tower front face, with the most visits and
    interfaces any entry needed. Deposits of refused walks are summed apart, in
    refused_deposited and refused_cells.
    """
    count = len(caustic_entries(BOX_SETS[key], coarse, renderer))
    step = max(1, -(-count // (WORKERS * 4)))
    tasks = [(key, dispersion, coarse, renderer, jitter, seed, begin, min(count, begin + step))
             for begin in range(0, count, step)]
    report = empty_caustic_report()
    for part in run_pool(caustic_chunk, tasks):
        for name in ("entries", "power", "blocked", "deposited", "refused_deposited"):
            report[name] += part[name]
        report["refusals"] += part["refusals"]
        for name in ("max_unresolved", "max_visits", "max_interfaces"):
            report[name] = max(report[name], part[name])
        for status, count in part["refused"].items():
            report["refused"][status] = report["refused"].get(status, 0) + count
        for name in ("cells", "refused_cells"):
            for cell, power in part[name].items():
                report[name][cell] = report[name].get(cell, 0.0) + power
    return report


def pattern(report, boxes=BOXES):
    """Caustic irradiance of each lit tower-front cell, relative to the bare sun on that face.

    Light reaches the tower only through the dome, so a ratio near 1 is an even
    glow, and bright and dark cells are the facets' caustic pattern. Cells the
    lantern shades from the sun are left out, as are partial cells at the edges.
    """
    tower, lantern = solid(boxes, TOWER), solid(boxes, LANTERN)
    (low_x, low_y, _), (high_x, high_y, front) = tower.bounds()
    columns, rows = int((high_x - low_x) / CELL_WIDTH + 1e-9), int((high_y - low_y) / CELL_HEIGHT + 1e-9)
    bare = CELL_WIDTH * CELL_HEIGHT * TOWARD_SUN[2]
    ratios = {}
    for column in range(columns):
        for row in range(rows):
            corners = [(low_x + (column + i) * CELL_WIDTH, low_y + (row + j) * CELL_HEIGHT, front)
                       for i in (0, 1) for j in (0, 1)]
            if any(lantern.hit(corner, TOWARD_SUN) is not None for corner in corners):
                continue
            ratios[column, row] = report["cells"].get((column, row), 0.0) / bare
    return ratios


def overlap(a, b):
    """True when two bounding boxes share volume (touching faces do not count)."""
    (alow, ahigh), (blow, bhigh) = a.bounds(), b.bounds()
    return all(alow[axis] < bhigh[axis] and blow[axis] < ahigh[axis] for axis in range(3))


def tree_nodes(count, leaf):
    """Nodes of the median-split tree glass_caustics::treeNodeCount builds over `count` items."""
    if count <= leaf:
        return 1
    return 1 + tree_nodes(count // 2, leaf) + tree_nodes(count - count // 2, leaf)


# fixture_glass_caustics.h: the starter's entrySubdivisions 4 (16 entries per triangle) over 8 bands, and 194
# queries per path.
MAXIMUM_ENTRY_TRIANGLES = 128
PATHS_PER_LIT_TRIANGLE, QUERIES_PER_PATH = 16 * 8, 194


def budget(boxes=BOXES):
    """(triangles, lit triangles, query allowance, frame query work) of the dome caustic."""
    dome = glass_solid(boxes)
    grid = STARTER["Ocean Grid"]
    meshes = [2 * (int(grid["columns"]) - 1) * (int(grid["rows"]) - 1)]
    meshes += [12 if isinstance(box, Box) else box.triangle_count for box in boxes]
    allowance = sum(tree_nodes(count, 4) + count for count in meshes) + tree_nodes(len(meshes), 1)
    lit = sum(1 for _ in dome.entry_triangles(TOWARD_SUN))
    return dome.triangle_count, lit, allowance, lit * PATHS_PER_LIT_TRIANGLE * QUERIES_PER_PATH * allowance


def dome_errors(boxes):
    """The authored dome: a closed outward sphere around a closed inward cavity cube, as the generators make them."""
    dome = glass_solid(boxes)
    errors = []
    if "rings" not in dome.outer_shape or "rings" in dome.cavity_shape or "radius" in dome.cavity_shape:
        errors.append("the dome must be a sphere shell around a cube cavity")
        return errors
    segments, rings = int(dome.outer_shape["segments"]), int(dome.outer_shape["rings"])
    # GraphInstrumentTests pins the scene totals: 2 + segments * (rings - 1) vertices and
    # 6 * segments * (rings - 1) indices for the sphere, 24 and 36 for the cube.
    for shape, shell, name, vertices, indices in (
            (dome.outer_shape, dome.outer, "shell", 2 + segments * (rings - 1), 6 * segments * (rings - 1)),
            (dome.cavity_shape, dome.cavity, "cavity", 24, 36)):
        positions, triangles = shape_mesh(shape)
        if len(positions) != vertices or 3 * len(triangles) != indices:
            errors.append("the %s mesh no longer has the generator's vertex and index counts" % name)
        if not shell.closed():
            errors.append("the dome %s is not a closed shell" % name)
        # Each shell is convex around its own centre: the outer shell's normals point
        # away from it and the cavity's toward it.
        for a, e1, e2, normal in shell.triangles:
            centroid = add(a, mul(add(e1, e2), 1.0 / 3.0))
            if shell.sign * dot(normal, sub(centroid, shell.centre)) <= 0.0:
                errors.append("a dome %s triangle is wound the wrong way" % name)
                break
    # The searched shape: an 8 by 4 shell turned half a segment. Its sun paths
    # circle inside the facets for up to 43 interfaces and 56 visits.
    if (segments, rings, dome.turn % (360.0 / segments)) != (8, 4, 22.5):
        errors.append("the dome must be an 8 by 4 shell turned 22.5 degrees")
    if dome.mirror[0] * dome.mirror[1] * dome.mirror[2] >= 0.0:
        errors.append("the cavity mirror must turn the cavity inward")
    outer_volume, cavity_volume = dome.outer.volume(), dome.cavity.volume()
    if not (outer_volume > 0.0 and cavity_volume < 0.0 and outer_volume + cavity_volume > 0.0):
        errors.append("the dome volumes %.6f and %.6f are not a shell around a cavity" % (outer_volume, cavity_volume))
    # Each shell is convex, so its half-space clip must agree with its triangle hits.
    generator, hits = random.Random(7), 0
    reach = dome.outer.radius
    for _ in range(6000):
        origin = add(dome.centre, tuple(generator.uniform(-1.5 * reach, 1.5 * reach) for _ in range(3)))
        target = add(dome.centre, tuple(generator.uniform(-0.8 * reach, 0.8 * reach) for _ in range(3)))
        direction = normalize(sub(target, origin))
        found = dome.hit(origin, direction)
        expected = None
        for shell in dome.shells:
            hit = shell.triangle_hit(origin, direction)
            if hit is not None and (expected is None or hit[0] < expected[0]):
                expected = hit
        hits += found is not None
        if (found is None) != (expected is None) or (found is not None and (
                abs(found[0] - expected[0]) > 1.0e-9 or dot(found[1], expected[1]) < 1.0 - 1.0e-9)):
            errors.append("a triangle hit disagrees with the convex half-space clip of the dome")
            break
    if hits < 1000:
        errors.append("the dome hit cross-check found only %d hits, too few to mean anything" % hits)
    return errors


# The glass keeps this much room from every other object, in metres.
CLEARANCE = 0.02


def project(point):
    inverse = (-CAMERA_ROTATION[0], -CAMERA_ROTATION[1], -CAMERA_ROTATION[2], CAMERA_ROTATION[3])
    local = rotate(inverse, sub(point, CAMERA))
    return ((local[0] / -local[2] / (TANGENT * WIDTH / HEIGHT) + 1) * WIDTH / 2,
            (1 - local[1] / -local[2] / TANGENT) * HEIGHT / 2)


def placement_errors(boxes):
    """The dome hangs clear of the rock and the waves, holds the lighthouse in its cavity, and is in frame."""
    dome = glass_solid(boxes)
    rock, tower, lantern = solid(boxes, ROCK), solid(boxes, TOWER), solid(boxes, LANTERN)
    errors = []
    (_, low_y, _), _ = dome.bounds()
    rock_top = rock.bounds()[1][1]
    crest = STARTER["Ocean Waves"]["amplitude"]
    if low_y - rock_top < CLEARANCE:
        errors.append("the dome must clear the rock top (gap %.6f)" % (low_y - rock_top))
    if low_y - crest < CLEARANCE:
        errors.append("the dome must clear the wave crests (gap %.6f)" % (low_y - crest))
    for box in (tower, lantern):
        margin = min(dome.cavity.contains(vertex) for vertex in box.vertices())
        if margin < CLEARANCE:
            errors.append("the %s must sit inside the dome cavity (margin %.6f)" % (NAMES[box.index], margin))
    # Glass thickness: the least depth of a cavity vertex inside the outer shell.
    # Checking only cavity vertices against the outer shell is enough here: both
    # shells are convex, the cavity lies inside the outer shell, and the depth of
    # a point inside a convex polyhedron (its least distance to a face plane) is
    # concave, so over the convex cavity it is least at one of its vertices.
    thickness = min(dome.outer.contains(vertex) for vertex in dome.cavity.points)
    if thickness < 0.08:
        errors.append("the dome glass must be at least 0.08 thick (found %.6f)" % thickness)
    for x, y in (project(vertex) for vertex in dome.outer.points):
        if not (0.0 <= x <= WIDTH and 0.0 <= y <= HEIGHT):
            errors.append("the camera does not show the whole dome (vertex at %.0f, %.0f)" % (x, y))
            break
    triangles, lit, allowance, work = budget(boxes)
    if triangles > MAXIMUM_ENTRY_TRIANGLES:
        errors.append("the dome has %d triangles, over the caustic's %d" % (triangles, MAXIMUM_ENTRY_TRIANGLES))
    print("dome: %d triangles, %d sunlit, allowance %d, frame query work %d; bottom %.3f over the rock "
          "and %.3f over the crests; lighthouse margin %.3f; glass at least %.3f thick"
          % (triangles, lit, allowance, work, low_y - rock_top, low_y - crest,
             min(dome.cavity.contains(v) for box in (tower, lantern) for v in box.vertices()), thickness))
    return errors


DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
WORKERS = max(1, int(os.environ.get("LIGHTHOUSE_MODEL_WORKERS", DEFAULT_WORKERS)))


def run_pool(function, tasks):
    if WORKERS == 1:
        return [function(task) for task in tasks]
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        return pool.map(function, tasks, chunksize=1)


def survey_chunk(task):
    """One worker's share of the pixel survey: rows of glass pixels and of sampled water pixels."""
    key, policy, dispersion, time, glass_wins_ties, glass_rows, water_rows, region, water_stride = task
    scene = Scene(time, glass_wins_ties, BOX_SETS[key])
    x0, x1 = region
    jobs = [(glass_pixel, x, y) for y in glass_rows for x in range(x0, x1)]
    jobs += [(water_pixel, x, y) for y in water_rows for x in range(0, WIDTH, water_stride)]
    report = {"glass": 0, "water": 0, "refused": {}, "first": None, "max_truncated": 0.0,
              "max_queries": 0, "shared_plane_hits": 0}
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
    report["shared_plane_hits"] = scene.shared_plane_hits
    return report


def survey(policy, dispersion=0.0, water_stride=24, glass_wins_ties=False, times=(0.0,), key="starter"):
    report = {"glass": 0, "water": 0, "refused": {}, "first": None, "max_truncated": 0.0,
              "max_queries": 0, "shared_plane_hits": 0}
    x0, x1, y0, y1 = glass_region(BOX_SETS[key])
    glass_rows, water_rows = list(range(y0, y1)), list(range(0, HEIGHT, water_stride))
    parts = max(1, WORKERS * 4)
    tasks = [(key, policy, dispersion, time, glass_wins_ties, glass_rows[part::parts], water_rows[part::parts],
              (x0, x1), water_stride) for time in times for part in range(parts)]
    for part in run_pool(survey_chunk, tasks):
        for name in ("glass", "water", "shared_plane_hits"):
            report[name] += part[name]
        report["max_truncated"] = max(report["max_truncated"], part["max_truncated"])
        report["max_queries"] = max(report["max_queries"], part["max_queries"])
        for status, count in part["refused"].items():
            report["refused"][status] = report["refused"].get(status, 0) + count
        first = part["first"]
        if first is not None and (report["first"] is None or (first[1], first[0]) < (report["first"][1], report["first"][0])):
            report["first"] = first
    return report


def describe(label, report):
    refused = sum(report["refused"].values())
    print("%s: %d glass and %d sampled water pixels, %d refused %s, first %s, max truncated %.5f, max queries %d, "
          "shared-plane hits %d" % (label, report["glass"], report["water"], refused, report["refused"], report["first"],
                                    report["max_truncated"], report["max_queries"], report["shared_plane_hits"]))
    return refused


def describe_caustic(label, report):
    refused = sum(report["refused"].values())
    print("%s: %d caustic entries, %d refused %s, max unresolved %.3g, at most %d of %d visits and %d of %d interfaces"
          % (label, report["entries"], refused, report["refused"], report["max_unresolved"],
             report["max_visits"], CAUSTIC_VISITS, report["max_interfaces"], CAUSTIC_INTERFACES))
    return refused


def cavity_edge_distance(point, boxes=BOXES):
    """Distance from a point to the nearest edge of the dome's box cavity."""
    cavity = glass_solid(boxes).cavity.points
    low = [min(p[a] for p in cavity) for a in range(3)]
    high = [max(p[a] for p in cavity) for a in range(3)]
    best = math.inf
    for a in range(3):
        b, c = [axis for axis in range(3) if axis != a]
        along = max(low[a] - point[a], 0.0, point[a] - high[a])
        for corner_b in (low[b], high[b]):
            for corner_c in (low[c], high[c]):
                best = min(best, math.sqrt(along ** 2 + (point[b] - corner_b) ** 2 + (point[c] - corner_c) ** 2))
    return best


def describe_grid(label, report):
    refused = describe_caustic(label, report)
    tower = sum(report["cells"].values())
    refused_tower = sum(report["refused_cells"].values())
    print("    %d of %d walks refused; refused walks carry %.2g of the deposited tower-front power, "
          "left out of the pattern" % (refused, report["entries"], refused_tower / max(tower + refused_tower, 1.0e-30)))
    for status, point, unresolved, visits, interfaces, origin in sorted(report["refusals"], key=lambda r: r[:5]):
        if status != "UNRESOLVED":
            print("    %s refusal at entry (%.6f, %.6f, %.6f): unresolved %.3g after %d visits, %d interfaces"
                  % ((status,) + point + (unresolved, visits, interfaces)))
        if status == "ORDER":
            # The real validator refuses the frame here too ("interface order is
            # invalid"), with the same 1e-5 ray minimum. A branch that leaves the
            # glass this close to a cavity edge skips the neighbouring face.
            print("    its last branch left the glass %.2g from a cavity edge; the ray minimum is %g"
                  % (cavity_edge_distance(origin), CAUSTIC_RAY_MINIMUM))
    return refused


def refusing(report):
    """Pixels that refuse the frame. VISITS without a spill completes as Trapped
    (dark, power absorbed), which the frame admits at any count."""
    return sum(report["refused"].values()) - report["refused"].get("VISITS", 0)


# The caustic pattern: lit tower-front cells at least this bright and at most
# this dark, relative to the bare sun on the face, must each cover this share of
# the lit cells.
BRIGHT, DARK, PATTERN_SHARE = 1.5, 0.5, 0.05


def main():
    errors = []
    errors.extend(dome_errors(BOXES))
    errors.extend(placement_errors(BOXES))
    shipped = survey("shipped")
    describe("shipped walk, starter", shipped)
    # A few camera rays bounce inside the glass by total internal reflection past
    # the visit budget. They publish as Trapped pixels; anything else refuses.
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
    # The renderer traces 16 entries on each sunlit triangle; every one must complete.
    traced = caustic_survey(renderer=True)
    if describe_caustic("caustic, renderer entries", traced) != 0:
        errors.append("the caustic branch walk refuses starter entries")
    if traced["max_unresolved"] > CAUSTIC_UNRESOLVED:
        errors.append("a caustic entry leaves more than the unresolved bound")
    # The renderer's entries again, each moved by up to JITTER, as a stand-in for
    # its float32 positions: none of them may refuse either.
    for seed in range(JITTER_SEEDS):
        if describe_caustic("caustic, renderer entries jittered %g, seed %d" % (JITTER, seed),
                            caustic_survey(renderer=True, jitter=JITTER, seed=seed)) != 0:
            errors.append("a jittered renderer caustic entry refuses (seed %d)" % seed)
    # The finer grid only measures the pattern, from its completed walks. The
    # renderer never traces these entries, so their refusals are reported, not
    # gated, with the share of tower-front power the refused walks would have
    # added and the raw and jittered counts side by side.
    caustic = caustic_survey()
    describe_grid("caustic, %d-per-edge pattern grid" % DOME_ENTRY_SAMPLES, caustic)
    for seed in range(JITTER_SEEDS):
        describe_grid("caustic, pattern grid jittered %g, seed %d" % (JITTER, seed),
                      caustic_survey(jitter=JITTER, seed=seed))
    # The dome must lay a readable pattern on the tower front: bright cells where
    # facets overlap their light and dark cells where they part it.
    ratios = pattern(caustic)
    values = sorted(ratios.values())
    bright = sum(1 for value in values if value >= BRIGHT) / max(len(values), 1)
    dark = sum(1 for value in values if value <= DARK) / max(len(values), 1)
    mean = sum(values) / max(len(values), 1)
    print("caustic pattern, completed walks only: %d lit tower-front cells, mean %.3f of the bare sun, range %.3f to %.3f, "
          "%.3f at or over %.1f and %.3f at or under %.1f; %.3f of the dome's sun power deposited, %.3f source-blocked"
          % (len(values), mean, values[0] if values else 0.0, values[-1] if values else 0.0, bright, BRIGHT,
             dark, DARK, caustic["deposited"] / max(caustic["power"], 1.0e-30),
             caustic["blocked"] / max(caustic["power"] + caustic["blocked"], 1.0e-30)))
    if "--pattern" in sys.argv:
        columns = 1 + max(column for column, _ in ratios) if ratios else 0
        rows = 1 + max(row for _, row in ratios) if ratios else 0
        for row in reversed(range(rows)):
            print(" ".join("%4.2f" % ratios[column, row] if (column, row) in ratios else "  - "
                           for column in range(columns)))
    if len(values) < 50:
        errors.append("too few lit tower-front cells to read a pattern (%d)" % len(values))
    if bright < PATTERN_SHARE or dark < PATTERN_SHARE:
        errors.append("the dome no longer lays a readable caustic pattern on the tower front")
    # The model reproduces the live refusal: 680 pixels, first at texel 1649,205.
    before = survey("channel", key="touching")
    if before["refused"].get("VISITS", 0) == 0 or before["first"][:2] != (1649, 205):
        errors.append("the earlier walk no longer reproduces the live VISITS refusal: " + str(before["first"]))
    # The live 4K Accurate export refusal is one texel (1396,1063) of the
    # 3840x2160 readback on build-20261024T163222Z-48c2d5fe: the thin band's
    # past-tolerance ownership hop starts inside the glass dome and returns
    # the dome exit, which lands before the water entry.
    diagnostic = thin_entry_diagnostic()
    print("thin entry texel 1396,1063: %r" % (diagnostic,))
    # The geometry the live refusal stands on: the ray's first surface is the
    # glass dome, as a facet sliver thinner than the ownership tolerance, with
    # the water entry far beyond it. The double-precision model says the
    # past-tolerance hop clears the sliver and reaches the water (refuses
    # False); the live float32 renderer refuses this texel, so the model's
    # verdict pins today's divergence for the fix slice to close.
    beyond = diagnostic.get("beyond")
    sliver = diagnostic.get("domeSliver")
    tolerance = diagnostic.get("tolerance")
    geometry_ok = (isinstance(sliver, float) and isinstance(tolerance, float) and sliver < tolerance
                   and isinstance(beyond, tuple) and beyond[0] == "water")
    if not (diagnostic.get("first", ("", 0.0))[0] == "glass" and geometry_ok
            and diagnostic.get("refuses") is False):
        errors.append("the thin-band diagnostic no longer describes the 4K first-hit refusal at texel 1396,1063: "
                      + str(diagnostic))
    if "--compare" in sys.argv:
        describe("earlier per-channel walk, touching glass", before)
        describe("shipped walk, touching glass, lantern wins the tie", survey("shipped", key="touching"))
        describe("shipped walk, touching glass, glass wins the tie", survey("shipped", glass_wins_ties=True, key="touching"))
        describe("earlier per-channel walk, starter", survey("channel"))
        describe("shipped walk, dispersion 0.004", survey("shipped", dispersion=0.004))
        describe("shipped walk, wave at 1 s and 2 s", survey("shipped", times=(1.0, 2.0)))
        describe_caustic("caustic, touching glass", caustic_survey("touching"))
        describe_caustic("caustic, dispersion 0.004", caustic_survey(dispersion=0.004, coarse=True))
    if errors:
        for error in errors:
            print("FAIL: " + error, file=sys.stderr)
        raise SystemExit(1)
    print("lighthouse starter walk model: PASS")


if __name__ == "__main__":
    main()
