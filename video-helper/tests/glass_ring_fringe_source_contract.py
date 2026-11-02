#!/usr/bin/env python3
# The Glass Ring's refraction stays clean only if the environment lookup stays
# bilinear on every ray and the starter blends its tube sides. Its bands need the
# starter's openly stronger Cauchy B and a path tracer that does not cap it.
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lookup = (root / "src/pathtrace/environment_map.cpp").read_text()
raster = (root / "src/gpu_backend/fixture_environment_image.h").read_text()
integrator = (root / "src/pathtrace/devices/integrator_body.inl").read_text()
modules = (root.parent / "plugin/Source/graph/BuiltInVisualModules.h").read_text()
limits = (root.parent / "shared/SurfaceMaterialIR.h").read_text()

radiance = lookup[lookup.index("Vec3 EnvironmentMapView::radiance"):]
radiance = radiance[:radiance.index("\n}\n")]
assert "mix (top, bottom, ty)" in radiance and "mix (texel (x0, y0), texel (x1, y0), tx)" in radiance
assert "Bilinear sampling is explicit on both APIs" in raster
assert "mip" not in radiance and "nearest" not in radiance.lower()
assert "nearest" not in integrator.lower()

ring = modules[modules.index("inline GraphModulePackage glassRingCausticsStarter()"):]
ring = ring[:ring.index("\ninline GraphModulePackage ", 1)]
assert '{"shadeMode", 3}, {"smoothAngleDegrees", 30.0}' in ring
assert '{{"value", 0.15}}' in ring
# renderRingFloor in pathtrace_refraction_tests.cpp keeps the flat-floor fixture and this key direction.
assert '{{"rotationX", -120}}' in ring
assert '{{"intensity", 3}, {"red", 1}, {"green", 0.9}, {"blue", 0.75}' in ring
assert '{0.07, 0.05, 0.19}' in ring
# The floor is a sweep with its side only: no plate edges and no cylinder caps in frame.
assert 'kinds::GeometryCube, 340, 2200' not in ring
assert '{{"radius", 1.6}, {"height", 8}, {"segments", 512}}' in ring
assert '{{"firstIndex", 512}, {"lastIndex", 639}, {"keepSelected", 1}}' in ring
assert '{{"ambientRed", 0}, {"ambientGreen", 0}, {"ambientBlue", 0}' in ring
refraction_tests = (root / "tests/pathtrace_refraction_tests.cpp").read_text()
assert "{ { 0.0f, 0.866f, -0.5f }, 10.0f, 0.03f, 0.043f }" in refraction_tests
assert "clearGlass (1.5f, 0.15f)" in refraction_tests
bsdf = (root / "src/pathtrace/bsdf.cpp").read_text()
cauchy = bsdf[bsdf.index("float materialCauchyB"):bsdf.index("bool materialDispersive")]
assert "0.05f" not in cauchy
assert "kMaximumCauchyB" not in limits  # no upper clamp: any finite nonnegative B admits

# The ring's lowest point touches the floor top at y 0, so the shadow starts at the contact.
assert '{{"translationY", 0.6}}' in ring, "ring centre = hoop 0.5 + tube 0.1, no gap above the floor"

# The sky tint is rgb * intensity. Spelling it out keeps the environment's scale in view.
assert '{"red", 0.5}, {"green", 0.5}, {"blue", 0.5}, {"intensity", 1}' in ring
ring_sky = ring[ring.index('TRANS("Ring Sky or HDR Environment")'):ring.index('TRANS("Ring Camera Position")')]
assert "environmentCameraVisible" not in ring_sky, "judged frames show the world, so the ring keeps the visible default"

# The body colour comes from a many-coloured environment, not from dispersion. Judged frames bind
# the colour studio HDR, and the other studio rows keep art_studio.
acceptance = (root.parent / "scripts/video-plan3-acceptance.py").read_text()
hdr = root.parent / "references/environment-hdri/glass_ring_colour_studio_2k.hdr"
assert hdr.is_file()
assert 'references/environment-hdri/glass_ring_colour_studio_2k.hdr' in acceptance
assert 'if module_id=="visual.glass-ring-caustics": return GLASS_RING_HDRI' in acceptance
generator = (root.parent / "scripts/generate-glass-ring-environment.py").read_text()
assert "art_studio" not in generator and "read_hdr" not in generator, "the ring's environment holds no photograph"
assert hashlib.sha256(hdr.read_bytes()).hexdigest() == "b5ba280628d303658da8e40bdbe3a7e79ad77368682321f2c29b163d96d02fbc"


# The 55 stripes in 11 identical groups were the 12 beads round the ring, and they survived at
# B 0 as white and black blobs, so the rig made them. Their key shadows were the polygon outlines.
assert "Ring Studio" not in ring and "studioAngles" not in ring, "the stripe grid stays out of this starter"
assert "edge(ringScene, 4, compose, 0); edge(floorScene, 4, compose, 1);" in ring

print("glass ring fringe source contract: PASS")
