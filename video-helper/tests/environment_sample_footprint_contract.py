#!/usr/bin/env python3
"""Raster reads an HDR sky through mip levels sized to each quadrature sample.

On 31 October 2026 visual.material.clearcoat-plastic showed petals meeting at the sphere's
centre and visual.material.polished-metal showed lumpy blobs. Neither starter authors a bump
or noise. 64 GGX and 32 diffuse samples in fixed directions each read level zero, so each drew
a sharp copy of the image, around a tangent frame that turns at n=+Z.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
text = (ROOT / "video-helper/src/gpu_backend/fixture_environment_image.h").read_text(encoding="utf-8")
sky = (ROOT / "video-helper/src/gpu_backend/fixture_analytic_sky.h").read_text(encoding="utf-8")
for anchor in ("result+=analyticSkyRadiance(l,zenith,horizon,ground,sun,sunDirection)\n                *fresnel*g1l/64.0;",
               "result+=analyticSkyRadiance(l,zenith,horizon,ground,sun,sunDirection)\n            *(1.0-fresnel)"):
    if anchor not in sky:
        print("FAIL: the analytic sky changed under a substitution anchor")
        sys.exit(1)
for needle in ("environmentImageFiltered(l,tint,control,4.0*nv/(64.0*", "environmentImageFiltered(l,tint,control,3.141592653589793/(32.0*",
               "buildMips(candidate.texels"):
    if needle not in text:
        print(f"FAIL: {needle} is missing")
        sys.exit(1)
for name, needle in (("backend_opengl.cpp", "GL_TEXTURE_MAX_LEVEL,GLint(environment.mips.size())"),
                     ("backend_sokol.mm", "desc.num_mipmaps=int(environment.mips.size())+1")):
    if needle not in (ROOT / "video-helper/src/gpu_backend" / name).read_text(encoding="utf-8"):
        print(f"FAIL: {name} does not upload the mip levels")
        sys.exit(1)
print("environment sample footprint: PASS")
