#!/usr/bin/env python3
# Contract for the admission-time program prewarm. The lighthouse water row's cold
# chain (caustic dispatch set plus the volume fixture program) used to start at the
# first frame's render steps and outran the 12-second acceptance window. Both links
# must start when the seam publishes the admitted material, and the fixture link
# must submit into the same in-flight state the first prepare polls.
from pathlib import Path
import sys

root = Path(sys.argv[1])
seam = (root / "src/glb_native_render_seam.cpp").read_text()
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
header = (root / "src/gpu_backend/backend.h").read_text()
errors = []

# The seam prewarms both links with the admitted material, only after the
# duplicate-publication early return, and only for an exact closed-volume glass.
publish = seam[seam.find("bool GlbNativeRenderSeam::publishSurfacePrograms"):
                seam.find("void GlbNativeRenderSeam::clearSurfaceMaterial")]
if not publish:
    errors.append("publishSurfacePrograms region not found")
else:
    for needle in (
        "if (caustic)",
        "arbitgpu::nativeSurfaceForObject",
        "surfacematerial::TransportMode::ClosedVolume",
        "arbitgpu::prepareOpenGlGlassCausticPrograms (rough, rough,",
        "arbitgpu::NativeFixturePrepareUse::Preview, dispatchError",
        "backend_.startFixtureProgramLink (candidate->nativeProgram(),",
        "causticVolumeNeedsSlot (",
    ):
        if needle not in publish:
            errors.append("publish region missing " + needle)
    # Prewarm must sit after the duplicate check so unchanged republishes pay nothing.
    if publish.find("lastSurfaceProgramsIdentity_") > publish.find("backend_.startFixtureProgramLink"):
        errors.append("prewarm must run after the duplicate-publication early return")

# The interface default keeps non-GL backends unchanged, and the OpenGL backend
# submits into the same program cache, in-flight map and counters the prepare uses.
if "virtual void startFixtureProgramLink (" not in header:
    errors.append("backend interface lost the admission-time warm hook")
for needle in (
    "void startFixtureProgramLink (",
    "override",
    "programCache_[glfwGetCurrentContext()]",
    "inFlight.emplace (programCacheKey, std::move (entry))",
    "fixture program warm-submitted",
    "kMaximumInFlightFixturePrograms",
    "if (distinctGrating) return;",
):
    if needle not in backend:
        errors.append("OpenGL warm missing " + needle)

# A caller with no current context must return before the fixture lock: taking
# fixtureMutex() first parks a context-less caller (the model_payload_preview
# RPC reaches here on its own thread) behind a whole export caustic frame and
# then has it return without doing anything. The context is thread-local, so
# the pre-lock check is safe; the post-lock check stays for a context that
# goes away while waiting.
link = backend[backend.find("void startFixtureProgramLink ("):]
pre_context = link.find("if (glfwGetCurrentContext() == nullptr) return;")
lock_at = link.find('lockScope.emplace ("gl:fixture-link-lock");')
if pre_context < 0:
    errors.append("startFixtureProgramLink lost its pre-lock context check")
elif lock_at < 0:
    errors.append("startFixtureProgramLink lost its lock scope anchor")
elif pre_context > lock_at:
    errors.append("startFixtureProgramLink must check the context before taking fixtureMutex")

if errors:
    print("fixture admission warm contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("fixture admission warm contract: PASS")
