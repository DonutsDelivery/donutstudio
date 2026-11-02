#!/usr/bin/env python3
"""Pins the OptiX device wiring that no test can run without the SDK."""
from pathlib import Path
import re
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
cmake = (root / "CMakeLists.txt").read_text()

start = cmake.index('include(cmake/OptiXDiscovery.cmake)')
block = cmake[start:cmake.index('if(GLFW3_FOUND AND OpenGL_FOUND)', start)]

# A machine without the SDK configures and builds.
assert "FATAL_ERROR" not in block, "a missing SDK or compiler never stops configure"
assert re.search(r"find_package\(CUDAToolkit REQUIRED\)", block) is None, "no REQUIRED toolkit"
assert "find_program(ARBIT_NVCC" in block and "if(ARBIT_NVCC)" in block, "probe the compiler first"
assert "--expt-relaxed-constexpr" in block, "device_optix.cu refuses a build without it"
assert re.search(r'message\(STATUS "OptiX device', block), "configure says what it found"
# The SDK is compiled in only behind discovery, and the kernel only with it.
assert "if(ARBIT_OPTIX_SDK_DIR)" in block
assert "device_optix_host.cpp" in block and "ARBIT_HAVE_OPTIX=1" in block
assert 'option(ARBIT_OPTIX_KERNEL_ENABLED "Compile the OptiX integrator kernel when the SDK and nvcc are found" ON)' in block
assert block.count("src/pathtrace/devices/device_optix.cu\" -o") == 1
assert block.count("arbit_add_optix_device(") == 2, "the helper and the route tests"
# Outside the block nothing names the SDK sources.
assert "device_optix_host.cpp" not in cmake.replace(block, "")

# The driver is only reached at run time.
registry = (root / "src/pathtrace/devices/device_registry.cpp").read_text()
assert "#include <optix" not in registry, "the registry builds without the SDK"
assert "optixQueryFunctionTable" in registry and "libnvoptix.so.1" in registry and "nvoptix.dll" in registry
assert "#if defined(ARBIT_HAVE_OPTIX_KERNEL)\n    environment.optixKernelBuilt = true;" in registry, "only a build with the kernel names it"

# A named device that fails stops the frame by name, and the replies say which device ran.
engine = (root / "src/pathtrace_fixture_engine.cpp").read_text()
render = engine[engine.index("bool render (const SceneEngineDraw& draw"):]
assert render.index("pathTraceDeviceSelection()") < render.index("admitInputs"), "the device check comes first"
assert "fail (error, selection.error)" in render, "the error reaches the user"
assert 'std::getenv ("ARBIT_PATHTRACE_DEVICE")' in engine
reply = (root / "src/path_trace_stats_reply.h").read_text()
assert '"device", stats->device' in reply and '"deviceRequested", stats->deviceRequested' in reply

# OptiX renders only through its own pass, and a failed pass stops the frame.
assert "makeOptixDevicePass" in engine and "deviceFailed" in (root / "src/pathtrace/engine.cpp").read_text()
# Under Auto a failed pass moves the frame to the CPU, by name in the log and in the stats.
assert 'std::strcmp (selection.requested, "Auto") == 0' in render and "[pt] device=CPU requested=Auto: %s" in render
assert "engine_.setDevicePass ({})" in render and "fellBackToCpu_ || sceneOnCpu_ ? pt::device::DeviceKind::Cpu" in engine
# The Render panel names the device and the reason, from both ways onto the CPU under Auto.
assert "autoPassedOverReason (selection)" in render and "deviceReason_ = std::make_shared<const std::string> (" in render
assert "stats_.deviceReason = deviceReason_" in engine and '"deviceReason"' in reply
# The fallback starts the CPU on an empty film, and a lost context stays lost for the process.
assert render.index("engine_.setDevicePass ({})") < render.index("engine_.restartPreview()") < render.index("rendered = renderOnDevice();\n        }")
assert "if (deviceLost())" in (root / "src/pathtrace/devices/device_optix_host.cpp").read_text()
assert "Restart DonutStudio to try the GPU again" in (root / "src/pathtrace/devices/device_registry.cpp").read_text()

# A failed setup leaves nothing. The pipeline is published once, after the last step,
# and each object made has its release. tests/pathtrace_device_failure_tests.cpp covers the unwind.
optix_host = (root / "src/pathtrace/devices/device_optix_host.cpp").read_text()
build = optix_host[optix_host.index("bool buildPipeline (SetupUnwind& undo"):optix_host.index("bool uploadBytes (const void* data")]
assert build.count("pipeline_") == 1 and build.rindex("uploadTables (tables, error)") < build.index("pipeline_ = pipeline;")
assert "if (built || deviceLost())" in optix_host
for made, released in (("cuCtxCreate (&cudaContext_", "cuCtxDestroy (cudaContext_)"),
                       ("optixDeviceContextCreate (cudaContext_", "optixDeviceContextDestroy (context_)"),
                       ("optixProgramGroupCreate (context_", "optixProgramGroupDestroy (group)"),
                       ("optixPipelineCreate (context_", "optixPipelineDestroy (pipeline)")):
    assert build.index(made) < build.index(released), made
# A failed task still leaves a module, so its release is registered before the compile.
assert "optixModuleCreateWithTasks (context_" in optix_host and build.index("optixModuleDestroy (module_)") < build.index("compileModule (")
assert build.index("cuCtxDestroy") < build.index("cuMemFree (address)") < build.index("optixDeviceContextDestroy"), "memory goes before its context"
assert "optixInAuto = false" not in registry, "Auto picks OptiX"

# A failed pass adds nothing to the film: only readBack writes it, after its last copy
# from the device. tests/pathtrace_device_failure_tests.cpp covers what happens next.
host = (root / "src/pathtrace/devices/device_optix_host.cpp").read_text()
read_back = host[host.index("bool readBack (PathTracer::DevicePass& pass"):host.index("// One context and pipeline for the process.")]
assert read_back.rindex("cuMemcpyDtoH") < read_back.index("pass.film."), "every copy comes before the first film write"
assert "pass.film." not in host.replace(read_back, "").replace("pass.film.width()", "").replace("pass.film.height()", "")

# An Auto failure outlives the engines: each engine asks the latch before it makes a
# device pass, and each draw reports its Render node's engine so a switch releases it.
engine_source = (root / "src/pathtrace_fixture_engine.cpp").read_text()
assert engine_source.index("autoDeviceLatch().failure()") < engine_source.index("pt::device::makeOptixDevicePass"), "the latch comes before the device pass"
assert "autoDeviceLatch().fail (deviceReason_)" in engine_source
assert "autoDeviceLatch().noteEngine(request.clipId, request.renderStableId" in (root / "src/imported_scene_payload_execution.cpp").read_text()

# The shim implements the whole device interface, and honours ignoreTriangle like the BVH.
shim = (root / "src/pathtrace/devices/device_optix.cu").read_text()
for member in ("using Accel", "using Film", "using Sampler", "makeSampler", "intersect", "occluded",
               "filmAdd", "filmSplat", "filmCountLightPath"):
    assert member in shim, member
assert "ray.ignoreTriangle" in shim and shim.count("__anyhit__") == 2 and "optixIgnoreIntersection" in shim

# The denoiser stays OIDN, and the device code has no size cap.
assert "optixDenoiser" not in shim and "optixDenoiser" not in registry
assert not re.search(r"\bkMax[A-Z]", shim + registry), "no device buffer caps"
print("pathtrace optix device source contract passed")
