#!/usr/bin/env python3
"""Pins the OptiX motion GAS that no test can run without the SDK.

A moving scene builds two motion keys from the ingested open and close vertices,
the pipeline admits motion, and every trace carries the ray's time.
"""
from pathlib import Path
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
host = (root / "src/pathtrace/devices/device_optix_host.cpp").read_text()
kernel = (root / "src/pathtrace/devices/device_optix.cu").read_text()
integrator = (root / "src/pathtrace/integrator.cpp").read_text()

assert "pipelineOptions.usesMotionBlur = 1;" in host, "the pipeline traces motion GASes"
assert "view_.motionOpen, error)" in host and "view_.motionClose, error)" in host, "both keys reach the device"
build = host[host.index("const bool moving = ! view_.motionOpen.empty();"):]
build = build[:build.index("optixAccelComputeMemoryUsage")]
assert "input.triangleArray.vertexBuffers = vertices;" in build, "one vertex buffer per motion key"
assert "options.motionOptions.numKeys = 2;" in build
assert "options.motionOptions.timeBegin = 0.0f;" in build and "options.motionOptions.timeEnd = 1.0f;" in build, \
    "the keys span Ray::time"

assert kernel.count("ray.tMin, ray.tMax, ray.time,") == 2, "closest and shadow rays both trace at the ray's time"
assert "hit.time = ray.time;" in kernel, "the hit keeps the time surfaceAt interpolates with"

assert "motion blur draws on the CPU only so far" not in integrator, "a moving scene stays on the device"
assert "motion blur has no device acceleration structure" not in integrator
print("OptiX motion source contract holds")
