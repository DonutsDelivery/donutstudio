#pragma once

// The OptiX device's render pass, without SDK types, so the fixture engine can
// call it. Exists only in a build with ARBIT_HAVE_OPTIX.

#include "../integrator.h"

namespace videohelper::pathtrace::device
{

// A device pass for PathTracer::setDevicePass. All passes share one context and
// pipeline, and each keeps its own uploaded scene and film; a change of sceneBuild
// uploads the scene again. Any failure
// fills DevicePass::error. A device asked for by name stops the frame by name, and
// Auto moves to the CPU. A lost context fails every later pass (device_registry.h).
PathTracer::DevicePassFunction makeOptixDevicePass (std::function<std::uint64_t()> sceneBuild);

// The kernel builds on its own thread, once for the process. This starts it if needed and
// says whether a pass can run now. A failed build says true, so the pass reports it.
bool optixPipelineReady();
void startOptixPipelineBuild();
// Stops a running build and returns when its thread has let go. For process exit.
void cancelOptixPipelineBuild();
// The user's Cancel and Start again. Cancel returns at once and is refused once the build is
// done or an export is blocked on it. A preview then waits by name until Start again.
bool stopOptixPipelineBuildByUser();
void startOptixPipelineBuildByUser();

} // namespace videohelper::pathtrace::device
