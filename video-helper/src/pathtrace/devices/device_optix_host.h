#pragma once

// The OptiX device's render pass, without SDK types, so the fixture engine can
// call it. Exists only in a build with ARBIT_HAVE_OPTIX.

#include "../integrator.h"
#include "kernel_warmup.h"

#include <cstdint>
#include <string>

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
// The driver's cache holds this kernel, so a build is the 0.1 s kind. False when it does not,
// and when nothing can say (no driver, no OptiX, the cache turned off).
bool optixModuleHashPresent();
// The host's go for the background warm-up (kernel_warmup.h), and its state for diagnostics.
void armOptixWarmup();
kernelcompile::WarmupReading readOptixWarmup();
// Stops a running build and returns when its thread has let go. For process exit.
void cancelOptixPipelineBuild();
// The user's Cancel and Start again. Cancel returns at once and is refused once the build is
// done or an export is blocked on it. A preview then waits by name until Start again.
bool stopOptixPipelineBuildByUser();
void startOptixPipelineBuildByUser();

// One frame for the OptiX AI denoiser (Plan 3, Denoising). Every image is width *
// height pixels of three floats, in any row order shared by all of them; flow has two.
struct OptixDenoiseImage
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    const float* color = nullptr;
    const float* albedo = nullptr;
    const float* normal = nullptr;  // camera space, as the temporal model reads it
    float* output = nullptr;
    // The temporal model, with the flow from the previous frame and that frame's
    // output kept on the device. usePrevious false starts a new sequence.
    bool temporal = false;
    bool usePrevious = false;
    const float* flow = nullptr;
};

// Why the OptiX AI denoiser cannot run now, or empty when it can: it needs the
// context the OptiX device made, so a session that never started that device has none.
std::string optixDenoiserProblem();
// Uploads the frame, runs the AOV or temporal model with the albedo and normal
// guides on the shared context and reads the output back. False with error.
bool optixDenoise (const OptixDenoiseImage& image, std::string& error);

} // namespace videohelper::pathtrace::device
