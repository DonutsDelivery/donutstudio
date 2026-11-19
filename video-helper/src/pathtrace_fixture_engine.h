#pragma once

// The pieces of the Path Trace fixture engine (pathtrace_fixture_engine.cpp)
// that tests check directly: the input refusals, the Surface material mapping
// and the engine factory the helper registers at static initialisation.

#include "fixture_scene_renderer.h"
#include "gpu_backend/cpu_scene_frame.h"
#include "pathtrace/bsdf.h"
#include "pathtrace/engine.h"

#include <memory>
#include <string>

namespace videorender::fixture3d::pathtracefixture
{

// Refuses runtime inputs the raster refuses too (validFixtureRuntimeInputs).
// Score frames, note instancing, glass caustics and an admitted diffraction
// material render in the tracer itself, and frame textures in the upload's GPU pass.
bool admitInputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs, std::string& error);
// Whether a draw shows an Image view, exports data passes or runs a pass
// composite, so its frame carries the render outputs.
bool readsRenderOutputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs) noexcept;

// Sets the setup's per-note instancing from the frame's score and mapping, as
// the raster draws it: every object once per note at objectMatrix * (position *
// meshScale + note position), tinted by mix (appearanceLow, appearanceHigh,
// value), and under motion blur from each note's prior placement. Inputs
// without a mapping clear it. Refuses what the raster refuses.
bool bindNoteInstances (const HarmonicMIDI::grid::Visual3DScene& scene,
                        const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs, bool motionBlur,
                        videohelper::pathtrace::EngineSceneSetup& setup, std::string& error);

// Maps one admitted Surface program onto the tracer's material. A closed
// volume or thin interface refracts with transmissionIorClearcoat.x and the
// d-line index transmissionIorClearcoat.y. A thin surface does not refract:
// like the raster, its transmission t lowers coverage to 1 - t (1 - F0) with
// F0 from the d-line index, and it blends. A diffraction program maps to the
// object's scene material, which shows where its foil's mask leaves the grating
// out. A frame-texture base colour maps its factor here; the engine applies the frame.
bool mapSurfaceProgram (const arbitgpu::NativeFixtureSurfaceMaterialProgram& program,
                        const HarmonicMIDI::grid::SceneMaterialRecord* record, float timeSeconds,
                        videohelper::pathtrace::Material& material, std::string& error);

// A new Path Trace engine, or null when it cannot be allocated.
std::shared_ptr<SceneRenderEngine> makeEngine();

// The same engine with another frame upload, so a test can read the exact bytes
// the engine would hand the compositor.
using CpuFrameUpload = std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> (*) (
    const std::string& backend, const arbitgpu::CpuSceneImage& image, std::string& error);
std::shared_ptr<SceneRenderEngine> makeEngineWithUpload (CpuFrameUpload upload);

// Starts the kernel build in the background where OptiX is this machine's device, so a
// first render after an update finds it done or joins it.
void prepareDeviceAtStartup();
// The host's go for the background kernel warm-up, sent once its window has painted and the
// app is idle. Only an OptiX device warms up, and only when the driver's cache lacks the
// kernel. The warm-up then waits for a quiet machine (kernel_warmup.h).
void armKernelWarmup();
// What a live test reads: the warm-up's state beside the kernel's own notice flags.
struct KernelWarmupInfo
{
    bool applies = false;            // the device is OptiX and this build has its kernel
    std::string state;               // idle, running, paused, done or failed
    std::string reason;
    bool armed = false;
    bool moduleHashPresent = false;  // the driver's cache holds this kernel
    bool joined = false;             // the viewport or an export joined the running build
    bool contextReleased = false;    // done with nobody using it: the GPU context was freed
    bool busy = false;               // playing, exporting or rendering
    bool lowPriority = false;
    int elapsedSeconds = 0;
    std::string error;               // of a failed warm-up, which the user never sees
};
KernelWarmupInfo kernelWarmupInfo();
// The machine's Path Trace device setting from Preferences: "auto", "optix", "cuda",
// "hip", "metal", "oneapi" or "cpu". ARBIT_PATHTRACE_DEVICE still overrides it. A
// change moves every engine to the new selection on its next draw and starts that
// device's kernel build. Returns the selection's error, empty when it can render.
std::string setDevicePreference (const std::string& value);
std::string devicePreference();
// The notice's Cancel and Start again. Cancel says whether it applied.
bool cancelDeviceBuild();
void startDeviceBuildAgain();
// Ends a build still running, before the process exits. Measured about 2 s on a loaded host.
void stopDeviceBuildAtExit();

} // namespace videorender::fixture3d::pathtracefixture
