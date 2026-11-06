#pragma once

// The pieces of the Path Trace fixture engine (pathtrace_fixture_engine.cpp)
// that tests check directly: the input refusals, the Surface material mapping
// and the engine factory the helper registers at static initialisation.

#include "fixture_scene_renderer.h"
#include "gpu_backend/cpu_scene_frame.h"
#include "pathtrace/bsdf.h"

#include <memory>
#include <string>

namespace videorender::fixture3d::pathtracefixture
{

// Refuses runtime inputs whose raster passes the tracer does not have
// (AOVs and raw exports, pass composites, deformed or composed frames). Glass caustics and an admitted diffraction material render
// in the tracer itself, and frame textures in the upload's GPU pass.
bool admitInputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs, std::string& error);
// Whether a draw shows an Image view, exports data passes or runs a pass
// composite, so its frame carries the render outputs.
bool readsRenderOutputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs) noexcept;

// The scene with its one mesh object instanced once per note the frame sends through
// the mapping. Note appearance and several objects are refused by name.
bool noteInstancedScene (const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
                         const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs,
                         std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& out, std::string& error);

// Maps one admitted Surface program onto the tracer's material. A closed
// volume or thin interface refracts with transmissionIorClearcoat.x and the
// d-line index transmissionIorClearcoat.y. A thin surface does not refract:
// like the raster, its transmission t lowers coverage to 1 - t (1 - F0) with
// F0 from the d-line index, and it blends. Diffraction programs are refused. A
// frame-texture base colour maps its factor here; the engine applies the frame.
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
