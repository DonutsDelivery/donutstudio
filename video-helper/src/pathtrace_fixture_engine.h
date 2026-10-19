#pragma once

// The pieces of the Path Trace fixture engine (pathtrace_fixture_engine.cpp)
// that tests check directly: the input refusals, the Surface material mapping
// and the engine factory the helper registers at static initialisation.

#include "fixture_scene_renderer.h"
#include "pathtrace/bsdf.h"

#include <memory>
#include <string>

namespace videorender::fixture3d::pathtracefixture
{

// Refuses runtime inputs whose raster passes the tracer does not have
// (AOVs and raw exports, pass composites, diffraction, glass caustics, note
// instancing, deformed or composed frames, frame textures).
bool admitInputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs, std::string& error);

// Maps one admitted Surface program onto the tracer's material. A closed
// volume or thin interface refracts with transmissionIorClearcoat.x and the
// d-line index transmissionIorClearcoat.y. A thin surface does not refract:
// like the raster, its transmission t lowers coverage to 1 - t (1 - F0) with
// F0 from the d-line index, and it blends. Diffraction programs and frame
// textures are refused.
bool mapSurfaceProgram (const arbitgpu::NativeFixtureSurfaceMaterialProgram& program,
                        const HarmonicMIDI::grid::SceneMaterialRecord* record, float timeSeconds,
                        videohelper::pathtrace::Material& material, std::string& error);

// A new Path Trace engine, or null when it cannot be allocated.
std::shared_ptr<SceneRenderEngine> makeEngine();

} // namespace videorender::fixture3d::pathtracefixture
