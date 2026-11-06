#pragma once

#include "gpu_backend/backend.h"
#include "vertex_program_cpu.h"

#include <memory>
#include <string>
#include <vector>

// Path Trace's pre-BVH hook, for this frame and the one before. Each object's
// vertex program moves a copy of the scene on the host. For motion, the prior
// frame's positions are the program run on the prior frame's scene positions
// with the prior frame's time, spectrum and control bank, as the raster's
// second vertex evaluation does. Without a prior sample the prior frame is
// this frame, so nothing moves.
namespace videohelper::vertexprogramcpu
{
inline Inputs currentInputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& in)
{
    Inputs inputs;
    inputs.timeSeconds = in.timeSeconds;
    inputs.spectrum = in.vertexSpectrum;
    inputs.controls = in.vertexControls;
    return inputs;
}

inline Inputs priorInputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& in)
{
    if (! in.previousMotion) return currentInputs (in);
    Inputs inputs;
    inputs.timeSeconds = in.previousMotion->timeSeconds;
    inputs.spectrum = in.previousMotion->vertexSpectrum;
    // A prior bank of another program reads as this frame's, as the raster's
    // control texture repeats row 0 (vertexcontrol::motionTexels).
    inputs.controls = in.previousMotion->vertexControls.size() == in.vertexControls.size()
        ? in.previousMotion->vertexControls : in.vertexControls;
    return inputs;
}

// Runs every program of the material on scene. False with the program's error.
inline bool displaceScene (HarmonicMIDI::grid::Visual3DScene& scene,
                           const arbitgpu::NativeFixtureSurfaceMaterialProgram& root, const Inputs& inputs,
                           std::string& error)
{
    const auto apply = [&] (const arbitgpu::NativeFixtureSurfaceMaterialProgram& program) {
        return ! program.vertexProgram || displaceObject (scene, program.object, *program.vertexProgram, inputs, error);
    };
    if (! apply (root)) return false;
    for (const auto& program : root.objectPrograms)
        if (program && ! apply (*program)) return false;
    return true;
}

// The prior frame's displaced positions of source, one per scene vertex. The
// prior sample's scene positions replace a prefix of the vertices before the
// programs run; a prior sample of another size keeps this frame's positions.
inline bool priorDisplacedPositions (const HarmonicMIDI::grid::Visual3DScene& source,
                                     const arbitgpu::NativeFixtureSurfaceMaterialProgram& root,
                                     const arbitgpu::NativeFixtureSceneRuntimeInputs& in,
                                     std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>>& out,
                                     std::string& error)
{
    auto prior = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (source);
    const auto& positions = in.previousMotion ? in.previousMotion->scenePositions : nullptr;
    if (positions != nullptr && positions->size() <= prior->vertexCount)
        for (std::size_t i = 0; i < positions->size(); ++i)
            prior->vertices[i].position = (*positions)[i];
    if (! displaceScene (*prior, root, priorInputs (in), error)) return false;
    auto moved = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>();
    moved->reserve (prior->vertexCount);
    for (std::size_t i = 0; i < prior->vertexCount; ++i)
        moved->push_back (prior->vertices[i].position);
    out = std::move (moved);
    return true;
}
} // namespace videohelper::vertexprogramcpu
