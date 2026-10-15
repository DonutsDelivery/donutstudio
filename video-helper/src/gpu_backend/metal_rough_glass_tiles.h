#pragma once

#include "fixture_rough_glass_tiles.h"

namespace arbitgpu
{
// Private bounded submission seam. Normal render selects this loop when the
// rough canvas exceeds one job; native parity can also submit a small canvas.
NativeFixtureSceneSubmission renderMetalRoughGlassTiledFrame(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const NativeFixtureSceneResources>& resources,
    std::uint32_t width,std::uint32_t height,NativeFixtureSceneRuntimeInputs runtime,
    bool reverseOrder = false);
bool readMetalRoughGlassTileAggregate(const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    std::shared_ptr<const roughglasstiles::Schedule>& schedule,
    roughglasstiles::Aggregate& aggregate,std::string& error);
}
