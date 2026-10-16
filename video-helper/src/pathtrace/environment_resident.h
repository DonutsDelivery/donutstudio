#pragma once

// Decoded environment image pixels for the CPU path tracer, taken from the
// helper's resident store (environment_image_payload_transport.h) instead of
// a second decode. The returned texels share ownership with the resident, so
// the store cannot evict it while a RenderScene holds them, the same lease a
// native preparation takes (fixture_environment_image.h prepare()).
//
// Pass residentEnvironmentPixels as PrepareOptions::environmentImageSource.

#include "environment_map.h"

#include <memory>

namespace videohelper::environmentimage
{
struct Resident;
}

namespace videohelper::pathtrace
{

// Empty pixels when the exact asset is not resident.
EnvironmentPixels residentEnvironmentPixels (const HarmonicMIDI::grid::SceneEnvironmentImage& image);
// Empty pixels for a null resident.
EnvironmentPixels residentEnvironmentPixels (const std::shared_ptr<const environmentimage::Resident>& resident);

} // namespace videohelper::pathtrace
