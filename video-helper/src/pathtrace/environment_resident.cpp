#include "environment_resident.h"

#include "../environment_image_payload_transport.h"

namespace videohelper::pathtrace
{

EnvironmentPixels residentEnvironmentPixels (const std::shared_ptr<const environmentimage::Resident>& resident)
{
    EnvironmentPixels pixels;
    if (resident == nullptr || resident->payload == nullptr)
        return pixels;
    pixels.asset = resident->payload->key();
    pixels.width = resident->image.metadata.width;
    pixels.height = resident->image.metadata.height;
    // Aliasing pointer: the texels live inside the resident and keep it alive.
    pixels.texels = std::shared_ptr<const EnvironmentTexels> (resident, &resident->image.pixels);
    if (! pixels.valid())
        return {};
    return pixels;
}

EnvironmentPixels residentEnvironmentPixels (const HarmonicMIDI::grid::SceneEnvironmentImage& image)
{
    if (image.version == 0)
        return {};
    return residentEnvironmentPixels (environmentimage::store().resolve (image.asset));
}

} // namespace videohelper::pathtrace
