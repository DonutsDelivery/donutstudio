// Realtime on an animated imported mesh: the deformation renderer draws Raster
// only, so a Realtime request (Render node renderEngine 2) on a skinned or
// morphed mesh bakes the frame's vertices and draws through the fixture scene
// renderer, as Path Trace does. Before this, a Realtime frame of the Note-Driven
// Particle and Light Rig starter was byte-identical to its Raster frame.
// realtime_animated_mesh_source_contract.py pins the routes and the program
// composition, which no test here can draw.

#include "VisualImportedSceneRenderOperationContract.h"

#include <iostream>
#include <string>

namespace
{

namespace render = visualimportedscenerender;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

render::Request requestFor (render::RenderEngine engine)
{
    render::Request request;
    request.sourceStableId = 81;
    request.renderStableId = 82;
    request.asset.id = "animated-model-asset";
    request.asset.version = 1;
    request.asset.contentSha256 = std::string (64, 'b');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 8192;
    request.sceneIndex = 0;
    request.renderEngine = engine;
    return request;
}

void wireTests()
{
    const auto request = requestFor (render::RenderEngine::Realtime);
    check (render::valid (request), "a Realtime request is valid");
    const auto wire = render::encode (request);
    check (! wire.empty(), "a Realtime request encodes");
    render::Request decoded;
    check (render::decode (wire, decoded) && decoded.renderEngine == render::RenderEngine::Realtime
               && render::sameRequest (request, decoded),
           "the helper decodes Realtime from the wire");
    // Old projects: a Raster request keeps its exact bytes, with no engine envelope.
    const auto raster = render::encode (requestFor (render::RenderEngine::Raster));
    check (! raster.empty() && raster != wire, "Raster keeps its own bytes");
    render::Request rasterDecoded;
    check (render::decode (raster, rasterDecoded) && rasterDecoded.renderEngine == render::RenderEngine::Raster,
           "a Raster request still decodes as Raster");
}

void bakedRouteTests()
{
    check (! render::animatedMeshDrawsBaked (render::RenderEngine::Raster),
           "Raster keeps the deformation renderer");
    check (render::animatedMeshDrawsBaked (render::RenderEngine::PathTrace),
           "Path Trace bakes the animated mesh");
    check (render::animatedMeshDrawsBaked (render::RenderEngine::Realtime),
           "Realtime bakes the animated mesh instead of drawing Raster");
    const auto realtimeRefusal = std::string (render::animatedMeshUnavailable (render::RenderEngine::Realtime));
    const auto pathTraceRefusal = std::string (render::animatedMeshUnavailable (render::RenderEngine::PathTrace));
    check (realtimeRefusal.find ("Realtime") != std::string::npos, "a Realtime refusal names Realtime");
    check (pathTraceRefusal == render::kPathTraceAnimatedMeshUnavailable, "the Path Trace refusal is unchanged");
}

} // namespace

int main()
{
    wireTests();
    bakedRouteTests();
    std::cout << checks - failures << '/' << checks << " realtime animated mesh checks passed\n";
    return failures == 0 ? 0 : 1;
}
