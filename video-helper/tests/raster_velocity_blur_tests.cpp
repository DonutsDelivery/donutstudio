// Raster and Realtime motion blur: a request on either engine carries the
// Render node's shutter and position on the V38 wire around its exact bytes
// without motion blur, and the shutter interval the OpenGL velocity blur
// gathers over follows the position. realtime_velocity_blur_source_contract.py
// pins the OpenGL pass, which no test here can draw.

#include "VisualImportedSceneRenderOperationContract.h"

#include <cmath>
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
    request.sourceStableId = 71;
    request.renderStableId = 72;
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string (64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    request.renderEngine = engine;
    return request;
}

void engineWireTests (render::RenderEngine engine, const std::string& name)
{
    const auto request = requestFor (engine);
    check (render::valid (request), name + " without motion blur is valid");
    const auto plain = render::encode (request);
    check (! plain.empty(), name + " without motion blur encodes");

    auto blurred = request;
    blurred.motionBlurShutter = 250;
    blurred.motionBlurPosition = render::MotionBlurPosition::Start;
    check (render::valid (blurred), name + " carries motion blur");
    const auto wire = render::encode (blurred);
    check (wire == std::string (render::kCompiledOperationHeaderV38) + "\n250 0 " + std::to_string (plain.size())
                       + "\n" + plain,
           name + " wraps its exact bytes in the shutter envelope");
    render::Request decoded;
    check (render::decode (wire, decoded) && decoded.renderEngine == engine && decoded.motionBlurShutter == 250
               && decoded.motionBlurPosition == render::MotionBlurPosition::Start
               && render::sameRequest (blurred, decoded),
           name + " round-trips its shutter");
}

void intervalTests()
{
    const auto start = render::shutterInterval (250, render::MotionBlurPosition::Start);
    check (start.active() && start.open == 0.0f && std::fabs (start.close - 0.25f) < 1.0e-6f,
           "a start shutter gathers from now to a quarter frame on");
    const auto off = render::shutterInterval (0, render::MotionBlurPosition::Centre);
    check (! off.active(), "no shutter draws no blur");
}

} // namespace

int main()
{
    engineWireTests (render::RenderEngine::Raster, "Raster");
    engineWireTests (render::RenderEngine::Realtime, "Realtime");
    intervalTests();
    std::cout << checks - failures << '/' << checks << " raster velocity blur checks passed\n";
    return failures == 0 ? 0 : 1;
}
