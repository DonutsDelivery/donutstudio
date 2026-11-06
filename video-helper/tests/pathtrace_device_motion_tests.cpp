// A moving scene stays on the named device. A fake DevicePass stands in for
// OptiX, so no GPU or SDK is needed: the tracer hands it moving geometry and a
// moving camera instead of skipping to the CPU or refusing the frame, and the
// scene it receives carries the open and close vertices the OptiX motion GAS is
// built from. pathtrace_optix_motion_source_contract.py pins the OptiX side.

#include "pathtrace/integrator.h"

#include <iostream>
#include <string>

namespace
{

namespace pt = videohelper::pathtrace;
using pt::Vec3;

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

struct FakeDevice
{
    int calls = 0;
    bool sawMotionVertices = false;
    bool sawMovingCamera = false;

    pt::PathTracer::DevicePassFunction function()
    {
        return [this] (pt::PathTracer::DevicePass& pass) {
            ++calls;
            sawMotionVertices = ! pass.scene.geometry.motionOpen.empty()
                                && pass.scene.geometry.motionClose.size() == pass.scene.geometry.motionOpen.size();
            sawMovingCamera = pass.camera.moving;
            const std::uint32_t samples = pass.samples;
            for (std::uint32_t y = 0; y < pass.film.height(); ++y)
                for (std::uint32_t x = 0; x < pass.film.width(); ++x)
                    pass.film.addSampleSum (x, y, { 0.0f, 0.0f, 0.0f }, samples);
            pass.stats.samplesAdded += static_cast<std::uint64_t> (samples) * pass.film.width() * pass.film.height();
            return true;
        };
    }
};

pt::RenderScene movingTriangle()
{
    pt::RenderScene scene;
    scene.geometry.positions = { { -0.5f, -0.5f, -2.0f }, { 0.5f, -0.5f, -2.0f }, { 0.0f, 0.5f, -2.0f } };
    scene.geometry.triangles.resize (1);
    scene.geometry.motionOpen = scene.geometry.positions;
    scene.geometry.motionClose = scene.geometry.positions;
    for (auto& p : scene.geometry.motionClose)
        p.x += 1.0f;
    scene.commit();
    return scene;
}

pt::PinholeCamera smallCamera()
{
    pt::PinholeCamera camera;
    camera.width = 2;
    camera.height = 2;
    return camera;
}

void movingGeometryRunsOnTheDevice (bool fallbackAllowed)
{
    const pt::RenderScene scene = movingTriangle();
    check (scene.bvh.hasMotion(), "the fixture scene moves");
    const pt::CancellationToken cancel;
    pt::PathTracer tracer (1);
    tracer.setDeviceSceneFallback (fallbackAllowed);
    FakeDevice device;
    tracer.setDevicePass (device.function());
    const auto stats = tracer.renderProgressive (scene, smallCamera(), 2, 2, 2, 1, cancel);
    const std::string mode = fallbackAllowed ? " (Auto)" : " (named device)";
    check (device.calls == 1, "a moving scene renders on the device" + mode);
    check (stats.deviceSkipped == nullptr, "a moving scene is not sent to the CPU" + mode);
    check (! stats.deviceFailed && tracer.deviceFailure().empty(), "a moving scene does not fail the device" + mode);
    check (device.sawMotionVertices, "the device receives the open and close vertices" + mode);
}

void movingCameraRunsOnTheDevice()
{
    pt::RenderScene scene;
    scene.geometry.positions = { { -0.5f, -0.5f, -2.0f }, { 0.5f, -0.5f, -2.0f }, { 0.0f, 0.5f, -2.0f } };
    scene.geometry.triangles.resize (1);
    scene.commit();
    pt::PinholeCamera camera = smallCamera();
    camera.moving = true;
    camera.closePosition = { 1.0f, 0.0f, 0.0f };
    const pt::CancellationToken cancel;
    pt::PathTracer tracer (1);
    FakeDevice device;
    tracer.setDevicePass (device.function());
    const auto stats = tracer.renderProgressive (scene, camera, 2, 2, 2, 1, cancel);
    check (device.calls == 1 && stats.deviceSkipped == nullptr && ! stats.deviceFailed,
           "a moving camera renders on the device");
    check (device.sawMovingCamera, "the device receives the moving camera");
}

}  // namespace

int main()
{
    movingGeometryRunsOnTheDevice (false);
    movingGeometryRunsOnTheDevice (true);
    movingCameraRunsOnTheDevice();
    std::cout << (checks - failures) << "/" << checks << " device motion checks passed\n";
    return failures == 0 ? 0 : 1;
}
