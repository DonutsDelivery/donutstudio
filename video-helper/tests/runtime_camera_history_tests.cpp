// The Render node's camera parameters evaluated one frame earlier: the reader,
// the previous-frame evaluator and the Motion uniforms built from that camera.

#include "runtime_camera_parameters.h"
#include "gpu_backend/backend.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>

namespace
{
int failures = 0;

void expect(bool condition, const char* what)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

bool approx(float a, float b) { return std::abs(a - b) < 1.0e-4f; }

struct FakeLayer
{
    std::function<bool(double, FakeLayer&, std::map<std::string, double>&, std::string&)> materialInputsAt;
};

struct LayerWithoutHistory {};

HarmonicMIDI::grid::SceneCameraRecord authoredCamera()
{
    HarmonicMIDI::grid::SceneCameraRecord camera;
    camera.id.value = 1;
    camera.transform.translation = { 0.0f, 1.0f, 5.0f };
    camera.verticalFovRadians = 0.8f;
    camera.nearPlane = 0.1f;
    camera.farPlane = 100.0f;
    return camera;
}

void readerTests()
{
    namespace rc = videohelper::runtimecamera;
    const auto prefix = rc::prefix(7);
    expect(prefix == "visual7/", "prefix names the Render node");
    std::map<std::string, double> values { { "visual7/cameraX", 2.5 }, { "visual7/cameraRotateY", 720.0 },
        { "visual7/cameraFovDegrees", 1.0e-9 }, { "visual3/cameraZ", 9.0 } };
    expect(rc::containsAny(values, prefix), "camera keys under the prefix are found");
    expect(!rc::containsAny(values, rc::prefix(8)), "another Render node has no camera keys");
    const auto camera = rc::read(values, prefix);
    expect(approx(camera.translationOffset[0], 2.5f) && approx(camera.translationOffset[2], 0.0f),
        "translation reads its own prefix only");
    expect(approx(camera.rotationDegrees[1], 360.0f), "rotation clamps to one turn");
    expect(camera.verticalFovDegrees == HarmonicMIDI::grid::kMinimumCameraVerticalFovDegrees,
        "a positive lens under the minimum rounds up");
    values["visual7/cameraFovDegrees"] = -4.0;
    expect(rc::read(values, prefix).verticalFovDegrees == 0.0f, "a negative lens keeps the authored one");
    values["visual7/cameraOrthographicHeight"] = 4.0;
    expect(rc::read(values, prefix).verticalFovDegrees == -4.0f,
        "an orthographic height travels as a negative field of view");
    expect(rc::containsAny({ { "visual7/cameraOrthographicHeight", 2.0 } }, prefix),
        "an orthographic height alone is a camera parameter");

    double at = 0.0;
    expect(rc::previousFrameSeconds(2.0, 25.0, at) && approx(static_cast<float>(at), 1.96f), "one frame earlier");
    expect(!rc::previousFrameSeconds(2.0, 0.0, at), "no frame rate, no previous frame");
}

void evaluatorTests()
{
    namespace rc = videohelper::runtimecamera;
    const auto prefix = rc::prefix(4);
    FakeLayer layer;
    double asked = -1.0;
    layer.materialInputsAt = [&](double at, FakeLayer&, std::map<std::string, double>& values, std::string&)
    {
        asked = at;
        values = { { prefix + "cameraX", at * 10.0 } };
        return true;
    };
    const auto earlier = rc::previous(layer, prefix, 1.0, 10.0);
    expect(earlier.has_value() && approx(static_cast<float>(asked), 0.9f), "the evaluator runs one frame earlier");
    expect(earlier && approx(earlier->translationOffset[0], 9.0f), "the earlier camera comes from the evaluator");

    layer.materialInputsAt = [](double, FakeLayer&, std::map<std::string, double>& values, std::string&)
    { values.clear(); return true; };
    expect(!rc::previous(layer, prefix, 1.0, 10.0), "no camera parameter means no history");
    layer.materialInputsAt = [](double, FakeLayer&, std::map<std::string, double>&, std::string& failure)
    { failure = "live routing"; return false; };
    expect(!rc::previous(layer, prefix, 1.0, 10.0), "a refusing evaluator means no history");
    layer.materialInputsAt = {};
    expect(!rc::previous(layer, prefix, 1.0, 10.0), "a layer without an evaluator has no history");
    expect(!rc::previous(LayerWithoutHistory {}, prefix, 1.0, 10.0), "a layer type without the evaluator compiles");
}

void motionTests()
{
    const auto camera = authoredCamera();
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.cameraTranslationOffset = { 1.0f, 0.0f, 0.0f };

    auto still = arbitgpu::sceneMotionUniforms(inputs, camera, 64, 32);
    expect(still.projection[3] == 0.0f, "no predecessor and no history is zero motion");

    inputs.previousCamera = arbitgpu::NativeRuntimeCameraHistory { { 0.5f, 0.0f, 0.0f }, {}, 0.0f };
    expect(arbitgpu::hasPriorMotionSample(inputs), "a camera history is a prior sample");
    const auto moved = arbitgpu::sceneMotionUniforms(inputs, camera, 64, 32);
    expect(moved.projection[3] == 1.0f, "the camera history turns Motion on");
    expect(approx(moved.cameraTranslation[0], 0.5f) && approx(moved.cameraTranslation[1], 1.0f)
        && approx(moved.cameraTranslation[2], 5.0f), "the prior camera is the authored one at the earlier offset");
    expect(moved.rotation[3] == 1.0f && moved.translationScale[3] == 1.0f, "objects keep identity motion");

    inputs.previousCamera->verticalFovDegrees = 60.0f;
    const auto zoomed = arbitgpu::sceneMotionUniforms(inputs, camera, 64, 32);
    expect(approx(zoomed.projection[0], std::tan(60.0f * 3.14159265f / 360.0f)), "the earlier lens is the prior projection");

    arbitgpu::NativeSceneMotionSample predecessor = arbitgpu::sceneMotionSample(inputs);
    predecessor.cameraTranslationOffset = { -2.0f, 0.0f, 0.0f };
    inputs.previousMotion = predecessor;
    const auto adjacent = arbitgpu::sceneMotionUniforms(inputs, camera, 64, 32);
    expect(approx(adjacent.cameraTranslation[0], -2.0f), "the adjacent predecessor outranks the history");
}
} // namespace

int main()
{
    readerTests();
    evaluatorTests();
    motionTests();
    if (failures != 0) return 1;
    std::cout << "runtime camera history tests passed\n";
    return 0;
}
