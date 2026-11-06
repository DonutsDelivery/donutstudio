// Live lens modulation on Render 3D: runtimeCameraRecord applies the focus
// distance, f-stop, blade rotation and aperture aspect a route sends, but only
// to a camera whose Depth of field is on. Zero keeps the authored lens.
#include "gpu_backend/backend.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool approx(float a, float b) { return std::abs(a - b) <= 1.0e-5f; }
} // namespace

int main()
{
    using namespace arbitgpu;
    HarmonicMIDI::grid::SceneCameraRecord lens;
    lens.fStop = 2.8f;
    lens.focusDistance = 5.0f;
    lens.apertureBlades = 6;
    lens.bladeRotationRadians = 0.25f;
    lens.apertureRatio = 1.0f;
    const std::array<float, 3> none {};

    const auto authored = runtimeCameraRecord(lens, none, none, 0.0f);
    check(HarmonicMIDI::grid::sameCameraLens(authored, lens), "zero lens values keep the authored lens");

    const auto live = runtimeCameraRecord(lens, none, none, 0.0f, { 2.0f, 8.0f, 90.0f, 1.5f });
    check(approx(live.focusDistance, 2.0f), "a live focus distance replaces the authored one");
    check(approx(live.fStop, 8.0f), "a live f-stop replaces the authored one");
    check(approx(live.bladeRotationRadians, 0.25f + 1.5707963f), "blade rotation adds degrees");
    check(approx(live.apertureRatio, 1.5f), "a live aperture aspect replaces the authored one");
    check(live.apertureBlades == 6, "the blade count stays authored");
    check(HarmonicMIDI::grid::validSceneCameraLens(live), "the modulated lens stays valid");
    // A smaller aperture (higher f-stop) shrinks the blur disc at the same depth.
    check(HarmonicMIDI::grid::sceneCameraCircleOfConfusion(live, 10.0f)
              < HarmonicMIDI::grid::sceneCameraCircleOfConfusion(
                  runtimeCameraRecord(lens, none, none, 0.0f, { 2.0f, 0.0f, 0.0f, 0.0f }), 10.0f),
          "f-stop modulation changes the blur");
    check(approx(HarmonicMIDI::grid::sceneCameraCircleOfConfusion(live, 2.0f), 0.0f),
          "the live focus plane is sharp");

    HarmonicMIDI::grid::SceneCameraRecord pinhole;
    const auto still = runtimeCameraRecord(pinhole, none, none, 0.0f, { 2.0f, 8.0f, 90.0f, 1.5f });
    check(!HarmonicMIDI::grid::sceneCameraHasLens(still)
              && HarmonicMIDI::grid::sameCameraLens(still, pinhole),
          "a camera with Depth of field off stays a pinhole");

    check(validRuntimeCameraLens({}), "all-zero lens inputs are valid");
    check(validRuntimeCameraLens({ 0.001f, 0.1f, -360.0f, 100.0f }), "the lowest admitted values are valid");
    check(!validRuntimeCameraLens({ 0.0005f, 0.0f, 0.0f, 0.0f }), "a focus distance below the camera range is refused");
    check(!validRuntimeCameraLens({ 0.0f, 200.0f, 0.0f, 0.0f }), "an f-stop above the camera range is refused");
    check(!validRuntimeCameraLens({ 0.0f, 0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f }),
          "a non-finite blade rotation is refused");

    if (failures != 0) return EXIT_FAILURE;
    std::cout << "runtime camera lens tests passed\n";
    return EXIT_SUCCESS;
}
