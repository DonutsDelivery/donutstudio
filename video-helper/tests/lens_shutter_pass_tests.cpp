// The lens and shutter pass inputs OpenGL and Metal share (lens_shutter_pass.h):
// which passes a frame runs, the bokeh blur scale and kernel, and the velocity
// tap count. The GPU gathers themselves are pinned by
// realtime_velocity_blur_source_contract.py.

#include "gpu_backend/lens_shutter_pass.h"

#include <cmath>
#include <iostream>
#include <string>

namespace
{

namespace lens = arbitgpu::lensshutter;
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

bool near (float a, float b, float tolerance = 1.0e-4f)
{
    return std::fabs (a - b) <= tolerance * std::max (1.0f, std::fabs (b));
}

HarmonicMIDI::grid::SceneCameraRecord lensCamera()
{
    HarmonicMIDI::grid::SceneCameraRecord camera;
    camera.verticalFovRadians = 2.0f * std::atan (0.5f);
    camera.nearPlane = 0.5f;
    camera.farPlane = 100.0f;
    camera.focusDistance = 4.0f;
    camera.fStop = 2.0f;
    camera.apertureBlades = 6;
    camera.bladeRotationRadians = 0.25f;
    camera.apertureRatio = 1.5f;
    return camera;
}

void passSelection()
{
    auto camera = lensCamera();
    const render::ShutterInterval shutter { -0.25f, 0.25f };
    const auto both = lens::colorPasses (true, shutter, camera);
    check (both.velocity && both.bokeh && both.any(), "a Color frame with a shutter and a lens runs both passes");
    check (! lens::colorPasses (false, shutter, camera).any(), "an AOV or inspection image runs neither pass");
    check (! lens::colorPasses (true, {}, camera).velocity, "no shutter, no velocity pass");
    camera.fStop = 0.0f;
    const auto pinhole = lens::colorPasses (true, shutter, camera);
    check (pinhole.velocity && ! pinhole.bokeh, "a pinhole camera runs only the velocity pass");
    check (! lens::colorPasses (true, {}, camera).any(), "a pinhole still frame runs nothing");
}

void bokehScale()
{
    const auto camera = lensCamera();
    const auto parameters = lens::bokehParameters (camera, 640, 480);
    const float aperture = HarmonicMIDI::grid::sceneCameraApertureRadius (camera);
    // tan(fov / 2) is 0.5, so the focus plane spans 2 * 0.5 * 4 = 4 units over 480 rows.
    const float pixelsPerDepth = aperture * 480.0f / 4.0f;
    check (near (parameters.lens[0], pixelsPerDepth), "blur scale is the aperture over the focus plane's pixels");
    check (parameters.lens[1] == 4.0f && parameters.lens[2] == 0.5f && parameters.lens[3] == 100.0f,
           "focus, near and far pass through");
    check (parameters.aperture[0] == 6.0f && parameters.aperture[1] == 0.25f && parameters.aperture[2] == 1.5f,
           "blades, rotation and ratio pass through");
    // The near plane blurs by |4 - 0.5| / 0.5 = 7 times the scale.
    check (near (parameters.aperture[3], std::min (800.0f, pixelsPerDepth * 7.0f)), "the kernel reaches the near plane's blur");

    auto wide = camera;
    wide.fStop = 0.01f;
    const auto clamped = lens::bokehParameters (wide, 640, 480);
    check (near (clamped.aperture[3], 800.0f), "the kernel never passes the image diagonal");
}

void velocityTaps()
{
    const render::ShutterInterval half { -0.25f, 0.25f };
    check (lens::velocityTaps (0.0f, 0.0f, half, 640, 480) == 1, "a still pixel takes one tap");
    check (lens::velocityTaps (30.0f, 40.0f, half, 640, 480) == 25, "one tap per pixel crossed over the shutter");
    check (lens::velocityTaps (30.0f, 40.0f, { 0.0f, 1.0f }, 640, 480) == 50, "a full shutter crosses the whole motion");
    check (lens::velocityTaps (1.0e6f, 0.0f, { 0.0f, 1.0f }, 640, 480) == 800, "never more taps than the diagonal");
}

void metalShader()
{
    const std::string source = lens::metalShaderSource();
    for (const char* entry : { "fragment float4 _copy(", "fragment float4 _velocity(", "fragment float4 _bokeh(" })
        check (source.find (entry) != std::string::npos, std::string ("Metal entry ") + entry);
    check (source.find ("float2(raw.x, -raw.y)") != std::string::npos, "Metal flips the +Y up Motion to its +Y down pixels");
    check (source.find ("pixel + float2(offset.x, -offset.y)") != std::string::npos, "Metal flips aperture offsets to match OpenGL");
}

} // namespace

int main()
{
    passSelection();
    bokehScale();
    velocityTaps();
    metalShader();
    if (failures != 0)
    {
        std::cerr << failures << " of " << checks << " lens and shutter checks failed\n";
        return 1;
    }
    std::cout << checks << " lens and shutter checks passed\n";
    return 0;
}
