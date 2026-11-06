// Path Trace denoisers (Plan 3, Denoising): the Render node's choice to the
// denoiser that runs, the named fallbacks and labels, the guide and flow
// layouts, the denoiser on the Render wire, and OIDN loaded at run time. An
// OIDN run needs the library, and its GPU run a GPU; each skips with the reason
// printed when that is missing.

#include "pathtrace_denoiser.h"
#include "pathtrace_oidn.h"
#include "pathtrace/accumulator.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{

namespace dn = videorender::fixture3d::pathtracedenoise;
namespace pt = videohelper::pathtrace;
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

bool contains (const std::string& text, const std::string& part)
{
    return text.find (part) != std::string::npos;
}

dn::DenoiserAvailability availability (bool tracedOnOptix, bool optix, bool oidn)
{
    dn::DenoiserAvailability available;
    available.tracedOnOptix = tracedOnOptix;
    if (! optix)
        available.optixProblem = "no OptiX context";
    if (! oidn)
        available.oidnProblem = "no OIDN library";
    return available;
}

void selectionTests()
{
    using Choice = dn::DenoiserChoice;
    using D = dn::Denoiser;
    for (bool traced : { false, true })
        for (bool optix : { false, true })
            for (bool oidn : { false, true })
                check (dn::planDenoiser (Choice::Off, availability (traced, optix, oidn)).denoiser == D::None,
                       "Off never runs a denoiser");

    auto plan = dn::planDenoiser (Choice::Auto, availability (true, true, true));
    check (plan.denoiser == D::OptixAi && plan.reason.empty(), "Auto on the OptiX device takes OptiX AI");
    plan = dn::planDenoiser (Choice::Auto, availability (false, true, true));
    check (plan.denoiser == D::Oidn && plan.reason.empty(), "Auto off the OptiX device takes OIDN without a reason");
    plan = dn::planDenoiser (Choice::Auto, availability (true, false, true));
    check (plan.denoiser == D::Oidn && contains (plan.reason, "OptiX AI did not run: no OptiX context"),
           "Auto on OptiX without its denoiser falls back to OIDN by name");
    plan = dn::planDenoiser (Choice::Auto, availability (false, false, false));
    check (plan.denoiser == D::NativeFilter && contains (plan.reason, "OIDN did not run: no OIDN library"),
           "Auto without either library names the native filter's reason");
    plan = dn::planDenoiser (Choice::Auto, availability (true, false, false));
    check (plan.denoiser == D::NativeFilter && contains (plan.reason, "OptiX AI") && contains (plan.reason, "OIDN"),
           "Auto on OptiX with nothing loaded names both reasons");

    plan = dn::planDenoiser (Choice::OptiX, availability (false, true, true));
    check (plan.denoiser == D::OptixAi && plan.reason.empty(), "OptiX runs on a frame traced elsewhere");
    plan = dn::planDenoiser (Choice::OptiX, availability (false, false, true));
    check (plan.denoiser == D::Oidn && contains (plan.reason, "OptiX AI did not run"), "OptiX falls back to OIDN by name");
    plan = dn::planDenoiser (Choice::OptiX, availability (false, false, false));
    check (plan.denoiser == D::NativeFilter && contains (plan.reason, "OptiX AI") && contains (plan.reason, "OIDN"),
           "OptiX with nothing loaded falls back to the native filter with both reasons");

    plan = dn::planDenoiser (Choice::Oidn, availability (true, true, true));
    check (plan.denoiser == D::Oidn && plan.reason.empty(), "OIDN runs even on the OptiX device");
    plan = dn::planDenoiser (Choice::Oidn, availability (true, true, false));
    check (plan.denoiser == D::NativeFilter && contains (plan.reason, "OIDN did not run: no OIDN library"),
           "OIDN never falls back to OptiX AI, only to the native filter by name");

    check (std::string (dn::choiceName (Choice::Off)) == "off" && std::string (dn::choiceName (Choice::Auto)) == "auto"
               && std::string (dn::choiceName (Choice::OptiX)) == "optix" && std::string (dn::choiceName (Choice::Oidn)) == "oidn",
           "choice names");
}

void labelTests()
{
    using D = dn::Denoiser;
    using O = dn::OidnDevice;
    check (std::string (dn::denoiserLabel (D::None, O::Cpu, false)) == "none", "off reports denoiser=none");
    check (std::string (dn::denoiserLabel (D::OptixAi, O::Cpu, false)) == "optix-ai", "OptiX AI label");
    check (std::string (dn::denoiserLabel (D::OptixAi, O::Cpu, true)) == "optix-ai-temporal", "temporal OptiX AI label");
    check (std::string (dn::denoiserLabel (D::Oidn, O::Cpu, false)) == "oidn-cpu", "CPU OIDN label");
    check (std::string (dn::denoiserLabel (D::Oidn, O::Cuda, false)) == "oidn-cuda", "CUDA OIDN label");
    check (std::string (dn::denoiserLabel (D::Oidn, O::Hip, false)) == "oidn-hip", "HIP OIDN label");
    check (std::string (dn::denoiserLabel (D::Oidn, O::Sycl, false)) == "oidn-sycl", "SYCL OIDN label");
    check (std::string (dn::oidnDeviceName (O::Sycl)) == "SYCL", "the SYCL device is named on the HUD");
    check (std::string (dn::denoiserLabel (D::NativeFilter, O::Cuda, true)) == "native-filter", "native filter label");

    check (dn::oidnDeviceFor ("OptiX") == O::Cuda && dn::oidnDeviceFor ("CUDA") == O::Cuda,
           "OIDN goes to CUDA for the NVIDIA devices");
    check (dn::oidnDeviceFor ("HIP") == O::Hip, "OIDN goes to HIP for HIP");
    check (dn::oidnDeviceFor ("oneAPI") == O::Sycl, "OIDN goes to SYCL for oneAPI");
    check (dn::oidnDeviceFor ("CPU") == O::Cpu && dn::oidnDeviceFor ("Metal") == O::Cpu && dn::oidnDeviceFor (nullptr) == O::Cpu,
           "every other device gets CPU OIDN");
}

void guideLayoutTests()
{
    check (dn::kGuideChannels == 3 && dn::kFlowChannels == 2, "three floats per guide pixel, two per flow pixel");
    check (dn::guideOffset (0, 0, 7) == 0 && dn::guideOffset (1, 0, 7) == 3 && dn::guideOffset (0, 1, 7) == 21
               && dn::guideOffset (6, 4, 7) == (4 * 7 + 6) * 3,
           "guide pixels are row-major from the bottom, three floats each");

    pt::Accumulator film;
    film.resize (2, 2);
    film.addGuide (1, 0, { 0.2f, 0.4f, 0.6f }, { 0.0f, 0.0f, 1.0f });
    film.addGuide (1, 0, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f });
    film.addGuideSum (0, 1, { 1.5f, 0.0f, 0.0f }, { 0.0f, 3.0f, 0.0f }, 3);
    pt::Rgb albedo, normal;
    film.guideMean (1, 0, albedo, normal);
    check (std::abs (albedo.r - 0.1f) < 1e-6f && std::abs (albedo.g - 0.2f) < 1e-6f && std::abs (albedo.b - 0.3f) < 1e-6f,
           "a miss adds a zero albedo, halving the mean");
    check (std::abs (normal.b - 1.0f) < 1e-6f && normal.r == 0.0f, "the mean normal is normalized");
    film.guideMean (0, 1, albedo, normal);
    check (std::abs (albedo.r - 0.5f) < 1e-6f && std::abs (normal.g - 1.0f) < 1e-6f, "a device pass's guide sum divides by its count");
    film.guideMean (0, 0, albedo, normal);
    check (albedo.r == 0.0f && normal.r == 0.0f && normal.g == 0.0f && normal.b == 0.0f, "a pixel without guide samples is zero");
    film.reset (film.generation() + 1);
    film.guideMean (1, 0, albedo, normal);
    check (albedo.r == 0.0f && normal.b == 0.0f, "reset clears the guides");
}

pt::PinholeCamera cameraAt (float x)
{
    HarmonicMIDI::grid::SceneCameraRecord record;
    record.transform.translation = { x, 0.0f, 0.0f };
    record.nearPlane = 0.5f;
    record.farPlane = 20.5f;
    pt::PinholeCamera camera;
    check (pt::makePinholeCamera (record, 16, 8, camera), "the test camera is valid");
    return camera;
}

void flowTests()
{
    const auto current = cameraAt (0.0f);
    // A plane at view depth 4.5, so depth = (4.5 - near) / (far - near) = 0.2.
    const std::vector<float> depth (16 * 8, 0.2f);
    std::vector<float> flow;
    check (dn::reprojectionFlow (current, current, depth, flow) && flow.size() == 16 * 8 * 2,
           "flow has two floats per pixel");
    bool still = true;
    for (float value : flow)
        still = still && std::abs (value) < 1e-3f;
    check (still, "a still camera has zero flow");

    // The camera moved right by 0.1, so the plane was 0.1 further right on the
    // previous frame: x - previousX is negative by the pixels 0.1 spans at depth 4.5.
    const auto previous = cameraAt (-0.1f);
    check (dn::reprojectionFlow (current, previous, depth, flow), "a moved camera reprojects");
    const float pixelsPerUnit = 16.0f / (2.0f * 4.5f * current.tanHalfFov * current.aspect);
    bool shifted = true;
    for (std::size_t index = 0; index < 16 * 8; ++index)
        shifted = shifted && std::abs (flow[index * 2] + 0.1f * pixelsPerUnit) < 1e-2f && std::abs (flow[index * 2 + 1]) < 1e-3f;
    check (shifted, "a sideways camera move gives a uniform horizontal flow of the expected size");

    check (! dn::reprojectionFlow (current, previous, std::vector<float> (3, 0.2f), flow), "a depth of another size is refused");
    auto smaller = cameraAt (0.0f);
    smaller.width = 8;
    check (! dn::reprojectionFlow (current, smaller, depth, flow), "a previous camera of another size is refused");
}

void colorTests()
{
    const std::size_t pixels = 2;
    std::vector<std::uint8_t> half (pixels * 8, 0);
    const float values[] { 0.25f, 1.5f, 4.0f, 1.0f, 0.0f, 0.5f, 2.0f, 0.75f };
    for (std::size_t index = 0; index < 8; ++index)
    {
        const std::uint16_t bits = pt::halfBits (values[index]);
        std::memcpy (half.data() + index * 2, &bits, 2);
    }
    std::vector<float> rgb;
    check (dn::colorToFloat3 (half, true, pixels, rgb) && rgb.size() == 6, "half colour reads as three floats per pixel");
    check (rgb[0] == 0.25f && rgb[1] == 1.5f && rgb[2] == 4.0f && rgb[3] == 0.0f && rgb[5] == 2.0f, "alpha is dropped");
    std::vector<std::uint8_t> back;
    dn::float3ToColor (rgb, true, half, back);
    check (back == half, "half colour round-trips, alpha copied");
    rgb[0] = -1.0f;
    rgb[1] = std::nanf ("");
    dn::float3ToColor (rgb, true, half, back);
    std::uint16_t first = 0, second = 0;
    std::memcpy (&first, back.data(), 2);
    std::memcpy (&second, back.data() + 2, 2);
    check (first == 0 && second == 0, "negative and NaN output becomes 0");

    const std::vector<std::uint8_t> display { 255, 128, 0, 77, 10, 20, 30, 255 };
    check (dn::colorToFloat3 (display, false, pixels, rgb) && std::abs (rgb[0] - 1.0f) < 1e-6f, "display bytes read as 0 to 1");
    dn::float3ToColor (rgb, false, display, back);
    check (back == display, "display bytes round-trip");
    check (! dn::colorToFloat3 (display, true, pixels, rgb), "a byte count that does not match is refused");
}

void wireTests()
{
    render::Request traced;
    traced.sourceStableId = 71;
    traced.renderStableId = 72;
    traced.asset.id = "model-asset-1";
    traced.asset.version = 3;
    traced.asset.contentSha256 = std::string (64, 'a');
    traced.asset.sourceMediaType = "model/gltf-binary";
    traced.asset.sourceByteSize = 4096;
    traced.renderEngine = render::RenderEngine::PathTrace;
    traced.pathTraceExportSamples = 128;

    auto automatic = traced;
    automatic.pathTraceDenoise = true;
    const auto autoWire = render::encode (automatic);
    check (autoWire.rfind (std::string (render::kCompiledOperationHeaderV30), 0) == 0,
           "Auto keeps the V30 bytes old projects wrote");
    check (automatic.pathTraceDenoise.denoiser == render::PathTraceDenoiser::Auto, "a switched-on denoise means Auto");

    for (const auto denoiser : { render::PathTraceDenoiser::OptiX, render::PathTraceDenoiser::Oidn })
    {
        auto chosen = traced;
        chosen.pathTraceDenoise = denoiser;
        const auto wire = render::encode (chosen);
        check (wire.rfind (std::string (render::kCompiledOperationHeaderV39), 0) == 0, "a named denoiser uses V39");
        check (wire.find (autoWire) != std::string::npos, "V39 carries the Auto request's exact bytes");
        render::Request decoded;
        check (render::decode (wire, decoded) && decoded.pathTraceDenoise.denoiser == denoiser
                   && render::sameRequest (chosen, decoded) && render::encode (decoded) == wire,
               "a named denoiser round-trips");
        check (! render::sameRequest (automatic, decoded), "the denoiser is part of request identity");
    }

    render::Request decodedAuto;
    check (render::decode (autoWire, decodedAuto) && decodedAuto.pathTraceDenoise.denoiser == render::PathTraceDenoiser::Auto,
           "an old denoised wire decodes as Auto");
    auto tampered = std::string (render::kCompiledOperationHeaderV39) + "\n7 " + std::to_string (autoWire.size()) + "\n" + autoWire;
    render::Request refused;
    check (! render::decode (tampered, refused), "an unknown denoiser is refused");
    const auto offWire = render::encode (traced);
    tampered = std::string (render::kCompiledOperationHeaderV39) + "\n2 " + std::to_string (offWire.size()) + "\n" + offWire;
    check (! render::decode (tampered, refused), "V39 around a request with denoise off is refused");
}

// A sharp step plus noise. OIDN must keep it finite and close to the clean image.
void fillNoisyFrame (std::uint32_t size, std::vector<float>& color, std::vector<float>& albedo, std::vector<float>& normal)
{
    const std::size_t pixels = static_cast<std::size_t> (size) * size;
    color.assign (pixels * 3, 0.0f);
    albedo.assign (pixels * 3, 0.5f);
    normal.assign (pixels * 3, 0.0f);
    std::uint32_t state = 12345u;
    for (std::size_t index = 0; index < pixels; ++index)
    {
        state = state * 1664525u + 1013904223u;
        const float noise = static_cast<float> (state >> 8) / 16777216.0f - 0.5f;
        const float clean = (index % size) < size / 2 ? 0.2f : 0.8f;
        for (int channel = 0; channel < 3; ++channel)
            color[index * 3 + channel] = std::max (0.0f, clean + 0.3f * noise);
        normal[index * 3 + 2] = 1.0f;
    }
}

void oidnTests()
{
    dn::setOidnLibraryForTests ("/nonexistent/libOpenImageDenoise.so.2");
    const auto missing = dn::oidnLoadProblem();
    check (contains (missing, "not installed") && contains (missing, "/nonexistent/libOpenImageDenoise.so.2"),
           "a missing OIDN library is named with where it was looked for");
    std::vector<float> color, albedo, normal, output;
    fillNoisyFrame (16, color, albedo, normal);
    output.assign (color.size(), -1.0f);
    dn::DenoiseFrame frame;
    frame.width = 16;
    frame.height = 16;
    frame.color = color.data();
    frame.albedo = albedo.data();
    frame.normal = normal.data();
    frame.output = output.data();
    dn::OidnDevice used = dn::OidnDevice::Cpu;
    std::string note, error;
    check (! dn::oidnDenoise (dn::OidnDevice::Cpu, frame, used, note, error) && error == missing,
           "OIDN without its library fails with the load reason");

    dn::setOidnLibraryForTests (nullptr);
    if (const auto problem = dn::oidnLoadProblem(); ! problem.empty())
    {
        std::cout << "SKIP OIDN runs: " << problem << '\n';
        return;
    }
    error.clear();
    const bool ran = dn::oidnDenoise (dn::OidnDevice::Cpu, frame, used, note, error);
    check (ran && used == dn::OidnDevice::Cpu && note.empty(), "CPU OIDN runs: " + error);
    if (ran)
    {
        double before = 0.0, after = 0.0;
        bool finite = true;
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            const float clean = ((index / 3) % 16) < 8 ? 0.2f : 0.8f;
            finite = finite && std::isfinite (output[index]) && output[index] >= 0.0f;
            before += std::abs (color[index] - clean);
            after += std::abs (output[index] - clean);
        }
        check (finite, "CPU OIDN writes finite, non-negative output");
        check (after < before, "CPU OIDN brings the noisy frame closer to the clean one");
    }

    // A oneAPI render asks for OIDN's SYCL device; without it OIDN runs on the CPU and says why.
    output.assign (color.size(), -1.0f);
    note.clear();
    error.clear();
    const bool sycl = dn::oidnDenoise (dn::OidnDevice::Sycl, frame, used, note, error);
    if (used == dn::OidnDevice::Sycl)
        check (sycl && note.empty() && std::isfinite (output[0]) && output[0] >= 0.0f, "SYCL OIDN runs: " + error);
    else
        check (sycl && used == dn::OidnDevice::Cpu && contains (note, "SYCL") && contains (note, "ran on the CPU"),
               "a SYCL request that ran on the CPU names SYCL and the CPU: " + note + error);

    output.assign (color.size(), -1.0f);
    note.clear();
    error.clear();
    const bool gpu = dn::oidnDenoise (dn::OidnDevice::Cuda, frame, used, note, error);
    if (used != dn::OidnDevice::Cuda)
    {
        std::cout << "SKIP OIDN CUDA run: " << (note.empty() ? error : note) << '\n';
        check (! gpu || ! note.empty(), "a CUDA request that ran on the CPU says why");
        return;
    }
    check (gpu && std::isfinite (output[0]) && output[0] >= 0.0f, "CUDA OIDN runs: " + error);
}

} // namespace

int main()
{
    selectionTests();
    labelTests();
    guideLayoutTests();
    flowTests();
    colorTests();
    wireTests();
    oidnTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
