#pragma once
#include "linear_scene_hdr_checks.h"
#include "../../src/renderer.h"
#include "../../src/raw_pass_export.h"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace hdrimagechecks
{
template <typename Renderer, typename Draw>
bool sceneComposite(Renderer& renderer, Draw&& draw, std::string& error, int width = 1, int height = 1)
{
    auto frame = linearscenechecks::capture(arbitgpu::nativeFixtureSceneBackend(), error);
    if (!frame) return false;
    videorender::LayerDesc layer;
    layer.nativeTextureDescriptor = frame->colorTextureDescriptor();
    layer.nativeTextureBackend = frame->backend();
    layer.nativeTextureView = frame->colorTextureViewHandle();
    layer.texture = frame->backend() == "opengl" ? static_cast<unsigned>(layer.nativeTextureView) : 0;
    layer.nativeTextureOwner = frame; layer.texWidth = 64; layer.texHeight = 64;
    renderer.setHdrImageCapture(true);
    std::vector<float> first, repeat;
    const auto center = (static_cast<std::size_t>(height / 2) * width + width / 2) * 4;
    if (!draw(&layer, 1) || !renderer.readLastCompositeFloat(first, error)
        || first.size() != static_cast<std::size_t>(width) * height * 4
        || !linearscenechecks::nearPixels(std::vector<float>(first.begin() + center, first.begin() + center + 4), error)
        || !draw(&layer, 1)
        || !renderer.readLastCompositeFloat(repeat, error) || first != repeat) return false;
    std::vector<float> wide;
    std::vector<std::uint8_t> preview;
    if (!hdrimage::convert(*hdrimage::find("linear-rec2020"), first, wide, preview, error)
        || wide.empty() || wide[center] <= 1 || preview.empty())
    { error = "Native Scene3D failed actual HDR primaries conversion/reference preview"; return false; }
    const auto root = std::filesystem::temp_directory_path() / ("donutstudio-linear-scene-sequence-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) { error = "HDR fixture directory unavailable"; return false; }
    struct Cleanup { std::filesystem::path path; ~Cleanup()
        { std::error_code ignored; std::filesystem::remove_all(path, ignored); } } cleanup {root};
    arbitgpu::NativeRawPassPixels pixels {arbitgpu::NativeTexturePixelFormat::Rgba32Float,
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
        std::vector<std::uint8_t>(wide.size() * sizeof(float))};
    std::memcpy(pixels.bytes.data(), wide.data(), pixels.bytes.size());
    {
        videohelper::rawexport::Session sequence((root / "linear.mov").string());
        for (int i = 0; i < 2; ++i)
            if (!sequence.appendFinal(i, i / 24.0, 24, frame->backend(), *hdrimage::find("linear-rec2020"), pixels, error))
                return false;
        if (!sequence.finish(error)) return false;
    }
    {
        videohelper::rawexport::Session cancelled((root / "cancelled.mov").string());
        if (!cancelled.appendFinal(0, 0, 24, frame->backend(), *hdrimage::find("linear-rec2020"), pixels, error))
            return false;
    }
    if (!std::filesystem::exists(root / "linear.mov.passes" / ".complete")
        || std::filesystem::exists(root / "cancelled.mov.passes"))
    { error = "Native HDR sequence completion/cancellation ownership differs"; return false; }
    auto undeclared = layer; undeclared.nativeTextureDescriptor.transfer = colortransform::TransferFunction::Unspecified;
    const std::array<videorender::LayerDesc, 2> mixed {layer, undeclared};
    if (draw(mixed.data(), 2) || renderer.readLastCompositeFloat(repeat, error))
    { error = "Mixed native transfer was accepted or left a stale HDR readback"; return false; }
    auto effect = layer; effect.blendMode = 1;
    if (draw(&effect, 1)) { error = "SDR clipping blend admitted to linear HDR"; return false; }
    renderer.setHdrImageCapture(false);
    if (draw(&layer, 1)) { error = "Linear scene silently entered SDR preview without conversion"; return false; }
    error.clear(); return true;
}

// The same fixture runs through production OpenGL and Metal compositor targets.
template <typename Renderer, typename Draw>
bool readback(Renderer& renderer, Draw&& draw, std::string& error)
{
    renderer.setHdrImageCapture(true);
    std::vector<float> first, repeat;
    if (renderer.readLastCompositeFloat(first, error))
    { error = "HDR capture accepted a stale result after mode change"; return false; }
    renderer.setBackgroundColor(0.25f, 0.5f, 0.75f, 1.0f);
    renderer.setPostFx(0.0f, 1.0f, 0.0f, 0, 4.0f);
    if (!draw() || !renderer.readLastCompositeFloat(first, error)
        || !renderer.readLastCompositeFloat(repeat, error) || first.empty() || first != repeat)
    { error = "HDR final readback failed or changed without another render: " + error; return false; }
    for (std::size_t i = 0; i < first.size(); i += 4)
        if (std::abs(first[i] - 1.0f) > 0.004f || std::abs(first[i + 1] - 2.0f) > 0.004f
            || std::abs(first[i + 2] - 3.0f) > 0.004f || first[i + 3] != 1.0f)
        { error = "HDR float readback clipped extended-range channels"; return false; }
    renderer.setHdrImageCapture(false);
    if (renderer.readLastCompositeFloat(repeat, error))
    { error = "HDR capture accepted a stale result after leaving capture mode"; return false; }
    if (!draw() || !renderer.readLastCompositeFloat(repeat, error)) return false;
    for (float value : repeat)
        if (value < 0.0f || value > 1.0f)
        { error = "Default SDR post transform changed"; return false; }
    renderer.setPostFx(0.0f, 1.0f, 0.0f, 0, 1.0f);
    renderer.setBackgroundColor(0, 0, 0, 1);
    error.clear(); return true;
}
} // namespace hdrimagechecks
