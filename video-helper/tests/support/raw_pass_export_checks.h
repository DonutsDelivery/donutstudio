#pragma once
#include "../../src/raw_pass_export.h"
#include <chrono>
#include <sstream>

namespace rawexportchecks
{
class Frame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    const std::string& backend() const noexcept override { static const std::string name = "opengl"; return name; }
    std::uint32_t width() const noexcept override { return 3; }
    std::uint32_t height() const noexcept override { return 2; }
    std::uintptr_t colorImageHandle() const noexcept override { return 1; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return 1; }
    bool unsupported = false, truncated = false;
    bool linearColor = false;
    arbitgpu::NativeTextureViewDescriptor passTextureDescriptor(renderpassoutput::Output output) const noexcept override
    {
        using F = arbitgpu::NativeTexturePixelFormat;
        constexpr std::array<F, 8> formats { F::Rgba8Unorm, F::R32Float, F::Rgba16Float,
            F::Rg16Float, F::Rgba16Float, F::R8Unorm, F::R32Uint, F::R32Uint };
        if (static_cast<unsigned>(output) >= formats.size()) return {};
        auto result = arbitgpu::NativeTextureViewDescriptor { backend(), arbitgpu::NativeTextureViewKind::Texture2D,
            unsupported ? F::Invalid : formats[static_cast<unsigned>(output)], 1, 1, width(), height(), 1, true, 1, 1 };
        if (linearColor && output == renderpassoutput::Output::Color)
        {
            result.format = F::Rgba16Float;
            result.colorSpace = colortransform::ColorSpace::LinearSRGB;
            result.transfer = colortransform::TransferFunction::Linear;
        }
        return result;
    }
    bool readRawPass(renderpassoutput::Output output, arbitgpu::NativeRawPassPixels& pixels, std::string& error) const override
    {
        const auto descriptor = passTextureDescriptor(output);
        if (!descriptor.complete()) { pixels = {}; error = "unavailable"; return false; }
        const auto scalarBytes = arbitgpu::rawPassScalarBytes(descriptor.format);
        pixels = { descriptor.format, width(), height(), std::vector<std::uint8_t>(
            width() * height() * arbitgpu::rawPassChannels(descriptor.format) * scalarBytes) };
        // Includes high unsigned bits and distinct rows/channels. The writer must preserve every bit.
        for (std::size_t i = 0; i < pixels.bytes.size(); ++i) pixels.bytes[i] = static_cast<std::uint8_t>(0xa5u + i * 17u);
        if (truncated) pixels.bytes.pop_back();
        error.clear();
        return true;
    }
};

inline std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}

template <typename Check>
void verify(Check&& check)
{
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("donutstudio-raw-pass-test-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!fs::create_directory(root)) { check(false, "raw export test directory creates"); return; }
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup {root};
    auto frame = std::make_shared<Frame>();
    std::string error;
    const auto append = [&](videohelper::rawexport::Session& session, std::uint32_t mask = 255u)
    { return session.append(7, 72, 0, 1.25, 24, mask, renderpassoutput::Output::Color, frame, error); };
    const auto bundle = root / "completed.mov.passes";
    {
        videohelper::rawexport::Session session((root / "completed.mov").string());
        check(append(session), "all selected raw passes write without a GPU or new codec dependency");
        check(!fs::exists(bundle / ".complete"), "partial bundle is never marked complete");
        check(session.finish(error), "successful export closes its manifest");
    }
    check(fs::exists(bundle / ".complete"), "completed bundle survives session teardown");
    {
        frame->linearColor = true;
        videohelper::rawexport::Session session((root / "native-linear.mov").string());
        check(append(session, 1) && session.finish(error), "native linear Image writes lossless half-float output");
        const auto metadata = nlohmann::json::parse(readFile(root / "native-linear.mov.passes" / "manifest.json"));
        const auto& file = metadata["frames"][0]["files"][0];
        check(file["dtype"] == "<f2" && file["primaries"] == "srgb-bt709-d65" && file["transfer"] == "linear",
            "native HDR AOV manifest describes the actual pixel representation");
        frame->linearColor = false;
    }
    const auto manifestBytes = readFile(bundle / "manifest.json");
    const auto manifest = nlohmann::json::parse(manifestBytes, nullptr, false);
    check(!manifest.is_discarded() && manifest.value("sampleCount", 0) == 1,
        "manifest is valid JSON with the exact committed sample count");
    if (!manifest.is_discarded() && manifest.contains("frames") && manifest["frames"].size() == 1)
    {
        const auto& entry = manifest["frames"][0];
        check(entry["clipId"] == 7 && entry["renderId"] == 72 && entry["timelineSeconds"] == 1.25
            && entry["files"].size() == 8, "manifest records clip, renderer, timeline and every selected file");
        for (const auto& file : entry["files"])
        {
            const auto output = renderpassoutput::outputFromToken(file["output"].get<std::string>());
            check(output.has_value(), "manifest output tokens use the existing attachment authority");
            if (!output) continue;
            arbitgpu::NativeRawPassPixels pixels;
            frame->readRawPass(*output, pixels, error);
            const auto bytes = readFile(bundle / file["path"].get<std::string>());
            check(bytes.size() >= 10 && bytes.substr(1, 5) == "NUMPY"
                && static_cast<unsigned char>(bytes[0]) == 0x93 && bytes[6] == 1 && bytes[7] == 0,
                "lossless pass files use the documented NPY 1.0 magic and version");
            if (bytes.size() < 10) continue;
            const auto headerLength = static_cast<unsigned char>(bytes[8])
                + 256u * static_cast<unsigned char>(bytes[9]);
            check((10u + headerLength) % 64u == 0u && bytes.size() == 10u + headerLength + pixels.bytes.size(),
                "NPY headers are aligned and payloads have exactly the typed byte length");
            const auto header = bytes.substr(10, headerLength);
            check(header.find("'shape': (2, 3, ") != std::string::npos
                && header.find(videohelper::rawexport::dtype(pixels.format)) != std::string::npos,
                "NPY header describes height, width, channels and explicit scalar byte order");
            const std::uint16_t endian = 1;
            if (*reinterpret_cast<const std::uint8_t*>(&endian) == 1 && bytes.size() == 10u + headerLength + pixels.bytes.size())
                check(std::memcmp(bytes.data() + 10u + headerLength, pixels.bytes.data(), pixels.bytes.size()) == 0,
                    "NPY serialization preserves float16, float32 and full uint32 bits without conversion");
        }
    }
    {
        videohelper::rawexport::Session replay((root / "replay.mov").string());
        check(append(replay) && replay.finish(error), "same sample can export to a fresh destination");
    }
    for (const auto& file : fs::directory_iterator(bundle))
        check(readFile(file.path()) == readFile(root / "replay.mov.passes" / file.path().filename()),
            "identical export samples produce byte-identical arrays and manifest");
    {
        videohelper::rawexport::Session cancelled((root / "cancelled.mov").string());
        check(append(cancelled), "partial sample writes before cancellation");
    }
    check(!fs::exists(root / "cancelled.mov.passes"), "cancellation removes the owned partial bundle");
    for (const bool unsupported : { false, true })
    {
        frame->truncated = !unsupported; frame->unsupported = unsupported;
        { videohelper::rawexport::Session failed((root / "failed.mov").string());
          check(!append(failed) && !error.empty(), "unsupported or truncated raw data fails export"); }
        check(!fs::exists(root / "failed.mov.passes"), "failed raw export removes partial arrays and manifest");
    }
    frame->truncated = frame->unsupported = false;
    {
        videohelper::rawexport::Session duplicate((root / "duplicate.mov").string());
        check(append(duplicate) && !append(duplicate), "duplicate frame paths never overwrite an earlier sample");
    }
    {
        videohelper::rawexport::Session exists((root / "completed.mov").string());
        check(!append(exists), "pre-existing output directory is not adopted or overwritten");
    }
    check(readFile(bundle / "manifest.json") == manifestBytes, "failed collision preserves the existing bundle");
    {
        videohelper::rawexport::Session unselected((root / "unselected.mov").string());
        check(append(unselected, 0) && unselected.finish(error), "no selections preserve ordinary video-only export");
    }
    check(!fs::exists(root / "unselected.mov.passes"), "unselected export creates no files");

    const auto* profile = hdrimage::find("linear-rec2020");
    check(profile != nullptr && hdrimage::find("pq") == nullptr && hdrimage::find("hlg") == nullptr,
        "HDR video and unknown profiles are explicitly unsupported");
    if (profile == nullptr) return;
    std::vector<float> linear;
    std::vector<std::uint8_t> preview;
    check(hdrimage::convert(*profile, {1, 0, 0, 1, 4, 4, 4, 1}, linear, preview, error),
        "HDR conversion accepts signed extended-range linear source data");
    check(linear.size() == 8 && std::abs(linear[0] - 0.6274039f) < 0.00001f
        && std::abs(linear[1] - 0.0690973f) < 0.00001f && std::abs(linear[2] - 0.0163914f) < 0.00001f
        && std::abs(linear[4] - 4.0f) < 0.00001f,
        "Rec.2020 conversion golden changes primary coordinates and retains highlights");
    check(preview.size() == 8 && preview[0] == 180 && preview[4] == 228 && preview[7] == 255,
        "SDR reference applies Reinhard and BT.709 OETF to actual pixels");
    arbitgpu::NativeRawPassPixels hdrPixels { arbitgpu::NativeTexturePixelFormat::Rgba32Float, 2, 1,
        std::vector<std::uint8_t>(linear.size() * sizeof(float)) };
    std::memcpy(hdrPixels.bytes.data(), linear.data(), hdrPixels.bytes.size());
    for (const char* name : {"hdr", "hdr-replay"})
    {
        videohelper::rawexport::Session session((root / (std::string(name) + ".mov")).string());
        check(session.appendFinal(0, 1.25, 24, "opengl", *profile, hdrPixels, error)
            && session.appendFinal(1, 1.25 + 1.0 / 24, 24, "opengl", *profile, hdrPixels, error)
            && session.finish(error), "HDR image sequence commits through existing bundle ownership");
    }
    const auto hdrBundle = root / "hdr.mov.passes";
    for (const auto& file : fs::directory_iterator(hdrBundle))
        check(readFile(file.path()) == readFile(root / "hdr-replay.mov.passes" / file.path().filename()),
            "HDR sequence replay is byte-identical including the color manifest");
    const auto hdrManifest = nlohmann::json::parse(readFile(hdrBundle / "manifest.json"), nullptr, false);
    check(!hdrManifest.is_discarded() && hdrManifest["frames"][0]["outputPrimaries"] == "rec2020-d65"
        && hdrManifest["frames"][0]["outputTransfer"] == "linear"
        && hdrManifest["frames"][0]["files"][0]["dtype"] == "<f4",
        "HDR manifest describes actual converted float pixels, not fake HDR video tags");
    {
        videohelper::rawexport::Session cancelled((root / "hdr-cancelled.mov").string());
        check(cancelled.appendFinal(0, 0, 24, "metal", *profile, hdrPixels, error), "Metal uses the same HDR writer");
    }
    check(!fs::exists(root / "hdr-cancelled.mov.passes"), "HDR cancellation cleans the partial image sequence");
    {
        videohelper::rawexport::Session duplicate((root / "hdr-duplicate.mov").string());
        check(duplicate.appendFinal(0, 0, 24, "metal", *profile, hdrPixels, error)
            && !duplicate.appendFinal(0, 0, 24, "metal", *profile, hdrPixels, error)
            && !duplicate.finish(error), "duplicate HDR frame rejects the entire incomplete bundle");
    }
    check(hdrimage::convert(*hdrimage::find("srgb-linear-p3"), {0.5f, 0.5f, 0.5f, 1}, linear, preview, error)
        && linear.size() == 4 && std::abs(linear[0] - 0.21404114f) < 0.00001f,
        "sRGB interpretation decodes its transfer before P3 primary conversion");
    check(hdrimage::convert(*hdrimage::find("linear-p3"), {1, 0, 0, 1}, linear, preview, error)
        && linear.size() == 4 && std::abs(linear[0] - 0.82246197f) < 0.00001f
        && std::abs(linear[1] - 0.03319420f) < 0.00001f
        && std::abs(linear[2] - 0.01708263f) < 0.00001f,
        "Display P3 primary conversion golden is not metadata-only");
    const std::vector<float> signedLinear {-0.5f, 2.0f, 4.0f, 1.0f};
    check(hdrimage::convert(*hdrimage::find("linear-srgb"), signedLinear, linear, preview, error)
        && linear == signedLinear && preview[0] == 0, "identity linear export preserves signed highlights exactly");
    check(!hdrimage::convert(*profile, {1, 2, 3, 0.5f}, linear, preview, error)
        && linear.empty() && preview.empty(), "unsupported transparent HDR output fails closed");
    check(!hdrimage::convert(*profile, {INFINITY, 2, 3, 1}, linear, preview, error), "nonfinite HDR pixels fail closed");
    check(hdrimage::halfToFloat(0x4400) == 4.0f && hdrimage::halfToFloat(0xbc00) == -1.0f
        && hdrimage::halfToFloat(1) == std::ldexp(1.0f, -24)
        && std::isinf(hdrimage::halfToFloat(0x7c00)) && std::isnan(hdrimage::halfToFloat(0x7c01)),
        "Metal half readback handles finite, subnormal and nonfinite IEEE values");
}
} // namespace rawexportchecks
