#pragma once

#include "gpu_backend/backend.h"
#include "../../shared/HdrImageOutputContract.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <locale>

namespace videohelper::rawexport
{
inline const char* dtype(arbitgpu::NativeTexturePixelFormat format) noexcept
{
    using F = arbitgpu::NativeTexturePixelFormat;
    switch (format)
    {
        case F::Rgba8Unorm: case F::R8Unorm: return "|u1";
        case F::Rgba16Float: case F::Rg16Float: return "<f2";
        case F::R32Float: case F::Rgba32Float: return "<f4";
        case F::R32Uint: return "<u4";
        default: return "";
    }
}

// NPY 1.0, C-order [height, width, channels], always little endian.
inline bool writeNpy(std::ostream& stream, const arbitgpu::NativeRawPassPixels& pixels)
{
    const auto channels = arbitgpu::rawPassChannels(pixels.format);
    const auto scalarBytes = arbitgpu::rawPassScalarBytes(pixels.format);
    if (*dtype(pixels.format) == '\0' || !arbitgpu::nativeFixtureDimensionsWithinBounds(pixels.width, pixels.height)
        || pixels.bytes.size() != static_cast<std::uint64_t>(pixels.width) * pixels.height * channels * scalarBytes)
        return false;
    std::string header = "{'descr': '" + std::string(dtype(pixels.format))
        + "', 'fortran_order': False, 'shape': (" + std::to_string(pixels.height) + ", "
        + std::to_string(pixels.width) + ", " + std::to_string(channels) + "), }";
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header += '\n';
    const unsigned char prefix[] { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
        static_cast<unsigned char>(header.size() & 255), static_cast<unsigned char>(header.size() >> 8) };
    stream.write(reinterpret_cast<const char*>(prefix), sizeof(prefix));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const std::uint8_t*>(&endian) == 1 || scalarBytes == 1)
        stream.write(reinterpret_cast<const char*>(pixels.bytes.data()), static_cast<std::streamsize>(pixels.bytes.size()));
    else
        for (std::size_t offset = 0; offset < pixels.bytes.size(); offset += scalarBytes)
            for (unsigned byte = scalarBytes; byte > 0; --byte)
                stream.put(static_cast<char>(pixels.bytes[offset + byte - 1]));
    return stream.good();
}

// The export job owns this sidecar alongside its partial video. No directory is
// created until a selected pass is rendered. Failure removes only our directory.
class Session final
{
public:
    explicit Session(const std::string& videoPath) : directory_(videoPath + ".passes") {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session()
    {
        manifest_.close();
        if (owned_ && !completed_) { std::error_code ignored; std::filesystem::remove_all(directory_, ignored); }
    }

    bool append(int clipId, std::uint32_t renderId, std::int64_t frameIndex, double timelineSeconds,
                double fps, std::uint32_t mask, renderpassoutput::Output imageOutput,
                const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame, std::string& error)
    {
        if (mask == 0) return true;
        if (completed_ || failed_)
        { error = "Raw pass export no longer accepts samples"; return false; }
        failed_ = true;
        if (mask > render3dimage::kRawExportMask || clipId < 0 || renderId == 0
            || frameIndex < 0 || !std::isfinite(timelineSeconds) || !std::isfinite(fps) || fps <= 0
            || !frame || !render3dimage::supported(imageOutput))
        { error = "Raw pass export has invalid frame metadata"; return false; }
        if (!ensureDirectory(error)) return false;
        const auto prefix = "clip-" + std::to_string(clipId) + "-render-" + std::to_string(renderId)
            + "-frame-" + std::to_string(frameIndex) + "-";
        nlohmann::json entry { {"clipId", clipId}, {"renderId", renderId}, {"frameIndex", frameIndex},
            {"stage", "scene-before-clip-composite"},
            {"timelineSeconds", timelineSeconds}, {"fps", fps}, {"backend", frame->backend()},
            {"imageOutput", renderpassoutput::token(imageOutput)}, {"files", nlohmann::json::array()} };
        for (const auto output : renderpassoutput::kOutputs)
        {
            if (!render3dimage::exports(mask, output)) continue;
            const auto descriptor = frame->passTextureDescriptor(output);
            arbitgpu::NativeRawPassPixels pixels;
            if (!descriptor.complete() || descriptor.backend != frame->backend()
                || descriptor.width != frame->width() || descriptor.height != frame->height()
                || !arbitgpu::rawPassFormatMatches(output, descriptor.format)
                || (output == renderpassoutput::Output::Color
                    && descriptor.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float
                    && !arbitgpu::isLinearSceneColor(descriptor)))
            { error = "Raw export selected an unavailable native attachment: " + std::string(renderpassoutput::token(output)); return false; }
            if (!frame->readRawPass(output, pixels, error)) return false;
            if (!arbitgpu::rawPassFormatMatches(output, pixels.format) || pixels.width != descriptor.width
                || pixels.height != descriptor.height || *dtype(pixels.format) == '\0')
            { error = "Raw pass readback returned a mismatched type or extent"; return false; }
            const auto filename = prefix + std::string(renderpassoutput::token(output)) + ".npy";
            std::error_code ec;
            if (std::filesystem::exists(directory_ / filename, ec) || ec)
            { error = "Raw pass frame was submitted more than once"; return false; }
            std::ofstream stream(directory_ / filename, std::ios::binary);
            if (!stream || !writeNpy(stream, pixels))
            { error = "Could not write raw pass array: " + filename; return false; }
            stream.close();
            if (!stream) { error = "Could not finish raw pass array: " + filename; return false; }
            entry["files"].push_back({ {"output", renderpassoutput::token(output)}, {"path", filename},
                {"dtype", dtype(pixels.format)}, {"shape", {pixels.height, pixels.width, arbitgpu::rawPassChannels(pixels.format)}} });
            if (output == renderpassoutput::Output::Color && arbitgpu::isLinearSceneColor(descriptor))
            {
                entry["files"].back()["primaries"] = "srgb-bt709-d65";
                entry["files"].back()["transfer"] = "linear";
            }
        }
        if (samples_++ != 0) manifest_ << ",\n";
        manifest_ << entry.dump();
        if (!manifest_) { error = "Could not write raw pass manifest"; return false; }
        failed_ = false;
        error.clear();
        return true;
    }

    bool appendFinal(std::int64_t frameIndex, double timelineSeconds, double fps,
                     const std::string& backend, const hdrimage::Profile& profile,
                     const arbitgpu::NativeRawPassPixels& pixels, std::string& error)
    {
        if (completed_ || failed_) { error = "HDR export no longer accepts samples"; return false; }
        failed_ = true;
        if (frameIndex < 0 || !std::isfinite(timelineSeconds) || !std::isfinite(fps) || fps <= 0
            || (backend != "opengl" && backend != "metal") || !hdrimage::enabled(profile.token)
            || pixels.format != arbitgpu::NativeTexturePixelFormat::Rgba32Float)
        { error = "HDR export has invalid final-frame metadata"; return false; }
        if (!ensureDirectory(error)) return false;
        const auto filename = "final-frame-" + std::to_string(frameIndex) + "-linear.npy";
        std::error_code ec;
        if (std::filesystem::exists(directory_ / filename, ec) || ec)
        { error = "HDR final frame was submitted more than once"; return false; }
        std::ofstream stream(directory_ / filename, std::ios::binary);
        if (!stream || !writeNpy(stream, pixels))
        { error = "Could not write HDR final-frame array"; return false; }
        stream.close();
        if (!stream) { error = "Could not finish HDR final-frame array"; return false; }
        const nlohmann::json entry { {"stage", "timeline-final-composite"}, {"frameIndex", frameIndex},
            {"timelineSeconds", timelineSeconds}, {"fps", fps}, {"backend", backend},
            {"hdrImageProfile", profile.token}, {"inputPrimaries", "srgb-bt709-d65"},
            {"inputTransfer", profile.inputTransfer == colortransform::TransferFunction::Linear ? "linear" : "srgb"},
            {"outputPrimaries", hdrimage::primaries(profile)}, {"outputTransfer", "linear"},
            {"alpha", "opaque"}, {"units", "relative-to-declared-input-white; no absolute luminance claim"},
            {"captureFormat", "rgba16f"}, {"sdrReference", "per-channel Reinhard then BT.709 OETF"},
            {"files", nlohmann::json::array({ { {"output", "linear-image"}, {"path", filename},
                {"dtype", "<f4"}, {"shape", {pixels.height, pixels.width, 4}} } })} };
        if (samples_++ != 0) manifest_ << ",\n";
        manifest_ << entry.dump();
        if (!manifest_) { error = "Could not write HDR manifest entry"; return false; }
        failed_ = false; error.clear(); return true;
    }

    bool finish(std::string& error)
    {
        if (failed_) { error = "Raw pass export contains a failed sample"; return false; }
        if (!owned_) { error.clear(); return true; }
        if (completed_) { error = "Raw pass export is already complete"; return false; }
        manifest_ << "\n],\"sampleCount\":" << samples_ << "}\n";
        manifest_.close();
        if (!manifest_) { error = "Could not finish raw pass manifest"; return false; }
        std::ofstream completion(directory_ / ".complete", std::ios::binary);
        completion << "complete\n";
        completion.close();
        if (!completion) { error = "Could not finish raw pass bundle"; return false; }
        completed_ = true;
        error.clear();
        return true;
    }

private:
    bool ensureDirectory(std::string& error)
    {
        if (owned_) return true;
        std::error_code ec;
        if (!std::filesystem::create_directory(directory_, ec))
        { error = "Raw pass destination exists or cannot be created: " + directory_.string(); return false; }
        owned_ = true;
        std::ofstream marker(directory_ / ".donutstudio-raw-pass-export", std::ios::binary);
        marker << "donutstudio.raw-render-passes.v1\n";
        marker.close();
        if (!marker) { error = "Could not write raw pass ownership marker"; return false; }
        manifest_.open(directory_ / "manifest.json", std::ios::binary);
        manifest_.imbue(std::locale::classic());
        manifest_ << "{\"schema\":\"donutstudio.raw-render-passes.v1\",\"stage\":\"per-frame-entry\","
                     "\"rowOrder\":\"top-to-bottom\",\"arrayOrder\":\"height,width,channels\","
                     "\"color\":\"scene Image is SDR; final linear-image interpretation is declared per entry\","
                     "\"depth\":\"linear camera near-to-far normalized [0,1]\","
                     "\"normal\":\"signed world-space XYZ; W is 1 on geometry, 0 on background\","
                     "\"motion\":\"current-minus-previous pixels XY; positive Y up; reset sample zero\","
                     "\"emission\":\"linear RGBA float16 attachment\","
                     "\"mask\":\"coverage UNORM8; divide by 255\","
                     "\"ids\":\"unsigned 32-bit scene material/object IDs; zero background\",\"frames\":[\n";
        if (!manifest_) { error = "Could not open raw pass manifest"; return false; }
        return true;
    }
    std::filesystem::path directory_;
    std::ofstream manifest_;
    std::uint64_t samples_ = 0;
    bool owned_ = false, completed_ = false, failed_ = false;
};
} // namespace videohelper::rawexport
