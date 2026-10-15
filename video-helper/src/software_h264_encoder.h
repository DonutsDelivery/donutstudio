#pragma once

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

#include <climits>
#include <cmath>
#include <string>

namespace videohelper
{
// The named x264 fallback preserves existing installations during dependency
// migration. Its availability does not establish a GPL-free dependency profile.
inline const char* softwareH264EncoderName()
{
    return avcodec_find_encoder_by_name("libopenh264") != nullptr
        ? "libopenh264" : "libx264";
}

inline bool isSoftwareH264Encoder(const std::string& name)
{
    return name == "libopenh264" || name == "libx264";
}

// OpenH264 quality mode has no x264 CRF or preset equivalent. Keep a native
// QP ceiling and a per-frame bitrate budget, with frame skipping forbidden.
// FFmpeg n6.1 maps these fields/options to OpenH264's encoder parameters.
inline std::string configureSoftwareH264(AVCodecContext* context,
                                         const std::string& name, int qualityQp,
                                         const char* x264Preset)
{
    if (context == nullptr || context->priv_data == nullptr)
        return "software H.264 encoder context unavailable";
    auto option = [&](const char* key, const char* value) -> std::string
    {
        const int rc = av_opt_set(context->priv_data, key, value, 0);
        if (rc >= 0) return {};
        char error[AV_ERROR_MAX_STRING_SIZE] {};
        av_strerror(rc, error, sizeof(error));
        return name + " option " + key + " failed: " + error;
    };
    if (name == "libx264")
    {
        if (auto error = option("preset", x264Preset); !error.empty()) return error;
        return option("crf", std::to_string(qualityQp).c_str());
    }
    if (name != "libopenh264") return "unsupported software H.264 encoder: " + name;
    if (context->width <= 0 || context->height <= 0
        || (context->width & 1) || (context->height & 1)
        || context->pix_fmt != AV_PIX_FMT_YUV420P
        || context->framerate.num <= 0 || context->framerate.den <= 0
        || qualityQp < 1 || qualityQp > 51)
        return "invalid OpenH264 dimensions, rate, pixel format or quality";
    const long double budget = static_cast<long double>(context->width)
        * context->height * 8 * context->framerate.num / (30.0L * context->framerate.den);
    // OpenH264's target/max bitrate parameters are signed 32-bit integers.
    if (!std::isfinite(budget) || budget < 1 || budget > INT_MAX)
        return "OpenH264 bitrate budget is outside its supported range";
    context->bit_rate = static_cast<int64_t>(std::llround(budget));
    context->rc_max_rate = context->bit_rate;
    context->qmin = 1;
    context->qmax = qualityQp;
    context->max_b_frames = 0; // OpenH264 has no B-frame encoding path.
    if (auto error = option("rc_mode", "quality"); !error.empty()) return error;
    if (auto error = option("allow_skip_frames", "0"); !error.empty()) return error;
    return option("profile", "main");
}

// Call only after avcodec_open2 succeeds, using the opened context's identity.
// MP4 preserves these standard format tags for independent ffprobe inspection.
inline void tagSoftwareH264Encoder(AVFormatContext* format, const AVCodecContext* context)
{
    if (format == nullptr || context == nullptr || context->codec == nullptr
        || !isSoftwareH264Encoder(context->codec->name)) return;
    const std::string name = context->codec->name;
    av_dict_set(&format->metadata, "encoder", name.c_str(), 0);
    const std::string policy = name == "libopenh264"
        ? "video-encoder=libopenh264;quality=openh264-quality-v1;target-bitrate="
            + std::to_string(context->bit_rate) + ";qp-min=1;qp-max="
            + std::to_string(context->qmax) + ";skip-frames=0"
        : "video-encoder=libx264;quality=x264-crf;dependency-migration=transitional";
    av_dict_set(&format->metadata, "comment", policy.c_str(), 0);
}
} // namespace videohelper
