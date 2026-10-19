#pragma once

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <thread>

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

// This also turns on multithreaded libx264: the exporter, the recorder and the
// stabilization and proxy encodes now run with automatic threads, bounded by the
// plan below, instead of one thread. Large exports and the 1080p veryfast
// recorder encode faster; the recorder's frame memory at 1080p on a 26-core
// machine is about 137 frames (0.8 GB) rather than about 23.
//
// libx264 keeps every frame inside its lookahead window and every frame-thread
// slot resident until the encoder drains. Measured at 3840x2160 with ffmpeg
// n9.0.2, medium preset: one queued frame costs about 3.05 bytes per pixel
// (24 MiB at 4K, 6 MiB at 1080p), and one frame thread holds about three
// frames' worth. Peak frames are roughly lookahead + 3 * threads + 10. With the
// automatic thread count on a 26-core machine that is about 4.2 GB at 4K (167 frames of 25 MB), which a
// short export held in full because it never reached the end of the window.
struct X264MemoryPlan
{
    int lookahead = -1;  // -1 keeps the preset's rc-lookahead
    int threads = 0;     // 0 keeps libx264's automatic thread count
};

inline constexpr std::uint64_t kX264FrameBudgetBytes = 1024ull * 1024ull * 1024ull;
inline constexpr int kX264FramesPerThread = 3;
inline constexpr int kX264FixedFrames = 10;  // B-frames, references, the frame being coded

inline std::uint64_t x264BytesPerFrame(int width, int height)
{
    return width <= 0 || height <= 0 ? 0
        : static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 61u / 20u;
}

// The preset's own rc-lookahead, from x264 r3222 x264_param_default_preset and
// checked against the options line libx264 prints at open: ultrafast 0 and
// superfast 0 (mbtree off, no rc_lookahead printed), veryfast 10, faster 20,
// fast 30, medium 40, slow 50, slower 60, veryslow 60, placebo 60. Unknown names
// count as medium.
inline int x264PresetLookahead(const std::string& preset)
{
    if (preset == "ultrafast" || preset == "superfast") return 0;
    if (preset == "veryfast") return 10;
    if (preset == "faster") return 20;
    if (preset == "fast") return 30;
    if (preset == "slow") return 50;
    if (preset == "slower" || preset == "veryslow" || preset == "placebo") return 60;
    return 40;
}

// libx264 resolves a thread count of 0 to 3/2 of the cores, capped at 128.
// AVCodecContext defaults thread_count to 1 (only the ffmpeg CLI asks for auto),
// so before this planner every helper libx264 encode ran on one thread.
inline int x264AutoThreads(unsigned hardwareThreads)
{
    return static_cast<int>(std::clamp<unsigned>(
        hardwareThreads == 0 ? 8u : hardwareThreads * 3u / 2u, 1u, 128u));
}

// currentThreads is the count the encoder would run with: the context's explicit
// thread_count, or x264AutoThreads when that is 0.
// Leaves the encoder alone when the unrestricted worst case fits the budget, so
// 1080p and smaller encodes keep the preset's lookahead and run automatic
// threads (they used to run one, see above). Above the budget it lowers the
// thread count first, then the lookahead, never raising either.
inline X264MemoryPlan planX264Memory(int width, int height, const std::string& preset,
                                     int currentThreads,
                                     std::uint64_t budgetBytes = kX264FrameBudgetBytes)
{
    const auto frameBytes = x264BytesPerFrame(width, height);
    if (frameBytes == 0) return {};
    const int presetLookahead = x264PresetLookahead(preset);
    const int autoThreads = std::clamp(currentThreads, 1, 128);
    const int maxFrames = static_cast<int>(std::clamp<std::uint64_t>(
        budgetBytes / frameBytes, 16u, 100000u));
    if (presetLookahead + kX264FramesPerThread * autoThreads + kX264FixedFrames <= maxFrames)
        return {};
    X264MemoryPlan plan;
    const int threads = std::min(std::max(2, maxFrames / 6), autoThreads);
    if (threads < autoThreads) plan.threads = threads;
    const int lookahead = std::min(presetLookahead, std::max(std::min(8, presetLookahead),
        maxFrames - kX264FramesPerThread * threads - kX264FixedFrames));
    if (lookahead < presetLookahead) plan.lookahead = lookahead;
    return plan;
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
        // Ask for automatic threading (the context default is 1, which also
        // serialised the 4K export), then let the plan lower it to the budget.
        // A thread_count of 0 or 1 (the context default) therefore becomes
        // automatic; only a caller that set an explicit count above 1 keeps it,
        // as the ceiling the plan may lower.
        if (context->thread_count <= 1) context->thread_count = 0;
        const auto plan = planX264Memory(context->width, context->height,
                                         x264Preset != nullptr ? x264Preset : "",
                                         context->thread_count > 0 ? context->thread_count
                                             : x264AutoThreads(std::thread::hardware_concurrency()));
        if (plan.threads > 0) context->thread_count = plan.threads;
        if (plan.lookahead >= 0)
            if (auto error = option("rc-lookahead", std::to_string(plan.lookahead).c_str()); !error.empty())
                return error;
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
