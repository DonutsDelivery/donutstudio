#include "software_h264_encoder.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void checkPolicy(const AVCodec* codec, AVRational fps, int qp, bool intraOnly)
{
    AVCodecContext* context = avcodec_alloc_context3(codec);
    require(context != nullptr, "allocate encoder");
    context->width = 160;
    context->height = 96;
    context->pix_fmt = AV_PIX_FMT_YUV420P;
    context->framerate = fps;
    context->time_base = av_inv_q(fps);
    context->gop_size = intraOnly ? 1 : static_cast<int>(std::lround(av_q2d(fps) * 2));
    const auto error = videohelper::configureSoftwareH264(context, codec->name, qp, "veryfast");
    require(error.empty(), error.c_str());
    int64_t value = -1;
    require(av_opt_get_int(context->priv_data, "allow_skip_frames", 0, &value) >= 0
        && value == 0, "frame skipping was not disabled");
    require(av_opt_get_int(context->priv_data, "profile", 0, &value) >= 0
        && value == AV_PROFILE_H264_MAIN, "main profile was not configured");
    // Resolve the named native RC value independently through the option API.
    AVCodecContext* defaults = avcodec_alloc_context3(codec);
    require(defaults != nullptr, "allocate reference option context");
    require(av_opt_set(defaults->priv_data, "rc_mode", "quality", 0) >= 0, "quality mode unavailable");
    int64_t expectedMode = -1;
    require(av_opt_get_int(defaults->priv_data, "rc_mode", 0, &expectedMode) >= 0
        && av_opt_get_int(context->priv_data, "rc_mode", 0, &value) >= 0
        && value == expectedMode, "wrong native RC mode");
    avcodec_free_context(&defaults);
    const auto expectedBitrate = static_cast<int64_t>(std::llround(160.0L * 96 * 8 * fps.num / (30.0L * fps.den)));
    require(context->bit_rate == expectedBitrate && context->rc_max_rate == expectedBitrate,
            "bitrate does not preserve the per-frame budget");
    require(context->qmin == 1 && context->qmax == qp && context->max_b_frames == 0,
            "native quality or reordering policy changed");
    require(avcodec_open2(context, codec, nullptr) >= 0, "OpenH264 did not open with native policy");
    AVFormatContext* format = avformat_alloc_context();
    require(format != nullptr, "allocate format context");
    av_dict_set(&format->metadata, "color-policy", "retained", 0);
    videohelper::tagSoftwareH264Encoder(format, context);
    const auto* tag = av_dict_get(format->metadata, "encoder", nullptr, 0);
    require(tag != nullptr && std::string(tag->value) == context->codec->name, "tag is not actual opened encoder");
    require(av_dict_get(format->metadata, "color-policy", nullptr, 0) != nullptr, "unrelated metadata removed");
    avformat_free_context(format);
    avcodec_free_context(&context);
}

// Hand-worked expectations, not read back from the planner. Frame cost is
// 61/20 bytes per pixel: 3840x2160 gives 25,297,920 bytes, so the 1 GiB budget
// holds 42 frames; 1920x1080 gives 6,324,480 bytes and 169 frames. An automatic
// thread count is 3/2 of the cores (26 cores give 39), and the
// unrestricted worst case is the preset lookahead + 3 frames per thread + 10 fixed frames.
void checkX264MemoryPlan()
{
    using videohelper::planX264Memory;
    // 4K, 26 cores: 40 + 3 * 39 + 10 = 167 > 42. Threads 42 / 6 = 7, lookahead 42 - 21 - 10 = 11.
    auto plan = planX264Memory(3840, 2160, "medium", 39);
    require(plan.threads == 7 && plan.lookahead == 11, "4K medium plan on 26 cores");
    // 1080p, 26 cores: 40 + 117 + 10 = 167 <= 169, so the encoder stays untouched.
    plan = planX264Memory(1920, 1080, "medium", 39);
    require(plan.threads == 0 && plan.lookahead == -1, "1080p must keep the automatic settings");
    // 1080p, 64 cores: 40 + 288 + 10 = 338 > 169. Threads 169 / 6 = 28; lookahead
    // 169 - 84 - 10 = 75 exceeds the preset's 40, so it is never raised.
    plan = planX264Memory(1920, 1080, "medium", 96);
    require(plan.threads == 28 && plan.lookahead == -1, "1080p on 64 cores lowers threads only");
    // veryfast looks 10 frames ahead; 4K computes 11, which is not lower.
    plan = planX264Memory(3840, 2160, "veryfast", 39);
    require(plan.threads == 7 && plan.lookahead == -1, "veryfast lookahead must not be raised");
    // ultrafast has no lookahead to lower.
    plan = planX264Memory(3840, 2160, "ultrafast", 39);
    require(plan.threads == 7 && plan.lookahead == -1, "ultrafast lookahead must stay absent");
    // A 2-core machine: auto 3 threads, 4K worst case 40 + 9 + 10 = 59 > 42.
    // Threads max(2, 7) capped by 3 gives 3 and equals auto, so unchanged (0);
    // lookahead 42 - 9 - 10 = 23.
    plan = planX264Memory(3840, 2160, "medium", 3);
    require(plan.threads == 0 && plan.lookahead == 23, "2-core 4K plan");
    // A zero budget falls back to the 16-frame floor: threads 16 / 6 = 2,
    // lookahead 16 - 6 - 10 = 0 raised to the floor of 8.
    plan = planX264Memory(3840, 2160, "medium", 39, 0);
    require(plan.threads == 2 && plan.lookahead == 8, "floor plan");
    // Unknown size keeps the defaults.
    plan = planX264Memory(0, 2160, "medium", 39);
    require(plan.threads == 0 && plan.lookahead == -1, "zero width must not plan");
    // A context that already runs 1 thread is never given more.
    plan = planX264Memory(3840, 2160, "medium", 1);
    require(plan.threads == 0 && plan.lookahead == 29, "single-thread 4K plan");
    // The thread-count resolution: 26 cores -> 39, no cores known -> 8, 200 cores -> 128 cap.
    require(videohelper::x264AutoThreads(26) == 39 && videohelper::x264AutoThreads(0) == 8
            && videohelper::x264AutoThreads(200) == 128 && videohelper::x264AutoThreads(1) == 1,
            "automatic thread resolution");
    // The preset table.
    require(videohelper::x264PresetLookahead("veryfast") == 10 && videohelper::x264PresetLookahead("medium") == 40
            && videohelper::x264PresetLookahead("slow") == 50 && videohelper::x264PresetLookahead("ultrafast") == 0,
            "preset lookahead table");
    // Whatever the machine, a restricted 4K plan stays inside the 42-frame budget.
    for (unsigned cores : { 1u, 2u, 4u, 8u, 16u, 26u, 64u, 128u, 512u })
    {
        const int autoThreads = static_cast<int>(std::min(128u, std::max(1u, cores * 3u / 2u)));
        const auto p = planX264Memory(3840, 2160, "medium", autoThreads);
        const int threads = p.threads > 0 ? p.threads : autoThreads;
        const int lookahead = p.lookahead >= 0 ? p.lookahead : 40;
        require(lookahead + 3 * threads + 10 <= 42 || (p.threads == 2 && p.lookahead == 8) || threads == 1,
                "4K plan exceeds the frame budget");
        require(threads <= autoThreads && lookahead <= 40, "plan raised a default");
    }
    // The real encoder context receives the plan when libx264 is present.
    if (const auto* x264 = avcodec_find_encoder_by_name("libx264"))
    {
        const int autoThreads = videohelper::x264AutoThreads(std::thread::hardware_concurrency());
        AVCodecContext* context = avcodec_alloc_context3(x264);
        require(context != nullptr, "allocate libx264 context");
        // The allocated default is whatever this FFmpeg picks (1 for the system
        // libavcodec). The call site must not depend on it.
        context->thread_count = 1;
        context->width = 3840;
        context->height = 2160;
        context->pix_fmt = AV_PIX_FMT_YUV420P;
        context->framerate = {30, 1};
        context->time_base = {1, 30};
        const auto error = videohelper::configureSoftwareH264(context, "libx264", 18, "medium");
        require(error.empty(), error.c_str());
        int64_t lookahead = -1;
        require(av_opt_get_int(context->priv_data, "rc-lookahead", 0, &lookahead) >= 0, "read rc-lookahead");
        if (autoThreads > 7)
        {
            // 42-frame budget: threads 42 / 6 = 7, lookahead 42 - 3 * 7 - 10 = 11.
            require(context->thread_count == 7, "4K medium context must run 7 threads");
            require(lookahead == 11, "4K medium context must look 11 frames ahead");
        }
        else
        {
            // Few cores: the automatic count (0) already sits below 7, and the
            // lookahead fills the remaining 42 - 3 * auto - 10 frames.
            require(context->thread_count == 0, "a small machine keeps automatic threads");
            require(lookahead == 42 - 3 * autoThreads - 10,
                    "4K medium lookahead on a small machine");
        }
        avcodec_free_context(&context);
        AVCodecContext* small = avcodec_alloc_context3(x264);
        require(small != nullptr, "allocate small libx264 context");
        small->thread_count = 1;
        small->width = 640;
        small->height = 360;
        require(videohelper::configureSoftwareH264(small, "libx264", 18, "medium").empty(), "configure small");
        require(small->thread_count == 0, "a small frame must run automatic threads");
        int64_t smallLookahead = 0;
        require(av_opt_get_int(small->priv_data, "rc-lookahead", 0, &smallLookahead) >= 0
                && smallLookahead == -1, "a small frame must keep the preset lookahead");
        avcodec_free_context(&small);
        // The live recorder caps its own threads at 4 so x264 holds few frames before
        // the first packet. 1080p veryfast is far inside the budget (10 + 3 * 4 + 10
        // frames of 169), so the plan leaves both the ceiling and the preset's
        // lookahead of 10 alone.
        AVCodecContext* recorder = avcodec_alloc_context3(x264);
        require(recorder != nullptr, "allocate recorder libx264 context");
        recorder->width = 1920;
        recorder->height = 1080;
        recorder->thread_count = 4;
        require(videohelper::configureSoftwareH264(recorder, "libx264", 20, "veryfast").empty(), "configure recorder");
        require(recorder->thread_count == 4, "the recorder must keep its 4-thread ceiling");
        int64_t recorderLookahead = 0;
        require(av_opt_get_int(recorder->priv_data, "rc-lookahead", 0, &recorderLookahead) >= 0
                && recorderLookahead == -1, "the recorder lookahead must be left at the preset default");
        avcodec_free_context(&recorder);
    }
}
}

int main()
{
    checkX264MemoryPlan();
    const auto* codec = avcodec_find_encoder_by_name("libopenh264");
    if (codec == nullptr)
    {
        std::cerr << "OpenH264 unavailable; native encoder policy is unverified\n";
        return 77;
    }
    require(std::string(videohelper::softwareH264EncoderName()) == "libopenh264", "OpenH264 is not preferred");
    for (const auto fps : { AVRational{24, 1}, AVRational{60, 1}, AVRational{30000, 1001} })
        for (const bool intraOnly : { false, true })
            for (const int qp : {18, 20}) checkPolicy(codec, fps, qp, intraOnly);

    AVCodecContext* context = avcodec_alloc_context3(codec);
    require(context != nullptr, "allocate invalid-input context");
    context->width = 160; context->height = 96;
    context->pix_fmt = AV_PIX_FMT_YUV420P;
    context->framerate = {30, 1};
    context->width = INT_MAX;
    require(!videohelper::configureSoftwareH264(context, codec->name, 18, "medium").empty(), "odd size accepted");
    context->width = INT_MAX - 1;
    context->height = INT_MAX - 1;
    require(!videohelper::configureSoftwareH264(context, codec->name, 18, "medium").empty(), "overflow budget accepted");
    context->width = 160; context->height = 96;
    context->framerate = {0, 1};
    require(!videohelper::configureSoftwareH264(context, codec->name, 18, "medium").empty(), "zero rate accepted");
    context->framerate = {30, 1}; context->pix_fmt = AV_PIX_FMT_YUV420P10LE;
    require(!videohelper::configureSoftwareH264(context, codec->name, 18, "medium").empty(), "HDR format accepted");
    avcodec_free_context(&context);
    std::cout << "OpenH264 native policy configured and opened at integer/fractional rates\n";
}
