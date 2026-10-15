#include "software_h264_encoder.h"

#include <iostream>
#include <stdexcept>

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
        && value == FF_PROFILE_H264_MAIN, "main profile was not configured");
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
}

int main()
{
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
