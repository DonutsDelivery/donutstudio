#include "media.h"
#include "optical_flow_source_resolver.h"
#include "sha256.h"
#include "software_h264_encoder.h"
#include "stabilization.h"

extern "C"
{
#include <libavutil/opt.h>
}

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <filesystem>
#include <unordered_set>
#include <utility>

#include "native_beat_detector.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace
{
std::string avErr (int code)
{
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror (code, buf, sizeof (buf));
    return std::string (buf);
}

double streamTimeToSec (const AVStream* st, int64_t ts)
{
    if (ts == AV_NOPTS_VALUE || st->time_base.num <= 0 || st->time_base.den <= 0)
        return 0.0;
    return (double) ts * st->time_base.num / st->time_base.den;
}

const AVHWDeviceType* preferredHwDeviceTypes()
{
    static const AVHWDeviceType preferred[] = {
#if defined(__APPLE__)
        AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
#elif defined(_WIN32)
        AV_HWDEVICE_TYPE_D3D11VA,
        AV_HWDEVICE_TYPE_CUDA,
#else
        AV_HWDEVICE_TYPE_CUDA,
        AV_HWDEVICE_TYPE_VAAPI,
#endif
        AV_HWDEVICE_TYPE_NONE
    };
    return preferred;
}

bool decoderSupportsDevice (const AVCodec* codec, AVHWDeviceType deviceType)
{
    for (int i = 0;; ++i)
    {
        const AVCodecHWConfig* cfg = avcodec_get_hw_config (codec, i);
        if (cfg == nullptr) return false;
        if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0
            && cfg->device_type == deviceType)
            return true;
    }
}

const AVCodec* findHardwareDecoder (AVCodecID codecId)
{
    for (const AVHWDeviceType* device = preferredHwDeviceTypes();
         *device != AV_HWDEVICE_TYPE_NONE; ++device)
    {
        void* iterator = nullptr;
        while (const AVCodec* candidate = av_codec_iterate (&iterator))
            if (av_codec_is_decoder (candidate) && candidate->id == codecId
                && decoderSupportsDevice (candidate, *device))
                return candidate;
    }
    return nullptr;
}
} // namespace

AVPixelFormat MediaContext::getHwFormat (AVCodecContext* context,
                                         const AVPixelFormat* formats)
{
    auto* media = static_cast<MediaContext*> (context->opaque);
    const auto desired = media != nullptr ? media->hwPixFmt_ : AV_PIX_FMT_NONE;
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format)
        if (*format == desired)
            return *format;

    // Device creation can succeed before FFmpeg checks the stream's profile,
    // bit depth, chroma, and dimensions. Record a rejected hardware format so
    // getFrame can reopen the original software decoder instead of leaving an
    // alternative native decoder in an accidental software mode.
    if (media != nullptr)
        media->hardwareFallbackRequested_.store (true, std::memory_order_release);
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format)
        if (av_pix_fmt_desc_get (*format) != nullptr
            && (av_pix_fmt_desc_get (*format)->flags & AV_PIX_FMT_FLAG_HWACCEL) == 0)
            return *format;
    return formats[0];
}

AVBufferRef* sharedCudaDeviceCtx()
{
    // Create exactly ONE CUDA device context for the whole process, lazily and
    // single-threaded (call_once). Sharing one hw_device_ctx across all decode
    // sessions — instead of av_hwdevice_ctx_create per MediaContext — avoids the
    // concurrent cuInit()/CUDA-context creation that deadlocked the driver when
    // two threads opened hw decoders at once. main() pokes this at startup so the
    // one-time init happens before any worker/render thread exists.
    static std::once_flag onceFlag;
    static AVBufferRef* ctx = nullptr;
    std::call_once (onceFlag, []
    {
        if (std::getenv ("ARBIT_DISABLE_HWDEC") != nullptr)
            return; // escape hatch: ctx stays nullptr -> software everywhere
        AVBufferRef* created = nullptr;
        if (av_hwdevice_ctx_create (&created, AV_HWDEVICE_TYPE_CUDA,
                                    nullptr, nullptr, 0) == 0)
            ctx = created; // permanent process-lifetime ref (never unref'd)
        // On failure (no NVIDIA / no CUDA in this FFmpeg build) ctx stays
        // nullptr and callers silently fall back to software decode.
    });
    return ctx;
}

bool MediaContext::tryOpenHwDecoder (const AVCodec* codec)
{
    // Each backend is tried only if this decoder, FFmpeg build, and running
    // machine support it. Failures fall through to the next backend, then software.
    for (const AVHWDeviceType* device = preferredHwDeviceTypes();
         *device != AV_HWDEVICE_TYPE_NONE; ++device)
    {
        AVPixelFormat hwFmt = AV_PIX_FMT_NONE;
        for (int i = 0; const AVCodecHWConfig* cfg = avcodec_get_hw_config (codec, i); ++i)
            if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0
                && cfg->device_type == *device)
            { hwFmt = cfg->pix_fmt; break; }
        if (hwFmt == AV_PIX_FMT_NONE) continue;

        AVBufferRef* deviceCtx = nullptr;
        if (*device == AV_HWDEVICE_TYPE_CUDA)
        {
            // Reference the ONE shared CUDA device context instead of creating a
            // new one here — av_buffer_ref only bumps a refcount, so no second
            // cuInit/CUDA-context creation runs on this thread (that concurrent
            // init is what deadlocked the driver). nullptr = unavailable or
            // ARBIT_DISABLE_HWDEC set -> fall through to the next device type
            // (then software).
            //
            // Thread-safety: FFmpeg's CUDA hwcontext makes the shared CUDA
            // context current per operation (cuCtxPushCurrent/PopCurrent) and
            // serializes access via the AVHWDeviceContext lock callbacks, so
            // multiple decode threads referencing the same hw_device_ctx is the
            // supported, recommended pattern — each MediaContext still has its
            // own AVCodecContext.
            AVBufferRef* shared = sharedCudaDeviceCtx();
            if (shared == nullptr) continue;
            deviceCtx = av_buffer_ref (shared);
            if (deviceCtx == nullptr) continue;
        }
        else if (av_hwdevice_ctx_create (&deviceCtx, *device, nullptr, nullptr, 0) < 0)
            continue;

        hwDeviceCtx_ = deviceCtx;
        hwPixFmt_ = hwFmt;
        hardwareFallbackRequested_.store (false, std::memory_order_release);
        videoDec_->hw_device_ctx = av_buffer_ref (deviceCtx);
        videoDec_->opaque = this;
        videoDec_->get_format = &MediaContext::getHwFormat;
        info_.hwaccel = av_hwdevice_get_type_name (*device);
        return true;
    }
    return false;
}

MediaContext::~MediaContext()
{
    if (lastFrame_ != nullptr) av_frame_free (&lastFrame_);
    if (hwTransferFrame_ != nullptr) av_frame_free (&hwTransferFrame_);
    if (hwDeviceCtx_ != nullptr) av_buffer_unref (&hwDeviceCtx_);
    if (sws_ != nullptr) sws_freeContext (sws_);
    if (videoDec_ != nullptr) avcodec_free_context (&videoDec_);
    if (audioDec_ != nullptr) avcodec_free_context (&audioDec_);
    if (fmt_ != nullptr) avformat_close_input (&fmt_);
}

std::string MediaContext::open (const std::string& path, bool allowHwDecode,
                                double seqFps, int seqStartNumber)
{
    path_ = path;

    // Image-sequence pattern ("img_%04d.png"): open through the image2
    // demuxer with an explicit frame rate so pts/duration are correct.
    const AVInputFormat* inputFormat = nullptr;
    AVDictionary* openOpts = nullptr;
    std::error_code pathError;
    if (path.find ('%') != std::string::npos
        && ! std::filesystem::is_regular_file (path, pathError))
    {
        inputFormat = av_find_input_format ("image2");
        char fpsBuf[32];
        std::snprintf (fpsBuf, sizeof (fpsBuf), "%.6f", seqFps > 0.0 ? seqFps : 30.0);
        av_dict_set (&openOpts, "framerate", fpsBuf, 0);
        if (seqStartNumber >= 0)
            av_dict_set_int (&openOpts, "start_number", seqStartNumber, 0);
    }

    int rc = avformat_open_input (&fmt_, path.c_str(), inputFormat, &openOpts);
    av_dict_free (&openOpts);
    if (rc < 0) return "open failed: " + avErr (rc);
    rc = avformat_find_stream_info (fmt_, nullptr);
    if (rc < 0) return "stream info failed: " + avErr (rc);

    videoStream_ = av_find_best_stream (fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    audioStream_ = av_find_best_stream (fmt_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (videoStream_ < 0 && audioStream_ < 0)
        return "no audio or video streams found";

    info_.container = fmt_->iformat != nullptr ? fmt_->iformat->name : "";
    if (fmt_->duration != AV_NOPTS_VALUE)
        info_.durationSec = (double) fmt_->duration / AV_TIME_BASE;

    if (videoStream_ >= 0)
    {
        auto* st = fmt_->streams[videoStream_];
        info_.streamIndex = videoStream_;
        info_.timeBaseNum = st->time_base.num; info_.timeBaseDen = st->time_base.den;
        const auto descriptorSar = av_guess_sample_aspect_ratio(fmt_,st,nullptr);
        info_.sarNum = descriptorSar.num > 0 && descriptorSar.den > 0 ? descriptorSar.num : 1;
        info_.sarDen = descriptorSar.num > 0 && descriptorSar.den > 0 ? descriptorSar.den : 1;

        // Alpha detection. Two cases:
        //  - the stream pix_fmt itself carries alpha (yuva420p, yuva444p10,
        //    rgba/bgra, ProRes 4444, qtrle 32-bit, PNG, ...)
        //  - webm/mkv VP8/VP9 alpha: the alpha plane travels as block
        //    additional data; the stream metadata carries alpha_mode=1 and
        //    only the libvpx decoders reassemble it (-> yuva420p).
        bool fmtHasAlpha = false;
        if (const auto* desc = av_pix_fmt_desc_get ((AVPixelFormat) st->codecpar->format))
            fmtHasAlpha = (desc->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
        bool vpxAlpha = false;
        if (st->codecpar->codec_id == AV_CODEC_ID_VP8
            || st->codecpar->codec_id == AV_CODEC_ID_VP9)
            if (const auto* e = av_dict_get (st->metadata, "alpha_mode", nullptr, 0))
                vpxAlpha = std::strcmp (e->value, "1") == 0;

        const AVCodec* softwareCodec = nullptr;
        if (vpxAlpha)
            softwareCodec = avcodec_find_decoder_by_name (
                st->codecpar->codec_id == AV_CODEC_ID_VP9 ? "libvpx-vp9" : "libvpx");
        if (softwareCodec == nullptr)
        {
            vpxAlpha = false; // native vp8/vp9 decoders drop the alpha plane
            softwareCodec = avcodec_find_decoder (st->codecpar->codec_id);
        }
        if (softwareCodec == nullptr) return "no video decoder for codec";
        softwareVideoCodec_ = softwareCodec;
        info_.hasAlpha = fmtHasAlpha || vpxAlpha;

        // avcodec_find_decoder() can prefer a software-only wrapper even when
        // FFmpeg also ships a decoder for the same codec with a hardware config.
        // AV1 commonly resolves to libdav1d this way, leaving 4K preview on the
        // CPU despite working NVDEC/VAAPI support. Pick the first decoder matching
        // our backend order when hardware decode is allowed, but retain the
        // original software decoder for a clean open-time fallback.
        const AVCodec* codec = softwareCodec;
        if (allowHwDecode && ! info_.hasAlpha)
            if (const AVCodec* hardwareCodec = findHardwareDecoder (st->codecpar->codec_id))
                codec = hardwareCodec;

        videoDec_ = avcodec_alloc_context3 (codec);
        avcodec_parameters_to_context (videoDec_, st->codecpar);
        videoDec_->thread_count = 0; // auto

        // Never hardware-decode alpha sources: hw surfaces are NV12/P010 and
        // the alpha plane would be silently dropped on download.
        const bool wantHw = allowHwDecode && ! info_.hasAlpha && tryOpenHwDecoder (codec);
        if (! wantHw && codec != softwareCodec)
        {
            // A matching FFmpeg decoder exists, but none of its advertised
            // devices opened on this machine. Do not leave playback on a slower
            // native decoder when the normal threaded software decoder is available.
            avcodec_free_context (&videoDec_);
            codec = softwareCodec;
            videoDec_ = avcodec_alloc_context3 (codec);
            avcodec_parameters_to_context (videoDec_, st->codecpar);
            videoDec_->thread_count = 0;
        }
        rc = avcodec_open2 (videoDec_, codec, nullptr);
        if (rc < 0 && wantHw)
        {
            // Hardware path failed at open — rebuild as a software decoder.
            avcodec_free_context (&videoDec_);
            if (hwDeviceCtx_ != nullptr) av_buffer_unref (&hwDeviceCtx_);
            hwPixFmt_ = AV_PIX_FMT_NONE;
            hardwareFallbackRequested_.store (false, std::memory_order_release);
            info_.hwaccel.clear();
            codec = softwareCodec;
            videoDec_ = avcodec_alloc_context3 (codec);
            avcodec_parameters_to_context (videoDec_, st->codecpar);
            videoDec_->thread_count = 0;
            rc = avcodec_open2 (videoDec_, codec, nullptr);
        }
        if (rc < 0) return "video decoder open failed: " + avErr (rc);

        info_.hasVideo = true;
        info_.width = videoDec_->width;
        info_.height = videoDec_->height;
        info_.videoCodec = codec->name;
        auto fr = av_guess_frame_rate (fmt_, st, nullptr);
        info_.fps = fr.den != 0 ? (double) fr.num / fr.den : 0.0;
        if (info_.durationSec <= 0.0 && st->duration != AV_NOPTS_VALUE)
            info_.durationSec = streamTimeToSec (st, st->duration);
    }

    if (audioStream_ >= 0)
    {
        auto* par = fmt_->streams[audioStream_]->codecpar;
        info_.hasAudio = true;
        info_.audioSampleRate = par->sample_rate;
        info_.audioChannels = par->ch_layout.nb_channels;
        const AVCodec* c = avcodec_find_decoder (par->codec_id);
        info_.audioCodec = c != nullptr ? c->name : "";
    }

    lastFrame_ = av_frame_alloc();
    return {};
}

bool MediaContext::openAudioDecoder (std::string& error)
{
    if (audioDec_ != nullptr) return true;
    if (audioStream_ < 0) { error = "media has no audio stream"; return false; }

    auto* st = fmt_->streams[audioStream_];
    const AVCodec* codec = avcodec_find_decoder (st->codecpar->codec_id);
    if (codec == nullptr) { error = "no audio decoder for codec"; return false; }
    audioDec_ = avcodec_alloc_context3 (codec);
    avcodec_parameters_to_context (audioDec_, st->codecpar);
    int rc = avcodec_open2 (audioDec_, codec, nullptr);
    if (rc < 0)
    {
        avcodec_free_context (&audioDec_);
        error = "audio decoder open failed: " + avErr (rc);
        return false;
    }
    return true;
}

void MediaContext::downloadIfHw (AVFrame* frame)
{
    if (hwPixFmt_ == AV_PIX_FMT_NONE)
        return;
    if (frame->format != hwPixFmt_)
    {
        hardwareFallbackRequested_.store (true, std::memory_order_release);
        return;
    }
    if (hwTransferFrame_ == nullptr)
        hwTransferFrame_ = av_frame_alloc();
    av_frame_unref (hwTransferFrame_);
    if (av_hwframe_transfer_data (hwTransferFrame_, frame, 0) != 0)
    {
        hardwareFallbackRequested_.store (true, std::memory_order_release);
    }
    else
    {
        av_frame_copy_props (hwTransferFrame_, frame);
        av_frame_unref (frame);
        av_frame_move_ref (frame, hwTransferFrame_);
    }
}

std::string MediaContext::reopenSoftwareVideoDecoder (double targetSec)
{
    if (softwareVideoCodec_ == nullptr)
        return "original software decoder is unavailable";

    av_frame_unref (lastFrame_);
    if (hwTransferFrame_ != nullptr)
        av_frame_unref (hwTransferFrame_);
    avcodec_free_context (&videoDec_);
    if (hwDeviceCtx_ != nullptr)
        av_buffer_unref (&hwDeviceCtx_);
    hwPixFmt_ = AV_PIX_FMT_NONE;
    hardwareFallbackRequested_.store (false, std::memory_order_release);
    info_.hwaccel.clear();

    auto* stream = fmt_->streams[videoStream_];
    videoDec_ = avcodec_alloc_context3 (softwareVideoCodec_);
    if (videoDec_ == nullptr)
        return "software decoder allocation failed";
    avcodec_parameters_to_context (videoDec_, stream->codecpar);
    videoDec_->thread_count = 0;
    const int openResult = avcodec_open2 (videoDec_, softwareVideoCodec_, nullptr);
    if (openResult < 0)
        return "software decoder open failed: " + avErr (openResult);

    const int64_t timestamp = (int64_t) std::llround (
        targetSec * stream->time_base.den / (double) stream->time_base.num);
    const int seekResult = av_seek_frame (fmt_, videoStream_, timestamp, AVSEEK_FLAG_BACKWARD);
    if (seekResult < 0)
        return "software fallback seek failed: " + avErr (seekResult);
    avcodec_flush_buffers (videoDec_);
    lastDecodedPts_ = -1.0e9;
    info_.videoCodec = softwareVideoCodec_->name;
    return {};
}

std::string MediaContext::decodeForwardUntil (double targetSec, AVFrame* frame, bool& gotFrame)
{
    gotFrame = false;
    AVPacket* pkt = av_packet_alloc();
    auto* st = fmt_->streams[videoStream_];
    const double frameDur = info_.fps > 0.0 ? 1.0 / info_.fps : 1.0 / 30.0;

    std::string error;
    bool draining = false;
    bool havePendingPacket = false;
    while (true)
    {
        // Always drain queued output before reading another packet. A prior
        // request can return as soon as it finds its target frame while the
        // decoder still holds later frames from the same accepted packet.
        int receiveResult = 0;
        while ((receiveResult = avcodec_receive_frame (videoDec_, frame)) == 0)
        {
            downloadIfHw (frame);
            gotFrame = true;
            lastDecodedPts_ = streamTimeToSec (st, frame->best_effort_timestamp);
            if (lastDecodedPts_ + frameDur > targetSec)
                break;
        }
        if (gotFrame && lastDecodedPts_ + frameDur > targetSec)
            break;
        if (receiveResult == AVERROR_EOF)
            break;
        if (receiveResult != AVERROR (EAGAIN))
        {
            if (hwPixFmt_ != AV_PIX_FMT_NONE)
                hardwareFallbackRequested_.store (true, std::memory_order_release);
            error = "receive frame failed: " + avErr (receiveResult);
            break;
        }

        if (draining)
        {
            const int sendResult = avcodec_send_packet (videoDec_, nullptr);
            if (sendResult == 0)
                continue;
            if (sendResult == AVERROR_EOF)
                break;
            if (hwPixFmt_ != AV_PIX_FMT_NONE)
                hardwareFallbackRequested_.store (true, std::memory_order_release);
            error = "flush packet failed: " + avErr (sendResult);
            break;
        }

        if (! havePendingPacket)
        {
            int readResult = 0;
            while (true)
            {
                readResult = av_read_frame (fmt_, pkt);
                if (readResult < 0 || pkt->stream_index == videoStream_)
                    break;
                av_packet_unref (pkt);
            }

            if (readResult == AVERROR_EOF)
            {
                draining = true;
                continue;
            }
            if (readResult < 0)
            {
                error = "read frame failed: " + avErr (readResult);
                break;
            }
            havePendingPacket = true;
        }

        const int sendResult = avcodec_send_packet (videoDec_, pkt);
        if (sendResult == AVERROR (EAGAIN))
        {
            // The packet was not accepted. Keep it intact, drain output, and
            // retry instead of silently skipping compressed input.
            continue;
        }
        av_packet_unref (pkt);
        havePendingPacket = false;
        if (sendResult < 0)
        {
            if (hwPixFmt_ != AV_PIX_FMT_NONE)
                hardwareFallbackRequested_.store (true, std::memory_order_release);
            error = "send packet failed: " + avErr (sendResult);
            break;
        }
    }

    av_packet_free (&pkt);
    return error;
}

// decodedframecolor::fromCicp takes H.273 code points; these pin the FFmpeg values it relies on.
static_assert(AVCOL_PRI_BT709 == 1 && AVCOL_PRI_BT2020 == 9 && AVCOL_PRI_SMPTE432 == 12,
              "fromCicp primaries codes must match FFmpeg");
static_assert(AVCOL_TRC_BT709 == 1 && AVCOL_TRC_GAMMA22 == 4 && AVCOL_TRC_LINEAR == 8
              && AVCOL_TRC_IEC61966_2_1 == 13 && AVCOL_TRC_BT2020_10 == 14
              && AVCOL_TRC_BT2020_12 == 15 && AVCOL_TRC_SMPTE2084 == 16
              && AVCOL_TRC_ARIB_STD_B67 == 18, "fromCicp transfer codes must match FFmpeg");

std::string MediaContext::scaleToRgba (AVFrame* frame, int maxW, int maxH, DecodedFrame& out)
{
    out.setPresentationTimestamp (frame->best_effort_timestamp,
                                  fmt_->streams[videoStream_]->time_base);
    const auto primaries = frame->color_primaries == AVCOL_PRI_UNSPECIFIED
        ? videoDec_->color_primaries : frame->color_primaries;
    const auto transfer = frame->color_trc == AVCOL_TRC_UNSPECIFIED
        ? videoDec_->color_trc : frame->color_trc;
    // The shared CICP mapping keeps the declared primaries and transfer; codes it
    // does not know stay Unspecified instead of defaulting to sRGB.
    out.color = decodedframecolor::fromCicp(static_cast<int>(primaries), static_cast<int>(transfer));
    if (colortransform::isHdrTransfer(out.color.transfer))
    {
        out.rgba.clear(); out.width = out.height = out.strideBytes = 0;
        return std::string("PQ/HLG decoded video requires a float HDR decode and display/output path; RGBA8 conversion is unavailable: ")
            + std::string(colortransform::token(colortransform::AdmissionFailure::UnsupportedPixelFormat));
    }
    int dstW = frame->width;
    int dstH = frame->height;
    if (maxW > 0 && maxH > 0 && (dstW > maxW || dstH > maxH))
    {
        const double s = std::min ((double) maxW / dstW, (double) maxH / dstH);
        dstW = std::max (2, (int) std::lround (dstW * s) & ~1);
        dstH = std::max (2, (int) std::lround (dstH * s) & ~1);
    }

    sws_ = sws_getCachedContext (sws_, frame->width, frame->height,
                                 (AVPixelFormat) frame->format,
                                 dstW, dstH, AV_PIX_FMT_RGBA,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (sws_ == nullptr) return "swscale context failed";
    const auto matrix = frame->colorspace == AVCOL_SPC_UNSPECIFIED ? videoDec_->colorspace : frame->colorspace;
    const int coefficients = matrix == AVCOL_SPC_BT709 ? SWS_CS_ITU709
        : matrix == AVCOL_SPC_BT2020_NCL ? SWS_CS_BT2020 : SWS_CS_DEFAULT;
    const auto range = frame->color_range == AVCOL_RANGE_UNSPECIFIED ? videoDec_->color_range : frame->color_range;
    if (sws_setColorspaceDetails(sws_,sws_getCoefficients(coefficients),range == AVCOL_RANGE_JPEG,
        sws_getCoefficients(coefficients),1,0,1 << 16,1 << 16) < 0)
        return "Decoded Frame color matrix/range declaration is unavailable";

    out.width = dstW;
    out.height = dstH;
    out.strideBytes = dstW * 4;
    out.rgba.resize ((size_t) out.strideBytes * dstH);

    uint8_t* dstData[4] = { out.rgba.data(), nullptr, nullptr, nullptr };
    int dstLines[4] = { out.strideBytes, 0, 0, 0 };
    sws_scale (sws_, frame->data, frame->linesize, 0, frame->height, dstData, dstLines);
    return {};
}

std::string MediaContext::scaleToHdrRgba (AVFrame* frame, int maxW, int maxH,
                                          DecodedHdrFrame& out)
{
    const auto primaries = frame->color_primaries == AVCOL_PRI_UNSPECIFIED
        ? videoDec_->color_primaries : frame->color_primaries;
    const auto transfer = frame->color_trc == AVCOL_TRC_UNSPECIFIED
        ? videoDec_->color_trc : frame->color_trc;
    const auto matrix = frame->colorspace == AVCOL_SPC_UNSPECIFIED
        ? videoDec_->colorspace : frame->colorspace;
    const auto range = frame->color_range == AVCOL_RANGE_UNSPECIFIED
        ? videoDec_->color_range : frame->color_range;
    if (primaries != AVCOL_PRI_BT2020
        || (transfer != AVCOL_TRC_SMPTE2084 && transfer != AVCOL_TRC_ARIB_STD_B67)
        || matrix != AVCOL_SPC_BT2020_NCL
        || (range != AVCOL_RANGE_MPEG && range != AVCOL_RANGE_JPEG))
        return "float HDR decode requires declared Rec.2020 primaries, BT.2020 NCL matrix, range, and PQ or HLG transfer";

    const auto sourceFormat = static_cast<AVPixelFormat> (frame->format);
    const auto* description = av_pix_fmt_desc_get (sourceFormat);
    if (description == nullptr || (description->flags & AV_PIX_FMT_FLAG_ALPHA) != 0)
        return "float HDR decode does not support this source pixel format or alpha";
    if (! sws_isSupportedInput (sourceFormat) || ! sws_isSupportedOutput (AV_PIX_FMT_GBRPF32))
        return "float HDR conversion is unavailable in this FFmpeg build";
    if (frame->width <= 0 || frame->height <= 0)
        return "invalid float HDR frame dimensions";

    int dstW = frame->width;
    int dstH = frame->height;
    if (maxW > 0 && maxH > 0 && (dstW > maxW || dstH > maxH))
    {
        const double scale = std::min ((double) maxW / dstW, (double) maxH / dstH);
        dstW = std::max (2, (int) std::lround (dstW * scale) & ~1);
        dstH = std::max (2, (int) std::lround (dstH * scale) & ~1);
    }
    // Cap the owned RGBA allocation at 1 GiB, including 8K UHD frames.
    constexpr size_t maxPixels = (size_t) 1024 * 1024 * 1024 / (4 * sizeof (float));
    const size_t pixels = (size_t) dstW * (size_t) dstH;
    if (dstW <= 0 || dstH <= 0 || pixels > maxPixels || dstW > INT_MAX / (int) sizeof (float))
        return "float HDR frame exceeds decode bounds";

    SwsContext* scaler = sws_getContext (frame->width, frame->height, sourceFormat,
                                         dstW, dstH, AV_PIX_FMT_GBRPF32,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (scaler == nullptr) return "float HDR swscale context failed";
    const int* coefficients = sws_getCoefficients (SWS_CS_BT2020);
    const int configured = sws_setColorspaceDetails (scaler, coefficients,
        range == AVCOL_RANGE_JPEG, coefficients, 1, 0, 1 << 16, 1 << 16);
    if (configured < 0)
    {
        sws_freeContext (scaler);
        return "float HDR BT.2020 matrix/range conversion is unavailable";
    }

    std::vector<float> planar (pixels * 3);
    DecodedHdrFrame decoded;
    decoded.rgba.resize (pixels * 4);
    const int planeStride = dstW * (int) sizeof (float);
    uint8_t* dstData[4] = {
        reinterpret_cast<uint8_t*> (planar.data()),
        reinterpret_cast<uint8_t*> (planar.data() + pixels),
        reinterpret_cast<uint8_t*> (planar.data() + pixels * 2), nullptr
    };
    int dstLines[4] = { planeStride, planeStride, planeStride, 0 };
    const int rows = sws_scale (scaler, frame->data, frame->linesize, 0,
                                frame->height, dstData, dstLines);
    sws_freeContext (scaler);
    if (rows != dstH) return "float HDR swscale conversion failed";

    for (size_t pixel = 0; pixel < pixels; ++pixel)
    {
        const float red = planar[pixel + pixels * 2];
        const float green = planar[pixel];
        const float blue = planar[pixel + pixels];
        if (! std::isfinite (red) || ! std::isfinite (green) || ! std::isfinite (blue))
            return "float HDR conversion produced non-finite code values";
        decoded.rgba[pixel * 4] = red;
        decoded.rgba[pixel * 4 + 1] = green;
        decoded.rgba[pixel * 4 + 2] = blue;
        decoded.rgba[pixel * 4 + 3] = 1.0f;
    }
    decoded.width = dstW;
    decoded.height = dstH;
    decoded.strideFloats = dstW * 4;
    decoded.ptsSec = lastDecodedPts_;
    decoded.timeBase = fmt_->streams[videoStream_]->time_base;
    if (decoded.timeBase.num <= 0 || decoded.timeBase.den <= 0)
        decoded.timeBase = { 0, 1 };
    decoded.ptsTicks = decoded.timeBase.num > 0
        ? frame->best_effort_timestamp : AV_NOPTS_VALUE;
    decoded.color.primaries = colortransform::ColorSpace::Rec2020;
    decoded.color.transfer = transfer == AVCOL_TRC_SMPTE2084
        ? colortransform::TransferFunction::PQ : colortransform::TransferFunction::HLG;
    decoded.matrix = matrix;
    decoded.range = range;
    out = std::move (decoded);
    return {};
}

std::string MediaContext::getFrame (double timeSec, int maxW, int maxH, DecodedFrame& out)
{
    std::lock_guard<std::mutex> lock (mutex_);
    const auto error = selectFrame (timeSec);
    if (! error.empty()) return error;
    out.ptsSec = lastDecodedPts_;
    return scaleToRgba (lastFrame_, maxW, maxH, out);
}

std::string MediaContext::getHdrFrame (double timeSec, int maxW, int maxH, DecodedHdrFrame& out)
{
    std::lock_guard<std::mutex> lock (mutex_);
    out = {};
    const auto error = selectFrame (timeSec);
    if (! error.empty()) return error;
    return scaleToHdrRgba (lastFrame_, maxW, maxH, out);
}

std::string MediaContext::selectFrame (double timeSec)
{
    if (! info_.hasVideo) return "media has no video stream";

    timeSec = std::clamp (timeSec, 0.0, std::max (0.0, info_.durationSec - 0.001));
    const double frameDur = info_.fps > 0.0 ? 1.0 / info_.fps : 1.0 / 30.0;

    // Already showing the right frame?
    if (lastFrame_->width > 0 && timeSec >= lastDecodedPts_
        && timeSec < lastDecodedPts_ + frameDur)
        return {};

    // Backward jump, or forward jump beyond ~1s: seek to keyframe before target.
    const bool sequential = timeSec > lastDecodedPts_ && timeSec < lastDecodedPts_ + 1.0;
    if (! sequential)
    {
        auto* st = fmt_->streams[videoStream_];
        const int64_t ts = (int64_t) std::llround (timeSec * st->time_base.den
                                                   / (double) st->time_base.num);
        int rc = av_seek_frame (fmt_, videoStream_, ts, AVSEEK_FLAG_BACKWARD);
        if (rc < 0) return "seek failed: " + avErr (rc);
        avcodec_flush_buffers (videoDec_);
        lastDecodedPts_ = -1.0e9;
    }

    bool gotFrame = false;
    auto error = decodeForwardUntil (timeSec, lastFrame_, gotFrame);
    if (hwPixFmt_ != AV_PIX_FMT_NONE
        && hardwareFallbackRequested_.exchange (false, std::memory_order_acq_rel))
    {
        std::fprintf (stderr,
                      "[media] hardware decode rejected at frame decode; reopening with %s\n",
                      softwareVideoCodec_ != nullptr ? softwareVideoCodec_->name : "software");
        const auto fallbackError = reopenSoftwareVideoDecoder (timeSec);
        if (! fallbackError.empty())
            return "hardware decode failed; " + fallbackError;
        gotFrame = false;
        error = decodeForwardUntil (timeSec, lastFrame_, gotFrame);
    }
    if (! error.empty()) return error;
    if (! gotFrame) return "no frame decoded at requested time";
    return {};
}

std::string MediaContext::decodePresentationFrame (
    double timeSec, int maxW, int maxH, DecodedFrame& out, uint64_t& ordinal)
{
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* selected = av_frame_alloc();
    if (packet == nullptr || frame == nullptr || selected == nullptr)
    {
        av_packet_free (&packet);
        av_frame_free (&frame);
        av_frame_free (&selected);
        return "Optical Flow presentation scan allocation failed";
    }

    uint64_t nextOrdinal = 0;
    bool found = false;
    std::unordered_set<int64_t> timestamps;
    std::string error;
    const auto drain = [&] () -> int
    {
        int result = 0;
        while ((result = avcodec_receive_frame (videoDec_, frame)) == 0)
        {
            const auto ticks = frame->best_effort_timestamp;
            if (ticks == AV_NOPTS_VALUE || ! timestamps.insert (ticks).second)
                error = "Optical Flow presentation scan has absent or duplicate timestamps";
            else
            {
                const double frameSec = streamTimeToSec (fmt_->streams[videoStream_], ticks);
                const bool atOrBefore = frameSec <= timeSec;
                if (! found || atOrBefore)
                {
                    av_frame_unref (selected);
                    if (av_frame_ref (selected, frame) < 0)
                        error = "Optical Flow presentation frame copy failed";
                    else
                    {
                        ordinal = nextOrdinal;
                        found = true;
                    }
                }
            }
            ++nextOrdinal;
            av_frame_unref (frame);
            if (! error.empty()) break;
        }
        return result;
    };

    while (error.empty())
    {
        const int read = av_read_frame (fmt_, packet);
        if (read == AVERROR_EOF) break;
        if (read < 0) { error = "Optical Flow presentation scan read failed: " + avErr (read); break; }
        if (packet->stream_index == videoStream_)
        {
            int sent = avcodec_send_packet (videoDec_, packet);
            if (sent == AVERROR (EAGAIN))
            {
                drain();
                if (error.empty()) sent = avcodec_send_packet (videoDec_, packet);
            }
            if (sent < 0 && error.empty())
                error = "Optical Flow presentation scan send failed: " + avErr (sent);
            if (error.empty())
            {
                const int received = drain();
                if (error.empty() && received != AVERROR (EAGAIN) && received != AVERROR_EOF)
                    error = "Optical Flow presentation scan receive failed: " + avErr (received);
            }
        }
        av_packet_unref (packet);
    }
    if (error.empty())
    {
        const int sent = avcodec_send_packet (videoDec_, nullptr);
        if (sent < 0 && sent != AVERROR_EOF)
            error = "Optical Flow presentation scan flush failed: " + avErr (sent);
        else
        {
            const int received = drain();
            if (error.empty() && received != AVERROR_EOF && received != AVERROR (EAGAIN))
                error = "Optical Flow presentation scan drain failed: " + avErr (received);
        }
    }
    av_packet_free (&packet);
    av_frame_free (&frame);
    if (error.empty() && found)
        error = scaleToRgba (selected, maxW, maxH, out);
    av_frame_free (&selected);
    if (! error.empty()) return error;
    return found ? std::string {} : "Optical Flow presentation scan found no video frame";
}

std::string MediaContext::getFrameWithPresentationOrdinal (
    double timeSec, int maxW, int maxH, DecodedFrame& out, uint64_t& ordinal)
{
    std::error_code pathError;
    if (! std::filesystem::is_regular_file (path_, pathError))
        return "Optical Flow presentation ordinal for image sequences is unavailable";
    MediaContext scanner;
    auto error = scanner.open (path_, false);
    if (! error.empty()) return "Optical Flow presentation scan: " + error;
    if (scanner.videoStream_ < 0)
        return "Optical Flow presentation scan has no video stream";
    return scanner.decodePresentationFrame (timeSec, maxW, maxH, out, ordinal);
}

namespace videoopticalflow
{
namespace
{
std::string fileSha256 (const std::string& path)
{
    std::ifstream input (path, std::ios::binary);
    if (! input) return {};
    videohelper::Sha256 hash;
    std::array<char, 65536> bytes {};
    while (input)
    {
        input.read (bytes.data(), bytes.size());
        const auto count = input.gcount();
        if (count > 0) hash.update (bytes.data(), static_cast<std::size_t> (count));
    }
    return input.eof() ? hash.finishHex() : std::string {};
}
}

std::string resolveDecodedPair (
    const videowire::CompiledVisualLayerPlan& plan,
    const opticalflowoperation::SourcePair& orderedPair,
    const std::vector<videowire::OpticalFlowSourceBinding>& catalog,
    const std::vector<videowire::RenderSegment>& segments,
    std::uint64_t snapshotGeneration, double displaySec,
    int maxW, int maxH, std::shared_ptr<const DecodedPairReceipt>& out)
{
    out.reset();
    if (! plan.producerValidated || plan.clipId <= 0 || plan.structuralRevision == 0
        || plan.identityMode != "authoredGraph"
        || snapshotGeneration == 0 || ! std::isfinite (displaySec)
        || maxW <= 0 || maxH <= 0 || catalog.size() != 2
        || orderedPair.images[0] == orderedPair.images[1])
        return "Optical Flow decoded pair has an invalid selection";
    const auto flow = std::find_if (plan.operations.begin(), plan.operations.end(), [] (const auto& op)
        { return op.kind == "visual.optical-flow"; });
    opticalflowoperation::SourcePair savedPair;
    if (flow == plan.operations.end()
        || ! opticalflowoperation::parseSourcePair (flow->payloadXml, savedPair)
        || savedPair.images != orderedPair.images)
        return "Optical Flow ordered pair changed after validation";

    DecodedPairReceipt candidate;
    candidate.snapshotGeneration = snapshotGeneration;
    for (std::size_t index = 0; index < 2; ++index)
    {
        const auto& endpoint = orderedPair.images[index];
        const auto operation = std::find_if (plan.operations.begin(), plan.operations.end(),
            [&] (const auto& op) { return op.nodeId == static_cast<int> (endpoint.nodeId); });
        std::uint32_t clipId = 0;
        if (operation == plan.operations.end()
            || (operation->kind != "video.source" && operation->kind != "video.layer.source")
            || ! opticalflowoperation::parseSourceClip (operation->payloadXml, clipId)
            || (operation->kind == "video.source" && clipId != static_cast<std::uint32_t> (plan.clipId))
            || (operation->kind == "video.layer.source" && clipId == static_cast<std::uint32_t> (plan.clipId))
            || (index == 1 && clipId == candidate.sources[0].clipId))
            return "Optical Flow saved source endpoint changed after validation";
        const auto binding = std::find_if (catalog.begin(), catalog.end(),
            [clipId] (const auto& value) { return value.clipId == static_cast<int> (clipId); });
        if (binding == catalog.end()
            || std::count_if (catalog.begin(), catalog.end(), [clipId] (const auto& value)
                { return value.clipId == static_cast<int> (clipId); }) != 1
            || binding->sourceKind != videowire::SourceKind::Media
            || binding->sourceGeneration == 0
            || binding->snapshotGeneration != snapshotGeneration)
            return "Optical Flow saved source catalog is missing or stale";
        const auto* segment = videowire::resolveClipSegmentAtDisplayTime (
            segments, static_cast<int> (clipId), displaySec);
        if (segment == nullptr || segment->sourceKind != videowire::SourceKind::Media
            || segment->sourcePath != binding->sourcePath)
            return "Optical Flow saved media segment is missing or stale";
        const double sourceSec = segment->inSec
            + (displaySec - segment->displayStartSec) * segment->rate;
        if (! std::isfinite (sourceSec) || sourceSec < 0.0 || sourceSec >= segment->outSec)
            return "Optical Flow source time is outside its saved segment";
        std::error_code fileError;
        if (! std::filesystem::is_regular_file (binding->sourcePath, fileError))
            return "Optical Flow saved media file is unavailable";
        const auto sourceDigest = fileSha256 (binding->sourcePath);
        if (sourceDigest.empty()) return "Optical Flow saved media content could not be hashed";

        MediaContext decoder;
        const auto openError = decoder.open (binding->sourcePath, false);
        if (! openError.empty()) return "Optical Flow source decode open failed: " + openError;
        if (! decoder.info().hasVideo || decoder.info().durationSec <= 0.0
            || sourceSec >= decoder.info().durationSec)
            return "Optical Flow source time exceeds the media content";
        auto& receipt = candidate.sources[index];
        receipt.clipId = clipId;
        receipt.sourceGeneration = binding->sourceGeneration;
        receipt.sourceTimeSec = sourceSec;
        receipt.sourceSha256 = sourceDigest;
        const auto decodeError = decoder.getFrameWithPresentationOrdinal (
            sourceSec, maxW, maxH, receipt.frame, receipt.presentationOrdinal);
        if (! decodeError.empty()) return decodeError;
        if (receipt.frame.width <= 0 || receipt.frame.height <= 0
            || receipt.frame.strideBytes != receipt.frame.width * 4
            || receipt.frame.rgba.size() != static_cast<std::size_t> (receipt.frame.strideBytes)
                                            * static_cast<std::size_t> (receipt.frame.height)
            || receipt.frame.color.primaries == colortransform::ColorSpace::Unspecified
            || receipt.frame.color.transfer == colortransform::TransferFunction::Unspecified)
            return "Optical Flow decoded source has unsupported color or pixel layout";
        receipt.pixelSha256 = videohelper::sha256Bytes (
            receipt.frame.rgba.data(), receipt.frame.rgba.size());
        if (fileSha256 (binding->sourcePath) != sourceDigest)
            return "Optical Flow saved media content changed during decode";
    }
    if (candidate.sources[0].frame.width != candidate.sources[1].frame.width
        || candidate.sources[0].frame.height != candidate.sources[1].frame.height)
        return "Optical Flow decoded source extents differ";
    out = std::make_shared<const DecodedPairReceipt> (std::move (candidate));
    return {};
}
} // namespace videoopticalflow

// ============================================================================
// Audio extraction (float32 WAV)
// ============================================================================

namespace
{
struct WavWriter
{
    FILE* f = nullptr;
    uint32_t dataBytes = 0;
    int channels = 0, sampleRate = 0;

    bool start (const std::string& path, int ch, int rate)
    {
        channels = ch; sampleRate = rate;
        f = std::fopen (path.c_str(), "wb");
        if (f == nullptr) return false;
        uint8_t hdr[44] = {};
        std::fwrite (hdr, 1, 44, f); // placeholder, patched in finish()
        return true;
    }

    void write (const float* interleaved, size_t frames)
    {
        const size_t bytes = frames * (size_t) channels * sizeof (float);
        std::fwrite (interleaved, 1, bytes, f);
        dataBytes += (uint32_t) bytes;
    }

    void finish()
    {
        auto put32 = [this] (uint32_t v) { std::fwrite (&v, 4, 1, f); };
        auto put16 = [this] (uint16_t v) { std::fwrite (&v, 2, 1, f); };
        std::fseek (f, 0, SEEK_SET);
        std::fwrite ("RIFF", 1, 4, f); put32 (36 + dataBytes);
        std::fwrite ("WAVE", 1, 4, f);
        std::fwrite ("fmt ", 1, 4, f); put32 (16);
        put16 (3 /*IEEE float*/); put16 ((uint16_t) channels);
        put32 ((uint32_t) sampleRate);
        put32 ((uint32_t) (sampleRate * channels * 4));
        put16 ((uint16_t) (channels * 4)); put16 (32);
        std::fwrite ("data", 1, 4, f); put32 (dataBytes);
        std::fclose (f);
        f = nullptr;
    }
};
} // namespace

std::string MediaContext::extractAudio (const std::string& outPath, double& durationSecOut,
                                        int& sampleRateOut, int& channelsOut)
{
    std::lock_guard<std::mutex> lock (mutex_);
    std::string error;
    if (! openAudioDecoder (error)) return error;

    auto* st = fmt_->streams[audioStream_];
    const int outChannels = std::min (2, std::max (1, audioDec_->ch_layout.nb_channels));
    const int outRate = audioDec_->sample_rate;

    SwrContext* swr = nullptr;
    AVChannelLayout outLayout;
    av_channel_layout_default (&outLayout, outChannels);
    int rc = swr_alloc_set_opts2 (&swr, &outLayout, AV_SAMPLE_FMT_FLT, outRate,
                                  &audioDec_->ch_layout, audioDec_->sample_fmt,
                                  audioDec_->sample_rate, 0, nullptr);
    if (rc < 0 || swr_init (swr) < 0) return "resampler init failed";

    WavWriter wav;
    if (! wav.start (outPath, outChannels, outRate))
    {
        swr_free (&swr);
        return "cannot write " + outPath;
    }

    // Decode the audio stream from the start with an independent read position:
    // use a second format context so the video decode state is untouched.
    AVFormatContext* afmt = nullptr;
    if (avformat_open_input (&afmt, path_.c_str(), nullptr, nullptr) < 0
        || avformat_find_stream_info (afmt, nullptr) < 0)
    {
        if (afmt != nullptr) avformat_close_input (&afmt);
        swr_free (&swr);
        wav.finish();
        return "reopen for audio failed";
    }

    avcodec_flush_buffers (audioDec_);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<float> buf;

    auto drainFrames = [&]()
    {
        while (avcodec_receive_frame (audioDec_, frame) == 0)
        {
            const int maxOut = swr_get_out_samples (swr, frame->nb_samples);
            buf.resize ((size_t) maxOut * outChannels);
            uint8_t* outPtr = reinterpret_cast<uint8_t*> (buf.data());
            const int got = swr_convert (swr, &outPtr, maxOut,
                                         (const uint8_t**) frame->extended_data,
                                         frame->nb_samples);
            if (got > 0) wav.write (buf.data(), (size_t) got);
        }
    };

    while (av_read_frame (afmt, pkt) >= 0)
    {
        if (pkt->stream_index == audioStream_)
        {
            if (avcodec_send_packet (audioDec_, pkt) == 0)
                drainFrames();
        }
        av_packet_unref (pkt);
    }
    avcodec_send_packet (audioDec_, nullptr);
    drainFrames();
    // flush resampler
    {
        const int maxOut = swr_get_out_samples (swr, 0);
        if (maxOut > 0)
        {
            buf.resize ((size_t) maxOut * outChannels);
            uint8_t* outPtr = reinterpret_cast<uint8_t*> (buf.data());
            const int got = swr_convert (swr, &outPtr, maxOut, nullptr, 0);
            if (got > 0) wav.write (buf.data(), (size_t) got);
        }
    }

    const uint32_t frames = wav.dataBytes / (uint32_t) (outChannels * sizeof (float));
    wav.finish();
    durationSecOut = outRate > 0 ? (double) frames / outRate : 0.0;
    sampleRateOut = outRate;
    channelsOut = outChannels;

    av_frame_free (&frame);
    av_packet_free (&pkt);
    avformat_close_input (&afmt);
    swr_free (&swr);
    avcodec_flush_buffers (audioDec_);
    (void) st;
    return {};
}

// ============================================================================
// Beat detection (owned constant-tempo fallback)
// ============================================================================

std::string MediaContext::decodeAudioMono (int targetRate, std::vector<float>& monoOut)
{
    std::string error;
    if (! openAudioDecoder (error)) return error;

    SwrContext* swr = nullptr;
    AVChannelLayout monoLayout = AV_CHANNEL_LAYOUT_MONO;
    int rc = swr_alloc_set_opts2 (&swr, &monoLayout, AV_SAMPLE_FMT_FLT, targetRate,
                                  &audioDec_->ch_layout, audioDec_->sample_fmt,
                                  audioDec_->sample_rate, 0, nullptr);
    if (rc < 0 || swr_init (swr) < 0)
    {
        swr_free (&swr);
        return "resampler init failed";
    }

    AVFormatContext* afmt = nullptr;
    if (avformat_open_input (&afmt, path_.c_str(), nullptr, nullptr) < 0
        || avformat_find_stream_info (afmt, nullptr) < 0)
    {
        if (afmt != nullptr) avformat_close_input (&afmt);
        swr_free (&swr);
        return "reopen for audio failed";
    }

    avcodec_flush_buffers (audioDec_);
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<float> buf;
    if (pkt == nullptr || frame == nullptr)
        error = "audio decode allocation failed";

    auto convert = [&](const uint8_t** input, int samples) -> int
    {
        const int maxOut = swr_get_out_samples (swr, samples);
        if (maxOut < 0) return maxOut;
        buf.resize ((size_t) maxOut);
        uint8_t* outPtr = reinterpret_cast<uint8_t*> (buf.data());
        const int got = swr_convert (swr, &outPtr, maxOut, input, samples);
        if (got > 0) monoOut.insert (monoOut.end(), buf.begin(), buf.begin() + got);
        return got;
    };
    auto drain = [&]() -> int
    {
        int result = 0;
        while ((result = avcodec_receive_frame (audioDec_, frame)) == 0)
        {
            const int got = convert ((const uint8_t**) frame->extended_data,
                                     frame->nb_samples);
            av_frame_unref (frame);
            if (got < 0) return got;
        }
        return result == AVERROR (EAGAIN) || result == AVERROR_EOF ? 0 : result;
    };

    if (error.empty())
    {
        while ((rc = av_read_frame (afmt, pkt)) >= 0)
        {
            if (pkt->stream_index == audioStream_)
            {
                int decoded = avcodec_send_packet (audioDec_, pkt);
                if (decoded >= 0) decoded = drain();
                if (decoded < 0) error = "audio decode failed: " + avErr (decoded);
            }
            av_packet_unref (pkt);
            if (! error.empty()) break;
        }
        if (error.empty() && rc != AVERROR_EOF)
            error = "audio read failed: " + avErr (rc);
        if (error.empty())
        {
            rc = avcodec_send_packet (audioDec_, nullptr);
            if (rc >= 0) rc = drain();
            if (rc < 0) error = "audio decode failed: " + avErr (rc);
        }
        if (error.empty())
            while ((rc = convert (nullptr, 0)) > 0) {}
        if (error.empty() && rc < 0)
            error = "audio resample failed: " + avErr (rc);
    }

    av_frame_free (&frame);
    av_packet_free (&pkt);
    avformat_close_input (&afmt);
    swr_free (&swr);
    avcodec_flush_buffers (audioDec_);
    return error;
}

std::string MediaContext::detectBeats (double& bpmOut, double& confidenceOut,
                                       std::vector<double>& beatTimesOut)
{
    std::lock_guard<std::mutex> lock (mutex_);
    bpmOut = 0.0;
    confidenceOut = 0.0;
    beatTimesOut.clear();
    constexpr int rate = 44100;
    std::vector<float> mono;
    auto error = decodeAudioMono (rate, mono);
    if (! error.empty()) return error;

    arbitnativebeat::Result result;
    error = arbitnativebeat::analyze (mono, rate, result);
    if (! error.empty()) return error;
    bpmOut = result.bpm;
    confidenceOut = result.confidence;
    beatTimesOut = std::move (result.beats);
    return {};
}

// ============================================================================
// Stabilization: owned rigid analysis and sequential render replay
// ============================================================================
namespace
{
namespace stab = videohelper::stabilization;
bool stabCancelled (StabilizationProgress* p) { return p && p->abort.load(); }
bool stabHdrTransfer (AVColorTransferCharacteristic transfer)
{
    return transfer == AVCOL_TRC_SMPTE2084 || transfer == AVCOL_TRC_ARIB_STD_B67;
}
bool stabAlphaFormat (int format)
{
    const auto* description = av_pix_fmt_desc_get(AVPixelFormat(format));
    return description && (description->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
}
std::string stabUnsupportedSource (const AVStream* stream, const AVCodecContext* decoder,
                                   const AVFrame* frame = nullptr)
{
    // A frame can introduce transfer/format declarations absent from the
    // container. Check both before any conversion to the SDR-only intermediate.
    if (stabHdrTransfer(stream->codecpar->color_trc)
        || (decoder && stabHdrTransfer(decoder->color_trc))
        || (frame && stabHdrTransfer(frame->color_trc)))
        return "stabilization HDR preservation is not implemented for PQ/HLG sources";
    const auto* alpha = av_dict_get(stream->metadata,"alpha_mode",nullptr,0);
    if (stabAlphaFormat(stream->codecpar->format) || (decoder && stabAlphaFormat(decoder->pix_fmt))
        || (frame && stabAlphaFormat(frame->format))
        || (alpha && std::strcmp(alpha->value,"1") == 0))
        return "stabilization alpha preservation is not implemented for alpha-bearing sources";
    return {};
}
std::string stabDigestHex (const std::array<uint8_t,32>& digest)
{
    static const char hex[] = "0123456789abcdef";
    std::string result(64,'0');
    for (size_t i = 0; i < digest.size(); ++i)
    { result[i*2] = hex[digest[i] >> 4]; result[i*2+1] = hex[digest[i] & 15]; }
    return result;
}
bool stabParseDigest (const std::string& hex, std::array<uint8_t,32>& result)
{
    if (hex.size() != 64) return false;
    const auto digit = [](char c) { return c >= '0' && c <= '9' ? c-'0' : c >= 'a' && c <= 'f' ? c-'a'+10 : -1; };
    for (size_t i = 0; i < result.size(); ++i)
    {
        const int a = digit(hex[i*2]), b = digit(hex[i*2+1]);
        if (a < 0 || b < 0) return false;
        result[i] = uint8_t(a*16+b);
    }
    return true;
}
std::string stabFileDigest (const std::string& path, std::string& result, StabilizationProgress* progress)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(std::filesystem::u8path(path),ec) || ec)
        return "stabilization requires a regular source/artifact file";
    std::ifstream input(std::filesystem::u8path(path),std::ios::binary);
    if (!input) return "cannot read stabilization source/artifact";
    videohelper::Sha256 hash; std::array<char,65536> buffer;
    while (input)
    {
        if (stabCancelled(progress)) return "cancelled";
        input.read(buffer.data(),buffer.size());
        const auto count = input.gcount();
        if (count > 0) hash.update(buffer.data(),size_t(count));
    }
    if (!input.eof()) return "stabilization source/artifact read failed";
    result = hash.finishHex(); return {};
}
bool stabSamePath (const std::string& a, const std::string& b)
{
    std::error_code ecA, ecB;
    const auto pa = std::filesystem::weakly_canonical(std::filesystem::u8path(a),ecA);
    const auto pb = std::filesystem::weakly_canonical(std::filesystem::u8path(b),ecB);
    if (!ecA && !ecB && pa == pb) return true;
    std::error_code equivalentError;
    return std::filesystem::equivalent(std::filesystem::u8path(a),std::filesystem::u8path(b),equivalentError)
        && !equivalentError;
}
struct StabScratch
{
    std::string path;
    explicit StabScratch (const std::string& destination)
    {
        static std::atomic<uint64_t> sequence {0};
        path = destination+".stab-part-"+std::to_string(
#if defined(_WIN32)
            GetCurrentProcessId()
#else
            getpid()
#endif
        )+"-"+std::to_string(sequence.fetch_add(1));
    }
    ~StabScratch() { std::error_code ec; std::filesystem::remove(std::filesystem::u8path(path),ec); }
    std::string publish (const std::string& destination, StabilizationProgress* progress)
    {
        if (stabCancelled(progress)) return "cancelled";
#if defined(_WIN32)
        if (!MoveFileExW(std::filesystem::u8path(path).c_str(),std::filesystem::u8path(destination).c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            return "cannot atomically publish stabilization artifact";
#else
        std::error_code ec;
        std::filesystem::rename(std::filesystem::u8path(path),std::filesystem::u8path(destination),ec);
        if (ec) return "cannot atomically publish stabilization artifact: "+ec.message();
#endif
        return {};
    }
};
bool stabMatchesDescriptor (const MediaInfo& info, const stab::SourceIdentity& source)
{
    return info.streamIndex == source.streamIndex && info.width == int(source.width)
        && info.height == int(source.height) && info.sarNum == source.sarNumerator
        && info.sarDen == source.sarDenominator && info.timeBaseNum == source.timeBaseNumerator
        && info.timeBaseDen == source.timeBaseDenominator;
}
void stabReceipt (StabilizationReceipt* receipt, const stab::SourceIdentity& s)
{
    if (!receipt) return;
    receipt->sourceSha256 = stabDigestHex(s.sha256);
    receipt->streamIndex = s.streamIndex; receipt->width = int(s.width); receipt->height = int(s.height);
    receipt->sarNum = s.sarNumerator; receipt->sarDen = s.sarDenominator;
    receipt->timeBaseNum = s.timeBaseNumerator; receipt->timeBaseDen = s.timeBaseDenominator;
    receipt->inSec = s.inSeconds; receipt->outSec = s.outSeconds;
}
std::string stabReadTrack (const std::string& path, stab::Track& track, std::string& digest)
{
    std::ifstream file(std::filesystem::u8path(path),std::ios::binary | std::ios::ate);
    if (!file) return "cannot open stabilization track";
    const auto length = file.tellg();
    if (length <= 0 || uint64_t(length) > stab::maximumTrackBytes) return "invalid stabilization track size";
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()))) return "cannot read stabilization track";
    std::string error;
    if (!stab::decodeTrack(bytes.data(),bytes.size(),track,error)) return error;
    digest = videohelper::sha256Bytes(bytes.data(),bytes.size()); return {};
}

struct StabDecodeResources
{
    AVFormatContext* fmt = nullptr;
    AVCodecContext* dec = nullptr;
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    SwsContext* grayScale = nullptr;
    void close()
    {
        sws_freeContext(grayScale); grayScale = nullptr;
        av_frame_free(&frame); av_packet_free(&packet);
        avcodec_free_context(&dec); avformat_close_input(&fmt);
    }
    ~StabDecodeResources() { close(); }
};
struct StabRenderResources
{
    AVFormatContext* output = nullptr;
    AVCodecContext* encoder = nullptr;
    AVPacket* packet = av_packet_alloc();
    AVFrame* converted = av_frame_alloc();
    AVFrame* warped = av_frame_alloc();
    SwsContext* converter = nullptr;
    int close()
    {
        sws_freeContext(converter); converter = nullptr;
        av_frame_free(&converted); av_frame_free(&warped); av_packet_free(&packet);
        avcodec_free_context(&encoder);
        int rc = 0;
        if (output)
        {
            if (output->pb) rc = avio_closep(&output->pb);
            avformat_free_context(output); output = nullptr;
        }
        return rc;
    }
    ~StabRenderResources() { close(); }
};

// Both passes use this exact decoder, trim comparison and Gray8 conversion.
// No filters, nominal-FPS indexing, hardware contexts or invented timestamps.
std::string runStabPass (const std::string& path, double inSec, double outSec,
                        const std::string& expectedSha, StabilizationProgress* progress,
                        const std::function<std::string(AVFrame*, const stab::GrayFrame&,
                                                       const stab::SourceIdentity&, stab::FrameKey)>& onFrame,
                        int& framesOut, stab::SourceIdentity& source)
{
    framesOut = 0;
    if (!std::isfinite(inSec) || !std::isfinite(outSec) || std::abs(inSec) > 1e9 || std::abs(outSec) > 1e9) return "invalid stabilization range";
    std::string initialDigest;
    auto error = stabFileDigest(path,initialDigest,progress);
    if (!error.empty()) return error;
    if (!expectedSha.empty() && expectedSha != initialDigest) return "stabilization source content changed";
    if (!stabParseDigest(initialDigest,source.sha256)) return "invalid stabilization source digest";
    std::error_code stampError;
    const auto initialStamp = std::filesystem::last_write_time(std::filesystem::u8path(path),stampError);
    if (stampError) return "cannot inspect stabilization source";
    StabDecodeResources resources;
    auto*& fmt = resources.fmt; auto*& dec = resources.dec;
    auto*& packet = resources.packet; auto*& frame = resources.frame;
    auto*& grayScale = resources.grayScale;
    std::vector<uint8_t> grayPixels;
    const auto cancelled = [&] { return stabCancelled(progress); };
    do
    {
        if (!packet || !frame) { error = "cannot allocate stabilization decode frame"; break; }
        fmt = avformat_alloc_context();
        if (!fmt) { error = "cannot allocate stabilization input context"; break; }
        fmt->interrupt_callback.callback = [](void* value)
            { return stabCancelled(static_cast<StabilizationProgress*>(value)) ? 1 : 0; };
        fmt->interrupt_callback.opaque = progress;
        if (avformat_open_input(&fmt,path.c_str(),nullptr,nullptr) < 0
            || avformat_find_stream_info(fmt,nullptr) < 0)
        { error = cancelled() ? "cancelled" : "stabilization source open failed"; break; }
        const int index = av_find_best_stream(fmt,AVMEDIA_TYPE_VIDEO,-1,-1,nullptr,0);
        if (index < 0) { error = "no video stream"; break; }
        auto* stream = fmt->streams[index];
        error = stabUnsupportedSource(stream,nullptr);
        if (!error.empty()) break;
        if (stream->time_base.num <= 0 || stream->time_base.den <= 0) { error = "invalid stabilization source time base"; break; }
        source.streamIndex = index;
        source.timeBaseNumerator = stream->time_base.num; source.timeBaseDenominator = stream->time_base.den;
        source.inSeconds = std::max(0.0,inSec);
        source.outSeconds = outSec;
        if (source.outSeconds <= 0)
        {
            const double start = fmt->start_time != AV_NOPTS_VALUE ? std::max(0.0,double(fmt->start_time)/AV_TIME_BASE) : 0;
            source.outSeconds = fmt->duration > 0 ? start+double(fmt->duration)/AV_TIME_BASE : 1.0e9;
        }
        if (source.outSeconds <= source.inSeconds) { error = "empty stabilization range"; break; }
        const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!codec) { error = "stabilization decoder unavailable"; break; }
        dec = avcodec_alloc_context3(codec);
        if (!dec || avcodec_parameters_to_context(dec,stream->codecpar) < 0
            || avcodec_open2(dec,codec,nullptr) < 0) { error = "stabilization decoder open failed"; break; }
        error = stabUnsupportedSource(stream,dec);
        if (!error.empty()) break;
        if (source.inSeconds > 0)
        {
            const auto seekTick = av_rescale_q(int64_t(std::llround(source.inSeconds*AV_TIME_BASE)),
                                             AVRational{1,AV_TIME_BASE},stream->time_base);
            if (av_seek_frame(fmt,index,seekTick,AVSEEK_FLAG_BACKWARD) < 0) { error = "stabilization seek failed"; break; }
        }
        bool rangeDone = false;
        auto drain = [&]() -> std::string
        {
            while (!rangeDone)
            {
                if (cancelled()) return "cancelled";
                const int rc = avcodec_receive_frame(dec,frame);
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return {};
                if (rc < 0) return "stabilization decode failed: "+avErr(rc);
                if (frame->best_effort_timestamp == AV_NOPTS_VALUE) return "stabilization frame has no presentation timestamp";
                const double seconds = double(frame->best_effort_timestamp)*av_q2d(stream->time_base);
                if (seconds < source.inSeconds) { av_frame_unref(frame); continue; }
                if (seconds >= source.outSeconds) { av_frame_unref(frame); rangeDone = true; return {}; }
                if (av_frame_apply_cropping(frame,AV_FRAME_CROP_UNALIGNED) < 0) return "invalid stabilization visible crop";
                if (auto unsupported = stabUnsupportedSource(stream,dec,frame); !unsupported.empty()) return unsupported;
                const auto sar = av_guess_sample_aspect_ratio(fmt,stream,frame);
                const AVRational validSar = sar.num > 0 && sar.den > 0 ? sar : AVRational{1,1};
                if (!framesOut)
                {
                    if (frame->width < 2 || frame->height < 2 || frame->width > 16384 || frame->height > 16384
                        || uint64_t(frame->width)*frame->height > 64u*1024u*1024u) return "stabilization extent limit exceeded";
                    source.width = unsigned(frame->width); source.height = unsigned(frame->height);
                    source.sarNumerator = validSar.num; source.sarDenominator = validSar.den;
                    grayPixels.resize(size_t(source.width)*source.height);
                }
                else if (unsigned(frame->width) != source.width || unsigned(frame->height) != source.height
                         || validSar.num != source.sarNumerator || validSar.den != source.sarDenominator)
                    return "stabilization visible extent or sample aspect changed";
                if (size_t(framesOut) >= stab::maximumFrames) return "stabilization frame limit exceeded";
                grayScale = sws_getCachedContext(grayScale,frame->width,frame->height,AVPixelFormat(frame->format),
                    frame->width,frame->height,AV_PIX_FMT_GRAY8,SWS_BILINEAR,nullptr,nullptr,nullptr);
                if (!grayScale) return "stabilization Gray8 conversion unavailable";
                uint8_t* destination[] = {grayPixels.data(),nullptr,nullptr,nullptr};
                const int strides[] = {frame->width,0,0,0};
                if (sws_scale(grayScale,frame->data,frame->linesize,0,frame->height,destination,strides) != frame->height)
                    return "stabilization Gray8 conversion failed";
                const stab::GrayFrame gray {grayPixels.data(),source.width,source.height,source.width,grayPixels.size()};
                const stab::FrameKey key {uint64_t(framesOut),frame->best_effort_timestamp};
                frame->pts = key.pts;
                auto callbackError = onFrame(frame,gray,source,key);
                av_frame_unref(frame);
                if (!callbackError.empty()) return callbackError;
                ++framesOut;
                if (progress) progress->frame.store(framesOut);
            }
            return {};
        };
        while (!rangeDone && error.empty())
        {
            if (cancelled()) { error = "cancelled"; break; }
            int rc = av_read_frame(fmt,packet);
            if (rc == AVERROR_EOF)
            {
                rc = avcodec_send_packet(dec,nullptr);
                if (rc == AVERROR(EAGAIN)) { error = drain(); if (error.empty()) rc = avcodec_send_packet(dec,nullptr); }
                if (error.empty() && rc < 0 && rc != AVERROR_EOF) error = "stabilization decoder flush failed: "+avErr(rc);
                if (error.empty()) error = drain();
                break;
            }
            if (rc < 0) { error = "stabilization source read failed: "+avErr(rc); break; }
            if (packet->stream_index == index)
            {
                rc = avcodec_send_packet(dec,packet);
                if (rc == AVERROR(EAGAIN)) { error = drain(); if (error.empty()) rc = avcodec_send_packet(dec,packet); }
                if (error.empty() && rc < 0) error = "stabilization packet decode failed: "+avErr(rc);
                if (error.empty()) error = drain();
            }
            av_packet_unref(packet);
        }
        if (error.empty() && !framesOut) error = "no frames in stabilization range";
    } while (false);
    resources.close();
    if (error.empty())
    {
        std::string finalDigest;
        error = stabFileDigest(path,finalDigest,progress);
        std::error_code finalStampError;
        const auto finalStamp = std::filesystem::last_write_time(std::filesystem::u8path(path),finalStampError);
        if (error.empty() && (initialDigest != finalDigest || finalStampError || initialStamp != finalStamp))
            error = "stabilization source changed during the pass";
    }
    return cancelled() ? "cancelled" : error;
}
} // namespace

std::string MediaContext::stabilizeDetect (const std::string& trfPath, double inSec, double outSec,
                                           int& framesOut, StabilizationProgress* progress,
                                           const std::string& expectedSha, StabilizationReceipt* receipt)
{
    std::lock_guard<std::mutex> lock(mutex_);
    framesOut = 0;
    if (receipt) *receipt = {};
    if (!info_.hasVideo) return "media has no video stream";
    if (info_.hasAlpha) return "stabilization alpha preservation is not implemented for alpha-bearing sources";
    if (auto unsupported = stabUnsupportedSource(fmt_->streams[videoStream_],videoDec_); !unsupported.empty()) return unsupported;
    if (trfPath.empty() || stabSamePath(path_,trfPath)) return "invalid stabilization track path";
    std::unique_ptr<stab::Analyzer> analyzer;
    stab::SourceIdentity source;
    const auto cancel = [&] { return stabCancelled(progress); };
    int count = 0;
    auto error = runStabPass(path_,inSec,outSec,expectedSha,progress,
        [&](AVFrame*, const stab::GrayFrame& gray, const stab::SourceIdentity& identity, stab::FrameKey key)
        {
            if (!stabMatchesDescriptor(info_,identity))
                return std::string("stabilization decoded stream descriptor does not match opened media");
            if (!analyzer) analyzer = std::make_unique<stab::Analyzer>(identity);
            std::string frameError;
            if (!analyzer->append(gray,key,frameError,cancel)) return cancel() ? std::string("cancelled") : frameError;
            return std::string();
        },count,source);
    stab::Track track; std::vector<uint8_t> bytes;
    if (error.empty() && (!analyzer || !analyzer->finish(track,error)))
        if (error.empty()) error = "stabilization analysis produced no track";
    if (error.empty() && !stab::encodeTrack(track,bytes,error)) return error;
    if (!error.empty()) return error;
    StabScratch scratch(trfPath);
    {
        std::ofstream output(std::filesystem::u8path(scratch.path),std::ios::binary | std::ios::trunc);
        if (!output || !output.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size())))
            return "cannot write stabilization track";
        output.close();
        if (!output) return "cannot finalize stabilization track";
    }
    std::string finalDigest;
    if (auto digestError = stabFileDigest(path_,finalDigest,progress); !digestError.empty()) return digestError;
    if (finalDigest != stabDigestHex(source.sha256)) return "stabilization source changed before publication";
    if (auto publishError = scratch.publish(trfPath,progress); !publishError.empty()) return publishError;
    framesOut = count; stabReceipt(receipt,source);
    if (receipt) receipt->trackSha256 = videohelper::sha256Bytes(bytes.data(),bytes.size());
    return {};
}

std::string MediaContext::stabilizeRender (const std::string& trfPath, const std::string& outPath,
                                           double inSec, double outSec, double strength,
                                           int& framesOut, StabilizationProgress* progress,
                                           const std::string& expectedSha, StabilizationReceipt* receipt)
{
    std::lock_guard<std::mutex> lock(mutex_);
    framesOut = 0; if (receipt) *receipt = {};
    if (!info_.hasVideo) return "media has no video stream";
    if (info_.hasAlpha) return "stabilization alpha preservation is not implemented for alpha-bearing sources";
    if (auto unsupported = stabUnsupportedSource(fmt_->streams[videoStream_],videoDec_); !unsupported.empty()) return unsupported;
    if (trfPath.empty() || outPath.empty() || stabSamePath(path_,outPath) || stabSamePath(trfPath,outPath))
        return "invalid stabilization output path";
    stab::Track track; std::string trackDigest;
    auto error = stabReadTrack(trfPath,track,trackDigest);
    if (!error.empty()) return error;
    stab::RenderPlan plan;
    const auto cancel = [&] { return stabCancelled(progress); };
    if (!stab::makeRenderPlan(track,strength,plan,error,1.35,cancel)) return cancel() ? "cancelled" : error;
    if (progress) progress->totalFrames.store(int(track.frames.size()));
    StabScratch scratch(outPath);
    StabRenderResources resources;
    auto*& output = resources.output; auto*& encoder = resources.encoder;
    auto*& packet = resources.packet; auto*& converted = resources.converted; auto*& warped = resources.warped;
    auto*& converter = resources.converter;
    AVStream* outputStream = nullptr; const AVRational encoderTimeBase {1,90000};
    const auto inOffset = int64_t(std::llround(track.source.inSeconds*encoderTimeBase.den));
    int64_t previousPts = AV_NOPTS_VALUE;
    std::string encoderName;
    auto drainEncoder = [&]() -> std::string
    {
        while (true)
        {
            if (cancel()) return "cancelled";
            const int rc = avcodec_receive_packet(encoder,packet);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return {};
            if (rc < 0) return "stabilization encode receive failed: "+avErr(rc);
            packet->stream_index = outputStream->index;
            if (packet->duration <= 0) packet->duration = std::max<int64_t>(1,av_rescale_q(1,av_inv_q(encoder->framerate),encoderTimeBase));
            av_packet_rescale_ts(packet,encoderTimeBase,outputStream->time_base);
            const int writeResult = av_interleaved_write_frame(output,packet);
            av_packet_unref(packet);
            if (writeResult < 0) return "stabilization encoded frame write failed: "+avErr(writeResult);
        }
    };
    stab::SourceIdentity replay; int count = 0;
    if (!packet || !converted || !warped) error = "cannot allocate stabilization render frames";
    if (error.empty()) error = runStabPass(path_,inSec,outSec,expectedSha,progress,
        [&](AVFrame* frame, const stab::GrayFrame& gray, const stab::SourceIdentity& identity, stab::FrameKey key) -> std::string
        {
            std::string frameError;
            if (!stabMatchesDescriptor(info_,identity)) return "stabilization decoded stream descriptor does not match opened media";
            if (!stab::sameSource(track.source,identity)) return "stabilization source metadata or range does not match analysis";
            if (!stab::matchesFrame(track,count,key,frameError)) return frameError;
            std::array<uint8_t,32> digest;
            if (!stab::grayFrameDigest(gray,digest,frameError)) return frameError;
            if (digest != track.frames[size_t(count)].graySha256) return "stabilization decoded Gray8 frame does not match analysis";
            if (!encoder)
            {
                encoderName = videohelper::softwareH264EncoderName();
                const AVCodec* codec = avcodec_find_encoder_by_name(encoderName.c_str());
                if (!codec) return "stabilization software H.264 encoder unavailable";
                if (avformat_alloc_output_context2(&output,nullptr,"mp4",scratch.path.c_str()) < 0 || !output)
                    return "cannot allocate stabilization output context";
                encoder = avcodec_alloc_context3(codec);
                if (!encoder) return "cannot allocate stabilization encoder";
                encoder->width = frame->width; encoder->height = frame->height;
                encoder->pix_fmt = AV_PIX_FMT_YUV420P; encoder->time_base = encoderTimeBase;
                encoder->sample_aspect_ratio = {identity.sarNumerator,identity.sarDenominator};
                encoder->framerate = av_d2q(info_.fps > 0 ? info_.fps : 30,100000);
                encoder->gop_size = std::max(1,int(std::lround(av_q2d(encoder->framerate)*2)));
                if (output->oformat->flags & AVFMT_GLOBALHEADER) encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                if (auto encoderError = videohelper::configureSoftwareH264(encoder,encoderName,18,"veryfast"); !encoderError.empty()) return encoderError;
                if (avcodec_open2(encoder,codec,nullptr) < 0) return "cannot open stabilization encoder: "+encoderName;
                videohelper::tagSoftwareH264Encoder(output,encoder);
                outputStream = avformat_new_stream(output,nullptr);
                if (!outputStream || avcodec_parameters_from_context(outputStream->codecpar,encoder) < 0)
                    return "cannot create stabilization output stream";
                outputStream->time_base = encoderTimeBase; outputStream->sample_aspect_ratio = encoder->sample_aspect_ratio;
                if (!(output->oformat->flags & AVFMT_NOFILE) && avio_open(&output->pb,scratch.path.c_str(),AVIO_FLAG_WRITE) < 0)
                    return "cannot open stabilization temporary output";
                if (avformat_write_header(output,nullptr) < 0) return "cannot write stabilization output header";
                converted->format = warped->format = AV_PIX_FMT_YUV420P;
                converted->width = warped->width = frame->width; converted->height = warped->height = frame->height;
                if (av_frame_get_buffer(converted,32) < 0 || av_frame_get_buffer(warped,32) < 0)
                    return "cannot allocate stabilization plane storage";
            }
            if (av_frame_make_writable(converted) < 0 || av_frame_make_writable(warped) < 0) return "stabilization frame storage unavailable";
            converter = sws_getCachedContext(converter,frame->width,frame->height,AVPixelFormat(frame->format),
                frame->width,frame->height,AV_PIX_FMT_YUV420P,SWS_BILINEAR,nullptr,nullptr,nullptr);
            if (!converter || sws_scale(converter,frame->data,frame->linesize,0,frame->height,converted->data,converted->linesize) != frame->height)
                return "stabilization YUV420P conversion failed";
            for (unsigned plane = 0; plane < 3; ++plane)
            {
                const unsigned width = plane ? (identity.width+1)/2 : identity.width;
                const unsigned height = plane ? (identity.height+1)/2 : identity.height;
                const stab::ConstPlane src {converted->data[plane],width,height,1,size_t(converted->linesize[plane]),size_t(converted->linesize[plane])*height,stab::SampleType::u8};
                const stab::MutablePlane dst {warped->data[plane],width,height,1,size_t(warped->linesize[plane]),size_t(warped->linesize[plane])*height,stab::SampleType::u8};
                if (!stab::warpPlane(plan,size_t(count),key,src,dst,frameError,cancel)) return cancel() ? "cancelled" : frameError;
            }
            warped->pts = av_rescale_q(key.pts,{identity.timeBaseNumerator,identity.timeBaseDenominator},encoderTimeBase)-inOffset;
            if (warped->pts < 0 || (previousPts != AV_NOPTS_VALUE && warped->pts <= previousPts))
                return "stabilization output timestamps are not strictly ordered";
            previousPts = warped->pts;
            warped->duration = frame->duration > 0 ? av_rescale_q(frame->duration,{identity.timeBaseNumerator,identity.timeBaseDenominator},encoderTimeBase)
                : std::max<int64_t>(1,av_rescale_q(1,av_inv_q(encoder->framerate),encoderTimeBase));
            warped->pict_type = AV_PICTURE_TYPE_NONE;
            const int rc = avcodec_send_frame(encoder,warped);
            if (rc < 0) return "stabilization encode failed: "+avErr(rc);
            return drainEncoder();
        },count,replay);
    if (error.empty() && size_t(count) != track.frames.size()) error = "stabilization replay frame count does not match analysis";
    if (error.empty())
    {
        const int rc = avcodec_send_frame(encoder,nullptr);
        if (rc < 0) error = "stabilization encoder flush failed: "+avErr(rc);
        else error = drainEncoder();
    }
    if (error.empty() && av_write_trailer(output) < 0) error = "cannot finalize stabilization output";
    if (const int closeResult = resources.close(); error.empty() && closeResult < 0)
        error = "cannot close stabilization output";
    std::string outputDigest;
    if (error.empty()) error = stabFileDigest(scratch.path,outputDigest,progress);
    // Recheck after encoding/hashing too: source mutations during finalization
    // must not publish a result whose content identity is already obsolete.
    if (error.empty())
    {
        std::string finalSource;
        error = stabFileDigest(path_,finalSource,progress);
        if (error.empty() && finalSource != stabDigestHex(track.source.sha256)) error = "stabilization source changed before publication";
    }
    if (error.empty()) error = scratch.publish(outPath,progress);
    if (!error.empty()) return error;
    framesOut = count; stabReceipt(receipt,replay);
    if (receipt)
    {
        receipt->trackSha256 = trackDigest; receipt->outputSha256 = outputDigest; receipt->encoder = encoderName;
        receipt->zoom = plan.zoom; receipt->smoothingRadius = plan.smoothingRadius;
    }
    return {};
}

// ============================================================================
// Edit proxies — intra-only H.264 transcode (PROTOCOL.md §Proxy media)
// ============================================================================

std::string MediaContext::generateProxy (const std::string& sourcePath,
                                         const std::string& outPath,
                                         double scale, ProxyProgress* progress)
{
    scale = std::max (0.1, std::min (1.0, scale)); // 10%..100% of source
    AVFormatContext* fmt = nullptr;
    AVCodecContext* dec = nullptr;
    AVFormatContext* ofmt = nullptr;
    AVCodecContext* enc = nullptr;
    AVStream* ostream = nullptr;
    SwsContext* sws = nullptr;
    AVPacket* pkt = av_packet_alloc();
    AVPacket* opkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* yuv = av_frame_alloc();
    bool headerWritten = false;
    const AVRational encTb { 1, 90000 };
    std::string error;

    auto fail = [&] (std::string msg) { error = std::move (msg); };

    // Encode to a temp sibling and atomically rename on success. The consumer
    // (videoBuildSegments) keys proxy substitution off existsAsFile(), so a
    // half-written file at the FINAL path is referenced and opened mid-encode —
    // before the mp4 moov atom is written — which fails and permanently blacks
    // the clip (the viewport latches openFailed). Publishing only a complete
    // file via rename closes that race.
    const std::string tmpPath = outPath + ".part";

    auto drainEncoder = [&]() -> std::string
    {
        while (true)
        {
            const int rc = avcodec_receive_packet (enc, opkt);
            if (rc == AVERROR (EAGAIN) || rc == AVERROR_EOF) return {};
            if (rc < 0) return "encode receive failed: " + avErr (rc);
            opkt->stream_index = ostream->index;
            if (opkt->duration <= 0)
                opkt->duration = std::max<int64_t>(1, av_rescale_q(1, av_inv_q(enc->framerate), encTb));
            av_packet_rescale_ts (opkt, encTb, ostream->time_base);
            if (av_interleaved_write_frame (ofmt, opkt) < 0)
                return "write frame failed";
        }
    };

    do
    {
        if (avformat_open_input (&fmt, sourcePath.c_str(), nullptr, nullptr) < 0
            || avformat_find_stream_info (fmt, nullptr) < 0)
        {
            fail ("open failed: " + sourcePath);
            break;
        }
        const int vIdx = av_find_best_stream (fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (vIdx < 0) { fail ("no video stream"); break; }
        AVStream* st = fmt->streams[vIdx];

        // Software decode only: the proxy worker runs alongside the viewport's
        // GL (+ possibly NVDEC) sessions — same rationale as the exporter.
        const AVCodec* codec = avcodec_find_decoder (st->codecpar->codec_id);
        if (codec == nullptr) { fail ("no video decoder for codec"); break; }
        dec = avcodec_alloc_context3 (codec);
        avcodec_parameters_to_context (dec, st->codecpar);
        dec->thread_count = 0;
        if (avcodec_open2 (dec, codec, nullptr) < 0) { fail ("video decoder open failed"); break; }

        auto fr = av_guess_frame_rate (fmt, st, nullptr);
        const double fps = fr.den != 0 ? (double) fr.num / fr.den : 30.0;
        double durationSec = 0.0;
        if (fmt->duration != AV_NOPTS_VALUE)
            durationSec = (double) fmt->duration / AV_TIME_BASE;
        else if (st->duration != AV_NOPTS_VALUE)
            durationSec = streamTimeToSec (st, st->duration);
        if (progress != nullptr)
            progress->totalFrames.store ((int) std::lround (durationSec * fps));

        // yuv420p needs even dimensions; scale (0.1..1.0) multiplies them (min 2).
        const int srcW = dec->width, srcH = dec->height;
        const int dstW = std::max (2, (int) std::lround (srcW * scale) & ~1);
        const int dstH = std::max (2, (int) std::lround (srcH * scale) & ~1);

        const char* encoderName = videohelper::softwareH264EncoderName();
        const AVCodec* vCodec = avcodec_find_encoder_by_name (encoderName);
        if (vCodec == nullptr) { fail (std::string("H.264 encoder unavailable: ") + encoderName); break; }

        if (avformat_alloc_output_context2 (&ofmt, nullptr, "mp4", tmpPath.c_str()) < 0
            || ofmt == nullptr)
        {
            fail ("cannot create output context for " + outPath);
            break;
        }

        enc = avcodec_alloc_context3 (vCodec);
        if (enc == nullptr) { fail ("cannot allocate proxy encoder"); break; }
        enc->width = dstW;
        enc->height = dstH;
        enc->pix_fmt = AV_PIX_FMT_YUV420P;
        enc->time_base = encTb;
        enc->framerate = av_d2q (fps, 100000);
        enc->gop_size = 1;       // every frame a keyframe: intra-only
        enc->max_b_frames = 0;   // no reordering — frame-independent seeks
        if (ofmt->oformat->flags & AVFMT_GLOBALHEADER)
            enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        error = videohelper::configureSoftwareH264(enc, vCodec->name, scale < 0.999 ? 20 : 18, "veryfast");
        if (!error.empty()) break;
        if (avcodec_open2 (enc, vCodec, nullptr) < 0)
        {
            fail (std::string("cannot open proxy encoder ") + vCodec->name);
            break;
        }
        videohelper::tagSoftwareH264Encoder(ofmt, enc);

        ostream = avformat_new_stream (ofmt, nullptr);
        if (ostream == nullptr) { fail ("cannot create output stream"); break; }
        avcodec_parameters_from_context (ostream->codecpar, enc);
        ostream->time_base = encTb;

        if (! (ofmt->oformat->flags & AVFMT_NOFILE)
            && avio_open (&ofmt->pb, tmpPath.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            fail ("cannot open output file " + outPath);
            break;
        }
        if (avformat_write_header (ofmt, nullptr) < 0) { fail ("write_header failed"); break; }
        headerWritten = true;

        yuv->format = AV_PIX_FMT_YUV420P;
        yuv->width = dstW;
        yuv->height = dstH;
        if (av_frame_get_buffer (yuv, 0) < 0) { fail ("frame alloc failed"); break; }

        auto encodeOne = [&] (AVFrame* in) -> std::string
        {
            sws = sws_getCachedContext (sws, in->width, in->height,
                                        (AVPixelFormat) in->format,
                                        dstW, dstH, AV_PIX_FMT_YUV420P,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (sws == nullptr) return "swscale context failed";
            if (av_frame_make_writable (yuv) < 0) return "frame not writable";
            sws_scale (sws, in->data, in->linesize, 0, in->height, yuv->data, yuv->linesize);

            // Preserve source timing: the proxy is a 1:1 timeline stand-in.
            const int64_t pts = in->best_effort_timestamp == AV_NOPTS_VALUE
                                    ? 0
                                    : av_rescale_q (in->best_effort_timestamp,
                                                    st->time_base, encTb);
            yuv->pts = std::max<int64_t> (0, pts);
            yuv->duration = in->duration > 0 ? av_rescale_q(in->duration, st->time_base, encTb)
                : std::max<int64_t>(1, av_rescale_q(1, av_inv_q(enc->framerate), encTb));
            yuv->pict_type = AV_PICTURE_TYPE_NONE;
            const int rc = avcodec_send_frame (enc, yuv);
            if (rc < 0) return "encode failed: " + avErr (rc);
            if (progress != nullptr)
                progress->frame.fetch_add (1);
            return drainEncoder();
        };

        bool eof = false;
        while (! eof && error.empty())
        {
            if (progress != nullptr && progress->abort.load())
            {
                fail ("cancelled");
                break;
            }
            int rc = av_read_frame (fmt, pkt);
            if (rc == AVERROR_EOF)
            {
                avcodec_send_packet (dec, nullptr);
                eof = true;
            }
            else if (rc < 0) { fail ("read frame failed: " + avErr (rc)); break; }
            else if (pkt->stream_index != vIdx) { av_packet_unref (pkt); continue; }
            else
            {
                rc = avcodec_send_packet (dec, pkt);
                av_packet_unref (pkt);
                if (rc < 0 && rc != AVERROR (EAGAIN))
                {
                    fail ("send packet failed: " + avErr (rc));
                    break;
                }
            }

            while (error.empty())
            {
                rc = avcodec_receive_frame (dec, frame);
                if (rc == AVERROR (EAGAIN) || rc == AVERROR_EOF) break;
                if (rc < 0) { fail ("decode failed: " + avErr (rc)); break; }
                error = encodeOne (frame);
                av_frame_unref (frame);
            }
        }

        if (error.empty())
        {
            avcodec_send_frame (enc, nullptr); // flush encoder
            error = drainEncoder();
        }
        if (error.empty() && av_write_trailer (ofmt) < 0)
            error = "write_trailer failed";
    } while (false);

    if (sws != nullptr) sws_freeContext (sws);
    if (enc != nullptr) avcodec_free_context (&enc);
    if (ofmt != nullptr)
    {
        if (headerWritten && ! (ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb != nullptr)
            avio_closep (&ofmt->pb);
        avformat_free_context (ofmt);
    }
    if (dec != nullptr) avcodec_free_context (&dec);
    if (fmt != nullptr) avformat_close_input (&fmt);
    av_frame_free (&yuv);
    av_frame_free (&frame);
    av_packet_free (&opkt);
    av_packet_free (&pkt);
    // Publish atomically: a complete file appears at outPath in one step, or
    // not at all. A failed/cancelled run leaves no partial file behind.
    if (! error.empty())
        std::remove (tmpPath.c_str());
    else if (std::rename (tmpPath.c_str(), outPath.c_str()) != 0)
    {
        std::remove (tmpPath.c_str());
        error = "could not publish proxy: " + outPath;
    }
    return error;
}

// ============================================================================
// Thumbnails (JPEG via libavcodec mjpeg encoder)
// ============================================================================

std::string MediaContext::writeThumbnails (const std::vector<double>& times, int maxW, int maxH,
                                           const std::string& outDir, const std::string& baseName,
                                           std::vector<std::string>& outPaths)
{
    const AVCodec* jpeg = avcodec_find_encoder (AV_CODEC_ID_MJPEG);
    if (jpeg == nullptr) return "mjpeg encoder unavailable";

    for (size_t i = 0; i < times.size(); ++i)
    {
        DecodedFrame df;
        auto error = getFrame (times[i], maxW, maxH, df);
        if (! error.empty()) return error;

        AVCodecContext* enc = avcodec_alloc_context3 (jpeg);
        enc->width = df.width;
        enc->height = df.height;
        enc->pix_fmt = AV_PIX_FMT_YUVJ420P;
        enc->time_base = { 1, 25 };
        enc->flags |= AV_CODEC_FLAG_QSCALE;
        enc->global_quality = FF_QP2LAMBDA * 6;
        if (avcodec_open2 (enc, jpeg, nullptr) < 0)
        {
            avcodec_free_context (&enc);
            return "jpeg encoder open failed";
        }

        AVFrame* yuv = av_frame_alloc();
        yuv->format = AV_PIX_FMT_YUVJ420P;
        yuv->width = df.width;
        yuv->height = df.height;
        av_frame_get_buffer (yuv, 0);

        SwsContext* toYuv = sws_getContext (df.width, df.height, AV_PIX_FMT_RGBA,
                                            df.width, df.height, AV_PIX_FMT_YUVJ420P,
                                            SWS_BILINEAR, nullptr, nullptr, nullptr);
        const uint8_t* srcData[4] = { df.rgba.data(), nullptr, nullptr, nullptr };
        int srcLines[4] = { df.strideBytes, 0, 0, 0 };
        sws_scale (toYuv, srcData, srcLines, 0, df.height, yuv->data, yuv->linesize);
        sws_freeContext (toYuv);

        AVPacket* pkt = av_packet_alloc();
        std::string outPath = outDir + "/" + baseName + "-" + std::to_string (i) + ".jpg";
        bool ok = avcodec_send_frame (enc, yuv) == 0
               && avcodec_receive_packet (enc, pkt) == 0;
        if (ok)
        {
            FILE* f = std::fopen (outPath.c_str(), "wb");
            if (f != nullptr)
            {
                std::fwrite (pkt->data, 1, (size_t) pkt->size, f);
                std::fclose (f);
                outPaths.push_back (outPath);
            }
            else ok = false;
        }

        av_packet_free (&pkt);
        av_frame_free (&yuv);
        avcodec_free_context (&enc);
        if (! ok) return "thumbnail encode/write failed: " + outPath;
    }
    return {};
}
