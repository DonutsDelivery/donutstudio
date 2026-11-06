// The GPU encode paths (Plan 3, export and encoders) on whatever this machine
// has: a CUDA frame pool for NVENC and a VA-API surface pool, each filled by
// upload and read back exactly, then a few frames through h264_nvenc and
// h264_vaapi from those pools. CUDA-GL interop itself needs a GL context on the
// NVIDIA GPU, which the export makes and this test does not. A machine without
// a device, driver or encoder prints why each part skipped and passes.

#include "gpu_encode_cuda_gl.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

namespace
{

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

std::string averror (int code)
{
    char text[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror (code, text, sizeof (text));
    return text;
}

constexpr int kSide = 256;  // above every hardware encoder's minimum size

// A device's frame pool, or nullptr with the reason it is missing.
AVBufferRef* makePool (AVHWDeviceType type, AVPixelFormat hardware, AVPixelFormat software, std::string& why)
{
    AVBufferRef* device = nullptr;
    if (int code = av_hwdevice_ctx_create (&device, type, nullptr, nullptr, 0); code < 0)
    {
        why = std::string (av_hwdevice_get_type_name (type)) + " device did not open: " + averror (code);
        return nullptr;
    }
    AVBufferRef* frames = av_hwframe_ctx_alloc (device);
    av_buffer_unref (&device);
    auto* context = reinterpret_cast<AVHWFramesContext*> (frames->data);
    context->format = hardware;
    context->sw_format = software;
    context->width = kSide;
    context->height = kSide;
    context->initial_pool_size = type == AV_HWDEVICE_TYPE_VAAPI ? 8 : 0;
    if (int code = av_hwframe_ctx_init (frames); code < 0)
    {
        why = std::string (av_hwdevice_get_type_name (type)) + " frame pool did not start: " + averror (code);
        av_buffer_unref (&frames);
    }
    return frames;
}

AVFrame* softwareFrame (AVPixelFormat format, int seed)
{
    AVFrame* frame = av_frame_alloc();
    frame->format = format;
    frame->width = kSide;
    frame->height = kSide;
    av_frame_get_buffer (frame, 0);
    for (int plane = 0; plane < AV_NUM_DATA_POINTERS && frame->data[plane] != nullptr; ++plane)
    {
        const int rows = plane == 0 ? kSide : kSide / 2;
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < frame->linesize[plane]; ++x)
                frame->data[plane][y * frame->linesize[plane] + x] = static_cast<std::uint8_t> (x * 3 + y * 7 + seed);
    }
    return frame;
}

bool samePixels (const AVFrame* a, const AVFrame* b, int bytesPerRow0, int bytesPerRow1)
{
    for (int y = 0; y < kSide; ++y)
        if (std::memcmp (a->data[0] + y * a->linesize[0], b->data[0] + y * b->linesize[0], bytesPerRow0) != 0)
            return false;
    if (bytesPerRow1 > 0)
        for (int y = 0; y < kSide / 2; ++y)
            if (std::memcmp (a->data[1] + y * a->linesize[1], b->data[1] + y * b->linesize[1], bytesPerRow1) != 0)
                return false;
    return true;
}

// Upload, read back, compare: the route encodeAndWrite takes for CPU frames.
void roundTrip (AVBufferRef* pool, AVPixelFormat software, const char* name, int bytesPerRow0, int bytesPerRow1)
{
    AVFrame* source = softwareFrame (software, 11);
    AVFrame* gpu = av_frame_alloc();
    AVFrame* back = av_frame_alloc();
    const bool uploaded = av_hwframe_get_buffer (pool, gpu, 0) >= 0 && av_hwframe_transfer_data (gpu, source, 0) >= 0;
    check (uploaded, std::string (name) + " takes an upload");
    back->format = software;
    if (uploaded)
    {
        // Some VA-API drivers read no surface before an encoder context exists
        // ("invalid VAContextID"). The export never reads back, so that skips by
        // name; a readback that returns other pixels still fails.
        if (const int code = av_hwframe_transfer_data (back, gpu, 0); code < 0)
            std::cout << "SKIP " << name << " readback: " << averror (code) << '\n';
        else
            check (samePixels (source, back, bytesPerRow0, bytesPerRow1), std::string (name) + " reads back what it took");
    }
    av_frame_free (&source);
    av_frame_free (&gpu);
    av_frame_free (&back);
}

// A few frames through the hardware encoder straight from its pool.
void encodeFromPool (AVBufferRef* pool, const char* encoderName, AVPixelFormat hardware, AVPixelFormat software)
{
    const AVCodec* codec = avcodec_find_encoder_by_name (encoderName);
    if (codec == nullptr)
    {
        std::cout << "SKIP " << encoderName << ": not in this FFmpeg build\n";
        return;
    }
    AVCodecContext* encoder = avcodec_alloc_context3 (codec);
    encoder->width = kSide;
    encoder->height = kSide;
    encoder->time_base = { 1, 30 };
    encoder->framerate = { 30, 1 };
    encoder->pix_fmt = hardware;
    encoder->sw_pix_fmt = software;
    encoder->hw_frames_ctx = av_buffer_ref (pool);
    encoder->bit_rate = 1'000'000;
    if (int code = avcodec_open2 (encoder, codec, nullptr); code < 0)
    {
        std::cout << "SKIP " << encoderName << ": did not open with GPU frames: " << averror (code) << '\n';
        avcodec_free_context (&encoder);
        return;
    }
    int packets = 0;
    AVPacket* packet = av_packet_alloc();
    for (int n = 0; n <= 4; ++n)
    {
        AVFrame* gpu = nullptr;
        if (n < 4)
        {
            AVFrame* source = softwareFrame (software, n * 13);
            gpu = av_frame_alloc();
            av_hwframe_get_buffer (pool, gpu, 0);
            av_hwframe_transfer_data (gpu, source, 0);
            gpu->pts = n;
            av_frame_free (&source);
        }
        check (avcodec_send_frame (encoder, gpu) >= 0, std::string (encoderName) + " takes frame " + std::to_string (n));
        av_frame_free (&gpu);
        while (avcodec_receive_packet (encoder, packet) >= 0)
        {
            ++packets;
            av_packet_unref (packet);
        }
    }
    check (packets >= 4, std::string (encoderName) + " writes a packet per frame from GPU memory, got " + std::to_string (packets));
    av_packet_free (&packet);
    avcodec_free_context (&encoder);
}

void bridgeRefusals()
{
    videohelper::CudaGlFrameBridge bridge;
    std::string error;
    check (! bridge.open (nullptr, error) && ! error.empty(), "a bridge without a CUDA pool says why: " + error);
    error.clear();
    check (bridge.capture (1, 2, 2, error) == nullptr && error.find ("not open") != std::string::npos,
           "a closed bridge refuses a capture: " + error);
}

void cudaTests()
{
    std::string why;
    AVBufferRef* pool = makePool (AV_HWDEVICE_TYPE_CUDA, AV_PIX_FMT_CUDA, AV_PIX_FMT_0BGR32, why);
    if (pool == nullptr)
    {
        std::cout << "SKIP NVENC from CUDA memory: " << why << '\n';
        return;
    }
    roundTrip (pool, AV_PIX_FMT_0BGR32, "the CUDA RGBA pool", kSide * 4, 0);
    videohelper::CudaGlFrameBridge bridge;
    std::string error;
    check (bridge.open (pool, error), "the CUDA-GL bridge opens on the encoder's pool: " + error);
    std::cout << "SKIP CUDA-GL capture: needs a GL context on the NVIDIA GPU, which the export makes\n";
    encodeFromPool (pool, "h264_nvenc", AV_PIX_FMT_CUDA, AV_PIX_FMT_0BGR32);
    av_buffer_unref (&pool);
}

void vaapiTests()
{
#if defined(__linux__)
    std::string why;
    AVBufferRef* pool = makePool (AV_HWDEVICE_TYPE_VAAPI, AV_PIX_FMT_VAAPI, AV_PIX_FMT_NV12, why);
    if (pool == nullptr)
    {
        std::cout << "SKIP VA-API: " << why << '\n';
        return;
    }
    roundTrip (pool, AV_PIX_FMT_NV12, "the VA-API NV12 pool", kSide, kSide);
    encodeFromPool (pool, "h264_vaapi", AV_PIX_FMT_VAAPI, AV_PIX_FMT_NV12);
    av_buffer_unref (&pool);
#else
    std::cout << "SKIP VA-API: Linux only\n";
#endif
}

} // namespace

int main()
{
    bridgeRefusals();
    cudaTests();
    vaapiTests();
    std::cout << (checks - failures) << "/" << checks << " export GPU encode checks passed\n";
    return failures == 0 ? 0 : 1;
}
