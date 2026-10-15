#include "kvazaar10_encoder.h"

#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
struct kvz_api;
const kvz_api* kvz_api_get(int);
const kvz_api* donut_kvz10_kvz_api_get(int);
int donut_kvz10_kvz_adapter_compiled_bit_depth(void);
}

bool loadedFrom(const void* symbol, const char* expected)
{
    Dl_info library {};
    std::error_code error;
    return dladdr(symbol, &library) && library.dli_fname
        && std::filesystem::equivalent(expected, library.dli_fname, error) && !error;
}

int main(int argc, char** argv)
{
    if (argc != 3 || !loadedFrom(reinterpret_cast<const void*>(&kvz_api_get), argv[1])
        || !loadedFrom(reinterpret_cast<const void*>(&avcodec_find_encoder_by_name), argv[2])) {
        std::cerr << "Runtime dylibs do not match the inspected candidates\n";
        return 1;
    }
    if (donut_kvz10_kvz_adapter_compiled_bit_depth() != 10
        || !kvz_api_get(8) || !donut_kvz10_kvz_api_get(10)
        || kvz_api_get(8) == donut_kvz10_kvz_api_get(10)) {
        std::cerr << "Private Main10 and ordinary Kvazaar bindings are not distinct\n";
        return 1;
    }
    const AVCodec* codec = avcodec_find_encoder_by_name("libkvazaar");
    if (!codec) {
        std::cerr << "FFmpeg profile lacks its ordinary libkvazaar encoder\n";
        return 1;
    }
    AVCodecContext* ordinary = avcodec_alloc_context3(codec);
    AVFrame* eight = av_frame_alloc();
    AVFrame* ten = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    int result = 1;
    videohelper::Kvazaar10Encoder privateEncoder;
    do {
        if (!ordinary || !eight || !ten || !packet) break;
        ordinary->width = ordinary->height = 64;
        ordinary->time_base = {1, 25};
        ordinary->framerate = {25, 1};
        ordinary->pix_fmt = AV_PIX_FMT_YUV420P;
        if (avcodec_open2(ordinary, codec, nullptr) < 0) break;
        videohelper::Kvazaar10Encoder::Settings settings;
        settings.width = settings.height = 64;
        settings.frameRate = {25, 1};
        if (auto error = privateEncoder.open(settings); !error.empty()) {
            std::cerr << error << '\n';
            break;
        }
        eight->format = AV_PIX_FMT_YUV420P;
        ten->format = AV_PIX_FMT_YUV420P10LE;
        eight->width = ten->width = eight->height = ten->height = 64;
        eight->pts = ten->pts = 0;
        ten->color_range = AVCOL_RANGE_MPEG;
        ten->color_primaries = AVCOL_PRI_BT2020;
        ten->color_trc = AVCOL_TRC_SMPTE2084;
        ten->colorspace = AVCOL_SPC_BT2020_NCL;
        ten->chroma_location = AVCHROMA_LOC_LEFT;
        if (av_frame_get_buffer(eight, 32) < 0 || av_frame_get_buffer(ten, 32) < 0) break;
        for (int plane = 0; plane < 3; ++plane) {
            int extent = plane ? 32 : 64;
            for (int y = 0; y < extent; ++y) {
                std::memset(eight->data[plane] + y * eight->linesize[plane], 128, extent);
                auto* row = ten->data[plane] + y * ten->linesize[plane];
                for (int x = 0; x < extent; ++x) {
                    row[x * 2] = 0;
                    row[x * 2 + 1] = 2;
                }
            }
        }
        if (avcodec_send_frame(ordinary, eight) < 0) break;
        int eightPackets = 0;
        int receive = 0;
        while ((receive = avcodec_receive_packet(ordinary, packet)) >= 0) {
            ++eightPackets;
            av_packet_unref(packet);
        }
        if (receive != AVERROR(EAGAIN) || avcodec_send_frame(ordinary, nullptr) < 0) break;
        while ((receive = avcodec_receive_packet(ordinary, packet)) >= 0) {
            ++eightPackets;
            av_packet_unref(packet);
        }
        if (receive != AVERROR_EOF || eightPackets != 1) break;
        bool hasPacket = false;
        if (auto error = privateEncoder.encode(ten, packet, hasPacket); !error.empty()) {
            std::cerr << error << '\n';
            break;
        }
        int tenPackets = hasPacket ? 1 : 0;
        std::string error;
        for (int count = 0; count < 64 && !privateEncoder.drained(); ++count) {
            error = privateEncoder.encode(nullptr, packet, hasPacket);
            if (!error.empty()) break;
            if (hasPacket) ++tenPackets;
        }
        if (!error.empty() || !privateEncoder.drained() || tenPackets != 1) break;
        std::cout << "ordinary_yuv420p_packets=" << eightPackets
                  << " private_main10_packets=" << tenPackets << " compiled_depth=10\n";
        result = 0;
    } while (false);
    privateEncoder.close();
    av_packet_free(&packet);
    av_frame_free(&eight);
    av_frame_free(&ten);
    avcodec_free_context(&ordinary);
    return result;
}
