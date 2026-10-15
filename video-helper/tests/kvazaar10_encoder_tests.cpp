#include "kvazaar10_encoder.h"
#include <kvazaar.h>

extern "C" {
#include <libavcodec/defs.h>
}

#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "Failed at line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace {
using Encoder = videohelper::Kvazaar10Encoder;
enum class Failure { none, configAlloc, configInit, preset, open, headers, encode, chunks };
Failure failure = Failure::none;
int depth = 10;
bool missingApi = false;
int configs = 0, encoders = 0, pictures = 0, chunks = 0;
int width = 0, height = 0;
int transfer = 0;
std::deque<int64_t> pending;

kvz_config* configAlloc() {
    if (failure == Failure::configAlloc) return nullptr;
    ++configs;
    return new kvz_config{};
}
int configInit(kvz_config* cfg) { cfg->gop_len = 16; return failure != Failure::configInit; }
int configDestroy(kvz_config* cfg) { if (cfg) { --configs; delete cfg; } return 1; }
int configParse(kvz_config*, const char* key, const char* value) {
    CHECK(std::strcmp(key, "preset") == 0 && std::strcmp(value, "medium") == 0);
    return failure != Failure::preset;
}
kvz_picture* pictureAlloc(int32_t w, int32_t h) {
    ++pictures;
    auto* pic = new kvz_picture{};
    pic->width = w; pic->height = h; pic->stride = w + 8; pic->chroma_format = KVZ_CSP_420;
    for (int plane = 0; plane < 3; ++plane) {
        const auto count = size_t(pic->stride >> (plane != 0)) * (h >> (plane != 0));
        pic->data[plane] = new kvz_pixel[count]{};
    }
    return pic;
}
void pictureFree(kvz_picture* pic) {
    if (!pic) return;
    --pictures;
    for (auto* data : pic->data) delete[] data;
    delete pic;
}
void chunkFree(kvz_data_chunk* chunk) {
    while (chunk) { auto* next = chunk->next; --chunks; delete chunk; chunk = next; }
}
kvz_encoder* encoderOpen(const kvz_config* cfg) {
    CHECK(cfg->input_bitdepth == 10 && cfg->input_format == KVZ_FORMAT_P420);
    CHECK(cfg->vui.fullrange == 0 && cfg->vui.colorprim == AVCOL_PRI_BT2020);
    CHECK(cfg->vui.colormatrix == AVCOL_SPC_BT2020_NCL && cfg->vui.chroma_loc == 0);
    CHECK(cfg->vui.transfer == AVCOL_TRC_SMPTE2084 || cfg->vui.transfer == AVCOL_TRC_ARIB_STD_B67);
    CHECK(cfg->framerate_num == 30000 && cfg->framerate_denom == 1001);
    CHECK(cfg->width == 16 && cfg->height == 8 && cfg->qp == 18 && cfg->intra_period == 64);
    if (failure == Failure::open) return nullptr;
    width = cfg->width; height = cfg->height; transfer = cfg->vui.transfer;
    ++encoders;
    return reinterpret_cast<kvz_encoder*>(new int{});
}
void encoderClose(kvz_encoder* encoder) {
    if (!encoder) return;
    // Terminal error cleanup must free caller-owned pictures before releasing the gate.
    CHECK(pictures == 0 && chunks == 0);
    --encoders; pending.clear(); delete reinterpret_cast<int*>(encoder);
}
void outputChunks(kvz_data_chunk** output, uint32_t* size) {
    *output = new kvz_data_chunk{};
    ++chunks;
    (*output)->len = 2;
    (*output)->data[0] = 0; (*output)->data[1] = 1;
    (*output)->next = new kvz_data_chunk{};
    ++chunks;
    (*output)->next->len = 2;
    (*output)->next->data[0] = 2; (*output)->next->data[1] = 3;
    *size = failure == Failure::chunks ? 3 : 4;
}
int encoderHeaders(kvz_encoder*, kvz_data_chunk** output, uint32_t* size) {
    outputChunks(output, size);
    return failure != Failure::headers;
}
int encoderEncode(kvz_encoder*, kvz_picture* input, kvz_data_chunk** output,
                  uint32_t* size, kvz_picture** reconstructed, kvz_picture**, kvz_frame_info* info) {
    *output = nullptr; *size = 0; *reconstructed = nullptr;
    if (input) {
        for (int plane = 0; plane < 3; ++plane) {
            const int w = width >> (plane != 0), h = height >> (plane != 0);
            const int stride = input->stride >> (plane != 0);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    CHECK(input->data[plane][y * stride + x] == 300 + plane * 100 + y * 10 + x);
        }
        pending.push_back(input->pts);
    }
    if (failure == Failure::encode || (!pending.empty() && (!input || pending.size() > 2))) {
        outputChunks(output, size);
        *reconstructed = pictureAlloc(width, height);
        (*reconstructed)->pts = pending.front();
        (*reconstructed)->dts = pending.front() - 2;
        pending.pop_front();
        info->nal_unit_type = (*reconstructed)->pts == 0 ? KVZ_NAL_IDR_N_LP : KVZ_NAL_TRAIL_R;
    }
    return failure != Failure::encode;
}
const kvz_api api = [] {
    kvz_api result{};
    result.config_alloc = configAlloc; result.config_init = configInit;
    result.config_destroy = configDestroy; result.config_parse = configParse;
    result.picture_alloc = pictureAlloc; result.picture_free = pictureFree;
    result.chunk_free = chunkFree; result.encoder_open = encoderOpen;
    result.encoder_close = encoderClose; result.encoder_headers = encoderHeaders;
    result.encoder_encode = encoderEncode;
    return result;
}();

Encoder::Settings settings() {
    Encoder::Settings value;
    value.width = 16; value.height = 8; value.frameRate = AVRational{30000, 1001};
    return value;
}
void clean() { CHECK(configs == 0 && encoders == 0 && pictures == 0 && chunks == 0 && pending.empty()); }
AVFrame* frame(int64_t pts) {
    auto* value = av_frame_alloc(); CHECK(value);
    value->width = 16; value->height = 8; value->format = AV_PIX_FMT_YUV420P10LE;
    value->pts = pts; value->color_range = AVCOL_RANGE_MPEG;
    value->color_primaries = AVCOL_PRI_BT2020; value->color_trc = static_cast<AVColorTransferCharacteristic>(transfer);
    value->colorspace = AVCOL_SPC_BT2020_NCL;
    CHECK(av_frame_get_buffer(value, 64) == 0);
    for (int plane = 0; plane < 3; ++plane) {
        const int w = 16 >> (plane != 0), h = 8 >> (plane != 0);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const int sample = 300 + plane * 100 + y * 10 + x;
            auto* at = value->data[plane] + y * value->linesize[plane] + x * 2;
            at[0] = sample & 255; at[1] = sample >> 8;
        }
    }
    return value;
}
}

extern "C" const kvz_api* donut_kvz10_kvz_api_get(int bitDepth) { CHECK(bitDepth == 10); return missingApi ? nullptr : &api; }
extern "C" int donut_kvz10_kvz_adapter_compiled_bit_depth() { return depth; }

int main() {
    for (auto fail : {Failure::configAlloc, Failure::configInit, Failure::preset,
                      Failure::open, Failure::headers, Failure::chunks}) {
        failure = fail;
        Encoder encoder; CHECK(!encoder.open(settings()).empty()); clean();
        failure = Failure::none;
        CHECK(encoder.open(settings()).empty()); encoder.close(); clean();
    }
    {
        Encoder encoder;
        auto cfg = settings(); cfg.width = 18;
        CHECK(!encoder.open(cfg).empty()); clean();
        cfg = settings(); cfg.height = 10;
        CHECK(!encoder.open(cfg).empty()); clean();
        cfg = settings(); cfg.frameRate.num = 0;
        CHECK(!encoder.open(cfg).empty()); clean();
        cfg = settings(); cfg.intraPeriod = 60;
        CHECK(!encoder.open(cfg).empty()); clean();
        depth = 8; CHECK(!encoder.open(settings()).empty()); clean(); depth = 10;
        missingApi = true; CHECK(!encoder.open(settings()).empty()); clean(); missingApi = false;
        CHECK(encoder.open(settings()).empty());
        CHECK(!encoder.open(settings()).empty());
        Encoder second; CHECK(!second.open(settings()).empty());
        encoder.close(); CHECK(second.open(settings()).empty()); second.close(); clean();
    }
    for (auto hdr : {Encoder::Transfer::pq, Encoder::Transfer::hlg}) {
        Encoder encoder; auto cfg = settings(); cfg.transfer = hdr;
        CHECK(encoder.open(cfg).empty());
        CHECK(encoder.timeBase().num == 1001 && encoder.timeBase().den == 30000);
        auto* parameters = avcodec_parameters_alloc(); CHECK(parameters);
        CHECK(encoder.copyCodecParameters(parameters).empty());
        CHECK(parameters->width == 16 && parameters->height == 8);
        CHECK(parameters->profile == AV_PROFILE_HEVC_MAIN_10);
        CHECK(parameters->format == AV_PIX_FMT_YUV420P10LE && parameters->bits_per_raw_sample == 10);
        CHECK(parameters->color_range == AVCOL_RANGE_MPEG && parameters->color_primaries == AVCOL_PRI_BT2020);
        CHECK(parameters->color_space == AVCOL_SPC_BT2020_NCL && parameters->color_trc == transfer);
        CHECK(parameters->extradata_size == 4 && parameters->extradata[3] == 3);
        for (int i = 0; i < AV_INPUT_BUFFER_PADDING_SIZE; ++i) CHECK(parameters->extradata[4 + i] == 0);
        avcodec_parameters_free(&parameters);
        auto* packet = av_packet_alloc(); CHECK(packet); bool hasPacket = true;
        auto* invalid = frame(0); invalid->format = AV_PIX_FMT_YUV420P;
        CHECK(!encoder.encode(invalid, packet, hasPacket).empty() && !hasPacket);
        invalid->format = AV_PIX_FMT_YUV420P10LE; invalid->linesize[1] = 1;
        CHECK(!encoder.encode(invalid, packet, hasPacket).empty());
        av_frame_free(&invalid);
        invalid = frame(AV_NOPTS_VALUE);
        CHECK(!encoder.encode(invalid, packet, hasPacket).empty());
        invalid->pts = 0; invalid->color_range = AVCOL_RANGE_JPEG;
        CHECK(!encoder.encode(invalid, packet, hasPacket).empty());
        invalid->color_range = AVCOL_RANGE_MPEG; invalid->data[0][1] = 4;
        CHECK(!encoder.encode(invalid, packet, hasPacket).empty());
        av_frame_free(&invalid); CHECK(pictures == 0 && pending.empty());
        int packetCount = 0;
        for (int64_t pts = 0; pts < 3; ++pts) {
            auto* input = frame(pts);
            CHECK(encoder.encode(input, packet, hasPacket).empty());
            av_frame_free(&input);
            if (hasPacket) { CHECK(packet->pts == 0 && packet->dts == -2); ++packetCount; CHECK(packet->flags & AV_PKT_FLAG_KEY); }
        }
        auto* duplicate = frame(2);
        CHECK(!encoder.encode(duplicate, packet, hasPacket).empty()); av_frame_free(&duplicate);
        while (!encoder.drained()) {
            CHECK(encoder.encode(nullptr, packet, hasPacket).empty());
            if (hasPacket) {
                CHECK(packet->pts == packetCount && packet->dts == packetCount - 2);
                CHECK(packet->duration == 1 && packet->size == 4 && packet->data[3] == 3);
                CHECK(!(packet->flags & AV_PKT_FLAG_KEY)); ++packetCount;
            }
        }
        CHECK(packetCount == 3);
        CHECK(encoder.encode(nullptr, packet, hasPacket).empty() && !hasPacket);
        auto* late = frame(3); CHECK(!encoder.encode(late, packet, hasPacket).empty()); av_frame_free(&late);
        av_packet_free(&packet); encoder.close(); clean();
    }
    for (auto fail : {Failure::encode, Failure::chunks, Failure::none}) {
        Encoder encoder; CHECK(encoder.open(settings()).empty());
        auto* packet = av_packet_alloc(); auto* input = frame(0); bool got = false;
        failure = fail;
        auto error = encoder.encode(input, packet, got);
        if (fail == Failure::encode) { CHECK(!error.empty()); clean(); }
        else if (fail == Failure::chunks) {
            CHECK(error.empty());
            CHECK(!encoder.encode(nullptr, packet, got).empty()); clean();
        } else {
            CHECK(error.empty() && !got && pending.size() == 1);
            encoder.close(); clean(); // cancellation discards delayed frames
        }
        av_frame_free(&input); av_packet_free(&packet); failure = Failure::none;
        Encoder replacement; CHECK(replacement.open(settings()).empty()); replacement.close(); clean();
    }
    std::cout << "Kvazaar ten-bit adapter leaf fixtures passed\n";
}
