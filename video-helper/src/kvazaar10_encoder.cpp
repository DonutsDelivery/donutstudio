#include "kvazaar10_encoder.h"

#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {
#include <kvazaar.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/defs.h>
#include <libavutil/mem.h>
const kvz_api* donut_kvz10_kvz_api_get(int bit_depth);
int donut_kvz10_kvz_adapter_compiled_bit_depth(void);
}

static_assert(KVZ_BIT_DEPTH == 10 && sizeof(kvz_pixel) == 2,
              "Kvazaar adapter must use the matched ten-bit header");

namespace videohelper {
namespace {
std::atomic<bool> encoderOwned {false};

// Chunks belong to Kvazaar even on a failed API call.
struct Output {
    const kvz_api* api;
    kvz_data_chunk* chunks = nullptr;
    kvz_picture* picture = nullptr;
    uint32_t size = 0;
    ~Output() { api->chunk_free(chunks); api->picture_free(picture); }
};

std::string copyChunks(const Output& output, uint8_t* destination)
{
    uint64_t copied = 0;
    for (auto* chunk = output.chunks; chunk; chunk = chunk->next) {
        if (chunk->len > KVZ_DATA_CHUNK_SIZE || copied + chunk->len > output.size)
            return "Kvazaar returned invalid chunk lengths";
        std::memcpy(destination + copied, chunk->data, chunk->len);
        copied += chunk->len;
    }
    return copied == output.size ? std::string{} : "Kvazaar returned incomplete chunks";
}
}

struct Kvazaar10Encoder::Impl {
    const kvz_api* api = nullptr;
    kvz_config* config = nullptr;
    kvz_encoder* encoder = nullptr;
    Settings settings;
    std::vector<uint8_t> headers;
    bool ownsEncoder = false;
    bool flushing = false;
    bool drained = false;
    int64_t lastPts = AV_NOPTS_VALUE;
    ~Impl() {
        if (api) { api->encoder_close(encoder); api->config_destroy(config); }
        if (ownsEncoder) encoderOwned.store(false, std::memory_order_release);
    }
};

Kvazaar10Encoder::Kvazaar10Encoder() = default;
Kvazaar10Encoder::~Kvazaar10Encoder() = default;
void Kvazaar10Encoder::close() { impl_.reset(); }
bool Kvazaar10Encoder::drained() const { return impl_ && impl_->drained; }
AVRational Kvazaar10Encoder::timeBase() const {
    return impl_ ? av_inv_q(impl_->settings.frameRate) : AVRational{0, 1};
}

std::string Kvazaar10Encoder::open(const Settings& settings)
{
    if (impl_) return "Kvazaar encoder is already open";
    if (settings.width <= 0 || settings.width > INT_MAX / 2 || settings.height <= 0
        || settings.width % 8 || settings.height % 8)
        return "Kvazaar requires visible width and height divisible by eight; padding is not implemented";
    if ((settings.transfer != Transfer::pq && settings.transfer != Transfer::hlg)
        || settings.frameRate.num <= 0 || settings.frameRate.den <= 0
        || settings.qp < 0 || settings.qp > 51 || settings.intraPeriod <= 0)
        return "Invalid Kvazaar frame rate or quality settings";
    if (donut_kvz10_kvz_adapter_compiled_bit_depth() != 10)
        return "Kvazaar library is not the private ten-bit build";

    auto state = std::make_unique<Impl>();
    state->api = donut_kvz10_kvz_api_get(10);
    if (!state->api) return "Kvazaar ten-bit API unavailable";
    bool available = false;
    if (!encoderOwned.compare_exchange_strong(available, true, std::memory_order_acq_rel))
        return "Kvazaar ten-bit encoder is busy";
    state->ownsEncoder = true;
    state->settings = settings;
    state->config = state->api->config_alloc();
    if (!state->config || !state->api->config_init(state->config))
        return "Cannot initialize Kvazaar configuration";
    auto& cfg = *state->config;
    if (!state->api->config_parse(&cfg, "preset", "medium"))
        return "Cannot set Kvazaar preset";
    cfg.width = settings.width;
    cfg.height = settings.height;
    cfg.framerate_num = settings.frameRate.num;
    cfg.framerate_denom = settings.frameRate.den;
    cfg.input_format = KVZ_FORMAT_P420;
    cfg.input_bitdepth = 10;
    cfg.qp = settings.qp;
    cfg.intra_period = settings.intraPeriod;
    if (cfg.gop_len && !cfg.gop_lowdelay && settings.intraPeriod > 1
        && settings.intraPeriod % cfg.gop_len != 0)
        return "Kvazaar intra period must be a multiple of its random-access GOP length";
    cfg.vps_period = 0;
    cfg.vui.sar_width = 1;
    cfg.vui.sar_height = 1;
    cfg.vui.fullrange = 0;
    cfg.vui.colorprim = AVCOL_PRI_BT2020;
    cfg.vui.transfer = settings.transfer == Transfer::hlg
        ? AVCOL_TRC_ARIB_STD_B67 : AVCOL_TRC_SMPTE2084;
    cfg.vui.colormatrix = AVCOL_SPC_BT2020_NCL;
    cfg.vui.chroma_loc = 0;
    state->encoder = state->api->encoder_open(&cfg);
    if (!state->encoder) return "Cannot open Kvazaar ten-bit encoder";
    Output headers {state->api};
    if (!state->api->encoder_headers(state->encoder, &headers.chunks, &headers.size)
        || !headers.chunks || !headers.size || headers.size > INT_MAX)
        return "Cannot obtain Kvazaar HEVC headers";
    state->headers.resize(headers.size);
    if (auto error = copyChunks(headers, state->headers.data()); !error.empty()) return error;
    impl_ = std::move(state);
    return {};
}

std::string Kvazaar10Encoder::copyCodecParameters(AVCodecParameters* parameters) const
{
    if (!impl_ || !parameters) return "Kvazaar encoder or codec parameters unavailable";
    auto* result = avcodec_parameters_alloc();
    if (!result) return "Cannot allocate HEVC codec parameters";
    result->codec_type = AVMEDIA_TYPE_VIDEO;
    result->codec_id = AV_CODEC_ID_HEVC;
    result->format = AV_PIX_FMT_YUV420P10LE;
    result->profile = AV_PROFILE_HEVC_MAIN_10;
    result->width = impl_->settings.width;
    result->height = impl_->settings.height;
    result->bits_per_raw_sample = 10;
    result->sample_aspect_ratio = AVRational{1, 1};
    result->color_range = AVCOL_RANGE_MPEG;
    result->color_primaries = AVCOL_PRI_BT2020;
    result->color_trc = impl_->settings.transfer == Transfer::hlg
        ? AVCOL_TRC_ARIB_STD_B67 : AVCOL_TRC_SMPTE2084;
    result->color_space = AVCOL_SPC_BT2020_NCL;
    result->chroma_location = AVCHROMA_LOC_LEFT;
    result->extradata_size = static_cast<int>(impl_->headers.size());
    result->extradata = static_cast<uint8_t*>(av_mallocz(impl_->headers.size() + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!result->extradata) { avcodec_parameters_free(&result); return "Cannot allocate HEVC headers"; }
    std::memcpy(result->extradata, impl_->headers.data(), impl_->headers.size());
    const int status = avcodec_parameters_copy(parameters, result);
    avcodec_parameters_free(&result);
    return status < 0 ? "Cannot copy HEVC codec parameters" : std::string{};
}

std::string Kvazaar10Encoder::encode(const AVFrame* frame, AVPacket* packet, bool& hasPacket)
{
    hasPacket = false;
    if (!packet) return "HEVC packet unavailable";
    av_packet_unref(packet);
    if (!impl_) return "Kvazaar encoder is not open";
    auto& state = *impl_;
    if (frame && (state.flushing || state.drained)) return "Cannot submit a frame after Kvazaar drain begins";
    if (state.drained) return {};
    bool terminate = false;
    struct CloseOnFailure {
        Kvazaar10Encoder* owner;
        bool& terminate;
        ~CloseOnFailure() { if (terminate) owner->close(); }
    } closeOnFailure {this, terminate};
    kvz_picture* input = nullptr;
    struct InputOwner {
        const kvz_api* api;
        kvz_picture*& picture;
        ~InputOwner() { api->picture_free(picture); }
    } inputOwner {state.api, input};
    if (frame) {
        const auto& settings = state.settings;
        const auto transfer = settings.transfer == Transfer::hlg ? AVCOL_TRC_ARIB_STD_B67 : AVCOL_TRC_SMPTE2084;
        if (frame->width != settings.width || frame->height != settings.height
            || frame->format != AV_PIX_FMT_YUV420P10LE
            || frame->color_range != AVCOL_RANGE_MPEG || frame->color_primaries != AVCOL_PRI_BT2020
            || frame->color_trc != transfer || frame->colorspace != AVCOL_SPC_BT2020_NCL)
            return "Kvazaar input must match limited Rec.2020 ten-bit PQ/HLG settings";
        if (frame->pts == AV_NOPTS_VALUE || (state.lastPts != AV_NOPTS_VALUE && frame->pts <= state.lastPts))
            return "Kvazaar input PTS must increase in encoder time base";
        for (int plane = 0; plane < 3; ++plane) {
            const int width = settings.width >> (plane != 0);
            if (!frame->data[plane] || static_cast<int64_t>(frame->linesize[plane]) < int64_t(width) * 2)
                return "Kvazaar input has an invalid plane or stride";
        }
        input = state.api->picture_alloc(settings.width, settings.height);
        if (!input) return "Cannot allocate Kvazaar input picture";
        if (input->width != settings.width || input->height != settings.height
            || input->stride < settings.width || input->stride % 2
            || input->chroma_format != KVZ_CSP_420)
            return "Kvazaar allocated an incompatible picture";
        for (int plane = 0; plane < 3; ++plane) {
            const int width = settings.width >> (plane != 0);
            const int height = settings.height >> (plane != 0);
            const int stride = input->stride >> (plane != 0);
            if (!input->data[plane]) return "Kvazaar allocated an empty picture plane";
            for (int y = 0; y < height; ++y) {
                const uint8_t* source = frame->data[plane] + ptrdiff_t(y) * frame->linesize[plane];
                kvz_pixel* destination = input->data[plane] + ptrdiff_t(y) * stride;
                for (int x = 0; x < width; ++x) {
                    const unsigned sample = unsigned(source[x * 2]) | unsigned(source[x * 2 + 1]) << 8;
                    if (sample > 1023) return "Kvazaar input contains a sample outside ten-bit range";
                    destination[x] = static_cast<kvz_pixel>(sample);
                }
            }
        }
        input->pts = frame->pts;
    } else {
        state.flushing = true;
    }
    Output output {state.api};
    kvz_frame_info info {};
    if (!state.api->encoder_encode(state.encoder, input, &output.chunks, &output.size,
                                   &output.picture, nullptr, &info)) {
        terminate = true;
        return "Kvazaar failed to encode frame";
    }
    if (frame) state.lastPts = frame->pts;
    if (!output.chunks) {
        if (output.size || output.picture) { terminate = true; return "Kvazaar returned incomplete frame output"; }
        if (!frame) state.drained = true;
        return {};
    }
    if (!output.picture || !output.size || output.size > INT_MAX) {
        terminate = true; return "Kvazaar returned invalid packet output";
    }
    if (av_new_packet(packet, static_cast<int>(output.size)) < 0) {
        terminate = true; return "Cannot allocate HEVC output packet";
    }
    if (auto error = copyChunks(output, packet->data); !error.empty()) {
        av_packet_unref(packet); terminate = true; return error;
    }
    packet->pts = output.picture->pts;
    packet->dts = output.picture->dts;
    packet->duration = 1;
    if (info.nal_unit_type >= KVZ_NAL_BLA_W_LP && info.nal_unit_type <= KVZ_NAL_RSV_IRAP_VCL23)
        packet->flags |= AV_PKT_FLAG_KEY;
    hasPacket = true;
    return {};
}

} // namespace videohelper
