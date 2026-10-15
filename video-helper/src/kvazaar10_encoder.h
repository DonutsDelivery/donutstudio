#pragma once

#include <memory>
#include <string>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavcodec/packet.h>
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

namespace videohelper {

// Helper-private encoder. Calls on one instance must be serialized by its owner.
// Only one instance can be open at once; a second open fails without waiting.
class Kvazaar10Encoder {
public:
    enum class Transfer { pq, hlg };
    struct Settings {
        int width = 0;
        int height = 0;
        AVRational frameRate {0, 1};
        Transfer transfer = Transfer::pq;
        int qp = 18;
        int intraPeriod = 64;
    };

    Kvazaar10Encoder();
    ~Kvazaar10Encoder();
    Kvazaar10Encoder(const Kvazaar10Encoder&) = delete;
    Kvazaar10Encoder& operator=(const Kvazaar10Encoder&) = delete;

    std::string open(const Settings& settings);
    // Supplies HEVC Main10 parameters and padded Annex B VPS/SPS/PPS extradata.
    // The caller owns the stream, timestamp rescale, mux, trailer and file cleanup.
    std::string copyCodecParameters(AVCodecParameters* parameters) const;
    AVRational timeBase() const;

    // Input is limited-range Rec.2020 YUV420P10LE with matching PQ/HLG tags.
    // Frame PTS uses timeBase(). The adapter copies pixels before returning.
    // Pass null repeatedly after the last frame until drained() is true.
    // packet is unreferenced first; hasPacket can be false while frames buffer.
    std::string encode(const AVFrame* frame, AVPacket* packet, bool& hasPacket);
    bool drained() const;
    // Cancellation discards buffered frames and releases encoder ownership.
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace videohelper
