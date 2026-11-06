#pragma once

// RGB to the encoder's YUV on the GPU (Plan 3, export and encoders). A GPU
// export converts its composite into NV12 (8-bit SDR) or P010 (10-bit HDR)
// planes before CUDA-GL interop copies them into the encoder's frame, so the
// matrix, transfer and range are the ones the stream's colour tags declare and
// not whatever the encoder's own RGB input assumes.
//
// This header holds the math the shader runs, with a CPU reference of it, and
// reads the conversion from the stream's tags. Tag values are the ITU-T H.273
// code points FFmpeg's AVColorPrimaries, AVColorTransferCharacteristic,
// AVColorSpace and AVColorRange use, so nothing here needs FFmpeg.
//
// Chroma is horizontally cosited and vertically centred (AVCHROMA_LOC_LEFT),
// as paintHdr420 in exporter.cpp: each chroma sample averages the pixel at its
// even column on two rows.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace videohelper::yuvencode
{

namespace tag
{
constexpr int kPrimariesBt709 = 1, kPrimariesBt2020 = 9;
constexpr int kTransferBt709 = 1, kTransferPq = 16, kTransferHlg = 18;
constexpr int kMatrixBt709 = 1, kMatrixBt2020Ncl = 9;
constexpr int kRangeLimited = 1;
} // namespace tag

enum class Matrix { Bt709, Bt2020Ncl };
// Sdr: the composite is already display-coded BT.709 RGB in [0, 1].
// Pq, Hlg: the composite is linear BT.709-primaries light with 1.0 at the
// 203-nit reference white, as the HDR movie readback takes it.
enum class Transfer { Sdr, Pq, Hlg };

struct Conversion
{
    Matrix matrix = Matrix::Bt709;
    Transfer transfer = Transfer::Sdr;
    int bits = 8;          // 8 for NV12, 10 for P010
    float peakNits = 0.0f; // HDR only
};

constexpr double kReferenceWhiteNits = 203.0;

// The conversion a stream's tags ask for, or the reason none fits.
inline std::optional<Conversion> fromStreamTags (int primaries, int transfer, int matrix, int range, int bits,
                                                 float peakNits, std::string& error)
{
    if (range != tag::kRangeLimited)
    {
        error = "GPU YUV conversion writes limited range only";
        return std::nullopt;
    }
    Conversion conversion;
    conversion.bits = bits;
    if (primaries == tag::kPrimariesBt709 && transfer == tag::kTransferBt709 && matrix == tag::kMatrixBt709)
    {
        if (bits != 8)
        {
            error = "GPU YUV conversion writes BT.709 SDR as 8-bit NV12";
            return std::nullopt;
        }
        return conversion;
    }
    if (primaries == tag::kPrimariesBt2020 && matrix == tag::kMatrixBt2020Ncl
        && (transfer == tag::kTransferPq || transfer == tag::kTransferHlg))
    {
        if (bits != 10)
        {
            error = "GPU YUV conversion writes BT.2020 HDR as 10-bit P010";
            return std::nullopt;
        }
        if (! std::isfinite (peakNits) || peakNits < kReferenceWhiteNits || peakNits > 10000.0f)
        {
            error = "GPU YUV conversion needs an HDR peak between 203 and 10000 nits";
            return std::nullopt;
        }
        conversion.matrix = Matrix::Bt2020Ncl;
        conversion.transfer = transfer == tag::kTransferPq ? Transfer::Pq : Transfer::Hlg;
        conversion.peakNits = peakNits;
        return conversion;
    }
    error = "GPU YUV conversion has no matrix for colour tags primaries " + std::to_string (primaries)
        + ", transfer " + std::to_string (transfer) + ", matrix " + std::to_string (matrix);
    return std::nullopt;
}

// Luma weights.
struct Weights
{
    double kr = 0.0, kb = 0.0;
};

inline Weights weights (Matrix matrix) noexcept
{
    return matrix == Matrix::Bt709 ? Weights { 0.2126, 0.0722 } : Weights { 0.2627, 0.0593 };
}

// Limited-range code values: Y' = offset + scale * Y, C = mid + scale * C.
struct Codes
{
    double lumaOffset = 0.0, lumaScale = 0.0, chromaMid = 0.0, chromaScale = 0.0;
    double maximum = 0.0;
    // A code's value in the plane texture: unorm8 for NV12; for P010 the 10-bit
    // code sits in the top bits of a unorm16, so code c stores as c * 64 / 65535.
    double storageUnit = 0.0;
};

inline Codes codes (int bits) noexcept
{
    if (bits == 10)
        return { 64.0, 876.0, 512.0, 896.0, 1023.0, 64.0 / 65535.0 };
    return { 16.0, 219.0, 128.0, 224.0, 255.0, 1.0 / 255.0 };
}

// Linear BT.709 to linear BT.2020 primaries, ITU-R BT.2087 (row major).
constexpr std::array<double, 9> kBt709ToBt2020 = {
    0.6274039, 0.3292830, 0.0433131,
    0.0690973, 0.9195404, 0.0113623,
    0.0163914, 0.0880133, 0.8955953,
};

// SMPTE ST 2084 inverse EOTF of absolute luminance in nits.
inline double pqEncode (double nits) noexcept
{
    const double m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const double c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    const double y = std::pow (std::clamp (nits / 10000.0, 0.0, 1.0), m1);
    return std::pow ((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

// The HLG system gamma for a nominal peak, BT.2100 Table 5 note 5f, extended
// above 2000 nits as hdr_movie_readback.h does.
inline double hlgGamma (double peakNits) noexcept
{
    return peakNits <= 2000.0 ? 1.2 + 0.42 * std::log10 (peakNits / 1000.0)
                              : 1.2 * std::pow (1.111, std::log2 (peakNits / 1000.0));
}

// BT.2100 HLG OETF of normalized scene light.
inline double hlgOetf (double scene) noexcept
{
    constexpr double a = 0.17883277, b = 1.0 - 4.0 * a;
    const double c = 0.5 - a * std::log (4.0 * a);
    scene = std::clamp (scene, 0.0, 1.0);
    return scene <= 1.0 / 12.0 ? std::sqrt (3.0 * scene) : a * std::log (12.0 * scene - b) + c;
}

// The non-linear R'G'B' the matrix takes, from one composite pixel. HDR light
// above the declared peak clips to it: a GPU frame cannot refuse a pixel.
inline std::array<double, 3> encodedRgb (const Conversion& conversion, double r, double g, double b) noexcept
{
    const auto unit = [] (double v) { return std::isfinite (v) ? std::clamp (v, 0.0, 1.0) : 0.0; };
    if (conversion.transfer == Transfer::Sdr)
        return { unit (r), unit (g), unit (b) };
    const auto finite = [] (double v) { return std::isfinite (v) ? std::max (v, 0.0) : 0.0; };
    const double in[3] = { finite (r), finite (g), finite (b) };
    double display[3] {};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            display[row] += kBt709ToBt2020[row * 3 + column] * in[column];
    const double peak = conversion.peakNits;
    for (double& value : display)
        value = std::min (value * kReferenceWhiteNits, peak);
    if (conversion.transfer == Transfer::Pq)
        return { pqEncode (display[0]), pqEncode (display[1]), pqEncode (display[2]) };
    // HLG: display light normalized to the peak, the inverse OOTF, then the OETF,
    // as hdr_movie_readback.h readHlg.
    for (double& value : display)
        value /= peak;
    const double gamma = hlgGamma (peak);
    const double luminance = 0.2627 * display[0] + 0.6780 * display[1] + 0.0593 * display[2];
    const double inverseGain = luminance > 0.0 ? std::pow (luminance, (1.0 - gamma) / gamma) : 0.0;
    return { hlgOetf (display[0] * inverseGain), hlgOetf (display[1] * inverseGain), hlgOetf (display[2] * inverseGain) };
}

struct YCbCr
{
    double y = 0.0, cb = 0.0, cr = 0.0;  // Y in [0, 1], Cb and Cr in [-0.5, 0.5]
};

inline YCbCr toYCbCr (Matrix matrix, const std::array<double, 3>& rgb) noexcept
{
    const auto w = weights (matrix);
    const double y = w.kr * rgb[0] + (1.0 - w.kr - w.kb) * rgb[1] + w.kb * rgb[2];
    return { y, (rgb[2] - y) / (2.0 * (1.0 - w.kb)), (rgb[0] - y) / (2.0 * (1.0 - w.kr)) };
}

inline std::uint16_t quantize (double value, double offset, double scale, double maximum) noexcept
{
    return static_cast<std::uint16_t> (std::clamp (std::floor (offset + scale * value + 0.5), 0.0, maximum));
}

inline std::uint16_t lumaCode (const Conversion& conversion, double r, double g, double b) noexcept
{
    const auto c = codes (conversion.bits);
    return quantize (toYCbCr (conversion.matrix, encodedRgb (conversion, r, g, b)).y, c.lumaOffset, c.lumaScale, c.maximum);
}

// The chroma pair for the pixels at (x, y) and (x, y + 1).
inline std::array<std::uint16_t, 2> chromaCodes (const Conversion& conversion, const std::array<double, 3>& top,
                                                 const std::array<double, 3>& bottom) noexcept
{
    const auto c = codes (conversion.bits);
    const auto a = toYCbCr (conversion.matrix, encodedRgb (conversion, top[0], top[1], top[2]));
    const auto b = toYCbCr (conversion.matrix, encodedRgb (conversion, bottom[0], bottom[1], bottom[2]));
    return { quantize ((a.cb + b.cb) * 0.5, c.chromaMid, c.chromaScale, c.maximum),
             quantize ((a.cr + b.cr) * 0.5, c.chromaMid, c.chromaScale, c.maximum) };
}

// The uniforms the GL pass takes, all derived above.
struct ShaderConstants
{
    std::array<float, 9> toBt2020 {};  // row major
    float kr = 0.0f, kb = 0.0f;
    float lumaOffset = 0.0f, lumaScale = 0.0f, chromaMid = 0.0f, chromaScale = 0.0f, maximum = 0.0f;
    float storageUnit = 0.0f;
    int transfer = 0;  // 0 SDR, 1 PQ, 2 HLG
    float peakNits = 0.0f, referenceWhite = 0.0f, hlgGamma = 0.0f;
};

inline ShaderConstants shaderConstants (const Conversion& conversion) noexcept
{
    ShaderConstants constants;
    for (std::size_t i = 0; i < 9; ++i)
        constants.toBt2020[i] = static_cast<float> (kBt709ToBt2020[i]);
    const auto w = weights (conversion.matrix);
    const auto c = codes (conversion.bits);
    constants.kr = static_cast<float> (w.kr);
    constants.kb = static_cast<float> (w.kb);
    constants.lumaOffset = static_cast<float> (c.lumaOffset);
    constants.lumaScale = static_cast<float> (c.lumaScale);
    constants.chromaMid = static_cast<float> (c.chromaMid);
    constants.chromaScale = static_cast<float> (c.chromaScale);
    constants.maximum = static_cast<float> (c.maximum);
    constants.storageUnit = static_cast<float> (c.storageUnit);
    constants.transfer = conversion.transfer == Transfer::Sdr ? 0 : conversion.transfer == Transfer::Pq ? 1 : 2;
    constants.peakNits = conversion.peakNits;
    constants.referenceWhite = static_cast<float> (kReferenceWhiteNits);
    constants.hlgGamma = conversion.transfer == Transfer::Hlg ? static_cast<float> (hlgGamma (conversion.peakNits)) : 0.0f;
    return constants;
}

// Bytes per row of each plane: NV12 1 byte per sample, P010 2.
inline int bytesPerSample (int bits) noexcept
{
    return bits == 10 ? 2 : 1;
}

} // namespace videohelper::yuvencode
