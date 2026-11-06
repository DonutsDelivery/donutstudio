// The GPU's RGB to YUV for encoder frames (Plan 3, export and encoders): the
// conversion read from the stream's colour tags, BT.709 and BT.2020 matrices,
// limited-range codes for NV12 and P010, the PQ and HLG transfers an HDR movie
// takes, the chroma siting, and the uniforms the GL pass gets.

#include "yuv_encode_math.h"

#include <cmath>
#include <iostream>
#include <string>

namespace
{

namespace yuv = videohelper::yuvencode;

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

bool near (double a, double b, double tolerance)
{
    return std::abs (a - b) <= tolerance;
}

yuv::Conversion sdr()
{
    std::string error;
    return *yuv::fromStreamTags (yuv::tag::kPrimariesBt709, yuv::tag::kTransferBt709, yuv::tag::kMatrixBt709,
                                 yuv::tag::kRangeLimited, 8, 0.0f, error);
}

yuv::Conversion hdr (int transfer, float peak)
{
    std::string error;
    return *yuv::fromStreamTags (yuv::tag::kPrimariesBt2020, transfer, yuv::tag::kMatrixBt2020Ncl,
                                 yuv::tag::kRangeLimited, 10, peak, error);
}

void tagTests()
{
    std::string error;
    auto conversion = yuv::fromStreamTags (1, 1, 1, 1, 8, 0.0f, error);
    check (conversion && conversion->matrix == yuv::Matrix::Bt709 && conversion->transfer == yuv::Transfer::Sdr,
           "BT.709 tags read as BT.709 SDR");
    conversion = yuv::fromStreamTags (9, 16, 9, 1, 10, 1000.0f, error);
    check (conversion && conversion->matrix == yuv::Matrix::Bt2020Ncl && conversion->transfer == yuv::Transfer::Pq
               && conversion->bits == 10, "BT.2020 PQ tags read as 10-bit PQ");
    conversion = yuv::fromStreamTags (9, 18, 9, 1, 10, 1000.0f, error);
    check (conversion && conversion->transfer == yuv::Transfer::Hlg, "ARIB STD-B67 reads as HLG");

    check (! yuv::fromStreamTags (1, 1, 1, 2, 8, 0.0f, error) && error.find ("limited range") != std::string::npos,
           "full range is refused by name: " + error);
    check (! yuv::fromStreamTags (1, 1, 1, 1, 10, 0.0f, error) && error.find ("8-bit NV12") != std::string::npos,
           "SDR in 10 bits is refused by name: " + error);
    check (! yuv::fromStreamTags (9, 16, 9, 1, 8, 1000.0f, error) && error.find ("P010") != std::string::npos,
           "HDR in 8 bits is refused by name: " + error);
    check (! yuv::fromStreamTags (9, 16, 9, 1, 10, 100.0f, error) && error.find ("peak") != std::string::npos,
           "an HDR peak under reference white is refused: " + error);
    check (! yuv::fromStreamTags (5, 6, 5, 1, 8, 0.0f, error) && error.find ("primaries 5") != std::string::npos,
           "BT.601 tags name themselves: " + error);
}

void sdrTests()
{
    const auto c = sdr();
    check (yuv::lumaCode (c, 1, 1, 1) == 235 && yuv::lumaCode (c, 0, 0, 0) == 16, "BT.709 white and black are 235 and 16");
    check (yuv::lumaCode (c, 1, 0, 0) == 63, "BT.709 red's luma is 16 + 219 x 0.2126");
    const auto grey = yuv::chromaCodes (c, { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 });
    check (grey[0] == 128 && grey[1] == 128, "grey has no chroma");
    const auto red = yuv::chromaCodes (c, { 1, 0, 0 }, { 1, 0, 0 });
    check (red[1] == 240 && red[0] == 102, "BT.709 red's chroma is Cr 240, Cb 102");
    const auto mixed = yuv::chromaCodes (c, { 1, 0, 0 }, { 0, 0, 0 });
    check (mixed[1] == 184, "chroma averages the two rows it covers: Cr " + std::to_string (mixed[1]));
    check (yuv::lumaCode (c, 2.0, 2.0, 2.0) == 235 && yuv::lumaCode (c, -1, -1, -1) == 16 && yuv::lumaCode (c, NAN, 0, 0) == 16,
           "out-of-range and non-finite SDR clamp");
}

void hdrTests()
{
    check (near (yuv::pqEncode (10000.0), 1.0, 1e-9) && yuv::pqEncode (0.0) < 1e-5, "PQ spans 0 to 10000 nits");
    check (near (yuv::pqEncode (100.0), 0.5081, 5e-4), "100 nits is PQ 0.508");
    check (near (yuv::hlgOetf (1.0 / 12.0), 0.5, 1e-9) && near (yuv::hlgOetf (1.0), 1.0, 1e-6), "the HLG OETF knee and top");
    check (near (yuv::hlgGamma (1000.0), 1.2, 1e-12), "HLG gamma is 1.2 at 1000 nits");

    const auto pq = hdr (yuv::tag::kTransferPq, 1000.0f);
    const auto white = static_cast<int> (std::floor (64.0 + 876.0 * yuv::pqEncode (203.0) + 0.5));
    check (yuv::lumaCode (pq, 1, 1, 1) == white, "PQ reference white sits at 203 nits: " + std::to_string (white));
    check (yuv::lumaCode (pq, 0, 0, 0) == 64, "PQ black is code 64");
    const auto peak = static_cast<int> (std::floor (64.0 + 876.0 * yuv::pqEncode (1000.0) + 0.5));
    check (yuv::lumaCode (pq, 100, 100, 100) == peak, "light past the peak clips to it, no refused frame");
    const auto greyChroma = yuv::chromaCodes (pq, { 1, 1, 1 }, { 1, 1, 1 });
    check (greyChroma[0] == 512 && greyChroma[1] == 512, "PQ white has no chroma: BT.709 white is BT.2020 white");

    const auto hlg = hdr (yuv::tag::kTransferHlg, 1000.0f);
    const auto encoded = yuv::encodedRgb (hlg, 1, 1, 1);
    check (near (encoded[0], 0.75, 0.01), "HLG reference white lands near 0.75: " + std::to_string (encoded[0]));
    check (yuv::lumaCode (hlg, 0, 0, 0) == 64, "HLG black is code 64");

    // The CPU painter (exporter.cpp paintHdr420) uses BT.2020 weights with the
    // same codes; a transfer-coded grey matches it.
    const double coded = yuv::encodedRgb (pq, 0.5, 0.5, 0.5)[0];
    check (yuv::lumaCode (pq, 0.5, 0.5, 0.5) == static_cast<int> (std::lround (64 + 876 * coded)),
           "GPU luma matches the CPU painter's");
}

void shaderTests()
{
    const auto ten = yuv::shaderConstants (hdr (yuv::tag::kTransferPq, 1000.0f));
    check (std::lround (ten.storageUnit * 65535.0f * 940.0f) == 940 * 64, "a P010 code stores in the top ten bits");
    check (ten.transfer == 1 && near (ten.kr, 0.2627, 1e-6) && ten.maximum == 1023.0f, "PQ uniforms");
    check (near (ten.toBt2020[0] + ten.toBt2020[1] + ten.toBt2020[2], 1.0, 1e-4), "the primaries matrix keeps white");
    const auto eight = yuv::shaderConstants (sdr());
    check (std::lround (eight.storageUnit * 255.0f * 235.0f) == 235 && eight.transfer == 0 && near (eight.kb, 0.0722, 1e-6),
           "NV12 uniforms");
    check (yuv::shaderConstants (hdr (yuv::tag::kTransferHlg, 1000.0f)).hlgGamma == 1.2f, "HLG gamma reaches the shader");
    check (yuv::bytesPerSample (10) == 2 && yuv::bytesPerSample (8) == 1, "P010 rows are two bytes a sample");
}

} // namespace

int main()
{
    tagTests();
    sdrTests();
    hdrTests();
    shaderTests();
    std::cout << (checks - failures) << "/" << checks << " yuv encode math checks passed\n";
    return failures == 0 ? 0 : 1;
}
