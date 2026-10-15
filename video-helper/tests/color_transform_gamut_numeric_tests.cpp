// Headless numeric checks for the sRGB, Display P3 and Rec.2020 colour paths.
// The GLSL in color_transform_gl.cpp and the Metal source in backend_sokol.mm
// implement the same transfer and matrix math; the scalar mirrors below follow it.
#include "../../shared/ColorTransformContract.h"
#include "../../shared/ColorTransformGpuMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace
{
using namespace colortransform;
using gpumath::Matrix3;

void require(bool condition, const char* message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

constexpr double kMatrixTolerance = 1.0e-5;
constexpr double kTransferTolerance = 1.0e-9;

bool near(double actual, double expected, double tolerance)
{
    return std::abs(actual - expected) <= tolerance;
}

Matrix3 identityMatrix()
{
    return { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
}

bool matrixNear(const Matrix3& actual, const Matrix3& expected, double tolerance)
{
    for (std::size_t index = 0; index < actual.size(); ++index)
        if (! near(actual[index], expected[index], tolerance))
            return false;
    return true;
}

std::array<double, 3> transform3(const Matrix3& m, const std::array<double, 3>& v)
{
    return { m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
             m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
             m[6] * v[0] + m[7] * v[1] + m[8] * v[2] };
}

Matrix3 inverse3(const Matrix3& m)
{
    const double a = m[0], b = m[1], c = m[2];
    const double d = m[3], e = m[4], f = m[5];
    const double g = m[6], h = m[7], i = m[8];
    const double cofactorA = e * i - f * h;
    const double cofactorB = f * g - d * i;
    const double cofactorC = d * h - e * g;
    const double determinant = a * cofactorA + b * cofactorB + c * cofactorC;
    return { cofactorA / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant,
             cofactorB / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant,
             cofactorC / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant };
}

// Published CIE 1931 xy chromaticities. All three spaces share the D65 white.
struct Chromaticity
{
    double rx, ry, gx, gy, bx, by;
};
constexpr Chromaticity kSrgbPrimaries { 0.640, 0.330, 0.300, 0.600, 0.150, 0.060 };      // IEC 61966-2-1, BT.709
constexpr Chromaticity kDisplayP3Primaries { 0.680, 0.320, 0.265, 0.690, 0.150, 0.060 }; // SMPTE EG 432-1 (P3-D65)
constexpr Chromaticity kRec2020Primaries { 0.708, 0.292, 0.170, 0.797, 0.131, 0.046 };   // ITU-R BT.2020
constexpr double kWhiteX = 0.3127;
constexpr double kWhiteY = 0.3290;

// RGB-to-XYZ from chromaticities: each primary column is scaled so RGB (1,1,1) maps to D65 at Y = 1.
Matrix3 matrixFromChromaticity(const Chromaticity& value)
{
    const auto column = [] (double x, double y) {
        return std::array<double, 3> { x / y, 1.0, (1.0 - x - y) / y };
    };
    const auto r = column(value.rx, value.ry);
    const auto g = column(value.gx, value.gy);
    const auto b = column(value.bx, value.by);
    const auto white = column(kWhiteX, kWhiteY);
    const Matrix3 primaries { r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2] };
    const auto scale = transform3(inverse3(primaries), white);
    return { r[0] * scale[0], g[0] * scale[1], b[0] * scale[2],
             r[1] * scale[0], g[1] * scale[1], b[1] * scale[2],
             r[2] * scale[0], g[2] * scale[1], b[2] * scale[2] };
}

void testMatrixRoundTrips()
{
    for (const auto space : { ColorSpace::SRGB, ColorSpace::DisplayP3, ColorSpace::Rec2020 })
    {
        const auto& toXyz = gpumath::rgbToXyz(space);
        const auto& fromXyz = gpumath::xyzToRgb(space);
        require(matrixNear(gpumath::multiply(fromXyz, toXyz), identityMatrix(), kMatrixTolerance),
                "XYZ to RGB does not invert RGB to XYZ");
        require(matrixNear(gpumath::multiply(toXyz, fromXyz), identityMatrix(), kMatrixTolerance),
                "RGB to XYZ does not invert XYZ to RGB");
        require(matrixNear(toXyz, inverse3(fromXyz), kMatrixTolerance),
                "published inverse disagrees with the forward matrix");
    }
}

void testPublishedChromaticityMatrices()
{
    require(matrixNear(gpumath::rgbToXyz(ColorSpace::SRGB), matrixFromChromaticity(kSrgbPrimaries), kMatrixTolerance),
            "sRGB RGB-to-XYZ matrix disagrees with BT.709 primaries");
    require(matrixNear(gpumath::rgbToXyz(ColorSpace::DisplayP3), matrixFromChromaticity(kDisplayP3Primaries), kMatrixTolerance),
            "Display P3 RGB-to-XYZ matrix disagrees with P3-D65 primaries");
    require(matrixNear(gpumath::rgbToXyz(ColorSpace::Rec2020), matrixFromChromaticity(kRec2020Primaries), kMatrixTolerance),
            "Rec.2020 RGB-to-XYZ matrix disagrees with BT.2020 primaries");

    // Each space maps RGB (1,1,1) to the D65 white point, X = 0.950456, Y = 1, Z = 1.089058.
    for (const auto space : { ColorSpace::SRGB, ColorSpace::DisplayP3, ColorSpace::Rec2020 })
    {
        const auto white = transform3(gpumath::rgbToXyz(space), { 1.0, 1.0, 1.0 });
        require(near(white[0], 0.950456, kMatrixTolerance) && near(white[1], 1.0, kMatrixTolerance)
                    && near(white[2], 1.089058, kMatrixTolerance),
                "RGB white does not map to D65");
    }
}

void testKnownPrimaryMappings()
{
    // sRGB red in Display P3 linear.
    const auto srgbRedInP3 = transform3(gpumath::multiply(gpumath::xyzToRgb(ColorSpace::DisplayP3),
                                                          gpumath::rgbToXyz(ColorSpace::SRGB)),
                                        { 1.0, 0.0, 0.0 });
    require(near(srgbRedInP3[0], 0.822462, kMatrixTolerance) && near(srgbRedInP3[1], 0.033194, kMatrixTolerance)
                && near(srgbRedInP3[2], 0.017083, kMatrixTolerance),
            "sRGB red did not map to the published Display P3 values");

    // Pure Display P3 red in linear sRGB is outside sRGB: R exceeds 1, G and B go negative.
    const auto p3RedInSrgb = transform3(gpumath::multiply(gpumath::xyzToRgb(ColorSpace::SRGB),
                                                          gpumath::rgbToXyz(ColorSpace::DisplayP3)),
                                        { 1.0, 0.0, 0.0 });
    require(near(p3RedInSrgb[0], 1.224940, kMatrixTolerance) && near(p3RedInSrgb[1], -0.042057, kMatrixTolerance)
                && near(p3RedInSrgb[2], -0.019638, kMatrixTolerance),
            "Display P3 red did not map to the expected linear sRGB values");
    require(p3RedInSrgb[1] < 0.0 && p3RedInSrgb[2] < 0.0, "Display P3 red was not out of sRGB gamut");

    // Pure Rec.2020 red in linear sRGB is far outside sRGB.
    const auto rec2020RedInSrgb = transform3(gpumath::multiply(gpumath::xyzToRgb(ColorSpace::SRGB),
                                                               gpumath::rgbToXyz(ColorSpace::Rec2020)),
                                             { 1.0, 0.0, 0.0 });
    require(near(rec2020RedInSrgb[0], 1.660491, kMatrixTolerance) && near(rec2020RedInSrgb[1], -0.124550, kMatrixTolerance)
                && near(rec2020RedInSrgb[2], -0.018151, kMatrixTolerance),
            "Rec.2020 red did not map to the expected linear sRGB values");

    // Pure Display P3 red in linear Rec.2020 keeps R and G positive, with a small negative B.
    const auto p3RedInRec2020 = transform3(gpumath::multiply(gpumath::xyzToRgb(ColorSpace::Rec2020),
                                                             gpumath::rgbToXyz(ColorSpace::DisplayP3)),
                                           { 1.0, 0.0, 0.0 });
    require(near(p3RedInRec2020[0], 0.753833, kMatrixTolerance) && near(p3RedInRec2020[1], 0.045744, kMatrixTolerance)
                && near(p3RedInRec2020[2], -0.001210, kMatrixTolerance),
            "Display P3 red did not map to the expected linear Rec.2020 values");
}

// Test-only mirror of the SDR output policy: negative linear light is clipped before the
// sRGB encode, and the 8-bit store clamps to [0, 1].
std::uint8_t sdrSrgbByte(double linear)
{
    constexpr double kThreshold = 0.0031308;
    const double clipped = std::max(linear, 0.0);
    const double encoded = clipped <= kThreshold ? 12.92 * clipped
                                                 : 1.055 * std::pow(clipped, 1.0 / 2.4) - 0.055;
    return static_cast<std::uint8_t>(std::lround(std::clamp(encoded, 0.0, 1.0) * 255.0));
}

void testSdrOutputClipsOutOfGamut()
{
    const auto p3Red = transform3(gpumath::multiply(gpumath::xyzToRgb(ColorSpace::SRGB),
                                                    gpumath::rgbToXyz(ColorSpace::DisplayP3)),
                                  { 1.0, 0.0, 0.0 });
    const auto rec2020Red = transform3(gpumath::multiply(gpumath::xyzToRgb(ColorSpace::SRGB),
                                                         gpumath::rgbToXyz(ColorSpace::Rec2020)),
                                       { 1.0, 0.0, 0.0 });
    require(sdrSrgbByte(p3Red[0]) == 255 && sdrSrgbByte(p3Red[1]) == 0 && sdrSrgbByte(p3Red[2]) == 0,
            "Display P3 red did not clip to sRGB byte (255, 0, 0)");
    require(sdrSrgbByte(rec2020Red[0]) == 255 && sdrSrgbByte(rec2020Red[1]) == 0 && sdrSrgbByte(rec2020Red[2]) == 0,
            "Rec.2020 red did not clip to sRGB byte (255, 0, 0)");
    require(sdrSrgbByte(1.0) == 255 && sdrSrgbByte(0.0) == 0, "SDR output endpoints changed");
}

// Scalar mirrors of decodeTransfer and encodeTransfer in color_transform_gl.cpp.
double decodeSrgb(double encoded)
{
    return encoded <= 0.04045 ? encoded / 12.92 : std::pow((encoded + 0.055) / 1.055, 2.4);
}

double encodeSrgb(double linear)
{
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

double decodeGamma22(double encoded)
{
    return std::pow(encoded, 2.2);
}

double encodeGamma22(double linear)
{
    return std::pow(linear, 1.0 / 2.2);
}

// BT.709 nominal OETF, also used for BT.2020 SDR. Its 1.099 and 0.018 constants leave a
// small step at the knee, so the encoded-domain round trip is checked away from that knee.
double decodeRec709(double encoded)
{
    return encoded < 0.081 ? encoded / 4.5 : std::pow((encoded + 0.099) / 1.099, 1.0 / 0.45);
}

double encodeRec709(double linear)
{
    return linear < 0.018 ? 4.5 * linear : 1.099 * std::pow(linear, 0.45) - 0.099;
}

constexpr double kHlgA = 0.17883277;
constexpr double kHlgB = 0.28466892;
constexpr double kHlgC = 0.55991073;

double decodeHlg(double encoded)
{
    return encoded <= 0.5 ? encoded * encoded / 3.0
                          : (std::exp((encoded - kHlgC) / kHlgA) + kHlgB) / 12.0;
}

double encodeHlg(double linear)
{
    return linear <= 1.0 / 12.0 ? std::sqrt(3.0 * linear)
                                : kHlgA * std::log(12.0 * linear - kHlgB) + kHlgC;
}

constexpr double kPqM1 = 2610.0 / 16384.0;
constexpr double kPqM2 = 2523.0 / 32.0;
constexpr double kPqC1 = 3424.0 / 4096.0;
constexpr double kPqC2 = 2413.0 / 128.0;
constexpr double kPqC3 = 2392.0 / 128.0;

double decodePqNits(double encoded)
{
    const double power = std::pow(encoded, 1.0 / kPqM2);
    return 10000.0 * std::pow(std::max(power - kPqC1, 0.0) / (kPqC2 - kPqC3 * power), 1.0 / kPqM1);
}

double encodePqNits(double nits)
{
    const double power = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), kPqM1);
    return std::pow((kPqC1 + kPqC2 * power) / (1.0 + kPqC3 * power), kPqM2);
}

template <typename Encode, typename Decode>
void requireLinearRoundTrip(Encode encode, Decode decode, const char* message)
{
    for (int index = 0; index <= 1024; ++index)
    {
        const double linear = index / 1024.0;
        require(near(decode(encode(linear)), linear, kTransferTolerance), message);
    }
}

template <typename Encode, typename Decode>
void requireEncodedRoundTrip(Encode encode, Decode decode, const char* message)
{
    for (int index = 0; index <= 1024; ++index)
    {
        const double encoded = index / 1024.0;
        require(near(encode(decode(encoded)), encoded, kTransferTolerance), message);
    }
}

void testTransferRoundTrips()
{
    requireLinearRoundTrip(encodeSrgb, decodeSrgb, "sRGB linear round trip failed");
    requireEncodedRoundTrip(encodeSrgb, decodeSrgb, "sRGB encoded round trip failed");
    requireLinearRoundTrip(encodeGamma22, decodeGamma22, "gamma 2.2 linear round trip failed");
    requireEncodedRoundTrip(encodeGamma22, decodeGamma22, "gamma 2.2 encoded round trip failed");
    requireLinearRoundTrip(encodeRec709, decodeRec709, "Rec.709 linear round trip failed");
    for (const double encoded : { 0.0, 0.02, 0.1, 0.5, 0.9, 1.0 })
        require(near(encodeRec709(decodeRec709(encoded)), encoded, kTransferTolerance),
                "Rec.709 encoded round trip failed away from the knee");
    requireLinearRoundTrip(encodeHlg, decodeHlg, "HLG linear round trip failed");
    requireEncodedRoundTrip(encodeHlg, decodeHlg, "HLG encoded round trip failed");

    for (const double nits : { 0.5, 1.0, 10.0, 100.0, 203.0, 1000.0, 4000.0, 10000.0 })
        require(near(decodePqNits(encodePqNits(nits)), nits, 1.0e-6 * std::max(1.0, nits)),
                "PQ nits round trip failed");
}

Description sdrPair(ColorSpace inputSpace, TransferFunction inputTransfer,
                    ColorSpace outputSpace, TransferFunction outputTransfer)
{
    Description value;
    value.input.extent = { 4, 4 };
    value.input.format = PixelFormat::RGBA8;
    value.input.colorSpace = inputSpace;
    value.input.transfer = inputTransfer;
    value.input.alpha = AlphaMode::Opaque;
    value.output = value.input;
    value.output.colorSpace = outputSpace;
    value.output.transfer = outputTransfer;
    return value;
}

void requireRefused(const Description& value, AdmissionFailure expected, const char* message)
{
    AdmissionFailure failure = AdmissionFailure::None;
    const auto admitted = admit(value, BackendCapability::NativeGpu, failure);
    require(! admitted && failure == expected, message);
}

// Admits the description and returns its stage order. Refusal fails the check.
std::vector<StageKind> requireAdmittedStages(const Description& value, const char* message)
{
    AdmissionFailure failure = AdmissionFailure::None;
    const auto admitted = admit(value, BackendCapability::NativeGpu, failure);
    require(admitted && failure == AdmissionFailure::None, message);
    std::vector<StageKind> stages;
    for (std::size_t index = 0; index < admitted->stageCount(); ++index)
        stages.push_back(admitted->stages()[index]);
    return stages;
}

void testBt2020AndP3Admission()
{
    // BT.2020 SDR input: Rec.709 transfer, then Rec.2020 to linear sRGB working primaries.
    const auto bt2020Input = requireAdmittedStages(
        sdrPair(ColorSpace::Rec2020, TransferFunction::Rec709, ColorSpace::SRGB, TransferFunction::SRGB),
        "BT.2020 SDR input was refused");
    require(bt2020Input == std::vector<StageKind> { StageKind::DecodeTransfer,
                                                    StageKind::ConvertToWorkingPrimaries,
                                                    StageKind::EncodeTransfer },
            "BT.2020 SDR input stage order changed");

    // BT.2020 SDR output: sRGB linear working primaries to Rec.2020, encoded with the Rec.709 transfer.
    const auto bt2020Output = requireAdmittedStages(
        sdrPair(ColorSpace::SRGB, TransferFunction::SRGB, ColorSpace::Rec2020, TransferFunction::Rec709),
        "BT.2020 SDR output was refused");
    require(bt2020Output == std::vector<StageKind> { StageKind::DecodeTransfer,
                                                     StageKind::ConvertFromWorkingPrimaries,
                                                     StageKind::EncodeTransfer },
            "BT.2020 SDR output stage order changed");

    // Display P3 scene-linear input to linear sRGB, admitted the same way the managed plane admits it.
    auto p3Linear = sdrPair(ColorSpace::DisplayP3, TransferFunction::Linear,
                            ColorSpace::LinearSRGB, TransferFunction::Linear);
    p3Linear.input.format = PixelFormat::RGBA16F;
    p3Linear.output.format = PixelFormat::RGBA16F;
    p3Linear.outputIntent = OutputIntent::SceneLinear;
    const auto p3Stages = requireAdmittedStages(p3Linear, "Display P3 linear input was refused");
    require(p3Stages == std::vector<StageKind> { StageKind::ConvertToWorkingPrimaries },
            "Display P3 linear stage order changed");

    requireRefused(sdrPair(ColorSpace::Rec2020, TransferFunction::SRGB, ColorSpace::SRGB, TransferFunction::SRGB),
                   AdmissionFailure::InvalidColorTransferPair,
                   "Rec.2020 with the sRGB transfer was admitted");
    requireRefused(sdrPair(ColorSpace::DisplayP3, TransferFunction::Gamma22, ColorSpace::SRGB, TransferFunction::SRGB),
                   AdmissionFailure::InvalidColorTransferPair,
                   "Display P3 with the gamma 2.2 transfer was admitted");

    auto hdrIntent = sdrPair(ColorSpace::SRGB, TransferFunction::SRGB, ColorSpace::Rec2020, TransferFunction::Rec709);
    hdrIntent.outputIntent = OutputIntent::HdrPq;
    requireRefused(hdrIntent, AdmissionFailure::InvalidOutputIntent,
                   "BT.2020 SDR output was admitted as an HDR PQ intent");
    requireRefused(sdrPair(ColorSpace::SRGB, TransferFunction::SRGB, ColorSpace::Rec2020, TransferFunction::PQ),
                   AdmissionFailure::InvalidOutputIntent,
                   "Rec.2020 PQ was admitted as an SDR output");
    requireRefused(sdrPair(ColorSpace::SRGB, TransferFunction::SRGB, ColorSpace::DisplayP3, TransferFunction::Linear),
                   AdmissionFailure::InvalidOutputIntent,
                   "linear Display P3 was admitted as an SDR output");
}
} // namespace

int main()
{
    testMatrixRoundTrips();
    testPublishedChromaticityMatrices();
    testKnownPrimaryMappings();
    testSdrOutputClipsOutOfGamut();
    testTransferRoundTrips();
    testBt2020AndP3Admission();
    std::cout << "color transform gamut numeric: all checks passed\n";
    return EXIT_SUCCESS;
}
