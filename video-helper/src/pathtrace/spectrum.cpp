#include "spectrum.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace videohelper::pathtrace
{
namespace
{

constexpr int kTableSize = 81;  // 380 to 780 nm every 5 nm
constexpr float kTableStepNm = 5.0f;

// CIE 1931 2-degree colour matching functions x, y, z at 380, 385, ..., 780 nm.
constexpr float kCie1931[kTableSize][3] {
    { 0.001368f, 3.9e-05f, 0.006450001f },  // 380
    { 0.002236f, 6.4e-05f, 0.01054999f },  // 385
    { 0.004243f, 0.00012f, 0.02005001f },  // 390
    { 0.00765f, 0.000217f, 0.03621f },  // 395
    { 0.01431f, 0.000396f, 0.06785001f },  // 400
    { 0.02319f, 0.00064f, 0.1102f },  // 405
    { 0.04351f, 0.00121f, 0.2074f },  // 410
    { 0.07763f, 0.00218f, 0.3713f },  // 415
    { 0.13438f, 0.004f, 0.6456f },  // 420
    { 0.21477f, 0.0073f, 1.03905f },  // 425
    { 0.2839f, 0.0116f, 1.3856f },  // 430
    { 0.3285f, 0.01684f, 1.62296f },  // 435
    { 0.34828f, 0.023f, 1.74706f },  // 440
    { 0.34806f, 0.0298f, 1.7826f },  // 445
    { 0.3362f, 0.038f, 1.77211f },  // 450
    { 0.3187f, 0.048f, 1.7441f },  // 455
    { 0.2908f, 0.06f, 1.6692f },  // 460
    { 0.2511f, 0.0739f, 1.5281f },  // 465
    { 0.19536f, 0.09098f, 1.28764f },  // 470
    { 0.1421f, 0.1126f, 1.0419f },  // 475
    { 0.09564f, 0.13902f, 0.8129501f },  // 480
    { 0.05795001f, 0.1693f, 0.6162f },  // 485
    { 0.03201f, 0.20802f, 0.46518f },  // 490
    { 0.0147f, 0.2586f, 0.3533f },  // 495
    { 0.0049f, 0.323f, 0.272f },  // 500
    { 0.0024f, 0.4073f, 0.2123f },  // 505
    { 0.0093f, 0.503f, 0.1582f },  // 510
    { 0.0291f, 0.6082f, 0.1117f },  // 515
    { 0.06327f, 0.71f, 0.07824999f },  // 520
    { 0.1096f, 0.7932f, 0.05725001f },  // 525
    { 0.1655f, 0.862f, 0.04216f },  // 530
    { 0.2257499f, 0.9148501f, 0.02984f },  // 535
    { 0.2904f, 0.954f, 0.0203f },  // 540
    { 0.3597f, 0.9803f, 0.0134f },  // 545
    { 0.4334499f, 0.9949501f, 0.008749999f },  // 550
    { 0.5120501f, 1.0f, 0.005749999f },  // 555
    { 0.5945f, 0.995f, 0.0039f },  // 560
    { 0.6784f, 0.9786f, 0.002749999f },  // 565
    { 0.7621f, 0.952f, 0.0021f },  // 570
    { 0.8425f, 0.9154f, 0.0018f },  // 575
    { 0.9163f, 0.87f, 0.001650001f },  // 580
    { 0.9786f, 0.8163f, 0.0014f },  // 585
    { 1.0263f, 0.757f, 0.0011f },  // 590
    { 1.0567f, 0.6949f, 0.001f },  // 595
    { 1.0622f, 0.631f, 0.0008f },  // 600
    { 1.0456f, 0.5668f, 0.0006f },  // 605
    { 1.0026f, 0.503f, 0.00034f },  // 610
    { 0.9384f, 0.4412f, 0.00024f },  // 615
    { 0.8544499f, 0.381f, 0.00019f },  // 620
    { 0.7514f, 0.321f, 0.0001f },  // 625
    { 0.6424f, 0.265f, 4.999999e-05f },  // 630
    { 0.5419f, 0.217f, 3e-05f },  // 635
    { 0.4479f, 0.175f, 2e-05f },  // 640
    { 0.3608f, 0.1382f, 1e-05f },  // 645
    { 0.2835f, 0.107f, -1.905824e-21f },  // 650
    { 0.2187f, 0.0816f, 0.0f },  // 655
    { 0.1649f, 0.061f, 0.0f },  // 660
    { 0.1212f, 0.04458f, 0.0f },  // 665
    { 0.0874f, 0.032f, 0.0f },  // 670
    { 0.0636f, 0.0232f, 0.0f },  // 675
    { 0.04677f, 0.017f, 0.0f },  // 680
    { 0.0329f, 0.01192f, 0.0f },  // 685
    { 0.0227f, 0.00821f, 0.0f },  // 690
    { 0.01584f, 0.005723f, 0.0f },  // 695
    { 0.01135916f, 0.004102f, 0.0f },  // 700
    { 0.008110916f, 0.002929f, 0.0f },  // 705
    { 0.005790346f, 0.002091f, 0.0f },  // 710
    { 0.004109457f, 0.001484f, 0.0f },  // 715
    { 0.002899327f, 0.001047f, 0.0f },  // 720
    { 0.00204919f, 0.00074f, 0.0f },  // 725
    { 0.001439971f, 0.00052f, 0.0f },  // 730
    { 0.0009999493f, 0.0003611f, 0.0f },  // 735
    { 0.0006900786f, 0.0002492f, 0.0f },  // 740
    { 0.0004760213f, 0.0001719f, 0.0f },  // 745
    { 0.0003323011f, 0.00012f, 0.0f },  // 750
    { 0.0002348261f, 8.48e-05f, 0.0f },  // 755
    { 0.0001661505f, 6e-05f, 0.0f },  // 760
    { 0.000117413f, 4.24e-05f, 0.0f },  // 765
    { 8.307527e-05f, 3e-05f, 0.0f },  // 770
    { 5.870652e-05f, 2.12e-05f, 0.0f },  // 775
    { 4.150994e-05f, 1.499e-05f, 0.0f },  // 780
};

// CIE standard illuminant D65, relative spectral power at 380, 385, ..., 780 nm.
constexpr float kD65[kTableSize] {
    49.9755f, 52.3118f, 54.6482f, 68.7015f, 82.7549f, 87.1204f, 91.486f, 92.4589f,
    93.4318f, 90.057f, 86.6823f, 95.7736f, 104.865f, 110.936f, 117.008f, 117.41f,
    117.812f, 116.336f, 114.861f, 115.392f, 115.923f, 112.367f, 108.811f, 109.082f,
    109.354f, 108.578f, 107.802f, 106.296f, 104.79f, 106.239f, 107.689f, 106.047f,
    104.405f, 104.225f, 104.046f, 102.023f, 100.0f, 98.1671f, 96.3342f, 96.0611f,
    95.788f, 92.2368f, 88.6856f, 89.3459f, 90.0062f, 89.8026f, 89.5991f, 88.6489f,
    87.6987f, 85.4936f, 83.2886f, 83.4939f, 83.6992f, 81.863f, 80.0268f, 80.1207f,
    80.2146f, 81.2462f, 82.2778f, 80.281f, 78.2842f, 74.0027f, 69.7213f, 70.6652f,
    71.6091f, 72.979f, 74.349f, 67.9765f, 61.604f, 65.7448f, 69.8856f, 72.4863f,
    75.087f, 69.3398f, 63.5927f, 55.0054f, 46.4182f, 56.6118f, 66.8054f, 65.0941f,
    63.3828f,
};

// Red, green and blue basis spectra before the exact correction, at 380, 385,
// ..., 780 nm. Fitted offline (softmax of cubic B-splines with 12 knots, with a
// small curvature penalty) so the three are smooth, non-negative and sum to 1,
// and so that each converts back close to its own primary.
constexpr float kRawBasis[kTableSize][3] {
    { 0.002356f, 0.046993f, 0.950651f },  // 380
    { 0.003283f, 0.049837f, 0.946880f },  // 385
    { 0.004370f, 0.049962f, 0.945668f },  // 390
    { 0.005579f, 0.047830f, 0.946592f },  // 395
    { 0.006861f, 0.044156f, 0.948983f },  // 400
    { 0.008162f, 0.039699f, 0.952139f },  // 405
    { 0.009434f, 0.035106f, 0.955460f },  // 410
    { 0.010641f, 0.030844f, 0.958515f },  // 415
    { 0.011769f, 0.027202f, 0.961029f },  // 420
    { 0.012824f, 0.024333f, 0.962842f },  // 425
    { 0.013827f, 0.022299f, 0.963874f },  // 430
    { 0.014792f, 0.021113f, 0.964095f },  // 435
    { 0.015740f, 0.020823f, 0.963437f },  // 440
    { 0.016698f, 0.021567f, 0.961736f },  // 445
    { 0.017698f, 0.023643f, 0.958659f },  // 450
    { 0.018777f, 0.027644f, 0.953580f },  // 455
    { 0.019965f, 0.034713f, 0.945322f },  // 460
    { 0.021275f, 0.047086f, 0.931639f },  // 465
    { 0.022663f, 0.069207f, 0.908130f },  // 470
    { 0.023919f, 0.109154f, 0.866927f },  // 475
    { 0.024542f, 0.178617f, 0.796841f },  // 480
    { 0.023773f, 0.289705f, 0.686522f },  // 485
    { 0.020993f, 0.442384f, 0.536622f },  // 490
    { 0.016504f, 0.610969f, 0.372527f },  // 495
    { 0.011637f, 0.756391f, 0.231972f },  // 500
    { 0.007642f, 0.857485f, 0.134873f },  // 505
    { 0.004900f, 0.918141f, 0.076959f },  // 510
    { 0.003197f, 0.951791f, 0.045012f },  // 515
    { 0.002188f, 0.970059f, 0.027753f },  // 520
    { 0.001603f, 0.980146f, 0.018252f },  // 525
    { 0.001279f, 0.985833f, 0.012888f },  // 530
    { 0.001127f, 0.989055f, 0.009817f },  // 535
    { 0.001113f, 0.990791f, 0.008096f },  // 540
    { 0.001248f, 0.991503f, 0.007249f },  // 545
    { 0.001606f, 0.991326f, 0.007068f },  // 550
    { 0.002402f, 0.990077f, 0.007521f },  // 555
    { 0.004220f, 0.987032f, 0.008748f },  // 560
    { 0.008613f, 0.980353f, 0.011034f },  // 565
    { 0.019671f, 0.965545f, 0.014783f },  // 570
    { 0.048017f, 0.931539f, 0.020444f },  // 575
    { 0.117562f, 0.854573f, 0.027865f },  // 580
    { 0.262961f, 0.702350f, 0.034689f },  // 585
    { 0.483036f, 0.480883f, 0.036081f },  // 590
    { 0.697747f, 0.271703f, 0.030550f },  // 595
    { 0.838657f, 0.138670f, 0.022673f },  // 600
    { 0.911879f, 0.071814f, 0.016307f },  // 605
    { 0.947611f, 0.040316f, 0.012073f },  // 610
    { 0.965617f, 0.024985f, 0.009398f },  // 615
    { 0.975129f, 0.017134f, 0.007737f },  // 620
    { 0.980310f, 0.012956f, 0.006734f },  // 625
    { 0.983084f, 0.010736f, 0.006180f },  // 630
    { 0.984363f, 0.009679f, 0.005958f },  // 635
    { 0.984572f, 0.009422f, 0.006006f },  // 640
    { 0.983873f, 0.009824f, 0.006303f },  // 645
    { 0.982266f, 0.010882f, 0.006852f },  // 650
    { 0.979657f, 0.012678f, 0.007665f },  // 655
    { 0.975877f, 0.015366f, 0.008757f },  // 660
    { 0.970701f, 0.019163f, 0.010137f },  // 665
    { 0.963886f, 0.024316f, 0.011798f },  // 670
    { 0.955249f, 0.031052f, 0.013699f },  // 675
    { 0.944777f, 0.039475f, 0.015748f },  // 680
    { 0.932774f, 0.049431f, 0.017795f },  // 685
    { 0.920006f, 0.060362f, 0.019632f },  // 690
    { 0.907694f, 0.071276f, 0.021030f },  // 695
    { 0.896751f, 0.081371f, 0.021878f },  // 700
    { 0.887708f, 0.090127f, 0.022165f },  // 705
    { 0.880906f, 0.097173f, 0.021921f },  // 710
    { 0.876486f, 0.102302f, 0.021212f },  // 715
    { 0.874409f, 0.105467f, 0.020124f },  // 720
    { 0.874488f, 0.106758f, 0.018754f },  // 725
    { 0.876429f, 0.106376f, 0.017195f },  // 730
    { 0.879868f, 0.104594f, 0.015537f },  // 735
    { 0.884465f, 0.101684f, 0.013850f },  // 740
    { 0.890144f, 0.097702f, 0.012154f },  // 745
    { 0.896876f, 0.092660f, 0.010464f },  // 750
    { 0.904569f, 0.086622f, 0.008809f },  // 755
    { 0.913069f, 0.079705f, 0.007227f },  // 760
    { 0.922164f, 0.072079f, 0.005757f },  // 765
    { 0.931598f, 0.063964f, 0.004438f },  // 770
    { 0.941087f, 0.055615f, 0.003298f },  // 775
    { 0.950340f, 0.047305f, 0.002355f },  // 780
};

// IEC 61966-2-1: CIE XYZ (D65 white) to linear sRGB.
constexpr double kXyzToSrgb[3][3] { { 3.2404542, -1.5371385, -0.4985314 },
                                    { -0.9692660, 1.8760108, 0.0415560 },
                                    { 0.0556434, -0.2040259, 1.0572252 } };

#if ! defined(__CUDACC__)
struct Tables
{
    float weight[kTableSize][3] {};  // linear sRGB per nm, each channel integrates to 1
    float basis[kTableSize][3] {};   // corrected basis spectra

    Tables() noexcept
    {
        // Weights: M * xyz * D65, normalized by the integral of the linear
        // interpolant (the trapezoid rule on the 5 nm grid).
        double raw[kTableSize][3] {};
        double integral[3] {};
        for (int k = 0; k < kTableSize; ++k)
            for (int c = 0; c < 3; ++c)
            {
                double sum = 0.0;
                for (int j = 0; j < 3; ++j)
                    sum += kXyzToSrgb[c][j] * static_cast<double> (kCie1931[k][j]);
                raw[k][c] = sum * static_cast<double> (kD65[k]);
            }
        for (int k = 0; k + 1 < kTableSize; ++k)
            for (int c = 0; c < 3; ++c)
                integral[c] += 0.5 * static_cast<double> (kTableStepNm) * (raw[k][c] + raw[k + 1][c]);
        double w[kTableSize][3] {};
        for (int k = 0; k < kTableSize; ++k)
            for (int c = 0; c < 3; ++c)
            {
                w[k][c] = raw[k][c] / integral[c];
                weight[k][c] = static_cast<float> (w[k][c]);
            }

        // Colour of each raw basis spectrum: the exact integral of the product
        // of two linear interpolants on the same grid.
        double colour[3][3] {};  // colour[basis][channel]
        for (int k = 0; k + 1 < kTableSize; ++k)
            for (int b = 0; b < 3; ++b)
                for (int c = 0; c < 3; ++c)
                {
                    const double a0 = kRawBasis[k][b], a1 = kRawBasis[k + 1][b];
                    const double w0 = w[k][c], w1 = w[k + 1][c];
                    colour[b][c] += static_cast<double> (kTableStepNm)
                                  * (2.0 * a0 * w0 + a0 * w1 + a1 * w0 + 2.0 * a1 * w1) / 6.0;
                }
        // basis' = inverse(colour) * basis, so basis' b converts to primary b.
        // Every column of colour sums to 1 (the raw spectra sum to 1 and each
        // weight integrates to 1), so every column of the inverse does too and
        // the corrected spectra still sum to 1.
        const double det = colour[0][0] * (colour[1][1] * colour[2][2] - colour[1][2] * colour[2][1])
                         - colour[0][1] * (colour[1][0] * colour[2][2] - colour[1][2] * colour[2][0])
                         + colour[0][2] * (colour[1][0] * colour[2][1] - colour[1][1] * colour[2][0]);
        double inverse[3][3] {};
        inverse[0][0] = (colour[1][1] * colour[2][2] - colour[1][2] * colour[2][1]) / det;
        inverse[0][1] = (colour[0][2] * colour[2][1] - colour[0][1] * colour[2][2]) / det;
        inverse[0][2] = (colour[0][1] * colour[1][2] - colour[0][2] * colour[1][1]) / det;
        inverse[1][0] = (colour[1][2] * colour[2][0] - colour[1][0] * colour[2][2]) / det;
        inverse[1][1] = (colour[0][0] * colour[2][2] - colour[0][2] * colour[2][0]) / det;
        inverse[1][2] = (colour[0][2] * colour[1][0] - colour[0][0] * colour[1][2]) / det;
        inverse[2][0] = (colour[1][0] * colour[2][1] - colour[1][1] * colour[2][0]) / det;
        inverse[2][1] = (colour[0][1] * colour[2][0] - colour[0][0] * colour[2][1]) / det;
        inverse[2][2] = (colour[0][0] * colour[1][1] - colour[0][1] * colour[1][0]) / det;
        // The table rounding leaves the sums within about 1e-6 of 1; dividing
        // by the sum makes a flat white exact again.
        for (int k = 0; k < kTableSize; ++k)
        {
            double corrected[3] {};
            double total = 0.0;
            for (int b = 0; b < 3; ++b)
            {
                for (int e = 0; e < 3; ++e)
                    corrected[b] += inverse[b][e] * static_cast<double> (kRawBasis[k][e]);
                corrected[b] = std::max (0.0, corrected[b]);
                total += corrected[b];
            }
            for (int b = 0; b < 3; ++b)
                basis[k][b] = static_cast<float> (corrected[b] / total);
        }
    }
};

const Tables& tables() noexcept
{
    static const Tables instance;
    return instance;
}
#endif

// Device code reads the same values from the buffers the host uploads.
using TableRow = float[3];

PT_SHARED_ENTRY const PT_TABLE_SPACE TableRow* basisTable() noexcept
{
#if defined(__CUDACC__)
    return device::deviceLookupTables().spectrumBasis;
#else
    return tables().basis;
#endif
}

PT_SHARED_ENTRY const PT_TABLE_SPACE TableRow* weightTable() noexcept
{
#if defined(__CUDACC__)
    return device::deviceLookupTables().spectrumWeight;
#else
    return tables().weight;
#endif
}

PT_SHARED_ENTRY Vec3 lookup (const PT_TABLE_SPACE TableRow* table, float nm) noexcept
{
    const float position = std::min (std::max ((nm - kWavelengthMinNm) / kTableStepNm, 0.0f),
                                     static_cast<float> (kTableSize - 1));
    const int k = std::min (static_cast<int> (position), kTableSize - 2);
    const float t = position - static_cast<float> (k);
    return { table[k][0] + (table[k + 1][0] - table[k][0]) * t,
             table[k][1] + (table[k + 1][1] - table[k][1]) * t,
             table[k][2] + (table[k + 1][2] - table[k][2]) * t };
}

PT_SHARED_ENTRY float wrapWavelength (float nm) noexcept
{
    if (nm >= kWavelengthMaxNm)
        nm -= kWavelengthSpanNm;
    return std::min (std::max (nm, float (kWavelengthMinNm)), std::nextafter (kWavelengthMaxNm, kWavelengthMinNm));
}

} // namespace

PT_SHARED_ENTRY Wavelengths rgbChannels() noexcept
{
    return {};
}

PT_SHARED_ENTRY Wavelengths sampleHeroWavelengths (float u) noexcept
{
    Wavelengths out;
    out.spectral = true;
    const float clamped = std::isfinite (u) ? std::min (std::max (u, 0.0f), 1.0f) : 0.0f;
    const float hero = wrapWavelength (kWavelengthMinNm + kWavelengthSpanNm * clamped);
    const float spacing = kWavelengthSpanNm / static_cast<float> (kSpectrumChannels);
    for (int i = 0; i < kSpectrumChannels; ++i)
        out.nm[static_cast<std::size_t> (i)] = wrapWavelength (hero + spacing * static_cast<float> (i));
    return out;
}

PT_SHARED_ENTRY Spectrum upsampleRgb (Vec3 rgb, const Wavelengths& wavelengths) noexcept
{
    Spectrum out;
    if (! wavelengths.spectral)
    {
        out[0] = rgb.x;
        out[1] = rgb.y;
        out[2] = rgb.z;
        return out;
    }
    for (int i = 0; i < kSpectrumChannels; ++i)
    {
        const Vec3 b = lookup (basisTable(), wavelengths.nm[static_cast<std::size_t> (i)]);
        out[i] = std::max (0.0f, rgb.x * b.x + rgb.y * b.y + rgb.z * b.z);
    }
    return out;
}

PT_SHARED_ENTRY Vec3 channelsToRgb (const Spectrum& value, const Wavelengths& wavelengths) noexcept
{
    if (! wavelengths.spectral)
        return { value[0], value[1], value[2] };
    const float scale = kWavelengthSpanNm / static_cast<float> (kSpectrumChannels);
    Vec3 rgb {};
    for (int i = 0; i < kSpectrumChannels; ++i)
    {
        if (value[i] == 0.0f)
            continue;
        rgb = rgb + lookup (weightTable(), wavelengths.nm[static_cast<std::size_t> (i)]) * (value[i] * scale);
    }
    return rgb;
}

PT_SHARED_ENTRY Vec3 rgbBasisAt (float nm) noexcept
{
    return lookup (basisTable(), nm);
}

PT_SHARED_ENTRY Vec3 rgbMatchingWeightAt (float nm) noexcept
{
    return lookup (weightTable(), nm);
}

PT_SHARED_ENTRY float cauchyIor (float iorD, float cauchyB, float nm) noexcept
{
    if (! (cauchyB != 0.0f) || ! std::isfinite (cauchyB))
        return iorD;
    const float micrometres = nm * 1.0e-3f;
    const float reference = kDLineNm * 1.0e-3f;
    return iorD + cauchyB * (1.0f / (micrometres * micrometres) - 1.0f / (reference * reference));
}

PT_SHARED_ENTRY float cauchyBFromAbbe (float iorD, float abbe) noexcept
{
    if (! std::isfinite (abbe) || ! (abbe > 0.0f) || ! std::isfinite (iorD) || ! (iorD > 1.0f))
        return 0.0f;
    const float f = kFLineNm * 1.0e-3f;
    const float c = kCLineNm * 1.0e-3f;
    // n_F - n_C = B (1 / f^2 - 1 / c^2) = (n_d - 1) / V.
    return (iorD - 1.0f) / abbe / (1.0f / (f * f) - 1.0f / (c * c));
}

#if ! defined(__CUDACC__)
void exportSpectrumTables (device::DeviceLookupTables& out) noexcept
{
    const Tables& t = tables();
    out.spectrumWeight = t.weight;
    out.spectrumBasis = t.basis;
    out.spectrumRows = kTableSize;
}
#endif

} // namespace videohelper::pathtrace
