#pragma once

#include <array>
#include <cstddef>

namespace diffractionmaterial
{
// Measured aluminium, from K. M. McPeak et al., "Plasmonic films can easily be better:
// Rules and recipes", ACS Photonics 2, 326-333 (2015), doi:10.1021/ph5004237. Thermally
// evaporated at 150 A/s and template stripped from a Si wafer. Values are every tenth
// row of the CC0 refractiveindex.info file database/data/main/Al/nk/McPeak.yml, unsmoothed.
// A pressed disc's sputtered film is a different film, so this is the nearest measurement.
inline constexpr double kAluminiumTableFirstNanometres = 380.0;
inline constexpr double kAluminiumTableStepNanometres = 10.0;
struct ComplexIndexRow final
{
    double n;
    double k;
};
inline constexpr std::array<ComplexIndexRow, 41> kAluminiumMcPeak2015 { {
    { 0.335956, 4.002607 }, // 380
    { 0.354902, 4.114238 }, // 390
    { 0.375151, 4.226433 }, // 400
    { 0.396086, 4.336806 }, // 410
    { 0.417648, 4.447407 }, // 420
    { 0.440996, 4.559109 }, // 430
    { 0.464233, 4.669102 }, // 440
    { 0.489220, 4.778319 }, // 450
    { 0.514817, 4.887948 }, // 460
    { 0.539877, 4.997346 }, // 470
    { 0.568005, 5.105941 }, // 480
    { 0.596705, 5.212874 }, // 490
    { 0.625686, 5.320478 }, // 500
    { 0.655710, 5.428163 }, // 510
    { 0.688336, 5.535956 }, // 520
    { 0.720794, 5.641768 }, // 530
    { 0.754468, 5.746497 }, // 540
    { 0.789405, 5.851937 }, // 550
    { 0.829205, 5.958236 }, // 560
    { 0.867377, 6.060226 }, // 570
    { 0.908570, 6.165184 }, // 580
    { 0.948956, 6.267417 }, // 590
    { 0.992466, 6.368986 }, // 600
    { 1.038146, 6.466819 }, // 610
    { 1.088160, 6.566589 }, // 620
    { 1.136329, 6.663126 }, // 630
    { 1.190265, 6.759418 }, // 640
    { 1.246364, 6.852330 }, // 650
    { 1.304383, 6.942364 }, // 660
    { 1.365410, 7.031952 }, // 670
    { 1.426024, 7.116287 }, // 680
    { 1.493231, 7.197481 }, // 690
    { 1.559752, 7.273914 }, // 700
    { 1.631622, 7.345760 }, // 710
    { 1.706781, 7.414250 }, // 720
    { 1.786399, 7.475374 }, // 730
    { 1.872189, 7.528353 }, // 740
    { 1.958355, 7.571404 }, // 750
    { 2.054224, 7.599463 }, // 760
    { 2.144529, 7.612162 }, // 770
    { 2.231500, 7.603854 }, // 780
} };

// Straight lines between rows. The caller keeps the wavelength inside 380 to 780 nm.
inline constexpr ComplexIndexRow aluminiumIndexAt(double wavelengthNanometres) noexcept
{
    const double position = (wavelengthNanometres - kAluminiumTableFirstNanometres)
        / kAluminiumTableStepNanometres;
    const auto row = position >= 40.0 ? std::size_t { 39 } : static_cast<std::size_t>(position);
    const double along = position - static_cast<double>(row);
    // Copied, since device code cannot bind a reference to a host global.
    constexpr auto table = kAluminiumMcPeak2015;
    const ComplexIndexRow low = table[row];
    const ComplexIndexRow high = table[row + 1];
    return { low.n + (high.n - low.n) * along, low.k + (high.k - low.k) * along };
}
} // namespace diffractionmaterial
