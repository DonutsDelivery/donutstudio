// Tests for the Environment Light's Physical sky (shared/PhysicalSky.h): a blue
// zenith under a high sun, reddening sun and horizon at a low sun, the sun as
// the brightest direction, the atmosphere controls, and the bake into the
// analytic sky every engine draws, and the dust's sun halo in the raster twin.

#include "../../shared/PhysicalSky.h"
#include "../src/gpu_backend/fixture_analytic_sky.h"

#include <cmath>
#include <iostream>
#include <string>

namespace
{

namespace sky = HarmonicMIDI::grid::physicalsky;
namespace grid = HarmonicMIDI::grid;

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

bool finite (const sky::Rgb& c)
{
    return std::isfinite (c[0]) && std::isfinite (c[1]) && std::isfinite (c[2])
        && c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0;
}

double redOverBlue (const sky::Rgb& c) { return c[0] / std::max (1.0e-30, c[2]); }

void earthTests()
{
    const sky::Atmosphere earth;
    const auto noon = sky::bake (earth, 60.0);
    const auto dusk = sky::bake (earth, 2.0);
    check (finite (noon.zenith) && finite (noon.horizon) && finite (noon.ground) && finite (noon.sunTransmittance),
           "the bake is finite and non-negative");
    check (noon.zenith[2] > noon.zenith[0], "a high sun gives a blue zenith");
    check (noon.sunTransmittance[2] > 0.5 && noon.sunTransmittance[0] > noon.sunTransmittance[2],
           "a high sun loses a little blue");
    check (redOverBlue (dusk.sunTransmittance) > 2.0 * redOverBlue (noon.sunTransmittance),
           "a low sun reddens");
    check (redOverBlue (dusk.horizon) > redOverBlue (noon.horizon), "the horizon reddens at a low sun");
    check (noon.zenith[1] > dusk.zenith[1], "the sky dims as the sun sets");

    const auto night = sky::bake (earth, -12.0);
    check (night.sunTransmittance[0] == 0.0 && night.sunTransmittance[2] == 0.0, "the planet hides a sun below the horizon");
    check (night.zenith[2] < 0.1 * noon.zenith[2], "twilight is far dimmer than day");
}

void controlTests()
{
    const auto noon = sky::bake (sky::Atmosphere {}, 45.0);
    const auto vacuum = sky::bake (sky::Atmosphere { 0.0f, 0.0f, 0.0f, 0.0f }, 45.0);
    check (vacuum.zenith[2] == 0.0 && vacuum.sunTransmittance[2] == 1.0, "no atmosphere scatters nothing and dims nothing");
    const auto dusty = sky::bake (sky::Atmosphere { 1.0f, 8.0f, 1.0f, 0.0f }, 45.0);
    check (redOverBlue (dusty.horizon) > redOverBlue (noon.horizon), "dust whitens the blue horizon");
    const auto thick = sky::bake (sky::Atmosphere { 4.0f, 1.0f, 1.0f, 0.0f }, 45.0);
    check (thick.sunTransmittance[2] < noon.sunTransmittance[2], "denser air dims the sun");
    const auto high = sky::bake (sky::Atmosphere { 1.0f, 1.0f, 1.0f, 40000.0f }, 45.0);
    check (high.zenith[2] < 0.3 * noon.zenith[2], "high altitude darkens the zenith");
    const auto noOzone = sky::bake (sky::Atmosphere { 1.0f, 1.0f, 0.0f, 0.0f }, 5.0);
    const auto ozone = sky::bake (sky::Atmosphere { 1.0f, 1.0f, 4.0f, 0.0f }, 5.0);
    check (ozone.sunTransmittance[1] < noOzone.sunTransmittance[1], "ozone absorbs green light");
    check (sky::validAtmosphere (sky::Atmosphere {}) && ! sky::validAtmosphere (sky::Atmosphere { -1.0f, 1.0f, 1.0f, 0.0f }),
           "atmosphere validity");
}

void analyticSkyTests()
{
    grid::SceneAnalyticSky analytic;
    analytic.version = 1;
    analytic.sunElevationDegrees = 30.0f;
    analytic.sunRadiusDegrees = 2.0f;
    sky::applyToAnalyticSky (sky::Atmosphere {}, 30.0f, analytic);
    check (analytic.zenith.z > analytic.zenith.x && analytic.zenith.z > 0.0f, "the baked zenith is blue");
    const float peak = analytic.sunIntensity * analytic.sunColor.y;
    check (peak > analytic.zenith.y && peak > analytic.horizon.y && peak > analytic.ground.y,
           "the sun is the brightest direction");
    const double radius = 2.0 * sky::kPi / 180.0;
    check (std::abs (analytic.sunIntensity * sky::kPi * (1.0 - std::cos (radius)) - 30.0) < 0.01 * 30.0,
           "the disk carries the sun's irradiance");
    check (analytic.sunAzimuthDegrees == 0.0f && analytic.sunElevationDegrees == 30.0f, "the sun keeps its direction");
}

void haloTests()
{
    const float clean = sky::sunHalo (sky::Atmosphere { 1.0f, 0.0f, 1.0f, 0.0f });
    const float earth = sky::sunHalo (sky::Atmosphere {});
    const float dusty = sky::sunHalo (sky::Atmosphere { 1.0f, 8.0f, 1.0f, 0.0f });
    const float high = sky::sunHalo (sky::Atmosphere { 1.0f, 1.0f, 1.0f, 20000.0f });
    check (clean == 0.0f && earth > 0.0f && dusty > earth && dusty <= 1.0f, "dust makes the halo");
    check (high < earth, "the halo thins with altitude");

    grid::SceneAnalyticSky analytic;
    analytic.version = 1;
    sky::applyToAnalyticSky (sky::Atmosphere {}, 30.0f, analytic);
    check (analytic.sunHalo == earth && grid::analyticSkyExtrasAuthored (analytic), "the bake writes the halo");
    check (! grid::analyticSkyExtrasAuthored (grid::SceneAnalyticSky {}), "a default sky has no halo, so its bytes stay");

    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Environment;
    light.color = { 1.0f, 1.0f, 1.0f };
    light.intensity = 1.0f;
    light.sky.version = 1;
    light.sky.sunElevationDegrees = 10.0f;
    light.sky.zenith = light.sky.horizon = light.sky.ground = { 0.0f, 0.0f, 0.0f };
    namespace raster = arbitgpu::analyticsky;
    const auto plain = raster::uniforms (light);
    check (plain[0][3] == 1.0f, "no halo leaves the version slot at one");
    light.sky.sunHalo = 0.5f;
    check (grid::validSceneAnalyticSky (light), "a halo in range is valid");
    const auto haloed = raster::uniforms (light);
    check (haloed[0][3] > 1.0f && haloed[0][3] < 1.5f, "the halo packs below the image field's slot value");
    // Toward the sun at the horizon against the opposite horizon.
    const std::array<float, 3> sunSide { 0.0f, 0.05f, 1.0f }, antiSun { 0.0f, 0.05f, -1.0f };
    const auto toward = raster::radiance (sunSide, haloed), away = raster::radiance (antiSun, haloed);
    const auto plainToward = raster::radiance (sunSide, plain), plainAway = raster::radiance (antiSun, plain);
    check (toward[1] > plainToward[1] && toward[1] > 2.0f * away[1], "the halo brightens the sun side of the sky");
    check (std::abs (away[1] - plainAway[1]) < 0.1f * std::max (plainAway[1], 0.01f) + 0.05f,
           "the halo leaves the far side nearly unchanged");
    check (plainToward[1] == plainAway[1], "without a halo the gradient has no sun side");
    light.sky.sunHalo = 2.0f;
    check (! grid::validSceneAnalyticSky (light), "a halo past one is refused");
}

} // namespace

int main()
{
    earthTests();
    controlTests();
    analyticSkyTests();
    haloTests();
    std::cout << checks - failures << " of " << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
