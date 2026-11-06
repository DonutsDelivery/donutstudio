// Tests for Raster's and Realtime's volumetric world fog (shared/RasterMedium.h):
// the fog rows a scene's or the Render node's fog lowers to, the frame scope the
// backends read, and the march's Beer-Lambert transmittance and shadowed single
// scattering, which draws light shafts where the scene blocks a light.

#include "../../shared/RasterMedium.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

namespace
{

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

bool close (float a, float b, float tolerance = 1.0e-5f) { return std::fabs (a - b) <= tolerance * std::max (1.0f, std::fabs (b)); }

grid::SceneVolume fogVolume()
{
    grid::SceneVolume volume;
    volume.density = 0.5f;
    volume.color = { 0.8f, 0.8f, 0.8f };
    return volume;
}

void loweringTests()
{
    check (! rastermedium::worldFogOn (rastermedium::packWorldFog (grid::SceneVolume {}, 100.0f)), "no fog packs off");
    auto volume = fogVolume();
    volume.field = grid::SceneVolumeField::Noise;
    volume.fieldDetail = 3;
    const auto fog = rastermedium::packWorldFog (volume, 250.0f);
    const auto absorption = rastermedium::absorption (volume), scattering = rastermedium::scattering (volume);
    check (rastermedium::worldFogOn (fog) && close (fog[0][0], absorption[0]) && close (fog[1][1], scattering[1]),
           "the fog carries Path Trace's coefficients and is on");
    check (fog[2][3] == 1.0f && fog[4][3] == 3.0f, "the noise field and its detail lower like a medium's");
    check (fog[4][0] == 250.0f && fog[4][1] == 0.0f && fog[5] == std::array<float, 4> {},
           "row 4 carries the sky's far distance and no bounds follow");
    check (rastermedium::packWorldFog (volume, std::nanf ("")) [4][0] == 0.0f, "a nonfinite far distance marches nothing");

    auto scene = std::make_unique<grid::Visual3DScene>();
    scene->cameras[0].id = { 4 };
    scene->cameras[0].farPlane = 80.0f;
    scene->cameraCount = 1;
    scene->activeCamera = { 4 };
    check (! rastermedium::worldFogOn (rastermedium::worldFogForScene (*scene, nullptr)), "a scene without fog marches none");
    scene->worldVolume = fogVolume();
    const auto sceneFog = rastermedium::worldFogForScene (*scene, nullptr);
    check (rastermedium::worldFogOn (sceneFog) && sceneFog[4][0] == 80.0f, "the scene's fog marches to the camera's far plane");
    auto renderFog = fogVolume();
    renderFog.density = 2.0f;
    const auto overridden = rastermedium::worldFogForScene (*scene, &renderFog);
    check (close (overridden[1][0], rastermedium::scattering (renderFog)[0]), "the Render node's fog wins over the scene's");
    scene->activeCamera = { 5 };
    check (! rastermedium::worldFogOn (rastermedium::worldFogForScene (*scene, nullptr)), "no active camera marches none");
}

void scopeTests()
{
    check (! rastermedium::worldFogOn (rastermedium::currentWorldFog()), "no frame has no fog");
    {
        const rastermedium::WorldFogScope scope (rastermedium::packWorldFog (fogVolume(), 10.0f));
        check (rastermedium::worldFogOn (rastermedium::currentWorldFog()), "a frame's scope sets its fog");
        {
            const rastermedium::WorldFogScope inner (rastermedium::Uniforms {});
            check (! rastermedium::worldFogOn (rastermedium::currentWorldFog()), "a nested frame without fog clears it");
        }
        check (rastermedium::currentWorldFog()[4][0] == 10.0f, "the outer frame's fog comes back");
    }
    check (! rastermedium::worldFogOn (rastermedium::currentWorldFog()), "the scope restores no fog");
}

void marchTests()
{
    // Homogeneous fog: transmittance is exp(-sigma_t L), and with ambient light
    // only the in-scattered radiance is sigma_s A (1 - exp(-sigma_t L)) / sigma_t.
    const auto volume = fogVolume();
    const auto fog = rastermedium::packWorldFog (volume, 100.0f);
    const float sigmaS = rastermedium::scattering (volume)[0];
    const float sigmaT = rastermedium::absorption (volume)[0] + sigmaS;
    const float o[3] = { 0.0f, 0.0f, 0.0f }, d[3] = { 0.0f, 0.0f, -1.0f };
    const float length = 6.0f;
    const auto none = [] (const float*, int& count) { count = 0; return std::array<rastermedium::Light, 8> {}; };
    const auto ambient = rastermedium::march (fog, o, d, 0.0f, length, { 1.0f, 1.0f, 1.0f }, none);
    check (close (ambient.transmittance[0], std::exp (-sigmaT * length)), "the fog's transmittance is Beer-Lambert");
    check (close (ambient.radiance[0], sigmaS * (1.0f - std::exp (-sigmaT * length)) / sigmaT),
           "ambient light scatters in over the path");

    // A light shaft: a light the scene blocks past the path's midpoint lights
    // only the first half, which is exactly the march of that half alone.
    rastermedium::Light light;
    light.toLight = { 1.0f, 0.0f, 0.0f };
    light.radiance = { 3.0f, 3.0f, 3.0f };
    const auto lit = [&] (const float*, int& count) {
        count = 1;
        return std::array<rastermedium::Light, 8> { light };
    };
    const auto shadowed = [&] (const float* p, int& count) {
        count = p[2] > -0.5f * length ? 1 : 0;
        return std::array<rastermedium::Light, 8> { light };
    };
    const auto full = rastermedium::march (fog, o, d, 0.0f, length, {}, lit);
    const auto shaft = rastermedium::march (fog, o, d, 0.0f, length, {}, shadowed);
    const auto half = rastermedium::march (fog, o, d, 0.0f, 0.5f * length, {}, lit);
    check (shaft.radiance[0] > 0.0f && shaft.radiance[0] < full.radiance[0], "a shadowed span scatters less light");
    check (close (shaft.radiance[0], half.radiance[0], 1.0e-4f), "only the lit span scatters the light");
    check (close (shaft.transmittance[0], full.transmittance[0]), "shadow changes the light, not the fog's transmittance");

    // The programs carry the march, and only the fixture program turns it on.
    const auto gl = rastermedium::glslShader(), metal = rastermedium::metalShader();
    check (gl.find ("uniform vec4 uWorldFog[5];") != std::string::npos && gl.find ("vec4 worldFogApply(vec4 color)") != std::string::npos,
           "the GL program declares the fog and applies it");
    check (metal.find ("FragmentOut worldFogged(FragmentOut value") != std::string::npos
               && metal.find ("mediumSourceOf(u, opaqueTexture, true, p, d, f)") != std::string::npos,
           "the Metal program fogs with the shadowed light source");
    check (gl.find ("uSkyBackground > 1.5") != std::string::npos && metal.find ("u.skyBackground.x > 1.5f") != std::string::npos,
           "both programs give the backdrop pass the fog's opacity");
}

// A frame with no sky draw (no sky, or a camera-hidden environment) fogs its
// cleared backdrop over the far distance, as the sky would be.
void backdropTests()
{
    using Mode = grid::SceneCameraBackgroundMode;
    grid::SceneCameraBackground background;
    const auto stage = rastermedium::backdropRow (false, background);
    check (close (stage[0], 7.0f / 255.0f) && close (stage[2], 18.0f / 255.0f) && stage[3] == 1.0f, "no sky clears to the stage colour");
    check (rastermedium::backdropRow (true, background) == std::array<float, 4> { 0.0f, 0.0f, 0.0f, 1.0f },
           "a hidden environment's World backdrop is opaque black");
    background.mode = Mode::Colour;
    background.color = { 0.25f, -1.0f, std::nanf ("") };
    check (rastermedium::backdropRow (true, background) == std::array<float, 4> { 0.25f, 0.0f, 0.0f, 1.0f },
           "a Colour backdrop is its linear colour, negative and nonfinite channels dark");
    background.mode = Mode::Transparent;
    const auto clear = rastermedium::backdropRow (true, background);
    check (clear == std::array<float, 4> {}, "a Transparent backdrop is zero alpha");

    const float o[3] = { 0.0f, 1.0f, 0.0f }, d[3] = { 0.0f, 0.0f, -1.0f };
    const auto none = [] (const float*, int& count) { count = 0; return std::array<rastermedium::Light, 8> {}; };
    check (rastermedium::fogBackdrop (rastermedium::Uniforms {}, stage, o, d, { 1.0f, 1.0f, 1.0f }, none) == stage,
           "without fog the backdrop is unchanged");

    const auto volume = fogVolume();
    const float farDistance = 4.0f;
    const auto fog = rastermedium::packWorldFog (volume, farDistance);
    const float sigmaS = rastermedium::scattering (volume)[0];
    const float sigmaT = rastermedium::absorption (volume)[0] + sigmaS;
    const float keep = std::exp (-sigmaT * farDistance);
    const float inscattered = sigmaS * (1.0f - keep) / sigmaT;
    const auto fogged = rastermedium::fogBackdrop (fog, stage, o, d, { 1.0f, 1.0f, 1.0f }, none);
    check (close (fogged[0], stage[0] * keep + inscattered, 1.0e-4f) && fogged[3] == 1.0f,
           "the stage backdrop fogs over the far distance and stays opaque");
    const auto covered = rastermedium::fogBackdrop (fog, clear, o, d, { 1.0f, 1.0f, 1.0f }, none);
    check (close (covered[0], inscattered, 1.0e-4f) && close (covered[3], 1.0f - keep, 1.0e-4f),
           "a transparent backdrop takes the fog's light and opacity");

    // A light shaft reaches the backdrop too: a light lit only over the near
    // half of the far distance adds exactly that half's scattering.
    rastermedium::Light light;
    light.toLight = { 0.0f, 1.0f, 0.0f };
    light.radiance = { 2.0f, 2.0f, 2.0f };
    const auto shadowed = [&] (const float* p, int& count) {
        count = p[2] > -0.5f * farDistance ? 1 : 0;
        return std::array<rastermedium::Light, 8> { light };
    };
    const auto lit = [&] (const float*, int& count) { count = 1; return std::array<rastermedium::Light, 8> { light }; };
    const auto shaft = rastermedium::fogBackdrop (fog, clear, o, d, {}, shadowed);
    const auto half = rastermedium::march (fog, o, d, 0.0f, 0.5f * farDistance, {}, lit);
    check (close (shaft[0], half.radiance[0], 1.0e-4f), "the backdrop shows the shadowed light shaft");
}

} // namespace

int main()
{
    loweringTests();
    scopeTests();
    marchTests();
    backdropTests();
    std::cout << checks - failures << " of " << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
