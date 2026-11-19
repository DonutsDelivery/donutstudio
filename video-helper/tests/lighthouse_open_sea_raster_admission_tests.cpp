// Lighthouse and Ocean in Raster: the open sea's water authors a base colour
// below white (BuiltInVisualModules.h), which is the light its medium leaves
// after one scene unit. Raster admission refused it as "Closed glass transport
// requires one dielectric with unit opacity and transmission" while Path Trace
// and Realtime drew it. These cases read that colour from the starter source,
// run water and lantern glass through the camera-only admission the renderer
// uses, and check the absorption the draws hand the medium hook. Caustics keep
// a white glass. CPU only, no GPU.

#include "support/glass_caustic_oracle.h"
#include "../src/gpu_backend/fixture_glass_caustics.h"
#include "../src/gpu_backend/fixture_smooth_glass.h"
#include "../../shared/RasterMedium.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace
{
namespace gt = glasscaustictests;
namespace sg = arbitgpu::smoothglass;
using namespace HarmonicMIDI::grid;

int failures = 0, checks = 0;
void check(bool condition, const std::string& text)
{
    ++checks;
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", text.c_str()); }
}

// The open variant's water base colour, as the starter authors it.
bool starterWaterColour(std::array<float, 3>& colour)
{
    std::ifstream file(ARBIT_PLUGIN_SOURCE_DIR "/graph/BuiltInVisualModules.h");
    std::stringstream text;
    text << file.rdbuf();
    const std::string source = text.str();
    const std::string marker = "glassDome ? std::array<double, 3> {1.0, 1.0, 1.0} : std::array<double, 3> {";
    const auto at = source.find(marker);
    if (at == std::string::npos) return false;
    return std::sscanf(source.c_str() + at + marker.size(), "%f, %f, %f", &colour[0], &colour[1], &colour[2]) == 3;
}

// The helper's glass slab over a receiver, with a thin water plane between
// them whose base colour is the given one.
gt::Fixture openSea(const std::array<float, 3>& waterColour, SceneObjectId& water)
{
    auto f = gt::ordinaryFixture();
    const auto first = f.scene->vertexCount;
    const auto o = f.scene->objectCount;
    videohelper::tests::smoothglassfixture::receiver(*f.scene, -6, {0, 0, 0});
    for (std::size_t v = first; v < f.scene->vertexCount; ++v) f.scene->vertices[v].position.z = -1.5f;
    water = f.scene->objects[o].id;
    auto p = std::make_shared<gt::Program>();
    p->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    p->object = water;
    p->bindingDigest = "open-sea-binding-water";
    p->programIdentity = "open-sea-program-water";
    p->authoredScene = f.scene->id;
    p->authoredMaterial = f.scene->objects[o].material;
    p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
    p->parameters.identifiers[0] = p->authoredMaterial.value;
    p->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ThinInterface);
    p->parameters.normalOpacity = {0, 0, 1, 1};
    p->parameters.baseColorMetallic = {waterColour[0], waterColour[1], waterColour[2], 0};
    p->parameters.transmissionIorClearcoat = {1, 1.333f, 0, 0};
    p->parameters.emissionRoughness = {0, 0, 0, 0};
    f.program->objectPrograms.push_back(p);
    return f;
}

bool near(float a, float b) { return std::abs(a - b) <= 1.0e-5f * std::max(1.0f, std::abs(b)); }

void tintedWaterAdmitted()
{
    std::array<float, 3> colour {};
    check(starterWaterColour(colour), "the starter source authors the open sea's water colour");
    check(colour[0] < 1 && colour[1] < 1 && colour[2] < 1, "the open sea's water is tinted");
    SceneObjectId water;
    auto f = openSea(colour, water);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    const auto glass = f.scene->objects[0].id;
    std::string error;

    // The renderer's Raster path: Glass Caustics off, camera-only selection.
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    check(derived && derived->cameraOnly && derived->object == glass, "the tinted water and glass derive a camera-only selection");
    sg::Prepared prepared;
    const bool admitted = derived && sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &*derived);
    check(admitted, "the tinted open sea is admitted in Raster: " + error);
    if (!admitted) return;
    check(prepared.object == water && prepared.transport == surfacematerial::TransportMode::ThinInterface
              && prepared.causticVolume.present && prepared.causticVolume.object == glass,
          "the water stays the camera dielectric with the glass in the second slot");
    for (std::size_t c = 0; c < 3; ++c)
        check(near(prepared.absorption[c], -std::log(colour[c])), "the water absorbs -ln(base colour) per scene unit, as Path Trace");
    check(prepared.causticVolume.absorption == std::array<float, 3> {}, "the white lantern glass absorbs nothing");

    // The draws hand the absorption to the medium hook: the water draw's own row,
    // and the glass draw's partner row for its walk through the water.
    rastermedium::Uniforms waterDraw {};
    rastermedium::addDielectricTint(waterDraw, sg::dielectricAbsorption(prepared, water), sg::dielectricAbsorption(prepared, glass));
    check(waterDraw[0][3] == float(int(rastermedium::Mode::Tint)), "the water draw's medium tint is on");
    for (std::size_t c = 0; c < 3; ++c)
        check(near(waterDraw[0][c], prepared.absorption[c]) && waterDraw[6][c] == 0, "the water draw carries its own absorption");
    rastermedium::Uniforms glassDraw {};
    rastermedium::addDielectricTint(glassDraw, sg::dielectricAbsorption(prepared, glass), sg::dielectricAbsorption(prepared, water));
    check(glassDraw[0][3] == 0, "the white glass draw keeps its medium off");
    for (std::size_t c = 0; c < 3; ++c)
        check(near(glassDraw[6][c], prepared.absorption[c]), "the glass draw tints its water segments by the water's absorption");
    // A receiver draw takes nothing.
    rastermedium::Uniforms receiverDraw {};
    rastermedium::addDielectricTint(receiverDraw, sg::dielectricAbsorption(prepared, f.scene->objects[1].id), {});
    check(receiverDraw == rastermedium::Uniforms {}, "a receiver draw takes no dielectric absorption");
}

// White water absorbs nothing, so the clear-water starters draw as before.
void whiteWaterUnchanged()
{
    SceneObjectId water;
    auto f = openSea({1, 1, 1}, water);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    std::string error;
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    sg::Prepared prepared;
    check(derived && sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &*derived), "white water is admitted: " + error);
    check(prepared.absorption == std::array<float, 3> {}, "white water absorbs nothing");
    rastermedium::Uniforms draw {};
    rastermedium::addDielectricTint(draw, sg::dielectricAbsorption(prepared, water), sg::dielectricAbsorption(prepared, f.scene->objects[0].id));
    check(draw == rastermedium::Uniforms {}, "white dielectrics leave the medium rows as they were");
}

void baseColourOutOfRangeRefused()
{
    SceneObjectId water;
    auto f = openSea({1.2f, 0.5f, 0.5f}, water);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    std::string error;
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    sg::Prepared prepared;
    check(derived && !sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &*derived)
              && error == "Closed glass transport requires one dielectric with unit opacity and transmission",
          "a base colour above one is still refused: " + error);
}

// The caustic photon walk has no medium, so a tinted caustic glass is refused
// there rather than depositing untinted light.
void tintedCausticGlassRefused()
{
    auto f = gt::ordinaryFixture();
    f.program->parameters.baseColorMetallic = {0.9f, 0.8f, 0.7f, 0};
    std::string error;
    check(!f.prepare(error) && error == "Glass caustics require a white glass base colour",
          "a tinted caustic glass is refused by the caustic preparation: " + error);
    sg::Prepared camera;
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    check(sg::prepare(scene, root, {}, camera, error) && near(camera.absorption[0], -std::log(0.9f)),
          "the same tinted glass is admitted for the camera walk: " + error);
}
} // namespace

int main()
{
    tintedWaterAdmitted();
    whiteWaterUnchanged();
    baseColourOutOfRangeRefused();
    tintedCausticGlassRefused();
    std::printf("lighthouse open sea raster admission: %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
