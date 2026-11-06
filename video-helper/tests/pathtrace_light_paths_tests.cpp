// Tests for the Render node's Path Trace light paths: the Filter Glossy
// roughness, the caustic lobe filter, the settings mapping, the V36 wire, and
// small CPU renders for the clamps, a kind limit and the caustic switches.
// Renders run at 4 samples per pixel.

#include "pathtrace/accumulator.h"
#include "pathtrace/bsdf.h"
#include "pathtrace/integrator.h"
#include "pathtrace/light_paths.h"
#include "pathtrace/scene_ingest.h"
#include "pathtrace/thread_pool.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
namespace render = visualimportedscenerender;
using pt::Vec3;

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

std::string describe (double value) { return std::to_string (value); }

// ---------------------------------------------------------------- scenes

void appendQuad (pt::RenderScene& scene, Vec3 centre, Vec3 halfU, Vec3 halfV, std::uint32_t material)
{
    const Vec3 corners[4] { centre - halfU - halfV, centre + halfU - halfV, centre + halfU + halfV, centre - halfU + halfV };
    const std::uint32_t indices[6] { 0, 1, 2, 0, 2, 3 };
    for (int i = 0; i < 6; i += 3)
    {
        const Vec3 p0 = corners[indices[i]], p1 = corners[indices[i + 1]], p2 = corners[indices[i + 2]];
        const Vec3 n = pt::normalizeOrZero (pt::cross (p1 - p0, p2 - p0));
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (scene.geometry.triangles.size());
        attributes.geometricNormal = n;
        attributes.shadingNormals[0] = n;
        attributes.shadingNormals[1] = n;
        attributes.shadingNormals[2] = n;
        for (const Vec3 p : { p0, p1, p2 })
        {
            scene.geometry.positions.push_back (p);
            scene.geometry.bounds.grow (p);
        }
        scene.geometry.triangles.push_back (attributes);
    }
}

// A closed box whose faces point away from its centre.
void appendBox (pt::RenderScene& scene, Vec3 centre, float half, std::uint32_t material)
{
    const float h = half;
    appendQuad (scene, centre + Vec3 { 0.0f, 0.0f, h }, { h, 0.0f, 0.0f }, { 0.0f, h, 0.0f }, material);
    appendQuad (scene, centre + Vec3 { 0.0f, 0.0f, -h }, { 0.0f, h, 0.0f }, { h, 0.0f, 0.0f }, material);
    appendQuad (scene, centre + Vec3 { h, 0.0f, 0.0f }, { 0.0f, h, 0.0f }, { 0.0f, 0.0f, h }, material);
    appendQuad (scene, centre + Vec3 { -h, 0.0f, 0.0f }, { 0.0f, 0.0f, h }, { 0.0f, h, 0.0f }, material);
    appendQuad (scene, centre + Vec3 { 0.0f, h, 0.0f }, { 0.0f, 0.0f, h }, { h, 0.0f, 0.0f }, material);
    appendQuad (scene, centre + Vec3 { 0.0f, -h, 0.0f }, { h, 0.0f, 0.0f }, { 0.0f, 0.0f, h }, material);
}

grid::SceneLightRecord pointRecord (Vec3 position, float intensity)
{
    grid::SceneLightRecord light;
    light.kind = grid::SceneLightKind::Point;
    light.transform.translation = { position.x, position.y, position.z };
    light.intensity = intensity;
    return light;
}

pt::PinholeCamera makeCamera (Vec3 position, std::uint32_t size, float tanHalfFov)
{
    pt::PinholeCamera camera;
    camera.position = position;  // looks down -Z
    camera.tanHalfFov = tanHalfFov;
    camera.aspect = 1.0f;
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.width = size;
    camera.height = size;
    return camera;
}

struct Image
{
    std::vector<Vec3> pixels;
    pt::RenderStats stats {};

    double mean() const
    {
        double sum = 0.0;
        for (const Vec3 p : pixels)
            sum += (p.x + p.y + p.z) / 3.0;
        return sum / static_cast<double> (std::max<std::size_t> (1, pixels.size()));
    }
    double peak() const
    {
        double most = 0.0;
        for (const Vec3 p : pixels)
            most = std::max ({ most, double (p.x), double (p.y), double (p.z) });
        return most;
    }
};

Image renderImage (const pt::RenderScene& scene, const pt::PinholeCamera& camera, const pt::IntegratorSettings& settings)
{
    pt::PathTracer tracer (4);
    tracer.settings = settings;
    const pt::CancellationToken cancel {};
    Image image;
    image.stats = tracer.renderFixed (scene, camera, camera.width, camera.height, 4, 1, cancel);
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x)
        {
            const pt::Rgb v = tracer.accumulator().mean (x, y);
            image.pixels.push_back ({ v.r, v.g, v.b });
        }
    return image;
}

// ---------------------------------------------------------------- tests

void filterGlossyTests()
{
    const float inf = std::numeric_limits<float>::infinity();
    check (pt::filteredGlossRoughness (0.1f, 0.0f, 0.5f) == 0.1f, "Filter Glossy 0 leaves the roughness");
    check (pt::filteredGlossRoughness (0.1f, 1.0f, inf) == 0.1f, "before a rough bounce nothing blurs");
    check (pt::filteredGlossRoughness (0.1f, 1.0f, 2.0f) == 0.1f, "a dense bounce (blur pdf 1 or more) does not blur");
    const float expected = std::sqrt (std::sqrt (1.0f - 0.25f) * 0.5f);
    check (std::fabs (pt::filteredGlossRoughness (0.0f, 2.0f, 0.5f) - expected) < 1.0e-6f,
           "blur alpha is sqrt(1 - minPdf / filter) / 2: " + describe (pt::filteredGlossRoughness (0.0f, 2.0f, 0.5f)));
    check (pt::filteredGlossRoughness (0.9f, 2.0f, 0.5f) == 0.9f, "a rougher lobe keeps its roughness");
    check (pt::filteredGlossRoughness (0.0f, 10.0f, 0.01f) > pt::filteredGlossRoughness (0.0f, 1.0f, 0.01f),
           "a larger filter blurs more");
}

void causticLobeTests()
{
    const Vec3 n { 0.0f, 0.0f, 1.0f };
    const Vec3 wo = pt::normalizeOrZero (Vec3 { 0.3f, 0.0f, 1.0f });
    pt::Material plastic;
    plastic.roughness = 0.3f;
    pt::Bsdf bsdf;
    pt::buildMaterialBsdf (plastic, n, n, wo, true, bsdf);
    const int plasticLobes = bsdf.lobeCount();
    bsdf.dropCausticLobes (true, false);
    bool onlyDiffuse = bsdf.lobeCount() >= 1;
    float total = 0.0f;
    for (int i = 0; i < bsdf.lobeCount(); ++i)
    {
        onlyDiffuse = onlyDiffuse && bsdf.lobe (i).kind == pt::LobeKind::Lambert;
        total += bsdf.selectionProbability (i);
    }
    check (plasticLobes >= 2 && onlyDiffuse && std::fabs (total - 1.0f) < 1.0e-6f,
           "Reflective off keeps the diffuse lobe and re-weights it");

    pt::Material glass;
    glass.transmission = 1.0f;
    glass.roughness = 0.0f;
    glass.ior = 1.5f;
    pt::buildMaterialBsdf (glass, n, n, wo, true, bsdf);
    const int glassLobes = bsdf.lobeCount();
    bsdf.dropCausticLobes (true, false);
    check (glassLobes > 0 && bsdf.lobeCount() == glassLobes, "Reflective off keeps a dielectric");
    bsdf.dropCausticLobes (false, true);
    check (bsdf.lobeCount() == 0, "Refractive off drops a dielectric");
}

void settingsTests()
{
    render::PathTraceLightPaths paths;
    pt::IntegratorSettings settings;
    pt::applyLightPaths (paths, settings);
    const pt::IntegratorSettings defaults;
    check (pt::sameLightPathSettings (settings, defaults), "the default light paths are the integrator's defaults");
    paths.diffuseBounces = 2;
    paths.glossyBounces = 0;
    paths.transmissionBounces = 7;
    paths.transparentBounces = 3;
    paths.directClamp = 5.0f;
    paths.indirectClamp = 2.5f;
    paths.filterGlossy = 1.0f;
    paths.reflectiveCaustics = false;
    pt::applyLightPaths (paths, settings);
    check (settings.maxDiffuseBounces == 2 && settings.maxGlossyBounces == 0 && settings.maxTransmissionBounces == 7
               && settings.maxTransparentHits == 3 && settings.directClamp == 5.0f && settings.indirectClamp == 2.5f
               && settings.filterGlossy == 1.0f && ! settings.reflectiveCaustics && settings.refractiveCaustics,
           "every light path reaches the integrator");
    check (! pt::sameLightPathSettings (settings, defaults), "a changed light path restarts the preview");
    paths.directClamp = std::numeric_limits<float>::quiet_NaN();
    pt::applyLightPaths (paths, settings);
    check (settings.directClamp == 0.0f, "an amount that is not a number reads as off");
}

void wireTests()
{
    render::Request traced;
    traced.sourceStableId = 71;
    traced.renderStableId = 72;
    traced.asset.id = "model-asset-1";
    traced.asset.version = 3;
    traced.asset.contentSha256 = std::string (64, 'a');
    traced.asset.sourceMediaType = "model/gltf-binary";
    traced.asset.sourceByteSize = 4096;
    traced.sceneIndex = 0;
    traced.renderEngine = render::RenderEngine::PathTrace;
    const auto tracedWire = render::encode (traced);
    check (! tracedWire.empty() && tracedWire.find (render::kCompiledOperationHeaderV36) == std::string::npos,
           "the default light paths keep the old bytes");

    auto paths = traced;
    paths.pathTraceLightPaths.diffuseBounces = 4;
    paths.pathTraceLightPaths.transparentBounces = 32;
    paths.pathTraceLightPaths.indirectClamp = 10.0f;
    paths.pathTraceLightPaths.filterGlossy = 0.5f;
    paths.pathTraceLightPaths.refractiveCaustics = false;
    const auto pathsWire = render::encode (paths);
    check (pathsWire.rfind (std::string (render::kCompiledOperationHeaderV36) + "\n4 65535 65535 32 0 "
                                + std::to_string (render::detail::floatBits (10.0f)) + " "
                                + std::to_string (render::detail::floatBits (0.5f)) + " 1 "
                                + std::to_string (tracedWire.size()) + "\n", 0) == 0
               && pathsWire.size() > tracedWire.size()
               && pathsWire.compare (pathsWire.size() - tracedWire.size(), tracedWire.size(), tracedWire) == 0,
           "V36 carries the light paths and the exact default bytes");
    render::Request decoded;
    check (render::decode (pathsWire, decoded)
               && render::sameLightPaths (decoded.pathTraceLightPaths, paths.pathTraceLightPaths)
               && render::sameRequest (paths, decoded) && ! render::sameRequest (traced, decoded),
           "light paths round-trip and are part of request identity");

    auto nested = paths;
    nested.pathTraceMaxBounces = 9;
    nested.pathTraceVolume = render::pathTraceVolume (2, false);
    check (render::decode (render::encode (nested), decoded) && decoded.pathTraceMaxBounces == 9
               && decoded.pathTraceVolume == nested.pathTraceVolume
               && render::sameLightPaths (decoded.pathTraceLightPaths, paths.pathTraceLightPaths),
           "V36 wraps the media and bounce limit envelopes");

    const auto header = std::string (render::kCompiledOperationHeaderV36) + "\n";
    check (! render::decode (header + "65535 65535 65535 16 0 0 0 3 " + std::to_string (tracedWire.size()) + "\n"
                                 + tracedWire, decoded),
           "the default written out is not canonical");
    check (! render::decode (header + "1 65535 65535 16 0 0 0 4 " + std::to_string (tracedWire.size()) + "\n"
                                 + tracedWire, decoded),
           "an unknown caustic flag is refused");
    check (! render::decode (header + "1 65535 65535 16 "
                                 + std::to_string (render::detail::floatBits (-1.0f)) + " 0 0 3 "
                                 + std::to_string (tracedWire.size()) + "\n" + tracedWire, decoded),
           "a negative clamp is refused");

    auto raster = paths;
    raster.renderEngine = render::RenderEngine::Raster;
    check (render::encode (raster).empty(), "Raster may not set light paths");
}

// A diffuse floor below a point light, seen from above.
pt::RenderScene litFloor()
{
    pt::RenderScene scene;
    scene.materials = { pt::Material {} };
    appendQuad (scene, { 0.0f, 0.0f, -3.0f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 3.0f, 0.0f }, 0);
    scene.lightRecords = { pointRecord ({ 0.0f, 0.0f, 0.0f }, 50.0f) };
    scene.commit();
    return scene;
}

void clampTests()
{
    const pt::RenderScene scene = litFloor();
    const auto camera = makeCamera ({ 0.0f, 0.0f, 5.0f }, 4, 0.3f);
    pt::IntegratorSettings settings;
    settings.maxDepth = 2;
    const Image open = renderImage (scene, camera, settings);
    settings.directClamp = 1.0e-3f;
    const Image clamped = renderImage (scene, camera, settings);
    check (open.peak() > 1.0e-2, "the lit floor is bright: " + describe (open.peak()));
    check (clamped.peak() <= 1.0e-3 * 1.0001 && clamped.stats.clampedContributions > 0,
           "Clamp Direct limits light after one scatter: " + describe (clamped.peak()));
    settings.directClamp = 0.0f;
    settings.indirectClamp = 1.0e-3f;
    const Image indirect = renderImage (scene, camera, settings);
    check (std::fabs (indirect.mean() - open.mean()) < 1.0e-6 * std::max (1.0, open.mean()),
           "Clamp Indirect leaves direct light alone");
}

void diffuseLimitTests()
{
    // A closed room lit from inside: its light after the first scatter is indirect.
    pt::RenderScene room;
    room.materials = { pt::Material {} };
    appendBox (room, { 0.0f, 0.0f, 0.0f }, 2.0f, 0);
    // The room's faces point outward; the camera and light are inside, so
    // every wall is seen and lit from its back face, which a two-sided
    // diffuse material shades alike.
    room.lightRecords = { pointRecord ({ 0.0f, 0.0f, 1.0f }, 20.0f) };
    room.commit();
    const auto camera = makeCamera ({ 0.0f, 0.0f, 1.5f }, 4, 0.5f);
    pt::IntegratorSettings settings;
    settings.maxDepth = 4;
    const Image full = renderImage (room, camera, settings);
    settings.maxDiffuseBounces = 0;
    const Image direct = renderImage (room, camera, settings);
    check (direct.mean() > 0.0 && direct.mean() < full.mean() && direct.stats.bounceLimitPaths > 0,
           "Diffuse Bounces 0 keeps direct light and ends the bounce: " + describe (direct.mean()) + " of "
               + describe (full.mean()));
}

void causticSwitchTests()
{
    // A point light inside a closed glass box above a diffuse floor. Shadow rays
    // stop at the glass, so the floor is lit only by refraction caustics.
    pt::RenderScene scene;
    pt::Material glass;
    glass.transmission = 1.0f;
    glass.roughness = 0.0f;
    glass.ior = 1.5f;
    scene.materials = { pt::Material {}, glass };
    appendQuad (scene, { 0.0f, 0.0f, -3.0f }, { 3.0f, 0.0f, 0.0f }, { 0.0f, 3.0f, 0.0f }, 0);
    appendBox (scene, { 0.0f, 0.0f, 0.0f }, 0.3f, 1);
    scene.lightRecords = { pointRecord ({ 0.0f, 0.0f, 0.0f }, 50.0f) };
    scene.commit();
    const auto camera = makeCamera ({ 0.0f, 0.0f, 5.0f }, 8, 0.3f);
    pt::IntegratorSettings settings;
    settings.maxDepth = 6;
    const Image on = renderImage (scene, camera, settings);
    settings.reflectiveCaustics = false;
    const Image reflectiveOff = renderImage (scene, camera, settings);
    settings.reflectiveCaustics = true;
    settings.refractiveCaustics = false;
    const Image refractiveOff = renderImage (scene, camera, settings);
    check (on.mean() > 0.0 && on.stats.causticSplats > 0, "refraction caustics light the floor: " + describe (on.mean()));
    check (reflectiveOff.mean() > 0.0, "Reflective off keeps refraction caustics");
    check (refractiveOff.mean() == 0.0 && refractiveOff.stats.causticSplats == 0,
           "Refractive off leaves the floor dark: " + describe (refractiveOff.mean()));
}

} // namespace

int main()
{
    filterGlossyTests();
    causticLobeTests();
    settingsTests();
    wireTests();
    clampTests();
    diffuseLimitTests();
    causticSwitchTests();
    std::cout << checks - failures << " of " << checks << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
