#pragma once

#include "../../src/gltf_glb.h"
#include "../../src/glb_scene_adapter.h"
#include "../../src/gpu_backend/backend.h"
#include "../../../shared/VideoScreenAsset.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace videohelper::tests::orientation
{
inline constexpr int width = 64, height = 36;

inline std::vector<std::uint8_t> pixels()
{
    const std::array<std::array<std::uint8_t, 4>, 4> colors {{
        {{255, 0, 0, 255}}, {{0, 255, 0, 255}}, {{0, 0, 255, 255}}, {{255, 255, 0, 255}} }};
    // White TL/TR/BL/BR text is asymmetric within each differently colored cell.
    const std::array<unsigned, 5> t {7, 2, 2, 2, 2}, l {4, 4, 4, 4, 7};
    const std::array<unsigned, 5> b {6, 5, 6, 5, 6}, r {6, 5, 6, 5, 5};
    std::vector<std::uint8_t> image(width * height * 4);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const int cell = (y >= height / 2 ? 2 : 0) + (x >= width / 2 ? 1 : 0);
            auto color = colors[cell];
            const int localX = x % (width / 2) - 4, localY = y % (height / 2) - 3;
            if (localY >= 0 && localY < 10 && localX >= 0 && localX < 14 && localX % 8 < 6)
            {
                const auto& glyph = localX < 8 ? (cell < 2 ? t : b) : (cell % 2 == 0 ? l : r);
                if ((glyph[localY / 2] >> (2 - (localX % 8) / 2)) & 1u) color = {255, 255, 255, 255};
            }
            std::copy(color.begin(), color.end(), image.begin() + (y * width + x) * 4);
        }
    return image;
}

inline std::shared_ptr<HarmonicMIDI::grid::Visual3DScene> screen(std::string& error)
{
    using namespace HarmonicMIDI::grid;
    gltf::GlbAdmissionOptions options;
    options.supportedRequiredExtensions = {"KHR_lights_punctual"};
    const auto document = gltf::decodeStaticGlb(videoscreen::kGlb.data(), videoscreen::kGlb.size(), options, error);
    const auto adapted = document ? gltf::adaptStaticGlbToVisual3DScene(*document, error) : std::nullopt;
    if (!adapted) return {};
    auto scene = std::make_shared<Visual3DScene>(*adapted);
    // Use the shipped geometry/UVs with a camera that maps one texel to one pixel.
    scene->cameras[0].verticalFovRadians = 2.0f * std::atan(0.45f / 1.5f);
    // Metallic=0 and roughness=1 give diffuseWeight=0.5 with no specular
    // contribution. Unit output requires ambient=2 for the exact RGB oracle.
    scene->ambientColor = {2, 2, 2}; scene->lightCount = 0;
    scene->materials[0].baseColor = {1, 1, 1};
    scene->materials[0].metallic = 0; scene->materials[0].roughness = 1;
    scene->materials[0].emissive = {};
    scene->materials[0].baseColorTexture = SceneTextureId {701};
    scene->textures[0].id = SceneTextureId {701};
    scene->textures[0].width = width; scene->textures[0].height = height;
    scene->textures[0].minFilter = scene->textures[0].magFilter = 9728; // glTF NEAREST
    scene->textureCount = 1; scene->textureTexelCount = width * height;
    scene->textureTexels.resize(scene->textureTexelCount);
    const auto image = pixels();
    for (std::size_t i = 0; i < scene->textureTexelCount; ++i)
        scene->textureTexels[i] = {image[i * 4], image[i * 4 + 1], image[i * 4 + 2], image[i * 4 + 3]};
    return scene;
}

inline std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> frameMaterial(
    const HarmonicMIDI::grid::Visual3DScene& scene, const std::string& backend)
{
    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = backend == "metal" ? arbitgpu::NativeFixtureMaterialBackend::Metal
        : arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object = scene.objects[0].id;
    material->bindingDigest = "four-quadrant-text-frame-binding";
    material->programIdentity = "four-quadrant-text-frame-program";
    material->baseColorSource = arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture;
    material->parameters.baseColorMetallic = {1, 1, 1, 0};
    material->parameters.emissionRoughness = {0, 0, 0, 1};
    material->parameters.normalOpacity = {0, 0, 1, 1};
    material->parameters.transmissionIorClearcoat = {0, 1.5f, 0, 0};
    material->parameters.identifiers[0] = scene.materials[0].id.value;
    return material;
}

inline bool upright(const std::vector<std::uint8_t>& actual, std::string& error, bool lit = false)
{
    const auto expected = pixels();
    if (actual.size() != expected.size()) { error = "Orientation fixture has no complete image"; return false; }
    for (std::size_t i = 0; i < actual.size(); i += 4)
    {
        bool match = actual[i + 3] == 255;
        if (!lit)
        {
            for (std::size_t channel = 0; channel < 3; ++channel)
                match &= std::abs(static_cast<int>(actual[i + channel]) - expected[i + channel]) <= 3;
        }
        else
        {
            // Lighting may alter brightness, but cannot exchange quadrants or
            // mirror the text. Test every pixel's color class independently.
            const bool white = expected[i] == 255 && expected[i + 1] == 255 && expected[i + 2] == 255;
            if (white)
                match &= *std::min_element(actual.begin() + i, actual.begin() + i + 3) > 10
                    && *std::max_element(actual.begin() + i, actual.begin() + i + 3)
                        - *std::min_element(actual.begin() + i, actual.begin() + i + 3) <= 4;
            else
                for (std::size_t high = 0; high < 3; ++high)
                    for (std::size_t low = 0; low < 3; ++low)
                        if (expected[i + high] == 255 && expected[i + low] == 0)
                            match &= actual[i + high] > actual[i + low] + 10;
        }
        if (!match)
        {
            error = "Quadrant/text orientation differs at " + std::to_string((i / 4) % width)
                + "," + std::to_string((i / 4) / width);
            error += ": expected " + std::to_string(expected[i]) + ","
                + std::to_string(expected[i + 1]) + "," + std::to_string(expected[i + 2])
                + "; actual " + std::to_string(actual[i]) + ","
                + std::to_string(actual[i + 1]) + "," + std::to_string(actual[i + 2])
                + "," + std::to_string(actual[i + 3]);
            return false;
        }
    }
    return true;
}

inline bool materialOrientationCases(arbitgpu::NativeFixtureSceneBackend& backend, std::string& error)
{
    const auto scene = screen(error);
    if (!scene) return false;
    const auto resources = backend.prepare(scene, nullptr);
    if (!resources.prepared) { error = resources.error; return false; }
    const auto source = backend.render(scene, resources.resources, width, height, {});
    if (!source.rendered) { error = source.error; return false; }
    const auto order = backend.info().backend == "opengl" ? arbitgpu::NativeTextureRowOrder::BottomFirst
        : arbitgpu::NativeTextureRowOrder::TopFirst;
    if (source.frame->colorTextureDescriptor().rowOrder != order
        || source.frame->depthTextureDescriptor().rowOrder != order)
    { error = "Native scene descriptors lost the attachment row convention"; return false; }
    arbitgpu::NativeRawPassPixels embedded;
    if (!source.frame->readRawPass(renderpassoutput::Output::Color, embedded, error)
        || !upright(embedded.bytes, error)) return false;
    const auto texturedResources = backend.prepare(scene, frameMaterial(*scene, backend.info().backend));
    if (!texturedResources.prepared) { error = texturedResources.error; return false; }
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.materialFrameTexture = source.frame;
    const auto preview = backend.render(scene, texturedResources.resources, width, height, inputs);
    const auto exported = backend.render(scene, texturedResources.resources, width, height, inputs);
    if (!preview.rendered || !exported.rendered) { error = preview.error + exported.error; return false; }
    arbitgpu::NativeRawPassPixels a, b, retained;
    if (!preview.frame->readRawPass(renderpassoutput::Output::Color, a, error)
        || !exported.frame->readRawPass(renderpassoutput::Output::Color, b, error)
        || !source.frame->readRawPass(renderpassoutput::Output::Color, retained, error)
        || !upright(a.bytes, error) || !upright(b.bytes, error)) return false;
    if (a.bytes != b.bytes || embedded.bytes != retained.bytes)
    { error = "Frame rendering changed its input or preview/export orientation"; return false; }
    return true;
}
} // namespace videohelper::tests::orientation
