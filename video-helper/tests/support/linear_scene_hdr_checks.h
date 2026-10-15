#pragma once
#include "../../src/gpu_backend/backend.h"
#include "../../../shared/HdrImageOutputContract.h"
#include <cstring>

namespace linearscenechecks
{
// An opaque quad fills every output pixel. The glTF emissive factor is linear;
// its RGB8 texture is sRGB. Runtime gain supplies highlights above one before
// the compositor (post exposure remains one).
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene()
{
    using namespace HarmonicMIDI::grid;
    auto value = std::make_shared<Visual3DScene>();
    value->id = Scene3DId {1}; value->activeCamera = SceneCameraId {4};
    value->ambientColor = {}; value->lightCount = 0;
    value->vertexCount = 4; value->indexCount = 6;
    value->vertices[0] = {{-2, 2, 0}, {0, 0, 1}, {}, {0, 0}};
    value->vertices[1] = {{2, 2, 0}, {0, 0, 1}, {}, {1, 0}};
    value->vertices[2] = {{-2, -2, 0}, {0, 0, 1}, {}, {0, 1}};
    value->vertices[3] = {{2, -2, 0}, {0, 0, 1}, {}, {1, 1}};
    const std::array<std::uint32_t, 6> indices {0, 2, 1, 1, 2, 3};
    std::copy(indices.begin(), indices.end(), value->indices.begin());
    value->objects[0].id = SceneObjectId {2}; value->objects[0].material = SceneMaterialId {3};
    value->objects[0].vertexCount = 4; value->objects[0].indexCount = 6; value->objectCount = 1;
    value->materials[0].id = SceneMaterialId {3}; value->materials[0].baseColor = {};
    value->materials[0].emissive = {0.5f, 0.25f, 0.125f};
    value->materials[0].emissiveTexture = SceneTextureId {5};
    value->materials[0].doubleSided = true; value->materialCount = 1;
    value->textures[0].id = SceneTextureId {5}; value->textures[0].width = 1; value->textures[0].height = 1;
    value->textureCount = 1; value->textureTexelCount = 1;
    value->textureTexels = {{255, 128, 64, 255}};
    value->cameras[0].id = SceneCameraId {4}; value->cameras[0].transform.translation.z = 2;
    value->cameras[0].verticalFovRadians = 0.7853981634f;
    value->cameras[0].nearPlane = 0.1f; value->cameras[0].farPlane = 100; value->cameraCount = 1;
    return value;
}
inline std::array<float, 4> expected()
{
    return {4, static_cast<float>(2 * hdrimage::decode(128.0 / 255, colortransform::TransferFunction::SRGB)),
        static_cast<float>(hdrimage::decode(64.0 / 255, colortransform::TransferFunction::SRGB)), 1};
}
inline bool nearPixels(const std::vector<float>& pixels, std::string& error)
{
    const auto oracle = expected();
    if (pixels.empty() || pixels.size() % 4 != 0) { error = "Missing linear scene pixels"; return false; }
    for (std::size_t i = 0; i < pixels.size(); ++i)
        if (!std::isfinite(pixels[i]) || std::abs(pixels[i] - oracle[i % 4]) > 0.006f)
        { error = "Linear scene RGB differs from the unclipped sRGB-texture/emission oracle"; return false; }
    return true;
}
inline bool raw(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame, std::string& error)
{
    arbitgpu::NativeRawPassPixels pixels, repeat;
    if (!frame || !arbitgpu::isLinearSceneColor(frame->colorTextureDescriptor())
        || !frame->readRawPass(renderpassoutput::Output::Color, pixels, error)
        || !frame->readRawPass(renderpassoutput::Output::Color, repeat, error)
        || pixels.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float || pixels.bytes != repeat.bytes)
    { error = "Native linear color descriptor/readback/replay differs: " + error; return false; }
    std::vector<float> values(pixels.bytes.size() / 2);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        std::uint16_t half; std::memcpy(&half, pixels.bytes.data() + i * 2, 2);
        values[i] = hdrimage::halfToFloat(half);
    }
    return nearPixels(values, error);
}
inline std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> capture(
    arbitgpu::NativeFixtureSceneBackend& backend, std::string& error)
{
    auto value = scene();
    auto resources = backend.prepare(value, nullptr);
    if (!resources.prepared) { error = resources.error; return {}; }
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true; inputs.emissionGain = 8;
    auto captured = backend.render(value, resources.resources, 64, 64, inputs);
    if (!captured.rendered || !raw(captured.frame, error)) { error += captured.error; return {}; }
    auto repeated = backend.render(value, resources.resources, 64, 64, inputs);
    if (!repeated.rendered || !raw(repeated.frame, error)) return {};
    inputs.passComposite = renderpasscomposite::Parameters {};
    inputs.passComposite->mode = renderpasscomposite::Mode::MaskedColor;
    inputs.passComposite->outputTransform = renderpasscomposite::OutputTransform::ReinhardToSRGB;
    auto pass = backend.render(value, resources.resources, 64, 64, inputs);
    if (!pass.rendered || !raw(pass.frame, error)) return {};
    inputs.passComposite.reset(); inputs.imageOutput = renderpassoutput::Output::Depth;
    if (backend.render(value, resources.resources, 64, 64, inputs).rendered)
    { error = "HDR admitted a data-pass visualization as linear color"; return {}; }
    inputs.imageOutput = renderpassoutput::Output::Color; inputs.linearColor = false;
    auto sdr = backend.render(value, resources.resources, 64, 64, inputs);
    if (!sdr.rendered || arbitgpu::isLinearSceneColor(sdr.frame->colorTextureDescriptor()))
    { error = "Native HDR changed the default SDR resource contract"; return {}; }
    return captured.frame;
}

inline bool diffraction(arbitgpu::NativeFixtureSceneBackend& backend,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>& resources, std::string& error)
{
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor = true;
    const auto first = backend.render(scene, resources, 96, 96, inputs);
    const auto again = backend.render(scene, resources, 96, 96, inputs);
    arbitgpu::NativeRawPassPixels a, b;
    if (!first.rendered || !again.rendered || !arbitgpu::isLinearSceneColor(first.frame->colorTextureDescriptor())
        || !first.frame->readRawPass(renderpassoutput::Output::Color, a, error)
        || !again.frame->readRawPass(renderpassoutput::Output::Color, b, error) || a.bytes != b.bytes)
    { error = "Diffraction linear color capture/replay failed: " + error + first.error + again.error; return false; }
    for (std::size_t i = 0; i + 1 < a.bytes.size(); i += 2)
    {
        std::uint16_t half; std::memcpy(&half, a.bytes.data() + i, 2);
        if (!std::isfinite(hdrimage::halfToFloat(half)))
        { error = "Diffraction HDR color contains a non-finite channel"; return false; }
    }
    return true;
}
} // namespace linearscenechecks
