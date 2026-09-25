#pragma once

#include "fixture_scene.h"
#include "../../src/gpu_backend/backend.h"
#include "../../../shared/HdrImageOutputContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace videohelper::tests
{
class MaterialFrameDescriptorOverride final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    MaterialFrameDescriptorOverride(std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> owner,
                                   arbitgpu::NativeTextureViewDescriptor descriptor)
        : owner_(std::move(owner)), descriptor_(std::move(descriptor)) {}
    const std::string& backend() const noexcept override { return descriptor_.backend; }
    std::uint32_t width() const noexcept override { return descriptor_.width; }
    std::uint32_t height() const noexcept override { return descriptor_.height; }
    std::uintptr_t colorImageHandle() const noexcept override { return descriptor_.imageHandle; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return descriptor_.textureViewHandle; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override { return descriptor_; }
private:
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> owner_;
    arbitgpu::NativeTextureViewDescriptor descriptor_;
};

// The same physical-backend cases run on GL and Metal. Readback exists only
// in the test harness; production binds the retained image without CPU pixels.
template <typename ReadRgba>
bool materialFrameBindingCases(arbitgpu::NativeFixtureSceneBackend& backend, ReadRgba readRgba)
{
    using namespace arbitgpu;
    bool ok = true;
    const auto check = [&](bool value, const char* message)
    {
        if (!value) { std::cerr << "FAIL: " << message << '\n'; ok = false; }
        return value;
    };
    auto value = fixture3d::makeScene();
    value.vertexCount = 4; value.indexCount = 6;
    value.vertices[0] = { { -2, 2, 0 }, { 0, 0, 1 }, {}, { 0, 0 } };
    value.vertices[1] = { { 2, 2, 0 }, { 0, 0, 1 }, {}, { 1, 0 } };
    value.vertices[2] = { { -2, -2, 0 }, { 0, 0, 1 }, {}, { 0, 1 } };
    value.vertices[3] = { { 2, -2, 0 }, { 0, 0, 1 }, {}, { 1, 1 } };
    const std::array<std::uint32_t, 6> indices { 0, 2, 1, 1, 2, 3 };
    std::copy(indices.begin(), indices.end(), value.indices.begin());
    value.objects[0].transform = {};
    value.objects[0].vertexCount = 4; value.objects[0].indexCount = 6;
    value.cameras[0].transform = {};
    value.cameras[0].transform.translation.z = 2;
    value.lightCount = 0; value.ambientColor = { 1, 1, 1 };
    value.materials[0].baseColor = { 128.0f / 255.0f, 0, 0 };
    value.materials[0].baseColorTexture = {};
    value.materials[0].metallic = 0; value.materials[0].roughness = 0;
    value.materials[0].emissive = {}; value.materials[0].doubleSided = true;
    value.textureCount = 0; value.textureTexelCount = 0;
    value.textureTexels.clear();
    auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(value);
    auto sourcePreparation = backend.prepare(scene, nullptr);
    auto source = backend.render(scene, sourcePreparation.resources, 64, 64, {});
    if (!check(source.rendered, "material Frame source renders on the physical backend")) return false;

    auto material = std::make_shared<NativeFixtureSurfaceMaterialProgram>();
    material->backend = backend.info().backend == "metal"
        ? NativeFixtureMaterialBackend::Metal : NativeFixtureMaterialBackend::OpenGl;
    material->object = value.objects[0].id;
    material->bindingDigest = "frame-texture-binding";
    material->programIdentity = "frame-texture-program";
    material->baseColorSource = NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture;
    material->parameters.baseColorMetallic = { 1, 1, 1, 0 };
    material->parameters.normalOpacity = { 0, 0, 1, 1 };
    // Index-matched, nontransmitting material: unit diffuse response isolates
    // texture transfer from the additional dielectric reflection term.
    material->parameters.transmissionIorClearcoat = { 0, 1.0f, 0, 0 };
    material->parameters.identifiers[0] = value.materials[0].id.value;
    auto preparation = backend.prepare(scene, material);
    if (!check(preparation.prepared, "Frame material prepares without an imported base texture")) return false;
    check(!backend.render(scene, preparation.resources, 64, 64, {}).rendered,
          "physical draw rejects an absent Frame lease");
    NativeFixtureSceneRuntimeInputs inputs;
    inputs.materialFrameTexture = source.frame;
    check(!backend.render(scene, sourcePreparation.resources, 64, 64, inputs).rendered,
          "an unrequested Frame cannot override an imported/default material");
    const auto descriptor = source.frame->colorTextureDescriptor();
    std::string error;
    check(!validateMaterialFrameForDraw(material, inputs, descriptor.backend.c_str(),
              descriptor.deviceOrContextIdentity + 1, error),
          "material Frame device/context identity is checked before drawing");
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        auto invalid = descriptor;
        if (mutation == 0) ++invalid.deviceOrContextIdentity;
        if (mutation == 1) --invalid.width;
        if (mutation == 2) invalid.rendererGeneration = 0;
        inputs.materialFrameTexture = std::make_shared<MaterialFrameDescriptorOverride>(source.frame, invalid);
        check(!backend.render(scene, preparation.resources, 64, 64, inputs).rendered,
              "physical draw rejects wrong-device, wrong-extent and stale-generation Frame descriptors");
    }
    inputs.materialFrameTexture = source.frame;
    auto preview = backend.render(scene, preparation.resources, 64, 64, inputs);
    auto exported = backend.render(scene, preparation.resources, 64, 64, inputs);
    if (!check(preview.rendered && exported.rendered,
               "preview/export draws consume the same owned Frame binding")) return false;
    const auto previewPixels = readRgba(preview.frame);
    const auto exportPixels = readRgba(exported.frame);
    const auto sourcePixels = readRgba(source.frame);
    constexpr std::size_t center = (32 * 64 + 32) * 4;
    if (check(previewPixels.size() == 64 * 64 * 4 && sourcePixels.size() == 64 * 64 * 4,
              "native Frame sampling has complete readback"))
    {
        const double encoded = sourcePixels[center] / 255.0;
        const int expected = static_cast<int>(std::lround(255 * (encoded <= 0.04045
            ? encoded / 12.92 : std::pow((encoded + 0.055) / 1.055, 2.4))));
        check(sourcePixels[center] >= 120 && sourcePixels[center] <= 136
                  && std::abs(static_cast<int>(previewPixels[center]) - expected) <= 3
                  && previewPixels[center + 1] < 3 && previewPixels[center + 2] < 3,
              "GL/Metal Frame sampling decodes sRGB once against the CPU oracle");
    }
    check(previewPixels == exportPixels, "unchanged Frame preview/export samples are identical");
    inputs.linearColor = true;
    check(!backend.render(scene, preparation.resources, 64, 64, inputs).rendered,
          "HDR rejects an undeclared Frame transfer instead of inventing video sRGB metadata");
    auto declaredSrgb = descriptor;
    declaredSrgb.colorSpace = colortransform::ColorSpace::SRGB;
    declaredSrgb.transfer = colortransform::TransferFunction::SRGB;
    inputs.materialFrameTexture = std::make_shared<MaterialFrameDescriptorOverride>(source.frame, declaredSrgb);
    auto hdr = backend.render(scene, preparation.resources, 64, 64, inputs);
    NativeRawPassPixels linearPixels;
    check(hdr.rendered && hdr.frame->readRawPass(renderpassoutput::Output::Color, linearPixels, error)
        && linearPixels.format == NativeTexturePixelFormat::Rgba16Float,
        "an explicitly sRGB owned Frame is decoded by the native HDR material sampler");
    if (linearPixels.bytes.size() == 64 * 64 * 8)
    {
        std::uint16_t half; std::memcpy(&half, linearPixels.bytes.data() + center * 2, 2);
        const float sampled = hdrimage::halfToFloat(half);
        check(std::abs(sampled - hdrimage::decode(sourcePixels[center] / 255.0,
                colortransform::TransferFunction::SRGB)) < 0.003,
              "GL/Metal HDR Frame sampling decodes the declared sRGB value exactly once");
    }
    hdr.frame.reset();
    std::weak_ptr<const NativeFixtureSceneFrame> lifetime = source.frame;
    source.frame.reset(); inputs.materialFrameTexture.reset();
    check(!lifetime.expired(), "draw outputs retain the input lease after scheduler release");
    preview.frame.reset(); exported.frame.reset();
    check(lifetime.expired(), "the input lease releases after the final consuming output");
    return ok;
}
} // namespace videohelper::tests
