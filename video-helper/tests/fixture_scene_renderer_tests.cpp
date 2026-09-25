#include "../src/fixture_scene_renderer.h"
#include "support/fixture_scene.h"
#include "support/vertex_modifier_fixture.h"
#include "../src/gpu_backend/fixture_vertex_modifier_shader.h"
#include "../src/gpu_backend/fixture_texture_upload.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/DiffractionProductPlans.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
using namespace surfacematerial;

static_assert (std::is_same_v<
    decltype (diffractionmaterial::DiffractionProductRenderRequest::plan),
    std::shared_ptr<const diffractionmaterial::ImmutableDiffractionScenePlan>>,
    "product requests must own immutable plans rather than caller pointers");

int failures = 0;
int checks = 0;

void check (bool value, const char* message)
{
    ++checks;
    if (! value)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

void checkFixtureTextureUploads()
{
    using namespace arbitgpu::fixturetexture;
    struct LayoutCase { std::uint32_t width, height; std::size_t levels, bytes; };
    for (const auto value : { LayoutCase { 8, 4, 4, 172 }, { 5, 3, 3, 72 },
                              { 1, 9, 4, 64 }, { 9, 1, 4, 64 }, { 1, 1, 1, 4 },
                              { 1024, 1024, 11, 5592404 } })
    {
        MipLayout layout;
        check (mipLayout (value.width, value.height, 9987, 16384, layout)
                   && layout.levelCount == value.levels && layout.bytes == value.bytes,
               "mip layout counts power-of-two, odd and one-axis extents exactly");
        std::size_t offset = 0;
        for (std::size_t level = 0; level < layout.levelCount; ++level)
        {
            const auto& mip = layout.levels[level];
            check (mip.width == std::max (1u, value.width >> level)
                       && mip.height == std::max (1u, value.height >> level)
                       && mip.offsetBytes == offset
                       && mip.bytes == static_cast<std::size_t> (mip.width) * mip.height * 4,
                   "every mip matches native floor-halving dimensions and contiguous bytes");
            offset += mip.bytes;
        }
    }
    struct FilterCase { std::uint32_t filter; bool linear, mipmapped, linearMipmap; };
    for (const auto value : { FilterCase { 9728, false, false, false },
            { 9729, true, false, false }, { 9984, false, true, false },
            { 9985, true, true, false }, { 9986, false, true, true }, { 9987, true, true, true } })
    {
        Minification sampling;
        MipLayout layout;
        check (minification (value.filter, sampling) && sampling.linear == value.linear
                   && sampling.mipmapped == value.mipmapped && sampling.linearMipmap == value.linearMipmap
                   && mipLayout (8, 4, value.filter, 16384, layout)
                   && layout.levelCount == (value.mipmapped ? 4u : 1u),
               "glTF minification separates texel filtering from mip filtering");
    }
    MipLayout fox;
    check (mipLayout (1024, 1024, 9987, 16384, fox), "original Fox extent admits eleven mip levels");
    std::size_t bytes = 4;
    check (accountBytes (fox, 1, kMaximumTextureBytes, bytes) && bytes == 5592408,
           "GL charges every Fox mip and one white fallback");
    bytes = 4;
    check (accountBytes (fox, 2, 11184812, bytes) && bytes == 11184812,
           "Metal charges both Fox role copies and one white fallback at the exact budget");
    bytes = 4;
    check (! accountBytes (fox, 2, 11184811, bytes) && bytes == 4
               && ! accountBytes (fox, 0, kMaximumTextureBytes, bytes)
               && ! accountBytes (fox, std::numeric_limits<std::size_t>::max(), kMaximumTextureBytes, bytes)
               && ! accountBytes (fox, 1, 0, bytes)
               && ! accountBytes (fox, 1, kMaximumTextureBytes + 1, bytes),
           "invalid budgets and copy counts fail without changing allocation accounting");
    MipLayout importedFallback;
    check (mipLayout (2, 2, 9728, 16384, importedFallback), "imported material fallback has one level");
    bytes = 0;
    check (accountBytes (importedFallback, 1, kMaximumTextureBytes, bytes) && bytes == 16,
           "an imported fallback replaces the white allocation without charging phantom texels");
    mipLayout (1024, 1024, 9728, 16384, importedFallback);
    bytes = 0;
    check (accountBytes (importedFallback, 1, kMaximumTextureBytes, bytes)
               && accountBytes (fox, 2, kMaximumTextureBytes, bytes) && bytes == 15379112,
           "both Fox role chains and an owned full-size imported fallback fit the retained budget");
    MipLayout rejected;
    check (! mipLayout (0, 1, 9987, 16384, rejected)
               && ! mipLayout (1, 0, 9987, 16384, rejected)
               && ! mipLayout (1024, 1025, 9987, 16384, rejected)
               && ! mipLayout (1024, 1024, 9987, 512, rejected)
               && ! mipLayout (1, 1, 9987, 0, rejected)
               && ! mipLayout (65536, 1, 9987, 65536, rejected)
               && ! mipLayout (UINT32_MAX, UINT32_MAX, 9987, UINT32_MAX, rejected)
               && ! mipLayout (1, 1, 9988, 16384, rejected)
               && rejected.levelCount == 0,
           "zero, overflow, source-cap, device-cap, mip-count and filter violations are rejected");

    auto scene = std::make_unique<Visual3DScene>();
    scene->textureTexels = { SceneTexelRgba8 {}, {}, {}, {} };
    scene->textureTexelCount = 4;
    scene->textureCount = 2;
    scene->textures[0].width = scene->textures[0].height = 2;
    scene->textures[1] = scene->textures[0];
    SceneMipLayouts layouts;
    bytes = 4;
    check (sceneMipLayouts (*scene, 2, 16384, 84, bytes, layouts) && bytes == 84,
           "aliased texture records charge separate uploads while preserving shared source storage");
    bytes = 4;
    check (! sceneMipLayouts (*scene, 2, 16384, 83, bytes, layouts) && bytes == 4,
           "whole-scene preflight fails atomically when the final copy exceeds budget");
    scene->textures[1].magFilter = 9987;
    check (! sceneMipLayouts (*scene, 1, 16384, kMaximumTextureBytes, bytes, layouts),
           "mipmap modes are invalid magnification filters");
    scene->textures[1].magFilter = 9729;
    scene->textures[1].wrapS = 0;
    check (! sceneMipLayouts (*scene, 1, 16384, kMaximumTextureBytes, bytes, layouts),
           "invalid horizontal wrapping is rejected before upload");
    scene->textures[1].wrapS = 10497;
    scene->textures[1].wrapT = 0;
    check (! sceneMipLayouts (*scene, 1, 16384, kMaximumTextureBytes, bytes, layouts),
           "invalid vertical wrapping is rejected before upload");
    scene->textures[1].wrapT = 10497;
    scene->textures[1].firstTexel = 1;
    check (! sceneMipLayouts (*scene, 1, 16384, kMaximumTextureBytes, bytes, layouts),
           "source range must fit the populated texels before upload");
    scene->textures[1].firstTexel = 0;
    scene->textureTexelCount = kMaximumSourceTexels + 1;
    check (! sceneMipLayouts (*scene, 1, 16384, kMaximumTextureBytes, bytes, layouts),
           "aggregate source storage cannot exceed one million RGBA8 texels");
    scene->textureTexelCount = 4;
    scene->textureCount = Visual3DScene::kMaxTextures + 1;
    check (! sceneMipLayouts (*scene, 1, 16384, kMaximumTextureBytes, bytes, layouts),
           "texture record count is checked before reading records");

    const std::array<SceneTexelRgba8, 4> image {{
        { 0, 0, 0, 0 }, { 255, 255, 255, 255 }, { 255, 255, 255, 255 }, { 0, 0, 0, 0 }
    }};
    const auto original = image;
    MipLayout layout;
    mipLayout (2, 2, 9987, 16384, layout);
    std::vector<SceneTexelRgba8> linear, color;
    const bool generated = mipTexels (image.data(), image.size(), layout, false, linear)
        && mipTexels (image.data(), image.size(), layout, true, color);
    check (generated && linear.size() == 5 && color.size() == 5
               && linear.back().red == 128 && color.back().red == 188
               && linear.back().green == 128 && color.back().green == 188
               && linear.back().blue == 128 && color.back().blue == 188
               && linear.back().alpha == 128 && color.back().alpha == 128,
           "color mips average RGB in linear light and keep alpha and data channels linear");
    check (generated && std::memcmp (image.data(), original.data(), sizeof (image)) == 0
               && std::memcmp (linear.data(), image.data(), sizeof (image)) == 0
               && std::memcmp (color.data(), image.data(), sizeof (image)) == 0,
           "source and both uploaded level-zero images remain byte-identical");
    std::vector<SceneTexelRgba8> odd (15);
    odd.back() = { 255, 255, 255, 255 };
    mipLayout (5, 3, 9987, 16384, layout);
    check (mipTexels (odd.data(), odd.size(), layout, false, linear)
               && linear.size() == 18 && linear[16].red == 34 && linear.back().red == 17,
           "odd-size mip footprints include the final row and column");
    const std::array<SceneTexelRgba8, 5> strip {{
        { 0, 0, 0, 255 }, { 10, 10, 10, 255 }, { 20, 20, 20, 255 },
        { 30, 30, 30, 255 }, { 40, 40, 40, 255 }
    }};
    for (const auto width : { 1u, 5u })
    {
        mipLayout (width, 6u - width, 9987, 16384, layout);
        check (mipTexels (strip.data(), strip.size(), layout, false, linear)
                   && linear.size() == 8 && linear[5].red == 8
                   && linear[6].red == 32 && linear.back().red == 20,
               "one-axis mips retain all samples and stop at one texel");
    }
    mipLayout (2, 2, 9729, 16384, layout);
    check (mipTexels (image.data(), image.size(), layout, true, color)
               && color.size() == 4 && std::memcmp (color.data(), image.data(), sizeof (image)) == 0,
           "non-mipmapped images retain only the unchanged source level");
    const auto retainedSize = color.size();
    check (! mipTexels (nullptr, image.size(), layout, false, color)
               && ! mipTexels (image.data(), image.size() - 1, layout, false, color),
           "missing and truncated source data fail before mip allocation");
    layout.levels[0].offsetBytes = 4;
    check (! mipTexels (image.data(), image.size(), layout, false, color)
               && color.size() == retainedSize,
           "forged mip offsets fail without replacing existing output");
}

bool near (float actual, float expected) noexcept
{
    return std::abs (actual - expected) <= 1.0e-6f;
}

ValueId append (ProgramDescription& program, OperationKind kind, ValueType type,
                std::initializer_list<ValueId> inputs = {})
{
    Operation operation;
    operation.id = static_cast<ValueId> (program.operations.size() + 1);
    operation.kind = kind;
    operation.resultType = type;
    operation.inputCount = static_cast<std::uint8_t> (inputs.size());
    std::copy (inputs.begin(), inputs.end(), operation.inputs.begin());
    program.operations.push_back (operation);
    return operation.id;
}

ValueId constant (ProgramDescription& program, ValueType type,
                  std::array<float, 4> literal = {})
{
    const auto id = append (program,
                            type == ValueType::UInt ? OperationKind::UIntConstant
                                                    : OperationKind::FloatConstant,
                            type);
    program.operations.back().literal = literal;
    return id;
}

void bindConstantPbrOutputs (ProgramDescription& program, ValueId baseColor,
                             std::uint32_t materialId,
                             std::array<float, 4> emission = {})
{
    program.outputs[static_cast<std::size_t> (OutputSemantic::BaseColor)] = baseColor;
    program.outputs[static_cast<std::size_t> (OutputSemantic::Metallic)]
        = constant (program, ValueType::Scalar, { 0.6f, 0.0f, 0.0f, 0.0f });
    program.outputs[static_cast<std::size_t> (OutputSemantic::Roughness)]
        = constant (program, ValueType::Scalar, { 0.25f, 0.0f, 0.0f, 0.0f });
    program.outputs[static_cast<std::size_t> (OutputSemantic::Emission)]
        = constant (program, ValueType::Vec3, emission);
    program.outputs[static_cast<std::size_t> (OutputSemantic::Opacity)]
        = constant (program, ValueType::Scalar, { 1.0f, 0.0f, 0.0f, 0.0f });
    program.outputs[static_cast<std::size_t> (OutputSemantic::Normal)]
        = constant (program, ValueType::Vec3, { 0.0f, 0.0f, 1.0f, 0.0f });
    program.outputs[static_cast<std::size_t> (OutputSemantic::Transmission)]
        = constant (program, ValueType::Scalar);
    program.outputs[static_cast<std::size_t> (OutputSemantic::Ior)]
        = constant (program, ValueType::Scalar, { 1.5f, 0.0f, 0.0f, 0.0f });
    program.outputs[static_cast<std::size_t> (OutputSemantic::Clearcoat)]
        = constant (program, ValueType::Scalar);
    program.outputs[static_cast<std::size_t> (OutputSemantic::MaterialId)]
        = constant (program, ValueType::UInt);
    program.operations.back().unsignedLiteral = materialId;
}

ProgramDescription constantSurfaceProgram (std::uint32_t materialId,
                                           float red = 0.9f,
                                           std::array<float, 4> emission = {})
{
    ProgramDescription program;
    const auto baseColor = constant (
        program, ValueType::Vec3, { red, 0.2f, 0.1f, 0.0f });
    bindConstantPbrOutputs (program, baseColor, materialId, emission);
    return program;
}

ProgramDescription importedTextureSurfaceProgram (std::uint32_t materialId)
{
    ProgramDescription program;
    program.textureSlotCount = 1;
    const auto uv = append (program, OperationKind::Input, ValueType::Vec2);
    program.operations.back().semantic = InputSemantic::TexCoord0;
    const auto sampled = append (
        program, OperationKind::TextureSample2D, ValueType::Vec4, { uv });
    program.operations.back().parameter = 0;
    std::array<ValueId, 3> components {};
    for (std::size_t lane = 0; lane < components.size(); ++lane)
    {
        components[lane] = append (
            program, OperationKind::Component, ValueType::Scalar, { sampled });
        program.operations.back().parameter = static_cast<std::uint16_t> (lane);
    }
    const auto baseColor = append (
        program, OperationKind::ComposeVec3, ValueType::Vec3,
        { components[0], components[1], components[2] });
    bindConstantPbrOutputs (program, baseColor, materialId);
    return program;
}

ProgramDescription tintedImportedTextureSurfaceProgram (
    std::uint32_t materialId,
    std::array<float, 4> tintValue = { 0.25f, 0.5f, 0.75f, 0.0f })
{
    auto program = importedTextureSurfaceProgram (materialId);
    const auto sampledRgb = program.outputs[
        static_cast<std::size_t> (OutputSemantic::BaseColor)];
    const auto tint = constant (program, ValueType::Vec3, tintValue);
    const auto tinted = append (
        program, OperationKind::Multiply, ValueType::Vec3, { sampledRgb, tint });
    program.outputs[static_cast<std::size_t> (OutputSemantic::BaseColor)] = tinted;
    return program;
}

ProgramDescription timeMixSurfaceProgram (std::uint32_t materialId)
{
    ProgramDescription program;
    const auto start = constant (
        program, ValueType::Vec3, { 0.1f, 0.2f, 0.3f, 0.0f });
    const auto end = constant (
        program, ValueType::Vec3, { 0.9f, 0.7f, 0.5f, 0.0f });
    const auto time = append (program, OperationKind::Input, ValueType::Scalar);
    program.operations.back().semantic = InputSemantic::Time;
    const auto lower = constant (program, ValueType::Scalar);
    const auto upper = constant (
        program, ValueType::Scalar, { 1.0f, 0.0f, 0.0f, 0.0f });
    const auto clamped = append (
        program, OperationKind::Clamp, ValueType::Scalar, { time, lower, upper });
    const auto baseColor = append (
        program, OperationKind::Mix, ValueType::Vec3, { start, end, clamped });
    bindConstantPbrOutputs (program, baseColor, materialId);
    return program;
}

surfacematerialbinding::ImportedSceneMaterialRequest materialRequest (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    ProgramDescription program,
    std::uint64_t revision = 1)
{
    std::string error;
    const auto admitted = admit (program, error);
    surfacematerialbinding::ImportedSceneMaterialRequest request;
    request.sceneSnapshot = scene;
    request.scene = scene->id;
    request.sceneRevision = 1;
    request.structuralRevision = revision;
    request.evaluationRevision = revision;
    request.programRevision = revision;
    request.program = std::move (program);
    request.binding.targetKind
        = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    request.binding.object = scene->objects[0].id;
    request.binding.surfaceMaterialDigest
        = admitted ? admitted->structuralDigest() : std::string {};
    request.binding.surfaceMaterialRevision = revision;
    if (request.program.textureSlotCount == 1)
    {
        request.binding.textures.push_back ({
            0, surfacematerialbinding::TextureSourceKind::ImportedBaseColor, {} });
    }
    return request;
}

diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest
diffractionRequest (
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    std::uint64_t revision = 1)
{
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
    request.scene = scene->id;
    request.sceneRevision = 1;
    request.structuralRevision = revision;
    request.evaluationRevision = revision;
    request.materialRevision = revision;
    request.object = scene->objects[0].id;
    request.material = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    request.lighting = diffractionmaterial::makeReferenceLighting();
    std::string error;
    const auto admitted = diffractionmaterial::admit (request.material, error);
    if (admitted)
        request.structuralDigest = admitted->structuralDigest();
    return request;
}

class FakeFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    FakeFrame (std::string backend, std::uint32_t width, std::uint32_t height,
               std::uintptr_t handle, int& deleted) noexcept
        : backend_ (std::move (backend)), width_ (width), height_ (height),
          handle_ (handle), deleted_ (deleted)
    {
    }

    ~FakeFrame() override
    {
        ++deleted_;
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return handle_; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return handle_ + 1000; }

private:
    std::string backend_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uintptr_t handle_ = 0;
    int& deleted_;
};

class FakeResources final : public arbitgpu::NativeFixtureSceneResources
{
public:
    explicit FakeResources (std::string backend) : backend_ (std::move (backend)) {}
    const std::string& backend() const noexcept override { return backend_; }
private:
    std::string backend_;
};

class FakeBackend final : public arbitgpu::NativeFixtureSceneBackend
{
public:
    arbitgpu::BackendInfo info() const override
    {
        auto result = backendInfo;
        return result;
    }

    arbitgpu::NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& material) override
    {
        ++preparations;
        preparedMaterial = material;
        arbitgpu::NativeFixtureScenePreparation result;
        result.prepared = true;
        result.resources = std::make_shared<FakeResources> (backendInfo.backend);
        result.stats.vertexBytes = scene->vertexCount * sizeof (scene->vertices[0]);
        result.stats.indexBytes = scene->indexCount * sizeof (scene->indices[0]);
        result.stats.textureBytes = scene->textureTexelCount > 0
            ? scene->textureTexelCount * sizeof (scene->textureTexels[0]) : 4;
        result.stats.staticUploadCount = static_cast<std::uint32_t> (scene->objectCount);
        result.stats.textureUploadCount = static_cast<std::uint32_t> (scene->objectCount);
        return result;
    }

    arbitgpu::NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
        std::uint32_t width,
        std::uint32_t height,
        arbitgpu::NativeFixtureSceneRuntimeInputs runtimeInputs) override
    {
        ++submissions;
        submittedRuntimeInputs.push_back (runtimeInputs);
        submittedScene = scene.get();
        submittedWidth = width;
        submittedHeight = height;
        arbitgpu::NativeFixtureSceneSubmission result;
        if (rejectSubmission)
        {
            result.error = "fake native submission rejected";
            return result;
        }
        result.rendered = true;
        result.frame = std::make_shared<FakeFrame> (
            backendInfo.backend, width, height,
            static_cast<std::uintptr_t> (submissions), deletedFrames);
        result.stats.drawCount = static_cast<std::uint32_t> (scene->objectCount);
        result.stats.materialBytes = sizeof (scene->materials[0]);
        return result;
    }

    arbitgpu::BackendInfo backendInfo { true, false, "metal", "fake", {} };
    bool rejectSubmission = false;
    int submissions = 0;
    int preparations = 0;
    int deletedFrames = 0;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> preparedMaterial;
    std::vector<arbitgpu::NativeFixtureSceneRuntimeInputs> submittedRuntimeInputs;
    const HarmonicMIDI::grid::Visual3DScene* submittedScene = nullptr;
    std::uint32_t submittedWidth = 0;
    std::uint32_t submittedHeight = 0;
};
} // namespace

int main()
{
    checkFixtureTextureUploads();
    using namespace HarmonicMIDI::grid;
    using namespace videorender::fixture3d;

    {
        using Texture = arbitgpu::NativeFixtureSurfaceMaterialProgram::ImportedSrgbTexture;
        static_assert(sizeof(Texture) < 1024, "native texture records must keep pixels off the stack");
        Texture texture;
        check(!texture.valid(), "empty native texture storage is not an admitted image");
        texture.width = texture.height = 1024;
        texture.texelCount = 1024u * 1024u;
        check(!texture.valid(), "native texture counts require actual owned pixels");
        texture.texels.resize(texture.texelCount);
        check(texture.valid(), "native material storage admits a full 4 MiB texture");
        texture.texels.pop_back();
        check(!texture.valid(), "native texture upload rejects truncated storage");
        texture.texels.resize(texture.texelCount);
        texture.width = Visual3DScene::kMaxTextureDimension + 1;
        check(!texture.valid(), "native texture upload rejects excessive dimensions");
        texture.width = 1024;
        ++texture.texelCount;
        check(!texture.valid(), "native texture upload rejects excessive declared texels");
    }

    auto scene = std::make_shared<const Visual3DScene> (videohelper::fixture3d::makeScene());
    FakeBackend backend;
    FixtureSceneRenderer renderer (backend);
    RenderedFrame preview;
    RenderedFrame exported;
    std::string error;
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs;
    sceneInputs.objectRotationDegrees[1] = 30.0f;
    sceneInputs.objectScale = 1.25f;

    check (renderer.renderPreview (scene, {}, {}, sceneInputs, { 96, 96 },
                                   kNativeGpuCapability, preview, error),
           "preview submits the immutable fixture to the native backend");
    check (renderer.renderExport (scene, { 96, 96 }, kNativeGpuCapability,
                                  exported, error),
           "export submits through the same native fixture renderer");
    check (backend.preparations == 1 && backend.submissions == 2
           && backend.submittedScene == scene.get()
           && backend.submittedWidth == 96 && backend.submittedHeight == 96
           && backend.submittedRuntimeInputs[0].objectRotationDegrees[1] == 30.0f
           && backend.submittedRuntimeInputs[0].objectScale == 1.25f,
           "preview and export retain one immutable snapshot contract");
    check (preview.use == RenderUse::Preview && exported.use == RenderUse::Export
           && preview.sceneId == scene->id && exported.sceneId == scene->id
           && preview.dimensions.width == exported.dimensions.width
           && preview.dimensions.height == exported.dimensions.height,
           "preview and export differ only in use metadata");
    check (preview.nativeFrame != nullptr && exported.nativeFrame != nullptr
           && preview.nativeFrame->backend() == "metal"
           && exported.nativeFrame->backend() == "metal"
           && preview.stats.drawCount == 1 && preview.stats.vertexBytes > 0
           && preview.stats.indexBytes > 0 && preview.stats.materialBytes > 0
           && preview.stats.textureBytes > 0,
           "native frame and bounded upload receipts reach both callers");
    check (preview.stats.staticUploadCount == 1
           && ! preview.stats.reusedStaticResources
           && exported.stats.staticUploadCount == 0
           && exported.stats.reusedStaticResources,
           "preview uploads static resources once and export reuses the exact GPU cache");

    auto materialSceneValue = videohelper::fixture3d::makeScene();
    materialSceneValue.textureCount = 0;
    materialSceneValue.textureTexelCount = 0;
    materialSceneValue.textureTexels.clear();
    materialSceneValue.materials[0].baseColorTexture = {};
    auto materialScene = std::make_shared<const Visual3DScene> (materialSceneValue);
    const auto materialDescription = constantSurfaceProgram (
        materialScene->materials[0].id.value);
    const auto admittedMaterial = admit (materialDescription, error);
    const auto reference = admittedMaterial
        ? evaluateReference (*admittedMaterial, {}, error) : std::nullopt;
    auto constantRequest = materialRequest (materialScene, materialDescription);
    auto materialBinding = admitSurfaceMaterialBinding (
        materialScene, constantRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    check (reference.has_value() && materialBinding != nullptr
           && materialBinding->sceneSnapshot() == materialScene,
           "the constant surface fixture independently admits and owns its exact scene snapshot");
    if (! reference || materialBinding == nullptr)
        return 1;

    FakeBackend materialBackend;
    FixtureSceneRenderer materialRenderer (materialBackend);
    RenderedFrame materialPreview;
    RenderedFrame materialExport;
    check (materialRenderer.renderPreview (
               materialScene, materialBinding, { 64, 64 }, kNativeGpuCapability,
               materialPreview, error)
           && materialBackend.preparedMaterial != nullptr,
           "the admitted constant surface program reaches native GPU preparation");
    const auto& nativePbr = materialBackend.preparedMaterial->parameters;
    check (materialBackend.preparedMaterial->backend
                == arbitgpu::NativeFixtureMaterialBackend::Metal
           && materialBackend.preparedMaterial->baseColorSource
                == arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear
           && ! materialBackend.preparedMaterial->importedBaseColorTexture
           && nativePbr.baseColorMetallic == reference->baseColorMetallic
           && nativePbr.emissionRoughness == reference->emissionRoughness
           && nativePbr.normalOpacity == reference->normalOpacity
           && nativePbr.transmissionIorClearcoat
                == reference->transmissionIorClearcoat
           && nativePbr.identifiers == reference->identifiers,
           "native constant values exactly match the independent reference evaluator");
    check (materialPreview.stats.materialProgramUploadCount == 1
           && ! materialPreview.stats.reusedMaterialProgram
           && materialRenderer.renderExport (
               materialScene, materialBinding, { 64, 64 }, kNativeGpuCapability,
               materialExport, error)
           && materialExport.stats.materialProgramUploadCount == 0
           && materialExport.stats.reusedMaterialProgram
           && materialBackend.preparations == 1,
           "preview and export retain one immutable program and pipeline preparation");

    auto vertexRequest = constantRequest;
    vertexRequest.vertexModifier = videohelper::test::audioNormalDisplacement();
    const auto vertexBinding = admitSurfaceMaterialBinding(materialScene, vertexRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    check(vertexBinding && vertexBinding->nativeProgram()->vertexProgram
          && vertexBinding->bindingDigest() != materialBinding->bindingDigest(),
          "native material admission retains the vertex stage in the exact binding identity");
    if (vertexBinding)
    {
        const auto shader = arbitgpu::fixtureVertexModifierShader("// ARBIT_VERTEX_MODIFIER",
            vertexBinding->nativeProgram()->vertexProgram.get(),
            videohelper::materialprogram::BackendTarget::Metal, error);
        check(shader.find("u.vertexSpectrum[1][2]") != std::string::npos
              && shader.find("modifiedPosition = vertexValue6") != std::string::npos,
              "the Metal shader executes the requested audio band and bounded displacement output");
        auto altered = vertexRequest;
        altered.vertexModifier->records[3].parameters[0] = 0.125;
        const auto alteredBinding = admitSurfaceMaterialBinding(materialScene, altered,
            videohelper::materialprogram::BackendTarget::Metal, error);
        check(alteredBinding && alteredBinding->bindingDigest() != vertexBinding->bindingDigest(),
              "changed vertex parameters cannot reuse a previous material binding");
    }
    auto unsupportedVertex = vertexRequest;
    unsupportedVertex.vertexModifier->records[2].operation = videowire::VertexModifierOperation::controlParameter;
    check(!admitSurfaceMaterialBinding(materialScene, unsupportedVertex,
              videohelper::materialprogram::BackendTarget::Metal, error)
          && error.find("control banks") != std::string::npos,
          "unavailable vertex control banks reject before native preparation");
    unsupportedVertex = vertexRequest;
    auto& noise = unsupportedVertex.vertexModifier->records[2];
    noise.operation = videowire::VertexModifierOperation::scalarNoise3d;
    noise.inputCount = 1;
    noise.inputs[0] = 1;
    noise.parameterCount = 2;
    noise.parameters[0] = 1.0;
    check(!admitSurfaceMaterialBinding(materialScene, unsupportedVertex,
              videohelper::materialprogram::BackendTarget::Metal, error)
          && error.find("noise") != std::string::npos,
          "unavailable vertex noise rejects before native preparation");
    unsupportedVertex = vertexRequest;
    auto& remap = unsupportedVertex.vertexModifier->records[3];
    remap.operation = videowire::VertexModifierOperation::scalarRemap;
    remap.inputCount = 1;
    remap.inputs[0] = 3;
    remap.parameterCount = 5;
    remap.parameters = {0.0, std::numeric_limits<double>::min(), 0.0, 1.0, 1.0, 0.0};
    check(!admitSurfaceMaterialBinding(materialScene, unsupportedVertex,
              videohelper::materialprogram::BackendTarget::Metal, error)
          && error.find("representable") != std::string::npos,
          "unrepresentable remap ranges reject before native division");

    auto noLightSceneValue = materialSceneValue;
    noLightSceneValue.lightCount = 0;
    noLightSceneValue.ambientColor = { 0.0f, 0.0f, 0.0f };
    auto noLightScene = std::make_shared<const Visual3DScene> (noLightSceneValue);
    const auto emissionDescription = constantSurfaceProgram (
        noLightScene->materials[0].id.value, 0.0f,
        { 0.8f, 0.08f, 0.02f, 0.0f });
    auto emissionRequest = materialRequest (noLightScene, emissionDescription, 2);
    auto emissionBinding = admitSurfaceMaterialBinding (
        noLightScene, emissionRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, error);
    FakeBackend emissionBackend;
    emissionBackend.backendInfo.backend = "opengl";
    FixtureSceneRenderer emissionRenderer (emissionBackend);
    RenderedFrame emissionPreview;
    const bool emissionRendered = emissionBinding != nullptr
        && emissionRenderer.renderPreview (
            noLightScene, emissionBinding, { 64, 64 }, kNativeGpuCapability,
            emissionPreview, error);
    if (! emissionRendered)
        std::fprintf (stderr, "emission fixture error: %s\n", error.c_str());
    check (emissionRendered
           && emissionBackend.preparedMaterial != nullptr
           && emissionBackend.preparedMaterial->parameters.emissionRoughness[0] == 0.8f
           && emissionBackend.preparedMaterial->parameters.emissionRoughness[1] == 0.08f
           && emissionBackend.preparedMaterial->parameters.emissionRoughness[2] == 0.02f,
           "the no-light fixture route admits and preserves constant surface emission");

    auto diffractionSceneValue = materialSceneValue;
    diffractionSceneValue.ambientColor = { 0.0f, 0.0f, 0.0f };
    diffractionSceneValue.lights[0].color = { 1.0f, 1.0f, 1.0f };
    diffractionSceneValue.lights[0].intensity = 0.05f;
    auto diffractionScene = std::make_shared<const Visual3DScene> (
        diffractionSceneValue);
    auto diffractionMaterial = diffractionRequest (diffractionScene, 7);
    auto diffractionBinding = admitDiffractionMaterialBinding (
        diffractionScene, diffractionMaterial,
        videohelper::materialprogram::BackendTarget::Metal, error);
    FakeBackend diffractionBackend;
    FixtureSceneRenderer diffractionRenderer (diffractionBackend);
    RenderedFrame diffractionPreview;
    RenderedFrame diffractionExport;
    check (diffractionBinding != nullptr
           && diffractionRenderer.renderPreview (
               diffractionScene, diffractionBinding, {}, { 64, 64 },
               kNativeGpuCapability, diffractionPreview, error)
           && diffractionRenderer.renderExport (
               diffractionScene, diffractionBinding, {}, { 64, 64 },
               kNativeGpuCapability, diffractionExport, error)
           && diffractionBackend.preparations == 1
           && diffractionBackend.preparedMaterial != nullptr
           && diffractionBackend.preparedMaterial->kind
                == arbitgpu::NativeFixtureMaterialKind::DiffractionReflective
           && diffractionBackend.preparedMaterial->diffractionPathCount == 3
           && diffractionBackend.preparedMaterial->diffractionPaths[0]
                  .incidentDirectionAndIntensity.w == 0.65f
           && diffractionBackend.preparedMaterial->diffractionPaths[1]
                  .incidentDirectionAndIntensity.w == 0.24f
           && diffractionBackend.preparedMaterial->diffractionPaths[2]
                  .incidentDirectionAndIntensity.w == 0.07f
           && diffractionBackend.preparedMaterial->diffractionPaths[0]
                  .kindBounceAndReserved[0]
                == static_cast<std::uint32_t>(diffractionmaterial::LightingPathKind::Direct)
           && diffractionBackend.preparedMaterial->diffractionPaths[1]
                  .kindBounceAndReserved[0]
                == static_cast<std::uint32_t>(diffractionmaterial::LightingPathKind::Environment)
           && diffractionBackend.preparedMaterial->diffractionPaths[2]
                  .kindBounceAndReserved[1] == 1u
           && diffractionExport.stats.reusedMaterialProgram,
            "preview and export share one exact one-dimensional diffraction program");

    auto foilPlan = diffractionmaterial::makeDiffractionReferenceScene (
        diffractionmaterial::ReferenceSceneKind::SpatialFoil);
    auto referencePlan = diffractionmaterial::makeDiffractionReferenceScene (
        diffractionmaterial::ReferenceSceneKind::LinearGrating);
    auto foilPreviewRequest = diffractionmaterial::makePreviewRequest (foilPlan);
    auto foilExportRequest = diffractionmaterial::makeExportRequest (foilPlan);
    auto referencePreviewRequest = diffractionmaterial::makePreviewRequest (referencePlan);
    auto referenceExportRequest = diffractionmaterial::makeExportRequest (referencePlan);
    foilPlan.material.maskCoverage = 0.0f;
    referencePlan.lighting.paths[0].incident.radiance.fill (0.0f);

    auto foilPreviewBinding = admitDiffractionProductPlan (
        diffractionScene, foilPreviewRequest, 1, 11, 11,
        videohelper::materialprogram::BackendTarget::Metal, error);
    auto foilExportBinding = admitDiffractionProductPlan (
        diffractionScene, foilExportRequest, 1, 11, 11,
        videohelper::materialprogram::BackendTarget::Metal, error);
    auto referencePreviewBinding = admitDiffractionProductPlan (
        diffractionScene, referencePreviewRequest, 1, 12, 12,
        videohelper::materialprogram::BackendTarget::Metal, error);
    auto referenceExportBinding = admitDiffractionProductPlan (
        diffractionScene, referenceExportRequest, 1, 12, 12,
        videohelper::materialprogram::BackendTarget::Metal, error);
    FakeBackend foilPreviewBackend;
    FakeBackend foilExportBackend;
    FixtureSceneRenderer foilPreviewRenderer (foilPreviewBackend);
    FixtureSceneRenderer foilExportRenderer (foilExportBackend);
    RenderedFrame productPreview;
    RenderedFrame productExport;
    const bool foilRendered = foilPreviewBinding != nullptr && foilExportBinding != nullptr
        && foilPreviewRenderer.renderPreview (
            diffractionScene, foilPreviewBinding, {}, { 64, 64 },
            kNativeGpuCapability, productPreview, error)
        && foilExportRenderer.renderExport (
            diffractionScene, foilExportBinding, {}, { 64, 64 },
            kNativeGpuCapability, productExport, error);
    check (foilRendered
           && foilPreviewRequest.plan != foilExportRequest.plan
           && foilPreviewBinding->productPlan() != foilExportBinding->productPlan()
           && foilPreviewBinding->productDigest() == foilExportBinding->productDigest()
           && foilPreviewBinding->nativeProgram()->programIdentity
                == foilExportBinding->nativeProgram()->programIdentity
           && foilPreviewBinding->productPlan()->material.maskCoverage == 0.72f
           && foilPreviewBackend.preparedMaterial->diffractionPathCount == 3
           && foilExportBackend.preparedMaterial->diffractionMaximumBounceDepth == 1,
           "Trading Card Foil survives isolated preview and export owners with exact product and lighting receipts");

    FakeBackend referencePreviewBackend;
    FakeBackend referenceExportBackend;
    FixtureSceneRenderer referencePreviewRenderer (referencePreviewBackend);
    FixtureSceneRenderer referenceExportRenderer (referenceExportBackend);
    const bool referenceRendered = referencePreviewBinding != nullptr
        && referenceExportBinding != nullptr
        && referencePreviewRenderer.renderPreview (
            diffractionScene, referencePreviewBinding, {}, { 64, 64 },
            kNativeGpuCapability, productPreview, error)
        && referenceExportRenderer.renderExport (
            diffractionScene, referenceExportBinding, {}, { 64, 64 },
            kNativeGpuCapability, productExport, error);
    check (referenceRendered
           && referencePreviewRequest.plan != referenceExportRequest.plan
           && referencePreviewBinding->productDigest()
                == referenceExportBinding->productDigest()
           && referencePreviewBinding->nativeProgram()->programIdentity
                == referenceExportBinding->nativeProgram()->programIdentity
           && referencePreviewBinding->productDigest()
                != foilPreviewBinding->productDigest()
           && referencePreviewBackend.preparedMaterial->diffractionPathCount == 3
           && referenceExportBackend.preparedMaterial->diffractionMaximumBounceDepth == 1,
           "the calibrated reference scene uses the same production preview and export executor");

    FakeBackend rejectedProductBackend;
    FixtureSceneRenderer rejectedProductRenderer (rejectedProductBackend);
    auto staleProduct = foilPreviewRequest;
    staleProduct.productDigest.push_back ('0');
    auto replayedProduct = foilPreviewRequest;
    replayedProduct.productDigest = referencePreviewRequest.productDigest;
    auto mutatedPlan = std::make_shared<diffractionmaterial::ImmutableDiffractionScenePlan> (
        *foilPreviewRequest.plan);
    mutatedPlan->lighting.paths[0].incident.radiance[0] = 0.0f;
    auto mutatedProduct = foilPreviewRequest;
    mutatedProduct.plan = mutatedPlan;
    const auto staleBinding = admitDiffractionProductPlan (
        diffractionScene, staleProduct, 1, 11, 11,
        videohelper::materialprogram::BackendTarget::Metal, error);
    const auto replayedBinding = admitDiffractionProductPlan (
        diffractionScene, replayedProduct, 1, 11, 11,
        videohelper::materialprogram::BackendTarget::Metal, error);
    const auto mutatedBinding = admitDiffractionProductPlan (
        diffractionScene, mutatedProduct, 1, 11, 11,
        videohelper::materialprogram::BackendTarget::Metal, error);
    bool rejectedRouteRendered = false;
    for (const auto* rejected : { &staleBinding, &replayedBinding, &mutatedBinding })
        if (*rejected != nullptr)
            rejectedRouteRendered = rejectedProductRenderer.renderPreview (
                diffractionScene, *rejected, {}, { 64, 64 }, kNativeGpuCapability,
                productPreview, error) || rejectedRouteRendered;
    check (staleBinding == nullptr && replayedBinding == nullptr
           && mutatedBinding == nullptr && ! rejectedRouteRendered
           && rejectedProductBackend.preparations == 0
           && rejectedProductBackend.submissions == 0,
           "stale, replayed, and mutated product plans fail before backend allocation");

    auto changedWeightRequest = diffractionMaterial;
    // Lighting is frozen to the canonical reference plan, so the anti-aliasing
    // property is exercised through the grooved microstructure: every revision
    // input still collides, the admitted digest legitimately differs, and the
    // program receipt must never alias the original.
    changedWeightRequest.material.microstructure.grooveDepthNanometres
        = std::nextafter(
            changedWeightRequest.material.microstructure.grooveDepthNanometres,
            0.0f);
    {
        std::string changedMaterialError;
        if (const auto admittedChanged = diffractionmaterial::admit (
                changedWeightRequest.material, changedMaterialError))
            changedWeightRequest.structuralDigest
                = admittedChanged->structuralDigest();
    }
    auto changedWeightBinding = admitDiffractionMaterialBinding(
        diffractionScene, changedWeightRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    const auto originalProgram = diffractionBinding != nullptr
        ? diffractionBinding->nativeProgram() : nullptr;
    const auto changedProgram = changedWeightBinding != nullptr
        ? changedWeightBinding->nativeProgram() : nullptr;
    check (originalProgram != nullptr && changedProgram != nullptr
           && changedWeightRequest.sceneRevision == diffractionMaterial.sceneRevision
           && changedWeightRequest.structuralRevision
                == diffractionMaterial.structuralRevision
           && changedWeightRequest.evaluationRevision
                == diffractionMaterial.evaluationRevision
           && originalProgram->programIdentity != changedProgram->programIdentity,
           "one microstructure change never aliases the program receipt while every revision input collides");
    if (originalProgram != nullptr && changedProgram != nullptr)
    {
        auto sourceOwner = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*originalProgram);
        const auto preparedSnapshot = arbitgpu::snapshotNativeSurfaceProgram(sourceOwner);
        arbitgpu::NativeFixtureSceneRuntimeInputs modulated;
        modulated.diffractionParameters = {{"grooveDepthNanometres",
            changedWeightRequest.material.microstructure.grooveDepthNanometres}};
        modulated.diffractionSourceProgram = sourceOwner;
        modulated.diffractionEvaluatedProgram = changedProgram;
        std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> resolved;
        check (preparedSnapshot != sourceOwner
               && arbitgpu::resolveDiffractionRuntimeProgram(preparedSnapshot, sourceOwner,
                   modulated, resolved, error) && resolved == changedProgram,
               "frame-local modulation retains its exact source authority across immutable preparation");

        // A mutable caller alias cannot change the prepared layout, even though
        // the original owner still authorizes scalar modulation.
        sourceOwner->baseColorSource = arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture;
        check (arbitgpu::resolveDiffractionRuntimeProgram(preparedSnapshot, sourceOwner,
                   modulated, resolved, error) && resolved == changedProgram
               && arbitgpu::resolveDiffractionRuntimeProgram(preparedSnapshot, sourceOwner,
                   {}, resolved, error) && resolved == preparedSnapshot,
               "caller alias mutation cannot replace the prepared material or its modulation layout");
        auto changedLayout = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*changedProgram);
        changedLayout->baseColorSource = sourceOwner->baseColorSource;
        modulated.diffractionEvaluatedProgram = changedLayout;
        check (!arbitgpu::resolveDiffractionRuntimeProgram(preparedSnapshot, sourceOwner,
                   modulated, resolved, error),
               "a changed Frame layout cannot inherit authority from a mutated caller alias");
        modulated.diffractionEvaluatedProgram = changedProgram;
        modulated.diffractionSourceProgram = std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(*sourceOwner);
        check (!arbitgpu::resolveDiffractionRuntimeProgram(preparedSnapshot, sourceOwner,
                   modulated, resolved, error),
               "a copied material owner cannot replay another preparation's scalar admission");

        auto replayed = *changedProgram;
        replayed.programIdentity = originalProgram->programIdentity;
        const bool metalAccepted = arbitgpu::validNativeFixtureDiffractionProgram(
            replayed, arbitgpu::NativeFixtureMaterialBackend::Metal,
            diffractionScene->objects[0].id);
        replayed.backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
        const bool openGlAccepted = arbitgpu::validNativeFixtureDiffractionProgram(
            replayed, arbitgpu::NativeFixtureMaterialBackend::OpenGl,
            diffractionScene->objects[0].id);
        check (! metalAccepted && ! openGlAccepted,
               "Metal and OpenGL reject a stale lighting receipt before backend allocation");
    }

    auto crossedMaterial = diffractionMaterial;
    crossedMaterial.material
        = diffractionmaterial::makeRealtimeCrossedTwoDimensionalGratingPreset();
    if (const auto admitted = diffractionmaterial::admit (
            crossedMaterial.material, error))
        crossedMaterial.structuralDigest = admitted->structuralDigest();
    auto crossedBinding = admitDiffractionMaterialBinding (
        diffractionScene, crossedMaterial,
        videohelper::materialprogram::BackendTarget::Metal, error);
    RenderedFrame crossedPreview;
    RenderedFrame crossedExport;
    check (crossedBinding != nullptr
           && diffractionRenderer.renderPreview (
               diffractionScene, crossedBinding, {}, { 64, 64 },
               kNativeGpuCapability, crossedPreview, error)
           && diffractionRenderer.renderExport (
               diffractionScene, crossedBinding, {}, { 64, 64 },
               kNativeGpuCapability, crossedExport, error)
           && diffractionBackend.preparations == 2
           && diffractionBackend.preparedMaterial != nullptr
           && diffractionBackend.preparedMaterial->diffractionPaths[0].material.secondaryGeometry.w
                == static_cast<float> (static_cast<unsigned> (
                    diffractionmaterial::GratingLattice::CrossedTwoDimensional))
           && crossedExport.stats.reusedMaterialProgram,
           "preview and export share one exact crossed two-dimensional diffraction program");

    check (diffractionBackend.preparedMaterial != nullptr,
           "the crossed preset prepares before workload estimation");
    if (diffractionBackend.preparedMaterial != nullptr)
    {
        const auto& crossedWorkload = *diffractionBackend.preparedMaterial;
        check (crossedWorkload.diffractionLightingAdmission != nullptr
               && crossedWorkload.diffractionLightingAdmission->description().version == 2,
               "crossed workload uses the physical environment integration contract");
        // Nine reciprocal-order pairs, eight wavelengths and eleven lighting
        // evaluations: one direct plus five environment and five indirect.
        check (arbitgpu::nativeFixtureDimensionsWithinBounds (960, 540)
               && arbitgpu::nativeFixtureDiffractionWorkWithinBudget (
                   crossedWorkload, 960, 540)
               && arbitgpu::nativeFixtureDiffractionLobeEvaluations (
                   crossedWorkload, 960, 540) == 410'572'800ull,
               "physical crossed diffraction admits a bounded 540p workload before dispatch");
        check (! arbitgpu::nativeFixtureDiffractionWorkWithinBudget (
                   crossedWorkload, 1920, 1080)
               && arbitgpu::nativeFixtureDiffractionLobeEvaluations (
                   crossedWorkload, 1920, 1080) == 1'642'291'200ull,
               "physical crossed diffraction accounts for environment integration at 1080p");
        check (arbitgpu::nativeFixtureDimensionsWithinBounds (3840, 2160)
               && ! arbitgpu::nativeFixtureDiffractionWorkWithinBudget (
                   crossedWorkload, 3840, 2160)
               && arbitgpu::nativeFixtureDiffractionLobeEvaluations (
                   crossedWorkload, 3840, 2160) == 6'569'164'800ull,
               "crossed diffraction rejects an unsafe 4K workload before dispatch");

        auto malformedWorkload = crossedWorkload;
        malformedWorkload.diffractionPaths[0].material.control.x
            = std::numeric_limits<float>::quiet_NaN();
        check (! arbitgpu::nativeFixtureDiffractionWorkWithinBudget (
                   malformedWorkload, 64, 64)
               && arbitgpu::nativeFixtureDiffractionLobeEvaluations (
                   malformedWorkload, 64, 64)
                    == std::numeric_limits<std::uint64_t>::max(),
               "diffraction workload estimation rejects malformed order fields before conversion");
        malformedWorkload = crossedWorkload;
        malformedWorkload.diffractionPaths[0].material.secondaryGeometry.w = 99.0f;
        check (! arbitgpu::nativeFixtureDiffractionWorkWithinBudget (
                   malformedWorkload, 64, 64),
               "diffraction workload estimation rejects unknown lattice tokens");
    }
    check (! arbitgpu::nativeFixtureDimensionsWithinBounds (0, 1080)
           && ! arbitgpu::nativeFixtureDimensionsWithinBounds (4097, 1),
           "fixture dimensions reject empty and oversized targets before allocation");

    auto zeroSlopeMaterial = diffractionMaterial;
    zeroSlopeMaterial.material.roughness.rmsSlope = 0.0f;
    if (const auto admitted = diffractionmaterial::admit (
            zeroSlopeMaterial.material, error))
        zeroSlopeMaterial.structuralDigest = admitted->structuralDigest();
    check (admitDiffractionMaterialBinding (
               diffractionScene, zeroSlopeMaterial,
               videohelper::materialprogram::BackendTarget::Metal, error) == nullptr
           && error
                == "native diffraction surface rendering requires RMS slope at least 0.0001 for a finite angular lobe",
           "the native surface slice rejects an unresolved delta lobe");

    auto coloredLightValue = diffractionSceneValue;
    coloredLightValue.lights[0].color = { 1.0f, 0.9f, 0.8f };
    auto coloredLightScene = std::make_shared<const Visual3DScene> (coloredLightValue);
    check (admitDiffractionMaterialBinding (
               coloredLightScene, diffractionMaterial,
               videohelper::materialprogram::BackendTarget::Metal, error) == nullptr
           && error.find("one white Directional Light") != std::string::npos,
           "physical direct lighting rejects an undeclared RGB-to-spectrum interpretation");
    auto environmentOnly = diffractionMaterial;
    environmentOnly.version = diffractionmaterialbinding::kEnvironmentWireVersion;
    environmentOnly.lighting.paths[0].incident.radiance.fill(0.0f);
    check (admitDiffractionMaterialBinding (
               coloredLightScene, environmentOnly,
               videohelper::materialprogram::BackendTarget::Metal, error) != nullptr,
           "environment-only spectral lighting is independent of unused RGB scene-light metadata");
    auto legacyEnvironment = environmentOnly;
    legacyEnvironment.version = diffractionmaterialbinding::kWireVersion;
    check (admitDiffractionMaterialBinding (
               coloredLightScene, legacyEnvironment,
               videohelper::materialprogram::BackendTarget::Metal, error) == nullptr
           && error == "imported diffraction lighting does not match the version-defined canonical plan",
           "legacy diffraction transport cannot override its fixed reference lighting");

    auto degenerateUvValue = diffractionSceneValue;
    degenerateUvValue.vertices[2].uv = degenerateUvValue.vertices[0].uv;
    auto degenerateUvScene = std::make_shared<const Visual3DScene> (degenerateUvValue);
    check (admitDiffractionMaterialBinding (
               degenerateUvScene, diffractionMaterial,
               videohelper::materialprogram::BackendTarget::Metal, error) == nullptr
           && error
                == "native diffraction surface rendering requires non-degenerate UV derivatives on every triangle",
           "the native surface slice rejects geometry without a UV tangent frame");

    auto wrongObjectRequest = constantRequest;
    wrongObjectRequest.binding.object.value += 1;
    check (admitSurfaceMaterialBinding (
               materialScene, wrongObjectRequest,
               videohelper::materialprogram::BackendTarget::Metal, error) == nullptr
           && error == "surface material request must bind the exact imported scene object"
           && materialBackend.preparations == 1,
           "an inexact object binding is rejected before native preparation");
    auto openGlBinding = admitSurfaceMaterialBinding (
        materialScene, constantRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, error);
    FakeBackend openGlBackend;
    openGlBackend.backendInfo.backend = "opengl";
    FixtureSceneRenderer openGlRenderer (openGlBackend);
    RenderedFrame openGlPreview;
    RenderedFrame openGlExport;
    check (openGlBinding != nullptr
           && openGlRenderer.renderPreview (
               materialScene, openGlBinding, { 64, 64 }, kNativeGpuCapability,
               openGlPreview, error)
           && openGlRenderer.renderExport (
               materialScene, openGlBinding, { 64, 64 }, kNativeGpuCapability,
               openGlExport, error)
           && openGlBackend.preparations == 1
           && openGlBackend.preparedMaterial != nullptr
           && openGlBackend.preparedMaterial->backend
                == arbitgpu::NativeFixtureMaterialBackend::OpenGl
           && openGlPreview.nativeFrame->backend() == "opengl"
           && openGlExport.nativeFrame->backend() == "opengl"
           && openGlExport.stats.reusedMaterialProgram,
           "OpenGL preview and export share one backend-matched immutable material owner");
    check (! openGlRenderer.renderPreview (
               materialScene, materialBinding, { 64, 64 }, kNativeGpuCapability,
               openGlPreview, error)
           && error == "native material program does not match the physical fixture backend"
           && openGlBackend.preparations == 1,
           "a Metal material cannot enter an OpenGL fixture backend");
    check (! materialRenderer.renderPreview (
               materialScene, openGlBinding, { 64, 64 }, kNativeGpuCapability,
               materialPreview, error)
           && error == "native material program does not match the physical fixture backend"
           && materialBackend.preparations == 1,
           "an OpenGL material cannot enter a Metal fixture backend");

    auto equivalentBinding = admitSurfaceMaterialBinding (
        materialScene, constantRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    check (equivalentBinding != nullptr && equivalentBinding != materialBinding
           && materialRenderer.renderPreview (
               materialScene, equivalentBinding, { 64, 64 }, kNativeGpuCapability,
               materialPreview, error)
           && materialBackend.preparations == 2
           && materialPreview.stats.materialProgramUploadCount == 1
           && ! materialPreview.stats.reusedMaterialProgram,
           "equal material values from a different immutable admission replace the cached owner");

    auto unsupportedDescription = constantSurfaceProgram (
        materialScene->materials[0].id.value);
    Operation add;
    add.id = static_cast<ValueId> (unsupportedDescription.operations.size() + 1);
    add.kind = OperationKind::Add;
    add.resultType = ValueType::Scalar;
    add.inputCount = 2;
    add.inputs[0] = unsupportedDescription.output (OutputSemantic::Metallic);
    add.inputs[1] = unsupportedDescription.output (OutputSemantic::Roughness);
    unsupportedDescription.operations.push_back (add);
    unsupportedDescription.outputs[static_cast<std::size_t> (OutputSemantic::Metallic)]
        = add.id;
    auto unsupportedRequest = materialRequest (materialScene, unsupportedDescription);
    check (admitSurfaceMaterialBinding (
               materialScene, unsupportedRequest,
               videohelper::materialprogram::BackendTarget::Metal, error) == nullptr
           && error == "native surface material PBR subset requires constant metallic, roughness, emission, opacity, normal, transmission, IOR, clearcoat, and materialId outputs"
           && materialBackend.preparations == 2,
           "an admitted operation outside the execution checkpoint fails closed");

    auto changedDescription = constantSurfaceProgram (
        materialScene->materials[0].id.value, 0.4f);
    auto changedRequest = materialRequest (materialScene, changedDescription, 2);
    auto changedBinding = admitSurfaceMaterialBinding (
        materialScene, changedRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    check (changedBinding != nullptr
           && materialRenderer.renderPreview (
               materialScene, changedBinding, { 64, 64 }, kNativeGpuCapability,
               materialPreview, error)
           && materialBackend.preparations == 3
           && materialPreview.stats.materialProgramUploadCount == 1,
           "a changed immutable program replaces the exact cached native owner once");

    const auto timeDescription = timeMixSurfaceProgram (
        materialScene->materials[0].id.value);
    const auto admittedTimeMaterial = admit (timeDescription, error);
    auto timeRequest = materialRequest (materialScene, timeDescription, 3);
    auto timeBinding = admitSurfaceMaterialBinding (
        materialScene, timeRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    MaterialEvaluationInputs timeZero {};
    MaterialEvaluationInputs timeMiddle {};
    timeMiddle.timeSeconds = 0.25f;
    MaterialEvaluationInputs timeEnd {};
    timeEnd.timeSeconds = 1.0f;
    const auto referenceZero = admittedTimeMaterial
        ? evaluateReference (*admittedTimeMaterial, timeZero, error) : std::nullopt;
    const auto referenceMiddle = admittedTimeMaterial
        ? evaluateReference (*admittedTimeMaterial, timeMiddle, error) : std::nullopt;
    const auto referenceEnd = admittedTimeMaterial
        ? evaluateReference (*admittedTimeMaterial, timeEnd, error) : std::nullopt;
    check (timeBinding != nullptr && referenceZero && referenceMiddle && referenceEnd
           && near (referenceZero->baseColorMetallic[0], 0.1f)
           && near (referenceZero->baseColorMetallic[1], 0.2f)
           && near (referenceZero->baseColorMetallic[2], 0.3f)
           && near (referenceMiddle->baseColorMetallic[0], 0.3f)
           && near (referenceMiddle->baseColorMetallic[1], 0.325f)
           && near (referenceMiddle->baseColorMetallic[2], 0.35f)
           && near (referenceEnd->baseColorMetallic[0], 0.9f)
           && near (referenceEnd->baseColorMetallic[1], 0.7f)
           && near (referenceEnd->baseColorMetallic[2], 0.5f),
           "the independent oracle resolves the admitted time mix at start, intermediate, and end");
    if (timeBinding == nullptr || ! referenceZero || ! referenceMiddle || ! referenceEnd)
        return 1;

    FakeBackend timeBackend;
    FixtureSceneRenderer timeRenderer (timeBackend);
    RenderedFrame timePreview;
    RenderedFrame timeExport;
    check (timeRenderer.renderPreview (
               materialScene, timeBinding, timeZero, { 64, 64 },
               kNativeGpuCapability, timePreview, error)
           && timeBackend.preparedMaterial != nullptr
           && timeBackend.preparedMaterial->baseColorSource
                == arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix
           && near (timeBackend.preparedMaterial->parameters.baseColorMetallic[0], 0.1f)
           && near (timeBackend.preparedMaterial->parameters.baseColorMetallic[1], 0.2f)
           && near (timeBackend.preparedMaterial->parameters.baseColorMetallic[2], 0.3f)
           && near (timeBackend.preparedMaterial->timeMixEndColor[0], 0.9f)
           && near (timeBackend.preparedMaterial->timeMixEndColor[1], 0.7f)
           && near (timeBackend.preparedMaterial->timeMixEndColor[2], 0.5f),
           "native preparation retains both linear endpoints without evaluating time on the CPU");
    if (timeBackend.preparedMaterial == nullptr)
        return 1;
    const auto timeProgramIdentity = timeBackend.preparedMaterial->programIdentity;
    check (timeRenderer.renderExport (
               materialScene, timeBinding, timeEnd, { 64, 64 },
               kNativeGpuCapability, timeExport, error)
           && timeRenderer.renderPreview (
               materialScene, timeBinding, timeMiddle, { 64, 64 },
               kNativeGpuCapability, timePreview, error)
           && timeBackend.submittedRuntimeInputs.size() == 3
           && timeBackend.submittedRuntimeInputs[0].timeSeconds == 0.0f
           && timeBackend.submittedRuntimeInputs[1].timeSeconds == 1.0f
           && timeBackend.submittedRuntimeInputs[2].timeSeconds == 0.25f,
           "preview and export submit the caller's exact runtime time for every draw");
    check (timeBackend.preparations == 1
           && timeBackend.preparedMaterial->programIdentity == timeProgramIdentity
           && timePreview.stats.materialProgramUploadCount == 0
           && timePreview.stats.reusedMaterialProgram
           && timeExport.stats.materialProgramUploadCount == 0
           && timeExport.stats.reusedMaterialProgram,
           "time-only draws share one immutable material and never change compilation identity");

    const auto retainedTimeHandle = timePreview.nativeFrame->colorImageHandle();
    auto invalidTime = timeMiddle;
    invalidTime.timeSeconds = std::numeric_limits<float>::quiet_NaN();
    check (! timeRenderer.renderPreview (
               materialScene, timeBinding, invalidTime, { 64, 64 },
               kNativeGpuCapability, timePreview, error)
           && error == "native surface material time input is non-finite or out of bounds"
           && timeBackend.preparations == 1 && timeBackend.submissions == 3
           && timePreview.nativeFrame->colorImageHandle() == retainedTimeHandle,
           "a non-finite runtime time fails before native preparation or draw and preserves last-good output");
    invalidTime.timeSeconds = surfacematerial::kMaximumEvaluationMagnitude * 2.0f;
    check (! timeRenderer.renderExport (
               materialScene, timeBinding, invalidTime, { 64, 64 },
               kNativeGpuCapability, timeExport, error)
           && error == "native surface material time input is non-finite or out of bounds"
           && timeBackend.preparations == 1 && timeBackend.submissions == 3,
           "an out-of-bounds runtime time fails closed before native preparation or draw");

    auto texturedRequest = materialRequest (
        scene, importedTextureSurfaceProgram (scene->materials[0].id.value));
    auto authoredFrameRequest = texturedRequest;
    authoredFrameRequest.version = surfacematerialbinding::kGraphFrameWireVersion;
    authoredFrameRequest.binding.textures[0].source = surfacematerialbinding::TextureSourceKind::GraphFrame;
    authoredFrameRequest.binding.textures[0].graphFrame
        = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint { 24, 0 };
    for (const auto target : { videohelper::materialprogram::BackendTarget::OpenGl,
                               videohelper::materialprogram::BackendTarget::Metal })
    {
        const auto binding = admitSurfaceMaterialBinding(scene, authoredFrameRequest, target, error);
        FakeBackend backend;
        FixtureSceneRenderer renderer(backend);
        RenderedFrame frame;
        check(binding != nullptr,
              "both native backends admit an exact authored Frame endpoint without requiring an imported texture");
        check(binding && !renderer.renderPreview(scene, binding, { 64, 64 },
                  kNativeGpuCapability, frame, error)
                  && error == "Native material Frame draw requires its exact owned texture"
                  && backend.preparations == 0 && backend.submissions == 0,
              "a Frame material without a resolved lease fails before GPU preparation and never samples the imported texture");
    }
    auto texturedBinding = admitSurfaceMaterialBinding (
        scene, texturedRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    check (texturedBinding != nullptr
           && texturedBinding->sceneSnapshot() == scene,
           "the imported texture admission retains the exact immutable decoded scene owner");
    texturedRequest.sceneSnapshot.reset();
    check (texturedBinding != nullptr && texturedBinding->sceneSnapshot() == scene,
           "the admitted material does not borrow scene ownership from the mutable request");

    FakeBackend texturedBackend;
    FixtureSceneRenderer texturedRenderer (texturedBackend);
    RenderedFrame texturedPreview;
    check (texturedBinding != nullptr
           && texturedRenderer.renderPreview (
               scene, texturedBinding, { 64, 64 }, kNativeGpuCapability,
               texturedPreview, error)
           && texturedBackend.preparedMaterial != nullptr,
           "the imported embedded texture reaches native GPU preparation without a CPU fallback");
    const auto& texturedProgram = texturedBackend.preparedMaterial;
    const auto* importedTexture = texturedProgram != nullptr
        && texturedProgram->importedBaseColorTexture
        ? &*texturedProgram->importedBaseColorTexture : nullptr;
    check (texturedProgram != nullptr
           && texturedProgram->baseColorSource
                == arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture
           && importedTexture != nullptr
           && importedTexture->width == scene->textures[0].width
           && importedTexture->height == scene->textures[0].height
           && importedTexture->texelCount == scene->textureTexelCount
           && importedTexture->texels[0].red == scene->textureTexels[0].red
           && importedTexture->texels[0].green == scene->textureTexels[0].green
           && importedTexture->texels[3].blue == scene->textureTexels[3].blue
           && texturedProgram->parameters.baseColorMetallic[0] == 1.0f
           && texturedProgram->parameters.baseColorMetallic[1] == 1.0f
           && texturedProgram->parameters.baseColorMetallic[2] == 1.0f
           && texturedProgram->parameters.baseColorMetallic[3] == 0.6f
           && texturedProgram->parameters.emissionRoughness[3] == 0.25f,
           "native preparation value-owns exact sRGB texels plus constant metallic and roughness");

    auto tintedRequest = materialRequest (
        scene, tintedImportedTextureSurfaceProgram (scene->materials[0].id.value), 2);
    auto tintedBinding = admitSurfaceMaterialBinding (
        scene, tintedRequest,
        videohelper::materialprogram::BackendTarget::Metal, error);
    FakeBackend tintedBackend;
    FixtureSceneRenderer tintedRenderer (tintedBackend);
    RenderedFrame tintedPreview;
    RenderedFrame tintedExport;
    check (tintedBinding != nullptr
           && tintedRenderer.renderPreview (
               scene, tintedBinding, { 64, 64 }, kNativeGpuCapability,
               tintedPreview, error)
           && tintedBackend.preparedMaterial != nullptr
           && tintedBackend.preparedMaterial->baseColorSource
                == arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture
           && tintedBackend.preparedMaterial->importedBaseColorTexture.has_value()
           && near (tintedBackend.preparedMaterial->parameters.baseColorMetallic[0], 0.25f)
           && near (tintedBackend.preparedMaterial->parameters.baseColorMetallic[1], 0.5f)
           && near (tintedBackend.preparedMaterial->parameters.baseColorMetallic[2], 0.75f),
           "an authored linear tint multiplies the imported sRGB texture in the native material program");
    auto excessiveTintRequest = materialRequest (
        scene, tintedImportedTextureSurfaceProgram (
            scene->materials[0].id.value, { 1.25f, 0.5f, 0.75f, 0.0f }), 3);
    check (admitSurfaceMaterialBinding (
               scene, excessiveTintRequest,
               videohelper::materialprogram::BackendTarget::Metal, error) == nullptr
           && error == "native surface material texture tint must be finite and within [0, 1]",
           "texture tint admission rejects finite values outside the linear unit interval");
    check (tintedRenderer.renderExport (
               scene, tintedBinding, { 64, 64 }, kNativeGpuCapability,
               tintedExport, error)
           && tintedBackend.preparations == 1
           && tintedBackend.submissions == 2
           && tintedExport.stats.materialProgramUploadCount == 0
           && tintedExport.stats.reusedMaterialProgram,
           "tinted texture preview and export share one immutable native GPU material");

    const auto clonedScene = std::make_shared<const Visual3DScene> (*scene);
    check (! texturedRenderer.renderPreview (
               clonedScene, texturedBinding, { 64, 64 }, kNativeGpuCapability,
               texturedPreview, error)
           && error == "native surface material admission does not match the exact scene snapshot"
           && texturedBackend.preparations == 1,
           "equal scene values cannot replace the exact immutable texture owner");

    const auto successfulPreviewHandle = preview.nativeFrame->colorImageHandle();
    check (! renderer.renderPreview (scene, { 96, 96 }, "source-decode", preview, error)
           && error == "native fixture rendering requires backendCapability native-gpu"
           && backend.submissions == 2
           && preview.nativeFrame->colorImageHandle() == successfulPreviewHandle,
           "non-native capability fails before dispatch without replacing the last frame");

    auto malformed = std::make_shared<Visual3DScene> (*scene);
    malformed->indices[0] = static_cast<std::uint32_t> (malformed->objects[0].vertexCount);
    check (! renderer.renderPreview (malformed, { 96, 96 }, kNativeGpuCapability,
                                     preview, error)
           && error.find ("Visual3DScene validation failed") == 0
           && backend.submissions == 2
           && preview.nativeFrame->colorImageHandle() == successfulPreviewHandle,
           "invalid scene data fails closed before backend allocation");

    auto expanded = std::make_shared<Visual3DScene> (*scene);
    expanded->lights[0].kind = SceneLightKind::Point;
    expanded->objectCount = 2;
    expanded->objects[1] = expanded->objects[0];
    expanded->objects[1].id = SceneObjectId { 2 };
    expanded->objects[1].transform.translation.x += 1.0f;
    FakeBackend expandedBackend;
    FixtureSceneRenderer expandedRenderer (expandedBackend);
    RenderedFrame expandedFrame;
    check (expandedRenderer.renderPreview (
               expanded, { 96, 96 }, kNativeGpuCapability, expandedFrame, error)
           && expandedBackend.submissions == 1
           && expandedFrame.stats.drawCount == 2,
           "bounded multi-object scenes and punctual lights reach native preparation");

    check (! renderer.renderPreview (scene, { FixtureSceneRenderer::kMaxExtent + 1, 1 },
                                     kNativeGpuCapability, preview, error)
           && error == "native fixture render dimensions exceed the bounded extent"
           && backend.submissions == 2,
           "render dimensions are bounded before backend allocation");

    backend.rejectSubmission = true;
    check (! renderer.renderExport (scene, { 96, 96 }, kNativeGpuCapability,
                                    exported, error)
           && error == "fake native submission rejected"
           && exported.nativeFrame != nullptr,
           "a backend draw failure preserves the previous export frame");
    backend.rejectSubmission = false;

    backend.backendInfo.available = false;
    backend.backendInfo.error = "fake native backend unavailable";
    check (! renderer.renderPreview (scene, { 96, 96 }, kNativeGpuCapability,
                                     preview, error)
           && error == "fake native backend unavailable"
           && backend.submissions == 3,
           "an unavailable backend fails before native submission");

    {
        FakeBackend ownershipBackend;
        FixtureSceneRenderer ownershipRenderer (ownershipBackend);
        RenderedFrame frame;
        check (ownershipRenderer.renderPreview (scene, { 32, 32 }, kNativeGpuCapability,
                                                frame, error),
               "ownership fixture creates its first native frame");
        check (ownershipRenderer.renderPreview (scene, { 32, 32 }, kNativeGpuCapability,
                                                frame, error)
               && ownershipBackend.deletedFrames == 1,
               "replacing a rendered frame releases its exact native owner once");
        frame = {};
        check (ownershipBackend.deletedFrames == 2,
               "dropping the final frame releases the remaining native owner once");
    }

    FixtureSceneRenderer stubRenderer (arbitgpu::nativeFixtureSceneBackend());
    RenderedFrame stubOutput;
    check (! stubRenderer.renderPreview (scene, { 96, 96 }, kNativeGpuCapability,
                                         stubOutput, error)
           && error == "native GPU backend not compiled in"
           && stubOutput.nativeFrame == nullptr,
           "the compiled stub has no CPU rendering fallback");

    preview = {};
    exported = {};
    check (backend.deletedFrames == 2,
           "successful preview and export frames each release one native owner");

    std::printf ("fixture-scene-renderer: %d/%d checks passed; submissions=%d deletions=%d\n",
                 checks - failures, checks, backend.submissions, backend.deletedFrames);
    return failures == 0 ? 0 : 1;
}
