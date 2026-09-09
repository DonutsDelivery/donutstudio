#include "glb_native_render_seam.h"
#include "material_program_compiler.h"
#include "surface_material_binding_admission.h"

#include <algorithm>
#include <string>
#include <utility>

namespace videohelper::gltf
{
namespace
{
using HarmonicMIDI::grid::Visual3DScene;

GlbAdmissionOptions nativeRenderOptions (
    std::optional<std::size_t> sceneIndex)
{
    GlbAdmissionOptions options;
    auto& limits = options.limits;
    limits.maxContainerBytes = GlbNativeRenderSeam::kMaxSourceBytes;
    limits.maxJsonBytes = GlbNativeRenderSeam::kMaxJsonBytes;
    limits.maxBinBytes = GlbNativeRenderSeam::kMaxBinBytes;
    limits.maxScenes = 64;
    limits.maxNodes = Visual3DScene::kMaxObjects
                    + Visual3DScene::kMaxCameras
                    + Visual3DScene::kMaxLights;
    limits.maxNodeDepth = 64;
    limits.maxMeshes = Visual3DScene::kMaxObjects;
    limits.maxPrimitives = Visual3DScene::kMaxObjects;
    limits.maxMaterials = Visual3DScene::kMaxMaterials;
    limits.maxTextures = Visual3DScene::kMaxTextures;
    limits.maxImages = Visual3DScene::kMaxTextures;
    limits.maxSamplers = Visual3DScene::kMaxTextures;
    limits.maxCameras = Visual3DScene::kMaxCameras;
    limits.maxLights = Visual3DScene::kMaxLights;
    limits.maxDecodedVertices = Visual3DScene::kMaxVertices;
    limits.maxDecodedIndices = Visual3DScene::kMaxIndices;
    limits.maxDecodedBytes = Visual3DScene::kMaxVertices
                           * sizeof (HarmonicMIDI::grid::SceneVertex)
                           + Visual3DScene::kMaxIndices * sizeof (std::uint32_t)
                           + Visual3DScene::kMaxTextureTexels
                               * sizeof (HarmonicMIDI::grid::SceneTexelRgba8);
    limits.maxEmbeddedImageBytes = Visual3DScene::kMaxTextureTexels * 4u;
    options.admitAnimations = true;
    options.admitSkins = true;
    options.supportedRequiredExtensions = {"KHR_lights_punctual"};
    options.sceneIndex = sceneIndex;
    return options;
}

bool fail (std::string& error, const char* stage, std::string diagnostic)
{
    error = stage;
    error += ": ";
    error += diagnostic.empty() ? "request rejected without a diagnostic"
                                : std::move (diagnostic);
    return false;
}
} // namespace

GlbNativeRenderSeam::GlbNativeRenderSeam (
    arbitgpu::NativeFixtureSceneBackend& backend) noexcept
    : backend_ (backend), renderer_ (backend)
{
}

bool GlbNativeRenderSeam::admit (
    const std::uint8_t* bytes,
    std::size_t size,
    std::optional<std::size_t> sceneIndex,
    GlbNativeSceneAdmission& output,
    std::string& error) const
{
    if (bytes == nullptr || size == 0)
        return fail (error, "GLB decode rejected", "source bytes are empty");
    if (size > kMaxSourceBytes)
        return fail (error, "GLB decode rejected", "source exceeds the 16 MiB native-render bound");

    std::string diagnostic;
    auto decoded = decodeAnimatedGlbBaseScene (
        bytes, size, nativeRenderOptions (sceneIndex), diagnostic);
    if (! decoded)
        return fail (error, "GLB decode rejected", std::move (diagnostic));

    auto scene = adaptStaticGlbToVisual3DScene (*decoded, diagnostic);
    if (! scene)
        return fail (error, "GLB scene adaptation rejected", std::move (diagnostic));

    GlbNativeSceneAdmission admitted;
    admitted.scene = std::make_shared<const Visual3DScene> (std::move (*scene));
    admitted.metadata = decoded->metadata;
    admitted.selectedScene = decoded->selectedScene;
    admitted.sourceBytes = size;
    admitted.decodedVertexCount = decoded->decodedVertexCount;
    admitted.decodedIndexCount = decoded->decodedIndexCount;
    admitted.decodedBytes = decoded->decodedBytes;
    output = std::move (admitted);
    error.clear();
    return true;
}

bool GlbNativeRenderSeam::renderPreview (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return renderPreview(admission, dimensions, {}, output, error);
}

bool GlbNativeRenderSeam::renderPreview (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return render (admission, dimensions,
                   videorender::fixture3d::RenderUse::Preview,
                   sceneInputs, output, error);
}

bool GlbNativeRenderSeam::renderExport (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return renderExport(admission, dimensions, {}, output, error);
}

bool GlbNativeRenderSeam::renderExport (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    return render (admission, dimensions,
                   videorender::fixture3d::RenderUse::Export,
                   sceneInputs, output, error);
}

bool GlbNativeRenderSeam::publishSurfaceMaterial (
    const GlbNativeSceneAdmission& admission,
    const surfacematerialbinding::ImportedSceneMaterialRequest& request,
    std::string& error)
{
    auto reject = [&] (std::string diagnostic)
    {
        if (request.programRevision != 0
            && request.programRevision == highestMaterialRevisionSeen_
            && request.programRevision > lastGoodMaterialRevision_)
            rejectedMaterialRevision_ = request.programRevision;
        return fail (error, "native GLB material publication rejected",
                     std::move (diagnostic));
    };
    if (request.programRevision != 0
        && request.programRevision < highestMaterialRevisionSeen_)
        return reject ("surface material request revision is stale");
    if (request.programRevision > highestMaterialRevisionSeen_)
        highestMaterialRevisionSeen_ = request.programRevision;
    if (request.programRevision != 0
        && request.programRevision == rejectedMaterialRevision_)
        return reject ("surface material request revision was already rejected");
    if (! admission.valid())
        return reject ("scene admission is empty");

    using videohelper::materialprogram::BackendTarget;
    const auto backendName = backend_.info().backend;
    const auto target = backendName == "metal" ? BackendTarget::Metal
                      : backendName == "opengl" ? BackendTarget::OpenGl
                                                 : BackendTarget::Invalid;
    if (target == BackendTarget::Invalid)
        return reject ("native fixture backend has no admitted material compiler target");

    std::string diagnostic;
    auto candidate = videorender::fixture3d::admitSurfaceMaterialBinding (
        admission.scene, request, target, diagnostic);
    if (candidate == nullptr)
        return reject (std::move (diagnostic));

    const auto materialScene = materialScene_.lock();
    if (request.programRevision == lastGoodMaterialRevision_)
    {
        if (materialScene == admission.scene && lastGoodMaterial_ != nullptr
            && lastGoodDiffractionMaterial_ == nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && lastGoodMaterial_->evaluationRevision() == candidate->evaluationRevision()
            && lastGoodMaterial_->bindingDigest() == candidate->bindingDigest())
        {
            error.clear();
            return true;
        }
        const bool nextEvaluation = materialScene == admission.scene
            && lastGoodMaterial_ != nullptr
            && lastGoodMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodMaterial_->structuralRevision() == candidate->structuralRevision()
            && candidate->evaluationRevision() > lastGoodMaterial_->evaluationRevision();
        if (! nextEvaluation)
            return reject ("surface material revision already names a different immutable request");
    }
    if (lastGoodMaterial_ != nullptr && materialScene == admission.scene
        && lastGoodMaterial_->sceneRevision() != candidate->sceneRevision())
        return reject ("surface material request scene revision does not match the immutable admission");

    materialScene_ = admission.scene;
    lastGoodMaterial_ = std::move (candidate);
    lastGoodDiffractionMaterial_.reset();
    lastGoodMaterialRevision_ = request.programRevision;
    rejectedMaterialRevision_ = 0;
    error.clear();
    return true;
}

void GlbNativeRenderSeam::clearSurfaceMaterial (
    const GlbNativeSceneAdmission& admission) noexcept
{
    if (materialScene_.lock() != admission.scene || lastGoodMaterial_ == nullptr)
        return;
    lastGoodMaterial_.reset();
    if (lastGoodDiffractionMaterial_ == nullptr)
    {
        materialScene_.reset();
        highestMaterialRevisionSeen_ = 0;
        lastGoodMaterialRevision_ = 0;
        rejectedMaterialRevision_ = 0;
    }
}

bool GlbNativeRenderSeam::publishDiffractionMaterial (
    const GlbNativeSceneAdmission& admission,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
    std::string& error)
{
    auto reject = [&] (std::string diagnostic)
    {
        if (request.materialRevision != 0
            && request.materialRevision == highestMaterialRevisionSeen_
            && request.materialRevision > lastGoodMaterialRevision_)
            rejectedMaterialRevision_ = request.materialRevision;
        return fail (error, "native GLB diffraction material publication rejected",
                     std::move (diagnostic));
    };
    if (request.materialRevision != 0
        && request.materialRevision < highestMaterialRevisionSeen_)
        return reject ("diffraction material request revision is stale");
    if (request.materialRevision > highestMaterialRevisionSeen_)
        highestMaterialRevisionSeen_ = request.materialRevision;
    if (request.materialRevision != 0
        && request.materialRevision == rejectedMaterialRevision_)
        return reject ("diffraction material request revision was already rejected");
    if (! admission.valid())
        return reject ("scene admission is empty");

    using videohelper::materialprogram::BackendTarget;
    const auto backendName = backend_.info().backend;
    const auto target = backendName == "metal" ? BackendTarget::Metal
                      : backendName == "opengl" ? BackendTarget::OpenGl
                                                 : BackendTarget::Invalid;
    if (target == BackendTarget::Invalid)
        return reject ("native fixture backend has no admitted diffraction target");

    std::string diagnostic;
    auto candidate = videorender::fixture3d::admitDiffractionMaterialBinding (
        admission.scene, request, target, diagnostic);
    if (candidate == nullptr)
        return reject (std::move (diagnostic));

    const auto materialScene = materialScene_.lock();
    if (request.materialRevision == lastGoodMaterialRevision_)
    {
        if (materialScene == admission.scene && lastGoodMaterial_ == nullptr
            && lastGoodDiffractionMaterial_ != nullptr
            && lastGoodDiffractionMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodDiffractionMaterial_->structuralRevision()
                == candidate->structuralRevision()
            && lastGoodDiffractionMaterial_->evaluationRevision()
                == candidate->evaluationRevision()
            && lastGoodDiffractionMaterial_->bindingDigest() == candidate->bindingDigest())
        {
            error.clear();
            return true;
        }
        const bool nextEvaluation = materialScene == admission.scene
            && lastGoodMaterial_ == nullptr && lastGoodDiffractionMaterial_ != nullptr
            && lastGoodDiffractionMaterial_->sceneRevision() == candidate->sceneRevision()
            && lastGoodDiffractionMaterial_->structuralRevision()
                == candidate->structuralRevision()
            && candidate->evaluationRevision()
                > lastGoodDiffractionMaterial_->evaluationRevision();
        if (! nextEvaluation)
            return reject ("diffraction material revision already names a different immutable request");
    }
    if (lastGoodDiffractionMaterial_ != nullptr && materialScene == admission.scene
        && lastGoodDiffractionMaterial_->sceneRevision() != candidate->sceneRevision())
        return reject ("diffraction material scene revision does not match the immutable admission");

    materialScene_ = admission.scene;
    lastGoodMaterial_.reset();
    lastGoodDiffractionMaterial_ = std::move (candidate);
    lastGoodMaterialRevision_ = request.materialRevision;
    rejectedMaterialRevision_ = 0;
    error.clear();
    return true;
}

void GlbNativeRenderSeam::clearDiffractionMaterial (
    const GlbNativeSceneAdmission& admission) noexcept
{
    if (materialScene_.lock() != admission.scene
        || lastGoodDiffractionMaterial_ == nullptr)
        return;
    lastGoodDiffractionMaterial_.reset();
    if (lastGoodMaterial_ == nullptr)
    {
        materialScene_.reset();
        highestMaterialRevisionSeen_ = 0;
        lastGoodMaterialRevision_ = 0;
        rejectedMaterialRevision_ = 0;
    }
}

bool GlbNativeRenderSeam::render (
    const GlbNativeSceneAdmission& admission,
    videorender::fixture3d::RenderDimensions dimensions,
    videorender::fixture3d::RenderUse use,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    GlbNativeFrameReceipt& output,
    std::string& error)
{
    if (! admission.valid())
        return fail (error, "native GLB render rejected", "scene admission is empty");
    if (use == videorender::fixture3d::RenderUse::Export
        && rejectedMaterialRevision_ > lastGoodMaterialRevision_)
        return fail (error, "native GLB render rejected",
                     "current material revision is not exportable");

    videorender::fixture3d::RenderedFrame frame;
    std::string diagnostic;
    const auto materialScene = materialScene_.lock();
    const auto material = materialScene == admission.scene
        ? lastGoodMaterial_ : nullptr;
    const auto diffractionMaterial = materialScene == admission.scene
        ? lastGoodDiffractionMaterial_ : nullptr;
    surfacematerial::MaterialEvaluationInputs materialInputs;
    materialInputs.timeSeconds = sceneInputs.timeSeconds;
    const auto rendered = diffractionMaterial != nullptr
        ? (use == videorender::fixture3d::RenderUse::Preview
            ? renderer_.renderPreview(admission.scene, diffractionMaterial, sceneInputs,
                                      dimensions, videorender::fixture3d::kNativeGpuCapability,
                                      frame, diagnostic)
            : renderer_.renderExport(admission.scene, diffractionMaterial, sceneInputs,
                                     dimensions, videorender::fixture3d::kNativeGpuCapability,
                                     frame, diagnostic))
        : (use == videorender::fixture3d::RenderUse::Preview
            ? renderer_.renderPreview(admission.scene, material, materialInputs, sceneInputs,
                                      dimensions, videorender::fixture3d::kNativeGpuCapability,
                                      frame, diagnostic)
            : renderer_.renderExport(admission.scene, material, materialInputs, sceneInputs,
                                     dimensions, videorender::fixture3d::kNativeGpuCapability,
                                     frame, diagnostic));
    if (! rendered)
        return fail (error, "native GLB render rejected", std::move (diagnostic));

    GlbNativeFrameReceipt receipt;
    receipt.rendered = std::move (frame);
    receipt.draws.reserve (admission.scene->objectCount);
    for (std::size_t index = 0; index < admission.scene->objectCount; ++index)
    {
        const auto& object = admission.scene->objects[index];
        receipt.draws.push_back ({ object.id, object.parent, object.material, object.transform,
            object.firstVertex, object.vertexCount, object.firstIndex, object.indexCount });
    }
    receipt.materials.reserve (admission.scene->materialCount);
    for (std::size_t index = 0; index < admission.scene->materialCount; ++index)
    {
        const auto& materialRecord = admission.scene->materials[index];
        receipt.materials.push_back ({ materialRecord.id,
            materialRecord.baseColorTexture, materialRecord.metallicRoughnessTexture,
            materialRecord.normalTexture, materialRecord.occlusionTexture,
            materialRecord.emissiveTexture, materialRecord.baseColor,
            materialRecord.opacity, materialRecord.metallic, materialRecord.roughness,
            materialRecord.emissive, materialRecord.normalScale,
            materialRecord.occlusionStrength, materialRecord.alphaMode,
            materialRecord.alphaCutoff,
            materialRecord.doubleSided });
        const auto addRole = [&] (HarmonicMIDI::grid::SceneTextureId texture,
                                  GlbTextureRole role, GlbTextureColorSpace colorSpace)
        {
            if (texture.isValid())
                receipt.textureRoles.push_back ({ materialRecord.id, texture, role, colorSpace });
        };
        addRole (materialRecord.baseColorTexture, GlbTextureRole::BaseColor,
                 GlbTextureColorSpace::Srgb);
        addRole (materialRecord.metallicRoughnessTexture, GlbTextureRole::MetallicRoughness,
                 GlbTextureColorSpace::Linear);
        addRole (materialRecord.normalTexture, GlbTextureRole::Normal,
                 GlbTextureColorSpace::Linear);
        addRole (materialRecord.occlusionTexture, GlbTextureRole::Occlusion,
                 GlbTextureColorSpace::Linear);
        addRole (materialRecord.emissiveTexture, GlbTextureRole::Emissive,
                 GlbTextureColorSpace::Srgb);
    }
    receipt.textures.reserve (admission.scene->textureCount);
    for (std::size_t index = 0; index < admission.scene->textureCount; ++index)
    {
        const auto& texture = admission.scene->textures[index];
        receipt.textures.push_back ({ texture.id, texture.width, texture.height,
            texture.firstTexel, texture.width * texture.height,
            texture.magFilter, texture.minFilter, texture.wrapS, texture.wrapT });
    }
    receipt.activeCamera = admission.scene->activeCamera;
    receipt.cameras.assign (admission.scene->cameras.begin(),
                            admission.scene->cameras.begin() + admission.scene->cameraCount);
    receipt.lights.assign (admission.scene->lights.begin(),
                           admission.scene->lights.begin() + admission.scene->lightCount);
    if (receipt.rendered.nativeFrame != nullptr)
    {
        const auto descriptor = receipt.rendered.nativeFrame->colorTextureDescriptor();
        receipt.nativeBackend = receipt.rendered.nativeFrame->backend();
        receipt.nativeResourceCacheIdentity
            = receipt.rendered.nativeFrame->nativeResourceCacheIdentity();
        receipt.nativeDeviceOrContextIdentity = descriptor.deviceOrContextIdentity;
        receipt.nativeRendererGeneration = descriptor.rendererGeneration;
        receipt.downstreamHandoffComplete = descriptor.complete()
            && receipt.nativeResourceCacheIdentity != 0;
    }
    output = std::move (receipt);
    error.clear();
    return true;
}

} // namespace videohelper::gltf
