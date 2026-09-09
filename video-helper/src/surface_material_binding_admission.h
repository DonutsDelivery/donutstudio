#pragma once

#include "../../shared/SurfaceMaterialBindingContract.h"
#include "render_snapshot.h"
#include "sha256.h"
#include "surface_material_admission.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace surfacematerialbinding
{
struct SurfaceMaterialProgramReceipt
{
    std::shared_ptr<const surfacematerial::AdmittedSurfaceMaterialIR> program;
    std::uint64_t revision = 0;
};

// The resource owner supplies the exact typed graph output descriptor and an
// opaque receipt identity. Admission does not resolve paths, inspect pixels,
// allocate a texture, or treat node labels as authority.
struct VideoTextureResourceReceipt
{
    VideoResourceIdentity identity {};
    std::uint64_t helperGeneration = 0;
    int sourceClipId = 0;
    std::uint64_t sourceStructuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
    videowire::CompiledVisualPortBinding imageDescriptor;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t byteCount = 0;
};

struct Description
{
    std::uint32_t version = kWireVersion;
    const HarmonicMIDI::grid::Visual3DScene* scene = nullptr;
    std::uint64_t sceneRevision = 0;
    std::uint64_t structuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
    std::uint64_t helperGeneration = 0;
    UnboundObjectFallback fallback = UnboundObjectFallback::ImportedMaterial;
    std::vector<SurfaceMaterialProgramReceipt> programs;
    std::vector<MaterialBinding> bindings;
    std::vector<VideoTextureResourceReceipt> videoResources;
    std::vector<ModulationInputReceipt> modulationInputs;
};

enum class AdmissionFailure : std::uint8_t
{
    None = 0,
    UnsupportedVersion,
    MissingScene,
    InvalidScene,
    MissingRevision,
    UnsupportedFallback,
    ProgramCapacityExceeded,
    BindingCapacityExceeded,
    TextureBindingCapacityExceeded,
    VideoResourceCapacityExceeded,
    ModulationCapacityExceeded,
    MissingProgram,
    InvalidProgramRevision,
    DuplicateProgramIdentity,
    MissingBinding,
    InvalidBindingTarget,
    MissingBindingTarget,
    DuplicateBindingTarget,
    MissingSurfaceMaterial,
    InvalidTextureBinding,
    MissingImportedTexture,
    MissingVideoResource,
    DuplicateVideoResource,
    InvalidVideoResourceRevision,
    InvalidVideoImageDescriptor,
    InvalidVideoResourceSize,
    UnreferencedProgram,
    UnreferencedVideoResource,
    InvalidModulationInput,
    DuplicateModulationInput,
    MissingModulationInput,
    UnreferencedModulationInput,
    AllocationFailed
};

inline constexpr const char* token(AdmissionFailure failure) noexcept
{
    switch (failure)
    {
        case AdmissionFailure::None: return "none";
        case AdmissionFailure::UnsupportedVersion: return "unsupportedVersion";
        case AdmissionFailure::MissingScene: return "missingScene";
        case AdmissionFailure::InvalidScene: return "invalidScene";
        case AdmissionFailure::MissingRevision: return "missingRevision";
        case AdmissionFailure::UnsupportedFallback: return "unsupportedFallback";
        case AdmissionFailure::ProgramCapacityExceeded: return "programCapacityExceeded";
        case AdmissionFailure::BindingCapacityExceeded: return "bindingCapacityExceeded";
        case AdmissionFailure::TextureBindingCapacityExceeded: return "textureBindingCapacityExceeded";
        case AdmissionFailure::VideoResourceCapacityExceeded: return "videoResourceCapacityExceeded";
        case AdmissionFailure::ModulationCapacityExceeded: return "modulationCapacityExceeded";
        case AdmissionFailure::MissingProgram: return "missingProgram";
        case AdmissionFailure::InvalidProgramRevision: return "invalidProgramRevision";
        case AdmissionFailure::DuplicateProgramIdentity: return "duplicateProgramIdentity";
        case AdmissionFailure::MissingBinding: return "missingBinding";
        case AdmissionFailure::InvalidBindingTarget: return "invalidBindingTarget";
        case AdmissionFailure::MissingBindingTarget: return "missingBindingTarget";
        case AdmissionFailure::DuplicateBindingTarget: return "duplicateBindingTarget";
        case AdmissionFailure::MissingSurfaceMaterial: return "missingSurfaceMaterial";
        case AdmissionFailure::InvalidTextureBinding: return "invalidTextureBinding";
        case AdmissionFailure::MissingImportedTexture: return "missingImportedTexture";
        case AdmissionFailure::MissingVideoResource: return "missingVideoResource";
        case AdmissionFailure::DuplicateVideoResource: return "duplicateVideoResource";
        case AdmissionFailure::InvalidVideoResourceRevision: return "invalidVideoResourceRevision";
        case AdmissionFailure::InvalidVideoImageDescriptor: return "invalidVideoImageDescriptor";
        case AdmissionFailure::InvalidVideoResourceSize: return "invalidVideoResourceSize";
        case AdmissionFailure::UnreferencedProgram: return "unreferencedProgram";
        case AdmissionFailure::UnreferencedVideoResource: return "unreferencedVideoResource";
        case AdmissionFailure::InvalidModulationInput: return "invalidModulationInput";
        case AdmissionFailure::DuplicateModulationInput: return "duplicateModulationInput";
        case AdmissionFailure::MissingModulationInput: return "missingModulationInput";
        case AdmissionFailure::UnreferencedModulationInput: return "unreferencedModulationInput";
        case AdmissionFailure::AllocationFailed: return "allocationFailed";
    }
    return "unknown";
}

enum class ResolutionKind : std::uint8_t
{
    ImportedMaterial = 0,
    MaterialSlot = 1,
    ObjectOverride = 2
};

struct MaterialResolution
{
    ResolutionKind kind = ResolutionKind::ImportedMaterial;
    HarmonicMIDI::grid::SceneMaterialId importedMaterial {};
    const MaterialBinding* binding = nullptr;
    const SurfaceMaterialProgramReceipt* program = nullptr;
};

class AdmittedMaterialBindings final
{
public:
    AdmittedMaterialBindings(const AdmittedMaterialBindings&) = default;
    AdmittedMaterialBindings(AdmittedMaterialBindings&&) = default;
    AdmittedMaterialBindings& operator=(const AdmittedMaterialBindings&) = delete;
    AdmittedMaterialBindings& operator=(AdmittedMaterialBindings&&) = delete;

    HarmonicMIDI::grid::Scene3DId scene() const noexcept { return scene_; }
    std::uint64_t sceneRevision() const noexcept { return sceneRevision_; }
    std::uint64_t structuralRevision() const noexcept { return structuralRevision_; }
    std::uint64_t evaluationRevision() const noexcept { return evaluationRevision_; }
    std::uint64_t helperGeneration() const noexcept { return helperGeneration_; }
    UnboundObjectFallback fallback() const noexcept { return fallback_; }
    const std::vector<SurfaceMaterialProgramReceipt>& programs() const noexcept
    {
        return programs_;
    }
    const std::vector<MaterialBinding>& bindings() const noexcept { return bindings_; }
    const std::vector<VideoTextureResourceReceipt>& videoResources() const noexcept
    {
        return videoResources_;
    }
    const std::vector<ModulationInputReceipt>& modulationInputs() const noexcept
    {
        return modulationInputs_;
    }
    const std::string& digest() const noexcept { return digest_; }

    std::optional<MaterialResolution> resolve(
        HarmonicMIDI::grid::SceneObjectId object) const noexcept
    {
        const auto imported = std::lower_bound(
            objects_.begin(), objects_.end(), object.value,
            [] (const ObjectMaterial& candidate, std::uint32_t value)
            { return candidate.object.value < value; });
        if (imported == objects_.end() || imported->object != object)
            return std::nullopt;

        const auto objectOverride = std::lower_bound(
            bindings_.begin(), bindings_.end(), object.value,
            [] (const MaterialBinding& candidate, std::uint32_t value)
            {
                return candidate.targetKind == BindingTargetKind::MaterialSlot
                    || candidate.object.value < value;
            });
        if (objectOverride != bindings_.end()
            && objectOverride->targetKind == BindingTargetKind::ObjectOverride
            && objectOverride->object == object)
            return makeResolution(ResolutionKind::ObjectOverride, imported->material,
                                  &*objectOverride);

        const auto materialSlot = std::lower_bound(
            bindings_.begin(), bindings_.end(), imported->material.value,
            [] (const MaterialBinding& candidate, std::uint32_t value)
            {
                if (candidate.targetKind == BindingTargetKind::ObjectOverride)
                    return false;
                return candidate.materialSlot.value < value;
            });
        if (materialSlot != bindings_.end()
            && materialSlot->targetKind == BindingTargetKind::MaterialSlot
            && materialSlot->materialSlot == imported->material)
            return makeResolution(ResolutionKind::MaterialSlot, imported->material,
                                  &*materialSlot);

        return MaterialResolution { ResolutionKind::ImportedMaterial,
                                    imported->material, nullptr, nullptr };
    }

private:
    struct ObjectMaterial
    {
        HarmonicMIDI::grid::SceneObjectId object {};
        HarmonicMIDI::grid::SceneMaterialId material {};
    };

    friend std::optional<AdmittedMaterialBindings> admit(
        const Description&, AdmissionFailure&, std::string&, const AdmissionLimits&);

    AdmittedMaterialBindings(
        HarmonicMIDI::grid::Scene3DId scene,
        std::uint64_t sceneRevision,
        std::uint64_t structuralRevision,
        std::uint64_t evaluationRevision,
        std::uint64_t helperGeneration,
        UnboundObjectFallback fallback,
        std::vector<ObjectMaterial> objects,
        std::vector<SurfaceMaterialProgramReceipt> programs,
        std::vector<MaterialBinding> bindings,
        std::vector<VideoTextureResourceReceipt> videoResources,
        std::vector<ModulationInputReceipt> modulationInputs,
        std::string digest)
        : scene_(scene),
          sceneRevision_(sceneRevision),
          structuralRevision_(structuralRevision),
          evaluationRevision_(evaluationRevision),
          helperGeneration_(helperGeneration),
          fallback_(fallback),
          objects_(std::move(objects)),
          programs_(std::move(programs)),
          bindings_(std::move(bindings)),
          videoResources_(std::move(videoResources)),
          modulationInputs_(std::move(modulationInputs)),
          digest_(std::move(digest))
    {
    }

    const SurfaceMaterialProgramReceipt* findProgram(
        const MaterialBinding& binding) const noexcept
    {
        const auto found = std::lower_bound(
            programs_.begin(), programs_.end(), binding,
            [] (const SurfaceMaterialProgramReceipt& receipt,
                const MaterialBinding& requested)
            {
                if (receipt.program->structuralDigest() != requested.surfaceMaterialDigest)
                    return receipt.program->structuralDigest()
                        < requested.surfaceMaterialDigest;
                return receipt.revision < requested.surfaceMaterialRevision;
            });
        return found != programs_.end()
                && found->program->structuralDigest() == binding.surfaceMaterialDigest
                && found->revision == binding.surfaceMaterialRevision
            ? &*found : nullptr;
    }

    MaterialResolution makeResolution(
        ResolutionKind kind,
        HarmonicMIDI::grid::SceneMaterialId importedMaterial,
        const MaterialBinding* binding) const noexcept
    {
        return { kind, importedMaterial, binding, findProgram(*binding) };
    }

    const HarmonicMIDI::grid::Scene3DId scene_;
    const std::uint64_t sceneRevision_;
    const std::uint64_t structuralRevision_;
    const std::uint64_t evaluationRevision_;
    const std::uint64_t helperGeneration_;
    const UnboundObjectFallback fallback_;
    const std::vector<ObjectMaterial> objects_;
    const std::vector<SurfaceMaterialProgramReceipt> programs_;
    const std::vector<MaterialBinding> bindings_;
    const std::vector<VideoTextureResourceReceipt> videoResources_;
    const std::vector<ModulationInputReceipt> modulationInputs_;
    const std::string digest_;
};

std::optional<AdmittedMaterialBindings> admit(
    const Description& untrusted,
    AdmissionFailure& failure,
    std::string& diagnostic,
    const AdmissionLimits& requestedLimits = {});

namespace detail
{
inline AdmissionLimits boundedLimits(const AdmissionLimits& requested) noexcept
{
    return {
        std::min(requested.programs, kMaximumPrograms),
        std::min(requested.bindings, kMaximumBindings),
        std::min(requested.textureBindings, kMaximumTextureBindings),
        std::min(requested.videoResources, kMaximumVideoResources),
        std::min(requested.modulationInputs, kMaximumModulationInputs),
        std::min(requested.maximumImageDimension, kMaximumImageDimension),
        std::min(requested.videoResourceBytes, kMaximumVideoResourceBytes)
    };
}

inline bool fail(AdmissionFailure code, AdmissionFailure& failure,
                 std::string& diagnostic)
{
    failure = code;
    diagnostic = token(code);
    return false;
}

inline bool knownColorSpace(const std::string& value) noexcept
{
    return value == "linearSRGB" || value == "sRGB" || value == "displayP3"
        || value == "rec709" || value == "rec2020";
}

inline std::uint64_t imageBytesPerPixel(const std::string& value) noexcept
{
    if (value == "rgba8") return 4;
    if (value == "rgba16f") return 8;
    if (value == "rgba32f") return 16;
    return 0;
}

inline bool identityLess(const VideoResourceIdentity& left,
                         const VideoResourceIdentity& right) noexcept
{
    return std::lexicographical_compare(left.begin(), left.end(),
                                        right.begin(), right.end());
}

inline bool identityEqual(const VideoResourceIdentity& left,
                          const VideoResourceIdentity& right) noexcept
{
    return left == right;
}

inline bool bindingLess(const MaterialBinding& left,
                        const MaterialBinding& right) noexcept
{
    if (left.targetKind != right.targetKind)
        return left.targetKind < right.targetKind;
    return left.targetKind == BindingTargetKind::MaterialSlot
        ? left.materialSlot.value < right.materialSlot.value
        : left.object.value < right.object.value;
}

inline bool sameTarget(const MaterialBinding& left,
                       const MaterialBinding& right) noexcept
{
    if (left.targetKind != right.targetKind)
        return false;
    return left.targetKind == BindingTargetKind::MaterialSlot
        ? left.materialSlot == right.materialSlot : left.object == right.object;
}

inline const HarmonicMIDI::grid::SceneMaterialRecord* findMaterial(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    HarmonicMIDI::grid::SceneMaterialId id) noexcept
{
    return HarmonicMIDI::grid::visual3d_detail::findById(
        scene.materials, scene.materialCount, id);
}

inline const HarmonicMIDI::grid::SceneObjectRecord* findObject(
    const HarmonicMIDI::grid::Visual3DScene& scene,
    HarmonicMIDI::grid::SceneObjectId id) noexcept
{
    return HarmonicMIDI::grid::visual3d_detail::findById(
        scene.objects, scene.objectCount, id);
}

inline const SurfaceMaterialProgramReceipt* findProgram(
    const std::vector<SurfaceMaterialProgramReceipt>& programs,
    const MaterialBinding& binding) noexcept
{
    const auto found = std::lower_bound(
        programs.begin(), programs.end(), binding,
        [] (const SurfaceMaterialProgramReceipt& receipt,
            const MaterialBinding& requested)
        {
            if (receipt.program->structuralDigest() != requested.surfaceMaterialDigest)
                return receipt.program->structuralDigest()
                    < requested.surfaceMaterialDigest;
            return receipt.revision < requested.surfaceMaterialRevision;
        });
    return found != programs.end()
            && found->program->structuralDigest() == binding.surfaceMaterialDigest
            && found->revision == binding.surfaceMaterialRevision
        ? &*found : nullptr;
}

inline const VideoTextureResourceReceipt* findVideoResource(
    const std::vector<VideoTextureResourceReceipt>& resources,
    const VideoResourceIdentity& identity) noexcept
{
    const auto found = std::lower_bound(
        resources.begin(), resources.end(), identity,
        [] (const VideoTextureResourceReceipt& receipt,
            const VideoResourceIdentity& requested)
        { return identityLess(receipt.identity, requested); });
    return found != resources.end() && found->identity == identity ? &*found : nullptr;
}

inline void appendU8(std::vector<std::uint8_t>& bytes, std::uint8_t value)
{
    bytes.push_back(value);
}

inline void appendU16(std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

inline void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value >> 24u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

inline void appendU64(std::vector<std::uint8_t>& bytes, std::uint64_t value)
{
    appendU32(bytes, static_cast<std::uint32_t>(value >> 32u));
    appendU32(bytes, static_cast<std::uint32_t>(value));
}

inline void appendFloat(std::vector<std::uint8_t>& bytes, float value)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t)
                  && std::numeric_limits<float>::is_iec559,
                  "Material binding digest requires IEEE-754 binary32 floats");
    if (value == 0.0f)
        value = 0.0f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(bytes, bits);
}

inline void appendString(std::vector<std::uint8_t>& bytes,
                         const std::string& value)
{
    appendU32(bytes, static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

inline void appendIdentity(std::vector<std::uint8_t>& bytes,
                           const VideoResourceIdentity& identity)
{
    bytes.insert(bytes.end(), identity.begin(), identity.end());
}

inline std::string digest(
    const Description& description,
    const std::vector<SurfaceMaterialProgramReceipt>& programs,
    const std::vector<MaterialBinding>& bindings,
    const std::vector<VideoTextureResourceReceipt>& videoResources,
    const std::vector<ModulationInputReceipt>& modulationInputs)
{
    static constexpr char domain[]
        = "DonutStudio/SurfaceMaterialBindingContract/Snapshot/v1";
    std::vector<std::uint8_t> bytes(std::begin(domain), std::end(domain) - 1);
    appendU32(bytes, description.version);
    appendU32(bytes, description.scene->id.value);
    appendU64(bytes, description.sceneRevision);
    appendU64(bytes, description.structuralRevision);
    appendU64(bytes, description.evaluationRevision);
    appendU64(bytes, description.helperGeneration);
    appendU8(bytes, static_cast<std::uint8_t>(description.fallback));

    appendU32(bytes, static_cast<std::uint32_t>(programs.size()));
    for (const auto& receipt : programs)
    {
        appendString(bytes, receipt.program->structuralDigest());
        appendU64(bytes, receipt.revision);
    }

    appendU32(bytes, static_cast<std::uint32_t>(bindings.size()));
    for (const auto& binding : bindings)
    {
        appendU8(bytes, static_cast<std::uint8_t>(binding.targetKind));
        appendU32(bytes, binding.materialSlot.value);
        appendU32(bytes, binding.object.value);
        appendString(bytes, binding.surfaceMaterialDigest);
        appendU64(bytes, binding.surfaceMaterialRevision);
        appendU32(bytes, static_cast<std::uint32_t>(binding.textures.size()));
        for (const auto& texture : binding.textures)
        {
            appendU16(bytes, texture.slot);
            appendU8(bytes, static_cast<std::uint8_t>(texture.source));
            appendIdentity(bytes, texture.videoResource);
        }
    }

    appendU32(bytes, static_cast<std::uint32_t>(videoResources.size()));
    for (const auto& resource : videoResources)
    {
        appendIdentity(bytes, resource.identity);
        appendU64(bytes, resource.helperGeneration);
        appendU32(bytes, static_cast<std::uint32_t>(resource.sourceClipId));
        appendU64(bytes, resource.sourceStructuralRevision);
        appendU64(bytes, resource.evaluationRevision);
        appendU32(bytes, static_cast<std::uint32_t>(resource.imageDescriptor.nodeId));
        appendU32(bytes, static_cast<std::uint32_t>(resource.imageDescriptor.port));
        appendU32(bytes, static_cast<std::uint32_t>(resource.imageDescriptor.channels));
        appendString(bytes, resource.imageDescriptor.direction);
        appendString(bytes, resource.imageDescriptor.carrier);
        appendString(bytes, resource.imageDescriptor.dataType);
        appendString(bytes, resource.imageDescriptor.pixelFormat);
        appendString(bytes, resource.imageDescriptor.colorSpace);
        appendU32(bytes, resource.width);
        appendU32(bytes, resource.height);
        appendU64(bytes, resource.byteCount);
    }

    appendU32(bytes, static_cast<std::uint32_t>(modulationInputs.size()));
    for (const auto& input : modulationInputs)
    {
        appendU8(bytes, static_cast<std::uint8_t>(input.semantic));
        appendFloat(bytes, input.value);
        appendU64(bytes, input.sourceRevision);
    }

    videohelper::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finishHex();
}
} // namespace detail

inline std::optional<AdmittedMaterialBindings> admit(
    const Description& untrusted,
    AdmissionFailure& failure,
    std::string& diagnostic,
    const AdmissionLimits& requestedLimits)
{
    failure = AdmissionFailure::None;
    diagnostic.clear();
    using namespace HarmonicMIDI::grid;

    if (untrusted.version != kWireVersion)
    {
        detail::fail(AdmissionFailure::UnsupportedVersion, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.scene == nullptr)
    {
        detail::fail(AdmissionFailure::MissingScene, failure, diagnostic);
        return std::nullopt;
    }
    if (!validateVisual3DScene(*untrusted.scene).valid())
    {
        detail::fail(AdmissionFailure::InvalidScene, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.sceneRevision == 0 || untrusted.structuralRevision == 0
        || untrusted.evaluationRevision == 0)
    {
        detail::fail(AdmissionFailure::MissingRevision, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.fallback != UnboundObjectFallback::ImportedMaterial)
    {
        detail::fail(AdmissionFailure::UnsupportedFallback, failure, diagnostic);
        return std::nullopt;
    }

    const auto limits = detail::boundedLimits(requestedLimits);
    if (untrusted.programs.size() > limits.programs)
    {
        detail::fail(AdmissionFailure::ProgramCapacityExceeded, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.bindings.size() > limits.bindings)
    {
        detail::fail(AdmissionFailure::BindingCapacityExceeded, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.videoResources.size() > limits.videoResources)
    {
        detail::fail(AdmissionFailure::VideoResourceCapacityExceeded, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.modulationInputs.size() > limits.modulationInputs)
    {
        detail::fail(AdmissionFailure::ModulationCapacityExceeded, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.programs.empty())
    {
        detail::fail(AdmissionFailure::MissingProgram, failure, diagnostic);
        return std::nullopt;
    }
    if (untrusted.bindings.empty())
    {
        detail::fail(AdmissionFailure::MissingBinding, failure, diagnostic);
        return std::nullopt;
    }

    try
    {
        auto programs = untrusted.programs;
        std::sort(programs.begin(), programs.end(),
                  [] (const auto& left, const auto& right)
                  {
                      if (!left.program || !right.program)
                          return static_cast<bool>(left.program)
                              < static_cast<bool>(right.program);
                      if (left.program->structuralDigest()
                          != right.program->structuralDigest())
                          return left.program->structuralDigest()
                              < right.program->structuralDigest();
                      return left.revision < right.revision;
                  });
        for (std::size_t index = 0; index < programs.size(); ++index)
        {
            if (!programs[index].program)
            {
                detail::fail(AdmissionFailure::MissingProgram, failure, diagnostic);
                return std::nullopt;
            }
            if (programs[index].revision == 0)
            {
                detail::fail(AdmissionFailure::InvalidProgramRevision,
                             failure, diagnostic);
                return std::nullopt;
            }
            if (index != 0
                && programs[index - 1].revision == programs[index].revision
                && programs[index - 1].program->structuralDigest()
                    == programs[index].program->structuralDigest())
            {
                detail::fail(AdmissionFailure::DuplicateProgramIdentity,
                             failure, diagnostic);
                return std::nullopt;
            }
        }

        auto videoResources = untrusted.videoResources;
        std::sort(videoResources.begin(), videoResources.end(),
                  [] (const auto& left, const auto& right)
                  { return detail::identityLess(left.identity, right.identity); });
        std::uint64_t totalVideoBytes = 0;
        for (std::size_t index = 0; index < videoResources.size(); ++index)
        {
            const auto& resource = videoResources[index];
            if (!hasIdentity(resource.identity))
            {
                detail::fail(AdmissionFailure::MissingVideoResource,
                             failure, diagnostic);
                return std::nullopt;
            }
            if (index != 0
                && detail::identityEqual(videoResources[index - 1].identity,
                                         resource.identity))
            {
                detail::fail(AdmissionFailure::DuplicateVideoResource,
                             failure, diagnostic);
                return std::nullopt;
            }
            if (untrusted.helperGeneration == 0
                || resource.helperGeneration != untrusted.helperGeneration
                || resource.sourceClipId <= 0
                || resource.sourceStructuralRevision == 0
                || resource.evaluationRevision != untrusted.evaluationRevision)
            {
                detail::fail(AdmissionFailure::InvalidVideoResourceRevision,
                             failure, diagnostic);
                return std::nullopt;
            }
            const auto& descriptor = resource.imageDescriptor;
            const auto pixelBytes = detail::imageBytesPerPixel(descriptor.pixelFormat);
            if (descriptor.nodeId <= 0 || descriptor.port < 0
                || descriptor.channels != 1 || descriptor.direction != "out"
                || descriptor.carrier != "frame" || descriptor.dataType != "image"
                || pixelBytes == 0 || !detail::knownColorSpace(descriptor.colorSpace))
            {
                detail::fail(AdmissionFailure::InvalidVideoImageDescriptor,
                             failure, diagnostic);
                return std::nullopt;
            }
            if (resource.width == 0 || resource.height == 0
                || resource.width > limits.maximumImageDimension
                || resource.height > limits.maximumImageDimension)
            {
                detail::fail(AdmissionFailure::InvalidVideoResourceSize,
                             failure, diagnostic);
                return std::nullopt;
            }
            const auto pixels = static_cast<std::uint64_t>(resource.width)
                              * static_cast<std::uint64_t>(resource.height);
            if (pixels > std::numeric_limits<std::uint64_t>::max() / pixelBytes
                || resource.byteCount != pixels * pixelBytes
                || resource.byteCount > limits.videoResourceBytes - totalVideoBytes)
            {
                detail::fail(AdmissionFailure::InvalidVideoResourceSize,
                             failure, diagnostic);
                return std::nullopt;
            }
            totalVideoBytes += resource.byteCount;
        }
        if (videoResources.empty() && untrusted.helperGeneration != 0)
        {
            detail::fail(AdmissionFailure::InvalidVideoResourceRevision,
                         failure, diagnostic);
            return std::nullopt;
        }

        auto bindings = untrusted.bindings;
        for (auto& binding : bindings)
            std::sort(binding.textures.begin(), binding.textures.end(),
                      [] (const auto& left, const auto& right)
                      { return left.slot < right.slot; });
        std::sort(bindings.begin(), bindings.end(), detail::bindingLess);
        if (std::adjacent_find(bindings.begin(), bindings.end(), detail::sameTarget)
            != bindings.end())
        {
            detail::fail(AdmissionFailure::DuplicateBindingTarget,
                         failure, diagnostic);
            return std::nullopt;
        }

        std::size_t textureBindingCount = 0;
        std::set<std::pair<std::string, std::uint64_t>> usedPrograms;
        std::set<VideoResourceIdentity, decltype(&detail::identityLess)>
            usedVideoResources(&detail::identityLess);
        std::array<bool, 14> requiredModulations {};
        for (const auto& binding : bindings)
        {
            SceneMaterialId importedMaterial {};
            if (binding.targetKind == BindingTargetKind::MaterialSlot)
            {
                if (!binding.materialSlot.isValid() || binding.object.isValid())
                {
                    detail::fail(AdmissionFailure::InvalidBindingTarget,
                                 failure, diagnostic);
                    return std::nullopt;
                }
                if (detail::findMaterial(*untrusted.scene, binding.materialSlot) == nullptr)
                {
                    detail::fail(AdmissionFailure::MissingBindingTarget,
                                 failure, diagnostic);
                    return std::nullopt;
                }
                importedMaterial = binding.materialSlot;
            }
            else if (binding.targetKind == BindingTargetKind::ObjectOverride)
            {
                if (!binding.object.isValid() || binding.materialSlot.isValid())
                {
                    detail::fail(AdmissionFailure::InvalidBindingTarget,
                                 failure, diagnostic);
                    return std::nullopt;
                }
                const auto* object = detail::findObject(*untrusted.scene, binding.object);
                if (object == nullptr)
                {
                    detail::fail(AdmissionFailure::MissingBindingTarget,
                                 failure, diagnostic);
                    return std::nullopt;
                }
                importedMaterial = object->material;
            }
            else
            {
                detail::fail(AdmissionFailure::InvalidBindingTarget,
                             failure, diagnostic);
                return std::nullopt;
            }

            const auto* program = detail::findProgram(programs, binding);
            if (program == nullptr)
            {
                detail::fail(AdmissionFailure::MissingSurfaceMaterial,
                             failure, diagnostic);
                return std::nullopt;
            }
            usedPrograms.emplace(binding.surfaceMaterialDigest,
                                 binding.surfaceMaterialRevision);
            const auto textureSlots = program->program->program().textureSlotCount;
            if (binding.textures.size() != textureSlots
                || binding.textures.size() > limits.textureBindings - textureBindingCount)
            {
                detail::fail(binding.textures.size() > limits.textureBindings
                                - textureBindingCount
                                 ? AdmissionFailure::TextureBindingCapacityExceeded
                                 : AdmissionFailure::InvalidTextureBinding,
                             failure, diagnostic);
                return std::nullopt;
            }
            textureBindingCount += binding.textures.size();
            const auto* imported = detail::findMaterial(*untrusted.scene, importedMaterial);
            for (std::size_t slot = 0; slot < binding.textures.size(); ++slot)
            {
                const auto& texture = binding.textures[slot];
                if (texture.slot != slot)
                {
                    detail::fail(AdmissionFailure::InvalidTextureBinding,
                                 failure, diagnostic);
                    return std::nullopt;
                }
                if (texture.source == TextureSourceKind::ImportedBaseColor)
                {
                    if (hasIdentity(texture.videoResource))
                    {
                        detail::fail(AdmissionFailure::InvalidTextureBinding,
                                     failure, diagnostic);
                        return std::nullopt;
                    }
                    if (imported == nullptr || !imported->baseColorTexture.isValid())
                    {
                        detail::fail(AdmissionFailure::MissingImportedTexture,
                                     failure, diagnostic);
                        return std::nullopt;
                    }
                }
                else if (texture.source == TextureSourceKind::VideoResource)
                {
                    if (!hasIdentity(texture.videoResource)
                        || detail::findVideoResource(videoResources,
                                                     texture.videoResource) == nullptr)
                    {
                        detail::fail(AdmissionFailure::MissingVideoResource,
                                     failure, diagnostic);
                        return std::nullopt;
                    }
                    usedVideoResources.insert(texture.videoResource);
                }
                else
                {
                    detail::fail(AdmissionFailure::InvalidTextureBinding,
                                 failure, diagnostic);
                    return std::nullopt;
                }
            }

            for (const auto& operation : program->program->program().operations)
                if (operation.kind == surfacematerial::OperationKind::Input
                    && isModulationSemantic(operation.semantic))
                    requiredModulations[static_cast<std::size_t>(operation.semantic)] = true;
        }

        if (usedPrograms.size() != programs.size())
        {
            detail::fail(AdmissionFailure::UnreferencedProgram, failure, diagnostic);
            return std::nullopt;
        }
        if (usedVideoResources.size() != videoResources.size())
        {
            detail::fail(AdmissionFailure::UnreferencedVideoResource,
                         failure, diagnostic);
            return std::nullopt;
        }

        auto modulationInputs = untrusted.modulationInputs;
        std::sort(modulationInputs.begin(), modulationInputs.end(),
                  [] (const auto& left, const auto& right)
                  { return left.semantic < right.semantic; });
        for (std::size_t index = 0; index < modulationInputs.size(); ++index)
        {
            const auto& input = modulationInputs[index];
            if (!isModulationSemantic(input.semantic) || !std::isfinite(input.value)
                || input.sourceRevision == 0)
            {
                detail::fail(AdmissionFailure::InvalidModulationInput,
                             failure, diagnostic);
                return std::nullopt;
            }
            if (index != 0 && modulationInputs[index - 1].semantic == input.semantic)
            {
                detail::fail(AdmissionFailure::DuplicateModulationInput,
                             failure, diagnostic);
                return std::nullopt;
            }
            const auto semantic = static_cast<std::size_t>(input.semantic);
            if (!requiredModulations[semantic])
            {
                detail::fail(AdmissionFailure::UnreferencedModulationInput,
                             failure, diagnostic);
                return std::nullopt;
            }
            requiredModulations[semantic] = false;
        }
        if (std::find(requiredModulations.begin(), requiredModulations.end(), true)
            != requiredModulations.end())
        {
            detail::fail(AdmissionFailure::MissingModulationInput,
                         failure, diagnostic);
            return std::nullopt;
        }

        std::vector<AdmittedMaterialBindings::ObjectMaterial> objects;
        objects.reserve(untrusted.scene->objectCount);
        for (std::size_t index = 0; index < untrusted.scene->objectCount; ++index)
            objects.push_back({ untrusted.scene->objects[index].id,
                                untrusted.scene->objects[index].material });
        std::sort(objects.begin(), objects.end(),
                  [] (const auto& left, const auto& right)
                  { return left.object.value < right.object.value; });

        auto exactDigest = detail::digest(untrusted, programs, bindings,
                                          videoResources, modulationInputs);
        return AdmittedMaterialBindings(
            untrusted.scene->id, untrusted.sceneRevision,
            untrusted.structuralRevision, untrusted.evaluationRevision,
            untrusted.helperGeneration, untrusted.fallback,
            std::move(objects), std::move(programs), std::move(bindings),
            std::move(videoResources), std::move(modulationInputs),
            std::move(exactDigest));
    }
    catch (const std::bad_alloc&)
    {
        detail::fail(AdmissionFailure::AllocationFailed, failure, diagnostic);
        return std::nullopt;
    }
}
} // namespace surfacematerialbinding
