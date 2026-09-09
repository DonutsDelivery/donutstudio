#include "fixture_scene_renderer.h"
#include "diffractive_foil_admission.h"
#include "diffraction_material_execution.h"
#include "surface_material_binding_admission.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace videorender::fixture3d
{
namespace
{
using namespace HarmonicMIDI::grid;

bool admitFixtureSubset (const Visual3DScene& scene, std::string& error)
{
    if (scene.objectCount == 0 || scene.materialCount == 0
        || scene.cameraCount == 0)
    {
        error = "native fixture rendering requires at least one object, material, and camera";
        return false;
    }

    // Keep sampler admission explicit here because these values select native
    // allocation descriptors. Unknown glTF enums must fail before a backend
    // creates an image or sampler.
    const auto admittedMagFilter = [] (std::uint32_t value)
    {
        return value == 9728 || value == 9729;
    };
    const auto admittedMinFilter = [] (std::uint32_t value)
    {
        return value == 9728 || value == 9729 || value == 9984
            || value == 9985 || value == 9986 || value == 9987;
    };
    const auto admittedWrap = [] (std::uint32_t value)
    {
        return value == 33071 || value == 33648 || value == 10497;
    };
    for (std::size_t index = 0; index < scene.textureCount; ++index)
    {
        const auto& texture = scene.textures[index];
        if (! admittedMagFilter (texture.magFilter)
            || ! admittedMinFilter (texture.minFilter)
            || ! admittedWrap (texture.wrapS)
            || ! admittedWrap (texture.wrapT))
        {
            error = "native fixture rendering rejects an unsupported texture sampler";
            return false;
        }
    }

    // validateVisual3DScene has already bounded every fixed-capacity collection,
    // geometry range, hierarchy, texture reference, material value, camera, and
    // punctual light before either backend can allocate.
    return true;
}

std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>
admitNativeSurfaceMaterial (const Visual3DScene& scene,
                            HarmonicMIDI::grid::SceneObjectId object,
                            const std::string& bindingDigest,
                            const videohelper::materialprogram::MaterialProgramRecord& program,
                            std::string& error)
{
    using namespace surfacematerial;
    using NativeProgram = arbitgpu::NativeFixtureSurfaceMaterialProgram;
    if (! object.isValid() || object != scene.objects[0].id)
    {
        error = "native surface material binding must target the exact admitted object";
        return {};
    }
    if (bindingDigest.empty())
    {
        error = "native surface material binding requires an exact admitted digest";
        return {};
    }
    if (! program.hasSurfaceProgram() || program.hasVertexProgram())
    {
        error = "native surface material execution requires one surface-only program";
        return {};
    }
    arbitgpu::NativeFixtureMaterialBackend nativeBackend
        = arbitgpu::NativeFixtureMaterialBackend::Invalid;
    switch (program.capabilities().backend.target)
    {
        case videohelper::materialprogram::BackendTarget::OpenGl:
            nativeBackend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
            break;
        case videohelper::materialprogram::BackendTarget::Metal:
            nativeBackend = arbitgpu::NativeFixtureMaterialBackend::Metal;
            break;
        case videohelper::materialprogram::BackendTarget::Invalid:
            error = "native surface material execution requires an OpenGL or Metal material program";
            return {};
    }
    if (scene.objects[0].material != scene.materials[0].id)
    {
        error = "native surface material execution requires the object's exact imported material";
        return {};
    }

    const auto& instructions = program.surfaceInstructions();
    const auto& outputs = program.surfaceOutputSlots();
    auto instructionAt = [&] (std::uint16_t slot)
        -> const videohelper::materialprogram::SurfaceInstruction*
    {
        return slot < instructions.size() ? &instructions[slot] : nullptr;
    };
    auto instructionFor = [&] (OutputSemantic semantic)
        -> const videohelper::materialprogram::SurfaceInstruction*
    {
        return instructionAt (outputs[static_cast<std::size_t> (semantic)]);
    };

    const auto* base = instructionFor (OutputSemantic::BaseColor);
    bool texturedBaseColor = false;
    const videohelper::materialprogram::SurfaceInstruction* textureTint = nullptr;
    std::array<std::uint16_t, 7> textureSlots {};
    auto admitsTextureRgb = [&] (
        const videohelper::materialprogram::SurfaceInstruction* candidate) -> bool
    {
        if (candidate == nullptr || candidate->operation != OperationKind::ComposeVec3
            || candidate->resultType != ValueType::Vec3 || candidate->inputCount != 3)
            return false;
        const videohelper::materialprogram::SurfaceInstruction* sample = nullptr;
        bool exact = true;
        textureSlots[0] = candidate->resultSlot;
        for (std::size_t lane = 0; lane < 3; ++lane)
        {
            const auto* component = instructionAt (candidate->inputs[lane]);
            if (component == nullptr || component->operation != OperationKind::Component
                || component->resultType != ValueType::Scalar || component->inputCount != 1
                || component->parameter != lane)
            {
                exact = false;
                break;
            }
            const auto* candidateSample = instructionAt (component->inputs[0]);
            if (candidateSample == nullptr
                || candidateSample->operation != OperationKind::TextureSample2D
                || candidateSample->resultType != ValueType::Vec4
                || candidateSample->inputCount != 1 || candidateSample->parameter != 0
                || (sample != nullptr && sample != candidateSample))
            {
                exact = false;
                break;
            }
            sample = candidateSample;
            textureSlots[lane + 1] = component->resultSlot;
        }
        if (exact && sample != nullptr)
        {
            const auto* uv = instructionAt (sample->inputs[0]);
            exact = uv != nullptr && uv->operation == OperationKind::Input
                 && uv->resultType == ValueType::Vec2 && uv->inputCount == 0
                 && uv->semantic == InputSemantic::TexCoord0;
            textureSlots[4] = sample->resultSlot;
            if (exact)
                textureSlots[5] = uv->resultSlot;
        }
        return exact;
    };
    texturedBaseColor = admitsTextureRgb (base);
    if (! texturedBaseColor && base != nullptr
        && base->operation == OperationKind::Multiply
        && base->resultType == ValueType::Vec3 && base->inputCount == 2)
    {
        for (std::size_t textureInput = 0; textureInput < 2; ++textureInput)
        {
            const auto* candidateTexture = instructionAt (base->inputs[textureInput]);
            const auto* candidateTint = instructionAt (base->inputs[1u - textureInput]);
            if (admitsTextureRgb (candidateTexture)
                && candidateTint != nullptr
                && candidateTint->operation == OperationKind::FloatConstant
                && candidateTint->resultType == ValueType::Vec3
                && candidateTint->inputCount == 0)
            {
                texturedBaseColor = true;
                textureTint = candidateTint;
                break;
            }
        }
        if (texturedBaseColor)
            textureSlots[6] = base->resultSlot;
    }

    bool timeMixedBaseColor = false;
    std::array<std::uint16_t, 3> timeMixSlots {};
    const videohelper::materialprogram::SurfaceInstruction* timeMixStart = nullptr;
    const videohelper::materialprogram::SurfaceInstruction* timeMixEnd = nullptr;
    if (base != nullptr && base->operation == OperationKind::Mix
        && base->resultType == ValueType::Vec3 && base->inputCount == 3)
    {
        timeMixStart = instructionAt (base->inputs[0]);
        timeMixEnd = instructionAt (base->inputs[1]);
        const auto* clamp = instructionAt (base->inputs[2]);
        if (timeMixStart != nullptr && timeMixEnd != nullptr && clamp != nullptr
            && timeMixStart->operation == OperationKind::FloatConstant
            && timeMixStart->resultType == ValueType::Vec3
            && timeMixStart->inputCount == 0
            && timeMixEnd->operation == OperationKind::FloatConstant
            && timeMixEnd->resultType == ValueType::Vec3
            && timeMixEnd->inputCount == 0
            && clamp->operation == OperationKind::Clamp
            && clamp->resultType == ValueType::Scalar && clamp->inputCount == 3)
        {
            const auto* time = instructionAt (clamp->inputs[0]);
            const auto* lower = instructionAt (clamp->inputs[1]);
            const auto* upper = instructionAt (clamp->inputs[2]);
            timeMixedBaseColor = time != nullptr && lower != nullptr && upper != nullptr
                && time->operation == OperationKind::Input
                && time->resultType == ValueType::Scalar && time->inputCount == 0
                && time->semantic == InputSemantic::Time
                && lower->operation == OperationKind::FloatConstant
                && lower->resultType == ValueType::Scalar && lower->inputCount == 0
                && lower->literal[0] == 0.0f
                && upper->operation == OperationKind::FloatConstant
                && upper->resultType == ValueType::Scalar && upper->inputCount == 0
                && upper->literal[0] == 1.0f;
            if (timeMixedBaseColor)
                timeMixSlots = { base->resultSlot, clamp->resultSlot, time->resultSlot };
        }
    }

    const bool constantBaseColor = base != nullptr
        && base->operation == OperationKind::FloatConstant
        && base->resultType == ValueType::Vec3 && base->inputCount == 0;
    if (! constantBaseColor && ! texturedBaseColor && ! timeMixedBaseColor)
    {
        error = "native surface material baseColor supports only a linear vec3 constant, exact imported sRGB texture RGB sampled with TEXCOORD_0 optionally multiplied by one linear vec3 tint, or mix of two linear vec3 constants driven by clamp(time, 0, 1)";
        return {};
    }
    if (((constantBaseColor || timeMixedBaseColor) && program.resources().textureSlots != 0)
        || (texturedBaseColor && program.resources().textureSlots != 1))
    {
        error = "native surface material texture declaration does not match the bounded baseColor program";
        return {};
    }

    for (const auto semantic : kSurfaceOutputs)
    {
        if (semantic == OutputSemantic::BaseColor)
            continue;
        const auto* instruction = instructionFor (semantic);
        const auto expectedOperation = semantic == OutputSemantic::MaterialId
            ? OperationKind::UIntConstant : OperationKind::FloatConstant;
        if (instruction == nullptr || instruction->operation != expectedOperation
            || instruction->resultType != outputType (semantic) || instruction->inputCount != 0)
        {
            error = "native surface material PBR subset requires constant metallic, roughness, emission, opacity, normal, transmission, IOR, clearcoat, and materialId outputs";
            return {};
        }
    }

    for (const auto& instruction : instructions)
    {
        const bool constant = (instruction.operation == OperationKind::FloatConstant
                            || instruction.operation == OperationKind::UIntConstant)
                           && instruction.inputCount == 0;
        const bool textureOperation = texturedBaseColor
            && std::find (textureSlots.begin(), textureSlots.end(), instruction.resultSlot)
                != textureSlots.end();
        const bool timeMixOperation = timeMixedBaseColor
            && std::find (timeMixSlots.begin(), timeMixSlots.end(), instruction.resultSlot)
                != timeMixSlots.end();
        if (! constant && ! textureOperation && ! timeMixOperation)
        {
            error = "native surface material PBR subset rejects operation "
                  + std::string (token (instruction.operation));
            return {};
        }
    }

    const auto* metallic = instructionFor (OutputSemantic::Metallic);
    const auto* roughness = instructionFor (OutputSemantic::Roughness);
    const auto* emission = instructionFor (OutputSemantic::Emission);
    const auto* opacity = instructionFor (OutputSemantic::Opacity);
    const auto* normal = instructionFor (OutputSemantic::Normal);
    const auto* transmission = instructionFor (OutputSemantic::Transmission);
    const auto* ior = instructionFor (OutputSemantic::Ior);
    const auto* clearcoat = instructionFor (OutputSemantic::Clearcoat);
    const auto* materialId = instructionFor (OutputSemantic::MaterialId);
    if (materialId->unsignedLiteral != scene.objects[0].material.value)
    {
        error = "native surface material materialId must match the rendered object material";
        return {};
    }
    if (textureTint != nullptr)
    {
        for (std::size_t channel = 0; channel < 3; ++channel)
        {
            const auto value = textureTint->literal[channel];
            if (! std::isfinite(value) || value < 0.0f || value > 1.0f)
            {
                error = "native surface material texture tint must be finite and within [0, 1]";
                return {};
            }
        }
    }

    auto native = std::make_shared<NativeProgram>();
    native->backend = nativeBackend;
    native->kind = arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
    native->object = object;
    native->bindingDigest = bindingDigest;
    native->programIdentity = program.programIdentity();
    native->baseColorSource = texturedBaseColor
        ? NativeProgram::BaseColorSource::ImportedSrgbTexture
        : (timeMixedBaseColor
            ? NativeProgram::BaseColorSource::TimeLinearMix
            : NativeProgram::BaseColorSource::ConstantLinear);
    if (texturedBaseColor)
    {
        if (! scene.materials[0].baseColorTexture.isValid())
        {
            error = "native surface material imported texture binding has no decoded base-color texture";
            return {};
        }
        const auto* source = visual3d_detail::findById (
            scene.textures, scene.textureCount, scene.materials[0].baseColorTexture);
        if (source == nullptr)
        {
            error = "native surface material imported texture binding does not resolve an admitted texture";
            return {};
        }
        const auto texelCount = static_cast<std::size_t> (source->width) * source->height;
        if (texelCount == 0 || texelCount > Visual3DScene::kMaxTextureTexels
            || source->firstTexel > scene.textureTexelCount
            || texelCount > scene.textureTexelCount - source->firstTexel)
        {
            error = "native surface material imported texture range is invalid";
            return {};
        }
        NativeProgram::ImportedSrgbTexture texture;
        texture.width = source->width;
        texture.height = source->height;
        texture.texelCount = texelCount;
        std::copy_n (scene.textureTexels.begin() + source->firstTexel,
                     texelCount, texture.texels.begin());
        native->importedBaseColorTexture = std::move (texture);
    }
    const auto* nativeBaseColor = timeMixedBaseColor ? timeMixStart : base;
    native->parameters.baseColorMetallic = {
        texturedBaseColor ? (textureTint != nullptr ? textureTint->literal[0] : 1.0f)
                          : nativeBaseColor->literal[0],
        texturedBaseColor ? (textureTint != nullptr ? textureTint->literal[1] : 1.0f)
                          : nativeBaseColor->literal[1],
        texturedBaseColor ? (textureTint != nullptr ? textureTint->literal[2] : 1.0f)
                          : nativeBaseColor->literal[2],
        metallic->literal[0] };
    if (timeMixedBaseColor)
    {
        std::copy_n (timeMixEnd->literal.begin(), native->timeMixEndColor.size(),
                     native->timeMixEndColor.begin());
    }
    native->parameters.emissionRoughness = {
        emission->literal[0], emission->literal[1], emission->literal[2],
        roughness->literal[0]
    };
    native->parameters.normalOpacity = {
        normal->literal[0], normal->literal[1], normal->literal[2], opacity->literal[0]
    };
    native->parameters.transmissionIorClearcoat = {
        transmission->literal[0], ior->literal[0], clearcoat->literal[0], 0.0f
    };
    native->parameters.identifiers[0] = materialId->unsignedLiteral;
    return native;
}

arbitgpu::NativeFixtureMaterialBackend nativeMaterialBackend (
    videohelper::materialprogram::BackendTarget target) noexcept
{
    switch (target)
    {
        case videohelper::materialprogram::BackendTarget::OpenGl:
            return arbitgpu::NativeFixtureMaterialBackend::OpenGl;
        case videohelper::materialprogram::BackendTarget::Metal:
            return arbitgpu::NativeFixtureMaterialBackend::Metal;
        case videohelper::materialprogram::BackendTarget::Invalid:
            break;
    }
    return arbitgpu::NativeFixtureMaterialBackend::Invalid;
}

bool hasValidUvDerivatives (const Visual3DScene& scene,
                            const SceneObjectRecord& object) noexcept
{
    if (object.indexCount == 0 || object.indexCount % 3u != 0u)
        return false;
    for (std::uint32_t index = 0; index < object.indexCount; index += 3u)
    {
        const auto first = scene.indices[object.firstIndex + index];
        const auto second = scene.indices[object.firstIndex + index + 1u];
        const auto third = scene.indices[object.firstIndex + index + 2u];
        if (first >= object.vertexCount || second >= object.vertexCount
            || third >= object.vertexCount)
            return false;
        const auto& a = scene.vertices[object.firstVertex + first].uv;
        const auto& b = scene.vertices[object.firstVertex + second].uv;
        const auto& c = scene.vertices[object.firstVertex + third].uv;
        const auto determinant = (b.x - a.x) * (c.y - a.y)
                               - (b.y - a.y) * (c.x - a.x);
        if (! std::isfinite (determinant) || std::abs (determinant) <= 1.0e-8f)
            return false;
    }
    return true;
}

bool hasUnitUvDomain(const Visual3DScene& scene,
                     const SceneObjectRecord& object) noexcept
{
    for (std::uint32_t index = 0; index < object.vertexCount; ++index)
    {
        const auto& uv = scene.vertices[object.firstVertex + index].uv;
        if (! std::isfinite (uv.x) || ! std::isfinite (uv.y)
            || uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f)
            return false;
    }
    return true;
}

std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>
admitNativeDiffractionMaterial (
    const Visual3DScene& scene,
    const diffractionmaterialbinding::AdmittedImportedSceneDiffractionMaterial& admitted,
    videohelper::materialprogram::BackendTarget target,
    std::string& error)
{
    using NativeProgram = arbitgpu::NativeFixtureSurfaceMaterialProgram;
    const auto backend = nativeMaterialBackend (target);
    if (backend == arbitgpu::NativeFixtureMaterialBackend::Invalid)
    {
        error = "native diffraction material execution requires an OpenGL or Metal target";
        return {};
    }
    if (admitted.scene() != scene.id || admitted.object() != scene.objects[0].id)
    {
        error = "native diffraction material binding must target the exact admitted object";
        return {};
    }
    const auto& description = admitted.material().description();
    if (description.geometry.lattice != diffractionmaterial::GratingLattice::OneDimensional
        && description.geometry.lattice
            != diffractionmaterial::GratingLattice::CrossedTwoDimensional)
    {
        error = "native diffraction surface rendering requires a supported diffraction lattice";
        return {};
    }
    if (description.roughness.rmsSlope
        < arbitgpu::kMinimumNativeFixtureDiffractionRmsSlope)
    {
        error = "native diffraction surface rendering requires RMS slope at least 0.0001 for a finite angular lobe";
        return {};
    }

    if (! hasValidUvDerivatives (scene, scene.objects[0]))
    {
        error = "native diffraction surface rendering requires non-degenerate UV derivatives on every triangle";
        return {};
    }
    if (description.grooveField.mode != diffractionmaterial::GrooveFieldMode::Constant
        && ! hasUnitUvDomain(scene, scene.objects[0]))
    {
        error = "native spatial diffraction rendering requires material UVs in the admitted unit domain";
        return {};
    }

    std::string validationError;
    const auto& lighting = admitted.lighting().description();
    for (std::size_t pathIndex = 0; pathIndex < lighting.pathCount; ++pathIndex)
    {
        if (! diffractionmaterial::physicalcheckpoint::validate (
                admitted.material(), lighting.paths[pathIndex].incident,
                target == videohelper::materialprogram::BackendTarget::OpenGl
                    ? "opengl" : "metal",
                validationError))
        {
            error = "native diffraction surface admission rejected at lighting path "
                + std::to_string(pathIndex) + ": " + validationError;
            return {};
        }
    }

    auto native = std::make_shared<NativeProgram>();
    native->backend = backend;
    native->kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    native->object = admitted.object();
    native->bindingDigest = admitted.material().structuralDigest();
    native->diffractionPathCount = lighting.pathCount;
    native->diffractionLightingAdmission
        = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(admitted.lighting());
    native->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
        native->bindingDigest, *native->diffractionLightingAdmission);
    for (std::size_t pathIndex = 0; pathIndex < lighting.pathCount; ++pathIndex)
    {
        native->diffractionPaths[pathIndex] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                admitted.material(), lighting.paths[pathIndex].incident),
            lighting.paths[pathIndex]);
        native->diffractionMaximumBounceDepth = std::max(
            native->diffractionMaximumBounceDepth,
            lighting.paths[pathIndex].bounceDepth);
    }
    error.clear();
    return native;
}
} // namespace

std::shared_ptr<const AdmittedSurfaceMaterialBinding>
admitSurfaceMaterialBinding (const std::shared_ptr<const Visual3DScene>& sceneSnapshot,
                             const surfacematerialbinding::ImportedSceneMaterialRequest& request,
                             videohelper::materialprogram::BackendTarget target,
                             std::string& error)
{
    using namespace surfacematerialbinding;
    if (sceneSnapshot == nullptr || request.sceneSnapshot == nullptr)
    {
        error = "surface material request requires an immutable owned scene snapshot";
        return {};
    }
    if (request.sceneSnapshot != sceneSnapshot)
    {
        error = "surface material request does not share the exact admitted scene snapshot";
        return {};
    }
    const auto& scene = *sceneSnapshot;
    if (request.version != kWireVersion)
    {
        error = "surface material request version is unsupported";
        return {};
    }
    if (request.scene != scene.id)
    {
        error = "surface material request targets a different admitted scene";
        return {};
    }
    if (request.sceneRevision == 0 || request.structuralRevision == 0
        || request.evaluationRevision == 0 || request.programRevision == 0
        || request.programRevision != request.structuralRevision
        || request.binding.surfaceMaterialRevision != request.programRevision)
    {
        error = "surface material request revisions are missing or inconsistent";
        return {};
    }
    if (request.binding.targetKind != BindingTargetKind::ObjectOverride
        || request.binding.object != scene.objects[0].id
        || request.binding.materialSlot.isValid())
    {
        error = "surface material request must bind the exact imported scene object";
        return {};
    }
    if (request.binding.textures.size() != request.program.textureSlotCount)
    {
        error = "native surface material request texture bindings do not match its declared slots";
        return {};
    }
    if (target != videohelper::materialprogram::BackendTarget::OpenGl
        && target != videohelper::materialprogram::BackendTarget::Metal)
    {
        error = "native surface material execution requires an OpenGL or Metal compiler target";
        return {};
    }

    std::string diagnostic;
    auto admittedProgram = surfacematerial::admit (request.program, diagnostic);
    if (! admittedProgram)
    {
        error = "surface material program admission failed: " + diagnostic;
        return {};
    }
    if (admittedProgram->structuralDigest()
        != request.binding.surfaceMaterialDigest)
    {
        error = "surface material program digest does not match its exact binding";
        return {};
    }

    auto immutableProgram
        = std::make_shared<const surfacematerial::AdmittedSurfaceMaterialIR> (
            std::move (*admittedProgram));
    Description description;
    description.scene = &scene;
    description.sceneRevision = request.sceneRevision;
    description.structuralRevision = request.structuralRevision;
    description.evaluationRevision = request.evaluationRevision;
    description.programs.push_back ({ immutableProgram, request.programRevision });
    description.bindings.push_back (request.binding);

    AdmissionFailure failure = AdmissionFailure::None;
    auto admittedBindings = admit (description, failure, diagnostic);
    if (! admittedBindings)
    {
        error = std::string (token (failure)) + ": " + diagnostic;
        return {};
    }
    const auto resolution = admittedBindings->resolve (scene.objects[0].id);
    if (! resolution || resolution->kind != ResolutionKind::ObjectOverride
        || resolution->binding == nullptr || resolution->program == nullptr
        || resolution->program->program == nullptr)
    {
        error = "admitted material does not resolve the exact imported scene object override";
        return {};
    }

    auto compiled = videohelper::materialprogram::compileMaterialProgram (
        resolution->program->program.get(), nullptr, target, diagnostic);
    if (! compiled)
    {
        error = "surface material compilation failed: " + diagnostic;
        return {};
    }
    auto immutableCompiled
        = std::make_shared<const videohelper::materialprogram::MaterialProgramRecord> (
            std::move (*compiled));
    auto native = admitNativeSurfaceMaterial (
        scene, scene.objects[0].id, admittedBindings->digest(),
        *immutableCompiled, diagnostic);
    if (native == nullptr)
    {
        error = std::move (diagnostic);
        return {};
    }

    error.clear();
    return std::shared_ptr<const AdmittedSurfaceMaterialBinding> (
        new AdmittedSurfaceMaterialBinding (
            scene.id, scene.objects[0].id, request.sceneRevision,
            request.structuralRevision, request.evaluationRevision,
            request.programRevision, admittedBindings->digest(), sceneSnapshot,
            std::move (native)));
}

std::shared_ptr<const AdmittedDiffractionMaterialBinding>
admitDiffractionMaterialBinding (
    const std::shared_ptr<const Visual3DScene>& sceneSnapshot,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
    videohelper::materialprogram::BackendTarget target,
    std::string& error)
{
    if (sceneSnapshot == nullptr)
    {
        error = "diffraction material request requires an immutable owned scene snapshot";
        return {};
    }
    if (request.scene != sceneSnapshot->id)
    {
        error = "diffraction material request targets a different admitted scene";
        return {};
    }
    std::string diagnostic;
    auto admitted = diffractionmaterialbinding::admit (request, diagnostic);
    if (! admitted)
    {
        error = std::move (diagnostic);
        return {};
    }
    auto native = admitNativeDiffractionMaterial (
        *sceneSnapshot, *admitted, target, diagnostic);
    if (native == nullptr)
    {
        error = std::move (diagnostic);
        return {};
    }
    auto spatial = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*native);
    if (admitted->spatialFoil())
    {
        const auto& foil = *admitted->spatialFoil();
        for (std::size_t index = 0; index < foil.description().grooveField.size(); ++index)
        {
            const auto& sample = foil.description().grooveField[index];
            spatial->diffractionFoilField[index] = { sample.reciprocalDirectionUv[0],
                sample.reciprocalDirectionUv[1], sample.grooveSpacingNanometres,
                sample.diffractionCoverage };
        }
        spatial->diffractionFoilMaximumEvaluations
            = foil.description().workBudget.maximumEvaluations;
        spatial->bindingDigest = foil.structuralDigest();
        spatial->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            spatial->bindingDigest, *spatial->diffractionLightingAdmission);
    }
    native = std::move(spatial);

    error.clear();
    return std::shared_ptr<const AdmittedDiffractionMaterialBinding> (
        new AdmittedDiffractionMaterialBinding (
            *admitted, request.structuralDigest, sceneSnapshot, std::move (native)));
}

std::shared_ptr<const AdmittedDiffractionMaterialBinding>
admitDiffractionProductPlan (
    const std::shared_ptr<const Visual3DScene>& sceneSnapshot,
    const diffractionmaterial::DiffractionProductRenderRequest& request,
    std::uint64_t sceneRevision,
    std::uint64_t structuralRevision,
    std::uint64_t evaluationRevision,
    videohelper::materialprogram::BackendTarget target,
    std::string& error)
{
    using namespace diffractionmaterial;
    if (sceneSnapshot == nullptr || sceneSnapshot->objectCount == 0
        || request.plan == nullptr)
    {
        error = "diffraction product request requires owned immutable plan and scene snapshots";
        return {};
    }
    const auto& plan = *request.plan;
    const auto kind = static_cast<std::uint8_t> (plan.kind);
    const auto mask = static_cast<std::uint8_t> (plan.material.patternMask);
    if (plan.version != 1 || kind < 1 || kind > 4 || mask < 1 || mask > 3
        || ! std::isfinite (plan.material.maskCoverage)
        || plan.material.maskCoverage < 0.0f || plan.material.maskCoverage > 1.0f)
    {
        error = "diffraction product plan version, kind, or pattern bounds are invalid";
        return {};
    }
    const auto& description = plan.material.diffractionBsdf;
    const auto& coating = plan.material.clearcoat;
    const auto& grooves = plan.material.embossedGrooves;
    if (coating.model != description.coating.model
        || coating.thicknessNanometres != description.coating.thicknessNanometres
        || coating.opticalConstants.refractiveIndex
            != description.coating.opticalConstants.refractiveIndex
        || coating.opticalConstants.extinctionCoefficient
            != description.coating.opticalConstants.extinctionCoefficient
        || grooves.mode != description.grooveField.mode
        || grooves.originUv != description.grooveField.originUv
        || grooves.axisUv != description.grooveField.axisUv
        || grooves.grooveSpacingDeltaNanometresPerUnit
            != description.grooveField.grooveSpacingDeltaNanometresPerUnit
        || grooves.secondarySpacingDeltaNanometresPerUnit
            != description.grooveField.secondarySpacingDeltaNanometresPerUnit
        || grooves.orientationDegreesPerUnit
            != description.grooveField.orientationDegreesPerUnit)
    {
        error = "diffraction product material mirrors do not match the executable material";
        return {};
    }
    const auto exactDigest = exactDiffractionProductDigest (plan);
    if (exactDigest.empty() || request.productDigest != exactDigest)
    {
        error = "diffraction product digest is stale or does not match its immutable plan";
        return {};
    }
    if (sceneRevision == 0 || structuralRevision == 0 || evaluationRevision == 0)
    {
        error = "diffraction product revisions must be nonzero";
        return {};
    }

    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest lowered;
    lowered.scene = sceneSnapshot->id;
    lowered.sceneRevision = sceneRevision;
    lowered.structuralRevision = structuralRevision;
    lowered.evaluationRevision = evaluationRevision;
    lowered.materialRevision = structuralRevision;
    lowered.object = sceneSnapshot->objects[0].id;
    lowered.material = description;
    lowered.lighting = plan.lighting;
    std::string diagnostic;
    const auto admittedMaterial = admit (lowered.material, diagnostic);
    if (! admittedMaterial)
    {
        error = "diffraction product material admission failed: " + diagnostic;
        return {};
    }
    lowered.structuralDigest = admittedMaterial->structuralDigest();
    auto binding = admitDiffractionMaterialBinding (
        sceneSnapshot, lowered, target, error);
    if (! binding)
        return {};
    auto productProgram = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
        *binding->nativeProgram());
    productProgram->diffractionPatternMask = mask;
    productProgram->diffractionMaskCoverage = plan.material.maskCoverage;
    productProgram->diffractionProductDigest = request.productDigest;
    const auto& occupancy = plan.material.occupancyMask;
    if (occupancy.rectangleCount == 0
        || occupancy.rectangleCount > occupancy.kMaximumRectangles
        || (plan.material.patternMask == PatternMask::FullSurface
            && (occupancy.rectangleCount != 1
                || occupancy.rectangles[0] != std::array<float, 4> { 0.0f, 0.0f, 1.0f, 1.0f }
                || plan.material.maskCoverage != 1.0f)))
    {
        error = "diffraction product occupancy geometry is invalid";
        return {};
    }
    productProgram->diffractionOccupancyRectangleCount = occupancy.rectangleCount;
    for (std::size_t index = 0; index < occupancy.rectangleCount; ++index)
    {
        const auto& rectangle = occupancy.rectangles[index];
        if (!std::all_of(rectangle.begin(), rectangle.end(), [] (float value) {
                return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
            }) || rectangle[0] >= rectangle[2] || rectangle[1] >= rectangle[3])
        {
            error = "diffraction product occupancy rectangle is invalid";
            return {};
        }
        productProgram->diffractionOccupancyRectangles[index]
            = { rectangle[0], rectangle[1], rectangle[2], rectangle[3] };
    }
    std::uint32_t coverageBits = 0;
    static_assert(sizeof(coverageBits) == sizeof(plan.material.maskCoverage));
    std::memcpy(&coverageBits, &plan.material.maskCoverage, sizeof(coverageBits));
    for (std::size_t index = 0; index < productProgram->diffractionPathCount; ++index)
    {
        productProgram->diffractionPaths[index].kindBounceAndReserved[2] = mask;
        productProgram->diffractionPaths[index].kindBounceAndReserved[3] = coverageBits;
    }
    productProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity (
        productProgram->bindingDigest, *productProgram->diffractionLightingAdmission,
        productProgram->diffractionPatternMask, productProgram->diffractionMaskCoverage,
        productProgram->diffractionProductDigest);
    auto productBinding = std::shared_ptr<const AdmittedDiffractionMaterialBinding> (
        new AdmittedDiffractionMaterialBinding (
            *binding, std::move(productProgram)));
    return std::shared_ptr<const AdmittedDiffractionMaterialBinding> (
        new AdmittedDiffractionMaterialBinding (
            *productBinding, request.plan, request.productDigest));
}

FixtureSceneRenderer::FixtureSceneRenderer (arbitgpu::NativeFixtureSceneBackend& backend) noexcept
    : backend_ (backend)
{
}

bool FixtureSceneRenderer::renderPreview (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    RenderDimensions dimensions, std::string_view backendCapability,
    RenderedFrame& output, std::string& error)
{
    return render (snapshot, dimensions, backendCapability,
                   RenderUse::Preview, {}, {}, {}, {}, output, error);
}

bool FixtureSceneRenderer::renderPreview (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return renderPreview (snapshot, material, {}, dimensions,
                          backendCapability, output, error);
}

bool FixtureSceneRenderer::renderPreview (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
    surfacematerial::MaterialEvaluationInputs runtimeInputs,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return renderPreview(snapshot, material, runtimeInputs, {}, dimensions,
                         backendCapability, output, error);
}

bool FixtureSceneRenderer::renderPreview (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
    surfacematerial::MaterialEvaluationInputs runtimeInputs,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return render (snapshot, dimensions, backendCapability,
                   RenderUse::Preview, material, {}, runtimeInputs, sceneInputs, output, error);
}

bool FixtureSceneRenderer::renderPreview (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& material,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return render (snapshot, dimensions, backendCapability,
                   RenderUse::Preview, {}, material, {}, sceneInputs, output, error);
}

bool FixtureSceneRenderer::renderExport (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    RenderDimensions dimensions, std::string_view backendCapability,
    RenderedFrame& output, std::string& error)
{
    return render (snapshot, dimensions, backendCapability,
                   RenderUse::Export, {}, {}, {}, {}, output, error);
}

bool FixtureSceneRenderer::renderExport (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return renderExport (snapshot, material, {}, dimensions,
                         backendCapability, output, error);
}

bool FixtureSceneRenderer::renderExport (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
    surfacematerial::MaterialEvaluationInputs runtimeInputs,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return renderExport(snapshot, material, runtimeInputs, {}, dimensions,
                        backendCapability, output, error);
}

bool FixtureSceneRenderer::renderExport (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
    surfacematerial::MaterialEvaluationInputs runtimeInputs,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return render (snapshot, dimensions, backendCapability,
                   RenderUse::Export, material, {}, runtimeInputs, sceneInputs, output, error);
}

bool FixtureSceneRenderer::renderExport (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& material,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    RenderDimensions dimensions,
    std::string_view backendCapability, RenderedFrame& output, std::string& error)
{
    return render (snapshot, dimensions, backendCapability,
                   RenderUse::Export, {}, material, {}, sceneInputs, output, error);
}

bool FixtureSceneRenderer::render (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    RenderDimensions dimensions, std::string_view backendCapability,
    RenderUse use,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& material,
    const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& diffractionMaterial,
    surfacematerial::MaterialEvaluationInputs runtimeInputs,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    RenderedFrame& output, std::string& error)
{
    if (snapshot == nullptr)
    {
        error = "native fixture rendering requires an immutable scene snapshot";
        return false;
    }
    if (backendCapability != kNativeGpuCapability)
    {
        error = "native fixture rendering requires backendCapability native-gpu";
        return false;
    }
    if (Visual3DSceneRenderContract::kBackendRequirement
            != Visual3DSceneBackendRequirement::NativeGpu
        || Visual3DSceneRenderContract::kAllowsCpuImageFallback)
    {
        error = "Visual3DScene production rendering contract does not admit this path";
        return false;
    }
    if (dimensions.width == 0 || dimensions.height == 0
        || dimensions.width > kMaxExtent || dimensions.height > kMaxExtent
        || static_cast<std::uint64_t> (dimensions.width) * dimensions.height > kMaxPixels)
    {
        error = "native fixture render dimensions exceed the bounded extent";
        return false;
    }

    const auto validation = validateVisual3DScene (*snapshot);
    if (! validation.valid())
    {
        error = "Visual3DScene validation failed with code "
            + std::to_string (static_cast<unsigned> (validation.code))
            + " at record " + std::to_string (validation.recordIndex);
        return false;
    }
    if (! admitFixtureSubset (*snapshot, error))
        return false;

    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> nativeMaterial;
    if (material != nullptr && diffractionMaterial != nullptr)
    {
        error = "native fixture rendering admits one exact material program";
        return false;
    }
    if (material != nullptr)
    {
        if (material->scene_ != snapshot->id
            || material->object_ != snapshot->objects[0].id
            || material->sceneSnapshot_ != snapshot
            || material->nativeProgram_ == nullptr)
        {
            error = "native surface material admission does not match the exact scene snapshot";
            return false;
        }
        nativeMaterial = material->nativeProgram_;
        if (nativeMaterial->baseColorSource
                == arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix
            && (! std::isfinite (runtimeInputs.timeSeconds)
                || std::abs (runtimeInputs.timeSeconds)
                    > surfacematerial::kMaximumEvaluationMagnitude))
        {
            error = "native surface material time input is non-finite or out of bounds";
            return false;
        }
    }

    if (diffractionMaterial != nullptr)
    {
        if (diffractionMaterial->scene_ != snapshot->id
            || diffractionMaterial->object_ != snapshot->objects[0].id
            || diffractionMaterial->sceneSnapshot_ != snapshot
            || diffractionMaterial->nativeProgram_ == nullptr)
        {
            error = "native diffraction material admission does not match the exact scene snapshot";
            return false;
        }
        nativeMaterial = diffractionMaterial->nativeProgram_;
    }

    const auto info = backend_.info();
    if (! info.available || info.backend.empty())
    {
        error = info.error.empty()
            ? "native fixture GPU backend is unavailable" : info.error;
        return false;
    }
    const bool materialBackendMatches = nativeMaterial == nullptr
        || (nativeMaterial->backend == arbitgpu::NativeFixtureMaterialBackend::OpenGl
            && info.backend == "opengl")
        || (nativeMaterial->backend == arbitgpu::NativeFixtureMaterialBackend::Metal
            && info.backend == "metal");
    if (! materialBackendMatches)
    {
        error = "native material program does not match the physical fixture backend";
        return false;
    }

    auto cachedSnapshot = cache_.snapshot.lock();
    // Prepared resources retain this exact immutable admission owner. Equal
    // digests from a separately admitted owner are not a cache hit.
    const bool exactMaterialHit = nativeMaterial == cache_.materialProgram;
    const bool exactCacheHit = cachedSnapshot == snapshot && cache_.native != nullptr
                            && exactMaterialHit;
    if (! exactCacheHit)
    {
        auto preparation = backend_.prepare (snapshot, nativeMaterial);
        if (! preparation.prepared || preparation.resources == nullptr)
        {
            error = preparation.error.empty()
                ? "native fixture GPU backend did not prepare static resources"
                : std::move (preparation.error);
            return false;
        }
        if (preparation.resources->backend() != info.backend)
        {
            error = "native fixture GPU backend returned incompatible static resources";
            return false;
        }

        cache_.snapshot = snapshot;
        cache_.materialProgram = nativeMaterial;
        cache_.native = std::move (preparation.resources);
        cache_.footprint = preparation.stats;
    }

    sceneInputs.timeSeconds = runtimeInputs.timeSeconds;
    auto submission = backend_.render (
        snapshot, cache_.native, dimensions.width, dimensions.height, sceneInputs);
    if (! submission.rendered || submission.frame == nullptr)
    {
        error = submission.error.empty()
            ? "native fixture GPU backend did not return a rendered frame"
            : std::move (submission.error);
        return false;
    }
    if (submission.frame->backend() != info.backend
        || submission.frame->width() != dimensions.width
        || submission.frame->height() != dimensions.height
        || submission.frame->colorImageHandle() == 0
        || submission.frame->colorTextureViewHandle() == 0)
    {
        error = "native fixture GPU backend returned an incompatible frame";
        return false;
    }

    RenderedFrame rendered;
    rendered.use = use;
    rendered.sceneId = snapshot->id;
    rendered.dimensions = dimensions;
    rendered.nativeFrame = std::move (submission.frame);
    rendered.stats = submission.stats;
    rendered.stats.vertexBytes = cache_.footprint.vertexBytes;
    rendered.stats.indexBytes = cache_.footprint.indexBytes;
    rendered.stats.textureBytes = cache_.footprint.textureBytes;
    rendered.stats.staticUploadCount = exactCacheHit
        ? 0u : cache_.footprint.staticUploadCount;
    rendered.stats.textureUploadCount = exactCacheHit
        ? 0u : cache_.footprint.textureUploadCount;
    rendered.stats.reusedStaticResources = exactCacheHit;
    rendered.stats.materialProgramUploadCount
        = nativeMaterial != nullptr && ! exactCacheHit ? 1u : 0u;
    rendered.stats.reusedMaterialProgram = nativeMaterial != nullptr && exactCacheHit;
    output = std::move (rendered);
    error.clear();
    return true;
}

} // namespace videorender::fixture3d
