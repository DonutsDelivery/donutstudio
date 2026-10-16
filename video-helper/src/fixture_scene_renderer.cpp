#include "fixture_scene_renderer.h"
#include "diffractive_foil_admission.h"
#include "diffraction_material_execution.h"
#include "surface_material_binding_admission.h"
#include "gpu_backend/fixture_vertex_modifier_shader.h"
#include "gpu_backend/fixture_glass_caustics.h"
#if ARBIT_HAVE_METAL_BACKEND
#include "gpu_backend/metal_glass_caustic_consumer.h"
#elif ARBIT_HAVE_OPENGL_SDF_BACKEND
#include "gpu_backend/fixture_glass_caustics_opengl.h"
#endif
#include "../../shared/VisualImportedSceneRenderOperationContract.h"
#include "loop_step_probe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace videorender::fixture3d
{
namespace
{
thread_local std::uint64_t previewPendingResults = 0;

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
                            bool graphFrame,
                            std::string& error)
{
    using namespace surfacematerial;
    using NativeProgram = arbitgpu::NativeFixtureSurfaceMaterialProgram;
    const auto* targetObject = visual3d_detail::findById(scene.objects, scene.objectCount, object);
    if (! object.isValid() || targetObject == nullptr)
    {
        error = "native surface material binding must target the exact admitted object";
        return {};
    }
    if (bindingDigest.empty())
    {
        error = "native surface material binding requires an exact admitted digest";
        return {};
    }
    if (! program.hasSurfaceProgram())
    {
        error = "native surface material execution requires one surface program";
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
    const auto* targetMaterial = visual3d_detail::findById(scene.materials, scene.materialCount, targetObject->material);
    if (targetMaterial == nullptr)
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
    const auto* dispersion = program.capabilities().surfaceIrVersion == kSpectralWireVersion
        ? instructionFor(OutputSemantic::Dispersion) : nullptr;
    if (dispersion && dispersion->literal[0] != 0.0f && program.transport() != TransportMode::ClosedVolume)
    {
        error = "nonzero dispersion requires explicit closed glass volume transport";
        return {};
    }
    if (materialId->unsignedLiteral != targetObject->material.value)
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
    // The native draw blends only a thin-surface program, so a translucent value
    // on any other transport would render opaque. Refuse it instead.
    const float authoredOpacity = opacity->literal[0];
    if (! std::isfinite (authoredOpacity) || authoredOpacity < 0.0f || authoredOpacity > 1.0f)
    {
        error = "native surface material opacity must be finite and within [0, 1]";
        return {};
    }
    if (authoredOpacity < 1.0f && program.transport() != TransportMode::ThinSurface)
    {
        error = "native surface material opacity below one requires thin-surface transport";
        return {};
    }

    auto native = std::make_shared<NativeProgram>();
    native->backend = nativeBackend;
    native->kind = arbitgpu::NativeFixtureMaterialKind::SurfacePbr;
    native->object = object;
    native->bindingDigest = bindingDigest;
    native->programIdentity = program.programIdentity();
    if (program.hasVertexProgram())
    {
        if (scene.objectCount != 1)
        {
            error = "Native vertex material execution requires one selected imported object";
            return {};
        }
        if (arbitgpu::fixtureVertexModifierShader ("// ARBIT_VERTEX_MODIFIER", &program,
                program.capabilities().backend.target, error).empty()) return {};
        native->vertexProgram = std::make_shared<const videohelper::materialprogram::MaterialProgramRecord> (program);
    }
    native->baseColorSource = texturedBaseColor
        ? (graphFrame ? NativeProgram::BaseColorSource::GraphFrameSrgbTexture
                      : NativeProgram::BaseColorSource::ImportedSrgbTexture)
        : (timeMixedBaseColor
            ? NativeProgram::BaseColorSource::TimeLinearMix
            : NativeProgram::BaseColorSource::ConstantLinear);
    if (texturedBaseColor && !graphFrame)
    {
        if (! targetMaterial->baseColorTexture.isValid())
        {
            error = "native surface material imported texture binding has no decoded base-color texture";
            return {};
        }
        const auto* source = visual3d_detail::findById (
            scene.textures, scene.textureCount, targetMaterial->baseColorTexture);
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
        texture.texels.assign (scene.textureTexels.begin() + source->firstTexel,
                              scene.textureTexels.begin() + source->firstTexel + texelCount);
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
        transmission->literal[0], ior->literal[0], clearcoat->literal[0], dispersion ? dispersion->literal[0] : 0.0f
    };
    native->parameters.identifiers[0] = materialId->unsignedLiteral;
    native->parameters.identifiers[1] = static_cast<std::uint32_t>(program.transport());
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
    std::string& error, bool exactComposedTarget,
    arbitgpu::NativeFixtureDiffractionConsumer consumer=arbitgpu::NativeFixtureDiffractionConsumer::FiniteLobe,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest* discreteRequest=nullptr)
{
    const auto backend = nativeMaterialBackend (target);
    if (backend == arbitgpu::NativeFixtureMaterialBackend::Invalid)
    {
        error = "native diffraction material execution requires an OpenGL or Metal target";
        return {};
    }
    const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount, admitted.object());
    if (admitted.scene() != scene.id || !object || object->indexCount == 0
        || (!exactComposedTarget && admitted.object() != scene.objects[0].id))
    {
        error = "native diffraction material binding must target the exact admitted object";
        return {};
    }
    if(consumer==arbitgpu::NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders) {
        if(!discreteRequest||discreteRequest->object!=admitted.object()||discreteRequest->scene!=admitted.scene()
            ||discreteRequest->sceneRevision!=admitted.sceneRevision()||discreteRequest->structuralRevision!=admitted.structuralRevision()
            ||discreteRequest->evaluationRevision!=admitted.evaluationRevision()||discreteRequest->materialRevision!=admitted.materialRevision()
            ||discreteRequest->structuralDigest!=admitted.material().structuralDigest()
            ||!diffractionmaterial::sameLightingDescription(discreteRequest->lighting,admitted.lighting().description()))
        {error="discrete grating compiler requires its exact original admitted request";return {};}
        return arbitgpu::glasscaustics::compileDiscreteGrating(scene,*discreteRequest,backend,error);
    }
    if(consumer!=arbitgpu::NativeFixtureDiffractionConsumer::FiniteLobe)
    {error="native diffraction compiler consumer is invalid";return {};}
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

    if (! hasValidUvDerivatives (scene, *object))
    {
        error = "native diffraction surface rendering requires non-degenerate UV derivatives on every triangle";
        return {};
    }
    if (description.grooveField.mode != diffractionmaterial::GrooveFieldMode::Constant
        && ! hasUnitUvDomain(scene, *object))
    {
        error = "native spatial diffraction rendering requires material UVs in the admitted unit domain";
        return {};
    }

    std::string validationError;
    const auto& lighting = admitted.lighting().description();
    if (lighting.version == 2
        && std::any_of(lighting.paths[0].incident.radiance.begin(),
                       lighting.paths[0].incident.radiance.end(),
                       [](float value) { return value > 0.0f; })
        && (scene.lightCount != 1
            || scene.lights[0].kind != SceneLightKind::Directional
            || scene.lights[0].color.x != 1.0f || scene.lights[0].color.y != 1.0f
            || scene.lights[0].color.z != 1.0f))
    {
        error = "spectral environment direct lighting requires one white Directional Light; set Direct Intensity to zero for environment-only rendering";
        return {};
    }
    for (std::size_t pathIndex = 0; pathIndex < lighting.pathCount; ++pathIndex)
    {
        auto validationLight = lighting.paths[pathIndex].incident;
        if (lighting.version != 1) validationLight.direction = { 0.0f, 0.0f, 1.0f };
        if (! diffractionmaterial::physicalcheckpoint::validate (
                admitted.material(), validationLight,
                target == videohelper::materialprogram::BackendTarget::OpenGl
                    ? "opengl" : "metal",
                validationError))
        {
            error = "native diffraction surface admission rejected at lighting path "
                + std::to_string(pathIndex) + ": " + validationError;
            return {};
        }
    }

    auto native = arbitgpu::glasscaustics::lowerNativeDiffractionMaterial(admitted,object->material,backend);
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
    const bool graphFrame = hasGraphFrameInput(request.binding);
    if (request.version != requestVersion(graphFrame, request.program.spectralReflectance.has_value(),request.roughGlassScatteringVersion)
        || !validRoughGlassVersion(request))
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
        || !HarmonicMIDI::grid::validateVisual3DScene(scene).valid()
        || HarmonicMIDI::grid::visual3d_detail::findById(scene.objects, scene.objectCount, request.binding.object) == nullptr
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
    description.version = requestVersion(graphFrame,request.program.spectralReflectance.has_value());
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
    const auto resolution = admittedBindings->resolve (request.binding.object);
    if (! resolution || resolution->kind != ResolutionKind::ObjectOverride
        || resolution->binding == nullptr || resolution->program == nullptr
        || resolution->program->program == nullptr)
    {
        error = "admitted material does not resolve the exact imported scene object override";
        return {};
    }

    std::optional<videohelper::vertexmodifier::AdmittedVertexModifierIr> vertex;
    if (request.vertexModifier)
    {
        vertex = videohelper::vertexmodifier::admitVertexModifierIr (
            *request.vertexModifier, {}, diagnostic);
        if (! vertex)
        {
            error = "Vertex modifier admission failed: " + diagnostic;
            return {};
        }
    }
    auto compiled = videohelper::materialprogram::compileMaterialProgram (
        resolution->program->program.get(), vertex ? &*vertex : nullptr, target, diagnostic);
    if (! compiled)
    {
        error = "surface material compilation failed: " + diagnostic;
        return {};
    }
    auto immutableCompiled
        = std::make_shared<const videohelper::materialprogram::MaterialProgramRecord> (
            std::move (*compiled));
    const auto bindingDigest = admittedBindings->digest()
        + (request.roughGlassScatteringVersion==2 ? ":rough-glass:2" : std::string {})
        + (vertex ? ":vertex:" + immutableCompiled->programIdentity() : std::string {});
    auto native = admitNativeSurfaceMaterial (
        scene, request.binding.object, bindingDigest,
        *immutableCompiled, graphFrame, diagnostic);
    if (native == nullptr)
    {
        error = std::move (diagnostic);
        return {};
    }

    {
        auto owned = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*native);
        if (graphFrame) owned->frameEndpoint = request.binding.textures.front().graphFrame;
        owned->spectralReflectance = immutableProgram->spectralReflectance();
        owned->roughGlassScatteringVersion = request.roughGlassScatteringVersion;
        owned->authoredScene = scene.id;
        owned->authoredMaterial = visual3d_detail::findById(
            scene.objects, scene.objectCount, request.binding.object)->material;
        owned->sceneRevision = request.sceneRevision;
        owned->structuralRevision = request.structuralRevision;
        owned->evaluationRevision = request.evaluationRevision;
        owned->materialRevision = request.programRevision;
        native = std::move(owned);
    }

    error.clear();
    return std::shared_ptr<const AdmittedSurfaceMaterialBinding> (
        new AdmittedSurfaceMaterialBinding (
            scene.id, request.binding.object, request.sceneRevision,
            request.structuralRevision, request.evaluationRevision,
            request.programRevision, bindingDigest, sceneSnapshot,
            std::move (native)));
}

std::shared_ptr<const AdmittedSurfaceMaterialBinding> admitSurfaceMaterialCollection (
    const std::shared_ptr<const Visual3DScene>& scene,
    const geometrysurfacematerial::ObjectPrograms& programs,
    const videowire::geometry::RuntimeFieldEvaluation& evaluation,
    videohelper::materialprogram::BackendTarget target, std::string& error)
{
    if (!scene || programs.empty()) { error = "Surface collection requires an owned scene and exact object programs"; return {}; }
    const auto& first = programs.front().program.material;
    visualimportedscenerender::Request request;
    request.sourceStableId = scene->id.value;
    request.renderStableId = request.sourceStableId == 1 ? 2 : 1;
    request.sceneSnapshot = scene;
    request.structuralRevision = first.structuralRevision;
    request.evaluationRevision = first.evaluationRevision;
    request.surfacePrograms = programs;
    if (!visualimportedscenerender::valid(request))
    { error = "Surface collection object, material, snapshot or Field identity is invalid"; return {}; }
    std::shared_ptr<arbitgpu::NativeFixtureSurfaceMaterialProgram> root;
    std::ostringstream evaluated;
    evaluated.imbue(std::locale::classic());
    for (const auto& object : programs)
    {
        const auto binding = admitSurfaceMaterialBinding(scene, object.program.material, target, error);
        if (!binding) return {};
        auto native = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*binding->nativeProgram());
        for (const auto& field : object.program.fields)
            if (!surfacematerialfield::evaluate(field, evaluation, object.instanceIndex,
                    native->parameters, native->timeMixEndColor, error)) return {};
        for (std::size_t channel = 0; channel < 3; ++channel)
        {
            native->parameters.baseColorMetallic[channel] *= object.appearance.color[channel];
            native->timeMixEndColor[channel] *= object.appearance.color[channel];
            native->parameters.emissionRoughness[channel] += object.appearance.emission[channel];
        }
        native->parameters.normalOpacity[3] *= object.appearance.color[3];
        if (object.appearance.metallic >= 0) native->parameters.baseColorMetallic[3] = object.appearance.metallic;
        if (object.appearance.roughness >= 0) native->parameters.emissionRoughness[3] = object.appearance.roughness;
        // Canonical bytes of the actual draw parameters distinguish sampled
        // Fields, while time-only shader inputs keep the same prepared program.
        for (const auto value : native->parameters.baseColorMetallic) evaluated << visualimportedscenerender::detail::floatBits(value) << ' ';
        for (const auto value : native->parameters.emissionRoughness) evaluated << visualimportedscenerender::detail::floatBits(value) << ' ';
        for (const auto value : native->parameters.normalOpacity) evaluated << visualimportedscenerender::detail::floatBits(value) << ' ';
        for (const auto value : native->parameters.transmissionIorClearcoat) evaluated << visualimportedscenerender::detail::floatBits(value) << ' ';
        for (const auto value : native->timeMixEndColor) evaluated << visualimportedscenerender::detail::floatBits(value) << ' ';
        if (!root) root = std::move(native); else root->objectPrograms.push_back(std::move(native));
    }
    if (!arbitgpu::validNativeSurfaceObjectPrograms(*root, *scene, root->backend))
    { error = "Surface collection exceeds the native exact-object shader subset"; return {}; }
    const auto digest = videohelper::sha256Text(visualimportedscenerender::detail::encodeSurfacePrograms(programs) + evaluated.str());
    error.clear();
    return std::shared_ptr<const AdmittedSurfaceMaterialBinding>(new AdmittedSurfaceMaterialBinding(
        scene->id, first.binding.object, first.sceneRevision, first.structuralRevision,
        first.evaluationRevision, first.programRevision, digest, scene, std::move(root)));
}

std::shared_ptr<const AdmittedDiffractionMaterialBinding>
admitDiffractionMaterialBinding (
    const std::shared_ptr<const Visual3DScene>& sceneSnapshot,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& request,
    videohelper::materialprogram::BackendTarget target,
    std::string& error, bool exactComposedTarget, arbitgpu::NativeFixtureDiffractionConsumer consumer)
{
    if (consumer == arbitgpu::NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders && !exactComposedTarget)
    { error = "discrete grating admission requires an exact composed target"; return {}; }
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
        *sceneSnapshot, *admitted, target, diagnostic, exactComposedTarget, consumer, &request);
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
            spatial->bindingDigest, *spatial->diffractionLightingAdmission, 0, 1.0f, {}, false, spatial.get());
    }
    if (request.graphFrame)
    {
        spatial->baseColorSource = arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::GraphFrameSrgbTexture;
        spatial->frameEndpoint = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint { request.graphFrame->node, request.graphFrame->port };
        spatial->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            spatial->bindingDigest, *spatial->diffractionLightingAdmission, 0, 1.0f, {}, true, spatial.get());
    }
    native = std::move(spatial);
    const auto bindingIdentity = request.graphFrame
        ? native->programIdentity + ":" + std::to_string(request.graphFrame->node)
            + ":" + std::to_string(request.graphFrame->port)
        : request.structuralDigest;

    error.clear();
    return std::shared_ptr<const AdmittedDiffractionMaterialBinding> (
        new AdmittedDiffractionMaterialBinding (
            *admitted, bindingIdentity, sceneSnapshot, std::move (native), nullptr, {}, request, consumer));
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
    if (plan.lighting.version == 2)
        lowered.version = diffractionmaterialbinding::kEnvironmentWireVersion;
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

bool FixtureSceneRenderer::renderMixed (
    const std::shared_ptr<const Visual3DScene>& snapshot,
    const std::shared_ptr<const AdmittedSurfaceMaterialBinding>& surfaces,
    const std::shared_ptr<const AdmittedDiffractionMaterialBinding>& diffraction,
    arbitgpu::NativeFixtureSceneRuntimeInputs sceneInputs,
    RenderDimensions dimensions, RenderUse use, std::string_view backendCapability,
    RenderedFrame& output, std::string& error)
{
    if (sceneInputs.glassCaustic)
        output = {};
    if (!surfaces || !diffraction) { error = "mixed material rendering requires both exact admissions"; return false; }
    surfacematerial::MaterialEvaluationInputs inputs;
    inputs.timeSeconds = sceneInputs.timeSeconds;
    return render(snapshot, dimensions, backendCapability, use, surfaces, diffraction, inputs, sceneInputs, output, error);
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
    // A selected caustic draw cannot leave an earlier ordinary or caustic frame visible.
    if (sceneInputs.glassCaustic)
        output = {};
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
    if (material != nullptr)
    {
        if (material->scene_ != snapshot->id
            || visual3d_detail::findById(snapshot->objects, snapshot->objectCount, material->object_) == nullptr
            || material->sceneSnapshot_ != snapshot
            || material->nativeProgram_ == nullptr)
        {
            error = "native surface material admission does not match the exact scene snapshot";
            return false;
        }
        nativeMaterial = material->nativeProgram_;
        if (arbitgpu::nativeSurfaceUsesTime(*nativeMaterial)
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
            || (!material && diffractionMaterial->object_ != snapshot->objects[0].id)
            || diffractionMaterial->sceneSnapshot_ != snapshot
            || diffractionMaterial->nativeProgram_ == nullptr)
        {
            error = "native diffraction material admission does not match the exact scene snapshot";
            return false;
        }
        nativeMaterial = diffractionMaterial->nativeProgram_;
    }

    if (material && diffractionMaterial)
    {
        if (material->sceneRevision_ != diffractionMaterial->sceneRevision_
            || material->structuralRevision_ != diffractionMaterial->structuralRevision_
            || material->evaluationRevision_ != diffractionMaterial->evaluationRevision_)
        { error = "mixed material admissions require identical scene and evaluation revisions"; return false; }
        if (cache_.mixedSurfaces == material->nativeProgram_ && cache_.mixedDiffraction == nativeMaterial)
            nativeMaterial = cache_.materialProgram;
        else
        {
            auto mixed = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*nativeMaterial);
            auto first = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*material->nativeProgram_);
            first->objectPrograms.clear();
            mixed->objectPrograms.push_back(std::move(first));
            mixed->objectPrograms.insert(mixed->objectPrograms.end(),
                material->nativeProgram_->objectPrograms.begin(), material->nativeProgram_->objectPrograms.end());
            if (!arbitgpu::validNativeFixtureMaterialPrograms(*mixed, *snapshot, mixed->backend,
                    backend_.supportsAuthoredSpectralReceiverTransport(), diffractionMaterial->consumer_))
            { error = "mixed material programs overlap or have invalid exact-object identities"; return false; }
            nativeMaterial = std::move(mixed);
        }
    }

    if (diffractionMaterial && diffractionMaterial->consumer_ == arbitgpu::NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders)
    {
        if (!material || !sceneInputs.glassCaustic || !diffractionMaterial->authoredRequest_
            || !sceneInputs.diffractionParameters.empty()
            || !visualimportedscenerender::validGlassCausticScene(*sceneInputs.glassCaustic, *snapshot))
        { error = "discrete grating requires its selected static smooth glass collection"; return false; }
        const arbitgpu::smoothglass::DistinctGratingAdmission target {
            diffractionMaterial->object_, diffractionMaterial->materialIR_ };
        arbitgpu::smoothglass::Prepared prepared;
        if (!arbitgpu::smoothglass::prepare(snapshot, nativeMaterial, {}, prepared, error, {}, &target)) return false;
        if (!prepared.enabled || prepared.object != sceneInputs.glassCaustic->glassObject
            || prepared.material != sceneInputs.glassCaustic->glassMaterial)
        { error = "discrete grating does not retain the selected closed glass owner"; return false; }
    }

    if (!sceneInputs.diffractionParameters.empty())
    {
        if (!diffractionMaterial || !diffractionMaterial->authoredRequest_ || diffractionMaterial->productPlan_)
        { error = "Diffraction modulation requires one graph-authored physical material"; return false; }
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest evaluated;
        if (!diffractionmaterialbinding::applyRuntimeParameters(*diffractionMaterial->authoredRequest_,
                sceneInputs.diffractionParameters, evaluated, error)) return false;
        const auto physical = diffractionmaterial::admit(evaluated.material, error);
        if (!physical) return false;
        evaluated.structuralDigest = physical->structuralDigest();
        if (evaluated.spatialFoil)
        {
            const auto foil = diffractivefoil::admit(*evaluated.spatialFoil, error);
            if (!foil) return false;
            evaluated.structuralDigest = foil->structuralDigest();
        }
        const auto target = nativeMaterial->backend == arbitgpu::NativeFixtureMaterialBackend::Metal
            ? videohelper::materialprogram::BackendTarget::Metal : videohelper::materialprogram::BackendTarget::OpenGl;
        const auto admission = admitDiffractionMaterialBinding(snapshot, evaluated, target, error, material != nullptr, diffractionMaterial->consumer_);
        if (!admission) return false;
        sceneInputs.diffractionSourceProgram = nativeMaterial;
        auto evaluatedProgram = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*admission->nativeProgram());
        evaluatedProgram->objectPrograms = nativeMaterial->objectPrograms;
        sceneInputs.diffractionEvaluatedProgram = std::move(evaluatedProgram);
    }
    else if (sceneInputs.diffractionSourceProgram || sceneInputs.diffractionEvaluatedProgram)
    { error = "Diffraction runtime admission must be derived from this draw's scalar parameters"; return false; }

    const auto info = backend_.info();
    const bool expectsFrame = nativeMaterial && arbitgpu::nativeSurfaceUsesFrame(*nativeMaterial);
    if (expectsFrame != static_cast<bool>(sceneInputs.materialFrameTexture)
        || (expectsFrame && (!arbitgpu::validMaterialFrameTexture(sceneInputs.materialFrameTexture)
            || sceneInputs.materialFrameTexture->backend() != info.backend)))
    {
        error = "Native material Frame draw requires its exact owned texture";
        return false;
    }
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

    sceneInputs.timeSeconds = runtimeInputs.timeSeconds;
    std::optional<arbitgpu::glasscaustics::AdmittedBranchPolicy> causticPolicy;
    std::optional<arbitgpu::glasscaustics::MultipleScatterPolicy> multipleScatterPolicy;
    if (sceneInputs.glassCaustic)
    {
        const auto& selection = *sceneInputs.glassCaustic;
        const auto* selectedMaterial = arbitgpu::nativeSurfaceForObject(nativeMaterial.get(), selection.glassObject);
        if (!visualimportedscenerender::validGlassCausticScene(selection, *snapshot)
            || !selectedMaterial || selectedMaterial->authoredScene != snapshot->id
            || selectedMaterial->authoredMaterial != selection.glassMaterial
            || selectedMaterial->parameters.identifiers[0] != selection.glassMaterial.value
            || selectedMaterial->parameters.identifiers[1]
                != static_cast<std::uint32_t>(surfacematerial::TransportMode::ClosedVolume)
            || !std::isfinite(selectedMaterial->parameters.emissionRoughness[3])
            || selectedMaterial->parameters.emissionRoughness[3] < 0.0f
            || selectedMaterial->parameters.emissionRoughness[3] > 1.0f)
        {
            error = "Glass caustic selection does not match an exact authored closed-volume material";
            return false;
        }
        if (selectedMaterial->parameters.emissionRoughness[3] > 0.0f)
            multipleScatterPolicy.emplace();
        // Material Fields have already consumed the caller's retained score owner.
        // Only an explicit mapping authorizes carrying it into a native instance draw.
        if (!sceneInputs.noteInstanceMapping)
            sceneInputs.canonicalBlockCFrame.reset();
        if (!sceneInputs.linearColor || !arbitgpu::validFixtureRuntimeInputs(sceneInputs)
            || sceneInputs.imageOutput != renderpassoutput::Output::Color
            || sceneInputs.rawExportMask != 0 || sceneInputs.passComposite || sceneInputs.passProgram
            || diffractionMaterial || !sceneInputs.diffractionParameters.empty()
            || sceneInputs.diffractionSourceProgram || sceneInputs.diffractionEvaluatedProgram
            || sceneInputs.noteInstanceMapping)
        {
            error = "Glass caustics require scene-linear Color without pass composition, diffraction or note instancing";
            return false;
        }
        const auto& policy = selection.policy;
        const auto admittedPolicy = arbitgpu::glasscaustics::AdmittedBranchPolicy::admit(
            {policy.version, policy.maximumUnresolvedFraction, policy.minimumBranchFraction,
             policy.maximumInterfaces, policy.maximumVisits, policy.maximumPending}, error);
        if (!admittedPolicy)
            return false;
        causticPolicy.emplace(*admittedPolicy);
#if ARBIT_HAVE_METAL_BACKEND
        const bool supportsCaustics = info.backend == "metal";
#elif ARBIT_HAVE_OPENGL_SDF_BACKEND && !defined(__APPLE__)
        const bool supportsCaustics = info.backend == "opengl";
#else
        const bool supportsCaustics = false;
#endif
        if (!supportsCaustics)
        {
            error = "Glass caustics have no native consumer in this renderer backend";
            return false;
        }
    }

    auto cachedSnapshot = cache_.snapshot.lock();
    // Prepared resources retain this exact immutable admission owner. Equal
    // digests from a separately admitted owner are not a cache hit.
    const bool exactMaterialHit = nativeMaterial == cache_.materialProgram;
    const bool exactCacheHit = cachedSnapshot == snapshot && cache_.native != nullptr
                            && exactMaterialHit;
    if (! exactCacheHit)
    {
        const auto prepareUse = use == RenderUse::Preview
            ? arbitgpu::NativeFixturePrepareUse::Preview
            : arbitgpu::NativeFixturePrepareUse::Blocking;
        // A cache miss builds the static GPU resources: buffers, textures and
        // the scene shader and pipelines. The backend's own scopes split it.
        std::optional<loopstep::Scope> prepareScope;
        prepareScope.emplace ("advance:imported-scene:fixture-prepare");
        auto preparation = backend_.prepare (snapshot, nativeMaterial, prepareUse);
        prepareScope.reset();
        if (preparation.pending)
        {
            // Pending is a retry state, not a failure. No cache or static resource changes.
            ++previewPendingResults;
            if (use != RenderUse::Preview)
            {
                error = "native fixture GPU backend returned pending for a blocking prepare";
                return false;
            }
            output = {};
            output.use = use;
            output.pending = true;
            error.clear();
            return true;
        }
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
        cache_.mixedSurfaces = material && diffractionMaterial ? material->nativeProgram_ : nullptr;
        cache_.mixedDiffraction = material && diffractionMaterial ? diffractionMaterial->nativeProgram_ : nullptr;
        cache_.native = std::move (preparation.resources);
        cache_.footprint = preparation.stats;
    }

    arbitgpu::NativeFixtureSceneSubmission submission;
    // Encodes, submits and (for checked paths) waits on the frame's GPU work.
    std::optional<loopstep::Scope> submitScope;
    submitScope.emplace (sceneInputs.glassCaustic ? "advance:imported-scene:fixture-caustic-submit"
                                                  : "advance:imported-scene:fixture-submit");
    if (sceneInputs.glassCaustic)
    {
        const auto& selection = *sceneInputs.glassCaustic;
#if ARBIT_HAVE_METAL_BACKEND
        arbitgpu::glasscaustics::Prepared prepared;
        if (!arbitgpu::prepareMetalGlassCaustic(cache_.native, selection.sourceLight,
                *causticPolicy, prepared, error))
            return false;
        const auto* selectedMaterial = arbitgpu::nativeSurfaceForObject(
            prepared.glass.sourceMaterial.get(), selection.glassObject);
        // The backend owns a deep immutable material snapshot derived during preparation.
        const auto* authoredMaterial = arbitgpu::nativeSurfaceForObject(nativeMaterial.get(), selection.glassObject);
        if (prepared.glass.sourceScene != snapshot || prepared.glass.object != selection.glassObject
            || !selectedMaterial || !authoredMaterial
            || selectedMaterial->authoredMaterial != selection.glassMaterial
            || selectedMaterial->bindingDigest != authoredMaterial->bindingDigest
            || selectedMaterial->programIdentity != authoredMaterial->programIdentity
            || selectedMaterial->sceneRevision != authoredMaterial->sceneRevision
            || selectedMaterial->structuralRevision != authoredMaterial->structuralRevision
            || selectedMaterial->evaluationRevision != authoredMaterial->evaluationRevision
            || prepared.glass.roughness != authoredMaterial->parameters.emissionRoughness[3])
        {
            error = "Metal caustic preparation differs from the exact selected native owners";
            return false;
        }
        submission = arbitgpu::renderMetalGlassCausticFrame(snapshot, cache_.native,
            dimensions.width, dimensions.height, sceneInputs, prepared, {},
            multipleScatterPolicy ? &*multipleScatterPolicy : nullptr);
#elif ARBIT_HAVE_OPENGL_SDF_BACKEND && !defined(__APPLE__)
        submission = arbitgpu::renderOpenGlGlassCausticScene(snapshot, cache_.native,
            selection.sourceLight, selection.glassObject, selection.glassMaterial, *causticPolicy,
            {}, dimensions.width, dimensions.height, sceneInputs,
            multipleScatterPolicy ? &*multipleScatterPolicy : nullptr);
#else
        (void)selection;
        error = "Glass caustics have no native consumer in this renderer backend";
        return false;
#endif
    }
    else
        submission = backend_.render (
            snapshot, cache_.native, dimensions.width, dimensions.height, sceneInputs);
    submitScope.reset();
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

    if (sceneInputs.glassCaustic)
    {
        const auto descriptor = submission.frame->colorTextureDescriptor();
        if (!descriptor.complete() || descriptor.backend != info.backend
            || descriptor.width != dimensions.width || descriptor.height != dimensions.height
            || descriptor.imageHandle != submission.frame->colorImageHandle()
            || descriptor.textureViewHandle != submission.frame->colorTextureViewHandle()
            || descriptor.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float
            || !arbitgpu::isLinearSceneColor(descriptor)
            || submission.frame->nativeResourceCacheIdentity()
                != reinterpret_cast<std::uintptr_t>(cache_.native.get()))
        {
            error = "Glass caustic native Color must retain its owned scene-linear frame";
            return false;
        }
    }

    // Preparation owns permanent textures. Selected native caustic submissions
    // report additional frame storage, excluding standard/replaced Color. Ordinary
    // draws retain their explicit rough-scratch accounting contract.
    std::uint64_t textureBytes = 0;
    if (!detail::frameTextureFootprint (cache_.footprint.textureBytes,
            submission.stats, sceneInputs.glassCaustic.has_value(), textureBytes))
    {
        error = "native fixture texture accounting exceeds its byte receipt range";
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
    rendered.stats.textureBytes = textureBytes;
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

std::uint64_t previewPendingResultsOnThisThread() noexcept
{
    return previewPendingResults;
}

} // namespace videorender::fixture3d
