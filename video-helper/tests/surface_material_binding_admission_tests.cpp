#include "../src/surface_material_binding_admission.h"
#include "support/fixture_scene.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

using namespace HarmonicMIDI::grid;
using namespace surfacematerial;
using namespace surfacematerialbinding;

static_assert(!std::is_copy_assignable_v<AdmittedMaterialBindings>);
static_assert(!std::is_move_assignable_v<AdmittedMaterialBindings>);
static_assert(std::is_same_v<decltype(std::declval<const AdmittedMaterialBindings&>().bindings()),
                             const std::vector<MaterialBinding>&>);
static_assert(static_cast<std::uint8_t>(BindingTargetKind::MaterialSlot) == 1);
static_assert(static_cast<std::uint8_t>(TextureSourceKind::VideoResource) == 2);

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

ValueId append(ProgramDescription& program,
               OperationKind kind,
               ValueType type,
               std::initializer_list<ValueId> inputs = {})
{
    Operation operation;
    operation.id = static_cast<ValueId>(program.operations.size() + 1);
    operation.kind = kind;
    operation.resultType = type;
    operation.inputCount = static_cast<std::uint8_t>(inputs.size());
    std::copy(inputs.begin(), inputs.end(), operation.inputs.begin());
    program.operations.push_back(operation);
    return operation.id;
}

void bindOutputs(ProgramDescription& program, ValueId scalar,
                 ValueId vector, ValueId unsignedValue)
{
    for (const auto output : kSurfaceOutputs)
    {
        const auto type = outputType(output);
        program.outputs[static_cast<std::size_t>(output)]
            = type == ValueType::Vec3 ? vector
            : type == ValueType::UInt ? unsignedValue
            : scalar;
    }
}

std::shared_ptr<const AdmittedSurfaceMaterialIR> makeProgram()
{
    ProgramDescription program;
    program.textureSlotCount = 1;

    const auto uv = append(program, OperationKind::Input, ValueType::Vec2);
    program.operations.back().semantic = InputSemantic::TexCoord0;
    const auto sample = append(program, OperationKind::TextureSample2D,
                               ValueType::Vec4, {uv});
    program.operations.back().parameter = 0;
    const auto scalar = append(program, OperationKind::Component,
                               ValueType::Scalar, {sample});
    program.operations.back().parameter = 0;
    const auto beat = append(program, OperationKind::Input, ValueType::Scalar);
    program.operations.back().semantic = InputSemantic::AudioBeat;
    const auto vector = append(program, OperationKind::ComposeVec3,
                               ValueType::Vec3, {scalar, beat, scalar});
    const auto materialId = append(program, OperationKind::UIntConstant,
                                   ValueType::UInt);
    program.operations.back().unsignedLiteral = 19;
    bindOutputs(program, scalar, vector, materialId);

    std::string error;
    auto admitted = surfacematerial::admit(program, error);
    check(admitted.has_value(), "surface material fixture is independently admitted");
    return admitted
        ? std::make_shared<const AdmittedSurfaceMaterialIR>(*admitted) : nullptr;
}

Visual3DScene makeScene()
{
    auto scene = videohelper::fixture3d::makeScene();
    scene.materials[1].id = SceneMaterialId {311};
    scene.materials[1].baseColor = {0.25f, 0.5f, 0.75f};
    scene.materials[1].baseColorTexture = {};
    scene.materials[1].metallic = 0.0f;
    scene.materials[1].roughness = 0.8f;
    scene.materialCount = 2;

    scene.objects[1] = scene.objects[0];
    scene.objects[1].id = SceneObjectId {211};
    scene.objects[1].transform.translation.x += 1.0f;
    scene.objects[2] = scene.objects[0];
    scene.objects[2].id = SceneObjectId {212};
    scene.objects[2].material = scene.materials[1].id;
    scene.objects[2].transform.translation.x += 2.0f;
    scene.objectCount = 3;
    check(validateVisual3DScene(scene).valid(), "binding scene fixture is valid");
    return scene;
}

VideoResourceIdentity videoIdentity()
{
    VideoResourceIdentity identity {};
    identity[0] = 0x91;
    identity[15] = 0x2a;
    return identity;
}

VideoTextureResourceReceipt videoReceipt()
{
    VideoTextureResourceReceipt receipt;
    receipt.identity = videoIdentity();
    receipt.helperGeneration = 23;
    receipt.sourceClipId = 77;
    receipt.sourceStructuralRevision = 41;
    receipt.evaluationRevision = 13;
    receipt.imageDescriptor.nodeId = 9;
    receipt.imageDescriptor.port = 2;
    receipt.imageDescriptor.channels = 1;
    receipt.imageDescriptor.direction = "out";
    receipt.imageDescriptor.carrier = "frame";
    receipt.imageDescriptor.dataType = "image";
    receipt.imageDescriptor.pixelFormat = "rgba16f";
    receipt.imageDescriptor.colorSpace = "linearSRGB";
    receipt.width = 32;
    receipt.height = 16;
    receipt.byteCount = 32 * 16 * 8;
    return receipt;
}

Description makeDescription(const Visual3DScene& scene,
                            const std::shared_ptr<const AdmittedSurfaceMaterialIR>& program)
{
    Description description;
    description.scene = &scene;
    description.sceneRevision = 5;
    description.structuralRevision = 8;
    description.evaluationRevision = 13;
    description.helperGeneration = 23;
    description.programs.push_back({program, 7});
    description.videoResources.push_back(videoReceipt());
    description.modulationInputs.push_back({InputSemantic::AudioBeat, 0.75f, 29});

    MaterialBinding materialSlot;
    materialSlot.targetKind = BindingTargetKind::MaterialSlot;
    materialSlot.materialSlot = scene.materials[0].id;
    materialSlot.surfaceMaterialDigest = program->structuralDigest();
    materialSlot.surfaceMaterialRevision = 7;
    materialSlot.textures.push_back({0, TextureSourceKind::ImportedBaseColor, {}});
    description.bindings.push_back(materialSlot);

    MaterialBinding objectOverride;
    objectOverride.targetKind = BindingTargetKind::ObjectOverride;
    objectOverride.object = scene.objects[1].id;
    objectOverride.surfaceMaterialDigest = program->structuralDigest();
    objectOverride.surfaceMaterialRevision = 7;
    objectOverride.textures.push_back(
        {0, TextureSourceKind::VideoResource, videoIdentity()});
    description.bindings.push_back(objectOverride);
    return description;
}

std::optional<AdmittedMaterialBindings> admitDescription(
    const Description& description, AdmissionFailure& failure,
    const surfacematerialbinding::AdmissionLimits& limits = {})
{
    std::string diagnostic;
    auto admitted = admit(description, failure, diagnostic, limits);
    check((admitted.has_value() && failure == AdmissionFailure::None
           && diagnostic.empty())
              || (!admitted.has_value() && diagnostic == token(failure)),
          "admission result and stable failure diagnostic agree");
    return admitted;
}

void expectRejected(const Description& description, AdmissionFailure expected,
                    const char* message,
                    const surfacematerialbinding::AdmissionLimits& limits = {})
{
    AdmissionFailure failure = AdmissionFailure::None;
    const auto admitted = admitDescription(description, failure, limits);
    check(!admitted && failure == expected, message);
}

void testResolutionImmutabilityAndDigest()
{
    auto scene = makeScene();
    const auto program = makeProgram();
    auto description = makeDescription(scene, program);
    AdmissionFailure failure = AdmissionFailure::None;
    auto admitted = admitDescription(description, failure);
    check(admitted.has_value(), "valid bounded material bindings are admitted");
    if (!admitted)
        return;

    check(admitted->scene() == scene.id && admitted->sceneRevision() == 5
              && admitted->structuralRevision() == 8
              && admitted->evaluationRevision() == 13
              && admitted->helperGeneration() == 23,
          "scene identity and separate stable revisions are retained");
    check(admitted->digest().size() == 64,
          "binding snapshot has a SHA-256 deterministic digest");

    const auto slot = admitted->resolve(scene.objects[0].id);
    const auto overridden = admitted->resolve(scene.objects[1].id);
    const auto fallback = admitted->resolve(scene.objects[2].id);
    check(slot && slot->kind == ResolutionKind::MaterialSlot
              && slot->binding && slot->program,
          "material-slot binding replaces matching imported material");
    check(overridden && overridden->kind == ResolutionKind::ObjectOverride
              && overridden->binding && overridden->program,
          "object override has exact precedence over material-slot binding");
    check(fallback && fallback->kind == ResolutionKind::ImportedMaterial
              && !fallback->binding && !fallback->program
              && fallback->importedMaterial == scene.materials[1].id,
          "unbound object falls back exactly to its imported material");
    check(!admitted->resolve(SceneObjectId {999}),
          "unknown object identity does not synthesize a fallback");

    auto reordered = description;
    std::reverse(reordered.bindings.begin(), reordered.bindings.end());
    auto second = admitDescription(reordered, failure);
    check(second && second->digest() == admitted->digest(),
          "digest and canonical bindings are independent of input ordering");

    auto changed = description;
    changed.modulationInputs[0].value = 0.5f;
    auto third = admitDescription(changed, failure);
    check(third && third->digest() != admitted->digest(),
          "evaluation receipt changes are digest-visible");

    scene.objects[0].material = scene.materials[1].id;
    const auto afterSourceMutation = admitted->resolve(
        videohelper::fixture3d::kCubeObjectId);
    check(afterSourceMutation && afterSourceMutation->kind == ResolutionKind::MaterialSlot,
          "admitted resolution is detached from later source scene mutation");
}

void testIdentityAndTargetRejections()
{
    const auto scene = makeScene();
    const auto program = makeProgram();
    const auto baseline = makeDescription(scene, program);

    auto malformed = baseline;
    malformed.bindings.push_back(malformed.bindings[0]);
    expectRejected(malformed, AdmissionFailure::DuplicateBindingTarget,
                   "duplicate target identities fail closed");

    malformed = baseline;
    malformed.bindings[0].materialSlot = SceneMaterialId {999};
    expectRejected(malformed, AdmissionFailure::MissingBindingTarget,
                   "missing imported material identity fails closed");

    malformed = baseline;
    malformed.bindings[1].object = SceneObjectId {999};
    expectRejected(malformed, AdmissionFailure::MissingBindingTarget,
                   "missing object identity fails closed");

    malformed = baseline;
    malformed.bindings[0].surfaceMaterialDigest[0] = '0';
    expectRejected(malformed, AdmissionFailure::MissingSurfaceMaterial,
                   "claimed material digest cannot replace admitted IR identity");

    malformed = baseline;
    malformed.bindings[0].surfaceMaterialRevision = 8;
    expectRejected(malformed, AdmissionFailure::MissingSurfaceMaterial,
                   "surface material revision is identity-bearing");

    malformed = baseline;
    malformed.programs.push_back(malformed.programs[0]);
    expectRejected(malformed, AdmissionFailure::DuplicateProgramIdentity,
                   "duplicate admitted material identities fail closed");

    malformed = baseline;
    malformed.programs.push_back({program, 9});
    expectRejected(malformed, AdmissionFailure::UnreferencedProgram,
                   "unreferenced program receipts fail closed");
}

void testTextureAndResourceRejections()
{
    const auto scene = makeScene();
    const auto program = makeProgram();
    const auto baseline = makeDescription(scene, program);

    auto malformed = baseline;
    malformed.bindings[0].textures.clear();
    expectRejected(malformed, AdmissionFailure::InvalidTextureBinding,
                   "every admitted IR texture slot requires an exact binding");

    malformed = baseline;
    malformed.bindings[0].textures[0].source = TextureSourceKind::VideoResource;
    expectRejected(malformed, AdmissionFailure::MissingVideoResource,
                   "video texture source requires a concrete resource receipt identity");

    malformed = baseline;
    malformed.bindings[1].textures[0].videoResource[0] ^= 1;
    expectRejected(malformed, AdmissionFailure::MissingVideoResource,
                   "missing video resource identity fails closed");

    malformed = baseline;
    malformed.videoResources[0].imageDescriptor.dataType = "mask";
    expectRejected(malformed, AdmissionFailure::InvalidVideoImageDescriptor,
                   "non-image typed resource descriptor fails closed");

    malformed = baseline;
    malformed.videoResources[0].imageDescriptor.direction = "in";
    expectRejected(malformed, AdmissionFailure::InvalidVideoImageDescriptor,
                   "input port cannot authorize a video texture resource");

    malformed = baseline;
    malformed.videoResources[0].evaluationRevision = 12;
    expectRejected(malformed, AdmissionFailure::InvalidVideoResourceRevision,
                   "stale evaluation resource receipt fails closed");

    malformed = baseline;
    malformed.videoResources[0].byteCount -= 1;
    expectRejected(malformed, AdmissionFailure::InvalidVideoResourceSize,
                   "resource byte extent must match its typed image descriptor");

    malformed = baseline;
    malformed.videoResources.push_back(malformed.videoResources[0]);
    expectRejected(malformed, AdmissionFailure::DuplicateVideoResource,
                   "duplicate video resource receipt identity fails closed");

    malformed = baseline;
    malformed.bindings[1].textures[0]
        = {0, TextureSourceKind::ImportedBaseColor, {}};
    malformed.videoResources.clear();
    malformed.helperGeneration = 0;
    const auto objectWithoutTexture = malformed;
    malformed.bindings[1].object = scene.objects[2].id;
    expectRejected(malformed, AdmissionFailure::MissingImportedTexture,
                   "imported base-color source requires an actual imported texture");

    malformed = objectWithoutTexture;
    malformed.videoResources.push_back(videoReceipt());
    malformed.helperGeneration = 23;
    expectRejected(malformed, AdmissionFailure::UnreferencedVideoResource,
                   "unreferenced video resource receipts fail closed");
}

void testModulationAndBoundsRejections()
{
    const auto scene = makeScene();
    const auto program = makeProgram();
    const auto baseline = makeDescription(scene, program);

    auto malformed = baseline;
    malformed.modulationInputs.clear();
    expectRejected(malformed, AdmissionFailure::MissingModulationInput,
                   "used material modulation input requires a receipt");

    malformed = baseline;
    malformed.modulationInputs.push_back(malformed.modulationInputs[0]);
    expectRejected(malformed, AdmissionFailure::DuplicateModulationInput,
                   "duplicate modulation semantic fails closed");

    malformed = baseline;
    malformed.modulationInputs[0].semantic = InputSemantic::AudioBass;
    expectRejected(malformed, AdmissionFailure::UnreferencedModulationInput,
                   "unreferenced modulation input fails closed");

    malformed = baseline;
    malformed.modulationInputs[0].value
        = std::numeric_limits<float>::quiet_NaN();
    expectRejected(malformed, AdmissionFailure::InvalidModulationInput,
                   "non-finite modulation value fails closed");

    surfacematerialbinding::AdmissionLimits limits;
    limits.bindings = 1;
    expectRejected(baseline, AdmissionFailure::BindingCapacityExceeded,
                   "caller can tighten bounded binding capacity", limits);

    limits = {};
    limits.videoResourceBytes = baseline.videoResources[0].byteCount - 1;
    expectRejected(baseline, AdmissionFailure::InvalidVideoResourceSize,
                   "caller can tighten aggregate resource byte capacity", limits);

    malformed = baseline;
    malformed.fallback = UnboundObjectFallback::Invalid;
    expectRejected(malformed, AdmissionFailure::UnsupportedFallback,
                   "fallback policy is exact rather than implicit");
}
} // namespace

int main()
{
    testResolutionImmutabilityAndDigest();
    testIdentityAndTargetRejections();
    testTextureAndResourceRejections();
    testModulationAndBoundsRejections();
    if (failures != 0)
    {
        std::cerr << failures << " material binding contract checks failed\n";
        return 1;
    }
    std::cout << "material binding contract checks passed\n";
    return 0;
}
