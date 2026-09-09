#include "../src/material_program_compiler.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace
{
using namespace surfacematerial;
using namespace videohelper::materialprogram;
using videowire::VertexModifierIr;
using videowire::VertexModifierOperation;
using videowire::VertexModifierRecord;
using videowire::VertexModifierStableId;

int failures = 0;

void check(bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
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

ProgramDescription surfaceFixture()
{
    ProgramDescription program;
    program.textureSlotCount = 2;

    const auto uv = append(program, OperationKind::Input, ValueType::Vec2);
    program.operations.back().semantic = InputSemantic::TexCoord0;
    const auto sampled = append(program, OperationKind::TextureSample2D,
                                ValueType::Vec4, { uv });
    program.operations.back().parameter = 1;
    const auto sampleRed = append(program, OperationKind::Component,
                                  ValueType::Scalar, { sampled });
    const auto control = append(program, OperationKind::Input, ValueType::Scalar);
    program.operations.back().semantic = InputSemantic::Control;
    program.operations.back().parameter = 5;
    const auto sum = append(program, OperationKind::Add,
                            ValueType::Scalar, { sampleRed, control });
    const auto color = append(program, OperationKind::ComposeVec3,
                              ValueType::Vec3, { sum, sampleRed, control });
    const auto materialId = append(program, OperationKind::UIntConstant,
                                   ValueType::UInt);
    program.operations.back().unsignedLiteral = 42;

    for (const auto output : kSurfaceOutputs)
    {
        const auto type = outputType(output);
        program.outputs[static_cast<std::size_t>(output)]
            = type == ValueType::Vec3 ? color
            : type == ValueType::UInt ? materialId
            : sum;
    }
    return program;
}

VertexModifierRecord vertexRecord(
    VertexModifierStableId id,
    VertexModifierOperation operation,
    std::initializer_list<VertexModifierStableId> inputs = {},
    std::initializer_list<double> parameters = {})
{
    const auto schema = videowire::vertexModifierOperationSchema(operation);
    VertexModifierRecord record;
    record.stableId = id;
    record.operation = operation;
    record.resultType = schema.resultType;
    record.inputCount = static_cast<std::uint8_t>(inputs.size());
    record.parameterCount = static_cast<std::uint8_t>(parameters.size());
    std::copy(inputs.begin(), inputs.end(), record.inputs.begin());
    std::copy(parameters.begin(), parameters.end(), record.parameters.begin());
    return record;
}

VertexModifierIr vertexFixture(bool reverseRecords, double signedZero)
{
    using Operation = VertexModifierOperation;
    VertexModifierIr ir;
    ir.rootId = 70;
    ir.records = {
        vertexRecord(10, Operation::importedPosition),
        vertexRecord(20, Operation::audioParameter, {}, { 3.0 }),
        vertexRecord(30, Operation::controlParameter, {}, { 7.0 }),
        vertexRecord(40, Operation::scalarAdd, { 20, 30 }),
        vertexRecord(50, Operation::vec3Compose, { 40, 20, 30 }),
        vertexRecord(55, Operation::vec3Constant, {}, { signedZero, 0.25, -0.25 }),
        vertexRecord(60, Operation::vec3Add, { 50, 55 }),
        vertexRecord(70, Operation::boundedDisplacementOutput, { 10, 60 }, { 0.5 })
    };
    if (reverseRecords) std::reverse(ir.records.begin(), ir.records.end());
    return ir;
}

bool contains(const std::string& value, const char* fragment)
{
    return value.find(fragment) != std::string::npos;
}
} // namespace

int main()
{
    static_assert(videowire::kMaterialProgramIrSchemaVersion == 1);
    static_assert(!videowire::MaterialProgramIR::containsShaderSource);
    static_assert(!videowire::MaterialProgramIR::performsNativeShaderCompilation);
    static_assert(!videowire::MaterialProgramIR::authorizesNativeGpuExecution);
    static_assert(!MaterialProgramRecord::containsShaderSource);
    static_assert(!MaterialProgramRecord::performsNativeShaderCompilation);
    static_assert(!MaterialProgramRecord::authorizesNativeGpuExecution);
    static_assert(!MaterialProgramRecord::allowsCpuProductionRenderingFallback);
    static_assert(std::is_same_v<
        decltype(std::declval<const MaterialProgramRecord&>().program()),
        const videowire::MaterialProgramIR&>);

    std::string error;
    const auto admittedSurface = admit(surfaceFixture(), error);
    check(admittedSurface.has_value() && error.empty(),
          "the surface fixture is admitted before compilation");
    const auto admittedVertex = videohelper::vertexmodifier::admitVertexModifierIr(
        vertexFixture(true, -0.0), {}, error);
    check(admittedVertex.has_value() && error.empty(),
          "the vertex fixture is admitted before compilation");
    if (!admittedSurface || !admittedVertex) return EXIT_FAILURE;

    const auto openGl = compileMaterialProgram(
        &*admittedSurface, &*admittedVertex, BackendTarget::OpenGl, error);
    check(openGl.has_value() && error.empty(),
          "the admitted surface and vertex programs compile to neutral IR");
    if (!openGl) return EXIT_FAILURE;

    const auto& capabilities = openGl->capabilities();
    check(capabilities.backend.target == BackendTarget::OpenGl
              && capabilities.backend.minimumApiMajor == 3
              && capabilities.backend.minimumApiMinor == 3
              && capabilities.backend.shaderLanguage == ShaderLanguageProfile::Glsl330Core
              && capabilities.backend.shaderLanguageVersion == 330,
          "OpenGL requirements are explicit without compiling a native shader");
    check(capabilities.compilerVersion == kCompilerVersion
              && capabilities.programRecordVersion
                   == videowire::kMaterialProgramIrSchemaVersion
              && capabilities.surfaceIrVersion == kWireVersion
              && capabilities.vertexIrVersion
                   == videowire::kVertexModifierIrSchemaVersion,
          "the program records every source and compiler contract version");
    check(openGl->program().schemaVersion == videowire::kMaterialProgramIrSchemaVersion,
          "the emitted program uses the shared neutral IR schema");

    const auto& surface = openGl->surfaceInstructions();
    check(surface.size() == 7 && surface[0].resultSlot == 0
              && surface[0].resultType == ValueType::Vec2
              && surface[1].operation == OperationKind::TextureSample2D
              && surface[1].inputs[0] == 0
              && surface[4].resultType == ValueType::Scalar
              && surface[4].inputs[0] == 2 && surface[4].inputs[1] == 3
              && surface[5].resultType == ValueType::Vec3,
          "surface lowering preserves exact admitted types and ordered value slots");
    for (const auto output : kSurfaceOutputs)
    {
        const auto slot = openGl->surfaceOutputSlots()[static_cast<std::size_t>(output)];
        const auto expected = outputType(output) == ValueType::Vec3 ? 5u
            : outputType(output) == ValueType::UInt ? 6u : 4u;
        check(slot == expected, "surface outputs bind to exact typed value slots");
    }

    const auto& vertex = openGl->vertexInstructions();
    check(vertex.size() == 8 && vertex.front().sourceStableId == 10
              && vertex.back().sourceStableId == 70
              && vertex[3].inputs[0] == 1 && vertex[3].inputs[1] == 2
              && vertex[4].resultType == videowire::VertexModifierValueType::vec3
              && vertex[5].parameters[0] == 0.0
              && !std::signbit(vertex[5].parameters[0])
              && openGl->vertexOutputSlot() == 7,
          "vertex lowering uses stable topological order and canonical finite parameters");
    for (std::size_t index = 1; index < vertex.size(); ++index)
        check(vertex[index - 1].sourceStableId < vertex[index].sourceStableId,
              "independent vertex records use stable-ID tie ordering");

    const auto& resources = openGl->resources();
    check(resources.surfaceInstructions == 7 && resources.vertexInstructions == 8
              && resources.valueSlots == 15 && resources.inputReferences == 16
              && resources.constantBytes == 52 && resources.textureSlots == 2
              && resources.controlParameterSlots == 8
              && resources.audioParameterSlots == 4
              && resources.surfaceOutputBindings == kSurfaceOutputCount
              && resources.surfaceMaximumDepth == admittedSurface->maximumDepth()
              && resources.vertexMaximumDepth == admittedVertex->maximumDepth()
              && resources.maximumVertexDisplacement == 0.5
              && resources.surfaceParameterBlockBytes
                   == sizeof(SurfacePbrParameterBlock)
              && resources.canonicalPayloadBytes > 0
              && resources.canonicalPayloadBytes <= kMaximumCanonicalPayloadBytes,
          "resource accounting is exact and bounded");
    check(openGl->sourceIdentity().size() == 64
              && openGl->programIdentity().size() == 64,
          "source and program identities are SHA-256 digests");
    check(openGl->sourceIdentity()
              == "b43f6feceba769a1261202fe922e937551307195b00360b45361aa41c6357b8b"
              && openGl->programIdentity()
                   == "4a087bf7d1c522e9a47d15179840f6c5a7d180da3af25c2806dff5b1b24aaceb",
          "the reference program has stable cross-process structural digests");

    const auto reorderedVertex = videohelper::vertexmodifier::admitVertexModifierIr(
        vertexFixture(false, 0.0), {}, error);
    const auto repeated = reorderedVertex
        ? compileMaterialProgram(&*admittedSurface, &*reorderedVertex,
                                 BackendTarget::OpenGl, error)
        : std::nullopt;
    check(repeated && repeated->sourceIdentity() == openGl->sourceIdentity()
              && repeated->programIdentity() == openGl->programIdentity(),
          "source order and negative zero do not change the structural digest");

    const auto metal = compileMaterialProgram(
        &*admittedSurface, &*admittedVertex, BackendTarget::Metal, error);
    check(metal && metal->capabilities().backend.target == BackendTarget::Metal
              && metal->capabilities().backend.shaderLanguage
                   == ShaderLanguageProfile::MetalSl20
              && metal->sourceIdentity() == openGl->sourceIdentity()
              && metal->programIdentity() != openGl->programIdentity(),
          "backend requirements participate in program identity but not source identity");

    check(compileMaterialProgram(&*admittedSurface, nullptr,
                                 BackendTarget::OpenGl, error)
              .has_value(),
          "a surface-only material program is valid");
    check(compileMaterialProgram(nullptr, &*admittedVertex,
                                 BackendTarget::OpenGl, error)
              .has_value(),
          "a vertex-only material program is valid");
    check(!compileMaterialProgram(nullptr, nullptr, BackendTarget::OpenGl, error)
               && contains(error, "requires a surface or vertex"),
          "an empty material program is rejected");
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::Invalid, error)
               && contains(error, "backend target"),
          "unsupported backends fail closed");

    CompileLimits limits;
    limits.surfaceInstructions = 6;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "stage instruction"),
          "surface instruction limits are enforced");
    limits = {};
    limits.vertexInstructions = 7;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "stage instruction"),
          "vertex instruction limits are enforced");
    limits = {};
    limits.combinedInstructions = 14;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "combined instruction"),
          "combined instruction limits are enforced");
    limits = {};
    limits.textureSlots = 1;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "binding resource"),
          "texture binding limits are enforced");
    limits = {};
    limits.controlParameterSlots = 7;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "binding resource"),
          "control binding limits are enforced");
    limits = {};
    limits.audioParameterSlots = 3;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "binding resource"),
          "audio binding limits are enforced");
    limits = {};
    limits.canonicalPayloadBytes = 1;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "payload byte"),
          "canonical output byte limits are enforced");
    limits = {};
    limits.surfaceInstructions = 0;
    check(!compileMaterialProgram(&*admittedSurface, &*admittedVertex,
                                  BackendTarget::OpenGl, error, limits)
               && contains(error, "hard contract bounds"),
          "caller limits cannot disable or exceed hard compiler bounds");

    auto badSurface = surfaceFixture();
    badSurface.operations[4].resultType = ValueType::Vec3;
    check(!admit(badSurface, error) && contains(error, "incompatible explicit types"),
          "surface admission enforces exact operation types before compilation");
    auto badVertex = vertexFixture(false, 0.0);
    badVertex.records[3].resultType = videowire::VertexModifierValueType::vec3;
    check(!videohelper::vertexmodifier::admitVertexModifierIr(badVertex, {}, error)
               && contains(error, "invalid typed operation shape"),
          "vertex admission enforces exact operation types before compilation");
    badVertex = vertexFixture(false, 0.0);
    badVertex.records[5].parameters[1] = std::numeric_limits<double>::infinity();
    check(!videohelper::vertexmodifier::admitVertexModifierIr(badVertex, {}, error)
               && contains(error, "non-finite"),
          "non-finite vertex parameters cannot reach the compiler");

    MaterialProgramCompilerCheckpoint checkpoint;
    check(checkpoint.compileAndPublish(1, &*admittedSurface, &*admittedVertex,
                                       BackendTarget::OpenGl, error),
          "the first good revision publishes");
    const auto firstGood = checkpoint.lastGood();
    limits = {};
    limits.combinedInstructions = 1;
    check(!checkpoint.compileAndPublish(2, &*admittedSurface, &*admittedVertex,
                                        BackendTarget::OpenGl, error, limits)
              && checkpoint.latestAttemptedRevision() == 2
              && checkpoint.lastGoodRevision() == 1
              && checkpoint.rejectedRevision() == 2
              && checkpoint.lastGood() == firstGood,
          "a failed edit retains the immutable last-good program");
    check(!checkpoint.compileAndPublish(2, &*admittedSurface, &*admittedVertex,
                                        BackendTarget::OpenGl, error)
              && contains(error, "stale") && checkpoint.lastGood() == firstGood,
          "stale authoring revisions cannot replace the last-good program");

    std::cout << "source identity: " << openGl->sourceIdentity() << '\n';
    std::cout << "OpenGL program identity: " << openGl->programIdentity() << '\n';
    std::cout << "canonical payload bytes: " << resources.canonicalPayloadBytes << '\n';
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "Material program compiler checks passed\n";
    return EXIT_SUCCESS;
}
