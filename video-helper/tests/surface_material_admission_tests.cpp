#include "../src/surface_material_admission.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace surfacematerial;

static_assert(std::is_copy_constructible_v<AdmittedSurfaceMaterialIR>);
static_assert(!std::is_constructible_v<AdmittedSurfaceMaterialIR,
                                       ProgramDescription,
                                       std::string,
                                       std::size_t>);
static_assert(!std::is_copy_assignable_v<AdmittedSurfaceMaterialIR>);
static_assert(!std::is_move_assignable_v<AdmittedSurfaceMaterialIR>);
static_assert(std::is_same_v<decltype(std::declval<const AdmittedSurfaceMaterialIR&>().program()),
                             const ProgramDescription&>);
static_assert(static_cast<std::uint8_t>(ValueType::UInt) == 5);
static_assert(static_cast<std::uint8_t>(InputSemantic::Control) == 14);
static_assert(static_cast<std::uint8_t>(OperationKind::ComposeVec4) == 21);
static_assert(static_cast<std::uint8_t>(OutputSemantic::MaterialId) == 9);

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

ValueId appendFloat(ProgramDescription& program,
                    ValueType type,
                    std::array<float, 4> value)
{
    const auto id = append(program, OperationKind::FloatConstant, type);
    program.operations.back().literal = value;
    return id;
}

ValueId appendScalar(ProgramDescription& program, float value)
{
    return appendFloat(program, ValueType::Scalar, { value, 0.0f, 0.0f, 0.0f });
}

ValueId appendUInt(ProgramDescription& program, std::uint32_t value)
{
    const auto id = append(program, OperationKind::UIntConstant, ValueType::UInt);
    program.operations.back().unsignedLiteral = value;
    return id;
}

ValueId appendInput(ProgramDescription& program,
                    InputSemantic semantic,
                    std::uint16_t parameter = 0)
{
    const auto id = append(program, OperationKind::Input, inputType(semantic));
    program.operations.back().semantic = semantic;
    program.operations.back().parameter = parameter;
    return id;
}

void bindOutputs(ProgramDescription& program,
                 ValueId scalar,
                 ValueId vector,
                 ValueId unsignedValue)
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

ProgramDescription comprehensiveProgram()
{
    ProgramDescription program;
    program.textureSlotCount = static_cast<std::uint16_t>(kMaximumTextureSlots);

    const auto uv = append(program, OperationKind::Input, ValueType::Vec2);
    program.operations.back().semantic = InputSemantic::TexCoord0;

    const auto sampled = append(program, OperationKind::TextureSample2D,
                                ValueType::Vec4, { uv });
    program.operations.back().parameter
        = static_cast<std::uint16_t>(kMaximumTextureSlots - 1);

    const auto red = append(program, OperationKind::Component,
                            ValueType::Scalar, { sampled });
    program.operations.back().parameter = 0;
    const auto half = appendScalar(program, 0.5f);
    const auto added = append(program, OperationKind::Add,
                              ValueType::Scalar, { red, half });
    const auto subtracted = append(program, OperationKind::Subtract,
                                   ValueType::Scalar, { added, half });
    const auto multiplied = append(program, OperationKind::Multiply,
                                   ValueType::Scalar, { subtracted, half });
    const auto divided = append(program, OperationKind::Divide,
                                ValueType::Scalar, { multiplied, half });
    const auto minimum = append(program, OperationKind::Minimum,
                                ValueType::Scalar, { divided, half });
    const auto maximum = append(program, OperationKind::Maximum,
                                ValueType::Scalar, { minimum, half });
    const auto zero = appendScalar(program, 0.0f);
    const auto one = appendScalar(program, 1.0f);
    const auto clamped = append(program, OperationKind::Clamp,
                                ValueType::Scalar, { maximum, zero, one });
    const auto mixed = append(program, OperationKind::Mix,
                              ValueType::Scalar, { zero, one, clamped });
    const auto absolute = append(program, OperationKind::Absolute,
                                 ValueType::Scalar, { mixed });
    const auto powered = append(program, OperationKind::Power,
                                ValueType::Scalar, { absolute, half });
    const auto vector2 = append(program, OperationKind::ComposeVec2,
                                ValueType::Vec2, { powered, powered });
    const auto dotted = append(program, OperationKind::Dot,
                               ValueType::Scalar, { vector2, vector2 });
    const auto normalized = append(program, OperationKind::Normalize,
                                   ValueType::Vec2, { vector2 });
    const auto length = append(program, OperationKind::Length,
                               ValueType::Scalar, { normalized });
    const auto vector3 = append(program, OperationKind::ComposeVec3,
                                ValueType::Vec3, { dotted, length, powered });
    const auto vector4 = append(program, OperationKind::ComposeVec4,
                                ValueType::Vec4,
                                { powered, dotted, length, clamped });
    const auto alpha = append(program, OperationKind::Component,
                              ValueType::Scalar, { vector4 });
    program.operations.back().parameter = 3;
    const auto materialId = appendUInt(program, 0x12345678u);
    bindOutputs(program, alpha, vector3, materialId);
    return program;
}

ProgramDescription programForInput(InputSemantic semantic)
{
    ProgramDescription program;
    const auto input = append(program, OperationKind::Input, inputType(semantic));
    program.operations.back().semantic = semantic;

    ValueId scalar = 0;
    ValueId vector = 0;
    if (inputType(semantic) == ValueType::Scalar)
    {
        scalar = input;
        vector = append(program, OperationKind::ComposeVec3,
                        ValueType::Vec3, { scalar, scalar, scalar });
    }
    else
    {
        scalar = append(program, OperationKind::Component,
                        ValueType::Scalar, { input });
        program.operations.back().parameter = 0;
        vector = inputType(semantic) == ValueType::Vec3
            ? input
            : append(program, OperationKind::ComposeVec3,
                     ValueType::Vec3, { scalar, scalar, scalar });
    }

    const auto materialId = appendUInt(program, 7);
    bindOutputs(program, scalar, vector, materialId);
    return program;
}

ProgramDescription maximumOperationProgram()
{
    ProgramDescription program;
    std::vector<ValueId> level;
    for (std::size_t i = 0; i < 62; ++i)
        level.push_back(appendScalar(program, static_cast<float>(i + 1)));

    while (level.size() > 1)
    {
        std::vector<ValueId> next;
        for (std::size_t i = 0; i < level.size(); i += 2)
        {
            if (i + 1 == level.size())
                next.push_back(level[i]);
            else
                next.push_back(append(program, OperationKind::Add,
                                      ValueType::Scalar, { level[i], level[i + 1] }));
        }
        level = std::move(next);
    }

    auto scalar = level.front();
    for (int i = 0; i < 3; ++i)
        scalar = append(program, OperationKind::Absolute, ValueType::Scalar, { scalar });
    const auto vector = append(program, OperationKind::ComposeVec3,
                               ValueType::Vec3, { scalar, scalar, scalar });
    const auto materialId = appendUInt(program, 1);
    bindOutputs(program, scalar, vector, materialId);
    return program;
}

ProgramDescription maximumDepthProgram()
{
    ProgramDescription program;
    auto scalar = appendScalar(program, 1.0f);
    for (int i = 0; i < 29; ++i)
        scalar = append(program, OperationKind::Absolute, ValueType::Scalar, { scalar });
    const auto vector = append(program, OperationKind::ComposeVec3,
                               ValueType::Vec3, { scalar, scalar, scalar });
    const auto normalized = append(program, OperationKind::Normalize,
                                   ValueType::Vec3, { vector });
    const auto materialId = appendUInt(program, 2);
    bindOutputs(program, scalar, normalized, materialId);
    return program;
}

ProgramDescription evaluationProgram()
{
    ProgramDescription program;
    const auto time = appendInput(program, InputSemantic::Time);
    const auto bass = appendInput(program, InputSemantic::AudioBass);
    const auto control = appendInput(program, InputSemantic::Control);
    const auto ratio = append(program, OperationKind::Divide,
                              ValueType::Scalar, { bass, control });
    const auto added = append(program, OperationKind::Add,
                              ValueType::Scalar, { control, bass });
    const auto half = appendScalar(program, 0.5f);
    const auto modulation = append(program, OperationKind::Multiply,
                                   ValueType::Scalar, { added, half });
    const auto red = appendScalar(program, 0.25f);
    const auto baseColor = append(program, OperationKind::ComposeVec3,
                                  ValueType::Vec3, { red, control, bass });
    const auto emission = append(program, OperationKind::ComposeVec3,
                                 ValueType::Vec3, { ratio, modulation, time });
    const auto zero = appendScalar(program, 0.0f);
    const auto one = appendScalar(program, 1.0f);
    const auto two = appendScalar(program, 2.0f);
    const auto normal = append(program, OperationKind::ComposeVec3,
                               ValueType::Vec3, { zero, zero, two });
    const auto materialId = appendUInt(program, 0xdeadbeefu);

    program.outputs[static_cast<std::size_t>(OutputSemantic::BaseColor)] = baseColor;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Metallic)] = control;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Roughness)] = bass;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Emission)] = emission;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Opacity)] = one;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Normal)] = normal;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Transmission)] = zero;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Ior)] = time;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Clearcoat)] = modulation;
    program.outputs[static_cast<std::size_t>(OutputSemantic::MaterialId)] = materialId;
    return program;
}

ProgramDescription completeValueEvaluationProgram()
{
    ProgramDescription program;
    const auto control = appendInput(program, InputSemantic::Control);
    const auto bass = appendInput(program, InputSemantic::AudioBass);
    const auto one = appendScalar(program, 1.0f);
    const auto two = appendScalar(program, 2.0f);
    const auto difference = append(program, OperationKind::Subtract,
                                   ValueType::Scalar, { control, bass });
    const auto minimum = append(program, OperationKind::Minimum,
                                ValueType::Scalar, { difference, one });
    const auto maximum = append(program, OperationKind::Maximum,
                                ValueType::Scalar, { minimum, bass });
    const auto zero = appendScalar(program, 0.0f);
    const auto clamped = append(program, OperationKind::Clamp,
                                ValueType::Scalar, { maximum, zero, one });
    const auto negative = append(program, OperationKind::Subtract,
                                 ValueType::Scalar, { zero, clamped });
    const auto absolute = append(program, OperationKind::Absolute,
                                 ValueType::Scalar, { negative });
    const auto powered = append(program, OperationKind::Power,
                                ValueType::Scalar, { absolute, two });
    const auto vector2 = append(program, OperationKind::ComposeVec2,
                                ValueType::Vec2, { powered, clamped });
    const auto dotted = append(program, OperationKind::Dot,
                               ValueType::Scalar, { vector2, vector2 });
    const auto normalized2 = append(program, OperationKind::Normalize,
                                    ValueType::Vec2, { vector2 });
    const auto length = append(program, OperationKind::Length,
                               ValueType::Scalar, { normalized2 });
    const auto component2 = append(program, OperationKind::Component,
                                   ValueType::Scalar, { vector2 });
    program.operations.back().parameter = 1;
    const auto added = append(program, OperationKind::Add,
                              ValueType::Scalar, { dotted, component2 });
    const auto multiplied = append(program, OperationKind::Multiply,
                                   ValueType::Scalar, { added, clamped });
    const auto divided = append(program, OperationKind::Divide,
                                ValueType::Scalar, { multiplied, two });
    const auto firstColor = append(program, OperationKind::ComposeVec3,
                                   ValueType::Vec3, { powered, dotted, divided });
    const auto vector4 = append(program, OperationKind::ComposeVec4,
                                ValueType::Vec4,
                                { powered, dotted, divided, length });
    const auto component4 = append(program, OperationKind::Component,
                                   ValueType::Scalar, { vector4 });
    program.operations.back().parameter = 3;
    const auto secondColor = append(program, OperationKind::ComposeVec3,
                                    ValueType::Vec3, { control, bass, clamped });
    const auto mixed = append(program, OperationKind::Mix,
                              ValueType::Vec3,
                              { firstColor, secondColor, powered });
    const auto normal = append(program, OperationKind::Normalize,
                               ValueType::Vec3, { mixed });
    const auto materialId = appendUInt(program, 0x10203040u);

    program.outputs[static_cast<std::size_t>(OutputSemantic::BaseColor)] = mixed;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Metallic)] = control;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Roughness)] = powered;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Emission)] = firstColor;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Opacity)] = clamped;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Normal)] = normal;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Transmission)] = divided;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Ior)] = component4;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Clearcoat)] = dotted;
    program.outputs[static_cast<std::size_t>(OutputSemantic::MaterialId)] = materialId;
    return program;
}

ProgramDescription hardBoundsProgram()
{
    ProgramDescription program;
    const auto baseColor = appendFloat(program, ValueType::Vec3,
                                       { -1.0f, 2.0f, 0.5f, 0.0f });
    const auto emission = appendFloat(program, ValueType::Vec3,
                                      { -2.0f, 70000.0f,
                                        kMaximumEvaluationMagnitude, 0.0f });
    const auto below = appendScalar(program, -2.0f);
    const auto above = appendScalar(program, 2.0f);
    const auto ior = appendScalar(program, 100.0f);
    const auto normal = appendFloat(program, ValueType::Vec3,
                                    { 0.0f, 0.0f,
                                      kMaximumEvaluationMagnitude, 0.0f });
    const auto materialId = appendUInt(program, std::numeric_limits<std::uint32_t>::max());

    program.outputs[static_cast<std::size_t>(OutputSemantic::BaseColor)] = baseColor;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Metallic)] = below;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Roughness)] = above;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Emission)] = emission;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Opacity)] = above;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Normal)] = normal;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Transmission)] = below;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Ior)] = ior;
    program.outputs[static_cast<std::size_t>(OutputSemantic::Clearcoat)] = above;
    program.outputs[static_cast<std::size_t>(OutputSemantic::MaterialId)] = materialId;
    return program;
}

ProgramDescription overflowingEvaluationProgram()
{
    ProgramDescription program;
    const auto maximum = appendScalar(program, kMaximumEvaluationMagnitude);
    const auto one = appendScalar(program, 1.0f);
    const auto overflow = append(program, OperationKind::Add,
                                 ValueType::Scalar, { maximum, one });
    const auto vector = append(program, OperationKind::ComposeVec3,
                               ValueType::Vec3, { overflow, one, one });
    const auto materialId = appendUInt(program, 1);
    bindOutputs(program, one, vector, materialId);
    return program;
}

ProgramDescription textureEvaluationProgram()
{
    ProgramDescription program;
    program.textureSlotCount = 1;
    const auto uv = appendFloat(program, ValueType::Vec2,
                                { 0.5f, 0.5f, 0.0f, 0.0f });
    const auto sampled = append(program, OperationKind::TextureSample2D,
                                ValueType::Vec4, { uv });
    const auto red = append(program, OperationKind::Component,
                            ValueType::Scalar, { sampled });
    const auto vector = append(program, OperationKind::ComposeVec3,
                               ValueType::Vec3, { red, red, red });
    const auto materialId = appendUInt(program, 1);
    bindOutputs(program, red, vector, materialId);
    return program;
}

void expectRejected(const ProgramDescription& program,
                    std::string_view diagnostic,
                    const char* message,
                    const AdmissionLimits& limits = {})
{
    std::string error;
    const auto result = admit(program, error, limits);
    check(!result, message);
    check(error.find(diagnostic) != std::string::npos,
          "rejection diagnostic identifies the violated contract");
}

void expectEvaluationRejected(const AdmittedSurfaceMaterialIR& admitted,
                              const MaterialEvaluationInputs& inputs,
                              std::string_view diagnostic,
                              const char* message)
{
    std::string error;
    const auto result = evaluateReference(admitted, inputs, error);
    check(!result, message);
    if (error.find(diagnostic) == std::string::npos)
    {
        ++failures;
        std::cerr << "FAIL: evaluation diagnostic '" << error
                  << "' does not contain '" << diagnostic << "'\n";
    }
}

void testStableVocabulary()
{
    constexpr std::array<std::string_view, 5> valueTokens {
        "scalar", "vec2", "vec3", "vec4", "uint"
    };
    constexpr std::array<std::string_view, 14> inputTokens {
        "position", "normal", "tangent", "texCoord0", "texCoord1",
        "vertexColor", "viewDirection", "time", "audioLevel", "audioBass",
        "audioMid", "audioTreble", "audioBeat", "control"
    };
    constexpr std::array<std::string_view, 21> operationTokens {
        "floatConstant", "uintConstant", "input", "textureSample2D", "add",
        "subtract", "multiply", "divide", "minimum", "maximum", "clamp",
        "mix", "dot", "normalize", "length", "absolute", "power",
        "component", "composeVec2", "composeVec3", "composeVec4"
    };
    constexpr std::array<std::string_view, 10> outputTokens {
        "baseColor", "metallic", "roughness", "emission", "opacity",
        "normal", "transmission", "ior", "clearcoat", "materialId"
    };
    constexpr std::array<ValueType, 10> outputTypes {
        ValueType::Vec3, ValueType::Scalar, ValueType::Scalar, ValueType::Vec3,
        ValueType::Scalar, ValueType::Vec3, ValueType::Scalar, ValueType::Scalar,
        ValueType::Scalar, ValueType::UInt
    };

    std::set<std::string_view> uniqueTokens;
    for (std::size_t i = 0; i < kValueTypes.size(); ++i)
    {
        check(static_cast<std::uint8_t>(kValueTypes[i]) == i + 1,
              "value type IDs remain contiguous and append-only");
        check(token(kValueTypes[i]) == valueTokens[i], "value type token is stable");
        uniqueTokens.insert(token(kValueTypes[i]));
    }
    check(uniqueTokens.size() == kValueTypes.size(), "value type tokens are unique");

    uniqueTokens.clear();
    for (std::size_t i = 0; i < kInputSemantics.size(); ++i)
    {
        check(static_cast<std::uint8_t>(kInputSemantics[i]) == i + 1,
              "input IDs remain contiguous and append-only");
        check(token(kInputSemantics[i]) == inputTokens[i], "input token is stable");
        uniqueTokens.insert(token(kInputSemantics[i]));
    }
    check(uniqueTokens.size() == kInputSemantics.size(), "input tokens are unique");

    uniqueTokens.clear();
    for (std::size_t i = 0; i < kOperationKinds.size(); ++i)
    {
        check(static_cast<std::uint8_t>(kOperationKinds[i]) == i + 1,
              "operation IDs remain contiguous and append-only");
        check(token(kOperationKinds[i]) == operationTokens[i], "operation token is stable");
        uniqueTokens.insert(token(kOperationKinds[i]));
    }
    check(uniqueTokens.size() == kOperationKinds.size(), "operation tokens are unique");

    uniqueTokens.clear();
    for (std::size_t i = 0; i < kSurfaceOutputs.size(); ++i)
    {
        check(static_cast<std::uint8_t>(kSurfaceOutputs[i]) == i,
              "output IDs remain contiguous and append-only");
        check(token(kSurfaceOutputs[i]) == outputTokens[i], "output token is stable");
        check(outputName(kSurfaceOutputs[i]) == outputTokens[i],
              "output name aliases its stable token");
        check(outputType(kSurfaceOutputs[i]) == outputTypes[i],
              "output type is stable");
        uniqueTokens.insert(token(kSurfaceOutputs[i]));
    }
    check(uniqueTokens.size() == kSurfaceOutputs.size(), "output tokens are unique");
    check(token(ValueType::Invalid).empty(), "invalid value type has no token");
    check(token(InputSemantic::Invalid).empty(), "invalid input has no token");
    check(token(OperationKind::Invalid).empty(), "invalid operation has no token");
    check(token(OutputSemantic::Count).empty(), "output sentinel has no token");
}

void testAdmissionAndDigest()
{
    auto program = comprehensiveProgram();
    std::string error = "stale";
    const auto admitted = admit(program, error);
    check(admitted.has_value(), "complete restricted operation vocabulary is admitted");
    check(error.empty(), "successful admission clears an old diagnostic");
    if (!admitted)
        return;

    check(admitted->program().operations.size() == 24,
          "admitted immutable copy retains every operation");
    check(admitted->maximumDepth() == 18, "maximum graph depth is deterministic");
    check(admitted->structuralDigest().size() == 64,
          "structural digest is lowercase SHA-256 hex");
    check(admitted->structuralDigest()
              == "3071cfeebe22ff91edcf4389164b9001528833bed5186771f8cf3f720963cbe4",
          "structural digest matches the independent serialization oracle");

    const auto originalDigest = admitted->structuralDigest();
    program.operations[3].literal[0] = 0.25f;
    check(admitted->program().operations[3].literal[0] == 0.5f,
          "admission owns an immutable deep copy of untrusted input");
    check(admitted->structuralDigest() == originalDigest,
          "source mutation cannot change an admitted digest");

    const auto changed = admit(program, error);
    check(changed && changed->structuralDigest() != originalDigest,
          "one structural literal change changes the digest");

    auto positiveZero = comprehensiveProgram();
    auto negativeZero = positiveZero;
    negativeZero.operations[10].literal[0] = -0.0f;
    const auto positive = admit(positiveZero, error);
    const auto negative = admit(negativeZero, error);
    check(positive && negative
              && positive->structuralDigest() == negative->structuralDigest(),
          "positive and negative zero have one canonical digest encoding");

    AdmissionLimits exactLimits;
    exactLimits.textureSlots = kMaximumTextureSlots;
    exactLimits.operations = program.operations.size();
    exactLimits.depth = kMaximumDepth;
    const auto customLimits = admit(program, error, exactLimits);
    check(changed && customLimits
              && customLimits->structuralDigest() == changed->structuralDigest(),
          "admission policy limits do not alter structural identity");

    auto firstControl = programForInput(InputSemantic::Control);
    auto secondControl = firstControl;
    secondControl.operations.front().parameter = 1;
    const auto firstControlDigest = admit(firstControl, error);
    const auto secondControlDigest = admit(secondControl, error);
    check(firstControlDigest && secondControlDigest
              && firstControlDigest->structuralDigest()
                  != secondControlDigest->structuralDigest(),
          "control input index participates in structural identity");
}

void testEveryInput()
{
    constexpr std::array<ValueType, 14> expectedTypes {
        ValueType::Vec3, ValueType::Vec3, ValueType::Vec4, ValueType::Vec2,
        ValueType::Vec2, ValueType::Vec4, ValueType::Vec3, ValueType::Scalar,
        ValueType::Scalar, ValueType::Scalar, ValueType::Scalar,
        ValueType::Scalar, ValueType::Scalar, ValueType::Scalar
    };

    for (std::size_t i = 0; i < kInputSemantics.size(); ++i)
    {
        check(inputType(kInputSemantics[i]) == expectedTypes[i],
              "canonical input type is stable");
        std::string error;
        check(admit(programForInput(kInputSemantics[i]), error).has_value(),
              "each declared surface input is admitted with its canonical type");
    }

    auto invalid = programForInput(InputSemantic::Position);
    invalid.operations.front().semantic = static_cast<InputSemantic>(255);
    expectRejected(invalid, "semantic and result type", "unknown input semantic rejected");

    auto wrongType = programForInput(InputSemantic::Position);
    wrongType.operations.front().resultType = ValueType::Vec4;
    expectRejected(wrongType, "semantic and result type", "input type mismatch rejected");
}

void testDeterministicEvaluation()
{
    auto source = evaluationProgram();
    std::string error = "stale";
    const auto admitted = admit(source, error);
    check(admitted.has_value(), "uniform scalar and color program is admitted");
    if (!admitted)
        return;

    MaterialEvaluationInputs inputs;
    inputs.timeSeconds = 1.5f;
    inputs.audioBass = 0.25f;
    inputs.controlCount = 1;
    inputs.controls[0] = 0.8f;

    const auto firstRun = evaluateReference(*admitted, inputs, error);
    const auto secondRun = evaluateReference(*admitted, inputs, error);
    check(firstRun.has_value() && secondRun.has_value(),
          "admitted uniform program evaluates for explicit runtime inputs");
    check(error.empty(), "successful evaluation clears an old diagnostic");
    if (!firstRun || !secondRun)
        return;

    check(std::memcmp(&*firstRun, &*secondRun, sizeof(*firstRun)) == 0,
          "repeated reference evaluation produces bit-identical fixed PBR blocks");
    check(firstRun->baseColorMetallic[0] == 0.25f
              && firstRun->baseColorMetallic[1] == 0.8f
              && firstRun->baseColorMetallic[2] == 0.25f,
          "base color remains linear RGB with no transfer-function conversion");
    check(firstRun->baseColorMetallic[3] == 0.8f
              && firstRun->emissionRoughness[0] == 0.3125f
              && firstRun->emissionRoughness[1] == 0.525f
              && firstRun->emissionRoughness[2] == 1.5f
              && firstRun->emissionRoughness[3] == 0.25f,
          "time, audio, control, and exact operation order drive the PBR block");
    check(firstRun->normalOpacity == std::array<float, 4> { 0.0f, 0.0f, 1.0f, 1.0f }
              && firstRun->transmissionIorClearcoat[0] == 0.0f
              && firstRun->transmissionIorClearcoat[1] == 1.5f
              && firstRun->transmissionIorClearcoat[2] == 0.525f
              && firstRun->identifiers[0] == 0xdeadbeefu,
          "fixed PBR lanes carry normalized normal, scalar outputs, and material ID");

    auto modulatedInputs = inputs;
    modulatedInputs.controls[0] = 0.4f;
    const auto modulated = evaluateReference(*admitted, modulatedInputs, error);
    check(modulated && modulated->baseColorMetallic[1] == 0.4f
              && modulated->emissionRoughness[0] == 0.625f
              && modulated->emissionRoughness[1] == 0.325f,
          "changing one explicit control deterministically modulates color outputs");

    source.operations[7].literal[0] = 0.9f;
    const auto afterSourceMutation = evaluateReference(*admitted, inputs, error);
    check(afterSourceMutation
              && std::memcmp(&*firstRun, &*afterSourceMutation, sizeof(*firstRun)) == 0,
          "source mutation cannot change immutable admitted evaluation");
}

void testInputMappingAndCompleteValueVocabulary()
{
    struct InputCase
    {
        InputSemantic semantic;
        float value;
    };
    constexpr std::array<InputCase, 7> cases {{
        { InputSemantic::Time, 0.125f },
        { InputSemantic::AudioLevel, 0.25f },
        { InputSemantic::AudioBass, 0.375f },
        { InputSemantic::AudioMid, 0.5f },
        { InputSemantic::AudioTreble, 0.625f },
        { InputSemantic::AudioBeat, 0.75f },
        { InputSemantic::Control, 0.875f }
    }};

    for (const auto& inputCase : cases)
    {
        std::string error;
        const auto admitted = admit(programForInput(inputCase.semantic), error);
        check(admitted.has_value(), "uniform reference-input fixture is admitted");
        if (!admitted)
            continue;

        MaterialEvaluationInputs inputs;
        switch (inputCase.semantic)
        {
            case InputSemantic::Time: inputs.timeSeconds = inputCase.value; break;
            case InputSemantic::AudioLevel: inputs.audioLevel = inputCase.value; break;
            case InputSemantic::AudioBass: inputs.audioBass = inputCase.value; break;
            case InputSemantic::AudioMid: inputs.audioMid = inputCase.value; break;
            case InputSemantic::AudioTreble: inputs.audioTreble = inputCase.value; break;
            case InputSemantic::AudioBeat: inputs.audioBeat = inputCase.value; break;
            case InputSemantic::Control:
                inputs.controlCount = 1;
                inputs.controls[0] = inputCase.value;
                break;
            default: break;
        }
        const auto block = evaluateReference(*admitted, inputs, error);
        check(block && block->emissionRoughness[0] == inputCase.value,
              "each uniform input semantic maps to its exact runtime value");
    }

    const auto program = completeValueEvaluationProgram();
    std::set<OperationKind> present;
    for (const auto& operation : program.operations)
        present.insert(operation.kind);
    for (const auto kind : kOperationKinds)
        if (kind != OperationKind::TextureSample2D)
            check(present.count(kind) == 1,
                  "complete value fixture covers every non-texture operation kind");

    std::string error;
    const auto admitted = admit(program, error);
    check(admitted.has_value(), "complete scalar/color operation fixture is admitted");
    if (!admitted)
        return;

    MaterialEvaluationInputs inputs;
    inputs.audioBass = 0.25f;
    inputs.controlCount = 1;
    inputs.controls[0] = 0.75f;
    const auto block = evaluateReference(*admitted, inputs, error);
    check(block.has_value(), "all admitted value-only operations evaluate");
    if (!block)
        return;

    check(block->baseColorMetallic
              == std::array<float, 4> { 0.375f, 0.296875f, 0.27734375f, 0.75f },
          "complete value evaluation preserves exact linear base-color lanes");
    check(block->emissionRoughness
              == std::array<float, 4> { 0.25f, 0.3125f, 0.203125f, 0.25f }
              && block->normalOpacity[3] == 0.5f
              && block->transmissionIorClearcoat
                  == std::array<float, 4> { 0.203125f, 1.0f, 0.3125f, 0.0f }
              && block->identifiers[0] == 0x10203040u,
          "complete value evaluation preserves the remaining material outputs");
    const auto normalLength = std::sqrt(
        block->normalOpacity[0] * block->normalOpacity[0]
        + block->normalOpacity[1] * block->normalOpacity[1]
        + block->normalOpacity[2] * block->normalOpacity[2]);
    check(std::abs(normalLength - 1.0f) <= 1.0e-6f,
          "complete value evaluation emits a unit normal");
}

void testMalformedEvaluationInputsAndBounds()
{
    std::string error;
    const auto admitted = admit(evaluationProgram(), error);
    check(admitted.has_value(), "runtime validation fixture is admitted");
    if (!admitted)
        return;

    MaterialEvaluationInputs valid;
    valid.timeSeconds = 1.5f;
    valid.audioBass = 0.25f;
    valid.controlCount = 1;
    valid.controls[0] = 0.8f;

    auto malformed = valid;
    malformed.timeSeconds = std::numeric_limits<float>::quiet_NaN();
    expectEvaluationRejected(*admitted, malformed, "time input",
                             "non-finite runtime time is rejected");
    malformed = valid;
    malformed.audioBass = 1.01f;
    expectEvaluationRejected(*admitted, malformed, "outside [0, 1]",
                             "audio above its closed hard range is rejected");
    malformed = valid;
    malformed.controlCount = 0;
    malformed.controls[0] = 0.0f;
    expectEvaluationRejected(*admitted, malformed, "control input is missing",
                             "missing referenced runtime control is rejected");
    malformed = valid;
    malformed.controlCount = static_cast<std::uint16_t>(kMaximumControlInputs + 1);
    expectEvaluationRejected(*admitted, malformed, "control count exceeds",
                             "runtime control count above the hard bound is rejected");
    malformed = valid;
    malformed.controls[0] = std::numeric_limits<float>::infinity();
    expectEvaluationRejected(*admitted, malformed, "control input is non-finite",
                             "non-finite runtime control is rejected");
    malformed = valid;
    malformed.controls[1] = 1.0f;
    expectEvaluationRejected(*admitted, malformed, "non-zero unused controls",
                             "non-zero unused runtime control is rejected");
    malformed = valid;
    malformed.controls[0] = 0.0f;
    expectEvaluationRejected(*admitted, malformed, "division by zero",
                             "runtime division by zero fails closed");

    auto exact = valid;
    exact.timeSeconds = kMaximumEvaluationMagnitude;
    exact.audioBass = 0.0f;
    exact.controls[0] = kMaximumEvaluationMagnitude;
    check(evaluateReference(*admitted, exact, error).has_value(),
          "runtime time and control values at the exact magnitude bound evaluate");
    exact.controls[0] = std::nextafter(kMaximumEvaluationMagnitude,
                                      std::numeric_limits<float>::infinity());
    expectEvaluationRejected(*admitted, exact, "out of bounds",
                             "runtime control above the exact magnitude bound is rejected");

    const auto boundedProgram = admit(hardBoundsProgram(), error);
    check(boundedProgram.has_value(), "PBR output-bound fixture is admitted");
    if (boundedProgram)
    {
        MaterialEvaluationInputs noInputs;
        const auto boundedBlock = evaluateReference(*boundedProgram, noInputs, error);
        check(boundedBlock
                  && boundedBlock->baseColorMetallic
                      == std::array<float, 4> { 0.0f, 1.0f, 0.5f, 0.0f }
                  && boundedBlock->emissionRoughness
                      == std::array<float, 4> {
                          0.0f, kMaximumLinearColorComponent,
                          kMaximumLinearColorComponent, 1.0f }
                  && boundedBlock->normalOpacity
                      == std::array<float, 4> { 0.0f, 0.0f, 1.0f, 1.0f }
                  && boundedBlock->transmissionIorClearcoat
                      == std::array<float, 4> {
                          0.0f, kMaximumMaterialIor, 1.0f, 0.0f }
                  && boundedBlock->identifiers[0]
                      == std::numeric_limits<std::uint32_t>::max(),
              "PBR output lanes clamp to documented finite hard bounds");
    }

    const auto overflowing = admit(overflowingEvaluationProgram(), error);
    check(overflowing.has_value(), "runtime overflow fixture is structurally admitted");
    if (overflowing)
    {
        MaterialEvaluationInputs noInputs;
        expectEvaluationRejected(*overflowing, noInputs, "out-of-bounds value",
                                 "operation result above the hard magnitude bound is rejected");
    }

    auto reversedClamp = completeValueEvaluationProgram();
    reversedClamp.operations[8].inputs = { 7, 3, 2, 0 };
    const auto reversedClampProgram = admit(reversedClamp, error);
    check(reversedClampProgram.has_value(),
          "runtime reversed-clamp fixture is structurally admitted");
    if (reversedClampProgram)
    {
        MaterialEvaluationInputs inputs;
        inputs.audioBass = 0.25f;
        inputs.controlCount = 1;
        inputs.controls[0] = 0.75f;
        expectEvaluationRejected(*reversedClampProgram, inputs, "clamp bounds are reversed",
                                 "runtime clamp with reversed bounds fails closed");
    }

    const auto textured = admit(textureEvaluationProgram(), error);
    check(textured.has_value(), "texture fixture remains structurally admitted");
    if (textured)
    {
        MaterialEvaluationInputs noInputs;
        expectEvaluationRejected(*textured, noInputs, "does not sample textures",
                                 "reference value evaluation does not claim texture or GPU rendering");
    }

    const auto fragmentInput = admit(programForInput(InputSemantic::Position), error);
    check(fragmentInput.has_value(), "per-fragment fixture remains structurally admitted");
    if (fragmentInput)
    {
        MaterialEvaluationInputs noInputs;
        expectEvaluationRejected(*fragmentInput, noInputs, "per-fragment inputs",
                                 "reference value evaluation does not claim per-fragment shading");
    }
}

void testExactBounds()
{
    std::string error;
    const auto maxOperations = maximumOperationProgram();
    check(maxOperations.operations.size() == kMaximumOperations,
          "maximum-operation fixture reaches the exact hard bound");
    const auto admittedOperations = admit(maxOperations, error);
    check(admittedOperations.has_value(), "exact hard operation bound is admitted");

    auto tooManyOperations = maxOperations;
    appendScalar(tooManyOperations, 1.0f);
    expectRejected(tooManyOperations, "operation limit exceeded",
                   "one operation above the hard bound rejected");

    const auto maxDepth = maximumDepthProgram();
    const auto admittedDepth = admit(maxDepth, error);
    check(admittedDepth && admittedDepth->maximumDepth() == kMaximumDepth,
          "exact hard graph depth is admitted");
    AdmissionLimits shallow;
    shallow.depth = kMaximumDepth - 1;
    expectRejected(maxDepth, "depth limit exceeded",
                   "one level below required depth rejects the graph", shallow);

    auto maxTexture = comprehensiveProgram();
    check(admit(maxTexture, error).has_value(),
          "highest valid texture slot index is admitted");
    maxTexture.operations[1].parameter
        = static_cast<std::uint16_t>(kMaximumTextureSlots);
    expectRejected(maxTexture, "unadmitted texture slot",
                   "texture slot equal to slot count rejected");

    auto tooManyTextures = comprehensiveProgram();
    tooManyTextures.textureSlotCount
        = static_cast<std::uint16_t>(kMaximumTextureSlots + 1);
    expectRejected(tooManyTextures, "texture slot limit exceeded",
                   "texture slot count above hard bound rejected");

    AdmissionLimits fewerTextures;
    fewerTextures.textureSlots = kMaximumTextureSlots - 1;
    expectRejected(comprehensiveProgram(), "texture slot limit exceeded",
                   "caller texture limit is exact", fewerTextures);

    auto component = comprehensiveProgram();
    check(component.operations[22].parameter == 3,
          "component fixture uses the highest vec4 lane");
    component.operations[22].parameter = 4;
    expectRejected(component, "invalid vector lane",
                   "component lane equal to vector width rejected");

    auto strayParameter = comprehensiveProgram();
    strayParameter.operations[4].parameter = 1;
    expectRejected(strayParameter, "unsupported payload fields",
                   "non-parameter operation rejects a parameter payload");

    auto tooManyInputs = comprehensiveProgram();
    tooManyInputs.operations[4].inputCount
        = static_cast<std::uint8_t>(kMaximumOperationInputs + 1);
    expectRejected(tooManyInputs, "input limit exceeded",
                   "operation input count above four rejected");

    auto control = programForInput(InputSemantic::Control);
    control.operations.front().parameter
        = static_cast<std::uint16_t>(kMaximumControlInputs - 1);
    check(admit(control, error).has_value(),
          "highest valid control input index is admitted");
    control.operations.front().parameter
        = static_cast<std::uint16_t>(kMaximumControlInputs);
    expectRejected(control, "control input limit exceeded",
                   "control input index at the hard bound is rejected");
    control.operations.front().parameter = 1;
    AdmissionLimits oneControl;
    oneControl.controlInputs = 1;
    expectRejected(control, "control input limit exceeded",
                   "caller control input limit is exact", oneControl);

    AdmissionLimits invalidLimits;
    invalidLimits.textureSlots = kMaximumTextureSlots + 1;
    expectRejected(comprehensiveProgram(), "limits exceed hard contract bounds",
                   "caller cannot raise the texture hard bound", invalidLimits);
    invalidLimits = {};
    invalidLimits.operations = 0;
    expectRejected(comprehensiveProgram(), "limits exceed hard contract bounds",
                   "zero operation policy limit rejected", invalidLimits);
    invalidLimits = {};
    invalidLimits.depth = 0;
    expectRejected(comprehensiveProgram(), "limits exceed hard contract bounds",
                   "zero depth policy limit rejected", invalidLimits);
    invalidLimits = {};
    invalidLimits.operations = kMaximumOperations + 1;
    expectRejected(comprehensiveProgram(), "limits exceed hard contract bounds",
                   "caller cannot raise the operation hard bound", invalidLimits);
    invalidLimits = {};
    invalidLimits.depth = kMaximumDepth + 1;
    expectRejected(comprehensiveProgram(), "limits exceed hard contract bounds",
                   "caller cannot raise the depth hard bound", invalidLimits);
    invalidLimits = {};
    invalidLimits.controlInputs = 0;
    check(admit(comprehensiveProgram(), error, invalidLimits).has_value(),
          "zero control-input policy admits programs without controls");
    expectRejected(programForInput(InputSemantic::Control), "control input limit exceeded",
                   "zero control-input policy rejects programs with controls", invalidLimits);
    invalidLimits = {};
    invalidLimits.controlInputs = kMaximumControlInputs + 1;
    expectRejected(comprehensiveProgram(), "limits exceed hard contract bounds",
                   "caller cannot raise the control-input hard bound", invalidLimits);
}

void testMalformedGraphs()
{
    ProgramDescription empty;
    expectRejected(empty, "operation limit exceeded", "empty program rejected");

    auto wrongVersion = comprehensiveProgram();
    ++wrongVersion.version;
    expectRejected(wrongVersion, "wire version is unsupported",
                   "unknown wire version rejected");

    auto missingOutput = comprehensiveProgram();
    missingOutput.outputs[0] = 0;
    expectRejected(missingOutput, "output is missing or out of range",
                   "missing output rejected");

    auto outOfRangeOutput = comprehensiveProgram();
    outOfRangeOutput.outputs[0] = 100;
    expectRejected(outOfRangeOutput, "output is missing or out of range",
                   "out-of-range output rejected");

    auto wrongOutputType = comprehensiveProgram();
    wrongOutputType.outputs[0] = wrongOutputType.outputs[1];
    expectRejected(wrongOutputType, "output has the wrong explicit type",
                   "wrong output type rejected");

    auto nonContiguous = comprehensiveProgram();
    nonContiguous.operations[4].id = 99;
    expectRejected(nonContiguous, "IDs must be contiguous and ordered",
                   "non-contiguous operation ID rejected");

    auto selfCycle = comprehensiveProgram();
    selfCycle.operations[4].inputs[0] = selfCycle.operations[4].id;
    expectRejected(selfCycle, "ordered acyclic graph", "self-cycle rejected");

    auto forwardCycle = comprehensiveProgram();
    forwardCycle.operations[4].inputs[0] = forwardCycle.operations[5].id;
    forwardCycle.operations[5].inputs[0] = forwardCycle.operations[4].id;
    expectRejected(forwardCycle, "ordered acyclic graph", "two-operation cycle rejected");

    auto missingReference = comprehensiveProgram();
    missingReference.operations[4].inputs[0] = 100;
    expectRejected(missingReference, "ordered acyclic graph", "missing input reference rejected");

    auto unusedInput = comprehensiveProgram();
    unusedInput.operations[3].inputs[0] = 1;
    expectRejected(unusedInput, "non-zero unused inputs", "non-zero unused input rejected");

    auto unreachable = comprehensiveProgram();
    appendScalar(unreachable, 1.0f);
    expectRejected(unreachable, "unreachable operations", "unreachable operation rejected");

    auto unknownKind = comprehensiveProgram();
    unknownKind.operations[3].kind = static_cast<OperationKind>(255);
    expectRejected(unknownKind, "kind is unsupported", "unknown operation kind rejected");

    auto unknownType = comprehensiveProgram();
    unknownType.operations[3].resultType = static_cast<ValueType>(255);
    expectRejected(unknownType, "unsupported result type", "unknown result type rejected");

    auto wrongArity = comprehensiveProgram();
    wrongArity.operations[4].inputCount = 1;
    wrongArity.operations[4].inputs[1] = 0;
    expectRejected(wrongArity, "invalid arity", "wrong operation arity rejected");

    auto nonFinite = comprehensiveProgram();
    nonFinite.operations[3].literal[0] = std::numeric_limits<float>::infinity();
    expectRejected(nonFinite, "non-finite component", "infinite literal rejected");
    nonFinite.operations[3].literal[0] = std::numeric_limits<float>::quiet_NaN();
    expectRejected(nonFinite, "non-finite component", "NaN literal rejected");

    auto unusedLiteral = comprehensiveProgram();
    unusedLiteral.operations[3].literal[1] = 1.0f;
    expectRejected(unusedLiteral, "non-zero unused components",
                   "non-zero unused literal component rejected");

    auto inputPayload = programForInput(InputSemantic::Time);
    inputPayload.operations.front().unsignedLiteral = 1;
    expectRejected(inputPayload, "canonical input carries unsupported payload fields",
                   "input rejects unrelated payload fields");

    auto texturePayload = comprehensiveProgram();
    texturePayload.operations[1].literal[0] = 1.0f;
    expectRejected(texturePayload, "texture sample carries unsupported payload fields",
                   "texture sample rejects unrelated payload fields");
}
} // namespace

int main()
{
    testStableVocabulary();
    testAdmissionAndDigest();
    testEveryInput();
    testDeterministicEvaluation();
    testInputMappingAndCompleteValueVocabulary();
    testMalformedEvaluationInputsAndBounds();
    testExactBounds();
    testMalformedGraphs();

    if (failures != 0)
    {
        std::cerr << failures << " surface material admission checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "surface material admission checks passed\n";
    return EXIT_SUCCESS;
}
