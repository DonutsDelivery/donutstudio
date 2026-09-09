#pragma once

#include "../../shared/SurfaceMaterialIR.h"
#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace surfacematerial
{
class AdmittedSurfaceMaterialIR;

std::optional<AdmittedSurfaceMaterialIR> admit(
    const ProgramDescription& untrusted,
    std::string& error,
    const AdmissionLimits& limits = {});

// Deterministic value-only oracle. This is not the production material backend.
std::optional<SurfacePbrParameterBlock> evaluateReference(
    const AdmittedSurfaceMaterialIR& admitted,
    const MaterialEvaluationInputs& inputs,
    std::string& error);

class AdmittedSurfaceMaterialIR final
{
public:
    AdmittedSurfaceMaterialIR(const AdmittedSurfaceMaterialIR&) = default;
    AdmittedSurfaceMaterialIR(AdmittedSurfaceMaterialIR&&) = default;
    AdmittedSurfaceMaterialIR& operator=(const AdmittedSurfaceMaterialIR&) = delete;
    AdmittedSurfaceMaterialIR& operator=(AdmittedSurfaceMaterialIR&&) = delete;

    const ProgramDescription& program() const noexcept { return program_; }
    const std::string& structuralDigest() const noexcept { return digest_; }
    std::size_t maximumDepth() const noexcept { return maximumDepth_; }

private:
    friend std::optional<AdmittedSurfaceMaterialIR> admit(
        const ProgramDescription&, std::string&, const AdmissionLimits&);

    AdmittedSurfaceMaterialIR(ProgramDescription program,
                              std::string digest,
                              std::size_t maximumDepth)
        : program_(std::move(program)),
          digest_(std::move(digest)),
          maximumDepth_(maximumDepth)
    {
    }

    const ProgramDescription program_;
    const std::string digest_;
    const std::size_t maximumDepth_;
};

namespace detail
{
inline bool reject(std::string& error, const char* diagnostic)
{
    error = diagnostic;
    return false;
}

inline bool isFloating(ValueType type) noexcept
{
    return type == ValueType::Scalar || type == ValueType::Vec2
        || type == ValueType::Vec3 || type == ValueType::Vec4;
}

inline bool isVector(ValueType type) noexcept
{
    return type == ValueType::Vec2 || type == ValueType::Vec3
        || type == ValueType::Vec4;
}

inline std::size_t width(ValueType type) noexcept
{
    switch (type)
    {
        case ValueType::Scalar: return 1;
        case ValueType::Vec2: return 2;
        case ValueType::Vec3: return 3;
        case ValueType::Vec4: return 4;
        case ValueType::UInt:
        case ValueType::Invalid:
            break;
    }
    return 0;
}

inline bool knownType(ValueType type) noexcept
{
    return isFloating(type) || type == ValueType::UInt;
}

inline bool literalsAreZero(const Operation& operation) noexcept
{
    return std::all_of(operation.literal.begin(), operation.literal.end(),
                       [] (float value) { return value == 0.0f; });
}

inline bool neutralPayload(const Operation& operation) noexcept
{
    return literalsAreZero(operation)
        && operation.semantic == InputSemantic::Invalid
        && operation.parameter == 0
        && operation.unsignedLiteral == 0;
}

inline bool hasArity(const Operation& operation,
                     std::uint8_t expected,
                     std::string& error)
{
    if (operation.inputCount != expected)
        return reject(error, "surface material operation has invalid arity");
    return true;
}

inline bool broadcastResult(ValueType left,
                            ValueType right,
                            ValueType result) noexcept
{
    if (!isFloating(result))
        return false;
    return (left == result && right == result)
        || (left == ValueType::Scalar && right == result)
        || (right == ValueType::Scalar && left == result);
}

inline bool validateOperation(const ProgramDescription& program,
                              const Operation& operation,
                              std::string& error)
{
    const auto inputTypeAt = [&] (std::size_t input) {
        return program.operations[operation.inputs[input] - 1].resultType;
    };
    const auto requireNeutral = [&] {
        return neutralPayload(operation)
            || reject(error, "surface material operation carries unsupported payload fields");
    };

    switch (operation.kind)
    {
        case OperationKind::FloatConstant:
        {
            if (!hasArity(operation, 0, error) || !isFloating(operation.resultType))
                return error.empty()
                    ? reject(error, "float constant has an invalid result type") : false;
            if (operation.semantic != InputSemantic::Invalid
                || operation.parameter != 0 || operation.unsignedLiteral != 0)
                return reject(error, "float constant carries unsupported payload fields");
            const auto componentCount = width(operation.resultType);
            for (std::size_t i = 0; i < operation.literal.size(); ++i)
            {
                if (i < componentCount && !std::isfinite(operation.literal[i]))
                    return reject(error, "float constant contains a non-finite component");
                if (i >= componentCount && operation.literal[i] != 0.0f)
                    return reject(error, "float constant has non-zero unused components");
            }
            return true;
        }
        case OperationKind::UIntConstant:
            if (!hasArity(operation, 0, error) || operation.resultType != ValueType::UInt)
                return error.empty()
                    ? reject(error, "unsigned constant has an invalid result type") : false;
            if (!literalsAreZero(operation)
                || operation.semantic != InputSemantic::Invalid || operation.parameter != 0)
                return reject(error, "unsigned constant carries unsupported payload fields");
            return true;
        case OperationKind::Input:
        {
            if (!hasArity(operation, 0, error))
                return false;
            const auto canonicalType = inputType(operation.semantic);
            if (canonicalType == ValueType::Invalid || operation.resultType != canonicalType)
                return reject(error, "canonical input semantic and result type do not match");
            if (!literalsAreZero(operation)
                || (operation.semantic != InputSemantic::Control && operation.parameter != 0)
                || operation.parameter >= kMaximumControlInputs
                || operation.unsignedLiteral != 0)
                return reject(error, "canonical input carries unsupported payload fields");
            return true;
        }
        case OperationKind::TextureSample2D:
            if (!hasArity(operation, 1, error))
                return false;
            if (operation.resultType != ValueType::Vec4
                || inputTypeAt(0) != ValueType::Vec2)
                return reject(error, "texture sample requires a vec2 UV and returns vec4");
            if (operation.parameter >= program.textureSlotCount)
                return reject(error, "texture sample references an unadmitted texture slot");
            if (!literalsAreZero(operation)
                || operation.semantic != InputSemantic::Invalid
                || operation.unsignedLiteral != 0)
                return reject(error, "texture sample carries unsupported payload fields");
            return true;
        case OperationKind::Add:
        case OperationKind::Subtract:
        case OperationKind::Multiply:
        case OperationKind::Divide:
        case OperationKind::Minimum:
        case OperationKind::Maximum:
            if (!hasArity(operation, 2, error))
                return false;
            if (!broadcastResult(inputTypeAt(0), inputTypeAt(1), operation.resultType))
                return reject(error, "binary operation has incompatible explicit types");
            return requireNeutral();
        case OperationKind::Clamp:
            if (!hasArity(operation, 3, error))
                return false;
            if (!isFloating(operation.resultType)
                || inputTypeAt(0) != operation.resultType
                || (inputTypeAt(1) != operation.resultType
                    && inputTypeAt(1) != ValueType::Scalar)
                || (inputTypeAt(2) != operation.resultType
                    && inputTypeAt(2) != ValueType::Scalar))
                return reject(error, "clamp operation has incompatible explicit types");
            return requireNeutral();
        case OperationKind::Mix:
            if (!hasArity(operation, 3, error))
                return false;
            if (!isFloating(operation.resultType)
                || inputTypeAt(0) != operation.resultType
                || inputTypeAt(1) != operation.resultType
                || inputTypeAt(2) != ValueType::Scalar)
                return reject(error, "mix operation has incompatible explicit types");
            return requireNeutral();
        case OperationKind::Dot:
            if (!hasArity(operation, 2, error))
                return false;
            if (operation.resultType != ValueType::Scalar
                || !isVector(inputTypeAt(0)) || inputTypeAt(0) != inputTypeAt(1))
                return reject(error, "dot operation requires equal vector input types");
            return requireNeutral();
        case OperationKind::Normalize:
            if (!hasArity(operation, 1, error))
                return false;
            if (!isVector(operation.resultType) || inputTypeAt(0) != operation.resultType)
                return reject(error, "normalize operation requires one matching vector type");
            return requireNeutral();
        case OperationKind::Length:
            if (!hasArity(operation, 1, error))
                return false;
            if (operation.resultType != ValueType::Scalar || !isVector(inputTypeAt(0)))
                return reject(error, "length operation requires a vector and returns scalar");
            return requireNeutral();
        case OperationKind::Absolute:
            if (!hasArity(operation, 1, error))
                return false;
            if (!isFloating(operation.resultType) || inputTypeAt(0) != operation.resultType)
                return reject(error, "absolute operation requires one matching float type");
            return requireNeutral();
        case OperationKind::Power:
            if (!hasArity(operation, 2, error))
                return false;
            if (!isFloating(operation.resultType)
                || inputTypeAt(0) != operation.resultType
                || (inputTypeAt(1) != ValueType::Scalar
                    && inputTypeAt(1) != operation.resultType))
                return reject(error, "power operation has incompatible explicit types");
            return requireNeutral();
        case OperationKind::Component:
            if (!hasArity(operation, 1, error))
                return false;
            if (operation.resultType != ValueType::Scalar || !isVector(inputTypeAt(0))
                || operation.parameter >= width(inputTypeAt(0)))
                return reject(error, "component operation has an invalid vector lane");
            if (!literalsAreZero(operation)
                || operation.semantic != InputSemantic::Invalid
                || operation.unsignedLiteral != 0)
                return reject(error, "component operation carries unsupported payload fields");
            return true;
        case OperationKind::ComposeVec2:
        case OperationKind::ComposeVec3:
        case OperationKind::ComposeVec4:
        {
            const auto expectedType = operation.kind == OperationKind::ComposeVec2
                ? ValueType::Vec2 : operation.kind == OperationKind::ComposeVec3
                ? ValueType::Vec3 : ValueType::Vec4;
            const auto expectedInputs = static_cast<std::uint8_t>(width(expectedType));
            if (!hasArity(operation, expectedInputs, error))
                return false;
            if (operation.resultType != expectedType)
                return reject(error, "compose operation has the wrong vector result type");
            for (std::size_t i = 0; i < expectedInputs; ++i)
                if (inputTypeAt(i) != ValueType::Scalar)
                    return reject(error, "compose operation requires scalar inputs");
            return requireNeutral();
        }
        case OperationKind::Invalid:
            break;
    }
    return reject(error, "surface material operation kind is unsupported");
}

inline bool validate(const ProgramDescription& program,
                     const AdmissionLimits& limits,
                     std::size_t& maximumDepth,
                     std::string& error)
{
    error.clear();
    maximumDepth = 0;

    if (program.version != kWireVersion)
        return reject(error, "surface material IR wire version is unsupported");
    if (limits.textureSlots > kMaximumTextureSlots
        || limits.operations == 0 || limits.operations > kMaximumOperations
        || limits.depth == 0 || limits.depth > kMaximumDepth
        || limits.controlInputs > kMaximumControlInputs)
        return reject(error, "surface material admission limits exceed hard contract bounds");
    if (program.textureSlotCount > limits.textureSlots)
        return reject(error, "surface material texture slot limit exceeded");
    if (program.operations.empty() || program.operations.size() > limits.operations)
        return reject(error, "surface material operation limit exceeded");

    std::vector<std::size_t> depths(program.operations.size(), 0);
    for (std::size_t index = 0; index < program.operations.size(); ++index)
    {
        const auto& operation = program.operations[index];
        if (operation.id != index + 1)
            return reject(error, "surface material operation IDs must be contiguous and ordered");
        if (!knownType(operation.resultType))
            return reject(error, "surface material operation has an unsupported result type");
        if (operation.inputCount > kMaximumOperationInputs)
            return reject(error, "surface material operation input limit exceeded");
        if (operation.kind == OperationKind::Input
            && operation.semantic == InputSemantic::Control
            && operation.parameter >= limits.controlInputs)
            return reject(error, "surface material control input limit exceeded");

        std::size_t operationDepth = 1;
        for (std::size_t input = 0; input < operation.inputs.size(); ++input)
        {
            if (input >= operation.inputCount)
            {
                if (operation.inputs[input] != 0)
                    return reject(error, "surface material operation has non-zero unused inputs");
                continue;
            }
            const auto source = operation.inputs[input];
            if (source == 0 || source >= operation.id)
                return reject(error, "surface material operations must form an ordered acyclic graph");
            operationDepth = std::max(operationDepth, depths[source - 1] + 1);
        }
        if (operationDepth > limits.depth)
            return reject(error, "surface material operation depth limit exceeded");
        depths[index] = operationDepth;
        maximumDepth = std::max(maximumDepth, operationDepth);

        if (!validateOperation(program, operation, error))
            return false;
    }

    std::vector<bool> reachable(program.operations.size(), false);
    std::vector<ValueId> pending;
    pending.reserve(program.operations.size());
    for (const auto output : kSurfaceOutputs)
    {
        const auto value = program.output(output);
        if (value == 0 || value > program.operations.size())
            return reject(error, "surface material output is missing or out of range");
        if (program.operations[value - 1].resultType != outputType(output))
            return reject(error, "surface material output has the wrong explicit type");
        pending.push_back(value);
    }

    while (!pending.empty())
    {
        const auto value = pending.back();
        pending.pop_back();
        if (reachable[value - 1])
            continue;
        reachable[value - 1] = true;
        const auto& operation = program.operations[value - 1];
        for (std::size_t input = 0; input < operation.inputCount; ++input)
            pending.push_back(operation.inputs[input]);
    }
    if (std::find(reachable.begin(), reachable.end(), false) != reachable.end())
        return reject(error, "surface material graph contains unreachable operations");
    return true;
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

inline void appendFloat(std::vector<std::uint8_t>& bytes, float value)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t)
                  && std::numeric_limits<float>::is_iec559,
                  "Surface material digest requires IEEE-754 binary32 floats");
    if (value == 0.0f)
        value = 0.0f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(bytes, bits);
}

inline std::string structuralDigest(const ProgramDescription& program)
{
    static constexpr char domain[] = "DonutStudio/SurfaceMaterialIR/Structure/v1";
    std::vector<std::uint8_t> bytes(std::begin(domain), std::end(domain) - 1);
    appendU32(bytes, program.version);
    appendU16(bytes, program.textureSlotCount);
    appendU32(bytes, static_cast<std::uint32_t>(program.operations.size()));
    for (const auto& operation : program.operations)
    {
        appendU16(bytes, operation.id);
        appendU8(bytes, static_cast<std::uint8_t>(operation.kind));
        appendU8(bytes, static_cast<std::uint8_t>(operation.resultType));
        appendU8(bytes, operation.inputCount);
        for (const auto input : operation.inputs)
            appendU16(bytes, input);
        for (const auto value : operation.literal)
            appendFloat(bytes, value);
        appendU8(bytes, static_cast<std::uint8_t>(operation.semantic));
        appendU16(bytes, operation.parameter);
        appendU32(bytes, operation.unsignedLiteral);
    }
    for (const auto output : program.outputs)
        appendU16(bytes, output);

    videohelper::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finishHex();
}

struct EvaluatedValue
{
    ValueType type = ValueType::Invalid;
    std::array<float, 4> floating {};
    std::uint32_t unsignedValue = 0;
};

inline float orderedAdd(float left, float right) noexcept
{
    volatile float result = left + right;
    return result;
}

inline float orderedSubtract(float left, float right) noexcept
{
    volatile float result = left - right;
    return result;
}

inline float orderedMultiply(float left, float right) noexcept
{
    volatile float result = left * right;
    return result;
}

inline float orderedDivide(float left, float right) noexcept
{
    volatile float result = left / right;
    return result;
}

inline bool bounded(float value) noexcept
{
    return std::isfinite(value)
        && std::abs(value) <= kMaximumEvaluationMagnitude;
}

inline bool validateEvaluationInputs(const MaterialEvaluationInputs& inputs,
                                     std::string& error)
{
    if (!bounded(inputs.timeSeconds))
        return reject(error, "surface material time input is non-finite or out of bounds");

    const std::array<float, 5> audio {
        inputs.audioLevel, inputs.audioBass, inputs.audioMid,
        inputs.audioTreble, inputs.audioBeat
    };
    for (const auto value : audio)
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
            return reject(error, "surface material audio input is outside [0, 1]");

    if (inputs.controlCount > kMaximumControlInputs)
        return reject(error, "surface material runtime control count exceeds the hard bound");
    for (std::size_t index = 0; index < inputs.controls.size(); ++index)
    {
        if (index < inputs.controlCount)
        {
            if (!bounded(inputs.controls[index]))
                return reject(error, "surface material control input is non-finite or out of bounds");
        }
        else if (inputs.controls[index] != 0.0f)
        {
            return reject(error, "surface material runtime input has non-zero unused controls");
        }
    }
    return true;
}

inline bool inputValue(const Operation& operation,
                       const MaterialEvaluationInputs& inputs,
                       EvaluatedValue& result,
                       std::string& error)
{
    result.type = ValueType::Scalar;
    switch (operation.semantic)
    {
        case InputSemantic::Time: result.floating[0] = inputs.timeSeconds; return true;
        case InputSemantic::AudioLevel: result.floating[0] = inputs.audioLevel; return true;
        case InputSemantic::AudioBass: result.floating[0] = inputs.audioBass; return true;
        case InputSemantic::AudioMid: result.floating[0] = inputs.audioMid; return true;
        case InputSemantic::AudioTreble: result.floating[0] = inputs.audioTreble; return true;
        case InputSemantic::AudioBeat: result.floating[0] = inputs.audioBeat; return true;
        case InputSemantic::Control:
            if (operation.parameter >= inputs.controlCount)
                return reject(error, "surface material runtime control input is missing");
            result.floating[0] = inputs.controls[operation.parameter];
            return true;
        case InputSemantic::Position:
        case InputSemantic::Normal:
        case InputSemantic::Tangent:
        case InputSemantic::TexCoord0:
        case InputSemantic::TexCoord1:
        case InputSemantic::VertexColor:
        case InputSemantic::ViewDirection:
            return reject(error, "surface material reference evaluation cannot consume per-fragment inputs");
        case InputSemantic::Invalid:
            break;
    }
    return reject(error, "surface material reference evaluation found an unsupported input");
}

inline float component(const EvaluatedValue& value, std::size_t lane) noexcept
{
    return value.floating[value.type == ValueType::Scalar ? 0 : lane];
}

inline bool boundedResult(const EvaluatedValue& result, std::string& error)
{
    if (result.type == ValueType::UInt)
        return true;
    for (std::size_t lane = 0; lane < width(result.type); ++lane)
        if (!bounded(result.floating[lane]))
            return reject(error, "surface material operation produced a non-finite or out-of-bounds value");
    return true;
}

inline bool evaluateOperation(const Operation& operation,
                              const std::array<EvaluatedValue, kMaximumOperations>& values,
                              const MaterialEvaluationInputs& inputs,
                              EvaluatedValue& result,
                              std::string& error)
{
    const auto value = [&] (std::size_t input) -> const EvaluatedValue& {
        return values[operation.inputs[input] - 1];
    };
    result.type = operation.resultType;

    switch (operation.kind)
    {
        case OperationKind::FloatConstant:
            result.floating = operation.literal;
            break;
        case OperationKind::UIntConstant:
            result.unsignedValue = operation.unsignedLiteral;
            break;
        case OperationKind::Input:
            if (!inputValue(operation, inputs, result, error))
                return false;
            break;
        case OperationKind::TextureSample2D:
            return reject(error, "surface material reference evaluation does not sample textures");
        case OperationKind::Add:
        case OperationKind::Subtract:
        case OperationKind::Multiply:
        case OperationKind::Divide:
        case OperationKind::Minimum:
        case OperationKind::Maximum:
            for (std::size_t lane = 0; lane < width(operation.resultType); ++lane)
            {
                const auto left = component(value(0), lane);
                const auto right = component(value(1), lane);
                switch (operation.kind)
                {
                    case OperationKind::Add: result.floating[lane] = orderedAdd(left, right); break;
                    case OperationKind::Subtract:
                        result.floating[lane] = orderedSubtract(left, right); break;
                    case OperationKind::Multiply:
                        result.floating[lane] = orderedMultiply(left, right); break;
                    case OperationKind::Divide:
                        if (right == 0.0f)
                            return reject(error, "surface material division by zero");
                        result.floating[lane] = orderedDivide(left, right);
                        break;
                    case OperationKind::Minimum:
                        result.floating[lane] = std::min(left, right); break;
                    case OperationKind::Maximum:
                        result.floating[lane] = std::max(left, right); break;
                    default: break;
                }
            }
            break;
        case OperationKind::Clamp:
            for (std::size_t lane = 0; lane < width(operation.resultType); ++lane)
            {
                const auto lower = component(value(1), lane);
                const auto upper = component(value(2), lane);
                if (lower > upper)
                    return reject(error, "surface material clamp bounds are reversed");
                result.floating[lane]
                    = std::min(std::max(component(value(0), lane), lower), upper);
            }
            break;
        case OperationKind::Mix:
            for (std::size_t lane = 0; lane < width(operation.resultType); ++lane)
            {
                const auto delta = orderedSubtract(component(value(1), lane),
                                                   component(value(0), lane));
                const auto scaled = orderedMultiply(delta, value(2).floating[0]);
                result.floating[lane]
                    = orderedAdd(component(value(0), lane), scaled);
            }
            break;
        case OperationKind::Dot:
        {
            float sum = 0.0f;
            for (std::size_t lane = 0; lane < width(value(0).type); ++lane)
                sum = orderedAdd(sum, orderedMultiply(value(0).floating[lane],
                                                      value(1).floating[lane]));
            result.floating[0] = sum;
            break;
        }
        case OperationKind::Normalize:
        case OperationKind::Length:
        {
            float squaredLength = 0.0f;
            for (std::size_t lane = 0; lane < width(value(0).type); ++lane)
                squaredLength = orderedAdd(
                    squaredLength,
                    orderedMultiply(value(0).floating[lane], value(0).floating[lane]));
            const auto vectorLength = std::sqrt(squaredLength);
            if (!std::isfinite(vectorLength))
                return reject(error, "surface material vector length is non-finite");
            if (operation.kind == OperationKind::Length)
            {
                result.floating[0] = vectorLength;
                break;
            }
            if (vectorLength == 0.0f)
                return reject(error, "surface material cannot normalize a zero vector");
            for (std::size_t lane = 0; lane < width(operation.resultType); ++lane)
                result.floating[lane]
                    = orderedDivide(value(0).floating[lane], vectorLength);
            break;
        }
        case OperationKind::Absolute:
            for (std::size_t lane = 0; lane < width(operation.resultType); ++lane)
                result.floating[lane] = std::abs(value(0).floating[lane]);
            break;
        case OperationKind::Power:
            for (std::size_t lane = 0; lane < width(operation.resultType); ++lane)
                result.floating[lane]
                    = std::pow(value(0).floating[lane], component(value(1), lane));
            break;
        case OperationKind::Component:
            result.floating[0] = value(0).floating[operation.parameter];
            break;
        case OperationKind::ComposeVec2:
        case OperationKind::ComposeVec3:
        case OperationKind::ComposeVec4:
            for (std::size_t lane = 0; lane < width(operation.resultType); ++lane)
                result.floating[lane] = value(lane).floating[0];
            break;
        case OperationKind::Invalid:
            return reject(error, "surface material reference evaluation found an unsupported operation");
    }
    return boundedResult(result, error);
}

inline float clampPbr(float value, float lower, float upper) noexcept
{
    return std::min(std::max(value, lower), upper);
}

inline bool makeParameterBlock(
    const ProgramDescription& program,
    const std::array<EvaluatedValue, kMaximumOperations>& values,
    SurfacePbrParameterBlock& block,
    std::string& error)
{
    const auto output = [&] (OutputSemantic semantic) -> const EvaluatedValue& {
        return values[program.output(semantic) - 1];
    };
    const auto& baseColor = output(OutputSemantic::BaseColor);
    const auto& emission = output(OutputSemantic::Emission);
    const auto& normal = output(OutputSemantic::Normal);

    float normalLengthSquared = 0.0f;
    for (std::size_t lane = 0; lane < 3; ++lane)
        normalLengthSquared = orderedAdd(
            normalLengthSquared,
            orderedMultiply(normal.floating[lane], normal.floating[lane]));
    const auto normalLength = std::sqrt(normalLengthSquared);
    if (!std::isfinite(normalLength) || normalLength == 0.0f)
        return reject(error, "surface material PBR normal is zero or non-finite");

    block = {};
    for (std::size_t lane = 0; lane < 3; ++lane)
    {
        block.baseColorMetallic[lane]
            = clampPbr(baseColor.floating[lane], 0.0f, 1.0f);
        block.emissionRoughness[lane]
            = clampPbr(emission.floating[lane], 0.0f, kMaximumLinearColorComponent);
        block.normalOpacity[lane]
            = orderedDivide(normal.floating[lane], normalLength);
    }
    block.baseColorMetallic[3]
        = clampPbr(output(OutputSemantic::Metallic).floating[0], 0.0f, 1.0f);
    block.emissionRoughness[3]
        = clampPbr(output(OutputSemantic::Roughness).floating[0], 0.0f, 1.0f);
    block.normalOpacity[3]
        = clampPbr(output(OutputSemantic::Opacity).floating[0], 0.0f, 1.0f);
    block.transmissionIorClearcoat[0]
        = clampPbr(output(OutputSemantic::Transmission).floating[0], 0.0f, 1.0f);
    block.transmissionIorClearcoat[1]
        = clampPbr(output(OutputSemantic::Ior).floating[0],
                   kMinimumMaterialIor, kMaximumMaterialIor);
    block.transmissionIorClearcoat[2]
        = clampPbr(output(OutputSemantic::Clearcoat).floating[0], 0.0f, 1.0f);
    block.identifiers[0] = output(OutputSemantic::MaterialId).unsignedValue;
    return true;
}

inline bool evaluateReferenceProgram(const ProgramDescription& program,
                                     const MaterialEvaluationInputs& inputs,
                                     SurfacePbrParameterBlock& block,
                                     std::string& error)
{
    error.clear();
    if (!validateEvaluationInputs(inputs, error))
        return false;

    std::array<EvaluatedValue, kMaximumOperations> values {};
    // Admission makes IDs contiguous, so ascending storage order is the stable
    // reference evaluation order.
    for (std::size_t index = 0; index < program.operations.size(); ++index)
        if (!evaluateOperation(program.operations[index], values, inputs,
                               values[index], error))
            return false;
    return makeParameterBlock(program, values, block, error);
}
} // namespace detail

inline std::optional<AdmittedSurfaceMaterialIR> admit(
    const ProgramDescription& untrusted,
    std::string& error,
    const AdmissionLimits& limits)
{
    std::size_t maximumDepth = 0;
    if (!detail::validate(untrusted, limits, maximumDepth, error))
        return std::nullopt;

    ProgramDescription candidate = untrusted;
    auto digest = detail::structuralDigest(candidate);
    return AdmittedSurfaceMaterialIR(
        std::move(candidate), std::move(digest), maximumDepth);
}

inline std::optional<SurfacePbrParameterBlock> evaluateReference(
    const AdmittedSurfaceMaterialIR& admitted,
    const MaterialEvaluationInputs& inputs,
    std::string& error)
{
    SurfacePbrParameterBlock block;
    if (!detail::evaluateReferenceProgram(admitted.program(), inputs, block, error))
        return std::nullopt;
    return block;
}
} // namespace surfacematerial
