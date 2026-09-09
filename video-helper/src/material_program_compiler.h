#pragma once

#include "../../shared/MaterialProgramIR.h"
#include "surface_material_admission.h"
#include "vertex_modifier_ir_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace videohelper::materialprogram
{
inline constexpr std::uint32_t kCompilerVersion = 1;
inline constexpr std::uint32_t kProgramRecordVersion
    = videowire::kMaterialProgramIrSchemaVersion;
inline constexpr std::size_t kMaximumSurfaceInstructions
    = surfacematerial::kMaximumOperations;
inline constexpr std::size_t kMaximumVertexInstructions = 512;
inline constexpr std::size_t kMaximumCombinedInstructions
    = kMaximumSurfaceInstructions + kMaximumVertexInstructions;
inline constexpr std::size_t kMaximumControlParameterSlots = 4096;
inline constexpr std::size_t kMaximumAudioParameterSlots = 64;
inline constexpr std::size_t kMaximumCanonicalPayloadBytes = 128 * 1024;
inline constexpr std::uint16_t kInvalidValueSlot
    = videowire::kMaterialProgramInvalidValueSlot;

using SurfaceInstruction = videowire::MaterialProgramSurfaceInstruction;
using VertexInstruction = videowire::MaterialProgramVertexInstruction;

// This checkpoint emits data for a future native backend lowerer. It never calls
// a graphics API, creates a native shader, authorizes a draw, or renders on CPU.
inline constexpr bool kPerformsNativeShaderCompilation = false;
inline constexpr bool kAuthorizesNativeGpuExecution = false;
inline constexpr bool kAllowsCpuProductionRenderingFallback = false;

enum class BackendTarget : std::uint8_t
{
    Invalid = 0,
    OpenGl = 1,
    Metal = 2
};

enum class ShaderLanguageProfile : std::uint8_t
{
    Invalid = 0,
    Glsl330Core = 1,
    MetalSl20 = 2
};

struct BackendRequirements final
{
    BackendTarget target = BackendTarget::Invalid;
    std::uint16_t minimumApiMajor = 0;
    std::uint16_t minimumApiMinor = 0;
    ShaderLanguageProfile shaderLanguage = ShaderLanguageProfile::Invalid;
    std::uint16_t shaderLanguageVersion = 0;
};

struct CompilerCapabilities final
{
    BackendRequirements backend;
    std::uint32_t compilerVersion = kCompilerVersion;
    std::uint32_t programRecordVersion = kProgramRecordVersion;
    std::uint32_t surfaceIrVersion = surfacematerial::kWireVersion;
    std::uint32_t vertexIrVersion = videowire::kVertexModifierIrSchemaVersion;
    std::uint64_t supportedSurfaceOperations = 0;
    std::uint64_t supportedVertexOperations = 0;
    std::uint32_t maximumSurfaceInstructions
        = static_cast<std::uint32_t>(kMaximumSurfaceInstructions);
    std::uint32_t maximumVertexInstructions
        = static_cast<std::uint32_t>(kMaximumVertexInstructions);
    std::uint32_t maximumCombinedInstructions
        = static_cast<std::uint32_t>(kMaximumCombinedInstructions);
    std::uint16_t maximumTextureSlots
        = static_cast<std::uint16_t>(surfacematerial::kMaximumTextureSlots);
    std::uint16_t maximumControlParameterSlots
        = static_cast<std::uint16_t>(kMaximumControlParameterSlots);
    std::uint16_t maximumAudioParameterSlots
        = static_cast<std::uint16_t>(kMaximumAudioParameterSlots);
    std::uint32_t maximumCanonicalPayloadBytes
        = static_cast<std::uint32_t>(kMaximumCanonicalPayloadBytes);
};

struct CompileLimits final
{
    std::size_t surfaceInstructions = kMaximumSurfaceInstructions;
    std::size_t vertexInstructions = kMaximumVertexInstructions;
    std::size_t combinedInstructions = kMaximumCombinedInstructions;
    std::size_t textureSlots = surfacematerial::kMaximumTextureSlots;
    std::size_t controlParameterSlots = kMaximumControlParameterSlots;
    std::size_t audioParameterSlots = kMaximumAudioParameterSlots;
    std::size_t canonicalPayloadBytes = kMaximumCanonicalPayloadBytes;
};

struct ProgramResourceUsage final
{
    std::uint32_t surfaceInstructions = 0;
    std::uint32_t vertexInstructions = 0;
    std::uint32_t valueSlots = 0;
    std::uint32_t inputReferences = 0;
    std::uint32_t constantBytes = 0;
    std::uint16_t textureSlots = 0;
    std::uint16_t controlParameterSlots = 0;
    std::uint16_t audioParameterSlots = 0;
    std::uint16_t surfaceOutputBindings = 0;
    std::uint16_t surfaceMaximumDepth = 0;
    std::uint16_t vertexMaximumDepth = 0;
    std::uint32_t surfaceInputSemanticMask = 0;
    std::uint64_t usedSurfaceOperationMask = 0;
    std::uint64_t usedVertexOperationMask = 0;
    double maximumVertexDisplacement = 0.0;
    std::uint32_t surfaceParameterBlockBytes = 0;
    std::uint64_t canonicalPayloadBytes = 0;
};

class MaterialProgramRecord final
{
public:
    MaterialProgramRecord(const MaterialProgramRecord&) = default;
    MaterialProgramRecord(MaterialProgramRecord&&) noexcept = default;
    MaterialProgramRecord& operator=(const MaterialProgramRecord&) = delete;
    MaterialProgramRecord& operator=(MaterialProgramRecord&&) = delete;

    const CompilerCapabilities& capabilities() const noexcept { return capabilities_; }
    const ProgramResourceUsage& resources() const noexcept { return resources_; }
    const std::string& sourceIdentity() const noexcept { return sourceIdentity_; }
    const std::string& programIdentity() const noexcept { return programIdentity_; }
    const videowire::MaterialProgramIR& program() const noexcept { return program_; }
    const std::vector<SurfaceInstruction>& surfaceInstructions() const noexcept
    {
        return program_.surfaceInstructions;
    }
    const std::array<std::uint16_t, surfacematerial::kSurfaceOutputCount>&
    surfaceOutputSlots() const noexcept
    {
        return program_.surfaceOutputSlots;
    }
    const std::vector<VertexInstruction>& vertexInstructions() const noexcept
    {
        return program_.vertexInstructions;
    }
    std::uint16_t vertexOutputSlot() const noexcept { return program_.vertexOutputSlot; }
    bool hasSurfaceProgram() const noexcept { return !program_.surfaceInstructions.empty(); }
    bool hasVertexProgram() const noexcept { return !program_.vertexInstructions.empty(); }

    static constexpr bool containsShaderSource = false;
    static constexpr bool performsNativeShaderCompilation = false;
    static constexpr bool authorizesNativeGpuExecution = false;
    static constexpr bool allowsCpuProductionRenderingFallback = false;

private:
    friend std::optional<MaterialProgramRecord> compileMaterialProgram(
        const surfacematerial::AdmittedSurfaceMaterialIR*,
        const vertexmodifier::AdmittedVertexModifierIr*, BackendTarget,
        std::string&, const CompileLimits&);

    MaterialProgramRecord(
        CompilerCapabilities capabilities,
        ProgramResourceUsage resources,
        std::string sourceIdentity,
        std::string programIdentity,
        std::vector<SurfaceInstruction> surfaceInstructions,
        std::array<std::uint16_t, surfacematerial::kSurfaceOutputCount> surfaceOutputSlots,
        std::vector<VertexInstruction> vertexInstructions,
        std::uint16_t vertexOutputSlot)
        : capabilities_(std::move(capabilities)), resources_(std::move(resources)),
          sourceIdentity_(std::move(sourceIdentity)),
          programIdentity_(std::move(programIdentity)),
          program_ { kProgramRecordVersion, std::move(surfaceInstructions),
                     surfaceOutputSlots, std::move(vertexInstructions), vertexOutputSlot }
    {
    }

    CompilerCapabilities capabilities_;
    ProgramResourceUsage resources_;
    std::string sourceIdentity_;
    std::string programIdentity_;
    videowire::MaterialProgramIR program_;
};

namespace detail
{
constexpr std::uint64_t operationBit(std::uint8_t value) noexcept
{
    return value > 0 && value < 64 ? (std::uint64_t { 1 } << value) : 0;
}

constexpr std::uint64_t surfaceOperationMask() noexcept
{
    std::uint64_t result = 0;
    for (std::uint8_t value = 1;
         value <= static_cast<std::uint8_t>(surfacematerial::OperationKind::ComposeVec4);
         ++value)
        result |= operationBit(value);
    return result;
}

constexpr std::uint64_t vertexOperationMask() noexcept
{
    std::uint64_t result = 0;
    for (std::uint8_t value = 1;
         value <= static_cast<std::uint8_t>(
             videowire::VertexModifierOperation::boundedDisplacementOutput);
         ++value)
        result |= operationBit(value);
    return result;
}

inline bool supported(surfacematerial::OperationKind operation) noexcept
{
    switch (operation)
    {
        case surfacematerial::OperationKind::FloatConstant:
        case surfacematerial::OperationKind::UIntConstant:
        case surfacematerial::OperationKind::Input:
        case surfacematerial::OperationKind::TextureSample2D:
        case surfacematerial::OperationKind::Add:
        case surfacematerial::OperationKind::Subtract:
        case surfacematerial::OperationKind::Multiply:
        case surfacematerial::OperationKind::Divide:
        case surfacematerial::OperationKind::Minimum:
        case surfacematerial::OperationKind::Maximum:
        case surfacematerial::OperationKind::Clamp:
        case surfacematerial::OperationKind::Mix:
        case surfacematerial::OperationKind::Dot:
        case surfacematerial::OperationKind::Normalize:
        case surfacematerial::OperationKind::Length:
        case surfacematerial::OperationKind::Absolute:
        case surfacematerial::OperationKind::Power:
        case surfacematerial::OperationKind::Component:
        case surfacematerial::OperationKind::ComposeVec2:
        case surfacematerial::OperationKind::ComposeVec3:
        case surfacematerial::OperationKind::ComposeVec4:
            return true;
        case surfacematerial::OperationKind::Invalid:
            break;
    }
    return false;
}

inline bool supported(videowire::VertexModifierOperation operation) noexcept
{
    switch (operation)
    {
        case videowire::VertexModifierOperation::importedPosition:
        case videowire::VertexModifierOperation::importedNormal:
        case videowire::VertexModifierOperation::importedUv:
        case videowire::VertexModifierOperation::importedVertexColor:
        case videowire::VertexModifierOperation::scalarConstant:
        case videowire::VertexModifierOperation::vec3Constant:
        case videowire::VertexModifierOperation::timeSeconds:
        case videowire::VertexModifierOperation::audioParameter:
        case videowire::VertexModifierOperation::controlParameter:
        case videowire::VertexModifierOperation::scalarAdd:
        case videowire::VertexModifierOperation::scalarSubtract:
        case videowire::VertexModifierOperation::scalarMultiply:
        case videowire::VertexModifierOperation::vec3Add:
        case videowire::VertexModifierOperation::vec3Subtract:
        case videowire::VertexModifierOperation::vec3Multiply:
        case videowire::VertexModifierOperation::vec3Scale:
        case videowire::VertexModifierOperation::vec3Compose:
        case videowire::VertexModifierOperation::componentX:
        case videowire::VertexModifierOperation::componentY:
        case videowire::VertexModifierOperation::componentZ:
        case videowire::VertexModifierOperation::scalarRemap:
        case videowire::VertexModifierOperation::vec3Remap:
        case videowire::VertexModifierOperation::scalarNoise3d:
        case videowire::VertexModifierOperation::vec3Noise3d:
        case videowire::VertexModifierOperation::boundedDisplacementOutput:
            return true;
        case videowire::VertexModifierOperation::invalid:
            break;
    }
    return false;
}

inline CompilerCapabilities capabilitiesFor(BackendTarget target)
{
    CompilerCapabilities capabilities;
    capabilities.backend.target = target;
    capabilities.supportedSurfaceOperations = surfaceOperationMask();
    capabilities.supportedVertexOperations = vertexOperationMask();
    switch (target)
    {
        case BackendTarget::OpenGl:
            capabilities.backend.minimumApiMajor = 3;
            capabilities.backend.minimumApiMinor = 3;
            capabilities.backend.shaderLanguage = ShaderLanguageProfile::Glsl330Core;
            capabilities.backend.shaderLanguageVersion = 330;
            break;
        case BackendTarget::Metal:
            capabilities.backend.minimumApiMajor = 2;
            capabilities.backend.minimumApiMinor = 0;
            capabilities.backend.shaderLanguage = ShaderLanguageProfile::MetalSl20;
            capabilities.backend.shaderLanguageVersion = 200;
            break;
        case BackendTarget::Invalid:
            capabilities = {};
            break;
    }
    return capabilities;
}

inline bool validLimits(const CompileLimits& limits) noexcept
{
    return limits.surfaceInstructions > 0
        && limits.surfaceInstructions <= kMaximumSurfaceInstructions
        && limits.vertexInstructions > 0
        && limits.vertexInstructions <= kMaximumVertexInstructions
        && limits.combinedInstructions > 0
        && limits.combinedInstructions <= kMaximumCombinedInstructions
        && limits.textureSlots <= surfacematerial::kMaximumTextureSlots
        && limits.controlParameterSlots <= kMaximumControlParameterSlots
        && limits.audioParameterSlots <= kMaximumAudioParameterSlots
        && limits.canonicalPayloadBytes > 0
        && limits.canonicalPayloadBytes <= kMaximumCanonicalPayloadBytes;
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
    for (int shift = 56; shift >= 0; shift -= 8)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline void appendFloat(std::vector<std::uint8_t>& bytes, float value)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t)
                      && std::numeric_limits<float>::is_iec559,
                  "material program records require IEEE-754 binary32 floats");
    if (value == 0.0f) value = 0.0f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(bytes, bits);
}

inline void appendDouble(std::vector<std::uint8_t>& bytes, double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t)
                      && std::numeric_limits<double>::is_iec559,
                  "material program records require IEEE-754 binary64 doubles");
    if (value == 0.0) value = 0.0;
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU64(bytes, bits);
}

inline void appendString(std::vector<std::uint8_t>& bytes, const std::string& value)
{
    appendU32(bytes, static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

inline std::string digest(const std::vector<std::uint8_t>& bytes)
{
    videohelper::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finishHex();
}

inline std::string sourceIdentity(
    const surfacematerial::AdmittedSurfaceMaterialIR* surface,
    const vertexmodifier::AdmittedVertexModifierIr* vertex)
{
    static constexpr char domain[] = "DonutStudio/MaterialProgramSource/v1";
    std::vector<std::uint8_t> bytes(std::begin(domain), std::end(domain) - 1);
    appendU8(bytes, surface != nullptr ? 1 : 0);
    if (surface != nullptr) appendString(bytes, surface->structuralDigest());
    appendU8(bytes, vertex != nullptr ? 1 : 0);
    if (vertex != nullptr) appendString(bytes, vertex->structuralDigest());
    return digest(bytes);
}

inline void appendCapabilities(std::vector<std::uint8_t>& bytes,
                               const CompilerCapabilities& capabilities)
{
    appendU8(bytes, static_cast<std::uint8_t>(capabilities.backend.target));
    appendU16(bytes, capabilities.backend.minimumApiMajor);
    appendU16(bytes, capabilities.backend.minimumApiMinor);
    appendU8(bytes, static_cast<std::uint8_t>(capabilities.backend.shaderLanguage));
    appendU16(bytes, capabilities.backend.shaderLanguageVersion);
    appendU32(bytes, capabilities.compilerVersion);
    appendU32(bytes, capabilities.programRecordVersion);
    appendU32(bytes, capabilities.surfaceIrVersion);
    appendU32(bytes, capabilities.vertexIrVersion);
    appendU64(bytes, capabilities.supportedSurfaceOperations);
    appendU64(bytes, capabilities.supportedVertexOperations);
    appendU32(bytes, capabilities.maximumSurfaceInstructions);
    appendU32(bytes, capabilities.maximumVertexInstructions);
    appendU32(bytes, capabilities.maximumCombinedInstructions);
    appendU16(bytes, capabilities.maximumTextureSlots);
    appendU16(bytes, capabilities.maximumControlParameterSlots);
    appendU16(bytes, capabilities.maximumAudioParameterSlots);
    appendU32(bytes, capabilities.maximumCanonicalPayloadBytes);
}

inline void appendResources(std::vector<std::uint8_t>& bytes,
                            const ProgramResourceUsage& resources)
{
    appendU32(bytes, resources.surfaceInstructions);
    appendU32(bytes, resources.vertexInstructions);
    appendU32(bytes, resources.valueSlots);
    appendU32(bytes, resources.inputReferences);
    appendU32(bytes, resources.constantBytes);
    appendU16(bytes, resources.textureSlots);
    appendU16(bytes, resources.controlParameterSlots);
    appendU16(bytes, resources.audioParameterSlots);
    appendU16(bytes, resources.surfaceOutputBindings);
    appendU16(bytes, resources.surfaceMaximumDepth);
    appendU16(bytes, resources.vertexMaximumDepth);
    appendU32(bytes, resources.surfaceInputSemanticMask);
    appendU64(bytes, resources.usedSurfaceOperationMask);
    appendU64(bytes, resources.usedVertexOperationMask);
    appendDouble(bytes, resources.maximumVertexDisplacement);
    appendU32(bytes, resources.surfaceParameterBlockBytes);
}

inline std::vector<std::uint8_t> canonicalPayload(
    const CompilerCapabilities& capabilities,
    const ProgramResourceUsage& resources,
    const std::string& sourceIdentityValue,
    const std::vector<SurfaceInstruction>& surfaceInstructions,
    const std::array<std::uint16_t, surfacematerial::kSurfaceOutputCount>& surfaceOutputs,
    const std::vector<VertexInstruction>& vertexInstructions,
    std::uint16_t vertexOutput)
{
    static constexpr char domain[] = "DonutStudio/MaterialProgramRecord/v1";
    std::vector<std::uint8_t> bytes(std::begin(domain), std::end(domain) - 1);
    appendCapabilities(bytes, capabilities);
    appendResources(bytes, resources);
    appendString(bytes, sourceIdentityValue);

    appendU32(bytes, static_cast<std::uint32_t>(surfaceInstructions.size()));
    for (const auto& instruction : surfaceInstructions)
    {
        appendU8(bytes, static_cast<std::uint8_t>(
                            videowire::MaterialProgramStage::surfaceFragment));
        appendU16(bytes, instruction.resultSlot);
        appendU8(bytes, static_cast<std::uint8_t>(instruction.operation));
        appendU8(bytes, static_cast<std::uint8_t>(instruction.resultType));
        appendU8(bytes, instruction.inputCount);
        for (const auto input : instruction.inputs) appendU16(bytes, input);
        for (const auto literal : instruction.literal) appendFloat(bytes, literal);
        appendU8(bytes, static_cast<std::uint8_t>(instruction.semantic));
        appendU16(bytes, instruction.parameter);
        appendU32(bytes, instruction.unsignedLiteral);
    }
    for (const auto output : surfaceOutputs) appendU16(bytes, output);

    appendU32(bytes, static_cast<std::uint32_t>(vertexInstructions.size()));
    for (const auto& instruction : vertexInstructions)
    {
        appendU8(bytes, static_cast<std::uint8_t>(
                            videowire::MaterialProgramStage::vertexModifier));
        appendU16(bytes, instruction.resultSlot);
        appendU64(bytes, instruction.sourceStableId);
        appendU8(bytes, static_cast<std::uint8_t>(instruction.operation));
        appendU8(bytes, static_cast<std::uint8_t>(instruction.resultType));
        appendU8(bytes, instruction.inputCount);
        for (const auto input : instruction.inputs) appendU16(bytes, input);
        appendU8(bytes, instruction.parameterCount);
        for (const auto parameter : instruction.parameters) appendDouble(bytes, parameter);
    }
    appendU16(bytes, vertexOutput);
    return bytes;
}

inline bool compileSurface(
    const surfacematerial::AdmittedSurfaceMaterialIR& admitted,
    std::vector<SurfaceInstruction>& instructions,
    std::array<std::uint16_t, surfacematerial::kSurfaceOutputCount>& outputs,
    ProgramResourceUsage& resources,
    std::string& error)
{
    const auto& program = admitted.program();
    instructions.reserve(program.operations.size());
    for (std::size_t index = 0; index < program.operations.size(); ++index)
    {
        const auto& source = program.operations[index];
        if (!supported(source.kind))
        {
            error = "surface material compiler does not support operation "
                + std::to_string(static_cast<unsigned>(source.kind));
            return false;
        }

        SurfaceInstruction instruction;
        instruction.resultSlot = static_cast<std::uint16_t>(index);
        instruction.operation = source.kind;
        instruction.resultType = source.resultType;
        instruction.inputCount = source.inputCount;
        for (std::size_t input = 0; input < source.inputCount; ++input)
            instruction.inputs[input] = static_cast<std::uint16_t>(source.inputs[input] - 1);
        instruction.literal = source.literal;
        for (auto& literal : instruction.literal)
            if (literal == 0.0f) literal = 0.0f;
        instruction.semantic = source.semantic;
        instruction.parameter = source.parameter;
        instruction.unsignedLiteral = source.unsignedLiteral;
        instructions.push_back(instruction);

        const auto operation = static_cast<std::uint8_t>(source.kind);
        resources.usedSurfaceOperationMask |= operationBit(operation);
        resources.inputReferences += source.inputCount;
        if (source.kind == surfacematerial::OperationKind::FloatConstant)
        {
            const auto width = static_cast<std::uint32_t>(source.resultType);
            resources.constantBytes += width * static_cast<std::uint32_t>(sizeof(float));
        }
        else if (source.kind == surfacematerial::OperationKind::UIntConstant)
        {
            resources.constantBytes += static_cast<std::uint32_t>(sizeof(std::uint32_t));
        }
        else if (source.kind == surfacematerial::OperationKind::Input)
        {
            const auto semantic = static_cast<std::uint8_t>(source.semantic);
            resources.surfaceInputSemanticMask |= std::uint32_t { 1 } << semantic;
            if (source.semantic == surfacematerial::InputSemantic::Control)
            {
                resources.controlParameterSlots = std::max(
                    resources.controlParameterSlots,
                    static_cast<std::uint16_t>(source.parameter + 1));
            }
        }
    }

    for (std::size_t index = 0; index < outputs.size(); ++index)
        outputs[index] = static_cast<std::uint16_t>(program.outputs[index] - 1);
    resources.surfaceInstructions = static_cast<std::uint32_t>(instructions.size());
    resources.textureSlots = program.textureSlotCount;
    resources.surfaceOutputBindings
        = static_cast<std::uint16_t>(surfacematerial::kSurfaceOutputCount);
    resources.surfaceMaximumDepth = static_cast<std::uint16_t>(admitted.maximumDepth());
    resources.surfaceParameterBlockBytes
        = static_cast<std::uint32_t>(sizeof(surfacematerial::SurfacePbrParameterBlock));
    return true;
}

inline bool compileVertex(
    const vertexmodifier::AdmittedVertexModifierIr& admitted,
    std::vector<VertexInstruction>& instructions,
    std::uint16_t& output,
    ProgramResourceUsage& resources,
    std::string& error)
{
    const auto& records = admitted.records();
    std::unordered_map<videowire::VertexModifierStableId, std::size_t> sourceIndices;
    sourceIndices.reserve(records.size());
    for (std::size_t index = 0; index < records.size(); ++index)
        sourceIndices.emplace(records[index].stableId, index);

    std::vector<std::size_t> dependencies(records.size(), 0);
    std::vector<std::vector<std::size_t>> dependants(records.size());
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        dependencies[index] = records[index].inputCount;
        for (std::size_t input = 0; input < records[index].inputCount; ++input)
            dependants[sourceIndices.at(records[index].inputs[input])].push_back(index);
    }

    using Ready = std::pair<videowire::VertexModifierStableId, std::size_t>;
    std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
    for (std::size_t index = 0; index < records.size(); ++index)
        if (dependencies[index] == 0) ready.emplace(records[index].stableId, index);

    std::vector<std::uint16_t> slots(records.size(), kInvalidValueSlot);
    instructions.reserve(records.size());
    while (!ready.empty())
    {
        const auto sourceIndex = ready.top().second;
        ready.pop();
        const auto& source = records[sourceIndex];
        if (!supported(source.operation))
        {
            error = "vertex modifier compiler does not support operation "
                + std::to_string(static_cast<unsigned>(source.operation));
            return false;
        }

        VertexInstruction instruction;
        instruction.resultSlot = static_cast<std::uint16_t>(instructions.size());
        instruction.sourceStableId = source.stableId;
        instruction.operation = source.operation;
        instruction.resultType = source.resultType;
        instruction.inputCount = source.inputCount;
        for (std::size_t input = 0; input < source.inputCount; ++input)
            instruction.inputs[input] = slots[sourceIndices.at(source.inputs[input])];
        instruction.parameterCount = source.parameterCount;
        instruction.parameters = source.parameters;
        for (auto& parameter : instruction.parameters)
            if (parameter == 0.0) parameter = 0.0;
        slots[sourceIndex] = instruction.resultSlot;
        instructions.push_back(instruction);

        resources.usedVertexOperationMask
            |= operationBit(static_cast<std::uint8_t>(source.operation));
        resources.inputReferences += source.inputCount;
        resources.constantBytes += static_cast<std::uint32_t>(source.parameterCount)
            * static_cast<std::uint32_t>(sizeof(double));
        if (source.operation == videowire::VertexModifierOperation::controlParameter)
        {
            resources.controlParameterSlots = std::max(
                resources.controlParameterSlots,
                static_cast<std::uint16_t>(source.parameters[0] + 1.0));
        }
        else if (source.operation == videowire::VertexModifierOperation::audioParameter)
        {
            resources.audioParameterSlots = std::max(
                resources.audioParameterSlots,
                static_cast<std::uint16_t>(source.parameters[0] + 1.0));
        }

        for (const auto dependant : dependants[sourceIndex])
        {
            if (--dependencies[dependant] == 0)
                ready.emplace(records[dependant].stableId, dependant);
        }
    }

    if (instructions.size() != records.size())
    {
        error = "vertex modifier compiler found an unsupported dependency cycle";
        return false;
    }
    output = slots[sourceIndices.at(admitted.rootId())];
    resources.vertexInstructions = static_cast<std::uint32_t>(instructions.size());
    resources.vertexMaximumDepth = static_cast<std::uint16_t>(admitted.maximumDepth());
    resources.maximumVertexDisplacement = admitted.maximumDisplacement();
    return true;
}
} // namespace detail

inline CompilerCapabilities compilerCapabilities(BackendTarget target)
{
    return detail::capabilitiesFor(target);
}

inline std::optional<MaterialProgramRecord> compileMaterialProgram(
    const surfacematerial::AdmittedSurfaceMaterialIR* surface,
    const vertexmodifier::AdmittedVertexModifierIr* vertex,
    BackendTarget target,
    std::string& error,
    const CompileLimits& limits = {})
{
    error.clear();
    if (surface == nullptr && vertex == nullptr)
    {
        error = "material program compilation requires a surface or vertex program";
        return std::nullopt;
    }
    if (!detail::validLimits(limits))
    {
        error = "material program compiler limits exceed hard contract bounds";
        return std::nullopt;
    }

    auto capabilities = detail::capabilitiesFor(target);
    if (capabilities.backend.target == BackendTarget::Invalid)
    {
        error = "material program compiler backend target is unsupported";
        return std::nullopt;
    }
    if ((surface != nullptr && surface->program().operations.size() > limits.surfaceInstructions)
        || (vertex != nullptr && vertex->operationCount() > limits.vertexInstructions))
    {
        error = "material program stage instruction limit exceeded";
        return std::nullopt;
    }
    const auto totalInstructions
        = (surface != nullptr ? surface->program().operations.size() : 0)
        + (vertex != nullptr ? vertex->operationCount() : 0);
    if (totalInstructions > limits.combinedInstructions)
    {
        error = "material program combined instruction limit exceeded";
        return std::nullopt;
    }

    ProgramResourceUsage resources;
    std::vector<SurfaceInstruction> surfaceInstructions;
    std::array<std::uint16_t, surfacematerial::kSurfaceOutputCount> surfaceOutputs;
    surfaceOutputs.fill(kInvalidValueSlot);
    std::vector<VertexInstruction> vertexInstructions;
    std::uint16_t vertexOutput = kInvalidValueSlot;

    if (surface != nullptr
        && !detail::compileSurface(*surface, surfaceInstructions, surfaceOutputs,
                                   resources, error))
        return std::nullopt;
    if (vertex != nullptr
        && !detail::compileVertex(*vertex, vertexInstructions, vertexOutput,
                                  resources, error))
        return std::nullopt;

    resources.valueSlots = resources.surfaceInstructions + resources.vertexInstructions;
    if (resources.textureSlots > limits.textureSlots
        || resources.controlParameterSlots > limits.controlParameterSlots
        || resources.audioParameterSlots > limits.audioParameterSlots)
    {
        error = "material program binding resource limit exceeded";
        return std::nullopt;
    }
    if ((resources.usedSurfaceOperationMask & ~capabilities.supportedSurfaceOperations) != 0
        || (resources.usedVertexOperationMask & ~capabilities.supportedVertexOperations) != 0)
    {
        error = "material program requires an unsupported backend operation";
        return std::nullopt;
    }

    auto sourceIdentityValue = detail::sourceIdentity(surface, vertex);
    auto payload = detail::canonicalPayload(
        capabilities, resources, sourceIdentityValue, surfaceInstructions,
        surfaceOutputs, vertexInstructions, vertexOutput);
    resources.canonicalPayloadBytes = payload.size();
    if (resources.canonicalPayloadBytes > limits.canonicalPayloadBytes)
    {
        error = "material program canonical payload byte limit exceeded";
        return std::nullopt;
    }
    const auto programIdentityValue = detail::digest(payload);

    return MaterialProgramRecord(
        std::move(capabilities), resources, std::move(sourceIdentityValue),
        programIdentityValue, std::move(surfaceInstructions), surfaceOutputs,
        std::move(vertexInstructions), vertexOutput);
}

class MaterialProgramCompilerCheckpoint final
{
public:
    bool compileAndPublish(
        std::uint64_t authoringRevision,
        const surfacematerial::AdmittedSurfaceMaterialIR* surface,
        const vertexmodifier::AdmittedVertexModifierIr* vertex,
        BackendTarget target,
        std::string& error,
        const CompileLimits& limits = {})
    {
        if (authoringRevision == 0 || authoringRevision <= latestAttemptedRevision_)
        {
            error = "material program authoring revision is zero or stale";
            return false;
        }
        latestAttemptedRevision_ = authoringRevision;
        auto candidate = compileMaterialProgram(surface, vertex, target, error, limits);
        if (!candidate)
        {
            rejectedRevision_ = authoringRevision;
            return false;
        }

        lastGood_ = std::make_shared<const MaterialProgramRecord>(std::move(*candidate));
        lastGoodRevision_ = authoringRevision;
        return true;
    }

    std::uint64_t latestAttemptedRevision() const noexcept
    {
        return latestAttemptedRevision_;
    }
    std::uint64_t lastGoodRevision() const noexcept { return lastGoodRevision_; }
    std::uint64_t rejectedRevision() const noexcept { return rejectedRevision_; }
    const std::shared_ptr<const MaterialProgramRecord>& lastGood() const noexcept
    {
        return lastGood_;
    }

private:
    std::uint64_t latestAttemptedRevision_ = 0;
    std::uint64_t lastGoodRevision_ = 0;
    std::uint64_t rejectedRevision_ = 0;
    std::shared_ptr<const MaterialProgramRecord> lastGood_;
};

static_assert(static_cast<std::uint8_t>(surfacematerial::OperationKind::ComposeVec4) < 64,
              "surface operation capabilities use a 64-bit mask");
static_assert(static_cast<std::uint8_t>(
                  videowire::VertexModifierOperation::boundedDisplacementOutput) < 64,
              "vertex operation capabilities use a 64-bit mask");
static_assert(!kPerformsNativeShaderCompilation);
static_assert(!kAuthorizesNativeGpuExecution);
static_assert(!kAllowsCpuProductionRenderingFallback);
static_assert(!MaterialProgramRecord::containsShaderSource);
static_assert(!MaterialProgramRecord::performsNativeShaderCompilation);
static_assert(!MaterialProgramRecord::authorizesNativeGpuExecution);
static_assert(!MaterialProgramRecord::allowsCpuProductionRenderingFallback);
} // namespace videohelper::materialprogram
