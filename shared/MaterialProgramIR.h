#pragma once

#include "SurfaceMaterialIR.h"
#include "VertexModifierIr.h"

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace videowire
{
inline constexpr std::uint32_t kMaterialProgramIrSchemaVersion = 1;
inline constexpr std::uint32_t kMaterialProgramInvalidValueSlot
    = std::numeric_limits<std::uint32_t>::max();

enum class MaterialProgramStage : std::uint8_t
{
    surfaceFragment = 1,
    vertexModifier = 2
};

struct MaterialProgramSurfaceInstruction final
{
    std::uint32_t resultSlot = kMaterialProgramInvalidValueSlot;
    surfacematerial::OperationKind operation = surfacematerial::OperationKind::Invalid;
    surfacematerial::ValueType resultType = surfacematerial::ValueType::Invalid;
    std::uint8_t inputCount = 0;
    std::array<std::uint32_t, surfacematerial::kMaximumOperationInputs> inputs {
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot,
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot
    };
    std::array<float, 4> literal {};
    surfacematerial::InputSemantic semantic = surfacematerial::InputSemantic::Invalid;
    std::uint16_t parameter = 0;
    std::uint32_t unsignedLiteral = 0;
};

struct MaterialProgramVertexInstruction final
{
    std::uint32_t resultSlot = kMaterialProgramInvalidValueSlot;
    VertexModifierStableId sourceStableId = 0;
    VertexModifierOperation operation = VertexModifierOperation::invalid;
    VertexModifierValueType resultType = VertexModifierValueType::invalid;
    std::uint8_t inputCount = 0;
    std::array<std::uint32_t, kVertexModifierMaximumInputsPerRecord> inputs {
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot,
        kMaterialProgramInvalidValueSlot
    };
    std::uint8_t parameterCount = 0;
    std::array<double, kVertexModifierMaximumParametersPerRecord> parameters {};
};

struct MaterialProgramIR final
{
    std::uint32_t schemaVersion = kMaterialProgramIrSchemaVersion;
    std::vector<MaterialProgramSurfaceInstruction> surfaceInstructions;
    std::array<std::uint32_t, surfacematerial::kProgramOutputCount> surfaceOutputSlots {
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot,
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot,
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot,
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot,
        kMaterialProgramInvalidValueSlot, kMaterialProgramInvalidValueSlot,
        kMaterialProgramInvalidValueSlot
    };
    std::vector<MaterialProgramVertexInstruction> vertexInstructions;
    std::uint32_t vertexOutputSlot = kMaterialProgramInvalidValueSlot;
    surfacematerial::TransportMode transport = surfacematerial::TransportMode::ThinSurface;

    static constexpr bool containsShaderSource = false;
    static constexpr bool performsNativeShaderCompilation = false;
    static constexpr bool authorizesNativeGpuExecution = false;
};

static_assert(surfacematerial::kSurfaceOutputCount == 10,
              "material program output initialization tracks the surface contract");
static_assert(std::is_standard_layout_v<MaterialProgramSurfaceInstruction>
                  && std::is_trivially_copyable_v<MaterialProgramSurfaceInstruction>,
              "surface program instructions contain data only");
static_assert(std::is_standard_layout_v<MaterialProgramVertexInstruction>
                  && std::is_trivially_copyable_v<MaterialProgramVertexInstruction>,
              "vertex program instructions contain data only");
static_assert(!MaterialProgramIR::containsShaderSource);
static_assert(!MaterialProgramIR::performsNativeShaderCompilation);
static_assert(!MaterialProgramIR::authorizesNativeGpuExecution);
} // namespace videowire
