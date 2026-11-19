#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>

namespace videowire
{

// This wire model contains data only. It carries no shader source, backend program,
// or CPU renderer. A separately admitted GPU compiler must lower accepted records.
using VertexModifierStableId = std::uint64_t;

inline constexpr std::uint32_t kVertexModifierIrSchemaVersion = 1;
inline constexpr std::size_t kVertexModifierMaximumInputsPerRecord = 3;
inline constexpr std::size_t kVertexModifierMaximumParametersPerRecord = 6;

enum class VertexModifierValueType : std::uint8_t
{
    invalid = 0,
    scalar = 1,
    vec3 = 2
};

inline constexpr const char* vertexModifierValueTypeWireToken(
    VertexModifierValueType type) noexcept
{
    switch (type)
    {
        case VertexModifierValueType::scalar: return "scalar";
        case VertexModifierValueType::vec3: return "vec3";
        case VertexModifierValueType::invalid: break;
    }
    return "invalid";
}

inline VertexModifierValueType vertexModifierValueTypeFromWireToken(
    std::string_view token) noexcept
{
    if (token == "scalar") return VertexModifierValueType::scalar;
    if (token == "vec3") return VertexModifierValueType::vec3;
    return VertexModifierValueType::invalid;
}

enum class VertexModifierOperation : std::uint8_t
{
    invalid = 0,
    importedPosition = 1,
    importedNormal = 2,
    // UV is carried as (u, v, 0), and vertex color as linear RGB. The restricted
    // value domain stays scalar/Vec3 rather than introducing backend vector ABI types.
    importedUv = 3,
    importedVertexColor = 4,
    scalarConstant = 5,
    vec3Constant = 6,
    timeSeconds = 7,
    // The single parameter is a stable integral index into the admitted runtime bank.
    audioParameter = 8,
    controlParameter = 9,
    scalarAdd = 10,
    scalarSubtract = 11,
    scalarMultiply = 12,
    vec3Add = 13,
    vec3Subtract = 14,
    vec3Multiply = 15,
    vec3Scale = 16,
    vec3Compose = 17,
    componentX = 18,
    componentY = 19,
    componentZ = 20,
    // Remap parameters are input min/max, output min/max, and a 0/1 clamp flag.
    scalarRemap = 21,
    vec3Remap = 22,
    // Noise parameters are positive frequency and an integral deterministic seed.
    scalarNoise3d = 23,
    vec3Noise3d = 24,
    // Inputs are the imported base position and a displacement Vec3. The parameter
    // is the maximum displacement length. Lowering must clamp before adding it.
    boundedDisplacementOutput = 25,
    // Sine and cosine of a scalar in radians.
    scalarSin = 26,
    scalarCos = 27,
    // Inputs are a rest position (its X and Y) and seconds. The result is the
    // Ocean displacement: the wind spectrum's waves summed in a loop over a wave
    // table that the backend compiler builds from the parameters, which are wind
    // speed (m/s), wind heading (degrees), smallest wave (m), size (m), an
    // integral seed and choppiness. Other Ocean settings keep their defaults.
    oceanWaves = 28
};

inline constexpr VertexModifierOperation kLastVertexModifierOperation = VertexModifierOperation::oceanWaves;

inline constexpr const char* vertexModifierOperationWireToken(
    VertexModifierOperation operation) noexcept
{
    switch (operation)
    {
        case VertexModifierOperation::importedPosition: return "imported-position";
        case VertexModifierOperation::importedNormal: return "imported-normal";
        case VertexModifierOperation::importedUv: return "imported-uv";
        case VertexModifierOperation::importedVertexColor: return "imported-vertex-color";
        case VertexModifierOperation::scalarConstant: return "scalar-constant";
        case VertexModifierOperation::vec3Constant: return "vec3-constant";
        case VertexModifierOperation::timeSeconds: return "time-seconds";
        case VertexModifierOperation::audioParameter: return "audio-parameter";
        case VertexModifierOperation::controlParameter: return "control-parameter";
        case VertexModifierOperation::scalarAdd: return "scalar-add";
        case VertexModifierOperation::scalarSubtract: return "scalar-subtract";
        case VertexModifierOperation::scalarMultiply: return "scalar-multiply";
        case VertexModifierOperation::vec3Add: return "vec3-add";
        case VertexModifierOperation::vec3Subtract: return "vec3-subtract";
        case VertexModifierOperation::vec3Multiply: return "vec3-multiply";
        case VertexModifierOperation::vec3Scale: return "vec3-scale";
        case VertexModifierOperation::vec3Compose: return "vec3-compose";
        case VertexModifierOperation::componentX: return "component-x";
        case VertexModifierOperation::componentY: return "component-y";
        case VertexModifierOperation::componentZ: return "component-z";
        case VertexModifierOperation::scalarRemap: return "scalar-remap";
        case VertexModifierOperation::vec3Remap: return "vec3-remap";
        case VertexModifierOperation::scalarNoise3d: return "scalar-noise-3d";
        case VertexModifierOperation::vec3Noise3d: return "vec3-noise-3d";
        case VertexModifierOperation::boundedDisplacementOutput:
            return "bounded-displacement-output";
        case VertexModifierOperation::scalarSin: return "scalar-sin";
        case VertexModifierOperation::scalarCos: return "scalar-cos";
        case VertexModifierOperation::oceanWaves: return "ocean-waves";
        case VertexModifierOperation::invalid: break;
    }
    return "invalid";
}

inline VertexModifierOperation vertexModifierOperationFromWireToken(
    std::string_view token) noexcept
{
    for (int raw = static_cast<int>(VertexModifierOperation::importedPosition);
         raw <= static_cast<int>(kLastVertexModifierOperation);
         ++raw)
    {
        const auto operation = static_cast<VertexModifierOperation>(raw);
        if (token == vertexModifierOperationWireToken(operation)) return operation;
    }
    return VertexModifierOperation::invalid;
}

inline constexpr const char* vertexModifierGraphKindToken(VertexModifierOperation operation) noexcept
{
    constexpr const char* kinds[] {
        "", "visual.vertex.imported-position", "visual.vertex.imported-normal",
        "visual.vertex.imported-uv", "visual.vertex.imported-color", "visual.vertex.scalar.constant",
        "visual.vertex.vec3.constant", "visual.vertex.time-seconds", "visual.vertex.audio-parameter",
        "visual.vertex.control-parameter", "visual.vertex.scalar.add", "visual.vertex.scalar.subtract",
        "visual.vertex.scalar.multiply", "visual.vertex.vec3.add", "visual.vertex.vec3.subtract",
        "visual.vertex.vec3.multiply", "visual.vertex.vec3.scale", "visual.vertex.vec3.compose",
        "visual.vertex.component.x", "visual.vertex.component.y", "visual.vertex.component.z",
        "visual.vertex.scalar.remap", "visual.vertex.vec3.remap", "visual.vertex.scalar.noise-3d",
        "visual.vertex.vec3.noise-3d", "visual.vertex.bounded-displacement",
        "visual.vertex.scalar.sin", "visual.vertex.scalar.cos", "visual.vertex.ocean-waves"
    };
    const auto index = static_cast<std::size_t>(operation);
    return index < std::size(kinds) ? kinds[index] : "";
}

struct VertexModifierOperationSchema
{
    VertexModifierValueType resultType = VertexModifierValueType::invalid;
    std::array<VertexModifierValueType, kVertexModifierMaximumInputsPerRecord> inputTypes {};
    std::uint8_t inputCount = 0;
    std::uint8_t parameterCount = 0;
    bool terminal = false;
    bool valid = false;
};

namespace detail
{
inline constexpr VertexModifierOperationSchema sourceSchema(
    VertexModifierValueType resultType, std::uint8_t parameters = 0) noexcept
{
    return { resultType, {}, 0, parameters, false, true };
}

inline constexpr VertexModifierOperationSchema unarySchema(
    VertexModifierValueType resultType, VertexModifierValueType inputType,
    std::uint8_t parameters = 0) noexcept
{
    return { resultType, { inputType }, 1, parameters, false, true };
}

inline constexpr VertexModifierOperationSchema binarySchema(
    VertexModifierValueType resultType, VertexModifierValueType first,
    VertexModifierValueType second) noexcept
{
    return { resultType, { first, second }, 2, 0, false, true };
}
} // namespace detail

inline constexpr VertexModifierOperationSchema vertexModifierOperationSchema(
    VertexModifierOperation operation) noexcept
{
    using Operation = VertexModifierOperation;
    using Type = VertexModifierValueType;
    switch (operation)
    {
        case Operation::importedPosition:
        case Operation::importedNormal:
        case Operation::importedUv:
        case Operation::importedVertexColor:
            return detail::sourceSchema(Type::vec3);
        case Operation::scalarConstant:
            return detail::sourceSchema(Type::scalar, 1);
        case Operation::vec3Constant:
            return detail::sourceSchema(Type::vec3, 3);
        case Operation::timeSeconds:
            return detail::sourceSchema(Type::scalar);
        case Operation::audioParameter:
            return detail::sourceSchema(Type::scalar, 1);
        // Slot index, then the authored value the control bank starts from.
        // A record written before the value existed carries only the slot.
        case Operation::controlParameter:
            return detail::sourceSchema(Type::scalar, 2);
        case Operation::scalarAdd:
        case Operation::scalarSubtract:
        case Operation::scalarMultiply:
            return detail::binarySchema(Type::scalar, Type::scalar, Type::scalar);
        case Operation::vec3Add:
        case Operation::vec3Subtract:
        case Operation::vec3Multiply:
            return detail::binarySchema(Type::vec3, Type::vec3, Type::vec3);
        case Operation::vec3Scale:
            return detail::binarySchema(Type::vec3, Type::vec3, Type::scalar);
        case Operation::vec3Compose:
            return { Type::vec3, { Type::scalar, Type::scalar, Type::scalar },
                     3, 0, false, true };
        case Operation::componentX:
        case Operation::componentY:
        case Operation::componentZ:
            return detail::unarySchema(Type::scalar, Type::vec3);
        case Operation::scalarRemap:
            return detail::unarySchema(Type::scalar, Type::scalar, 5);
        case Operation::vec3Remap:
            return detail::unarySchema(Type::vec3, Type::vec3, 5);
        case Operation::scalarNoise3d:
            return detail::unarySchema(Type::scalar, Type::vec3, 2);
        case Operation::vec3Noise3d:
            return detail::unarySchema(Type::vec3, Type::vec3, 2);
        case Operation::boundedDisplacementOutput:
            return { Type::vec3, { Type::vec3, Type::vec3 }, 2, 1, true, true };
        case Operation::scalarSin:
        case Operation::scalarCos:
            return detail::unarySchema(Type::scalar, Type::scalar);
        case Operation::oceanWaves:
            return { Type::vec3, { Type::vec3, Type::scalar }, 2, 6, false, true };
        case Operation::invalid:
            break;
    }
    return {};
}

// A Control Parameter record may omit its authored value, which then reads as 0.
inline constexpr bool vertexModifierParameterCountAdmitted(
    VertexModifierOperation operation, std::uint8_t count) noexcept
{
    const auto expected = vertexModifierOperationSchema(operation).parameterCount;
    return count == expected
        || (operation == VertexModifierOperation::controlParameter && count == 1);
}

struct VertexModifierRecord
{
    VertexModifierStableId stableId = 0;
    VertexModifierOperation operation = VertexModifierOperation::invalid;
    VertexModifierValueType resultType = VertexModifierValueType::invalid;
    std::array<VertexModifierStableId, kVertexModifierMaximumInputsPerRecord> inputs {};
    std::uint8_t inputCount = 0;
    std::array<double, kVertexModifierMaximumParametersPerRecord> parameters {};
    std::uint8_t parameterCount = 0;
};

struct VertexModifierIr
{
    std::uint32_t schemaVersion = kVertexModifierIrSchemaVersion;
    VertexModifierStableId rootId = 0;
    std::vector<VertexModifierRecord> records;
};

// The prior deformed geometry a Motion pass feeds to the second, prior-time
// evaluation of a vertex program. That evaluation reads the prior position.
// Every other imported input still comes from the current vertex, so a
// program is exact only when those inputs did not change between the frames.
// Skinning and morphing change positions and normals and copy UV and color.
// A retained timeline Geometry3D frame keeps positions only.
enum class VertexModifierPriorGeometry : std::uint8_t
{
    none = 0,
    skinnedPose = 1,
    retainedPositions = 2
};

// Returns the refusal for a program whose prior evaluation cannot be exact
// over the given prior geometry, or nullptr when it is exact. Works on both
// wire records and compiled instructions.
template<class Records>
inline const char* vertexModifierPriorGeometryRefusal(
    const Records& records, VertexModifierPriorGeometry prior) noexcept
{
    if (prior == VertexModifierPriorGeometry::none) return nullptr;
    for (const auto& record : records)
    {
        if (record.operation == VertexModifierOperation::importedNormal)
            return "Motion output is unavailable for a graph vertex program that reads Normal over "
                   "deformed geometry, because prior normals are not kept";
        if (prior == VertexModifierPriorGeometry::retainedPositions
            && (record.operation == VertexModifierOperation::importedUv
                || record.operation == VertexModifierOperation::importedVertexColor))
            return "Motion output is unavailable for a graph vertex program that reads UV or Color over "
                   "timeline Geometry3D deformation, because only prior positions are kept";
    }
    return nullptr;
}

} // namespace videowire
