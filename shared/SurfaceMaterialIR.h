#pragma once

#include "DiffractionLayerIR.h"
#include "DiffractionMaterialIR.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

namespace surfacematerial
{
using ValueId = std::uint32_t;

inline constexpr std::uint32_t kWireVersion = 1;
inline constexpr std::uint32_t kSpectralWireVersion = 2;
inline constexpr std::size_t kProgramOutputCount = 11;
// Wavelengths are in micrometres. IOR remains the sodium d-line index.
inline constexpr std::array<float, 3> kGlassWavelengths { 0.6563f, 0.5461f, 0.4358f };
// Finiteness at these endpoints covers every visible index, because nonnegative
// B is monotone across the visible interval.
inline constexpr std::array<float, 2> kGlassIorAdmissionWavelengths {0.38f,0.78f};
inline constexpr float kGlassReferenceWavelength = 0.58756f;
enum class TransportMode : std::uint8_t { ThinSurface = 0, ClosedVolume = 1, ThinInterface = 2 };
inline constexpr std::size_t outputCount(std::uint32_t version) noexcept
{
    return version == kWireVersion ? 10 : version == kSpectralWireVersion ? kProgramOutputCount : 0;
}
// B has no upper bound. Far enough into the red, Cauchy's formula drops below
// vacuum, where it no longer describes a dielectric, so that band stops at index
// 1 and passes straight through. Blue keeps rising, so the fringe keeps widening.
inline float wavelengthIor(float ior, float dispersion, float wavelength) noexcept
{
    const auto index = ior + dispersion * (1.0f / (wavelength * wavelength)
        - 1.0f / (kGlassReferenceWavelength * kGlassReferenceWavelength));
    return index < 1.0f ? 1.0f : index;
}
// Abbe number V = (n_d - 1) / (n_F - n_C), with the Fraunhofer F and C lines in
// micrometres. Under wavelengthIor, n_F - n_C = B (1/F^2 - 1/C^2), so a glass
// with d-line index n and Abbe number V has B = (n - 1) / (V (1/F^2 - 1/C^2)).
// Crown glass is about V 60, flint about V 30; a lower number disperses more.
inline constexpr double kAbbeFLine = 0.4861327;
inline constexpr double kAbbeCLine = 0.6562725;
inline constexpr double kMinimumAbbeNumber = 5;
inline constexpr double kMaximumAbbeNumber = 150;
inline double cauchyBPerIorExcess(double abbe) noexcept
{
    return 1.0 / (abbe * (1.0 / (kAbbeFLine * kAbbeFLine) - 1.0 / (kAbbeCLine * kAbbeCLine)));
}
inline double cauchyBFromAbbe(double ior, double abbe) noexcept
{
    return (ior - 1.0) * cauchyBPerIorExcess(abbe);
}
inline double abbeFromCauchyB(double ior, double b) noexcept
{
    return (ior - 1.0) / (b * (1.0 / (kAbbeFLine * kAbbeFLine) - 1.0 / (kAbbeCLine * kAbbeCLine)));
}
inline constexpr std::size_t kMaximumTextureSlots = 8;
// Value identities are 32 bits and zero means no value, so this is the most a
// program can address. It comes from the program format.
inline constexpr std::size_t kMaximumOperations = std::numeric_limits<ValueId>::max();
// Admission and lowering walk with explicit stacks, so depth has no limit of its own.
inline constexpr std::size_t kMaximumDepth = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kMaximumOperationInputs = 4;
// Operation parameters are uint16 and a program needs parameter + 1 slots, so 65535 is the format limit.
inline constexpr std::size_t kMaximumControlInputs = 65535;
inline constexpr float kMaximumEvaluationMagnitude = 1.0e6f;
inline constexpr float kMaximumLinearColorComponent = 65504.0f;
inline constexpr float kMinimumMaterialIor = 1.0f;
inline constexpr float kMaximumMaterialIor = 3.0f;

enum class ValueType : std::uint8_t
{
    Invalid = 0,
    Scalar = 1,
    Vec2 = 2,
    Vec3 = 3,
    Vec4 = 4,
    UInt = 5
};

enum class InputSemantic : std::uint8_t
{
    Invalid = 0,
    Position = 1,
    Normal = 2,
    Tangent = 3,
    TexCoord0 = 4,
    TexCoord1 = 5,
    VertexColor = 6,
    ViewDirection = 7,
    Time = 8,
    AudioLevel = 9,
    AudioBass = 10,
    AudioMid = 11,
    AudioTreble = 12,
    AudioBeat = 13,
    Control = 14
};

enum class OperationKind : std::uint8_t
{
    Invalid = 0,
    FloatConstant = 1,
    UIntConstant = 2,
    Input = 3,
    TextureSample2D = 4,
    Add = 5,
    Subtract = 6,
    Multiply = 7,
    Divide = 8,
    Minimum = 9,
    Maximum = 10,
    Clamp = 11,
    Mix = 12,
    Dot = 13,
    Normalize = 14,
    Length = 15,
    Absolute = 16,
    Power = 17,
    Component = 18,
    ComposeVec2 = 19,
    ComposeVec3 = 20,
    ComposeVec4 = 21
};

enum class OutputSemantic : std::uint8_t
{
    BaseColor = 0,
    Metallic = 1,
    Roughness = 2,
    Emission = 3,
    Opacity = 4,
    Normal = 5,
    Transmission = 6,
    Ior = 7,
    Clearcoat = 8,
    MaterialId = 9,
    Dispersion = 10,
    Count = 11
};

inline constexpr std::array<ValueType, 5> kValueTypes {
    ValueType::Scalar,
    ValueType::Vec2,
    ValueType::Vec3,
    ValueType::Vec4,
    ValueType::UInt
};

inline constexpr std::array<InputSemantic, 14> kInputSemantics {
    InputSemantic::Position,
    InputSemantic::Normal,
    InputSemantic::Tangent,
    InputSemantic::TexCoord0,
    InputSemantic::TexCoord1,
    InputSemantic::VertexColor,
    InputSemantic::ViewDirection,
    InputSemantic::Time,
    InputSemantic::AudioLevel,
    InputSemantic::AudioBass,
    InputSemantic::AudioMid,
    InputSemantic::AudioTreble,
    InputSemantic::AudioBeat,
    InputSemantic::Control
};

inline constexpr std::array<OperationKind, 21> kOperationKinds {
    OperationKind::FloatConstant,
    OperationKind::UIntConstant,
    OperationKind::Input,
    OperationKind::TextureSample2D,
    OperationKind::Add,
    OperationKind::Subtract,
    OperationKind::Multiply,
    OperationKind::Divide,
    OperationKind::Minimum,
    OperationKind::Maximum,
    OperationKind::Clamp,
    OperationKind::Mix,
    OperationKind::Dot,
    OperationKind::Normalize,
    OperationKind::Length,
    OperationKind::Absolute,
    OperationKind::Power,
    OperationKind::Component,
    OperationKind::ComposeVec2,
    OperationKind::ComposeVec3,
    OperationKind::ComposeVec4
};

inline constexpr std::size_t kSurfaceOutputCount
    = 10; // Released v1 outputs and terminal port identities remain fixed.

inline constexpr std::array<OutputSemantic, kSurfaceOutputCount> kSurfaceOutputs {
    OutputSemantic::BaseColor,
    OutputSemantic::Metallic,
    OutputSemantic::Roughness,
    OutputSemantic::Emission,
    OutputSemantic::Opacity,
    OutputSemantic::Normal,
    OutputSemantic::Transmission,
    OutputSemantic::Ior,
    OutputSemantic::Clearcoat,
    OutputSemantic::MaterialId
};

inline constexpr std::string_view token(ValueType type) noexcept
{
    switch (type)
    {
        case ValueType::Scalar: return "scalar";
        case ValueType::Vec2: return "vec2";
        case ValueType::Vec3: return "vec3";
        case ValueType::Vec4: return "vec4";
        case ValueType::UInt: return "uint";
        case ValueType::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(InputSemantic input) noexcept
{
    switch (input)
    {
        case InputSemantic::Position: return "position";
        case InputSemantic::Normal: return "normal";
        case InputSemantic::Tangent: return "tangent";
        case InputSemantic::TexCoord0: return "texCoord0";
        case InputSemantic::TexCoord1: return "texCoord1";
        case InputSemantic::VertexColor: return "vertexColor";
        case InputSemantic::ViewDirection: return "viewDirection";
        case InputSemantic::Time: return "time";
        case InputSemantic::AudioLevel: return "audioLevel";
        case InputSemantic::AudioBass: return "audioBass";
        case InputSemantic::AudioMid: return "audioMid";
        case InputSemantic::AudioTreble: return "audioTreble";
        case InputSemantic::AudioBeat: return "audioBeat";
        case InputSemantic::Control: return "control";
        case InputSemantic::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(OperationKind kind) noexcept
{
    switch (kind)
    {
        case OperationKind::FloatConstant: return "floatConstant";
        case OperationKind::UIntConstant: return "uintConstant";
        case OperationKind::Input: return "input";
        case OperationKind::TextureSample2D: return "textureSample2D";
        case OperationKind::Add: return "add";
        case OperationKind::Subtract: return "subtract";
        case OperationKind::Multiply: return "multiply";
        case OperationKind::Divide: return "divide";
        case OperationKind::Minimum: return "minimum";
        case OperationKind::Maximum: return "maximum";
        case OperationKind::Clamp: return "clamp";
        case OperationKind::Mix: return "mix";
        case OperationKind::Dot: return "dot";
        case OperationKind::Normalize: return "normalize";
        case OperationKind::Length: return "length";
        case OperationKind::Absolute: return "absolute";
        case OperationKind::Power: return "power";
        case OperationKind::Component: return "component";
        case OperationKind::ComposeVec2: return "composeVec2";
        case OperationKind::ComposeVec3: return "composeVec3";
        case OperationKind::ComposeVec4: return "composeVec4";
        case OperationKind::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(OutputSemantic output) noexcept
{
    switch (output)
    {
        case OutputSemantic::BaseColor: return "baseColor";
        case OutputSemantic::Metallic: return "metallic";
        case OutputSemantic::Roughness: return "roughness";
        case OutputSemantic::Emission: return "emission";
        case OutputSemantic::Opacity: return "opacity";
        case OutputSemantic::Normal: return "normal";
        case OutputSemantic::Transmission: return "transmission";
        case OutputSemantic::Ior: return "ior";
        case OutputSemantic::Clearcoat: return "clearcoat";
        case OutputSemantic::MaterialId: return "materialId";
        case OutputSemantic::Dispersion: return "dispersion";
        case OutputSemantic::Count: break;
    }
    return {};
}

inline constexpr std::string_view outputName(OutputSemantic output) noexcept
{
    return token(output);
}

inline constexpr ValueType outputType(OutputSemantic output) noexcept
{
    switch (output)
    {
        case OutputSemantic::BaseColor:
        case OutputSemantic::Emission:
        case OutputSemantic::Normal:
            return ValueType::Vec3;
        case OutputSemantic::Metallic:
        case OutputSemantic::Roughness:
        case OutputSemantic::Opacity:
        case OutputSemantic::Transmission:
        case OutputSemantic::Ior:
        case OutputSemantic::Clearcoat:
        case OutputSemantic::Dispersion:
            return ValueType::Scalar;
        case OutputSemantic::MaterialId:
            return ValueType::UInt;
        case OutputSemantic::Count:
            break;
    }
    return ValueType::Invalid;
}

inline constexpr ValueType inputType(InputSemantic input) noexcept
{
    switch (input)
    {
        case InputSemantic::Position:
        case InputSemantic::Normal:
        case InputSemantic::ViewDirection:
            return ValueType::Vec3;
        case InputSemantic::Tangent:
        case InputSemantic::VertexColor:
            return ValueType::Vec4;
        case InputSemantic::TexCoord0:
        case InputSemantic::TexCoord1:
            return ValueType::Vec2;
        case InputSemantic::Time:
        case InputSemantic::AudioLevel:
        case InputSemantic::AudioBass:
        case InputSemantic::AudioMid:
        case InputSemantic::AudioTreble:
        case InputSemantic::AudioBeat:
        case InputSemantic::Control:
            return ValueType::Scalar;
        case InputSemantic::Invalid:
            break;
    }
    return ValueType::Invalid;
}

struct Operation
{
    ValueId id = 0;
    OperationKind kind = OperationKind::Invalid;
    ValueType resultType = ValueType::Invalid;
    std::uint8_t inputCount = 0;
    std::array<ValueId, kMaximumOperationInputs> inputs {};
    std::array<float, 4> literal {};
    InputSemantic semantic = InputSemantic::Invalid;
    std::uint16_t parameter = 0;
    std::uint32_t unsignedLiteral = 0;
};

// Explicit measured/authored reflectance at the production diffraction bands.
// RGB BaseColor has no unique inverse spectrum and never supplies these values.
inline constexpr std::uint32_t kReflectanceWireVersion = 1;
static_assert(diffractionmaterial::kMaximumSpectralSamples == 8,
              "The reflectance companion has exactly eight production diffraction bands");
struct SpectralReflectanceDescription
{
    std::uint32_t version = kReflectanceWireVersion;
    std::array<float, diffractionmaterial::kMaximumSpectralSamples> coefficients {};
};

inline bool validSpectralReflectance(const SpectralReflectanceDescription& source) noexcept
{
    if (source.version != kReflectanceWireVersion) return false;
    for (const auto coefficient : source.coefficients)
        if (!std::isfinite(coefficient) || coefficient < 0.0f || coefficient > 1.0f)
            return false;
    return true;
}

struct ProgramDescription
{
    std::uint32_t version = kWireVersion;
    std::uint16_t textureSlotCount = 0;
    std::vector<Operation> operations;
    std::array<ValueId, kProgramOutputCount> outputs {};
    TransportMode transport = TransportMode::ThinSurface;
    // Optional companion. Released program output and uniform layouts stay fixed.
    std::optional<SpectralReflectanceDescription> spectralReflectance;
    // Optional companion. Absent means no diffraction layer, which is how every
    // document written before the layer existed renders (DiffractionLayerIR.h).
    std::optional<DiffractionLayerDescription> diffractionLayer;

    ValueId output(OutputSemantic semantic) const noexcept
    {
        const auto index = static_cast<std::size_t>(semantic);
        return index < outputs.size() ? outputs[index] : ValueId {};
    }
};

struct AdmissionLimits
{
    std::size_t textureSlots = kMaximumTextureSlots;
    std::size_t operations = kMaximumOperations;
    std::size_t depth = kMaximumDepth;
    std::size_t controlInputs = kMaximumControlInputs;
};

struct MaterialEvaluationInputs
{
    float timeSeconds = 0.0f;
    float audioLevel = 0.0f;
    float audioBass = 0.0f;
    float audioMid = 0.0f;
    float audioTreble = 0.0f;
    float audioBeat = 0.0f;
    std::uint16_t controlCount = 0;
    // Sized by the caller; at least controlCount entries.
    std::vector<float> controls;
};

// Reference output layout for compiler/backend contract checks. Colors stay in
// linear space. Producing this block does not sample textures or render pixels.
struct alignas(16) SurfacePbrParameterBlock
{
    std::array<float, 4> baseColorMetallic {};
    std::array<float, 4> emissionRoughness {};
    std::array<float, 4> normalOpacity {};
    std::array<float, 4> transmissionIorClearcoat {};
    std::array<std::uint32_t, 4> identifiers {};
};

static_assert(kSurfaceOutputCount == 10,
              "The admitted surface material contract has exactly ten outputs");
static_assert(static_cast<std::uint8_t>(ValueType::UInt) == kValueTypes.size(),
              "Value type IDs are stable and contiguous");
static_assert(static_cast<std::uint8_t>(InputSemantic::Control) == kInputSemantics.size(),
              "Input semantic IDs are stable and contiguous");
static_assert(static_cast<std::uint8_t>(OperationKind::ComposeVec4) == kOperationKinds.size(),
              "Operation kind IDs are stable and contiguous");
static_assert(std::is_standard_layout_v<SurfacePbrParameterBlock>
                  && std::is_trivially_copyable_v<SurfacePbrParameterBlock>,
              "The PBR parameter block must remain directly uploadable");
static_assert(sizeof(SurfacePbrParameterBlock) == 80
                  && alignof(SurfacePbrParameterBlock) == 16,
              "The PBR parameter block layout is fixed at five 16-byte lanes");
} // namespace surfacematerial
