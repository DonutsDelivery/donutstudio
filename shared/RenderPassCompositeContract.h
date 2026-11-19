#pragma once

#include "RenderPassOutputContract.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

namespace renderpasscomposite
{
inline constexpr const char* kNodeKind = "visual.3d.pass-composite";
// Persisted modes. Pass branches share one Render 3D draw and its raw textures.
enum class Mode : std::uint32_t
{
    MaskedColor, DepthFog, DepthView, NormalView, EmissionView, MaskView,
    MaterialMatte, ObjectMatte, MotionView, Mix, Add, Count
};
enum class ColorInput : std::uint32_t { Linear, SRGB, Count };
enum class OutputTransform : std::uint32_t { LegacyClip, LinearToSRGB, ReinhardToSRGB, Count };
inline constexpr std::size_t kMaximumPasses = 8;
inline constexpr int kLinearOutputPort = 9, kBranchA = 10, kBranchB = 11;
struct Parameters
{
    Mode mode = Mode::DepthFog;
    float nearDepth = 0.0f;
    float farDepth = 1.0f;
    float amount = 1.0f;
    std::array<float, 3> fogColor { 0.5f, 0.5f, 0.5f };
    std::uint32_t identity = 1;
    ColorInput fogColorSpace = ColorInput::Linear;
    OutputTransform outputTransform = OutputTransform::LegacyClip;
    float exposure = 0.0f;
    bool inspectionImage = false;
    bool invertDepth = false;
};
inline bool branches(Mode mode) noexcept { return mode == Mode::Mix || mode == Mode::Add; }
inline bool sameParameters(const Parameters& a, const Parameters& b) noexcept
{
    return a.mode == b.mode && a.nearDepth == b.nearDepth && a.farDepth == b.farDepth
        && a.amount == b.amount && a.fogColor == b.fogColor && a.identity == b.identity
        && a.fogColorSpace == b.fogColorSpace && a.outputTransform == b.outputTransform
        && a.exposure == b.exposure && a.inspectionImage == b.inspectionImage && a.invertDepth == b.invertDepth;
}
inline constexpr std::array<renderpassoutput::Output, 8> kInputs {
    renderpassoutput::Output::Color, renderpassoutput::Output::Depth,
    renderpassoutput::Output::Normal, renderpassoutput::Output::Emission,
    renderpassoutput::Output::Mask, renderpassoutput::Output::MaterialId,
    renderpassoutput::Output::ObjectId, renderpassoutput::Output::Motion
};
// Output 7 is persisted. Motion appends input 8 without moving existing ports.
inline constexpr int inputPort(std::size_t index) noexcept { return index == 7 ? 8 : static_cast<int>(index); }
inline constexpr int renderPort(renderpassoutput::Output output) noexcept
{
    switch (output)
    {
        case renderpassoutput::Output::Color: return 1;
        case renderpassoutput::Output::Depth: return 4;
        case renderpassoutput::Output::Normal: return 7;
        case renderpassoutput::Output::Emission: return 8;
        case renderpassoutput::Output::Mask: return 9;
        case renderpassoutput::Output::MaterialId: return 10;
        case renderpassoutput::Output::ObjectId: return 11;
        case renderpassoutput::Output::Motion: return 12;
        default: return -1;
    }
}
inline bool valid(const Parameters& value) noexcept
{
    if (value.mode >= Mode::Count || !std::isfinite(value.nearDepth)
        || !std::isfinite(value.farDepth) || value.nearDepth < 0.0f
        || value.farDepth > 1.0f || value.nearDepth >= value.farDepth
        || !std::isfinite(value.amount) || value.amount < 0.0f || value.amount > 1.0f
        || value.fogColorSpace >= ColorInput::Count || value.outputTransform >= OutputTransform::Count
        || !std::isfinite(value.exposure) || value.exposure < -8.0f || value.exposure > 8.0f
        || (value.inspectionImage && value.mode != Mode::DepthView && value.mode != Mode::NormalView)
        || (value.invertDepth && (!value.inspectionImage || value.mode != Mode::DepthView)))
        return false;
    for (const auto channel : value.fogColor)
        if (!std::isfinite(channel) || channel < 0.0f || channel > 64.0f) return false;
    return true;
}
inline std::string encode(const Parameters& value)
{
    if (!valid(value)) return {};
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    const bool legacy = value.fogColorSpace == ColorInput::Linear
        && value.outputTransform == OutputTransform::LegacyClip && value.exposure == 0.0f
        && !value.inspectionImage && value.mode < Mode::Mix && value.fogColor[0] <= 1.0f
        && value.fogColor[1] <= 1.0f && value.fogColor[2] <= 1.0f;
    const bool typedInspection = value.inspectionImage;
    stream << (typedInspection ? "RenderPassCompositeV3 " : legacy ? "RenderPassCompositeV1 " : "RenderPassCompositeV2 ") << static_cast<unsigned>(value.mode)
           << std::setprecision(9) << ' ' << value.nearDepth << ' ' << value.farDepth
           << ' ' << value.amount;
    for (const auto channel : value.fogColor) stream << ' ' << channel;
    stream << ' ' << value.identity;
    if (!legacy) stream << ' ' << static_cast<unsigned>(value.fogColorSpace)
        << ' ' << static_cast<unsigned>(value.outputTransform) << ' ' << value.exposure;
    if (typedInspection) stream << ' ' << (value.invertDepth ? 1 : 0);
    return stream.str();
}
inline bool decode(const std::string& encoded, Parameters& destination)
{
    Parameters value;
    unsigned mode = 0;
    std::string version;
    std::istringstream stream(encoded);
    stream.imbue(std::locale::classic());
    if (!(stream >> version >> mode >> value.nearDepth >> value.farDepth >> value.amount
          >> value.fogColor[0] >> value.fogColor[1] >> value.fogColor[2] >> value.identity)
        || (version != "RenderPassCompositeV1" && version != "RenderPassCompositeV2" && version != "RenderPassCompositeV3")
        || mode >= static_cast<unsigned>(Mode::Count))
        return false;
    value.mode = static_cast<Mode>(mode);
    if (version == "RenderPassCompositeV2" || version == "RenderPassCompositeV3")
    {
        unsigned color = 0, transform = 0;
        if (!(stream >> color >> transform >> value.exposure)) return false;
        value.fogColorSpace = static_cast<ColorInput>(color);
        value.outputTransform = static_cast<OutputTransform>(transform);
    }
    if (version == "RenderPassCompositeV3")
    {
        unsigned invert = 0;
        if (!(stream >> invert) || invert > 1) return false;
        value.inspectionImage = true;
        value.invertDepth = invert != 0;
    }
    if (!valid(value) || encode(value) != encoded) return false;
    destination = value;
    return true;
}

struct Step
{
    Parameters parameters;
    int inputA = -1, inputB = -1;
};
struct Program
{
    std::array<Step, kMaximumPasses> steps {};
    std::size_t count = 0, output = 0;
};
inline bool valid(const Program& value) noexcept
{
    if (value.count == 0 || value.count > kMaximumPasses || value.output >= value.count) return false;
    for (std::size_t i = 0; i < value.count; ++i)
    {
        const auto& step = value.steps[i];
        if (!valid(step.parameters)) return false;
        if (branches(step.parameters.mode))
        {
            if (step.inputA < 0 || step.inputB < 0
                || static_cast<std::size_t>(step.inputA) >= i
                || static_cast<std::size_t>(step.inputB) >= i) return false;
        }
        else if (step.inputA != -1 || step.inputB != -1) return false;
    }
    return true;
}
inline bool usesMotion(const Program& value) noexcept
{
    for (std::size_t i = 0; i < value.count && i < kMaximumPasses; ++i)
        if (value.steps[i].parameters.mode == Mode::MotionView) return true;
    return false;
}
// Float32 registers keep HDR branch values until the one selected display output.
// No intermediate RGBA8 texture or output transform occurs between branches.
struct GpuProgram
{
    std::array<float, 4> control {};
    std::array<std::array<float, 4>, kMaximumPasses> modes {}, colors {};
    std::array<std::array<float, 4>, kMaximumPasses> references {};
    std::array<std::array<float, 4>, kMaximumPasses> transforms {};
};
static_assert(sizeof(GpuProgram) == 528);
inline GpuProgram gpuProgram(const Program& program) noexcept
{
    GpuProgram gpu;
    gpu.control = {static_cast<float>(program.count), static_cast<float>(program.output), 0, 0};
    for (std::size_t i = 0; i < program.count && i < kMaximumPasses; ++i)
    {
        const auto& step = program.steps[i];
        const auto& p = step.parameters;
        gpu.modes[i] = {static_cast<float>(p.mode), p.nearDepth, p.farDepth, p.amount};
        gpu.colors[i] = {p.fogColor[0], p.fogColor[1], p.fogColor[2], 0};
        gpu.references[i] = {static_cast<float>(p.identity & 65535u), static_cast<float>(p.identity >> 16u),
            static_cast<float>(step.inputA + 1), static_cast<float>(step.inputB + 1)};
        gpu.transforms[i] = {p.exposure, static_cast<float>(p.fogColorSpace),
            static_cast<float>(p.outputTransform), (p.inspectionImage ? 1.0f : 0.0f) + (p.invertDepth ? 2.0f : 0.0f)};
    }
    return gpu;
}

// The native raw depth attachment clears to one and Normal clears to zero.
// Inspection maps those uncovered pixels exactly as it maps covered pixels.
inline std::array<float, 4> inspectionClear(const Parameters& parameters) noexcept
{
    const float value = parameters.mode == Mode::DepthView
        ? (parameters.invertDepth ? 0.0f : 1.0f) : 0.5f;
    return {value, value, value, 1.0f};
}
} // namespace renderpasscomposite
