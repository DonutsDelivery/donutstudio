#pragma once

#include "DiffractionMaterialIR.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace diffractivefoil
{
inline constexpr std::uint32_t kWireVersion = 1;
inline constexpr std::size_t kFieldSampleCount = 4;
// One evaluation is one output-pixel slot reserved for one admitted lighting
// path. The reservation is charged whether rasterization covers that slot or
// not, which makes admission independent of GPU coverage and draw ordering.
inline constexpr std::uint32_t kMaximumEvaluations = 67'108'864;

// A fixed 2x2 lattice over normalized material UV. Bilinear interpolation gives
// bounded spatial variation without admitting shader source, textures, or an
// unbounded procedural graph. Period remains in physical nanometres.
struct GrooveFieldSample final
{
    std::array<float, 2> reciprocalDirectionUv { 1.0f, 0.0f };
    float grooveSpacingNanometres = 1000.0f;
    float diffractionCoverage = 1.0f;
};

struct WorkBudget final
{
    std::uint32_t maximumEvaluations = kMaximumEvaluations;
};

// Canonical row-major schedule shared by CPU admission and both native
// backends. Evaluation index = path * (width * height) + y * width + x.
// Every output-pixel slot is charged, including uncovered raster slots.
struct EvaluationSchedule final
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t pathCount = 0;
    std::uint32_t maximumEvaluations = 0;
    std::uint32_t requiredEvaluations = 0;
};

inline bool makeEvaluationSchedule(std::uint32_t maximumEvaluations,
                                   std::uint32_t width,
                                   std::uint32_t height,
                                   std::uint32_t pathCount,
                                   EvaluationSchedule& result) noexcept
{
    result = {};
    if (maximumEvaluations == 0 || maximumEvaluations > kMaximumEvaluations
        || width == 0 || height == 0 || pathCount == 0)
        return false;

    const auto pixels = static_cast<std::uint64_t>(width) * height;
    if (pixels > std::numeric_limits<std::uint32_t>::max()
        || pathCount > std::numeric_limits<std::uint32_t>::max() / pixels)
        return false;
    const auto required = static_cast<std::uint32_t>(pixels * pathCount);
    if (required > maximumEvaluations)
        return false;

    result = { width, height, pathCount, maximumEvaluations, required };
    return true;
}

inline std::uint32_t evaluationIndex(const EvaluationSchedule& schedule,
                                     std::uint32_t x,
                                     std::uint32_t y,
                                     std::uint32_t pathIndex) noexcept
{
    if (x >= schedule.width || y >= schedule.height
        || pathIndex >= schedule.pathCount)
        return std::numeric_limits<std::uint32_t>::max();
    return pathIndex * (schedule.width * schedule.height)
        + y * schedule.width + x;
}

struct Description final
{
    std::uint32_t version = kWireVersion;
    diffractionmaterial::Description physicalBsdf;
    // Row-major corners: (0,0), (1,0), (0,1), (1,1).
    std::array<GrooveFieldSample, kFieldSampleCount> grooveField {};
    WorkBudget workBudget;
};

enum class ProductUse : std::uint8_t
{
    Invalid = 0,
    Preview = 1,
    Export = 2,
    CpuReferenceValidation = 3
};

enum class Backend : std::uint8_t
{
    Invalid = 0,
    CpuReference = 1,
    OpenGl = 2,
    Metal = 3
};

inline constexpr std::string_view token(ProductUse use) noexcept
{
    switch (use)
    {
        case ProductUse::Preview: return "preview";
        case ProductUse::Export: return "export";
        case ProductUse::CpuReferenceValidation: return "cpu-reference-validation";
        case ProductUse::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(Backend backend) noexcept
{
    switch (backend)
    {
        case Backend::CpuReference: return "cpu-reference";
        case Backend::OpenGl: return "open-gl";
        case Backend::Metal: return "metal";
        case Backend::Invalid: break;
    }
    return {};
}
} // namespace diffractivefoil
