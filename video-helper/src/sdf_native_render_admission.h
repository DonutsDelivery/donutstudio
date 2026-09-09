#pragma once

#include "gpu_backend/backend.h"
#include "sdf_ir_admission.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace videohelper::sdf
{
struct NativeSdfRenderDimensions final
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct NativeSdfRenderControls final
{
    std::uint32_t maximumSteps = 128;
    double epsilon = 0.001;
    double maximumDistance = 100.0;
    arbitgpu::NativeSdfQuality adaptiveQuality = arbitgpu::NativeSdfQuality::medium;
    arbitgpu::NativeSdfQuality normalQuality = arbitgpu::NativeSdfQuality::medium;
    arbitgpu::NativeSdfQuality shadowQuality = arbitgpu::NativeSdfQuality::medium;
    arbitgpu::NativeSdfOutput output = arbitgpu::NativeSdfOutput::color;
};

class AdmittedNativeSdfRender final
{
public:
    const AdmittedSdfIr& geometry() const noexcept { return geometry_; }
    NativeSdfRenderDimensions dimensions() const noexcept { return dimensions_; }
    NativeSdfRenderControls controls() const noexcept { return controls_; }
    const std::string& backend() const noexcept { return backend_; }

private:
    friend std::optional<AdmittedNativeSdfRender> admitNativeSdfRender (
        const AdmittedSdfIr&, NativeSdfRenderDimensions, NativeSdfRenderControls,
        const arbitgpu::NativeSdfExecutionCapabilities&, std::string&);

    AdmittedNativeSdfRender (AdmittedSdfIr geometry,
                             NativeSdfRenderDimensions dimensions,
                             NativeSdfRenderControls controls,
                             std::string backend)
        : geometry_ (std::move (geometry)), dimensions_ (dimensions), controls_ (controls),
          backend_ (std::move (backend))
    {
    }

    AdmittedSdfIr geometry_;
    NativeSdfRenderDimensions dimensions_;
    NativeSdfRenderControls controls_;
    std::string backend_;
};

namespace detail
{
inline bool validQuality (arbitgpu::NativeSdfQuality quality) noexcept
{
    return static_cast<std::uint8_t> (quality)
        < static_cast<std::uint8_t> (arbitgpu::NativeSdfQuality::count);
}

inline bool validOutput (arbitgpu::NativeSdfOutput output) noexcept
{
    return static_cast<std::uint8_t> (output)
        < static_cast<std::uint8_t> (arbitgpu::NativeSdfOutput::count);
}
} // namespace detail

inline std::optional<AdmittedNativeSdfRender> admitNativeSdfRender (
    const AdmittedSdfIr& geometry,
    NativeSdfRenderDimensions dimensions,
    NativeSdfRenderControls controls,
    const arbitgpu::NativeSdfExecutionCapabilities& capabilities,
    std::string& error)
{
    error.clear();
    if (! capabilities.available)
    {
        error = capabilities.error.empty()
            ? "native GPU SDF execution is unavailable" : capabilities.error;
        return std::nullopt;
    }
    if (capabilities.backend.empty() || capabilities.maxOperations == 0
        || capabilities.maxDepth == 0 || capabilities.maxExtent == 0
        || capabilities.maxPixels == 0 || capabilities.maxSteps == 0
        || ! std::isfinite (capabilities.minEpsilon)
        || ! std::isfinite (capabilities.maxEpsilon)
        || ! std::isfinite (capabilities.maxDistance)
        || capabilities.minEpsilon <= 0.0
        || capabilities.maxEpsilon < capabilities.minEpsilon
        || capabilities.maxDistance < capabilities.minEpsilon)
    {
        error = "native GPU SDF execution capabilities are invalid";
        return std::nullopt;
    }
    if (dimensions.width == 0 || dimensions.height == 0
        || dimensions.width > capabilities.maxExtent
        || dimensions.height > capabilities.maxExtent
        || static_cast<std::uint64_t> (dimensions.width) * dimensions.height
            > capabilities.maxPixels)
    {
        error = "native GPU SDF render dimensions exceed backend limits";
        return std::nullopt;
    }
    if (controls.maximumSteps == 0 || controls.maximumSteps > capabilities.maxSteps
        || ! std::isfinite (controls.epsilon)
        || ! std::isfinite (controls.maximumDistance)
        || controls.epsilon < capabilities.minEpsilon
        || controls.epsilon > capabilities.maxEpsilon
        || controls.maximumDistance < controls.epsilon
        || controls.maximumDistance > capabilities.maxDistance
        || ! detail::validQuality (controls.adaptiveQuality)
        || ! detail::validQuality (controls.normalQuality)
        || ! detail::validQuality (controls.shadowQuality)
        || ! detail::validOutput (controls.output))
    {
        error = "native GPU SDF raymarch controls exceed backend limits";
        return std::nullopt;
    }
    if (geometry.operationCount() > capabilities.maxOperations
        || geometry.maximumDepth() > capabilities.maxDepth)
    {
        error = "native GPU SDF geometry exceeds backend limits";
        return std::nullopt;
    }
    for (const auto& record : geometry.records())
        if (! capabilities.supports (record.operation))
        {
            error = "native GPU SDF backend does not support operation ";
            error += videowire::sdfOperationWireToken (record.operation);
            return std::nullopt;
        }
    if (! capabilities.supports (controls.output))
    {
        error = "native GPU SDF backend does not support output ";
        error += arbitgpu::nativeSdfOutputToken (controls.output);
        return std::nullopt;
    }

    return AdmittedNativeSdfRender (
        geometry, dimensions, controls, capabilities.backend);
}
} // namespace videohelper::sdf