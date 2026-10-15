#pragma once

#include "gpu_backend/backend.h"
#include "sdf_ir_admission.h"
#include "sdf_native_program.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <limits>
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
    bool normalInspection = false;
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

// Bound all actual distance-query branches: Color+Normal pass and raw Depth
// trace, then the selected diagnostic or shading path. Hit work is charged for
// every pixel; adaptive early exits do not weaken the admission bound.
inline bool admitNativeSdfTextWork (const videowire::SdfIr& geometry,
    NativeSdfRenderDimensions dimensions, NativeSdfRenderControls controls,
    arbitgpu::NativeSdfWorkReceipt& receipt, std::string& error,
    std::uint64_t maximumRecordVisits = 100000000,
    std::uint64_t maximumSegmentVisits = 100000000)
{
    receipt = {};
    error.clear();
    if (geometry.schemaVersion != videowire::kSdfIrTextSchemaVersion) return true;
    if (dimensions.width == 0 || dimensions.height == 0 || controls.maximumSteps == 0
        || ! detail::validQuality (controls.normalQuality)
        || ! detail::validQuality (controls.shadowQuality) || ! detail::validOutput (controls.output))
    {
        error = "native GPU SDF text work controls are invalid";
        return false;
    }
    std::uint64_t perPixel = 2ull * controls.maximumSteps + 6;
    switch (controls.output)
    {
        case arbitgpu::NativeSdfOutput::materialId: perPixel += 1; break;
        case arbitgpu::NativeSdfOutput::curvature: perPixel += 19; break;
        case arbitgpu::NativeSdfOutput::ambientOcclusion:
            perPixel += 4 + 4 * static_cast<std::uint8_t> (controls.normalQuality); break;
        case arbitgpu::NativeSdfOutput::color:
        case arbitgpu::NativeSdfOutput::softShadow:
            perPixel += 8ull << static_cast<std::uint8_t> (controls.shadowQuality); break;
        case arbitgpu::NativeSdfOutput::edgeDistance:
            perPixel += 40ull * controls.maximumSteps; break;
        case arbitgpu::NativeSdfOutput::depth:
        case arbitgpu::NativeSdfOutput::normal: break;
        case arbitgpu::NativeSdfOutput::count: return false;
    }
    const auto pixels = static_cast<std::uint64_t> (dimensions.width) * dimensions.height;
    if (pixels > std::numeric_limits<std::uint64_t>::max() / perPixel)
    {
        error = "native GPU SDF text query count overflows";
        return false;
    }
    const auto queries = pixels * perPixel;
    videowire::SdfTextWorkBudget budget;
    budget.maximumRecordVisits = maximumRecordVisits;
    budget.maximumSegmentVisits = maximumSegmentVisits;
    if (! videowire::chargeSdfTextIrEvaluationWork (geometry, queries, budget, error)) return false;
    receipt = { queries, budget.recordVisits, budget.segmentVisits };
    error.clear();
    return true;
}

// Direct backend consumers must use this before allocating or submitting text
// resources; callers cannot supply their own lower work receipt.
inline bool admitNativeSdfTextWork (const arbitgpu::NativeSdfDrawRequest& request,
    arbitgpu::NativeSdfWorkReceipt& receipt, std::string& error,
    std::uint64_t maximumRecordVisits = 100000000,
    std::uint64_t maximumSegmentVisits = 100000000)
{
    NativeSdfRenderControls controls;
    controls.maximumSteps = request.maximumSteps;
    controls.normalQuality = request.normalQuality;
    controls.shadowQuality = request.shadowQuality;
    controls.output = request.output;
    return admitNativeSdfTextWork (request.geometry, { request.width, request.height }, controls,
                                 receipt, error, maximumRecordVisits, maximumSegmentVisits);
}

// A shared primary trace may feed Color, raw Normal and raw Depth. These bounds
// describe real-arithmetic geometry only. Until native arithmetic and adaptive
// hit thresholds are certified, every primary pixel and displaced edge ray is
// charged, including rays whose mathematical zero set is outside the viewport.
struct NativeSdfRayWorkPlan final
{
    std::optional<arbitgpu::NativeSdfSceneBounds> sceneBounds;
    std::array<std::uint32_t, 2> firstPixel {}, endPixel {};
    std::uint64_t activePixelCount = 0;
    std::uint64_t primaryQueryCount = 0;
    std::uint64_t hitQueryCount = 0;
    std::uint32_t edgeHaloPixels = 0;
    bool certifiedRayCulling = false;
    // Lower bound only. Consumers must add real native padding, programs,
    // uniforms, primary intermediates and upload peaks against the same 2 GiB cap.
    std::uint64_t minimumStorageBytes = 0;
};

inline bool nativeSdfRayWorkPlan (const arbitgpu::NativeSdfDrawRequest& request,
    NativeSdfRayWorkPlan& plan, std::string& error)
{
    plan = {};
    if ((request.normalInspection && request.output != arbitgpu::NativeSdfOutput::normal)
        || request.width == 0 || request.height == 0 || request.maximumSteps == 0
        || ! std::isfinite (request.epsilon) || request.epsilon <= 0
        || ! std::isfinite (request.maximumDistance) || request.maximumDistance < request.epsilon
        || ! detail::validQuality (request.adaptiveQuality)
        || ! detail::validQuality (request.normalQuality)
        || ! detail::validQuality (request.shadowQuality) || ! detail::validOutput (request.output))
    {
        error = "native GPU SDF staged work controls are invalid";
        return false;
    }
    auto program = request.compiledProgram;
    if (program)
    {
        if (! validateNativeSdfProgram (*program, request.geometry, error, true)) return false;
    }
    else program = compileNativeSdfProgram (request.geometry, error, true);
    if (! program) return false;
    std::uint64_t hitQueries = 6;
    switch (request.output)
    {
        case arbitgpu::NativeSdfOutput::materialId: hitQueries += 1; break;
        case arbitgpu::NativeSdfOutput::curvature: hitQueries += 19; break;
        case arbitgpu::NativeSdfOutput::ambientOcclusion:
            hitQueries += 4 + 4 * static_cast<std::uint8_t> (request.normalQuality); break;
        case arbitgpu::NativeSdfOutput::color:
        case arbitgpu::NativeSdfOutput::softShadow:
            hitQueries += 8ull << static_cast<std::uint8_t> (request.shadowQuality); break;
        case arbitgpu::NativeSdfOutput::edgeDistance:
            hitQueries += 40ull * request.maximumSteps; break;
        case arbitgpu::NativeSdfOutput::depth:
        case arbitgpu::NativeSdfOutput::normal: break;
        case arbitgpu::NativeSdfOutput::count: return false;
    }
    const auto pixels = std::uint64_t (request.width) * request.height;
    const auto queriesPerPixel = std::uint64_t (request.maximumSteps) + hitQueries;
    if (pixels > std::numeric_limits<std::uint64_t>::max() / queriesPerPixel)
    {
        error = "native GPU SDF staged query count overflows";
        return false;
    }
    // Permanent RGBA16F Color+Normal and R32F Depth at full extent. Native
    // consumers account their actual primary intermediate format and padding.
    constexpr std::uint64_t bytesPerPixel = 8 + 8 + 4;
    const auto cpuBytes = request.geometryCacheBytes == 0 ? program->storageBytes() : request.geometryCacheBytes;
    const auto tableBytes = program->textTableBytes();
    constexpr auto maximumBytes = UINT64_C(2) * 1024 * 1024 * 1024;
    if (cpuBytes < program->storageBytes() || cpuBytes > maximumBytes
        || tableBytes > maximumBytes - cpuBytes
        || pixels > (maximumBytes - cpuBytes - tableBytes) / bytesPerPixel)
    {
        error = "native GPU SDF staged storage exceeds 2 GiB";
        return false;
    }
    plan.sceneBounds = program->sceneBounds();
    plan.endPixel = { request.width, request.height };
    plan.activePixelCount = pixels;
    plan.primaryQueryCount = pixels * request.maximumSteps;
    plan.hitQueryCount = pixels * hitQueries;
    plan.edgeHaloPixels = request.output == arbitgpu::NativeSdfOutput::edgeDistance ? 8 : 0;
    plan.minimumStorageBytes = cpuBytes + tableBytes + pixels * bytesPerPixel;
    error.clear();
    return true;
}

// Opt in only after the consumer reuses one primary result for every permanent
// attachment. Caller-provided receipts and BVH average visits cannot reduce it.
inline bool admitNativeSdfStagedTextWork (const arbitgpu::NativeSdfDrawRequest& request,
    arbitgpu::NativeSdfWorkReceipt& receipt, std::string& error,
    std::uint64_t maximumRecordVisits = 100000000,
    std::uint64_t maximumSegmentVisits = 100000000,
    std::uint64_t maximumIndexNodeVisits = 100000000)
{
    receipt = {};
    error.clear();
    if (request.geometry.schemaVersion != videowire::kSdfIrTextSchemaVersion) return true;
    NativeSdfRayWorkPlan plan;
    if (! nativeSdfRayWorkPlan (request, plan, error)) return false;
    auto program = request.compiledProgram;
    if (! program) program = compileNativeSdfProgram (request.geometry, error, true);
    if (! program) return false;
    const auto queries = plan.primaryQueryCount + plan.hitQueryCount;
    videowire::SdfTextWorkBudget budget;
    budget.maximumRecordVisits = maximumRecordVisits;
    budget.maximumSegmentVisits = maximumSegmentVisits;
    if (! videowire::chargeSdfTextIrEvaluationWork (request.geometry, queries, budget, error)) return false;
    const auto nodesPerQuery = program->textIndexNodeVisitsPerQuery();
    if (nodesPerQuery != 0 && queries > maximumIndexNodeVisits / nodesPerQuery)
    {
        error = "native GPU SDF text index node work budget exceeded";
        return false;
    }
    receipt = { queries, budget.recordVisits, budget.segmentVisits, queries * nodesPerQuery };
    error.clear();
    return true;
}

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
    if ((controls.normalInspection && controls.output != arbitgpu::NativeSdfOutput::normal)
        || controls.maximumSteps == 0 || controls.maximumSteps > capabilities.maxSteps
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

    if (! geometry.textOutlines().empty())
    {
        videowire::SdfIr source;
        source.schemaVersion = geometry.schemaVersion();
        source.rootId = geometry.rootId();
        source.records = geometry.records();
        for (const auto& outline : geometry.textOutlines()) source.textOutlines.push_back (outline.outline());
        arbitgpu::NativeSdfWorkReceipt work;
        if (capabilities.reusesPrimaryTrace)
        {
            arbitgpu::NativeSdfDrawRequest request;
            request.geometry = std::move (source);
            request.width = dimensions.width; request.height = dimensions.height;
            request.maximumSteps = controls.maximumSteps;
            request.epsilon = controls.epsilon; request.maximumDistance = controls.maximumDistance;
            request.adaptiveQuality = controls.adaptiveQuality; request.normalQuality = controls.normalQuality;
            request.shadowQuality = controls.shadowQuality; request.output = controls.output;
            if (! admitNativeSdfStagedTextWork (request, work, error)) return std::nullopt;
        }
        else if (! admitNativeSdfTextWork (source, dimensions, controls, work, error)) return std::nullopt;
    }
    return AdmittedNativeSdfRender (
        geometry, dimensions, controls, capabilities.backend);
}
} // namespace videohelper::sdf
