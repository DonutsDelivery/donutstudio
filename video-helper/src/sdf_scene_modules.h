#pragma once

#include "sdf_ir_admission.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace videohelper::sdf
{
inline constexpr const char* kRaymarchPackId = "donutstudio-3d-raymarch-v1";
inline constexpr const char* kScoreMonolithProgramId = "score-monolith";
inline constexpr const char* kScoreMonolithSourcePath
    = "arbit-3d-raymarch/shaders/score_monolith.fs";
inline constexpr const char* kScoreMonolithSourceSha256
    = "c43e4add4e4e181a4ac5e25a454c9744776c1c441d0b31f50ed3b8083a660d69";
inline constexpr const char* kHoloFoilProgramId = "holo-foil";
inline constexpr const char* kHoloFoilSourcePath
    = "arbit-3d-raymarch/shaders/holo_foil.fs";
inline constexpr const char* kHoloFoilSourceSha256
    = "88cd6f7fd88fd53c0426895574125dce9267786e50d8aa26362baab9bd133265";

enum class SdfSceneConversionScope : std::uint8_t
{
    exactStaticGeometry = 1,
    staticBaseGeometryOnly = 2
};

enum SdfSceneOmittedFeature : std::uint32_t
{
    omittedCamera = 1u << 0u,
    omittedLighting = 1u << 1u,
    omittedShading = 1u << 2u,
    omittedClockModulation = 1u << 3u,
    omittedAudioModulation = 1u << 4u,
    omittedScoreModulation = 1u << 5u,
    omittedSurfaceDisplacement = 1u << 6u
};

struct SdfSceneSource final
{
    std::string packId;
    std::string programId;
    std::string sourcePath;
    std::string sourceSha256;
    SdfSceneConversionScope conversionScope = SdfSceneConversionScope::exactStaticGeometry;
    std::uint32_t omittedFeatures = 0;
};

struct SdfMaterialBinding final
{
    videowire::SdfStableId primitiveId = 0;
    std::uint32_t materialId = 0;
};

class SdfSceneModule final
{
public:
    const std::string& stableId() const noexcept { return stableId_; }
    const std::string& displayName() const noexcept { return displayName_; }
    const SdfSceneSource& source() const noexcept { return source_; }
    const AdmittedSdfIr& geometry() const noexcept { return geometry_; }
    const std::vector<SdfMaterialBinding>& materialBindings() const noexcept
    {
        return materialBindings_;
    }

private:
    friend std::optional<SdfSceneModule> makeScoreMonolith (std::string&);
    friend std::optional<SdfSceneModule> makeHoloFoil (std::string&);

    SdfSceneModule (std::string stableId, std::string displayName,
                    SdfSceneSource source,
                    AdmittedSdfIr geometry,
                    std::vector<SdfMaterialBinding> materialBindings)
        : stableId_ (std::move (stableId)), displayName_ (std::move (displayName)),
          source_ (std::move (source)), geometry_ (std::move (geometry)),
          materialBindings_ (std::move (materialBindings))
    {
    }

    std::string stableId_;
    std::string displayName_;
    SdfSceneSource source_;
    AdmittedSdfIr geometry_;
    std::vector<SdfMaterialBinding> materialBindings_;
};

namespace detail
{
inline videowire::SdfRecord sceneModuleRecord (
    videowire::SdfStableId id, videowire::SdfOperation operation,
    std::initializer_list<videowire::SdfStableId> inputs = {},
    std::initializer_list<double> parameters = {})
{
    videowire::SdfRecord value;
    value.stableId = id;
    value.operation = operation;
    value.inputCount = static_cast<std::uint8_t> (inputs.size());
    value.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), value.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), value.parameters.begin());
    return value;
}

inline bool validateMaterialBindings (
    const AdmittedSdfIr& geometry,
    const std::vector<SdfMaterialBinding>& bindings,
    std::string& error)
{
    std::vector<videowire::SdfStableId> primitiveIds;
    for (const auto& record : geometry.records())
        if (videowire::sdfOperationSchema (record.operation).inputCount == 0)
            primitiveIds.push_back (record.stableId);

    if (bindings.size() != primitiveIds.size())
    {
        error = "SDF scene module must bind every primitive to one material";
        return false;
    }

    std::vector<videowire::SdfStableId> boundIds;
    boundIds.reserve (bindings.size());
    for (const auto& binding : bindings)
    {
        if (binding.primitiveId == 0 || binding.materialId == 0)
        {
            error = "SDF scene module material bindings require non-zero identities";
            return false;
        }
        boundIds.push_back (binding.primitiveId);
    }
    std::sort (boundIds.begin(), boundIds.end());
    if (std::adjacent_find (boundIds.begin(), boundIds.end()) != boundIds.end()
        || boundIds != primitiveIds)
    {
        error = "SDF scene module material bindings must name each primitive exactly once";
        return false;
    }
    return true;
}

} // namespace detail

inline std::optional<SdfSceneModule> makeScoreMonolith (std::string& error)
{
    error.clear();
    videowire::SdfIr source;
    source.rootId = 1004;
    source.records = {
        detail::sceneModuleRecord (1001, videowire::SdfOperation::box,
                                   {}, { 1.05, 1.5, 0.16 }),
        detail::sceneModuleRecord (1002, videowire::SdfOperation::translate,
                                   { 1001 }, { 0.0, 1.5, 0.0 }),
        detail::sceneModuleRecord (1003, videowire::SdfOperation::plane,
                                   {}, { 0.0, 1.0, 0.0, 0.0 }),
        detail::sceneModuleRecord (1004, videowire::SdfOperation::unionOp,
                                   { 1002, 1003 })
    };

    SdfAdmissionLimits limits;
    limits.maxOperations = 4;
    limits.maxDepth = 3;
    auto admitted = admitSdfIr (source, limits, error);
    std::vector<SdfMaterialBinding> bindings { { 1001, 1101 }, { 1003, 1102 } };
    if (! admitted || ! detail::validateMaterialBindings (*admitted, bindings, error))
        return std::nullopt;
    SdfSceneSource provenance {
        kRaymarchPackId, kScoreMonolithProgramId, kScoreMonolithSourcePath,
        kScoreMonolithSourceSha256, SdfSceneConversionScope::exactStaticGeometry,
        omittedCamera | omittedLighting | omittedShading | omittedClockModulation
            | omittedAudioModulation | omittedScoreModulation
    };
    return SdfSceneModule ("score-monolith-sdf-v1", "Score Monolith SDF",
                           std::move (provenance), std::move (*admitted),
                           std::move (bindings));
}

inline std::optional<SdfSceneModule> makeHoloFoil (std::string& error)
{
    error.clear();
    videowire::SdfIr source;
    source.rootId = 2001;
    source.records = {
        detail::sceneModuleRecord (2001, videowire::SdfOperation::roundedBox,
                                   {}, { 1.18, 1.62, 0.055, 0.055 })
    };

    SdfAdmissionLimits limits;
    limits.maxOperations = 1;
    limits.maxDepth = 1;
    auto admitted = admitSdfIr (source, limits, error);
    std::vector<SdfMaterialBinding> bindings { { 2001, 2101 } };
    if (! admitted || ! detail::validateMaterialBindings (*admitted, bindings, error))
        return std::nullopt;
    SdfSceneSource provenance {
        kRaymarchPackId, kHoloFoilProgramId, kHoloFoilSourcePath,
        kHoloFoilSourceSha256, SdfSceneConversionScope::staticBaseGeometryOnly,
        omittedCamera | omittedLighting | omittedShading | omittedClockModulation
            | omittedAudioModulation | omittedSurfaceDisplacement
    };
    return SdfSceneModule ("holo-foil-sdf-v1", "Holo Foil SDF",
                           std::move (provenance), std::move (*admitted),
                           std::move (bindings));
}
} // namespace videohelper::sdf
