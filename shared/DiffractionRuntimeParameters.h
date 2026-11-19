#pragma once

#include "DiffractionMaterialBindingContract.h"
#include "DiffractionMaterialPresets.h"

#include <map>
#include <string>
#include <cmath>
#include <utility>

namespace diffractionmaterialbinding
{
using RuntimeParameters = std::map<std::string, double>;

// These are the existing graph parameter IDs, not a second automation format.
// Mode changes require a new authored admission, never an interpolated enum.
inline RuntimeParameters runtimeParameterValues(const ImportedSceneDiffractionMaterialRequest& request)
{
    using namespace diffractionmaterial;
    const auto& m = request.material;
    constexpr double degrees = 57.2957795130823208768;
    RuntimeParameters p {
        {"lattice", static_cast<int>(m.geometry.lattice) - static_cast<int>(GratingLattice::OneDimensional)},
        {"profile", static_cast<int>(m.microstructure.profile) - static_cast<int>(GrooveProfile::BinaryRectangular)},
        {"coating", static_cast<int>(m.coating.model) - static_cast<int>(CoatingModel::Uncoated)},
        {"grooveFieldMode", static_cast<int>(m.grooveField.mode) - static_cast<int>(GrooveFieldMode::Constant)},
        {"spectralLighting", request.lighting.version == 2 ? 1 : 0},
        {"grooveDepthNanometres", m.microstructure.grooveDepthNanometres},
        {"dutyCycle", m.microstructure.dutyCycle},
        {"rmsHeightNanometres", m.roughness.rmsHeightNanometres},
        {"rmsSlope", m.roughness.rmsSlope},
        {"substrateRefractiveIndex", m.substrate.refractiveIndex},
        {"substrateExtinctionCoefficient", m.substrate.extinctionCoefficient}
    };
    if (m.microstructure.profile == GrooveProfile::Sinusoidal) p.erase("dutyCycle");
    if (request.spatialFoil)
    {
        constexpr const char* suffix[] {"00", "10", "01", "11"};
        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto& sample = request.spatialFoil->grooveField[i];
            p[std::string("angle") + suffix[i] + "Degrees"]
                = std::atan2(sample.reciprocalDirectionUv[1], sample.reciprocalDirectionUv[0]) * degrees;
            p[std::string("period") + suffix[i] + "Nanometres"] = sample.grooveSpacingNanometres;
            p[std::string("mask") + suffix[i]] = sample.diffractionCoverage;
        }
    }
    else
    {
        p["directionAngleDegrees"] = std::atan2(m.geometry.directionUv[1], m.geometry.directionUv[0]) * degrees;
        p["grooveSpacingNanometres"] = m.geometry.grooveSpacingNanometres;
        if (m.geometry.lattice == GratingLattice::CrossedTwoDimensional)
            p["secondaryGrooveSpacingNanometres"] = m.geometry.secondaryGrooveSpacingNanometres;
    }
    if (m.coating.model != CoatingModel::Uncoated)
    {
        p["coatingThicknessNanometres"] = m.coating.thicknessNanometres;
        p["coatingRefractiveIndex"] = m.coating.opticalConstants.refractiveIndex;
        p["coatingExtinctionCoefficient"] = m.coating.opticalConstants.extinctionCoefficient;
    }
    if (m.grooveField.mode != GrooveFieldMode::Constant)
    {
        p["grooveFieldOriginU"] = m.grooveField.originUv[0];
        p["grooveFieldOriginV"] = m.grooveField.originUv[1];
        p["grooveSpacingDeltaNanometresPerUnit"] = m.grooveField.grooveSpacingDeltaNanometresPerUnit;
        p["grooveOrientationDegreesPerUnit"] = m.grooveField.orientationDegreesPerUnit;
        if (m.geometry.lattice == GratingLattice::CrossedTwoDimensional)
            p["secondarySpacingDeltaNanometresPerUnit"] = m.grooveField.secondarySpacingDeltaNanometresPerUnit;
        if (m.grooveField.mode == GrooveFieldMode::Linear)
            p["grooveFieldAxisAngleDegrees"] = std::atan2(m.grooveField.axisUv[1], m.grooveField.axisUv[0]) * degrees;
    }
    if (request.lighting.version == 2)
        p["directIntensity"] = request.lighting.paths[0].incident.radiance[0];
    return p;
}

inline bool fixedRuntimeMode(const std::string& name)
{
    return name == "lattice" || name == "profile" || name == "coating"
        || name == "grooveFieldMode" || name == "spectralLighting";
}

// Applies only changed scalar values to a frame-local copy. The caller must
// re-admit the resulting physical/foil IR and recompute its digest before use.
inline bool applyRuntimeParameters(const ImportedSceneDiffractionMaterialRequest& authored,
    const RuntimeParameters& values, ImportedSceneDiffractionMaterialRequest& output, std::string& error)
{
    const auto baseline = runtimeParameterValues(authored);
    if (values.size() > baseline.size()) { error = "Diffraction modulation has unsupported parameters"; return false; }
    auto candidate = authored;
    auto& m = candidate.material;
    using namespace diffractionmaterial;
    constexpr double radians = 0.0174532925199432957692;
    const auto direction = [&](double angle) -> std::array<float, 2>
        { return {static_cast<float>(std::cos(angle * radians)), static_cast<float>(std::sin(angle * radians))}; };
    for (const auto& [name, value] : values)
    {
        const auto original = baseline.find(name);
        if (original == baseline.end() || !std::isfinite(value))
        { error = "Diffraction modulation requires an available finite scalar: " + name; return false; }
        if (value == original->second) continue;
        if (fixedRuntimeMode(name))
        { error = "Diffraction lattice, profile, coating and field modes require an authored material edit"; return false; }
        double minimum = 0, maximum = 0;
        float* target = nullptr;
        if (name == "grooveDepthNanometres") { target = &m.microstructure.grooveDepthNanometres; minimum = kMinimumGrooveDepthNanometres; maximum = kMaximumGrooveDepthNanometres; }
        else if (name == "dutyCycle") { target = &m.microstructure.dutyCycle; minimum = 0.01; maximum = 1; }
        else if (name == "rmsHeightNanometres") { target = &m.roughness.rmsHeightNanometres; maximum = kMaximumRmsHeightNanometres; }
        else if (name == "rmsSlope") { target = &m.roughness.rmsSlope; maximum = kMaximumRmsSlope; }
        else if (name == "substrateRefractiveIndex") { target = &m.substrate.refractiveIndex; minimum = kMinimumSubstrateRefractiveIndex; maximum = kMaximumSubstrateRefractiveIndex; }
        else if (name == "substrateExtinctionCoefficient") { target = &m.substrate.extinctionCoefficient; maximum = kMaximumExtinctionCoefficient; }
        else if (name == "coatingThicknessNanometres") { target = &m.coating.thicknessNanometres; minimum = 0.1; maximum = kMaximumCoatingThicknessNanometres; }
        else if (name == "coatingRefractiveIndex") { target = &m.coating.opticalConstants.refractiveIndex; minimum = kMinimumSubstrateRefractiveIndex; maximum = kMaximumSubstrateRefractiveIndex; }
        else if (name == "coatingExtinctionCoefficient") { target = &m.coating.opticalConstants.extinctionCoefficient; maximum = kMaximumExtinctionCoefficient; }
        else if (name == "grooveSpacingNanometres") { target = &m.geometry.grooveSpacingNanometres; minimum = kMinimumGrooveSpacingNanometres; maximum = kMaximumGrooveSpacingNanometres; }
        else if (name == "secondaryGrooveSpacingNanometres") { target = &m.geometry.secondaryGrooveSpacingNanometres; minimum = kMinimumGrooveSpacingNanometres; maximum = kMaximumGrooveSpacingNanometres; }
        else if (name == "grooveFieldOriginU") { target = &m.grooveField.originUv[0]; maximum = 1; }
        else if (name == "grooveFieldOriginV") { target = &m.grooveField.originUv[1]; maximum = 1; }
        else if (name == "grooveSpacingDeltaNanometresPerUnit") { target = &m.grooveField.grooveSpacingDeltaNanometresPerUnit; minimum = -kMaximumGrooveSpacingNanometres; maximum = -minimum; }
        else if (name == "secondarySpacingDeltaNanometresPerUnit") { target = &m.grooveField.secondarySpacingDeltaNanometresPerUnit; minimum = -kMaximumGrooveSpacingNanometres; maximum = -minimum; }
        else if (name == "grooveOrientationDegreesPerUnit") { target = &m.grooveField.orientationDegreesPerUnit; minimum = -kMaximumFieldOrientationDegreesPerUnit; maximum = -minimum; }
        else if (name == "directionAngleDegrees" || name == "grooveFieldAxisAngleDegrees")
        {
            minimum = -180; maximum = 180;
            if (name == "grooveFieldAxisAngleDegrees") m.grooveField.axisUv = direction(value);
            else
            {
                // The second axis keeps its angle to the first, whatever that is.
                const auto before = m.geometry.directionUv;
                const auto secondary = m.geometry.secondaryDirectionUv;
                const float relativeCosine = before[0] * secondary[0] + before[1] * secondary[1];
                const float relativeSine = before[0] * secondary[1] - before[1] * secondary[0];
                m.geometry.directionUv = direction(value);
                if (m.geometry.lattice == GratingLattice::CrossedTwoDimensional)
                    m.geometry.secondaryDirectionUv = {
                        relativeCosine * m.geometry.directionUv[0] - relativeSine * m.geometry.directionUv[1],
                        relativeSine * m.geometry.directionUv[0] + relativeCosine * m.geometry.directionUv[1] };
            }
        }
        else if (name == "directIntensity")
        { maximum = 16; candidate.lighting.paths[0].incident.radiance.fill(static_cast<float>(value)); }
        else if (candidate.spatialFoil)
        {
            constexpr const char* suffix[] {"00", "10", "01", "11"};
            for (std::size_t i = 0; i < 4; ++i)
            {
                auto& sample = candidate.spatialFoil->grooveField[i];
                if (name == std::string("angle") + suffix[i] + "Degrees")
                { minimum = -89; maximum = 89; sample.reciprocalDirectionUv = direction(value); }
                else if (name == std::string("period") + suffix[i] + "Nanometres")
                { target = &sample.grooveSpacingNanometres; minimum = kMinimumGrooveSpacingNanometres; maximum = kMaximumGrooveSpacingNanometres; }
                else if (name == std::string("mask") + suffix[i]) { target = &sample.diffractionCoverage; maximum = 1; }
            }
        }
        if (value < minimum || value > maximum)
        { error = "Diffraction modulation is outside its physical range: " + name; return false; }
        if (target) *target = static_cast<float>(value == 0 ? 0 : value);
    }
    if (candidate.spatialFoil)
    {
        m.geometry.directionUv = candidate.spatialFoil->grooveField[0].reciprocalDirectionUv;
        m.geometry.grooveSpacingNanometres = candidate.spatialFoil->grooveField[0].grooveSpacingNanometres;
    }
    if (m.microstructure.profile == GrooveProfile::BlazedSawtooth)
        m.microstructure.blazeAngleDegrees = static_cast<float>(std::atan(
            static_cast<double>(m.microstructure.grooveDepthNanometres)
                / (static_cast<double>(m.microstructure.dutyCycle) * m.geometry.grooveSpacingNanometres)) / radians);
    if (candidate.spatialFoil) candidate.spatialFoil->physicalBsdf = m;
    output = std::move(candidate);
    error.clear();
    return true;
}
} // namespace diffractionmaterialbinding
