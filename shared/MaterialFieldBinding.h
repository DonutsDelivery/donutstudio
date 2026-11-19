#pragma once

#include "GeometryCoreTransport.h"
#include "DiffractionRuntimeParameters.h"
#include <new>
#include <tuple>
#include <iterator>

namespace materialfield
{
inline constexpr int kInputPort = 2;
// The field's size comes from its input. Only the authored sample index has a
// range, because node parameters are floats and integers stay exact up to 2^24.
inline constexpr std::size_t kMaximumSampleIndex = (std::size_t{1} << 24u) - 1u;
inline constexpr const char* kTargets[] {
    "rmsHeightNanometres", "rmsSlope", "grooveDepthNanometres", "dutyCycle",
    "grooveSpacingNanometres", "directionAngleDegrees", "secondaryGrooveSpacingNanometres",
    "substrateRefractiveIndex", "substrateExtinctionCoefficient", "coatingThicknessNanometres",
    "coatingRefractiveIndex", "coatingExtinctionCoefficient", "directIntensity",
    "period00Nanometres", "period10Nanometres", "period01Nanometres", "period11Nanometres",
    "angle00Degrees", "angle10Degrees", "angle01Degrees", "angle11Degrees",
    "mask00", "mask10", "mask01", "mask11", "grooveFieldOriginU", "grooveFieldOriginV",
    "grooveSpacingDeltaNanometresPerUnit", "secondarySpacingDeltaNanometresPerUnit",
    "grooveOrientationDegreesPerUnit", "grooveFieldAxisAngleDegrees"
};
enum class Reduction : std::uint8_t { index, mean, minimum, maximum };
struct Binding final
{
    int materialNode = -1;
    std::uint32_t target = 0;
    Reduction reduction = Reduction::mean;
    std::uint32_t index = 0;
    double gain = 1;
    std::string fieldPlan;
};

// Topology projection for the original physical-material pure vertex route.
// Surface sampling additionally admits canonical, retained Geometry Core plans.
#if defined(__GNUC__)
#pragma GCC diagnostic push
// These switches name the operation codes they accept and reject the rest in default.
#pragma GCC diagnostic ignored "-Wswitch-enum"
#endif
inline std::string nodeKind(const videowire::geometry::OperationRecord& op)
{
    using namespace videowire::geometry;
    if (op.field.domain != Domain::vertex) return {};
    const char* token = nullptr;
    switch (op.code)
    {
        case OperationCode::fieldConstant:
            return std::string("visual.geometry.field.") + toWire(op.field.valueType);
        case OperationCode::fieldTimeline: token = "field-timeline"; break;
        case OperationCode::fieldScoreSample: token = "field-score-sample"; break;
        case OperationCode::fieldMath: token = "field-math"; break;
        case OperationCode::fieldVectorMath: token = "field-vector-math"; break;
        case OperationCode::fieldVectorMeasure: token = "field-vector-measure"; break;
        case OperationCode::fieldCompare: token = "field-compare"; break;
        case OperationCode::fieldBoolean: token = "field-boolean"; break;
        case OperationCode::fieldMapRange: token = "field-map-range"; break;
        case OperationCode::fieldClamp: token = "field-clamp"; break;
        case OperationCode::fieldNoise: token = "field-noise"; break;
        case OperationCode::fieldGradient: token = "field-gradient"; break;
        case OperationCode::fieldRandom: token = "field-random"; break;
        case OperationCode::fieldFalloff: token = "field-falloff"; break;
        case OperationCode::fieldComposeVector: token = "field-compose-vector"; break;
        case OperationCode::fieldIntegerToFloat: token = "field-integer-to-float"; break;
        case OperationCode::fieldFloatToInteger: token = "field-float-to-integer"; break;
        case OperationCode::fieldMix:
            return std::string("visual.geometry.field-mix-") + toWire(op.field.valueType) + ".vertex";
        case OperationCode::fieldSelect:
            return std::string("visual.geometry.field-select-") + toWire(op.field.valueType) + ".vertex";
        default: return {};
    }
    return std::string("visual.geometry.") + token + ".vertex";
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
inline int inputCount(videowire::geometry::OperationCode code)
{
    using C = videowire::geometry::OperationCode;
    if (code == C::fieldConstant) return 0;
    if (code == C::fieldMix || code == C::fieldSelect || code == C::fieldComposeVector) return 3;
    if (code == C::fieldMath || code == C::fieldVectorMath || code == C::fieldVectorMeasure
        || code == C::fieldCompare || code == C::fieldBoolean || code == C::fieldScoreSample) return 2;
    return 1;
}

struct Topology final
{
    std::map<int, std::string> nodes;
    std::set<std::tuple<int, int, int, int>> edges;
    std::map<int, int> outputPorts;
    std::set<int> scoreSources;
};
inline Topology topology(const Binding& binding, const videowire::geometry::ValueDescriptor& value, int inputPort = kInputPort)
{
    using namespace videowire::geometry;
    Topology result;
    for (const auto& op : value.operations)
    {
        const auto id = static_cast<int>(op.stableId - 1);
        result.nodes[id] = nodeKind(op);
        result.outputPorts[id] = inputCount(op.code);
        if (op.code == OperationCode::fieldScoreSample)
        {
            const auto source = static_cast<int>(op.field.values[4] - 1);
            result.nodes[source] = "visual.score.note-collection";
            result.outputPorts[source] = 0;
            result.scoreSources.insert(source);
            result.edges.insert({source, 0, id, 1});
        }
    }
    const auto edge = [&](StableId from, int to, int port)
    {
        const auto source = static_cast<int>(from - 1);
        const auto found = result.outputPorts.find(source);
        if (from && found != result.outputPorts.end()) result.edges.insert({source, found->second, to, port});
    };
    for (const auto& op : value.operations)
    {
        const auto id = static_cast<int>(op.stableId - 1);
        if (op.code != OperationCode::fieldConstant) edge(op.inputStableId, id, 0);
        edge(op.secondaryInputStableId, id, 1);
        edge(op.field.auxiliaryInputStableId, id, 2);
    }
    edge(value.stableId, binding.materialNode, inputPort);
    return result;
}

inline std::optional<videowire::geometry::AdmittedValue> admitBounded(const Binding& binding, std::string& error,
    bool geometryBacked)
{
    using namespace videowire::geometry;
    if (binding.materialNode < 0 || binding.target >= std::size(kTargets)
        || binding.reduction > Reduction::maximum
        || !std::isfinite(binding.gain) || std::abs(binding.gain) > 1000 || binding.fieldPlan.empty())
    { error = "Material Field requires bounded scalar sampling controls"; return std::nullopt; }
    const auto bytes = decodeLoweredPlanText(binding.fieldPlan, error);
    const auto plan = bytes ? decodeLoweredRuntimePlan(*bytes, error) : std::nullopt;
    if (!plan) return std::nullopt;
    const auto value = decodeRuntimeValue(plan->runtimeValue, plan->contract, ResourceLimits{}, {}, error,
        [geometryBacked](const OperationRecord& op, std::string& failure)
        {
            if ((geometryBacked && ((op.retainedMesh && !op.retainedMesh->importedAnimation.empty())
                    || op.code == OperationCode::audioDeformer || op.code == OperationCode::spectrumInstancer))
                || (!geometryBacked && nodeKind(op).empty()))
            { failure = "Material Field rejects unsupported operations before materialization"; return false; }
            return true;
        });
    if (!value) return std::nullopt;
    const auto& d = value->descriptor();
    if (d.carrier != CarrierKind::field || (!geometryBacked && d.fieldDomain != Domain::vertex)
        || d.fieldValueType != ValueType::floatValue || !d.attributes.empty()
        || !d.spectrumFields.empty() || !d.scoreFields.empty() || d.operations.empty()
        || std::get<FieldData>(d.data).elements.empty()
        || (binding.reduction == Reduction::index && binding.index >= std::get<FieldData>(d.data).elements.size())
        || std::any_of(d.operations.begin(), d.operations.end(), [geometryBacked](const auto& op)
            { return (!geometryBacked && nodeKind(op).empty()) || op.stableId > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
                || (op.code == OperationCode::fieldScoreSample && op.field.values[4] > std::numeric_limits<int>::max()); })
        || encodeLoweredPlanText(lowerRuntimePlan(plan->contract, *value)) != binding.fieldPlan)
    { error = geometryBacked ? "Surface Material Field requires a canonical Float field"
        : "Material Field requires a canonical pure vertex Float field"; return std::nullopt; }
    if (geometryBacked) {
        if (std::any_of(d.operations.begin(),d.operations.end(),[&](const auto& op) {
            return op.stableId==static_cast<StableId>(binding.materialNode)+1;
        })) { error="Material Field source and material identities overlap"; return std::nullopt; }
        return value;
    }
    const auto graph = topology(binding, d);
    if (graph.nodes.count(binding.materialNode) || graph.nodes.size() != d.operations.size() + graph.scoreSources.size())
    { error = "Material Field source and material identities overlap"; return std::nullopt; }
    return value;
}

// The field is sized from its input plan. Only a real allocation failure
// refuses it, and the error names the node.
inline std::optional<videowire::geometry::AdmittedValue> admit(const Binding& binding, std::string& error,
    bool geometryBacked = false)
{
    try { return admitBounded(binding, error, geometryBacked); }
    catch (const std::bad_alloc&)
    {
        error = geometryBacked ? "Surface Material Field could not allocate memory for its field"
                               : "Material Field could not allocate memory for its field";
        return std::nullopt;
    }
}

inline bool valid(const Binding& binding,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& material)
{
    std::string error;
    return binding.target < std::size(kTargets)
        && diffractionmaterialbinding::runtimeParameterValues(material).count(kTargets[binding.target]) != 0
        && admit(binding, error).has_value();
}
inline bool same(const Binding& a, const Binding& b)
{
    return a.materialNode == b.materialNode && a.target == b.target && a.reduction == b.reduction
        && a.index == b.index && std::memcmp(&a.gain, &b.gain, sizeof(double)) == 0 && a.fieldPlan == b.fieldPlan;
}

inline bool sample(const Binding& binding, const videowire::geometry::RuntimeFieldEvaluation& runtime,
    double& result, std::string& error, bool geometryBacked = false)
{
    const auto admitted = admit(binding, error, geometryBacked);
    if (!admitted) return false;
    if (!std::isfinite(runtime.timelineSeconds)) { error = "Material Field requires finite rational frame time"; return false; }
    videowire::geometry::ValueDescriptor evaluated;
    try
    {
        if (!videowire::geometry::validateOperationPlan(admitted->descriptor(), {}, nullptr, error, &evaluated, &runtime))
            return false;
    }
    catch (const std::bad_alloc&)
    {
        error = geometryBacked ? "Surface Material Field could not allocate memory for its samples"
                               : "Material Field could not allocate memory for its samples";
        return false;
    }
    const auto& values = std::get<videowire::geometry::FieldData>(evaluated.data).elements;
    if (values.empty() || (binding.reduction == Reduction::index && binding.index >= values.size()))
    { error = "Material Field sample index is past the field's samples"; return false; }
    double reduced = values[binding.reduction == Reduction::index ? binding.index : 0].components[0];
    if (binding.reduction == Reduction::mean)
    {
        // Fixed index order and compensated summation make seeks independent
        // of evaluation history and avoid a backend-dependent parallel sum.
        double sum = 0, compensation = 0;
        for (const auto& value : values)
        {
            const auto next = value.components[0] - compensation;
            const auto total = sum + next;
            compensation = (total - sum) - next;
            sum = total;
        }
        reduced = sum / static_cast<double>(values.size());
    }
    else if (binding.reduction != Reduction::index)
        for (const auto& value : values)
            reduced = binding.reduction == Reduction::minimum ? std::min(reduced, value.components[0])
                : std::max(reduced, value.components[0]);
    result = binding.gain * reduced;
    if (!std::isfinite(result)) { error = "Material Field produced a nonfinite sample"; return false; }
    return true;
}

inline bool evaluate(const Binding& binding, const videowire::geometry::RuntimeFieldEvaluation& runtime,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& material,
    diffractionmaterialbinding::RuntimeParameters& parameters, std::string& error)
{
    double sampled = 0;
    if (!sample(binding, runtime, sampled, error)) return false;
    const auto defaults = diffractionmaterialbinding::runtimeParameterValues(material);
    const std::string name = kTargets[binding.target];
    const auto baseline = defaults.find(name);
    if (baseline == defaults.end()) { error = "Material Field targets an inactive physical parameter"; return false; }
    const auto previous = parameters.find(name);
    const auto result = (previous == parameters.end() ? baseline->second : previous->second) + sampled;
    if (!std::isfinite(result)) { error = "Material Field produced a nonfinite physical parameter"; return false; }
    parameters[name] = result;
    return true;
}
} // namespace materialfield
