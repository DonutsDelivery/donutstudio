#pragma once

#include "gpu_backend/backend.h"
#include "sdf_ir_admission.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace videohelper::sdf
{
inline std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> compileNativeSdfProgram (
    const videowire::SdfIr& source, std::string& error)
{
    SdfAdmissionLimits limits;
    limits.maxOperations = arbitgpu::kNativeSdfMaximumRecords;
    limits.maxDepth = arbitgpu::kNativeSdfMaximumDepth;
    auto admitted = admitSdfIr (source, limits, error);
    if (! admitted) return {};

    std::shared_ptr<arbitgpu::NativeSdfCompiledProgram> program (
        new arbitgpu::NativeSdfCompiledProgram());
    program->maximumDepth_ = static_cast<std::uint32_t> (admitted->maximumDepth());
    program->structuralDigest_ = admitted->structuralDigest();
    program->records_.reserve (admitted->records().size());

    std::unordered_map<videowire::SdfStableId, std::uint32_t> indices;
    indices.reserve (admitted->records().size());
    for (std::size_t index = 0; index < admitted->records().size(); ++index)
        indices.emplace (admitted->records()[index].stableId, static_cast<std::uint32_t> (index));

    for (const auto& sourceRecord : admitted->records())
    {
        arbitgpu::NativeSdfCompiledRecord record;
        record.operation = static_cast<std::uint32_t> (sourceRecord.operation);
        record.parameterCount = sourceRecord.parameterCount;
        if (sourceRecord.inputCount > 0) record.input0 = indices.at (sourceRecord.inputs[0]);
        if (sourceRecord.inputCount > 1) record.input1 = indices.at (sourceRecord.inputs[1]);
        for (std::size_t parameter = 0; parameter < sourceRecord.parameterCount; ++parameter)
        {
            const auto value = static_cast<float> (sourceRecord.parameters[parameter]);
            if (! std::isfinite (value))
            {
                error = "native GPU SDF parameter cannot be represented as a finite float";
                return {};
            }
            record.parameters[parameter] = value;
        }
        program->records_.push_back (record);
    }

    std::vector<std::size_t> expandedSteps (program->records_.size(), 0);
    std::function<std::size_t (std::size_t)> countSteps = [&] (std::size_t index)
    {
        if (expandedSteps[index] != 0) return expandedSteps[index];
        const auto& record = program->records_[index];
        const auto inputs = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (record.operation)).inputCount;
        std::size_t steps = 1;
        const auto addChild = [&] (std::uint32_t child)
        {
            const auto childSteps = countSteps (child);
            if (childSteps > arbitgpu::kNativeSdfMaximumEvaluationSteps - steps)
                steps = arbitgpu::kNativeSdfMaximumEvaluationSteps + 1;
            else
                steps += childSteps;
        };
        if (inputs > 0) addChild (record.input0);
        if (inputs > 1 && steps <= arbitgpu::kNativeSdfMaximumEvaluationSteps)
            addChild (record.input1);
        expandedSteps[index] = steps;
        return steps;
    };
    program->rootIndex_ = indices.at (admitted->rootId());
    const auto evaluationSteps = countSteps (program->rootIndex_);
    if (evaluationSteps > arbitgpu::kNativeSdfMaximumEvaluationSteps)
    {
        error = "native GPU SDF expanded evaluation capacity exceeded";
        return {};
    }
    program->evaluationSteps_ = static_cast<std::uint32_t> (evaluationSteps);
    error.clear();
    return program;
}

inline bool validateNativeSdfProgram (
    const arbitgpu::NativeSdfCompiledProgram& program, std::string& error)
{
    if (program.records().empty() || program.records().size() > arbitgpu::kNativeSdfMaximumRecords
        || program.rootIndex() >= program.records().size()
        || program.maximumDepth() > arbitgpu::kNativeSdfMaximumDepth
        || program.evaluationSteps() == 0
        || program.evaluationSteps() > arbitgpu::kNativeSdfMaximumEvaluationSteps
        || program.structuralDigest().empty())
    {
        error = "native GPU SDF compiled payload is invalid";
        return false;
    }
    for (std::size_t index = 0; index < program.records().size(); ++index)
    {
        const auto& record = program.records()[index];
        if (record.operation < static_cast<std::uint32_t> (videowire::SdfOperation::sphere)
            || record.operation > static_cast<std::uint32_t> (videowire::SdfOperation::domainWarp))
        {
            error = "native GPU SDF compiled payload has an invalid operation";
            return false;
        }
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (record.operation));
        if (record.parameterCount != schema.parameterCount
            || (schema.inputCount > 0 && record.input0 >= index)
            || (schema.inputCount > 1 && record.input1 >= index)
            || (schema.inputCount == 0
                && (record.input0 != std::numeric_limits<std::uint32_t>::max()
                    || record.input1 != std::numeric_limits<std::uint32_t>::max()))
            || (schema.inputCount == 1
                && record.input1 != std::numeric_limits<std::uint32_t>::max()))
        {
            error = "native GPU SDF compiled payload has an invalid record";
            return false;
        }
        for (std::size_t parameter = 0; parameter < record.parameterCount; ++parameter)
        {
            if (! std::isfinite (record.parameters[parameter]))
            {
                error = "native GPU SDF compiled payload has a non-finite parameter";
                return false;
            }
        }
        for (std::size_t parameter = record.parameterCount;
             parameter < record.parameters.size(); ++parameter)
            if (record.parameters[parameter] != 0.0f)
            {
                error = "native GPU SDF compiled payload has an active unused parameter";
                return false;
            }
        if (static_cast<videowire::SdfOperation> (record.operation)
                == videowire::SdfOperation::roundedBox
            && record.parameters[3] > std::min ({ record.parameters[0], record.parameters[1],
                                                   record.parameters[2] }))
        {
            error = "native GPU SDF rounded-box radius exceeds the minimum half extent";
            return false;
        }
    }
    std::vector<bool> reachable (program.records().size(), false);
    std::function<void (std::uint32_t)> visit = [&] (std::uint32_t index)
    {
        if (reachable[index]) return;
        reachable[index] = true;
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (program.records()[index].operation));
        if (schema.inputCount > 0) visit (program.records()[index].input0);
        if (schema.inputCount > 1) visit (program.records()[index].input1);
    };
    visit (program.rootIndex());
    if (std::find (reachable.begin(), reachable.end(), false) != reachable.end())
    {
        error = "native GPU SDF compiled payload has unreachable records";
        return false;
    }
    error.clear();
    return true;
}

// The compiled type is factory-only. Re-admit the untrusted source and bind it
// to the factory result by the canonical structural digest before GPU use.
inline bool validateNativeSdfProgram (
    const arbitgpu::NativeSdfCompiledProgram& program,
    const videowire::SdfIr& source,
    std::string& error)
{
    if (! validateNativeSdfProgram (program, error))
        return false;
    SdfAdmissionLimits limits;
    limits.maxOperations = arbitgpu::kNativeSdfMaximumRecords;
    limits.maxDepth = arbitgpu::kNativeSdfMaximumDepth;
    const auto admitted = admitSdfIr (source, limits, error);
    if (! admitted
        || program.structuralDigest() != admitted->structuralDigest()
        || program.maximumDepth() != admitted->maximumDepth()
        || program.records().size() != admitted->records().size())
    {
        if (error.empty())
            error = "native GPU SDF compiled payload does not match admitted geometry";
        return false;
    }
    std::unordered_map<videowire::SdfStableId, std::uint32_t> indices;
    for (std::size_t index = 0; index < admitted->records().size(); ++index)
        indices.emplace (admitted->records()[index].stableId, static_cast<std::uint32_t> (index));
    std::vector<std::size_t> steps (program.records().size(), 0);
    std::function<std::size_t (std::uint32_t)> countSteps = [&] (std::uint32_t index)
    {
        if (steps[index] != 0) return steps[index];
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (program.records()[index].operation));
        std::size_t result = 1;
        if (schema.inputCount > 0) result += countSteps (program.records()[index].input0);
        if (schema.inputCount > 1) result += countSteps (program.records()[index].input1);
        return steps[index] = result;
    };
    for (std::size_t index = 0; index < admitted->records().size(); ++index)
    {
        const auto& sourceRecord = admitted->records()[index];
        const auto& record = program.records()[index];
        if (record.operation != static_cast<std::uint32_t> (sourceRecord.operation)
            || record.parameterCount != sourceRecord.parameterCount
            || (sourceRecord.inputCount > 0 && record.input0 != indices.at (sourceRecord.inputs[0]))
            || (sourceRecord.inputCount > 1 && record.input1 != indices.at (sourceRecord.inputs[1])))
        {
            error = "native GPU SDF compiled payload records do not match admitted geometry";
            return false;
        }
        for (std::size_t parameter = 0; parameter < sourceRecord.parameterCount; ++parameter)
            if (record.parameters[parameter] != static_cast<float> (sourceRecord.parameters[parameter]))
            {
                error = "native GPU SDF compiled payload values do not match admitted geometry";
                return false;
            }
    }
    if (program.rootIndex() != indices.at (admitted->rootId())
        || program.evaluationSteps() != countSteps (program.rootIndex()))
    {
        error = "native GPU SDF compiled payload metadata does not match admitted geometry";
        return false;
    }
    error.clear();
    return true;
}
} // namespace videohelper::sdf
