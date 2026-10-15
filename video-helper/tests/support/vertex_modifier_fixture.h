#pragma once

#include "../../../shared/VertexModifierIr.h"

#include <algorithm>
#include <initializer_list>

namespace videohelper::test
{
inline void appendVertexModifierRecord(videowire::VertexModifierIr& ir, videowire::VertexModifierOperation operation,
                                       std::initializer_list<std::uint64_t> inputs,
                                       std::initializer_list<double> parameters)
{
    const auto schema = videowire::vertexModifierOperationSchema(operation);
    videowire::VertexModifierRecord record;
    record.stableId = ir.records.size() + 1u;
    record.operation = operation;
    record.resultType = schema.resultType;
    record.inputCount = schema.inputCount;
    record.parameterCount = schema.parameterCount;
    std::copy(inputs.begin(), inputs.end(), record.inputs.begin());
    std::copy(parameters.begin(), parameters.end(), record.parameters.begin());
    ir.records.push_back(record);
}

// position + (rate * timeSeconds, 0, 0). It reads only position and time, so
// its prior evaluation over a prior deformed pose is exact.
inline videowire::VertexModifierIr timeAxisDisplacement(double rate = 0.6, double bound = 10.0)
{
    using Operation = videowire::VertexModifierOperation;
    videowire::VertexModifierIr ir;
    const auto append = [&ir](Operation operation, std::initializer_list<std::uint64_t> inputs,
                              std::initializer_list<double> parameters)
    { appendVertexModifierRecord(ir, operation, inputs, parameters); };
    append(Operation::importedPosition, {}, {});
    append(Operation::timeSeconds, {}, {});
    append(Operation::scalarConstant, {}, {rate});
    append(Operation::scalarMultiply, {2, 3}, {});
    append(Operation::scalarConstant, {}, {0.0});
    append(Operation::vec3Compose, {4, 5, 5}, {});
    append(Operation::boundedDisplacementOutput, {1, 6}, {bound});
    ir.rootId = 7;
    return ir;
}

inline videowire::VertexModifierIr audioNormalDisplacement(double gain = 0.75, double bound = 0.25)
{
    using Operation = videowire::VertexModifierOperation;
    videowire::VertexModifierIr ir;
    const auto append = [&ir](Operation operation, std::initializer_list<std::uint64_t> inputs,
                              std::initializer_list<double> parameters)
    { appendVertexModifierRecord(ir, operation, inputs, parameters); };
    append(Operation::importedPosition, {}, {});
    append(Operation::importedNormal, {}, {});
    append(Operation::audioParameter, {}, {6.0});
    append(Operation::scalarConstant, {}, {gain});
    append(Operation::scalarMultiply, {3, 4}, {});
    append(Operation::vec3Scale, {2, 5}, {});
    append(Operation::boundedDisplacementOutput, {1, 6}, {bound});
    ir.rootId = 7;
    return ir;
}
} // namespace videohelper::test
