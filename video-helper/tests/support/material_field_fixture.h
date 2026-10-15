#pragma once
#include "../../../shared/MaterialFieldBinding.h"

namespace videohelper::tests
{
inline materialfield::Binding materialFieldFixture(int materialNode, bool score = false, std::uint32_t count = 4)
{
    using namespace videowire::geometry;
    PortContract contract;
    contract.carrier = CarrierKind::field;
    contract.fieldDomain = Domain::vertex;
    contract.fieldValueType = ValueType::floatValue;
    contract.fieldInterpolation = Interpolation::constant;
    contract.maxFieldElements = kMaximumFieldElements;
    FieldOperationSettings constant;
    constant.domain = Domain::vertex; constant.valueType = score ? ValueType::vector : ValueType::floatValue;
    constant.values[4] = count;
    std::string error;
    auto coordinateContract = contract;
    coordinateContract.fieldValueType = constant.valueType;
    const auto coordinate = lowerConstructedField(9001, OperationCode::fieldConstant, constant,
        nullptr, nullptr, nullptr, coordinateContract, error);
    FieldOperationSettings time;
    time.domain = Domain::vertex; time.valueType = ValueType::floatValue;
    time.mode = 2; time.values = {4, 0, 1, 0, 1, 0, 0, 0};
    if (score) { time.mode = 1; time.values = {2, 1, 0, 0.25, 9003, 0, 0, 0}; }
    const auto field = coordinate ? lowerConstructedField(9002, score ? OperationCode::fieldScoreSample : OperationCode::fieldTimeline, time,
        &*coordinate, nullptr, nullptr, contract, error) : std::nullopt;
    const auto admitted = field ? admitValue(*field, contract, {}, error) : std::nullopt;
    materialfield::Binding binding;
    binding.materialNode = materialNode;
    binding.gain = 10;
    if (admitted) binding.fieldPlan = encodeLoweredPlanText(lowerRuntimePlan(contract, *admitted));
    return binding;
}
} // namespace videohelper::tests
