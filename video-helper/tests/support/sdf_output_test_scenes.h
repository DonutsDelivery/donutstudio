#pragma once

#include "../../src/sdf_ir_admission.h"

#include <algorithm>
#include <initializer_list>
#include <memory>

namespace videohelper::sdf::test
{
enum class OutputScene
{
    Plane, Sphere, Box, Corner, OccludedPlane, Pair, Union, Intersection,
    SmoothUnion, SmoothIntersection, CarvedBox, SmoothCarvedBox
};
inline constexpr videowire::SdfStableId kLeftPrimitive = UINT64_C(0x1000000000000011);
inline constexpr videowire::SdfStableId kRightPrimitive = UINT64_C(0x2000000000000011);

inline videowire::SdfIr outputTestSource (OutputScene scene)
{
    using Op = videowire::SdfOperation;
    using Id = videowire::SdfStableId;
    const auto record = [] (Id id, Op op, std::initializer_list<Id> inputs,
                           std::initializer_list<double> parameters)
    {
        videowire::SdfRecord r;
        r.stableId = id; r.operation = op;
        r.inputCount = static_cast<std::uint8_t> (inputs.size());
        r.parameterCount = static_cast<std::uint8_t> (parameters.size());
        std::copy (inputs.begin(), inputs.end(), r.inputs.begin());
        std::copy (parameters.begin(), parameters.end(), r.parameters.begin());
        return r;
    };
    videowire::SdfIr source;
    source.rootId = 17;
    source.records = {record (17, Op::plane, {}, {0,0,1,0})};
    switch (scene)
    {
        case OutputScene::Plane: break;
        case OutputScene::Sphere:
            source.records = {record (17, Op::sphere, {}, {1})};
            break;
        case OutputScene::Box:
            source.records = {record (17, Op::box, {}, {0.9,0.9,0.25})};
            break;
        case OutputScene::Corner:
            source.rootId = 33;
            source.records.push_back (record (31, Op::plane, {}, {-1,0,0,0.15}));
            source.records.push_back (record (33, Op::unionOp, {17,31}, {}));
            break;
        case OutputScene::OccludedPlane:
            source.rootId = 33;
            source.records.push_back (record (31, Op::sphere, {}, {0.2}));
            source.records.push_back (record (32, Op::translate, {31}, {-0.45,0.75,0.6}));
            source.records.push_back (record (33, Op::unionOp, {17,32}, {}));
            break;
        case OutputScene::Pair:
        case OutputScene::Union:
        case OutputScene::Intersection:
        case OutputScene::SmoothUnion:
        case OutputScene::SmoothIntersection:
        {
            const bool separated = scene == OutputScene::Pair;
            const auto radius = separated ? 0.45 : 0.7;
            const auto offset = separated ? 0.65 : 0.35;
            const auto operation = scene == OutputScene::Intersection ? Op::intersection
                : scene == OutputScene::SmoothUnion ? Op::smoothUnion
                : scene == OutputScene::SmoothIntersection ? Op::smoothIntersection : Op::unionOp;
            source.rootId = 33;
            source.records = {
                record (kLeftPrimitive, Op::sphere, {}, {radius}),
                record (31, Op::translate, {kLeftPrimitive}, {-offset,0,0}),
                record (kRightPrimitive, Op::sphere, {}, {radius}),
                record (32, Op::translate, {kRightPrimitive}, {offset,0,0}),
                record (33, operation, {31,32}, {})};
            if (scene == OutputScene::SmoothUnion || scene == OutputScene::SmoothIntersection)
            {
                source.records.back().parameterCount = 1;
                source.records.back().parameters[0] = 0.2;
            }
            break;
        }
        case OutputScene::CarvedBox:
        case OutputScene::SmoothCarvedBox:
            source.rootId = 141;
            source.records = {
                record (111, Op::box, {}, {0.75,0.6,0.6}),
                record (kLeftPrimitive, Op::sphere, {}, {0.5}),
                record (131, Op::translate, {kLeftPrimitive}, {0,0,0.6}),
                record (141, Op::subtraction, {111,131}, {})};
            if (scene == OutputScene::SmoothCarvedBox)
            {
                source.records.back().operation = Op::smoothSubtraction;
                source.records.back().parameterCount = 1;
                source.records.back().parameters[0] = 0.2;
            }
            break;
    }
    return source;
}

inline std::shared_ptr<const AdmittedSdfIr> outputTestGeometry (OutputScene scene, std::string& error)
{
    auto admitted = admitSdfIr (outputTestSource (scene), {}, error);
    return admitted ? std::make_shared<const AdmittedSdfIr> (std::move (*admitted)) : nullptr;
}
} // namespace videohelper::sdf::test
