#include "../src/sdf_scene_modules.h"
#include "support/sdf_reference_evaluator.h"
#include "support/sdf_output_test_scenes.h"
#include "../src/sdf_native_program.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using videohelper::sdf::AdmittedSdfIr;
using videohelper::sdf::SdfAdmissionLimits;
using videohelper::sdf::reference::Point3;
using videowire::SdfIr;
using videowire::SdfOperation;
using videowire::SdfRecord;
using videowire::SdfStableId;

int failures = 0;

void check (bool condition, const std::string& message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

SdfRecord record (SdfStableId id, SdfOperation operation,
                  std::initializer_list<SdfStableId> inputs = {},
                  std::initializer_list<double> parameters = {})
{
    SdfRecord value;
    value.stableId = id;
    value.operation = operation;
    value.inputCount = static_cast<std::uint8_t> (inputs.size());
    value.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), value.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), value.parameters.begin());
    return value;
}

std::vector<double> parametersFor (SdfOperation operation)
{
    switch (operation)
    {
        case SdfOperation::sphere: return { 1.15 };
        case SdfOperation::box: return { 0.7, 0.4, 0.9 };
        case SdfOperation::roundedBox: return { 0.8, 0.6, 1.0, 0.18 };
        case SdfOperation::plane: return { 0.25, 1.0, -0.5, 0.2 };
        case SdfOperation::torus: return { 0.95, 0.22 };
        case SdfOperation::capsule: return { -0.4, -0.7, 0.2, 0.6, 0.8, -0.3, 0.19 };
        case SdfOperation::cylinder: return { 0.62, 0.85 };
        case SdfOperation::cone: return { 0.72, 1.1 };
        case SdfOperation::gyroid: return { 2.4, 0.12 };
        case SdfOperation::smoothUnion:
        case SdfOperation::smoothIntersection:
        case SdfOperation::smoothSubtraction: return { 0.35 };
        case SdfOperation::translate: return { 0.25, -0.35, 0.45 };
        case SdfOperation::rotate: return { 1.0, 2.0, -0.5, 0.7 };
        case SdfOperation::scale: return { -1.4, 0.75, 1.8 };
        case SdfOperation::repeat: return { 1.7, 2.1, 2.5 };
        case SdfOperation::polarRepeat: return { 1.0, 5.0, 0.23 };
        case SdfOperation::mirror: return { 1.0, 0.0, 1.0 };
        case SdfOperation::twist: return { 2.0, 0.65 };
        case SdfOperation::bend: return { 0.0, -0.45 };
        case SdfOperation::taper: return { 1.0, 0.3 };
        case SdfOperation::displacement: return { 0.17, 2.3 };
        case SdfOperation::domainWarp: return { 0.2, -0.15, 0.1, 1.2, 1.7, 2.1 };
        case SdfOperation::unionOp:
        case SdfOperation::intersection:
        case SdfOperation::subtraction: return {};
    }
    return {};
}

SdfRecord makeRecord (SdfStableId id, SdfOperation operation,
                      std::initializer_list<SdfStableId> inputs,
                      const std::vector<double>& parameters)
{
    SdfRecord value;
    value.stableId = id;
    value.operation = operation;
    value.inputCount = static_cast<std::uint8_t> (inputs.size());
    value.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), value.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), value.parameters.begin());
    return value;
}

AdmittedSdfIr fixtureFor (SdfOperation operation)
{
    const auto schema = videowire::sdfOperationSchema (operation);
    SdfIr ir;
    if (schema.inputCount == 0)
    {
        ir.rootId = 1;
        ir.records.clear();
        ir.records.push_back (makeRecord (1, operation, {}, parametersFor (operation)));
    }
    else if (schema.inputCount == 2)
    {
        ir.rootId = 3;
        ir.records.clear();
        ir.records.push_back (record (1, SdfOperation::sphere, {}, { 0.9 }));
        ir.records.push_back (record (2, SdfOperation::box, {}, { 0.65, 0.45, 0.8 }));
        ir.records.push_back (makeRecord (3, operation, { 1, 2 }, parametersFor (operation)));
    }
    else
    {
        ir.rootId = 3;
        ir.records.clear();
        ir.records.push_back (record (1, SdfOperation::sphere, {}, { 0.72 }));
        ir.records.push_back (record (2, SdfOperation::translate, { 1 }, { 0.25, -0.35, 0.45 }));
        ir.records.push_back (makeRecord (3, operation, { 2 }, parametersFor (operation)));
        if (operation == SdfOperation::translate)
        {
            ir.rootId = 2;
            ir.records.resize (2);
        }
    }
    std::string error;
    auto admitted = videohelper::sdf::admitSdfIr (ir, SdfAdmissionLimits {}, error);
    if (! admitted) throw std::runtime_error ("golden fixture admission failed: " + error);
    return std::move (*admitted);
}

struct Golden final
{
    SdfOperation operation;
    Point3 point;
    double expected;
};

constexpr Point3 operationPoint { 0.83, -0.57, 1.11 };

const std::array<Golden, 26> operationGoldens {{
    { SdfOperation::sphere, operationPoint, 0.34863271017284281 },
    { SdfOperation::box, operationPoint, 0.29983328701129902 },
    { SdfOperation::roundedBox, operationPoint, 0.20820097887563338 },
    { SdfOperation::plane, operationPoint, -0.60085965716609202 },
    { SdfOperation::torus, operationPoint, 0.49763309448400272 },
    { SdfOperation::capsule, operationPoint, 1.2553620406567445 },
    { SdfOperation::cylinder, operationPoint, 0.76600144300069195 },
    { SdfOperation::cone, operationPoint, 0.7978165613872292 },
    { SdfOperation::gyroid, operationPoint, 0.012260299898138705 },
    { SdfOperation::unionOp, operationPoint, 0.378021163428716 },
    { SdfOperation::intersection, operationPoint, 0.5986327101728427 },
    { SdfOperation::subtraction, operationPoint, 0.5986327101728427 },
    { SdfOperation::smoothUnion, operationPoint, 0.36606304068875367 },
    { SdfOperation::smoothIntersection, operationPoint, 0.61059083291280503 },
    { SdfOperation::smoothSubtraction, { -0.775, 0.0, 0.0 }, -0.0375 },
    { SdfOperation::translate, operationPoint, 0.1857593499379403 },
    { SdfOperation::rotate, operationPoint, 0.37649795635290628 },
    { SdfOperation::scale, operationPoint, 0.17399288640485397 },
    { SdfOperation::repeat, { 2.13, -0.57, 1.11 }, -0.0013902310711326749 },
    { SdfOperation::polarRepeat, operationPoint, 0.18214413091459181 },
    { SdfOperation::mirror, { -0.83, -0.57, -1.11 }, 0.1857593499379403 },
    { SdfOperation::twist, operationPoint, 0.1897936515390598 },
    { SdfOperation::bend, operationPoint, 0.15665102969980205 },
    { SdfOperation::taper, operationPoint, 0.41655618470887767 },
    { SdfOperation::displacement, operationPoint, 0.09971127042651895 },
    { SdfOperation::domainWarp, operationPoint, 0.23528932114751133 }
}};

bool close (double actual, double expected)
{
    return std::abs (actual - expected) <= 1.0e-12;
}
} // namespace

int main()
{
    std::set<SdfOperation> covered;
    for (const auto& golden : operationGoldens)
    {
        const auto geometry = fixtureFor (golden.operation);
        const auto actual = videohelper::sdf::reference::evaluatePoint (geometry, golden.point);
        covered.insert (golden.operation);
        check (close (actual, golden.expected),
               std::string (videowire::sdfOperationWireToken (golden.operation))
                   + " point-distance golden changed");
        const auto sample = videohelper::sdf::reference::evaluateSample (geometry, golden.point);
        check (close (sample.distance, golden.expected)
                   && std::any_of (geometry.records().begin(), geometry.records().end(),
                       [&] (const auto& r) { return r.stableId == sample.primitiveId && r.inputCount == 0; }),
               "all 26 operations retain an actual primitive contributor and unchanged distance");
    }
    check (covered.size() == 26, "goldens cover every Phase 10 operation exactly once");
    const auto subtraction = fixtureFor (SdfOperation::subtraction);
    check (close (videohelper::sdf::reference::evaluatePoint (
                      subtraction, { 0.0, 0.0, 0.0 }),
                  0.45),
          "subtraction oracle removes the overlapping box from the sphere");
    const auto cone = fixtureFor (SdfOperation::cone);
    check (close (videohelper::sdf::reference::evaluatePoint (
                      cone, { 0.36, 0.0, 0.0 }),
                  0.0),
          "cone oracle places the side surface between the apex and base rim");
    check (videohelper::sdf::reference::evaluatePoint (cone, { 0.0, 0.0, 0.0 }) < 0.0
            && videohelper::sdf::reference::evaluatePoint (cone, { 0.5, 0.0, 0.0 }) > 0.0
            && close (videohelper::sdf::reference::evaluatePoint (cone, { 0.0, 1.1, 0.0 }), 0.0)
            && close (videohelper::sdf::reference::evaluatePoint (cone, { 0.72, -1.1, 0.0 }), 0.0)
            && close (videohelper::sdf::reference::evaluatePoint (cone, { 0.36, 0.0, 0.0 }), 0.0),
          "cone base radius and half height agree at inside, outside, apex, base, and midpoint");

    videowire::SdfIr roundedBox;
    roundedBox.rootId = 1;
    roundedBox.records.push_back (
        record (1, SdfOperation::roundedBox, {}, { 1.0, 0.8, 0.9, 0.8 }));
    std::string roundedBoxError;
    const auto admittedRoundedBox = videohelper::sdf::admitSdfIr (roundedBox, {}, roundedBoxError);
    check (admittedRoundedBox.has_value()
            && videohelper::sdf::reference::evaluatePoint (*admittedRoundedBox, { 0.0, 0.0, 0.0 }) < 0.0,
          "rounded-box equality boundary keeps the origin inside");
    roundedBox.records[0].parameters[3] = 0.800001;
    check (! videohelper::sdf::admitSdfIr (roundedBox, {}, roundedBoxError).has_value(),
           "rounded-box malformed radius fails before the CPU oracle evaluates it");

    videowire::SdfIr composite;
    composite.rootId = 5;
    composite.records.push_back (record (1, SdfOperation::sphere, {}, { 0.55 }));
    composite.records.push_back (
        makeRecord (2, SdfOperation::translate, { 1 }, { -0.7, 0.0, 0.0 }));
    composite.records.push_back (record (3, SdfOperation::box, {}, { 0.38, 0.5, 0.45 }));
    composite.records.push_back (
        makeRecord (4, SdfOperation::translate, { 3 }, { 0.7, 0.0, 0.0 }));
    composite.records.push_back (
        makeRecord (5, SdfOperation::smoothUnion, { 2, 4 }, { 0.18 }));
    std::string compositeError;
    const auto admittedComposite = videohelper::sdf::admitSdfIr (composite, {}, compositeError);
    check (admittedComposite.has_value()
            && close (videohelper::sdf::reference::evaluatePoint (
                          *admittedComposite, { -0.7, 0.0, 0.0 }),
                      -0.55)
            && close (videohelper::sdf::reference::evaluatePoint (
                          *admittedComposite, { 0.7, 0.0, 0.0 }),
                      -0.38)
            && close (videohelper::sdf::reference::evaluatePoint (
                          *admittedComposite, { 0.0, 0.0, 0.0 }),
                      0.1498611111111111),
          "transformed smooth union oracle keeps both lobes and their gap");

    std::string error;
    const auto score = videohelper::sdf::makeScoreMonolith (error);
    const auto foil = videohelper::sdf::makeHoloFoil (error);
    check (score.has_value() && foil.has_value(), "scene-module goldens use admitted source-bound IR");
    if (score && foil)
    {
        const std::array<std::pair<Point3, double>, 3> scoreGoldens {{
            { { 0.0, 1.5, 0.0 }, -0.16 },
            { { 1.2, 1.5, 0.0 }, 0.14999999999999991 },
            { { 0.0, 0.4, 0.5 }, 0.33999999999999997 }
        }};
        const std::array<std::pair<Point3, double>, 3> foilGoldens {{
            { { 0.0, 0.0, 0.0 }, -0.054999999999999993 },
            { { 1.3, 0.0, 0.0 }, 0.12000000000000005 },
            { { 0.7, 1.7, 0.2 }, 0.1862985702402731 }
        }};
        for (const auto& golden : scoreGoldens)
        {
            const auto actual = videohelper::sdf::reference::evaluatePoint (score->geometry(), golden.first);
            check (close (actual, golden.second), "Score Monolith point-distance golden changed");
        }
        for (const auto& golden : foilGoldens)
        {
            const auto actual = videohelper::sdf::reference::evaluatePoint (foil->geometry(), golden.first);
            check (close (actual, golden.second), "Holo Foil point-distance golden changed");
        }
    }

    const auto gyroid = fixtureFor (SdfOperation::gyroid);
    const auto period = 2.0 * 3.141592653589793238462643383279502884 / 2.4;
    const auto first = videohelper::sdf::reference::evaluatePoint (gyroid, operationPoint);
    const auto repeated = videohelper::sdf::reference::evaluatePoint (
        gyroid, { operationPoint.x + period, operationPoint.y, operationPoint.z });
    check (close (first, repeated),
           "gyroid cellScale is the normative reciprocal cell scale on a 2*pi/cellScale period");

    using namespace videohelper::sdf;
    using namespace videohelper::sdf::reference;
    using namespace videohelper::sdf::test;
    NativeSdfRenderControls controls;
    controls.maximumDistance = 12;
    controls.shadowQuality = arbitgpu::NativeSdfQuality::ultra;
    const auto plane = outputTestGeometry (OutputScene::Plane,error);
    const auto sphere = outputTestGeometry (OutputScene::Sphere,error);
    const auto corner = outputTestGeometry (OutputScene::Corner,error);
    const auto blocked = outputTestGeometry (OutputScene::OccludedPlane,error);
    const auto carved = outputTestGeometry (OutputScene::CarvedBox,error);
    const auto pair = outputTestGeometry (OutputScene::Pair,error);
    const auto cube = outputTestGeometry (OutputScene::Box,error);
    check (plane != nullptr, "plane oracle scene admits");
    check (sphere != nullptr, "sphere oracle scene admits");
    check (corner != nullptr, "corner oracle scene admits");
    check (blocked != nullptr, "occluded plane oracle scene admits");
    check (carved != nullptr, "carved box oracle scene admits");
    check (pair != nullptr, "pair oracle scene admits");
    check (cube != nullptr, "box oracle scene admits");
    if (!plane || !sphere || !corner || !blocked || !carved || !pair || !cube) return EXIT_FAILURE;
    const auto flat = evaluateSurface (*plane,{0,0,0},controls);
    const auto convex = evaluateSurface (*sphere,{0,0,1},controls);
    const auto concave = evaluateSurface (*carved,{0,0,0.1},controls);
    check (std::abs (flat.meanCurvature) < 1.0e-10
               && std::abs (convex.meanCurvature - 1.0) < 0.002
               && std::abs (concave.meanCurvature + 2.0) < 0.005,
           "mean curvature is zero on a plane, +1/R on a sphere and -1/R in a spherical cut");
    check (flat.ambientVisibility > 0.999
               && evaluateSurface (*corner,{0,0,0},controls).ambientVisibility < 0.85
               && evaluateSurface (*corner,{-1,0,0},controls).ambientVisibility > 0.999,
           "AO darkens the nearby corner and returns to open-plane visibility beyond its radius");
    check (flat.lightVisibility > 0.999
               && evaluateSurface (*blocked,{0,0,0},controls).lightVisibility < 0.01
               && evaluateSurface (*blocked,{1,0,0},controls).lightVisibility > 0.999
               && evaluateSurface (*sphere,{0,0,-1},controls).lightVisibility == 0.0,
           "directional shadow follows a separate occluder while an unblocked plane remains lit");
    check (evaluateSample (*pair,{-0.65,0,0.45}).primitiveId == kLeftPrimitive
               && evaluateSample (*pair,{0.65,0,0.45}).primitiveId == kRightPrimitive
               && evaluateSample (*carved,{0,0,0.1}).primitiveId == kLeftPrimitive
               && evaluateSample (*carved,{0.7,0,0.6}).primitiveId == 111,
           "translated lobes and subtraction cuts retain the winning primitive's full identity");
    check (nativeSdfMaterialColorCode (17) == 0x3c71beu
               && nativeSdfMaterialColorCode (kLeftPrimitive) == 0x307c62u
               && nativeSdfMaterialColorCode (kRightPrimitive) == 0x30a274u
               && nativeSdfMaterialColorCode (UINT64_MAX) == 0xebbd7bu,
           "fixed material visualization goldens include high ID bits and the uint64 maximum");
    const auto compiled = compileNativeSdfProgram (outputTestSource (OutputScene::Pair),error);
    check (compiled && validateNativeSdfProgram (*compiled,outputTestSource (OutputScene::Pair),error)
               && std::any_of (compiled->records().begin(),compiled->records().end(),
                   [] (const auto& r) { return r.stableId == kRightPrimitive; }),
           "native compiled records preserve full primitive identities alongside source admission");
    if (compiled)
    {
        auto changed = outputTestSource (OutputScene::Pair);
        changed.records[0].stableId += 1;
        changed.records[1].inputs[0] += 1;
        check (!validateNativeSdfProgram (*compiled,changed,error), "compiled identity cannot be reused for renamed geometry");
    }
    // Analytic silhouette locations are independent of the raymarch oracle.
    controls.output = arbitgpu::NativeSdfOutput::edgeDistance;
    const auto centerEdge = evaluateOutputPixel (*sphere,controls,33,33,16,16)[0]*8.0;
    const auto nearEdge = evaluateOutputPixel (*sphere,controls,33,33,26,16)[0]*8.0;
    const double projectedSphereRadius = 16.5*1.8/std::sqrt (8.0);
    check (centerEdge == 8.0 && nearEdge >= projectedSphereRadius-10.0
               && nearEdge <= projectedSphereRadius-10.0+0.6,
           "sphere silhouette distance is in pixels with bounded half-pixel refinement");
    check (evaluateOutputPixel (*cube,controls,33,33,16,16)[0] > 0.99
               && evaluateOutputPixel (*cube,controls,33,33,25,16)[0] < 0.2
               && evaluateOutputPixel (*plane,controls,33,33,0,0)[0] == 1.0,
           "edge distance varies over a flat box face and does not invent a viewport edge on an infinite plane");
    check (evaluateOutputPixel (*sphere,controls,17,17,8,8)[0] < 0.8,
           "resizing changes silhouette pixel distance rather than reusing normalized depth");
    // All hard and smooth Boolean variants have deterministic ownership.
    for (auto op : {SdfOperation::unionOp,SdfOperation::intersection,SdfOperation::subtraction,
                    SdfOperation::smoothUnion,SdfOperation::smoothIntersection,SdfOperation::smoothSubtraction})
    {
        SdfIr source;
        source.rootId = 3;
        source.records = {record (kLeftPrimitive,SdfOperation::sphere,{}, {1}),
                          record (kRightPrimitive,SdfOperation::sphere,{}, {0.5})};
        const auto isSmooth = static_cast<unsigned> (op) >= static_cast<unsigned> (SdfOperation::smoothUnion);
        auto operation = record (3,op,{kLeftPrimitive,kRightPrimitive},{});
        if (isSmooth) { operation.parameterCount = 1; operation.parameters[0] = 0.1; }
        source.records.push_back (operation);
        const auto admitted = admitSdfIr (source,{},error);
        const bool isUnion = op == SdfOperation::unionOp || op == SdfOperation::smoothUnion;
        check (admitted && evaluateSample (*admitted,{0,0,0.5}).primitiveId
                   == (isUnion ? kLeftPrimitive : kRightPrimitive),
               "hard and smooth Boolean ownership agrees with the dominant signed distance");
        source.records[1].parameters[0] = 1;
        const auto tied = admitSdfIr (source,{},error);
        check (tied && evaluateSample (*tied,{0,0,1}).primitiveId == kLeftPrimitive,
               "ordered A wins an equal-weight tie in every hard and smooth Boolean");
        std::swap (source.records.back().inputs[0],source.records.back().inputs[1]);
        const auto reversedTie = admitSdfIr (source,{},error);
        check (reversedTie && evaluateSample (*reversedTie,{0,0,1}).primitiveId == kRightPrimitive,
               "reversing Boolean inputs reverses tied ownership without sorting primitive IDs");
        // Distances differ by less than the smooth radius. The larger blend
        // weight must select a leaf even when both leaves affect the field.
        std::swap (source.records.back().inputs[0],source.records.back().inputs[1]);
        source.records[1].parameters[0] = 0.96;
        const auto blended = admitSdfIr (source,{},error);
        const bool cut = op == SdfOperation::subtraction || op == SdfOperation::smoothSubtraction;
        check (blended && evaluateSample (*blended,{0,0,0.99}).primitiveId
                   == (isUnion || cut ? kLeftPrimitive : kRightPrimitive)
                   && evaluateSample (*blended,{0,0,0.97}).primitiveId
                   == (isUnion ? kLeftPrimitive : kRightPrimitive),
               "Boolean contributors follow dominant weights inside the smooth blend band");
    }

    for (const auto invalidOutput : {static_cast<arbitgpu::NativeSdfOutput> (-1),
                                     arbitgpu::NativeSdfOutput::count,
                                     static_cast<arbitgpu::NativeSdfOutput> (9)})
    {
        auto invalid = controls;
        invalid.output = invalidOutput;
        bool rejected = false;
        try { evaluateOutputPixel (*sphere,invalid,33,33,16,16); }
        catch (const std::invalid_argument&) { rejected = true; }
        check (rejected,"the scalar output oracle rejects invalid output selections");
    }
    for (unsigned badControl = 0; badControl < 3; ++badControl)
    {
        auto invalid = controls;
        if (badControl == 0) invalid.normalQuality = static_cast<arbitgpu::NativeSdfQuality> (-1);
        if (badControl == 1) invalid.shadowQuality = arbitgpu::NativeSdfQuality::count;
        if (badControl == 2) invalid.epsilon = std::numeric_limits<double>::quiet_NaN();
        bool rejected = false;
        try { evaluateSurface (*sphere,{0,0,1},invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        check (rejected,"the scalar surface oracle bounds quality indexing and rejects nonfinite controls");
    }

    if (failures != 0) return EXIT_FAILURE;
    std::cout << "SDF reference oracle checks passed\n";
    return EXIT_SUCCESS;
}
