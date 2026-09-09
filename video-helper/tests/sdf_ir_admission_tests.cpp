#include "../src/sdf_ir_admission.h"
#include "../src/sdf_scene_modules.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using videowire::SdfIr;
using videowire::SdfOperation;
using videowire::SdfRecord;
using videowire::SdfStableId;
using videohelper::sdf::AdmittedSdfIr;
using videohelper::sdf::SdfAdmissionLimits;

int failures = 0;

void check (bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
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

SdfIr completeFixture()
{
    SdfIr ir;
    ir.rootId = 28;
    ir.records = {
        record (1, SdfOperation::sphere, {}, { 1.0 }),
        record (2, SdfOperation::box, {}, { 1.0, 2.0, 3.0 }),
        record (3, SdfOperation::roundedBox, {}, { 1.0, 1.5, 2.0, 0.2 }),
        record (4, SdfOperation::plane, {}, { 0.0, 1.0, 0.0, -0.5 }),
        record (5, SdfOperation::torus, {}, { 2.0, 0.4 }),
        record (6, SdfOperation::capsule, {}, { 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.25 }),
        record (7, SdfOperation::cylinder, {}, { 0.75, 1.5 }),
        record (8, SdfOperation::cone, {}, { 1.25, 2.0 }),
        record (9, SdfOperation::gyroid, {}, { 3.0, 0.1 }),
        record (10, SdfOperation::unionOp, { 1, 2 }),
        record (11, SdfOperation::intersection, { 10, 3 }),
        record (12, SdfOperation::subtraction, { 11, 4 }),
        record (13, SdfOperation::smoothUnion, { 12, 5 }, { 0.15 }),
        record (14, SdfOperation::smoothIntersection, { 13, 6 }, { 0.2 }),
        record (15, SdfOperation::smoothSubtraction, { 14, 7 }, { 0.25 }),
        record (16, SdfOperation::unionOp, { 15, 8 }),
        record (17, SdfOperation::unionOp, { 16, 9 }),
        record (18, SdfOperation::translate, { 17 }, { 0.0, 1.0, -2.0 }),
        record (19, SdfOperation::rotate, { 18 }, { 0.0, 1.0, 0.0, 0.5 }),
        record (20, SdfOperation::scale, { 19 }, { 1.0, 2.0, -1.0 }),
        record (21, SdfOperation::repeat, { 20 }, { 4.0, 5.0, 6.0 }),
        record (22, SdfOperation::polarRepeat, { 21 }, { 1.0, 8.0, 0.25 }),
        record (23, SdfOperation::mirror, { 22 }, { 1.0, 0.0, 1.0 }),
        record (24, SdfOperation::twist, { 23 }, { 1.0, 0.2 }),
        record (25, SdfOperation::bend, { 24 }, { 0.0, -0.1 }),
        record (26, SdfOperation::taper, { 25 }, { 2.0, 0.05 }),
        record (27, SdfOperation::displacement, { 26 }, { 0.1, 3.0 }),
        record (28, SdfOperation::domainWarp, { 27 }, { 0.1, 0.2, 0.3, 1.0, 2.0, 3.0 })
    };
    return ir;
}

SdfRecord* findRecord (SdfIr& ir, SdfStableId id)
{
    const auto found = std::find_if (ir.records.begin(), ir.records.end(),
        [id] (const SdfRecord& candidate) { return candidate.stableId == id; });
    return found == ir.records.end() ? nullptr : &*found;
}

bool rejected (const SdfIr& ir, SdfAdmissionLimits limits = {})
{
    std::string error;
    return ! videohelper::sdf::admitSdfIr (ir, limits, error).has_value() && ! error.empty();
}

bool rejectedWith (const SdfIr& ir, const std::string& diagnostic)
{
    std::string error;
    return ! videohelper::sdf::admitSdfIr (ir, {}, error).has_value()
        && error.find (diagnostic) != std::string::npos;
}

bool sameBindings (const std::vector<videohelper::sdf::SdfMaterialBinding>& left,
                   const std::vector<videohelper::sdf::SdfMaterialBinding>& right)
{
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index)
        if (left[index].primitiveId != right[index].primitiveId
            || left[index].materialId != right[index].materialId)
            return false;
    return true;
}

bool usesOnly (const AdmittedSdfIr& geometry,
               std::initializer_list<SdfOperation> operations)
{
    for (const auto& recordValue : geometry.records())
        if (std::find (operations.begin(), operations.end(), recordValue.operation)
            == operations.end())
            return false;
    return true;
}

std::string readCatalogSource (const std::string& relativePath)
{
#ifndef SDF_SCENE_SOURCE_ROOT
#error "SDF_SCENE_SOURCE_ROOT must name the checked-in shader-pack root"
#endif
    std::ifstream input (std::string (SDF_SCENE_SOURCE_ROOT) + "/" + relativePath,
                         std::ios::binary);
    return { std::istreambuf_iterator<char> (input), std::istreambuf_iterator<char>() };
}
} // namespace

int main()
{
    static_assert (std::is_same<decltype (std::declval<const AdmittedSdfIr&>().records()),
                                const std::vector<SdfRecord>&>::value,
                   "admitted SDF records must be exposed read-only");
    static_assert (std::is_same<decltype (std::declval<const AdmittedSdfIr&>().structuralDigest()),
                                const std::string&>::value,
                   "the structural digest must be exposed read-only");

    std::set<std::string> tokens;
    for (int raw = static_cast<int> (SdfOperation::sphere);
         raw <= static_cast<int> (SdfOperation::domainWarp); ++raw)
    {
        const auto operation = static_cast<SdfOperation> (raw);
        const auto schema = videowire::sdfOperationSchema (operation);
        check (schema.valid, "every declared SDF operation has a schema");
        check (tokens.emplace (videowire::sdfOperationWireToken (operation)).second,
               "every declared SDF operation has a unique wire token");
    }
    check (tokens.size() == 26, "the contract covers all bounded Phase 10 operations");

    auto ir = completeFixture();
    std::reverse (ir.records.begin(), ir.records.end());
    findRecord (ir, 18)->parameters[0] = -0.0;
    std::string error;
    const auto admitted = videohelper::sdf::admitSdfIr (ir, {}, error);
    check (admitted.has_value() && error.empty(), "the complete backend-neutral fixture is admitted");
    check (admitted && admitted->operationCount() == 28 && admitted->maximumDepth() == 20,
           "admission reports bounded operation count and graph depth");
    check (admitted && admitted->records().front().stableId == 1
                    && admitted->records().back().stableId == 28,
           "admitted records use canonical stable-ID order");
    check (admitted && admitted->structuralDigest().size() == 64,
           "the admitted structure has a SHA-256 digest");

    auto reordered = completeFixture();
    const auto reorderedAdmission = videohelper::sdf::admitSdfIr (reordered, {}, error);
    check (admitted && reorderedAdmission
           && admitted->structuralDigest() == reorderedAdmission->structuralDigest(),
           "record order and negative zero do not change the structural digest");
    check (reorderedAdmission
           && reorderedAdmission->structuralDigest()
                == "2beb7117724b5949ee9b014fc6176424d951814fb694442bd5a6561d676e8f3d",
           "the reference fixture has a stable cross-process digest");

    auto changed = completeFixture();
    std::swap (findRecord (changed, 12)->inputs[0], findRecord (changed, 12)->inputs[1]);
    const auto changedAdmission = videohelper::sdf::admitSdfIr (changed, {}, error);
    check (reorderedAdmission && changedAdmission
           && reorderedAdmission->structuralDigest() != changedAdmission->structuralDigest(),
           "ordered Boolean operands participate in the digest");

    auto sourceMutation = completeFixture();
    const auto immutable = videohelper::sdf::admitSdfIr (sourceMutation, {}, error);
    findRecord (sourceMutation, 1)->parameters[0] = 99.0;
    check (immutable && immutable->records().front().parameters[0] == 1.0,
           "admission owns an immutable copy rather than aliasing producer records");

    auto duplicate = completeFixture();
    duplicate.records.back().stableId = 1;
    check (rejected (duplicate), "duplicate stable IDs are rejected");

    auto missing = completeFixture();
    findRecord (missing, 10)->inputs[0] = 999;
    check (rejected (missing), "missing stable-ID references are rejected");

    SdfIr cycle;
    cycle.rootId = 1;
    cycle.records = {
        record (1, SdfOperation::translate, { 2 }, { 0.0, 0.0, 0.0 }),
        record (2, SdfOperation::translate, { 1 }, { 0.0, 0.0, 0.0 })
    };
    check (rejected (cycle), "cycles are rejected independently of record order");

    auto unreachable = completeFixture();
    unreachable.records.push_back (record (99, SdfOperation::sphere, {}, { 1.0 }));
    check (rejected (unreachable), "records outside the root dependency graph are rejected");

    auto badShape = completeFixture();
    findRecord (badShape, 1)->parameterCount = 0;
    check (rejected (badShape), "operation arity and parameter shape are exact");

    auto hiddenData = completeFixture();
    findRecord (hiddenData, 1)->parameters[7] = 1.0;
    check (rejected (hiddenData), "undeclared parameter slots cannot carry hidden data");

    auto nonFinite = completeFixture();
    findRecord (nonFinite, 1)->parameters[0] = std::numeric_limits<double>::infinity();
    check (rejected (nonFinite), "non-finite parameters are rejected");

    auto outOfRange = completeFixture();
    findRecord (outOfRange, 18)->parameters[0] = 1000001.0;
    check (rejected (outOfRange), "parameter magnitude bounds are enforced");

    auto badSemantic = completeFixture();
    findRecord (badSemantic, 22)->parameters[1] = 8.5;
    check (rejected (badSemantic), "operation-specific parameter bounds are enforced");

    auto badTorus = completeFixture();
    findRecord (badTorus, 5)->parameters[1] = 0.0000009;
    check (rejectedWith (badTorus, "positive major and minor radii"),
           "torus major and minor radii use the shared positive lower bound");

    auto roundedBoxBoundary = completeFixture();
    findRecord (roundedBoxBoundary, 3)->parameters[3] = 1.0;
    check (! rejected (roundedBoxBoundary),
           "rounded-box radius equal to the minimum half extent is admitted");
    auto roundedBoxTooLarge = roundedBoxBoundary;
    findRecord (roundedBoxTooLarge, 3)->parameters[3] = 1.000001;
    check (rejectedWith (roundedBoxTooLarge, "radius exceeds the minimum half extent"),
           "rounded-box radius above the minimum half extent is rejected without normalization");

    auto badCylinder = completeFixture();
    findRecord (badCylinder, 7)->parameters[1] = 1000001.0;
    check (rejectedWith (badCylinder, "parameter is non-finite or out of bounds"),
           "cylinder half height uses the shared absolute upper bound");

    badCylinder = completeFixture();
    findRecord (badCylinder, 7)->parameters[0] = 0.0;
    check (rejectedWith (badCylinder, "positive radius and half height"),
           "cylinder radius and half height fail closed with an exact diagnostic");

    auto badStableId = completeFixture();
    findRecord (badStableId, 1)->stableId = 9007199254740992ull;
    check (rejected (badStableId), "stable IDs outside the exact transport range are rejected");

    SdfAdmissionLimits operationLimit;
    operationLimit.maxOperations = 27;
    check (rejected (completeFixture(), operationLimit), "operation count is bounded");

    SdfAdmissionLimits depthLimit;
    depthLimit.maxDepth = 19;
    check (rejected (completeFixture(), depthLimit), "dependency depth is bounded");

    SdfAdmissionLimits parameterLimit;
    parameterLimit.maxParametersPerRecord = 6;
    check (rejected (completeFixture(), parameterLimit), "per-record parameter count is bounded");

    const auto scoreMonolith = videohelper::sdf::makeScoreMonolith (error);
    const auto repeatedScoreMonolith = videohelper::sdf::makeScoreMonolith (error);
    check (scoreMonolith && repeatedScoreMonolith,
           "Score Monolith produces admitted SDF IR");
    if (scoreMonolith && repeatedScoreMonolith)
    {
        const auto source = readCatalogSource (scoreMonolith->source().sourcePath);
        check (! source.empty()
               && videohelper::sha256Text (source) == scoreMonolith->source().sourceSha256,
               "Score Monolith provenance matches the checked-in shader bytes");
        check (scoreMonolith->source().packId == videohelper::sdf::kRaymarchPackId
               && scoreMonolith->source().programId
                      == videohelper::sdf::kScoreMonolithProgramId,
               "Score Monolith names the exact shipped catalog program");
        check (scoreMonolith->stableId() == "score-monolith-sdf-v1"
               && scoreMonolith->source().conversionScope
                      == videohelper::sdf::SdfSceneConversionScope::exactStaticGeometry,
               "Score Monolith exposes stable identity and exact static-geometry scope");
        check ((scoreMonolith->source().omittedFeatures
                    & videohelper::sdf::omittedScoreModulation) != 0,
               "Score Monolith does not claim its omitted score-reactive shading");
        check (scoreMonolith->geometry().structuralDigest()
                   == repeatedScoreMonolith->geometry().structuralDigest()
               && sameBindings (scoreMonolith->materialBindings(),
                                repeatedScoreMonolith->materialBindings()),
               "Score Monolith conversion is deterministic");
        check (scoreMonolith->geometry().operationCount() == 4
               && scoreMonolith->geometry().maximumDepth() == 3
               && usesOnly (scoreMonolith->geometry(),
                            { SdfOperation::box, SdfOperation::translate,
                              SdfOperation::plane, SdfOperation::unionOp }),
               "Score Monolith converts the shipped box, floor, and union exactly");
        check (scoreMonolith->materialBindings().size() == 2
               && scoreMonolith->materialBindings()[0].primitiveId == 1001
               && scoreMonolith->materialBindings()[1].primitiveId == 1003,
               "Score Monolith binds both source primitives explicitly");
    }

    const auto holoFoil = videohelper::sdf::makeHoloFoil (error);
    const auto repeatedHoloFoil = videohelper::sdf::makeHoloFoil (error);
    check (holoFoil && repeatedHoloFoil, "Holo Foil produces admitted SDF IR");
    if (holoFoil && repeatedHoloFoil)
    {
        const auto source = readCatalogSource (holoFoil->source().sourcePath);
        check (! source.empty()
               && videohelper::sha256Text (source) == holoFoil->source().sourceSha256,
               "Holo Foil provenance matches the checked-in shader bytes");
        check (holoFoil->source().packId == videohelper::sdf::kRaymarchPackId
               && holoFoil->source().programId == videohelper::sdf::kHoloFoilProgramId,
               "Holo Foil names the exact shipped catalog program");
        check (holoFoil->stableId() == "holo-foil-sdf-v1"
               && holoFoil->source().conversionScope
                      == videohelper::sdf::SdfSceneConversionScope::staticBaseGeometryOnly,
               "Holo Foil reports its static-base-only conversion scope");
        check ((holoFoil->source().omittedFeatures
                    & videohelper::sdf::omittedSurfaceDisplacement) != 0,
               "Holo Foil does not claim its omitted animated relief");
        check (holoFoil->geometry().structuralDigest()
                   == repeatedHoloFoil->geometry().structuralDigest()
               && sameBindings (holoFoil->materialBindings(),
                                repeatedHoloFoil->materialBindings()),
               "Holo Foil conversion is deterministic");
        check (holoFoil->geometry().operationCount() == 1
               && holoFoil->geometry().maximumDepth() == 1
               && usesOnly (holoFoil->geometry(), { SdfOperation::roundedBox }),
               "Holo Foil converts only the source rounded-card base");
        check (holoFoil->materialBindings().size() == 1
               && holoFoil->materialBindings()[0].primitiveId == 2001,
               "Holo Foil binds its source primitive explicitly");
    }

    if (reorderedAdmission)
        std::cout << "fixture digest: " << reorderedAdmission->structuralDigest() << '\n';
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "SDF IR admission checks passed\n";
    return EXIT_SUCCESS;
}
