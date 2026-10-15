#include "../src/sdf_ir_admission.h"
#include "../src/sdf_scene_modules.h"
#include "../../shared/SdfRaymarchOperationContract.h"

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

videowire::SdfTextOutline analyticTextOutline (bool letterA = false)
{
    videowire::SdfTextOutline outline;
    outline.sourceText = letterA ? "A" : "O";
    // The source hash oracle uses the existing helper implementation, separately
    // from the new shared codec. These polygon glyphs test field semantics;
    // they do not claim to reproduce the embedded font's actual glyph outlines.
    outline.sourceTextSha256 = videohelper::sha256Text (outline.sourceText);
    outline.embeddedFontToken = "InterRegular_ttf";
    outline.embeddedFontSha256 = "40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82";
    outline.glyphs.push_back ({ letterA ? 36u : 50u, 0, 0.0, 0.0, 4.0, 0.0 });
    if (letterA)
    {
        outline.contours = { { 0, 0, 3, 1 }, { 0, 3, 3, -1 } };
        outline.segments = { {0,0,4,0}, {4,0,2,4}, {2,4,0,0},
                             {1.5,1,2,2}, {2,2,2.5,1}, {2.5,1,1.5,1} };
    }
    else
    {
        outline.contours = { { 0, 0, 4, 1 }, { 0, 4, 4, -1 } };
        outline.segments = { {-2,-2,2,-2}, {2,-2,2,2}, {2,2,-2,2}, {-2,2,-2,-2},
                             {-1,-1,-1,1}, {-1,1,1,1}, {1,1,1,-1}, {1,-1,-1,-1} };
    }
    outline.stableId = videowire::sdfTextOutlineStableId (outline);
    return outline;
}

void textOutlineContractChecks()
{
    using namespace videowire;
    static_assert (static_cast<int> (SdfOperation::domainWarp) == 26);
    static_assert (static_cast<int> (SdfOperation::textExtrusion) == 27);
    static_assert (! sdfOperationSchema (SdfOperation::textExtrusion).valid);
    static_assert (sdfOperationSchema (SdfOperation::textExtrusion, kSdfIrTextSchemaVersion).valid);
    std::string error;
    auto source = analyticTextOutline();
    const auto outline = admitSdfTextOutline (source, error);
    check (outline && error.empty(), "analytic O contours admit with exact source identity");
    check (sdfTextSourceSha256 ("O") == source.sourceTextSha256, "public source hash matches independently computed UTF-8 identity");
    check (outline && outline->identitySha256() == videohelper::sha256Text (sdfTextOutlineIdentityBytes (source)),
           "full resource SHA-256 matches the independent existing hash implementation");
    if (! outline) return;
    const auto sample = [&] (double x, double y, double z, double expected)
    {
        SdfTextWorkBudget budget;
        const auto value = evaluateSdfTextExtrusion (*outline, x, y, z, 2.0, budget);
        return value && std::abs (*value - expected) < 1.0e-12
            && budget.recordVisits == 1 && budget.segmentVisits == 8;
    };
    check (sample (0, 0, 0, 1), "O counter stays outside by its analytic distance to the inner square");
    check (sample (1.5, 0, 0, -0.5), "O stroke is inside by the nearest contour distance");
    check (sample (3, 0, 0, 1), "O external side distance uses the outer contour");
    check (sample (1.5, 0, 1, 0) && sample (1.5, 0, -1, 0),
           "front and back zero sets use full extrusion depth symmetrically");
    check (sample (3, 0, 2, std::sqrt (2.0)) && sample (3, 0, -2, std::sqrt (2.0)),
           "O outside XY/Z corners use independent Euclidean distance to the cap edge");
    check (sample (1.5, 0, 1.25, 0.25), "extrusion outside Z is an analytic cap distance");
    check (sample (1, 1, 0, 0) && sample (2, 2, 0, 0), "contour vertices are zero under half-open winding crossings");
    const auto a = admitSdfTextOutline (analyticTextOutline (true), error);
    SdfTextWorkBudget aBudget;
    const auto counter = a ? evaluateSdfTextExtrusion (*a, 2, 1.5, 0, 2, aBudget) : std::nullopt;
    // Inner sides have slope +/-2. Distance at (2,1.5) is 0.5/sqrt(5).
    check (counter && std::abs (*counter - 0.5 / std::sqrt (5.0)) < 1.0e-12,
           "A triangular counter has the independent analytic oblique-edge distance");
    const auto aCorner = a ? evaluateSdfTextExtrusion (*a, 2, 1.5, 2, 2, aBudget) : std::nullopt;
    check (aCorner && std::abs (*aCorner - std::sqrt (1.05)) < 1.0e-12,
           "A counter above the cap uses independent Euclidean distance to the oblique counter edge");
    const auto stroke = a ? evaluateSdfTextExtrusion (*a, 2, 0.5, 0, 2, aBudget) : std::nullopt;
    check (stroke && std::abs (*stroke + 0.5) < 1.0e-12, "A base stroke remains filled below its counter");
    source.segments[0].x0 = 999;
    check (outline->outline().segments[0].x0 == -2, "admitted outline owns its immutable copy");
    auto bad = analyticTextOutline();
    bad.segments.back().x1 = 0;
    check (! admitSdfTextOutline (bad, error), "open contours refuse before evaluation");
    bad = analyticTextOutline(); bad.contours[1].winding = 1;
    check (! admitSdfTextOutline (bad, error), "counter winding must match signed contour area");
    bad = analyticTextOutline(); bad.segments[0].x0 = std::numeric_limits<double>::quiet_NaN();
    check (! admitSdfTextOutline (bad, error), "non-finite outline coordinates refuse");
    bad = analyticTextOutline(); bad.segments[0].x1 = bad.segments[0].x0; bad.segments[0].y1 = bad.segments[0].y0;
    check (! admitSdfTextOutline (bad, error), "zero-length vector segments refuse");
    bad = analyticTextOutline(); bad.glyphs[0].glyphId = 0;
    check (! admitSdfTextOutline (bad, error), "missing glyph IDs refuse rather than using a font fallback");
    bad = analyticTextOutline(); bad.embeddedFontToken = "SystemSans";
    check (! admitSdfTextOutline (bad, error), "system fonts are outside the embedded source contract");
    bad = analyticTextOutline(); bad.embeddedFontSha256[0] = '0';
    check (! admitSdfTextOutline (bad, error), "embedded font bytes must match the pinned source hash");
    bad = analyticTextOutline(); bad.shapingRevision += 1;
    check (! admitSdfTextOutline (bad, error), "unknown shaping revisions refuse");
    bad = analyticTextOutline(); bad.sourceText = std::string ("\xc0\xaf", 2);
    check (! admitSdfTextOutline (bad, error), "overlong UTF-8 source text refuses");
    bad = analyticTextOutline(); bad.sourceText = "A";
    check (! admitSdfTextOutline (bad, error), "exact source text SHA-256 is checked before admission");
    bad = analyticTextOutline(); bad.sourceText = std::string (257, 'O');
    check (! admitSdfTextOutline (bad, error), "source text is bounded before identity allocation");
    bad = analyticTextOutline(); bad.segments.resize (kSdfTextMaximumSegments + 1);
    check (sdfTextOutlineIdentityBytes (bad).empty() && ! admitSdfTextOutline (bad, error),
           "oversized segment resources refuse before codec allocation");
    bad = analyticTextOutline(); bad.contours[0].segmentCount = std::numeric_limits<std::uint32_t>::max();
    check (! admitSdfTextOutline (bad, error), "hostile contour ranges cannot wrap or access segments");
    auto encodedOutline = encodeSdfTextOutline (analyticTextOutline());
    SdfTextOutline decodedOutline;
    check (decodeSdfTextOutline (encodedOutline, decodedOutline)
           && encodeSdfTextOutline (decodedOutline) == encodedOutline, "outline roundtrip retains all source and vector bytes");
    const auto originalId = decodedOutline.stableId;
    check (! decodeSdfTextOutline (encodedOutline + "x", decodedOutline) && decodedOutline.stableId == originalId,
           "outline decoder rejects trailing bytes without replacing destination");
    // Offset: ID/version, four length-prefixed strings, revisions, emSize.
    const std::size_t glyphCountOffset = 8 + 4 + 16 + 1 + 64 + 16 + 64 + 8 + 8;
    auto hostile = encodedOutline;
    for (std::size_t i = 0; i < 4; ++i) hostile[glyphCountOffset + i] = static_cast<char> (0xff);
    check (! decodeSdfTextOutline (hostile, decodedOutline), "hostile glyph count refuses before resize");
    SdfTextWorkBudget bounded; bounded.maximumSegmentVisits = 7;
    check (! evaluateSdfTextExtrusion (*outline, 0, 0, 0, 2, bounded)
           && bounded.recordVisits == 0 && bounded.segmentVisits == 0, "failed segment work admission is atomic");
    bounded = {};
    check (! evaluateSdfTextExtrusion (*outline, 0, 0, 0, 0, bounded)
           && ! evaluateSdfTextExtrusion (*outline, 0, 0, 0, std::numeric_limits<double>::infinity(), bounded),
           "degenerate and non-finite extrusion depths refuse");

    SdfRaymarchOperation request;
    request.terminalStableId = 10;
    request.geometry.schemaVersion = kSdfIrTextSchemaVersion;
    request.geometry.rootId = 1;
    auto text = record (1, SdfOperation::textExtrusion, {}, { 2 });
    request.geometry.textOutlines = { analyticTextOutline() };
    text.resourceStableId = request.geometry.textOutlines[0].stableId;
    request.geometry.records = { text };
    const auto wire = encodeSdfRaymarchOperation (request);
    SdfRaymarchOperation decoded;
    check (! wire.empty() && wire.find ("sdf-raymarch-v2:") == 0
           && decodeSdfRaymarchOperation (wire, decoded) && encodeSdfRaymarchOperation (decoded) == wire,
           "resource-bearing v2 raymarch roundtrip preserves typed text resource and depth");
    if (wire.empty()) return;
    check (rejected (request.geometry), "legacy native IR admission does not advertise text support");
    check (! decodeSdfRaymarchOperation (wire + "00", decoded)
           && ! decodeSdfRaymarchOperation (wire.substr (0, wire.size() - 2), decoded),
           "v2 trailing and truncated payloads refuse without partial destination replacement");
    auto hostileWire = wire;
    const auto resourceCountHexOffset = kSdfRaymarchTextOperationPrefix.size() + (52 + 100) * 2;
    hostileWire.replace (resourceCountHexOffset, 8, "ffffffff");
    check (! decodeSdfRaymarchOperation (hostileWire, decoded), "hostile v2 resource count refuses before reserve");
    hostileWire = wire;
    hostileWire.replace (resourceCountHexOffset + 8, 8, "ffffffff");
    check (! decodeSdfRaymarchOperation (hostileWire, decoded), "hostile v2 resource byte count refuses before allocation");
    auto aggregate = request.geometry;
    auto firstResource = analyticTextOutline();
    auto secondResource = analyticTextOutline (true);
    firstResource.sourceText = std::string (129, 'O');
    firstResource.sourceTextSha256 = videohelper::sha256Text (firstResource.sourceText);
    firstResource.stableId = sdfTextOutlineStableId (firstResource);
    secondResource.sourceText = std::string (129, 'A');
    secondResource.sourceTextSha256 = videohelper::sha256Text (secondResource.sourceText);
    secondResource.stableId = sdfTextOutlineStableId (secondResource);
    aggregate.textOutlines = { firstResource, secondResource };
    std::sort (aggregate.textOutlines.begin(), aggregate.textOutlines.end(),
               [] (const SdfTextOutline& left, const SdfTextOutline& right) { return left.stableId < right.stableId; });
    check (! validateSdfTextIrResources (aggregate, error)
           && error == "SDF text graph source, glyph, payload or segment capacity exceeded", "source byte budget applies across every outline in the graph");
    firstResource = analyticTextOutline(); secondResource = analyticTextOutline (true);
    firstResource.glyphs.resize (65, firstResource.glyphs.front());
    secondResource.glyphs.resize (65, secondResource.glyphs.front());
    firstResource.stableId = sdfTextOutlineStableId (firstResource);
    secondResource.stableId = sdfTextOutlineStableId (secondResource);
    aggregate.textOutlines = { firstResource, secondResource };
    std::sort (aggregate.textOutlines.begin(), aggregate.textOutlines.end(),
               [] (const SdfTextOutline& left, const SdfTextOutline& right) { return left.stableId < right.stableId; });
    check (! validateSdfTextIrResources (aggregate, error)
           && error == "SDF text graph source, glyph, payload or segment capacity exceeded", "glyph budget applies across every outline in the graph");
    SdfTextWorkBudget graphBudget;
    request.geometry.records.push_back (record (2, SdfOperation::unionOp, { 1, 1 }));
    request.geometry.rootId = 2;
    check (chargeSdfTextIrEvaluationWork (request.geometry, 4, graphBudget, error)
           && graphBudget.recordVisits == 12 && graphBudget.segmentVisits == 64,
           "repeated DAG edges charge record and segment visits for every retained/raymarch query");
    graphBudget = {}; graphBudget.maximumSegmentVisits = 63;
    check (! chargeSdfTextIrEvaluationWork (request.geometry, 4, graphBudget, error)
           && graphBudget.recordVisits == 0 && graphBudget.segmentVisits == 0, "graph work refuses atomically at segment budget");
    graphBudget = {};
    check (! chargeSdfTextIrEvaluationWork (request.geometry, std::numeric_limits<std::uint64_t>::max(), graphBudget, error),
           "query multiplication is bounded before overflow");
    request.geometry.records.back().inputs[1] = 2;
    check (encodeSdfRaymarchOperation (request).empty(), "cyclic v2 graphs refuse wire admission");
    request.geometry.records.back().inputs[1] = 1;
    request.geometry.records[0].resourceStableId = 42;
    check (encodeSdfRaymarchOperation (request).empty(), "unresolved text resource references refuse wire admission");
    request.geometry = {}; request.geometry.rootId = 1;
    request.geometry.records = { record (1, SdfOperation::sphere, {}, { 1 }) };
    const auto legacy = encodeSdfRaymarchOperation (request);
    // Independent fixed legacy request bytes, terminal=10, root=1, sphere r=1.
    const std::string golden = "sdf-raymarch-v1:010000000a0000000000000080000000fca9f1d24d62503f0000000000005940010101000100000001000000000000000100000001000000000000000100010000000000000000000000000000000000000000000000f03f0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    check (legacy == golden && decodeSdfRaymarchOperation (golden, decoded)
           && decoded.geometry.textOutlines.empty(), "v1 golden wire bytes and decoding remain unchanged");
    request.geometry.records[0] = text;
    check (encodeSdfRaymarchOperation (request).empty(), "text cannot be encoded in a resource-free v1 envelope");
}
} // namespace

int main()
{
    textOutlineContractChecks();
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
