#include "../src/path_mask_geometry.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace videowire::pathmask;
using namespace videohelper::pathmask;

static_assert(std::is_copy_constructible_v<AdmittedPathMask>);
static_assert(!std::is_copy_assignable_v<AdmittedPathMask>);
static_assert(!std::is_move_assignable_v<AdmittedPathMask>);
static_assert(std::is_same_v<decltype(std::declval<const AdmittedPathMask&>().contours()),
                             const std::vector<Contour>&>);
static_assert(std::is_same_v<decltype(std::declval<const AdmittedPathMask&>().segments()),
                             const std::vector<Segment>&>);
static_assert(!kAllowsCpuProductionImageRendering);
static_assert(!kAuthorizesGraphBypass);

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (condition)
        return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

Segment line(double x, double y)
{
    Segment result;
    result.kind = SegmentKind::line;
    result.end = { x, y };
    return result;
}

Description square(FillRule rule = FillRule::nonZero)
{
    Description description;
    description.assetId = 41;
    description.revision = 7;
    description.fillRule = rule;
    description.contours = { { 101, { 0.2, 0.2 }, 0, 3, true } };
    description.segments = {
        line(0.8, 0.2),
        line(0.8, 0.8),
        line(0.2, 0.8)
    };
    return description;
}

std::optional<AdmittedPathMask> admitted(const Description& description,
                                         Limits limits = {})
{
    AdmissionFailure failure = AdmissionFailure::unsupportedVersion;
    auto result = admit(description, limits, failure);
    check(result.has_value() && failure == AdmissionFailure::none,
          "valid path-mask fixture is admitted");
    return result;
}

void expectRejected(const Description& description,
                    AdmissionFailure expected,
                    const char* message,
                    Limits limits = {})
{
    AdmissionFailure failure = AdmissionFailure::none;
    const auto result = admit(description, limits, failure);
    check(!result, message);
    check(failure == expected, "rejection reports the exact path-mask contract failure");
    check(!token(failure).empty(), "path-mask rejection has a stable token");
}

GeometrySample sample(const AdmittedPathMask& asset, Point point)
{
    EvaluationFailure failure = EvaluationFailure::invalidCoordinate;
    const auto result = evaluate(asset, point, failure);
    check(result.has_value() && failure == EvaluationFailure::none,
          "valid normalized geometry sample succeeds");
    return result.value_or(GeometrySample {});
}

void testAdmissionAndFootprint()
{
    auto source = square();
    source.feather = -0.0;
    source.choke = -0.0;
    const auto asset = admitted(source);
    if (!asset)
        return;

    check(asset->assetId() == 41 && asset->revision() == 7,
          "stable asset identity and immutable revision survive admission");
    check(asset->feather() == 0.0 && !std::signbit(asset->feather())
              && asset->choke() == 0.0 && !std::signbit(asset->choke()),
          "signed zero scalar parameters are canonicalized");
    check(asset->contours().size() == 1 && asset->segments().size() == 3,
          "admission preserves exact bounded contour and segment records");

    const auto footprint = asset->footprint();
    check(footprint.logicalPointCount == 4,
          "memory accounting counts one contour start and three line endpoints");
    check(footprint.flattenedLineCount == 4,
          "memory accounting includes the one implicit closing line");
    check(footprint.canonicalScalarBytes == 45 + 33 + 3 * 49,
          "canonical scalar payload byte accounting is exact");
    check(footprint.loweredLineBytes == 4 * sizeof(FlattenedLine),
          "lowered geometry byte accounting is exact");
    check(footprint.admittedValueBytes == sizeof(AdmittedPathMask)
              + asset->contours().capacity() * sizeof(Contour)
              + asset->segments().capacity() * sizeof(Segment),
          "admitted value storage reports exact owned element capacity");

    source.assetId = 999;
    source.contours[0].start = { 0.0, 0.0 };
    source.segments.clear();
    check(asset->assetId() == 41 && asset->contours()[0].start.x == 0.2
              && asset->segments().size() == 3,
          "admission owns immutable values independent of the source description");

    Description reopened;
    reopened.assetId = asset->assetId();
    reopened.revision = asset->revision();
    reopened.fillRule = asset->fillRule();
    reopened.feather = asset->feather();
    reopened.choke = asset->choke();
    reopened.contours = asset->contours();
    reopened.segments = asset->segments();
    const auto reopenedAsset = admitted(reopened);
    check(reopenedAsset && reopenedAsset->footprint().canonicalScalarBytes
                               == asset->footprint().canonicalScalarBytes,
          "scalar save and reopen reconstruction retains the exact contract");
    if (reopenedAsset)
    {
        const auto original = sample(*asset, { 0.35, 0.55 });
        const auto restored = sample(*reopenedAsset, { 0.35, 0.55 });
        check(original.inside == restored.inside
                  && original.windingNumber == restored.windingNumber
                  && original.signedDistance == restored.signedDistance
                  && original.coverage == restored.coverage,
              "save and reopen reconstruction evaluates identically");
    }
}

void testClosedAndOpenSemantics()
{
    auto description = square();
    description.contours.push_back({ 202, { 0.0, 0.5 }, 3, 1, false });
    description.segments.push_back(line(1.0, 0.5));
    const auto asset = admitted(description);
    if (!asset)
        return;

    std::vector<FlattenedLine> lines;
    visitFlattenedLines(*asset, [&](const FlattenedLine& value) { lines.push_back(value); });
    check(lines.size() == 5 && lines[3].implicitClosingLine
              && lines[3].contourId == 101,
          "closed contours receive exactly one implicit closing line");
    check(!lines[4].contourClosed && !lines[4].implicitClosingLine
              && !lines[4].contributesToFill && lines[4].contourId == 202,
          "open contours remain open and do not contribute to fill");

    const auto center = sample(*asset, { 0.5, 0.5 });
    const auto outsideOnOpenLine = sample(*asset, { 0.1, 0.5 });
    check(center.inside && center.windingNumber == 1 && center.coverage == 1.0,
          "closed square fills its center under nonzero winding");
    check(!outsideOnOpenLine.inside && !outsideOnOpenLine.onBoundary
              && outsideOnOpenLine.coverage == 0.0,
          "an open contour never creates fill or a mask boundary");

    const auto boundary = sample(*asset, { 0.2, 0.5 });
    check(boundary.inside && boundary.onBoundary && boundary.signedDistance == 0.0,
          "closed contour boundary membership is deterministic");
}

void testFillRules()
{
    auto description = square(FillRule::nonZero);
    description.contours.push_back({ 102, { 0.3, 0.3 }, 3, 3, true });
    description.segments.push_back(line(0.7, 0.3));
    description.segments.push_back(line(0.7, 0.7));
    description.segments.push_back(line(0.3, 0.7));

    const auto nonZero = admitted(description);
    description.fillRule = FillRule::evenOdd;
    const auto evenOdd = admitted(description);
    if (!nonZero || !evenOdd)
        return;

    const auto nonZeroCenter = sample(*nonZero, { 0.5, 0.5 });
    const auto evenOddCenter = sample(*evenOdd, { 0.5, 0.5 });
    check(nonZeroCenter.inside && nonZeroCenter.windingNumber == 2,
          "nonzero fill retains same-direction nested contours");
    check(!evenOddCenter.inside && evenOddCenter.windingNumber == 2,
          "even-odd fill removes a same-direction nested region");
}

void testCurveEvaluationAndDeterminism()
{
    Description description;
    description.assetId = 88;
    description.revision = 3;
    description.contours = { { 301, { 0.1, 0.1 }, 0, 2, true } };
    Segment quadratic;
    quadratic.kind = SegmentKind::quadratic;
    quadratic.control1 = { 0.5, 0.9 };
    quadratic.end = { 0.9, 0.1 };
    Segment cubic;
    cubic.kind = SegmentKind::cubic;
    cubic.control1 = { 0.8, 0.4 };
    cubic.control2 = { 0.2, 0.4 };
    cubic.end = { 0.2, 0.2 };
    description.segments = { quadratic, cubic };

    const auto asset = admitted(description);
    if (!asset)
        return;
    check(asset->footprint().logicalPointCount == 1 + 2 + 3,
          "quadratic and cubic controls count against the point limit");
    check(asset->footprint().flattenedLineCount
              == kQuadraticLineSegments + kCubicLineSegments + 1,
          "curve lowering uses fixed subdivision counts plus implicit closure");

    std::vector<FlattenedLine> lines;
    visitFlattenedLines(*asset, [&](const FlattenedLine& value) { lines.push_back(value); });
    bool continuous = !lines.empty() && lines.front().from.x == 0.1
        && lines.front().from.y == 0.1 && lines.back().implicitClosingLine;
    for (std::size_t index = 1; continuous && index < lines.size(); ++index)
        continuous = lines[index - 1].to.x == lines[index].from.x
            && lines[index - 1].to.y == lines[index].from.y;
    check(continuous, "fixed curve lowering is continuous in authored segment order");

    const auto baseline = sample(*asset, { 0.45, 0.3 });
    bool repeatedEqual = true;
    for (int repeat = 0; repeat < 1000 && repeatedEqual; ++repeat)
    {
        const auto current = sample(*asset, { 0.45, 0.3 });
        repeatedEqual = current.inside == baseline.inside
            && current.onBoundary == baseline.onBoundary
            && current.windingNumber == baseline.windingNumber
            && current.signedDistance == baseline.signedDistance
            && current.coverage == baseline.coverage;
    }
    check(repeatedEqual, "repeated geometry samples are bit-identical");
}

void testFeatherAndChoke()
{
    auto description = square();
    description.feather = 0.2;
    const auto feathered = admitted(description);
    if (!feathered)
        return;

    const auto boundary = sample(*feathered, { 0.2, 0.5 });
    const auto deepInside = sample(*feathered, { 0.5, 0.5 });
    const auto farOutside = sample(*feathered, { 0.05, 0.5 });
    check(boundary.coverage == 0.5,
          "feather transition is centered at the exact mask boundary");
    check(deepInside.coverage == 1.0 && farOutside.coverage == 0.0,
          "feather coverage clamps outside its bounded transition");

    description.feather = 0.0;
    description.choke = 0.05;
    const auto contracted = admitted(description);
    description.choke = -0.05;
    const auto expanded = admitted(description);
    if (!contracted || !expanded)
        return;
    check(sample(*contracted, { 0.22, 0.5 }).coverage == 0.0,
          "positive choke contracts the filled mask");
    check(sample(*expanded, { 0.18, 0.5 }).coverage == 1.0,
          "negative choke expands the filled mask");
}

void testFailClosedAdmission()
{
    auto value = square();
    value.schemaVersion = 2;
    expectRejected(value, AdmissionFailure::unsupportedVersion,
                   "unknown schema versions fail closed");

    value = square();
    value.assetId = 0;
    expectRejected(value, AdmissionFailure::invalidAssetIdentity,
                   "zero asset identity is rejected");
    value = square();
    value.revision = kMaximumScalarIdentity + 1;
    expectRejected(value, AdmissionFailure::invalidRevision,
                   "revision outside exact scalar range is rejected");
    value = square();
    value.fillRule = static_cast<FillRule>(99);
    expectRejected(value, AdmissionFailure::invalidFillRule,
                   "unknown fill rules fail closed");
    value = square();
    value.feather = std::numeric_limits<double>::quiet_NaN();
    expectRejected(value, AdmissionFailure::invalidFeather,
                   "nonfinite feather is rejected");
    value = square();
    value.choke = kMaximumAbsoluteChoke + 0.001;
    expectRejected(value, AdmissionFailure::invalidChoke,
                   "out-of-range choke is rejected");

    value = square();
    value.contours.clear();
    expectRejected(value, AdmissionFailure::emptyAsset,
                   "empty contour data is rejected");
    value = square();
    value.contours.resize(kMaximumContours + 1, value.contours.front());
    expectRejected(value, AdmissionFailure::contourLimitExceeded,
                   "contour budget is checked before traversal");
    value = square();
    value.segments.resize(kMaximumSegments + 1, value.segments.front());
    expectRejected(value, AdmissionFailure::segmentLimitExceeded,
                   "segment budget is checked before traversal");

    value = square();
    value.contours[0].id = 0;
    expectRejected(value, AdmissionFailure::invalidContourIdentity,
                   "zero contour identity is rejected");
    value = square();
    value.contours.push_back(value.contours.front());
    value.contours[1].firstSegment = 3;
    value.contours[1].segmentCount = 1;
    value.contours[1].closed = false;
    value.segments.push_back(line(0.5, 0.5));
    expectRejected(value, AdmissionFailure::duplicateContourIdentity,
                   "duplicate stable contour identities are rejected");
    value = square();
    value.contours[0].firstSegment = 1;
    expectRejected(value, AdmissionFailure::noncanonicalSegmentRange,
                   "gapped segment ranges are rejected");
    value = square();
    value.contours[0].segmentCount = 2;
    expectRejected(value, AdmissionFailure::noncanonicalSegmentRange,
                   "unclaimed trailing segments are rejected");
    value = square();
    value.contours[0].segmentCount = 0;
    expectRejected(value, AdmissionFailure::emptyContour,
                   "empty contours are rejected");
    value = square();
    value.contours[0].segmentCount = 1;
    value.segments.resize(1);
    expectRejected(value, AdmissionFailure::closedContourTooShort,
                   "closed contours require three implicit-edge vertices");
    value = square();
    value.contours[0].closed = false;
    expectRejected(value, AdmissionFailure::noClosedContour,
                   "open-only data cannot masquerade as a fill mask");

    value = square();
    value.contours[0].start.x = -0.001;
    expectRejected(value, AdmissionFailure::invalidCoordinate,
                   "coordinates outside normalized space are rejected");
    value = square();
    value.segments[0].end.y = std::numeric_limits<double>::infinity();
    expectRejected(value, AdmissionFailure::invalidCoordinate,
                   "nonfinite coordinates are rejected");
    value = square();
    value.segments[0].kind = static_cast<SegmentKind>(77);
    expectRejected(value, AdmissionFailure::invalidSegmentKind,
                   "unknown segment kinds fail closed");
    value = square();
    value.segments[0].control1 = { 0.5, 0.5 };
    expectRejected(value, AdmissionFailure::noncanonicalUnusedControl,
                   "line segments cannot retain hidden control points");
    value = square();
    value.segments.back().end = value.contours.front().start;
    expectRejected(value, AdmissionFailure::explicitClosureForbidden,
                   "closed contours cannot duplicate the implicit closing point");

    value = square();
    Limits lowPointLimit;
    lowPointLimit.maximumLogicalPoints = 3;
    expectRejected(value, AdmissionFailure::pointLimitExceeded,
                   "logical point budget includes contour starts and endpoints",
                   lowPointLimit);
    value = square();
    Limits invalidLimits;
    invalidLimits.maximumSegments = kMaximumSegments + 1;
    expectRejected(value, AdmissionFailure::invalidLimits,
                   "caller limits cannot weaken hard segment bounds",
                   invalidLimits);
}

void testFailClosedEvaluation()
{
    const auto asset = admitted(square());
    if (!asset)
        return;
    EvaluationFailure failure = EvaluationFailure::none;
    const auto nonfinite = evaluate(*asset,
                                    { std::numeric_limits<double>::infinity(), 0.5 },
                                    failure);
    check(!nonfinite && failure == EvaluationFailure::invalidCoordinate
              && token(failure) == "invalidCoordinate",
          "nonfinite evaluation coordinates fail closed with a stable token");
    const auto outside = evaluate(*asset, { 1.01, 0.5 }, failure);
    check(!outside && failure == EvaluationFailure::invalidCoordinate,
          "evaluation coordinates outside normalized space fail closed");
}
} // namespace

int main()
{
    testAdmissionAndFootprint();
    testClosedAndOpenSemantics();
    testFillRules();
    testCurveEvaluationAndDeterminism();
    testFeatherAndChoke();
    testFailClosedAdmission();
    testFailClosedEvaluation();

    if (failures != 0)
    {
        std::cerr << failures << " path-mask contract checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "path-mask contract checks passed\n";
    return EXIT_SUCCESS;
}
