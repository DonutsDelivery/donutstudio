#include "../src/sdf_native_program.h"
#include "../src/sdf_native_render_admission.h"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace
{
int failures = 0;
void check (bool value, const char* message)
{
    if (! value) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

videowire::SdfTextOutline rectangleOutline (const char* text = "O", bool counter = true)
{
    videowire::SdfTextOutline outline;
    outline.sourceText = text;
    outline.sourceTextSha256 = videowire::sdfTextSourceSha256 (text);
    outline.embeddedFontToken = "InterRegular_ttf";
    outline.embeddedFontSha256 = videowire::sdfTextEmbeddedFontSha256 (outline.embeddedFontToken);
    outline.glyphs.push_back ({ 1, 0, 0, 0, 2, 0 });
    // Analytic rectangles are vector-consumer fixtures, not font-origin proof.
    outline.contours.push_back ({ 0, 0, 4, 1 });
    outline.segments = { { -1,-1,1,-1 }, { 1,-1,1,1 }, { 1,1,-1,1 }, { -1,1,-1,-1 } };
    if (counter)
    {
        outline.contours.push_back ({ 0, 4, 4, -1 });
        const videowire::SdfTextSegment hole[] {
            { -.5,-.5,-.5,.5 }, { -.5,.5,.5,.5 }, { .5,.5,.5,-.5 }, { .5,-.5,-.5,-.5 }
        };
        outline.segments.insert (outline.segments.end(), std::begin (hole), std::end (hole));
    }
    outline.stableId = videowire::sdfTextOutlineStableId (outline);
    return outline;
}

videowire::SdfIr textIr (videowire::SdfTextOutline outline = rectangleOutline())
{
    videowire::SdfIr source;
    source.schemaVersion = videowire::kSdfIrTextSchemaVersion;
    source.rootId = 41;
    videowire::SdfRecord text;
    text.stableId = source.rootId;
    text.operation = videowire::SdfOperation::textExtrusion;
    text.resourceStableId = outline.stableId;
    text.parameterCount = 1;
    text.parameters[0] = .5;
    source.records.push_back (text);
    source.textOutlines.push_back (std::move (outline));
    return source;
}

// Consumer-table exercise only, never a product renderer. The independent
// expected values below come from closed-form rectangle distances.
double packedDistance (const arbitgpu::NativeSdfCompiledProgram& program,
                       double x, double y, double z)
{
    const auto& header = program.textResources()[program.textResourceIndices()[program.rootIndex()]];
    double distance = std::numeric_limits<double>::infinity();
    int winding = 0;
    for (std::size_t i = header.firstSegment; i < header.firstSegment + header.segmentCount; ++i)
    {
        const auto& edge = program.textSegments()[i];
        const double dx = edge.x1 - edge.x0, dy = edge.y1 - edge.y0;
        const double px = x - edge.x0, py = y - edge.y0;
        const auto projection = std::clamp ((dx * px + dy * py) / (dx * dx + dy * dy), 0.0, 1.0);
        distance = std::min (distance, std::hypot (px - projection * dx, py - projection * dy));
        const auto side = dx * py - dy * px;
        if (edge.y0 <= y && edge.y1 > y && side > 0) ++winding;
        if (edge.y0 > y && edge.y1 <= y && side < 0) --winding;
    }
    distance *= winding == 0 ? 1 : -1;
    const auto cap = std::abs (z) - program.records()[program.rootIndex()].parameters[0] / 2;
    return std::hypot (std::max (distance, 0.0), std::max (cap, 0.0))
        + std::min (std::max (distance, cap), 0.0);
}

double rectangleDistance (double x, double y, double halfExtent)
{
    const auto dx = std::abs (x) - halfExtent, dy = std::abs (y) - halfExtent;
    return std::hypot (std::max (dx, 0.0), std::max (dy, 0.0)) + std::min (std::max (dx, dy), 0.0);
}

void geometryAndIdentity()
{
    std::string error;
    auto source = textIr();
    const auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    const auto program = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (admitted && program, "v2 exact outline admission and table compilation");
    if (! admitted || ! program) std::exit (2);
    check (admitted->schemaVersion() == 2 && admitted->textOutlines().size() == 1,
           "admission retains schema and immutable outlines");
    check (videohelper::sdf::validateNativeSdfProgram (*program, source, error, true),
           "source-bound validation includes exact tables and full resource identity");
    check (! videohelper::sdf::compileNativeSdfProgram (source, error)
           && ! videohelper::sdf::validateNativeSdfProgram (*program, source, error),
           "unimplemented native consumers refuse text before dispatch");
    arbitgpu::NativeSdfExecutionCapabilities unavailable;
    check (! unavailable.supports (videowire::SdfOperation::textExtrusion), "text capability defaults unavailable");
    check (program->textResourceIdentities()[0] == admitted->textOutlines()[0].identitySha256()
           && program->textResourceIdentities()[0].size() == 64, "full SHA-256 stays in native receipt");
    check (program->textResourceIndices()[0] == 0 && program->textResources()[0].segmentCount == 8
           && program->textContours()[1].winding == -1, "exact resource index and clockwise counter packing");
    check (program->textTableBytes() == 4 + 16 + 2 * 16 + 8 * 16 + 3 * 32 + 8 * 4 + 4
           && program->storageBytes() > sizeof (arbitgpu::NativeSdfCompiledRecord) + program->textTableBytes(),
           "compiled bytes include variable tables and full identity storage");
    for (const auto& xy : std::array<std::array<double, 2>, 7> {{
            {{ 0,0 }}, {{ .75,0 }}, {{ 1,0 }}, {{ .5,.5 }}, {{ 1.2,1.3 }}, {{ .2,.1 }}, {{ -.8,.7 }} }})
        for (const auto z : { -.55, -.25, 0.0, .25, .55 })
        {
            const auto field2 = std::max (rectangleDistance (xy[0], xy[1], 1),
                                         -rectangleDistance (xy[0], xy[1], .5));
            const auto dz = std::abs (z) - .25;
            const auto expected = std::hypot (std::max (field2, 0.0), std::max (dz, 0.0))
                + std::min (std::max (field2, dz), 0.0);
            check (std::abs (packedDistance (*program, xy[0], xy[1], z) - expected) < 1e-12,
                   "polygon oracle covers counter, side, front/back cap and outside corners");
        }
    check (packedDistance (*program, 0,0,0) > 0 && packedDistance (*program, .75,0,0) < 0,
           "counter remains empty and solid is negative inside");
    auto changed = textIr (rectangleOutline ("A"));
    const auto changedProgram = videohelper::sdf::compileNativeSdfProgram (changed, error, true);
    check (changedProgram && changedProgram->structuralDigest() != program->structuralDigest()
           && ! videohelper::sdf::validateNativeSdfProgram (*program, changed, error, true),
           "same vector geometry with changed source identity cannot reuse stale program");
    changed = source;
    changed.textOutlines[0].segments[0].x0 -= .125;
    check (! videohelper::sdf::admitSdfIr (changed, {}, error), "changed outline with stale identity is refused");
    changed = source;
    changed.records[0].resourceStableId ^= 1;
    check (! videohelper::sdf::compileNativeSdfProgram (changed, error, true), "missing exact resource reference refuses");
    source.textOutlines[0].segments.clear();
    check (admitted->textOutlines()[0].outline().segments.size() == 8 && program->textSegments().size() == 8,
           "source mutation cannot change admitted or factory-owned resources");
}

// Exercise the derived traversal without claiming native float pruning. The
// rectangle oracle above is independent of this table/index implementation.
double indexedDistance (const arbitgpu::NativeSdfCompiledProgram& program, double x, double y)
{
    const auto root = program.textIndexRoots()[0];
    double nearest = std::numeric_limits<double>::infinity();
    int winding = 0;
    for (auto nodeIndex = root; nodeIndex < program.textIndexNodes()[root].escape;)
    {
        const auto& node = program.textIndexNodes()[nodeIndex];
        if (node.indexCount == 0) { ++nodeIndex; continue; }
        const bool crossing = y >= node.minY && y < node.maxY;
        for (std::size_t j = 0; j < node.indexCount; ++j)
        {
            const auto& edge = program.textSegments()[program.textSegmentIndices()[node.firstIndex + j]];
            const double dx = edge.x1 - edge.x0, dy = edge.y1 - edge.y0;
            const double px = x - edge.x0, py = y - edge.y0;
            const auto along = std::clamp ((dx * px + dy * py) / (dx * dx + dy * dy), 0.0, 1.0);
            nearest = std::min (nearest, std::hypot (px - dx * along, py - dy * along));
            if (crossing)
            {
                const auto side = dx * py - dy * px;
                if (edge.y0 <= y && edge.y1 > y && side > 0) ++winding;
                if (edge.y0 > y && edge.y1 <= y && side < 0) --winding;
            }
        }
        nodeIndex = node.escape;
    }
    return winding == 0 ? nearest : -nearest;
}

void derivedIndexAndBounds()
{
    std::string error;
    auto source = textIr();
    const auto program = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    if (! program) std::exit (2);
    check (program->textIndexNodes().size() == 3 && program->textIndexRoots() == std::vector<std::uint32_t> { 0 }
        && program->textIndexNodeVisitsPerQuery() == 3, "deterministic threaded hierarchy and worst-case node bound");
    for (double x : { -2., -1., -.5, 0., .5, .75, 1., 2. })
        for (double y : { -2., -1., -.5, 0., .5, .75, 1., 2. })
            check (std::abs (indexedDistance (*program, x, y)
                    - std::max (rectangleDistance (x, y, 1), -rectangleDistance (x, y, .5))) < 1e-12,
                "indexed nearest and nonzero half-open winding match independent rectangle/counter oracle");
    for (const auto& node : program->textIndexNodes())
    {
        check (node.escape <= program->textIndexNodes().size() && node.reserved == 0
            && node.indexCount <= arbitgpu::kNativeSdfTextIndexLeafSegments,
            "threaded traversal and leaf capacity are bounded");
        if (node.indexCount != 0)
            for (std::uint32_t i = 0; i < node.indexCount; ++i)
            {
                const auto& edge = program->textSegments()[program->textSegmentIndices()[node.firstIndex + i]];
                check (node.minX < std::min (edge.x0, edge.x1) && node.maxX > std::max (edge.x0, edge.x1)
                    && node.minY < std::min (edge.y0, edge.y1) && node.maxY > std::max (edge.y0, edge.y1),
                    "outward float boxes strictly enclose both packed segment endpoints");
            }
    }
    auto invalid = *program;
    auto& nodes = const_cast<std::vector<arbitgpu::NativeSdfTextIndexNode>&> (invalid.textIndexNodes());
    nodes[0].escape = std::numeric_limits<std::uint32_t>::max();
    check (! videohelper::sdf::validateNativeSdfProgram (invalid, error, true), "invalid escape refuses before traversal");
    invalid = *program;
    auto& indices = const_cast<std::vector<std::uint32_t>&> (invalid.textSegmentIndices());
    indices[0] = indices[1];
    check (! videohelper::sdf::validateNativeSdfProgram (invalid, error, true), "duplicate segment index refuses");
    invalid = *program;
    auto& roots = const_cast<std::vector<std::uint32_t>&> (invalid.textIndexRoots());
    roots[0] = 1;
    check (! videohelper::sdf::validateNativeSdfProgram (invalid, error, true), "invalid resource root refuses");
    check (program->sceneBounds() && program->sceneBounds()->minimum[0] < -1
        && program->sceneBounds()->maximum[1] > 1 && program->sceneBounds()->minimum[2] < -.25,
        "exact text bounds enclose polygon and extrusion caps");
    videowire::SdfRecord transform;
    transform.stableId = 42; transform.operation = videowire::SdfOperation::translate;
    transform.inputCount = 1; transform.inputs[0] = 41; transform.parameterCount = 3;
    transform.parameters = { 10, -4, 2 };
    source.records.push_back (transform); source.rootId = 42;
    auto transformed = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (transformed && transformed->sceneBounds() && transformed->sceneBounds()->minimum[0] < 9
        && transformed->sceneBounds()->maximum[0] > 11 && transformed->sceneBounds()->maximum[2] > 2.25,
        "translation propagates outward finite zero-set bounds");
    source.records[1].operation = videowire::SdfOperation::scale;
    source.records[1].parameters = { -2, 2, -2 };
    transformed = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (transformed && transformed->sceneBounds() && transformed->sceneBounds()->minimum[0] < -2
        && transformed->sceneBounds()->maximum[2] > .5, "signed uniform scale preserves conservative bounds");
    source.records[1].parameters[1] = 3;
    transformed = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (transformed && ! transformed->sceneBounds(), "uncertified nonuniform scale keeps unknown bound");
    source.records[1].operation = videowire::SdfOperation::repeat;
    source.records[1].parameters = { 2, 2, 2 };
    transformed = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (transformed && ! transformed->sceneBounds(), "infinite repetition does not invent finite scene bounds");
}

void stagedWorkPlan()
{
    using namespace videohelper::sdf;
    std::string error;
    arbitgpu::NativeSdfDrawRequest request;
    request.geometry = textIr(); request.width = 2; request.height = 3;
    request.maximumSteps = 10; request.epsilon = .001; request.maximumDistance = 100;
    request.compiledProgram = compileNativeSdfProgram (request.geometry, error, true);
    arbitgpu::NativeSdfWorkReceipt receipt;
    const std::array<std::uint64_t, 8> perPixel { 32,16,16,17,35,24,32,416 };
    for (std::size_t i = 0; i < perPixel.size(); ++i)
    {
        request.output = static_cast<arbitgpu::NativeSdfOutput> (i);
        NativeSdfRayWorkPlan plan;
        check (nativeSdfRayWorkPlan (request, plan, error) && ! plan.certifiedRayCulling
            && plan.sceneBounds && plan.activePixelCount == 6 && plan.primaryQueryCount == 60
            && plan.primaryQueryCount + plan.hitQueryCount == 6 * perPixel[i]
            && plan.endPixel == std::array<std::uint32_t,2> {2,3}
            && plan.edgeHaloPixels == (i == 7 ? 8u : 0u),
            "full-frame staged schedule reserves one primary plus every output branch and edge halo");
        check (admitNativeSdfStagedTextWork (request, receipt, error)
            && receipt.distanceQueryCount == 6 * perPixel[i]
            && receipt.segmentVisits == 8 * receipt.distanceQueryCount
            && receipt.textIndexNodeVisits == 3 * receipt.distanceQueryCount,
            "staged receipt charges all segments and every possible hierarchy node");
    }
    request.output = arbitgpu::NativeSdfOutput::normal;
    check (! admitNativeSdfStagedTextWork (request, receipt, error, 1000, 1000, 287)
        && receipt.distanceQueryCount == 0 && receipt.textIndexNodeVisits == 0,
        "node budget failure leaves atomic zero receipt");
    request.maximumSteps = 128; request.output = arbitgpu::NativeSdfOutput::color;
    for (const auto extent : { std::array<std::uint32_t,2> {1920,1080}, {3840,2160} })
    {
        request.width = extent[0]; request.height = extent[1];
        NativeSdfRayWorkPlan plan;
        check (nativeSdfRayWorkPlan (request, plan, error) && plan.activePixelCount == std::uint64_t (extent[0]) * extent[1]
            && ! admitNativeSdfStagedTextWork (request, receipt, error),
            "ordinary1080/4K refuses until certified culling or a measured device policy exists");
    }
    const auto admittedGeometry = admitSdfIr (request.geometry, {}, error);
    arbitgpu::NativeSdfExecutionCapabilities capabilities;
    capabilities.available = true; capabilities.backend = "staged-fixture";
    capabilities.supportedOperations.fill (true); capabilities.supportedOutputs.fill (true);
    capabilities.maxOperations = 64; capabilities.maxDepth = 32;
    capabilities.maxExtent = 4096; capabilities.maxPixels = 4096ull * 4096;
    capabilities.maxSteps = 512; capabilities.minEpsilon = .000001;
    capabilities.maxEpsilon = .1; capabilities.maxDistance = 1000;
    NativeSdfRenderControls controls; controls.output = arbitgpu::NativeSdfOutput::normal;
    check (admittedGeometry && ! admitNativeSdfRender (*admittedGeometry, {320,180}, controls, capabilities, error),
        "legacy consumer retains double-primary segment refusal");
    capabilities.reusesPrimaryTrace = true;
    check (admittedGeometry && admitNativeSdfRender (*admittedGeometry, {320,180}, controls, capabilities, error),
        "explicit primary reuse capability selects staged admission without lowering quality");
    request.geometryCacheBytes = UINT64_C(2) * 1024 * 1024 * 1024;
    NativeSdfRayWorkPlan plan;
    check (! nativeSdfRayWorkPlan (request, plan, error) && plan.minimumStorageBytes == 0,
        "aggregate cache/index/fulltexture/intermediate storage cannot exceed2GiB");
    request.geometryCacheBytes = 0; request.width = 2; request.height = 3;
    request.epsilon = std::numeric_limits<double>::quiet_NaN();
    check (! admitNativeSdfStagedTextWork (request, receipt, error), "nonfinite staged controls refuse");
}

void conversionAndBounds()
{
    std::string error;
    auto outline = rectangleOutline ("I", false);
    outline.segments = { {100,0,100.000001,0}, {100.000001,0,100.000001,1},
                         {100.000001,1,100,1}, {100,1,100,0} };
    outline.stableId = videowire::sdfTextOutlineStableId (outline);
    auto source = textIr (outline);
    check (videohelper::sdf::admitSdfIr (source, {}, error).has_value()
           && ! videohelper::sdf::compileNativeSdfProgram (source, error, true),
           "finite double outline with collapsed float segment refuses");
    outline.contours[0].segmentCount = 3;
    outline.segments = { {0,0,1,1}, {1,1,2,2.00000001}, {2,2.00000001,0,0} };
    outline.stableId = videowire::sdfTextOutlineStableId (outline);
    source = textIr (outline);
    check (videohelper::sdf::admitSdfIr (source, {}, error).has_value()
           && ! videohelper::sdf::compileNativeSdfProgram (source, error, true),
           "float contour area collapse refuses despite distinct segments");
    source = textIr();
    source.textOutlines[0].segments[0].x1 += .25;
    source.textOutlines[0].stableId = videowire::sdfTextOutlineStableId (source.textOutlines[0]);
    source.records[0].resourceStableId = source.textOutlines[0].stableId;
    check (! videohelper::sdf::compileNativeSdfProgram (source, error, true), "open contour refuses even with renewed identity");
    source = textIr();
    source.textOutlines[0].contours[0].winding = -1;
    source.textOutlines[0].stableId = videowire::sdfTextOutlineStableId (source.textOutlines[0]);
    source.records[0].resourceStableId = source.textOutlines[0].stableId;
    check (! videohelper::sdf::compileNativeSdfProgram (source, error, true), "wrong contour winding refuses even with renewed identity");
    source = textIr();
    source.textOutlines[0].segments[0].x0 = std::numeric_limits<double>::infinity();
    check (! videohelper::sdf::compileNativeSdfProgram (source, error, true), "non-finite contour refuses");
    source = textIr();
    source.records[0].parameters[0] = videowire::kSdfTextMinimumDepth;
    const auto minimum = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (minimum && videohelper::sdf::validateNativeSdfProgram (*minimum, source, error, true),
           "minimum shared depth survives float boundary conversion");
    source.textOutlines.push_back (rectangleOutline ("A"));
    check (! videohelper::sdf::compileNativeSdfProgram (source, error, true), "unowned outline refuses");
    source = textIr();
    source.textOutlines[0].segments.resize (videowire::kSdfTextMaximumSegments + 1);
    check (! videohelper::sdf::compileNativeSdfProgram (source, error, true), "segment capacity refuses before table allocation");
}

void resourceOrderAndWork()
{
    std::string error;
    auto source = textIr();
    const auto second = rectangleOutline ("A", false);
    source.textOutlines.push_back (second);
    auto other = source.records[0]; other.stableId = 42; other.resourceStableId = second.stableId;
    source.records.push_back (other);
    videowire::SdfRecord unite;
    unite.stableId = 43; unite.operation = videowire::SdfOperation::unionOp;
    unite.inputCount = 2; unite.inputs = { 41,42 };
    source.records.push_back (unite); source.rootId = 43;
    std::sort (source.textOutlines.begin(), source.textOutlines.end(), [] (const auto& a, const auto& b)
        { return a.stableId < b.stableId; });
    const auto program = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (program && program->evaluationSteps() == 3 && program->segmentVisitsPerQuery() == 12,
           "expanded query charges each referenced outline");
    if (program)
    {
        check (program->textResourceStableIds()[program->textResourceIndices()[0]]
                   == source.records[0].resourceStableId
               && program->textResourceStableIds()[program->textResourceIndices()[1]]
                   == source.records[1].resourceStableId
               && program->textResourceIndices()[2] == std::numeric_limits<std::uint32_t>::max(),
               "sorted resources preserve record references and non-text sentinel");
    }
    // Shared DAG edges are repeated shader traversals, not unique-node work.
    source.records[2].inputs = { 41,41 }; source.records.erase (source.records.begin() + 1);
    source.textOutlines.erase (std::remove_if (source.textOutlines.begin(), source.textOutlines.end(),
        [&](const auto& resource) { return resource.stableId == second.stableId; }), source.textOutlines.end());
    const auto repeated = videohelper::sdf::compileNativeSdfProgram (source, error, true);
    check (repeated && repeated->evaluationSteps() == 3 && repeated->segmentVisitsPerQuery() == 16
           && repeated->textIndexNodeVisitsPerQuery() == 6,
           "repeated DAG child edges charge record and segment work twice");
    source = textIr();
    videohelper::sdf::NativeSdfRenderControls controls;
    controls.maximumSteps = 10;
    arbitgpu::NativeSdfWorkReceipt receipt;
    const std::array<std::uint64_t, 8> perPixel { 42,26,26,27,45,34,42,426 };
    for (std::size_t i = 0; i < perPixel.size(); ++i)
    {
        controls.output = static_cast<arbitgpu::NativeSdfOutput> (i);
        check (videohelper::sdf::admitNativeSdfTextWork (source, {2,3}, controls, receipt, error)
               && receipt.distanceQueryCount == 6 * perPixel[i]
               && receipt.recordVisits == receipt.distanceQueryCount
               && receipt.segmentVisits == 8 * receipt.distanceQueryCount,
               "query receipt covers actual output branches and all pixel slots");
    }
    controls.output = arbitgpu::NativeSdfOutput::normal;
    check (! videohelper::sdf::admitNativeSdfTextWork (source, {2,3}, controls, receipt, error, 155, 2000)
           && receipt.distanceQueryCount == 0, "record budget refuses atomically");
    check (! videohelper::sdf::admitNativeSdfTextWork (source, {2,3}, controls, receipt, error, 2000, 1247),
           "segment budget includes normal and raw Depth work");
    controls.maximumSteps = 128; controls.output = arbitgpu::NativeSdfOutput::color;
    check (! videohelper::sdf::admitNativeSdfTextWork (source, {1920,1080}, controls, receipt, error),
           "default 1080p text remains refused by bounded policy, without reducing extent");
    controls.maximumSteps = std::numeric_limits<std::uint32_t>::max();
    controls.output = arbitgpu::NativeSdfOutput::edgeDistance;
    check (! videohelper::sdf::admitNativeSdfTextWork (source,
        {std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max()},
        controls, receipt, error), "query multiplication overflow refuses");
}

void legacyRegression()
{
    static_assert (sizeof (arbitgpu::NativeSdfCompiledRecord) == 56, "v1 fixed record bytes changed");
    static_assert (std::is_same_v<decltype (std::declval<const arbitgpu::NativeSdfCompiledProgram&>().textSegments()),
                   const std::vector<arbitgpu::NativeSdfTextSegment>&>, "tables must be readonly");
    std::string error;
    videowire::SdfIr source;
    source.rootId = 1;
    videowire::SdfRecord sphere; sphere.stableId = 1; sphere.parameterCount = 1; sphere.parameters[0] = 1;
    source.records.push_back (sphere);
    const auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    const auto program = videohelper::sdf::compileNativeSdfProgram (source, error);
    check (admitted && program && videohelper::sdf::validateNativeSdfProgram (*program, source, error)
           && program->textTableBytes() == 0 && program->storageBytes() == 56
           && program->textResourceIndices().empty() && program->segmentVisitsPerQuery() == 0,
           "v1 compilation and receipt bytes remain unchanged");
    if (admitted)
        check (admitted->structuralDigest() == "11bc60410188114a77e2a0b51d670d0ece83280f689d61be377aa278bb728d3d",
               "v1 retains pinned canonical SHA-256 with no additive text fields");
    source.records[0].resourceStableId = 1;
    check (! videohelper::sdf::admitSdfIr (source, {}, error), "v1 nonzero text resource refuses");
}
}

int main()
{
    geometryAndIdentity(); derivedIndexAndBounds(); stagedWorkPlan(); conversionAndBounds(); resourceOrderAndWork(); legacyRegression();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
