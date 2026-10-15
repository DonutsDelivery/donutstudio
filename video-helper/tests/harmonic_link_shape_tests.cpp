#include "../src/harmonic_link_shape.h"
#include "../src/harmonic_link_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{
using videohelper::harmonicgeometry::appendLinkShape;
using videohelper::harmonicgeometry::linkSideHalfWidth;
using videowire::geometry::GeometryData;
using videowire::geometry::Vec3;
namespace contract = visualharmonicgeometry;
using contract::LinkShape;

int failures = 0;

void require(bool condition, const std::string& message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool near(float actual, float expected, float tolerance = 1e-4f)
{
    return std::fabs(actual - expected) <= tolerance;
}

bool nearPoint(Vec3 actual, Vec3 expected)
{
    return near(actual.x, expected.x) && near(actual.y, expected.y) && near(actual.z, expected.z);
}

Vec3 centroid(const GeometryData& mesh, std::size_t first, std::size_t count)
{
    Vec3 sum;
    for (std::size_t index = first; index < first + count; ++index)
    {
        sum.x += mesh.positions[index].x;
        sum.y += mesh.positions[index].y;
        sum.z += mesh.positions[index].z;
    }
    const auto n = static_cast<float>(count);
    return { sum.x / n, sum.y / n, sum.z / n };
}

struct TestLink
{
    Vec3 start;
    Vec3 end;
    Vec3 side;
};

// Oblique link with a half width of 0.5. The side vector is perpendicular to the link.
TestLink obliqueLink()
{
    const Vec3 start { 0.0f, 0.0f, 0.0f };
    const Vec3 end { 1.0f, 2.0f, 0.5f };
    const Vec3 direction { end.x - start.x, end.y - start.y, end.z - start.z };
    return { start, end, linkSideHalfWidth(direction, 1.0f) };
}

// Emits one link at the requested detail. A set of one link always keeps the requested detail.
void appendRequested(GeometryData& mesh, LinkShape shape, std::uint32_t curvature,
                     std::uint32_t segments, Vec3 start, Vec3 end, Vec3 side)
{
    const auto effective = contract::effectiveLinkLayout(shape, curvature, segments, 1);
    appendLinkShape(mesh, shape, curvature, effective, start, end, side);
}

void testLayoutCounts()
{
    struct Case
    {
        LinkShape shape;
        std::uint32_t curvature;
        std::uint32_t segments;
        std::size_t vertices;
        std::size_t triangles;
        const char* name;
    };
    const Case cases[] {
        { LinkShape::ribbon, 50, 4, 4, 2, "ribbon" },
        { LinkShape::beam, 50, 4, 8, 12, "beam" },
        { LinkShape::curve, 50, 4, 10, 8, "curve" },
        { LinkShape::curve, 50, 1, 4, 2, "curve with one segment" },
        { LinkShape::spring, 50, 4, 34, 32, "spring with four coils" },
        { LinkShape::spring, 0, 4, 10, 8, "spring with one coil" },
        { LinkShape::strut, 50, 4, 16, 12, "strut" },
    };
    const auto link = obliqueLink();
    for (const auto& c : cases)
    {
        GeometryData mesh;
        appendRequested(mesh, c.shape, c.curvature, c.segments, link.start, link.end, link.side);
        const auto layout = contract::effectiveLinkLayout(c.shape, c.curvature, c.segments, 1).layout;
        const std::string name = c.name;
        require(mesh.positions.size() == c.vertices, name + " vertex count");
        require(mesh.indices.size() == c.triangles * 3, name + " index count");
        require(layout.vertices == c.vertices && layout.triangles == c.triangles,
                name + " layout matches emitted counts");
        const bool inRange = std::all_of(mesh.indices.begin(), mesh.indices.end(),
            [&mesh](std::uint32_t index) { return index < mesh.positions.size(); });
        require(inRange, name + " indices stay inside the vertex list");
    }
}

void testBeamIsClosedAndOutward()
{
    const auto link = obliqueLink();
    GeometryData mesh;
    appendRequested(mesh, LinkShape::beam, 50, 4, link.start, link.end, link.side);
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> undirected;
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> directed;
    double volume = 0.0;
    for (std::size_t t = 0; t + 2 < mesh.indices.size(); t += 3)
    {
        const std::uint32_t corners[3] { mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2] };
        for (int edge = 0; edge < 3; ++edge)
        {
            const auto from = corners[edge];
            const auto to = corners[(edge + 1) % 3];
            ++directed[{ from, to }];
            ++undirected[{ std::min(from, to), std::max(from, to) }];
        }
        const auto p0 = mesh.positions[corners[0]];
        const auto p1 = mesh.positions[corners[1]];
        const auto p2 = mesh.positions[corners[2]];
        // Signed volume of the tetrahedron formed with the origin. Outward winding gives a positive sum.
        volume += (p0.x * (p1.y * p2.z - p1.z * p2.y) - p0.y * (p1.x * p2.z - p1.z * p2.x)
                   + p0.z * (p1.x * p2.y - p1.y * p2.x)) / 6.0;
    }
    bool everyEdgeShared = undirected.size() == 18;
    for (const auto& entry : undirected)
        everyEdgeShared = everyEdgeShared && entry.second == 2;
    require(everyEdgeShared, "every beam edge is shared by exactly two triangles");
    bool consistent = true;
    for (const auto& entry : directed)
    {
        const auto reverse = directed.find({ entry.first.second, entry.first.first });
        consistent = consistent && entry.second == 1 && reverse != directed.end() && reverse->second == 1;
    }
    require(consistent, "beam triangles are consistently oriented");
    const auto halfWidth = std::sqrt(link.side.x * link.side.x + link.side.y * link.side.y
                                     + link.side.z * link.side.z);
    const Vec3 axis { link.end.x - link.start.x, link.end.y - link.start.y, link.end.z - link.start.z };
    const auto length = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    // Square section of side 2 * half width, so the volume is (2h)^2 * length.
    const auto expected = 4.0 * halfWidth * halfWidth * length;
    require(std::fabs(volume - expected) <= 1e-3 * expected, "beam volume matches its square section");
}

void testEndpointsMatchLink()
{
    const auto link = obliqueLink();
    const LinkShape shapes[] { LinkShape::ribbon, LinkShape::curve, LinkShape::spring,
                               LinkShape::beam, LinkShape::strut };
    for (const auto shape : shapes)
    {
        GeometryData mesh;
        appendRequested(mesh, shape, 50, 4, link.start, link.end, link.side);
        const auto count = mesh.positions.size();
        const std::string name = contract::shapeName(shape);
        if (shape == LinkShape::beam)
        {
            require(nearPoint(centroid(mesh, 0, 4), link.start), name + " start section sits on the start");
            require(nearPoint(centroid(mesh, 4, 4), link.end), name + " end section sits on the end");
        }
        else if (shape == LinkShape::strut)
        {
            require(nearPoint(centroid(mesh, 0, 4), link.start), name + " thin start section sits on the start");
            require(nearPoint(centroid(mesh, 4, 4), link.end), name + " thin end section sits on the end");
            require(nearPoint(centroid(mesh, 8, 4), link.start), name + " start plate sits on the start");
            require(nearPoint(centroid(mesh, 12, 4), link.end), name + " end plate sits on the end");
        }
        else
        {
            require(nearPoint(centroid(mesh, 0, 2), link.start), name + " first sample sits on the start");
            require(nearPoint(centroid(mesh, count - 2, 2), link.end), name + " last sample sits on the end");
        }
    }
    GeometryData spring;
    appendRequested(spring, LinkShape::spring, 0, 4, link.start, link.end, link.side);
    require(nearPoint(centroid(spring, 0, 2), link.start) && nearPoint(centroid(spring, 8, 2), link.end),
            "single coil spring keeps its endpoints");
}

void testRibbonMatchesPinnedFixture()
{
    // Output of the pre-change lower() for a link from (0, 0, 0) to (1, 0, 0) with Ribbon Width 1.
    const Vec3 start { 0.0f, 0.0f, 0.0f };
    const Vec3 end { 1.0f, 0.0f, 0.0f };
    const auto side = linkSideHalfWidth({ 1.0f, 0.0f, 0.0f }, 1.0f);
    GeometryData mesh;
    appendRequested(mesh, LinkShape::ribbon, 50, 4, start, end, side);
    const Vec3 expected[4] {
        { 0.0f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f }, { 1.0f, -0.5f, 0.0f }, { 1.0f, 0.5f, 0.0f } };
    require(mesh.positions.size() == 4, "ribbon fixture has four corners");
    for (std::size_t index = 0; index < 4 && index < mesh.positions.size(); ++index)
    {
        const auto& actual = mesh.positions[index];
        require(actual.x == expected[index].x && actual.y == expected[index].y
                && actual.z == expected[index].z, "ribbon fixture corner " + std::to_string(index));
    }
    const std::vector<std::uint32_t> expectedIndices { 0, 2, 1, 1, 2, 3 };
    require(mesh.indices == expectedIndices, "ribbon fixture triangles");
}

void testBudgetRefusal()
{
    // Level of detail steps down until a layout fits, so the bound is the capacity at minimum detail.
    require(contract::maximumLinks(LinkShape::ribbon) == 256, "ribbon budget is kMaximumLinks");
    require(contract::maximumLinks(LinkShape::beam) == 42, "beam budget is the index bound");
    require(contract::maximumLinks(LinkShape::strut) == 42, "strut budget is the index bound");
    require(contract::maximumLinks(LinkShape::curve) == 256, "curve budget is kMaximumLinks at one segment");
    require(contract::maximumLinks(LinkShape::spring) == 256, "spring budget is kMaximumLinks at one coil");
    std::string error;
    require(contract::admitLinkCount(LinkShape::beam, 50, 4, 42, error) && error.empty(),
            "beam at budget is admitted");
    require(!contract::admitLinkCount(LinkShape::beam, 50, 4, 43, error), "beam over budget is refused");
    require(error.find("Beam") != std::string::npos && error.find("42") != std::string::npos,
            "refusal names the shape and the admitted count");
    require(contract::admitLinkCount(LinkShape::ribbon, 50, 4, 256, error), "ribbon at kMaximumLinks is admitted");
    require(!contract::admitLinkCount(LinkShape::ribbon, 50, 4, 257, error), "ribbon over kMaximumLinks is refused");
    require(contract::admitLinkCount(LinkShape::spring, 100, 8, 256, error),
            "spring at minimum detail is admitted");
}

void testLevelOfDetailFitsBudget()
{
    // The requested detail holds while the budgets admit the link count.
    const auto single = contract::effectiveLinkLayout(LinkShape::spring, 100, 8, 1);
    require(single.admitted && single.coils == 6 && single.segments == 8,
            "one spring link keeps its requested coils and segments");

    // At six coils and eight segments the segments are the larger value, so they step down first.
    // Six links fit at seven segments.
    const auto six = contract::effectiveLinkLayout(LinkShape::spring, 100, 8, 6);
    require(six.admitted && six.coils == 6 && six.segments == 7,
            "six spring links step segments down to seven");
    require(six.layout.vertices == 86 && six.layout.triangles == 84,
            "six spring links use the stepped layout");
    const auto link = obliqueLink();
    GeometryData springMesh;
    for (int index = 0; index < 6; ++index)
        appendLinkShape(springMesh, LinkShape::spring, 100, six, link.start, link.end, link.side);
    require(springMesh.positions.size() == 516 && springMesh.indices.size() == 1512,
            "six stepped spring links emit 516 vertices and 1512 indices");

    // Two hundred spring links fall through to one coil and one segment, which admits 256 links.
    const auto minimal = contract::effectiveLinkLayout(LinkShape::spring, 100, 8, 200);
    require(minimal.admitted && minimal.coils == 1 && minimal.segments == 1,
            "two hundred spring links step down to one coil and one segment");
    GeometryData minimalMesh;
    for (int index = 0; index < 200; ++index)
        appendLinkShape(minimalMesh, LinkShape::spring, 100, minimal, link.start, link.end, link.side);
    require(minimalMesh.positions.size() == 800 && minimalMesh.indices.size() == 1200,
            "two hundred minimal spring links emit 800 vertices and 1200 indices");

    // Curve steps its segments. Beam never steps, and every shape stops at its floor.
    const auto curveEight = contract::effectiveLinkLayout(LinkShape::curve, 50, 8, 32);
    require(curveEight.admitted && curveEight.segments == 8, "32 curve links keep eight segments");
    const auto curveFour = contract::effectiveLinkLayout(LinkShape::curve, 50, 8, 64);
    require(curveFour.admitted && curveFour.segments == 4, "64 curve links step down to four segments");
    const auto curveThree = contract::effectiveLinkLayout(LinkShape::curve, 50, 8, 65);
    require(curveThree.admitted && curveThree.segments == 3, "65 curve links step down to three segments");
    const auto curveFloor = contract::effectiveLinkLayout(LinkShape::curve, 50, 8, 100000);
    require(!curveFloor.admitted && curveFloor.segments == 1, "curve never steps below one segment");
    const auto springFloor = contract::effectiveLinkLayout(LinkShape::spring, 100, 8, 100000);
    require(!springFloor.admitted && springFloor.coils == 1 && springFloor.segments == 1,
            "spring never steps below one coil and one segment");
    const auto beamAt = contract::effectiveLinkLayout(LinkShape::beam, 50, 4, 42);
    require(beamAt.admitted && beamAt.segments == 4, "beam admits 42 links at the requested detail");
    const auto beamOver = contract::effectiveLinkLayout(LinkShape::beam, 50, 4, 43);
    require(!beamOver.admitted && beamOver.layout.vertices == 8, "beam refuses 43 links rather than stepping");

    std::string error;
    require(!contract::admitLinkCount(LinkShape::spring, 100, 8, 257, error)
                && error.find("Spring") != std::string::npos,
            "spring refusal names the shape");
}

void testTransportVersion()
{
    require(contract::transportVersion("visual.harmonic-link-geometry.v1\n81\n") == 1, "version 1 header");
    require(contract::transportVersion("visual.harmonic-link-geometry.v2\n81\n1 50 4\n") == 2,
            "version 2 header");
    require(contract::transportVersion("visual.harmonic-link-geometry.v3\n81\n") == 3,
            "a newer header reports version 3");
    require(contract::transportVersion("visual.harmonic-link-geometry.v\n81\n") == 0,
            "missing version digits are not a version");
    require(contract::transportVersion("visual.harmonic-link-geometry.v2x\n81\n") == 0,
            "a non digit suffix is not a version");
    require(contract::transportVersion("visual.harmonic-link-geometry.v123456\n81\n") == 0,
            "a long version suffix is not a version");
    require(contract::transportVersion("visual.other-geometry.v3\n81\n") == 0,
            "other headers are not harmonic link versions");
    require(contract::transportVersion("") == 0, "an empty payload has no version");
}

void testTransportRoundTripAndLegacyDecode()
{
    const auto notes = visualnoteinstancing::encode(visualnoteinstancing::Mapping {});
    contract::Mapping mapping;
    mapping.stableId = 81;
    const auto ribbonText = contract::encode(mapping);
    require(ribbonText.rfind("visual.harmonic-link-geometry.v1\n81\n", 0) == 0,
            "ribbon keeps the version 1 header");

    contract::Mapping decoded;
    require(contract::decode(ribbonText, decoded) && decoded.shape == LinkShape::ribbon
                && decoded.curvaturePercent == 50 && decoded.segments == 4,
            "version 1 payload decodes as Ribbon");
    const std::string handWritten = std::string("visual.harmonic-link-geometry.v1\n81\n") + notes;
    require(contract::decode(handWritten, decoded) && decoded.shape == LinkShape::ribbon,
            "hand written version 1 payload decodes as Ribbon");

    const LinkShape shapes[] { LinkShape::beam, LinkShape::curve, LinkShape::spring, LinkShape::strut };
    for (const auto shape : shapes)
    {
        contract::Mapping source;
        source.stableId = 81;
        source.shape = shape;
        source.curvaturePercent = 30;
        source.segments = 6;
        const auto text = contract::encode(source);
        const std::string name = contract::shapeName(shape);
        require(text.rfind("visual.harmonic-link-geometry.v2\n81\n", 0) == 0, name + " uses version 2");
        contract::Mapping reopened;
        require(contract::decode(text, reopened) && reopened.shape == shape
                    && reopened.curvaturePercent == 30 && reopened.segments == 6,
                name + " version 2 round trip");
        require(contract::encode(reopened) == text, name + " re-encodes canonically");
    }

    contract::Mapping invalid;
    invalid.stableId = 81;
    invalid.shape = LinkShape::beam;
    invalid.segments = 0;
    require(contract::encode(invalid).empty(), "zero segments are refused");
    invalid.segments = 9;
    require(contract::encode(invalid).empty(), "nine segments are refused");
    invalid.segments = 4;
    invalid.curvaturePercent = 101;
    require(contract::encode(invalid).empty(), "curvature above one hundred percent is refused");
    invalid.curvaturePercent = 50;
    invalid.shape = static_cast<LinkShape>(5);
    require(contract::encode(invalid).empty(), "unknown shape is refused");

    const std::string ribbonInShapedHeader = std::string("visual.harmonic-link-geometry.v2\n81\n0 50 4\n") + notes;
    require(!contract::decode(ribbonInShapedHeader, decoded), "version 2 with Ribbon is not canonical");
    const std::string shortShapeLine = std::string("visual.harmonic-link-geometry.v2\n81\n1 50\n") + notes;
    require(!contract::decode(shortShapeLine, decoded), "short shape line is refused");
    const std::string extraShapeField = std::string("visual.harmonic-link-geometry.v2\n81\n1 50 4 9\n") + notes;
    require(!contract::decode(extraShapeField, decoded), "extra shape field is refused");
}

// Lowering sets the bright link material after its copy. Each channel of that material is at least 0.8.
void testLinkMaterialIsBright()
{
    namespace harmonic = videohelper::harmonicgeometry;
    contract::Mapping mapping;
    mapping.stableId = 81;
    std::string error;
    const auto value = harmonic::lower(mapping, nullptr, error);
    require(value.has_value() && error.empty(), "a link mapping lowers without a score frame");
    if (!value) return;
    const bool setsMaterial = std::any_of(value->operations.begin(), value->operations.end(),
        [&mapping](const videowire::geometry::OperationRecord& operation)
        {
            return operation.code == videowire::geometry::OperationCode::setMaterial
                && operation.stableId == harmonic::kLinkMaterialId
                && operation.inputStableId == mapping.stableId
                && operation.outputStableId == mapping.stableId;
        });
    require(setsMaterial, "link lowering sets the bright material after its copy");
    require(videowire::geometry::retainedMaterialIdentity(*value) == harmonic::kLinkMaterialId,
            "the retained link material is the bright material");
    // Mirrors the channel lambda in geometry_core_backend.cpp. That lambda is local, so it is not reachable here.
    for (const unsigned shift : { 0u, 8u, 16u })
    {
        const float channel = 0.2f + 0.7f * static_cast<float>((harmonic::kLinkMaterialId >> shift) & 0xffu) / 255.0f;
        require(channel >= 0.8f, "the link base colour is at least 0.8 in every channel");
    }
    require(videowire::geometry::admitValue(*value, contract::geometryContract(), {}, error).has_value(),
            "Geometry Core admits the bright link material");
}
} // namespace

int main()
{
    testLayoutCounts();
    testBeamIsClosedAndOutward();
    testEndpointsMatchLink();
    testRibbonMatchesPinnedFixture();
    testBudgetRefusal();
    testLevelOfDetailFitsBudget();
    testTransportVersion();
    testTransportRoundTripAndLegacyDecode();
    testLinkMaterialIsBright();
    if (failures != 0)
    {
        std::cerr << failures << " harmonic link shape checks failed\n";
        return 1;
    }
    std::cout << "harmonic link shape tests passed\n";
    return 0;
}
