// Content-sized collider lists. ParticleGeometryFrame held fixed arrays of
// 256 mesh vertices, 128 triangles and 64 points. The lists now size from the
// geometry, so a 1000-vertex mesh and 300 points validate, the heap bytes
// count what the lists hold, and a read past the end stays zero. Particle
// points, curves and instances in a plan are no longer held to 64/256/64.

#include "../src/particle_solid_state.h"
#include "../../shared/GeometryFieldLowering.h"
#include "../../shared/ParticleGeometryContract.h"

#include <cstdio>
#include <string>

namespace
{
int failures = 0;

void check (bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}
}

int main ()
{
    using videorender::ParticleGeometryFrame;

    // A 25 x 40 grid in the z=0 plane: 1000 vertices, 2 triangles per cell.
    constexpr std::uint32_t columns = 25, rows = 40;
    ParticleGeometryFrame frame;
    frame.meshIdentity = 7;
    frame.meshVertexCount = columns * rows;
    frame.meshVertices.resize (frame.meshVertexCount);
    frame.meshVertexIdentities.resize (frame.meshVertexCount);
    for (std::uint32_t row = 0; row < rows; ++row)
        for (std::uint32_t column = 0; column < columns; ++column)
        {
            const auto vertex = row * columns + column;
            frame.meshVertices[vertex] = { -6.0f + 0.5f * float (column), -10.0f + 0.5f * float (row), 0.0f };
            frame.meshVertexIdentities[vertex] = 1000 + vertex;
        }
    for (std::uint32_t row = 0; row + 1 < rows; ++row)
        for (std::uint32_t column = 0; column + 1 < columns; ++column)
        {
            const auto a = row * columns + column, b = a + 1, c = a + columns, d = c + 1;
            frame.meshTriangles.push_back ({ a, b, c });
            frame.meshTriangles.push_back ({ b, d, c });
        }
    frame.meshTriangleCount = static_cast<std::uint32_t> (frame.meshTriangles.size());
    check (frame.meshVertexCount == 1000 && frame.meshTriangleCount == 2 * 24 * 39,
           "the grid holds 1000 vertices and 1872 triangles");

    videorender::SolidTriangleWorkBudget work;
    std::string error;
    check (videorender::validateSolidTriangleFrame (frame, work, error) && error.empty(),
           "a 1000-vertex collider validates past the old 256-vertex array");

    frame.meshVertexIdentities[999] = frame.meshVertexIdentities[3];
    videorender::SolidTriangleWorkBudget duplicateWork;
    check (! videorender::validateSolidTriangleFrame (frame, duplicateWork, error),
           "a duplicate identity in the heap-sorted copy still refuses");
    frame.meshVertexIdentities[999] = 1999;

    frame.pointCount = 300;
    for (int point = 0; point < frame.pointCount; ++point)
    {
        frame.solidPoints.push_back ({ float (point) * 0.01f, 0.0f, 1.0f });
        frame.pointIdentities.push_back (5000 + std::uint64_t (point));
    }
    check (frame.solidPoints.size() == 300 && frame.pointIdentities[299] == 5299,
           "300 point colliders fit past the old 64-point array");

    const auto heap = videorender::particleGeometryFrameHeapBytes (frame);
    check (heap >= 1000 * (sizeof (std::array<float, 3>) + sizeof (std::uint64_t))
                       + 1872 * sizeof (std::array<std::uint32_t, 3>)
                       + 300 * (sizeof (std::array<float, 3>) + sizeof (std::uint64_t)),
           "heap bytes count every vertex, triangle and point the lists hold");

    const ParticleGeometryFrame& readOnly = frame;
    check (readOnly.segments[40][0] == 0 && readOnly.segments.size() == 0,
           "a read past the end is zero and does not grow the list");

    ParticleGeometryFrame copy = frame;
    copy.segments.resize (4);
    check (copy.segments == frame.segments, "zero padding compares equal like the old zeroed array");
    copy.segments[1] = { 1, 2, 3, 4 };
    check (copy.segments != frame.segments, "a written segment compares unequal");

    // Particle mesh barriers, imported geometry and collider family children
    // size from the mesh: a 25 x 40 grid of 1000 vertices and 1872 triangles
    // validates past the old 256-vertex and 128-triangle contract.
    {
        using namespace videowire::geometry;
        PortContract contract; contract.carrier = CarrierKind::geometry3D;
        contract.maxVertices = 4096; contract.maxIndices = 12288;
        std::string error;
        const auto grid = lowerGrid (18400, 18400, 1, columns, rows, 0.05f, contract, error);
        check (grid && std::get<GeometryData> (grid->data).indices.size() == 6u * (columns - 1) * (rows - 1),
               "the 25 x 40 grid lowers");
        check (grid && validateParticleGeometry (*grid, CarrierKind::geometry3D, error),
               "a 1000-vertex particle mesh barrier validates");
        if (grid)
        {
            auto partial = *grid;
            std::get<GeometryData> (partial.data).indices.pop_back();
            check (! validateParticleGeometry (partial, CarrierKind::geometry3D, error)
                       && error.find ("whole triangles") != std::string::npos,
                   "a partial triangle is still refused");
        }
    }

    // Points, curve points and instances size from the plan: 300 points, 1000
    // curve points and 300 instances fit where 64, 256 and 64 used to refuse.
    {
        using namespace videowire::geometry;
        std::string error;
        ValueDescriptor points;
        points.carrier = CarrierKind::points3D;
        PointsData data;
        for (std::uint64_t i = 0; i < 300; ++i)
            data.points.push_back ({ { float (i % 20) * 0.01f, float (i / 20) * 0.01f, 0 }, i + 1 });
        points.data = data;
        check (validateParticleGeometry (points, CarrierKind::points3D, error, false), "300 particle points validate");
        points.data = PointsData {};
        check (! validateParticleGeometry (points, CarrierKind::points3D, error, false)
                   && error.find ("at least one") != std::string::npos,
               "an empty point set is still refused");
        MaterializedInstancePlan plan;
        plan.points[1] = data;
        CurvesData curve;
        curve.points.resize (1000);
        plan.curves[2] = curve;
        plan.instances.instances.resize (300);
        check (particleGeometryIntermediatesFit (plan, error), "300 points, 1000 curve points and 300 instances fit");
        GeometryData partial;
        partial.positions.resize (3);
        partial.indices = { 0, 1 };
        plan.geometries[3] = partial;
        check (! particleGeometryIntermediatesFit (plan, error), "a partial intermediate triangle is still refused");
    }

    if (failures != 0)
    {
        std::fprintf (stderr, "%d particle collider check(s) failed\n", failures);
        return 1;
    }
    std::puts ("particle collider content-sized tests passed");
    return 0;
}
