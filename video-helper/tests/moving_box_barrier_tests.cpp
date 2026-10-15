// Translation-only moving-box barrier pins (Plan 3 item 03.10).
//
// Scope: fixed-orientation box proxies against one triangle, through the
// certified sweeps solidtriangle::boxSweep (static triangle) and
// solidtriangle::movingBoxSweep (linearly moving triangle, old -> current).
// Rotation and the solidBoxTriangleContact dispatch are not exercised here.
//
// Every expected time and normal below is derived by hand in its case
// comment. None is copied from the sweep output.
//
// Shared geometry. The triangle lies in a horizontal plane with vertices
// A=(-10,-10,z), B=(10,-10,z), C=(0,10,z), shifted laterally by dx. Edge AC
// is x=-5+y/2 and edge BC is x=5-y/2, so for |y|<=1/2 the triangle spans at
// least [-4.75,4.75] before the shift. A box with half extent 1/2 has
// footprint |x-cx|<=1/2 and |y|<=1/2. When |cx-dx|<=4.25 that footprint lies
// over the triangle, the nearest feature is a horizontal face, and the
// box-to-triangle distance equals the vertical gap between that face and the
// plane. The sweeps use reach=0, so contact means distance within a
// tolerance of about 1e-5.

#include "../src/particle_solid_state.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace
{
using videorender::SolidBoxProxy;
using videorender::SolidTriangleWorkBudget;
using videorender::SolidVector;
namespace solidtriangle = videorender::solidtriangle;

int failures = 0;

void check (bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

bool closeTo (double value, double expected, double tolerance)
{
    return std::abs (value - expected) <= tolerance;
}

// Identity orientation with one half extent on every axis.
SolidBoxProxy alignedBox (float halfExtent)
{
    return { { halfExtent, halfExtent, halfExtent },
             {{ { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }} };
}

// The triangle in plane z, translated laterally by dx.
std::array<SolidVector,3> triangleAt (float z, float dx)
{
    return {{ { -10.0f + dx, -10.0f, z }, { 10.0f + dx, -10.0f, z }, { dx, 10.0f, z } }};
}
}

int main ()
{
    check (solidtriangle::arithmeticSupported(),
        "moving box oracles require IEEE binary64 nearest-rounding arithmetic");
    const SolidBoxProxy box = alignedBox (0.5f);
    const SolidVector origin { 0.0f, 0.0f, 0.0f };

    // Case 1: a box translating into a static triangle.
    // Triangle at z=0, dx=0. Box center path (2t,0,3-4t), from (0,0,3) to
    // (2,0,-1). The bottom face sits at 2.5-4t, so it meets z=0 at
    // t = 2.5/4 = 0.625. The box centre stays within |x|<=1.25 over that
    // interval, so the footprint lies over the triangle. The contact normal
    // points from the triangle toward the box, which is +z.
    {
        SolidTriangleWorkBudget work;
        solidtriangle::Query contact;
        bool hit = false;
        std::string error;
        const bool ok = solidtriangle::boxSweep (SolidVector { 0.0f, 0.0f, 3.0f },
            SolidVector { 2.0f, 0.0f, -1.0f }, box, triangleAt (0.0f, 0.0f), 0.0,
            work, contact, hit, error);
        check (ok && error.empty() && hit
            && closeTo (contact.time, 0.625, 1.0e-4)
            && closeTo (contact.normal[0], 0.0, 1.0e-6)
            && closeTo (contact.normal[1], 0.0, 1.0e-6)
            && closeTo (contact.normal[2], 1.0, 1.0e-6),
            "a box translating into a static triangle contacts at t=0.625 with normal +z");
    }

    // Case 2: a static box hit by a translating triangle mesh.
    // Box at the origin, top face at z=0.5. The triangle moves from old z=3
    // to current z=-1 with lateral shift dx from 0 to -2, so its plane is
    // z(t)=3-4t and its shift is -2t. Contact is 3-4t=0.5, i.e. t=0.625.
    // Over the interval the shift is at least -1.25, and the footprint stays
    // within 4.25 of the triangle centre, so the face is still the nearest
    // feature. The triangle is above the box, so the normal from the triangle
    // toward the box is -z.
    {
        SolidTriangleWorkBudget work;
        solidtriangle::Query contact;
        bool hit = false;
        std::string error;
        const bool ok = solidtriangle::movingBoxSweep (origin, origin, box,
            triangleAt (3.0f, 0.0f), triangleAt (-1.0f, -2.0f), 0.0,
            work, contact, hit, error);
        check (ok && error.empty() && hit
            && closeTo (contact.time, 0.625, 1.0e-4)
            && closeTo (contact.normal[0], 0.0, 1.0e-6)
            && closeTo (contact.normal[1], 0.0, 1.0e-6)
            && closeTo (contact.normal[2], -1.0, 1.0e-6),
            "a static box hit by a translating triangle contacts at t=0.625 with normal -z");
    }

    // Case 3: a box and a triangle translating in parallel at (2,0,-4).
    // The box center runs (2t,0,5-4t) and the triangle shifts by (2t,0,-4t)
    // from z=3. The relative position is fixed, and the bottom face stays
    // 1.5 above the triangle plane (4.5-4t against 3-4t). Relative motion is
    // zero, so no contact is possible.
    {
        SolidTriangleWorkBudget work;
        solidtriangle::Query contact;
        bool hit = true;
        std::string error;
        const bool ok = solidtriangle::movingBoxSweep (SolidVector { 0.0f, 0.0f, 5.0f },
            SolidVector { 2.0f, 0.0f, 1.0f }, box,
            triangleAt (3.0f, 0.0f), triangleAt (-1.0f, 2.0f), 0.0,
            work, contact, hit, error);
        check (ok && error.empty() && !hit,
            "a box and triangle moving in parallel at the same velocity report no contact");
    }

    // Case 4a: a clearance miss against a static triangle.
    // Triangle at z=0. The box center runs (-3+6t,0,0.75), so its bottom face
    // is at z=0.25 throughout. Its x extent stays within [-3.5,3.5], inside
    // the footprint, so the distance is exactly 0.25 and never reaches the
    // tolerance.
    {
        SolidTriangleWorkBudget work;
        solidtriangle::Query contact;
        bool hit = true;
        std::string error;
        const bool ok = solidtriangle::boxSweep (SolidVector { -3.0f, 0.0f, 0.75f },
            SolidVector { 3.0f, 0.0f, 0.75f }, box, triangleAt (0.0f, 0.0f), 0.0,
            work, contact, hit, error);
        check (ok && error.empty() && !hit,
            "a box passing 0.25 above a static triangle reports no contact");
    }

    // Case 4b: a clearance miss against a laterally translating triangle.
    // The static box top is at z=0.5. The triangle plane is z=0.75, which
    // leaves a 0.25 gap, and its shift runs from -3 to +3. The triangle is
    // always in plane z=0.75, so the distance from the box is at least 0.25
    // at every time. Its footprint covers the box for |shift|<=4.25, so the
    // distance is exactly 0.25 and never reaches the tolerance.
    {
        SolidTriangleWorkBudget work;
        solidtriangle::Query contact;
        bool hit = true;
        std::string error;
        const bool ok = solidtriangle::movingBoxSweep (origin, origin, box,
            triangleAt (0.75f, -3.0f), triangleAt (0.75f, 3.0f), 0.0,
            work, contact, hit, error);
        check (ok && error.empty() && !hit,
            "a box under a laterally translating triangle with 0.25 clearance reports no contact");
    }

    if (failures != 0) {
        std::fprintf (stderr, "%d moving box barrier check(s) failed\n", failures);
        return 1;
    }
    std::printf ("moving box barrier checks passed\n");
    return 0;
}
