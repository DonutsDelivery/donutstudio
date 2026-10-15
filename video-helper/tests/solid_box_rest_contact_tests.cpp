// Headless Solid 3D rest proofs for an oriented box against a static triangle
// environment (Plan 3 items 53 and 54).
//
// Every expected value below is derived in its case comment. None is copied
// from solver output. The replay is the production historical path,
// replayParticleBodies, with a static triangle collider and an oriented-box
// body whose half extents come from the body appearance bound.
//
// Scene. The environment is one triangle with base vertices A=(-20,-20),
// B=(20,-20), C=(0,20) in the plane z=0 before tilt. Tilt rotates that plane
// about the X axis by theta, so a base point (x,y) maps to (x, y cos, y sin).
// The box is a cube appearance with corners at +-1. Its bounding box gives
// half extents bodyRadius/sqrt(3) on every axis, with bodyRadius = 0.1.
//
// Settings. Restitution 0, friction 0 unless stated, drag 0, no attraction, no
// links, one body, gravity -9.8 along Z, colliderThickness 0.005, mass 1. The
// body spawns at rest, so impulseZ and force are zero. Replay steps are 1/120 s.

#include "../src/particle_body_replay.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

namespace
{
using videorender::ParticleGeometryBinding;
using videorender::ParticleHistorySource;
using videorender::ParticleParams;
using videorender::ParticleSolidState;
namespace geometry = videowire::geometry;

int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}

bool closeTo(double value, double expected, double tolerance)
{
    return std::abs(value - expected) <= tolerance;
}

constexpr double kPi = 3.14159265358979323846;
constexpr float kBodyRadius = 0.1f;
constexpr float kThickness = 0.005f;
constexpr float kGravity = 9.8f;
// Enclosing radius of the unit-corner cube is sqrt(3), and each half extent is
// bodyRadius * 1 / sqrt(3). The bounding-box formula gives the same value.
const float kHalf = kBodyRadius / std::sqrt(3.0f);

// Cube appearance. Only its bounding box is read, for the contact proxy.
std::shared_ptr<const geometry::ValueDescriptor> cubeAppearance()
{
    geometry::GeometryData data;
    for (const float x : {-1.0f, 1.0f})
        for (const float y : {-1.0f, 1.0f})
            for (const float z : {-1.0f, 1.0f})
                data.positions.push_back({x, y, z});
    data.vertexIds = {501, 502, 503, 504, 505, 506, 507, 508};
    data.indices = {0, 1, 2};
    auto descriptor = std::make_shared<geometry::ValueDescriptor>();
    descriptor->carrier = geometry::CarrierKind::geometry3D;
    descriptor->stableId = 5001;
    descriptor->data = std::move(data);
    return descriptor;
}

// The base triangle rotated about X by tiltDegrees. World = (x, y c, y s).
std::shared_ptr<const geometry::ValueDescriptor> groundAt(float tiltDegrees)
{
    const double angle = double(tiltDegrees) * kPi / 180.0;
    const float c = float(std::cos(angle)), s = float(std::sin(angle));
    geometry::GeometryData data;
    const float base[3][2] = {{-20.0f, -20.0f}, {20.0f, -20.0f}, {0.0f, 20.0f}};
    for (const auto& b : base) data.positions.push_back({b[0], b[1] * c, b[1] * s});
    data.vertexIds = {4301, 4302, 4303};
    data.indices = {0, 1, 2};
    auto descriptor = std::make_shared<geometry::ValueDescriptor>();
    descriptor->carrier = geometry::CarrierKind::geometry3D;
    descriptor->stableId = 4201;
    descriptor->data = std::move(data);
    return descriptor;
}

// One box spawned at rest at (x, y, depth), with its orientation set by Euler
// degrees about X only. Those degrees give quaternion (sin(t/2), 0, 0, cos(t/2)).
ParticleParams makeParams(double x, double y, double depth, float friction,
    float tiltDegrees, std::shared_ptr<const geometry::ValueDescriptor> ground)
{
    ParticleParams p;
    p.motionMode = 2;
    p.simulationSpace = 1;
    p.historicalReplay = true;
    auto history = std::make_shared<ParticleHistorySource>();
    history->score = std::make_shared<const arbitmod::Score>();
    p.history = history;
    p.count = 1;
    p.geometryCount = 1;
    p.lifetime = 4;
    p.force = 0;
    p.impulseZ = 0;
    p.geometryAnchors[0] = {float(x), float(y), float(x), float(y)};
    p.geometryDepths[0] = {float(depth), float(depth)};
    p.geometryIdentities[0] = 9007199254741001ull;
    p.bodyShape = 2;
    p.collisionMode = 2;
    p.bodyRadius = kBodyRadius;
    p.restitution = 0;
    p.friction = friction;
    p.drag = 0;
    p.attraction = 0;
    p.gravityZ = -kGravity;
    p.colliderThickness = kThickness;
    p.bodyMass = 1;
    p.orientationDegrees = {tiltDegrees, 0.0f, 0.0f};
    p.angularVelocity = {};
    p.angularDrag = 0;
    auto binding = std::make_shared<ParticleGeometryBinding>();
    binding->scale = 1;
    binding->centerX = binding->centerY = binding->centerZ = 0;
    binding->mesh = std::move(ground);
    binding->body = cubeAppearance();
    p.geometryBinding = binding;
    return p;
}

// Replays the production historical path at one project time. Each call builds
// fresh replay state, so the result is a pure function of the time.
ParticleSolidState sample(const ParticleParams& source, double seconds, std::string& error)
{
    ParticleParams p = source;
    p.historyProjectSeconds = seconds;
    ParticleSolidState state;
    error.clear();
    videorender::replayParticleBodies(p, seconds, 1.0f, nullptr, &error, &state);
    return state;
}

bool sameBodies(const ParticleSolidState& a, const ParticleSolidState& b)
{
    if (a.count != b.count) return false;
    for (int i = 0; i < a.count; ++i) {
        const auto& x = a.bodies[static_cast<std::size_t>(i)];
        const auto& y = b.bodies[static_cast<std::size_t>(i)];
        if (videorender::solidPosition(x) != videorender::solidPosition(y)
            || videorender::solidVelocity(x) != videorender::solidVelocity(y)
            || x.orientation != y.orientation || x.angular != y.angular
            || x.identity != y.identity || x.geometryIdentity != y.geometryIdentity)
            return false;
    }
    return true;
}

// Center height of a resting unit cube on the plane z=0, with zero spin.
// Free fall stops when the box face is colliderThickness from the plane, and
// the response then keeps it there with only a 2*tolerance excess. So the
// center sits at half + thickness, within float rounding.
void checkRestingFlat(const ParticleSolidState& state, const char* label)
{
    const auto& b = state.bodies[0];
    check(state.count == 1, label);
    check(closeTo(b.z, double(kHalf) + kThickness, 2.0e-4), "resting center height is half extent plus collider thickness");
    check(closeTo(b.x, 0.0, 1.0e-6) && closeTo(b.y, -5.0, 1.0e-6),
        "resting box keeps its drop column without drift");
    check(closeTo(b.vx, 0.0, 1.0e-6) && closeTo(b.vy, 0.0, 1.0e-6) && closeTo(b.vz, 0.0, 1.0e-3),
        "resting box has no residual velocity");
    check(closeTo(b.orientation[0], 0.0, 1.0e-6) && closeTo(b.orientation[1], 0.0, 1.0e-6)
        && closeTo(b.orientation[2], 0.0, 1.0e-6) && closeTo(b.orientation[3], 1.0, 1.0e-6),
        "resting box keeps the identity orientation");
    check(closeTo(b.angular[0], 0.0, 1.0e-6) && closeTo(b.angular[1], 0.0, 1.0e-6)
        && closeTo(b.angular[2], 0.0, 1.0e-6), "resting box has no spin");
}
}

int main()
{
    std::string error;

    // Case A: free fall before contact, from depth 1.0 at (0,-5), flat ground.
    // With semi-implicit steps, vz after n steps is -g n dt and z is
    // 1 - g dt^2 n(n+1)/2. Sampling at 0.2 s gives n=24, so z = 1 - 0.2041667
    // = 0.7958333 and vz = -1.96. Contact begins near 0.44 s, so this sample is
    // still in free fall.
    {
        const auto ground = groundAt(0.0f);
        const auto p = makeParams(0.0, -5.0, 1.0, 0.0f, 0.0f, ground);
        const auto fall = sample(p, 0.2, error);
        const double n = 24.0;
        check(error.empty() && fall.count == 1, "free-fall replay admits one box above the ground");
        check(closeTo(fall.bodies[0].z, 1.0 - double(kGravity) * n * (n + 1) / (2.0 * 120.0 * 120.0), 1.0e-3),
            "free fall before contact matches the semi-implicit integrator");
        check(closeTo(fall.bodies[0].vz, -double(kGravity) * n / 120.0, 1.0e-3),
            "free-fall velocity matches -g t");

        // Rest after contact. Flat ground, zero friction. The expected center
        // height is kHalf + kThickness (see checkRestingFlat). The sample at
        // 1.0 s comes after contact at about 0.44 s.
        const auto rest = sample(p, 1.0, error);
        check(error.empty(), "box dropped on flat ground replays through contact without refusal");
        checkRestingFlat(rest, "box dropped on flat ground settles at rest");

        // Seek and replay determinism. The replay is a pure function of time, so
        // seeking away and back must reproduce bit-identical bodies.
        const auto early = sample(p, 0.25, error);
        const auto far = sample(p, 3.5, error);
        const auto back = sample(p, 1.0, error);
        check(error.empty() && early.count == 1 && far.count == 1,
            "seek samples admit the box at every probed time");
        check(sameBodies(rest, back), "seek back to the rest time reproduces the same bodies bit for bit");
        const auto repeat = sample(p, 1.0, error);
        check(sameBodies(back, repeat), "repeated replay at one time is bit-identical");
    }

    // Case C: the drop height does not change the rest height. Same ground, drop
    // from depth 0.5, which contacts at about 0.31 s. The rest sample must
    // match case A to 2e-4.
    {
        const auto ground = groundAt(0.0f);
        const auto p = makeParams(0.0, -5.0, 0.5, 0.0f, 0.0f, ground);
        const auto rest = sample(p, 1.0, error);
        check(error.empty(), "shorter drop onto flat ground replays through contact");
        checkRestingFlat(rest, "shorter drop settles at the same rest height");
    }

    // Case D: a box resting on a tilted plane is refused, not silently replayed.
    // Plane normal n=(0,-sin t, cos t), and the box is oriented flat on it by a
    // quaternion about X with the same tilt. Gravity has a tangent component of
    // g sin t, so friction 0.5 acts on contact and its torque about the face
    // arm -h n is nonzero. The next step then changes the orientation, which
    // is a spin about a non-coordinate normal. The solver refuses that interval
    // with "Rotating box triangle sweeps are unsupported". Tilt 20 degrees and a
    // start height of kHalf + kThickness + 0.2 above the plane.
    {
        const float tilt = 20.0f;
        const double angle = double(tilt) * kPi / 180.0;
        const double c = std::cos(angle), s = std::sin(angle);
        const double clearance = double(kHalf) + kThickness + 0.2;
        // Plane point under base (0,-5) plus clearance along n.
        const double ax = 0.0;
        const double ay = -5.0 * c - clearance * s;
        const double az = -5.0 * s + clearance * c;
        const auto ground = groundAt(tilt);
        const auto p = makeParams(ax, ay, az, 0.5f, tilt, ground);
        std::string tiltError;
        sample(p, 1.0, tiltError);
        check(tiltError.find("Rotating box") != std::string::npos,
            "a box at rest on a tilted triangle surface is refused by the spin refusal");
    }

    if (failures == 0) std::printf("solid box rest contact proofs passed\n");
    return failures == 0 ? 0 : 1;
}
