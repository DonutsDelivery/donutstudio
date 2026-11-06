// Content-sized coupled bodies. Planar replay held 64 bodies and Solid 3D
// 128, and planar emitters held 64 anchors. Body state, anchors and the
// planar shader's anchor/body texture are now sized from Count and the
// emitter points, so 300 bodies and 300 anchors replay and pack.

#include "../src/particle_body_replay.h"
#include "../src/particle_timeline_data.h"

#include <cstdio>
#include <limits>
#include <memory>
#include <string>

using namespace videorender;

namespace
{
int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

// 300 emitter anchors on a grid inside the unit box, each its own body.
ParticleParams anchoredBodies(int simulationSpace)
{
    ParticleParams p;
    p.motionMode = 2; p.simulationSpace = simulationSpace; p.historicalReplay = true;
    auto history = std::make_shared<ParticleHistorySource>();
    history->score = std::make_shared<const arbitmod::Score>();
    auto& key = history->key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = key.backendGeneration = key.deviceGeneration = 1;
    key.scoreGeneration = key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    p.history = history;
    p.count = 300; p.geometryCount = 300; p.lifetime = 2; p.force = 0; p.bodyRadius = 0.002f;
    for (int i = 0; i < 300; ++i)
    {
        const auto at = static_cast<std::size_t>(i);
        const float x = 0.1f + 0.04f * float(i % 20), y = 0.1f + 0.05f * float(i / 20);
        p.geometryAnchors[at] = {x, y, x, y};
        p.geometryIdentities[at] = 9007199254740001ull + at;
        p.geometryDepths[at] = {0.5f, 0.5f};
    }
    return p;
}
}

int main()
{
    // An unread body slot reads as a dead body, as the filled array did.
    const ParticleBodyState none;
    check(none[299] == std::array<float, 4>{-10, -10, 0, 0} && none.size() == 0,
          "a body past the list reads dead without growing it");

    auto planar = anchoredBodies(0);
    check(planar.geometryAnchors.size() == 300 && planar.geometryIdentities[299] == 9007199254740001ull + 299,
          "300 anchors fit past the old 64-entry emitter arrays");
    std::string error;
    planar.historyProjectSeconds = 0.25;
    const auto bodies = replayParticleBodies(planar, 0.25, 1, nullptr, &error);
    check(error.empty() && bodies.size() == 300, "planar replay sizes 300 bodies from Count");
    int live = 0;
    for (const auto& body : bodies) if (body[0] > -1) ++live;
    check(live == 300, "every one of 300 planar anchors births a body");

    const auto data = packParticleTimelineData(planar, &bodies);
    const auto entry = [&](int index)
    {
        const auto at = static_cast<std::size_t>(index) * 4;
        return std::array<float, 4>{data.values[at], data.values[at + 1], data.values[at + 2], data.values[at + 3]};
    };
    check(data.anchors == 300 && data.bodies == 300 && data.rows == 1
              && data.values.size() == std::size_t(kParticleTimelineDataWidth) * 4,
          "300 anchors and 300 bodies pack into one 1024-entry row");
    check(entry(299) == planar.geometryAnchors[299] && entry(300) == bodies[0] && entry(599) == bodies[299]
              && entry(600) == std::array<float, 4>{},
          "anchors come first, then bodies, then zero padding");

    auto wide = planar; wide.geometryCount = 900;
    for (int i = 300; i < 900; ++i) wide.geometryAnchors[static_cast<std::size_t>(i)] = {0.5f, 0.5f, 0.5f, 0.5f};
    const auto wrapped = packParticleTimelineData(wide, &bodies);
    check(wrapped.rows == 2 && wrapped.values.size() == 2 * std::size_t(kParticleTimelineDataWidth) * 4,
          "1200 entries wrap onto a second texture row");
    check(packParticleTimelineData(ParticleParams {}, nullptr).rows == 1,
          "no anchors and no bodies still hold one zero row");

    auto solid = anchoredBodies(1);
    solid.historyProjectSeconds = 0.25;
    ParticleSolidState state;
    replayParticleBodies(solid, 0.25, 1, nullptr, &error, &state);
    check(error.empty() && state.count == 300 && state.bodies.size() >= 300
              && state.bodies[299].geometryIdentity == 9007199254740001ull + 299,
          "Solid 3D replay holds 300 bodies past the old 128-body array");

    const auto perBody = ParticleBodyReplayState::callerScratchBytesPerBody();
    check(ParticleBodyReplayState::callerScratchLogicalBytes(300, 300)
              == ParticleBodyReplayState::callerScratchLogicalBytes(0, 0) + 300 * perBody
                     + ParticleBodyReplayState::callerAnchorBytes(300),
          "the caller scratch charges each body and anchor it holds");
    check(ParticleBodyReplayState::callerOutputLogicalBytes(true, true, 300)
              > ParticleBodyReplayState::callerOutputLogicalBytes(true, true, 128),
          "the publication charge grows with the body count");

    // Triangle replay work has no fixed ceiling: 300 bodies over 600 steps
    // against a 20,000-triangle mesh is far past the old 100M units.
    SolidTriangleWorkBudget work;
    check(work.limit == std::numeric_limits<std::uint64_t>::max()
              && work.preflight(600, 300, 10000, 20000, 300, 96),
          "a large Solid 3D replay is admitted without a work ceiling");
    check(work.charge(1000000000ull) && work.used == 1000000000ull,
          "a billion checked units charge without refusal");
    SolidTriangleWorkBudget explicitLimit; explicitLimit.limit = 10;
    check(!explicitLimit.charge(11) && explicitLimit.used == 0, "an explicit caller limit still refuses past itself");
    check(!work.preflight(std::numeric_limits<std::uint64_t>::max(), 2, 3, 1, 0),
          "only work that cannot be counted in 64 bits refuses");

    // Body geometry sizes from its mesh: a 600-vertex, 1200-triangle retained
    // body is past the old 256-vertex and 384-index limits.
    {
        using videowire::geometry::OperationCode;
        ParticleBodyGeometryScalarCensus census;
        census.operations.resize(2);
        auto& root = census.operations[0];
        root.code = OperationCode::retainedMesh; root.emptyPayloads = true;
        root.vertices = root.retainedVertices = 600; root.indices = root.retainedIndices = 3600;
        auto& timeline = census.operations[1];
        timeline.code = OperationCode::fieldTimeline; timeline.emptyPayloads = true; timeline.fields = 600;
        census.operationCount = 2; census.terminalVertices = 600; census.terminalIndices = 3600;
        ParticleBodyGeometryScalarCost cost;
        auto refusal = ParticleBodyGeometryCostRefusal::none;
        check(ParticleBodyGeometryAdmission::deriveScalarCost(census, 10, ParticleBodyGeometryProxy::sphere, cost, refusal)
                  && cost.terminalVertices == 600 && cost.terminalIndices == 3600,
              "a 600-vertex body schedule costs from its mesh");
        auto broken = census; broken.terminalIndices = 3601;
        check(!ParticleBodyGeometryAdmission::deriveScalarCost(broken, 10, ParticleBodyGeometryProxy::sphere, cost, refusal)
                  && refusal == ParticleBodyGeometryCostRefusal::unsupported,
              "a partial triangle is still refused");
        using rotatingexact::Cost;
        check(rotatingplane::ResponseCost::interval(40, 600, 1200, 4) != Cost::unavailable
                  && rotatingplane::ResponseCost::interval(40, 600, 1200, 4)
                         > rotatingplane::ResponseCost::interval(40, 256, 128, 4),
              "rotating-plane contact work grows with the mesh past 256 vertices");
        check(work.rotatingWidth(60, 2, 600, 1200, 4) != 0, "a 600-vertex rotating body keeps an exact width");
        // Point colliders size from the scene: 300 points are charged, not refused.
        check(rotatingplane::ResponseCost::interval(40, 600, 1200, 300) != Cost::unavailable
                  && rotatingplane::ResponseCost::interval(40, 600, 1200, 300)
                         > rotatingplane::ResponseCost::interval(40, 600, 1200, 64),
              "rotating-plane guards grow with 300 point colliders past 64");
        check(work.preflight(60, 2, 600, 1200, 300, 96) && work.rotatingWidth(60, 2, 600, 1200, 300) != 0,
              "a body with 300 point colliders admits a rotating width");
        check(rotatingplane::ResponseCost::interval(40, 600, 1200, std::numeric_limits<std::uint64_t>::max()) == Cost::unavailable
                  && !work.preflight(60, 2, 600, 1200, std::numeric_limits<std::uint64_t>::max(), 96),
              "only a point count whose work leaves the uint64 range is unavailable");
        check(ParticleBodyGeometryLimits{}.maxReplayBoundaries == std::numeric_limits<std::size_t>::max(),
              "replay boundaries default to the size_t range, not 100000");
        constexpr auto sizeRange = std::numeric_limits<std::size_t>::max();
        const ParticleBodyGeometryLimits geometry;
        check(geometry.maxCensusVisits == sizeRange && geometry.maxFactoryVisits == sizeRange
                  && geometry.maxReplayVisits == sizeRange,
              "census, factory and replay visits default to the size_t range, not 100000, 1e6 and 1e9");
        const ParticleBodyGeometrySampleLimits sample;
        check(sample.maxDerivationVisits == sizeRange && sample.maxTotalVisits == sizeRange,
              "sampler derivation and total visits default to the size_t range");
        check(ParticleBodyCallerLimits{}.maxWork == sizeRange, "caller work defaults to the size_t range, not 1e12");
        check(ParticleBodyCallerLimits{}.maxLogicalBytes == sizeRange, "caller bytes default to the size_t range, not 256 MB");
    }

    if (failures != 0)
    {
        std::fprintf(stderr, "%d particle body check(s) failed\n", failures);
        return 1;
    }
    return 0;
}
