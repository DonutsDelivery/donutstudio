// Coupled water checkpoints. Bodies live for the particle lifetime; after that the
// water keeps stepping on its own. A late or sought frame resumes from the latest
// water checkpoint at or before it and must match a replay from project zero bit
// for bit. Header-only, like the other particle body replay targets.
#include "../src/particle_body_replay.h"
#include "water_surface_replay_fixture.h"
#include "../../shared/GeometryFieldLowering.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace videorender;

namespace
{
int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct Setup
{
    videowire::geometry::PortContract contract;
    std::optional<videowire::geometry::ValueDescriptor> rest, water;
    std::shared_ptr<ParticleHistorySource> history;
    ParticleParams p;
};

// One 9 by 9 positive-mass Water Surface Simulation, one sphere dropped onto it
// and score notes that keep exciting the water after the body window ends.
// Other grid sizes, square or not, couple the same way.
std::optional<Setup> makeSetup(std::string& error, std::uint32_t columns = 9, std::uint32_t rows = 9)
{
    using namespace videowire::geometry;
    Setup setup;
    setup.contract.carrier = CarrierKind::geometry3D;
    setup.contract.maxVertices = std::max<std::uint32_t>(256, columns * rows);
    setup.contract.maxIndices = std::max<std::uint32_t>(384, 6 * (columns - 1) * (rows - 1));
    setup.rest = lowerGrid(18000, 18000, 1, columns, rows, 2.0f / float(std::max(columns, rows) - 1),
                           setup.contract, error);
    setup.water = setup.rest
        ? lowerWaterSurfaceSimulation(*setup.rest, 18001, 18002, 1.5, 0.2, 0.5, 0, setup.contract, error, 100)
        : std::nullopt;
    if (!setup.water) return std::nullopt;
    auto source = waterreplayfixture::source();
    auto score = std::make_shared<arbitmod::Score>(*source.score);
    for (const auto& [id, beat, pitch] : std::vector<std::tuple<int, float, float>>{
             {1, 4, 40}, {2, 9, 70}, {3, 15, 55}, {4, 19.5f, 90}})
        score->notes.push_back(waterreplayfixture::note(id, beat, pitch));
    source.score = score;
    setup.history = std::make_shared<ParticleHistorySource>(source);
    auto binding = std::make_shared<ParticleGeometryBinding>(); binding->scale = .25f;
    binding->mesh = std::make_shared<const ValueDescriptor>(*setup.water);
    ValueDescriptor emitter;
    emitter.carrier = CarrierKind::points3D; emitter.stableId = 18100;
    emitter.sourceStableId = 18100; emitter.sourceRevision = 1;
    emitter.data = PointsData{};
    std::get<PointsData>(emitter.data).points.push_back({{-.4f, -.2f, .12f}, 18101, 1, {0, 0, 0, 1}});
    binding->emitters = std::make_shared<const ValueDescriptor>(emitter);
    auto& p = setup.p;
    p.motionMode = 2; p.simulationSpace = 1; p.historicalReplay = true;
    p.history = setup.history; p.geometryBinding = binding; p.count = 1; p.lifetime = .5f;
    p.force = 0; p.impulseZ = -4; p.bodyRadius = .015f; p.colliderThickness = 0;
    p.restitution = 1; p.bodyMass = 1; p.resetMode = 1;
    return setup;
}

struct Frame
{
    bool ok = false;
    std::string error;
    ParticleBodyState poses {};
    ParticleSolidState solids;
    std::vector<double> velocity;
    WaterCoupledFrame water;
};

Frame replayAt(const Setup& setup, std::int64_t frameIndex, WaterCoupledCheckpointStore* store,
               ParticleParams p = {})
{
    if (!p.history) p = setup.p;
    const double seconds = double(frameIndex) / 60;
    p.historyProjectSeconds = seconds;
    const auto canonical = waterreplayfixture::frame(*setup.history, seconds, 60);
    Frame result;
    result.ok = replayWaterCoupledBodies(p, seconds, 1, canonical.get(), result.poses, result.solids, nullptr,
                                         result.error, &result.velocity, &result.water, store);
    if (!result.ok) std::fprintf(stderr, "frame %lld: %s\n", static_cast<long long>(frameIndex), result.error.c_str());
    return result;
}

bool identical(const Frame& a, const Frame& b)
{
    return a.ok && b.ok && a.poses == b.poses && a.solids.count == b.solids.count
        && a.solids.colliders.meshVertexCount == b.solids.colliders.meshVertexCount
        && a.solids.colliders.meshVertices == b.solids.colliders.meshVertices
        && a.velocity == b.velocity && a.water.heights == b.water.heights
        && a.water.projectSeconds == b.water.projectSeconds;
}

void frame600ResumesFromCheckpoint()
{
    std::string error;
    const auto setup = makeSetup(error);
    check(setup.has_value(), "the coupled water fixture lowers");
    if (!setup) return;
    const auto full = replayAt(*setup, 600, nullptr);
    check(full.ok, "a replay from project zero reaches frame 600, past the body window");
    check(full.ok && std::any_of(full.water.heights.begin(), full.water.heights.end(),
                                 [](double height) { return height != 0; }),
          "the water still moves at frame 600");
    WaterCoupledCheckpointStore store;
    const auto earlier = replayAt(*setup, 599, &store);
    check(earlier.ok && store.recorded() >= 9 && store.resumes() == 0,
          "the first late frame replays from zero and records a checkpoint every simulated second");
    const auto resumed = replayAt(*setup, 600, &store);
    check(store.resumes() == 1, "frame 600 resumes from a checkpoint");
    check(identical(full, resumed), "frame 600 from a checkpoint matches the full replay bit for bit");
    const auto sought = replayAt(*setup, 300, &store);
    check(store.resumes() == 2 && identical(replayAt(*setup, 300, nullptr), sought),
          "a backward seek resumes from an earlier checkpoint and matches the full replay");
    const auto inside = replayAt(*setup, 15, &store);
    check(store.resumes() == 2 && identical(replayAt(*setup, 15, nullptr), inside),
          "a frame inside the body window replays in full");
    const auto later = replayAt(*setup, 1800, &store);
    check(store.resumes() == 3 && identical(replayAt(*setup, 1800, nullptr), later),
          "frame 1800 evaluates past any fixed length and matches the full replay");
    auto changed = setup->p; changed.impulseZ = -3;
    const auto replanned = replayAt(*setup, 600, &store, changed);
    check(store.resumes() == 3 && identical(replayAt(*setup, 600, nullptr, changed), replanned),
          "a control change drops the old checkpoints and replays from zero");
}
// Water without bodies resumes from the same per-second snapshots.
void uncoupledWaterResumesFromCheckpoint()
{
    using namespace videorender;
    auto history = waterreplayfixture::source();
    auto score = std::make_shared<arbitmod::Score>(*history.score);
    for (const float beat : {4.0f, 9.0f, 15.0f, 19.5f})
        score->notes.push_back(waterreplayfixture::note(int(beat * 2) + 1, beat, 64));
    history.score = score;
    const auto rest = waterreplayfixture::mesh(9, 9);
    const auto operation = waterreplayfixture::operation(1.5, 0.2, 0.5, 0);
    videowire::geometry::WaterSurfaceGrid grid;
    std::string error;
    check(videowire::geometry::validateWaterSurfaceGrid(rest, grid, error), "the 9 by 9 rest grid validates");
    WaterSurfaceCheckpointStore store;
    std::vector<double> full, warm, resumed, seekBack;
    check(replayWaterSurface(history, operation, rest, grid, 599.0 / 60, warm, error, &store)
              && store.recorded() >= 9 && store.resumes() == 0,
          "frame 599 records one checkpoint per simulated second");
    check(replayWaterSurface(history, operation, rest, grid, 10.0, full, error, nullptr)
              && replayWaterSurface(history, operation, rest, grid, 10.0, resumed, error, &store)
              && store.resumes() == 1 && full == resumed,
          "frame 600 resumes and matches a full replay bit for bit");
    std::vector<double> fullBack;
    check(replayWaterSurface(history, operation, rest, grid, 5.0, fullBack, error, nullptr)
              && replayWaterSurface(history, operation, rest, grid, 5.0, seekBack, error, &store)
              && store.resumes() == 2 && fullBack == seekBack,
          "a seek back to frame 300 resumes and matches");
    check(std::any_of(full.begin(), full.end(), [](double h) { return h != 0; }), "the score moved the water by frame 600");
    auto changed = operation; changed.field.values[1] = 0.3;
    std::vector<double> changedFull, changedResumed;
    check(replayWaterSurface(history, changed, rest, grid, 10.0, changedFull, error, nullptr)
              && replayWaterSurface(history, changed, rest, grid, 10.0, changedResumed, error, &store)
              && store.resumes() == 2 && changedFull == changedResumed && changedFull != full,
          "a damping change drops the old checkpoints and replays from zero");
}
// A 24 by 16 grid (384 vertices, 690 triangles) is past the old 9 by 9 and
// 256-vertex bounds, and 20 seconds is past the old 10-second bound.
void largeRectangularGridCouples()
{
    std::string error;
    const auto setup = makeSetup(error, 24, 16);
    check(setup.has_value(), "a 24 by 16 water grid lowers");
    if (!setup) return;
    const auto frame = replayAt(*setup, 1200, nullptr);
    check(frame.ok, "a body couples to a 24 by 16 grid at 20 seconds");
    check(frame.ok && frame.water.heights.size() == 24u * 16u, "the coupled heights size from the grid");
    WaterCoupledCheckpointStore store;
    const auto warm = replayAt(*setup, 1199, &store);
    const auto resumed = replayAt(*setup, 1200, &store);
    check(warm.ok && store.resumes() == 1 && identical(frame, resumed),
          "the large grid resumes from a checkpoint bit for bit");
}
} // namespace

int main()
{
    frame600ResumesFromCheckpoint();
    uncoupledWaterResumesFromCheckpoint();
    largeRectangularGridCouples();
    if (failures != 0) {
        std::fprintf(stderr, "%d coupled water checkpoint check(s) failed\n", failures);
        return 1;
    }
    std::printf("coupled water checkpoint tests passed\n");
    return 0;
}
