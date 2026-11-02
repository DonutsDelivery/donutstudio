// The process-wide tile pool: a default PathTracer renders on it without adding
// threads, concurrent passes stay within the core count, a pass submitted from a
// tile finishes, and its films equal a private pool's bit for bit.

#include "pathtrace/integrator.h"
#include "pathtrace/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
using pt::Vec3;

std::atomic<int> checks { 0 };
std::atomic<int> failures { 0 };

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

// Threads in this process, or 0 where /proc does not exist.
int processThreadCount()
{
    std::ifstream status ("/proc/self/status");
    std::string key;
    while (status >> key)
    {
        if (key == "Threads:")
        {
            int count = 0;
            status >> count;
            return count;
        }
        std::getline (status, key);
    }
    return 0;
}

void appendQuad (pt::RenderScene& scene, Vec3 centre, float half, std::uint32_t material)
{
    const Vec3 p[4] = { { centre.x - half, centre.y - half, centre.z }, { centre.x + half, centre.y - half, centre.z },
                        { centre.x + half, centre.y + half, centre.z }, { centre.x - half, centre.y + half, centre.z } };
    const std::uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    for (int t = 0; t < 2; ++t)
    {
        pt::TriangleAttributes attributes;
        attributes.materialIndex = material;
        attributes.primitiveIndex = static_cast<std::uint32_t> (t);
        attributes.geometricNormal = { 0.0f, 0.0f, 1.0f };
        for (int k = 0; k < 3; ++k)
        {
            scene.geometry.positions.push_back (p[indices[t * 3 + k]]);
            attributes.shadingNormals[static_cast<std::size_t> (k)] = attributes.geometricNormal;
            attributes.tangents[static_cast<std::size_t> (k)] = { 1.0f, 0.0f, 0.0f, 1.0f };
        }
        scene.geometry.triangles.push_back (attributes);
    }
}

constexpr std::uint32_t kWidth = 48;
constexpr std::uint32_t kHeight = 32;
constexpr std::uint32_t kSamples = 16;
constexpr std::uint64_t kSeed = 5;

void buildScene (pt::RenderScene& scene)
{
    pt::Material wall;
    wall.baseColor = { 0.7f, 0.5f, 0.3f };
    wall.roughness = 0.6f;
    scene.materials.push_back (wall);
    appendQuad (scene, { 0.0f, 0.0f, 0.0f }, 1.0f, 0);
    pt::Material lamp;
    lamp.emission = { 4.0f, 3.0f, 2.0f };
    scene.materials.push_back (lamp);
    appendQuad (scene, { 0.3f, 0.2f, -0.5f }, 0.6f, 1);
    grid::SceneLightRecord environment;
    environment.kind = grid::SceneLightKind::Environment;
    environment.color = { 0.25f, 0.25f, 0.25f };
    environment.intensity = 1.0f;
    scene.lightRecords.push_back (environment);
    scene.commit();
}

pt::PinholeCamera makeCamera()
{
    pt::PinholeCamera camera;
    camera.position = { 0.1f, -0.2f, 3.0f };
    camera.tanHalfFov = 0.5f;
    camera.aspect = static_cast<float> (kWidth) / static_cast<float> (kHeight);
    camera.nearPlane = 0.01f;
    camera.farPlane = 1000.0f;
    camera.width = kWidth;
    camera.height = kHeight;
    return camera;
}

std::vector<float> renderFilm (pt::PathTracer& tracer, const pt::RenderScene& scene, const pt::PinholeCamera& camera)
{
    const pt::CancellationToken cancel {};
    tracer.renderFixed (scene, camera, kWidth, kHeight, kSamples, kSeed, cancel);
    return tracer.accumulator().resolveRgba32f();
}

bool sameFilm (const std::vector<float>& a, const std::vector<float>& b)
{
    return ! a.empty() && a.size() == b.size() && std::memcmp (a.data(), b.data(), a.size() * sizeof (float)) == 0;
}

void filmTests()
{
    pt::RenderScene scene;
    buildScene (scene);
    const pt::PinholeCamera camera = makeCamera();

    pt::PathTracer privateTracer (1);
    const std::vector<float> reference = renderFilm (privateTracer, scene, camera);

    pt::PathTracer sharedA;
    pt::PathTracer sharedB;
    check (sameFilm (renderFilm (sharedA, scene, camera), reference), "the shared pool's film equals a one-thread film");

    // Two tracers at once take turns on the same workers, as preview and export do.
    std::vector<float> filmA;
    std::vector<float> filmB;
    std::thread other ([&] { filmB = renderFilm (sharedB, scene, camera); });
    filmA = renderFilm (sharedA, scene, camera);
    other.join();
    check (sameFilm (filmA, reference), "the first of two concurrent shared passes keeps its film");
    check (sameFilm (filmB, reference), "the second of two concurrent shared passes keeps its film");
}

void threadCountTests()
{
    const unsigned cores = std::max (1u, std::thread::hardware_concurrency());
    const int before = processThreadCount();
    pt::ThreadPool& shared = pt::sharedThreadPool();
    check (shared.concurrency() == cores, "the shared pool has one thread per core");
    check (&pt::sharedThreadPool() == &shared, "every caller gets the same pool");

    const int started = processThreadCount();
    if (before != 0)
        check (started - before <= static_cast<int> (cores) - 1, "the shared pool starts cores - 1 workers at most");

    pt::PathTracer first;
    pt::PathTracer second;
    check (first.concurrency() == cores && second.concurrency() == cores, "default tracers report the core count");
    if (before != 0)
        check (processThreadCount() == started, "default tracers add no threads");
    check (pt::PathTracer (3).concurrency() == 3, "an explicit count keeps a private pool");

    // Three passes at once, the shape of two engines and a denoise.
    const pt::TileGrid grid (256, 256, 8);
    std::atomic<int> running { 0 };
    std::atomic<int> peak { 0 };
    std::atomic<std::uint32_t> ran { 0 };
    std::atomic<bool> badIndex { false };
    std::vector<std::atomic<int>> seatsInUse (shared.concurrency());
    const auto tile = [&] (const pt::Tile&, unsigned threadIndex) {
        const int now = running.fetch_add (1) + 1;
        int seen = peak.load();
        while (now > seen && ! peak.compare_exchange_weak (seen, now)) {}
        if (threadIndex >= cores || seatsInUse[threadIndex].fetch_add (1) != 0)
            badIndex = true;
        std::this_thread::sleep_for (std::chrono::microseconds (100));
        if (threadIndex < cores)
            seatsInUse[threadIndex].fetch_sub (1);
        running.fetch_sub (1);
        ran.fetch_add (1);
    };
    const auto pass = [&] {
        const pt::CancellationToken cancel {};
        const auto result = shared.runTiles (grid, cancel, tile);
        check (result.tilesCompleted == grid.tileCount() && ! result.cancelled, "a concurrent pass runs every tile");
    };
    std::thread a (pass);
    std::thread b (pass);
    pass();
    a.join();
    b.join();
    check (ran.load() == 3 * grid.tileCount(), "three concurrent passes run every tile once");
    check (peak.load() <= static_cast<int> (cores), "concurrent passes never run more tiles than cores, peak "
                                                        + std::to_string (peak.load()));
    check (! badIndex.load(), "no two running tiles share a threadIndex");
}

void cancelWhileWaitingTest()
{
    // The worker and the caller seat are both on another pass, so the cancelled
    // pass never runs a tile. The next tile to finish must wake its caller.
    pt::ThreadPool pool (2);
    const pt::TileGrid busyGrid (64, 64, 8);
    const pt::CancellationToken neverCancelled {};
    std::atomic<bool> release { false };
    std::atomic<int> started { 0 };
    std::thread busy ([&] {
        pool.runTiles (busyGrid, neverCancelled, [&] (const pt::Tile&, unsigned) {
            started.fetch_add (1);
            while (! release.load())
                std::this_thread::sleep_for (std::chrono::milliseconds (1));
        });
    });
    while (started.load() < 2)
        std::this_thread::sleep_for (std::chrono::milliseconds (1));

    pt::CancellationToken cancel;
    std::atomic<bool> returned { false };
    std::thread waiting ([&] {
        const auto result = pool.runTiles (pt::TileGrid (64, 64, 8), cancel, [] (const pt::Tile&, unsigned) {});
        check (result.cancelled, "a pass cancelled before it ran reports cancelled");
        returned = true;
    });
    cancel.cancel();
    release = true;
    busy.join();
    waiting.join();
    check (returned.load(), "a caller whose pass was cancelled while it slept returns");
}

void nestedTests()
{
    for (const unsigned threads : { 1u, 2u, 4u })
    {
        const std::string label = "pool(" + std::to_string (threads) + ")";
        pt::ThreadPool pool (threads);
        const pt::TileGrid outer (64, 64, 16);
        const pt::TileGrid inner (32, 32, 8);
        const pt::CancellationToken cancel {};
        std::atomic<std::uint32_t> innerTiles { 0 };
        std::atomic<bool> indexMoved { false };
        const auto result = pool.runTiles (outer, cancel, [&] (const pt::Tile&, unsigned outerIndex) {
            pool.runTiles (inner, cancel, [&] (const pt::Tile&, unsigned innerIndex) {
                innerTiles.fetch_add (1);
                if (innerIndex != outerIndex)
                    indexMoved = true;
            });
        });
        check (result.tilesCompleted == outer.tileCount(), label + ": a pool job submitting to its own pool finishes");
        check (innerTiles.load() == outer.tileCount() * inner.tileCount(), label + ": every nested tile runs");
        check (! indexMoved.load(), label + ": a nested pass keeps the thread's index");

        bool thrown = false;
        try
        {
            pool.runTiles (outer, cancel, [&] (const pt::Tile&, unsigned) {
                pool.runTiles (inner, cancel, [] (const pt::Tile&, unsigned) { throw std::runtime_error ("nested"); });
            });
        }
        catch (const std::runtime_error&)
        {
            thrown = true;
        }
        check (thrown, label + ": a nested exception reaches the outer caller");
        check (pool.runTiles (inner, cancel, [] (const pt::Tile&, unsigned) {}).tilesCompleted == inner.tileCount(),
               label + ": the pool still runs after a nested exception");
    }
}

} // namespace

int main()
{
    threadCountTests();
    filmTests();
    cancelWhileWaitingTest();
    nestedTests();
    std::cout << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
