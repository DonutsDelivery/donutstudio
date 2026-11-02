// A render tile flushes denormals on every thread that runs one, and the caller gets its
// own mode back, even from a tile that throws. x86 only, as the pool sets it only there.

#include "pathtrace/thread_pool.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>

namespace
{
namespace pt = videohelper::pathtrace;

constexpr unsigned kFlushBits = 0x8040;
bool flushed() { return (_mm_getcsr() & kFlushBits) == kFlushBits; }

int failures = 0;
void check (bool condition, const char* message)
{
    if (! condition)
    {
        ++failures;
        std::printf ("FAIL: %s\n", message);
    }
}

// Runs a pass and reports whether every tile was flushed, and on how many threads.
void checkPool (pt::ThreadPool& pool, const char* name)
{
    const pt::CancellationToken cancel;
    std::atomic<int> unflushed { 0 };
    std::mutex mutex;
    std::set<std::thread::id> threads;
    const auto caller = std::this_thread::get_id();
    bool callerTile = false;
    pool.runTiles (pt::TileGrid (512, 512, 8), cancel, [&] (const pt::Tile&, unsigned) {
        if (! flushed())
            ++unflushed;
        const std::lock_guard<std::mutex> lock (mutex);
        threads.insert (std::this_thread::get_id());
        callerTile = callerTile || std::this_thread::get_id() == caller;
    });
    std::printf ("%s: %zu threads ran tiles, caller among them %d\n", name, threads.size(), callerTile ? 1 : 0);
    check (unflushed == 0, "a tile ran without flushing denormals");
    check (callerTile, "the caller's seat was not exercised");
    check (! flushed(), "the caller did not get its mode back");
}
} // namespace

int main()
{
    check (! flushed(), "the test must start unflushed");

    pt::ThreadPool one (1);
    checkPool (one, "one thread");
    pt::ThreadPool four (4);
    checkPool (four, "four threads");
    checkPool (pt::sharedThreadPool(), "shared pool");

    const pt::CancellationToken cancel;
    bool nestedFlushed = false;
    four.runTiles (pt::TileGrid (8, 8, 8), cancel, [&] (const pt::Tile&, unsigned) {
        four.runTiles (pt::TileGrid (8, 8, 8), cancel, [&] (const pt::Tile&, unsigned) { nestedFlushed = flushed(); });
        check (flushed(), "a nested pass ended the outer tile's flush");
    });
    check (nestedFlushed, "a nested tile ran without flushing denormals");
    check (! flushed(), "the caller did not get its mode back after a nested pass");

    try
    {
        pt::ThreadPool alone (1);
        alone.runTiles (pt::TileGrid (8, 8, 8), cancel, [] (const pt::Tile&, unsigned) { throw std::runtime_error ("tile"); });
        check (false, "the tile's exception did not reach the caller");
    }
    catch (const std::runtime_error&)
    {
    }
    check (! flushed(), "the caller did not get its mode back from a tile that threw");

    std::printf ("%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
#else
int main()
{
    std::printf ("no denormal flush on this architecture\n");
    return 0;
}
#endif
