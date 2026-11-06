#pragma once

// Fixed worker pool that renders an image as square tiles. Each tile of a pass
// runs exactly once unless the pass is cancelled. Cancellation stops new tiles
// from starting; tiles already running finish, so a viewport restart waits for
// at most one tile per thread. Several passes can be live at once. Workers take
// turns between them, and at most concurrency() threads run tiles at any moment.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace videohelper::pathtrace
{

// Shared stop flag. One owner calls cancel(); any number of passes poll it.
// reset() re-arms it once the cancelled pass has returned.
//
// It can also watch a generation counter: once the counter moves past the
// value captured with the pass's inputs, the pass is obsolete and stops at the
// same polls (render_input_generation.h). The owner sets the watch between
// passes, never while one runs.
class CancellationToken
{
public:
    void cancel() noexcept { cancelled_.store (true, std::memory_order_release); }
    void reset() noexcept { cancelled_.store (false, std::memory_order_release); }
    bool isCancelled() const noexcept { return cancelled_.load (std::memory_order_acquire) || obsolete(); }

    void watch (const std::atomic<std::uint64_t>* generation, std::uint64_t captured) noexcept
    {
        generation_ = generation;
        captured_ = captured;
    }
    // The watched generation moved: the inputs this pass renders were edited.
    bool obsolete() const noexcept
    {
        return generation_ != nullptr && generation_->load (std::memory_order_acquire) != captured_;
    }

private:
    std::atomic<bool> cancelled_ { false };
    const std::atomic<std::uint64_t>* generation_ = nullptr;
    std::uint64_t captured_ = 0;
};

// Half-open pixel rectangle [x0, x1) x [y0, y1).
struct Tile
{
    std::uint32_t index = 0;
    std::uint32_t x0 = 0;
    std::uint32_t y0 = 0;
    std::uint32_t x1 = 0;
    std::uint32_t y1 = 0;
};

// Row-major tiles, top row first. Edge tiles are clipped to the image.
struct TileGrid
{
    static constexpr std::uint32_t kDefaultTileSize = 32;

    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t tileSize = kDefaultTileSize;

    TileGrid() = default;
    TileGrid (std::uint32_t w, std::uint32_t h, std::uint32_t size = kDefaultTileSize) noexcept
        : width (w), height (h), tileSize (size == 0 ? 1 : size) {}

    std::uint32_t tilesX() const noexcept { return (width + tileSize - 1) / tileSize; }
    std::uint32_t tilesY() const noexcept { return (height + tileSize - 1) / tileSize; }
    std::uint32_t tileCount() const noexcept { return tilesX() * tilesY(); }
    Tile tile (std::uint32_t index) const noexcept;
};

struct TilePassResult
{
    std::uint32_t tileCount = 0;
    std::uint32_t tilesCompleted = 0;
    bool cancelled = false;  // the token stopped the pass before every tile ran
};

class ThreadPool
{
public:
    // threadCount counts every thread that renders, including the caller of
    // runTiles. 0 picks std::thread::hardware_concurrency(). 1 spawns no
    // workers and runs passes on the caller.
    explicit ThreadPool (unsigned threadCount = 0);
    ~ThreadPool();
    ThreadPool (const ThreadPool&) = delete;
    ThreadPool& operator= (const ThreadPool&) = delete;

    // Number of distinct threadIndex values passed to a tile function.
    unsigned concurrency() const noexcept { return static_cast<unsigned> (workers_.size()) + 1; }

    using TileFunction = std::function<void (const Tile& tile, unsigned threadIndex)>;

    // Runs fn once per tile and blocks until the pass ends. One caller at a time
    // renders its own pass as threadIndex 0; any other caller sleeps until that
    // seat is free or the workers finish its pass. threadIndex is unique among
    // the threads running tiles. If fn throws, the pass stops claiming tiles and
    // runTiles rethrows the first exception after every thread has left the pass.
    // Called from inside a tile of this pool, it runs the new pass on that thread
    // under the same threadIndex, because waiting for the pool's other threads
    // could deadlock.
    TilePassResult runTiles (const TileGrid& grid, const CancellationToken& cancel, const TileFunction& fn);

private:
    // mutex_ guards nextTile, running and the error, along with the pool's state.
    struct Pass
    {
        const TileGrid* grid = nullptr;
        const CancellationToken* cancel = nullptr;
        const TileFunction* fn = nullptr;
        std::uint32_t tileCount = 0;
        std::uint32_t nextTile = 0;
        std::uint32_t completed = 0;
        unsigned running = 0;
        bool failed = false;
        std::exception_ptr error;

        bool claimable() const noexcept { return ! failed && nextTile < tileCount && ! cancel->isCancelled(); }
    };

    void workerLoop (unsigned threadIndex);
    Pass* claimFromAnyPass();
    void runClaimedTile (Pass& pass, std::uint32_t tileIndex, unsigned threadIndex);
    TilePassResult runNested (const TileGrid& grid, const CancellationToken& cancel, const TileFunction& fn);

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable progress_;
    std::vector<Pass*> passes_;
    std::size_t nextPass_ = 0;
    bool callerSeatTaken_ = false;
    bool stopping_ = false;
};

// The process's one pool for pure-CPU render work, sized to the core count, so
// that every Path Trace engine and the denoiser together stay within the cores.
// Starts its threads on first use and lives until the process exits.
ThreadPool& sharedThreadPool();

} // namespace videohelper::pathtrace
