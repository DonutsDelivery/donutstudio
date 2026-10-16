#pragma once

// Fixed worker pool that renders an image as square tiles. Workers claim tiles
// from one atomic counter, so each tile of a pass runs exactly once unless the
// pass is cancelled. Cancellation stops new tiles from starting; tiles already
// running finish, so a viewport restart waits for at most one tile per thread.

#include <atomic>
#include <condition_variable>
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
class CancellationToken
{
public:
    void cancel() noexcept { cancelled_.store (true, std::memory_order_release); }
    void reset() noexcept { cancelled_.store (false, std::memory_order_release); }
    bool isCancelled() const noexcept { return cancelled_.load (std::memory_order_acquire); }

private:
    std::atomic<bool> cancelled_ { false };
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

    // Runs fn once per tile and blocks until the pass ends. The caller renders
    // too, as threadIndex 0. Passes from several threads run one at a time. If
    // fn throws, the pass stops claiming tiles and runTiles rethrows the first
    // exception after every thread has left the pass.
    TilePassResult runTiles (const TileGrid& grid, const CancellationToken& cancel, const TileFunction& fn);

private:
    struct Pass
    {
        const TileGrid* grid = nullptr;
        const CancellationToken* cancel = nullptr;
        const TileFunction* fn = nullptr;
        std::atomic<std::uint32_t> nextTile { 0 };
        std::atomic<std::uint32_t> completed { 0 };
        std::atomic<bool> failed { false };
        std::exception_ptr error;
        std::mutex errorMutex;
    };

    void workerLoop (unsigned threadIndex);
    static void drain (Pass& pass, unsigned threadIndex);

    std::vector<std::thread> workers_;
    std::mutex submitMutex_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    Pass* pass_ = nullptr;
    std::uint64_t passGeneration_ = 0;
    unsigned busyWorkers_ = 0;
    bool stopping_ = false;
};

} // namespace videohelper::pathtrace
