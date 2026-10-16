#include "thread_pool.h"

#include <algorithm>

namespace videohelper::pathtrace
{

Tile TileGrid::tile (std::uint32_t index) const noexcept
{
    Tile t;
    const std::uint32_t columns = tilesX();
    if (columns == 0)
        return t;
    t.index = index;
    t.x0 = (index % columns) * tileSize;
    t.y0 = (index / columns) * tileSize;
    t.x1 = std::min (width, t.x0 + tileSize);
    t.y1 = std::min (height, t.y0 + tileSize);
    return t;
}

ThreadPool::ThreadPool (unsigned threadCount)
{
    if (threadCount == 0)
        threadCount = std::max (1u, std::thread::hardware_concurrency());
    workers_.reserve (threadCount - 1);
    for (unsigned i = 1; i < threadCount; ++i)
        workers_.emplace_back ([this, i] { workerLoop (i); });
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock (mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    for (auto& worker : workers_)
        worker.join();
}

void ThreadPool::drain (Pass& pass, unsigned threadIndex)
{
    const std::uint32_t count = pass.grid->tileCount();
    for (;;)
    {
        if (pass.cancel->isCancelled() || pass.failed.load (std::memory_order_acquire))
            return;
        const std::uint32_t index = pass.nextTile.fetch_add (1, std::memory_order_relaxed);
        if (index >= count)
            return;
        try
        {
            (*pass.fn) (pass.grid->tile (index), threadIndex);
            pass.completed.fetch_add (1, std::memory_order_relaxed);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock (pass.errorMutex);
            if (! pass.error)
                pass.error = std::current_exception();
            pass.failed.store (true, std::memory_order_release);
            return;
        }
    }
}

void ThreadPool::workerLoop (unsigned threadIndex)
{
    std::uint64_t seenGeneration = 0;
    for (;;)
    {
        Pass* pass = nullptr;
        {
            std::unique_lock<std::mutex> lock (mutex_);
            wake_.wait (lock, [&] { return stopping_ || passGeneration_ != seenGeneration; });
            if (stopping_)
                return;
            seenGeneration = passGeneration_;
            pass = pass_;
        }
        drain (*pass, threadIndex);
        {
            std::lock_guard<std::mutex> lock (mutex_);
            if (--busyWorkers_ == 0)
                done_.notify_all();
        }
    }
}

TilePassResult ThreadPool::runTiles (const TileGrid& grid, const CancellationToken& cancel, const TileFunction& fn)
{
    std::lock_guard<std::mutex> submit (submitMutex_);
    Pass pass;
    pass.grid = &grid;
    pass.cancel = &cancel;
    pass.fn = &fn;
    {
        // Every worker joins every pass, even an empty one, so busyWorkers_
        // counts exactly the threads that still hold a pointer to pass.
        std::lock_guard<std::mutex> lock (mutex_);
        pass_ = &pass;
        ++passGeneration_;
        busyWorkers_ = static_cast<unsigned> (workers_.size());
    }
    wake_.notify_all();
    drain (pass, 0);
    {
        std::unique_lock<std::mutex> lock (mutex_);
        done_.wait (lock, [&] { return busyWorkers_ == 0; });
        pass_ = nullptr;
    }

    TilePassResult result;
    result.tileCount = grid.tileCount();
    result.tilesCompleted = pass.completed.load (std::memory_order_relaxed);
    result.cancelled = result.tilesCompleted < result.tileCount && cancel.isCancelled();
    if (pass.error)
        std::rethrow_exception (pass.error);
    return result;
}

} // namespace videohelper::pathtrace
