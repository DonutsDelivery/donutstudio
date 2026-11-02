#include "thread_pool.h"

#include <algorithm>

#if defined(__SSE2__) || defined(_M_X64)
#define PT_TILE_FLUSHES_DENORMALS 1
#include <xmmintrin.h>
#endif

namespace videohelper::pathtrace
{

namespace
{
// The pool and threadIndex of the tile running on this thread.
thread_local const ThreadPool* tilePool = nullptr;
thread_local unsigned tileThreadIndex = 0;

// A tile flushes denormals to zero. Float order powers underflow on a crossed grating, and
// x86 takes a slow path for each one: the stock orb on CPU went from 5.85 to 16.7 s/sample.
// A caller keeps its own mode afterwards, as it may go on to audio or encoding work.
#if PT_TILE_FLUSHES_DENORMALS
class DenormalFlush
{
public:
    DenormalFlush() noexcept : previous (_mm_getcsr()) { _mm_setcsr (previous | kFlushToZero | kDenormalsAreZero); }
    ~DenormalFlush() { _mm_setcsr (previous); }

private:
    static constexpr unsigned kFlushToZero = 0x8000, kDenormalsAreZero = 0x0040;
    const unsigned previous;
};
#else
struct DenormalFlush {};
#endif

struct TileScope
{
    TileScope (const ThreadPool* pool, unsigned threadIndex) noexcept
        : previousPool (tilePool), previousIndex (tileThreadIndex)
    {
        tilePool = pool;
        tileThreadIndex = threadIndex;
    }
    ~TileScope()
    {
        tilePool = previousPool;
        tileThreadIndex = previousIndex;
    }
    TileScope (const TileScope&) = delete;
    TileScope& operator= (const TileScope&) = delete;

    const ThreadPool* previousPool;
    unsigned previousIndex;
    const DenormalFlush flush;
};
} // namespace

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

ThreadPool::Pass* ThreadPool::claimFromAnyPass()
{
    for (std::size_t offset = 0; offset < passes_.size(); ++offset)
    {
        const std::size_t index = (nextPass_ + offset) % passes_.size();
        if (passes_[index]->claimable())
        {
            nextPass_ = index + 1;
            return passes_[index];
        }
    }
    return nullptr;
}

// Called with mutex_ held, returns with it held, and releases it around fn.
void ThreadPool::runClaimedTile (Pass& pass, std::uint32_t tileIndex, unsigned threadIndex)
{
    std::exception_ptr error;
    mutex_.unlock();
    try
    {
        const TileScope scope (this, threadIndex);
        (*pass.fn) (pass.grid->tile (tileIndex), threadIndex);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    mutex_.lock();
    if (error)
    {
        if (! pass.error)
            pass.error = error;
        pass.failed = true;
    }
    else
    {
        ++pass.completed;
    }
    --pass.running;
    // A waiting caller's pass may have been cancelled, so every tile wakes them.
    progress_.notify_all();
}

void ThreadPool::workerLoop (unsigned threadIndex)
{
    std::unique_lock<std::mutex> lock (mutex_);
    for (;;)
    {
        if (stopping_)
            return;
        Pass* pass = claimFromAnyPass();
        if (pass == nullptr)
        {
            wake_.wait (lock);
            continue;
        }
        const std::uint32_t tileIndex = pass->nextTile++;
        ++pass->running;
        runClaimedTile (*pass, tileIndex, threadIndex);
    }
}

TilePassResult ThreadPool::runNested (const TileGrid& grid, const CancellationToken& cancel, const TileFunction& fn)
{
    TilePassResult result;
    result.tileCount = grid.tileCount();
    for (std::uint32_t index = 0; index < result.tileCount && ! cancel.isCancelled(); ++index)
    {
        fn (grid.tile (index), tileThreadIndex);
        ++result.tilesCompleted;
    }
    result.cancelled = result.tilesCompleted < result.tileCount;
    return result;
}

TilePassResult ThreadPool::runTiles (const TileGrid& grid, const CancellationToken& cancel, const TileFunction& fn)
{
    if (tilePool == this)
        return runNested (grid, cancel, fn);

    Pass pass;
    pass.grid = &grid;
    pass.cancel = &cancel;
    pass.fn = &fn;
    pass.tileCount = grid.tileCount();

    std::unique_lock<std::mutex> lock (mutex_);
    passes_.push_back (&pass);
    wake_.notify_all();
    for (;;)
    {
        if (pass.claimable() && ! callerSeatTaken_)
        {
            callerSeatTaken_ = true;
            while (pass.claimable())
            {
                const std::uint32_t tileIndex = pass.nextTile++;
                ++pass.running;
                runClaimedTile (pass, tileIndex, 0);
            }
            callerSeatTaken_ = false;
            progress_.notify_all();
        }
        if (! pass.claimable() && pass.running == 0)
            break;
        progress_.wait (lock);
    }
    passes_.erase (std::find (passes_.begin(), passes_.end(), &pass));

    TilePassResult result;
    result.tileCount = pass.tileCount;
    result.tilesCompleted = pass.completed;
    result.cancelled = result.tilesCompleted < result.tileCount && cancel.isCancelled();
    lock.unlock();
    if (pass.error)
        std::rethrow_exception (pass.error);
    return result;
}

ThreadPool& sharedThreadPool()
{
    // Never destroyed: a render thread can still be inside a pass at exit.
    static ThreadPool* const pool = new ThreadPool();
    return *pool;
}

} // namespace videohelper::pathtrace
