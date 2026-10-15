#pragma once

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace videohelper::compositediagnostic
{
// Numeric mappings are also documented beside the composite probe contract.
enum class Phase : std::uint8_t
{
    request = 1, admissionBegin = 2, snapshotAdmitted = 3,
    sourceAdmissionBegin = 4, sourceAdmitted = 5,
    contextBegin = 6, contextReady = 7, planAdmitted = 8,
    frameLoopBegin = 9, preparationBegin = 10, preparationEnd = 11,
    rendererBegin = 12, metalLockBegin = 13, metalLockAcquired = 14,
    metalEncodingBegin = 15, metalCommitBegin = 16, metalCommitEnd = 17,
    fenceCommitBegin = 18, fenceWaitBegin = 19, fenceWaitEnd = 20,
    readbackBegin = 21, readbackEnd = 22, drainBegin = 23, drainEnd = 24,
    frameLoopEnd = 25, contextShutdownBegin = 26, contextShutdownEnd = 27,
    requestSucceeded = 28, requestFailed = 29
};
enum class Backend : std::uint8_t { unknown = 0, openGL = 1, metal = 2 };

struct Record
{
    std::uint64_t rpcId = 0;
    Phase phase = Phase::request;
    Backend backend = Backend::unknown;
    std::uint64_t revision = 0, generation = 0;
    std::uint32_t width = 0, height = 0;
    std::uint64_t elapsedMs = 0;
    std::uint8_t commandBufferStatus = 0;
};

inline bool enabledValue(const char* value) noexcept
{
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline int format(char (&line)[512], const Record& record) noexcept
{
    return std::snprintf(line, sizeof(line),
        "[composite-render-diagnostic] rpcId=%" PRIu64 " phase=%u backend=%u"
        " revision=%" PRIu64 " generation=%" PRIu64 " width=%" PRIu32
        " height=%" PRIu32 " elapsedMs=%" PRIu64 " commandBufferStatus=%u\n",
        record.rpcId, static_cast<unsigned>(record.phase), static_cast<unsigned>(record.backend),
        record.revision, record.generation, record.width, record.height, record.elapsedMs,
        static_cast<unsigned>(record.commandBufferStatus <= 5 ? record.commandBufferStatus : 0));
}

class RecordBudget
{
public:
    bool claim() noexcept
    {
        auto count = count_.load(std::memory_order_relaxed);
        while (count < 2048)
            if (count_.compare_exchange_weak(count, count + 1, std::memory_order_relaxed))
                return true;
        return false;
    }
private:
    std::atomic<unsigned> count_ { 0 };
};

struct Context
{
    Record record;
    std::chrono::steady_clock::time_point started;
};
inline thread_local Context* current = nullptr;

// The synchronous composite probe stays on the request thread. Other helper
// workers and the viewport thread never inherit this context.
class ScopedRequest
{
public:
    ScopedRequest(bool requested, std::uint64_t rpcId) noexcept : previous_(current)
    {
        current = nullptr;
        if (requested && enabledValue(std::getenv("ARBIT_COMPOSITE_RENDER_DIAGNOSTICS")))
        {
            context_.record.rpcId = rpcId;
            context_.started = std::chrono::steady_clock::now();
            current = &context_;
        }
    }
    ~ScopedRequest() { current = previous_; }
    ScopedRequest(const ScopedRequest&) = delete;
    ScopedRequest& operator=(const ScopedRequest&) = delete;
private:
    Context context_;
    Context* previous_;
};

inline void dimensions(int width, int height) noexcept
{
    if (current == nullptr) return;
    current->record.width = width > 0 ? static_cast<std::uint32_t>(width) : 0;
    current->record.height = height > 0 ? static_cast<std::uint32_t>(height) : 0;
}

inline void identity(std::uint64_t revision, std::uint64_t generation,
                     int width, int height) noexcept
{
    if (current == nullptr) return;
    current->record.revision = revision;
    current->record.generation = generation;
    dimensions(width, height);
}

inline void backend(Backend value) noexcept
{
    if (current != nullptr) current->record.backend = value;
}

inline void emit(Phase phase, std::uint8_t commandBufferStatus = 0) noexcept
{
    if (current == nullptr) return;
    static RecordBudget budget;
    if (!budget.claim()) return;
    auto record = current->record;
    record.phase = phase;
    record.commandBufferStatus = commandBufferStatus;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - current->started).count();
    record.elapsedMs = elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
    char line[512];
    const int size = format(line, record);
    if (size > 0 && size < static_cast<int>(sizeof(line)))
    {
        std::fwrite(line, 1, static_cast<std::size_t>(size), stderr);
        std::fflush(stderr);
    }
}
} // namespace videohelper::compositediagnostic
