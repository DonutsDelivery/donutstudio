#pragma once

#include "optical_flow_source_lease.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace videoopticalflow
{
// Only owned pixels cross this queue. Native upload and execution stay with the
// viewport's current GPU context on its render thread.
class PreviewDecodeCoordinator final
{
public:
    struct Key
    {
        std::uint64_t timelineGeneration = 0;
        std::uint64_t snapshotGeneration = 0;
        std::uint64_t structuralRevision = 0;
        std::uint64_t helperGeneration = 0;
        std::uint64_t loopGeneration = 0;
        std::uint64_t seekGeneration = 0;
        int clipId = 0;
        double displaySec = 0.0;
        int maxW = 0, maxH = 0;

        bool operator== (const Key& other) const noexcept
        {
            return timelineGeneration == other.timelineGeneration
                && snapshotGeneration == other.snapshotGeneration
                && structuralRevision == other.structuralRevision
                && helperGeneration == other.helperGeneration
                && loopGeneration == other.loopGeneration
                && seekGeneration == other.seekGeneration
                && clipId == other.clipId && displaySec == other.displaySec
                && maxW == other.maxW && maxH == other.maxH;
        }

        // Playback may show a completed earlier frame from the same owner.
        // Paused preview remains exact at the export frame time.
        bool usableAt (const Key& current, bool playing) const noexcept
        {
            return timelineGeneration == current.timelineGeneration
                && snapshotGeneration == current.snapshotGeneration
                && structuralRevision == current.structuralRevision
                && helperGeneration == current.helperGeneration
                && loopGeneration == current.loopGeneration
                && seekGeneration == current.seekGeneration
                && clipId == current.clipId
                && maxW == current.maxW && maxH == current.maxH
                && (displaySec == current.displaySec
                    || (playing && displaySec < current.displaySec));
        }
    };

    struct Completion
    {
        Key key;
        std::shared_ptr<const DecodedPairReceipt> decoded;
        std::string error;
    };

    using Resolver = std::function<std::string (
        const videowire::ResolvedVisualSnapshot&, int, double, int, int,
        std::shared_ptr<const DecodedPairReceipt>&)>;

    explicit PreviewDecodeCoordinator (Resolver resolver)
        : resolver_ (std::move (resolver)),
          worker_ ([this] { run(); }) {}

    PreviewDecodeCoordinator (const PreviewDecodeCoordinator&) = delete;
    PreviewDecodeCoordinator& operator= (const PreviewDecodeCoordinator&) = delete;
    ~PreviewDecodeCoordinator() { stop(); }

    void request (Key key, std::shared_ptr<const videowire::ResolvedVisualSnapshot> snapshot)
    {
        std::lock_guard<std::mutex> lock (mutex_);
        if (stopping_) return;
        if (pending_ && pending_->key == key) return;
        if ((active_ && activeSerial_ == serial_ && *active_ == key)
            || (completion_ && completion_->key == key))
            pending_.reset();
        else
            pending_ = Job { key, std::move (snapshot), serial_ };
        wake_.notify_one();
    }

    // Taking a completion transfers its only queue ownership to the caller.
    // Until the caller retains it, the same key is retryable. A requested key
    // alone must never suppress work after its pixels have left this queue.
    std::optional<Completion> takeCompletion()
    {
        std::lock_guard<std::mutex> lock (mutex_);
        auto result = std::move (completion_);
        completion_.reset();
        if (result) wake_.notify_one();
        return result;
    }

    // Discard work for a hidden or changed owner without cancelling a different
    // pending owner. The old in-flight job cannot republish after this discard.
    void discardUnclaimed (const Key& key)
    {
        std::lock_guard<std::mutex> lock (mutex_);
        if (completion_ && completion_->key == key) completion_.reset();
        if (pending_ && pending_->key == key) pending_.reset();
        if (active_ && activeSerial_ == serial_ && *active_ == key)
        {
            ++serial_;
            if (pending_) pending_->serial = serial_;
        }
        wake_.notify_one();
    }

    const char* stateFor (const Key& key) const
    {
        std::lock_guard<std::mutex> lock (mutex_);
        if (completion_ && completion_->key == key) return "decoded";
        if (active_ && activeSerial_ == serial_ && *active_ == key)
            return "decoding";
        if (pending_ && pending_->key == key) return "queued";
        return "unclaimed";
    }

    // Cancels the pending request and prevents an in-flight decode from
    // publishing. The worker checks the serial again before handing back bytes.
    void cancel()
    {
        std::lock_guard<std::mutex> lock (mutex_);
        ++serial_;
        pending_.reset();
        completion_.reset();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock (mutex_);
            if (stopping_) return;
            stopping_ = true;
            ++serial_;
            pending_.reset();
            completion_.reset();
            wake_.notify_one();
        }
        if (worker_.joinable()) worker_.join();
    }

private:
    struct Job
    {
        Key key;
        std::shared_ptr<const videowire::ResolvedVisualSnapshot> snapshot;
        std::uint64_t serial = 0;
    };

    void run()
    {
        for (;;)
        {
            Job job;
            {
                std::unique_lock<std::mutex> lock (mutex_);
                wake_.wait (lock, [this]
                { return stopping_ || (pending_.has_value() && ! completion_); });
                if (stopping_) return;
                job = std::move (*pending_);
                pending_.reset();
                active_ = job.key;
                activeSerial_ = job.serial;
            }
            Completion result;
            result.key = job.key;
            if (job.snapshot)
                result.error = resolver_ (*job.snapshot, job.key.clipId,
                    job.key.displaySec, job.key.maxW, job.key.maxH, result.decoded);
            else
                result.error = "Optical Flow preview lost its saved snapshot";
            if (result.error.empty() && result.decoded)
                for (const auto& source : result.decoded->sources)
                {
                    SourceFrameV2 expected;
                    colortransform::Description color;
                    if (! detail::expectedSource (source, expected, color, result.error))
                    {
                        result.decoded.reset();
                        break;
                    }
                }
            std::lock_guard<std::mutex> lock (mutex_);
            active_.reset();
            if (! stopping_ && job.serial == serial_)
                completion_ = std::move (result);
        }
    }

    Resolver resolver_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::optional<Key> active_;
    std::uint64_t activeSerial_ = 0;
    std::optional<Job> pending_;
    std::optional<Completion> completion_;
    std::uint64_t serial_ = 0;
    bool stopping_ = false;
    std::thread worker_;
};

// Each visible clip retains at most one upload and one newer decode. The
// upload keeps its pixels while a native digest is in flight; stale owners are
// discarded before a queued completion can become that upload.
class PreviewDecodeOwner final
{
public:
    using Key = PreviewDecodeCoordinator::Key;
    using Completion = PreviewDecodeCoordinator::Completion;

    void discardStale (const Key& current, bool playing)
    {
        if (upload_ && ! upload_->key.usableAt (current, playing)) upload_.reset();
        if (queued_ && ! queued_->key.usableAt (current, playing)) queued_.reset();
        promote();
    }

    bool retain (Completion completion, const Key& current, bool playing)
    {
        discardStale (current, playing);
        if (! completion.key.usableAt (current, playing)) return false;
        if (! upload_)
            upload_ = std::move (completion);
        else if (completion.key.displaySec > upload_->key.displaySec
                 && (! queued_ || completion.key.displaySec > queued_->key.displaySec))
            queued_ = std::move (completion);
        return true;
    }

    const Completion* completion() const noexcept
    {
        return upload_ ? &*upload_ : nullptr;
    }

    bool owns (const Key& key) const noexcept
    {
        return (upload_ && upload_->key == key) || (queued_ && queued_->key == key);
    }

    const char* stateFor (const Key& key) const noexcept
    {
        if (upload_ && upload_->key == key) return "viewportRetained";
        if (queued_ && queued_->key == key) return "viewportQueued";
        return nullptr;
    }

    void release()
    {
        upload_.reset();
        promote();
    }

private:
    void promote()
    {
        if (! upload_ && queued_)
        {
            upload_ = std::move (queued_);
            queued_.reset();
        }
    }

    std::optional<Completion> upload_;
    std::optional<Completion> queued_;
};
} // namespace videoopticalflow
