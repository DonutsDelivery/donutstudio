#include "../src/optical_flow_preview_decode.h"

#include <atomic>
#include <chrono>
#include <cstdio>

namespace
{
using Coordinator = videoopticalflow::PreviewDecodeCoordinator;

std::shared_ptr<const videoopticalflow::DecodedPairReceipt> pair (std::uint64_t generation)
{
    videoopticalflow::DecodedPairReceipt receipt;
    receipt.snapshotGeneration = generation;
    for (int index = 0; index < 2; ++index)
    {
        auto& source = receipt.sources[static_cast<std::size_t> (index)];
        source.clipId = static_cast<std::uint32_t> (index + 11);
        source.sourceGeneration = 8;
        source.sourceSha256 = std::string (64, index == 0 ? 'a' : 'b');
        source.presentationOrdinal = static_cast<std::uint64_t> (index + 3);
        source.frame.width = 1;
        source.frame.height = 1;
        source.frame.strideBytes = 4;
        source.frame.rgba = { static_cast<std::uint8_t> (index), 2, 3, 255 };
        source.frame.ptsTicks = index + 10;
        source.frame.timeBase = { 1, 30 };
        source.frame.color.primaries = colortransform::ColorSpace::SRGB;
        source.frame.color.transfer = colortransform::TransferFunction::SRGB;
        source.pixelSha256 = videohelper::sha256Bytes (
            source.frame.rgba.data(), source.frame.rgba.size());
    }
    return std::make_shared<const videoopticalflow::DecodedPairReceipt> (std::move (receipt));
}

std::optional<Coordinator::Completion> waitFor (Coordinator& coordinator)
{
    for (int attempt = 0; attempt < 200; ++attempt)
    {
        if (auto completed = coordinator.takeCompletion()) return completed;
        std::this_thread::sleep_for (std::chrono::milliseconds (2));
    }
    return std::nullopt;
}
}

int main()
{
    std::mutex gateMutex;
    std::condition_variable gateWake;
    bool firstStarted = false;
    bool releaseFirst = false;
    int calls = 0;
    auto resolver = [&] (const videowire::ResolvedVisualSnapshot&, int,
                         double, int, int,
                         std::shared_ptr<const videoopticalflow::DecodedPairReceipt>& out)
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        ++calls;
        if (calls == 1)
        {
            firstStarted = true;
            gateWake.notify_all();
            gateWake.wait (lock, [&] { return releaseFirst; });
        }
        out = pair (7);
        return std::string {};
    };
    auto snapshot = std::make_shared<const videowire::ResolvedVisualSnapshot>();
    Coordinator::Key first { 1, 7, 2, 3, 4, 5, 20, 1.0, 1, 1 };
    auto second = first;
    second.displaySec = 2.0;
    Coordinator coordinator (resolver);
    coordinator.request (first, snapshot);
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        if (! gateWake.wait_for (lock, std::chrono::seconds (2),
                                 [&] { return firstStarted; }))
        {
            releaseFirst = true;
            lock.unlock();
            gateWake.notify_all();
            return 2;
        }
    }
    coordinator.request (second, snapshot);
    auto newest = second;
    for (int frame = 3; frame < 30; ++frame)
    {
        newest.displaySec = static_cast<double> (frame);
        coordinator.request (newest, snapshot);
    }
    {
        std::lock_guard<std::mutex> lock (gateMutex);
        releaseFirst = true;
    }
    gateWake.notify_all();
    const auto firstCompletion = waitFor (coordinator);
    if (! firstCompletion || ! (firstCompletion->key == first)
        || ! firstCompletion->decoded) return 1;
    if (! firstCompletion->key.usableAt (newest, true)
        || firstCompletion->key.usableAt (newest, false)
        || ! firstCompletion->key.usableAt (first, false)
        || newest.usableAt (first, true)) return 1;
    auto changedOwner = newest;
    ++changedOwner.snapshotGeneration;
    if (firstCompletion->key.usableAt (changedOwner, true)) return 1;
    changedOwner = newest;
    ++changedOwner.structuralRevision;
    if (firstCompletion->key.usableAt (changedOwner, true)) return 1;
    changedOwner = newest;
    ++changedOwner.helperGeneration;
    if (firstCompletion->key.usableAt (changedOwner, true)) return 1;
    changedOwner = newest;
    ++changedOwner.loopGeneration;
    if (firstCompletion->key.usableAt (changedOwner, true)) return 1;
    changedOwner = newest;
    ++changedOwner.seekGeneration;
    if (firstCompletion->key.usableAt (changedOwner, true)) return 1;
    changedOwner = newest;
    ++changedOwner.maxW;
    if (firstCompletion->key.usableAt (changedOwner, true)) return 1;
    const auto latestCompletion = waitFor (coordinator);
    if (! latestCompletion || ! (latestCompletion->key == newest)
        || ! latestCompletion->decoded || ! latestCompletion->error.empty()
        || latestCompletion->decoded->sources[0].clipId != 11
        || latestCompletion->decoded->sources[1].clipId != 12
        || latestCompletion->decoded->sources[0].presentationOrdinal != 3
        || latestCompletion->decoded->sources[1].presentationOrdinal != 4)
        return 1;
    coordinator.cancel();
    if (coordinator.takeCompletion()) return 1;
    coordinator.stop();

    bool resetStarted = false;
    bool releaseReset = false;
    int resetCalls = 0;
    Coordinator resetting ([&] (const videowire::ResolvedVisualSnapshot&, int,
                               double, int, int,
                               std::shared_ptr<const videoopticalflow::DecodedPairReceipt>& out)
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        if (++resetCalls == 1)
        {
            resetStarted = true;
            gateWake.notify_all();
            gateWake.wait (lock, [&] { return releaseReset; });
        }
        out = pair (7);
        return std::string {};
    });
    resetting.request (first, snapshot);
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        if (! gateWake.wait_for (lock, std::chrono::seconds (2),
                                 [&] { return resetStarted; }))
        {
            releaseReset = true;
            lock.unlock();
            gateWake.notify_all();
            return 2;
        }
    }
    std::atomic<bool> resetReturned { false };
    std::thread resetter ([&]
    {
        resetting.cancel();
        resetReturned.store (true);
    });
    for (int attempt = 0; attempt < 50 && ! resetReturned.load(); ++attempt)
        std::this_thread::sleep_for (std::chrono::milliseconds (2));
    const bool nonblockingReset = resetReturned.load();
    if (! nonblockingReset)
    {
        std::lock_guard<std::mutex> lock (gateMutex);
        releaseReset = true;
        gateWake.notify_all();
    }
    resetter.join();
    auto afterReset = first;
    ++afterReset.seekGeneration;
    resetting.request (afterReset, snapshot);
    {
        std::lock_guard<std::mutex> lock (gateMutex);
        releaseReset = true;
    }
    gateWake.notify_all();
    const auto resetCompletion = waitFor (resetting);
    if (! nonblockingReset || ! resetCompletion
        || ! (resetCompletion->key == afterReset)
        || first.usableAt (afterReset, true)
        || ! resetCompletion->key.usableAt (afterReset, false)
        || resetting.takeCompletion()) return 1;
    resetting.stop();

    bool sameKeyStarted = false;
    bool releaseSameKey = false;
    std::atomic<int> sameKeyCalls { 0 };
    Coordinator sameKey ([&] (const videowire::ResolvedVisualSnapshot&, int,
                             double, int, int,
                             std::shared_ptr<const videoopticalflow::DecodedPairReceipt>& out)
    {
        if (++sameKeyCalls == 1)
        {
            std::unique_lock<std::mutex> lock (gateMutex);
            sameKeyStarted = true;
            gateWake.notify_all();
            gateWake.wait (lock, [&] { return releaseSameKey; });
        }
        out = pair (7);
        return std::string {};
    });
    sameKey.request (first, snapshot);
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        if (! gateWake.wait_for (lock, std::chrono::seconds (2),
                                 [&] { return sameKeyStarted; }))
        {
            releaseSameKey = true;
            lock.unlock();
            gateWake.notify_all();
            return 2;
        }
    }
    sameKey.cancel();
    sameKey.request (first, snapshot);
    const bool queuedAfterCancel = sameKey.stateFor (first) == std::string ("queued")
        && ! sameKey.takeCompletion();
    {
        std::lock_guard<std::mutex> lock (gateMutex);
        releaseSameKey = true;
    }
    gateWake.notify_all();
    const auto sameKeyCompletion = waitFor (sameKey);
    if (! queuedAfterCancel || ! sameKeyCompletion
        || ! (sameKeyCompletion->key == first)
        || ! sameKeyCompletion->decoded || ! sameKeyCompletion->error.empty()
        || sameKeyCalls != 2 || sameKey.takeCompletion()) return 1;
    if (sameKey.stateFor (first) != std::string ("unclaimed")) return 1;
    videoopticalflow::PreviewDecodeOwner retainedOwner;
    if (! retainedOwner.retain (*sameKeyCompletion, first, false)) return 1;
    for (int frame = 0; frame < 100; ++frame)
        if (! retainedOwner.owns (first)) sameKey.request (first, snapshot);
    if (sameKeyCalls != 2
        || retainedOwner.stateFor (first) != std::string ("viewportRetained")
        || ! retainedOwner.completion()
        || ! retainedOwner.completion()->decoded) return 1;
    retainedOwner.release();
    // A consumed completion can be lost before a viewport claims it. No
    // explicit discard is needed to make its exact key retryable.
    sameKey.request (first, snapshot);
    const auto retriedCompletion = waitFor (sameKey);
    if (! retriedCompletion || ! (retriedCompletion->key == first)
        || ! retriedCompletion->decoded || sameKeyCalls != 3) return 1;
    sameKey.discardUnclaimed (retriedCompletion->key);
    sameKey.request (first, snapshot);
    const auto visibleAgain = waitFor (sameKey);
    if (! visibleAgain || ! (visibleAgain->key == first)
        || ! visibleAgain->decoded || sameKeyCalls != 4) return 1;
    sameKey.stop();

    videoopticalflow::PreviewDecodeOwner uploadOwner;
    Coordinator::Completion oldUpload { first, pair (7), {} };
    Coordinator::Completion newerUpload { second, pair (7), {} };
    if (! uploadOwner.retain (oldUpload, second, true)
        || ! uploadOwner.retain (newerUpload, second, true)
        || ! uploadOwner.owns (second)
        || uploadOwner.stateFor (second) != std::string ("viewportQueued")
        || ! uploadOwner.completion()
        || ! (uploadOwner.completion()->key == first)) return 1;
    uploadOwner.discardStale (second, false);
    if (! uploadOwner.completion()
        || ! (uploadOwner.completion()->key == second)
        || uploadOwner.stateFor (second) != std::string ("viewportRetained")
        || uploadOwner.retain (oldUpload, second, false)) return 1;
    // Retaining an older or duplicate frame cannot replace current pixels.
    if (! uploadOwner.retain (newerUpload, second, false)
        || ! (uploadOwner.completion()->key == second)) return 1;
    uploadOwner.release();
    if (uploadOwner.completion() || uploadOwner.owns (second)) return 1;

    auto third = second;
    third.displaySec = 3.0;
    auto fourth = third;
    fourth.displaySec = 4.0;
    if (! uploadOwner.retain (oldUpload, fourth, true)
        || ! uploadOwner.retain (newerUpload, fourth, true)
        || ! uploadOwner.retain ({ third, pair (7), {} }, fourth, true)
        || ! uploadOwner.retain ({ fourth, pair (7), {} }, fourth, true)
        || ! uploadOwner.owns (fourth) || uploadOwner.owns (second)
        || uploadOwner.owns (third)) return 1;
    uploadOwner.release();
    if (! uploadOwner.completion()
        || ! (uploadOwner.completion()->key == fourth)) return 1;
    uploadOwner.release();
    if (uploadOwner.completion()) return 1;

    for (int changed = 0; changed < 9; ++changed)
    {
        auto changedKey = second;
        switch (changed)
        {
            case 0: ++changedKey.timelineGeneration; break;
            case 1: ++changedKey.snapshotGeneration; break;
            case 2: ++changedKey.structuralRevision; break;
            case 3: ++changedKey.helperGeneration; break;
            case 4: ++changedKey.loopGeneration; break;
            case 5: ++changedKey.seekGeneration; break;
            case 6: ++changedKey.clipId; break;
            case 7: ++changedKey.maxW; break;
            case 8: ++changedKey.maxH; break;
        }
        if (! uploadOwner.retain (newerUpload, second, false)) return 1;
        uploadOwner.discardStale (changedKey, true);
        if (uploadOwner.completion() || uploadOwner.owns (second)
            || uploadOwner.retain (newerUpload, changedKey, true)) return 1;
    }
    // Hidden clips destroy their retained owner; showing the same key again
    // must claim fresh ownership rather than inherit a delivered marker.
    if (! uploadOwner.retain (newerUpload, second, false)) return 1;
    uploadOwner = videoopticalflow::PreviewDecodeOwner {};
    if (uploadOwner.completion() || uploadOwner.owns (second)) return 1;

    bool discardedStarted = false;
    bool releaseDiscarded = false;
    std::atomic<int> discardedCalls { 0 };
    Coordinator discarded ([&] (const videowire::ResolvedVisualSnapshot&, int,
                               double, int, int,
                               std::shared_ptr<const videoopticalflow::DecodedPairReceipt>& out)
    {
        if (++discardedCalls == 1)
        {
            std::unique_lock<std::mutex> lock (gateMutex);
            discardedStarted = true;
            gateWake.notify_all();
            gateWake.wait (lock, [&] { return releaseDiscarded; });
        }
        out = pair (7);
        return std::string {};
    });
    discarded.request (first, snapshot);
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        if (! gateWake.wait_for (lock, std::chrono::seconds (2),
                                 [&] { return discardedStarted; }))
        {
            releaseDiscarded = true;
            lock.unlock();
            gateWake.notify_all();
            return 2;
        }
    }
    discarded.request (second, snapshot);
    discarded.discardUnclaimed (first);
    const bool discardedOwnerReleased = discarded.stateFor (first) == std::string ("unclaimed")
        && discarded.stateFor (second) == std::string ("queued");
    {
        std::lock_guard<std::mutex> lock (gateMutex);
        releaseDiscarded = true;
    }
    gateWake.notify_all();
    const auto otherOwnerCompletion = waitFor (discarded);
    if (! discardedOwnerReleased || ! otherOwnerCompletion
        || ! (otherOwnerCompletion->key == second)
        || ! otherOwnerCompletion->decoded || discardedCalls != 2) return 1;
    discarded.request (first, snapshot);
    const auto discardedRetry = waitFor (discarded);
    if (! discardedRetry || ! (discardedRetry->key == first)
        || ! discardedRetry->decoded || discardedCalls != 3) return 1;
    discarded.stop();

    bool closingStarted = false;
    bool releaseClosing = false;
    Coordinator closing ([&] (const videowire::ResolvedVisualSnapshot&, int,
                             double, int, int,
                             std::shared_ptr<const videoopticalflow::DecodedPairReceipt>& out)
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        closingStarted = true;
        gateWake.notify_all();
        gateWake.wait (lock, [&] { return releaseClosing; });
        out = pair (7);
        return std::string {};
    });
    closing.request (first, snapshot);
    {
        std::unique_lock<std::mutex> lock (gateMutex);
        if (! gateWake.wait_for (lock, std::chrono::seconds (2),
                                 [&] { return closingStarted; }))
        {
            releaseClosing = true;
            lock.unlock();
            gateWake.notify_all();
            return 2;
        }
    }
    std::thread closer ([&] { closing.stop(); });
    {
        std::lock_guard<std::mutex> lock (gateMutex);
        releaseClosing = true;
    }
    gateWake.notify_all();
    closer.join();
    if (closing.takeCompletion()) return 1;

    // Audio callbacks continue to publish transport while a stopped seek waits
    // for its exact image pair. Repeating that request must keep one decode owner.
    std::atomic<int> pausedCalls { 0 };
    Coordinator paused ([&] (const videowire::ResolvedVisualSnapshot&, int,
                             double displaySec, int, int,
                             std::shared_ptr<const videoopticalflow::DecodedPairReceipt>& out)
    {
        ++pausedCalls;
        auto resolved = std::make_shared<videoopticalflow::DecodedPairReceipt> (*pair (7));
        for (auto& source : resolved->sources)
        {
            source.sourceTimeSec = displaySec;
            source.frame.ptsTicks = 15;
        }
        out = std::move (resolved);
        return std::string {};
    });
    auto pausedKey = first;
    pausedKey.displaySec = 0.5;
    paused.request (pausedKey, snapshot);
    for (int transportWrite = 0; transportWrite < 500; ++transportWrite)
        paused.request (pausedKey, snapshot);
    const auto pausedPair = waitFor (paused);
    if (! pausedPair || ! (pausedPair->key == pausedKey)
        || ! pausedPair->key.usableAt (pausedKey, false)
        || ! pausedPair->decoded || ! pausedPair->error.empty()
        || pausedCalls.load() != 1
        || paused.stateFor (pausedKey) != std::string ("unclaimed")) return 1;
    videoopticalflow::PreviewDecodeOwner pausedOwner;
    if (! pausedOwner.retain (*pausedPair, pausedKey, false)) return 1;
    for (int transportWrite = 0; transportWrite < 500; ++transportWrite)
        if (! pausedOwner.owns (pausedKey)) paused.request (pausedKey, snapshot);
    if (pausedCalls.load() != 1 || ! pausedOwner.completion()
        || ! (pausedOwner.completion()->key == pausedKey)) return 1;
    for (std::size_t index = 0; index < pausedPair->decoded->sources.size(); ++index)
    {
        const auto& source = pausedPair->decoded->sources[index];
        if (source.clipId != index + 11 || source.sourceGeneration != 8
            || source.sourceSha256 != std::string (64, index == 0 ? 'a' : 'b'))
            return 1;
        if (source.sourceTimeSec != 0.5 || source.frame.ptsTicks != 15
            || source.frame.timeBase.num != 1 || source.frame.timeBase.den != 30)
            return 1;
    }
    auto nextSeek = pausedKey;
    ++nextSeek.seekGeneration;
    if (pausedPair->key.usableAt (nextSeek, false)) return 1;
    pausedOwner.discardStale (nextSeek, false);
    if (pausedOwner.completion() || pausedOwner.owns (pausedKey)) return 1;
    paused.request (nextSeek, snapshot);
    const auto nextSeekPair = waitFor (paused);
    if (! nextSeekPair || ! (nextSeekPair->key == nextSeek)
        || ! nextSeekPair->decoded || pausedCalls.load() != 2) return 1;
    paused.stop();
    std::printf ("optical-flow-preview-decode PASS\n");
    return 0;
}
