#pragma once

#include "canonical_block_c_frame.h"

#include <utility>

namespace canonicalblockc
{

struct OwnerIdentity final
{
    std::uint64_t projectGeneration = 0;
    std::uint64_t sourceGeneration = 0;
    std::uint64_t helperGeneration = 0;
    std::uint64_t backendGeneration = 0;
    std::uint64_t deviceGeneration = 0;
    std::uint64_t scoreGeneration = 0;
    std::uint64_t beatMapGeneration = 0;
    std::uint64_t fpsGeneration = 0;
    std::uint64_t loopGeneration = 0;
    std::uint64_t seekGeneration = 0;
};

struct PreviewFrameRequest final
{
    OwnerIdentity identity;
    std::int64_t targetFrame = -1;
    double fps = 0.0;
};

struct ExportFrameRequest final
{
    OwnerIdentity identity;
    std::int64_t targetFrame = -1;
    double fps = 0.0;
};

inline FrameKey makePreviewFrameKey(const PreviewFrameRequest& request) noexcept;
inline FrameKey makeExportFrameKey(const ExportFrameRequest& request) noexcept;

class FrameOwner final
{
public:
    template <typename BeatAtFrame>
    std::shared_ptr<const CanonicalBlockCFrame> frameAt(
        const PreviewFrameRequest& request,
        const std::shared_ptr<const arbitmod::Score>& score,
        BeatAtFrame&& beatAtFrame)
    {
        return frameAt(makePreviewFrameKey(request), request.targetFrame, request.fps,
                       score, std::forward<BeatAtFrame>(beatAtFrame));
    }

    template <typename BeatAtFrame>
    std::shared_ptr<const CanonicalBlockCFrame> frameAt(
        const ExportFrameRequest& request,
        const std::shared_ptr<const arbitmod::Score>& score,
        BeatAtFrame&& beatAtFrame)
    {
        return frameAt(makeExportFrameKey(request), request.targetFrame, request.fps,
                       score, std::forward<BeatAtFrame>(beatAtFrame));
    }

    const std::shared_ptr<const CanonicalBlockCFrame>& current() const noexcept { return cached_; }
    std::uint64_t packCount() const noexcept { return producer_.packCount(); }
    std::int64_t packedFrame() const noexcept { return packedFrame_; }

private:
    template <typename BeatAtFrame>
    std::shared_ptr<const CanonicalBlockCFrame> frameAt(
        FrameKey key, std::int64_t targetFrame, double fps,
        const std::shared_ptr<const arbitmod::Score>& score,
        BeatAtFrame&& beatAtFrame)
    {
        key.fps = fps;
        return warmFrameProducerTo(producer_, packedFrame_, cached_, key, score,
                                   targetFrame, std::forward<BeatAtFrame>(beatAtFrame));
    }

    FrameProducer producer_;
    std::int64_t packedFrame_ = -1;
    std::shared_ptr<const CanonicalBlockCFrame> cached_;
};

inline FrameKey makeFrameKey(const OwnerIdentity& identity) noexcept
{
    FrameKey key;
    key.projectGeneration = identity.projectGeneration;
    key.sourceGeneration = identity.sourceGeneration;
    key.helperGeneration = identity.helperGeneration;
    key.backendGeneration = identity.backendGeneration;
    key.deviceGeneration = identity.deviceGeneration;
    key.scoreGeneration = identity.scoreGeneration;
    key.beatMapGeneration = identity.beatMapGeneration;
    key.fpsGeneration = identity.fpsGeneration;
    key.loopGeneration = identity.loopGeneration;
    key.seekGeneration = identity.seekGeneration;
    return key;
}

inline FrameKey makePreviewFrameKey(const PreviewFrameRequest& request) noexcept
{
    return makeFrameKey(request.identity);
}

inline FrameKey makeExportFrameKey(const ExportFrameRequest& request) noexcept
{
    return makeFrameKey(request.identity);
}
} // namespace canonicalblockc
