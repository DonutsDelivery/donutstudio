#pragma once

#include <cstdint>

namespace videotemporal
{

enum class EvaluationMode
{
    preview,
    offlineSequential
};

enum class TransitionAction
{
    resetAndAdvance,
    advance,
    hold,
    reject
};

enum class ResetCause
{
    none,
    firstEvaluation,
    backwardSeek,
    loopDiscontinuity,
    structuralRevision,
    helperRestart,
    explicitReset
};

enum class RejectionCause
{
    none,
    offlineNonSequential
};

struct EvaluationPoint
{
    int64_t frameIndex = 0;
    uint64_t structuralRevision = 0;
    uint64_t loopDiscontinuitySerial = 0;
    uint64_t helperGeneration = 0;
    EvaluationMode mode = EvaluationMode::preview;
    bool paused = false;
    bool explicitReset = false;
};

struct Transition
{
    TransitionAction action = TransitionAction::reject;
    ResetCause resetCause = ResetCause::none;
    RejectionCause rejectionCause = RejectionCause::none;
};

class ResourceState final
{
public:
    Transition evaluate(const EvaluationPoint& point) noexcept
    {
        if (! initialized_)
            return resetAndAdvance(point, ResetCause::firstEvaluation);
        if (point.explicitReset)
            return resetAndAdvance(point, ResetCause::explicitReset);
        if (point.helperGeneration != helperGeneration_)
            return resetAndAdvance(point, ResetCause::helperRestart);
        if (point.structuralRevision != structuralRevision_)
            return resetAndAdvance(point, ResetCause::structuralRevision);
        if (point.loopDiscontinuitySerial != loopDiscontinuitySerial_)
            return resetAndAdvance(point, ResetCause::loopDiscontinuity);
        if (point.frameIndex < frameIndex_)
            return resetAndAdvance(point, ResetCause::backwardSeek);

        if (point.paused || point.frameIndex == frameIndex_)
            return { TransitionAction::hold, ResetCause::none, RejectionCause::none };

        if (point.mode == EvaluationMode::offlineSequential
            && point.frameIndex != frameIndex_ + 1)
        {
            return { TransitionAction::reject, ResetCause::none,
                     RejectionCause::offlineNonSequential };
        }

        remember(point);
        return { TransitionAction::advance, ResetCause::none, RejectionCause::none };
    }

    void reset() noexcept
    {
        initialized_ = false;
        frameIndex_ = 0;
        structuralRevision_ = 0;
        loopDiscontinuitySerial_ = 0;
        helperGeneration_ = 0;
    }

    bool initialized() const noexcept { return initialized_; }
    int64_t frameIndex() const noexcept { return frameIndex_; }

private:
    Transition resetAndAdvance(const EvaluationPoint& point, ResetCause cause) noexcept
    {
        remember(point);
        return { TransitionAction::resetAndAdvance, cause, RejectionCause::none };
    }

    void remember(const EvaluationPoint& point) noexcept
    {
        initialized_ = true;
        frameIndex_ = point.frameIndex;
        structuralRevision_ = point.structuralRevision;
        loopDiscontinuitySerial_ = point.loopDiscontinuitySerial;
        helperGeneration_ = point.helperGeneration;
    }

    bool initialized_ = false;
    int64_t frameIndex_ = 0;
    uint64_t structuralRevision_ = 0;
    uint64_t loopDiscontinuitySerial_ = 0;
    uint64_t helperGeneration_ = 0;
};

} // namespace videotemporal
