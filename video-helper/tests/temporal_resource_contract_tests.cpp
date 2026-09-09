#include "temporal_resource_contract.h"
#include "temporal_resource_state.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

void checkTransition(const videotemporal::Transition& transition,
                     videotemporal::TransitionAction action,
                     videotemporal::ResetCause resetCause,
                     videotemporal::RejectionCause rejectionCause,
                     const char* message)
{
    check(transition.action == action
              && transition.resetCause == resetCause
              && transition.rejectionCause == rejectionCause,
          message);
}

videotemporal::EvaluationPoint point(int64_t frameIndex,
                                     videotemporal::EvaluationMode mode =
                                         videotemporal::EvaluationMode::preview)
{
    videotemporal::EvaluationPoint result;
    result.frameIndex = frameIndex;
    result.structuralRevision = 11;
    result.loopDiscontinuitySerial = 2;
    result.helperGeneration = 7;
    result.mode = mode;
    return result;
}
} // namespace

int main()
{
    using namespace videotemporal;

    ResourceLimits limits;
    limits.maximumHistoryLength = 16;
    limits.maximumHistoryBytes = 256ull * 1024ull * 1024ull;

    ResourceFootprint footprint;
    std::string error;
    const ResourceContract delay(Mode::frameDelay, PixelFormat::rgba16f,
                                 { 1920, 1080 }, 4);
    check(admitResource(delay, limits, false, footprint, error),
          "an explicit bounded frame delay is admitted");
    check(footprint.retainedImages == 4 && footprint.transientImages == 1,
          "frame delay retains its exact ring and one admitted result image");
    check(footprint.bytesPerImage == 1920ull * 1080ull * 8ull
              && footprint.historyBytes == 1920ull * 1080ull * 8ull * 4ull
              && footprint.transientBytes == 1920ull * 1080ull * 8ull,
          "format and extent determine exact history and result bytes");

    const ResourceContract feedback(Mode::feedback, PixelFormat::rgba8,
                                    { 1280, 720 }, 2);
    const ResourceContract echo(Mode::echo, PixelFormat::rgba8,
                                { 1280, 720 }, 6);
    const ResourceContract stutter(Mode::stutter, PixelFormat::rgba8,
                                   { 1280, 720 }, 1);
    const ResourceContract exposure(Mode::longExposure, PixelFormat::rgba16f,
                                    { 640, 360 }, 8);
    check(admitResource(feedback, limits, false, footprint, error),
          "two-image ping-pong feedback is admitted");
    check(admitResource(echo, limits, false, footprint, error)
              && footprint.retainedImages == 6,
          "echo retains its declared preceding inputs");
    check(admitResource(stutter, limits, false, footprint, error),
          "single held-image stutter is admitted");
    check(admitResource(exposure, limits, false, footprint, error)
              && footprint.retainedImages == 8,
          "long exposure retains a bounded rolling window");

    const ResourceContract invalidFeedback(Mode::feedback, PixelFormat::rgba8,
                                           { 1280, 720 }, 1);
    check(! admitResource(invalidFeedback, limits, false, footprint, error)
              && error == "temporal feedback requires exactly two ping-pong history images"
              && footprint.historyBytes == 0,
          "feedback rejects incomplete ping-pong history before allocation");

    const ResourceContract invalidStutter(Mode::stutter, PixelFormat::rgba8,
                                          { 1280, 720 }, 2);
    check(! admitResource(invalidStutter, limits, false, footprint, error)
              && error == "temporal stutter requires exactly one history image"
              && footprint.retainedImages == 0,
          "stutter rejects excess history before allocation");

    const ResourceContract noHistory(Mode::echo, PixelFormat::rgba8,
                                     { 1280, 720 }, 0);
    check(! admitResource(noHistory, limits, false, footprint, error)
              && error == "temporal resource history length must be positive",
          "zero history is rejected");

    const ResourceContract noExtent(Mode::frameDelay, PixelFormat::rgba8,
                                    { 0, 1080 }, 1);
    check(! admitResource(noExtent, limits, false, footprint, error)
              && error == "temporal resource requires a positive explicit image extent",
          "implicit image extent is rejected");

    const ResourceContract noFormat(Mode::frameDelay,
                                    static_cast<PixelFormat>(255),
                                    { 1920, 1080 }, 1);
    check(! admitResource(noFormat, limits, false, footprint, error)
              && error == "temporal resource requires a declared image format",
          "undeclared image format is rejected");

    const ResourceContract tooManyFrames(Mode::echo, PixelFormat::rgba8,
                                         { 1280, 720 }, 17);
    check(! admitResource(tooManyFrames, limits, false, footprint, error)
              && error == "temporal resource history length capacity exceeded"
              && footprint.historyBytes == 0,
          "history count is bounded before allocation");

    ResourceLimits byteLimits = limits;
    byteLimits.maximumHistoryBytes = 1024;
    check(! admitResource(delay, byteLimits, false, footprint, error)
              && error == "temporal resource history byte capacity exceeded"
              && footprint.historyBytes == 0,
          "history bytes are bounded before allocation");

    ResourceLimits transientLimits = limits;
    transientLimits.maximumTransientBytes = 1024;
    check(! admitResource(delay, transientLimits, false, footprint, error)
              && error == "temporal resource transient byte capacity exceeded"
              && footprint.transientBytes == 0,
          "frame-delay result bytes are bounded before allocation");

    ResourceLimits overflowLimits;
    overflowLimits.maximumImageDimension = 0xffffffffu;
    overflowLimits.maximumHistoryLength = 1;
    overflowLimits.maximumHistoryBytes = 0xffffffffffffffffull;
    const ResourceContract overflowing(Mode::frameDelay, PixelFormat::rgba32f,
                                       { 0xffffffffu, 0xffffffffu }, 1);
    check(! admitResource(overflowing, overflowLimits, false, footprint, error)
              && error == "temporal resource byte accounting overflow"
              && footprint.historyBytes == 0,
          "byte multiplication overflow is rejected before allocation");

    check(! admitResource(delay, limits, true, footprint, error)
              && error == "temporal resources do not authorize graph cycles"
              && footprint.retainedImages == 0,
          "temporal state never authorizes an implicit graph cycle");

    ResourceState preview;
    checkTransition(preview.evaluate(point(10)), TransitionAction::resetAndAdvance,
                    ResetCause::firstEvaluation, RejectionCause::none,
                    "the first preview frame clears history before advancing");
    checkTransition(preview.evaluate(point(11)), TransitionAction::advance,
                    ResetCause::none, RejectionCause::none,
                    "strictly forward preview time advances once");

    auto paused = point(12);
    paused.paused = true;
    checkTransition(preview.evaluate(paused), TransitionAction::hold,
                    ResetCause::none, RejectionCause::none,
                    "paused preview holds without advancing");
    check(preview.frameIndex() == 11,
          "paused hold does not consume the requested frame");
    checkTransition(preview.evaluate(point(12)), TransitionAction::advance,
                    ResetCause::none, RejectionCause::none,
                    "unpausing advances the previously held frame once");
    checkTransition(preview.evaluate(point(12)), TransitionAction::hold,
                    ResetCause::none, RejectionCause::none,
                    "equal-time presentation holds current history");

    checkTransition(preview.evaluate(point(3)), TransitionAction::resetAndAdvance,
                    ResetCause::backwardSeek, RejectionCause::none,
                    "backward seek clears history deterministically");

    auto looped = point(4);
    looped.loopDiscontinuitySerial = 3;
    checkTransition(preview.evaluate(looped), TransitionAction::resetAndAdvance,
                    ResetCause::loopDiscontinuity, RejectionCause::none,
                    "loop discontinuity clears history deterministically");

    auto revised = looped;
    revised.structuralRevision = 12;
    checkTransition(preview.evaluate(revised), TransitionAction::resetAndAdvance,
                    ResetCause::structuralRevision, RejectionCause::none,
                    "structural revision clears history deterministically");

    auto restarted = revised;
    restarted.helperGeneration = 8;
    checkTransition(preview.evaluate(restarted), TransitionAction::resetAndAdvance,
                    ResetCause::helperRestart, RejectionCause::none,
                    "helper restart clears history deterministically");

    auto explicitlyReset = restarted;
    explicitlyReset.frameIndex += 1;
    explicitlyReset.explicitReset = true;
    checkTransition(preview.evaluate(explicitlyReset), TransitionAction::resetAndAdvance,
                    ResetCause::explicitReset, RejectionCause::none,
                    "explicit reset clears history deterministically");

    ResourceState offline;
    checkTransition(offline.evaluate(point(100, EvaluationMode::offlineSequential)),
                    TransitionAction::resetAndAdvance, ResetCause::firstEvaluation,
                    RejectionCause::none,
                    "offline evaluation initializes at its declared range start");
    checkTransition(offline.evaluate(point(101, EvaluationMode::offlineSequential)),
                    TransitionAction::advance, ResetCause::none, RejectionCause::none,
                    "offline evaluation accepts the next frame");
    checkTransition(offline.evaluate(point(103, EvaluationMode::offlineSequential)),
                    TransitionAction::reject, ResetCause::none,
                    RejectionCause::offlineNonSequential,
                    "offline evaluation rejects a forward gap");
    check(offline.frameIndex() == 101,
          "rejected offline evaluation does not mutate history state");
    checkTransition(offline.evaluate(point(102, EvaluationMode::offlineSequential)),
                    TransitionAction::advance, ResetCause::none, RejectionCause::none,
                    "offline evaluation resumes from the last accepted frame");
    checkTransition(offline.evaluate(point(102, EvaluationMode::offlineSequential)),
                    TransitionAction::hold, ResetCause::none, RejectionCause::none,
                    "offline equal-time presentation holds without accumulation");
    checkTransition(offline.evaluate(point(99, EvaluationMode::offlineSequential)),
                    TransitionAction::resetAndAdvance, ResetCause::backwardSeek,
                    RejectionCause::none,
                    "offline backward seek starts a new deterministic sequence");

    ResourceState independent;
    independent.evaluate(point(50));
    check(preview.frameIndex() == explicitlyReset.frameIndex
              && independent.frameIndex() == 50,
          "separate viewport and export state owners remain isolated");

    offline.reset();
    check(! offline.initialized(), "explicit owner reset drops temporal state");
    checkTransition(offline.evaluate(point(200, EvaluationMode::offlineSequential)),
                    TransitionAction::resetAndAdvance, ResetCause::firstEvaluation,
                    RejectionCause::none,
                    "evaluation after explicit reset starts a fresh sequence");

    if (failures != 0) return EXIT_FAILURE;
    std::cout << "Temporal resource contract checks passed\n";
    return EXIT_SUCCESS;
}
