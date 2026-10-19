// viewport_publish_gate.h: when the docked viewport loop composites and
// publishes a frame.
//
// The shm and shared-GPU (dmabuf) rungs used to composite and publish on every
// tick at targetFps, even with the transport stopped, nothing edited, and no
// viewport on screen. The plugin then copied and repainted each frame. This
// gate lets the render loop skip a tick unless the output can differ from the
// frame the consumer already holds.
//
// Each tick, after the transport and audio state are read and before the scene
// is evaluated, the loop calls shouldProduce(). It produces while:
//   - the viewport is visible and an input changed since the last render. The
//     inputs are the content revision (state RPCs), the playhead or clock, the
//     baked audio mix, the render size, and the Speed Warp suspension;
//   - the transport plays, or has just stopped;
//   - the last render is not final. A live-preview script, scopes, optical flow,
//     a layer or transition source still decoding, or a Speed Warp frame waiting
//     on RIFE leaves the render unfinished, so the next tick renders again and
//     can pick up the result without any new input;
//   - the last render may not have reached the consumer. The shm rung publishes
//     a render one produce late, so a changed render, or one that follows an
//     unfinished render, needs one more produce. An unchanged, finished render
//     does not.
// requestProduce() forces one produce without an input change. The render loop
// calls it when a shared-GPU consumer connects, so a settled scene still sends
// that consumer its first frame. A hidden viewport never produces, even when
// asked. This header has no GL dependency, so the helper tests drive the decision
// tick by tick.
#pragma once

#include <cstdint>

namespace arbitviewport
{

class PublishGate
{
public:
    // What the render loop observed on one tick.
    struct Snapshot
    {
        bool visible = true;
        bool playing = false;
        uint64_t revision = 0;       // content revision, bumped by state RPCs
        uint64_t mixGeneration = 0;  // baked audio mix generation
        double displaySec = 0.0;
        double displayBeat = 0.0;
        double bpm = 0.0;
        double beatsPerBar = 0.0;
        int width = 0;
        int height = 0;
        bool interpSuspended = false; // Speed Warp off while an export holds the GPU
    };

    // True when this tick must composite and publish. Until finish() runs, the
    // gate keeps the next tick producing, so a `continue` in the produce path
    // retries on the next tick instead of going quiet.
    bool shouldProduce (const Snapshot& now)
    {
        if (! now.visible)
        {
            pending_ = false;
            return false;
        }
        const bool changed = ! haveRendered_
            || now.revision != last_.revision
            || now.mixGeneration != last_.mixGeneration
            || now.playing || last_.playing
            || now.displaySec != last_.displaySec
            || now.displayBeat != last_.displayBeat
            || now.bpm != last_.bpm
            || now.beatsPerBar != last_.beatsPerBar
            || now.width != last_.width
            || now.height != last_.height
            || now.interpSuspended != last_.interpSuspended;
        if (! changed && ! pending_)
            return false;
        changed_ = changed;
        pending_ = true;
        return true;
    }

    // Asks for one produce that no input change would otherwise trigger. A hidden
    // tick drops the request, as it drops any pending produce.
    void requestProduce()
    {
        pending_ = true;
    }

    // Records the outcome of the produce that shouldProduce() admitted.
    //   published: a frame reached the consumer. On the shm rung this is the
    //              render of the previous produce, read back now.
    //   settled:   this render is final. Nothing in it changes without new input.
    //   pipelined: the shm rung, where a render reaches the consumer one produce late.
    void finish (const Snapshot& now, bool published, bool settled, bool pipelined)
    {
        const bool previousSettled = renderSettled_;
        last_ = now;
        haveRendered_ = true;
        renderSettled_ = settled;
        if (! published)
        {
            pending_ = true;
            return;
        }
        if (! pipelined)
        {
            pending_ = ! settled;
            return;
        }
        // Until the next produce, the consumer holds the render before this one.
        // That produce is needed when the inputs changed, or when either render
        // was not final.
        pending_ = ! settled || changed_ || ! previousSettled;
    }

private:
    Snapshot last_ {};
    bool haveRendered_ = false;
    bool renderSettled_ = false;
    bool changed_ = false;
    bool pending_ = false;
};

// Bounds how often the render loop re-evaluates a frame that a rejected visual plan
// keeps failing. A held plan error used to retry on every tick, so a refused frame
// rebuilt the same scene a hundred times a second and starved the helper's RPC
// loop. The governor retries a few times with a growing delay, then stops until
// something that can change the outcome moves: a state RPC (content revision), the
// playhead, or the render size. The banner stays up meanwhile, since the loop keeps
// the plan error it already published. Time is passed in, so tests drive it tick by
// tick without a clock.
//
// While the transport plays, the playhead moves every tick, so an exact-frame key
// would give every frame a fresh set of attempts and the retries would never stop.
// A playing caller therefore leaves displaySec at 0 and names the set of clips that
// are active at the playhead instead (clipSet). The failure then holds for as long
// as the same clips stay under the playhead, and ends when the playhead leaves them.
class FrameFailureGovernor
{
public:
    struct Key
    {
        uint64_t revision = 0;   // content revision, bumped by state RPCs
        double displaySec = 0.0;
        int width = 0;
        int height = 0;
        uint64_t clipSet = 0;    // active clips at the playhead while playing, else 0

        bool operator== (const Key& other) const
        {
            return revision == other.revision && displaySec == other.displaySec
                && width == other.width && height == other.height
                && clipSet == other.clipSet;
        }
        bool operator!= (const Key& other) const { return ! (*this == other); }
    };

    // Order-sensitive token of the clips active at the playhead. Start from
    // kEmptyClipSet and fold each active clip id in timeline order.
    static constexpr uint64_t kEmptyClipSet = 1469598103934665603ULL;
    static uint64_t foldClip (uint64_t token, int clipId)
    {
        return (token ^ static_cast<uint64_t> (static_cast<uint32_t> (clipId))) * 1099511628211ULL;
    }

    // Attempts at one (revision, frame or clip set) before the loop stops retrying it.
    static constexpr int kMaxAttempts = 4;

    // Wait after the nth failure before the next attempt.
    static int64_t backoffMs (int failures)
    {
        switch (failures)
        {
            case 1:  return 100;
            case 2:  return 400;
            default: return 1600;
        }
    }

    // True when this tick must not evaluate the frame. A changed key forgets the
    // earlier failures, so the new frame gets a full set of attempts.
    bool shouldHold (const Key& now, int64_t nowMs)
    {
        if (failures_ == 0)
            return false;
        if (now != key_)
        {
            failures_ = 0;
            return false;
        }
        return failures_ >= kMaxAttempts || nowMs < retryAtMs_;
    }

    // Records a failed attempt. Returns true when it used the last attempt, so the
    // caller can log the stop once.
    bool noteFailure (const Key& now, int64_t nowMs)
    {
        if (failures_ == 0 || now != key_)
        {
            key_ = now;
            failures_ = 0;
        }
        ++failures_;
        retryAtMs_ = nowMs + backoffMs (failures_);
        return failures_ == kMaxAttempts;
    }

    // A frame rendered, or the failure came from something other than the plan.
    void noteSuccess()
    {
        failures_ = 0;
    }

    int failures() const { return failures_; }
    bool exhausted() const { return failures_ >= kMaxAttempts; }

private:
    Key key_ {};
    int failures_ = 0;
    int64_t retryAtMs_ = 0;
};

} // namespace arbitviewport
