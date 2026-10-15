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

} // namespace arbitviewport
