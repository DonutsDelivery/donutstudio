# Cross-frame feedback history (ping-pong FBO)

The `feedback_trail` effect is the only **history-dependent** pass in the
renderer: its output depends on the previous frame's output, not just the
current source. This note documents the persistent double-buffer it uses.
The same mechanism backs any future PERSISTENT-buffer effect (e.g. ISF
`PERSISTENT` passes) — extend it rather than inventing a parallel one.

All section numbers referenced from `renderer.{h,cpp}` comments live here.

## §1 — Why it is separate from the stateless geometric passes

Kaleidoscope, mirror, tile, warp, displace, polar-swirl, pixelate, etc. are
pure functions of the current frame: `out = f(src)`. They ping-pong the two
scratch textures `fxTex_[0]/[1]` within a single frame and need no memory.

Feedback is `out_n = blend(src_n, decay · warp(out_{n-1}))`. It must read a
texture that survived from the previous `renderComposite` call, so it cannot
use the per-frame scratch textures (those are overwritten every frame).

## §2 — The double buffer

Each `(clipId, slot)` owns a `FeedbackHistory { unsigned tex[2]; int w, h; }`
keyed by `(clipId << 8) | slot` in `feedbackHistory_`.

`frameParity_` increments once at the top of every `renderComposite` (i.e.
once per composited frame). Within the feedback pass:

```
readTex  = tex[(frameParity_ + 1) & 1]   // last frame wrote here
writeTex = tex[frameParity_ & 1]         // we write here this frame
```

Because parity flips each frame, this frame's `writeTex` is next frame's
`readTex` — trails accumulate with no read-after-write hazard on a single
texture. The pass renders into `writeTex` and returns it as the layer's
processed texture, so the composited frame and the stored history are the
same image (the trail is visible *and* fed back).

## §3 — Reset rules (when history is cleared)

History begins from transparent black and is cleared in exactly these cases:

1. **First use** — the map entry is value-initialised (`tex[*] == 0`), so the
   pass allocates both buffers and clears them.
2. **Canvas resize** — `w/h` no longer match `outW_/outH_`; both buffers are
   freed and reallocated cleared at the new size. Also enforced centrally:
   `releaseTargets()` frees the whole map on every resize/shutdown.
3. **Shutdown** — `releaseTargets()` (via shutdown) frees all buffers.

There is deliberately **no** per-seek reset: the renderer has no concept of
the timeline. See §4.

## §4 — Export / preview parity

The exporter constructs a **fresh `FrameRenderer` per export**, so its history
starts cleared. For a clip exported from its start, the export therefore
warms the trail through exactly the same frame sequence as live linear
playback → identical output.

The one divergence is a mid-timeline export range (or a live scrub/seek):
the trail at `rangeStart` depends on frames before it that the export never
rendered. The project policy (generative-visuals-research.md risk 13) is a
**pre-roll**: render N beats (default 8, per-clip override) before the range
start to warm the buffers, then begin emitting frames. This is owned by the
export driver, not the renderer — the renderer just keeps accumulating.
