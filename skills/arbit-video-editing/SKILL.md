---
name: arbit-video-editing
description: Drive Arbit's video editor programmatically through its MCP server (or raw JSON-RPC on port 9900). Use when asked to edit video in Arbit — import clips, beat-sync footage to the project tempo, apply effects/LUTs/transitions/fades, add text overlays, automate parameters, and export — with state verification after every step.
---

# Arbit Video Editing

Arbit is a beat-grid video editor coupled to a MIDI/microtonal music engine. You edit
by calling tools, and you verify by reading state back — the compositor can describe
its own render graph, measure its own frames, and screenshot itself. **Silence is not
success**: after every mutation, confirm the change landed before moving on.

## Prerequisites

1. **Arbit standalone must be running.** It binds a command server on TCP port 9900
   (override with `ARBIT_PORT`/`ARBIT_HOST`). Verify with the `ping` tool — it should
   return "pong". If the connection is refused, ask the user to launch Arbit.
2. **MCP server** (preferred): Arbit ships an MCP server at `mcp-server/index.js`.
   Register it once:

   ```bash
   claude mcp add arbit -- node /path/to/arbit/mcp-server/index.js
   ```

   The server name is `arbit`; tools appear as `mcp__arbit__<tool>` in Claude Code.
3. **Raw TCP fallback** (no MCP needed): the protocol is newline-delimited JSON-RPC 2.0
   — send `{"method": "...", "params": {...}, "id": 1}\n`, read one line back
   (`{"result": ...}` or `{"error": "..."}`). Tool names below are the `method` names.

   ```js
   const net = require("net");
   const sock = net.connect(9900, "127.0.0.1", () =>
     sock.write(JSON.stringify({ method: "video_probe", params: { path: "/abs/clip.mp4" }, id: 1 }) + "\n"));
   sock.on("data", (d) => { console.log(JSON.parse(d.toString())); sock.destroy(); });
   ```

4. **Video helper sidecar**: video tools run through a sidecar process that starts when
   the Video Editor is opened or a clip is imported. If video tools error with the
   helper not running, ask the user to open the Video Editor window (or import a clip
   first via `video_import_clip`, which spins it up).

## Beat-domain thinking (read this first)

**Everything on the timeline is measured in beats, not seconds.** Clip placement
(`startBeat`, `lengthBeats`), transitions (`durationBeats`), fades (`inBeats`/`outBeats`),
automation points (`beat`), text overlays (`startBeat`/`lengthBeats`), and export ranges
(`startBeat`/`endBeat`) are all beat positions. Seconds are derived from the project BPM
at render time — change the BPM and the whole edit re-times.

- Read the BPM with `get_transport`; set it with `set_bpm`.
- `beats = seconds * bpm / 60`. A 4-beat transition at 120 BPM is 2 seconds.
- A clip only *follows* the grid once you set a stretch mode (see step 4 below).
  Importing and detecting beats does **not** snap anything by itself.
- Externally generated media (AI video, motion graphics, anything authored at a known
  tempo) can ride the grid: `video_set_stretch` mode 2 with the BPM it was authored at
  stretches it uniformly to the project BPM, audio pitch-preserved.

## The canonical loop

Work through these steps in order, verifying after each mutation batch (see the
Verification protocol below).

**1. Probe the media** — inspect before importing:

```json
video_probe { "path": "/abs/footage.mp4" }
```

Returns duration, fps, width/height, codecs, hasAudio. For image sequences pass
`fps` and `startNumber`. If the file is unreadable, stop and report — don't import blind.

**2. Import onto the timeline:**

```json
video_import_clip { "path": "/abs/footage.mp4", "trackId": 0, "startBeat": 0 }
```

Returns `{clipId, trackId, startBeat, lengthBeats, videoFps, videoSourceDurationSec, ...}`.
Record the `clipId` — every later call needs it. Confirm with `get_clips` (clip `kind`:
0=midi, 1=audio, 2=video).

**3. Detect beats** (footage with music/rhythm in its audio):

```json
video_detect_clip_beats { "trackId": 0, "clipId": 7 }
```

Returns `{detectedBpm, markerCount}`. This only *stores* beat markers on the clip.

**4. Set the stretch mode — the step that makes the edit BPM-adaptive.** Detection
alone does NOT snap the clip; this does:

```json
video_set_stretch { "trackId": 0, "clipId": 7, "mode": 1 }
```

Modes:

| mode | name | behavior |
|------|------|----------|
| 0 | none | real-time playback; length-in-beats recomputes on BPM change |
| 1 | tap_tempo | warp video so detected/tapped beat markers land on the project beat grid (requires markers from `video_detect_clip_beats` or tap tempo) |
| 2 | manual_bpm | uniform stretch from `manualBpm` to the project BPM — use for externally generated clips authored at a known tempo |

Mode 2 requires `manualBpm` (the clip's source BPM). Audio is re-rendered
pitch-preserved (Signalsmith). Returns `{stretchMode, audioFilePath, lengthBeats}` —
check `stretchMode` in the response matches what you asked for.

**5. Fit to canvas:**

```json
video_set_canvas { "width": 1920, "height": 1080 }
video_smart_fit  { "clipId": 7, "mode": "fill" }
```

Fit modes: `fit` (letterbox), `fill` (crop), `fit_width`, `fit_height`,
`one_to_one` (pixel-exact), `center` (no rescale).

**6. Cut and arrange.** Split at beat boundaries; transitions/fades go on the
resulting clips:

```json
split_clip { "trackId": 0, "clipId": 7, "beat": 16 }
```

Returns `{leftClipId, rightClipId}` — the original ID may no longer exist; refresh
your clip IDs from `get_clips` after splitting.

**7. Look: effects, LUT, transitions, fades.**

```json
video_set_effect     { "clipId": 7, "slot": 0, "type": 2, "params": { "amount": 1.2 } }
video_set_lut        { "clipId": 7, "path": "/abs/grade.cube" }
video_set_transition { "clipId": 8, "type": 2, "durationBeats": 1 }
video_set_fade       { "clipId": 7, "inBeats": 1, "inCurve": 3, "outBeats": 2, "outCurve": 1 }
```

- Each clip has 8 effect rack slots (0-7). `video_set_effect` **replaces** the slot;
  params not given reset to defaults. `type: -1` clears a slot. See the reference
  tables for type indices and per-effect wire param names.
- LUTs are `.cube` 3D LUTs (LUT_3D_SIZE 2-129; 17/33/65 common). Empty path clears.
- Transitions sit on a clip's **leading edge** and blend from the previous clip (or
  black) — so to transition between two clips, set the transition on the *second* one.
- Fades are **audio** fades on the clip edges.

**8. Speed ramps** (optional, clip-local rate curve):

```json
video_set_speed_ramp {
  "clipId": 7, "enabled": true, "maintainPitch": true,
  "keys": [ { "t": 0, "s": 1 }, { "t": 0.5, "s": 0.25, "i": 3 }, { "t": 1, "s": 1 } ]
}
```

`t` = normalized clip time 0-1, `s` = speed multiplier (1 = normal, range 0-4),
`i` = interpolation to next key: 0=linear, 1=ease_in, 2=ease_out, 3=ease_in_out, 4=hold.
Keys are sorted by time automatically. Clear with `enabled: false` and empty/omitted keys.

**9. Text overlays** (project-level, render above all clips):

```json
video_add_text {
  "text": "TITLE", "startBeat": 0, "lengthBeats": 8,
  "fontSize": 96, "bold": true, "textColour": "ffffffff",
  "outlineEnabled": true, "outlineColour": "ff000000", "outlineWidth": 3,
  "posX": 0.5, "posY": 0.5, "hAlign": 1, "vAlign": 1
}
```

Returns `{textId}`. Colours are ARGB hex strings (`"ffff0000"` = opaque red).
`hAlign`: 0=left, 1=center, 2=right; `vAlign`: 0=top, 1=middle, 2=bottom;
`posX`/`posY` normalized 0-1. Update with `video_update_text` (only the fields you
pass change), remove with `video_remove_text`.

**10. Automation lanes** (persisted, baked into export):

```json
video_add_automation {
  "clipId": 7, "subParam": 1,
  "points": [ { "beat": 0, "value": 1.0 }, { "beat": 8, "value": 1.3 } ]
}
```

Point values are in real parameter units (normalized internally). See the subParam
packing table below. For text-overlay lanes (subParam 3001-3003) pass `textId`
instead of `clipId`. For a *live, non-persisted* tweak there is also `video_set_param`
(paramId grammar below) — it bypasses Arbit's clip state, so prefer
`video_set_effect`/`video_smart_fit`/`video_add_automation` for anything that must
survive save/export.

**11. Export:**

```json
video_export {
  "path": "/abs/out.mp4", "width": 1920, "height": 1080, "fps": 30,
  "codec": "h264", "encoder": "libx264", "interpolation": "none",
  "startBeat": 0, "endBeat": 64, "async": true
}
```

Effects, transitions, text, and the audio mix are baked in. Omit
`startBeat`/`endBeat` for the full timeline. With `async: true` it returns
immediately — poll `video_export_progress` until done.

**12. Verify the export** (see protocol below), then probe the output file with
`video_probe` to confirm duration/fps/resolution match expectations.

## Verification protocol

Never assume a mutation worked. The verification toolkit:

- **`video_graph_describe`** — call after **every mutation batch**. Describes the
  helper's live render graph: per-clip nodes with current param values, effects
  `[{slot, type, enabled, params}]`, and text overlays. This is the primary check
  that your params actually reached the compositor (not just Arbit's UI state).
- **`screenshot { "window": "video" }`** — visual check of the Video Editor window
  (it must be open; errors otherwise). Use after layout/look changes to confirm the
  frame looks right, not just that numbers were stored.
- **`video_viewport_info`** — `{open, width, height, measuredFps, avOffsetSec,
  interpolationActive, interpolationBackend, gpuPath, framesPresented, ...}`. Check
  `measuredFps` for playback health and `avOffsetSec` for A/V sync (should be well
  under one frame). Errors if the video sidecar is not running — which is itself
  diagnostic.
- **`video_scope_data { "types": ["histogram", "waveform", "vectorscope"] }`** —
  measures the composited viewport frame *after* effects/LUTs/transitions/text. Use
  for grading checks (e.g. confirm a LUT shifted the histogram, confirm black levels
  after a contrast effect). Requires an open viewport; the first sample lands ~200ms
  after enabling, so poll until `samples > 0`.
- **`video_export_progress`** — plugin-side `{pluginActive, pluginDone, pluginOk,
  pluginError, pluginEncoder}` merged with the helper's per-frame progress
  (phase, frame n/total, fps). Poll during async exports; on completion check
  `pluginOk` and read `pluginError` if it failed. Cancel with `video_export_cancel`
  (returns `{ok, wasActive}`).

Minimum discipline: mutate → `video_graph_describe` → (visual change?) `screenshot`
→ proceed. Before export: `get_clips` + `get_transport` sanity pass. After export:
`video_export_progress` shows `pluginDone` + `pluginOk`, then `video_probe` the output.

## Reference tables

### Effect types (`video_set_effect` `type`) and wire param names

| type | effect | params |
|------|--------|--------|
| 0 | brightness | `amount` |
| 1 | contrast | `amount` |
| 2 | saturation | `amount` |
| 3 | hue | `shift` |
| 4 | exposure | `stops` |
| 5 | gamma | `value` |
| 6 | blur | `radius` |
| 7 | sharpen | `amount` |
| 8 | vignette | `amount`, `softness` |
| 9 | warm | `intensity` |
| 10 | cool | `intensity` |
| 11 | vintage | `intensity` |
| 12 | sepia | `intensity` |
| 13 | black_and_white | `intensity` |
| 14 | invert | `amount` |
| 15 | posterize | `levels` |
| 16 | noise | `amount` |
| 17 | chroma_key | `keyR`, `keyG`, `keyB`, `tolerance`, `softness`, `spillSuppress` |
| 18 | luma_key | `low`, `high`, `softness`, `invert` |
| 19 | color_wheels | `liftR`, `liftG`, `liftB`, `gammaR`, `gammaG`, `gammaB`, `gainR`, `gainG`, `gainB` |
| -1 | (clear slot) | — |

### Transition types (`video_set_transition` `type`)

0=none, 1=fade, 2=dissolve, 3=wipe_left, 4=wipe_right, 5=wipe_up, 6=wipe_down.

### Fade curves (`video_set_fade` `inCurve`/`outCurve`)

0=linear, 1=exponential (t^2), 2=logarithmic (sqrt t), 3=s-curve (smoothstep).

### `video_set_param` paramId grammar

`clip<id>/<node>/<param>`, with nodes:

| node | params |
|------|--------|
| `source` | `opacity`, `visible`, `zOrder`, `blendMode` (0=normal, 1=add, 2=multiply, 3=screen, 4=overlay) |
| `transform2d` | `scale`, `translateX`, `translateY`, `rotation`, `cropLeft`/`cropRight`/`cropTop`/`cropBottom` |
| `effect<slot>` | `type`, `enabled`, then per-effect wire names (e.g. `clip7/effect2/radius`) |
| `mask` | `type`, `cx`, `cy`, `w`, `h`, `feather`, `invert` |

Text overlays: `text<id>/{opacity, translateX, translateY}`.
Optional `atBeat` applies the value when the playhead passes that beat (omit = now).

### `video_add_automation` subParam packing

| subParam | parameter |
|----------|-----------|
| 1-8 | transform2d: 1=scale, 2=translateX, 3=translateY, 4=rotation, 5-8=crop L/R/T/B |
| 1001 | opacity |
| 1101-1105 | mask: cx, cy, w, h, feather |
| 2000 + slot*100 + paramIdx | effect slot param (paramIdx in kEffectDefs order — same order as the wire-params column above; paramIdx 99 = enabled toggle) |
| 3001-3003 | text overlay: opacity, translateX, translateY (pass `textId` instead of `clipId`) |

Example: slot 2 vignette `softness` (param index 1) → `2000 + 200 + 1 = 2201`.

## Determinism

For reproducible output (frame-hash comparisons, regression tests, CI renders), pin:

```json
{ "encoder": "libx264", "interpolation": "none" }
```

The default `encoder: "auto"` tries hardware encoders first (h264_nvenc/hevc_nvenc/VAAPI)
and `interpolation: "rife"`/`"auto"` uses GPU optical flow — both vary per machine and
per driver. `libx264` + `none` is bit-stable for the same project on the same media.

## Failure modes

| symptom | cause | fix |
|---------|-------|-----|
| Connection refused on 9900 | Arbit not running | Ask the user to launch the standalone; `ping` to confirm |
| Video tools error / `video_viewport_info` fails | Video helper sidecar not running | Open the Video Editor window, or `video_import_clip` a file to spin it up |
| "clip not found" | stale clipId (e.g. after `split_clip`) | Refresh IDs with `get_clips` |
| Clip plays black / silent | media file moved | Relink: 9900 method `video_media_relink` with `{oldPath, newPath}` (not in the MCP tool list — use the raw TCP fallback) |
| Stretch had no effect | only ran detection | `video_detect_clip_beats` stores markers; you must also call `video_set_stretch` mode 1 (or mode 2 + `manualBpm`) |
| Export "finished" but file wrong | error swallowed | Always check `video_export_progress` → `pluginOk`/`pluginError`, then `video_probe` the output |
| Scopes return no data | viewport closed or sampled too soon | Open viewport; poll until `samples > 0` (~200ms after enabling) |

## Tool index

Media & clips: `video_probe`, `video_import_clip`, `get_clips`, `split_clip`,
`video_detect_clip_beats`, `video_set_stretch`, `video_smart_fit`, `video_set_canvas`.
Look: `video_set_effect`, `video_set_lut`, `video_set_transition`, `video_set_fade`,
`video_set_speed_ramp`. Text: `video_add_text`, `video_update_text`, `video_remove_text`.
Automation: `video_set_param`, `video_add_automation`. Export: `video_export`,
`video_export_progress`, `video_export_cancel`. Verification: `video_graph_describe`,
`video_scope_data`, `video_viewport_info`, `screenshot`. Supporting: `ping`,
`get_transport`, `set_bpm`, `set_playhead`, `play`, `stop`, `export_audio`.
