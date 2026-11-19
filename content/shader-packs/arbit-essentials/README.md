# DonutStudio Essentials — Generative Shaders

Five free, original GLSL shaders for the **DonutStudio Media Machine** generative-visuals
system. Together they exercise every reactivity source the shader contract
exposes — the Block A clock, Block B audio analysis, and the Block C symbolic
score / JI lattice — so they double as worked examples of the DonutStudio shader
dialect.

**License: CC0-1.0 (public domain).** These are original works written for DonutStudio;
they use Arbit-specific uniforms (`arbitNote`, `arbitLink`, `uRootFreq`,
`arbitBand`, `lattice2`, …) and are not derived from any external shader.

| Shader | Reacts to | Notes |
|---|---|---|
| `aurora_drift` | clock + audio | Aurora curtains over a star field. No notes required. |
| `prime_lattice` | score + clock | JI lattice of the sounding chord with harmonic-link edges. Spreads only when notes carry prime structure (harmonic links / ratios). |
| `spectrum_rings` | audio + clock | Radial spectrum analyzer — bass core, treble rim. |
| `pulse_grid` | clock | Synthwave perspective grid, beat/bar flashes. No notes required. |
| `harmonic_bloom` | score + audio + clock | Chord blooms into radial petals. |

## Using a shader

Each `.fs` is a single-pass fragment shader in the DonutStudio dialect (Shadertoy-style
`void mainImage(out vec4 fragColor, in vec2 fragCoord)`; bare `void main()` and
ISF also work). Load one by either:

- **In DonutStudio** — create a shader generator clip and paste/point it at the file, or
- **Via the command server / MCP** — `video_import_shader { path }` imports the
  file as a shader clip in one call.

The shader runs in the live viewport and exports byte-identically (export ==
preview by construction).

## The dialect, in one glance

Uniforms available to every shader (see `video-helper/src/shader_dialect.h` for
the authoritative list):

- **Clock (A):** `uResolution uTime uTimeDelta uFrame uBeat uBPM uBeatPhase
  uBarPhase uBeatsPerBar uClipBeat uClipLength uPlaying`
- **Audio (B):** `uRMS uPeak uOnset uOnsetAge`, `arbitBand(float band /*0..63*/)`,
  `uAudioBands` / `uAudioBands2D`
- **Score (C):** `uNoteCount uLinkCount uRootFreq`, `arbitNote(int row)` →
  `ArbitNote{t0..t3}`, `arbitLink(int edge)` → `ArbitLink{t0}`, `lattice2(e3,e5)`
- **Harmony and structure:** `uChord` = (root pitch class / 12, quality class,
  tension 0..1, beats since the chord changed), `uKey` = (tonic / 12, mode + 1,
  beats since the key changed, brightness 0..1), `uPhrase` = (bar index, phrase
  index, phrase phase 0..1, song progress 0..1), `uLastOnsetBeat`. Accessors:
  `arbitChordRoot() arbitChordQuality() arbitChordTension() arbitChordAge()
  arbitChordPulse(float width) arbitKeyTonic() arbitKeyMode() arbitKeyAge()
  arbitKeyBrightness() arbitBarIndex() arbitPhraseIndex() arbitPhrasePhase()
  arbitSongProgress() arbitLastOnsetAge()`
- **Helpers:** `hsv2rgb(vec3)`, `beatPulse(float width)`

Quality classes: 0 none, 1 major, 2 minor, 3 diminished, 4 augmented,
5 suspended, 6 dominant 7th, 7 major 7th, 8 minor 7th, 9 other (clusters and
unnamed sets). Modes (`uKey.y`): 0 none, 1 Ionian, 2 Dorian, 3 Phrygian,
4 Lydian, 5 Mixolydian, 6 Aeolian, 7 Locrian, 8 harmonic minor, 9 melodic minor.
These values follow the music, not the volume: a shader that reads
`arbitChordPulse(0.5)` flashes on each chord change, and one that maps
`arbitChordTension()` to colour tells a cluster from a triad. A project without chord data keeps
all of them at 0.

`ArbitNote` fields: `t0 = (midi, velocity, ageBeats, remainingBeats)`,
`t1 = (freqHz, cents, track, isRoot)`, `t2 = primeExp(2,3,5,7)`,
`t3 = (primeExp11, primeExp13, masterRow, _)`. `ArbitLink.t0 =
(slaveRow, masterRow, ratioNum, ratioDen)`.

## Publishing as store content

This directory is the source for a downloadable **Visuals** store pack. The store
catalog is authoritative on the VPS, not in the repo (see the project store
notes) — to publish: zip this folder, host it (e.g. a GitHub Release), and add a
catalog entry on the VPS. A ready-to-edit catalog snippet and the zip/upload
steps live in `PUBLISHING.md`.
