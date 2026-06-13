# Editorial Rules — music-synced edits (v0)

Seeded 2026-06-13 from a production-pipeline study. These are taste rules, not
tool mechanics: check every edit plan against them before applying it.
Accumulate new rules and lock them with dates, same convention as
`lessons.md`.

## Rhythm

- **Cut on the beat grid.** Cuts land on beats or bars; transitions span
  musical units (1 beat, 2 beats, 1 bar) — never arbitrary frame counts.
  Arbit's timeline is beat-domain; there is no excuse for an off-grid cut.
- **Visual-beat cap.** No single static visual holds longer than ~2 bars at
  moderate tempi without internal motion (camera move, zoom, automation, or
  in-frame action). A static shot past the cap reads as a stall.
- **Pattern-interrupt density scales with energy.** Chorus / drop sections get
  denser cuts and more movement; verses get longer holds. Map the cut rate to
  the song's energy curve, not a uniform interval.
- **Exactly one deliberate slow-down per piece.** One moment — the drop, the
  reveal, the verdict — gets room to breathe: a longer hold or a half-speed
  ramp. More than one and none of them land; zero and the edit never exhales.

## Structure

- **Clip length distribution follows the song's section structure.** Section
  boundaries (intro/verse/chorus/bridge) are the first-class cut points; the
  histogram of clip lengths should change at each section, not stay flat
  across the whole timeline.
- **Open ON action.** The first frame of every clip is the action already
  underway — trim off the wind-up. Dead air at a clip head is the most common
  amateur tell.

## Motivation

- **Every effect/zoom needs a musical reason.** Anchor each effect, zoom, or
  speed change to a beat, a detected onset, or a section change — and record
  it in the plan's `reason` field. If you cannot name the musical reason,
  delete the effect.
- **Mute test.** Would the edit still read with the sound off? The rhythm of
  the cuts alone should communicate the energy curve. If it only works because
  the music carries it, the visuals are passive.
