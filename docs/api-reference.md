# API Reference

Scripts call Arbit through the global `arbit` table.

This reference tracks the public scripting surface exposed by Arbit's embedded
Lua engine. Some functions may depend on the current license tier or the current
platform build.

## Notes

### `arbit.get_notes()`

Returns the current notes.

Common note fields include:

- `id`
- `midiNote`
- `startBeat`
- `lengthBeats`
- `velocity`
- `trackId`
- `centsOffset`
- `muted`
- `isRoot`

### `arbit.create_note(params)`

Creates a note.

Parameters:

- `midiNote` - MIDI note number, 0-127.
- `startBeat` - note start in beats.
- `lengthBeats` - note duration in beats.
- `velocity` - velocity, 1-127.
- `trackId` - target track.

### Note Editing

- `arbit.delete_note({ noteId = id })`
- `arbit.move_note({ noteId = id, midiNote = 64, startBeat = 1.0 })`
- `arbit.resize_note({ noteId = id, lengthBeats = 2.0 })`
- `arbit.set_note_velocity({ noteId = id, velocity = 100 })`
- `arbit.set_note_muted({ noteId = id, muted = true })`
- `arbit.set_note_root({ noteId = id, isRoot = true })`
- `arbit.transpose_note({ noteId = id, semitones = 12 })`
- `arbit.set_cents_offset({ noteId = id, cents = 14 })`
- `arbit.set_note_lyrics({ noteId = id, lyrics = "la" })`
- `arbit.clear_notes()`

### `arbit.batch_transform(params)`

Runs a batch transform. Pass `noteIds` to target specific notes, or omit it to
target the active/default selection behavior exposed by Arbit.

Common transforms include:

- `quantize`
- `double_length`
- `halve_length`
- `reverse`
- `legato`
- `staccato`
- `transpose`
- `set_velocity`
- `scale_velocity`
- `humanize`
- `mute`
- `set_track`

Example:

```lua
arbit.batch_transform({
  transform = "transpose",
  semitones = 7
})
```

## Harmonic Links

- `arbit.get_links()`
- `arbit.create_link({ slaveNoteId = childId, masterNoteId = parentId })`
- `arbit.delete_link({ linkId = id })`
- `arbit.set_link_ratio({ linkId = id, slaveHarmonic = 3, masterHarmonic = 2 })`

## Pitch Bend

- `arbit.set_pitch_bend({ noteId = id, position = 0.5, semitones = 1.0 })`
- `arbit.set_pitch_bend({ noteId = id, points = points })`
- `arbit.get_pitch_bend({ noteId = id })`
- `arbit.clear_pitch_bend({ noteId = id })`

`points` is an array of tables:

```lua
{
  { position = 0.0, semitones = 0.0 },
  { position = 0.5, semitones = 1.0 },
  { position = 1.0, semitones = 0.0 }
}
```

## Transport

- `arbit.play()`
- `arbit.stop()`
- `arbit.get_transport()`
- `arbit.set_playhead({ beats = 0 })`
- `arbit.set_bpm({ bpm = 120 })`

## Preview

- `arbit.preview_note({ noteId = id })`
- `arbit.preview_note({ midiNote = 60, trackId = 0 })`
- `arbit.stop_previews()`

## Tracks And Instruments

- `arbit.get_tracks()`
- `arbit.set_track_instrument(params)`
- `arbit.set_track_volume({ trackId = 0, volume = 0.8 })`
- `arbit.set_track_name({ trackId = 0, name = "Lead" })`
- `arbit.set_track_adsr({ trackId = 0, attack = 0.01, decay = 0.2, sustain = 0.8, release = 0.5 })`
- `arbit.set_track_reverb({ trackId = 0, size = 0.4, mix = 0.15 })`
- `arbit.set_track_muted({ trackId = 0, muted = true })`
- `arbit.set_track_soloed({ trackId = 0, soloed = true })`

Instrument examples:

```lua
arbit.set_track_instrument({
  trackId = 0,
  synthType = "soundfont",
  soundfontName = "GeneralUser-GS",
  presetIndex = 0
})
```

```lua
arbit.set_track_instrument({
  trackId = 1,
  synthType = "dx7",
  presetIndex = 42
})
```

## Instrument Libraries

- `arbit.get_soundfonts()`
- `arbit.get_presets({ soundfontName = "GeneralUser-GS" })`
- `arbit.get_dx7_presets()`

## File I/O

- `arbit.import_midi({ path = "/absolute/path/file.mid" })`
- `arbit.export_midi({ path = "/absolute/path/file.mid" })`
- `arbit.import_arbit({ path = "/absolute/path/file.arbit" })`
- `arbit.export_arbit({ path = "/absolute/path/file.arbit" })`
- `arbit.export_audio({ path = "/absolute/path/file.wav" })`

Use absolute paths.

## State And Diagnostics

- `arbit.get_state()`
- `arbit.undo()`
- `arbit.redo()`
- `arbit.get_note_frequency({ noteId = id })`
- `arbit.get_audio_meters()`
- `arbit.reset_audio_meters()`

## Tempo Map

- `arbit.get_tempo_map()`
- `arbit.set_tempo_map({ points = points })`
- `arbit.add_tempo_point({ beat = 16, bpm = 140 })`
- `arbit.remove_tempo_point({ index = 2 })`

## CC Automation

- `arbit.set_cc_automation(params)`
- `arbit.get_cc_automation(params)`
- `arbit.clear_cc_automation(params)`

## Master And Metronome

- `arbit.set_master_volume({ volume = 0.8 })`
- `arbit.get_master_volume()`
- `arbit.set_metronome(params)`
- `arbit.get_metronome()`
