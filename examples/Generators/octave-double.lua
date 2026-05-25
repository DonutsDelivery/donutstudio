-- Duplicate every note one octave higher
local notes = arbit.get_notes()
local created = 0

for _, note in ipairs(notes) do
  if note.midiNote + 12 <= 127 then
    arbit.create_note({
      midiNote = note.midiNote + 12,
      startBeat = note.startBeat,
      lengthBeats = note.lengthBeats,
      velocity = math.max(1, math.floor(note.velocity * 0.7)),
      trackId = note.trackId
    })
    created = created + 1
  end
end

print("Created " .. created .. " octave doubles")
