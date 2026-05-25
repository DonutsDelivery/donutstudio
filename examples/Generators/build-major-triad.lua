-- Create a C major triad with harmonic links
arbit.clear_notes()

local root = arbit.create_note({
  midiNote = 60,
  startBeat = 0,
  lengthBeats = 2,
  velocity = 100,
  trackId = 0
})

local third = arbit.create_note({
  midiNote = 64,
  startBeat = 0,
  lengthBeats = 2,
  velocity = 92,
  trackId = 0
})

local fifth = arbit.create_note({
  midiNote = 67,
  startBeat = 0,
  lengthBeats = 2,
  velocity = 92,
  trackId = 0
})

arbit.create_link({
  slaveNoteId = third.noteId or third.id,
  masterNoteId = root.noteId or root.id
})

arbit.create_link({
  slaveNoteId = fifth.noteId or fifth.id,
  masterNoteId = root.noteId or root.id
})

for _, link in ipairs(arbit.get_links()) do
  if link.slaveNoteId == (third.noteId or third.id) then
    arbit.set_link_ratio({
      linkId = link.id,
      slaveHarmonic = 5,
      masterHarmonic = 4
    })
  elseif link.slaveNoteId == (fifth.noteId or fifth.id) then
    arbit.set_link_ratio({
      linkId = link.id,
      slaveHarmonic = 3,
      masterHarmonic = 2
    })
  end
end

print("Created a linked 5:4:3 C major triad")
