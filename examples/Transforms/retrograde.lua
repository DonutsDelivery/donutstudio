-- Reverse all notes in time
local notes = arbit.get_notes()

if #notes < 2 then
  print("Need at least two notes to reverse")
  return
end

local minStart = math.huge
local maxEnd = 0

for _, note in ipairs(notes) do
  minStart = math.min(minStart, note.startBeat)
  maxEnd = math.max(maxEnd, note.startBeat + note.lengthBeats)
end

for _, note in ipairs(notes) do
  local newStart = maxEnd - (note.startBeat + note.lengthBeats) + minStart
  arbit.move_note({
    noteId = note.id,
    midiNote = note.midiNote,
    startBeat = newStart
  })
end

print("Reversed " .. #notes .. " notes")
