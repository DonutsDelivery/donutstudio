-- Snap all notes to the nearest C major pitch class
local scale = { 0, 2, 4, 5, 7, 9, 11 }

local function nearest_in_scale(midiNote)
  local pc = midiNote % 12
  local octave = midiNote - pc
  local bestDistance = 99
  local bestNote = midiNote

  for _, degree in ipairs(scale) do
    local distance = math.abs(pc - degree)
    if distance > 6 then
      distance = 12 - distance
    end

    if distance < bestDistance then
      bestDistance = distance
      bestNote = octave + degree
    end
  end

  return bestNote
end

local notes = arbit.get_notes()
local moved = 0

for _, note in ipairs(notes) do
  local target = nearest_in_scale(note.midiNote)
  if target ~= note.midiNote then
    arbit.move_note({
      noteId = note.id,
      midiNote = target,
      startBeat = note.startBeat
    })
    moved = moved + 1
  end
end

print("Snapped " .. moved .. " notes to C major")
