-- Randomize note velocities by a small amount
local amount = 12
local notes = arbit.get_notes()
local changed = 0

for _, note in ipairs(notes) do
  local offset = math.random(-amount, amount)
  local velocity = math.max(1, math.min(127, note.velocity + offset))

  if velocity ~= note.velocity then
    arbit.set_note_velocity({
      noteId = note.id,
      velocity = velocity
    })
    changed = changed + 1
  end
end

print("Humanized " .. changed .. " notes")
