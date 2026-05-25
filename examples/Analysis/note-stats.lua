-- Print basic statistics about the current arrangement
local notes = arbit.get_notes()
local links = arbit.get_links()

if #notes == 0 then
  print("No notes in the arrangement")
  return
end

local minPitch = 127
local maxPitch = 0
local totalVelocity = 0
local totalLength = 0
local byTrack = {}

for _, note in ipairs(notes) do
  minPitch = math.min(minPitch, note.midiNote)
  maxPitch = math.max(maxPitch, note.midiNote)
  totalVelocity = totalVelocity + note.velocity
  totalLength = totalLength + note.lengthBeats

  local trackId = note.trackId or 0
  byTrack[trackId] = (byTrack[trackId] or 0) + 1
end

print("Notes: " .. #notes)
print("Harmonic links: " .. #links)
print("Pitch range: " .. minPitch .. "-" .. maxPitch)
print("Average velocity: " .. math.floor(totalVelocity / #notes))
print("Average length: " .. string.format("%.2f", totalLength / #notes) .. " beats")

for trackId, count in pairs(byTrack) do
  print("Track " .. trackId .. ": " .. count .. " notes")
end
