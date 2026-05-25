-- Print track names, instruments, and levels
local tracks = arbit.get_tracks()

for _, track in ipairs(tracks) do
  local name = track.name or ("Track " .. tostring(track.trackId))
  local synth = track.synthType or "unknown"
  local volume = track.volume or 0
  print(track.trackId .. ": " .. name .. " | " .. synth .. " | volume " .. string.format("%.2f", volume))
end
