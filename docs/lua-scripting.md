# Lua Scripting

Arbit scripting uses an embedded Lua 5.4 runtime exposed through a global
`arbit` table. Script files use the `.lua` extension.

## Script Location

Open Arbit and use:

`Settings -> Scripts -> Open Folder`

Arbit scans that folder recursively. Subfolders become categories in the Scripts
tab, so this layout works well:

```text
scripts/
  Analysis/
    note-stats.lua
  Generators/
    octave-double.lua
  Transforms/
    humanize-velocity.lua
```

## Script Metadata

The first line comment is used as a short description in the Scripts tab:

```lua
-- Humanize note velocities by a small random amount
```

## Runtime Model

- Each script runs in a fresh Lua state.
- Scripts can use `base`, `table`, `string`, `math`, and `utf8`.
- File loading and module loading are disabled for safety.
- `print()` output is captured and shown after the script finishes.
- Scripts call Arbit through the global `arbit` table.

## Parameter Style

Most mutating functions take a table of named parameters:

```lua
arbit.create_note({
  midiNote = 60,
  startBeat = 0,
  lengthBeats = 1,
  velocity = 100,
  trackId = 0
})
```

Read functions usually take no arguments:

```lua
local notes = arbit.get_notes()
```

## Best Practices

- Read the current state first with `arbit.get_notes()`, `arbit.get_links()`, or
  `arbit.get_tracks()`.
- Keep transforms reversible by using Arbit operations instead of trying to edit
  project files directly.
- Avoid destructive scripts unless the script name makes that obvious.
- Use `print()` to report what changed.
- Prefer small scripts that do one thing well.

## Example

```lua
-- Set quiet notes to a minimum velocity
local notes = arbit.get_notes()
local changed = 0

for _, note in ipairs(notes) do
  if note.velocity < 50 then
    arbit.set_note_velocity({
      noteId = note.id,
      velocity = 50
    })
    changed = changed + 1
  end
end

print("Raised " .. changed .. " quiet notes")
```
