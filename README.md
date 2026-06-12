# Arbit Developer Resources

Developer resources, Lua scripting examples, and download links for Arbit.

Arbit itself is closed source. This repository contains public documentation,
examples, and integration resources for scripting and extension workflows. It
does not contain the Arbit application source code, build system, license server,
DSP engines, model integrations, or private product assets.

## Downloads

- Arbit Free: https://donutsdelivery.online/arbit/download
- Arbit Pro: https://donutsdelivery.online/arbit/buy
- Product pages: https://donutsdelivery.online/arbit

## What Is Here

- `docs/lua-scripting.md` - scripting overview and install location.
- `docs/api-reference.md` - current public `arbit` scripting API surface.
- `examples/` - ready-to-edit scripts grouped by category.
- `skills/` - agent skills for AI coding tools.
- `.github/ISSUE_TEMPLATE/` - issue templates for script bugs and API requests.

## Quick Start

1. Open Arbit.
2. Open `Settings -> Scripts -> Open Folder`.
3. Copy a `.lua` file from `examples/` into that scripts folder.
4. Press `Refresh` in Arbit's Scripts tab.
5. Run the script.

Scripts run inside Arbit and receive a global `arbit` table. Most functions take
a Lua table of named parameters and return regular Lua values converted from
Arbit's JSON-style command responses.

```lua
local notes = arbit.get_notes()

for _, note in ipairs(notes) do
  if note.velocity < 40 then
    arbit.set_note_velocity({
      noteId = note.id,
      velocity = 80
    })
  end
end

print("Updated " .. #notes .. " notes")
```

## Agent Skills

AI coding agents (Claude Code, Cursor, Codex) can drive Arbit directly through
its MCP server or the port-9900 command server. The
[`arbit-video-editing` skill](skills/arbit-video-editing/SKILL.md) teaches an
agent the full video workflow — import, beat-sync, effects, transitions, text
overlays, automation, export, and verification — with exact tool names and
parameter tables. See [`skills/README.md`](skills/README.md) for what skills
are and how to install them.

## Contributing

Contributions are welcome for documentation, examples, script ideas, and API
requests. Keep scripts small, readable, and focused on one useful workflow.

Do not submit proprietary Arbit source code, reverse-engineered internals, paid
content, private beta keys, or third-party assets you do not have permission to
redistribute.

## License

Documentation and example scripts in this repository are MIT licensed unless a
file states otherwise.
