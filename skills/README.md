# Agent Skills

Skills are procedure manuals for AI coding agents (Claude Code, Cursor, Codex,
and similar tools). Each skill is a directory containing a `SKILL.md` file: YAML
frontmatter (`name`, `description`) followed by markdown instructions the agent
loads when the task matches.

They teach an agent how to drive Arbit programmatically — exact tool names,
parameter enums, the order of operations, and how to verify each step actually
worked — so the agent edits reliably instead of guessing.

## Available Skills

- `arbit-video-editing/` - end-to-end video editing through Arbit's MCP server
  or the port-9900 JSON-RPC command server: import, beat-sync, effects, LUTs,
  transitions, text overlays, automation, export, and verification. Ships a
  `knowledge/` folder of dated, locked field lessons and music-synced editorial
  rules that agents read first and append to over time.

## Install

Copy a skill directory into your project's skills folder:

```bash
# Claude Code (project-level)
cp -r skills/arbit-video-editing /path/to/your-project/.claude/skills/

# or user-level, available in every project
cp -r skills/arbit-video-editing ~/.claude/skills/
```

Or with the `skills` CLI:

```bash
npx skills add DonutsDelivery/arbit-developer-resources
```

The skill expects the Arbit standalone to be running (its command server binds
TCP port 9900). See the Prerequisites section of each `SKILL.md` for MCP
registration and the raw TCP fallback.
