# Lessons (dated, locked)

Field lessons from real agent sessions driving DonutStudio's video editor. Each entry
is dated and **locked**: do not re-litigate, re-debug, or "verify whether this
is still true" — it is, until the underlying behavior changes, at which point
the entry is updated or removed. When you learn a new hard-won lesson (a silent
no-op, an RPC gotcha, a timing constant, a verification trick), **append it
here** with a date and `locked`.

---

- **Beat detection stores markers only; stretch mode applies the warp.**
  `video_detect_clip_beats` writes beat markers onto the clip and nothing else.
  Without a follow-up `video_set_stretch` mode 1, nothing snaps to the grid —
  this is the #1 silent no-op in agent sessions. (2026-06-12, locked)

- **Stretch mode 2 + `manualBpm` is how authored media follows tempo changes.**
  Externally generated/authored clips (AI video, motion graphics, anything made
  at a known tempo) set `video_set_stretch` mode 2 with the BPM they were
  authored at; they then re-stretch automatically when the project BPM changes.
  (2026-06-12, locked)

- **Determinism: pin `encoder: "libx264"` + `interpolation: "none"`** for any
  frame-hash workflow (regression tests, CI renders, before/after comparisons).
  Hardware encoders (nvenc/VAAPI) and RIFE optical flow vary per GPU and per
  driver. (2026-06-12, locked)

- **MCP clients must restart to pick up new DonutStudio MCP tools.** The MCP server
  is spawned per client session; a tool added to `mcp-server/index.js` (or a
  new DonutStudio build exposing a new method) is invisible until the client
  restarts. If a tool you expect is missing, restart the MCP client before
  concluding it doesn't exist. (2026-06-13, locked)

- **After `split_clip`, old clipIds may be stale.** The original clip ID may no
  longer exist. Refresh with `get_clips` before any further edits in that
  region — "clip not found" mid-batch usually means you skipped this.
  (2026-06-12, locked)

- **`video_media_relink` is port-9900-only.** It is not in the MCP tool list;
  call it over the raw TCP JSON-RPC fallback with `{oldPath, newPath}`.
  (2026-06-12, locked)

- **Video tools error when the helper sidecar isn't running — and `ping` does
  not prove the helper is alive.** `ping` only proves the DonutStudio command server
  is up. The video helper starts when the Video Editor window opens or a clip
  is imported; open the Video Editor or `video_import_clip` first.
  (2026-06-12, locked)

- **Verify by re-measurement, never by assuming.** Call `video_graph_describe`
  after every mutation batch — it is the only proof params reached the
  compositor. After export, `video_probe` the output file; when beat sync
  matters, re-run beat detection on the exported audio and confirm the
  detected grid still lands on the project grid. A clean tool response is not
  evidence the edit took effect. (2026-06-13, locked)
