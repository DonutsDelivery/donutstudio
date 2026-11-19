# Contributing a shader pack

DonutStudio's Media Machine loads **shader packs** — self-contained folders of
GLSL/ISF visual generators that show up in the in-app **Shader Library** browser
and the `video_get_shader_packs` RPC. This document is the format reference for
community contributors. (For publishing a finished pack to the in-app store, see
`arbit-essentials/PUBLISHING.md`.)

## Folder layout

```
my-pack/
  pack.json               # manifest (required)
  shaders/                # the shader sources referenced by the manifest
    aurora_drift.fs
    ...
  thumbnails/             # optional pre-baked PNG previews, named <shaderId>.png
    aurora_drift.png
    ...
  README.md               # optional
  LICENSE                 # include if your license notice must be retained
```

Packs are discovered from (in priority order): the installed data dir
(`shader-packs/` under DonutStudio's data directory, where store downloads land),
next to the executable, and the dev repo's `content/shader-packs/`. A pack `id`
found in an earlier root shadows the same id in a later one.

## `pack.json` schema

```jsonc
{
  "pack": {
    "id":          "my-pack-v1",          // unique pack id (dedup key)
    "name":        "My Pack",             // display name
    "version":     "1.0",
    "author":      "Your Name",
    "license":     "CC0-1.0",             // SPDX id; ship only what you may
    "format":      "glsl",                // "glsl" or "isf"
    "description": "One-paragraph summary."
  },
  "shaders": [
    {
      "id":          "aurora_drift",       // unique within the pack; also the
                                           // thumbnail base name (thumbnails/aurora_drift.png)
      "file":        "shaders/aurora_drift.fs",  // path relative to the pack folder
      "name":        "Aurora Drift",       // display name
      "category":    "nature",             // OPTIONAL curation theme (see below)
      "reactsTo":    ["clock", "audio"],   // musical signals that drive the visual
      "needs":       [],                   // human-readable prerequisites, e.g.
                                           // ["harmonic links / JI tuning"] or
                                           // ["an input image / clip"]
      "description": "Flowing aurora curtains that scroll with the beat."
    }
  ]
}
```

### Field notes

- **`reactsTo`** documents which reactivity sources the shader consumes — any of
  `"clock"` (Block A transport/beat), `"audio"` (Block B mix analysis: RMS/peak/
  onset/FFT), `"score"` (Block C symbolic notes / JI lattice). The browser renders
  these as coloured chips on each card.
- **`needs`** lists human-readable prerequisites shown to the user (e.g. a filter
  that needs an input clip, or a shader that only spreads out when notes carry
  harmonic-link / JI prime structure).
- **Thumbnails** are static PNGs named `thumbnails/<shaderId>.png`. The manifest
  does not list them; the browser resolves them by convention. Packs may omit a
  preview when the shader cannot produce a representative frame from available
  inputs; the browser then shows its "no preview" placeholder.
  Run `tools/generate_shader_pack_thumbnails.py` with DonutStudio open to render
  missing bundled previews. Every preview must be a frame produced by the listed
  shader through DonutStudio's production compositor. Load representative score
  or audio input before rendering shaders that depend on it.

### `category` — curation theme (optional)

The Shader Library groups cards into a filterable theme strip. Set a shader's
`"category"` to one of the canonical ids to file it under that theme; the value is
matched case-insensitively. Any shader with no `category` (or an unrecognised one)
is grouped under **Uncategorized**.

| id             | theme         | what belongs here                                                        |
|----------------|---------------|--------------------------------------------------------------------------|
| `psychedelic`  | Psychedelic   | Warping, kaleidoscopic, feedback-heavy — tunnels, swirls, plasmas, trails |
| `minimalist`   | Minimalist    | Clean geometry, restrained motion — grids, gradients, single shapes       |
| `retro`        | Retro         | Vintage screen looks — CRT, VHS, film burn, synthwave grids, TV static    |
| `nature`       | Nature        | Organic/atmospheric — auroras, water, smoke, light rays, blooms           |

The canonical list (with descriptions used in docs/store copy) lives in
`categories.json` next to this file. The theme strip only shows a tab for a theme
that has at least one shader, plus **All** and (when present) **Uncategorized**.

## Checklist before submitting

- [ ] `pack.json` parses as JSON and every `file` path resolves.
- [ ] Each shader has a unique `id` and a short, human `description`.
- [ ] `category` (if set) is one of the four canonical ids.
- [ ] `reactsTo` / `needs` reflect what the shader actually consumes/requires.
- [ ] Every bundled generator has a `thumbnails/<shaderId>.png` preview.
- [ ] You have the right to distribute every source under the stated `license`.
