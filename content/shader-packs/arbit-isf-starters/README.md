# DonutStudio ISF Starters

Five free, original **ISF** (Interactive Shader Format) generators for the
**DonutStudio Media Machine**. Where the [DonutStudio Essentials](../arbit-essentials) pack
shows off the native DonutStudio dialect, this pack shows off **ISF import**: each
shader declares ISF `INPUTS`, and every INPUT becomes a real, addressable DonutStudio
clip param.

**License: CC0-1.0 (public domain).** These are original works written for DonutStudio
(see *Provenance* below) — they are **not** taken from the ISF gallery or
interactiveshaderformat.com.

| Shader | INPUTS | Reacts to | Showcases |
|---|---|---|---|
| `warp_tunnel` | speed, twist, ribs, colorA, colorB | clock + audio | float + color inputs |
| `kaleido` | segments, zoom, spin, tint | clock | a clean mod-matrix target (kick→segments, bass→zoom) |
| `plasma` | scale, speed, hueShift, brightness | audio + clock | float inputs, audio-reactive backdrop |
| `orbit` | center, count, size, spread, color | audio + clock | the less-common **point2D** and **long** input types |
| `light_trails` | decay, spin, zoom, tint | clock + audio | **ISF multipass** — a `PERSISTENT` feedback buffer (PASSES) smearing a beat-driven source into endless trails |

## Why ISF here

When you import one of these, DonutStudio parses the ISF header and exposes each INPUT
as a generator param under the one param namespace:

```
clip<id>/gen/<inputName>      e.g.  clip7/gen/segments,  clip7/gen/zoom
```

That id is the same one automation lanes and the **mod matrix** speak, so an ISF
INPUT is automatable on a lane *and* a valid mod-routing destination with no
extra wiring. For example, with `kaleido` on clip 7:

- route a kick (`NoteTrigger`, pitch-filtered) → `clip7/gen/segments`
- route a bassline (`NotePitch`) → `clip7/gen/zoom`

…and the kaleidoscope fans and breathes with the track. That is the M6×M7
synthesis — *the mod matrix driving a shader's own inputs* — using only stock
ISF.

## Using a shader

Each `.fs` is a single-pass ISF fragment shader. Import one with:

- **Via the command server / MCP** — `video_import_shader { path }` (it reads the
  file, validates the ISF header, and seeds each INPUT's default), or
- **In DonutStudio** — create a shader generator clip and point it at the file.

The shader runs in the live viewport and exports byte-identically (export ==
preview by construction).

## The dialect, in one glance

These use **standard ISF** uniforms, which DonutStudio shims onto its own contract, so
real ISF shaders just work:

- `TIME`, `TIMEDELTA`, `FRAMEINDEX`, `RENDERSIZE`, `isf_FragNormCoord`,
  `IMG_PIXEL` / `IMG_NORM_PIXEL` / `IMG_SIZE`
- INPUTS arrive as `uniform`s of the matching GLSL type (`float`→`float`,
  `bool`→`bool`, `long`→`int`, `color`→`vec4`, `point2D`→`vec2`,
  `image`→`sampler2D`).

On top of that, DonutStudio exposes its **musical prelude** to ISF shaders.
The starters demonstrate `uRMS`, `uOnset`, spectrum bands, `beatPulse(width)`,
and the Block C score API (`arbitNote`, `arbitLink`, `lattice2`). See
`video-helper/src/shader_dialect.h` for the authoritative list.

## Publishing as store content

This directory is the source for a downloadable **Visuals** store pack, the same
as the Essentials pack. The store catalog is authoritative on the VPS, not in the
repo — to publish, zip this folder, host it, and add a catalog entry on the VPS.
The zip/upload steps and a ready-to-edit catalog snippet are in
[`../arbit-essentials/PUBLISHING.md`](../arbit-essentials/PUBLISHING.md) (use the
same recipe with this pack's id `arbit-isf-starters-v1`).

## Provenance

All five shaders are original works authored for DonutStudio. The visual techniques
(tunnel, kaleidoscope, plasma, orbiting glows, feedback trails) are generic primitives, written
here from scratch with original math, and combined with DonutStudio's own dialect
uniforms. They are not derived from any specific gallery/Shadertoy/ISF shader, so
they carry no CC-BY-NC-SA / unverifiable-upload lineage and are safe to ship as
CC0 (master-plan principle #4, similarity review). Each was verified on the real
helper: `shader_compile` reports `dialect=isf, ok=true` with every INPUT
discovered, and a full offscreen-GL export renders a non-black, structured,
time-reactive frame.
