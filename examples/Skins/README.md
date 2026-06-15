# Skin Examples

Skin packs are example **looks** for Arbit, the same way the other folders here
hold example scripts:

```text
examples/
  Analysis/      Lua scripts that read your project
  Generators/    Lua scripts that create notes
  Transforms/    Lua scripts that change notes
  Skins/         skin packs that change how controls are drawn  <- you are here
```

A skin is not a script. It is a small folder named `Something.arbitskin` that
holds a `skin.json` manifest and image artwork. Arbit reads it and draws certain
controls from your art instead of its built-in look. See
[`docs/skinning.md`](../../docs/skinning.md) for the full format.

> **Preview / Status**
>
> The skin engine is an early **preview** feature. Treat these examples as
> describing an experimental, unstable format.
>
> - Requires a build of Arbit that includes the skin engine. It is **not** in an
>   earlier numbered release; if your build cannot list or set skins, it does not
>   ship the engine yet.
> - In this preview, **knobs are the only thing a skin can replace.** Panels,
>   buttons, meters, fonts, and colors are **not** skinnable yet. Do not author
>   for them.
> - There is **no Theme-tab picker yet.** Today a skin is activated only by
>   dropping the folder in place and switching to it over the command server. A
>   graphical picker is planned.
> - **The format will grow and may change.** Future versions are expected to add
>   color, metric, and font settings and more asset types. Keep your packs small
>   and expect to revisit them.

## No Binary Art In This Repo

A knob skin needs an image (a filmstrip). Image files are **not** committed to
this repository — finished, shareable packs belong in Arbit's online content
store, and committing binary art here is out of scope (see the repo's
`CONTRIBUTING.md`).

So each example here ships a small, readable **generator script** that produces
the artwork on your machine. You run the script, it writes a complete
`*.arbitskin` pack next to itself, and then you install that pack. This also
shows you exactly how the art is built, so you can edit it into your own knob.

## Available Examples

- [`console-knob/`](console-knob/) — a metallic console-style filmstrip knob.
  Run its generator to produce the pack, then install it.

## Using an Example

Every example here follows the same three steps:

1. **Generate the pack.** Run the example's Python generator. It writes a
   `*.arbitskin` folder (manifest + `assets/` art) next to the script.

   ```bash
   cd console-knob
   python3 make_console_knob.py     # writes console-knob.arbitskin/
   ```

2. **Install it.** Copy the generated `*.arbitskin` folder into Arbit's
   **Themes** folder for your platform:

   ```text
   Linux:    ~/.local/share/Arbit/Themes/
   macOS:    ~/Library/Application Support/Arbit/Themes/
   Windows:  %APPDATA%\Arbit\Themes\
   ```

3. **Activate it.** There is no in-app picker yet. With Arbit running, switch to
   the skin over its command server (newline-delimited JSON-RPC on raw TCP port
   `9900` — not HTTP). See [`docs/skinning.md`](../../docs/skinning.md#activating-a-skin)
   for a ready-to-paste client; the short version:

   ```bash
   printf '{"method":"set_skin","params":{"name":"Console Knob"},"id":1}\n' | nc localhost 9900
   printf '{"method":"set_skin","params":{"name":"Default"},"id":1}\n'      | nc localhost 9900
   ```

## Making Your Own

Edit a generator, re-run it, and you have a new pack. Knob art is a
**filmstrip**: one image holding every rotation frame in order, from the knob
fully turned down to fully turned up. The pack's own `README.md` and
[`docs/skinning.md`](../../docs/skinning.md) explain the filmstrip rules and the
pointer-angle convention your frames must follow.

## Sharing

Use original artwork or art you are clearly allowed to redistribute (for example
CC0 or your own work), and say so in the pack. Finished packs are shared through
Arbit's online content store, not committed here.
