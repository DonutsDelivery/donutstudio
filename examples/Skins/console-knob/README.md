# Console Knob

A skin pack that replaces Arbit's knobs with a metallic console-style rotary:
a solid dished body, a brushed-metal ring, and a bright cyan accent pointer.
It is deliberately different from Arbit's built-in glass-arc knob, so the swap
is unmistakable. Only knobs change; everything else stays the default look.

> **Preview / Status**
>
> The skin engine is an early **preview** feature. Treat this pack as
> describing an experimental, unstable format.
>
> - Requires a build of Arbit that includes the skin engine. It is **not** in an
>   earlier numbered release; if your build cannot list or set skins, it does not
>   ship the engine yet.
> - In this preview, **knobs are the only thing a skin can replace.** Panels,
>   buttons, meters, fonts, and colors are **not** skinnable yet.
> - There is **no Theme-tab picker yet.** Today a skin is activated only by
>   dropping the folder in place and switching to it over the command server. A
>   graphical picker is planned.
> - **The format will grow and may change.** Keep packs small and expect to
>   revisit them.

## What Is In This Folder

```text
console-knob/
  README.md               this file
  make_console_knob.py     the generator — run it to build the pack
```

No image is committed here. You **generate** the pack by running the script,
which writes:

```text
console-knob.arbitskin/
  skin.json                manifest: names the skin, points at the art
  assets/
    knob.png               64-frame vertical filmstrip (128 x 8192)
```

The generated `skin.json` reads:

```json
{
  "name": "Console Knob",
  "assets": {
    "knob": {
      "type": "filmstrip",
      "file": "knob.png",
      "frameCount": 64,
      "orientation": "vertical"
    }
  }
}
```

The `knob.png` is a 64-frame filmstrip stacked top to bottom on a transparent
background, drawn from the knob fully turned down (frame 0, at the top) to fully
turned up (frame 63, at the bottom). See
[`docs/skinning.md`](../../../docs/skinning.md) for the full manifest reference.

## 1. Generate It

```bash
python3 make_console_knob.py
```

This writes `console-knob.arbitskin/` (manifest + `assets/knob.png`) next to the
script.

**Dependency:** [pycairo](https://pypi.org/project/pycairo/) — `pip install
pycairo`. It is the Python binding for the system Cairo library, used to draw and
write the PNG. On Linux you may also need Cairo's development package (for
example `libcairo2-dev`) for pycairo to build. Pillow is **not** required.

## 2. Install It

Copy the generated `console-knob.arbitskin` folder into Arbit's **Themes** folder
for your platform:

```text
Linux:    ~/.local/share/Arbit/Themes/
macOS:    ~/Library/Application Support/Arbit/Themes/
Windows:  %APPDATA%\Arbit\Themes\
```

So a finished install looks like:

```text
~/.local/share/Arbit/Themes/console-knob.arbitskin/skin.json
~/.local/share/Arbit/Themes/console-knob.arbitskin/assets/knob.png
```

Arbit scans the Themes folder for `*.arbitskin` folders that contain a
`skin.json`.

## 3. Activate It

There is no in-app picker yet. With Arbit running, switch skins over its command
server. It is **newline-delimited JSON-RPC on raw TCP port `9900`** — not HTTP,
so `curl` will not work. See
[`docs/skinning.md`](../../../docs/skinning.md#activating-a-skin) for a
ready-to-paste Python client. The quick `nc` version (some `nc` builds need
`-q1` or `-N` to wait for the reply):

```bash
# Confirm Arbit found the pack
printf '{"method":"list_skins","id":1}\n' | nc localhost 9900

# Turn it on (the name is the manifest "name", not the folder name)
printf '{"method":"set_skin","params":{"name":"Console Knob"},"id":1}\n' | nc localhost 9900

# Go back to the built-in look
printf '{"method":"set_skin","params":{"name":"Default"},"id":1}\n' | nc localhost 9900
```

Every knob in Arbit repaints immediately when the skin changes.

## Make Your Own Knob

`make_console_knob.py` is plain Python; the drawing happens in
`draw_knob(ctx, value)`, where `value` runs from `0.0` (knob fully down) to `1.0`
(knob fully up). Edit it, re-run it, and you have a new pack.

### The Pointer-Angle Convention (Read This Before Changing the Pointer)

Arbit places a knob's pointer using a fixed angle convention, and your frames
must match it or the pointer will point the wrong way. The rule (degrees,
`0` = straight up, increasing clockwise) is:

```text
juceAngle = 225 + 270 * value
```

So a knob fully down (`value = 0`) points to the lower-left (about 7:30) and a
knob fully up (`value = 1`) points to the lower-right (about 4:30), sweeping
270 degrees clockwise across the top. The generator converts that to screen
coordinates (y pointing down) before drawing:

```python
juce_angle = math.radians(225 + 270 * value)
std_angle  = juce_angle - math.pi / 2.0   # screen coords, y down
```

Keep this formula when you redraw the pointer. The body, ring, and accent color
are yours to change freely, but the pointer must follow `225 + 270 * value` so
it lines up with the value Arbit is showing.

### frameCount

`frameCount` is how many rotation frames are stacked in the filmstrip, and it
must match the image exactly. The generator's `FRAMES` constant drives both the
rendered image and the `frameCount` written into `skin.json`, so they always
agree:

```python
FRAMES = 64   # change this and both the PNG and skin.json update together
```

Arbit chooses a frame from the knob's value with:

```text
frameIndex = round(value * (frameCount - 1))
```

More frames means smoother motion (and a taller image); fewer frames gives a
chunky, stepped look. If you author the PNG by hand instead of with this script,
make sure the image height is an exact multiple of `frameCount` (every frame the
same size), and set `frameCount` to the real number of frames — otherwise the
wrong slice is drawn.

## License

The generator is MIT (per this repository's `LICENSE`) and the artwork it
produces is CC0 — `knob.png` is generated procedurally by `make_console_knob.py`
with no third-party imagery. Reuse, edit, and redistribute freely.
