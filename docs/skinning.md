# Skinning

Arbit skins are optional, community-made packs that replace the look of certain
controls with your own artwork. A skin is a folder named `Something.arbitskin`
that contains a `skin.json` manifest and, optionally, image assets.

> **Preview / Status**
>
> The skin engine is an early **preview** feature. Treat this guide as
> describing an experimental, unstable format.
>
> - Requires a build of Arbit that includes the skin engine. It is **not** in an
>   earlier numbered release; if your build cannot list or set skins (see
>   *Activating a Skin* below), it does not ship the engine yet.
> - In this preview, **knobs are the only thing a skin can replace.** Panels,
>   buttons, meters, fonts, and colors are **not** skinnable yet. Do not author
>   for them.
> - There is **no Theme-tab picker yet.** Today a skin is activated only by
>   dropping the folder in place and switching to it over the command server
>   (described below). A graphical picker is planned.
> - **The format will grow and may change.** Future versions are expected to add
>   color, metric, and font settings and more asset types. Keep your packs small
>   and expect to revisit them.

## What a Skin Is

Arbit's normal look is drawn from its built-in color and shape system. A skin
does not throw that away — it **layers on top of the default look**. Anything a
skin does not provide keeps using the built-in default.

This means a pack can override **just the knob** and leave everything else
exactly as it ships. If a skin has no usable artwork for a control, Arbit
automatically falls back to its built-in vector drawing for that control. There
is no way to make Arbit "blank" — a missing or broken asset simply means the
default is used.

A skin that contains only a `skin.json` (and no images) is valid. It loads and
behaves exactly like the default look until you add artwork.

## Skin Folder Layout

A skin pack is a single folder ending in `.arbitskin`:

```text
Console Knob.arbitskin/
  skin.json
  assets/
    knob.png
```

- `skin.json` is the manifest. It is required. Without it the folder is ignored.
- `assets/` holds the image files the manifest points at. Arbit looks for an
  asset first inside `assets/`, and if it is not there, next to `skin.json` in
  the top of the folder. Keeping art in `assets/` is the recommended layout.

## The skin.json Manifest

`skin.json` is plain JSON. In this preview the loader reads only the fields shown
below. Any other field you add (for example a `description` or `author` for your
own notes) is **ignored** by the current build — it is harmless, but it does
nothing yet.

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

### Fields the loader reads

| Field | Meaning |
|-------|---------|
| `name` | The display name shown when listing skins. If you leave it out, the folder name (without `.arbitskin`) is used instead. |
| `assets` | An object holding one entry per skinnable control. In this preview the only entry that does anything is the knob. |
| `assets.knob` | The knob artwork. The key may also be written `knob.default`; if both are present, `knob` wins. |

### Fields inside the knob entry

| Field | Meaning |
|-------|---------|
| `type` | Must be `filmstrip`. Any other value means the knob entry is skipped and the default knob is used. Case does not matter. |
| `file` | The image file name, e.g. `knob.png`. Looked up in `assets/` first, then at the top of the folder. |
| `frameCount` | The number of frames in the filmstrip image (a whole number). |
| `orientation` | `vertical` (frames stacked top-to-bottom) or `horizontal` (frames laid left-to-right). Any value other than `horizontal` is treated as `vertical`. |

If the image cannot be loaded, or `frameCount` is 1 or less, the knob silently
falls back to the built-in vector knob.

## How a Filmstrip Knob Works

A knob skin is a **filmstrip**: a single image holding every frame of the knob's
rotation, from fully turned down to fully turned up, packed in order along one
axis.

- With `orientation: "vertical"`, the frames are stacked top to bottom. The
  image is one frame wide and `frameCount` frames tall. Each frame's height is
  the image height divided by `frameCount`.
- With `orientation: "horizontal"`, the frames run left to right. The image is
  `frameCount` frames wide and one frame tall.

When Arbit draws a knob at a value between 0 (fully down) and 1 (fully up), it
picks the frame nearest that value. The rule is:

```text
frameIndex = round(value * (frameCount - 1))
```

So for a 64-frame strip: value 0 shows frame 0, value 1 shows frame 63, and a
value of 0.5 shows frame 32 (`round(0.5 * 63) = round(31.5) = 32`). The chosen
frame is drawn centered in a square inside the knob's space.

Practical tips for the art itself:

- Make the image divide evenly: image height (vertical) or width (horizontal)
  should be a clean multiple of `frameCount`, so every frame is the same size.
  For example, a 64-frame vertical strip that is 128 pixels wide works well at
  `128 x 8192` (64 frames of `128 x 128`).
- Use a transparent background. Arbit fills behind the knob, so your frames
  should be cut out, not on a solid block.
- Draw the frames in order from minimum to maximum. Frame 0 is the lowest value;
  the last frame is the highest.
- More frames means smoother motion; fewer frames is fine for a chunky,
  stepped look.

## Install Location

Skins live in Arbit's **Themes** data folder. Create the folder if it does not
exist, and drop your `Something.arbitskin` folder inside it:

```text
Linux:    ~/.local/share/Arbit/Themes/
macOS:    ~/Library/Application Support/Arbit/Themes/
Windows:  %APPDATA%\Arbit\Themes\
```

So a finished pack ends up like:

```text
~/.local/share/Arbit/Themes/Console Knob.arbitskin/skin.json
~/.local/share/Arbit/Themes/Console Knob.arbitskin/assets/knob.png
```

Arbit scans the Themes folder for `*.arbitskin` directories that contain a
`skin.json`.

## Activating a Skin

There is **no graphical skin picker yet** (a Theme-tab picker is planned). In
this preview you switch skins through Arbit's command server, which the Arbit
standalone app runs on TCP port `9900` while it is open. This is the same server
the scripting and agent tooling use.

That command server is **not** an HTTP server, so `curl` will not work. It speaks
**newline-delimited JSON-RPC over a raw TCP socket**: you send one JSON object on
a single line ending in a newline, and read one JSON line back. A normal user
with no automation tooling cannot switch skins yet — this path is for people
comfortable talking to that port.

Two methods are available:

- `list_skins` — returns the skins Arbit found in the Themes folder (always
  including `Default`) and which one is active.
- `set_skin` — switches to a skin by its `name`. Pass `Default` to return to the
  built-in look. The visible knobs repaint immediately.

A small client in Python 3 (no extra packages needed):

```python
import socket, json

def arbit_rpc(method, params=None):
    s = socket.create_connection(("localhost", 9900), timeout=5)
    s.sendall((json.dumps({"method": method, "params": params or {}, "id": 1}) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return buf.decode().strip()

print(arbit_rpc("list_skins"))                          # what's available, and the active one
print(arbit_rpc("set_skin", {"name": "Console Knob"}))  # switch to a skin by its manifest name
print(arbit_rpc("set_skin", {"name": "Default"}))       # back to the built-in look
```

If you prefer the shell, `nc` works too — it is one line in, one line out (some
`nc` builds need a short linger flag such as `-q1` or `-N` so they wait for the
reply):

```bash
printf '{"method":"list_skins","id":1}\n' | nc localhost 9900
printf '{"method":"set_skin","params":{"name":"Console Knob"},"id":1}\n' | nc localhost 9900
```

The `name` you pass to `set_skin` is the `name` from the manifest (or the folder
name if you left `name` out). `set_skin` returns whether the switch worked and
which skin ended up active.

## A Note on Sharing Skins

Knob filmstrip images are binary art, not text, so finished skin packs are meant
to be shared through Arbit's online content store rather than committed into the
developer-resources repository. When you make a pack to share, use original
artwork or art you are clearly allowed to redistribute (for example, CC0 or your
own work), and say so in the pack.

The [`examples/Skins/console-knob`](../examples/Skins/console-knob) example ships
a small Python **generator** instead of a committed image, so you can produce the
artwork yourself and see exactly how a working pack is built.

## Example

Here is a complete, minimal pack. It overrides only the knob and lets everything
else stay default.

Folder:

```text
My First Knob.arbitskin/
  skin.json
  assets/
    knob.png
```

`skin.json`:

```json
{
  "name": "My First Knob",
  "assets": {
    "knob": {
      "type": "filmstrip",
      "file": "knob.png",
      "frameCount": 48,
      "orientation": "vertical"
    }
  }
}
```

`assets/knob.png` is a vertical filmstrip: 48 frames stacked top to bottom, each
the same size, drawn from the knob fully turned down (frame 0, at the top) to
fully turned up (frame 47, at the bottom), on a transparent background. A clean
size would be `128 x 6144` (48 frames of `128 x 128`).

To use it:

1. Copy the `My First Knob.arbitskin` folder into the Themes folder for your
   platform (see *Install Location*).
2. With Arbit running, send `list_skins` to confirm `My First Knob` shows up.
3. Send `set_skin` with `name` set to `My First Knob` (see *Activating a Skin*).
   The visible knobs repaint immediately; send `set_skin` with `Default` to
   switch back.
