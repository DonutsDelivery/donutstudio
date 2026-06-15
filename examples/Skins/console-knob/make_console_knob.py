#!/usr/bin/env python3
"""Generate the "Console Knob" Arbit skin pack.

Running this script writes a complete, installable skin pack next to itself:

    console-knob.arbitskin/
      skin.json
      assets/knob.png        <- a 64-frame metallic-knob filmstrip

No binary art ships in this repository on purpose (see ../README.md); you
generate it locally with this script. Edit draw_knob() to make your own knob.

Dependency: pycairo  (pip install pycairo). On Linux you may also need Cairo's
dev package, e.g. libcairo2-dev, for pycairo to build. Pillow is NOT required.

Pointer-angle convention (must match Arbit, see docs/skinning.md):
    juceAngle = 225 + 270 * value     # degrees, 0 = straight up, clockwise
    stdAngle  = juceAngle - 90        # screen coords (y points down)
so value 0 points lower-left (~7:30) and value 1 lower-left->lower-right (~4:30),
sweeping 270 degrees clockwise across the top.
"""
import math
import os
import cairo

FRAMES = 64          # number of rotation frames; drives both the PNG and skin.json
SIZE = 128           # pixels per square frame
HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "console-knob.arbitskin")
ASSETS = os.path.join(PACK, "assets")


def draw_knob(ctx, value):
    cx = cy = SIZE / 2.0
    R = SIZE * 0.5 - 8.0

    ctx.set_operator(cairo.OPERATOR_CLEAR)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    # Outer brushed-metal ring
    ring = cairo.RadialGradient(cx - R * 0.3, cy - R * 0.3, R * 0.2, cx, cy, R + 4)
    ring.add_color_stop_rgb(0.0, 0.55, 0.57, 0.60)
    ring.add_color_stop_rgb(0.6, 0.28, 0.29, 0.32)
    ring.add_color_stop_rgb(1.0, 0.10, 0.10, 0.12)
    ctx.set_source(ring)
    ctx.arc(cx, cy, R + 4, 0, 2 * math.pi)
    ctx.fill()

    # Knob body — dark dished cylinder with a top-left sheen
    body = cairo.RadialGradient(cx - R * 0.35, cy - R * 0.35, R * 0.1, cx, cy, R)
    body.add_color_stop_rgb(0.0, 0.32, 0.33, 0.36)
    body.add_color_stop_rgb(0.55, 0.16, 0.16, 0.18)
    body.add_color_stop_rgb(1.0, 0.05, 0.05, 0.06)
    ctx.set_source(body)
    ctx.arc(cx, cy, R, 0, 2 * math.pi)
    ctx.fill()

    # Inner bevel ring (light top, dark bottom)
    ctx.set_source_rgba(1, 1, 1, 0.10)
    ctx.set_line_width(1.4)
    ctx.arc(cx, cy, R - 1.5, math.radians(200), math.radians(340))
    ctx.stroke()
    ctx.set_source_rgba(0, 0, 0, 0.5)
    ctx.arc(cx, cy, R - 1.5, math.radians(20), math.radians(160))
    ctx.stroke()

    # Accent pointer — follows Arbit's 225 + 270*value convention
    juce_angle = math.radians(225 + 270 * value)
    std_angle = juce_angle - math.pi / 2.0
    tipx = cx + (R - 4) * math.cos(std_angle)
    tipy = cy + (R - 4) * math.sin(std_angle)
    basex = cx + (R * 0.30) * math.cos(std_angle)
    basey = cy + (R * 0.30) * math.sin(std_angle)

    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_source_rgba(0.27, 0.71, 1.0, 0.35)   # glow
    ctx.set_line_width(7.5)
    ctx.move_to(basex, basey)
    ctx.line_to(tipx, tipy)
    ctx.stroke()
    ctx.set_source_rgb(0.45, 0.82, 1.0)          # core
    ctx.set_line_width(3.2)
    ctx.move_to(basex, basey)
    ctx.line_to(tipx, tipy)
    ctx.stroke()

    # Center cap
    cap = cairo.RadialGradient(cx - 2, cy - 2, 1, cx, cy, R * 0.32)
    cap.add_color_stop_rgb(0.0, 0.40, 0.41, 0.44)
    cap.add_color_stop_rgb(1.0, 0.12, 0.12, 0.14)
    ctx.set_source(cap)
    ctx.arc(cx, cy, R * 0.30, 0, 2 * math.pi)
    ctx.fill()


def main():
    os.makedirs(ASSETS, exist_ok=True)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, SIZE, SIZE * FRAMES)
    ctx = cairo.Context(surface)
    for i in range(FRAMES):
        value = i / (FRAMES - 1)
        ctx.save()
        ctx.translate(0, i * SIZE)
        ctx.rectangle(0, 0, SIZE, SIZE)
        ctx.clip()
        draw_knob(ctx, value)
        ctx.restore()
    png_path = os.path.join(ASSETS, "knob.png")
    surface.write_to_png(png_path)
    print("wrote", png_path, "(%d x %d)" % (surface.get_width(), surface.get_height()))

    manifest = (
        '{\n'
        '  "name": "Console Knob",\n'
        '  "assets": {\n'
        '    "knob": {\n'
        '      "type": "filmstrip",\n'
        '      "file": "knob.png",\n'
        '      "frameCount": %d,\n'
        '      "orientation": "vertical"\n'
        '    }\n'
        '  }\n'
        '}\n'
    ) % FRAMES
    manifest_path = os.path.join(PACK, "skin.json")
    with open(manifest_path, "w") as f:
        f.write(manifest)
    print("wrote", manifest_path)
    print("\nPack ready:", PACK)
    print("Install it by copying that folder into Arbit's Themes folder.")


if __name__ == "__main__":
    main()
