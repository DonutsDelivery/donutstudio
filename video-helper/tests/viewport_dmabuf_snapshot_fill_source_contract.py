#!/usr/bin/env python3
# Contract for the docked viewport's Dmabuf rung paint(). On the non-Mac
# zero-copy rung component painting is disabled, so paint() only feeds
# software snapshots (screenshots). The opaque component must therefore fill
# its bounds on that path before the overlays, or the snapshot shows
# uninitialised memory. The Mac transparent layer must stay as it is.
from pathlib import Path
import re
import sys

if len(sys.argv) > 1:
    video_helper = Path(sys.argv[1]).resolve()
else:
    video_helper = Path(__file__).resolve().parent.parent
header_path = video_helper.parent / "plugin/Source/VideoViewportComponent.h"
text = header_path.read_text()
errors = []


def strip_line_comments(source):
    return re.sub(r"//[^\n]*", "", source)


def block_after(source, start_pattern):
    """Return the brace-balanced block that follows the first start_pattern match."""
    match = re.search(start_pattern, source)
    if match is None:
        errors.append("missing: " + start_pattern)
        return ""
    open_brace = source.find("{", match.end())
    if open_brace < 0:
        errors.append("no body after: " + start_pattern)
        return ""
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace:index + 1]
    errors.append("unbalanced braces after: " + start_pattern)
    return ""


def non_mac_lines(block):
    """Yield (line_text, is_non_mac) for each line, honouring JUCE_MAC conditionals."""
    stack = []
    out = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#if"):
            if re.fullmatch(r"#if\s+JUCE_MAC", stripped):
                stack.append("mac")
            elif re.fullmatch(r"#if\s*!\s*JUCE_MAC", stripped):
                stack.append("notmac")
            else:
                stack.append("other")
            out.append((line, False))
            continue
        if stripped == "#else" and stack:
            top = stack[-1]
            stack[-1] = {"mac": "notmac", "notmac": "mac"}.get(top, "other")
            out.append((line, False))
            continue
        if stripped.startswith("#endif") and stack:
            stack.pop()
            out.append((line, False))
            continue
        is_non_mac = bool(stack) and "mac" not in stack and "notmac" in stack
        out.append((line, is_non_mac))
    return out


paint_body = block_after(text, r"void paint \(juce::Graphics& g\) override")
dmabuf_match = re.search(r"if \(rung == Rung::Dmabuf\)", paint_body)
if dmabuf_match is None:
    errors.append("paint() has no 'if (rung == Rung::Dmabuf)' branch")
    dmabuf_block = ""
else:
    dmabuf_block = block_after(paint_body[dmabuf_match.start():], r"if \(rung == Rung::Dmabuf\)")
code = strip_line_comments(dmabuf_block)

if dmabuf_block:
    lines = non_mac_lines(code)
    fill_index = None
    offset = 0
    for line, is_non_mac in lines:
        if is_non_mac and "g.fillAll (juce::Colours::black);" in line:
            fill_index = offset
            break
        offset += len(line) + 1
    overlay_index = code.find("paintViewportOverlay (g);")
    if fill_index is None:
        errors.append("non-Mac path of Dmabuf branch does not fillAll black")
    elif overlay_index < 0:
        errors.append("Dmabuf branch no longer paints viewport overlays")
    elif fill_index > overlay_index:
        errors.append("non-Mac fillAll comes after paintViewportOverlay")

    mac_transparent = [line for line, is_non_mac in lines
                       if "transparentBlack" in line and not is_non_mac]
    if not mac_transparent:
        errors.append("Mac transparentBlack handling missing from Dmabuf branch")
    if "isGuiComposited" not in code or "softwareSnapshotFrame_" not in code:
        errors.append("Mac composited snapshot path missing from Dmabuf branch")

ctor_body = block_after(text, r"explicit VideoViewportComponent \(VideoSidecar& sidecarToUse\)")
if "setComponentPaintingEnabled (false);" not in ctor_body:
    errors.append("constructor no longer disables component painting "
                  "(the fill would reach the screen, reason not established)")

if errors:
    print("viewport dmabuf snapshot fill contract failed: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport dmabuf snapshot fill source contract: PASS")
