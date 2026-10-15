#!/usr/bin/env python3
# Contract for the shm docked viewport open. A larger frame ring is created beside
# the live one and swapped in only after the helper accepts the open, so a refused
# open keeps the live ring. A docked test window whose viewport never starts is torn
# down and reports the viewport's error.
from pathlib import Path
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
plugin = root / "../plugin/Source"
sidecar = (plugin / "VideoSidecar.h").read_text()
commands = (plugin / "CommandServerVideoClips.cpp").read_text()
errors = []


def region(source, begin, end):
    start = source.find(begin)
    if start < 0:
        errors.append("missing: " + begin)
        return ""
    stop = source.find(end, start + len(begin))
    return source[start:stop] if stop >= 0 else source[start:]


open_body = region(sidecar, "bool viewportOpenShm (int width, int height, juce::String& errorOut)",
                   "/** Frame ring of the shm docked viewport.")
send_text = 'sendSync ("viewport_open_shm", params, errorOut, 15000);'
send = open_body.find(send_text)
refused = open_body.find("if (errorOut.isNotEmpty())")
swap = open_body.find("viewportFrames_.swap (pendingFrames);")

# (a) The grow path sends the open with the pending name and changes nothing before
# the helper answers. The live ring and its name are only replaced after the send.
if send < 0:
    errors.append("viewportOpenShm no longer sends viewport_open_shm")
if "const std::string frameName = grow ? pendingName : viewportFramesName_;" not in open_body:
    errors.append("viewportOpenShm does not choose the pending name for a grow")
if '"shmName", juce::String (frameName)' not in open_body:
    errors.append("viewport_open_shm does not send the chosen frame name")
if send >= 0:
    before_send = open_body[:send]
    for needle in ("viewportFrames_.close", "viewportFrames_.open", "viewportFrames_.swap", "viewportFramesName_ ="):
        if needle in before_send:
            errors.append("grow path touches the live ring before the helper accepts: " + needle)
    # Closing a creator unlinks its name, so the live ring must own the new name.
    if "viewportFrames_.open (pendingName)" in open_body:
        errors.append("the live ring re-opens the pending name instead of owning it")
    if swap < 0 or swap < send:
        errors.append("the pending ring is not swapped in after the accepted open")
    if refused < 0 or refused > swap:
        errors.append("the refusal check does not come before the ring swap")
if "videoshm::Region pendingFrames;" not in open_body or "pendingFrames.create (pendingName, 3, slotBytes)" not in open_body:
    errors.append("the grow path does not create the ring as a pending region")

# (b) Every failure path closes the pending ring and leaves viewportFrames_ alone.
refusal_branch = region(open_body, "if (errorOut.isNotEmpty())", "return false;")
if "pendingFrames.close();" not in refusal_branch:
    errors.append("a refused open does not close the pending ring")
if "viewportFrames_.close" in refusal_branch:
    errors.append("a refused open closes the live ring")

# (c) A docked test window whose viewport path is none is torn down and reports why.
handler = region(commands, "juce::String CommandServer::handleVideoViewportOpenDocked",
                 "juce::String CommandServer::handleVideoViewportDockedInfo")
dead = region(handler, 'if (started.path == "none")', "return videoErrorJson(")
if "g_dockedTestWindow.reset();" not in dead:
    errors.append("a dead docked test window is not torn down")
if 'videoErrorJson("docked test viewport failed to open: " + started.lastError)' not in handler:
    errors.append("a dead docked test viewport does not return its lastError")
if 'obj->setProperty("ok", true);' in handler and \
        handler.find('"docked test viewport failed to open') > handler.find('obj->setProperty("ok", true);'):
    errors.append("the dead-viewport error comes after the success reply")

if errors:
    print("missing viewport shm open refusal contracts: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("viewport shm open refusal source contract: PASS")
