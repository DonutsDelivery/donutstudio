#!/usr/bin/env python3
from pathlib import Path
import sys


root = Path(sys.argv[1])
plugin = root / "../plugin/Source"
viewport = (root / "src/viewport.cpp").read_text()
close_reason = (plugin / "SharedGpuSurfaceCloseReason.h").read_text()

# One shared definition holds the cap, the bare EOF text and the UTF-8 cut.
assert "inline constexpr std::size_t kMaxCloseReasonBytes = 512;" in close_reason
assert 'inline constexpr const char* kProducerClosedText = "producer closed the surface channel";' in close_reason
assert "while (cut > 0 && cut < size && (text[cut] & 0xC0u) == 0x80u)" in close_reason
assert '#include "SharedGpuSurfaceCloseReason.h"' in viewport
assert "gpusurf::closeReasonPrefixBytes (" in viewport


def region(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


# Each Linux, Windows and IOSurface consumer keeps the BuffersGone reason and
# appends it to its EOF text after the parse. The Mac consumer keeps the same
# reason, so it is checked here too.
compose = "if (producerReason_.isNotEmpty() && lastError_ == gpusurf::kProducerClosedText)"
append = 'lastError_ = juce::String (gpusurf::kProducerClosedText) + ": " + producerReason_;'
for name in ("SharedGpuSurfaceDmabuf.h", "SharedGpuSurfaceVkFd.h",
             "SharedGpuSurfaceD3D.h", "SharedGpuSurfaceIOSurface.h"):
    source = (plugin / name).read_text()
    assert '#include "SharedGpuSurfaceCloseReason.h"' in source, name
    assert "juce::String producerReason_;" in source, name
    connect = region(source, "bool connect (const juce::String& endpoint) override", "bool pump() override")
    assert "producerReason_.clear();" in connect, name
    buffers_gone = region(source, "case gpusurf::MsgType::BuffersGone:", "break;")
    assert "gpusurf::closeReasonPrefixBytes (payload, payloadBytes)" in buffers_gone, name
    assert "producerReason_ = reasonBytes > 0" in buffers_gone, name
    pump = region(source, "bool pump() override", "\n    }\n")
    assert "lastError_ = gpusurf::kProducerClosedText;" in pump, name
    assert compose in pump and append in pump, name
    assert pump.rindex("handleMessage (header") < pump.index(compose), name
    assert pump.index(compose) < pump.index("return ! connectionLost_;"), name

# A failed release send records its errno text as the consumer error, so the viewport
# reports why the socket was lost. A short send has no errno and says so instead.
for name in ("SharedGpuSurfaceDmabuf.h", "SharedGpuSurfaceVkFd.h",
             "SharedGpuSurfaceD3D.h", "SharedGpuSurfaceIOSurface.h"):
    source = (plugin / name).read_text()
    release = region(source, "void sendRelease (uint32_t bufferIndex)", "void disconnect()")
    assert 'lastError_ = sent == SOCKET_ERROR' in release or 'lastError_ = sent < 0' in release, name
    assert 'consumer release failed: ' in release, name
    assert '"consumer release failed: short send"' in release, name
    assert release.index('lastError_ = ') < release.index("connectionLost_ = true;"), name
    if name == "SharedGpuSurfaceD3D.h":
        assert '"consumer release failed: socket error "' in release, name
        assert "WSAGetLastError()" in release, name
    else:
        assert "std::strerror (errno)" in release, name
    # The D3D AcquireSync failure text is set before its release, so a release failure replaces it.
    if name == "SharedGpuSurfaceD3D.h":
        acquire_fail = region(source, "if (hr != S_OK && hr != (HRESULT) WAIT_ABANDONED)", "buf.mutexHeld = true;")
        assert (acquire_fail.index('lastError_ = "AcquireSync(1) failed: 0x"')
                < acquire_fail.index("sendRelease (pending_.bufferIndex);")), name

mac = (plugin / "MacMetalVideoSurface.mm").read_text()
assert '#include "SharedGpuSurfaceCloseReason.h"' in mac
mac_buffers_gone = region(mac, "case gpusurf::MsgType::BuffersGone:", "case gpusurf::MsgType::FrameReady:")
assert "gpusurf::closeReasonPrefixBytes (payload, payloadBytes)" in mac_buffers_gone
assert "kMaxReasonBytes" not in mac

# When pump() reports producer loss, the Linux viewport copies the consumer's reason
# into its error, as the macOS path does, so video_editor_info carries it.
video_viewport = (plugin / "VideoViewportComponent.h").read_text()
assert video_viewport.count("if (! surface_->pump())") == 1
producer_loss = region(video_viewport, "if (! surface_->pump())", "const auto transfer = surface_->transferStats();")
assert "if (const auto reason = surface_->lastError(); reason.isNotEmpty())" in producer_loss
assert "setLastError (reason);" in producer_loss
assert producer_loss.index("setLastError (reason);") < producer_loss.index("connected_ = false;")

# The Linux and macOS viewports keep the consumer's loss text as producerLossError_ and
# clear it on the first fresh frame. The clear is skipped while connected_ is false,
# so a frame shown during the loss keeps the error.
clear_helper = region(video_viewport, "void clearProducerLossError()", "juce::String lastErrorCopy() const")
assert "if (producerLossError_.isEmpty() || ! connected_.load())" in clear_helper
assert "if (lastError_ == producerLossError_)" in clear_helper
assert "lastError_.clear();" in clear_helper
assert video_viewport.count("clearProducerLossError();") == 2
acquire_ok = region(video_viewport, "if (surface_->acquire (frame))", "else if (surface_->consecutiveImportFailures() >= 3)")
assert "clearProducerLossError();" in acquire_ok
mac_present = region(video_viewport, "if (metalSurface_->pumpAndPresent())", "else if (! metalSurface_->isConnected())")
assert "clearProducerLossError();" in mac_present
mac_loss = region(video_viewport, "else if (! metalSurface_->isConnected())", "if (resizePending_")
assert "producerLossError_ = reason;" in mac_loss
assert mac_loss.index("setLastError (reason);") < mac_loss.index("producerLossError_ = reason;")

# A device mismatch is an explicit fallback to shm, not a viewport error. The first
# connect and every helper reconnect check the producer device before connect(), so no
# cross-device import can run. The timer then opens shm once, with no retry of the
# zero-copy rung. The cause becomes transportFallbackReason, and lastError_ is cleared
# only when it still holds that same cause.
device_guard = region(video_viewport, "bool deviceMatchesOrDemote()", "\n    }\n")
assert "const auto deviceErr = surface_->deviceCompatibilityError (currentProducerDevice());" in device_guard
assert "requestDemote (deviceErr, true);" in device_guard
assert video_viewport.count("deviceMatchesOrDemote()") == 3
draw = region(video_viewport, "bool drawZeroCopyFrame (", "const auto transfer = surface_->transferStats();")
connect_calls = [i for i in range(len(draw)) if draw.startswith("surface_->connect (currentSocketPath())", i)]
assert len(connect_calls) == 2, connect_calls
for at in connect_calls:
    guard = draw.rfind("if (! deviceMatchesOrDemote())", 0, at)
    assert guard >= 0 and "return false;" in draw[guard:at], at
request = region(video_viewport, "void requestDemote (const juce::String& why, bool deviceMismatch = false)", "bool deviceMatchesOrDemote()")
assert "deviceFallbackReason_ = why;" in request and "deviceFallbackReason_.clear();" in request
assert "demoteRequested_ = true;" in request
demote = region(video_viewport, "const auto deviceFallback = takeDeviceFallbackReason();", "if (rung_.load() == Rung::Shm)")
assert demote.index("demoteRequested_ = false;") < demote.index("sidecar_.viewportCloseShared();")
assert demote.index("sidecar_.viewportCloseShared();") < demote.index("if (tryShm())")
assert demote.index("if (tryShm())") < demote.index("recordDeviceFallback (deviceFallback);") < demote.index("demoteToWindow();")
assert "sidecar_.onSharedViewportReopened = nullptr;" in demote
record = region(video_viewport, "void recordDeviceFallback (const juce::String& reason)", "void setRung (Rung r)")
assert "transportFallbackReason_ = reason;" in record
assert "if (lastError_ == reason)" in record and "lastError_.clear();" in record
assert "juce::String path, lastError, transportFallbackReason;" in video_viewport
stats_fn = region(video_viewport, "Stats stats() const", "bool copyLatestShmFrame")
assert "s.transportFallbackReason = transportFallbackReason_;" in stats_fn
start_block = region(video_viewport, "socketPath_ = socketPath;", "setRung (Rung::DmabufConnecting);")
assert "transportFallbackReason_.clear();" in start_block
stop_fn = region(video_viewport, "void stop()", "bool reDock()")
assert "transportFallbackReason_.clear();" in stop_fn and "deviceFallbackReason_.clear();" in stop_fn
command = (plugin / "CommandServerEditorCommands.cpp").read_text()
assert 'obj->setProperty("viewportTransportFallbackReason", stats.transportFallbackReason);' in command
docs = (root / "../docs/agents/mcp-command-reference.md").read_text()
assert "viewportTransportFallbackReason" in docs
