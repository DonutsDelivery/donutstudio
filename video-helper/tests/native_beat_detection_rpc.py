#!/usr/bin/env python3
"""Exercise decoded native beat analysis through the retained helper RPC."""

import argparse
import array
import math
from pathlib import Path
import random
import sys
import tempfile
import wave

from rpc_smoke_common import RpcHelper


def write_audio(path, rate, channels, duration, bpm=None, noise=False, phase=0.25):
    frames = round(rate * duration)
    pulse_starts = []
    mono = [0.0] * frames
    if bpm is not None:
        beat = phase
        while beat + 0.02 < duration:
            start = round(beat * rate)
            pulse_starts.append(start)
            for offset in range(round(0.02 * rate)):
                t = offset / rate
                mono[start + offset] = 0.8 * math.exp(-250 * t) * math.cos(2 * math.pi * 1000 * t)
            beat += 60.0 / bpm
    elif noise:
        rng = random.Random(8713)
        mono = [rng.uniform(-0.04, 0.04) for _ in range(frames)]
    pcm = array.array("h", (round(value * 32767) for value in mono for _ in range(channels)))
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())

    # Read the actual fixture back through an independent PCM reader. Its pulse
    # transitions establish the expected tempo and phase without the DSP engine.
    with wave.open(str(path), "rb") as source:
        assert source.getnframes() == frames
        assert source.getframerate() == rate
        assert source.getnchannels() == channels
        decoded = array.array("h", source.readframes(frames))
    if sys.byteorder != "little":
        decoded.byteswap()
    if bpm is not None:
        found = []
        active = False
        for index in range(frames):
            loud = abs(decoded[index * channels]) > 15000
            if loud and not active and (not found or index - found[-1] > rate * 0.05):
                found.append(index)
            active = loud
        assert found == pulse_starts, (path, found, pulse_starts)
        periods = [(right - left) / rate for left, right in zip(found, found[1:])]
        if periods:
            expected_bpm = 60.0 / (sum(periods) / len(periods))
            assert abs(expected_bpm - bpm) < 0.01
    return frames / rate, [start / rate for start in pulse_starts]


def valid_result(result, duration):
    assert set(result) == {"bpm", "confidence", "beats", "engine", "timingModel"}, result
    assert result["engine"] == "donutstudio-native"
    assert result["timingModel"] == "constant-tempo"
    assert isinstance(result["bpm"], (int, float)) and not isinstance(result["bpm"], bool)
    assert math.isfinite(result["bpm"]) and 40 <= result["bpm"] <= 220, result
    assert isinstance(result["confidence"], (int, float))
    assert math.isfinite(result["confidence"]) and 0 <= result["confidence"] <= 1, result
    assert isinstance(result["beats"], list) and len(result["beats"]) >= 2, result
    previous = -1.0
    for beat in result["beats"]:
        assert isinstance(beat, (int, float)) and math.isfinite(beat)
        assert 0 <= beat <= duration and beat > previous, result
        previous = beat


def expect_error(helper, method, params, message):
    try:
        helper.call(method, params)
    except RuntimeError as error:
        assert str(error) == f"{method}: {message}", str(error)
    else:
        raise AssertionError(f"{method} unexpectedly succeeded")


def analyze(helper, path):
    opened = helper.call("open", {"path": str(path)})
    media_id = opened["mediaId"]
    try:
        return helper.call("detect_beats", {"mediaId": media_id})
    finally:
        helper.call("close", {"mediaId": media_id})


def run(executable):
    with tempfile.TemporaryDirectory(prefix="native-beat-rpc-") as temporary:
        root = Path(temporary)
        fixtures = []
        for bpm, rate, channels, phase in [(120, 44100, 2, 0.25),
                                           (120, 48000, 1, 0.36),
                                           (109.5, 44100, 1, 0.31)]:
            path = root / f"click-{bpm}-{rate}.wav"
            duration, pulses = write_audio(path, rate, channels, 12.0, bpm=bpm, phase=phase)
            fixtures.append((path, duration, pulses, bpm))
        silence = root / "silence.wav"
        write_audio(silence, 44100, 1, 4.0)
        short = root / "short.wav"
        write_audio(short, 44100, 1, 0.5, bpm=120)
        noise = root / "noise.wav"
        noise_duration, _ = write_audio(noise, 44100, 1, 8.0, noise=True)
        invalid = root / "invalid.wav"
        invalid.write_bytes(b"not an audio container")

        helper = RpcHelper(executable)
        try:
            version = helper.call("version")
            assert version["beatDetection"] is True
            assert version["beatDetectionEngine"] == "donutstudio-native"
            assert version["beatDetectionTimingModel"] == "constant-tempo"
            expect_error(helper, "detect_beats", {}, "missing mediaId")
            expect_error(helper, "detect_beats", {"mediaId": 4294967295}, "unknown mediaId")
            for path, duration, pulses, bpm in fixtures:
                opened = helper.call("open", {"path": str(path)})
                assert opened["hasAudio"] is True
                assert abs(opened["durationSec"] - duration) < 0.001
                media_id = opened["mediaId"]
                first = helper.call("detect_beats", {"mediaId": media_id})
                valid_result(first, duration)
                assert abs(first["bpm"] - bpm) <= 2.0, (bpm, first)
                # Ignore leading/trailing grid extrapolation. Interior markers
                # must align with the independently read pulse positions.
                interior = [beat for beat in first["beats"] if pulses[0] <= beat <= pulses[-1]]
                assert len(interior) >= len(pulses) - 2, (bpm, first)
                assert all(min(abs(beat - pulse) for pulse in pulses) < 0.040 for beat in interior)
                assert helper.call("detect_beats", {"mediaId": media_id}) == first
                helper.call("close", {"mediaId": media_id})
                expect_error(helper, "detect_beats", {"mediaId": media_id}, "unknown mediaId")

            for path, message in [(silence, "native beat detector found no stable tempo"),
                                  (short, "audio too short for beat detection")]:
                opened = helper.call("open", {"path": str(path)})
                expect_error(helper, "detect_beats", {"mediaId": opened["mediaId"]}, message)
                helper.call("close", {"mediaId": opened["mediaId"]})

            # Stationary noise need not imply a stable musical tempo. The native
            # engine may return a grid or its existing explicit failure; both
            # paths must remain deterministic and preserve the RPC contract.
            opened = helper.call("open", {"path": str(noise)})
            noise_results = []
            for _ in range(2):
                try:
                    result = helper.call("detect_beats", {"mediaId": opened["mediaId"]})
                    valid_result(result, noise_duration)
                    noise_results.append(result)
                except RuntimeError as error:
                    assert str(error) in {
                        "detect_beats: native beat detector found no stable tempo",
                        "detect_beats: native beat detector found too few rhythmic events",
                    }, str(error)
                    noise_results.append(str(error))
            assert noise_results[0] == noise_results[1]
            helper.call("close", {"mediaId": opened["mediaId"]})

            video = Path(__file__).parent / "fixtures/decoded-frame-pts/frames.mp4"
            opened = helper.call("open", {"path": str(video)})
            assert opened["hasAudio"] is False
            expect_error(helper, "detect_beats", {"mediaId": opened["mediaId"]}, "media has no audio stream")
            helper.call("close", {"mediaId": opened["mediaId"]})

            # Keep the media handle, then remove its pathname. Analysis must
            # report its independent audio reopen failure without killing RPC.
            disappearing = root / "disappearing.wav"
            write_audio(disappearing, 44100, 1, 4.0, bpm=120)
            opened = helper.call("open", {"path": str(disappearing)})
            disappearing.unlink()
            expect_error(helper, "detect_beats", {"mediaId": opened["mediaId"]}, "reopen for audio failed")
            helper.call("close", {"mediaId": opened["mediaId"]})
            for path in [invalid, root / "missing.wav"]:
                try:
                    helper.call("open", {"path": str(path)})
                except RuntimeError as error:
                    assert str(error).startswith("open: open failed:"), str(error)
                else:
                    raise AssertionError(f"invalid audio opened: {path}")
            recovered = analyze(helper, fixtures[1][0])
            valid_result(recovered, fixtures[1][1])
            assert helper.call("ping") == "pong"
        finally:
            helper.close()
    print("PASS: native beat detection RPC, click timing, resampling and failure recovery")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("helper")
    run(parser.parse_args().helper)
