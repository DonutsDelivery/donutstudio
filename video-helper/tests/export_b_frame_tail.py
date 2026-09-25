#!/usr/bin/env python3
"""Check that a delayed H.264 export displays every submitted frame."""

import argparse
import json
import math
import pathlib
import subprocess
import tempfile
import wave

from export_runtime_fixture import Helper, poll_export


def probe_frames(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_frames", "-show_entries",
         "stream=has_b_frames,nb_read_frames:frame=best_effort_timestamp_time",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=pathlib.Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="arbit-b-frame-tail-") as directory:
        root = pathlib.Path(directory)
        output = root / "export.mp4"
        audio = root / "silence.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(b"\0" * 48000 * 4)

        helper = Helper(str(args.helper))
        try:
            export_id = helper.send("export", {
                "outPath": str(output), "width": 160, "height": 90,
                "fps": 30.0, "durationSec": 1.0, "endSec": 1.0,
                "codec": "h264", "encoder": "software", "intraOnly": False,
                "interpolation": "none", "audioPath": str(audio),
                "segments": [], "clips": [],
            })
            status, _, reply = poll_export(helper, export_id)
            assert "error" not in reply, reply
            assert (status["frame"], status["totalFrames"]) == (30, 30), status
        finally:
            helper.close()

        probe = probe_frames(output)
        stream = probe["streams"][0]
        assert stream["has_b_frames"] > 0, stream
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
        assert int(stream["nb_read_frames"]) == len(timestamps) == status["frame"], stream
        assert all(math.isclose(timestamp, n / 30.0, abs_tol=0.00001)
                   for n, timestamp in enumerate(timestamps)), timestamps

        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:v:0",
             "-f", "framemd5", "-"],
            check=True, capture_output=True, text=True, timeout=30)
        display_frames = [line for line in decoded.stdout.splitlines()
                          if line and not line.startswith("#")]
        assert len(display_frames) == status["frame"], len(display_frames)
        print(f"PASS: {status['frame']} submitted, {len(display_frames)} decoded B-frame displays")


if __name__ == "__main__":
    main()
