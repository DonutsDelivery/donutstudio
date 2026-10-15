#!/usr/bin/env python3
"""Exercise the real H.264 export, proxy and stabilization encoder owners.

Independent FFmpeg decoding checks every display frame, PTS, keyframe and
image error against a lossless production export or exact source/scale oracle.
The default acceptance requires OpenH264. An explicit libx264 run can check
transitional compatibility but cannot establish dependency migration.
"""

import argparse
import json
import math
import pathlib
import subprocess
import tempfile
import time

from export_runtime_fixture import Helper, poll_export, require


WIDTH, HEIGHT, FRAMES = 160, 96, 72


def ffmpeg(arguments, data=None):
    return subprocess.run(["ffmpeg", "-v", "error", "-y", *arguments], input=data,
                          check=True, capture_output=True, timeout=90).stdout


def probe(path):
    process = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-count_packets",
        "-show_frames", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    return json.loads(process.stdout)


def source(root, rate, frames=FRAMES, static=False):
    # Exact limited-range YUV420 planes. The moving luma patch, gradient and
    # fine edges reveal missing/repeated frames and excessive quantization.
    raw = bytearray()
    for frame in range(frames):
        motion = 0 if static else frame
        for y in range(HEIGHT):
            for x in range(WIDTH):
                value = 40 + (x * 96 // WIDTH) + (y * 48 // HEIGHT)
                if ((x - 2 * motion) % WIDTH) < 24 and 24 <= y < 72:
                    value = 200 if ((x // 4 + y // 4) & 1) else 64
                raw.append(value)
        raw.extend(bytes([104]) * (WIDTH * HEIGHT // 4))
        raw.extend(bytes([152]) * (WIDTH * HEIGHT // 4))
    path = root / f"source-{str(rate).replace('/', '_')}-{'static' if static else 'moving'}.mkv"
    ffmpeg(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(rate), "-i", "pipe:0", "-c:v", "ffv1",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", str(path)], bytes(raw))
    return path


def decode(path, pixel_format="yuv420p", video_filter=None):
    arguments = ["-i", str(path), "-map", "0:v:0"]
    if video_filter:
        arguments.extend(["-vf", video_filter])
    return ffmpeg([*arguments, "-fps_mode", "passthrough", "-pix_fmt", pixel_format,
                   "-f", "rawvideo", "pipe:1"])


def quality(actual, expected, frame_bytes, frames, row, minimum_psnr=35.0):
    require(len(actual) == len(expected) == frame_bytes * frames,
            f"{row}: independent decode byte count differs: {len(actual)}, {len(expected)}")
    scores = []
    for frame in range(frames):
        begin, end = frame * frame_bytes, (frame + 1) * frame_bytes
        mse = sum((a - b) ** 2 for a, b in zip(actual[begin:end], expected[begin:end])) / frame_bytes
        scores.append(100.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse))
    require(min(scores) >= minimum_psnr,
            f"{row}: worst decoded frame PSNR {min(scores):.2f} below {minimum_psnr}")
    return round(min(scores), 3)


def inspect(path, encoder, timestamps, row, intra_only=False):
    evidence = probe(path)
    stream = evidence["streams"][0]
    frames = evidence["frames"]
    tags = evidence["format"].get("tags", {})
    require(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p",
            f"{row}: wrong codec/format {stream}")
    require(tags.get("encoder") == encoder, f"{row}: actual encoder tag mismatch: {tags}")
    require(f"video-encoder={encoder}" in tags.get("comment", ""), f"{row}: policy identity missing")
    require(len(frames) == int(stream["nb_read_frames"]) == int(stream["nb_read_packets"]) == len(timestamps),
            f"{row}: missing coded or decoded frames")
    for frame, expected in zip(frames, timestamps):
        require(math.isclose(float(frame["best_effort_timestamp_time"]), expected, abs_tol=0.00002),
                f"{row}: display PTS does not match source/grid")
    require(frames[0]["key_frame"] == 1, f"{row}: first frame is not independently decodable")
    if intra_only:
        require(stream["has_b_frames"] == 0 and all(frame["key_frame"] == 1 for frame in frames),
                f"{row}: intra-only seeks would need another frame")
    return evidence


def export_job(path, input_path, fps, frames, intra_only, codec="h264"):
    duration = (frames + 0.25) / fps
    return {"outPath": str(path), "width": WIDTH, "height": HEIGHT,
            "fps": fps, "durationSec": duration, "endSec": duration,
            "codec": codec, "encoder": "software", "interpolation": "none",
            "intraOnly": intra_only,
            "segments": [{"sourcePath": str(input_path), "clipId": 1, "trackLayer": 0,
                          "inSec": 0.0, "outSec": duration, "rate": 1.0,
                          "displayStartSec": 0.0}],
            "clips": [{"clipId": 1, "opacity": 1.0, "visible": True, "scale": 1.0}]}


def export(helper, job):
    request_id = helper.send("export", job)
    status, _, response = poll_export(helper, request_id)
    require("error" not in response, f"export failed: {response}")
    require(status["frame"] == status["totalFrames"] == FRAMES, f"incomplete export: {status}")
    return response["result"]


def await_proxy(helper, request_id):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = helper.call("proxy_progress")
        if status["done"]:
            return status, helper.response(request_id)
        time.sleep(0.01)
    raise RuntimeError("proxy did not complete within 90 seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=pathlib.Path)
    parser.add_argument("--expected-encoder", choices=["libopenh264", "libx264"], default="libopenh264")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="arbit-software-h264-") as directory:
        root = pathlib.Path(directory)
        helper = Helper(str(args.helper.resolve()))
        try:
            for rate in ("24", "60", "30000/1001"):
                fps = float(rate) if "/" not in rate else 30000 / 1001
                input_path = source(root, rate)
                timestamps = [n / fps for n in range(FRAMES)]
                reference = root / "reference.mkv"
                export(helper, export_job(reference, input_path, fps, FRAMES, True, "ffv1"))
                expected = decode(reference, "rgb24")
                # The lossless compositor reference must still show the moving
                # authored scene; a blank or frozen reference cannot pass.
                frame_bytes = WIDTH * HEIGHT * 3
                require(len(set(expected[:frame_bytes])) > 32, "reference lost authored gradient")
                require(expected[:frame_bytes] != expected[-frame_bytes:], "reference froze source motion")
                for intra_only in (False, True):
                    output = root / f"export-{rate.replace('/', '_')}-{intra_only}.mp4"
                    result = export(helper, export_job(output, input_path, fps, FRAMES, intra_only))
                    require(result["encoder"] == args.expected_encoder, f"RPC actual encoder mismatch: {result}")
                    row = f"export-{rate}-intra-{intra_only}"
                    evidence = inspect(output, args.expected_encoder, timestamps, row, intra_only)
                    # The two-second GOP stays bounded even when scene detection
                    # adds earlier keyframes; include the trailing GOP interval.
                    keys = [n for n, f in enumerate(evidence["frames"]) if f["key_frame"]]
                    require(max(b - a for a, b in zip(keys, keys[1:] + [FRAMES])) <= round(fps * 2),
                            f"{row}: keyframe interval exceeded configured GOP")
                    score = quality(decode(output, "rgb24"), expected, frame_bytes, FRAMES, row)
                    print(json.dumps({"row": row, "encoder": result["encoder"],
                                      "decodedFrames": FRAMES, "worstFramePsnr": score}))

            input_path = source(root, "24")
            input_pts = [float(f["best_effort_timestamp_time"]) for f in probe(input_path)["frames"]]
            for scale in (1.0, 0.5):
                output = root / f"proxy-{scale}.mp4"
                request = helper.send("proxy_generate", {"sourcePath": str(input_path),
                                      "outPath": str(output), "scale": scale})
                status, response = await_proxy(helper, request)
                require("error" not in response and status["frame"] == FRAMES, f"proxy failed: {response}")
                row = f"proxy-{scale}"
                inspect(output, args.expected_encoder, input_pts, row, True)
                width, height = int(WIDTH * scale), int(HEIGHT * scale)
                expected = decode(input_path, video_filter=f"scale={width}:{height}:flags=bilinear")
                score = quality(decode(output), expected, width * height * 3 // 2, FRAMES, row)
                # Independently seek to the final keyframe, then compare its
                # decoded pixels to the sequential decode's final frame.
                seek = ffmpeg(["-ss", str(input_pts[-1]), "-i", str(output), "-frames:v", "1",
                               "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"])
                require(seek == decode(output)[-width * height * 3 // 2:], f"{row}: final independent seek differs")
                print(json.dumps({"row": row, "encoder": args.expected_encoder,
                                  "decodedFrames": FRAMES, "worstFramePsnr": score}))

            static_path = source(root, "24", static=True)
            media = helper.call("open", {"path": str(static_path)})
            try:
                transform, output = root / "motion.trf", root / "stabilized.mp4"
                params = {"mediaId": media["mediaId"], "trfPath": str(transform),
                          "inSec": 0.25, "outSec": 1.25}
                detected = helper.call("stabilize_detect", params, timeout=90)
                rendered = helper.call("stabilize_render", {**params, "outPath": str(output),
                                                             "strength": 0.5}, timeout=90)
                require(detected["frames"] == rendered["frames"] == 24, "stabilization range/frame count differs")
                source_pts = [float(f["best_effort_timestamp_time"]) for f in probe(static_path)["frames"]][6:30]
                inspect(output, args.expected_encoder, [t - 0.25 for t in source_pts], "stabilize")
                # Static camera has zero global motion. Check the centre to
                # exclude any stabilization border interpolation policy.
                crop = "crop=128:64:16:16"
                expected = decode(static_path, video_filter="trim=start_frame=6:end_frame=30," + crop)
                score = quality(decode(output, video_filter=crop), expected, 128 * 64 * 3 // 2, 24, "stabilize")
                print(json.dumps({"row": "stabilize", "encoder": args.expected_encoder,
                                  "decodedFrames": 24, "worstFramePsnr": score}))
            finally:
                helper.call("close", {"mediaId": media["mediaId"]})

            long_source = root / "long-source.mkv"
            ffmpeg(["-stream_loop", "49", "-i", str(input_path), "-map", "0:v:0",
                    "-c:v", "copy", str(long_source)])
            cancelled = root / "cancelled.mp4"
            request = helper.send("proxy_generate", {"sourcePath": str(long_source),
                                  "outPath": str(cancelled), "scale": 1.0})
            helper.call("proxy_cancel")
            status, response = await_proxy(helper, request)
            require("error" in response and status["cancelled"], f"proxy cancellation failed: {response}")
            require(not cancelled.exists() and not pathlib.Path(str(cancelled) + ".part").exists(),
                    "cancelled proxy published a partial file")
            print(json.dumps({"row": "proxy-cancel", "cancelled": True, "partialFile": False}))
        finally:
            helper.close()


if __name__ == "__main__":
    main()
