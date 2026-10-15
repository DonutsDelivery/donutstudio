#!/usr/bin/env python3
"""Prepared owned-stabilization RPC timing, identity and cancellation checks.

This fixture uses the installed helper plus FFmpeg/ffprobe. Its B-frame input
uses FFmpeg's native MPEG-4 encoder, not x264 or vid.stab. Run the independent
stabilization_core image RMS oracle as a separate required quality check.
"""
import argparse
import hashlib
import json
import math
import pathlib
import signal
import struct
import subprocess
import tempfile
import time

from export_runtime_fixture import Helper


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_streams", "-show_entries",
        "stream=index,width,height,time_base,has_b_frames,sample_aspect_ratio:"
        "frame=best_effort_timestamp,best_effort_timestamp_time", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def read_track(path):
    data = path.read_bytes()
    assert data[:4] == b"STAB" and hashlib.sha256(data[:-32]).digest() == data[-32:]
    schema, estimator, count = struct.unpack_from("<III", data, 4)
    assert (schema, estimator) == (2, 1)
    assert len(data) == 92 + 88 * count + 32
    metadata = struct.unpack_from("<iIIiiii dd", data, 48)
    frames = []
    for index in range(count):
        ordinal, pts = struct.unpack_from("<Qq", data, 92 + 88 * index)
        gray_sha = data[92 + 88 * index + 56:92 + 88 * index + 88].hex()
        frames.append((ordinal, pts, gray_sha))
    return data[16:48].hex(), metadata, frames


def expect_error(helper, method, params, contains):
    response = helper.response(helper.send(method, params), timeout=180)
    assert "error" in response and contains in response["error"]["message"], response
    return response


def make_vfr_bframes(path):
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24",
        "-frames:v", "96", "-vf", "settb=1/1000,setpts=2000+N*37+floor(N/3)*7,setsar=4/3",
        "-fps_mode", "vfr", "-c:v", "mpeg4", "-bf", "2", "-q:v", "3",
        "-enc_time_base", "1:1000", "-an", "-y", str(path),
    ], check=True, capture_output=True, timeout=60)


def wait_active_frame(helper, request, minimum):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = helper.call("stabilize_progress")
        if request in helper.responses:
            raise AssertionError(f"pass finished before owned cancellation boundary: {helper.responses[request]}")
        if status["active"] and status["frame"] >= minimum:
            return status
        time.sleep(0.01)
    raise AssertionError("stabilization did not reach a bounded frame boundary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="arbit-owned-stabilization-") as directory:
        root = pathlib.Path(directory)
        source = root / "vfr-bframes.mp4"
        make_vfr_bframes(source)
        source_probe = probe(source)
        assert source_probe["streams"][0]["has_b_frames"] > 0
        source_pts = [int(frame["best_effort_timestamp"]) for frame in source_probe["frames"]]
        source_seconds = [float(frame["best_effort_timestamp_time"]) for frame in source_probe["frames"]]
        assert source_seconds[0] >= 2 and len(set(round(b-a, 6) for a, b in zip(source_seconds, source_seconds[1:]))) > 1
        track, output = root / "motion.trf", root / "motion.mp4"
        helper = Helper(str(args.helper.resolve()))
        try:
            version = helper.call("version")
            assert version["stabilization"] and version["vidstab"]
            assert version["stabilizationEngine"] == "donutstudio-native"
            assert version["stabilizationTrackSchema"] == 2 and not version["stabilizationHdr"]
            descriptor = helper.call("open", {"path": str(source)})
            media = descriptor["mediaId"]
            assert descriptor["streamIndex"] == source_probe["streams"][0]["index"]
            assert (descriptor["sarNum"], descriptor["sarDen"]) == (4, 3)
            numerator, denominator = map(int, source_probe["streams"][0]["time_base"].split("/"))
            assert (descriptor["timeBaseNum"], descriptor["timeBaseDen"]) == (numerator, denominator)
            incoming, outgoing = 2.231, 4.497
            params = {"mediaId": media, "trfPath": str(track), "inSec": incoming,
                      "outSec": outgoing, "sourceSha256": sha(source), "requestToken": "timing-detect"}
            detected = helper.call("stabilize_detect", params, timeout=180)
            assert detected["sourceSha256"] == sha(source) and detected["trackSha256"] == sha(track)
            assert all(detected[field] == descriptor[field] for field in
                       ("streamIndex", "width", "height", "sarNum", "sarDen", "timeBaseNum", "timeBaseDen"))
            digest, metadata, records = read_track(track)
            assert digest == sha(source)
            expected = [(ordinal, pts) for ordinal, pts in enumerate(
                pts for pts, seconds in zip(source_pts, source_seconds) if incoming <= seconds < outgoing)]
            assert [(ordinal, pts) for ordinal, pts, _ in records] == expected
            assert all(len(gray_sha) == 64 and gray_sha != "0" * 64 for _, _, gray_sha in records)
            assert (metadata[-2], metadata[-1]) == (incoming, outgoing)
            render = dict(params, outPath=str(output), strength=0.2, requestToken="timing-render")
            rendered = helper.call("stabilize_render", render, timeout=180)
            assert rendered["frames"] == detected["frames"] == len(records)
            assert rendered["outputSha256"] == sha(output) and rendered["trackSha256"] == sha(track)
            assert rendered["encoder"] == version["stabilizationEncoder"]
            output_probe = probe(output)
            assert (output_probe["streams"][0]["width"], output_probe["streams"][0]["height"]) == (320, 240)
            assert output_probe["streams"][0]["sample_aspect_ratio"] == "4:3"
            output_seconds = [float(frame["best_effort_timestamp_time"]) for frame in output_probe["frames"]]
            expected_seconds = [seconds-incoming for seconds in source_seconds if incoming <= seconds < outgoing]
            assert len(output_seconds) == len(expected_seconds)
            assert all(math.isclose(a, b, abs_tol=2/90000) for a, b in zip(output_seconds, expected_seconds)), (output_seconds, expected_seconds)
            previous_output = sha(output)
            expect_error(helper, "stabilize_render", dict(render, inSec=incoming+0.001), "does not match analysis")
            assert sha(output) == previous_output

            # Recompute the outer checksum after corrupting a frame key. This
            # reaches semantic sequence validation rather than only its checksum.
            corrupted = root / "corrupt.trf"
            data = bytearray(track.read_bytes())
            struct.pack_into("<Q", data, 92+88, 999)
            data[-32:] = hashlib.sha256(data[:-32]).digest()
            corrupted.write_bytes(data)
            expect_error(helper, "stabilize_render", dict(render, trfPath=str(corrupted)), "presentation order")
            data = bytearray(track.read_bytes()); struct.pack_into("<i", data, 60, 5)
            data[-32:] = hashlib.sha256(data[:-32]).digest(); corrupted.write_bytes(data)
            expect_error(helper, "stabilize_render", dict(render, trfPath=str(corrupted)), "does not match analysis")
            # A validly checksummed but false Gray8 digest must fail replay.
            data = bytearray(track.read_bytes()); data[92+56] ^= 1
            data[-32:] = hashlib.sha256(data[:-32]).digest(); corrupted.write_bytes(data)
            expect_error(helper, "stabilize_render", dict(render, trfPath=str(corrupted)), "Gray8 frame does not match")
            assert sha(output) == previous_output
            short_track = root / "short.trf"
            original = track.read_bytes()
            count = struct.unpack_from("<I", original, 12)[0]
            shortened = bytearray(original[:92+88*(count-1)])
            struct.pack_into("<I", shortened, 12, count-1)
            shortened.extend(hashlib.sha256(shortened).digest())
            short_track.write_bytes(shortened)
            expect_error(helper, "stabilize_render", dict(render, trfPath=str(short_track)), "does not match analysis")
            assert sha(output) == previous_output
            # Whole-byte source change is rejected even if decoding remains valid.
            source.write_bytes(source.read_bytes()+b"source-identity-change")
            expect_error(helper, "stabilize_render", render, "source content changed")
            assert sha(output) == previous_output

            mutated_track = root / "mutated-during-pass.trf"
            mutate_params = dict(params, trfPath=str(mutated_track), sourceSha256=sha(source), requestToken="owned-mutation")
            request = helper.send("stabilize_detect", mutate_params)
            wait_active_frame(helper, request, 1)
            with source.open("ab") as handle:
                handle.write(b"changed-during-analysis")
            response = helper.response(request, timeout=180)
            assert "source changed" in response.get("error", {}).get("message", ""), response
            assert not mutated_track.exists()

            cancelled_track = root / "cancelled.trf"
            cancel_params = dict(params, trfPath=str(cancelled_track), sourceSha256=sha(source), requestToken="owned-cancel")
            request = helper.send("stabilize_detect", cancel_params)
            status = wait_active_frame(helper, request, 1)
            wrong = helper.call("stabilize_cancel", {"requestToken": "other-request"})
            assert wrong["active"] is False
            right = helper.call("stabilize_cancel", {"requestToken": status["requestToken"]})
            assert right["active"] is True
            response = helper.response(request, timeout=30)
            assert response.get("error", {}).get("message") == "cancelled", response
            assert not cancelled_track.exists() and not list(root.glob("*.stab-part-*"))

            replay_track = root / "replay.trf"
            fresh = dict(cancel_params, trfPath=str(replay_track), requestToken="fresh-replay-track")
            helper.call("stabilize_detect", fresh, timeout=180)
            cancelled_output = root / "cancelled.mp4"
            request = helper.send("stabilize_render", dict(fresh, outPath=str(cancelled_output), strength=0.2, requestToken="owned-render-cancel"))
            status = wait_active_frame(helper, request, 1)
            assert helper.call("stabilize_cancel", {"requestToken": status["requestToken"]})["active"]
            response = helper.response(request, timeout=30)
            assert response.get("error", {}).get("message") == "cancelled", response
            assert not cancelled_output.exists() and not list(root.glob("*.stab-part-*"))

            # Kill a render after a frame entered its real temporary muxer.
            # Restart must not promote that orphan or replay the interrupted job.
            restart_output = root / "interrupted.mp4"
            restart_render = dict(fresh, outPath=str(restart_output), strength=0.2, requestToken="owned-restart")
            request = helper.send("stabilize_render", restart_render)
            wait_active_frame(helper, request, 1)
            helper.proc.send_signal(signal.SIGKILL)
            helper.proc.wait(timeout=10)
            assert not restart_output.exists()
        finally:
            helper.close()
        restarted = Helper(str(args.helper.resolve()))
        try:
            assert restarted.call("stabilize_progress")["active"] is False
            assert not restart_output.exists()
            media = restarted.call("open", {"path": str(source)})["mediaId"]
            retry = dict(restart_render, mediaId=media, requestToken="explicit-retry-render")
            analysis_retry = dict(fresh, mediaId=media, requestToken="explicit-retry-detect")
            restarted.call("stabilize_detect", analysis_retry, timeout=180)
            result = restarted.call("stabilize_render", retry, timeout=180)
            assert result["frames"] > 0 and result["outputSha256"] == sha(restart_output)
        finally:
            restarted.close()
        print("PASS: owned stabilization exact VFR/B-frame trim, replay/source guards, cancellation and restart")


if __name__ == "__main__":
    main()
