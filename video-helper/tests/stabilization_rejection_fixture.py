#!/usr/bin/env python3
"""Prepared HDR/alpha rejection checks and a narrow opaque SDR control."""
import argparse
import json
import pathlib
import subprocess
import tempfile

from export_runtime_fixture import Helper
from stabilization_rpc_fixture import expect_error, sha


def make_source(path, transfer=None, alpha=False):
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
               "color=c=red@0.25:size=160x120:rate=12,format=rgba" if alpha
               else "color=c=gray:size=160x120:rate=12",
               "-frames:v", "4", "-an"]
    if alpha:
        command += ["-c:v", "qtrle", "-pix_fmt", "argb"]
    else:
        command += ["-c:v", "ffv1", "-pix_fmt", "yuv420p10le" if transfer else "yuv420p"]
    if transfer:
        command += ["-color_trc", transfer, "-color_primaries", "bt2020",
                    "-colorspace", "bt2020nc", "-color_range", "tv"]
    subprocess.run(command + ["-y", str(path)], check=True, capture_output=True, timeout=30)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=pix_fmt,color_transfer", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(probe.stdout)["streams"][0]
    if transfer:
        assert stream["color_transfer"] == transfer, stream
    if alpha:
        assert stream["pix_fmt"] == "argb", stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="arbit-stabilization-rejection-") as directory:
        root = pathlib.Path(directory)
        sdr = root / "opaque-sdr.mkv"
        make_source(sdr)
        helper = Helper(str(args.helper.resolve()))
        try:
            version = helper.call("version")
            assert version["stabilization"] and not version["stabilizationHdr"]
            assert version["stabilizationAlpha"] is False
            track, output = root / "complete.trf", root / "complete.mp4"
            media = helper.call("open", {"path": str(sdr)})["mediaId"]
            params = {"mediaId": media, "trfPath": str(track), "inSec": 0.0,
                      "outSec": 0.3, "sourceSha256": sha(sdr), "requestToken": "sdr-detect"}
            detected = helper.call("stabilize_detect", params, timeout=30)
            rendered = helper.call("stabilize_render", dict(params, outPath=str(output),
                strength=0.1, requestToken="sdr-render"), timeout=30)
            assert detected["frames"] == rendered["frames"] == 4
            track_digest, output_digest = sha(track), sha(output)
            for name, transfer, alpha, error in (
                ("pq", "smpte2084", False, "HDR preservation is not implemented"),
                ("hlg", "arib-std-b67", False, "HDR preservation is not implemented"),
                ("alpha", None, True, "alpha preservation is not implemented"),
            ):
                source = root / (name + (".mov" if alpha else ".mkv"))
                make_source(source, transfer, alpha)
                info = helper.call("open", {"path": str(source)})
                if alpha:
                    assert info["hasAlpha"] is True
                reject = dict(params, mediaId=info["mediaId"], sourceSha256=sha(source),
                              requestToken=name+"-detect")
                expect_error(helper, "stabilize_detect", reject, error)
                expect_error(helper, "stabilize_render", dict(reject, outPath=str(output),
                    strength=0.1, requestToken=name+"-render"), error)
                assert sha(track) == track_digest and sha(output) == output_digest
                fresh_track, fresh_output = root / (name+".trf"), root / (name+".mp4")
                expect_error(helper, "stabilize_detect", dict(reject, trfPath=str(fresh_track)), error)
                expect_error(helper, "stabilize_render", dict(reject, outPath=str(fresh_output)), error)
                assert not fresh_track.exists() and not fresh_output.exists()
                assert not list(root.glob("*.stab-part-*"))
        finally:
            helper.close()
        print("PASS: opaque SDR stabilizes; PQ/HLG/alpha fail visibly without changing complete artifacts")


if __name__ == "__main__":
    main()
