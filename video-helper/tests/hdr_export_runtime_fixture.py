#!/usr/bin/env python3
"""Real helper HDR sequence/export regression. No compiler or NumPy dependency."""
import argparse
import ast
import json
import pathlib
import struct
import tempfile

from export_runtime_fixture import Helper, canonical_job, poll_export, require, ffprobe_verify


def float_array(path):
    data = path.read_bytes()
    require(data[:8] == b"\x93NUMPY\x01\x00", "HDR file must be NPY 1.0")
    size = struct.unpack_from("<H", data, 8)[0]
    header = ast.literal_eval(data[10:10 + size].decode("ascii"))
    require(header == {"descr": "<f4", "fortran_order": False, "shape": (32, 64, 4)},
            "HDR NPY type, shape or layout differs")
    return struct.unpack("<" + "f" * (32 * 64 * 4), data[10 + size:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="donutstudio-hdr-export-") as directory:
        root = pathlib.Path(directory)
        manifests = []
        for name in ("first", "replay"):
            helper = Helper(args.helper)
            try:
                output = root / (name + ".mp4")
                job = canonical_job(output, 64, 32, 3)
                job["hdrImageProfile"] = "linear-rec2020"
                job["post"] = {"exposure": 128.0}
                export_id = helper.send("export", job)
                _, _, response = poll_export(helper, export_id)
                require("error" not in response, f"HDR export failed: {response}")
                ffprobe_verify(output, 64, 32, 3)
                bundle = pathlib.Path(str(output) + ".passes")
                require((bundle / ".complete").read_text() == "complete\n", "HDR bundle not committed")
                manifest = json.loads((bundle / "manifest.json").read_text())
                frames = manifest["frames"]
                require([entry["frameIndex"] for entry in frames] == [0, 1, 2], "HDR sequence dropped or duplicated frames")
                for entry in frames:
                    require(entry["stage"] == "timeline-final-composite" and entry["outputTransfer"] == "linear"
                            and entry["outputPrimaries"] == "rec2020-d65", "HDR color declaration differs")
                    pixels = float_array(bundle / entry["files"][0]["path"])
                    require(max(pixels) > 1.0, "HDR export clipped every highlight")
                    require(all(pixels[index] == 1.0 for index in range(3, len(pixels), 4)), "HDR alpha differs")
                manifests.append({file.name: file.read_bytes() for file in bundle.iterdir()})
            finally:
                helper.close()
        require(manifests[0] == manifests[1], "HDR replay changed image or manifest bytes")

        helper = Helper(args.helper)
        try:
            output = root / "cancelled.mp4"
            job = canonical_job(output, 1920, 1080, 24)
            job["hdrImageProfile"] = "linear-srgb"
            export_id = helper.send("export", job)
            _, _, response = poll_export(helper, export_id, cancel_after_frame=1)
            require(response.get("error", {}).get("message") == "cancelled", "HDR cancellation did not reach export owner")
            require(not pathlib.Path(str(output) + ".passes").exists(), "HDR cancellation retained partial bundle")
        finally:
            helper.close()
    print("PASS: HDR final images, reference video, deterministic replay and cancellation")


if __name__ == "__main__":
    main()
