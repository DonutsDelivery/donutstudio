#!/usr/bin/env python3
"""Inspect native arm64 bindings before exercising both real Kvazaar libraries."""
import argparse
import pathlib
import re
import subprocess
import sys


def run(*argv):
    return subprocess.run(argv, check=True, text=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "archive", "avcodec", "kvazaar", "nm"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise RuntimeError("Native Mach-O link acceptance requires macOS")
    for path in (args.executable, args.archive, args.avcodec, args.kvazaar):
        if run("/usr/bin/lipo", "-archs", path).strip() != "arm64":
            raise RuntimeError(f"Candidate is not thin arm64: {path}")
    for path in (args.executable, args.avcodec, args.kvazaar):
        if "TWOLEVEL" not in run("/usr/bin/otool", "-hv", path):
            raise RuntimeError(f"Candidate lacks Mach-O two-level namespace: {path}")
    kvazaar_id = run("/usr/bin/otool", "-D", args.kvazaar).splitlines()[1].strip()
    dependencies = run("/usr/bin/otool", "-L", args.avcodec)
    if not any(line.strip().startswith(kvazaar_id + " (") for line in dependencies.splitlines()[1:]):
        raise RuntimeError("FFmpeg libavcodec does not load the recorded ordinary Kvazaar dylib")
    # Mach-O's two-level import ordinal is printed as the supplying library.
    library = pathlib.PurePosixPath(kvazaar_id).name.removesuffix(".dylib")
    # LLVM strips a single version component, such as .7 in libkvazaar.7.dylib.
    library = re.sub(r"\.[^.]$", "", library)
    for path in (args.avcodec, args.executable):
        binding = run(args.nm, "--format=darwin", "--undefined-only", path)
        if not any(re.search(r"\b_kvz_api_get\s+\(from " + re.escape(library) + r"\)", line)
                   for line in binding.splitlines()):
            raise RuntimeError(f"Ordinary getter import does not bind the recorded Kvazaar dylib: {path}")
    private = run(args.nm, "-P", "-g", "--defined-only", args.archive)
    private_names = {line.split()[0] for line in private.splitlines()
                     if line.strip() and not line.rstrip().endswith(":")}
    if not {"_donut_kvz10_kvz_api_get", "_donut_kvz10_kvz_adapter_compiled_bit_depth"} <= private_names:
        raise RuntimeError("Private archive lacks namespaced API or depth marker")
    if any(not name.startswith("_donut_kvz10_") for name in private_names):
        raise RuntimeError("Private archive retains a nonprivate external definition")
    ordinary = run(args.nm, "-P", "-g", "--defined-only", args.kvazaar)
    if not any(line.startswith("_kvz_api_get ") for line in ordinary.splitlines()):
        raise RuntimeError("Ordinary Kvazaar dylib lacks its original getter")
    print(run(args.executable, args.kvazaar, args.avcodec), end="")
    print(f"arm64 twolevel binding={kvazaar_id}")


if __name__ == "__main__":
    main()
