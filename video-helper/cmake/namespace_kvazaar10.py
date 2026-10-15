#!/usr/bin/env python3
"""Rename defined archive symbols, preserving external runtime imports."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import struct
import subprocess

PREFIX = "donut_kvz10_"
MIN_LLVM_VERSION = (20, 1, 8)


def llvm_tools(nm, objcopy, ranlib, version):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version or ""):
        raise RuntimeError("Mach-O requires a recorded LLVM release version")
    if tuple(map(int, version.split("."))) < MIN_LLVM_VERSION:
        raise RuntimeError("Mach-O requires LLVM 20.1.8 or newer")
    recorded = {}
    for name, path in (("nm", nm), ("objcopy", objcopy), ("ranlib", ranlib)):
        tool = pathlib.Path(path)
        if not tool.is_absolute() or not tool.is_file() or not os.access(tool, os.X_OK):
            raise RuntimeError(f"Mach-O requires an explicit absolute executable LLVM {name}")
        result = subprocess.run([str(tool), "--version"], check=True, text=True,
                                capture_output=True)
        match = re.search(r"LLVM version\s+([0-9]+\.[0-9]+\.[0-9]+)(?![\w.-])",
                          result.stdout + result.stderr)
        if not match or match[1] != version:
            raise RuntimeError(f"LLVM {name} does not match recorded release {version}")
        resolved = tool.resolve()
        recorded[name] = {"path": str(tool), "resolved_path": str(resolved),
                          "version": match[1],
                          "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}
    return recorded


def macho_arm64_archive(archive):
    """Check every regular archive member, independently of nm's symbol census."""
    data = pathlib.Path(archive).read_bytes()
    if not data.startswith(b"!<arch>\n"):
        raise RuntimeError("Expected a regular Mach-O archive")
    offset, objects = 8, 0
    while offset < len(data):
        header = data[offset:offset + 60]
        if len(header) != 60 or header[58:] != b"`\n":
            raise RuntimeError("Malformed Mach-O archive member")
        try:
            size = int(header[48:58])
            name = header[:16].decode("ascii").strip()
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("Malformed Mach-O archive member") from error
        start, end = offset + 60, offset + 60 + size
        if size < 0 or end > len(data):
            raise RuntimeError("Truncated Mach-O archive member")
        member = data[start:end]
        if name.startswith("#1/"):
            length = int(name[3:])
            if length < 1 or length > len(member):
                raise RuntimeError("Malformed BSD archive member name")
            name = member[:length].rstrip(b"\0").decode("ascii")
            member = member[length:]
        if name.rstrip("/") not in ("__.SYMDEF", "__.SYMDEF SORTED", "__.SYMDEF_64",
                                    "__.SYMDEF_64 SORTED", "", "/"):
            if len(member) < 32 or struct.unpack_from("<IIII", member) != (
                    0xFEEDFACF, 0x0100000C, 0, 1):
                raise RuntimeError(f"Archive member {name} is not a thin arm64 Mach-O object")
            objects += 1
        offset = end + size % 2
    if offset != len(data) or not objects:
        raise RuntimeError("Malformed or empty Mach-O archive")
    return objects


def renamed(name, macho):
    if macho:
        if not name.startswith("_"):
            raise RuntimeError(f"Mach-O external symbol lacks its ABI underscore: {name}")
        return "_" + PREFIX + name[1:]
    return PREFIX + name


def symbols(nm, archive, defined=False):
    command = [nm, "-P", "-g"]
    if defined:
        command.append("--defined-only")
    result = subprocess.run(command + [str(archive)], check=True, text=True, capture_output=True)
    entries = []
    for line in result.stdout.splitlines():
        if not line.strip() or line.rstrip().endswith(":"):
            continue
        fields = line.split()
        if len(fields) < 2 or len(fields[1]) != 1:
            raise RuntimeError(f"Unsupported nm archive output: {line}")
        entries.append((fields[0], fields[1]))
    return entries


def transform(args):
    macho = args.format == "macho-arm64"
    tools = llvm_tools(args.nm, args.objcopy, args.ranlib, args.llvm_version) if macho else {}
    object_count = macho_arm64_archive(args.input) if macho else None
    initial = symbols(args.nm, args.input)
    original = {name for name, kind in symbols(args.nm, args.input, True)
                if kind.upper() != "U"}
    required = {"kvz_api_get", "kvz_adapter_compiled_bit_depth"}
    logical = {name[1:] if macho and name.startswith("_") else name for name in original}
    if not required <= logical or any(name.startswith(PREFIX) for name in logical):
        raise RuntimeError("Input is not the unmodified private Kvazaar archive")
    renames = {name: renamed(name, macho) for name in original}
    imports = {(name, kind) for name, kind in initial if name not in original}
    if any(name in renames.values() for name, _ in imports):
        raise RuntimeError("Runtime import collides with the private namespace")
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = output.with_suffix(".symbols")
    mapping.write_text("".join(f"{name} {renames[name]}\n" for name in sorted(original)))
    temporary = output.with_suffix(".pending.a")
    try:
        subprocess.run([args.objcopy, "--redefine-syms", str(mapping), args.input,
                        str(temporary)], check=True)
        subprocess.run([args.ranlib, str(temporary)], check=True)
        if macho and macho_arm64_archive(temporary) != object_count:
            raise RuntimeError("Archive transformation changed the arm64 object count")
        final = symbols(args.nm, temporary)
        definitions = {name for name, kind in symbols(args.nm, temporary, True)
                       if kind.upper() != "U"}
        expected = set(renames.values())
        final_imports = {(name, kind) for name, kind in final if name not in expected}
        if definitions != expected or any(name in original for name, _ in final):
            raise RuntimeError("Private archive retains original definitions or internal references")
        if final_imports != imports:
            raise RuntimeError("Private archive changed external runtime imports")
        temporary.replace(output)
        receipt = {"format": args.format, "tools": tools, "objects": object_count,
                   "definitions": sorted(expected), "runtime_imports": sorted(imports),
                   "input_sha256": hashlib.sha256(pathlib.Path(args.input).read_bytes()).hexdigest(),
                   "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        output.with_suffix(".namespace.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nm", "objcopy", "ranlib"):
        parser.add_argument("--" + name, required=True)
    for name in ("input", "output"):
        parser.add_argument("--" + name)
    parser.add_argument("--format", choices=("gnu", "macho-arm64"), default="gnu")
    parser.add_argument("--llvm-version")
    parser.add_argument("--check-tools", action="store_true")
    args = parser.parse_args()
    if args.check_tools:
        if args.format != "macho-arm64":
            parser.error("--check-tools requires --format=macho-arm64")
        print(json.dumps(llvm_tools(args.nm, args.objcopy, args.ranlib, args.llvm_version)))
    else:
        if not args.input or not args.output:
            parser.error("--input and --output are required for archive transformation")
        transform(args)


if __name__ == "__main__":
    main()
