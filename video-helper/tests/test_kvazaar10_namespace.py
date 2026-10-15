#!/usr/bin/env python3
"""Archive namespace admission fixtures. Native archive acceptance is separate."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "cmake" / "namespace_kvazaar10.py"
SPEC = importlib.util.spec_from_file_location("kvz10_namespace", SCRIPT)
NS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NS)


def member(name, data):
    header = f"{name:<16}{0:<12}{0:<6}{0:<6}{644:<8}{len(data):<10}`\n".encode()
    return header + data + (b"\n" if len(data) % 2 else b"")


def archive(cpu=0x0100000C, magic=0xFEEDFACF):
    obj = struct.pack("<IIIIIIII", magic, cpu, 0, 1, 0, 0, 0, 0)
    return b"!<arch>\n" + member("__.SYMDEF/", b"index") + member("one.o/", obj)


class NamespaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.source, self.output = self.root / "raw.a", self.root / "private.a"
        self.source.write_bytes(archive())

    def transform(self, macho=True, fault=None):
        args = argparse.Namespace(nm="nm", objcopy="objcopy", ranlib="ranlib",
                                  input=str(self.source), output=str(self.output),
                                  format="macho-arm64" if macho else "gnu", llvm_version="20.1.8")
        names = {"kvz_api_get", "kvz_adapter_compiled_bit_depth", "helper_global", "_helper"}
        names = {"_" + name if macho else name for name in names}
        imports = {("_malloc" if macho else "malloc", "U")}
        renamed = {NS.renamed(name, macho) for name in names}

        def census(nm, path, defined=False):
            is_original = pathlib.Path(path) == self.source
            definitions = names if is_original else renamed
            entries = {(name, "T") for name in definitions}
            if not defined:
                entries |= imports
                if not is_original and fault == "reference":
                    entries.add((sorted(names)[0], "U"))
                if not is_original and fault == "import":
                    entries.remove(next(iter(imports)))
                if not is_original and fault == "new-import":
                    entries.add(("_new_import" if macho else "new_import", "U"))
            if not is_original and fault == "definition":
                entries.add(("_stray" if macho else "stray", "T"))
            return list(entries)

        def command(argv, **kwargs):
            if argv[0] == "objcopy":
                self.assertEqual(argv[1], "--redefine-syms")
                pathlib.Path(argv[-1]).write_bytes(self.source.read_bytes())
            return mock.Mock(returncode=0)

        with mock.patch.object(NS, "llvm_tools", return_value={}), \
             mock.patch.object(NS, "symbols", side_effect=census), \
             mock.patch.object(NS.subprocess, "run", side_effect=command):
            NS.transform(args)
        return renamed, imports

    def test_macho_full_definition_closure_and_abi_underscore(self):
        names, imports = self.transform()
        mapping = self.output.with_suffix(".symbols").read_text()
        self.assertIn("_kvz_api_get _donut_kvz10_kvz_api_get\n", mapping)
        self.assertIn("__helper _donut_kvz10__helper\n", mapping)
        self.assertNotIn("malloc", mapping)
        receipt = json.loads(self.output.with_suffix(".namespace.json").read_text())
        self.assertEqual(set(receipt["definitions"]), names)
        self.assertEqual({tuple(item) for item in receipt["runtime_imports"]}, imports)
        self.assertEqual(receipt["objects"], 1)

    def test_gnu_closure_preserves_existing_symbol_spelling(self):
        self.transform(macho=False)
        self.assertIn("_helper donut_kvz10__helper\n", self.output.with_suffix(".symbols").read_text())

    def test_post_census_rejects_original_reference_and_definition_drift(self):
        for fault in ("reference", "definition", "import", "new-import"):
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                self.transform(fault=fault)
            self.assertFalse(self.output.exists())
            self.assertFalse(self.output.with_suffix(".pending.a").exists())

    def test_archive_architecture_is_independent_of_symbol_output(self):
        self.assertEqual(NS.macho_arm64_archive(self.source), 1)
        for data in (archive(cpu=0x01000007), archive(magic=0xFEEDFACE), b"!<thin>\n",
                     b"!<arch>\n", archive()[:-1]):
            with self.subTest(data=data):
                self.source.write_bytes(data)
                with self.assertRaises(RuntimeError):
                    NS.macho_arm64_archive(self.source)

    def test_bsd_extended_member_names(self):
        name = b"long-object-name.o\0\0\0"
        obj = struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 1, 0, 0, 0, 0)
        self.source.write_bytes(b"!<arch>\n" + member(f"#1/{len(name)}", name + obj))
        self.assertEqual(NS.macho_arm64_archive(self.source), 1)

    def test_macho_rejects_symbols_without_abi_underscore(self):
        with self.assertRaises(RuntimeError):
            NS.renamed("kvz_api_get", True)

    def test_missing_api_and_preexisting_namespace_are_rejected(self):
        args = argparse.Namespace(nm="nm", objcopy="objcopy", ranlib="ranlib",
                                  input=str(self.source), output=str(self.output),
                                  format="macho-arm64", llvm_version="20.1.8")
        for names in ({"_kvz_api_get"}, {"_kvz_api_get", "_kvz_adapter_compiled_bit_depth",
                                         "_donut_kvz10_helper"}):
            with self.subTest(names=names), \
                 mock.patch.object(NS, "llvm_tools", return_value={}), \
                 mock.patch.object(NS, "symbols", return_value=[(name, "T") for name in names]), \
                 self.assertRaises(RuntimeError):
                NS.transform(args)

    def test_tools_require_recorded_supported_matching_release_and_absolute_paths(self):
        for version in (None, "", "20", "20.1.7", "19.1.8"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                NS.llvm_tools("nm", "objcopy", "ranlib", version)
        with self.assertRaises(RuntimeError):
            NS.llvm_tools("nm", "objcopy", "ranlib", "20.1.8")
        tool = self.root / "llvm-tool"
        tool.write_text("placeholder")
        tool.chmod(0o700)
        versions = ("LLVM version 20.1.8\n", "GNU nm 20.1.8\n",
                    "LLVM version 20.1.8git\n", "LLVM version 21.1.0\n")
        for version in versions:
            with mock.patch.object(NS.subprocess, "run", return_value=mock.Mock(stdout=version, stderr="")):
                if version == versions[0]:
                    tools = NS.llvm_tools(str(tool), str(tool), str(tool), "20.1.8")
                    for recorded in tools.values():
                        self.assertEqual(recorded["version"], "20.1.8")
                        self.assertEqual(recorded["sha256"], hashlib.sha256(b"placeholder").hexdigest())
                else:
                    with self.assertRaises(RuntimeError):
                        NS.llvm_tools(str(tool), str(tool), str(tool), "20.1.8")
        with mock.patch.object(NS.subprocess, "run", side_effect=[
                mock.Mock(stdout="LLVM version 20.1.8\n", stderr=""),
                mock.Mock(stdout="LLVM version 21.1.0\n", stderr="")]), \
             self.assertRaises(RuntimeError):
            NS.llvm_tools(str(tool), str(tool), str(tool), "20.1.8")


if __name__ == "__main__":
    unittest.main()
