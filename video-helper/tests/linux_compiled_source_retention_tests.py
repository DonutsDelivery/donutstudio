#!/usr/bin/env python3
"""Compiler receipt retention fixtures. No compiler, Docker or app is invoked."""
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/video-helper-dependencies'))
import compiled_source
SPEC = importlib.util.spec_from_file_location('linux_compiled_retention',
    ROOT / 'scripts/video-helper-dependencies/linux.py')
linux = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linux)


class CompilerRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        self.build = self.root / 'build'
        self.output = self.root / 'candidate'
        self.private = self.root / 'private'
        for path in (self.repo, self.build, self.output, self.private):
            path.mkdir()
        self.source = self.repo / 'video-helper/src/helper.cpp'
        self.header = self.repo / 'shared/used.h'
        for path in (self.source, self.header):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('retained authored input ' + path.name)
        self.commands = [{'directory': str(self.build), 'file': str(self.source),
                          'command': 'c++ -c ' + str(self.source) + ' -o helper.o'}]
        (self.build / 'compile_commands.json').write_text(json.dumps(self.commands))
        self.deps = 'helper.o: #deps 2, deps mtime 1 (VALID)\n    ' + str(self.source) + '\n    ' + str(self.header) + '\n'
        self.command_text = self.commands[0]['command'] + '\nc++ helper.o libvendor.a -o arbit-video-helper\n'

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self):
        with patch.object(linux.subprocess, 'check_output', side_effect=[self.deps, self.command_text]) as query:
            result = linux.capture_helper_compilation(self.build, self.output)
        self.assertEqual([call.args[0][3:] for call in query.call_args_list],
                         [['-t', 'deps'], ['-t', 'commands', 'arbit-video-helper']])
        return result

    def test_actual_dependency_header_and_raw_commands_are_retained(self):
        paths, link, deps = self.capture()
        self.assertEqual(paths, sorted([self.source, self.header]))
        self.assertEqual(link, 'c++ helper.o libvendor.a -o arbit-video-helper')
        raw = self.output / 'evidence/helper-compilation'
        self.assertEqual((raw / 'ninja-deps.txt').read_text(), deps)
        self.assertEqual(json.loads((raw / 'compile_commands.json').read_text()), self.commands)

    def test_missing_database_or_ambiguous_helper_link_is_refused(self):
        database = self.build / 'compile_commands.json'
        database.unlink()
        with self.assertRaisesRegex(ValueError, 'compiler command database'):
            linux.capture_helper_compilation(self.build, self.output)
        database.write_text(json.dumps(self.commands))
        self.command_text += 'c++ helper.o -o arbit-video-helper\n'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.capture()

    def test_fetched_source_notices_patch_content_and_original_download_are_kept(self):
        vendor = self.build / '_deps/ncnn-src'
        vendor.mkdir(parents=True)
        (vendor / 'gpu.cpp').write_text('actual patched source')
        (vendor / 'LICENSE').write_text('vendor license')
        download = self.build / '_deps/ncnn-subbuild/ncnn-20250503-full-source.zip'
        download.parent.mkdir(parents=True)
        download.write_bytes(b'original pinned download fixture')
        (self.output / 'evidence/helper-compilation').mkdir(parents=True)
        rows, installed, packages = linux.retain_compiler_sources(
            self.build, self.output, {'entries': {}}, [vendor / 'gpu.cpp'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['notices'], ['LICENSE'])
        self.assertEqual(installed, [])
        self.assertEqual(packages, set())
        self.assertEqual((self.output / rows[0]['retainedPath'] / 'gpu.cpp').read_text(), 'actual patched source')
        self.assertEqual((self.output / 'dependencies/helper-archives' / download.name).read_bytes(), download.read_bytes())
        source_archive = self.private / 'helper-source.tar'
        with tarfile.open(source_archive, 'w') as archive:
            archive.add(self.header, arcname='shared/used.h')
        linux.append_retained_helper_sources(source_archive, self.output)
        with tarfile.open(source_archive) as archive:
            self.assertIn('shared/used.h', archive.getnames())
            self.assertIn('dependencies/helper-source/ncnn-src/gpu.cpp', archive.getnames())
            self.assertIn('dependencies/helper-source/ncnn-src/LICENSE', archive.getnames())

    def fetched_archive(self, dependency, root, members):
        vendor = compiled_source.FETCHED_VENDORS[dependency]
        fetch = compiled_source.FETCHED_VENDORS['ncnn'] if vendor['fetch'] == 'ncnn' else vendor
        path = self.private / 'compiled-source/archives' / fetch['download']
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'w') as archive:
            for name, blob in members.items():
                archive.writestr(name, blob)
        for name, blob in members.items():
            prefix = vendor['archivePrefix']
            if prefix and not name.startswith(prefix + '/'):
                continue
            relative = name[len(prefix) + 1:] if prefix else name
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        return path

    def collect_fetched_origins(self, rows, pins):
        raw = self.private / 'compiled-source/raw'
        raw.mkdir(parents=True, exist_ok=True)
        (raw / 'link.map').write_text('fixture map')
        with patch.object(compiled_source, 'pinned_archives', return_value=pins), \
             patch.object(compiled_source, 'collect_compiled_source', side_effect=lambda **kwargs: kwargs['origins']):
            return linux.compiled_source_evidence(self.build, self.private, 'a' * 40, rows)

    def test_flat_ncnn_archive_binds_distinct_glslang_and_bison_origins(self):
        # The source directory name supplies no license authority.
        root = self.build / '_deps/unexpected-name-src'
        members = {'LICENSE.txt': b'NCNN notice', 'src/layer.h': b'NCNN source',
                   'glslang/LICENSE.txt': b'exact full glslang and Bison exception notice',
                   'glslang/glslang/Include/intermediate.h': b'glslang source'}
        archive = self.fetched_archive('ncnn', root, members)
        pins = {name: (linux.sha256(archive), vendor['license'], vendor['licenses'])
                for name, vendor in compiled_source.FETCHED_VENDORS.items() if vendor['fetch'] == 'ncnn'}
        origins = self.collect_fetched_origins([{'root': str(root)}], pins)
        fetched = {o['dependency']: o for o in origins if o['kind'] == 'archive'}
        self.assertEqual(set(fetched), {'ncnn', 'ncnn-glslang', 'ncnn-glslang-bison'})
        self.assertEqual(fetched['ncnn']['archivePrefix'], '')
        self.assertEqual(fetched['ncnn-glslang']['root'], root / 'glslang')
        self.assertEqual(fetched['ncnn-glslang-bison']['archivePrefix'], 'glslang')
        self.assertIn('Bison-exception-2.2', fetched['ncnn-glslang-bison']['license'])

    def test_vulkan_and_json_origin_notices_require_exact_archive_members(self):
        for dependency in ('vulkan-headers', 'nlohmann-json'):
            with self.subTest(dependency=dependency):
                vendor = compiled_source.FETCHED_VENDORS[dependency]
                root = self.build / ('source-for-' + dependency)
                members = {compiled_source.archive_member(vendor['archivePrefix'], notice):
                           ('actual notice ' + notice).encode() for notice in vendor['licenses']}
                archive = self.fetched_archive(dependency, root, members)
                pins = {dependency: (linux.sha256(archive), vendor['license'], vendor['licenses'])}
                origins = self.collect_fetched_origins([{'root': str(root)}], pins)
                self.assertEqual([o['dependency'] for o in origins if o['kind'] == 'archive'], [dependency])
                (root / vendor['licenses'][0]).write_bytes(b'wrong notice')
                origins = self.collect_fetched_origins([{'root': str(root)}], pins)
                self.assertFalse(any(o['kind'] == 'archive' for o in origins))
                archive.write_bytes(archive.read_bytes() + b'changed download')
                result = self.collect_fetched_origins([{'root': str(root)}], pins)
                self.assertEqual(result['status'], 'UNVERIFIED')
                self.assertIn('Changed original pinned vendor', result['reason'])

    def test_original_vulkan_and_json_downloads_are_retained(self):
        (self.output / 'evidence/helper-compilation').mkdir(parents=True)
        downloads = []
        for dependency in ('vulkan-headers', 'nlohmann-json'):
            filename = compiled_source.FETCHED_VENDORS[dependency]['download']
            path = self.build / '_deps' / dependency / filename
            path.parent.mkdir(parents=True)
            path.write_bytes(('exact original ' + dependency).encode())
            downloads.append(path)
        linux.retain_compiler_sources(self.build, self.output, {'entries': {}}, [])
        for path in downloads:
            self.assertEqual((self.output / 'dependencies/helper-archives' / path.name).read_bytes(), path.read_bytes())

    def test_conflicting_downloads_with_the_same_pinned_name_are_refused(self):
        (self.output / 'evidence/helper-compilation').mkdir(parents=True)
        filename = compiled_source.FETCHED_VENDORS['nlohmann-json']['download']
        for directory in ('first', 'second'):
            path = self.build / '_deps' / directory / filename
            path.parent.mkdir(parents=True)
            path.write_bytes(directory.encode())
        with self.assertRaisesRegex(ValueError, 'Ambiguous original fetched vendor download'):
            linux.retain_compiler_sources(self.build, self.output, {'entries': {}}, [])

    def test_installed_header_package_identity_and_exact_bytes_are_retained(self):
        header = self.root / 'installed/include/json.hpp'
        header.parent.mkdir(parents=True)
        header.write_text('system-resolved header')
        (self.output / 'evidence/helper-compilation').mkdir(parents=True)
        fields = ['json-source', '3.11', 'json-dev', '3.11-1']
        with patch.object(linux, 'ROOT', self.repo), patch.object(linux, 'system_source_package', return_value=fields):
            rows, installed, packages = linux.retain_compiler_sources(
                self.build, self.output, {'entries': {}}, [header])
        self.assertEqual(rows, [])
        self.assertEqual(packages, {tuple(fields)})
        self.assertEqual(installed[0]['sha256'], linux.sha256(header))
        self.assertEqual((self.output / installed[0]['retainedPath']).read_bytes(), header.read_bytes())
        self.assertEqual(installed[0]['sourcePackage'], fields[0])

    def test_private_object_and_archive_member_inventory_do_not_claim_link_proof(self):
        (self.build / 'helper.o').write_bytes(b'object fixture')
        (self.build / 'libvendor.a').write_bytes(b'archive fixture')
        with patch.object(linux.subprocess, 'check_output', return_value='member.o\n'):
            rows = linux.retain_relink_materials(self.build, self.private,
                                                'c++ helper.o libvendor.a -o arbit-video-helper')
        self.assertEqual(len(rows), 2)
        self.assertEqual((self.private / rows[1]['memberInventory']).read_text(), 'member.o\n')
        proof = linux.compiled_source_evidence(self.build, self.private, 'a' * 40, [])
        self.assertEqual(proof['status'], 'UNVERIFIED')
        self.assertIn('link map', proof['reason'])


if __name__ == '__main__':
    unittest.main()
