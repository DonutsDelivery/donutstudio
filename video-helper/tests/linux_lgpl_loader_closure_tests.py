#!/usr/bin/env python3
"""Tiny ELF loader fixtures; no plugin, Docker, display or application build."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = ROOT / 'scripts/video-helper-dependencies'
sys.path.insert(0, str(DEPENDENCIES))
SPEC = importlib.util.spec_from_file_location('linux_loader', DEPENDENCIES / 'linux.py')
linux = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linux)


class LoaderEnvironmentTests(unittest.TestCase):
    def test_final_resolution_clears_loader_environment(self):
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/untrusted', 'LD_PRELOAD': '/injected.so',
                                     'LD_AUDIT': '/audit.so'}), \
                patch.object(linux.subprocess, 'check_output', return_value='') as call:
            linux.resolve_needed(Path('/fixture/helper'))
            env = call.call_args.kwargs['env']
            self.assertTrue(all(key not in env for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT')))
            linux.resolve_needed(Path('/fixture/helper'), [Path('/admitted-prefix/lib')])
            env = call.call_args.kwargs['env']
            self.assertEqual(env['LD_LIBRARY_PATH'], '/admitted-prefix/lib')
            self.assertNotIn('LD_PRELOAD', env)
            self.assertNotIn('LD_AUDIT', env)

    def test_recorded_hash_does_not_admit_an_escaped_or_unknown_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'candidate'
            output.mkdir()
            helper = output / 'arbit-video-helper'
            helper.write_bytes(b'helper')
            library = output / 'libavcodec.so.60'
            library.write_bytes(b'library')
            outside = root / library.name
            outside.write_bytes(library.read_bytes())
            row = {'soname': library.name, 'bundledPath': library.name, 'sha256': linux.sha256(library),
                   'origin': 'source-prefix'}
            metadata = {'needed': [library.name], 'loaderPaths': ['$ORIGIN']}
            with patch.object(linux, 'elf_metadata', return_value=metadata), \
                    patch.object(linux, 'resolve_needed', return_value={library.name: outside}):
                with self.assertRaisesRegex(ValueError, 'escaped the candidate'):
                    linux.validate_installed_closure(output, helper.name, [row])
            row['origin'] = 'unknown'
            with patch.object(linux, 'elf_metadata', return_value=metadata), \
                    patch.object(linux, 'resolve_needed', return_value={library.name: library}):
                with self.assertRaisesRegex(ValueError, 'Unknown final ELF source origin'):
                    linux.validate_installed_closure(output, helper.name, [row])


class RealTransitiveLoaderTests(unittest.TestCase):
    def test_helper_runpath_does_not_resolve_grandchild_until_every_elf_is_sealed(self):
        missing = [name for name in ('cc', 'patchelf', 'readelf', 'ldd') if not shutil.which(name)]
        if missing:
            self.skipTest('Bounded ELF fixture unavailable; missing tools: ' + ', '.join(missing))
        with tempfile.TemporaryDirectory(prefix='donutstudio-loader-fixture-') as temporary:
            root = Path(temporary)
            leaf = root / 'libdonut-fixture-leaf.so.1'
            middle = root / 'libdonut-fixture-middle.so.1'
            helper = root / 'arbit-video-helper'
            (root / 'leaf.c').write_text('int fixture_leaf(void) { return 17; }\n')
            (root / 'middle.c').write_text(
                'int fixture_leaf(void); int fixture_middle(void) { return fixture_leaf() + 1; }\n')
            (root / 'helper.c').write_text(
                'int fixture_middle(void); int main(void) { return fixture_middle() != 18; }\n')
            # Three tiny translation units reproduce glibc's non-transitive
            # DT_RUNPATH search. No repository target or build cache is touched.
            subprocess.run(['cc', '-shared', '-fPIC', str(root / 'leaf.c'),
                            '-Wl,-soname,' + leaf.name, '-o', str(leaf)], check=True)
            subprocess.run(['cc', '-shared', '-fPIC', str(root / 'middle.c'), '-L' + str(root),
                            '-l:' + leaf.name, '-Wl,-soname,' + middle.name, '-o', str(middle)], check=True)
            subprocess.run(['cc', str(root / 'helper.c'), '-L' + str(root), '-l:' + middle.name,
                            '-Wl,-rpath,$ORIGIN', '-Wl,-rpath-link,' + str(root), '-o', str(helper)], check=True)
            self.assertEqual(linux.resolve_needed(helper, [root])[leaf.name], leaf)
            # Ambient LD_LIBRARY_PATH must not conceal the missing middle RUNPATH.
            with patch.dict(os.environ, {'LD_LIBRARY_PATH': str(root)}):
                with self.assertRaisesRegex(ValueError, 'Unresolved ELF dependency'):
                    linux.resolve_needed(helper)
                # This fixture tests real ELF rewriting, not package attribution.
                with patch.object(linux, 'system_source_package', return_value=['fixture'] * 4):
                    tool = linux.seal_loader_paths([helper, middle, leaf])
                self.assertEqual(tool['sha256'], linux.sha256(Path(tool['path'])))
                resolved = linux.resolve_needed(helper)
                self.assertEqual(resolved[middle.name], middle)
                self.assertEqual(resolved[leaf.name], leaf)
                self.assertTrue(all(linux.elf_metadata(p)['loaderPaths'] == ['$ORIGIN']
                                    for p in (helper, middle, leaf)))
                subprocess.run(['patchelf', '--remove-rpath', str(middle)], check=True)
                with self.assertRaisesRegex(ValueError, 'Unresolved ELF dependency'):
                    linux.resolve_needed(helper)


if __name__ == '__main__':
    unittest.main()
