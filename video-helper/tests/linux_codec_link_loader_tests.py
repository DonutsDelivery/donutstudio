#!/usr/bin/env python3
"""Codec linker fixtures. Parent validation runs bounded Make/ELF probes only."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/video-helper-dependencies'))
SPEC = importlib.util.spec_from_file_location('linux_codec_recipe',
    ROOT / 'scripts/video-helper-dependencies/build-linux.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
POLICY = json.loads((ROOT / 'scripts/ffmpeg-lgpl-candidate/policy-6.1.6.json').read_text())


class CodecLinkRecipeTests(unittest.TestCase):
    def test_literal_origin_escaping_matches_direct_and_eval_make_recipes(self):
        self.assertEqual(builder.KVAZAAR_LDFLAGS, "-Wl,-rpath,'$$ORIGIN' -Wl,-z,relro,-z,now")
        self.assertEqual(builder.OPENH264_LDFLAGS, '-lpthread ' + builder.KVAZAAR_LDFLAGS)
        args = POLICY['profiles']['linux-x64']['args']
        self.assertIn("--extra-ldsoflags=-Wl,-rpath,'$$$$ORIGIN' -Wl,-z,relro,-z,now", args)
        self.assertFalse(any('ORIGIN' in flag for flag in POLICY['configureFlags']))
        self.assertFalse(any('norelro' in flag for flag in args))

    def test_link_flags_are_recipe_inputs_and_audit_precedes_success_handoff(self):
        source = (ROOT / 'scripts/video-helper-dependencies/build-linux.py').read_text()
        self.assertIn("'LDFLAGS=' + OPENH264_LDFLAGS", source)
        self.assertIn("'LDFLAGS=' + KVAZAAR_LDFLAGS", source)
        self.assertIn("pathlib.Path(__file__)", source)
        self.assertLess(source.index('loader_protections = audit_loader_protections(prefix)'),
                        source.index('source_builds = SOURCE_BUILDS.write()'))
        self.assertLess(source.index("evidence / 'loader-protections.json'"),
                        source.index('linux.write_selection(selection_path, stage)'))

    def audit_fixture(self, dynamic, headers):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        prefix = Path(temporary.name)
        (prefix / 'lib').mkdir()
        names = ['lib' + name + '.so.1' for name in sorted(builder.audit.candidate.CORE)]
        names += ['libopenh264.so.1', 'libkvazaar.so.1']
        for name in names:
            (prefix / 'lib' / name).write_bytes(b'fixture bytes only')
        with patch.object(builder.subprocess, 'check_output',
                          side_effect=lambda argv, **kw: dynamic if '--dynamic' in argv else headers):
            return builder.audit_loader_protections(prefix)

    def test_loader_audit_requires_real_metadata_for_all_selected_libraries(self):
        observed = self.audit_fixture(' (RUNPATH) Library runpath: [$ORIGIN]\n (FLAGS) BIND_NOW\n',
                                      ' GNU_RELRO 0x0 0x0 0x0 0x100 0x100 R 0x1\n')
        self.assertEqual(len(observed), 9)

    def test_loader_audit_refuses_expanded_empty_prefix_or_duplicate_paths(self):
        for path in ('', '/tmp/prefix/lib', 'ORIGIN', '$ORIGIN:/tmp/prefix'):
            with self.assertRaisesRegex(ValueError, 'exact link-time ORIGIN'):
                self.audit_fixture(' (RUNPATH) Library runpath: [' + path + ']\n (FLAGS) BIND_NOW\n',
                                   ' GNU_RELRO 0x0\n')
        with self.assertRaisesRegex(ValueError, 'exact link-time ORIGIN'):
            self.audit_fixture(' (RUNPATH) Library runpath: [$ORIGIN]\n (RPATH) Library rpath: [$ORIGIN]\n',
                               ' GNU_RELRO 0x0\n')

    def test_loader_audit_refuses_missing_relro_or_lazy_binding(self):
        for dynamic, headers in [(' (RUNPATH) Library runpath: [$ORIGIN]\n (FLAGS) BIND_NOW\n', ''),
                                 (' (RUNPATH) Library runpath: [$ORIGIN]\n', ' GNU_RELRO 0x0\n')]:
            with self.assertRaisesRegex(ValueError, 'full RELRO'):
                self.audit_fixture(dynamic, headers)


class RealLinkOriginTests(unittest.TestCase):
    def setUp(self):
        missing = [name for name in ('cc', 'make', 'readelf', 'patchelf') if not shutil.which(name)]
        if missing:
            self.skipTest('Missing bounded ELF fixture tools: ' + ', '.join(missing))
        self.temporary = tempfile.TemporaryDirectory(prefix='donutstudio-codec-link-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'fixture.c').write_text('int fixture(void) { return 17; }\n')

    def run_tool(self, args, cwd=None, env=None):
        return subprocess.check_output(args, cwd=cwd or self.root, env=env, text=True,
                                       stderr=subprocess.STDOUT)

    def assert_final_loader_and_protections(self, binary):
        def metadata():
            dynamic = self.run_tool(['readelf', '--dynamic', '--wide', str(binary)])
            headers = self.run_tool(['readelf', '--program-headers', '--wide', str(binary)])
            paths = re.findall(r'\((?:RPATH|RUNPATH)\).*?\[(.*?)\]', dynamic)
            self.assertEqual(paths, ['$ORIGIN'])
            self.assertRegex(dynamic, r'\(BIND_NOW\)|\(FLAGS(?:_1)?\).*\b(?:BIND_NOW|NOW)\b')
            self.assertRegex(headers, r'GNU_RELRO\s+')
            return headers, self.run_tool(['readelf', '--sections', '--wide', str(binary)])
        before = metadata()
        self.run_tool(['patchelf', '--set-rpath', '$ORIGIN', str(binary)])
        # Independent readelf checks require final dynamic ORIGIN and exactly
        # preserved segment/section protections, including GNU RELRO.
        self.assertEqual(metadata(), before)

    def test_openh264_direct_make_preserves_literal_origin_and_full_relro(self):
        (self.root / 'Makefile').write_text(
            'all: libfixture.so.1\n'
            'libfixture.so.1: fixture.c\n'
            '\t$(CC) -shared -fPIC $< -Wl,-soname,$@ $(LDFLAGS) -o $@\n')
        self.run_tool(['make', 'CC=' + shutil.which('cc'), 'LDFLAGS=' + builder.OPENH264_LDFLAGS])
        self.assert_final_loader_and_protections(self.root / 'libfixture.so.1')

    def test_ffmpeg_eval_make_preserves_literal_origin_and_full_relro(self):
        flags = next(arg.split('=', 1)[1] for arg in POLICY['profiles']['linux-x64']['args']
                     if arg.startswith('--extra-ldsoflags='))
        (self.root / 'Makefile').write_text(
            'LDSOFLAGS=' + flags + '\n'
            'define RULES\n'
            'all: libfixture.so.1\n'
            'libfixture.so.1: fixture.c\n'
            '\t$$(CC) -shared -fPIC $$< -Wl,-soname,$$@ $(LDSOFLAGS) -o $$@\n'
            'endef\n'
            '$(eval $(RULES))\n')
        self.run_tool(['make', 'CC=' + shutil.which('cc')])
        self.assert_final_loader_and_protections(self.root / 'libfixture.so.1')

    def test_kvazaar_autoconf_make_libtool_roundtrip_preserves_origin_and_full_relro(self):
        required = ('autoreconf', 'autoconf', 'automake', 'libtoolize')
        missing = [name for name in required if not shutil.which(name)]
        if missing:
            self.skipTest('Missing bounded Autotools fixture tools: ' + ', '.join(missing))
        (self.root / 'configure.ac').write_text(
            'AC_INIT([codec-origin-fixture], [1.0])\nAM_INIT_AUTOMAKE([foreign])\n'
            'AC_PROG_CC\nLT_INIT([disable-static])\nAC_CONFIG_FILES([Makefile])\nAC_OUTPUT\n')
        (self.root / 'Makefile.am').write_text(
            'lib_LTLIBRARIES = libfixture.la\nlibfixture_la_SOURCES = fixture.c\n'
            'libfixture_la_LDFLAGS = -no-undefined -version-number 1:0:0\n')
        self.run_tool(['autoreconf', '-fi'])
        environment = os.environ.copy()
        for key in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'LIBRARY_PATH', 'CPATH'):
            environment.pop(key, None)
        self.run_tool([str(self.root / 'configure'), '--prefix=' + str(self.root / 'prefix'),
                       'LDFLAGS=' + builder.KVAZAAR_LDFLAGS], env=environment)
        self.run_tool(['make', '-j1'], env=environment)
        self.assert_final_loader_and_protections(self.root / '.libs/libfixture.so.1.0.0')


if __name__ == '__main__':
    unittest.main()
