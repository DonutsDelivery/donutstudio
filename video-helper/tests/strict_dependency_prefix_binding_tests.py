#!/usr/bin/env python3
"""Receipt and binding fixtures only; no encoder, compiler or runtime admission."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[1] / 'cmake/strict_dependency_prefix.py'
SPEC = importlib.util.spec_from_file_location('strict_dependency_prefix', MODULE)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


class PrefixFixture:
    def __init__(self, directory, system='Linux'):
        self.system = system
        self.root = directory / ('repo-' + system)
        profile_id, profile_path, builder_path, layout = policy.PROFILES[system]
        self.profile = {'schemaVersion': 1, 'id': profile_id, 'sources': [
            {'name': 'ffmpeg', 'version': '6.1.6', 'sha256': 'd4fcb164028dd3beee5d92c0ac72e46aac6973c75ea12dc14de07bf8f407370a'}]}
        owned = [profile_path, builder_path, *policy.RECIPE_INPUTS[system]]
        for relative in owned:
            text = json.dumps(self.profile) if relative == profile_path else 'fixture recipe\n'
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        inputs = {relative: policy.sha256(self.root / relative) for relative in owned}
        toolchain = {'architecture': 'arm64', 'sdk': '/fixture-sdk'}
        identity_inputs = ({'recipe': inputs, 'toolchain': toolchain} if system == 'Darwin' else
                           {'inputs': inputs, 'packages': 'fixture packages\n', 'image': 'fixture-image'}
                           if system == 'Windows' else
                           {'recipe': inputs, 'installedPackages': 'fixture packages\n', 'image': 'fixture-image'})
        identity = hashlib.sha256(json.dumps(identity_inputs, sort_keys=True).encode()).hexdigest()
        self.stage = directory / profile_id / identity
        self.prefix = self.stage / layout
        (self.prefix / 'include').mkdir(parents=True)
        (self.prefix / 'lib/pkgconfig').mkdir(parents=True)
        (self.stage / 'evidence').mkdir()
        (self.stage / 'evidence/installed-packages.txt').write_text('fixture packages\n')
        self.libraries = []
        suffix = '.dylib' if system == 'Darwin' else '.dll.a' if system == 'Windows' else '.so.60'
        for name, header in policy.HEADERS.items():
            include = self.prefix / 'include' / ('lib' + name)
            include.mkdir()
            (include / header).write_text('fixture public header\n')
            (include / 'version.h').write_text('fixture version header\n')
            if name == 'avcodec':
                (include / 'version_major.h').write_text('#define LIBAVCODEC_VERSION_MAJOR 60\n')
            (self.prefix / 'lib/pkgconfig' / ('lib' + name + '.pc')).write_text('fixture pc\n')
            library = self.prefix / 'lib' / ('lib' + name + suffix)
            library.write_bytes(b'fixture library')
            self.libraries.append(str(library))
        self.receipt = {'schemaVersion': 1, 'status': 'PASS', 'profile': profile_id,
                        'scope': 'Mac-codec-source-prefix-only' if system == 'Darwin' else 'dependency-staging-only',
                        'sources': self.profile['sources'], 'inputHashes': inputs,
                        'artifactHashes': {str(p.relative_to(self.stage)): policy.sha256(p)
                                           for p in self.prefix.rglob('*') if p.is_file()}}
        if system == 'Darwin':
            self.receipt.update(prefix=str(self.prefix), identity=identity, toolchain=toolchain)
        elif system == 'Windows':
            self.receipt.update(profileSha256=policy.sha256(self.root / profile_path), image='fixture-image')
        else:
            self.receipt.update(prefix=str(self.prefix), containerImageId='fixture-image')
        self.save()

    def save(self):
        (self.stage / 'DEPENDENCY_BUILD.json').write_text(json.dumps(self.receipt))

    def admit(self, **kwargs):
        settings = {'system': self.system, 'processor': 'arm64' if self.system == 'Darwin' else 'x86_64',
                    'mingw': 'ON' if self.system == 'Windows' else ''}
        settings.update(kwargs)
        return policy.admit_prefix(self.root, self.prefix, **settings)

    def bind(self, **kwargs):
        settings = {'includes': [str(self.prefix / 'include')],
                    'library_dirs': [str(self.prefix / 'lib')],
                    'pkg_dirs': [str(self.prefix / 'lib/pkgconfig')] * len(policy.HEADERS),
                    'libraries': self.libraries}
        settings.update(kwargs)
        return policy.admit_binding(self.prefix, **settings)


class StrictPrefixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.fixture = PrefixFixture(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_positive_each_reviewed_target_and_layout(self):
        self.fixture.admit()
        self.fixture.bind()
        for system in ('Windows', 'Darwin'):
            fixture = PrefixFixture(self.directory, system)
            fixture.admit()
            fixture.bind()

    def test_missing_receipt_and_incomplete_shared_closure(self):
        (self.fixture.stage / 'DEPENDENCY_BUILD.json').unlink()
        with self.assertRaises(OSError):
            self.fixture.admit()
        with self.assertRaisesRegex(ValueError, 'Incomplete shared'):
            self.fixture.bind(libraries=self.fixture.libraries[:-1])

    def test_receipt_status_profile_sources_and_recipe_hash(self):
        original = copy.deepcopy(self.fixture.receipt)
        for key, value in [('status', 'FAIL'), ('profile', 'system-ffmpeg'), ('sources', [])]:
            self.fixture.receipt = copy.deepcopy(original)
            self.fixture.receipt[key] = value
            self.fixture.save()
            with self.assertRaises(ValueError):
                self.fixture.admit()
        self.fixture.receipt = original
        self.fixture.receipt['inputHashes'][policy.PROFILES['Linux'][1]] = '0' * 64
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, 'recipe hash'):
            self.fixture.admit()

    def test_identity_and_artifact_drift(self):
        (self.fixture.stage / 'evidence/installed-packages.txt').write_text('different packages\n')
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.fixture.admit()
        (self.fixture.stage / 'evidence/installed-packages.txt').write_text('fixture packages\n')
        Path(self.fixture.libraries[0]).write_bytes(b'changed library')
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            self.fixture.admit()

    def test_added_and_missing_artifacts_and_escape_symlink(self):
        added = self.fixture.prefix / 'lib/contaminant.so'
        added.write_bytes(b'unknown library')
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            self.fixture.admit()
        added.unlink()
        outside = self.directory / 'system-header.h'
        outside.write_text('system')
        added.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes prefix'):
            self.fixture.admit()
        added.unlink()
        Path(self.fixture.libraries[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'artifact is missing'):
            self.fixture.admit()

    def test_reject_system_include_library_pkg_dir_and_bare_link(self):
        outside = self.directory / 'system'
        outside.mkdir()
        library = outside / 'libavcodec.so'
        library.write_bytes(b'system library')
        for arguments in ({'includes': [str(outside)]}, {'library_dirs': [str(outside)]},
                          {'pkg_dirs': [str(outside)] * len(policy.HEADERS)},
                          {'libraries': [str(library), *self.fixture.libraries[1:]]},
                          {'libraries': ['avcodec', *self.fixture.libraries[1:]]}):
            with self.assertRaises(ValueError):
                self.fixture.bind(**arguments)

    def test_reject_static_and_unreviewed_link_library(self):
        for name in ('libavcodec.a', 'libx264.so'):
            path = self.fixture.prefix / 'lib' / name
            path.write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'Unreviewed'):
                self.fixture.bind(libraries=[str(path), *self.fixture.libraries[1:]])

    def test_target_architecture_and_toolchain_fences(self):
        with self.assertRaisesRegex(ValueError, 'x86_64'):
            self.fixture.admit(processor='aarch64')
        mac = PrefixFixture(self.directory, 'Darwin')
        with self.assertRaisesRegex(ValueError, 'matching PASS profile receipt'):
            policy.admit_prefix(mac.root, self.fixture.prefix, 'Darwin', 'arm64')
        windows = PrefixFixture(self.directory, 'Windows')
        with self.assertRaisesRegex(ValueError, 'MinGW'):
            windows.admit(mingw='OFF')
        with self.assertRaisesRegex(ValueError, 'arm64 helper'):
            mac.admit(osx_architectures='arm64;x86_64')
        mac.receipt['toolchain']['architecture'] = 'x86_64'
        mac.save()
        with self.assertRaisesRegex(ValueError, 'toolchain'):
            mac.admit()

    def header_markers(self):
        return ''.join('# 1 ' + json.dumps(str(self.fixture.prefix / 'include' / ('lib' + family) / header)) + '\n'
                       for family, public in policy.HEADERS.items()
                       for header in ((public, 'version.h', 'version_major.h')
                                      if family == 'avcodec' else (public, 'version.h')))

    def test_consumed_headers_must_match_exact_prefix_and_versions(self):
        headers = policy.admit_consumed_headers(self.fixture.prefix, self.header_markers(),
                                               self.fixture.receipt['artifactHashes'])
        self.assertEqual(len(headers), 15)
        for suffix in ('avcodec.h', 'version.h', 'version_major.h'):
            header = self.fixture.prefix / 'include/libavcodec' / suffix
            old = header.read_text()
            header.write_text('different version or contents\n')
            with self.assertRaisesRegex(ValueError, 'differs from admitted prefix'):
                policy.admit_consumed_headers(self.fixture.prefix, self.header_markers(),
                                              self.fixture.receipt['artifactHashes'])
            header.write_text(old)
        with self.assertRaisesRegex(ValueError, 'Missing compiler-selected'):
            policy.admit_consumed_headers(self.fixture.prefix, '', self.fixture.receipt['artifactHashes'])

    def test_actual_ambient_headers_and_reparse_escape_are_refused(self):
        ambient = self.directory / 'homebrew/include/libavcodec'
        ambient.mkdir(parents=True)
        header = ambient / 'avcodec.h'
        header.write_text('ambient major 63\n')
        markers = self.header_markers() + '# 1 ' + json.dumps(str(header)) + '\n'
        with self.assertRaisesRegex(ValueError, 'escapes prefix'):
            policy.admit_consumed_headers(self.fixture.prefix, markers, self.fixture.receipt['artifactHashes'])
        owned = self.fixture.prefix / 'include/libavcodec/avcodec.h'
        owned.unlink()
        owned.symlink_to(header)
        with self.assertRaisesRegex(ValueError, 'escapes prefix'):
            policy.admit_consumed_headers(self.fixture.prefix, self.header_markers(),
                                          self.fixture.receipt['artifactHashes'])

    def compile_fixture(self):
        build = self.directory / 'generated-build'
        build.mkdir()
        source = self.fixture.root / 'video-helper/src/exporter.cpp'
        source.parent.mkdir(parents=True)
        source.touch()
        compiler = '/Library/Developer/CommandLineTools/usr/bin/clang++'
        cache = ('CMAKE_HOME_DIRECTORY:INTERNAL=' + str(self.fixture.root / 'video-helper') + '\n'
                 'ARBIT_VIDEO_DEPENDENCY_PREFIX:PATH=' + str(self.fixture.prefix) + '\n'
                 'CMAKE_CXX_COMPILER:UNINITIALIZED=' + compiler + '\n'
                 'CMAKE_CXX_COMPILER_LAUNCHER:STRING=ccache\n')
        (build / 'CMakeCache.txt').write_text(cache)
        (build / 'arbit-prefix-av-consumers.txt').write_text('helper\nfixture\n')
        rows = [{'directory': str(build), 'file': str(source),
                 'arguments': [compiler, '-isystem', '/opt/homebrew/include',
                               '-I' + str(self.fixture.prefix / 'include'), '-arch', 'arm64',
                               '-isysroot', '/recorded/SDK', '-DKEEP_CODEC_FEATURE=1',
                               '-MD', '-MF', 'dep.d', '-o', 'object.o', '-c', str(source)],
                 'output': 'CMakeFiles/' + target + '.dir/exporter.cpp.o'} for target in ('helper', 'fixture')]
        (build / 'compile_commands.json').write_text(json.dumps(rows))
        return build, rows

    def test_bound_commands_deduplicate_equivalent_consumers_only(self):
        build, rows = self.compile_fixture()
        with patch.object(policy.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=self.header_markers(), stderr='')) as run:
            policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[0], '/Library/Developer/CommandLineTools/usr/bin/clang++')
            self.assertNotIn('ccache', command)
            receipt = json.loads((build / 'arbit-prefix-av-header-admission.json').read_text())
            self.assertEqual(receipt['configuredCompilerLaunchers']['CXX'], 'ccache')
            self.assertIn('-DKEEP_CODEC_FEATURE=1', command)
            self.assertIn('/opt/homebrew/include', command)
            self.assertIn('/recorded/SDK', command)
            self.assertIn('-arch', command)
            for removed in ('-c', '-o', '-MD', '-MF', str(rows[0]['file'])):
                self.assertNotIn(removed, command)
            self.assertEqual(command[-4:], ['-E', '-x', 'c++', '-'])
            self.assertNotIn('shell', run.call_args.kwargs)
        rows[1]['arguments'].insert(1, '-DDISTINCT_PIXEL_CONTEXT=1')
        (build / 'compile_commands.json').write_text(json.dumps(rows))
        with patch.object(policy.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=self.header_markers(), stderr='')) as run:
            policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
            self.assertEqual(run.call_count, 2)

    def test_real_cmake_command_string_omits_cache_only_launcher(self):
        build, rows = self.compile_fixture()
        for row in rows:
            row['command'] = policy.shlex.join(row.pop('arguments'))
        (build / 'compile_commands.json').write_text(json.dumps(rows))
        with patch.object(policy.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=self.header_markers(), stderr='')) as run:
            policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][0], '/Library/Developer/CommandLineTools/usr/bin/clang++')
            self.assertNotIn('ccache', run.call_args.args[0])

    def test_represented_launcher_must_match_configured_identity(self):
        build, rows = self.compile_fixture()
        row = rows[0]
        row['arguments'].insert(0, 'ccache')
        cache_path = build / 'CMakeCache.txt'
        cache = dict(policy.re.findall(r'^([^/#\s:]+):[^=\n]*=(.*)$', cache_path.read_text(), policy.re.M))
        self.assertEqual(policy.bound_probe_command(row, cache, build, self.fixture.root)[:2],
                         ['ccache', '/Library/Developer/CommandLineTools/usr/bin/clang++'])
        for configured, represented in (('ccache', '/ambient/ccache'), ('sccache', 'ccache'), ('', 'ccache')):
            cache['CMAKE_CXX_COMPILER_LAUNCHER'] = configured
            row['arguments'][0] = represented
            with self.assertRaisesRegex(ValueError, 'Compiler/launcher differs'):
                policy.bound_probe_command(row, cache, build, self.fixture.root)

    def test_cpp_source_explicit_objective_cpp_context_is_preserved(self):
        build, rows = self.compile_fixture()
        cache_path = build / 'CMakeCache.txt'
        cache_path.write_text(cache_path.read_text() +
                             'CMAKE_OBJCXX_COMPILER:FILEPATH=/recorded/objc-clang++\n'
                             'CMAKE_OBJCXX_COMPILER_LAUNCHER:STRING=ccache\n')
        for row in rows:
            self.assertEqual(Path(row['file']).suffix, '.cpp')
            row['arguments'][0] = '/recorded/objc-clang++'
            row['arguments'][1:1] = ['-x', 'objective-c++']
        (build / 'compile_commands.json').write_text(json.dumps(rows))
        with patch.object(policy.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=self.header_markers(), stderr='')) as run:
            policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][0], '/recorded/objc-clang++')
            self.assertEqual(run.call_args.args[0][-4:], ['-E', '-x', 'objective-c++', '-'])
        joined = copy.deepcopy(rows[0])
        joined['arguments'][1:3] = ['-xobjective-c++']
        self.assertEqual(policy.header_probe_command(joined)[-2], 'objective-c++')
        for language in ('assembler', 'none'):
            joined['arguments'][1] = '-x' + language
            with self.assertRaisesRegex(ValueError, 'Unreviewed explicit compiler language'):
                policy.header_probe_command(joined)
        rows[0]['arguments'][0] = '/Library/Developer/CommandLineTools/usr/bin/clang++'
        (build / 'compile_commands.json').write_text(json.dumps(rows))
        with patch.object(policy.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Compiler/launcher differs'):
                policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
            run.assert_not_called()

    def test_generated_source_prefix_and_command_escapes_refuse(self):
        build, rows = self.compile_fixture()
        cache_path = build / 'CMakeCache.txt'
        cache = cache_path.read_text()
        for old, new in ((str(self.fixture.prefix), '/ambient/prefix'),
                         (str(self.fixture.root / 'video-helper'), '/another/source')):
            cache_path.write_text(cache.replace(old, new))
            with patch.object(policy.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'Compiler configuration differs'):
                    policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
                run.assert_not_called()
        cache_path.write_text(cache)
        for token in ('@unbound.rsp', '&&', '|'):
            row = copy.deepcopy(rows[0])
            row['arguments'].insert(2, token)
            with self.assertRaisesRegex(ValueError, 'Unreviewed compiler command'):
                policy.header_probe_command(row)

    def test_unbound_compiler_missing_consumer_and_probe_failure_refuse(self):
        build, rows = self.compile_fixture()
        rows[0]['arguments'][0] = '/ambient/clang++'
        (build / 'compile_commands.json').write_text(json.dumps(rows))
        with patch.object(policy.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Compiler/launcher differs'):
                policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
            run.assert_not_called()
        rows[0]['arguments'][0] = '/Library/Developer/CommandLineTools/usr/bin/clang++'
        (build / 'compile_commands.json').write_text(json.dumps(rows[:1]))
        with patch.object(policy.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=self.header_markers(), stderr='')):
            with self.assertRaisesRegex(ValueError, 'Missing actual compiler command'):
                policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)
        with patch.object(policy.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stdout='', stderr='bad SDK')):
            with self.assertRaisesRegex(ValueError, 'preprocessor failed.*bad SDK'):
                policy.admit_compile_headers(self.fixture.prefix, build, self.fixture.root)


if __name__ == '__main__':
    unittest.main()
