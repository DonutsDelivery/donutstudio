#!/usr/bin/env python3
"""Exercise package copy paths with sealed synthetic ELF fixtures, without builds."""
import hashlib
import importlib.util
import json
import io
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import tarfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('candidate_package', ROOT / 'installers/linux-video-helper-candidate.py')
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
COMMIT = 'a' * 40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def elf(path, needed, soname=None, loader='$ORIGIN'):
    strings = bytearray(b'\0')
    tags = []
    for tag, text in [(1, n) for n in needed] + ([(14, soname)] if soname else []) + [(29, loader)]:
        tags.append((tag, len(strings)))
        strings.extend(text.encode() + b'\0')
    dynamic_offset = 176
    string_offset = dynamic_offset + (len(tags) + 3) * 16
    tags += [(5, 0x400000 + string_offset), (10, len(strings)), (0, 0)]
    data = bytearray(string_offset + len(strings))
    data[:16] = b'\x7fELF\x02\x01\x01' + b'\0' * 9
    struct.pack_into('<HHIQQQIHHHHHH', data, 16, 3 if soname else 2, 62, 1, 0, 64, 0, 0, 64, 56, 2, 0, 0, 0)
    struct.pack_into('<IIQQQQQQ', data, 64, 1, 5, 0, 0x400000, 0, len(data), len(data), 4096)
    struct.pack_into('<IIQQQQQQ', data, 120, 2, 6, dynamic_offset, 0x400000 + dynamic_offset, 0, len(tags) * 16, len(tags) * 16, 8)
    for i, item in enumerate(tags):
        struct.pack_into('<qQ', data, dynamic_offset + i * 16, *item)
    data[string_offset:] = strings
    path.write_bytes(data)
    path.chmod(0o755)


class PackageFixture:
    def __init__(self, root):
        self.root = root
        self.package = root / 'package'
        self.candidate = self.package / 'video-helper'
        self.candidate.mkdir(parents=True)
        self.sources = json.loads((ROOT / 'scripts/ffmpeg-lgpl-candidate/policy-6.1.6.json').read_text())['sources']
        for source in self.sources:
            path = self.candidate / 'dependencies/archives' / source['archive']
            path.parent.mkdir(parents=True, exist_ok=True)
            if source['name'] == 'kvazaar':
                self.authored = {'CMakeLists.txt': b'cmake authored generator', 'src/kvazaar.h': b'kvazaar API header',
                                'src/version.h.in': b'version @PROJECT_VERSION@ compiler @KVZ_COMPILER_STRING@ date @CMAKE_BUILD_DATE@\n',
                                'src/kvazaarCMake.pc.in': b'prefix=@DEFERRED@CMAKE_INSTALL_PREFIX@DEFERRED@\nlibdir=@KVAZAAR_PC_LIBDIR@\nincdir=@KVAZAAR_PC_INCDIR@\n'}
                with tarfile.open(path, 'w:gz') as archive:
                    for name, content in self.authored.items():
                        member = tarfile.TarInfo('kvazaar-2.3.2/' + name)
                        member.size = len(content)
                        archive.addfile(member, io.BytesIO(content))
                        retained = self.candidate / 'dependencies/source/kvazaar-2.3.2' / name
                        retained.parent.mkdir(parents=True, exist_ok=True)
                        retained.write_bytes(content)
                source['headerSha256'] = hashlib.sha256(self.authored['src/kvazaar.h']).hexdigest()
            else:
                path.write_text('fixture pinned archive ' + source['name'])
            source['sha256'] = validator.digest(path)
        self.pinned_sources = hashlib.sha256(json.dumps(self.sources, sort_keys=True).encode()).hexdigest()
        validator.PINNED_SOURCES_SHA256 = self.pinned_sources
        self.packages = 'libc6:amd64\t2.39\tglibc\t2.39\npatchelf\t0.18\tpatchelf\t0.18\n'
        self.loader = {'loaderSearchPath': '$ORIGIN', 'loaderClosureValidation': 'environment-cleared',
            'loaderSealingTool': {'path': '/usr/bin/patchelf', 'sha256': 'e' * 64, 'version': 'patchelf 0.18',
                                  'systemPackage': ['patchelf', '0.18', 'patchelf', '0.18']}}
        self.dependency = {'schemaVersion': 1, 'status': 'PASS', 'scope': 'dependency-staging-only',
            'profile': validator.PROFILE, 'sourceCommit': COMMIT, 'inputHashes': {'recipe': 'b' * 64},
            'containerImageId': 'sha256:' + 'c' * 64, 'sources': self.sources,
            'sourcePaths': {'kvazaar': 'source/kvazaar-2.3.2'}, 'artifactHashes': {},
            'archiveHashes': {s['archive']: s['sha256'] for s in self.sources}}
        self.identity = hashlib.sha256(json.dumps({'recipe': self.dependency['inputHashes'],
            'image': self.dependency['containerImageId'], 'installedPackages': self.packages}, sort_keys=True).encode()).hexdigest()
        self.stage = '/cache/' + self.identity
        self.entries = {'ARBIT_VIDEO_DEPENDENCY_PREFIX': self.stage + '/prefix', 'ARBIT_REQUIRE_VIEWPORT': 'ON',
            'CMAKE_BUILD_TYPE': 'Release', 'ARBIT_BUILD_KVAZAAR10_ADAPTER': 'ON',
            'FETCHCONTENT_SOURCE_DIR_ARBIT_KVAZAAR10_SOURCE': '/work/.video-helper-lgpl-build/' + self.identity + '/kvazaar10-source',
            'CMAKE_HOME_DIRECTORY': '/work/video-helper', 'CMAKE_INSTALL_LIBDIR': 'lib',
            'CMAKE_INSTALL_INCLUDEDIR': 'include', 'CMAKE_INSTALL_PREFIX': '/usr/local'}
        for directory in ('dependencies/archives', 'dependencies/evidence', 'dependencies/source',
                          'evidence/kvazaar10', 'evidence/notices/system', 'evidence/system-sources'):
            path = self.candidate / directory / 'retained.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(directory)
        (self.candidate / 'evidence/CMakeCache.txt').write_text(''.join(k + ':STRING=' + v + '\n' for k, v in self.entries.items()))
        write_json(self.candidate / 'evidence/system-packages.json', [['libc6:amd64', 'glibc', '2.39']])
        elf(self.candidate / 'arbit-video-helper', ['libavcodec.so.60'])
        elf(self.candidate / 'libavcodec.so.60', ['libc.so.6'], 'libavcodec.so.60')
        other_av = ('libavformat.so.60', 'libavutil.so.58', 'libavdevice.so.60',
                    'libavfilter.so.9', 'libswscale.so.7', 'libswresample.so.4')
        for name in other_av:
            elf(self.candidate / name, ['libc.so.6'], name)
            self.dependency['artifactHashes']['prefix/lib/' + name] = validator.digest(self.candidate / name)
        self.dependency['artifactHashes']['prefix/lib/libavcodec.so.60'] = validator.digest(self.candidate / 'libavcodec.so.60')
        inventory_path = self.candidate / 'dependencies/evidence/installed-packages.txt'
        inventory_path.write_text(self.packages)
        self.dependency['artifactHashes']['evidence/installed-packages.txt'] = validator.digest(inventory_path)
        write_json(self.candidate / 'DEPENDENCY_BUILD.json', self.dependency)
        dep_hash = validator.digest(self.candidate / 'DEPENDENCY_BUILD.json')
        write_json(self.candidate / 'DEPENDENCY_SELECTION.json', {'schemaVersion': 1, 'status': 'PASS',
            'profile': validator.PROFILE, 'identity': self.identity, 'stage': self.stage, 'prefix': self.stage + '/prefix',
            'dependencyReceiptSha256': dep_hash})
        source_inputs = {'schemaVersion': 1, 'profile': validator.PROFILE, 'dependencyIdentity': self.identity,
            'dependencyReceiptSha256': dep_hash, 'source': self.sources[2], 'sourceArchiveSha256': self.sources[2]['sha256'],
            'admittedSourceDirectory': self.stage + '/source/kvazaar-2.3.2',
            'helperSourceDirectory': self.entries['FETCHCONTENT_SOURCE_DIR_ARBIT_KVAZAAR10_SOURCE'],
            'authoredHashes': {name: hashlib.sha256(value).hexdigest() for name, value in self.authored.items()},
            'generator': {'path': 'CMakeLists.txt', 'sha256': hashlib.sha256(self.authored['CMakeLists.txt']).hexdigest()}}
        source_input_path = self.candidate / 'evidence/kvazaar10/SOURCE_INPUTS.json'
        write_json(source_input_path, source_inputs)
        descriptor = self.candidate / 'evidence/kvazaar10/CMakeCCompiler.cmake'
        descriptor.write_text('set(CMAKE_C_COMPILER_ID "GNU")\nset(CMAKE_C_COMPILER_VERSION "13.3.0")\n')
        generation = {'sourceDateEpoch': 0, 'cmakeCacheSha256': validator.digest(self.candidate / 'evidence/CMakeCache.txt'),
            'compilerConfiguration': {'path': '/work/.video-helper-lgpl-build/' + self.identity + '/CMakeFiles/3.28/CMakeCCompiler.cmake',
                                      'sha256': validator.digest(descriptor), 'id': 'GNU', 'version': '13.3.0'},
            'substitutions': {'PROJECT_VERSION': '2.3.2', 'KVZ_COMPILER_STRING': 'GNU 13.3.0', 'CMAKE_BUILD_DATE': '1970-01-01',
                'DEFERRED': '@', 'KVAZAAR_PC_LIBDIR': '${exec_prefix}/lib', 'KVAZAAR_PC_INCDIR': '${prefix}/include',
                'CMAKE_PROJECT_DESCRIPTION': '', 'CMAKE_PROJECT_HOMEPAGE_URL': '', 'LIBS': ''}}
        generated_contents = {'src/version.h': 'version 2.3.2 compiler GNU 13.3.0 date 1970-01-01\n',
                              'src/kvazaar.pc.temp': 'prefix=@CMAKE_INSTALL_PREFIX@\nlibdir=${exec_prefix}/lib\nincdir=${prefix}/include\n'}
        generated = {}
        for name, template in {'src/version.h': 'src/version.h.in', 'src/kvazaar.pc.temp': 'src/kvazaarCMake.pc.in'}.items():
            output = self.candidate / 'evidence/kvazaar10/generated' / Path(name).name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(generated_contents[name])
            generated[name] = {'template': template, 'templateSha256': source_inputs['authoredHashes'][template],
                               'sha256': validator.digest(output)}
        self.source_binding = {**source_inputs, 'status': 'PASS',
            'preparedReceiptSha256': validator.digest(source_input_path), 'generatedOutputs': generated,
            'generationConfiguration': generation}
        write_json(self.candidate / 'evidence/kvazaar10/SOURCE_BINDING.json', self.source_binding)
        self.provenance = {**self.loader, 'schemaVersion': 1, 'status': 'PASS', 'scope': 'ELF-dependency-provenance-only',
            'profile': validator.PROFILE, 'identity': self.identity, 'dependencyReceiptSha256': dep_hash,
            'helperSha256': validator.digest(self.candidate / 'arbit-video-helper'), 'dependencies': [
                {'soname': 'libavcodec.so.60', 'path': self.stage + '/prefix/lib/libavcodec.so.60',
                 'sha256': validator.digest(self.candidate / 'libavcodec.so.60'), 'needed': ['libc.so.6'],
                 'origin': 'source-prefix', 'source': self.sources[0], 'prefixPath': 'prefix/lib/libavcodec.so.60',
                 'sourceSha256': validator.digest(self.candidate / 'libavcodec.so.60'), 'bundledPath': 'libavcodec.so.60'},
                {'soname': 'libc.so.6', 'path': '/usr/lib/x86_64-linux-gnu/libc.so.6', 'sha256': 'd' * 64,
                 'needed': [], 'origin': 'system-package', 'sourcePackage': 'glibc', 'sourceVersion': '2.39',
                 'binaryPackage': 'libc6:amd64', 'binaryVersion': '2.39', 'platformABI': True}]}
        for name in other_av:
            self.provenance['dependencies'].append({'soname': name, 'path': self.stage + '/prefix/lib/' + name,
                'sha256': validator.digest(self.candidate / name), 'needed': ['libc.so.6'], 'origin': 'source-prefix',
                'source': self.sources[0], 'prefixPath': 'prefix/lib/' + name,
                'sourceSha256': validator.digest(self.candidate / name), 'bundledPath': name})
        write_json(self.candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', self.provenance)
        private = root / 'private'
        private.mkdir()
        (private / 'helper-source.tar').write_text('private first party source')
        (private / 'video-helper.symbols.tar.gz').write_text('private detached symbols')
        self.receipt = {**self.loader, 'schemaVersion': 1, 'status': 'PASS', 'scope': 'helper-candidate-only', 'profile': validator.PROFILE,
            'sourceCommit': COMMIT, 'identity': self.identity, 'containerImageId': self.dependency['containerImageId'],
            'installedPackages': self.packages, 'helperSha256': self.provenance['helperSha256'],
            'kvazaar10SourceBinding': self.source_binding,
            'configuration': {'cacheSha256': validator.digest(self.candidate / 'evidence/CMakeCache.txt'), 'entries': self.entries},
            'buildProfile': 'shipping', 'debugSymbolsEmbedded': False, 'midiDebugBundled': False,
            'sourceArchive': {'path': str(private / 'helper-source.tar'), 'sha256': validator.digest(private / 'helper-source.tar'),
                              'sourceCommit': COMMIT, 'scope': 'private-helper-source'},
            'symbolArtifact': {'localPath': str(private / 'video-helper.symbols.tar.gz'),
                               'sha256': validator.digest(private / 'video-helper.symbols.tar.gz')},
            'privateEvidence': {p.name: validator.digest(p) for p in private.iterdir()}}
        self.reseal_candidate()
        for name in (*validator.APP_FILES, 'DonutStudio.svg', 'DonutStudioUpdater.svg',
                     'DonutStudioSidecar/whisper/setup_venv.sh'):
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('app ' + name)
            path.chmod(0o755)
        (self.package / 'sidecar').mkdir()
        (self.package / 'sidecar/retained.txt').write_text('sidecar')
        write_json(self.package / 'RELEASE_BUILD.json', {'schemaVersion': 1, 'sourceCommit': COMMIT,
            'buildProfile': 'shipping', 'debugSymbolsEmbedded': False, 'midiDebugBundled': False})
        shutil.copy2(ROOT / 'installers/install-linux.sh', self.package)
        validator.seal_package(self.package, COMMIT)

    def reseal_candidate(self):
        self.receipt['artifactHashes'] = {p.relative_to(self.candidate).as_posix(): validator.digest(p)
            for p in self.candidate.rglob('*') if p.is_file() and p.name != 'CANDIDATE_BUILD.json'}
        write_json(self.candidate / 'CANDIDATE_BUILD.json', self.receipt)

    def fixture_validator(self, directory):
        destination = directory / 'linux-video-helper-candidate.py'
        code = (ROOT / 'installers/linux-video-helper-candidate.py').read_text()
        import re
        code = re.sub(r"PINNED_SOURCES_SHA256 = '[0-9a-f]+'", "PINNED_SOURCES_SHA256 = '" + self.pinned_sources + "'", code)
        destination.write_text(code)

    def install(self):
        home = self.root / 'home'
        home.mkdir(exist_ok=True)
        installed = home / '.local/share/Arbit/video-helper'
        installed.mkdir(parents=True, exist_ok=True)
        (installed / 'old.txt').write_text('old installed helper')
        app = home / '.local/bin/DonutStudio'
        app.parent.mkdir(parents=True, exist_ok=True)
        app.write_text('old installed app')
        result = subprocess.run(['bash', str(self.package / 'install-linux.sh'), '--update'],
            env={**os.environ, 'HOME': str(home)}, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        return result, installed, app

    def appimage(self):
        repo = self.root / 'appimage-repo'
        (repo / 'installers').mkdir(parents=True, exist_ok=True)
        for name in ('make-appimage.sh', 'linux-video-helper-candidate.py'):
            shutil.copy2(ROOT / 'installers' / name, repo / 'installers')
        self.fixture_validator(repo / 'installers')
        icon = repo / 'plugin/Source/DonutStudio.svg'
        icon.parent.mkdir(parents=True, exist_ok=True)
        icon.write_text('<svg/>')
        tool = repo / '.tmp/appimage/tools/linuxdeploy-x86_64.AppImage'
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.write_text('#!/bin/bash\ntouch DonutStudio-x86_64.AppImage\n')
        tool.chmod(0o755)
        commands = repo / 'fake-bin'
        commands.mkdir(exist_ok=True)
        (commands / 'desktop-file-validate').write_text('#!/bin/bash\nexit 0\n')
        (commands / 'desktop-file-validate').chmod(0o755)
        result = subprocess.run(['bash', str(repo / 'installers/make-appimage.sh'), str(self.package), str(repo / 'out')],
            env={**os.environ, 'PATH': str(commands) + ':' + os.environ['PATH']}, capture_output=True, text=True)
        return result, repo / '.tmp/appimage/DonutStudio.AppDir'

    def docker_package(self, candidate=True):
        repo = self.root / 'docker-repo'
        (repo / 'scripts').mkdir(parents=True, exist_ok=True)
        (repo / 'installers').mkdir()
        for name in ('install-linux.sh', 'make-linux-installer.sh', 'linux-video-helper-candidate.py', 'LICENSE.txt'):
            shutil.copy2(ROOT / 'installers' / name, repo / 'installers')
        self.fixture_validator(repo / 'installers')
        shutil.copy2(ROOT / 'scripts/build-linux-docker-build.sh', repo / 'scripts')
        (repo / 'scripts/prepare-release-symbols.py').write_text(
            "import sys,json,pathlib\np=pathlib.Path(sys.argv[sys.argv.index('--report')+1])\n"
            "p.write_text(json.dumps({'buildProfile':'shipping','debugSymbolsEmbedded':False,'midiDebugBundled':False,'symbolArtifact':{}}))\n")
        shutil.copy2(ROOT / 'scripts/release-build-provenance.sh', repo / 'scripts')
        relative_files = [
            'build-linux-docker/HarmonicMIDI_artefacts/Release/VST3/DonutStudio.vst3/Contents/x86_64-linux/DonutStudio.so',
            'build-linux-docker/HarmonicMIDI_artefacts/Release/VST3/DonutStudio.vst3/Contents/x86_64-linux/ArbitPluginScanner',
            'build-linux-docker/HarmonicMIDI_artefacts/Release/CLAP/DonutStudio.clap',
            'build-linux-docker/HarmonicMIDI_artefacts/Release/CLAP/ArbitPluginScanner',
            'build-linux-docker/HarmonicMIDI_artefacts/Release/Standalone/DonutStudio',
            'build-linux-docker/DonutStudioUpdater_artefacts/Release/DonutStudio Updater',
            'build-linux-docker/ArbitPluginHost_artefacts/Release/ArbitPluginHost',
            'plugin/Soundfonts/sound.txt', 'bitwig-script/DonutStudio.control.js',
            'plugin/Source/DonutStudio.svg', 'plugin/Source/UpdaterIcon.svg', 'plugin/THIRD_PARTY_LICENSES.md',
            'content/shader-packs/shader.txt', 'content/mod-presets/preset.txt', 'content/example-projects/project.txt',
            'sidecar/whisper/setup_venv.sh', 'sidecar/music_ai/download_notochord.py', 'sidecar/music_ai/setup_venv.sh']
        for name in relative_files:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture ' + name)
            path.chmod(0o755)
        commands = repo / 'fake-bin'
        commands.mkdir()
        (commands / 'cmake').write_text('#!/bin/bash\nexit 0\n')
        (commands / 'cmake').chmod(0o755)
        env = {**os.environ, 'PATH': str(commands) + ':' + os.environ['PATH'],
               'DONUTSTUDIO_RELEASE_SOURCE_COMMIT': COMMIT, 'DONUTSTUDIO_RELEASE_VERSION': 'fixture',
               'DONUTSTUDIO_RELEASE_BUILD_TIMESTAMP': '2026-10-01T00:00:00Z'}
        env.pop('DONUTSTUDIO_LINUX_VIDEO_HELPER_CANDIDATE', None)
        if candidate:
            env['DONUTSTUDIO_LINUX_VIDEO_HELPER_CANDIDATE'] = str(self.candidate)
        else:
            helper = repo / 'video-helper/build-docker'
            helper.mkdir(parents=True)
            shutil.copy2(self.candidate / 'arbit-video-helper', helper)
            (helper / 'libvulkan.so.1').write_text('transitional Vulkan')
        result = subprocess.run(['bash', str(repo / 'scripts/build-linux-docker-build.sh')],
                                env=env, capture_output=True, text=True)
        return result, repo / 'dist-linux-local'


class PackageInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='linux-helper-package-')
        self.fixture = PackageFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_candidate_and_private_evidence(self):
        receipt = validator.validate_candidate(self.fixture.candidate, COMMIT)
        validator.validate_private_evidence(self.fixture.candidate, receipt)
        validator.validate_package(self.fixture.package, COMMIT)

    def test_installer_copies_complete_candidate_without_private_source(self):
        result, installed, _ = self.fixture.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('All 13 components installed successfully.', result.stdout)
        self.assertTrue((installed / 'arbit-video-helper').is_file())
        for path in self.fixture.candidate.rglob('*'):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (installed / path.relative_to(self.fixture.candidate)).read_bytes())
        self.assertFalse((installed / 'helper-source.tar').exists())
        self.assertFalse((installed / 'old.txt').exists())

    def test_appimage_copies_complete_candidate_and_colocated_library(self):
        result, appdir = self.fixture.appimage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((appdir / 'usr/bin/arbit-video-helper').resolve(), appdir / 'usr/bin/video-helper/arbit-video-helper')
        for path in self.fixture.candidate.rglob('*'):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (appdir / 'usr/bin/video-helper' / path.relative_to(self.fixture.candidate)).read_bytes())

    def test_zip_and_self_extracting_installer_preserve_complete_candidate(self):
        result, out = self.fixture.docker_package()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with zipfile.ZipFile(out / 'DonutStudio-Linux.zip') as archive:
            for path in self.fixture.candidate.rglob('*'):
                if path.is_file():
                    name = 'video-helper/' + path.relative_to(self.fixture.candidate).as_posix()
                    self.assertEqual(archive.read(name), path.read_bytes())
            self.assertIn('LINUX_HELPER_PACKAGE.sha256', archive.namelist())
            self.assertNotIn('video-helper/helper-source.tar', archive.namelist())
        self.assertTrue((out / 'DonutStudio-Linux-Installer.sh').is_file())
        validator.validate_package(out / 'linux', COMMIT)

    def test_transitional_docker_route_preserves_default_helper_input(self):
        result, out = self.fixture.docker_package(candidate=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((out / 'linux/video-helper/libvulkan.so.1').read_text(), 'transitional Vulkan')
        self.assertFalse((out / 'linux/LINUX_HELPER_PACKAGE.json').exists())
        self.assertFalse((out / 'linux/video-helper/libavcodec.so.60').exists())

    def test_transitional_appimage_route_keeps_original_layout(self):
        for name in ('LINUX_HELPER_PACKAGE.json', 'LINUX_HELPER_PACKAGE.sha256'):
            (self.fixture.package / name).unlink()
        (self.fixture.candidate / 'CANDIDATE_BUILD.json').unlink()
        (self.fixture.candidate / 'libvulkan.so.1').write_text('transitional Vulkan')
        result, appdir = self.fixture.appimage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((appdir / 'usr/bin/arbit-video-helper').is_symlink())
        self.assertEqual((appdir / 'usr/bin/libvulkan.so.1').read_text(), 'transitional Vulkan')
        self.assertFalse((appdir / 'usr/bin/video-helper').exists())

    def test_missing_or_altered_library_preserves_installed_app(self):
        for change in ('missing', 'altered'):
            with self.subTest(change=change):
                library = self.fixture.candidate / 'libavcodec.so.60'
                original = library.read_bytes()
                if change == 'missing':
                    library.unlink()
                else:
                    library.write_bytes(original + b'changed')
                result, installed, app = self.fixture.install()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((installed / 'old.txt').read_text(), 'old installed helper')
                self.assertEqual(app.read_text(), 'old installed app')
                library.write_bytes(original)

    def test_missing_receipt_rejected_before_appimage_mutation(self):
        (self.fixture.candidate / 'DEPENDENCY_SELECTION.json').unlink()
        result, appdir = self.fixture.appimage()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(appdir.exists())

    def test_wrong_source_rejected(self):
        with self.assertRaises(ValueError):
            validator.validate_candidate(self.fixture.candidate, 'f' * 40)

    def test_false_pass_cannot_hide_unresolved_loader_dependency(self):
        elf(self.fixture.candidate / 'arbit-video-helper', ['libmissing.so.1'])
        self.fixture.receipt['helperSha256'] = validator.digest(self.fixture.candidate / 'arbit-video-helper')
        self.fixture.provenance['helperSha256'] = self.fixture.receipt['helperSha256']
        write_json(self.fixture.candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', self.fixture.provenance)
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_wrong_source_association_rejected_even_if_resealed(self):
        self.fixture.provenance['dependencies'][0]['source'] = self.fixture.sources[1]
        write_json(self.fixture.candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', self.fixture.provenance)
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'wrong source'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_adapter_authored_source_drift_rejected_even_if_resealed(self):
        (self.fixture.candidate / 'dependencies/source/kvazaar-2.3.2/src/kvazaar.h').write_text('changed authored source')
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'retained adapter source drift'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_adapter_generated_output_drift_rejected_even_if_resealed(self):
        (self.fixture.candidate / 'evidence/kvazaar10/generated/version.h').write_text('changed generated output')
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'generated output/template identity'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_adapter_arbitrary_generated_bytes_rejected_even_if_receipts_resealed(self):
        output = self.fixture.candidate / 'evidence/kvazaar10/generated/version.h'
        output.write_text('arbitrary generated content')
        self.fixture.source_binding['generatedOutputs']['src/version.h']['sha256'] = validator.digest(output)
        write_json(self.fixture.candidate / 'evidence/kvazaar10/SOURCE_BINDING.json', self.fixture.source_binding)
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'generated bytes differ'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_adapter_copy_cannot_reuse_dependency_stage_source(self):
        self.fixture.source_binding['helperSourceDirectory'] = self.fixture.source_binding['admittedSourceDirectory']
        write_json(self.fixture.candidate / 'evidence/kvazaar10/SOURCE_BINDING.json', self.fixture.source_binding)
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'source copy'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_presealing_pass_receipt_is_rejected(self):
        self.fixture.receipt.pop('loaderClosureValidation')
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'loader closure admission'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_escaped_loader_path_rejected_even_if_resealed(self):
        elf(self.fixture.candidate / 'arbit-video-helper', ['libavcodec.so.60'], loader='/tmp/outside')
        self.fixture.receipt['helperSha256'] = validator.digest(self.fixture.candidate / 'arbit-video-helper')
        self.fixture.provenance['helperSha256'] = self.fixture.receipt['helperSha256']
        write_json(self.fixture.candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', self.fixture.provenance)
        self.fixture.reseal_candidate()
        with self.assertRaisesRegex(ValueError, 'loader search path'):
            validator.validate_candidate(self.fixture.candidate, COMMIT)

    def test_extra_unsealed_helper_file_rejected_at_install(self):
        (self.fixture.candidate / 'stale.so').write_text('stale')
        result, installed, _ = self.fixture.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((installed / 'old.txt').exists())

    def test_symlink_escape_rejected_at_install(self):
        library = self.fixture.candidate / 'libavcodec.so.60'
        other = self.fixture.root / 'outside.so'
        library.rename(other)
        library.symlink_to(other)
        result, installed, _ = self.fixture.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((installed / 'old.txt').exists())

    def test_altered_app_bytes_rejected(self):
        (self.fixture.package / 'DonutStudio').write_text('different app')
        with self.assertRaisesRegex(ValueError, 'application bytes'):
            validator.validate_package(self.fixture.package)
        result, installed, app = self.fixture.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(app.read_text(), 'old installed app')
        self.assertTrue((installed / 'old.txt').exists())

    def test_transitional_install_route_retains_existing_helper_copy(self):
        for name in ('LINUX_HELPER_PACKAGE.json', 'LINUX_HELPER_PACKAGE.sha256'):
            (self.fixture.package / name).unlink()
        (self.fixture.candidate / 'CANDIDATE_BUILD.json').unlink()
        result, installed, _ = self.fixture.install()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('All 13 components installed successfully.', result.stdout)
        self.assertTrue((installed / 'arbit-video-helper').is_file())
        self.assertTrue((installed / 'libavcodec.so.60').is_file())


if __name__ == '__main__':
    unittest.main()
