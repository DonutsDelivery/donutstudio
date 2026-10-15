#!/usr/bin/env python3
"""Saved-output Windows admission fixtures. No compiler or Windows app is used."""
import copy
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('windows_candidate', ROOT / 'scripts/video-helper-dependencies/windows.py')
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
COMMIT = '1' * 40
IDENTITY = {'sourceCommit': COMMIT, 'inputHashes': {'video-helper/src/exporter.cpp': 'fixture'}}


class Wiring(unittest.TestCase):
    def test_default_and_candidate_remain_explicit(self):
        helper = (ROOT / 'scripts/build-windows-videohelper-docker.sh').read_text()
        self.assertIn('CANDIDATE_CMAKE_ARGS=()\ncase', helper)
        candidate = helper.split('lgpl-windows-mingw-v1)', 1)[1].split('*)', 1)[0]
        for option in ('-DARBIT_BUILD_KVAZAAR10_ADAPTER=ON', '-DARBIT_VIDEO_DEPENDENCY_PREFIX=',
                       '-DARBIT_REQUIRE_VIEWPORT=ON', '-DFETCHCONTENT_SOURCE_DIR_ARBIT_KVAZAAR10_SOURCE='):
            self.assertIn(option, candidate)
        for name, path in w.GNU_TOOLS.items():
            self.assertIn('-DCMAKE_' + name.upper() + '=' + path, candidate)
        self.assertIn('transitional-msys2) ;;', helper)
        self.assertLess(helper.index('--strip-unneeded'), helper.index('--seal-candidate'))
        self.assertLess(helper.index('--add-gnu-debuglink='), helper.index('--seal-candidate'))

    def test_compile_and_link_evidence_is_final_helper_candidate_only(self):
        helper = (ROOT / 'scripts/build-windows-videohelper-docker.sh').read_text()
        candidate = helper.split('lgpl-windows-mingw-v1)', 1)[1].split('*)', 1)[0]
        self.assertIn('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', candidate)
        self.assertIn('-DARBIT_HELPER_LINK_MAP_FILE="$BUILD_DIR/arbit-video-helper.link.map"', candidate)
        self.assertNotIn('CMAKE_EXE_LINKER_FLAGS', candidate)
        outside = helper.split('lgpl-windows-mingw-v1)', 1)[0] + helper.split('*)', 1)[1]
        self.assertNotIn('CMAKE_EXPORT_COMPILE_COMMANDS', outside)
        self.assertNotIn('ARBIT_HELPER_LINK_MAP_FILE', outside)
        self.assertIn('"${CANDIDATE_CMAKE_ARGS[@]}"', outside)
        self.assertLess(helper.index('ARBIT_HELPER_LINK_MAP_FILE'), helper.index('cmake -S video-helper'))

    def test_candidate_transform_trace_remains_private_and_legacy_steps_remain(self):
        helper = (ROOT / 'scripts/build-windows-videohelper-docker.sh').read_text()
        trace = helper.split("<<'PYTRACE'\n", 1)[1].split('\nPYTRACE\n', 1)[0]
        self.assertIn('"$OUT/private-candidate-evidence"', helper)
        self.assertIn("Path('/usr/bin/x86_64-w64-mingw32-' + name)", trace)
        self.assertIn("for name in ('strip', 'objcopy'):", trace)
        for name in ('windows-helper-linked.exe', 'windows-helper-stripped.exe',
                     'windows-helper-debuglinked.exe', 'WINDOWS_HELPER_TRANSFORMATIONS.json'):
            self.assertIn(name, trace)
        for field in ('resolvedPath', 'sha256', 'version', 'commandSha256', 'inputSha256', 'outputSha256'):
            self.assertIn(field, trace)
        self.assertIn('shlex.join(argv)', trace)
        self.assertIn('subprocess.run(argv, check=True)', trace)
        self.assertIn("map_path = build / 'arbit-video-helper.link.map'", trace)
        self.assertIn("if digest(linked) != original_sha", trace)
        legacy = helper.split('\nPYTRACE\nelse\n', 1)[1].split('\nfi\n', 1)[0]
        for command in ('x86_64-w64-mingw32-objcopy --only-keep-debug',
                        'x86_64-w64-mingw32-strip --strip-unneeded', '--add-gnu-debuglink='):
            self.assertIn(command, legacy)
        self.assertLess(helper.index('WINDOWS_HELPER_TRANSFORMATIONS.json'), helper.index('--seal-candidate'))

    def test_full_pipeline_forwards_profile_and_admits_before_zip(self):
        full = (ROOT / 'scripts/build-windows-local.sh').read_text()
        self.assertIn('--lgpl-candidate-package)', full)
        self.assertIn('VH_IMAGE=arbit-windows-videohelper:lgpl-mingw-v1', full)
        self.assertIn('-e DONUTSTUDIO_WINDOWS_DEPENDENCY_PROFILE', full)
        self.assertIn('-e DONUTSTUDIO_DEPENDENCY_IMAGE_ID', full)
        self.assertIn('-e ARBIT_DIST_WINDOWS_DIR="$WINDOWS_OUT_CONTAINER"', full)
        self.assertIn('--stage "$WINDOWS_STAGE_CONTAINER"', full)
        self.assertLess(full.index('--seal-app'), full.index('"$VH_IMAGE" bash'))
        self.assertLess(full.index('--seal-package'), full.index('--admit-package'))
        self.assertLess(full.index('--admit-package'), full.index('zip -rq'))
        self.assertLess(full.index('python3 scripts/prepare-release-symbols.py'), full.index('--admit-app'))
        self.assertLess(full.index('--admit-app'), full.index('--seal-package'))
        self.assertNotIn('--strip-unneeded', full)
        standalone = (ROOT / 'scripts/build-windows-videohelper-local.sh').read_text()
        self.assertLess(standalone.index('Windows helper candidate staged'), standalone.index('zip -rq'))


    def test_source_mesa_stager_and_resource_guard_are_candidate_only(self):
        docker = (ROOT / 'scripts/build-windows-videohelper-docker.sh').read_text()
        stage = docker.split('# A separate helper copy loads Mesa', 1)[1].split('if [ -n "$DEPENDENCY_RECEIPT" ]; then', 2)
        self.assertIn('windows_mesa.py', stage[1])
        self.assertIn('--stage "$STAGE" --receipt "$DEPENDENCY_RECEIPT"', stage[1])
        self.assertIn('else\n  bash scripts/stage-windows-mesa-fallback.sh "$STAGE"\nfi', stage[1])
        local = (ROOT / 'scripts/build-windows-videohelper-local.sh').read_text()
        self.assertIn('WINDOWS_HELPER_DOCKER_ARGS=()\ncase', local)
        candidate = local.split('lgpl-windows-mingw-v1)', 1)[1].split('*)', 1)[0]
        self.assertIn('source "$ROOT/scripts/docker-release-resource-guard.sh"', candidate)
        self.assertIn('release_docker_resource_guard_init', candidate)
        self.assertIn('"$ROOT/video-helper/dependency-cache-windows" "$ROOT/dist-windows-local"', candidate)
        self.assertIn('WINDOWS_HELPER_DOCKER_ARGS=("${RELEASE_DOCKER_RUN_ARGS[@]}")', candidate)
        self.assertIn('"${WINDOWS_HELPER_DOCKER_ARGS[@]}"', local.split('$DOCKER run --rm', 1)[1])


class MesaPolicyGlue(unittest.TestCase):
    def test_renderer_sources_join_the_dependency_source_receipt(self):
        base, renderer = {'name': 'ffmpeg'}, {'name': 'mesa'}
        self.assertEqual(w.dependency_sources({'sources': [base]}), [base])
        self.assertEqual(w.dependency_sources({'sources': [base], 'softwareRenderer': {'sources': [renderer]}}),
                         [base, renderer])

    def test_renderer_policy_and_receipt_refusals_propagate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = json.loads(w.PROFILE.read_text())
            profile['softwareRenderer'] = {'fixture': True}
            policy = root / 'profile.json'
            policy.write_text(json.dumps(profile))
            artifact = root / 'source.bin'
            artifact.write_bytes(b'fixture dependency source')
            receipt_path = root / 'DEPENDENCY_BUILD.json'
            receipt = {'status': 'PASS', 'profileSha256': w.sha256(policy), 'inputHashes': {},
                       'artifactHashes': {'source.bin': w.sha256(artifact)}}
            receipt_path.write_text(json.dumps(receipt))
            engine = Mock()
            with patch.object(w, 'PROFILE', policy), patch.object(w, 'mesa_engine', return_value=engine):
                w.verify_policy()
                engine.verify_profile.assert_called_once_with(profile)
                w.validate_receipt(receipt_path)
                engine.validate_receipt.assert_called_once_with(receipt_path, receipt)
                engine.validate_receipt.side_effect = ValueError('renderer source mismatch')
                with self.assertRaisesRegex(ValueError, 'renderer source mismatch'):
                    w.validate_receipt(receipt_path)
                engine.reset_mock()
                artifact.write_bytes(b'changed dependency source')
                with self.assertRaisesRegex(ValueError, 'Dependency artifact changed'):
                    w.validate_receipt(receipt_path)
                engine.validate_receipt.assert_not_called()

    def test_fallback_runtime_is_copied_from_the_admitted_primary_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage, prefix = root / 'stage', root / 'sysroot/mingw64'
            fallback = stage / 'software-renderer'
            fallback.mkdir(parents=True)
            (prefix / 'bin').mkdir(parents=True)
            (stage / 'arbit-video-helper.exe').write_bytes(b'MZ helper fixture')
            (fallback / 'arbit-video-helper.exe').write_bytes(b'MZ helper fixture')
            known = {}
            for name in ('opengl32.dll', 'libgallium_wgl.dll'):
                (fallback / name).write_bytes(b'MZ source Mesa fixture ' + name.encode())
                known[name] = {'sha256': w.sha256(fallback / name), 'source': 'mesa',
                               'prefixPath': 'software-renderer/' + name}
            runtime = prefix / 'bin/libwinpthread-1.dll'
            runtime.write_bytes(b'MZ admitted MinGW runtime fixture')
            known[runtime.name] = {'sha256': w.sha256(runtime), 'source': 'mingw-w64',
                                   'prefixPath': 'bin/' + runtime.name}
            receipt = root / 'DEPENDENCY_BUILD.json'
            receipt.write_text('fixture receipt')
            dependency = {'profile': 'fixture', 'dllProvenance': known,
                          'executableProvenance': {}, 'limitations': []}
            def imported(path):
                return {'opengl32.dll': ['libgallium_wgl.dll'],
                        'libgallium_wgl.dll': ['libwinpthread-1.dll']}.get(path.name, ['KERNEL32.dll'])
            with patch.object(w, 'validate_receipt', return_value=dependency), \
                    patch.object(w, 'imports', side_effect=imported):
                closure = w.audit_closure(stage, receipt, copy=True)
            self.assertEqual(w.sha256(fallback / runtime.name), w.sha256(runtime))
            self.assertIn('software-renderer/libwinpthread-1.dll', closure['peHashes'])
            self.assertFalse((stage / 'opengl32.dll').exists())


class PackageAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage = self.root / 'windows'
        self.helper = self.stage / 'video-helper'
        self.helper.mkdir(parents=True)
        (self.stage / 'DonutStudio.exe').write_bytes(b'MZ app fixture')
        (self.helper / 'arbit-video-helper.exe').write_bytes(b'MZ helper fixture')
        (self.helper / 'candidate-evidence').mkdir()
        (self.helper / 'candidate-evidence/notice.txt').write_text('matching notice')
        self.dependency = self.root / 'dependency'
        self.dependency.mkdir()
        self.receipt = self.dependency / 'DEPENDENCY_BUILD.json'
        self.receipt.write_text('dependency fixture')
        (self.root / 'video-helper').mkdir()
        (self.root / 'video-helper/.windows-lgpl-dependency-selection').write_text(str(self.dependency))
        self.helper_record = {'status': 'PASS', 'scope': 'sealed-helper-candidate-only',
                              'source': copy.deepcopy(IDENTITY),
                              'closure': {'dependencyReceiptSha256': w.sha256(self.receipt)},
                              'artifactHashes': w.file_hashes(self.helper)}
        (self.helper / w.HELPER_RECEIPT).write_text(json.dumps(self.helper_record))
        self.app_record = {'status': 'PASS', 'source': copy.deepcopy(IDENTITY),
                           'artifactHashes': {'DonutStudio.exe': w.sha256(self.stage / 'DonutStudio.exe')}}
        (self.stage / w.APP_RECEIPT).write_text(json.dumps(self.app_record))
        (self.stage / 'RELEASE_BUILD.json').write_text(json.dumps({
            'sourceCommit': COMMIT, 'buildProfile': 'shipping', 'debugSymbolsEmbedded': False}))
        self.source_patch = patch.object(w, 'source_identity', return_value=copy.deepcopy(IDENTITY))
        self.root_patch = patch.object(w, 'ROOT', self.root)
        self.source_patch.start()
        self.root_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def seal(self):
        with patch.object(w, 'audit_closure', return_value={'status': 'PASS', 'peHashes': {}}):
            w.seal_package(self.stage, COMMIT)

    def test_matching_candidate_is_admitted_with_notice_evidence(self):
        self.seal()
        record = w.admit_package(self.stage, COMMIT)
        self.assertIn('video-helper/candidate-evidence/notice.txt', record['artifactHashes'])
        self.assertFalse(any('first-party-source.tar' in name for name in record['artifactHashes']))

    def test_changed_helper_and_notice_are_refused(self):
        for relative in ('arbit-video-helper.exe', 'candidate-evidence/notice.txt'):
            path = self.helper / relative
            original = path.read_bytes()
            path.write_bytes(original + b' changed')
            with self.assertRaisesRegex(ValueError, 'helper bytes or evidence changed'):
                w.validate_helper(self.helper, COMMIT)
            path.write_bytes(original)

    def test_missing_helper_receipt_is_refused(self):
        (self.helper / w.HELPER_RECEIPT).unlink()
        with self.assertRaises(FileNotFoundError):
            w.validate_helper(self.helper, COMMIT)

    def test_wrong_source_and_changed_app_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'source identity mismatch'):
            w.validate_helper(self.helper, '2' * 40)
        (self.stage / 'DonutStudio.exe').write_bytes(b'MZ changed app')
        with self.assertRaisesRegex(ValueError, 'app artifact changed'):
            w.validate_app(self.stage, COMMIT)

    def test_unadmitted_extra_pe_is_refused_before_sealing(self):
        (self.stage / 'unexpected.exe').write_bytes(b'MZ unknown')
        with self.assertRaisesRegex(ValueError, 'Unadmitted candidate package files'):
            self.seal()

    def test_changed_or_removed_packaged_file_is_refused(self):
        self.seal()
        (self.stage / 'RELEASE_BUILD.json').unlink()
        with self.assertRaisesRegex(ValueError, 'package files changed'):
            w.admit_package(self.stage, COMMIT)

    def test_unknown_import_and_disguised_pe_are_refused(self):
        dependency = {'profile': 'fixture', 'dllProvenance': {}, 'executableProvenance': {},
                      'limitations': []}
        with patch.object(w, 'validate_receipt', return_value=dependency), \
                patch.object(w, 'imports', return_value=['unknown-runtime.dll']):
            with self.assertRaisesRegex(ValueError, 'Unresolved/unproven PE import'):
                w.audit_closure(self.helper, self.receipt)
        (self.helper / 'hidden.bin').write_bytes(b'MZ disguised PE')
        with patch.object(w, 'validate_receipt', return_value=dependency), \
                patch.object(w, 'imports', return_value=['KERNEL32.dll']):
            with self.assertRaisesRegex(ValueError, 'Unproven bundled product PE'):
                w.audit_closure(self.helper, self.receipt)

    def test_app_receipt_does_not_allow_changed_pe_or_unknown_imports(self):
        dependency = {'profile': 'fixture', 'dllProvenance': {}, 'executableProvenance': {},
                      'limitations': []}
        with patch.object(w, 'validate_receipt', return_value=dependency), \
                patch.object(w, 'imports', return_value=['not-a-system.dll']):
            with self.assertRaisesRegex(ValueError, 'Unresolved/unproven PE import'):
                w.audit_closure(self.stage, self.receipt, app_artifacts=self.app_record['artifactHashes'])


    def test_windows_symbol_collection_preserves_the_sealed_app_bytes(self):
        spec = importlib.util.spec_from_file_location('shipping_symbols', ROOT / 'scripts/prepare-release-symbols.py')
        symbols = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(symbols)
        app = self.stage / 'DonutStudio.exe'
        app.write_bytes(b'MZ fixture RSDS' + b'0' * 20 + b'DonutStudio.pdb\0')
        self.app_record['artifactHashes']['DonutStudio.exe'] = w.sha256(app)
        (self.stage / w.APP_RECEIPT).write_text(json.dumps(self.app_record))
        build, private = self.root / 'msvc', self.root / 'private-symbols'
        build.mkdir()
        private.mkdir()
        (build / 'DonutStudio.pdb').write_bytes(b'private PDB fixture')
        before = w.file_hashes(self.stage)
        symbols.prepare_windows(self.stage, build, private, None)
        self.assertEqual(w.file_hashes(self.stage), before)
        self.assertEqual(w.validate_app(self.stage, COMMIT), self.app_record)


class KvazaarGeneratedBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.src = self.root / 'source'
        (self.src / 'src').mkdir(parents=True)
        authored = {
            'CMakeLists.txt': 'project(kvazaar VERSION 2.3.2)\nconfigure_file(src/version.h.in src/version.h @ONLY)\n'
                              'configure_file(src/kvazaarCMake.pc.in src/kvazaar.pc.temp @ONLY)\n',
            'src/kvazaar.h': '/* immutable authored header */\n',
            'src/version.h.in': '#define KVZ_VERSION @PROJECT_VERSION@\n'
                                '#define KVZ_COMPILER_STRING "@KVZ_COMPILER_STRING@"\n'
                                '#define KVZ_COMPILE_DATE "@CMAKE_BUILD_DATE@"\n',
            'src/kvazaarCMake.pc.in': 'prefix=@DEFERRED@CMAKE_INSTALL_PREFIX@DEFERRED@\n'
                                     'exec_prefix=${prefix}\nlibdir=@KVAZAAR_PC_LIBDIR@\n'
                                     'incdir=@KVAZAAR_PC_INCDIR@\nVersion: @PROJECT_VERSION@\n'
                                     'Description: @CMAKE_PROJECT_DESCRIPTION@\n'
                                     'URL: @CMAKE_PROJECT_HOMEPAGE_URL@\nLibs.private: @LIBS@\n'}
        for name, text in authored.items():
            (self.src / name).write_text(text)
        self.archive = self.root / 'kvazaar.tar.gz'
        with tarfile.open(self.archive, 'w:gz') as bundle:
            for name in authored:
                bundle.add(self.src / name, arcname='kvazaar-2.3.2/' + name)
        self.inputs = {'kvazaar': {'version': '2.3.2', 'sha256': w.sha256(self.archive)},
                       'compiler': {'version': '13.3.0'}, 'configureEpoch': 1767225600,
                       'kvazaarSourceHashes': w.file_hashes(self.src)}
        self.cache = {'CMAKE_INSTALL_LIBDIR': 'lib', 'CMAKE_INSTALL_INCLUDEDIR': 'include',
                      'CMAKE_INSTALL_PREFIX': '/candidate/install'}
        # Independent expected output from the pinned upstream @ONLY operations.
        self.version = '#define KVZ_VERSION 2.3.2\n#define KVZ_COMPILER_STRING "GNU 13.3.0"\n'
        self.version += '#define KVZ_COMPILE_DATE "2026-01-01"\n'
        self.pc = 'prefix=@CMAKE_INSTALL_PREFIX@\nexec_prefix=${prefix}\n'
        self.pc += 'libdir=${exec_prefix}/lib\nincdir=${prefix}/include\nVersion: 2.3.2\n'
        self.pc += 'Description: \nURL: \nLibs.private: \n'
        (self.src / 'src/version.h').write_text(self.version)
        (self.src / 'src/kvazaar.pc.temp').write_text(self.pc)
        self.addCleanup(self.temp.cleanup)

    def binding(self):
        return w.kvazaar_source_binding(self.src, self.inputs, self.archive, self.cache)

    def test_exact_configure_outputs_and_optional_install_output_are_bound(self):
        record = self.binding()
        self.assertEqual(set(record['generated']), set(w.KVAZAAR_CONFIGURED_FILES))
        self.assertEqual(record['generatorSha256'], self.inputs['kvazaarSourceHashes']['CMakeLists.txt'])
        (self.src / 'src/kvazaar.pc').write_text(self.pc.replace('@CMAKE_INSTALL_PREFIX@', '/candidate/install'))
        self.assertEqual(self.binding()['generated']['src/kvazaar.pc']['templateSha256'],
                         w.sha256(self.src / 'src/kvazaar.pc.temp'))

    def test_missing_or_forged_generated_outputs_are_refused(self):
        path = self.src / 'src/version.h'
        for text in (self.version.replace('2026-01-01', '2026-01-02'),
                     self.version.replace('13.3.0', '99.0'), self.version + 'arbitrary source'):
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'generated output differs'):
                self.binding()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'generated output differs'):
            self.binding()

    def test_authored_changes_additions_and_removals_are_refused(self):
        header = self.src / 'src/kvazaar.h'
        original = header.read_bytes()
        header.write_bytes(b'changed header')
        with self.assertRaisesRegex(ValueError, 'authored source changed'):
            self.binding()
        header.write_bytes(original)
        extra = self.src / 'src/unadmitted.c'
        extra.write_text('/* arbitrary addition */')
        with self.assertRaisesRegex(ValueError, 'authored source changed'):
            self.binding()
        extra.unlink()
        header.unlink()
        with self.assertRaisesRegex(ValueError, 'authored source changed'):
            self.binding()

    def test_resealing_authored_hashes_cannot_override_the_pinned_archive(self):
        (self.src / 'src/kvazaar.h').write_text('/* changed */')
        self.inputs['kvazaarSourceHashes']['src/kvazaar.h'] = w.sha256(self.src / 'src/kvazaar.h')
        with self.assertRaisesRegex(ValueError, 'authored source differs from pinned archive'):
            self.binding()

    def test_archive_and_install_output_changes_are_refused(self):
        (self.src / 'src/kvazaar.pc').write_text('prefix=/unadmitted\n')
        with self.assertRaisesRegex(ValueError, 'generated output differs'):
            self.binding()
        self.archive.write_bytes(self.archive.read_bytes() + b'changed archive')
        with self.assertRaisesRegex(ValueError, 'source archive changed'):
            self.binding()


class Evidence(unittest.TestCase):
    def test_dependency_sources_and_notices_match_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'tiny/LICENSE'
            source.parent.mkdir()
            source.write_text('dependency notice')
            archive = root / 'evidence/source-archives/tiny.tar.gz'
            archive.parent.mkdir(parents=True)
            with tarfile.open(archive, 'w:gz') as bundle:
                bundle.add(source, arcname='tiny/LICENSE')
            profile = {'sources': [{'name': 'tiny', 'archive': archive.name,
                                    'sha256': w.sha256(archive), 'licenseFiles': ['LICENSE']}]}
            receipt = {'sources': profile['sources'], 'artifactHashes': {
                'evidence/source-archives/tiny.tar.gz': w.sha256(archive),
                'evidence/notices/tiny/LICENSE': w.sha256(source)}}
            with patch.object(w, 'validate_receipt', return_value=receipt), \
                    patch.object(w, 'verify_policy', return_value=profile):
                self.assertEqual(w.validate_dependency_sources(root / 'DEPENDENCY_BUILD.json'), receipt)
                receipt['artifactHashes']['evidence/notices/tiny/LICENSE'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'notice differs from source archive'):
                    w.validate_dependency_sources(root / 'DEPENDENCY_BUILD.json')

    def test_fetched_and_vendored_sources_and_notices_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / 'build'
            for source in (build / '_deps/ncnn-src', root / 'video-helper/third_party/quickjs',
                           root / 'video-helper/src/rife_net'):
                source.mkdir(parents=True)
                (source / 'LICENSE').write_text('fixture source notice')
                (source / 'source.c').write_text('/* matching fixture source */')
            patch_path = root / 'video-helper/patches/ncnn_enable_external_memory_fd.cmake'
            patch_path.parent.mkdir(parents=True)
            patch_path.write_bytes((ROOT / 'video-helper/patches/ncnn_enable_external_memory_fd.cmake').read_bytes())
            evidence = root / 'evidence'
            evidence.mkdir()
            with patch.object(w, 'ROOT', root):
                rows = w.retain_helper_dependencies(build, evidence)
            self.assertEqual(set(rows), {'ncnn-src', 'quickjs', 'rife_net'})
            for name, row in rows.items():
                archive = evidence / row['archive']
                self.assertEqual(w.sha256(archive), row['sha256'])
                self.assertTrue(all((evidence / notice).is_file() for notice in row['notices']))
                with tarfile.open(archive) as bundle:
                    self.assertEqual(bundle.extractfile(name + '/source.c').read(),
                                     b'/* matching fixture source */')
            self.assertEqual(rows['ncnn-src']['patches'], {
                'helper-patches/ncnn_enable_external_memory_fd.cmake': w.sha256(patch_path)})

    def test_missing_ncnn_patch_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'build/_deps/ncnn-src'
            source.mkdir(parents=True)
            (source / 'LICENSE').write_text('fixture source notice')
            (root / 'video-helper/third_party').mkdir(parents=True)
            evidence = root / 'evidence'
            evidence.mkdir()
            with patch.object(w, 'ROOT', root):
                with self.assertRaises(FileNotFoundError):
                    w.retain_helper_dependencies(root / 'build', evidence)

    def test_private_source_evidence_never_enters_helper_package(self):
        source = (ROOT / 'scripts/video-helper-dependencies/windows.py').read_text()
        self.assertIn("first_party = private / 'windows-helper-first-party-source.tar.gz'", source)
        self.assertNotIn('shutil.copy2(first_party', source)
        self.assertIn('shutil.copytree(receipt_path.parent / \'evidence\'', source)
        self.assertIn("'helper-dependency-sources'", source)
        self.assertIn("'detachedSymbolsSha256': sha256(debug)", source)


if __name__ == '__main__':
    unittest.main()
