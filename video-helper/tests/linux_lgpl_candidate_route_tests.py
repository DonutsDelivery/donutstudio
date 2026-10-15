#!/usr/bin/env python3
"""Linux candidate route and rejection fixtures; no compiler or ELF tools run."""
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = ROOT / 'scripts/video-helper-dependencies'
sys.path.insert(0, str(DEPENDENCIES))
SPEC = importlib.util.spec_from_file_location('linux_candidate', DEPENDENCIES / 'linux.py')
linux = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linux)
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    'binding_fixtures', Path(__file__).with_name('strict_dependency_prefix_binding_tests.py'))
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


class CandidateFixture:
    def __init__(self, directory):
        base = fixtures.PrefixFixture(directory)
        self.root = base.root
        source = base.stage / 'source/kvazaar-2.3.2'
        (source / 'src').mkdir(parents=True)
        (source / 'src/kvazaar.h').write_text('fixture kvazaar API')
        (source / 'CMakeLists.txt').write_text('fixture upstream cmake')
        (source / 'src/version.h.in').write_text('#define VERSION @PROJECT_VERSION@\n'
                                                     '#define COMPILER "@KVZ_COMPILER_STRING@"\n'
                                                     '#define BUILD_DATE "@CMAKE_BUILD_DATE@"\n')
        (source / 'src/kvazaarCMake.pc.in').write_text('Version: @PROJECT_VERSION@\n')
        sources = []
        archives = base.stage.parent / 'archives'
        archives.mkdir()
        for name in ('ffmpeg', 'kvazaar'):
            archive = archives / (name + '.tar.gz')
            if name == 'kvazaar':
                with tarfile.open(archive, 'w:gz') as bundle:
                    bundle.add(source, arcname=source.name)
            else:
                archive.write_bytes(('fixture archive ' + name).encode())
            row = {'name': name, 'version': '2.3.2' if name == 'kvazaar' else '6.1.6',
                   'archive': archive.name, 'sha256': linux.sha256(archive)}
            if name == 'kvazaar':
                row['headerSha256'] = linux.sha256(source / 'src/kvazaar.h')
            sources.append(row)
        profile_path = self.root / linux.strict.PROFILES['Linux'][1]
        base.profile['sources'] = sources
        profile_path.write_text(json.dumps(base.profile))
        auditor = self.root / 'scripts/video-helper-dependencies/linux.py'
        auditor.write_bytes((DEPENDENCIES / 'linux.py').read_bytes())
        inputs = {p: linux.sha256(self.root / p) for p in base.receipt['inputHashes']}
        inputs['scripts/video-helper-dependencies/linux.py'] = linux.sha256(auditor)
        identity = hashlib.sha256(json.dumps({'recipe': inputs, 'installedPackages': 'fixture packages\n',
                                             'image': 'fixture-image'}, sort_keys=True).encode()).hexdigest()
        self.stage = base.stage.with_name(identity)
        base.stage.rename(self.stage)
        self.prefix = self.stage / 'prefix'
        self.receipt = dict(base.receipt, sources=sources, prefix=str(self.prefix), inputHashes=inputs,
                            sourcePaths={'kvazaar': 'source/kvazaar-2.3.2'},
                            archiveHashes={s['archive']: s['sha256'] for s in sources})
        self.receipt['artifactHashes'] = {str(p.relative_to(self.stage)): linux.sha256(p)
                                          for p in self.stage.rglob('*') if p.is_file()
                                          and p.name != 'DEPENDENCY_BUILD.json'}
        self.selection = self.stage.parent / '.selection'
        self.save()

    def save(self):
        (self.stage / 'DEPENDENCY_BUILD.json').write_text(json.dumps(self.receipt))


class CandidateRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.fixture = CandidateFixture(self.directory)
        self.root_patch = patch.object(linux, 'ROOT', self.fixture.root)
        self.root_patch.start()
        self.env_patch = patch.dict(os.environ, {'DONUTSTUDIO_DEPENDENCY_IMAGE_ID': 'fixture-image',
                                                  'SOURCE_DATE_EPOCH': '1760000000', 'TZ': 'UTC'})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.root_patch.stop()
        self.temporary.cleanup()

    def select(self):
        linux.write_selection(self.fixture.selection, self.fixture.stage)
        return linux.resolve_selection(self.fixture.selection)

    def test_pass_handoff_binds_exact_prefix_identity_and_kvazaar_source(self):
        prefix, receipt = self.select()
        self.assertEqual(prefix, self.fixture.prefix)
        self.assertEqual(linux.kvazaar_source(prefix, receipt), self.fixture.stage / 'source/kvazaar-2.3.2')
        selection = json.loads(self.fixture.selection.read_text())
        self.assertEqual(selection['identity'], self.fixture.stage.name)
        self.assertEqual(selection['dependencyReceiptSha256'], linux.sha256(self.fixture.stage / 'DEPENDENCY_BUILD.json'))

    def test_missing_handoff_never_guesses_an_existing_stage(self):
        with self.assertRaises(OSError):
            linux.resolve_selection(self.fixture.selection)

    def test_failed_stage_cannot_publish_a_pass_handoff(self):
        self.select()
        self.fixture.receipt['status'] = 'FAIL'
        self.fixture.save()
        with self.assertRaises(ValueError):
            linux.write_selection(self.fixture.selection, self.fixture.stage)
        self.assertFalse(self.fixture.selection.exists())

    def test_selection_status_identity_path_and_receipt_drift(self):
        self.select()
        original = json.loads(self.fixture.selection.read_text())
        for key, value in [('status', 'FAIL'), ('identity', '0' * 64), ('prefix', '/usr'),
                           ('dependencyReceiptSha256', '0' * 64)]:
            self.fixture.selection.write_text(json.dumps(dict(original, **{key: value})))
            with self.assertRaises((ValueError, OSError)):
                linux.resolve_selection(self.fixture.selection)
        self.fixture.selection.write_text(json.dumps(original))
        with patch.dict(os.environ, {'DONUTSTUDIO_DEPENDENCY_IMAGE_ID': 'different-image'}):
            with self.assertRaisesRegex(ValueError, 'container image'):
                linux.resolve_selection(self.fixture.selection)

    def test_recipe_prefix_evidence_source_and_archive_hash_drift(self):
        self.select()
        paths = [self.fixture.root / 'scripts/build-linux-local.sh',
                 self.fixture.prefix / 'lib/libavcodec.so.60',
                 self.fixture.stage / 'evidence/installed-packages.txt',
                 self.fixture.stage / 'source/kvazaar-2.3.2/src/kvazaar.h',
                 self.fixture.stage.parent / 'archives/kvazaar.tar.gz']
        for path in paths:
            before = path.read_bytes()
            path.write_bytes(before + b'drift')
            with self.assertRaises(ValueError):
                linux.resolve_selection(self.fixture.selection)
            path.write_bytes(before)

    def test_added_retained_source_or_evidence_is_rejected(self):
        self.select()
        for relative in ('source/kvazaar-2.3.2/unrecorded.c', 'evidence/unrecorded.json'):
            path = self.fixture.stage / relative
            path.write_text('unrecorded')
            with self.assertRaisesRegex(ValueError, 'Unrecorded dependency artifact'):
                linux.resolve_selection(self.fixture.selection)
            path.unlink()

    def test_unknown_denied_wrong_origin_and_hash_drift_codec_rejected(self):
        prefix, receipt = self.select()
        codec = prefix / 'lib/libavcodec.so.60'
        for name, path in [('libunknown.so.1', codec), ('libx265.so.199', codec),
                           ('libavcodec.so.60', self.directory / 'system-codec.so')]:
            with self.assertRaises((ValueError, OSError)):
                linux.dependency_origin(name, path, prefix, receipt)
        codec.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash drift'):
            linux.dependency_origin(codec.name, codec, prefix, receipt)
        with self.assertRaisesRegex(ValueError, 'system ELF origin'):
            linux.dependency_origin('libvulkan.so.1', codec, prefix, receipt)

    def test_recursive_dt_needed_and_missing_transitive_edge(self):
        prefix, receipt = self.select()
        helper = self.directory / 'arbit-video-helper'
        helper.write_bytes(b'fixture helper')
        libs = {name: prefix / ('lib/lib' + name + '.so.60') for name in linux.audit.candidate.CORE}
        names = {path: path.name for path in libs.values()}
        metadata = {helper: {'soname': None, 'needed': list(names.values())},
                    **{path: {'soname': name, 'needed': []} for path, name in names.items()}}
        resolutions = {helper: {path.name: path for path in libs.values()}, **{p: {} for p in libs.values()}}
        with patch.object(linux, 'elf_metadata', side_effect=lambda p: metadata[p]), \
                patch.object(linux, 'resolve_needed', side_effect=lambda p, search: resolutions[p]):
            rows = linux.audit_closure(helper, prefix, receipt)
            self.assertEqual(len(rows), 7)
            metadata[libs['avcodec']]['needed'] = ['libunknown.so.1']
            with self.assertRaisesRegex(ValueError, 'Unresolved ELF DT_NEEDED'):
                linux.audit_closure(helper, prefix, receipt)
            resolutions[libs['avcodec']]['libunknown.so.1'] = libs['avutil']
            with self.assertRaisesRegex(ValueError, 'Unknown ELF dependency'):
                linux.audit_closure(helper, prefix, receipt)

    def prepare_adapter(self):
        prefix, receipt = self.select()
        build = self.directory / 'helper-build-cache' / prefix.parent.name
        source = linux.prepare_kvazaar_copy(prefix, receipt, build)
        compiler = build / 'CMakeFiles/3.28.3/CMakeCCompiler.cmake'
        compiler.parent.mkdir(parents=True)
        compiler.write_text('set(CMAKE_C_COMPILER_ID "GNU")\nset(CMAKE_C_COMPILER_VERSION "13.3.0")\n')
        (build / 'CMakeCache.txt').write_text('CMAKE_INSTALL_PREFIX:PATH=/usr/local\n'
                                            'CMAKE_INSTALL_LIBDIR:PATH=lib\n'
                                            'CMAKE_INSTALL_INCLUDEDIR:PATH=include\n')
        generation = linux.generated_configuration(build, next(s for s in receipt['sources'] if s['name'] == 'kvazaar'))
        for output, template in linux.KVAZAAR_GENERATED_OUTPUTS.items():
            (source / output).write_bytes(linux.configured_template(source / template, generation['substitutions']))
        return prefix, receipt, build, source

    def test_generated_adapter_outputs_are_separate_from_exact_admitted_source(self):
        prefix, receipt, build, source = self.prepare_adapter()
        self.assertNotEqual(source, linux.kvazaar_source(prefix, receipt))
        first = linux.kvazaar_source_binding(prefix, receipt, build)
        os.environ['SOURCE_DATE_EPOCH'] = str(int(os.environ['SOURCE_DATE_EPOCH']) + 86400)
        generation = linux.generated_configuration(build, first['source'])
        (source / 'src/version.h').write_bytes(linux.configured_template(
            source / 'src/version.h.in', generation['substitutions']))
        second = linux.kvazaar_source_binding(prefix, receipt, build)
        self.assertEqual(first['authoredHashes'], second['authoredHashes'])
        self.assertNotEqual(first['generatedOutputs'], second['generatedOutputs'])
        self.assertEqual(second['dependencyIdentity'], prefix.parent.name)
        self.assertEqual(second['dependencyReceiptSha256'], linux.sha256(prefix.parent / 'DEPENDENCY_BUILD.json'))
        self.assertEqual(second['generator']['sha256'], second['authoredHashes']['CMakeLists.txt'])
        for output, template in linux.KVAZAAR_GENERATED_OUTPUTS.items():
            row = second['generatedOutputs'][output]
            self.assertEqual(row['template'], template)
            self.assertEqual(row['templateSha256'], second['authoredHashes'][template])
            self.assertEqual(row['sha256'], linux.sha256(source / output))
        # Generated files were written only into the adapter copy. Exact source
        # and artifact re-admission of the original dependency stage still passes.
        linux.resolve_selection(self.fixture.selection)
        self.assertFalse((linux.kvazaar_source(prefix, receipt) / 'src/version.h').exists())

    def test_adapter_authored_template_added_file_and_archive_drift_reject(self):
        prefix, receipt, build, source = self.prepare_adapter()
        for relative in ('src/kvazaar.h', 'src/version.h.in', 'CMakeLists.txt'):
            path = source / relative
            before = path.read_bytes()
            path.write_bytes(before + b'drift')
            with self.assertRaisesRegex(ValueError, 'authored source drift'):
                linux.kvazaar_source_binding(prefix, receipt, build)
            with self.assertRaisesRegex(ValueError, 'authored source drift'):
                linux.prepare_kvazaar_copy(prefix, receipt, build)
            path.write_bytes(before)
        added = source / 'src/unrecorded.c'
        added.write_text('new authored source')
        with self.assertRaisesRegex(ValueError, 'authored source drift'):
            linux.kvazaar_source_binding(prefix, receipt, build)
        added.unlink()
        removed = source / 'src/kvazaar.h'
        before = removed.read_bytes()
        removed.unlink()
        with self.assertRaisesRegex(ValueError, 'authored source drift'):
            linux.kvazaar_source_binding(prefix, receipt, build)
        removed.write_bytes(before)
        archive = prefix.parent.parent / 'archives/kvazaar.tar.gz'
        archive.write_bytes(archive.read_bytes() + b'drift')
        with self.assertRaisesRegex(ValueError, 'archive differs'):
            linux.kvazaar_source_binding(prefix, receipt, build)

    def test_arbitrary_generated_bytes_or_unknown_template_variable_reject(self):
        prefix, receipt, build, source = self.prepare_adapter()
        for output in linux.KVAZAAR_GENERATED_OUTPUTS:
            path = source / output
            before = path.read_bytes()
            path.write_bytes(before + b'unapproved generated bytes')
            with self.assertRaisesRegex(ValueError, 'differs from pinned template'):
                linux.kvazaar_source_binding(prefix, receipt, build)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing/escaped'):
                linux.kvazaar_source_binding(prefix, receipt, build)
            path.write_bytes(before)
        unknown = self.directory / 'unknown-template.in'
        unknown.write_text('@UNKNOWN_VARIABLE@')
        with self.assertRaisesRegex(ValueError, 'Unknown private adapter'):
            linux.configured_template(unknown, {})

    def test_prepared_source_baseline_cannot_be_rehashed_after_authored_drift(self):
        prefix, receipt, build, source = self.prepare_adapter()
        prepared = build / 'kvazaar10-source-inputs.json'
        baseline = json.loads(prepared.read_text())
        authored = source / 'src/kvazaar.h'
        authored.write_text('changed API')
        baseline['authoredHashes']['src/kvazaar.h'] = linux.sha256(authored)
        prepared.write_text(json.dumps(baseline))
        with self.assertRaisesRegex(ValueError, 'source association drift'):
            linux.prepare_kvazaar_copy(prefix, receipt, build)
        with self.assertRaisesRegex(ValueError, 'source association drift'):
            linux.kvazaar_source_binding(prefix, receipt, build)

    def test_dependency_receipt_refresh_keeps_authored_baseline_exact(self):
        prefix, receipt, build, source = self.prepare_adapter()
        first = linux.kvazaar_source_binding(prefix, receipt, build)
        self.fixture.receipt['builtAt'] = 'new dependency retry timestamp'
        self.fixture.save()
        prefix, receipt = self.select()
        linux.prepare_kvazaar_copy(prefix, receipt, build)
        second = linux.kvazaar_source_binding(prefix, receipt, build)
        self.assertEqual(first['authoredHashes'], second['authoredHashes'])
        self.assertNotEqual(first['dependencyReceiptSha256'], second['dependencyReceiptSha256'])
        (source / 'src/version.h.in').write_text('altered template')
        with self.assertRaisesRegex(ValueError, 'authored source drift'):
            linux.prepare_kvazaar_copy(prefix, receipt, build)

    def test_failed_candidate_retry_removes_old_pass_receipts(self):
        output = self.directory / 'candidate'
        output.mkdir()
        for name in ('CANDIDATE_BUILD.json', 'LINUX_DEPENDENCY_PROVENANCE.json'):
            (output / name).write_text('{"status":"PASS"}')
        with self.assertRaises(OSError):
            linux.build_candidate(self.fixture.selection, self.directory / 'build', output)
        self.assertFalse((output / 'CANDIDATE_BUILD.json').exists())
        self.assertFalse((output / 'LINUX_DEPENDENCY_PROVENANCE.json').exists())

    def test_retained_source_links_are_materialized_and_escape_rejected(self):
        root = self.directory / 'sources'
        root.mkdir()
        target = root / 'original.c'
        target.write_text('source')
        link = root / 'alias.c'
        link.symlink_to(target.name)
        linux.materialize_source_links(root)
        self.assertFalse(link.is_symlink())
        self.assertEqual(link.read_text(), target.read_text())
        outside = self.directory / 'outside.c'
        outside.write_text('foreign')
        link.unlink()
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes prefix'):
            linux.materialize_source_links(root)

    def test_copy_rejects_changed_artifact(self):
        source = self.directory / 'artifact'
        source.write_bytes(b'first')
        expected = linux.sha256(source)
        source.write_bytes(b'second')
        with self.assertRaisesRegex(ValueError, 'changed before staging'):
            linux.copy_verified(source, self.directory / 'copied', expected)


class RouteSourceContract(unittest.TestCase):
    def test_default_routes_and_caches_remain_unchanged(self):
        local = (ROOT / 'scripts/build-linux-local.sh').read_text()
        docker = (ROOT / 'scripts/build-linux-docker-videohelper.sh').read_text()
        default_helper = 'echo "==> [video-helper] building in $VH_IMAGE ..."\nrun_in "$VH_IMAGE" scripts/build-linux-docker-videohelper.sh'
        selected_helper = local[local.index('if [ -n "${DONUTSTUDIO_LINUX_VIDEO_HELPER_CANDIDATE:-}" ]; then',
                                           local.index('build_image "$VH_IMAGE"')):local.index('echo "==> [plugin + package]')]
        restored = local.replace(selected_helper.rstrip(), default_helper)
        self.assertEqual(hashlib.sha256(restored[restored.index('build_image "$VH_IMAGE"'):].encode()).hexdigest(),
                         'df6cdb40de4b9a7cca80b89c92f65c93168e93a624a53e8d1ba819c47a6d3523')
        self.assertEqual(hashlib.sha256(docker[docker.index('VH_BUILD_DIR="$ROOT/video-helper/build-docker"'):].encode()).hexdigest(),
                         'c3a730e458491bb4442d4da2e4bb46ac1c0e4e0ee9cb585bfc72b83631cc38d2')
        for value in ('video-helper-build', 'video-helper-ccache'):
            self.assertIn(value, local)

    def test_candidate_stops_before_plugin_and_package_with_separate_identity_cache(self):
        local = (ROOT / 'scripts/build-linux-local.sh').read_text()
        candidate = local.split('if [ "${LGPL_CANDIDATE:-0}" = 1 ]; then', 1)[1].split('build_image "$VH_IMAGE"', 1)[0]
        self.assertIn('--lgpl-dependencies', candidate)
        self.assertIn('exit 0', candidate)
        self.assertNotIn('scripts/build-linux-docker-build.sh', candidate)
        self.assertNotIn('installers/make-appimage.sh', candidate)
        docker = (ROOT / 'scripts/build-linux-docker-videohelper.sh').read_text()
        candidate = docker.split('if [ "${1:-}" = --lgpl-dependencies', 1)[1].split('if [ "$#" != 0 ]', 1)[0]
        self.assertIn('/work/.video-helper-lgpl-build/$IDENTITY', candidate)
        self.assertIn('/work/.video-helper-lgpl-ccache/$IDENTITY', candidate)
        self.assertIn('-DARBIT_VIDEO_DEPENDENCY_PREFIX="$PREFIX"', candidate)
        self.assertIn('-DARBIT_REQUIRE_VIEWPORT=ON', candidate)
        self.assertIn('-DARBIT_BUILD_KVAZAAR10_ADAPTER=ON', candidate)
        self.assertIn('-DFETCHCONTENT_SOURCE_DIR_ARBIT_KVAZAAR10_SOURCE="$KVAZAAR_SOURCE"', candidate)
        self.assertLess(candidate.index('--print-selection'), candidate.index('cmake -S'))
        self.assertIn('/lgpl-candidate/linux/video-helper', candidate)
        self.assertIn('-e DONUTSTUDIO_LINUX_VIDEO_HELPER_CANDIDATE', local)
        self.assertIn('/work/${HELPER_CANDIDATE_HOST#', local)

    def test_candidate_job_guard_rejects_unbounded_values_and_defaults_to_two(self):
        docker = (ROOT / 'scripts/build-linux-docker-videohelper.sh').read_text()
        guard = '  CANDIDATE_JOBS=' + docker.split('  CANDIDATE_JOBS=', 1)[1].split(
            '  python3 scripts/video-helper-dependencies/build-linux.py', 1)[0]
        command = ['bash', '-c', guard + '\nprintf "%s" "$ARBIT_BUILD_JOBS"']
        env = os.environ.copy()
        env.pop('ARBIT_BUILD_JOBS', None)
        self.assertEqual(subprocess.check_output(command, env=env, text=True), '2')
        for jobs in ('1', '2'):
            env['ARBIT_BUILD_JOBS'] = jobs
            self.assertEqual(subprocess.check_output(command, env=env, text=True), jobs)
        for jobs in ('0', '-1', '3', '999', 'automatic', '2;true'):
            env['ARBIT_BUILD_JOBS'] = jobs
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        candidate = docker.split('if [ "${1:-}" = --lgpl-dependencies', 1)[1].split('if [ "$#" != 0 ]', 1)[0]
        ninja_lines = [line.strip() for line in candidate.splitlines() if line.strip().startswith('ninja ')]
        self.assertEqual(ninja_lines, [
            'ninja -j "$CANDIDATE_JOBS" -C "$VH_BUILD_DIR" arbit-video-helper',
            'ninja -j "$CANDIDATE_JOBS" -C "$VH_BUILD_DIR" arbit-kvazaar10-adapter-tests'])
        self.assertIn("-R '^kvazaar10_adapter$'", candidate)
        self.assertNotIn('-LE', candidate)

    def test_candidate_exports_compile_database_and_helper_only_link_map(self):
        docker = (ROOT / 'scripts/build-linux-docker-videohelper.sh').read_text()
        candidate = docker.split('if [ "${1:-}" = --lgpl-dependencies', 1)[1].split('if [ "$#" != 0 ]', 1)[0]
        self.assertIn('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', candidate)
        self.assertIn('-DARBIT_HELPER_LINK_MAP_FILE="$VH_BUILD_DIR/arbit-video-helper.link.map"', candidate)
        # The adapter test executable must not overwrite the helper map, and
        # existing cached or environment linker flags stay under CMake's control.
        self.assertNotIn('-DCMAKE_EXE_LINKER_FLAGS=', candidate)
        default = docker[docker.index('VH_BUILD_DIR="$ROOT/video-helper/build-docker"'):]
        self.assertNotIn('CMAKE_EXPORT_COMPILE_COMMANDS', default)
        self.assertNotIn('ARBIT_HELPER_LINK_MAP_FILE', default)

    def test_private_evidence_and_shipping_order(self):
        auditor = (DEPENDENCIES / 'linux.py').read_text()
        self.assertIn("private_evidence / 'helper-source.tar'", auditor)
        self.assertIn("private_evidence / 'video-helper.symbols.tar.gz'", auditor)
        self.assertLess(auditor.index("'scripts/prepare-release-symbols.py'"),
                        auditor.index("(output / 'CANDIDATE_BUILD.json').write_text"))
        self.assertIn("row['sourceSha256'] = row['sha256']", auditor)
        self.assertIn("materialize_source_links(output / 'dependencies/source')", auditor)
        self.assertLess(auditor.index('loader_tool = seal_loader_paths(runtime_paths,'),
                        auditor.index('prepare_candidate_symbols(output, symbols, shipping_path'))
        self.assertIn('validate_installed_closure(output, helper.name, rows)', auditor)
        self.assertNotIn('resolve_needed(current, [output])', auditor)


if __name__ == '__main__':
    unittest.main()
