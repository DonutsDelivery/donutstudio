#!/usr/bin/env python3
"""Prepared Windows private receipt/format fixtures; no build or app invocation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('windows_private', ROOT / 'scripts/video-helper-dependencies/windows.py')
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
COMMIT = '1' * 40
SOURCE = {'sourceCommit': COMMIT, 'helperTreeSha256': '2' * 64,
          'compiledInputs': {'video-helper/src/main.cpp': '7' * 64},
          'firstPartyLicense': {'path': 'video-helper/FIRST_PARTY_LICENSE.txt', 'sha256': hashlib.sha256(b'owner declaration').hexdigest()}}


class FinalPrivateReceipt(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stage = self.root / 'windows'
        self.helper = self.stage / 'video-helper'
        self.helper.mkdir(parents=True)
        self.private = self.root / 'private-candidate-evidence'
        self.private.mkdir()
        (self.private / 'windows-helper-first-party-source.tar.gz').write_bytes(b'private source archive fixture')
        (self.private / 'retained-object.o').write_bytes(b'private object fixture')
        for name, blob in {
            'DonutStudio.exe': b'MZ admitted app', 'RELEASE_BUILD.json': b'{}',
            'video-helper/arbit-video-helper.exe': b'MZ admitted helper',
            'video-helper/runtime.dll': b'MZ admitted source runtime',
            'video-helper/software-renderer/arbit-video-helper.exe': b'MZ admitted helper',
            'video-helper/software-renderer/renderer.dll': b'MZ admitted renderer',
            'video-helper/DEPENDENCY_BUILD.json': b'{"status":"PASS"}',
            'video-helper/candidate-evidence/source.tar': b'admitted vendor source',
            'video-helper/candidate-evidence/NOTICE': b'admitted vendor notice',
            'video-helper/candidate-evidence/configuration': b'admitted configuration',
            'video-helper/candidate-evidence/relink.a': b'admitted LGPL replacement payload',
            'video-helper/candidate-evidence/FIRST_PARTY_LICENSE.txt': b'owner declaration',
            w.APP_RECEIPT: b'original app seal', w.PACKAGE_RECEIPT: b'original package seal',
            'video-helper/' + w.HELPER_RECEIPT: b'original helper seal',
            'DonutStudio.clap': b'MZ optional CLAP',
            'DonutStudio.vst3/Contents/x86_64-win/DonutStudio.vst3': b'MZ optional VST3',
            'Soundfonts/optional.sf2': b'optional externally installed content',
            'DonutStudio.control.js': b'optional externally installed Bitwig',
        }.items():
            path = self.stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        self.closure = []
        for name, dependency, imports in (
            ('video-helper/arbit-video-helper.exe', 'first-party',
             [{'name': 'runtime.dll', 'target': 'video-helper/runtime.dll'}]),
            ('video-helper/runtime.dll', 'source-runtime', [{'name': 'KERNEL32.dll', 'target': None}]),
            ('video-helper/software-renderer/arbit-video-helper.exe', 'first-party',
             [{'name': 'renderer.dll', 'target': 'video-helper/software-renderer/renderer.dll'}]),
            ('video-helper/software-renderer/renderer.dll', 'renderer-source', []),
        ):
            self.closure.append({'path': name, 'sha256': w.sha256(self.stage / name),
                                 'dependency': dependency, 'imports': imports})
        self.package_files = w.file_hashes(self.stage)
        self.package = {'artifactHashes': copy.deepcopy(self.package_files), 'closure': {
            'status': 'PASS', 'binaries': self.closure,
            'peHashes': {row['path']: row['sha256'] for row in self.closure}}}
        prefix = 'video-helper/candidate-evidence/'
        self.dependencies = [{'name': 'source-runtime', 'version': 'fixture', 'license': 'LGPL-2.1-or-later',
            'sources': {prefix + 'source.tar': self.package_files[prefix + 'source.tar']},
            'configuration': {prefix + 'configuration': self.package_files[prefix + 'configuration']},
            'patches': {}, 'licenses': {prefix + 'NOTICE': self.package_files[prefix + 'NOTICE']},
            'notices': {prefix + 'NOTICE': self.package_files[prefix + 'NOTICE']},
            'relink': {prefix + 'relink.a': self.package_files[prefix + 'relink.a']}}]
        self.dependencies.append({**copy.deepcopy(self.dependencies[0]), 'name': 'renderer-source',
                                  'license': 'MIT', 'relink': {}})
        # These fixtures isolate Windows format plumbing. They do not assert
        # that the current archive/notice-only record is an admitted vendor build.
        self.helper_record = {'helperDependencies': {'ncnn-src': {'archive': 'retained.tar'}},
            'source': {'sourceCommit': COMMIT, 'inputHashes': SOURCE['compiledInputs']},
            'compiledSource': {'repositoryInputs': SOURCE['compiledInputs'], 'retainedFiles': {'fixture': '8' * 64}},
            'privateEvidence': w.private_evidence_binding(self.private),
            'privateSourceArchiveSha256': w.sha256(self.private / 'windows-helper-first-party-source.tar.gz')}
        self.engine = w.distribution_engine()
        for patcher in (patch.object(w, 'ROOT', self.root),
                        patch.object(w, 'distribution_engine', return_value=self.engine),
                        patch.object(w, 'admit_package', side_effect=self.admit_original),
                        patch.object(w, 'validate_helper', return_value=self.helper_record),
                        patch.object(self.engine, 'committed_source', return_value=SOURCE),
                        patch.object(self.engine, 'verify_committed_source'),
                        patch.object(self.engine, 'verify_package')):
            patcher.start()
            self.addCleanup(patcher.stop)

    def admit_original(self, stage, commit, *, private=False, private_evidence=None):
        self.assertEqual(stage, self.stage)
        self.assertEqual(commit, COMMIT)
        current = w.file_hashes(stage, (w.DISTRIBUTION_RECEIPT,) if private else ())
        if current != self.package_files:
            raise ValueError('original package byte drift')
        return copy.deepcopy(self.package)

    def emit(self):
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            return w.emit_private_receipt(self.stage, COMMIT)

    def test_current_static_vendor_records_fail_before_emission(self):
        self.helper_record['compiledSource'] = {'status': 'UNVERIFIED', 'reason': 'missing compiler proof'}
        with self.assertRaisesRegex(ValueError, 'static/vendor source-to-object/link.*ncnn-src'):
            w.emit_private_receipt(self.stage, COMMIT)
        self.assertFalse((self.stage / w.DISTRIBUTION_RECEIPT).exists())

    def test_final_receipt_preserves_existing_seals_and_all_vendor_payload(self):
        before = copy.deepcopy(self.package_files)
        result = self.emit()
        self.assertEqual(result['receipt']['packageFiles'], before)
        self.assertEqual(result['receipt']['closure'], self.closure)
        self.assertEqual(result['receipt']['dependencies'], self.dependencies)
        self.assertNotIn(w.DISTRIBUTION_RECEIPT, result['receipt']['packageFiles'])
        self.assertEqual(w.sha256(self.stage / w.DISTRIBUTION_RECEIPT), result['receiptSha256'])
        self.assertEqual(w.file_hashes(self.stage, (w.DISTRIBUTION_RECEIPT,)), before)
        self.assertEqual(result['receipt']['dependencyReceipt']['path'], 'video-helper/' + w.HELPER_RECEIPT)
        self.assertEqual(result['privateEvidence'], {'root': str(self.private), **self.helper_record['privateEvidence']})
        self.assertNotIn('retained-object.o', result['receipt']['packageFiles'])
        self.engine.committed_source.assert_called_with(self.root, COMMIT,
            compiled_inputs=SOURCE['compiledInputs'])

    def test_existing_receipt_cannot_bypass_missing_static_authority(self):
        self.emit()
        self.helper_record['compiledSource'] = {'status': 'UNVERIFIED', 'reason': 'missing map'}
        with self.assertRaisesRegex(ValueError, 'static/vendor source-to-object/link'):
            w.verify_private_receipt(self.stage, COMMIT)

    def test_unadmitted_files_are_rejected_before_shared_inventory(self):
        (self.stage / 'unadmitted.txt').write_text('not part of prior seal')
        with self.assertRaisesRegex(ValueError, 'original package byte drift'):
            self.emit()

    def test_changed_source_app_helper_notice_and_relink_are_refused(self):
        self.emit()
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            for name in ('DonutStudio.exe', 'video-helper/arbit-video-helper.exe',
                         'video-helper/candidate-evidence/NOTICE', 'video-helper/candidate-evidence/relink.a'):
                path = self.stage / name
                before = path.read_bytes()
                path.write_bytes(before + b'changed')
                with self.assertRaisesRegex(ValueError, 'original package byte drift'):
                    w.verify_private_receipt(self.stage, COMMIT)
                path.write_bytes(before)
            with patch.object(self.engine, 'committed_source', return_value={**SOURCE, 'sourceCommit': '4' * 40}):
                with self.assertRaisesRegex(ValueError, 'source/platform mismatch'):
                    w.verify_private_receipt(self.stage, COMMIT)

    def test_closure_requires_complete_audited_import_and_source_rows(self):
        for edited in ({'status': 'PASS', 'peHashes': {'video-helper/hidden.dll': '0' * 64}},
                       {**self.package['closure'], 'binaries': self.closure[:-1]}):
            with self.assertRaisesRegex(ValueError, 'PE.*(?:associations|incomplete)'):
                w.private_closure({'closure': edited})
        self.assertEqual(w.private_closure(self.package), self.closure)

    def test_installer_uses_exact_flattened_core_and_preserves_evidence(self):
        self.emit()
        payload = self.root / 'installer-core'
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            candidate = w.prepare_private_installer(self.stage, payload, COMMIT)
            w.verify_private_installer(self.stage, payload, COMMIT)
        receipt = candidate['receipt']
        self.assertEqual(receipt['helper']['path'], 'arbit-video-helper.exe')
        self.assertEqual(receipt['dependencyReceipt']['path'], w.HELPER_RECEIPT)
        self.assertEqual(candidate['privateEvidence'], {'root': str(self.private), **self.helper_record['privateEvidence']})
        self.assertEqual(receipt['closure'][0]['imports'][0]['target'], 'runtime.dll')
        self.assertEqual(receipt['dependencies'][0]['sources'], {
            'candidate-evidence/source.tar': self.package_files['video-helper/candidate-evidence/source.tar']})
        self.assertIn(w.APP_RECEIPT, receipt['packageFiles'])
        self.assertIn(w.HELPER_RECEIPT, receipt['packageFiles'])
        self.assertIn(w.PACKAGE_RECEIPT, receipt['packageFiles'])
        for name in ('DonutStudio.clap', 'Soundfonts/optional.sf2', 'DonutStudio.control.js'):
            self.assertNotIn(name, receipt['packageFiles'])
        self.assertEqual(self.engine.platform_identity(receipt), self.engine.platform_identity(
            json.loads((self.stage / w.DISTRIBUTION_RECEIPT).read_text())))

    def test_installer_projection_cannot_reseal_added_removed_or_changed_files(self):
        self.emit()
        payload = self.root / 'installer-core'
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            w.prepare_private_installer(self.stage, payload, COMMIT)
            original = payload / 'candidate-evidence/NOTICE'
            data = original.read_bytes()
            for action in ('changed', 'removed', 'extra'):
                if action == 'changed':
                    original.write_bytes(data + b'drift')
                elif action == 'removed':
                    original.unlink()
                else:
                    (payload / 'unknown.dll').write_bytes(b'MZ extra')
                with self.assertRaisesRegex(ValueError, 'projection changed'):
                    w.verify_private_installer(self.stage, payload, COMMIT)
                original.write_bytes(data)
                (payload / 'unknown.dll').unlink(missing_ok=True)

    def test_installer_mapping_refuses_helper_app_collision(self):
        (self.stage / 'video-helper/DonutStudio.exe').write_bytes(b'MZ collision')
        with self.assertRaisesRegex(ValueError, 'path collision'):
            w.installer_payload_paths(self.stage)

    def test_private_inventory_drift_is_rejected_for_zip_and_installer(self):
        self.emit()
        payload = self.root / 'installer-core'
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            w.prepare_private_installer(self.stage, payload, COMMIT)
            path = self.private / 'retained-object.o'
            original = path.read_bytes()
            for action in ('changed', 'removed', 'extra'):
                if action == 'changed':
                    path.write_bytes(original + b'drift')
                elif action == 'removed':
                    path.unlink()
                else:
                    (self.private / 'unassociated.o').write_bytes(b'extra object')
                with self.assertRaisesRegex(ValueError, 'sealed candidate inventory'):
                    w.verify_private_receipt(self.stage, COMMIT)
                with self.assertRaisesRegex(ValueError, 'sealed candidate inventory'):
                    w.verify_private_installer(self.stage, payload, COMMIT)
                path.write_bytes(original)
                (self.private / 'unassociated.o').unlink(missing_ok=True)

    def test_caller_root_cannot_supply_its_own_self_consistent_inventory(self):
        other = self.root / 'caller-evidence'
        other.mkdir()
        (other / 'windows-helper-first-party-source.tar.gz').write_bytes(b'unadmitted source')
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            with self.assertRaisesRegex(ValueError, 'sealed candidate inventory'):
                w.emit_private_receipt(self.stage, COMMIT, private=other)
        self.assertFalse((self.stage / w.DISTRIBUTION_RECEIPT).exists())

    def test_wrong_dependency_receipt_cannot_replace_final_helper_authority(self):
        self.emit()
        path = self.stage / w.DISTRIBUTION_RECEIPT
        receipt = json.loads(path.read_text())
        receipt['dependencyReceipt'] = {'path': 'video-helper/DEPENDENCY_BUILD.json',
            'sha256': self.package_files['video-helper/DEPENDENCY_BUILD.json']}
        path.write_text(json.dumps(receipt))
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            with self.assertRaisesRegex(ValueError, 'differs from admitted dependency'):
                w.verify_private_receipt(self.stage, COMMIT)

    def test_all_package_helper_checks_receive_explicit_private_root(self):
        with patch.object(w, 'private_dependency_evidence', return_value=self.dependencies):
            w.emit_private_receipt(self.stage, COMMIT, private=self.private)
        calls = w.validate_helper.call_args_list
        self.assertTrue(calls)
        self.assertTrue(all(call.kwargs['private'] == self.private for call in calls))

    def test_nominal_pass_does_not_activate_private_compiler_authority(self):
        self.helper_record['compiledSource'] = {'status': 'PASS'}
        with self.assertRaisesRegex(ValueError, 'static/vendor source-to-object/link'):
            w.private_dependency_evidence(self.stage, self.helper_record, {'status': 'PASS'}, private=self.private)

    def test_missing_native_source_binding_remains_refused(self):
        (self.helper / w.HELPER_RECEIPT).write_text(json.dumps(self.helper_record))
        (self.stage / w.PACKAGE_RECEIPT).write_text(json.dumps(self.package))
        with self.assertRaisesRegex(ValueError, 'native dependency source-build authority is incomplete'):
            w.private_dependency_evidence(self.stage, self.helper_record, {'status': 'PASS'}, private=self.private)

    def test_caller_cannot_replace_selected_sealed_helper_source_map(self):
        (self.helper / w.HELPER_RECEIPT).write_text(json.dumps(self.helper_record))
        forged = copy.deepcopy(self.helper_record)
        forged['compiledSource']['repositoryInputs'] = {'borrowed/unbound.cpp': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'exact original helper and codec receipts'):
            w.private_dependency_evidence(self.stage, forged, {'status': 'PASS'}, private=self.private)


class AuditedPEAssociations(unittest.TestCase):
    def test_complete_import_targets_and_actual_source_associations_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / 'arbit-video-helper.exe'
            runtime = root / 'unrelated-name.dll'
            helper.write_bytes(b'MZ first-party helper')
            runtime.write_bytes(b'MZ source-built runtime')
            dependency = {'profile': 'fixture', 'limitations': [], 'executableProvenance': {},
                          'dllProvenance': {runtime.name: {'source': 'actual-source-name',
                              'sha256': w.sha256(runtime), 'sourceSha256': '4' * 64}}}
            path = root.parent / 'receipt.json'
            path_hash = '5' * 64
            def inspected(pe):
                return [runtime.name] if pe == helper else ['KERNEL32.dll']
            with patch.object(w, 'validate_receipt', return_value=dependency), \
                    patch.object(w, 'imports', side_effect=inspected), \
                    patch.object(w, 'sha256', side_effect=lambda p: path_hash if p == path else hashlib.sha256(p.read_bytes()).hexdigest()):
                closure = w.audit_closure(root, path)
            rows = {row['path']: row for row in closure['binaries']}
            self.assertEqual(rows[helper.name]['imports'], [{'name': runtime.name, 'target': runtime.name}])
            self.assertEqual(rows[runtime.name]['dependency'], 'actual-source-name')
            self.assertEqual(rows[runtime.name]['imports'], [{'name': 'KERNEL32.dll', 'target': None}])
            self.assertEqual({name: row['sha256'] for name, row in rows.items()}, closure['peHashes'])


class PrivateWiring(unittest.TestCase):
    def test_emission_after_final_transformations_before_zip(self):
        full = (ROOT / 'scripts/build-windows-local.sh').read_text()
        self.assertIn('private Windows helper distribution requires --lgpl-candidate-package', full)
        self.assertLess(full.index('scripts/prepare-release-symbols.py'), full.index('--emit-private-distribution'))
        self.assertLess(full.index('--seal-package'), full.index('--emit-private-distribution'))
        self.assertLess(full.index('--emit-private-distribution'), full.index('zip -rq'))
        helper = (ROOT / 'scripts/build-windows-videohelper-local.sh').read_text()
        self.assertLess(helper.index('private Windows distribution requires the matching app pipeline'),
                        helper.index('$DOCKER run'))

    def test_inno_only_changes_private_core_keeps_optional_external_destinations(self):
        installer = (ROOT / 'installers/donutstudio-installer.iss').read_text()
        private, legacy = installer.split('#ifdef PrivateHelperPayload', 1)[1].split('#else', 1)
        self.assertIn('Components: standalone;', private)
        self.assertIn('Source: "{#PrivateHelperPayload}\\*"', private)
        self.assertIn('Source: "{#ArtifactsDir}\\video-helper\\*"', legacy)
        for component in ('vst3', 'clap', 'soundfonts', 'soundfonts_premium', 'soundfonts_sfz', 'bitwig'):
            self.assertIn('Components: ' + component + ';', installer)
        wrapper = (ROOT / 'scripts/build-windows-installer-local.sh').read_text()
        self.assertLess(wrapper.index('--prepare-private-installer'), wrapper.index('wine "$ISCC"'))
        self.assertLess(wrapper.index('wine "$ISCC"'), wrapper.index('--admit-private-installer'))
        self.assertIn('"${PRIVATE_INNO_ARGS[@]}"', wrapper)


if __name__ == '__main__':
    unittest.main()
