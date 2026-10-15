#!/usr/bin/env python3
"""Layout and inventory fixtures with a mocked source authority; no native acceptance."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('linux_package_fixture', Path(__file__).with_name('linux_candidate_package_inputs_tests.py'))
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
validator = fixtures.validator
COMMIT = fixtures.COMMIT
LICENSE = b'fixture owner declaration\n'
REAL_PRIVATE_INPUTS = validator.private_dependency_inputs
REAL_DISTRIBUTION_API = validator.distribution_api


class InventoryApi:
    """Exercise format plumbing only. Production compiled-source proof stays blocked."""
    def __init__(self):
        self.calls = []
        self.source = {'sourceCommit': COMMIT, 'helperTreeSha256': 'b' * 64,
                       'compiledInputs': {'video-helper/src/main.cpp': 'c' * 64},
                       'firstPartyLicense': {'path': 'video-helper/FIRST_PARTY_LICENSE.txt',
                                            'sha256': hashlib.sha256(LICENSE).hexdigest()}}

    def committed_source(self, repository, commit, *, compiled_inputs):
        if commit != COMMIT or compiled_inputs != self.source['compiledInputs']:
            raise ValueError('source mismatch')
        return self.source

    @staticmethod
    def private_materials(root):
        return {p.relative_to(root).as_posix(): validator.digest(p)
                for p in root.rglob('*') if p.is_file() and not p.is_symlink()}

    @staticmethod
    def inventory_digest(files):
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    def verify_compiled_source_associations(self, receipt, read, repository, **private):
        raise ValueError('fixture raw verifier refusal')

    def verify_committed_source(self, source, repository):
        if source != self.source:
            raise ValueError('source tree/declaration mismatch')

    def emit_receipt(self, root, **inputs):
        self.calls.append(inputs)
        names, read, _ = validator.storage_view(root)
        files = {name: hashlib.sha256(read(name)).hexdigest() for name in names
                 if name != validator.DISTRIBUTION_RECEIPT}
        receipt = {'schemaVersion': 1, 'profile': validator.PRIVATE_PROFILE,
                   'platform': inputs['platform'], 'source': inputs['source'],
                   'helperRoots': inputs['helper_roots'], 'closure': inputs['closure'],
                   'dependencies': inputs['dependencies'], 'packageFiles': files}
        for label, field in (('helper', 'helper_path'), ('application', 'application_path'),
                             ('dependencyReceipt', 'dependency_receipt_path')):
            receipt[label] = {'path': inputs[field], 'sha256': files[inputs[field]]}
        fixtures.write_json(root / validator.DISTRIBUTION_RECEIPT, receipt)
        return {'receiptPath': validator.DISTRIBUTION_RECEIPT,
                'receiptSha256': validator.digest(root / validator.DISTRIBUTION_RECEIPT), 'receipt': receipt,
                'privateEvidence': self.private_binding(inputs['private_evidence_root'])}

    def private_binding(self, root):
        files = self.private_materials(root)
        return {'root': str(root), 'files': files, 'contentSha256': self.inventory_digest(files)}

    def verify_package(self, candidate, names, read, links, repository):
        if candidate['privateEvidence'] != self.private_binding(Path(candidate['privateEvidence']['root'])):
            raise ValueError('private compiled evidence drift')
        receipt = candidate['receipt']
        if names - {validator.DISTRIBUTION_RECEIPT} != set(receipt['packageFiles']):
            raise ValueError('missing or unaccounted files')
        for name, expected in receipt['packageFiles'].items():
            if hashlib.sha256(read(name)).hexdigest() != expected:
                raise ValueError('retained payload drift: ' + name)


class LinuxPrivateEmitterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = fixtures.PackageFixture(self.root)
        self.api = InventoryApi()
        candidate = self.fixture.candidate
        def evidence(name):
            return {name: validator.digest(candidate / name)}
        self.dependencies = [{'name': 'ffmpeg', 'version': 'fixture', 'license': 'LGPL-2.1-only',
            'sources': evidence('dependencies/archives/' + self.fixture.sources[0]['archive']),
            'configuration': evidence('evidence/CMakeCache.txt'), 'patches': {},
            'licenses': evidence('dependencies/evidence/retained.txt'),
            'notices': evidence('evidence/notices/system/retained.txt'),
            'relink': evidence('dependencies/source/retained.txt')}]
        self.api_patch = patch.object(validator, 'distribution_api', return_value=self.api)
        self.api_patch.start()
        private_root = Path(validator.read_json(candidate / 'CANDIDATE_BUILD.json')['sourceArchive']['path']).parent
        self.admission = {'dependencies': self.dependencies, 'source': self.api.source,
                          'privateRoot': private_root, 'privateEvidence': self.api.private_binding(private_root)}
        self.admission_patch = patch.object(validator, 'private_dependency_inputs', return_value=self.admission)
        self.admission_patch.start()

    def tearDown(self):
        self.admission_patch.stop()
        self.api_patch.stop()
        self.temporary.cleanup()

    def emit(self, package, layout='archive'):
        with patch.object(validator.subprocess, 'check_output', return_value=LICENSE):
            return validator.emit_distribution_package(package, COMMIT, layout)

    def appdir(self):
        appdir = self.root / 'squashfs-root'
        helper, doc, files = validator.package_paths('appimage')
        shutil.copytree(self.fixture.candidate, appdir / helper)
        for label, path in files.items():
            target = appdir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.fixture.package / label, target)
        (appdir / doc).mkdir(parents=True)
        shutil.copy2(self.fixture.package / 'RELEASE_BUILD.json', appdir / doc / 'RELEASE_BUILD.json')
        (appdir / 'usr/bin/arbit-video-helper').symlink_to('video-helper/arbit-video-helper')
        validator.seal_package(appdir, COMMIT, 'appimage', self.fixture.package)
        return appdir

    def test_archive_zip_and_shell_tar_share_final_payload_inventory(self):
        package = self.fixture.package
        before = {name: validator.digest(package / name) for name in
                  ('LINUX_HELPER_PACKAGE.json', 'LINUX_HELPER_PACKAGE.sha256', 'video-helper/CANDIDATE_BUILD.json')}
        candidate = self.emit(package)
        files = candidate['receipt']['packageFiles']
        self.assertEqual(candidate['receipt']['dependencyReceipt']['path'],
                         'video-helper/LINUX_DEPENDENCY_PROVENANCE.json')
        self.assertEqual(candidate['privateEvidence'], self.admission['privateEvidence'])
        self.assertEqual(self.api.calls[-1]['private_evidence_root'], self.admission['privateRoot'])
        self.assertEqual(candidate['receipt']['source']['compiledInputs'], self.api.source['compiledInputs'])
        self.assertNotIn(validator.DISTRIBUTION_RECEIPT, files)
        self.assertEqual(set(files), validator.storage_view(package)[0] - {validator.DISTRIBUTION_RECEIPT})
        self.assertIn('helper-distribution/FIRST_PARTY_LICENSE.txt', files)
        self.assertEqual(before, {name: validator.digest(package / name) for name in before})
        validator.validate_distribution_package(package, COMMIT, 'archive')
        # Both container seams consume this exact root, without re-stripping or
        # copying a reduced helper-only inventory.
        build = (validator.ROOT / 'scripts/build-linux-docker-build.sh').read_text()
        installer = (validator.ROOT / 'installers/make-linux-installer.sh').read_text()
        self.assertLess(build.index('emit-distribution'), build.index('zip -rq'))
        self.assertIn('tar czf "$PAYLOAD_FILE" -C "$ARTIFACTS_DIR" .', installer)
        self.assertLess(installer.index('linux-video-helper-candidate.py'), installer.index('tar czf'))

    def test_appimage_paths_and_alias_use_final_squashfs_root(self):
        appdir = self.appdir()
        candidate = self.emit(appdir, 'appimage')
        receipt = candidate['receipt']
        self.assertEqual(receipt['application']['path'], 'usr/bin/DonutStudio')
        self.assertEqual(receipt['helperRoots'], ['usr/bin/video-helper', 'usr/bin/arbit-video-helper'])
        self.assertIn('usr/bin/arbit-video-helper', {row['path'] for row in receipt['closure']})
        self.assertEqual(receipt['packageFiles']['usr/bin/arbit-video-helper'],
                         hashlib.sha256(b'video-helper/arbit-video-helper').hexdigest())
        for dependency in receipt['dependencies']:
            for label in ('sources', 'configuration', 'licenses', 'notices', 'relink'):
                self.assertTrue(all(name.startswith('usr/bin/video-helper/') for name in dependency[label]))
        script = (validator.ROOT / 'installers/make-appimage.sh').read_text()
        self.assertLess(script.index('emit-distribution'), script.index('"$OUTPUT_PLUGIN" --appdir'))
        self.assertIn('--layout appimage --reference-package', script)
        helper = next(row for row in receipt['closure'] if row['path'] == receipt['helper']['path'])
        self.assertEqual(helper['imports'], [{'name': 'libavcodec.so.60',
                                             'target': 'usr/bin/video-helper/libavcodec.so.60'}])

    def test_missing_receipt_or_wrong_source_profile_layout_reject(self):
        package = self.fixture.package
        with self.assertRaisesRegex(ValueError, 'missing regular'):
            validator.validate_distribution_package(package, COMMIT, 'archive')
        fixtures.write_json(package / validator.DISTRIBUTION_RECEIPT, {'source': []})
        with self.assertRaisesRegex(ValueError, 'malformed'):
            validator.validate_distribution_package(package, COMMIT, 'archive')
        self.emit(package)
        for field, value in (('profile', 'legacy'), ('platform', 'windows')):
            receipt = fixtures.validator.read_json(package / validator.DISTRIBUTION_RECEIPT)
            receipt[field] = value
            fixtures.write_json(package / validator.DISTRIBUTION_RECEIPT, receipt)
            with self.assertRaisesRegex(ValueError, 'source/profile/layout'):
                validator.validate_distribution_package(package, COMMIT, 'archive')
            self.emit(package)
        with self.assertRaisesRegex(ValueError, 'source/profile/layout'):
            validator.validate_distribution_package(package, 'f' * 40, 'archive')
        with self.assertRaisesRegex(ValueError, 'source/profile/layout'):
            validator.validate_distribution_package(package, COMMIT, 'appimage')
        receipt = fixtures.validator.read_json(package / validator.DISTRIBUTION_RECEIPT)
        receipt['source']['helperTreeSha256'] = 'e' * 64
        fixtures.write_json(package / validator.DISTRIBUTION_RECEIPT, receipt)
        with self.assertRaisesRegex(ValueError, 'compiled source/declaration'):
            validator.validate_distribution_package(package, COMMIT, 'archive')

    def test_final_inventory_rejects_added_removed_and_retained_payload_drift(self):
        package = self.fixture.package
        self.emit(package)
        retained = package / 'helper-distribution/FIRST_PARTY_LICENSE.txt'
        retained.write_bytes(LICENSE + b'drift')
        with self.assertRaisesRegex(ValueError, 'retained payload drift'):
            validator.validate_distribution_package(package, COMMIT, 'archive')
        retained.write_bytes(LICENSE)
        extra = package / 'unadmitted.bin'
        extra.write_bytes(b'unknown bytes')
        with self.assertRaisesRegex(ValueError, 'unaccounted'):
            validator.validate_distribution_package(package, COMMIT, 'archive')
        extra.unlink()
        retained.unlink()
        with self.assertRaisesRegex(ValueError, 'unaccounted'):
            validator.validate_distribution_package(package, COMMIT, 'archive')

    def test_appimage_transform_drift_and_unknown_dynamic_origin_reject(self):
        appdir = self.appdir()
        (appdir / 'usr/bin/DonutStudio').write_bytes(b'transformed after archive seal')
        with self.assertRaisesRegex(ValueError, 'application bytes differ from archive'):
            validator.seal_package(appdir, COMMIT, 'appimage', self.fixture.package)
        with self.assertRaisesRegex(ValueError, 'no retained source evidence'):
            validator.private_closure(self.fixture.candidate, 'video-helper', [])

    def test_legacy_candidate_seal_still_admits_without_private_profile(self):
        validator.validate_package(self.fixture.package, COMMIT)
        self.assertFalse((self.fixture.package / validator.DISTRIBUTION_RECEIPT).exists())
        self.assertFalse(self.api.calls)

    def test_current_candidate_and_fabricated_pass_cannot_supply_missing_compiled_proof(self):
        self.admission_patch.stop()
        with self.assertRaisesRegex(ValueError, 'compiled/static/vendor'):
            validator.emit_distribution_package(self.fixture.package, COMMIT)
        self.assertFalse((self.fixture.package / validator.DISTRIBUTION_RECEIPT).exists())
        # No external assertion format is accepted by the internal resolver.
        fixtures.write_json(self.fixture.candidate / 'arbitrary-proof.json', {'status': 'PASS', 'staticDependencies': ['ffmpeg']})
        with self.assertRaisesRegex(ValueError, 'compiled/static/vendor'):
            REAL_PRIVATE_INPUTS(self.fixture.candidate)

    def set_proof(self, proof):
        candidate = self.fixture.candidate
        receipt = validator.read_json(candidate / 'CANDIDATE_BUILD.json')
        receipt['compiledSource'] = proof
        fixtures.write_json(candidate / 'CANDIDATE_BUILD.json', receipt)
        provenance = validator.read_json(candidate / 'LINUX_DEPENDENCY_PROVENANCE.json')
        provenance['compiledSource'] = proof
        stage = validator.read_json(candidate / 'DEPENDENCY_BUILD.json')
        material_root = 'linux-native-prefix'
        native_receipt = self.admission['privateRoot'] / material_root / 'DEPENDENCY_BUILD.json'
        native_receipt.parent.mkdir(parents=True, exist_ok=True)
        fixtures.write_json(native_receipt, stage)
        for name in stage['artifactHashes']:
            source = candidate / (Path(name).name if name.startswith('prefix/') else 'dependencies/' + name)
            target = self.admission['privateRoot'] / material_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        provenance['nativeSourceBindings'] = {row['sha256']: {'dependency': row['source']['name'],
            'receiptPath': material_root + '/DEPENDENCY_BUILD.json', 'materialRoot': material_root,
            'prefixBinary': row['prefixPath'], 'prefixSha256': row['sourceSha256'],
            'transformations': []} for row in provenance['dependencies'] if row.get('bundledPath')}
        receipt['nativeSourceBindings'] = provenance['nativeSourceBindings']
        receipt['privateEvidence'] = self.api.private_materials(self.admission['privateRoot'])
        fixtures.write_json(candidate / 'CANDIDATE_BUILD.json', receipt)
        fixtures.write_json(candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', provenance)
        self.admission['privateEvidence'] = self.api.private_binding(self.admission['privateRoot'])
        return receipt

    def test_diagnostic_linked_proof_and_status_pass_never_promote(self):
        for proof in ({'status': 'UNVERIFIED', 'linkedHelperProof': {'retainedFiles': {}}},
                      {'status': 'PASS', 'retainedFiles': {}, 'repositoryInputs': {}},
                      {'retainedFiles': {}, 'repositoryInputs': []}):
            self.set_proof(proof)
            with self.assertRaisesRegex(ValueError, 'diagnostic or claimed status'):
                REAL_PRIVATE_INPUTS(self.fixture.candidate)

    def test_raw_proof_verifier_receives_exact_private_readers_and_compiled_inputs(self):
        receipt = self.set_proof({'retainedFiles': {}, 'repositoryInputs': self.api.source['compiledInputs'], 'materials': {}})
        def refuse(authority, read, repository, *, private_read, private_files):
            self.assertEqual(authority['source']['compiledInputs'], self.api.source['compiledInputs'])
            self.assertEqual(authority['dependencyReceipt']['path'], 'LINUX_DEPENDENCY_PROVENANCE.json')
            self.assertEqual(json.loads(read(authority['dependencyReceipt']['path']))['compiledSource'],
                             receipt['compiledSource'])
            self.assertEqual(private_files, self.admission['privateEvidence']['files'])
            self.assertEqual(hashlib.sha256(private_read('helper-source.tar')).hexdigest(),
                             private_files['helper-source.tar'])
            raise ValueError('raw transformation refused')
        with patch.object(validator, 'validate_candidate', return_value=receipt), \
             patch.object(self.api, 'verify_compiled_source_associations', side_effect=refuse) as raw, \
             patch.object(validator, 'private_retained_dependencies', return_value=self.dependencies) as transport:
            with self.assertRaisesRegex(ValueError, 'raw transformation refused'):
                REAL_PRIVATE_INPUTS(self.fixture.candidate)
            self.assertEqual(raw.call_count, 1)
            transport.assert_called_once()

    def test_private_material_addition_drift_and_proof_mismatch_refuse_before_transport(self):
        proof = {'retainedFiles': {}, 'repositoryInputs': self.api.source['compiledInputs'], 'materials': {}}
        receipt = self.set_proof(proof)
        extra = self.admission['privateRoot'] / 'unadmitted-private.o'
        extra.write_bytes(b'not admitted')
        with patch.object(validator, 'validate_candidate', return_value=receipt):
            with self.assertRaisesRegex(ValueError, 'extra or changed'):
                REAL_PRIVATE_INPUTS(self.fixture.candidate)
        extra.unlink()
        retained = self.admission['privateRoot'] / 'helper-source.tar'
        retained.write_bytes(retained.read_bytes() + b'drift')
        with patch.object(validator, 'validate_candidate', return_value=receipt):
            with self.assertRaisesRegex(ValueError, 'source/symbol hash mismatch'):
                REAL_PRIVATE_INPUTS(self.fixture.candidate)
        provenance = validator.read_json(self.fixture.candidate / 'LINUX_DEPENDENCY_PROVENANCE.json')
        provenance['compiledSource']['repositoryInputs'] = {}
        fixtures.write_json(self.fixture.candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', provenance)
        with self.assertRaisesRegex(ValueError, 'differs from sealed candidate'):
            REAL_PRIVATE_INPUTS(self.fixture.candidate)

    def test_raw_admission_is_required_after_retained_declarations_and_keeps_private_material_external(self):
        receipt = self.set_proof({'retainedFiles': {}, 'repositoryInputs': self.api.source['compiledInputs'], 'materials': {}})
        with patch.object(validator, 'validate_candidate', return_value=receipt), \
             patch.object(self.api, 'verify_compiled_source_associations'), \
             patch.object(validator, 'private_retained_dependencies', return_value=self.dependencies):
            admission = REAL_PRIVATE_INPUTS(self.fixture.candidate)
        self.assertEqual(admission['source'], self.api.source)
        self.assertEqual(admission['privateEvidence'], self.admission['privateEvidence'])
        self.assertFalse(admission['privateRoot'].is_relative_to(self.fixture.package))
        self.assertEqual(admission['dependencies'], self.dependencies)

    def test_missing_retained_vendor_payload_refuses_without_admitting_raw_proof(self):
        receipt = self.set_proof({'retainedFiles': {}, 'repositoryInputs': self.api.source['compiledInputs'], 'materials': {}})
        with patch.object(validator, 'validate_candidate', return_value=receipt), \
             patch.object(self.api, 'verify_compiled_source_associations'), \
             patch.object(validator, 'private_retained_dependencies', side_effect=ValueError('vendor notice missing')):
            with self.assertRaisesRegex(ValueError, 'vendor notice missing'):
                REAL_PRIVATE_INPUTS(self.fixture.candidate)

    def test_native_prefix_mismatch_and_unknown_system_origin_refuse(self):
        receipt = self.set_proof({'retainedFiles': {}, 'repositoryInputs': self.api.source['compiledInputs'], 'materials': {}})
        candidate = self.fixture.candidate
        provenance = validator.read_json(candidate / 'LINUX_DEPENDENCY_PROVENANCE.json')
        provenance['nativeSourceBindings'][provenance['dependencies'][0]['sha256']]['prefixBinary'] = 'prefix/lib/not-admitted.so'
        receipt['nativeSourceBindings'] = provenance['nativeSourceBindings']
        fixtures.write_json(candidate / 'CANDIDATE_BUILD.json', receipt)
        fixtures.write_json(candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', provenance)
        with patch.object(validator, 'validate_candidate', return_value=receipt), \
             patch.object(self.api, 'verify_compiled_source_associations'):
            with self.assertRaisesRegex(ValueError, 'differs from admitted prefix'):
                REAL_PRIVATE_INPUTS(candidate)
        row = provenance['dependencies'][0]
        row.update(origin='system-package', sourcePackage='unknown-system-source')
        fixtures.write_json(candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', provenance)
        with patch.object(validator, 'validate_candidate', return_value=receipt), \
             patch.object(self.api, 'verify_compiled_source_associations'):
            with self.assertRaisesRegex(ValueError, 'lacks pinned source/license authority'):
                REAL_PRIVATE_INPUTS(candidate)

    def test_actual_shared_native_seam_checks_retained_declarations(self):
        receipt = self.set_proof({'retainedFiles': {}, 'repositoryInputs': self.api.source['compiledInputs'], 'materials': {}})
        candidate = self.fixture.candidate
        api = REAL_DISTRIBUTION_API()
        # Exercise actual shared source/native dispatch and declaration matching.
        # Only the lower compiler/prefix prerequisites are stubbed; this fixture
        # supplies no actual native source admission.
        compiler = SimpleNamespace(verify_compiled_source=lambda *a, **k: None,
                                   verify_transformations=lambda *a, **k: None)
        prefix = SimpleNamespace(verify_native_source_binding=lambda *a, **k:
                                 {'name': 'ffmpeg', 'version': 'fixture', 'license': 'LGPL-2.1-only'})
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        tool_root = self.root / 'shared-tool-root'
        native_path = tool_root / 'scripts/video-helper-dependencies/prefix_source.py'
        native_path.parent.mkdir(parents=True)
        native_path.write_text('# Lower native proof prerequisite stub only.\n')
        original = api.verify_compiled_source_associations
        def verify(authority, read, repository, **private_inputs):
            self.assertEqual(authority['dependencies'], self.dependencies)
            return original(authority, read, tool_root, **private_inputs)
        with patch.object(validator, 'validate_candidate', return_value=receipt), \
             patch.object(validator, 'private_retained_dependencies', return_value=self.dependencies), \
             patch.object(self.api, 'verify_compiled_source_associations', side_effect=verify), \
             patch.object(api, 'compiled_source_module', return_value=compiler), \
             patch.object(api.importlib.util, 'spec_from_file_location', return_value=spec), \
             patch.object(api.importlib.util, 'module_from_spec', return_value=prefix):
            admission = REAL_PRIVATE_INPUTS(candidate)
            self.assertEqual(admission['dependencies'], self.dependencies)
            self.dependencies[0]['version'] = 'arbitrary-declaration'
            with self.assertRaisesRegex(ValueError, 'native declared source/version/terms differ'):
                REAL_PRIVATE_INPUTS(candidate)

    def test_private_and_candidate_directory_overlap_refuses_both_directions(self):
        receipt = self.set_proof({'retainedFiles': {}, 'repositoryInputs': self.api.source['compiledInputs'], 'materials': {}})
        candidate = self.fixture.candidate
        for private in (candidate / 'nested-private', candidate.parent):
            private.mkdir(exist_ok=True)
            receipt['sourceArchive']['path'] = str(private / 'helper-source.tar')
            fixtures.write_json(candidate / 'CANDIDATE_BUILD.json', receipt)
            with patch.object(validator, 'validate_candidate', return_value=receipt), \
                 patch.object(validator, 'validate_private_evidence'):
                with self.assertRaisesRegex(ValueError, 'directories overlap'):
                    REAL_PRIVATE_INPUTS(candidate)

    def retained_mapper_fixture(self):
        # Synthetic bytes isolate transport from the independently verified raw
        # compiler authority; they are never a native/private release PASS.
        candidate = self.root / 'retained-transport'
        candidate.mkdir()
        def write(name, content):
            target = candidate / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return target
        policy = {'sources': json.loads(json.dumps(self.fixture.sources))}
        for source in policy['sources']:
            target = write('dependencies/archives/' + source['archive'], b'')
            with tarfile.open(target, 'w:gz') as archive:
                for notice in source['licenseFiles']:
                    raw = ('pinned notice ' + source['name'] + '/' + notice).encode()
                    member = tarfile.TarInfo(source['name'] + '-src/' + notice)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
                    write('dependencies/evidence/notices/' + source['name'] + '/' + notice.replace('/', '-'), raw)
            source['sha256'] = validator.digest(target)
        write('dependencies/evidence/configure.log', b'actual selected configure recipe')
        fixtures.write_json(candidate / 'DEPENDENCY_BUILD.json', {'sources': policy['sources']})
        library = write('libavcodec.so.60', b'actual native source-bound replacement bytes')
        fixtures.write_json(candidate / 'LINUX_DEPENDENCY_PROVENANCE.json', {'dependencies': [{
            'source': policy['sources'][0], 'bundledPath': library.name, 'sha256': validator.digest(library)}]})
        private = self.root / 'retained-mapper-private'
        private.mkdir()
        (private / 'vendor.tar').write_bytes(b'pinned vendor archive')
        (private / 'cache').write_bytes(b'exact raw compiler configuration')
        write('dependencies/helper-archives/vendor.tar', (private / 'vendor.tar').read_bytes())
        notice = write('dependencies/helper-source/vendor/LICENSE', b'verified vendor notice')
        source = write('dependencies/helper-source/vendor/vendor.c', b'verified vendor source')
        write('evidence/helper-compilation/cache', (private / 'cache').read_bytes())
        proof = {'repositoryRoot': '/repository', 'origins': [{
            'dependency': 'ncnn', 'kind': 'archive', 'root': '/build/vendor', 'archive': 'vendor.tar',
            'licenses': ['LICENSE'], 'patches': [], 'license': 'BSD-3-Clause'}],
            'materials': {'/build/vendor/LICENSE': {'dependency': 'ncnn', 'kind': 'source', 'sha256': validator.digest(notice)},
                          '/build/vendor/vendor.c': {'dependency': 'ncnn', 'kind': 'source', 'sha256': validator.digest(source)}},
            'evidence': {'cache': 'cache', 'depfiles': {}, 'ninjaDeps': None}}
        receipt = {'sourceCommit': COMMIT, 'artifactHashes': {p.relative_to(candidate).as_posix(): validator.digest(p)
                   for p in candidate.rglob('*') if p.is_file()}}
        original_read = validator.read_json
        def read(path):
            return policy if path == validator.ROOT / 'scripts/ffmpeg-lgpl-candidate/policy-6.1.6.json' else original_read(path)
        return candidate, receipt, proof, private, read

    def test_retained_mapper_binds_archives_notices_configuration_and_replacement_bytes(self):
        candidate, receipt, proof, private, read = self.retained_mapper_fixture()
        with patch.object(validator, 'read_json', side_effect=read):
            dependencies = validator.private_retained_dependencies(candidate, receipt, proof, private)
        by_name = {row['name']: row for row in dependencies}
        self.assertEqual(set(by_name), {'ffmpeg', 'openh264', 'kvazaar', 'nv-codec-headers', 'ncnn'})
        self.assertEqual(by_name['ffmpeg']['relink'], {'libavcodec.so.60': validator.digest(candidate / 'libavcodec.so.60')})
        self.assertIn('dependencies/helper-archives/vendor.tar', by_name['ncnn']['sources'])
        self.assertIn('dependencies/helper-source/vendor/LICENSE', by_name['ncnn']['licenses'])
        self.assertEqual(by_name['ncnn']['configuration'], {'evidence/helper-compilation/cache': validator.digest(private / 'cache')})
        self.assertTrue(all(not Path(name).is_absolute() for row in dependencies
                            for label in ('sources', 'licenses', 'configuration', 'relink') for name in row[label]))

    def test_retained_mapper_missing_notice_patch_and_authored_drift_refuse(self):
        candidate, receipt, proof, private, read = self.retained_mapper_fixture()
        authored = candidate / 'dependencies/helper-source/vendor/vendor.c'
        authored.write_bytes(b'arbitrary source edit')
        with patch.object(validator, 'read_json', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                validator.private_retained_dependencies(candidate, receipt, proof, private)
        authored.write_bytes(b'verified vendor source')
        proof['origins'][0]['licenses'] = ['UNKNOWN-NOTICE']
        with patch.object(validator, 'read_json', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'notice lacks exact source association'):
                validator.private_retained_dependencies(candidate, receipt, proof, private)
        proof['origins'][0]['licenses'] = ['LICENSE']
        proof['origins'][0]['patches'] = ['video-helper/patches/not-retained.cmake']
        with patch.object(validator, 'read_json', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'patch lacks exact source association'):
                validator.private_retained_dependencies(candidate, receipt, proof, private)


if __name__ == '__main__':
    unittest.main()
