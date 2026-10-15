#!/usr/bin/env python3
"""Synthetic producer fixtures. No compiler, native tool, Docker or app runs."""
import copy
import importlib.util
import json
import os
import shlex
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/video-helper-dependencies'))
import compiled_source
import prefix_source


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


linux = load('linux_native_producer', 'scripts/video-helper-dependencies/linux.py')
validator = load('linux_native_package', 'installers/linux-video-helper-candidate.py')


class NativeProducerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.stage = self.root / 'stage'
        self.private = self.root / 'private'
        self.output = self.root / 'candidate'
        for path in (self.stage, self.private, self.output):
            path.mkdir()
        self.prefix = self.stage / 'prefix'
        self.receipt = {'sources': [], 'artifactHashes': {}}
        for name, blob in {'prefix/lib/libfixture.so.1': b'\x7fELFfixture-linked',
                           'source/fixture/input.c': b'fixture authored source',
                           'evidence/recipe.py': b'fixture executed recipe'}.items():
            target = self.stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            self.receipt['artifactHashes'][name] = linux.sha256(target)
        self.write_receipt()
        self.tool = self.root / 'patchelf'
        self.tool.write_bytes(b'fixture tool bytes')
        self.binary = self.output / 'libfixture.so.1'
        self.binary.write_bytes((self.prefix / 'lib/libfixture.so.1').read_bytes())
        self.chains = {}

    def tearDown(self):
        self.temporary.cleanup()

    def write_receipt(self):
        (self.stage / 'DEPENDENCY_BUILD.json').write_text(json.dumps(self.receipt))

    def transform(self, operation='elf-set-rpath'):
        command = [str(self.tool), '--set-rpath', '$ORIGIN', str(self.binary)]
        linux.record_transformation(self.private, self.chains, self.binary, operation, command,
            lambda: self.binary.write_bytes(self.binary.read_bytes() + b'-' + operation.encode()),
            tool=self.tool if operation == 'elf-set-rpath' else None)

    def retain(self):
        return linux.retain_codec_stage(self.prefix, self.receipt, self.private)

    def native_row(self):
        return {'soname': self.binary.name, 'bundledPath': self.binary.name, 'origin': 'source-prefix',
                'source': {'name': 'fixture'}, 'prefixPath': 'prefix/lib/libfixture.so.1',
                'sourceSha256': self.receipt['artifactHashes']['prefix/lib/libfixture.so.1'],
                'sha256': linux.sha256(self.binary)}

    def test_complete_codec_stage_materializes_aliases_and_keeps_source_private(self):
        alias = self.prefix / 'lib/libfixture.so'
        alias.symlink_to('libfixture.so.1')
        self.receipt['artifactHashes']['prefix/lib/libfixture.so'] = linux.sha256(alias)
        self.write_receipt()
        retained, files = self.retain()
        self.assertEqual(files, {**self.receipt['artifactHashes'],
                                'DEPENDENCY_BUILD.json': linux.sha256(self.stage / 'DEPENDENCY_BUILD.json')})
        self.assertFalse((retained / 'prefix/lib/libfixture.so').is_symlink())
        self.assertEqual(list(self.output.iterdir()), [self.binary])
        self.assertFalse(retained.is_relative_to(self.output))

    def test_unknown_missing_changed_and_escaped_codec_material_refuse(self):
        extra = self.stage / 'source/unrecorded.h'
        extra.write_text('unrecorded')
        with self.assertRaisesRegex(ValueError, 'Unrecorded'):
            self.retain()
        extra.unlink()
        original = self.stage / 'source/fixture/input.c'
        original.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact drift'):
            self.retain()
        original.unlink()
        original.symlink_to(self.tool)
        with self.assertRaises(ValueError):
            self.retain()

    def test_overlapping_private_codec_root_refuses(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            linux.retain_codec_stage(self.prefix, self.receipt, self.stage)

    def test_executed_rpath_mutation_retains_exact_tool_command_and_bytes(self):
        before = self.binary.read_bytes()
        self.transform()
        step = self.chains[str(self.binary)][0]
        self.assertEqual(step['operation'], 'elf-set-rpath')
        self.assertEqual(shlex.split((self.private / step['command']).read_text()),
                         [str(self.tool), '--set-rpath', '$ORIGIN', str(self.binary)])
        self.assertEqual((self.private / step['input']).read_bytes(), before)
        self.assertEqual((self.private / step['output']).read_bytes(), self.binary.read_bytes())
        self.assertEqual(step['tool']['sha256'], linux.sha256(self.tool))
        self.assertEqual((self.private / step['tool']['binary']).read_bytes(), self.tool.read_bytes())

    def test_failed_mutation_does_not_append_a_completed_step(self):
        with self.assertRaisesRegex(ValueError, 'execution failed'):
            linux.record_transformation(self.private, self.chains, self.binary, 'elf-set-rpath', [],
                lambda: (_ for _ in ()).throw(ValueError('execution failed')), tool=self.tool)
        self.assertEqual(self.chains[str(self.binary)], [])

    def test_symbol_workflow_hook_keeps_the_actual_contiguous_mutations(self):
        self.transform()
        strip, objcopy = self.root / 'strip', self.root / 'objcopy'
        strip.write_bytes(b'fixture strip')
        objcopy.write_bytes(b'fixture objcopy')
        def execute(command, *, capture=False):
            self.binary.write_bytes(self.binary.read_bytes() + b'-' + Path(command[0]).name.encode())
        module = SimpleNamespace(run=execute)
        def main():
            module.run(['strip', '--strip-unneeded', str(self.binary)])
            module.run(['objcopy', '--add-gnu-debuglink=/temporary/fixture.debug', str(self.binary)])
        module.main = main
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        previous = sys.argv
        with patch.object(linux.importlib.util, 'spec_from_file_location', return_value=spec), \
             patch.object(linux.importlib.util, 'module_from_spec', return_value=module), \
             patch.object(linux.shutil, 'which', side_effect=lambda name: str(self.root / name)), \
             patch.dict(os.environ, {'DONUTSTUDIO_RELEASE_VERSION': 'fixture'}):
            linux.prepare_candidate_symbols(self.output, self.private / 'symbols.tar.gz',
                self.output / 'report.json', 'a' * 40, self.private, self.chains)
        self.assertIs(sys.argv, previous)
        steps = self.chains[str(self.binary)]
        self.assertEqual([step['operation'] for step in steps], ['elf-set-rpath', 'strip', 'debuglink'])
        for before, after in zip(steps, steps[1:]):
            self.assertEqual((self.private / before['output']).read_bytes(),
                             (self.private / after['input']).read_bytes())
        self.assertEqual(shlex.split((self.private / steps[1]['command']).read_text())[0], str(strip))

    def test_final_native_binding_calls_raw_prefix_and_transform_authorities(self):
        _, files = self.retain()
        self.transform()
        row = self.native_row()
        # Positive synthetic plumbing only. Neither lower authority is claimed.
        with patch.object(prefix_source, 'verify_native_source_binding', return_value={'name': 'fixture'}) as origin, \
             patch.object(compiled_source, 'verify_transformations') as transform:
            bindings, diagnostics = linux.retain_native_source_bindings(self.output, [row], self.receipt,
                self.private, self.chains, files)
        self.assertEqual(diagnostics, [])
        binding = bindings[row['sha256']]
        self.assertEqual(binding['prefixSha256'], row['sourceSha256'])
        self.assertEqual(binding['transformations'], self.chains[str(self.binary)])
        self.assertEqual(origin.call_args.kwargs['expected_sha256'], row['sourceSha256'])
        self.assertEqual(transform.call_args.kwargs['expected_sha256'], row['sha256'])

    def test_unknown_system_and_rejected_transformation_never_emit_binding(self):
        _, files = self.retain()
        row = self.native_row()
        row['origin'] = 'system-package'
        bindings, diagnostics = linux.retain_native_source_bindings(self.output, [row], self.receipt,
            self.private, self.chains, files)
        self.assertEqual(bindings, {})
        self.assertIn('lacks pinned source build', diagnostics[0])
        row['origin'] = 'source-prefix'
        with patch.object(prefix_source, 'verify_native_source_binding', return_value={'name': 'fixture'}), \
             patch.object(compiled_source, 'verify_transformations', side_effect=ValueError('changed executable data')):
            bindings, diagnostics = linux.retain_native_source_bindings(self.output, [row], self.receipt,
                self.private, self.chains, files)
        self.assertEqual(bindings, {})
        self.assertIn('changed executable data', diagnostics[0])

    def package_binding(self):
        retained, files = self.retain()
        self.transform()
        row = self.native_row()
        binding = {'dependency': 'fixture', 'receiptPath': 'linux-native-prefix/DEPENDENCY_BUILD.json',
            'materialRoot': 'linux-native-prefix', 'prefixBinary': row['prefixPath'],
            'prefixSha256': row['sourceSha256'], 'transformations': self.chains[str(self.binary)]}
        receipt = {'nativeSourceBindings': {row['sha256']: binding}}
        provenance = {**copy.deepcopy(receipt), 'dependencies': [row]}
        private_files = {str(p.relative_to(self.private)): linux.sha256(p)
                         for p in self.private.rglob('*') if p.is_file()}
        return receipt, provenance, private_files

    def test_package_gate_accepts_exact_sealed_inventory_before_lower_verification(self):
        receipt, provenance, files = self.package_binding()
        validator.validate_native_source_bindings(receipt, provenance, self.receipt, self.private, files)

    def test_package_gate_refuses_extra_missing_and_changed_stage_material(self):
        receipt, provenance, files = self.package_binding()
        for change in ('extra', 'missing', 'changed'):
            drift = dict(files)
            if change == 'extra':
                drift['linux-native-prefix/unadmitted.o'] = 'b' * 64
            elif change == 'missing':
                drift.pop('linux-native-prefix/source/fixture/input.c')
            else:
                drift['linux-native-prefix/source/fixture/input.c'] = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'missing, extra or changed'):
                validator.validate_native_source_bindings(receipt, provenance, self.receipt, self.private, drift)

    def test_package_gate_refuses_unsealed_extra_binding_and_wrong_prefix(self):
        receipt, provenance, files = self.package_binding()
        changed = copy.deepcopy(provenance)
        changed['nativeSourceBindings']['b' * 64] = {}
        with self.assertRaisesRegex(ValueError, 'differs from sealed candidate'):
            validator.validate_native_source_bindings(receipt, changed, self.receipt, self.private, files)
        receipt = {'nativeSourceBindings': changed['nativeSourceBindings']}
        with self.assertRaisesRegex(ValueError, 'incomplete or has extra'):
            validator.validate_native_source_bindings(receipt, changed, self.receipt, self.private, files)
        receipt, provenance, files = self.package_binding()
        binding = next(iter(provenance['nativeSourceBindings'].values()))
        binding['prefixBinary'] = 'prefix/lib/unrelated.so'
        receipt['nativeSourceBindings'] = provenance['nativeSourceBindings']
        with self.assertRaisesRegex(ValueError, 'differs from admitted prefix'):
            validator.validate_native_source_bindings(receipt, provenance, self.receipt, self.private, files)

    def test_completed_helper_proof_is_returned_without_a_status_wrapper(self):
        raw = self.private / 'compiled-source/raw'
        raw.mkdir(parents=True)
        (raw / 'link.map').write_bytes(b'fixture link map')
        proof = {'retainedFiles': {'fixture': 'b' * 64}, 'repositoryInputs': {}}
        with patch.object(compiled_source, 'collect_compiled_source', return_value=proof) as collect:
            result = linux.compiled_source_evidence(self.root / 'build', self.private, 'a' * 40, [])
        self.assertEqual(result, proof)
        self.assertEqual(collect.call_args.kwargs['helper_path'], 'compiled-source/helper-final')
        with patch.object(compiled_source, 'collect_compiled_source', side_effect=ValueError('unknown header origin')):
            result = linux.compiled_source_evidence(self.root / 'build', self.private, 'a' * 40, [])
        self.assertEqual(result, {'status': 'UNVERIFIED', 'reason': 'unknown header origin'})


if __name__ == '__main__':
    unittest.main()
