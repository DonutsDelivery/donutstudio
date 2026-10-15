#!/usr/bin/env python3
"""Source archive admission fixtures; no compiler or native app is invoked."""
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/video-helper-dependencies'))
SPEC = importlib.util.spec_from_file_location('macos_archive_builder',
    ROOT / 'scripts/video-helper-dependencies/build-macos-source.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ArchiveExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / 'source'
        self.root.mkdir()
        self.archive = self.base / 'source.tar'

    def tearDown(self):
        self.temporary.cleanup()

    def write_archive(self, entries):
        with tarfile.open(self.archive, 'w') as bundle:
            for name, kind, link in entries:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = link
                member.mode = 0o7777
                if kind == tarfile.REGTYPE:
                    member.size = 4
                    bundle.addfile(member, io.BytesIO(b'data'))
                else:
                    bundle.addfile(member)

    def extract_legacy(self):
        with patch.object(tarfile, 'data_filter', None, create=True), \
                patch.object(tarfile.TarFile, 'extractall', side_effect=AssertionError('unfiltered extraction')):
            return builder.extract(self.archive, self.root)

    def test_legacy_regular_files_keep_executable_without_special_modes(self):
        self.write_archive([('codec', tarfile.DIRTYPE, ''),
                            ('codec/configure', tarfile.REGTYPE, '')])
        path = self.extract_legacy()
        self.assertEqual((path / 'configure').read_bytes(), b'data')
        self.assertEqual((path / 'configure').stat().st_mode & 0o7777, 0o755)
        self.assertEqual(self.extract_legacy(), path)

    def test_traversal_absolute_and_multiple_roots_are_refused(self):
        for name in ('../escape', '/escape', 'codec/../../escape', 'other/file', ''):
            with self.subTest(name=name):
                self.write_archive([('codec', tarfile.DIRTYPE, ''), (name, tarfile.REGTYPE, '')])
                with self.assertRaises(ValueError):
                    self.extract_legacy()
                self.assertFalse((self.root / 'codec').exists())

    def test_special_file_types_are_refused(self):
        for kind in (tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind):
                self.write_archive([('codec', tarfile.DIRTYPE, ''), ('codec/device', kind, '')])
                with self.assertRaisesRegex(ValueError, 'file type'):
                    self.extract_legacy()

    def test_escaping_links_are_refused(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            for target in ('../../escape', '/escape'):
                with self.subTest(kind=kind, target=target):
                    self.write_archive([('codec', tarfile.DIRTYPE, ''), ('codec/link', kind, target)])
                    with self.assertRaisesRegex(ValueError, 'link'):
                        self.extract_legacy()

    def test_legacy_internal_links_require_filtered_python_before_writes(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            target = 'file' if kind == tarfile.SYMTYPE else 'codec/file'
            self.write_archive([('codec', tarfile.DIRTYPE, ''),
                                ('codec/file', tarfile.REGTYPE, ''), ('codec/link', kind, target)])
            with self.assertRaisesRegex(ValueError, 'normal build wrapper'):
                self.extract_legacy()
            self.assertFalse((self.root / 'codec').exists())

    def test_preexisting_symlink_root_or_ancestor_escape_is_refused_on_cache_reuse(self):
        outside = self.base / 'outside'
        outside.mkdir()
        codec = self.root / 'codec'
        codec.symlink_to(outside, target_is_directory=True)
        self.write_archive([('codec/file', tarfile.REGTYPE, '')])
        with self.assertRaises(ValueError):
            self.extract_legacy()
        codec.unlink()
        codec.mkdir()
        (codec / 'nested').symlink_to(outside, target_is_directory=True)
        self.write_archive([('codec/nested/file', tarfile.REGTYPE, '')])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.extract_legacy()
        self.assertFalse((outside / 'file').exists())

    @unittest.skipUnless(callable(getattr(tarfile, 'data_filter', None)), 'requires tarfile data filter')
    def test_filtered_python_accepts_internal_links(self):
        self.write_archive([('codec', tarfile.DIRTYPE, ''), ('codec/file', tarfile.REGTYPE, ''),
                            ('codec/link', tarfile.SYMTYPE, 'file')])
        path = builder.extract(self.archive, self.root)
        self.assertEqual((path / 'link').read_bytes(), b'data')

    def assert_no_partial_source(self):
        self.assertFalse((self.root / 'codec').exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_legacy_duplicate_and_file_directory_conflicts_never_publish_partial_root(self):
        for entries in (
                [('codec/file', tarfile.REGTYPE, ''), ('codec/file', tarfile.REGTYPE, '')],
                [('codec/file', tarfile.REGTYPE, ''), ('codec/file/nested', tarfile.REGTYPE, '')],
                [('codec/file', tarfile.REGTYPE, ''), ('codec/file', tarfile.DIRTYPE, '')]):
            with self.subTest(entries=entries):
                self.write_archive(entries)
                for attempt in range(2):
                    with self.assertRaises(OSError):
                        self.extract_legacy()
                    self.assert_no_partial_source()
        self.write_archive([('codec/file', tarfile.REGTYPE, '')])
        self.assertEqual((self.extract_legacy() / 'file').read_bytes(), b'data')

    def test_legacy_copy_failure_cleans_owned_staging_and_allows_retry(self):
        self.write_archive([('codec/first', tarfile.REGTYPE, ''), ('codec/second', tarfile.REGTYPE, '')])
        real_copy = builder.shutil.copyfileobj
        calls = 0
        def fail_second_copy(source, output):
            nonlocal calls
            calls += 1
            if calls == 2:
                output.write(b'par')
                raise OSError('injected copy failure')
            real_copy(source, output)
        with patch.object(builder.shutil, 'copyfileobj', side_effect=fail_second_copy):
            with self.assertRaisesRegex(OSError, 'injected copy failure'):
                self.extract_legacy()
        self.assert_no_partial_source()
        self.assertEqual((self.extract_legacy() / 'second').read_bytes(), b'data')

    @unittest.skipUnless(callable(getattr(tarfile, 'data_filter', None)), 'requires tarfile data filter')
    def test_filtered_extraction_failure_cleans_owned_staging_and_allows_retry(self):
        self.write_archive([('codec/file', tarfile.REGTYPE, '')])
        def fail_extract(bundle, destination, **kwargs):
            (Path(destination) / 'codec').mkdir()
            (Path(destination) / 'codec/file').write_bytes(b'partial')
            raise OSError('injected filtered extraction failure')
        with patch.object(tarfile.TarFile, 'extractall', new=fail_extract):
            with self.assertRaisesRegex(OSError, 'injected filtered'):
                builder.extract(self.archive, self.root)
        self.assert_no_partial_source()
        self.assertEqual((builder.extract(self.archive, self.root) / 'file').read_bytes(), b'data')

    def test_unreceipted_old_partial_source_is_preserved_and_refused(self):
        self.write_archive([('codec/file', tarfile.REGTYPE, '')])
        codec = self.root / 'codec'
        codec.mkdir()
        (codec / 'file').write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError, 'Unverified existing source extraction'):
            self.extract_legacy()
        self.assertEqual((codec / 'file').read_bytes(), b'partial')

    def test_receipted_cache_reuse_preserves_generated_files_and_checks_archive_identity(self):
        self.write_archive([('codec/file', tarfile.REGTYPE, '')])
        codec = self.extract_legacy()
        (codec / 'file').write_bytes(b'configured source')
        self.assertEqual(self.extract_legacy(), codec)
        self.write_archive([('codec/other', tarfile.REGTYPE, '')])
        with self.assertRaisesRegex(ValueError, 'Unverified existing source extraction'):
            self.extract_legacy()
        self.assertEqual((codec / 'file').read_bytes(), b'configured source')

    def test_empty_archive_is_refused(self):
        self.write_archive([])
        with self.assertRaisesRegex(ValueError, 'Empty'):
            self.extract_legacy()


if __name__ == '__main__':
    unittest.main()
