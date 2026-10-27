"""Synthetic CSV label and run-planning tests; no SQL or ML execution."""
import tempfile
import unittest
from datetime import timezone
from pathlib import Path
from uuid import uuid4
from Storage.ArtifactContract import StorageScope, StorageKind
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.CSVDatasetCatalog import list_csv_dataset_labels
from Storage.CSVRunContext import prepare_csv_run


class Revision193Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = FileArtifactRepository(self.root / 'archive')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.csv = self.root / 'sample.csv'
        self.csv.write_text('uid;label;x\n001;A;first\n002;B;second\n', encoding='utf-8')

    def register(self, name='sample.csv'):
        return register_csv_dataset(self.repo, self.scope, self.csv,
            id_column='uid', class_column='label', feature_columns=('x',),
            delimiter=';', original_name=name)

    def test_filename_timestamp_and_reopen(self):
        registered = self.register()
        rows = list_csv_dataset_labels(FileArtifactRepository(self.root / 'archive'), self.scope, display_timezone=timezone.utc)
        self.assertEqual(len(rows), 1)
        self.assertIn('sample.csv', rows[0][0])
        self.assertIn('UTC+0000', rows[0][0])
        self.assertEqual(rows[0][1], registered.reference.artifact_id)
        self.assertNotIn(str(self.root), str(rows))

    def test_legacy_entries_have_safe_fallback(self):
        register_csv_dataset(self.repo, self.scope, self.csv,
            id_column='uid', class_column='label', feature_columns=('x',), delimiter=';')
        self.assertTrue(list_csv_dataset_labels(self.repo, self.scope)[0][0].startswith('Dataset '))

    def test_scope_and_path_sanitization(self):
        self.register(r'C:\clients\private\source.csv')
        labels = list_csv_dataset_labels(self.repo, self.scope)
        self.assertIn('source.csv', labels[0][0])
        self.assertNotIn('clients', labels[0][0])
        other = StorageScope(str(uuid4()), str(uuid4()))
        self.assertEqual(list_csv_dataset_labels(self.repo, other), ())

    def test_run_context_and_reject_cross_scope(self):
        registered = self.register()
        ctx = prepare_csv_run(self.repo, self.scope, registered.reference.artifact_id,
            id_column='uid', class_column='label', feature_columns=('x',))
        self.assertEqual(ctx.storage_kind, StorageKind.FILES)
        self.assertEqual(ctx.dataset_id, registered.dataset_id)
        with self.assertRaises(FileNotFoundError):
            prepare_csv_run(self.repo, StorageScope(str(uuid4()), str(uuid4())),
                registered.reference.artifact_id, id_column='uid', class_column='label', feature_columns=('x',))
        with self.assertRaises(ValueError):
            prepare_csv_run(self.repo, self.scope, registered.reference.artifact_id,
                id_column='uid', class_column='uid', feature_columns=('x',))

if __name__ == '__main__':
    unittest.main()
