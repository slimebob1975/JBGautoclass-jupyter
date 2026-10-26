"""Revision 197: persisted dialect and backwards compatibility."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVModelConfiguration import save_csv_model_configuration, load_csv_model_configuration
from Storage.CSVConfiguredRun import prepare_configured_csv_run


class CSVFormatPersistenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = FileArtifactRepository(Path(temp.name) / 'archive')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                         'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.dataset,
                                BytesIO(b'uid;label;value\n001;A;hello\n002;B;world\n'))

    def save(self, **kwargs):
        return save_csv_model_configuration(self.repo, self.scope, self.dataset.artifact_id,
                                            id_column='uid', class_column='label',
                                            data_columns=('value',), **kwargs)

    def test_saved_format_used_by_preparation_and_roundtrips(self):
        dialect = {'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8-sig'}
        config = self.save(csv_format=dialect)
        loaded = load_csv_model_configuration(self.repo, self.scope, config.configuration_id)
        self.assertEqual(loaded.csv_format, dialect)
        self.assertEqual(prepare_configured_csv_run(self.repo, self.scope, config.configuration_id).row_count, 2)

    def test_conflicting_runtime_override_is_rejected(self):
        config = self.save(csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})
        with self.assertRaisesRegex(ValueError, 'CSV format'):
            prepare_configured_csv_run(self.repo, self.scope, config.configuration_id, delimiter=',')

    def test_legacy_configuration_retains_explicit_override(self):
        config = self.save()
        self.assertIsNone(load_csv_model_configuration(self.repo, self.scope, config.configuration_id).csv_format)
        self.assertEqual(prepare_configured_csv_run(self.repo, self.scope, config.configuration_id,
                                                    delimiter=';').row_count, 2)

    def test_invalid_or_unexpected_format_settings_rejected(self):
        for value in ({'delimiter': ';'}, {'delimiter': ';', 'quotechar': ';', 'encoding': 'utf-8'},
                      {'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8', 'path': 'private'},
                      {'delimiter': ';', 'quotechar': '"', 'encoding': 'unknown-codec'}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Invalid CSV format'):
                self.save(csv_format=value)


if __name__ == '__main__':
    unittest.main()
