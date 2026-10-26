"""Regression tests for artifact-backed CSV configuration and run linkage."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVConfiguredRun import prepare_configured_csv_run
from Storage.CSVRunArtifacts import publish_csv_run


class ConfiguredCSVRunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = FileArtifactRepository(Path(tmp.name) / 'archive')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                         'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.dataset,
                                BytesIO(b'uid;target;x;message\n001;A;12;hello\n002;B;24;world\n'))

    def config(self, **roles):
        return save_csv_model_configuration(self.repo, self.scope, self.dataset.artifact_id,
                                            **roles)

    def test_prepares_and_links_manifest_without_leaking_roles(self):
        cfg = self.config(id_column='uid', class_column='target',
                          data_columns=('x',), text_columns=('message',))
        prepared = prepare_configured_csv_run(self.repo, self.scope, cfg.configuration_id,
                                              delimiter=';')
        self.assertEqual(prepared.row_count, 2)
        self.assertEqual(prepared.context.dataset_id, cfg.dataset_id)
        result = publish_csv_run(self.repo, self.scope, prepared.context,
                                 configuration_id=cfg.configuration_id)
        with self.repo.open_artifact(self.scope, result.manifest.artifact_id) as f:
            raw = f.read().decode()
        self.assertEqual(json.loads(raw)['configuration_id'], cfg.configuration_id)
        for private in ('uid', 'target', 'message', 'hello'):
            self.assertNotIn(private, raw)

    def test_missing_actual_column_rejected(self):
        cfg = self.config(id_column='uid', class_column='target', data_columns=('not_in_csv',))
        with self.assertRaisesRegex(ValueError, 'Invalid CSV data'):
            prepare_configured_csv_run(self.repo, self.scope, cfg.configuration_id, delimiter=';')

    def test_wrong_delimiter_rejected(self):
        cfg = self.config(id_column='uid', class_column='target', data_columns=('x',))
        with self.assertRaisesRegex(ValueError, 'Invalid CSV data'):
            prepare_configured_csv_run(self.repo, self.scope, cfg.configuration_id)

    def test_mismatched_configuration_cannot_be_published(self):
        a = self.config(id_column='uid', class_column='target', data_columns=('x',))
        b = self.config(id_column='uid', class_column='x', data_columns=('target',))
        context = prepare_configured_csv_run(self.repo, self.scope, a.configuration_id,
                                             delimiter=';').context
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            publish_csv_run(self.repo, self.scope, context, configuration_id=b.configuration_id)

    def test_cross_scope_isolated(self):
        cfg = self.config(id_column='uid', class_column='target', data_columns=('x',))
        other = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises(FileNotFoundError):
            prepare_configured_csv_run(self.repo, other, cfg.configuration_id, delimiter=';')


if __name__ == '__main__':
    unittest.main()
