"""Revision 202: data-source adapter contract tests, no SQL dependencies."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVTrainingAdapter import CSVTrainingAdapter


class StubConfig:
    def __init__(self, names, train=True, predict=True, limit=100):
        self.names, self.train, self.predict, self.limit = names, train, predict, limit

    def get_column_names(self): return self.names
    def get_id_column_name(self): return 'uid'
    def get_class_column_name(self): return 'target'
    def should_train(self): return self.train
    def should_predict(self): return self.predict
    def get_data_limit(self): return self.limit


class AdapterTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.repository = FileArtifactRepository(Path(td.name) / 'archive')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.repository.save_artifact(self.scope, self.ref, BytesIO(
            b'uid;target;numeric;text\n001;A;12;hello\n002;;23;world\n003;B;34;again\n'))
        self.config = save_csv_model_configuration(
            self.repository, self.scope, self.ref.artifact_id, id_column='uid',
            class_column='target', data_columns=('numeric',), text_columns=('text',),
            csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})

    def adapter(self, **kwargs):
        return CSVTrainingAdapter(self.repository, self.scope, self.config.configuration_id,
                                  StubConfig(['text', 'numeric', 'uid', 'target'], **kwargs))

    def test_order_and_lossless_string_id(self):
        rows = self.adapter().get_dataset()
        self.assertEqual(rows, [('hello', '12', '001', 'A'), ('world', '23', '002', ''),
                                ('again', '34', '003', 'B')])

    def test_modes_and_limit(self):
        self.assertEqual(len(self.adapter(train=True, predict=False).get_dataset()), 2)
        self.assertEqual(self.adapter(train=False, predict=True).get_dataset(),
                         [('world', '23', '002', '')])
        self.assertEqual(len(self.adapter(limit=1).get_dataset()), 1)

    def test_role_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'roles'):
            CSVTrainingAdapter(self.repository, self.scope, self.config.configuration_id,
                               StubConfig(['numeric', 'text', 'uid', 'target']))

    def test_scope_isolated(self):
        with self.assertRaises(FileNotFoundError):
            CSVTrainingAdapter(self.repository, StorageScope(str(uuid4()), str(uuid4())),
                               self.config.configuration_id,
                               StubConfig(['text', 'numeric', 'uid', 'target']))

    def test_invalid_duplicate_id_rejected(self):
        self.repository.save_artifact(self.scope, ArtifactReference(
            self.scope, str(uuid4()), ArtifactKind.DATASET, 'text/csv', dataset_id=str(uuid4())),
            BytesIO(b'uid;target;numeric;text\n001;A;12;hello\n001;B;13;world\n'))
        # Separate invalid dataset and configuration; no reliance on user files.
        refs = [r for r in self.repository.list_artifacts(self.scope) if r.kind == ArtifactKind.DATASET
                and r.artifact_id != self.ref.artifact_id]
        bad = save_csv_model_configuration(self.repository, self.scope, refs[0].artifact_id,
            id_column='uid', class_column='target', data_columns=('numeric',), text_columns=('text',),
            csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})
        adapter = CSVTrainingAdapter(self.repository, self.scope, bad.configuration_id,
                                     StubConfig(['text', 'numeric', 'uid', 'target']))
        with self.assertRaisesRegex(ValueError, 'Unable to prepare'):
            adapter.get_dataset()


if __name__ == '__main__':
    unittest.main()
