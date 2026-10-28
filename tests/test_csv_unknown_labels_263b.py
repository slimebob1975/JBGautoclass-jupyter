"""263b: CSV-only unknown-class tokens, tested against real file artifacts."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.FileArtifactRepository import FileArtifactRepository


class ConfigStub:
    def __init__(self, train, predict):
        self.train, self.predict = train, predict

    def get_column_names(self): return ['feature', 'uid', 'label']
    def get_id_column_name(self): return 'uid'
    def get_class_column_name(self): return 'label'
    def should_train(self): return self.train
    def should_predict(self): return self.predict
    def get_data_limit(self): return 0


class CSVUnknownLabels263b(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = FileArtifactRepository(Path(tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        lines = ['feature,uid,label']
        for label in ('A', 'B', 'NULL', 'null', ' NuLl ', '', '   ', 'NULLIFICATION', 'C'):
            lines.append(f'NULL,{len(lines)},{label}')
        self.raw = ('\n'.join(lines) + '\n').encode('utf-8')
        self.repo.save_artifact(self.scope, self.ref, BytesIO(self.raw))
        self.roles = save_csv_model_configuration(
            self.repo, self.scope, self.ref.artifact_id, id_column='uid',
            class_column='label', data_columns=('feature',),
            csv_format={'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8'})

    def reader(self, train, predict):
        return CSVTrainingAdapter(self.repo, self.scope,
                                  self.roles.configuration_id, ConfigStub(train, predict))

    def test_train_excludes_all_unknown_label_tokens(self):
        rows = self.reader(True, False).get_dataset()
        self.assertEqual(len(rows), 4)
        self.assertEqual([r[2] for r in rows], ['A', 'B', 'NULLIFICATION', 'C'])
        self.assertTrue(all(r[0] == 'NULL' for r in rows))

    def test_predict_selects_only_unknown_label_tokens(self):
        rows = self.reader(False, True).get_dataset()
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r[2] == '' for r in rows))

    def test_combined_mode_returns_all_rows_with_canonical_unknown(self):
        rows = self.reader(True, True).get_dataset()
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(row[2] == '' for row in rows), 5)

    def test_original_artifact_is_not_modified(self):
        self.reader(True, False).get_dataset()
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            self.assertEqual(stream.read(), self.raw)

    def test_iris_partition_has_135_known_and_15_unknown(self):
        lines = ['feature,uid,label']
        for cls in ('setosa', 'versicolor', 'virginica'):
            for j in range(50):
                lines.append(f'{j},{cls}-{j},{cls if j < 45 else "NULL"}')
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.ref, BytesIO(('\n'.join(lines) + '\n').encode()))
        self.roles = save_csv_model_configuration(
            self.repo, self.scope, self.ref.artifact_id, id_column='uid',
            class_column='label', data_columns=('feature',),
            csv_format={'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8'})
        self.assertEqual(len(self.reader(True, False).get_dataset()), 135)
        self.assertEqual(len(self.reader(False, True).get_dataset()), 15)


if __name__ == '__main__':
    unittest.main()
