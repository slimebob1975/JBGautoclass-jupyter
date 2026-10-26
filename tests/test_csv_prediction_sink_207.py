"""SQL-independent prediction persistence regression tests (revision 207)."""
import json
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, StorageScope
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.CSVRunContext import prepare_csv_run
from Storage.CSVPredictionResults import CSVPredictionResultSink
from Storage.FileArtifactRepository import FileArtifactRepository


class Rate(Enum):
    PROBABILITY = 1


class CSVPredictionSinkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.repo = FileArtifactRepository(root / 'artifacts')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        csv = root / 'synthetic.csv'
        csv.write_text('uid;class;feature\n001;A;one\n002;B;two\n', encoding='utf-8')
        registered = register_csv_dataset(self.repo, self.scope, csv,
            id_column='uid', class_column='class', feature_columns=('feature',), delimiter=';')
        self.ctx = prepare_csv_run(self.repo, self.scope, registered.reference.artifact_id,
            id_column='uid', class_column='class', feature_columns=('feature',))
        self.sink = CSVPredictionResultSink(self.repo, self.scope, self.ctx)

    def test_persists_private_rows_with_public_manifest_only_identifiers(self):
        result = self.sink.save_prediction_data(
            [{'key': '001', 'prediction': 'positive', 'rate': 0.7, 'probabilities': '0.3,0.7'}],
            Rate.PROBABILITY, 'model', ['negative', 'positive'])
        self.assertEqual(result['results']['row_count'], 1)
        refs = self.repo.list_artifacts(self.scope, run_id=result['results']['run_id'])
        self.assertEqual({r.kind for r in refs}, {ArtifactKind.MANIFEST, ArtifactKind.PREDICTIONS})
        manifest_ref = next(ref for ref in refs if ref.kind == ArtifactKind.MANIFEST)
        data_ref = next(ref for ref in refs if ref.kind == ArtifactKind.PREDICTIONS)
        with self.repo.open_artifact(self.scope, manifest_ref.artifact_id) as f:
            manifest = json.load(f)
        with self.repo.open_artifact(self.scope, data_ref.artifact_id) as f:
            private = json.load(f)
        self.assertEqual(private['rows'][0]['key'], '001')
        self.assertNotIn('positive', str(manifest))
        self.assertNotIn('001', str(manifest))
        self.assertNotIn('feature', str(manifest))
        self.assertEqual(manifest['predictions_artifact_id'], data_ref.artifact_id)

    def test_invalid_rows_do_not_publish_artifacts(self):
        before = len(self.repo.list_artifacts(self.scope))
        for rows in ([{'key': 'x', 'prediction': 'yes', 'rate': float('nan'), 'probabilities': ''}],
                     [{'key': '1', 'prediction': 'yes', 'rate': .5, 'probabilities': ''},
                      {'key': '1', 'prediction': 'no', 'rate': .4, 'probabilities': ''}],
                     [{'key': '1', 'prediction': 'yes', 'rate': .5}],
                     [{'key': True, 'prediction': 'yes', 'rate': .5, 'probabilities': ''}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.sink.save_prediction_data(rows, Rate.PROBABILITY, 'model', [])
        self.assertEqual(before, len(self.repo.list_artifacts(self.scope)))

    def test_isolated_scope_and_wrong_dataset_rejected(self):
        other = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises(ValueError):
            CSVPredictionResultSink(self.repo, other, self.ctx)
        from dataclasses import replace
        sink = CSVPredictionResultSink(self.repo, self.scope,
                                       replace(self.ctx, dataset_id=str(uuid4())))
        with self.assertRaises(ValueError):
            sink.save_prediction_data([], Rate.PROBABILITY, 'model', [])

    def test_manifest_not_published_when_payload_write_fails(self):
        class FailingRepo:
            def get_reference(_, scope, artifact_id):
                return self.repo.get_reference(scope, artifact_id)
            def save_artifact(_, scope, ref, data):
                raise OSError('write failed')
        sink = CSVPredictionResultSink(FailingRepo(), self.scope, self.ctx)
        with self.assertRaises(OSError):
            sink.save_prediction_data([], Rate.PROBABILITY, 'model', [])
        self.assertFalse(any(ref.run_id for ref in self.repo.list_artifacts(self.scope)))

    def test_existing_sql_prediction_path_not_modified(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        handler = (root / 'src' / 'JBGHandler.py').read_text(encoding='utf-8')
        sql = (root / 'src' / 'SQLDataLayer.py').read_text(encoding='utf-8')
        self.assertIn('self.handler.datalayer.save_prediction_data(', handler)
        self.assertIn('def save_prediction_data(', sql)


if __name__ == '__main__':
    unittest.main()
