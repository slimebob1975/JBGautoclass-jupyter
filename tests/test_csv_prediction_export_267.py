"""Synthetic scope, manifest, and CSV join regression checks for revision 267."""
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope, StorageKind
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVRunContext import prepare_csv_run
from Storage.CSVPredictionResults import CSVPredictionResultSink
from Storage.CSVPredictionExport import list_csv_prediction_runs, export_csv_prediction_run


class Rate:
    name = 'TEST'


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = FileArtifactRepository(Path(self.temp.name) / 'objects')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset_id = str(uuid4())
        self.dataset_artifact = str(uuid4())
        self.csv_bytes = b'id,class,data\n001,,10\n002,,20\n003,known,30\n'
        ref = ArtifactReference(self.scope, self.dataset_artifact, ArtifactKind.DATASET,
                                'text/csv', dataset_id=self.dataset_id)
        self.repo.save_artifact(self.scope, ref, BytesIO(self.csv_bytes))
        from Storage.CSVModelConfiguration import save_csv_model_configuration
        self.config = save_csv_model_configuration(self.repo, self.scope, self.dataset_artifact,
                                                   id_column='id', class_column='class', data_columns=('data',),
                                                   csv_format={'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8-sig'})
        context = prepare_csv_run(self.repo, self.scope, self.dataset_artifact,
                                  id_column='id', class_column='class', feature_columns=('data',))
        sink = CSVPredictionResultSink(self.repo, self.scope, context)
        info = sink.save_prediction_data([
            {'key': '001', 'prediction': 'A', 'rate': .9, 'probabilities': '0.9,0.1'},
            {'key': '002', 'prediction': 'B', 'rate': .8, 'probabilities': '0.2,0.8'}
        ], Rate(), 'model', ('A', 'B'))
        self.manifest_id = info['results']['artifact_id']

    def test_list_and_export_default(self):
        entries = list_csv_prediction_runs(self.repo, self.scope, self.dataset_artifact)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][1], self.manifest_id)
        result = export_csv_prediction_run(self.repo, self.scope, self.dataset_artifact, self.manifest_id)
        rows = list(csv.DictReader(StringIO(result.decode('utf-8-sig'))))
        self.assertEqual([r['id'] for r in rows], ['001', '002'])
        self.assertEqual([r['predicted_class'] for r in rows], ['A', 'B'])
        self.assertNotIn('data', rows[0])

    def test_join_uses_unique_string_id(self):
        data = export_csv_prediction_run(self.repo, self.scope, self.dataset_artifact,
                                         self.manifest_id, include_original=True)
        rows = list(csv.DictReader(StringIO(data.decode('utf-8-sig'))))
        self.assertEqual([r['id'] for r in rows], ['001', '002'])
        self.assertEqual([r['data'] for r in rows], ['10', '20'])
        self.assertEqual(len(rows), 2)

    def test_wrong_scope_and_dataset_blocked(self):
        wrong = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises((ValueError, FileNotFoundError)):
            export_csv_prediction_run(self.repo, wrong, self.dataset_artifact, self.manifest_id)
        other_id = str(uuid4())
        self.repo.save_artifact(self.scope, ArtifactReference(self.scope, other_id, ArtifactKind.DATASET,
                                'text/csv', dataset_id=str(uuid4())), BytesIO(self.csv_bytes))
        with self.assertRaises(ValueError):
            export_csv_prediction_run(self.repo, self.scope, other_id, self.manifest_id)

    def test_deleted_or_incomplete_run_not_discoverable(self):
        self.repo.delete_artifact(self.scope, self.manifest_id)
        self.assertEqual(list_csv_prediction_runs(self.repo, self.scope, self.dataset_artifact), ())

    def test_duplicate_source_id_rejected(self):
        # Another immutable source artifact with duplicate IDs cannot silently multiply joined rows.
        duplicate = str(uuid4())
        did = str(uuid4())
        self.repo.save_artifact(self.scope, ArtifactReference(self.scope, duplicate, ArtifactKind.DATASET,
                                'text/csv', dataset_id=did), BytesIO(b'id,class,data\n001,,10\n001,,11\n'))
        # A separate complete run for this dataset is required, thus wrong dataset guard fires first.
        with self.assertRaises(ValueError):
            export_csv_prediction_run(self.repo, self.scope, duplicate, self.manifest_id, include_original=True)


if __name__ == '__main__':
    unittest.main()
