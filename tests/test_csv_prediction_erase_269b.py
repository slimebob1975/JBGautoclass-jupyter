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
from Storage.CSVPredictionErasure import erase_csv_prediction_run


class Rate:
    name = 'TEST'


class ErasePredictionTests(unittest.TestCase):
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

    def test_erases_only_selected_run(self):
        ref = self.repo.get_reference(self.scope, self.manifest_id)
        self.assertEqual(ref.kind, ArtifactKind.MANIFEST)
        other = self.repo.get_reference(self.scope, self.dataset_artifact)
        self.assertEqual(other.kind, ArtifactKind.DATASET)
        erase_csv_prediction_run(self.repo, self.scope, self.dataset_artifact, self.manifest_id)
        self.assertEqual(list_csv_prediction_runs(self.repo, self.scope, self.dataset_artifact), ())
        self.repo.get_reference(self.scope, self.dataset_artifact)
        self.repo.get_reference(self.scope, self.config.configuration_id)

    def test_rejects_foreign_project_and_dataset(self):
        foreign = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises((ValueError, FileNotFoundError)):
            erase_csv_prediction_run(self.repo, foreign, self.dataset_artifact, self.manifest_id)
        with self.assertRaises((ValueError, FileNotFoundError)):
            erase_csv_prediction_run(self.repo, self.scope, str(uuid4()), self.manifest_id)
        self.repo.get_reference(self.scope, self.manifest_id)

    def test_cannot_erase_twice(self):
        erase_csv_prediction_run(self.repo, self.scope, self.dataset_artifact, self.manifest_id)
        with self.assertRaises((ValueError, FileNotFoundError)):
            erase_csv_prediction_run(self.repo, self.scope, self.dataset_artifact, self.manifest_id)

    def test_gui_confirmation_and_color_wired(self):
        widget = (Path(__file__).resolve().parents[1] / 'src/GUI/Widgets.py').read_text()
        panel = (Path(__file__).resolve().parents[1] / 'src/GUI/CSVSourcePanel.py').read_text()
        self.assertIn('self._csv_source_panel.erase_predictions_button.on_click(self._csv_erase_prediction_run)', widget)
        self.assertIn('panel._erase_predictions_confirmation != key', widget)
        self.assertIn('set_action_state(button, "warning")', widget)
        self.assertIn('self.erase_predictions_button', panel)
