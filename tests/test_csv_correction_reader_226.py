"""CSV correction snapshot tests; synthetic records only, no SQL or GUI."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVReclassification import correct_csv_class, read_corrected_csv
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.FileArtifactRepository import FileArtifactRepository


class StubConfig:
    def __init__(self, train, predict):
        self.train, self.predict = train, predict

    def get_column_names(self): return ['feature', 'uid', 'label']
    def get_id_column_name(self): return 'uid'
    def get_class_column_name(self): return 'label'
    def should_train(self): return self.train
    def should_predict(self): return self.predict
    def get_data_limit(self): return 0


class CSVCorrectionReader226(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = FileArtifactRepository(Path(temp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.original = b'feature;uid;label\n1;005;A\n2;ABC-17;\n3;009;B\n'
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.ref, BytesIO(self.original))
        self.roles = save_csv_model_configuration(
            self.repo, self.scope, self.ref.artifact_id,
            id_column='uid', class_column='label', data_columns=('feature',),
            csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})

    def adapter(self, *, train=True, predict=True):
        return CSVTrainingAdapter(self.repo, self.scope, self.roles.configuration_id,
                                  StubConfig(train, predict))

    def edit(self, row, label):
        return correct_csv_class(self.repo, self.scope, self.roles.configuration_id,
                                 row, label, allowed_classes=('A', 'B'))

    def test_new_adapters_read_latest_overlay_with_text_ids(self):
        self.edit('005', 'B')
        self.edit('ABC-17', 'A')
        adapter = self.adapter()
        self.assertEqual(adapter.get_dataset(), [('1', '005', 'B'),
                                                  ('2', 'ABC-17', 'A'), ('3', '009', 'B')])
        self.assertEqual(adapter.load_dataframe()['uid'].tolist(), ['005', 'ABC-17', '009'])
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            self.assertEqual(stream.read(), self.original)

    def test_run_without_edits_stays_on_original_after_new_edits(self):
        original_run = self.adapter()
        self.assertIsNone(original_run.correction_revision)
        self.edit('005', 'B')
        self.assertEqual(original_run.get_dataset()[0], ('1', '005', 'A'))
        self.assertEqual(self.adapter().get_dataset()[0], ('1', '005', 'B'))

    def test_run_with_edits_stays_on_pinned_revision(self):
        first = self.edit('005', 'B')
        pinned = self.adapter()
        self.assertEqual(pinned.correction_revision, first.artifact_id)
        self.edit('005', 'A')
        self.edit('ABC-17', 'A')
        self.assertEqual(pinned.get_dataset(), [('1', '005', 'B'),
                                                ('2', 'ABC-17', ''), ('3', '009', 'B')])
        self.assertEqual(self.adapter().get_dataset(), [('1', '005', 'A'),
                                                        ('2', 'ABC-17', 'A'), ('3', '009', 'B')])

    def test_label_filters_apply_after_overlay(self):
        self.edit('ABC-17', 'A')
        self.assertEqual(len(self.adapter(train=True, predict=False).get_dataset()), 3)
        self.assertEqual(self.adapter(train=False, predict=True).get_dataset(), [])

    def test_unmodified_export_is_original_and_revision_is_explicit(self):
        first = self.edit('005', 'B')
        self.edit('005', 'A')
        self.assertIn(b'1;005;B', read_corrected_csv(
            self.repo, self.scope, self.roles.configuration_id, revision=first.artifact_id))
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            self.assertEqual(self.original, stream.read())
        with self.assertRaises(ValueError):
            read_corrected_csv(self.repo, self.scope, self.roles.configuration_id,
                               revision=str(uuid4()))


if __name__ == '__main__':
    unittest.main()
