"""Synthetic, SQL-independent checks for explicit CSV editing and export."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import unittest

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVReclassification import list_csv_corrections
from Storage.CSVReclassificationWorkflow import reclassify_csv_row, corrected_csv_download
from Storage.FileArtifactRepository import FileArtifactRepository


class CSVReclassificationWorkflow227(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repository = FileArtifactRepository(Path(tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.original = b'uid;label;value\n005;no;1\nA-17;yes;2\n'
        reference = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                      'text/csv', dataset_id=str(uuid4()))
        self.reference = reference
        self.repository.save_artifact(self.scope, reference, BytesIO(self.original))
        config = save_csv_model_configuration(self.repository, self.scope, reference.artifact_id,
                                              id_column='uid', class_column='label',
                                              data_columns=['value'], csv_format={
                                                  'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8-sig'})
        self.config_id = config.configuration_id

    def edit(self, row_id, label, confirmed=True):
        return reclassify_csv_row(self.repository, self.scope, self.config_id,
                                  row_id, label, allowed_classes=('yes', 'no'),
                                  confirmed=confirmed)

    def test_confirmation_required_without_side_effects(self):
        with self.assertRaises(ValueError):
            self.edit('005', 'yes', confirmed=False)
        self.assertEqual(list_csv_corrections(self.repository, self.scope, self.config_id), ())

    def test_download_latest_with_edits_and_original_unchanged(self):
        self.edit('005', 'yes')
        data, encoding = corrected_csv_download(self.repository, self.scope, self.config_id)
        self.assertEqual(encoding, 'utf-8-sig')
        self.assertIn(b'005;yes;1', data)
        with self.repository.open_artifact(self.scope, self.reference.artifact_id) as stream:
            self.assertEqual(stream.read(), self.original)

    def test_historical_revision_download(self):
        first = self.edit('005', 'yes')
        self.edit('005', 'no')
        old, _ = corrected_csv_download(self.repository, self.scope, self.config_id,
                                        revision=first.artifact_id)
        latest, _ = corrected_csv_download(self.repository, self.scope, self.config_id)
        self.assertIn(b'005;yes;1', old)
        self.assertIn(b'005;no;1', latest)
        self.assertEqual(len([r for r in self.repository.list_artifacts(self.scope)
                              if r.kind == ArtifactKind.DATASET]), 1)

    def test_unknown_revision_is_rejected(self):
        with self.assertRaises(ValueError):
            corrected_csv_download(self.repository, self.scope, self.config_id,
                                   revision=str(uuid4()))

    def test_scoped_edit_and_download_fail_closed(self):
        wrong_scope = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises((ValueError, FileNotFoundError)):
            reclassify_csv_row(self.repository, wrong_scope, self.config_id, '005', 'yes',
                               allowed_classes=('yes', 'no'), confirmed=True)
        with self.assertRaises((ValueError, FileNotFoundError)):
            corrected_csv_download(self.repository, wrong_scope, self.config_id)


if __name__ == '__main__':
    unittest.main()
