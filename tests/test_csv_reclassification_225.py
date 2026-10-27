"""Synthetic tests for immutable base CSV plus compact correction events."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import csv
import unittest

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVReclassification import correct_csv_class, read_corrected_csv, list_csv_corrections
from Storage.FileArtifactRepository import FileArtifactRepository


class CSVCorrectionOverlay225(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = FileArtifactRepository(Path(tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.original = b'uid;label;value\n005;no;1\nA-17;yes;2\n'
        ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                'text/csv', dataset_id=str(uuid4()))
        self.ref = ref
        self.repo.save_artifact(self.scope, ref, BytesIO(self.original))
        config = save_csv_model_configuration(self.repo, self.scope, ref.artifact_id,
                                              id_column='uid', class_column='label',
                                              data_columns=['value'], csv_format={
                                                  'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8-sig'})
        self.config_id = config.configuration_id

    def change(self, key, label):
        return correct_csv_class(self.repo, self.scope, self.config_id, key, label,
                                 allowed_classes=('yes', 'no'))

    def test_multiple_edits_do_not_duplicate_csv(self):
        one = self.change('005', 'yes')
        two = self.change('A-17', 'no')
        self.assertEqual(one.kind, ArtifactKind.MANIFEST)
        self.assertEqual(two.dataset_id, self.ref.dataset_id)
        self.assertEqual(len([r for r in self.repo.list_artifacts(self.scope) if r.kind == ArtifactKind.DATASET]), 1)
        self.assertEqual(list_csv_corrections(self.repo, self.scope, self.config_id),
                         (one.artifact_id, two.artifact_id))
        rows = list(csv.DictReader(read_corrected_csv(self.repo, self.scope, self.config_id).decode('utf-8-sig').splitlines(), delimiter=';'))
        self.assertEqual([(row['uid'], row['label'], row['value']) for row in rows],
                         [('005', 'yes', '1'), ('A-17', 'no', '2')])
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            self.assertEqual(stream.read(), self.original)

    def test_same_row_repeated_edit_and_historical_revision(self):
        first = self.change('005', 'yes')
        second = self.change('005', 'no')
        self.assertIn(b'005;yes;1', read_corrected_csv(self.repo, self.scope, self.config_id,
                                                      revision=first.artifact_id))
        self.assertIn(b'005;no;1', read_corrected_csv(self.repo, self.scope, self.config_id,
                                                     revision=second.artifact_id))
        self.assertEqual(len([r for r in self.repo.list_artifacts(self.scope) if r.kind == ArtifactKind.DATASET]), 1)

    def test_invalid_input_creates_no_events(self):
        for key, label in [('missing', 'yes'), ('005', 'other'), ('005', 'no'), ('', 'yes')]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.change(key, label)
        self.assertEqual(list_csv_corrections(self.repo, self.scope, self.config_id), ())

    def test_scopes_and_unknown_revisions_fail_closed(self):
        with self.assertRaises(ValueError):
            read_corrected_csv(self.repo, self.scope, self.config_id, revision=str(uuid4()))
        with self.assertRaises((ValueError, FileNotFoundError)):
            correct_csv_class(self.repo, StorageScope(str(uuid4()), str(uuid4())),
                              self.config_id, '005', 'yes', allowed_classes=('yes', 'no'))

    def test_original_is_returned_byte_for_byte_before_edits(self):
        self.assertEqual(read_corrected_csv(self.repo, self.scope, self.config_id), self.original)


if __name__ == '__main__':
    unittest.main()
