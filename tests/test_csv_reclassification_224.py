"""Synthetic-only regression tests for CSV correction overlays."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import csv
import unittest

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from Storage.CSVReclassification import correct_csv_class, read_corrected_csv
from Storage.FileArtifactRepository import FileArtifactRepository


class CSVReclassification224(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = FileArtifactRepository(Path(self.tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.original = b'uid;label;value\n005;no;1\nA-17;yes;2\n'
        self.repo.save_artifact(self.scope, self.ref, BytesIO(self.original))
        c = save_csv_model_configuration(self.repo, self.scope, self.ref.artifact_id,
                                         id_column='uid', class_column='label',
                                         data_columns=['value'],
                                         csv_format={'delimiter': ';', 'quotechar': '"',
                                                     'encoding': 'utf-8-sig'})
        self.config_id = c.configuration_id

    def test_publishes_new_dataset_and_keeps_original(self):
        new_ref = correct_csv_class(self.repo, self.scope, self.config_id, '005', 'yes',
                                    allowed_classes=['no', 'yes'])
        self.assertEqual(new_ref.dataset_id, self.ref.dataset_id)
        self.assertEqual(new_ref.kind, ArtifactKind.MANIFEST)
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as f:
            self.assertEqual(f.read(), self.original)
        rows = list(csv.DictReader(read_corrected_csv(self.repo, self.scope, self.config_id).decode('utf-8-sig').splitlines(), delimiter=';'))
        self.assertEqual([r['uid'] for r in rows], ['005', 'A-17'])
        self.assertEqual([r['label'] for r in rows], ['yes', 'yes'])

    def test_opaque_id_and_original_config_not_mutated(self):
        ref = correct_csv_class(self.repo, self.scope, self.config_id, 'A-17', 'no',
                                allowed_classes=('yes', 'no'))
        self.assertNotEqual(ref.artifact_id, self.ref.artifact_id)
        self.assertEqual(ref.kind, ArtifactKind.MANIFEST)
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as f:
            self.assertEqual(f.read(), self.original)

    def test_invalid_row_or_label_does_not_publish(self):
        before = len(self.repo.list_artifacts(self.scope))
        for key, label in [('missing', 'yes'), ('005', 'unknown'), ('005', 'no'), ('', 'yes')]:
            with self.subTest(key=key, label=label):
                with self.assertRaises(ValueError):
                    correct_csv_class(self.repo, self.scope, self.config_id, key, label,
                                      allowed_classes=('yes', 'no'))
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), before)

    def test_wrong_scope_fails_closed(self):
        other = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises((ValueError, FileNotFoundError)):
            correct_csv_class(self.repo, other, self.config_id, '005', 'yes',
                              allowed_classes=('yes', 'no'))

    def test_duplicate_ids_rejected_without_publication(self):
        bad = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, bad, BytesIO(b'uid;label;value\n005;no;1\n005;yes;2\n'))
        cfg = save_csv_model_configuration(self.repo, self.scope, bad.artifact_id,
                                          id_column='uid', class_column='label',
                                          data_columns=['value'], csv_format={
                                              'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8-sig'})
        before = len(self.repo.list_artifacts(self.scope))
        with self.assertRaises(ValueError):
            correct_csv_class(self.repo, self.scope, cfg.configuration_id, '005', 'yes',
                              allowed_classes=('yes', 'no'))
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), before)


if __name__ == '__main__':
    unittest.main()
