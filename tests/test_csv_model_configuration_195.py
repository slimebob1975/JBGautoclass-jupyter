"""Regression tests for separate dataset and private model-configuration identity."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVModelConfiguration import save_csv_model_configuration, load_csv_model_configuration


class CSVModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = FileArtifactRepository(Path(temp.name) / 'archive')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                         'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.dataset, BytesIO(b'uid,label,x,y\n001,A,1,2\n'))

    def save(self, **kwargs):
        return save_csv_model_configuration(self.repo, self.scope, self.dataset.artifact_id,
                                            id_column='uid', **kwargs)

    def test_same_dataset_different_configurations_and_roundtrip(self):
        a = self.save(class_column='label', data_columns=('x',), text_columns=('y',))
        b = self.save(class_column='x', data_columns=('label',), text_columns=('y',))
        self.assertNotEqual(a.configuration_id, b.configuration_id)
        self.assertEqual(a.dataset_id, b.dataset_id)
        self.assertEqual(load_csv_model_configuration(self.repo, self.scope, a.configuration_id), a)
        self.assertEqual(load_csv_model_configuration(self.repo, self.scope, b.configuration_id), b)
        datasets = [r for r in self.repo.list_artifacts(self.scope) if r.kind == ArtifactKind.DATASET]
        self.assertEqual(len(datasets), 1)

    def test_roles_do_not_overlap(self):
        with self.assertRaises(ValueError):
            self.save(class_column='label', data_columns=('uid',))
        with self.assertRaises(ValueError):
            self.save(class_column='label', data_columns=(), text_columns=())

    def test_scope_isolation(self):
        saved = self.save(class_column='label', data_columns=('x',))
        other = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises(FileNotFoundError):
            load_csv_model_configuration(self.repo, other, saved.configuration_id)
        with self.assertRaises(FileNotFoundError):
            save_csv_model_configuration(self.repo, other, self.dataset.artifact_id,
                                         id_column='uid', class_column='label', data_columns=('x',))

    def test_private_columns_not_exposed_in_reference(self):
        saved = self.save(class_column='label', data_columns=('x',))
        self.assertNotIn('label', repr(saved.reference))
        self.assertNotIn('uid', repr(saved.reference))
        self.assertEqual(saved.reference.dataset_id, self.dataset.dataset_id)


if __name__ == '__main__':
    unittest.main()
