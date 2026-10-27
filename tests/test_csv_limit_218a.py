"""Regression: JBG's unlimited (zero) row limit is valid for CSV."""
import unittest
from unittest.mock import patch
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.ArtifactContract import ArtifactReference, ArtifactKind, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration
from uuid import uuid4
from tempfile import TemporaryDirectory
from pathlib import Path
from io import BytesIO


class CSVLimit218a(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = FileArtifactRepository(Path(tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, ref, BytesIO(b'value;uid;label\n0;001;\n1;002;\n2;003;\n'))
        roles = save_csv_model_configuration(self.repo, self.scope, ref.artifact_id,
            id_column='uid', class_column='label', data_columns=('value',), text_columns=(),
            csv_format={'delimiter': ';', 'quotechar': '"', 'encoding': 'utf-8'})
        self.config_id = roles.configuration_id

    def _read(self, limit):
        config = unittest.mock.Mock()
        config.get_column_names.return_value = ['value', 'uid', 'label']
        config.get_id_column_name.return_value = 'uid'
        config.get_class_column_name.return_value = 'label'
        config.data_source.kind = 'csv'
        # Leave source identity guard inactive for isolated limit validation.
        config.data_source = None
        config.should_train.return_value = False
        config.should_predict.return_value = True
        config.get_data_limit.return_value = limit
        return CSVTrainingAdapter(self.repo, self.scope, self.config_id, config).get_dataset()

    def test_zero_means_unlimited(self):
        self.assertEqual(len(self._read(0)), 3)

    def test_none_means_unlimited(self):
        self.assertEqual(len(self._read(None)), 3)

    def test_positive_limit_caps_rows(self):
        self.assertEqual(len(self._read(2)), 2)

    def test_negative_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unable to prepare CSV rows'):
            self._read(-1)

    def test_boolean_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unable to prepare CSV rows'):
            self._read(False)


if __name__ == '__main__':
    unittest.main()
