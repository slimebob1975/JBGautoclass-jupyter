"""Regression: shared JSON-v2 payloads and legacy CSV configurations."""
import json
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVModelConfiguration import save_csv_model_configuration, load_csv_model_configuration


class CSVSharedConfig200Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = FileArtifactRepository(Path(self.tmp.name))
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET, 'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.dataset, BytesIO(b'id,target,feature\n1,A,2\n'))
        self.dialect = {'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8-sig'}
        self.payload = {
            'data_source': {'kind': 'csv', 'dataset_id': self.dataset.dataset_id,
                            'artifact_id': self.dataset.artifact_id, **self.dialect},
            'connection': {'id_column': 'id', 'class_column': 'target',
                           'data_numerical_columns': ['feature'], 'data_text_columns': [],
                           'sql_password':'', 'sql_username':'', 'host':'',
                           'data_catalog':'', 'data_table':'', 'class_catalog':'', 'class_table':''},
            'mail': {'notification_email': ''},
            'mode': {'test_size': 0.25}
        }
        fake = types.ModuleType('Config')
        class StubConfig:
            @staticmethod
            def _validate_json_payload(payload):
                if 'mode' not in payload:
                    raise ValueError('missing mode')
        fake.Config = StubConfig
        self.patch = patch.dict(sys.modules, {'Config': fake})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def save(self, shared):
        return save_csv_model_configuration(self.repo, self.scope, self.dataset.artifact_id,
                        id_column='id', class_column='target', data_columns=('feature',),
                        csv_format=self.dialect, shared_json=shared)

    def test_new_config_keeps_complete_settings(self):
        config = self.save(self.payload)
        loaded = load_csv_model_configuration(self.repo, self.scope, config.configuration_id)
        self.assertEqual(loaded.shared_json['mode']['test_size'], 0.25)
        self.assertEqual(loaded.artifact_id, self.dataset.artifact_id)

    def test_legacy_configuration_still_loads(self):
        config = self.save(None)
        self.assertIsNone(load_csv_model_configuration(self.repo, self.scope, config.configuration_id).shared_json)

    def test_wrong_dataset_identity_rejected(self):
        self.payload['data_source']['dataset_id'] = str(uuid4())
        with self.assertRaises(ValueError):
            self.save(self.payload)

    def test_sql_identity_rejected(self):
        self.payload['connection']['sql_username'] = 'private'
        with self.assertRaises(ValueError):
            self.save(self.payload)

    def test_role_mismatch_rejected(self):
        self.payload['connection']['class_column'] = 'other'
        with self.assertRaises(ValueError):
            self.save(self.payload)

if __name__ == '__main__':
    unittest.main()
