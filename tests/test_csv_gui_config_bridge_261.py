"""Revision 261: scoped CSV GUI-to-training configuration bridge."""
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVGUITrainingConfig import prepare_csv_gui_training_config
from Storage.CSVModelConfiguration import load_csv_model_configuration

try:
    from Config import Config
except ImportError:
    Config = None


@unittest.skipIf(Config is None, 'JBG runtime dependencies unavailable')
class CSVGUIConfigBridge261(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = FileArtifactRepository(self.root / 'objects')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.ref,
                                BytesIO(b'row_id,measurement,species\n1,2.2,a\n2,3.3,b\n'))
        self.base = Config()
        self.base.connection.host = 'sql-host-sentinel'
        self.base.connection.sql_username = 'sql-user-sentinel'
        self.base.connection.sql_password = 'sql-password-sentinel'
        self.base.mode.predict = True
        self.base.mail.notification_email = 'private@example.invalid'

    def prepare(self, **overrides):
        params = dict(id_column='row_id', class_column='species',
                      data_columns=('measurement',), text_columns=(),
                      csv_format=dict(delimiter=',', quotechar='"', encoding='utf-8-sig'))
        params.update(overrides)
        return prepare_csv_gui_training_config(self.base, self.repo, self.scope,
                                                self.ref.artifact_id, **params)

    def test_persisted_config_has_exact_csv_identity_and_no_sql_credentials(self):
        config, saved = self.prepare()
        self.assertEqual(config.data_source.kind, 'csv')
        self.assertEqual(config.data_source.artifact_id, self.ref.artifact_id)
        self.assertTrue(config.should_train())
        self.assertFalse(config.should_predict())
        self.assertEqual(config.get_column_names(), ['measurement', 'row_id', 'species'])
        self.assertEqual(config.connection.host, '')
        self.assertEqual(config.connection.sql_password, '')
        self.assertEqual(config.connection.sql_username, '')
        self.assertEqual(config.mail.notification_email, '')
        self.assertEqual(self.base.connection.host, 'sql-host-sentinel')
        self.assertEqual(self.base.connection.sql_password, 'sql-password-sentinel')
        loaded = load_csv_model_configuration(self.repo, self.scope, saved.configuration_id)
        self.assertEqual(loaded.shared_json['data_source']['kind'], 'csv')
        self.assertEqual(loaded.shared_json['connection']['sql_password'], '')
        self.assertEqual(loaded.shared_json['connection']['host'], '')

    def test_role_conflict_rejected_without_writing_configuration(self):
        with self.assertRaisesRegex(ValueError, 'column roles'):
            self.prepare(data_columns=('row_id',))
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), 1)

    def test_wrong_scope_rejected_without_configuration_publication(self):
        wrong = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises((ValueError, KeyError, OSError)):
            prepare_csv_gui_training_config(self.base, self.repo, wrong, self.ref.artifact_id,
                                             id_column='row_id', class_column='species',
                                             data_columns=('measurement',))
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), 1)

    def test_requires_validated_config_object(self):
        with self.assertRaisesRegex(ValueError, 'validated configuration'):
            prepare_csv_gui_training_config(object(), self.repo, self.scope, self.ref.artifact_id,
                                             id_column='row_id', class_column='species',
                                             data_columns=('measurement',))


if __name__ == '__main__':
    unittest.main()
