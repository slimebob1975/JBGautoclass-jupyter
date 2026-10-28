"""Revision 262: source-scoped CSV model location and GUI preparation boundary."""
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVGUITrainingConfig import prepare_csv_gui_training_config

try:
    from Config import Config
except ImportError:
    Config = None


@unittest.skipIf(Config is None, 'JBG application dependencies unavailable')
class CSVRuntimePreparation262(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = FileArtifactRepository(self.root / 'objects')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.ref,
                                BytesIO(b'row_id,feature,label\n1,2.5,a\n2,3.1,b\n'))

    def prepare(self, scope=None, model_root=None):
        return prepare_csv_gui_training_config(
            Config(), self.repo, scope or self.scope, self.ref.artifact_id,
            id_column='row_id', class_column='label', data_columns=('feature',),
            csv_format=dict(delimiter=',', quotechar='"', encoding='utf-8-sig'),
            model_root=model_root or (self.root / 'models'))

    def test_model_storage_uses_opaque_scope_and_dataset_identity(self):
        config, saved = self.prepare()
        self.assertIn(self.scope.tenant_id, config.io.model_path)
        self.assertIn(self.scope.project_id, config.io.model_path)
        self.assertIn(self.ref.dataset_id, config.io.model_path)
        self.assertEqual(config.io.model_name, 'csv_model_' + self.ref.artifact_id)
        self.assertNotEqual(config.io.model_path, Config().io.model_path)
        self.assertTrue(Path(config.io.model_path).is_dir())
        self.assertEqual(saved.shared_json['io']['model_path'], config.io.model_path)
        self.assertEqual(saved.shared_json['io']['model_name'], config.io.model_name)

    def test_invalid_storage_root_rejected_before_publication(self):
        before = len(self.repo.list_artifacts(self.scope))
        with self.assertRaisesRegex(ValueError, 'storage root'):
            self.prepare(model_root=Path('relative-models'))
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), before)

    def test_invalid_scope_cannot_publish(self):
        wrong = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaises((ValueError, KeyError, OSError)):
            self.prepare(scope=wrong)
        self.assertEqual(len(self.repo.list_artifacts(self.scope)), 1)

    def test_gui_start_handler_routes_csv_away_from_sql_start(self):
        from GUI.Widgets import EventHandler
        dummy = Mock()
        dummy._csv_view_active.return_value = True
        dummy.models_dropdown.value = Config.DEFAULT_TRAIN_OPTION
        EventHandler.start_button_was_clicked(EventHandler(dummy), Mock())
        dummy._csv_start_training.assert_called_once_with()
        dummy.start_button_actions.assert_not_called()
        dummy.deactivate_section.assert_not_called()


if __name__ == '__main__':
    unittest.main()
