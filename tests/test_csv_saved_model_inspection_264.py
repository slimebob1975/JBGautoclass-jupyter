"""Revision 264: scoped read-only CSV checkpoint discovery."""
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVSavedModelInspection import (
    _model_filename, inspect_saved_csv_model, list_saved_csv_models)


class CSVCheckpointPath264(unittest.TestCase):
    def test_rejects_relative_root(self):
        scope = StorageScope(str(uuid4()), str(uuid4()))
        with self.assertRaisesRegex(ValueError, "root"):
            _model_filename(Path("relative"), scope, str(uuid4()), str(uuid4()))

    def test_rejects_traversal(self):
        scope = StorageScope(str(uuid4()), str(uuid4()))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _model_filename(Path(directory), scope, "../escape", str(uuid4()))

    def test_uses_scoped_opaque_filename(self):
        scope = StorageScope(str(uuid4()), str(uuid4()))
        dataset, artifact = str(uuid4()), str(uuid4())
        with tempfile.TemporaryDirectory() as directory:
            candidate = _model_filename(Path(directory), scope, dataset, artifact)
            self.assertEqual(candidate.name, "csv_model_" + artifact + ".sav")
            self.assertEqual(candidate.parent.parts[-3:], (scope.tenant_id, scope.project_id, dataset))

try:
    from Config import Config
except ImportError:
    Config = None


@unittest.skipIf(Config is None, 'Full JBG dependencies unavailable')
class CSVCheckpointInspection264(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = FileArtifactRepository(self.root / 'objects')
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                     'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.ref,
                                BytesIO(b'id,feature,class\n1,1.2,a\n2,2.1,b\n'))
        from Storage.CSVGUITrainingConfig import prepare_csv_gui_training_config
        self.runtime, self.saved = prepare_csv_gui_training_config(
            Config(), self.repo, self.scope, self.ref.artifact_id,
            id_column='id', class_column='class', data_columns=('feature',),
            csv_format={'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8'},
            model_root=self.root / 'models')
        self.filename = Path(self.runtime.get_model_filename())

    def inspect(self, **kwargs):
        return inspect_saved_csv_model(self.repo, self.scope, self.ref.artifact_id,
                                       self.saved.configuration_id,
                                       model_root=self.root / 'models', **kwargs)

    def test_missing_checkpoint_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'No saved CSV model'):
            self.inspect()
        self.assertEqual(list_saved_csv_models(self.repo, self.scope, self.ref.artifact_id,
                                              model_root=self.root / 'models'), ())

    def test_other_dataset_cannot_reuse_checkpoint(self):
        another = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                    'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, another, BytesIO(b'id,feature,class\n3,1.1,a\n'))
        with self.assertRaisesRegex(ValueError, 'another dataset'):
            inspect_saved_csv_model(self.repo, self.scope, another.artifact_id,
                                    self.saved.configuration_id, model_root=self.root / 'models')

    def test_untrusted_path_is_rejected(self):
        with self.assertRaises(ValueError):
            _model_filename(self.root / 'models', self.scope, '../../escape', self.ref.artifact_id)
        with self.assertRaises(ValueError):
            _model_filename(Path('relative'), self.scope, self.ref.dataset_id, self.ref.artifact_id)

    def test_sql_model_metadata_rejected(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.filename.write_bytes(b'synthetic-checkpoint')
        with patch('JBGModelPersistence.load_model_artifact_with_compatibility', return_value=(
            (self.runtime, None), {'source_kind': 'sql'})):
            with self.assertRaises(ValueError):
                self.inspect()

    def test_valid_metadata_requires_semantic_validation(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.filename.write_bytes(b'synthetic-checkpoint')
        with patch('JBGModelPersistence.load_model_artifact_with_compatibility', return_value=(
            (self.runtime, None), {'source_kind': 'csv'})), \
             patch('JBGModelCompatibility.review_model_source_boundary', return_value=SimpleNamespace(ok=True)), \
             patch('JBGModelCompatibility.review_artifact_semantics', return_value=SimpleNamespace(ok=False)):
            with self.assertRaisesRegex(ValueError, 'compatibility'):
                self.inspect()


if __name__ == '__main__':
    unittest.main()

class CSVCheckpointGUI264(unittest.TestCase):
    def test_gui_selection_inspects_without_execution(self):
        import ast
        from types import SimpleNamespace
        from unittest.mock import Mock
        source = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_csv_inspect_selected_model')
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
            name='Target', bases=[], keywords=[], body=[fn], decorator_list=[])], type_ignores=[]))
        namespace = {'Config': SimpleNamespace(DEFAULT_TRAIN_OPTION='New model')}
        exec(compile(module, str(source), 'exec'), namespace)
        target = SimpleNamespace(
            models_dropdown=SimpleNamespace(value='synthetic-config'),
            _csv_catalog_storage=Mock(return_value=(Mock(), Mock())),
            _csv_current_dataset_artifact=Mock(return_value='synthetic-dataset'),
            _csv_source_panel=SimpleNamespace(status=SimpleNamespace(value='')),
            _csv_show_saved_pipeline=Mock(), _csv_set_model_status=Mock())
        with patch('Storage.CSVSavedModelInspection.inspect_saved_csv_model') as inspection:
            result = namespace['Target']._csv_inspect_selected_model(target)
        self.assertTrue(result)
        inspection.assert_called_once()
        self.assertIn('verified', target._csv_source_panel.status.value)
        target._csv_show_saved_pipeline.assert_called_once()

    def test_gui_rejects_unverified_checkpoint(self):
        import ast
        from types import SimpleNamespace
        from unittest.mock import Mock
        source = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_csv_inspect_selected_model')
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
            name='Target', bases=[], keywords=[], body=[fn], decorator_list=[])], type_ignores=[]))
        namespace = {'Config': SimpleNamespace(DEFAULT_TRAIN_OPTION='New model')}
        exec(compile(module, str(source), 'exec'), namespace)
        target = SimpleNamespace(models_dropdown=SimpleNamespace(value='synthetic-config'),
            _csv_catalog_storage=Mock(return_value=(Mock(), Mock())),
            _csv_current_dataset_artifact=Mock(return_value='synthetic-dataset'),
            _csv_source_panel=SimpleNamespace(status=SimpleNamespace(value='')),
            _csv_set_model_status=Mock())
        with patch('Storage.CSVSavedModelInspection.inspect_saved_csv_model',
                   side_effect=ValueError('synthetic mismatch')):
            self.assertFalse(namespace['Target']._csv_inspect_selected_model(target))
        self.assertIn('cannot be reused', target._csv_source_panel.status.value)
        target._csv_set_model_status.assert_called_once()
