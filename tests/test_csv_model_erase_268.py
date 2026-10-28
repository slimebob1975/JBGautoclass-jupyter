"""Synthetic regression coverage for scoped model deletion and GUI safety."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVModelConfiguration import save_csv_model_configuration, load_csv_model_configuration
from Storage.CSVSavedModelInspection import _model_filename
from Storage.CSVModelErasure import erase_saved_csv_model


class ModelErasureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.repo = FileArtifactRepository(root / 'objects')
        self.models = root / 'models'
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                         'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, self.dataset, BytesIO(b'id,class,x\n001,A,1\n'))
        self.first = self.create_model(self.scope, self.dataset)

    def create_model(self, scope, dataset):
        registration = save_csv_model_configuration(self.repo, scope, dataset.artifact_id,
            id_column='id', class_column='class', data_columns=('x',))
        filename = _model_filename(self.models.resolve(), scope,
                                   registration.dataset_id, registration.artifact_id)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(b'test-only checkpoint')
        return registration, filename

    def test_delete_only_selected_model_and_keep_dataset_and_other_checkpoint(self):
        second_dataset = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                           'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, second_dataset, BytesIO(b'id,class,x\n002,B,2\n'))
        other, other_path = self.create_model(self.scope, second_dataset)
        registration, filename = self.first
        erase_saved_csv_model(self.repo, self.scope, self.dataset.artifact_id,
                              registration.configuration_id, model_root=self.models.resolve())
        self.assertFalse(filename.exists())
        self.assertTrue(other_path.exists())
        self.assertEqual(self.repo.get_reference(self.scope, self.dataset.artifact_id), self.dataset)
        self.assertEqual(load_csv_model_configuration(self.repo, self.scope, other.configuration_id), other)
        with self.assertRaises(FileNotFoundError):
            self.repo.get_reference(self.scope, registration.configuration_id)

    def test_shared_checkpoint_is_preserved_for_other_registration(self):
        registered, filename = self.first
        second = save_csv_model_configuration(self.repo, self.scope, self.dataset.artifact_id,
            id_column='id', class_column='class', data_columns=('x',))
        erase_saved_csv_model(self.repo, self.scope, self.dataset.artifact_id,
            registered.configuration_id, model_root=self.models.resolve())
        self.assertTrue(filename.exists())
        self.assertEqual(load_csv_model_configuration(self.repo, self.scope, second.configuration_id), second)

    def test_wrong_dataset_does_not_remove_checkpoint(self):
        other = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.DATASET,
                                  'text/csv', dataset_id=str(uuid4()))
        self.repo.save_artifact(self.scope, other, BytesIO(b'id,class,x\n'))
        registered, filename = self.first
        with self.assertRaises(ValueError):
            erase_saved_csv_model(self.repo, self.scope, other.artifact_id,
                registered.configuration_id, model_root=self.models.resolve())
        self.assertTrue(filename.exists())

    def test_wrong_scope_does_not_remove_checkpoint(self):
        registered, filename = self.first
        with self.assertRaises(FileNotFoundError):
            erase_saved_csv_model(self.repo, StorageScope(str(uuid4()), str(uuid4())),
                self.dataset.artifact_id, registered.configuration_id,
                model_root=self.models.resolve())
        self.assertTrue(filename.exists())

    def test_missing_checkpoint_allows_stale_registration_cleanup(self):
        registered, filename = self.first
        filename.unlink()
        erase_saved_csv_model(self.repo, self.scope, self.dataset.artifact_id,
            registered.configuration_id, model_root=self.models.resolve())
        with self.assertRaises(FileNotFoundError):
            self.repo.get_reference(self.scope, registered.configuration_id)

    def test_symlink_checkpoint_rejected(self):
        registered, filename = self.first
        filename.unlink()
        destination = filename.parent / 'unrelated.sav'
        destination.write_bytes(b'other')
        try:
            filename.symlink_to(destination)
        except (OSError, NotImplementedError):
            self.skipTest('Symlinks not available')
        with self.assertRaises(ValueError):
            erase_saved_csv_model(self.repo, self.scope, self.dataset.artifact_id,
                registered.configuration_id, model_root=self.models.resolve())
        self.assertEqual(destination.read_bytes(), b'other')
        self.assertEqual(load_csv_model_configuration(self.repo, self.scope, registered.configuration_id), registered)


if __name__ == '__main__':
    unittest.main()

class ModelErasureGUIStructureTests(unittest.TestCase):
    def test_button_only_in_csv_handler_and_has_confirmation(self):
        import ast
        source = (Path(__file__).resolve().parents[1] / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        methods = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        body = ast.get_source_segment(source, methods['_csv_erase_model'])
        self.assertIn('self._csv_view_active()', body)
        self.assertIn('Config.DEFAULT_TRAIN_OPTION', body)
        self.assertIn('Confirm erase', body)
        self.assertIn('erase_saved_csv_model(', body)
        self.assertIn('self._csv_refresh_saved_models()', body)
        self.assertNotIn('run_sql', body.lower())
