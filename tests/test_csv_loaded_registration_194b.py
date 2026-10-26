"""Loaded CSV registrations must retain identity, filename and project scope."""
import ast
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from Storage.ArtifactContract import StorageScope
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.CSVDatasetCatalog import csv_dataset_display_name, restore_csv_dataset, list_csv_dataset_labels
from Storage.FileArtifactRepository import FileArtifactRepository


class SavedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.other = StorageScope(str(uuid4()), str(uuid4()))
        self.repo = FileArtifactRepository(self.base / 'objects')
        self.source = self.base / 'input.csv'
        self.source.write_text('uid,target,x\n001,A,yes\n002,B,no\n', encoding='utf-8')
        self.entry = register_csv_dataset(self.repo, self.scope, self.source,
                                          id_column='uid', class_column='target',
                                          feature_columns=('x',), original_name='input.csv')

    def test_saved_name_is_original_basename_and_scoped(self):
        self.assertEqual(csv_dataset_display_name(self.repo, self.scope, self.entry.dataset_id), 'input.csv')
        with self.assertRaises(FileNotFoundError):
            csv_dataset_display_name(self.repo, self.other, self.entry.dataset_id)

    def test_restoring_does_not_generate_new_registration(self):
        target = self.base / 'restored.csv'
        restore_csv_dataset(self.repo, self.scope, self.entry.reference.artifact_id, target)
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        self.assertEqual(len(list_csv_dataset_labels(self.repo, self.scope)), 1)

    def test_legacy_filename_fallback(self):
        legacy = register_csv_dataset(self.repo, self.scope, self.source,
                                      id_column='uid', class_column='target', feature_columns=('x',))
        self.assertEqual(csv_dataset_display_name(self.repo, self.scope, legacy.dataset_id), 'Registered CSV')

    def test_gui_reuses_existing_artifact_and_clears_on_new_upload(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        code = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertIn('loaded_saved_artifact_id', code['_csv_register_dataset'])
        self.assertIn('repository.get_reference(scope, loaded_id)', code['_csv_register_dataset'])
        self.assertIn('No duplicate created', code['_csv_register_dataset'])
        self.assertIn('csv_dataset_display_name(', code['_csv_load_saved'])
        self.assertIn('artifact_id=ref.artifact_id', code['_csv_load_saved'])
        panel = (Path(__file__).resolve().parents[1] / 'src/GUI/CSVSourcePanel.py').read_text(encoding='utf-8')
        self.assertIn('self.loaded_saved_artifact_id = None', panel)


if __name__ == '__main__':
    unittest.main()
