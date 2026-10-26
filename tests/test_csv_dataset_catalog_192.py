"""Scoped durable CSV retrieval tests; synthetic data only."""
import tempfile
import unittest
from pathlib import Path
from Storage.ArtifactContract import StorageScope
from Storage.CSVDatasetCatalog import list_csv_datasets, restore_csv_dataset
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.FileArtifactRepository import FileArtifactRepository
from uuid import uuid4


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = FileArtifactRepository(self.root / "archive")
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.other = StorageScope(str(uuid4()), str(uuid4()))
        self.source = self.root / "input.csv"
        self.source.write_text('uid;label;feature\n001;A;"one;two"\n002;B;three\n', encoding="utf-8")
        self.registered = register_csv_dataset(self.repo, self.scope, self.source, id_column="uid", class_column="label", feature_columns=("feature",), delimiter=";")

    def test_catalog_and_restore_survive_reopen(self):
        reopened = FileArtifactRepository(self.root / "archive")
        entries = list_csv_datasets(reopened, self.scope)
        self.assertEqual(entries, ((self.registered.dataset_id, self.registered.reference.artifact_id),))
        target = self.root / "restored.csv"
        ref = restore_csv_dataset(reopened, self.scope, entries[0][1], target)
        self.assertEqual(ref.dataset_id, self.registered.dataset_id)
        self.assertEqual(target.read_bytes(), self.source.read_bytes())

    def test_scope_isolation_and_no_overwrite(self):
        self.assertEqual(list_csv_datasets(self.repo, self.other), ())
        with self.assertRaises(FileNotFoundError):
            restore_csv_dataset(self.repo, self.other, self.registered.reference.artifact_id, self.root / "forbidden.csv")
        self.assertFalse((self.root / "forbidden.csv").exists())
        target = self.root / "existing.csv"
        target.write_text("do not overwrite")
        with self.assertRaises(ValueError):
            restore_csv_dataset(self.repo, self.scope, self.registered.reference.artifact_id, target)
        self.assertEqual(target.read_text(), "do not overwrite")

    def test_no_server_paths_in_public_catalog(self):
        self.assertNotIn(str(self.root), str(list_csv_datasets(self.repo, self.scope)))


if __name__ == "__main__":
    unittest.main()

class GUIWiring192Tests(unittest.TestCase):
    def test_saved_catalog_actions_are_separate_from_execution(self):
        import ast
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Widgets')
        methods = {node.name: ast.unparse(node) for node in cls.body if isinstance(node, ast.FunctionDef)}
        self.assertIn('refresh_saved_button.on_click(self._csv_refresh_saved)', methods['connection_form'])
        self.assertIn('load_saved_button.on_click(self._csv_load_saved)', methods['connection_form'])
        self.assertIn('restore_csv_dataset(', methods['_csv_load_saved'])
        self.assertIn('stage_saved_csv(', methods['_csv_load_saved'])
        self.assertNotIn('run_classifier', methods['_csv_load_saved'])
        self.assertNotIn('start_button_actions', methods['_csv_load_saved'])
