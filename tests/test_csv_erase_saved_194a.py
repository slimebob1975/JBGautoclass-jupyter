"""Scoped CSV deletion and GUI confirmation; synthetic-only fixtures."""
import ast
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVDatasetRegistration import register_csv_dataset
from Storage.CSVDatasetCatalog import erase_csv_dataset, list_csv_datasets, list_csv_dataset_labels


class EraseSavedCSVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = FileArtifactRepository(self.root / "store")
        self.scope = StorageScope(str(uuid4()), str(uuid4()))
        self.other = StorageScope(str(uuid4()), str(uuid4()))
        source = self.root / "sample.csv"
        source.write_text("uid,target,x\n001,A,one\n002,B,two\n", encoding="utf-8")
        self.one = register_csv_dataset(self.repo, self.scope, source, id_column="uid", class_column="target", feature_columns=("x",), original_name="sample.csv")
        self.two = register_csv_dataset(self.repo, self.scope, source, id_column="uid", class_column="target", feature_columns=("x",), original_name="sample.csv")

    def test_delete_removes_content_and_label_but_not_other_dataset(self):
        erase_csv_dataset(self.repo, self.scope, self.one.reference.artifact_id)
        self.assertEqual(list_csv_datasets(self.repo, self.scope), ((self.two.dataset_id, self.two.reference.artifact_id),))
        self.assertEqual(len(list_csv_dataset_labels(self.repo, self.scope)), 1)
        with self.assertRaises(FileNotFoundError):
            self.repo.open_artifact(self.scope, self.one.reference.artifact_id)

    def test_wrong_scope_and_non_dataset_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            erase_csv_dataset(self.repo, self.other, self.one.reference.artifact_id)
        self.assertEqual(len(list_csv_datasets(self.repo, self.scope)), 2)
        manifest = next(ref for ref in self.repo.list_artifacts(self.scope) if ref.kind == ArtifactKind.MANIFEST)
        with self.assertRaises(ValueError):
            erase_csv_dataset(self.repo, self.scope, manifest.artifact_id)

    def test_related_results_block_erase(self):
        ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.METRICS, "application/json", run_id=str(uuid4()), dataset_id=self.one.dataset_id)
        self.repo.save_artifact(self.scope, ref, BytesIO(b"{}"))
        with self.assertRaises(ValueError):
            erase_csv_dataset(self.repo, self.scope, self.one.reference.artifact_id)
        self.assertEqual(len(list_csv_datasets(self.repo, self.scope)), 2)

    def test_gui_requires_second_click(self):
        src = (Path(__file__).resolve().parents[1] / "src/GUI/Widgets.py").read_text()
        cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Widgets")
        methods = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("erase_saved_button.on_click(self._csv_erase_saved)", methods["connection_form"])
        self.assertIn("panel._erase_confirmation != artifact_id", methods["_csv_erase_saved"])
        self.assertNotIn("run_classifier", methods["_csv_erase_saved"])


if __name__ == "__main__":
    unittest.main()
