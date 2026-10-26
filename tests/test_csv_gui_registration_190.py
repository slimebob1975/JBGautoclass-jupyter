"""Local CSV registration and GUI boundaries; synthetic files only."""
import ast
from pathlib import Path
import tempfile
import unittest
from uuid import UUID
from Storage.LocalCSVScope import local_csv_scope
from Storage.FileArtifactRepository import FileArtifactRepository
from Storage.CSVDatasetRegistration import register_csv_dataset


class CSVRegistrationGUI190Tests(unittest.TestCase):
    def test_project_scope_is_stable_without_exposing_project_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = local_csv_scope(tmp, "synthetic-project")
            b = local_csv_scope(tmp, "synthetic-project")
            c = local_csv_scope(tmp, "another-project")
            self.assertEqual(a, b)
            self.assertNotEqual(a.project_id, c.project_id)
            self.assertEqual(UUID(a.project_id).version, 4)
            self.assertNotIn("synthetic-project", (Path(tmp) / ".local-tenant-id").read_text())

    def test_local_registration_survives_repository_reopening(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.csv"
            source.write_text("uid;target;feature\n001;A;one\n002;B;two\n", encoding="utf-8")
            scope = local_csv_scope(root / "store", "project")
            repo_path = root / "store" / "objects"
            registered = register_csv_dataset(FileArtifactRepository(repo_path), scope, source,
                                              id_column="uid", class_column="target",
                                              feature_columns=("feature",), delimiter=";")
            with FileArtifactRepository(repo_path).open_artifact(scope, registered.reference.artifact_id) as fh:
                self.assertEqual(fh.read(), source.read_bytes())

    def test_gui_registration_is_explicit_and_not_a_training_action(self):
        path = Path(__file__).resolve().parents[1] / "src/GUI/Widgets.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Widgets")
        methods = {x.name: ast.unparse(x) for x in cls.body if isinstance(x, ast.FunctionDef)}
        self.assertIn("register_button.on_click(self._csv_register_dataset)", methods["connection_form"])
        self.assertIn("on_file_changed = self._csv_file_changed", methods["connection_form"])
        self.assertIn("_csv_roles_loaded_188 = False", methods["_csv_file_changed"])
        registration = methods["_csv_register_dataset"]
        self.assertIn("register_csv_dataset(", registration)
        self.assertNotIn("run_classifier", registration)
        self.assertNotIn("start_button_actions", registration)


if __name__ == "__main__":
    unittest.main()
