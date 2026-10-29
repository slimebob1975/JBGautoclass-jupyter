"""Revision 273m: real CSV GUI readiness and mode propagation checks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(name):
    path = ROOT / "src" / "GUI" / "Widgets.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(methods) == 1
    cls = ast.ClassDef(name="Subject", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"Config": SimpleNamespace(DEFAULT_TRAIN_OPTION="New model")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Subject"]


class CSVMispredictedGUIGate273m(unittest.TestCase):
    def setUp(self):
        self.w = extract("_csv_training_readiness")()
        self.w._csv_view_active = lambda: True
        self.w._csv_roles_loaded_188 = True
        self.w._csv_current_dataset_artifact = lambda: "synthetic-artifact"
        self.w.models_dropdown = SimpleNamespace(value="New model")
        self.w.id_column = SimpleNamespace(value="row_id")
        self.w.class_column = SimpleNamespace(value="target")
        self.w.data_columns = SimpleNamespace(value=("feature",))
        for field in ("algorithm_dropdown", "preprocess_dropdown", "reduction_dropdown"):
            setattr(self.w, field, SimpleNamespace(value=("test",)))
        self.w.predict_checkbox = SimpleNamespace(value=False)
        self.w.mispredicted_checkbox = SimpleNamespace(value=False)

    def test_train_only_still_ready_without_diagnostics(self):
        self.assertEqual(self.w._csv_training_readiness()[0], "ready")

    def test_diagnostics_require_combined_mode(self):
        self.w.mispredicted_checkbox.value = True
        state, reason = self.w._csv_training_readiness()
        self.assertEqual(state, "waiting")
        self.assertIn("Predict", reason)

    def test_combined_diagnostics_are_ready(self):
        self.w.predict_checkbox.value = True
        self.w.mispredicted_checkbox.value = True
        self.assertEqual(self.w._csv_training_readiness()[0], "ready")

    def test_preparation_passes_actual_checkbox_choice(self):
        text = (ROOT / "src" / "GUI" / "Widgets.py").read_text(encoding="utf-8")
        self.assertIn("mispredicted=bool(self.mispredicted_checkbox.value)", text)
        config = (ROOT / "src" / "Storage" / "CSVGUITrainingConfig.py").read_text(encoding="utf-8")
        self.assertIn("config.mode.mispredicted = bool(mispredicted)", config)
        self.assertIn("if mispredicted and not predict:", config)


if __name__ == "__main__":
    unittest.main()
