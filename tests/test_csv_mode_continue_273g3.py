"""Mode-panel choices and explicit fail-closed CSV limitations."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "GUI" / "Widgets.py"


def methods(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(found) != len(names):
        raise AssertionError("Missing CSV method")
    cls = ast.ClassDef(name="Subject", bases=[], keywords=[], body=found, decorator_list=[])
    ns = {"Config": SimpleNamespace(DEFAULT_TRAIN_OPTION="New model")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), ns)
    return ns["Subject"]


class ContinueModes273g3(unittest.TestCase):
    def test_mispredicted_training_is_ready_without_predict(self):
        w = methods("_csv_training_readiness")()
        w._csv_view_active = lambda: True
        w._csv_roles_loaded_188 = True
        w.models_dropdown = SimpleNamespace(value="New model")
        w._csv_current_dataset_artifact = lambda: "opaque-id"
        w.id_column = SimpleNamespace(value="id")
        w.class_column = SimpleNamespace(value="label")
        w.data_columns = SimpleNamespace(value=("feature",))
        w.algorithm_dropdown = SimpleNamespace(value=("model",))
        w.preprocess_dropdown = SimpleNamespace(value=("none",))
        w.reduction_dropdown = SimpleNamespace(value=("none",))
        w.predict_checkbox = SimpleNamespace(value=False)
        w.mispredicted_checkbox = SimpleNamespace(value=True)
        self.assertEqual(w._csv_training_readiness()[0], "ready")
        w.mispredicted_checkbox.value = False
        self.assertEqual(w._csv_training_readiness()[0], "ready")

    def test_saved_metadata_fails_closed(self):
        w = methods("_csv_saved_prediction_readiness")()
        w._csv_view_active = lambda: True
        w._csv_roles_loaded_188 = True
        w.models_dropdown = SimpleNamespace(value="saved-id")
        w._csv_current_dataset_artifact = lambda: "opaque-id"
        w.metas_checkbox = SimpleNamespace(value=True)
        self.assertEqual(w._csv_saved_prediction_readiness()[0], "waiting")
        w.metas_checkbox.value = False
        self.assertEqual(w._csv_saved_prediction_readiness()[0], "ready")


if __name__ == "__main__":
    unittest.main()
