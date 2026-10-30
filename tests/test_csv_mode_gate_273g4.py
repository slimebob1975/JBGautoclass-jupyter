"""Regression checks for CSV Mode gating after Continue (revision 273g4)."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "GUI" / "Widgets.py"

def subject(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names):
        raise AssertionError("Missing Mode method")
    cls = ast.ClassDef(name="Subject", bases=[], keywords=[], body=functions, decorator_list=[])
    ns = {"Config": SimpleNamespace(DEFAULT_TRAIN_OPTION="New model")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), ns)
    return ns["Subject"]

class CSVModeGate273g4(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.w = subject("_csv_continue_actions", "_csv_training_readiness")()
        self.w._csv_view_active = lambda: True
        self.w._csv_continue_ready = lambda: True
        self.w._CSV_TRAINING_EDITABLE_FIELDS = ("algorithm_dropdown",)
        self.w._CSV_TRAINING_LOCKED_FIELDS = ("train_checkbox", "mispredicted_checkbox", "metas_checkbox")
        self.w._CSV_DARK_NUMBER_FIELDS = ("dark_number_method", "dark_number_alpha")
        self.w.eventhandler = SimpleNamespace(lock_observe_1=False)
        for name in ("train", "predict", "mispredicted", "metas"):
            setattr(self.w, name + "_checkbox", SimpleNamespace(value=False, disabled=True))
        self.w.models_dropdown = SimpleNamespace(value="New model")
        self.w.dark_numbers_checkbox = SimpleNamespace(value=True, disabled=True)
        self.w.sync_dark_number_control_state = lambda: None
        self.w.sync_feature_selection_controls = lambda: None
        self.w._csv_source_panel = SimpleNamespace(status=SimpleNamespace(value=""))
        self.w.enable_items = lambda names: [setattr(getattr(self.w,n),"disabled",False) for n in names if hasattr(self.w,n)]
        self.w.disable_items = lambda names: [setattr(getattr(self.w,n),"disabled",True) for n in names if hasattr(self.w,n)]
        self.w.update_values = lambda values: [setattr(getattr(self.w,n),"value",v) for n,v in values.items()]
        self.w._csv_refresh_training_action = lambda: None
        self.w.update_ready = lambda: None

    def test_new_model_after_continue_is_runnable_without_unimplemented_diagnostics(self):
        self.w._csv_continue_actions()
        self.assertTrue(self.w.train_checkbox.value)
        self.assertFalse(self.w.mispredicted_checkbox.value)
        self.assertFalse(self.w.predict_checkbox.disabled)
        self.assertTrue(self.w.metas_checkbox.disabled)
        self.w._csv_roles_loaded_188 = True
        self.w._csv_current_dataset_artifact = lambda: "test-dataset"
        self.w.id_column = SimpleNamespace(value="row_id")
        self.w.class_column = SimpleNamespace(value="class")
        self.w.data_columns = SimpleNamespace(value=("feature",))
        for name in ("algorithm_dropdown", "preprocess_dropdown", "reduction_dropdown"):
            setattr(self.w, name, SimpleNamespace(value=("one",)))
        self.assertEqual(self.w._csv_training_readiness()[0], "ready")

    def test_saved_model_can_choose_metadata_after_continue(self):
        self.w.models_dropdown.value = "saved-id"
        self.w._csv_continue_actions()
        self.assertFalse(self.w.metas_checkbox.disabled)
        self.assertFalse(self.w.predict_checkbox.disabled)
        self.assertTrue(self.w.train_checkbox.disabled)

    def test_neutral_model_label_is_present(self):
        self.assertIn("Train new or pick ready model", SOURCE.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
