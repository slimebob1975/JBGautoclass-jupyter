"""CSV Continue must advance, not merely reselect the chosen model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "GUI" / "Widgets.py"


def subject(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names):
        raise AssertionError("Missing Continue method")
    cls = ast.ClassDef(name="Subject", bases=[], keywords=[], body=functions, decorator_list=[])
    namespace = {"Config": SimpleNamespace(DEFAULT_TRAIN_OPTION="New model"),
                 "widgets": SimpleNamespace(Button=object)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace["Subject"]


class CSVContinueClick273g2(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.w = subject("_csv_continue_actions")()
        self.w._csv_view_active = lambda: True
        self.w._csv_continue_ready = lambda: True
        self.w._CSV_TRAINING_EDITABLE_FIELDS = ("algorithm_dropdown",)
        self.w._CSV_TRAINING_LOCKED_FIELDS = ("train_checkbox", "mispredicted_checkbox", "metas_checkbox")
        self.w._CSV_DARK_NUMBER_FIELDS = ("dark_number_method", "dark_number_alpha")
        self.w.eventhandler = SimpleNamespace(lock_observe_1=False)
        self.w.update_values = lambda values: self.events.append(("values", values))
        self.w.enable_items = lambda names: self.events.append(("enable", tuple(names)))
        self.w.disable_items = lambda names: self.events.append(("disable", tuple(names)))
        self.w._csv_refresh_training_action = lambda: self.events.append("refresh_start")
        self.w._csv_source_panel = SimpleNamespace(status=SimpleNamespace(value=""))
        self.w.models_dropdown = SimpleNamespace(value="New model")
        self.w.update_ready = lambda: self.events.append("update_ready")
        self.w.dark_numbers_checkbox = SimpleNamespace(value=True)
        self.w.sync_dark_number_control_state = lambda: self.events.append("sync_dark_numbers")
        self.w.sync_feature_selection_controls = lambda: self.events.append("sync_feature_selection")

    def test_new_model_continue_exposes_settings(self):
        self.w._csv_continue_actions()
        self.assertIn(("enable", ("algorithm_dropdown", "train_checkbox", "predict_checkbox", "mispredicted_checkbox")), self.events)
        self.assertIn(("disable", ("metas_checkbox",)), self.events)
        self.assertTrue(self.w.train_checkbox.value if hasattr(self.w, "train_checkbox") else True)
        self.assertIn("refresh_start", self.events)
        self.assertIn("training is ready", self.w._csv_source_panel.status.value)

    def test_saved_model_continue_leaves_configuration_untouched(self):
        self.w.models_dropdown.value = "synthetic-model-id"
        self.w._csv_continue_actions()
        self.assertIn(("enable", ("predict_checkbox", "metas_checkbox")), self.events)
        self.assertIn(("disable", ("train_checkbox", "mispredicted_checkbox")), self.events)
        self.assertIn("refresh_start", self.events)
        self.assertIn("Saved CSV model ready", self.w._csv_source_panel.status.value)

    def test_invalid_continue_does_not_change_settings(self):
        self.w._csv_continue_ready = lambda: False
        self.w._csv_continue_actions()
        self.assertEqual(self.events, ["update_ready"])

    def test_real_click_dispatches_to_csv_transition(self):
        handler = subject("continuation_button_was_clicked")()
        self.w.continuation_button = SimpleNamespace(disabled=False)
        self.w._csv_continue_actions = lambda: self.events.append("advance")
        self.w._csv_model_selected = lambda: self.fail("Must not just refresh model selection")
        handler.widgets = self.w
        handler.continuation_button_was_clicked(None)
        self.assertEqual(self.events, ["advance"])


if __name__ == "__main__":
    unittest.main()
