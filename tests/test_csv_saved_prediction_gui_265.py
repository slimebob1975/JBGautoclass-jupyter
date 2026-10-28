"""Regression for CSV GUI prediction dispatch; never invoke SQL classification."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


class SavedPredictionDispatchTests(unittest.TestCase):
    def test_csv_saved_model_calls_saved_prediction_not_training(self):
        try:
            from GUI.Widgets import EventHandler
        except ImportError as exc:
            self.skipTest(str(exc))
        selected = SimpleNamespace(value='a-valid-config-id')
        gui = SimpleNamespace(
            _csv_view_active=lambda: True,
            models_dropdown=selected,
            _csv_start_saved_prediction=Mock(),
            _csv_start_training=Mock(),
        )
        EventHandler(gui).start_button_was_clicked(None)
        gui._csv_start_saved_prediction.assert_called_once_with()
        gui._csv_start_training.assert_not_called()

    def test_csv_new_model_still_trains(self):
        try:
            from GUI.Widgets import EventHandler
            from Config import Config
        except ImportError as exc:
            self.skipTest(str(exc))
        gui = SimpleNamespace(
            _csv_view_active=lambda: True,
            models_dropdown=SimpleNamespace(value=Config.DEFAULT_TRAIN_OPTION),
            _csv_start_saved_prediction=Mock(),
            _csv_start_training=Mock(),
        )
        EventHandler(gui).start_button_was_clicked(None)
        gui._csv_start_training.assert_called_once_with()
        gui._csv_start_saved_prediction.assert_not_called()

    def test_prediction_is_rejected_without_selected_registered_dataset(self):
        try:
            from GUI.Widgets import Widgets
        except ImportError as exc:
            self.skipTest(str(exc))
        fake = SimpleNamespace(
            _csv_view_active=lambda: True, _csv_roles_loaded_188=True,
            models_dropdown=SimpleNamespace(value='valid-id'),
            _csv_current_dataset_artifact=lambda: None,
        )
        self.assertEqual(Widgets._csv_saved_prediction_readiness(fake)[0], 'waiting')


if __name__ == '__main__':
    unittest.main()

class SavedPredictionStructureTests(unittest.TestCase):
    def test_dispatch_never_calls_legacy_sql_runner(self):
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/GUI/Widgets.py').read_text())
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'start_button_was_clicked']
        self.assertEqual(len(handlers), 1)
        source = ast.unparse(handlers[0])
        self.assertIn('_csv_start_saved_prediction()', source)
        self.assertIn('_csv_start_training()', source)
        self.assertIn('if self.widgets._csv_view_active()', source)

    def test_prediction_uses_scoped_application_boundary(self):
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/GUIHandler.py').read_text())
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run_csv_saved_prediction']
        self.assertEqual(len(methods), 1)
        source = ast.unparse(methods[0])
        self.assertIn('prepare_registered_csv_prediction_classifier', source)
        self.assertNotIn('run_classifier(', source)
