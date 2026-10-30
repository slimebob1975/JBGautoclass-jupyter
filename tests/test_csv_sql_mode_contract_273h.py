"""273h: source-parity contract and explicit remaining integration gaps.

These checks intentionally do not claim that CSV misprediction or metadata
publication is implemented. They protect the common ML TaskRunner contract
and identify the adapter boundary to finish before unlocking GUI modes.
"""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'


def function_in(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f'Expected one {name} in {path.name}; found {len(matches)}')
    return matches[0]


class CSVSQLModeContract273h(unittest.TestCase):
    def test_shared_task_sequence_includes_misprediction_for_training(self):
        node = function_in(SRC / 'JBGTaskRunner.py', 'get_tasks')
        source = ast.unparse(node)
        self.assertIn('display_mispredicted', source)
        self.assertIn('config.should_train()', source)
        self.assertIn('config.should_predict()', source)

    def test_misprediction_task_is_shared_not_a_csv_specific_duplicate(self):
        node = function_in(SRC / 'JBGTaskRunner.py', 'display_mispredicted__task')
        source = ast.unparse(node)
        self.assertIn('self.config.should_display_mispredicted()', source)
        self.assertIn('self.ph.most_mispredicted(', source)
        self.assertIn('self.ph.evaluate_mispredictions(', source)

    def test_csv_mode_diagnostic_is_explicit_not_silently_forced_off(self):
        """The CSV configuration must preserve validated diagnostic choice."""
        node = function_in(SRC / 'Storage' / 'CSVGUITrainingConfig.py',
                           'prepare_csv_gui_training_config')
        source = ast.unparse(node)
        self.assertIn('mispredicted', source)
        self.assertIn('predict', source)
        self.assertNotIn('mode.mispredicted = False', source)

    def test_csv_mode_requires_explicit_validation_before_start(self):
        widget = SRC / 'GUI' / 'Widgets.py'
        training = ast.unparse(function_in(widget, '_csv_training_readiness'))
        saved = ast.unparse(function_in(widget, '_csv_saved_prediction_readiness'))
        self.assertIn('self.predict_checkbox.value', training)
        self.assertIn('self.algorithm_dropdown.value', training)
        self.assertIn('self.metas_checkbox.value', saved)


if __name__ == '__main__':
    unittest.main()
