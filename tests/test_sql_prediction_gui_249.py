"""Synthetic SQL GUI regression cases; no connection or private data."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    func = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name)
    ns = {}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), 'exec'), ns)
    return ns[method_name]


class SQLPredictionGUI249(unittest.TestCase):
    def test_target_choices_do_not_mix_raw_values_and_pairs(self):
        normalize = method('src/GUI/Widgets.py', 'Widgets', '_with_saved_target_options')
        for original in ((('All classes', ''), ('Positive', 'positive')), ('All classes', 'positive')):
            result = normalize(original, 'positive')
            self.assertTrue(all(isinstance(v, tuple) and len(v) == 2 for v in result))
            self.assertEqual([v for _, v in result].count('positive'), 1)
        self.assertEqual(normalize((('All classes', ''),), 'negative')[-1], ('negative', 'negative'))

    def test_saved_model_assisted_profile_must_not_disable_continue(self):
        # The mode restriction applies only to editable training datasets.
        text = (ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sync_feature_selection_controls')
        source = ast.get_source_segment(text, func)
        self.assertIn('if dataset_editable and (running or', source)
        self.assertIn('if dataset_editable and mode in ("assisted", "reuse")', source)

    def test_zero_row_sql_prediction_returns_without_connection(self):
        text = (ROOT / 'src/SQLDataLayer.py').read_text(encoding='utf-8')
        self.assertIn('if num_rows == 0:', text)
        self.assertLess(text.index('if num_rows == 0:'), text.index('sqlHelper = self.get_connection()', text.index('if num_rows == 0:')))

    def test_empty_result_does_not_switch_to_rerun(self):
        text = (ROOT / 'src/GUIHandler.py').read_text(encoding='utf-8')
        self.assertIn('if set_rerun and result is not False:', text)
        self.assertIn('Classification ended without eligible data', text)


if __name__ == '__main__':
    unittest.main()
