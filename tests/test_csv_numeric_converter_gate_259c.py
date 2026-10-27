"""259c: numeric-only CSV training still fits the model input-schema converter."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def production_get_tasks():
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_tasks')
    namespace = {"Config": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['get_tasks']


def config(kind, train, text=()):
    return SimpleNamespace(data_source=SimpleNamespace(kind=kind),
                           get_text_column_names=lambda: list(text),
                           should_train=lambda: train,
                           should_predict=lambda: not train,
                           should_calculate_feature_importance=lambda: False)


class CSVNumericConverterGate259c(unittest.TestCase):
    def test_numeric_only_csv_training_fits_converter_before_train(self):
        tasks = production_get_tasks()(config('csv', True))
        self.assertEqual(tasks.count('convert_to_numbers'), 1)
        self.assertLess(tasks.index('separate_dataset'), tasks.index('convert_to_numbers'))
        self.assertLess(tasks.index('convert_to_numbers'), tasks.index('train_model'))

    def test_sql_numeric_training_keeps_existing_sequence(self):
        self.assertNotIn('convert_to_numbers', production_get_tasks()(config('sql', True)))

    def test_text_training_keeps_single_converter(self):
        for kind in ('sql', 'csv'):
            with self.subTest(kind=kind):
                self.assertEqual(production_get_tasks()(config(kind, True, ('text',))).count('convert_to_numbers'), 1)

    def test_numeric_csv_prediction_sequence_not_changed_by_training_fix(self):
        self.assertNotIn('convert_to_numbers', production_get_tasks()(config('csv', False)))


if __name__ == '__main__':
    unittest.main()
