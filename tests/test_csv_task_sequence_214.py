"""CSV task dispatch must not allow training by passing manual task names."""
from pathlib import Path
import ast
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer


def _method():
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'TaskRunner')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'run')
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), ns)
    return ns['run']


BASE = ['load_model', 'fetch_data', 'load_dataset', 'separate_dataset', 'make_predictions']


def _runner(kind='csv', text=False):
    config = SimpleNamespace(data_source=SimpleNamespace(kind=kind), should_train=lambda: False,
                             get_text_column_names=lambda: ['feature'] if text else [])
    logger = Mock()
    handler = Mock()
    handler.get_handler.return_value = Mock()
    runner = SimpleNamespace(config=config, datalayer=CSVPredictionDataLayer(Mock(), Mock()),
                             logger=logger, handler=handler, regression_suite=False)
    return runner


class CSVTaskSequenceTests(TestCase):
    def test_valid_sequence_enters_runner(self):
        runner = _runner()
        runner.load_model__task = Mock(return_value={})
        runner.fetch_data__task = Mock(return_value={'early_exit': True})
        self.assertTrue(_method()(runner, BASE))

    def test_optional_conversion_and_email_are_valid(self):
        runner = _runner(text=True)
        runner.load_model__task = Mock(return_value={'early_exit': True})
        ordered = BASE[:4] + ['convert_to_numbers'] + BASE[4:] + ['send_completetion_email']
        self.assertTrue(_method()(runner, ordered))

    def test_training_task_rejected_before_progress(self):
        runner = _runner()
        with self.assertRaisesRegex(ValueError, 'task sequence'):
            _method()(runner, BASE[:4] + ['train_model'] + BASE[4:])
        runner.logger.initiate_progress.assert_not_called()

    def test_missing_reordered_and_duplicate_tasks_rejected(self):
        for tasks in (BASE[:-1], BASE[::-1], BASE + ['make_predictions'], [],
                      BASE[:4] + ['display_mispredicted'] + BASE[4:]):
            runner = _runner()
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                _method()(runner, tasks)
            runner.logger.initiate_progress.assert_not_called()

    def test_conversion_must_match_text_mode(self):
        for text, tasks in ((True, BASE), (False, BASE[:4] + ['convert_to_numbers'] + BASE[4:])):
            runner = _runner(text=text)
            with self.assertRaisesRegex(ValueError, 'conversion'):
                _method()(runner, tasks)

    def test_sql_dispatch_retains_legacy_empty_tasks(self):
        runner = _runner(kind='sql')
        self.assertFalse(_method()(runner, []))
        runner.logger.initiate_progress.assert_called_once()
