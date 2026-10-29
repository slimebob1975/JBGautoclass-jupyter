"""273d: combined CSV dispatch is guarded; partitions are loaded once."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Storage.CSVCombinedDataLayer import CSVCombinedDataLayer
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVPredictionResults import CSVPredictionResultSink


def task_method(name):
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskRunner')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    context = {'get_tasks': lambda *_args, **_kwargs: []}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), context)
    return context[name]


class CombinedDispatch273d(unittest.TestCase):
    def setUp(self):
        repo, scope = Mock(), Mock()
        reader = object.__new__(CSVTrainingAdapter)
        reader.repository, reader.scope = repo, scope
        reader.configuration = Mock(dataset_id='dataset', artifact_id='artifact')
        reader.correction_revision = None
        reader.get_dataset = Mock()
        sink = object.__new__(CSVPredictionResultSink)
        sink.repository, sink.scope = repo, scope
        sink.context = Mock(dataset_id='dataset', artifact_id='artifact')
        self.layer = CSVCombinedDataLayer(reader, sink)
        self.config = Mock()
        self.config.data_source.kind = 'csv'
        self.config.should_train.return_value = True
        self.config.should_predict.return_value = True
        self.config.should_calculate_feature_importance.return_value = False
        self.config.should_display_mispredicted.return_value = False
        self.logger = Mock()
        self.handler = Mock(datalayer=self.layer)
        self.runner = SimpleNamespace(config=self.config, datalayer=self.layer,
            handler=self.handler, logger=self.logger, regression_suite=False)
        self.runner.run = task_method('run').__get__(self.runner)
        self.runner.fetch_data__task = task_method('fetch_data__task').__get__(self.runner)
        self.tasks = ['load_model', 'fetch_data', 'load_dataset', 'separate_dataset',
            'convert_to_numbers', 'train_model', 'evaluate_model', 'retrain_model',
            'display_mispredicted', 'make_predictions', 'send_completetion_email']

    def test_combined_runs_exact_sequence_using_shared_dispatch(self):
        invoked = []
        for task in self.tasks:
            setattr(self.runner, task + '__task', lambda task=task, **kw:
                (invoked.append(task) or {}))
        with patch.dict(self.runner.run.__func__.__globals__,
                        {'get_tasks': lambda *a, **kw: self.tasks}):
            self.assertFalse(self.runner.run(self.tasks))
        self.assertEqual(invoked, self.tasks)

    def test_wrong_shared_layer_rejected_before_any_task(self):
        self.handler.datalayer = Mock()
        with self.assertRaisesRegex(ValueError, 'prepared training and shared'):
            self.runner.run(self.tasks)
        self.logger.initiate_progress.assert_not_called()

    def test_unexpected_tasks_or_diagnostics_rejected(self):
        with patch.dict(self.runner.run.__func__.__globals__,
                        {'get_tasks': lambda *a, **kw: self.tasks}):
            with self.assertRaisesRegex(ValueError, 'Unsupported CSV combined'):
                self.runner.run(self.tasks[:-1])
            self.config.should_calculate_feature_importance.return_value = True
            with self.assertRaisesRegex(ValueError, 'Unsupported CSV combined'):
                self.runner.run(self.tasks)
        self.logger.initiate_progress.assert_not_called()

    def test_fetch_reads_one_snapshot_for_both_partitions(self):
        self.layer.read_partitions = Mock(return_value=(
            (('a', '001', 'A'), ('b', '002', 'B')),
            (('c', '003', ''),)))
        outcome = self.runner.fetch_data__task()
        self.assertEqual(outcome['data'], [('a', '001', 'A'),
                                           ('b', '002', 'B'), ('c', '003', '')])
        self.layer.read_partitions.assert_called_once_with()

    def test_invalid_partition_aborts_without_returning_data(self):
        self.layer.read_partitions = Mock(side_effect=ValueError('invalid synthetic classes'))
        with self.assertRaisesRegex(ValueError, 'invalid synthetic classes'):
            self.runner.fetch_data__task()


if __name__ == '__main__':
    unittest.main()
