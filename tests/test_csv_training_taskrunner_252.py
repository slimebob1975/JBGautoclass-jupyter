"""Revision 252: training is read-only and uses the shared task dispatcher."""
import ast
from uuid import uuid4
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from Storage.ArtifactContract import StorageScope
from Storage.CSVTrainingAdapter import CSVTrainingAdapter
from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer, prepare_csv_training_datalayer


def runner_dispatch():
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskRunner')
    run = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    run.decorator_list = []
    scope = {'get_tasks': lambda config, regression_suite=False: []}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[])), str(path), 'exec'), scope)
    return scope['run']


class CSVTrainingTaskrunner252(TestCase):
    def setUp(self):
        self.config = Mock()
        self.config.data_source.kind = 'csv'
        self.config.should_train.return_value = True
        self.config.should_predict.return_value = False
        self.config.get_text_column_names.return_value = []
        self.config.should_calculate_feature_importance.return_value = False
        self.reader = object.__new__(CSVTrainingAdapter)
        self.reader.correction_revision = None
        self.reader.get_dataset = Mock(return_value=[('3.1', '005', 'setosa')])
        self.layer = CSVTrainingDataLayer(self.reader)
        self.logger = Mock()
        self.handler = Mock(datalayer=self.layer)
        self.runner = SimpleNamespace(config=self.config, datalayer=self.layer,
                                      handler=self.handler, logger=self.logger,
                                      regression_suite=False)
        self.runner.run = runner_dispatch().__get__(self.runner)

    def test_training_follows_unmodified_common_task_order(self):
        # Dispatch is genuine; individual ML tasks are counters, not heavy model fits.
        tasks = ['load_model', 'fetch_data', 'load_dataset', 'separate_dataset',
                 'train_model', 'evaluate_model', 'retrain_model', 'display_mispredicted',
                 'send_completetion_email']
        executed = []
        for name in tasks:
            setattr(self.runner, name + '__task', lambda name=name, **kwargs:
                    (executed.append(name) or {}))
        with patch.dict(self.runner.run.__func__.__globals__, {'get_tasks': lambda c, regression_suite=False: tasks}):
            self.assertFalse(self.runner.run(tasks))
        self.assertEqual(executed, tasks)

    def test_wrong_data_layer_or_mixed_mode_stops_before_dispatch(self):
        tasks = ['load_model']
        with patch.dict(self.runner.run.__func__.__globals__, {'get_tasks': lambda c, regression_suite=False: tasks}):
            self.handler.datalayer = Mock()
            with self.assertRaisesRegex(ValueError, 'prepared training'):
                self.runner.run(tasks)
            self.handler.datalayer = self.layer
            self.config.should_predict.return_value = True
            with self.assertRaisesRegex(ValueError, 'prepared training'):
                self.runner.run(tasks)
        self.logger.initiate_progress.assert_not_called()

    def test_unknown_training_task_stops_before_dispatch(self):
        with patch.dict(self.runner.run.__func__.__globals__, {'get_tasks': lambda c, regression_suite=False: ['load_model']}):
            with self.assertRaisesRegex(ValueError, 'task sequence'):
                self.runner.run(['load_model', 'make_predictions'])
        self.logger.initiate_progress.assert_not_called()

    def test_training_facade_never_exposes_sql_publication(self):
        # The single-row fixture tests the read-only facade, not ML preflight.
        self.assertEqual(self.reader.get_dataset(), [('3.1', '005', 'setosa')])
        self.assertIsNone(self.layer.correction_revision)
        self.assertFalse(hasattr(self.layer, 'save_prediction_data'))

    def test_factory_rejects_prediction_and_source_mismatch(self):
        self.config.should_predict.return_value = True
        with self.assertRaisesRegex(ValueError, 'train-only'):
            prepare_csv_training_datalayer(self.config, Mock(),
                 StorageScope(str(uuid4()), str(uuid4())), 'config')
        self.config.should_predict.return_value = False
        saved = SimpleNamespace(dataset_id='dataset', artifact_id='artifact')
        self.config.data_source.dataset_id = 'other'
        with patch('Storage.CSVTrainingDataLayer.load_csv_model_configuration', return_value=saved):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                prepare_csv_training_datalayer(self.config, Mock(),
                     StorageScope(str(uuid4()), str(uuid4())), 'config')


if __name__ == '__main__':
    main()
