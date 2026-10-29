"""Revision 254: native classifier/dispatcher orchestration; no heavy fitting."""
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

try:
    from AutomaticClassifier import AutomaticClassifier
    from JBGTaskRunner import TaskRunner, get_tasks
    from JBGExceptions import DatasetException
    from Storage.CSVTrainingDataLayer import CSVTrainingDataLayer
    from Storage.CSVTrainingAdapter import CSVTrainingAdapter
    import pyodbc
except (ImportError, OSError):
    AutomaticClassifier = None


@unittest.skipIf(AutomaticClassifier is None, 'Optional JBG dependencies unavailable')
class CSVTrainingClassifier254(unittest.TestCase):
    def setUp(self):
        reader = object.__new__(CSVTrainingAdapter)
        reader.correction_revision = None
        reader.get_dataset = Mock(return_value=[(5.1, '001', 'setosa'), (6.0, '002', 'versicolor')])
        self.layer = CSVTrainingDataLayer(reader)
        self.config = Mock()
        self.config.data_source.kind = 'csv'
        self.config.should_train.return_value = True
        self.config.should_predict.return_value = False
        self.config.should_display_mispredicted.return_value = False
        self.config.should_calculate_feature_importance.return_value = False
        self.config.get_text_column_names.return_value = []
        self.logger = Mock()
        self.handler = Mock(datalayer=self.layer)
        self.handler.get_handler.side_effect = lambda name: getattr(self, name)
        self.dataset = Mock()
        self.model = Mock()
        self.predictions = Mock()
        self.dataset.classes = ['setosa', 'versicolor']

    def classifier(self):
        def handlers(classifier):
            classifier.handler = self.handler
            return self.dataset, self.model, self.predictions
        with patch.object(AutomaticClassifier, 'create_handlers', handlers):
            return AutomaticClassifier(config=self.config, logger=self.logger,
                                       datalayer=self.layer)

    def test_native_classifier_runs_csv_training_tasks_and_calls_both_save_steps(self):
        classifier = self.classifier()
        tasks = get_tasks(self.config)
        seen = []
        def step(name):
            def invoke(runner, **kwargs):
                seen.append(name)
                if name == 'fetch_data':
                    return {'data': self.layer.get_dataset()}
                if name == 'load_dataset':
                    self.assertEqual(len(kwargs['data']), 2)
                if name == 'train_model' or name == 'retrain_model':
                    self.model.save_model_to_file(self.config.get_model_filename())
                return {}
            return invoke
        methods = {name + '__task': step(name) for name in tasks}
        with patch.multiple(TaskRunner, **methods), patch.object(pyodbc, 'connect', side_effect=AssertionError('SQL access')) as sql:
            result = classifier.run()
        self.assertEqual(result, {'mispredicted': None})
        self.assertEqual(seen, tasks)
        self.assertEqual(self.model.save_model_to_file.call_count, 2)
        sql.assert_not_called()

    def test_empty_csv_training_raises_and_does_not_reach_training(self):
        self.layer._reader.get_dataset.return_value = []
        runner = TaskRunner(self.layer, self.config, self.logger, self.handler)
        with patch.object(runner, 'train_model__task') as trainer:
            with self.assertRaisesRegex(DatasetException, 'no labelled rows'):
                runner.fetch_data__task()
        trainer.assert_not_called()

    def test_empty_csv_prediction_still_uses_early_exit(self):
        self.layer._reader.get_dataset.return_value = []
        self.config.should_train.return_value = False
        self.config.should_predict.return_value = True
        runner = TaskRunner(self.layer, self.config, self.logger, self.handler)
        self.assertEqual(runner.fetch_data__task(), {'early_exit': True})

    def test_sql_none_dataset_keeps_legacy_early_exit(self):
        self.config.data_source.kind = 'sql'
        self.layer._reader.get_dataset.return_value = None
        runner = TaskRunner(self.layer, self.config, self.logger, self.handler)
        self.assertEqual(runner.fetch_data__task(), {'early_exit': True})


class CSVTrainingFetchPortable254(unittest.TestCase):
    """Portable production-method coverage when heavyweight ML deps are absent."""
    @staticmethod
    def fetch_method():
        import ast
        from pathlib import Path
        source = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'TaskRunner')
        method = next(x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name == 'fetch_data__task')
        env = {'DatasetException': ValueError}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(source), 'exec'), env)
        return env['fetch_data__task']

    def test_empty_training_raises_using_actual_fetch_method(self):
        layer = Mock(get_dataset=Mock(return_value=[]))
        cfg = Mock()
        cfg.data_source.kind = 'csv'
        cfg.should_train.return_value = True
        cfg.should_predict.return_value = False
        runner = SimpleNamespace(config=cfg, datalayer=layer, logger=Mock())
        with self.assertRaisesRegex(ValueError, 'no labelled rows'):
            self.fetch_method()(runner)

    def test_prediction_empty_and_sql_none_remain_no_work(self):
        cfg = Mock()
        cfg.data_source.kind = 'csv'
        cfg.should_train.return_value = False
        runner = SimpleNamespace(config=cfg, datalayer=Mock(get_dataset=Mock(return_value=[])), logger=Mock())
        self.assertEqual(self.fetch_method()(runner), {'early_exit': True})
        cfg.data_source.kind = 'sql'
        runner.datalayer.get_dataset.return_value = None
        self.assertEqual(self.fetch_method()(runner), {'early_exit': True})


if __name__ == '__main__':
    unittest.main()
