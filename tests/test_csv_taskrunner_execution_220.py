"""Revision 220: execute the production TaskRunner task methods against a CSV fixture.

Uses real TaskRunner dispatch and task implementations, a real sklearn estimator
and scoped CSV persistence. Dataset/model/prediction handlers are controlled test
adapters; this does NOT validate the complete JBG handler lifecycle or GUI.
"""
import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from Storage.ArtifactContract import ArtifactKind
from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer
import test_csv_prediction_integration_216 as fixture216


TASKS = ['load_model', 'fetch_data', 'load_dataset', 'separate_dataset', 'make_predictions']


def production_task_methods():
    """Compile unmodified production dispatch/task methods, avoiding GUI bootstrap."""
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskRunner')
    names = {'run'} | {name + '__task' for name in TASKS}
    nodes = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'os': __import__('os'), 'ModelException': RuntimeError,
                 'DatasetException': RuntimeError, 'PredictionsException': RuntimeError,
                 'Helpers': SimpleNamespace(count_value_distr_as_dict=lambda arr: {
                     str(x): arr.count(x) for x in set(arr)}),
                 'CSVPredictionDataLayer': CSVPredictionDataLayer}
    for node in nodes:
        node.decorator_list = []
    # Production run() imports the facade itself; production tasks are unchanged.
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


class CSVTaskRunnerExecution220(unittest.TestCase):
    def setUp(self):
        fixture216.CSVPredictionIntegration216.setUp(self)
        self.saved, self.layer = fixture216.CSVPredictionIntegration216._prepare(self)
        self.logger = Mock()
        self.logger.abort_cleanly.side_effect = RuntimeError('Task execution aborted')
        model_handler = SimpleNamespace(model=None)
        def load_model(path):
            with open(path, 'rb') as stream:
                estimator = __import__('pickle').load(stream)
            model_handler.model = SimpleNamespace(pipeline=estimator)
        model_handler.load_model = load_model
        class Dataset:
            X_prediction = None
            classes = ['A', 'B']
            def load_data(inner, data):
                inner.data = data
            def separate_dataset(inner):
                inner.X_prediction = np.array([[float(row[0])] for row in inner.data])
            def split_dataset_for_training_and_validation(inner):
                pass
        dataset_handler = Dataset()
        class Predictions:
            could_predict_proba = True
            def make_predictions(inner, estimator, features, classes):
                inner.predictions = estimator.predict(features)
                inner.probabilities = estimator.predict_proba(features)
            def get_rates(inner, as_string=False):
                return [float(max(v)) for v in inner.probabilities]
        predictions_handler = Predictions()
        self.handlers = {'model': model_handler, 'dataset': dataset_handler,
                         'predictions': predictions_handler}
        def save():
            rows = [dict(key=row[1], prediction=prediction,
                         rate=float(max(probs)),
                         probabilities=','.join(str(float(v)) for v in probs))
                    for row, prediction, probs in zip(dataset_handler.data,
                        predictions_handler.predictions, predictions_handler.probabilities)]
            return self.layer.save_prediction_data(rows, fixture216.Rate.PROBABILITY,
                'synthetic-model', model_handler.model.pipeline.classes_)
        self.handler = SimpleNamespace(datalayer=self.layer,
             get_handler=lambda name: self.handlers[name], save_predictions=save)
        self.config = SimpleNamespace(data_source=SimpleNamespace(kind='csv'),
             should_train=lambda: False, get_text_column_names=lambda: [],
             get_model_filename=lambda: str(self.model_path))
        self.runner = SimpleNamespace(datalayer=self.layer, handler=self.handler,
             config=self.config, logger=self.logger, regression_suite=False)
        namespace = production_task_methods()
        for name in ('run', *(n + '__task' for n in TASKS)):
            setattr(self.runner, name, namespace[name].__get__(self.runner))

    def test_complete_dispatch_load_predict_publish(self):
        self.assertFalse(self.runner.run(TASKS))
        self.logger.update_progress.assert_called()
        refs = self.repo.list_artifacts(self.scope)
        private = [r for r in refs if r.kind == ArtifactKind.PREDICTIONS]
        self.assertEqual(len(private), 1)
        with self.repo.open_artifact(self.scope, private[0].artifact_id) as stream:
            rows = json.load(stream)['rows']
        self.assertEqual([r['key'] for r in rows], ['005', '006'])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['prediction'] in ('A', 'B') for r in rows))
        self.assertEqual(len([r for r in refs if r.kind == ArtifactKind.MANIFEST
                              and r.run_id == private[0].run_id]), 1)

    def test_sql_write_target_rejected_without_publication(self):
        self.handler.datalayer = Mock()
        with self.assertRaisesRegex(ValueError, 'data-layer mismatch'):
            self.runner.run(TASKS)
        self.assertFalse(any(r.kind == ArtifactKind.PREDICTIONS
                             for r in self.repo.list_artifacts(self.scope)))

    def test_forbidden_training_task_rejected_before_data_access(self):
        self.layer.get_dataset = Mock(side_effect=AssertionError('read must not happen'))
        with self.assertRaisesRegex(ValueError, 'task sequence'):
            self.runner.run(TASKS[:4] + ['train_model'] + TASKS[4:])
        self.layer.get_dataset.assert_not_called()

    def test_source_file_reference_is_project_bound(self):
        self.assertEqual([row[1] for row in self.layer.get_dataset()], ['005', '006'])
        self.assertIs(self.handler.datalayer, self.runner.datalayer)


if __name__ == '__main__':
    unittest.main()
