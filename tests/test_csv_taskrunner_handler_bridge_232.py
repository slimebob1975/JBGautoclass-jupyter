"""Revision 232: production TaskRunner dispatch with real dataset/prediction methods.

The task functions come from production source, but optional-dependency-heavy
application bootstrap and the ModelHandler model artifact format are still isolated.
This is deliberately NOT a complete live JBG bootstrap test.
"""
import ast
import json
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from Storage.ArtifactContract import ArtifactKind
from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer
import test_csv_handler_bridge_231 as bridge

TASKS = ['load_model', 'fetch_data', 'load_dataset', 'separate_dataset', 'make_predictions']


def taskrunner_methods():
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGTaskRunner.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskRunner')
    names = {'run'} | {name + '__task' for name in TASKS}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
    namespace = {'os': __import__('os'), 'ModelException': RuntimeError,
                 'DatasetException': RuntimeError, 'PredictionsException': RuntimeError,
                 'Helpers': SimpleNamespace(count_value_distr_as_dict=lambda a: {
                     str(value): a.count(value) for value in set(a)}),
                 'CSVPredictionDataLayer': CSVPredictionDataLayer}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


class CSVTaskRunnerHandlerBridge232(bridge.CSVHandlerBridge231):
    # Parent tests are run separately; this class changes model state in setUp.
    test_dataset_index_and_prediction_keys_remain_aligned = None
    test_saved_csv_rows_to_real_prediction_handlers = None
    test_real_handler_chain_writes_scoped_prediction_artifacts = None
    test_real_handler_reject_duplicate_csv_keys = None
    test_real_handler_rejects_duplicate_csv_keys = None
    test_real_handler_rejects_unknown_label_before_publication = None
    test_real_handler_requires_registered_handlers = None

    def setUp(self):
        super().setUp()
        # Real prediction implementation; only the heavyweight input utilities are
        # passed through, since the fitted test model accepts the numeric DataFrame.
        path = Path(__file__).resolve().parents[1] / 'src' / 'JBGHandler.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PredictionsHandler')
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('make_predictions', 'calculate_probability', 'get_rates')]
        for node in methods:
            node.returns = None
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                arg.annotation = None
        ns = {'np': np, 'prepare_model_input': lambda x, model: x,
              'call_with_sparse_input_retry': lambda fn, x, **kwargs: fn(x)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                     str(path), 'exec'), ns)
        for name in ('make_predictions', 'calculate_probability', 'get_rates'):
            setattr(self.predictions, name, ns[name].__get__(self.predictions))

        self.model.model = None
        def load_model(filename):
            with open(filename, 'rb') as stream:
                estimator = pickle.load(stream)  # Synthetic model created locally by fixture.
            self.model.model = SimpleNamespace(pipeline=estimator,
                class_labels=estimator.classes_, get_name=lambda: 'synthetic-model')
        self.model.load_model = load_model
        self.config.get_model_filename = lambda: str(self.model_path)
        self.config.get_text_column_names = lambda: []
        self.logger.abort_cleanly.side_effect = RuntimeError('task aborted')
        self.runner = SimpleNamespace(datalayer=self.layer, handler=self.handler,
                                      logger=self.logger, config=self.config,
                                      regression_suite=False)
        ns = taskrunner_methods()
        for name in ('run', *(n + '__task' for n in TASKS)):
            setattr(self.runner, name, ns[name].__get__(self.runner))

    def _artifacts(self):
        return [ref for ref in self.repo.list_artifacts(self.scope)
                if ref.kind == ArtifactKind.PREDICTIONS]

    def test_dispatch_loads_rows_predicts_and_publishes(self):
        self.assertFalse(self.runner.run(TASKS))
        refs = self._artifacts()
        self.assertEqual(len(refs), 1)
        with self.repo.open_artifact(self.scope, refs[0].artifact_id) as stream:
            result = json.load(stream)
        self.assertEqual([row['key'] for row in result['rows']], ['005', '006'])
        self.assertEqual(len(result['rows']), 2)
        self.assertEqual(self.logger.update_progress.call_count, len(TASKS))

    def test_sql_layer_mismatch_rejected_before_read(self):
        self.handler.datalayer = Mock()
        self.layer.get_dataset = Mock(side_effect=AssertionError('unexpected read'))
        with self.assertRaisesRegex(ValueError, 'data-layer mismatch'):
            self.runner.run(TASKS)
        self.layer.get_dataset.assert_not_called()
        self.assertFalse(self._artifacts())

    def test_training_sequence_rejected_before_read(self):
        self.layer.get_dataset = Mock(side_effect=AssertionError('unexpected read'))
        with self.assertRaisesRegex(ValueError, 'task sequence'):
            self.runner.run(TASKS[:-1] + ['train_model', 'make_predictions'])
        self.layer.get_dataset.assert_not_called()
        self.assertFalse(self._artifacts())

    def test_prediction_fails_closed_for_invalid_class(self):
        original = self.estimator.predict
        self.estimator.predict = lambda x: np.array(['UNEXPECTED'] * len(x))
        # Model loader must return the tampered, in-memory estimator only here.
        self.model.load_model = lambda path: setattr(self.model, 'model',
            SimpleNamespace(pipeline=self.estimator, class_labels=self.estimator.classes_,
                            get_name=lambda: 'synthetic-model'))
        with self.assertRaises(RuntimeError):
            self.runner.run(TASKS)
        self.assertFalse(self._artifacts())
        self.estimator.predict = original

if __name__ == '__main__':
    unittest.main()
