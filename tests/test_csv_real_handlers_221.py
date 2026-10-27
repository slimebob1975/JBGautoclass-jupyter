"""Revision 221: real JBG prediction-handler methods with a scoped CSV sink.

Uses unmodified JBGHandler/PredictionsHandler methods compiled from the
production source, avoiding optional GUI/SQL bootstrap. Dataset/model lifecycle
is still controlled; this is NOT a full live classifier smoke test.
"""
import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from JBGPredictionKeys import prediction_key
from Storage.ArtifactContract import ArtifactKind
import test_csv_prediction_integration_216 as fixture216


def handler_methods():
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGHandler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    wanted = {
        'JBGHandler': ('get_handler', 'save_predictions'),
        'PredictionsHandler': ('get_prediction_results', 'get_probablities_as_string',
                               'save_predictions', 'get_rate_type'),
        'ModelHandler': ('get_class_labels',),
    }
    out = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in wanted:
            nodes = [m for m in cls.body if isinstance(m, ast.FunctionDef)
                     and m.name in wanted[cls.name]]
            for node in nodes:
                node.returns = None
                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    arg.annotation = None
            names = {'prediction_key': prediction_key, 'HandlerException': RuntimeError,
                     'RateType': SimpleNamespace(I=fixture216.Rate.PROBABILITY,
                                                 U=fixture216.Rate.PROBABILITY,
                                                 A=fixture216.Rate.PROBABILITY)}
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                         str(path), 'exec'), names)
            for name in wanted[cls.name]:
                out[(cls.name, name)] = names[name]
    return out


METHODS = handler_methods()


class CSVRealHandlers221(unittest.TestCase):
    def setUp(self):
        fixture216.CSVPredictionIntegration216.setUp(self)
        self.config, self.layer = fixture216.CSVPredictionIntegration216._prepare(self)
        import pickle
        with self.model_path.open('rb') as stream:
            self.estimator = pickle.load(stream)
        rows = self.layer.get_dataset()
        self.keys = pd.Series([r[1] for r in rows])
        features = np.array([[float(r[0])] for r in rows])
        self.logger = Mock()
        self.handler = SimpleNamespace(datalayer=self.layer, config=self.config,
                                       logger=self.logger, handlers={})
        self.handler.get_handler = METHODS['JBGHandler', 'get_handler'].__get__(self.handler)
        self.handler.save_predictions = METHODS['JBGHandler', 'save_predictions'].__get__(self.handler)
        self.dataset = SimpleNamespace(unpredicted_keys=self.keys, Y=pd.Series(dtype=object))
        self.model = SimpleNamespace(model=SimpleNamespace(
            class_labels=self.estimator.classes_, get_name=lambda: 'synthetic-model'))
        self.model.get_class_labels = METHODS['ModelHandler', 'get_class_labels'].__get__(self.model)
        self.predictions = SimpleNamespace(handler=self.handler,
            predictions=self.estimator.predict(features),
            probabilites=self.estimator.predict_proba(features))
        self.predictions.rates = np.max(self.predictions.probabilites, axis=1)
        self.predictions.could_predict_proba = True
        for method in ('get_prediction_results', 'get_probablities_as_string',
                       'save_predictions', 'get_rate_type'):
            setattr(self.predictions, method,
                    METHODS['PredictionsHandler', method].__get__(self.predictions))
        self.handler.handlers.update(dataset=self.dataset, model=self.model,
                                     predictions=self.predictions)

    def test_real_handler_chain_writes_scoped_prediction_artifacts(self):
        result = self.handler.save_predictions()
        self.assertIsNone(result['error'])
        refs = self.repo.list_artifacts(self.scope, run_id=result['results']['run_id'])
        self.assertEqual({r.kind for r in refs},
                         {ArtifactKind.PREDICTIONS, ArtifactKind.MANIFEST})
        payload = next(r for r in refs if r.kind == ArtifactKind.PREDICTIONS)
        with self.repo.open_artifact(self.scope, payload.artifact_id) as stream:
            stored = json.load(stream)['rows']
        self.assertEqual([r['key'] for r in stored], ['005', '006'])
        self.assertEqual([r['prediction'] for r in stored],
                         self.predictions.predictions.tolist())
        self.assertTrue(all(len(r['probabilities'].split(',')) == 2 for r in stored))

    def test_real_handler_rejects_unknown_label_before_publication(self):
        self.predictions.predictions[0] = 'UNKNOWN'
        with self.assertRaises(RuntimeError):
            self.handler.save_predictions()
        self.assertFalse(any(r.kind == ArtifactKind.PREDICTIONS
                             for r in self.repo.list_artifacts(self.scope)))

    def test_real_handler_requires_registered_handlers(self):
        self.handler.handlers.pop('model')
        with self.assertRaises(RuntimeError):
            self.handler.save_predictions()
        self.assertFalse(any(r.kind == ArtifactKind.PREDICTIONS
                             for r in self.repo.list_artifacts(self.scope)))

    def test_real_handler_rejects_duplicate_csv_keys(self):
        self.dataset.unpredicted_keys = pd.Series(['005', '005'])
        with self.assertRaises(RuntimeError):
            self.handler.save_predictions()
        self.assertFalse(any(r.kind == ArtifactKind.PREDICTIONS
                             for r in self.repo.list_artifacts(self.scope)))


if __name__ == '__main__':
    unittest.main()
