"""Revision 233: genuine versioned JBG serialization and ModelHandler loader.

The on-disk format and production loader are real. Compatibility semantics and
heavy application bootstrap are isolated; this is not a live JBG startup test.
"""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from JBGModelPersistence import save_model_artifact, load_model_artifact_with_compatibility
from Storage.ArtifactContract import ArtifactKind
import test_csv_taskrunner_handler_bridge_232 as bridge


def production_model_loader():
    path = Path(__file__).resolve().parents[1] / 'src' / 'JBGHandler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'ModelHandler')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('load_model', 'load_model_from_file')]
    for node in methods:
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
    class RestoredModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.class_labels = kwargs['pipeline'].classes_
        def get_name(self):
            return 'synthetic-model'
    ns = {'load_model_artifact_with_compatibility': load_model_artifact_with_compatibility,
          'review_artifact_semantics': lambda config, converter, metadata:
              SimpleNamespace(issues=() if metadata.get('schema_version') == 'test.v1' else ('invalid metadata',), warnings=()),
          'Model': RestoredModel, 'ModelInitializationException': ValueError,
          'keras': SimpleNamespace(models=SimpleNamespace(load_model=Mock(side_effect=AssertionError('keras forbidden')))),
          'resolve_keras_model_sidecar_path': Mock(side_effect=AssertionError('keras forbidden')),
          'attach_keras_model_to_pipeline': Mock(side_effect=AssertionError('keras forbidden'))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(path), 'exec'), ns)
    return ns


class CSVModelReload233(bridge.CSVTaskRunnerHandlerBridge232):
    test_dispatch_loads_rows_predicts_and_publishes = None
    test_sql_layer_mismatch_rejected_before_read = None
    test_training_sequence_rejected_before_read = None
    test_prediction_fails_closed_for_invalid_class = None

    def setUp(self):
        super().setUp()
        # Save the existing fitted estimator in JBG's actual versioned format.
        self.jbg_model_path = self.model_path.with_name('trusted-current-jbg-model.pkl')
        save_model_artifact(self.jbg_model_path, bridge.bridge.handlers.fixture216.SavedConfig(self.ref), None,
                            (None, None, None, None, None), self.estimator,
                            None, 1, compatibility={'schema_version': 'test.v1'})
        self.config.get_model_filename = lambda: str(self.jbg_model_path)
        self.config.set_num_selected_features = Mock()
        loader = production_model_loader()
        self.model.handler = self.handler
        for method in ('load_model', 'load_model_from_file'):
            setattr(self.model, method, loader[method].__get__(self.model))

    def _predictions(self):
        refs = [ref for ref in self.repo.list_artifacts(self.scope)
                if ref.kind == ArtifactKind.PREDICTIONS]
        self.assertEqual(len(refs), 1)
        with self.repo.open_artifact(self.scope, refs[0].artifact_id) as handle:
            return json.load(handle)['rows']

    def test_versioned_jbg_model_reloads_via_production_method_and_runner(self):
        self.assertFalse(self.runner.run(bridge.TASKS))
        self.assertEqual([row['key'] for row in self._predictions()], ['005', '006'])
        self.assertEqual(self.model.model.n_features_out, 1)
        self.assertEqual(set(self.model.model.class_labels), {'A', 'B'})
        self.assertEqual(self.model.model.pipeline.predict(np.array([[0.2], [0.8]])).shape, (2,))

    def test_corrupt_model_format_refuses_publication(self):
        self.jbg_model_path.write_bytes(b'not a saved JBG model')
        # Production loader handles failures by returning None. TaskRunner must
        # not publish predictions when the model cannot be restored.
        with self.assertRaises(Exception):
            self.runner.run(bridge.TASKS)
        self.assertFalse([r for r in self.repo.list_artifacts(self.scope)
                          if r.kind == ArtifactKind.PREDICTIONS])

    def test_versioned_artifact_contains_six_payload_fields(self):
        payload, metadata = load_model_artifact_with_compatibility(self.jbg_model_path)
        self.assertEqual(len(payload), 6)
        self.assertEqual(metadata['schema_version'], 'test.v1')
        self.assertIsNone(payload[4])
        self.assertEqual(payload[5], 1)


if __name__ == '__main__':
    unittest.main()
