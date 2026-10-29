"""Revision 230: production DatasetHandler ingestion preserves CSV keys before splitting.

Runs the real handler methods via dependency-light source extraction. This is NOT
full JBG TaskRunner/ModelHandler initialization or proof of SQL isolation.
"""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from JBGPredictionKeys import prediction_key


def production_methods():
    source = Path(__file__).resolve().parents[1] / 'src' / 'JBGHandler.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DatasetHandler')
    required = {'validate_column', 'validate_dataset', '_normalize_class_column',
                'shuffle_dataset', 'split_keys_from_dataset', 'load_data',
                'separate_dataset', 'get_num_unpredicted_rows',
                'split_dataset_for_training_and_validation'}
    nodes = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in required:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                arg.annotation = None
            nodes.append(node)
    namespace = {'np': np, 'pd': pd, 'Parallel': Parallel, 'delayed': delayed,
                 'prediction_key': prediction_key, 'DatasetException': ValueError,
                 'Helpers': SimpleNamespace(
                     clean_list=lambda values: list(dict.fromkeys(v for v in values if v is not None and str(v).strip())),
                     bytes_to_suffix=lambda size: str(size),
                     get_rid_of_decimals=lambda v: int(float(v)))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), namespace)
    return {name: namespace[name] for name in required}


METHODS = production_methods()


class RealDatasetCSVIngest230(unittest.TestCase):
    def build(self, source='csv'):
        config = SimpleNamespace(
            data_source=SimpleNamespace(kind=source),
            get_column_names=lambda: ['value', 'uid', 'label'],
            get_id_column_name=lambda: 'uid',
            get_class_column_name=lambda: 'label',
            column_is_text=lambda col: False,
            should_train=lambda: False)
        logger = Mock()
        handler = SimpleNamespace(config=config, logger=logger, STANDARD_DESIRED_N_JOBS=1)
        obj = SimpleNamespace(handler=handler)
        for name in METHODS:
            setattr(obj, name, METHODS[name] if name == "_normalize_class_column" else METHODS[name].__get__(obj))
        return obj

    def test_csv_real_load_data_retains_text_ids(self):
        obj = self.build()
        obj.load_data([('0.1', '005', ''), ('0.8', 'A-17', ''), ('0.2', '010', '')])
        self.assertEqual(obj.keys.tolist(), ['005', '010', 'A-17'])
        self.assertEqual(obj.dataset.index.tolist(), ['005', '010', 'A-17'])
        self.assertEqual(obj.dataset.value.tolist(), [0.1, 0.2, 0.8])
        obj.separate_dataset()
        self.assertEqual(obj.unpredicted_keys.tolist(), ['005', '010', 'A-17'])
        self.assertEqual(obj.X_prediction.index.tolist(), ['005', '010', 'A-17'])
        self.assertFalse(obj.split_dataset_for_training_and_validation())

    def test_csv_opaque_numeric_ids_do_not_collapse(self):
        obj = self.build()
        obj.load_data([('1', '5', ''), ('2', '005', '')])
        self.assertEqual(obj.keys.tolist(), ['005', '5'])

    def test_sql_null_and_blank_labels_are_prediction_rows(self):
        # SQL NULL may reach pandas as None, np.nan or pd.NA; all are unknown.
        obj = self.build('sql')
        obj.load_data([
            ('0.1', '101', 'negative'), ('0.2', '102', np.nan),
            ('0.3', '103', 'positive'), ('0.4', '104', None),
            ('0.5', '105', '   '), ('0.6', '106', pd.NA),
        ])
        self.assertEqual(obj.get_num_unpredicted_rows(), 4)
        obj.separate_dataset()
        self.assertEqual(obj.Y.tolist(), ['negative', 'positive'])
        self.assertEqual(len(obj.X), 2)
        self.assertEqual(len(obj.X_prediction), 4)
        self.assertEqual(obj.unpredicted_keys.tolist(), [102, 104, 105, 106])

    def test_unknown_sql_rows_are_not_in_training_when_prediction_disabled(self):
        obj = self.build('sql')
        obj.handler.config.should_predict = lambda: False
        obj.handler.config.should_train = lambda: True
        # The common dataset partitioner excludes unknown labels from training,
        # while the task sequence decides whether prediction is performed.
        obj.load_data([
            ('0.1', '101', 'negative'), ('0.2', '102', np.nan),
            ('0.3', '103', 'positive'), ('0.4', '104', ''),
        ])
        obj.separate_dataset()
        self.assertEqual(obj.Y.tolist(), ['negative', 'positive'])
        self.assertFalse(obj.Y.isna().any())
        self.assertEqual(len(obj.X_prediction), 2)

    def test_sql_numeric_id_validation_unchanged(self):
        obj = self.build('sql')
        result = obj.validate_column('uid', pd.Series(['005', '010']))
        self.assertEqual(result.tolist(), [5, 10])

    def test_non_id_csv_feature_still_numeric(self):
        obj = self.build()
        result = obj.validate_column('value', pd.Series(['0.2', '0.8']))
        self.assertEqual(result.tolist(), [0.2, 0.8])


if __name__ == '__main__':
    unittest.main()
