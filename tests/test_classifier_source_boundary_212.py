"""Source-aware saved-model initialization without importing heavy ML libraries."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from Storage.CSVPredictionDataLayer import CSVPredictionDataLayer


class _ConfigError(Exception):
    pass


def _constructor():
    source = Path(__file__).resolve().parents[1] / 'src' / 'AutomaticClassifier.py'
    module = ast.parse(source.read_text(encoding='utf-8'))
    klass = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'AutomaticClassifier')
    func = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    func.decorator_list = []
    func.returns = None
    func.args.args = [ast.arg(arg=a.arg) for a in func.args.args]
    # Remove annotations from AST so this test doesn't require third-party imports.
    for arg in func.args.args:
        arg.annotation = None
    scope = {'ConfigException': _ConfigError, 'Config': SimpleNamespace(Config=SimpleNamespace(
        load_config_from_model_file=Mock(side_effect=lambda path, config: config))),
        'load_model_compatibility': Mock(return_value='compat'),
        'review_current_source_compatibility': Mock(return_value=SimpleNamespace(issues=[], warnings=[])),
        'time': SimpleNamespace(time=lambda: 0)}
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(source), 'exec'), scope, namespace)
    return namespace['__init__'], scope


def _input(kind, layer):
    config = SimpleNamespace(should_train=lambda: False, get_model_filename=lambda: 'model',
        data_source=SimpleNamespace(kind=kind), connection=SimpleNamespace(
            data_catalog='catalog', data_table='table', odbc_driver='driver', host='host'))
    obj = SimpleNamespace(create_handlers=lambda: (1, 2, 3))
    return obj, config, layer


class ClassifierSourceBoundaryTests(TestCase):
    def test_csv_skips_sql_probes(self):
        ctor, scope = _constructor()
        obj, config, layer = _input('csv', CSVPredictionDataLayer(Mock(), Mock()))
        ctor(obj, config, Mock(), layer)
        layer._reader.get_dataset.assert_not_called()
        scope['load_model_compatibility'].assert_not_called()
        scope['review_current_source_compatibility'].assert_not_called()
        scope['Config'].Config.load_config_from_model_file.assert_called_once_with('model', config=config)

    def test_csv_rejects_sql_layer(self):
        ctor, scope = _constructor()
        obj, config, layer = _input('csv', Mock())
        with self.assertRaises(_ConfigError):
            ctor(obj, config, Mock(), layer)
        scope['Config'].Config.load_config_from_model_file.assert_not_called()

    def test_sql_preserves_compatibility_review(self):
        ctor, scope = _constructor()
        obj, config, layer = _input('sql', Mock())
        layer.get_table_columns.return_value = 'schema'
        ctor(obj, config, Mock(), layer)
        layer.get_table_columns.assert_called_once_with('catalog', 'table')
        scope['review_current_source_compatibility'].assert_called_once()
        scope['Config'].Config.load_config_from_model_file.assert_called_once()

    def test_sql_rejects_incompatible_schema(self):
        ctor, scope = _constructor()
        scope['review_current_source_compatibility'].return_value = SimpleNamespace(issues=['incompatible'], warnings=[])
        obj, config, layer = _input('sql', Mock())
        with self.assertRaises(_ConfigError):
            ctor(obj, config, Mock(), layer)
        scope['Config'].Config.load_config_from_model_file.assert_not_called()

    def test_unknown_source_fails_closed(self):
        ctor, scope = _constructor()
        obj, config, layer = _input('other', Mock())
        with self.assertRaises(_ConfigError):
            ctor(obj, config, Mock(), layer)
        scope['Config'].Config.load_config_from_model_file.assert_not_called()
