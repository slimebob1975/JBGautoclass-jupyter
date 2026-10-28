"""Revision 264c: restored role selections and immutable run-settings contract."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def method(name):
    path = ROOT / 'src/GUI/Widgets.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    target = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name='Harness', bases=[], keywords=[], body=[target], decorator_list=[])], type_ignores=[]))
    env = {}
    exec(compile(module, str(path), 'exec'), env)
    return getattr(env['Harness'], name)


class RestoredRoles264c(TestCase):
    def setUp(self):
        self.schema = SimpleNamespace(columns=('id', 'x', 'y', 'class'),
                                      class_candidates=('class',), unique_id_candidates=('id',))
        self.panel = SimpleNamespace(path=SimpleNamespace(value='synthetic.csv'),
                                     delimiter=SimpleNamespace(value=','),
                                     encoding=SimpleNamespace(value='utf-8'),
                                     status=SimpleNamespace(value=''))
        self.target = SimpleNamespace(
            _csv_view_active=lambda: True, _csv_current_dataset_artifact=lambda: 'dataset',
            _csv_catalog_storage=lambda: ('repo', 'scope'),
            _csv_model_selected=Mock(), _csv_refresh_saved_models=Mock(),
            _restore_source_roles=Mock(), models_dropdown=SimpleNamespace(disabled=True),
        )

    def run_validation(self, configs, lookup):
        with patch('CSVColumnSchema.inspect_csv_columns', return_value=self.schema), \
             patch('Storage.CSVModelConfiguration.list_csv_model_configurations', return_value=configs), \
             patch('Storage.CSVModelConfiguration.load_csv_model_configuration', side_effect=lambda *args: lookup[args[-1]]):
            method('_csv_validated_columns')(self.target, self.panel)
        return self.target._restore_source_roles.call_args.args[0]

    def test_saved_numeric_features_are_selected_after_validate(self):
        saved = SimpleNamespace(class_column='class', id_column='id', data_columns=('x', 'y'), text_columns=())
        roles = self.run_validation((('first', 'config'),), {'config': saved})
        self.assertEqual(roles['data_columns'], (('x', 'y'), ('x', 'y')))
        self.assertEqual(roles['text_columns'], ((), ()))
        self.target._csv_refresh_saved_models.assert_called_once()

    def test_conflicting_saved_roles_do_not_guess(self):
        a = SimpleNamespace(class_column='class', id_column='id', data_columns=('x',), text_columns=())
        b = SimpleNamespace(class_column='class', id_column='id', data_columns=('y',), text_columns=())
        roles = self.run_validation((('a', 'a'), ('b', 'b')), {'a': a, 'b': b})
        self.assertEqual(roles['data_columns'][1], ())

    def test_no_saved_configuration_keeps_empty_selection(self):
        roles = self.run_validation((), {})
        self.assertEqual(roles['data_columns'][1], ())


class ImmutableSettings264c(TestCase):
    def test_model_parameter_grid_tuning_not_used_as_registration_identity(self):
        source = (ROOT / 'src/Storage/CSVSavedModelInspection.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'inspect_saved_csv_model')
        # Dataset and role identity are checked independently from training choices.
        # Hyperparameter tuning must not become part of dataset registration identity.
        self.assertNotIn('model_parameters', ast.unparse(fn))
        self.assertNotIn('registered_mode', ast.unparse(fn))
        self.assertIn('get_numerical_column_names', ast.unparse(fn))
        self.assertIn('get_text_column_names', ast.unparse(fn))
