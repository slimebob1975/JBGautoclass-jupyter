"""Synthetic regression checks for read-only CSV checkpoint discovery diagnostics."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Storage.CSVSavedModelInspection import diagnose_saved_csv_models, list_saved_csv_models


class CSVModelDiagnostics264b(unittest.TestCase):
    def setUp(self):
        self.repository = Mock()
        self.scope = Mock()
        self.dataset = 'synthetic-dataset'

    def test_configuration_mismatch_is_visible_without_exposing_details(self):
        with patch('Storage.CSVSavedModelInspection.list_csv_model_configurations',
                   return_value=(('synthetic', 'config'),)), \
             patch('Storage.CSVSavedModelInspection.inspect_saved_csv_model',
                   side_effect=ValueError('Saved CSV checkpoint configuration differs from registration')):
            entries, reasons, count = diagnose_saved_csv_models(
                self.repository, self.scope, self.dataset, model_root=Path.cwd())
        self.assertEqual(entries, ())
        self.assertEqual(count, 1)
        self.assertEqual(reasons, {'configuration_mismatch': 1})

    def test_missing_checkpoint_is_not_offered(self):
        with patch('Storage.CSVSavedModelInspection.list_csv_model_configurations',
                   return_value=(('synthetic', 'config'),)), \
             patch('Storage.CSVSavedModelInspection.inspect_saved_csv_model',
                   side_effect=ValueError('No saved CSV model exists for this dataset')):
            entries, reasons, count = diagnose_saved_csv_models(
                self.repository, self.scope, self.dataset, model_root=Path.cwd())
        self.assertEqual((entries, reasons, count), ((), {'checkpoint_missing': 1}, 1))

    def test_only_verified_models_are_selectable(self):
        records = (('valid', 'one'), ('invalid', 'two'))
        def inspect(*args, **kwargs):
            if args[3] == 'two':
                raise ValueError('CSV model source identity mismatch')
            return Path('synthetic.sav')
        with patch('Storage.CSVSavedModelInspection.list_csv_model_configurations', return_value=records), \
             patch('Storage.CSVSavedModelInspection.inspect_saved_csv_model', side_effect=inspect):
            entries, reasons, count = diagnose_saved_csv_models(
                self.repository, self.scope, self.dataset, model_root=Path.cwd())
            self.assertEqual(len(list_saved_csv_models(
                self.repository, self.scope, self.dataset, model_root=Path.cwd())), 1)
        self.assertEqual(len(entries), 1)
        self.assertEqual((reasons, count), ({'source_mismatch': 1}, 2))

    def test_no_configuration_is_distinct_from_missing_checkpoint(self):
        with patch('Storage.CSVSavedModelInspection.list_csv_model_configurations', return_value=()):
            self.assertEqual(diagnose_saved_csv_models(
                self.repository, self.scope, self.dataset, model_root=Path.cwd()), ((), {}, 0))


class GUIMessage264b(unittest.TestCase):
    def test_gui_status_reports_mismatch_and_does_not_unlock(self):
        source = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_csv_refresh_saved_models')
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
            name='Target', bases=[], keywords=[], body=[fn], decorator_list=[])], type_ignores=[]))
        ns = {'Config': SimpleNamespace(DEFAULT_TRAIN_OPTION='New model')}
        exec(compile(module, str(source), 'exec'), ns)
        handler = SimpleNamespace(lock_observe_1=False)
        target = SimpleNamespace(
            _csv_current_dataset_artifact=lambda: 'dataset', _csv_view_active=lambda: True,
            _csv_catalog_storage=lambda: (Mock(), Mock()),
            _csv_update_erase_model_button=Mock(), _csv_set_model_status=Mock(),
            update_ready=Mock(),
            _csv_source_panel=SimpleNamespace(status=SimpleNamespace(value='')),
            eventhandler=handler,
            models_dropdown=SimpleNamespace(options=(), value=None))
        with patch('Storage.CSVSavedModelInspection.diagnose_saved_csv_models',
                   return_value=((), {'configuration_mismatch': 1}, 1)):
            result = ns['Target']._csv_refresh_saved_models(target)
        self.assertEqual(result, 0)
        self.assertIn('training configuration', target._csv_source_panel.status.value)
        self.assertNotIn('synthetic-dataset', target._csv_source_panel.status.value)
        self.assertEqual(len(target.models_dropdown.options), 2)
        self.assertFalse(handler.lock_observe_1)


if __name__ == '__main__':
    unittest.main()
