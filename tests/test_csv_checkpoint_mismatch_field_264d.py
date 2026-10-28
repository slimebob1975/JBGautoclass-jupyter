"""264d: safely identify mismatching configuration keys, never payload values."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from Storage.CSVSavedModelInspection import diagnose_saved_csv_models


class MismatchField264d(unittest.TestCase):
    def test_safe_known_field_is_reported(self):
        with patch('Storage.CSVSavedModelInspection.list_csv_model_configurations', return_value=(("synthetic", "config"),)), patch(
            'Storage.CSVSavedModelInspection.inspect_saved_csv_model',
            side_effect=ValueError('Saved CSV checkpoint configuration differs from registration: preprocessor')):
            choices, reasons, total = diagnose_saved_csv_models(Mock(), Mock(), 'dataset', model_root=Path.cwd())
        self.assertEqual((choices, reasons, total), ((), {'configuration_mismatch_preprocessor': 1}, 1))

    def test_unexpected_detail_is_not_reflected(self):
        with patch('Storage.CSVSavedModelInspection.list_csv_model_configurations', return_value=(("synthetic", "config"),)), patch(
            'Storage.CSVSavedModelInspection.inspect_saved_csv_model',
            side_effect=ValueError('Saved CSV checkpoint configuration differs from registration: private_value')):
            _, reasons, _ = diagnose_saved_csv_models(Mock(), Mock(), 'dataset', model_root=Path.cwd())
        self.assertEqual(reasons, {'configuration_mismatch': 1})

    def test_inspection_no_longer_compares_training_recipe(self):
        path = Path(__file__).resolve().parents[1] / 'src/Storage/CSVSavedModelInspection.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inspect_saved_csv_model')
        self.assertNotIn('registered_mode', ast.unparse(target))
        self.assertNotIn('persisted_mode', ast.unparse(target))

if __name__ == '__main__': unittest.main()
