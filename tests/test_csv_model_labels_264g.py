"""Revision 264g: saved CSV labels describe checkpoint choices, not opaque IDs."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from Storage.CSVSavedModelInspection import saved_csv_model_label


class SavedCSVLabels264g(unittest.TestCase):
    def label(self, mode):
        config = SimpleNamespace(_config_to_json=lambda: {'mode': mode})
        with patch('JBGModelPersistence.load_model_artifact_with_compatibility',
                   return_value=((config, None), {})):
            return saved_csv_model_label('verified-local-model', '12345678-0000-0000-0000-000000000000')

    def test_labels_show_training_choices_and_distinguish_models(self):
        first = self.label({'algorithm': ['LRN'], 'preprocessor': 'STA',
                            'feature_selection': 'NOR', 'oversampler': 'NOS',
                            'undersampler': 'NUS'})
        second = self.label({'algorithm': 'RFC', 'preprocessor': ['NOS'],
                             'feature_selection': 'RFE', 'oversampler': 'NOS',
                             'undersampler': 'NUS'})
        self.assertNotEqual(first, second)
        self.assertIn('LRN | STA', first)
        self.assertIn('RFC | NOS', second)
        self.assertIn('[12345678]', first)

    def test_names_and_column_data_are_never_displayed(self):
        label = self.label({'algorithm': '../../secret', 'preprocessor': 'STA',
                            'feature_selection': 'NOR'})
        self.assertIn('CSV ? | STA', label)
        self.assertNotIn('secret', label)

    def test_missing_options_use_placeholder(self):
        self.assertIn('CSV ? | ?', self.label({}))


if __name__ == '__main__':
    unittest.main()
