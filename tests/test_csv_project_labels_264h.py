"""Revision 264h: local project prefix on verified CSV checkpoint labels."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from Storage.CSVSavedModelInspection import saved_csv_model_label


class CSVProjectLabels264h(unittest.TestCase):
    def make_label(self, project):
        config = SimpleNamespace(_config_to_json=lambda: {'mode': {
            'algorithm': ['LRN'], 'preprocessor': 'STA',
            'feature_selection': 'NOR', 'oversampler': 'NOS', 'undersampler': 'NUS'}})
        with patch('JBGModelPersistence.load_model_artifact_with_compatibility',
                   return_value=((config, None), {})):
            return saved_csv_model_label('local-verified-checkpoint',
                                         '12345678-0000-0000-0000-000000000000', project)

    def test_project_prefix_and_model_identity_are_visible(self):
        label = self.make_label('Iris project')
        self.assertTrue(label.startswith('Iris project | CSV LRN | STA'))
        self.assertIn('[12345678]', label)

    def test_control_characters_and_separator_are_not_inserted(self):
        label = self.make_label('Sample\nprivate|project')
        self.assertTrue(label.startswith('Sampleprivateproject | CSV'))
        self.assertNotIn('\n', label)

    def test_empty_and_long_names_are_handled(self):
        self.assertTrue(self.make_label(' ').startswith('Project | CSV'))
        self.assertLessEqual(len(self.make_label('x'*200).split(' | CSV')[0]), 48)

if __name__ == '__main__':
    unittest.main()
