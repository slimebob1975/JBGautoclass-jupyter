"""Regression checks for CSV GUI configuration lifecycle consolidation."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def method_names(path, class_name):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


class CSVLifecycleCleanupTests(unittest.TestCase):
    def test_csv_panel_uses_dataset_only_actions(self):
        source = (ROOT / 'src/GUI/CSVSourcePanel.py').read_text(encoding='utf-8')
        for obsolete in ('save_configuration_button', 'saved_configurations',
                         'refresh_configurations_button', 'load_configuration_button'):
            self.assertNotIn(obsolete, source)
        for needed in ('register_button', 'refresh_saved_button', 'load_saved_button', 'erase_saved_button'):
            self.assertIn(needed, source)

    def test_csv_gui_no_independent_model_configuration_lifecycle(self):
        source = (ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
        for obsolete in ('_csv_save_configuration', '_csv_load_configuration',
                         '_csv_refresh_configurations', '_csv_loaded_shared_json_200',
                         'save_configuration_button'):
            self.assertNotIn(obsolete, source)
        self.assertIn('register_csv_dataset(', source)
        self.assertIn('restore_csv_dataset(', source)

    def test_existing_sql_model_config_load_untouched(self):
        funcs = method_names('src/GUI/Widgets.py', 'Widgets')
        load = ast.get_source_segment((ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8'), funcs['load_model_config'])
        self.assertIn('Config.load_config_from_model_file', load)
        artifact = (ROOT / 'src/JBGModelPersistence.py').read_text(encoding='utf-8')
        self.assertIn('def load_model_config(', artifact)
        self.assertIn('def save_model_artifact(', artifact)

    def test_legacy_readers_kept_for_transition(self):
        source = (ROOT / 'src/Storage/CSVModelConfiguration.py').read_text(encoding='utf-8')
        self.assertIn('def load_csv_model_configuration(', source)
        self.assertIn('def list_csv_model_configurations(', source)


if __name__ == '__main__':
    unittest.main()
