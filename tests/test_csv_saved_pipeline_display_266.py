"""Revision 266: saved CSV widgets reflect checkpoint metadata only."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class PipelineValuesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        gui = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        names = ('_csv_saved_pipeline_values', '_csv_show_saved_pipeline')
        cls.methods = [n for n in gui.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
        cls.subject = SimpleNamespace()
        code = compile(ast.fix_missing_locations(ast.Module(body=cls.methods, type_ignores=[])), '<csv-display>', 'exec')
        namespace = {'staticmethod': staticmethod}
        exec(code, namespace)
        cls.pipeline_values = namespace['_csv_saved_pipeline_values']
        cls.show = namespace['_csv_show_saved_pipeline']

    def test_saved_values_come_from_checkpoint(self):
        config = Mock()
        config.get_algorithm_abbreviations.return_value = ['LRN']
        config.get_preprocessor_abbreviations.return_value = ['STA']
        config.get_feature_selection_abbreviations.return_value = ['NOR']
        config.get_oversampler_abbreviations.return_value = ['NOG']
        config.get_undersampler_abbreviations.return_value = ['NUG']
        config.mode.scoring.name = 'MCC'
        self.assertEqual(self.pipeline_values(config), {
            'algorithm_dropdown': ('LRN',), 'preprocess_dropdown': ('STA',),
            'reduction_dropdown': ('NOR',), 'oversampler_dropdown': ('NOG',),
            'undersampler_dropdown': ('NUG',), 'scoremetric_dropdown': 'MCC'})

    def test_only_pipeline_widget_fields_are_updated(self):
        config = Mock()
        for name in ('get_algorithm_abbreviations', 'get_preprocessor_abbreviations',
                     'get_feature_selection_abbreviations', 'get_oversampler_abbreviations',
                     'get_undersampler_abbreviations'):
            getattr(config, name).return_value = []
        self.assertEqual(set(self.pipeline_values(config).keys()), {
            'algorithm_dropdown', 'preprocess_dropdown', 'reduction_dropdown',
            'oversampler_dropdown', 'undersampler_dropdown', 'scoremetric_dropdown'})

    def test_display_is_only_called_after_checkpoint_inspection(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
        start = text.index('    def _csv_inspect_selected_model(self):')
        end = text.index('    def _csv_model_selected(self):', start)
        section = text[start:end]
        self.assertLess(section.index('filename = inspect_saved_csv_model('),
                        section.index('self._csv_show_saved_pipeline(filename)'))


if __name__ == '__main__':
    unittest.main()
