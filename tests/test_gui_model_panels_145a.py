"""GUI-only regression checks for revision 145a's collapsible panels."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'


class ModelPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding='utf-8')
        ast.parse(cls.source)

    def test_selection_observer_and_dynamic_children(self):
        self.assertIn('self.algorithm_dropdown.observe(self._model_selection_changed, names="value")', self.source)
        self.assertIn('for code in selected if code in self._model_parameter_panels', self.source)
        self.assertIn('tabs.children = tuple(content for _, content in visible)', self.source)

    def test_keep_controls_when_deselected(self):
        self.assertIn('self._model_parameter_panels[algorithm] = content', self.source)
        self.assertIn('for algorithm, controls in self._model_parameter_controls.items()', self.source)

    def test_feature_groups_are_accordions(self):
        self.assertIn('accordion = widgets.Accordion(children=[content], selected_index=None,', self.source)
        for heading in ('Search strategy', 'Validation and computational budget',
                        'Investigative model parameters (only the selected family is active)'):
            self.assertIn(f'setting_group("{heading}"', self.source)


if __name__ == '__main__':
    unittest.main()
