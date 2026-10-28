"""Revision 260: GUI training preflight remains fail-closed until source binding exists.

No private datasets or SQL connections are used by these tests.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'


def load_methods():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    methods = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
            '_csv_training_readiness', '_csv_refresh_training_action'):
            methods[node.name] = node
    assert len(methods) == 2
    cls = ast.ClassDef(name='Target', bases=[], keywords=[], body=list(methods.values()), decorator_list=[])
    mod = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {'Config': SimpleNamespace(DEFAULT_TRAIN_OPTION='New model')}
    exec(compile(mod, str(SOURCE), 'exec'), namespace)
    return namespace['Target']


class CSVGUITrainingReadiness260(unittest.TestCase):
    def make_widget(self):
        instance = load_methods()()
        instance._csv_view_active = lambda: True
        instance._csv_roles_loaded_188 = True
        instance.models_dropdown = SimpleNamespace(value='New model')
        instance._csv_current_dataset_artifact = lambda: 'synthetic-artifact'
        instance.id_column = SimpleNamespace(value='row_id')
        instance.class_column = SimpleNamespace(value='label')
        instance.data_columns = SimpleNamespace(value=('measurement',))
        instance.start_button = SimpleNamespace(tooltip='', disabled=False, button_style='')
        return instance

    def test_validated_registered_csv_still_requires_runtime_binding(self):
        w = self.make_widget()
        state, explanation = w._csv_training_readiness()
        self.assertEqual(state, 'blocked')
        self.assertIn('runtime binding', explanation)

    def test_unregistered_and_incomplete_csv_are_not_ready(self):
        w = self.make_widget()
        w._csv_current_dataset_artifact = lambda: None
        self.assertEqual(w._csv_training_readiness()[0], 'waiting')
        w._csv_current_dataset_artifact = lambda: 'synthetic-artifact'
        w.data_columns.value = ()
        self.assertEqual(w._csv_training_readiness()[0], 'waiting')
        w.data_columns.value = ('measurement',)
        w.models_dropdown.value = ''
        self.assertEqual(w._csv_training_readiness()[0], 'waiting')

    def test_source_switch_and_validation_requirements(self):
        w = self.make_widget()
        w._csv_view_active = lambda: False
        self.assertEqual(w._csv_training_readiness()[0], 'blocked')
        w._csv_view_active = lambda: True
        w._csv_roles_loaded_188 = False
        self.assertEqual(w._csv_training_readiness()[0], 'waiting')

    def test_start_remains_disabled_by_semantic_action_state(self):
        w = self.make_widget()
        # Run the real lightweight button-state implementation with a fake widget.
        from GUI.ButtonStates import set_action_state
        class Button:
            disabled = False
            button_style = ''
            tooltip = ''
            style = SimpleNamespace(button_color=None)
            def remove_class(self, value): pass
            def add_class(self, value): pass
        w.start_button = Button()
        with patch('GUI.ButtonStates.set_action_state', wraps=set_action_state):
            state, explanation = w._csv_refresh_training_action()
        self.assertEqual(state, 'blocked')
        self.assertTrue(w.start_button.disabled)
        self.assertEqual(w.start_button.tooltip, explanation)
        self.assertEqual(w.start_button.button_style, 'danger')


if __name__ == '__main__':
    unittest.main()
