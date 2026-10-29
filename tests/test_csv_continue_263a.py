"""CSV Continue gate regression; synthetic state and no SQL connections."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'


def method_owner(method_names):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in method_names]
    assert len(found) == len(method_names)
    cls = ast.ClassDef(name='Subject', bases=[], keywords=[], body=found, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    ns = {'Config': SimpleNamespace(DEFAULT_TRAIN_OPTION='New model'), 'widgets': SimpleNamespace(Button=object)}
    exec(compile(module, str(SOURCE), 'exec'), ns)
    return ns['Subject']


class CSVContinue263a(unittest.TestCase):
    def widget(self):
        w = method_owner(('_csv_continue_ready', 'update_ready'))()
        w._csv_view_active = lambda: True
        w._csv_roles_loaded_188 = True
        w.models_dropdown = SimpleNamespace(value='New model')
        w._csv_current_dataset_artifact = lambda: 'synthetic-dataset'
        w.id_column = SimpleNamespace(value='synthetic_id')
        w.class_column = SimpleNamespace(value='target')
        w.data_columns = SimpleNamespace(value=('feature_1',))
        w.num_variables = SimpleNamespace(value=0)
        w.states = {}
        w.enable_button = lambda name: w.states.__setitem__(name, False)
        w.disable_button = lambda name: w.states.__setitem__(name, True)
        return w

    def test_registered_csv_enables_continue_without_sql(self):
        w = self.widget()
        w.update_ready()
        self.assertFalse(w.states['continuation_button'])
        self.assertTrue(w.states['test_profile_button'])
        self.assertEqual(w.num_variables.value, 1)

    def test_unregistered_csv_remains_blocked(self):
        w = self.widget()
        w._csv_current_dataset_artifact = lambda: None
        w.update_ready()
        self.assertTrue(w.states['continuation_button'])

    def test_csv_requires_valid_roles_and_new_model(self):
        w = self.widget()
        w.models_dropdown.value = ''
        self.assertFalse(w._csv_continue_ready())
        w.models_dropdown.value = 'New model'
        w.data_columns.value = ()
        self.assertFalse(w._csv_continue_ready())

    def test_verified_saved_model_enables_continue(self):
        w = self.widget()
        w.models_dropdown.value = 'saved-synthetic-model'
        w._csv_verified_model_selection = 'saved-synthetic-model'
        w.update_ready()
        self.assertFalse(w.states['continuation_button'])
        w._csv_verified_model_selection = None
        w.update_ready()
        self.assertTrue(w.states['continuation_button'])

    def test_saved_model_requires_registered_dataset(self):
        w = self.widget()
        w.models_dropdown.value = 'saved-synthetic-model'
        w._csv_verified_model_selection = 'saved-synthetic-model'
        w._csv_current_dataset_artifact = lambda: None
        self.assertFalse(w._csv_continue_ready())

    def test_continue_dispatch_does_not_use_sql_actions(self):
        e = method_owner(('continuation_button_was_clicked',))()
        w = self.widget()
        w.continuation_button = SimpleNamespace(disabled=False)
        calls = []
        w._csv_continue_actions = lambda: calls.append('csv')
        w.continuation_button_actions = lambda: self.fail('SQL continuation invoked')
        w.deactivate_section = lambda x: self.fail('SQL section invoked')
        e.widgets = w
        e.continuation_button_was_clicked(None)
        self.assertEqual(calls, ['csv'])

    def test_programmatic_click_when_unregistered_is_blocked(self):
        e = method_owner(('continuation_button_was_clicked',))()
        w = self.widget()
        w._csv_current_dataset_artifact = lambda: None
        w.continuation_button = SimpleNamespace(disabled=False)
        w._csv_continue_actions = lambda: self.fail('CSV settings opened without registration')
        e.widgets = w
        e.continuation_button_was_clicked(None)
        self.assertTrue(w.states['continuation_button'])


if __name__ == '__main__':
    unittest.main()
