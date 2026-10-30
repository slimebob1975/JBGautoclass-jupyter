"""Source-safe Dataset columns binding tests with synthetic column names only."""
import ast
from pathlib import Path
import tempfile
import unittest

import sys
import types
from unittest.mock import patch

try:
    import ipywidgets as widget_library
except ImportError:
    class StubWidget:
        def __init__(self, value="", description="", children=(), **kwargs):
            self.value = value
            self.description = description
            self.accept = kwargs.get("accept", "")
            self.multiple = kwargs.get("multiple", False)
            self.children = tuple(children)
            self._observers = []
        def on_click(self, callback):
            self.callback = callback
        def observe(self, callback, names=None):
            self._observers.append(callback)
    widget_library = types.ModuleType("ipywidgets")
    for name in ("Text", "HTML", "Button", "VBox", "HBox", "ToggleButtons", "FileUpload", "Layout", "Dropdown"):
        setattr(widget_library, name, StubWidget)
    with patch.dict(sys.modules, {"ipywidgets": widget_library}):
        from GUI.CSVSourcePanel import CSVSourcePanel, DataSourceSelector
else:
    from GUI.CSVSourcePanel import CSVSourcePanel, DataSourceSelector
from CSVColumnSchema import inspect_csv_columns

ROOT = Path(__file__).resolve().parents[1]


class CSVColumnBindingTests(unittest.TestCase):
    def test_validation_notifies_only_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'example.csv'
            path.write_text('uid,target,x\n001,A,one\n002,B,two\n', encoding='utf-8')
            panel = CSVSourcePanel()
            received = []
            panel.on_validated = lambda value: received.append(value)
            panel.path.value = str(path)
            self.assertTrue(panel.validate())
            self.assertEqual(received, [panel])
            panel.path.value = str(path / 'invalid')
            self.assertFalse(panel.validate())
            self.assertEqual(received, [panel])

    def test_source_switch_invokes_callback(self):
        selector = DataSourceSelector(widget_library.HTML(), widget_library.HTML())
        received = []
        selector.on_source_change = received.append
        selector.choice.value = 'csv'
        selector._source_changed({'name': 'value', 'new': 'csv'})
        selector.choice.value = 'sql'
        selector._source_changed({'name': 'value', 'new': 'sql'})
        self.assertIn('csv', received)
        self.assertIn('sql', received)

    def test_csv_roles_can_be_edited_only_after_validation(self):
        import ast
        tree = ast.parse((ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        widgets_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        methods = {n.name: n for n in widgets_class.body if isinstance(n, ast.FunctionDef)}
        switch = ast.unparse(methods['_csv_source_selected'])
        validated = ast.unparse(methods['_csv_validated_columns'])
        self.assertIn('_csv_set_role_editability(False)', switch)
        self.assertIn('_csv_model_selected()', validated)
        self.assertIn('_sql_role_disabled_188d', switch)
        event_handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EventHandler')
        data_method = next(n for n in event_handler.body if isinstance(n, ast.FunctionDef) and n.name == 'data_columns')
        self.assertIn('_csv_column_role_changed(previous_features=change.old)', ast.unparse(data_method))

    def test_source_change_resets_model_without_firing_model_observer(self):
        tree = ast.parse((ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        switch = ast.unparse(methods['_csv_source_selected'])
        self.assertIn('self._reset_model_on_source_change()', switch)

        class Model:
            options = ('', 'New model', 'Previously saved')
            value = 'Previously saved'
        class Event:
            lock_observe_1 = False
        class Fake:
            models_dropdown = Model()
            eventhandler = Event()
            resets = []
            def reset_field_status(self, name):
                self.resets.append(name)

        namespace = {}
        definition = ast.Module(body=[methods['_reset_model_on_source_change']], type_ignores=[])
        exec(compile(ast.fix_missing_locations(definition), '<test>', 'exec'), namespace)
        fake = Fake()
        namespace['_reset_model_on_source_change'](fake)
        self.assertEqual(fake.models_dropdown.value, '')
        self.assertFalse(fake.eventhandler.lock_observe_1)
        self.assertEqual(fake.resets, ['models'])

    def test_csv_columns_include_text_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.csv'
            path.write_text('uid,target,x\n001,A,one\n002,B,two\n', encoding='utf-8')
            schema = inspect_csv_columns(path)
            self.assertIn('uid', schema.unique_id_candidates)
            self.assertIn('target', schema.class_candidates)

    def test_csv_view_is_framed_like_sql_connection(self):
        tree = ast.parse((ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'connection_form')
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        section_calls = [node for node in calls if isinstance(node.func, ast.Attribute)
                         and node.func.attr == 'create_section_form']
        titles = {node.args[0].value: node for node in section_calls
                  if node.args and isinstance(node.args[0], ast.Constant)}
        self.assertIn('Database connection', titles)
        self.assertIn('CSV file', titles)
        self.assertEqual(ast.unparse(titles['CSV file'].args[2]), 'widgets.Box')
        selector = next(node for node in calls if isinstance(node.func, ast.Name)
                        and node.func.id == 'DataSourceSelector')
        self.assertEqual(ast.unparse(selector.args[1]), 'csv_form')

    def test_execution_entrypoints_have_csv_guard(self):
        tree = ast.parse((ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        for cls, methods in {
            'EventHandler': ('continuation_button_was_clicked', 'regression_suite_button_was_clicked',
                             'repeat_last_run_button_was_clicked', 'start_button_was_clicked'),
            'Widgets': ('start_button_actions', 'repeat_last_run_button_actions',
                        '_launch_feature_selection_investigation',),
        }.items():
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            for method in methods:
                func = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
                prefix = ast.dump(ast.Module(body=func.body[:20], type_ignores=[]))
                self.assertIn('_csv_view_active', prefix, f'{cls}.{method}')


if __name__ == '__main__':
    unittest.main()

class CSVModelWorkflow188fTests(unittest.TestCase):
    def test_csv_model_gate_avoids_sql_observer(self):
        tree = ast.parse((ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EventHandler')
        event = next(n for n in handler.body if isinstance(n, ast.FunctionDef) and n.name == 'models_dropdown')
        code = ast.unparse(event)
        self.assertLess(code.index('_csv_model_selected()'), code.index('update_class_id_data_columns()'))
        widgets = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        methods = {n.name: ast.unparse(n) for n in widgets.body if isinstance(n, ast.FunctionDef)}
        self.assertIn('Config.DEFAULT_TRAIN_OPTION', methods['_csv_model_selected'])
        self.assertIn('self.models_dropdown.disabled = False', methods['_csv_validated_columns'])
        self.assertNotIn('_csv_set_role_editability(True)', methods['_csv_validated_columns'])
        self.assertIn('_sql_model_options_188f', methods['_csv_source_selected'])

    def test_model_gate_only_unlocks_new_model(self):
        tree = ast.parse((ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Widgets')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_csv_model_selected')
        namespace = {'Config': type('Config', (), {'DEFAULT_TRAIN_OPTION': 'New model'})}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<test>', 'exec'), namespace)
        class Fake:
            _csv_roles_loaded_188 = True
            models_dropdown = types.SimpleNamespace(value='')
            id_column_diagnostic = types.SimpleNamespace(value='')
            results = []
            dark_numbers_checkbox = types.SimpleNamespace(value=False)
            def _csv_view_active(self): return False
            def sync_dark_number_control_state(self): pass
            def sync_feature_selection_controls(self): pass
            def _csv_update_erase_model_button(self): pass
            def _csv_set_training_controls(self, enabled): pass
            def _csv_set_model_status(self, message): pass
            def _csv_refresh_training_action(self): pass
            def _csv_set_role_editability(self, enabled):
                self.results.append(enabled)
            def update_ready(self): pass
            def disable_button(self, name): pass
        fake = Fake()
        namespace['_csv_model_selected'](fake)
        self.assertEqual(fake.results[-1], False)
        fake.models_dropdown.value = 'New model'
        namespace['_csv_model_selected'](fake)
        self.assertEqual(fake.results[-1], True)
