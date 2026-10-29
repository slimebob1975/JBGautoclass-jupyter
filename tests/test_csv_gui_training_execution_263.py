"""Revision 263: dependency-light GUI dispatch and dedicated CSV execution boundary."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
WIDGETS_SOURCE = ROOT / 'src' / 'GUI' / 'Widgets.py'
HANDLER_SOURCE = ROOT / 'src' / 'GUIHandler.py'


def load_method(source, name, namespace=None):
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    cls = ast.ClassDef(name='Target', bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    values = dict(namespace or {})
    exec(compile(module, str(source), 'exec'), values)
    return getattr(values['Target'], name)


class Button:
    def __init__(self):
        self.disabled = False
        self.button_style = ''
        self.tooltip = ''
        self.description = 'Start'
        self.style = SimpleNamespace(button_color=None)
        self.classes = set()

    def remove_class(self, value):
        self.classes.discard(value)

    def add_class(self, value):
        self.classes.add(value)


class CSVGUITrainingExecution263(unittest.TestCase):
    def make_start_target(self):
        method = load_method(WIDGETS_SOURCE, '_csv_start_training')
        target = SimpleNamespace()
        target._csv_start_training = method.__get__(target, type(target))
        target._csv_training_readiness = Mock(return_value=('ready', 'ready'))
        target._csv_refresh_training_action = Mock(return_value=('ready', 'ready'))
        target._csv_source_panel = SimpleNamespace(status=SimpleNamespace(value=''))
        target.output = Mock()
        target.clear_mispredicted_output = Mock()
        target.start_button = Button()
        target.predict_checkbox = SimpleNamespace(value=False)
        target._csv_prepare_training_runtime = Mock(return_value=(
            SimpleNamespace(), SimpleNamespace(configuration_id='synthetic-config'),
            object(), object()))
        target.guihandler = SimpleNamespace(run_csv_classifier=Mock(
            return_value={'mispredicted': None}))
        return target

    def test_csv_start_prepares_runtime_and_uses_dedicated_handler(self):
        target = self.make_start_target()
        result = target._csv_start_training()
        self.assertEqual(result, {'mispredicted': None})
        target._csv_prepare_training_runtime.assert_called_once_with()
        target.guihandler.run_csv_classifier.assert_called_once()
        args = target.guihandler.run_csv_classifier.call_args.args
        self.assertEqual(args[3], 'synthetic-config')
        self.assertIs(args[4], target.output)
        self.assertIn('model checkpoint was saved', target._csv_source_panel.status.value)
        target._csv_refresh_training_action.assert_called_once_with()

    def test_csv_start_failure_is_visible_and_retryable(self):
        target = self.make_start_target()
        target.guihandler.run_csv_classifier.side_effect = RuntimeError('synthetic failure')
        self.assertFalse(target._csv_start_training())
        self.assertIn('RuntimeError: synthetic failure', target._csv_source_panel.status.value)
        self.assertFalse(target.start_button.disabled)
        self.assertEqual(target.start_button.button_style, 'warning')
        self.assertIn('jbg-action-warning', target.start_button.classes)

    def test_csv_start_refuses_programmatic_click_when_not_ready(self):
        target = self.make_start_target()
        target._csv_training_readiness.return_value = ('waiting', 'register first')
        self.assertFalse(target._csv_start_training())
        target._csv_prepare_training_runtime.assert_not_called()
        target.guihandler.run_csv_classifier.assert_not_called()
        self.assertEqual(target._csv_source_panel.status.value, 'register first')

    def test_source_specific_start_dispatch_preserves_sql_path(self):
        widgets = SimpleNamespace(Button=object)
        from Config import Config
        method = load_method(WIDGETS_SOURCE, 'start_button_was_clicked', {'widgets': widgets, 'Config': Config})
        target = SimpleNamespace(widgets=Mock())
        target.start_button_was_clicked = method.__get__(target, type(target))

        target.widgets._csv_view_active.return_value = True
        target.widgets.models_dropdown.value = Config.DEFAULT_TRAIN_OPTION
        target.start_button_was_clicked(Mock())
        target.widgets._csv_start_training.assert_called_once_with()
        target.widgets.start_button_actions.assert_not_called()

        target.widgets.reset_mock()
        target.widgets._csv_view_active.return_value = False
        target.start_button_was_clicked(Mock())
        target.widgets.deactivate_section.assert_called_once_with('classifier')
        target.widgets.start_button_actions.assert_called_once_with()
        target.widgets._csv_start_training.assert_not_called()

    def test_dedicated_handler_builds_classifier_without_sql_setup(self):
        method = load_method(HANDLER_SOURCE, 'run_csv_classifier', {
            'Output': object, 'time': __import__('time'), 'traceback': __import__('traceback')})
        handler = SimpleNamespace()
        handler.run_csv_classifier = method.__get__(handler, type(handler))
        handler._classifier_diagnostic = Mock()
        handler.logger = SimpleNamespace(
            set_enable_quiet=Mock(), capture_console_output=Mock(return_value=nullcontext()),
            print_warning=Mock(), print_error=Mock())
        runtime = SimpleNamespace(data_source=SimpleNamespace(kind='csv'),
                                  io=SimpleNamespace(verbose=True),
                                  should_predict=lambda: False)
        classifier = SimpleNamespace(run=Mock(return_value={'mispredicted': None}))
        prepare = Mock(return_value=classifier)
        fake_module = ModuleType('Storage.CSVTrainingApplication')
        fake_module.prepare_registered_csv_training_classifier = prepare
        repository, scope, output = object(), object(), nullcontext()
        with patch.dict(sys.modules, {'Storage.CSVTrainingApplication': fake_module}):
            result = handler.run_csv_classifier(runtime, repository, scope, 'config', output)
        self.assertEqual(result, {'mispredicted': None})
        prepare.assert_called_once_with(runtime, repository, scope, 'config', handler.logger)
        classifier.run.assert_called_once_with()
        handler.logger.set_enable_quiet.assert_called_once_with(False)

    def test_dedicated_handler_rejects_non_csv_before_classifier_preparation(self):
        method = load_method(HANDLER_SOURCE, 'run_csv_classifier', {
            'Output': object, 'time': __import__('time'), 'traceback': __import__('traceback')})
        handler = SimpleNamespace()
        handler.run_csv_classifier = method.__get__(handler, type(handler))
        handler._classifier_diagnostic = Mock()
        handler.logger = SimpleNamespace(
            set_enable_quiet=Mock(), capture_console_output=Mock(return_value=nullcontext()),
            print_warning=Mock(), print_error=Mock())
        runtime = SimpleNamespace(data_source=SimpleNamespace(kind='sql'),
                                  io=SimpleNamespace(verbose=True))
        with self.assertRaisesRegex(ValueError, 'CSV runtime configuration'):
            handler.run_csv_classifier(runtime, object(), object(), 'config', nullcontext())
        handler.logger.set_enable_quiet.assert_not_called()


try:
    from test_csv_native_handler_bridge_258 import AutomaticClassifier, CSVNativeHandlerBridge258
except (ImportError, OSError):
    AutomaticClassifier = None
    CSVNativeHandlerBridge258 = unittest.TestCase


@unittest.skipIf(AutomaticClassifier is None, 'Full JBG ML dependencies unavailable')
class CSVGUITrainingIntegration263(CSVNativeHandlerBridge258):
    """Windows/ML acceptance probe for the new GUIHandler CSV execution bridge."""
    test_native_dataset_handler_and_real_estimator_roundtrip = None
    test_registered_train_only_reader_does_not_include_unlabelled_rows = None

    def test_gui_handler_runs_real_train_only_csv_pipeline_and_saves_model(self):
        from contextlib import ExitStack
        from GUIHandler import GUIHandler
        from Helpers import build_model_performance_matrix, save_matrix_as_csv
        from JBGMeta import (
            Algorithm, AlgorithmTuple, Oversampling, Undersampling,
            Preprocess, PreprocessTuple, Reduction, ReductionTuple,
        )
        from JBGModelPersistence import load_model_artifact_with_compatibility

        config = self.config
        config.script_path = self.root
        config.io.model_path = 'models'
        config.mode.algorithm = AlgorithmTuple([Algorithm.LRN])
        config.mode.preprocessor = PreprocessTuple([Preprocess.STA])
        config.mode.feature_selection = ReductionTuple([Reduction.NOR])
        config.mode.oversampler = Oversampling.NOG
        config.mode.undersampler = Undersampling.NUG
        config.mode.oversamplers = ['NOG']
        config.mode.undersamplers = ['NUG']
        config.mode.max_iterations = 300
        config.mode.test_size = .2
        config.mode.calculate_dark_numbers = False
        config.mode.calculate_feature_importance = False
        (self.root / 'models').mkdir()
        (self.root / 'output' / 'csvs').mkdir(parents=True)

        def export_cv_results(results, filepath):
            save_matrix_as_csv(build_model_performance_matrix(results), filepath)

        self.logger.capture_console_output.return_value = nullcontext()
        self.logger.print_test_performance.side_effect = export_cv_results
        self.logger.get_log_output.return_value = ''
        self.logger.get_log_output_html.return_value = ''
        handler = object.__new__(GUIHandler)
        handler.logger = self.logger
        model_path = Path(config.get_model_filename())

        with ExitStack() as guards:
            guards.enter_context(patch('socket.create_connection',
                                       side_effect=AssertionError('Network access forbidden')))
            import pyodbc
            guards.enter_context(patch.object(
                pyodbc, 'connect', side_effect=AssertionError('SQL connection forbidden')))
            from SQLDataLayer import DataLayer as SQLDataLayer
            guards.enter_context(patch.object(
                SQLDataLayer, '__init__', side_effect=AssertionError('SQL layer forbidden')))
            guards.enter_context(patch('JBGTaskRunner.send_email', return_value=False))
            result = handler.run_csv_classifier(
                config, self.repo, self.scope, self.roles.configuration_id, nullcontext())

        self.assertIsInstance(result, dict)
        self.assertTrue(model_path.is_file())
        _, compatibility = load_model_artifact_with_compatibility(model_path)
        self.assertEqual(compatibility['source_kind'], 'csv')
        self.assertEqual(compatibility['schema_version'], 2)



if __name__ == '__main__':
    unittest.main()
