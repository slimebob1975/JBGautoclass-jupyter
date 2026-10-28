"""Revision 265c: exercise logger reporting without optional GUI dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class CSVPredictionLoggerTests(unittest.TestCase):
    def _subject(self, terminal=False):
        source = (Path(__file__).resolve().parents[1] / 'src/JBGStreamedLogger.py').read_text()
        module = ast.parse(source)
        method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
                      and n.name == 'print_prediction_results')
        displays = Mock()
        safe_sql = Mock(return_value='safe sql summary')
        namespace = {'print_html': displays, 'safe_prediction_query_log_message': safe_sql}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<logger-test>', 'exec'), namespace)
        logger = SimpleNamespace(_enable_quiet=False, in_terminal=terminal,
                                 log_file=SimpleNamespace(write_message=Mock()),
                                 writeln=Mock())
        return namespace['print_prediction_results'], logger, displays, safe_sql

    def test_csv_saved_result_has_no_query_and_no_ids_in_output(self):
        method, logger, displays, _ = self._subject()
        result = {'row_count': 15, 'artifact_id': 'private-artifact', 'run_id': 'private-run'}
        method(logger, result)
        message = displays.call_args.args[0]
        self.assertIn('15 CSV predictions', message)
        self.assertNotIn('private-artifact', message)
        self.assertNotIn('private-run', message)
        self.assertNotIn('SQL query', message)
        logger.log_file.write_message.assert_called_once_with(message)

    def test_csv_terminal_uses_safe_summary(self):
        method, logger, _, _ = self._subject(terminal=True)
        method(logger, {'row_count': 15, 'artifact_id': 'secret', 'run_id': 'secret'})
        logger.writeln.assert_called_once_with('unformatted', 'Saved 15 CSV predictions to the CSV artifact store.')

    def test_sql_query_reporting_unchanged(self):
        method, logger, displays, safe_sql = self._subject()
        method(logger, {'row_count': 2, 'query': 'SELECT data'})
        self.assertIn('SQL query:', displays.call_args.args[0])
        logger.log_file.write_message.assert_called_once_with('safe sql summary')
        safe_sql.assert_called_once_with(2, 'SELECT data')

    def test_invalid_csv_count_is_rejected(self):
        method, logger, _, _ = self._subject()
        with self.assertRaises(ValueError):
            method(logger, {'row_count': -1, 'artifact_id': 'a', 'run_id': 'r'})


if __name__ == '__main__':
    unittest.main()
