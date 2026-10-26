"""Regression for internal data_source metadata breaking the welcome banner."""
import ast
from pathlib import Path
import unittest
from functools import partial
from types import SimpleNamespace


class LoggerConfigDisplayTests(unittest.TestCase):
    @staticmethod
    def get_method():
        source = Path(__file__).resolve().parents[1] / 'src' / 'JBGStreamedLogger.py'
        module = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'JBGLogger')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'print_config_settings')
        namespace = {
            'Config': object,
            'partial': partial,
            'html_wrapper': lambda tag, value: value,
            'pd': SimpleNamespace(DataFrame=SimpleNamespace(from_dict=lambda **kwargs: kwargs['data'])),
            'sensitive_log_data_enabled': lambda: False,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
        return namespace['print_config_settings']

    def test_source_metadata_does_not_crash_or_leak(self):
        method = self.get_method()
        events = []
        logger = SimpleNamespace(
            print_unformatted=lambda *args, **kwargs: events.append(('title', args[0])),
            display_matrix=lambda *args, **kwargs: events.append(('section', args[0])),
        )
        payload = {
            'title': 'Configuration settings',
            'connection': {'title': 'Database settings', 'host': 'localhost'},
            'data_source': {'kind': 'csv', 'dataset_id': 'private-dataset-id', 'artifact_id': 'private-artifact-id'},
            'mode': {'title': 'Classification mode settings', 'train': True},
            'io': {'title': 'I/O specifications'},
            'debug': {'title': 'Debug settings'},
        }
        config = SimpleNamespace(to_dict=lambda: payload)
        method(logger, config)
        self.assertEqual([value for kind, value in events if kind == 'section'], [
            '1. Database settings', '2. Classification mode settings',
            '3. I/O specifications', '4. Debug settings',
        ])
        self.assertNotIn('private-dataset-id', str(events))
        self.assertNotIn('private-artifact-id', str(events))

    def test_legacy_sql_payload_still_displays(self):
        method = self.get_method()
        events = []
        logger = SimpleNamespace(
            print_unformatted=lambda *args, **kwargs: None,
            display_matrix=lambda *args, **kwargs: events.append(args[0]),
        )
        config = SimpleNamespace(to_dict=lambda: {
            'title': 'Configuration settings',
            'connection': {'title': 'Database settings'},
            'mode': {'title': 'Classification mode settings'},
            'io': {'title': 'I/O specifications'},
            'debug': {'title': 'Debug settings'},
        })
        method(logger, config)
        self.assertEqual(len(events), 4)


if __name__ == '__main__':
    unittest.main()
