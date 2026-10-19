import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from Config import Config
from JBGExceptions import ConfigException


class Config155Tests(unittest.TestCase):
    def test_generated_configuration_to_json_never_executes(self):
        source = """name = 'demo'\nconnection = {'sql_password': 'secret'}\nmode = {}\nio = {}\ndebug = {}\n"""
        payload = Config._legacy_config_to_json(source)
        self.assertEqual(payload['schema_version'], 1)
        self.assertEqual(payload['connection']['sql_password'], '')

    def test_bad_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'config.json'
            file.write_text(json.dumps({'schema_version': 22}), encoding='utf-8')
            with self.assertRaises(ConfigException):
                Config.load_config_from_json(file)

    def test_json_dispatch_and_password_sanitization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'schema_version': 1, 'name': 'demo',
                'connection': {'sql_password': 'must-not-trust'},
                'mode': {}, 'io': {}, 'debug': {}}), encoding='utf-8')
            with patch.object(Config, 'load_config', return_value='loaded') as load:
                self.assertEqual(Config.load_config_from_module(['main', '-f', str(path)]), 'loaded')
                self.assertEqual(load.call_args.args[0].connection['sql_password'], '')

if __name__ == '__main__':
    unittest.main()
