"""Dependency-free checks for the shared config source schema boundary."""
import ast
from pathlib import Path
import unittest


class SharedJSONSource199Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/Config.py').read_text(encoding='utf-8'))
        cls.source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Config')
        cls.methods = {n.name: ast.unparse(n) for n in cls.source.body if isinstance(n, ast.FunctionDef)}

    def test_legacy_sql_config_optional_source(self):
        text = self.methods['_validate_json_payload']
        self.assertIn("optional_top = {'data_source'}", text)
        self.assertIn('cls.DataSource(**section).validate()', text)
        self.assertIn("payload.get('data_source')", self.methods['load_config_from_json'])

    def test_one_serializer_and_clean_config_carries_source(self):
        self.assertIn("'data_source': self._json_section(self.data_source)", self.methods['_config_to_json'])
        self.assertIn('configuration.data_source = copy.deepcopy(self.data_source)', self.methods['get_clean_config'])
        self.assertIn('config.data_source = cls.DataSource(**source)', self.methods['load_config'])

    def test_csv_references_have_no_paths_or_credentials(self):
        source = next(n for n in self.source.body if isinstance(n, ast.ClassDef) and n.name == 'DataSource')
        fields = {n.target.id for n in source.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        self.assertEqual(fields, {'kind', 'dataset_id', 'artifact_id', 'display_name', 'delimiter', 'quotechar', 'encoding'})
        self.assertIn("self.kind not in ('sql', 'csv')", ast.unparse(source))


if __name__ == '__main__':
    unittest.main()
