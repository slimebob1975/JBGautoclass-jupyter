"""Revision 162c: GUI metadata must use the original connection query."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'SQLDataLayer.py'


class CatalogMetadata162b(unittest.TestCase):
    def test_gui_metadata_uses_original_query(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        layer = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'DataLayer')
        methods = [x for x in layer.body if isinstance(x, ast.FunctionDef)
                   and x.name in ('get_id_columns', 'get_table_columns')]
        module = ast.Module(body=[ast.ClassDef(name='DataLayer', bases=[], keywords=[],
                                             body=methods, decorator_list=[])], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {'to_quoted_string': lambda s: "'" + s + "'"}
        exec(compile(module, str(SOURCE), 'exec'), namespace)
        layer = namespace['DataLayer']()
        calls = []
        layer.get_data_list_from_query = lambda q: (calls.append(q) or [('id', 'int'), ('class', 'nvarchar')])
        self.assertEqual(layer.get_id_columns('OtherDatabase', 'dbo.iris'),
                         {'id': 'int', 'class': 'nvarchar'})
        self.assertIn('FROM INFORMATION_SCHEMA.COLUMNS', calls[0])
        self.assertIn("TABLE_CATALOG = 'OtherDatabase'", calls[0])

    def test_bracket_escaped_catalog(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        layer = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'DataLayer')
        methods = [x for x in layer.body if isinstance(x, ast.FunctionDef)
                   and x.name in ('get_id_columns', 'get_table_columns')]
        module = ast.Module(body=[ast.ClassDef(name='DataLayer', bases=[], keywords=[],
                                             body=methods, decorator_list=[])], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {'to_quoted_string': lambda s: "'" + s.replace("'", "''") + "'"}
        exec(compile(module, str(SOURCE), 'exec'), namespace)
        layer = namespace['DataLayer']()
        calls = []
        layer.get_data_list_from_query = lambda q: (calls.append(q) or [])
        layer.get_id_columns('My]Db', 'dbo.iris')
        self.assertIn('FROM INFORMATION_SCHEMA.COLUMNS', calls[0])
