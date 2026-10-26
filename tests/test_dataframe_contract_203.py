"""Revision 203: provider-neutral DataFrame contract without training changes."""
import unittest

from DataLayer.FrameContract import dataframe_from_rows, load_dataframe


class FakeConfig:
    def get_column_names(self):
        return ['text', 'numeric', 'uid', 'target']


class FakeSource:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get_dataset(self, num_rows=None):
        self.calls.append(num_rows)
        return self.rows[:num_rows] if num_rows is not None else self.rows


class FrameContractTests(unittest.TestCase):
    def test_column_order_and_id_values(self):
        df = load_dataframe(FakeSource([('x', '12', '001', 'A')]), FakeConfig())
        self.assertEqual(list(df.columns), ['text', 'numeric', 'uid', 'target'])
        self.assertEqual(df.iloc[0]['uid'], '001')
        self.assertEqual(df.iloc[0]['numeric'], '12')

    def test_empty_dataset_retains_columns(self):
        df = load_dataframe(FakeSource([]), FakeConfig())
        self.assertEqual(df.shape, (0, 4))
        self.assertEqual(list(df.columns), FakeConfig().get_column_names())

    def test_limit_delegated_not_implemented_twice(self):
        source = FakeSource([('x', 1, '01', 'A'), ('y', 2, '02', '')])
        df = load_dataframe(source, FakeConfig(), 1)
        self.assertEqual(source.calls, [1])
        self.assertEqual(len(df), 1)

    def test_malformed_rows_rejected(self):
        with self.assertRaises(ValueError):
            dataframe_from_rows([('x', 'y')], ['only'])
        with self.assertRaises(ValueError):
            dataframe_from_rows([], ['id', 'id'])

    def test_sql_and_csv_method_delegate_to_same_contract(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / 'src'
        sql = (root / 'SQLDataLayer.py').read_text(encoding='utf8')
        csv = (root / 'Storage' / 'CSVTrainingAdapter.py').read_text(encoding='utf8')
        for code in (sql, csv):
            self.assertIn('def load_dataframe(self, num_rows=None):', code)
            self.assertIn('return load_dataframe(self, self.config, num_rows)', code)


if __name__ == '__main__':
    unittest.main()
