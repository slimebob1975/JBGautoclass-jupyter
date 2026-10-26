"""Revision 204: source-semantic diagnostics must be safe and observational."""
from decimal import Decimal
import unittest

import pandas as pd

from DataLayer.FrameContract import dataframe_from_rows
from DataLayer.FrameDiagnostics import compare_frame_profiles, profile_dataframe


class FrameDiagnosticsTests(unittest.TestCase):
    def test_sql_null_vs_csv_empty_preserved(self):
        names = ['uid', 'target', 'feature']
        sql = dataframe_from_rows([('001', None, Decimal('12.5'))], names)
        csv = dataframe_from_rows([('001', '', '12.5')], names)
        sql_before = sql.copy(deep=True)
        csv_before = csv.copy(deep=True)
        result = compare_frame_profiles(sql, csv)
        self.assertFalse(result['null_cell_count_equal'])
        self.assertFalse(result['empty_string_count_equal'])
        pd.testing.assert_frame_equal(sql, sql_before)
        pd.testing.assert_frame_equal(csv, csv_before)
        self.assertEqual(csv.loc[0, 'uid'], '001')
        self.assertIsNone(sql.loc[0, 'target'])

    def test_profile_aggregates_only_no_values_or_column_names(self):
        frame = dataframe_from_rows([('sensitive-id-999', 'secret-field')], ['private-id', 'private-label'])
        profile = profile_dataframe(frame)
        output = repr(profile)
        for sensitive in ('sensitive-id-999', 'secret-field', 'private-id', 'private-label'):
            self.assertNotIn(sensitive, output)
        self.assertEqual(profile.rows, 1)
        self.assertEqual(profile.columns, 2)
        self.assertGreater(profile.memory_bytes, 0)

    def test_missing_vs_empty_vs_zero(self):
        frame = dataframe_from_rows([(None, '', 0), (float('nan'), ' ', False)], ['a', 'b', 'c'])
        profile = profile_dataframe(frame)
        self.assertEqual(profile.null_cells, 2)
        self.assertEqual(profile.empty_string_cells, 1)

    def test_empty_frame_profile(self):
        frame = dataframe_from_rows([], ['uid', 'target'])
        profile = profile_dataframe(frame)
        self.assertEqual((profile.rows, profile.columns, profile.null_cells), (0, 2, 0))
        self.assertGreaterEqual(profile.memory_bytes, 0)

    def test_shape_and_dtype_comparison_no_column_disclosure(self):
        a = dataframe_from_rows([('001', 2)], ['hidden-a', 'hidden-b'])
        b = dataframe_from_rows([('001', '2'), ('002', '3')], ['other-a', 'other-b'])
        report = compare_frame_profiles(a, b)
        self.assertFalse(report['row_count_equal'])
        self.assertTrue(report['column_count_equal'])
        self.assertFalse(report['dtype_distribution_equal'])
        self.assertNotIn('hidden-a', repr(report))
        self.assertNotIn('other-a', repr(report))


if __name__ == '__main__':
    unittest.main()
