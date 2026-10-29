"""277c: CSV preview is read-only; SQL reclassification controls are not created."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_widget_render_163 import load_widgets_contract, make_view, pd


class CSVMostMispredicted277c(unittest.TestCase):
    def test_csv_preview_uses_shared_widget_without_sql_correction_controls(self):
        view = make_view()
        def sql_correction_not_allowed(*args):
            raise AssertionError('CSV GUI must not create SQL correction dropdowns')
        view.reclassify_dropdown = sql_correction_not_allowed
        frame = pd.DataFrame({'Actual': ['a', 'b'], 'Predicted': ['b', 'a']}, index=['001', '002'])
        rendered = load_widgets_contract().handle_mispredicted(
            view, frame, ['a', 'b'], allow_reclassification=False)
        self.assertTrue(rendered)
        # Index + two data columns, header + two rows.
        self.assertEqual(len(view.mispredicted_gridbox.children), 9)
        self.assertEqual(view.mispredicted_gridbox.children[3].value, '001')


if __name__ == '__main__':
    unittest.main()
