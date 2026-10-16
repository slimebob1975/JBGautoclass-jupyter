"""Regression guards for the class cardinality and Compare selectors."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDGETS = (ROOT / 'src/GUI/Widgets.py').read_text(encoding='utf-8')
SQL = (ROOT / 'src/SQLDataLayer.py').read_text(encoding='utf-8')


class GuiColumnFilterTests(unittest.TestCase):
    def test_class_cardinality_limit_and_refresh(self):
        self.assertIn('IntSlider(value=3, min=2, max=30', WIDGETS)
        self.assertIn('continuous_update=False', WIDGETS)
        self.assertIn('self.refresh_class_column_options()', WIDGETS)
        self.assertIn('self.class_column.options = eligible', WIDGETS)
        self.assertIn('self._class_cardinalities.get(name, ceiling + 1)', WIDGETS)

    def test_slider_above_class_radio_group(self):
        self.assertIn('widgets.VBox([self.class_cardinality_limit, self.class_column])', WIDGETS)
        self.assertIn('[class_controls, self.id_column, self.data_columns, self.text_columns]', WIDGETS)

    def test_sql_cardinality_cached_per_dataset(self):
        self.assertIn('def get_column_cardinalities(', SQL)
        self.assertIn('COUNT(DISTINCT', SQL)
        self.assertIn('CASE WHEN', SQL)
        self.assertIn('self._class_cardinalities = self.datalayer.get_column_cardinalities(', WIDGETS)

    def test_compare_by_selected_categorical_or_integer_columns(self):
        self.assertIn('self._feature_search_subgroup_column_widget = widgets.Dropdown(', WIDGETS)
        self.assertIn('for name in self.data_columns.value', WIDGETS)
        self.assertIn('self.widgets.refresh_feature_search_compare_options()', WIDGETS)
        self.assertIn('name not in (self.class_column.value, self.id_column.value)', WIDGETS)
        self.assertIn('(self.feature_search_subgroup_column.value or None)', WIDGETS)


if __name__ == '__main__':
    unittest.main()
