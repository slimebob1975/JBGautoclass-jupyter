"""Regression contracts for opt-in saved recommendation application."""
from pathlib import Path
import unittest

WIDGETS = (Path(__file__).resolve().parents[1] / 'src/GUI/Widgets.py').read_text(encoding='utf-8')

class SavedRecommendationApplyTests(unittest.TestCase):
    def test_reuse_requires_reviewable_compatibility(self):
        self.assertIn('reuse and self._saved_recommendation_issues() == []', WIDGETS)
        self.assertIn('if issues:', WIDGETS)
        self.assertIn('record = find_candidate(', WIDGETS)
        self.assertIn('sql_schema=getattr(self, "datatype_dict", None)', WIDGETS)
        self.assertIn('Current column type metadata is incomplete', WIDGETS)

    def test_saved_text_settings_and_explicit_apply(self):
        self.assertIn('old_text = tuple(saved_text.get("text_columns", ()))', WIDGETS)
        self.assertIn('old_categories = tuple(saved_text.get("category_columns", ()))', WIDGETS)
        self.assertIn('self.data_columns.value = selected', WIDGETS)
        self.assertIn('self.filter_checkbox.value = bool(saved_text["stop_words"])', WIDGETS)
        self.assertIn('self.ngram_range_dropdown.value = saved_ngram_name', WIDGETS)
        self.assertIn('self.encryption_checkbox.value = bool(saved_text["use_encryption"])', WIDGETS)
        self.assertIn('self.categorize_checkbox.value = bool(saved_text["use_categorization"])', WIDGETS)
        self.assertIn('with its recorded text representation', WIDGETS)
        self.assertIn('handler=self.eventhandler.feature_selection_profile', WIDGETS)
        self.assertIn('not independently validated', WIDGETS)

if __name__ == '__main__': unittest.main()
