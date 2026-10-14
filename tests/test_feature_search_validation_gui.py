"""Regression guards for review-only validation wiring."""
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'


class ValidationGuiContractTests(unittest.TestCase):
    def test_gui_wires_explicit_cohorts_to_frozen_validation(self):
        source = SRC.read_text(encoding='utf-8')
        for token in ('feature_search_development_csv', 'feature_search_external_csv',
                      'feature_search_cohort_description',
                      'feature_search_validate_button_was_clicked',
                      'evaluate_external_candidate(', 'development_ids=dev[id_col]',
                      'external_ids=ext[id_col]', 'review_compatibility(',
                      'Optional external validation', 'profile_id'):
            self.assertIn(token, source)

    def test_reuse_is_not_approved_by_validation_gui(self):
        source = SRC.read_text(encoding='utf-8')
        validation = source.split('def validate_feature_search_candidate(self)', 1)[1].split('def save_feature_search_candidate', 1)[0]
        self.assertNotIn('apply_feature_search_recommendation(', validation)
        self.assertNotIn('eligible_for_reuse(', validation)
        self.assertIn('Candidate remains ineligible for reuse', validation)

if __name__ == '__main__':
    unittest.main()
