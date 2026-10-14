"""The optional external-cohort API is not part of the standard GUI."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "src" / "GUI" / "Widgets.py"
API = ROOT / "src" / "JBGFeatureValidation.py"


class ValidationGuiContractTests(unittest.TestCase):
    def test_external_validation_controls_are_removed(self):
        source = GUI.read_text(encoding="utf-8")
        for token in ("feature_search_development_csv", "feature_search_external_csv",
                      "feature_search_cohort_description", "feature_search_validate_button",
                      "Optional external validation", "validate_feature_search_candidate("):
            self.assertNotIn(token, source)

    def test_standalone_validation_api_is_preserved(self):
        self.assertTrue(API.is_file())
        self.assertIn("def evaluate_external_candidate(", API.read_text(encoding="utf-8"))
        gui = GUI.read_text(encoding="utf-8")
        self.assertIn("def apply_feature_search_recommendation(", gui)
        self.assertIn("def review_feature_search_candidate(", gui)


if __name__ == "__main__":
    unittest.main()
