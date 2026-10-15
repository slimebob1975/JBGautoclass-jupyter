"""Revision 131 GUI integration contract without requiring notebook/SQL runtimes."""
from pathlib import Path
import re
import unittest

from JBGFeatureSearch import SearchOptions


ROOT = Path(__file__).parents[1]
WIDGETS = (ROOT / "src/GUI/Widgets.py").read_text(encoding="utf-8")
CONFIG = (ROOT / "src/Config.py").read_text(encoding="utf-8")
HANDLER = (ROOT / "src/GUIHandler.py").read_text(encoding="utf-8")
SERVICE = (ROOT / "src/JBGFeatureSearchService.py").read_text(encoding="utf-8")
TEMPLATE = (ROOT / "src/config/autoclassconfig_template.py.txt").read_text(encoding="utf-8")


class FeatureSearchGuiContractTests(unittest.TestCase):
    def test_small_gui_budget_is_66_fit_ceiling(self):
        options = SearchOptions(trials=4, sizes=(5, 10, 20), inner_folds=3, outer_folds=3,
                                top_sets=1, refine_rounds=0, max_evaluations=5)
        self.assertEqual(options.planned_fit_upper_bound, 66)

    def test_three_required_mode_labels_exist(self):
        for label in (
            "I will pick my own data settings",
            "Help me pick the best data settings",
            "Use a recommended data setting",
        ):
            self.assertIn(label, WIDGETS)

    def test_panel_is_inserted_before_dataset_actions(self):
        self.assertIn('display_items.index("continuation_form")', WIDGETS)
        self.assertIn('"feature_selection_modelling_form"', WIDGETS)

    def test_required_investigative_families_exist(self):
        self.assertIn('"options": ["Linear", "Non-linear", "Tree", "Combination"]', WIDGETS)

    def test_reuse_mode_has_explicit_empty_profile_state(self):
        self.assertIn("No saved recommendations", WIDGETS)
        self.assertIn("Select and review a saved recommendation", WIDGETS)

    def test_family_specific_controls_are_gated(self):
        self.assertIn('family == "Linear"', WIDGETS)
        self.assertIn('family == "Non-linear"', WIDGETS)
        self.assertIn('family == "Tree"', WIDGETS)

    def test_apply_is_explicit_and_candidate_only(self):
        self.assertIn('"description": "Apply"', WIDGETS)
        self.assertIn('recommendation.get("validation_status") != "candidate"', WIDGETS)

    def test_apply_preserves_reviewed_text_and_category_choices(self):
        self.assertIn("old_text = tuple(self.text_columns.value)", WIDGETS)
        self.assertIn("old_categories = tuple(self.categorize_columns.value)", WIDGETS)
        self.assertIn("if column in selected", WIDGETS)

    def test_baseline_retained_preview_wording_exists(self):
        self.assertIn("Full-column baseline retained; no improvement found within this search budget.", WIDGETS)

    def test_saved_preview_and_two_step_deletion_wired(self):
        self.assertIn('self.feature_search_delete_button', WIDGETS)
        self.assertIn('Confirm delete', WIDGETS)
        self.assertIn('self.cancel_feature_search_deletion()', WIDGETS)
        self.assertIn('self.widgets.review_feature_search_candidate()', WIDGETS)
        self.assertIn('Nested selected MCC ', WIDGETS)
        self.assertIn('Macro precision ', WIDGETS)
        self.assertIn('Recommended columns: ', WIDGETS)

    def test_compact_profile_actions_and_tooltip_coverage(self):
        self.assertIn('profile_row = widgets.HBox(', WIDGETS)
        self.assertIn('self.feature_search_review_button,', WIDGETS)
        self.assertIn('self.feature_search_delete_button,', WIDGETS)
        self.assertIn('action_row = widgets.HBox(', WIDGETS)
        self.assertNotIn('grid_template_columns="repeat(3, minmax(0, 1fr))"', WIDGETS)
        for name in ('feature_selection_mode', 'feature_selection_profile',
                     'feature_search_fit_budget', 'feature_search_rows',
                     'feature_search_tree_max_depth'):
            self.assertIn('"' + name + '": "', WIDGETS)

    def test_generated_config_contains_mode_model_and_budget_fields(self):
        required = (
            "feature_selection_mode", "feature_selection_profile_id", "feature_search_model_type",
            "feature_search_trials", "feature_search_sizes", "feature_search_inner_folds",
            "feature_search_outer_folds", "feature_search_max_evaluations",
            "feature_search_linear_max_iterations", "feature_search_non_linear_max_iterations",
            "feature_search_tree_n_estimators",
        )
        for field in required:
            self.assertIn(f'"{field}"', TEMPLATE)
            self.assertIn(f'"mode.{field}"', CONFIG)

    def test_repeat_last_serializes_feature_search_contract(self):
        for field in (
            "feature_selection_mode", "feature_selection_profile_id", "feature_search_model_type",
            "feature_search_scaler", "feature_search_trials", "feature_search_sizes",
            "feature_search_inner_folds", "feature_search_outer_folds", "feature_search_max_evaluations",
        ):
            self.assertGreaterEqual(HANDLER.count(f'"{field}"'), 2)

    def test_training_only_service_splits_before_search(self):
        separate = SERVICE.index("dataset.separate_dataset()")
        split = SERVICE.index("dataset.split_dataset_for_training_and_validation()")
        search = SERVICE.index("run_experiment(", split)
        self.assertLess(separate, split)
        self.assertLess(split, search)

    def test_training_only_service_reserves_normal_holdout(self):
        self.assertIn("reserved_holdout_signature=partition_signature", SERVICE)
        self.assertIn("dataset.X_validation.index", SERVICE)
        self.assertIn("dataset.Y_validation", SERVICE)

    def test_investigation_service_does_not_run_classifier_or_repeat_last(self):
        self.assertNotIn("TaskRunner", SERVICE)
        self.assertNotIn("save_last_classifier_run", SERVICE)
        self.assertNotIn("save_model", SERVICE)
        self.assertNotIn("make_predictions", SERVICE)

    def test_investigation_blocks_continue_and_duplicate_start_until_finally(self):
        self.assertIn('if getattr(self.widgets, "_feature_search_running", False):', WIDGETS)
        self.assertIn('self._feature_search_running = True', WIDGETS)
        self.assertIn('self.disable_button("continuation_button")', WIDGETS)
        self.assertIn('self._feature_search_running = False', WIDGETS)
        self.assertIn('self.update_ready()', WIDGETS)

    def test_investigation_diagnostic_survives_ready_refresh(self):
        # update_ready calls sync_feature_selection_controls without preserve_status.
        method = WIDGETS.split('    def run_feature_selection_investigation(self) -> None:', 1)[1]
        method = method.split('    @staticmethod', 1)[0]
        self.assertIn('final_status = self.feature_search_status.value', method)
        self.assertLess(method.index('final_status = self.feature_search_status.value'),
                        method.index('self.update_ready()'))
        self.assertLess(method.index('self.update_ready()'),
                        method.index('self.feature_search_status.value = final_status'))
        self.assertIn('logging.getLogger(__name__).exception', method)

    def test_run_summary_exposes_active_mode_profile_and_model(self):
        self.assertIn('"Feature selection modelling"', CONFIG)
        self.assertIn('"Feature selection profile"', CONFIG)
        self.assertIn('"Investigative model"', CONFIG)


if __name__ == "__main__":
    unittest.main()
