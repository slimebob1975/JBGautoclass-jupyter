"""Targeted regression guards for unified SQL/CSV reclassification controls."""
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py').read_text(encoding='utf-8')


class ReclassificationApplyTests(unittest.TestCase):
    def test_explicit_apply_in_both_modes(self):
        start = SOURCE.index('    def handle_mispredicted(')
        end = SOURCE.index('    def create_inpage_styles(', start)
        view = SOURCE[start:end]
        self.assertIn('apply_button.on_click(apply_choice)', view)
        self.assertIn('self.correct_mispredicted_data(dropdown.value, row_id)', view)
        self.assertIn('csv_correction_callback(row_id, dropdown.value)', view)
        self.assertNotIn("('Keep'", view)
        self.assertNotIn('reclassify_dropdown__value', view)
        self.assertIn("value=actual", view)
        self.assertIn("change['new'] != original", view)
        self.assertIn("set_action_state(button, 'waiting')", view)
        self.assertIn("set_action_state(button, 'warning')", view)
        self.assertIn("set_action_state(button, 'ready'", view)

    def test_csv_model_selection_does_not_unlock_training_controls(self):
        start = SOURCE.index('    def _csv_model_selected(')
        end = SOURCE.index('    def _csv_source_selected(', start)
        self.assertIn('self._csv_set_training_controls(False)', SOURCE[start:end])
        continuation = SOURCE[SOURCE.index('    def _csv_continue_actions('):SOURCE.index('    def continuation_button_actions(')]
        self.assertIn('self.enable_items(list(self._CSV_TRAINING_EDITABLE_FIELDS)', continuation)

if __name__ == '__main__':
    unittest.main()
