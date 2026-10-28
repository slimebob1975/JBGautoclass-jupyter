"""Regression coverage of CSV model panel layout and state messages."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/GUI/Widgets.py").read_text(encoding="utf-8")

class ModelPanelStatusTests(unittest.TestCase):
    def test_models_row_order(self):
        tree = ast.parse(SOURCE)
        form = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                    and any(isinstance(k, ast.Constant) and k.value == "models" for k in node.keys))
        key_index = next(i for i,k in enumerate(form.keys) if isinstance(k,ast.Constant) and k.value=="models")
        row = ast.unparse(form.values[key_index])
        self.assertLess(row.index("models_dropdown"), row.index("field_status"))
        self.assertLess(row.index("field_status"), row.index("_csv_erase_model_button"))

    def test_model_states_are_explicit(self):
        for message in ("Ready to train a new CSV model", "Saved model verified - ready to predict",
                        "Saved CSV model unavailable (verification failed)",
                        "Confirm deletion of selected CSV model", "Saved CSV model erased",
                        "Unable to erase selected CSV model"):
            self.assertIn(message, SOURCE)

    def test_status_escapes_html(self):
        self.assertIn('html.escape(message)', SOURCE)

if __name__ == "__main__":
    unittest.main()
