"""Synthetic CSV Iris-like role inference and semantic button contract."""
import tempfile
import unittest
from pathlib import Path

from CSVColumnSchema import inspect_csv_columns
from GUI.ButtonStates import set_action_state, COLORS


class CSVTypeInference250(unittest.TestCase):
    def test_numeric_measurements_not_class_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.csv"
            path.write_text("id,sepal-length,sepal-width,petal-length,petal-width,class\n"
                            "001,5.1,3.5,1.4,0.2,setosa\n"
                            "002,4.9,3.5,1.4,0.2,setosa\n"
                            "003,6.3,2.9,5.6,2.1,virginica\n"
                            "004,5.5,2.9,4.0,2.1,versicolor\n")
            schema = inspect_csv_columns(path, class_cardinality_limit=3)
            self.assertEqual(schema.class_candidates, ("class",))
            self.assertIn("id", schema.unique_id_candidates)

    def test_blank_numeric_values_and_decimal_comma(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.csv"
            path.write_text("id;measurement;class\n1;1,2;red\n2;2,3;blue\n3;;red\n")
            schema = inspect_csv_columns(path, delimiter=";")
            self.assertEqual(schema.class_candidates, ("class",))

    def test_text_numeric_mixed_is_not_silently_numeric(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.csv"
            path.write_text("id,measurement,class\n1,1,red\n2,unknown,blue\n3,2,red\n")
            schema = inspect_csv_columns(path)
            self.assertIn("measurement", schema.class_candidates)


class FakeButton:
    def __init__(self):
        self.disabled = False
        self.button_style = ""
        self.classes = set()
    def add_class(self, name):
        self.classes.add(name)
    def remove_class(self, name):
        self.classes.discard(name)


class ButtonStates250(unittest.TestCase):
    def test_state_transitions(self):
        button = FakeButton()
        for state, expected_disabled, style in [("blocked", True, "danger"),
                                                ("ready", False, "success"),
                                                ("waiting", True, "warning"),
                                                ("warning", False, "warning")]:
            set_action_state(button, state)
            self.assertEqual(button.disabled, expected_disabled)
            self.assertEqual(button.button_style, style)
            self.assertEqual(button.classes, {"jbg-action-" + state})
        self.assertEqual(len(set(COLORS.values())), 4)

    def test_unknown_state_rejected(self):
        with self.assertRaises(ValueError):
            set_action_state(FakeButton(), "unknown")


if __name__ == "__main__":
    unittest.main()
