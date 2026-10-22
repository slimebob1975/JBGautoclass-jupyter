"""Revision 168: saved-model target restoration for prediction-only tables."""
import ast
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
WIDGETS_PATH = ROOT / "src" / "GUI" / "Widgets.py"
WIDGETS = WIDGETS_PATH.read_text(encoding="utf-8")


class PredictionOnlyClass168(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(WIDGETS)
        cls.methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        helper_source = textwrap.dedent(
            ast.get_source_segment(WIDGETS, cls.methods["ensure_saved_model_class_option"])
        )
        dummy_config_type = type(
            "Config",
            (),
            {"TEXT_DATATYPES": ["nvarchar", "varchar"], "INT_DATATYPES": ["int", "bigint"]},
        )
        namespace = {"Config": dummy_config_type}
        exec(helper_source, namespace)
        cls.ensure_saved_model_class_option = staticmethod(
            namespace["ensure_saved_model_class_option"]
        )

    def _source(self, name: str) -> str:
        return ast.get_source_segment(WIDGETS, self.methods[name])

    @staticmethod
    def _dummy_widgets(options=()):
        return SimpleNamespace(
            datatype_dict={"class": "int", "other": "int", "when": "datetime"},
            class_column=SimpleNamespace(options=tuple(options)),
            _with_saved_options=lambda current, values: tuple(
                list(current) + [value for value in values if value not in current]
            ),
            _without_stale_na_placeholder=lambda current: tuple(
                value for value in current if value != "N/A"
            ),
        )

    def test_saved_model_target_is_restored_before_widget_values(self):
        source = self._source("update_from_model_config")
        self.assertIn("self.ensure_saved_model_class_option(config)", source)
        self.assertLess(
            source.index("self.ensure_saved_model_class_option(config)"),
            source.index("self.update_values({"),
        )

    def test_exact_saved_target_is_added_even_when_generic_options_are_empty(self):
        widgets = self._dummy_widgets()
        config = SimpleNamespace(get_class_column_name=lambda: "class")
        self.ensure_saved_model_class_option(widgets, config)
        self.assertEqual(widgets.class_column.options, ("class",))

    def test_unrelated_empty_columns_are_not_promoted(self):
        widgets = self._dummy_widgets(options=("other",))
        config = SimpleNamespace(get_class_column_name=lambda: "class")
        self.ensure_saved_model_class_option(widgets, config)
        self.assertEqual(widgets.class_column.options, ("other", "class"))
        self.assertNotIn("when", widgets.class_column.options)

    def test_missing_or_unsupported_saved_target_is_not_added(self):
        no_metadata = self._dummy_widgets(options=("other",))
        no_metadata.datatype_dict = None
        self.ensure_saved_model_class_option(
            no_metadata, SimpleNamespace(get_class_column_name=lambda: "class")
        )
        self.assertEqual(no_metadata.class_column.options, ("other",))

        missing = self._dummy_widgets(options=("other",))
        self.ensure_saved_model_class_option(
            missing, SimpleNamespace(get_class_column_name=lambda: "missing")
        )
        self.assertEqual(missing.class_column.options, ("other",))

        unsupported = self._dummy_widgets(options=("other",))
        self.ensure_saved_model_class_option(
            unsupported, SimpleNamespace(get_class_column_name=lambda: "when")
        )
        self.assertEqual(unsupported.class_column.options, ("other",))

    def test_manual_training_cardinality_rule_is_unchanged(self):
        source = self._source("refresh_class_column_options")
        self.assertIn("2 <= self._class_cardinalities.get(name, ceiling + 1) <= ceiling", source)


if __name__ == "__main__":
    unittest.main()
