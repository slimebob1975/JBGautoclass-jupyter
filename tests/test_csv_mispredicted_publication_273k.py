"""Source-aware publication for the existing shared misprediction calculation.

Extract the production method by AST so these boundary tests do not require the
entire optional ML dependency stack. Full Windows/Voila acceptance remains needed.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
methods = [node for klass in tree.body if isinstance(klass, ast.ClassDef)
           for node in klass.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name == "evaluate_mispredictions"]
assert len(methods) == 1, "Expected one production misprediction evaluation method"
namespace = {"Helpers": SimpleNamespace(save_matrix_as_csv=Mock(), create_download_link=Mock(return_value="link"))}
exec(compile(ast.Module(body=[methods[0]], type_ignores=[]), str(SOURCE), "exec"), namespace)
evaluate = namespace["evaluate_mispredictions"]


class Publication273k(TestCase):
    def setUp(self):
        namespace["Helpers"].save_matrix_as_csv.reset_mock()
        namespace["Helpers"].create_download_link.reset_mock()
        self.frame = pd.DataFrame({"prediction": ["B"]}, index=["00017"])
        self.logger = Mock()
        self.sink = Mock()

    def _predictions(self, source="csv", sink=True, enabled=True):
        cfg = SimpleNamespace(data_source=SimpleNamespace(kind=source),
                              should_display_mispredicted=lambda: enabled,
                              get_id_column_name=lambda: "row_id")
        layer = SimpleNamespace(save_mispredicted_data=self.sink) if sink else object()
        owner = SimpleNamespace(X_most_mispredicted=self.frame, X_mispredicted=self.frame,
                                num_mispredicted=1,
                                handler=SimpleNamespace(config=cfg, datalayer=layer,
                                                        logger=self.logger, read_data_query="SELECT * FROM data"),
                                _get_rid_of_TOP_stuffs=lambda q: q)
        return owner

    def test_csv_publishes_existing_dataframe_without_sql_or_unscoped_file(self):
        evaluate(self._predictions(), "unused.csv")
        self.sink.assert_called_once_with(self.frame)
        self.logger.print_query.assert_not_called()
        namespace["Helpers"].save_matrix_as_csv.assert_not_called()

    def test_csv_missing_sink_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "sink is unavailable"):
            evaluate(self._predictions(sink=False), "unused.csv")
        namespace["Helpers"].save_matrix_as_csv.assert_not_called()

    def test_sql_preserves_existing_file_publication(self):
        evaluate(self._predictions(source="sql"), "report.csv")
        self.sink.assert_not_called()
        namespace["Helpers"].save_matrix_as_csv.assert_called_once_with(self.frame, "report.csv", "row_id")
        self.logger.print_query.assert_called_once()

    def test_disabled_option_writes_nothing(self):
        evaluate(self._predictions(enabled=False), "unused.csv")
        self.sink.assert_not_called()
        namespace["Helpers"].save_matrix_as_csv.assert_not_called()


if __name__ == "__main__":
    main()
