"""273i: exercise the shared misprediction task without opening CSV GUI mode.

This is an execution-level boundary test. CSV artifact publication and the
GUI switch remain gated until the destination is scoped and validated.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "JBGTaskRunner.py"


def load_shared_task():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name == "display_mispredicted__task"]
    if len(methods) != 1:
        raise AssertionError("Expected exactly one shared misprediction task")
    holder = ast.ClassDef(name="TaskUnderTest", bases=[], keywords=[],
                          body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[holder], type_ignores=[]))
    namespace = {"Pipeline": object}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["TaskUnderTest"]


class CSVSharedMisprediction273i(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.task = load_shared_task()()
        calls = self.calls
        self.task.config = SimpleNamespace(
            should_display_mispredicted=lambda: True,
            should_calculate_dark_numbers=lambda: False,
            get_output_filepath=lambda kind: "isolated-test-output.csv",
        )
        self.task.regression_suite = False
        self.task.logger = SimpleNamespace(
            print_task_header=lambda **kwargs: calls.append(("header", kwargs)),
            print_progress=lambda **kwargs: calls.append(("progress", kwargs)),
        )
        self.task.dh = SimpleNamespace(X_original="original", X="features", Y="labels")
        self.task.ph = SimpleNamespace(
            most_mispredicted=lambda *args: calls.append(("calculate", args)),
            evaluate_mispredictions=lambda path: calls.append(("publish", path)),
        )

    def test_shared_task_computes_and_publishes_using_existing_handlers(self):
        result = self.task.display_mispredicted__task("cross", "trained")
        self.assertEqual(result, {})
        self.assertIn(("calculate", ("original", "trained", "cross", "features", "labels")),
                      self.calls)
        self.assertIn(("publish", "isolated-test-output.csv"), self.calls)

    def test_disabled_diagnostic_does_not_compute_or_publish(self):
        self.task.config.should_display_mispredicted = lambda: False
        self.task.display_mispredicted__task("cross", "trained")
        self.assertFalse(any(event in ("calculate", "publish") for event, _ in self.calls))

    def test_diagnostic_failure_is_not_silently_swallowed(self):
        def fail(*args):
            raise RuntimeError("synthetic analysis failure")
        self.task.ph.most_mispredicted = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic analysis failure"):
            self.task.display_mispredicted__task("cross", "trained")
        self.assertFalse(any(event == "publish" for event, _ in self.calls))


if __name__ == "__main__":
    unittest.main()
