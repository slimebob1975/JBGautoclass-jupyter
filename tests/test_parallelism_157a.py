"""Regression tests for negative global worker caps (revision 157a)."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def _worker_resolver():
    """Load the pure worker-count method without importing ML dependencies."""
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_resolve_n_jobs":
            module = ast.Module(body=[node], type_ignores=[])
            scope = {}
            exec(compile(module, str(SOURCE), "exec"), scope)
            return scope[node.name]
    raise AssertionError("_resolve_n_jobs not found")


class Parallelism157aTests(unittest.TestCase):
    def setUp(self):
        self.resolve = _worker_resolver()

    def handler(self, cap):
        return SimpleNamespace(handler=SimpleNamespace(STANDARD_DESIRED_N_JOBS=cap))

    def test_minus_one_uses_cores(self):
        self.assertEqual(self.resolve(self.handler(-1), 16, 8), 8)

    def test_explicit_cap(self):
        self.assertEqual(self.resolve(self.handler(3), 16, 8), 3)

    def test_small_job_count(self):
        self.assertEqual(self.resolve(self.handler(-1), 2, 8), 2)

    def test_no_cpu_info(self):
        self.assertEqual(self.resolve(self.handler(-1), 16, 0), 1)

    def test_both_problematic_paths_use_resolver(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("n_jobs_desired = self._resolve_n_jobs(total_jobs, max_cores)", source)
        self.assertIn("available_cores = max(1, psutil.cpu_count(logical=True) or 1)", source)
        self.assertNotIn("min(total_jobs, max_cores, self.handler.STANDARD_DESIRED_N_JOBS)", source)


if __name__ == "__main__":
    unittest.main()
