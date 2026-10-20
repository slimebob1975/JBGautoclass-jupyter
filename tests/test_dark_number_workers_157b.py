"""Execute the real Dark Numbers worker-count statements without ML dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "JBGHandler.py"

def compute_dark_workers(global_cap, folds=5, repeats=2, logical_cpus=12):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PredictionsHandler")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_calculate_dark_number_corrections")
    # Extract exact code from the implementation, not a parallel reimplementation.
    begin = next(i for i, node in enumerate(ast.walk(method))
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "available_cores" for t in node.targets))
    # The six contiguous statements define the worker count and cap.
    siblings = None
    for node in ast.walk(method):
        for _, child in ast.iter_fields(node):
            if isinstance(child, list) and begin in child:
                siblings = child
    # Locate the containing statement list by identity instead.
    anchor = next(node for node in ast.walk(method) if isinstance(node, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == "available_cores" for t in node.targets))
    for node in ast.walk(method):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and anchor in value:
                i = value.index(anchor)
                siblings = value[i:i+5]
                break
        if siblings is not None: break
    assert siblings is not None and len(siblings) == 5
    module = ast.fix_missing_locations(ast.Module(body=siblings, type_ignores=[]))
    ctx = dict(self=SimpleNamespace(handler=SimpleNamespace(STANDARD_DESIRED_N_JOBS=global_cap)),
               psutil=SimpleNamespace(cpu_count=lambda logical=True: logical_cpus),
               n_splits=folds, n_repeats=repeats)
    exec(compile(module, str(SOURCE), "exec"), ctx)
    return ctx["total_jobs"]

class DarkNumberWorkerCount157bTests(unittest.TestCase):
    def test_minus_one_means_available_parallelism(self):
        self.assertEqual(compute_dark_workers(-1, 5, 2, 12), 10)
    def test_explicit_positive_cap(self):
        self.assertEqual(compute_dark_workers(3, 5, 2, 12), 3)
    def test_missing_cpu_info(self):
        self.assertEqual(compute_dark_workers(-1, 5, 2, None), 1)
    def test_no_more_workers_than_jobs(self):
        self.assertEqual(compute_dark_workers(-1, 2, 2, 32), 4)

if __name__ == "__main__": unittest.main()
