"""Revision 169: automatic feature-search parallelism and panel cleanup."""
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from JBGFeatureSearch import resolve_feature_search_workers
from JBGResourceMonitor157 import resolve_parallel_workers


ROOT = Path(__file__).resolve().parents[1]
WIDGETS_SOURCE = (ROOT / "src" / "GUI" / "Widgets.py").read_text(encoding="utf-8")


class ParallelPolicy169Tests(unittest.TestCase):
    def test_shared_cpu_cap_semantics(self):
        self.assertEqual(resolve_parallel_workers(-1, max_cores=12, configured_cap=-1), 12)
        self.assertEqual(resolve_parallel_workers(None, max_cores=12, configured_cap=4), 4)
        self.assertEqual(resolve_parallel_workers(7, max_cores=12, configured_cap=4), 4)
        self.assertEqual(resolve_parallel_workers(3, max_cores=12, configured_cap=8), 3)

    def test_environment_global_cap_is_respected(self):
        with patch.dict(os.environ, {"DEFAULT_N_JOBS_DESIRED": "3"}):
            self.assertEqual(resolve_parallel_workers(-1, max_cores=16), 3)

    def test_feature_search_is_bounded_by_search_budget(self):
        workers, note = resolve_feature_search_workers(
            5, max_cores=16, configured_cap=-1,
            available=8 * 1024**3, total=16 * 1024**3,
        )
        self.assertEqual(workers, 5)
        self.assertIsNone(note)

    def test_feature_search_uses_memory_guard(self):
        workers, note = resolve_feature_search_workers(
            20, max_cores=16, configured_cap=8,
            available=512 * 1024**2, total=16 * 1024**3,
        )
        self.assertEqual(workers, 1)
        self.assertIn("Feature search memory guard", note)


class PanelLayout169Tests(unittest.TestCase):
    def test_search_workers_control_is_removed(self):
        self.assertNotIn("Search workers:", WIDGETS_SOURCE)
        self.assertNotIn("feature_search_parallel_workers", WIDGETS_SOURCE)

    def test_search_rows_has_dedicated_row(self):
        self.assertIn(
            "search_rows_row = widgets.HBox([\n            self.feature_search_rows,",
            WIDGETS_SOURCE,
        )

    def test_pipeline_controls_follow_execution_order(self):
        start = WIDGETS_SOURCE.index("pipeline_row = widgets.HBox([")
        end = WIDGETS_SOURCE.index("recovery_row = widgets.HBox([", start)
        block = WIDGETS_SOURCE[start:end]
        positions = [block.index(name) for name in (
            "self.feature_search_scaler",
            "self.feature_search_oversampling",
            "self.feature_search_reduction",
            "self.feature_search_model_type",
        )]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("feature_search_fit_budget", block)

    def test_fit_budget_reports_automatic_workers(self):
        self.assertIn("Candidate workers: automatic", WIDGETS_SOURCE)


if __name__ == "__main__":
    unittest.main()
