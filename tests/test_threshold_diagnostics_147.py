import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from JBGThresholdDiagnostics import build_threshold_table, identity_overlap

class ThresholdDiagnosticsTests(unittest.TestCase):
    def test_confusion_and_metrics(self):
        y = np.array(["bad", "good", "bad", "good"])
        p = np.array([0.8, 0.3, 0.1, 0.01])
        r = build_threshold_table(y, p, "bad", thresholds=[0.2]).iloc[0]
        self.assertEqual((r.tp, r.fp, r.fn, r.tn), (1, 1, 1, 1))
        self.assertEqual(r.precision, .5)
        self.assertEqual(r.recall, .5)
    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            build_threshold_table([0, 1], [0.5, float('nan')], 1)
    def test_overlap(self):
        self.assertEqual(identity_overlap(pd.DataFrame({"iafId": [1, 1, 2]}),
                                          pd.DataFrame({"iafId": [2, 3]})), (1, 2, 2))
    def test_nonidentity(self):
        self.assertIsNone(identity_overlap(pd.DataFrame({"x":[1]}),
                                           pd.DataFrame({"x":[2]})))

if __name__ == '__main__': unittest.main()
