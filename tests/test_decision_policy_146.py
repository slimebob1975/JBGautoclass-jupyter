import unittest
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from JBGDecisionPolicy import DecisionPipeline, validate_decision_policy

class TestDecisionPolicy(unittest.TestCase):
    def test_default_matches_estimator(self):
        x = np.arange(30).reshape(-1, 1)
        y = np.array(["rare"] * 4 + ["common"] * 26)
        p = DecisionPipeline([("model", DummyClassifier(strategy="prior"))]).fit(x,y)
        self.assertTrue(np.array_equal(p.predict(x), p.steps[-1][1].predict(x)))

    def test_threshold_clone_pickle(self):
        import pickle
        x = np.arange(30).reshape(-1, 1)
        y = np.array(["rare"] * 4 + ["common"] * 26)
        p = DecisionPipeline([("model", DummyClassifier(strategy="prior"))],
                             decision_policy="fixed", probability_threshold=0.1,
                             positive_label="rare")
        q = clone(p).fit(x,y)
        self.assertTrue(np.all(q.predict(x) == "rare"))
        self.assertEqual(q.probability_threshold, 0.1)
        self.assertTrue(np.array_equal(pickle.loads(pickle.dumps(q)).predict(x), q.predict(x)))
        self.assertAlmostEqual(q.predict_proba(x)[0, list(q.classes_).index("rare")], 4/30)

    def test_native_catboost_pipeline(self):
        import pandas as pd
        from JBGNativeCatBoost import RawCatBoostColumns, OrdinaryNativeCatBoostClassifier
        frame = pd.DataFrame({"category": ["one", "two"] * 24,
                              "amount": np.arange(48, dtype=float)})
        labels = np.array(["rare" if i % 7 == 0 else "common" for i in range(48)])
        pipeline = DecisionPipeline([
            ("native_columns", RawCatBoostColumns(tuple(frame.columns), ("category",))),
            ("CBST", OrdinaryNativeCatBoostClassifier(iterations=20, depth=3))],
            decision_policy="fixed", probability_threshold=0.1,
            positive_label="rare")
        fitted = pipeline.fit(frame, labels)
        self.assertEqual(len(fitted.predict(frame)), len(frame))
        self.assertEqual(fitted.predict_proba(frame).shape, (len(frame), 2))

    def test_reject_bad_parameters(self):
        for value in (-1, 0, 1, float("nan")):
            with self.assertRaises(ValueError):
                validate_decision_policy("fixed", value)
        with self.assertRaises(ValueError):
            validate_decision_policy("unknown", 0.1)

if __name__ == "__main__":
    unittest.main()
