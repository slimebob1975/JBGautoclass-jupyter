"""Read-only permutation importance of original input columns on a final holdout."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import check_scoring
from sklearn.utils.metaestimators import available_if

from Helpers import prepare_estimator_input


class FittedInputClassifier(ClassifierMixin, BaseEstimator):
    """Expose a fitted converter + classifier pipeline without fitting either again."""

    def __init__(self, pipeline, converter=None):
        self.pipeline = pipeline
        self.converter = converter

    @property
    def classes_(self):
        return self.pipeline.classes_

    def __sklearn_is_fitted__(self):
        return hasattr(self.pipeline, "classes_")

    def fit(self, X, y=None):
        raise RuntimeError("Feature importance uses a fixed fitted model; fitting is disabled")

    def _call(self, method, X):
        converted = self.converter.transform(X) if self.converter is not None else X
        call = getattr(self.pipeline, method)
        try:
            return call(prepare_estimator_input(converted))
        except TypeError:
            # Keep the application's existing dense-estimator compatibility path.
            return call(converted.to_numpy())

    def predict(self, X):
        return self._call("predict", X)

    @available_if(lambda self: hasattr(self.pipeline, "predict_proba"))
    def predict_proba(self, X):
        return self._call("predict_proba", X)

    @available_if(lambda self: hasattr(self.pipeline, "decision_function"))
    def decision_function(self, X):
        return self._call("decision_function", X)


@dataclass
class FeatureImportanceReport:
    table: pd.DataFrame
    baseline_score: float
    rows: int
    repeats: int
    seed: int
    seconds: float


def compute_feature_importance(pipeline, X, y, scoring, converter=None,
                               repeats=5, seed=42, progress=None):
    """Shuffle whole original columns, preserving sparse conversion and the full pipeline.

    All holdout rows are used: subsampling could discard rare positive labels.
    Repeated score decreases measure reliance of this particular fitted model,
    not causality or importance intrinsic to the dataset. No model/transform fit occurs.
    """
    if not isinstance(X, pd.DataFrame) or X.empty or X.shape[1] == 0:
        raise ValueError("Feature importance needs a non-empty original-column holdout DataFrame")
    if X.columns.has_duplicates:
        raise ValueError("Feature importance needs unique original column names")
    if len(y) != len(X) or isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("Feature importance holdout features and labels are not aligned")
    if type(repeats) is not int or not 2 <= repeats <= 30:
        raise ValueError("Feature importance repeats must be an integer between 2 and 30")

    adapter = FittedInputClassifier(pipeline, converter)
    scorer = check_scoring(adapter, scoring=scoring)
    total = 1 + X.shape[1] * repeats
    completed = 0
    baseline = None

    def score(estimator, features, labels):
        nonlocal completed, baseline
        value = float(scorer(estimator, features, labels))
        if not np.isfinite(value):
            raise ValueError("Feature importance scorer returned a nonfinite value")
        if baseline is None:
            baseline = value
        completed += 1
        if progress is not None:
            progress(completed, total)
        return value

    started = perf_counter()
    result = permutation_importance(
        adapter, X, y, scoring=score, n_repeats=repeats, random_state=seed,
        n_jobs=1, max_samples=1.0,
    )
    table = pd.DataFrame({
        "Feature": [str(column) for column in X.columns],
        "Mean score decrease": result.importances_mean,
        "Std. across repeats": result.importances_std,
    })
    for repeat in range(repeats):
        table[f"Repeat {repeat + 1}"] = result.importances[:, repeat]
    table = table.sort_values("Mean score decrease", ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return FeatureImportanceReport(table, baseline, len(X), repeats, seed, perf_counter() - started)
