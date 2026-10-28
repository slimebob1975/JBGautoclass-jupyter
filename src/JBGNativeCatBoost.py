"""Native mixed-type input for ordinary JBG CatBoost (revision 144)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

class RawCatBoostColumns(BaseEstimator, TransformerMixin):
    """Restore named mixed-type columns even after legacy correction code uses NumPy.

    Categorical identities are fixed from the original schema before cross-validation,
    never inferred from a held-out validation fold or a transformed NumPy dtype.
    """
    def __init__(self, columns, categorical_columns):
        self.columns = columns
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.loc[:, list(self.columns)].copy()
        else:
            arr = np.asarray(X)
            if arr.ndim != 2 or arr.shape[1] != len(self.columns):
                raise ValueError("CatBoost input shape does not match the saved column schema")
            frame = pd.DataFrame(arr, columns=list(self.columns))
        categories = set(self.categorical_columns)
        for name in self.columns:
            if name in categories:
                frame[name] = frame[name].astype("string").fillna("__MISSING__").astype(object)
            else:
                frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
        return frame


def is_native_catboost_pipeline(pipeline):
    return bool(getattr(pipeline, "steps", None)) and any(
        isinstance(step, RawCatBoostColumns) for _, step in pipeline.steps
    )


def prepare_model_input(data, pipeline, *, prefer_numpy=False):
    from Helpers import prepare_estimator_input
    if is_native_catboost_pipeline(pipeline):
        return data
    return prepare_estimator_input(data, prefer_numpy=prefer_numpy)


class OrdinaryNativeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """The ordinary sklearn-compatible CatBoost estimator (independent of GUI imports)."""
    def __init__(self, iterations=400, depth=6, learning_rate=0.05,
                 class_weight='None', random_seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.random_seed = random_seed

    def fit(self, X, y):
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError('CatBoost requires the catboost package: pip install catboost') from exc
        if not isinstance(X, pd.DataFrame):
            raise TypeError('CatBoost native categorical input must be a DataFrame')
        categorical = [column for column in X
                       if not pd.api.types.is_numeric_dtype(X[column])]
        self.estimator_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, random_seed=self.random_seed,
            auto_class_weights=('Balanced' if self.class_weight == 'Balanced' else None),
            cat_features=categorical, thread_count=1, verbose=False,
            allow_writing_files=False)
        self.estimator_.fit(X, y)
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).ravel()

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
