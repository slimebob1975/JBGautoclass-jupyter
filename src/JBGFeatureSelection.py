"""Feature-selection guards evaluated inside each fitted training partition."""

import numpy as np
from sklearn.feature_selection import SelectFromModel


class EmptyFeatureSelectionError(ValueError):
    """A fitted selector would send zero columns to the next pipeline step."""


class NonEmptySelectFromModel(SelectFromModel):
    """Keep sklearn's selection policy, but reject empty output before transform.

    The inherited fit_transform fits on the current CV training fold before calling
    this method. No full-data probe, extra fit, threshold change or forced feature
    is needed. Inheriting the constructor preserves sklearn cloning/grid parameters.
    """

    def transform(self, X):
        support = self.get_support()
        if not np.any(support):
            estimator = getattr(self, "estimator_", self.estimator)
            threshold = (
                f"{self.threshold_:g}" if hasattr(self, "estimator_")
                else f"setting {self.threshold!r}"
            )
            raise EmptyFeatureSelectionError(
                f"SelectFromModel({type(estimator).__name__}) selected 0 of "
                f"{support.size} input features at threshold {threshold} "
                "in this fitted training partition. The downstream classifier was "
                "not run. Try a different preprocessing/model combination and check "
                "the training data for usable signal or excessive regularization."
            )
        return super().transform(X)
