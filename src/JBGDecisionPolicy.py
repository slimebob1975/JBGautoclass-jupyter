"""Binary fixed-probability classification decision, shared by ordinary JBG estimators.

This policy changes only predict(), never predict_proba().  Stored on the
pipeline so CV clones, persistence and dark-number predictions use one rule.
"""
import numpy as np
from imblearn.pipeline import Pipeline


def validate_decision_policy(policy, threshold):
    if policy not in ("default", "fixed"):
        raise ValueError("Unknown classification decision policy")
    try:
        threshold = float(threshold)
    except (ValueError, TypeError) as error:
        raise ValueError("Probability threshold must be numeric") from error
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Probability threshold must lie strictly between 0 and 1")
    return policy, threshold


class DecisionPipeline(Pipeline):
    """Cloneable imblearn pipeline with optional binary probability threshold."""

    def __init__(self, steps, *, memory=None, verbose=False,
                 decision_policy="default", probability_threshold=0.5,
                 positive_label=None):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.decision_policy = decision_policy
        self.probability_threshold = probability_threshold
        self.positive_label = positive_label

    def predict(self, X, **params):
        policy, threshold = validate_decision_policy(
            self.decision_policy, self.probability_threshold)
        if policy == "default":
            return super().predict(X, **params)
        if not hasattr(self.steps[-1][1], "predict_proba"):
            raise ValueError("Fixed probability threshold requires predict_proba()")
        classes = np.asarray(self.classes_)
        if len(classes) != 2:
            raise ValueError("Fixed probability threshold requires binary classification")
        if self.positive_label not in classes:
            raise ValueError("Configured positive class is absent from fitted classes")
        proba = np.asarray(self.predict_proba(X, **params))
        index = list(classes).index(self.positive_label)
        negative = classes[1 - index]
        return np.where(proba[:, index] >= threshold, self.positive_label, negative)
