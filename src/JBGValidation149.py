"""Dataset-agnostic group CV and predictor exclusion (Revision 149)."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def group_folds(X, y, groups, n_splits=10, random_state=1):
    """Return a reproducible, validated sequence of disjoint-group CV folds."""
    groups = np.asarray(groups)
    labels = np.asarray(y)
    if len(groups) != len(labels):
        raise ValueError("Group identifiers and labels have different lengths")
    if len(groups) < n_splits or len(set(groups)) < n_splits:
        raise ValueError("Not enough distinct groups for requested CV folds")
    if any(g is None or (isinstance(g, float) and np.isnan(g)) for g in groups):
        raise ValueError("Group identifiers contain nulls")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(X, labels, groups=groups))
    for train, test in folds:
        if not set(groups[train]).isdisjoint(set(groups[test])):
            raise ValueError("Group overlap detected in CV fold")
        if len(set(labels[train])) != len(set(labels)) or len(set(labels[test])) != len(set(labels)):
            raise ValueError("A group CV fold lacks a target class; reduce folds")
    return folds


def validation_folds(X, y, *, groups=None, n_splits=10, random_state=1):
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return group_folds(X, y, groups, n_splits, random_state)


def parse_excluded_columns(value):
    return tuple(dict.fromkeys(v.strip() for v in str(value or "").split(",") if v.strip()))
