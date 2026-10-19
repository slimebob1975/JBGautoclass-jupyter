"""Deterministic, non-leaking holdout split for recurring individuals (Revision 148).

This changes only holdout membership. It does NOT change JBG's row-based CV,
model selection, or training-only threshold diagnostics.
"""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def person_disjoint_indices(X, y, test_size=0.2, group_column="iafId"):
    if group_column not in X.columns:
        raise ValueError(f"Person-disjoint validation requires {group_column!r} in input columns")
    groups = X[group_column]
    if groups.isna().any():
        raise ValueError(f"Person-disjoint validation requires non-null {group_column} values")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between zero and one")
    labels = np.asarray(y)
    if len(np.unique(labels)) != 2:
        raise ValueError("Person-disjoint validation currently requires binary classification")
    unique_groups = groups.nunique()
    folds = round(1.0 / test_size)
    if folds < 2 or unique_groups < folds:
        raise ValueError("Not enough distinct individuals for the requested holdout fraction")
    splits = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    candidates = []
    for train_idx, test_idx in splits.split(X, labels, groups=groups):
        train_classes = set(labels[train_idx]); test_classes = set(labels[test_idx])
        if len(train_classes) != 2 or len(test_classes) != 2:
            continue
        prevalence = np.mean(labels == Counter(labels).most_common()[-1][0])
        observed = np.mean(labels[test_idx] == Counter(labels).most_common()[-1][0])
        score = abs(len(test_idx) / len(labels) - test_size) + abs(observed - prevalence)
        candidates.append((score, train_idx, test_idx))
    if not candidates:
        raise ValueError("Cannot create person-disjoint holdout containing both classes")
    _, train_idx, test_idx = min(candidates, key=lambda entry: entry[0])
    assert not set(groups.iloc[train_idx]).intersection(groups.iloc[test_idx])
    return train_idx, test_idx
