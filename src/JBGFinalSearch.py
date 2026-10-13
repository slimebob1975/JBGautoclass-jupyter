"""Bounded Extra Trees tuning and training-only final-search diagnostics."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import ParameterGrid


def extra_trees_search_parameters():
    """24 profiles including the exact current spot-check factory defaults."""
    return {
        'criterion': ('gini',),
        'n_estimators': (100,),
        'max_depth': (None, 12, 24),
        'max_features': ('sqrt', 0.5),
        'min_samples_leaf': (1, 3),
        'class_weight': (None, 'balanced'),
    }


def matches_parameters(candidate, baseline):
    """Unspecified parameters inherit the unchanged pipeline settings."""
    return all(np.array_equal(value, baseline[key]) for key, value in candidate.items())


def include_extra_trees_baseline(model, grid):
    """Keep custom Extra Trees baselines eligible without changing other grids."""
    if not hasattr(model, 'steps') or not isinstance(model.steps[-1][1], ExtraTreesClassifier):
        return grid, None
    candidates = list(ParameterGrid(grid))
    current = model.get_params(deep=True)
    baseline = {key: current[key] for candidate in candidates for key in candidate}
    if any(matches_parameters(candidate, baseline) for candidate in candidates):
        return grid, baseline
    branches = [dict(grid)] if isinstance(grid, dict) else [dict(branch) for branch in grid]
    branches.append({key: [value] for key, value in baseline.items()})
    return branches, baseline


def final_search_summary(search, baseline, metric, spot_check_score=None):
    """Summarize existing CV results; never fit or evaluate holdout data."""
    results = search.cv_results_
    best_index = int(search.best_index_)
    baseline_rows = [i for i, params in enumerate(results['params'])
                     if baseline is not None and matches_parameters(params, baseline)]
    baseline_index = baseline_rows[0] if baseline_rows else None
    best_score = float(search.best_score_)
    base_score = float(results['mean_test_score'][baseline_index]) if baseline_index is not None else None
    return {
        'metric': metric,
        'data_scope': 'Training-partition cross-validation; final holdout excluded',
        'candidate_count': len(results['params']),
        'cv_folds': int(search.n_splits_),
        'best_cv_mean': best_score,
        'best_cv_std': float(results['std_test_score'][best_index]),
        'best_params': dict(search.best_params_),
        'baseline_cv_mean': base_score,
        'baseline_cv_std': float(results['std_test_score'][baseline_index]) if baseline_index is not None else None,
        'best_minus_baseline': best_score - base_score if base_score is not None else None,
        'spot_check_cv_mean': float(spot_check_score) if spot_check_score is not None else None,
    }, baseline_rows
