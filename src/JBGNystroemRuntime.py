"""Transient CV diagnostics for the expensive Nystroem/MLP combinations.

Only CV clones are instrumented. Original pipelines and persisted models retain
ordinary sklearn Nystroem instances and their existing parameters.
"""
from time import perf_counter

import numpy as np
from sklearn import get_config
from sklearn.base import clone
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import check_scoring
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_info


PROFILE_ALGORITHMS = frozenset({'MLPC', 'MLP2', 'FUTV', 'FUTS'})
METRICS = (
    'nys_fit_transform_seconds', 'nys_score_transform_seconds',
    'nys_training_rows', 'nys_input_features', 'nys_output_features', 'nys_output_bytes',
    'mlp_count', 'mlp_iterations_min', 'mlp_iterations_max',
    'mlp_max_iter_min', 'mlp_max_iter_max', 'mlp_at_limit_count', 'native_threads_max',
)


class TimedNystroem(Nystroem):
    """Same transformer, with scalar measurements on a disposable CV clone."""
    def fit_transform(self, X, y=None, **fit_params):
        started = perf_counter()
        result = super().fit_transform(X, y, **fit_params)
        self.runtime_fit_transform_seconds_ = perf_counter() - started
        self.runtime_training_rows_ = result.shape[0]
        self.runtime_input_features_ = self.n_features_in_
        self.runtime_output_features_ = result.shape[1]
        self.runtime_output_bytes_ = result.nbytes
        # Training transforms are part of the fit measurement, not scoring.
        self.runtime_score_transform_seconds_ = 0.0
        return result

    def transform(self, X):
        started = perf_counter()
        result = super().transform(X)
        self.runtime_score_transform_seconds_ = (
            getattr(self, 'runtime_score_transform_seconds_', 0.0) + perf_counter() - started
        )
        return result


def _fitted_mlps(estimator):
    """Inspect retained base estimators; never run extra fits or predictions.

    Stacking's internal out-of-fold estimators are discarded by sklearn. Counts
    here describe retained fitted base MLPs only, not those internal fits.
    """
    if isinstance(estimator, MLPClassifier):
        return [estimator]
    result = []
    for child in getattr(estimator, 'named_estimators_', {}).values():
        if child is not None and not isinstance(child, str):
            result.extend(_fitted_mlps(child))
    final = getattr(estimator, 'final_estimator_', None)
    if final is not None:
        result.extend(_fitted_mlps(final))
    return result


def _fold_metrics(pipeline):
    nys = pipeline.named_steps['NYS']
    mlps = _fitted_mlps(pipeline.steps[-1][1])
    iterations = [model.n_iter_ for model in mlps]
    budgets = [model.max_iter for model in mlps]
    pools = threadpool_info()
    return dict(zip(METRICS, (
        nys.runtime_fit_transform_seconds_, nys.runtime_score_transform_seconds_,
        nys.runtime_training_rows_, nys.runtime_input_features_,
        nys.runtime_output_features_, nys.runtime_output_bytes_,
        len(mlps), min(iterations, default=np.nan), max(iterations, default=np.nan),
        min(budgets, default=np.nan), max(budgets, default=np.nan),
        sum(model.n_iter_ >= model.max_iter for model in mlps),
        max((pool['num_threads'] for pool in pools), default=0),
    )))


class RuntimeScorer:
    """Return the original score plus numeric metadata from that same fold."""
    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, pipeline, X, y):
        # Score errors retain the original behavior; only optional diagnostics
        # are allowed to fail without affecting model selection.
        score = self.scorer(pipeline, X, y)
        try:
            metrics = _fold_metrics(pipeline)
            available = 1.0
        except Exception:
            metrics = dict.fromkeys(METRICS, np.nan)
            available = 0.0
        return {'score': score, 'runtime_available': available, **metrics}


def prepare_runtime_cv(pipeline, scorer, algorithm_name):
    """Instrument eligible CV clones without changing the caller's pipeline."""
    steps = getattr(pipeline, 'named_steps', {})
    nys = steps.get('NYS')
    if (algorithm_name not in PROFILE_ALGORITHMS or type(nys) is not Nystroem
            or getattr(nys, '_sklearn_output_config', {}).get('transform', 'default') != 'default'
            or get_config().get('transform_output', 'default') != 'default'
            or getattr(pipeline, 'memory', None) is not None):
        return pipeline, scorer, False
    cv_pipeline = clone(pipeline)
    cv_pipeline.set_params(NYS=TimedNystroem(**cv_pipeline.named_steps['NYS'].get_params(deep=False)))
    return cv_pipeline, RuntimeScorer(check_scoring(pipeline, scoring=scorer)), True


def build_runtime_rows(pipeline, results, workers, wall_seconds):
    """Keep fold times distinct from wall time; residual fit work is not MLP-only."""
    label = '-'.join(name for name, _ in pipeline.steps)
    classifier = pipeline.steps[-1][1]
    inner_cv = getattr(classifier, 'cv', None)
    inner_cv = inner_cv if isinstance(inner_cv, int) else None
    nys = pipeline.named_steps['NYS']
    rows = []
    for index, fit_seconds in enumerate(results['fit_time']):
        measured = results['test_runtime_available'][index] == 1.0
        metrics = {name: float(results[f'test_{name}'][index]) for name in METRICS}
        nys_seconds = metrics['nys_fit_transform_seconds']
        rows.append({
            'Pipeline': label, 'Fold': index + 1, 'Folds': len(results['fit_time']),
            'Workers': workers, 'CV wall seconds': wall_seconds,
            'Score': float(results['test_score'][index]),
            'Fit seconds': float(fit_seconds),
            'Score seconds': float(results['score_time'][index]),
            'Other fit seconds': max(0.0, float(fit_seconds) - nys_seconds) if measured else np.nan,
            'Runtime available': measured, 'Kernel': str(nys.kernel),
            'Gamma setting': nys.gamma, 'Configured components': nys.n_components,
            'Ensemble internal CV': inner_cv,
            **metrics,
        })
    return rows


def format_runtime_summary(rows):
    valid = [row for row in rows if row['Runtime available']]
    if not valid:
        return f"Nystroem runtime — {rows[0]['Pipeline']}: diagnostics unavailable; CV score retained."
    mean = lambda key: float(np.mean([row[key] for row in valid]))
    fits = mean('Fit seconds')
    nys = mean('nys_fit_transform_seconds')
    share = f'{100 * nys / fits:.1f}%' if fits > 0 else 'unavailable'
    mlp_count = sum(row['mlp_count'] for row in valid)
    reached = sum(row['mlp_at_limit_count'] for row in valid)
    iterations = [row['mlp_iterations_max'] for row in valid if row['mlp_count'] > 0]
    budgets = [row['mlp_max_iter_max'] for row in valid if row['mlp_count'] > 0]
    mlp = (
        f"retained MLP n_iter_ maximum {max(iterations):.0f}, max_iter maximum {max(budgets):.0f}; "
        f"{reached:.0f}/{mlp_count:.0f} retained MLP fits reached their iteration cap"
        if iterations else 'retained MLP iteration counts unavailable'
    )
    inner_cv = valid[0]['Ensemble internal CV']
    stacking = (f'; stacking internal CV={inner_cv}, whose discarded internal fit iterations are not measured'
                if inner_cv is not None else '')
    return (
        f"Nystroem runtime — {valid[0]['Pipeline']}: "
        f"CV wall {valid[0]['CV wall seconds']:.2f} s, {valid[0]['Workers']} workers; "
        f"mean fold fit {fits:.2f} s = NYS fit/transform {nys:.2f} s ({share}) "
        f"+ other pipeline fit work {mean('Other fit seconds'):.2f} s; "
        f"mean NYS scoring transforms {mean('nys_score_transform_seconds'):.3f} s. "
        f"{mlp}; observed native threads maximum {max(row['native_threads_max'] for row in valid):.0f}"
        f"{stacking}. Measured {len(valid)}/{len(rows)} folds; other work includes preprocessing, "
        "sampling and classifier fitting. Fold times are concurrent work, not additive wall time."
    )
