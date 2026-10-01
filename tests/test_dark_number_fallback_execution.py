import os
from pickle import PicklingError
import threading
from time import sleep

from joblib import delayed
import numpy as np
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD

import JBGDarkNumberExecution as execution
import JBGDarkNumberCorrectionFactor as correction


class RecordingLogger:
    def __init__(self):
        self.info = []
        self.warnings = []
        self.updates = []
        self.starts = []
        self.ends = []
        self.threads = []

    def print_info(self, message):
        self.info.append(message)

    def print_warning(self, message):
        self.warnings.append(message)

    def start_inline_progress(self, key, description, final_count, tooltip):
        self.starts.append((key, final_count))

    def update_inline_progress(self, key, current_count, terminal_text):
        self.updates.append(current_count)
        self.threads.append(threading.get_ident())

    def end_inline_progress(self, key, set_100=True):
        self.ends.append(set_100)


def worker_identity():
    return os.getpid()


def delayed_value(value, delay):
    sleep(delay)
    return value


def fail_fit():
    raise ValueError("fit is invalid")


@pytest.mark.parametrize("requested,cpus,fits,expected", [
    (-1, 24, 24, 8), (3, 24, 24, 3), (24, 2, 24, 2),
    (24, 24, 2, 2), (1, 24, 24, 1), (None, 24, 24, 8),
])
def test_worker_resolution_respects_cpu_config_fit_and_fallback_caps(monkeypatch, requested, cpus, fits, expected):
    monkeypatch.setattr(execution, "cpu_count", lambda: cpus)
    assert execution.resolve_fallback_workers(LogisticRegression(), requested, fits)[0] == expected


def test_framework_model_remains_sequential(monkeypatch):
    class TorchCheckpointClassifier(BaseEstimator):
        pass
    monkeypatch.setattr(execution, "cpu_count", lambda: 24)
    pipeline = Pipeline([("model", TorchCheckpointClassifier())])
    workers, reason = execution.resolve_fallback_workers(pipeline, -1, 24)
    assert workers == 1
    assert "checkpoint" in reason


def test_existing_checkpoint_classifier_remains_sequential(monkeypatch):
    class NNClassifier3PL(BaseEstimator):
        CHECKPOINT_DIR = 'nn_checkpoints'
    monkeypatch.setattr(execution, "cpu_count", lambda: 24)
    assert execution.resolve_fallback_workers(NNClassifier3PL(), -1, 24)[0] == 1


def test_old_joblib_uses_serial_fit_progress_without_new_dependency(monkeypatch):
    monkeypatch.setattr(execution, '_HAS_STREAMING', False)
    assert execution.resolve_fallback_workers(LogisticRegression(), -1, 24)[0] == 1
    counts = []
    values, workers = execution.run_isolated_tasks(
        [delayed(delayed_value)(1, 0), delayed(delayed_value)(2, 0)], 2,
        progress=lambda current, total: counts.append(current),
    )
    assert values == [1, 2] and workers == 1 and counts == [1, 2]


def test_parallel_fits_run_in_processes_and_callback_stays_in_parent():
    parent = os.getpid()
    callback_threads = []
    results, workers = execution.run_isolated_tasks(
        [delayed(worker_identity)() for _ in range(4)], 2,
        progress=lambda current, total: callback_threads.append(threading.get_ident()),
    )
    assert workers == 2
    assert all(pid != parent for pid in results)
    assert callback_threads == [threading.get_ident()] * 4


def test_unordered_completions_do_not_reorder_recovery_results():
    counts = []
    results, _ = execution.run_isolated_tasks(
        [delayed(delayed_value)(1, 0.15), delayed(delayed_value)(2, 0.0)], 2,
        progress=lambda current, total: counts.append((current, total)),
    )
    assert results == [1, 2]
    assert counts == [(1, 2), (2, 2)]


@pytest.mark.parametrize("error,expected_workers", [(MemoryError(), 2), (PicklingError(), 1)])
def test_resource_retry_keeps_completed_results_and_retries_only_missing(monkeypatch, error, expected_workers):
    calls = []
    original = execution.Parallel
    class FailingPool:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __call__(self, tasks):
            tasks = list(tasks)
            def partial():
                function, args, kwargs = tasks[0]
                yield function(*args, **kwargs)
                raise error
            return partial()
    first = True
    def pool(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            return FailingPool()
        # Use threads only in this retry test so calls are directly inspectable.
        return original(n_jobs=kwargs['n_jobs'], backend="threading", return_as="generator_unordered")
    monkeypatch.setattr(execution, "Parallel", pool)
    def calculate(value):
        calls.append(value)
        return value / 10
    logger = RecordingLogger()
    results, workers = execution.run_isolated_tasks(
        [delayed(calculate)(value) for value in range(4)], 4, logger=logger,
    )
    assert results == [0.0, 0.1, 0.2, 0.3]
    assert sorted(calls) == [0, 1, 2, 3]
    assert workers == expected_workers
    assert "Retained 1/4" in logger.warnings[0]


def test_model_failure_is_not_retried_as_worker_failure():
    with pytest.raises(ValueError, match="fit is invalid"):
        execution.run_isolated_tasks([delayed(fail_fit)()], 2)


def test_progress_closes_incomplete_on_interruption():
    logger = RecordingLogger()
    with pytest.raises(KeyboardInterrupt):
        with execution.FallbackProgress(logger, 5, 4, 2, "test") as progress:
            progress.start_clone(0)
            progress.fit_completed(1, 4)
            raise KeyboardInterrupt
    assert logger.ends == [False]
    assert logger.updates == [1]


def test_parallel_and_sequential_fallback_have_same_seeded_experiment():
    X, y = make_classification(n_samples=180, n_features=8, n_informative=5,
                               class_sep=1.5, random_state=123)
    X = sparse.csr_matrix(X)
    model = LogisticRegression(C=3, max_iter=500, random_state=12, n_jobs=3)
    kwargs = dict(flip_fraction=0.2, n_splits=2, n_repeats=2, random_state=42, positive_class=1)
    serial = correction.estimate_perturbed_same_model_correction(model, X, y, n_jobs=1, **kwargs)
    logger = RecordingLogger()
    parallel = correction.estimate_perturbed_same_model_correction(model, X, y, n_jobs=2, logger=logger, **kwargs)
    assert serial[:2] == parallel[:2]
    for field in ('clone_results', 'clone_corrections', 'accepted', 'reason', 'valid_clone_count'):
        assert serial[2][field] == parallel[2][field]
    assert parallel[2]['planned_fits'] == parallel[2]['completed_fits'] == 20
    assert parallel[2]['skipped_fits'] == 0
    assert logger.updates[-1] == 20
    assert all(a <= b for a, b in zip(logger.updates, logger.updates[1:]))
    assert logger.ends == [True]
    assert logger.threads == [threading.get_ident()] * len(logger.threads)
    assert model.n_jobs == 3 and model.random_state == 12
    assert not hasattr(model, 'coef_')


def test_sparse_tsvd_voting_pipeline_supports_parallel_fallback():
    X, y = make_classification(n_samples=120, n_features=10, n_informative=6,
                               class_sep=2.5, random_state=100)
    pipeline = Pipeline([
        ('MAX', MaxAbsScaler()), ('TSVD', TruncatedSVD(n_components=6)),
        ('FUTV', VotingClassifier([
            ('mlpc', MLPClassifier(hidden_layer_sizes=(5,), max_iter=60, tol=0.05)),
            ('rfcl', RandomForestClassifier(n_estimators=5)),
            ('abc', AdaBoostClassifier(n_estimators=5)),
        ], voting='soft')),
    ])
    result = correction.estimate_perturbed_same_model_correction(
        pipeline, sparse.csr_matrix(X), y, flip_fraction=0.2, n_splits=2,
        n_repeats=1, random_state=42, positive_class=1, n_jobs=2,
    )
    assert len(result[2]['clone_results']) == 5
    assert result[2]['completed_fits'] == 10
    assert result[2]['skipped_fits'] == 0
    assert not hasattr(pipeline.named_steps['TSVD'], 'components_')


def test_zero_recovery_keeps_all_five_clones_and_rejects_without_skipping():
    X = np.ones((100, 2))
    y = np.array([0, 1] * 50)
    logger = RecordingLogger()
    corr, source, details = correction.estimate_perturbed_same_model_correction(
        DummyClassifier(strategy='constant', constant=0), X, y,
        flip_fraction=0.2, n_splits=2, n_repeats=1, random_state=42,
        positive_class=1, n_jobs=1, logger=logger,
    )
    assert corr == 1.0 and source == 'fallback/unestimable'
    assert details['valid_clone_count'] == 0
    assert len(details['clone_results']) == 5
    assert details['completed_fits'] == 10 and details['skipped_fits'] == 0
    assert logger.ends == [True]  # Completed work does not imply accepted correction.


def test_failed_clone_accounts_for_skipped_work_without_faking_fit_completions(monkeypatch):
    def fail(self, X, y):
        raise ValueError('fit failed before any results')
    monkeypatch.setattr(correction.DarkNumberCorrectionFactorEstimator, 'fit', fail)
    logger = RecordingLogger()
    _, source, details = correction.estimate_perturbed_same_model_correction(
        LogisticRegression(), np.ones((100, 2)), np.array([0, 1] * 50),
        flip_fraction=0.2, n_splits=2, n_repeats=1, random_state=42,
        positive_class=1, logger=logger,
    )
    assert source == 'fallback/unestimable'
    assert details['completed_fits'] == 0 and details['skipped_fits'] == 10
    assert logger.updates == [2, 4, 6, 8, 10]
    assert logger.ends == [True]
    assert 'completed=0, skipped=10' in logger.info[-1]


@pytest.mark.parametrize("values,accepted,reason", [
    ([2, 2.1, 2.2, 2.3, 2.4], True, 'accepted'),
    ([1, 1, 1, 10, 10], False, 'unstable_correction_factors'),
    ([2, 2.1, np.inf, np.inf, np.inf], False, 'insufficient_valid_clones'),
    ([2, 2.1, 2.2, np.inf, np.inf], True, 'accepted'),
])
def test_existing_clone_acceptance_guards_and_seed_schedule_are_preserved(monkeypatch, values, accepted, reason):
    seen = []
    class ControlledEstimator:
        def __init__(self, *, estimator, random_state, n_jobs, progress, **kwargs):
            self.value = values[len(seen)]
            self.n_jobs = n_jobs
            self.progress = progress
            self.is_valid_for_regression_ = np.isfinite(self.value)
            self.correction_status_ = 'estimated' if self.is_valid_for_regression_ else 'zero_recovery'
            seen.append((random_state, estimator.random_state, estimator.C, kwargs))
        def fit(self, X, y):
            assert dict(zip(*np.unique(y, return_counts=True))) == {0: 50, 1: 50}
            self.progress(1, 2)
            self.progress(2, 2)
        def score(self):
            return self.value
    monkeypatch.setattr(correction, 'DarkNumberCorrectionFactorEstimator', ControlledEstimator)
    _, _, details = correction.estimate_perturbed_same_model_correction(
        LogisticRegression(C=7), np.ones((100, 2)), np.array([0, 1] * 50),
        flip_fraction=0.2, n_splits=2, n_repeats=1, random_state=42, positive_class=1,
    )
    assert len(seen) == 5
    assert [item[0] for item in seen] == [42 + 1009 * i for i in range(1, 6)]
    assert all(seed == model_seed and C == 7 for seed, model_seed, C, _ in seen)
    assert all(kwargs['predict_mode'] == 'predict' and kwargs['sample_size'] == 1 for *_, kwargs in seen)
    assert details['accepted'] is accepted and details['reason'] == reason
    assert details['completed_fits'] == 10 and details['skipped_fits'] == 0


@pytest.mark.parametrize('kwargs', [
    {'perturbation_clones': 0}, {'perturbation_min_valid': 6},
    {'perturbation_max_cv': -1}, {'n_repeats': 0},
])
def test_invalid_work_is_rejected_before_starting_progress(kwargs):
    logger = RecordingLogger()
    args = dict(flip_fraction=0.2, n_splits=2, n_repeats=1, random_state=42, positive_class=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        correction.estimate_perturbed_same_model_correction(
            LogisticRegression(), np.ones((100, 2)), np.array([0, 1] * 50), logger=logger, **args,
        )
    assert logger.starts == []
