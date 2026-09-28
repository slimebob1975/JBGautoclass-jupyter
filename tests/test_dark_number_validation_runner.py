from types import SimpleNamespace

import pandas as pd
import pytest
from scipy import sparse as scipy_sparse

import JBGDarkNumberValidationRunner as runner


class _FakeConfig:
    def __init__(self, script_path=None):
        self.script_path = script_path
        self.io = SimpleNamespace(model_name="perturbed validation model")

    def get_dark_number_flip_fraction(self):
        return 0.2

    def get_test_size(self):
        return 0.1

    def get_data_limit(self):
        return None


class _FakeHarness:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def compare_perturbed_same_model_fallback(self, X, y, n_runs):
        target = self.kwargs["positive_class"]
        paired = pd.DataFrame(
            {
                "random_state": [42],
                "true_dark_number": [0.25],
                "baseline_corr_cv_source": ["fallback/unestimable"],
                "candidate_corr_cv_source": ["perturbed_same_model"],
                "baseline_corr_retrained_source": ["fallback/unestimable"],
                "candidate_corr_retrained_source": ["perturbed_same_model"],
                "baseline_interval_covered": [False],
                "candidate_interval_covered": [True],
                "baseline_d_cv_test": [0.0],
                "candidate_d_cv_test": [0.2],
                "baseline_abs_error_d_cv_test": [0.25],
                "candidate_abs_error_d_cv_test": [0.05],
                "baseline_d_cv_full": [0.0],
                "candidate_d_cv_full": [0.2],
                "baseline_abs_error_d_cv_full": [0.25],
                "candidate_abs_error_d_cv_full": [0.05],
                "baseline_d_retrained_full": [0.0],
                "candidate_d_retrained_full": [0.2],
                "baseline_abs_error_d_retrained_full": [0.25],
                "candidate_abs_error_d_retrained_full": [0.05],
            }
        )
        summary = {
            "n_runs": n_runs,
            "baseline": {"coverage_rate": 0.0},
            "perturbed_same_model": {"coverage_rate": 1.0},
            "baseline_mean_absolute_error": {
                "d_cv_test": 0.25,
                "d_cv_full": 0.25,
                "d_retrained_full": 0.25,
            },
            "perturbed_same_model_mean_absolute_error": {
                "d_cv_test": 0.05,
                "d_cv_full": 0.05,
                "d_retrained_full": 0.05,
            },
            "mean_absolute_error_delta": {
                "d_cv_test": -0.20,
                "d_cv_full": -0.20,
                "d_retrained_full": -0.20,
            },
            "coverage_rate_delta": 1.0,
            "estimability_rate_delta": 1.0,
            "interval_coverage_rate_delta": 1.0,
            "perturbed_same_model_activation_rate": 1.0,
            "baseline_unestimable_rate": 1.0,
            "candidate_unestimable_rate": 0.0,
            "perturbed_estimable_error_profile": {
                name: {
                    "estimable_count": 1,
                    "total_count": 1,
                    "estimable_rate": 1.0,
                    "mean_absolute_error": 0.05,
                    "median_absolute_error": 0.05,
                    "p90_absolute_error": 0.05,
                    "mean_signed_error": -0.05,
                    "signed_error_std": 0.0,
                }
                for name in ("d_cv_test", "d_cv_full", "d_retrained_full")
            },
            "perturbed_correction_stability": {
                name: {
                    "accepted_count": 1,
                    "total_count": 1,
                    "activation_rate": 1.0,
                    "mean": 3.0,
                    "median": 3.0,
                    "std": 0.0,
                    "coefficient_of_variation": 0.0,
                    "iqr": 0.0,
                    "min": 3.0,
                    "max": 3.0,
                }
                for name in ("corr_cv", "corr_retrained")
            },
            "target_seen_by_fake": target,
        }
        return paired, summary


class _NullLogger:
    def print_info(self, *args, **kwargs):
        return None


def test_validation_input_fetch_accepts_nonempty_numpy_array(monkeypatch):
    import numpy as np

    class _DatasetHandler:
        def __init__(self):
            self.X = pd.DataFrame({"x": [1.0, 2.0]})
            self.Y = pd.Series(["incident", "request"])

        def load_data(self, data):
            assert isinstance(data, np.ndarray)
            assert len(data) == 2

        def separate_dataset(self):
            return None

    dataset_handler = _DatasetHandler()

    class _Handler:
        def __init__(self, **kwargs):
            pass

        def add_handler(self, name):
            assert name == "dataset"
            return dataset_handler

        def get_dataset(self):
            return np.array([[1.0, "incident"], [2.0, "request"]], dtype=object)

    class _Converter:
        def transform(self, X):
            return X

    class _Pipeline:
        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    monkeypatch.setattr(runner, "DataLayer", lambda config, logger: object())
    monkeypatch.setattr(runner, "JBGHandler", _Handler)
    monkeypatch.setattr(
        runner,
        "load_model_artifact",
        lambda model_path: (None, _Converter(), None, _Pipeline(), None, 1),
    )

    X, y, pipeline = runner.load_validation_inputs(
        _FakeConfig(),
        _NullLogger(),
        model_path="unused.sav",
    )

    assert list(X["x"]) == [1.0, 2.0]
    assert list(y) == ["incident", "request"]
    assert hasattr(pipeline, "predict_proba")




def test_validation_input_converts_sparse_dataframe_to_csr(monkeypatch):
    import numpy as np

    class _DatasetHandler:
        def __init__(self):
            self.X = pd.DataFrame({"text": ["a", "b"]})
            self.Y = pd.Series(["incident", "request"])
        def load_data(self, data):
            return None
        def separate_dataset(self):
            return None

    dataset_handler = _DatasetHandler()

    class _Handler:
        def __init__(self, **kwargs):
            pass
        def add_handler(self, name):
            return dataset_handler
        def get_dataset(self):
            return np.array([["a", "incident"], ["b", "request"]], dtype=object)

    class _Converter:
        def transform(self, X):
            return pd.DataFrame({
                "f1": pd.arrays.SparseArray([1.0, 0.0], fill_value=0.0),
                "f2": pd.arrays.SparseArray([0.0, 2.0], fill_value=0.0),
            })

    class _Pipeline:
        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (X.shape[0], 1))

    monkeypatch.setattr(runner, "DataLayer", lambda config, logger: object())
    monkeypatch.setattr(runner, "JBGHandler", _Handler)
    monkeypatch.setattr(
        runner, "load_model_artifact",
        lambda model_path: (None, _Converter(), None, _Pipeline(), None, 2),
    )

    X, y, _ = runner.load_validation_inputs(_FakeConfig(), _NullLogger(), "unused.sav")

    assert scipy_sparse.isspmatrix_csr(X)
    assert X.shape == (2, 2)
    assert list(y) == ["incident", "request"]

def test_real_dataset_runner_validates_all_observed_targets_with_same_model(monkeypatch):
    _FakeHarness.calls = []
    monkeypatch.setattr(runner, "DarkNumberValidationHarness", _FakeHarness)

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series(["incident", "request", "incident", "request"])
    paired, summaries = runner.run_real_dataset_comparison(
        _FakeConfig(),
        _NullLogger(),
        X,
        y,
        pipeline=object(),
        targets=None,
        n_runs=1,
        random_state=42,
        injected_noise_fraction=0.2,
        correction_n_splits=3,
        correction_n_repeats=1,
    )

    assert set(paired["target"]) == {"incident", "request"}
    assert set(summaries) == {"incident", "request"}
    assert [call["positive_class"] for call in _FakeHarness.calls] == ["incident", "request"]
    assert all(call["estimator"] is _FakeHarness.calls[0]["estimator"] for call in _FakeHarness.calls)
    assert all(call["correction_flip_fraction"] == pytest.approx(0.2) for call in _FakeHarness.calls)


def test_real_dataset_runner_rejects_unknown_target(monkeypatch):
    monkeypatch.setattr(runner, "DarkNumberValidationHarness", _FakeHarness)
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.Series(["incident", "request"])

    with pytest.raises(ValueError, match="not present"):
        runner.run_real_dataset_comparison(
            _FakeConfig(),
            _NullLogger(),
            X,
            y,
            pipeline=object(),
            targets=["missing"],
            n_runs=1,
            random_state=42,
            injected_noise_fraction=0.2,
            correction_n_splits=3,
            correction_n_repeats=1,
        )


def test_validation_outputs_are_written_next_to_existing_csv_outputs(tmp_path):
    config = _FakeConfig(script_path=tmp_path)
    paired = pd.DataFrame({"target": ["incident"], "true_dark_number": [0.25]})
    summaries = {"incident": {"perturbed_same_model_activation_rate": 1.0}}

    csv_path, json_path = runner.save_validation_outputs(config, paired, summaries)

    assert csv_path.parent == tmp_path / "output" / "csvs"
    assert json_path.parent == csv_path.parent
    assert "dark_number_perturbed_same_model_robustness_perturbed_validation_model_" in csv_path.name
    assert csv_path.exists()
    assert json_path.exists()


def test_robustness_runner_defaults_to_nine_paired_seeds():
    args = runner.build_argument_parser().parse_args([])
    assert args.runs == 9
