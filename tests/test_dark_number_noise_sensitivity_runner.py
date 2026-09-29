from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

import JBGDarkNumberNoiseSensitivityRunner as runner


class _Config:
    def __init__(self, target="Ja", fallback=True, script_path=None):
        self._target = target
        self._fallback = fallback
        self.script_path = script_path
        self.io = SimpleNamespace(model_name="noise sensitivity model")

    def get_dark_number_target(self):
        return self._target

    def get_test_size(self):
        return 0.2

    def should_use_experimental_perturbed_dark_number_fallback(self):
        return self._fallback

    def get_dark_number_calculation_type(self):
        return "base"


class _Logger:
    def print_info(self, *args, **kwargs):
        return None

    def print_warning(self, *args, **kwargs):
        return None


def test_defaults_cover_requested_flip_fraction_grid_and_nine_seeds():
    args = runner.build_argument_parser().parse_args([])
    assert runner._validated_fractions(args.fraction) == [0.05, 0.10, 0.15, 0.20]
    assert args.runs == 9
    assert args.seed == 42
    assert args.split_seed == 42


def test_model_identity_records_pipeline_steps_and_final_estimator():
    pipeline = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    identity = runner.describe_model_pipeline(pipeline)

    assert identity["type"] == "Pipeline"
    assert identity["steps"][0]["name"] == "clf"
    assert identity["steps"][0]["type"] == "DummyClassifier"
    assert identity["final_estimator_type"] == "DummyClassifier"
    formatted = runner.format_model_identity(identity)
    assert "clf=DummyClassifier" in formatted
    assert "strategy='most_frequent'" in formatted


def test_resolve_targets_prefers_configured_dark_number_target():
    y = np.array(["Ja", "Nej", "Nej"], dtype=object)
    assert runner.resolve_targets(_Config(target="Ja"), y, None) == ["Ja"]
    assert runner.resolve_targets(_Config(target=""), y, None) == ["Ja", "Nej"]
    assert runner.resolve_targets(_Config(target="Ja"), y, ["Nej"]) == ["Nej"]
    with pytest.raises(ValueError, match="not present"):
        runner.resolve_targets(_Config(target="Ja"), y, ["Missing"])


def test_recovery_count_diagnostics_reconstructs_hard_counts():
    y = np.array(["Ja"] * 50 + ["Nej"] * 50, dtype=object)
    diagnostics = runner.recovery_count_diagnostics(
        y,
        positive_class="Ja",
        flip_fraction=0.20,
        n_splits=5,
        n_repeats=2,
        random_state=42,
        recovery_results=[0.5] * 10,
    )

    # Each 5-fold training partition contains 40 positives: 20% => 8 flipped.
    assert diagnostics["correction_fits"] == 10
    assert diagnostics["mean_flipped_per_fit"] == pytest.approx(8.0)
    assert diagnostics["total_flipped"] == 80
    assert diagnostics["total_recovered"] == 40
    assert diagnostics["pooled_recovery"] == pytest.approx(0.5)


def test_recovery_count_diagnostics_keeps_planned_flip_counts_when_preflight_skips():
    y = np.array(["Ja"] * 55 + ["Nej"] * 45, dtype=object)
    diagnostics = runner.recovery_count_diagnostics(
        y,
        positive_class="Ja",
        flip_fraction=0.05,
        n_splits=5,
        n_repeats=2,
        random_state=42,
        recovery_results=None,
    )

    assert diagnostics["correction_fits"] == 10
    assert diagnostics["mean_flipped_per_fit"] == pytest.approx(2.0)
    assert diagnostics["total_flipped"] == 20
    assert np.isnan(diagnostics["total_recovered"])
    assert np.isnan(diagnostics["pooled_recovery"])


def test_summary_excludes_unestimable_sentinel_from_corr_and_dark_number_stability():
    runs = pd.DataFrame(
        {
            "target": ["Ja", "Ja", "Ja"],
            "flip_fraction": [0.2, 0.2, 0.2],
            "mean_flipped_per_fit": [10.0, 10.0, 10.0],
            "total_flipped": [100, 100, 100],
            "total_recovered": [20, np.nan, np.nan],
            "pooled_recovery": [0.2, np.nan, np.nan],
            "direct_status": ["estimated", "zero_recovery", "insufficient_sample"],
            "corr_source": ["direct", "perturbed_same_model", "fallback/unestimable"],
            "fallback_attempted": [False, True, False],
            "low_recovery_warning": [False, False, False],
            "direct_valid": [True, False, False],
            "direct_mean_recovery": [0.2, np.nan, np.nan],
            "corr": [5.0, 6.0, 1.0],
            "d_cv_full_estimable": [0.10, 0.12, np.nan],
        }
    )

    summary = runner.summarize_sensitivity(runs).iloc[0]
    assert summary["direct_rate"] == pytest.approx(1 / 3)
    assert summary["fallback_activation_rate"] == pytest.approx(1 / 3)
    assert summary["perturbed_accept_rate"] == pytest.approx(1 / 3)
    assert summary["estimable_rate"] == pytest.approx(2 / 3)
    assert summary["corr_mean"] == pytest.approx(5.5)
    assert summary["d_cv_full_mean"] == pytest.approx(0.11)


def test_dataset_and_split_fingerprints_are_stable_and_sensitive():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = np.array(["Ja", "Nej"], dtype=object)
    first = runner.dataset_fingerprint(X, y)
    second = runner.dataset_fingerprint(X.copy(), y.copy())
    changed = runner.dataset_fingerprint(X.assign(a=[1.0, 9.0]), y)
    assert first == second
    assert first != changed

    split_a = runner.split_fingerprint(np.array([0, 2]), np.array([1, 3]))
    split_b = runner.split_fingerprint(np.array([0, 2]), np.array([1, 3]))
    split_c = runner.split_fingerprint(np.array([0, 3]), np.array([1, 2]))
    assert split_a == split_b
    assert split_a != split_c


def test_save_outputs_writes_detail_summary_and_metadata(tmp_path):
    config = _Config(script_path=tmp_path)
    detailed = pd.DataFrame({"target": ["Ja"], "flip_fraction": [0.2]})
    summary = pd.DataFrame({"target": ["Ja"], "flip_fraction": [0.2], "corr_mean": [5.0]})
    metadata = {"dataset_fingerprint": "abc", "split_fingerprint": "def"}

    detail_path, summary_path, metadata_path = runner.save_outputs(
        config, detailed, summary, metadata
    )

    assert detail_path.exists()
    assert summary_path.exists()
    assert metadata_path.exists()
    assert "dark_number_noise_sensitivity_noise_sensitivity_model_" in detail_path.name
