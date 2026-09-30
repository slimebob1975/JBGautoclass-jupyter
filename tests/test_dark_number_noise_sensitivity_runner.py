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
    assert args.checkpoint is None
    assert runner.build_argument_parser().parse_args(["--checkpoint", "study.json"]).checkpoint.name == "study.json"


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


def _small_experiment(tmp_path, **overrides):
    settings = dict(
        targets=["Ja"], fractions=[0.2, 0.3], runs=2, seed=42, split_seed=42,
        correction_n_splits=2, correction_n_repeats=1,
        perturbation_clones=3, perturbation_min_valid=2, perturbation_max_cv=0.5,
        checkpoint_path=tmp_path / "study.json", source_artifact_fingerprint="saved-model-abc",
    )
    settings.update(overrides)
    X = np.arange(240, dtype=float).reshape(120, 2)
    y = np.array(["Ja", "Nej"] * 60)
    return _Config(fallback=False, script_path=tmp_path), X, y, settings


def test_interrupted_cells_resume_without_refit_or_duplicates(tmp_path, monkeypatch):
    import json
    config, X, y, settings = _small_experiment(tmp_path)
    pipeline = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    actual_cell = runner.run_sensitivity_cell
    calls = []

    def interrupted(*args, **kwargs):
        calls.append((kwargs["flip_fraction"], kwargs["correction_seed"]))
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated logout")
        return actual_cell(*args, **kwargs)

    monkeypatch.setattr(runner, "run_sensitivity_cell", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run_noise_sensitivity(config, _Logger(), X, y, pipeline, **settings)
    saved = json.loads(settings["checkpoint_path"].read_text())
    assert saved["state"] == "running"
    assert saved["completed_cells"] == 1
    assert saved["metadata"]["fixed_model_fingerprint"]
    assert saved["rows"][0]["target"] == "Ja"

    monkeypatch.setattr(runner, "run_sensitivity_cell", actual_cell)
    expected, expected_summary, _ = runner.run_noise_sensitivity(
        config, _Logger(), X, y, pipeline, **dict(settings, checkpoint_path=None)
    )

    def forbid_refit(*args, **kwargs):
        raise AssertionError("The original fixed model must be restored, not refitted.")

    monkeypatch.setattr(runner, "fit_fixed_cross_trained_model", forbid_refit)
    resumed_calls = []

    def resumed_cell(*args, **kwargs):
        resumed_calls.append((kwargs["flip_fraction"], kwargs["correction_seed"]))
        return actual_cell(*args, **kwargs)

    monkeypatch.setattr(runner, "run_sensitivity_cell", resumed_cell)
    detailed, summary, metadata = runner.run_noise_sensitivity(config, _Logger(), X, y, pipeline, **settings)
    assert resumed_calls == [(0.2, 43), (0.3, 42), (0.3, 43)]
    pd.testing.assert_frame_equal(detailed, expected)
    pd.testing.assert_frame_equal(summary, expected_summary)
    assert metadata["completed_cells"] == metadata["total_cells"] == 4
    assert json.loads(settings["checkpoint_path"].read_text())["state"] == "complete"

    # A completed checkpoint also rebuilds final outputs without running any cells.
    monkeypatch.setattr(runner, "run_sensitivity_cell", forbid_refit)
    replay, replay_summary, _ = runner.run_noise_sensitivity(config, _Logger(), X, y, pipeline, **settings)
    pd.testing.assert_frame_equal(replay, expected)
    pd.testing.assert_frame_equal(replay_summary, expected_summary)


@pytest.mark.parametrize("change", [
    "data", "feature_names", "split", "model", "artifact", "fractions", "target",
    "seed", "runs", "correction_splits", "correction_repeats", "fallback",
    "formula", "clones", "min_valid", "max_cv", "runtime",
])
def test_checkpoint_refuses_changed_experiment_before_training(tmp_path, monkeypatch, change):
    config, X, y, settings = _small_experiment(tmp_path)
    X = pd.DataFrame(X, columns=["a", "b"])
    pipeline = DummyClassifier(strategy="most_frequent")
    runner.run_noise_sensitivity(config, _Logger(), X, y, pipeline, **settings)
    original = settings["checkpoint_path"].read_bytes()
    if change == "data":
        X.iloc[0, 0] += 1
    elif change == "feature_names":
        X.columns = ["different", "b"]
    elif change == "split":
        settings["split_seed"] += 1
    elif change == "model":
        pipeline = DummyClassifier(strategy="prior")
    elif change == "artifact":
        settings["source_artifact_fingerprint"] = "different-artifact"
    elif change == "fractions":
        settings["fractions"] = [0.2, 0.4]
    elif change == "target":
        settings["targets"] = ["Nej"]
    elif change == "seed":
        settings["seed"] += 1
    elif change == "runs":
        settings["runs"] += 1
    elif change == "correction_splits":
        settings["correction_n_splits"] += 1
    elif change == "correction_repeats":
        settings["correction_n_repeats"] += 1
    elif change == "fallback":
        config._fallback = True
    elif change == "formula":
        monkeypatch.setattr(config, "get_dark_number_calculation_type", lambda: "single_alpha")
    elif change == "clones":
        settings["perturbation_clones"] += 1
    elif change == "min_valid":
        settings["perturbation_min_valid"] += 1
    elif change == "max_cv":
        settings["perturbation_max_cv"] = 0.4
    elif change == "runtime":
        monkeypatch.setattr(runner, "runtime_identity", lambda: {"code": "changed"})

    def unexpected(*args, **kwargs):
        raise AssertionError("Mismatch must be detected before training/correction.")

    monkeypatch.setattr(runner, "fit_fixed_cross_trained_model", unexpected)
    monkeypatch.setattr(runner, "run_sensitivity_cell", unexpected)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        runner.run_noise_sensitivity(config, _Logger(), X, y, pipeline, **settings)
    assert settings["checkpoint_path"].read_bytes() == original


def test_initial_checkpoint_survives_failure_during_fixed_model_fit(tmp_path, monkeypatch):
    import json
    config, X, y, settings = _small_experiment(tmp_path)
    actual_fit = runner.fit_fixed_cross_trained_model

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("before first cell")

    monkeypatch.setattr(runner, "fit_fixed_cross_trained_model", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run_noise_sensitivity(config, _Logger(), X, y, DummyClassifier(), **settings)
    saved = json.loads(settings["checkpoint_path"].read_text())
    assert saved["state"] == "prepared"
    assert saved["completed_cells"] == 0
    assert saved["experiment"]["dataset_fingerprint"]
    assert saved["metadata"]["source_model_identity"]
    monkeypatch.setattr(runner, "fit_fixed_cross_trained_model", actual_fit)
    detailed, _, _ = runner.run_noise_sensitivity(config, _Logger(), X, y, DummyClassifier(), **settings)
    assert len(detailed) == 4


def test_duplicate_grid_is_rejected_before_checkpoint_creation(tmp_path):
    config, X, y, settings = _small_experiment(tmp_path, fractions=[0.2, 0.2])
    with pytest.raises(ValueError, match="duplicate"):
        runner.run_noise_sensitivity(config, _Logger(), X, y, DummyClassifier(), **settings)
    assert not settings["checkpoint_path"].exists()


@pytest.mark.parametrize("explicit_checkpoint", [False, True])
def test_cli_automatically_creates_and_resumes_checkpoint(tmp_path, monkeypatch, explicit_checkpoint):
    from contextlib import nullcontext
    import json

    config, X, y, _ = _small_experiment(tmp_path)
    model_path = tmp_path / "model.sav"
    model_path.write_bytes(b"source-model-fixture")
    config.get_model_filename = lambda: str(model_path)
    monkeypatch.setattr(runner, "load_last_run_snapshot", lambda path: {})
    monkeypatch.setattr(runner, "resolve_sql_credentials", lambda snapshot, user: ("", ""))
    monkeypatch.setattr(runner, "config_from_last_run_snapshot", lambda *args: config)
    monkeypatch.setattr(runner, "load_validation_inputs", lambda *args: (X, y, DummyClassifier()))

    class Logger(_Logger):
        def __init__(self, **kwargs):
            pass

        def get_log_filename(self):
            return "fixture.log"

        def capture_console_output(self):
            return nullcontext()

        def display_matrix(self, *args, **kwargs):
            pass

    monkeypatch.setattr(runner, "JBGLogger", Logger)
    args = ["--runs", "1", "--fraction", "0.2", "--correction-splits", "2", "--correction-repeats", "1"]
    expected = tmp_path / "output" / "csvs" / "dark_number_noise_sensitivity_noise_sensitivity_model_checkpoint.json"
    if explicit_checkpoint:
        expected = tmp_path / "explicit.json"
        args += ["--checkpoint", str(expected)]
    assert runner.main(args) == 0
    saved = json.loads(expected.read_text())
    assert saved["completed_cells"] == saved["total_cells"] == 1
    assert saved["experiment"]["source_artifact_fingerprint"] == runner.file_fingerprint(model_path)
    assert expected.with_name(expected.name + ".model.joblib").exists()
    assert expected.with_name(expected.name + ".inputs.joblib").exists()
    assert expected.with_name(expected.name + ".inputs.json").exists()

    def unexpected(*args, **kwargs):
        raise AssertionError("Completed CLI invocation must regenerate outputs without model refit/cells.")

    monkeypatch.setattr(runner, "fit_fixed_cross_trained_model", unexpected)
    monkeypatch.setattr(runner, "run_sensitivity_cell", unexpected)
    monkeypatch.setattr(runner, "load_validation_inputs", unexpected)
    assert runner.main(args) == 0
    assert list((tmp_path / "output" / "csvs").glob("*_summary.csv"))
