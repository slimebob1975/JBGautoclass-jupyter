"""285m3: bounded structured feature importance without row-level leakage."""
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from JBGApplicationContracts import DataSourceKind, RunStatus
from JBGRunJobStore import RunJobStore
from JBGRunOverview import project_run_overview
from JBGRunWorker import StoreProgressLogger
from JBGStructuredFeatureImportance import (
    feature_importance_from_report, sanitize_stored_feature_importance,
)


def report():
    return SimpleNamespace(
        table=pd.DataFrame({
            "Rank": [1, 2], "Feature": ["petal length", "petal width"],
            "Mean score decrease": [.15, -.04], "Std. across repeats": [.03, .02],
            "Repeat 1": [.12, -.02], "Repeat 2": [.18, -.06],
        }), baseline_score=.95, rows=30, repeats=2, seed=42, seconds=.5)


def test_report_is_aggregate_and_drops_repeat_level_data():
    value = feature_importance_from_report(report())
    assert value == {
        "schema_version": 1, "method": "permutation", "baseline_score": .95,
        "holdout_rows": 30, "repeats": 2, "total_features": 2,
        "features": [
            {"rank": 1, "name": "petal length", "mean_score_decrease": .15,
             "std_across_repeats": .03},
            {"rank": 2, "name": "petal width", "mean_score_decrease": -.04,
             "std_across_repeats": .02},
        ],
    }
    assert "Repeat" not in str(value)
    assert "seed" not in str(value)
    assert "seconds" not in str(value)


def test_stored_allowlist_strips_unknown_values_and_rejects_invalid():
    value = feature_importance_from_report(report())
    assert sanitize_stored_feature_importance({**value, "secret": "private"}) == value
    malformed = {**value, "features": [{**value["features"][0], "name": "secret\nline"},
                                        value["features"][1]]}
    assert sanitize_stored_feature_importance(malformed) is None
    malformed = {**value, "features": [{**value["features"][0],
                                        "mean_score_decrease": float("nan")},
                                        value["features"][1]]}
    assert sanitize_stored_feature_importance(malformed) is None
    malformed = {**value, "features": [value["features"][0]] * 2}
    assert sanitize_stored_feature_importance(malformed) is None
    assert sanitize_stored_feature_importance({**value, "total_features": 10000}) is None


def test_worker_captures_and_overview_exposes_for_success_only(tmp_path):
    store = RunJobStore.sqlite(tmp_path / "runs.db")
    store.create_tables()
    run = store.create("alice", DataSourceKind.SQL, {"spec_version": 1})
    assert store.claim(run, "worker-1")
    observer = StoreProgressLogger(Mock(), store, run, SimpleNamespace(lost=False))
    observer.record_feature_importance(report())
    assert observer.feature_importance["features"][0]["name"] == "petal length"
    assert store.finish(run, "worker-1", RunStatus.SUCCEEDED,
                        result={"kind": "classification", "feature_importance": observer.feature_importance})
    record = store.get(run)
    result = project_run_overview(run.key, record.status, store.progress(run), record.result)
    assert result["results"]["feature_importance"] == observer.feature_importance
    assert "feature_importance" not in project_run_overview(
        run.key, RunStatus.FAILED, [], record.result)["results"]
    assert "feature_importance" not in project_run_overview(
        run.key, RunStatus.SUCCEEDED, [], {"kind": "classification"})["results"]
    store.close()


def test_invalid_computation_never_creates_payload():
    obj = report()
    obj.table.loc[0, "Mean score decrease"] = float("inf")
    assert feature_importance_from_report(obj) is None
    obj = report()
    obj.table.loc[0, "Feature"] = "duplicate"
    obj.table.loc[1, "Feature"] = "duplicate"
    assert feature_importance_from_report(obj) is None
