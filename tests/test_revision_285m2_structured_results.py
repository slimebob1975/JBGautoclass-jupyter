"""285m2: exact task stages and label-free evaluation persistence."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from JBGApplicationContracts import DataSourceKind, RunStatus
from JBGRunJobStore import RunJobStore
from JBGRunOverview import project_run_overview
from JBGRunWorker import StoreProgressLogger
from JBGStructuredEvaluation import evaluation_from_report, sanitize_stored_evaluation


def _report():
    return dict(accuracy_score=.75, confusion_matrix=[[2, 1], [0, 1]],
                class_labels=["PRIVATE_A", "PRIVATE_B"],
                classification_matrix={"macro avg": {"precision": .8, "recall": .75, "f1-score": .73},
                                       "weighted avg": {"precision": .82, "recall": .75, "f1-score": .74},
                                       "PRIVATE_A": {"private": "should not appear"}},
                sample_rates=None)


def test_structured_evaluation_excludes_labels_and_arbitrary_values():
    dto = evaluation_from_report(**_report())
    assert dto["accuracy"] == .75
    assert dto["confusion_matrix"] == [[2, 1], [0, 1]]
    assert dto["precision_macro"] == .8
    assert dto["mcc"] is not None
    assert "PRIVATE" not in str(dto)
    assert "should not appear" not in str(dto)
    assert sanitize_stored_evaluation({**dto, "private": "SECRET"}) == dto


def test_invalid_matrix_or_metric_rejected():
    report = _report()
    report["confusion_matrix"] = [[2, -1], [0, 1]]
    assert evaluation_from_report(**report) is None
    report = _report()
    report["accuracy_score"] = float("nan")
    assert evaluation_from_report(**report) is None
    dto = evaluation_from_report(**_report())
    assert sanitize_stored_evaluation({**dto, "confusion_matrix": [[1], [2]]}) is None


def test_worker_logger_captures_exact_stage_and_evaluation(tmp_path):
    store = RunJobStore.sqlite(tmp_path / "runs.db")
    store.create_tables()
    run = store.create("alice", DataSourceKind.SQL, {"spec_version": 1})
    assert store.claim(run, "worker-1")
    logger = Mock()
    heartbeat = SimpleNamespace(lost=False)
    observer = StoreProgressLogger(logger, store, run, heartbeat)
    observer.record_task_stage("train_model")
    observer.record_task_stage("arbitrary SQL text")
    observer.print_prediction_report(**_report())
    progress = store.progress(run)
    assert len(progress) == 1
    assert progress[0].message == "jbg_stage:train_model"
    assert observer.evaluation["accuracy"] == .75
    assert store.finish(run, "worker-1", RunStatus.SUCCEEDED,
                        result={"kind": "classification", "evaluation": observer.evaluation})
    overview = project_run_overview(run.key, store.get(run).status,
                                    store.progress(run), store.get(run).result)
    assert overview["results"]["evaluation"]["confusion_matrix"] == [[2, 1], [0, 1]]
    assert "PRIVATE" not in str(overview)
    store.close()


def test_exact_stage_only_when_running_and_safe_marker():
    stage = SimpleNamespace(percent=.4, message="jbg_stage:feature_importance")
    overview = project_run_overview("id", RunStatus.RUNNING, [stage])
    assert overview["stage"] == {"name": "feature_importance", "precision": "exact", "percent": .4}
    spoofed = SimpleNamespace(percent=.4, message="jbg_stage:secret")
    assert project_run_overview("id", RunStatus.RUNNING, [spoofed])["stage"]["precision"] == "estimated"
    assert project_run_overview("id", RunStatus.FAILED, [stage])["stage"]["name"] == "failed"


def test_legacy_result_still_valid_and_no_made_up_metrics():
    legacy = project_run_overview("id", RunStatus.SUCCEEDED, [],
                                  {"kind": "classification", "class_count": 3})
    assert "evaluation" not in legacy["results"]
    assert legacy["results"]["summary"]["class_count"] == 3
