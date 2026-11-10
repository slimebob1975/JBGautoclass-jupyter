"""297b: an empty row limit reads all rows in SQL runs, and a training run that
trains no model fails instead of succeeding (found 2026-11-10: a new SQL
project's first training run fetched no data and still ended "succeeded")."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGProjectRuns import SQL_ALL_ROWS, _read_all_rows_unless_limited  # noqa: E402
from JBGRunJobStore import RunJobStore  # noqa: E402


# --- The engine: why it happened, and what the runtime limit does ------------------------------

class _Stop(BaseException):
    """Stops get_dataset at its first query, past its ``except Exception``."""


def _fetch(limit, train=True, predict=False):
    from test_revision_295c_predictions import _layer
    layer, _ = _layer(train, predict, {"id": "int", "sepal": "float", "species": "nvarchar"})
    layer.config.debug.data_limit = limit
    warnings = []
    layer.logger = SimpleNamespace(print_formatted_info=lambda *a, **k: None,
                                   print_warning=warnings.append,
                                   print_query=lambda *a, **k: None)
    layer.count_class_distribution = lambda: {"a": 3, "b": 2, "NULL": 1}
    layer.get_connection = lambda: object()
    asked = []

    def query(rows, *args):
        asked.append(rows)
        raise _Stop()
    layer.get_data_query_with_order = query
    try:
        result = layer.get_dataset()
    except _Stop:
        result = "fetching"
    return result, asked, warnings


def test_the_sql_engine_reads_an_empty_limit_as_no_rows():
    """The cause: Voilà always fills in the limit; the web interface left it empty."""
    for empty in (None, 0):
        result, asked, warnings = _fetch(empty)
        assert result is None and asked == []
        assert warnings == ["No eligible SQL rows for this mode; no data was fetched."]


@pytest.mark.parametrize("train, predict, eligible", [
    (True, False, 5),          # training: the rows with a class
    (False, True, 1),          # prediction: the rows without
    (True, True, 6),
])
def test_the_runtime_limit_reads_every_eligible_row(train, predict, eligible):
    result, asked, _ = _fetch(SQL_ALL_ROWS, train, predict)
    assert result == "fetching" and asked == [eligible]


def test_a_limit_the_user_chose_is_kept():
    _, asked, _ = _fetch(4)
    assert asked == [4]


@pytest.mark.parametrize("limit, expected", [
    (None, SQL_ALL_ROWS), (0, SQL_ALL_ROWS), (-3, SQL_ALL_ROWS), ("", SQL_ALL_ROWS),
    (True, SQL_ALL_ROWS), (1, 1), (5000, 5000),
])
def test_only_an_unset_limit_is_replaced(limit, expected):
    config = SimpleNamespace(debug=SimpleNamespace(data_limit=limit))
    _read_all_rows_unless_limited(config)
    assert config.debug.data_limit == expected


# --- Submitting SQL runs -------------------------------------------------------------------------

def test_an_sql_training_run_without_a_row_limit_reads_all_rows(tmp_path):
    from JBGApplicationContracts import RunReference
    from test_revision_288b_sql_project_runs import _client, _setup, _sign_in, _sql_server
    with _sql_server(), _client(tmp_path) as client:
        project, _, configuration = _setup(client)
        _sign_in(client)
        created = client.post(f"/api/projects/{project}/runs",
                              json={"configuration_id": configuration})
        assert created.status_code == 201, created.text
        spec = client.app.state.jbg["store"].get(RunReference(created.json()["run_id"])).spec
        saved = client.get(f"/api/projects/{project}/configurations/{configuration}").json()
    assert spec["configuration"]["debug"]["data_limit"] == SQL_ALL_ROWS
    # Runtime only: the saved configuration keeps its empty limit.
    assert saved["settings"]["debug"]["data_limit"] is None


def test_an_sql_training_run_keeps_a_chosen_row_limit(tmp_path):
    from JBGApplicationContracts import RunReference
    from test_revision_288b_sql_project_runs import ROLES, _client, _setup, _sign_in, _sql_server
    settings = json.loads(json.dumps(ROLES))
    settings["debug"] = {"data_limit": 750}
    with _sql_server(), _client(tmp_path) as client:
        project, _, configuration = _setup(client, roles=settings)
        _sign_in(client)
        created = client.post(f"/api/projects/{project}/runs",
                              json={"configuration_id": configuration})
        spec = client.app.state.jbg["store"].get(RunReference(created.json()["run_id"])).spec
    assert spec["configuration"]["debug"]["data_limit"] == 750


def test_predicting_with_a_model_trained_without_a_limit_reads_all_rows(tmp_path):
    """Models trained before 297b stored the empty limit with their configuration."""
    from JBGApplicationContracts import RunReference
    from test_revision_295c_predictions import (_client, _sign_in, _sql_signed_in,
                                                _trained_sql_model)
    with _sql_signed_in(), _client(tmp_path) as client:
        project, model_id, prediction = _trained_sql_model(client, tmp_path)
        _sign_in(client)
        started = client.post(f"/api/projects/{project}/predictions",
                              json={"model_id": model_id, "dataset_id": prediction})
        assert started.status_code == 201, started.text
        spec = client.app.state.jbg["store"].get(RunReference(started.json()["run_id"])).spec
    assert spec["configuration"]["debug"]["data_limit"] == SQL_ALL_ROWS


# --- The worker: no model, no success ----------------------------------------------------------

def _job(tmp_path, *, kind="training", train=True, with_model=False):
    from Config import Config
    store = RunJobStore.sqlite(tmp_path / "jobs.sqlite")
    store.create_tables()
    config = Config()
    config.mode.train, config.mode.predict = train, not train
    config.mode.mispredicted = train
    config.mail.notification_email = ""
    config.io.model_path, config.io.model_name = str(tmp_path), "model-key"
    if with_model:
        Path(config.get_model_filename()).write_bytes(b"model")
    project_run = {"project_id": "p", "dataset_id": "d"}
    if kind is not None:
        project_run["kind"] = kind
    run = store.create("owner", DataSourceKind.CSV, {
        "spec_version": 1, "configuration": config._config_to_json(username=""),
        "project_run": project_run})
    return store, run, config


def _run(store, run, config):
    from JBGRunWorker import run_job
    composed = SimpleNamespace(
        preparation=SimpleNamespace(prepare=lambda *a: "prepared"),
        execution=SimpleNamespace(execute=lambda prepared: False))   # "Process finished"
    with patch("JBGRunWorker._compose", return_value=(composed, "reference", config)), \
            patch("JBGRunWorker.register_trained_model") as register:
        run_job(store, run, "w", logger=SimpleNamespace(print_progress=lambda **k: None))
    return store.get(run), register


def test_a_training_run_that_trained_no_model_fails(tmp_path, capsys):
    store, run, config = _job(tmp_path)
    record, register = _run(store, run, config)
    assert record.status is RunStatus.FAILED and record.error_type == "NoModelTrained"
    assert not register.called
    assert "the training run produced no model" in capsys.readouterr().err


def test_a_training_run_with_its_model_still_succeeds(tmp_path):
    store, run, config = _job(tmp_path, with_model=True)
    record, register = _run(store, run, config)
    assert record.status is RunStatus.SUCCEEDED and register.called


def test_prediction_runs_need_no_model_of_their_own(tmp_path):
    store, run, config = _job(tmp_path, kind="prediction", train=False)
    record, _ = _run(store, run, config)
    assert record.status is RunStatus.SUCCEEDED


def test_runs_outside_projects_keep_the_engines_outcome(tmp_path):
    from JBGRunWorker import trained_nothing
    record = SimpleNamespace(spec={"spec_version": 1})
    assert trained_nothing(record, SimpleNamespace()) is False
    broken = SimpleNamespace(should_train=lambda: True,
                             get_model_filename=lambda: (_ for _ in ()).throw(OSError()))
    assert trained_nothing(SimpleNamespace(spec={"project_run": {}}), broken) is False
