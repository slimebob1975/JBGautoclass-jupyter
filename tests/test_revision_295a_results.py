"""295a: per-run result folders, owner-only details, the Results tab."""
import json
import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from Config import Config

LEADERBOARD = ';Algorithm;Mean cv;Stdev.\n0;LRN - Logistic Regression;0,9512;0,0213\n1;KNN - K-Neighbors;0,9333;0,0300\n'


# --- The engine's output folder ---------------------------------------------------------

def test_output_path_defaults_to_script_path_and_is_never_saved(tmp_path):
    config = Config()
    assert Path(config.get_output_filepath("misplaced")).parent == Path(config.script_path) / "output" / "csvs"
    config.output_path = tmp_path / "run"
    assert Path(config.get_output_filepath("misplaced")).parent == tmp_path / "run" / "output" / "csvs"
    assert "output_path" not in json.dumps(config._config_to_json(username=""), default=str)


def _predictions_handler(tmp_path, output_path):
    from JBGHandler import PredictionsHandler
    config = Config()
    config.data_source.kind = "csv"
    config.connection.id_column = "id"
    config.output_path = output_path
    published = []
    handler = PredictionsHandler.__new__(PredictionsHandler)
    handler.handler = SimpleNamespace(
        config=config, datalayer=SimpleNamespace(save_mispredicted_data=published.append),
        logger=SimpleNamespace(print_key_value_pair=lambda *a, **k: None))
    frame = pd.DataFrame({"Actual": ["a"], "Predicted": ["b"]}, index=pd.Index([7], name="id"))
    handler.X_most_mispredicted, handler.X_mispredicted, handler.num_mispredicted = frame, frame, 1
    target = tmp_path / "misplaced.csv"
    handler.evaluate_mispredictions(str(target))
    return published, target


def test_csv_mispredictions_go_to_a_run_folder_only_when_the_run_has_one(tmp_path):
    published, target = _predictions_handler(tmp_path / "api", tmp_path / "run")
    assert len(published) == 1 and target.is_file()                 # also kept for the owner
    published, target = _predictions_handler(tmp_path / "voila", None)
    assert len(published) == 1 and not target.exists()               # Voila: storage only


# --- The results service -------------------------------------------------------------------------

def _store(tmp_path):
    from JBGRunJobStore import RunJobStore
    store = RunJobStore.sqlite(tmp_path / "jobs.sqlite")
    store.create_tables()
    return store


def _project_run(store, owner="me", project="11111111-1111-4111-8111-111111111111"):
    from JBGApplicationContracts import DataSourceKind
    return store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "model_id": "m", "configuration_version": 1}}), project


def _write(root, project, run, name, text):
    folder = root / project / run.key / "output" / "csvs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")


def test_details_are_read_with_decimal_commas_and_paged(tmp_path):
    from JBGRunResults import RunResultsService
    store = _store(tmp_path)
    run, project = _project_run(store)
    root = tmp_path / "runs"
    _write(root, project, run, "crossval_p__train.csv", LEADERBOARD)
    _write(root, project, run, "crossval_p__train_grid_search.csv", ";Other\n0;x\n")   # companion
    rows = "".join(f"{i};{i};a;b\n" for i in range(120))
    _write(root, project, run, "misplaced_p__train.csv", "id;id;Actual;Predicted\n" + rows)
    _write(root, project, run, "feature_importance_p__train.csv", ";Feature;Mean score decrease\n0;sepal;0,25\n")
    _write(root, project, run, "feature_importance_details_p__train.csv", ";Value\nmethod;permutation\n")
    (root / project / run.key / "class_labels.json").write_text(json.dumps({"class_labels": ["a", "b"]}))
    service = RunResultsService(store, results_root=root)
    board = service.detail("me", run.key, "leaderboard")
    assert board["columns"] == ["Algorithm", "Mean cv", "Stdev."]          # index column dropped
    assert board["rows"][0] == ["LRN - Logistic Regression", 0.9512, 0.0213]
    page = service.detail("me", run.key, "mispredictions", page=3)
    assert (page["total_rows"], len(page["rows"]), page["page"]) == (120, 20, 3)
    assert service.detail("me", run.key, "mispredictions", page_size=10000)["page_size"] == 500
    assert service.detail("me", run.key, "feature_importance")["columns"] == ["Feature", "Mean score decrease"]
    assert service.class_labels("me", run.key) == ["a", "b"] and service.has_folder("me", run.key)


def test_details_are_for_the_owner_of_a_project_run_only(tmp_path):
    from JBGApplicationContracts import DataSourceKind
    from JBGRunResults import RunResultsService, RunResultsUnavailable
    store = _store(tmp_path)
    run, project = _project_run(store)
    _write(tmp_path / "runs", project, run, "crossval_p__train.csv", LEADERBOARD)
    service = RunResultsService(store, results_root=tmp_path / "runs")
    other_kind = store.create("me", DataSourceKind.CSV, {"spec_version": 1})       # not a project run
    for owner, run_id, kind in (("bob", run.key, "leaderboard"), ("me", run.key, "nope"),
                                ("me", other_kind.key, "leaderboard"), ("me", "not-a-run", "leaderboard"),
                                ("me", run.key, "mispredictions")):
        with pytest.raises(RunResultsUnavailable):
            service.detail(owner, run_id, kind)


# --- Endpoints and the Results tab ------------------------------------------------------------------

def _client(tmp_path, subject="developer"):
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _finished_run(client):
    from JBGApplicationContracts import RunReference, RunStatus
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    store = client.app.state.jbg["store"]
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    from JBGApplicationContracts import DataSourceKind
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    run = store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "configuration_name": "C", "configuration_version": 2}})
    store.claim(run, "w")
    store.finish(run, "w", RunStatus.SUCCEEDED, result={
        "schema_version": 1, "kind": "classification", "class_count": 2, "mispredicted_count": 0,
        "evaluation": {"schema_version": 1, "accuracy": 0.75, "balanced_accuracy": 0.75, "mcc": 0.5,
                       "precision_macro": 0.75, "recall_macro": 0.75, "f1_macro": 0.7,
                       "precision_weighted": 0.75, "recall_weighted": 0.75, "f1_weighted": 0.7,
                       "confusion_matrix": [[2, 0], [1, 1]]}})
    return project, run


def test_endpoints_are_owner_bound(tmp_path):
    with _client(tmp_path) as client:
        project, run = _finished_run(client)
        _write(tmp_path / "runs", project, run, "crossval_p__train.csv", LEADERBOARD)
        assert client.get(f"/api/runs/{run.key}/details/leaderboard").json()["total_rows"] == 2
        assert client.get(f"/api/runs/{run.key}/class-labels").json()["class_labels"] == []
    with _client(tmp_path, subject="bob") as theirs:
        for url in (f"/api/runs/{run.key}/details/leaderboard", f"/api/runs/{run.key}/class-labels"):
            assert theirs.get(url).status_code == 404


def test_results_tab_names_the_classes_and_shows_the_details(tmp_path):
    with _client(tmp_path) as client:
        project, run = _finished_run(client)
        root = tmp_path / "runs"
        (root / project / run.key).mkdir(parents=True)
        (root / project / run.key / "class_labels.json").write_text(
            json.dumps({"class_labels": ["setosa", "<b>virginica</b>"]}), encoding="utf-8")
        _write(root, project, run, "crossval_p__train.csv", LEADERBOARD)
        page = client.get("/ui/results").text
        assert "<th class=\"num\">setosa</th>" in page and "&lt;b&gt;virginica" in page   # escaped
        assert "Avslutade körningar" in page and "C v2" in page
        board = client.get(f"/ui/results?run={run.key}&view=leaderboard").text
        assert "<td class=\"num\">0,9512</td>" in board
        empty = client.get(f"/ui/results?run={run.key}&view=mispredictions").text
        assert "Inget att visa för den här körningen" in empty


def test_runs_from_before_295a_say_so(tmp_path):
    with _client(tmp_path) as client:
        _, run = _finished_run(client)                         # no run folder at all
        page = client.get(f"/ui/results?run={run.key}&view=leaderboard").text
        assert "tidigare körningar sparade dem inte" in page
        overview = client.get(f"/ui/results?run={run.key}").text
        assert '<th class="num">1</th>' in overview            # numbered without class names


def test_paging_links_and_translations(tmp_path):
    import JBGWebUI
    with _client(tmp_path) as client:
        project, run = _finished_run(client)
        rows = "".join(f"{i};{i};a;b\n" for i in range(120))
        _write(tmp_path / "runs", project, run, "misplaced_p__train.csv", "id;id;Actual;Predicted\n" + rows)
        page = client.get(f"/ui/results?run={run.key}&view=mispredictions&page=2").text
        assert "Sida 2 / 3" in page and "&page=1" in page and "&page=3" in page
    from JBGWebUIResults import VIEWS
    assert not [label for _, label in VIEWS if label not in JBGWebUI._CATALOGS["sv"]]
