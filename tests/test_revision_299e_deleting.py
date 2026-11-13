"""299e (agreed 2026-11-13): deleting runs, models and configurations.

What owns something takes it with it: a configuration its training runs,
their models and the predictions made with them; a model its files and its
prediction runs; a run its log and result folder. A training run whose model
still exists is hidden (its settings stay, so the model still predicts).
Confirmation pages list what goes; queued or running runs block a deletion.
Also: "Välj för körning" follows the traffic lights, and the Dark Numbers
confusion matrices show the estimate left of the actual class.
"""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGRunJobStore import RunJobStore  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

CSV = b"id;sepal;species\n1;5.1;a\n2;7.0;b\n3;6.3;a\n4;4.9;b\n"
OWNER = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
ARTIFACT = "00000000-0000-4000-8000-0000000000aa"


def _client(tmp_path):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _open(client):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _files(tmp_path, project, run, stem=None):
    """The files a finished run leaves: result folder, logs, model file."""
    folder = tmp_path / "runs" / project / run.key / "output" / "csvs"
    folder.mkdir(parents=True)
    (folder / "predictions_x.csv").write_text("a;b\n", encoding="utf-8")
    (tmp_path / "run-logs").mkdir(exist_ok=True)
    (tmp_path / "run-logs" / f"run_{run.key}.log").write_text("log", encoding="utf-8")
    (tmp_path / "run-logs" / f"run_{run.key}_engine.log").write_text("log", encoding="utf-8")
    if stem:
        (tmp_path / "models" / project).mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / project / f"{stem}.sav").write_bytes(b"model")
        (tmp_path / "models" / project / f"{stem}.sav.net.keras").write_bytes(b"sidecar")


def _run(store, project, *, finish=True, **meta):
    run = store.create(OWNER, DataSourceKind.CSV, {"spec_version": 1, "project_run": dict(
        project_id=project, configuration_name="LR", configuration_version=1, **meta)})
    store.record_progress(run, "x", 0.5)
    if finish:
        store.claim(run, "w")
        store.finish(run, "w", RunStatus.SUCCEEDED, result={"schema_version": 1})
    return run


def _scene(client, tmp_path):
    """Configuration C; training run T made with it and its model M; a
    prediction run P with M; another configuration's training run U."""
    from JBGModelCatalog import ModelState
    project = _open(client)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    configuration = client.post(f"/api/projects/{project}/configurations",
                                json={"name": "LR"}).json()["configuration_id"]
    other = client.post(f"/api/projects/{project}/configurations",
                        json={"name": "Other"}).json()["configuration_id"]
    store = client.app.state.jbg["store"]
    training = _run(store, project, configuration_id=configuration, dataset_id=dataset,
                    model_id=ARTIFACT)
    catalog = client.app.state.jbg["models"]
    model = catalog.register(OWNER, project, "LR v1 · test", ARTIFACT, dataset_id=dataset,
                             run_id=training.key)
    catalog.set_state(OWNER, project, model.model_id, ModelState.AVAILABLE)
    prediction = _run(store, project, kind="prediction", model_id=model.model_id,
                      dataset_id=dataset)
    unrelated = _run(store, project, configuration_id=other, dataset_id=dataset,
                     model_id="00000000-0000-4000-8000-0000000000bb")
    _files(tmp_path, project, training, ARTIFACT)
    _files(tmp_path, project, prediction)
    _files(tmp_path, project, unrelated, "00000000-0000-4000-8000-0000000000bb")
    return dict(project=project, configuration=configuration, other=other, model=model.model_id,
                training=training, prediction=prediction, unrelated=unrelated, store=store,
                catalog=catalog)


def _gone(tmp_path, project, run):
    return not (tmp_path / "runs" / project / run.key).exists() and not (
        tmp_path / "run-logs" / f"run_{run.key}.log").exists() and not (
        tmp_path / "run-logs" / f"run_{run.key}_engine.log").exists()


# --- The store ---------------------------------------------------------------------------

def test_hidden_runs_are_left_out_of_lists_and_owner_reads(tmp_path):
    store = RunJobStore.sqlite(tmp_path / "s.sqlite")
    store.create_tables()
    run = _run(store, "p")
    store.hide(run)
    assert store.list_for_owner(OWNER) == ()
    assert [r.run for r in store.list_for_owner(OWNER, None, include_hidden=True)] == [run]
    with pytest.raises(KeyError):
        store.get_owned(OWNER, run)
    record = store.get_owned(OWNER, run, include_hidden=True)
    assert record.result is None and store.progress(run) == ()
    assert store.hidden_runs(OWNER) == {run.key}
    store.remove(run)
    with pytest.raises(KeyError):
        store.get(run)
    store.close()


def test_unfinished_runs_are_never_removed(tmp_path):
    store = RunJobStore.sqlite(tmp_path / "s.sqlite")
    store.create_tables()
    run = _run(store, "p", finish=False)
    for action in (store.hide, store.remove):
        with pytest.raises(ValueError):
            action(run)
    assert store.get(run).status is RunStatus.PENDING
    store.close()


# --- Runs --------------------------------------------------------------------------------

def test_deleting_a_prediction_run(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        page = client.get(f"/ui/runs/{s['prediction'].key}/delete?back=/ui/prediction").text
        assert "Ta bort körning" in page and "Prediktionskörning" in page
        assert '<button class="btn act warning">Ta bort</button>' in page
        assert '<a href="/ui/prediction" class="on"' in page
        done = client.post(f"/ui/runs/{s['prediction'].key}/delete", data={"back": "/ui/prediction"},
                           follow_redirects=False)
        assert done.headers["location"] == "/ui/prediction?message=run_deleted"
        assert "Körningen är borttagen." in client.get(done.headers["location"]).text
        with pytest.raises(KeyError):
            s["store"].get(s["prediction"])
        assert _gone(tmp_path, s["project"], s["prediction"])
        assert (tmp_path / "models" / s["project"] / f"{ARTIFACT}.sav").is_file()


def test_a_training_run_with_a_model_is_hidden_and_the_model_still_predicts(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        page = client.get(f"/ui/runs/{s['training'].key}/delete").text
        assert "Modellen som körningen tränade finns kvar" in page and "LR v1 · test" in page
        client.post(f"/ui/runs/{s['training'].key}/delete", data={"back": "/ui/run"})
        store = s["store"]
        assert store.is_hidden(s["training"]) and _gone(tmp_path, s["project"], s["training"])
        assert (tmp_path / "models" / s["project"] / f"{ARTIFACT}.sav").is_file()
        assert s["training"].key not in client.get("/ui/run").text
        assert client.get(f"/ui/results?run={s['training'].key}").status_code == 200
        assert client.get(f"/api/runs/{s['training'].key}").status_code == 404
        card = client.get(f"/ui/models?model={s['model']}").text
        assert "<dd>Borttagen</dd>" in card and "Prediktionskörning" in card
        # The prediction path reads the hidden run's settings.
        assert store.get_owned(OWNER, s["training"], include_hidden=True).spec["project_run"]


def test_a_training_run_without_a_model_takes_its_model_file(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        stem = "00000000-0000-4000-8000-0000000000bb"
        client.post(f"/ui/runs/{s['unrelated'].key}/delete", data={"back": "/ui/run"})
        with pytest.raises(KeyError):
            s["store"].get(s["unrelated"])
        assert not list((tmp_path / "models" / s["project"]).glob(f"{stem}*"))
        assert (tmp_path / "models" / s["project"] / f"{ARTIFACT}.sav").is_file()


def test_a_queued_run_cannot_be_deleted(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        run = _run(client.app.state.jbg["store"], project, finish=False)
        page = client.get(f"/ui/runs/{run.key}/delete").text
        assert "står i kö eller pågår" in page and "btn act warning" not in page
        refused = client.post(f"/ui/runs/{run.key}/delete", data={"back": "/ui/run"},
                              follow_redirects=False)
        assert refused.headers["location"].startswith(f"/ui/runs/{run.key}/delete")
        assert client.app.state.jbg["store"].get(run).status is RunStatus.PENDING


def test_a_run_still_held_by_its_worker_is_refused(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        deletion = client.app.state.jbg["deletion"]
        deletion._active_runs = lambda: (s["prediction"],)
        page = client.get(f"/ui/runs/{s['prediction'].key}/delete").text
        assert "avslutas fortfarande på servern" in page
        client.post(f"/ui/runs/{s['prediction'].key}/delete", data={"back": "/ui/run"})
        assert s["store"].get(s["prediction"])


def test_other_users_and_other_projects_cannot_delete(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        foreign = s["store"].create("someone else", DataSourceKind.CSV, {"spec_version": 1,
                                    "project_run": {"project_id": s["project"]}})
        assert client.get(f"/ui/runs/{foreign.key}/delete", follow_redirects=False) \
            .headers["location"] == "/ui/run?message=delete_unavailable"
        client.post(f"/ui/runs/{foreign.key}/delete", data={"back": "/ui/run"})
        assert s["store"].get(foreign)
        client.post("/ui/close-project", data={"back": "/ui/run"})
        client.post(f"/ui/runs/{s['prediction'].key}/delete", data={"back": "/ui/run"})
        assert s["store"].get(s["prediction"])                  # no open project


# --- Models ------------------------------------------------------------------------------

def test_deleting_a_model_takes_its_files_and_predictions(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        client.post(f"/ui/models/{s['model']}/choose")
        page = client.get(f"/ui/models/{s['model']}/delete").text
        assert "LR v1 · test" in page and "Prediktionskörning" in page
        assert "Träningskörningen och dess resultat" in page          # stays
        done = client.post(f"/ui/models/{s['model']}/delete", follow_redirects=False)
        assert done.headers["location"] == "/ui/models?message=model_deleted"
        assert any(h.startswith("jbg_model=") for h in done.headers.get_list("set-cookie"))
        assert s["catalog"].list_for_project(OWNER, s["project"]) == ()
        assert not list((tmp_path / "models" / s["project"]).glob(f"{ARTIFACT}*"))
        with pytest.raises(KeyError):
            s["store"].get(s["prediction"])
        assert _gone(tmp_path, s["project"], s["prediction"])
        assert s["store"].get_owned(OWNER, s["training"])               # the run stays


def test_a_model_whose_run_was_deleted_takes_the_hidden_record(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        client.post(f"/ui/runs/{s['training'].key}/delete", data={"back": "/ui/run"})
        client.post(f"/ui/models/{s['model']}/delete")
        with pytest.raises(KeyError):
            s["store"].get(s["training"])
        assert s["store"].hidden_runs(OWNER) == frozenset()


# --- Configurations ----------------------------------------------------------------------

def test_deleting_a_configuration_takes_its_runs_and_models(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        client.post(f"/ui/configuration/{s['configuration']}/select")
        page = client.get(f"/ui/configuration/{s['configuration']}/delete").text
        assert "Konfigurationen med alla sina versioner: 1" in page
        assert "Träningskörning" in page and "Prediktionskörning" in page and "LR v1 · test" in page
        done = client.post(f"/ui/configuration/{s['configuration']}/delete", follow_redirects=False)
        assert done.headers["location"] == "/ui/configuration?message=configuration_deleted"
        assert any(h.startswith("jbg_configuration=") for h in done.headers.get_list("set-cookie"))
        names = [c["name"] for c in client.get(f"/api/projects/{s['project']}/configurations").json()]
        assert names == ["Other"]
        for run in (s["training"], s["prediction"]):
            with pytest.raises(KeyError):
                s["store"].get(run)
            assert _gone(tmp_path, s["project"], run)
        assert s["catalog"].list_for_project(OWNER, s["project"]) == ()
        assert s["store"].get(s["unrelated"])                           # another configuration's
        assert (tmp_path / "models" / s["project"] /
                "00000000-0000-4000-8000-0000000000bb.sav").is_file()


def test_a_configuration_in_use_by_a_queued_run_is_refused(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        _run(s["store"], s["project"], finish=False, configuration_id=s["configuration"])
        assert "står i kö eller pågår" in client.get(
            f"/ui/configuration/{s['configuration']}/delete").text
        client.post(f"/ui/configuration/{s['configuration']}/delete")
        assert len(client.get(f"/api/projects/{s['project']}/configurations").json()) == 2


# --- The entry points and the two fixes --------------------------------------------------

def test_pages_offer_deleting(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        assert f"/ui/runs/{s['training'].key}/delete?back=/ui/run" in client.get(
            f"/ui/run?run={s['training'].key}").text
        assert f"/ui/runs/{s['prediction'].key}/delete?back=/ui/prediction" in client.get(
            f"/ui/prediction?run={s['prediction'].key}").text
        assert f"/ui/runs/{s['training'].key}/delete?back=/ui/results" in client.get(
            f"/ui/results?run={s['training'].key}").text
        assert f"/ui/models/{s['model']}/delete" in client.get(f"/ui/models?model={s['model']}").text
        assert "Ta bort konfigurationen…" in client.get(
            f"/ui/configuration?configuration={s['configuration']}").text


def test_choose_for_runs_follows_the_traffic_lights(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        page = client.get(f"/ui/configuration?configuration={s['configuration']}").text
        assert '<button class="btn act ready">Välj för körning</button>' in page
        client.post(f"/ui/configuration/{s['configuration']}/select")
        page = client.get(f"/ui/configuration?configuration={s['configuration']}").text
        assert ('<button class="btn act blocked" disabled title="Konfigurationen är redan vald '
                'för körning.">Välj för körning</button>') in page
        assert "Vald för körning" in page


def test_the_estimate_comes_before_the_actual_class():
    from JBGWebUIResults import estimate_first
    detail = {"columns": ["", "Model type", "a", "b"],
              "rows": [["a", "Korstränad modell, testdata", 9, 1], ["b", "", 2, 8]]}
    estimate_first(detail)
    assert detail["columns"] == ["Model type", "", "a", "b"]
    assert detail["rows"] == [["Korstränad modell, testdata", "a", 9, 1], ["", "b", 2, 8]]
    from JBGWebUI import WEB
    template = (WEB / "templates" / "results.html").read_text(encoding="utf-8")
    assert '_("Actual class") if c == ""' in template


def test_the_worker_names_the_engine_log_after_the_run():
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent / "src" / "JBGRunWorker.py").read_text(
        encoding="utf-8")
    assert 'log_filename=f"run_{run.key}_engine.log"' in source


def test_the_new_texts_are_translated():
    import JBGWebUI
    from JBGWebUIDeletion import REFUSALS
    catalog = JBGWebUI._load_catalog("sv")
    texts = list(REFUSALS.values()) + [m[1] for k, m in JBGWebUI.MESSAGES.items()
                                       if k.endswith("deleted") or k.startswith("delete_")]
    assert all(text in catalog for text in texts)


def test_the_hidden_table_compiles_for_sql_server_in_a_schema():
    from sqlalchemy.dialects import mssql
    from sqlalchemy.schema import CreateTable
    from JBGRunJobStore import _hidden_table, _metadata
    metadata, jobs, _ = _metadata("example_schema")
    ddl = str(CreateTable(_hidden_table(metadata, jobs)).compile(dialect=mssql.dialect()))
    assert "example_schema.jbg_run_hidden" in ddl and "REFERENCES example_schema.jbg_run_jobs" in ddl
