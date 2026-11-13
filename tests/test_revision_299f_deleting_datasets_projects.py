"""299f (agreed 2026-11-13): deleting datasets and projects.

A dataset takes the training runs on it, the models trained on it, the
prediction runs on it or with those models, its column roles and every
version of its CSV file; configurations are not tied to a dataset and stay.
CSV file versions nothing refers to any more are cleaned up at the same time.
A project takes everything in it and is deleted only when its name is typed.
"""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
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


def _project(client, name="P", open_it=True):
    location = client.post("/ui/projects", data={"name": name, "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    if open_it:
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _dataset(client, project, name, role):
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": name, "role": role},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    repository = client.app.state.jbg["datasets"]
    return repository.get_owned(OWNER, project, dataset)


def _run(store, project, artifact, *, finish=True, **meta):
    run = store.create(OWNER, DataSourceKind.CSV, {
        "spec_version": 1, "csv": {"artifact_id": artifact},
        "project_run": dict(project_id=project, configuration_name="LR", **meta)})
    if finish:
        store.claim(run, "w")
        store.finish(run, "w", RunStatus.SUCCEEDED, result={"schema_version": 1})
    return run


def _groups(tmp_path, project):
    """The staging IDs of the project's CSV artifacts."""
    from JBGProjectStorage import project_csv_storage
    repository, scope = project_csv_storage(project, root=tmp_path / "csv")
    return {ref.dataset_id for ref in repository.list_artifacts(scope) if ref.dataset_id}


def _staging(tmp_path, project, artifact_id):
    from JBGProjectStorage import project_csv_storage
    repository, scope = project_csv_storage(project, root=tmp_path / "csv")
    return repository.get_reference(scope, artifact_id).dataset_id


def _upload(tmp_path, project):
    """A file version as a correction leaves it (registered, not yet referenced)."""
    from JBGProjectStorage import project_csv_storage, register_csv_upload
    repository, scope = project_csv_storage(project, root=tmp_path / "csv")
    path = tmp_path / "v.csv"
    path.write_bytes(CSV)
    return register_csv_upload(repository, scope, path, delimiter=";", encoding="utf-8").artifact_id


def _scene(client, tmp_path):
    """Training dataset D (corrected once: two file versions), prediction
    dataset E, configuration C; training run T on D's first version with model
    M; prediction runs with M on D and on E; one unused file version."""
    from JBGModelCatalog import ModelState
    project = _project(client)
    d = _dataset(client, project, "D", "training")
    e = _dataset(client, project, "E", "prediction")
    configuration = client.post(f"/api/projects/{project}/configurations",
                                json={"name": "C"}).json()["configuration_id"]
    store = client.app.state.jbg["store"]
    training = _run(store, project, d.registration_id, dataset_id=d.dataset_id,
                    configuration_id=configuration, model_id=ARTIFACT)
    catalog = client.app.state.jbg["models"]
    model = catalog.register(OWNER, project, "M", ARTIFACT, dataset_id=d.dataset_id,
                             run_id=training.key)
    catalog.set_state(OWNER, project, model.model_id, ModelState.AVAILABLE)
    on_d = _run(store, project, d.registration_id, kind="prediction", model_id=model.model_id,
                dataset_id=d.dataset_id)
    on_e = _run(store, project, e.registration_id, kind="prediction", model_id=model.model_id,
                dataset_id=e.dataset_id)
    corrected = _upload(tmp_path, project)                       # D's current version
    client.app.state.jbg["datasets"].replace_csv_file(OWNER, project, d.dataset_id, corrected)
    unused = _upload(tmp_path, project)                          # nothing refers to it
    (tmp_path / "models" / project).mkdir(parents=True)
    (tmp_path / "models" / project / f"{ARTIFACT}.sav").write_bytes(b"model")
    for run in (training, on_d, on_e):
        (tmp_path / "runs" / project / run.key).mkdir(parents=True)
    return dict(project=project, d=d, e=e, configuration=configuration, store=store,
                catalog=catalog, model=model.model_id, training=training, on_d=on_d, on_e=on_e,
                stages={"d1": _staging(tmp_path, project, d.registration_id),
                        "d2": _staging(tmp_path, project, corrected),
                        "e": _staging(tmp_path, project, e.registration_id),
                        "unused": _staging(tmp_path, project, unused)})


# --- Datasets ----------------------------------------------------------------------------

def test_deleting_a_training_dataset_takes_its_runs_models_and_files(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        client.post(f"/ui/models/{s['model']}/choose")
        page = client.get(f"/ui/dataset/{s['d'].dataset_id}/delete").text
        assert "Ta bort dataset" in page and "<b>D</b>" in page and "<b>M</b>" in page
        assert "Datasetets versioner av CSV-filen, även korrigerade: 2" in page
        assert "som inget längre använder" in page and ": 1</li>" in page
        assert "Konfigurationer: de hör inte till något dataset." in page
        done = client.post(f"/ui/dataset/{s['d'].dataset_id}/delete", follow_redirects=False)
        assert done.headers["location"] == "/ui/dataset?message=dataset_deleted"
        assert any(h.startswith("jbg_model=") for h in done.headers.get_list("set-cookie"))
        assert "Datasetet är borttaget." in client.get(done.headers["location"]).text
        names = [d["name"] for d in client.get(f"/api/projects/{s['project']}/datasets").json()]
        assert names == ["E"]
        for run in (s["training"], s["on_d"], s["on_e"]):            # on E: made with M
            with pytest.raises(KeyError):
                s["store"].get(run)
        assert s["catalog"].list_for_project(OWNER, s["project"]) == ()
        assert not (tmp_path / "models" / s["project"] / f"{ARTIFACT}.sav").exists()
        assert _groups(tmp_path, s["project"]) == {s["stages"]["e"]}
        assert [c["name"] for c in client.get(
            f"/api/projects/{s['project']}/configurations").json()] == ["C"]


def test_deleting_a_prediction_dataset_keeps_the_model(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        client.post(f"/ui/dataset/{s['e'].dataset_id}/delete")
        with pytest.raises(KeyError):
            s["store"].get(s["on_e"])
        assert s["store"].get(s["on_d"]) and s["store"].get(s["training"])
        assert len(s["catalog"].list_for_project(OWNER, s["project"])) == 1
        groups = _groups(tmp_path, s["project"])
        assert s["stages"]["e"] not in groups and s["stages"]["unused"] not in groups
        assert {s["stages"]["d1"], s["stages"]["d2"]} <= groups     # D's run and current file


def test_a_dataset_with_a_queued_run_is_refused(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        _run(s["store"], s["project"], s["d"].registration_id, finish=False,
             dataset_id=s["d"].dataset_id)
        assert "står i kö eller pågår" in client.get(f"/ui/dataset/{s['d'].dataset_id}/delete").text
        client.post(f"/ui/dataset/{s['d'].dataset_id}/delete")
        assert len(client.get(f"/api/projects/{s['project']}/datasets").json()) == 2


# --- Projects ----------------------------------------------------------------------------

def test_a_project_is_deleted_only_with_its_name(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        page = client.get(f"/ui/projects/{s['project']}/delete").text
        assert "Projektet och allt i det" in page and "Skriv projektets namn" in page
        assert 'data-wait="match"' in page and 'data-wait-match="P"' in page
        wrong = client.post(f"/ui/projects/{s['project']}/delete", data={"confirm_name": "p"},
                            follow_redirects=False)
        assert "mismatch=1" in wrong.headers["location"]
        assert "Namnet stämde inte" in client.get(wrong.headers["location"]).text
        assert client.get(f"/api/projects/{s['project']}").status_code == 200


def test_deleting_a_project_takes_everything_in_it(tmp_path):
    with _client(tmp_path) as client:
        other = _project(client, "Other", open_it=False)
        s = _scene(client, tmp_path)
        client.post(f"/ui/runs/{s['training'].key}/delete", data={"back": "/ui/run"})  # hidden
        done = client.post(f"/ui/projects/{s['project']}/delete", data={"confirm_name": " P "},
                           follow_redirects=False)
        assert done.headers["location"] == "/ui/projects?message=project_deleted"
        cookies = done.headers.get_list("set-cookie")
        assert any(h.startswith("jbg_project=") for h in cookies)
        assert client.get(f"/api/projects/{s['project']}").status_code == 404
        assert client.get(f"/api/projects/{other}").status_code == 200
        assert s["store"].list_for_owner(OWNER, None, include_hidden=True) == ()
        from JBGProjectStorage import project_csv_storage
        _, scope = project_csv_storage(s["project"], root=tmp_path / "csv")
        for folder in (tmp_path / "runs" / s["project"], tmp_path / "models" / s["project"],
                       tmp_path / "csv" / "objects" / scope.tenant_id / s["project"]):
            assert not folder.exists()
        assert "Projektet är borttaget." in client.get("/ui/projects?message=project_deleted").text


def test_a_project_with_a_queued_run_is_refused(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        _run(s["store"], s["project"], s["d"].registration_id, finish=False)
        page = client.get(f"/ui/projects/{s['project']}/delete").text
        assert "står i kö eller pågår" in page and "confirm-name" not in page
        client.post(f"/ui/projects/{s['project']}/delete", data={"confirm_name": "P"})
        assert client.get(f"/api/projects/{s['project']}").status_code == 200


def test_another_users_project_is_unknown(tmp_path):
    with _client(tmp_path) as client:
        projects = client.app.state.jbg["projects"]
        foreign = projects.create("someone else", "F", DataSourceKind.CSV)
        assert client.get(f"/ui/projects/{foreign.project_id}/delete",
                          follow_redirects=False).headers["location"] == \
            "/ui/projects?message=delete_unavailable"
        client.post(f"/ui/projects/{foreign.project_id}/delete", data={"confirm_name": "F"})
        assert projects.get_owned("someone else", foreign.project_id)


# --- Entry points, script and texts ------------------------------------------------------

def test_pages_offer_deleting(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        assert f"/ui/dataset/{s['d'].dataset_id}/delete" in client.get(
            f"/ui/dataset?dataset={s['d'].dataset_id}").text
        assert f"/ui/projects/{s['project']}/delete" in client.get(
            f"/ui/projects?selected={s['project']}").text


def test_waiting_warning_buttons_turn_orange_and_can_wait_for_a_match():
    from JBGWebUI import WEB
    script = (WEB / "static" / "jbg.js").read_text(encoding="utf-8")
    assert 'contains("warning") ? "warning" : "ready"' in script
    assert 'mode === "match"' in script and "data-wait-match" in script


def test_the_new_texts_are_translated():
    import JBGWebUI
    catalog = JBGWebUI._load_catalog("sv")
    for code in ("dataset_deleted", "project_deleted"):
        assert JBGWebUI.MESSAGES[code][1] in catalog
    template = (JBGWebUI.WEB / "templates" / "delete.html").read_text(encoding="utf-8")
    for text in re.findall(r'_\("([^"]+)"\)', template):
        assert text in catalog, text


def test_the_sql_line_is_only_for_sql_projects(tmp_path):
    with _client(tmp_path) as client:
        s = _scene(client, tmp_path)
        page = client.get(f"/ui/dataset/{s['d'].dataset_id}/delete").text
        assert "Allt i SQL Server" not in page
        assert "CSV-filerna på din egen dator" in client.get(
            f"/ui/projects/{s['project']}/delete").text
