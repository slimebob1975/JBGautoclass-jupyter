"""288c: models of project runs are registered in the catalog and listed."""
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGRunWorker import register_trained_model  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

MISSING = "00000000-0000-4000-8000-000000000000"
CSV = b"row_id;f1;f2;target\n1;5.1;1.4;a\n2;7.0;4.7;b\n3;6.3;6.0;a\n4;4.9;1.4;b\n"
ROLES = {"connection": {"id_column": "row_id", "class_column": "target",
                        "data_numerical_columns": ["f1", "f2"], "data_text_columns": []}}


class _Launcher:
    def __call__(self, command, **options):
        stdin = type("S", (), {"write": lambda s, d: None, "close": lambda s: None})()
        return type("P", (), {"stdin": stdin, "poll": lambda s: None})()


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5), "launcher": _Launcher()}))


def _queued_run(client):
    project = client.post("/api/projects", json={"name": "Iris", "source": "csv"}).json()["project_id"]
    dataset = client.post(f"/api/projects/{project}/datasets/csv",
                          data={"name": "Data", "role": "training"},
                          files={"file": ("data.csv", CSV, "text/csv")}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    configuration = client.post(f"/api/projects/{project}/configurations",
                                json={"name": "Standard", "settings": ROLES}).json()["configuration_id"]
    run = client.post(f"/api/projects/{project}/runs",
                      json={"configuration_id": configuration}).json()["run_id"]
    return project, dataset, RunReference(run)


def _finish(client, run, status=RunStatus.SUCCEEDED, write_model=True):
    """Do what a worker does, without training."""
    store = client.app.state.jbg["store"]
    store.claim(run, "worker")
    io = store.get(run).spec["configuration"]["io"]
    from pathlib import Path
    model_file = Path(io["model_path"]) / (io["model_name"] + ".sav")
    if write_model:
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"model")
    store.finish(run, "worker", status)
    config = SimpleNamespace(get_model_filename=lambda: str(model_file))
    return store, store.get(run), config


def test_a_succeeded_project_run_registers_its_model(tmp_path):
    with _client(tmp_path) as client:
        project, dataset, run = _queued_run(client)
        store, record, config = _finish(client, run)
        assert register_trained_model(store, record, config)
        listed = client.get(f"/api/projects/{project}/models").json()
        assert len(listed) == 1
        model = listed[0]
        assert model["name"] == f"Standard v1 · {run.key[:8]}"
        assert (model["state"], model["source"], model["run_id"], model["dataset_id"]) == (
            "available", "csv", run.key, dataset)
        assert not {"artifact_key", "model_path", "path"} & set(model)
        assert client.get(f"/api/projects/{project}/models/{model['model_id']}").json() == model


@pytest.mark.parametrize("case", ["no model file", "failed run", "not a project run"])
def test_nothing_is_registered_without_a_model_of_a_succeeded_project_run(tmp_path, case):
    with _client(tmp_path) as client:
        project, _, run = _queued_run(client)
        store, record, config = _finish(
            client, run, status=RunStatus.FAILED if case == "failed run" else RunStatus.SUCCEEDED,
            write_model=case != "no model file")
        if case == "not a project run":
            record = SimpleNamespace(**{**record.__dict__, "spec": {
                k: v for k, v in record.spec.items() if k != "project_run"}})
        assert register_trained_model(store, record, config) is False      # never raises
        assert client.get(f"/api/projects/{project}/models").json() == []


def test_a_catalog_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    import JBGModelCatalog
    with _client(tmp_path) as client:
        project, _, run = _queued_run(client)
        store, record, config = _finish(client, run)
        monkeypatch.setattr(JBGModelCatalog.ModelCatalog, "register",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
        assert register_trained_model(store, record, config) is False
        assert store.get(run).status is RunStatus.SUCCEEDED


def test_models_are_owner_bound(tmp_path):
    with _client(tmp_path) as mine:
        project, _, run = _queued_run(mine)
        store, record, config = _finish(mine, run)
        register_trained_model(store, record, config)
        model_id = mine.get(f"/api/projects/{project}/models").json()[0]["model_id"]
        assert mine.get(f"/api/projects/{project}/models/{MISSING}").json() == {
            "detail": "model unavailable"}
    with _client(tmp_path, subject="bob") as theirs:
        for response in (theirs.get(f"/api/projects/{project}/models"),
                         theirs.get(f"/api/projects/{project}/models/{model_id}")):
            assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})
