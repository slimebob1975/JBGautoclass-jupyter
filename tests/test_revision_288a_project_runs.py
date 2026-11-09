"""288a: runs from a CSV project's configuration and dataset."""
import time
from datetime import timedelta
from io import BytesIO
from uuid import UUID, uuid4

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import RunReference  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

MISSING = "00000000-0000-4000-8000-000000000000"
CSV = ("row_id;f1;f2;target\n1;5.1;1.4;a\n2;7.0;4.7;b\n3;6.3;6.0;a\n4;4.9;1.4;b\n").encode()
ROLES = {"connection": {"id_column": "row_id", "class_column": "target",
                        "data_numerical_columns": ["f1", "f2"], "data_text_columns": []}}


class _Launcher:
    """Records worker starts without running anything."""

    def __init__(self):
        self.commands = []

    def __call__(self, command, **options):
        self.commands.append(command)
        stdin = type("S", (), {"write": lambda s, d: None, "close": lambda s: None})()
        return type("P", (), {"stdin": stdin, "poll": lambda s: None})()


def _client(tmp_path, subject="developer", launcher=None, **dispatcher):
    options = {"poll_seconds": 60, "heartbeat_seconds": 1, "stale_after": timedelta(minutes=5),
               "launcher": launcher or _Launcher()}
    options.update(dispatcher)
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv",
        model_root=tmp_path / "models", sql_server=SQLServerSettings(),
        dispatcher_options=options))


def _setup(client, *, roles=ROLES, data=CSV, role="training", activate=True):
    project = client.post("/api/projects", json={"name": "Iris", "source": "csv"}).json()["project_id"]
    dataset = client.post(f"/api/projects/{project}/datasets/csv",
                          data={"name": "Data", "role": role},
                          files={"file": ("data.csv", data, "text/csv")}).json()["dataset_id"]
    if activate:
        client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    if role != "prediction":
        # 293a: the roles belong to the dataset (decision A).
        from jbg_test_roles import store_roles
        store_roles(client, project, dataset, roles)
    configuration = client.post(f"/api/projects/{project}/configurations",
                                json={"name": "Standard", "settings": roles}).json()
    return project, dataset, configuration["configuration_id"]


def _spec(client, run_id):
    return client.app.state.jbg["store"].get(RunReference(run_id)).spec


# --- Starting runs ----------------------------------------------------------------------

def test_run_records_configuration_version_dataset_and_its_own_model(tmp_path):
    with _client(tmp_path) as client:
        project, dataset, configuration = _setup(client)
        response = client.post(f"/api/projects/{project}/runs",
                               json={"configuration_id": configuration})
        assert response.status_code == 201, response.text
        assert response.json()["status"] == "pending"
        spec = _spec(client, response.json()["run_id"])
    run = spec["project_run"]
    assert (run["project_id"], run["configuration_id"], run["configuration_version"],
            run["dataset_id"]) == (project, configuration, 1, dataset)
    assert UUID(run["model_id"]).version == 4
    io = spec["configuration"]["io"]
    assert io["model_name"] == run["model_id"]
    assert io["model_path"] == str(tmp_path / "models" / project)
    assert spec["configuration"]["data_source"]["delimiter"] == ";"
    assert spec["configuration"]["connection"]["class_column"] == "target"
    assert not (tmp_path / "models" / "csv_artifacts").exists()


def test_a_pinned_version_is_used(tmp_path):
    with _client(tmp_path) as client:
        project, _, configuration = _setup(client)
        url = f"/api/projects/{project}/configurations/{configuration}"
        client.put(url, json={"settings": {"debug": {"data_limit": 2}}, "expected_version": 1})
        latest = client.post(f"/api/projects/{project}/runs",
                             json={"configuration_id": configuration}).json()
        pinned = client.post(f"/api/projects/{project}/runs",
                             json={"configuration_id": configuration, "version": 1}).json()
        latest_spec, pinned_spec = _spec(client, latest["run_id"]), _spec(client, pinned["run_id"])
    assert latest_spec["project_run"]["configuration_version"] == 2
    assert latest_spec["configuration"]["debug"]["data_limit"] == 2
    assert pinned_spec["project_run"]["configuration_version"] == 1
    assert pinned_spec["configuration"]["debug"]["data_limit"] != 2
    assert latest_spec["project_run"]["model_id"] != pinned_spec["project_run"]["model_id"]


@pytest.mark.parametrize("case, status, detail", [
    ("no active dataset", 409, "the project has no active training dataset"),
    ("prediction dataset", 409, "a training run needs a training or mixed dataset"),
    ("wrong column", 422, "the column roles do not fit the dataset"),
    ("non-unique id column", 422, "the column roles do not fit the dataset"),
])
def test_runs_that_cannot_start_are_refused_with_fixed_messages(tmp_path, case, status, detail):
    roles = {"connection": dict(ROLES["connection"])}
    body_extra = {}
    if case == "non-unique id column":
        roles["connection"]["id_column"] = "target"          # values repeat
        roles["connection"]["class_column"] = "row_id"
    if case == "wrong column":
        roles["connection"]["class_column"] = "no_such_column"
    with _client(tmp_path) as client:
        project, dataset, configuration = _setup(
            client, roles=roles, activate=case not in ("no active dataset", "prediction dataset"),
            role="prediction" if case == "prediction dataset" else "training")
        if case == "prediction dataset":
            body_extra["dataset_id"] = dataset
        response = client.post(f"/api/projects/{project}/runs",
                               json={"configuration_id": configuration, **body_extra})
        assert (response.status_code, response.json()) == (status, {"detail": detail})
        assert client.get("/api/runs").json() == []


def test_prediction_only_configurations_cannot_even_be_saved_for_csv(tmp_path):
    # The Config loader refuses them; the run service also refuses them (288b SQL).
    with _client(tmp_path) as client:
        project, _, _ = _setup(client)
        response = client.post(f"/api/projects/{project}/configurations", json={
            "name": "Predict", "settings": {"mode": {"train": False, "predict": True}}})
        assert (response.status_code, response.json()) == (422, {"detail": "configuration is invalid"})


def test_a_second_run_on_the_same_dataset_starts(tmp_path):
    # A first run writes another manifest for the dataset (its roles record).
    with _client(tmp_path) as client:
        project, _, configuration = _setup(client)
        for _ in range(3):
            response = client.post(f"/api/projects/{project}/runs",
                                   json={"configuration_id": configuration})
            assert response.status_code == 201, response.text


def test_sql_projects_need_sql_server_settings(tmp_path):
    # 288a refused all SQL projects; since 288b they need the server's SQL settings.
    with _client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
        configuration = client.post(f"/api/projects/{project}/configurations",
                                    json={"name": "C"}).json()["configuration_id"]
        response = client.post(f"/api/projects/{project}/runs",
                               json={"configuration_id": configuration})
        assert (response.status_code, response.json()) == (
            409, {"detail": "no SQL Server is configured on this server"})


def test_unknown_or_foreign_references(tmp_path):
    with _client(tmp_path) as mine:
        project, dataset, configuration = _setup(mine)
        url = f"/api/projects/{project}/runs"
        assert mine.post(url, json={"configuration_id": MISSING}).json() == {
            "detail": "configuration unavailable"}
        assert mine.post(url, json={"configuration_id": configuration,
                                    "dataset_id": MISSING}).json() == {"detail": "dataset unavailable"}
    with _client(tmp_path, subject="bob") as theirs:
        response = theirs.post(url, json={"configuration_id": configuration})
        assert (response.status_code, response.json()) == (404, {"detail": "project unavailable"})


def test_uploads_from_286b_without_a_recorded_format_are_detected(tmp_path):
    from JBGProjectStorage import csv_upload_format, project_csv_storage
    from Storage.ArtifactContract import ArtifactKind, ArtifactReference
    project_id = str(uuid4())
    repository, scope = project_csv_storage(project_id, root=tmp_path)
    reference = ArtifactReference(scope, str(uuid4()), ArtifactKind.DATASET, "text/csv",
                                  dataset_id=str(uuid4()))
    repository.save_artifact(scope, reference, BytesIO(CSV.replace(b";", b"|")))
    assert csv_upload_format(repository, scope, reference.artifact_id) == {
        "delimiter": "|", "quotechar": '"', "encoding": "utf-8-sig"}


# --- Native: a real worker trains the project's model ---------------------------------------

# Import the module, not the class, so pytest does not collect 281c's tests here.
import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class ProjectRunNative288a(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_project_run_trains_and_saves_its_own_model_file(self):
        self._configure_training()
        self.config.mail.notification_email = ""
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            data = stream.read()
        work = self.root / "api"
        work.mkdir()
        client = TestClient(create_server_app(
            settings=MetadataStoreSettings(sqlite_path=work / "jobs.sqlite"),
            run_logs=work / "run-logs", csv_root=work / "csv", model_root=work / "models",
            results_root=work / "runs",
            dispatcher_options={"script_path": self.root, "heartbeat_seconds": 5,
                                "stale_after": timedelta(minutes=10), "poll_seconds": 1}))
        with client:
            # 293c: the full configuration; the service ignores its server-owned
            # parts (roles included), the dataset gets the roles from it.
            project, _, configuration = _setup(
                client, data=data, roles=self.config._config_to_json(username=""))
            created = client.post(f"/api/projects/{project}/runs",
                                  json={"configuration_id": configuration})
            self.assertEqual(created.status_code, 201, created.text)
            run_id = created.json()["run_id"]
            deadline, status = time.time() + 900, None
            while time.time() < deadline:
                status = client.get(f"/api/runs/{run_id}").json()["status"]
                if status in ("succeeded", "failed", "cancelled", "interrupted"):
                    break
                time.sleep(1)
            log = work / "run-logs" / f"run_{run_id}.log"
            tail = log.read_bytes().decode("utf-8", "replace")[-3000:] if log.exists() else ""
            self.assertEqual(status, "succeeded", tail)
            model_id = client.app.state.jbg["store"].get(RunReference(run_id)).spec[
                "project_run"]["model_id"]
            # 288c: the worker registered the model in the catalog. It does so
            # right after marking the run succeeded (the catalog accepts only
            # succeeded runs), so wait briefly instead of racing it (296a).
            deadline, models = time.time() + 30, []
            while time.time() < deadline:
                models = client.get(f"/api/projects/{project}/models").json()
                if models:
                    break
                time.sleep(0.2)
            self.assertEqual([(m["run_id"], m["state"]) for m in models], [(run_id, "available")])
            # 295a: the run's own folder holds its details and class names.
            folder = work / "runs" / project / run_id
            self.assertTrue((folder / "class_labels.json").is_file(), "no class names")
            self.assertTrue(list((folder / "output" / "csvs").glob("crossval_*.csv")))
            leaderboard = client.get(f"/api/runs/{run_id}/details/leaderboard").json()
            self.assertGreaterEqual(leaderboard["total_rows"], 1)
            labels = client.get(f"/api/runs/{run_id}/class-labels").json()["class_labels"]
            self.assertGreaterEqual(len(labels), 2)
            dispatcher = client.app.state.jbg["dispatcher"]
            for _ in range(50):                      # release the run log before cleanup
                dispatcher.poll_once()
                if not dispatcher.active_runs():
                    break
                time.sleep(0.2)
        self.assertTrue((work / "models" / project / f"{model_id}.sav").is_file(),
                        "the run did not save its own model file")
