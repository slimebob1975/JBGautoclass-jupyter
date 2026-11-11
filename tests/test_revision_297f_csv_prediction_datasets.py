"""297f: CSV projects predict another CSV dataset of the project (decision 4
changed 2026-11-10): prediction and mixed datasets whose columns fit the model;
the class column may be missing (every row is predicted)."""
import csv
import io
import re
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

FORMAT = {"delimiter": ";", "quotechar": '"', "encoding": "utf-8-sig"}
ROLES = dict(id_column="id", class_column="species", numeric_columns=("sepal",),
             text_columns=("colour",))


def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject="developer",
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _stored(tmp_path, data: str):
    """A CSV upload in real project storage: (repository, scope, artifact_id)."""
    from JBGProjectStorage import project_csv_storage
    from Storage.ArtifactContract import ArtifactKind
    with _client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "P", "source": "csv"}).json()["project_id"]
        client.post(f"/api/projects/{project}/datasets/csv", data={"name": "B", "role": "prediction"},
                    files={"file": ("b.csv", data.encode("utf-8"), "text/csv")})
    repository, scope = project_csv_storage(project, root=tmp_path / "csv")
    artifact = next(ref.artifact_id for ref in repository.list_artifacts(scope)
                    if ref.kind == ArtifactKind.DATASET)
    return repository, scope, artifact


def _check(tmp_path, data):
    from JBGProjectStorage import check_csv_prediction_columns
    repository, scope, artifact = _stored(tmp_path, data)
    return check_csv_prediction_columns(repository, scope, artifact, FORMAT, **ROLES)


# --- Does a file fit the model? ------------------------------------------------------------------

@pytest.mark.parametrize("data, has_class", [
    ("id;sepal;colour;extra\n1;5.1;red;x\n2;7,0;blue;y\n", False),        # no class column
    ("id;sepal;colour;species\n1;5.1;red;a\n2;7.0;blue;\n", True),         # one row to predict
    ("id;colour;sepal;species\n1;red;5.1;NULL\n2;blue;6;\n", True),        # other order, NULL
])
def test_files_that_fit(tmp_path, data, has_class):
    assert _check(tmp_path, data) is has_class


@pytest.mark.parametrize("data, missing, different", [
    ("id;colour;species\n1;red;\n2;blue;\n", ("sepal",), ()),
    ("id;sepal;species\n1;5.1;\n2;7.0;\n", ("colour",), ()),
    ("id;sepal;colour\n1;n/a;red\n2;7.0;blue\n", (), ("sepal",)),         # not numbers here
    ("id;sepal;colour\n1;5.1;red\n1;7.0;blue\n", (), ("id",)),            # ID not unique
    ("id;sepal;colour\n1;5.1;red\n;7.0;blue\n", (), ("id",)),             # ID not filled in
])
def test_files_that_do_not_fit_are_named_by_column(tmp_path, data, missing, different):
    from JBGDatasetRoles import IncompatibleDataset
    with pytest.raises(IncompatibleDataset) as refused:
        _check(tmp_path, data)
    assert (refused.value.missing, refused.value.different) == (missing, different)


def test_a_file_where_every_row_has_a_class_has_nothing_to_predict(tmp_path):
    with pytest.raises(ValueError, match="nothing to predict"):
        _check(tmp_path, "id;sepal;colour;species\n1;5.1;red;a\n2;7.0;blue;b\n")


# --- The reader adds the class column only with the opt-in ----------------------------------------

def _reader(tmp_path, *, model_source):
    from Config import Config
    from Storage.CSVModelConfiguration import save_csv_model_configuration
    from Storage.CSVTrainingAdapter import CSVTrainingAdapter
    repository, scope, artifact = _stored(tmp_path, "id;sepal;colour\n1;5.1;red\n2;7.0;blue\n")
    registered = save_csv_model_configuration(
        repository, scope, artifact, id_column="id", class_column="species",
        data_columns=("sepal",), text_columns=("colour",), csv_format=FORMAT)
    config = Config()
    config.connection.id_column, config.connection.class_column = "id", "species"
    config.connection.data_text_columns = ["colour"]
    config.connection.data_numerical_columns = ["sepal"]
    config.mode.train, config.mode.predict = False, True
    config.data_source = Config.DataSource(kind="csv", dataset_id=registered.dataset_id,
                                           artifact_id=artifact, **FORMAT)
    config.model_source = model_source
    return CSVTrainingAdapter(repository, scope, registered.configuration_id, config)


def test_with_the_opt_in_a_missing_class_column_means_every_row_is_predicted(tmp_path):
    from Config import Config
    source = Config.DataSource(kind="csv", dataset_id="11111111-1111-4111-8111-111111111111",
                               artifact_id="22222222-2222-4222-8222-222222222222")
    rows = _reader(tmp_path, model_source=source).get_dataset()
    assert rows == [("red", "5.1", "1", ""), ("blue", "7.0", "2", "")]


def test_without_the_opt_in_the_class_column_is_required_as_before(tmp_path):
    with pytest.raises(ValueError, match="Unable to prepare CSV rows"):
        _reader(tmp_path, model_source=None).get_dataset()


# --- The Prediction tab -------------------------------------------------------------------------

def _csv_project(client):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    for name, role in (("Träning", "training"), ("Nya", "prediction"), ("Blandat", "mixed")):
        client.post("/ui/dataset/csv", data={"name": name, "role": role},
                    files={"file": (f"{name}.csv", b"id;x;c\n1;1;a\n2;2;b\n", "text/csv")})
    return project


def test_csv_projects_offer_their_prediction_and_mixed_datasets(tmp_path):
    with _client(tmp_path) as client:
        _csv_project(client)
        page = client.get("/ui/prediction").text
    select = re.search(r'<select id="predict-dataset" name="dataset_id">(.*?)</select>', page, re.S)
    options = re.findall(r'<option value="[^"]*">([^<]+)</option>', select.group(1))
    assert options[0] == "Rader utan klass i träningsdatasetet"
    assert sorted(options[1:]) == ["Blandat", "Nya"]                # not the training dataset
    assert "en fil utan klasskolumn predikteras i sin helhet" in page


@pytest.mark.parametrize("error, text", [
    ("RunDoesNotMatchDataset", "Datasetet passar inte modellen"),
    ("NothingToPredict", "Alla rader i datasetet har en klass"),
])
def test_refusals_say_why(tmp_path, error, text, monkeypatch):
    import JBGProjectRuns
    failing = getattr(JBGProjectRuns, error)

    def refuse(*args, **kwargs):
        raise failing("refused")
    monkeypatch.setattr(JBGProjectRuns.ProjectRunService, "submit_prediction", refuse)
    with _client(tmp_path) as client:
        _csv_project(client)
        client.cookies.set("jbg_model", "00000000-0000-4000-8000-000000000001", path="/ui")
        page = client.post("/ui/prediction", data={"dataset_id": "x"}).text
    assert text in page


# --- Through the API with a real worker (native) --------------------------------------------------

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class CSVPredictionDatasetsNative297f(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_a_csv_model_predicts_other_datasets_of_its_project(self):
        import pandas as pd
        from JBGFastAPIServer import create_server_app
        from JBGMetadataStore import MetadataStoreSettings
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import CSVPredictionNative296b, _wait_for_workers_to_exit
        self._configure_training()
        self.config.mail.notification_email = ""
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            text = stream.read().decode("utf-8-sig")
        dialect = csv.Sniffer().sniff(text.splitlines()[0])
        rows = list(csv.reader(io.StringIO(text), dialect))
        header, body = rows[0], rows[1:]
        class_at = header.index(self.config.connection.class_column)
        id_at = header.index(self.config.connection.id_column)

        def file(header, rows):
            out = io.StringIO()
            csv.writer(out, dialect).writerows([header] + rows)
            return out.getvalue().encode("utf-8")

        # B: no class column at all; every row is predicted.
        b_rows = [[v for i, v in enumerate(row) if i != class_at] for row in body[:4]]
        for number, row in enumerate(b_rows):
            row[id_at if id_at < class_at else id_at - 1] = f"8{number:03d}"
        b_header = [h for i, h in enumerate(header) if i != class_at]
        # C: the class column with two rows without a class.
        c_rows = [list(row) for row in body[:5]]
        for number, row in enumerate(c_rows):
            row[id_at] = f"7{number:03d}"
        c_rows[1][class_at] = c_rows[3][class_at] = ""
        # D: a feature column missing. E: every row has a class.
        feature = next(h for h in header if h not in (header[class_at], header[id_at]))
        d_header = [h for h in header if h != feature]
        d_rows = [[v for h, v in zip(header, row) if h != feature] for row in body[:3]]
        e_rows = [list(row) for row in body[:3] if row[class_at].strip()]
        work = self.root / "api"
        work.mkdir()
        client = TestClient(create_server_app(
            settings=MetadataStoreSettings(sqlite_path=work / "jobs.sqlite"),
            run_logs=work / "run-logs", csv_root=work / "csv", model_root=work / "models",
            results_root=work / "runs",
            dispatcher_options={"script_path": self.root, "heartbeat_seconds": 5,
                                "stale_after": timedelta(minutes=10), "poll_seconds": 1}))
        with client:
            settings = self.config._config_to_json(username="")
            settings["mode"]["predict"] = False
            project, _, configuration = _setup(client, data=text.encode("utf-8"), roles=settings)
            training = client.post(f"/api/projects/{project}/runs",
                                   json={"configuration_id": configuration})
            self.assertEqual(training.status_code, 201, training.text)
            CSVPredictionNative296b._wait(self, client, training.json()["run_id"], work)
            deadline, models = time.time() + 30, []
            while time.time() < deadline and not models:
                models = client.get(f"/api/projects/{project}/models").json()
                time.sleep(0.2)
            model_id = models[0]["model_id"]

            def upload(name, data):
                response = client.post(f"/api/projects/{project}/datasets/csv",
                                       data={"name": name, "role": "prediction"},
                                       files={"file": (f"{name}.csv", data, "text/csv")})
                self.assertEqual(response.status_code, 201, (name, response.text))
                return response.json()["dataset_id"]

            def predict(dataset_id):
                return client.post(f"/api/projects/{project}/predictions",
                                   json={"model_id": model_id, "dataset_id": dataset_id})

            # A missing training column is refused already at upload (decision 4).
            rejected = client.post(f"/api/projects/{project}/datasets/csv",
                                   data={"name": "D", "role": "prediction"},
                                   files={"file": ("D.csv", file(d_header, d_rows), "text/csv")})
            self.assertEqual(rejected.status_code, 409, rejected.text)
            # What the upload does not check is refused when predicting: here a
            # duplicated ID (422, the dataset does not fit the model).
            dup_rows = [list(row) for row in body[:3]]
            for row in dup_rows:
                row[id_at], row[class_at] = "6000", ""
            refused = predict(upload("Dubbel", file(header, dup_rows)))
            self.assertEqual(refused.status_code, 422, refused.text)
            if e_rows:
                nothing = predict(upload("E", file(header, e_rows)))
                self.assertEqual(nothing.status_code, 409, nothing.text)
            for name, data, expected in (
                    ("B", file(b_header, b_rows), [f"8{n:03d}" for n in range(4)]),
                    ("C", file(header, c_rows), ["7001", "7003"])):
                started = predict(upload(name, data))
                self.assertEqual(started.status_code, 201, started.text)
                run_id = started.json()["run_id"]
                CSVPredictionNative296b._wait(self, client, run_id, work)
                download = client.get(f"/api/runs/{run_id}/predictions.csv")
                self.assertEqual(download.status_code, 200, name)
                frame = pd.read_csv(io.StringIO(download.text), sep=";", decimal=",", dtype=str)
                self.assertEqual(sorted(frame.iloc[:, 0]), sorted(expected), name)
            # Prediction runs register no models of their own.
            self.assertEqual(len(client.get(f"/api/projects/{project}/models").json()), 1)
            _wait_for_workers_to_exit(client)
