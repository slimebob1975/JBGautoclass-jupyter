"""296b: CSV predictions with a saved model; predictions kept per run."""
import csv
import io
import json
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from Config import Config


# --- The engine keeps predictions in a run's own folder ----------------------------------------

def _handler(output_path):
    from JBGHandler import PredictionsHandler
    config = Config()
    config.connection.id_column = "id"
    config.output_path = output_path
    handler = PredictionsHandler.__new__(PredictionsHandler)
    handler.handler = SimpleNamespace(config=config)
    return handler, config


RESULTS = [{"key": "7", "prediction": "setosa", "rate": 0.91, "probabilities": "0.91,0.06,0.03"},
           {"key": "8", "prediction": "virginica", "rate": 0.55, "probabilities": "0.1,0.35,0.55"}]


def test_predictions_are_kept_in_the_run_folder_with_class_columns(tmp_path):
    handler, config = _handler(tmp_path)
    handler._keep_predictions_in_run_folder(RESULTS, ["setosa", "versicolor", "virginica"])
    frame = pd.read_csv(config.get_output_filepath("predictions"), sep=";", decimal=",", dtype={"id": str})
    assert list(frame.columns) == ["id", "Predicted", "Rate", "P(setosa)", "P(versicolor)", "P(virginica)"]
    assert frame["id"].tolist() == ["7", "8"] and frame["P(virginica)"].tolist() == [0.03, 0.55]


def test_without_a_run_folder_nothing_is_written(tmp_path):
    handler, config = _handler(None)                       # the Voilà GUI
    config.script_path = tmp_path
    handler._keep_predictions_in_run_folder(RESULTS, ["setosa", "versicolor", "virginica"])
    assert not (tmp_path / "output").exists()


def test_the_base_logger_reports_csv_predictions_without_a_query(tmp_path):
    from JBGLogger import JBGLogger
    lines = []
    logger = JBGLogger(quiet=False, in_terminal=True, log_dir=tmp_path)   # not the repository
    with patch.object(logger, "writeln", side_effect=lambda kind, text: lines.append(text)):
        logger.print_prediction_results({"artifact_id": "a", "run_id": "r", "row_count": 3})
    assert lines == ["Saved 3 CSV predictions to the CSV artifact store."]


def test_the_csv_runtime_predicts_with_the_file_name_as_text_and_the_run_folder(tmp_path):
    from JBGApplicationContracts import DataSourceKind, DatasetReference
    from JBGProjectStorage import project_csv_storage
    from JBGRunExecutionService import RegisteredCSVClassifierRuntime
    from JBGRunPreparationService import RunConfiguration
    from Storage.ArtifactContract import ArtifactKind, ArtifactReference
    repository, scope = project_csv_storage("11111111-1111-4111-8111-111111111111", root=tmp_path)
    from uuid import uuid4
    artifact, dataset = str(uuid4()), str(uuid4())
    runtime = RegisteredCSVClassifierRuntime(repository, scope, "cfg", object())
    config = Config()
    config.mode.train, config.mode.predict = False, True
    config.data_source.kind, config.data_source.artifact_id, config.data_source.dataset_id = "csv", artifact, dataset
    config.output_path = tmp_path / "run"
    reference = ArtifactReference(scope=scope, artifact_id=artifact, kind=ArtifactKind.DATASET,
                                  content_type="text/csv", dataset_id=dataset, run_id=None)
    prepared = SimpleNamespace(dataset=DatasetReference(DataSourceKind.CSV, artifact),
                               configuration=RunConfiguration(DataSourceKind.CSV, config),
                               resolved_dataset=reference)
    restored, seen = SimpleNamespace(output_path=None), {}

    def prepare(filename, *args):
        seen["filename"] = filename
        return SimpleNamespace(config=restored, run=lambda: "ran")

    with patch("Storage.CSVApplicationRuntime.prepare_registered_csv_prediction_classifier", prepare):
        assert runtime.execute(prepared) == "ran"
    # The CSV boundary takes the file name as text; the run folder reaches the
    # configuration the classifier actually runs with.
    assert isinstance(seen["filename"], str)
    assert restored.output_path == tmp_path / "run"


# --- A real CSV prediction through the API (native: trains and predicts) -------------------------------

pytest.importorskip("fastapi")

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402
from test_revision_288a_project_runs import _setup  # noqa: E402


def _wait_for_workers_to_exit(client, seconds=120):
    """A worker reports "succeeded" and then exits; until then it holds its log
    file open (its output goes there). Windows cannot delete open files, so the
    temporary folder's cleanup failed with WinError 32 when the test ended
    first (gate 2026-11-09). Wait until the dispatcher has collected them."""
    dispatcher = client.app.state.jbg["dispatcher"]
    deadline = time.time() + seconds
    while time.time() < deadline:
        dispatcher.poll_once()                  # collects workers that have exited
        if not dispatcher.active_runs():
            return
        time.sleep(0.5)
    raise AssertionError("a worker did not exit in time")


class CSVPredictionNative296b(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def _wait(self, client, run_id, work):
        deadline, status = time.time() + 900, None
        while time.time() < deadline:
            status = client.get(f"/api/runs/{run_id}").json()["status"]
            if status in ("succeeded", "failed", "cancelled", "interrupted"):
                break
            time.sleep(1)
        log = work / "run-logs" / f"run_{run_id}.log"
        tail = log.read_bytes().decode("utf-8", "replace")[-3000:] if log.exists() else ""
        self.assertEqual(status, "succeeded", tail)

    def test_a_csv_model_predicts_its_training_datasets_rows_without_a_class(self):
        from fastapi.testclient import TestClient
        from JBGFastAPIServer import create_server_app
        from JBGMetadataStore import MetadataStoreSettings
        self._configure_training()
        self.config.mail.notification_email = ""
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            text = stream.read().decode("utf-8-sig")
        # Remove the class of three rows: these are the rows to predict.
        dialect = csv.Sniffer().sniff(text.splitlines()[0])
        rows = list(csv.reader(io.StringIO(text), dialect))
        class_at = rows[0].index(self.config.connection.class_column)
        id_at = rows[0].index(self.config.connection.id_column)
        for row in rows[-3:]:
            row[class_at] = ""
        # Every row without a class is predicted: the fixture already has some.
        unlabelled = [row[id_at] for row in rows[1:] if not row[class_at].strip()]
        out = io.StringIO()
        csv.writer(out, dialect).writerows(rows)
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
            settings["mode"]["predict"] = False                   # CSV training is train-only
            project, _, configuration = _setup(client, data=out.getvalue().encode("utf-8"),
                                               roles=settings)
            training = client.post(f"/api/projects/{project}/runs",
                                   json={"configuration_id": configuration})
            self.assertEqual(training.status_code, 201, training.text)
            self._wait(client, training.json()["run_id"], work)
            deadline, models = time.time() + 30, []
            while time.time() < deadline and not models:
                models = client.get(f"/api/projects/{project}/models").json()
                time.sleep(0.2)
            prediction = client.post(f"/api/projects/{project}/predictions",
                                     json={"model_id": models[0]["model_id"]})
            self.assertEqual(prediction.status_code, 201, prediction.text)
            run_id = prediction.json()["run_id"]
            self._wait(client, run_id, work)
            download = client.get(f"/api/runs/{run_id}/predictions.csv")
            self.assertEqual(download.status_code, 200)
            frame = pd.read_csv(io.StringIO(download.text), sep=";", decimal=",", dtype=str)
            self.assertEqual(sorted(frame.iloc[:, 0]), sorted(unlabelled))
            self.assertTrue(all(column.startswith("P(") for column in frame.columns[3:]))
            # A prediction run registers no model of its own.
            self.assertEqual(len(client.get(f"/api/projects/{project}/models").json()), 1)
            _wait_for_workers_to_exit(client)


# --- The Prediction tab shows and offers the predictions --------------------------------------------

def _client(tmp_path, subject="developer"):
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


def test_the_prediction_tab_shows_and_offers_the_predictions(tmp_path):
    import re
    from JBGApplicationContracts import DataSourceKind, RunStatus
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    with _client(tmp_path) as client:
        location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        store = client.app.state.jbg["store"]
        run = store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
            "project_id": project, "kind": "prediction"}})
        store.claim(run, "w")
        store.finish(run, "w", RunStatus.SUCCEEDED, result={"schema_version": 1})
        folder = tmp_path / "runs" / project / run.key / "output" / "csvs"
        folder.mkdir(parents=True)
        lines = "".join(f"{i};a;0,9;0,9;0,1\n" for i in range(60))
        (folder / "predictions_p__predict.csv").write_text("id;Predicted;Rate;P(a);P(b)\n" + lines,
                                                          encoding="utf-8")
        page = client.get(f"/ui/prediction?run={run.key}").text
        assert "Ladda ned prediktionerna" in page and "<th>P(a)</th>" in page
        assert "Sida 1 / 2" in page and "<td class=\"num\">0,9</td>" in page
        assert "prediktionstabellen i SQL Server" not in page              # CSV project
        # 297g: the download starts with a UTF-8 byte-order mark (for Excel).
        assert client.get(f"/api/runs/{run.key}/predictions.csv").content.startswith(
            b"\xef\xbb\xbfid;Predicted")
    with _client(tmp_path, subject="bob") as theirs:
        assert theirs.get(f"/api/runs/{run.key}/predictions.csv").status_code == 404
