"""297e: a CSV model may predict another CSV dataset of its project (opt-in,
runtime only: Config.model_source), and the live view says why a run failed."""
import copy
import csv
import io
import re
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")

from Config import Config  # noqa: E402
from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGModelSourceIdentity import check_model_source_identity  # noqa: E402

A = ("11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222")
B = ("33333333-3333-4333-8333-333333333333", "44444444-4444-4444-8444-444444444444")


def _csv(identity):
    return Config.DataSource(kind="csv", dataset_id=identity[0], artifact_id=identity[1])


def _configs(saved=A, data=B, model_source=None):
    saved_config = SimpleNamespace(data_source=_csv(saved))
    runtime = SimpleNamespace(data_source=_csv(data), model_source=model_source)
    return saved_config, runtime


# --- The identity rule ---------------------------------------------------------------------------

def test_without_a_model_source_the_exact_identity_rule_is_unchanged():
    saved, runtime = _configs()
    with pytest.raises(ValueError, match="different dataset"):
        check_model_source_identity(saved, runtime)
    check_model_source_identity(*_configs(data=A))                    # its own dataset: fine


def test_with_a_model_source_the_model_is_checked_against_it():
    check_model_source_identity(*_configs(model_source=_csv(A)))      # B's rows, A's model


def test_the_model_source_must_be_the_models_own_dataset():
    with pytest.raises(ValueError, match="different dataset"):
        check_model_source_identity(*_configs(model_source=_csv(B)))


def test_a_model_source_applies_to_csv_only():
    saved, runtime = _configs(model_source=_csv(A))
    runtime.data_source = Config.DataSource()                          # SQL
    with pytest.raises(ValueError, match="CSV predictions only"):
        check_model_source_identity(saved, runtime)


def test_the_model_source_is_runtime_only():
    config = Config()
    config.model_source = _csv(A)
    assert "model_source" not in config._config_to_json(username="")
    assert "model_source" not in config.to_dict()
    assert config.get_clean_config().model_source is None              # not in model files
    assert Config().model_source is None                               # off by default


# --- The worker sets it from the server-built run metadata ---------------------------------------

def _record(project_run, source=DataSourceKind.CSV):
    return SimpleNamespace(source=source, spec={"project_run": project_run})


def test_the_worker_sets_the_model_source_of_a_csv_prediction_run():
    from JBGRunWorker import apply_model_source
    config = Config()
    record = _record({"kind": "prediction",
                      "model_source": {"dataset_id": A[0], "artifact_id": A[1]}})
    assert apply_model_source(record, config) is True
    assert (config.model_source.kind, config.model_source.dataset_id,
            config.model_source.artifact_id) == ("csv", A[0], A[1])
    plain = Config()
    assert apply_model_source(_record({"kind": "prediction"}), plain) is False
    assert plain.model_source is None


@pytest.mark.parametrize("project_run, source", [
    ({"kind": "training", "model_source": {"dataset_id": A[0], "artifact_id": A[1]}},
     DataSourceKind.CSV),
    ({"kind": "prediction", "model_source": {"dataset_id": A[0], "artifact_id": A[1]}},
     DataSourceKind.SQL),
    ({"kind": "prediction", "model_source": {"dataset_id": A[0]}}, DataSourceKind.CSV),
    ({"kind": "prediction", "model_source": {"dataset_id": "x", "artifact_id": "y"}},
     DataSourceKind.CSV),
    ({"kind": "prediction", "model_source": "A"}, DataSourceKind.CSV),
])
def test_a_malformed_model_source_is_refused(project_run, source):
    from JBGRunWorker import apply_model_source
    with pytest.raises(Exception):
        apply_model_source(_record(project_run, source), Config())


# --- Why a run failed ---------------------------------------------------------------------------

@pytest.mark.parametrize("status, error_type, code", [
    (RunStatus.FAILED, "NoModelTrained", "no_model"),
    (RunStatus.FAILED, "SystemExit", "engine_stopped"),
    (RunStatus.FAILED, "WorkerStartFailed", "worker_start"),
    (RunStatus.FAILED, "ConfigException", "model_mismatch"),
    (RunStatus.FAILED, "KeyError", "other"),
    (RunStatus.FAILED, None, "other"),
    (RunStatus.INTERRUPTED, None, "interrupted"),
])
def test_failures_have_fixed_codes(status, error_type, code):
    from JBGRunOverview import project_run_overview
    assert project_run_overview("r", status, [], error_type=error_type)["failure"] == code


@pytest.mark.parametrize("status", [RunStatus.SUCCEEDED, RunStatus.RUNNING, RunStatus.PENDING,
                                    RunStatus.CANCELLED])
def test_other_runs_have_no_failure(status):
    from JBGRunOverview import project_run_overview
    assert "failure" not in project_run_overview("r", status, [], error_type="SystemExit")


def _client(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("jinja2")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject="developer",
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _failed(client, kind, status, error_type):
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    store = client.app.state.jbg["store"]
    run = store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "kind": kind}})
    store.claim(run, "w")
    store.finish(run, "w", status, error_type=error_type)
    return run


@pytest.mark.parametrize("kind, path, status, error_type, text", [
    ("training", "/ui/run", RunStatus.FAILED, "NoModelTrained",
     "Ingen modell tränades; loggen visar varför."),
    ("training", "/ui/run", RunStatus.FAILED, "SystemExit",
     "Programmet avbröt körningen; loggen visar varför."),
    ("prediction", "/ui/prediction", RunStatus.FAILED, "ConfigException",
     "Modellen passar inte de här data (kolumner eller roller); loggen säger mer."),
    ("prediction", "/ui/prediction", RunStatus.FAILED, "OSError",
     "Körningen misslyckades; loggen visar varför."),
])
def test_the_live_view_says_why_a_run_failed(tmp_path, kind, path, status, error_type, text):
    with _client(tmp_path) as client:
        run = _failed(client, kind, status, error_type)
        page = client.get(f"{path}?run={run.key}").text
        overview = client.get(f"/api/runs/{run.key}/overview").json()
    assert f'<p class="message error run-failure">{text}</p>' in page
    assert overview["failure"] and error_type not in str(overview)     # codes, not class names


def test_a_successful_run_shows_no_failure(tmp_path):
    with _client(tmp_path) as client:
        run = _failed(client, "training", RunStatus.SUCCEEDED, None)
        page = client.get(f"/ui/run?run={run.key}").text
    assert "run-failure" not in page


# --- A real CSV model predicts another dataset (native) -----------------------------------------

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class CSVModelSourceNative297e(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_a_csv_model_predicts_another_dataset_of_its_project(self):
        from fastapi.testclient import TestClient
        from JBGApplicationContracts import RunReference
        from JBGFastAPIServer import create_server_app
        from JBGLogger import JBGLogger
        from JBGMetadataStore import MetadataStoreSettings
        from JBGProjectRuns import _load_config
        from JBGProjectStorage import project_csv_storage, csv_upload_format
        from Storage.ArtifactContract import ArtifactKind
        from Storage.CSVApplicationRuntime import prepare_registered_csv_prediction_classifier
        from Storage.CSVModelConfiguration import save_csv_model_configuration
        from Storage.CSVPredictionExport import list_csv_prediction_runs
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import CSVPredictionNative296b, _wait_for_workers_to_exit
        self._configure_training()
        self.config.mail.notification_email = ""
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            text = stream.read().decode("utf-8-sig")
        dialect = csv.Sniffer().sniff(text.splitlines()[0])
        rows = list(csv.reader(io.StringIO(text), dialect))
        class_at = rows[0].index(self.config.connection.class_column)
        # B: the same columns, five rows, none with a class; new IDs.
        id_at = rows[0].index(self.config.connection.id_column)
        new_rows = [list(row) for row in rows[1:6]]
        for number, row in enumerate(new_rows):
            row[class_at], row[id_at] = "", f"9{number:03d}"
        other = io.StringIO()
        csv.writer(other, dialect).writerows([rows[0]] + new_rows)
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
            run_id = training.json()["run_id"]
            CSVPredictionNative296b._wait(self, client, run_id, work)
            _wait_for_workers_to_exit(client)
            client.post(f"/api/projects/{project}/datasets/csv", data={"name": "B", "role": "prediction"},
                        files={"file": ("b.csv", other.getvalue().encode("utf-8"), "text/csv")})
            record = client.app.state.jbg["store"].get(RunReference(run_id))
        trained = _load_config(record.spec["configuration"])
        repository, scope = project_csv_storage(project, root=work / "csv")
        b_artifact = next(ref for ref in repository.list_artifacts(scope)
                          if ref.kind == ArtifactKind.DATASET
                          and ref.artifact_id != trained.data_source.artifact_id)
        csv_format = csv_upload_format(repository, scope, b_artifact.artifact_id)
        connection = trained.connection
        text_columns = tuple(connection.data_text_columns)
        registered = save_csv_model_configuration(
            repository, scope, b_artifact.artifact_id, id_column=connection.id_column,
            class_column=connection.class_column, csv_format=csv_format,
            data_columns=tuple(c for c in connection.data_numerical_columns
                               if c not in text_columns), text_columns=text_columns)

        def runtime(model_source):
            config = copy.deepcopy(trained)
            config.mode.train, config.mode.predict = False, True
            config.mode.mispredicted = config.mode.use_metas = False
            config.mode.calculate_feature_importance = config.mode.calculate_dark_numbers = False
            config.data_source = Config.DataSource(
                kind="csv", dataset_id=b_artifact.dataset_id,
                artifact_id=b_artifact.artifact_id, **csv_format)
            config.model_source = model_source
            config.output_path = work / "out"
            return config

        logger = JBGLogger(quiet=True, in_terminal=True, log_dir=work)
        try:
            self._predict_b(logger, trained, runtime, repository, scope, registered, b_artifact,
                            prepare_registered_csv_prediction_classifier,
                            list_csv_prediction_runs, work)
        finally:
            # The logger keeps its log file open; Windows cannot delete an open
            # file, so the temporary folder's cleanup failed (gate 2026-11-11).
            logger.log_file.close()

    def _predict_b(self, logger, trained, runtime, repository, scope, registered, b_artifact,
                   prepare_registered_csv_prediction_classifier, list_csv_prediction_runs,
                   work):
        model = str(trained.get_model_filename())
        # Without the opt-in, the exact-identity rule refuses B (as in Voilà).
        with self.assertRaises(Exception):
            prepare_registered_csv_prediction_classifier(
                model, runtime(None), repository, scope, registered.configuration_id, logger)
        # A model source that is not the model's own dataset is refused too.
        with self.assertRaises(Exception):
            prepare_registered_csv_prediction_classifier(
                model, runtime(copy.deepcopy(runtime(None).data_source)), repository, scope,
                registered.configuration_id, logger)
        classifier = prepare_registered_csv_prediction_classifier(
            model, runtime(copy.deepcopy(trained.data_source)), repository, scope,
            registered.configuration_id, logger)
        self.assertEqual(classifier.config.data_source.artifact_id, b_artifact.artifact_id)
        self.assertEqual(classifier.config.model_source.artifact_id,
                         trained.data_source.artifact_id)
        classifier.config.output_path = work / "out"
        classifier.run()
        published = list_csv_prediction_runs(repository, scope, b_artifact.artifact_id)
        self.assertEqual(len(published), 1)                               # one run for B
        self.assertIn("| 5 predictions |", published[0][0])             # B's five rows
        # The training dataset's own predictions are untouched.
        self.assertEqual(list_csv_prediction_runs(repository, scope,
                                                  trained.data_source.artifact_id), ())
