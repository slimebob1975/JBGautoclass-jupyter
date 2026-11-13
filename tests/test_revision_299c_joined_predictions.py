"""299c: CSV predictions joined with the predicted rows' original columns
(the Voila GUI's "Export joined"), headings of the predictions table in the
interface's language, and Översikt's empty state with the tab name as the link
(asked 2026-11-13)."""
import csv
import io
import re
import time
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def test_overview_without_a_project_links_the_tab_in_the_sentence(tmp_path):
    with _client(tmp_path) as client:
        swedish = client.get("/ui/overview").text
        client.post("/ui/language", data={"language": "en", "back": "/ui/projects"},
                    follow_redirects=False)
        english = client.get("/ui/overview").text
    assert ('<p class="empty">Öppna först ett projekt under fliken '
            '<a href="/ui/projects">Projekt</a>.</p>') in swedish
    assert 'Fliken <a href="/ui/about">Om</a> beskriver hur det används.' in swedish
    assert ('Open a project in the <a href="/ui/projects">Projects</a> tab first.') in english


def test_the_joined_download_is_for_finished_csv_prediction_runs_only(tmp_path):
    from JBGApplicationContracts import DataSourceKind, RunStatus
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    with _client(tmp_path) as client:
        location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        store = client.app.state.jbg["store"]
        owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
        training = store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
            "project_id": project, "kind": "training"}})
        store.claim(training, "w")
        store.finish(training, "w", RunStatus.SUCCEEDED, result={})
        refused = client.get(f"/ui/prediction/{training.key}/joined.csv", follow_redirects=False)
        unknown = client.get("/ui/prediction/00000000-0000-4000-8000-000000000000/joined.csv",
                             follow_redirects=False)
    assert refused.status_code == 303 and "message=joined_unavailable" in refused.headers["location"]
    assert unknown.status_code == 303


class JoinedPredictionsNative299c(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_predictions_with_the_original_columns(self):
        from JBGFastAPIServer import create_server_app
        from JBGMetadataStore import MetadataStoreSettings
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
            try:
                self._check(client, work, data)
            finally:
                from test_revision_296b_csv_predictions import _wait_for_workers_to_exit
                _wait_for_workers_to_exit(client)

    def _check(self, client, work, data):
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import CSVPredictionNative296b
        settings = self.config._config_to_json(username="")
        settings["mode"].update(predict=False)
        project, dataset, configuration = _setup(client, data=data, roles=settings)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        training = client.post(f"/api/projects/{project}/runs",
                               json={"configuration_id": configuration}).json()["run_id"]
        CSVPredictionNative296b._wait(self, client, training, work)
        deadline, models = time.time() + 30, []
        while time.time() < deadline and not models:
            models = client.get(f"/api/projects/{project}/models").json()
            time.sleep(0.2)
        # The training dataset's rows without a class (296b).
        predicted = client.post(f"/api/projects/{project}/predictions",
                                json={"model_id": models[0]["model_id"]})
        self.assertEqual(predicted.status_code, 201, predicted.text)
        run = predicted.json()["run_id"]
        CSVPredictionNative296b._wait(self, client, run, work)
        page = client.get(f"/ui/prediction?run={run}").text
        self.assertIn(f'href="/ui/prediction/{run}/joined.csv"', page)
        self.assertIn("<th>Predikterad klass</th>", page)
        self.assertIn("<th>Säkerhet</th>", page)
        joined = client.get(f"/ui/prediction/{run}/joined.csv")
        self.assertEqual(joined.status_code, 200)
        self.assertTrue(joined.content.startswith(b"\xef\xbb\xbf"))
        table = list(csv.reader(io.StringIO(joined.content.decode("utf-8-sig")), delimiter=";"))
        original = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        header = original[0]
        self.assertEqual(table[0][:len(header)], header)
        self.assertEqual(table[0][len(header)], "Predicted")
        by_id = {row[header.index("row_id")]: row for row in original[1:]}
        self.assertGreater(len(table), 1)
        for row in table[1:]:
            self.assertEqual(row[:len(header)], by_id[row[header.index("row_id")]])
            self.assertEqual(row[header.index("target")], "")          # rows without a class
