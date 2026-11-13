"""299b: correcting mispredicted rows of CSV training runs (agreed 2026-11-12;
the Voila GUI's CSV corrections were never connected to runs), and Dark
Number estimates as percentages (asked 2026-11-12).

A confirmed correction stores a new version of the dataset's current file
with the classes changed; the dataset points to it, so every reader sees the
corrected rows. The uploaded file and earlier versions stay (runs and models
refer to them). Row IDs are compared as written ("0036" is not 36)."""
import csv
import io
import json
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

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGReclassification import CorrectionFailed  # noqa: E402
from JBGWebUIResults import percent_text, with_percentages  # noqa: E402

CSV = "nr;x;art\n0036;1,5;katt\n36;2,5;katt\n0009;3,5;hund\n"
MISPLACED = ("nr;P(hund);P(katt);Actual;Predicted;Model;x\n"
             "0036;0,8;0,2;katt;hund;cross trained;1,5\n"
             "0009;0,1;0,9;hund;katt;cross trained;3,5\n")


def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _owner():
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    return AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer")


def _setting(client, tmp_path, misplaced=MISPLACED):
    from JBGConfigurationService import _server_template
    location = client.post("/ui/projects", data={"name": "C", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "Djur å", "role": "training"},
                          files={"file": ("d.csv", CSV.encode("utf-8"), "text/csv")}).json()["dataset_id"]
    settings = _server_template()
    settings["connection"].update(id_column="nr", class_column="art")
    store = client.app.state.jbg["store"]
    run = store.create(_owner().principal_key, DataSourceKind.CSV, {
        "spec_version": 1, "configuration": settings,
        "project_run": {"project_id": project, "dataset_id": dataset, "kind": "training",
                        "model_id": "m", "configuration_version": 1}})
    store.claim(run, "w")
    store.finish(run, "w", RunStatus.SUCCEEDED, result={})
    folder = tmp_path / "runs" / project / run.key
    (folder / "output" / "csvs").mkdir(parents=True)
    (folder / "output" / "csvs" / "misplaced_r__train.csv").write_text(misplaced, encoding="utf-8")
    (folder / "class_labels.json").write_text(json.dumps({"class_labels": ["hund", "katt"]}))
    return SimpleNamespace(project=project, dataset=dataset, run=run.key, folder=folder)


def _registration(client, s):
    return client.app.state.jbg["datasets"].get_owned(_owner().principal_key, s.project, s.dataset)


def _rows(client, s):
    content = client.get(f"/ui/dataset/{s.dataset}/download").content
    return list(csv.reader(io.StringIO(content.decode("utf-8-sig")), delimiter=";"))


# --- Correcting the file ------------------------------------------------------------------------

def test_a_confirmed_correction_makes_a_new_file_version(tmp_path):
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        before = _registration(client, s)
        client.post(f"/api/projects/{s.project}/datasets/{s.dataset}/validate")
        written = client.post(f"/ui/results/{s.run}/corrections", data={"corr:0036": "hund"},
                              follow_redirects=False)
        after = _registration(client, s)
        rows = _rows(client, s)
        page = client.get(written.headers["location"]).text
    assert "message=corrected_csv" in written.headers["location"]
    assert after.registration_id != before.registration_id
    assert after.validation_state.value == "unverified"              # the data changed
    assert rows[1] == ["0036", "1,5", "hund"] and rows[2] == ["36", "2,5", "katt"]   # IDs as written
    assert "Korrigeringarna är sparade i en ny version av datasetets fil." in page
    assert "korrigerad till hund" in page


def test_the_uploaded_file_is_kept(tmp_path):
    from JBGProjectStorage import project_csv_storage
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        original = _registration(client, s).registration_id
        client.post(f"/ui/results/{s.run}/corrections", data={"corr:0036": "hund"})
        repository, scope = project_csv_storage(s.project, root=tmp_path / "csv")
        with repository.open_artifact(scope, original) as stream:
            assert stream.read().decode("utf-8") == CSV


def test_undo_makes_another_version_with_the_old_class(tmp_path):
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        client.post(f"/ui/results/{s.run}/corrections", data={"corr:0036": "hund"})
        corrected = _registration(client, s).registration_id
        undone = client.post(f"/ui/results/{s.run}/corrections/0/undo", follow_redirects=False)
        rows = _rows(client, s)
        latest = _registration(client, s).registration_id
    assert "message=undone" in undone.headers["location"]
    assert rows[1] == ["0036", "1,5", "katt"]
    assert latest not in (corrected,)                                   # another version


def test_a_row_missing_from_the_file_changes_nothing(tmp_path):
    misplaced = MISPLACED + "0077;0,5;0,5;katt;hund;cross trained;9\n"
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path, misplaced)
        before = _registration(client, s).registration_id
        failed = client.post(f"/ui/results/{s.run}/corrections",
                             data={"corr:0036": "hund", "corr:0077": "hund"}, follow_redirects=False)
        page = client.get(failed.headers["location"]).text
        after = _registration(client, s).registration_id
        log = s.folder / "corrections.json"
    assert "message=correction_failed_csv" in failed.headers["location"]
    assert before == after and not log.exists()
    assert "Datasetets fil kunde inte uppdateras, så inget ändrades." in page


def test_the_review_and_the_page_speak_of_the_file(tmp_path):
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        page = client.get(f"/ui/results?run={s.run}&view=mispredictions").text
        review = client.post(f"/ui/results/{s.run}/corrections/review", data={"corr:0036": "hund"})
    assert 'name="corr:0036"' in page and "innan de sparas i datasetets fil" in page
    assert '<td class="">0036</td>' in page                             # shown as written
    assert "sparas i en ny version av datasetets CSV-fil" in review.text
    assert "SQL Server" not in review.text.split("Bekräfta korrigeringarna")[1].split("</form>")[0]


def test_the_dataset_tab_offers_the_current_file(tmp_path):
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        client.post(f"/ui/results/{s.run}/corrections", data={"corr:0036": "hund"})
        page = client.get(f"/ui/dataset?dataset={s.dataset}").text
        download = client.get(f"/ui/dataset/{s.dataset}/download")
    assert f'href="/ui/dataset/{s.dataset}/download">Hämta CSV</a>' in page
    assert download.headers["content-type"].startswith("text/csv")
    assert "filename*=UTF-8''Djur%20%C3%A5.csv" in download.headers["content-disposition"]
    assert "0036;1,5;hund" in download.content.decode("utf-8-sig")


# --- Dark Numbers in percent --------------------------------------------------------------------

def test_percent_texts():
    assert percent_text(0.001042, "sv") == "0,10 %" and percent_text(0.01432, "en") == "1.43 %"
    assert percent_text(0.0, "sv") == "0,00 %" and percent_text(0.00001, "sv") == "< 0,01 %"


def test_the_estimates_get_percentages_and_bars():
    summary = [{"points": [{"value": 0.001042}, {"value": 0.004082}, {"value": None},
                           {"value": 0.01432}]}]
    points = with_percentages(summary, "sv")[0]["points"]
    assert [p["percent"] for p in points] == ["0,10 %", "0,41 %", None, "1,43 %"]
    assert [p["bar"] for p in points] == [7.3, 28.5, 0.0, 100.0]
    assert summary[0]["range_text"] == "0,10 % – 1,43 %"


def test_the_dark_numbers_view_shows_percentages(tmp_path):
    calculations = (";Model type;type;target;alphas;corr;corr_model;corr_source;dark_number\n"
                    "0;D_cv_test - Cross-trained model;single_alpha;a;0,5;2,0;Cross-trained model;estimated;0,01432\n"
                    "0;D_cv_full - Cross-trained model;single_alpha;a;0,5;2,0;Cross-trained model;estimated;0,004082\n"
                    "0;D_retrained_full - Retrained model;single_alpha;a;0,5;2,0;Retrained model;estimated;0,001042\n"
                    "0;Combined (legacy full-data);single_alpha;a;0,5;2,0;Cross-trained model;estimated;0,004427\n")
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        (s.folder / "output" / "csvs" / "dark_numbers_r__train.csv").write_text(calculations, encoding="utf-8")
        page = client.get(f"/ui/results?run={s.run}&view=dark_numbers").text
    assert "<th>I procent</th>" in page
    assert re.findall(r'class="dark-percent">([^<]+)<', page) == ["0,10 %", "0,41 %", "0,44 %", "1,43 %"]
    assert "Intervall: <b>0,10 % – 1,43 %</b>" in page
    assert 'style="width: 100.0%"' in page


# --- A real run (native) ---------------------------------------------------------------------------

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class CSVReclassificationNative299b(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_a_correction_reaches_the_next_training(self):
        from JBGFastAPIServer import create_server_app
        from JBGMetadataStore import MetadataStoreSettings
        self._configure_training()
        self.config.mail.notification_email = ""
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            text = stream.read().decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        target = rows[0].index("target")
        for number, row in enumerate(rows[1:]):                  # a few wrong labels
            if number % 9 == 0 and row[target]:
                row[target] = {"a": "b", "b": "c", "c": "a"}[row[target]]
        out = io.StringIO()
        csv.writer(out).writerows(rows)
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
                self._correct_and_retrain(client, work, out.getvalue().encode("utf-8"))
            finally:
                from test_revision_296b_csv_predictions import _wait_for_workers_to_exit
                _wait_for_workers_to_exit(client)

    def _correct_and_retrain(self, client, work, data):
        from JBGApplicationContracts import RunReference
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import CSVPredictionNative296b
        settings = self.config._config_to_json(username="")
        settings["mode"].update(predict=False, mispredicted=True)
        project, dataset, configuration = _setup(client, data=data, roles=settings)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        first = client.post(f"/api/projects/{project}/runs",
                            json={"configuration_id": configuration}).json()["run_id"]
        CSVPredictionNative296b._wait(self, client, first, work)
        deadline, page = time.time() + 30, ""
        while time.time() < deadline and 'name="corr:' not in page:
            page = client.get(f"/ui/results?run={first}&view=mispredictions").text
            time.sleep(0.3)
        row_id, = re.findall(r'name="corr:([^"]+)"', page)[:1]
        labels = ["a", "b", "c"]
        written = client.post(f"/ui/results/{first}/corrections",
                              data={f"corr:{row_id}": labels[0]}, follow_redirects=False)
        if "correction_refused" in written.headers["location"]:          # already "a"
            written = client.post(f"/ui/results/{first}/corrections",
                                  data={f"corr:{row_id}": labels[1]}, follow_redirects=False)
        self.assertIn("message=corrected_csv", written.headers["location"])
        new_class = json.loads(next((work / "runs").rglob("corrections.json")).read_text(
            encoding="utf-8"))[-1]["new"]
        registration = client.app.state.jbg["datasets"].get_owned(
            client.app.state.jbg["store"].get(RunReference(first)).owner, project, dataset).registration_id
        second = client.post(f"/api/projects/{project}/runs",
                             json={"configuration_id": configuration}).json()["run_id"]
        CSVPredictionNative296b._wait(self, client, second, work)
        spec = client.app.state.jbg["store"].get(RunReference(second)).spec
        self.assertEqual(spec["csv"]["artifact_id"], registration)      # the corrected file
        corrected = client.get(f"/ui/dataset/{dataset}/download").content.decode("utf-8-sig")
        table = list(csv.reader(io.StringIO(corrected)))
        id_at, class_at = table[0].index("row_id"), table[0].index("target")
        self.assertEqual(next(r for r in table[1:] if r[id_at] == row_id)[class_at], new_class)
