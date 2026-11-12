"""298j (noted 2026-11-12): the Results tab shows Dark Numbers (they were
computed and saved in the run folder, but not shown anywhere; the log leaves
them out as they may name classes): per target class the engine's four
estimates in their expected order with a status, the calculation table and
the confusion matrices. Wide tables get a second horizontal scrollbar above
them (jbg.js; checked in headless Chromium)."""
import re
import time
from datetime import timedelta

import pytest

from JBGRunResults import DETAILS, KEEP_FIRST_COLUMN, RunResultsService

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402

CALCULATIONS = (";Model type;type;target;alphas;corr;corr_model;corr_source;dark_number\n"
                "0;D_cv_test - Cross-trained model;single_alpha;a;0,5;2,0;Cross-trained model;estimated;0,3\n"
                "0;D_cv_full - Cross-trained model;single_alpha;a;0,5;2,0;Cross-trained model;estimated;0,2\n"
                "0;D_retrained_full - Retrained model;single_alpha;a;0,5;2,0;Retrained model;estimated;0,1\n"
                "0;Combined (legacy full-data);single_alpha;a;0,5;2,0;Cross-trained model;estimated;0,25\n")
MATRICES = ";Model type;a;b\na;D_cv_test - Cross-trained model;9;1\nb;;2;8\n"


def _folder(tmp_path, calculations=CALCULATIONS):
    csvs = tmp_path / "output" / "csvs"
    csvs.mkdir(parents=True)
    (csvs / "dark_numbers_r__train.csv").write_text(calculations, encoding="utf-8")
    (csvs / "dark_numbers_conf_matrix_r__train.csv").write_text(MATRICES, encoding="utf-8")
    service = RunResultsService(store=None, results_root=tmp_path)
    service._folder = lambda owner, run_id: tmp_path
    return service


def test_the_two_dark_number_tables_are_told_apart(tmp_path):
    service = _folder(tmp_path)
    calculations = service.detail("o", "r", "dark_numbers")
    matrices = service.detail("o", "r", "dark_numbers_matrix")
    assert calculations["columns"][0] == "Model type" and calculations["total_rows"] == 4
    assert matrices["columns"] == ["", "Model type", "a", "b"]          # the true class is kept
    assert matrices["rows"][0][:2] == ["a", "D_cv_test - Cross-trained model"]
    assert "dark_numbers_matrix" in KEEP_FIRST_COLUMN
    assert service.available("o", "r")["dark_numbers"] is True


def test_the_summary_is_the_engines(tmp_path):
    summary = _folder(tmp_path).dark_number_summary("o", "r")
    assert len(summary) == 1 and summary[0]["target"] == "a"
    assert [p["label"] for p in summary[0]["points"]] == ["D_re_full", "D_cv_full", "D_comb", "D_cv_test"]
    assert [p["value"] for p in summary[0]["points"]] == [0.1, 0.2, 0.25, 0.3]
    assert summary[0]["status"] == "ordered"


def test_a_run_without_dark_numbers_has_no_summary(tmp_path):
    service = RunResultsService(store=None, results_root=tmp_path)
    service._folder = lambda owner, run_id: tmp_path
    assert service.dark_number_summary("o", "r") == []
    assert set(DETAILS) >= {"dark_numbers", "dark_numbers_matrix"}


def test_wide_tables_get_a_top_scrollbar():
    from JBGWebUI import WEB
    script = (WEB / "static" / "jbg.js").read_text(encoding="utf-8")
    assert "addTopScrollbar" in script and 'querySelectorAll(".scroll")' in script


# --- A real run (native) --------------------------------------------------------------------------

class DarkNumbersResultsNative298j(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_the_results_tab_shows_dark_numbers(self):
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
        settings["mode"].update(predict=False, calculate_dark_numbers=True)
        project, _, configuration = _setup(client, data=data, roles=settings)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        run_id = client.post(f"/api/projects/{project}/runs",
                             json={"configuration_id": configuration}).json()["run_id"]
        CSVPredictionNative296b._wait(self, client, run_id, work)
        deadline, page = time.time() + 30, ""
        while time.time() < deadline and "Beräkningar" not in page:
            page = client.get(f"/ui/results?run={run_id}&view=dark_numbers").text
            time.sleep(0.3)
        self.assertIn("Mörkertal</a>", page)
        self.assertIn("Slutlig modell, alla data", page)
        self.assertIn("Korstränad modell, testdata", page)
        self.assertRegex(page, r"Målklass: \w+")
        self.assertIn("Förväxlingsmatriser", page)
        self.assertIn("<th>Faktisk klass</th>", page)
        self.assertIn("<th>Skattning</th>", page)
        self.assertEqual(len(re.findall(r'<table class="grid dark-range">', page)), 3)  # a, b, c


def test_readable_names_never_touch_class_names():
    from JBGWebUI import translator
    from JBGWebUIResults import readable_dark_table
    detail = {"columns": ["", "Model type", "Combined", "corr_model"],
              "rows": [["Combined", "D_cv_test - Cross-trained model", 3, "Retrained model"],
                       ["Retrained model", "", 4, "Cross-trained model"]]}
    readable_dark_table(detail, translator("sv"))
    assert detail["rows"][0] == ["Combined", "Korstränad modell, testdata", 3, "Slutlig modell"]
    assert detail["rows"][1] == ["Retrained model", "", 4, "Korstränad modell"]   # a class named so
