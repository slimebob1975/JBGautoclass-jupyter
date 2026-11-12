"""298h (agreed 2026-11-12): a "date" column role ("Datumvariabel").

A date predictor becomes features every model type can use: the year as a
number, month and weekday as points on a circle (sine and cosine: December
lies next to January), and whether the date was missing. The input converter
(saved with the model) derives them, so predictions get exactly the same
features. Dates are never suggested automatically (agreed): a date column is
suggested as ignored, and its "treated as" text says it can be a date feature.
Engine: opt-in through the runtime field mode.date_columns (empty by default:
the Voila GUI and earlier models are unchanged).
"""
import csv
import io
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from JBGDatasetRoles import DatasetRoles, suggest_roles
from JBGTransformers import TextDataToNumbersConverter

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


# --- The converter ---------------------------------------------------------------------------------

def _frame():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0],
                         "when": ["2024-12-31", "2025-01-01 10:00:00", "nan", "2023-06-15"],
                         "note": ["katt hund", "hund ko", "ko får", "får katt"]})


def _converter(**options):
    return TextDataToNumbersConverter(text_columns=["note", "when"], use_encryption=False,
                                      use_categorization=False, **options)


def test_a_date_becomes_year_month_weekday_and_missing():
    converter = _converter(date_columns=["when"])
    out = converter.fit(_frame()) or converter.transform(_frame())
    names = [c for c in out.columns if c.startswith("when_")]
    assert names == ["when_year", "when_month_sin", "when_month_cos", "when_weekday_sin",
                     "when_weekday_cos", "when_missing"]
    assert "when" not in out.columns and converter.text_columns_ == ["note"]
    assert list(out["when_year"]) == [2024.0, 2025.0, 2024.0, 2023.0]   # missing: median year
    assert list(out["when_missing"]) == [0.0, 0.0, 1.0, 0.0]
    circle = out[["when_month_sin", "when_month_cos"]].to_numpy()
    december_january = np.hypot(*(circle[0] - circle[1]))
    december_june = np.hypot(*(circle[0] - circle[3]))
    assert december_january < 0.6 < 1.9 < december_june


def test_predictions_get_the_training_fill_year():
    converter = _converter(date_columns=["when"])
    converter.fit(_frame())
    later = _frame().assign(when=["nan", "2030-02-03", "nan", "nan"])
    assert list(converter.transform(later)["when_year"]) == [2024.0, 2030.0, 2024.0, 2024.0]


def test_without_dates_nothing_changes():
    plain = _converter()
    assert plain.configured_date_columns_ == () and plain._dates() == []
    legacy = _converter()
    del legacy.date_columns_                       # a converter saved before 298h
    frame = _frame().drop(columns="when")
    legacy.text_columns_ = ["note"]
    legacy.fit(frame)
    assert list(legacy.transform(frame).columns)[:1] == ["x"]


def test_compatibility_metadata_only_mentions_dates_when_there_are_any():
    from Config import Config
    from JBGModelCompatibility import _converter_config
    config = Config()
    assert "date_columns" not in _converter_config(config)          # earlier models unchanged
    DatasetRoles.from_mapping({"id": "id", "x": "numeric", "note": "text", "when": "date",
                               "label": "class"}).apply_to(config)
    settings = _converter_config(config)
    assert settings["date_columns"] == ("when",) and settings["text_columns"] == ("note",)
    assert config.connection.data_text_columns == ["note", "when"]     # read as text


# --- Roles, suggestions and validation --------------------------------------------------------------

def test_dates_are_suggested_as_ignored():
    columns = [{"name": "nr", "type": "number", "distinct": 10, "missing": 0},
               {"name": "when", "type": "text", "distinct": 9, "missing": 0, "date": True},
               {"name": "made", "type": "datetime2", "distinct": 9, "missing": 0, "date": True}]
    assert suggest_roles(columns, 10) == {"nr": "id", "when": "ignored", "made": "ignored"}


CSV = ("id;x;when;label\n1;1.0;2024-01-05;katt\n2;2.0;2024-02-06;hund\n"
       "3;3.0;2024-03-07;katt\n4;4.0;;hund\n").encode()


def _client(tmp_path):
    pytest.importorskip("python_multipart")
    pytest.importorskip("jinja2")
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _dataset(client, data=CSV):
    import re
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                          files={"file": ("d.csv", data, "text/csv")}).json()["dataset_id"]
    return project, dataset


def test_the_roles_view_offers_dates_and_points_them_out(tmp_path):
    import re
    with _client(tmp_path) as client:
        _, dataset = _dataset(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
    select = re.search(r'<select name="role:when"[^>]*>(.*?)</select>', page, re.S).group(1)
    assert '<option value="ignored" selected>' in select and "Datumvariabel" in select
    assert "Används inte (datum: kan bli datumvariabel)" in page


def test_validation_checks_date_features(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _dataset(client)
        url = f"/api/projects/{project}/datasets/{dataset}"
        client.put(url + "/roles", json={"columns": {"id": "id", "x": "numeric", "when": "date",
                                                     "label": "class"}})
        good = client.post(url + "/validate").json()
        client.put(url + "/roles", json={"columns": {"id": "id", "x": "numeric", "label": "date",
                                                     "when": "class"}})
        bad = client.post(url + "/validate").json()
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
    date_check = next(c for c in good["checks"] if c["code"] == "date_types")
    assert date_check["status"] == "passed" and good["validation_state"] == "valid"
    assert next(c for c in bad["checks"] if c["code"] == "date_types")["status"] == "failed"
    assert "År, månad och veckodag" in page


def test_no_date_check_without_date_features(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _dataset(client)
        url = f"/api/projects/{project}/datasets/{dataset}"
        client.put(url + "/roles", json={"columns": {"id": "id", "x": "numeric", "label": "class"}})
        result = client.post(url + "/validate").json()
    assert "date_types" not in [c["code"] for c in result["checks"]]


# --- Through the API with a real worker (native) --------------------------------------------------

class DateRoleNative298h(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_a_date_feature_trains_and_predicts(self):
        from JBGFastAPIServer import create_server_app
        from JBGMetadataStore import MetadataStoreSettings
        self._configure_training()
        self.config.mail.notification_email = ""
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            text = stream.read().decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        header = rows[0] + ["when"]
        start = date(2023, 1, 1)
        body = [row + ["" if number % 11 == 0 else (start + timedelta(days=7 * number)).isoformat()]
                for number, row in enumerate(rows[1:])]
        out = io.StringIO()
        csv.writer(out).writerows([header] + body)
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
                self._train_and_predict(client, work, header, body, out.getvalue().encode("utf-8"))
            finally:
                from test_revision_296b_csv_predictions import _wait_for_workers_to_exit
                _wait_for_workers_to_exit(client)

    def _train_and_predict(self, client, work, header, body, data):
        from JBGApplicationContracts import RunReference
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import CSVPredictionNative296b
        settings = self.config._config_to_json(username="")
        settings["mode"].update(predict=False, calculate_feature_importance=True)
        project, dataset, configuration = _setup(client, data=data, roles=settings)
        url = f"/api/projects/{project}/datasets/{dataset}/roles"
        roles = client.get(url).json()["columns"]
        client.put(url, json={"columns": dict(roles, when="date")})
        training = client.post(f"/api/projects/{project}/runs", json={"configuration_id": configuration})
        self.assertEqual(training.status_code, 201, training.text)
        run_id = training.json()["run_id"]
        CSVPredictionNative296b._wait(self, client, run_id, work)
        spec = client.app.state.jbg["store"].get(RunReference(run_id)).spec["configuration"]
        self.assertEqual(spec["mode"]["date_columns"], "when")
        deadline, models = time.time() + 30, []
        while time.time() < deadline and not models:
            models = client.get(f"/api/projects/{project}/models").json()
            time.sleep(0.2)
        # The derived features reached the model: four numeric inputs plus the
        # date's six features (feature importance reports the original column).
        log = (work / "run-logs" / f"run_{run_id}.log").read_text(encoding="utf-8", errors="replace")
        self.assertIn("Number of features after text and category conversion: 10", log)
        # A prediction with the saved model: the converter derives the same features.
        class_at = header.index("target")
        pred_header = [h for i, h in enumerate(header) if i != class_at]
        pred_rows = [[v for i, v in enumerate(row) if i != class_at] for row in body[:4]]
        for number, row in enumerate(pred_rows):
            row[pred_header.index("row_id")] = f"9{number:03d}"
        out = io.StringIO()
        csv.writer(out).writerows([pred_header] + pred_rows)
        accepted = client.post(f"/api/projects/{project}/datasets/csv",
                               data={"name": "Ny", "role": "prediction"},
                               files={"file": ("n.csv", out.getvalue().encode("utf-8"), "text/csv")})
        self.assertEqual(accepted.status_code, 201, accepted.text)
        predicted = client.post(f"/api/projects/{project}/predictions",
                                json={"model_id": models[0]["model_id"],
                                      "dataset_id": accepted.json()["dataset_id"]})
        self.assertEqual(predicted.status_code, 201, predicted.text)
        CSVPredictionNative296b._wait(self, client, predicted.json()["run_id"], work)
