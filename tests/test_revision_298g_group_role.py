"""298g (chosen 2026-11-11): a "group" column role on the Dataset tab.

A group column (for example a person's identity) keeps its rows together
when data is split (grouped cross-validation, groups kept apart in the
holdout), without being a predictor. It belongs to the dataset (decision A),
so configurations no longer hold it; runs pass it to the engine as a text
column, the excluded predictor and the validation group column. Prediction
datasets must contain it (the trained model's runs read it, as in Voila).
"""
import csv
import io
import time
from datetime import timedelta

import pytest

from JBGDatasetRoles import DatasetRoles, check_compatible, IncompatibleDataset

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import re  # noqa: E402

ROLES = {"id": "id", "x": "numeric", "note": "text", "person": "group", "label": "class"}
CSV = "id;x;note;person;label\n1;1.0;a b;P1;katt\n2;2.0;c d;P1;hund\n3;3.0;e f;P2;katt\n4;4.0;g h;P2;hund\n".encode()


# --- The role ------------------------------------------------------------------------------------

def test_the_group_column_is_no_feature():
    roles = DatasetRoles.from_mapping(ROLES)
    assert roles.group_column == "person" and "person" not in roles.features
    roles.check(["id", "x", "note", "person", "label"])


def test_at_most_one_group_column():
    roles = DatasetRoles.from_mapping(dict(ROLES, note="group"))
    with pytest.raises(ValueError, match="at most one column can be the group column"):
        roles.check(["id", "x", "note", "person", "label"])


def test_runs_read_the_group_column_but_do_not_learn_from_it():
    from Config import Config
    config = Config()
    config.mode.validation_group_column = "iafId"                        # a stale value
    DatasetRoles.from_mapping(ROLES).apply_to(config)
    assert config.connection.data_text_columns == ["note", "person"]     # read as text
    assert config.mode.excluded_predictor_columns == "person"            # not a predictor
    assert config.mode.validation_group_column == "person"
    DatasetRoles.from_mapping({k: v for k, v in ROLES.items() if v != "group"}).apply_to(config)
    assert config.mode.validation_group_column == "" and config.mode.excluded_predictor_columns == ""


def test_prediction_datasets_need_the_group_column():
    roles = DatasetRoles.from_mapping(ROLES)
    kinds = {"id": "number", "x": "number", "note": "text", "person": "text"}
    check_compatible(roles, kinds, kinds)
    with pytest.raises(IncompatibleDataset) as refused:
        check_compatible(roles, kinds, {k: v for k, v in kinds.items() if k != "person"})
    assert refused.value.missing == ("person",)


def test_the_converter_settings_leave_out_excluded_columns():
    from Config import Config
    from JBGModelCompatibility import _converter_config
    config = Config()
    DatasetRoles.from_mapping(ROLES).apply_to(config)
    assert _converter_config(config)["text_columns"] == ("note",)
    config.mode.excluded_predictor_columns = ""
    assert _converter_config(config)["text_columns"] == ("note", "person")   # unchanged rule


# --- The tabs and the checks ----------------------------------------------------------------------

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


def _project(client, roles=ROLES, mode=None):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    saved = client.put(f"/api/projects/{project}/datasets/{dataset}/roles", json={"columns": roles})
    assert saved.status_code == 200, saved.text
    configuration = client.post(f"/api/projects/{project}/configurations", json={
        "name": "C", "settings": {"mode": dict({"predict": False}, **(mode or {}))}}).json()
    client.post(f"/ui/configuration/{configuration['configuration_id']}/select",
                follow_redirects=False)
    return project, dataset, configuration["configuration_id"]


def test_the_roles_view_offers_the_group_column(tmp_path):
    with _client(tmp_path) as client:
        _, dataset, _ = _project(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
    select = re.search(r'<select name="role:person"[^>]*>(.*?)</select>', page, re.S).group(1)
    assert '<option value="group" selected>Gruppkolumn</option>' in select
    assert "Håller ihop sina rader; ingen variabel" in page


def test_two_group_columns_are_refused_on_the_tab(tmp_path):
    with _client(tmp_path) as client:
        _, dataset, _ = _project(client)
        form = {f"role:{k}": v for k, v in dict(ROLES, note="group").items()}
        form["categorical_limit"] = "30"
        response = client.post(f"/ui/dataset/{dataset}/roles", data=form)
    assert response.status_code == 422 and "Högst en kolumn kan vara gruppkolumn." in response.text


def test_the_configuration_names_the_datasets_group_column(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client)
        page = client.get(f"/ui/configuration?configuration={configuration}").text
        schema = client.get("/api/config/schema", params={"source": "csv"}).json()
    assert 'name="f:mode.validation_group_column"' not in page
    assert "mode.validation_group_column" not in {f["key"] for f in schema["fields"]}
    assert "Gruppkolumn: <b>person</b> (en kolumnroll i det aktiva träningsdatasetet)." in page


def test_without_a_group_column_the_configuration_points_to_the_roles(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client, roles={k: v for k, v in ROLES.items() if v != "group"})
        page = client.get(f"/ui/configuration?configuration={configuration}").text
    assert "ge en kolumn rollen Gruppkolumn under fliken Dataset" in page


@pytest.mark.parametrize("mode", [{"grouped_cross_validation": True},
                                  {"holdout_split_mode": "person_disjoint"}])
def test_keeping_groups_apart_needs_a_group_column(tmp_path, mode):
    without = {k: v for k, v in ROLES.items() if v != "group"}
    with _client(tmp_path) as client:
        project, dataset, configuration = _project(client, roles=without, mode=mode)
        page = client.get("/ui/run").text
        refused = client.post(f"/api/projects/{project}/runs", json={"configuration_id": configuration})
        client.put(f"/api/projects/{project}/datasets/{dataset}/roles", json={"columns": ROLES})
        fixed = client.get("/ui/run").text
    failed = re.search(r'<li class="check-failed">(.*?)</li>', page, re.S).group(1)
    assert "Konfigurationen håller grupper isär, men datasetet har ingen gruppkolumn" in failed
    assert f'href="/ui/dataset?dataset={dataset}&amp;view=roles"' in failed
    assert refused.status_code == 409 and "no group column" in refused.json()["detail"]
    assert "Datasetet har en gruppkolumn för att hålla grupper isär." in fixed


def test_the_check_is_not_shown_when_groups_are_not_kept_apart(tmp_path):
    with _client(tmp_path) as client:
        _project(client)
        page = client.get("/ui/run").text
    assert "gruppkolumn" not in page.lower().split("före körningen")[1].split("</ul>")[0]


def test_the_picture_names_the_group_column(tmp_path):
    with _client(tmp_path) as client:
        _, _, configuration = _project(client, mode={"grouped_cross_validation": True})
        page = client.get(f"/ui/configuration?configuration={configuration}").text
    assert "Testandel 20 %, grupper hålls isär (person)" in page


# --- Through the API with a real worker (native) --------------------------------------------------

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class GroupRoleNative298g(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_a_group_column_keeps_groups_apart_and_is_no_predictor(self):
        from JBGFastAPIServer import create_server_app
        from JBGMetadataStore import MetadataStoreSettings
        from JBGApplicationContracts import RunReference
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import CSVPredictionNative296b
        self._configure_training()
        self.config.mail.notification_email = ""
        with self.repo.open_artifact(self.scope, self.ref.artifact_id) as stream:
            text = stream.read().decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        # Two classes: the engine's group-disjoint holdout needs binary classes.
        header = rows[0] + ["person"]
        body = [row for row in rows[1:] if row[header.index("target")] != "c"]
        # Persons as text ("P0", ...; numbers would also do), assigned round-robin
        # so that every person has rows of both classes: then every grouped
        # fold holds both classes, whatever the library version splits (a fold
        # without a class made the run stop on Windows, 2026-11-12).
        body = [row + [f"P{number % 17}"] for number, row in enumerate(body)]
        out = io.StringIO()
        csv.writer(out).writerows([header] + body)
        data = out.getvalue().encode("utf-8")
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
                self._run_and_predict(client, work, header, body, data)
            finally:
                # Windows cannot remove files a worker still holds (WinError 32).
                from test_revision_296b_csv_predictions import _wait_for_workers_to_exit
                _wait_for_workers_to_exit(client)

    def _run_and_predict(self, client, work, header, body, data):
        from JBGApplicationContracts import RunReference
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import CSVPredictionNative296b
        settings = self.config._config_to_json(username="")
        settings["mode"].update(predict=False, grouped_cross_validation=True,
                                holdout_split_mode="person_disjoint")
        project, dataset, configuration = _setup(client, data=data, roles=settings)
        url = f"/api/projects/{project}/datasets/{dataset}/roles"
        roles = client.get(url).json()["columns"]
        client.put(url, json={"columns": dict(roles, person="group")})
        training = client.post(f"/api/projects/{project}/runs",
                               json={"configuration_id": configuration})
        self.assertEqual(training.status_code, 201, training.text)
        run_id = training.json()["run_id"]
        CSVPredictionNative296b._wait(self, client, run_id, work)
        spec = client.app.state.jbg["store"].get(RunReference(run_id)).spec["configuration"]
        self.assertEqual(spec["mode"]["validation_group_column"], "person")
        self.assertEqual(spec["mode"]["excluded_predictor_columns"], "person")
        log = (work / "run-logs" / f"run_{run_id}.log").read_text(encoding="utf-8", errors="replace")
        self.assertIn("Group-disjoint holdout", log)
        self.assertIn("0 groups appear in both training and holdout", log)
        deadline, models = time.time() + 30, []
        while time.time() < deadline and not models:
            models = client.get(f"/api/projects/{project}/models").json()
            time.sleep(0.2)
        model_id = models[0]["model_id"]
        # Predictions: the group column is required, as the model's runs read it.
        class_at = header.index("target")
        pred_header = [h for i, h in enumerate(header) if i != class_at]
        pred_rows = [[v for i, v in enumerate(row) if i != class_at] for row in body[:4]]
        for number, row in enumerate(pred_rows):
            row[pred_header.index("row_id")] = f"9{number:03d}"

        def file(columns, rows):
            out = io.StringIO()
            csv.writer(out).writerows([columns] + rows)
            return out.getvalue().encode("utf-8")
        refused = client.post(f"/api/projects/{project}/datasets/csv",
                              data={"name": "Utan", "role": "prediction"},
                              files={"file": ("u.csv", file(pred_header[:-1],
                                                           [r[:-1] for r in pred_rows]), "text/csv")})
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(refused.json()["detail"]["missing_columns"], ["person"])
        accepted = client.post(f"/api/projects/{project}/datasets/csv",
                               data={"name": "Med", "role": "prediction"},
                               files={"file": ("m.csv", file(pred_header, pred_rows), "text/csv")})
        self.assertEqual(accepted.status_code, 201, accepted.text)
        predicted = client.post(f"/api/projects/{project}/predictions",
                                json={"model_id": model_id,
                                      "dataset_id": accepted.json()["dataset_id"]})
        self.assertEqual(predicted.status_code, 201, predicted.text)
        CSVPredictionNative296b._wait(self, client, predicted.json()["run_id"], work)
