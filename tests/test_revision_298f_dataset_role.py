"""298f (noted 2026-11-11): a CSV dataset uploaded as mixed by mistake could
not be changed back. Training and mixed datasets may now switch role (Dataset
tab, PATCH on the API); the column roles, the validation state and the
active choice are kept; prediction datasets and the prediction role are
refused. The checks before a training run link to where a failed check is
put right (the blocking "CSV runs cannot also predict" to the configuration's
run type). Checked in headless Chromium with the reported situation."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

CSV = b"id;x;label\n1;1.0;a\n2;2.0;b\n3;3.0;a\n4;4.0;b\n5;5.0;\n"


def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _project(client, role="mixed"):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": role},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    return project, dataset


def _set_up_training(client, project, dataset):
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    client.put(f"/api/projects/{project}/datasets/{dataset}/roles",
               json={"columns": {"id": "id", "x": "numeric", "label": "class"}})


# --- API ---------------------------------------------------------------------------------------

def test_mixed_and_training_switch_and_keep_everything(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        _set_up_training(client, project, dataset)
        url = f"/api/projects/{project}/datasets/{dataset}"
        training = client.patch(url, json={"role": "training"})
        back = client.patch(url, json={"role": "mixed"})
        roles = client.get(url + "/roles").json()["columns"]
    assert training.status_code == 200 and training.json()["role"] == "training"
    assert training.json()["active_training"] is True
    assert back.json()["role"] == "mixed" and roles == {"id": "id", "x": "numeric", "label": "class"}


def test_the_same_role_is_accepted(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client, role="training")
        response = client.patch(f"/api/projects/{project}/datasets/{dataset}", json={"role": "training"})
    assert response.status_code == 200 and response.json()["role"] == "training"


def test_prediction_is_not_a_switch(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client, role="training")
        _set_up_training(client, project, dataset)
        to_prediction = client.patch(f"/api/projects/{project}/datasets/{dataset}",
                                     json={"role": "prediction"})
        prediction = client.post(f"/api/projects/{project}/datasets/csv",
                                 data={"name": "Ny", "role": "prediction"},
                                 files={"file": ("n.csv", CSV, "text/csv")}).json()["dataset_id"]
        from_prediction = client.patch(f"/api/projects/{project}/datasets/{prediction}",
                                       json={"role": "training"})
        unknown = client.patch(f"/api/projects/{project}/datasets/00000000-0000-4000-8000-000000000000",
                               json={"role": "training"})
    assert to_prediction.status_code == 422                                   # not offered
    assert from_prediction.status_code == 422
    assert from_prediction.json()["detail"] == "only training and mixed datasets can switch role"
    assert unknown.status_code == 404


# --- Dataset tab -------------------------------------------------------------------------------

def test_the_tab_offers_the_other_role(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        page = client.get(f"/ui/dataset?dataset={dataset}").text
        changed = client.post(f"/ui/dataset/{dataset}/role", data={"role": "training"},
                              follow_redirects=False)
        after = client.get(changed.headers["location"]).text
        role = client.get(f"/api/projects/{project}/datasets/{dataset}").json()["role"]
    form = page.split('class="change-role"')[1].split("</form>")[0]
    assert 'name="role" value="training"' in form and "Byt roll till träning" in form
    assert "message=role_changed" in changed.headers["location"] and role == "training"
    assert "Datasetets roll är ändrad." in after and "Byt roll till blandad" in after


def test_prediction_datasets_have_no_role_switch(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _project(client, role="prediction")
        page = client.get(f"/ui/dataset?dataset={dataset}").text
        refused = client.post(f"/ui/dataset/{dataset}/role", data={"role": "training"},
                              follow_redirects=False)
    assert "change-role" not in page
    assert "message=cannot_change_role" in refused.headers["location"]


# --- Checks before a training run --------------------------------------------------------------

def test_a_failed_check_links_to_where_it_is_put_right(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        _set_up_training(client, project, dataset)
        configuration = client.post(f"/api/projects/{project}/configurations", json={
            "name": "C", "settings": {"mode": {"predict": True}}}).json()["configuration_id"]
        client.post(f"/ui/configuration/{configuration}/select", follow_redirects=False)
        page = client.get("/ui/run").text
    failed = re.search(r'<li class="check-failed">(.*?)</li>', page, re.S).group(1)
    assert "CSV-körningar kan inte samtidigt prediktera" in failed
    assert (f'href="/ui/configuration?configuration={configuration}#section-run_type" '
            'class="fix-link">Åtgärda ›') in failed
    passed = re.findall(r'<li class="check-passed">(.*?)</li>', page, re.S)
    assert passed and not any("fix-link" in item for item in passed)


def test_missing_roles_link_to_the_roles_view(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
        page = client.get("/ui/run").text
    assert f'href="/ui/dataset?dataset={dataset}&amp;view=roles" class="fix-link"' in page
