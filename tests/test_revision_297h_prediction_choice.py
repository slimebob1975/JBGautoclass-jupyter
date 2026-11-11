"""297h: the Prediction tab's "Rows to predict" keeps showing the chosen
dataset after "Start prediction" (found 2026-11-11: it fell back to its first
option, which looked as if the choice had been ignored)."""
import re

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402
from test_revision_297f_csv_prediction_datasets import _client  # noqa: E402

OWNER = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key


def _project(client):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    ids = {}
    for name, role in (("Träning", "training"), ("Nya", "prediction"), ("Blandat", "mixed")):
        response = client.post("/ui/dataset/csv", data={"name": name, "role": role},
                               files={"file": (f"{name}.csv", b"id;x;c\n1;1;a\n2;2;b\n", "text/csv")},
                               follow_redirects=False)
        ids[name] = re.search(r"dataset=([0-9a-f-]{36})", response.headers["location"]).group(1)
    return project, ids


def _prediction_run(client, project, dataset_id):
    store = client.app.state.jbg["store"]
    run = store.create(OWNER, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "kind": "prediction", "dataset_id": dataset_id}})
    store.claim(run, "w")
    return run


def _selected(page):
    select = re.search(r'<select id="predict-dataset" name="dataset_id">(.*?)</select>',
                       page, re.S).group(1)
    return re.findall(r'<option value="([^"]*)" selected>', select)


def test_the_choice_shows_the_dataset_of_the_run_on_show(tmp_path):
    with _client(tmp_path) as client:
        project, ids = _project(client)
        run = _prediction_run(client, project, ids["Blandat"])
        page = client.get(f"/ui/prediction?run={run.key}&message=prediction_started").text
        latest = client.get("/ui/prediction").text                    # the latest run is shown
    assert _selected(page) == [ids["Blandat"]] and _selected(latest) == [ids["Blandat"]]


def test_a_run_on_the_training_datasets_rows_shows_the_first_choice(tmp_path):
    with _client(tmp_path) as client:
        project, ids = _project(client)
        run = _prediction_run(client, project, ids["Träning"])    # not offered: the default
        page = client.get(f"/ui/prediction?run={run.key}").text
    assert _selected(page) == []                                   # first option, as chosen


def test_a_refused_choice_is_kept(tmp_path, monkeypatch):
    import JBGProjectRuns

    def refuse(*args, **kwargs):
        raise JBGProjectRuns.NothingToPredict("refused")
    monkeypatch.setattr(JBGProjectRuns.ProjectRunService, "submit_prediction", refuse)
    with _client(tmp_path) as client:
        _, ids = _project(client)
        client.cookies.set("jbg_model", "00000000-0000-4000-8000-000000000001", path="/ui")
        response = client.post("/ui/prediction", data={"dataset_id": ids["Nya"]},
                               follow_redirects=False)
        page = client.get(response.headers["location"]).text
    assert f"dataset={ids['Nya']}" in response.headers["location"]
    assert _selected(page) == [ids["Nya"]] and "Alla rader i datasetet har en klass" in page


@pytest.mark.parametrize("dataset", ["00000000-0000-4000-8000-000000000009", "x\"><b>", ""])
def test_only_the_projects_offered_datasets_can_be_shown_as_chosen(tmp_path, dataset):
    with _client(tmp_path) as client:
        _project(client)
        page = client.get("/ui/prediction", params={"dataset": dataset}).text
    assert _selected(page) == [] and "<b>" not in page.split("predict-dataset")[1][:2000]


def test_the_run_list_names_each_runs_dataset(tmp_path):
    with _client(tmp_path) as client:
        project, ids = _project(client)
        _prediction_run(client, project, ids["Nya"])
        from JBGDurableRuns import DurableRunService
        runs = DurableRunService(client.app.state.jbg["store"],
                                 scope_for=lambda principal: None).list_project_runs(OWNER, project)
    assert [r["dataset_id"] for r in runs] == [ids["Nya"]]
