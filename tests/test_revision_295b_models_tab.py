"""295b: Models tab: catalog, model card, choosing a model."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402

CSV = b"id;sepal;species\n1;5.1;a\n2;7.0;b\n3;6.3;a\n4;4.9;b\n"
OWNER = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _open(client, name="P"):
    location = client.post("/ui/projects", data={"name": name, "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _model(client, project, state="available"):
    """A finished project run and its model in the catalog, as the worker leaves them."""
    from JBGModelCatalog import ModelState
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "Iris", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    store = client.app.state.jbg["store"]
    run = store.create(OWNER, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "dataset_id": dataset, "configuration_name": "LR",
        "configuration_version": 2, "model_id": "m"}})
    store.claim(run, "w")
    store.finish(run, "w", RunStatus.SUCCEEDED, result={
        "schema_version": 1, "kind": "classification", "class_count": 2, "mispredicted_count": 0,
        "evaluation": {"schema_version": 1, "accuracy": 0.9, "balanced_accuracy": 0.9, "mcc": 0.8,
                       "precision_macro": 0.9, "recall_macro": 0.9, "f1_macro": 0.9,
                       "precision_weighted": 0.9, "recall_weighted": 0.9, "f1_weighted": 0.9,
                       "confusion_matrix": [[2, 0], [0, 2]]}})
    catalog = client.app.state.jbg["models"]
    entry = catalog.register(OWNER, project, "LR v2 · test", "00000000-0000-4000-8000-0000000000aa",
                             dataset_id=dataset, run_id=run.key)
    if state != "registered":
        catalog.set_state(OWNER, project, entry.model_id, ModelState(state))
    return entry.model_id, run


def test_no_models_yet(tmp_path):
    with _client(tmp_path) as client:
        _open(client)
        page = client.get("/ui/models").text
        assert "Inga modeller ännu." in page and '<a href="/ui/models" class="on"' in page


def test_the_model_card_shows_origin_and_figures(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        model_id, run = _model(client, project)
        page = client.get(f"/ui/models?model={model_id}").text
    assert "LR v2 · test" in page and "Tillgänglig" in page
    assert "<dd>Iris</dd>" in page and "LR v2</dd>" in page
    assert f'href="/ui/results?run={run.key}"' in page and "90,0 %" in page
    assert '<button class="btn act ready">Välj modell</button>' in page
    assert "model_path" not in page and ".sav" not in page                 # no file paths


def test_choosing_a_model_fills_the_context_bar(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        model_id, _ = _model(client, project)
        chosen = client.post(f"/ui/models/{model_id}/choose", follow_redirects=False)
        assert any(h.startswith(f"jbg_model={model_id}") for h in chosen.headers.get_list("set-cookie"))
        page = client.get(chosen.headers["location"]).text
        assert "Modellen är vald." in page and re.search(r"<span class=\"k\">Modell</span><b>LR v2 · test</b>", page)
        assert "Den här modellen är vald." in page and "btn act blocked" not in page


@pytest.mark.parametrize("state", ["registered", "missing", "retired"])
def test_only_available_models_can_be_chosen(tmp_path, state):
    with _client(tmp_path) as client:
        project = _open(client)
        model_id, _ = _model(client, project, state=state)
        response = client.post(f"/ui/models/{model_id}/choose", follow_redirects=False)
        assert "message=not_available" in response.headers["location"]
        assert "set-cookie" not in response.headers


def test_closing_or_switching_forgets_the_model(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        model_id, _ = _model(client, project)
        client.post(f"/ui/models/{model_id}/choose", follow_redirects=False)
        other = _open(client, "Other")
        assert "LR v2 · test" not in client.get("/ui/run").text            # chip empty again
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        client.post(f"/ui/models/{model_id}/choose", follow_redirects=False)
        closed = client.post("/ui/close-project", follow_redirects=False)
        assert any(h.startswith("jbg_model=") and "Max-Age=0" in h
                   for h in closed.headers.get_list("set-cookie"))
        assert other


def test_other_users_cannot_see_or_choose_my_models(tmp_path):
    with _client(tmp_path) as mine:
        project = _open(mine)
        model_id, _ = _model(mine, project)
    with _client(tmp_path, subject="bob") as theirs:
        theirs.cookies.set("jbg_project", project, path="/ui")
        assert "LR v2 · test" not in theirs.get(f"/ui/models?model={model_id}").text
        refused = theirs.post(f"/ui/models/{model_id}/choose", follow_redirects=False)
        assert refused.headers["location"] == "/ui/models?message=unavailable"


def test_texts_are_translated():
    import JBGWebUI
    from JBGWebUIModels import MESSAGES, STATES
    texts = set(STATES.values()) | {t for _, t in MESSAGES.values()}
    assert not sorted(t for t in texts if t not in JBGWebUI._CATALOGS["sv"])
