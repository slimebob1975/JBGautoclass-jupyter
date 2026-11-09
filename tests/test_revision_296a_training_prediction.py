"""296a: live progress, Training and Prediction tabs, display name, run logs."""
import json
import re
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402

OWNER = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key


# --- Live progress ------------------------------------------------------------------------------

def test_the_engines_progress_reaches_the_overview():
    from JBGLogger import JBGLogger
    from JBGRunJobStore import RunJobStore
    from JBGRunOverview import project_run_overview
    from JBGRunWorker import StoreProgressLogger

    class Beat:
        lost = False

        def touch(self):
            pass

    store = RunJobStore.sqlite(Path(tempfile.mkdtemp()) / "jobs.sqlite")
    store.create_tables()
    run = store.create("me", DataSourceKind.CSV, {"spec_version": 1})
    store.claim(run, "w")
    observed = StoreProgressLogger(JBGLogger(quiet=True, in_terminal=False), store, run, Beat())
    observed.initiate_progress(4)
    seen = []
    for step in range(3):
        observed.update_progress(message=f"step {step}")       # how the engine reports
        seen.append(project_run_overview(run.key, store.get(run).status,
                                         store.progress(run))["stage"]["percent"])
    assert seen == [0.25, 0.5, 0.75]
    observed.update_progress(percent=0.9)                       # sums above 1 are clamped
    assert store.progress(run)[-1].percent == 1.0


# --- The interface ---------------------------------------------------------------------------------

def _client(tmp_path, subject="developer", trust=False):
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        trust_app_service_headers=trust,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _open(client, name="P"):
    location = client.post("/ui/projects", data={"name": name, "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _finished(client, project, kind="training", evaluation=True):
    store = client.app.state.jbg["store"]
    run = store.create(OWNER, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "kind": kind, "configuration_name": "C", "configuration_version": 1}})
    store.claim(run, "w")
    result = {"schema_version": 1, "kind": "classification", "class_count": 2, "mispredicted_count": 0}
    if evaluation:
        result["evaluation"] = {"schema_version": 1, "accuracy": 0.75, "balanced_accuracy": 0.75,
                                "mcc": 0.5, "precision_macro": 0.75, "recall_macro": 0.75,
                                "f1_macro": 0.7, "precision_weighted": 0.75, "recall_weighted": 0.75,
                                "f1_weighted": 0.7, "confusion_matrix": [[2, 0], [1, 1]]}
    store.finish(run, "w", RunStatus.SUCCEEDED, result=result)
    return run


def test_tabs_follow_the_work_order(tmp_path):
    with _client(tmp_path) as client:
        page = client.get("/ui/projects").text
    labels = re.findall(r'<(?:a href="/ui/[a-z]+"[^>]*|span class="off"[^>]*)>([^<]+)</(?:a|span)>',
                        page.split('<nav class="tabs"')[1].split("</nav>")[0])
    assert labels == ["Översikt", "Projekt", "Dataset", "Konfiguration", "Träning", "Resultat",
                      "Modeller", "Prediktion", "Körhistorik", "Om"]          # 295d: About last
    assert '<a href="/ui/prediction"' in page


def test_training_and_prediction_runs_are_kept_apart(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        training = _finished(client, project)
        prediction = _finished(client, project, kind="prediction", evaluation=False)
        run_page, predict_page = client.get("/ui/run").text, client.get("/ui/prediction").text
        results_page = client.get("/ui/results").text
    assert f"run={training.key}" in run_page and f"run={prediction.key}" not in run_page
    assert f"run={prediction.key}" in predict_page and f"run={training.key}" not in predict_page
    assert f"run={prediction.key}" not in results_page                     # results: training only


def test_the_training_tab_names_the_classes(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        run = _finished(client, project)
        folder = tmp_path / "runs" / project / run.key
        folder.mkdir(parents=True)
        (folder / "class_labels.json").write_text(json.dumps({"class_labels": ["setosa", "virginica"]}))
        page = client.get(f"/ui/run?run={run.key}").text
    assert '<th class="num">setosa</th>' in page and "<th>virginica</th>" in page


def test_cancel_returns_to_the_tab_it_came_from(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        store = client.app.state.jbg["store"]
        run = store.create(OWNER, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
            "project_id": project, "kind": "prediction"}})
        back = client.post(f"/ui/run/{run.key}/cancel", data={"back": "/ui/prediction"},
                           follow_redirects=False)
        assert back.headers["location"].startswith("/ui/prediction?run=")
        elsewhere = client.post(f"/ui/run/{run.key}/cancel", data={"back": "https://evil.example/"},
                                follow_redirects=False)
        assert elsewhere.headers["location"].startswith("/ui/run")


# --- Display name -----------------------------------------------------------------------------------

def test_the_header_shows_unknown_user_unless_app_service_headers_are_trusted(tmp_path):
    name = {"x-ms-client-principal-name": "anna.andersson@example.se"}
    with _client(tmp_path) as client:
        page = client.get("/ui/projects", headers=name).text
        assert "Inloggad som Okänd användare" in page and "anna.andersson" not in page
        assert "developer" not in page.split("<header")[1].split("</header>")[0]
        english = client.get("/ui/projects", headers={"accept-language": "en"}).text
        assert "Signed in as Unknown user" in english
    with _client(tmp_path / "azure", trust=True) as client:
        assert "Inloggad som anna.andersson@example.se" in client.get("/ui/projects", headers=name).text
        odd = client.get("/ui/projects", headers={"x-ms-client-principal-name": "<b>x</b>\u0007" + "y" * 300}).text
        shown = re.search(r"Inloggad som (.*?)</span>", odd).group(1)
        assert shown.startswith("&lt;b&gt;x&lt;/b&gt;y") and "\u0007" not in shown
        assert len(shown.replace("&lt;", "<").replace("&gt;", ">")) == 200


# --- Run logs -----------------------------------------------------------------------------------------

def test_logs_are_shown_to_the_owner_and_downloadable(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        run = _finished(client, project)
        (tmp_path / "run-logs").mkdir(exist_ok=True)
        (tmp_path / "run-logs" / f"run_{run.key}.log").write_text(
            "start\n" + "x" * 250_000 + "\n<the end>\n", encoding="utf-8")
        assert f'href="/ui/runs/{run.key}/log"' in client.get(f"/ui/run?run={run.key}").text
        page = client.get(f"/ui/runs/{run.key}/log").text
        assert "&lt;the end&gt;" in page and "start\n" not in page          # the end, escaped
        assert "Bara slutet av loggen visas" in page
        download = client.get(f"/api/runs/{run.key}/log")
        assert download.status_code == 200 and download.text.startswith("start")
        assert "attachment" in download.headers["content-disposition"]
        missing = _finished(client, project)
        assert client.get(f"/ui/runs/{missing.key}/log").status_code == 404
        assert "Visa logg" not in client.get(f"/ui/run?run={missing.key}").text
    from fastapi.testclient import TestClient  # noqa: F401
    with _client(tmp_path, subject="bob") as theirs:
        assert theirs.get(f"/api/runs/{run.key}/log").status_code == 404
        assert theirs.get(f"/ui/runs/{run.key}/log").status_code == 404


def test_every_new_text_is_translated():
    import JBGWebUI
    from JBGWebUIPrediction import MESSAGES, PREFLIGHT
    texts = {t for _, t in MESSAGES.values()} | {t for _, a, b in PREFLIGHT for t in (a, b)}
    texts |= {"Training", "Prediction", "Unknown user", "Start training", "Show log"}
    assert not sorted(t for t in texts if t not in JBGWebUI._current_catalog("sv"))
