"""294a: Run tab: preflight, start, follow, cancel, results."""
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import RunReference, RunStatus  # noqa: E402
from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402
from JBGSQLServer import SQLServerSettings  # noqa: E402
from JBGWebUIRun import MESSAGES, PREFLIGHT, STAGES  # noqa: E402

CSV = b"id;sepal;species\n1;5.1;a\n2;7.0;b\n3;6.3;a\n4;4.9;b\n"
ROLES = {"id": "id", "sepal": "numeric", "species": "class"}


class _Launcher:
    def __call__(self, command, **options):
        stdin = type("S", (), {"write": lambda s, d: None, "close": lambda s: None})()
        return type("P", (), {"stdin": stdin, "poll": lambda s: None})()


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5), "launcher": _Launcher()}))


def _open(client, name="P"):
    location = client.post("/ui/projects", data={"name": name, "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _ready(client, *, predict=False):
    project = _open(client)
    dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                          files={"file": ("d.csv", CSV, "text/csv")}).json()["dataset_id"]
    client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
    client.put(f"/api/projects/{project}/datasets/{dataset}/roles", json={"columns": ROLES})
    settings = {"mode": {"predict": True}} if predict else {}
    created = client.post("/ui/configuration", data={"name": "C"}, follow_redirects=False)
    configuration = re.search(r"configuration=([0-9a-f-]{36})", created.headers["location"]).group(1)
    if predict:
        client.put(f"/api/projects/{project}/configurations/{configuration}",
                   json={"settings": settings, "expected_version": 1})
    return project, dataset, configuration


def _checks(page):
    return re.findall(r'<li class="check-(\w+)">', page)


# --- Preflight ------------------------------------------------------------------------------

def test_preflight_blocks_until_everything_is_ready(tmp_path):
    with _client(tmp_path) as client:
        _open(client)
        page = client.get("/ui/run").text
        assert _checks(page)[:3] == ["failed", "failed", "failed"]
        assert re.search(r'<button class="btn act blocked" disabled[^>]*>Starta körning', page)
    with _client(tmp_path / "ready") as client:
        _ready(client)
        page = client.get("/ui/run").text
        assert "Konfigurationen tränar bara (CSV)." in page
        assert _checks(page)[-1] == "warning"                      # not validated: advisory only
        assert '<button class="btn act ready">Starta körning</button>' in page


def test_csv_configurations_start_train_only_and_predicting_ones_are_refused(tmp_path):
    with _client(tmp_path) as client:
        project, _, configuration = _ready(client)
        stored = client.get(f"/api/projects/{project}/configurations/{configuration}").json()
        assert stored["settings"]["mode"]["predict"] is False            # CSV default
    with _client(tmp_path / "predict") as client:
        project, _, configuration = _ready(client, predict=True)
        page = client.get("/ui/run").text
        assert "CSV-körningar kan inte samtidigt prediktera" in page
        refused = client.post("/ui/run", follow_redirects=True).text      # also refused directly
        assert "Körningen kunde inte startas." in refused
        direct = client.post(f"/api/projects/{project}/runs", json={"configuration_id": configuration})
        assert direct.json() == {"detail": "CSV training runs cannot also predict"}


def test_sql_configurations_keep_predicting_by_default(tmp_path):
    with _client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "S", "source": "sql"}).json()["project_id"]
        created = client.post(f"/api/projects/{project}/configurations", json={"name": "C"}).json()
        assert created["settings"]["mode"]["predict"] is True


# --- Start, follow, cancel ---------------------------------------------------------------------

def test_start_follow_and_cancel(tmp_path):
    with _client(tmp_path) as client:
        project, _, _ = _ready(client)
        started = client.post("/ui/run", follow_redirects=False)
        run_id = re.search(r"run=([0-9a-f-]{36})", started.headers["location"]).group(1)
        page = client.get(started.headers["location"]).text
        assert "Körningen har startat." in page and "I kö" in page
        panel = re.search(r'data-run-poll="([^"]+)"[^>]*data-stage-labels="([^"]+)"', page)
        assert panel.group(1) == run_id
        labels = json.loads(panel.group(2).replace("&#34;", '"'))
        assert labels["train_model"] == "Tränar och jämför modeller"
        assert run_id in [r["run_id"] for r in client.get("/api/runs").json()]
        cancelled = client.post(f"/ui/run/{run_id}/cancel", follow_redirects=True).text
        assert "Körningen avbryts." in cancelled and "data-run-poll" not in cancelled


def test_a_finished_run_shows_its_results(tmp_path):
    with _client(tmp_path) as client:
        project, _, _ = _ready(client)
        run_id = re.search(r"run=([0-9a-f-]{36})",
                           client.post("/ui/run", follow_redirects=False).headers["location"]).group(1)
        store = client.app.state.jbg["store"]
        store.claim(RunReference(run_id), "worker")
        store.finish(RunReference(run_id), "worker", RunStatus.SUCCEEDED, result={
            "schema_version": 1, "kind": "classification", "class_count": 2, "mispredicted_count": 1,
            "evaluation": {"schema_version": 1, "accuracy": 0.75, "balanced_accuracy": 0.75, "mcc": 0.5,
                           "precision_macro": 0.75, "recall_macro": 0.75, "f1_macro": 0.733,
                           "precision_weighted": 0.75, "recall_weighted": 0.75, "f1_weighted": 0.733,
                           "confusion_matrix": [[2, 0], [1, 1]]},
            "feature_importance": {"schema_version": 1, "method": "permutation", "baseline_score": 0.5,
                                   "holdout_rows": 4, "repeats": 5, "total_features": 1,
                                   "features": [{"rank": 1, "name": "sepal", "mean_score_decrease": 0.2,
                                                 "std_across_repeats": 0.01}]}})
        page = client.get(f"/ui/run?run={run_id}").text
    assert "data-run-poll" not in page and "Klar" in page
    assert "75,0 %" in page and "0,500" in page and "0,733" in page
    assert '<td class="num diag">2</td>' in page and "sepal" in page


# --- Ownership, listing, texts --------------------------------------------------------------------

def test_runs_of_other_projects_and_users_stay_out(tmp_path):
    with _client(tmp_path) as mine:
        project, _, _ = _ready(mine)
        run_id = re.search(r"run=([0-9a-f-]{36})",
                           mine.post("/ui/run", follow_redirects=False).headers["location"]).group(1)
        _open(mine, "Other")                                      # another project, now open
        assert run_id not in mine.get("/ui/run").text
    with _client(tmp_path, subject="bob") as theirs:
        theirs.cookies.set("jbg_project", project, path="/ui")
        page = theirs.get(f"/ui/run?run={run_id}").text
        # The requested address is echoed in the language switch; the run's
        # panel and data are not shown.
        assert run_id[:8] + "</span></h2>" not in page and "data-run-poll" not in page
        assert "Öppna först ett projekt" in page
        refused = theirs.post(f"/ui/run/{run_id}/cancel", follow_redirects=False)
        assert refused.headers["location"] == "/ui/run?message=unavailable"


def test_times_are_shown_in_local_time():
    from JBGWebUI import _local_time
    utc = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)
    assert _local_time(utc.isoformat()) == utc.astimezone().strftime("%Y-%m-%d %H:%M")
    assert _local_time("not a time") == ""


def test_the_script_follows_runs_and_every_text_is_translated(tmp_path):
    import JBGWebUI
    with _client(tmp_path) as client:
        script = client.get("/ui/static/jbg.js").text
    assert "data-run-poll" in script and "/overview" in script and "location.reload" in script
    texts = set(STAGES.values()) | {t for _, t in MESSAGES.values()}
    texts |= {text for _, passed, failed, _ in PREFLIGHT for text in (passed, failed)}
    assert not sorted(t for t in texts if t not in JBGWebUI._CATALOGS["sv"])
