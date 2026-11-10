"""297a: inner progress (e.g. "12 of 40" algorithm combinations) in the live view
of a run, and progress pruning that keeps what the live view reads."""
import re
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus  # noqa: E402
from JBGRunJobStore import RunJobStore  # noqa: E402
from JBGRunOverview import INNER_KINDS, INNER_PROGRESS, project_run_overview  # noqa: E402


def _store(tmp_path, **options):
    store = RunJobStore(f"sqlite:///{(tmp_path / 'jobs.sqlite').as_posix()}", **options)
    store.create_tables()
    run = store.create("owner", DataSourceKind.CSV, {"spec_version": 1})
    store.claim(run, "w")
    return store, run


def _messages(store, run):
    return [event.message for event in store.progress(run)]


def _stage(store, run, status=RunStatus.RUNNING):
    return project_run_overview(run.key, status, store.progress(run))["stage"]


# --- The store -----------------------------------------------------------------------------------

def test_inner_progress_is_one_event_that_replaces_the_previous(tmp_path):
    store, run = _store(tmp_path)
    for done in range(5):
        store.record_inner_progress(run, "models", done, 8)
    inner = [m for m in _messages(store, run) if m.startswith("jbg_inner:")]
    assert inner == ["jbg_inner:models:4:8"]
    store.record_inner_progress(run, None)
    assert not any(m.startswith("jbg_inner:") for m in _messages(store, run))


@pytest.mark.parametrize("kind, done, total, error", [
    ("secret_table", 1, 2, ValueError), ("models", 3, 2, ValueError),
    ("models", -1, 2, ValueError), ("models", 0, 0, ValueError),
    ("models", 1.5, 2, TypeError), ("models", True, 2, TypeError),
])
def test_only_reviewed_kinds_and_sane_counts_are_stored(tmp_path, kind, done, total, error):
    store, run = _store(tmp_path)
    with pytest.raises(error):
        store.record_inner_progress(run, kind, done, total)
    assert _messages(store, run) == []


def test_unknown_runs_are_refused(tmp_path):
    store, _ = _store(tmp_path)
    with pytest.raises(KeyError):
        store.record_inner_progress(RunReference("00000000-0000-4000-8000-000000000000"),
                                    "models", 0, 1)


def test_pruning_keeps_the_stage_the_percentage_and_the_inner_progress(tmp_path):
    """A spot check records one event per combination; with more than the cap the
    live view lost its stage and percentage (found 2026-11-10)."""
    store, run = _store(tmp_path, max_events=20)
    store.record_progress(run, None, 0.35)
    store.record_progress(run, "jbg_stage:train_model")
    store.record_inner_progress(run, "models", 3, 40)
    for number in range(60):
        store.record_progress(run, f"Check and train algorithms (P-{number})")
    assert len(store.progress(run)) == 20
    assert _stage(store, run) == {"name": "train_model", "precision": "exact", "percent": 0.35,
                                  "inner": {"kind": "models", "done": 3, "total": 40}}


def test_pruning_still_drops_older_stages_and_percentages(tmp_path):
    store, run = _store(tmp_path, max_events=5)
    for number in range(10):
        store.record_progress(run, f"jbg_stage:{'fetch_data' if number < 9 else 'train_model'}",
                              number / 10)
    messages = _messages(store, run)
    assert len(messages) == 5 and messages[-1] == "jbg_stage:train_model"


# --- The overview --------------------------------------------------------------------------------

def _events(*messages):
    return [SimpleNamespace(message=message, percent=None) for message in messages]


def test_the_inner_progress_is_shown_only_while_running():
    events = _events("jbg_stage:train_model", "jbg_inner:models:2:9")
    running = project_run_overview("r", RunStatus.RUNNING, events)["stage"]
    assert running["inner"] == {"kind": "models", "done": 2, "total": 9}
    for status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
                   RunStatus.PENDING, RunStatus.INTERRUPTED):
        assert "inner" not in project_run_overview("r", status, events)["stage"], status


def test_a_later_stage_ends_the_inner_progress():
    events = _events("jbg_stage:train_model", "jbg_inner:models:9:9", "jbg_stage:evaluate_model")
    assert "inner" not in project_run_overview("r", RunStatus.RUNNING, events)["stage"]


@pytest.mark.parametrize("message", [
    "jbg_inner:secret:1:2", "jbg_inner:models:3:2", "jbg_inner:models:1:0",
    "jbg_inner:models:-1:2", "jbg_inner:models:1", "jbg_inner:models:a:b",
    "jbg_inner:models:1:2:extra",
])
def test_malformed_inner_markers_are_ignored(message):
    stage = project_run_overview("r", RunStatus.RUNNING, _events(message))["stage"]
    assert "inner" not in stage


def test_every_kind_has_a_label_in_the_panel():
    from pathlib import Path
    panel = (Path(__file__).resolve().parents[1] / "src" / "web" / "templates"
             / "_run_panel.html").read_text(encoding="utf-8")
    for kind in INNER_KINDS:
        assert f'"{kind}":' in panel, kind


# --- The worker's logger proxy -------------------------------------------------------------------

class _Logger:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append(name)


def _proxy(tmp_path, interval=0.0):
    from JBGRunWorker import StoreProgressLogger
    store, run = _store(tmp_path)
    proxy = StoreProgressLogger(_Logger(), store, run, SimpleNamespace(lost=False))
    proxy.INNER_INTERVAL = interval
    return proxy, store, run


def test_the_proxy_records_a_spot_check_and_forwards_every_call(tmp_path):
    proxy, store, run = _proxy(tmp_path)
    proxy.start_inline_progress("k", "Checking & Training models", 4, "Percent models checked")
    assert _stage(store, run)["inner"] == {"kind": "models", "done": 0, "total": 4}
    proxy.update_inline_progress("k", 2, "Percent models checked")
    assert _stage(store, run)["inner"]["done"] == 2
    proxy.reset_inline_progress("k")                    # "All algorithms failed. Restarting"
    assert _stage(store, run)["inner"]["done"] == 0
    proxy.update_inline_progress("k", 99, "x")          # beyond the total: capped
    assert _stage(store, run)["inner"]["done"] == 4
    proxy.end_inline_progress("k")
    # 297c: a finished bar stays full until the next stage marker hides it.
    assert _stage(store, run)["inner"] == {"kind": "models", "done": 4, "total": 4}
    store.record_progress(run, "jbg_stage:evaluate_model")
    assert "inner" not in _stage(store, run)
    assert proxy._logger.calls == ["start_inline_progress", "update_inline_progress",
                                   "reset_inline_progress", "update_inline_progress",
                                   "end_inline_progress"]


def test_the_proxy_throttles_the_writes(tmp_path):
    proxy, store, run = _proxy(tmp_path, interval=60.0)
    proxy.start_inline_progress("k", "Checking & Training models", 100, "t")
    for done in range(1, 99):
        proxy.update_inline_progress("k", done, "t")
    assert _stage(store, run)["inner"]["done"] == 0       # within the interval
    proxy.update_inline_progress("k", 100, "t")           # the last step is always recorded
    assert _stage(store, run)["inner"]["done"] == 100


def test_unknown_bars_are_forwarded_but_not_recorded(tmp_path):
    proxy, store, run = _proxy(tmp_path)
    proxy.start_inline_progress("k", "Something new with a table name", 5, "t")
    proxy.update_inline_progress("k", 3, "t")
    proxy.end_inline_progress("k")
    assert _messages(store, run) == [] and len(proxy._logger.calls) == 3


def test_parsing_a_dataset_is_recorded_through_the_proxy(tmp_path):
    proxy, store, run = _proxy(tmp_path)
    proxy.parse_dataset_progress("fetch", 250, 1000)
    assert _stage(store, run)["inner"] == {"kind": "fetch_data", "done": 250, "total": 1000}


def test_a_store_failure_never_stops_the_run(tmp_path, capsys):
    proxy, store, run = _proxy(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("database away")
    store.record_inner_progress = broken
    proxy.start_inline_progress("k", "Checking & Training models", 4, "t")
    proxy.update_inline_progress("k", 4, "t")
    proxy.end_inline_progress("k")
    assert "inner progress not recorded: RuntimeError" in capsys.readouterr().err


def test_every_engine_description_maps_to_a_known_kind():
    assert set(INNER_PROGRESS.values()) == INNER_KINDS


# --- The live view -------------------------------------------------------------------------------

def _client(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("jinja2")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject="developer",
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _running(client, kind="training"):
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    store = client.app.state.jbg["store"]
    run = store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "kind": kind}})
    store.claim(run, "w")
    store.record_progress(run, "jbg_stage:train_model", 0.3)
    return store, run


def _panel(page):
    return page.split('class="inner-progress"')[1].split("</div>\n    </div>")[0]


def test_the_training_tab_shows_the_inner_progress(tmp_path):
    with _client(tmp_path) as client:
        store, run = _running(client)
        store.record_inner_progress(run, "models", 3, 8)
        page = client.get(f"/ui/run?run={run.key}").text
        overview = client.get(f"/api/runs/{run.key}/overview").json()
    panel = _panel(page)
    assert " hidden" not in panel.split(">")[0]
    assert "<span data-inner-label>Algoritmkombinationer</span>" in panel
    assert "<span data-inner-count>3 av 8</span>" in panel
    assert 'aria-valuenow="38"' in panel and "width: 38%" in panel
    assert overview["stage"]["inner"] == {"kind": "models", "done": 3, "total": 8}


def test_without_inner_progress_the_block_is_hidden(tmp_path):
    with _client(tmp_path) as client:
        _, run = _running(client, kind="prediction")
        page = client.get(f"/ui/prediction?run={run.key}").text
    assert " hidden" in _panel(page).split(">")[0]


def test_a_finished_run_has_no_inner_block(tmp_path):
    with _client(tmp_path) as client:
        store, run = _running(client)
        store.record_inner_progress(run, "models", 8, 8)
        store.finish(run, "w", RunStatus.CANCELLED)
        page = client.get(f"/ui/run?run={run.key}").text
    assert "inner-progress" not in page


# --- A real training run (native: real worker, real spot check) ----------------------------------

import test_revision_281c_csv_application_e2e as revision_281c  # noqa: E402


class InnerProgressNative297a(revision_281c.CSVApplicationEndToEnd281c):
    test_registered_csv_trains_and_reloads_model_through_application = None
    test_other_scope_cannot_prepare_csv_run = None

    def test_a_spot_check_reports_its_algorithm_combinations(self):
        from fastapi.testclient import TestClient
        from JBGFastAPIServer import create_server_app
        from JBGMetadataStore import MetadataStoreSettings
        from test_revision_288a_project_runs import _setup
        from test_revision_296b_csv_predictions import _wait_for_workers_to_exit
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
            settings = self.config._config_to_json(username="")
            settings["mode"]["predict"] = False
            settings["mode"]["algorithm"] = ["LRN", "GNB", "DTC"]
            settings["mode"]["preprocessor"] = ["NOS", "STA", "MAX"]
            project, _, configuration = _setup(client, data=data, roles=settings)
            started = client.post(f"/api/projects/{project}/runs",
                                  json={"configuration_id": configuration})
            self.assertEqual(started.status_code, 201, started.text)
            run_id = started.json()["run_id"]
            seen, status, deadline = [], None, time.time() + 900
            while time.time() < deadline:
                overview = client.get(f"/api/runs/{run_id}/overview").json()
                status = overview["status"]
                if status in ("succeeded", "failed", "cancelled", "interrupted"):
                    break
                if "inner" in overview["stage"]:
                    seen.append(overview["stage"]["inner"])
                time.sleep(0.1)
            log = work / "run-logs" / f"run_{run_id}.log"
            tail = log.read_bytes().decode("utf-8", "replace")[-3000:] if log.exists() else ""
            self.assertEqual(status, "succeeded", tail)
            models = [inner for inner in seen if inner["kind"] == "models"]
            self.assertTrue(models, f"no algorithm combinations seen: {seen}")
            self.assertEqual({inner["total"] for inner in models}, {9})        # 3 x 3
            self.assertTrue(all(0 <= inner["done"] <= 9 for inner in models))
            done = [inner["done"] for inner in models]
            self.assertEqual(done, sorted(done))                             # never backwards
            # 297c: the finished spot check is left full; later stages hide it, and a
            # finished run shows no inner progress.
            events = client.app.state.jbg["store"].progress(RunReference(run_id))
            self.assertEqual([e.message for e in events
                              if (e.message or "").startswith("jbg_inner:")],
                             ["jbg_inner:models:9:9"])
            overview = client.get(f"/api/runs/{run_id}/overview").json()
            self.assertNotIn("inner", overview["stage"])
            _wait_for_workers_to_exit(client)
