"""299a (agreed 2026-11-12): correcting mispredicted rows of SQL training runs.

The Results tab's mispredictions get a "right class" choice per row; the
corrections are reviewed and only written after confirmation, under the user's
own SQL sign-in, in one parameterised transaction (the Voila GUI wrote per
click and put the class into the SQL text). A log in the run's folder records
each correction; each can be undone.
"""
import json
import re
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGReclassification import (CorrectionFailed, CorrectionRefused,  # noqa: E402
                                 SQLReclassificationService)
from JBGSQLServer import SQLServerSettings  # noqa: E402

SERVER = SQLServerSettings(host="sql.example.invalid", driver="SQL Server", probe_catalog="Db")
MISPLACED = ("Nr;P(hund);P(katt);Actual;Predicted;Model;x\n"
             "36;0,8;0,2;katt;hund;cross trained;1,5\n"
             "36;0,7;0,3;katt;hund;final;1,5\n"
             "9;0,1;0,9;hund;katt;cross trained;2,5\n")


class Helper:
    """A fake SQL connection: records statements, parameters and the outcome."""

    def __init__(self, fail_on=None):
        self.executed, self.outcome, self.fail_on = [], None, fail_on

    def execute_query_with_params(self, query, params, get_data=False, commit=True):
        assert commit is False                                    # one transaction
        if self.fail_on is not None and len(self.executed) == self.fail_on:
            raise RuntimeError("[driver] UPDATE permission denied on dbo.Djur")
        self.executed.append((query, params))
        return 1

    def disconnect(self, commit=False):
        self.outcome = "committed" if commit else "rolled back"


def _settings():
    from JBGConfigurationService import _server_template
    settings = _server_template()
    settings["connection"].update(data_catalog="Db", data_table="dbo.Djur", id_column="Nr",
                                  class_column="Art", data_numerical_columns="x")
    return settings


def _run(store, results_root, project, *, source=DataSourceKind.SQL, kind="training",
         status=RunStatus.SUCCEEDED, owner=None):
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    owner = owner or AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    run = store.create(owner, source, {"spec_version": 1, "configuration": _settings(),
                                       "project_run": {"project_id": project, "kind": kind,
                                                       "model_id": "m", "configuration_version": 1}})
    store.claim(run, "w")
    store.finish(run, "w", status, result={})
    folder = results_root / project / run.key
    (folder / "output" / "csvs").mkdir(parents=True)
    (folder / "output" / "csvs" / "misplaced_r__train.csv").write_text(MISPLACED, encoding="utf-8")
    (folder / "class_labels.json").write_text(json.dumps({"class_labels": ["hund", "katt"]}))
    return run.key, folder


@pytest.fixture
def setting(tmp_path):
    from JBGRunJobStore import RunJobStore
    from JBGRunResults import RunResultsService
    store = RunJobStore.sqlite(tmp_path / "jobs.sqlite")
    store.create_tables()
    results = RunResultsService(store, results_root=tmp_path / "runs")
    helper = Helper()
    sessions = SimpleNamespace(data_layer_for=lambda principal, config, logger: SimpleNamespace(
        get_connection=lambda: helper))
    service = SQLReclassificationService(store, results, sessions, SERVER)
    run, folder = _run(store, tmp_path / "runs", "11111111-1111-4111-8111-111111111111")
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    principal = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer")
    return SimpleNamespace(store=store, service=service, helper=helper, run=run, folder=folder,
                           principal=principal, owner=principal.principal_key, root=tmp_path / "runs")


# --- The service --------------------------------------------------------------------------------

def test_review_lists_real_changes_only(setting):
    changes = setting.service.review(setting.owner, setting.run, {"36": "hund", "9.0": "hund"})
    assert changes == [{"id": 36, "old": "katt", "new": "hund"}]           # 9 is already hund
    with pytest.raises(CorrectionRefused, match="unknown class"):
        setting.service.review(setting.owner, setting.run, {"36": "älg"})
    with pytest.raises(CorrectionRefused, match="unknown row"):
        setting.service.review(setting.owner, setting.run, {"77": "hund"})
    with pytest.raises(CorrectionRefused, match="nothing to change"):
        setting.service.review(setting.owner, setting.run, {"36": ""})


def test_corrections_are_parameterised_in_one_transaction_and_logged(setting):
    count = setting.service.apply(setting.principal, setting.run, {"36": "hund", "9": "katt"})
    assert count == 2 and setting.helper.outcome == "committed"
    statement = "UPDATE [Db].[dbo].[Djur] SET [Art] = ? WHERE [Nr] = ?"
    assert setting.helper.executed == [(statement, ("katt", 9)), (statement, ("hund", 36))]
    log = json.loads((setting.folder / "corrections.json").read_text(encoding="utf-8"))
    assert [(e["id"], e["old"], e["new"], e["undone_at"]) for e in log] == [
        (9, "hund", "katt", None), (36, "katt", "hund", None)]
    assert setting.service.current(setting.owner, setting.run) == {9: "katt", 36: "hund"}


def test_a_failed_write_writes_nothing_and_says_little(setting):
    setting.helper.fail_on = 1
    with pytest.raises(CorrectionFailed) as failed:
        setting.service.apply(setting.principal, setting.run, {"36": "hund", "9": "katt"})
    assert setting.helper.outcome == "rolled back"
    assert "dbo.Djur" not in str(failed.value) and "permission" not in str(failed.value)
    assert not (setting.folder / "corrections.json").exists()


def test_undo_writes_the_old_class_back(setting):
    setting.service.apply(setting.principal, setting.run, {"36": "hund"})
    setting.helper.executed.clear()
    entry = setting.service.undo(setting.principal, setting.run, 0)
    assert setting.helper.executed[0][1] == ("katt", 36) and entry["undone_at"]
    assert setting.service.current(setting.owner, setting.run) == {}
    with pytest.raises(CorrectionRefused):
        setting.service.undo(setting.principal, setting.run, 0)            # already undone


@pytest.mark.parametrize("change", [{"source": DataSourceKind.CSV}, {"kind": "prediction"},
                                    {"status": RunStatus.FAILED}, {"owner": "someone-else"}])
def test_only_finished_own_sql_training_runs(setting, change):
    run, _ = _run(setting.store, setting.root, "22222222-2222-4222-8222-222222222222", **change)
    assert setting.service.correctable(setting.owner, run) is False
    with pytest.raises(CorrectionRefused):
        setting.service.review(setting.owner, run, {"36": "hund"})


def test_a_class_with_quotes_stays_a_parameter(setting):
    (setting.folder / "class_labels.json").write_text(json.dumps({"class_labels": ["hund", "k'att"]}))
    setting.service.apply(setting.principal, setting.run, {"9": "k'att"})
    statement, params = setting.helper.executed[0]
    assert "k'att" not in statement and params == ("k'att", 9)


# --- The Results tab ----------------------------------------------------------------------------

def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SERVER,
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _tab(client, tmp_path, helper):
    location = client.post("/ui/projects", data={"name": "S", "source": "sql"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    sessions = client.app.state.jbg["sessions"]
    sessions.data_layer_for = lambda principal, config, logger: SimpleNamespace(
        get_connection=lambda: helper)
    run, _ = _run(client.app.state.jbg["store"], tmp_path / "runs", project)
    return run


def test_pick_review_confirm_and_undo(tmp_path):
    helper = Helper()
    with _client(tmp_path) as client:
        run = _tab(client, tmp_path, helper)
        page = client.get(f"/ui/results?run={run}&view=mispredictions").text
        review = client.post(f"/ui/results/{run}/corrections/review",
                             data={"corr:36": "hund", "corr:9": ""})
        nothing_yet = list(helper.executed)
        written = client.post(f"/ui/results/{run}/corrections", data={"corr:36": "hund"},
                              follow_redirects=False)
        after = client.get(written.headers["location"]).text
        undone = client.post(f"/ui/results/{run}/corrections/0/undo", follow_redirects=False)
        final = client.get(undone.headers["location"]).text
    assert page.count('name="corr:36"') == 1 and 'name="corr:9"' in page   # once per row ID
    assert "Granska korrigeringarna" in page
    assert "Bekräfta korrigeringarna" in review.text and "<b>hund</b>" in review.text
    assert nothing_yet == []                                            # review writes nothing
    assert "message=corrected" in written.headers["location"]
    assert "Korrigeringarna är skrivna till källtabellen." in after
    assert "korrigerad till hund" in after and 'name="corr:36"' not in after
    assert "Korrigeringen är ångrad" in final and 'name="corr:36"' in final
    assert "ångrad" in final


def test_an_empty_review_and_a_refused_write_explain_themselves(tmp_path):
    helper = Helper(fail_on=0)
    with _client(tmp_path) as client:
        run = _tab(client, tmp_path, helper)
        empty = client.post(f"/ui/results/{run}/corrections/review", data={"corr:36": ""},
                            follow_redirects=False)
        failed = client.post(f"/ui/results/{run}/corrections", data={"corr:36": "hund"},
                             follow_redirects=False)
        page = client.get(failed.headers["location"]).text
    assert "message=correction_refused" in empty.headers["location"]
    assert "message=correction_failed" in failed.headers["location"]
    assert "Källtabellen kunde inte uppdateras, så inget skrevs." in page


def test_csv_runs_show_no_correction_controls(tmp_path):
    with _client(tmp_path) as client:
        location = client.post("/ui/projects", data={"name": "C", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        run, _ = _run(client.app.state.jbg["store"], tmp_path / "runs", project,
                      source=DataSourceKind.CSV)
        page = client.get(f"/ui/results?run={run}&view=mispredictions").text
    assert "corr:" not in page and "Granska korrigeringarna" not in page
