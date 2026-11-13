"""299a2 (asked 2026-11-12): the result tables' column headings, and the
mispredictions' model names, in the interface's language. Class names,
P(<class>) and feature names are data and stay as they are."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import DataSourceKind  # noqa: E402

LEADERBOARD = (";Oversampling;Undersampling;Preprocessor;Feature Reduction;Algorithm (Library);Mean cv;"
               "Stdev.;Elapsed Time;Exception\n0;NOG;NUG;NOS;NOR;LRN - Logistic Regression;0,9;0,1;1,2;\n")
MISPLACED = "Nr;P(Actual);P(katt);Actual;Predicted;Model\n7;0,8;0,2;Actual;katt;cross trained\n8;0,4;0,6;katt;Actual;retrained\n"
IMPORTANCE = ";Rank;Feature;Mean score decrease;Std. across repeats;Repeat 1;Repeat 2\n0;1;Model;0,2;0,01;0,19;0,21\n"


def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _run(client, tmp_path):
    from JBGApplicationContracts import RunStatus
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    store = client.app.state.jbg["store"]
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    run = store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "model_id": "m", "configuration_version": 1}})
    store.claim(run, "w")
    store.finish(run, "w", RunStatus.SUCCEEDED, result={})
    csvs = tmp_path / "runs" / project / run.key / "output" / "csvs"
    csvs.mkdir(parents=True)
    (csvs / "crossval_p__train.csv").write_text(LEADERBOARD, encoding="utf-8")
    (csvs / "misplaced_p__train.csv").write_text(MISPLACED, encoding="utf-8")
    (csvs / "feature_importance_p__train.csv").write_text(IMPORTANCE, encoding="utf-8")
    return run.key


def _panel(page):
    return page.split('<aside class="card wide">')[1]


def _heads(page):
    table = _panel(page).split('<table class="grid">')[1].split("</thead>")[0]
    return re.findall(r"<th>([^<]*)</th>", table)


def _cells(page):
    body = _panel(page).split('<table class="grid">')[1].split("</tbody>")[0].split("<tbody>")[1]
    return re.findall(r"<td[^>]*>([^<]*)</td>", body)


def test_swedish_headings_and_model_names(tmp_path):
    with _client(tmp_path) as client:
        run = _run(client, tmp_path)
        mispredicted = client.get(f"/ui/results?run={run}&view=mispredictions").text
        board = client.get(f"/ui/results?run={run}&view=leaderboard").text
        importance = client.get(f"/ui/results?run={run}&view=feature_importance").text
    assert _heads(mispredicted) == ["Nr", "P(Actual)", "P(katt)", "Faktisk klass",
                                    "Predikterad klass", "Modell"]
    cells = _cells(mispredicted)
    assert "Korstränad modell" in cells and "Slutlig modell" in cells
    assert cells.count("Actual") == 2                          # a class named "Actual" stays
    assert _heads(board) == ["Översampling", "Undersampling", "Skalning", "Dimensionsreduktion",
                             "Algoritm (bibliotek)", "Medelvärde CV", "Standardavvikelse",
                             "Tid (s)", "Fel"]
    assert _heads(importance) == ["Rang", "Variabel", "Medelminskning av poäng",
                                  "Standardavvikelse över upprepningar", "Upprepning 1",
                                  "Upprepning 2"]
    assert "Model" in _cells(importance)                       # a feature named "Model" stays


def test_english_headings(tmp_path):
    with _client(tmp_path) as client:
        run = _run(client, tmp_path)
        client.post("/ui/language", data={"language": "en", "back": "/ui/projects"},
                    follow_redirects=False)
        mispredicted = client.get(f"/ui/results?run={run}&view=mispredictions").text
        board = client.get(f"/ui/results?run={run}&view=leaderboard").text
    assert _heads(mispredicted)[3:] == ["Actual class", "Predicted class", "Model"]
    assert "Cross-trained model" in _cells(mispredicted)
    assert _heads(board)[4:7] == ["Algorithm (library)", "Mean CV score", "Standard deviation"]
