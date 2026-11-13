"""295d: the Overview tab, the About tab (user guide and version) and the title link."""
import re
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402

OWNER = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
CSV = b"id;sepal;species\n1;5.1;a\n2;7.0;b\n3;6.3;a\n4;4.9;b\n"
DOCS = Path(__file__).resolve().parents[1] / "docs"


def _client(tmp_path, subject="developer", version_line=None, monkeypatch=None):
    import JBGFastAPIServer
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    if version_line is not None:
        monkeypatch.setattr(JBGFastAPIServer, "_version_line", lambda: version_line)
    return TestClient(JBGFastAPIServer.create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _open(client, name="Iris-projektet"):
    location = client.post("/ui/projects", data={"name": name, "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    return project


def _run(client, project, kind="training", status=RunStatus.SUCCEEDED):
    store = client.app.state.jbg["store"]
    run = store.create(OWNER, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
        "project_id": project, "kind": kind, "configuration_name": "LR",
        "configuration_version": 1}})
    if status is RunStatus.PENDING:
        return run
    store.claim(run, "w")
    result = None
    if status is RunStatus.SUCCEEDED and kind == "training":
        result = {"schema_version": 1, "kind": "classification", "class_count": 2,
                  "mispredicted_count": 0,
                  "evaluation": {"schema_version": 1, "accuracy": 0.75, "balanced_accuracy": 0.625,
                                 "mcc": 0.5, "precision_macro": 0.75, "recall_macro": 0.75,
                                 "f1_macro": 0.7, "precision_weighted": 0.75,
                                 "recall_weighted": 0.75, "f1_weighted": 0.7,
                                 "confusion_matrix": [[2, 0], [1, 1]]}}
    store.finish(run, "w", status, result=result)
    return run


def _main(page):
    return page.split("<main>")[1].split("</main>")[0]


# --- Tabs and the title link ---------------------------------------------------------------------

def test_overview_and_about_are_linked_and_about_is_last(tmp_path):
    with _client(tmp_path) as client:
        page = client.get("/ui/projects").text
    nav = page.split('<nav class="tabs"')[1].split("</nav>")[0]
    labels = re.findall(r'<(?:a href="/ui/[a-z]+"[^>]*|span class="off"[^>]*)>([^<]+)</(?:a|span)>', nav)
    assert labels[0] == "Översikt" and labels[-2:] == ["Körhistorik", "Om"]
    assert '<a href="/ui/overview"' in nav and '<a href="/ui/about"' in nav
    assert re.search(r'<span class="off"[^>]*>Körhistorik</span>', nav)       # still to come


@pytest.mark.parametrize("path", ["/ui/projects", "/ui/models", "/ui/about"])
def test_the_title_links_to_the_overview(tmp_path, path):
    with _client(tmp_path) as client:
        _open(client)
        page = client.get(path).text
    brand = page.split('<div class="brand">')[1].split("</div>")[0]
    link = re.search(r'<a href="/ui/overview" class="brand-link"[^>]*>(.*?)</a>', brand, re.S)
    assert link and "JBG" in link.group(1) and "Automatisk klassificering" in link.group(1)


# --- Overview ----------------------------------------------------------------------------------

def test_overview_without_a_project_is_empty_apart_from_a_hint(tmp_path):
    with _client(tmp_path) as client:
        response = client.get("/ui/overview")
    main = _main(response.text)
    assert response.status_code == 200 and "overview-grid" not in main
    # 299c: the tab name in the sentence is the link.
    assert 'Öppna först ett projekt under fliken <a href="/ui/projects">Projekt</a>.' in main
    assert 'href="/ui/projects"' in main and 'href="/ui/about"' in main


def test_a_new_project_points_to_the_tabs_that_fill_it(tmp_path):
    with _client(tmp_path) as client:
        _open(client)
        main = _main(client.get("/ui/overview").text)
    assert "Iris-projektet" in main and "CSV" in main
    for text in ("Inget aktivt träningsdataset.", "Ingen konfiguration är vald för körning.",
                 "Inga körningar ännu.", "Modeller i projektet: 0 · tillgängliga: 0"):
        assert text in main, text
    for target in ("/ui/dataset", "/ui/configuration", "/ui/run", "/ui/prediction", "/ui/models"):
        assert f'href="{target}' in main, target


def test_the_overview_shows_dataset_configuration_runs_and_models(tmp_path):
    from JBGModelCatalog import ModelState
    with _client(tmp_path) as client:
        project = _open(client)
        uploaded = client.post("/ui/dataset/csv", data={"name": "Iris", "role": "training"},
                               files={"file": ("d.csv", CSV, "text/csv")}, follow_redirects=False)
        dataset = re.search(r"dataset=([0-9a-f-]{36})", uploaded.headers["location"]).group(1)
        client.post(f"/ui/dataset/{dataset}/activate", follow_redirects=False)
        created = client.post("/ui/configuration", data={"name": "Standard"}, follow_redirects=False)
        configuration = re.search(r"configuration=([0-9a-f-]{36})",
                                  created.headers["location"]).group(1)
        client.post(f"/ui/configuration/{configuration}/select", follow_redirects=False)
        training = _run(client, project)
        prediction = _run(client, project, kind="prediction", status=RunStatus.FAILED)
        catalog = client.app.state.jbg["models"]
        entry = catalog.register(OWNER, project, "LR v1 · test", "00000000-0000-4000-8000-0000000000aa",
                                 dataset_id=dataset, run_id=training.key)
        catalog.set_state(OWNER, project, entry.model_id, ModelState.AVAILABLE)
        client.post(f"/ui/models/{entry.model_id}/choose", follow_redirects=False)
        main = _main(client.get("/ui/overview").text)
    assert f'href="/ui/dataset?dataset={dataset}"' in main and "<b>Iris</b>" in main
    assert "<b>Standard</b> v1" in main and "Konfigurationer i projektet: 1" in main
    assert f'href="/ui/results?run={training.key}"' in main                  # finished: its results
    assert f'href="/ui/prediction?run={prediction.key}"' in main and "Misslyckades" in main
    assert "75,0 %" in main and "62,5 %" in main                              # Swedish decimal comma
    assert f'href="/ui/models?model={entry.model_id}"' in main
    assert "Modeller i projektet: 1 · tillgängliga: 1" in main


def test_an_unfinished_training_run_links_to_the_training_tab(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client)
        queued = _run(client, project, status=RunStatus.PENDING)
        main = _main(client.get("/ui/overview").text)
    assert f'href="/ui/run?run={queued.key}"' in main and "results?run=" not in main
    assert "I kö" in main


@pytest.mark.parametrize("claimed, label", [(False, "I kö"), (True, "Körs")])
def test_run_lists_name_the_status_of_runs_in_progress(tmp_path, claimed, label):
    """The run lists showed the stored words "pending" and "running" untranslated."""
    with _client(tmp_path) as client:
        project = _open(client)
        store = client.app.state.jbg["store"]
        for kind in ("training", "prediction"):
            run = store.create(OWNER, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
                "project_id": project, "kind": kind}})
            if claimed:
                store.claim(run, "w")
        for path in ("/ui/run", "/ui/prediction", "/ui/overview"):
            badges = re.findall(r'<span class="badge [^"]*">([^<]*)</span>', client.get(path).text)
            assert label in badges and not {"pending", "running"} & set(badges), path


def test_the_overview_never_shows_another_users_project(tmp_path):
    with _client(tmp_path) as client:
        project = _open(client, name="Hemligt")
    with _client(tmp_path, subject="someone-else") as other:
        other.cookies.set("jbg_project", project, path="/ui")
        main = _main(other.get("/ui/overview").text)
    assert "Hemligt" not in main and "Öppna först ett projekt" in main


# --- About -------------------------------------------------------------------------------------

def test_about_shows_the_swedish_guide_and_the_version(tmp_path, monkeypatch):
    with _client(tmp_path, version_line="v9.9.9 · Git: test", monkeypatch=monkeypatch) as client:
        page = client.get("/ui/about").text
    main = _main(page)
    assert '<article class="guide" lang="sv">' in main and "Arbetsgången" in main
    assert "<h2>Version</h2>" in main and "v9.9.9 · Git: test" in main
    assert '<a href="#arbetsgången">Arbetsgången</a>' in main                  # contents
    assert '<h3 id="arbetsgången">' in main


def test_about_follows_the_language_choice(tmp_path):
    with _client(tmp_path) as client:
        client.cookies.set("jbg_lang", "en", path="/ui")
        main = _main(client.get("/ui/about").text)
    assert '<article class="guide" lang="en">' in main and "The work order" in main
    assert "Arbetsgången" not in main


def test_about_without_a_guide_says_so(tmp_path, monkeypatch):
    import JBGWebUIAbout
    monkeypatch.setattr(JBGWebUIAbout, "GUIDES", tmp_path / "no-docs")
    with _client(tmp_path) as client:
        response = client.get("/ui/about")
    assert response.status_code == 200
    assert "Användarhandledningen är inte tillgänglig." in response.text


def test_a_missing_language_falls_back_to_the_other(tmp_path):
    from JBGWebUIAbout import load_guide
    (tmp_path / "user_guide.sv.md").write_text("# Hej", encoding="utf-8")
    assert load_guide("en", tmp_path) == ("# Hej", "sv")
    assert load_guide("sv", tmp_path) == ("# Hej", "sv")
    assert load_guide("../etc", tmp_path) == ("# Hej", "sv")          # only the known languages
    assert load_guide("sv", tmp_path / "missing") == (None, None)


# --- The Markdown subset -------------------------------------------------------------------------

def test_the_renderer_escapes_all_text():
    from JBGWebUIAbout import render_markdown
    html, _ = render_markdown('# <b>T</b>\n\n<script>alert(1)</script> & "x"\n\n- <i>y</i>')
    assert "<script>" not in html and "&lt;script&gt;" in html and "&amp;" in html
    assert "<b>T</b>" not in html and "<i>" not in html


@pytest.mark.parametrize("target, linked", [
    ("/ui/projects", True), ("#arbetsgangen", True), ("https://example.org/a", True),
    ("javascript:alert(1)", False), ("//evil.example/ui/", False), ("/ui//evil", False),
    ("http://example.org/", False), ("/docs", False), ("data:text/html,x", False), ("#", False),
])
def test_links_go_only_to_the_interface_anchors_or_https(target, linked):
    from JBGWebUIAbout import render_markdown
    html, _ = render_markdown(f"See [the page]({target}).")
    assert ("<a href=" in html) is linked
    assert "the page" in html                                     # the text stays either way
    if target.startswith("https://"):
        assert 'rel="noopener noreferrer"' in html


def test_the_supported_blocks():
    from JBGWebUIAbout import render_markdown
    text = ("# Title\n\nOne\nparagraph with **bold** and `co**de`.\n\n"
            "## Part\n\n- first\n  continued\n- second\n\n1. one\n2. two\n\n"
            "> a note\n> on two lines\n\n### Detail\n\n## Part\n")
    html, contents = render_markdown(text)
    assert '<h2 id="title">Title</h2>' in html
    assert "<p>One paragraph with <strong>bold</strong> and <code>co**de</code>.</p>" in html
    assert "<ul><li>first continued</li><li>second</li></ul>" in html
    assert "<ol><li>one</li><li>two</li></ol>" in html
    assert "<blockquote><p>a note on two lines</p></blockquote>" in html
    assert '<h4 id="detail">Detail</h4>' in html
    assert contents == [("part", "Part"), ("part-2", "Part")]           # unique anchors


# --- The guides --------------------------------------------------------------------------------

def _headings(language, level):
    text = (DOCS / f"user_guide.{language}.md").read_text(encoding="utf-8")
    return [line[level + 1:].strip() for line in text.splitlines()
            if line.startswith("#" * level + " ")]


def test_both_guides_have_the_same_structure():
    for level in (1, 2, 3):
        assert len(_headings("sv", level)) == len(_headings("en", level)), level


@pytest.mark.parametrize("language", ["sv", "en"])
def test_the_guide_has_a_section_for_every_tab(language):
    from JBGWebUI import TABS, translator
    _ = translator(language)
    sections = _headings(language, 2)
    for key, label, _available in TABS:
        if key != "about":                                    # the guide is the About tab
            assert _(label) in sections, (language, label)


@pytest.mark.parametrize("language", ["sv", "en"])
def test_the_guide_renders_cleanly(language):
    from JBGWebUIAbout import load_guide, render_markdown
    text, shown = load_guide(language)
    assert shown == language
    html, contents = render_markdown(text)
    plain = re.sub(r"<code>.*?</code>", "", str(html))
    assert "**" not in plain and "`" not in plain                # no unbalanced markup
    assert len(contents) >= 8
