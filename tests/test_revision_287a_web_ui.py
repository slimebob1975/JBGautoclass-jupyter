"""287a: web interface shell and Projects tab, cross-site protection."""
import re
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")
pytest.importorskip("babel")

from fastapi.testclient import TestClient  # noqa: E402

from JBGFastAPIServer import create_server_app  # noqa: E402
from JBGMetadataStore import MetadataStoreSettings  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


def _client(tmp_path, subject="developer"):
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject=subject,
        run_logs=tmp_path / "run-logs", csv_artifacts=tmp_path / "csv" / "objects",
        csv_root=tmp_path / "csv",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _create(client, name, source="sql", description=""):
    response = client.post("/ui/projects", data={"name": name, "source": source,
                                                 "description": description},
                           follow_redirects=False)
    assert response.status_code == 303
    return response.headers["location"]


# --- Navigation and language ------------------------------------------------------------

def test_root_leads_to_the_interface_and_docs_remain(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/", follow_redirects=False).headers["location"] == "/ui/"
        assert client.get("/ui/", follow_redirects=False).headers["location"] == "/ui/projects"
        assert client.get("/docs").status_code == 200
        assert client.get("/ui/static/jbg.css").status_code == 200


def test_swedish_by_default_english_on_request_and_choice_remembered(tmp_path):
    with _client(tmp_path) as client:
        page = client.get("/ui/projects").text
        assert "Dina projekt" in page and '<html lang="sv">' in page
        english = client.get("/ui/projects", headers={"accept-language": "en-GB,en;q=0.8"}).text
        assert "Your projects" in english
        chosen = client.post("/ui/language", data={"language": "en", "back": "/ui/projects"},
                             follow_redirects=False)
        assert chosen.headers["location"] == "/ui/projects"
        cookie = chosen.headers["set-cookie"]
        assert "jbg_lang=en" in cookie and "HttpOnly" in cookie and "Path=/ui" in cookie
        assert "Your projects" in client.get("/ui/projects", headers={"accept-language": "sv"}).text


@pytest.mark.parametrize("back", ["https://evil.example/", "//evil.example/ui/", "/docs", ""])
def test_language_switch_never_redirects_elsewhere(tmp_path, back):
    with _client(tmp_path) as client:
        response = client.post("/ui/language", data={"language": "sv", "back": back},
                               follow_redirects=False)
        assert response.headers["location"] == "/ui/projects"


def test_unavailable_tabs_are_shown_but_not_linked(tmp_path):
    with _client(tmp_path) as client:
        page = client.get("/ui/projects").text
    # 291a, 292a and 294a made Dataset, Configuration and Run available.
    for label in ("Översikt", "Resultat", "Modeller", "Körhistorik"):
        assert re.search(rf'<span class="off" aria-disabled="true"[^>]*>{label}</span>', page), label
    assert '<a href="/ui/projects" class="on"' in page
    assert '<a href="/ui/dataset"' in page and '<a href="/ui/configuration"' in page
    assert '<a href="/ui/run"' in page


# --- Projects tab -----------------------------------------------------------------------------

def test_create_rename_and_open_a_project(tmp_path):
    with _client(tmp_path) as client:
        location = _create(client, "Ärenden", description="Fem ärendetyper")
        page = client.get(location).text
        assert "Projektet är skapat." in page and "Fem ärendetyper" in page
        project_id = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        renamed = client.post(f"/ui/projects/{project_id}", data={"name": "Ärenden 2026",
                                                                  "description": "Ny"},
                              follow_redirects=True).text
        assert "Ändringarna är sparade." in renamed and "Ärenden 2026" in renamed
        opened = client.post(f"/ui/projects/{project_id}/open", follow_redirects=False)
        assert "jbg_project=" + project_id in opened.headers["set-cookie"]
        page = client.get(opened.headers["location"]).text
        assert "Projektet är öppnat." in page
        assert re.search(r'<b>Ärenden 2026</b>\s*</span>', page)          # context bar
        assert "Det här projektet är öppet." in page                         # footer hint


def test_errors_are_fixed_messages_and_names_are_escaped(tmp_path):
    with _client(tmp_path) as client:
        _create(client, "Iris")
        duplicate = client.get(_create(client, "IRIS")).text
        assert "Det finns redan ett projekt med det namnet." in duplicate
        invalid = client.get(_create(client, "x" * 201)).text
        assert "Uppgifterna godtogs inte." in invalid and "x" * 201 not in invalid
        wrong_source = client.get(_create(client, "Y", source="file")).text
        assert "Uppgifterna godtogs inte." in wrong_source
        page = client.get(_create(client, '<script>alert("x")</script>')).text
        assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_another_users_project_cannot_be_seen_changed_or_opened(tmp_path):
    with _client(tmp_path) as mine:
        location = _create(mine, "Mine")
        project_id = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    with _client(tmp_path, subject="bob") as theirs:
        assert "Mine" not in theirs.get(f"/ui/projects?selected={project_id}").text
        changed = theirs.post(f"/ui/projects/{project_id}", data={"name": "Bob's"},
                              follow_redirects=True).text
        assert "Projektet är inte längre tillgängligt." in changed
        opened = theirs.post(f"/ui/projects/{project_id}/open", follow_redirects=False)
        assert "set-cookie" not in opened.headers
        theirs.cookies.set("jbg_project", project_id, path="/ui")
        assert "Inget valt" in theirs.get("/ui/projects").text            # not resolved for bob
    with _client(tmp_path) as mine:
        assert "Mine" in mine.get("/ui/projects").text


# --- Cross-site protection ------------------------------------------------------------------

@pytest.mark.parametrize("headers", [{"origin": "http://evil.example"},
                                     {"origin": "null"},
                                     {"sec-fetch-site": "cross-site"},
                                     {"sec-fetch-site": "same-site"}])
def test_cross_site_changes_are_refused(tmp_path, headers):
    with _client(tmp_path) as client:
        for method, url, kwargs in (
                ("post", "/ui/projects", {"data": {"name": "X", "source": "sql"}}),
                ("post", "/api/projects", {"json": {"name": "X", "source": "sql"}})):
            response = getattr(client, method)(url, headers=headers, follow_redirects=False, **kwargs)
            assert (response.status_code, response.json()) == (
                403, {"detail": "cross-site request refused"})
        assert client.get("/api/projects").json() == []
        assert client.get("/ui/projects", headers=headers).status_code == 200    # reading is fine


def test_same_origin_changes_are_allowed(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/ui/projects", data={"name": "X", "source": "sql"},
                               headers={"origin": "http://testserver",
                                        "sec-fetch-site": "same-origin"},
                               follow_redirects=False)
        assert response.status_code == 303


# --- Translations ---------------------------------------------------------------------------

def test_every_interface_text_has_a_swedish_translation():
    from JBGWebUI import MESSAGES, TABS, translator
    texts = set()
    for template in (SRC / "web" / "templates").glob("*.html"):
        source = template.read_text(encoding="utf-8")
        texts |= set(re.findall(r'_\("([^"]+)"\)', source))
        texts |= set(re.findall(r"_\('([^']+)'\)", source))
        texts |= set(re.findall(r'\{% set title = "([^"]+)" %\}', source))
    texts |= {label for _, label, _ in TABS} | {text for _, text in MESSAGES.values()}
    # 291a: the Dataset tab's messages, label maps and the dataset check list.
    import JBGWebUIDatasets
    from JBGDatasetInspection import CHECKS
    texts |= {text for _, text in JBGWebUIDatasets.MESSAGES.values()}
    texts |= set(JBGWebUIDatasets.READ_ERRORS.values())
    texts |= {text for _, passed, failed in CHECKS for text in (passed, failed)}
    from JBGDatasetInspection import COMPATIBILITY_CHECK            # 293a
    texts |= set(COMPATIBILITY_CHECK[1:])
    for labels in re.findall(r"\{% set \w+_labels = (\{[^}]+\}) %\}",
                             (SRC / "web" / "templates" / "dataset.html").read_text(encoding="utf-8")):
        texts |= set(re.findall(r'"[a-z]+": "([^"]+)"', labels))
    texts |= {"Preview", "Profile", "Classes"}
    # An entry must exist; words spelled the same in Swedish (e.g. "Status",
    # "SQL Server", "Dataset") are recorded as such in the catalogue.
    import JBGWebUI
    missing = sorted(text for text in texts if text not in JBGWebUI._CATALOGS["sv"])
    assert not missing, missing
