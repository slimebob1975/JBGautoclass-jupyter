"""295e: the JBG logo in the header, a header that wraps on narrow screens, and
the interface opening on the Overview."""
import json
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


def _client(tmp_path):
    from JBGFastAPIServer import create_server_app
    from JBGMetadataStore import MetadataStoreSettings
    from JBGSQLServer import SQLServerSettings
    return TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"), subject="developer",
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs", sql_server=SQLServerSettings(),
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))


def _brand(page):
    return page.split('<div class="brand">')[1].split("</a>")[0]


@pytest.mark.parametrize("path", ["/ui", "/ui/"])
def test_the_interface_opens_on_the_overview(tmp_path, path):
    with _client(tmp_path) as client:
        assert client.get(path, follow_redirects=False).headers["location"] == "/ui/overview"
        assert client.get(path).status_code == 200


def test_the_web_and_voila_interfaces_share_one_logo_file():
    import JBGWebUI
    settings = json.loads((SRC / "GUI" / "default_settings.json").read_text(encoding="utf-8"))
    assert JBGWebUI.LOGO == (SRC / "GUI" / settings["logo"]).resolve()
    assert JBGWebUI.LOGO.is_file()


def test_the_header_shows_the_logo_image(tmp_path):
    import JBGWebUI
    with _client(tmp_path) as client:
        brand = _brand(client.get("/ui/projects").text)
        image = client.get("/ui/logo.png")
    assert '<img class="logo" src="/ui/logo.png" alt="JBG"' in brand
    assert '<span class="logo">' not in brand
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.content == JBGWebUI.LOGO.read_bytes()
    assert "max-age" in image.headers["cache-control"]


def test_without_the_file_the_header_keeps_the_text_logo(tmp_path, monkeypatch):
    import JBGWebUI
    monkeypatch.setattr(JBGWebUI, "LOGO", tmp_path / "missing.png")
    with _client(tmp_path) as client:
        brand = _brand(client.get("/ui/projects").text)
        image = client.get("/ui/logo.png")
    assert '<span class="logo">JBG</span>' in brand and "<img" not in brand   # no broken image
    assert image.status_code == 404


def test_the_header_wraps_instead_of_squeezing_the_language_switch():
    css = (SRC / "web" / "static" / "jbg.css").read_text(encoding="utf-8")
    assert ".masthead { flex-wrap: wrap;" in css
    assert ".lang { flex-shrink: 0; }" in css
