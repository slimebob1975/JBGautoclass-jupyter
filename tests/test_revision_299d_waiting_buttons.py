"""299d (asked 2026-11-13): action buttons follow the traffic-light convention
(design decision: ready green, blocked red with the reason on hover) while
there is nothing to do yet, and the footer's actions also stand at the top of
long pages, as do buttons below long tables. The behaviour itself (jbg.js)
was checked in headless Chromium; these tests cover what the server renders."""
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

sys.path.insert(0, str(Path(__file__).parent))
from test_revision_299b_csv_reclassification import _client, _setting  # noqa: E402


def _buttons(page, text):
    return re.findall(r"<button[^>]*>\s*" + re.escape(text) + r"\s*</button>", page)


def test_reviewing_corrections_waits_for_a_chosen_class(tmp_path):
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        page = client.get(f"/ui/results?run={s.run}&view=mispredictions").text
    buttons = _buttons(page, "Granska korrigeringarna")
    assert len(buttons) == 2                                      # above and below the table
    for button in buttons:
        assert 'data-wait="fields-any"' in button and "select[name^='corr:']" in button
        assert 'data-wait-reason="Välj rätt klass för minst en rad först."' in button


def test_saving_roles_waits_for_a_change_unless_unsaved(tmp_path):
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        suggested = client.get(f"/ui/dataset?dataset={s.dataset}&view=roles").text
        client.put(f"/api/projects/{s.project}/datasets/{s.dataset}/roles",
                   json={"columns": {"nr": "id", "x": "numeric", "art": "class"}})
        saved = client.get(f"/ui/dataset?dataset={s.dataset}&view=roles").text
        refused = client.post(f"/ui/dataset/{s.dataset}/roles",
                              data={"role:nr": "id", "role:x": "numeric", "categorical_limit": "30"})
    assert all("data-wait-dirty" in b for b in _buttons(suggested, "Spara kolumnroller"))
    assert len(_buttons(saved, "Spara kolumnroller")) == 2
    assert not any("data-wait-dirty" in b for b in _buttons(saved, "Spara kolumnroller"))
    assert refused.status_code == 422
    assert all("data-wait-dirty" in b for b in _buttons(refused.text, "Spara kolumnroller"))


def test_saving_a_configuration_waits_for_a_change_unless_refused(tmp_path):
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        cfg = client.post(f"/api/projects/{s.project}/configurations",
                          json={"name": "K", "settings": {}}).json()["configuration_id"]
        page = client.get(f"/ui/configuration?configuration={cfg}").text
        refused = client.post(f"/ui/configuration/{cfg}", data={"expected_version": "1",
                                                                "f:mode.test_size": "lots"})
    saves = _buttons(page, "Spara konfiguration")
    assert len(saves) == 2                                        # footer and its top copy
    assert all('data-wait="change"' in b and "data-wait-dirty" not in b for b in saves)
    assert refused.status_code == 422
    assert all("data-wait-dirty" in b for b in _buttons(refused.text, "Spara konfiguration"))


def test_forms_wait_for_their_fields(tmp_path):
    with _client(tmp_path) as client:
        projects = client.get("/ui/projects").text
        s = _setting(client, tmp_path)
        dataset = client.get("/ui/dataset").text
    create = _buttons(projects, "Skapa projekt")[0]
    assert 'data-wait="fields"' in create and 'data-wait-reason="Ange ett namn först."' in create
    upload = _buttons(dataset, "Ladda upp")[0]
    assert 'data-wait-fields="#csv-file, #csv-name"' in upload


def test_footer_actions_have_a_top_copy_shown_by_the_script(tmp_path):
    from JBGWebUI import WEB
    with _client(tmp_path) as client:
        s = _setting(client, tmp_path)
        page = client.get(f"/ui/dataset?dataset={s.dataset}").text
    top = page.split('data-top-actions hidden>')[1].split("</div>")[0]
    assert "Validera dataset" in top                              # hidden until the page is long
    script = (WEB / "static" / "jbg.js").read_text(encoding="utf-8")
    for name in ("wireWaiting", "wireTopActions", "aria-disabled", "data-wait-dirty"):
        assert name in script


def test_blocked_footer_buttons_say_why(tmp_path):
    with _client(tmp_path) as client:
        _setting(client, tmp_path)
        page = client.get("/ui/dataset").text
    blocked = _buttons(page, "Validera dataset")[0]
    assert "disabled" in blocked and 'title="Välj ett dataset i listan."' in blocked
