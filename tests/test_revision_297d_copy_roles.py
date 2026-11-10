"""297d: copy the column roles of another dataset of the project (asked
2026-11-10). The roles view is filled in from the other dataset; nothing is
saved until the user saves, with the usual checks."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGDatasetRoles import DatasetRoles, copy_roles  # noqa: E402

SOURCE = b"id;sepal;petal;colour;note;species\n1;5.1;1.4;red;x;a\n2;7.0;4.7;blue;y;b\n3;6.3;6.0;red;z;a\n"
# Same columns, but no "petal", and "sepal" holds text here.
TARGET = b"id;sepal;colour;note;species;extra\n1;n/a;red;x;a;1\n2;7.0;blue;y;b;2\n3;6.3;red;z;a;3\n"
ROLES = {"role:id": "id", "role:sepal": "numeric", "role:petal": "numeric",
         "role:colour": "categorical", "role:note": "ignored", "role:species": "class",
         "categorical_limit": "12"}


# --- The helper ----------------------------------------------------------------------------------

def test_copy_keeps_the_roles_of_columns_that_exist():
    source = DatasetRoles.from_mapping({"id": "id", "sepal": "numeric", "petal": "numeric",
                                        "colour": "categorical", "note": "ignored",
                                        "species": "class"})
    kinds = {"id": "number", "sepal": "text", "colour": "text", "note": "text",
             "species": "text", "extra": "number"}
    mapping, missing, not_numeric = copy_roles(source, kinds)
    assert mapping == {"id": "id", "sepal": "numeric", "colour": "categorical",
                       "species": "class"}                    # "note" was ignored: not copied
    assert missing == ["petal"] and not_numeric == ["sepal"]


def test_identical_datasets_copy_everything():
    source = DatasetRoles.from_mapping({"id": "id", "x": "numeric", "c": "class"})
    assert copy_roles(source, {"id": "number", "x": "number", "c": "text"}) == (
        {"id": "id", "x": "numeric", "c": "class"}, [], [])


# --- The roles view ------------------------------------------------------------------------------

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


def _upload(client, name, data, role="training"):
    response = client.post("/ui/dataset/csv", data={"name": name, "role": role},
                           files={"file": (f"{name}.csv", data, "text/csv")},
                           follow_redirects=False)
    return re.search(r"dataset=([0-9a-f-]{36})", response.headers["location"]).group(1)


def _project(client):
    location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                           follow_redirects=False).headers["location"]
    project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
    client.post(f"/ui/projects/{project}/open", follow_redirects=False)
    source = _upload(client, "Källa", SOURCE)
    saved = client.post(f"/ui/dataset/{source}/roles", data=ROLES, follow_redirects=False)
    assert "roles_saved" in saved.headers["location"]
    target = _upload(client, "Mål", TARGET)
    return project, source, target


def _selected(page, column):
    select = re.search(rf'<select name="role:{column}"[^>]*>(.*?)</select>', page, re.S).group(1)
    return re.search(r'<option value="(\w+)" selected>', select).group(1)


def test_the_roles_view_offers_the_other_datasets_roles(tmp_path):
    with _client(tmp_path) as client:
        _, source, target = _project(client)
        page = client.get(f"/ui/dataset?dataset={target}&view=roles").text
    form = page.split('class="inline copy-roles"')[1].split("</form>")[0]
    assert f'<option value="{source}" >Källa</option>' in form
    assert "Kopiera kolumnroller från" in form and "Fyll i" in form


def test_filling_in_copies_the_roles_without_saving(tmp_path):
    with _client(tmp_path) as client:
        project, source, target = _project(client)
        page = client.get(f"/ui/dataset?dataset={target}&view=roles&roles_from={source}").text
        later = client.get(f"/ui/dataset?dataset={target}&view=roles").text
    assert "Kopierade från datasetet “Källa”" in page
    assert [_selected(page, c) for c in ("id", "sepal", "colour", "note", "species", "extra")] == [
        "id", "numeric", "categorical", "ignored", "class", "ignored"]
    assert 'name="categorical_limit" value="12"' in page          # the source's limit
    assert re.search(r'class="message warn">[^<]*kopierades inte:\s*petal</p>', page)
    assert re.search(r'class="message warn">[^<]*välj en annan roll:\s*sepal</p>', page)
    assert "Inga kolumnroller ännu" in later                        # nothing saved


def test_saving_the_copy_keeps_it(tmp_path):
    with _client(tmp_path) as client:
        project, source, target = _project(client)
        page = client.get(f"/ui/dataset?dataset={target}&view=roles&roles_from={source}").text
        form = {f"role:{c}": _selected(page, c)
                for c in ("id", "sepal", "colour", "note", "species", "extra")}
        form["role:sepal"] = "text"                  # the user follows the warning
        form["categorical_limit"] = "12"
        saved = client.post(f"/ui/dataset/{target}/roles", data=form, follow_redirects=False)
        after = client.get(f"/ui/dataset?dataset={target}&view=roles").text
    assert "roles_saved" in saved.headers["location"]
    assert _selected(after, "colour") == "categorical" and _selected(after, "sepal") == "text"


@pytest.mark.parametrize("which", ["itself", "prediction", "unknown", "other_project"])
def test_only_this_projects_other_training_datasets_are_offered(tmp_path, which):
    with _client(tmp_path) as client:
        project, _, target = _project(client)
        stranger = {"itself": target,
                    "prediction": _upload(client, "Ny", TARGET, role="prediction"),
                    "unknown": "00000000-0000-4000-8000-000000000000"}.get(which)
        if which == "other_project":
            location = client.post("/ui/projects", data={"name": "Q", "source": "csv"},
                                   follow_redirects=False).headers["location"]
            other = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
            client.post(f"/ui/projects/{other}/open", follow_redirects=False)
            stranger = _upload(client, "Annan", SOURCE)
            client.post(f"/ui/dataset/{stranger}/roles", data=ROLES, follow_redirects=False)
            client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        page = client.get(f"/ui/dataset?dataset={target}&view=roles&roles_from={stranger}").text
    assert "Kopierade från datasetet" not in page
    assert f'<option value="{stranger}"' not in page.split("copy-roles")[1].split("</form>")[0]


def test_without_other_datasets_with_roles_there_is_nothing_to_copy(tmp_path):
    with _client(tmp_path) as client:
        location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        only = _upload(client, "Ensam", SOURCE)
        page = client.get(f"/ui/dataset?dataset={only}&view=roles").text
    assert "copy-roles" not in page
