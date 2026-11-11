"""298a: suggested column roles for a dataset without roles (requested
2026-11-11), and the column shown on the Classes view as the class column.
Agreed 2026-11-11: suggestions only fill the form in; nothing is saved until
the user chooses "Spara kolumnroller"."""
import re
from datetime import timedelta

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from JBGDatasetRoles import suggest_roles, with_class_column  # noqa: E402


def _profile_column(name, kind, distinct, missing=0):
    return {"name": name, "type": kind, "distinct": distinct, "missing": missing}


# --- The suggestion --------------------------------------------------------------------------

def test_each_column_gets_a_role_except_the_class_column():
    columns = [_profile_column("ref", "text", 9, missing=1),      # unique but not filled in
               _profile_column("nr", "number", 10),               # first filled-in unique column
               _profile_column("size", "number", 7),
               _profile_column("colour", "text", 3),
               _profile_column("note", "text", 10),
               _profile_column("empty", "text", 0, missing=10),
               _profile_column("same", "text", 1),
               _profile_column("seq", "number", 10),              # unique too, but not the first
               _profile_column("species", "text", 3)]
    assert suggest_roles(columns, 10, categorical_limit=5) == {
        "ref": "text", "nr": "id", "size": "numeric", "colour": "categorical", "note": "text",
        "empty": "ignored", "same": "ignored", "seq": "numeric", "species": "categorical"}


def test_the_class_column_is_only_the_one_given():
    columns = [_profile_column("label", "text", 10), _profile_column("nr", "number", 10),
               _profile_column("x", "number", 4)]
    assert suggest_roles(columns, 10, class_column="label") == {
        "label": "class", "nr": "id", "x": "numeric"}                # the ID skips the class
    assert suggest_roles(columns, 10, class_column="gone")["label"] == "id"


def test_sql_type_names_are_understood():
    columns = [_profile_column("Key", "uniqueidentifier", 50), _profile_column("Amount", "decimal", 20),
               _profile_column("Kind", "NVARCHAR", 4), _profile_column("Text", "ntext", 45),
               _profile_column("Created", "datetime", 30), _profile_column("Blob", "varbinary", None)]
    assert suggest_roles(columns, 50) == {
        "Key": "id", "Amount": "numeric", "Kind": "categorical", "Text": "text",
        "Created": "ignored", "Blob": "ignored"}


def test_one_row_has_no_id_column():
    assert suggest_roles([_profile_column("a", "number", 1)], 1) == {"a": "ignored"}


def test_a_suggested_class_column_replaces_the_former_one():
    suggested = {"nr": "id", "kind": "categorical", "species": "categorical"}
    assert with_class_column({"nr": "id", "kind": "class"}, "species", suggested) == {
        "nr": "id", "kind": "categorical", "species": "class"}
    # The ID column chosen as class: the suggested ID column takes over.
    suggested = {"nr": "class", "seq": "id", "kind": "categorical"}
    assert with_class_column({"nr": "id", "kind": "class"}, "nr", suggested) == {
        "nr": "class", "kind": "categorical", "seq": "id"}


# --- The Dataset tab -------------------------------------------------------------------------

def _csv():
    lines = ["ref;nr;size;colour;note;empty;same;species"]
    for i in range(1, 41):
        ref = "" if i == 1 else f"R{i}"
        lines.append(f"{ref};{i};{i % 7 + 0.5};{('red', 'blue', 'green')[i % 3]};note {i};;x;"
                     f"{('a', 'b')[i % 2]}")
    return ("\n".join(lines) + "\n").encode()


SUGGESTED = {"ref": "text", "nr": "id", "size": "numeric", "colour": "categorical",
             "note": "text", "empty": "ignored", "same": "ignored", "species": "categorical"}


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
    return project, _upload(client, "Ny", _csv())


def _roles(page):
    return {name: re.search(r'<option value="(\w+)" selected', body).group(1)
            for name, body in re.findall(r'<select name="role:([^"]+)"[^>]*>(.*?)</select>',
                                         page, re.S)}


def _stored(client, project, dataset):
    return client.get(f"/api/projects/{project}/datasets/{dataset}/roles").json()["columns"]


def test_a_new_dataset_gets_suggested_roles_that_are_not_saved(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        stored = _stored(client, project, dataset)
    assert _roles(page) == SUGGESTED
    assert ("Inga kolumnroller ännu. Rollerna nedan är föreslagna utifrån datasetets innehåll. "
            "Granska rollerna och spara för att behålla dem.") in page
    assert "Klasskolumnen föreslås inte: välj den här, eller visa en kolumns fördelning" in page
    assert stored is None                                               # nothing saved


def test_the_column_shown_on_the_classes_view_becomes_the_class_column(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _project(client)
        classes = client.get(f"/ui/dataset?dataset={dataset}&view=classes&column=species").text
        link = re.search(r'href="([^"]*view=roles[^"]*)"', classes).group(1).replace("&amp;", "&")
        page = client.get(link).text
    assert "Tills kolumnrollerna har sparats föreslås kolumnen som visas här som klasskolumn" in classes
    assert link == f"/ui/dataset?dataset={dataset}&view=roles&column=species"
    assert _roles(page) == dict(SUGGESTED, species="class")
    assert "Klasskolumnen är den kolumn som senast visades i vyn Klasser: “species”." in page
    assert "Klasskolumnen föreslås inte" not in page


def test_saving_keeps_the_suggestion_and_then_the_saved_roles_stand(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles&column=species").text
        form = {f"role:{name}": role for name, role in _roles(page).items()}
        form["categorical_limit"] = "30"
        saved = client.post(f"/ui/dataset/{dataset}/roles", data=form, follow_redirects=False)
        stored = _stored(client, project, dataset)
        later = client.get(f"/ui/dataset?dataset={dataset}&view=roles&column=colour").text
        classes = client.get(f"/ui/dataset?dataset={dataset}&view=classes&column=colour").text
    assert "roles_saved" in saved.headers["location"]
    assert stored == {name: role for name, role in SUGGESTED.items() if role != "ignored"} | {
        "species": "class"}
    assert _roles(later)["species"] == "class" and _roles(later)["colour"] == "categorical"
    assert "föreslagna utifrån" not in later and "senast visades" not in later
    assert "Tills kolumnrollerna har sparats" not in classes


def test_an_unknown_column_is_not_the_class_column(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _project(client)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles&column=gone").text
    assert _roles(page) == SUGGESTED and "Klasskolumnen föreslås inte" in page


def test_a_refused_save_keeps_the_users_choices(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _project(client)
        form = {f"role:{name}": role for name, role in SUGGESTED.items()}   # no class column
        form["categorical_limit"] = "30"
        response = client.post(f"/ui/dataset/{dataset}/roles", data=form)
    assert response.status_code == 422
    assert "Exakt en kolumn måste vara klasskolumn." in response.text
    assert "föreslagna utifrån" not in response.text
    assert _roles(response.text) == SUGGESTED


def test_copied_roles_win_over_the_suggestion(tmp_path):
    with _client(tmp_path) as client:
        _, dataset = _project(client)
        other = _upload(client, "Annan", _csv())
        client.post(f"/ui/dataset/{other}/roles", data={
            "role:nr": "id", "role:size": "numeric", "role:colour": "class",
            "categorical_limit": "30"}, follow_redirects=False)
        page = client.get(f"/ui/dataset?dataset={dataset}&view=roles&roles_from={other}"
                          f"&column=species").text
    assert "Kopierade från datasetet “Annan”" in page and "föreslagna utifrån" not in page
    assert {k: v for k, v in _roles(page).items() if v != "ignored"} == {
        "nr": "id", "size": "numeric", "colour": "class"}


def test_prediction_datasets_get_no_suggestion(tmp_path):
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        prediction = _upload(client, "Pred", _csv(), role="prediction")
        roles = client.get(f"/ui/dataset?dataset={prediction}&view=roles&column=species").text
        classes = client.get(f"/ui/dataset?dataset={prediction}&view=classes&column=species").text
    assert "Prediktionsdataset använder träningsdatasetets kolumnroller." in roles
    assert "föreslagna utifrån" not in roles and "Tills kolumnrollerna har sparats" not in classes


def test_a_configuration_suggestion_takes_the_chosen_class_column(tmp_path):
    from JBGConfigurationService import _server_template
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    with _client(tmp_path) as client:
        project, dataset = _project(client)
        # A configuration stored before 293c, roles included (as in the 293b2 tests).
        settings = _server_template()
        settings["data_source"]["kind"] = "csv"
        settings["connection"].update({"id_column": "nr", "class_column": "colour",
                                       "data_numerical_columns": ["size"],
                                       "data_text_columns": ["species"]})
        owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
        client.app.state.jbg["configuration_service"]._repository.create(
            owner, project, "Gammal", settings)
        plain = client.get(f"/ui/dataset?dataset={dataset}&view=roles").text
        chosen = client.get(f"/ui/dataset?dataset={dataset}&view=roles&column=species").text
    assert "Förslag från konfigurationen “Gammal”." in plain
    assert {k: v for k, v in _roles(plain).items() if v != "ignored"} == {
        "nr": "id", "size": "numeric", "species": "text", "colour": "class"}
    assert "Förslag från konfigurationen “Gammal”." in chosen and "senast visades" in chosen
    assert {k: v for k, v in _roles(chosen).items() if v != "ignored"} == {
        "nr": "id", "size": "numeric", "species": "class", "colour": "categorical"}
