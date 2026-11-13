"""299b2 (found 2026-11-13): a configuration saved before 298h lacks
mode.date_columns, and Config refused to load it ("JSON schema v2 section
'mode' is missing keys: date_columns"), so a CSV training run failed with a
server error. Keys added after the schema-v2 baseline must be listed in
Config.LATER_KEYS with their default; this test guards every mode field
added since 298h."""
import json
import re
from datetime import timedelta

import pytest

from Config import Config


def _without(*keys):
    settings = Config()._config_to_json(username="")
    for key in keys:
        settings["mode"].pop(key)
    return settings


def _load(settings, tmp_path):
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return Config.load_config_from_json(path)


def test_a_configuration_from_before_298h_loads(tmp_path):
    config = _load(_without("date_columns"), tmp_path)
    assert config.mode.date_columns == ""


def test_every_later_key_has_a_default(tmp_path):
    later = list(Config.LATER_KEYS["mode"])
    assert {"categorical_limit", "date_columns"} <= set(later)
    config = _load(_without(*later), tmp_path)
    assert config.mode.categorical_limit == 30 and config.mode.date_columns == ""


def test_other_missing_keys_are_still_refused(tmp_path):
    from JBGExceptions import ConfigException
    with pytest.raises(ConfigException, match="missing keys: train"):
        _load(_without("train"), tmp_path)


def test_a_stored_configuration_from_before_298h_starts_a_csv_run(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("python_multipart")
    from fastapi.testclient import TestClient
    from JBGFastAPIServer import create_server_app
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    from JBGMetadataStore import MetadataStoreSettings
    client = TestClient(create_server_app(
        settings=MetadataStoreSettings(sqlite_path=tmp_path / "jobs.sqlite"),
        run_logs=tmp_path / "run-logs", csv_root=tmp_path / "csv", model_root=tmp_path / "models",
        results_root=tmp_path / "runs",
        dispatcher_options={"poll_seconds": 60, "heartbeat_seconds": 1,
                            "stale_after": timedelta(minutes=5)}))
    rows = "".join(f"{i};{i % 7}.5;{'ab'[i % 2]}\n" for i in range(40))
    with client:
        location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        client.post(f"/ui/projects/{project}/open", follow_redirects=False)
        dataset = client.post(f"/api/projects/{project}/datasets/csv", data={"name": "D", "role": "training"},
                              files={"file": ("d.csv", ("nr;x;art\n" + rows).encode(), "text/csv")}).json()["dataset_id"]
        client.post(f"/api/projects/{project}/datasets/{dataset}/activate")
        client.put(f"/api/projects/{project}/datasets/{dataset}/roles",
                   json={"columns": {"nr": "id", "x": "numeric", "art": "class"}})
        configuration = client.post(f"/api/projects/{project}/configurations",
                                    json={"name": "C", "settings": {"mode": {"predict": False}}}).json()
        # Make the stored settings look as they did before 298h.
        owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
        service = client.app.state.jbg["configuration_service"]
        repository = service._repository
        stored = service.stored_settings(owner, project, configuration["configuration_id"])
        stored["mode"].pop("date_columns")
        from sqlalchemy import update
        revisions = repository._revisions                          # every saved version
        with repository._engine.begin() as connection:
            connection.execute(update(revisions).where(
                revisions.c.configuration_id == configuration["configuration_id"]).values(
                settings_json=json.dumps(stored)))
        assert "date_columns" not in service.stored_settings(
            owner, project, configuration["configuration_id"])["mode"]
        started = client.post(f"/api/projects/{project}/runs",
                              json={"configuration_id": configuration["configuration_id"]})
    assert started.status_code == 201, started.text
