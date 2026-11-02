"""282b: offline bootstrap with no SQL, Entra, CSV or notebook dependencies."""
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
from JBGFastAPILocal import app, create_local_app


def test_default_offline_server_starts_with_read_endpoints():
    with TestClient(app) as client:
        assert client.get('/api/health').json() == {'status': 'ok'}
        assert client.get('/openapi.json').status_code == 200
        assert client.get('/docs').status_code == 200
        for path in ('', '/progress', '/result'):
            assert client.get('/api/runs/missing' + path).status_code == 404


def test_separate_local_app_bootstraps_are_isolated():
    first, second = create_local_app(), create_local_app(subject='bob')
    assert first is not second
    with TestClient(first) as a, TestClient(second) as b:
        assert a.get('/api/health').status_code == 200
        assert b.get('/api/health').status_code == 200


@pytest.mark.parametrize('host', ['0.0.0.0', '192.168.1.4', 'localhost', '::'])
def test_local_server_rejects_non_numeric_loopback(host):
    with pytest.raises(ValueError):
        create_local_app(host=host)


def test_windows_launcher_has_loopback_binding_and_no_implicit_install_or_pull():
    root = Path(__file__).resolve().parents[1]
    script = (root / 'scripts' / 'start_fastapi_local.ps1').read_text(encoding='utf-8')
    assert ('--host 127.0.0.1' in script or "--host '127.0.0.1'" in script)
    assert '$InstallDependencies' in script
    assert 'git pull' not in script.lower()
    assert 'requirements-fastapi.txt' in script
