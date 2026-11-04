"""282b4: Swagger resources must be served locally, never from a CDN."""
from html.parser import HTMLParser
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from JBGFastAPILocal import create_local_app


class _Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == 'script' and data.get('src'):
            self.urls.append(data['src'])
        if tag == 'link' and data.get('href'):
            self.urls.append(data['href'])


def test_swagger_resources_are_self_hosted():
    with TestClient(create_local_app()) as client:
        docs = client.get('/docs')
        assert docs.status_code == 200
        assert 'Swagger UI' in docs.text
        assert 'cdn.jsdelivr.net' not in docs.text
        assert 'fastapi.tiangolo.com' not in docs.text
        assets = _Assets()
        assets.feed(docs.text)
        assert any('.js' in url for url in assets.urls)
        assert any('.css' in url for url in assets.urls)
        for url in assets.urls:
            parsed = urlparse(url)
            assert not parsed.scheme and not parsed.netloc, url
            assert parsed.path.startswith('/'), url
            response = client.get(parsed.path)
            assert response.status_code == 200, url
            assert response.content, url


def test_root_redirects_to_docs_and_api_unchanged():
    with TestClient(create_local_app()) as client:
        response = client.get('/', follow_redirects=False)
        assert response.status_code == 307
        assert response.headers['location'] == '/docs'
        assert client.get('/api/health').json() == {'status': 'ok'}
        assert client.get('/openapi.json').status_code == 200
        assert client.get('/api/runs/unknown').status_code == 404


def test_launcher_reinstalls_new_offline_dependency():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = (root / 'scripts' / 'start_fastapi_local.ps1').read_text(encoding='utf-8')
    requirements = (root / 'requirements-fastapi.txt').read_text(encoding='utf-8')
    # 285d: no dedicated venv any more; the launcher installs the API packages
    # into the JBG environment whenever one of them cannot be imported.
    assert 'import fastapi, uvicorn, fastapi_offline, sqlalchemy' in script
    assert 'fastapi-offline==1.7.7' in requirements
