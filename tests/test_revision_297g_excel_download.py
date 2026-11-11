"""297g: the predictions download opens in Excel (found 2026-11-11: Excel took
a file starting with "ID" for a SYLK file and refused it)."""
import re

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

from JBGApplicationContracts import DataSourceKind, RunStatus  # noqa: E402
from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode  # noqa: E402
from test_revision_296b_csv_predictions import _client  # noqa: E402

BOM = b"\xef\xbb\xbf"


def _download(tmp_path, content: bytes, subject="developer"):
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    with _client(tmp_path) as client:
        location = client.post("/ui/projects", data={"name": "P", "source": "csv"},
                               follow_redirects=False).headers["location"]
        project = re.search(r"selected=([0-9a-f-]{36})", location).group(1)
        store = client.app.state.jbg["store"]
        run = store.create(owner, DataSourceKind.CSV, {"spec_version": 1, "project_run": {
            "project_id": project, "kind": "prediction"}})
        store.claim(run, "w")
        store.finish(run, "w", RunStatus.SUCCEEDED, result={"schema_version": 1})
        folder = tmp_path / "runs" / project / run.key / "output" / "csvs"
        folder.mkdir(parents=True)
        stored = folder / "predictions_p__predict.csv"
        stored.write_bytes(content)
        response = client.get(f"/api/runs/{run.key}/predictions.csv")
    return response, stored, run


def test_a_file_starting_with_id_gets_a_byte_order_mark(tmp_path):
    content = "IDnummer;Predicted;Rate;P(ä);P(ö)\n1;ä;0,9;0,9;0,1\n".encode("utf-8")
    response, stored, run = _download(tmp_path, content)
    assert response.status_code == 200
    assert response.content == BOM + content            # Excel: UTF-8 text, not SYLK
    assert stored.read_bytes() == content                # the stored file is unchanged
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="jbg_predictions_{run.key[:8]}.csv"')


def test_a_file_that_has_one_already_gets_no_second(tmp_path):
    content = BOM + b"id;Predicted\n1;a\n"
    response, _, _ = _download(tmp_path, content)
    assert response.content == content


def test_large_files_are_sent_whole(tmp_path):
    content = b"id;Predicted\n" + b"".join(b"%d;klass\n" % i for i in range(30000))
    assert len(content) > 3 * 65536
    response, _, _ = _download(tmp_path, content)
    assert response.content == BOM + content


def test_an_empty_file_is_just_the_mark(tmp_path):
    response, _, _ = _download(tmp_path, b"")
    assert response.content == BOM
