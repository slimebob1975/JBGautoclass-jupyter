"""Predictions joined with the predicted rows' original columns (revision 299c).

The Voila GUI's "Export joined" for CSV predictions: every original column of
each predicted row, followed by the prediction's columns. The rows come from
the very file version the prediction run read (the run's own CSV artifact),
so a later correction (299b) does not change an earlier export. Row IDs are
matched as written ("0036" is not 36). For the run's owner only, like the
predictions themselves.

The file is written like the plain predictions download (297g): semicolons,
a UTF-8 byte-order mark (Excel reads it as UTF-8 and never as SYLK) and the
values as stored.
"""
from __future__ import annotations

import csv
import io

from JBGApplicationContracts import DataSourceKind, RunReference, RunStatus


class JoinedExportUnavailable(LookupError):
    """Not the owner's finished CSV prediction run, or its files are gone."""


def joined_predictions(store, results, csv_root, owner: str, run_id: str) -> tuple[bytes, str]:
    """The joined file's bytes and a file name."""
    from JBGProjectStorage import csv_upload_format, project_csv_storage
    from JBGRunResults import RunResultsUnavailable
    try:
        record = store.get_owned(owner, RunReference(run_id))
    except (KeyError, ValueError):
        raise JoinedExportUnavailable("run unavailable") from None
    spec = record.spec if isinstance(record.spec, dict) else {}
    project_run = spec.get("project_run") or {}
    artifact_id = (spec.get("csv") or {}).get("artifact_id")
    if (record.source is not DataSourceKind.CSV or project_run.get("kind") != "prediction"
            or record.status is not RunStatus.SUCCEEDED or not artifact_id):
        raise JoinedExportUnavailable("not a finished CSV prediction run")
    id_column = spec["configuration"]["connection"]["id_column"]
    try:
        path = results.predictions_file(owner, run_id)
    except RunResultsUnavailable:
        raise JoinedExportUnavailable("predictions unavailable") from None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        predicted = list(csv.reader(stream, delimiter=";", quotechar='"'))
    if not predicted:
        raise JoinedExportUnavailable("predictions unavailable")
    repository, scope = project_csv_storage(project_run["project_id"], root=csv_root)
    try:
        form = csv_upload_format(repository, scope, artifact_id)
        with repository.open_artifact(scope, artifact_id) as stream:
            text = stream.read().decode(form["encoding"])
    except (OSError, ValueError, UnicodeError):
        raise JoinedExportUnavailable("the run's file is unavailable") from None
    source = list(csv.reader(io.StringIO(text, newline=""), delimiter=form["delimiter"],
                             quotechar=form["quotechar"]))
    columns = source[0] if source else []
    if id_column not in columns:
        raise JoinedExportUnavailable("the run's file is unavailable")
    at = columns.index(id_column)
    original = {}
    for row in source[1:]:
        if len(row) > at:
            original.setdefault(row[at].strip(), row)
    # The prediction's own columns, after the original ones (its ID is the same).
    added = predicted[0][1:]
    added = [name if name not in columns else f"{name} (prediction)" for name in added]
    out = io.StringIO(newline="")
    writer = csv.writer(out, delimiter=";", quotechar='"', lineterminator="\r\n")
    writer.writerow(columns + added)
    blank = [""] * len(columns)
    for row in predicted[1:]:
        if not row:
            continue
        values = original.get(row[0].strip(), blank)
        values = (list(values) + blank)[:len(columns)]
        writer.writerow(values + row[1:])
    return out.getvalue().encode("utf-8-sig"), f"jbg_predictions_joined_{run_id[:8]}.csv"
