"""Reclassification of mispredicted rows (revision 299a, SQL).

Agreed 2026-11-12, replacing the Voila GUI's write-per-click grid: on the
Results tab the owner of a finished SQL training run picks the right class for
mispredicted rows, reviews the list and confirms. Only then are the classes
written to the run's source table, under the user's own SQL sign-in, in one
transaction: every confirmed correction is written, or none.

The statement is parameterised (UPDATE <table> SET [class] = ? WHERE [id] = ?).
The Voila GUI's statement put the class into the SQL text without escaping it.
A new class must be one of the run's classes; an ID must be one of the run's
mispredicted rows.

Every written correction is logged in the run's own folder
(``corrections.json``: ID, old class, new class, time), which only the run's
owner can see, like all class names (decision 5). A logged correction can be
undone, which writes the old class back. The next training run reads the
corrected classes from the table; the reviewed run's tables stay as they were.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from JBGApplicationContracts import DataSourceKind, RunReference

LOG_NAME = "corrections.json"


class CorrectionRefused(ValueError):
    """Not a correctable run, an unknown row or class, or nothing to change."""


class CorrectionFailed(RuntimeError):
    """The source table could not be updated (nothing was written)."""


def _bracket(name: str) -> str:
    return "[" + str(name).replace("]", "]]") + "]"


def _row_id(value) -> int:
    """SQL IDs are whole numbers (298i); the table file may show 36 as 36.0 or 0036."""
    number = float(str(value).strip().replace(",", "."))
    if not number.is_integer():
        raise CorrectionRefused("unknown row")
    return int(number)


class SQLReclassificationService:
    def __init__(self, store, results, sessions, sql_server) -> None:
        self._store = store
        self._results = results
        self._sessions = sessions
        self._sql_server = sql_server

    # --- Reading -----------------------------------------------------------------------

    def _record(self, owner: str, run_id: str):
        from JBGApplicationContracts import RunStatus
        try:
            record = self._store.get_owned(owner, RunReference(run_id))
        except (KeyError, ValueError):
            raise CorrectionRefused("run unavailable") from None
        project_run = record.spec.get("project_run") if isinstance(record.spec, dict) else None
        if (record.source is not DataSourceKind.SQL or not isinstance(project_run, dict)
                or project_run.get("kind", "training") != "training"
                or record.status is not RunStatus.SUCCEEDED):
            raise CorrectionRefused("only finished SQL training runs can be corrected")
        return record

    def correctable(self, owner: str, run_id: str) -> bool:
        try:
            self._record(owner, run_id)
        except CorrectionRefused:
            return False
        return True

    def mispredicted(self, owner: str, run_id: str) -> dict:
        """The run's mispredicted rows: ID -> the class the run saw."""
        from JBGRunResults import DETAILS
        self._record(owner, run_id)
        folder = self._results._folder(owner, run_id)
        path = self._results._file(folder / "output" / "csvs", DETAILS["mispredictions"])
        if path is None:
            return {}
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter=";", quotechar='"')
            header = next(reader, [])
            if "Actual" not in header:
                return {}
            actual = header.index("Actual")
            rows = {}
            for row in reader:
                if row and len(row) > actual:
                    rows.setdefault(_row_id(row[0]), row[actual])
        return rows

    def log(self, owner: str, run_id: str) -> list:
        self._record(owner, run_id)
        path = self._log_path(owner, run_id)
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def current(self, owner: str, run_id: str) -> dict:
        """ID -> the class it was last corrected to (undone corrections left out)."""
        result = {}
        for entry in self.log(owner, run_id):
            if entry.get("undone_at"):
                result.pop(entry["id"], None)
            else:
                result[entry["id"]] = entry["new"]
        return result

    # --- Reviewing and writing ----------------------------------------------------------

    def review(self, owner: str, run_id: str, chosen: dict) -> list:
        """The corrections a form asks for: [{"id", "old", "new"}], unchanged ones left out."""
        self._record(owner, run_id)                     # the owner's own SQL training run first
        labels = [str(label) for label in self._results.class_labels(owner, run_id)]
        rows = self.mispredicted(owner, run_id)
        changes = []
        for raw_id, new_class in chosen.items():
            if not str(new_class):
                continue
            row_id = _row_id(raw_id)
            if row_id not in rows:
                raise CorrectionRefused("unknown row")
            if str(new_class) not in labels:
                raise CorrectionRefused("unknown class")
            if str(new_class) != str(rows[row_id]):
                changes.append({"id": row_id, "old": str(rows[row_id]), "new": str(new_class)})
        if not changes:
            raise CorrectionRefused("nothing to change")
        return sorted(changes, key=lambda change: change["id"])

    def apply(self, principal, run_id: str, chosen: dict) -> int:
        """Write the reviewed corrections in one transaction and log them."""
        owner = principal.principal_key
        changes = self.review(owner, run_id, chosen)
        record = self._record(owner, run_id)
        self._write(principal, record, [(change["id"], change["new"]) for change in changes])
        moment = datetime.now(timezone.utc).isoformat()
        entries = self.log(owner, run_id)
        entries.extend(dict(change, at=moment, undone_at=None) for change in changes)
        self._save_log(owner, run_id, entries)
        return len(changes)

    def undo(self, principal, run_id: str, index: int) -> dict:
        """Write a logged correction's old class back and mark it undone."""
        owner = principal.principal_key
        record = self._record(owner, run_id)
        entries = self.log(owner, run_id)
        if not 0 <= index < len(entries) or entries[index].get("undone_at"):
            raise CorrectionRefused("unknown correction")
        entry = entries[index]
        self._write(principal, record, [(entry["id"], entry["old"])])
        entry["undone_at"] = datetime.now(timezone.utc).isoformat()
        self._save_log(owner, run_id, entries)
        return entry

    def _write(self, principal, record, pairs) -> None:
        from JBGProjectRuns import _QuietLogger, _load_config
        config = _load_config(record.spec["configuration"])
        self._sql_server.apply_to(config)
        # PermissionError("SQL sign-in required") without a sign-in.
        layer = self._sessions.data_layer_for(principal, config, _QuietLogger())
        statement = (f"UPDATE {config.get_connection().get_formatted_data_table()} "
                     f"SET {_bracket(config.get_class_column_name())} = ? "
                     f"WHERE {_bracket(config.get_id_column_name())} = ?")
        helper = layer.get_connection()
        try:
            for row_id, new_class in pairs:
                helper.execute_query_with_params(statement, (new_class, int(row_id)), commit=False)
            helper.disconnect(commit=True)
        except Exception:
            try:
                helper.disconnect(commit=False)
            except Exception:
                pass
            # The driver's message may name the table or values; it stays server-side.
            raise CorrectionFailed("the source table could not be updated") from None

    def _log_path(self, owner: str, run_id: str) -> Path:
        return self._results._folder(owner, run_id) / LOG_NAME

    def _save_log(self, owner: str, run_id: str, entries: list) -> None:
        path = self._log_path(owner, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(path)
