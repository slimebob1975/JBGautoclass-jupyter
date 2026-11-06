"""Detailed results of a project run, for its owner only (revision 295a).

Agreed 2026-11-06: every project run writes its outputs into its own folder,
``<results root>/<project ID>/<run ID>/`` (the engine's ``Config.output_path``),
so later runs never overwrite them. Class names and row values stay in these
files and are served only to the run's owner; the summary in the metadata
database remains free of them. The engine writes its detail tables with
``Helpers.save_matrix_as_csv`` (semicolons, decimal commas, an index column).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from JBGApplicationContracts import RunReference

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = ROOT / "output" / "runs"
# kind -> file name prefix in <run folder>/output/csvs
DETAILS = {
    "leaderboard": "crossval_",
    "mispredictions": "misplaced_",
    # The per-feature table; "feature_importance_details_" holds only the
    # calculation's metadata and is excluded below.
    "feature_importance": "feature_importance_",
}
PAGE_SIZE = 50
MAX_PAGE_SIZE = 500


class RunResultsUnavailable(LookupError):
    """Unknown run, another user's run, or no such detail for it."""


def _number(text: str):
    """The engine writes decimal commas; return a float where it is a number."""
    candidate = text.strip().replace(" ", "").replace(",", ".")
    try:
        return float(candidate)
    except ValueError:
        return text


class RunResultsService:
    def __init__(self, store, *, results_root: Path = DEFAULT_RESULTS_ROOT) -> None:
        self._store = store
        self._root = Path(results_root)

    def _folder(self, owner: str, run_id: str) -> Path:
        try:
            record = self._store.get_owned(owner, RunReference(run_id))
        except (KeyError, ValueError):
            raise RunResultsUnavailable("run unavailable") from None
        project_run = record.spec.get("project_run") if isinstance(record.spec, dict) else None
        if not isinstance(project_run, dict):
            raise RunResultsUnavailable("run unavailable")
        return self._root / str(project_run["project_id"]) / record.run.key

    def class_labels(self, owner: str, run_id: str) -> list:
        path = self._folder(owner, run_id) / "class_labels.json"
        if not path.is_file():
            return []
        try:
            return [str(label) for label in json.loads(path.read_text(encoding="utf-8"))["class_labels"]]
        except (ValueError, KeyError, TypeError):
            return []

    def has_folder(self, owner: str, run_id: str) -> bool:
        """False for runs from before 295a, which kept no details of their own."""
        return self._folder(owner, run_id).is_dir()

    def available(self, owner: str, run_id: str) -> dict:
        folder = self._folder(owner, run_id) / "output" / "csvs"
        return {kind: self._file(folder, prefix) is not None for kind, prefix in DETAILS.items()}

    @staticmethod
    def _file(folder: Path, prefix: str):
        if not folder.is_dir():
            return None
        matches = [p for p in folder.glob(f"{prefix}*.csv")
                   if not (prefix == "feature_importance_" and "details" in p.name)]
        # The engine writes companions with longer names next to the main table
        # (e.g. crossval_..._grid_search.csv); the main table has the shortest.
        return min(matches, key=lambda p: (len(p.name), p.name)) if matches else None

    def detail(self, owner: str, run_id: str, kind: str, *, page: int = 1,
               page_size: int = PAGE_SIZE) -> dict:
        """One page of a detail table: columns, rows (numbers as numbers), totals."""
        if kind not in DETAILS:
            raise RunResultsUnavailable("detail unavailable")
        page = max(int(page), 1)
        page_size = min(max(int(page_size), 1), MAX_PAGE_SIZE)
        path = self._file(self._folder(owner, run_id) / "output" / "csvs", DETAILS[kind])
        if path is None:
            raise RunResultsUnavailable("detail unavailable")
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter=";", quotechar='"')
            header = next(reader, [])
            rows = [row for row in reader if row]
        # The engine's index column has an empty header; it is not data.
        drop_index = bool(header) and header[0].strip() == ""
        columns = header[1:] if drop_index else header
        start = (page - 1) * page_size
        selected = [row[1:] if drop_index else row for row in rows[start:start + page_size]]
        return {"run_id": run_id, "kind": kind, "columns": columns,
                "rows": [[_number(value) for value in row] for row in selected],
                "total_rows": len(rows), "page": page, "page_size": page_size}
