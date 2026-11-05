"""Validate and preview project datasets, the same way for SQL and CSV (289a).

Agreed 2026-11-05: one check list with the same codes and fixed texts for both
sources, so the Dataset tab can show them identically; optionally with the
column roles of a chosen configuration. The preview shows columns, row count
and the first 50 rows. Values go to the owner's browser only: they are never
stored, logged or cached here. SQL access uses the user's own sign-in for the
server (288b); every column name in a query is first checked against the
table's real columns and quoted.
"""
from __future__ import annotations

import csv
import datetime
import decimal
import shutil
import tempfile
from pathlib import Path
from typing import Any

from JBGApplicationContracts import DataSourceKind
from JBGDatasets import DatasetRepository, DatasetRole, DatasetValidationState

PREVIEW_ROWS = 50
CLASS_LIMIT = 30            # the CSV structure check's default class cardinality limit
NUMERIC_SQL_TYPES = frozenset({"bigint", "int", "smallint", "tinyint", "bit", "decimal",
                               "numeric", "money", "smallmoney", "float", "real"})

CHECKS = (
    ("source_reachable", "The data source can be reached.", "The data source could not be reached."),
    ("source_readable", "The data could be read.", "The data could not be read."),
    ("has_rows", "The dataset contains rows.", "The dataset contains no rows."),
    ("id_column", "The ID column exists and is unique.",
     "The ID column is missing, has empty values or repeats values."),
    ("class_column", "The class column exists and has between 2 and 30 classes.",
     "The class column is missing, is the ID column or does not have between 2 and 30 classes."),
    ("feature_columns", "The feature columns exist and do not overlap with ID or class.",
     "A feature column is missing, repeated, or is the ID or class column."),
    ("feature_types", "Numeric feature columns contain numbers only.",
     "A numeric feature column contains values that are not numbers."),
)
ROLE_CHECKS = ("id_column", "class_column", "feature_columns", "feature_types")


class DatasetUnreadable(ValueError):
    """The preview could not be read (fixed message)."""


def _quote(name: str) -> str:
    return "[" + str(name).replace("]", "]]") + "]"


def _table(catalog: str, table: str) -> str:
    return ".".join(_quote(part) for part in [catalog, *str(table).split(".")])


def _json_value(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None                      # binary data is not previewed
    return str(value)


class _CSVFacts:
    def __init__(self, schema, path: Path, csv_format: dict):
        self.columns = list(schema.columns)
        self.rows = schema.rows
        self._schema = schema
        self._path, self._format = path, csv_format

    def id_unique(self, column):
        return column in self._schema.unique_id_candidates

    def class_ok(self, column):
        return column in self._schema.class_candidates

    def numeric_ok(self, column):
        return column not in self._schema.text_feature_candidates

    def column_types(self):
        return [{"name": c, "type": "text" if c in self._schema.text_feature_candidates
                 else "number"} for c in self.columns]

    def first_rows(self):
        with self._path.open("r", encoding=self._format["encoding"], newline="") as stream:
            reader = csv.reader(stream, delimiter=self._format["delimiter"])
            next(reader, None)
            return [list(row) for _, row in zip(range(PREVIEW_ROWS), reader)]


class _SQLFacts:
    def __init__(self, layer, catalog: str, table: str):
        self._layer, self._catalog, self._table = layer, catalog, table
        self._types = layer.get_table_columns(catalog, table)
        self.columns = list(self._types)
        self.rows = layer.count_data_rows(catalog, table) if self.columns else 0

    def id_unique(self, column):
        stats = self._layer.get_column_uniqueness_stats(self._catalog, self._table, column)
        return stats["total"] > 0 and stats["nonnull"] == stats["total"] == stats["distinct"]

    def class_ok(self, column):
        count = self._layer.get_column_cardinalities(self._catalog, self._table, [column])[column]
        return 2 <= count <= CLASS_LIMIT

    def numeric_ok(self, column):
        return str(self._types.get(column, "")).lower() in NUMERIC_SQL_TYPES

    def column_types(self):
        return [{"name": c, "type": str(t)} for c, t in self._types.items()]

    def first_rows(self):
        columns = ", ".join(_quote(c) for c in self.columns)       # real columns only
        query = f"SELECT TOP({PREVIEW_ROWS}) {columns} FROM {_table(self._catalog, self._table)}"
        return [list(row) for row in self._layer.get_data_list_from_query(query)]


class _QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class DatasetInspectionService:
    def __init__(self, datasets: DatasetRepository, projects, configurations, *,
                 sessions=None, sql_server=None, csv_root=None) -> None:
        from JBGProjectStorage import DEFAULT_CSV_ROOT
        self._datasets = datasets
        self._projects = projects
        self._configurations = configurations
        self._sessions = sessions
        self._sql_server = sql_server
        self._csv_root = Path(csv_root) if csv_root is not None else DEFAULT_CSV_ROOT

    def _dataset(self, owner, project_id, dataset_id):
        project = self._projects.require(owner, project_id)
        try:
            return project, self._datasets.get_owned(owner, project_id, dataset_id)
        except KeyError:
            from JBGDatasetService import DatasetUnavailable
            raise DatasetUnavailable("dataset unavailable") from None

    # --- Reading the source ---------------------------------------------------------

    def _with_facts(self, principal, project, dataset, action, reached=lambda: None):
        """Run ``action(facts)`` on the dataset's source; facts never outlive it.

        ``reached`` is called as soon as the source itself was reached (file
        found / connection opened), before anything is read from it.
        """
        if dataset.source is DataSourceKind.CSV:
            from CSVColumnSchema import inspect_csv_columns
            from JBGProjectStorage import csv_upload_format, project_csv_storage
            repository, scope = project_csv_storage(project.project_id, root=self._csv_root)
            csv_format = csv_upload_format(repository, scope, dataset.registration_id)
            with tempfile.TemporaryDirectory(prefix="jbg_inspect_") as folder:
                path = Path(folder) / "data.csv"
                with repository.open_artifact(scope, dataset.registration_id) as source, \
                        path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                reached()
                schema = inspect_csv_columns(path, delimiter=csv_format["delimiter"],
                                             encoding=csv_format["encoding"])
                return action(_CSVFacts(schema, path, csv_format))
        if self._sql_server is None or not self._sql_server.configured or self._sessions is None:
            from JBGProjectRuns import RunRequestRefused
            raise RunRequestRefused("no SQL Server is configured on this server")
        config = self._sql_server.probe_config()
        config.connection.data_catalog = dataset.catalog
        config.connection.data_table = dataset.table
        # PermissionError("SQL sign-in required") without the user's sign-in.
        layer = self._sessions.data_layer_for(principal, config, _QuietLogger())
        reached()
        return action(_SQLFacts(layer, dataset.catalog, dataset.table))

    # --- Validation -----------------------------------------------------------------

    def validate(self, principal, project_id: str, dataset_id: str, *,
                 configuration_id: str | None = None, version: int | None = None) -> dict:
        owner = principal.principal_key
        project, dataset = self._dataset(owner, project_id, dataset_id)
        roles = None
        if configuration_id is not None:
            settings = self._configurations.stored_configuration(
                owner, project_id, configuration_id, version).settings["connection"]
            text = list(settings.get("data_text_columns") or [])
            numeric = [c for c in settings.get("data_numerical_columns") or [] if c not in text]
            roles = {"id": settings.get("id_column", ""), "class": settings.get("class_column", ""),
                     "numeric": numeric, "text": text}

        outcome = {code: None for code, _, _ in CHECKS}

        def run_checks(facts):
            outcome["source_reachable"] = True              # the source answered
            outcome["source_readable"] = bool(facts.columns)
            if not facts.columns:
                return
            outcome["has_rows"] = facts.rows > 0
            if roles is None or not outcome["has_rows"]:
                return
            columns = set(facts.columns)
            outcome["id_column"] = roles["id"] in columns and facts.id_unique(roles["id"])
            known_class = roles["class"] in columns and roles["class"] != roles["id"]
            outcome["class_column"] = known_class and (
                dataset.role is DatasetRole.PREDICTION or facts.class_ok(roles["class"]))
            features = roles["numeric"] + roles["text"]
            outcome["feature_columns"] = bool(features) and len(set(features)) == len(features) \
                and all(c in columns and c not in (roles["id"], roles["class"]) for c in features)
            if outcome["feature_columns"]:
                outcome["feature_types"] = all(facts.numeric_ok(c) for c in roles["numeric"])

        state = {"reached": False}
        try:
            self._with_facts(principal, project, dataset, run_checks,
                             reached=lambda: state.update(reached=True))
        except (PermissionError, LookupError):
            raise
        except Exception as error:
            from JBGProjectRuns import RunRequestRefused
            if isinstance(error, RunRequestRefused):
                raise
            # The first check not yet answered failed; details stay server-side.
            outcome["source_reachable"] = state["reached"]
            if state["reached"]:
                for code, _, _ in CHECKS[1:]:
                    if outcome[code] is None:
                        outcome[code] = False
                        break

        checks, valid = [], True
        for code, passed_text, failed_text in CHECKS:
            if code in ROLE_CHECKS and roles is None:
                continue
            result = outcome[code]
            if result is None:
                checks.append({"code": code, "status": "skipped", "message": None})
                valid = False
            else:
                checks.append({"code": code, "status": "passed" if result else "failed",
                               "message": passed_text if result else failed_text})
                valid = valid and result
        state = DatasetValidationState.VALID if valid else DatasetValidationState.INVALID
        self._datasets.set_validation(owner, project_id, dataset_id, state)
        return {"dataset_id": dataset_id, "validation_state": state.value,
                "configuration_id": configuration_id, "checks": checks}

    # --- Preview ----------------------------------------------------------------------

    def preview(self, principal, project_id: str, dataset_id: str) -> dict[str, Any]:
        owner = principal.principal_key
        project, dataset = self._dataset(owner, project_id, dataset_id)

        def read(facts):
            if not facts.columns:
                raise DatasetUnreadable("the dataset could not be read")
            rows = [[_json_value(value) for value in row] for row in facts.first_rows()]
            return {"dataset_id": dataset_id, "columns": facts.column_types(),
                    "row_count": facts.rows, "rows": rows}
        try:
            return self._with_facts(principal, project, dataset, read)
        except (PermissionError, LookupError, DatasetUnreadable):
            raise
        except Exception as error:
            from JBGProjectRuns import RunRequestRefused
            if isinstance(error, RunRequestRefused):
                raise
            raise DatasetUnreadable("the dataset could not be read") from None
