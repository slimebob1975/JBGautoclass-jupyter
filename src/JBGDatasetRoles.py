"""Column roles belong to the training dataset (revision 293a, decision A).

One role per column: ``id``, ``class``, ``numeric``, ``text``, ``categorical``
or ``ignored``; columns without a role are ignored. Training and mixed
datasets carry roles; prediction datasets use the roles of the project's
active training dataset and must be compatible with it. Runs take the roles
from their dataset; configurations hold only the machine-learning settings.

Mapping to the engine's Config: ``categorical`` columns are text columns that
are categorised, so they appear in ``data_text_columns`` and in
``mode.category_text_columns``.

298g: at most one ``group`` column (for example a person's identity) keeps
its rows together when data is split, without being a predictor. The engine
needs it among its data columns to read it, so it is passed as a text column
(numbers stay exact as text), named as the excluded predictor (dropped right
after the split, before any conversion) and as ``mode.validation_group_column``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

ROLES = ("id", "class", "numeric", "text", "categorical", "date", "group", "ignored")
FEATURE_ROLES = ("numeric", "text", "categorical", "date")
# 298i: the engine's SQL path orders, splits and writes back rows by whole-number
# IDs (Helpers.get_rid_of_decimals), so an SQL ID column must have one of these
# types (a uniqueidentifier ID made a run fail after fetching every row). CSV IDs
# are opaque text and need no such type.
INTEGER_SQL_TYPES = frozenset({"bigint", "int", "smallint", "tinyint"})
MAX_COLUMNS = 2000


class IncompatibleDataset(ValueError):
    """A prediction dataset lacks training columns or has other column kinds."""

    def __init__(self, missing=(), different=()):
        super().__init__("the dataset is not compatible with the training dataset")
        self.missing, self.different = tuple(missing), tuple(different)


DEFAULT_CATEGORICAL_LIMIT = 30


@dataclass(frozen=True)
class DatasetRoles:
    columns: tuple            # ((column name, role), ...) in the dataset's order
    # 293a2: text columns with at most this many distinct values are
    # categorised; "categorical" columns are categorised regardless.
    categorical_limit: int = DEFAULT_CATEGORICAL_LIMIT

    @classmethod
    def from_mapping(cls, mapping, categorical_limit=DEFAULT_CATEGORICAL_LIMIT) -> "DatasetRoles":
        if type(categorical_limit) is not int or not 2 <= categorical_limit <= 1000:
            raise ValueError("the categorical limit must be a whole number between 2 and 1000")
        if not isinstance(mapping, dict) or len(mapping) > MAX_COLUMNS:
            raise ValueError("column roles must map column names to roles")
        pairs = []
        for name, role in mapping.items():
            if not isinstance(name, str) or not name or len(name) > 200:
                raise ValueError("column names must be text of 1-200 characters")
            if role not in ROLES:
                raise ValueError("unknown column role")
            pairs.append((name, role))
        return cls(tuple(pairs), categorical_limit)

    @classmethod
    def from_json(cls, text: str) -> "DatasetRoles":
        stored = json.loads(text)
        if "columns" in stored and isinstance(stored["columns"], dict):
            return cls.from_mapping(stored["columns"],
                                    stored.get("categorical_limit", DEFAULT_CATEGORICAL_LIMIT))
        return cls.from_mapping(dict(stored))            # 293a format: the mapping only

    def to_json(self) -> str:
        return json.dumps({"columns": dict(self.columns),
                           "categorical_limit": self.categorical_limit})

    def as_mapping(self) -> dict:
        return dict(self.columns)

    def by_role(self, role: str) -> list:
        return [name for name, given in self.columns if given == role]

    @property
    def id_column(self) -> str:
        return (self.by_role("id") or [""])[0]

    @property
    def class_column(self) -> str:
        return (self.by_role("class") or [""])[0]

    @property
    def group_column(self) -> str:
        return (self.by_role("group") or [""])[0]

    @property
    def features(self) -> list:
        return [name for name, role in self.columns if role in FEATURE_ROLES]

    def check(self, available, *, require_class: bool = True) -> None:
        """Fixed messages; one role per column is guaranteed by the structure."""
        unknown = [name for name, _ in self.columns if name not in set(available)]
        if unknown:
            raise ValueError("a column with a role does not exist in the dataset")
        if len(self.by_role("id")) != 1:
            raise ValueError("exactly one column must be the ID column")
        if len(self.by_role("class")) > 1 or (require_class and not self.by_role("class")):
            raise ValueError("exactly one column must be the class column")
        if not self.features:
            raise ValueError("at least one column must be a feature")
        if len(self.by_role("group")) > 1:
            raise ValueError("at most one column can be the group column")

    def apply_to(self, config) -> None:
        """Set the engine's role fields of a Config from these roles."""
        text, categorical = self.by_role("text"), self.by_role("categorical")
        config.connection.id_column = self.id_column
        config.connection.class_column = self.class_column
        config.connection.data_numerical_columns = self.by_role("numeric")
        group = [self.group_column] if self.group_column else []
        # 298h: date features are read as text and converted as dates.
        dates = self.by_role("date")
        config.connection.data_text_columns = text + categorical + dates + group
        config.mode.date_columns = ",".join(dates)
        config.mode.category_text_columns = categorical
        config.mode.categorical_limit = self.categorical_limit
        # 298g: the group column belongs to the dataset; it replaces whatever a
        # configuration held (configurations no longer set it).
        config.mode.validation_group_column = self.group_column
        config.mode.excluded_predictor_columns = self.group_column


def roles_from_configuration(settings: dict, columns=None) -> dict:
    """293b2: column roles as an older configuration expressed them, as a
    suggestion for a dataset without roles (decision A moved roles to the
    dataset). Only columns in ``columns`` (when given) are kept."""
    connection = settings.get("connection", {}) or {}
    categorical = set(settings.get("mode", {}).get("category_text_columns") or [])
    mapping = {}
    for name in connection.get("data_numerical_columns") or []:
        mapping[name] = "numeric"
    for name in connection.get("data_text_columns") or []:
        mapping[name] = "categorical" if name in categorical else "text"
    if connection.get("class_column"):
        mapping[connection["class_column"]] = "class"
    if connection.get("id_column"):
        mapping[connection["id_column"]] = "id"
    if columns is not None:
        mapping = {name: role for name, role in mapping.items() if name in set(columns)}
    return mapping


def _column_kind(type_name) -> str:
    """"number", "text" or "other" for a profile column type: CSV columns are
    "number" or "text"; SQL columns carry their type name (nvarchar, int, ...)."""
    from JBGDatasetInspection import _TEXT_SQL_TYPES, NUMERIC_SQL_TYPES
    name = str(type_name or "").lower()
    if name == "number" or name in NUMERIC_SQL_TYPES:
        return "number"
    if name == "text" or name in _TEXT_SQL_TYPES:
        return "text"
    return "other"                                  # dates, binary data, GUIDs, ...


def suggest_roles(columns, rows: int, categorical_limit: int = DEFAULT_CATEGORICAL_LIMIT,
                  class_column: str = "", source: str = "csv") -> dict:
    """298a: preliminary roles for a dataset without roles, from its profile
    (``columns``: name, type, missing and distinct per column; ``rows``: the
    rows the profile counted). A suggestion to review and save; nothing here
    is stored.

    The class column is the user's choice: only ``class_column`` (when it is
    a column of the dataset) gets that role. Of the other columns, the first
    filled-in column with a distinct value on every row (and at least two
    rows) is the ID column; empty and single-valued columns are ignored;
    numbers are numeric features; text with at most ``categorical_limit``
    distinct values is categorical, other text is free text. Other SQL types
    (dates, binary data, ...) are ignored unless they are the ID column.
    298i: in SQL datasets (``source="sql"``) only whole-number columns can be
    the suggested ID.
    """
    mapping, id_given = {}, False
    names = [c["name"] for c in columns]
    for column in columns:
        name, distinct = column["name"], column.get("distinct")
        if name == class_column:
            mapping[name] = "class"
        elif column.get("date"):
            mapping[name] = "ignored"        # 298h: dates are an informed choice
        elif (not id_given and rows >= 2 and distinct == rows
                and not column.get("missing")
                and (source != "sql" or str(column.get("type", "")).lower() in INTEGER_SQL_TYPES)):
            mapping[name], id_given = "id", True
        elif distinct is None or distinct <= 1:
            mapping[name] = "ignored"
        else:
            kind = _column_kind(column.get("type"))
            if kind == "number":
                mapping[name] = "numeric"
            elif kind == "text":
                mapping[name] = "categorical" if distinct <= categorical_limit else "text"
            else:
                mapping[name] = "ignored"
    return {name: mapping[name] for name in names}


def with_class_column(mapping: dict, column: str, suggested: dict) -> dict:
    """298a: ``mapping`` with ``column`` as its class column. A former class
    column gets its role from ``suggested`` (see ``suggest_roles``); so does
    the ID column when ``column`` was the ID column."""
    result = {name: (suggested.get(name, "ignored") if role == "class" else role)
              for name, role in mapping.items()}
    if mapping.get(column) == "id":
        result.update({name: "id" for name, role in suggested.items()
                       if role == "id" and name != column})
    result[column] = "class"
    return result


def check_compatible(roles: DatasetRoles, training_kinds: dict, candidate_kinds: dict) -> None:
    """A prediction dataset must contain the ID and feature columns of the
    training roles; numeric features must stay numeric. The class column may
    be missing (it is predicted); other columns are allowed. 298g: the group
    column too, as the trained model's runs read it (as in the Voila GUI)."""
    required = [roles.id_column] + roles.features + (
        [roles.group_column] if roles.group_column else [])
    missing = [name for name in required if name not in candidate_kinds]
    different = [name for name in roles.by_role("numeric")
                 if name in candidate_kinds and candidate_kinds[name] != "number"
                 and training_kinds.get(name) == "number"]
    if missing or different:
        raise IncompatibleDataset(missing, different)


def copy_roles(source: DatasetRoles, target_kinds: dict) -> tuple[dict, list, list]:
    """297d: another dataset's roles for a dataset with the columns (and kinds,
    "number" or "text") in ``target_kinds``, as a suggestion to review and save.

    Returns (mapping, missing, not_numeric): the roles of the columns that
    exist in the target; the columns with a role that do not (left out); and
    numeric columns that are not numbers in the target (kept, for the user to
    review: validation will refuse them). Ignored columns are not copied.
    """
    mapping, missing, not_numeric = {}, [], []
    for name, role in source.columns:
        if role == "ignored":
            continue
        if name not in target_kinds:
            missing.append(name)
            continue
        mapping[name] = role
        if role == "numeric" and target_kinds[name] != "number":
            not_numeric.append(name)
    return mapping, missing, not_numeric
