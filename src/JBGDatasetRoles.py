"""Column roles belong to the training dataset (revision 293a, decision A).

One role per column: ``id``, ``class``, ``numeric``, ``text``, ``categorical``
or ``ignored``; columns without a role are ignored. Training and mixed
datasets carry roles; prediction datasets use the roles of the project's
active training dataset and must be compatible with it. Runs take the roles
from their dataset; configurations hold only the machine-learning settings.

Mapping to the engine's Config: ``categorical`` columns are text columns that
are categorised, so they appear in ``data_text_columns`` and in
``mode.category_text_columns``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

ROLES = ("id", "class", "numeric", "text", "categorical", "ignored")
FEATURE_ROLES = ("numeric", "text", "categorical")
MAX_COLUMNS = 2000


class IncompatibleDataset(ValueError):
    """A prediction dataset lacks training columns or has other column kinds."""

    def __init__(self, missing=(), different=()):
        super().__init__("the dataset is not compatible with the training dataset")
        self.missing, self.different = tuple(missing), tuple(different)


@dataclass(frozen=True)
class DatasetRoles:
    columns: tuple            # ((column name, role), ...) in the dataset's order

    @classmethod
    def from_mapping(cls, mapping) -> "DatasetRoles":
        if not isinstance(mapping, dict) or len(mapping) > MAX_COLUMNS:
            raise ValueError("column roles must map column names to roles")
        pairs = []
        for name, role in mapping.items():
            if not isinstance(name, str) or not name or len(name) > 200:
                raise ValueError("column names must be text of 1-200 characters")
            if role not in ROLES:
                raise ValueError("unknown column role")
            pairs.append((name, role))
        return cls(tuple(pairs))

    @classmethod
    def from_json(cls, text: str) -> "DatasetRoles":
        return cls.from_mapping(dict(json.loads(text)))

    def to_json(self) -> str:
        return json.dumps(dict(self.columns))

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

    def apply_to(self, config) -> None:
        """Set the engine's role fields of a Config from these roles."""
        text, categorical = self.by_role("text"), self.by_role("categorical")
        config.connection.id_column = self.id_column
        config.connection.class_column = self.class_column
        config.connection.data_numerical_columns = self.by_role("numeric")
        config.connection.data_text_columns = text + categorical
        config.mode.category_text_columns = categorical


def check_compatible(roles: DatasetRoles, training_kinds: dict, candidate_kinds: dict) -> None:
    """A prediction dataset must contain the ID and feature columns of the
    training roles; numeric features must stay numeric. The class column may
    be missing (it is predicted); other columns are allowed."""
    required = [roles.id_column] + roles.features
    missing = [name for name in required if name not in candidate_kinds]
    different = [name for name in roles.by_role("numeric")
                 if name in candidate_kinds and candidate_kinds[name] != "number"
                 and training_kinds.get(name) == "number"]
    if missing or different:
        raise IncompatibleDataset(missing, different)
