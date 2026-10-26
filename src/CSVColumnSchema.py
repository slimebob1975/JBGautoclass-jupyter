"""CSV column-role discovery independent of SQL and GUI widgets.

This is a structural contract only; it does not activate CSV model execution.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CSVColumnSchema:
    columns: tuple[str, ...]
    unique_id_candidates: tuple[str, ...]
    class_candidates: tuple[str, ...]
    feature_candidates: tuple[str, ...]
    rows: int


def inspect_csv_columns(path, *, delimiter=",", encoding="utf-8-sig", class_cardinality_limit=30):
    """Inspect roles without making assumptions about SQL integer IDs.

    The caller must explicitly choose distinct ID and class roles. Column names
    and row values are never included in exception messages.
    """
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in "\r\n\x00":
        raise ValueError("Invalid CSV delimiter")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("Invalid CSV encoding")
    if not isinstance(class_cardinality_limit, int) or class_cardinality_limit < 2:
        raise ValueError("Invalid class cardinality limit")
    try:
        with Path(path).open("r", encoding=encoding, newline="") as stream:
            reader = csv.reader(stream, delimiter=delimiter, strict=True)
            headers = next(reader)
            if not headers or any(not h.strip() for h in headers) or len(set(headers)) != len(headers):
                raise ValueError("Invalid CSV column headers")
            unique = [set() for _ in headers]
            complete = [True for _ in headers]
            categories = [set() for _ in headers]
            count = 0
            for row in reader:
                if len(row) != len(headers):
                    raise ValueError("Inconsistent CSV row shape")
                count += 1
                for i, value in enumerate(row):
                    if not value:
                        complete[i] = False
                    if value in unique[i]:
                        complete[i] = False
                    unique[i].add(value)
                    if value:
                        categories[i].add(value)
            if not count:
                raise ValueError("CSV requires at least one data row")
    except (OSError, UnicodeError, LookupError, csv.Error, StopIteration):
        raise ValueError("Unable to inspect CSV columns") from None
    id_columns = tuple(h for i, h in enumerate(headers) if complete[i])
    class_columns = tuple(h for i, h in enumerate(headers)
                          if 2 <= len(categories[i]) <= class_cardinality_limit)
    return CSVColumnSchema(tuple(headers), id_columns, class_columns, tuple(headers), count)


def validate_csv_column_roles(schema, *, id_column, class_column, feature_columns):
    """Fail closed on missing, overlapping or empty roles; no raw values in errors."""
    if not isinstance(schema, CSVColumnSchema):
        raise ValueError("Invalid CSV schema")
    if id_column not in schema.unique_id_candidates:
        raise ValueError("CSV ID must be populated and unique")
    if class_column not in schema.class_candidates or class_column == id_column:
        raise ValueError("Invalid CSV class selection")
    try:
        selected = tuple(feature_columns)
        if not selected or len(set(selected)) != len(selected) or any(
            col not in schema.feature_candidates or col in (id_column, class_column)
            for col in selected
        ):
            raise ValueError("Invalid CSV feature selection")
    except TypeError:
        raise ValueError("Invalid CSV feature selection") from None
    return selected
