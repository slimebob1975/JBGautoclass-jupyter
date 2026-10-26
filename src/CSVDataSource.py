"""Opt-in, file-backed tabular input for a future dual-source data layer.

This module deliberately does not change SQLDataLayer or GUI source selection.
No source file name, row contents or column names are logged by this module.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CSVSourceSettings:
    """Explicit CSV input settings; no automatic encoding/delimiter guessing."""

    path: Path
    id_column: str
    class_column: str
    delimiter: str = ","
    encoding: str = "utf-8-sig"

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("A CSV file path is required")
        object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.delimiter, str) or len(self.delimiter) != 1 or self.delimiter in "\r\n\x00":
            raise ValueError("CSV delimiter must be one non-newline character")
        if not isinstance(self.encoding, str) or not self.encoding.strip():
            raise ValueError("CSV encoding is required")
        if not self.id_column or not self.class_column or self.id_column == self.class_column:
            raise ValueError("CSV ID and class columns must be distinct and nonempty")


class CSVDataSource:
    """Load string-preserving CSV data, with explicit column role checks.

    CSV is intentionally not wired to model persistence, prediction export or
    GUI actions until their source-identity and output contracts are defined.
    """

    def __init__(self, settings: CSVSourceSettings):
        self.settings = settings

    def read_frame(self) -> pd.DataFrame:
        if not self.settings.path.is_file():
            raise ValueError("CSV input file is unavailable")
        try:
            frame = pd.read_csv(
                self.settings.path,
                sep=self.settings.delimiter,
                encoding=self.settings.encoding,
                dtype=str,
                keep_default_na=False,
                index_col=False,
                on_bad_lines="error",
            )
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, LookupError, ValueError):
            # Never propagate parser diagnostics containing user rows or paths.
            raise ValueError("Unable to read CSV input with the specified settings") from None
        if not len(frame.columns) or not len(frame):
            raise ValueError("CSV input must contain headers and at least one row")
        columns = list(frame.columns)
        if any(not name or name.startswith("Unnamed:") for name in columns):
            raise ValueError("CSV input contains an empty column header")
        if len(columns) != len(set(columns)) or any(
            name.rsplit(".", 1)[-1].isdigit() and name.rsplit(".", 1)[0] in columns
            for name in columns
        ):
            # pandas disambiguates duplicate headers by appending .1, .2, etc.
            raise ValueError("CSV input contains potentially duplicate column headers")
        if self.settings.id_column not in columns or self.settings.class_column not in columns:
            raise ValueError("CSV ID or class column is not present")
        if frame[self.settings.id_column].eq("").any() or not frame[self.settings.id_column].is_unique:
            raise ValueError("CSV ID values must be populated and unique")
        return frame
