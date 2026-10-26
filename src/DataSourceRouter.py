"""Explicit SQL/CSV source choice; existing SQL execution remains unchanged."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from CSVDataSource import CSVDataSource, CSVSourceSettings


class SourceKind(str, Enum):
    SQL_SERVER = "sql_server"
    CSV = "csv"


@dataclass(frozen=True)
class DataSourceChoice:
    kind: SourceKind = SourceKind.SQL_SERVER
    csv_settings: Optional[CSVSourceSettings] = None

    def __post_init__(self):
        if not isinstance(self.kind, SourceKind):
            raise ValueError("Unknown data source kind")
        if self.kind is SourceKind.CSV:
            if not isinstance(self.csv_settings, CSVSourceSettings):
                raise ValueError("CSV source settings are required")
        elif self.csv_settings is not None:
            raise ValueError("CSV settings are not valid for SQL Server")


def create_data_source(choice: DataSourceChoice, *, config=None, logger=None, validate=True):
    """Construct a source without loading SQL dependencies in CSV mode.

    SQL still uses its legacy data API; CSV is not yet wired into model runs.
    """
    if not isinstance(choice, DataSourceChoice):
        raise ValueError("An explicit data source choice is required")
    if choice.kind is SourceKind.CSV:
        return CSVDataSource(choice.csv_settings)
    if config is None or logger is None:
        raise ValueError("SQL Server requires a configuration and logger")
    from SQLDataLayer import DataLayer

    return DataLayer(config=config, logger=logger, validate=validate)
