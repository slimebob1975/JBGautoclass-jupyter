"""GUI-independent, non-executing run preparation (revision 280y).

Source-specific validators are trusted dependencies supplied by a composition
root. They must validate the existing backend configuration for that source;
this service deliberately does not duplicate Config or ML-engine logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from JBGApplicationContracts import DataSourceKind, DatasetReference
from JBGDatasetApplicationService import DatasetApplicationService


@runtime_checkable
class RunConfigurationValidator(Protocol):
    def validate(self, configuration: object, resolved_dataset: object) -> None:
        """Raise if the configuration is invalid for this source/dataset."""
        ...


@dataclass(frozen=True)
class RunConfiguration:
    """Explicit source binding around an existing backend Config object.

    The payload is passed by identity, never copied or normalized. The caller
    must treat it as immutable between preparation and execution.
    """
    source: DataSourceKind
    payload: object

    def __post_init__(self) -> None:
        if not isinstance(self.source, DataSourceKind):
            raise TypeError("configuration source must be a DataSourceKind")
        if self.payload is None:
            raise ValueError("configuration payload is required")


@dataclass(frozen=True)
class PreparedRun:
    """Validated pairing, not an authorization token or executable job."""
    dataset: DatasetReference
    resolved_dataset: object
    configuration: RunConfiguration


class RunPreparationService:
    """Resolve, then validate with the *same* source's trusted validator."""

    def __init__(self, datasets: DatasetApplicationService, *,
                 sql: RunConfigurationValidator | None = None,
                 csv: RunConfigurationValidator | None = None) -> None:
        if not isinstance(datasets, DatasetApplicationService):
            raise TypeError("datasets must be a DatasetApplicationService")
        self._datasets = datasets
        self._validators = {}
        for kind, validator in ((DataSourceKind.SQL, sql), (DataSourceKind.CSV, csv)):
            if validator is not None:
                if not isinstance(validator, RunConfigurationValidator):
                    raise TypeError("run validator must implement validate(configuration, dataset)")
                self._validators[kind] = validator

    def prepare(self, reference: DatasetReference,
                configuration: RunConfiguration) -> PreparedRun:
        if not isinstance(reference, DatasetReference):
            raise TypeError("reference must be a DatasetReference")
        if not isinstance(reference.source, DataSourceKind):
            raise ValueError("invalid dataset source")
        if not isinstance(configuration, RunConfiguration):
            raise TypeError("configuration must be a RunConfiguration")
        if (not isinstance(configuration.source, DataSourceKind)
                or configuration.source != reference.source):
            raise ValueError("dataset and configuration sources differ")
        if configuration.payload is None:
            raise ValueError("configuration payload is required")
        validator = self._validators.get(reference.source)
        if validator is None:
            raise ValueError("run validator for dataset source is not configured")
        # No fallback; dataset resolver enforces its own scope/identity boundary.
        resolved = self._datasets.resolve(reference)
        # Validation failure propagates unchanged and cannot yield PreparedRun.
        validator.validate(configuration.payload, resolved)
        return PreparedRun(reference, resolved, configuration)
