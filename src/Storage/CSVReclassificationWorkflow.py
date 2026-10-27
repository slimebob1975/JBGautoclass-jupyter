"""Explicit, scoped CSV reclassification and download boundary.

This service is deliberately independent from GUI state and SQL connectors.
The GUI may use it only after a validated CSV run context is available.
"""
from dataclasses import dataclass

from Storage.ArtifactContract import StorageScope
from Storage.CSVReclassification import correct_csv_class, read_corrected_csv
from Storage.CSVModelConfiguration import load_csv_model_configuration


def reclassify_csv_row(repository, scope, configuration_id, row_id, new_class,
                       *, allowed_classes, confirmed=False):
    """Commit one confirmed edit; never silently dispatch to the SQL writer."""
    if confirmed is not True:
        raise ValueError('CSV correction must be confirmed')
    return correct_csv_class(repository, scope, configuration_id, row_id, new_class,
                             allowed_classes=allowed_classes)


def corrected_csv_download(repository, scope, configuration_id, *, revision=None):
    """Return a validated CSV payload for an explicit download action.

    The caller controls delivery; no file is written to the user machine.
    A non-None revision provides an exact historical view.
    """
    configuration = load_csv_model_configuration(repository, scope, configuration_id)
    data = read_corrected_csv(repository, scope, configuration_id, revision=revision)
    fmt = configuration.csv_format or {'delimiter': ',', 'quotechar': '"', 'encoding': 'utf-8-sig'}
    return data, fmt['encoding']


@dataclass(frozen=True)
class CSVReclassificationContext:
    """A validated, explicit CSV edit target; never inferred from SQL GUI widgets."""
    repository: object
    scope: StorageScope
    configuration_id: str
    dataset_id: str
    dataset_artifact_id: str
    id_column: str
    class_column: str
    allowed_classes: tuple[str, ...]

    def correct(self, row_id, new_class, *, confirmed=False):
        # Fail closed if the referenced configuration was removed or changed.
        configuration = load_csv_model_configuration(self.repository, self.scope,
                                                      self.configuration_id)
        if (configuration.dataset_id != self.dataset_id or
                configuration.artifact_id != self.dataset_artifact_id or
                configuration.id_column != self.id_column or
                configuration.class_column != self.class_column):
            raise ValueError('CSV reclassification context no longer matches dataset')
        return reclassify_csv_row(self.repository, self.scope, self.configuration_id,
                                  row_id, new_class, allowed_classes=self.allowed_classes,
                                  confirmed=confirmed)


def prepare_csv_reclassification_context(repository, scope, configuration_id, *,
                                         dataset_id, dataset_artifact_id,
                                         id_column, class_column, allowed_classes):
    """Validate an explicit run identity before enabling a CSV correction action.

    This constructor does not infer scope or class labels from mutable GUI state.
    """
    if not isinstance(scope, StorageScope):
        raise ValueError('Invalid CSV reclassification scope')
    configuration = load_csv_model_configuration(repository, scope, configuration_id)
    if (configuration.dataset_id != dataset_id or
            configuration.artifact_id != dataset_artifact_id or
            configuration.id_column != id_column or
            configuration.class_column != class_column):
        raise ValueError('CSV reclassification source identity mismatch')
    if (not isinstance(allowed_classes, (tuple, list)) or not allowed_classes or
            any(not isinstance(v, str) or not v.strip() for v in allowed_classes) or
            len(set(allowed_classes)) != len(allowed_classes)):
        raise ValueError('Invalid CSV reclassification classes')
    return CSVReclassificationContext(repository, scope, configuration_id, dataset_id,
                                      dataset_artifact_id, id_column, class_column,
                                      tuple(allowed_classes))
