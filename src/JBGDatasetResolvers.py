"""Trusted, source-specific adapters for DatasetApplicationService (280x).

These adapters resolve existing dataset identities; they do not authorize callers,
open CSV content, execute SQL or create new training configurations.
"""
from __future__ import annotations

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageScope


class ScopedCSVDatasetResolver:
    """Use the existing repository's scoped identity checks, without staging bytes.

    Construct this only after the calling identity has been authenticated and
    authorized for the supplied scope. StorageScope alone is not authorization.
    DatasetReference.key is an artifact_id (not a dataset_id or file path).
    """

    def __init__(self, repository, scope: StorageScope):
        if not isinstance(scope, StorageScope):
            raise TypeError("CSV resolver requires a trusted StorageScope")
        self._repository = repository
        self._scope = scope

    def resolve(self, key: str) -> ArtifactReference:
        # Repository performs UUID validation and scoped lookup; do not interpret
        # the opaque key as a path or retry in another scope.
        reference = self._repository.get_reference(self._scope, key)
        if (not isinstance(reference, ArtifactReference)
                or reference.scope != self._scope
                or reference.artifact_id != key
                or reference.kind != ArtifactKind.DATASET
                or reference.content_type != "text/csv"
                or not reference.dataset_id):
            raise ValueError("Not a registered CSV dataset in this scope")
        return reference


class ConfiguredSQLDatasetResolver:
    """Resolve only the selected table of an existing, trusted SQL DataLayer.

    The caller passes an already configured SQL DataLayer and its matching
    config. No connection is opened and no query or table interpolation occurs.
    """

    def __init__(self, data_layer, config):
        self._data_layer = data_layer
        self._config = config

    def resolve(self, key: str):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Invalid SQL dataset key")
        configured = self._config.get_data_table(include_database=True)
        if not isinstance(configured, str) or not configured.strip() or key != configured:
            raise ValueError("SQL dataset does not match configured table")
        return self._data_layer
