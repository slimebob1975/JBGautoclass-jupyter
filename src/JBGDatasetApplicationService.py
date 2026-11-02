"""GUI-free dataset dispatch boundary (revision 280w).

Adapters are supplied by a trusted composition root. For CSV, that adapter
must enforce StorageScope and dataset/artifact identity using existing storage
contracts. SQL remains the responsibility of its existing data layer. This
service deliberately neither loads frames nor constructs SQL queries.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from JBGApplicationContracts import DataSourceKind, DatasetReference


@runtime_checkable
class DatasetResolver(Protocol):
    """Source-local validation; raise on missing, unauthorized or invalid data."""

    def resolve(self, key: str) -> object:
        ...


class DatasetApplicationService:
    """Resolve opaque references via explicit, source-scoped adapters.

    No implicit default source, cross-source retry, untrusted path conversion,
    or normalization of keys is permitted. The resolved value is adapter-owned.
    """

    def __init__(self, *, sql: DatasetResolver | None = None,
                 csv: DatasetResolver | None = None) -> None:
        self._resolvers = {}
        for kind, resolver in ((DataSourceKind.SQL, sql), (DataSourceKind.CSV, csv)):
            if resolver is not None:
                if not isinstance(resolver, DatasetResolver):
                    raise TypeError("dataset adapter must implement resolve(key)")
                self._resolvers[kind] = resolver

    def resolve(self, reference: DatasetReference) -> object:
        if not isinstance(reference, DatasetReference):
            raise TypeError("reference must be a DatasetReference")
        # Recheck all fields: dataclass frozen is not an authentication boundary;
        # object.__new__ / custom deserializers can bypass __post_init__.
        if not isinstance(reference.source, DataSourceKind):
            raise ValueError("invalid dataset source")
        if not isinstance(reference.key, str) or not reference.key.strip():
            raise ValueError("invalid dataset key")
        resolver = self._resolvers.get(reference.source)
        if resolver is None:
            raise ValueError("dataset source is not configured")
        # Propagate adapter errors. In particular do not fallback to another
        # resolver, since that could violate source or storage-scope isolation.
        return resolver.resolve(reference.key)
