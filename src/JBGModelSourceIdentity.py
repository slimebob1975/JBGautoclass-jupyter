"""Source guard for reusing JBG's existing model files and embedded Config.

A CSV filename is display metadata, never the identity of an uploaded dataset.
SQL model loading is intentionally unchanged.
"""

from __future__ import annotations


def check_model_source_identity(saved_config, runtime_config) -> None:
    """Reject cross-source and cross-dataset model reuse, without touching SQL.

    Compatibility: historical model configurations lacking ``data_source``
    are SQL. This check does not authorize access to an artifact repository;
    scoped access must still be enforced by the data adapter.
    """
    saved = getattr(saved_config, "data_source", None)
    runtime = getattr(runtime_config, "data_source", None)
    saved_kind = getattr(saved, "kind", "sql")
    runtime_kind = getattr(runtime, "kind", "sql")
    if saved_kind not in ("sql", "csv") or runtime_kind not in ("sql", "csv"):
        raise ValueError("Unsupported model data source")
    if saved_kind == runtime_kind == "sql":
        return
    if saved_kind != runtime_kind:
        raise ValueError("Saved model data source differs from the selected source")
    # A model trained on one uploaded dataset must not silently run against
    # another upload with the same display filename.
    for name in ("dataset_id", "artifact_id"):
        saved_id = getattr(saved, name, None)
        runtime_id = getattr(runtime, name, None)
        if not isinstance(saved_id, str) or not saved_id or saved_id != runtime_id:
            raise ValueError("Saved CSV model references a different dataset")
