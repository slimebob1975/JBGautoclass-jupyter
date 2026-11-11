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
    model_source = getattr(runtime_config, "model_source", None)
    if model_source is not None:
        # 297e (opt-in, runtime only): a prediction run reads another CSV dataset
        # of the same project. The model must still be the one trained on
        # model_source, and both sides must be CSV.
        if getattr(runtime, "kind", None) != "csv" or getattr(model_source, "kind", None) != "csv":
            raise ValueError("A model source applies to CSV predictions only")
        runtime = model_source
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
