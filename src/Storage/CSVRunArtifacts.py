"""SQL-independent, non-executing CSV run artifact publication (revision 194).

The authorized caller supplies a trusted scope and a prepared CSVRunContext.
No authentication, ML execution, SQL operations, or web API is implemented here.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageKind, StorageScope
from Storage.CSVRunContext import CSVRunContext


@dataclass(frozen=True)
class PublishedCSVRun:
    run_id: str
    manifest: ArtifactReference
    metrics: ArtifactReference | None


def publish_csv_run(repository, scope: StorageScope, context: CSVRunContext, *,
                    metrics: dict | None = None, configuration_id: str | None = None) -> PublishedCSVRun:
    """Publish a new run manifest and optional numeric metrics, never raw data.

    The final manifest is written last; it is the discovery marker. A failed
    metrics write leaves no published manifest, though an orphan blob may remain.
    """
    if not isinstance(scope, StorageScope) or not isinstance(context, CSVRunContext):
        raise ValueError("Invalid CSV run context")
    if context.scope != scope or context.storage_kind != StorageKind.FILES:
        raise ValueError("CSV run scope mismatch")
    dataset = repository.get_reference(scope, context.artifact_id)
    if (dataset.kind != ArtifactKind.DATASET or dataset.content_type != "text/csv"
            or dataset.dataset_id != context.dataset_id):
        raise ValueError("Invalid CSV run dataset")
    if configuration_id is not None:
        # Resolve and verify the private configuration against the prepared run.
        from Storage.CSVModelConfiguration import load_csv_model_configuration
        config = load_csv_model_configuration(repository, scope, configuration_id)
        if (config.dataset_id != context.dataset_id or
                config.artifact_id != context.artifact_id or
                config.id_column != context.id_column or
                config.class_column != context.class_column or
                config.data_columns + config.text_columns != context.feature_columns):
            raise ValueError("CSV run configuration mismatch")
    # Store only non-sensitive finite numeric aggregate metrics. Never serialize
    # feature names, row IDs, local paths, arbitrary objects or free-form strings.
    if metrics is not None:
        import math
        if (not isinstance(metrics, dict) or len(metrics) > 64 or
            any(not isinstance(key, str) or not key.isascii() or
                not key.replace('_', '').isalnum() or len(key) > 48 or
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) for key, value in metrics.items())):
            raise ValueError("Invalid CSV run metrics")
    run_id = str(uuid4())
    metrics_ref = None
    if metrics is not None:
        metrics_ref = ArtifactReference(scope, str(uuid4()), ArtifactKind.METRICS,
                                        "application/json", run_id=run_id,
                                        dataset_id=context.dataset_id)
        repository.save_artifact(scope, metrics_ref,
                                 BytesIO(json.dumps(metrics, allow_nan=False,
                                                    sort_keys=True).encode('utf-8')))
    manifest_ref = ArtifactReference(scope, str(uuid4()), ArtifactKind.MANIFEST,
                                     "application/json", run_id=run_id,
                                     dataset_id=context.dataset_id)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_id": context.dataset_id,
        "dataset_artifact_id": context.artifact_id,
        "configuration_id": configuration_id,
        "storage_kind": StorageKind.FILES.value,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics_artifact_id": metrics_ref.artifact_id if metrics_ref else None,
    }
    repository.save_artifact(scope, manifest_ref,
                             BytesIO(json.dumps(manifest, sort_keys=True).encode('utf-8')))
    return PublishedCSVRun(run_id, manifest_ref, metrics_ref)
