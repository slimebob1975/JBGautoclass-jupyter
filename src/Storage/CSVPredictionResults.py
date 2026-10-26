"""Scoped CSV prediction result sink matching the existing data-layer save API.

This module does not enable CSV ML execution or authorize a storage scope.
The caller must authorize the scope and provide a verified CSVRunContext.
"""

from datetime import datetime, timezone
from io import BytesIO
import json
import math
from uuid import uuid4

from Storage.ArtifactContract import ArtifactKind, ArtifactReference, StorageKind, StorageScope
from Storage.CSVRunContext import CSVRunContext


class CSVPredictionResultSink:
    """Persist prediction rows privately; publish a discoverable manifest last."""

    def __init__(self, repository, scope: StorageScope, context: CSVRunContext):
        if (not isinstance(scope, StorageScope) or not isinstance(context, CSVRunContext)
                or context.scope != scope or context.storage_kind != StorageKind.FILES):
            raise ValueError("Invalid CSV prediction scope")
        self.repository = repository
        self.scope = scope
        self.context = context

    def save_prediction_data(self, results: list, class_rate_type, model_name: str,
                             class_labels, create_tables: bool = True, test_run: int = 0) -> dict:
        """Use the SQL adapter's call signature, without accessing SQL Server.

        The saved row payload is private and may contain individual predictions.
        Never include it in logs, UI summaries, or a public manifest.
        """
        context = self.context
        dataset = self.repository.get_reference(self.scope, context.artifact_id)
        if (dataset.kind != ArtifactKind.DATASET or dataset.content_type != "text/csv"
                or dataset.dataset_id != context.dataset_id):
            raise ValueError("CSV prediction dataset mismatch")
        if not isinstance(results, list) or not isinstance(model_name, str) or not model_name:
            raise ValueError("Invalid CSV prediction input")
        if not isinstance(test_run, int) or isinstance(test_run, bool) or test_run < 0:
            raise ValueError("Invalid CSV prediction test run")
        # class labels and algorithm metadata are deliberately excluded from the
        # public manifest; the model's JSON configuration already stores them.
        if not isinstance(class_labels, (list, tuple)) or not all(isinstance(c, str) for c in class_labels):
            raise ValueError("Invalid CSV prediction class labels")
        try:
            rate_type = class_rate_type.name
        except AttributeError:
            raise ValueError("Invalid CSV prediction rate type") from None
        if not isinstance(rate_type, str) or not rate_type:
            raise ValueError("Invalid CSV prediction rate type")
        normalized = []
        seen = set()
        # The model supplies labels in probability-column order. Validate the
        # optional probability vector before publishing any row artifact.
        if len(set(class_labels)) != len(class_labels):
            raise ValueError("Duplicate CSV prediction class label")
        for item in results:
            if not isinstance(item, dict) or set(item) != {"key", "prediction", "rate", "probabilities"}:
                raise ValueError("Invalid CSV prediction row")
            # IDs must not undergo integer coercion: leading zeroes matter.
            key = item["key"]
            if not isinstance(key, (str, int)) or isinstance(key, bool) or not str(key):
                raise ValueError("Invalid CSV prediction ID")
            key = str(key)
            if key in seen:
                raise ValueError("Duplicate CSV prediction ID")
            seen.add(key)
            prediction = item["prediction"]
            rate = item["rate"]
            probabilities = item["probabilities"]
            if (not isinstance(prediction, (str, int, float)) or isinstance(prediction, bool)
                    or not isinstance(rate, (int, float)) or isinstance(rate, bool)
                    or not math.isfinite(rate) or
                    not isinstance(probabilities, (str, int, float, type(None)))):
                raise ValueError("Invalid CSV prediction value")
            if isinstance(prediction, float) and not math.isfinite(prediction):
                raise ValueError("Invalid CSV prediction value")
            if isinstance(probabilities, float) and not math.isfinite(probabilities):
                raise ValueError("Invalid CSV prediction value")
            if class_labels and prediction not in class_labels:
                raise ValueError("CSV prediction class is not in the model labels")
            if isinstance(probabilities, str) and probabilities.strip():
                try:
                    vector = [float(part.strip()) for part in probabilities.split(",")]
                except ValueError:
                    raise ValueError("Invalid CSV prediction probability vector") from None
                if (any(not math.isfinite(value) or value < 0 or value > 1 for value in vector)
                        or (class_labels and len(vector) != len(class_labels))):
                    raise ValueError("Invalid CSV prediction probability vector")
            normalized.append({"key": key, "prediction": prediction,
                               "rate": float(rate), "probabilities": probabilities})
        run_id = str(uuid4())
        payload_ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.PREDICTIONS,
                                        "application/json", run_id=run_id,
                                        dataset_id=context.dataset_id)
        manifest_ref = ArtifactReference(self.scope, str(uuid4()), ArtifactKind.MANIFEST,
                                         "application/json", run_id=run_id,
                                         dataset_id=context.dataset_id)
        # Private row artifact first, discovery manifest last.
        payload = {"schema_version": 1, "rows": normalized, "rate_type": rate_type}
        self.repository.save_artifact(self.scope, payload_ref, BytesIO(
            json.dumps(payload, allow_nan=False).encode("utf-8")))
        manifest = {"schema_version": 1, "run_id": run_id,
                    "dataset_id": context.dataset_id,
                    "dataset_artifact_id": context.artifact_id,
                    "predictions_artifact_id": payload_ref.artifact_id,
                    "row_count": len(normalized), "storage_kind": StorageKind.FILES.value,
                    "status": "predictions_saved",
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.repository.save_artifact(self.scope, manifest_ref, BytesIO(
            json.dumps(manifest, sort_keys=True).encode("utf-8")))
        return {"error": None, "results": {"run_id": run_id,
                "row_count": len(normalized), "artifact_id": manifest_ref.artifact_id}}
