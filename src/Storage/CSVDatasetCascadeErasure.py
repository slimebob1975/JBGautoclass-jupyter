"""Preflighted, scope-bound cascade deletion for a registered CSV dataset.

Only trusted local GUI callers may authorize this destructive operation.
"""
import json
from pathlib import Path

from Storage.ArtifactContract import ArtifactKind, StorageScope
from Storage.CSVModelConfiguration import load_csv_model_configuration
from Storage.CSVPredictionExport import _run_manifest
from Storage.CSVSavedModelInspection import _model_filename


def erase_csv_dataset_cascade(repository, scope, artifact_id, *, model_root):
    """Remove one dataset, its model registrations and prediction history.

    Preflight all relationships and filesystem paths before mutating anything.
    An interrupted deletion can leave orphan bytes; it must not expose a
    completed run missing its payload or erase unrelated scoped artifacts.
    """
    if not isinstance(scope, StorageScope):
        raise ValueError('Invalid CSV scope')
    dataset = repository.get_reference(scope, artifact_id)
    if (dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv'
            or not dataset.dataset_id):
        raise ValueError('Invalid registered CSV dataset')
    refs = repository.list_artifacts(scope)
    # Do not infer ownership if multiple dataset objects share an identity.
    if any(ref.kind == ArtifactKind.DATASET and ref.dataset_id == dataset.dataset_id
           and ref.artifact_id != artifact_id for ref in refs):
        raise ValueError('Ambiguous CSV dataset identity')
    associated = [ref for ref in refs if ref.dataset_id == dataset.dataset_id
                  and ref.artifact_id != artifact_id]
    to_remove = set()
    checkpoints = set()
    for ref in associated:
        if ref.kind == ArtifactKind.MANIFEST and ref.run_id is None:
            # Model config or original dataset display manifest.
            try:
                with repository.open_artifact(scope, ref.artifact_id) as stream:
                    record = json.load(stream)
            except (ValueError, OSError, TypeError, UnicodeError):
                raise ValueError('Unreadable CSV dataset dependency') from None
            if isinstance(record, dict) and record.get('configuration_id') is not None:
                config = load_csv_model_configuration(repository, scope, ref.artifact_id)
                if config.artifact_id != artifact_id:
                    raise ValueError('CSV model belongs to another dataset')
                checkpoints.add(_model_filename(model_root, scope, config.dataset_id, config.artifact_id))
            elif (isinstance(record, dict) and record.get('dataset_id') == dataset.dataset_id
                  and isinstance(record.get('filename'), str)
                  and record.get('filename').lower().endswith('.csv')):
                pass  # Display manifest for this dataset.
            else:
                raise ValueError('Unrecognized CSV dataset dependency')
            to_remove.add(ref.artifact_id)
        elif ref.kind == ArtifactKind.MANIFEST and ref.run_id is not None:
            manifest = _run_manifest(repository, scope, artifact_id, ref.artifact_id)
            to_remove.add(ref.artifact_id)
            to_remove.add(manifest['predictions_artifact_id'])
        elif ref.kind == ArtifactKind.PREDICTIONS:
            # Refuse orphan results, rather than silently destroying unknown data.
            continue
        else:
            raise ValueError('Unrecognized CSV dataset dependency')
    if any(ref.kind == ArtifactKind.PREDICTIONS and ref.artifact_id not in to_remove
           for ref in associated):
        raise ValueError('Unreferenced CSV prediction payload')
    # Do not delete any artifact referenced by another dataset/run.
    if any(ref.artifact_id in to_remove and ref.dataset_id != dataset.dataset_id for ref in refs):
        raise ValueError('Shared CSV artifact')
    for filename in checkpoints:
        if filename.is_symlink() or filename.parent.is_symlink():
            raise ValueError('Unsafe CSV checkpoint')
        if filename.exists() and not filename.is_file():
            raise ValueError('Invalid CSV checkpoint')
    # Recheck all deletable artifacts exist before beginning a destructive pass.
    for ident in to_remove:
        repository.get_reference(scope, ident)
    # Unpublish manifests first, then prediction payloads, then model configs.
    # Finally remove dataset and associated display metadata.
    manifests = [ref.artifact_id for ref in associated
                 if ref.kind == ArtifactKind.MANIFEST and ref.run_id is not None]
    payloads = [ref.artifact_id for ref in associated if ref.kind == ArtifactKind.PREDICTIONS]
    other_manifests = [ref.artifact_id for ref in associated
                       if ref.kind == ArtifactKind.MANIFEST and ref.run_id is None]
    for ident in manifests + payloads + other_manifests:
        repository.delete_artifact(scope, ident)
    for filename in checkpoints:
        filename.unlink(missing_ok=True)
    repository.delete_artifact(scope, artifact_id)
    return True
