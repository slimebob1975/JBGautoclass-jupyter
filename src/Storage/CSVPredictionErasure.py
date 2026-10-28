"""Delete one authorized CSV prediction run; never touch SQL, datasets, or models."""
from Storage.CSVPredictionExport import _run_manifest
from Storage.ArtifactContract import ArtifactKind


def erase_csv_prediction_run(repository, scope, dataset_artifact_id, manifest_id):
    """Remove the published run and its private payload within a verified scope.

    Unpublish manifest first so an interrupted deletion cannot advertise a
    completed run with missing prediction rows. A failed payload removal may
    leave an orphan, but must never erase another run or its dataset.
    """
    manifest = _run_manifest(repository, scope, dataset_artifact_id, manifest_id)
    payload_id = manifest['predictions_artifact_id']
    # Refuse shared payloads: another manifest may still need these rows.
    for ref in repository.list_artifacts(scope):
        if ref.kind != ArtifactKind.MANIFEST or ref.artifact_id == manifest_id:
            continue
        if ref.run_id == manifest['run_id']:
            raise ValueError('CSV prediction run has ambiguous manifests')
    repository.delete_artifact(scope, manifest_id)
    repository.delete_artifact(scope, payload_id)
    return True
