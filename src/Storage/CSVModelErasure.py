"""Delete exactly one scoped CSV checkpoint and its configuration registration.

Only trusted local GUI code may call this function. No SQL or dataset deletion.
"""
from Storage.ArtifactContract import ArtifactKind
from Storage.CSVModelConfiguration import load_csv_model_configuration, list_csv_model_configurations
from Storage.CSVSavedModelInspection import _model_filename


def erase_saved_csv_model(repository, scope, dataset_artifact_id, configuration_id, *, model_root):
    """Remove a model; permit an already absent checkpoint for stale registrations.

    Confirm the dataset/configuration relationship before touching the filesystem.
    Never follow symlinks, and never erase results, dataset, or other models.
    """
    dataset = repository.get_reference(scope, dataset_artifact_id)
    if dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv':
        raise ValueError('Expected registered CSV dataset')
    roles = load_csv_model_configuration(repository, scope, configuration_id)
    if roles.artifact_id != dataset_artifact_id or roles.dataset_id != dataset.dataset_id:
        raise ValueError('CSV model belongs to another dataset')
    filename = _model_filename(model_root, scope, roles.dataset_id, roles.artifact_id)
    if filename.is_symlink() or filename.parent.is_symlink():
        raise ValueError('Unsafe CSV checkpoint path')
    # Several configurations can refer to the same physical checkpoint when
    # they share a dataset artifact. Never remove a file still referenced by
    # another registration, even if that registration is not GUI-selectable.
    others = [identifier for _, identifier in list_csv_model_configurations(
        repository, scope, dataset_artifact_id) if identifier != configuration_id]
    if not others:
        # Remove checkpoint first so an I/O failure keeps registration intact.
        filename.unlink(missing_ok=True)
    repository.delete_artifact(scope, configuration_id)
    return True
