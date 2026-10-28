"""Read-only discovery of trusted, source-bound CSV model checkpoints.

This does not enable prediction, training, or SQL model selection. The caller
must authorize the local storage scope and choose a registered CSV dataset.
"""
from pathlib import Path
from uuid import UUID

from Storage.ArtifactContract import ArtifactKind, StorageScope
from Storage.CSVModelConfiguration import load_csv_model_configuration, list_csv_model_configurations


def _model_filename(model_root, scope, dataset_id, artifact_id):
    if not isinstance(scope, StorageScope):
        raise ValueError('Invalid CSV scope')
    for value in (scope.tenant_id, scope.project_id, dataset_id, artifact_id):
        UUID(value)  # Reject traversal or uncontrolled filenames.
    root = Path(model_root)
    if not root.is_absolute() or root.is_symlink() or '..' in root.parts:
        raise ValueError('Invalid CSV model root')
    folder = root / scope.tenant_id / scope.project_id / dataset_id
    candidate = folder / ('csv_model_' + artifact_id + '.sav')
    if folder.is_symlink() or candidate.is_symlink():
        raise ValueError('Unsafe CSV checkpoint path')
    return candidate


def _registered_algorithm_matches_checkpoint(registered, persisted, components):
    """Compare algorithm identities, not the representation or search breadth.

    A selected subset is valid only if the persisted pipeline identifies the
    same selected algorithm. Never accept an out-of-registration algorithm.
    """
    def codes(value):
        if isinstance(value, str):
            value = value.split(',')
        if not isinstance(value, (list, tuple)) or not value:
            return None
        if not all(isinstance(item, str) and item.strip() for item in value):
            return None
        result = tuple(item.strip() for item in value)
        return result if len(result) == len(set(result)) else None

    requested, selected = codes(registered), codes(persisted)
    if requested is None or selected is None or not set(selected).issubset(requested):
        return False
    if set(selected) == set(requested):
        return True
    # A narrowed choice must be a single fitted algorithm, not an arbitrary
    # subset of a search list. The saved pipeline descriptor is independent of
    # the mutable configuration and was already semantically reviewed above.
    if len(selected) != 1 or len(components) < 3:
        return False
    descriptor = components[2]
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) < 5:
        return False
    algorithm = descriptor[4]
    return getattr(algorithm, 'name', None) == selected[0]


def inspect_saved_csv_model(repository, scope, dataset_artifact_id, configuration_id,
                            *, model_root):
    """Verify one locally generated model, without executing the classifier.

    Saved pickle/dill artifacts must be trusted local artifacts: loading them
    can execute serialized code. Never pass arbitrary uploaded model filenames.
    """
    dataset = repository.get_reference(scope, dataset_artifact_id)
    if dataset.kind != ArtifactKind.DATASET or dataset.content_type != 'text/csv':
        raise ValueError('Expected a registered CSV dataset')
    roles = load_csv_model_configuration(repository, scope, configuration_id)
    if roles.artifact_id != dataset_artifact_id or roles.dataset_id != dataset.dataset_id:
        raise ValueError('CSV checkpoint belongs to another dataset')
    filename = _model_filename(model_root, scope, roles.dataset_id, roles.artifact_id)
    if not filename.is_file():
        raise ValueError('No saved CSV model exists for this dataset')

    from JBGModelPersistence import load_model_artifact_with_compatibility
    from JBGModelCompatibility import review_artifact_semantics, review_model_source_boundary
    from Config import Config
    components, compatibility = load_model_artifact_with_compatibility(filename)
    saved_config = components[0]
    Config._validate_current_saved_config(saved_config)
    if not isinstance(roles.shared_json, dict):
        raise ValueError('Saved CSV registration lacks a runtime configuration')
    source = saved_config.data_source
    if (getattr(source, 'kind', None) != 'csv'
            or source.dataset_id != roles.dataset_id
            or source.artifact_id != roles.artifact_id):
        raise ValueError('CSV model source identity mismatch')
    boundary = review_model_source_boundary(saved_config, saved_config, compatibility)
    semantics = review_artifact_semantics(saved_config, components[1], compatibility)
    if not boundary.ok or not semantics.ok:
        raise ValueError('Saved CSV model compatibility verification failed')
    if (saved_config.get_id_column_name() != roles.id_column
            or saved_config.get_class_column_name() != roles.class_column
            or tuple(saved_config.get_numerical_column_names()) != roles.data_columns
            or tuple(saved_config.get_text_column_names()) != roles.text_columns):
        raise ValueError('Saved CSV model roles differ from the registration')
    # Dataset registration describes the input schema, not the training recipe.
    # A saved model may use different estimators, scaling, sampling, scoring,
    # or holdout settings. Its own persisted pipeline is authoritative for
    # those choices and was checked by review_artifact_semantics above.
    if compatibility.get('source_kind') != 'csv':
        raise ValueError('Saved model lacks CSV source compatibility')
    return filename


def diagnose_saved_csv_models(repository, scope, dataset_artifact_id, *, model_root):
    """Return verified options and aggregated, non-sensitive rejection reasons.

    Do not expose artifact IDs, filesystem paths, configuration payloads or
    exception text in the GUI. A rejected model is never selectable.
    """
    options = []
    rejected = {}
    configurations = list_csv_model_configurations(repository, scope, dataset_artifact_id)
    for label, configuration_id in configurations:
        try:
            filename = inspect_saved_csv_model(repository, scope, dataset_artifact_id,
                                               configuration_id, model_root=model_root)
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            message = str(exc)
            if isinstance(exc, ValueError) and message == 'No saved CSV model exists for this dataset':
                reason = 'checkpoint_missing'
            elif isinstance(exc, ValueError) and message.startswith('Saved CSV checkpoint configuration differs from registration'):
                field = message.partition(': ')[2]
                allowed_fields = ('algorithm', 'preprocessor', 'feature_selection',
                                  'oversampler', 'undersampler', 'scoring', 'test_size')
                reason = ('configuration_mismatch_' + field if field in allowed_fields
                          else 'configuration_mismatch')
            elif isinstance(exc, ValueError) and ('source' in message.lower() or 'another dataset' in message.lower()):
                reason = 'source_mismatch'
            elif isinstance(exc, ValueError) and ('roles' in message.lower()):
                reason = 'column_roles_mismatch'
            else:
                reason = 'verification_failed'
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        options.append((label, configuration_id, filename))
    return tuple(options), rejected, len(configurations)


def list_saved_csv_models(repository, scope, dataset_artifact_id, *, model_root):
    """List only checkpoints passing the same strict read-only inspection."""
    verified, _rejected, _total = diagnose_saved_csv_models(
        repository, scope, dataset_artifact_id, model_root=model_root)
    return verified


def saved_csv_model_label(filename, configuration_id, project_name=""):
    """Summarize a verified local checkpoint, not the original search recipe.

    Never include column names, filenames, user-supplied configuration names or
    arbitrary serialized values in a dropdown label. Caller must first inspect
    the model with inspect_saved_csv_model.
    """
    from JBGModelPersistence import load_model_artifact_with_compatibility

    components, _compatibility = load_model_artifact_with_compatibility(filename)
    mode = components[0]._config_to_json().get('mode', {})

    def codes(key):
        value = mode.get(key, '')
        if isinstance(value, str):
            values = value.split(',')
        elif isinstance(value, (tuple, list)):
            values = value
        else:
            return '?'
        # Only compact enum-like codes; never expose unbounded private text.
        if not values or any(not isinstance(v, str) or not v.strip().isalnum()
                             or len(v.strip()) > 16 for v in values):
            return '?'
        return ','.join(v.strip() for v in values[:3]) + ('+' if len(values) > 3 else '')

    # Project is a user-facing local name, not a checkpoint identity. Never
    # substitute the project name for scope or source verification.
    # Keep labels single-line and bounded even for unusual input values.
    safe_project = ''.join(ch for ch in str(project_name) if ch.isprintable() and ch not in '|\r\n').strip()
    safe_project = safe_project[:48].rstrip() or 'Project'

    return (safe_project + ' | CSV ' + codes('algorithm') + ' | ' + codes('preprocessor')
            + ' | FS:' + codes('feature_selection')
            + ' | OS:' + codes('oversampler')
            + ' | US:' + codes('undersampler')
            + ' [' + configuration_id[:8] + ']')
