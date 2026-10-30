"""Prepare a source-bound train-only CSV config without reusing SQL credentials.

The CSV GUI uses this boundary for training; prediction and saved-model reuse stay separate.
"""
from copy import deepcopy
from pathlib import Path
from uuid import UUID

from Storage.ArtifactContract import ArtifactKind, StorageScope
from Storage.CSVModelConfiguration import save_csv_model_configuration


def prepare_csv_gui_training_config(base_config, repository, scope, dataset_artifact_id, *,
                                    id_column, class_column, data_columns,
                                    text_columns=(), csv_format=None, model_root=None,
                                    predict=False, mispredicted=False):
    """Return (runtime_config, registered_config) after fail-closed validation.

    The caller supplies the already-collected ML settings as a Config instance;
    only the data-source/mode/column-role boundary is adapted here. SQL-related
    fields and notification destinations are not carried into the saved artifact.
    """
    from Config import Config

    if not isinstance(base_config, Config) or not isinstance(scope, StorageScope):
        raise ValueError('CSV training requires a validated configuration and scope')
    if (not isinstance(dataset_artifact_id, str) or not dataset_artifact_id
            or not isinstance(data_columns, (tuple, list))
            or not isinstance(text_columns, (tuple, list))):
        raise ValueError('CSV training requires registered data and column roles')
    role_names = (id_column, class_column, *data_columns, *text_columns)
    if (len(role_names) != len(set(role_names)) or
            any(not isinstance(name, str) or not name.strip() for name in role_names)
            or not (data_columns or text_columns)):
        raise ValueError('Invalid CSV column roles')
    ref = repository.get_reference(scope, dataset_artifact_id)
    if (ref.kind != ArtifactKind.DATASET or ref.content_type != 'text/csv'
            or not ref.dataset_id):
        raise ValueError('CSV training requires a registered CSV dataset')

    config = deepcopy(base_config)
    config.connection = Config.Connection(odbc_driver='',
        class_column=class_column, id_column=id_column,
        data_numerical_columns=list(data_columns), data_text_columns=list(text_columns))
    config.data_source = Config.DataSource(
        kind='csv', dataset_id=ref.dataset_id, artifact_id=ref.artifact_id,
        **(csv_format or {}))
    # Never save a CSV model into the legacy SQL model directory. The local
    # scope is not an authentication boundary, but prevents cross-project
    # artifact selection and accidental SQL/CSV filename collisions.
    if model_root is None:
        model_root = Path.cwd() / "output" / "csv_artifacts" / "models"
    root = Path(model_root)
    if not root.is_absolute() or root.is_symlink() or ".." in root.parts:
        raise ValueError('CSV model storage root must be an absolute safe directory')
    try:
        tenant_id = str(UUID(scope.tenant_id))
        project_id = str(UUID(scope.project_id))
    except (ValueError, AttributeError, TypeError):
        raise ValueError('Invalid CSV model storage scope') from None
    model_path = root / tenant_id / project_id / ref.dataset_id
    model_path.mkdir(parents=True, exist_ok=True)
    config.io.model_path = str(model_path)
    config.io.model_name = "csv_model_" + ref.artifact_id
    config.mode.train = True
    config.mode.predict = bool(predict)
    # Training diagnostics do not require prediction of unlabelled rows.
    config.mode.mispredicted = bool(mispredicted)
    config.mail = Config.Mail(notification_email='')
    # Check the resulting runtime configuration, not just its source metadata.
    config.connection.validate()
    config.data_source.validate()
    config.mode.validate()
    config.io.validate()
    config.debug.validate()
    shared_json = config._config_to_json()
    saved = save_csv_model_configuration(
        repository, scope, dataset_artifact_id,
        id_column=id_column, class_column=class_column,
        data_columns=tuple(data_columns), text_columns=tuple(text_columns),
        csv_format=csv_format, shared_json=shared_json)
    return config, saved
