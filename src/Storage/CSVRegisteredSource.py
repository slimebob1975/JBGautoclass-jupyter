"""Shared registered CSV identity and column-role validation at the I/O boundary."""


def registered_csv_mismatch(config, saved):
    """Return the mismatch category, or None, without reading or writing rows.

    Preserve the configured feature order and the opaque registered source
    identity. The caller owns mode/scope checks and its public error message.
    """
    source = getattr(config, 'data_source', None)
    if (getattr(source, 'kind', None) != 'csv'
            or getattr(source, 'dataset_id', None) != saved.dataset_id
            or getattr(source, 'artifact_id', None) != saved.artifact_id):
        return 'identity'
    if (config.get_id_column_name() != saved.id_column
            or config.get_class_column_name() != saved.class_column
            or tuple(config.get_column_names()) !=
            tuple(saved.text_columns) + tuple(saved.data_columns) +
            (saved.id_column, saved.class_column)):
        return 'roles'
    return None
