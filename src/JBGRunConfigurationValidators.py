"""Source-bound validation of existing Config instances for prepared runs (280z).

No new I/O, SQL connection, training, or alternative CSV configuration
format is introduced here. Existing Config.validate methods can normalize settings.
Resolved datasets must come from trusted resolvers.
"""
from __future__ import annotations

from Storage.ArtifactContract import ArtifactKind, ArtifactReference


def _validate_existing_config(configuration):
    from Config import Config
    if not isinstance(configuration, Config):
        raise TypeError('Run configuration must be a backend Config')
    # Reuse the existing backend validation rather than implement new ML rules.
    for component in (configuration.data_source, configuration.connection,
                      configuration.mode, configuration.io, configuration.debug):
        component.validate()


class CSVRunConfigurationValidator:
    """Bind CSV Config identity and column roles to a scoped dataset reference.

    This is not an authorization boundary: the caller must authorize the scope,
    and ScopedCSVDatasetResolver must have verified its repository reference.
    """

    def validate(self, configuration, resolved_dataset):
        if (not isinstance(resolved_dataset, ArtifactReference)
                or resolved_dataset.kind != ArtifactKind.DATASET
                or resolved_dataset.content_type != 'text/csv'
                or not resolved_dataset.dataset_id):
            raise ValueError('Expected a resolved CSV dataset artifact')
        _validate_existing_config(configuration)
        source = configuration.data_source
        if (source.kind != 'csv' or source.dataset_id != resolved_dataset.dataset_id
                or source.artifact_id != resolved_dataset.artifact_id):
            raise ValueError('CSV configuration does not match resolved dataset')
        conn = configuration.connection
        # Existing column-role constraints; reuse rather than reimplement.
        from Storage.CSVModelConfiguration import _roles
        _roles(conn.id_column, conn.class_column,
               conn.data_numerical_columns, conn.data_text_columns)
        # Do not permit a CSV job to inherit SQL connection identity.
        if (conn.host or conn.data_catalog or conn.data_table or
                conn.class_catalog or conn.class_table or
                conn.sql_username or conn.sql_password or
                configuration.mail.notification_email):
            raise ValueError('CSV configuration retains SQL or email identity')


class SQLRunConfigurationValidator:
    """Bind existing SQL Config to the already selected trusted SQL DataLayer.

    Does not create connections or execute queries. Actual SQL permissions and
    connection validity remain the SQL DataLayer's responsibility.
    """

    def validate(self, configuration, resolved_dataset):
        _validate_existing_config(configuration)
        if configuration.data_source.kind != 'sql':
            raise ValueError('Expected SQL configuration')
        # Require a preconfigured layer bound to the same config instance.
        # Reject duck-typed/foreign resolved datasets before use.
        from SQLDataLayer import DataLayer
        if (not isinstance(resolved_dataset, DataLayer)
                or resolved_dataset.config is not configuration):
            raise ValueError('SQL configuration is not bound to resolved DataLayer')
        table = configuration.get_data_table(include_database=True)
        if not isinstance(table, str) or not table.strip():
            raise ValueError('SQL data table is missing')
