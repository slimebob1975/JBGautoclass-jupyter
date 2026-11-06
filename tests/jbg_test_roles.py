"""Test helper (293a): store column roles on a dataset directly.

Before 293a, tests put roles into the configuration. Since decision A the
roles belong to the training dataset; this helper converts configuration-style
roles and stores them through the repository, bypassing the column check, so
tests can also store deliberately wrong roles.
"""
from JBGDatasetRoles import DatasetRoles


def roles_from_settings(settings: dict) -> DatasetRoles:
    from JBGDatasetRoles import roles_from_configuration
    return DatasetRoles.from_mapping(roles_from_configuration(settings))


def store_roles(client, project_id: str, dataset_id: str, settings: dict) -> None:
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    client.app.state.jbg["datasets"].set_roles(owner, project_id, dataset_id,
                                               roles_from_settings(settings))
