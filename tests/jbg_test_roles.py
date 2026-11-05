"""Test helper (293a): store column roles on a dataset directly.

Before 293a, tests put roles into the configuration. Since decision A the
roles belong to the training dataset; this helper converts configuration-style
roles and stores them through the repository, bypassing the column check, so
tests can also store deliberately wrong roles.
"""
from JBGDatasetRoles import DatasetRoles


def roles_from_settings(settings: dict) -> DatasetRoles:
    connection = settings.get("connection", {})
    categorical = list(settings.get("mode", {}).get("category_text_columns") or [])
    mapping = {}
    for name in connection.get("data_numerical_columns") or []:
        mapping[name] = "numeric"
    for name in connection.get("data_text_columns") or []:
        mapping[name] = "categorical" if name in categorical else "text"
    if connection.get("class_column"):
        mapping[connection["class_column"]] = "class"
    if connection.get("id_column"):
        mapping[connection["id_column"]] = "id"
    return DatasetRoles.from_mapping(mapping)


def store_roles(client, project_id: str, dataset_id: str, settings: dict) -> None:
    from JBGIdentity import AuthenticatedPrincipal, AuthenticationMode
    owner = AuthenticatedPrincipal(AuthenticationMode.LOCAL, "developer").principal_key
    client.app.state.jbg["datasets"].set_roles(owner, project_id, dataset_id,
                                               roles_from_settings(settings))
