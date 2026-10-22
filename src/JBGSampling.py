"""Compatible sampling selections; fitted models retain one enum per sampler."""

from collections import Counter


def partial_undersampling_counts(y, majority_multiple):
    """Cap each class at a multiple of the smallest training-fold class.

    Never remove minority rows or request additional rows when a fold is
    already less imbalanced than the requested profile. For multiclass data,
    the same cap applies independently to every class. This top-level callable
    can be cloned/pickled with the pipeline for CV workers and saved models.
    """
    if isinstance(majority_multiple, bool) or not isinstance(majority_multiple, int) or majority_multiple < 1:
        raise ValueError("majority_multiple must be a positive integer")
    counts = Counter(y)
    if not counts:
        raise ValueError("Partial undersampling requires nonempty training labels")
    cap = min(counts.values()) * majority_multiple
    return {label: min(count, cap) for label, count in counts.items()}


def sampling_names(value, enum_type, fallback):
    """Normalize old scalar names/enums and new nonempty selections in stable order."""
    if value is None:
        value = [fallback]
    elif isinstance(value, str):
        value = value.split(",")
    elif isinstance(value, enum_type):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Select at least one {enum_type.__name__} method (including no sampling).")
    names = []
    for item in value:
        try:
            name = item.name if isinstance(item, enum_type) else str(item).strip()
            enum_type[name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Unsupported {enum_type.__name__} method: {item}") from error
        if name not in names:
            names.append(name)
    return names


def categorical_feature_indices(feature_names, categorical_names):
    """Resolve fitted categorical feature names to stable converted-column indices.

    Ordinary JBG conversion turns categorical text into ordinal numeric columns, so
    dtype-based sampler auto-detection cannot recover their semantics later.
    """
    names = list(feature_names)
    if len(names) != len(set(names)):
        raise ValueError("Categorical SMOTE requires unique converted feature names.")

    categories = tuple(categorical_names or ())
    if len(categories) != len(set(categories)):
        raise ValueError("Categorical SMOTE received duplicate fitted categorical feature names.")

    missing = [name for name in categories if name not in names]
    if missing:
        raise ValueError(
            "Categorical SMOTE contract is inconsistent: "
            f"{len(missing)} fitted categorical column(s) are absent after conversion."
        )

    selected = set(categories)
    return tuple(index for index, name in enumerate(names) if name in selected)


def validate_categorical_oversampling(method_name, categorical_indices, n_features):
    """Validate the converted-feature contract for SMOTE-NC or SMOTE-N.

    Returns a normalized tuple of categorical indices. The caller can pass the SMOTE-NC
    result directly to ``categorical_features``. SMOTE-N is accepted only when every
    converted feature is explicitly known to be categorical.
    """
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
        raise ValueError("Categorical SMOTE requires a positive converted feature count.")

    indices = tuple(categorical_indices or ())
    invalid = [
        index for index in indices
        if isinstance(index, bool) or not isinstance(index, int)
        or index < 0 or index >= n_features
    ]
    if invalid or len(set(indices)) != len(indices):
        raise ValueError("Categorical SMOTE received an invalid converted-feature index contract.")

    method_name = str(method_name).upper()
    if method_name == "SNC":
        if not indices:
            raise ValueError(
                "SMOTE-NC requires fitted categorical feature roles; no categorical converted "
                "features were identified. Configure categorical text columns or use a numerical "
                "oversampler."
            )
        if len(indices) >= n_features:
            raise ValueError(
                "SMOTE-NC requires both categorical and non-categorical converted features; "
                "use SMOTE-N only for an all-categorical representation."
            )
    elif method_name == "SNN":
        if tuple(sorted(indices)) != tuple(range(n_features)):
            raise ValueError(
                "SMOTE-N requires every converted predictor to be categorical. The fitted "
                "representation contains non-categorical features; use SMOTE-NC or another "
                "oversampler."
            )
    else:
        raise ValueError(f"Unsupported categorical oversampler contract: {method_name}")

    return indices
