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
