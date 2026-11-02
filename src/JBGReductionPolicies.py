"""Pure dimension-reduction parameter policies (no estimator or GUI imports)."""


def choose_pca_components(n_samples: int, n_features: int,
                          n_components: int, variance_target: float = 0.95):
    """Preserve Reduction.get_PCA's historical component selection.

    The dimension arguments remain accepted for compatibility with the original
    nested helper; they have no effect on this specific PCA policy.
    """
    if n_components is not None:
        return int(n_components)

    return float(variance_target)
