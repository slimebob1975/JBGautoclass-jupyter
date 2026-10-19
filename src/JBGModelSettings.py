"""Declarative ordinary-classifier parameter settings (revision 145)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    label: str
    kind: str
    default: object
    minimum: object = None
    maximum: object = None
    options: tuple = ()


REGISTRY = {
    "CBST": {
        "iterations": Parameter("Iterations", "int", 400, 50, 3000),
        "depth": Parameter("Tree depth", "int", 6, 2, 12),
        "learning_rate": Parameter("Learning rate", "float", 0.05, 0.001, 0.3),
        "class_weight": Parameter("Class weighting", "choice", "None", options=("None", "Balanced")),
    },
    "RFCL": {
        "n_estimators": Parameter("Trees", "int", 100, 10, 2000),
        "min_samples_leaf": Parameter("Minimum samples per leaf", "int", 1, 1, 100),
    },
}

# Expose estimator-level settings only when set_params maps directly to the
# ordinary JBG estimator. Composite/optional-backend classifiers deliberately
# retain their existing configuration until explicit routing is implemented.
REGISTRY.update({
    "LRN": {
        "C": Parameter("Inverse regularization (C)", "float", 1.0, 0.0001, 1000.0),
        "class_weight": Parameter("Class weighting", "choice", "None", options=("None", "balanced")),
    },
    "KNN": {
        "n_neighbors": Parameter("Neighbors", "int", 5, 1, 500),
        "weights": Parameter("Neighbor weighting", "choice", "uniform", options=("uniform", "distance")),
    },
    "RADN": {
        "radius": Parameter("Neighbor radius", "float", 1.0, 0.001, 10000.0),
        "weights": Parameter("Neighbor weighting", "choice", "uniform", options=("uniform", "distance")),
    },
    "DTC": {
        "min_samples_leaf": Parameter("Minimum samples per leaf", "int", 1, 1, 500),
        "criterion": Parameter("Split criterion", "choice", "gini", options=("gini", "entropy", "log_loss")),
    },
    "GNB": {"var_smoothing": Parameter("Variance smoothing", "float", 1e-9, 1e-12, 0.1)},
    "MNB": {"alpha": Parameter("Smoothing alpha", "float", 1.0, 0.00001, 100.0)},
    "BNB": {"alpha": Parameter("Smoothing alpha", "float", 1.0, 0.00001, 100.0)},
    "CNB": {"alpha": Parameter("Smoothing alpha", "float", 1.0, 0.00001, 100.0)},
    "REC": {"alpha": Parameter("Regularization alpha", "float", 1.0, 0.00001, 1000.0)},
    "PCN": {"alpha": Parameter("Regularization alpha", "float", 0.0001, 0.0000001, 1.0)},
    "PAC": {"alpha": Parameter("Regularization alpha", "float", 0.0001, 0.0000001, 1.0)},
    "LSVC": {"C": Parameter("Inverse regularization (C)", "float", 1.0, 0.0001, 1000.0)},
    "SVC": {"C": Parameter("Inverse regularization (C)", "float", 1.0, 0.0001, 1000.0)},
    "SGDE": {"alpha": Parameter("Regularization alpha", "float", 0.0001, 0.0000001, 1.0)},
    "LDA": {"solver": Parameter("Solver", "choice", "svd", options=("svd", "lsqr"))},
    "QDA": {"reg_param": Parameter("Covariance regularization", "float", 0.1, 0.0, 1.0)},
    "BGC": {"n_estimators": Parameter("Ensemble estimators", "int", 10, 1, 2000)},
    "ETC": {"n_estimators": Parameter("Trees", "int", 100, 10, 2000),
            "min_samples_leaf": Parameter("Minimum samples per leaf", "int", 1, 1, 500)},
    "ABC": {"n_estimators": Parameter("Boosting estimators", "int", 50, 1, 2000),
            "learning_rate": Parameter("Learning rate", "float", 1.0, 0.001, 10.0)},
    "GBC": {"n_estimators": Parameter("Boosting estimators", "int", 100, 10, 2000),
            "learning_rate": Parameter("Learning rate", "float", 0.1, 0.001, 1.0)},
    "HIST": {"learning_rate": Parameter("Learning rate", "float", 0.1, 0.001, 1.0),
             "max_iter": Parameter("Boosting iterations", "int", 100, 10, 2000)},
    "MLPC": {"alpha": Parameter("Regularization alpha", "float", 0.0001, 0.0000001, 1.0)},
    "MLP2": {"alpha": Parameter("Regularization alpha", "float", 0.0001, 0.0000001, 1.0)},
    "GPC": {"n_restarts_optimizer": Parameter("Optimizer restarts", "int", 0, 0, 20)},
})



def defaults():
    return {algorithm: {key: spec.default for key, spec in specs.items()}
            for algorithm, specs in REGISTRY.items()}


def validate(overrides):
    """Return normalized overrides, rejecting unknown algorithms/keys and invalid ranges."""
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("Model settings must be a mapping")
    result = {}
    for algorithm, params in overrides.items():
        if algorithm not in REGISTRY or not isinstance(params, dict):
            raise ValueError(f"Unknown model settings group: {algorithm}")
        result[algorithm] = {}
        for key, value in params.items():
            if key not in REGISTRY[algorithm]:
                raise ValueError(f"Unknown parameter: {algorithm}.{key}")
            spec = REGISTRY[algorithm][key]
            if spec.kind == "choice":
                if value not in spec.options:
                    raise ValueError(f"Invalid choice: {algorithm}.{key}")
            elif spec.kind == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"Expected integer: {algorithm}.{key}")
            elif spec.kind == "float":
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise ValueError(f"Expected number: {algorithm}.{key}")
                value = float(value)
            if spec.kind != "choice" and not (spec.minimum <= value <= spec.maximum):
                raise ValueError(f"Parameter outside range: {algorithm}.{key}")
            result[algorithm][key] = value
    return result


def configure_estimator(algorithm, estimator, settings):
    """Apply registered settings without modifying other algorithms."""
    overrides = validate(settings).get(algorithm, {})
    if overrides:
        if algorithm == "LRN" and overrides.get("class_weight") == "None":
            overrides = dict(overrides, class_weight=None)
        estimator.set_params(**overrides)
    return estimator
