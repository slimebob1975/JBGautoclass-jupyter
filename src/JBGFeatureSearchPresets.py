"""Deterministic investigation budgets; estimator hyperparameters are independent."""
from dataclasses import dataclass

PRESETS = {
    "Quick": dict(trials=4, sizes=(5, 10, 20), inner_folds=3, outer_folds=3,
                  top_sets=1, refine_rounds=0, max_evaluations=5, min_gain=0.005),
    "Standard": dict(trials=30, sizes=(10, 20, 40, 60, 80), inner_folds=3, outer_folds=3,
                     top_sets=2, refine_rounds=1, max_evaluations=120, min_gain=0.005),
    "Extended": dict(trials=80, sizes=(5, 10, 20, 40, 60, 80, 90), inner_folds=5, outer_folds=3,
                     top_sets=3, refine_rounds=2, max_evaluations=300, min_gain=0.005),
}
SEARCH_FIELDS = tuple(PRESETS["Quick"])


def preset_values(name):
    if name not in PRESETS:
        raise ValueError(f"Unknown investigation preset: {name}")
    return dict(PRESETS[name])


def detected_preset(settings):
    """A manually edited configuration is Custom, never falsely labelled a preset."""
    for name, values in PRESETS.items():
        if all(tuple(settings[key]) == tuple(value) if key == "sizes" else settings[key] == value
               for key, value in values.items()):
            return name
    return "Custom"
