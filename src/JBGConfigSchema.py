"""Schema of the user-owned configuration settings, for generated forms (290a).

Agreed 2026-11-05:
1. Option values are derived from code wherever they exist there (JBGMeta code
   lists, the Dark Numbers enums in Config, JBGModelSettings.REGISTRY, the
   feature-search presets). About a dozen choice fields are validated with
   inline literals in Config or at run time; their values are declared below
   and a test checks them against that code, so Config.py stays untouched.
2. Grouping, labels and help texts for every user-owned field live in FIELDS;
   a test fails if a configuration field has no entry. Swedish texts come from
   the interface catalogue (src/web/i18n/sv.po), with its completeness test.
3. Labels follow the user's language; technical option names (algorithm and
   method names, SMOTE, PCA, ...) are never translated.
4. The schema guides the form only; the Config loader validates every save.
"""
from __future__ import annotations

from typing import Callable

# Group key -> English label (in display order). The GUI decides on which tab
# each group appears (e.g. column roles on the Dataset tab, run type on Run).
GROUPS = (
    ("run_type", "Run type"),
    ("roles", "Column roles"),
    ("preprocessing", "Preprocessing"),
    ("feature_modelling", "Feature modelling"),
    ("sampling", "Sampling"),
    ("algorithms", "Algorithms"),
    ("validation", "Validation & decision"),
    ("dark_numbers", "Dark Numbers"),
    ("predictions", "Predictions"),
    ("advanced", "Advanced"),
)

# Values declared here because Config (or the run) checks them inline.
INLINE_OPTIONS = {
    "mode.feature_selection_mode": ("manual", "assisted", "reuse"),
    "mode.feature_search_model_type": ("Linear", "Non-linear", "Tree", "Combination", "CatBoost"),
    "mode.feature_search_tree_estimator": ("Extra Trees", "Random Forest"),
    "mode.feature_search_tree_class_weight": ("None", "balanced", "balanced_subsample"),
    "mode.feature_search_catboost_class_weight": ("None", "Balanced"),
    "mode.feature_search_reduction": ("NOR", "PCA"),
    "mode.feature_search_oversampling": ("NOG", "RND", "SME"),
    "mode.feature_search_scaler": ("STA", "MAX"),
    "mode.decision_policy": ("default", "fixed"),            # checked at run time
    "mode.holdout_split_mode": ("stratified_rows", "person_disjoint"),   # run time
}
# Words among those values that are interface text, not technical names.
OPTION_LABELS = {
    "manual": "Manual", "assisted": "Assisted search", "reuse": "Reuse a saved result",
    "default": "Default (most likely class)", "fixed": "Fixed probability threshold",
    "stratified_rows": "Stratified rows", "person_disjoint": "Groups kept apart",
    "None": "None",
}


def _meta(name: str) -> Callable[[], list]:
    def options():
        import JBGMeta
        return [(member.name, member.full_name) for member in getattr(JBGMeta, name)]
    return options


def _dark(name: str) -> Callable[[], list]:
    def options():
        import Config
        return [(member.name, member.display_name) for member in getattr(Config, name)]
    return options


def _presets() -> list:
    from JBGFeatureSearchPresets import PRESETS
    return [(name, name) for name in (*PRESETS, "Custom")]


def _inline(key: str) -> Callable[[], list]:
    return lambda: [(value, OPTION_LABELS.get(value, value)) for value in INLINE_OPTIONS[key]]


# key -> (group, kind, label, help, extras). Kinds: boolean, integer, number,
# text, choice, multi_choice, column, columns, integer_list, parameters.
F = {}


def _f(key, group, kind, label, help=None, **extras):
    F[key] = {"group": group, "kind": kind, "label": label, "help": help, **extras}


_f("mode.train", "run_type", "boolean", "Train a model")
_f("mode.predict", "run_type", "boolean", "Predict unlabelled rows")
_f("mode.mispredicted", "run_type", "boolean", "Show mispredictions")
# 293c: column roles (and the derived excluded columns) belong to the dataset
# and are set on the Dataset tab; the group column is a splitting setting.
_f("mode.validation_group_column", "validation", "text", "Group column",
   "Rows of one group stay together when data is split.")
_f("mode.preprocessor", "preprocessing", "multi_choice", "Scalers", options=_meta("Preprocess"),
   allows_all=True)
_f("mode.use_metas", "preprocessing", "boolean", "Pass on metadata")
_f("mode.use_stop_words", "preprocessing", "boolean", "Filter stop words")
_f("mode.ngram_range", "preprocessing", "choice", "N-grams", options=_meta("NgramRange"))
_f("mode.hex_encode", "preprocessing", "boolean", "Encrypt text features")
_f("mode.use_categorization", "preprocessing", "boolean", "Categorize text columns")
_f("mode.feature_selection", "feature_modelling", "multi_choice", "Dimensionality reduction",
   options=_meta("Reduction"), allows_all=True)
_f("mode.num_selected_features", "feature_modelling", "integer", "Number of features",
   "Empty selects automatically.", nullable=True, minimum=1)
_f("mode.feature_selection_mode", "feature_modelling", "choice", "Feature selection mode",
   options=_inline("mode.feature_selection_mode"))
_f("mode.feature_selection_profile_id", "feature_modelling", "text", "Saved feature profile",
   advanced=True)
_INVESTIGATION = (
    ("preset", "choice", "Search preset", _presets),
    ("model_type", "choice", "Model type", _inline("mode.feature_search_model_type")),
    ("scaler", "choice", "Scaler", _inline("mode.feature_search_scaler")),
    ("oversampling", "choice", "Oversampling", _inline("mode.feature_search_oversampling")),
    ("reduction", "choice", "Reduction", _inline("mode.feature_search_reduction")),
    ("trials", "integer", "Trials", None), ("sizes", "integer_list", "Subset sizes", None),
    ("inner_folds", "integer", "Inner folds", None), ("outer_folds", "integer", "Outer folds", None),
    ("top_sets", "integer", "Top sets", None), ("refine_rounds", "integer", "Refinement rounds", None),
    ("max_evaluations", "integer", "Maximum evaluations", None),
    ("min_gain", "number", "Minimum gain", None),
    ("linear_c", "number", "Linear: inverse regularization (C)", None),
    ("linear_max_iterations", "integer", "Linear: maximum iterations", None),
    ("non_linear_max_iterations", "integer", "Non-linear: maximum iterations", None),
    ("non_linear_alpha", "number", "Non-linear: alpha", None),
    ("non_linear_hidden_layers", "integer_list", "Non-linear: hidden layers", None),
    ("non_linear_learning_rate", "number", "Non-linear: learning rate", None),
    ("tree_estimator", "choice", "Tree: estimator", _inline("mode.feature_search_tree_estimator")),
    ("tree_class_weight", "choice", "Tree: class weighting",
     _inline("mode.feature_search_tree_class_weight")),
    ("tree_positive_threshold", "number", "Tree: positive threshold", None),
    ("tree_n_estimators", "integer", "Tree: number of trees", None),
    ("tree_max_depth", "integer", "Tree: maximum depth", None),
    ("tree_min_samples_leaf", "integer", "Tree: minimum samples per leaf", None),
    ("catboost_class_weight", "choice", "CatBoost: class weighting",
     _inline("mode.feature_search_catboost_class_weight")),
    ("catboost_iterations", "integer", "CatBoost: iterations", None),
)
for _name, _kind, _label, _options in _INVESTIGATION:
    _extras = {"subgroup": "Feature investigation"}
    if _options is not None:
        _extras["options"] = _options
    _f(f"mode.feature_search_{_name}", "feature_modelling", _kind, _label, **_extras)
_f("mode.oversamplers", "sampling", "multi_choice", "Oversampling methods", nullable=True,
   options=_meta("Oversampling"), allows_all=True)
_f("mode.undersamplers", "sampling", "multi_choice", "Undersampling methods", nullable=True,
   options=_meta("Undersampling"), allows_all=True)
_f("mode.oversampler", "sampling", "choice", "Oversampling (single, legacy)", advanced=True,
   options=_meta("Oversampling"))
_f("mode.undersampler", "sampling", "choice", "Undersampling (single, legacy)", advanced=True,
   options=_meta("Undersampling"))
_f("mode.algorithm", "algorithms", "multi_choice", "Algorithms", options=_meta("Algorithm"),
   allows_all=True)
_f("mode.model_parameters", "algorithms", "parameters", "Algorithm settings",
   "Settings per selected algorithm; empty uses the defaults.")
_f("mode.scoring", "algorithms", "choice", "Scoring metric", options=_meta("ScoreMetric"))
_f("mode.max_iterations", "algorithms", "integer", "Maximum iterations", minimum=1)
_f("mode.test_size", "validation", "number", "Holdout share", "Share of rows kept for testing.",
   minimum=0.0, maximum=1.0)
_f("mode.holdout_split_mode", "validation", "choice", "Holdout split",
   options=_inline("mode.holdout_split_mode"))
_f("mode.grouped_cross_validation", "validation", "boolean", "Keep groups apart in cross-validation")
_f("mode.decision_policy", "validation", "choice", "Decision policy",
   options=_inline("mode.decision_policy"))
_f("mode.probability_threshold", "validation", "number", "Probability threshold",
   "Used with the fixed decision policy (binary classes).", minimum=0.0, maximum=1.0)
_f("mode.threshold_diagnostics", "validation", "boolean", "Threshold diagnostics")
_f("mode.model_benchmark_150", "validation", "boolean", "Out-of-fold benchmark", advanced=True)
_f("mode.calculate_feature_importance", "validation", "boolean", "Feature importance")
_f("mode.feature_importance_repeats", "validation", "integer", "Feature importance repeats",
   minimum=2, maximum=30)
_f("mode.calculate_dark_numbers", "dark_numbers", "boolean", "Estimate Dark Numbers")
_f("mode.dark_number_method", "dark_numbers", "choice", "Method", options=_dark("DarkNumberMethod"))
_f("mode.dark_number_alpha", "dark_numbers", "choice", "Alpha", options=_dark("DarkNumberAlpha"))
_f("mode.dark_number_target", "dark_numbers", "text", "Target class")
_f("mode.dark_number_flip_fraction", "dark_numbers", "number", "Flip fraction",
   minimum=0.0, maximum=1.0)
_f("mode.experimental_perturbed_dark_number_fallback", "dark_numbers", "boolean",
   "Experimental fallback", advanced=True)
_f("connection.class_catalog", "predictions", "text", "Prediction catalog",
   "Empty uses the server's default.", sources=("sql",))
_f("connection.class_table", "predictions", "text", "Prediction table",
   "Empty uses the server's default (schema.table).", sources=("sql",))
_f("io.verbose", "advanced", "boolean", "Detailed log")
_f("debug.on", "advanced", "boolean", "Debug mode", advanced=True)
_f("debug.data_limit", "advanced", "integer", "Row limit", "Empty reads all rows.",
   nullable=True, minimum=1)
_f("mail.notification_email", "advanced", "text", "Notification e-mail")
FIELDS = F


def _translate_options(options, translate):
    # Only interface words are translated; technical names stay as they are.
    return [{"value": value, "label": translate(label) if label in OPTION_LABELS.values()
             else label} for value, label in options]


def configuration_schema(source: str, translate: Callable[[str], str], defaults: dict) -> dict:
    """The form schema for ``source`` ("sql" or "csv"), labels via ``translate``."""
    from JBGModelSettings import REGISTRY
    if source not in ("sql", "csv"):
        raise ValueError("source must be 'sql' or 'csv'")
    fields = []
    for key, entry in FIELDS.items():
        if source not in entry.get("sources", ("sql", "csv")):
            continue
        section, name = key.split(".", 1)
        field = {"key": key, "group": entry["group"], "kind": entry["kind"],
                 "label": translate(entry["label"]),
                 "help": translate(entry["help"]) if entry["help"] else None,
                 "default": defaults[section][name],
                 "nullable": entry.get("nullable", False),
                 "advanced": entry.get("advanced", False)}
        if entry.get("allows_all"):
            field["allows_all"] = True                   # 293c: an "All" (ALL) choice
        for extra in ("minimum", "maximum"):
            if extra in entry:
                field[extra] = entry[extra]
        if "subgroup" in entry:
            field["subgroup"] = translate(entry["subgroup"])
        if "options" in entry:
            field["options"] = _translate_options(entry["options"](), translate)
        fields.append(field)
    used = {field["group"] for field in fields}
    parameters = {
        algorithm: [{"name": name, "label": translate(p.label), "kind": p.kind,
                     "default": p.default, "minimum": p.minimum, "maximum": p.maximum,
                     "options": list(p.options)} for name, p in settings.items()]
        for algorithm, settings in REGISTRY.items()}
    return {"schema_version": 1, "source": source,
            "groups": [{"key": key, "label": translate(label)} for key, label in GROUPS
                       if key in used],
            "fields": fields, "model_parameters": parameters}


def interface_texts() -> set[str]:
    """Every text this schema shows, for the translation completeness test."""
    from JBGModelSettings import REGISTRY
    texts = {label for _, label in GROUPS} | set(OPTION_LABELS.values())
    for entry in FIELDS.values():
        texts.add(entry["label"])
        if entry["help"]:
            texts.add(entry["help"])
        if "subgroup" in entry:
            texts.add(entry["subgroup"])
    texts |= {p.label for settings in REGISTRY.values() for p in settings.values()}
    return texts
