"""The pipeline picture in the Configuration tab's left panel (revision 298e).

Requested and agreed on 2026-11-11: below the list of configurations, the
steps of the pipeline in the order the engine runs them (the order of the
form's sections, 298b), each marked from the configuration on show:

- "used": the step runs, with what it does as a short subtitle;
- "alternatives": the run compares several choices (several ticked, or "All");
- "unused": the step does nothing in this configuration.

Scaling, sampling, dimensionality reduction and the algorithm are fitted
together in every cross-validation fold; the picture frames them. Every box
links to its section of the form. Texts go through the interface's
translation, so the picture follows the chosen language.
"""
from __future__ import annotations

USED, ALTERNATIVES, UNUSED = "used", "alternatives", "unused"
CROSS_VALIDATED = ("scaling", "oversampling", "undersampling", "reduction", "algorithms")
# Step -> (list field, the code that means "nothing").
LISTS = {"scaling": ("preprocessor", "NOS"), "oversampling": ("oversamplers", "NOG"),
         "undersampling": ("undersamplers", "NUG"), "reduction": ("feature_selection", "NOR")}


def _percent(value) -> str:
    return f"{float(value or 0) * 100:.0f} %"


def pipeline_steps(schema: dict, values: dict, *, translate, language: str = "sv",
                   dataset_name: str = "", roles=None) -> list[dict]:
    """The steps for ``values`` (the form's values) in pipeline order.

    Each step: key (the form section, or "dataset"), label, state and
    subtitle. ``roles`` (the active training dataset's roles, if known)
    tells whether there is text to process."""
    fields = {field["key"]: field for field in schema["fields"]}
    labels = {group["key"]: group["label"] for group in schema["groups"]}
    mode = (values or {}).get("mode") or {}
    source = schema.get("source", "csv")

    def options(key):
        return {o["value"]: o["label"] for o in fields.get(key, {}).get("options", [])}

    def label_of(key, value):
        return options(key).get(value, str(value))

    def step(key, state, subtitle, label=None):
        return {"key": key, "label": label or labels.get(key, key), "state": state,
                "subtitle": subtitle}

    def choices(key, none_code, name):
        chosen = list(mode.get(name) or [])
        choose_key = f"mode.{name}"
        everything = options(choose_key)
        if "ALL" in chosen:
            return step(key, ALTERNATIVES, f"{translate('All')}, {len(everything)} "
                                           f"{translate('alternatives')}")
        if len(chosen) > 1:
            return step(key, ALTERNATIVES, f"{len(chosen)} {translate('alternatives')}")
        if not chosen or chosen[0] == none_code:
            return step(key, UNUSED, translate("Not used"))
        return step(key, USED, label_of(choose_key, chosen[0]))

    steps = [step("dataset", USED if dataset_name else UNUSED,
                  dataset_name or translate("No active training dataset"),
                  label=translate("Dataset"))]

    text_columns = None if roles is None else roles.by_role("text") + roles.by_role("categorical")
    if text_columns == []:
        steps.append(step("text_processing", UNUSED, translate("No text features")))
    else:
        parts = [label_of("mode.ngram_range", mode.get("ngram_range"))]
        if mode.get("use_categorization"):
            parts.insert(0, translate("categorisation"))
        if mode.get("use_stop_words"):
            parts.append(translate("stop words"))
        if mode.get("hex_encode"):
            parts.append(translate("encryption"))
        text = ", ".join(parts)
        steps.append(step("text_processing", USED, text[:1].upper() + text[1:]))

    steps.append(step("feature_selection", USED,
                      label_of("mode.feature_selection_mode", mode.get("feature_selection_mode"))))

    split = f"{translate('Test share')} {_percent(mode.get('test_size'))}"
    if mode.get("grouped_cross_validation") or mode.get("holdout_split_mode") == "person_disjoint":
        split += f", {translate('groups kept apart')}"
        if roles is not None and roles.group_column:
            split += f" ({roles.group_column})"                   # 298g
    steps.append(step("splitting", USED, split))

    for key in ("scaling", "oversampling", "undersampling", "reduction"):
        choose, none_code = LISTS[key]
        steps.append(choices(key, none_code, choose))

    algorithms = list(mode.get("algorithm") or [])
    metric = label_of("mode.scoring", mode.get("scoring"))
    if "ALL" in algorithms or len(algorithms) > 1:
        count = len(options("mode.algorithm")) if "ALL" in algorithms else len(algorithms)
        steps.append(step("algorithms", ALTERNATIVES,
                          f"{count} {translate('alternatives')}, {metric}"))
    elif algorithms:
        steps.append(step("algorithms", USED,
                          f"{label_of('mode.algorithm', algorithms[0])}, {metric}"))
    else:
        steps.append(step("algorithms", UNUSED, translate("No algorithm chosen")))

    decision = label_of("mode.decision_policy", mode.get("decision_policy"))
    if mode.get("decision_policy") == "fixed":
        threshold = mode.get("probability_threshold")
        decision += f" {str(threshold).replace('.', ',') if language == 'sv' else threshold}"
    steps.append(step("decision", USED, decision))

    steps.append(step("evaluation", USED, translate("Feature importance"))
                 if mode.get("calculate_feature_importance")
                 else step("evaluation", UNUSED, translate("Not used")))

    if mode.get("calculate_dark_numbers"):
        dark = (f"{label_of('mode.dark_number_method', mode.get('dark_number_method'))}, "
                f"alpha {label_of('mode.dark_number_alpha', mode.get('dark_number_alpha'))}")
        if mode.get("dark_number_target"):
            dark += f", {mode['dark_number_target']}"
        steps.append(step("dark_numbers", USED, dark))
    else:
        steps.append(step("dark_numbers", UNUSED, translate("Not used")))

    prediction_label = labels.get("predictions") or translate("Predictions")
    if source == "sql":
        steps.append(step("predictions", USED if mode.get("predict") else UNUSED,
                          translate("Rows without a class") if mode.get("predict")
                          else translate("Not used"), label=prediction_label))
    else:
        # Decision 9: CSV training runs only train; predictions have their own tab.
        steps.append(step("predictions", UNUSED, translate("On the Prediction tab"),
                          label=prediction_label))
    return steps
