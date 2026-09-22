from GUIHandler import GUIHandler


def test_regression_suite_includes_wine_fixture():
    datasets = {dataset["label"]: dataset for dataset in GUIHandler.REGRESSION_SUITE_DATASETS}

    assert tuple(datasets) == ("Iris", "Breast Cancer", "Wine")
    assert datasets["Wine"] == {
        "label": "Wine",
        "slug": "wine",
        "table_name": "wine",
        "class_column": "class",
        "id_column": "id",
    }


def test_regression_suite_dataset_labels_follow_fixture_list():
    assert GUIHandler._regression_suite_dataset_labels() == "Iris, Breast Cancer, Wine"


def test_regression_suite_sampling_profiles_target_breast_cancer():
    profiles = GUIHandler.REGRESSION_SUITE_PROFILES

    assert tuple(profile["label"] for profile in profiles) == (
        "Breast Cancer / Random oversampling",
        "Breast Cancer / Random undersampling",
    )
    assert all(profile["dataset_slug"] == "breast_cancer" for profile in profiles)
    assert tuple((profile["oversampler"], profile["undersampler"]) for profile in profiles) == (
        ("RND", "NUG"),
        ("NOG", "RND"),
    )
    assert all(profile["scoring"] == "f1_macro" for profile in profiles)
    assert all(profile["algorithms"] == GUIHandler.REGRESSION_SUITE_SAMPLING_ALGORITHMS for profile in profiles)
    assert all(profile["preprocessors"] == ("NOS", "STA") for profile in profiles)
    assert all(profile["reductions"] == ("NOR", "PCA") for profile in profiles)


def test_regression_suite_sampling_profile_overrides_only_targeted_training_settings():
    from types import SimpleNamespace

    handler = GUIHandler.__new__(GUIHandler)
    base = {
        "connection": SimpleNamespace(
            data_catalog="",
            data_table="",
            class_column="",
            id_column="",
            data_text_columns=[],
            data_numerical_columns=[],
        ),
        "mode": SimpleNamespace(
            category_text_columns=["stale"],
            oversampler=None,
            undersampler=None,
            scoring=None,
            algorithm=None,
            preprocessor=None,
            feature_selection=None,
        ),
        "debug": SimpleNamespace(data_limit=0),
        "io": SimpleNamespace(model_name="test_suite"),
        "name": "test_suite",
    }
    dataset = {
        "label": "Breast Cancer",
        "slug": "breast_cancer",
        "table_name": "breast_cancer",
        "class_column": "diagnosis",
        "id_column": "id",
        "catalog": "AIdata",
        "table": "jbg_ai.breast_cancer",
        "text_columns": [],
        "numerical_columns": ["radius_mean", "texture_mean"],
        "row_count": 569,
    }
    profile = GUIHandler.REGRESSION_SUITE_PROFILES[0]

    config = handler._build_regression_suite_config(base, dataset, profile=profile)

    assert config["connection"].data_catalog == "AIdata"
    assert config["connection"].data_table == "jbg_ai.breast_cancer"
    assert config["connection"].class_column == "diagnosis"
    assert config["connection"].id_column == "id"
    assert config["debug"].data_limit == 569
    assert config["mode"].category_text_columns == []
    assert config["mode"].oversampler.name == "RND"
    assert config["mode"].undersampler.name == "NUG"
    assert config["mode"].scoring.name == "f1_macro"
    assert config["mode"].algorithm.get_abbreviations() == list(GUIHandler.REGRESSION_SUITE_SAMPLING_ALGORITHMS)
    assert config["mode"].preprocessor.get_abbreviations() == ["NOS", "STA"]
    assert config["mode"].feature_selection.get_abbreviations() == ["NOR", "PCA"]
    assert config["io"].model_name == "test_suite_breast_cancer_random_oversampling"
    assert config["name"] == "test_suite_breast_cancer_random_oversampling"

    # The suite builder must not mutate the broad base profile used by the
    # established Iris/Breast Cancer/Wine runs.
    assert base["mode"].category_text_columns == ["stale"]
    assert base["mode"].oversampler is None
    assert base["io"].model_name == "test_suite"
    assert base["name"] == "test_suite"


def test_regression_suite_appends_sampling_profiles_after_base_datasets():
    from types import MethodType, SimpleNamespace

    handler = GUIHandler.__new__(GUIHandler)

    def fake_find(self, dataset, preferred_catalog):
        return {
            **dataset,
            "catalog": "AIdata",
            "table": f"jbg_ai.{dataset['table_name']}",
            "text_columns": [],
            "numerical_columns": ["feature"],
            "row_count": 100,
        }

    handler._find_regression_suite_dataset = MethodType(fake_find, handler)
    base = {
        "connection": SimpleNamespace(
            data_catalog="AIdata",
            data_table="",
            class_column="",
            id_column="",
            data_text_columns=[],
            data_numerical_columns=[],
        ),
        "mode": SimpleNamespace(
            category_text_columns=[],
            oversampler=None,
            undersampler=None,
            scoring=None,
            algorithm=None,
            preprocessor=None,
            feature_selection=None,
        ),
        "debug": SimpleNamespace(data_limit=0),
        "io": SimpleNamespace(model_name="test_suite"),
        "name": "test_suite",
    }

    configs, missing = handler.get_regression_suite_configs(base)

    assert missing == []
    assert [label for label, _ in configs] == [
        "Iris",
        "Breast Cancer",
        "Wine",
        "Breast Cancer / Random oversampling",
        "Breast Cancer / Random undersampling",
    ]
    assert [config["io"].model_name for _, config in configs] == [
        "test_suite_iris",
        "test_suite_breast_cancer",
        "test_suite_wine",
        "test_suite_breast_cancer_random_oversampling",
        "test_suite_breast_cancer_random_undersampling",
    ]
