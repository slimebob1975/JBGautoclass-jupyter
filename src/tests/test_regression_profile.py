from pathlib import Path

import ipywidgets

from GUI.Widgets import Widgets


class MockDataLayer:
    def get_trained_models_from_files(self, *args, **kwargs):
        return ["config-save.sav", "model-save.sav"]


class MockGUIHandler:
    def __init__(self):
        self.classifier_calls = []
        self.suite_calls = []
        self.class_distribution_calls = []

    @property
    def datalayer(self):
        return MockDataLayer()

    def run_classifier(self, config_params, output):
        self.classifier_calls.append(config_params)

    def run_regression_suite(self, base_config_params, output):
        self.suite_calls.append(base_config_params)

    def get_class_distribution(self, data_settings, current_class):
        self.class_distribution_calls.append((data_settings, current_class))
        raise AssertionError("Regression suite must not query manual class distribution")


def make_widgets() -> Widgets:
    src_path = Path(__file__).parents[1] / "JBGclassification"
    model_path = Path(__file__).parent / "fixtures"
    return Widgets(src_path=src_path, GUIhandler=MockGUIHandler(), model_path=model_path)


def test_regression_test_profile_values():
    widgets = make_widgets()
    widgets.apply_regression_test_profile()

    assert widgets.algorithm_dropdown.value == (
        "LRN", "RFCL", "LSVC", "GNB", "KNN", "LDA",
        "DTC", "REC", "PAC", "SGDE", "QDA", "SVC", "HIST", "NCT", "BGC",
    )
    assert widgets.preprocess_dropdown.value == ("NOS", "STA", "MIX", "MAX")
    assert widgets.reduction_dropdown.value == ("NOR", "PCA", "RFE")
    assert widgets.scoremetric_dropdown.value == "balanced_accuracy"
    assert widgets.oversampler_dropdown.value == "NOG"
    assert widgets.undersampler_dropdown.value == "NUG"
    assert widgets.testdata_slider.value == 20
    assert widgets.iterations_slider.value == 20000
    assert widgets.encryption_checkbox.value is True
    assert widgets.categorize_checkbox.value is True
    assert widgets.categorize_columns.value == ()
    assert widgets.filter_checkbox.value is False
    assert widgets.ngram_range_dropdown.value == "UNI_GRAM"
    assert widgets.show_info_checkbox.value is True


def test_regression_test_profile_button_is_temporary_continue_button():
    widgets = make_widgets()
    button = widgets.test_profile_button

    assert isinstance(button, ipywidgets.Button)
    assert button.disabled is True
    assert button.description == "Regr. test"
    assert button.icon == "flask"


def test_regression_test_profile_button_shares_continue_row():
    widgets = make_widgets()
    form = widgets.continuation_form()

    assert isinstance(form, ipywidgets.HBox)
    assert tuple(form.children) == (
        widgets.continuation_button,
        widgets.test_profile_button,
    )
    assert form.layout.justify_content == "space-between"
    assert form.layout.width == "100%"


def test_regression_suite_has_connection_level_form():
    widgets = make_widgets()
    form = widgets.regression_suite_form()

    assert isinstance(form, ipywidgets.HBox)
    assert tuple(form.children) == (widgets.regression_suite_button,)
    assert form.layout.justify_content == "flex-end"
    assert form.layout.width == "100%"


def test_regression_suite_button_is_available_as_separate_action():
    widgets = make_widgets()
    button = widgets.regression_suite_button

    assert isinstance(button, ipywidgets.Button)
    assert button.disabled is True
    assert button.description == "Regr. suite"
    assert button.icon == "tasks"


def test_regression_suite_becomes_ready_from_connection_without_dataset_selection():
    widgets = make_widgets()
    widgets.sql_username.value = "test-user"
    widgets.sql_password.value = "test-password"
    widgets.data_catalogs_dropdown.options = ("", "CatalogA", "CatalogB")
    widgets.data_catalogs_dropdown.disabled = False

    widgets.update_regression_suite_ready()

    assert widgets.regression_suite_button.disabled is False
    assert widgets.continuation_button.disabled is True
    assert widgets.class_column.value == "N/A"
    assert widgets.id_column.value == "N/A"
    assert widgets.data_columns.value == ()


def test_regression_suite_button_does_not_require_manual_dataset_selection():
    widgets = make_widgets()

    widgets.eventhandler.regression_suite_button_was_clicked(widgets.regression_suite_button)

    assert widgets.regression_suite_state is True
    assert widgets.train_checkbox.value is True
    assert widgets.predict_checkbox.value is False
    assert widgets.mispredicted_checkbox.value is False
    assert widgets.guihandler.class_distribution_calls == []
    assert widgets.start_button.disabled is False
    assert widgets.start_button.description == "Run suite"
    assert widgets.algorithm_dropdown.value == (
        "LRN", "RFCL", "LSVC", "GNB", "KNN", "LDA",
        "DTC", "REC", "PAC", "SGDE", "QDA", "SVC", "HIST", "NCT", "BGC",
    )


def test_regression_suite_base_config_forces_training_and_disables_reclassification():
    widgets = make_widgets()
    widgets.eventhandler.regression_suite_button_was_clicked(widgets.regression_suite_button)

    params = widgets.get_regression_suite_base_config_params()

    assert params["mode"].train is True
    assert params["mode"].predict is False
    assert params["mode"].mispredicted is False
    assert params["mode"].use_metas is False
    assert params["connection"].data_catalog == ""
    assert params["connection"].data_table == ""
    assert params["connection"].class_column == ""
    assert params["connection"].id_column == ""


def test_regression_suite_rerun_does_not_query_manual_class_distribution():
    widgets = make_widgets()
    widgets.regression_suite_state = True
    widgets.rerun_state = True
    widgets.get_regression_suite_base_config_params = lambda: {"sentinel": "suite config"}

    widgets.start_button_actions()

    assert widgets.guihandler.class_distribution_calls == []
    assert widgets.guihandler.suite_calls == [{"sentinel": "suite config"}]


def test_start_button_routes_suite_mode_to_suite_runner():
    widgets = make_widgets()
    params = {"sentinel": "suite config"}
    widgets.get_regression_suite_base_config_params = lambda: params
    widgets.regression_suite_state = True

    widgets.start_button_actions()

    assert widgets.guihandler.suite_calls == [params]
    assert widgets.guihandler.classifier_calls == []


def test_suite_rerun_label_is_distinct():
    widgets = make_widgets()
    widgets.regression_suite_state = True

    widgets.set_rerun()

    assert widgets.start_button.description == "Rerun suite"
    assert widgets.start_button.tooltip == "Rerun the regression suite"
