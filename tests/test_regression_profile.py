from pathlib import Path
import json
from types import SimpleNamespace

import ipywidgets

from Config import Config, DarkNumberAlpha, DarkNumberMethod
from GUI.Widgets import Widgets
from JBGMeta import (
    AlgorithmTuple, NgramRange, Oversampling, PreprocessTuple, ReductionTuple,
    ScoreMetric, Undersampling,
)


class MockDataLayer:
    def __init__(self):
        self.config = SimpleNamespace(
            connection=SimpleNamespace(sql_username="", sql_password="")
        )

    def get_catalogs_as_options(self):
        return ["", "CatalogA", "CatalogB"]

    def get_trained_models_from_files(self, *args, **kwargs):
        return ["config-save.sav", "model-save.sav"]


class MockGUIHandler:
    def __init__(self):
        self.classifier_calls = []
        self.suite_calls = []
        self.repeat_calls = []
        self.class_distribution_calls = []
        self.saved_run_available = False

    @property
    def datalayer(self):
        return MockDataLayer()

    def run_classifier(self, config_params, output):
        self.classifier_calls.append(config_params)

    def run_regression_suite(self, base_config_params, output):
        self.suite_calls.append(base_config_params)

    def has_saved_classifier_run(self):
        return self.saved_run_available

    def repeat_last_classifier_run(self, output, sql_username, sql_password):
        self.repeat_calls.append((sql_username, sql_password))

    def get_class_distribution(self, data_settings, current_class):
        self.class_distribution_calls.append((data_settings, current_class))
        raise AssertionError("Regression suite must not query manual class distribution")


def make_widgets() -> Widgets:
    src_path = Path(__file__).parents[1] / "src" / "JBGclassification"
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
    row = form.children[1]

    assert isinstance(form, ipywidgets.VBox)
    assert isinstance(row, ipywidgets.HBox)
    assert tuple(row.children) == (
        widgets.continuation_button,
        widgets.test_profile_button,
    )
    assert row.layout.justify_content == "space-between"
    assert row.layout.width == "100%"


def test_regression_suite_has_connection_level_form():
    widgets = make_widgets()
    form = widgets.regression_suite_form()
    row = form.children[1]

    assert isinstance(form, ipywidgets.VBox)
    assert isinstance(row, ipywidgets.HBox)
    assert tuple(row.children) == (
        widgets.repeat_last_run_button,
        widgets.regression_suite_button,
    )
    assert row.layout.justify_content == "space-between"
    assert row.layout.width == "100%"


def test_regression_suite_button_is_available_as_separate_action():
    widgets = make_widgets()
    button = widgets.regression_suite_button

    assert isinstance(button, ipywidgets.Button)
    assert button.disabled is True
    assert button.description == "Regr. suite"
    assert button.icon == "tasks"
    assert button.tooltip == "Run the configured regression suite without selecting a dataset first"


def test_repeat_last_run_button_is_available_beside_regression_suite():
    widgets = make_widgets()
    button = widgets.repeat_last_run_button

    assert isinstance(button, ipywidgets.Button)
    assert button.disabled is True
    assert button.description == "Repeat last"
    assert button.icon == "repeat"
    assert button.tooltip == "Repeat the previous classifier settings; data is fetched again"


def test_repeat_last_run_becomes_ready_when_saved_run_and_connection_are_available():
    widgets = make_widgets()
    widgets.guihandler.saved_run_available = True
    widgets.sql_username.value = "test-user"
    widgets.sql_password.value = "test-password"
    widgets.data_catalogs_dropdown.options = ("", "CatalogA")
    widgets.data_catalogs_dropdown.disabled = False

    widgets.update_regression_suite_ready()

    assert widgets.repeat_last_run_button.disabled is False


def test_repeat_last_run_uses_current_credentials_without_manual_dataset_setup():
    widgets = make_widgets()
    widgets.guihandler.saved_run_available = True
    widgets.sql_username.value = "test-user"
    widgets.sql_password.value = "test-password"

    widgets.repeat_last_run_button_actions()

    assert widgets.guihandler.repeat_calls == [("test-user", "test-password")]
    assert widgets.regression_suite_state is False


def test_repeat_last_restore_updates_visible_gui_state_and_next_rerun_config():
    widgets = make_widgets()
    widgets.sql_username.value = "current-user"
    widgets.sql_password.value = "current-password"

    saved = {
        "connection": Config.Connection(
            sql_username="saved-user",
            sql_password="saved-password",
            data_catalog="SavedCatalog",
            data_table="dbo.saved_table",
            class_column="target",
            data_text_columns=["comment"],
            data_numerical_columns=["amount", "age"],
            id_column="row_id",
        ),
        "mode": Config.Mode(
            train=False,
            predict=True,
            mispredicted=False,
            use_metas=True,
            use_stop_words=True,
            ngram_range=NgramRange.UNI_BI_GRAM,
            hex_encode=False,
            use_categorization=True,
            category_text_columns=["comment"],
            test_size=0.3,
            calculate_dark_numbers=False,
            dark_number_method=DarkNumberMethod.NON_LINEAR,
            dark_number_alpha=DarkNumberAlpha.SINGLE,
            oversampler=Oversampling.RND,
            undersampler=Undersampling.NUG,
            algorithm=AlgorithmTuple(("LRN", "RFCL")),
            preprocessor=PreprocessTuple(("NOS", "STA")),
            feature_selection=ReductionTuple(("NOR", "PCA")),
            scoring=ScoreMetric.f1_macro,
            max_iterations=12300,
        ),
        "io": Config.IO(verbose=True, model_name="saved_model"),
        "debug": Config.Debug(on=True, data_limit=321),
        "mail": Config.Mail(),
        "name": "saved-project",
        "save": True,
    }

    widgets.restore_classifier_config(saved)

    assert widgets.sql_username.value == "current-user"
    assert widgets.sql_password.value == "current-password"
    assert widgets.project.value == "saved-project"
    assert widgets.data_catalogs_dropdown.value == "SavedCatalog"
    assert widgets.data_tables_dropdown.value == "dbo.saved_table"
    assert widgets.models_dropdown.value == "saved_model.sav"
    assert widgets.class_column.value == "target"
    assert widgets.id_column.value == "row_id"
    assert widgets.data_columns.value == ("amount", "age", "comment")
    assert "N/A" not in widgets.class_column.options
    assert "N/A" not in widgets.id_column.options
    assert "N/A" not in widgets.data_columns.options
    assert widgets.text_columns.value == ("comment",)
    assert widgets.train_checkbox.value is False
    assert widgets.predict_checkbox.value is True
    assert widgets.mispredicted_checkbox.value is False
    assert widgets.metas_checkbox.value is True
    assert widgets.dark_numbers_checkbox.value is False
    assert widgets.dark_number_method.value == "NON_LINEAR"
    assert widgets.dark_number_alpha.value == "SINGLE"
    assert widgets.algorithm_dropdown.value == ("LRN", "RFCL")
    assert widgets.preprocess_dropdown.value == ("NOS", "STA")
    assert widgets.reduction_dropdown.value == ("NOR", "PCA")
    assert widgets.scoremetric_dropdown.value == "f1_macro"
    assert widgets.oversampler_dropdown.value == "RND"
    assert widgets.undersampler_dropdown.value == "NUG"
    assert widgets.testdata_slider.value == 30
    assert widgets.iterations_slider.value == 12300
    assert widgets.encryption_checkbox.value is False
    assert widgets.categorize_checkbox.value is True
    assert widgets.categorize_columns.value == ("comment",)
    assert widgets.filter_checkbox.value is True
    assert widgets.ngram_range_dropdown.value == "UNI_BI_GRAM"
    assert widgets.data_limit.value == 321
    assert widgets.show_info_checkbox.value is True
    assert widgets.num_variables.value == 3

    rerun = widgets.get_config_params()
    assert rerun["connection"].sql_username == "current-user"
    assert rerun["connection"].sql_password == "current-password"
    assert rerun["connection"].data_catalog == "SavedCatalog"
    assert rerun["connection"].data_table == "dbo.saved_table"
    assert rerun["connection"].class_column == "target"
    assert rerun["connection"].id_column == "row_id"
    assert rerun["connection"].data_numerical_columns == ["amount", "age"]
    assert rerun["connection"].data_text_columns == ["comment"]
    assert rerun["mode"].algorithm.get_abbreviations() == ["LRN", "RFCL"]
    assert rerun["mode"].preprocessor.get_abbreviations() == ["NOS", "STA"]
    assert rerun["mode"].feature_selection.get_abbreviations() == ["NOR", "PCA"]
    assert rerun["mode"].calculate_dark_numbers is False
    assert rerun["mode"].dark_number_method is DarkNumberMethod.NON_LINEAR
    assert rerun["mode"].dark_number_alpha is DarkNumberAlpha.SINGLE
    assert rerun["io"].model_name == "saved_model"
    assert rerun["debug"].data_limit == 321

    saved["mode"].train = True
    saved["mode"].predict = False
    widgets.restore_classifier_config(saved)
    assert widgets.models_dropdown.value == Config.DEFAULT_TRAIN_OPTION
    assert widgets.train_checkbox.value is True
    assert widgets.predict_checkbox.value is False

def test_repeat_last_restore_suppresses_class_summary_queries_while_state_is_incomplete():
    widgets = make_widgets()
    widgets.sql_username.value = "current-user"
    widgets.sql_password.value = "current-password"

    saved = {
        "connection": Config.Connection(
            data_catalog="SavedCatalog",
            data_table="dbo.saved_table",
            class_column="target",
            data_numerical_columns=["amount"],
            id_column="row_id",
        ),
        "mode": Config.Mode(
            train=True,
            predict=False,
            mispredicted=True,
            algorithm=AlgorithmTuple(("LDA",)),
            preprocessor=PreprocessTuple(("NOS",)),
            feature_selection=ReductionTuple(("NOR",)),
        ),
        "io": Config.IO(model_name="saved_model"),
        "debug": Config.Debug(on=True, data_limit=100),
        "mail": Config.Mail(),
        "name": "saved-project",
        "save": True,
    }

    # A fresh GUI still has N/A as the class selection. Restoring the saved
    # options/value must not let the class-column observer query SQL midway
    # through restoration with that transient placeholder.
    widgets.restore_classifier_config(saved)

    assert widgets.guihandler.class_distribution_calls == []
    assert widgets.class_column.value == "target"
    assert widgets.data_catalogs_dropdown.value == "SavedCatalog"
    assert widgets.data_tables_dropdown.value == "dbo.saved_table"


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
    assert widgets.project.value == "test_suite"
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

    assert params["name"] == "test_suite"
    assert params["io"].model_name == "test_suite"
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


def test_dark_number_checkbox_controls_method_widget_independently_of_mispredictions():
    widgets = make_widgets()
    widgets.dark_numbers_checkbox.disabled = False
    widgets.dark_number_method.disabled = False
    widgets.dark_number_alpha.disabled = False
    widgets.mispredicted_checkbox.value = True

    widgets.dark_numbers_checkbox.value = False
    assert widgets.dark_number_method.disabled is True
    assert widgets.dark_number_alpha.disabled is True
    assert widgets.mispredicted_checkbox.value is True

    widgets.dark_numbers_checkbox.value = True
    assert widgets.dark_number_method.disabled is False
    assert widgets.dark_number_alpha.disabled is False
    widgets.dark_number_method.value = "NON_LINEAR"
    widgets.dark_number_alpha.value = "SINGLE"

    config = widgets.get_config_params()["mode"]
    assert config.calculate_dark_numbers is True
    assert config.dark_number_method is DarkNumberMethod.NON_LINEAR
    assert config.dark_number_alpha is DarkNumberAlpha.SINGLE


def test_dark_number_checkbox_still_controls_dependencies_after_continue_locks_observers():
    widgets = make_widgets()
    widgets.dark_numbers_checkbox.disabled = False
    widgets.dark_number_method.disabled = False
    widgets.dark_number_alpha.disabled = False
    widgets.eventhandler.lock_observe_1 = True

    widgets.dark_numbers_checkbox.value = False
    assert widgets.dark_number_method.disabled is True
    assert widgets.dark_number_alpha.disabled is True

    widgets.dark_numbers_checkbox.value = True
    assert widgets.dark_number_method.disabled is False
    assert widgets.dark_number_alpha.disabled is False


def test_classifier_activation_preserves_dark_number_dependency_state():
    widgets = make_widgets()
    widgets.dark_numbers_checkbox.value = False

    widgets.activate_section("classifier")

    assert widgets.dark_numbers_checkbox.disabled is False
    assert widgets.dark_number_method.disabled is True
    assert widgets.dark_number_alpha.disabled is True


def test_regression_suite_keeps_dark_number_controls_read_only():
    widgets = make_widgets()

    widgets.regression_suite_button_actions()

    assert widgets.dark_numbers_checkbox.value is True
    assert widgets.dark_numbers_checkbox.disabled is True
    assert widgets.dark_number_method.disabled is True
    assert widgets.dark_number_alpha.disabled is True


def test_dark_number_alpha_options_follow_selected_method_without_inventing_formula():
    widgets = make_widgets()
    widgets.dark_number_method.disabled = False
    widgets.dark_number_alpha.disabled = False

    widgets.dark_number_method.value = "LINEAR"
    assert tuple(widgets.dark_number_alpha.options) == (
        ("None", "NONE"), ("Single", "SINGLE"), ("Separated", "SEPARATED")
    )
    widgets.dark_number_alpha.value = "SEPARATED"

    widgets.dark_number_method.value = "NON_LINEAR"
    assert tuple(widgets.dark_number_alpha.options) == (("None", "NONE"), ("Single", "SINGLE"))
    assert widgets.dark_number_alpha.value == "NONE"


def test_repeat_last_and_regression_suite_are_aligned_to_opposite_edges():
    widgets = make_widgets()
    form = widgets.regression_suite_form()
    row = form.children[1]

    assert tuple(row.children) == (
        widgets.repeat_last_run_button,
        widgets.regression_suite_button,
    )
    assert row.layout.justify_content == "space-between"
    assert row.layout.width == "100%"


def test_legacy_local_settings_labels_are_migrated_without_overriding_custom_text():
    src_path = Path(__file__).parents[1] / "src" / "JBGclassification"
    model_path = Path(__file__).parent / "fixtures"
    settings_path = src_path / "GUI" / "default_settings.json"
    settings = json.loads(settings_path.read_text())
    settings["widgets"]["train_checkbox"]["params"]["description"] = "Mode: Train"
    settings["widgets"]["dark_numbers_checkbox"]["params"]["description"] = "Dark Numbers: Calculate"
    settings["widgets"]["encryption_checkbox"]["params"]["description"] = "Text: Encryption"
    settings["widgets"]["categorize_checkbox"]["params"]["description"] = "Text: Categorize"
    settings["widgets"]["filter_checkbox"]["params"]["description"] = "Text: Filter"
    settings["widgets"]["ngram_range_dropdown"]["params"]["description"] = "Text: Ngrams"
    settings["widgets"]["predict_checkbox"]["params"]["description"] = "Custom prediction label"

    widgets = Widgets(
        src_path=src_path,
        GUIhandler=MockGUIHandler(),
        model_path=model_path,
        settings=settings,
    )

    assert widgets.train_checkbox.description == "Train"
    assert widgets.dark_numbers_checkbox.description == ""
    assert widgets.encryption_checkbox.description == "Encryption"
    assert widgets.categorize_checkbox.description == "Categorizer"
    assert widgets.filter_checkbox.description == "Filter"
    assert widgets.ngram_range_dropdown.description == "Ngrams"
    assert widgets.predict_checkbox.description == "Custom prediction label"


def test_pre_065_categorize_label_is_migrated_to_categorizer():
    src_path = Path(__file__).parents[1] / "src" / "JBGclassification"
    model_path = Path(__file__).parent / "fixtures"
    settings_path = src_path / "GUI" / "default_settings.json"
    settings = json.loads(settings_path.read_text())
    settings["widgets"]["categorize_checkbox"]["params"]["description"] = "Categorize"

    widgets = Widgets(
        src_path=src_path,
        GUIhandler=MockGUIHandler(),
        model_path=model_path,
        settings=settings,
    )

    assert widgets.categorize_checkbox.description == "Categorizer"


def test_classifier_checkbox_labels_are_scoped_by_section_titles():
    widgets = make_widgets()

    assert widgets.train_checkbox.description == "Train"
    assert widgets.predict_checkbox.description == "Predict"
    assert widgets.mispredicted_checkbox.description == "Display mispredictions"
    assert widgets.metas_checkbox.description == "Pass on meta data"
    assert widgets.dark_numbers_checkbox.description == ""


def test_horizontal_forms_have_consistent_titles_and_frames():
    widgets = make_widgets()
    sections = (
        (widgets.connection_form, "Database connection"),
        (widgets.regression_suite_form, "Run history & regression"),
        (widgets.models_form, "Model"),
        (widgets.data_form, "Dataset columns"),
        (widgets.continuation_form, "Dataset actions"),
        (widgets.checkboxes_form, "Mode"),
        (widgets.dark_numbers_form, "Dark Numbers"),
        (widgets.algorithm_form, "Model selection"),
        (widgets.data_handling_form, "Training data"),
        (widgets.text_handling_form, "Text processing"),
        (widgets.debug_form, "Run settings"),
    )

    for factory, title in sections:
        form = factory()
        heading, row = form.children
        assert isinstance(form, ipywidgets.VBox)
        assert isinstance(heading, ipywidgets.HTML)
        assert title in heading.value
        assert isinstance(row, ipywidgets.Box)
        assert form.layout.border == "2px solid #d0d0d0"
        assert form.layout.width == "100%"


def test_progress_form_has_no_section_title_or_frame():
    widgets = make_widgets()

    form = widgets.progress_form()

    assert isinstance(form, ipywidgets.HBox)
    assert tuple(form.children) == (widgets.progress_bar, widgets.progress_label)
    assert form.layout.border in (None, "")


def test_new_model_uses_separated_alpha_by_default():
    widgets = make_widgets()

    widgets.set_checkboxes(new_model=True)

    assert widgets.dark_number_method.value == "LINEAR"
    assert widgets.dark_number_alpha.value == "SEPARATED"
