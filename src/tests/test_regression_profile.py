from pathlib import Path

import ipywidgets

from GUI.Widgets import Widgets


class MockDataLayer:
    def get_trained_models_from_files(self, *args, **kwargs):
        return ["config-save.sav", "model-save.sav"]


class MockGUIHandler:
    @property
    def datalayer(self):
        return MockDataLayer()


def make_widgets() -> Widgets:
    src_path = Path(__file__).parents[1] / "JBGclassification"
    model_path = Path(__file__).parent / "fixtures"
    return Widgets(src_path=src_path, GUIhandler=MockGUIHandler(), model_path=model_path)


def test_regression_test_profile_values():
    widgets = make_widgets()
    widgets.apply_regression_test_profile()

    assert widgets.algorithm_dropdown.value == ("LRN", "RFCL", "LSVC", "GNB", "KNN", "LDA")
    assert widgets.preprocess_dropdown.value == ("NOS", "STA", "MIX")
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
    assert tuple(form.children) == (widgets.continuation_button, widgets.test_profile_button)
    assert form.layout.justify_content == "space-between"
    assert form.layout.align_items == "center"
    assert form.layout.width == "100%"
