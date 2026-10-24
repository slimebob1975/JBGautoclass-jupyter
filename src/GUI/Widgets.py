from __future__ import annotations
from enum import Enum
import html
import os
from pathlib import Path
import sys
import json
import logging
from IPython.display import HTML as DisplayHTML, display

from typing import Callable, Iterable, Protocol
import ipywidgets as widgets
import pandas as pd
from pandas import DataFrame
from sklearn.utils import Bunch

from Config import (Config, Reduction, ReductionTuple, Algorithm, 
                    AlgorithmTuple, Preprocess, PreprocessTuple, 
                    ScoreMetric, Oversampling, Undersampling, NgramRange, DarkNumberAlpha, DarkNumberMethod)
from JBGTransformers import TextDataToNumbersConverter
from JBGDarkNumbers import DarkNumberCalculator
from JBGScoring import score_metric_help
        
from JBGExceptions import GuiWidgetsException
import Helpers
from JBGModelSettings import REGISTRY, defaults, validate
from JBGSampling import sampling_names
from JBGFeatureSearch import SearchOptions, resolve_feature_search_workers
from JBGFeatureProfiles import save_candidate, load_candidates, find_candidate, review_compatibility, profile_display_name, delete_candidate
from JBGModelCompatibility import review_current_source_compatibility
from JBGModelPersistence import load_model_compatibility
from JBGVersion import application_version_line

class WidgetConstant(Enum):
    DATA_NUMERICAL = 1
    DATA_TEXT = 2

class WidgetStatus(Enum):
    UNCHECKED = 0
    OK = 1
    WARNING = 2
    ERROR = 3

class EventHandler:
    """ Handles interactions between widgets """

    def __init__(self, gui_widgets: Widgets):
        """ Initialises handler things """

        self.widgets = gui_widgets

        # TODO: Can we fix this in a neater way?
        # Some data elements might get lost unless we lock a few callback functions
        self.lock_observe_1 = False

    def should_run_event(self, change: Bunch) -> bool:
        """ This only returns True if event.name is 'index' and event.new > 0 or lock_observe_1 == False """
        if self.lock_observe_1:
            return False

        if change.name != "index":
            return False
        
        if change.new == 0:
            return False

        return True
    
    def print_event_info(self, caller: str, change: Bunch) -> None:
        """ Just prints some info about the event to the output, for debug """
        print(f"{caller}: {change.name=}, {change.old=}, {change.new=}")

    def sql_username_and_password(self, change: Bunch) -> None:
        
        # Handler for changes in sql username and password text fields
        if change['name'] == 'value':                
            if self.widgets.sql_username.value != "" and self.widgets.sql_password.value != "":
                self.widgets.update_sql_username_and_password(
                    self.widgets.sql_username.value,
                    self.widgets.sql_password.value
                )
                self.widgets.update_data_catalogs_dropdown()
                self.widgets.data_catalogs_dropdown.disabled = False
                self.widgets.update_regression_suite_ready()
            else:
                self.widgets.data_catalogs_dropdown.disabled = True
                self.widgets.update_regression_suite_ready()
    
    def data_catalogs_dropdown(self, change: Bunch) -> None:
        """ Handler for data_catalogs_dropdown
            Events:
                value_change: Updates the data tables dropdown when catalog is picked
        """
        # Change to default values below if it's changing in ways
        if not self.should_run_event(change):
            return
        self.widgets.update_data_catalog()
        self.widgets.update_data_tables_dropdown()
    
    def data_tables_dropdown(self, change: Bunch) -> None:
        """ Handler for data_tables_dropdown
            Events:
                value_change: Sets the models dropdown to enabled when table is picked
        """
        # Change to default values below if it's changing in ways
        if not self.should_run_event(change):
            return
        
        self.widgets.update_item("models_dropdown", {"disabled": False})

    def models_dropdown(self, change: Bunch) -> None:
        """ Handler for models_dropdown
            Events:
                value_change: Triggers several changes when model is set
        """
        # Programmatic Repeat Last restoration must not reload model metadata over
        # the persisted run state that is being made visible.
        if self.lock_observe_1 or change.name != "index":
            return
        
        if change.new == 0: # Moves to the empty value
            return
        
        if not self.lock_observe_1:
            self.widgets.update_class_id_data_columns()
        
        self.widgets.disable_post_model_dropdowns() # Locks all except project name
        
        self.widgets.form_field_enabled_toggle("data", disabled=True) # Disable before updating

        self.widgets.reset_field_status("models")
        self.widgets.update_field_status("models")

        if self.widgets.models_status == WidgetStatus.OK:
            if change.new == 1: # This is the default train option
                self.widgets.form_field_enabled_toggle("data", disabled=False)

                self.widgets.disable_button("continuation_button")
                self.widgets.disable_button("test_profile_button")
                return
            
            self.widgets.update_from_model_config()
        

    def class_column(self, change: Bunch) -> None:
        """ Handler for class_column
            Events:
                value_change: Changes the id and data columns, updates summary
        """
        # Programmatic state restoration mutates options/value while the GUI can
        # still contain the default ``N/A`` selection. Do not query the database
        # until the persisted dataset/class state has been restored completely.
        if self.lock_observe_1 or getattr(self.widgets, "_updating_column_options", False):
            return

        if change.name not in ["index", "disabled"]:
            return
        
        if change.name == "disabled" and change.name:
            # This is still disabled
            return

        if self.widgets.class_column.value is None:
            self.widgets.summarise_state = False
            self.widgets.update_id_column()
            self.widgets.update_data_columns()
            return

        self.widgets.summarise_state = True # NB: It will not update the summary if summarise_state = False
 
        """ update_id_and_data_columns
        """
        self.widgets.update_class_summary()
        self.widgets.update_id_column() # Event.name "options"
        self.widgets.update_data_columns()

    def id_column(self, change: Bunch) -> None:
        """ Handler for id_column
            Events:
                value_change: Updates the data column option when id is set
        """
        if self.lock_observe_1: 
            return
        
        if change.name not in ["index", "options"]:
            return
        
        self.widgets.update_data_columns()

    def data_columns(self, change: Bunch) -> None:
        """ Handler for data_columns
            Events:
                value_change: Updates the text columns and continuation_button states when data options changes
        """
        if self.lock_observe_1: 
            return
        
        if change.name not in ["index", "options"]:
            return

        self.widgets.update_text_columns()
        self.widgets.refresh_feature_search_compare_options()
        self.widgets.invalidate_feature_search_result()
        self.widgets.update_ready() # At the moment enables/disables the continuation_button
        self.widgets.sync_feature_selection_controls()

    def feature_selection_mode(self, change: Bunch) -> None:
        if self.lock_observe_1:
            return
        if change.get("name") == "value":
            self.widgets._feature_search_applied = False
            self.widgets.update_ready()

    def feature_selection_profile(self, change: Bunch) -> None:
        if not self.lock_observe_1 and change.get("name") == "value":
            self.widgets._feature_search_applied = False
            self.widgets.cancel_feature_search_deletion()
            self.widgets.update_ready()
            self.widgets.sync_feature_selection_controls(preserve_status=True)
            if self.widgets.feature_selection_mode.value == "reuse" and change.get("new"):
                self.widgets.review_feature_search_candidate()

    def feature_search_model_type(self, change: Bunch) -> None:
        if self.lock_observe_1:
            return
        if change.get("name") == "value":
            self.widgets.invalidate_feature_search_result()
            self.widgets.update_ready()
            self.widgets.sync_feature_selection_controls()

    def feature_search_preset(self, change: Bunch) -> None:
        if self.lock_observe_1 or change.get("name") != "value":
            return
        if change.get("new") != "Custom":
            self.widgets.invalidate_feature_search_result()
            self.widgets.set_feature_search_preset(change["new"])
            self.widgets.update_ready()
        else:
            self.widgets.invalidate_feature_search_result()
            self.widgets.update_feature_search_fit_budget()

    def feature_search_setting(self, change: Bunch) -> None:
        if self.lock_observe_1:
            return
        if change.get("name") == "value":
            from JBGFeatureSearchPresets import detected_preset, SEARCH_FIELDS
            self.widgets.invalidate_feature_search_result()
            if any(change.get("owner") is getattr(self.widgets, "feature_search_" + name)
                   for name in SEARCH_FIELDS):
                self.lock_observe_1 = True
                try:
                    settings = self.widgets.feature_search_options()
                    preset = detected_preset({name: getattr(settings, name) for name in SEARCH_FIELDS})
                    self.widgets.feature_search_preset.value = preset
                except ValueError:
                    self.widgets.feature_search_preset.value = "Custom"
                finally:
                    self.lock_observe_1 = False
            self.widgets.update_feature_search_fit_budget()
            self.widgets.update_ready()

    def feature_search_representation_setting(self, change: Bunch) -> None:
        """Invalidate a candidate when preprocessing choices used by the search change."""
        if self.lock_observe_1:
            return
        if change.get("name") == "value":
            self.widgets.invalidate_feature_search_result()
            self.widgets.update_ready()
            self.widgets.sync_feature_selection_controls()

    def feature_search_investigate_button_was_clicked(self, button: widgets.Button) -> None:
        self.widgets.run_feature_selection_investigation()

    def feature_search_apply_button_was_clicked(self, button: widgets.Button) -> None:
        self.widgets.apply_feature_search_recommendation()

    def feature_search_save_button_was_clicked(self, button: widgets.Button) -> None:
        self.widgets.save_feature_search_candidate()

    def feature_search_review_button_was_clicked(self, button: widgets.Button) -> None:
        self.widgets.review_feature_search_candidate()

    def feature_search_delete_button_was_clicked(self, button: widgets.Button) -> None:
        self.widgets.delete_feature_search_candidate()

    def algorithm_dropdown_handler(self, change: Bunch) -> None:
        self.check_num_alg_pre_red(change)

    def preprocess_dropdown_handler(self, change: Bunch) -> None:
        self.check_num_alg_pre_red(change)

    def reduction_dropdown_handler(self, change: Bunch) -> None:
        self.check_num_alg_pre_red(change)
    
    def check_num_alg_pre_red(self, change: Bunch) -> None:
        num_alg = len(self.widgets.algorithm_dropdown.value)
        num_pre = len(self.widgets.preprocess_dropdown.value)
        num_red = len(self.widgets.reduction_dropdown.value)
        if num_alg < 1 or num_pre < 1 or num_red < 1:
            if not self.widgets.start_button.disabled:
                self.widgets.start_button.disabled = True
        elif num_alg >= 1 and num_pre >= 1 and num_red >= 1:
            if self.widgets.start_button.disabled:
                self.widgets.start_button.disabled = False
    
    def reclassify_dropdown__value(self, change: Bunch) -> None:
        """ Handler for mispredicted/reclassify value-event
            Triggers a correction into the database
        """
        if change.new != 0 and change.old != change.new:
            new_class, new_index = change.new
            self.widgets.correct_mispredicted_data(new_class, new_index)

    def predict_checkbox(self, change: Bunch) -> None:
        #Default is that predictions should use meta values, e.g., for comparison afterwards
        if change['name'] == 'value':                
            if self.widgets.predict_checkbox.value:
                self.widgets.metas_checkbox.disabled = False
                self.widgets.metas_checkbox.value = True
            else:
                self.widgets.metas_checkbox.disabled = True
                self.widgets.metas_checkbox.value = False

    def dark_numbers_checkbox(self, change: Bunch) -> None:
        if change.get("name") != "value":
            return
        self.widgets.sync_dark_number_control_state()

    def dark_number_method(self, change: Bunch) -> None:
        if change.get("name") != "value":
            return
        self.widgets.sync_dark_number_alpha_options()
        self.widgets.sync_dark_number_equation()

    def dark_number_alpha(self, change: Bunch) -> None:
        if change.get("name") != "value":
            return
        self.widgets.sync_dark_number_equation()

    def dark_number_target(self, change: Bunch) -> None:
        if change.get("name") != "value":
            return
        self.widgets.sync_dark_number_equation()

    def dark_number_failure_mode(self, change: Bunch) -> None:
        if change.get("name") != "value":
            return
        self.widgets.sync_dark_number_equation()

    def continuation_button_was_clicked(self, button: widgets.Button) -> None:
        """ Callback: Sets various states based on the value in models dropdown. """
        if getattr(self.widgets, "_feature_search_running", False):
            return
        if self.widgets.feature_selection_mode.value in ("assisted", "reuse") and not getattr(self.widgets, "_feature_search_applied", False):
            self.widgets.update_ready()
            return
        self.lock_observe_1 = True
        
        self.widgets.deactivate_section("data")
        self.widgets.deactivate_section("feature_selection")
        self.widgets.continuation_button_actions()

    def test_profile_button_was_clicked(self, button: widgets.Button) -> None:
        """Continue with a temporary regression profile for manual test runs."""
        self.continuation_button_was_clicked(button)
        self.widgets.apply_regression_test_profile()

    def regression_suite_button_was_clicked(self, button: widgets.Button) -> None:
        """Enter regression-suite mode without requiring a manually selected dataset."""
        self.lock_observe_1 = True
        self.widgets.deactivate_section("data")
        self.widgets.deactivate_section("feature_selection")
        self.widgets.regression_suite_button_actions()
        self.widgets.apply_regression_test_profile()

    def repeat_last_run_button_was_clicked(self, button: widgets.Button) -> None:
        """Rerun the last persisted manual classifier configuration immediately."""
        button.disabled = True
        self.lock_observe_1 = True
        self.widgets.deactivate_section("data")
        self.widgets.deactivate_section("feature_selection")
        self.widgets.deactivate_section("classifier")
        self.widgets.repeat_last_run_button_actions()
        

    def start_button_was_clicked(self, button: widgets.Button) -> None:
        """ Callback: Changes the state to be read-only, starts the classifier (or the class summary) """

        self.widgets.deactivate_section("classifier")
        self.widgets.start_button_actions()
    
    def reset_button_was_clicked(self, button: widgets.Button) -> None:
        """ Callback: Resets all widgets back to the way they were """

        self.widgets.regression_suite_state = False
        self.widgets._load_default_widgets(clear=True)
    
    def unimplemented_handler(self, change) -> None:
        """ Just a basic handler before I've done the correct one """
        self.print_event_info("Unimplemented", change)

    def print_to_output(self, thing) -> None:
        with self.widgets.output:
            print(thing)


class DataLayer(Protocol):
    """ The minimum amount of datalayer needed for the Widgets """
    
    def get_catalogs_as_options(self) -> list:
        """ Used in the GUI, to get the catalogs """
    
    def get_tables_as_options(self) -> list:
        """ Used in the GUI, to get the tables """

    def get_trained_models_from_files(self, model_path: str, model_file_extension: str, preface: list = None) -> list:
        """ Used in the GUI, get a list of pretrained models (base implementation assumes files)"""

    def get_id_columns(self, **kwargs) -> dict:
        """ Used in the GUI, gets name and type for columns """

    def get_column_cardinalities(self, data_catalog: str, data_table: str, columns: list[str]) -> dict[str, int]:
        """Return distinct value counts for candidate classification columns."""

    def get_unique_int_id_columns(self, data_catalog: str, data_table: str, int_columns: list[str]) -> list[str]:
        """ Return integer columns that are globally non-null and unique. """

    def get_table_columns(self, database: str, table: str) -> dict:
        """ Used by the regression suite to inspect tables across catalogs """
        
class GUIhandler(Protocol):
    def get_class_distribution(self, data_settings: dict, current_class: str) -> dict:
        """ Help function to avoid Widgets knowing about classifier DataLayer """

    def run_classifier(
        self,
        config_params: dict,
        output: widgets.Output,
        set_rerun: bool = True,
        regression_suite: bool = False,
    ) -> None:
        """ Sets up the classifier and then runs it"""

    def run_regression_suite(self, base_config_params: dict, output: widgets.Output) -> None:
        """ Run the configured regression datasets sequentially """

    def run_feature_selection_investigation(self, config_params: dict, options, on_event=None):
        """Run a training-only original-column investigation."""

    def has_saved_classifier_run(self) -> bool:
        """ Whether a previous manual classifier run can be repeated """

    def repeat_last_classifier_run(
        self,
        output: widgets.Output,
        sql_username: str,
        sql_password: str,
    ) -> None:
        """ Repeat the previous manual classifier run """

    def correct_mispredicted_data(self, new_class: str, index: int) -> None:
        """ Changes the original dataset """

    @property
    def datalayer(self) -> DataLayer:
        """ Returns the datalayer """

    def set_data_catalog(self, data_catalog: str) -> None:
        """ Updates the GUI datalayer with the data catalog """

class Widgets:
    """ Creates and populates the widgets of the GUI """
    TEXT_MIN_LIMIT = 30
    TEXT_AREA_MIN_LIMIT = 60
    REGRESSION_SUITE_PROJECT = "test_suite"

    # Temporary manual regression profile. Keep this in one place so the
    # button can be repurposed as later patches need different coverage.
    REGRESSION_TEST_PROFILE = {
        "algorithm_dropdown": (
            "LRN", "RFCL", "LSVC", "GNB", "KNN", "LDA",
            "DTC", "REC", "PAC", "SGDE", "QDA", "SVC", "HIST", "NCT", "BGC",
        ),
        "preprocess_dropdown": ("NOS", "STA", "MIX", "MAX"),
        "reduction_dropdown": ("NOR", "PCA", "RFE"),
        "scoremetric_dropdown": "balanced_accuracy",
        "oversampler_dropdown": ("NOG",),
        "undersampler_dropdown": ("NUG",),
        "testdata_slider": 20,
        "iterations_slider": 20000,
        "encryption_checkbox": True,
        "categorize_checkbox": True,
        "categorize_columns": (),
        "filter_checkbox": False,
        "ngram_range_dropdown": "UNI_GRAM",
        "show_info_checkbox": True,
    }
    
    def __init__(self, src_path: Path, GUIhandler: GUIhandler, model_path: Path = None, settings: dict = None) -> None:

        if settings:
            self.settings = settings
        else:
            with open(self.get_sibling_file_path("default_settings.json")) as f: # The default_settings.json is sibling
                self.settings = json.load(f)
        
        self.eventhandler = EventHandler(self)
        self.guihandler = GUIhandler
        self.model_path = model_path if model_path else src_path / Config.DEFAULT_MODELS_PATH
        
        self.default_widgets = self.settings.get("widgets")
        self._migrate_legacy_widget_labels()
        self._migrate_default_scoring_metric()
        self._migrate_sampling_widgets()
        self.sections = self.settings.get("sections")
        self.logo_image =  self.get_sibling_file_path(self.settings.get("logo"))
        self.field_statuses = {}
        
        self.states = {
            "rerun": False,
            "summarise": False,
            "regression_suite": False,
        }
        self.feature_search_result = None
        self.widgets = {}
        self._load_default_widgets()
        #print(self.widgets)
        
        self._model_parameter_controls = {}
        self._model_settings_panel = self._create_model_settings_panel()
        # Keep this observer independent of the readiness/locking event handler.
        self.algorithm_dropdown.observe(self._model_selection_changed, names="value")
        self.refresh_model_settings_panel()
        self.forms = {
            "connection": [self.sql_username, self.sql_password, self.data_catalogs_dropdown, self.data_tables_dropdown],
            "models": [self.models_dropdown, self.field_status("models")],
            "data": [self.class_cardinality_limit, self.class_column, self.id_column, self.data_columns, self.text_columns],
            "feature_selection": [
                self.feature_selection_mode, self.feature_selection_profile, self.feature_search_rows, self.feature_search_model_type, self.feature_search_preset, self.feature_search_tree_estimator,
                self.feature_search_scaler, self.feature_search_oversampling, self.feature_search_reduction, self.feature_search_trials, self.feature_search_sizes,
                self.feature_search_inner_folds, self.feature_search_outer_folds, self.feature_search_top_sets,
                self.feature_search_refine_rounds, self.feature_search_max_evaluations, self.feature_search_min_gain,
                self.feature_search_linear_c, self.feature_search_linear_max_iterations,
                self.feature_search_non_linear_max_iterations, self.feature_search_non_linear_alpha,
                self.feature_search_non_linear_hidden_layers, self.feature_search_non_linear_learning_rate,
                self.feature_search_tree_estimator, self.feature_search_tree_n_estimators, self.feature_search_tree_max_depth,
                self.feature_search_tree_min_samples_leaf, self.feature_search_tree_class_weight, self.feature_search_tree_positive_threshold,
                self.feature_search_catboost_class_weight, self.feature_search_catboost_iterations, self.feature_search_investigate_button,
                self.feature_search_pause_button, self.feature_search_resume_button,
                self.feature_search_apply_button, self.feature_search_save_button, self.feature_search_review_button,
                self.feature_search_delete_button,
            ],
            "checkboxes": [self.train_checkbox, self.predict_checkbox, self.mispredicted_checkbox, self.metas_checkbox],
            "dark_numbers": [self.dark_numbers_checkbox, self.dark_number_target, self.dark_number_method, self.dark_number_alpha, self.dark_number_failure_mode],
            "feature_importance": [self.feature_importance_checkbox, self.feature_importance_repeats],
            "algorithm": [self.preprocess_dropdown, self.oversampler_dropdown, self.undersampler_dropdown, self.reduction_dropdown, self.algorithm_dropdown, self.scoremetric_dropdown],
            "data_handling": [self.testdata_slider, self.iterations_slider],
            "text_handling": [self.categorize_checkbox, self.categorize_columns, self.encryption_checkbox, self.filter_checkbox, self.ngram_range_dropdown],
            "debug": [self.data_limit, self.show_info_checkbox, self.num_variables],
            "progress": [self.progress_bar, self.progress_label]
        }

        # We need a dictionary to keep track of datatypes and the subset that
        # satisfies the GUI's "Unique id" contract for the current table.
        self.datatype_dict = None
        self.unique_id_columns = []
        self._class_cardinalities = {}

    def _migrate_default_scoring_metric(self) -> None:
        """Move local pre-176c GUI defaults from Accuracy to MCC.

        Saved run/config values are restored later and remain authoritative. This
        migration only updates the legacy GUI-default value in locally persisted
        ``settings.json`` files so a normal fresh run receives the new MCC default.
        """
        definition = self.default_widgets.get("scoremetric_dropdown", {})
        params = definition.get("params", {})
        if params.get("value") == "accuracy":
            params["value"] = "mcc"

    def _migrate_sampling_widgets(self) -> None:
        """Upgrade old local settings without changing the selected methods."""
        for name, enum_type, fallback in (
            ("oversampler_dropdown", Oversampling, Oversampling.NOG),
            ("undersampler_dropdown", Undersampling, Undersampling.NUG),
        ):
            definition = self.default_widgets[name]
            definition["type"] = "SelectMultiple"
            params = definition.setdefault("params", {})
            params["value"] = sampling_names(params.get("value"), enum_type, fallback)
            params["rows"] = 4
            params["tooltip"] = (
                "Select one or more methods (Ctrl/Cmd-click); include no sampling "
                "to compare it with active methods. Each selected pair is evaluated."
                + (" Random R20/R10/R05/R02 cap majority rows at 20/10/5/2 times "
                   "the smallest training-fold class; RND uses full balancing. "
                   "Already less imbalanced folds keep their rows."
                   if name == "undersampler_dropdown" else "")
            )

    def _migrate_legacy_widget_labels(self) -> None:
        """Migrate exact legacy labels while preserving user-customized text."""
        # Locally saved GUI settings predate this optional analysis section.
        new_widgets = {
            "feature_importance_checkbox": {
                "type": "Checkbox", "params": {
                    "value": False, "disabled": True, "description": "Feature importance",
                    "tooltip": "Measure score decrease when original inputs are shuffled on the final holdout; adds scoring work",
                },
            },
            "feature_importance_repeats": {
                "type": "BoundedIntText", "params": {
                    "value": 5, "min": 2, "max": 30, "disabled": True, "description": "Repeats:",
                    "tooltip": "Shuffle each input this many times; more repeats add scoring work",
                },
            },
            "feature_selection_mode": {"type": "RadioButtons", "params": {
                "options": [
                    ["I will pick my own data settings", "manual"],
                    ["Help me pick the best data settings", "assisted"],
                    ["Use a recommended data setting", "reuse"],
                ], "value": "manual", "disabled": True, "description": "Mode:",
            }},
            "feature_selection_profile": {"type": "Dropdown", "params": {
                "options": [["No saved recommendations", ""]], "value": "", "disabled": True,
                "description": "Profile:",
            }},
            "feature_search_rows": {"type": "BoundedIntText", "params": {"value": 1, "min": 1, "max": 2147483647, "disabled": True, "description": "Search rows:", "tooltip": "Maximum labelled rows fetched for investigation; separate from the normal ML limit."}},
            "feature_search_model_type": {"type": "Dropdown", "params": {
                "options": ["Linear", "Non-linear", "Tree", "Combination", "CatBoost"], "value": "Linear", "disabled": True,
                "description": "Model type:",
            }},
            "feature_search_preset": {"type": "Dropdown", "params": {"options": ["Quick", "Standard", "Extended", "Custom"], "value": "Quick", "disabled": True, "description": "Budget preset:", "tooltip": "Quick for smoke tests; Standard or Extended for larger searches. Manual changes set Custom."}},
            "feature_search_tree_estimator": {"type": "Dropdown", "params": {"options": ["Extra Trees", "Random Forest"], "value": "Extra Trees", "disabled": True, "description": "Tree model:", "tooltip": "Extra Trees uses randomized split thresholds; Random Forest optimizes candidate splits."}},
            "feature_search_scaler": {"type": "Dropdown", "params": {
                "options": ["STA", "MAX"], "value": "STA", "disabled": True, "description": "Scaler:",
            }},
            "feature_search_oversampling": {"type": "Dropdown", "params": {
                "options": [("No Oversampling", "NOG"), ("Random", "RND"), ("SMOTE", "SME")],
                "value": "NOG", "disabled": True, "description": "Oversampling:",
            }},
            "feature_search_reduction": {"type": "Dropdown", "params": {
                "options": [("No reduction", "NOR"), ("PCA (95% variance)", "PCA")],
                "value": "NOR", "disabled": True, "description": "Reduction:",
            }},
            "feature_search_trials": {"type": "IntSlider", "params": {"value": 4, "min": 1, "max": 1000, "step": 1, "continuous_update": False, "disabled": True, "description": "Trials:"}},
            "feature_search_sizes": {"type": "Text", "params": {"value": "5,10,20", "disabled": True, "description": "Subset sizes:"}},
            "feature_search_inner_folds": {"type": "IntSlider", "params": {"value": 3, "min": 2, "max": 20, "step": 1, "continuous_update": False, "disabled": True, "description": "Inner CV:"}},
            "feature_search_outer_folds": {"type": "IntSlider", "params": {"value": 3, "min": 2, "max": 20, "step": 1, "continuous_update": False, "disabled": True, "description": "Outer CV:"}},
            "feature_search_top_sets": {"type": "IntSlider", "params": {"value": 1, "min": 1, "max": 20, "step": 1, "continuous_update": False, "disabled": True, "description": "Top sets:"}},
            "feature_search_refine_rounds": {"type": "IntSlider", "params": {"value": 0, "min": 0, "max": 20, "step": 1, "continuous_update": False, "disabled": True, "description": "Refine rounds:"}},
            "feature_search_max_evaluations": {"type": "BoundedIntText", "params": {"value": 5, "min": 2, "max": 10000, "disabled": True, "description": "Max subsets:"}},
            "feature_search_min_gain": {"type": "FloatSlider", "params": {"value": 0.005, "min": 0.0, "max": 1.0, "step": 0.001, "readout_format": ".3f", "continuous_update": False, "disabled": True, "description": "Min gain:"}},
            "feature_search_linear_c": {"type": "BoundedFloatText", "params": {"value": 1.0, "min": 0.000001, "max": 1000000.0, "disabled": True, "description": "Linear C:"}},
            "feature_search_linear_max_iterations": {"type": "BoundedIntText", "params": {"value": 10000, "min": 1, "max": 1000000, "disabled": True, "description": "Linear iter.:"}},
            "feature_search_non_linear_max_iterations": {"type": "BoundedIntText", "params": {"value": 2000, "min": 1, "max": 1000000, "disabled": True, "description": "MLP iter.:"}},
            "feature_search_non_linear_alpha": {"type": "BoundedFloatText", "params": {"value": 0.0001, "min": 0.0, "max": 1000000.0, "step": 0.0001, "disabled": True, "description": "MLP alpha:"}},
            "feature_search_non_linear_hidden_layers": {"type": "Text", "params": {"value": "100", "disabled": True, "description": "MLP layers:"}},
            "feature_search_non_linear_learning_rate": {"type": "BoundedFloatText", "params": {"value": 0.001, "min": 0.000000001, "max": 1000000.0, "step": 0.001, "disabled": True, "description": "MLP learning:"}},
            "feature_search_catboost_class_weight": {"type": "Dropdown", "params": {"options": ["None", "Balanced"], "value": "None", "disabled": True, "description": "CB weights:"}},
            "feature_search_catboost_iterations": {"type": "BoundedIntText", "params": {"value": 400, "min": 1, "max": 10000, "disabled": True, "description": "CB iterations:"}},
            "feature_search_tree_n_estimators": {"type": "BoundedIntText", "params": {"value": 100, "min": 1, "max": 100000, "disabled": True, "description": "Trees:"}},
            "feature_search_tree_max_depth": {"type": "BoundedIntText", "params": {"value": 0, "min": 0, "max": 100000, "disabled": True, "description": "Max depth:"}},
            "feature_search_tree_class_weight": {"type": "Dropdown", "params": {"options": ["None", "balanced", "balanced_subsample"], "value": "None", "disabled": True, "description": "Class weight:"}},
            "feature_search_tree_positive_threshold": {"type": "BoundedFloatText", "params": {"value": 0.5, "min": 0.01, "max": 0.99, "step": 0.01, "disabled": True, "description": "Positive cutoff:"}},
            "feature_search_tree_min_samples_leaf": {"type": "BoundedIntText", "params": {"value": 1, "min": 1, "max": 100000, "disabled": True, "description": "Min leaf:"}},
            "feature_search_fit_budget": {"type": "HTML", "params": {"value": "Expected fit ceiling: 66 | Candidate workers: automatic"}},
            "feature_search_status": {"type": "HTML", "params": {"value": "Manual data settings; no investigation will run."}},
            "feature_search_preview": {"type": "HTML", "params": {"value": ""}},
            "feature_search_investigate_button": {"type": "Button", "params": {"description": "Investigate", "button_style": "primary", "disabled": True, "icon": "search"}},
            "feature_search_apply_button": {"type": "Button", "params": {"description": "Apply", "button_style": "primary", "disabled": True, "icon": "check"}},
            "feature_search_save_button": {"type": "Button", "params": {"description": "Save", "button_style": "", "disabled": True, "icon": "save"}},
            "feature_search_review_button": {"type": "Button", "params": {"description": "Review", "button_style": "", "disabled": True, "icon": "search"}},
            "feature_search_delete_button": {"type": "Button", "params": {"description": "Delete", "button_style": "danger", "disabled": True, "icon": "trash"}},
        }
        classifier_items = self.settings.get("sections", {}).get("classifier")
        for name, definition in new_widgets.items():
            self.default_widgets.setdefault(name, definition)
            if classifier_items is not None and name not in classifier_items:
                classifier_items.append(name)
        self.default_widgets.setdefault(
            "build_identity",
            {"type": "HTML", "params": {"value": "", "description": ""}},
        )
        display_items = self.settings.get("display", [])
        if "build_identity" not in display_items:
            position = display_items.index("welcome") + 1 if "welcome" in display_items else 0
            display_items.insert(position, "build_identity")
        if "feature_importance_form" not in display_items:
            position = display_items.index("algorithm_form") if "algorithm_form" in display_items else len(display_items)
            display_items.insert(position, "feature_importance_form")
        if "feature_selection_modelling_form" not in display_items:
            position = display_items.index("continuation_form") if "continuation_form" in display_items else len(display_items)
            display_items.insert(position, "feature_selection_modelling_form")
        self.settings.setdefault("sections", {})["feature_selection"] = [
            "feature_selection_mode", "feature_selection_profile", "feature_search_rows", "feature_search_model_type",
            "feature_search_scaler", "feature_search_oversampling", "feature_search_reduction", "feature_search_trials", "feature_search_sizes",
            "feature_search_inner_folds", "feature_search_outer_folds", "feature_search_top_sets",
            "feature_search_refine_rounds", "feature_search_max_evaluations", "feature_search_min_gain",
            "feature_search_linear_c", "feature_search_linear_max_iterations",
            "feature_search_non_linear_max_iterations", "feature_search_non_linear_alpha",
            "feature_search_non_linear_hidden_layers", "feature_search_non_linear_learning_rate",
            "feature_search_tree_n_estimators", "feature_search_tree_max_depth",
            "feature_search_tree_min_samples_leaf", "feature_search_tree_class_weight", "feature_search_tree_positive_threshold",
            "feature_search_catboost_class_weight", "feature_search_catboost_iterations", "feature_search_investigate_button",
            "feature_search_apply_button", "feature_search_save_button", "feature_search_review_button", "feature_search_delete_button",
        ]
        legacy_fallback = self.default_widgets.pop("dark_number_perturbed_fallback_checkbox", None)
        for section_name, section_items in self.settings.get("sections", {}).items():
            self.settings["sections"][section_name] = [
                "dark_number_failure_mode"
                if item == "dark_number_perturbed_fallback_checkbox"
                else item
                for item in section_items
            ]
        classifier_items = self.settings.get("sections", {}).get("classifier")
        if classifier_items is not None:
            # Keep the target choice immediately after the main Estimate checkbox,
            # including for locally persisted settings created before revision 097.
            classifier_items[:] = [item for item in classifier_items if item != "dark_number_target"]
            insert_at = (
                classifier_items.index("dark_numbers_checkbox") + 1
                if "dark_numbers_checkbox" in classifier_items
                else len(classifier_items)
            )
            classifier_items.insert(insert_at, "dark_number_target")
        if "dark_number_target" not in self.default_widgets:
            self.default_widgets["dark_number_target"] = {
                "type": "RadioButtons",
                "params": {
                    "options": [["All classes", ""]],
                    "value": "",
                    "disabled": True,
                    "description": "Target:",
                    "tooltip": "Choose which class to estimate Dark Numbers for, or keep all classes",
                    "layout": widgets.Layout(width="170px", min_width="170px", margin="0 0 0 6px"),
                    "style": {"description_width": "initial"},
                },
            }

        if classifier_items is not None and "dark_number_failure_mode" not in classifier_items:
            insert_at = (
                classifier_items.index("dark_number_alpha") + 1
                if "dark_number_alpha" in classifier_items
                else len(classifier_items)
            )
            classifier_items.insert(insert_at, "dark_number_failure_mode")
        if "dark_number_failure_mode" not in self.default_widgets:
            legacy_value = True
            if legacy_fallback is not None:
                legacy_value = bool(legacy_fallback.get("params", {}).get("value", True))
            self.default_widgets["dark_number_failure_mode"] = {
                "type": "RadioButtons",
                "params": {
                    "options": [["No fallback", False], ["Experimental", True]],
                    "value": legacy_value,
                    "disabled": True,
                    "description": "Failure:",
                    "tooltip": (
                        "Choose no fallback or the experimental perturbed same-model fallback "
                        "when ordinary correction is statistically unestimable"
                    ),
                    "layout": widgets.Layout(width="210px", min_width="210px", margin="0 0 0 8px"),
                    "style": {"description_width": "initial"},
                },
            }

        failure_widget = self.default_widgets.get("dark_number_failure_mode")
        if failure_widget is not None:
            params = failure_widget.setdefault("params", {})
            params["options"] = [["No fallback", False], ["Experimental", True]]
            params["description"] = "Failure:"
            params["tooltip"] = (
                "Choose no fallback or the experimental perturbed same-model fallback "
                "when ordinary correction is statistically unestimable"
            )
            params["layout"] = widgets.Layout(
                width="210px", min_width="210px", margin="0 0 0 8px"
            )
            params["style"] = {"description_width": "initial"}

        replacements = {
            "train_checkbox": (("Mode: Train",), "Train"),
            "predict_checkbox": (("Mode: Predict",), "Predict"),
            "mispredicted_checkbox": (("Mode: Display mispredictions",), "Display mispredictions"),
            "metas_checkbox": (("Mode: Pass on meta data",), "Pass on meta data"),
            "dark_numbers_checkbox": (("Dark Numbers: Calculate", ""), "Estimate"),
            "encryption_checkbox": (("Text: Encryption",), "Encryption"),
            "categorize_checkbox": (("Text: Categorize", "Categorize"), "Categorizer"),
            "filter_checkbox": (("Text: Filter",), "Filter"),
            "ngram_range_dropdown": (("Text: Ngrams",), "Ngrams"),
        }
        for widget_name, (legacy_labels, current) in replacements.items():
            params = self.default_widgets.get(widget_name, {}).get("params", {})
            if params.get("description") in legacy_labels:
                params["description"] = current

    def field_status(self, type: str) -> widgets.HTML:
        """ HTML fields for issues, connected to a particular field """
        
        if type in self.field_statuses:
            return self.update_field_status(type)

        self.field_statuses[type] = {
            "check": WidgetStatus.UNCHECKED,
            "note": widgets.HTML()
        }
        self.field_statuses[type]["note"].add_class(f"{type}-status")
        return self.field_statuses[type]["note"]

    def reset_field_status(self, type: str) -> None:
        self.field_statuses[type]["note"].value = "<em>Checking status</em>"

    
    def update_field_status(self, type: str) -> widgets.HTML:
        """ Updates the note based on the status of the field """
        check, note = self.get_field_status(type)

        if check == WidgetStatus.UNCHECKED:
            value = ""
        elif check == WidgetStatus.OK:
            value = "OK"
        elif check == WidgetStatus.WARNING:
            value = "<b>Warning</b>"
        elif check == WidgetStatus.ERROR:
            value = "<b>Error</b>"
        else:
            value = "Unknown value from status"

        self.field_statuses[type]["check"] = check
        if note:
            value = f"{value}: {note}"
        
        self.field_statuses[type]["note"].value = value

        return self.field_statuses[type]["note"]


    def get_field_status(self, field: str) -> tuple(WidgetStatus, str):
        """ Returns a constant based on the state of the field """
        func_name = f"check_{field}_field_status"
        func = getattr(self, func_name)
        return func()
        
    @property
    def models_status(self) -> WidgetStatus:
        return self.field_statuses["models"]["check"]


    def check_models_field_status(self) -> tuple(WidgetStatus, str):
        """ The specific code for checking the models_dropdown field """
        if not self.data_form_fields_valid:
            return WidgetStatus.WARNING, "Columns from table not available"
        
        if self.models_dropdown.value == "":
            return WidgetStatus.UNCHECKED, ""

        if self.new_model:
            return WidgetStatus.OK, ""

        try:
            config = self.load_model_config()
        except Exception as e:
            return WidgetStatus.ERROR, str(e)

        columns = list(self.datatype_dict.keys())
        if not config.check_config_against_data_configuration(columns):
            note = f"Model <em>{self.model_name}</em> contains columns not in chosen table"
            return WidgetStatus.ERROR, note

        try:
            model_path = self.model_path / self.models_dropdown.value
            compatibility = load_model_compatibility(model_path)
        except Exception as exc:
            return WidgetStatus.ERROR, "Could not read saved-model compatibility metadata: " + html.escape(str(exc))

        review = review_current_source_compatibility(
            compatibility,
            driver=os.environ.get("DEFAULT_ODBC_DRIVER", ""),
            host=os.environ.get("DEFAULT_HOST", ""),
            catalog=self.data_catalogs_dropdown.value,
            table=self.data_tables_dropdown.value,
            source_schema=self.datatype_dict,
        )
        if review.issues:
            return WidgetStatus.ERROR, "Saved-model compatibility check failed: " + html.escape("; ".join(review.issues))
        if review.warnings:
            return WidgetStatus.OK, html.escape("; ".join(review.warnings))
        return WidgetStatus.OK, "Saved-model source and SQL schema compatibility verified"


    def get_sibling_file_path(self, file: str) -> Path:
        """ Gets a file in the same directory as the GUI module """
        return Path(__file__).parent / file

    @property
    def datalayer(self) -> DataLayer:
        """ This returns the datalayer from the gui handler """
        return self.guihandler.datalayer

    @property
    def rerun_state(self) -> bool:
        return self.states["rerun"]

    @rerun_state.setter
    def rerun_state(self, state: bool) -> None:
        self.states["rerun"] = state
    
    @property
    def regression_suite_state(self) -> bool:
        return self.states["regression_suite"]

    @regression_suite_state.setter
    def regression_suite_state(self, state: bool) -> None:
        self.states["regression_suite"] = state

    @property
    def summarise_state(self) -> bool:
        return self.states["summarise"]

    @summarise_state.setter
    def summarise_state(self, state: bool) -> None:
        self.states["summarise"] = state

    
    def load_contents(self) -> None:
        """ This is when we set content loaded from datasource """
        if not self.datalayer:
            raise GuiWidgetsException("Data Layer not initialized")

        #self.update_data_catalogs_dropdown()
        self.update_models_dropdown_options()

    def load_model_config(self, model: str = None) -> Config:
        """ Break out the loading so it can be checked earlier """
        value = model if model else self.models_dropdown.value
        model_path = self.model_path / value
        
        return Config.load_config_from_model_file(model_path)

    def update_from_model_config(self, model: str = None ) -> None:
        """ Sets values based on the config in the chosen model """
        config = self.load_model_config(model)
        self.ensure_saved_model_class_option(config)
        self.restore_model_parameters(getattr(config.mode, "model_parameters", {}))
        self.restore_decision_policy(config.mode)
        self.categorize_columns.options = config.get_text_column_names()
        saved_dark_number_target = config.get_dark_number_target()
        if saved_dark_number_target:
            self.dark_number_target.options = self._with_saved_options(
                self.dark_number_target.options, (saved_dark_number_target,)
            )
            
        # Values from config
        self.update_values({
            "class_column": config.get_class_column_name(),
            "id_column": config.get_id_column_name(),
            "data_columns": config.get_data_column_names(),
            "text_columns": config.get_text_column_names(),
            "algorithm_dropdown": tuple(config.get_algorithm_abbreviations()),
            "preprocess_dropdown": tuple(config.get_preprocessor_abbreviations()), 
            "reduction_dropdown": tuple(config.get_feature_selection_abbreviations()),
            "num_variables": config.get_num_selected_features(),
            "filter_checkbox": config.use_stop_words(),
            "ngram_range_dropdown": config.get_ngram_range(),
            "encryption_checkbox": config.should_hex_encode(),
            "categorize_checkbox": config.use_categorization(),
            "categorize_columns":  config.get_categorical_text_column_names(),
            "testdata_slider": config.get_test_size_percentage(),
            "oversampler_dropdown": tuple(config.get_oversampler_abbreviations()),
            "undersampler_dropdown": tuple(config.get_undersampler_abbreviations()),
            "dark_numbers_checkbox": config.should_calculate_dark_numbers(),
            "feature_importance_checkbox": config.should_calculate_feature_importance(),
            "feature_importance_repeats": config.get_feature_importance_repeats(),
            "feature_selection_mode": getattr(config.mode, "feature_selection_mode", "manual"),
            "feature_selection_profile": getattr(config.mode, "feature_selection_profile_id", ""),
            "feature_search_model_type": getattr(config.mode, "feature_search_model_type", "Linear"),
            "feature_search_preset": getattr(config.mode, "feature_search_preset", "Quick"),
            "feature_search_tree_estimator": getattr(config.mode, "feature_search_tree_estimator", "Extra Trees"),
            "feature_search_scaler": getattr(config.mode, "feature_search_scaler", "STA"),
            "feature_search_oversampling": getattr(config.mode, "feature_search_oversampling", "NOG"),
            "feature_search_reduction": getattr(config.mode, "feature_search_reduction", "NOR"),
            "feature_search_trials": getattr(config.mode, "feature_search_trials", 4),
            "feature_search_sizes": ",".join(str(value) for value in getattr(config.mode, "feature_search_sizes", [5, 10, 20])),
            "feature_search_inner_folds": getattr(config.mode, "feature_search_inner_folds", 3),
            "feature_search_outer_folds": getattr(config.mode, "feature_search_outer_folds", 3),
            "feature_search_top_sets": getattr(config.mode, "feature_search_top_sets", 1),
            "feature_search_refine_rounds": getattr(config.mode, "feature_search_refine_rounds", 0),
            "feature_search_max_evaluations": getattr(config.mode, "feature_search_max_evaluations", 5),
            "feature_search_min_gain": getattr(config.mode, "feature_search_min_gain", 0.005),
            "feature_search_linear_c": getattr(config.mode, "feature_search_linear_c", 1.0),
            "feature_search_linear_max_iterations": getattr(config.mode, "feature_search_linear_max_iterations", 10000),
            "feature_search_non_linear_max_iterations": getattr(config.mode, "feature_search_non_linear_max_iterations", 2000),
            "feature_search_non_linear_alpha": getattr(config.mode, "feature_search_non_linear_alpha", 0.0001),
            "feature_search_non_linear_hidden_layers": ",".join(str(value) for value in getattr(config.mode, "feature_search_non_linear_hidden_layers", [100])),
            "feature_search_non_linear_learning_rate": getattr(config.mode, "feature_search_non_linear_learning_rate", 0.001),
            "feature_search_tree_n_estimators": getattr(config.mode, "feature_search_tree_n_estimators", 100),
            "feature_search_tree_max_depth": getattr(config.mode, "feature_search_tree_max_depth", 0),
            "feature_search_tree_min_samples_leaf": getattr(config.mode, "feature_search_tree_min_samples_leaf", 1),
            "feature_search_tree_class_weight": getattr(config.mode, "feature_search_tree_class_weight", "None"),
            "feature_search_tree_positive_threshold": getattr(config.mode, "feature_search_tree_positive_threshold", 0.5),
            "feature_search_catboost_class_weight": getattr(config.mode, "feature_search_catboost_class_weight", "None"),
            "feature_search_catboost_iterations": getattr(config.mode, "feature_search_catboost_iterations", 400),
            "dark_number_method": config.get_dark_number_method().name,
            "dark_number_alpha": config.get_dark_number_alpha().name,
            "dark_number_target": config.get_dark_number_target(),
            "dark_number_failure_mode": config.should_use_experimental_perturbed_dark_number_fallback(),
        })
        
        # Disabled items:
        self.disable_items(["class_column", "id_column", "data_columns", "text_columns", "categorize_columns"])
        
        # Enable and disable buttons
        self.enable_button("continuation_button")
        self.disable_button("test_profile_button")
        self.update_regression_suite_ready()
        self.sync_feature_selection_controls()
        self.disable_button("start_button")
            
        
    def ensure_saved_model_class_option(self, config: Config) -> None:
        """Keep the persisted target selectable when reusing a saved model.

        Generic class discovery deliberately requires at least two observed target
        values, which is correct for training but excludes prediction-only tables
        whose target column is entirely NULL.  A saved model already identifies
        its target unambiguously, so retain that exact column when it exists in the
        selected source table.  Unrelated zero-cardinality columns remain hidden.
        """
        saved_class = config.get_class_column_name()
        if not saved_class or not self.datatype_dict or saved_class not in self.datatype_dict:
            return

        allowed_types = Config.TEXT_DATATYPES + Config.INT_DATATYPES
        if self.datatype_dict[saved_class] not in allowed_types:
            return

        self.class_column.options = self._without_stale_na_placeholder(
            self._with_saved_options(self.class_column.options, (saved_class,))
        )

    @staticmethod
    def _with_saved_options(options, values) -> tuple:
        """Include persisted values in widget options without discarding current choices."""
        current = list(options)
        for value in values:
            if value not in (None, "") and value not in current:
                current.append(value)
        return tuple(current)

    @staticmethod
    def _without_stale_na_placeholder(options) -> tuple:
        """Drop the initial N/A option once at least one real option is available."""
        current = tuple(options)
        if any(option not in (None, "", "N/A") for option in current):
            return tuple(option for option in current if option != "N/A")
        return current

    def restore_classifier_config(self, config_params: dict) -> None:
        """Restore the visible GUI state for a persisted manual classifier run.

        Current SQL credentials are deliberately left untouched. Repeat Last executes
        the persisted Config objects directly, while this synchronization also makes a
        subsequent ordinary Rerun reflect the same table/model/settings in the GUI.
        """
        connection = config_params["connection"]
        mode = config_params["mode"]
        self.restore_model_parameters(getattr(mode, "model_parameters", {}))
        self.restore_decision_policy(mode)
        io = config_params["io"]
        debug = config_params["debug"]

        configured_columns = list(connection.data_numerical_columns) + list(connection.data_text_columns)
        selected_columns = [
            column for column in self.data_columns.options if column in configured_columns
        ]
        selected_columns.extend(
            column for column in configured_columns if column not in selected_columns
        )
        selected_columns = tuple(selected_columns)
        text_columns = tuple(connection.data_text_columns)
        category_columns = tuple(
            column for column in mode.category_text_columns if column in text_columns
        )

        model_value = Config.DEFAULT_TRAIN_OPTION
        if not mode.train:
            model_value = str(io.model_name)
            if not model_value.endswith(Config.DEFAULT_MODEL_EXTENSION):
                model_value += Config.DEFAULT_MODEL_EXTENSION

        previous_lock = self.eventhandler.lock_observe_1
        self.eventhandler.lock_observe_1 = True
        try:
            self.data_catalogs_dropdown.options = self._with_saved_options(
                self.data_catalogs_dropdown.options, (connection.data_catalog,)
            )
            self.data_tables_dropdown.options = self._with_saved_options(
                self.data_tables_dropdown.options, (connection.data_table,)
            )
            self.models_dropdown.options = self._with_saved_options(
                self.models_dropdown.options, (model_value,)
            )
            self.class_column.options = self._without_stale_na_placeholder(
                self._with_saved_options(self.class_column.options, (connection.class_column,))
            )
            self.id_column.options = self._without_stale_na_placeholder(
                self._with_saved_options(self.id_column.options, (connection.id_column,))
            )
            self.data_columns.options = self._without_stale_na_placeholder(
                self._with_saved_options(self.data_columns.options, selected_columns)
            )
            self.text_columns.options = selected_columns
            self.categorize_columns.options = text_columns
            saved_dark_number_target = str(getattr(mode, "dark_number_target", "") or "")
            if saved_dark_number_target:
                self.dark_number_target.options = self._with_saved_options(
                    self.dark_number_target.options, (saved_dark_number_target,)
                )

            self.update_values({
                "project": config_params["name"],
                "data_catalogs_dropdown": connection.data_catalog,
                "data_tables_dropdown": connection.data_table,
                "class_column": connection.class_column,
                "id_column": connection.id_column,
                "data_columns": selected_columns,
                "text_columns": text_columns,
                "models_dropdown": model_value,
                "train_checkbox": mode.train,
                "predict_checkbox": mode.predict,
                "mispredicted_checkbox": mode.mispredicted,
                "feature_importance_checkbox": bool(mode.train and getattr(mode, "calculate_feature_importance", False)),
                "feature_importance_repeats": getattr(mode, "feature_importance_repeats", 5),
                "feature_selection_mode": getattr(mode, "feature_selection_mode", "manual"),
                "feature_selection_profile": getattr(mode, "feature_selection_profile_id", ""),
                "feature_search_model_type": getattr(mode, "feature_search_model_type", "Linear"),
                "feature_search_preset": getattr(mode, "feature_search_preset", "Quick"),
                "feature_search_tree_estimator": getattr(mode, "feature_search_tree_estimator", "Extra Trees"),
                "feature_search_scaler": getattr(mode, "feature_search_scaler", "STA"),
                "feature_search_oversampling": getattr(mode, "feature_search_oversampling", "NOG"),
                "feature_search_reduction": getattr(mode, "feature_search_reduction", "NOR"),
                "feature_search_trials": getattr(mode, "feature_search_trials", 4),
                "feature_search_sizes": ",".join(str(value) for value in getattr(mode, "feature_search_sizes", [5, 10, 20])),
                "feature_search_inner_folds": getattr(mode, "feature_search_inner_folds", 3),
                "feature_search_outer_folds": getattr(mode, "feature_search_outer_folds", 3),
                "feature_search_top_sets": getattr(mode, "feature_search_top_sets", 1),
                "feature_search_refine_rounds": getattr(mode, "feature_search_refine_rounds", 0),
                "feature_search_max_evaluations": getattr(mode, "feature_search_max_evaluations", 5),
                "feature_search_min_gain": getattr(mode, "feature_search_min_gain", 0.005),
                "feature_search_linear_c": getattr(mode, "feature_search_linear_c", 1.0),
                "feature_search_linear_max_iterations": getattr(mode, "feature_search_linear_max_iterations", 10000),
                "feature_search_non_linear_max_iterations": getattr(mode, "feature_search_non_linear_max_iterations", 2000),
                "feature_search_non_linear_alpha": getattr(mode, "feature_search_non_linear_alpha", 0.0001),
                "feature_search_non_linear_hidden_layers": ",".join(str(value) for value in getattr(mode, "feature_search_non_linear_hidden_layers", [100])),
                "feature_search_non_linear_learning_rate": getattr(mode, "feature_search_non_linear_learning_rate", 0.001),
                "feature_search_tree_n_estimators": getattr(mode, "feature_search_tree_n_estimators", 100),
                "feature_search_tree_max_depth": getattr(mode, "feature_search_tree_max_depth", 0),
                "feature_search_tree_min_samples_leaf": getattr(mode, "feature_search_tree_min_samples_leaf", 1),
                "feature_search_tree_class_weight": getattr(mode, "feature_search_tree_class_weight", "None"),
                "feature_search_tree_positive_threshold": getattr(mode, "feature_search_tree_positive_threshold", 0.5),
                "feature_search_catboost_class_weight": getattr(mode, "feature_search_catboost_class_weight", "None"),
                "feature_search_catboost_iterations": getattr(mode, "feature_search_catboost_iterations", 400),
                "metas_checkbox": mode.use_metas,
                "dark_numbers_checkbox": getattr(mode, "calculate_dark_numbers", mode.mispredicted),
                "dark_number_method": DarkNumberMethod.from_config_value(
                    getattr(mode, "dark_number_method", "LINEAR")
                ).name,
                "dark_number_alpha": DarkNumberAlpha.from_config_value(
                    getattr(mode, "dark_number_alpha", "NONE")
                ).name,
                "dark_number_target": str(getattr(mode, "dark_number_target", "") or ""),
                "dark_number_failure_mode": getattr(
                    mode, "experimental_perturbed_dark_number_fallback", True
                ),
                "algorithm_dropdown": tuple(mode.algorithm.get_abbreviations()),
                "preprocess_dropdown": tuple(mode.preprocessor.get_abbreviations()),
                "reduction_dropdown": tuple(mode.feature_selection.get_abbreviations()),
                "scoremetric_dropdown": mode.scoring.name,
                "oversampler_dropdown": tuple(sampling_names(getattr(mode, "oversamplers", None), Oversampling, mode.oversampler)),
                "undersampler_dropdown": tuple(sampling_names(getattr(mode, "undersamplers", None), Undersampling, mode.undersampler)),
                "testdata_slider": int(round(float(mode.test_size) * 100)),
                "iterations_slider": mode.max_iterations,
                "encryption_checkbox": mode.hex_encode,
                "categorize_checkbox": mode.use_categorization,
                "categorize_columns": category_columns,
                "filter_checkbox": mode.use_stop_words,
                "ngram_range_dropdown": mode.ngram_range.name,
                "data_limit": debug.data_limit,
                "show_info_checkbox": io.verbose,
                "num_variables": len(selected_columns),
            })
        finally:
            self.eventhandler.lock_observe_1 = previous_lock

        # Repeat Last executes immediately. Keep restored values visible but read-only
        # during execution; set_rerun() reopens the intended controls afterwards.
        self.deactivate_section("data")
        self.deactivate_section("feature_selection")
        self.deactivate_section("classifier")

    def activate_section(self, name: str) -> None:
        """ This should probably be a toggle, but for the moment we'll do it this way"""

        if section := self.sections.get(name):
            for item in section:
                if item.endswith("_button"):
                    self.enable_button(item)
                else:
                    widget = self.get_item_or_error(item)
                    if hasattr(widget, "disabled"):
                        setattr(widget, "disabled", False)

        if name == "classifier":
            self.sync_dark_number_control_state()
            self.sync_feature_importance_control_state()

    def deactivate_section(self, name: str) -> None:
        """ This should probably be a toggle, but for the moment we'll do it this way"""
        if section := self.sections.get(name):
            for item in section:
                if item.endswith("_button"):
                    self.disable_button(item)
                else:
                    widget = self.get_item_or_error(item)
                    if hasattr(widget, "disabled"):
                        setattr(widget, "disabled", True)

    def source_can_be_predicted(self) -> bool:
        """ Prediction is only available if the class-variable have empty values """
        current_class = self.class_column.value
        distribution = self.guihandler.get_class_distribution(self.data_settings, current_class)

        if "NULL" in distribution:
            return True

        if "" in distribution:
            return True

        # It needs to have empty elements (None which is translated to "NULL" or "") to predict
        return False

    def set_checkboxes(self, new_model: bool) -> None:
        """ Sets the values and enabled-state based on new vs trained model"""
        if new_model:
            self.enable_items([
                "train_checkbox",
                "mispredicted_checkbox",
                "dark_numbers_checkbox",
                "dark_number_method",
                "dark_number_alpha",
                "dark_number_target",
                "dark_number_failure_mode",
            ])

            if self.source_can_be_predicted():
                self.enable_items([
                    "predict_checkbox"
                ])
            else: 
                self.disable_items([
                    "predict_checkbox"
                ])
            

            self.update_values({
                "train_checkbox": True,
                "predict_checkbox": False,
                "mispredicted_checkbox": True,
                "feature_importance_checkbox": False,
                "feature_importance_repeats": 5,
                "dark_numbers_checkbox": True,
                "dark_number_method": "LINEAR",
                "dark_number_alpha": "SEPARATED",
                "dark_number_target": "",
                "dark_number_failure_mode": True,
            })
        else:
            self.disable_items([
                "train_checkbox",
                "predict_checkbox",
                "mispredicted_checkbox",
                "dark_numbers_checkbox",
                "dark_number_method",
                "dark_number_alpha",
                "dark_number_target",
                "dark_number_failure_mode",
            ])
            self.update_values({
                "train_checkbox": False,
                "predict_checkbox": True,
                "mispredicted_checkbox": False,
                "feature_importance_checkbox": False,
                "dark_numbers_checkbox": False,
                "dark_number_method": "LINEAR",
                "dark_number_alpha": "NONE",
                "dark_number_target": "",
                "dark_number_failure_mode": True,
            })
            
    def apply_regression_test_profile(self) -> None:
        """Apply the temporary numeric regression profile to classifier widgets."""
        self.update_values(self.REGRESSION_TEST_PROFILE)

    def continuation_button_actions(self) -> None:
        """ Complex actions when button is clicked """
        self.regression_suite_state = False
        self.start_button.description = "Start"
        self.start_button.tooltip = "Run the classifier using the current settings"

        new_model = self.new_model
        self.set_checkboxes(new_model)
        default_enable_list = [
            "data_limit",
            "show_info_checkbox",
            "start_button"
        ]
        new_model_list = []
        if new_model:
            new_model_list = [
                "algorithm_dropdown",
                "preprocess_dropdown",
                "reduction_dropdown",
                "scoremetric_dropdown", 
                "oversampler_dropdown",
                "undersampler_dropdown",
                "testdata_slider",
                "iterations_slider",
                "encryption_checkbox",
                "categorize_checkbox",
                "categorize_columns",
                "filter_checkbox",
                "ngram_range_dropdown"
            ]
        
            self.categorize_columns.options = self.text_columns.value
        
        self.update_data_limit()
        self.enable_items(default_enable_list + new_model_list)

    def regression_suite_button_actions(self) -> None:
        """Enable classifier controls without touching manual dataset validation."""
        self.regression_suite_state = True
        self.project.value = self.REGRESSION_SUITE_PROJECT

        # Do not call set_checkboxes() here. That helper asks whether the currently
        # selected dataset can be predicted, which queries the manual class column.
        # A suite run intentionally has no manual dataset/class selection yet.
        self.enable_items(["train_checkbox"])
        self.disable_items([
            "predict_checkbox", "mispredicted_checkbox", "metas_checkbox",
            "dark_numbers_checkbox", "dark_number_method", "dark_number_alpha",
            "dark_number_target", "dark_number_failure_mode",
            "feature_importance_checkbox", "feature_importance_repeats",
        ])
        self.update_values({
            "train_checkbox": True,
            "predict_checkbox": False,
            "mispredicted_checkbox": False,
            "metas_checkbox": False,
            "dark_numbers_checkbox": True,
            "dark_number_method": "LINEAR",
            "dark_number_alpha": "NONE",
            "dark_number_target": "",
            "dark_number_failure_mode": True,
            "feature_importance_checkbox": False,
        })
        self.categorize_columns.options = ()

        self.enable_items([
            "show_info_checkbox",
            "start_button",
            "algorithm_dropdown",
            "preprocess_dropdown",
            "reduction_dropdown",
            "scoremetric_dropdown",
            "oversampler_dropdown",
            "undersampler_dropdown",
            "testdata_slider",
            "iterations_slider",
            "encryption_checkbox",
            "categorize_checkbox",
            "categorize_columns",
            "filter_checkbox",
            "ngram_range_dropdown",
        ])
        self.start_button.description = "Run suite"
        self.start_button.tooltip = "Run the regression suite across configured datasets"

    def repeat_last_run_button_actions(self) -> None:
        """Run the persisted previous configuration without rebuilding it from current widgets."""
        self.regression_suite_state = False
        self.output.clear_output(wait=True)
        self.clear_mispredicted_output()
        self.guihandler.repeat_last_classifier_run(
            output=self.output,
            sql_username=self.sql_username.value,
            sql_password=self.sql_password.value,
        )

    def start_button_actions(self) -> None:
        """ Complex actions when button is clicked """
        self.guihandler._classifier_diagnostic("start button action entered")
        if self.rerun_state:
            if not self.regression_suite_state:
                self.update_class_summary()
        else:
            self.rerun_state = True
        
        # The disk log must reveal whether a blank Script output is caused by
        # clearing the widget, constructing settings, or executing the runner.
        self.guihandler._classifier_diagnostic("Script output widget: clear begin")
        self.output.clear_output(wait=True)
        self.clear_mispredicted_output()
        self.guihandler._classifier_diagnostic("Script output widget: clear end")
        if self.regression_suite_state:
            config_params = self.get_regression_suite_base_config_params()
            self.guihandler.run_regression_suite(base_config_params=config_params, output=self.output)
        else:
            config_params = self.get_config_params()
            self.guihandler.run_classifier(config_params=config_params, output=self.output)

    def set_rerun(self) -> None:
        """ Updates buttons and checkboxes for doing a rerun with the settings """
        self.enable_items(["start_button", "show_info_checkbox"])
        self.update_repeat_last_run_ready()

        if self.regression_suite_state:
            self.start_button.description = "Rerun suite"
            self.start_button.tooltip = "Rerun the regression suite"
        else:
            self.start_button.description = "Rerun"
            self.start_button.tooltip = "Rerun the classifier with the same setting as last time"
        
    # Each GridBox cell creates a comm-backed ipywidget. Limit the initial view to
    # keep large results from overwhelming the Voilà/Jupyter frontend.
    MISPREDICTED_RENDER_LIMIT = 40
    MISPREDICTED_WIDGET_LIMIT = 640

    def clear_mispredicted_output(self) -> None:
        """Clear the preview and release comm-backed widgets from the previous run.

        Clearing an Output widget removes the visible view but does not close the
        child widget models that were created for the reclassification table. On
        repeated runs those stale models can otherwise accumulate in the frontend.
        """
        logger = logging.getLogger(__name__)
        grid = self.mispredicted_gridbox
        children = tuple(getattr(grid, "children", ()))
        display_widgets = tuple(getattr(self, "_mispredicted_display_widgets", ()))

        try:
            # Remove the old view before closing child models referenced by it.
            self.mispredicted_output.clear_output(wait=False)
        except Exception:
            logger.exception("Could not clear the previous mispredicted Output widget")

        try:
            grid.children = ()
        except Exception:
            logger.exception("Could not detach previous mispredicted grid children")

        closed = 0
        for widget in (*children, *display_widgets):
            try:
                widget.close()
                closed += 1
            except Exception:
                # Cleanup diagnostics must never prevent a new model run.
                logger.debug("Could not close a stale mispredicted widget", exc_info=True)
        self._mispredicted_display_widgets = ()

        if children or display_widgets:
            logger.info(
                "Mispredicted widget cleanup: detached=%d display_widgets=%d closed=%d",
                len(children), len(display_widgets), closed,
            )

    def handle_mispredicted(self, mispredicted: DataFrame, unique_classes: list[str]) -> bool:
        """Render a bounded, editable preview; never fail the completed ML run.

        Full results remain in the classifier output. Rendering is intentionally
        capped because one comm-backed widget is allocated per table cell.
        """
        total_rows = len(mispredicted)
        total_columns = len(mispredicted.columns) + 2
        logger = logging.getLogger(__name__)
        widget_budget = max(1, int(getattr(self, "MISPREDICTED_WIDGET_LIMIT", 640)))
        row_limit = max(0, int(getattr(self, "MISPREDICTED_RENDER_LIMIT", 40)))
        grid = self.mispredicted_gridbox
        items = []
        display_items = []

        # This also makes direct/programmatic calls safe; GUI button actions clear
        # earlier so stale widget models do not remain alive throughout training.
        Widgets.clear_mispredicted_output(self)

        # Headers also consume widget models. If a non-empty preview cannot fit
        # both its header and at least one data row, skip the interactive grid
        # instead of asking the frontend to render an effectively unusable model
        # graph. The full result remains in the classifier output.
        minimum_preview_widgets = total_columns * (2 if total_rows else 1)
        if minimum_preview_widgets > widget_budget:
            logger.warning(
                "Mispredicted widget skipped: rows=%d columns=%d minimum_widgets=%d "
                "widget_budget=%d",
                total_rows, total_columns, minimum_preview_widgets, widget_budget,
            )
            try:
                with self.mispredicted_output:
                    print(
                        "Interactive reclassification preview skipped because its header and "
                        f"first row require {minimum_preview_widgets} grid widgets, exceeding "
                        f"the frontend widget budget of {widget_budget}. Training results are "
                        "unaffected; use the full "
                        "classifier output for inspection."
                    )
            except Exception:
                logger.exception("Could not display the mispredicted budget fallback message")
            return False

        max_rows_by_widget_budget = max(0, widget_budget // total_columns - 1)
        displayed_rows = min(total_rows, row_limit, max_rows_by_widget_budget)
        try:
            preview = mispredicted.iloc[:displayed_rows]
            headers = [str(mispredicted.index.name or "Index"), *map(str, mispredicted.columns), "Reclassify as"]
            items = [widgets.Label(value=label).add_class("header") for label in headers]
            # iterrows, not loc[index]: duplicate index labels must not turn a row
            # into an entire DataFrame or duplicate rendered output.
            for row_index, row in preview.iterrows():
                row_name = str(row_index)
                items.append(widgets.Label(value=row_name))
                items.extend(self.get_text_widget(value) for value in row)
                items.append(self.reclassify_dropdown(unique_classes, row_name))

            grid.layout.grid_template_columns = f"repeat({total_columns}, max-content)"
            grid.children = tuple(items)

            # Static decoration does not need a widget model/comm. In Voilà a
            # completed run can otherwise leave the frontend resolving several
            # short-lived HTML widget models at the same time as the large grid,
            # which may surface as "Error displaying widget: model not found".
            display_items = [
                self.create_inpage_styles("mispredicted"),
                DisplayHTML("<h3>Reclassification table for most mispredicted</h3>"),
            ]
            if total_rows > displayed_rows:
                display_items.append(DisplayHTML(
                    f"<p>Displaying the first {displayed_rows} of {total_rows} rows "
                    "to limit frontend widget load. Full results are available "
                    "in the classifier output.</p>"
                ))
            # Revision 163 may leave comm-backed decorations from an older live
            # kernel, so clear_mispredicted_output still knows how to close them.
            # New renders deliberately keep no comm-backed decoration widgets.
            self._mispredicted_display_widgets = ()

            with self.mispredicted_output:
                for item in display_items:
                    if item is not None:
                        display(item)
                display(grid)
            logger.info(
                "Mispredicted widget rendered: rows=%d/%d columns=%d grid_widgets=%d budget=%d",
                displayed_rows, total_rows, total_columns, len(items), widget_budget,
            )
            return True
        except Exception:
            logger.exception("Mispredicted widget rendering failed: rows=%d columns=%d "
                             "attempted_rows=%d", total_rows, total_columns, displayed_rows)
            try:
                grid.children = ()
            except Exception:
                logger.debug("Could not detach failed mispredicted grid children", exc_info=True)
            for widget in items:
                try:
                    widget.close()
                except Exception:
                    logger.debug("Could not close failed mispredicted widget", exc_info=True)
            self._mispredicted_display_widgets = ()
            # A failed display must not mask a completed model training run.
            try:
                with self.mispredicted_output:
                    print(
                        "Could not display the interactive reclassification table. "
                        "Training results are unaffected; see the server log for details."
                    )
            except Exception:
                logger.exception("Could not display the mispredicted fallback message")
            return False

    def create_inpage_styles(self, section: str) -> DisplayHTML | None:
        """ 
            Defined in default_settings.json it uses the section to get a dict
            Section (as a class) scopes the selectorsss
        """
        styles_dict = self.settings.get("styles")
        section_styles = styles_dict.get(section)
        if not section_styles:
            print(f"Section {section} does not exist in settings.styles")
            return

        styles = []
        for selector, properties in section_styles.items():
            scoped_selector = f".{section} {selector}" #ex .mispredicted .widget-dropdown
            
            scoped_styles = "; ".join(properties) # ex: width: fit-content; border: blue
            
            styles.append(f"{scoped_selector} {{ {scoped_styles} }}")

        html_string = Helpers.html_wrapper("style", " ".join(styles))
        return DisplayHTML(html_string)


    def reclassify_dropdown(self, unique_classes: list[str], row_name: str) -> widgets.Dropdown:
        dropdown_options = [('Keep', 0)] + [(label, (label, row_name)) for label in unique_classes]
        reclassify_dropdown = widgets.Dropdown(options = dropdown_options, value = 0, description = '', disabled = False)
        reclassify_dropdown.observe(self.eventhandler.reclassify_dropdown__value,'value')

        return reclassify_dropdown

    
    def correct_mispredicted_data(self, new_class: str, index: int) -> None:
        """ 
            From the reclassify dropdown change 
            Wrapper function since EventHandler does not have access to GUIhandler
        """
        self.guihandler.correct_mispredicted_data(new_class, index)


    def get_text_widget(self, element):
        """ Gets the right type of text widget based on the value """
        if Helpers.is_str(element):
            element_string = str(element)
            string_length = len(element_string)
        
            kwargs = {
                "value": element_string,
                "placeholder": "NULL"
            }
            if string_length > self.TEXT_AREA_MIN_LIMIT:
                return widgets.Textarea(**kwargs)
            
            if string_length > self.TEXT_MIN_LIMIT:
                return widgets.Text(**kwargs)
        
        if Helpers.is_float(element):
            element = round(float(element), 2)
        
        return widgets.Label(str(element))

    
    def get_config_params(self) -> dict:
        """ Creates a Config params object"""
        data_settings = self.data_settings
        params = {
            "connection": Config.Connection(
                odbc_driver = os.environ.get("DEFAULT_ODBC_DRIVER"),
                host = os.environ.get("DEFAULT_HOST"),
                sql_username = data_settings["connection"]["sql_username"],
                sql_password = data_settings["connection"]["sql_password"],
                class_catalog = os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
                class_table = os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
                data_catalog = data_settings["data"]["catalog"],
                data_table = data_settings["data"]["table"],
                class_column = data_settings["columns"]["class"], 
                data_text_columns =  data_settings["columns"]["data_text"], 
                data_numerical_columns = data_settings["columns"]["data_numerical"], 
                id_column = data_settings["columns"]["id"]
            ),
            "mode": Config.Mode(
                train = self.train_checkbox.value,
                predict = self.predict_checkbox.value,
                mispredicted = self.mispredicted_checkbox.value,
                calculate_feature_importance = self.train_checkbox.value and self.feature_importance_checkbox.value,
                feature_importance_repeats = self.feature_importance_repeats.value,
                feature_selection_mode = self.feature_selection_mode.value,
                feature_selection_profile_id = self.feature_selection_profile.value,
                feature_search_model_type = self.feature_search_model_type.value,
                feature_search_preset = self.feature_search_preset.value,
                feature_search_tree_estimator = self.feature_search_tree_estimator.value,
                feature_search_scaler = self.feature_search_scaler.value,
                feature_search_oversampling = self.feature_search_oversampling.value,
                feature_search_reduction = self.feature_search_reduction.value,
                feature_search_trials = self.feature_search_trials.value,
                feature_search_sizes = self._parse_positive_int_list(self.feature_search_sizes.value, "Subset sizes"),
                feature_search_inner_folds = self.feature_search_inner_folds.value,
                feature_search_outer_folds = self.feature_search_outer_folds.value,
                feature_search_top_sets = self.feature_search_top_sets.value,
                feature_search_refine_rounds = self.feature_search_refine_rounds.value,
                feature_search_max_evaluations = self.feature_search_max_evaluations.value,
                feature_search_min_gain = self.feature_search_min_gain.value,
                feature_search_linear_c = self.feature_search_linear_c.value,
                feature_search_linear_max_iterations = self.feature_search_linear_max_iterations.value,
                feature_search_non_linear_max_iterations = self.feature_search_non_linear_max_iterations.value,
                feature_search_non_linear_alpha = self.feature_search_non_linear_alpha.value,
                feature_search_non_linear_hidden_layers = self._parse_positive_int_list(self.feature_search_non_linear_hidden_layers.value, "MLP hidden layers"),
                feature_search_non_linear_learning_rate = self.feature_search_non_linear_learning_rate.value,
                feature_search_tree_n_estimators = self.feature_search_tree_n_estimators.value,
                feature_search_tree_max_depth = self.feature_search_tree_max_depth.value,
                feature_search_tree_min_samples_leaf = self.feature_search_tree_min_samples_leaf.value,
                feature_search_tree_class_weight = self.feature_search_tree_class_weight.value,
                feature_search_tree_positive_threshold = self.feature_search_tree_positive_threshold.value,
                feature_search_catboost_class_weight = self.feature_search_catboost_class_weight.value,
                feature_search_catboost_iterations = self.feature_search_catboost_iterations.value,
                use_metas = self.metas_checkbox.value,
                use_stop_words = self.filter_checkbox.value,
                ngram_range = NgramRange[self.ngram_range_dropdown.value],
                hex_encode = self.encryption_checkbox.value,
                use_categorization = self.categorize_checkbox.value,
                category_text_columns = list(self.categorize_columns.value),
                test_size = float(self.testdata_slider.value) / 100.0,
                calculate_dark_numbers = self.dark_numbers_checkbox.value,
                dark_number_method = DarkNumberMethod[self.dark_number_method.value],
                dark_number_alpha = DarkNumberAlpha[self.dark_number_alpha.value],
                dark_number_target = str(self.dark_number_target.value or ""),
                experimental_perturbed_dark_number_fallback = bool(self.dark_number_failure_mode.value),
                oversampler = Oversampling[self.oversampler_dropdown.value[0]] if self.oversampler_dropdown.value else Oversampling.NOG,
                undersampler = Undersampling[self.undersampler_dropdown.value[0]] if self.undersampler_dropdown.value else Undersampling.NUG,
                oversamplers = list(self.oversampler_dropdown.value),
                undersamplers = list(self.undersampler_dropdown.value),
                algorithm = AlgorithmTuple(self.algorithm_dropdown.value),
                model_parameters = self.get_model_parameters(),
                decision_policy = self.decision_policy_widget.value,
                probability_threshold = self.probability_threshold_widget.value,
                threshold_diagnostics = self.threshold_diagnostics_widget.value,
                model_benchmark_150 = self.model_benchmark_150_widget.value,
                holdout_split_mode = self.holdout_split_mode_widget.value,
                validation_group_column = self.validation_group_column_widget.value.strip(),
                grouped_cross_validation = self.grouped_cv_widget.value,
                excluded_predictor_columns = self.excluded_predictors_widget.value,
                preprocessor = PreprocessTuple(self.preprocess_dropdown.value),
                feature_selection = ReductionTuple(self.reduction_dropdown.value),
                num_selected_features = None,
                scoring = ScoreMetric[self.scoremetric_dropdown.value],
                max_iterations = self.iterations_slider.value
            ),
            "io": Config.IO(
                verbose=self.show_info_checkbox.value,
                model_name=self.model_name
            ),
            "mail": Config.Mail(
                smtp_server=os.environ.get("DEFAULT_SMTP_SERVER"),
                notification_email=os.environ.get("DEFAULT_NOTIFICATION_EMAIL")  
            ),
            "debug": Config.Debug(
                on=True,
                data_limit=self.data_limit.value
            ),
            "name": self.project.value,
            "save": True
        }

        return params

    def get_regression_suite_base_config_params(self) -> dict:
        """Build suite settings without depending on the manually selected dataset widgets."""
        params = self.get_config_params()
        connection = params["connection"]
        connection.data_catalog = ""
        connection.data_table = ""
        connection.class_column = ""
        connection.data_text_columns = []
        connection.data_numerical_columns = []
        connection.id_column = ""

        # Suite runs are always fresh training runs. Reclassification/mispredicted
        # output is deliberately disabled because the suite is not a targeted
        # dataset-cleanup workflow. Regr. test and normal runs remain unchanged.
        mode = params["mode"]
        mode.train = True
        mode.predict = False
        mode.mispredicted = False
        mode.use_metas = False
        mode.calculate_feature_importance = False

        params["io"].model_name = self.project.value or "regression"
        return params

    @property
    def new_model(self) -> bool:
        return self.models_dropdown.value == Config.DEFAULT_TRAIN_OPTION

    def update_data_limit(self) -> None:
        """ Gets the number of rows from the database """
        num_rows = self.datalayer.count_data_rows(data_catalog=self.data_catalogs_dropdown.value, data_table=self.data_tables_dropdown.value)
        
        self.data_limit.value = num_rows


    def disable_button(self, name: str) -> None:
        """ Disables and sets to warning style """
        item = self.get_item_or_error(name)
        setattr(item, "disabled", True)
        setattr(item, "button_style", "primary")

    def enable_button(self, name: str) -> None:
        """ Enables and sets to success style """
        item = self.get_item_or_error(name)
        setattr(item, "disabled", False)
        setattr(item, "button_style", "success")

    def disable_items(self, items: list) -> None:
        """ Disable all items in the list """
        for name in items:
            if name.endswith("_button"):
                self.disable_button(name)
            else:
                item = self.get_item_or_error(name)
                if hasattr(item, "disabled"):
                    setattr(item, "disabled", True)

    def enable_items(self, items: list) -> None:
        """ Enable all items in the list """
        for name in items:
            if name.endswith("_button"):
                self.enable_button(name)
            else:
                item = self.get_item_or_error(name)
                if hasattr(item, "disabled"):
                    setattr(item, "disabled", False)

    def update_data_catalog(self) -> None:
        """ Update datalayer with data_catalog """
        # TODO: This doesn't work as expected (changing the catalog) currently because the datalayer *only* looks at class_catalog
        self.guihandler.set_data_catalog(self.data_catalogs_dropdown.value)

    def update_sql_username_and_password(self, username, password) -> None:
        self.datalayer.config.connection.sql_username = username
        self.datalayer.config.connection.sql_password = password
    
    def update_data_catalogs_dropdown(self) -> None:
        catalog_list = self.datalayer.get_catalogs_as_options()
        updates = {
            "options": catalog_list,
            "default": catalog_list[0],
            "disabled": False
        }
        self.update_item("data_catalogs_dropdown", updates)

    def update_data_tables_dropdown(self) -> None:
        """ Update the tables dropdown """
        tables_list = self.datalayer.get_tables_as_options()
        updates = {
            "options": tables_list,
            "default": tables_list[0],
            "disabled": False
        }
        self.update_item("data_tables_dropdown", updates)

    def disable_post_model_dropdowns(self) -> None:
        """ Disables the Catalogs, Tables and Models dropdowns once model is set """
        self.disable_items([
                "data_catalogs_dropdown",
                "data_tables_dropdown"
            ])

    def update_models_dropdown_options(self) -> None:
        preface = ["", Config.DEFAULT_TRAIN_OPTION]
        
        models_list = self.datalayer.get_trained_models_from_files(self.model_path, Config.DEFAULT_MODEL_EXTENSION, preface)

        updates = {
            "options": models_list,
            "default": models_list[0]
        }
        self.update_item("models_dropdown", updates)

    @property
    def class_cardinality_limit(self) -> widgets.IntSlider:
        if not hasattr(self, '_class_cardinality_limit_widget'):
            control = widgets.IntSlider(value=3, min=2, max=30, step=1,
                                        description='Max class values:',
                                        continuous_update=False,
                                        style={'description_width': 'initial'},
                                        layout=widgets.Layout(width='390px'))
            control.observe(self._class_cardinality_changed, names='value')
            self._class_cardinality_limit_widget = control
        return self._class_cardinality_limit_widget

    def _class_cardinality_changed(self, change) -> None:
        if not self.eventhandler.lock_observe_1:
            self.refresh_class_column_options()

    def refresh_class_column_options(self) -> None:
        """Filter precomputed cardinalities without additional SQL on slider changes."""
        if not self.datatype_dict:
            return
        old = self.class_column.value
        ceiling = self.class_cardinality_limit.value
        eligible = [name for name, dtype in self.datatype_dict.items()
                    if dtype in (Config.TEXT_DATATYPES + Config.INT_DATATYPES)
                    and 2 <= self._class_cardinalities.get(name, ceiling + 1) <= ceiling]
        # Changing options fires the existing class_column observer, which updates
        # the unique ID, feature selection, and the remaining data widgets.
        self._updating_column_options = True
        try:
            self.class_column.options = eligible
            self.class_column.value = old if old in eligible else (eligible[0] if eligible else None)
        finally:
            self._updating_column_options = False
        # Update dependent widgets only after class choice is internally consistent.
        if hasattr(self, "unique_id_columns"):
            self.update_id_column()
            self.update_data_columns()
        self.refresh_feature_search_compare_options()
        if self.class_column.value is not None and old != self.class_column.value:
            self.summarise_state = True
            self.update_class_summary()
        elif self.class_column.value is None:
            self.summarise_state = False

    def refresh_feature_search_compare_options(self) -> None:
        """Follow eligible selected source features, preserving a valid selection."""
        if not self.datatype_dict:
            return
        current = self.feature_search_subgroup_column.value
        allowed_types = Config.TEXT_DATATYPES + Config.INT_DATATYPES
        allowed = [name for name in self.data_columns.value
                   if name in self.datatype_dict and self.datatype_dict[name] in allowed_types
                   and name not in (self.class_column.value, self.id_column.value)]
        options = [('', '')] + [(name, name) for name in allowed]
        self.feature_search_subgroup_column.options = options
        self.feature_search_subgroup_column.value = current if current in allowed else ''

    def update_class_id_data_columns(self) -> None:
        if self.data_catalogs_dropdown.value == "" or self.data_tables_dropdown.value == "":
            return
        
        columns = self.datalayer.get_id_columns(self.data_catalogs_dropdown.value, self.data_tables_dropdown.value)
        self.datatype_dict = columns
        columns_list = list(columns.keys())
        eligible = [name for name in columns_list
                    if columns[name] in (Config.TEXT_DATATYPES + Config.INT_DATATYPES)]
        self._class_cardinalities = self.datalayer.get_column_cardinalities(
            self.data_catalogs_dropdown.value, self.data_tables_dropdown.value, eligible)
        integer_columns = [
            var for var in columns_list if self.datatype_dict[var] in Config.INT_DATATYPES
        ]
        self.unique_id_columns = (
            self.datalayer.get_unique_int_id_columns(
                self.data_catalogs_dropdown.value, self.data_tables_dropdown.value, integer_columns
            )
            if integer_columns else []
        )
        self.refresh_class_column_options()
        self.id_column.options = [
            var for var in self.unique_id_columns if var != self.class_column.value
        ]
        # The SQL datatype (int NOT NULL) does not guarantee uniqueness.
        # Explain the exclusion rather than leaving the ID selector empty.
        self.id_column_diagnostic.value = ""
        if "id" in integer_columns and "id" not in self.unique_id_columns:
            try:
                stats = self.datalayer.get_column_uniqueness_stats(
                    self.data_catalogs_dropdown.value, self.data_tables_dropdown.value, "id"
                )
                self.id_column_diagnostic.value = (
                    "<b>Column 'id' is not eligible as a unique identifier.</b> "
                    f"Rows: {stats['total']}; non-NULL: {stats['nonnull']}; "
                    f"distinct: {stats['distinct']}. "
                    "The ID column must be non-NULL and distinct in every row."
                )
            except Exception as exc:
                self.id_column_diagnostic.value = (
                    "<b>Column 'id' failed the uniqueness check.</b> "
                    "Inspect NULL and duplicate values in SQL Server. "
                    + html.escape(str(exc))
                )
        elif not self.id_column.options:
            self.id_column_diagnostic.value = (
                "<b>No eligible ID column.</b> JBG requires a globally unique, "
                "non-NULL integer column. SQL int NOT NULL alone does not guarantee uniqueness."
            )
        if self.id_column.options:
            self.id_column.value = self.id_column.options[0]
        else:
            self.id_column.value = None
        
        self.data_columns.options = \
            [var for var in columns_list if var not in (self.class_column.value, self.id_column.value)]
        self.data_columns.value = []
        self.refresh_feature_search_compare_options()

    @property
    def data_settings(self) -> dict:
        """ Settings on which data to classify """
        return {
            "project": self.project.value,
            "connection": {
                "sql_username": self.sql_username.value,
                "sql_password": self.sql_password.value
            },
            "data": {
                "catalog": self.data_catalogs_dropdown.value,
                "table": self.data_tables_dropdown.value
            },
            "model": self.models_dropdown.value,
            "columns": {
                "class": self.class_column.value,
                "id": self.id_column.value,
                "data_text": self.data_text_columns,
                "data_numerical": self.data_numerical_columns
            }
        }

    @property
    def model_name(self) -> str:
        """ Gets the model name based on values """
        return Config.get_model_name(self.models_dropdown.value, self.project.value)

    def _get_data_columns(self, type: WidgetConstant) -> list:
        """ Gets a list of columns connected to data """
        text_columns = list(self.text_columns.value)
        
        if type == WidgetConstant.DATA_NUMERICAL:
            return [col for col in self.data_columns.value if not col in text_columns]

        if type == WidgetConstant.DATA_TEXT:
            return text_columns

    def update_class_summary(self):
        """ Sets the class summary """
        if not self.summarise_state:
            return
        
        current_class = self.class_column.value
        
        distribution = self.guihandler.get_class_distribution(self.data_settings, current_class)
        self.sync_dark_number_target_options(distribution)
        number_of_categories = len(distribution)
        
        dist_items = '; '.join(f'{key} ({value})' for key, value in distribution.items())
        texts = [

        ]
        texts.append(f"<em>Class column</em>: {current_class}")
        distro_title = "<em>Distribution</em>"
        distro_text = dist_items
        if number_of_categories > TextDataToNumbersConverter.LIMIT_IS_CATEGORICAL:
            distro_text = f"<b>Invalid class column</b> (category count {number_of_categories} higher than limit of {TextDataToNumbersConverter.LIMIT_IS_CATEGORICAL})"
        
        texts.append(f"{distro_title}: {distro_text}")
        total_rows = sum(distribution.values())
        # Match the labelled-row eligibility used by SQLDataLayer.get_data_query:
        # NULL and empty class labels are excluded from training investigations.
        unknown_rows = sum(distribution.get(key, 0) for key in ("", "NULL", None))
        labelled_rows = total_rows - unknown_rows
        texts.append(f"<em>Total rows</em>: {total_rows}")
        texts.append(f"<em>Labelled rows (available for feature selection)</em>: {labelled_rows}")
        if unknown_rows:
            texts.append(f"<em>Unlabelled rows excluded</em>: {unknown_rows}")
        # Refresh the suggested investigation limit only when the dataset identity
        # changes. Preserve a smaller value explicitly chosen by the user.
        identity = (self.data_catalogs_dropdown.value, self.data_tables_dropdown.value, current_class)
        if distribution and labelled_rows > 0 and identity != getattr(self, "_feature_search_rows_source", None):
            self._feature_search_rows_source = identity
            self.feature_search_rows.value = int(labelled_rows)


        self.class_summary.value = "<br>".join(texts)
        

    def update_id_column(self) -> None:
        """ Removes class_column from the id_column options """

        self.id_column.options = [
            var for var in self.unique_id_columns if var != self.class_column.value
        ]
       
    
    def update_data_columns(self) -> None:
        """ Removes id column from data columns """

        self.data_columns.options = \
            [var for var in list(self.datatype_dict.keys()) if var not in (self.class_column.value, self.id_column.value)]


    def options_excluding_selected(self, source: widgets.Select) -> list:
        """ Gets the chosen select and returns all options except the current one """
        
        return [option for option in source.options if option != source.value]

    def update_text_columns(self) -> None:
        """ Sets any text-columns as pickable """
        updates = {
            "options": self.data_columns.value,
            "value": [col for col in self.data_columns.value if self.datatype_dict[col] in Config.TEXT_DATATYPES],
            "disabled": False
        }
        self.update_item("text_columns", updates)

    def update_ready(self) -> None:
        """ Sets whether it can continue into Classifier Settings  """
        num_vars = len(self.data_columns.value)
        self.num_variables.value = num_vars
        if getattr(self, "_feature_search_running", False):
            self.disable_button("continuation_button")
            return

        ready_for_classifier = num_vars > 0 and self.id_column.value and self.class_column.value
        if self.feature_selection_mode.value in ("assisted", "reuse") and not getattr(self, "_feature_search_applied", False):
            ready_for_classifier = False
        if ready_for_classifier:
            self.enable_button("continuation_button")
            if self.new_model:
                self.enable_button("test_profile_button")
            else:
                self.disable_button("test_profile_button")
        else:
            self.disable_button("continuation_button")
            self.disable_button("test_profile_button")

        self.sync_feature_selection_controls()
        self.update_regression_suite_ready()

    def update_regression_suite_ready(self) -> None:
        """Enable connection-level actions as soon as SQL login/catalog discovery is ready."""
        connection_ready = (
            bool(self.sql_username.value)
            and bool(self.sql_password.value)
            and not self.data_catalogs_dropdown.disabled
            and any(option not in ("", "N/A") for option in self.data_catalogs_dropdown.options)
        )
        if connection_ready:
            self.enable_button("regression_suite_button")
        else:
            self.disable_button("regression_suite_button")
        self.update_repeat_last_run_ready(connection_ready=connection_ready)

    def update_repeat_last_run_ready(self, connection_ready: bool | None = None) -> None:
        """Enable repeat only when current credentials and a persisted manual run are available."""
        if connection_ready is None:
            connection_ready = bool(self.sql_username.value) and bool(self.sql_password.value)

        if connection_ready and self.guihandler.has_saved_classifier_run():
            self.enable_button("repeat_last_run_button")
        else:
            self.disable_button("repeat_last_run_button")


    def get_item_or_error(self, name: str) -> widgets.Widget:
        if not hasattr(self, name):
            raise ValueError(f"Item {name} does not exist")

        return getattr(self, name)

    def update_item(self, name: str, updates: dict) -> None:
        item = self.get_item_or_error(name)
        
        for attribute, value in updates.items():
            if hasattr(item, attribute):
                setattr(item, attribute, value)

    def update_values(self, updates: dict) -> None:
        """ Given a dict with item: value, update the values """
        for name, value in updates.items():
            item = self.get_item_or_error(name)
            if hasattr(item, "value"):
                setattr(item, "value", value)
            else:
                print(f"Couldn't set value on {name}")

    @staticmethod
    def _parse_positive_int_list(value: str, label: str) -> list[int]:
        try:
            result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
        except ValueError as ex:
            raise ValueError(f"{label} must be comma-separated positive integers") from ex
        if not result or any(item < 1 for item in result):
            raise ValueError(f"{label} must be comma-separated positive integers")
        return result

    def feature_search_options(self) -> SearchOptions:
        parallel_workers, _ = resolve_feature_search_workers(self.feature_search_max_evaluations.value)
        return SearchOptions(
            trials=self.feature_search_trials.value,
            sizes=tuple(self._parse_positive_int_list(self.feature_search_sizes.value, "Subset sizes")),
            inner_folds=self.feature_search_inner_folds.value,
            outer_folds=self.feature_search_outer_folds.value,
            top_sets=self.feature_search_top_sets.value,
            refine_rounds=self.feature_search_refine_rounds.value,
            max_evaluations=self.feature_search_max_evaluations.value,
            min_gain=self.feature_search_min_gain.value,
            scaler=self.feature_search_scaler.value,
            oversampling=self.feature_search_oversampling.value,
            reduction=self.feature_search_reduction.value,
            C=self.feature_search_linear_c.value,
            max_iterations=self.feature_search_linear_max_iterations.value,
            model_type=self.feature_search_model_type.value,
            non_linear_max_iterations=self.feature_search_non_linear_max_iterations.value,
            non_linear_alpha=self.feature_search_non_linear_alpha.value,
            non_linear_hidden_layers=tuple(self._parse_positive_int_list(
                self.feature_search_non_linear_hidden_layers.value, "MLP hidden layers"
            )),
            non_linear_learning_rate=self.feature_search_non_linear_learning_rate.value,
            tree_estimator=self.feature_search_tree_estimator.value,
            tree_n_estimators=self.feature_search_tree_n_estimators.value,
            tree_max_depth=(self.feature_search_tree_max_depth.value or None),
            tree_min_samples_leaf=self.feature_search_tree_min_samples_leaf.value,
            tree_class_weight=self.feature_search_tree_class_weight.value,
            tree_positive_threshold=self.feature_search_tree_positive_threshold.value,
            parallel_workers=parallel_workers,
            catboost_class_weight=self.feature_search_catboost_class_weight.value,
            catboost_iterations=self.feature_search_catboost_iterations.value,
        )

    def set_feature_search_preset(self, name: str) -> None:
        from JBGFeatureSearchPresets import preset_values
        values = preset_values(name)
        self.eventhandler.lock_observe_1 = True
        try:
            mapping = {"trials": "feature_search_trials", "sizes": "feature_search_sizes",
                       "inner_folds": "feature_search_inner_folds", "outer_folds": "feature_search_outer_folds",
                       "top_sets": "feature_search_top_sets", "refine_rounds": "feature_search_refine_rounds",
                       "max_evaluations": "feature_search_max_evaluations", "min_gain": "feature_search_min_gain"}
            for key, widget_name in mapping.items():
                value = ",".join(map(str, values[key])) if key == "sizes" else values[key]
                getattr(self, widget_name).value = value
        finally:
            self.eventhandler.lock_observe_1 = False
        self.invalidate_feature_search_result()
        self.update_feature_search_fit_budget()

    def update_feature_search_fit_budget(self) -> None:
        try:
            options = self.feature_search_options()
            options.validate()
            self.feature_search_fit_budget.value = (
                f"Expected fit ceiling: {options.planned_fit_upper_bound} | "
                f"Candidate workers: automatic ({options.parallel_workers})"
            )
        except Exception as ex:
            self.feature_search_fit_budget.value = f"Fit budget invalid: {html.escape(str(ex))}"

    def invalidate_feature_search_result(self) -> None:
        self._feature_search_applied = False
        self.feature_search_result = None
        if "feature_search_apply_button" in self.widgets:
            self.disable_button("feature_search_apply_button")
        if "feature_search_save_button" in self.widgets:
            self.disable_button("feature_search_save_button")
        if "feature_search_preview" in self.widgets:
            self.feature_search_preview.value = ""

    def sync_feature_selection_controls(self, preserve_status: bool = False) -> None:
        mode = self.feature_selection_mode.value
        dataset_editable = not self.data_columns.disabled and self.new_model
        self.feature_selection_mode.disabled = not dataset_editable
        assisted = dataset_editable and mode == "assisted"
        reuse = dataset_editable and mode == "reuse"
        self.feature_selection_profile.disabled = not reuse
        self.feature_search_review_button.disabled = not reuse
        self.feature_search_delete_button.disabled = not reuse or not bool(self.feature_selection_profile.value)
        if not reuse:
            self.cancel_feature_search_deletion()
        if reuse:
            self.refresh_feature_search_profiles()
            self.feature_search_delete_button.disabled = not bool(self.feature_selection_profile.value)

        common = [
            self.feature_search_rows, self.feature_search_model_type, self.feature_search_preset, self.feature_search_scaler, self.feature_search_oversampling, self.feature_search_reduction, self.feature_search_trials,
            self.feature_search_sizes, self.feature_search_inner_folds, self.feature_search_outer_folds,
            self.feature_search_top_sets, self.feature_search_refine_rounds,
            self.feature_search_max_evaluations, self.feature_search_min_gain,
        ]
        for widget in common:
            widget.disabled = not assisted
        family = self.feature_search_model_type.value
        for widget in (self.feature_search_linear_c, self.feature_search_linear_max_iterations):
            widget.disabled = not (assisted and family == "Linear")
        for widget in (
            self.feature_search_non_linear_max_iterations, self.feature_search_non_linear_alpha,
            self.feature_search_non_linear_hidden_layers, self.feature_search_non_linear_learning_rate,
        ):
            widget.disabled = not (assisted and family == "Non-linear")
        for widget in (
            self.feature_search_tree_estimator, self.feature_search_tree_n_estimators, self.feature_search_tree_max_depth,
            self.feature_search_tree_min_samples_leaf, self.feature_search_tree_class_weight, self.feature_search_tree_positive_threshold,
        ):
            widget.disabled = not (assisted and family == "Tree")

        for widget in (self.feature_search_catboost_class_weight, self.feature_search_catboost_iterations):
            widget.disabled = not (assisted and family == "CatBoost")
        if assisted and family == "CatBoost":
            # CatBoost operates on original mixed-type features, so do not apply PCA/SMOTE.
            self.feature_search_oversampling.value = 'NOG'
            self.feature_search_reduction.value = 'NOR'
            self.feature_search_oversampling.disabled = True
            self.feature_search_reduction.disabled = True
            self.feature_search_scaler.disabled = True  # CatBoost uses raw numeric values.

        ready = assisted and bool(self.data_columns.value) and bool(self.class_column.value) and bool(self.id_column.value)
        running = getattr(self, "_feature_search_running", False)
        self.feature_search_investigate_button.disabled = running or not ready
        if running or (mode in ("assisted", "reuse") and not getattr(self, "_feature_search_applied", False)):
            self.disable_button("continuation_button")
        if not ready:
            self.feature_search_investigate_button.button_style = "primary"
        else:
            self.feature_search_investigate_button.button_style = "success"
        self.feature_search_apply_button.disabled = not (
            (assisted and self.feature_search_result and self.feature_search_result.get("status") == "completed" and self.feature_search_result.get("recommendation") is not None)
            or (reuse and self._saved_recommendation_issues() == [])
        )
        self.feature_search_save_button.disabled = not (
            assisted and self.feature_search_result and self.feature_search_result.get("status") == "completed" and self.feature_search_result.get("recommendation") is not None
        )
        if mode in ("assisted", "reuse") and not getattr(self, "_feature_search_applied", False):
            self.disable_button("continuation_button")
        if not preserve_status:
            if mode == "manual":
                self.feature_search_status.value = "Manual data settings; no investigation will run."
            elif mode == "reuse":
                self.feature_search_status.value = "Select and review a saved recommendation, then apply explicitly. Exploratory results are not independently validated."
            elif self.feature_search_result is None:
                self.feature_search_status.value = "Ready for an exploratory training-only investigation. The JBG holdout remains reserved."
        self.update_feature_search_fit_budget()

    def _feature_search_event(self, event: dict) -> None:
        if event.get("event") == "fit_started":
            self.feature_search_status.value = (
                f"Investigating {html.escape(str(event.get('model_type', 'model')))}: attempted fits "
                f"{event['attempted_fits']}/{event['planned_fit_upper_bound']}"
            )
        elif event.get("event") == "candidate_completed":
            score = "failed" if event.get("error") else f"MCC {event['mean_mcc']:.4f}"
            self.feature_search_status.value = (
                f"{html.escape(event['scope'])}: candidate {event['candidate']}/"
                f"{event['max_evaluations']}, {event['n_columns']} columns, {score}; "
                f"{event.get('completed_candidates', 0)} checkpoints, "
                f"{event.get('elapsed_seconds', 0) / 60:.1f} min this session"
            )

    @property
    def feature_search_pause_button(self):
        if not hasattr(self, '_feature_search_pause_widget'):
            self._feature_search_pause_widget = widgets.Button(description='Pause', icon='pause', disabled=True)
            self._feature_search_pause_widget.on_click(lambda _: self.pause_feature_selection_investigation())
        return self._feature_search_pause_widget

    @property
    def feature_search_resume_button(self):
        if not hasattr(self, '_feature_search_resume_widget'):
            self._feature_search_resume_widget = widgets.Button(description='Resume', icon='play')
            self._feature_search_resume_widget.on_click(lambda _: self.resume_feature_selection_investigation())
        return self._feature_search_resume_widget

    def pause_feature_selection_investigation(self):
        if getattr(self, '_feature_search_running', False) and hasattr(self, '_feature_search_pause_event'):
            self._feature_search_pause_event.set()
            self.feature_search_pause_button.disabled = True
            self.feature_search_status.value = 'Pause requested; finishing the current evaluation before checkpointing.'

    def _feature_search_recovery_choices(self):
        """List all incomplete studies explicitly; never guess based on mtime."""
        root = Path(__file__).resolve().parents[2]
        directories = set((root / 'output').glob('feature_search_gui_*'))
        directories.update((root / 'src' / 'output').glob('feature_search_gui_*'))
        choices = []
        for directory in sorted(directories, key=lambda value: value.name, reverse=True):
            summary = directory / 'summary.json'
            if not summary.is_file() or not (directory / 'development_snapshot.pkl').is_file():
                continue
            try:
                data = json.loads(summary.read_text(encoding='utf-8'))
                if data.get('status') not in ('paused', 'interrupted', 'running', 'failed'):
                    continue
                context = data.get('provenance', {}).get('context', {})
                tables = context.get('source_tables') or ['unknown table']
                description = (f"{directory.name} | {tables[0]} | "
                               f"{data.get('options', {}).get('model_type', '?')} | "
                               f"{data.get('attempted_fits', '?')}/"
                               f"{data.get('planned_fit_upper_bound', '?')} fits | "
                               f"{data.get('status')}")
                choices.append((description, str(directory.resolve())))
            except (ValueError, OSError, TypeError, KeyError):
                continue
        return choices

    @property
    def feature_search_recovery_selector(self):
        if not hasattr(self, '_feature_search_recovery_selector_widget'):
            self._feature_search_recovery_selector_widget = widgets.Dropdown(
                options=[('Select a saved investigation (Refresh to list)', '')],
                description='Recover run:', layout=widgets.Layout(width='480px'),
                style={'description_width': 'initial'},
            )
            self.refresh_feature_search_recovery_choices()
        return self._feature_search_recovery_selector_widget

    def refresh_feature_search_recovery_choices(self):
        selector = self.feature_search_recovery_selector if hasattr(self, '_feature_search_recovery_selector_widget') else None
        if selector is None:
            return
        previous = selector.value
        options = self._feature_search_recovery_choices()
        selector.options = [('Select a saved investigation', '')] + options
        if previous and previous in [value for _, value in options]:
            selector.value = previous

    @property
    def feature_search_recovery_refresh_button(self):
        if not hasattr(self, '_feature_search_recovery_refresh_widget'):
            button = widgets.Button(description='Refresh investigations', icon='refresh')
            button.on_click(lambda _: self.refresh_feature_search_recovery_choices())
            self._feature_search_recovery_refresh_widget = button
        return self._feature_search_recovery_refresh_widget

    def resume_feature_selection_investigation(self):
        checkpoint = self.feature_search_recovery_selector.value
        if not checkpoint:
            self.feature_search_status.value = 'Select a saved investigation to resume, then click Resume.'
            return
        if checkpoint not in [value for _, value in self._feature_search_recovery_choices()]:
            self.feature_search_status.value = 'Selected checkpoint is no longer available; refresh the list.'
            return
        self._launch_feature_selection_investigation(resume_directory=Path(checkpoint))

    @property
    def feature_search_investigation_mode(self):
        if not hasattr(self, '_feature_search_investigation_mode_widget'):
            self._feature_search_investigation_mode_widget = widgets.ToggleButtons(
                options=[('Diagnostics only', 'diagnostics_only'),
                         ('Feature selection + diagnostics', 'selection_plus_diagnostics')],
                value='selection_plus_diagnostics', description='Investigation:',
                style={'description_width': 'initial'})
        return self._feature_search_investigation_mode_widget

    @property
    def feature_search_subgroup_column(self):
        if not hasattr(self, '_feature_search_subgroup_column_widget'):
            self._feature_search_subgroup_column_widget = widgets.Dropdown(
                options=[('', '')], value='', description='Compare by:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='330px'))
        return self._feature_search_subgroup_column_widget

    @property
    def feature_search_result_tabs(self):
        if not hasattr(self, '_feature_search_result_tabs_widget'):
            self._feature_search_diagnostics_html = widgets.HTML()
            self._feature_search_details_html = widgets.HTML()
            tabs = widgets.Tab(children=[self.feature_search_preview,
                                        self._feature_search_diagnostics_html,
                                        self._feature_search_details_html])
            for i, name in enumerate(('Recommendation', 'Diagnostics', 'Search details')):
                tabs.set_title(i, name)
            self._feature_search_result_tabs_widget = tabs
        return self._feature_search_result_tabs_widget

    def _display_feature_search_diagnostics(self, result):
        reports = result.get('diagnostics', {})
        sections = []
        for name, report in reports.items():
            rows = ''.join('<tr>' + ''.join(f'<td>{v:.4f}</td>' if isinstance(v, float)
                                               else f'<td>{v}</td>' for v in
                                               (r['threshold'], r['precision'], r['recall'],
                                                r['specificity'], r['mcc'])) + '</tr>'
                           for r in report['threshold_analysis'])
            sections.append(
                f'<h4>{html.escape(name)}</h4>'
                f"ROC AUC: {report['roc_auc']:.4f}; Average precision: {report['average_precision']:.4f}; "
                f"Prevalence: {report['prevalence']:.4f}; Brier: {report['brier_score']:.4f}"
                '<p>Descriptive out-of-fold thresholds, NOT independently optimized:</p>'
                '<table><tr><th>Cutoff</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>MCC</th></tr>'
                + rows + '</table>'
                '<p>Calibration bins (mean predicted probability / observed fraction / count):</p>'
                + '<ul>' + ''.join(f"<li>{r['mean_probability']:.3f} / {r['observed_rate']:.3f} / {r['count']}</li>"
                                      for r in report['calibration_bins']) + '</ul>')
        for name, report in reports.items():
            if 'subgroups' not in report:
                continue
            heading = html.escape(str(report['subgroup_column']))
            rows = []
            for item in report['subgroups']:
                def num(value):
                    return '—' if value is None else f'{value:.4f}'
                cells = [html.escape(item['value']), str(item['rows']), str(item['positives']),
                         num(item['prevalence']), num(item['roc_auc']),
                         num(item['average_precision']), html.escape(item['status'])]
                rows.append('<tr>' + ''.join('<td>' + c + '</td>' for c in cells) + '</tr>')
            sections.append('<h4>Subgroups by ' + heading + ' — ' + html.escape(name) + '</h4>'
                            '<table><tr><th>Value</th><th>Rows</th><th>Positive</th>'
                            '<th>Prevalence</th><th>ROC AUC</th><th>AP</th><th>Status</th></tr>'
                            + ''.join(rows) + '</table>')
        self._feature_search_diagnostics_html.value = ''.join(sections) or 'No diagnostics available.'
        self._feature_search_details_html.value = (
            'Completed fits: ' + str(result.get('attempted_fits', 0)) +
            '; output: ' + html.escape(result.get('output_directory', '')))

    def run_feature_selection_investigation(self) -> None:
        self._launch_feature_selection_investigation()

    def _launch_feature_selection_investigation(self, resume_directory=None):
        import threading
        if getattr(self, '_feature_search_running', False):
            return
        # Snapshot GUI state on the kernel callback thread. Workers must not
        # read mutable widget values while an investigation is running.
        try:
            if resume_directory is None:
                params = self.get_config_params()
                options = self.feature_search_options()
                options.validate()
                row_limit = int(self.feature_search_rows.value)
            else:
                params, options, row_limit = None, None, None
        except Exception as ex:
            self.feature_search_status.value = 'Could not start investigation: ' + html.escape(str(ex))
            return
        investigation_mode = self.feature_search_investigation_mode.value if resume_directory is None else None
        subgroup_column = (self.feature_search_subgroup_column.value or None) if resume_directory is None else None
        self._feature_search_running = True
        self._feature_search_pause_event = threading.Event()
        self.feature_search_pause_button.disabled = False
        self.feature_search_resume_button.disabled = True
        self.feature_search_investigate_button.disabled = True
        self._feature_search_job = threading.Thread(
            target=self._run_feature_selection_investigation_worker,
            args=(params, options, row_limit, resume_directory, investigation_mode, subgroup_column), daemon=True, name='JBGFeatureInvestigation')
        self._feature_search_job.start()

    def _run_feature_selection_investigation_worker(self, params, options, row_limit, resume_directory=None, investigation_mode=None, subgroup_column=None) -> None:
        self.disable_button("continuation_button")
        self.feature_search_investigate_button.disabled = True
        self.invalidate_feature_search_result()
        try:
            self.feature_search_investigate_button.disabled = True
            self.feature_search_status.value = "Starting training-only investigation..."
            if resume_directory is not None:
                result = self.guihandler.resume_saved_feature_selection_investigation(
                    resume_directory, on_event=self._feature_search_event,
                    pause_event=self._feature_search_pause_event,
                )
            else:
                result = self.guihandler.run_feature_selection_investigation(
                    params, options, on_event=self._feature_search_event,
                    row_limit=row_limit, pause_event=self._feature_search_pause_event,
                    investigation_mode=investigation_mode, subgroup_column=subgroup_column,
                )
            self.feature_search_result = result
            self._display_feature_search_diagnostics(result)
            if result.get('investigation_mode') == 'diagnostics_only':
                self.feature_search_preview.value = ('<b>Diagnostics only: no feature selection or '
                    'applicable recommendation was produced.</b><br>Results: <code>' +
                    html.escape(result.get('output_directory', '')) + '</code>')
                self.feature_search_status.value = 'Classification diagnostics completed.'
                return
            selected = result["nested_validation"]["selected"]
            baseline = result["nested_validation"]["full_baseline"]
            columns = result["recommendation"]["selected_columns"]
            retained = columns == list(result.get("original_columns", ()))
            outcome = (
                "Full-column baseline retained; no improvement found within this search budget."
                if retained else
                f"Candidate recommendation contains {len(columns)} of {len(result.get('original_columns', ())) } original columns."
            )
            is_multiclass = result.get("evaluation", {}).get("type") == "multiclass"
            macro_report = (
                f"Macro precision {selected['macro_precision']:.3f}, "
                f"macro recall {selected['macro_recall']:.3f}, macro F1 {selected['macro_f1']:.3f}.<br>"
                if is_multiclass else ""
            )
            self.feature_search_preview.value = (
                f"<b>{html.escape(outcome)}</b><br>"
                f"Status: candidate; model: {html.escape(result['investigative_model']['model_type'])}. "
                f"Nested selected MCC {selected['mean_mcc']:.4f} vs full {baseline['mean_mcc']:.4f}; "
                f"Target {html.escape(str(result['target']))}: precision {selected['precision']:.3f}, "
                f"recall {selected['recall']:.3f}, F1 {selected['f1']:.3f}; "
                f"TP {selected['tp']}, FP {selected['fp']}, FN {selected['fn']}.<br>"
                f"{macro_report}"
                f"Recommended columns: {html.escape(', '.join(columns))}<br>"
                f"Results: <code>{html.escape(result.get('output_directory', ''))}</code>"
            )
            if result.get('baseline_warning'):
                self.feature_search_preview.value += '<br><b>Validation warning:</b> ' + html.escape(result['baseline_warning'])
            self.feature_search_status.value = "Investigation completed. Review the candidate before applying it."
        except Exception as ex:
            # Preserve the failure in the GUI and emit a traceback in the server log.
            import logging
            from JBGFeatureSearch import InvestigationPaused
            if not isinstance(ex, InvestigationPaused):
                logging.getLogger(__name__).exception("Feature-selection investigation failed")
            if isinstance(ex, InvestigationPaused):
                self.feature_search_status.value = 'Investigation paused. Use Resume to continue.'
            else:
                self.feature_search_status.value = f"Investigation failed: {html.escape(type(ex).__name__ + ': ' + str(ex))}"
        finally:
            # update_ready() also synchronizes controls and otherwise replaces the
            # diagnostic status with the generic Ready prompt.
            final_status = self.feature_search_status.value
            self._feature_search_running = False
            self.feature_search_pause_button.disabled = True
            self.feature_search_resume_button.disabled = False
            self.update_ready()
            self.sync_feature_selection_controls(preserve_status=True)
            self.feature_search_status.value = final_status

    @staticmethod
    def _feature_profile_store():
        return Path(__file__).resolve().parents[2] / "output" / "feature_search_profiles"

    def cancel_feature_search_deletion(self) -> None:
        self._pending_feature_profile_delete = None
        if "feature_search_delete_button" in self.widgets:
            self.feature_search_delete_button.description = "Delete"
            self.feature_search_delete_button.button_style = "danger"

    def delete_feature_search_candidate(self) -> None:
        if self.feature_selection_mode.value != "reuse" or getattr(self, "_feature_search_running", False):
            return
        profile_id = self.feature_selection_profile.value
        record = find_candidate(self._feature_profile_store(), profile_id)
        if record is None:
            self.cancel_feature_search_deletion()
            self.feature_search_status.value = "Select an existing recommendation to delete."
            return
        if getattr(self, "_pending_feature_profile_delete", None) != profile_id:
            self._pending_feature_profile_delete = profile_id
            self.feature_search_delete_button.description = "Confirm delete"
            self.feature_search_status.value = (
                "Click Confirm delete to permanently delete ONLY this saved recommendation: "
                + html.escape(profile_display_name(record))
                + ". The currently applied data settings and trained models will not change."
            )
            return
        self.cancel_feature_search_deletion()
        try:
            delete_candidate(self._feature_profile_store(), profile_id)
        except (OSError, ValueError) as ex:
            self.feature_search_status.value = "Deletion failed: " + html.escape(str(ex))
            return
        self.feature_search_preview.value = ""
        self.feature_search_status.value = "Saved recommendation deleted. Current data selections remain unchanged."
        self.refresh_feature_search_profiles()
        self.sync_feature_selection_controls(preserve_status=True)

    def refresh_feature_search_profiles(self) -> None:
        """Populate saved recommendations; IDs remain stable across label changes."""
        existing = self.feature_selection_profile.value
        records = load_candidates(self._feature_profile_store())
        options = [("No saved candidate profiles", "")]
        for record in records:
            label = profile_display_name(record)
            options.append((label, record["profile_id"]))
        if list(self.feature_selection_profile.options) != options:
            old_lock = self.eventhandler.lock_observe_1
            self.eventhandler.lock_observe_1 = True
            try:
                self.feature_selection_profile.options = options
                self.feature_selection_profile.value = existing if existing in dict(options).values() else ""
            finally:
                self.eventhandler.lock_observe_1 = old_lock

    def _saved_recommendation_issues(self) -> list[str]:
        """Fail closed on missing identity, source or original feature types."""
        record = find_candidate(self._feature_profile_store(), self.feature_selection_profile.value)
        if record is None:
            return ["Select a saved recommendation"]
        params = self.get_config_params()
        connection = params["connection"]
        problems = review_compatibility(
            record, project=params["name"],
            tables=[f"{connection.data_catalog}.{connection.data_table}"],
            class_column=connection.class_column, id_column=connection.id_column,
            columns=list(self.data_columns.options), sql_schema=getattr(self, "datatype_dict", None),
        )
        selected = record.get("recommendation", {}).get("selected_columns") or []
        if not isinstance(getattr(self, "datatype_dict", None), dict):
            problems.append("Current column types are unavailable")
        elif any(c not in self.datatype_dict for c in selected):
            problems.append("Current column type metadata is incomplete")
        if not (record.get("provenance", {}).get("original_column_schema") or []):
            problems.append("Saved feature schema is missing")
        return list(dict.fromkeys(problems))

    def review_feature_search_candidate(self) -> None:
        if self.feature_selection_mode.value != "reuse":
            return
        record = find_candidate(self._feature_profile_store(), self.feature_selection_profile.value)
        if record is None:
            self.feature_search_status.value = "Select a saved candidate to review."
            return
        params = self.get_config_params()
        connection = params["connection"]
        # Source identity from current GUI values; do not fetch data or train.
        context = dict(
            project=params["name"],
            tables=[f"{connection.data_catalog}.{connection.data_table}"],
            class_column=connection.class_column,
            id_column=connection.id_column,
            columns=list(self.data_columns.options),
        )
        reasons = self._saved_recommendation_issues()
        selected = record.get("recommendation", {}).get("selected_columns", [])
        nested = record.get("nested_validation") or {}
        result = nested.get("selected") or {}
        score = result.get("mean_mcc")
        score_text = f"Nested MCC: {score:.4f}<br>" if isinstance(score, (int, float)) else ""
        findings = "<br>".join(html.escape(x) for x in reasons) if reasons else "Source identity, recorded SQL schema and converter representation match; feature meaning still requires domain review."
        baseline = nested.get("full_baseline") or {}
        original = record.get("original_columns") or []
        retained = list(selected) == list(original)
        outcome = ("Full-column baseline retained; no improvement found within this search budget."
                   if retained else f"Candidate recommendation contains {len(selected)} of {len(original)} original columns.")
        model = (record.get("investigative_model") or {}).get("model_type", "Unknown")
        target = record.get("target", "Unknown")
        def metric(data, name, digits=3):
            value = data.get(name)
            return f"{value:.{digits}f}" if isinstance(value, (float, int)) else "unavailable"
        macro = ("Macro precision " + metric(result, "macro_precision") +
                 ", macro recall " + metric(result, "macro_recall") +
                 ", macro F1 " + metric(result, "macro_f1") + ".<br>"
                 if (record.get("evaluation") or {}).get("type") == "multiclass" else "")
        self.feature_search_preview.value = (
            "<b>" + html.escape(outcome) + "</b><br>"
            + "Status: candidate (saved; not independently validated); model: " + html.escape(str(model)) + ". "
            + "Nested selected MCC " + metric(result, "mean_mcc", 4)
            + " vs full " + metric(baseline, "mean_mcc", 4) + "; "
            + "Target " + html.escape(str(target)) + ": precision " + metric(result, "precision")
            + ", recall " + metric(result, "recall") + ", F1 " + metric(result, "f1")
            + "; TP " + str(result.get("tp", "unavailable")) + ", FP " + str(result.get("fp", "unavailable"))
            + ", FN " + str(result.get("fn", "unavailable")) + ".<br>"
            + macro + "Recommended columns: " + html.escape(", ".join(selected)) + "<br>"
            + "Results: <code>" + html.escape(str(record.get("output_directory") or "unavailable")) + "</code><br>"
            + "Compatibility: " + findings + "<br>Application is an explicit user decision."
        )
        self.feature_search_status.value = "Recommendation reviewed. No columns applied."
        self.sync_feature_selection_controls(preserve_status=True)

    def save_feature_search_candidate(self) -> None:
        """Save evidence as a candidate; never make it eligible for profile reuse."""
        result = self.feature_search_result
        if self.feature_selection_mode.value != "assisted" or not result:
            return
        try:
            directory = self._feature_profile_store()
            path = save_candidate(directory, result)
            self.feature_search_status.value = (
                "Saved candidate profile (NOT validated for reuse): "
                + html.escape(str(path))
            )
        except Exception as ex:
            self.feature_search_status.value = (
                "Could not save candidate: " + html.escape(type(ex).__name__ + ": " + str(ex))
            )

    def apply_feature_search_recommendation(self) -> None:
        mode = self.feature_selection_mode.value
        record = None
        if mode == "reuse":
            issues = self._saved_recommendation_issues()
            if issues:
                self.feature_search_status.value = "Cannot apply saved recommendation: " + html.escape("; ".join(issues))
                return
            record = find_candidate(self._feature_profile_store(), self.feature_selection_profile.value)
            recommendation = record["recommendation"]
        elif mode == "assisted" and self.feature_search_result:
            recommendation = self.feature_search_result.get("recommendation", {})
        else:
            return
        if recommendation.get("validation_status") != "candidate":
            self.feature_search_status.value = "Only saved candidate recommendations are supported."
            return
        selected = tuple(recommendation.get("selected_columns", ()))
        if not selected or any(column not in self.data_columns.options for column in selected):
            self.feature_search_status.value = "Recommendation is no longer compatible with the current dataset columns."
            return
        old_text = tuple(self.text_columns.value)
        old_categories = tuple(self.categorize_columns.value)
        saved_text = None
        saved_ngram_name = None
        if record is not None:
            saved_text = record.get("text_options")
            if not isinstance(saved_text, dict):
                self.feature_search_status.value = "Saved preprocessing settings are unavailable; cannot apply."
                return
            old_text = tuple(saved_text.get("text_columns", ()))
            old_categories = tuple(saved_text.get("category_columns", ()))
            saved_ngram = tuple(saved_text.get("ngram_range", ()))
            saved_ngram_name = next(
                (item.name for item in NgramRange if tuple(item.ngram_range) == saved_ngram),
                None,
            )
            if saved_ngram_name is None:
                self.feature_search_status.value = "Saved n-gram representation is unsupported; cannot apply."
                return
        previous_lock = self.eventhandler.lock_observe_1
        self.eventhandler.lock_observe_1 = True
        try:
            self.data_columns.value = selected
            self.text_columns.options = selected
            self.text_columns.value = tuple(column for column in old_text if column in selected)
            self.categorize_columns.options = self.text_columns.value
            self.categorize_columns.value = tuple(column for column in old_categories if column in self.text_columns.value)
            if saved_text is not None:
                # Reuse the exact configurable representation that produced the
                # saved recommendation instead of silently keeping today's GUI values.
                self.filter_checkbox.value = bool(saved_text["stop_words"])
                self.ngram_range_dropdown.value = saved_ngram_name
                self.encryption_checkbox.value = bool(saved_text["use_encryption"])
                self.categorize_checkbox.value = bool(saved_text["use_categorization"])
        finally:
            self.eventhandler.lock_observe_1 = previous_lock
        self._feature_search_applied = True
        self.update_ready()
        representation_note = " with its recorded text representation" if record else ""
        self.feature_search_status.value = (
            f"Applied {'saved ' if record else ''}candidate recommendation{representation_note}: "
            f"{len(selected)} original columns selected (not independently validated)."
        )
        self.disable_button("feature_search_apply_button")

    def form_field_enabled_toggle(self, form_name: str, disabled: bool = False) -> None:
        """ Sets disabled to <disabled> for all fields in the form group """
        fields = self.forms.get(form_name, [])

        for field in fields:
            setattr(field, "disabled", disabled)
        if form_name == "data" and "feature_selection_mode" in self.widgets:
            self.sync_feature_selection_controls()

    
    def models_form(self) -> widgets.Box:
        return self.create_section_form("Model", self.forms["models"], widgets.HBox)

    
    @property
    def data_form_fields_valid(self) -> bool:
        """ Checks so that fields in the data form have a non-default value """
        fields = self.forms.get("data", [])
        
        for field in fields:
            if hasattr(field, "options") and "N/A" in field.options: # N/A is default option
                return False

        return True

    def connection_form(self) -> widgets.Box:
        return self.create_section_form("Database connection", self.forms["connection"], widgets.Box)

    @property
    def id_column_diagnostic(self) -> widgets.HTML:
        if not hasattr(self, "_id_column_diagnostic_widget"):
            self._id_column_diagnostic_widget = widgets.HTML(value="")
        return self._id_column_diagnostic_widget

    def data_form(self) -> widgets.Box:
        # Keep the slider directly above its class choices, not as a separate
        # column beside the class selector. Preserve the original form fields
        # for enable/disable handling and existing observer propagation.
        class_controls = widgets.VBox([self.class_cardinality_limit, self.class_column])
        return self.create_section_form(
            "Dataset columns",
            [class_controls, widgets.VBox([self.id_column, self.id_column_diagnostic]),
             self.data_columns, self.text_columns],
            widgets.Box,
        )

    def regression_suite_form(self) -> widgets.Box:
        """Keep Repeat Last left-aligned and the regression suite right-aligned."""
        return self.create_section_form(
            "Run history & regression",
            [self.repeat_last_run_button, self.regression_suite_button],
            widgets.HBox,
            justify_content="space-between",
        )

    def feature_selection_modelling_form(self) -> widgets.Box:
        """Feature-selection controls between dataset settings and dataset actions."""
        # Keep profile management adjacent to the saved-profile selector.
        # Native button widths match the other application actions.
        primary_actions = [
            self.feature_search_investigate_button,
            self.feature_search_pause_button,
            self.feature_search_resume_button,
            self.feature_search_apply_button,
            self.feature_search_save_button,
        ]
        profile_actions = [
            self.feature_search_review_button,
            self.feature_search_delete_button,
        ]
        mode_row = widgets.HBox([self.feature_selection_mode])
        self.feature_selection_profile.layout.width = '480px'
        profile_row = widgets.HBox(
            [self.feature_selection_profile, *profile_actions],
            layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
        )
        search_rows_row = widgets.HBox([
            self.feature_search_rows,
        ], layout=widgets.Layout(flex_flow="row wrap", width="100%"))
        pipeline_row = widgets.HBox([
            self.feature_search_scaler, self.feature_search_oversampling,
            self.feature_search_reduction, self.feature_search_model_type,
        ], layout=widgets.Layout(flex_flow="row wrap", width="100%"))
        recovery_row = widgets.HBox([
            self.feature_search_recovery_selector,
            self.feature_search_recovery_refresh_button,
        ], layout=widgets.Layout(flex_flow="row wrap", width="100%"))
        action_row = widgets.HBox(
            primary_actions,
            layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
        )
        actions = primary_actions + profile_actions
        # Tooltip content is set through the widget description's title where
        # supported; avoid adding unknown constructor traits on old ipywidgets.
        help_text = {
            "feature_selection_mode": "Choose manual settings, run an investigation, or apply a saved recommendation.",
            "feature_selection_profile": "Choose a saved recommendation; Review shows its evidence without applying it.",
            "feature_search_fit_budget": "Upper bound on estimator fits for the selected nested search configuration; not a time estimate. Candidate workers are selected automatically from the shared JBG CPU/resource policy.",
            "feature_search_rows": "Maximum number of labelled records to retrieve for investigation (independent of normal training).",
            "feature_search_trials": "Number of random subset proposals; the maximum-subsets cap still applies.",
            "feature_search_sizes": "Comma-separated numbers of original columns sampled per proposal, e.g. 5,10,20.",
            "feature_search_inner_folds": "Inner cross-validation folds used to rank candidate subsets.",
            "feature_search_outer_folds": "Outer cross-validation folds used to evaluate the complete selection process.",
            "feature_search_top_sets": "Number of leading subsets used as starting points for add/remove refinement; larger values consume more fits.",
            "feature_search_refine_rounds": "Maximum greedy add/remove rounds per starting subset; zero disables refinement.",
            "feature_search_max_evaluations": "Maximum distinct subsets per inner search (repeated inside every outer fold). Refinement can be budget-limited.",
            "feature_search_min_gain": "Minimum improvement in mean inner-CV MCC needed to accept a refinement step.",
            "feature_search_linear_c": "Inverse regularization strength for logistic regression; larger C means weaker regularization.",
            "feature_search_linear_max_iterations": "Maximum optimizer iterations for the Linear estimator.",
            "feature_search_non_linear_max_iterations": "Maximum training epochs for the MLP estimator.",
            "feature_search_non_linear_alpha": "L2 regularization strength for the MLP estimator.",
            "feature_search_non_linear_hidden_layers": "Comma-separated hidden-layer widths, e.g. 100,50.",
            "feature_search_non_linear_learning_rate": "Initial learning rate of the MLP Adam optimizer.",
            "feature_search_tree_n_estimators": "Number of trees in the selected Tree ensemble.",
            "feature_search_tree_estimator": "Choose Extra Trees (default) or Random Forest; fixed during each search.",
            "feature_search_preset": "Quick/Standard/Extended set the search budget; Custom allows manual settings.",
            "feature_search_tree_max_depth": "Maximum Extra Trees depth; zero means unlimited.",
            "feature_search_tree_min_samples_leaf": "Minimum records per leaf for the selected Tree estimator.",
            "feature_search_tree_class_weight": "None: unweighted. balanced: inverse class-frequency weights. balanced_subsample: per-bootstrap weights (Random Forest).",
            "feature_search_tree_positive_threshold": "Fixed probability cutoff for the target class in BINARY tree investigations; 0.5 is the default. Not tuned by CV.",
            "feature_search_model_type": "Investigative estimator family, separate from the ordinary JBG classifier. CatBoost uses native categorical predictors, without PCA/SMOTE or scaling.",
            "feature_search_scaler": "Fixed scaling method used within each investigation CV fold.",
            "feature_search_oversampling": "Reuse JBG Random or SMOTE balancing inside training CV folds only; NOG disables oversampling. Validation remains untouched.",
            "feature_search_reduction": "Fold-local 95% PCA fitted after optional training-only oversampling; does not alter which original columns are selected.",
        }
        for widget_name, tooltip in help_text.items():
            control = getattr(self, widget_name)
            if "tooltip" in control.traits():
                control.tooltip = tooltip
            elif "description_tooltip" in control.traits():
                control.description_tooltip = tooltip
        # Explicit mapping avoids positional drift when new buttons are inserted.
        action_help = {
            "feature_search_investigate_button": "Start an exploratory feature-subset investigation using labelled training rows.",
            "feature_search_pause_button": "Request a safe pause after the current evaluation and save a checkpoint.",
            "feature_search_resume_button": "Resume the selected incomplete investigation from its checkpoint.",
            "feature_search_apply_button": "Apply the current compatible feature recommendation to selected columns.",
            "feature_search_save_button": "Save the completed investigation as a candidate recommendation (not yet validated).",
            "feature_search_review_button": "Review a saved recommendation and its validation evidence without applying it.",
            "feature_search_delete_button": "Delete the selected saved recommendation after confirmation.",
        }
        for name, tooltip in action_help.items():
            control = getattr(self, name)
            if "tooltip" in control.traits():
                control.tooltip = tooltip

        # Only the group contents collapse: controls remain the same widget
        # instances so existing enable/disable rules and saved settings survive.
        def setting_group(label, controls):
            content = widgets.HBox(controls, layout=widgets.Layout(
                flex_flow="row wrap", width="100%"))
            accordion = widgets.Accordion(children=[content], selected_index=None,
                                          layout=widgets.Layout(width="100%"))
            accordion.set_title(0, label)
            return accordion

        strategy_group = setting_group("Search strategy", [
            self.feature_search_preset, self.feature_search_trials, self.feature_search_sizes,
            self.feature_search_top_sets, self.feature_search_refine_rounds,
            self.feature_search_min_gain,
        ])
        validation_group = setting_group("Validation and computational budget", [
            self.feature_search_inner_folds, self.feature_search_outer_folds,
            self.feature_search_max_evaluations, self.feature_search_fit_budget,
        ])
        family_group = setting_group("Investigative model parameters (only the selected family is active)", [
            self.feature_search_linear_c, self.feature_search_linear_max_iterations,
            self.feature_search_non_linear_max_iterations, self.feature_search_non_linear_alpha,
            self.feature_search_non_linear_hidden_layers, self.feature_search_non_linear_learning_rate,
            self.feature_search_tree_estimator, self.feature_search_tree_n_estimators, self.feature_search_tree_max_depth,
            self.feature_search_tree_min_samples_leaf, self.feature_search_tree_class_weight, self.feature_search_tree_positive_threshold,
            self.feature_search_catboost_class_weight, self.feature_search_catboost_iterations,
        ])
        body = widgets.VBox([
            mode_row, profile_row, search_rows_row, pipeline_row,
            self.feature_search_investigation_mode, self.feature_search_subgroup_column, recovery_row, action_row,
            strategy_group, validation_group, family_group,
            self.feature_search_status, self.feature_search_result_tabs,
        ], layout=widgets.Layout(width="100%"))
        return self.create_section_form("Feature selection modelling", [body], widgets.Box)

    def continuation_form(self) -> widgets.Box:
        """Display actions that depend on the manually selected dataset."""
        return self.create_section_form(
            "Dataset actions",
            [self.continuation_button, self.test_profile_button],
            widgets.HBox,
            justify_content="space-between",
        )
    
    def checkboxes_form(self) -> widgets.Box:
        return self.create_section_form("Mode", self.forms["checkboxes"], widgets.HBox)

    def feature_importance_form(self) -> widgets.Box:
        return self.create_section_form("Feature analysis", self.forms["feature_importance"], widgets.HBox)

    def dark_numbers_form(self) -> widgets.Box:
        controls = widgets.HBox(
            self.forms["dark_numbers"],
            layout=widgets.Layout(
                width="72%",
                justify_content="flex-start",
                align_items="flex-start",
                flex_flow="row nowrap",
            ),
        )
        controls.add_class("dark-number-controls")

        form = self.create_section_form(
            "Dark Numbers",
            [controls, self.dark_number_equation],
            widgets.HBox,
            justify_content="space-between",
        )
        form.children[1].layout.align_items = "flex-start"
        return form
    
    def _create_model_settings_panel(self):
        """Create an initially collapsed panel; no extra permanent row in the main GUI."""
        sections = []
        self._model_parameter_panels = {}
        for algorithm, definitions in REGISTRY.items():
            controls = {}
            for key, spec in definitions.items():
                common = dict(description=spec.label, style={"description_width": "initial"},
                              layout=widgets.Layout(width="280px"))
                if spec.kind == "int":
                    widget = widgets.BoundedIntText(value=spec.default, min=spec.minimum,
                                                    max=spec.maximum, **common)
                elif spec.kind == "float":
                    widget = widgets.BoundedFloatText(value=spec.default, min=spec.minimum,
                                                      max=spec.maximum, **common)
                else:
                    widget = widgets.Dropdown(value=spec.default, options=spec.options, **common)
                controls[key] = widget
            self._model_parameter_controls[algorithm] = controls
            content = widgets.Box(list(controls.values()), layout=widgets.Layout(
                display="flex", flex_flow="row wrap", width="100%"))
            self._model_parameter_panels[algorithm] = content
            sections.append((algorithm, content))
        tabs = widgets.Accordion(children=[], selected_index=None)
        self._model_settings_tabs = tabs
        self.decision_policy_widget = widgets.Dropdown(
            description="Decision policy", options=[("Default classifier prediction", "default"),
              ("Fixed probability threshold (binary)", "fixed")], value="default",
            style={"description_width": "initial"}, layout=widgets.Layout(width="390px"))
        self.probability_threshold_widget = widgets.BoundedFloatText(
            description="Positive-class threshold", value=0.5, min=0.001, max=0.999,
            step=0.01, disabled=True, style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"))
        self.decision_policy_widget.observe(
            lambda change: setattr(self.probability_threshold_widget, "disabled",
                                   change["new"] != "fixed"), names="value")
        self.threshold_diagnostics_widget = widgets.Checkbox(
            value=False, description="Analyze training-only CV thresholds (extra fits)",
            indent=False, style={"description_width": "initial"},
            layout=widgets.Layout(width="460px"))
        self.model_benchmark_150_widget = widgets.Checkbox(
            value=False, description="OOF model benchmark (extra CV fits)",
            indent=False, layout=widgets.Layout(width="460px"))
        self.holdout_split_mode_widget = widgets.Dropdown(
            description="Holdout validation",
            options=[("Stratified rows (existing)", "stratified_rows"),
                     ("Group-disjoint (selected column)", "person_disjoint")],
            value="stratified_rows",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="410px"))
        self.validation_group_column_widget = widgets.Text(
            description="Group column", value="iafId",
            style={"description_width": "initial"}, layout=widgets.Layout(width="390px"))
        self.grouped_cv_widget = widgets.Checkbox(
            description="Group-disjoint cross-validation (extra validation safeguard)",
            value=False, indent=False, layout=widgets.Layout(width="490px"))
        self.excluded_predictors_widget = widgets.Text(
            description="Exclude predictors (CSV)", value="",
            style={"description_width": "initial"}, layout=widgets.Layout(width="510px"))
        decision_panel = widgets.VBox([self.holdout_split_mode_widget,
                                       self.validation_group_column_widget,
                                       self.grouped_cv_widget,
                                       self.excluded_predictors_widget,
                                       self.decision_policy_widget,
                                       self.probability_threshold_widget,
                                       self.threshold_diagnostics_widget,
                                       self.model_benchmark_150_widget])
        inner = widgets.Accordion(children=[tabs, decision_panel], selected_index=None)
        inner.set_title(0, "Classifier hyperparameters")
        inner.set_title(1, "Classification decision policy")
        panel = widgets.Accordion(children=[inner], selected_index=None)
        panel.set_title(0, "Model settings (advanced)")
        panel.layout.width = "100%"
        return panel

    def _model_selection_changed(self, change):
        if change.get("name") == "value":
            self.refresh_model_settings_panel()

    def refresh_model_settings_panel(self):
        """Show only configured, selected model families; retain all widget values."""
        if not hasattr(self, "_model_settings_tabs"):
            return
        selected = tuple(dict.fromkeys(self.algorithm_dropdown.value))
        visible = [(code, self._model_parameter_panels[code])
                   for code in selected if code in self._model_parameter_panels]
        tabs = self._model_settings_tabs
        if tuple(tabs.children) == tuple(content for _, content in visible):
            return
        tabs.children = tuple(content for _, content in visible)
        tabs.selected_index = None
        for index, (code, _) in enumerate(visible):
            tabs.set_title(index, f"{Algorithm[code].full_name} ({code})")

    def restore_decision_policy(self, mode):
        policy = getattr(mode, "decision_policy", "default")
        self.decision_policy_widget.value = policy if policy in ("default", "fixed") else "default"
        self.probability_threshold_widget.value = float(getattr(mode, "probability_threshold", 0.5))
        self.threshold_diagnostics_widget.value = bool(getattr(mode, "threshold_diagnostics", False))
        self.model_benchmark_150_widget.value = bool(getattr(mode, "model_benchmark_150", False))
        self.validation_group_column_widget.value = str(getattr(mode, "validation_group_column", "iafId"))
        self.grouped_cv_widget.value = bool(getattr(mode, "grouped_cross_validation", False))
        self.excluded_predictors_widget.value = str(getattr(mode, "excluded_predictor_columns", ""))
        split_mode = getattr(mode, "holdout_split_mode", "stratified_rows")
        self.holdout_split_mode_widget.value = (
            split_mode if split_mode in ("stratified_rows", "person_disjoint")
            else "stratified_rows")

    def get_model_parameters(self):
        return validate({algorithm: {key: control.value for key, control in controls.items()}
                         for algorithm, controls in self._model_parameter_controls.items()})

    def restore_model_parameters(self, settings):
        validated = validate(settings)
        baseline = defaults()
        for algorithm, controls in self._model_parameter_controls.items():
            for key, control in controls.items():
                control.value = validated.get(algorithm, {}).get(key, baseline[algorithm][key])

    def algorithm_form(self) -> widgets.Box:
        form = self.create_section_form("Model selection", self.forms["algorithm"], widgets.Box)
        form.children[1].layout.flex_flow = "row wrap"
        form.children[1].layout.width = "100%"
        form.children = tuple(form.children) + (self._model_settings_panel,)
        return form
    
    def data_handling_form(self) -> widgets.Box:
        return self.create_section_form("Training data", self.forms["data_handling"], widgets.HBox)
    
    def text_handling_form(self) -> widgets.Box:
        return self.create_section_form("Text processing", self.forms["text_handling"], widgets.HBox)
    
    def debug_form(self) -> widgets.Box:
        return self.create_section_form("Run settings", self.forms["debug"], widgets.HBox)
    
    def progress_form(self) -> widgets.Box:
        return self.create_form(self.forms["progress"], widgets.HBox)
    

    def create_form(self, children: list, boxtype: Callable, justify_content: str = None) -> widgets.Box:
        """Create one horizontal row while preserving the existing widget grouping semantics."""
        if boxtype == widgets.Box:
            layout = self.get_box_layout()
        elif boxtype == widgets.HBox:
            layout = widgets.Layout(width="100%") if justify_content else None
            if layout and justify_content:
                layout.justify_content = justify_content
        else:
            raise ValueError("boxtype needs to be a Box or HBox")

        return boxtype(children, layout=layout) if layout else boxtype(children)

    def create_section_form(
        self,
        title: str,
        children: list,
        boxtype: Callable,
        justify_content: str = None,
    ) -> widgets.VBox:
        """Wrap a horizontal form row in a consistently titled, framed GUI section."""
        row = self.create_form(children, boxtype, justify_content=justify_content)
        row.add_class("jbg-section-row")

        heading = widgets.HTML(value=f"<strong>{title}</strong>")
        heading.add_class("jbg-section-title")

        section = widgets.VBox(
            [heading, row],
            layout=widgets.Layout(
                border="2px solid #d0d0d0",
                padding="6px 8px 8px 8px",
                margin="4px 0 4px 0",
                width="100%",
            ),
        )
        section.add_class("jbg-section")
        return section
    
    def get_box_layout(self) -> widgets.Layout:
        return widgets.Layout(
            display="flex",
            flex_flow="row",
            border="solid 0",
            align_items="stretch",
            width="auto",
        )
    
    def _load_default_widgets(self, clear: bool = False) -> None:
        if clear:
            self.widgets = {}

        for item in self.default_widgets.keys():
            if hasattr(self, item):
                getattr(self, item)

    
    def _load_widget(self, name: str, calculated_params: dict = None, callback: Callable = None, handler: Callable = None) -> widgets.Widget:
        if name not in self.widgets:
            config = self.default_widgets.get(name)

            calculated_params = calculated_params if calculated_params else {}

            if config:
                widget_class = getattr(widgets, config["type"])
                widget_params = config["params"] | calculated_params
                self.widgets[name] = widget_class(**widget_params)

                if callback:
                    self.widgets[name].on_click(callback=callback)

                if handler:
                    self.widgets[name].observe(handler=handler)

                if css_class := config.get("class"):
                    self.widgets[name].add_class(css_class)
        
        return self.widgets[name]
        
    @property
    def progress(self) -> tuple:
        return (self.progress_bar, self.progress_label)

    
    @property
    def logo(self) -> widgets.Image:
        """ This checks the if here, since the initial state requires loading an image """
        name = sys._getframe(  ).f_code.co_name # Current function name

        if name not in self.widgets:
            with open(self.logo_image, "rb") as logo:
                logo_data = logo.read()

            return self._load_widget(name, {
                "value": logo_data
            })

        return self.widgets[name]

    @property
    def welcome(self) -> widgets.HTML:
        """ The initial data is entirely from the json """
        name = sys._getframe(  ).f_code.co_name # Current function name
        
        return self._load_widget(name)
    
    @property
    def build_identity(self) -> widgets.HTML:
        """Application version and repository provenance shown below the title."""
        name = sys._getframe().f_code.co_name
        widget = self._load_widget(name)
        widget.value = (
            '<div style="font-size: 0.9em; opacity: 0.75; margin-top: -6px; margin-bottom: 10px;">'
            + html.escape(application_version_line())
            + "</div>"
        )
        return widget

    @property
    def project(self) -> widgets.Text:
        """ The initial data is entirely from the json """
        name = sys._getframe(  ).f_code.co_name # Current function name
        
        return self._load_widget(name)
    
    @property
    def sql_username(self) -> widgets.Text:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.sql_username_and_password)
        
        return self.widgets[name]
    
    @property
    def sql_password(self) -> widgets.Password:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.sql_username_and_password)
        
        return self.widgets[name]

    @property
    def data_catalogs_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.data_catalogs_dropdown)
            
        return self.widgets[name]
    
    @property
    def data_tables_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.data_tables_dropdown)
            
        return self.widgets[name]
    
    @property
    def models_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.models_dropdown)
            
        return self.widgets[name]


    @property
    def class_column(self) -> widgets.RadioButtons:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.class_column)
        
        return self.widgets[name]
    
    @property
    def id_column(self) -> widgets.RadioButtons:
        name = sys._getframe().f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.id_column)
        
        return self.widgets[name]

    @property
    def data_columns(self) -> widgets.SelectMultiple:
        name = sys._getframe().f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.data_columns)
            
        return self.widgets[name]

    @property
    def text_columns(self) -> widgets.SelectMultiple:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.feature_search_representation_setting)

    @property
    def data_numerical_columns(self) -> list:
        return self._get_data_columns(WidgetConstant.DATA_NUMERICAL)

    @property
    def data_text_columns(self) -> list:
        return self._get_data_columns(WidgetConstant.DATA_TEXT)
    
    @property
    def class_summary(self) -> widgets.HTML:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
       
    
    @property
    def continuation_button(self) -> widgets.Button:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, callback=self.eventhandler.continuation_button_was_clicked)
        
        return self.widgets[name]

    @property
    def test_profile_button(self) -> widgets.Button:
        name = sys._getframe().f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, callback=self.eventhandler.test_profile_button_was_clicked)

        return self.widgets[name]

    @property
    def repeat_last_run_button(self) -> widgets.Button:
        name = sys._getframe().f_code.co_name # Current function name
        if name not in self.default_widgets:
            # Keep older local settings.json files compatible with this newly added action.
            self.default_widgets[name] = {
                "type": "Button",
                "params": {
                    "description": "Repeat last",
                    "button_style": "primary",
                    "disabled": True,
                    "tooltip": "Repeat the previous classifier settings; data is fetched again",
                    "icon": "repeat",
                },
            }
        if name not in self.widgets:
            self._load_widget(name, callback=self.eventhandler.repeat_last_run_button_was_clicked)

        return self.widgets[name]

    @property
    def regression_suite_button(self) -> widgets.Button:
        name = sys._getframe().f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, callback=self.eventhandler.regression_suite_button_was_clicked)

        return self.widgets[name]
    
    @property
    def feature_selection_mode(self) -> widgets.RadioButtons:
        return self._load_widget("feature_selection_mode", handler=self.eventhandler.feature_selection_mode)

    @property
    def feature_selection_profile(self) -> widgets.Dropdown:
        return self._load_widget("feature_selection_profile", handler=self.eventhandler.feature_selection_profile)

    @property
    def feature_search_model_type(self) -> widgets.Dropdown:
        return self._load_widget("feature_search_model_type", handler=self.eventhandler.feature_search_model_type)

    def _feature_search_setting_widget(self, name: str):
        return self._load_widget(name, handler=self.eventhandler.feature_search_setting)

    @property
    def feature_search_preset(self):
        return self._load_widget("feature_search_preset", handler=self.eventhandler.feature_search_preset)
    @property
    def feature_search_tree_estimator(self):
        return self._feature_search_setting_widget("feature_search_tree_estimator")

    @property
    def feature_search_rows(self): return self._feature_search_setting_widget("feature_search_rows")
    @property
    def feature_search_scaler(self): return self._feature_search_setting_widget("feature_search_scaler")

    @property
    def feature_search_oversampling(self): return self._feature_search_setting_widget("feature_search_oversampling")

    @property
    def feature_search_reduction(self): return self._feature_search_setting_widget("feature_search_reduction")
    @property
    def feature_search_trials(self): return self._feature_search_setting_widget("feature_search_trials")
    @property
    def feature_search_sizes(self): return self._feature_search_setting_widget("feature_search_sizes")
    @property
    def feature_search_inner_folds(self): return self._feature_search_setting_widget("feature_search_inner_folds")
    @property
    def feature_search_outer_folds(self): return self._feature_search_setting_widget("feature_search_outer_folds")
    @property
    def feature_search_top_sets(self): return self._feature_search_setting_widget("feature_search_top_sets")
    @property
    def feature_search_refine_rounds(self): return self._feature_search_setting_widget("feature_search_refine_rounds")
    @property
    def feature_search_max_evaluations(self): return self._feature_search_setting_widget("feature_search_max_evaluations")
    @property
    def feature_search_min_gain(self): return self._feature_search_setting_widget("feature_search_min_gain")
    @property
    def feature_search_linear_c(self): return self._feature_search_setting_widget("feature_search_linear_c")
    @property
    def feature_search_linear_max_iterations(self): return self._feature_search_setting_widget("feature_search_linear_max_iterations")
    @property
    def feature_search_non_linear_max_iterations(self): return self._feature_search_setting_widget("feature_search_non_linear_max_iterations")
    @property
    def feature_search_non_linear_alpha(self): return self._feature_search_setting_widget("feature_search_non_linear_alpha")
    @property
    def feature_search_non_linear_hidden_layers(self): return self._feature_search_setting_widget("feature_search_non_linear_hidden_layers")
    @property
    def feature_search_non_linear_learning_rate(self): return self._feature_search_setting_widget("feature_search_non_linear_learning_rate")
    @property
    def feature_search_catboost_class_weight(self): return self._feature_search_setting_widget("feature_search_catboost_class_weight")

    @property
    def feature_search_catboost_iterations(self): return self._feature_search_setting_widget("feature_search_catboost_iterations")

    @property
    def feature_search_tree_n_estimators(self): return self._feature_search_setting_widget("feature_search_tree_n_estimators")
    @property
    def feature_search_tree_max_depth(self): return self._feature_search_setting_widget("feature_search_tree_max_depth")
    @property
    def feature_search_tree_min_samples_leaf(self): return self._feature_search_setting_widget("feature_search_tree_min_samples_leaf")
    @property
    def feature_search_tree_class_weight(self): return self._feature_search_setting_widget("feature_search_tree_class_weight")
    @property
    def feature_search_tree_positive_threshold(self): return self._feature_search_setting_widget("feature_search_tree_positive_threshold")
    @property
    def feature_search_fit_budget(self): return self._load_widget("feature_search_fit_budget")
    @property
    def feature_search_status(self): return self._load_widget("feature_search_status")
    @property
    def feature_search_preview(self): return self._load_widget("feature_search_preview")

    @property
    def feature_search_investigate_button(self) -> widgets.Button:
        return self._load_widget("feature_search_investigate_button", callback=self.eventhandler.feature_search_investigate_button_was_clicked)

    @property
    def feature_search_apply_button(self) -> widgets.Button:
        return self._load_widget("feature_search_apply_button", callback=self.eventhandler.feature_search_apply_button_was_clicked)

    @property
    def feature_search_save_button(self) -> widgets.Button:
        return self._load_widget("feature_search_save_button", callback=self.eventhandler.feature_search_save_button_was_clicked)

    @property
    def feature_search_review_button(self) -> widgets.Button:
        return self._load_widget("feature_search_review_button", callback=self.eventhandler.feature_search_review_button_was_clicked)

    @property
    def feature_search_delete_button(self) -> widgets.Button:
        return self._load_widget("feature_search_delete_button", callback=self.eventhandler.feature_search_delete_button_was_clicked)

    @property
    def train_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name, handler=self._feature_importance_controls_changed)

    @property
    def feature_importance_checkbox(self) -> widgets.Checkbox:
        return self._load_widget("feature_importance_checkbox", handler=self._feature_importance_controls_changed)

    @property
    def feature_importance_repeats(self) -> widgets.BoundedIntText:
        return self._load_widget("feature_importance_repeats")

    def _feature_importance_controls_changed(self, change) -> None:
        if change.get("name") in ("value", "disabled"):
            self.sync_feature_importance_control_state()

    def sync_feature_importance_control_state(self) -> None:
        # Display-only synchronization also runs while Repeat Last observers are locked.
        checkbox = self.feature_importance_checkbox
        checkbox.disabled = self.train_checkbox.disabled or not self.train_checkbox.value or self.regression_suite_state
        self.feature_importance_repeats.disabled = checkbox.disabled or not checkbox.value
        
    
    @property
    def predict_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, handler=self.eventhandler.predict_checkbox)
        return self._load_widget(name)
        
    
    @property
    def mispredicted_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)

    @property
    def metas_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)

    @property
    def dark_numbers_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe().f_code.co_name
        return self._load_widget(name, handler=self.eventhandler.dark_numbers_checkbox)

    @property
    def dark_number_method(self) -> widgets.RadioButtons:
        name = sys._getframe().f_code.co_name
        widget = self._load_widget(name, handler=self.eventhandler.dark_number_method)
        widget.layout.width = "150px"
        widget.layout.min_width = "150px"
        widget.style.description_width = "initial"
        return widget

    @property
    def dark_number_alpha(self) -> widgets.RadioButtons:
        name = sys._getframe().f_code.co_name
        widget = self._load_widget(name, handler=self.eventhandler.dark_number_alpha)
        widget.layout.width = "145px"
        widget.layout.min_width = "145px"
        widget.style.description_width = "initial"
        return widget

    @property
    def dark_number_target(self) -> widgets.RadioButtons:
        name = sys._getframe().f_code.co_name
        if name not in self.default_widgets:
            self.default_widgets[name] = {
                "type": "RadioButtons",
                "params": {
                    "options": (("All classes", ""),),
                    "value": "",
                    "disabled": True,
                    "description": "Target:",
                    "tooltip": "Choose which class to estimate Dark Numbers for, or keep all classes",
                },
            }
        widget = self._load_widget(name)
        widget.description = "Target:"
        widget.layout.width = "170px"
        widget.layout.min_width = "170px"
        widget.layout.margin = "0 0 0 6px"
        widget.style.description_width = "initial"
        if not getattr(widget, "_jbg_target_handler_attached", False):
            widget.observe(self.eventhandler.dark_number_target)
            widget._jbg_target_handler_attached = True
        return widget

    @property
    def dark_number_failure_mode(self) -> widgets.RadioButtons:
        name = sys._getframe().f_code.co_name
        if name not in self.default_widgets:
            # Older local settings may predate the radio control entirely. Keep the
            # production default on the experimental fallback while exposing the choice.
            self.default_widgets[name] = {
                "type": "RadioButtons",
                "params": {
                    "options": (("No fallback", False), ("Experimental", True)),
                    "value": True,
                    "disabled": True,
                    "description": "Failure:",
                    "tooltip": (
                        "Choose no fallback or the experimental perturbed same-model fallback "
                        "when ordinary correction is statistically unestimable"
                    ),
                    "layout": widgets.Layout(width="210px", min_width="210px", margin="0 0 0 8px"),
                    "style": {"description_width": "initial"},
                },
            }
        widget = self._load_widget(name)
        current_value = bool(widget.value)
        widget.options = (("No fallback", False), ("Experimental", True))
        widget.value = current_value
        widget.description = "Failure:"
        widget.layout.width = "210px"
        widget.layout.min_width = "210px"
        widget.layout.margin = "0 0 0 8px"
        widget.style.description_width = "initial"
        if not getattr(widget, "_jbg_failure_handler_attached", False):
            widget.observe(self.eventhandler.dark_number_failure_mode)
            widget._jbg_failure_handler_attached = True
        return widget

    @property
    def dark_number_equation(self) -> widgets.HTMLMath:
        """Read-only formula card kept outside persisted widget settings."""
        name = sys._getframe().f_code.co_name
        if name not in self.widgets:
            self.widgets[name] = widgets.HTMLMath(
                value="",
                layout=widgets.Layout(width="24%", min_width="200px", margin="0 0 0 28px", overflow_x="auto"),
            )
            self.widgets[name].add_class("dark-number-equation")
            self.sync_dark_number_equation()
        return self.widgets[name]

    def sync_dark_number_control_state(self) -> None:
        """Keep dependent Dark Number controls aligned with the Dark Numbers checkbox."""
        disabled = self.dark_numbers_checkbox.disabled or not self.dark_numbers_checkbox.value
        self.dark_number_method.disabled = disabled
        self.dark_number_alpha.disabled = disabled
        self.dark_number_target.disabled = disabled
        self.dark_number_failure_mode.disabled = disabled
        self.sync_dark_number_equation()

    def sync_dark_number_equation(self) -> None:
        """Render the selected calculator formula and correction-failure policy."""
        equation = self.widgets.get("dark_number_equation")
        if equation is None:
            return

        spec = DarkNumberCalculator.get_formula_spec(
            self.dark_number_method.value,
            self.dark_number_alpha.value,
            root_degree=3,
        )
        failure_mode = "Experimental" if self.dark_number_failure_mode.value else "None"
        target_mode = html.escape(str(self.dark_number_target.value or "All classes"))
        enabled = bool(self.dark_numbers_checkbox.value)
        opacity = "1.0" if enabled else "0.38"
        state_note = "" if enabled else "<div style='margin-top:4px'><em>Estimate is off</em></div>"
        root_note = (
            " &middot; root=3"
            if spec["calculation_type"].startswith("non_linear")
            else ""
        )
        equation.value = (
            f"<div style='border-left:1px solid #d0d0d0;padding:1px 0 1px 12px;opacity:{opacity};font-size:0.88em'>"
            "<div style='font-weight:600;margin-bottom:2px'>Formula</div>"
            f"<div style='font-size:0.94em'>\\({spec['latex']}\\)</div>"
            f"<div style='margin-top:4px;font-size:0.78em'>{spec['alpha_note']}{root_note}</div>"
            "<div style='margin-top:2px;font-size:0.78em'>c: correction &middot; TN<sub>r</sub>/TP<sub>r</sub>: true-neg/true-pos rate</div>"
            f"<div style='margin-top:2px;font-size:0.78em'>Target: {target_mode}</div>"
            f"<div style='margin-top:2px;font-size:0.78em'>Failure: {failure_mode}</div>"
            f"{state_note}"
            "</div>"
        )

    def sync_dark_number_target_options(self, distribution: dict | None = None) -> None:
        """Populate target choices from the selected class column while preserving All classes."""
        if distribution is None:
            distribution = {}
        labels = [
            str(label) for label in distribution.keys()
            if label is not None and str(label).strip() != ""
        ]
        options = [("All classes", "")] + [(label, label) for label in labels]
        current = str(self.dark_number_target.value or "")
        self.dark_number_target.options = tuple(options)
        self.dark_number_target.value = current if current in {value for _, value in options} else ""
        self.sync_dark_number_equation()

    def sync_dark_number_alpha_options(self) -> None:
        """Expose only alpha variants implemented for the selected formula family."""
        current = self.dark_number_alpha.value
        if self.dark_number_method.value == "NON_LINEAR":
            options = (("None", "NONE"), ("Single", "SINGLE"))
            if current == "SEPARATED":
                current = "NONE"
        else:
            options = (("None", "NONE"), ("Single", "SINGLE"), ("Separated", "SEPARATED"))
        self.dark_number_alpha.options = options
        self.dark_number_alpha.value = current if current in dict(options).values() else "NONE"
    
    @property
    def algorithm_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.algorithm_dropdown_handler, \
            calculated_params={"options": [(label, value) for label, value in Algorithm.get_sorted_list()
                                          if value != "CBST" and getattr(value, "name", value) != "CBST"]})
        
    @property
    def preprocess_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.preprocess_dropdown_handler, \
            calculated_params={"options": Preprocess.get_sorted_list()})
        
    
    @property
    def scoremetric_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        if name not in self.widgets:
            widget = self._load_widget(
                name, handler=self._scoremetric_tooltip_changed,
                calculated_params={"options": ScoreMetric.get_sorted_list()},
            )
            widget.tooltip = score_metric_help(widget.value)
        return self.widgets[name]

    def _scoremetric_tooltip_changed(self, change) -> None:
        # Display-only updates must also run during locked Repeat Last/model restore.
        if change["name"] == "value":
            change["owner"].tooltip = score_metric_help(change["new"])
        

    @property
    def reduction_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.reduction_dropdown_handler, \
            calculated_params={"options": Reduction.get_sorted_list()
        })
    
    
    @property
    def oversampler_dropdown(self) -> widgets.SelectMultiple:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, calculated_params={
            "options": Oversampling.get_sorted_list()
        })
        
    
    @property
    def undersampler_dropdown(self) -> widgets.SelectMultiple:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, calculated_params={
            "options":Undersampling.get_sorted_list()
        })
        
    
    @property
    def testdata_slider(self) -> widgets.IntSlider:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def iterations_slider(self) -> widgets.IntSlider:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def encryption_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.feature_search_representation_setting)
        
    
    @property
    def categorize_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.feature_search_representation_setting)
        
    @property
    def categorize_columns(self) -> widgets.SelectMultiple:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.feature_search_representation_setting)
        
    
    @property
    def filter_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.feature_search_representation_setting)
        
    
    @property
    def ngram_range_dropdown(self) -> widgets.DropDown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, calculated_params={
            "options":NgramRange.get_plain_list()
        }, handler=self.eventhandler.feature_search_representation_setting)
        
    
    @property
    def data_limit(self) -> widgets.IntText:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def show_info_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        


    @property
    def num_variables(self) -> widgets.IntText:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def start_button(self) -> widgets.Button:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self._load_widget(name, callback=self.eventhandler.start_button_was_clicked)
        return self.widgets[name]

    @property
    def progress_bar(self) -> widgets.FloatProgress:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def progress_label(self) -> widgets.HTML:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
    
    @property
    def output(self) -> widgets.Output:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name)

    @property
    def output_title(self) -> widgets.HTML:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name)
    
    @property
    def mispredicted_gridbox(self) -> widgets.GridBox:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name)

    @property
    def mispredicted_output(self) -> widgets.Output:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name)

    def _set_css_styles(self, css_file_paths: Iterable = None) -> widgets.HTML:
        """
        Read the custom CSS files and load them into Jupyter.
        Pass the file path to the CSS file.
        """
        if not css_file_paths:
            css_file_paths = self.settings.get("stylesheets")
        
        styles = []
        for path in css_file_paths:
            with open(self.get_sibling_file_path(path), "r", encoding="utf-8") as stylesheet:
                styles.append(stylesheet.read())

        s = '<style>%s</style>' % " ".join(styles)
        return widgets.HTML(s)
        
    def display_gui(self) -> None:
        """ Displays the elements in the given order """
        
        display(self._set_css_styles())

        for item in self.settings.get("display"):
            if hasattr(self, item):
                widget = getattr(self, item)
                if item.endswith("_form"):
                    widget = widget()

                try:
                    display
                except NameError:
                    """ Empty for now """
                    #print(item)
                else:
                    display(widget, display_id=item)
