from __future__ import annotations
import dis
from enum import Enum
import os
from pathlib import Path
import sys
import json

from typing import Callable, Protocol
import ipywidgets as widgets
from sklearn.utils import Bunch

from Config import (Config, Reduction, ReductionTuple, Algorithm, 
    AlgorithmTuple, Preprocess, PreprocessTuple, ScoreMetric)
from JBGExceptions import GuiWidgetsException
from AutomaticClassifier import AutomaticClassifier
import Helpers

class WidgetConstant(Enum):
    DATA_NUMERICAL = 1
    DATA_TEXT = 2

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
        # Change to default values below if it's changing in ways
        if change.name != "index":
            return
        
        if change.new == 0: # Moves to the empty value
            return
        
        if not self.lock_observe_1:
            self.widgets.update_class_id_data_columns()
        
        if change.new == 1: # This is the default train option
            self.widgets.disable_button("continuation_button")
            return
        
        self.widgets.update_from_model_config()

    def class_column(self, change: Bunch) -> None:
        """ Handler for class_column
            Events:
                value_change: Changes the id and data columns, updates summary
        """
        if change.name not in ["index", "disabled"]:
            return
        
        if change.name == "disabled" and change.name:
            # This is still disabled
            return

        if change.name == "index" and change.old != change.new: # This is not done automatically
            self.widgets.summarise_state = True
 
        """ update_id_and_data_columns
        """
        self.widgets.update_class_summary()
        self.widgets.update_id_column() # Event.name "options"

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
        self.widgets.update_ready() # At the moment enables/disables the continuation_button
    
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

    def continuation_button_was_clicked(self, button: widgets.Button) -> None:
        """ Callback: Sets various states based on the value in models dropdown. """
        self.lock_observe_1 = True
        
        self.widgets.deactivate_section("data")
        self.widgets.continuation_button_actions()
        

    def start_button_was_clicked(self, button: widgets.Button) -> None:
        """ Callback: Changes the state to be read-only, starts the classifier (or the class summary) """

        self.widgets.deactivate_section("classifier")
        self.widgets.start_button_actions()
    
    def reset_button_was_clicked(self, button: widgets.Button) -> None:
        """ Callback: Resets all widgets back to the way they were """

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

    def get_id_columns(self, **kwargs) -> list:
        """ Used in the GUI, gets name and type for columns """
        
class GUIhandler(Protocol):
    def get_class_distribution(self, data_settings: dict) -> dict:
        """ Help function to avoid Widgets knowing about classifier DataLayer """

    def run_classifier(self) -> None:
        """ Sets up the classifier and then runs it"""

    def correct_mispredicted_data(self, new_class: str, index: int) -> None:
        """ Changes the original dataset """

    @property
    def logger(self):
        """ Signifies the logger property """

    @property
    def datalayer(self) -> DataLayer:
        """ Returns the datalayer """

    def set_data_catalog(self, data_catalog: str) -> None:
        """ Updates the GUI datalayer with the data catalog """

class Widgets:
    """ Creates and populates the widgets of the GUI """
    TEXT_MIN_LIMIT = 30
    TEXT_AREA_MIN_LIMIT = 60
    
    def __init__(self, src_path: Path, GUIhandler: GUIhandler, model_path: Path = None, settings: dict = None) -> None:

        if settings:
            self.settings = settings
        else:
            with open(Path(__file__).parent / "default_settings.json") as f: # The default_settings.json is sibling
                self.settings = json.load(f)
        
        self.eventhandler = EventHandler(self)
        self.guihandler = GUIhandler
        self.model_path = model_path if model_path else src_path / Config.DEFAULT_MODELS_PATH
        
        self.default_widgets = self.settings.get("widgets")
        self.sections = self.settings.get("sections")
        self.logo_image = src_path / self.settings.get("logo")

        self.states = {
            "rerun": False,
            "summarise": False,
        }
        self.widgets = {}
        self._load_default_widgets()
        self.forms = {
            "catalog": [self.data_catalogs_dropdown, self.data_tables_dropdown],
            "data": [self.class_column, self.id_column, self.data_columns, self.text_columns],
            "checkboxes": [self.train_checkbox, self.predict_checkbox, self.mispredicted_checkbox, self.metas_checkbox],
            "algorithm": [self.preprocess_dropdown, self.reduction_dropdown, self.algorithm_dropdown, self.scoremetric_dropdown],
            "data_handling": [self.smote_checkbox, self.undersample_checkbox, self.testdata_slider, self.iterations_slider],
            "text_handling": [self.categorize_checkbox, self.categorize_columns, self.encryption_checkbox, self.filter_checkbox, self.filter_slider],
            "debug": [self.data_limit, self.show_info_checkbox, self.num_variables],
            "progress": [self.progress_bar, self.progress_label]
        }

        # We need a dictionary to keep track of datatypes
        self.datatype_dict = None
    
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
    def summarise_state(self) -> bool:
        return self.states["summarise"]

    @summarise_state.setter
    def summarise_state(self, state: bool) -> None:
        self.states["summarise"] = state

    
    def load_contents(self) -> None:
        """ This is when we set content loaded from datasource """
        if not self.datalayer:
            raise GuiWidgetsException("Data Layer not initialized")

        self.update_data_catalogs_dropdown()
        self.update_models_dropdown_options()

    def update_from_model_config(self, model: str = None ) -> None:
        """ Sets values based on the config in the chosen model """
        value = model if model else self.models_dropdown.value
        model_path = self.model_path / value
        
        config = Config.load_config_from_model_file(model_path)
        self.categorize_columns.options = config.get_text_column_names()
        
        # Values from config
        self.update_values({
            "class_column": config.get_class_column_name(),
            "id_column": config.get_id_column_name(),
            "data_columns": config.get_data_column_names(),
            "text_columns": config.get_text_column_names(),
            "algorithm_dropdown": tuple([str(config.get_algorithm().name)]),
            "preprocess_dropdown": tuple([str(config.get_preprocessor().name)]), 
            "reduction_dropdown": tuple([str(config.get_feature_selection().name)]),
            "num_variables": config.get_num_selected_features(),
            "filter_checkbox": config.use_stop_words(),
            "filter_slider": config.get_stop_words_threshold_percentage(),
            "encryption_checkbox": config.should_hex_encode(),
            "categorize_checkbox": config.use_categorization(),
            "categorize_columns":  config.get_categorical_text_column_names(),
            "testdata_slider": config.get_test_size_percentage(),
            "smote_checkbox": config.use_smote(),
            "undersample_checkbox": config.use_undersample(),
        })
        
        # Disabled items:
        self.disable_items(["class_column", "id_column", "data_columns", "text_columns", "categorize_columns"])
        
        # Enabled items:
        self.enable_button("continuation_button")
    
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

    def set_checkboxes(self, new_model: bool) -> None:
        """ Sets the values and enabled-state based on new vs trained model"""
        if new_model:
            self.enable_items([
                "train_checkbox",
                "predict_checkbox",
                "mispredicted_checkbox",
                #"metas_checkbox"
            ])
            self.update_values({
                "train_checkbox": True,
                "predict_checkbox": False,
                "mispredicted_checkbox": True,
                "metas_checkbox": False
            })
        else:
            self.disable_items([
                "train_checkbox",
                "predict_checkbox",
                "mispredicted_checkbox",
                "metas_checkbox"
            ])
            self.update_values({
                "train_checkbox": False,
                "predict_checkbox": True,
                "mispredicted_checkbox": False,
                "metas_checkbox": True
            })
            
    def continuation_button_actions(self) -> None:
        """ Complex actions when button is clicked """
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
                "smote_checkbox",
                "undersample_checkbox",
                "testdata_slider",
                "iterations_slider",
                "encryption_checkbox",
                "categorize_checkbox",
                "categorize_columns",
                "filter_checkbox",
                "filter_slider"
            ]
        
            self.categorize_columns.options = self.text_columns.value
        
        self.update_data_limit()
        self.enable_items(default_enable_list + new_model_list)

    def start_button_actions(self) -> None:
        """ Complex actions when button is clicked """
        if self.rerun_state:
            self.update_class_summary()
        else:
            self.rerun_state = True
        
        self.output.clear_output(wait=True)
        
        self.guihandler.run_classifier(config_params=self.get_config_params(), output=self.output)

    def set_rerun(self) -> None:
        """ Updates buttons and checkboxes for doing a rerun with the settings """
        self.enable_items(["start_button", "show_info_checkbox"])

        self.start_button.description = "Rerun"
        self.start_button.tooltip = "Rerun the classifier with the same setting as last time"
        
    def handle_mispredicted(self, classifier: AutomaticClassifier) -> None:
        mispredicted = classifier.get_mispredicted_dataframe()
        if mispredicted is None:
            return
        items = [widgets.Label(mispredicted.index.name)] + \
            [widgets.Label(item) for item in mispredicted.columns] + \
            [widgets.Label("Reclassify as")]
        cols = len(items)
        for i in mispredicted.index:
            row = mispredicted.loc[i]
            row_items = [widgets.Label(str(row.name))]
            for item in row.index: 
                row_items.append(self.get_text_widget(row[item]))
            dropdown_options = [('Keep', 0)]
            for label in classifier.get_unique_classes():
                dropdown_options.append((label, (label, row.name)))
            reclassify_dropdown = widgets.Dropdown(options = dropdown_options, value = 0, description = '', disabled = False)
            reclassify_dropdown.observe(self.eventhandler.reclassify_dropdown__value,'value')
            row_items += [reclassify_dropdown]
            items += row_items 
        
        gridbox_layout = widgets.Layout(grid_template_columns="repeat("+ str(cols) +", auto)", border="4px solid grey")
        self.widgets["mispredicted_gridbox"] = widgets.GridBox(items, layout=gridbox_layout)
        display(self.mispredicted_gridbox)

    def correct_mispredicted_data(self, new_class: str, index: int) -> None:
        """ From the reclassify dropdown change"""
        self.guihandler.classifier_datalayer.correct_mispredicted_data(new_class, index)


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
                class_catalog = os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
                class_table = os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
                trusted_connection = True,
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
                use_metas = self.metas_checkbox.value,
                use_stop_words = self.filter_checkbox.value,
                specific_stop_words_threshold = float(self.filter_slider.value) / 100.0,
                hex_encode = self.encryption_checkbox.value,
                use_categorization = self.categorize_checkbox.value,
                category_text_columns = list(self.categorize_columns.value),
                test_size = float(self.testdata_slider.value) / 100.0,
                smote = self.smote_checkbox.value,
                undersample = self.undersample_checkbox.value,
                algorithm = AlgorithmTuple(self.algorithm_dropdown.value),
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
            "debug": Config.Debug(
                on=True,
                data_limit=self.data_limit.value
            ),
            "name": self.project.value,
            "save": self.new_model
        }

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

    def update_models_dropdown_options(self) -> None:
        preface = ["", Config.DEFAULT_TRAIN_OPTION]
        
        models_list = self.datalayer.get_trained_models_from_files(self.model_path, Config.DEFAULT_MODEL_EXTENSION, preface)

        updates = {
            "options": models_list,
            "default": models_list[0]
        }
        self.update_item("models_dropdown", updates)

    def update_class_id_data_columns(self) -> None:
        if self.data_catalogs_dropdown.value == "" or self.data_tables_dropdown.value == "":
            return
        
        columns = self.datalayer.get_id_columns(self.data_catalogs_dropdown.value, self.data_tables_dropdown.value)
        
        self.datatype_dict = columns
        columns_list = list(columns.keys())

        self.class_column.options = \
            [var for var in columns_list if self.datatype_dict[var] in (Config.TEXT_DATATYPES + Config.INT_DATATYPES)] 
        if self.class_column.options:
            self.class_column.value = self.class_column.options[0]
        else:
            self.class_column.value = None
        self.class_column.disabled = False
        
        self.id_column.options =  \
            [var for var in columns_list if self.datatype_dict[var] in Config.INT_DATATYPES and var != self.class_column.value]
        if self.id_column.options:
            self.id_column.value = self.id_column.options[0]
        else:
            self.id_column.value = None
        self.id_column.disabled = False
        
        self.data_columns.options = \
            [var for var in columns_list if var not in (self.class_column.value, self.id_column.value)]
        self.data_columns.value = []
        self.data_columns.disabled = False
    
    @property
    def data_settings(self) -> dict:
        """ Settings on which data to classify """
        return {
            "project": self.project.value,
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
        
        try:
            distribution = self.guihandler.get_class_distribution(self.data_settings)
        except Exception as e:
            message = f"Could not update summary for class: {current_class} because {e}"
            self.guihandler.logger.print_info(message)
        else: # No exception
            dist_items = '; '.join(f'{key} ({value})' for key, value in distribution.items())
            new_text = f"<em>Class column</em>: {current_class}<br>"
            new_text += f"<em>Distribution</em>: {dist_items}<br>"
            new_text += f"<em>Total rows</em>: {sum(distribution.values())}" 
            
            self.class_summary.value = new_text
        

    def update_id_column(self) -> None:
        """ Removes class_column from the id_column options """

        self.id_column.options = \
            [var for var in list(self.datatype_dict.keys()) if var != self.class_column.value and self.datatype_dict[var] in Config.INT_DATATYPES]
       
        #self.update_item("id_column", { "options": self.options_excluding_selected(self.class_column) })

    def update_data_columns(self) -> None:
        """ Removes id column from data columns """

        self.data_columns.options = \
            [var for var in list(self.datatype_dict.keys()) if var not in (self.class_column.value, self.id_column.value)]

        #self.update_item("data_columns", { "options": self.options_excluding_selected(self.id_column) })


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

        ready_for_classifier = num_vars > 0 and self.id_column.value and self.class_column.value
        if ready_for_classifier:
            self.enable_button("continuation_button")
        else:
            self.disable_button("continuation_button")
        


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

    def data_form(self) -> widgets.Box:
        return self.create_form(self.forms["data"], widgets.Box)
    
    def checkboxes_form(self) -> widgets.Box:
        return self.create_form(self.forms["checkboxes"], widgets.HBox)
    
    def algorithm_form(self) -> widgets.Box:
        return self.create_form(self.forms["algorithm"], widgets.Box)
    
    def data_handling_form(self) -> widgets.Box:
        return self.create_form(self.forms["data_handling"], widgets.HBox)
    
    def text_handling_form(self) -> widgets.Box:
        return self.create_form(self.forms["text_handling"], widgets.HBox)
    
    def debug_form(self) -> widgets.Box:
        return self.create_form(self.forms["debug"], widgets.HBox)
    
    def progress_form(self) -> widgets.Box:
        return self.create_form(self.forms["progress"], widgets.HBox)
    

    def create_form(self, children: list, boxtype: Callable) -> widgets.Box:
        if boxtype == widgets.Box:
            return boxtype(children, layout=self.get_box_layout())
        
        if boxtype == widgets.HBox:
            return boxtype(children)
        
        raise ValueError(f"boxtype needs to be a Box or HBox")
    
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
        
        return self.widgets[name]
        
    @property
    def progress(self) -> tuple:
        return (self.progress_bar, self.progress_label)

    
    @property
    def logo(self) -> widgets.Image:
        """ This checks the if here, since the initial state requires loading an image """
        name = sys._getframe(  ).f_code.co_name # Current function name

        if name not in self.widgets:
            logo = open(self.logo_image, "rb")

            return self._load_widget(name, {
                "value": logo.read()
            })

        return self.widgets[name]

    @property
    def welcome(self) -> widgets.HTML:
        """ The initial data is entirely from the json """
        name = sys._getframe(  ).f_code.co_name # Current function name
        
        return self._load_widget(name)
    
    @property
    def project(self) -> widgets.Text:
        """ The initial data is entirely from the json """
        name = sys._getframe(  ).f_code.co_name # Current function name
        
        return self._load_widget(name)

    
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
        return self._load_widget(name)

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
    def train_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
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
    def algorithm_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.algorithm_dropdown_handler, \
            calculated_params={"options": Algorithm.get_sorted_list()})
        
    @property
    def preprocess_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.preprocess_dropdown_handler, \
            calculated_params={"options": Preprocess.get_sorted_list()})
        
    
    @property
    def scoremetric_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, calculated_params={
            "options": ScoreMetric.get_sorted_list()
        })
        

    @property
    def reduction_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name, handler=self.eventhandler.reduction_dropdown_handler, \
            calculated_params={"options": Reduction.get_sorted_list()
        })
    
    
    @property
    def smote_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def undersample_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
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
        return self._load_widget(name)
        
    
    @property
    def categorize_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    @property
    def categorize_columns(self) -> widgets.SelectMultiple:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def filter_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
    @property
    def filter_slider(self) -> widgets.IntSlider:
        name = sys._getframe(  ).f_code.co_name # Current function name
        return self._load_widget(name)
        
    
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
    def mispredicted_gridbox(self) -> widgets.Label:
        name = sys._getframe().f_code.co_name # Current function name
        return self._load_widget(name)
        
    def display_gui(self) -> None:
        """ Displays the elements in the given order """
        
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
                    display(widget)