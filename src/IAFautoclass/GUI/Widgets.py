from __future__ import annotations
from enum import Enum
from pathlib import Path
import sys

from typing import Callable, Protocol
import ipywidgets as widgets
from sklearn.utils import Bunch

from Config import Config, Reduction, Algorithm, Preprocess, ScoreMetric
from IAFExceptions import GuiWidgetsException

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

    def should_run_event(self, event: Bunch) -> bool:
        """ This only returns True if event.name is 'index' and event.new > 0 or lock_observe_1 == False """
        if self.lock_observe_1:
            return False

        if event.name != "index":
            return False
        
        if event.new == 0:
            return False

        return True
    
    def print_event_info(self, caller: str, event: Bunch) -> None:
        """ Just prints some info about the event to the output, for debug """
        print(f"{caller}: {event.name=}, {event.old=}, {event.new=}")

    def data_catalogs_dropdown(self, event: Bunch) -> None:
        """ Handler for data_catalogs_dropdown
            Events:
                value_change: Updates the data tables dropdown when catalog is picked
        """
        # Change to default values below if it's changing in ways
        if not self.should_run_event(event):
            return
        
        self.widgets.update_data_tables_dropdown()
    
    def data_tables_dropdown(self, event: Bunch) -> None:
        """ Handler for data_tables_dropdown
            Events:
                value_change: Sets the models dropdown to enabled when table is picked
        """
        # Change to default values below if it's changing in ways
        if not self.should_run_event(event):
            return
        
        self.widgets.update_item("models_dropdown", {"disabled": False})

    def models_dropdown(self, event: Bunch) -> None:
        """ Handler for models_dropdown
            Events:
                value_change: Triggers several changes when model is set
        """
        # Change to default values below if it's changing in ways
        if event.name != "index":
            return
        
        if event.new == 0: # Moves to the empty value
            return
        
        if not self.lock_observe_1:
            self.widgets.update_class_id_data_columns()
        
        if event.new == 1: # This is the default train option
            self.widgets.update_item("continuation_button", {"disabled": True})
            return
        
        self.widgets.update_from_model_config()

    def class_column(self, event: Bunch) -> None:
        """ Handler for class_column
            Events:
                value_change: Changes the id and data columns, updates summary
        """
        if event.name not in ["index", "disabled"]:
            return
        
        if event.name == "disabled" and event.name:
            # This is still disabled
            return
        
        """ update_id_and_data_columns
        """
        self.widgets.update_class_summary()
        self.widgets.update_id_column() # Event.name "options"

    def id_column(self, event: Bunch) -> None:
        """ Handler for id_column
            Events:
                value_change: Updates the data column option when id is set
            update_data_columns
        """
        if self.lock_observe_1: 
            return
        
        if event.name not in ["index", "options"]:
            return
        
        self.widgets.update_data_columns()

    def data_columns(self, event: Bunch) -> None:
        """ Handler for data_columns
            Events:
                value_change: Updates the text columns and continuation_button states when data options changes
            update_text_columns_and_enable_button
        """
        if self.lock_observe_1: 
            return
        
        if event.name not in ["index", "options"]:
            return

        self.widgets.update_text_columns()
        self.widgets.update_ready() # At the moment enables/disables the continuation_button
    
    def continuation_button_was_clicked(self, event: Bunch) -> None:
        """ Callback: Sets various states based on the value in models dropdown. """
        self.lock_observe_1 = True

    def start_button_was_clicked(self, event: Bunch) -> None:
        """ Callback: Changes the state to be read-only, starts the classifier (or the class summary) """
    
        
    def unimplemented_handler(self, event) -> None:
        """ Just a basic handler before I've done the correct one """
        self.print_event_info("Unimplemented", event)


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
    """ Empty for now"""
    def get_class_distribution(self, data_settings: dict) -> dict:
        """ Help function to avoid Widgets knowing about classifier DataLayer """

class Widgets:
    """ Creates and populates the widgets of the GUI """

    def __init__(self, src_path: Path, GUIhandler: GUIhandler, model_path: Path = None, datalayer: DataLayer = None) -> None:
        self.logo_image = src_path / "images/iaf-logo.png"
        self.src_dir = src_path # TODO: Remove if not needed, but keep for now
        self.datalayer = datalayer
        self.eventhandler = EventHandler(self)
        self.guihandler = GUIhandler
        self.model_path = model_path if model_path else src_path / Config.DEFAULT_MODELS_PATH
        
        self.widgets = {}
        self.forms = {
            "catalog": [self.data_catalogs_dropdown, self.data_tables_dropdown],
            "data": [self.class_column, self.id_column, self.data_columns, self.text_columns],
            "checkboxes": [self.train_checkbox, self.predict_checkbox, self.mispredicted_checkbox],
            "algorithm": [self.reduction_dropdown, self.algorithm_dropdown, self.preprocess_dropdown, self.scoremetric_dropdown],
            "data_handling": [self.smote_checkbox, self.undersample_checkbox, self.testdata_slider, self.iterations_slider],
            "text_handling": [self.categorize_checkbox, self.categorize_columns, self.encryption_checkbox, self.filter_checkbox, self.filter_slider],
            "debug": [self.num_rows, self.show_info_checkbox, self.num_variables],
            "progress": [self.progress_bar, self.progress_label]
        }

        # We need a dictionary to keep track of datatypes
        self.datatype_dict = None
    
    def load_contents(self, datalayer: DataLayer = None) -> None:
        """ This is when we set content loaded from datasource """
        if datalayer:
            self.datalayer = datalayer

        if not self.datalayer:
            raise GuiWidgetsException("Data Layer not initialized")

        self.update_data_catalogs_dropdown()
        self.update_models_dropdown_options()

    def update_from_model_config(self, model: str = None ) -> None:
        """ Sets values based on the config in the chosen model """
        value = model if model else self.models_dropdown.value
        model_path = self.model_path / value
        
        config = Config.load_config_from_model_file(model_path)

        # Values from config
        self.class_column.value = config.get_class_column_name()
        self.id_column.value = config.get_id_column_name()
        self.data_columns.value = config.get_data_column_names()
        self.text_columns.value = config.get_text_column_names
        self.algorithm_dropdown.value = config.get_algorithm().name
        self.preprocess_dropdown.value = config.get_preprocessor().name
        self.reduction_dropdown.value = config.get_feature_selection().name
        self.num_variables.value = config.get_num_selected_features()
        self.filter_checkbox.value = config.use_stop_words()
        self.filter_slider.value = config.get_stop_words_threshold_percentage()
        self.encryption_checkbox.value = config.should_hex_encode()
        self.categorize_checkbox.value = config.use_categorization()
        self.categorize_columns.options = config.get_text_column_names
        self.categorize_columns.value =  config.get_categorical_text_column_names()
        self.testdata_slider.value = config.get_test_size_percentage()
        self.smote_checkbox.value = config.use_smote()
        self.undersample_checkbox.value = config.use_undersample()

        # Disabled items:
        self.disable_items(["class_column", "id_column", "data_columns", "text_columns", "categorize_columns"])
        
        # Enabled items:
        self.enable_items(["continuation_button"])
        
    def disable_items(self, items: list) -> None:
        """ Disable all items in the list """
        for name in items:
            item = self.get_item_or_error(name)
            if hasattr(item, "disabled"):
                setattr(item, "disabled", True)

    def enable_items(self, items: list) -> None:
        """ Enable all items in the list """
        for name in items:
            item = self.get_item_or_error(name)
            if hasattr(item, "disabled"):
                setattr(item, "disabled", False)

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

        self.class_column.options = columns_list
        self.class_column.value = columns_list[0]
        self.class_column.disabled = False
        
        self.id_column.options = columns_list[1:]
        self.id_column.value = columns_list[1]
        self.id_column.disabled = False
        
        self.data_columns.options = columns_list[2:]
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

    def _get_data_columns(self, type: WidgetConstant) -> list:
        """ Gets a list of columns connected to data """
        text_columns = list(self.text_columns.value)
        
        if type == WidgetConstant.DATA_NUMERICAL:
            return [col for col in self.data_columns.value if not col in text_columns]

        if type == WidgetConstant.DATA_TEXT:
            return text_columns


    def update_class_summary(self):
        """ Sets the class summary """
        current_class = self.class_column.value
        
        try:
            distribution = self.guihandler.get_class_distribution(self.data_settings)
        except Exception as e:
            message = f"Could not update summary for class: {current_class} because {e}"
            self.guihandler.logger.print_info(message)
        else: # No exception
            new_text = f"Class column: '{current_class}', with distribution: {str(distribution)[1:-1]}, in total: {sum(distribution.values())} rows"
            
            self.class_summary.value = new_text
        

    def update_id_column(self) -> None:
        """ Removes class_column from the id_column options """
        self.update_item("id_column", { "options": self.options_excluding_selected(self.class_column) })

    def update_data_columns(self) -> None:
        """ Removes id column from data columns """
        self.update_item("data_columns", { "options": self.options_excluding_selected(self.id_column) })


    def options_excluding_selected(self, source: widgets.Select) -> list:
        """ Gets the chosen select and returns all options except the current one """
        
        return [option for option in source.options if option != source.value]

    def update_text_columns(self) -> None:
        """ Sets any text-columns as pickable """
        updates = {
            "options": self.data_columns.value,
            "values": [col for col in self.data_columns.value if self.datatype_dict[col] in Config.TEXT_DATATYPES],
            "disabled": False
        }
        self.update_item("text_columns", updates)

    def update_ready(self) -> None:
        """ Sets whether it can continue into Classifier Settings  """
        num_vars = len(self.data_columns.value)
        self.num_variables.value = num_vars

        ready_for_classifier = num_vars > 0
        self.update_item("continuation_button", {"disabled": ready_for_classifier})

    def get_item_or_error(self, name: str) -> widgets.Widget:
        if not hasattr(self, name):
            raise ValueError(f"Item {name} does not exist")

        return getattr(self, name)

    def update_item(self, name: str, updates: dict) -> None:
        item = self.get_item_or_error(name)
        
        for attribute, value in updates.items():
            if hasattr(item, attribute):
                setattr(item, attribute, value)
        
        #if hasattr(item, "foo"):
        #    print(f"Weirdly, {name} has a foo")
        #else:
        #     print(f"Correct, {name} doesn't have a foo")

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
    
    @property
    def progress(self) -> tuple:
        return (self.progress_bar, self.progress_label)

    
    @property
    def logo(self) -> widgets.Image:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            logo = open(self.logo_image, "rb")
            self.widgets[name] = widgets.Image(
                value = logo.read(),
                format = "png",
            )
        
        return self.widgets[name]

    @property
    def welcome(self) -> widgets.HTML:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.HTML(
                value="<h1>Welcome to IAF automatic classification!</h1>"
            )
        
        return self.widgets[name]
    
    @property
    def project(self) -> widgets.Text:
        name = sys._getframe(  ).f_code.co_name # Current function name
        # This is defined outside of the rest to test proof-of-concept of defining the
        # intial values in a json file
        kwargs = {
            "value": "default",
            "placeholder": "Project name",
            "description": "Project name:",
            "description_tooltip": "Enter a distinct project name for the model"
        }
        if name not in self.widgets:
            self.widgets[name] =  widgets.Text(
                **kwargs
            )
        
        return self.widgets[name]

    
    @property
    def data_catalogs_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Dropdown(
                options = ["N/A"],
                value = "N/A",
                description = "Catalogs:",
                disabled = True,
                description_tooltip = "These are the catalogs of choice"
            )
            self.widgets[name].observe(handler=self.eventhandler.data_catalogs_dropdown)
        
        return self.widgets[name]
    
    @property
    def data_tables_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Dropdown(
                options = ["N/A"],
                value = "N/A",
                description = "Tables:",
                disabled = True,
                description_tooltip = "These are the tables of choice"
            )
            self.widgets[name].observe(handler=self.eventhandler.data_tables_dropdown)
        
        return self.widgets[name]
    
    @property
    def models_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Dropdown(
                options = ["N/A"],
                value = "N/A",
                description = "Models:",
                disabled = True,
                description_tooltip = "You can train a new model or use a previously trained one"
            )
            self.widgets[name].observe(handler=self.eventhandler.models_dropdown)
        
        return self.widgets[name]


    @property
    def class_column(self) -> widgets.RadioButtons:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] = widgets.RadioButtons(
                options = ["N/A"],
                value = "N/A",
                description = "Class:",
                disabled = True,
                description_tooltip = "Pick the column to use as class label"
            )
            self.widgets[name].observe(handler=self.eventhandler.class_column)
            
        
        return self.widgets[name]
    
    @property
    def id_column(self) -> widgets.RadioButtons:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.RadioButtons(
                options = ["N/A"],
                value = "N/A",
                description = "Unique id:",
                disabled = True,
                description_tooltip = "Pick the column to use as unique identifier"
            )
            self.widgets[name].observe(handler=self.eventhandler.id_column)
            
        
        return self.widgets[name]

    @property
    def data_columns(self) -> widgets.SelectMultiple:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.SelectMultiple(
                options = ["N/A"],
                value = [],
                description = "Pick data columns:",
                disabled = True,
                description_tooltip = "Pick the columns with the data to be used in the classification"
            )
            self.widgets[name].observe(handler=self.eventhandler.data_columns)

        return self.widgets[name]

    @property
    def text_columns(self) -> widgets.SelectMultiple:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.SelectMultiple(
            options = ["N/A"],
            value = [],
            description="Is text:",
            disabled = True,
            description_tooltip = "Mark the data columns to be interpreted as text (some are marked by default from their SQL datatype)"
        )
        
        return self.widgets[name]

    @property
    def data_numerical_columns(self) -> list:
        return self._get_data_columns(WidgetConstant.DATA_NUMERICAL)

    @property
    def data_text_columns(self) -> list:
        return self._get_data_columns(WidgetConstant.DATA_TEXT)
    
    @property
    def class_summary(self) -> widgets.HTML:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.HTML(
                value="<em>Class summary</em>: N/A"
            )
        
        return self.widgets[name]
    
    @property
    def continuation_button(self) -> widgets.Button:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =   widgets.Button(
                description="Continue",
                disabled=True,
                button_style="success", 
                tooltip="Continue with the process using these settings",
                icon="check" 
            )
            self.widgets[name].on_click(callback=self.eventhandler.continuation_button_was_clicked)
        return self.widgets[name]
    
    @property
    def train_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = False,
                description = "Mode: Train",
                disabled = True,
                indent = True,
                description_tooltip = "A new model will be trained"
            )
        
        return self.widgets[name]
    
    @property
    def predict_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = False,
                description = "Mode: Predict",
                disabled = True,
                indent = True,
                description_tooltip = "The model of choice will be used to make predictions"
            )
            
        return self.widgets[name]
    
    @property
    def mispredicted_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = False,
                description = "Mode: Display mispredictions",
                disabled = True,
                indent = True,
                description_tooltip = "The classifier will display mispredicted training data for manual inspection and correction"
            )
        
        return self.widgets[name]
    
    @property
    def algorithm_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Dropdown(
                options = Algorithm.get_sorted_list(),
                description = "Algorithm:",
                disabled = True,
                description_tooltip = "Pick which algorithms to use"
            )
        
        return self.widgets[name]
    
    @property
    def preprocess_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Dropdown(
                options = Preprocess.get_sorted_list(),
                description = "Preprocess:",
                disabled = True,
                description_tooltip = "Pick which data preprocessors to use"
            )
        
        return self.widgets[name]
    
    @property
    def scoremetric_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Dropdown(
                options = ScoreMetric.get_sorted_list(),
                description = "Score metric:",
                disabled = True,
                description_tooltip = "Pick by which method algorithm performances are measured and compared"
            )
        
        return self.widgets[name]

    @property
    def reduction_dropdown(self) -> widgets.Dropdown:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Dropdown(
                options = Reduction.get_sorted_list(),
                description = "Reduction:",
                disabled = True,
                description_tooltip = "Pick by which method the number of features (variables) are reduced"
            )
        
        return self.widgets[name]
    
    @property
    def smote_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = False,
                description = "SMOTE",
                disabled = True,
                indent = True,
                description_tooltip = "Use SMOTE (synthetic minority oversampling technique) for training data"
            )
        
        return self.widgets[name]
    
    @property
    def undersample_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = False,
                description = "Undersampling",
                disabled = True,
                indent = True,
                description_tooltip = "Use undersampling of majority training data"
            )
        
        return self.widgets[name]
    
    @property
    def testdata_slider(self) -> widgets.IntSlider:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.IntSlider(
                value = 20,
                min = 0,
                max = 100,
                step = 1,
                description = "Testdata (%):",
                disabled = True,
                continuous_update = False,
                orientation = "horizontal",
                readout = True,
                readout_format = "d",
                description_tooltip = "Set how large evaluation size of training data will be"
            )
        
        return self.widgets[name]
    
    @property
    def iterations_slider(self) -> widgets.IntSlider:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.IntSlider(
                value = 20000,
                min = 1000,
                max = 100000,
                step = 100,
                description = "Max.iter:",
                disabled = True,
                continuous_update = False,
                orientation = "horizontal",
                readout = True,
                readout_format = "d",
                description_tooltip = "Set how many iterations to use at most"
            )
        
        return self.widgets[name]
    
    @property
    def encryption_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = True,
                description = "Text: Encryption",
                description_tooltip = "Use encryption on text",
                disabled = True,
                indent = True
            )
        
        return self.widgets[name]
    
    @property
    def categorize_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = True,
                description = "Text: Categorize",
                description_tooltip = "Use categorization on text",
                disabled = True,
                indent = True
            )
        
        return self.widgets[name]
    
    @property
    def categorize_columns(self) -> widgets.SelectMultiple:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.SelectMultiple(
                options = ["N/A"],
                value = [],
                description="Categorize:",
                disabled = True,
                description_tooltip = "Mark text columns to force them to be categorized (text columns with up to 30 distinct values will be if checkbox is checked)"
            )
        
        return self.widgets[name]
    
    @property
    def filter_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = False,
                description = "Text: Filter",
                description_tooltip = "Use Better Tooltip",
                disabled = True,
                indent = True
            )
        
        return self.widgets[name]
    
    @property
    def filter_slider(self) -> widgets.IntSlider:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.IntSlider(
                value = 100,
                min = 0,
                max = 100,
                step = 1,
                description = "Doc.freq. (%):",
                disabled = True,
                continuous_update = False,
                orientation = "horizontal",
                readout = True,
                readout_format = "d",
                description_tooltip = "Set document frequency limit for filtering of stop words"
            )
        
        return self.widgets[name]
    
    @property
    def num_rows(self) -> widgets.IntText:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.IntText(
                value = 0,
                description = "Data limit:",
                disabled = True,
                description_tooltip = "During debugging or testing, limiting the number of data rows can be beneficial"
            )
        
        return self.widgets[name]
    
    @property
    def show_info_checkbox(self) -> widgets.Checkbox:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Checkbox(
                value = False,
                description = "Show info",
                description_tooltip = "Show detailed printout in output",
                disabled = True,
                indent = True
            )
        
        return self.widgets[name]


    @property
    def num_variables(self) -> widgets.IntText:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.IntText(
                value = 0,
                description = "Variables:",
                disabled = True,
                description_tooltip = "The number of variables used is shown in this box"
            )
        
        return self.widgets[name]
    
    @property
    def start_button(self) -> widgets.Button:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =   widgets.Button(
                description="Start",
                disabled=True,
                button_style="success", 
                tooltip="Run the classifier using the current settings",
                icon="check" 
            )
            self.widgets[name].on_click(callback = self.eventhandler.start_button_was_clicked)
        return self.widgets[name]

    @property
    def progress_bar(self) -> widgets.FloatProgress:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.FloatProgress(
                value=0.0,
                min=0.0,
                max=1.0,
                description="Progress:",
                bar_style="info",
                style={"bar_color": "#004B99"},
                orientation="horizontal",
                description_tooltip = "The computational process is shown here"
            )
        
        return self.widgets[name]
    
    @property
    def progress_label(self) -> widgets.HTML:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.HTML(
                value=""
            )
        
        return self.widgets[name]
    
    @property
    def output(self) -> widgets.Output:
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =  widgets.Output(
                layout={"border": "2px solid grey"}
            )  
        
        return self.widgets[name]
    
    @property
    def mispredicted_gridbox(self) -> widgets.Label:
        # TODO: Why is this a label?
        name = sys._getframe(  ).f_code.co_name # Current function name
        if name not in self.widgets:
            self.widgets[name] =   widgets.Label(
                value = "No mispredicted training data was detected yet",
                description_tooltip = "Use to manually inspect and correct mispredicted training data"
            )
        
        return self.widgets[name]


    def display_gui(self) -> None:
        """ Displays the elements in the given order """
        
        items = [
            "logo",
            "welcome",
            "project",
            "data_catalogs_dropdown",
            "data_tables_dropdown",
            "models_dropdown",
            "data_form",
            "class_summary",
            "continuation_button",
            "checkboxes_form",
            "algorithm_form",
            "data_handling_form",
            "text_handling_form",
            "debug_form",
            "start_button",
            "progress_form",
            "output"
        ]

        for item in items:
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