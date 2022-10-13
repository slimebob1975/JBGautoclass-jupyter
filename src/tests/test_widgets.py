from pathlib import Path
import pytest
import ipywidgets

from GUI.Widgets import Widgets

class MockDataLayer:
    """ The minimum amount of datalayer needed for the Widgets """
    
    def get_catalogs_as_options(self) -> list:
        """ Used in the GUI, to get the catalogs """
        return ["catalog", "catalog2", "catalog3"]
    
    def get_tables_as_options(self) -> list:
        """ Used in the GUI, to get the tables """
        return ["table1", "table2", "table3"]

    def get_trained_models_from_files(self, model_path: str, model_file_extension: str, preface: list = None) -> list:
        """ Used in the GUI, get a list of pretrained models (base implementation assumes files)"""
        return ["config-save.sav", "model-save.sav"]

    def get_id_columns(self, **kwargs) -> list:
        """ Used in the GUI, gets name and type for columns """



class MockGUIhandler:
    """ Empty for now"""

def get_src_path() -> Path:
    this_file = Path(__file__)
    return this_file.parents[1] / "IAFautoclass"

def get_model_path() -> Path:
    this_file = Path(__file__)
    return this_file.parent / "fixtures/"

def get_logo_path() -> Path:
    return get_src_path() / "images/iaf-logo.png"

@pytest.fixture
def widgets() -> Widgets:
    dl = MockDataLayer()
    gh = MockGUIhandler()
    
    return Widgets(src_path=get_src_path(), GUIhandler=gh, model_path=get_model_path(), datalayer=dl)

@pytest.fixture
def loaded_widgets(widgets) -> Widgets:
    widgets.load_contents()
    widgets.display_gui()

    return widgets

@pytest.fixture
def widget_parameters() -> dict:
    logo = open(get_logo_path(), "rb")
    return {
        "logo": {
            "value": logo.read(),
            "format": "png"
        },
        "welcome": {
            "value": "<h1>Welcome to IAF automatic classification!</h1>"
        },
        "project": {
            "pre__load": {
                "value": "default",
                "placeholder": "Project name",
                "description": "Project name:",
                "description_tooltip": "Enter a distinct project name for the model"
            }
        },
        "data_catalogs_dropdown": {
            "pre__load": {
                "options": ["N/A"],
                "value": "N/A",
                "disabled": True,
                "description": "Catalogs:",
                "description_tooltip": "These are the catalogs of choice"
            },
            "post__load": {
                "options": ("catalog", "catalog2", "catalog3"),
                "value": "catalog",
                "disabled": False,
                "description": "Catalogs:",
                "description_tooltip": "These are the catalogs of choice"
            }
        },
        "data_tables_dropdown": {
            "pre__load": {
                "options": ["N/A"],
                "value": "N/A",
                "disabled": True,
                "description": "Tables:",
                "description_tooltip": "These are the tables of choice"
            }
        },
        "models_dropdown": {
            "pre__load": {
                "options": ["N/A"],
                "value": "N/A",
                "description": "Models:",
                "disabled": True,
                "description_tooltip": "You can train a new model or use a previously trained one"
            },
            "post__load": {
                "options": ("config-save.sav", "model-save.sav"),
                "value": "config-save.sav",
                "description": "Models:",
                "disabled": True,
                "description_tooltip": "You can train a new model or use a previously trained one"
            }
        },
        "class_column": {
            "pre__load": {
                "options": ["N/A"],
                "value": "N/A",
                "disabled": True,
                "description": "Class:",
                "description_tooltip": "Pick the column to use as class label"
            }
        },
        "id_column": {
            "pre__load": {
                "options": ["N/A"],
                "value": "N/A",
                "disabled": True,
                "description": "Unique id:",
                "description_tooltip": "Pick the column to use as unique identifier"
            }
        },
        "data_columns": {
            "pre__load": {
                "options": ["N/A"],
                "value": (),
                "disabled": True,
                "description": "Pick data columns:",
                "description_tooltip": "Pick the columns with the data to be used in the classification"
            }
        },
        "text_columns": {
            "pre__load": {
                "options": ["N/A"],
                "value": (),
                "disabled": True,
                "description": "Is text:",
                "description_tooltip": "Mark the data columns to be interpreted as text (some are marked by default from their SQL datatype)"
            }
        },
        "class_summary": {
            "pre__load": {
                "value": "<em>Class summary</em>: N/A"
            }
        },
        "continuation_button": {
            "description": "Continue",
            "button_style": "primary", 
            "disabled": True,
            "tooltip": "Continue with the process using these settings",
            "icon": "check" 
        },
        "train_checkbox": {
            "pre__load": {
                "value": False,
                "disabled": True,
                "indent": True,
                "description": "Mode: Train",
                "description_tooltip": "A new model will be trained"
            }
        },
        "predict_checkbox": {
            "pre__load": {
                "value": False,
                "disabled": True,
                "indent": True,
                "description": "Mode: Predict",
                "description_tooltip": "The model of choice will be used to make predictions"
            }
        },
        "mispredicted_checkbox": {
            "pre__load": {
                "value": False,
                "disabled": True,
                "indent": True,
                "description": "Mode: Display mispredictions",
                "description_tooltip": "The classifier will display mispredicted training data for manual inspection and correction"
            }
        },
        "algorithm_dropdown": {
            "pre__load": {
                "value": "ALL",
                "description": "Algorithm:",
                "disabled": True,
                "description_tooltip": "Pick which algorithms to use"
            },
        },
        "preprocess_dropdown": {
            "pre__load": {
                "value": "ALL",
                "description": "Preprocess:",
                "disabled": True,
                "description_tooltip": "Pick which data preprocessors to use"
            },
        },
        "scoremetric_dropdown": {
            "pre__load": {
                "value": "accuracy",
                "description": "Score metric:",
                "disabled": True,
                "description_tooltip": "Pick by which method algorithm performances are measured and compared"
            },
        },
        "reduction_dropdown": {
            "pre__load": {
                "value": "NON",
                "description": "Reduction:",
                "disabled": True,
                "description_tooltip": "Pick by which method the number of features (variables) are reduced"
            },
        },
        "smote_checkbox": {
            "pre__load": {
                "value": False,
                "disabled": True,
                "indent": True,
                "description": "SMOTE",
                "description_tooltip": "Use SMOTE (synthetic minority oversampling technique) for training data"
            }
        },
        "undersample_checkbox": {
            "pre__load": {
                "value": False,
                "disabled": True,
                "indent": True,
                "description": "Undersampling",
                "description_tooltip": "Use undersampling of majority training data"
            }
        },
        "testdata_slider": {
            "pre__load": {
                "value": 20,
                "min": 0,
                "max": 100,
                "step": 1,
                "disabled": True,
                "continuous_update": False,
                "orientation": "horizontal",
                "readout": True,
                "readout_format": "d",
                "description": "Testdata (%):",
                "description_tooltip": "Set how large evaluation size of training data will be"
            }
        },
        "iterations_slider": {
            "pre__load": {
                "value": 20000,
                "min": 1000,
                "max": 100000,
                "step": 100,
                "disabled": True,
                "continuous_update": False,
                "orientation": "horizontal",
                "readout": True,
                "readout_format": "d",
                "description": "Max.iter:",
                "description_tooltip": "Set how many iterations to use at most"
            }
        },
        "encryption_checkbox": {
            "pre__load": {
                "value": True,
                "disabled": True,
                "indent": True,
                "description": "Text: Encryption",
                "description_tooltip": "Use encryption on text"
            }
        },
        "categorize_checkbox": {
            "pre__load": {
                "value": True,
                "disabled": True,
                "indent": True,
                "description": "Text: Categorize",
                "description_tooltip": "Use categorization on text"
            }
        },
        "categorize_columns": {
            "pre__load": {
                "options": ["N/A"],
                "value": (),
                "disabled": True,
                "description": "Categorize:",
                "description_tooltip": "Mark text columns to force them to be categorized (text columns with up to 30 distinct values will be if checkbox is checked)"
            }
        },
        "filter_checkbox": {
            "pre__load": {
                "value": False,
                "disabled": True,
                "indent": True,
                "description": "Text: Filter",
                "description_tooltip": "Use Better Tooltip"
            }
        },
        "filter_slider": {
            "pre__load": {
                "value": 100,
                "min": 0,
                "max": 100,
                "step": 1,
                "disabled": True,
                "continuous_update": False,
                "orientation": "horizontal",
                "readout": True,
                "readout_format": "d",
                "description": "Doc.freq. (%):",
                "description_tooltip": "Set document frequency limit for filtering of stop words"
            }
        },
        "data_limit": {
            "pre__load": {
                "value": 0,
                "disabled": True,
                "description": "Data limit:",
                "description_tooltip": "During debugging or testing, limiting the number of data rows can be beneficial"
            }
        },
        "show_info_checkbox": {
            "pre__load": {
                "value": False,
                "disabled": True,
                "indent": True,
                "description": "Show info",
                "description_tooltip": "Show detailed printout in output"
            }
        },
        "num_variables": {
            "pre__load": {
                "value": 0,
                "disabled": True,
                "description": "Variables:",
                "description_tooltip": "The number of variables used is shown in this box"
            }
        },
        "start_button": {
            "pre__load": {
                "description": "Start",
                "button_style": "primary", 
                "disabled": True,
                "tooltip": "Run the classifier using the current settings",
                "icon": "check" 
            }
        },
        "progress_bar": {
            "pre__load": {
                "value": 0.0,
                "min": 0.0,
                "max": 1.0,
                "description": "Progress:",
                "bar_style": "info",
                "orientation": "horizontal",
                "description_tooltip": "The computational process is shown here"
            }
        },
        "progress_label": {
            "pre__load": {
                "value": ""
            }
        },
        "output": {
        },
        "mispredicted_gridbox": {
            "pre__load": {
                "value": "<p>No mispredicted training data was detected yet</p>"
            }
        },
    }

class TestWidgets:
    """ Testing the main class """

    def test_load_contents(self, widgets, widget_parameters) -> None:
        """ Ensures that the things that are initialized in load_contents have the right values """
        widgets.load_contents()
        
        data_catalogs_dropdown = widgets.data_catalogs_dropdown
        assert data_catalogs_dropdown.options == widget_parameters["data_catalogs_dropdown"]["post__load"]["options"]
        assert data_catalogs_dropdown.value == widget_parameters["data_catalogs_dropdown"]["post__load"]["value"]
        assert data_catalogs_dropdown.disabled == widget_parameters["data_catalogs_dropdown"]["post__load"]["disabled"]

        models_dropdown = widgets.models_dropdown
        assert models_dropdown.options == widget_parameters["models_dropdown"]["post__load"]["options"]
        assert models_dropdown.value == widget_parameters["models_dropdown"]["post__load"]["value"]
        assert models_dropdown.disabled == widget_parameters["models_dropdown"]["post__load"]["disabled"]

    def test_widget_properties(self, widgets, widget_parameters) -> None:
        """ Checks that the widget properties are correct initialized with the right parameters
        """

        # Logo, image
        logo = widgets.logo
        assert isinstance(logo, ipywidgets.Image)
        assert logo.value == widget_parameters["logo"]["value"]
        assert logo.format == widget_parameters["logo"]["format"]

        # Welcome, html
        welcome = widgets.welcome
        assert isinstance(welcome, ipywidgets.HTML)
        assert welcome.value == widget_parameters["welcome"]["value"]

        # Project, text
        project = widgets.project
        assert isinstance(project, ipywidgets.Text)
        assert project.value == widget_parameters["project"]["pre__load"]["value"]
        assert project.placeholder == widget_parameters["project"]["pre__load"]["placeholder"]
        assert project.description == widget_parameters["project"]["pre__load"]["description"]
        assert project.description_tooltip == widget_parameters["project"]["pre__load"]["description_tooltip"]

        # Data Catalogs, Dropdown
        data_catalogs_dropdown = widgets.data_catalogs_dropdown
        assert isinstance(data_catalogs_dropdown, ipywidgets.Dropdown)
        assert list(data_catalogs_dropdown.options) == widget_parameters["data_catalogs_dropdown"]["pre__load"]["options"]
        assert data_catalogs_dropdown.value == widget_parameters["data_catalogs_dropdown"]["pre__load"]["value"]
        assert data_catalogs_dropdown.disabled == widget_parameters["data_catalogs_dropdown"]["pre__load"]["disabled"]
        assert data_catalogs_dropdown.description == widget_parameters["data_catalogs_dropdown"]["pre__load"]["description"]
        assert data_catalogs_dropdown.description_tooltip == widget_parameters["data_catalogs_dropdown"]["pre__load"]["description_tooltip"]
        
        # Data Tables, Dropdown
        data_tables_dropdown = widgets.data_tables_dropdown
        assert isinstance(data_tables_dropdown, ipywidgets.Dropdown)
        assert list(data_tables_dropdown.options) == widget_parameters["data_tables_dropdown"]["pre__load"]["options"]
        assert data_tables_dropdown.value == widget_parameters["data_tables_dropdown"]["pre__load"]["value"]
        assert data_tables_dropdown.disabled == widget_parameters["data_tables_dropdown"]["pre__load"]["disabled"]
        assert data_tables_dropdown.description == widget_parameters["data_tables_dropdown"]["pre__load"]["description"]
        assert data_tables_dropdown.description_tooltip == widget_parameters["data_tables_dropdown"]["pre__load"]["description_tooltip"]

        # Models, Dropdown
        models_dropdown = widgets.models_dropdown
        assert isinstance(models_dropdown, ipywidgets.Dropdown)
        assert list(models_dropdown.options) == widget_parameters["models_dropdown"]["pre__load"]["options"]
        assert models_dropdown.value == widget_parameters["models_dropdown"]["pre__load"]["value"]
        assert models_dropdown.disabled == widget_parameters["models_dropdown"]["pre__load"]["disabled"]
        assert models_dropdown.description == widget_parameters["models_dropdown"]["pre__load"]["description"]
        assert models_dropdown.description_tooltip == widget_parameters["models_dropdown"]["pre__load"]["description_tooltip"]
        
        # Class column, RadioButton
        class_column = widgets.class_column
        assert isinstance(class_column, ipywidgets.RadioButtons)
        assert list(class_column.options) == widget_parameters["class_column"]["pre__load"]["options"]
        assert class_column.value == widget_parameters["class_column"]["pre__load"]["value"]
        assert class_column.disabled == widget_parameters["class_column"]["pre__load"]["disabled"]
        assert class_column.description == widget_parameters["class_column"]["pre__load"]["description"]
        assert class_column.description_tooltip == widget_parameters["class_column"]["pre__load"]["description_tooltip"]
        
        # ID column, RadioButton
        id_column = widgets.id_column
        assert isinstance(id_column, ipywidgets.RadioButtons)
        assert list(id_column.options) == widget_parameters["id_column"]["pre__load"]["options"]
        assert id_column.value == widget_parameters["id_column"]["pre__load"]["value"]
        assert id_column.disabled == widget_parameters["id_column"]["pre__load"]["disabled"]
        assert id_column.description == widget_parameters["id_column"]["pre__load"]["description"]
        assert id_column.description_tooltip == widget_parameters["id_column"]["pre__load"]["description_tooltip"]
        
        # Data Columns, SelectMultiple
        data_columns = widgets.data_columns
        assert isinstance(data_columns, ipywidgets.SelectMultiple)
        assert list(data_columns.options) == widget_parameters["data_columns"]["pre__load"]["options"]
        assert data_columns.value == widget_parameters["data_columns"]["pre__load"]["value"]
        assert data_columns.disabled == widget_parameters["data_columns"]["pre__load"]["disabled"]
        assert data_columns.description == widget_parameters["data_columns"]["pre__load"]["description"]
        assert data_columns.description_tooltip == widget_parameters["data_columns"]["pre__load"]["description_tooltip"]
        
        # Text Columns, SelectMultiple
        text_columns = widgets.text_columns
        assert isinstance(text_columns, ipywidgets.SelectMultiple)
        assert list(text_columns.options) == widget_parameters["text_columns"]["pre__load"]["options"]
        assert text_columns.value == widget_parameters["text_columns"]["pre__load"]["value"]
        assert text_columns.disabled == widget_parameters["text_columns"]["pre__load"]["disabled"]
        assert text_columns.description == widget_parameters["text_columns"]["pre__load"]["description"]
        assert text_columns.description_tooltip == widget_parameters["text_columns"]["pre__load"]["description_tooltip"]
        
        # class summary, html
        class_summary = widgets.class_summary
        assert isinstance(class_summary, ipywidgets.HTML)
        assert class_summary.value == widget_parameters["class_summary"]["pre__load"]["value"]

        # Continuation button, Button
        continuation_button = widgets.continuation_button
        assert isinstance(continuation_button, ipywidgets.Button)
        assert continuation_button.disabled == widget_parameters["continuation_button"]["disabled"]
        assert continuation_button.description == widget_parameters["continuation_button"]["description"]
        assert continuation_button.tooltip == widget_parameters["continuation_button"]["tooltip"]
        assert continuation_button.button_style == widget_parameters["continuation_button"]["button_style"]
        assert continuation_button.icon == widget_parameters["continuation_button"]["icon"]

        # Train Checkbox
        train_checkbox = widgets.train_checkbox
        assert isinstance(train_checkbox, ipywidgets.Checkbox)
        assert train_checkbox.value == widget_parameters["train_checkbox"]["pre__load"]["value"]
        assert train_checkbox.disabled == widget_parameters["train_checkbox"]["pre__load"]["disabled"]
        assert train_checkbox.indent == widget_parameters["train_checkbox"]["pre__load"]["indent"]
        assert train_checkbox.description == widget_parameters["train_checkbox"]["pre__load"]["description"]
        assert train_checkbox.description_tooltip == widget_parameters["train_checkbox"]["pre__load"]["description_tooltip"]
       
        # Predict Checkbox
        predict_checkbox = widgets.predict_checkbox
        assert isinstance(predict_checkbox, ipywidgets.Checkbox)
        assert predict_checkbox.value == widget_parameters["predict_checkbox"]["pre__load"]["value"]
        assert predict_checkbox.disabled == widget_parameters["predict_checkbox"]["pre__load"]["disabled"]
        assert predict_checkbox.indent == widget_parameters["predict_checkbox"]["pre__load"]["indent"]
        assert predict_checkbox.description == widget_parameters["predict_checkbox"]["pre__load"]["description"]
        assert predict_checkbox.description_tooltip == widget_parameters["predict_checkbox"]["pre__load"]["description_tooltip"]
       
        # Mispredicted Checkbox
        mispredicted_checkbox = widgets.mispredicted_checkbox
        assert isinstance(mispredicted_checkbox, ipywidgets.Checkbox)
        assert mispredicted_checkbox.value == widget_parameters["mispredicted_checkbox"]["pre__load"]["value"]
        assert mispredicted_checkbox.disabled == widget_parameters["mispredicted_checkbox"]["pre__load"]["disabled"]
        assert mispredicted_checkbox.indent == widget_parameters["mispredicted_checkbox"]["pre__load"]["indent"]
        assert mispredicted_checkbox.description == widget_parameters["mispredicted_checkbox"]["pre__load"]["description"]
        assert mispredicted_checkbox.description_tooltip == widget_parameters["mispredicted_checkbox"]["pre__load"]["description_tooltip"]
       
        # Algorithm, Dropdown
        algorithm_dropdown = widgets.algorithm_dropdown
        assert isinstance(algorithm_dropdown, ipywidgets.Dropdown)
        # Note, explicitly not checking the options, since they are based on Algorithm and subject to change too quickly
        assert algorithm_dropdown.value == widget_parameters["algorithm_dropdown"]["pre__load"]["value"]
        assert algorithm_dropdown.disabled == widget_parameters["algorithm_dropdown"]["pre__load"]["disabled"]
        assert algorithm_dropdown.description == widget_parameters["algorithm_dropdown"]["pre__load"]["description"]
        assert algorithm_dropdown.description_tooltip == widget_parameters["algorithm_dropdown"]["pre__load"]["description_tooltip"]
        
        # Preprocess, Dropdown
        preprocess_dropdown = widgets.preprocess_dropdown
        assert isinstance(preprocess_dropdown, ipywidgets.Dropdown)
        # Note, explicitly not checking the options, since they are based on Preprocess and subject to change too quickly
        assert preprocess_dropdown.value == widget_parameters["preprocess_dropdown"]["pre__load"]["value"]
        assert preprocess_dropdown.disabled == widget_parameters["preprocess_dropdown"]["pre__load"]["disabled"]
        assert preprocess_dropdown.description == widget_parameters["preprocess_dropdown"]["pre__load"]["description"]
        assert preprocess_dropdown.description_tooltip == widget_parameters["preprocess_dropdown"]["pre__load"]["description_tooltip"]
        
        # Score Metric, Dropdown
        scoremetric_dropdown = widgets.scoremetric_dropdown
        assert isinstance(scoremetric_dropdown, ipywidgets.Dropdown)
        # Note, explicitly not checking the options, since they are based on ScoreMetric and subject to change too quickly
        assert scoremetric_dropdown.value == widget_parameters["scoremetric_dropdown"]["pre__load"]["value"]
        assert scoremetric_dropdown.disabled == widget_parameters["scoremetric_dropdown"]["pre__load"]["disabled"]
        assert scoremetric_dropdown.description == widget_parameters["scoremetric_dropdown"]["pre__load"]["description"]
        assert scoremetric_dropdown.description_tooltip == widget_parameters["scoremetric_dropdown"]["pre__load"]["description_tooltip"]
        
        # Reduction, Dropdown
        reduction_dropdown = widgets.reduction_dropdown
        assert isinstance(reduction_dropdown, ipywidgets.Dropdown)
        # Note, explicitly not checking the options, since they are based on Reduction and subject to change too quickly
        assert reduction_dropdown.value == widget_parameters["reduction_dropdown"]["pre__load"]["value"]
        assert reduction_dropdown.disabled == widget_parameters["reduction_dropdown"]["pre__load"]["disabled"]
        assert reduction_dropdown.description == widget_parameters["reduction_dropdown"]["pre__load"]["description"]
        assert reduction_dropdown.description_tooltip == widget_parameters["reduction_dropdown"]["pre__load"]["description_tooltip"]

        # SMOTE Checkbox
        smote_checkbox = widgets.smote_checkbox
        assert isinstance(smote_checkbox, ipywidgets.Checkbox)
        assert smote_checkbox.value == widget_parameters["smote_checkbox"]["pre__load"]["value"]
        assert smote_checkbox.disabled == widget_parameters["smote_checkbox"]["pre__load"]["disabled"]
        assert smote_checkbox.indent == widget_parameters["smote_checkbox"]["pre__load"]["indent"]
        assert smote_checkbox.description == widget_parameters["smote_checkbox"]["pre__load"]["description"]
        assert smote_checkbox.description_tooltip == widget_parameters["smote_checkbox"]["pre__load"]["description_tooltip"]
       

        # Undersample Checkbox
        undersample_checkbox = widgets.undersample_checkbox
        assert isinstance(undersample_checkbox, ipywidgets.Checkbox)
        assert undersample_checkbox.value == widget_parameters["undersample_checkbox"]["pre__load"]["value"]
        assert undersample_checkbox.disabled == widget_parameters["undersample_checkbox"]["pre__load"]["disabled"]
        assert undersample_checkbox.indent == widget_parameters["undersample_checkbox"]["pre__load"]["indent"]
        assert undersample_checkbox.description == widget_parameters["undersample_checkbox"]["pre__load"]["description"]
        assert undersample_checkbox.description_tooltip == widget_parameters["undersample_checkbox"]["pre__load"]["description_tooltip"]
        
        # Testdata Slider
        testdata_slider = widgets.testdata_slider
        assert isinstance(testdata_slider, ipywidgets.IntSlider)
        assert testdata_slider.value == widget_parameters["testdata_slider"]["pre__load"]["value"]
        assert testdata_slider.min == widget_parameters["testdata_slider"]["pre__load"]["min"]
        assert testdata_slider.max == widget_parameters["testdata_slider"]["pre__load"]["max"]
        assert testdata_slider.step == widget_parameters["testdata_slider"]["pre__load"]["step"]
        assert testdata_slider.disabled == widget_parameters["testdata_slider"]["pre__load"]["disabled"]
        assert testdata_slider.continuous_update == widget_parameters["testdata_slider"]["pre__load"]["continuous_update"]
        assert testdata_slider.orientation == widget_parameters["testdata_slider"]["pre__load"]["orientation"]
        assert testdata_slider.readout == widget_parameters["testdata_slider"]["pre__load"]["readout"]
        assert testdata_slider.readout_format == widget_parameters["testdata_slider"]["pre__load"]["readout_format"]
        assert testdata_slider.description == widget_parameters["testdata_slider"]["pre__load"]["description"]
        assert testdata_slider.description_tooltip == widget_parameters["testdata_slider"]["pre__load"]["description_tooltip"]
       
        # Iterations Slider
        iterations_slider = widgets.iterations_slider
        assert isinstance(iterations_slider, ipywidgets.IntSlider)
        assert iterations_slider.value == widget_parameters["iterations_slider"]["pre__load"]["value"]
        assert iterations_slider.min == widget_parameters["iterations_slider"]["pre__load"]["min"]
        assert iterations_slider.max == widget_parameters["iterations_slider"]["pre__load"]["max"]
        assert iterations_slider.step == widget_parameters["iterations_slider"]["pre__load"]["step"]
        assert iterations_slider.disabled == widget_parameters["iterations_slider"]["pre__load"]["disabled"]
        assert iterations_slider.continuous_update == widget_parameters["iterations_slider"]["pre__load"]["continuous_update"]
        assert iterations_slider.orientation == widget_parameters["iterations_slider"]["pre__load"]["orientation"]
        assert iterations_slider.readout == widget_parameters["iterations_slider"]["pre__load"]["readout"]
        assert iterations_slider.readout_format == widget_parameters["iterations_slider"]["pre__load"]["readout_format"]
        assert iterations_slider.description == widget_parameters["iterations_slider"]["pre__load"]["description"]
        assert iterations_slider.description_tooltip == widget_parameters["iterations_slider"]["pre__load"]["description_tooltip"]
       
        # Encryption Checkbox
        encryption_checkbox = widgets.encryption_checkbox
        assert isinstance(encryption_checkbox, ipywidgets.Checkbox)
        assert encryption_checkbox.value == widget_parameters["encryption_checkbox"]["pre__load"]["value"]
        assert encryption_checkbox.disabled == widget_parameters["encryption_checkbox"]["pre__load"]["disabled"]
        assert encryption_checkbox.indent == widget_parameters["encryption_checkbox"]["pre__load"]["indent"]
        assert encryption_checkbox.description == widget_parameters["encryption_checkbox"]["pre__load"]["description"]
        assert encryption_checkbox.description_tooltip == widget_parameters["encryption_checkbox"]["pre__load"]["description_tooltip"]
       
        # Categorize Checkbox
        categorize_checkbox = widgets.categorize_checkbox
        assert isinstance(categorize_checkbox, ipywidgets.Checkbox)
        assert categorize_checkbox.value == widget_parameters["categorize_checkbox"]["pre__load"]["value"]
        assert categorize_checkbox.disabled == widget_parameters["categorize_checkbox"]["pre__load"]["disabled"]
        assert categorize_checkbox.indent == widget_parameters["categorize_checkbox"]["pre__load"]["indent"]
        assert categorize_checkbox.description == widget_parameters["categorize_checkbox"]["pre__load"]["description"]
        assert categorize_checkbox.description_tooltip == widget_parameters["categorize_checkbox"]["pre__load"]["description_tooltip"]

        # Categorize Columns, SelectMultiple
        categorize_columns = widgets.categorize_columns
        assert isinstance(categorize_columns, ipywidgets.SelectMultiple)
        assert list(categorize_columns.options) == widget_parameters["categorize_columns"]["pre__load"]["options"]
        assert categorize_columns.value == widget_parameters["categorize_columns"]["pre__load"]["value"]
        assert categorize_columns.disabled == widget_parameters["categorize_columns"]["pre__load"]["disabled"]
        assert categorize_columns.description == widget_parameters["categorize_columns"]["pre__load"]["description"]
        assert categorize_columns.description_tooltip == widget_parameters["categorize_columns"]["pre__load"]["description_tooltip"]
        
        # Filter Checkbox
        filter_checkbox = widgets.filter_checkbox
        assert isinstance(filter_checkbox, ipywidgets.Checkbox)
        assert filter_checkbox.value == widget_parameters["filter_checkbox"]["pre__load"]["value"]
        assert filter_checkbox.disabled == widget_parameters["filter_checkbox"]["pre__load"]["disabled"]
        assert filter_checkbox.indent == widget_parameters["filter_checkbox"]["pre__load"]["indent"]
        assert filter_checkbox.description == widget_parameters["filter_checkbox"]["pre__load"]["description"]
        assert filter_checkbox.description_tooltip == widget_parameters["filter_checkbox"]["pre__load"]["description_tooltip"]

        # Filter Slider
        filter_slider = widgets.filter_slider
        assert isinstance(filter_slider, ipywidgets.IntSlider)
        assert filter_slider.value == widget_parameters["filter_slider"]["pre__load"]["value"]
        assert filter_slider.min == widget_parameters["filter_slider"]["pre__load"]["min"]
        assert filter_slider.max == widget_parameters["filter_slider"]["pre__load"]["max"]
        assert filter_slider.step == widget_parameters["filter_slider"]["pre__load"]["step"]
        assert filter_slider.disabled == widget_parameters["filter_slider"]["pre__load"]["disabled"]
        assert filter_slider.continuous_update == widget_parameters["filter_slider"]["pre__load"]["continuous_update"]
        assert filter_slider.orientation == widget_parameters["filter_slider"]["pre__load"]["orientation"]
        assert filter_slider.readout == widget_parameters["filter_slider"]["pre__load"]["readout"]
        assert filter_slider.readout_format == widget_parameters["filter_slider"]["pre__load"]["readout_format"]
        assert filter_slider.description == widget_parameters["filter_slider"]["pre__load"]["description"]
        assert filter_slider.description_tooltip == widget_parameters["filter_slider"]["pre__load"]["description_tooltip"]
       
        # Num Rows IntText
        data_limit = widgets.data_limit
        assert isinstance(data_limit, ipywidgets.IntText)
        assert data_limit.value == widget_parameters["data_limit"]["pre__load"]["value"]
        assert data_limit.disabled == widget_parameters["data_limit"]["pre__load"]["disabled"]
        assert data_limit.description == widget_parameters["data_limit"]["pre__load"]["description"]
        assert data_limit.description_tooltip == widget_parameters["data_limit"]["pre__load"]["description_tooltip"]

        # Show Info Checkbox
        show_info_checkbox = widgets.show_info_checkbox
        assert isinstance(show_info_checkbox, ipywidgets.Checkbox)
        assert show_info_checkbox.value == widget_parameters["show_info_checkbox"]["pre__load"]["value"]
        assert show_info_checkbox.disabled == widget_parameters["show_info_checkbox"]["pre__load"]["disabled"]
        assert show_info_checkbox.indent == widget_parameters["show_info_checkbox"]["pre__load"]["indent"]
        assert show_info_checkbox.description == widget_parameters["show_info_checkbox"]["pre__load"]["description"]
        assert show_info_checkbox.description_tooltip == widget_parameters["show_info_checkbox"]["pre__load"]["description_tooltip"]

        # Num Variables IntText
        num_variables = widgets.num_variables
        assert isinstance(num_variables, ipywidgets.IntText)
        assert num_variables.value == widget_parameters["num_variables"]["pre__load"]["value"]
        assert num_variables.disabled == widget_parameters["num_variables"]["pre__load"]["disabled"]
        assert num_variables.description == widget_parameters["num_variables"]["pre__load"]["description"]
        assert num_variables.description_tooltip == widget_parameters["num_variables"]["pre__load"]["description_tooltip"]

        # Start Button
        start_button = widgets.start_button
        assert isinstance(start_button, ipywidgets.Button)
        assert start_button.disabled == widget_parameters["start_button"]["pre__load"]["disabled"]
        assert start_button.button_style == widget_parameters["start_button"]["pre__load"]["button_style"]
        assert start_button.icon == widget_parameters["start_button"]["pre__load"]["icon"]
        assert start_button.description == widget_parameters["start_button"]["pre__load"]["description"]
        assert start_button.tooltip == widget_parameters["start_button"]["pre__load"]["tooltip"]

        # Progress Bar, FloatProgress
        progress_bar = widgets.progress_bar
        assert isinstance(progress_bar, ipywidgets.FloatProgress)
        assert progress_bar.value == widget_parameters["progress_bar"]["pre__load"]["value"]
        assert progress_bar.min == widget_parameters["progress_bar"]["pre__load"]["min"]
        assert progress_bar.max == widget_parameters["progress_bar"]["pre__load"]["max"]
        assert progress_bar.bar_style == widget_parameters["progress_bar"]["pre__load"]["bar_style"]
        assert progress_bar.orientation == widget_parameters["progress_bar"]["pre__load"]["orientation"]
        assert progress_bar.description == widget_parameters["progress_bar"]["pre__load"]["description"]
        assert progress_bar.description_tooltip == widget_parameters["progress_bar"]["pre__load"]["description_tooltip"]
        
        # Progress Label, HTML
        progress_label = widgets.progress_label
        assert isinstance(progress_label, ipywidgets.HTML)
        assert progress_label.value == widget_parameters["progress_label"]["pre__load"]["value"]
        
        # Output
        output = widgets.output
        assert isinstance(output, ipywidgets.Output)

        # Mispredicted Gridbox, HTML
        mispredicted_gridbox = widgets.mispredicted_gridbox
        assert isinstance(mispredicted_gridbox, ipywidgets.HTML)
        assert mispredicted_gridbox.value == widget_parameters["mispredicted_gridbox"]["pre__load"]["value"]

    def test_properties(self, widgets, widget_parameters) -> None:
        """ Tests properties that are not widgets """

        # Data Settings, dict
        data_settings = widgets.data_settings
        expected_data_settings = {
            'project': 'default',
            'data': {
                'catalog': 'N/A',
                'table': 'N/A'
            },
            'model': 'N/A',
            'columns': {
                'class': 'N/A',
                'id': 'N/A',
                'data_text': [],
                'data_numerical': []
            }
        }
        
        assert data_settings == expected_data_settings

        # data_numerical_columns(self) -> list:
        # data_text_columns(self) -> list:
        
        # Progress, tupe
        progress = widgets.progress
        assert isinstance(progress, tuple)
        assert isinstance(progress[0], ipywidgets.FloatProgress)
        assert isinstance(progress[1], ipywidgets.HTML)
       
        