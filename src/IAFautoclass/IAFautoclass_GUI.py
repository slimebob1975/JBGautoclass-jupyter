# Alternate GUI for IAFautoclassification script using widgets from Jupyter
# Written by: Robert Granat, Jan-Feb 2022.
# Broken into module by: Marie Hogebrandt June 2022

import errno
import os
import sys
from pathlib import Path

# Imports
import ipywidgets as widgets
from sklearn.utils import Bunch

base_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(os.getcwd(), '.env')
sys.path.append(base_dir)

from dotenv import load_dotenv

import Helpers
from IAFExceptions import DataLayerException
from IAFLogger import IAFLogger
import IAFautomaticClassifier as autoclass
from Config import Algorithm, Config, Preprocess, Reduction, Scoretype
from DataLayer import DataLayer

if os.path.isfile(env_path):
    load_dotenv(env_path)  # take environment variables from .env
else:
    raise FileNotFoundError(
        errno.ENOENT, 
        os.strerror(errno.ENOENT), 
        env_path
    )

# Class definition for the GUI
class IAFautoclass_GUI:
    
    # Constants
    DEFAULT_HOST = os.environ.get("DEFAULT_HOST")
    DEFAULT_ODBC_DRIVER = os.environ.get("DEFAULT_ODBC_DRIVER")
    DEFAULT_DATA_CATALOG = os.environ.get("DEFAULT_DATA_CATALOG")
    DEFAULT_DATA_TABLE = os.environ.get("DEFAULT_DATA_TABLE")
    DEFAULT_CLASSIFICATION_CATALOG = os.environ.get("DEFAULT_CLASSIFICATION_CATALOG")
    DEFAULT_CLASSIFICATION_TABLE = os.environ.get("DEFAULT_CLASSIFICATION_TABLE")

    DEFAULT_DATA_COLUMNS = "petal-width,sepal-length,petal-length,sepal-width"
    DEFAULT_CLASS_COLUMN = "class"
    DEFAULT_TRAIN_OPTION = "Train a new model!"
    TEXT_WIDGET_LIMIT = 30
    TEXT_AREA_LIMIT = 60
    NULL = "NULL"
    IMAGE_FILE = base_dir + "/images/iaf-logo.png"
    TEXT_DATATYPES = ["nvarchar", "varchar", "char", "text", "enum", "set"]


    # Constructor
    def __init__(self):    
        self.config = None
        # The classifier object is a data element in our GUI
        self.the_classifier = None
        
        # We need a dictionary to keep track of datatypes
        self.datatype_dict = None
        
        # TODO: Can we fix this in a neater way?
        # Some data elements might get lost unless we lock a few callback functions
        self.lock_observe_1 = False
        
        # Keep track of if this is a rerun or not
        self.rerun = False
        
        # All the widgets moved into this function to be able to be hidden easier
        self.setup_GUI()
        self.logger = IAFLogger(False, (self.progress_bar, self.progress_label))
        connection = Config.Connection(
            odbc_driver=os.environ.get("DEFAULT_ODBC_DRIVER"),
            host=os.environ.get("DEFAULT_HOST"),
            class_catalog=os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
            class_table=os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
            data_catalog=os.environ.get("DEFAULT_DATA_CATALOG"),
            data_table=os.environ.get("DEFAULT_DATA_TABLE")
            )
        self.data_layer = DataLayer(connection, self.logger)
        try:
            self.data_layer.get_sql_connection() # This checks so that the SQL connection works
        except Exception as ex:
            sys.exit("GUI class could not connect to SQL Server: {0}".format(str(ex)))
        
        # Update databases list
        self.update_databases()
        
        self.update_algorithm_form()
                
    # Destructor
    def __del__(self):
        pass
    
    # Print the class 
    def __str__(self):
        return str(type(self))

    def setup_GUI(self) -> None:
        # We put a logo on top
        image_file = open(self.IMAGE_FILE, "rb")
        self.logo = widgets.Image(
                value = image_file.read(),
                format = 'png',
        )
        
        # Welcoming message label
        self.welcome = widgets.Label(
            value = "*** Welcome to IAF automatic classification! ***"
        )
        
        # Project element data element
        self.project = widgets.Text(
            value = 'default',
            placeholder = 'Project name',
            description = 'Project name:',
            disabled = False,
            description_tooltip = 'Enter a distinct project name for the model'
        )
        
        # Database dropdown list
        self.database_dropdown = widgets.Dropdown(
            options = ['N/A'],
            value = 'N/A',
            description = 'Databases:',
            description_tooltip = 'These are the databases of choice'
        )
        
        # Tables dropdown list
        self.tables_dropdown = widgets.Dropdown(
            options = ['N/A'],
            value = 'N/A',
            description = 'Tables:',
            disabled = True,
            description_tooltip = 'These are the database tables of choice'
        )
        
        # Models dropdown list
        self.models_dropdown = widgets.Dropdown(
            options = ['N/A'],
            value = 'N/A',
            description = 'Models:',
            disabled = True,
            description_tooltip = 'You can train a new model or use a previously trained one'
        )
        
        # Radiobuttons for choosing the class column
        self.class_column = widgets.RadioButtons(
            options = ['N/A'],
            value = 'N/A',
            description = 'Class:',
            disabled = True,
            description_tooltip = 'Pick the column to use as class labels'
        )
    
        # Radiobuttons for choosing the unique id column
        self.id_column = widgets.RadioButtons(
            options = ['N/A'],
            value = 'N/A',
            description = 'Unique id:',
            disabled = True,
            description_tooltip = 'Pick the column to use as unique identifier'
        )

        # Multiple select for choosing the data columns
        self.data_columns = widgets.SelectMultiple(
            options = ['N/A'],
            value = [],
            description = 'Pick data:',
            disabled = True,
            description_tooltip = 'Pick the columns with the data to be used in the classification'
        )
        
        # Multiple select for picking text columns of the available data columns
        self.text_columns = widgets.SelectMultiple(
            options = ['N/A'],
            value = [],
            description='Is text:',
            disabled = True,
            description_tooltip = 'Mark the data columns to be interpreted as text (some are marked by default from their SQL datatype)'
        )
        
        # Put together the data widgets in a horizontal box widget
        self.data_form = widgets.Box( 
            [self.class_column, self.id_column, self.data_columns, self.text_columns], 
            layout = widgets.Layout(
                display='flex',
                flex_flow='row',
                border='solid 0px',
                align_items='stretch',
                width='auto',
            ),
        )
        
         # Continuation message label
        self.class_summary_text = widgets.Label(
            description = "",
            disabled = False,
        )
        
        # Button to continue after the basic settings was set
        self.continuation_button = widgets.Button(
            description='Continue',
            disabled=True,
            button_style='success', 
            tooltip='Continue with the process using these settings',
            icon='check' 
        )
        
        # Checkboxes for different model modes
        self.train_checkbox = widgets.Checkbox(
            value = False,
            description = 'Mode: Train',
            disabled = True,
            indent = True,
            description_tooltip = 'A new model will be trained'
        )
        
        self.predict_checkbox = widgets.Checkbox(
            value = False,
            description = 'Mode: Predict',
            disabled = True,
            indent = True,
            description_tooltip = 'The model of choice will be used to make predictions'
        )
        
        self.mispredicted_checkbox = widgets.Checkbox(
            value = False,
            description = 'Mode: display mispredictions',
            disabled = True,
            indent = True,
            description_tooltip = 'The classifier will display mispredicted training data for manual inspection and correction'
        )
                
        self.checkboxes_form = widgets.HBox([self.train_checkbox, self.predict_checkbox, self.mispredicted_checkbox ])
        
        # Algorithms dropdown list
        self.algorithm_dropdown = widgets.Dropdown(
            options = ['N/A'],
            value = 'N/A',
            description = 'Algorithm:',
            disabled = True,
            description_tooltip = 'Pick what algorithms to use'
        )
        
        # Preprocessor dropdown list
        self.preprocessor_dropdown = widgets.Dropdown(
            options = ['N/A'],
            value = 'N/A',
            description = 'Preprocess:',
            disabled = True,
            description_tooltip = 'Pick what data preprocessor to use'
        )
        
        # Metric dropdown list
        self.metric_dropdown = widgets.Dropdown(
            options = ['N/A'],
            value = 'N/A',
            description = 'Metric:',
            disabled = True,
            description_tooltip = 'Pick by what method algorithms performances are measured and compared'
        )
        
         # Variable reduction dropdown list
        self.reduction_dropdown = widgets.Dropdown(
            options = ['N/A'],
            value = 'N/A',
            description = 'Reduction:',
            disabled = True,
            description_tooltip = 'Pick by what method the number of features (variables) are reduced'
        )
        
        # Put everything together in a horizontal box and populate it
        self.algorithm_form = widgets.Box( 
            [ self.reduction_dropdown, self.algorithm_dropdown, self.preprocessor_dropdown, self.metric_dropdown ], 
            layout = widgets.Layout(
                display='flex',
                flex_flow='row',
                border='solid 0px',
                align_items='stretch',
                width='auto',
            ),
        )
        
        # Two checkboxes related to SMOTE and undersampling of dominant class
        self.smote_checkbox = widgets.Checkbox(
            value = False,
            description = 'SMOTE',
            disabled = True,
            indent = True,
            description_tooltip = 'Use SMOTE (synthetic minority oversampling technique) for training data'
        )
        
        self.undersample_checkbox = widgets.Checkbox(
            value = False,
            description = 'Undersampling',
            disabled = True,
            indent = True,
            description_tooltip = 'Use undersampling of majority training data'
        )
        
        # Test datasize slider
        self.testdata_slider = widgets.IntSlider(
            value = 20,
            min = 0,
            max = 100,
            step = 1,
            description = 'Testdata (%):',
            disabled = True,
            continuous_update = False,
            orientation = 'horizontal',
            readout = True,
            readout_format = 'd',
            description_tooltip = 'Set how large evaluation size of training data will be'
        )
        
         # Iterations slider
        self.iterations_slider = widgets.IntSlider(
            value = 20000,
            min = 1000,
            max = 100000,
            step = 100,
            description = 'Max.iter:',
            disabled = True,
            continuous_update = False,
            orientation = 'horizontal',
            readout = True,
            readout_format = 'd',
            description_tooltip = 'Set how many iterations to use at most'
        )
            
        # Put everything together in a horizontal box
        self.data_handling_form = widgets.HBox([ self.smote_checkbox, self.undersample_checkbox, self.testdata_slider, self.iterations_slider ])
        
        # Four widgets regarding handling of text data
        self.encryption_checkbox = widgets.Checkbox(
            value = True,
            description = 'Text: Encryption',
            disabled = True,
            indent = True,
        )
        
        self.categorize_checkbox = widgets.Checkbox(
            value = True,
            description = 'Text: Categorize',
            disabled = True,
            indent = True,
        )
        
        # Multiple select for picking text columns to force categorization on (text columns with 30 distinct values are automatically categorized)
        self.categorize_columns = widgets.SelectMultiple(
            options = ['N/A'],
            value = [],
            description='Categorize:',
            disabled = True,
            description_tooltip = 'Mark text columns to force them to be categorized (text columns with up to 30 distinct values will be if checkbox is checked)'
        )
        
        self.filter_checkbox = widgets.Checkbox(
            value = False,
            description = 'Text: Filter',
            disabled = True,
            indent = True,
        )
        
        self.filter_slider = widgets.IntSlider(
            value = 100,
            min = 0,
            max = 100,
            step = 1,
            description = 'Doc.freq. (%):',
            disabled = True,
            continuous_update = False,
            orientation = 'horizontal',
            readout = True,
            readout_format = 'd',
            description_tooltip = 'Set document frequency limit for filtering of stop words'
        )
        
        # Put everything together in a horizontal box
        self.text_handling_form = widgets.HBox([ self.categorize_checkbox, self.categorize_columns, self.encryption_checkbox, self.filter_checkbox, self.filter_slider ])
        
        # Two debug widget, one to limit the number of considered data rows, one for verbosity
        self.num_rows = widgets.IntText(
            value = 0,
            description = 'Data limit:',
            disabled = True,
            description_tooltip = 'During debugging or testing, limiting the number of data rows can be beneficial'
        )
        
        self.verbose_checkbox = widgets.Checkbox(
            value = False,
            description = 'Verbose',
            disabled = True,
            indent = True,
        )
        
         # Two debug widget, one to limit the number of considered data rows, one for verbosity
        self.num_variables = widgets.IntText(
            value = 0,
            description = 'Variables:',
            disabled = True,
            description_tooltip = 'The number of variables used is shown in this box'
        )
        
        # Put together horisontally as before
        self.debug_form = widgets.HBox([self.num_rows, self.verbose_checkbox, self.num_variables])
        
        # Start Button to continue after all settings was set
        self.start_button = widgets.Button(
            description='Start',
            disabled=True,
            button_style='success', 
            tooltip='Run the classifier using the current settings',
            icon='check' 
        )
        
        # A progress bar that displays the computational process
        self.progress_bar = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=1.0,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#004B99'},
            orientation='horizontal',
            description_tooltip = 'The computational process is shown here'
        )
        
        # Progress message label
        self.progress_label = widgets.Label(
            value = "",
        )
        
        # We put the progress widgets together in a form
        self.progress_form = widgets.HBox([self.progress_bar, self.progress_label])
        
        # All output is directed to this place
        self.output = widgets.Output(layout={'border': '2px solid grey'})  
        
        # The gridbox widget for handling mispredicted data elements.
        # We first declare a Label Placeholder, and update it into a gridbox later.
        self.mispredicted_gridbox = widgets.Label(
            value = "No mispredicted training data was detected yet",
            description_tooltip = 'Use to manually inspect and correct mispredicted training data'
        )
    
    # Internal methods used to populate or update widget settings
    def update_databases(self) -> None:
        data = self.data_layer.get_databases()
        database_list = [''] + [database[0] for database in data]
        self.database_dropdown.options = database_list
        self.database_dropdown.value = database_list[0]
        
        self.database_dropdown.observe(handler=self.update_tables)
                
    def update_tables(self, event: Bunch) -> None:
        """ Handler: This updates the tables dropdown when the database dropdown changes. """
        if self.lock_observe_1:
            return
        
        tables = self.data_layer.get_tables()
        tables_list = [""] + [table[0]+'.'+table[1] for table in tables]
        self.tables_dropdown.options = tables_list
        self.tables_dropdown.value = tables_list[0]
        self.tables_dropdown.disabled = False

        self.tables_dropdown.observe(handler=self.update_models)
        
    def update_models(self, event: Bunch) -> None:
        """ Handler: This updates the models dropdown when the tables dropdown changes. """
        if self.lock_observe_1 or self.tables_dropdown.value == "":
            return
        
        models_list = [""] + [self.DEFAULT_TRAIN_OPTION] + self.data_layer.get_trained_models(Config.DEFAULT_MODELS_PATH, Config.DEFAULT_MODEL_EXTENSION)
        self.models_dropdown.options = models_list
        self.models_dropdown.value = models_list[0]
        self.models_dropdown.disabled = False

        self.models_dropdown.observe(handler=self.use_old_model_or_train_new) # TODO: Rework this to a "model_handler method"
    
    def use_old_model_or_train_new(self, event: Bunch) -> None:
        """ Handler: Sets the GUI to a new or existing model, as necessary. """
        self.update_class_id_data_columns()
        if self.models_dropdown.value == self.DEFAULT_TRAIN_OPTION:
            self.continuation_button.disabled = True
            return
        
        if not self.models_dropdown.value:
            return

        model_path = Path(Config.DEFAULT_MODELS_PATH) / self.models_dropdown.value
        tc_config = Config.load_config_from_model_file(model_path)
        
        try:
            self.class_column.value = tc_config.get_class_column_name()
        except Exception as ex:
            print("Could not set class column as: {0} because of exception: {1}". \
                    format(tc_config.get_class_column_name(), str(ex)))
        self.class_column.disabled = True
        try:
            self.id_column.value = tc_config.get_id_column_name()
        except Exception as ex:
            print("Could not set id column as: {0} because of exception: {1}". \
                    format(tc_config.get_id_column_name(), str(ex)))
        self.id_column.disabled = True 
        
        
        try:
            temp_numerical_columns = tc_config.get_numerical_column_names()
            temp_text_columns = tc_config.get_text_column_names()
        
            self.data_columns.value = temp_numerical_columns + temp_text_columns
        except Exception as ex:
            print("Could not set data columns as: {0} because of exception: {1}". \
                    format(temp_numerical_columns + temp_text_columns, str(ex)))
        self.data_columns.disabled = True
        try:
            self.text_columns.value = temp_text_columns
        except Exception as ex:
            print("Could not set text columns as: {0} because of exception: {1}". \
                    format(temp_text_columns, str(ex)))
        self.text_columns.disabled = True

        self.algorithm_dropdown.value = tc_config.get_algorithm().name
        self.preprocessor_dropdown.value = tc_config.get_preprocessor().name
        self.reduction_dropdown.value = tc_config.get_feature_selection().name
        self.num_variables.value = tc_config.get_num_selected_features()
        self.filter_checkbox.value = tc_config.use_stop_words()
        self.filter_slider.value = tc_config.get_stop_words_threshold_percentage()
        self.encryption_checkbox.value = tc_config.should_hex_encode()
        self.categorize_checkbox.value = tc_config.use_categorization()
        self.categorize_columns.options = self.text_columns.value
        
        try:
            self.categorize_columns.value =  tc_config.get_categorical_text_column_names()
        except Exception as ex:
            print("Could not set category columns as: {0} because of exception: {1}". \
                    format(tc_config.get_categorical_text_column_names(), str(ex)))
        self.categorize_columns.disabled = True
        
        self.testdata_slider.value = tc_config.get_test_size_percentage()
        self.smote_checkbox.value = False
        self.undersample_checkbox.value = False
        
        self.continuation_button.on_click(callback=self.continuation_button_was_clicked)
        self.continuation_button.disabled = False
        
    def update_class_id_data_columns(self) -> None:
        """ Updates the options for class, id and data_columns"""
        if self.lock_observe_1:
            return
        
        columns = self.data_layer.get_id_columns(self.database_dropdown.value, self.tables_dropdown.value)
        columns_list = [column[0] for column in columns]
        self.datatype_dict = {column[0]:column[1] for column in columns}

        if len(columns_list) >= 3:
            self.class_column.options = columns_list
            self.class_column.value = columns_list[0]
            self.class_column.disabled = False
            self.class_column.observe(handler=self.update_id_and_data_columns)

            self.id_column.options = columns_list[1:]
            self.id_column.value = columns_list[1]
            self.id_column.disabled = False
            self.id_column.observe(handler=self.update_data_columns)

            self.data_columns.options = columns_list[2:]
            self.data_columns.value = []
            self.data_columns.disabled = False
            self.data_columns.observe(handler=self.update_text_columns_and_enable_button)
        else:
            sys.exit("Selected data table has to few columns. Minimum is three (3).")
            
    def update_id_and_data_columns(self, event:Bunch) -> None:
        """ Handler: Changes the id and data columns when the class option is changed. """
        try:
            self.update_class_summary()
        except Exception as ex:
            print("Notice. Could not update summary for class: {0}, for reason: {1}". \
                format(self.class_column.value, str(ex)))
        self.id_column.options = \
            [option for option in self.class_column.options if option != self.class_column.value]
        self.update_data_columns(event)

    def update_data_columns(self, event:Bunch) -> None:
        """ Handler: Updates the data column option when id is changed. """
        if self.lock_observe_1: 
            return

        self.data_columns.options = \
            [option for option in self.id_column.options if option != self.id_column.value]
            
    def update_text_columns_and_enable_button(self, event:Bunch) -> None:
        """ Updates the text columns and continuation_button states when data options changes. """
        if self.lock_observe_1:
            return

        self.text_columns.options = self.data_columns.value
        self.text_columns.value = \
            [col for col in self.data_columns.value if self.datatype_dict[col] in self.TEXT_DATATYPES]
        self.text_columns.disabled = False

        self.num_variables.value = len(self.data_columns.value)
        if self.num_variables.value >= 1:
            self.continuation_button.disabled = False
            self.continuation_button.on_click(callback=self.continuation_button_was_clicked)
        else:
            self.continuation_button.disabled = True
                
    def update_class_summary(self):
        """ Updates the data_layer's connection values too """
        try:
            distrib = self.data_layer.count_class_distribution(
                class_column=self.class_column.value,
                data_catalog=self.database_dropdown.value,
                data_table=self.tables_dropdown.value)
        except DataLayerException as e:
            self.logger.abort_cleanly(str(e))
        
        self.class_summary_text.value = \
            "Class column: \'{0}\', with distribution: {2}, in total: {1} rows". \
            format(self.class_column.value, sum(distrib.values()), str(distrib)[1:-1])
        
    def continuation_button_was_clicked(self, event:Bunch) -> None:
        """ Callback: Sets various states based on the value in models dropdown. """
        self.lock_observe_1 = True
        if self.models_dropdown.value != self.DEFAULT_TRAIN_OPTION:
            self.train_checkbox.value = False
            self.train_checkbox.disabled = True
            self.predict_checkbox.value = True
            self.predict_checkbox.disabled = True
            self.mispredicted_checkbox.value = False
            self.mispredicted_checkbox.disabled = True
        else:
            self.train_checkbox.value = True
            self.train_checkbox.disabled = False
            self.predict_checkbox.value = False
            self.predict_checkbox.disabled = False
            self.mispredicted_checkbox.value = True
            self.mispredicted_checkbox.disabled = False
            self.algorithm_dropdown.disabled = False
            self.preprocessor_dropdown.disabled = False
            self.reduction_dropdown.disabled = False
            self.metric_dropdown.disabled = False 
            self.smote_checkbox.disabled = False
            self.undersample_checkbox.disabled = False
            self.testdata_slider.disabled = False
            self.iterations_slider.disabled = False
            self.encryption_checkbox.disabled = False
            self.categorize_checkbox.disabled = False
            self.categorize_columns.disabled = False
            self.categorize_columns.options = self.text_columns.value
            self.filter_checkbox.disabled = False
            self.filter_slider.disabled = False
        self.num_rows.disabled = False
        self.update_num_rows()
        self.verbose_checkbox.disabled = False
        self.project.disabled = True
        self.database_dropdown.disabled = True
        self.tables_dropdown.disabled = True
        self.models_dropdown.disabled = True
        self.class_column.disabled = True
        self.id_column.disabled = True
        self.data_columns.disabled = True
        self.text_columns.disabled = True
        self.continuation_button.disabled = True
        self.start_button.disabled = False
        self.start_button.on_click(callback = self.start_button_was_clicked)
        
    def update_algorithm_form(self):
        self.reduction_dropdown.options = Reduction.get_sorted_list()
        self.algorithm_dropdown.options = Algorithm.get_sorted_list()
        self.preprocessor_dropdown.options = Preprocess.get_sorted_list()
        self.metric_dropdown.options = Scoretype.get_sorted_list()
        
        
    def update_num_rows(self):
        try:
            self.num_rows.value = self.data_layer.count_data_rows(data_catalog=self.database_dropdown.value, data_table=self.tables_dropdown.value)
        except DataLayerException as e:
             self.logger.abort_cleanly(str(e))
        
        
    def start_button_was_clicked(self, event:Bunch) -> None:
        """ Callback: Changes the state to be read-only, starts the classifier (or the class summary) """
        self.train_checkbox.disabled = True
        self.predict_checkbox.disabled = True
        self.mispredicted_checkbox.disabled = True
        self.algorithm_dropdown.disabled = True
        self.preprocessor_dropdown.disabled = True
        self.reduction_dropdown.disabled = True
        self.metric_dropdown.disabled = True 
        self.smote_checkbox.disabled = True
        self.undersample_checkbox.disabled = True
        self.testdata_slider.disabled = True
        self.iterations_slider.disabled = True
        self.encryption_checkbox.disabled = True
        self.categorize_checkbox.disabled = True
        self.categorize_columns.disabled = True
        self.filter_checkbox.disabled = True
        self.filter_slider.disabled = True
        self.num_rows.disabled = True
        self.verbose_checkbox.disabled = True
        self.start_button.disabled = True
        
        if self.rerun:
            self.update_class_summary()
        else:
            self.rerun = True
        
        self.output.clear_output(wait=True)
        self.start_classifier()
    
    def start_classifier(self):
        
        # This isn't pretty and probably needs to be tweaked, but it works for now
        # Sometimes you want to load the config from a different place than the name of the project
        if self.models_dropdown.value == self.DEFAULT_TRAIN_OPTION:
            model_name = self.project.value
        else:
            model_name = self.models_dropdown.value.replace(Config.DEFAULT_MODEL_EXTENSION, "")
        
        
        data_numerical_columns = \
                [col for col in self.data_columns.value if not col in self.text_columns.value]
        connection = Config.Connection(
                odbc_driver = self.DEFAULT_ODBC_DRIVER,
                host = self.DEFAULT_HOST,
                trusted_connection = True,
                data_catalog = self.database_dropdown.value,
                data_table = self.tables_dropdown.value,
                class_column = self.class_column.value,
                class_catalog = self.DEFAULT_CLASSIFICATION_CATALOG,
                class_table = self.DEFAULT_CLASSIFICATION_TABLE,
                data_text_columns = ','.join(list(self.text_columns.value)),
                data_numerical_columns = ','.join(data_numerical_columns),
                id_column = self.id_column.value
            )
        config = Config(
            connection,
            Config.Mode(
                train = self.train_checkbox.value,
                predict = self.predict_checkbox.value,
                mispredicted = self.mispredicted_checkbox.value,
                use_stop_words = self.filter_checkbox.value,
                specific_stop_words_threshold = float(self.filter_slider.value) / 100.0,
                hex_encode = self.encryption_checkbox.value,
                use_categorization = self.categorize_checkbox.value,
                category_text_columns = ','.join(list(self.categorize_columns.value)),
                test_size = float(self.testdata_slider.value) / 100.0,
                smote = self.smote_checkbox.value,
                undersample = self.undersample_checkbox.value,
                algorithm = Algorithm[self.algorithm_dropdown.value],
                preprocessor = Preprocess[self.preprocessor_dropdown.value],
                feature_selection = Reduction[self.reduction_dropdown.value],
                num_selected_features = None,
                scoring = Scoretype[self.metric_dropdown.value],
                max_iterations = self.iterations_slider.value
            ),
            Config.IO(
                verbose=self.verbose_checkbox.value,
                model_name=model_name
            ),
            Config.Debug(
                on=True,
                num_rows=self.num_rows.value
            ),
            name=self.project.value,
            save=(self.models_dropdown.value == self.DEFAULT_TRAIN_OPTION),
        )
        
        self.data_layer.update_connection(connection)
        self.the_classifier = autoclass.IAFautomaticClassiphyer(config=config, logger=self.logger, datalayer=self.data_layer)
        
        with self.output:
            worked = self.the_classifier.run()
            if worked == -1:
                print("Notice: no data was fetched from database!")
            else:
                if self.mispredicted_checkbox.value:
                    self.update_mispredicted_gridbox()
                    display(self.mispredicted_gridbox)
        self.start_button.description = "Rerun"
        self.start_button.tooltip = "Rerun the classifier with the same setting as last time"
        self.start_button.disabled = False
        self.verbose_checkbox.disabled = False
    
    def update_mispredicted_gridbox(self): 
        if not self.the_classifier or self.the_classifier.no_mispredicted_elements():
            return   
        
        mispredicted = self.the_classifier.get_mispredicted_dataframe()
        print("\nNotice! There are mispredicted training data elements:")
        items = [widgets.Label(mispredicted.index.name)] + \
            [widgets.Label(item) for item in mispredicted.columns] + \
            [widgets.Label("Reclassify as")]
        cols = len(items)
        for i in mispredicted.index:
            row = mispredicted.loc[i]
            row_items = [widgets.Label(str(row.name))]
            for item in row.index: 
                elem = row[item]
                if Helpers.is_str(elem) and len(str(elem)) > self.TEXT_AREA_LIMIT:
                    row_items.append(widgets.Textarea(value=str(elem), Placeholder=self.NULL))
                elif Helpers.is_str(elem) and len(str(elem)) > self.TEXT_WIDGET_LIMIT:
                    row_items.append(widgets.Text(value=str(elem), Placeholder=self.NULL))
                else:
                    if Helpers.is_float(elem):
                        elem = round(float(elem), 2)
                    row_items.append(widgets.Label(str(elem)))
            dropdown_options = [('Keep', 0)]
            for label in self.the_classifier.get_unique_classes():
                dropdown_options.append((label, (label, row.name)))
            reclassify_dropdown = widgets.Dropdown(options = dropdown_options, value = 0, description = '', disabled = False)
            reclassify_dropdown.observe(self.changed_class,'value')
            row_items += [reclassify_dropdown]
            items += row_items 
        gridbox_layout = widgets.Layout(grid_template_columns="repeat("+ str(cols) +", auto)", border="4px solid grey")
        self.mispredicted_gridbox = widgets.GridBox(items, layout=gridbox_layout)
        
    
    def changed_class(self, change):
        if change.new != 0 and change.old != change.new:
            new_class, new_index = change.new
            self.data_layer.correct_mispredicted_data(new_class, new_index)

    def display_gui(self) -> None:
        elements = [
            "logo",
            "welcome",
            "project",
            "database_dropdown",
            "tables_dropdown",
            "models_dropdown",
            "data_form",
            "class_summary_text",
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

        for element in elements:
            display(getattr(self, element))
        

def main():
    gui = IAFautoclass_GUI()


if __name__ == "__main__":
    main()
