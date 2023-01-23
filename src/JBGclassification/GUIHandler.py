# Alternate GUI for Autoclassification script using widgets from Jupyter
# Written by: Robert Granat, Jan-Feb 2022.
# Broken into module by: Marie Hogebrandt June-oct 2022

import errno
import json
import os
import sys
from pathlib import Path

from ipywidgets import Output

src_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(os.getcwd(), ".env")

sys.path.append(src_dir)

from dotenv import load_dotenv

from JBGExceptions import DataLayerException
from JBGLogger import JBGLogger
from AutomaticClassifier import AutomaticClassifier as autoclass
from Config import Config
from SQLDataLayer import DataLayer
from GUI.Widgets import Widgets

# Class definition for the GUI
class GUIHandler:
    # Constructor
    def __init__(self):
        settings = None
        settings_file = os.path.join(os.getcwd(), "settings.json")
        if os.path.isfile(settings_file):
            with open(settings_file) as f:
                settings = json.load(f)

        self.widgets = Widgets(src_path=Path(src_dir), GUIhandler=self, settings=settings)
        
        # This datalayer object is the one working with the classifier data
        self.classifier_datalayer = None

        # This datalayer object only works with the GUI
        self.gui_datalayer = None 
        
        self.logger = JBGLogger(False, self.widgets.progress) # Quiet is set to false here
        
        config = Config(
            connection=Config.Connection(
            odbc_driver=os.environ.get("DEFAULT_ODBC_DRIVER"),
            host=os.environ.get("DEFAULT_HOST"),
            class_catalog=os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
            class_table=os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
            data_catalog=os.environ.get("DEFAULT_DATA_CATALOG"),
            data_table=os.environ.get("DEFAULT_DATA_TABLE")
            )
        )
        self.gui_datalayer = DataLayer(config=config, logger=self.logger)
        
        if not self.gui_datalayer.can_connect(verbose=True):
            sys.exit("GUI Handler could not connect to Server")
        
        self.widgets.load_contents()
        
    @property
    def datalayer(self) -> DataLayer:
        """ This returns the GUI datalayer """
        return self.gui_datalayer

    def set_data_catalog(self, data_catalog: str) -> None:
        """ Updates the GUI datalayer with the data catalog """
        self.gui_datalayer.config.update_connection_columns({
            "data_catalog": data_catalog
            
        })

    def get_class_distribution(self, data_settings: dict, current_class: str) -> dict:
        """ Widgets does not need to know about Classifier Datalayer """
        datalayer = self.get_classifier_datalayer(data_settings=data_settings)
        try:
            distribution = datalayer.count_class_distribution()
        except DataLayerException as e:
            self.logger.abort_cleanly(str(e))
        except Exception as e:
            message = f"Could not update summary for class: {current_class} because {e}"
            self.logger.anaconda_debug(message)

        return distribution

    def correct_mispredicted_data(self, new_class: str, index: int) -> None:
        """ Changes the original dataset """
        self.classifier_datalayer.correct_mispredicted_data(new_class, index)

    def get_classifier_datalayer(self, data_settings: dict = None, config_params: dict = None) -> DataLayer:
        """ This will create or update the layer, and should probably also include the start_classifier() stuff
        """
        if config_params:
            # This creates a new classifier_datalayer whether one exists or not, because it has all the info
            self.classifier_datalayer = DataLayer(Config(**config_params), self.logger)

            return self.classifier_datalayer

        if data_settings:
            if not self.classifier_datalayer:
                connection = Config.Connection(
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
                    )

                self.classifier_datalayer = DataLayer(Config(connection), self.logger)
            else:
                updated_columns = {
                    "class_column": data_settings["columns"]["class"], 
                    "data_text_columns":  data_settings["columns"]["data_text"], 
                    "data_numerical_columns": data_settings["columns"]["data_numerical"], 
                    "id_column": data_settings["columns"]["id"]
                }
                self.classifier_datalayer.config.update_connection_columns(updated_columns)

        return self.classifier_datalayer        
        
    def run_classifier(self, config_params: dict, output: Output) -> None:
        """ Sets up the classifier and then runs it"""
        
        self.get_classifier_datalayer(config_params = config_params)

        self.logger.set_enable_quiet(not config_params["io"].verbose)
        the_classifier = autoclass(config=self.classifier_datalayer.get_config(), logger=self.logger, datalayer=self.classifier_datalayer)
        
        with output:
            result = the_classifier.run()
            if not result:
                self.logger.print_info("No data was fetched from database!")
            
        
        if result["mispredicted"] is not None:
            self.widgets.handle_mispredicted(**result)    
        
        self.widgets.set_rerun()
        

    def display_gui(self) -> None:
        self.widgets.display_gui()
        
        

def main():
    load_dotenv()
    gui = GUIHandler()
    gui.display_gui()


if __name__ == "__main__":
    main()
else:
    if os.path.isfile(env_path):
        load_dotenv(env_path)  # take environment variables from .env
    else:
        raise FileNotFoundError(
            errno.ENOENT, 
            os.strerror(errno.ENOENT), 
            env_path
        )

