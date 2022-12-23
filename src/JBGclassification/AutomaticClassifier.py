#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# General code for automatic classification of texts and numbers in databases
#
# Implemented by Robert Granat, JBG, March - May 2021
# Updated by Robert Granat, August 2021 - May 2022.
# Updated by Marie Hogebrandt, May 2022-October 2022
#
# Major revisions:
# 
# * Jan 26 - Febr 27, 2022: Rewritten the code as a Python Class
# * May-October, 2022: Breaking the script class into smaller, more manageable parts
# * November-?, 2022-?: Version 3 with new namespace and optimisations
# Standard imports
import os
import sys

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from scipy.sparse import SparseEfficiencyWarning

from JBGExceptions import HandlerException

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    
# General imports
import time
from datetime import datetime, timedelta

import pandas
from pandas import DataFrame, concat

# Imports of local help class for communication with SQL Server
import Config
import SQLDataLayer
import JBGLogger
from JBGHandler import JBGHandler
import Helpers

import warnings
# Sklearn issue a lot of warnings sometimes, we suppress them here
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

class AutomaticClassifier:

    # Internal constants
    LOWER_LIMIT_REDUCTION = 100
    NON_LINEAR_REDUCTION_COMPONENTS = 2
    MAX_HEAD_COLUMNS = 10
    
    # Constructor with arguments
    def __init__(self, config: Config.Config, logger: JBGLogger.JBGLogger, datalayer: SQLDataLayer.DataLayer):
        self.config = config

        self.logger = logger

        self.progression = {
            "majorTasks": 12,
            "progress": 0.0,
            "percentPerMajorTask": 0.03
        }

        self.logger.print_progress(message="Starting up ...", percent=self.progression["progress"])
        
         # Init some internal variables
        self.scriptpath = os.path.dirname(os.path.realpath(__file__))
        self.scriptname = sys.argv[0][2:]

        self.datalayer = datalayer

        # Internal settings for panda
        # TODO: This gives an error that it's not specific enough under 1.4.3, earlier 1.3.4 works
        # pandas.set_option("max_columns", self.MAX_HEAD_COLUMNS) 
        # Guessed display.max_columns, but could also be styler.render.max_columns
        pandas.set_option("display.max_columns", self.MAX_HEAD_COLUMNS)
        pandas.set_option("display.width", 80)

        # Get some timestamps
        self.date_now = datetime.now()
        self.clock1 = time.time()
        
    # Destructor
    def __del__(self):
        pass
    
    # Print the class 
    def __str__(self):
        return str(type(self))
    
    # Functions below

    # Function to simplify the paths/names of output files
    def get_output_filename(self, type: str) -> str:
        types = {
            "misplaced": {
                "suffix": "csv",
                "prefix": "misplaced_"
            }
        }

        type_dict = types[type]

        output_path = self.scriptpath + "\\output\\"
        output_name = self.config.name + "_" + self.config.connection.data_username

        return output_path + type_dict["prefix"] + output_name + "." + type_dict["suffix"]

    # Updates the progress and notifies the logger
    def update_progress(self, percent: float, message: str = None) -> float:
        self.progression["progress"] += percent

        if message is None:
            self.logger.print_progress(percent = self.progression["progress"])
        else:
            self.logger.print_progress(message=message, percent = self.progression["progress"])

        return self.progression["progress"]

    def create_handlers(self):
        """ Creates the handler and returns a tuple with handlers """

        self.handler = JBGHandler(self.datalayer, self.config, self.logger, self.progression)
        dh = self.handler.add_handler("dataset")
        mh = self.handler.add_handler("model")
        ph = self.handler.add_handler("predictions")

        return dh, mh, ph

    # The main classification function for the class
    def run(self):
        if not self.config.should_train():
            self.config = Config.Config.load_config_from_model_file(self.config.get_model_filename(), config=self.config)
        
        dh, mh, ph = self.create_handlers()
        
        # Print a welcoming message for the audience
        self.logger.print_welcoming_message(config=self.config, date_now=self.date_now)

        # Print out what mode we use: training a new model or not
        if self.config.should_train():
            self.logger.print_formatted_info("We will train our ml model")
            mh.load_model()
        elif self.config.should_predict():
            model_path = self.config.get_model_filename()
            if os.path.exists(model_path):
                self.logger.print_formatted_info("We will reload and use old model")
                mh.load_model(model_path)
            else:
                self.logger.abort_cleanly(f"No trained model exists at {model_path}")
        else:
            self.logger.abort_cleanly("User must choose either to train a new model or use an old one for predictions")
        
        self.update_progress(self.progression["percentPerMajorTask"])

        # Do some things prior to running the classification itself
        self.pre_run()

        # Major task: Read in data, validate and process it
        # Perhaps TODO: move validate_dataset out of read_in_data and make it another major task
        try:
            if not dh.read_in_data(): #should return true or false
                self.logger.print_progress(message="Process finished", percent=1.0)
                return -1 # Kanske inte behövs?
        except Exception as e:
            self.logger.abort_cleanly(f"Load of dataset failed: {e}")
        
        self.update_progress(self.progression["percentPerMajorTask"])

        # Separate data with know classifications from data with unknown class
        dh.separate_dataset()
        self.update_progress(self.progression["percentPerMajorTask"])
        
        # Split data in training and test sets
        dh.split_dataset_for_training_and_validation()

        # Major task: Text and categorical variables must be converted to numbers at this point
        self.logger.print_progress(message="Convert dataset to numbers only")
        if dh.handler.config.get_text_column_names():
            mh.model.update_field(field="text_converter", value=dh.convert_text_and_categorical_features(mh.model))

        # TODO: remove one progress-bar-updates corresponding to these lines
        self.update_progress(self.progression["percentPerMajorTask"])

        # TODO: Since we moved feature reduction into the pipeline, is this really necessary???
        try:
            self.config.set_num_selected_features(mh.model.get_num_selected_features(dh.X))
        except Exception as e:
            self.logger.abort_cleanly(f"Transform error: {e}")
        
        self.update_progress(self.progression["percentPerMajorTask"])

        # Major task: Check algorithms for best model and train that model. K-value should be 10 or below.
        # Or just use the model previously trained.
        # NOTICE: This major task uses another progressbar share inside DatasetHandler.spot_check_machine_learning_models,
        # so the number of progress bar shares will be the total number of "Major task":s + 1.
        if self.config.should_train():
            try:
                self.logger.print_progress(message="Check and train algorithms for best model")
                mh.train_model(dh)
                mh.save_model_to_file(mh.handler.config.get_model_filename())
            except Exception as ex:
                self.logger.abort_cleanly(f"Training or saving model failed: {ex}")

        self.update_progress(percent=self.progression["percentPerMajorTask"], message=f"Best model is: ({mh.model.get_name()}) with number of features: {self.config.get_num_selected_features()}")
        
        # Major task: Evalutate trained model on know testdata
        if self.config.should_train():
           
            if dh.X_validation.shape[0] > 0:
                
                # Make predictions on known testdata and report the results
                self.logger.print_progress(message="Make predictions on known testdata")
                
                ph.make_predictions(mh.model.pipeline, dh.X_validation, dh.classes, dh.Y_validation)
                
                ph.report_results(dh.Y_validation, mh.model)
                
            self.update_progress(percent=self.progression["percentPerMajorTask"])

        # Major task: Now RETRAIN the best model on whole dataset with known classification
        if self.config.should_train():
            self.logger.print_progress(message="Retrain model on whole dataset")
            
            cross_trained_model = mh.load_pipeline_from_file(self.config.get_model_filename())
            trained_model = mh.train_picked_model( mh.model.pipeline, dh.X, dh.Y)
                
            mh.save_model_to_file(self.config.get_model_filename())
                
        # Major task: Compute and display mot mispredicted data samples for possible manual correction
        if self.config.should_display_mispredicted():    
            ph.most_mispredicted(dh.X_original, trained_model, cross_trained_model, dh.X, dh.Y)

            ph.evaluate_mispredictions(self.get_output_filename("misplaced"))
            
            self.update_progress(percent=self.progression["percentPerMajorTask"])

        # Major task: Now make predictions on non-classified dataset: X_unknown -> Y_unknown
        if self.config.should_predict() and dh.X_prediction.shape[0] > 0:
            ph.make_predictions(mh.model.pipeline, dh.X_prediction, dh.classes)
            
            self.logger.print_formatted_info("Predictions for the unknown data")
            self.logger.print_info("Predictions:", str(Helpers.count_value_distr_as_dict(ph.predictions.tolist())))
            
            self.logger.print_training_rates(ph)
            
            self.update_progress(percent=self.progression["percentPerMajorTask"])

            try:
                saved_results = self.handler.save_predictions()
                if (saved_results["error"] is not None):
                    self.logger.print_error(f"Saving predictions failed: {saved_results['error']}")

                else:
                    results = saved_results["results"]
                    line =  f"Added {results['row_count']} rows to prediction table. Get them with SQL query:\n\n{results['query']}"
                    self.logger.print_info(line)
            except HandlerException as e:
                self.logger.abort_cleanly(f"Save of predictions failed: {e}")

        self.update_progress(percent=self.progression["percentPerMajorTask"])

        return self.post_run()
        

    def pre_run(self) -> None:
        """ Empty for now """
        # This used to have a function to set a flag in the database, which is no longer used.
        # However, pre_run is a good place to put things that needs to be run in the first stages
        # of the classifier

    def post_run(self) -> int:
        elapsed_time = time.time() - self.clock1
        date_again = datetime.now()
        message = f"Ending program after {timedelta(seconds=round(elapsed_time))} at {date_again:%Y-%m-%d %H:%M}"
        self.logger.print_progress(message, 1.0)
        # Return positive signal
        return 0

    def no_mispredicted_elements(self) -> bool:
        ph = self.handler.get_handler("predictions")
        
        return ph.num_mispredicted == 0

    def get_mispredicted_dataframe(self) -> DataFrame:
        ph = self.handler.get_handler("predictions")

        return ph.get_mispredicted_dataframe()

    def get_unique_classes(self) -> list[str]:
        dh = self.handler.get_handler("dataset")

        return dh.classes


# Main program
def main(argv):

    if len(argv) > 1:
        config = Config.Config.load_config_from_module(argv)
    else:
        config = Config.Config()
        #pwd = os.path.dirname(os.path.realpath(__file__))
        #
        #model_path = Path(pwd) / "./model/"
        #
        #filename =  model_path / "hpl_förutsättningar_2020.sav"
        #config = Config.Config.load_config_from_model_file(filename)
        #config.io.model_name = "hpl_förutsättningar_2020"

    logger = JBGLogger.JBGLogger(quiet=not config.io.verbose, in_terminal=True)
    
    datalayer = SQLDataLayer.DataLayer(config=config, logger=logger)
    # Use the loaded configuration module argument
    # or create a classifier object with only standard settings
    myClassiphyer = AutomaticClassifier(config=config, logger=logger, datalayer=datalayer)

    # Run the classifier
    myClassiphyer.run()

# Start main
if __name__ == "__main__":
    sys.exit(main(sys.argv))
