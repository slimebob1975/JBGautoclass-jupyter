#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# General code for automatic classification of texts and numbers in databases
#
# Implemented by Robert Granat, IAF, March - May 2021
# Updated by Robert Granat, August 2021 - May 2022.
# Updated by Marie Hogebrandt, May 2022-October 2022
#
# Major revisions:
# 
# * Jan 26 - Febr 27, 2022: Rewritten the code as a Python Class
# * May-October, 2022: Breaking the script class into smaller, more manageable parts
#
# Standard imports
import os
import sys

from IAFExceptions import HandlerException

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    
# General imports
import time
import warnings
from datetime import datetime, timedelta
#import ipywidgets as widgets

import pandas
from pandas import DataFrame, concat
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# Imports of local help class for communication with SQL Server
import Config
import SQLDataLayer
import IAFLogger
from IAFHandler import IAFHandler

# Sklearn issue a lot of warnings sometimes, we suppress them here
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

class IAFautomaticClassiphyer:

    # Internal constants
    LOWER_LIMIT_REDUCTION = 100
    NON_LINEAR_REDUCTION_COMPONENTS = 2
    MAX_HEAD_COLUMNS = 10
    
    # Constructor with arguments
    def __init__(self, config: Config.Config, logger: IAFLogger.IAFLogger, datalayer: SQLDataLayer.DataLayer):
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

    # The main classification function for the class
    def run(self):
        if not self.config.should_train():
            self.config = Config.Config.load_config_from_model_file(self.config.get_model_filename(), config=self.config)
        
        # TODO: This should be broken out into it's own function, creating IAFHandler and the necessary handlers
        self.handler = IAFHandler(self.datalayer, self.config, self.logger, self.progression)
        dh = self.handler.add_handler("dataset")
        mh = self.handler.add_handler("model")
        ph = self.handler.add_handler("predictions")
        
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
        
        # Create the classification table, if it does not exist already
        self.datalayer.prepare_for_classification()
        self.update_progress(self.progression["percentPerMajorTask"])

        self.pre_run()

        try:
            if not dh.read_in_data(): #should return true or false
                self.logger.print_progress(message="Process finished", percent=1.0)
                return -1 # Kanske inte behövs?
        except Exception as e:
            self.logger.abort_cleanly(f"Load of dataset failed: {e}")
        
        self.update_progress(self.progression["percentPerMajorTask"])

        dh.set_training_data()
        
        # Rearrange dataset such that all text columns are merged into one single
        # column, and convert text to numbers. Return the data in left-hand and
        # right hand side parts
        self.logger.print_progress(message="Rearrange dataset for possible textclassification, etc.")
        mh.model.update_fields(fields=["label_binarizers", "count_vectorizer", "tfid_transformer"], update_function=dh.convert_textdata_to_numbers)

        self.update_progress(self.progression["percentPerMajorTask"])

        mh.model.update_field("transform", dh.perform_feature_selection(mh.model))
        self.update_progress(self.progression["percentPerMajorTask"])

        # TODO: This was originally in perform_feature_selection, but I didn't want it as a sideeffect
        # Still need to decide where this belongs, however
        if mh.model.transform is None:
            num_selected_features = dh.X.shape[1]
        else:
            if hasattr(mh.model.transform, "n_components_"):
                num_selected_features = mh.model.transform.n_components_
            elif hasattr(mh.model.transform, "n_components"):
                num_selected_features = mh.model.transform.n_components
            else:
                self.logger.abort_cleanly(f"Transform {type(mh.model.transform)} have neither 'n_components_' 'or n_components'. Please check")

        self.config.set_num_selected_features(num_selected_features)
        
        # Split dataset for machine learning
        # TODO: Byt namn?
        dh.split_dataset()
        
        self.update_progress(self.progression["percentPerMajorTask"])

         # This is where the Model starts
        # Check algorithms for best model and train that model. K-value should be 10 or below.
        # Or just use the model previously trained.
        if self.config.should_train():
            try:
                mh.train_model(dh.X_train, dh.Y_train)
            except Exception as e:
                self.logger.abort_cleanly(f"Training model failed: {e}")

        self.update_progress(percent=self.progression["percentPerMajorTask"], message=f"Best model is: ({mh.model.get_name()}) with number of features: {self.config.get_num_selected_features()}")
        
        if self.config.should_train():
            # shape[0] = number of rows
            # shape[1] = number of columns
            if dh.X_validation.shape[0] > 0:
                # Make predictions on known testdata
                self.logger.print_progress(message="Make predictions on known testdata")
                
                ph.make_predictions(mh.model.model, dh.X_validation, dh.classes)

                self.logger.print_training_rates(ph)
                
                # Evaluate predictions (optional)
                ph.evaluate_predictions(dh.Y_validation, "ML algorithm: " + mh.model.get_name())

                # Get accumulated classification score report for all predictions
                self.logger.print_classification_report(*ph.get_classification_report(dh.Y_validation, mh.model))
                

            self.update_progress(percent=self.progression["percentPerMajorTask"])


            # RETRAIN best model on whole dataset: Xtrain + Xvalidation
            # Y and X here are both derived from the dataset without unknowns/non-classified elements
            if (dh.X_train.shape[0] + dh.X_validation.shape[0]) > 0:
                self.logger.print_progress(message="Retrain model on whole dataset")

                cross_trained_model = mh.load_pipeline_from_file(self.config.get_model_filename()) # Returns a 
                dh.X_transformed =  concat([pandas.DataFrame(dh.X_train), pandas.DataFrame(dh.X_validation)], axis = 0)
                # TODO: Maybe create this one up in the split_dataset, so we save Y_known, since neither Y_train nor Y_validation changes after calculation
                Y_known = concat([dh.Y_train, dh.Y_validation], axis = 0)

                trained_model = mh.train_picked_model( mh.model.model, dh.X_transformed, Y_known)
                
                mh.save_model_to_file(self.config.get_model_filename())
                
                ph.most_mispredicted(dh.X_original, trained_model, cross_trained_model, dh.X_transformed, Y_known)

                ph.evaluate_mispredictions(self.get_output_filename("misplaced"))
            
            self.update_progress(percent=self.progression["percentPerMajorTask"])

        # Now make predictions on non-classified dataset: X_unknown -> Y_unknown
        if self.config.should_predict() and dh.X_unknown.shape[0] > 0:
            ph.make_predictions(mh.model.model, dh.X_unknown)
            
            self.logger.print_formatted_info("Predictions for the unknown data")
            self.logger.print_info("Predictions:", str(ph.predictions))
            
            self.logger.print_training_rates(ph)
            
            # TODO: move this into make_predictions, since this is backup if predictions
            # couldn't be made (IE predict_proba == False)
            ph.calculate_probability()
           
            self.update_progress(percent=self.progression["percentPerMajorTask"])

            try:
                self.handler.save_classification_data()
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
        date_again = str(datetime.now())
        self.logger.print_formatted_info(f"Ending program after {timedelta(seconds=round(elapsed_time))} at {date_again}")

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

    logger = IAFLogger.IAFLogger(not config.io.verbose)
    
    datalayer = SQLDataLayer.DataLayer(config=config, logger=logger)
    # Use the loaded configuration module argument
    # or create a classifier object with only standard settings
    myClassiphyer = IAFautomaticClassiphyer(config=config, logger=logger, datalayer=datalayer)

    # Run the classifier
    myClassiphyer.run()

# Start main
if __name__ == "__main__":
    main(sys.argv)
