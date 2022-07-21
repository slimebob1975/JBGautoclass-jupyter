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

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    
# General imports
import time
import warnings
from datetime import datetime, timedelta
#import ipywidgets as widgets

import pandas
from pandas import concat
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# Imports of local help class for communication with SQL Server
import Config
import DataLayer
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
    def __init__(self, config: Config.Config, logger: IAFLogger.IAFLogger, datalayer: DataLayer.DataLayer):
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


    # TODO: used in GUI
    # Get a list of pretrained models
    def get_trained_models(self):
        models = []
        for file in os.listdir(self.model_path):
            if file[-len(self.DEFAULT_MODEL_FILE_EXTENSION):] == self.DEFAULT_MODEL_FILE_EXTENSION:
                models.append(file)
        return models

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
        
        # Print a welcoming message for the audience
        self.logger.print_welcoming_message(config=self.config, date_now=self.date_now)

        if not self.config.should_train():
            self.config = Config.Config.load_config_from_model_file(self.config.get_model_filename(), config=self.config)
        
        # TODO: This should be broken out into it's own function, creating IAFHandler and the necessary handlers
        handler = IAFHandler(self.datalayer, self.config, self.logger, self.progression)
        dh = handler.add_handler("dataset")
        mh = handler.add_handler("model")
        ph = handler.add_handler("predictions")
        

        # Print out what mode we use: training a new model or not
        # TODO: This is a good place to start splitting. 
        # Separate into two python files: one with train + "no if" and one with "no if" and not-train/predict
        # This turned out to be a bad idea at it's core, so starting with dividing things into dataset vs model instead
        if self.config.should_train():
            self.logger.print_formatted_info("We will train our ml model")
        elif self.config.should_predict():
            if os.path.exists(self.config.get_model_filename()):
                self.logger.print_formatted_info("We will reload and use old model")
            else:
                self.logger.abort_cleanly(f"No trained model exists at {self.config.get_model_filename()}")
        else:
            self.logger.abort_cleanly("User must choose either to train a new model or use an old one for predictions")
        
        # Create the classification table, if it does not exist already
        self.datalayer.create_classification_table()
        self.update_progress(self.progression["percentPerMajorTask"])

        try:
            # Set a flag in the classification database that execution has started
            self.datalayer.mark_execution_started()
        except Exception as e:
            self.logger.abort_cleanly(f"Mark of executionstart failed: {e}")


        try:
            if not dh.read_in_data(): #should return true or false
                self.logger.print_progress(message="Process finished", percent=1.0)
                return -1
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
            num_selected_features = mh.model.transform.n_components_

        self.config.set_num_selected_features(num_selected_features)

        
        # Split dataset for machine learning
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

        ml_algorithm = mh.model.get_name()
        self.update_progress(percent=self.progression["percentPerMajorTask"], message=f"Best model is: ({ml_algorithm}) with number of features: {self.config.get_num_selected_features()}")
        
        if self.config.should_train():
            # shape[0] = number of rows
            # shape[1] = number of columns
            if dh.X_validation.shape[0] > 0:
                # Make predictions on known testdata
                self.logger.print_progress(message="Make predictions on known testdata")
                
                could_proba = ph.make_predictions(mh.model.model, dh.X_validation)

                self.logger.print_training_probabilities(ph)
                
                # Evaluate predictions (optional)
                ph.evaluate_predictions(dh.Y_validation, "ML algorithm: " + ml_algorithm)

                # Get accumulated classification score report for all predictions
                self.logger.print_classification_report(*ph.get_classification_report(dh.Y_validation, mh.model))
                

            self.update_progress(percent=self.progression["percentPerMajorTask"])


            # RETRAIN best model on whole dataset: Xtrain + Xvalidation
            if (dh.X_train.shape[0] + dh.X_validation.shape[0]) > 0:
                self.logger.print_progress(message="Retrain model on whole dataset")

                cross_trained_model = mh.load_pipeline_from_file(self.config.get_model_filename())
                dh.X_transformed =  concat([pandas.DataFrame(dh.X_train), pandas.DataFrame(dh.X_validation)], axis = 0)
                Y = concat([dh.Y_train, dh.Y_validation], axis = 0)

                trained_model = mh.train_picked_model( mh.model.model, dh.X_transformed, dh.Y)
                
                mh.save_model_to_file(self.config.get_model_filename())
                
                # TODO: maybe replace Y with dh.Y
                ph.most_mispredicted(dh.X_original, trained_model, cross_trained_model, dh.X_transformed, Y)

                ph.evaluate_mispredictions(dh.queries["read_data"], self.get_output_filename("misplaced"))
            
            self.update_progress(percent=self.progression["percentPerMajorTask"])

        # Now make predictions on non-classified dataset: X_unknown -> Y_unknown
        if self.config.should_predict() and dh.X_unknown.shape[0] > 0:
            ph.make_predictions(mh.model.model, dh.X_unknown)
            
            self.logger.print_formatted_info("Predictions for the unknown data")
            self.logger.print_info("Predictions:", str(dh.Y_unknown))
            
            self.logger.print_training_probabilities(ph)
            
            ph.calculate_probability()
           
            self.update_progress(percent=self.progression["percentPerMajorTask"])

            # Save new classifications for X_unknown in classification database
            self.logger.print_progress(message="Save new classifications in database")
            try:
                results_saved = self.datalayer.save_data(
                    dh.unpredicted_keys.values,
                    dh.Y_unknown,
                    ph.probabilites,
                    ph.get_rate_type(),
                    mh.model.model.classes_,
                    ph.probabilites,
                    mh.model.get_name()
                )
            except Exception as e:
                self.logger.abort_cleanly(f"Save of predictions failed: {e}")
            
            saved_query = self.datalayer.get_sql_command_for_recently_classified_data(results_saved)
            self.logger.print_info(f"Added {results_saved} rows to classification table. Get them with SQL query:\n\n{saved_query}")

        self.update_progress(percent=self.progression["percentPerMajorTask"])

        elapsed_time = time.time() - self.clock1
        date_again = str(datetime.now())
        self.logger.print_formatted_info(f"Ending program after {timedelta(seconds=round(elapsed_time))} at {date_again}")

        try:
            # Remove flag in database, signaling all was alright
            self.datalayer.mark_execution_ended()
        except Exception as e:
            self.logger.abort_cleanly(f"Mark of execution-end failed: {e}")

        # Make sure progressbar is completed if not before
        self.logger.print_progress(message="Process finished", percent=1.0)

        # Return positive signal
        return 0
    
# Main program
def main(argv):

    if len(argv) > 1:
        config = Config.Config.load_config_from_module(argv)
    else:
       config = Config.Config()

    logger = IAFLogger.IAFLogger(not config.io.verbose)
    
    datalayer = DataLayer.DataLayer(connection=config.connection, logger=logger)
    # Use the loaded configuration module argument
    # or create a classifier object with only standard settings
    myClassiphyer = IAFautomaticClassiphyer(config=config, logger=logger, datalayer=datalayer)

    # Run the classifier
    myClassiphyer.run()

# Start main
if __name__ == "__main__":
    main(sys.argv)
