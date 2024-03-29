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

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    
# General imports
import time
from datetime import datetime, timedelta

# Imports of local help class for communication with SQL Server
import Config
import SQLDataLayer
import JBGLogger
import JBGTaskRunner
from JBGHandler import JBGHandler

import warnings
# Sklearn issue a lot of warnings sometimes, we suppress them here
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
#warnings.filterwarnings("ignore", category=ResourceWarning)

class AutomaticClassifier:

    # Constructor with arguments
    def __init__(self, config: Config.Config, logger: JBGLogger.JBGLogger, datalayer: SQLDataLayer.DataLayer):
        self.config = config

        self.logger = logger

        self.datalayer = datalayer

        # Replaces, if needed, the given config
        if not self.config.should_train():
            self.config = Config.Config.load_config_from_model_file(self.config.get_model_filename(), config=self.config)

        self.dh, self.mh, self.ph = self.create_handlers()
        

        # Get some timestamps
        self.start_time = time.time()

    def create_handlers(self):
        """ Creates the handler and returns a tuple with handlers """

        self.handler = JBGHandler(self.datalayer, self.config, self.logger)
        dh = self.handler.add_handler("dataset")
        mh = self.handler.add_handler("model")
        ph = self.handler.add_handler("predictions")

        return dh, mh, ph

    def run(self) -> dict:
        """ The classification function """
        self.logger.print_progress(message="Starting up ...", percent=0.0)
        
        # Print a welcoming message for the audience
        self.logger.print_welcoming_message(config=self.config, date_now=datetime.now())
        
        # Do some things prior to running the classification itself
        self.pre_run()
        tr = JBGTaskRunner.TaskRunner( self.datalayer, self.config, self.logger, self.handler)
        early_exit = tr.run(JBGTaskRunner.get_tasks(self.config))

        if early_exit:
            self.logger.print_progress(message="Process finished", percent=1.0)
            return False

        return self.post_run()
        

    def pre_run(self) -> None:
        """ Empty for now """
        # This used to have a function to set a flag in the database, which is no longer used.
        # However, pre_run is a good place to put things that needs to be run in the first stages
        # of the classifier

    def post_run(self) -> dict:
        elapsed_time = time.time() - self.start_time
        date_again = datetime.now()
        message = f"Ending program after {timedelta(seconds=round(elapsed_time))} at {date_again:%Y-%m-%d %H:%M}"
        self.logger.print_progress(message, 1.0)
        
        if self.config.should_display_mispredicted():
            return {
                "mispredicted": self.ph.get_mispredicted_dataframe(),
                "unique_classes": self.dh.classes
            }
        
        return {
            "mispredicted": None
        }
        

# Main program
def main(argv):
    # python AutomaticClassifier.py -f .\config\test_iris.py
    # python AutomaticClassifier.py -f .\config\test_hpl.py
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
    myClassifier = AutomaticClassifier(config=config, logger=logger, datalayer=datalayer)

    # Run the classifier
    myClassifier.run()

# Start main
if __name__ == "__main__":
    sys.exit(main(sys.argv))
