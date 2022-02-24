#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# General IAF code for automatic classification of texts and numbers in databases
#
# Implemented by Robert Granat, March - May 2021
# Updated by Robert Granat, August 2021 - February 2022.
#
# Major revisions:
# 
# * Jan 26 - Febr 27, 2022: Rewritten the code as a Python Class
#
# Standard imports
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Special imports that might need installations below
# Make sure pip is the latest version for the current user
checked_pip = False

def check_pip_installation():
    print("Upgrading pip if necessary...")
    try:
        os.system('python.exe -m pip install -q --upgrade pip')
        return True
    except:
        return False

# Check that importlib is avaiable
try:
    try:
        import importlib
    except:
        checked_pip = check_pip_installation()
        os.system('pip install importlib')
        import importlib
  
    # Install special libs that might cause problems for a new user
    packages = ['pyodbc','wheel','scipy','numpy','pickle','pandas','sklearn', \
        'matplotlib','stop_words', 'langdetect', 'lexicalrichness', 'textblob', \
        'anytree', 'imblearn', 'ipywidgets']

    for package in packages: 
        try:
            importlib.import_module(package)
        except:
            print("Please wait...installing missing package: {0}".format(package))
            if not checked_pip:
                checked_pip = check_pip_installation()
            os.system("pip install " + package)
            try:
                importlib.import_module(package)
            except:
                sys.exit("Please restart script to enable import of added modules!")
                            
except Exception as ex:
    sys.exit("Aborting. Package installation failed: {0}".format(str(ex)))
    

# Imports of local help class for communication with SQL Server
import SqlHelper.IAFSqlHelper as sql

# General imports
import base64, getopt, pickle, scipy, numpy as np
import importlib, getpass
import time as time
import getpass
import copy
from datetime import datetime, timedelta
from numpy.linalg import lstsq, norm
from math import ceil
import matplotlib, pandas, sklearn
from pandas import read_csv, concat
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, Binarizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.preprocessing import LabelBinarizer
import time
import threading
from math import isnan
from stop_words import get_stop_words
import langdetect
from lexicalrichness import LexicalRichness
import warnings
#import ipywidgets as widgets
from pathlib import Path

# Sklearn issue a lot of warnings sometimes, we suppress them here
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class IAFautomaticClassifier:

    # Internal constants
    ALGORITHMS = [ "ALL", "LRN", "KNN", "CART", "GNB", "MNB", "BNB", "CNB", "REC", "PCN", \
        "PAC", "RFC1", "RFC2", "LIN1", "LIN2", "LINP", "SGD", "SGD1", "SGD2", "SGDE", "NCT", \
        "SVC", "LDA", "BDT", "ETC", "ABC", "GBC", "MLPR", "MLPL" ]
    ALGORITHM_NAMES = ["All", "Logistic Regression", "K-Neighbors Classifier", "Decision Tree Classifier", \
        "Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes", "Complement Naive Bayes", \
        "Ridge Classifier", "Perceptron", "Passive Aggressive Classifier", "Random Forest Classifier 1", \
        "Random Forest Classifier 2", "Linear Support Vector L1", "Linear Support Vector L2", \
        "Linear SV L1+L2", "Stochastic Gradient Descent", "Stochastic GD L1", "Stochastic GD L2", \
        "Stochastic GD Elast.", "Nearest Centroid", "Support Vector Classification", \
        "Linear Discriminant Analysis", "Bagging CLassifier", "Extra Trees Classifier", \
        "Ada Boost Classifier", "Gradient Boosting Classifier", "ML Neural Network Relu", \
        "ML Neural Network Sigm" ]
    PREPROCESS = [ "ALL", "NON", "STA", "MIX", "MMX", "NRM", "BIN" ]
    PREPROCESS_NAMES = ["All", "None", "Standard Scaler", "Min-Max Scaler", "Max-Absolute Scaler", "Normalizer", "Binarizer"]
    REDUCTIONS = [ "NON", "RFE", "PCA", "NYS" ]
    REDUCTION_NAMES = ["None", "Recursive Feature Elimination", "Principal Component Analysis", "Nystroem Method"]
    SCORETYPES = [ "accuracy", "balanced_accuracy", "f1_micro", "recall_micro", "precision_micro" ]
    SCORETYPE_NAMES =  ["Accuracy", "Balanced Accuracy", "Balanced F1 Micro", "Recall Micro", "Precision Micro"]
    STANDARD_LANG = "sv"
    SQL_CHUNKSIZE = 1000
    SQL_USE_CHUNKS = True
    LIMIT_IS_CATEGORICAL = 30
    LIMIT_NYSTROEM = 100
    LIMIT_SVC = 10000
    LIMIT_MISPREDICTED = 10
    MAX_HEAD_COLUMNS = 10
    MAX_ITERATIONS = 20000
    DEFAULT_MODEL_FILE_EXTENSION = ".sav"
    CONFIG_FILENAME_PATH = "./config/"
    CONFIG_FILENAME_START = "autoclassconfig_"
    CONFIG_SAMPLE_FILE = CONFIG_FILENAME_START + "sample.py"
    CONFIGURATION_TAGS = ["<name>", "<odbc_driver>", "<host>", "<trusted_connection>", "<class_catalog>", \
                          "<class_table>", "<class_table_script>", "<class_username>", \
                          "<class_password>", "<data_catalog>", "<data_table>", "<class_column>", \
                          "<hierarchical_class>", "<data_text_columns>", "<data_numerical_columns>", "<id_column>", \
                          "<data_username>", "<data_password>", "<train>", "<predict>", "<mispredicted>", \
                          "<use_stop_words>", "<specific_stop_words_threshold>", "<hex_encode>", \
                          "<use_categorization>", "<test_size>", "<smote>", "<undersample>", "<algorithm>", \
                          "<preprocessor>", "<feature_selection>", "<num_selected_features>", "<scoring>", \
                          "<max_iterations>", "<verbose>", "<model_path>", "<model_name>",  "<debug_on>", "<num_rows>"]
    
    # Constructor with arguments
    def __init__(self, config = None, name = "iris", \
        odbc_driver = "ODBC Driver 17 for SQL Server", \
        host = "tcp:sql-stat.iaf.local", trusted_connection = False, \
        class_catalog = "Arbetsdatabas", \
        class_table = "aterkommande_automat.AutoKlassificering", \
        class_table_script = "./sql/autoClassCreateTable.sql", \
        class_username = "robert_tmp", class_password = "robert", \
        data_catalog = "Arbetsdatabas", data_table = "aterkommande_automat.iris", \
        class_column = "class", hierarchical_class = False, data_text_columns = "", \
        data_numerical_columns = "petal-length,petal-width,sepal-length,sepal-width", \
        id_column = "id", data_username = "robert_tmp", data_password = "robert", \
        train = True, predict = True, mispredicted = True, use_stop_words = True, \
        specific_stop_words_threshold = 1.0, hex_encode = True, use_categorization = True, \
        test_size = 0.2, smote = False, undersample = False, algorithm = "ALL", \
        preprocessor = "NON", feature_selection = "NON", num_selected_features = None, \
        scoring = "accuracy", max_iterations = None, verbose = True, redirect_output = False, \
        model_path = "./model/", model_name = "iris", debug_on = True, num_rows = None, \
        progress_bar = None, progress_label = None, save_config_to_file = False):
        
        # In case of a trusted connection, update the sql usernames and password accordingly
        if trusted_connection:
            class_username = os.getlogin()
            class_password = ""
            data_username = os.getlogin()
            data_password = ""
        
        # If configuration module was loaded from command line call, use it and ignore 
        # the rest of the input arguments to the constructor
        if config != None:
            self.config = self.Config( config.project["name"], config.sql["odbc_driver"], \
                config.sql["host"], config.sql["trusted_connection"] , \
                config.sql["class_catalog"], config.sql["class_table"], \
                config.sql["class_table_script"], config.sql["class_username"], \
                config.sql["class_password"], config.sql["data_catalog"],  config.sql["data_table"], \
                config.sql["class_column"], config.sql["hierarchical_class"] , \
                config.sql["data_text_columns"], config.sql["data_numerical_columns"], \
                config.sql["id_column"], config.sql["data_username"], config.sql["data_password"], \
                config.mode["train"] , config.mode["predict"] , \
                config.mode["mispredicted"] , config.mode["use_stop_words"] , \
                float(config.mode["specific_stop_words_threshold"]), config.mode["hex_encode"] , \
                config.mode["use_categorization"] , float(config.mode["test_size"]), \
                config.mode["smote"] , config.mode["undersample"] , \
                config.mode["algorithm"], config.mode["preprocessor"], \
                config.mode["feature_selection"], config.mode["num_selected_features"], \
                config.mode["scoring"], int(config.mode["max_iterations"]), \
                config.io["verbose"] , False, config.io["model_path"], config.io["model_name"], \
                config.debug["debug_on"] , int(config.debug["num_rows"]) )

        # Otherwise, Test input arguments to constructor
        else:
            if not self.is_str(name):
                raise ValueError("Argument name must be a string!")
            elif sql.IAFSqlHelper.drivers().find(odbc_driver) == -1:
                raise ValueError("Specified ODBC driver cannot be found!")
            elif not self.is_str(host) or not self.is_str(class_catalog) or \
                not self.is_str(class_table) or not self.is_str(class_table_script) or \
                not self.is_str(class_username) or not self.is_str(class_password) or \
                not self.is_str(data_catalog) or not self.is_str(data_table):
                raise ValueError("Specified database connection information is invalid!")
            elif not self.is_bool(trusted_connection):
                raise ValueError("Trusted connection must be true or false!")
            elif not self.is_str(class_column) or not self.is_str(data_text_columns) or \
                not self.is_str(data_numerical_columns) or not self.is_str(id_column) or \
                not self.is_str(data_username) or not self.is_str(data_password):
                raise ValueError("Specified data columns or login credentials are invalid!")
            elif not self.is_bool(hierarchical_class) or hierarchical_class:
                raise ValueError("Argument for hierachical class is invalid or currently not supported!")
            elif not self.is_bool(train) or not self.is_bool(predict) or not self.is_bool(mispredicted) or \
                not (train or predict or mispredicted):
                raise ValueError("Class must be set for either training, predictions or mispredictions!") 
            elif not self.is_bool(use_stop_words):
                raise ValueError("Argument use_stop_words must be true or false!")
            elif not self.is_bool(hex_encode):
                raise ValueError("Argument hex_encode must be true or false!")
            elif not self.is_bool(use_categorization):
                raise ValueError("Argument use_categorization must be true or false!")
            elif specific_stop_words_threshold > 1.0 or specific_stop_words_threshold < 0.0:
                raise ValueError("Argument specific_stop_words_threshold must be between 0 and 1!")
            elif not self.is_float(test_size) or test_size > 1.0 or test_size < 0.0:
                raise ValueError("Argument test_size must be between 0 and 1!")
            elif not self.is_bool(smote) or not self.is_bool(undersample):
                raise ValueError("Arguments for SMOTE and/or undersampling are invalid!")
            elif not self.is_str(algorithm) or not algorithm in self.ALGORITHMS:
                raise ValueError("Argument algorithm is invalid!")
            elif not self.is_str(preprocessor) or not preprocessor in self.PREPROCESS:
                raise ValueError("Argument preprocessor is invalid!")
            elif not self.is_str(feature_selection) or not feature_selection in self.REDUCTIONS:
                raise ValueError("Argument feature_selection is invalid!")
            elif num_selected_features != None and not (self.is_int(num_selected_features) and num_selected_features > 0):
                raise ValueError("Argument num_selected_features is invalid!")
            elif not self.is_str(scoring) or not scoring in self.SCORETYPES:
                raise ValueError("Argument scoring is invalid!")
            elif max_iterations != None and not (self.is_int(max_iterations) and max_iterations > 0):
                raise ValueError("Argument max_iterations is invalid!")
            elif not self.is_bool(verbose):
                raise ValueError("Arguments for verbose is invalid!")
            elif not self.is_bool(redirect_output):
                raise ValueError("Arguments for redirect_output is invalid!")
            elif num_rows != None and not (self.is_int(num_rows) and num_rows > 0):
                raise ValueError("Argument num_rows is invalid!")

            # If input okay, create configuration object with given settings
            self.config = self.Config( name, odbc_driver, host, trusted_connection, class_catalog, \
                class_table, class_table_script, class_username, class_password, data_catalog, data_table, \
                class_column, hierarchical_class, data_text_columns, data_numerical_columns, \
                id_column, data_username, data_password, train, predict, mispredicted, use_stop_words, \
                specific_stop_words_threshold, hex_encode, use_categorization, test_size, smote, \
                undersample, algorithm, preprocessor, feature_selection, num_selected_features, \
                scoring, max_iterations, verbose, redirect_output, model_path, model_name, debug_on, num_rows )

        # In case no specific row number was specified, fall back on the total 
        if self.config.debug["num_rows"] == None:
            self.config.debug["num_rows"] = self.count_data_rows()

        # In case of no specific maximum iterations specified, fall back on standard
        if self.config.mode["max_iterations"] == None:
            self.config.mode["max_iterations"] = self.MAX_ITERATIONS
        
        # For the external progressbar (hopefully iPython widgets)
        self.ProgressBar = progress_bar
        self.ProgressLabel = progress_label

        self.numMajorTasks = 12
        self.percentPermajorTask = 0.03
        if self.ProgressBar: self.ProgressBar.value = 0.0
        if self.ProgressLabel: self.ProgressLabel.value = "Starting up..."
        self.majorTask = 0
        self.use_progress_bar = True
        
         # Init some internal variables
        self.scriptpath = os.path.dirname(os.path.realpath(__file__))
        self.scriptname = sys.argv[0][2:]

        # Setting path for misplaced data output file
        if self.config.mode["mispredicted"]:
            self.misplaced_filepath = os.path.dirname(os.path.realpath(__file__)) + \
                    "\\output\\misplaced_"+ self.config.project["name"] + "_" + \
                    self.config.sql["data_username"] + ".csv"
        else:
            self.misplaced_filepath = None
            
        # Prepare variables for unique class lables and misplaced data elements
        self.unique_classes = None
        self.X_most_mispredicted = None
        
        # Extract parameters that are not textual
        self.text_data = self.config.sql["data_text_columns"] != ""
        self.numerical_data = self.config.sql["data_numerical_columns"] != ""
        self.use_feature_selection = self.config.mode["feature_selection"] != "NON"
        
         # Set the name and path of the model file depending on training or predictions
        pwd = os.path.dirname(os.path.realpath(__file__))
        self.model_path = Path(pwd) / self.config.io["model_path"]
        if self.config.mode["train"]:
            self.model_filename = self.model_path / (self.config.project["name"] + ".sav")
        else:
            self.model_filename = self.model_path / self.config.io["model_name"]

        # Redirect standard output to debug files
        if self.config.debug["debug_on"] and self.config.io["redirect_output"]:
            output_file = os.path.dirname(os.path.realpath(__file__)) + \
                "\\output\\output_"+ self.config.project["name"] + "_" + self.config.sql["data_username"] + ".txt"
            error_file = os.path.dirname(os.path.realpath(__file__)) + \
                "\\output\\error_"+ self.config.project["name"] + "_" + self.config.sql["data_username"] + ".txt"

            if self.config.io["verbose"]:
                print("Redirecting standard output to {0} and standard error to {1}".format(output_file,error_file))
            try:
                sys.stdout = open(output_file,'w')
                sys.stderr = open(error_file,'w')
            except Exception as ex:
                sys.exit("Something went wrong with redirection of standard output and error: {0}".format(str(ex)))
        
        # Write configuration to file for easy start from command line
        if save_config_to_file:
            self.export_configuration_to_file()
    
        # Internal settings for panda
        pandas.set_option("max_columns", self.MAX_HEAD_COLUMNS)
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

    # Internal configuration class, used by IAFautomaticClassifier. No default
    # arguments for this class constructor, since they are given in the outer class.
    class Config:
        
        # Constructor with arguments
        def __init__( self, name="", odbc_driver="", host="", trusted_connection=False, \
                     class_catalog="", class_table="", class_table_script="", class_username="", \
                     class_password="", data_catalog="",  data_table="", class_column="", \
                     hierarchical_class=False, data_text_columns="", data_numerical_columns="", \
                     id_column="", data_username="", data_password="", train=True, predict=False, \
                     mispredicted=False, use_stop_words=True, specific_stop_words_threshold=1.0, \
                     hex_encode=True, use_categorization=True, test_size=0.2, smote=False, \
                     undersample=False, algorithm="ALL", preprocessor="ALL", feature_selection="NON", \
                     num_selected_features="", scoring="accuracy", max_iterations="20000", \
                     verbose=True, redirect_output=False, model_path="", model_name="Model", \
                     debug_on=True, num_rows="" ):  

            # Project information, will be used to distinguish data and generated model
            # from other
            self.project = { "name": name }

            # Database configuration
            self.sql = { "odbc_driver": odbc_driver, "host": host, "trusted_connection": trusted_connection, \
                "class_catalog": class_catalog, "class_table": class_table, "class_table_script": class_table_script, \
                "class_username": class_username, "class_password": class_password, \
                "data_catalog": data_catalog, "data_table": data_table, \
                "class_column": class_column, "hierarchical_class": hierarchical_class, \
                "data_text_columns": data_text_columns, "data_numerical_columns": \
                data_numerical_columns, "id_column": id_column, "data_username": data_username, \
                "data_password": data_password }

            # Specified modes of operation
            self.mode = {"train": train, "predict": predict, "mispredicted": mispredicted, \
                "use_stop_words": use_stop_words, "specific_stop_words_threshold": specific_stop_words_threshold, \
                "hex_encode": hex_encode, "use_categorization": use_categorization, \
                "test_size": test_size, "smote": smote, "undersample": undersample, \
                "algorithm": algorithm, "preprocessor": preprocessor, \
                "feature_selection": feature_selection, "num_selected_features": num_selected_features, \
                "scoring": scoring, "max_iterations": max_iterations}

            # Specifies how to direct output, where to save model, etc
            self.io = { "verbose": verbose, "redirect_output": redirect_output, "model_path": model_path, \
                 "model_name": model_name}

            # Some debug settings
            self.debug = { "debug_on": debug_on, "num_rows": num_rows }
        
        # Destructor
        def __del__(self):
            pass
        
        # Print the class 
        def __str__(self):
            output = \
            "Project:" + str(self.project) + "\n" + \
            "SQL:    " + str(self.sql) + "\n" + \
            "Mode:   " + str(self.mode) + "\n" + \
            "I/O:    " + str(self.io) + "\n" + \
            "Debug:  " + str(self.debug)
            return output
    
    # Functions below

    # Create the classification table
    def create_classification_table(self):

        # Read in the sql-query from file
        pwd = os.path.dirname(os.path.realpath(__file__))
        sql_file = open(pwd + "/sql/autoClassCreateTable.sql.txt", mode="rt")
        query = ""
        nextline = sql_file.readline()
        while nextline:
            query += nextline.strip() + "\n"
            nextline = sql_file.readline()
        sql_file.close()

        if self.config.io["verbose"]: print("Creating classification table if not exists...")

        # Get a sql handler and connect to data database
        sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
            host = self.config.sql["host"], catalog = self.config.sql["class_catalog"], \
            trusted_connection = self.config.sql["trusted_connection"], \
            username = self.config.sql["class_username"], \
            password = self.config.sql["class_password"])
        sqlHelper.connect()

        # Execute query
        sqlHelper.execute_query(query, get_data=False, commit = True)

        sqlHelper.disconnect(commit=False)

    # Function to set a new row in classification database which marks that execution has started.
    # If this new row persists when exeuction has ended, this signals that something went wrong.
    # This new row should be deleted before the program ends, which signals all is okay.
    #
    def mark_execution_started(self):

        try:
            if self.config.io["verbose"]: print("Marking execution started in database...-please wait!")

            # Get a sql handler and connect to data database
            sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
                host = self.config.sql["host"], catalog = self.config.sql["class_catalog"], \
                trusted_connection = self.config.sql["trusted_connection"], \
                username = self.config.sql["class_username"], \
                password = self.config.sql["class_password"])
            sqlHelper.connect()

            # Mark execution started by setting unique key to -1 and algorithm to "Not set"

            # Set together SQL code for the insert
            query = "INSERT INTO " + self.config.sql["class_catalog"] + "." 
            query +=  self.config.sql["class_table"] + " (catalog_name,table_name,column_names," 
            query +=  "unique_key,class_result,class_rate,class_rate_type,"
            query += "class_labels,class_probabilities,class_algorithm," 
            query +=  "class_script,class_user) VALUES(\'" + self.config.sql["data_catalog"] + "\'," 
            query +=  "\'" + self.config.sql["data_table"] + "\',\'" 
            if self.text_data:
                query += self.config.sql["data_text_columns"]
            if self.text_data and self.numerical_data:
                query += ","
            if self.numerical_data:
                query +=  self.config.sql["data_numerical_columns"]  
            query +=  "\',-1" + ",\' N/A \',0.0," 
            query +=  "" + "\'U\',\' N/A \',\' N/A \',\'" + "Not set" + "\',\'" 
            query += self.scriptpath + "\',\'" + self.config.sql["data_username"] + "\')"

            # Execute a query without getting any data
            sqlHelper.execute_query(query, get_data=False, commit=True)   

            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=False)

            # Return the number of inserted rows
            return 1
        except Exception as e:
            print("Mark of executionstart failed: " + str(e))
            sys.exit("Program aborted.")

    def mark_execution_ended(self):
        
        try:
            if self.config.io["verbose"]: print("Marking execution ended in database...")

            # Get a sql handler and connect to data database
            sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
                host = self.config.sql["host"], catalog = self.config.sql["class_catalog"], \
                trusted_connection = self.config.sql["trusted_connection"], \
                username = self.config.sql["class_username"], \
                password = self.config.sql["class_password"])
            sqlHelper.connect()

            # Mark execution ended by removing first row with unique key as -1 
            # for the current user and currect script and so forth

            # Set together SQL code for the deletion operation
            query = "DELETE FROM " + self.config.sql["class_catalog"] + "." 
            query +=  self.config.sql["class_table"] + " WHERE "
            query += "catalog_name = \'" + self.config.sql["data_catalog"] + "\' AND "
            query += "table_name = \'" + self.config.sql["data_table"] + "\' AND "
            query += "unique_key = -1 AND "
            query += "class_algorithm = \'" + "Not set" + "\' AND " 
            query += "class_script = \'" + self.scriptpath + "\' AND "
            query += "class_user = \'" + self.config.sql["data_username"] + "\'"

            # Execute a query without getting any data
            sqlHelper.execute_query(query, get_data=False, commit=True)   

            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=False)

            # Return the number of inserted rows
            return 1
        except Exception as e:
            print("Mark of executionend failed: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")

    # Function for reading in data to classify from database
    def read_in_data(self):

        try:
            if self.config.io["verbose"]: print("Reading in dataset from database...-please wait!")

            # Get a sql handler and connect to data database
            sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
                host = self.config.sql["host"], catalog = self.config.sql["data_catalog"], \
                trusted_connection = self.config.sql["trusted_connection"], \
                username = self.config.sql["data_username"], \
                password = self.config.sql["data_password"])
            sqlHelper.connect()

            # Setup and execute a query to get the wanted data. 
            # 
            # The query has the form 
            # 
            #   SELECT columns FROM ( SELECT columns FROM table ORDER BY NEWID()) ORDER BY class_column
            #
            # to ensure a random selection (if not all rows are chosen) and ordering of data, and a 
            # subsequent sorting by class_column to put the NULL-classification last
            #
            query = "SELECT "
            query += "TOP(" + str(self.config.debug["num_rows"]) + ") "
            column_groups = (self.config.sql["id_column"],self.config.sql["class_column"], \
                self.config.sql["data_text_columns"], self.config.sql["data_numerical_columns"])
            for column_group in column_groups:
                columns = column_group.split(',')
                for column in columns:
                    if column != "":
                        query += "[" + column + "],"

            # Remove last comma sign
            query = query[:-1] # Remove last comma sign from last statement above
            query += " FROM "
            outer_query = query
            query += "[" + self.config.sql["data_catalog"] + "].[" + \
                self.config.sql["data_table"].replace(".","].[") + "]"

            # Take care of the special case of only training or only predictions
            if self.config.mode["train"] and not self.config.mode["predict"]:
                query += " WHERE " + self.config.sql["class_column"] + " IS NOT NULL OR " + \
                     self.config.sql["class_column"] + " != \'\' "
            elif not self.config.mode["train"] and self.config.mode["predict"]:
                query += " WHERE " + self.config.sql["class_column"] + " IS NULL OR " + \
                     self.config.sql["class_column"] + " = \'\' "

            # Since sorting the DataFrames directly does not seem to work right now (see below)
            # we sort the data in retreiving in directly in SQL. The "DESC" keyword makes sure
            # all NULL values (unclassified data) is placed last.
            # TODO: If self.config.debug["num_rows"] is less than the number of available records, this query will fetch
            # self.config.debug["num_rows"] randomly selected rows (NEWID does this trick)
            query += " ORDER BY NEWID() "
            query = outer_query + "(" + query + ") A "
            non_ordered_query = query
            query += " ORDER BY [" + str(self.config.sql["class_column"]) + "] DESC"

            if self.config.io["verbose"]: print("Query for classification data: ", query)

            # Now we are ready to execute the sql query
            # By default, all fetched data is placed in one long 1-dim list. The alternative is to read by chunks.
            num_lines = 0
            percent_fetched = 0
            data = []
            if sqlHelper.execute_query(query, get_data=True):

                # Read line by line from the query
                if not self.SQL_USE_CHUNKS:
                    row = sqlHelper.read_data()
                    data = [elem for elem in row]
                    num_lines = 1
                    percent_fetched = round(100.0 * float(num_lines) / float(self.config.debug["num_rows"]))
                    if not self.config.io["redirect_output"]:
                        print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')
                    while row:
                        if self.config.debug["debug_on"] and num_lines >= self.config.debug["num_rows"]:
                            break;
                        row = sqlHelper.read_data()
                        if row:
                            data = data + [elem for elem in row]
                            num_lines += 1
                            old_percent_fetched = percent_fetched
                            percent_fetched = round(100.0 * float(num_lines) / float(self.config.debug["num_rows"]))
                            if not self.config.io["redirect_output"] and percent_fetched > old_percent_fetched:
                                print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')

                    # Rearrange the long 1-dim array into a 2-dim numpy array which resembles the
                    # database table in question
                    data = np.asarray(data).reshape(num_lines,int(len(data)/num_lines)) 

                # Read data lines in chunks from the query
                else:
                    data_chunk = sqlHelper.read_many_data(chunksize=self.SQL_CHUNKSIZE)
                    data = np.asarray(data_chunk)
                    num_lines = len(data_chunk)
                    percent_fetched = round(100.0 * float(num_lines) / float(self.config.debug["num_rows"]))
                    if self.config.io["verbose"] and not self.config.io["redirect_output"]:
                        print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')
                    while data_chunk:
                        if self.config.debug["debug_on"] and num_lines >= self.config.debug["num_rows"]:
                            break;
                        data_chunk = sqlHelper.read_many_data(chunksize=self.SQL_CHUNKSIZE)
                        if data_chunk:
                            data = np.append(data, data_chunk, axis = 0)
                            num_lines += len(data_chunk)
                            old_percent_fetched = percent_fetched
                            percent_fetched = round(100.0 * float(num_lines) / float(self.config.debug["num_rows"]))
                            if self.config.io["verbose"] and not self.config.io["redirect_output"] and percent_fetched > old_percent_fetched:
                                print("Data fetched of available: " + str(percent_fetched) + " %", end='\r')

            if self.config.io["verbose"]: print("\n--- Totally fetched {0} data rows ---".format(num_lines))

            # Disconnect from database
            sqlHelper.disconnect()

            # Set the column names of the data array
            column_names = [self.config.sql["id_column"]] + [self.config.sql["class_column"]] + \
                    self.config.sql["data_text_columns"].split(',') + \
                    self.config.sql["data_numerical_columns"].split(',')
            try:
                column_names.remove("") # Remove any empty column name
            except Exception as ex:
                print("\nNotice: attempt to remove empty column name failed.")
            dataset = pandas.DataFrame(data, columns = column_names)
            
            # Extract unique class labels from dataset
            unique_classes = list(set(dataset[self.config.sql["class_column"]].tolist()))

            # Make an extensive search through the data for any inconsistencies (like NaNs and NoneType). 
            # Also convert datetime numerical variables to ordinals, i.e., convert them to the number of days 
            # or similar from a certain starting point, if any are left after the conversion above.
            if self.config.io["verbose"]: print("\nConsistency check")
            change = False
            percent_checked = 0
            try:
                for index in dataset.index:
                    old_percent_checked = percent_checked
                    percent_checked = round(100.0*float(index)/float(len(dataset.index)))
                    if self.config.io["verbose"] and not self.config.io["redirect_output"] and percent_checked > old_percent_checked:
                        print("Data checked of fetched: " + str(percent_checked) + " %", end='\r')
                    for key in dataset.columns:
                        item = dataset.at[index,key]

                        # Set NoneType objects  as zero or empty strings
                        if (key in self.config.sql["data_numerical_columns"].split(",") or \
                            key in self.config.sql["data_text_columns"].split(",")) and item == None:
                            if key in self.config.sql["data_numerical_columns"].split(","):
                                item = 0
                            else:
                                item = ""
                            change = True

                        # Convert numerical datetime values to ordinals
                        elif key in self.config.sql["data_numerical_columns"].split(",") and self.is_datetime(item):
                            item = datetime.toordinal(item)
                            change = True

                        # Set remaining numerical values that cannot be casted as integer or floating point numbers to zero, i.e., do not
                        # take then into account
                        elif key in self.config.sql["data_numerical_columns"].split(",") and \
                            not (self.is_int(item) or self.is_float(item)):
                            item = 0
                            change = True

                        # Set text values that cannot be casted as strings to empty strings
                        elif key in self.config.sql["data_text_columns"].split(",") and type(item) != str and not self.is_str(item):
                            item = ""
                            change = True

                        # Remove line breaks from text strings
                        if key in self.config.sql["data_text_columns"].split(","):
                            item = item.replace('\n'," ").replace('\r'," ").strip()
                            change = True

                        # Save new value
                        if change:
                            dataset.at[index,key] = item
                            change = False
            except Exception as ex:
                print("Warning: {0}. Something went wrong in inconsistency check at {1}: {2}. Continuing, but at risk.".format(str(ex),key,item))

            # Shuffle the upper part of the data, such that all already classified material are
            # put in random order keeping the unclassified material in the bottom
            # Notice: perhaps this operation is now obsolete, since we now use a NEWID() in the 
            # data query above
            if self.config.io["verbose"]: print("\n\n---Shuffle data")
            num_un_pred = self.get_num_unpredicted_rows(dataset)
            dataset['rnd'] = np.concatenate([np.random.rand(num_lines - num_un_pred), [num_lines]*num_un_pred])
            dataset.sort_values(by = 'rnd', inplace = True )
            dataset.drop(['rnd'], axis = 1, inplace = True )

            # Use the unique id column from the data as the index column and take a copy, 
            # since it will not be used in the classification but only to reconnect each 
            # data row with classification results later on
            keys = dataset[self.config.sql["id_column"]].copy(deep = True).apply(IAFautomaticClassifier.get_rid_of_decimals)
            try:
                dataset.set_index(keys.astype('int64'), drop=False, append=False, inplace=True, \
                                  verify_integrity=False)
            except Exception as ex:
                sys.exit("Could not set index for dataset: {0}".format(str(ex)))
            dataset = dataset.drop([self.config.sql["id_column"]], axis = 1)

        except Exception as e:
            print("Load of dataset failed: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")

        # Return bort data and keys
        return dataset, keys, non_ordered_query, unique_classes
    
    @staticmethod
    def get_rid_of_decimals(x):
        try:
            return int(round(float(x)))
        except Exception as ex:
            sys.exit("Could not convert {0} to integer: {1}".format(x,str(ex)))

    # Function for counting the number of rows in data to classify
    def count_data_rows(self):

        try:

            # Get a sql handler and connect to data database
            sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
                host = self.config.sql["host"], catalog = self.config.sql["data_catalog"], \
                trusted_connection = self.config.sql["trusted_connection"], \
                username = self.config.sql["data_username"], \
                password = self.config.sql["data_password"])
            sqlHelper.connect()
            #
            query = "SELECT COUNT(*) FROM "
            query += "[" + self.config.sql["data_catalog"] + "].[" + self.config.sql["data_table"].replace(".","].[") + "]"

            # Now we are ready to execute the sql query
            # By default, all fetched data is placed in one long 1-dim list. The alternative is to read by chunks.
            num_lines = 0
            data = []
            if sqlHelper.execute_query(query, get_data=True):
                count = sqlHelper.read_data()[0]

            # Disconnect from database
            sqlHelper.disconnect()

        except Exception as e:
            print("Count of dataset failed: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")

        # Return 
        return count
    
    # Function for counting the number of rows in data corresponding to each class
    def count_class_distribution(self):
        try:

            # Get a sql handler and connect to data database
            sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
                host = self.config.sql["host"], catalog = self.config.sql["data_catalog"], \
                trusted_connection = self.config.sql["trusted_connection"], \
                username = self.config.sql["data_username"], \
                password = self.config.sql["data_password"])
            sqlHelper.connect()
            
            # Construct the query
            query = "SELECT " + self.config.sql["class_column"] + ", COUNT(*) FROM "
            query += "[" + self.config.sql["data_catalog"] + "].[" + self.config.sql["data_table"].replace(".","].[") + "] "
            query += "GROUP BY " + self.config.sql["class_column"] + " ORDER BY " + self.config.sql["class_column"] + " DESC"
            
            # Now we are ready to execute the sql query
            # By default, all fetched data is placed in one long 1-dim list. The alternative is to read by chunks.
            num_lines = 0
            data = []
            if sqlHelper.execute_query(query, get_data=True):
                data = sqlHelper.read_all_data()
            dict = {}
            for pair in data:
                key, value = pair
                if key == None:
                    key = "NULL"
                dict[key] = value
            
            # Disconnect from database
            sqlHelper.disconnect()

        except Exception as e:
            print("Count of dataset distribution failed: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")

        # Return 
        return dict
    
    # Get a list of pretrained models
    def get_trained_models(self):
        models = []
        for file in os.listdir(self.model_path):
            if file[-len(self.DEFAULT_MODEL_FILE_EXTENSION):] == self.DEFAULT_MODEL_FILE_EXTENSION:
                models.append(file)
        return models
                
    # Help routines for determining consistency of input data
    @staticmethod
    def is_float(val):
        try:
            num = float(val)
        except ValueError:
            return False
        return True
    
    @staticmethod
    def is_int(val):
        try:
            num = int(val)
        except ValueError:
            return False
        return True
    
    @staticmethod
    def is_bool(val):
        return type(val)==bool

    @staticmethod
    def is_str(val):
        is_other_type = \
            IAFautomaticClassifier.is_float(val) or \
            IAFautomaticClassifier.is_int(val) or \
            IAFautomaticClassifier.is_bool(val) or \
            IAFautomaticClassifier.is_datetime(val)
        return not is_other_type 
    
    @staticmethod
    def is_datetime(val):
        try:
            if isinstance(val,datetime): #Simple, is already instance of datetime
                return True
        except ValueError:
            pass
        # Harder: test the value for many different datetime formats and see if any is correct.
        # If so, return true, otherwise, return false.
        the_formats = ['%Y-%m-%d %H:%M:%S','%Y-%m-%d','%Y-%m-%d %H:%M:%S.%f','%Y-%m-%d %H:%M:%S,%f', \
                       '%d/%m/%Y %H:%M:%S','%d/%m/%Y','%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S,%f']
        for the_format in the_formats:
            try:
                date_time = datetime.strptime(str(val), the_format)
                return isinstance(date_time, datetime)
            except ValueError:
                pass
        return False

    # Calculate the number of unclassified rows in data matrix
    def get_num_unpredicted_rows(self, dataset):

        num = 0
        for item in dataset[self.config.sql["class_column"]]:
            if item == None: # Corresponds to empty string and SQL NULL
                num += 1
        return num   

    # Find out if a DataFrame column contains categorical data or not
    def is_categorical_data(self, column):

        is_categorical = column.value_counts().count() <= self.LIMIT_IS_CATEGORICAL

        return is_categorical

    # Collapse all data text columns into a new column, which is necessary
    # for word-in-a-bag-technique
    def convert_textdata_to_numbers(self, dataset, label_binarizers = {}, count_vectorizer = None, \
        tfid_transformer = None):

        # Pick out the classification column. This is the 
        # "right hand side" of the problem.
        Y = dataset[self.config.sql["class_column"]]

        # Continue with "left hand side":
        # Prepare two empty DataFrames
        text_dataset = pandas.DataFrame()
        num_dataset = pandas.DataFrame()
        categorical_dataset = pandas.DataFrame()
        binarized_dataset = pandas.DataFrame()

        # First, separate all the text data from the numerical data, and
        # make sure to find the categorical data automatically
        if self.text_data:
            text_columns = self.config.sql["data_text_columns"].split(',')
            for column in text_columns:
                if not self.config.mode["use_categorization"] or not self.is_categorical_data(dataset[column]) \
                    or (len(label_binarizers) > 0 and not column in label_binarizers.keys()):
                    if self.config.io["verbose"]: print("Text data NOT picked for categorization: ",column)
                    text_dataset = \
                        concat([text_dataset, dataset[column]], axis = 1)
                else:
                    if self.config.io["verbose"]: print("Text data picked for categorization: ",column)
                    categorical_dataset = \
                        concat([categorical_dataset, dataset[column]], axis = 1)
        if self.numerical_data:
            num_columns = self.config.sql["data_numerical_columns"].split(',')
            for column in num_columns:
                num_dataset = concat([num_dataset, dataset[column]], axis = 1)

        # For concatenation, we need to make sure all text data are 
        # really treated as text, and categorical data as categories
        if self.text_data:
            if self.config.io["verbose"]: print("Text Columns:",text_dataset.columns)
            if len(text_dataset.columns) > 0:

                text_dataset = text_dataset.applymap(str)

                # Concatenating text data such that the result is another DataFrame  
                # with a single column
                text_dataset = text_dataset.agg(' '.join, axis = 1)

                # Convert text data to numbers using word-in-a-bag technique
                text_dataset, count_vectorizer, tfid_transformer = \
                    self.word_in_a_bag_conversion(text_dataset, count_vectorizer, tfid_transformer)
                text_dataset = pandas.DataFrame(text_dataset)

            # Continue with handling the categorical data.
            # Binarize the data, resulting in more columns.
            if len(categorical_dataset.columns) > 0:
                categorical_dataset = categorical_dataset.applymap(str)
                generate_new_label_binarizers = (len(label_binarizers) == 0)
                for column in categorical_dataset.columns:
                    lb = None
                    if generate_new_label_binarizers:
                        lb = LabelBinarizer()
                        lb.fit(categorical_dataset[column])
                        label_binarizers[column] = lb
                    else:
                        lb = label_binarizers[column]
                    lb_results = lb.transform(categorical_dataset[column])
                    try:
                        if lb_results.shape[1] > 1:
                            lb_results_df = pandas.DataFrame(lb_results, columns=lb.classes_)
                            if self.config.io["verbose"]: 
                                print("Column {0} was categorized with categories: {1}".format(column,lb.classes_))
                        else:
                            lb_results_df = pandas.DataFrame(lb_results, columns=[column])
                            if self.config.io["verbose"]:
                                print("Column {0} was binarized".format(column))
                    except ValueError as ex:
                        print("Column {0} could not be binarized: {1}".format(column,str(ex)))
                    binarized_dataset = concat([binarized_dataset, lb_results_df], axis = 1 )

        # Set together all the data before returning
        X = None
        if self.text_data:
            text_dataset.set_index(dataset.index, drop=False, append=False, inplace=True, \
                                   verify_integrity=False)
            X = text_dataset
        if self.numerical_data:
            num_dataset.set_index(dataset.index, drop=False, append=False, inplace=True, \
                                  verify_integrity=False)
            X = concat([num_dataset, X], axis = 1)
        if self.text_data and binarized_dataset.shape[1] > 0:
            binarized_dataset.set_index(dataset.index, drop=False, append=False, inplace=True, \
                                        verify_integrity=False)
            X = concat([binarized_dataset, X], axis = 1)
            
        return X, Y, label_binarizers, count_vectorizer, tfid_transformer

    # Use the bag of words technique to convert text corpus into numbers
    def word_in_a_bag_conversion(self, dataset, count_vectorizer = None, tfid_transformer = None):

        # Start working with datavalues in array
        X = dataset.values

        # Find actual languange if stop words are used
        if count_vectorizer == None:
            my_language = None
            if self.config.mode["use_stop_words"]:
                try:
                    my_language = langdetect.detect(' '.join(X))
                except Exception as ex:
                    my_language = self.STANDARD_LANG
                    print("Language could not be detected automatically: {0}. Fallback option, use: {1}.". \
                        format(str(ex),my_language))
                else:
                    if self.config.io["verbose"]: print("\nDetected language is: {0}".format(my_language))

            # Calculate the lexical richness
            try:
                lex = LexicalRichness(' '.join(X)) 
                if self.config.io["verbose"]: 
                    print("#Words, #Terms and TTR for original text is {0}, {1}, {2:5.2f} %"
                      .format(lex.words,lex.terms,100*float(lex.ttr)))
            except Exception as ex:
                print("Could not calculate lexical richness: {0}".format(str(ex)))

        # Mask all material by encryption (optional)
        if (self.config.mode["hex_encode"]):
            X = self.do_hex_base64_encode_on_data(X)

        # Text must be turned into numerical feature vectors ("bag-of-words"-technique).
        # If selected, remove stop words
        if count_vectorizer == None:
            my_stop_words = None
            if self.config.mode["use_stop_words"]:

                # Get the languange specific stop words and encrypt them if necessary
                my_stop_words = get_stop_words(my_language)
                if self.config.io["verbose"]: print("\nUsing standard stop words: ", my_stop_words)
                if (self.config.mode["hex_encode"]):
                    for word in my_stop_words:
                        word = self.cipher_encode_string(str(word))

                # Collect text specific stop words (already encrypted if encryption is on)
                text_specific_stop_words = []
                if self.config.mode["specific_stop_words_threshold"] < 1.0:
                    try:
                        stop_vectorizer = CountVectorizer(min_df = self.config.mode["specific_stop_words_threshold"])
                        stop_vectorizer.fit_transform(X)
                        text_specific_stop_words = stop_vectorizer.get_feature_names()
                        if self.config.io["verbose"]: print("\nUsing specific stop words: ", text_specific_stop_words)
                    except ValueError as ex:
                        if self.config.io["verbose"]: print( \
                            "\nNotice: Specified stop words threshold at {0} generated no stop words." \
                            .format(self.config.mode["specific_stop_words_threshold"]))
                    my_stop_words = sorted(set(my_stop_words + text_specific_stop_words))
                    if self.config.io["verbose"]: print("\nTotal list of stop words:", my_stop_words)

            # Use the stop words and count the words in the matrix        
            count_vectorizer = CountVectorizer(stop_words = my_stop_words)
            count_vectorizer.fit(X)

        # Do the word in a bag now
        X = count_vectorizer.transform(X)

        # Also generate frequencies instead of occurences to normalize the information.
        if tfid_transformer == None:
            tfid_transformer = TfidfTransformer(use_idf = False)
            tfid_transformer.fit(X) 

        # Generate the sequences
        X = (tfid_transformer.transform(X)).toarray()

        return X, count_vectorizer, tfid_transformer

    # Investigating dataset -- make some printouts to standard output
    def investigate_dataset(self, dataset, show_class_distribution = True):

        try:
            if self.config.io["verbose"]: print("Looking at dataset:")
            # 1. shape
            if self.config.io["verbose"]: print("\nShape:", dataset.shape)
        except Exception as ex:
           if self.config.io["verbose"]: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))
        try:
            # 2. head
            if self.config.io["verbose"]: print("\nHead:",dataset.head(20))
        except Exception as ex:
            if self.config.io["verbose"]: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))
        try:
            # 3. Data types
            if self.config.io["verbose"]: print("\nDatatypes:",dataset.dtypes)
        except Exception as ex:
            if self.config.io["verbose"]: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))
        if show_class_distribution:
            try:
            # 4. Class distribution
                if self.config.io["verbose"]: print("Class distribution: ")
                if self.config.io["verbose"]: print(dataset.groupby(dataset.columns[0]).size())
            except Exception as ex:
                if self.config.io["verbose"]: print("Notice: an error occured in investigate_dataset: {0}".format(str(ex)))

    # Show statistics in standard output
    def show_statistics_on_dataset(self, dataset):

        # 1. Descriptive statistics
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.describe(datetime_is_numeric = True)
        if self.config.io["verbose"]: print("\nDescription:")
        if self.config.io["verbose"]: print(description)

        # 2. Correlations
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.corr('pearson')
        if self.config.io["verbose"]: print("\nCorrelation between attributes:")
        if self.config.io["verbose"]: print(description)

        # 3. Skew
        skew = dataset.skew()
        if self.config.io["verbose"]: print("\nSkew of Univariate descriptions")
        if self.config.io["verbose"]: print(skew)
        if self.config.io["verbose"]: print("\n")

    # Convert dataset to unreadable hex code
    @staticmethod
    def do_hex_base64_encode_on_data(X):

        XX = X.copy()

        with np.nditer(XX, op_flags=['readwrite'], flags=["refs_ok"]) as iterator:
            for x in iterator:
                xval = str(x)
                xhex = IAFautomaticClassifier.cipher_encode_string(xval)
                x[...] = xhex 

        return XX

    @staticmethod 
    def cipher_encode_string(a):

        aa = a.split()
        b = ""
        for i in range(len(aa)):
            b += (str(
                    base64.b64encode(
                        bytes(aa[i].encode("utf-8").hex(),'utf-8')
                    )
                ) + " ")

        return b.strip()

    # Feature selection (reduction) function for PCA or Nystroem transformation of data.
    # (RFE feature selection is built into the model while training and does not need to be
    # considered here.)
    def perform_feature_selection(self, X, number_of_components = None, feature_selection_transform = None ):
        components = None

        # In the case of training, we compute a new transformation
        if feature_selection_transform == None:

            # PCA first. If not the number of features is specified by
            # the user, we rely on  Minka’s MLE to guess the optimal dimension.
            # Notice, MLE can only be used if the number of samples exceeds the number
            # of features, which is common, but not always. If not, rely on some
            # heuristic choice of the number of features.
            # Notice also, that PCA cannot be use together with PLS.
            if self.config.mode["feature_selection"] == "PCA" and self.config.mode["algorithm"] != "PLS":
                if self.config.io["verbose"]: print("\nPCA conversion of dataset under way...")
                if self.ProgressLabel: self.ProgressLabel.value = "PCA conversion of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                elif X.shape[0] >= X.shape[1]:
                    components = 'mle'
                else:
                    components = X.shape[0]-1
                    print("Notice: PCA n_components is set to n_samples-1: {0}".format(components))
                # Make transformation
                try:
                    feature_selection_transform = PCA(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: PCA could not be used: {0}".format(str(ex)))
                else:
                    if self.config.io["verbose"]: print("...new shape of data matrix is: ({0},{1})\n".
                                      format(X.shape[0],X.shape[1]))
                components = feature_selection_transform.n_components_

            elif self.use_feature_selection and self.config.mode["feature_selection"] == "PCA" and self.config.mode["algorithm"] == "PLS":
                if self.config.io["verbose"]: print("PCA feature reduction was not used because algorithm PLS was chosen...")

            # Nystroem transformation next
            elif self.config.mode["feature_selection"] == "NYS":
                if self.ProgressLabel: self.ProgressLabel.value = "Nystroem conversion of dataset under way..."
                if number_of_components != None and number_of_components > 0:
                    components = number_of_components
                else:
                    components = min(self.LIMIT_NYSTROEM,X.shape[1])
                    print("Notice: Nystroem n_components is set to minimum of number of data column and {1}: {0}".\
                        format(components,self.LIMIT_NYSTROEM))
                # Make transformation
                try:
                    feature_selection_transform = Nystroem(n_components=components)
                    feature_selection_transform.fit(X)
                    X = feature_selection_transform.transform(X)
                except Exception as ex:
                    print("Notice: Nystroem transform could not be used: {0}".format(str(ex)))
                else:
                    if self.config.io["verbose"]: print("...new shape of data matrix is: ({0},{1})\n".format(X.shape[0],X.shape[1])) 

        # For only predictions, use the saved transform associated with trained model
        else:
            X = feature_selection_transform.transform(X)

        return X, components, feature_selection_transform

    # Split dataset into training and validation parts
    def split_dataset(self, X, Y, num_lower):

        # First, split X and Y in two parts: the upper part, to be used in training,
        # and the lower part, to be classified later on
        [X_upper, X_lower] = np.split(X, [X.shape[0]-num_lower], axis = 0)
        [Y_upper, Y_lower] = np.split(Y, [Y.shape[0]-num_lower], axis = 0)

        # Split-out validation dataset from the upper part (do not use random order here)
        testsize = float(self.config.mode["test_size"])
        X_train, X_validation, Y_train, Y_validation = \
                 train_test_split( X_upper, Y_upper, test_size = testsize, \
                 shuffle = False, random_state = None, stratify = None )

        # Return all parts of the data, including the unclassified
        return X_train, X_validation, X_lower, Y_train, Y_validation, Y_lower

    # Spot Check Algorithms.
    # We do an extensive search of the best algorithm in comparison with the best
    # preprocessing.
    def spot_check_ml_algorithms(self, X_train, Y_train, k=10):

        # Save standard progress text
        if self.ProgressLabel:
            standardProgressText = self.ProgressLabel.value

        # Add all algorithms in a list
        if self.config.io["verbose"]: print("Spot check ml algorithms...")
        models = []
        models.append(('LRN', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('GNB', GaussianNB()))
        models.append(('MNB', MultinomialNB(alpha=.01)))
        models.append(('BNB', BernoulliNB(alpha=.01)))
        models.append(('CNB', ComplementNB(alpha=.01)))
        models.append(("REC", RidgeClassifier(tol=1e-2, solver="sag")))
        models.append(("PCN", Perceptron(max_iter=int(self.config.mode["max_iterations"]))))
        models.append(("PAC", PassiveAggressiveClassifier(max_iter=int(self.config.mode["max_iterations"]))))
        models.append(("RFC1", RandomForestClassifier(n_estimators=100, max_features= 3)))
        models.append(("RFC2", RandomForestClassifier()))
        models.append(("LIN1", LinearSVC(penalty="l1", dual=False, tol=1e-3, max_iter=int(self.config.mode["max_iterations"]))))
        models.append(("LIN2", LinearSVC(penalty="l2", dual=False, tol=1e-3, max_iter=int(self.config.mode["max_iterations"]))))
        models.append(("LINP", Pipeline([ \
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3, max_iter=int(self.config.mode["max_iterations"])))), \
            ('classification', LinearSVC(penalty="l2", max_iter=int(self.config.mode["max_iterations"])))])))
        models.append(('SGD', SGDClassifier()))
        models.append(('SGD1', SGDClassifier(alpha=.0001, max_iter=int(self.config.mode["max_iterations"]), penalty="l1")))
        models.append(('SGD2', SGDClassifier(alpha=.0001, max_iter=int(self.config.mode["max_iterations"]), penalty="l2")))
        models.append(('SGDE', SGDClassifier(alpha=.0001, max_iter=int(self.config.mode["max_iterations"]), penalty="elasticnet")))
        models.append(('NCT', NearestCentroid()))
        if X_train.shape[0] < self.LIMIT_SVC:
            models.append(('SVC', SVC(gamma='auto', probability=True)))
        elif self.config.mode["algorithm"] != "ALL":
            models.append(('SVC', LinearSVC(penalty="l1", dual=False, tol=1e-3, max_iter=int(self.config.mode["max_iterations"]))))
            if self.config.io["verbose"]: print("\nNotice: SVC model was exchange for LinearSVC since n_samples > {0}\n".format(self.LIMIT_SVC))
        else:
            if self.config.io["verbose"]: print("\nNotice: SVC model was ignored since n_samples > {0}\n".format(self.LIMIT_SVC))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('BDT', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators = 100, random_state = 7)))
        models.append(('ETC', ExtraTreesClassifier(n_estimators = 100)))
        models.append(('ABC', AdaBoostClassifier(n_estimators = 30, random_state = 7)))
        models.append(('GBC', GradientBoostingClassifier(n_estimators = 100, random_state = 7)))
        models.append(('MLPR', MLPClassifier(activation = 'relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), \
            max_iter=int(self.config.mode["max_iterations"]), random_state=1)))
        models.append(('MLPL', MLPClassifier(activation = 'logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), \
            max_iter=int(self.config.mode["max_iterations"]), random_state=1)))

        # Add different preprocessing methods in another list
        preprocessors = []
        preprocessors.append(('NON', None))
        preprocessors.append(('STA', StandardScaler(with_mean=False)))
        preprocessors.append(('MIX', MinMaxScaler()))
        preprocessors.append(('MMX', MaxAbsScaler()))
        preprocessors.append(('NRM', Normalizer()))
        if self.text_data:
            preprocessors.append(('BIN', Binarizer()))

        # Evaluate each model in turn in combination with all preprocessing methods
        results = []
        names = []
        best_mean = 0.0
        best_std = 1.0
        trained_model = None
        temp_model = None
        trained_model_name = None
        best_feature_selection = X_train.shape[1]
        first_round = True
        if self.config.io["verbose"]:
                print("{0:>4s}-{1:<6s}{2:>6s}{3:>8s}{4:>8s}{5:>11s}".format("Name","Prep.","#Feat.","Mean","Std","Time"))
                print("="*45)
        numMinorTasks = len(models) * len(preprocessors)
        percentAddPerMinorTask = (1.0-self.percentPermajorTask*self.numMajorTasks) / float(numMinorTasks)

        # Loop over the models
        for name, model in models:

            # Loop over pre-processing methods
            for preprocessor_name, preprocessor in preprocessors:

                # Update progressbar percent and label
                if self.ProgressLabel: self.ProgressLabel.value = standardProgressText + " (" + name + "-" + preprocessor_name + ")"
                if not first_round:
                    if self.ProgressBar: self.ProgressBar.value += percentAddPerMinorTask
                else:
                    first_round = False;

                # If the user has specified a specific algorithm and/or preprocessor, restrict
                # computations accordingly            
                if (name != self.config.mode["algorithm"] and not self.config.mode["algorithm"] == "ALL") or \
                    (preprocessor_name != self.config.mode["preprocessor"] and not self.config.mode["preprocessor"] == "ALL"):
                    continue

                # Add feature selection if selected, i.e., the option of reducing the number of variables used.
                # Make a binary search for the optimal dimensions.
                max_features_selection = X_train.shape[1]

                # Make sure all numbers are propely set for feature selection interval
                if self.use_feature_selection and self.config.mode.num_selected_features == "":
                    min_features_selection = 0
                elif self.use_feature_selection and self.config.mode.num_selected_features > 0:
                    min_features_selection = self.config.mode.num_selected_features
                    max_features_selection = self.config.mode.num_selected_features
                else:
                    min_features_selection = max_features_selection # No or minimal number of features are eliminated

                # Loop over feature selections span: break this loop when min and max reach the same value
                score = 0.0                                                 # Save the best values
                stdev = 1.0                                                 # so far.
                num_features = max_features_selection                       # Start with all features.
                first_feature_selection = True                              # Make special first round: use all features
                counter = 0
                while first_feature_selection or min_features_selection < max_features_selection:
                    counter += 1

                    # Update limits for binary search and break loop if we are done
                    if not first_feature_selection:
                        num_features = ceil((min_features_selection+max_features_selection) / 2)
                        if num_features == max_features_selection:          
                            break
                    else:
                        first_feature_selection = False
                        num_features = max_features_selection

                    # Calculate the time for this setting
                    t0 = time.time()

                    # Apply feature selection to current model and number of features.
                    # If feature selection is not applicable, set a flag to the loop is 
                    # ending after one iteration
                    temp_model = model
                    if self.use_feature_selection and self.config.mode["feature_selection"] == "RFE":     
                        try:
                            rfe = RFE(temp_model, n_features_to_select=num_features)
                            temp_model = rfe.fit(X_train, Y_train)
                        except ValueError as ex:
                            break

                    # Both algorithm and preprocessor should be used. Move on.
                    # Build pipline of model and preprocessor.
                    names.append(name+preprocessor_name)
                    if not self.config.mode["smote"] and not self.config.mode["undersample"]:
                        if preprocessor != None:
                            pipe = make_pipeline(preprocessor, temp_model)
                        else:
                            pipe = temp_model

                    # For use SMOTE and undersampling, different Pipeline is used
                    else:
                        steps = None
                        smote = None
                        undersampler = None
                        if self.config.mode["smote"]:
                            smote = SMOTE(sampling_strategy='auto')
                        if self.config.mode["undersample"]:
                            undersampler = RandomUnderSampler(sampling_strategy='auto')
                        steps = [('smote', smote ), ('under', undersampler), \
                                ('preprocessor', preprocessor), ('model', temp_model)]
                        pipe = ImbPipeline(steps=steps)

                    # Make evaluation of model
                    try:
                        kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
                    except Exception as ex:
                        print("NOTICE: StratifiedKfold raised an exception with message: {1}. Killing the program".format(names[-1], str(ex)))
                        if self.ProgressBar: self.ProgressBar.value = 1.0
                        sys.exit()

                    # Now make kfolded cross evaluation
                    cv_results = None
                    try:
                        cv_results = cross_val_score(pipe, X_train, Y_train, cv=kfold, scoring=self.config.mode["scoring"]) 
                    except Exception as ex:
                        if self.config.io["verbose"]: print("NOTICE: Model {0} raised an exception in cross_val_score with message: {1}. Skipping to next".
                                          format(names[-1], str(ex)))
                    else:
                        # Stop the stopwatch
                        t = time.time() - t0

                        # For current settings, calculate score
                        temp_score = cv_results.mean()
                        temp_stdev = cv_results.std()

                        # Print results to screen
                        if self.config.io["verbose"]:
                            print("{0:>4s}-{1:<6s}{2:6d}{3:8.3f}{4:8.3f}{5:11.3f} s.".
                                  format(name,preprocessor_name,num_features,temp_score,temp_stdev,t))

                        # Evaluate if feature selection changed accuracy or not. 
                        # Notice: Better or same score with less variables are both seen as an improvement,
                        # since the chance of finding an improvement increases when number of variables decrease
                        if  temp_score >= score:
                            score = temp_score
                            stdev = temp_stdev
                            max_features_selection = num_features   # We need to reduce more features
                        else:
                            min_features_selection = num_features   # We have reduced too much already  

                         # Save result if it is the overall best
                         # Notice the difference from above, here we demand a better score.
                        if temp_score > best_mean or \
                           (temp_score == best_mean and temp_stdev < best_std):
                            trained_model = pipe
                            trained_model_name = name + '-' + preprocessor_name
                            best_mean = temp_score
                            best_std = temp_stdev
                            best_feature_selection = num_features

        # Return best model for start making predictions
        return trained_model_name, trained_model, best_feature_selection

    # Train ml model
    def train_picked_model(self, model, X, Y):

        # Train model
        try:
            model.fit(X, Y)
        except Exception as ex:
            sys.exit("Something went wrong on training picked model {0}: {1}".format(str(model),str(ex)))

        return model

    # Save ml model and corresponding configuration
    def save_model_to_file(self, label_binarizers, count_vectorizer, tfid_transformer, \
        feature_selection_transform, model_name, model, filename):
        
        try:
            save_config = self.get_configuration_to_save()
            data = [save_config, label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, model_name, model]
            pickle.dump(data, open(filename,'wb'))

        except Exception as ex:
            print("Something went wrong on saving model to file: {0}".format(str(ex)))
    
    # A help method to extract the config information to save
    def get_configuration_to_save(self):
        configuration = self.Config()
        configuration.sql = copy.deepcopy(self.config.sql)
        configuration.sql.pop("data_catalog")
        configuration.sql.pop("data_table")
        configuration.mode = copy.deepcopy(self.config.mode)
        configuration.mode.pop("train")
        configuration.mode.pop("predict")
        configuration.mode.pop("mispredicted")
        configuration.io = copy.deepcopy(self.config.io)
        configuration.io.pop("model_name")
        configuration.debug = copy.deepcopy(self.config.debug)
        configuration.debug.pop("num_rows")
        return configuration

    # Load ml model with corresponding configuration
    def load_model_from_file(self, filename):

        try:
            saved_config, label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, model_name, model = pickle.load(open(filename, 'rb'))
            saved_config.mode["train"] = self.config.mode["train"]
            saved_config.mode["predict"] = self.config.mode["predict"]
            saved_config.mode["mispredicted"] = self.config.mode["mispredicted"]
            saved_config.sql["data_catalog"] = self.config.sql["data_catalog"]
            saved_config.sql["data_table"] = self.config.sql["data_table"]
            saved_config.io["model_name"] = self.config.io["model_name"]
            saved_config.debug["num_rows"] = self.config.debug["num_rows"]
            self.config = saved_config
        except Exception as ex:
            print("Something went wrong on loading model from file: {0}".format(str(ex)))

        return label_binarizers, count_vectorizer, tfid_transformer, feature_selection_transform, \
            model_name, model

    # Make predictions on dataset
    def make_predictions(self, model, X):

        try:
            predictions = model.predict(X)
        except ValueError as e:
            print("It seems like you need to regenerate your prediction model: " + str(e))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")
        try:
            probabilities = model.predict_proba(X)
            rates = np.amax(probabilities, axis=1)
        except Exception as e:
            if self.config.io["verbose"]:
                print("Probablity prediction not available for current model: " + \
                        str(e))
            probabilities = np.array([None])
            rates = np.array([None])
        return predictions, rates, probabilities

    # Evaluate predictions
    def evaluate_predictions(self, predictions, Y, message="Unknown"):
        print("\nEvaluation performed with evaluation data: " + message)
        print("\nAccuracy score for evaluation data: ", accuracy_score(Y, predictions))
        print("\nConfusion matrix for evaluation data: \n\n", confusion_matrix(Y, predictions))
        print("\nClassification matrix for evaluation data: \n\n", classification_report(Y, predictions, zero_division='warn'))

    # Function for finding the n most mispredicted data rows
    def most_mispredicted(self, X_original, model, ct_model, X_transformed, Y, n_limit):

        # Calculate predictions for both total model and cross trained model
        for what_model, the_model in [("model retrained on all data",model), ("modell cross trained on training data",ct_model)]:
            Y_pred = pandas.DataFrame(the_model.predict(X_transformed), index = Y.index)

            # Find the data rows where the real category is different from the predictions
            # Iterate over the indexes (they are now not in order)
            X_not = []
            for i in Y.index:
                X_not.append((Y.loc[i] != Y_pred.loc[i]).bool())

            # Quick end of loop if possible
            num_mispredicted = sum(elem == True for elem in X_not)
            if num_mispredicted != 0:
                break
 
        # Quick return if possible
        if num_mispredicted == 0:
            return "no model produced mispredictions", pandas.DataFrame()

        # Select the found data
        X_mispredicted = X_transformed.loc[X_not]

        # Predict probabilites
        try:
            Y_prob = the_model.predict_proba(X_mispredicted)
            could_predict_proba = True
        except Exception as ex:
            print("Could not predict probabilities: {0}".format(str(ex)))
            could_predict_proba = False

        #  Re-insert original data columns but drop the class column
        X_mispredicted = X_original.loc[X_not]
        X_mispredicted = X_mispredicted.drop(self.config.sql["class_column"], axis = 1)

        # Add other columns to mispredicted data
        X_mispredicted.insert(0, "Actual", Y.loc[X_not])
        X_mispredicted.insert(0, "Predicted", Y_pred.loc[X_not].values)
        
        # Add probabilities and sort only if they could be calculated above, otherwise
        # return a random sample of mispredicted
        if not could_predict_proba:
            for i in range(len(the_model.classes_)):
                X_mispredicted.insert(0, "P(" + the_model.classes_[i] + ")", "N/A")
            return what_model, X_mispredicted.sample(n=n_limit)
        else:
            Y_prob_max = np.amax(Y_prob, axis = 1)
            for i in reversed(range(Y_prob.shape[1])):
                X_mispredicted.insert(0, "P(" + the_model.classes_[i] + ")", Y_prob[:,i])
            X_mispredicted.insert(0, "__Sort__", Y_prob_max)

            # Sort the dataframe on the first column and remove it
            X_mispredicted = X_mispredicted.sort_values("__Sort__", ascending = False)
            X_mispredicted = X_mispredicted.drop("__Sort__", axis = 1)

            # Keep only the top n_limit rows and return
            return what_model, X_mispredicted.head(n_limit)

    # For saving results in database
    def save_data(self, keys, Y, rates, prob_mode = 'U', labels='N/A', probabilities='N/A', alg = "Unknown"):

        try:
            if self.config.io["verbose"]: print("Save to database...-please wait!")

            # Get a sql handler and connect to data database
            sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
                host = self.config.sql["host"], catalog = self.config.sql["class_catalog"], \
                trusted_connection = self.config.sql["trusted_connection"], \
                username = self.config.sql["class_username"], \
                password = self.config.sql["class_password"])
            sqlHelper.connect()

            # Loop through the data
            num_lines = 0
            for i in range(len(keys)):

                # Set together SQL code for the insert
                query = "INSERT INTO " + self.config.sql["class_catalog"] + "." 
                query +=  self.config.sql["class_table"] + " (catalog_name,table_name,column_names," 
                query +=  "unique_key,class_result,class_rate,class_rate_type,"
                query += "class_labels,class_probabilities,class_algorithm," 
                query +=  "class_script,class_user) VALUES(\'" + self.config.sql["data_catalog"] + "\'," 
                query +=  "\'" + self.config.sql["data_table"] + "\',\'" 
                if self.text_data:
                    query += self.config.sql["data_text_columns"]
                if self.text_data and self.numerical_data:
                    query += ","
                if self.numerical_data:
                    query +=  self.config.sql["data_numerical_columns"]  
                query += "\'," + str(keys[i]) + ",\'" + str(Y[i]) + "\'," + str(rates[i]) + "," 
                query += "\'" + prob_mode + "\'," + "\'" + ','.join(labels) + "\'," + "\'" + \
                    ','.join([str(elem) for elem in probabilities[i].tolist()]) + "\',"
                query += "\'" + alg + "\',\'" + self.scriptpath + "\',\'" + self.config.sql["data_username"] + "\')"

                # Execute a query without getting any data
                # Delay the commit until the connection is closed
                if sqlHelper.execute_query(query, get_data=False, commit=False):
                    num_lines += 1

            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=True)

            # Return the number of inserted rows
            return num_lines

        except Exception as e:
            print("Save of predictions: {0} failed: {1}".format(query,str(e)))
            if self.ProgressBar: self.ProgressBar.value = 1.0
            sys.exit("Program aborted.")

    # For saving results in database
    def correct_mispredicted_data(self, index, new_class):

        try:
            if self.config.io["verbose"]: print("Changing data row {0} to {1}".format(index, new_class))

            # Get a sql handler and connect to data database
            sqlHelper = sql.IAFSqlHelper(driver = self.config.sql["odbc_driver"], \
                host = self.config.sql["host"], catalog = self.config.sql["data_catalog"], \
                trusted_connection = self.config.sql["trusted_connection"], \
                username = self.config.sql["data_username"], \
                password = self.config.sql["data_password"])
            sqlHelper.connect()

            # Set together SQL code for the insert
            query =  "UPDATE " + self.config.sql["data_catalog"] + "." + self.config.sql["data_table"]
            query += " SET " + self.config.sql["class_column"] + " = \'" + str(new_class) + "\'"  
            query += " WHERE " + self.config.sql["id_column"] + " = " + str(index)
            
            # Execute a query without getting any data
            # Delay the commit until the connection is closed
            if sqlHelper.execute_query(query, get_data=False, commit=False):
                num_lines = 1
            else:
                num_lines = 0

            # Disconnect from database and commit all inserts
            sqlHelper.disconnect(commit=True)

            # Return the number of inserted rows
            return num_lines

        except Exception as e:
            print("Correction of mispredicted data: {0} failed: {1}".format(query,str(e)))
            return 0  
        
    # Produces an SQL command that can be executed to get a hold of the recently classified
    # data elements
    def get_sql_command_for_recently_classified_data(self, num_rows):
        
        selcols = ("A.[" + \
            "],A.[".join((self.config.sql["id_column"] + "," + \
            self.config.sql["data_numerical_columns"] + "," + \
            self.config.sql["data_text_columns"]).split(',')) + \
            "] ").replace("A.[]", "").replace(",,",",")
        
        query = \
            "SELECT TOP(" + str(num_rows) + ") " + selcols + \
            "B.[class_result], B.[class_rate],  B.[class_time], B.[class_algorithm] " + \
            "FROM [" + self.config.sql["data_catalog"].replace('.',"].[") + "].[" + \
            self.config.sql["data_table"].replace('.',"].[") + "] A " + \
            " INNER JOIN [" + self.config.sql["class_catalog"].replace('.',"].[") + "].[" + \
            self.config.sql["class_table"].replace('.',"].[") + "] B " + \
            "ON A." + self.config.sql["id_column"] + " = B.[unique_key] " + \
            "WHERE B.[class_user] = \'" + self.config.sql["class_username"] + "\' AND " + \
            "B.[table_name] = \'" + self.config.sql["data_table"] + "\' " + \
            "ORDER BY B.[class_time] DESC "
        return query
    
    # Make absolute file paths into relative
    @staticmethod
    def convert_absolut_path_to_relative(absolute_path):
        pwd = os.path.dirname(os.path.realpath(__file__))
        return absolute_path.replace(pwd, '.')

    # Find the number of data rows belonging to to smallest class in an array
    @staticmethod
    def find_smallest_class_number(Y):

        class_count = {}
        for elem in Y:
            if elem not in class_count:
                class_count[elem] = 1
            else:
                class_count[elem] += 1
        return max(1, min(class_count.values()));
    
    # The main classification function for the class
    def run(self):
        
         # Print a welcoming message for the audience
        if self.config.io["verbose"]:
            self._print_welcoming_message()
        
        # Create the classification table, if it does not exist already
        if self.ProgressLabel: self.ProgressLabel.value = "Create the classification table"
        self.create_classification_table()
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Set a flag in the classification database that execution has started
        self.mark_execution_started()

        # Read in all data, with classifications or not
        if self.ProgressLabel: self.ProgressLabel.value = "Read in data from database"
        dataset, unique_keys, data_query, self.unique_classes = self.read_in_data()
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Investigate dataset (optional)
        if self.config.mode["train"]:
            if self.ProgressLabel: self.ProgressLabel.value = "Investigate dataset (see console)"
            if self.config.io["verbose"]: self.investigate_dataset(dataset)

            # Give some statistical overview of the training material
            if self.config.io["verbose"]: self.show_statistics_on_dataset(dataset)
            if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # We need to know the number of data rows that are currently not
        # classified (assuming that they are last in the dataset because they
        # were sorted above). Separate keys from data elements that are
        # already classified from the rest.
        num_un_pred = self.get_num_unpredicted_rows(dataset)
        unpred_keys = unique_keys[-num_un_pred:]

        # Original training data are stored for later reference
        if num_un_pred > 0:
            X_original = dataset.head(-num_un_pred)
        else:
            X_original = dataset.copy(deep=True)

        # If we are making predictions, reload the necessary transforms and the model
        label_binarizers = {}
        count_vectorizer = None
        tfid_transformer = None
        feature_selection_transform = None
        if not self.config.mode["train"]:
            label_binarizers, count_vectorizer, tfid_transformer, feature_selection_transform, \
                trained_model_name, trained_model = self.load_model_from_file(self.model_filename)

        # Rearrange dataset such that all text columns are merged into one single
        # column, and convert text to numbers. Return the data in left-hand and
        # right hand side parts
        if self.ProgressLabel: self.ProgressLabel.value = "Rearrange dataset for possible textclassification, etc."
        X, Y, label_binarizers, count_vectorizer, tfid_transformer = \
            self.convert_textdata_to_numbers(dataset, label_binarizers, count_vectorizer, tfid_transformer)

        if self.text_data:
            if self.config.io["verbose"]: 
                print("After conversion of text data to numerical data:")
                self.investigate_dataset( X, False )
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # In case of PCA or Nystreom feature reduction, here it goes
        if self.use_feature_selection and self.config.mode["feature_selection"] != "RFE":
            t0 = time.time()
            X, self.config.mode["num_selected_features"], feature_selection_transform = \
                self.perform_feature_selection( X, self.config.mode["num_selected_features"], feature_selection_transform )

            t = time.time() - t0
            if self.config.io["verbose"]: print("Feature reduction took " + str(round(t,2)) + " sec.\n")

        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Split dataset for machine learning
        if self.config.mode["train"]:
            if self.ProgressLabel: self.ProgressLabel.value = "Split dataset for machine learning"
            X_train, X_validation, X_unknown, Y_train, Y_validation, Y_unknown = self.split_dataset(X, Y, num_un_pred)
        else:
            X_train = None
            X_validation = None
            Y_train = None
            Y_validation = None
            X_unknown = X
            Y_unknown = Y

        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Check algorithms for best model and train that model. K-value should be 10 or below.
        # Or just use the model previously trained.
        num_features = None
        if self.config.mode["train"]:
            if self.ProgressLabel: self.ProgressLabel.value = "Check and train algorithms for best model"
            k = min(10, self.find_smallest_class_number(Y_train))
            if k < 10:
                if self.config.io["verbose"]: print("Using non-standard k-value for spotcheck of algorithms: {0}".format(k))
            trained_model_name, trained_model, num_features = \
                self.spot_check_ml_algorithms(X_train, Y_train, k)
            trained_model = self.train_picked_model(trained_model, X_train, Y_train)
            self.save_model_to_file(label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, trained_model_name, trained_model, self.model_filename )
        elif self.config.mode["predict"]:
            num_features = trained_model.n_features_in_
        else:
            sys.exit("Aborting. User must choose either to train a new model or use old one for predictions!")
        if self.config.io["verbose"]: print("Best model is: {0} with number of features: {1}".format(trained_model_name, num_features))
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Make predictions on known testdata
        if self.config.mode["train"] and X_validation.shape[0] > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Make predictions on known testdata"
            pred, prob, probs = self.make_predictions(trained_model, X_validation)
            if not prob.any() == None :
                #print("Training Probabilities:",prob)
                print("Training Classification prob: Mean, Std.dev: ", \
                        np.mean(prob),np.std(prob))
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Evaluate predictions (optional)
        if self.config.mode["train"] and Y_validation.shape[0] > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Evaluate predictions"
            self.evaluate_predictions(pred, Y_validation, "ML algorithm: " + trained_model_name)
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Get accumulated classification score report for all predictions
        if self.config.mode["train"] and Y_validation.shape[0] > 0:
            class_report = classification_report(Y_validation, pred, output_dict = True)
            print("Classification report for {0} with #features: {1}".format(trained_model_name, num_features))
            for key in class_report.keys():
                print("{0}: {1}".format(key, class_report[key]))

        # RETRAIN best model on whole dataset: Xtrain + Xvalidation
        if self.config.mode["train"] and (X_train.shape[0] + X_validation.shape[0]) > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Retrain model on whole dataset"
            _label_binarizers, _count_vectorizer, _tfid_transformer, _feature_selection_transform, \
                _trained_model_name, cross_trained_model = self.load_model_from_file(self.model_filename)
            trained_model = \
                self.train_picked_model( trained_model, \
                    concat([X_train, X_validation], axis = 0), \
                    concat([Y_train, Y_validation], axis = 0) )
            self.save_model_to_file(label_binarizers, count_vectorizer, tfid_transformer, \
                feature_selection_transform, trained_model_name, trained_model, self.model_filename )
            what_model, self.X_most_mispredicted = self.most_mispredicted(X_original, trained_model, cross_trained_model, \
                concat([X_train, X_validation], axis = 0), \
                concat([Y_train, Y_validation], axis = 0), \
                self.LIMIT_MISPREDICTED)
            joiner = self.config.sql["id_column"] + " = \'"
            most_mispredicted_query = data_query + "WHERE " +  joiner \
                + ("\' OR " + joiner).join([str(number) for number in self.X_most_mispredicted.index.tolist()]) + "\'"
            if not self.X_most_mispredicted.empty and self.config.mode["mispredicted"]: 
                if self.config.io["verbose"]:
                    print("\n---Most mispredicted during training (using {0}):".format(what_model))
                    print(self.X_most_mispredicted.head(self.LIMIT_MISPREDICTED))
                    print("\nGet the most misplaced data by SQL query:\n {0}".format(most_mispredicted_query))
                    print("\nOr open the following csv-data file: \n\t {0}".format(self.misplaced_filepath))
                self.X_most_mispredicted.to_csv(path_or_buf = self.misplaced_filepath, sep = ';', na_rep='N/A', \
                                           float_format=None, columns=None, header=True, index=True, \
                                           index_label=self.config.sql["id_column"], mode='w', encoding='utf-8', \
                                           compression='infer', quoting=None, quotechar='"', line_terminator=None, \
                                           chunksize=None, date_format=None, doublequote=True, decimal=',', \
                                           errors='strict')
        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        # Now make predictions on non-classified dataset: X_unknown -> Y_unknown
        if self.config.mode["predict"] and X_unknown.shape[0] > 0:
            if self.ProgressLabel: self.ProgressLabel.value = "Make predictions on un-classified dataset"
            Y_unknown, Y_prob, Y_probs = self.make_predictions(trained_model, X_unknown)

            print("\n---Predictions for the unknown data---\n")
            if self.config.io["verbose"]: print("Predictions:", Y_unknown)
            if not Y_prob.any() == None :
                #print("Probabilities:",Y_prob)
                print("Classification Y_prob: Mean, Std.dev: ", np.mean(Y_prob), np.std(Y_prob))

            rate_type = 'I'
            if Y_prob.any() == None:
                Y_prob = []
                if self.config.mode["train"]:
                    rate_type = 'A'
                    for i in range(len(Y_unknown)):
                        try:
                            Y_prob = Y_prob + [class_report[Y_unknown[i]]['precision']]
                        except KeyError as e:
                            print("Warning: probability collection failed for key {0} with error {1}".format(Y_unknown[i]),str(e))
                else:
                    rate_type = 'U'
                    for i in range(len(Y_unknown)):
                        Y_prob = Y_prob + [0.0]
                if self.config.io["verbose"]: print("Probabilities:",Y_prob)
            if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

            # Save new classifications for X_unknown in classification database
            if self.ProgressLabel: self.ProgressLabel.value = "Save new classifications in database"
            results_saved = self.save_data( unpred_keys.values, Y_unknown, Y_prob, rate_type, \
               trained_model.classes_, Y_probs, trained_model_name ) 
            print("Added {0} rows to classification table. Get them with SQL query:\n\n{1}".
                  format(results_saved,self.get_sql_command_for_recently_classified_data(results_saved)))

        if self.ProgressBar: self.ProgressBar.value += self.percentPermajorTask

        elapsed_time = time.time() - self.clock1
        date_again = str(datetime.now())
        if self.config.io["verbose"]: print("\n--- Ending program after " + str(timedelta(seconds=round(elapsed_time))) + " at " + str(date_again) + " ---\n")

        # Remove flag in database, signaling all was allright
        self.mark_execution_ended()

        # Make sure progressbar is completed if not before
        if self.ProgressLabel: self.ProgressLabel.value = "Process finished"
        if self.ProgressBar: self.ProgressBar.value = 1.0

        # Close redirect of standard output in case of self.config.debug["debug_on"]ging
        if self.config.debug["debug_on"] and self.config.io["redirect_output"]:
            sys.stdout.close()
            sys.stderr.close()
    
    # Export configuration file for specified settings
    def export_configuration_to_file(self):
        
        pwd = os.path.dirname(os.path.realpath(__file__))
        config_path = Path(pwd) / self.CONFIG_FILENAME_PATH
        fin = open(config_path / self.CONFIG_SAMPLE_FILE, "r")
        lines = fin.readlines()
        fin.close()

        for tag in self.CONFIGURATION_TAGS:
            for i in range(len(lines)):
                try:
                    lines[i] = lines[i].replace(tag, str(self.config.project[tag[1:-1]]))
                except KeyError:
                    try:
                        lines[i] = lines[i].replace(tag, str(self.config.sql[tag[1:-1]]))
                    except KeyError:
                        try:
                            lines[i] = lines[i].replace(tag, str(self.config.mode[tag[1:-1]]))
                        except KeyError:
                            try:
                                lines[i] = lines[i].replace(tag, str(self.config.io[tag[1:-1]]))
                            except KeyError:
                                try:
                                    lines[i] = lines[i].replace(tag, str(self.config.debug[tag[1:-1]]))
                                except Exception as ex:
                                    write("Could not write configuration tag {0} to file: {1}". \
                                         format(tag, str(ex)))
        
        fout_name = self.CONFIG_FILENAME_START + self.config.project["name"] + "_" + \
                    self.config.sql["data_username"] + ".py"
        fout = open(config_path / fout_name, "w")
        fout.writelines(lines)
        fout.close()
    
    # Some welcoming message to the audience
    def _print_welcoming_message(self):
        
        # Print welcoming message
        if self.config.io["verbose"]: print("\n *** WELCOME TO IAF AUTOMATIC CLASSIFICATION SCRIPT ***\n")
        if self.config.io["verbose"]: print("Execution started at: {0:>30s} \n".format(str(self.date_now)))

        # Print configuration settings
        if self.config.io["verbose"]: print(" -- Configuration settings --")
        if self.config.io["verbose"]: print(" 1. Database settings ")
        if self.config.io["verbose"]: print(" * ODBC driver (when applicable):           " + self.config.sql["odbc_driver"])
        if self.config.io["verbose"]: print(" * Classification Host:                     " + self.config.sql["host"])
        if self.config.io["verbose"]: print(" * Trusted connection:                      " + str(self.config.sql["trusted_connection"]))
        if self.config.io["verbose"]: print(" * Classification Table:                    " + self.config.sql["class_catalog"])
        if self.config.io["verbose"]: print(" * Classification Table:                    " + self.config.sql["class_table"])
        if self.config.io["verbose"]: print(" * Classification Table creation script:    " + self.config.sql["class_table_script"])
        if self.config.io["verbose"]: print(" * Classification Db username (optional):   " + self.config.sql["class_username"])
        if self.config.io["verbose"]: print(" * Classification Db password (optional)    " + self.config.sql["class_password"])
        if self.config.io["verbose"]: print("");
        if self.config.io["verbose"]: print(" * Data Catalog:                            " + self.config.sql["data_catalog"])
        if self.config.io["verbose"]: print(" * Data Table:                              " + self.config.sql["data_table"])
        if self.config.io["verbose"]: print(" * Classification column:                   " + self.config.sql["class_column"])
        if self.config.io["verbose"]: print(" * Text Data columns (CSV):                 " + self.config.sql["data_text_columns"])
        if self.config.io["verbose"]: print(" * Numerical Data columns (CSV):            " + self.config.sql["data_numerical_columns"])
        if self.config.io["verbose"]: print(" * Unique data id column:                   " + self.config.sql["id_column"])
        if self.config.io["verbose"]: print(" * Data username (optional):                " + self.config.sql["data_username"])
        if self.config.io["verbose"]: print(" * Data password (optional):                " + self.config.sql["data_password"])
        if self.config.io["verbose"]: print("")
        if self.config.io["verbose"]: print(" 2. Classification mode settings ")
        if self.config.io["verbose"]: print(" * Train new model:                         " + str(self.config.mode["train"]))
        if self.config.io["verbose"]: print(" * Make predictions with model:             " + str(self.config.mode["predict"]))
        if self.config.io["verbose"]: print(" * Display mispredicted training data:      " + str(self.config.mode["mispredicted"]))
        if self.config.io["verbose"]: print(" * Use stop words:                          " + str(self.config.mode["use_stop_words"]))
        if self.config.io["verbose"]: print(" * Material specific stop words threshold:  " + str(self.config.mode["specific_stop_words_threshold"]))
        if self.config.io["verbose"]: print(" * Hex encode text data:                    " + str(self.config.mode["hex_encode"]))
        if self.config.io["verbose"]: print(" * Categorize text data where applicable:   " + str(self.config.mode["use_categorization"]))
        if self.config.io["verbose"]: print(" * Test size for trainings:                 " + str(self.config.mode["test_size"]))
        if self.config.io["verbose"]: print(" * Use SMOTE:                               " + str(self.config.mode["smote"]))
        if self.config.io["verbose"]: print(" * Use undersampling of majority class:     " + str(self.config.mode["undersample"]))
        if self.config.io["verbose"]: print(" * Algorithm of choice:                     " + self.config.mode["algorithm"])
        if self.config.io["verbose"]: print(" * Preprocessing method of choice:          " + self.config.mode["preprocessor"])
        if self.config.io["verbose"]: print(" * Scoring method:                          " + self.config.mode["scoring"])
        if self.config.io["verbose"]: print(" * Feature selection:                       " + self.config.mode["feature_selection"])
        if self.config.io["verbose"]: print(" * Number of selected features:             " + str(self.config.mode["num_selected_features"]))
        if self.config.io["verbose"]: print(" * Maximum iterations (where applicable):   " + str(self.config.mode["max_iterations"]))
        if self.config.io["verbose"]: print("")
        if self.config.io["verbose"]: print(" 3. I/O specifications ")
        if self.config.io["verbose"]: print(" * Verbosity:                               " + str(self.config.io["verbose"]))
        if self.config.io["verbose"]: print(" * Path where to save generated model:      " + self.config.io["model_path"])
        if self.config.io["verbose"]: print(" * Name of generated or loaded model:       " + self.config.io["model_name"])
        if self.config.io["verbose"]: print("")
        if self.config.io["verbose"]: print(" 4. Debug settings ")
        if self.config.io["verbose"]: print(" * Debugging on:                            " + str(self.config.debug["debug_on"]))
        if self.config.io["verbose"]: print(" * How many data rows to consider:          " + str(self.config.debug["num_rows"]))

        # Print out what mode we use: training a new model or not
        if self.config.mode["train"]:
            if self.config.io["verbose"]: print(" -- We will train our ml model --")
        else:
            try:
                f = open(self.model_filename,'br')
            except IOError:
                sys.exit("Aborting. No trained model exists at {0}".format(self.model_filename))
            else:
                if self.config.io["verbose"]: print(" -- We will reload and use old model --")

# In case the user has specified some input arguments to command line call
def check_input_arguments(argv):

    command_line_instructions = \
    "Usage: " + argv[0] + ' [-h/--help] [-f/--file <configfilename>]'

    try:
        short_options = "hf:"
        long_options = ["help", "file"]
        opts, args = getopt.getopt(argv[1:], short_options, long_options)
    except getopt.GetoptError:
        print(command_line_instructions)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h' or opt == '--help':
            print(command_line_instructions)
            sys.exit()
        elif opt == '-f' or opt == '--file':
            if arg.count("..") > 0:
                print("Configuration file must be in a subfolder to {0}".format(argv[0]));
                sys.exit();
            print("Importing specified configuration file:",arg)
            if not arg[0] == '.':
                arg = IAFautomaticClassifier.convert_absolut_path_to_relative(arg)
            file = arg.split('\\')[-1]
            filename = file.split('.')[0]
            filepath = '\\'.join(arg.split('\\')[:-1])
            paths = arg.split('\\')[:-1]
            try:
                paths.pop(paths.index('.'))
            except Exception as e:
                print("Filepath {0} does not seem to be relative (even after conversion)".format(filepath))
                sys.exit()
            pack = '.'.join(paths)
            sys.path.insert(0, filepath)
            try:
                config = importlib.import_module(pack+"."+filename)
                return config
            except Exception as e:
                print("Filename {0} and pack {1} could not be imported dynamically".format(filename,pack))
                sys.exit(str(e))
        else:
            print("Illegal argument to " + argv[0] + "!")
            print(command_line_instructions)
            sys.exit()
            
# Main program
def main(argv):

     # Handle the case when the user has specified input arguments
    if len(argv) > 1:
        config = check_input_arguments(argv)
    else:
        config = None

    # Use the loaded configuration module argument
    # or create a classifier object with only standard settings
    myClassiphyer = IAFautomaticClassifier(config=config)
    
    # Run the classifier
    myClassiphyer.run()

# Start main
if __name__ == "__main__":
    main(sys.argv)
