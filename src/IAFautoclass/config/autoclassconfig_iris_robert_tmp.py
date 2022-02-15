#!/usr/bin/env python
import datetime

# Project information, will be used to distinguish data and generated model
# from other
project = {
    
    # Name of the project
    "name": "iris",

    }

# Database configuration
sql = {

    # If applicable in a Windows enviroment, use this database driver
    "odbc_driver": "{ODBC Driver 17 for SQL Server}",

    # Database host
    "host": "tcp:sql-stat.iaf.local",   
    
    # Trusted connection or not
    "trusted_connection": "false",

    # Database catalog for results of classification
    "class_catalog": "Arbetsdatabas",         

    # Database table for results for classification
    "class_table": "aterkommande_automat.AutoKlassificering",

    # Creation script for classification table results
    "class_table_script": "./sql/autoClassCreateTable.sql",

    # User information for classification
    # Notice: ignored if trusted_connection = "true".
    "class_username": "robert_tmp",
    "class_password": "robert",

    # Database catalog for data to be classified
    "data_catalog": "Arbetsdatabas",

    # Database table for data to be classified
    "data_table": "aterkommande_automat.iris",

    # The columns in data table to be classified, in CSV-style.
    # First the classification column:
    "class_column": "class",

    # If the class column should be treated as a hierarchical class
    # this setting should be "true", otherwise false
    "hierarchical_class": "false",

    # Secondly, the data columns containing TEXT data, in CSV-style 
    # Set to "" (empty string) if none.
    "data_text_columns": "",

    # Thirdly, the data columns containing numerical data, in CSV-style
    # Set to "" (empty string) if none.
    "data_numerical_columns": "petal-length,petal-width,sepal-length,sepal-width",

    # The unique key index in the data table by which each row is
    # uniquely identied
    "id_column": "id",

    # User information for data table access.
    # Notice: ignored if trusted_connection = "true"
    "data_username": "robert_tmp",
    "data_password": "robert",
}

# Specified modes of operation
mode = {

    # The classifier can train, predict and display mispredicted data
    "train": "true",
    "predict": "true",
    "mispredicted": "true",

    # Ignore standard stop words in classification and,
    # possibly, material specific stop words (see below)
    "use_stop_words": "true",

     # Lower percentage limit for specific stop words retrieval. 
     # Words which have a document frequency higher than
     # this threshold will be considered as stop words.
     # Set to 1.0 to not consider any specific stop words
    "specific_stop_words_threshold": "1",

    # Use hex encoding on text data before classification.
    "hex_encode": "true",

     # Use hex encoding on text data before classification.
    "use_categorization": "true",

    # How large part, between 0 and 1, of already classified data
    # should be used for classification tests (the rest will be
    # used for prediction)
    "test_size": "0.2",

    # Use Synthetic Minority Oversampling Technique
    "smote": "false",

    # Use Random Undersampling of majority class
    "undersample": "false",

    # What algorithm to use
    "algorithm": "LDA",

    # What pre-processing method to use
    "preprocessor": "NON",

    # If to use feature_selection of not
    "feature_selection": "NON",

    # If feature selection is on, and we want to specify the 
    # numbers of features to use
    "num_selected_features": "",

    # How to score algoithms in cross-score evaluation,
    "scoring": "accuracy",

    # For iterative algorithms where we can specify the maximum
    # number of iterations, we do this here
    "max_iterations": "20000",
}

# Specifies how to direct output, where to save model, etc
io = {

    # Output level
    "verbose": "true",

    # Where to save/load trained and generated model
    "model_path": "./model/"
}

# Some debug settings
debug = {

    # Debugging on/off
    "debug_on": "true",
    
    # Number of records to consider in data table
    "num_rows": "150",    
    
}