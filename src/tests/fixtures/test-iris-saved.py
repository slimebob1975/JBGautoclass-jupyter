#!/usr/bin/env python

# Name of the project
name = "test"

version = "2.0"


# Database configuration
connection = {

    # If applicable in a Windows enviroment, use this database driver
    "odbc_driver": "Mock Server",

    # Database host
    "host": "tcp:database.iaf.mock", 
    
    # Trusted connection or not
    "trusted_connection": True,

    # Database catalog for results of classification
    "class_catalog": "DatabaseOne",         

    # Database table for results for classification
    "class_table": "ResultTable",

    # Creation script for classification table results
    "class_table_script": "createtable.sql.txt",

    # User information for classification
    # Notice: ignored if trusted_connection = "true".
    "class_username": "some_fake_name",
    "class_password": "",

    # Database catalog for data to be classified
    "data_catalog": "DatabaseTwo",

    # Database table for data to be classified
    "data_table": "InputTable",

    # The columns in data table to be classified, in CSV-style.
    # First the classification column:
    "class_column": "class",

    # Secondly, the data columns containing TEXT data, in CSV-style 
    # Set to "" (empty string) if none.
    "data_text_columns": "",

    # Thirdly, the data columns containing numerical data, in CSV-style
    # Set to "" (empty string) if none.
    "data_numerical_columns": "sepal-length,sepal-width,petal-length,petal-width",

    # The unique key index in the data table by which each row is
    # uniquely identied
    "id_column": "id",

    # User information for data table access.
    # Notice: ignored if trusted_connection = "true"
    "data_username": "some_fake_name",
    "data_password": "",
}

# Specified modes of operation
mode = {

    # The classifier can train, predict and display mispredicted data
    "train": False,
    "predict": True,
    "mispredicted": False,

    # Ignore standard stop words in classification and,
    # possibly, material specific stop words (see below)
    "use_stop_words": False,

     # Lower percentage limit for specific stop words retrieval. 
     # Words which have a document frequency higher than
     # this threshold will be considered as stop words.
     # Set to 1.0 to not consider any specific stop words
    "specific_stop_words_threshold": "1.0",

    # Use hex encoding on text data before classification.
    "hex_encode": True,

     # Use automatic categorization on text data before classification.
    "use_categorization": True,
    
    # Force the following columns to be categorized even though they are not automatically categorized
    "category_text_columns": "",

    # How large part, between 0 and 1, of already classified data
    # should be used for classification tests (the rest will be
    # used for prediction)
    "test_size": "0.2",

    # Use Synthetic Minority Oversampling Technique
    "smote": False,

    # Use Random Undersampling of majority class
    "undersample": False,

    # What algorithm to use
    "algorithm": "LDA",

    # What pre-processing method to use
    "preprocessor": "STA",

    # If to use feature_selection of not
    "feature_selection": "PCA",

    # If feature selection is on, and we want to specify the 
    # numbers of features to use
    "num_selected_features": "None",

    # How to score algoithms in cross-score evaluation,
    "scoring": "accuracy",

    # For iterative algorithms where we can specify the maximum
    # number of iterations, we do this here
    "max_iterations": "20000",
}

# Specifies how to direct output, where to save model, etc
io = {

    # Output level
    "verbose": True,

    # Where to save/load trained and generated model
    "model_path": "./model",
    
    # Name of previously trained model
    "model_name": "test",
}

# Some debug settings
debug = {

    # Debugging on/off
    "on": True,
    
    # Number of records to consider in data table
    "num_rows": "150",    
    
}