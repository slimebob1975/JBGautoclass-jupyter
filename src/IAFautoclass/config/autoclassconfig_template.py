#!/usr/bin/env python

# Name of the project
name = "<name>"

version = "2.0"


# Database configuration
connection = {

    # If applicable in a Windows enviroment, use this database driver
    "odbc_driver": "<odbc_driver>",

    # Database host
    "host": "<host>", 
    
    # Trusted connection or not
    "trusted_connection": <trusted_connection>,

    # Database catalog for results of classification
    "class_catalog": "<class_catalog>",         

    # Database table for results for classification
    "class_table": "<class_table>",

    # Creation script for classification table results
    "class_table_script": "<class_table_script>",

    # User information for classification
    # Notice: ignored if trusted_connection = "true".
    "class_username": "<class_username>",
    "class_password": "<class_password>",

    # Database catalog for data to be classified
    "data_catalog": "<data_catalog>",

    # Database table for data to be classified
    "data_table": "<data_table>",

    # The columns in data table to be classified, in CSV-style.
    # First the classification column:
    "class_column": "<class_column>",

    # Secondly, the data columns containing TEXT data, in CSV-style 
    # Set to "" (empty string) if none.
    "data_text_columns": "<data_text_columns>",

    # Thirdly, the data columns containing numerical data, in CSV-style
    # Set to "" (empty string) if none.
    "data_numerical_columns": "<data_numerical_columns>",

    # The unique key index in the data table by which each row is
    # uniquely identied
    "id_column": "<id_column>",

    # User information for data table access.
    # Notice: ignored if trusted_connection = "true"
    "data_username": "<data_username>",
    "data_password": "<data_password>",
}

# Specified modes of operation
mode = {

    # The classifier can train, predict and display mispredicted data
    "train": <train>,
    "predict": <predict>,
    "mispredicted": <mispredicted>,

    # Ignore standard stop words in classification and,
    # possibly, material specific stop words (see below)
    "use_stop_words": <use_stop_words>,

     # Lower percentage limit for specific stop words retrieval. 
     # Words which have a document frequency higher than
     # this threshold will be considered as stop words.
     # Set to 1.0 to not consider any specific stop words
    "specific_stop_words_threshold": "<specific_stop_words_threshold>",

    # Use hex encoding on text data before classification.
    "hex_encode": <hex_encode>,

     # Use automatic categorization on text data before classification.
    "use_categorization": <use_categorization>,
    
    # Force the following columns to be categorized even though they are not automatically categorized
    "category_text_columns": "<category_text_columns>",

    # How large part, between 0 and 1, of already classified data
    # should be used for classification tests (the rest will be
    # used for prediction)
    "test_size": "<test_size>",

    # Use Synthetic Minority Oversampling Technique
    "smote": <smote>,

    # Use Random Undersampling of majority class
    "undersample": <undersample>,

    # What algorithm to use
    "algorithm": "<algorithm>",

    # What pre-processing method to use
    "preprocessor": "<preprocessor>",

    # If to use feature_selection of not
    "feature_selection": "<feature_selection>",

    # If feature selection is on, and we want to specify the 
    # numbers of features to use
    "num_selected_features": "<num_selected_features>",

    # How to score algoithms in cross-score evaluation,
    "scoring": "<scoring>",

    # For iterative algorithms where we can specify the maximum
    # number of iterations, we do this here
    "max_iterations": "<max_iterations>",
}

# Specifies how to direct output, where to save model, etc
io = {

    # Output level
    "verbose": <verbose>,

    # Where to save/load trained and generated model
    "model_path": "<model_path>",
    
    # Name of previously trained model
    "model_name": "<model_name>",
}

# Some debug settings
debug = {

    # Debugging on/off
    "debug_on": <debug_on>,
    
    # Number of records to consider in data table
    "num_rows": "<num_rows>",    
    
}