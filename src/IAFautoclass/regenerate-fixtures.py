import os
import pickle
from Config import Config, Algorithm, Preprocess, Reduction, Scoretype

""" This is a script to quick-and-dirty regenerate some fixtures that might be annoying otherwise

    Each generator is a function, and can be commented out/in in main()

    Current generators:
    - config-save.sav: Creates a model file with just a config file. Enough for test_config, but not for test_handler
    - model-save.sav: Creates a model file with default (IE None). Enough to test basics in test_handler
"""

def bare_iris_config() -> Config:
    """ This is the bare_iris_config from test_config, so if that one has changed, this one should too """
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.iaf.mock",
            trusted_connection=True,
            class_catalog="DatabaseOne",
            class_table="ResultTable",
            class_table_script="createtable.sql.txt",
            class_username="some_fake_name",
            class_password="",
            data_catalog="DatabaseTwo",
            data_table="InputTable",
            class_column="class",
            data_text_columns=[],
            data_numerical_columns=["sepal-length","sepal-width","petal-length","petal-width"],
            id_column="id",
            data_username="some_fake_name",
            data_password=""
        ),
        Config.Mode(
            train=False,
            predict=True,
            mispredicted=False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            smote=False,
            undersample=False,
            algorithm=Algorithm.LDA,
            preprocessor=Preprocess.STA,
            feature_selection=Reduction.PCA,
            num_selected_features=None,
            scoring=Scoretype.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            num_rows=150
        ),
        name="iris",
        filename="autoclassconfig_iris_.py"
    )

    config.connection.data_catalog = ""
    config.connection.data_table = ""
    config.mode.train = None
    config.mode.predict = None
    config.mode.mispredicted = None
    config.io.model_name = ""
    config.debug.num_rows = 0
    

    return config

def save_model_to_file(filename, config, model = None):
    """ This is a copy of save_model_to_file in IAFHandler, update if necessary"""
    try:
        save_config = config.get_clean_config()
        if model is None:
            data = [
                save_config
            ]
        else:
            data = [
                save_config,
                {},
                None,
                None,
                None,
                (None, None),
                None
            ]
        
        pickle.dump(data, open(filename,'wb'))

    except Exception as e:
        print(f"Something went wrong on saving model to file: {e}")

def regenerate_model_save():
    config = bare_iris_config()
    
    srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    path = os.path.join(srcdir, "tests", "fixtures") # This is where all the fixtures should be saved
    
    modelSaveName = os.path.join(path, "model-save.sav")

    save_model_to_file(modelSaveName, config, "")

def regenerate_config_save():
    config = bare_iris_config()
    
    srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    path = os.path.join(srcdir, "tests", "fixtures") # This is where all the fixtures should be saved
    
    configSaveName = os.path.join(path, "config-save.sav")

    save_model_to_file(configSaveName, config)

def main():
    """ Regenerates config-save.sav """
    #regenerate_config_save()

    """ Regenerates model-save.sav """
    #regenerate_model_save()
    


if __name__ == "__main__":
    main()