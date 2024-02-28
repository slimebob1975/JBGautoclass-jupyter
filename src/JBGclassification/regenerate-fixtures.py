import os
import dill
from imblearn.pipeline import Pipeline
from Config import Config
from JBGMeta import (Algorithm, Preprocess, Reduction, ScoreMetric, 
                     AlgorithmTuple, PreprocessTuple, ReductionTuple,
                    Oversampling, Undersampling)


""" This is a script to quick-and-dirty regenerate some fixtures that might be annoying otherwise

    Each generator is a function, with a dictionary that sets whether it should be ran or not.
    Changed it from the original ("comment in/out what you need") as I didn't realise it wasn't regenerating.

    Current generators:
    - config-save.sav: Creates a model file with just a config file. Enough for test_config, but not for test_handler
    - model-save.sav: Creates a model file with default (IE None). Enough to test basics in test_handler
"""

def bare_iris_config() -> Config:
    """ This is the bare_iris_config from conftest.py, so if that one has changed, this one should too """
    config = Config(
        Config.Connection(
            odbc_driver="Mock Server",
            host="tcp:database.jbg.mock",
            trusted_connection=True,
            class_catalog="DatabaseOne",
            class_table="ResultTable",
            sql_username="some_fake_name",
            sql_password="",
            data_catalog="DatabaseTwo",
            data_table="InputTable",
            class_column="class",
            data_text_columns=[],
            data_numerical_columns=["sepal-length","sepal-width","petal-length","petal-width"],
            id_column="id",
        ),
        Config.Mode(
            train=False,
            predict=True,
            mispredicted=False,
            use_metas= False,
            use_stop_words=False,
            specific_stop_words_threshold=1.0,
            hex_encode=True,
            use_categorization=True,
            category_text_columns=[],
            test_size=0.2,
            oversampler=Oversampling.NOG,
            undersampler=Undersampling.NUG,
            algorithm=AlgorithmTuple([Algorithm.LDA]),
            preprocessor=PreprocessTuple([Preprocess.NOS]),
            feature_selection=ReductionTuple([Reduction.NOR]),
            num_selected_features=None,
            scoring=ScoreMetric.accuracy,
            max_iterations=20000
        ),
        Config.IO(
            verbose=True,
            model_path="./model",
            model_name="test"
        ),
        Config.Debug(
            on=True,
            data_limit=150
        ),
        name="iris",
        _filename="autoclassconfig_iris_.py"
    )

    config.connection.data_catalog = ""
    config.connection.data_table = ""
    config.mode.train = None
    config.mode.predict = None
    config.mode.mispredicted = None
    config.io.model_name = ""
    config.debug.data_limit = 0
    

    return config

def save_model_to_file(filename, config):
    """ Based on ModelHandler.save_model_to_file, update as needed """
    preprocess = Preprocess.NOS
    reduction = Reduction.NOR
    algorithm = Algorithm.DUMY
    oversampler = Oversampling.NOG
    undersampler = Undersampling.NUG
    try:
        save_config = config.get_clean_config()
        data = {
            "config": save_config,
            "text_converter": None,
            "pipeline names": (preprocess, reduction, algorithm),
            "pipeline": None,
            "n_features_out": 4
        }
        
        dill.dump(list(data.values()), open(filename,'wb'))
    except Exception as e:
        print(f"Something went wrong on saving model to file: {e}")

def regenerate_model_save():
    config = bare_iris_config()
    
    srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    path = os.path.join(srcdir, "tests", "fixtures") # This is where all the fixtures should be saved
    
    modelSaveName = os.path.join(path, "model-save.sav")

    save_model_to_file(modelSaveName, config)

def regenerate_config_save():
    config = bare_iris_config()
    
    srcdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    path = os.path.join(srcdir, "tests", "fixtures") # This is where all the fixtures should be saved
    
    configSaveName = os.path.join(path, "config-save.sav")

    save_model_to_file(configSaveName, config)

def main():
    options = {
        "config": False,
        "model": False
    }

    for type, run in options.items():
        if run:
            print(f"Regenerated {type}-save.sav")
            if type == "config":
                regenerate_config_save()
            elif type == "model":
                regenerate_model_save()
        else:
            print(f"Skipped {type}-save.sav")
    
    


if __name__ == "__main__":
    main()