from __future__ import annotations

import copy
import enum
import os
import dill
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Type, TypeVar, Union

import Helpers
from JBGExceptions import ConfigException, ODBCDriverException
from JBGMeta import (Algorithm, AlgorithmTuple, Preprocess, PreprocessTuple,
                     Reduction, ReductionTuple, ScoreMetric, MetaTuple, Oversampling,
                     Undersampling)


T = TypeVar('T', bound='Config')
ODBC_drivers = ["SQL Server", "ODBC Driver 13 for SQL Server", "ODBC Driver 17 for SQL Server"]


@dataclass
class Config:
    
    MAX_ITERATIONS = 20000
    CONFIG_FILENAME_START = "autoclassconfig_"
    CONFIG_SAMPLE_FILE = CONFIG_FILENAME_START + "template.py.txt"

    CLASS_TABLE_SCRIPT = "./sql/CreatePredictionTables.sql.txt"

    DEFAULT_MODELS_PATH =  ".\\model\\"
    DEFAULT_MODEL_EXTENSION = ".sav"
    DEFAULT_TRAIN_OPTION = "Train a new model"

    TEXT_DATATYPES = ["nvarchar", "varchar", "char", "text", "enum", "set"]
    INT_DATATYPES = ["bigint", "int", "smallint", "tinyint"]

    TEMPLATE_TAGS = {
        "name": "<name>",
        "connection.odbc_driver": "<odbc_driver>",
        "connection.host": "<host>",
        "connection.trusted_connection": "<trusted_connection>",
        "connection.class_catalog": "<class_catalog>", 
        "connection.class_table": "<class_table>",
        "connection.sql_username": "<sql_username>",
        "connection.sql_password": "<sql_password>",
        "connection.data_catalog": "<data_catalog>",
        "connection.data_table": "<data_table>",
        "connection.class_column": "<class_column>",
        "connection.data_text_columns": "<data_text_columns>",
        "connection.data_numerical_columns": "<data_numerical_columns>",
        "connection.id_column": "<id_column>",
        "mode.train": "<train>",
        "mode.predict": "<predict>",
        "mode.mispredicted": "<mispredicted>",
        "mode.use_metas": "<use_metas>",
        "mode.use_stop_words": "<use_stop_words>",
        "mode.specific_stop_words_threshold": "<specific_stop_words_threshold>",
        "mode.hex_encode": "<hex_encode>",
        "mode.use_categorization": "<use_categorization>",
        "mode.category_text_columns": "<category_text_columns>",
        "mode.test_size": "<test_size>",
        "mode.oversampler": "<oversampler>",
        "mode.undersampler": "<undersampler>",
        "mode.algorithm": "<algorithm>",
        "mode.preprocessor": "<preprocessor>",
        "mode.feature_selection": "<feature_selection>",
        "mode.num_selected_features": "<num_selected_features>",
        "mode.scoring": "<scoring>",
        "mode.max_iterations": "<max_iterations>",
        "io.verbose": "<verbose>",
        "io.model_path": "<model_path>",
        "io.model_name": "<model_name>",
        "debug.on": "<on>",
        "debug.data_limit": "<data_limit>"
    }

    @dataclass
    class Connection:
        odbc_driver: str = "SQL Server"
        host: str = ""
        trusted_connection: bool = False
        class_catalog: str = ""
        class_table: str = ""
        sql_username: str = ""
        sql_password: str = ""
        data_catalog: str = ""
        data_table: str = ""
        class_column: str = ""
        data_text_columns: list = field(default_factory=list)
        data_numerical_columns: list = field(default_factory=list)
        id_column: str = "id"

        def validate(self) -> None:
            """ Throws TypeError if invalid """
            # database connection information
            database_connection_information = [
                isinstance(self.host, str),
                isinstance(self.class_catalog, str),
                isinstance(self.class_table, str),
                isinstance(self.sql_username, str),
                isinstance(self.sql_password, str),
                isinstance(self.data_catalog, str),
                isinstance(self.data_table, str),
                isinstance(self.class_column, str),
                isinstance(self.data_text_columns, list),
                isinstance(self.data_numerical_columns, list),
                isinstance(self.id_column, str),
                isinstance(self.trusted_connection, bool)
            ]
            
            if not all(database_connection_information):
                raise TypeError(
                    "Specified database connection information is invalid")

            if not all(isinstance(x,str) for x in self.data_text_columns):
                raise TypeError("Data text columns needs to be a list of strings")
            if not all(isinstance(x,str) for x in self.data_numerical_columns):
                raise TypeError("Data numerical columns needs to be a list of strings")

            
            # Login credentials
            login_credentials = [
                isinstance(self.sql_username, str),
                isinstance(self.sql_password, str)
            ]

            if not all(login_credentials):
                raise TypeError("Specified login credentials are invalid!")

            # Overriding values
            if self.trusted_connection:
                self.sql_password = ""

                username = Config.get_username()
                self.sql_username = username

        def update_columns(self, updated_columns: dict) -> dict:
            """ Given the dictionary updated_columns set attributes to values """
            errors = []
            for attribute, value in updated_columns.items():
                if hasattr(self, attribute):
                    setattr(self, attribute, value)
                else:
                    errors.append(attribute)
            
            if errors:
                return {
                    "response": f"Following columns do not exist: {', '.join(errors)}",
                    "success": False
                }

            return {
                "response": "Columns updated",
                "success": True
            }

        def driver_is_implemented(self) -> bool:
            """ Returns whether the driver of the config is implemented """
            # TODO: Probably want to make this sturdier, but that would require
            # rewriting the dependancy on odbc_driver and the str
            if self.odbc_driver in ODBC_drivers:
                return True
            
            return False

        def _get_formatted_catalog(self, type) -> str:
            """ Gets a class table formatted for the type, which is based on odbc_driver
            """
            if not self.driver_is_implemented():
                return ""

            catalog_attr = type + "_catalog"
            catalog = getattr(self, catalog_attr)
            return f"[{ period_to_brackets(catalog) }]"

        def get_formatted_class_catalog(self) -> str:
            """ Gets a class table formatted for the type, which is based on odbc_driver
            """
            return self._get_formatted_catalog("class")

        def get_formatted_data_catalog(self) -> str:
            """ Gets a data table formatted for the type, which is based on odbc_driver
            """
            return self._get_formatted_catalog("data")

        def _get_formatted_table(self, type: str, include_database: bool = True, before: str = None, after: str = None) -> str:
            """ Gets a class table formatted for the type, which is based on odbc_driver
            """
            if not self.driver_is_implemented():
                raise ODBCDriverException(message="The current driver has not been tested. Please add it to the Connection.driver_is_implemented function.")

            table_attr = type + "_table"
            table = getattr(self, table_attr)
            if before:
                table = before + table
            if after:
                table = table + after
            
            formatted_table = f"[{ period_to_brackets(table) }]"

            if not include_database:
                return formatted_table

            formatted_catalog = self._get_formatted_catalog(type)

            return f"{formatted_catalog}.{formatted_table}"

        def get_formatted_class_table(self, include_database: bool = True, before: str = None, after: str = None) -> str:
            """ Gets the class table as a formatted string for the correct driver
                In the type of [schema].[catalog].[table]
            """
            return self._get_formatted_table("class", include_database, before, after)

        def get_formatted_data_table(self, include_database: bool = True) -> str:
            """ Gets the data table as a formatted string for the correct driver
                In the type of [schema].[catalog].[table]
            """
            return self._get_formatted_table("data", include_database)

        def get_formatted_prediction_tables(self, include_database: bool = True) -> dict:
            """ Gets header and row tables based on the class table in the config """
            
            results = {
                "header": self.get_formatted_class_table(include_database, after="Header"),
                "row": self.get_formatted_class_table(include_database, after="Row")
            }
            
            return results

            
        def get_catalog_params(self, type) -> dict:
            """ Gets params to connect to a database """
            params = {
                "driver": self.odbc_driver,
                "host": self.host,
                "catalog": "",
                "trusted_connection": self.trusted_connection,
                "username": "",
                "password": ""
            }

            params["username"] = self.sql_username
            params["password"] = self.sql_password
            if type == "class":
                params["catalog"] = self.class_catalog
            elif type == "data":
                params["catalog"] = self.data_catalog
            else:
                raise ConfigException(f"Type {type} not acceptable as a connection type")
            
            return params

        def update_catalogs(self, type: str, catalogs: list, checking_func: Callable) -> list:
            """ Updates the config to only contain accessible catalogs """

            default_catalog = self.data_catalog
            for catalog in catalogs: 
                if catalog != "":
                    self.data_catalog = catalog
                    try:
                        _ = checking_func("tables", "{}.{}")
                    except Exception:
                        catalogs.remove(catalog)
            self.data_catalog = default_catalog
            
            return catalogs

        def __str__(self) -> str:
            order = 1

            return Helpers.config_dict_to_list(order, self.to_dict())
            

        def to_dict(self) -> dict[str, str]:
            """ A dict with the headers for each value """
            sql_username = self.sql_username if self.sql_username else None
            sql_password= self.sql_password if self.sql_password else None

            str_dict = {
                "title": "Database settings", # 1
                "Driver":                         self.odbc_driver,
                "Classification Host":            self.host,
                "Trusted connection":             self.trusted_connection,
                "Classification Table":           self.class_catalog,
                "Classification Table":           self.class_table,
                "MS SQL username (*)":            sql_username,
                "MS SQL password (*)":            sql_password,
                "Data Catalog":                   self.data_catalog,
                "Data Table":                     self.data_table,
                "Classification column":          self.class_column,
                "Text Data columns (CSV)":        ', '.join(self.data_text_columns),
                "Numerical Data columns (CSV)":   ', '.join(self.data_numerical_columns),
                "Unique data id column":          self.id_column,
            }

            return str_dict

    @dataclass
    class IO:
        verbose: bool = True
        model_path: str = "./model/"
        model_name: str = "iris"

        def validate(self) -> None:
            """ Throws TypeError if invalid """
        
            for item in [
                "verbose",
            ]:
                if not isinstance(getattr(self, item), bool):
                    raise TypeError(f"Argument {item} must be True or False")

        
        def __str__(self) -> str:
            order = 3

            return Helpers.config_dict_to_list(order, self.to_dict())
            

        def to_dict(self) -> dict[str, str]:
            """ A dict with the headers for each value """
            str_dict = {
                "title": "I/O specifications", #3
                "Show full info":                             self.verbose,
                "Path where to save generated model":    self.model_path,
                "Name of generated or loaded model":     self.model_name
            }

            return str_dict    

    @dataclass
    class Mode:
        train: bool = True
        predict: bool = True
        mispredicted: bool = True
        use_metas: bool = True
        use_stop_words: bool = True
        specific_stop_words_threshold: float = 1.0
        hex_encode: bool = True
        use_categorization: bool = True
        category_text_columns: list = field(default_factory=list)
        test_size: float = 0.2
        oversampler: Oversampling = field(default_factory=Oversampling.defaultOversampler)
        undersampler: Undersampling = field(default_factory=Undersampling.defaultUndersampler)
        algorithm: AlgorithmTuple = field(default_factory=AlgorithmTuple.defaultAlgorithmTuple)        
        preprocessor: PreprocessTuple = field(default_factory=PreprocessTuple.defaultPreprocessTuple)
        feature_selection: ReductionTuple = field(default_factory=ReductionTuple.defaultReductionTuple)
        num_selected_features: int = None
        scoring: ScoreMetric = field(default_factory=ScoreMetric.defaultScoreMetric)
        max_iterations: int = None

        def validate(self) -> None:
            """ Throws TypeError if invalid """

            if not isinstance(self.category_text_columns, list):
                raise TypeError(f"Argument category_text_columns must be a list of strings")

            if not all(isinstance(x,str) for x in self.category_text_columns):
                raise TypeError(f"Argument category_text_columns must be a list of strings")
            
            if not Helpers.positive_int_or_none(self.num_selected_features):
                raise ValueError(
                    "Argument num_selected_features must be a positive integer")

            if self.max_iterations is None:
                self.max_iterations = Config.MAX_ITERATIONS
            elif not Helpers.positive_int_or_none(self.max_iterations):
                raise ValueError(
                    "Argument max_iterations must be a positive integer")

            # Type checking + at least one is True
            mode_types = [
                isinstance(self.train, bool),
                isinstance(self.predict, bool),
                isinstance(self.mispredicted, bool),
                isinstance(self.use_metas, bool),
                (self.train or self.predict)
            ]
            
            if not all(mode_types):
                raise ValueError(
                    "Class must be set for either training, predictions and/or mispredictions!")

            if self.mispredicted and not self.train:
                raise ValueError(
                    "Class must be set for training if it is set for misprediction")

            # Stop words threshold and test size
            if isinstance(self.specific_stop_words_threshold, float):
                if self.specific_stop_words_threshold > 1.0 or self.specific_stop_words_threshold < 0.0:
                    raise ValueError(
                        "Argument specific_stop_words_threshold must be between 0 and 1!")
            else:
                raise TypeError(
                    "Argument specific_stop_words_threshold must be a float between 0 and 1!")

            
            if isinstance(self.test_size, float):
                if self.test_size > 1.0 or self.test_size < 0.0:
                    raise ValueError(
                        "Argument test_size must be between 0 and 1!")
            else:
                raise TypeError(
                    "Argument test_size must be a float between 0 and 1!")

            if not (isinstance(self.algorithm, AlgorithmTuple)):
                raise TypeError("Argument algorithm is invalid")

            if not (isinstance(self.preprocessor, PreprocessTuple)):
                raise TypeError("Argument preprocessor is invalid")

            if not (isinstance(self.feature_selection, ReductionTuple)):
                raise TypeError(f"Argument feature_selection is invalid: {str(self.feature_selection)}")
            
            if not (isinstance(self.scoring, ScoreMetric)):
                raise TypeError(f"Argument scoring is invalid: {str(self.scoring)}")

            if not (isinstance(self.oversampler, Oversampling)):
                raise TypeError(f"Argument oversampler is invalid: {str(self.oversampler)}")

            if not (isinstance(self.undersampler, Undersampling)):
                raise TypeError(f"Argument undersampler is invalid: {str(self.undersampler)}")

            for item in [
                "use_stop_words",
                "hex_encode",
            ]:
                if not isinstance(getattr(self, item), bool):
                    raise TypeError(f"Argument {item} must be True or False")

        def __str__(self) -> str:
            order = 2

            return Helpers.config_dict_to_list(order, self.to_dict())
            

        def to_dict(self) -> dict[str, str]:
            """ A dict with the headers for each value """
            forced_columns = ', '.join(self.category_text_columns) if len(self.category_text_columns) > 0 else None

            str_dict = {
                "title": "Classification mode settings", #2
                "Train new model":                       self.train,
                "Make predictions with model":           self.predict,
                "Display mispredicted training data":    self.mispredicted,
                "Pass on meta data for predictions":     self.use_metas,
                "Use stop words":                        self.use_stop_words,
                "Material specific stop words threshold":self.specific_stop_words_threshold,
                "Hex encode text data":                  self.hex_encode,
                "Categorize text data where applicable": self.use_categorization,
                "Force categorization to these columns": forced_columns,
                "Test size for trainings":               self.test_size,
                "Oversampling technique":                self.oversampler.full_name,
                "Undersampling technique":               self.undersampler.full_name,
                "Algorithms of choice":                  self.algorithm.full_name,
                "Preprocessing method of choice":        self.preprocessor.full_name,
                "Scoring method":                        self.scoring.full_name,
                "Feature selection":                     self.feature_selection.full_name,
                "Number of selected features":           self.num_selected_features,
                "Maximum iterations (where applicable)": self.max_iterations
            }

            return str_dict

    @dataclass
    class Debug:
        on: bool = True
        data_limit: int = None

        def validate(self) -> None:
            """ Throws TypeError if invalid """

            # TODO: Set the value based on count_data_rows(), but first decide where that method should be
            # The method is (and should probably stay) in DataLayer--question is where we set this
            if not Helpers.positive_int_or_none(self.data_limit):
                raise ValueError(
                    "Argument data_limit must be a positive integer")

        def __str__(self) -> str:
            order = 4

            return Helpers.config_dict_to_list(order, self.to_dict())
            

        def to_dict(self) -> dict[str, str]:
            """ A dict with the headers for each value """
            str_dict = {
                "title": "Debug settings", #4
                "Debugging on":                          self.on,
                "How many data rows to consider":        self.data_limit
            }

            return str_dict

    @dataclass
    class Mail:
        smtp_server: str = "localhost"
        notification_email: str = ""
    
    connection: Connection = field(default_factory=Connection)
    mode: Mode  = field(default_factory=Mode)
    io: IO = field(default_factory=IO)
    debug: Debug = field(default_factory=Debug)
    mail: Mail = field(default_factory=Mail)
    name: str = "iris"
    config_path: Path = None
    script_path: Path = None
    _filename: str = None
    save: bool = False
    
    @property
    def filename(self) -> str:
        if self._filename is None:
            return self.get_filename("config")

        return self._filename

    @property
    def mode_suffix(self) -> str:
        """ Sets a suffix based on the predict/train mode(s) """
        suffixes = []
        if self.should_train():
            suffixes.append("train")
        if self.should_predict():
            suffixes.append("predict")

        return "_".join(suffixes)

    def get_filename(self, type: str) -> str:
        """ Simplifies the names of config and output files"""

        types = {
            "misplaced": {
                "suffix": "csv",
                "prefix": "misplaced_"
            },
            "cross_validation": {
                "suffix": "csv",
                "prefix": "crossval_"
            },
            "config": {
                "suffix": "py",
                "prefix": self.CONFIG_FILENAME_START
            }
        }

        type_dict = types[type]

        shared_parts = f"{self.name}_{self.connection.sql_username}_{self.mode_suffix}"

        return f"{type_dict['prefix']}{shared_parts}.{type_dict['suffix']}"

    def get_output_filepath(self, type: str, pwd: Path = None) -> str:
        """ Simplifies the path/names of output files """
        if pwd is None:
            pwd = self.script_path
        

        output_path = pwd / Path("output")
        
        return output_path / Path(self.get_filename(type))
    
    @property
    def filepath(self) -> Path:
        return self.config_path / self.filename

    def __post_init__(self) -> None:
        pwd = os.path.dirname(os.path.realpath(__file__))
        
        if self.config_path is None:
            self.config_path = Path(pwd) / "./config/"

        if self.script_path is None:
            self.script_path = Path(pwd)

        
        """Post init is called after init, which is the best place to check the types & values"""

        # 1: Top config params
        if not isinstance(self.name, str):
            raise TypeError(f"Argument name must be a string")

        # 2: Connection params
        self.connection.validate()

        # 3: Mode/training
        self.mode.validate()
        
        # 4: IO
        self.io.validate()

        # 5: Debug
        self.debug.validate()

        # This is True if running from GUI
        if self.save:
            self.save_to_file()

    def __str__(self) -> str:
        str_list = [
            " -- Configuration settings --",
            str(self.connection) + "\n",
            str(self.mode) + "\n",
            str(self.io) + "\n",
            str(self.debug)
        ]
        return  "\n".join(str_list)

    def to_dict(self) -> dict[str, Union[str, dict[str, str]]]:
        """ Gets all subdicts as dicts """
        str_dict = {
            "title": "Configuration settings",
            "connection": self.connection.to_dict(),
            "mode": self.mode.to_dict(),
            "io": self.io.to_dict(),
            "debug": self.debug.to_dict()
        }

        return str_dict

    @classmethod
    def get_username(self) -> str:
        """ calculates username """

        # TODO: Change to not use os.getlogin()
        return os.getlogin()
    
    def update_connection_columns(self, updated_columns = dict) -> dict:
        """ Wrapper function to not show inner workings """
        return self.connection.update_columns(updated_columns)

    def get_model_filename(self, pwd: Path = None) -> str:
        """ Set the name and path of the model file
            The second parameter allows for injecting the path for reliable testing
        """
        if pwd is None:
            pwd = self.script_path
        
        model_path = pwd / Path(self.io.model_path)
        
        return model_path / (self.io.model_name + Config.DEFAULT_MODEL_EXTENSION)

    def get_clean_config(self):
        """ Extracts the config information to save with a model """
        try:
            configuration = Config()
            configuration.connection = copy.deepcopy(self.connection)
            configuration.connection.data_catalog = ""
            configuration.connection.data_table = ""
            configuration.mode = copy.deepcopy(self.mode)
            configuration.mode.train = None
            configuration.mode.predict = None
            configuration.mode.mispredicted = None
            configuration.io = copy.deepcopy(self.io)
            configuration.io.model_name = ""
            configuration.debug = copy.deepcopy(self.debug)
            configuration.debug.data_limit = 0
            configuration.save = False
        except Exception as ex:
            print(f"Failed to extract configuration information: {str(self)} to save with a model: {str(ex)}")
        
        return configuration

    # Saves config to be read from the command line
    def save_to_file(self, filepath: Path = None, username: str = None) -> None:
        template_path = self.config_path / self.CONFIG_SAMPLE_FILE
        
        with open(template_path, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
        
        for tag in self.TEMPLATE_TAGS:
            #print(tag)
            template = self.TEMPLATE_TAGS[tag]
            location = tag.split(".")
            
            if (len(location) == 1):
                replace = getattr(self, location[0])
            else:
                head = getattr(self, location[0])
                replace = getattr(head, location[1])

                # Exception for class/data username, if given
                if username is not None and "username" in location[1]:
                    replace = username

            # Check if it's one of the enum variables
            if (isinstance(replace, enum.Enum)):
                replace = replace.name

            # Replace the lists with text representations
            if (isinstance(replace, list)):
                replace = ",".join(replace)

            for i in range(len(lines)):
                lines[i] = lines[i].replace(template, str(replace))

        filepath = filepath if filepath else self.filepath
        
        with open(filepath, "w", encoding="utf-8") as fout:
           fout.writelines(lines)

    def check_config_against_data_configuration(self, columns: list) -> bool:
        """ Check that the columns in the given table matches the config """
        class_check = [self.get_class_column_name() in columns]
        id_check = [self.get_id_column_name() in columns]
        data_check = [item in columns for item in self.get_data_column_names()]
        
        return all(class_check + id_check + data_check)
        

    @classmethod
    def load_config_from_model_file(cls: Type[T], filename: str, config: T = None) -> T:
        try:
            file_values = dill.load(open(filename, 'rb'))
            saved_config = file_values[0]
        except Exception as e:
            raise ConfigException(f"Something went wrong on loading model from file: {e}")
        
        if config is not None:
            saved_config.mode.train = config.mode.train
            saved_config.mode.predict = config.mode.predict
            saved_config.mode.mispredicted = config.mode.mispredicted
            saved_config.mode.use_metas = config.mode.use_metas
            saved_config.connection.data_catalog = config.connection.data_catalog
            saved_config.connection.data_table = config.connection.data_table
            saved_config.io.model_name = config.io.model_name
            saved_config.debug.data_limit = config.debug.data_limit
        
        return saved_config


    @classmethod
    def load_config_from_module(cls: Type[T], argv) -> T:
        module = Helpers.check_input_arguments(argv)
        
        version = '1.0'
        if (hasattr(module, 'version')):
            version = module.version

        if (version == '1.0'):
            raise ConfigException("V1.0 config is deprecated")
        
        return Config.load_config(module)

    @classmethod
    def load_config(cls: Type[T], module) -> T:
        if "num_rows" in module.debug:
            data_limit = Helpers.set_none_or_int(module.debug["num_rows"])
        else:
            data_limit = Helpers.set_none_or_int(module.debug["data_limit"])
        num_selected_features = Helpers.set_none_or_int(module.mode["num_selected_features"])
        max_iterations = Helpers.set_none_or_int(module.mode["max_iterations"])
        data_text_columns = Helpers.get_from_string_or_list(module.connection["data_text_columns"])
        data_numerical_columns = Helpers.get_from_string_or_list( module.connection["data_numerical_columns"])
        category_text_columns = Helpers.get_from_string_or_list(module.mode["category_text_columns"])
        use_metas = module.mode["predict"]
        if "use_metas" in module.mode:
            use_metas = module.mode["use_metas"]
       
        config = cls(
            Config.Connection(
                odbc_driver=module.connection["odbc_driver"],
                host=module.connection["host"],
                trusted_connection=module.connection["trusted_connection"],
                class_catalog=module.connection["class_catalog"],
                class_table=module.connection["class_table"],
                sql_username=module.connection["sql_username"],
                sql_password=module.connection["sql_password"],
                data_catalog=module.connection["data_catalog"],
                data_table=module.connection["data_table"],
                class_column=module.connection["class_column"],
                data_text_columns=data_text_columns,
                data_numerical_columns=data_numerical_columns,
                id_column=module.connection["id_column"],
            ),
            Config.Mode(
                train=module.mode["train"],
                predict=module.mode["predict"],
                mispredicted=module.mode["mispredicted"],
                use_metas=use_metas,
                use_stop_words=module.mode["use_stop_words"],
                specific_stop_words_threshold=float(
                    module.mode["specific_stop_words_threshold"]),
                hex_encode=module.mode["hex_encode"],
                use_categorization=module.mode["use_categorization"],
                category_text_columns=category_text_columns,
                test_size=float(module.mode["test_size"]),
                oversampler=Oversampling[module.mode["oversampler"]],
                undersampler=Undersampling[module.mode["undersampler"]],
                algorithm=AlgorithmTuple.from_string(module.mode["algorithm"]),
                preprocessor=PreprocessTuple.from_string(module.mode["preprocessor"]),
                feature_selection=ReductionTuple.from_string(module.mode["feature_selection"]),
                num_selected_features=num_selected_features,
                scoring=ScoreMetric[module.mode["scoring"]],
                max_iterations=max_iterations
            ),
            Config.IO(
                verbose=module.io["verbose"],
                model_path=module.io["model_path"],
                model_name=module.io["model_name"]
            ),
            Config.Mail(
                smtp_server=module.mail["smtp_server"],
                notification_email=module.mail["notification_email"]
            ),
            Config.Debug(
                on=module.debug["on"],
                data_limit=data_limit
            ),
            name=module.name
        )

        return config
        
    
    # Methods to hide implementation of Config
    def update_configuration(self, updates: dict) -> bool:
        """ Updates the config with new, wholesale, bits """
        # TODO: Break out validation to be able to call that here as well
        for key, item in updates.items():
            if not hasattr(self, key):
                raise ConfigException(f"Key {key} does not exist in Config")

            setattr(self, key, item)

        self.__post_init__()


    def is_numerical_data(self) -> bool:
        return len(self.connection.data_numerical_columns) > 0

    def force_categorization(self) -> bool:
        return  len(self.mode.category_text_columns) > 0

    def column_is_numeric(self, column: str) -> bool:
        """ Checks if the column is numerical """
        return column in self.get_numerical_column_names()

    def column_is_text(self, column: str) -> bool:
        """ Checks if the column is text based """
        return column in self.get_text_column_names()

    def use_imb_pipeline(self) -> bool:
        """ Returns True if either oversampling or undersampling is used """
        if self.mode.oversampler != Oversampling.NOG or self.mode.undersampler != Undersampling.NUG:
            return True

        return False

    def get_classification_script_path(self) -> Path:
        """ Gives a calculated path based on config"""
        
        return self.script_path / Config.CLASS_TABLE_SCRIPT
    
    def get_none_or_positive_value(self, attribute: str) -> int:
        value = self.get_attribute(attribute)
        
        if value is None or value == "":
            return 0

        return value

    def get_attribute(self, attribute: str):
        """ Gets an attribute from a attribute.subattribute string """
        location = attribute.split(".")
        length = len(location)
        if length > 2:
            raise ConfigException(f"Invalid format {attribute}")

        if length == 1:
            try:
                value = getattr(self, attribute)
            except AttributeError:
                raise ConfigException(f"There is no attribute {attribute}")

            return value

        try:
            head = getattr(self, location[0])
            value = getattr(head, location[1])
        except AttributeError:
            raise ConfigException(f"There is no attribute {attribute}")

        return value    
    
    def get_connection(self) -> Connection:
        """ Returns the connection object """
        return self.connection

    def get_quoted_attribute(self, attribute: str, quotes: str = "\'") -> str:
        """ Gets an attribute as per get_attribute, returns it in quotation marks """

        return to_quoted_string(self.get_attribute(attribute), quotes)
    
    @staticmethod
    def get_model_name(model: str, project_name: str) -> str:
        if model == Config.DEFAULT_TRAIN_OPTION:
            return project_name
        
        return model.replace(Config.DEFAULT_MODEL_EXTENSION, "")

    def get_num_selected_features(self) -> int:
        return self.get_none_or_positive_value("mode.num_selected_features")
        
    def get_test_size(self) -> float:
        """ Gets the test_size """
        return self.mode.test_size

    def get_test_size_percentage(self) -> int:
        """ Gets the test_size as a percentage """
        return int(self.mode.test_size * 100.0)

    def get_data_limit(self) -> int:
        """ Get the data limit"""
        return self.get_none_or_positive_value("debug.data_limit")

    def get_max_iterations(self) -> int:
        """ Get max iterations """
        return self.get_none_or_positive_value("mode.max_iterations")

    def is_verbose(self) -> bool:
        """ Returns what the io.verbose is set to"""
        return self.io.verbose
        
    def get_column_names(self) -> list[str]:
        """ Gets the column names based on connection columns """
        columns = self.get_data_column_names()

        if id_column := self.get_id_column_name():
            columns.append(id_column)

        if class_column := self.get_class_column_name():
            columns.append(class_column)
        
        return columns
        

    def get_categorical_text_column_names(self) -> list[str]:
        """ Gets the specified categorical text columns"""
        return self.mode.category_text_columns
        
        
    def get_text_column_names(self) -> list[str]:
        """ Gets the specified text columns"""
        return self.connection.data_text_columns


    def get_numerical_column_names(self) -> list[str]:
        """ Gets the specified numerical columns"""
        return self.connection.data_numerical_columns


    def get_data_column_names(self) -> list[str]:
        """ Gets data columns, so not Class or ID """
        return self.get_text_column_names() + self.get_numerical_column_names()
       

    def get_class_column_name(self) -> str:
        """ Gets the name of the column"""
        return self.connection.class_column

    def get_id_column_name(self) -> str:
        """ Gets the name of the ID column"""
        return self.connection.id_column

    def is_categorical(self, column_name: str) -> bool:
        """ Returns if a specific column is categorical"""
        return self.force_categorization() and column_name in self.get_categorical_text_column_names()

    def should_train(self) -> bool:
        """ Returns if this is a training config """
        return self.mode.train

    def should_predict(self) -> bool:
        """ Returns if this is a prediction config """
        return self.mode.predict

    def should_display_mispredicted(self) -> bool:
        """ Returns if this is a misprediction config """
        return self.mode.mispredicted

    def should_use_metas(self) -> bool:
        """ Returns if this is a use metas config """
        return self.mode.use_metas

    def use_stop_words(self) -> bool:
        """ Returns whether stop words should be used """
        return self.mode.use_stop_words

    def get_stop_words_threshold(self) -> float:
        """ Returns the threshold for the stop words """
        return self.mode.specific_stop_words_threshold

    def get_stop_words_threshold_percentage(self) -> int:
        """ Returns the threshold as an integer between 0-100"""
        return int(self.mode.specific_stop_words_threshold * 100.0)

    def should_hex_encode(self) -> bool:
        """ Returns whether dataset should be hex encoded """
        return self.mode.hex_encode

    def use_categorization(self) -> bool:
        """ Returns if categorization should be used """
        return self.mode.use_categorization

    def get_oversampler(self) -> Union[Oversampling, None]:
        """ Gets the oversampler for the model """
        return self.mode.oversampler.get_callable_oversampler()

    def set_oversampler(self, oversampler: Oversampling) -> None:
        """ Sets oversampler """

        self.mode.oversampler = oversampler

    def use_oversampling(self) -> bool:
        return self.mode.oversampler != Oversampling.NOG
    
    def get_undersampler(self) -> Union[Undersampling, None]:
        """ Gets the UnderSampler """
        return self.mode.undersampler.get_callable_undersampler()
    
    def set_undersampler(self, undersampler: Oversampling) -> None:
        """ Sets undersampler """

        self.mode.undersampler = undersampler

    def use_undersampling(self) -> bool:
        return self.mode.undersampler != Undersampling.NUG

    def set_num_selected_features(self, num_features: int) -> None:
        """ Updates the config with the number """
        self.mode.num_selected_features = num_features

    
    def get_callable_reductions(self, num_samples: int, num_features: int, num_selected_features: int = None) -> list:
        """ Returns callable reductions from mode.feature_selection """

        return self.mode.feature_selection.list_callable_reductions(num_samples, num_features, num_selected_features)

    def get_callable_algorithms(self, size: int, max_iterations: int) -> list:
        """ Returns callable algorithms from mode.algorithm """

        return self.mode.algorithm.list_callable_algorithms(size, max_iterations)

    def get_callable_preprocessors(self) -> list:
        """ Returns callable preprocessors from mode.algorithm """

        return self.mode.preprocessor.list_callable_preprocessors()

    def update_attribute(self, attribute: Union[str, dict], new_value) -> None:
        """ attribute is either a string on the form 'name.subname' 
        or a dict on the form { "name" : "subname", "type": "name"}
        """
        
        if isinstance(attribute, dict):
            try:
                head = getattr(self, attribute["type"])
                setattr(head, attribute["name"], new_value)
            except KeyError:
                raise ConfigException(f"Incorrect attribute format { attribute }")
            except AttributeError:
                raise ConfigException(f"There is no attribute { attribute }")

            return

        location = attribute.split(".")

        if len(location) > 2:
            raise ConfigException(f"There is no attribute {attribute}")
        try:
            if len(location) == 1:
                if hasattr(self, attribute):
                    setattr(self, attribute, new_value)
                else:
                    raise ConfigException(f"There is no attribute {attribute}")
            else:
                head = getattr(self, location[0])
                if hasattr(head, location[1]):
                    setattr(head, location[1], new_value)
                else:
                    raise ConfigException(f"There is no attribute {attribute}")
                
        except AttributeError:
            raise ConfigException(f"There is no attribute {attribute}")

    def update_attributes(self, updates: dict,  type: str = None) -> None:
        """ Updates several attributes inside the config """
        try:
            if type is None:
                """ Attempts to find the fields based on a split, useful if values belong to different parts """
                for attribute, value in updates.items():
                    self.update_attribute(attribute, value)
            else:
                attribute_dict = {"type": type}
                for attribute, value in updates.items():
                    attribute_dict["name"] = attribute
                    self.update_attribute(attribute_dict, value)
        except ConfigException:
            raise

    def get_scoring_mechanism(self)  -> Union[str, Callable]:
        """ While the actual function is in the mechanism, this allows us to hide where Scoring is """
        return self.mode.scoring.get_mechanism()

    def get_algorithm_abbreviations(self) -> list[str]:
        """ Get algorithm names """
        item = self.mode.algorithm
        if isinstance(item, MetaTuple):
            return item.get_abbreviations()

        return [item.name]
        

    def get_preprocessor_abbreviations(self) -> list[str]:
        """ get preprocessor names """
        item = self.mode.preprocessor
        if isinstance(item, MetaTuple):
            return item.get_abbreviations()

        return [item.name]
    
    def get_feature_selection_abbreviations(self) -> list[str]:
        """ Gets the given feature selection names """
        item = self.mode.feature_selection
        if isinstance(item, MetaTuple):
            return item.get_abbreviations()

        return [item.name]
    
    def get_oversampler_abbreviation(self) -> str:
        item = self.mode.oversampler
        if isinstance(item, tuple):
            return item.get_abbreviations()
        return item.name
    
    def get_undersampler_abbreviation(self) -> str:
        item = self.mode.undersampler
        if isinstance(item, tuple):
            return item.get_abbreviations()
        return item.name

    def get_class_catalog(self) -> str:
        """ Gets the class catalog """
        return self.connection.get_formatted_class_catalog()

    def get_class_table(self, include_database: bool = True) -> str:
        """ Gets the class table """
        return self.connection.get_formatted_class_table(include_database=include_database)
    
    def get_prediction_tables(self, include_database: bool = True) -> dict:
        """ Gets a dict with 'header' and 'row', based on class_table"""
        return self.connection.get_formatted_prediction_tables(include_database)

    def get_data_catalog(self) -> str:
        """ Gets the data catalog """
        return self.connection.get_formatted_data_catalog()

    def get_data_table(self, include_database: bool = False) -> str:
        """ Gets the data table """
        return self.connection.get_formatted_data_table(include_database=include_database)

    def get_data_username(self) -> str:
        """ Gets the data user name """
        return self.connection.sql_username

    def get_class_catalog_params(self) -> dict:
        """ Gets params to connect to class database """
        return self.connection.get_catalog_params("class")

    def get_data_catalog_params(self) -> dict:
        """ Gets params to connect to data database """
        return self.connection.get_catalog_params("data")


def to_quoted_string(x, quotes:str = '\'') -> str:
        value = str(x)

        return quotes + value + quotes

def period_to_brackets(string: str) -> str:
    """ Takes a string and replaces . with ].[
        Used with tables, schemas and databases    
    """

    return string.replace(".", "].[")


def main():
    # python Config.py -f .\config\test_iris.py
    if len(sys.argv) > 1:
        config = Config.load_config_from_module(sys.argv)
    else:
       config = Config()

    
    print(config)
    
if __name__ == "__main__":
    main()
