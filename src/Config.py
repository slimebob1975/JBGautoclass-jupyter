from __future__ import annotations

import copy
import getopt
import json
from types import SimpleNamespace
import enum
import os
import sys
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Callable, Type, TypeVar, Union

import Helpers
from JBGSampling import sampling_names
from JBGModelPersistence import load_model_config
from JBGModelSourceIdentity import check_model_source_identity
from JBGPaths import relocate_saved_config_paths
from JBGExceptions import ConfigException, ODBCDriverException
from JBGMeta import (Algorithm, AlgorithmTuple, Preprocess, PreprocessTuple,
                     Reduction, ReductionTuple, ScoreMetric, MetaTuple, Oversampling,
                     Undersampling, NgramRange)


T = TypeVar('T', bound='Config')
ODBC_drivers = ["SQL Server", "ODBC Driver 13 for SQL Server", "ODBC Driver 17 for SQL Server", "ODBC Driver 18 for SQL Server"]


def _positive_int_list(value, default: str) -> list[int]:
    """Accept generated CSV strings and direct list/tuple config values."""
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


class DarkNumberAlpha(enum.Enum):
    NONE = "none"
    SINGLE = "single"
    SEPARATED = "separated"

    @property
    def display_name(self) -> str:
        return {
            DarkNumberAlpha.NONE: "None",
            DarkNumberAlpha.SINGLE: "Single",
            DarkNumberAlpha.SEPARATED: "Separated",
        }[self]

    @classmethod
    def from_config_value(cls, value) -> "DarkNumberAlpha":
        if isinstance(value, cls):
            return value
        text = str(value or "NONE").strip()
        if text in cls.__members__:
            return cls[text]
        for alpha in cls:
            if text == alpha.value:
                return alpha
        raise ValueError(f"Unsupported Dark Number alpha mode: {value}")


class DarkNumberMethod(enum.Enum):
    LINEAR = "base"
    NON_LINEAR = "non_linear"

    @property
    def display_name(self) -> str:
        return "Linear" if self is DarkNumberMethod.LINEAR else "Non-linear"

    @classmethod
    def from_config_value(cls, value) -> "DarkNumberMethod":
        if isinstance(value, cls):
            return value
        text = str(value or "LINEAR").strip()
        if text in cls.__members__:
            return cls[text]
        for method in cls:
            if text == method.value:
                return method
        raise ValueError(f"Unsupported Dark Number method: {value}")


@dataclass
class Config:
    
    MAX_ITERATIONS = 20000
    CONFIG_FILENAME_START = "autoclassconfig_"
    JSON_CONFIG_SCHEMA_VERSION = 2

    CLASS_TABLE_SCRIPT = "./sql/CreatePredictionTables.sql.txt"

    DEFAULT_MODELS_PATH =  ".\\model\\"
    DEFAULT_MODEL_EXTENSION = ".sav"
    DEFAULT_TRAIN_OPTION = "Train a new model"
    SQL_PASSWORD_ENV = "JBG_SQL_PASSWORD"

    TEXT_DATATYPES = ["nvarchar", "varchar", "char", "text", "enum", "set"]
    INT_DATATYPES = ["bigint", "int", "smallint", "tinyint"]

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
#                "MS SQL password (*)":            sql_password,
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
        ngram_range: NgramRange = field(default_factory=NgramRange.defaultNgramRange)
        hex_encode: bool = True
        use_categorization: bool = True
        category_text_columns: list = field(default_factory=list)
        test_size: float = 0.2
        calculate_dark_numbers: bool = True
        dark_number_method: DarkNumberMethod = field(default_factory=lambda: DarkNumberMethod.LINEAR)
        dark_number_alpha: DarkNumberAlpha = None
        dark_number_target: str = ""
        dark_number_flip_fraction: float = 0.2
        experimental_perturbed_dark_number_fallback: bool = True
        oversampler: Oversampling = field(default_factory=Oversampling.defaultOversampler)
        undersampler: Undersampling = field(default_factory=Undersampling.defaultUndersampler)
        oversamplers: list[str] = None
        undersamplers: list[str] = None
        algorithm: AlgorithmTuple = field(default_factory=AlgorithmTuple.defaultAlgorithmTuple)
        model_parameters: dict = field(default_factory=dict)
        decision_policy: str = "default"
        probability_threshold: float = 0.5
        threshold_diagnostics: bool = False
        model_benchmark_150: bool = False
        holdout_split_mode: str = "stratified_rows"
        validation_group_column: str = "iafId"
        grouped_cross_validation: bool = False
        excluded_predictor_columns: str = ""
        preprocessor: PreprocessTuple = field(default_factory=PreprocessTuple.defaultPreprocessTuple)
        feature_selection: ReductionTuple = field(default_factory=ReductionTuple.defaultReductionTuple)
        num_selected_features: int = None
        scoring: ScoreMetric = field(default_factory=ScoreMetric.defaultScoreMetric)
        max_iterations: int = None
        calculate_feature_importance: bool = False
        feature_importance_repeats: int = 5
        feature_selection_mode: str = "manual"
        feature_selection_profile_id: str = ""
        feature_search_model_type: str = "Linear"
        feature_search_scaler: str = "STA"
        feature_search_oversampling: str = "NOG"
        feature_search_reduction: str = "NOR"
        feature_search_trials: int = 4
        feature_search_sizes: list[int] = field(default_factory=lambda: [5, 10, 20])
        feature_search_inner_folds: int = 3
        feature_search_outer_folds: int = 3
        feature_search_top_sets: int = 1
        feature_search_refine_rounds: int = 0
        feature_search_max_evaluations: int = 5
        feature_search_min_gain: float = 0.005
        feature_search_linear_c: float = 1.0
        feature_search_linear_max_iterations: int = 10000
        feature_search_non_linear_max_iterations: int = 2000
        feature_search_non_linear_alpha: float = 0.0001
        feature_search_non_linear_hidden_layers: list[int] = field(default_factory=lambda: [100])
        feature_search_non_linear_learning_rate: float = 0.001
        feature_search_tree_estimator: str = "Extra Trees"
        feature_search_tree_class_weight: str = "None"
        feature_search_tree_positive_threshold: float = 0.5
        feature_search_preset: str = "Quick"
        feature_search_tree_n_estimators: int = 100
        feature_search_tree_max_depth: int = 0
        feature_search_tree_min_samples_leaf: int = 1
        feature_search_catboost_class_weight: str = "None"
        feature_search_catboost_iterations: int = 400

        def __post_init__(self) -> None:
            # New configurations default to the separated alpha adjustment for
            # the published linear method. The non-linear family has no
            # separated-alpha implementation, so its implicit default remains none.
            if self.dark_number_alpha is None:
                self.dark_number_alpha = (
                    DarkNumberAlpha.SEPARATED
                    if self.dark_number_method is DarkNumberMethod.LINEAR
                    else DarkNumberAlpha.NONE
                )

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

            # Ngram range 
            if not (isinstance(self.ngram_range, NgramRange)):
                raise TypeError(f"Argument ngram_range is invalid: {str(self.ngram_range)}")
            
            # Test size
            if isinstance(self.test_size, float):
                if self.test_size > 1.0 or self.test_size < 0.0:
                    raise ValueError(
                        "Argument test_size must be between 0 and 1!")
            else:
                raise TypeError(
                    "Argument test_size must be a float between 0 and 1!")

            if not isinstance(self.calculate_dark_numbers, bool):
                raise TypeError("Argument calculate_dark_numbers must be True or False")

            if not isinstance(self.calculate_feature_importance, bool):
                raise TypeError("Argument calculate_feature_importance must be True or False")
            if type(self.feature_importance_repeats) is not int or not 2 <= self.feature_importance_repeats <= 30:
                raise ValueError("Argument feature_importance_repeats must be an integer between 2 and 30")

            if self.feature_selection_mode not in ("manual", "assisted", "reuse"):
                raise ValueError("Argument feature_selection_mode must be manual, assisted or reuse")
            if not isinstance(self.feature_selection_profile_id, str):
                raise TypeError("Argument feature_selection_profile_id must be a string")
            if self.feature_search_model_type not in ("Linear", "Non-linear", "Tree", "Combination", "CatBoost"):
                raise ValueError("Argument feature_search_model_type must be Linear, Non-linear, Tree, Combination or CatBoost")
            if self.feature_search_catboost_class_weight not in ("None", "Balanced"):
                raise ValueError("Unknown CatBoost class weighting")
            if type(self.feature_search_catboost_iterations) is not int or not 1 <= self.feature_search_catboost_iterations <= 10000:
                raise ValueError("CatBoost iterations must be in [1, 10000]")
            if self.feature_search_preset not in ("Quick", "Standard", "Extended", "Custom"):
                raise ValueError("Unknown feature search preset")
            if self.feature_search_tree_estimator not in ("Extra Trees", "Random Forest"):
                raise ValueError("Unknown feature search tree estimator")
            if self.feature_search_tree_class_weight not in ("None", "balanced", "balanced_subsample"):
                raise ValueError("Unknown feature search tree class weighting")
            if not isinstance(self.feature_search_tree_positive_threshold, (int, float)) or not 0 < self.feature_search_tree_positive_threshold < 1:
                raise ValueError("Feature search positive cutoff must lie between 0 and 1")
            if self.feature_search_reduction not in ("NOR", "PCA"):
                raise ValueError("Argument feature_search_reduction must be NOR or PCA")
            if self.feature_search_oversampling not in ("NOG", "RND", "SME"):
                raise ValueError("Unknown feature search oversampling method")
            if self.feature_search_scaler not in ("STA", "MAX"):
                raise ValueError("Argument feature_search_scaler must be STA or MAX")
            for name in (
                "feature_search_trials", "feature_search_inner_folds", "feature_search_outer_folds",
                "feature_search_top_sets", "feature_search_max_evaluations",
                "feature_search_linear_max_iterations", "feature_search_non_linear_max_iterations",
                "feature_search_tree_n_estimators", "feature_search_tree_min_samples_leaf",
            ):
                if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                    raise ValueError(f"Argument {name} must be a positive integer")
            if type(self.feature_search_refine_rounds) is not int or self.feature_search_refine_rounds < 0:
                raise ValueError("Argument feature_search_refine_rounds must be a nonnegative integer")
            if self.feature_search_inner_folds < 2 or self.feature_search_outer_folds < 2:
                raise ValueError("Feature-search inner and outer CV each require at least two folds")
            if self.feature_search_max_evaluations < self.feature_search_trials + 1:
                raise ValueError("Feature-search max evaluations must allow the baseline plus all trials")
            if (not isinstance(self.feature_search_sizes, list) or not self.feature_search_sizes
                    or any(type(value) is not int or value < 1 for value in self.feature_search_sizes)):
                raise ValueError("Argument feature_search_sizes must be a nonempty list of positive integers")
            if (not isinstance(self.feature_search_non_linear_hidden_layers, list)
                    or not self.feature_search_non_linear_hidden_layers
                    or any(type(value) is not int or value < 1 for value in self.feature_search_non_linear_hidden_layers)):
                raise ValueError("Argument feature_search_non_linear_hidden_layers must be a nonempty list of positive integers")
            if not isinstance(self.feature_search_min_gain, (int, float)) or self.feature_search_min_gain < 0:
                raise ValueError("Argument feature_search_min_gain must be nonnegative")
            if not isinstance(self.feature_search_linear_c, (int, float)) or self.feature_search_linear_c <= 0:
                raise ValueError("Argument feature_search_linear_c must be positive")
            if not isinstance(self.feature_search_non_linear_alpha, (int, float)) or self.feature_search_non_linear_alpha < 0:
                raise ValueError("Argument feature_search_non_linear_alpha must be nonnegative")
            if not isinstance(self.feature_search_non_linear_learning_rate, (int, float)) or self.feature_search_non_linear_learning_rate <= 0:
                raise ValueError("Argument feature_search_non_linear_learning_rate must be positive")
            if type(self.feature_search_tree_max_depth) is not int or self.feature_search_tree_max_depth < 0:
                raise ValueError("Argument feature_search_tree_max_depth must be zero or a positive integer")

            if not isinstance(self.dark_number_method, DarkNumberMethod):
                raise TypeError("Argument dark_number_method is invalid")

            if not isinstance(self.dark_number_alpha, DarkNumberAlpha):
                raise TypeError("Argument dark_number_alpha is invalid")
            if self.dark_number_method is DarkNumberMethod.NON_LINEAR and self.dark_number_alpha is DarkNumberAlpha.SEPARATED:
                raise ValueError("Separated alpha is only supported by the linear Dark Number method")

            if not isinstance(self.dark_number_target, str):
                raise TypeError("Argument dark_number_target must be a string")

            if not isinstance(self.experimental_perturbed_dark_number_fallback, bool):
                raise TypeError("Argument experimental_perturbed_dark_number_fallback must be True or False")

            if not isinstance(self.dark_number_flip_fraction, float):
                raise TypeError("Argument dark_number_flip_fraction must be a float between 0 and 1")
            if not 0.0 < self.dark_number_flip_fraction < 1.0:
                raise ValueError("Argument dark_number_flip_fraction must be greater than 0 and less than 1")

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

            # Normalize configured candidate selections to canonical abbreviations.
            for field_name, enum_type, fallback in (
                ("oversamplers", Oversampling, self.oversampler),
                ("undersamplers", Undersampling, self.undersampler),
            ):
                selection = getattr(self, field_name, None)
                if selection is not None:
                    setattr(self, field_name, sampling_names(selection, enum_type, fallback))

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

            feature_selection_mode_display = {
                "manual": "I will pick my own data settings",
                "assisted": "Help me pick the best data settings",
                "reuse": "Use a recommended data setting",
            }[self.feature_selection_mode]
            str_dict = {
                "title": "Classification mode settings", #2
                "Train new model":                       self.train,
                "Make predictions with model":           self.predict,
                "Display mispredicted training data":    self.mispredicted,
                "Pass on meta data for predictions":     self.use_metas,
                "Use stop words":                        self.use_stop_words,
                "Ngram range":                           self.ngram_range,
                "Hex encode text data":                  self.hex_encode,
                "Categorize text data where applicable": self.use_categorization,
                "Force categorization to these columns": forced_columns,
                "Test size for trainings":               self.test_size,
                "Calculate feature importance":            self.calculate_feature_importance,
                "Feature importance repeats":              self.feature_importance_repeats,
                "Feature selection modelling":             feature_selection_mode_display,
                "Feature selection profile":               self.feature_selection_profile_id or "None",
                "Investigative model":                      self.feature_search_model_type,
                "Calculate Dark Numbers":                  self.calculate_dark_numbers,
                "Dark Number method":                      self.dark_number_method.display_name,
                "Dark Number alpha":                       self.dark_number_alpha.display_name,
                "Dark Number target":                      self.dark_number_target or "All classes",
                "Correction failure handling":               (
                    "Experimental fallback"
                    if self.experimental_perturbed_dark_number_fallback
                    else "Controlled failure"
                ),
                "Dark-number injected label-noise fraction": self.dark_number_flip_fraction,
                "Oversampling technique":                self.oversampler.full_name,
                "Oversampling candidates":               ", ".join(sampling_names(getattr(self, "oversamplers", None), Oversampling, self.oversampler)),
                "Undersampling candidates":              ", ".join(sampling_names(getattr(self, "undersamplers", None), Undersampling, self.undersampler)),
                "Undersampling technique":               self.undersampler.full_name,
                "Algorithms of choice":                  self.algorithm.full_name,
                "Model parameters":                      str(getattr(self, "model_parameters", {})),
                "Decision policy":                       self.decision_policy,
                "Probability threshold":                 self.probability_threshold,
                "Training-only threshold diagnostics":  getattr(self, "threshold_diagnostics", False),
                "OOF model benchmark (extra fits)": getattr(self, "model_benchmark_150", False),
                "Holdout split mode":                  getattr(self, "holdout_split_mode", "stratified_rows"),
                "Validation group column": getattr(self, "validation_group_column", "iafId"),
                "Grouped cross-validation": getattr(self, "grouped_cross_validation", False),
                "Excluded predictor columns": getattr(self, "excluded_predictor_columns", ""),
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
    
    @dataclass
    class DataSource:
        """Backend identity shared by SQL and CSV in the existing JSON config.

        A CSV file name is display metadata, never a storage locator. Dataset
        and artifact IDs resolve through the scoped artifact repository.
        """
        kind: str = "sql"
        dataset_id: str = ""
        artifact_id: str = ""
        display_name: str = ""
        delimiter: str = ","
        quotechar: str = '"'
        encoding: str = "utf-8-sig"

        def validate(self) -> None:
            from uuid import UUID
            if self.kind not in ("sql", "csv"):
                raise ConfigException("Invalid data source type")
            if self.kind == "sql":
                if self.dataset_id or self.artifact_id:
                    raise ConfigException("SQL data source cannot reference CSV artifacts")
                return
            try:
                if (str(UUID(self.dataset_id)) != self.dataset_id
                        or str(UUID(self.artifact_id)) != self.artifact_id):
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                raise ConfigException("Invalid CSV dataset reference") from None
            if (not isinstance(self.display_name, str)
                    or '/' in self.display_name or "\\" in self.display_name
                    or not isinstance(self.delimiter, str) or len(self.delimiter) != 1
                    or self.delimiter in "\r\n\x00"
                    or not isinstance(self.quotechar, str) or len(self.quotechar) != 1
                    or self.quotechar in "\r\n\x00" or self.quotechar == self.delimiter
                    or self.encoding not in ("utf-8", "utf-8-sig", "latin-1", "cp1252")):
                raise ConfigException("Invalid CSV format or display metadata")

    connection: Connection = field(default_factory=Connection)
    data_source: DataSource = field(default_factory=DataSource)
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
            "dark_numb_conf_matrix": {
                "suffix": "csv",
                "prefix": "dark_numbers_conf_matrix_"
            },
            "dark_numbers": {
                "suffix": "csv",
                "prefix": "dark_numbers_"
            },
            "dark_number_fallback_events": {
                "suffix": "csv",
                "prefix": "dark_number_experimental_fallback_"
            },
            "misplaced": {
                "suffix": "csv",
                "prefix": "misplaced_"
            },
            "cross_validation": {
                "suffix": "csv",
                "prefix": "crossval_"
            },
            "feature_importance": {"suffix": "csv", "prefix": "feature_importance_"},
            "feature_importance_details": {"suffix": "csv", "prefix": "feature_importance_details_"},
            "config": {
                "suffix": "json",
                "prefix": self.CONFIG_FILENAME_START
            }
        }

        type_dict = types[type]

        shared_parts = f"{self.name}_{self.connection.sql_username}_{self.mode_suffix}"

        return f"{type_dict['prefix']}{shared_parts}.{type_dict['suffix']}"

    def get_output_filepath(self, type: str, pwd: Path = None) -> str:
        """ Simplifies the path/names of generated CSV output files. """
        if pwd is None:
            pwd = self.script_path

        output_path = pwd / Path("output") / Path("csvs")

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

        relocate_saved_config_paths(self)

        
        """Post init is called after init, which is the best place to check the types & values"""

        # 1: Top config params
        if not isinstance(self.name, str):
            raise TypeError(f"Argument name must be a string")

        # 2: Connection params
        self.connection.validate()
        self.data_source.validate()

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
            "data_source": self._json_section(self.data_source),
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
            configuration.data_source = copy.deepcopy(self.data_source)
            # Database passwords are runtime credentials and must never be embedded
            # in serialized model artifacts. Keep the username/connection metadata,
            # but require the password to be supplied again when the model is used.
            configuration.connection.sql_password = ""
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
        """Persist the complete configuration as schema-versioned JSON.

        Revision 176 intentionally ends generated Python configuration support.
        Configuration files are data and must never be imported or executed.
        """
        target = Path(filepath) if filepath is not None else self.filepath
        if target.suffix.lower() != ".json":
            raise ConfigException(
                "Python configuration files are no longer supported. "
                "Use a .json configuration generated by JBG revision 176 or newer."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_config(target, username=username)

    def _write_json_config(self, filepath: Path, username: str = None) -> None:
        payload = self._config_to_json(username=username)
        temporary_path = filepath.with_name(filepath.name + ".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as fout:
                json.dump(
                    payload, fout, ensure_ascii=False, indent=2, sort_keys=True,
                    allow_nan=False,
                )
                fout.write("\n")
            temporary_path.replace(filepath)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _json_value(value):
        if isinstance(value, enum.Enum):
            return value.name
        if hasattr(value, "get_abbreviations") and callable(value.get_abbreviations):
            return list(value.get_abbreviations())
        if isinstance(value, dict):
            return {str(key): Config._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Config._json_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        raise TypeError(f"Configuration value of type {type(value).__name__} is not JSON serializable")

    @classmethod
    def _json_section(cls, section) -> dict:
        return {
            item.name: cls._json_value(getattr(section, item.name))
            for item in dataclass_fields(section)
        }

    def _config_to_json(self, username: str = None) -> dict:
        """Return the complete schema-v2 data-only configuration payload."""
        connection = self._json_section(self.connection)
        connection["sql_password"] = ""
        if username is not None:
            connection["sql_username"] = username

        return {
            "schema_version": self.JSON_CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "connection": connection,
            "data_source": self._json_section(self.data_source),
            "mode": self._json_section(self.mode),
            "io": self._json_section(self.io),
            "debug": self._json_section(self.debug),
            "mail": self._json_section(self.mail),
        }

    @classmethod
    def _validate_json_payload(cls, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise ConfigException("JSON configuration must contain an object at the top level")
        if payload.get("schema_version") != cls.JSON_CONFIG_SCHEMA_VERSION:
            raise ConfigException(
                f"Unsupported JSON configuration schema_version; expected {cls.JSON_CONFIG_SCHEMA_VERSION}. "
                "Regenerate the configuration with the current JBG version."
            )
        if not isinstance(payload.get("name"), str):
            raise ConfigException("JSON configuration name must be a string")

        section_types = {
            "connection": cls.Connection,
            "mode": cls.Mode,
            "io": cls.IO,
            "debug": cls.Debug,
            "mail": cls.Mail,
        }
        expected_top = {"schema_version", "name", *section_types}
        optional_top = {"data_source"}
        missing_top = expected_top - set(payload)
        unknown_top = set(payload) - expected_top - optional_top
        if missing_top:
            raise ConfigException(
                "JSON schema v2 configuration is missing top-level keys: "
                + ", ".join(sorted(missing_top))
            )
        if unknown_top:
            raise ConfigException(
                "JSON schema v2 configuration has unknown top-level keys: "
                + ", ".join(sorted(unknown_top))
            )

        if "data_source" in payload:
            section = payload["data_source"]
            expected = {item.name for item in dataclass_fields(cls.DataSource)}
            if not isinstance(section, dict) or set(section) != expected:
                raise ConfigException("Invalid data source configuration section")
            try:
                cls.DataSource(**section).validate()
            except (TypeError, ValueError) as exc:
                raise ConfigException("Invalid data source configuration") from exc

        for section_name, section_type in section_types.items():
            section = payload[section_name]
            if not isinstance(section, dict):
                raise ConfigException(f"JSON configuration section '{section_name}' must be an object")
            expected = {item.name for item in dataclass_fields(section_type)}
            actual = set(section)
            missing = expected - actual
            unknown = actual - expected
            if missing:
                raise ConfigException(
                    f"JSON schema v2 section '{section_name}' is missing keys: "
                    + ", ".join(sorted(missing))
                )
            if unknown:
                raise ConfigException(
                    f"JSON schema v2 section '{section_name}' has unknown keys: "
                    + ", ".join(sorted(unknown))
                )

    @classmethod
    def load_config_from_json(cls: Type[T], filepath: Path) -> T:
        """Load a schema-v2 JSON configuration without executing code."""
        path = Path(filepath)
        if path.suffix.lower() != ".json":
            raise ConfigException(
                "Python configuration files are no longer supported. "
                "Use a .json configuration generated by JBG revision 176 or newer."
            )
        try:
            with path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, ValueError) as exc:
            raise ConfigException(f"Unable to read JSON configuration {path}: {exc}") from exc

        cls._validate_json_payload(payload)
        connection = dict(payload["connection"])
        connection["sql_password"] = ""
        module = SimpleNamespace(
            name=payload["name"],
            connection=connection,
            mode=payload["mode"],
            io=payload["io"],
            debug=payload["debug"],
            mail=payload["mail"],
            data_source=payload.get("data_source"),
        )
        try:
            return cls.load_config(module)
        except ConfigException:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigException(f"Invalid JSON configuration values: {exc}") from exc

    @classmethod
    def load_config_from_arguments(cls: Type[T], argv) -> T:
        """Load the JSON config selected by -f/--file on a command line."""
        usage = f"Usage: {argv[0]} [-h/--help] [-f/--file <config.json>]"
        try:
            options, remaining = getopt.getopt(argv[1:], "hf:", ["help", "file="])
        except getopt.GetoptError as exc:
            raise ConfigException(f"{exc}. {usage}") from exc
        if remaining:
            raise ConfigException(f"Unexpected command-line arguments: {' '.join(remaining)}. {usage}")

        filepath = None
        for option, value in options:
            if option in ("-h", "--help"):
                print(usage)
                raise SystemExit(0)
            if option in ("-f", "--file"):
                filepath = Path(value)

        if filepath is None:
            raise ConfigException(f"No configuration file specified. {usage}")
        return cls.load_config_from_json(filepath)

    def check_config_against_data_configuration(self, columns: list) -> bool:
        """ Check that the columns in the given table matches the config """
        class_check = [self.get_class_column_name() in columns]
        id_check = [self.get_id_column_name() in columns]
        data_check = [item in columns for item in self.get_data_column_names()]
        
        return all(class_check + id_check + data_check)
        

    @classmethod
    def _validate_current_saved_config(cls, saved_config) -> None:
        if not isinstance(saved_config, cls):
            raise ConfigException(
                "Saved model configuration is from an unsupported pre-revision-176 artifact. "
                "Retrain and save a new model."
            )
        sections = {
            "connection": cls.Connection,
            "mode": cls.Mode,
            "io": cls.IO,
            "debug": cls.Debug,
            "mail": cls.Mail,
        }
        for section_name, section_type in sections.items():
            section = getattr(saved_config, section_name, None)
            missing = [
                item.name for item in dataclass_fields(section_type)
                if section is None or not hasattr(section, item.name)
            ]
            if missing:
                raise ConfigException(
                    "Saved model configuration is missing current fields in "
                    f"{section_name}: {', '.join(missing)}. Retrain and save a new model."
                )

    @classmethod
    def load_config_from_model_file(cls: Type[T], filename: str, config: T = None) -> T:
        try:
            saved_config = load_model_config(filename)
        except Exception as e:
            raise ConfigException(f"Something went wrong on loading model from file: {e}")

        cls._validate_current_saved_config(saved_config)

        if config is not None:
            # Same model artifact and embedded JSON config for both sources.
            # SQL compatibility remains unchanged; CSV requires exact dataset
            # identity to prevent accidental cross-upload model reuse.
            try:
                check_model_source_identity(saved_config, config)
            except ValueError as exc:
                raise ConfigException(str(exc)) from exc
            saved_config.mode.train = config.mode.train
            saved_config.mode.predict = config.mode.predict
            saved_config.mode.mispredicted = config.mode.mispredicted
            saved_config.mode.use_metas = config.mode.use_metas
            saved_config.mode.calculate_feature_importance = config.should_calculate_feature_importance()
            saved_config.mode.feature_importance_repeats = config.get_feature_importance_repeats()
            saved_config.mode.calculate_dark_numbers = config.mode.calculate_dark_numbers
            saved_config.mode.dark_number_method = config.mode.dark_number_method
            saved_config.mode.dark_number_alpha = config.mode.dark_number_alpha
            saved_config.mode.dark_number_target = config.mode.dark_number_target
            saved_config.mode.experimental_perturbed_dark_number_fallback = config.mode.experimental_perturbed_dark_number_fallback
            saved_config.mode.dark_number_flip_fraction = config.mode.dark_number_flip_fraction
            # Saved model metadata deliberately contains no SQL password. Inject the
            # credentials from the current runtime configuration when loading a model.
            saved_config.connection.sql_username = config.connection.sql_username
            saved_config.connection.sql_password = config.connection.sql_password
            saved_config.connection.data_catalog = config.connection.data_catalog
            saved_config.connection.data_table = config.connection.data_table
            saved_config.io.model_name = config.io.model_name
            saved_config.debug.data_limit = config.debug.data_limit
            # Mail settings are runtime configuration and are intentionally not
            # persisted in model metadata.  Keep the current GUI/CLI values when
            # a prediction-only run replaces the live config with the saved one.
            saved_config.mail = copy.deepcopy(config.mail)
        
        return saved_config


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
        calculate_dark_numbers = bool(module.mode.get("calculate_dark_numbers", module.mode["mispredicted"]))
        dark_number_method = DarkNumberMethod.from_config_value(module.mode.get("dark_number_method", "LINEAR"))
        dark_number_alpha = DarkNumberAlpha.from_config_value(module.mode.get("dark_number_alpha", "NONE"))
        dark_number_target = str(module.mode.get("dark_number_target", "") or "")
        dark_number_flip_fraction = float(module.mode.get("dark_number_flip_fraction", 0.2))
        experimental_perturbed_dark_number_fallback = bool(module.mode.get("experimental_perturbed_dark_number_fallback", True))
        use_metas = module.mode["predict"]
        if "use_metas" in module.mode:
            use_metas = module.mode["use_metas"]
       
        configured_password = module.connection.get("sql_password", "")
        runtime_password = configured_password or os.environ.get(cls.SQL_PASSWORD_ENV, "")
        mail_defaults = cls.Mail()
        mail_settings = getattr(module, "mail", {})

        config = cls(
            connection=Config.Connection(
                odbc_driver=module.connection["odbc_driver"],
                host=module.connection["host"],
                trusted_connection=module.connection["trusted_connection"],
                class_catalog=module.connection["class_catalog"],
                class_table=module.connection["class_table"],
                sql_username=module.connection["sql_username"],
                sql_password=runtime_password,
                data_catalog=module.connection["data_catalog"],
                data_table=module.connection["data_table"],
                class_column=module.connection["class_column"],
                data_text_columns=data_text_columns,
                data_numerical_columns=data_numerical_columns,
                id_column=module.connection["id_column"],
            ),
            mode=Config.Mode(
                train=module.mode["train"],
                predict=module.mode["predict"],
                mispredicted=module.mode["mispredicted"],
                use_metas=use_metas,
                use_stop_words=module.mode["use_stop_words"],
                ngram_range=(module.mode["ngram_range"] if isinstance(module.mode["ngram_range"], NgramRange)
                             else NgramRange[module.mode["ngram_range"]]),
                hex_encode=module.mode["hex_encode"],
                use_categorization=module.mode["use_categorization"],
                category_text_columns=category_text_columns,
                test_size=float(module.mode["test_size"]),
                calculate_dark_numbers=calculate_dark_numbers,
                dark_number_method=dark_number_method,
                dark_number_alpha=dark_number_alpha,
                dark_number_target=dark_number_target,
                dark_number_flip_fraction=dark_number_flip_fraction,
                experimental_perturbed_dark_number_fallback=experimental_perturbed_dark_number_fallback,
                oversampler=Oversampling[module.mode["oversampler"]],
                undersampler=Undersampling[module.mode["undersampler"]],
                oversamplers=module.mode.get("oversamplers") or None,
                undersamplers=module.mode.get("undersamplers") or None,
                algorithm=(AlgorithmTuple.from_string(module.mode["algorithm"])
                           if isinstance(module.mode["algorithm"], str)
                           else AlgorithmTuple(module.mode["algorithm"])),
                model_parameters=module.mode.get("model_parameters", {}),
                decision_policy=module.mode.get("decision_policy", "default"),
                probability_threshold=float(module.mode.get("probability_threshold", 0.5)),
                threshold_diagnostics=bool(module.mode.get("threshold_diagnostics", False)),
                model_benchmark_150=bool(module.mode.get("model_benchmark_150", False)),
                holdout_split_mode=module.mode.get("holdout_split_mode", "stratified_rows"),
                validation_group_column=module.mode.get("validation_group_column", "iafId"),
                grouped_cross_validation=bool(module.mode.get("grouped_cross_validation", False)),
                excluded_predictor_columns=module.mode.get("excluded_predictor_columns", ""),
                preprocessor=(PreprocessTuple.from_string(module.mode["preprocessor"])
                              if isinstance(module.mode["preprocessor"], str)
                              else PreprocessTuple(module.mode["preprocessor"])),
                feature_selection=(ReductionTuple.from_string(module.mode["feature_selection"])
                                   if isinstance(module.mode["feature_selection"], str)
                                   else ReductionTuple(module.mode["feature_selection"])),
                num_selected_features=num_selected_features,
                scoring=ScoreMetric[module.mode["scoring"]],
                max_iterations=max_iterations,
                calculate_feature_importance=module.mode.get("calculate_feature_importance", False),
                feature_importance_repeats=module.mode.get("feature_importance_repeats", 5),
                feature_selection_mode=module.mode.get("feature_selection_mode", "manual"),
                feature_selection_profile_id=module.mode.get("feature_selection_profile_id", ""),
                feature_search_model_type=module.mode.get("feature_search_model_type", "Linear"),
                feature_search_scaler=module.mode.get("feature_search_scaler", "STA"),
                feature_search_oversampling=module.mode.get("feature_search_oversampling", "NOG"),
                feature_search_reduction=module.mode.get("feature_search_reduction", "NOR"),
                feature_search_trials=int(module.mode.get("feature_search_trials", 4)),
                feature_search_sizes=_positive_int_list(module.mode.get("feature_search_sizes"), "5,10,20"),
                feature_search_inner_folds=int(module.mode.get("feature_search_inner_folds", 3)),
                feature_search_outer_folds=int(module.mode.get("feature_search_outer_folds", 3)),
                feature_search_top_sets=int(module.mode.get("feature_search_top_sets", 1)),
                feature_search_refine_rounds=int(module.mode.get("feature_search_refine_rounds", 0)),
                feature_search_max_evaluations=int(module.mode.get("feature_search_max_evaluations", 5)),
                feature_search_min_gain=float(module.mode.get("feature_search_min_gain", 0.005)),
                feature_search_linear_c=float(module.mode.get("feature_search_linear_c", 1.0)),
                feature_search_linear_max_iterations=int(module.mode.get("feature_search_linear_max_iterations", 10000)),
                feature_search_non_linear_max_iterations=int(module.mode.get("feature_search_non_linear_max_iterations", 2000)),
                feature_search_non_linear_alpha=float(module.mode.get("feature_search_non_linear_alpha", 0.0001)),
                feature_search_non_linear_hidden_layers=_positive_int_list(module.mode.get("feature_search_non_linear_hidden_layers"), "100"),
                feature_search_non_linear_learning_rate=float(module.mode.get("feature_search_non_linear_learning_rate", 0.001)),
                feature_search_preset=module.mode.get("feature_search_preset", "Quick"),
                feature_search_tree_estimator=module.mode.get("feature_search_tree_estimator", "Extra Trees"),
                feature_search_tree_class_weight=module.mode.get("feature_search_tree_class_weight", "None"),
                feature_search_tree_positive_threshold=module.mode.get("feature_search_tree_positive_threshold", 0.5),
                feature_search_tree_n_estimators=int(module.mode.get("feature_search_tree_n_estimators", 100)),
                feature_search_tree_max_depth=int(module.mode.get("feature_search_tree_max_depth", 0)),
                feature_search_tree_min_samples_leaf=int(module.mode.get("feature_search_tree_min_samples_leaf", 1)),
                feature_search_catboost_class_weight=module.mode.get("feature_search_catboost_class_weight", "None"),
                feature_search_catboost_iterations=int(module.mode.get("feature_search_catboost_iterations", 400)),
            ),
            io=Config.IO(
                verbose=module.io["verbose"],
                model_path=module.io["model_path"],
                model_name=module.io["model_name"]
            ),
            mail=Config.Mail(
                smtp_server=mail_settings.get("smtp_server", mail_defaults.smtp_server),
                notification_email=mail_settings.get("notification_email", mail_defaults.notification_email),
            ),
            debug=Config.Debug(
                on=module.debug["on"],
                data_limit=data_limit
            ),
            name=module.name
        )

        source = getattr(module, "data_source", None)
        if source is not None:
            config.data_source = cls.DataSource(**source)
            config.data_source.validate()
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

    def should_calculate_dark_numbers(self) -> bool:
        """Whether Dark Number estimation is enabled for this run."""
        return bool(self.mode.calculate_dark_numbers)

    def get_dark_number_method(self) -> DarkNumberMethod:
        """Return the configured Dark Number formula family."""
        return DarkNumberMethod.from_config_value(self.mode.dark_number_method)

    def get_dark_number_alpha(self) -> DarkNumberAlpha:
        """Return the configured Dark Number alpha adjustment."""
        return DarkNumberAlpha.from_config_value(self.mode.dark_number_alpha)

    def get_dark_number_calculation_type(self) -> str:
        """Return the calculator branch used by JBGDarkNumbers."""
        from JBGDarkNumbers import DarkNumberCalculator

        return DarkNumberCalculator.resolve_calculation_type(
            self.get_dark_number_method(),
            self.get_dark_number_alpha(),
        )

    def get_dark_number_target(self) -> str:
        """Return the selected Dark Number target class, or an empty string for all classes."""
        return str(self.mode.dark_number_target or "")

    def should_use_experimental_perturbed_dark_number_fallback(self) -> bool:
        """Whether statistical Dark Number failures may use the experimental shadow-clone fallback."""
        return bool(self.mode.experimental_perturbed_dark_number_fallback)

    def get_dark_number_flip_fraction(self) -> float:
        """Fraction of positive labels hidden when estimating Dark Number correction factors."""
        return self.mode.dark_number_flip_fraction

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

    def should_calculate_feature_importance(self) -> bool:
        """Optional inspection is performed only in a fresh training/holdout lifecycle."""
        return bool(self.should_train() and getattr(self.mode, "calculate_feature_importance", False))

    def get_feature_importance_repeats(self) -> int:
        return getattr(self.mode, "feature_importance_repeats", 5)

    def should_use_metas(self) -> bool:
        """ Returns if this is a use metas config """
        return self.mode.use_metas

    def use_stop_words(self) -> bool:
        """ Returns whether stop words should be used """
        return self.mode.use_stop_words

    def get_ngram_range(self) -> NgramRange:
        """ Returns the ngram range"""
        return self.mode.ngram_range

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

        from JBGModelSettings import configure_estimator
        # Native CatBoost uses raw categorical columns and cannot share the ordinary
        # global text converter, so do not advertise it as a candidate in exhaustive search.
        if "CBST" in self.get_algorithm_abbreviations():
            raise ValueError(
                "CBST is not supported in ordinary exhaustive pipeline search. "
                "Use CatBoost in Feature modelling instead."
            )
        algorithms = self.mode.algorithm.list_callable_algorithms(size, max_iterations)
        return [(algorithm, configure_estimator(algorithm.name, estimator,
                 getattr(self.mode, "model_parameters", {}))) for algorithm, estimator in algorithms]

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
    
    def get_oversampler_abbreviations(self) -> list[str]:
        return sampling_names(getattr(self.mode, "oversamplers", None), Oversampling, self.mode.oversampler)

    def get_undersampler_abbreviations(self) -> list[str]:
        return sampling_names(getattr(self.mode, "undersamplers", None), Undersampling, self.mode.undersampler)

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
    # python Config.py -f .\config\autoclassconfig_example.json
    if len(sys.argv) > 1:
        config = Config.load_config_from_arguments(sys.argv)
    else:
       config = Config()

    
    print(config)
    
if __name__ == "__main__":
    main()
