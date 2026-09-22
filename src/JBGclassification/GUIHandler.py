# Alternate GUI for Autoclassification script using widgets from Jupyter
# Written by: Robert Granat, Jan-Feb 2022.
# Broken into module by: Marie Hogebrandt June-oct 2022

import copy
import errno
import json
import os
import sys
from pathlib import Path

from ipywidgets import Output

src_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(os.getcwd(), ".env")

sys.path.append(src_dir)

from dotenv import load_dotenv

from JBGExceptions import DataLayerException
from JBGStreamedLogger import JBGLogger
from AutomaticClassifier import AutomaticClassifier as autoclass
from Config import Config
from SQLDataLayer import DataLayer
from GUI.Widgets import Widgets

# Class definition for the GUI
class GUIHandler:
    REGRESSION_SUITE_DATASETS = (
        {
            "label": "Iris",
            "slug": "iris",
            "table_name": "iris",
            "class_column": "class",
            "id_column": "id",
        },
        {
            "label": "Breast Cancer",
            "slug": "breast_cancer",
            "table_name": "breast_cancer",
            "class_column": "diagnosis",
            "id_column": "id",
        },
    )

    # Constructor
    def __init__(self):

        # Start persistent logging before constructing the GUI so startup warnings
        # and direct stdout/stderr output are captured as well.
        self.logger = JBGLogger(False) # Quiet is set to false here

        with self.logger.capture_console_output():
            settings = None
            settings_file = os.path.join(os.getcwd(), "settings.json")
            if os.path.isfile(settings_file):
                with open(settings_file) as f:
                    settings = json.load(f)

            self.widgets = Widgets(src_path=Path(src_dir), GUIhandler=self, settings=settings)
            self.logger.set_progress_widgets(self.widgets.progress)

            # This datalayer object is the one working with the classifier data
            self.classifier_datalayer = None

            # This datalayer object only works with the GUI
            self.gui_datalayer = None

            config = Config(
                connection=Config.Connection(
                odbc_driver=os.environ.get("DEFAULT_ODBC_DRIVER"),
                host=os.environ.get("DEFAULT_HOST"),
                class_catalog=os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
                class_table=os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
                data_catalog=os.environ.get("DEFAULT_DATA_CATALOG"),
                data_table=os.environ.get("DEFAULT_DATA_TABLE"),
                sql_username="",
                sql_password="",
                trusted_connection=False
                )
            )
            self.gui_datalayer = DataLayer(config=config, logger=self.logger)

            self.widgets.load_contents()

    @property
    def datalayer(self) -> DataLayer:
        """ This returns the GUI datalayer """
        return self.gui_datalayer

    def set_data_catalog(self, data_catalog: str) -> None:
        """ Updates the GUI datalayer with the data catalog """
        self.gui_datalayer.config.update_connection_columns({
            "data_catalog": data_catalog
            
        })

    def get_class_distribution(self, data_settings: dict, current_class: str) -> dict:
        """ Widgets does not need to know about Classifier Datalayer """
        datalayer = self.get_classifier_datalayer(data_settings=data_settings)
        distribution = {}
        try:
            distribution = datalayer.count_class_distribution()
        except DataLayerException as e:
            self.logger.abort_cleanly(str(e))
        except Exception as e:
            message = f"Could not update summary for class: {current_class} because {e}"
            self.logger.anaconda_debug(message)

        return distribution

    def correct_mispredicted_data(self, new_class: str, index: int) -> None:
        """ Changes the original dataset """
        self.classifier_datalayer.correct_mispredicted_data(new_class, index)

    def get_classifier_datalayer(self, data_settings: dict = None, config_params: dict = None) -> DataLayer:
        """ This will create or update the layer, and should probably also include the start_classifier() stuff
        """
        if config_params:
            # This creates a new classifier_datalayer whether one exists or not, because it has all the info
            self.classifier_datalayer = DataLayer(Config(**config_params), self.logger)

            return self.classifier_datalayer

        if data_settings:
            if not self.classifier_datalayer:
                connection = Config.Connection(
                        odbc_driver = os.environ.get("DEFAULT_ODBC_DRIVER"),
                        host = os.environ.get("DEFAULT_HOST"),
                        class_catalog = os.environ.get("DEFAULT_CLASSIFICATION_CATALOG"),
                        class_table = os.environ.get("DEFAULT_CLASSIFICATION_TABLE"),
                        sql_username = data_settings["connection"]["sql_username"],
                        sql_password = data_settings["connection"]["sql_password"],
                        data_catalog = data_settings["data"]["catalog"],
                        data_table = data_settings["data"]["table"],
                        class_column = data_settings["columns"]["class"], 
                        data_text_columns =  data_settings["columns"]["data_text"], 
                        data_numerical_columns = data_settings["columns"]["data_numerical"], 
                        id_column = data_settings["columns"]["id"]
                    )

                self.classifier_datalayer = DataLayer(Config(connection), self.logger)
            else:
                updated_columns = {
                    "class_column": data_settings["columns"]["class"], 
                    "data_text_columns":  data_settings["columns"]["data_text"], 
                    "data_numerical_columns": data_settings["columns"]["data_numerical"], 
                    "id_column": data_settings["columns"]["id"]
                }
                self.classifier_datalayer.config.update_connection_columns(updated_columns)

        return self.classifier_datalayer        
        
    def run_classifier(
        self,
        config_params: dict,
        output: Output,
        set_rerun: bool = True,
        regression_suite: bool = False,
    ):
        """ Sets up the classifier and then runs it"""
        
        self.get_classifier_datalayer(config_params = config_params)

        self.logger.set_enable_quiet(not config_params["io"].verbose)
        
        result = {"mispredicted": None} # Fullösning för nu
        with output, self.logger.capture_console_output():
            the_classifier = autoclass(
                config=self.classifier_datalayer.get_config(),
                logger=self.logger,
                datalayer=self.classifier_datalayer,
                regression_suite=regression_suite,
            )
            result = the_classifier.run()
            if not result:
                self.logger.print_info("No data was fetched from database!")
            
        
        if result and result["mispredicted"] is not None:
            self.widgets.handle_mispredicted(**result)    
        
        if set_rerun:
            self.widgets.set_rerun()
        return result

    @staticmethod
    def _table_basename(table: str) -> str:
        return str(table).rsplit(".", 1)[-1].casefold()

    def _regression_suite_catalogs(self, preferred_catalog: str) -> list[str]:
        catalogs = [catalog for catalog in self.datalayer.get_catalogs_as_options() if catalog]
        if preferred_catalog in catalogs:
            catalogs.remove(preferred_catalog)
            catalogs.insert(0, preferred_catalog)
        return catalogs

    def _find_regression_suite_dataset(self, dataset: dict, preferred_catalog: str) -> dict | None:
        original_catalog = self.datalayer.config.connection.data_catalog
        try:
            for catalog in self._regression_suite_catalogs(preferred_catalog):
                try:
                    self.set_data_catalog(catalog)
                    tables = [table for table in self.datalayer.get_tables_as_options() if table]
                except Exception as ex:
                    self.logger.print_warning(
                        f"Regression suite: could not inspect catalog {catalog}: {type(ex).__name__}: {ex}"
                    )
                    continue

                matching_tables = [
                    table for table in tables
                    if self._table_basename(table) == dataset["table_name"].casefold()
                ]
                for table in matching_tables:
                    try:
                        columns = self.datalayer.get_table_columns(catalog, table)
                        row_count = self.datalayer.count_data_rows(catalog, table)
                    except Exception as ex:
                        self.logger.print_warning(
                            f"Regression suite: could not inspect {catalog}.{table}: "
                            f"{type(ex).__name__}: {ex}"
                        )
                        continue

                    required = (dataset["class_column"], dataset["id_column"])
                    missing = [column for column in required if column not in columns]
                    if missing:
                        self.logger.print_warning(
                            f"Regression suite: skipping {dataset['label']} candidate {catalog}.{table}; "
                            f"missing required columns: {', '.join(missing)}."
                        )
                        continue

                    feature_columns = [
                        column for column in columns
                        if column not in (dataset["class_column"], dataset["id_column"])
                    ]
                    text_columns = [
                        column for column in feature_columns
                        if str(columns[column]).casefold() in Config.TEXT_DATATYPES
                    ]
                    numerical_columns = [
                        column for column in feature_columns if column not in text_columns
                    ]

                    if not feature_columns:
                        self.logger.print_warning(
                            f"Regression suite: skipping {dataset['label']} candidate {catalog}.{table}; "
                            "no feature columns were found."
                        )
                        continue

                    return {
                        **dataset,
                        "catalog": catalog,
                        "table": table,
                        "text_columns": text_columns,
                        "numerical_columns": numerical_columns,
                        "row_count": row_count,
                    }
        finally:
            self.set_data_catalog(original_catalog)

        return None

    def _build_regression_suite_config(self, base_config_params: dict, dataset: dict) -> dict:
        config_params = copy.deepcopy(base_config_params)
        connection = config_params["connection"]
        connection.data_catalog = dataset["catalog"]
        connection.data_table = dataset["table"]
        connection.class_column = dataset["class_column"]
        connection.id_column = dataset["id_column"]
        connection.data_text_columns = list(dataset["text_columns"])
        connection.data_numerical_columns = list(dataset["numerical_columns"])

        config_params["mode"].category_text_columns = []
        config_params["debug"].data_limit = dataset["row_count"]

        base_model_name = config_params["io"].model_name or "regression"
        config_params["io"].model_name = f"{base_model_name}_{dataset['slug']}"
        config_params["name"] = f"{config_params['name']}_{dataset['slug']}"
        return config_params

    def get_regression_suite_configs(self, base_config_params: dict) -> tuple[list[tuple[str, dict]], list[str]]:
        preferred_catalog = base_config_params["connection"].data_catalog
        configs = []
        missing = []

        for dataset in self.REGRESSION_SUITE_DATASETS:
            resolved = self._find_regression_suite_dataset(dataset, preferred_catalog)
            if resolved is None:
                missing.append(dataset["label"])
                continue
            configs.append((dataset["label"], self._build_regression_suite_config(base_config_params, resolved)))

        return configs, missing

    def run_regression_suite(self, base_config_params: dict, output: Output) -> None:
        """Run known regression datasets sequentially and isolate dataset-level failures."""
        with output, self.logger.capture_console_output():
            self.logger.print_info("Regression suite: resolving Iris and Breast Cancer datasets.")
            self.logger.print_info(
                "Regression suite: reclassification/mispredicted output is disabled for suite runs."
            )
            configs, missing = self.get_regression_suite_configs(base_config_params)
            for label in missing:
                self.logger.print_warning(
                    f"Regression suite: {label} was not found in the accessible SQL catalogs and will be skipped."
                )

        completed = []
        failed = []
        for index, (label, config_params) in enumerate(configs, start=1):
            connection = config_params["connection"]
            with output, self.logger.capture_console_output():
                self.logger.print_info(
                    f"Regression suite [{index}/{len(configs)}]: starting {label} "
                    f"from {connection.data_catalog}.{connection.data_table}."
                )

            try:
                result = self.run_classifier(
                    config_params=config_params,
                    output=output,
                    set_rerun=False,
                    regression_suite=True,
                )
                if result:
                    completed.append(label)
                    with output, self.logger.capture_console_output():
                        self.logger.print_info(f"Regression suite: completed {label}.")
                else:
                    failed.append(label)
                    with output, self.logger.capture_console_output():
                        self.logger.print_warning(f"Regression suite: {label} returned no result.")
            except SystemExit as ex:
                failed.append(label)
                with output, self.logger.capture_console_output():
                    self.logger.print_error(f"Regression suite: {label} aborted: {ex}")
            except Exception as ex:
                failed.append(label)
                with output, self.logger.capture_console_output():
                    self.logger.print_error(f"Regression suite: {label} failed: {type(ex).__name__}: {ex}")

        with output, self.logger.capture_console_output():
            self.logger.print_info(
                "Regression suite summary: "
                f"{len(completed)} completed, {len(failed)} failed, {len(missing)} missing."
            )

        self.widgets.set_rerun()

    def display_gui(self) -> None:
        self.widgets.display_gui()
        
        

def main():
    load_dotenv()
    gui = GUIHandler()
    gui.display_gui()


if __name__ == "__main__":
    main()
else:
    if os.path.isfile(env_path):
        load_dotenv(env_path)  # take environment variables from .env
    else:
        raise FileNotFoundError(
            errno.ENOENT, 
            os.strerror(errno.ENOENT), 
            env_path
        )

