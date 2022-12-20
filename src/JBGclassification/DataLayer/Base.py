
from dataclasses import dataclass
import os
from typing import Protocol

from JBGExceptions import DataLayerException
from Config import Config


class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_warning(self, *args) -> None:
        """ print warning """

@dataclass
class DataLayerBase:
    config: Config
    logger: Logger
    validate: bool = True
    """ This is the base class that all DataLayers inherit from """

    def validate_parameters(self) -> None:
        """ Allow child classes to validate parameters they are given
            Raise ValueError if improperly-configurated parameters are found
        """
        return

    def get_config(self) -> Config:
        """ Getter/setter to not directly touch the config """

        if not self.config:
            raise DataLayerException(f"Config is not yet defined")
        
        return self.config

    def update_config(self, updates: dict) -> bool:
        """ Updates the config by running config.update_configuration """

        if isinstance(updates, dict):
            self.config.update_configuration(updates)
            return True

        return False


    def get_connnection(self):
        """ Gets the connection based on the type"""
        raise NotImplementedError

    def can_connect(self, verbose: bool = False) -> bool:
        """ Checks that the connection to the database works """
        con = self.get_connection()
        
        success = con.can_connect()

        if success:
            return True

        if verbose:
            self.logger.print_warning(f"Connection to server failed: {con}")
        
        return False

    def get_trained_models_from_files(self, model_path: str, model_file_extension: str, preface: list = None) -> list:
        """ Get a list of pretrained models (base implementation assumes files)"""
        # TODO: model_path + model_file_extension is Config-based, using constants. Think on this
        models = []

        if preface:
            models = preface
        
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                if file[-len(model_file_extension):] == model_file_extension:
                    models.append(file)

        return models

    def get_catalogs_as_options(self) -> list:
        """ Used in the GUI, to get the databases """
        raise NotImplementedError

    def get_tables_as_options(self) -> list:
        """ Used in the GUI, to get the tables """
        raise NotImplementedError

    def get_id_columns(self, **kwargs) -> list:
        """ Used in the GUI, gets name and type for columns """
        raise NotImplementedError

    def get_data_from_query(self, query:str) -> list:
        """ Parses data returned from data source """
        raise NotImplementedError

    def count_data_rows(self, data_catalog: str, data_table: str) -> int:
        """ Function for counting the number of rows in data to classify """
        raise NotImplementedError

# Main method
def main():
    dl = DataLayerBase(Config, Logger)

# Start main
if __name__ == "__main__":
    main()