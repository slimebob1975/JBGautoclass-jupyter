
from dataclasses import dataclass, field
import os
from typing import Protocol, Union

from IAFExceptions import DataLayerException
from Config import Config


class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_warning(self, *args) -> None:
        """ print warning """

@dataclass
class DataLayerBase:
    config: Config = field(init=False)
    logger: Logger
    """ This is the base class that all DataLayers inherit from """

    def get_config(self) -> Config:
        """ Getter/setter to not directly touch the config """

        if not self.config:
            raise DataLayerException(f"Config is not yet defined")
        
        return self.config

    def update_config(self, updates: Union[dict, Config]) -> bool:
        """ Updates the config
            If updates is a Config, it replaces the current config
            If updates is a dict, it runs config.update_configuration
        """

        if isinstance(updates, Config):
            self.config = updates
            return True

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

    def get_databases(self) -> list:
        """ Used in the GUI, to get the databases """
        raise NotImplementedError

    def get_tables(self) -> list:
        """ Used in the GUI, to get the tables """
        raise NotImplementedError

    def get_id_columns(self, database: str, table: str) -> list:
        """ Gets name and type for columns in specified <database> and <table> """
        raise NotImplementedError

    def get_trained_models_from_files(self, model_path: str, model_file_extension: str) -> list:
        """ Get a list of pretrained models (base implementation assumes files)"""
        # TODO: model_path + model_file_extension is Config-based, using constants. Think on this
        models = []
        for file in os.listdir(model_path):
            if file[-len(model_file_extension):] == model_file_extension:
                models.append(file)

        return models

    def prepare_for_classification(self) -> bool:
        """ Setting up tables or similar things in preparation for the classifictions """
        raise NotImplementedError

# Main method
def main():
    dl = DataLayerBase(Config, Logger)

# Start main
if __name__ == "__main__":
    main()