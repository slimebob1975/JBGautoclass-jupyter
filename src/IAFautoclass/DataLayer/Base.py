
from dataclasses import dataclass, field
from typing import Protocol


class Logger(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def print_info(self, *args) -> None:
        """printing info"""

    def print_warning(self, *args) -> None:
        """ print warning """

class Config(Protocol):
    # Methods to hide implementation of Config
    def is_text_data(self) -> bool:
        """True or False"""
    
    def is_numerical_data(self) -> bool:
        """True or False"""

@dataclass
class DataLayerBase:
    config: Config = field(init=False)
    logger: Logger
    """ This is the base class that all DataLayers inherit from """

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

# Main method
def main():
    dl = DataLayerBase(Config, Logger)

# Start main
if __name__ == "__main__":
    main()