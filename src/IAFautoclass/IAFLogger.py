from datetime import datetime
import sys
import pandas
import terminal
from typing import Protocol

from IAFHandler import Model, PredictionsHandler

# Using Protocol to simplify imports
class Config(Protocol):
    def __str__(self) -> str:
        print("Config Protocol")

class IAFLogger(terminal.Logger):

    # TODO: In terminal we can set more levels than our program. We have verbose which means "show info too", so has been
    # translated into quiet (IE "not verbose"). This can be tweaked, IE:
    # * verbose = show verbose errors, warnings and info
    # * quiet = don't show info

    def __init__(self, quiet: bool = True, progress: tuple = None):
        # Setup any widgets to report to
        self.widgets = {}
        if progress is not None:
            self.widgets["progress_bar"] = progress[0]
            self.widgets["progress_label"] = progress[1]

        terminal.Logger.__init__(self, quiet=quiet)
        

    def print_welcoming_message(self, config: Config, date_now: datetime) -> None:
        
        # Print welcoming message
        self.print_unformatted("\n *** WELCOME TO IAF AUTOMATIC CLASSIFICATION SCRIPT ***\n")
        self.print_unformatted("Execution started at: {0:>30s} \n".format(str(date_now)))

        # Print configuration settings
        self.print_unformatted(config)

    def get_table_format(self) -> str:
        return "{0:>4s}-{1:<6s}{2:>6s}{3:>8s}{4:>8s}{5:>11s} {6:<30s}"

    def print_table_row(self, items: list[str], divisor: str = None) -> None:
        """ Prints a row with optional divisor"""
        self.print_unformatted(self.get_table_format().format(*items))

        if divisor is not None:
            self.print_unformatted(divisor*45)

    def print_components(self, component, components, exception = None) -> None:
        if exception is None:
            self.print_unformatted(f"{component} n_components is set to {components}")

        else:
            self.warn(f"{component} could not be used with n_components = {components}: {exception}")

    def print_formatted_info(self, message: str) -> None:
        self.print_unformatted(f" -- {message} --")

    def print_info(self, *args) -> None:
        self.print_unformatted(' '.join(args))

    def print_progress(self, message: str = None, percent: float = None) -> None:
        if message is not None:
            #self.print_unformatted(message)
            self._set_widget_value("progress_label", message)

        if percent is not None:
            #self.print_unformatted(f"{percent*100}% completed")
            self._set_widget_value("progress_bar", percent)

    def print_error(self, *args) -> None:
        self.error(' '.join(args))

    def print_warning(self, *args) -> None:
        self.warn(' '.join(args))

    def print_exit_error(self, *args) -> None:
        self.print_error(' '.join(args))
        self._set_widget_value("progress_bar", 1.0)

    def _set_widget_value(self, widget: str, value) -> None:
        if widget in self.widgets:
            self.widgets[widget] = value

    def investigate_dataset(self, dataset: pandas.DataFrame, class_column: str, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        if self._enable_quiet:
            # This function should only run if info can be shown
            return False
        self._set_widget_value("progress_label", "Investigate dataset (see console)")
        self.print_dataset_investigation(dataset, class_column, show_class_distribution)

        if show_statistics:
            #Give some statistical overview of the training material
            self.show_statistics_on_dataset(dataset)

        return True

    # Investigating dataset -- make some printouts to standard output
    def print_dataset_investigation(self, dataset, class_column: str, show_class_distribution: bool = True):

        try: 
            self.start("Looking at dataset:")
            # 1. shape
            self.print_unformatted("Shape:", dataset.shape)
            
            # 2. head
            self.print_unformatted("Head:",dataset.head(20))
            
            # 3. Data types
            self.print_unformatted("Datatypes:",dataset.dtypes)
            
            if show_class_distribution:
                # 4. Class distribution
                self.print_unformatted("Class distribution: ")
                self.print_unformatted(dataset.groupby(dataset[class_column]).size()) 
        except Exception as e:
            self.print_warning(f"An error occured in investigate_dataset: {str(e)}")

        self.end()

    
    # Show statistics in standard output
    def show_statistics_on_dataset(self, dataset):

        # 1. Descriptive statistics
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.describe(datetime_is_numeric = True)
        self.print_unformatted("Description:")
        self.print_unformatted(description)

        # 2. Correlations
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.corr('pearson')
        self.print_unformatted("Correlation between attributes:")
        self.print_unformatted(description)

        # 3. Skew
        skew = dataset.skew()
        self.print_unformatted("Skew of Univariate descriptions")
        self.print_unformatted(skew, "\n")

    def print_classification_report(self, report: dict, model: Model, num_features: int):
        self.start(f"Classification report for {model.algorithm.name}/{model.preprocess.name} with #features: {num_features}")
        for key, value in report.items():
            self.print_unformatted(f"{key}: {value}")

        self.end()

    def is_quiet(self) -> bool:
        return self._enable_quiet

    def print_query(self, type: str, query: str) -> None:
        message = f"Query for {type}: {query}"
        self.debug(message)

    # This is to avoid the annoying "info:" in front of all lines. Debug/warning/Error should still use the normal
    def print_unformatted(self, *args) -> None:
        if self._enable_quiet:
            return self
        
        return self.writeln('unformatted', *args)

    def print_percentage_checked(self, text: str, old_percent, percent_checked) -> None:
        if self._enable_quiet or old_percent <= percent_checked:
            return

        print(f"{text}: " + str(percent_checked) + " %", end='\r')

    #
    def print_training_rates(self, ph: PredictionsHandler) -> None:
        if not ph.could_predict_proba:
            return
        
        mean, std = ph.get_rates(as_string = False)
        self.print_info("Sample prediction probability rate, mean: {0:5.3f}, std.dev: {1:5.3f}".format(mean, std))
    

    # Makes sure the GUI isn't left hanging if exceptions crash the program
    def abort_cleanly(self, message: str) -> None:
        self.print_exit_error(message)
        sys.exit("Program aborted.")


def main():
    config = Config()
    date_now = datetime.now()
    myLogger = IAFLogger(quiet=False)

    myLogger.print_welcoming_message(config=config, date_now=date_now)

    


if __name__ == "__main__":
    main()
