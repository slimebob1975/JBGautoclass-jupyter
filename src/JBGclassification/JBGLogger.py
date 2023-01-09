from datetime import datetime
import sys
from numpy import ndarray
import pandas as pd
import terminal
from typing import Protocol
import IPython.display

from JBGHandler import Model, PredictionsHandler

# Using Protocol to simplify imports
class Config(Protocol):
    def __str__(self) -> str:
        print("Config Protocol")

class JBGLogger(terminal.Logger):

    # TODO: In terminal we can set more levels than our program. We have verbose which means "show info too", so has been
    # translated into quiet (IE "not verbose"). This can be tweaked, IE:
    # * verbose = show verbose errors, warnings and info
    # * quiet = don't show info

    def __init__(self, quiet: bool = True, progress: tuple = None, in_terminal: bool = False):
        # Setup any widgets to report to
        self.widgets = {}
        if progress is not None:
            self.widgets["progress_bar"] = progress[0]
            self.widgets["progress_label"] = progress[1]

        self.in_terminal = in_terminal
        terminal.Logger.__init__(self, quiet=quiet)
        

    def set_enable_quiet(self, enable_quiet: bool) -> None:
        """ Sets quiet after init """
        self._enable_quiet = enable_quiet
    
    def print_welcoming_message(self, config: Config, date_now: datetime) -> None:
        
        # Print welcoming message
        self.print_unformatted("\n *** WELCOME TO JBG AUTOMATIC CLASSIFICATION SCRIPT ***\n")
        self.print_unformatted("Execution started at: {0:>30s} \n".format(str(date_now)))

        # Print configuration settings
        self.print_unformatted(config)

    def get_table_format(self) -> str:
        return "{0:>4s}{1:>4s}{2:<6s}{3:>6s}{4:>8s}{5:>8s}{6:>8s}{7:>11s} {8:<30s}"

    def print_table_row(self, items: list[str], divisor: str = None) -> None:
        """ Prints a row with optional divisor"""
        self.print_unformatted(self.get_table_format().format(*items))

        if divisor is not None:
            self.print_unformatted(divisor*65)

    def print_components(self, component, components, exception = None) -> None:
        if exception is None:
            self.print_unformatted(f"{component} n_components is set to {components}")

        else:
            self.warn(f"{component} could not be used with n_components = {components}: {exception}")

    def print_formatted_info(self, message: str) -> None:
        self.print_unformatted(f" -- {message} --")

    def print_info(self, *args) -> None:
        self.print_unformatted(' '.join(args))

    def print_always(self, *args) -> None:
        """ This ignores the quiet flag and should always be printed out """
        
        return self.writeln("always", *args)

    def print_prediction_report(self, evaluation_data: str, accuracy_score: float, confusion_matrix: ndarray, class_labels: list, \
        classification_matrix: dict) -> None:
        """ Printing out info about the prediction"""
        self.print_progress(message="Evaluate predictions")

        self.print_always(f"Evaluation performed with evaluation data: " + evaluation_data)
        self.print_always(f"Accuracy score for evaluation data: {accuracy_score}")
        self.display_matrix(f"Confusion matrix for evaluation data:", pd.DataFrame(confusion_matrix, columns=class_labels, index=class_labels))
        self.display_matrix(f"Classification matrix for evaluation data:", pd.DataFrame.from_dict(classification_matrix).transpose())

    def print_progress(self, message: str = None, percent: float = None) -> None:
        if message is not None:
            if self.in_terminal:
                self.print_unformatted(message)
            self._set_widget_value("progress_label", message)

        if percent is not None:
            if self.in_terminal:
                self.print_unformatted(f"{percent*100:.0f}% completed")
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
            self.widgets[widget].value = value

    def investigate_dataset(self, dataset: pd.DataFrame, class_column: str, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
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
            #self.print_unformatted("Head:",dataset.head(20))
            self.display_matrix("Head:", dataset.head(20))
            
            # 3. Data types
            #self.print_unformatted("Datatypes:",dataset.dtypes)
            self.display_matrix("Datatypes:", pd.DataFrame(dataset.dtypes, columns=["Datatype"]))
            
            if show_class_distribution:
                # 4. Class distribution
                #self.print_unformatted("Class distribution: ")
                #self.print_unformatted(dataset.groupby(dataset[class_column]).size()) 
                self.display_matrix("Class distribution: ", pd.DataFrame(dataset.groupby(dataset[class_column]).size(), \
                    columns=["Support"]))
        except Exception as e:
            self.print_warning(f"An error occured in investigate_dataset: {str(e)}")

        self.end()

    
    # Show statistics in standard output
    def show_statistics_on_dataset(self, dataset):

        # 1. Descriptive statistics
        pd.set_option('display.width', 100)
        pd.set_option('display.precision', 3)
        description = dataset.describe(datetime_is_numeric = True)
        #self.print_unformatted("Description:")
        #self.print_unformatted(description)
        self.display_matrix("Description:", pd.DataFrame(description))
        
        # 2. Correlations
        pd.set_option('display.width', 100)
        pd.set_option('display.precision', 3)
        try:
            correlations = dataset.corr('pearson', numeric_only=False)
        except Exception as ex:
            correlations = dataset.corr('pearson')
        #self.print_unformatted("Correlation between attributes:")
        #self.print_unformatted(description)
        self.display_matrix("Correlation between attributes:", pd.DataFrame(correlations))

        # 3. Skew
        try:
            skew = dataset.skew(numeric_only=False)
        except Exception as ex:
            skew = dataset.skew()
        #self.print_unformatted("Skew of Univariate descriptions")
        #self.print_unformatted(skew, "\n")
        self.display_matrix("Skew of Univariate descriptions:", pd.DataFrame(skew, columns=["Skew"]))


    def print_classification_report(self, report: dict, model: Model, num_features: int):
        """ Should only be printed if verbose """
        if self._enable_quiet:
            return self

        self.start(f"Classification report for {model.preprocess.name}-{model.reduction.name}-{model.algorithm.name} with #features: {num_features}")
        for key, value in report.items():
            self.print_unformatted(f"{key}: {value}")

        self.end()

    def is_quiet(self) -> bool:
        return self._enable_quiet

    def print_query(self, type: str, query: str) -> None:
        message = f"Query for {type}: {query}"
        self.debug(message)

    def print_dragon(self, exception: Exception) -> None:
        """ Type of Unhandled Exceptions, to handle them for the future """
        message = f"Here be dragons: {type(exception)}"
        self.print_warning(message)
    
    def print_result_line(self, reduction_name: str, algorithm_name: str, preprocessor_name: str, num_features: float, \
        temp_score, temp_stdev, test_score, t, failure: str, ending: str = '\n') -> None:
        """ Prints information about a specific result line """
        if self._enable_quiet:
            return self

        print(
            "{0:>4s}-{1:>4s}-{2:<6s}{3:6d}{4:8.3f}{5:8.3f}{6:8.3f}{7:11.3f} {8:<30s}".
                format(reduction_name,algorithm_name,preprocessor_name,num_features,temp_score,temp_stdev,test_score,t,failure),
                end=ending
        )
        
    def clear_last_printed_result_line(self):
        print(" "*200, end='\r')

    # This is to avoid the annoying "info:" in front of all lines. Debug/warning/Error should still use the normal
    def print_unformatted(self, *args) -> None:
        if self._enable_quiet:
            return self
        
        return self.writeln('unformatted', *args)

    def print_percentage(self, text: str, percent: float, old_percent: float = 0) -> None:
        if self._enable_quiet or (old_percent > 0 and old_percent >= percent):
            return

        print(f"{text}: {percent} %", end='\r')

    def print_linebreak(self) -> None:
        """ Important after using \r for updates """
        if self._enable_quiet:
            return self

        print("\n")
    #
    def print_training_rates(self, ph: PredictionsHandler) -> None:
        if not ph.could_predict_proba:
            return
        
        mean, std = ph.get_rates(as_string = False)
        self.print_info("Sample prediction probability rate, mean: {0:5.3f}, std.dev: {1:5.3f}".format(mean, std))
    
    def display_matrix(self, title: str, matrix: pd.DataFrame) -> None:
        self.print_unformatted(title)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 2)
        IPython.display.display(matrix)

    # Makes sure the GUI isn't left hanging if exceptions crash the program
    def abort_cleanly(self, message: str) -> None:
        self.print_exit_error(message)
        sys.exit("Program aborted.")


def main():
    config = Config()
    date_now = datetime.now()
    myLogger = JBGLogger(quiet=False)

    myLogger.print_welcoming_message(config=config, date_now=date_now)

    


if __name__ == "__main__":
    main()
