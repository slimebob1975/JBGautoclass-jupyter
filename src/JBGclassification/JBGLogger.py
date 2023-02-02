from datetime import datetime
from functools import partial
import sys
from numpy import ndarray
import pandas as pd
import terminal
from typing import Protocol, Union
import IPython.display
import ipywidgets as widgets

from Helpers import html_wrapper, print_html, save_matrix_as_csv

# Using Protocol to simplify imports
class Config(Protocol):
    def to_dict(self) -> dict[str, Union[str, dict[str, str]]]:
        """ Gets all subdicts as dicts """

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
        self.inline_bars = {}
    
    def initiate_progress(self, number_of_tasks: int):
        """ Initiate the progress counter """
        self.progress = 0
        self.number_of_tasks = number_of_tasks
        self.percentage = 1/number_of_tasks

    def update_progress(self, percent: float = None, message: str = None) -> float:
        """ Tracks the progress through the run """
        if percent is None:
            self.progress += self.percentage
        else:
            self.progress += percent

        if message is None:
            self.print_progress(percent = self.progress)
        else:
            self.print_progress(message=message, percent = self.progress)

        return self.progress
    
    def get_minor_percentage(self, minor_tasks: int) -> float:
        """ Given a number of minor tasks, returns what percentage each is worth """
        return self.percentage/float(minor_tasks)

    
    def set_enable_quiet(self, enable_quiet: bool) -> None:
        """ Sets quiet after init """
        self._enable_quiet = enable_quiet
    
    def print_welcoming_message(self, config: Config, date_now: datetime) -> None:
        """ Only when show info is True, this prints out information about the config and run """
        # Print welcoming message
        if self.in_terminal:
            title = "\n *** WELCOME TO JBG AUTOMATIC CLASSIFICATION SCRIPT ***\n"
            self.print_unformatted(title)
        
        self.print_unformatted("Execution started at: {0:>30s} \n".format(str(date_now)))

        self.print_config_settings(config)

    def print_code(self, key: str, code: str) -> None:
        """ Prints out a text with a (in output) code-tagged end """

        if not self.in_terminal:
            key = f"<em>{key}</em>"
            code = f"<code>{code}</code>"

        else:
            code = f"\n\t{code}"
        
        self.print_unformatted(f"{key}: {code}")

    def print_key_value_pair(self, key: str, value, print_always: bool = False) -> None:
        """ Prints out '<key>: <value>'"""

        if not self.in_terminal:
            key = f"<em>{key}</em>"
        
        printed = f"{key}: {value}"

        if print_always:
            self.print_always(printed)
        else:
            self.print_unformatted(printed)

    def print_config_settings(self, config: Config) -> None:
        """ Prints out the dicts with various information as matrixes """
        config_dict = config.to_dict()

        h2 = partial(html_wrapper, "h2")
        config_title = config_dict.pop("title")
        self.print_unformatted(config_title, html_function=h2)
        self.print_unformatted("Headers marked with (*) are optional in the config")

        # config_dict is now a dictionary of string keys and dict[str, str] containing the data of each subclass
        for index, subclass_dict in enumerate(config_dict.values()):
            subclass_title = f"{index + 1}. {subclass_dict.pop('title')}"
            df = pd.DataFrame.from_dict(data=subclass_dict, orient="index", columns=[""])
            self.display_matrix(subclass_title, df, print_always=False)


    def print_components(self, component, components, exception = None) -> None:
        if exception is None:
            self.print_unformatted(f"{component} n_components is set to {components}")

        else:
            self.print_warning(f"{component} could not be used with n_components = {components}: {exception}")

    def print_formatted_info(self, message: str) -> None:
        """ Prints information with a bit of markup """
        if self.in_terminal:
            message = f" -- {message} --"
        else:
            message = f"<b>{message}</b"
        
        self.print_unformatted(message)

    def print_info(self, *args, **kwargs) -> None:
        """ Wrapper that either uses print_unformatted or print_always, depending on the kwarg print_always"""
        print_always = kwargs.pop("print_always", False)
        if print_always:
            self.print_always(*args, **kwargs)
        else:
            self.print_unformatted(*args, **kwargs)

    
    def print_prediction_results(self, results: dict) -> None:
        """ Prints a nicely formatted query and the number of rows """
        if self._enable_quiet:
            return self

        if self.in_terminal:
            query = f"\n\n{results['query']}"
        else:
            query = f"<pre>{results['query']}</pre>"
            
        line = f"Added {results['row_count']} rows to prediction table. Get them with SQL query: {query}"
        self.print_unformatted(line)


    def print_prediction_info(self, predictions: dict, rates: tuple = None) -> None:
        """ Info before trying to save predictions """
        if self._enable_quiet:
            return self

        h3 = partial(html_wrapper, "h3")
        self.print_unformatted("Predictions for the unknown data", html_function=h3)
        self.print_unformatted(f"Predictions: {predictions}")
        
        if rates:
            mean, std = rates
            self.print_unformatted("Sample prediction probability rate, mean: {0:5.3f}, std.dev: {1:5.3f}".format(mean, std))
        
    def print_task_header(self, title: str) -> None:
        """ Prints an h2 for each task"""
        if self._enable_quiet:
            return self    
        h2 = partial(html_wrapper, "h2")
        self.print_unformatted(title, html_function=h2)

    def print_always(self, *args, **kwargs) -> None:
        """ This ignores the quiet flag and should always be printed out """
        if self.in_terminal:
            return self.writeln("always", *args)
        
        print_html(*args, **kwargs)

        return self


    def print_prediction_report(self, 
        accuracy_score: float, 
        confusion_matrix: ndarray, 
        class_labels: list,
        classification_matrix: dict, 
        sample_rates: tuple[str, str] = None) -> None:
        """ Printing out info about the prediction"""
        self.print_progress(message="Evaluate predictions")
        evaluation_dict = {}
        if sample_rates and not self._enable_quiet:
            rate_string = "Mean: {0:5.3f}, std.dev: {1:5.3f}".format(*sample_rates)
            evaluation_dict["Sample prediction probability rate"] = rate_string
        evaluation_dict["Accuracy score for evaluation data"] = str(accuracy_score)
    
        self.display_matrix("Evaluation information", pd.DataFrame.from_dict(data=evaluation_dict, orient="index", columns=[""]))
        self.display_matrix(f"Confusion matrix for evaluation data", pd.DataFrame(confusion_matrix, columns=class_labels, index=class_labels))
        self.display_matrix(f"Classification matrix for evaluation data", pd.DataFrame.from_dict(classification_matrix).transpose())

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """ Updates the progress bar and prints out a value in the terminal of no bar"""
        if message is not None:
            if self.in_terminal:
                self.print_unformatted(message)

            self._set_widget_value("progress_label", message)

        if percent is not None:
            if self.in_terminal:
                self.print_unformatted(f"{percent*100:.0f}% completed")
            self._set_widget_value("progress_bar", percent)

    def print_error(self, *args, **kwargs) -> None:
        if self.in_terminal:
            return self.error(*args)

        print_html(*args, **kwargs)

        return self
        

    def print_warning(self, *args, **kwargs) -> None:
        if self.in_terminal:
            return self.warn(*args)

        print_html(*args, **kwargs)

        return self

    def print_exit_error(self, *args) -> None:
        self.print_error(*args)
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
            h2 = partial(html_wrapper, "h2")
            self.print_unformatted("Looking at dataset", html_function=h2)
            # 1. shape
            self.print_unformatted("Shape", dataset.shape)
            
            # 2. head
            self.display_matrix("Head", dataset.head(20))
            
            # 3. Data types
            self.display_matrix("Datatypes", pd.DataFrame(dataset.dtypes, columns=["Datatype"]))
            
            if show_class_distribution:
                # 4. Class distribution
                self.display_matrix("Class distribution", pd.DataFrame(dataset.groupby(dataset[class_column]).size(), \
                    columns=["Support"]))
        except Exception as e:
            self.print_warning(f"An error occured in investigate_dataset: {str(e)}")

    
    # Show statistics in standard output
    def show_statistics_on_dataset(self, dataset):

        # 1. Descriptive statistics
        pd.set_option('display.width', 100)
        pd.set_option('display.precision', 3)
        description = dataset.describe(datetime_is_numeric = True)
        self.display_matrix("Description", pd.DataFrame(description))
        
        # 2. Correlations
        pd.set_option('display.width', 100)
        pd.set_option('display.precision', 3)
        try:
            correlations = dataset.corr('pearson', numeric_only=False)
        except Exception:
            correlations = dataset.corr('pearson')
        self.display_matrix("Correlation between attributes", pd.DataFrame(correlations))

        # 3. Skew
        try:
            skew = dataset.skew(numeric_only=False)
        except Exception:
            skew = dataset.skew()
        self.display_matrix("Skew of Univariate descriptions", pd.DataFrame(skew, columns=["Skew"]))


    def is_quiet(self) -> bool:
        return self._enable_quiet

    def print_query(self, type: str, query: str) -> None:
        if self.in_terminal:
            message = f"Query for {type}: {query}"
            self.debug(message)
        else:
            message = f"Query for <em>{type}</em>: <pre>{query}</pre>"
            self.print_unformatted(message)
        

    def print_correcting_mispredicted(self, new_class: str, index: int, query: str) -> None:
        """ Prints out notice about correcting mispredicted class in class_catalog """
        self.print_unformatted(f"Changing data row {index} to {new_class}: ")
        self.print_query("mispredicted", query)

    def print_dragon(self, exception: Exception) -> None:
        """ Type of Unhandled Exceptions, to handle them for the future """
        message = f"Here be dragons: {type(exception)}"
        self.print_warning(message)
    
    def print_result_line(self, result: list, ending: str = "\r") -> None:
        #def print_result_line(self, reduction_name: str, algorithm_name: str, preprocessor_name: str, num_features: float, \
        #temp_score, temp_stdev, test_score, t, failure: str, ending: str = '\n') -> None:
        """ Prints information about a specific result line """
        if self._enable_quiet:
            return self

        result = '\t'.join([str(x) for x in result])
        print(f"{result}", end=ending)
        
    def clear_last_printed_result_line(self):
        print(" "*200, end='\r')

    # This is to avoid the annoying "info:" in front of all lines. Debug/warning/Error should still use the normal
    def print_unformatted(self, *args, **kwargs) -> None:
        if self._enable_quiet:
            return self
        
        if self.in_terminal:
            return self.writeln('unformatted', *args)

        print_html(*args, **kwargs)

        return self

    def anaconda_debug(self, message: str):
        """ This is specifically for calls that will not get redirected to an Output """
        return self.debug(message)
    

    def start_inline_progress(self, key: str, description: str, final_count: int, tooltip: str) -> None:
        """ This will overwrite any prior bars with the same key """
        if self._enable_quiet:
            return

        self.inline_bars[key] = {
            "final_count": float(final_count)
        }
        self.inline_bars[key]["bar"] = widgets.FloatProgress(
            value= 0.0,
            min= 0.0,
            max= 1.0,
            description= f"{description}:",
            bar_style= "",
            style={"bar_color": "#C0C0C0"},
            orientation= "horizontal",
            description_tooltip= tooltip
        )

        self.inline_bars[key]["percent_label"] = widgets.HTML("0%")
        self.inline_bars[key]["percent_label"].add_class("inline_percent")
    
        if self.in_terminal:
            self.print_formatted_info(description)
            return

        progress_box = widgets.HBox([self.inline_bars[key]["bar"], self.inline_bars[key]["percent_label"]])
        IPython.display.display(progress_box)

    
    def update_inline_progress(self, key: str, current_count: int, terminal_text: str) -> None:
        """ Updates progress bars within the script"""
        
        if self._enable_quiet:
            return
        
        information = self.inline_bars[key]
        old_percent = information["bar"].value
        float_percent = float(current_count)/information["final_count"]
        
        if (old_percent > 0 and old_percent >= float_percent):
            return

        information["bar"].value = float_percent
        percent = round(100.0*float_percent)
            
        if self.in_terminal:
            print(f"{terminal_text}: {percent} %", end='\r')
            return

        self.inline_bars[key]["percent_label"].value = f"{percent}%"

    
    def end_inline_progress(self, key: str, set_100: bool = True) -> None:
        """ Ensures that any loose ends are tied up after the progress is done """
        if self._enable_quiet:
            return
        if self.in_terminal:
            print("\n") # terminal progress uses \r, so this is important to end it
            return

        if set_100:
            self.inline_bars[key]["bar"].value = 1 # Since it's completed
            self.inline_bars[key]["percent_label"].value = "100%"
            
    def parse_dataset_progress(self, key: str, num_lines: int, row_count: int, ) -> None:
        """ Groups the start of the parse_dataset functions print-outs """
        self.start_inline_progress(key, "Fetching data", row_count, "Percent data fetched of available")
        self.update_inline_progress(key, num_lines, "Data fetched of available")
        
    
    def print_test_performance(self, listOfResults: list, cross_validation_filepath: str) -> None:
        """ 
            Prints out the matrix of model test performance, given a list of results
            Both to screen and CSV
        """
        resultsMatrix = pd.DataFrame(listOfResults,
            columns=["Preprocessor","Feature Reduction","Algorithm","Components","Mean cv","Stdev.","Test data","Elapsed Time","Exception"])
        resultsMatrix.sort_values(by = ["Test data","Mean cv","Stdev."], axis = 0, ascending = [False, False, True], \
            inplace = True, ignore_index = True)

        # Save cross validation results to csv file
        save_matrix_as_csv(resultsMatrix, cross_validation_filepath)
        self.display_matrix(f"Model test performance in descending order", resultsMatrix)

    def display_matrix(self, title: str, matrix: pd.DataFrame, print_always: bool = True) -> None:
        """ Prints out a matrix, but only if it's verbose, or print_always is True """
        if self._enable_quiet and not print_always:
            return self
        
        h3 = partial(html_wrapper, "h3")
        
        self.print_always(title, html_function=h3)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', None) # This makes sure no column info gets cut off due to long content
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 2)
        IPython.display.display(matrix)

        if (self.in_terminal):
            print("\n")

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
