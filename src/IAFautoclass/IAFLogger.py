from datetime import datetime
import pandas
import terminal
import Config

class IAFLogger:

    # TODO: In terminal we can set more levels than our program. We have verbose which means "show info too", so has been
    # translated into quiet (IE "not verbose"). This can be tweaked, IE:
    # * verbose = show verbose errors, warnings and info
    # * quiet = don't show info

    def __init__(self, quiet: bool = True, progress: tuple = None):
        # Setup any widgets to report to
        self.widgets = {}

        if progress is not None:
            self.widget["progress_bar"] = progress[0]
            self.widget["progress_label"] = progress[1]
        
        self.logger = terminal.Logger(quiet=quiet, verbose=False, indent=0) # Default indent and verbose

    def print_welcoming_message(self, config: Config.Config, date_now: datetime) -> None:
        
        # Print welcoming message
        self.logger.info("\n *** WELCOME TO IAF AUTOMATIC CLASSIFICATION SCRIPT ***\n")
        self.logger.info("Execution started at: {0:>30s} \n".format(str(date_now)))

        # Print configuration settings
        self.logger.info(config)

    def print_components(self, component, components, exception = None) -> None:
        if exception is None:
            self.logger.info(f"{component} n_components is set to {components}")

        else:
            self.logger.warn(f"{component} could not be used with n_components = {components}: {exception}")

    def print_formatted_info(self, message: str) -> None:
        self.logger.info(f" -- {message} --")

    def print_info(self, *args) -> None:
        self.logger.info(' '.join(args))

    def print_progress(self, message: str = None, percent: float = None) -> None:
        if message is not None:
            self.logger.info(message)
            self._set_widget_value("progress_label", message)

        if percent is not None:
            self.logger.info(f"{percent*100}% completed")
            self._set_widget_value("progress_bar", percent)

    def print_error(self, *args) -> None:
        self.logger.error(' '.join(args))

    def print_warning(self, *args) -> None:
        self.logger.warning(' '.join(args))

    def print_exit_error(self, *args) -> None:
        self.print_error(' '.join(args))
        self._set_widget_value("progress_bar", 1.0)

    def _set_widget_value(self, widget: str, value) -> None:
        if widget in self.widgets:
            self.widgets[widget] = value

    def investigate_dataset(self, dataset: pandas.DataFrame, show_class_distribution: bool = True, show_statistics: bool = True) -> bool:
        if self.logger._enable_quiet:
            # This function should only run if info can be shown
            return False
        self._set_widget_value("progress_label", "Investigate dataset (see console)")
        self.print_dataset_investigation(dataset, show_class_distribution)

        if show_statistics:
            #Give some statistical overview of the training material
            self.show_statistics_on_dataset(dataset)

        return True

    # Investigating dataset -- make some printouts to standard output
    def print_dataset_investigation(self, dataset, show_class_distribution = True):

        try:
            self.logger.info("Looking at dataset:")
            # 1. shape
            self.logger.info("Shape:", dataset.shape)
        except Exception as e:
           self.logger.warning(f"An error occured in investigate_dataset: {str(e)}")
        try:
            # 2. head
            self.logger.info("Head:",dataset.head(20))
        except Exception as e:
            self.logger.warning(f"An error occured in investigate_dataset: {str(e)}")
        try:
            # 3. Data types
            self.logger.info("Datatypes:",dataset.dtypes)
        except Exception as e:
            self.logger.warning(f"An error occured in investigate_dataset: {str(e)}")
        if show_class_distribution:
            try:
            # 4. Class distribution
                self.logger.info("Class distribution: ")
                self.logger.info(dataset.groupby(dataset.columns[0]).size())
            except Exception as e:
                self.logger.warning(f"An error occured in investigate_dataset: {str(e)}")

    
    # Show statistics in standard output
    def show_statistics_on_dataset(self, dataset):

        # 1. Descriptive statistics
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.describe(datetime_is_numeric = True)
        self.logger.info("Description:")
        self.logger.info(description)

        # 2. Correlations
        pandas.set_option('display.width', 100)
        pandas.set_option('precision', 3)
        description = dataset.corr('pearson')
        self.logger.info("Correlation between attributes:")
        self.logger.info(description)

        # 3. Skew
        skew = dataset.skew()
        self.logger.info("Skew of Univariate descriptions")
        self.logger.info(skew, "\n")

    def print_classification_report(self, report: dict, model: tuple, num_features: int):
        self.logger.start(f"Classification report for {model[0].name}/{model[1].name} with #features: {num_features}")
        for key, value in report.items():
            self.logger.info(f"{key}: {value}")

        self.logger.end()

def main():
    config = Config.Config()
    date_now = datetime.now()
    myLogger = IAFLogger(quiet=False)

    myLogger.print_welcoming_message(config=config, date_now=date_now)

    


if __name__ == "__main__":
    main()
