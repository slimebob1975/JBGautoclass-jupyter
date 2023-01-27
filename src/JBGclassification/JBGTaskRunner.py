from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Protocol

from sklearn.pipeline import Pipeline

from JBGExceptions import PredictionsException, ModelException, DatasetException, ConfigException
from JBGHandler import JBGHandler
import Helpers

class Logger(Protocol):
    """ Hides implementation """
    def initiate_progress(self, number_of_tasks: int):
        """ Initiate the progress counter """

    def print_progress(self, message: str = None, percent: float = None) -> None:
        """Printing progress"""

    def update_progress(self, percent: float = None, message: str = None) -> float:
        """ Tracks the progress through the run """

    def print_unformatted(self, *args) -> None:
        """ Prints without info: """

    def abort_cleanly(self, message: str) -> None:
        """ Exits the process """

    def print_formatted_info(self, message: str) -> None:
        """ Printing info with """
    
    def print_info(self, *args) -> None:
        """printing info"""

    def print_error(self, *args) -> None:
        """ Printing error """
    
    def print_code(self, title: str, code: str) -> None:
        """ Prints out a text with a (in output) code-tagged end """

    def print_prediction_results(self, results: dict) -> None:
        """ Prints a nicely formatted query and the number of rows """
    
    def print_prediction_info(self, predictions: dict, rates: tuple = None) -> None:
        """ Info before trying to save predictions """
    
    def print_task_header(self, title: str) -> None:
        """ Prints an h2 for each task"""


class DataLayer(Protocol):
    """To avoid the issue of circular imports, we use Protocols with the defined functions/properties"""
    def get_dataset(self, num_rows: int = None) -> list:
        """ Gets the needed data from the database """

class Config(Protocol):
    """ Hides implementation"""
    def get_text_column_names(self) -> list[str]:
        """ Gets the specified text columns"""
    
    def should_train(self) -> bool:
        """ Returns if this is a training config """

    def should_predict(self) -> bool:
        """ Returns if this is a prediction config """

    def should_display_mispredicted(self) -> bool:
        """ Returns if this is a misprediction config """

    def get_model_filename(self, pwd: Path = None) -> str:
        """ Set the filename based on prediction or training """

    def get_output_filepath(self, type: str, pwd: Path = None) -> str:
        """ Simplifies the path/names of output files """

    def get_model_filename(self, pwd: Path = None) -> str:
        """ Set the name and path of the model file
            The second parameter allows for injecting the path for reliable testing
        """

@dataclass
class TaskRunner:
    """ Runs the tasks of the classification"""
    datalayer: DataLayer
    config: Config
    logger: Logger
    handler: JBGHandler
    
    def run(self, tasks: list):
        """
            Runs through a list of tasks (defined by function-name string).
            Each function returns a dict:
                - "print": string, gets printed using print_formatted_info
                - "early_exit": boolean, returns early if True
                - "progress": string, updates progress with this message
                - Other keys as used by the following task
        """
        number_of_tasks = len(tasks)
        if self.config.should_train():
            number_of_tasks += 1 # train_model__task has an extra share
        self.logger.initiate_progress(number_of_tasks=number_of_tasks)
        
        self.dh = self.handler.get_handler("dataset")
        self.mh = self.handler.get_handler("model")
        self.ph = self.handler.get_handler("predictions")
        
        params = {}
        early_exit = False
        for function_name in tasks:
            func = getattr(self, function_name + "__task")
            try:
                params = func(**params)
                printable = params.pop("print", None)
                if printable is not None:
                    self.logger.print_formatted_info(printable)

                early_exit = params.pop("early_exit", False)
                if early_exit:
                    break

            except Exception as e:
                self.logger.abort_cleanly(str(e))

            progress_message = params.pop("progress", None)

            self.logger.update_progress(message=progress_message)

        return early_exit

    def load_model__task(self) -> dict:
        """ Load the appropriate model """
        self.logger.print_task_header(title="Loading model")
        # Print out what mode we use: training a new model or not
        if self.config.should_train():
            self.mh.load_model()
            return {"print": "We will train our ml model"}
        
        if self.config.should_predict():
            model_path = self.config.get_model_filename()
            if os.path.exists(model_path):
                self.mh.load_model(model_path)
                return {"print": "We will reload and use old model"}
            
            raise ModelException(f"No trained model exists at {model_path}")
        
        raise ModelException("User must choose either to train a new model or use an old one for predictions")
        
    
    def fetch_data__task(self) -> dict:
        """ 
        Fetches the dataset from the database, and if there is none calls for an early exit
        """
        self.logger.print_task_header(title="Fetching data")
        data =  self.datalayer.get_dataset()

        if data is None:
            return {"early_exit": True}

        return {"data": data}


    def load_dataset__task(self, data: list) -> dict:
        """ Validates the data and loads it into the datset handler """
        self.logger.print_task_header(title="Loading dataset")
        try:
            self.dh.load_data(data)
        except Exception as e:
            raise DatasetException(f"Processing dataset failed: {e}")

        return {}

    
    def separate_dataset__task(self) -> dict:
        """ 
            Separates the known from unknown data, and splits into training/test sets
        """
        self.logger.print_task_header(title="Separate dataset into known and unknown")
        # Separate data with know classifications from data with unknown class
        self.dh.separate_dataset()
        
        # Split data in training and test sets
        self.dh.split_dataset_for_training_and_validation()

        return {}

    
    def convert_to_numbers__task(self) -> dict:
        """ Text and categorical variables must be converted to numbers """
        self.logger.print_task_header(title="Convert text to numbers")
        
        self.logger.print_progress(message="Convert dataset to numbers only")
        
        self.mh.model.update_field(
            field="text_converter",
            value=self.dh.convert_text_and_categorical_features(self.mh.model)
        )

        return {}


    def train_model__task(self) -> dict:
        """ 
            Check algorithms for best model and train that model. 
            K-value should be 10 or below.
            Or just use the model previously trained.
        """
        self.logger.print_task_header(title="Train model")
        
        # NOTICE: This major task uses another progressbar share inside DatasetHandler.spot_check_machine_learning_models,
        # so the number of progress bar shares will be the total number of "Major task":s + 1.
        try:
            output_filename = self.config.get_output_filepath("cross_validation")
            self.logger.print_progress(message="Check and train algorithms for best model")
            self.logger.print_code("Cross validation filepath", output_filename)
            self.mh.train_model(self.dh, output_filename)
            self.mh.save_model_to_file(self.mh.handler.config.get_model_filename())
        except Exception as e:
            raise Exception(f"Training or saving model failed: {e}")

        return {"progress": f"Best model is: ({self.mh.model.get_name()}) with number of features: {self.config.get_num_selected_features()}"}
        

    def evaluate_model__task(self) -> dict:
        """ Evaluate trained model on know testdata """
        self.logger.print_task_header(title="Evaluate trained model")
        if self.dh.X_validation.shape[0] > 0:
            self.logger.print_progress(message="Make predictions on known testdata")
            
            self.ph.make_predictions(self.mh.model.pipeline, self.dh.X_validation, self.dh.classes, self.dh.Y_validation)
            
            self.ph.report_results(self.dh.Y_validation)
            
            return {}
        else:
            raise ConfigException("Test size set too low")
        

    def retrain_model__task(self) -> dict:
        """ RETRAIN the best model on whole dataset with known classification """
        self.logger.print_task_header(title="Retraining model")
        self.logger.print_progress(message="Retrain model on whole dataset")
        
        cross_trained_model = self.mh.load_pipeline_from_file(self.config.get_model_filename())
        trained_model = self.mh.train_picked_model( self.mh.model.pipeline, self.dh.X, self.dh.Y)
            
        self.mh.save_model_to_file(self.config.get_model_filename())

        return {"cross_trained_model": cross_trained_model, "trained_model": trained_model}

    
    def display_mispredicted__task(self, cross_trained_model: Pipeline, trained_model: Pipeline) -> dict:
        """ Compute and display most mispredicted data samples for possible manual correction """
        if not self.config.should_display_mispredicted():    
            return {}

        self.logger.print_task_header(title="Calculating mispredictions")
            
        self.ph.most_mispredicted(self.dh.X_original, trained_model, cross_trained_model, self.dh.X, self.dh.Y)

        self.ph.evaluate_mispredictions(self.config.get_output_filepath("misplaced"))
        
        return {}


    def make_predictions__task(self) -> dict:
        """ Make predictions on non-classified dataset: X_unknown -> Y_unknown """
        self.logger.print_task_header(title="Make predictions")
        
        if self.dh.X_prediction is None:
            raise PredictionsException("No data to predict on")
        
        if self.dh.X_prediction.shape[0] > 0:
            self.ph.make_predictions(self.mh.model.pipeline, self.dh.X_prediction, self.dh.classes)
            
            rates = self.ph.get_rates(as_string = False) if self.ph.could_predict_proba else None
            predictions = Helpers.count_value_distr_as_dict(self.ph.predictions.tolist())
            self.logger.print_prediction_info(predictions, rates)
            saved_results = self.handler.save_predictions()
            if (saved_results["error"] is not None):
                self.logger.print_error(f"Saving predictions failed: {saved_results['error']}")

            else:
                self.logger.print_prediction_results(saved_results["results"])
        else:
            raise PredictionsException("No data to predict on")

        return {}

def get_tasks(config: Config) -> list:
    """ Gets a list of tasks for the given config """
    tasks = [
        "load_model",
        "fetch_data",
        "load_dataset",
        "separate_dataset"
    ]

    if config.get_text_column_names():
        tasks.append("convert_to_numbers")

    if config.should_train():
        training_tasks = [
            "train_model",
            "evaluate_model",
            "retrain_model",
            "display_mispredicted"
        ]
        tasks.extend(training_tasks)

    if config.should_predict():
        predict_tasks = [
            "make_predictions"
        ]
        tasks.extend(predict_tasks)

    return tasks