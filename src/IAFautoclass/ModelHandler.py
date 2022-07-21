
from dataclasses import dataclass, field
from math import ceil
import pickle
import time
import typing

import pandas
from sklearn.feature_selection import RFE
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from DataLayer import DataLayer
from IAFLogger import IAFLogger
import Config

class ModelException(Exception):
    def __init__(self, message):
        super().__init__(f"ModelException: {message}")

@dataclass
class Model:
    label_binarizes: dict = field(default_factory=dict)
    count_vectorizer: CountVectorizer = field(default=None)
    tfid_transformer: TfidfTransformer = field(default=None)
    algorithm: Config.Algorithm = field(default=None)
    preprocess: Config.Preprocess = field(default=None)
    model: Pipeline = field(default=None)
    transform: typing.Any = field(default=None)

@dataclass
class ModelHandler:
    datalayer: DataLayer
    config: Config.Config
    logger: IAFLogger

    #label_binarizes: dict = field(default_factory=dict)
    #count_vectorizer: CountVectorizer = field(init=False)
    #tfid_transformer: TfidfTransformer = field(init=False)
    #model_name: tuple = field(init=False)
    #model: Pipeline = field(init=False)
    #feature_selection_transform: typing.Any = field(init=False)
    model: Model = field(init=False)
    

    use_feature_selection: bool = field(init=False)
    text_data: bool = field(init=False)
    
    def __post_init__(self) -> None:
        self.text_data = self.config.connection.data_text_columns != ""
        self.use_feature_selection = self.config.mode.feature_selection != Config.Reduction.NON

        self.model = self.load_model()

    # Loads model based on config
    def load_model(self) -> Model:
        if self.config.mode.train:
            return self.load_empty_model()
        
        return self.load_model_from_file(self.config.get_model_filename())
            

    # Load ml model
    def load_model_from_file(self, filename: str) -> Model:
        try:
            _, label_binarizers, count_vectorizer, tfid_transformer, feature_selection_transform, model_name, model = pickle.load(open(filename, 'rb'))
            
        except Exception as e:
            self.logger.print_warning(f"Something went wrong on loading model from file: {e}")
            return None
        
        model_class = Model(
            label_binarizes=label_binarizers,
            count_vectorizer=count_vectorizer,
            tfid_transformer=tfid_transformer,
            transform=feature_selection_transform,
            algorithm=model_name[0],
            preprocess=model_name[1]
        )
        #self.label_binarizes = label_binarizers
        #self.count_vectorizer = count_vectorizer
        #self.tfid_transformer = tfid_transformer
        #self.feature_selection_transform = feature_selection_transform
        #self.model_name = model_name
        #self.model = model

        if self.config.mode.predict:
            self.config.mode.num_selected_features = model.n_features_in_

        return model_class
    
    # Sets default (read: empty) values
    def load_empty_model(self) -> Model:
        #self.label_binarizers = {}
        #self.count_vectorizer = None
        #self.tfid_transformer = None
        #self.feature_selection_transform = None
        #self.model = None

        return Model(label_binarizes={}, count_vectorizer=None, tfid_transformer=None, transform=None)

    def train_model(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame):
        self.logger.print_progress(message="Check and train algorithms for best model")
        
        try:
            self.model = self.get_model_from(X_train, Y_train)
    
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise ModelException(f"Something went wrong on training picked model: {str(e)}")


        self.save_model_to_file(self.config.get_model_filename())

    def get_model_from(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame):
        k = min(10, find_smallest_class_number(Y_train))
        if k < 10:
            self.logger.print_info(f"Using non-standard k-value for spotcheck of algorithms: {k}")
        
        model = self.spot_check_ml_algorithms(X_train, Y_train, k)
        model.fit(X_train, Y_train)
        return model

    # While more code, this should (hopefully) be easier to read
    def should_run_computation(self, current_algorithm: Config.Algorithm, current_preprocessor: Config.Preprocess) -> bool:
        chosen_algorithm = self.config.mode.algorithm
        chosen_preprocessor = self.config.mode.preprocessor
        
        # If both of these are ALL, it doesn't matter where in the set we are
        if chosen_algorithm == Config.Algorithm.ALL and chosen_preprocessor == Config.Preprocess.ALL:
            return True

        # If both the current one are equal to the chosen ones, carry on
        if current_algorithm == chosen_algorithm and current_preprocessor == chosen_preprocessor:
            return True

        # Two edge cases: A is All and P is Current, or A is Current and P is All
        if chosen_algorithm == Config.Algorithm.ALL and current_preprocessor == chosen_preprocessor:
            return True

        if current_algorithm == chosen_algorithm and chosen_preprocessor == Config.Preprocess.ALL:
            return True

        return False

    def prepare_models_preprocessors(self, size):
        # Add all algorithms in a list
        self.logger.print_info("Spot check ml algorithms...")
        models = []
        for algo in Config.Algorithm:
            if algo.has_algorithm_function():
                models.append((algo, algo.call_algorithm(max_iterations=self.config.mode.max_iterations, size=size)))
        

        # Add different preprocessing methods in another list
        preprocessors = []
        for preprocessor in Config.Preprocess:
            # This checks that the preprocessor has the function, and also that BIN is only added if self.text_data
            if preprocessor.has_preprocess_function() and (preprocessor.name != "BIN" or self.text_data):
                preprocessors.append((preprocessor, preprocessor.call_preprocess()))
                
        
        preprocessors.append((Config.Preprocess.NON, None)) # In case they choose no preprocessor in config

        return models, preprocessors

    # Spot Check Algorithms.
    # We do an extensive search of the best algorithm in comparison with the best
    # preprocessing.
    def spot_check_ml_algorithms(self, X_train: pandas.DataFrame, Y_train: pandas.DataFrame, k:int=10) -> Pipeline:

        # Save standard progress text
        standardProgressText = "Check and train algorithms for best model"

        models, preprocessors = self.prepare_models_preprocessors(size=X_train.shape[0])
        
        # Evaluate each model in turn in combination with all preprocessing methods
        names = []
        best_mean = 0.0
        best_std = 1.0
        trained_model = None
        temp_model = None
        algorithm = None
        pprocessor = None
        
        scorer_mechanism = self.config.mode.scoring.name
        if self.config.mode.scoring == Config.Scoretype.mcc:
            scorer_mechanism = make_scorer(matthews_corrcoef)

        smote = None
        undersampler = None
        if self.config.mode.smote:
            smote = SMOTE(sampling_strategy='auto')
        if self.config.mode.undersample:
            undersampler = RandomUnderSampler(sampling_strategy='auto')

        # Make evaluation of model
        try:
            kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
        except Exception as e:
            print(f"Here be dragons: {type(e)}") # Remove this once we've narrowed the exception types down
            raise ModelException(f"StratifiedKfold raised an exception with message: {e}")
        
        best_feature_selection = X_train.shape[1]
        first_round = True
        if self.config.io.verbose: # TODO: Can this be printed nicer?
                print("{0:>4s}-{1:<6s}{2:>6s}{3:>8s}{4:>8s}{5:>11s}".format("Name","Prep.","#Feat.","Mean","Std","Time"))
                print("="*45)
        #numMinorTasks = len(models) * len(preprocessors)
        #percentAddPerMinorTask = (1.0-self.percentPermajorTask*self.numMajorTasks) / float(numMinorTasks)

        # Loop over the models
        for name, model in models:
            # Loop over pre-processing methods
            for preprocessor_name, preprocessor in preprocessors:
                if not self.should_run_computation(name, preprocessor_name):
                    self.logger.print_progress(message=f"Skipping ({name.name}-{preprocessor_name.name}) due to config")
                    continue
                # Update progressbar percent and label
                self.logger.print_progress(message=f"{standardProgressText} ({name.name}-{preprocessor_name.name})")
                if not first_round: 
                    #self.update_progress(percent=percentAddPerMinorTask)
                    print("need to fix percent")
                else:
                    first_round = False


                # Add feature selection if selected, i.e., the option of reducing the number of variables used.
                # Make a binary search for the optimal dimensions.
                max_features_selection = X_train.shape[1]

                # Make sure all numbers are propely set for feature selection interval
                if self.use_feature_selection and self.config.mode.num_selected_features in ["", None]:
                    min_features_selection = 0
                elif self.use_feature_selection and self.config.mode.num_selected_features > 0:
                    min_features_selection = self.config.mode.num_selected_features
                    max_features_selection = self.config.mode.num_selected_features
                else:
                    min_features_selection = max_features_selection # No or minimal number of features are eliminated

                # Loop over feature selections span: break this loop when min and max reach the same value
                score = 0.0                                                 # Save the best values
                stdev = 1.0                                                 # so far.
                num_features = max_features_selection                       # Start with all features.
                first_feature_selection = True                              # Make special first round: use all features
                counter = 0
                while first_feature_selection or min_features_selection < max_features_selection:
                    counter += 1

                    # Update limits for binary search and break loop if we are done
                    if not first_feature_selection:
                        num_features = ceil((min_features_selection+max_features_selection) / 2)
                        if num_features == max_features_selection:          
                            break
                    else:
                        first_feature_selection = False
                        num_features = max_features_selection

                    # Calculate the time for this setting
                    t0 = time.time()

                    # Apply feature selection to current model and number of features.
                    # If feature selection is not applicable, set a flag to the loop is 
                    # ending after one iteration
                    temp_model = model
                    if self.use_feature_selection and self.config.mode.feature_selection == Config.Reduction.RFE:     
                        try:
                            rfe = RFE(temp_model, n_features_to_select=num_features)
                            temp_model = rfe.fit(X_train, Y_train)
                        except ValueError as e:
                            break

                    # Both algorithm and preprocessor should be used. Move on.
                    # Build pipline of model and preprocessor.
                    names.append((name,preprocessor_name))
                    if not self.config.mode.smote and not self.config.mode.undersample:
                        if preprocessor != None:
                            pipe = make_pipeline(preprocessor, temp_model)
                        else:
                            pipe = temp_model

                    # For use SMOTE and undersampling, different Pipeline is used
                    else:
                        steps = [('smote', smote ), ('under', undersampler), \
                                ('preprocessor', preprocessor), ('model', temp_model)]
                        pipe = ImbPipeline(steps=steps)
                        

                    # Now make kfolded cross evaluation
                    cv_results = None
                    try:
                        cv_results = cross_val_score(pipe, X_train, Y_train, cv=kfold, scoring=scorer_mechanism) 
                    except ValueError as e:
                        self.logger.print_warning(f"Model {names[-1]} raised a ValueError in cross_val_score. Skipping to next")
                    else:
                        # Stop the stopwatch
                        t = time.time() - t0

                        # For current settings, calculate score
                        temp_score = cv_results.mean()
                        temp_stdev = cv_results.std()

                        # Print results to screen
                        if self.config.io.verbose: # TODO: print prettier
                            print("{0:>4s}-{1:<6s}{2:6d}{3:8.3f}{4:8.3f}{5:11.3f} s.".
                                  format(name,preprocessor_name,num_features,temp_score,temp_stdev,t))

                        # Evaluate if feature selection changed accuracy or not. 
                        # Notice: Better or same score with less variables are both seen as an improvement,
                        # since the chance of finding an improvement increases when number of variables decrease
                        if  temp_score >= score:
                            score = temp_score
                            stdev = temp_stdev
                            max_features_selection = num_features   # We need to reduce more features
                        else:
                            min_features_selection = num_features   # We have reduced too much already  

                         # Save result if it is the overall best
                         # Notice the difference from above, here we demand a better score.
                        if temp_score > best_mean or \
                           (temp_score == best_mean and temp_stdev < best_std):
                            trained_model = pipe
                            algorithm = name
                            pprocessor = preprocessor_name
                            best_mean = temp_score
                            best_std = temp_stdev
                            best_feature_selection = num_features

        self.config.mode.algorithm = algorithm
        self.config.mode.preprocessor = pprocessor
        self.config.mode.num_selected_features = best_feature_selection
        self.model_name = (algorithm, pprocessor)
        
        # Return best model for start making predictions
        return trained_model

    # Save ml model and corresponding configuration
    def save_model_to_file(self, filename):
        
        try:
            save_config = self.config.get_clean_config()
            data = [save_config, self.label_binarizers, self.count_vectorizer, self.tfid_transformer, self.feature_selection_transform, self.model_name, self.model]
            pickle.dump(data, open(filename,'wb'))

        except Exception as e:
            self.logger.print_warning(f"Something went wrong on saving model to file: {e}")

def find_smallest_class_number(Y):
    class_count = {}
    for elem in Y:
        if elem not in class_count:
            class_count[elem] = 1
        else:
            class_count[elem] += 1
    return max(1, min(class_count.values()))
