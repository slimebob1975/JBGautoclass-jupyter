import os
import shutil
from pathlib import Path

import Helpers
import langdetect
import numpy as np
import pandas as pd
import torch
from JBGNeuralNetworks import _NeuralNetwork3PL
from skclean.detectors import (MCS, InstanceHardness, PartitioningDetector,
                               RandomForestDetector)
from skclean.models import Centroid, RobustLR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint
from stop_words import get_stop_words
from torch import nn, optim
from Helpers import recreate_dir
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from typing import Dict, Iterable, Any


""" ESTIMATOR """
class BaseNeuralNetClassifier(BaseEstimator):
    """ The base neural network classifier """
    
    OUTPUT_DIR = "output"
    CHECKPOINT_DIR = "nn_checkpoint"

    def __init__(self):

        super().__init__()
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Placeholders for the input and output layers
        self.num_features = None
        self.num_classes = None
        
        # For handling string classes
        self.label_encoder = LongLabelEncoder()
        
        # The net, this is defined by inheriting classes
        self.net = None
    
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.label_encoder = self.label_encoder.fit(y)
        self._setup_net(X, y)
        self.net.fit( X.astype(np.float32), self.label_encoder.transform(y))
        
    def predict(self, X):
        return self.label_encoder.inverse_transform(self.net.predict(X.astype(np.float32)))
        
    def predict_proba(self, X):
        return self.net.predict_proba(X.astype(np.float32))


class NNClassifier3PL(BaseNeuralNetClassifier):
    """ classifier based on the neural network with at least one hidden layer """
    def __init__(self, num_hidden_layers=2, hidden_layer_size=48, activation='tanh', learning_rate=0.02, max_epochs=20, \
        optimizer='adam', dropout_prob=0.1, verbose=True, train_split=True):
        
        super().__init__()
        
        # Set internal information variables
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.dropout_prob = dropout_prob

        self.verbose = verbose
        self.train_split = train_split  # Whether to split the data into training and test sets internally

        recreate_dir(self.history_file_dir) # Removes the directory and recreates it
        
    
    def _setup_net(self, X, y):
        self.num_features = X.shape[1]
        self.num_classes = len(unique_labels(y))
        the_net = _NeuralNetwork3PL(
            self.num_features,
            self.num_classes,
            self.num_hidden_layers,
            self.hidden_layer_size,
            self._get_activation_function(self.activation),
            self._get_optimizer(self.optimizer),
            self.dropout_prob
        )

        nn_classifier_kwargs = {
            "max_epochs": self.max_epochs,
            "lr": self.learning_rate,
            "device": self.device,
            "verbose": self.verbose,
            "callbacks": [self._get_early_stopping_callback()]
        }
        
        if not self.train_split:
            nn_classifier_kwargs["train_split"] = None

        self.net = NeuralNetClassifier(the_net, **nn_classifier_kwargs)
        
    
    def _get_activation_function(self, activation: str):
        mapped = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }

        func = mapped.get(activation)

        if func is not None:
            return func
        
        raise ValueError(f"Activation function {activation} not recognized!")
        
    
    def _get_optimizer(self, optimizer: str):
        mapped = {
            "adam": optim.Adam,
            "sgd": optim.SGD
        }

        func = mapped.get(optimizer)

        if func is not None:
            return func
        
        raise ValueError(f"Optimizer {optimizer} not recognized!")
        
    
    def _get_early_stopping_callback(self):
        if self.train_split:
            monitor = lambda net: all(net.history[-1, ('train_loss_best', 'valid_loss_best')])
        else:
            monitor = 'train_loss_best'
        dirname = self.history_file_dir
        
        return Checkpoint(monitor=monitor, dirname=dirname, load_best=True)
    
    
    @property
    def history_file_dir(self) -> Path:
        """ Defines the directory to save the best checkpoint"""
        pwd = Path(os.path.dirname(os.path.realpath(__file__)))
        dir = pwd / self.OUTPUT_DIR / self.CHECKPOINT_DIR

        return dir
        
class MLPKerasClassifier(KerasClassifier):

    def __init__(
        self,
        hidden_layer_sizes=(100, ),
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=200,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_))
        model.add(inp)
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(hidden_layer_size, activation="relu")
            model.add(layer)
        if self.target_type_ == "binary":
            n_output_units = 1
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        elif self.target_type_ == "multiclass":
            n_output_units = self.n_classes_
            output_activation = "softmax"
            loss = "sparse_categorical_crossentropy"
        else:
            raise NotImplementedError(f"Unsupported task type: {self.target_type_}")
        out = keras.layers.Dense(n_output_units, activation=output_activation)
        model.add(out)
        model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
        return model

class JBGRobustLogisticRegression(RobustLR):
    def __init__(self, PN=.2, NP=.2, C=np.inf, max_iter=4000, random_state=None):
        super().__init__(PN=.2, NP=.2, C=np.inf, max_iter=4000, random_state=None)
        self.label_encoder = LabelEncoder()

    def fit(self, X, y, sample_weight=None):
        self.label_encoder = self.label_encoder.fit(y)
        super().fit(X, self.label_encoder.transform(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.label_encoder.inverse_transform(super().predict(X))


class JBGRobustCentroid(Centroid):
    
    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        super().fit(X, self.label_encoder.transform(y))
        return self

    def predict(self, X):
        return self.label_encoder.inverse_transform(super().predict(X))


""" DETECTER """
class JBGPartitioningDetector(PartitioningDetector):

    def __init__(self, classifier=None, n_partitions=5, n_jobs=1, random_state=None):
        super().__init__(classifier=classifier, n_partitions=n_partitions, n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))


class JBGMCS(MCS):

    def __init__(self, classifier=None, n_steps=20, n_jobs=1, random_state=None):
        super().__init__(classifier=classifier, n_steps=n_steps, n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))


class JBGInstanceHardness(InstanceHardness):
    
    def __init__(self, classifiers=None, cv=None, n_jobs=1, random_state=None):
        super().__init__(classifiers=classifiers, cv=cv, n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))
    

class JBGRandomForestDetector(RandomForestDetector):
    def __init__(self, n_estimators=101, sampling_ratio=None, n_jobs=1, random_state=None):
        super().__init__(n_estimators=n_estimators, sampling_ratio=sampling_ratio, 
                        n_jobs=n_jobs, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def detect(self, X, y):
        self.label_encoder = self.label_encoder.fit(y)
        return super().detect(X, self.label_encoder.transform(y))


""" TRANSFORM """
class LongLabelEncoder(LabelEncoder):
    """ Handles the case of Long integer class labels """

    def __init__(self):
        super().__init__()
        
    def fit(self, X):
        return super().fit(X)

    def transform(self, X):
        return super().transform(X).astype(np.int64)
        
    def inverse_transform(self, X): 
        return super().inverse_transform(X.astype(int))
    

class TextDataToNumbersConverter(TransformerMixin, BaseEstimator):

    STANDARD_LANGUAGE = 'sv'
    LIMIT_IS_CATEGORICAL = 30

    # Create new instance of converter object
    def __init__(self, text_columns: list[str] = None, category_columns: list[str] = None, \
        limit_categorize: int = LIMIT_IS_CATEGORICAL, language: str = STANDARD_LANGUAGE, \
        stop_words: bool = True, df: float = 1.0, use_encryption: bool = True):
                
        # Take care of input
        if text_columns:
            self.text_columns_ = text_columns.copy()
        else:
            self.text_columns_ = []
        if category_columns:
            self.category_columns_ = category_columns.copy()
        else:
            self.category_columns_ = []
        self.limit_categorize_ = limit_categorize
        self.language_ = language
        self.stop_words_ = stop_words
        self.df_ = df
        self.use_encryption_ = use_encryption

        # Internal transforms for conversion (placeholders)
        self.tfidvectorizer_ = None
        self.ordinalencoder_ = None
        
        return None
        
    # Fit converter to data
    def fit(self, X: pd.DataFrame):
        
        if self.text_columns_:
        
            # Investigate language option
            if not self.language_:
                try:
                    self.language_ = langdetect.detect(' '.join(X))
                except Exception as e:
                    self.language_ = TextDataToNumbersConverter.STANDARD_LANGUAGE  
            
            # Handle stop words option
            if self.stop_words_:
                the_stop_words = self._get_stop_words()
            else:
                the_stop_words = None

            # Find out what text columns in text list that are categorical and separate them into
            # list of categories
            for column in self.text_columns_:
                if self._is_categorical_column(X, column):
                    self.category_columns_.append(column)
            if self.category_columns_:
                self.text_columns_ = [col for col in self.text_columns_ if col not in self.category_columns_]

        # Prepare data for transform
        if self.text_columns_ or self.category_columns_:
            X_document, X_category = self._separate_and_encrypt_input_data(X)

        # Depending on the division of columns, create conversion objects using fit.
        if self.text_columns_:
            self.tfidvectorizer_ = TfidfVectorizer(stop_words=the_stop_words, max_df=self.df_).fit(X_document)
        if self.category_columns_:
            self.ordinalencoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_category)

    def transform(self, X: pd.DataFrame):
        
        check_is_fitted(self)

        # Prepare placeholders for converted data (are dropped silently in concat below if None)
        X_document = None
        X_category = None

        # Prepare data for transform
        if self.text_columns_ or self.category_columns_:
            X_document, X_category = self._separate_and_encrypt_input_data(X)

        # Depending on the division of columns, transform text and categories. Add new columns names.
        if self.text_columns_:
            X_document = pd.DataFrame.sparse.from_spmatrix(self.tfidvectorizer_.transform(X_document), \
                columns = self.tfidvectorizer_.get_feature_names_out(), index=X.index)
        if self.category_columns_:
            X_category = pd.DataFrame(self.ordinalencoder_.transform(X_category), \
                columns = self.ordinalencoder_.get_feature_names_out(), index=X.index)

        # Remove text and category columns from X and put conversion result there instead.
        # Concatenate matrices and column names. Any Nones are dropped silently as long as X is not None.
        if self.text_columns_ or self.category_columns_:
            X = X.drop(self.text_columns_ + self.category_columns_, axis=1)
        
        X = pd.concat([X, X_category, X_document], axis=1, ignore_index=False)

        return X
    
    # Do we really need this?
    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)
    
    # Some help functions below
    def _is_categorical_column(self, X: pd.DataFrame, column: str) -> bool:
        
        if X is None or X[column] is None:
            return X[column].value_counts().count() <= self.limit_categorize_
        else:
            return False

    def _separate_and_encrypt_input_data(self, X: pd.DataFrame):
         
        # Separate text data from categorical data and collapse text data into one "document" column
        if self.text_columns_:
            X_document = X[self.text_columns_].astype(str).agg(' '.join, axis=1)
        else:
            X_document = None
        if self.category_columns_:
            X_category = X[self.category_columns_].astype(str)
        else:
            X_category = None

        # Use encryption on document part if set
        if X_document is not None and self.use_encryption_:
            X_document = Helpers.do_hex_base64_encode_on_data(X_document)

        return X_document, X_category

    def _get_stop_words(self):

        the_stop_words = get_stop_words(self.language_)
        
        # Use encrypton of stop words if set
        if self.use_encryption_:
            for word in the_stop_words:
                word = Helpers.cipher_encode_string(str(word))

        return the_stop_words
    

def main():
    print("Testing NNClassifier3PL!")
    
    import numpy as np
    from sklearn.datasets import make_classification
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # This is a toy dataset for binary classification, 1000 data points with 5 features each
    X, y = make_classification(1000, 5, n_classes=2, random_state=0)
    X = X.astype(np.float64)
    y = ["class " + str(elem) for elem in y]
    
    # Create the net in question
    net = NNClassifier3PL(train_split=True)
    
    # Fit the net to the data
    net.fit(X, y)
    
    # Making prediction for first 5 data points of X
    y_pred = net.predict(X[:5])
    print(y_pred)
    
    # Checking probarbility of each class for first 5 data points of X
    y_proba = net.predict_proba(X[:5])
    print(y_proba)
    


# Start main
if __name__ == "__main__":
    main()
