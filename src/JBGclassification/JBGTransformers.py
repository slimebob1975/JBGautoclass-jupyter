import os
import pickle
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import Helpers
import langdetect
import numpy as np
import pandas as pd
import torch
from JBGNeuralNetworks import _NeuralNetwork3PL
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint
from stop_words import get_stop_words
from torch import nn, optim
from Helpers import recreate_dir
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from typing import Dict, Iterable, Any


""" ESTIMATOR """
class BaseNeuralNetClassifier(ClassifierMixin, BaseEstimator):
    """ The base neural network classifier """
    
    OUTPUT_DIR = "output"
    CHECKPOINT_DIR = "nn_checkpoints"

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

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("net") is not None:
            # Skorch supports standard pickle. Encode only that component so
            # outer dill artifacts do not traverse Torch optimizer internals.
            state["_jbg_skorch_pickle"] = pickle.dumps(state.pop("net"))
        return state

    def __setstate__(self, state):
        state = state.copy()
        encoded_net = state.pop("_jbg_skorch_pickle", None)
        if encoded_net is not None:
            state["net"] = pickle.loads(encoded_net)
        # Historical artifacts store net directly and require no conversion.
        self.__dict__.update(state)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        # Each fit (including clones/folds/retries) owns its checkpoint files.
        # Load the chosen checkpoint inside fit before deleting temporary files.
        self.net = None
        self.__dict__.pop("classes_", None)
        self.__dict__.pop("n_features_in_", None)
        self.classes_ = unique_labels(y)
        self.label_encoder = self.label_encoder.fit(y)
        checkpoint_root = self.history_file_dir
        try:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="fit-", dir=checkpoint_root) as dirname:
                self._checkpoint_fit_dir = Path(dirname)
                self.checkpoint_fit_dir_ = dirname  # diagnostic only; not a reload dependency
                self._setup_net(X, y)
                self.net.fit(X, self.label_encoder.transform(y))
            self.n_features_in_ = X.shape[1]
        except BaseException:
            self.net = None
            self.__dict__.pop("classes_", None)
            raise
        finally:
            self.__dict__.pop("_checkpoint_fit_dir", None)
        return self

    def _prediction_input(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X, dtype=np.float32)
        # num_features is retained for artifacts saved before revision 116.
        expected = getattr(self, "n_features_in_", self.num_features)
        if X.shape[1] != expected:
            raise ValueError(f"Expected {expected} input features, got {X.shape[1]}.")
        return X
        
    def predict(self, X):
        X = self._prediction_input(X)
        return self.label_encoder.inverse_transform(self.net.predict(X))
        
    def predict_proba(self, X):
        X = self._prediction_input(X)
        return self.net.predict_proba(X)


class NNClassifier3PL(BaseNeuralNetClassifier):
    """Skorch classifier; num_hidden_layers counts stages after the first hidden layer."""
    def __init__(self, num_hidden_layers=2, hidden_layer_size=48, activation='tanh', learning_rate=0.001, max_epochs=50, \
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
            "optimizer": self._get_optimizer(self.optimizer),
            "criterion": nn.NLLLoss,
            "batch_size": 128,
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
        """Historical name: restore the best checkpoint, without stopping early."""
        monitor = 'valid_loss_best' if self.train_split else 'train_loss_best'
        dirname = self.history_file_dir
        
        return Checkpoint(monitor=monitor, dirname=dirname, load_best=True)
    
    
    @property
    def history_file_dir(self) -> Path:
        """Active fit's private directory, or the historical checkpoint root."""
        if getattr(self, "_checkpoint_fit_dir", None) is not None:
            return self._checkpoint_fit_dir
        pwd = Path(os.path.dirname(os.path.realpath(__file__)))
        dir = pwd / self.OUTPUT_DIR / self.CHECKPOINT_DIR

        return dir
        
class MLPKerasClassifier(KerasClassifier):

    def __init__(
        self,
        model=None,
        hidden_layer_sizes=(100, ),
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=50,
        batch_size=32,
        verbose=0,
        **kwargs,
    ):
        self.model = model
        super().__init__(self.model, **kwargs)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.optimizer__learning_rate = optimizer__learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_,))
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
        stop_words: bool = True, df: float = 1.0, ngram_range: tuple = (1,1), use_encryption: bool = True, \
        use_categorization: bool = True ):
                
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
        self.ngram_range_ = ngram_range
        self.use_categorization_ = use_categorization

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
            if self.use_categorization_:
                for column in self.text_columns_:
                    if column not in self.category_columns_ and self._is_categorical_column(X, column):
                        self.category_columns_.append(column)
            if self.category_columns_:
                self.text_columns_ = [col for col in self.text_columns_ if col not in self.category_columns_]

        # Prepare data for transform
        if self.text_columns_ or self.category_columns_:
            X_document, X_category = self._separate_and_encrypt_input_data(X)

        # Depending on the division of columns, create conversion objects using fit.
        if self.text_columns_:
            self.tfidvectorizer_ = TfidfVectorizer(stop_words=the_stop_words, ngram_range=self.ngram_range_ ,
                                                   max_df=self.df_).fit(X_document)
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
        if X is None or column not in X.columns:
            return False

        return X[column].nunique(dropna=True) <= self.limit_categorize_

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
