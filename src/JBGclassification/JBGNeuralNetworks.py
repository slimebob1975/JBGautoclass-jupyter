# Experimental test of PyTorch functionality via SKORCH
# https://skorch.readthedocs.io/en/stable/index.html

import os
from pathlib import Path
import torch
from torch import nn, optim 
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels

# The base neural network classifier
class BaseNeuralNetClassifier(BaseEstimator):
    
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

# A neural network with at least one hidden layer
class _NeuralNetwork3PL(nn.Module):
    def __init__(self, input_features, output_features, num_hidden_layers, hidden_layer_size,\
        activation, optimizer, dropout_prob):

        # Set up base class
        super(_NeuralNetwork3PL, self).__init__()
        
        # Some internal variables
        self.activation = activation
        self.dropout_prob = dropout_prob
        
        # Set up layers
        self.layers = nn.ModuleList()
    
        # Input layer
        self.layers.append(nn.Linear(input_features, hidden_layer_size))

        # The hidden layers
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        
        # The output layer
        self.layers.append(nn.Linear(hidden_layer_size, output_features))
        
        # Construct the opti
        self.optimizer = optimizer(params=self.layers.parameters(), lr=0.001)

    def forward(self, X, **kwargs):
        for i in range(len(self.layers)-1):
            X = self.activation()(self.layers[i](X))
            X = nn.Dropout(self.dropout_prob)(X)
        X = nn.Softmax(dim=-1)(self.layers[-1](X))
        return X

# A classifier based on the neural network with at least one hidden layer
class NNClassifier3PL(BaseNeuralNetClassifier):
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
        
    def _setup_net(self, X, y):
        self.num_features = X.shape[1]
        self.num_classes = len(unique_labels(y))
        the_net = _NeuralNetwork3PL(self.num_features, self.num_classes, self.num_hidden_layers, self.hidden_layer_size, \
            self._get_activation_function(self.activation), self._get_optimizer(self.optimizer), self.dropout_prob )
        if self.train_split:
            self.net = NeuralNetClassifier(the_net, max_epochs=self.max_epochs, lr=self.learning_rate, device=self.device, \
                verbose=self.verbose, callbacks=[self._get_early_stopping_callback()])

        else:
            self.net = NeuralNetClassifier(the_net, max_epochs=self.max_epochs, lr=self.learning_rate, device=self.device, \
                verbose=self.verbose, train_split=None, callbacks=[self._get_early_stopping_callback()])
            
    def _get_activation_function(self, activation):
        if activation =='relu':
            return nn.ReLU
        elif activation == 'tanh':
            return nn.Tanh
        elif activation =='sigmoid':
            return nn.Sigmoid
        else:
            raise ValueError(f"Activation function {activation} not recognized!")
        
    def _get_optimizer(self, optimizer):
        if optimizer == 'adam':
            return optim.Adam
        elif optimizer == 'sgd':
            return optim.SGD
        else:
            raise ValueError(f"Optimizer {optimizer} not recognized!")
        
    def _get_early_stopping_callback(self):
        if self.train_split:
            monitor = lambda net: all(net.history[-1, ('train_loss_best', 'valid_loss_best')])
        else:
            monitor = 'train_loss_best'
        dirname = self._get_history_file_dir()
        return Checkpoint(monitor=monitor, dirname=dirname, load_best=True)
    
    def _get_history_file_dir(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        dir = Path(pwd) / "./.nn_checkpoint_tmp/"
        try:
            os.rmdir(dir)
        except OSError:
            pass
        try:
            os.makedirs(dir)
        except OSError:
            return Path(pwd)
        else:
            return dir

# Class which handles the case of Long integer class labels
class LongLabelEncoder(LabelEncoder):
    def __init__(self):
        super().__init__()
        
    def fit(self, X):
        return super().fit(X)

    def transform(self, X):
        return super().transform(X).astype(np.int64)
        
    def inverse_transform(self, X): 
        return super().inverse_transform(X.astype(int))

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
        

    
