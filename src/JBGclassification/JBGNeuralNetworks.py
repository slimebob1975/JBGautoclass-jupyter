# Experimental test of PyTorch functionality via SKORCH
# https://skorch.readthedocs.io/en/stable/index.html

import torch
from torch import nn, optim
import torch.nn.functional as F 
from skorch import NeuralNetClassifier
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels

MAX_HIDDEN_LAYER_SIZE = 100

# The base neural network classifier
class BaseNNClassifier(BaseEstimator):
    
    def __init__(self):
        
        super().__init__()
        
        # The classes
        self.classes_ = None
        
        # For handling string classes
        self.label_encoder = LongLabelEncoder()
        
        # The net, this is defined by inheriting classes
        self.net = None
    
    def fit(self, X, y):
        if self.classes_ is None:
            self.classes_ = unique_labels(y)

        self.label_encoder = self.label_encoder.fit(y)
        
        if not self.net:
            self._setup_net(X, y)
        self.net.fit( X.astype(np.float32), self.label_encoder.transform(y) )
        
    def predict(self, X):
        return self.label_encoder.inverse_transform(self.net.predict(X.astype(np.float32)))
        
    def predict_proba(self, X):
        return self.net.predict_proba(X.astype(np.float32))

# A neural network with 2 layers of hidden units
class NeuralNet2L(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            hidden_layer_size,
            nonlin=F.relu,
            dropout=0.5
            
    ):
        super(NeuralNet2L, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_layer_size = hidden_layer_size
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(self.num_features, hidden_layer_size)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output = nn.Linear(hidden_layer_size, self.num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X

# A classifier based on the neural network with 2 layers of hidden units
class NNClassifier2L(BaseNNClassifier):
    def __init__(
        self,  
        hidden_layer_size=MAX_HIDDEN_LAYER_SIZE,
        nonlin=F.relu,
        dropout=0.5,
        max_epochs=20,
        lr=0.01,
        dev='cpu',
        verbose=True,
    ):
        super().__init__()
        # Parameters we know right now
        self.hidden_layer_size = hidden_layer_size
        self.nonlin = nonlin
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.dev = dev
        self.verbose = verbose
        
        # Placeholders for the real net setup first time we call fit
        self.net = None     
        self.num_features = None
        self.num_classes = None
        
        # For handling string classes
        self.label_encoder = LongLabelEncoder()
        
    def _setup_net(self, X, y):
        self.num_features = X.shape[1]
        self.num_classes = len(set(y))
        self.net = NeuralNetClassifier(
            NeuralNet2L(self.num_features, self.num_classes, self.hidden_layer_size, self.nonlin, self.dropout), \
            max_epochs=self.max_epochs, lr=self.lr, device=self.dev, verbose=self.verbose)
         
# A neural network with num_classes layers of hidden units
class NeuralNetXL(nn.Module):
    
    def __init__(self, input_size, layers_data: list, learning_rate=0.01, optimizer=optim.Adam):
        super(NeuralNetXL, self).__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(activation, nn.Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X = torch.nn.functional.softmax(X, dim=-1)
        return X
    
# A classifier based on the neural network with num_classes layers of hidden units
class NNClassifierXL(BaseNNClassifier):
    def __init__(self, verbose=True):
        
        super().__init__()

        # Set verbosity level
        self.verbose = verbose

        # Placeholders for the real net setup first time we call fit
        self.net = None     
        self.num_features = None
        self.num_classes = None
        
    def _setup_net(self, X, y):
        self.num_features = X.shape[1]
        self.num_classes = len(set(y))
        layers_data = []
        
        # Add hidden layers
        for i in range(self.num_classes):
            layers_data.append((MAX_HIDDEN_LAYER_SIZE, nn.ReLU()))
            
        # Add output layer
        layers_data.append((self.num_classes, nn.Sigmoid()))
                    
        # Construct the PyTorch neural net
        the_net = NeuralNetXL(self.num_features, layers_data)
        
        # Put the SKORCH wrapper around the classifier
        self.net = NeuralNetClassifier(the_net, max_epochs=20, verbose=self.verbose)
    
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
    print("Testing NNClassifier2L!")
    
    import numpy as np
    from sklearn.datasets import make_classification
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # This is a toy dataset for binary classification, 1000 data points with 20 features each
    X, y = make_classification(1000, 5, n_classes=2, random_state=0)
    X = X.astype(np.float64)
    y = ["class " + str(elem) for elem in y]

    # Create the net in question
    net = NNClassifier2L()
    
    # Fit the net to the data
    net.fit(X, y)
    
    # Making prediction for first 5 data points of X
    y_pred = net.predict(X[:5])
    print(y_pred)
    
    # Checking probarbility of each class for first 5 data points of X
    y_proba = net.predict_proba(X[:5])
    print(y_proba)
    
    print("Testing NNClassifierXL!")
    
    # This is a toy dataset for binary classification, 1000 data points with 20 features each
    X, y = make_classification(1000, 5, n_classes=2, random_state=0)
    X = X.astype(np.float64)
    y = ["class" + str(elem) for elem in y]

    # Create the net in question
    net = NNClassifierXL()
    
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
        

    
