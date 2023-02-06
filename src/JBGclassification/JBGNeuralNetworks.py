# Experimental test of PyTorch functionality via SKORCH
# https://skorch.readthedocs.io/en/stable/index.html

import torch
from torch import nn
import torch.nn.functional as F 
from skorch import NeuralNetClassifier

class NeuralNet(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            hidden_layer_size,
            nonlin=F.relu,
            dropout=0.5
            
    ):
        super(NeuralNet, self).__init__()
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
    
class NNClassifier():
    def __init__(
        self,  
        hidden_layer_size=100,
        nonlin=F.relu,
        dropout=0.5,
        max_epochs=20,
        lr=0.1,
        dev='cpu'
    ):
        # Parameters we know right now
        self.hidden_layer_size = hidden_layer_size
        self.nonlin = nonlin
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.dev = dev
        
        # Placeholders for the real net setup first time we call fit
        self.net = None     
        self.num_features = None
        self.num_classes = None
        
    def _setup_net(self, X, y):
        self.num_features = X.shape[1]
        self.num_classes = len(set(y))
        self.net = NeuralNetClassifier(
            NeuralNet(self.num_features, self.num_classes, self.hidden_layer_size, self.nonlin, self.dropout), \
            max_epochs=self.max_epochs, lr=self.lr, device=self.dev)

    
    def fit(self, X, y):
        if not self.net:
            self._setup_net(X, y)
        self.net.fit( X, y )
        
    def predict(self, y):
        return self.net.predict(y)
        
    def predict_proba(self, X):
        return self.net.predict_proba(X)
        
        
def main():
    print("Testing JBGNeuralNetworks!")
    
    import numpy as np
    from sklearn.datasets import make_classification
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # This is a toy dataset for binary classification, 1000 data points with 20 features each
    X, y = make_classification(1000, 200, n_informative=100, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.int64)
    print(X.shape, y.shape, y.mean())
    
    # Create the net in question
    net = NNClassifier(hidden_layer_size=100, dev='cpu')
    
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
        

    
