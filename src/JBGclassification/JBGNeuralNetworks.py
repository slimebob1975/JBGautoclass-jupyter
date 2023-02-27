# Experimental test of PyTorch functionality via SKORCH
# https://skorch.readthedocs.io/en/stable/index.html

from torch import nn

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
    
        

    
